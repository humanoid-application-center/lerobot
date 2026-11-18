#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Feature transforms: preprocessor steps that change what the policy sees, named in the train config.

`feature_transforms` in `TrainPipelineConfig` is a list of `{"type": <registered name>, ...arguments}`,
as `train_transforms`, but each names a `FeatureTransformStep`, a `ProcessorStep` registered with
`ProcessorStepRegistry`. Unlike a train transform, a feature transform

- runs inside the preprocessor, right before its normalizer: on raw values, a whole batch at a time,
  on the policy's device. The preprocessor is saved with the policy, so the eval loop and whoever
  loads the checkpoint run the same transform;
- may ask for more frames than the policy does (`delta_indices`, e.g. the previous state), and may
  change a feature's shape and names (`transform_dataset_features`);
- defines the normalization statistics of the keys it changes (`transformed_keys`): lerobot-train
  passes the training frames through the transforms (no image decoded) and computes the statistics of
  the result, padded steps left out; `adjust_stats` may then change them (e.g. floor a std).

At inference the caller gives the raw features the transform takes: a key with `delta_indices` is a
window, `(B, len(delta_indices), ...)`, batched by the caller (the preprocessor's batch step only adds
the batch dimension to a `(D,)` state).

lerobot registers none. A package registers its steps with `@ProcessorStepRegistry.register(name)` in a
module of the `lerobot.train_hooks` entry-point group (see lerobot.common.train_hooks); the registry
loads that group when it does not know a name, so a saved preprocessor finds its steps at inference too.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.datasets.compute_stats import RunningQuantileStats
from lerobot.processor import NormalizerProcessorStep, ProcessorStep, ProcessorStepRegistry
from lerobot.processor.converters import batch_to_transition, transition_to_batch

from .train_hooks import load_train_hook_plugins

STATS_BATCH_SIZE = 1024


class FeatureTransformStep(ProcessorStep):
    """A preprocessor step that changes features before normalization (see the module docstring).

    A subclass is a dataclass of its config arguments, registered with
    `@ProcessorStepRegistry.register(name)`, and implements `__call__` (an `EnvTransition`: raw values,
    batched) and `transformed_keys`.
    """

    def delta_indices(self) -> dict[str, list[int]]:
        """Frame offsets the transform needs per key, e.g. {"observation.state": [-1, 0]}. A key the policy
        already windows must have the same offsets."""
        return {}

    def transformed_keys(self) -> list[str]:
        """The keys whose values the transform changes: their statistics are recomputed."""
        raise NotImplementedError

    def transform_dataset_features(self, features: dict[str, dict]) -> dict[str, dict]:
        """Dataset features ({key: {"dtype", "shape", "names"}}) as the policy sees them after the transform.
        `names` may be missing."""
        return features

    def adjust_stats(self, key: str, stats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The statistics of `key` computed over the transformed values, changed if need be."""
        return stats

    def representation(self) -> dict[str, Any]:
        """What a deployment reads to feed the policy, beyond the step itself (which the saved preprocessor
        has): e.g. a sidecar file's `representation`."""
        return {}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        out = {}
        for ftype, group in features.items():
            changed = self.transform_dataset_features(
                {key: {"dtype": "float32", "shape": tuple(ft.shape)} for key, ft in group.items()}
            )
            out[ftype] = {
                key: PolicyFeature(type=ft.type, shape=tuple(changed[key]["shape"]))
                for key, ft in group.items()
                if key in changed
            }
        return out


def build_feature_transforms(specs: list[dict[str, Any]]) -> list[FeatureTransformStep]:
    if specs:
        load_train_hook_plugins()
    steps = []
    for spec in specs:
        spec = dict(spec)
        name = spec.pop("type", None)
        cls = ProcessorStepRegistry.get(name)
        if not issubclass(cls, FeatureTransformStep):
            raise ValueError(f"feature transform {name!r} is a {cls.__name__}, not a FeatureTransformStep")
        steps.append(cls(**spec))
    return steps


def merge_delta_timestamps(
    delta_timestamps: dict[str, list[float]] | None, steps: list[FeatureTransformStep], fps: int
) -> dict[str, list[float]] | None:
    """The policy's delta_timestamps plus the windows the transforms need."""
    out = dict(delta_timestamps or {})
    for step in steps:
        for key, offsets in step.delta_indices().items():
            wanted = [i / fps for i in offsets]
            if key in out and [round(t * fps) for t in out[key]] != list(offsets):
                raise ValueError(
                    f"{type(step).__name__} needs {key} at frames {list(offsets)}, the policy at "
                    f"{[round(t * fps) for t in out[key]]}"
                )
            out[key] = wanted
    return out or None


def transformed_meta(meta, steps: list[FeatureTransformStep], stats: dict[str, dict[str, np.ndarray]]):
    """A copy of a dataset's metadata as the policy sees it: transformed features, and `stats` in place
    of the statistics of the keys the transforms change. The dataset keeps its own."""
    out = copy.copy(meta)
    out.info = copy.deepcopy(meta.info)
    features = copy.deepcopy(meta.info.features)
    for step in steps:
        features = step.transform_dataset_features(features)
    out.info.features = features
    out.stats = {**(meta.stats or {}), **stats}
    return out


def _windows(
    columns: dict[str, np.ndarray], episode: np.ndarray, rows: np.ndarray, delta: dict[str, list[int]]
):
    """The batch lerobot's dataset gives for `rows` (positions in `columns`), without images: a key with
    delta indices is a window clamped to its episode, with `<key>_is_pad`."""
    batch: dict[str, Any] = {}
    # episodes are contiguous runs of rows (whole episodes are loaded)
    change = np.flatnonzero(np.diff(episode)) + 1
    start = np.repeat(np.r_[0, change], np.diff(np.r_[0, change, len(episode)]))
    end = np.repeat(np.r_[change, len(episode)], np.diff(np.r_[0, change, len(episode)]))
    for key, values in columns.items():
        if key not in delta:
            batch[key] = torch.from_numpy(values[rows])
            continue
        want = rows[:, None] + np.asarray(delta[key])[None, :]
        lo, hi = start[rows][:, None], end[rows][:, None]
        batch[key] = torch.from_numpy(values[np.clip(want, lo, hi - 1)])
        batch[f"{key}_is_pad"] = torch.from_numpy((want < lo) | (want >= hi))
    return batch


def compute_transformed_stats(
    dataset, steps: list[FeatureTransformStep], delta_indices: dict[str, list[int]] | None
) -> dict[str, dict[str, np.ndarray]]:
    """Statistics of the transformed keys over every frame of `dataset`, windows as in training.

    Only the columns that are not images are read. A transformed value with a window (`(B, T, ...)`)
    counts the steps whose `<key>_is_pad` is False."""
    keys = sorted({k for step in steps for k in step.transformed_keys()})
    camera = set(dataset.meta.camera_keys)
    hf = dataset.hf_dataset
    names = [
        k
        for k in dataset.meta.features
        if k not in camera and k in hf.column_names and dataset.meta.features[k]["dtype"].startswith("float")
    ]
    table = hf.select_columns([*names, "episode_index"]).with_format("numpy")[:]
    columns = {k: np.stack(table[k]).astype(np.float32).reshape(len(table[k]), -1) for k in names}
    episode = np.asarray(table["episode_index"]).reshape(-1)
    delta = {k: v for k, v in (delta_indices or {}).items() if k in columns}
    features = copy.deepcopy(dataset.meta.features)
    for step in steps:
        features = step.transform_dataset_features(features)

    running = {k: RunningQuantileStats() for k in keys}
    for first in range(0, len(episode), STATS_BATCH_SIZE):
        rows = np.arange(first, min(first + STATS_BATCH_SIZE, len(episode)))
        transition = batch_to_transition(_windows(columns, episode, rows, delta))
        for step in steps:
            transition = step(transition)
        batch = transition_to_batch(transition)
        for key in keys:
            value = batch[key].detach().to("cpu", torch.float64).numpy()
            ndim = len(features[key]["shape"])
            if value.ndim == ndim + 2:  # a window: its real steps
                pad = batch.get(f"{key}_is_pad")
                value = value[~pad.cpu().numpy()] if pad is not None else value.reshape(-1, *value.shape[2:])
            running[key].update(value.reshape(len(value), -1))

    stats = {}
    for key in keys:
        s = running[key].get_statistics()
        for step in steps:
            if key in step.transformed_keys():
                s = step.adjust_stats(key, s)
        stats[key] = s
        logging.info(
            f"Feature transforms: {key} stats over transformed values, std "
            f"{np.round(np.asarray(s['std']), 4).tolist()}"
        )
    return stats


def insert_feature_transforms(preprocessor, steps: list[FeatureTransformStep]) -> None:
    """Put `steps` right before the preprocessor's normalizer. A preprocessor loaded from a checkpoint
    already has its own: they must be the same transforms."""
    present = [s for s in preprocessor.steps if isinstance(s, FeatureTransformStep)]
    if present:
        have = [(type(s), s.get_config()) for s in present]
        want = [(type(s), s.get_config()) for s in steps]
        if have != want:
            raise ValueError(f"the preprocessor has feature transforms {have}, the config {want}")
        return
    if not steps:
        return
    at = next((i for i, s in enumerate(preprocessor.steps) if isinstance(s, NormalizerProcessorStep)), None)
    if at is None:
        raise ValueError("feature transforms go before the preprocessor's normalizer; this one has none")
    preprocessor.steps = [*preprocessor.steps[:at], *steps, *preprocessor.steps[at:]]
