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
from dataclasses import dataclass
from types import SimpleNamespace

import datasets
import numpy as np
import pytest
import torch

from lerobot.common import feature_transforms as ft
from lerobot.common.feature_transforms import (
    FeatureTransformStep,
    build_feature_transforms,
    compute_transformed_stats,
    insert_feature_transforms,
    merge_delta_timestamps,
    transformed_meta,
)
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
)
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.types import TransitionKey

STATE, ACTION = "observation.state", "action"


@ProcessorStepRegistry.register("test_motion")
@dataclass
class Motion(FeatureTransformStep):
    """The state becomes [state(t) - state(t-1), state(t)] (twice as wide); the action relative to state(t)."""

    scale: float = 1.0

    def delta_indices(self):
        return {STATE: [-1, 0]}

    def transformed_keys(self):
        return [STATE, ACTION]

    def transform_dataset_features(self, features):
        out = dict(features)
        state = dict(features[STATE])
        state["shape"] = (2 * state["shape"][0],)
        if state.get("names"):
            state["names"] = [f"d.{n}" for n in state["names"]] + list(state["names"])
        out[STATE] = state
        return out

    def adjust_stats(self, key, stats):
        return {**stats, "std": np.maximum(stats["std"], 0.5)} if key == STATE else stats

    def __call__(self, transition):
        obs = dict(transition[TransitionKey.OBSERVATION])
        window = obs[STATE]
        now = window[:, -1]
        obs[STATE] = torch.cat([(window[:, -1] - window[:, 0]) * self.scale, now], -1)
        obs.pop(f"{STATE}_is_pad", None)  # batch_to_transition puts it in both
        out = {**transition, TransitionKey.OBSERVATION: obs}
        comp = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        comp.pop(f"{STATE}_is_pad", None)
        out[TransitionKey.COMPLEMENTARY_DATA] = comp
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            out[TransitionKey.ACTION] = action - (now[:, None, :] if action.ndim == 3 else now)
        return out

    def get_config(self):
        return {"scale": self.scale}


def toy_dataset():
    """Two episodes (3 and 4 frames) of a 2-value state and action."""
    episode = np.array([0, 0, 0, 1, 1, 1, 1])
    state = np.arange(14, dtype=np.float32).reshape(7, 2) ** 1.5
    action = state + 1.0
    hf = datasets.Dataset.from_dict(
        {
            STATE: state.tolist(),
            ACTION: action.tolist(),
            "episode_index": episode.tolist(),
            "index": list(range(7)),
        }
    )
    features = {
        STATE: {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        ACTION: {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "observation.images.cam": {"dtype": "video", "shape": (3, 4, 4), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    meta = Meta()
    meta.info = SimpleNamespace(features=features)
    meta.stats = {STATE: {"mean": np.zeros(2)}, "observation.images.cam": {"mean": np.zeros(3)}}
    return SimpleNamespace(meta=meta, hf_dataset=hf), state, action, episode


class Meta:
    """What the feature transforms read of LeRobotDatasetMetadata (features is info.features)."""

    camera_keys = ["observation.images.cam"]
    fps = 10

    @property
    def features(self):
        return self.info.features


def test_merge_delta_timestamps():
    steps = [Motion()]
    assert merge_delta_timestamps({ACTION: [0.0, 0.1]}, steps, 10) == {ACTION: [0.0, 0.1], STATE: [-0.1, 0.0]}
    assert merge_delta_timestamps(None, [], 10) is None
    assert merge_delta_timestamps({STATE: [-0.1, 0.0]}, steps, 10) == {STATE: [-0.1, 0.0]}
    with pytest.raises(ValueError, match="needs observation.state"):
        merge_delta_timestamps({STATE: [-0.2, 0.0]}, steps, 10)


def test_build_refuses_other_steps():
    assert isinstance(build_feature_transforms([{"type": "test_motion", "scale": 2.0}])[0], Motion)
    with pytest.raises(ValueError, match="not a FeatureTransformStep"):
        build_feature_transforms([{"type": "device_processor"}])


def test_windows_as_the_dataset_reader():
    """Clamped to the episode, padding where the dataset reader marks it."""
    _, state, _, episode = toy_dataset()
    batch = ft._windows({STATE: state}, episode, np.arange(7), {STATE: [-1, 0, 1]})
    for row in range(7):
        start, end = (0, 3) if row < 3 else (3, 7)
        want = [min(max(row + d, start), end - 1) for d in (-1, 0, 1)]
        pad = [not (start <= row + d < end) for d in (-1, 0, 1)]
        np.testing.assert_array_equal(batch[STATE][row].numpy(), state[want])
        assert batch[f"{STATE}_is_pad"][row].tolist() == pad


def test_stats_of_the_transformed_values(monkeypatch):
    monkeypatch.setattr(ft, "STATS_BATCH_SIZE", 3)  # batches across the episode boundary
    dataset, state, action, episode = toy_dataset()
    chunk = [0, 1, 2]
    stats = compute_transformed_stats(dataset, [Motion()], {STATE: [-1, 0], ACTION: chunk})

    prev = np.array([0, 0, 1, 3, 3, 4, 5])
    want_state = np.concatenate([state - state[prev], state], 1)
    np.testing.assert_allclose(stats[STATE]["mean"], want_state.mean(0), rtol=1e-6)
    np.testing.assert_allclose(stats[STATE]["std"], np.maximum(want_state.std(0), 0.5), rtol=1e-5)
    rel = [
        action[r + k] - state[r]
        for r in range(7)
        for k in chunk
        if episode[min(r + k, 6)] == episode[r] and r + k < 7
    ]
    np.testing.assert_allclose(stats[ACTION]["mean"], np.mean(rel, 0), rtol=1e-6)
    np.testing.assert_allclose(stats[ACTION]["std"], np.std(rel, 0), rtol=1e-5)
    assert int(stats[ACTION]["count"][0]) == len(rel)


def test_transformed_meta_leaves_the_dataset_alone():
    dataset, *_ = toy_dataset()
    meta = transformed_meta(dataset.meta, [Motion()], {STATE: {"mean": np.ones(4)}})
    assert meta.features[STATE]["shape"] == (4,)
    assert meta.features[STATE]["names"] == ["d.a", "d.b", "a", "b"]
    assert dataset.meta.features[STATE]["shape"] == (2,)
    assert meta.stats[STATE]["mean"].shape == (4,) and dataset.meta.stats[STATE]["mean"].shape == (2,)
    assert "observation.images.cam" in meta.stats


def pipeline(stats):
    features = {
        STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.ACTION: NormalizationMode.MEAN_STD}
    return PolicyProcessorPipeline(
        steps=[
            DeviceProcessorStep(device="cpu"),
            NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats),
        ],
        name="policy_preprocessor",
    )


def test_inserted_before_the_normalizer_and_saved(tmp_path):
    stats = {
        STATE: {"mean": np.full(4, 1.0), "std": np.full(4, 2.0)},
        ACTION: {"mean": np.zeros(2), "std": np.ones(2)},
    }
    pre = pipeline(stats)
    insert_feature_transforms(pre, [Motion(scale=3.0)])
    assert [type(s) for s in pre.steps] == [DeviceProcessorStep, Motion, NormalizerProcessorStep]

    batch = {STATE: torch.tensor([[[1.0, 2.0], [2.0, 4.0]]]), ACTION: torch.tensor([[3.0, 5.0]])}
    out = pre(batch)
    np.testing.assert_allclose(out[STATE].numpy(), [[(3 - 1) / 2, (6 - 1) / 2, (2 - 1) / 2, (4 - 1) / 2]])
    np.testing.assert_allclose(out[ACTION].numpy(), [[1.0, 1.0]])

    pre.save_pretrained(tmp_path)
    loaded = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    assert [type(s) for s in loaded.steps] == [DeviceProcessorStep, Motion, NormalizerProcessorStep]
    assert loaded.steps[1].scale == 3.0
    np.testing.assert_allclose(loaded(batch)[STATE].numpy(), out[STATE].numpy())

    insert_feature_transforms(loaded, [Motion(scale=3.0)])  # already there: kept as is
    assert len(loaded.steps) == 3
    with pytest.raises(ValueError, match="feature transforms"):
        insert_feature_transforms(loaded, [Motion(scale=1.0)])


def test_registry_loads_plugins_for_an_unknown_name(monkeypatch):
    import lerobot.common.train_hooks as hooks

    calls = []

    def load():
        calls.append(1)
        if "test_late" not in ProcessorStepRegistry._registry:
            ProcessorStepRegistry.register("test_late")(type("Late", (Motion,), {}))

    monkeypatch.setattr(hooks, "load_train_hook_plugins", load)
    try:
        assert ProcessorStepRegistry.get("test_late").__name__ == "Late"
        assert calls == [1]
    finally:
        ProcessorStepRegistry.unregister("test_late")


def test_transition_round_trip_keeps_padding():
    """The stats pass hands the steps an EnvTransition, as the preprocessor does."""
    batch = {STATE: torch.zeros(2, 2, 2), f"{STATE}_is_pad": torch.tensor([[True, False], [False, False]])}
    transition = batch_to_transition(batch)
    assert f"{STATE}_is_pad" in transition[TransitionKey.COMPLEMENTARY_DATA]
    assert f"{STATE}_is_pad" not in transition_to_batch(Motion()(transition))
