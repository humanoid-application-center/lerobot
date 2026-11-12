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
"""Training-time hooks: batch transforms and probes, named in the train config.

`train_transforms` and `train_probes` in `TrainPipelineConfig` are lists of
`{"type": <registered name>, ...arguments}`. lerobot itself registers none: third-party
packages declare an entry point in the group `lerobot.train_hooks`, whose module registers its
classes with `@register_train_transform(name)` / `@register_train_probe(name)` when imported.

- A transform runs on every training batch after the preprocessor (normalized space), in the
  order listed. Training only: the eval loop and inference never see it, which is why it is not a
  `ProcessorStep` (those are saved with the policy and run at inference).
- A probe observes the run: it returns extra metrics, logged with the training ones
  (`after_backward` once gradients are complete, before the optimizer step; `on_log` at each log
  step; `on_eval` at each eval step), and may write files beside a checkpoint (`on_checkpoint`,
  main process only, after lerobot has saved it). The metric methods run on every rank; a probe that
  needs a cross-rank figure reduces it itself.
"""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENTRY_POINT_GROUP = "lerobot.train_hooks"

_TRANSFORMS: dict[str, type[TrainTransform]] = {}
_PROBES: dict[str, type[TrainProbe]] = {}
_entry_points_loaded = False


@dataclass
class TrainContext:
    """What a transform or a probe may need, built once the policy and data loaders are prepared."""

    cfg: Any  # TrainPipelineConfig
    policy: Any  # as prepared by accelerate; use accelerator.unwrap_model for the bare policy
    preprocessor: Any
    dataset: Any
    eval_dataset: Any | None
    accelerator: Any
    output_dir: Path | None
    # The dataset's metadata as the policy sees it: with feature transforms (lerobot.common.feature_transforms),
    # their features and statistics; else the dataset's own.
    policy_meta: Any | None = None


class TrainTransform:
    """A training-batch transform. Subclasses take their config arguments as keyword arguments."""

    def __init__(self, ctx: TrainContext, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError(f"{type(self).__name__} got unknown arguments: {sorted(kwargs)}")
        self.ctx = ctx

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class TrainProbe:
    """A source of extra training metrics. Every method may return an empty dict."""

    def __init__(self, ctx: TrainContext, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError(f"{type(self).__name__} got unknown arguments: {sorted(kwargs)}")
        self.ctx = ctx

    def after_backward(self, step: int) -> dict[str, float]:
        """Gradients of optimizer step `step` are complete (and clipped); the update has not run."""
        return {}

    def on_log(self, step: int, batch: dict[str, Any], output: dict | None) -> dict[str, float]:
        return {}

    def on_eval(self, step: int) -> dict[str, float]:
        return {}

    def on_checkpoint(self, step: int, checkpoint_dir: Path) -> None:
        """Checkpoint `step` is saved in `checkpoint_dir` (its policy in `pretrained_model/`)."""


def register_train_transform(name: str):
    def wrap(cls: type[TrainTransform]) -> type[TrainTransform]:
        if name in _TRANSFORMS and _TRANSFORMS[name] is not cls:
            raise ValueError(f"train transform {name!r} is already registered by {_TRANSFORMS[name]}")
        _TRANSFORMS[name] = cls
        return cls

    return wrap


def register_train_probe(name: str):
    def wrap(cls: type[TrainProbe]) -> type[TrainProbe]:
        if name in _PROBES and _PROBES[name] is not cls:
            raise ValueError(f"train probe {name!r} is already registered by {_PROBES[name]}")
        _PROBES[name] = cls
        return cls

    return wrap


def load_train_hook_plugins() -> None:
    """Import every module declared in the `lerobot.train_hooks` entry-point group (once)."""
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        ep.load()
        logging.info("Loaded train hooks plugin %s (%s)", ep.name, ep.value)


def _build(specs: list[dict[str, Any]], registry: dict, kind: str, ctx: TrainContext) -> list:
    if specs:
        load_train_hook_plugins()
    built = []
    for spec in specs:
        spec = dict(spec)
        name = spec.pop("type", None)
        if name not in registry:
            raise ValueError(
                f"unknown train {kind} {name!r}; registered: {sorted(registry)}. A plugin registers "
                f"its {kind}s through the {ENTRY_POINT_GROUP!r} entry-point group."
            )
        built.append(registry[name](ctx, **spec))
    return built


class TrainHooks:
    """The transforms and probes of one training run."""

    def __init__(self, transforms: list[TrainTransform], probes: list[TrainProbe]) -> None:
        self.transforms = transforms
        self.probes = probes

    @classmethod
    def from_config(cls, ctx: TrainContext) -> TrainHooks:
        return cls(
            _build(ctx.cfg.train_transforms, _TRANSFORMS, "transform", ctx),
            _build(ctx.cfg.train_probes, _PROBES, "probe", ctx),
        )

    def transform(self, batch: dict[str, Any]) -> dict[str, Any]:
        for t in self.transforms:
            batch = t(batch)
        return batch

    def after_backward(self, step: int) -> dict[str, float]:
        return self._collect("after_backward", step)

    def on_log(self, step: int, batch: dict[str, Any], output: dict | None) -> dict[str, float]:
        return self._collect("on_log", step, batch, output)

    def on_eval(self, step: int) -> dict[str, float]:
        return self._collect("on_eval", step)

    def on_checkpoint(self, step: int, checkpoint_dir: Path) -> None:
        for p in self.probes:
            p.on_checkpoint(step, checkpoint_dir)

    def _collect(self, method: str, *args: Any) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in self.probes:
            out.update(getattr(p, method)(*args) or {})
        return out
