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
from types import SimpleNamespace

import pytest

from lerobot.common.train_hooks import (
    TrainContext,
    TrainHooks,
    TrainProbe,
    TrainTransform,
    register_train_probe,
    register_train_transform,
)


@register_train_transform("test_add")
class Add(TrainTransform):
    def __init__(self, ctx, value: float = 1.0):
        super().__init__(ctx)
        self.value = value

    def __call__(self, batch):
        return {**batch, "x": batch["x"] + self.value}


@register_train_transform("test_double")
class Double(TrainTransform):
    def __call__(self, batch):
        return {**batch, "x": batch["x"] * 2}


@register_train_probe("test_probe")
class Probe(TrainProbe):
    def after_backward(self, step):
        return {"grad_probe": float(step)}

    def on_log(self, step, batch, output):
        return {"seen_x": float(batch["x"])}

    def on_checkpoint(self, step, checkpoint_dir):
        (checkpoint_dir / "probe.txt").write_text(str(step))


def ctx(transforms=(), probes=()):
    cfg = SimpleNamespace(train_transforms=list(transforms), train_probes=list(probes))
    return TrainContext(cfg, None, None, None, None, None, None)


def test_transforms_run_in_config_order():
    hooks = TrainHooks.from_config(ctx([{"type": "test_add", "value": 3}, {"type": "test_double"}]))
    assert hooks.transform({"x": 1.0})["x"] == 8.0
    hooks = TrainHooks.from_config(ctx([{"type": "test_double"}, {"type": "test_add", "value": 3}]))
    assert hooks.transform({"x": 1.0})["x"] == 5.0


def test_probes_collect_their_metrics():
    hooks = TrainHooks.from_config(ctx(probes=[{"type": "test_probe"}]))
    assert hooks.after_backward(7) == {"grad_probe": 7.0}
    assert hooks.on_log(7, {"x": 2.0}, None) == {"seen_x": 2.0}
    assert hooks.on_eval(7) == {}


def test_probes_write_beside_a_checkpoint(tmp_path):
    hooks = TrainHooks.from_config(ctx(probes=[{"type": "test_probe"}]))
    hooks.on_checkpoint(30, tmp_path)
    assert (tmp_path / "probe.txt").read_text() == "30"


def test_no_hooks_is_a_no_op():
    hooks = TrainHooks.from_config(ctx())
    batch = {"x": 1.0}
    assert hooks.transform(batch) is batch
    assert hooks.after_backward(1) == {}


def test_unknown_type_and_arguments_are_refused():
    with pytest.raises(ValueError, match="unknown train transform 'nope'"):
        TrainHooks.from_config(ctx([{"type": "nope"}]))
    with pytest.raises(TypeError, match="unknown arguments"):
        TrainHooks.from_config(ctx([{"type": "test_double", "factor": 3}]))


def test_a_name_cannot_be_taken_twice():
    with pytest.raises(ValueError, match="already registered"):

        @register_train_transform("test_double")
        class Other(TrainTransform):
            pass
