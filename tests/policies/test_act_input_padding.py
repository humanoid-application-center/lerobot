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
"""ACT leaves the observations marked `<key>_is_pad` out of the attention."""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

CAMS = ["observation.images.left", "observation.images.right"]


@pytest.fixture
def model():
    torch.manual_seed(0)
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            **{c: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)) for c in CAMS},
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
        chunk_size=5,
        n_action_steps=5,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=2,
        n_vae_encoder_layers=1,
        latent_dim=8,
        pretrained_backbone_weights=None,
    )
    return ACT(config).eval()


def make_batch(b: int = 3) -> dict:
    batch = {OBS_STATE: torch.randn(b, 4), **{c: torch.randn(b, 3, 64, 64) for c in CAMS}}
    batch[OBS_IMAGES] = [batch[c] for c in CAMS]
    return batch


def predict(model, batch: dict) -> torch.Tensor:
    with torch.no_grad():
        return model(batch)[0]


def test_no_mask_and_all_false_mask_agree(model):
    batch = make_batch()
    masked = dict(batch, **{f"{CAMS[0]}_is_pad": torch.zeros(3, dtype=torch.bool)})
    torch.testing.assert_close(predict(model, batch), predict(model, masked))


@pytest.mark.parametrize("key", [CAMS[0], CAMS[1], OBS_STATE])
def test_a_padded_observation_does_not_reach_the_output(model, key):
    batch = make_batch()
    is_pad = torch.tensor([True, False, False])
    changed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    changed[key][0] = torch.randn_like(changed[key][0]) * 10
    changed[OBS_IMAGES] = [changed[c] for c in CAMS]

    out = predict(model, dict(batch, **{f"{key}_is_pad": is_pad}))
    out_changed = predict(model, dict(changed, **{f"{key}_is_pad": is_pad}))

    torch.testing.assert_close(out[0], out_changed[0])  # the masked sample ignores its new value
    assert not torch.allclose(predict(model, batch)[0], predict(model, changed)[0])  # unmasked, it does not
    torch.testing.assert_close(out[1:], out_changed[1:])


def test_a_window_mask_uses_its_newest_step(model):
    batch = make_batch()
    window = torch.tensor([[False, True], [True, False], [False, False]])
    newest = window[:, -1]
    torch.testing.assert_close(
        predict(model, dict(batch, **{f"{CAMS[0]}_is_pad": window})),
        predict(model, dict(batch, **{f"{CAMS[0]}_is_pad": newest})),
    )


def test_training_forward_with_vae_and_padded_state(model):
    model.train()
    batch = make_batch()
    batch[ACTION] = torch.randn(3, 5, 4)
    batch["action_is_pad"] = torch.zeros(3, 5, dtype=torch.bool)
    batch[f"{OBS_STATE}_is_pad"] = torch.tensor([True, False, True])
    actions, (mu, log_sigma_x2) = model(batch)
    assert actions.shape == (3, 5, 4)
    assert torch.isfinite(actions).all() and torch.isfinite(mu).all() and torch.isfinite(log_sigma_x2).all()
