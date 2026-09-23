"""R-039 splice gates, on the CPU stand-in head from conftest.

The splice is a wrapper on `get_action_with_features`, the same seam the
capture uses. Per forward it builds five feature sets from the target's own
inputs and a SOURCE (nominal) feature set -- P target, N source, T text
positions from source, I image positions from source, S state token from source
-- runs the head on each under the SAME fixed noise, records all five action
chunks, and returns the one arm that drives the rollout.
"""
from __future__ import annotations

import pytest
import torch

from conftest import FakeActionHead, FakePolicy, STATE_DIM, STATE_TOKENS, VL_DIM, VL_TOKENS

from vla_harness.capture.splice import attach_splice, transfer_fraction


class MixingHead(FakeActionHead):
    """conftest's head decodes token-by-token, so its action rows never see
    `vl_embeds` or the state token -- fine for the capture gates, useless for
    a splice test. This one adds a pooled VL context and the state token into
    every action row before decoding, the way cross-attention and the
    state-token prefix do in the real DiT, so a splice CAN move the action."""

    def get_action_with_features(self, backbone_features, state_features,
                                 backbone_output, action_input, options=None):
        vl_embeds = backbone_features
        batch_size = vl_embeds.shape[0]
        actions = torch.randn(size=(batch_size, self.action_horizon, self.action_dim),
                              dtype=vl_embeds.dtype, device=vl_embeds.device)
        dt = 1.0 / self.num_inference_timesteps
        ctx = vl_embeds.mean(dim=1, keepdim=True) + state_features.mean(dim=1, keepdim=True)
        for _ in range(self.num_inference_timesteps):
            action_features = self.action_encoder(actions) + ctx
            sa_embs = torch.cat((state_features, action_features), dim=1)
            pred = self.action_decoder(sa_embs)
            actions = actions + dt * pred[:, -self.action_horizon:]
        return {"action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features}


@pytest.fixture
def policy() -> FakePolicy:
    torch.manual_seed(0)
    p = FakePolicy()
    p.action_head = MixingHead()
    return p.eval()


def _inputs(seed: int, n_text: int = 3):
    torch.manual_seed(seed)
    mask = torch.zeros(1, VL_TOKENS, dtype=torch.bool)
    mask[0, :VL_TOKENS - n_text] = True
    backbone_output = {"backbone_features": torch.randn(1, VL_TOKENS, VL_DIM),
                       "image_mask": mask}
    action_input = {"state": torch.randn(1, STATE_TOKENS, STATE_DIM)}
    return backbone_output, action_input


def _fresh(bo, ai):
    """`process_backbone_output` overwrites `backbone_features` IN PLACE
    (groot_n1_7.py:563-564, reproduced by the stand-in), so an input dict is
    single-use. Two calls need two copies, not one dict twice."""
    return ({k: v.clone() for k, v in bo.items()}, {k: v.clone() for k, v in ai.items()})


def _source_from(policy: FakePolicy, seed: int, n_text: int = 3):
    """What a paired render yields: the head's OWN encoded features for the
    nominal inputs, i.e. post-`vl_self_attention` tokens and the state token."""
    bo, ai = _inputs(seed, n_text)
    feats = policy.action_head._encode_features(bo, ai)
    return {"backbone_features": feats["backbone_features"],
            "state_features": feats["state_features"],
            "image_mask": bo["image_mask"]}


def test_fixed_noise_makes_the_draw_a_function_of_the_forward_index(policy):
    src = _source_from(policy, seed=7)
    bo, ai = _inputs(1)                       # built ONCE: the helper reseeds
    h = attach_splice(policy, source=lambda i: src, drive=None, noise_key=lambda i: 1000 + i)
    a1 = policy.action_head.get_action(*_fresh(bo, ai))["action_pred"].clone()
    h.begin_episode()
    a2 = policy.action_head.get_action(*_fresh(bo, ai))["action_pred"].clone()
    h.detach()
    assert torch.equal(a1, a2), "same inputs, same forward index, different action"
    b1 = policy.action_head.get_action(*_fresh(bo, ai))["action_pred"]
    b2 = policy.action_head.get_action(*_fresh(bo, ai))["action_pred"]
    assert not torch.equal(b1, b2), "the unwrapped head is supposed to be stochastic"


def test_fixed_noise_leaves_the_outer_generator_untouched(policy):
    src = _source_from(policy, seed=7)
    bo, ai = _inputs(1)
    h = attach_splice(policy, source=lambda i: src, drive=None, noise_key=lambda i: 5)
    before = torch.get_rng_state()
    policy.action_head.get_action(bo, ai)
    after = torch.get_rng_state()
    h.detach()
    assert torch.equal(before, after)


def test_records_hold_one_action_per_arm_per_forward(policy):
    src = _source_from(policy, seed=7)
    h = attach_splice(policy, source=lambda i: src, drive=None, noise_key=lambda i: 5)
    out = policy.action_head.get_action(*_inputs(1))
    policy.action_head.get_action(*_inputs(2))
    assert len(h.records) == 2
    rec = h.records[0]
    assert rec["forward_idx"] == 0
    for arm in ("P", "N", "T", "I", "S"):
        assert rec["action"][arm].shape == out["action_pred"].shape, arm
    assert torch.equal(rec["action"]["P"], out["action_pred"]), "drive=None returns P"
    h.detach()


def test_identical_source_and_target_give_five_identical_actions(policy):
    bo, ai = _inputs(3)
    src = _source_from(policy, seed=3)
    h = attach_splice(policy, source=lambda i: src, drive=None, noise_key=lambda i: 5)
    policy.action_head.get_action(bo, ai)
    acts = h.records[0]["action"]
    h.detach()
    for arm in ("N", "T", "I", "S"):
        assert torch.equal(acts["P"], acts[arm]), arm


def test_text_arm_replaces_only_the_text_token_positions(policy):
    bo, ai = _inputs(1)
    src = _source_from(policy, seed=7)
    h = attach_splice(policy, source=lambda i: src, drive="T", noise_key=lambda i: 5)
    out = policy.action_head.get_action(bo, ai)
    h.detach()
    used = out["backbone_features"]
    text = ~bo["image_mask"]
    assert torch.equal(used[text], src["backbone_features"][text])
    target_feats = policy.action_head._encode_features(*_inputs(1))["backbone_features"]
    assert torch.equal(used[~text], target_feats[~text])
    assert torch.equal(out["state_features"],
                       policy.action_head._encode_features(*_inputs(1))["state_features"])


def test_image_arm_replaces_only_the_image_token_positions(policy):
    bo, ai = _inputs(1)
    src = _source_from(policy, seed=7)
    h = attach_splice(policy, source=lambda i: src, drive="I", noise_key=lambda i: 5)
    out = policy.action_head.get_action(bo, ai)
    h.detach()
    used = out["backbone_features"]
    img = bo["image_mask"]
    assert torch.equal(used[img], src["backbone_features"][img])
    target_feats = policy.action_head._encode_features(*_inputs(1))["backbone_features"]
    assert torch.equal(used[~img], target_feats[~img])


def test_state_arm_replaces_the_state_token_and_nothing_else(policy):
    bo, ai = _inputs(1)
    src = _source_from(policy, seed=7)
    h = attach_splice(policy, source=lambda i: src, drive="S", noise_key=lambda i: 5)
    out = policy.action_head.get_action(bo, ai)
    h.detach()
    assert torch.equal(out["state_features"], src["state_features"])
    target_feats = policy.action_head._encode_features(*_inputs(1))["backbone_features"]
    assert torch.equal(out["backbone_features"], target_feats)


def test_driven_arm_action_is_the_recorded_one(policy):
    src = _source_from(policy, seed=7)
    h = attach_splice(policy, source=lambda i: src, drive="I", noise_key=lambda i: 5)
    out = policy.action_head.get_action(*_inputs(1))
    assert torch.equal(out["action_pred"], h.records[0]["action"]["I"])
    assert not torch.equal(out["action_pred"], h.records[0]["action"]["P"])
    h.detach()


def test_token_count_mismatch_refuses_rather_than_broadcasting(policy):
    src = _source_from(policy, seed=7, n_text=3)
    src["backbone_features"] = src["backbone_features"][:, :-1]
    src["image_mask"] = src["image_mask"][:, :-1]
    h = attach_splice(policy, source=lambda i: src, drive=None, noise_key=lambda i: 5)
    with pytest.raises(RuntimeError, match="token"):
        policy.action_head.get_action(*_inputs(1))
    h.detach()


def test_image_mask_mismatch_between_source_and_target_refuses(policy):
    src = _source_from(policy, seed=7, n_text=3)
    h = attach_splice(policy, source=lambda i: src, drive=None, noise_key=lambda i: 5)
    with pytest.raises(RuntimeError, match="mask"):
        policy.action_head.get_action(*_inputs(1, n_text=4))
    h.detach()


def test_detach_restores_the_head_and_stops_recording(policy):
    orig = policy.action_head.get_action_with_features
    src = _source_from(policy, seed=7)
    h = attach_splice(policy, source=lambda i: src, drive=None, noise_key=lambda i: 5)
    h.detach()
    assert "get_action_with_features" not in vars(policy.action_head), \
        "detach must remove the instance override, not install a new one"
    assert policy.action_head.get_action_with_features.__func__ is orig.__func__
    policy.action_head.get_action(*_inputs(1))
    assert h.records == []


def test_transfer_fraction_is_one_when_patched_equals_source_and_zero_at_target():
    src = torch.randn(1, 6, 12)
    tgt = torch.randn(1, 6, 12)
    assert transfer_fraction(src, tgt, src, live_dims=5) == pytest.approx(1.0)
    assert transfer_fraction(tgt, tgt, src, live_dims=5) == pytest.approx(0.0)


def test_transfer_fraction_ignores_padding_dims():
    src = torch.zeros(1, 6, 12)
    tgt = torch.zeros(1, 6, 12)
    tgt[..., :5] = 1.0
    patched = tgt.clone()
    patched[..., 5:] = 100.0            # padding only; must not count
    assert transfer_fraction(patched, tgt, src, live_dims=5) == pytest.approx(0.0)
