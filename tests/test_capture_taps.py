"""Tier-1 gates for the embedding capture (V1-V7, V13-V14).

Every gate here runs on CPU against `tests/conftest.py`'s stand-in in
milliseconds. Tier 2 (V8-V12) re-asserts them against the real checkpoint in
`experiments/capture_smoke.py`; neither tier substitutes for the other.

Gate numbering follows docs/superpowers/specs/2026-09-18-embedding-capture-design.md.
"""
from __future__ import annotations

import json

import numpy
import pytest
import torch

from conftest import (ACTION_DIM, ACTION_HORIZON, N_TIMESTEPS, LIVE_DIMS, STATE_TOKENS,
                      VL_DIM, FakeActionHead, FakePolicy)

from vla_harness.capture import CaptureSink, Role
from vla_harness.capture.groot_features import attach_groot_capture


# --- V1: the S1/S2 trap -----------------------------------------------------

def test_v1_vl_encoder_is_pre_vlln_not_a_second_copy_of_vl_adapted(policy):
    """A hook reading `backbone_features` after process_backbone_output captures
    the post-attention tensor twice. vlln sits between the two reads, so the
    duplicate looks plausible -- this paired assertion is what catches it."""
    sink = CaptureSink()
    attach_groot_capture(policy, sink)
    sink.begin_episode()
    policy.one_forward(seed=1)
    rec = sink.current()[0]

    s1 = torch.as_tensor(rec["vl_encoder"])
    s1_5 = torch.as_tensor(rec["vl_normed"])
    s2 = torch.as_tensor(rec["vl_adapted"])

    # the captured pre-vlln tensor must actually be pre-vlln
    torch.testing.assert_close(policy.action_head.vlln(s1), s1_5, rtol=1e-4, atol=1e-4)
    # and must not be the post-attention tensor wearing a different name
    assert not torch.allclose(s1, s2, rtol=1e-3, atol=1e-3)


# --- V2: S4t reconstruction -------------------------------------------------

def test_v2_euler_path_reconstructs_the_models_own_action(policy):
    """With RTC off, vel_strength is identically 1, so the denoise loop is
    exactly actions_0 + dt * sum(pred[:, -H:]). If that does not reproduce the
    model's returned action_pred, the decoder hook is on the wrong module.

    This is only a test because actions_0 is OBSERVED. Deriving it as
    action_pred - dt*sum(pred) would make this assertion arithmetic and it would
    pass for any hook at all."""
    sink = CaptureSink()
    attach_groot_capture(policy, sink)
    sink.begin_episode()
    out = policy.one_forward(seed=2)
    rec = sink.current()[0]

    actions_0 = torch.as_tensor(rec["actions_init"])
    path = torch.as_tensor(rec["decode_path"])          # (n_timesteps, B, H, dim)
    dt = 1.0 / N_TIMESTEPS

    rebuilt = actions_0 + dt * path.sum(dim=0)
    torch.testing.assert_close(rebuilt, out["action_pred"], rtol=1e-4, atol=1e-4)


def test_v2b_decode_path_is_sliced_exactly_as_the_model_slices_it(policy):
    """action_decoder emits a state-token prefix: sa_embs is
    cat(state_features, action_features), so pred is (B, STATE_TOKENS+H, dim)
    and the model uses pred[:, -H:]. Storing it unsliced makes V2 sum one row
    too many -- which reads as a tolerance failure, not a wrong-rows failure."""
    sink = CaptureSink()
    attach_groot_capture(policy, sink)
    sink.begin_episode()
    policy.one_forward(seed=3)
    rec = sink.current()[0]

    assert rec["decode_path_preslice_tokens"] == STATE_TOKENS + ACTION_HORIZON
    path = torch.as_tensor(rec["decode_path"])
    assert path.shape == (N_TIMESTEPS, 1, ACTION_HORIZON, ACTION_DIM)


# --- V4: RNG neutrality -----------------------------------------------------

class _DrawsAfterDenoise(FakeActionHead):
    """A head that consumes RNG AFTER the denoise loop's initial draw.

    The shared stand-in does not, and neither -- as far as the code reads -- does
    GR00T under inference_mode with dropout inert. That makes the plain fake
    unable to distinguish a wrapper that restores the generator from one that
    leaves it advanced: V4 against it passed whether or not the restore was
    there, which is a gate reporting green by not being exercised.

    So this subclass exists to make the CONTRACT observable -- "the generator
    ends where an unwrapped run would leave it, whatever the model does
    internally" -- rather than to claim GR00T draws here. If GR00T ever does,
    this is the test that already covers it."""

    def get_action_with_features(self, *args, **kwargs):
        out = super().get_action_with_features(*args, **kwargs)
        torch.randn(3)          # the post-draw the plain fake lacks
        return out


def test_v4_capture_leaves_the_generator_where_an_unwrapped_run_would():
    """Observing actions_0 means rewinding the generator. Output identity alone
    would not catch a wrapper that leaves RNG advanced -- which would silently
    change the noise every subsequent episode draws."""
    def run(attach: bool):
        torch.manual_seed(0)
        p = FakePolicy().eval()
        p.action_head = _DrawsAfterDenoise().eval()
        detach = None
        if attach:
            sink = CaptureSink()
            detach = attach_groot_capture(p, sink)
            sink.begin_episode()
        torch.manual_seed(7)
        p.one_forward()
        state = torch.get_rng_state()
        if detach:
            detach()
        return state

    assert torch.equal(run(attach=False), run(attach=True))


# --- V3: no behaviour change ------------------------------------------------

def test_v3_wrapper_returns_the_models_own_object_not_a_copy(policy):
    """Asserted with `is`, not `==`: a wrapper that rebuilt an equal BatchFeature
    would compare equal while having silently replaced what the caller gets."""
    seen = {}
    original = policy.action_head.get_action

    def spy(*a, **k):
        out = original(*a, **k)
        seen["out"] = out
        return out

    policy.action_head.get_action = spy
    sink = CaptureSink()
    attach_groot_capture(policy, sink)
    sink.begin_episode()
    returned = policy.one_forward(seed=4)

    assert returned is seen["out"]


# --- V5: shapes and dtypes --------------------------------------------------

def test_v5_action_pred_is_captured_unsliced_at_the_tap(policy):
    """Captured at full width and sliced afterwards, with `sliced_from`
    recorded -- so the slice can never quietly become the thing measured."""
    sink = CaptureSink()
    attach_groot_capture(policy, sink)
    sink.begin_episode()
    policy.one_forward(seed=5)
    rec = sink.current()[0]

    assert torch.as_tensor(rec["action_pred"]).shape == (1, ACTION_HORIZON, ACTION_DIM)
    assert torch.as_tensor(rec["vl_encoder"]).shape[-1] == VL_DIM
    assert torch.as_tensor(rec["state_encoded"]).shape[-1] == VL_DIM


# --- V6: one row per model forward ------------------------------------------

def test_v6_one_capture_row_per_get_action_call(policy):
    sink = CaptureSink()
    attach_groot_capture(policy, sink)
    sink.begin_episode()
    for i in range(3):
        policy.one_forward(seed=10 + i)

    assert len(sink.current()) == 3


# --- V7: off is off ---------------------------------------------------------

def test_v7_no_hooks_registered_when_capture_is_not_attached(policy):
    """This package must stay off the critical path of every other run."""
    head = policy.action_head
    assert len(head.vlln._forward_pre_hooks) == 0
    assert len(head.vlln._forward_hooks) == 0
    assert len(head.action_decoder._forward_hooks) == 0


def test_v7b_detach_removes_every_hook_and_restores_every_method(policy):
    head = policy.action_head
    before = (head.get_action, head.get_action_with_features)
    detach = attach_groot_capture(policy, CaptureSink())
    detach()

    assert len(head.vlln._forward_pre_hooks) == 0
    assert len(head.vlln._forward_hooks) == 0
    assert len(head.action_decoder._forward_hooks) == 0
    assert (head.get_action, head.get_action_with_features) == before


# --- V13: episode-boundary leak on the error path ---------------------------

def test_v13_unflushed_episode_is_discarded_loudly_not_carried_forward(policy):
    """If an episode raises part-way, the flush never runs. Carrying its
    forwards into the next episode is the same silent misjoin the explicit
    rollout_id flush exists to prevent, arriving by the exception path."""
    sink = CaptureSink()
    attach_groot_capture(policy, sink)
    sink.begin_episode()
    policy.one_forward(seed=6)          # episode 1 forward, then it "crashes"

    with pytest.warns(UserWarning, match="discarding"):
        sink.begin_episode()            # episode 2 starts with a dirty buffer

    policy.one_forward(seed=7)
    assert len(sink.current()) == 1     # episode 1's forward did not leak in


# --- writer: pooling, slicing, manifest -------------------------------------

def _captured(policy, n=2):
    sink = CaptureSink()
    attach_groot_capture(policy, sink)
    sink.begin_episode()
    for i in range(n):
        policy.one_forward(seed=20 + i)
    return sink


def test_writer_stores_both_poolings_because_the_choice_predates_the_result(policy, tmp_path):
    """EXP_EMBEDDING_OOD.md §6: mean-pooling discards spatial structure, so both
    are recorded. Storing one would let the choice be made after seeing which
    separated better."""
    sink = _captured(policy)
    path = sink.flush(tmp_path, rollout_id="r1", live_dims=LIVE_DIMS, meta={})
    with numpy.load(path) as z:
        assert "vl_encoder_mean" in z and "vl_encoder_max" in z
        assert not numpy.allclose(z["vl_encoder_mean"], z["vl_encoder_max"])


def test_writer_slices_to_live_dims_and_records_what_it_sliced_from(policy, tmp_path):
    sink = _captured(policy)
    path = sink.flush(tmp_path, rollout_id="r1", live_dims=LIVE_DIMS, meta={})
    with numpy.load(path) as z:
        assert z["action_pred"].shape == (2, ACTION_HORIZON, LIVE_DIMS)
        assert z["decode_path"].shape == (2, N_TIMESTEPS, ACTION_HORIZON, LIVE_DIMS)
        assert int(z["sliced_from"]) == ACTION_DIM
        assert z["action_pred"].dtype == numpy.float16


def test_v14_manifest_is_unique_on_rollout_id_across_a_resume(policy, tmp_path):
    """Resumable runs re-enter run_cell's cache-skip path. A second manifest row
    for the same rollout_id would double-count that episode in any downstream
    analysis, silently."""
    for _ in range(2):                      # same episode captured twice
        sink = _captured(policy)
        sink.flush(tmp_path, rollout_id="same-episode", live_dims=LIVE_DIMS, meta={})

    rows = [json.loads(l) for l in open(tmp_path / "capture_manifest.jsonl")]
    assert len(rows) == 1
    assert rows[0]["rollout_id"] == "same-episode"


def test_writer_records_the_forward_index_to_env_step_mapping(policy, tmp_path):
    sink = _captured(policy, n=3)
    sink.flush(tmp_path, rollout_id="r1", live_dims=LIVE_DIMS,
               meta={}, env_steps=[0, 16, 32])
    rows = [json.loads(l) for l in open(tmp_path / "capture_manifest.jsonl")]
    assert rows[0]["env_step"] == [0, 16, 32]
    assert rows[0]["n_forwards"] == 3


# --- S4: resample spread ----------------------------------------------------

def test_s4_resampling_records_spread_without_disturbing_the_returned_action(policy):
    """The k-1 extra draws are the only self-consistency signal available:
    RTC is off on this path, so there is no chunk overlap and no true
    chunk-to-chunk disagreement to measure."""
    sink = CaptureSink()
    attach_groot_capture(policy, sink, k_resample=4)
    sink.begin_episode()
    out = policy.one_forward(seed=8)
    rec = sink.current()[0]

    spread = torch.as_tensor(rec["resample_spread"])
    assert spread.shape == (1, ACTION_HORIZON, ACTION_DIM)
    assert (spread > 0).any()                       # the draws actually differ
    # the model's own action is untouched by the extra draws
    torch.testing.assert_close(torch.as_tensor(rec["action_pred"]), out["action_pred"])


def test_s4_resampling_still_leaves_the_generator_where_it_was():
    """k-1 extra draws consume RNG. Without the restore they would shift the
    noise of every subsequent episode."""
    def run(k):
        torch.manual_seed(0)
        p = FakePolicy().eval()
        p.action_head = _DrawsAfterDenoise().eval()
        sink = CaptureSink()
        detach = attach_groot_capture(p, sink, k_resample=k)
        sink.begin_episode()
        torch.manual_seed(7)
        p.one_forward()
        state = torch.get_rng_state()
        detach()
        return state

    assert torch.equal(run(1), run(4))


def test_s4_resamples_do_not_pollute_the_captured_decode_path(policy):
    """The decoder hook fires during resamples too. Only the model's own
    denoise loop is the signal."""
    sink = CaptureSink()
    attach_groot_capture(policy, sink, k_resample=4)
    sink.begin_episode()
    policy.one_forward(seed=9)
    path = torch.as_tensor(sink.current()[0]["decode_path"])

    assert path.shape[0] == N_TIMESTEPS      # 4 Euler steps, not 4 x k


def test_writer_stores_resample_spread_sliced_like_the_other_action_tensors(policy, tmp_path):
    sink = CaptureSink()
    attach_groot_capture(policy, sink, k_resample=4)
    sink.begin_episode()
    policy.one_forward(seed=30)
    sink.flush(tmp_path, rollout_id="r1", live_dims=LIVE_DIMS, meta={})

    with numpy.load(tmp_path / "r1.npz") as z:
        assert z["resample_spread"].shape == (1, ACTION_HORIZON, LIVE_DIMS)


# --- fp16 overflow on the pre-norm tap --------------------------------------

def test_prenorm_tap_survives_values_that_would_overflow_fp16(policy, tmp_path):
    """vl_encoder is PRE-LayerNorm, so its activations are large by
    construction -- that is what vlln exists to fix. Measured on the real
    checkpoint, vl_encoder_max peaked at 15,296 against an fp16 ceiling of
    65,504: 4.3x headroom. One brighter-scene excursion over 723 LIBERO-Plus
    episodes turns it into inf, and an inf in a max-pooled feature puts that
    episode infinitely far from the reference cloud -- a fabricated OOD hit in
    the signal the whole experiment is about."""
    sink = CaptureSink()
    sink.begin_episode()
    big = torch.full((1, 4, 8), 70000.0)          # over the fp16 ceiling
    sink.stage(Role.VL_ENCODER, big)
    sink.stage(Role.VL_ADAPTED, torch.randn(1, 4, 8))
    sink.commit_forward()
    path = sink.flush(tmp_path, rollout_id="big", live_dims=LIVE_DIMS, meta={})

    with numpy.load(path) as z:
        assert numpy.isfinite(z["vl_encoder_max"]).all(), "overflowed to inf"
        assert z["vl_encoder_max"].max() > 65504, "value was silently clipped"


def test_flush_refuses_to_write_non_finite_features(tmp_path):
    """A NaN or inf reaching the reference cloud poisons every distance computed
    against it. Better to fail the episode than to write it."""
    sink = CaptureSink()
    sink.begin_episode()
    bad = torch.randn(1, 4, 8)
    bad[0, 0, 0] = float("nan")
    sink.stage(Role.VL_ADAPTED, bad)
    sink.commit_forward()

    with pytest.raises(ValueError, match="non-finite"):
        sink.flush(tmp_path, rollout_id="nan", live_dims=LIVE_DIMS, meta={})


# --- modality-aware pooling -------------------------------------------------

def test_vl_roles_are_pooled_separately_over_image_and_text_tokens(policy, tmp_path):
    """The VL sequence is image AND text tokens interleaved
    (groot_n1_7.py:451), and the model drives them through separate attention
    masks. Pooling the whole sequence averages two channels the model keeps
    apart -- and because instruction length varies, the mixture ratio drifts
    between episodes for reasons unrelated to what the camera saw."""
    sink = CaptureSink()
    attach_groot_capture(policy, sink)
    sink.begin_episode()
    policy.one_forward(seed=40)
    path = sink.flush(tmp_path, rollout_id="m", live_dims=LIVE_DIMS, meta={})

    with numpy.load(path) as z:
        for k in ("vl_adapted_img_mean", "vl_adapted_txt_mean",
                  "vl_encoder_img_mean", "vl_encoder_txt_mean"):
            assert k in z, f"missing {k}"
        img = z["vl_adapted_img_mean"].astype(numpy.float64)
        txt = z["vl_adapted_txt_mean"].astype(numpy.float64)
        assert not numpy.allclose(img, txt), "image and text pools are identical"


def test_combined_pool_is_the_token_weighted_mix_of_the_two(policy, tmp_path):
    """Sanity: the old combined mean must be recoverable from the two new ones,
    which proves the split is a partition of the same tokens and not a
    different quantity."""
    sink = CaptureSink()
    attach_groot_capture(policy, sink)
    sink.begin_episode()
    policy.one_forward(seed=41, n_text=3)
    path = sink.flush(tmp_path, rollout_id="m", live_dims=LIVE_DIMS, meta={})

    with numpy.load(path) as z:
        n_img, n_txt = int(z["n_image_tokens"][0]), int(z["n_text_tokens"][0])
        mix = (z["vl_adapted_img_mean"].astype(numpy.float64) * n_img
               + z["vl_adapted_txt_mean"].astype(numpy.float64) * n_txt) / (n_img + n_txt)
        numpy.testing.assert_allclose(mix, z["vl_adapted_mean"].astype(numpy.float64),
                                      rtol=2e-2, atol=2e-2)


def test_token_counts_are_recorded_so_mixture_drift_is_visible(policy, tmp_path):
    """R-036 could not check this because the capture stored no token count --
    the modality-mixing defect had to be established from source rather than
    from the data. It is recorded now."""
    sink = CaptureSink()
    attach_groot_capture(policy, sink)
    sink.begin_episode()
    policy.one_forward(seed=42, n_text=2)
    policy.one_forward(seed=43, n_text=5)
    path = sink.flush(tmp_path, rollout_id="m", live_dims=LIVE_DIMS, meta={})

    with numpy.load(path) as z:
        assert list(z["n_text_tokens"]) == [2, 5]
        assert list(z["n_image_tokens"]) == [5, 2]
