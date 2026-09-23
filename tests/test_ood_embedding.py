"""§4A: swap the feature space, keep the method.

Mirrors experiments/ood_selfref.py -- reference = the policy's own successes,
threshold = leave-one-out p95, report separation and Mann-Whitney p -- but over
captured embeddings instead of 5 proprioceptive dims. The number to beat is
S0's 6.9x.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from vla_harness.capture.analysis import (calibrate_threshold, load_capture,
                                          nn_distance, score_run)


def _write(tmp_path, rid, feats, success, task="libero_spatial/task0", **extra):
    np.savez_compressed(tmp_path / f"{rid}.npz", vl_adapted_mean=feats.astype(np.float16))
    row = {"rollout_id": rid, "path": f"{rid}.npz", "n_forwards": len(feats),
           "success": success, "task_id": task, "env_step": list(range(len(feats))),
           **extra}
    with open(tmp_path / "capture_manifest.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")


def test_load_capture_joins_features_to_labels_on_rollout_id(tmp_path):
    rng = np.random.default_rng(0)
    _write(tmp_path, "a", rng.normal(size=(6, 8)), True)
    _write(tmp_path, "b", rng.normal(size=(5, 8)), False)

    eps = load_capture(tmp_path, "vl_adapted_mean")
    assert {e.rollout_id for e in eps} == {"a", "b"}
    assert {e.success for e in eps} == {True, False}
    assert next(e for e in eps if e.rollout_id == "a").features.shape == (6, 8)


def test_load_capture_refuses_an_npz_with_no_manifest_row(tmp_path):
    """An unjoined .npz means the episode's label is unknown. Scoring it as
    though it were a success would corrupt the reference cloud silently."""
    rng = np.random.default_rng(0)
    _write(tmp_path, "a", rng.normal(size=(4, 8)), True)
    np.savez_compressed(tmp_path / "orphan.npz",
                        vl_adapted_mean=rng.normal(size=(4, 8)).astype(np.float16))

    with pytest.raises(ValueError, match="orphan"):
        load_capture(tmp_path, "vl_adapted_mean")


def test_threshold_calibration_hits_the_requested_false_positive_rate(tmp_path):
    """alpha is a DESIGN PARAMETER, not an outcome: the reference must score
    ~alpha against itself, leave-one-out. If it does not, the threshold is
    wrong and every downstream ratio is meaningless."""
    rng = np.random.default_rng(1)
    ref = [rng.normal(size=(40, 6)) for _ in range(25)]

    thr, rate = calibrate_threshold(ref, alpha=0.05)
    assert 0.02 < rate < 0.09, f"reference scored {rate:.3f} OOD, wanted ~0.05"


def test_separation_detects_a_genuinely_shifted_query(tmp_path):
    rng = np.random.default_rng(2)
    ref = [rng.normal(size=(30, 6)) for _ in range(20)]
    thr, _ = calibrate_threshold(ref, alpha=0.05)
    cloud = np.concatenate(ref)

    near = np.mean((nn_distance(rng.normal(size=(30, 6)), cloud) > thr))
    far = np.mean((nn_distance(rng.normal(size=(30, 6)) + 6.0, cloud) > thr))
    assert far > near * 3


def test_score_run_reports_separation_and_p_value(tmp_path):
    rng = np.random.default_rng(3)
    for i in range(12):
        _write(tmp_path, f"s{i}", rng.normal(size=(20, 6)), True)
    for i in range(12):
        _write(tmp_path, f"f{i}", rng.normal(size=(20, 6)) + 5.0, False)

    res = score_run(tmp_path, "vl_adapted_mean", alpha=0.05)
    assert res.n_reference == 12 and res.n_fail == 12
    assert res.separation > 2.0
    assert res.p_value < 0.05


def test_success_episodes_are_scored_leave_one_out_not_against_themselves(tmp_path):
    """Reference and query are the SAME run here -- forced by the unseeded
    randn at groot_n1_7.py:657, which means a re-run does not reproduce its own
    success/failure split.

    ood_selfref.py never hit this because its --ref and --query are different
    runs. Scoring a success episode against a cloud that contains its own points
    gives nearest-neighbour distance 0, so success rate collapses to 0%, and the
    separation ratio becomes a division by epsilon -- 4.9e7x on the real stage-0
    capture, which is not a result, it is an artefact.

    The tell is that the reference rate and the success rate must AGREE: both
    describe the same episodes.
    """
    rng = np.random.default_rng(5)
    for i in range(14):
        _write(tmp_path, f"s{i}", rng.normal(size=(20, 6)), True)
    for i in range(10):
        _write(tmp_path, f"f{i}", rng.normal(size=(20, 6)) + 4.0, False)

    res = score_run(tmp_path, "vl_adapted_mean", alpha=0.05)
    assert res.mean_ood_success > 0.0, "successes scored against themselves"
    assert abs(res.mean_ood_success - res.reference_rate) < 0.03, (
        f"success rate {res.mean_ood_success:.3f} disagrees with reference rate "
        f"{res.reference_rate:.3f}; they describe the same episodes")
    assert res.separation < 100, "ratio is a division by epsilon, not a result"


# --- Q1: detection (does the perturbation move the signal at all?) ----------

def test_detection_rate_is_high_when_the_arm_is_genuinely_shifted(tmp_path):
    """Q1 asks whether a perturbation moves a signal AT ALL, outcome aside.
    It gates Q2: if a lighting change does not move the VL embedding, a
    pass/fail null in lighting episodes says the instrument is deaf, not that
    the model is."""
    from vla_harness.capture.analysis import detection_rate

    rng = np.random.default_rng(11)
    nominal = [rng.normal(size=(8, 6)) for _ in range(20)]
    shifted = [rng.normal(size=(8, 6)) + 5.0 for _ in range(20)]
    same = [rng.normal(size=(8, 6)) for _ in range(20)]

    thr, ref_rate = calibrate_threshold(nominal, alpha=0.05)
    assert detection_rate(shifted, nominal, thr) > 0.5
    # an unshifted arm must land near alpha, not at zero -- it is drawn from
    # the same distribution but is NOT in the reference cloud
    assert detection_rate(same, nominal, thr) < 0.25


def test_detection_excludes_an_episode_from_its_own_reference(tmp_path):
    """Scoring the nominal arm against itself must be leave-one-out, or it
    reads 0% and every ratio against it is a division by epsilon (O8)."""
    from vla_harness.capture.analysis import detection_rate

    rng = np.random.default_rng(12)
    nominal = [rng.normal(size=(8, 6)) for _ in range(20)]
    thr, ref_rate = calibrate_threshold(nominal, alpha=0.05)

    self_rate = detection_rate(nominal, nominal, thr, leave_one_out=True)
    assert self_rate > 0.0
    assert abs(self_rate - ref_rate) < 0.03
