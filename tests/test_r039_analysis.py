"""R-039 aggregation, on synthetic per-instance arrays shaped like the runner's."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from vla_harness.analysis.r039 import (aggregate, dominant_arm, onset_forward,
                                       load_instances, RESIDUAL_THRESHOLD)


def _write(tmp_path, rows):
    d = tmp_path / "splice"
    d.mkdir()
    with open(d / "manifest.jsonl", "w") as f:
        for r in rows:
            arrs = r.pop("_arrays")
            np.savez(d / f"{r['rollout_id']}.npz", **arrs)
            f.write(json.dumps(r) + "\n")
    return str(tmp_path)


def _row(rid, cat, tf_T, tf_I, tf_S, success=False, level=4, d_pn0=1.0):
    F = len(tf_T)
    return {"rollout_id": rid, "task_id": int(rid), "category": cat, "level": level,
            "label": f"{cat} {rid}", "success": success, "forwards": F,
            "anchor_forward": F - 1, "d_pn_f0": d_pn0,
            "_arrays": {"tf_T": np.array(tf_T, np.float32), "tf_I": np.array(tf_I, np.float32),
                        "tf_S": np.array(tf_S, np.float32),
                        "env_steps": np.arange(F) * 16, "d_pn": np.full(F, d_pn0, np.float32),
                        "closest_approach": np.linspace(0.3, 0.1, F * 16, dtype=np.float32)}}


def test_onset_is_the_first_forward_crossing_half():
    assert onset_forward(np.array([0.1, 0.4, 0.6, 0.9])) == 2
    assert onset_forward(np.array([0.7, 0.2])) == 0
    assert onset_forward(np.array([0.1, 0.2])) is None


def test_dominant_arm_is_the_largest_at_the_anchor_or_none_below_threshold():
    assert dominant_arm({"T": 0.1, "I": 0.95, "S": 0.2}) == "I"
    assert dominant_arm({"T": 0.1, "I": 0.5, "S": 0.4}) is None
    assert RESIDUAL_THRESHOLD == 0.9


def test_aggregate_reports_per_category_medians_and_the_residual_bucket(tmp_path):
    root = _write(tmp_path, [
        _row("1", "Camera Viewpoints", [0.1, 0.1], [0.95, 0.96], [0.0, 0.1]),
        _row("2", "Camera Viewpoints", [0.2, 0.1], [0.5, 0.6], [0.0, 0.4]),
        _row("3", "Robot Initial States", [0.0, 0.0], [0.0, 0.0], [0.9, 0.95]),
    ])
    inst = load_instances(root)
    assert len(inst) == 3
    agg = aggregate(inst)
    cam = agg["by_category"]["Camera Viewpoints"]
    assert cam["n"] == 2
    assert cam["median_tf_f0"]["I"] == pytest.approx(0.725)
    assert cam["median_tf_anchor"]["I"] == pytest.approx(0.78)
    assert cam["dominant"] == {"I": 1, "none": 1}
    ris = agg["by_category"]["Robot Initial States"]
    assert ris["median_tf_f0"]["S"] == pytest.approx(0.9)
    assert agg["residual"]["n"] == 1 and agg["residual"]["ids"] == ["2"]


def test_aggregate_flags_the_state_arm_growing_with_forward_index(tmp_path):
    root = _write(tmp_path, [
        _row("1", "Light Conditions", [0.0] * 4, [0.9] * 4, [0.0, 0.2, 0.5, 0.8]),
        _row("2", "Light Conditions", [0.0] * 4, [0.9] * 4, [0.1, 0.1, 0.1, 0.1]),
    ])
    agg = aggregate(load_instances(root))
    slopes = agg["by_category"]["Light Conditions"]["S_slope_per_forward"]
    assert slopes["median"] > 0 and slopes["n_positive"] == 1
