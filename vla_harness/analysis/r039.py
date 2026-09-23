"""R-039 aggregation: per-instance arrays -> the signal x category table.

Reads runs/<run>/splice/manifest.jsonl and the per-instance .npz the runner
wrote. Nothing here touches the model or the simulator.
"""
from __future__ import annotations

import json
import os

import numpy as np

ARMS = ("T", "I", "S")
ONSET_THRESHOLD = 0.5       # RESULTS.md R-039 revision: first forward with transfer >= 0.5
RESIDUAL_THRESHOLD = 0.9    # RESULTS.md R-039 expectation 6: no single arm >= 0.9


def onset_forward(tf: np.ndarray, threshold: float = ONSET_THRESHOLD):
    idx = np.flatnonzero(np.asarray(tf) >= threshold)
    return int(idx[0]) if len(idx) else None


def dominant_arm(tf_at: dict, threshold: float = RESIDUAL_THRESHOLD):
    arm = max(tf_at, key=lambda k: tf_at[k])
    return arm if tf_at[arm] >= threshold else None


def load_instances(run_root: str) -> list[dict]:
    d = os.path.join(run_root, "splice")
    out = []
    for line in open(os.path.join(d, "manifest.jsonl")):
        row = json.loads(line)
        z = np.load(os.path.join(d, f"{row['rollout_id']}.npz"))
        row["tf"] = {a: z[f"tf_{a}"].astype(np.float64) for a in ARMS}
        row["env_steps"] = z["env_steps"]
        row["d_pn"] = z["d_pn"].astype(np.float64)
        row["closest_approach"] = z["closest_approach"].astype(np.float64)
        out.append(row)
    return out


def _at(inst: dict, fwd: int) -> dict:
    return {a: float(inst["tf"][a][fwd]) for a in ARMS}


def _slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    x = np.arange(len(y), dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def aggregate(instances: list[dict]) -> dict:
    per = []
    for inst in instances:
        F = len(inst["tf"]["T"])
        anchor = inst.get("anchor_forward")
        if anchor is None or anchor < 0 or anchor >= F:
            anchor = F - 1
        at_f0, at_anchor = _at(inst, 0), _at(inst, anchor)
        per.append({
            "rollout_id": inst["rollout_id"], "task_id": inst["task_id"],
            "category": inst["category"], "level": inst.get("level"), "label": inst.get("label"),
            "success": inst.get("success"), "forwards": F, "anchor_forward": anchor,
            "d_pn_f0": float(inst["d_pn"][0]) if len(inst["d_pn"]) else None,
            "tf_f0": at_f0, "tf_anchor": at_anchor,
            "tf_mean": {a: float(np.nanmean(inst["tf"][a])) for a in ARMS},
            "onset": {a: onset_forward(inst["tf"][a]) for a in ARMS},
            "dominant": dominant_arm(at_anchor),
            "S_slope": _slope(inst["tf"]["S"]),
        })
    by_cat: dict = {}
    for cat in sorted({p["category"] for p in per}):
        rows = [p for p in per if p["category"] == cat]
        dom = {}
        for p in rows:
            dom[p["dominant"] or "none"] = dom.get(p["dominant"] or "none", 0) + 1
        slopes = np.array([p["S_slope"] for p in rows], dtype=np.float64)
        by_cat[cat] = {
            "n": len(rows),
            "n_fail": sum(1 for p in rows if p["success"] is False),
            "median_tf_f0": {a: float(np.median([p["tf_f0"][a] for p in rows])) for a in ARMS},
            "median_tf_anchor": {a: float(np.median([p["tf_anchor"][a] for p in rows])) for a in ARMS},
            "median_tf_mean": {a: float(np.median([p["tf_mean"][a] for p in rows])) for a in ARMS},
            "median_d_pn_f0": float(np.median([p["d_pn_f0"] for p in rows if p["d_pn_f0"] is not None])),
            "dominant": dict(sorted(dom.items())),
            "S_slope_per_forward": {"median": float(np.nanmedian(slopes)),
                                    "n_positive": int(np.sum(slopes > 0.05))},
        }
    residual = [p["rollout_id"] for p in per if p["dominant"] is None]
    return {"n": len(per), "by_category": by_cat,
            "residual": {"n": len(residual), "ids": residual,
                         "fraction": len(residual) / len(per) if per else float("nan")},
            "instances": per}
