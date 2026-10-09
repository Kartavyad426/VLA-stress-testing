"""R-059 part A (pre-registered 2026-09-29 16:40, RESULTS.md R-059): CPU, stored R-047.

  A1  failure type over env steps 0-79 (forwards 0-4):
        M1 reversal rate  = share of consecutive executed actions whose (dx,dy,dz)
                            have cosine < 0;
        M2 gripper toggles = changes of the executed gripper command;
        M3 progress       = drop in _gt_eef_to_object[target] from step 0 to 79,
                            target = the non-plate object in _gt_object_pos.
      Thresholds from the SUCCESSES of the same axis. Erratic = M1 or M2 above the
      successes' p95; stalled = not erratic and M3 below the successes' p5;
      smooth-progressing = the rest. Every failure classified, per axis.
  A2  clock baseline: FPR(f) = share of successes still running at forward f
      (= having a forward f, i.e. forwards > f); f_c = smallest f with FPR <= 0.05.
  A3  single-draw control (= E1): on act_P (F x 40 x 7), the overlap distance
      ||chunk_f[16:40] - chunk_{f+1}[0:24]|| over dims 0-5, f in 0..4; AUROC for
      failure vs success among episodes still running at f+1.

Expectations under test: (1) <= 40% of start-pose failures erratic; (2) A3 AUROC
at f <= 2 in [0.40, 0.60]; (3) f_c >= 9 on both axes.

Choices the pre-registration leaves open, fixed here before looking at results:
  * "per axis" = R-047's two axes, each INCLUDING its magnitude-0 rows (the join
    in root_signals_common assigns those rows their axis); a sensitivity
    without magnitude 0 is reported alongside.
  * Episodes shorter than 80 steps (fast successes) use their last step for M3
    and their available pairs for M1/M2; their count is reported.
  * Zero-norm translation pairs are excluded from M1's denominator (counted).
  * Gripper command = executed action[6] > 0 (env actions are +-1).
  * A3 is computed on act_P as stored (GR00T-normalised, as R-041's script did)
    and in env units (executed = s_d * act_P + o_d per dim, s_d fitted here and
    recorded); differences make o_d cancel.
  * A3 CIs: 95% bootstrap over scenes (clustered), 2000 resamples; extra, not
    pre-registered (the pre-registration asks scene-clustered CIs for part C).

  python experiments/r059_partA.py      -> runs/r059/partA/{summary.json, per_rollout.jsonl, code_state/}
Not a result until PIPELINE_CHECKS has a PASS; until then tag UNVERIFIED.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from root_signals_common import join_r047, AXES  # noqa: E402

OUT = os.path.join(ROOT, "runs/r059/partA")
WIN = 80                 # env steps 0-79
K, H, D = 16, 40, 6      # executed steps per forward, chunk length, dims kept
FS = range(5)            # overlap indices f = 0..4
RNG = np.random.default_rng(0)
TAG = "UNVERIFIED (no PIPELINE_CHECKS PASS yet)"


def target_of(st):
    objs = list(st["_gt_object_pos"])
    non = [o for o in objs if "plate" not in o]
    assert len(non) == 1, f"expected one non-plate object, got {objs}"
    return non[0]


def a1_metrics(row):
    S = row["steps"]
    A = np.array([s["action"] for s in S if s.get("action")], dtype=np.float64)[:WIN]
    tr = A[:, :3]
    n = np.linalg.norm(tr, axis=1)
    ok = (n[:-1] > 0) & (n[1:] > 0)
    cos = (tr[:-1] * tr[1:]).sum(1)[ok] / (n[:-1] * n[1:])[ok]
    m1 = float((cos < 0).mean()) if len(cos) else float("nan")
    cmd = A[:, 6] > 0
    m2 = int((cmd[1:] != cmd[:-1]).sum())
    tgt = target_of(S[0]["obs_state"])
    last = min(WIN - 1, len(S) - 1)
    m3 = float(S[0]["obs_state"]["_gt_eef_to_object"][tgt] - S[last]["obs_state"]["_gt_eef_to_object"][tgt])
    return dict(M1=m1, M2=m2, M3=m3, target=tgt, n_actions_in_window=len(A),
                zero_norm_pairs=int((~ok).sum()), short_window=len(S) < WIN)


def classify(rec, succ):
    p95_1 = np.nanpercentile([r["M1"] for r in succ], 95)
    p95_2 = np.percentile([r["M2"] for r in succ], 95)
    p5_3 = np.percentile([r["M3"] for r in succ], 5)
    erratic = rec["M1"] > p95_1 or rec["M2"] > p95_2
    if erratic:
        return "erratic"
    return "stalled" if rec["M3"] < p5_3 else "smooth_progressing"


def thresholds(succ):
    return {"M1_p95": float(np.nanpercentile([r["M1"] for r in succ], 95)),
            "M2_p95": float(np.percentile([r["M2"] for r in succ], 95)),
            "M3_p5": float(np.percentile([r["M3"] for r in succ], 5)), "n_successes": len(succ)}


def a1(recs, axis, include_mag0=True):
    sub = [r for r in recs if r["axis"] == axis and (include_mag0 or r["magnitude"] > 0)]
    succ = [r for r in sub if r["success"]]; fail = [r for r in sub if not r["success"]]
    cls = [classify(r, succ) for r in fail]
    counts = {c: cls.count(c) for c in ("erratic", "stalled", "smooth_progressing")}
    return {"n_failures": len(fail), "counts": counts,
            "erratic_share": round(counts["erratic"] / max(len(fail), 1), 3),
            "thresholds": thresholds(succ),
            "short_window_successes": sum(r["short_window"] for r in succ)}, cls


def a2(recs, axis, include_mag0=True):
    sub = [r for r in recs if r["axis"] == axis and (include_mag0 or r["magnitude"] > 0)]
    F = np.array([r["forwards"] for r in sub if r["success"]])
    fpr = {f: float((F > f).mean()) for f in range(0, 19)}
    fc = min(f for f, v in fpr.items() if v <= 0.05)
    fail_F = sorted({r["forwards"] for r in sub if not r["success"]})
    return {"f_c": fc, "fpr_by_forward": {k: round(v, 4) for k, v in fpr.items()},
            "n_successes": len(F), "success_forwards_median": float(np.median(F)),
            "failure_forwards_values": fail_F}


def overlap(P, f, scale=None):
    a = P[f, K:H, :D].astype(np.float64); b = P[f + 1, :H - K, :D].astype(np.float64)
    d = a - b
    if scale is not None:
        d = d * scale[:D]
    return float(np.linalg.norm(d))


def auroc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    r = rankdata(np.r_[pos, neg]); n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(items, n=2000):
    """items: (scene, y, score); resample scenes with replacement."""
    scenes = sorted({s for s, _, _ in items})
    by = {s: [(y, v) for ss, y, v in items if ss == s] for s in scenes}
    vals = []
    for _ in range(n):
        pick = RNG.choice(scenes, len(scenes))
        pts = [p for s in pick for p in by[s]]
        pos = [v for y, v in pts if y == 1]; neg = [v for y, v in pts if y == 0]
        if pos and neg:
            vals.append(auroc(pos, neg))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def fit_scale(rows, meta, n=60):
    """Executed action = s_d * act_P[f, :16, d] + o_d: least squares per dim on n rollouts."""
    X, Y = [], []
    for r, m in list(zip(rows, meta))[:n]:
        P = np.load(m["npz"])["act_P"].astype(np.float64)
        A = np.array([s["action"] for s in r["steps"] if s.get("action")], dtype=np.float64)
        for f in range(P.shape[0]):
            a = A[K * f:K * f + K]
            X.append(P[f, :len(a)]); Y.append(a)
    X, Y = np.concatenate(X), np.concatenate(Y)
    s, o, res = np.zeros(7), np.zeros(7), np.zeros(7)
    for d in range(7):
        c = np.polyfit(X[:, d], Y[:, d], 1); s[d], o[d] = c
        res[d] = np.abs(np.polyval(c, X[:, d]) - Y[:, d]).max()
    return s, o, res


def a3(recs, scale):
    out = {}
    for units, sc in (("normalised", None), ("env", scale)):
        out[units] = {}
        for ax in list(AXES) + ["pooled"]:
            sub = [r for r in recs if ax == "pooled" or r["axis"] == ax]
            row = {}
            for f in FS:
                pts = [(r["scene"], 0 if r["success"] else 1, r[f"ov_{units}"][f])
                       for r in sub if r["forwards"] >= f + 2]
                pos = [v for _, y, v in pts if y]; neg = [v for _, y, v in pts if not y]
                row[f] = {"auroc_fail_high": round(auroc(pos, neg), 3), "n_fail": len(pos), "n_succ": len(neg),
                          "ci95_scene_bootstrap": boot_ci(pts),
                          "median_fail": round(float(np.median(pos)), 4) if pos else None,
                          "median_succ": round(float(np.median(neg)), 4) if neg else None}
            out[units][ax] = row
    return out


def a3_strat(recs):
    """Exploratory: A3 AUROC within each (axis, magnitude) stratum, pooled with
    n1*n0 weights, because failures concentrate at large magnitudes."""
    out = {}
    for ax in AXES:
        out[ax] = {}
        for f in FS:
            num = den = 0.0; per = {}
            for mag in sorted({r["magnitude"] for r in recs if r["axis"] == ax}):
                sub = [r for r in recs if r["axis"] == ax and r["magnitude"] == mag and r["forwards"] >= f + 2]
                pos = [r["ov_normalised"][f] for r in sub if not r["success"]]
                neg = [r["ov_normalised"][f] for r in sub if r["success"]]
                if pos and neg:
                    a = auroc(pos, neg); w = len(pos) * len(neg)
                    num += a * w; den += w; per[mag] = {"auroc": round(a, 3), "n_fail": len(pos), "n_succ": len(neg)}
            out[ax][f] = {"pooled_within_stratum_auroc": round(num / den, 3) if den else None, "strata": per}
    return out


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def code_state(out):
    d = os.path.join(out, "code_state"); os.makedirs(d, exist_ok=True)
    g = lambda *a: subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout
    open(os.path.join(d, "HEAD"), "w").write(g("rev-parse", "HEAD"))
    open(os.path.join(d, "status.txt"), "w").write(g("status", "--porcelain"))
    open(os.path.join(d, "uncommitted.patch"), "w").write(g("diff", "HEAD"))
    for f in ("r059_partA.py", "root_signals_common.py"):
        shutil.copy(os.path.join(ROOT, "experiments", f), d)


def main():
    os.makedirs(OUT, exist_ok=True)
    code_state(OUT)
    rows, meta = join_r047()
    scale, offset, resid = fit_scale(rows, meta)
    recs = []
    for r, m in zip(rows, meta):
        P = np.load(m["npz"])["act_P"]
        assert P.shape[0] == m["forwards"], f"line {m['line']}: npz has {P.shape[0]} forwards, manifest {m['forwards']}"
        rec = dict(line=m["line"], scene=m["scene"], axis=m["axis"], magnitude=m["magnitude"], dir=m["dir"],
                   noise_seed=m["noise_seed"], success=m["success"], forwards=m["forwards"], **a1_metrics(r))
        rec["ov_normalised"] = {f: overlap(P, f) for f in FS if P.shape[0] >= f + 2}
        rec["ov_env"] = {f: overlap(P, f, scale) for f in FS if P.shape[0] >= f + 2}
        recs.append(rec)
    res = {"tag": TAG, "design": "RESULTS.md R-059 part A (registered 2026-09-29 16:40)",
           "inputs": {"rollouts.jsonl_sha256": sha(os.path.join(ROOT, "runs/r047/rollouts.jsonl")),
                      "n_rows": len(recs), "n_successes": sum(r["success"] for r in recs)},
           "act_P_to_env_map": {"scale": scale.round(5).tolist(), "offset": offset.round(5).tolist(),
                                "max_abs_residual": resid.round(5).tolist(),
                                "note": "gripper dim is binarised in env (residual large by design); dropped in A3"},
           "A1": {}, "A1_sensitivity_no_mag0": {}, "A2": {}, "A2_sensitivity_no_mag0": {}}
    for ax in AXES:
        res["A1"][ax], cls = a1(recs, ax)
        k = 0
        for r in recs:
            if r["axis"] == ax and not r["success"]:
                r["A1_class"] = cls[k]; k += 1
        res["A1_sensitivity_no_mag0"][ax] = a1(recs, ax, False)[0]
        res["A2"][ax] = a2(recs, ax)
        res["A2_sensitivity_no_mag0"][ax] = a2(recs, ax, False)
    res["A3"] = a3(recs, scale)
    res["A3_extra_within_magnitude_NOT_PREREGISTERED"] = a3_strat(recs)
    sp = res["A1"]["joint_radius_rad"]["erratic_share"]
    a3n = res["A3"]["normalised"]
    res["expectations"] = {
        "E1_start_pose_erratic_le_40pct": {"value": sp, "holds": sp <= 0.40},
        "E2_A3_auroc_f_le_2_in_0.40_0.60": {
            ax: {f: {"auroc": a3n[ax][f]["auroc_fail_high"],
                     "holds": 0.40 <= a3n[ax][f]["auroc_fail_high"] <= 0.60} for f in (0, 1, 2)}
            for ax in AXES},
        "E3_f_c_ge_9_both_axes": {ax: {"f_c": res["A2"][ax]["f_c"], "holds": res["A2"][ax]["f_c"] >= 9}
                                  for ax in AXES}}
    res["checks_for_val"] = {
        "known_answer": {"rows": len(recs), "successes": sum(r["success"] for r in recs),
                         "expect_rows_successes": [720, 544],
                         "f_c_expect_12_both_axes (D26 draft, measured 10-07)":
                             {ax: res["A2"][ax]["f_c"] for ax in AXES},
                         "every_failure_18_forwards": {ax: res["A2"][ax]["failure_forwards_values"] for ax in AXES}},
        "invariants": {"npz_forwards_equal_manifest": True,
                       "executed_equals_act_P_first16_after_affine_max_residual_dims0_5": float(resid[:6].max())},
        "null": "overlap of a chunk with itself is 0 by construction; see check script (val)"}
    with open(os.path.join(OUT, "per_rollout.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r, default=float) + "\n")
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("expectations", "checks_for_val", "act_P_to_env_map")}, indent=1, default=float))
    print(json.dumps(res["A1"], indent=1, default=float))


if __name__ == "__main__":
    main()
