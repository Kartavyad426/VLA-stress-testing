"""E3 (EXPLORATORY, not pre-registered): commanded vs achieved end-effector motion.

ROOT_SIGNAL_EXPERIMENTS.html E3 (signal c), scope from prim's 10-09 message.
Commanded translation per step c_t = action[t, :3] * 0.05 m (OSC_POSE: input
+-1 -> +-0.05 m; kp 150, so it's compliant and never fully achieved).
Achieved d_t = eef_pos[t+1] - eef_pos[t] (steps[t].obs_state is the
observation BEFORE action t; the last step has no action).

Controller model, fitted on SUCCESSES only: a per-axis distributed-lag (FIR)
kernel d_t[k] = sum_{l=0..L} g_l[k] * c_{t-l}[k], L = 8. A single-lag fit
(checked 10-09: best lag 1, R^2 0.95, gain ~0.24, world-aligned axes with
cross terms <= 0.03) is confounded because commands are autocorrelated, so the
FIR is what defines the baseline. It is fitted on successes from half the
scenes and evaluated on the other half (and vice versa) so the baseline isn't
scored on its own fit. Residual e_t = d_t - FIR(c)_t; |e_t| in metres/step.

Questions (prim, 10-09):
  1 does the residual spike at the symptom anchor (first failing grasp onset;
    the first onset in successes)? Time course of |e| at offsets -16..+40
    around the anchor, and the anchor-window vs elsewhere ratio, per episode;
  2 does it tell miss from slip? Failing onsets split as in poc_toy_instance:
    slip if the target's max lift within 40 steps > 0.015 m (DEFAULT_LIFT_M),
    else miss; AUROC of residual features before/after the onset;
  3 does it separate failures from successes? Per-episode mean |e| on the
    matched window (steps < 80) and per forward among episodes still running
    at f+1, pooled and within (axis, magnitude).
Rotation (action[3:6] x 0.5 rad) is not used: eef_quat's convention differs
from MuJoCo's site frame (poc_toy_common docstring).

Built-in controls (checks.json): known answer (720/544, row 14 anchor t=31,
the FIR recovered from a synthetic trajectory with a known kernel), null
(synthetic achieved = FIR(commanded) gives zero residual), planted (eef frozen
for steps [50, 60) in held-out successes: the residual must spike there).

  python experiments/root_signals_e3_cmd_vs_achieved.py -> runs/root_signals/e3/
Outputs are "not val-checked (exempt 10-09)".
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

OUT = os.path.join(ROOT, "runs/root_signals/e3")
SCALE = 0.05
L = 8
K = 16
MATCHED_T = 80
LIFT_M, LIFT_WIN, SLIP_M = 0.05, 40, 0.015
PRE, POST = 16, 40
RNG = np.random.default_rng(0)
TAG = "EXPLORATORY, not val-checked (exempt 10-09)"


def auroc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    r = rankdata(np.r_[pos, neg]); n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def strat_auroc(items):
    num = den = 0.0
    for g in sorted({s for s, _, _ in items}):
        pos = [v for s, y, v in items if s == g and y]; neg = [v for s, y, v in items if s == g and not y]
        if pos and neg:
            w = len(pos) * len(neg); num += auroc(pos, neg) * w; den += w
    return num / den if den else float("nan")


def lagged(c, L=L):
    """(T, 3) -> (T, 3, L+1): column l holds c_{t-l} (zero before t=0)."""
    T = len(c); X = np.zeros((T, 3, L + 1))
    for l in range(L + 1):
        X[l:, :, l] = c[:T - l]
    return X


def fit_fir(eps):
    g = np.zeros((3, L + 1))
    for k in range(3):
        X = np.concatenate([lagged(e["c"])[:, k, :] for e in eps])
        y = np.concatenate([e["d"][:, k] for e in eps])
        g[k] = np.linalg.lstsq(X, y, rcond=None)[0]
    return g


def predict(c, g):
    return np.einsum("tkl,kl->tk", lagged(c), g)


def r2(eps, g):
    y = np.concatenate([e["d"] for e in eps]); p = np.concatenate([predict(e["c"], g) for e in eps])
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def target_of(st):
    non = [o for o in st["_gt_object_pos"] if "plate" not in o]
    assert len(non) == 1
    return non[0]


def load():
    rows, meta = join_r047()
    eps = []
    for r, m in zip(rows, meta):
        S = r["steps"]
        A = np.array([s["action"] for s in S if s.get("action")], dtype=np.float64)
        P = np.array([s["obs_state"]["eef_pos"] for s in S], dtype=np.float64)
        assert len(P) == len(A) + 1, f"line {m['line']}: {len(P)} obs vs {len(A)} actions"
        tgt = target_of(S[0]["obs_state"])
        z = np.array([(s["obs_state"].get("_gt_scene_object_pos") or s["obs_state"]["_gt_object_pos"])[tgt][2] for s in S])
        cmd = A[:, 6] > 0.5
        ons = []
        for t in range(1, len(cmd)):
            if cmd[t] and not cmd[t - 1]:
                lift = float(z[t:t + LIFT_WIN + 1].max() - z[0])
                ons.append(dict(t=t, lift=lift, ok=lift >= LIFT_M))
        if m["success"]:
            anc = ons[0] if ons else None
        else:
            anc = next((o for o in ons if not o["ok"]), None)
        eps.append(dict(line=m["line"], scene=m["scene"], axis=m["axis"], mag=m["magnitude"],
                        success=m["success"], forwards=m["forwards"], T=len(A),
                        c=np.clip(A[:, :3], -1, 1) * SCALE, d=P[1:] - P[:-1], onsets=ons, anchor=anc))
    return eps


def residuals(eps, g_by_scene):
    for e in eps:
        g = g_by_scene[e["scene"]]
        e["e"] = np.linalg.norm(e["d"] - predict(e["c"], g), axis=1)
        e["ez"] = (e["d"] - predict(e["c"], g))[:, 2]


def window_mean(x, lo, hi):
    lo, hi = max(lo, 0), min(hi, len(x))
    return float(x[lo:hi].mean()) if hi > lo else float("nan")


def q1(eps):
    out = {}
    for grp, cond in (("failing_first_onset", lambda e: not e["success"]),
                      ("successful_first_onset", lambda e: e["success"])):
        sub = [e for e in eps if cond(e) and e["anchor"] is not None]
        prof = np.full((len(sub), PRE + POST), np.nan)
        ratio = []
        for i, e in enumerate(sub):
            t = e["anchor"]["t"]
            for j, o in enumerate(range(-PRE, POST)):
                if 0 <= t + o < e["T"]:
                    prof[i, j] = e["e"][t + o]
            a = window_mean(e["e"], t - 4, t + 12)
            rest = np.r_[e["e"][:max(t - 20, 0)], e["e"][t + 28:]]
            if len(rest):
                ratio.append(a / max(rest.mean(), 1e-12))
        med = np.nanmedian(prof, 0)
        out[grp] = {"n": len(sub), "median_profile_mm_offsets_-16_to_+39": [round(1e3 * x, 3) for x in med],
                    "peak_offset": int(np.nanargmax(med)) - PRE,
                    "anchor_window_over_rest_median": round(float(np.median(ratio)), 3),
                    "share_episodes_ratio_gt_1": round(float(np.mean(np.array(ratio) > 1)), 3)}
    f = [e for e in eps if not e["success"] and e["anchor"] is not None]
    s = [e for e in eps if e["success"] and e["anchor"] is not None]
    fa = [window_mean(e["e"], e["anchor"]["t"] - 4, e["anchor"]["t"] + 12) for e in f]
    sa = [window_mean(e["e"], e["anchor"]["t"] - 4, e["anchor"]["t"] + 12) for e in s]
    out["failing_vs_successful_onset_window_auroc"] = {
        "auroc_fail_high": round(auroc(fa, sa), 3), "n_fail": len(fa), "n_succ": len(sa),
        "within_magnitude": round(strat_auroc([(f"{e['axis']}|{e['mag']}", 1, v) for e, v in zip(f, fa)] +
                                              [(f"{e['axis']}|{e['mag']}", 0, v) for e, v in zip(s, sa)]), 3)}
    return out


def q2(eps):
    ev = [e for e in eps if not e["success"] and e["anchor"] is not None]
    slip = [e for e in ev if e["anchor"]["lift"] > SLIP_M]; miss = [e for e in ev if e["anchor"]["lift"] <= SLIP_M]
    feats = {
        "pre_onset_mean_e[-8,0)": lambda e: window_mean(e["e"], e["anchor"]["t"] - 8, e["anchor"]["t"]),
        "post_onset_mean_e[0,40)": lambda e: window_mean(e["e"], e["anchor"]["t"], e["anchor"]["t"] + 40),
        "pre_onset_mean_ez[-8,0)": lambda e: window_mean(e["ez"], e["anchor"]["t"] - 8, e["anchor"]["t"]),
        "post_onset_mean_ez[8,40)": lambda e: window_mean(e["ez"], e["anchor"]["t"] + 8, e["anchor"]["t"] + 40),
    }
    return {"n_slip": len(slip), "n_miss": len(miss), "slip_rule": f"max lift within {LIFT_WIN} steps > {SLIP_M} m",
            "auroc_slip_high": {k: round(auroc([f(e) for e in slip], [f(e) for e in miss]), 3) for k, f in feats.items()},
            "medians_mm": {k: [round(1e3 * float(np.nanmedian([f(e) for e in slip])), 3),
                               round(1e3 * float(np.nanmedian([f(e) for e in miss])), 3)] for k, f in feats.items()}}


def q3(eps):
    out = {}
    for ax in list(AXES) + ["pooled"]:
        sub = [e for e in eps if ax == "pooled" or e["axis"] == ax]
        items = [(f"{e['axis']}|{e['mag']}", not e["success"], window_mean(e["e"], 0, MATCHED_T)) for e in sub]
        per_f = {}
        for f in range(10):
            it = [(f"{e['axis']}|{e['mag']}", not e["success"], window_mean(e["e"], K * f, K * f + K))
                  for e in sub if e["forwards"] >= f + 2]
            pos = [v for _, y, v in it if y]; neg = [v for _, y, v in it if not y]
            per_f[f] = {"auroc_fail_high": round(auroc(pos, neg), 3), "within_magnitude": round(strat_auroc(it), 3),
                        "n_fail": len(pos), "n_succ": len(neg)}
        pos = [v for _, y, v in items if y]; neg = [v for _, y, v in items if not y]
        out[ax] = {"matched_window_mean_e": {"auroc_fail_high": round(auroc(pos, neg), 3),
                                             "within_magnitude": round(strat_auroc(items), 3),
                                             "median_fail_mm": round(1e3 * float(np.median(pos)), 3),
                                             "median_succ_mm": round(1e3 * float(np.median(neg)), 3)},
                   "per_forward_mean_e": per_f}
    return out


def checks(eps, g_by_scene, gall):
    ck = {"known_answer": {"rows": len(eps), "successes": sum(e["success"] for e in eps),
                           "expect": [720, 544], "row14_anchor_t": eps[14]["anchor"]["t"], "expect_row14": 31}}
    # FIR recovery: synthetic d from known kernel on real commands
    true = gall.copy()
    syn = [dict(c=e["c"], d=predict(e["c"], true)) for e in eps[:100]]
    rec = fit_fir(syn)
    ck["known_answer"]["fir_recovery_max_abs_err"] = float(np.abs(rec - true).max())
    # null: residual of exact synthetic is zero
    ck["null_synthetic_residual_max_m"] = float(max(np.abs(s["d"] - predict(s["c"], rec)).max() for s in syn))
    # planted: freeze eef for [50, 60) in held-out successes
    succ = [e for e in eps if e["success"] and e["T"] >= 80][:200]
    hits, peaks = [], []
    for e in succ:
        P = np.r_[[np.zeros(3)], np.cumsum(e["d"], 0)]
        P[51:61] = P[50]          # eef doesn't move over steps 50..59
        d = P[1:] - P[:-1]
        r = np.linalg.norm(d - predict(e["c"], g_by_scene[e["scene"]]), axis=1)
        inwin = r[50:60].mean(); rest = np.r_[r[:45], r[65:]].mean()
        hits.append(inwin > rest); peaks.append(int(np.argmax(r)))
    ck["planted_freeze_50_60"] = {"n": len(succ), "share_window_mean_above_rest": round(float(np.mean(hits)), 3),
                                  "share_argmax_in_[50,61)": round(float(np.mean([50 <= p <= 60 for p in peaks])), 3),
                                  "pass": float(np.mean(hits)) > 0.95}
    ck["invariants"] = {"obs_equals_actions_plus_1": True, "no_nan_residual": bool(all(np.isfinite(e["e"]).all() for e in eps))}
    return ck


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def code_state(out):
    d = os.path.join(out, "code_state"); os.makedirs(d, exist_ok=True)
    g = lambda *a: subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout
    open(os.path.join(d, "HEAD"), "w").write(g("rev-parse", "HEAD"))
    open(os.path.join(d, "status.txt"), "w").write(g("status", "--porcelain"))
    open(os.path.join(d, "uncommitted.patch"), "w").write(g("diff", "HEAD"))
    for f in ("root_signals_e3_cmd_vs_achieved.py", "root_signals_common.py"):
        shutil.copy(os.path.join(ROOT, "experiments", f), d)


def main():
    os.makedirs(OUT, exist_ok=True)
    code_state(OUT)
    eps = load()
    scenes = sorted({e["scene"] for e in eps})
    half = [set(scenes[0::2]), set(scenes[1::2])]
    g_half = [fit_fir([e for e in eps if e["success"] and e["scene"] in h]) for h in half]
    # each scene's residuals use the kernel fitted on the OTHER half's successes
    g_by_scene = {s: g_half[1] if s in half[0] else g_half[0] for s in scenes}
    gall = fit_fir([e for e in eps if e["success"]])
    residuals(eps, g_by_scene)
    held = {f"half{i}_fit_eval_on_other_half_successes_R2": round(r2([e for e in eps if e["success"] and e["scene"] not in h], g_half[i]), 4)
            for i, h in enumerate(half)}
    held["all_successes_in_sample_R2"] = round(r2([e for e in eps if e["success"]], gall), 4)
    held["failures_R2_with_all_success_kernel"] = round(r2([e for e in eps if not e["success"]], gall), 4)
    res = {"tag": TAG, "plan": "ROOT_SIGNAL_EXPERIMENTS E3 + prim 10-09 (3 questions)",
           "inputs": {"rollouts.jsonl_sha256": sha(os.path.join(ROOT, "runs/r047/rollouts.jsonl"))},
           "controller_model": {"fir_lags": L, "kernel_all_successes": np.round(gall, 4).tolist(),
                                "dc_gain_per_axis": np.round(gall.sum(1), 3).tolist(),
                                "peak_lag_per_axis": [int(np.argmax(np.abs(k))) for k in gall], **held},
           "success_residual_mm": {"median": round(1e3 * float(np.median(np.concatenate([e["e"] for e in eps if e["success"]]))), 3),
                                   "p95": round(1e3 * float(np.percentile(np.concatenate([e["e"] for e in eps if e["success"]]), 95)), 3)},
           "q1_anchor": q1(eps), "q2_miss_vs_slip": q2(eps), "q3_fail_vs_success": q3(eps)}
    ck = checks(eps, g_by_scene, gall)
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    json.dump(ck, open(os.path.join(OUT, "checks.json"), "w"), indent=1, default=float)
    with open(os.path.join(OUT, "per_rollout.jsonl"), "w") as f:
        for e in eps:
            f.write(json.dumps({k: e[k] for k in ("line", "scene", "axis", "mag", "success", "forwards", "T", "anchor")}
                               | {"e_mm": np.round(1e3 * e["e"], 3).tolist()}) + "\n")
    print(json.dumps(ck, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float)[:7000])


if __name__ == "__main__":
    main()
