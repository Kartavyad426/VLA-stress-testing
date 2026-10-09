"""E2 (EXPLORATORY, not pre-registered): d_pn per forward as a root signal.

ROOT_SIGNAL_EXPERIMENTS.html E2 (signal b, the privileged paired-nominal
reference for E1), scope from prim's 10-09 message:
  (a) AUROC failure vs success per forward f, among episodes still running at
      f+1 (every R-047 failure is an 18-forward timeout, so this is the matched
      comparison), per axis, pooled and within (axis, magnitude) strata;
      plus the max over the first k forwards;
  (b) does a d_pn jump line up with the rule-based symptom anchor (first
      failing grasp onset, poc_toy_common.grasp_events' definition)? Lead/lag
      in forwards. Sanity case: row 14, root t = 31 (forward 1).

d_pn as stored (r047_run.py:187) is ||act_P[f] - act_N[f]|| over ALL 7 normalised
dims, gripper included (a gripper-timing disagreement alone adds up to 2 per
step). So every analysis runs on d_pn as stored ("all7") and on the arm-only
distance recomputed from act_P/act_N over dims 0-5 ("arm6").

What N is (res21's audit, confirmed 10-08): start pose: the recorded
magnitude-0 rollout of the same (scene, dir, noise) aligned by FORWARD INDEX,
not a paired render, so later forwards compare different states. Camera yaw:
a paired re-render of the current state under the nominal camera
(nominal_frames(), force_update). Upstream caveats (G0c follow-ups, res21
pm1): in-loop force_update re-renders shift the observable sampling phase of
later frames.

Jump definition (fixed before looking at results): the first forward f at
which d_pn[f] exceeds gamma_f = the 95th percentile of d_pn[f] among the SAME
AXIS's successes still running at f (calibrated per magnitude stratum when it
has >= 10 such successes, else per axis), the Sentinel-style per-forward band.
Lead = anchor_forward - jump_forward (> 0: d_pn fires first), with
anchor_forward = anchor_t // 16 (forward k executes env steps 16k..16k+15).
A second, threshold-free jump: argmax over f of d_pn[f] - d_pn[f-1].

  python experiments/root_signals_e2_dpn.py -> runs/root_signals/e2/{summary.json, per_rollout.jsonl, code_state/}
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

OUT = os.path.join(ROOT, "runs/root_signals/e2")
K = 16
LIFT_M, LIFT_WIN = 0.05, 40          # poc_toy_common.grasp_events criterion
FMAX = 17                            # forwards 0..17 (failures run 18)
MIN_CAL = 10
RNG = np.random.default_rng(0)
TAG = "EXPLORATORY, UNVERIFIED (no PIPELINE_CHECKS PASS yet)"


def auroc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    r = rankdata(np.r_[pos, neg]); n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def target_of(st):
    non = [o for o in st["_gt_object_pos"] if "plate" not in o]
    assert len(non) == 1
    return non[0]


def anchor(row):
    """First grasp onset whose target doesn't rise LIFT_M within LIFT_WIN steps
    (failures); first onset (successes). Same definition as poc_toy_common."""
    S = row["steps"]
    tgt = target_of(S[0]["obs_state"])
    pos = lambda s: (s["obs_state"].get("_gt_scene_object_pos") or s["obs_state"]["_gt_object_pos"])[tgt]
    z = np.array([pos(s)[2] for s in S])
    cmd = [bool(s.get("action")) and s["action"][6] > 0.5 for s in S]
    ons = []
    for t in range(1, len(cmd)):
        if cmd[t] and not cmd[t - 1]:
            ons.append((t, float(z[t:t + LIFT_WIN + 1].max() - z[0]) >= LIFT_M))
    if not ons:
        return None, ons
    if row["success"]:
        return ons[0][0], ons
    bad = [t for t, ok in ons if not ok]
    return (bad[0] if bad else None), ons


def strat_auroc(items):
    """items: (stratum, y, score) -> n1*n0-weighted within-stratum AUROC."""
    num = den = 0.0
    for g in sorted({s for s, _, _ in items}):
        pos = [v for s, y, v in items if s == g and y]; neg = [v for s, y, v in items if s == g and not y]
        if pos and neg:
            w = len(pos) * len(neg); num += auroc(pos, neg) * w; den += w
    return num / den if den else float("nan")


def boot_ci(items, key, n=2000):
    scenes = sorted({it["scene"] for it in items})
    by = {s: [it for it in items if it["scene"] == s] for s in scenes}
    vals = []
    for _ in range(n):
        pts = [it for s in RNG.choice(scenes, len(scenes)) for it in by[s]]
        pos = [it[key] for it in pts if it["y"]]; neg = [it[key] for it in pts if not it["y"]]
        if pos and neg:
            vals.append(auroc(pos, neg))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def section_a(recs, sig):
    out = {}
    for ax in list(AXES) + ["pooled"]:
        sub = [r for r in recs if ax == "pooled" or r["axis"] == ax]
        per_f = {}
        for f in range(FMAX):
            items = [dict(scene=r["scene"], y=not r["success"], v=r[sig][f], strat=f"{r['axis']}|{r['magnitude']}")
                     for r in sub if r["forwards"] >= f + 2]
            pos = [i["v"] for i in items if i["y"]]; neg = [i["v"] for i in items if not i["y"]]
            if len(neg) < 5:
                break
            per_f[f] = {"auroc_fail_high": round(auroc(pos, neg), 3), "n_fail": len(pos), "n_succ": len(neg),
                        "within_magnitude_auroc": round(strat_auroc([(i["strat"], i["y"], i["v"]) for i in items]), 3),
                        "ci95_scene_bootstrap": boot_ci(items, "v")}
        maxk = {}
        for k in (1, 2, 3, 5):
            items = [dict(scene=r["scene"], y=not r["success"], v=max(r[sig][:k]), strat=f"{r['axis']}|{r['magnitude']}")
                     for r in sub if r["forwards"] >= k + 1]
            pos = [i["v"] for i in items if i["y"]]; neg = [i["v"] for i in items if not i["y"]]
            maxk[k] = {"auroc_fail_high": round(auroc(pos, neg), 3), "n_fail": len(pos), "n_succ": len(neg),
                       "within_magnitude_auroc": round(strat_auroc([(i["strat"], i["y"], i["v"]) for i in items]), 3)}
        out[ax] = {"per_forward": per_f, "max_over_first_k": maxk}
    return out


def gammas(recs, sig):
    """gamma[(axis, magnitude-or-None, f)] = p95 of d_pn[f] over that cell's successes running at f."""
    g = {}
    for ax in AXES:
        for f in range(FMAX + 1):
            base = [r for r in recs if r["axis"] == ax and r["success"] and r["forwards"] > f]
            if len(base) >= MIN_CAL:
                g[(ax, None, f)] = float(np.percentile([r[sig][f] for r in base], 95))
            for mag in {r["magnitude"] for r in recs if r["axis"] == ax}:
                b = [r for r in base if r["magnitude"] == mag]
                if len(b) >= MIN_CAL:
                    g[(ax, mag, f)] = float(np.percentile([r[sig][f] for r in b], 95))
    return g


def jump(r, sig, g):
    for f in range(r["forwards"]):
        thr = g.get((r["axis"], r["magnitude"], f), g.get((r["axis"], None, f)))
        if thr is not None and r[sig][f] > thr:
            return f
    return None


def section_b(recs, sig):
    g = gammas(recs, sig)
    out = {}
    for grp, cond in (("failures", lambda r: not r["success"]), ("successes", lambda r: r["success"])):
        leads, leads_dx, fired, nanchor = [], [], 0, 0
        for ax in AXES:
            pass
        for r in recs:
            if not cond(r):
                continue
            j = jump(r, sig, g)
            fired += j is not None
            dx = int(np.argmax(np.diff(r[sig]))) + 1 if r["forwards"] >= 2 else None
            if r["anchor_t"] is None:
                continue
            nanchor += 1
            af = r["anchor_t"] // K
            if j is not None:
                leads.append(af - j)
            if dx is not None:
                leads_dx.append(af - dx)
        n = sum(cond(r) for r in recs)
        summ = lambda L: ({"n": len(L), "median": float(np.median(L)), "iqr": [float(np.percentile(L, 25)), float(np.percentile(L, 75))],
                           "share_within_1_forward": round(float(np.mean(np.abs(L) <= 1)), 3),
                           "share_jump_before_anchor": round(float(np.mean(np.array(L) > 0)), 3)} if L else {"n": 0})
        out[grp] = {"n_episodes": n, "n_with_anchor": nanchor, "share_fired_any_forward": round(fired / max(n, 1), 3),
                    "lead_threshold_jump": summ(leads), "lead_max_increase": summ(leads_dx)}
    # per axis, failures only
    for ax in AXES:
        L = [r["anchor_t"] // K - jump(r, sig, g) for r in recs if r["axis"] == ax and not r["success"]
             and r["anchor_t"] is not None and jump(r, sig, g) is not None]
        out[f"failures_{ax}"] = {"n": len(L), "median_lead": float(np.median(L)) if L else None,
                                 "share_within_1_forward": round(float(np.mean(np.abs(L) <= 1)), 3) if L else None}
    return out, g


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def code_state(out):
    d = os.path.join(out, "code_state"); os.makedirs(d, exist_ok=True)
    g = lambda *a: subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout
    open(os.path.join(d, "HEAD"), "w").write(g("rev-parse", "HEAD"))
    open(os.path.join(d, "status.txt"), "w").write(g("status", "--porcelain"))
    open(os.path.join(d, "uncommitted.patch"), "w").write(g("diff", "HEAD"))
    for f in ("root_signals_e2_dpn.py", "root_signals_common.py"):
        shutil.copy(os.path.join(ROOT, "experiments", f), d)


def main():
    os.makedirs(OUT, exist_ok=True)
    code_state(OUT)
    rows, meta = join_r047()
    recs, recompute_err = [], 0.0
    for r, m in zip(rows, meta):
        z = np.load(m["npz"])
        P, N = z["act_P"].astype(np.float32), z["act_N"].astype(np.float32)
        all7 = np.linalg.norm((P - N).reshape(len(P), -1), axis=1)
        arm6 = np.linalg.norm((P[..., :6] - N[..., :6]).reshape(len(P), -1), axis=1)
        recompute_err = max(recompute_err, float(np.abs(all7 - z["d_pn"]).max()))
        at, ons = anchor(r)
        recs.append(dict(line=m["line"], scene=m["scene"], axis=m["axis"], magnitude=m["magnitude"], dir=m["dir"],
                         noise_seed=m["noise_seed"], success=m["success"], forwards=m["forwards"],
                         all7=z["d_pn"].astype(float).tolist(), arm6=arm6.tolist(), anchor_t=at,
                         onsets=[t for t, _ in ons], n_failing_onsets=sum(not ok for _, ok in ons)))
    res = {"tag": TAG, "plan": "ROOT_SIGNAL_EXPERIMENTS E2 + prim 10-09 (a) AUROC per forward, (b) lead/lag vs symptom anchor",
           "inputs": {"rollouts.jsonl_sha256": sha(os.path.join(ROOT, "runs/r047/rollouts.jsonl")), "n_rows": len(recs)},
           "invariants": {"d_pn_recomputed_all7_vs_stored_max_abs": recompute_err,
                          "npz_forwards_equal_manifest": all(len(r["all7"]) == r["forwards"] for r in recs),
                          "failures_without_anchor": sum(1 for r in recs if not r["success"] and r["anchor_t"] is None),
                          "successes_without_onset": sum(1 for r in recs if r["success"] and r["anchor_t"] is None)},
           "a": {}, "b": {}}
    for sig in ("all7", "arm6"):
        res["a"][sig] = section_a(recs, sig)
        res["b"][sig], g = section_b(recs, sig)
    r14 = recs[14]
    g7 = gammas(recs, "all7"); g6 = gammas(recs, "arm6")
    res["sanity_row14"] = {"scene": r14["scene"], "magnitude": r14["magnitude"], "dir": r14["dir"],
                           "noise_seed": r14["noise_seed"], "anchor_t": r14["anchor_t"],
                           "anchor_forward": None if r14["anchor_t"] is None else r14["anchor_t"] // K,
                           "expect_anchor_t": 31, "onsets": r14["onsets"],
                           "all7_first10": [round(x, 3) for x in r14["all7"][:10]],
                           "arm6_first10": [round(x, 3) for x in r14["arm6"][:10]],
                           "jump_all7": jump(r14, "all7", g7), "jump_arm6": jump(r14, "arm6", g6)}
    with open(os.path.join(OUT, "per_rollout.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("invariants", "sanity_row14", "b")}, indent=1, default=float))
    for sig in ("all7", "arm6"):
        for ax, v in res["a"][sig].items():
            print(sig, ax, {f: (x["auroc_fail_high"], x["within_magnitude_auroc"], x["n_fail"], x["n_succ"])
                            for f, x in list(v["per_forward"].items())[:6]}, "maxk", v["max_over_first_k"])


if __name__ == "__main__":
    main()
