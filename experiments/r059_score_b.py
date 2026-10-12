"""R-059 part B smoke scorer (CPU). Reads the multi-draw run dirs written by r047_run.py --draws B --seed-noise.

  .venvs/groot/bin/python experiments/r059_score_b.py --runs r059_b8,r059_b32,r059_b32_rep,r059_b1 --axis joint_radius_rad

Per run and rollout: forwards, wall per forward (head_s), peak GPU memory (manifest), and per forward f:
  STAC_f   unbiased MMD² (RBF) between the B draws of chunk f, steps 16:40, and the B draws of chunk f+1, steps 0:24;
           dims 0-5 (gripper dropped), flattened 24 x 6.
  NULL_f   the same statistic between two independent B-sets at the SAME observation (act_B vs act_B_null, steps
           16:40), needs --null-extra in the run.
  SPREAD_f mean pairwise L2 within the B draws at f (steps 16:40, dims 0-5).
  SINGLE_f ||draw0_f[16:40] - draw0_{f+1}[0:24]|| (dims 0-5), the A3 statistic.
Bandwidth: median heuristic over pooled draws of radius-0 rollouts, computed once here and written out (frozen for C).
Controls:
  bit identity  draw 0 (act_P) and all of act_B equal across two runs with the same seed (--rep pairs a,b);
  B-invariance  draw 0 of the B=1 seeded run vs draw 0 of the B=32 run at matched (scene, radius, dir, noise, f);
                reported, not required (GPU batch-size kernels; draw 0 is executed, so the trajectories may diverge).
Gate for C (pre-registered): median STAC_0 >= 3 x the noise floor at radius 0.4. The unbiased MMD² (U) null is
centred on 0 and can be negative, so a ratio to it is undefined; the ratio is reported on the biased V-statistic
(V_ratio_stac0_over_null), with U's null 95th percentile and the share of STAC_0 above it alongside. Which one
the gate uses: DECIDED by my prim 10-09, before data: the gate is median STAC_0 >= 3 x the null on the biased
V-statistic (same estimator for both), at radius 0.4; secondary: U-MMD² share of STAC_0 above the null's p95.
GATE (my prim, 10-09, before data): median V-STAC_0 / median V-null at the SAME f = 0 observations >= 3 at radius
0.4 (GATE_V_ratio_stac0_over_null0, GATE_pass_ge_3); the pooled-over-forwards null ratio is secondary.
Bandwidth: median heuristic on the radius-0 SUCCESSES of --bw-run, computed once, frozen in runs/r059/bandwidth.json
and reused for every later scoring (radius 0.4, part C). `--b0 <run>`: the default-flags run must reproduce R-047's
stored act_P bitwise; exit code 3 (stop, report) if not.
Writes runs/r059/score_b.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

DIMS = slice(0, 6)


def flat(x):            # (B, 24, 7) -> (B, 144)
    return np.asarray(x, np.float64)[:, :, DIMS].reshape(len(x), -1)


def mmd2(X, Y, sigma):
    """Unbiased MMD² with k = exp(-||x-y||² / (2 sigma²))."""
    def k(A, B):
        d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-d / (2 * sigma ** 2))
    m, n = len(X), len(Y)
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    return float((Kxx.sum() - np.trace(Kxx)) / (m * (m - 1)) + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1)) - 2 * Kxy.mean())


def mmd2_v(X, Y, sigma):
    """Biased (V-statistic) MMD², always >= 0: the form a 'k x the null' ratio is defined for."""
    def k(A, B):
        return np.exp(-((A[:, None, :] - B[None, :, :]) ** 2).sum(-1) / (2 * sigma ** 2))
    return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())


def load(run, axis):
    d = os.path.join("runs", run, axis)
    rows = [json.loads(l) for l in open(os.path.join(d, "manifest.jsonl"))]
    out = []
    for r in rows:
        tag = f"{r['task_id']}_{r['magnitude']:.6f}_d{r['dir']}_n{r['noise_seed']}".replace(".", "p")
        z = np.load(os.path.join(d, f"{tag}.npz"))
        out.append((r, {k: z[k] for k in z.files}))
    return out


def key(r):
    return (r["task_id"], r["magnitude"], r["dir"], r["noise_seed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None, help="required unless --b0")
    ap.add_argument("--axis", default="joint_radius_rad")
    ap.add_argument("--rep", default=None, help="a,b: two same-seed runs for the bit-identity check")
    ap.add_argument("--b1", default=None, help="run id of the B=1 seeded run for the B-invariance check")
    ap.add_argument("--bw-run", default=None, help="run whose radius-0 SUCCESSES set the bandwidth (once; frozen to runs/r059/bandwidth.json)")
    ap.add_argument("--b0", default=None, help="run id of the default-flags run: act_P must equal R-047's stored npz bitwise; exits 3 if not")
    a = ap.parse_args()
    if a.b0:
        r0 = load(a.b0, a.axis)
        res = {}
        for r, z in r0:
            tag = f"{r['task_id']}_{r['magnitude']:.6f}_d{r['dir']}_n{r['noise_seed']}".replace(".", "p")
            ref = np.load(os.path.join("runs/r047", a.axis, f"{tag}.npz"))["act_P"]
            nf = min(len(z["act_P"]), len(ref))
            res[tag] = {"bitwise": bool(np.array_equal(z["act_P"][0], ref[0])),          # gate: forward 0 only
                        "later_forwards_max_abs": [float(np.abs(z["act_P"][f].astype(float) - ref[f].astype(float)).max()) for f in range(1, nf)],
                        "shapes": [list(z["act_P"].shape), list(ref.shape)],
                        "note": "pass = act_P[0] bitwise (my prim 10-09); later forwards are reported, not gated (closed-loop render jitter)"}
        os.makedirs("runs/r059", exist_ok=True)
        json.dump(res, open("runs/r059/b0_check.json", "w"), indent=1)
        print("B0", json.dumps(res))
        if not all(v["bitwise"] for v in res.values()):
            raise SystemExit(3)
        return
    if not a.runs:
        ap.error("--runs is required unless --b0 is given")
    runs = {r: load(r, a.axis) for r in a.runs.split(",")}
    bw_p = "runs/r059/bandwidth.json"
    if os.path.exists(bw_p):
        bw = json.load(open(bw_p))                       # frozen: never recomputed
    else:
        if not a.bw_run:
            raise SystemExit("no frozen bandwidth yet: pass --bw-run <run> (its radius-0 successes set it)")
        pool = [flat(z["act_B"][f, :, 16:40]) for r, z in runs[a.bw_run] if r["magnitude"] == 0 and r["success"]
                for f in range(len(z["act_B"]))]
        if not pool:
            raise SystemExit("no radius-0 successes in --bw-run to set the bandwidth")
        P = np.concatenate(pool)
        rng = np.random.default_rng(0)
        S = P[rng.choice(len(P), min(len(P), 2000), replace=False)]
        bw = {"sigma": float(np.median(np.sqrt(((S[:, None] - S[None]) ** 2).sum(-1))[np.triu_indices(len(S), 1)])),
              "from_run": a.bw_run, "pool": "radius-0 successes, act_B steps 16:40, dims 0-5", "pool_n": int(len(P))}
        os.makedirs("runs/r059", exist_ok=True)
        json.dump(bw, open(bw_p, "w"), indent=1)
    sigma = bw["sigma"]
    out = {"bandwidth": bw, "runs": {}}
    for run, rs in runs.items():
        R = []
        for r, z in rs:
            B = z["act_B"].shape[1]
            F = len(z["act_B"])
            e = {"key": key(r), "success": r["success"], "forwards": F, "B": B, "head_s_mean": r.get("head_s_mean"),
                 "peak_gpu_alloc_gib": r.get("peak_gpu_alloc_gib"), "wall_s": r.get("wall_s"),
                 "single": [float(np.linalg.norm(z["act_P"][f, 16:40, DIMS].astype(float) - z["act_P"][f + 1, 0:24, DIMS].astype(float))) for f in range(F - 1)]}
            if B > 1:
                e["stac_v"] = [mmd2_v(flat(z["act_B"][f, :, 16:40]), flat(z["act_B"][f + 1, :, 0:24]), sigma) for f in range(F - 1)]
                e["stac"] = [mmd2(flat(z["act_B"][f, :, 16:40]), flat(z["act_B"][f + 1, :, 0:24]), sigma) for f in range(F - 1)]
                e["spread"] = [float(np.mean([np.linalg.norm(x - y) for i, x in enumerate(flat(z["act_B"][f, :, 16:40]))
                                              for y in flat(z["act_B"][f, :, 16:40])[i + 1:]])) for f in range(F)]
                if "act_B_null" in z:
                    e["null_v"] = [mmd2_v(flat(z["act_B"][f, :, 16:40]), flat(z["act_B_null"][f, :, 16:40]), sigma) for f in range(F)]
                    e["null"] = [mmd2(flat(z["act_B"][f, :, 16:40]), flat(z["act_B_null"][f, :, 16:40]), sigma) for f in range(F)]
            R.append(e)
        s = {"n_rollouts": len(R), "rollouts": R,
             "head_ms_per_forward_median": float(np.median([e["head_s_mean"] for e in R])) * 1e3 if R and R[0]["head_s_mean"] else None,
             "peak_gpu_alloc_gib_max": max((e["peak_gpu_alloc_gib"] or 0) for e in R) if R else None}
        for mag in sorted({e["key"][1] for e in R}):
            sub = [e for e in R if e["key"][1] == mag]
            st0 = [e["stac"][0] for e in sub if e.get("stac")]
            nu0 = [e["null"][0] for e in sub if e.get("null")]
            sv0 = [e["stac_v"][0] for e in sub if e.get("stac_v")]
            nv = [x for e in sub for x in e.get("null_v", [])]
            nu_all = [x for e in sub for x in e.get("null", [])]
            s[f"radius_{mag}"] = {"n": len(sub), "median_stac0": float(np.median(st0)) if st0 else None,
                                  "median_null0": float(np.median(nu0)) if nu0 else None,
                                  "ratio_stac0_over_null0": float(np.median(st0) / np.median(nu0)) if st0 and nu0 and np.median(nu0) > 0 else None,
                                  "V_median_stac0": float(np.median(sv0)) if sv0 else None,
                                  "V_median_null0": float(np.median([e["null_v"][0] for e in sub if e.get("null_v")])) if any(e.get("null_v") for e in sub) else None,
                                  "GATE_V_ratio_stac0_over_null0": (float(np.median(sv0) / np.median([e["null_v"][0] for e in sub if e.get("null_v")]))
                                                                    if sv0 and any(e.get("null_v") for e in sub) else None),
                                  "GATE_pass_ge_3": None,
                                  "V_median_null_all_f": float(np.median(nv)) if nv else None,
                                  "secondary_V_ratio_stac0_over_pooled_null": float(np.median(sv0) / np.median(nv)) if sv0 and nv and np.median(nv) > 0 else None,
                                  "U_null_p95_all_f": float(np.percentile(nu_all, 95)) if nu_all else None,
                                  "secondary_U_frac_stac0_above_null_p95": float(np.mean(np.array(st0) > np.percentile(nu_all, 95))) if st0 and nu_all else None,
                                  "median_null_all_f": float(np.median([x for e in sub for x in e.get("null", [])])) if any(e.get("null") for e in sub) else None,
                                  "median_stac_all_f": float(np.median([x for e in sub for x in e.get("stac", [])])) if any(e.get("stac") for e in sub) else None}
        for k_, v_ in s.items():
            if k_.startswith("radius_") and v_.get("GATE_V_ratio_stac0_over_null0") is not None:
                v_["GATE_pass_ge_3"] = bool(v_["GATE_V_ratio_stac0_over_null0"] >= 3)
        out["runs"][run] = s
    if a.rep:
        ra, rb = a.rep.split(",")
        A = {key(r): z for r, z in runs[ra]}; Bm = {key(r): z for r, z in runs[rb]}
        out["bit_identity"] = {str(k): {"act_P_equal": bool(np.array_equal(A[k]["act_P"], Bm[k]["act_P"])),
                                        "act_B_equal": bool(np.array_equal(A[k]["act_B"], Bm[k]["act_B"])),
                                        "forwards": [len(A[k]["act_P"]), len(Bm[k]["act_P"])]} for k in A if k in Bm}
    if a.b1:
        one = {key(r): z for r, z in runs[a.b1]}
        inv = {}
        for run, rs in runs.items():
            if run == a.b1:
                continue
            for r, z in rs:
                k = key(r)
                if k in one:
                    f0 = 0
                    inv[f"{run}|{k}"] = {"draw0_f0_equal": bool(np.array_equal(z["act_P"][f0], one[k]["act_P"][f0])),
                                         "draw0_f0_max_abs": float(np.abs(z["act_P"][f0].astype(float) - one[k]["act_P"][f0].astype(float)).max()),
                                         "forwards_equal_prefix": int(next((f for f in range(min(len(z["act_P"]), len(one[k]["act_P"])))
                                                                            if not np.array_equal(z["act_P"][f], one[k]["act_P"][f])),
                                                                           min(len(z["act_P"]), len(one[k]["act_P"]))))}
        out["b_invariance_draw0"] = inv
    os.makedirs("runs/r059", exist_ok=True)
    json.dump(out, open("runs/r059/score_b.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1, default=str))
    for run, s in out["runs"].items():
        print(run, {k: v for k, v in s.items() if k != "rollouts"})


if __name__ == "__main__":
    main()
