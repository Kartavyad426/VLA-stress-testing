"""R-053 scoring: the start-pose no-op baseline. Writes runs/r053/score.json.

  .venvs/groot/bin/python experiments/r053_score.py [--runs-root runs]

Per noise seed and pooled over seeds 0-2, on the ten R-044 instances:
  * P failures (the no-op re-run, runs/r053_P_s<k>);
  * N flips: P-failures of the SAME seed that succeed under N (runs/r053_N_s<k>);
  * re-run flips: failures of the ORIGINAL P run at that noise seed that succeed
    in the re-run (seed 0 -> R-042, seeds 1/2 -> R-044's seed runs), and the
    reverse (original successes that fail on re-run);
  * median closest approach per drive, and N-beats-P on closest approach;
  * ||P-N|| at forward 0 against R-044's at the same seed (tolerance 0.05).
Checks the four pre-registered expectations and the uninterpretability rule.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

SEEDS = (0, 1, 2)
# the original P run per noise seed, and R-044's forward-0 ||P-N|| per seed
ORIGINAL_P = {0: "r042", 1: "r044_seed1", 2: "r044_seed2"}
R044_DPN = {0: "r044_rescue_N", 1: "r044_seed1", 2: "r044_seed2"}
DPN_TOL = 0.05


def load(root, run_id, ids=None) -> dict[int, dict]:
    p = os.path.join(root, run_id, "splice", "manifest.jsonl")
    if not os.path.exists(p):
        return {}
    rows = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            if ids is None or r["task_id"] in ids:
                rows[r["task_id"]] = r         # last row wins (a resumed rerun)
    return rows


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def queue_rcs(path) -> list[str]:
    if not os.path.exists(path):
        return []
    return [l.strip() for l in open(path) if re.search(r" rc=(?!0\b)\d+", l)]


def score(root: str) -> dict:
    sel = json.load(open("experiments/repro/r044_selection_ris.json"))
    ids = [i["task_id"] for i in sel["instances"]]
    per_seed, pooled = {}, {"n": 0, "P_fail": 0, "N_success": 0, "N_flips": 0,
                            "orig_fail": 0, "rerun_flips": 0, "rerun_breaks": 0,
                            "ca_N_beats_P": 0, "ca_pairs": 0, "dpn_match": 0, "dpn_n": 0}
    ca_all = {"P": [], "N": []}
    missing = []
    for s in SEEDS:
        P, N = load(root, f"r053_P_s{s}", ids), load(root, f"r053_N_s{s}", ids)
        orig, ref = load(root, ORIGINAL_P[s], ids), load(root, R044_DPN[s], ids)
        both = [t for t in ids if t in P and t in N]
        missing += [f"s{s}:{t}" for t in ids if t not in both]
        p_fail = [t for t in both if not P[t]["success"]]
        n_flips = [t for t in p_fail if N[t]["success"]]
        o_ids = [t for t in both if t in orig]
        o_fail = [t for t in o_ids if not orig[t]["success"]]
        rerun_flips = [t for t in o_fail if P[t]["success"]]
        rerun_breaks = [t for t in o_ids if orig[t]["success"] and not P[t]["success"]]
        ca_pairs = [t for t in both if P[t].get("closest_approach_m") is not None
                    and N[t].get("closest_approach_m") is not None]
        ca_beats = [t for t in ca_pairs if N[t]["closest_approach_m"] < P[t]["closest_approach_m"]]
        dpn = {}
        for t in both:
            if t in ref and ref[t].get("d_pn_f0") is not None:
                # P and N runs both record every arm at forward 0; either gives ||P-N||
                dpn[t] = {"r053_P": P[t]["d_pn_f0"], "r053_N": N[t]["d_pn_f0"], "r044": ref[t]["d_pn_f0"],
                          "ok": abs(P[t]["d_pn_f0"] - ref[t]["d_pn_f0"]) <= DPN_TOL
                                and abs(N[t]["d_pn_f0"] - ref[t]["d_pn_f0"]) <= DPN_TOL}
        ca_all["P"] += [P[t].get("closest_approach_m") for t in both]
        ca_all["N"] += [N[t].get("closest_approach_m") for t in both]
        per_seed[s] = {
            "n": len(both), "P_fail": len(p_fail), "P_fail_ids": p_fail,
            "N_success": sum(N[t]["success"] for t in both),
            "N_flips": len(n_flips), "N_flip_ids": n_flips,
            "N_breaks": [t for t in both if P[t]["success"] and not N[t]["success"]],
            "original_run": ORIGINAL_P[s], "orig_fail": len(o_fail),
            "rerun_flips": len(rerun_flips), "rerun_flip_ids": rerun_flips,
            "rerun_breaks": len(rerun_breaks), "rerun_break_ids": rerun_breaks,
            "median_ca_P": med([P[t].get("closest_approach_m") for t in both]),
            "median_ca_N": med([N[t].get("closest_approach_m") for t in both]),
            "ca_N_beats_P": len(ca_beats), "ca_pairs": len(ca_pairs),
            "dpn_f0": dpn, "dpn_match": sum(v["ok"] for v in dpn.values()), "dpn_n": len(dpn),
        }
        for k in ("n", "P_fail", "N_success", "N_flips", "orig_fail", "rerun_flips",
                  "rerun_breaks", "ca_N_beats_P", "ca_pairs", "dpn_match", "dpn_n"):
            pooled[k] += per_seed[s][k]
    pooled["median_ca_P"], pooled["median_ca_N"] = med(ca_all["P"]), med(ca_all["N"])
    rate = lambda a, b: (a / b) if b else None
    pooled["N_flip_rate"] = rate(pooled["N_flips"], pooled["P_fail"])
    pooled["rerun_flip_rate"] = rate(pooled["rerun_flips"], pooled["orig_fail"])

    s0 = per_seed.get(0, {})
    exp = {
        "E1_N_success_ge_21_of_30": pooled["N_success"] >= 21 if pooled["n"] == 30 else None,
        "E2_N_flip_rate_minus_rerun_ge_30pp": (
            (pooled["N_flip_rate"] - pooled["rerun_flip_rate"]) >= 0.30
            if None not in (pooled["N_flip_rate"], pooled["rerun_flip_rate"]) else None),
        "E3_rerun_seed0_flips_le_3_of_R042_failures": (s0["rerun_flips"] <= 3) if s0.get("n") else None,
        "E4_ca_N_beats_P_ge_20_of_30": pooled["ca_N_beats_P"] >= 20 if pooled["n"] == 30 else None,
    }
    bad_rc = queue_rcs(os.path.join(root, "r053", "queue.log"))
    uninterp = {
        "dpn_seed0_match_ge_9_of_10": (s0.get("dpn_match", 0) >= 9) if s0.get("n") else None,
        "nonzero_rc": bad_rc,
        "missing": missing,
    }
    uninterp["interpretable"] = bool(uninterp["dpn_seed0_match_ge_9_of_10"]) and not bad_rc and not missing
    return {"per_seed": per_seed, "pooled": pooled, "expectations": exp, "uninterpretable_checks": uninterp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = score(a.runs_root)
    out = a.out or os.path.join(a.runs_root, "r053", "score.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1, default=float)
    for s, v in res["per_seed"].items():
        print(f"seed {s}: n={v['n']} P-fail {v['P_fail']} | N success {v['N_success']} flips {v['N_flips']}/{v['P_fail']}"
              f" | rerun vs {v['original_run']}: flips {v['rerun_flips']}/{v['orig_fail']} breaks {v['rerun_breaks']}"
              f" | median CA P {v['median_ca_P']} N {v['median_ca_N']} | ||P-N|| f0 match {v['dpn_match']}/{v['dpn_n']}")
    p = res["pooled"]
    print(f"pooled: n={p['n']} N success {p['N_success']} N flip rate {p['N_flip_rate']} rerun flip rate {p['rerun_flip_rate']}"
          f" CA N<P {p['ca_N_beats_P']}/{p['ca_pairs']}")
    print("expectations:", json.dumps(res["expectations"]))
    print("interpretable:", res["uninterpretable_checks"]["interpretable"])
    print("wrote", out)


if __name__ == "__main__":
    main()
