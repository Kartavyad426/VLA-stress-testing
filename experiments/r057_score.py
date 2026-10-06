"""R-057 scoring: when is the episode decided (drive-until sweep). Writes runs/r057/score.json.

  .venvs/groot/bin/python experiments/r057_score.py [--runs-root runs]
  .venvs/groot/bin/python experiments/r057_score.py --check-reuse     # exit 0 iff R-053 is reusable

Per (row, k) with k in {P, k1, k2, k4, all}, pooled over noise seeds 0-2:
  * success /30 with a Wilson 95% interval;
  * flips (and breaks) relative to P at the same instance and seed;
  * median closest approach;
  * the post-hand-back gap: per instance, mean ||P-N|| over forwards k..k+2 of
    the driven run divided by the same forwards of the P run; median over
    instances x seeds (R-050's measure). Not defined for P or "all".
  * k* = the smallest k in {1, 2, 4} with success within 10 pp of "all" and
    >= 20 pp above P; else "throughout (> 4 forwards)".
Start-pose P and k1 come from r057_ris_{P,k1}_s* when present, else from
R-053's r053_{P,N}_s* (runs/r057/reuse_r053 records which).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from vla_harness.runner import wilson_ci  # noqa: E402

SEEDS = (0, 1, 2)
LABELS = ("P", "k1", "k2", "k4", "all")
K_OF = {"k1": 1, "k2": 2, "k4": 4}
# forward-0 references at seed 0 (uninterpretability rule)
F0_REF = {"ris": "r044_rescue_N", "cam": "r048_rescue_N"}
SEL = {"ris": "experiments/repro/r044_selection_ris.json", "cam": "experiments/repro/r048_selection_cam.json"}
CODE_PATHS = ("vla_harness/", "experiments/r039_run.py")
R053_FOR = {"P": "r053_P_s{s}", "k1": "r053_N_s{s}"}


def reused(root) -> bool:
    p = os.path.join(root, "r057", "reuse_r053")
    return os.path.exists(p) and open(p).read().strip() == "1"


def run_dir(root, row, lab, s):
    """R-053's runs stand in only when the queue decided to reuse them, never as a silent fallback."""
    if row == "ris" and lab in R053_FOR and reused(root):
        return R053_FOR[lab].format(s=s)
    return f"r057_{row}_{lab}_s{s}"


def load(root, run_id):
    p = os.path.join(root, run_id, "splice", "manifest.jsonl")
    rows = {}
    if os.path.exists(p):
        for line in open(p):
            if line.strip():
                r = json.loads(line); rows[r["task_id"]] = r
    return rows


def dpn_series(root, run_id, row) -> np.ndarray | None:
    p = os.path.join(root, run_id, "splice", f"{row['rollout_id']}.npz")
    if not os.path.exists(p):
        return None
    return np.load(p)["d_pn"].astype(np.float64)


# --- reuse check: R-053's code_state against this checkout -----------------
def _code_sections(patch: str) -> dict[str, str]:
    """The per-file sections of a `git diff` that touch rollout-shaping code."""
    out, cur, buf = {}, None, []
    for line in patch.splitlines(keepends=True):
        m = re.match(r"diff --git a/(\S+) b/", line)
        if m:
            if cur is not None:
                out[cur] = "".join(buf)
            cur, buf = m.group(1), []
        buf.append(line)
    if cur is not None:
        out[cur] = "".join(buf)
    return {f: t for f, t in out.items() if f.startswith(CODE_PATHS)}


def check_reuse(root) -> tuple[bool, list[str]]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    now = _code_sections(subprocess.run(["git", "diff", "--", *CODE_PATHS], capture_output=True, text=True).stdout)
    why = []
    for s in SEEDS:
        for lab in ("P", "k1"):
            rid = R053_FOR[lab].format(s=s)
            cs = os.path.join(root, rid, "code_state")
            if not os.path.exists(os.path.join(root, rid, "splice", "manifest.jsonl")):
                why.append(f"{rid}: no manifest"); continue
            if len(load(root, rid)) < 10:
                why.append(f"{rid}: incomplete"); continue
            h = open(os.path.join(cs, "HEAD")).read().strip() if os.path.exists(os.path.join(cs, "HEAD")) else None
            if h != head:
                # a different commit is still a match if the code paths did not change between them
                d = subprocess.run(["git", "diff", "--quiet", str(h), head, "--", *CODE_PATHS]) if h else None
                if d is None or d.returncode != 0:
                    why.append(f"{rid}: HEAD {h} differs on {CODE_PATHS}"); continue
            then = _code_sections(open(os.path.join(cs, "uncommitted.patch")).read()) \
                if os.path.exists(os.path.join(cs, "uncommitted.patch")) else {}
            if then != now:
                why.append(f"{rid}: uncommitted code differs ({sorted(set(then) ^ set(now)) or 'content'})")
    return not why, why


# --- scoring ----------------------------------------------------------------
def score(root) -> dict:
    out = {"reuse_r053": open(os.path.join(root, "r057", "reuse_r053")).read().strip()
           if os.path.exists(os.path.join(root, "r057", "reuse_r053")) else None, "rows": {}}
    for row in ("ris", "cam"):
        ids = [i["task_id"] for i in json.load(open(SEL[row]))["instances"]]
        runs = {(lab, s): run_dir(root, row, lab, s) for lab in LABELS for s in SEEDS}
        data = {k: load(root, v) for k, v in runs.items()}
        res = {}
        for lab in LABELS:
            succ, n, flips, breaks, cas, gaps = 0, 0, 0, 0, [], []
            for s in SEEDS:
                d, p = data[(lab, s)], data[("P", s)]
                for t in ids:
                    if t not in d:
                        continue
                    n += 1; succ += bool(d[t]["success"])
                    if d[t].get("closest_approach_m") is not None:
                        cas.append(d[t]["closest_approach_m"])
                    if t in p and lab != "P":
                        flips += (not p[t]["success"]) and d[t]["success"]
                        breaks += p[t]["success"] and not d[t]["success"]
                    k = K_OF.get(lab)
                    if k is not None and t in p:
                        a = dpn_series(root, runs[(lab, s)], d[t]); b = dpn_series(root, runs[("P", s)], p[t])
                        if a is not None and b is not None and len(a) > k and len(b) > k:
                            w = slice(k, k + 3)
                            den = float(np.mean(b[w]))
                            if den > 0:
                                gaps.append(float(np.mean(a[w])) / den)
            lo, hi = wilson_ci(succ, n)
            res[lab] = {"n": n, "success": succ, "rate": succ / n if n else None, "wilson95": [lo, hi],
                        "flips_vs_P": flips if lab != "P" else None, "breaks_vs_P": breaks if lab != "P" else None,
                        "median_closest_approach": float(np.median(cas)) if cas else None,
                        "post_handback_gap_ratio_median": float(np.median(gaps)) if gaps else None,
                        "post_handback_gap_n": len(gaps), "runs": [runs[(lab, s)] for s in SEEDS]}
        r_all, r_p = res["all"]["rate"], res["P"]["rate"]
        kstar = None
        if r_all is not None and r_p is not None:
            for lab in ("k1", "k2", "k4"):
                r = res[lab]["rate"]
                if r is not None and r >= r_all - 0.10 and r >= r_p + 0.20:
                    kstar = K_OF[lab]; break
        # forward-0 check at seed 0 against R-044 (start pose) / R-048 (camera)
        ref = load(root, F0_REF[row]); f0 = {}
        for lab in LABELS:
            d = data[(lab, 0)]
            m = [t for t in ids if t in d and t in ref and d[t].get("d_pn_f0") is not None]
            f0[lab] = {"match": sum(abs(d[t]["d_pn_f0"] - ref[t]["d_pn_f0"]) <= 0.05 for t in m), "n": len(m)}
        out["rows"][row] = {"by_k": res, "k_star": kstar if kstar is not None else "throughout (> 4 forwards)",
                            "f0_vs_reference": {"reference": F0_REF[row], **f0}}
    R, C = out["rows"]["ris"]["by_k"], out["rows"]["cam"]["by_k"]
    rate = lambda d, k: d[k]["rate"]
    ok = lambda *xs: all(x is not None for x in xs)
    out["expectations"] = {
        "E1_ris_k1_within_10pp_of_k4": abs(rate(R, "k1") - rate(R, "k4")) <= 0.10 if ok(rate(R, "k1"), rate(R, "k4")) else None,
        "E2_ris_gap_k1_le_0.6": R["k1"]["post_handback_gap_ratio_median"] <= 0.6 if ok(R["k1"]["post_handback_gap_ratio_median"]) else None,
        "E3_cam_all_ge_27_of_30": C["all"]["success"] >= 27 if C["all"]["n"] == 30 else None,
        "E4_cam_k4_ge_15pp_below_all": rate(C, "all") - rate(C, "k4") >= 0.15 if ok(rate(C, "all"), rate(C, "k4")) else None,
        "E5_cam_gap_ge_0.8_every_finite_k": all(C[k]["post_handback_gap_ratio_median"] >= 0.8 for k in ("k1", "k2", "k4"))
        if ok(*(C[k]["post_handback_gap_ratio_median"] for k in ("k1", "k2", "k4"))) else None,
        "E6_cam_monotone_within_one": all(C[b]["success"] >= C[a]["success"] - 1 for a, b in zip(LABELS, LABELS[1:]))
        if all(C[k]["n"] for k in LABELS) else None,
    }
    bad_rc = [l.strip() for l in open(os.path.join(root, "r057", "queue.log"))
              if re.search(r" rc=(?!0\b)\d+", l)] if os.path.exists(os.path.join(root, "r057", "queue.log")) else []
    f0_ok = all(v["match"] >= 9 for row in ("ris", "cam") for lab, v in out["rows"][row]["f0_vs_reference"].items()
                if lab != "reference" and v["n"])
    out["uninterpretable_checks"] = {"f0_match_ge_9_of_10_every_run": f0_ok,
                                     "cam_all_ge_24_of_30": C["all"]["success"] >= 24 if C["all"]["n"] == 30 else None,
                                     "nonzero_rc": bad_rc}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check-reuse", action="store_true")
    a = ap.parse_args()
    if a.check_reuse:
        ok, why = check_reuse(a.runs_root)
        print("R-053 reusable" if ok else "R-053 NOT reusable: " + "; ".join(why))
        sys.exit(0 if ok else 1)
    res = score(a.runs_root)
    out = a.out or os.path.join(a.runs_root, "r057", "score.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1, default=float)
    for row, v in res["rows"].items():
        print(f"== {row}: k* = {v['k_star']}; f0 vs {v['f0_vs_reference']['reference']}: "
              + " ".join(f"{k}:{x['match']}/{x['n']}" for k, x in v["f0_vs_reference"].items() if k != "reference"))
        for lab, x in v["by_k"].items():
            print(f"  {lab:4s} {x['success']}/{x['n']} [{x['wilson95'][0]:.2f},{x['wilson95'][1]:.2f}] flips {x['flips_vs_P']} "
                  f"breaks {x['breaks_vs_P']} CA {x['median_closest_approach']} gap {x['post_handback_gap_ratio_median']}")
    print("expectations:", json.dumps(res["expectations"]))
    print("checks:", json.dumps(res["uninterpretable_checks"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
