"""Slice a LIBERO-Plus run by perturbation category, difficulty or task, and
report success rate and self-referenced OOD separation for each slice.

Generalises the one-off analyses: pick a grouping and it prints the whole table.

    # every perturbation category
    .venvs/lerobot/bin/python experiments/compare.py \
        --ref groot_harness_parity --query lplus_fail_groot --group-by category

    # by difficulty level
    ... --group-by difficulty

    # one category, broken out by task
    ... --category "Camera Viewpoints" --group-by task

    # what categories exist
    ... --list

Columns:
  n / succ%          how hard that slice is for the policy
  OOD succ / fail    mean % of steps outside the reference cloud, by outcome
  sep                fail/succ ratio -- how well OOD separates outcome in this slice
  p                  Mann-Whitney on per-episode OOD fraction

The reference is the policy's OWN successful rollouts in --ref, with the
threshold calibrated leave-one-out at p95, so "OOD succ" lands near 5% by
construction. That is the calibration check, not a result.

CAVEAT that belongs with every number here: the feature space is 5
proprioceptive dims (eef xyz + both gripper fingers). It is blind to anything
that does not move the arm -- so a low separation in Light Conditions or
Background Textures is expected and is NOT evidence those perturbations are
harmless. See docs/EXP_EMBEDDING_OOD.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vla_harness.schema import TraceStore            # noqa: E402
from ood_selfref import norm_params, traj, nn        # noqa: E402

CLS = ("third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json")


def classification() -> dict:
    """suite -> {task_id: (category, difficulty)}."""
    raw = json.load(open(CLS))
    return {s: {e["id"]: (e["category"], e["difficulty_level"]) for e in v}
            for s, v in raw.items()}


def annotate(r, cls):
    """(category, difficulty) for a rollout, or (None, None) if not LIBERO-Plus."""
    env = r.fingerprint.get("env", {}) if isinstance(r.fingerprint, dict) else {}
    suite, tid = env.get("suite"), env.get("task_id")
    return cls.get(suite, {}).get(tid, (None, None))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--group-by", default="category",
                    choices=["category", "difficulty", "task", "none"])
    ap.add_argument("--category", help="filter to one perturbation category")
    ap.add_argument("--difficulty", type=int, help="filter to one difficulty level")
    ap.add_argument("--task", type=int, help="filter to one task id")
    ap.add_argument("--suite", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-n", type=int, default=5,
                    help="skip groups smaller than this")
    ap.add_argument("--list", action="store_true",
                    help="list categories and difficulties in the query, then exit")
    a = ap.parse_args()

    cls = classification()
    q_all = TraceStore("runs", a.query).load()
    if a.suite:
        q_all = [r for r in q_all if a.suite in r.task_id]

    if a.list:
        cats, difs = defaultdict(int), defaultdict(int)
        for r in q_all:
            c, d = annotate(r, cls)
            cats[c] += 1
            difs[d] += 1
        print(f"{a.query}: {len(q_all)} episodes")
        print("\ncategories:")
        for c, n in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"   {str(c):26s} {n:5d}")
        print("difficulty levels:")
        for d, n in sorted(difs.items(), key=lambda x: (x[0] is None, x[0])):
            print(f"   {str(d):26s} {n:5d}")
        return

    # --- reference cloud + calibrated threshold -----------------------------
    mu, sd = norm_params()
    ref = [r for r in TraceStore("runs", a.ref).load()
           if r.success and (not a.suite or a.suite in r.task_id)]
    T = [t for t in (traj(r, mu, sd) for r in ref) if t is not None]
    if len(T) < 5:
        sys.exit(f"reference {a.ref}: only {len(T)} usable successful rollouts")
    cloud = np.concatenate(T)
    loo = [nn(t, np.concatenate([x for j, x in enumerate(T) if j != i]))
           for i, t in enumerate(T)]
    thr = float(np.percentile(np.concatenate(loo), 100 * (1 - a.alpha)))
    print(f"reference {a.ref}: {len(T)} successful rollouts, {len(cloud)} frames"
          f"   threshold p{100*(1-a.alpha):.0f} = {thr:.3f}")

    # --- score and group ----------------------------------------------------
    groups = defaultdict(lambda: {True: [], False: []})
    for r in q_all:
        c, d = annotate(r, cls)
        if a.category and c != a.category:
            continue
        if a.difficulty is not None and d != a.difficulty:
            continue
        tid = r.fingerprint.get("env", {}).get("task_id")
        if a.task is not None and tid != a.task:
            continue
        t = traj(r, mu, sd)
        if t is None:
            continue
        frac = float((nn(t, cloud) > thr).mean())
        key = {"category": c, "difficulty": f"L{d}", "task": tid,
               "none": "all"}[a.group_by]
        groups[key][r.success].append(frac)

    if not groups:
        sys.exit("no episodes matched the filters")

    from scipy.stats import mannwhitneyu
    print(f"\nquery {a.query}   grouped by {a.group_by}"
          + (f"   [category={a.category}]" if a.category else "")
          + (f"   [difficulty=L{a.difficulty}]" if a.difficulty is not None else ""))
    print(f"  {'group':26s}{'n':>5s}{'succ%':>7s}{'OOD succ':>10s}"
          f"{'OOD fail':>10s}{'sep':>7s}{'p':>11s}")
    rows = sorted(groups.items(), key=lambda kv: -(len(kv[1][True]) + len(kv[1][False])))
    for key, d in rows:
        S, F = d[True], d[False]
        n = len(S) + len(F)
        if n < a.min_n:
            continue
        sr = len(S) / n * 100
        ms = np.mean(S) * 100 if S else float("nan")
        mf = np.mean(F) * 100 if F else float("nan")
        sep = (np.mean(F) / np.mean(S)) if S and F and np.mean(S) > 0 else float("nan")
        if len(S) >= 3 and len(F) >= 3:
            _, p = mannwhitneyu(S, F)
            ps = f"{p:.1e}" + ("" if p < 0.05 else " ns")
        else:
            ps = "n/a"
        print(f"  {str(key):26s}{n:5d}{sr:7.1f}{ms:9.1f}%{mf:9.1f}%"
              f"{sep:7.1f}{ps:>11s}")
    print("\n  OOD succ ~5% is the calibration check, not a result.")
    print("  5 proprioceptive dims only: low separation on appearance-only")
    print("  perturbations is EXPECTED, not evidence they are harmless.")


if __name__ == "__main__":
    main()
