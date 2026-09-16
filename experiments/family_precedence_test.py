"""Test 2 of docs/FAILURE_FAMILY_AUDIT.md -- precedence sensitivity.

READ-ONLY. Imports `vla_harness.mining.classify` and does not modify it.

The question: `classify()` evaluates its rules most-specific-first and the first
match wins, so every family's catchment is its own condition MINUS every earlier
condition. If the conditions overlap, the reported counts are partly a fact
about the rule ORDER rather than about the rollouts.

Rather than permute the order and re-run (which conflates the effect with
re-segmentation noise), this computes the full PREDICATE VECTOR per failure --
which rules would fire, independent of order -- and then applies orderings to
that. Same answer, one segmentation pass, and it also exposes the overlap
directly, which is the thing worth knowing.

Usage:  .venvs/lerobot/bin/python experiments/family_precedence_test.py
"""
from __future__ import annotations

import glob
import itertools
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla_harness.schema import TraceStore
from vla_harness.mining import classify as C
from vla_harness.mining import signals as sig
from vla_harness.mining.phases import PhaseSegmenter, terminal_behaviour
from vla_harness.mining.phases_libero import LiberoPhaseSegmenter

# The rules of classify.classify(), as (name, family, predicate), in the order
# the shipped classifier evaluates them. Kept in lockstep with classify.py:130+.
RULES = [
    ("never_reached", "planning",
     lambda f: f["terminal"] == "never_reached"),
    ("wrong_object", "spatial_reasoning",
     lambda f: f["wrong_object"]),
    ("lost_target", "visual_grounding",
     lambda f: f["final_err"] is not None and f["final_err"] > C.LOST_TARGET_M),
    ("repeat_attempts", "recovery",
     lambda f: f["attempts"] >= 2 and f["spread"] is not None
     and f["spread"] < C.REPEAT_SPREAD_M),
    ("any_attempt", "manipulation",
     lambda f: f["attempts"] >= 1),
]
SHIPPED = [r[0] for r in RULES]


def _segmenter_for(r):
    """Same selection the production entry points make (experiments/e2e.py:58):
    LIBERO traces get the LIBERO segmenter, everything else the toy one."""
    if r.steps and "_gt_eef_to_object" in r.steps[0].obs_state:
        return LiberoPhaseSegmenter()
    return PhaseSegmenter()


def features(r):
    """Everything the decision rules read, computed once."""
    seg = _segmenter_for(r)
    segs, info = seg(r)
    if info.get("skipped"):
        return None, info["skipped"]
    S = sig.pick(r)
    if S is None:
        return None, "no signal set"
    return {
        "terminal": terminal_behaviour(r, segs),
        "wrong_object": bool(C._wrong_object(r, S)),
        "final_err": S.final_error_m(r),
        "attempts": S.grasp_attempts(r),
        "spread": sig.spread_m(S.attempt_points(r)),
    }, None


def assign(f, order):
    for name in order:
        rule = next(x for x in RULES if x[0] == name)
        if rule[2](f):
            return rule[1]
    return "ambiguous"


def main():
    dirs = sorted(glob.glob("runs/camp_*"))
    if not dirs:
        sys.exit("no runs/camp_* directories found")

    feats, skipped, n_total, n_fail = [], Counter(), 0, 0
    for d in dirs:
        suite = os.path.basename(d).split("_", 2)[-1]
        for r in TraceStore("runs", os.path.basename(d)).load():
            n_total += 1
            if r.success:
                continue
            n_fail += 1
            f, why = features(r)
            if f is None:
                skipped[why] += 1
                continue
            f["suite"] = suite
            feats.append(f)

    print(f"rollouts {n_total}  failures {n_fail}  "
          f"featurised {len(feats)}  skipped {sum(skipped.values())}")
    for k, v in skipped.items():
        print(f"    skipped: {k}: {v}")

    # --- 1. how many rules fire per failure, independent of order -----------
    fires = [[n for n, _, p in RULES if p(f)] for f in feats]
    print("\n1. RULES FIRING PER FAILURE (order-independent)")
    for k, v in sorted(Counter(len(x) for x in fires).items()):
        print(f"   {k} rule(s) fire: {v:4d}  ({v/len(feats)*100:.1f}%)")
    multi = sum(1 for x in fires if len(x) > 1)
    print(f"   -> {multi} of {len(feats)} ({multi/len(feats)*100:.1f}%) are "
          f"decided by ORDER, not by the taxonomy")

    print("\n   most common co-firing sets:")
    for combo, n in Counter(tuple(x) for x in fires).most_common(8):
        print(f"     {n:4d}  {' + '.join(combo) if combo else '(none)'}")

    # --- 2. counts under the shipped order ----------------------------------
    shipped_counts = Counter(assign(f, SHIPPED) for f in feats)
    print("\n2. SHIPPED ORDER")
    for fam, n in shipped_counts.most_common():
        print(f"   {fam:20s} {n:4d}  ({n/len(feats)*100:.1f}%)")

    # --- 3. range of each family's count over ALL orderings -----------------
    print("\n3. RANGE OVER ALL 120 RULE ORDERINGS")
    lo, hi = defaultdict(lambda: 10**9), defaultdict(int)
    for order in itertools.permutations(SHIPPED):
        c = Counter(assign(f, order) for f in feats)
        for fam in set(r[1] for r in RULES) | {"ambiguous"}:
            lo[fam] = min(lo[fam], c[fam])
            hi[fam] = max(hi[fam], c[fam])
    print(f"   {'family':20s} {'shipped':>8s} {'min':>6s} {'max':>6s} "
          f"{'swing':>7s}  {'swing/shipped':>14s}")
    for fam in sorted(hi, key=lambda k: -shipped_counts[k]):
        s = shipped_counts[fam]
        sw = hi[fam] - lo[fam]
        ratio = f"{sw/s:.1f}x" if s else "inf"
        print(f"   {fam:20s} {s:8d} {lo[fam]:6d} {hi[fam]:6d} {sw:7d}  "
              f"{ratio:>14s}")

    # --- 4. what each family would be if its rule were evaluated first ------
    print("\n4. IF EACH RULE WERE EVALUATED FIRST (its unconditioned size)")
    for name, fam, pred in RULES:
        n = sum(1 for f in feats if pred(f))
        s = shipped_counts[fam]
        print(f"   {name:16s} -> {fam:20s} alone {n:4d}   shipped {s:4d}   "
              f"suppressed {n - s:4d}")

    # --- 5. per-suite shipped counts ----------------------------------------
    print("\n5. SHIPPED COUNTS BY SUITE")
    per = defaultdict(Counter)
    for f in feats:
        per[f["suite"]][assign(f, SHIPPED)] += 1
    fams = [r[1] for r in RULES] + ["ambiguous"]
    print(f"   {'suite':18s}" + "".join(f"{x[:12]:>14s}" for x in fams))
    for suite in sorted(per):
        print(f"   {suite:18s}" + "".join(f"{per[suite][x]:14d}" for x in fams))


if __name__ == "__main__":
    main()
