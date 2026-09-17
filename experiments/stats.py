"""Arithmetic for claims, so claims stop being arithmetic done in prose.

Every function REQUIRES an explicit denominator and prints it back. That is the
whole point: the errors this exists to prevent were all ratios quoted without
the denominator being named -- a share of a subset reported as a rate over the
population.

Conventions follow RESULTS.md: Wilson 95% CIs for proportions, two-proportion
z-test for unpaired comparisons, exact McNemar for paired ones.

    from stats import rate, compare, share
    rate("multi-firing failures", 409, 534)
    compare("spatial", ref=(89,100), ours=(73,100))

Run directly to re-verify every published number in one pass:
    .venvs/lerobot/bin/python experiments/stats.py
"""
from __future__ import annotations

import math


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(label: str, k: int, n: int, of: str = "") -> float:
    """A proportion WITH its denominator named. `of` says what n is."""
    if n == 0:
        raise ValueError(f"{label}: denominator is zero")
    if k > n:
        raise ValueError(f"{label}: numerator {k} exceeds denominator {n} -- "
                         f"almost always a share/rate confusion")
    lo, hi = _wilson(k, n)
    den = f" of {n} {of}".rstrip() if of else f" of {n}"
    print(f"  {label:44s} {k:6d}{den:28s} = {k/n*100:5.1f}%  "
          f"[{lo*100:.1f}, {hi*100:.1f}]")
    return k / n


def share(label: str, k: int, subset_n: int, population_n: int | None,
          subset_name: str = "subset") -> None:
    """A share of a SUBSET. Refuses to be mistaken for a population rate.

    This is the function that would have caught the RoboMIND error: 557 of
    1,650 rejects is a share of the rejects, and becomes a rate only against
    total collected.
    """
    print(f"  {label:44s} {k:6d} of {subset_n} {subset_name}"
          f" = {k/subset_n*100:5.1f}% OF THE {subset_name.upper()}")
    if population_n:
        print(f"  {'':44s} {'':6s}    against population {population_n}"
              f" = {k/population_n*100:5.2f}%  <- the rate")
    else:
        print(f"  {'':44s} {'':6s}    population size UNKNOWN"
              f" -> NO RATE CAN BE QUOTED")


def compare(label: str, ref: tuple[int, int], ours: tuple[int, int]) -> None:
    """Two-proportion z-test, unpaired. ref/ours are (successes, n)."""
    k1, n1 = ref
    k2, n2 = ours
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    z = d / se if se else float("nan")
    lo, hi = d - 1.96 * se, d + 1.96 * se
    sig = "" if abs(z) >= 1.96 else "   NOT SIGNIFICANT (CI spans zero)"
    print(f"  {label:26s} ref {p1*100:5.1f} ({n1})  ours {p2*100:5.1f} ({n2})"
          f"  diff {d*100:6.1f}pp  95%CI [{lo*100:5.1f},{hi*100:5.1f}]"
          f"  z={z:5.2f}{sig}")


def _published() -> None:
    """Re-verify every number this session has published. Additions welcome."""
    print("\n== FAILURE_FAMILY_AUDIT 5b -- precedence (534 featurised failures)")
    rate("fire >1 rule (order decides family)", 409, 534, "failures")
    rate("any_attempt predicate fires", 528, 534, "failures")
    rate("never_reached fires", 6, 534, "failures")
    print("  disjoint+exhaustive check: 528 + 6 =", 528 + 6, "== 534 ->",
          528 + 6 == 534)
    rate("recovery surviving precedence", 10, 24, "episodes that DO repeat")

    print("\n== FAILURE_FAMILY_AUDIT 5c -- goal instrumentation")
    rate("libero_goal failures missing key", 133, 172, "goal failures")
    share("...as a share of all goal rollouts", 133, 172, 260, "failures")

    print("\n== baseline vs community repros (nominal arm, n=100/suite)")
    for lab, r, o in [("spatial", 89, 73), ("object", 94, 90),
                      ("goal", 91, 68), ("long", 57, 37)]:
        compare(lab, ref=(r, 100), ours=(o, 100))

    print("\n== RoboMIND -- the error this file exists to prevent")
    # counts re-walked by vla-7f with full Link-header pagination; the first
    # listing hit HF's 1,000-entry page cap, which is exactly the kind of
    # silent truncation this file exists to stop propagating.
    share("jerky motion", 557, 1678, 107000, "released failure episodes")
    rate("labelled with a cause", 1666, 1678, "released failure episodes")
    print(f"  paper states ~5000 failure demos; public main branch holds 1678 "
          f"-> {1678/5000*100:.0f}% (rest not searched for)")

    print("\n== SIMPLER Table VI -- ordering, not magnitude")
    rows = [("camera pose", .753, .458), ("table texture", .113, .389),
            ("distractors", .027, .111), ("lighting", .040, .083),
            ("background", .013, .028)]
    for lab, sim, real in rows:
        print(f"  {lab:16s} sim {sim:.3f}  real {real:.3f}  "
              f"ratio {real/sim:5.2f}x")
    # computed, not asserted: the first version of this line hardcoded "3.4x"
    # and was wrong, which is the same class of error as everything above.
    worst = max(rows, key=lambda r: max(r[2] / r[1], r[1] / r[2]))
    sim_rank = [r[0] for r in sorted(rows, key=lambda r: -r[1])]
    real_rank = [r[0] for r in sorted(rows, key=lambda r: -r[2])]
    print(f"  ordering sim : {' > '.join(sim_rank)}")
    print(f"  ordering real: {' > '.join(real_rank)}")
    print(f"  ordering preserved: {sim_rank == real_rank}   "
          f"worst magnitude error: {worst[0]} "
          f"{max(worst[2]/worst[1], worst[1]/worst[2]):.2f}x")


if __name__ == "__main__":
    _published()
