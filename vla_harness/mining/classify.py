"""L3 -- taxonomy assignment.

Tier 1 (deterministic) decides everything it can from simulator state.
Tier 3 (LLM judge) is consulted ONLY for what tier 1 leaves ambiguous, and
NEVER overrides a tier-1 verdict. Every label records its source so a reader
can tell which numbers are reproducible.

Two rules here were added because the oracle test caught the miner getting
them wrong -- see the comments on `_wrong_object` and `_attempt_spread`.
"""
from __future__ import annotations

import math

from .phases import PhaseSegmenter, terminal_behaviour, divergence_point
from . import signals as sig

FAMILIES = ["visual_grounding", "language_grounding", "spatial_reasoning",
            "planning", "manipulation", "recovery", "distribution_shift",
            "ambiguous"]

WRONG_OBJECT_M = 0.09      # ended this close to a distractor -> wrong object
LOST_TARGET_M = 0.10       # ended this far from target -> went somewhere else
REPEAT_SPREAD_M = 0.015    # grasp attempts within this radius -> identical


def _wrong_object(r, S=None) -> bool:
    if S is not None:
        return S.wrong_object(r)
    return _wrong_object_toy(r)


def _wrong_object_toy(r) -> bool:
    """Did it end up at a distractor rather than the named target?

    The first version of this classifier inferred object-selection failures
    from error MAGNITUDE, which is wrong: a large error can equally mean a
    grounding failure with no distractor involved. Selection has its own
    signal -- proximity to a distractor -- and it must be checked directly.
    """
    dd = r.series("_gt_dist_to_distractor")
    dt = r.series("_gt_ee_to_obj")
    if not dd or not dt or dd[-1] == float("inf"):
        return False
    return dd[-1] < WRONG_OBJECT_M and dd[-1] < dt[-1] * 0.6


def _attempt_spread(r) -> float | None:
    """How far apart were the grasp attempts?

    Distinguishes two failures that look identical in aggregate:
      * repeated attempts at the SAME point -> the policy cannot recover,
      * attempts spread over a search pattern -> the policy is recovering
        correctly but its grasp precision is the problem.
    Without this the miner labelled every failed-then-retried episode
    'recovery', masking the manipulation fault underneath.
    """
    pts = [s.obs_state["ee_xy"] for s in r.steps
           if s.obs_state.get("gripper", 0) > 0.5
           and not s.obs_state.get("holding")]
    if len(pts) < 2:
        return None
    return max(math.dist(a, b) for a in pts for b in pts)


# C8/DG-10 -- failure COST, not just failure RATE (PLAN.md §7b.3).
# The proposal counts failures; buyers care what a failure DOES. A timeout with
# the gripper open is free; a dropped part stops a line; a collision is a safety
# event. Same success rate, very different risk. Rated our most novel
# contribution by the landscape survey, and previously unimplemented -- which
# meant no row the harness could emit satisfied the schema's own ✱ rule.
COST_BENIGN, COST_DISRUPTIVE, COST_SAFETY = "benign", "disruptive", "safety"


def failure_cost(r, diagnosis) -> dict:
    """Classify a failure by CONSEQUENCE, from terminal state.

    Env-agnostic: reads only declared state keys, and abstains when it cannot
    tell (G8) rather than guessing a cost a client would act on.
    """
    if r.success:
        return {"cost": None, "reason": "succeeded"}
    term = diagnosis.get("terminal")
    held = r.series("holding")
    dropped = bool(held) and any(held) and not held[-1]

    if dropped:
        return {"cost": COST_DISRUPTIVE,
                "reason": "object was grasped and then released before the goal"}
    if term in ("retry_loop", "failed_grasp_no_retry"):
        return {"cost": COST_BENIGN,
                "reason": f"{term}: ended without acquiring the object"}
    if term == "never_reached":
        return {"cost": COST_BENIGN, "reason": "no contact with the scene"}
    return {"cost": None, "reason": f"no rule for terminal={term!r}",
            "needs_review": True}


def classify(r, nominal=None, segmenter=None) -> dict:
    seg = segmenter or PhaseSegmenter()
    segs, info = seg(r)
    if info.get("skipped"):
        # A SUCCESS needs no failure diagnosis, instrumented or not. Checking
        # abstention first labelled successful episodes with missing state keys
        # `ambiguous` -- e.g. libero_goal/task2 c3560775c39c, a clean pick-and-place
        # shown as an undiagnosable failure. Abstention is about failures only.
        if r.success:
            return {"family": None, "families": [], "predicates": {},
                    "source": "tier1", "confident": True,
                    "reason": "succeeded (detectors abstained: "
                              f"{info['skipped']})"}
        return {"family": "ambiguous", "families": [], "predicates": {},
                "source": "tier1", "reason": info["skipped"], "confident": False}

    term = terminal_behaviour(r, segs)
    # Signals are resolved per-env. The classifier previously read the toy's
    # keys directly, so on LIBERO every rule fell through to `ambiguous` while
    # reporting zero detector abstentions.
    S = sig.pick(r)
    if S is None:
        return {"family": "ambiguous", "families": [], "predicates": {},
                "source": "tier1", "confident": False,
                "reason": "no signal set matches this rollout's state keys"}
    final_err = S.final_error_m(r)
    attempts = S.grasp_attempts(r)
    spread = sig.spread_m(S.attempt_points(r))

    d = {"source": "tier1", "confident": True, "signals": S.name,
         "phases": [s.phase for s in segs], "terminal": term,
         "divergence": divergence_point(r, nominal),
         "final_error_m": final_err, "grasp_attempts": attempts,
         "attempt_spread_m": spread}

    if r.success:
        d.update(family=None, families=[], predicates={}, reason="succeeded")
        d["failure_cost"] = failure_cost(r, d)
        return d

    # --- rules: ALL are evaluated, none is suppressed -----------------------
    # This used to be first-match-wins. 76.6% of LIBERO failures satisfy more
    # than one rule, so the reported family was decided by rule ORDER, and every
    # family except manipulation could be driven to zero by reordering alone
    # (RESULTS.md R-008). A family defined by precedence is not meaningful on its
    # own -- `manipulation` really meant "attempted AND not far AND not at another
    # object AND not repeating AND did reach pre-grasp".
    #
    # So every rule is evaluated independently and the diagnosis reports all that
    # matched. `family` is kept only as an ORDER-INDEPENDENT key for grouping
    # (families joined in the fixed FAMILIES order), so clustering and the
    # manifest keep working. Use `families` / `predicates` for analysis.
    matched = {}
    if term == "never_reached":
        matched["planning"] = "never entered pre-grasp"
    if _wrong_object(r, S):
        matched["spatial_reasoning"] = ("ended nearer another task object than "
                                        "the target (may be the destination)")
    if final_err is not None and final_err > LOST_TARGET_M:
        matched["visual_grounding"] = (f"ended {final_err*100:.1f} cm from target "
                                       f"(> {LOST_TARGET_M*100:.0f} cm)")
    if attempts >= 2 and spread is not None and spread < REPEAT_SPREAD_M:
        matched["recovery"] = (f"{attempts} attempts within {spread*100:.1f} cm "
                               f"— repeating, not searching")
    if attempts >= 1:
        matched["manipulation"] = (f"{attempts} grasp attempt(s)"
                                   + (f"; searched over {spread*100:.1f} cm"
                                      if spread else ""))

    d["predicates"] = {"never_reached": "planning" in matched,
                       "wrong_object": "spatial_reasoning" in matched,
                       "lost_target": "visual_grounding" in matched,
                       "repeat_attempts": "recovery" in matched,
                       "any_attempt": "manipulation" in matched}
    families = [f for f in FAMILIES if f in matched]
    d["families"] = families
    d["reasons"] = {f: matched[f] for f in families}
    if families:
        d["family"] = "+".join(families)
        d["reason"] = "; ".join(f"{f}: {matched[f]}" for f in families)
    else:
        d.update(family="ambiguous", families=[], confident=False,
                 reason="no rule matched")
    d["failure_cost"] = failure_cost(r, d)
    return d


def llm_adjudicate(r, diagnosis, judge=None) -> dict:
    """Tier 3. Consulted only when tier 1 is not confident.

    `judge` is injected (LLM or human). Unset in the prototype -- the point is
    that the seam exists and is narrow, and anything it returns is tagged
    `tier3` so it can be excluded from headline numbers.
    """
    if diagnosis.get("confident"):
        return diagnosis
    if judge is None:
        return {**diagnosis, "needs_review": True}
    v = judge(r, diagnosis)
    return {**diagnosis, "family": v["family"], "source": "tier3",
            "llm_reasoning": v.get("reasoning"), "needs_review": True}
