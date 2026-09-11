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

FAMILIES = ["visual_grounding", "language_grounding", "spatial_reasoning",
            "planning", "manipulation", "recovery", "distribution_shift",
            "ambiguous"]

WRONG_OBJECT_M = 0.09      # ended this close to a distractor -> wrong object
LOST_TARGET_M = 0.10       # ended this far from target -> went somewhere else
REPEAT_SPREAD_M = 0.015    # grasp attempts within this radius -> identical


def _wrong_object(r) -> bool:
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


def classify(r, nominal=None, segmenter=None) -> dict:
    seg = segmenter or PhaseSegmenter()
    segs, info = seg(r)
    if info.get("skipped"):
        return {"family": "ambiguous", "source": "tier1",
                "reason": info["skipped"], "confident": False}

    term = terminal_behaviour(r, segs)
    final_err = (r.series("_gt_ee_to_obj") or [None])[-1]
    attempts = (r.series("grasp_attempts") or [0])[-1]
    spread = _attempt_spread(r)

    d = {"source": "tier1", "confident": True,
         "phases": [s.phase for s in segs], "terminal": term,
         "divergence": divergence_point(r, nominal),
         "final_error_m": final_err, "grasp_attempts": attempts,
         "attempt_spread_m": spread}

    if r.success:
        d.update(family=None, reason="succeeded")
        return d

    # --- decision rules, most-specific first --------------------------------
    if term == "never_reached":
        d.update(family="planning", reason="never entered pre-grasp")
    elif _wrong_object(r):
        d.update(family="spatial_reasoning",
                 reason="terminated at a distractor, not the named target")
    elif final_err is not None and final_err > LOST_TARGET_M:
        d.update(family="visual_grounding",
                 reason=f"went confidently to the wrong place "
                        f"({final_err*100:.1f} cm from target)")
    elif attempts >= 2 and spread is not None and spread < REPEAT_SPREAD_M:
        d.update(family="recovery",
                 reason=f"{attempts} attempts within {spread*100:.1f} cm — "
                        f"repeating, not searching")
    elif attempts >= 1:
        d.update(family="manipulation",
                 reason=f"reached target ({final_err*100:.1f} cm) but grasp "
                        f"failed" + (f"; searched over {spread*100:.1f} cm"
                                     if spread else ""))
    else:
        d.update(family="ambiguous", confident=False, reason="no rule matched")
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
