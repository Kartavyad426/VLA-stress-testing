"""Signal extraction -- the layer that lets ONE classifier serve many envs.

`classify()` reasons about four quantities: how far from the target the rollout
ended, how many grasp attempts there were, how far apart those attempts were,
and whether it ended at the wrong object. Those are ENV-INDEPENDENT concepts.

The bug this module fixes: the classifier read the toy's state keys directly
(`_gt_ee_to_obj`, `gripper`, `holding`), none of which LIBERO emits. Every rule
fell through and every real trace came back `ambiguous` -- detectors reporting
zero abstentions, because the segmenter HAD been adapted and the classifier had
not. Silent, and it looked like a threshold problem.

Thresholds stay in metres and transfer, because both envs are metric.
"""
from __future__ import annotations

import math


class ToySignals:
    """The original toy reach env."""
    name = "toy"

    @staticmethod
    def available(r) -> bool:
        return bool(r.steps) and "_gt_ee_to_obj" in r.steps[0].obs_state

    @staticmethod
    def final_error_m(r):
        s = r.series("_gt_ee_to_obj")
        return s[-1] if s else None

    @staticmethod
    def grasp_attempts(r) -> int:
        s = r.series("grasp_attempts")
        return int(s[-1]) if s else 0

    @staticmethod
    def attempt_points(r):
        return [s.obs_state["ee_xy"] for s in r.steps
                if s.obs_state.get("gripper", 0) > 0.5
                and not s.obs_state.get("holding")]

    @staticmethod
    def wrong_object(r) -> bool:
        dd, dt = r.series("_gt_dist_to_distractor"), r.series("_gt_ee_to_obj")
        if not dd or not dt or dd[-1] == float("inf"):
            return False
        return dd[-1] < 0.09 and dd[-1] < dt[-1] * 0.6


class LiberoSignals:
    """LIBERO via robosuite/MuJoCo.

    `_gt_eef_to_object` is a DICT of distances per BDDL object, and there is no
    `gripper` flag or `grasp_attempts` counter -- both are derived here.
    """
    name = "libero"
    # Measured on 60 real rollouts, commanded-close steps only: the aperture is
    # cleanly bimodal two steps after the command --
    #     fully closed  median 0.0059   (nothing between the fingers)
    #     blocked open  median 0.0597   (an object is)
    CLOSED_M = 0.030

    @staticmethod
    def available(r) -> bool:
        return bool(r.steps) and "_gt_eef_to_object" in r.steps[0].obs_state

    @staticmethod
    def _target(r):
        """First BDDL object -- `obj_of_interest` lists the MANIPULAND first.

        This used to `sorted(d)[0]`, which destroys that order. On task
        `['cream_cheese_1', 'akita_black_bowl_1']` sorting picks the bowl (the
        DESTINATION) over the cheese (the thing being moved), so every distance
        was measured to the wrong object. Dicts preserve insertion order and
        `_object_body_names()` builds them in BDDL order, so take the first.
        """
        d = r.steps[0].obs_state.get("_gt_eef_to_object") or {}
        return next(iter(d), None)

    # --- gripper: INTENT vs RESPONSE ------------------------------------
    # `action[6]` is what the policy COMMANDED (+1 close, -1 open, binary in
    # the demonstrations). `gripper_qpos` is how the hardware RESPONDED. The
    # disagreement is the signal:
    #     commanded close + fingers open   -> something is between them = HOLDING
    #     commanded close + fingers closed -> closed on empty air
    # Intent has no lag and needs no threshold, and the holding test works with
    # NO privileged object state -- which is what a real robot has.
    @staticmethod
    def commanded_closed(step) -> bool:
        a = step.action
        return bool(a) and len(a) >= 7 and a[6] > 0.5

    @staticmethod
    def _closed(st) -> bool:
        q = st.get("gripper_qpos") or []
        return bool(q) and sum(abs(x) for x in q) < LiberoSignals.CLOSED_M

    HOLD_STEPS = 6
    STALL_M = 0.004

    @staticmethod
    def holding_by_gripper(r, i: int) -> bool:
        """Grasp inferred from intent-vs-response alone. No object poses.

        Fingers STALLED while still open -- not merely "not closed yet". Free
        travel to full closure takes ~8 steps, so a 6-step "it hasn't closed"
        window sits inside the transient and excludes nothing (O3 in
        IMPLEMENTATION_OVERSIGHTS.md). A blocked finger is stationary at an open
        aperture; a travelling finger is not.

        Mirrors `LiberoPhaseSegmenter._holding_by_gripper`; the two are asserted
        equivalent in `experiments/phase_segmenter_test.py`, because this
        expression living in two modules is exactly how O1 happened.
        """
        n = len(r.steps)
        if i + LiberoSignals.HOLD_STEPS > n:
            return False
        ap = []
        for k in range(i, i + LiberoSignals.HOLD_STEPS):
            q = r.steps[k].obs_state.get("gripper_qpos") or []
            if not q or not LiberoSignals.commanded_closed(r.steps[k]):
                return False
            a = sum(abs(x) for x in q)
            if a < LiberoSignals.CLOSED_M:
                return False
            ap.append(a)
        return max(ap) - min(ap) < LiberoSignals.STALL_M

    @staticmethod
    def final_error_m(r):
        t = LiberoSignals._target(r)
        if t is None:
            return None
        d = r.steps[-1].obs_state.get("_gt_eef_to_object") or {}
        return d.get(t)

    @staticmethod
    def _closure_steps(r):
        """Rising edges of COMMANDED closure -- one per attempt, not per step.

        Keyed on intent rather than measured aperture: the response lags by a
        step and can be blocked by the object, so counting response edges both
        misses attempts and double-counts settling. Review finding A1 noted the
        old version was "correct by accident" because a grasp action has
        dx=dy=0, so the arm had not moved when the lagged value was read.
        """
        out, prev = [], False
        for s in r.steps:
            c = LiberoSignals.commanded_closed(s)
            if c and not prev:
                out.append(s)
            prev = c
        return out

    @staticmethod
    def grasp_attempts(r) -> int:
        return len(LiberoSignals._closure_steps(r))

    @staticmethod
    def attempt_points(r):
        return [s.obs_state["eef_pos"] for s in LiberoSignals._closure_steps(r)
                if s.obs_state.get("eef_pos")]

    @staticmethod
    def wrong_object(r) -> bool:
        """Ended nearer a NON-target BDDL object than the target.

        Abstains when the object list is incomplete (LE-2) rather than
        computing over a subset, which would give a confident wrong answer.
        """
        st = r.steps[-1].obs_state
        if st.get("_gt_object_pos_complete") is False:
            return False
        d = st.get("_gt_eef_to_object") or {}
        t = LiberoSignals._target(r)
        if not d or t is None or len(d) < 2:
            return False
        others = {k: v for k, v in d.items() if k != t}
        if not others:
            return False
        near = min(others.values())
        return near < 0.09 and near < d.get(t, 1e9) * 0.6


def pick(r):
    """Choose the signal set this rollout supports, or None to abstain (G8)."""
    for S in (LiberoSignals, ToySignals):
        if S.available(r):
            return S
    return None


def spread_m(points) -> float | None:
    if len(points) < 2:
        return None
    return max(math.dist(a, b) for a in points for b in points)
