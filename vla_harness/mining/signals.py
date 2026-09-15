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
    CLOSED_M = 0.030          # sum of both finger positions, below this = closed

    @staticmethod
    def available(r) -> bool:
        return bool(r.steps) and "_gt_eef_to_object" in r.steps[0].obs_state

    @staticmethod
    def _target(r):
        d = r.steps[0].obs_state.get("_gt_eef_to_object") or {}
        return sorted(d)[0] if d else None      # BDDL lists the manipuland first

    @staticmethod
    def _closed(st) -> bool:
        q = st.get("gripper_qpos") or []
        return bool(q) and sum(abs(x) for x in q) < LiberoSignals.CLOSED_M

    @staticmethod
    def final_error_m(r):
        t = LiberoSignals._target(r)
        if t is None:
            return None
        d = r.steps[-1].obs_state.get("_gt_eef_to_object") or {}
        return d.get(t)

    @staticmethod
    def _closure_steps(r):
        """Rising edges of gripper closure -- one per attempt, not per step."""
        out, prev = [], False
        for s in r.steps:
            c = LiberoSignals._closed(s.obs_state)
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
