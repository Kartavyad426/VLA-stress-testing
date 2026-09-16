"""LIBERO phase segmentation.

The toy `PhaseSegmenter` abstains on LIBERO because its detectors are bound to
toy state keys. This supplies the LIBERO equivalents:

    toy                 LIBERO
    _gt_ee_to_obj       _gt_eef_to_object   (a DICT, one entry per BDDL object)
    gripper (0/1)       gripper_qpos        (two finger positions, metres)
    holding (bool)      -- ABSENT, must be DERIVED

`holding` is the interesting one and the main risk. The toy env decides it; here
it is inferred, and if the inference is wrong every downstream phase is wrong.
Thresholds are constructor arguments, to be fitted from demonstrations rather
than guessed (see `fit_from_demos`).
"""
from __future__ import annotations

from .phases import (PhaseSegment, APPROACH, PREGRASP, GRASP, TRANSPORT, RETRY)

# Panda gripper: ~0.04 m per finger open, ~0 closed. Midpoint is a placeholder
# until fitted from demonstrations.
DEFAULT_CLOSED_M = 0.030          # sum of both fingers, below this = closed
DEFAULT_PREGRASP_M = 0.10         # eef within this of the target = pre-grasp
DEFAULT_LIFT_M = 0.015            # object risen this far above its resting z
# Aperture movement below this over a window = the fingers have STOPPED. Free
# travel moves ~0.005 m per step (0.075 -> 0.030 in ~8 steps), so this is an
# order of magnitude under the motion it must exclude.
DEFAULT_STALL_M = 0.004


class LiberoPhaseSegmenter:
    """Phases from LIBERO simulator state.

    `target` selects which BDDL object counts as the manipulation target. The
    BDDL `obj_of_interest` list is ordered with the manipuland first, so index 0
    is the default -- but a task whose first object is a destination (a plate, a
    drawer) needs this overridden, which is why it is a parameter and not a
    constant.
    """

    requires = ["_gt_eef_to_object", "gripper_qpos", "_gt_object_pos"]

    def __init__(self, pregrasp_m=DEFAULT_PREGRASP_M, closed_m=DEFAULT_CLOSED_M,
                 lift_m=DEFAULT_LIFT_M, target_index=0, hold_steps=6,
                 stall_m=DEFAULT_STALL_M):
        self.pregrasp_m = pregrasp_m
        self.closed_m = closed_m
        self.lift_m = lift_m
        self.target_index = target_index
        self.hold_steps = hold_steps
        self.stall_m = stall_m

    # --- required-key check, same contract as the toy segmenter (G8) -------
    def missing(self, r) -> list[str]:
        if not r.steps:
            return list(self.requires)
        have = r.steps[0].obs_state
        miss = [k for k in self.requires if k not in have]
        # completeness matters as much as presence: a partial object list gives
        # a confident wrong distance rather than an absent one (LE-2).
        if not miss and have.get("_gt_object_pos_complete") is False:
            miss.append("_gt_object_pos(incomplete)")
        return miss

    def _target(self, st) -> str | None:
        """The BDDL manipuland -- insertion order, NOT alphabetical.

        This used to `sorted(d)[i]`, which destroys BDDL order and picks the
        DESTINATION on any task whose manipuland sorts later than its container
        (`['cream_cheese_1', 'basket_1']` -> the basket). That was 210/260
        libero_object episodes and 110/185 libero_10 episodes segmenting against
        a body that never moves.

        The identical bug was fixed in `signals.py` and left here, because the
        expression lived in two modules and only one had a failing test on it
        (O1). `experiments/phase_segmenter_test.py` now asserts the two agree,
        since that duplication is what let them drift in the first place.
        """
        d = st.get("_gt_eef_to_object") or {}
        if not d:
            return None
        names = list(d)                       # BDDL order; dicts preserve it
        return names[min(self.target_index, len(names) - 1)]

    def _closed(self, st) -> bool:
        q = st.get("gripper_qpos") or []
        return bool(q) and sum(abs(x) for x in q) < self.closed_m

    @staticmethod
    def _commanded_closed(step) -> bool:
        a = step.action
        return bool(a) and len(a) >= 7 and a[6] > 0.5

    # --- the two holding routes, each callable ALONE --------------------
    # They are exposed separately because a disjunction cannot be validated by
    # asking whether the disjunction fired. `holding = by_gripper or by_lift`
    # passed "every success shows TRANSPORT" at 79/79 in every suite while BOTH
    # branches were broken -- their failures were complementary, so the OR hid
    # them (O4). Anything that scores these must score them one at a time.

    def _holding_by_gripper(self, r, i: int) -> bool:
        """Fingers STALLED while still open -- no privileged state required.

        The premise: the policy commanded the fingers shut and they did not
        shut, so something is between them. The trap is that "did not shut" is
        also true while the fingers are still TRAVELLING.

        The first version tested one frame and fired on 258/260. A persistence
        requirement (`hold_steps=6`) was then added without measuring the
        transient: free travel to full closure takes ~8 steps, so a 6-step
        window sits entirely inside it and excluded nothing (O3).

        Persistence is the wrong question anyway -- it asks "has enough time
        passed?" when the thing meant is "have the fingers STOPPED?". A blocked
        finger is stationary at an open aperture; a travelling finger is not.
        So: commanded closed throughout, still open throughout, and the aperture
        no longer falling.
        """
        n = len(r.steps)
        if i + self.hold_steps > n:
            return False
        ap = []
        for k in range(i, i + self.hold_steps):
            st = r.steps[k].obs_state
            q = st.get("gripper_qpos") or []
            if not q or not self._commanded_closed(r.steps[k]):
                return False
            a = sum(abs(x) for x in q)
            if a < self.closed_m:             # fingers met -- nothing between them
                return False
            ap.append(a)
        return max(ap) - min(ap) < self.stall_m

    def _holding_by_lift(self, r, i: int, tgt: str, z0: float) -> bool:
        """The target rose off its rest height. Needs simulator truth.

        NOT conjoined with `closed`. It used to be, and `closed` means aperture
        < closed_m -- which a held object is exactly what prevents. The lift
        route was therefore switched off precisely on the objects it should
        detect, reaching 1/79 libero_object successes (O2).
        """
        st = r.steps[i].obs_state
        zs = (st.get("_gt_object_pos", {}).get(tgt) or [0, 0, z0])[2]
        return (zs - z0) > self.lift_m

    def _holding(self, r, i: int, tgt: str = None, z0: float = 0.0) -> bool:
        if tgt is None:
            tgt = self._target(r.steps[0].obs_state)
            z0 = (r.steps[0].obs_state.get("_gt_object_pos", {}).get(tgt)
                  or [0, 0, 0])[2]
        return (self._holding_by_gripper(r, i)
                or self._holding_by_lift(r, i, tgt, z0))

    def __call__(self, r) -> tuple[list[PhaseSegment], dict]:
        miss = self.missing(r)
        if miss:
            return [], {"skipped": f"missing state keys: {miss}"}

        tgt = self._target(r.steps[0].obs_state)
        if tgt is None:
            return [], {"skipped": "no BDDL target object in trace"}

        # Resting height, from the first step -- the reference `holding` needs.
        z0 = (r.steps[0].obs_state.get("_gt_object_pos", {}).get(tgt) or [0, 0, 0])[2]

        labels, holding_flags = [], []
        for i, s in enumerate(r.steps):
            st = s.obs_state
            dist = (st.get("_gt_eef_to_object") or {}).get(tgt)
            commanded = self._commanded_closed(s)
            closed = self._closed(st)
            # DERIVED `holding`, two independent routes -- see _holding_by_*.
            # Either suffices. Closure alone never does: closing on empty air is
            # exactly the failure being detected.
            holding = self._holding(r, i, tgt, z0)
            holding_flags.append(holding)
            closed = closed or commanded

            if holding:
                labels.append(TRANSPORT)
            elif closed:
                labels.append(GRASP)
            elif dist is not None and dist < self.pregrasp_m:
                labels.append(PREGRASP)
            else:
                labels.append(APPROACH)

        segs, cur, start = [], labels[0], 0
        for i, lab in enumerate(labels[1:], 1):
            if lab != cur:
                segs.append(PhaseSegment(cur, start, i - 1))
                cur, start = lab, i
        segs.append(PhaseSegment(cur, start, len(labels) - 1))

        seen_grasp = False
        for sg in segs:
            if sg.phase == GRASP:
                seen_grasp = True
            elif seen_grasp and sg.phase in (APPROACH, PREGRASP):
                sg.phase = RETRY

        return segs, {"target": tgt, "derived_holding_steps": sum(holding_flags),
                      "thresholds": {"pregrasp_m": self.pregrasp_m,
                                     "closed_m": self.closed_m,
                                     "lift_m": self.lift_m,
                                     "stall_m": self.stall_m,
                                     "hold_steps": self.hold_steps}}


def fit_from_demos(demo_rollouts, target_index=0, pct=95):
    """Fit thresholds from DEMONSTRATIONS rather than guessing them.

    Framed as COVERAGE, deliberately: `pregrasp_m` is *the region from which
    successful grasps are observed to occur* (95th percentile of the demo
    distribution), NOT "the distance separating manipulation failures from
    grounding failures". Demonstrations contain only successes and cannot speak
    to the second. Fitting a discriminative boundary on one class and reporting
    it as discriminative is the error this docstring exists to prevent.
    """
    import statistics
    closures = []
    for r in demo_rollouts:
        prev_open = True
        for s in r.steps:
            st = s.obs_state
            q = st.get("gripper_qpos") or []
            closed = bool(q) and sum(abs(x) for x in q) < DEFAULT_CLOSED_M
            if closed and prev_open:
                d = st.get("_gt_eef_to_object") or {}
                if d:
                    closures.append(d[sorted(d)[min(target_index, len(d) - 1)]])
            prev_open = not closed
    if not closures:
        return {"fitted": False, "reason": "no gripper closure events in demos"}
    closures.sort()
    k = max(0, min(len(closures) - 1, int(len(closures) * pct / 100)))
    return {"fitted": True, "n_closure_events": len(closures),
            "pregrasp_m": round(closures[k], 4),
            "median_m": round(statistics.median(closures), 4),
            "interpretation": "coverage of observed successful grasps, not a "
                              "failure/success boundary"}
