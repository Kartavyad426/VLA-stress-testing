"""L3 tier 1 -- deterministic phase detectors.

Each detector declares the state keys it needs. If a trace lacks them the
detector is SKIPPED with a recorded reason, never crashes (G8). That is what
lets the same miner run over toy traces and LIBERO traces.

The miner never imports an env (G9). It consumes Rollout objects only.
"""
from __future__ import annotations

from dataclasses import dataclass

APPROACH, PREGRASP, GRASP, TRANSPORT, RETRY, IDLE = (
    "approach", "pre_grasp", "grasp", "transport", "retry", "idle")


@dataclass
class PhaseSegment:
    phase: str
    t_start: int
    t_end: int


class PhaseSegmenter:
    """Segment a rollout into manipulation phases from simulator state.

    Thresholds are in metres and are env-specific; they are constructor args
    rather than constants so a LIBERO adapter can supply its own.
    """
    requires = ["_gt_ee_to_obj", "gripper", "holding"]

    def __init__(self, pregrasp_radius=0.12, grasp_radius=0.06):
        self.pregrasp_radius = pregrasp_radius
        self.grasp_radius = grasp_radius

    def missing(self, r) -> list[str]:
        if not r.steps:
            return list(self.requires)
        have = r.steps[0].obs_state
        return [k for k in self.requires if k not in have]

    def __call__(self, r) -> tuple[list[PhaseSegment], dict]:
        miss = self.missing(r)
        if miss:
            return [], {"skipped": f"missing state keys: {miss}"}

        labels = []
        for s in r.steps:
            d = s.obs_state["_gt_ee_to_obj"]
            grip = s.obs_state["gripper"]
            if s.obs_state["holding"]:
                labels.append(TRANSPORT)
            elif grip > 0.5:
                labels.append(GRASP)
            elif d < self.grasp_radius:
                labels.append(PREGRASP)
            elif d < self.pregrasp_radius:
                labels.append(PREGRASP)
            else:
                labels.append(APPROACH)

        # a grasp phase that is followed by more approach is a retry
        segs, cur, start = [], labels[0], 0
        for i, lab in enumerate(labels[1:], 1):
            if lab != cur:
                segs.append(PhaseSegment(cur, start, i - 1))
                cur, start = lab, i
        segs.append(PhaseSegment(cur, start, len(labels) - 1))

        seen_grasp = False
        for s in segs:
            if s.phase == GRASP:
                seen_grasp = True
            elif seen_grasp and s.phase in (APPROACH, PREGRASP):
                s.phase = RETRY
        return segs, {}


def divergence_point(r, nominal, tol=0.03) -> dict:
    """First timestep where the perturbed trace leaves the nominal envelope.

    Compares end-effector position against a reference rollout at the same seed.
    Trajectory distance is a SECONDARY signal -- successful manipulation admits
    many valid trajectories -- so this is reported alongside phase segmentation,
    never instead of it.
    """
    if nominal is None:
        return {"skipped": "no nominal reference at this seed"}
    a, b = r.series("ee_xy"), nominal.series("ee_xy")
    for t, (p, q) in enumerate(zip(a, b)):
        d = ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5
        if d > tol:
            return {"t": t, "distance_m": round(d, 4), "tolerance_m": tol}
    return {"t": None, "note": "trace stayed inside nominal envelope"}


def terminal_behaviour(r, segs) -> str:
    """What the policy was doing when the episode ended."""
    if r.success:
        return "success"
    retries = sum(1 for s in segs if s.phase == RETRY)
    grasps = sum(1 for s in segs if s.phase == GRASP)
    if retries >= 2:
        return "retry_loop"
    if grasps >= 1:
        return "failed_grasp_no_retry"
    if segs and segs[-1].phase == APPROACH:
        return "never_reached"
    return "unknown"
