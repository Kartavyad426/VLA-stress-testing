"""L1 -- the scripted oracle.

Test fixture for the whole mining layer. Each `bug` plants a KNOWN fault with
a KNOWN trigger knob, so the miner's output can be checked against ground
truth (PLAN.md section 4).

Design rule for a fixture: a planted fault must be LATENT at nominal and only
fire under its trigger knob. A fault that fails everywhere tests nothing (the
policy is simply broken); a fault that never fails tests nothing either.

A real VLA adapter implements this same protocol and drops in unchanged.
"""
from __future__ import annotations

import math

from ..schema import Observation, Action, derive_id

ACTION_DIMS = ["dx", "dy", "gripper"]

# planted fault -> (family the miner should assign, knob the probe should find)
GROUND_TRUTH = {
    "camera_misalignment": ("visual_grounding",  "camera_yaw_deg"),
    "nearest_object":      ("spatial_reasoning", "distractor_count"),
    "unfiltered_grasp":    ("manipulation",      "pixel_noise_std"),
    "no_recovery":         ("recovery",          "pixel_noise_std"),
}


class ScriptedReachPolicy:
    """Solves the toy task. Competent unless a fault is planted.

    A competent policy does two things a naive one does not:
      * averages the last few perceptions, so sensor noise does not decide
        when to close the gripper;
      * widens its search after a failed grasp instead of repeating itself.
    Each planted fault removes one of those.
    """

    action_dims = ACTION_DIMS

    def __init__(self, bugs=(), step=0.15, grasp_tol=0.045, filter_n=5):
        self.bugs = set(bugs)
        self.step = step
        self.grasp_tol = grasp_tol
        # `unfiltered_grasp`: trust a single noisy reading
        self.filter_n = 1 if "unfiltered_grasp" in self.bugs else filter_n
        self.policy_id = derive_id(self.identity())
        self.reset()

    def identity(self) -> dict:
        """Everything that changes what this policy DOES.

        On a real VLA this is where the checkpoint revision and decoding config
        go -- a fine-tuned policy that keeps its predecessor's identity would be
        served the PRE-fine-tune rollouts on re-measurement, and report that the
        remediation data did nothing.
        """
        return {"name": "scripted-reach-v1[" +
                        (",".join(sorted(self.bugs)) or "clean") + "]",
                "bugs": sorted(self.bugs), "step": self.step,
                "grasp_tol": self.grasp_tol, "filter_n": self.filter_n,
                "action_dims": list(ACTION_DIMS),
                "checkpoint_revision": None}

    def reset(self):
        self._hist = []
        self._attempts = 0
        self._last_attempt_t = -99

    def _perceive(self, obs: Observation) -> tuple[float, float]:
        cam = obs.get("obj_cam_xy")

        if "nearest_object" in self.bugs:
            # selects whichever blob is closest rather than the one named
            cands = [cam] + list(obs.get("distractors_cam_xy", []))
            ee_cam = obs.get("ee_xy")
            cam = min(cands, key=lambda p: math.dist(p, ee_cam))

        if "camera_misalignment" in self.bugs:
            world = cam                       # believes camera frame == world
        else:
            yaw = math.radians(obs.state.get("_env_camera_yaw_deg", 0.0))
            c, s = math.cos(yaw), math.sin(yaw)
            world = (c * cam[0] - s * cam[1], s * cam[0] + c * cam[1])

        # temporal filtering: averaging N readings shrinks perception noise
        self._hist.append(world)
        h = self._hist[-self.filter_n:]
        return (sum(p[0] for p in h) / len(h), sum(p[1] for p in h) / len(h))

    def __call__(self, obs: Observation) -> Action:
        if obs.get("holding"):
            return Action([0.0, 0.0, 1.0], ACTION_DIMS)

        tx, ty = self._perceive(obs)
        ex, ey = obs.get("ee_xy")
        dx, dy = tx - ex, ty - ey
        dist = math.hypot(dx, dy)

        attempts = obs.get("grasp_attempts", 0)

        if dist < self.grasp_tol:
            if attempts >= 1 and "no_recovery" not in self.bugs:
                # competent: spiral outward to search after a miss
                a = attempts * 1.7
                return Action([0.035 * math.cos(a), 0.035 * math.sin(a), 0.0],
                              ACTION_DIMS)
            return Action([0.0, 0.0, 1.0], ACTION_DIMS)

        scale = min(self.step, dist) / dist
        return Action([dx * scale, dy * scale, 0.0], ACTION_DIMS)
