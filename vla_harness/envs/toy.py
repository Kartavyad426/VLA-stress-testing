"""L2 -- a toy reaching/grasping env with LIBERO-plus-shaped knobs.

Stands in for LIBERO while the harness is being validated. It exposes the same
Env protocol and the same knob names, so swapping in the real simulator is a
one-file change (G6, G4).
"""
from __future__ import annotations

import math
import random

from ..schema import Observation, Action, PerturbationSpec, derive_id

ACTION_DIMS = ["dx", "dy", "gripper"]


class ToyReachEnv:
    """Reach the target object and close the gripper on it.

    The knobs perturb the *observation* and the *initial state*. The task, the
    goal predicate and the object's world position are never touched -- which is
    what makes a success-rate drop attributable to the perturbation.
    """

    action_dims = ACTION_DIMS

    def __init__(self, task_id="pick_bowl", max_steps=40, grasp_radius=0.06):
        self.task_id = task_id
        self.max_steps = max_steps
        self.grasp_radius = grasp_radius
        self.instruction = "pick up the black bowl"
        self.env_id = derive_id(self.identity())

    def identity(self) -> dict:
        """Everything that changes what this env DOES.

        `grasp_radius` is the worked example: two ToyReachEnvs differing only in
        it are different tasks -- one is unsolvable -- yet both were "toy-reach-v1"
        before this existed, so a cached rollout from one was served for the
        other. Anything omitted here becomes that bug again.

        Note `grasp_radius` is a TASK parameter (it decides success), not a
        mining-layer detector threshold. Detector config is not covered by this
        fingerprint -- see the scope note on the Env protocol.
        """
        return {"name": "toy-reach-v1", "task_id": self.task_id,
                "max_steps": self.max_steps, "grasp_radius": self.grasp_radius,
                "action_dims": list(ACTION_DIMS), "control_mode": "delta"}

    def reset(self, seed: int, spec: PerturbationSpec) -> Observation:
        self.rng = random.Random(seed)             # G5: determinism
        self.spec = spec
        k = spec.as_dict()

        # Nominal object placement varies a little per seed -- this is the
        # stochasticity that makes N episodes per cell necessary.
        base_r, base_th = 0.70, math.radians(30)
        jitter = self.rng.uniform(-0.03, 0.03)
        self.object_xy = (base_r * math.cos(base_th) + jitter,
                          base_r * math.sin(base_th) + jitter)

        # object_layout knob: shift the target in the world
        shift = k.get("object_shift_m", 0.0)
        self.object_xy = (self.object_xy[0] + shift, self.object_xy[1])

        # object_layout knob: distractors the policy may confuse for the target
        self.distractors = []
        for i in range(int(k.get("distractor_count", 0))):
            a = base_th + math.radians(18 * (i + 1))
            # closer to the gripper's start than the target is: a policy that
            # picks "nearest blob" instead of the named object goes to these
            rr = base_r * (0.72 - 0.06 * i)
            self.distractors.append((rr * math.cos(a), rr * math.sin(a)))

        # initial_state knob: start the end-effector somewhere else
        self.ee_xy = (k.get("ee_offset_x_m", 0.0), k.get("ee_offset_y_m", 0.0))
        self.camera_yaw = math.radians(k.get("camera_yaw_deg", 0.0))
        self.noise = k.get("pixel_noise_std", 0.0)

        self.gripper, self.holding, self.t = 0.0, False, 0
        self.grasp_attempts = 0
        return self._obs()

    # --- rendering: project the world into the camera frame -----------------
    def _to_cam(self, p):
        a = -self.camera_yaw
        c, s = math.cos(a), math.sin(a)
        x, y = p
        px, py = c * x - s * y, s * x + c * y
        if self.noise:
            px += self.rng.gauss(0, self.noise)
            py += self.rng.gauss(0, self.noise)
        return (px, py)

    def _obs(self) -> Observation:
        return Observation(
            instruction=self.instruction,
            t=self.t,
            state={
                # what a camera would show (the "pixels")
                "obj_cam_xy": self._to_cam(self.object_xy),
                "distractors_cam_xy": [self._to_cam(d) for d in self.distractors],
                # proprioception -- always available on a real robot
                "ee_xy": self.ee_xy,
                "gripper": self.gripper,
                "holding": self.holding,
                # camera extrinsics: a real robot knows these from calibration,
                # so this is legitimate policy input, not privileged state.
                "_env_camera_yaw_deg": math.degrees(self.camera_yaw),
                # privileged simulator state: for DETECTORS ONLY, never the policy.
                # LIBERO exposes the equivalent (object poses, contact flags).
                "_gt_obj_xy": self.object_xy,
                "_gt_ee_to_obj": math.dist(self.ee_xy, self.object_xy),
                # the robot's own action history: legitimately its own
                "grasp_attempts": self.grasp_attempts,
                # "did it go to the wrong object" is a first-class signal, not
                # something to infer from error magnitude. LIBERO exposes the
                # equivalent via per-object poses.
                "_gt_dist_to_distractor": (
                    min((math.dist(self.ee_xy, d) for d in self.distractors),
                        default=float("inf"))),
            },
        )

    def step(self, action: Action) -> tuple[Observation, bool, bool, str]:
        a = action.as_dict()
        self.t += 1
        self.ee_xy = (self.ee_xy[0] + a["dx"], self.ee_xy[1] + a["dy"])
        self.gripper = a["gripper"]

        if a["gripper"] > 0.5 and not self.holding:
            self.grasp_attempts += 1
            d = math.dist(self.ee_xy, self.object_xy)
            if d < self.grasp_radius:
                self.holding = True
                return self._obs(), True, True, "grasped"

        done = self.t >= self.max_steps
        return self._obs(), False, done, ("timeout" if done else "")
