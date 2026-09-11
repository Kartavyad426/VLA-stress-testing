"""
A policy is a function: observation -> action.

That is the whole concept. This file makes it concrete with a toy 2-D
reaching-and-grasping environment you can actually run, plus a scripted
policy that solves it. Then it breaks that policy with a camera-viewpoint
perturbation, which is the exact failure mode the VLA Stress-Test project
is built to find.

No dependencies. Run:  python3 demo/policy_demo.py
"""

import math
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# Observation and action: the two ends of the policy function
# ----------------------------------------------------------------------------

@dataclass
class Observation:
    """What the robot can see and feel at one timestep.

    A real VLA gets `image` as a 224x224 RGB array from a wrist and/or third-
    person camera. Here it is the object's position *as measured in the camera
    frame* -- which is the part that matters, because that is what changes when
    the camera moves and the world does not.
    """
    instruction: str          # natural language, constant for the episode
    image_xy: tuple           # object position in CAMERA frame  (the "pixels")
    ee_xy: tuple              # end-effector position, world frame (proprioception)
    gripper: float            # 0.0 open, 1.0 closed
    holding: bool


@dataclass
class Action:
    """What the policy emits. LIBERO uses 7 dims; we use 3.

    Real: (dx, dy, dz, droll, dpitch, dyaw, gripper) at ~10-20 Hz.
    Here: (dx, dy, gripper) -- same shape of thing, fewer axes.
    """
    dx: float
    dy: float
    gripper: float            # target gripper state, 0.0 open / 1.0 closed

    def __repr__(self):
        return f"({self.dx:+.3f}, {self.dy:+.3f}, grip={self.gripper:.0f})"


# ----------------------------------------------------------------------------
# THE POLICY
# ----------------------------------------------------------------------------

class ScriptedReachPolicy:
    """A hand-written policy. Deterministic, inspectable, and solves the task.

    A VLA is this same function signature with ~7 billion learned parameters
    instead of twelve lines of arithmetic. The interface is identical:

        action = policy(observation)

    That interchangeability is the point. The stress-test harness never needs
    to know which one it is holding.

    `cam_to_world` is this policy's baked-in belief about how the camera relates
    to the world. A scripted policy states it as a number. A VLA absorbs the
    same assumption implicitly from its training demonstrations -- which is
    precisely why it breaks when the camera moves.
    """

    name = "scripted-reach-v1"

    def __init__(self, cam_to_world_yaw_deg=0.0, step=0.15, grasp_tol=0.05):
        self.cam_to_world_yaw = math.radians(cam_to_world_yaw_deg)
        self.step = step
        self.grasp_tol = grasp_tol

    def _perceive(self, obs):
        """Camera frame -> world frame. The policy's learned spatial prior."""
        cx, cy = obs.image_xy
        c, s = math.cos(self.cam_to_world_yaw), math.sin(self.cam_to_world_yaw)
        return (c * cx - s * cy, s * cx + c * cy)

    def __call__(self, obs):
        if obs.holding:                                   # phase: transport
            return Action(0.0, 0.0, 1.0)

        tx, ty = self._perceive(obs)                      # where I think it is
        ex, ey = obs.ee_xy
        dx, dy = tx - ex, ty - ey
        dist = math.hypot(dx, dy)

        if dist < self.grasp_tol:                         # phase: grasp
            return Action(0.0, 0.0, 1.0)

        scale = min(self.step, dist) / dist               # phase: approach
        return Action(dx * scale, dy * scale, 0.0)


class VLAPolicyStub:
    """The same interface, backed by a real VLA. Shown, not run -- OpenVLA-OFT
    is a 7B model and needs a GPU with far more than the 8 GB on this machine.

        from experiments.robot.openvla_utils import get_vla, get_vla_action

        vla = get_vla(cfg)                       # load checkpoint, freeze it

        def policy(obs):
            return get_vla_action(
                vla, processor, base_vla_name,
                obs["full_image"],               # 224x224 RGB from the sim
                obs["task_description"],         # the language instruction
                unnorm_key="libero_spatial",     # de-normalize to real units
            )                                    # -> 7-D action chunk

    Note `unnorm_key`. Action values come out of the model normalized; the
    wrong un-normalization key produces confident, plausible, completely wrong
    motion. The project doc flags this specifically: environment mismatch
    "can manufacture apparent model failures."
    """
    name = "openvla-oft (not loaded)"


# ----------------------------------------------------------------------------
# THE ENVIRONMENT -- and the perturbation
# ----------------------------------------------------------------------------

@dataclass
class ReachEnv:
    """Reach the object and close the gripper on it.

    `camera_yaw_deg` is the perturbation. It rotates the CAMERA. The object
    never moves in the world. Nothing about the task changes. Only the pixels.
    """
    object_xy: tuple = (0.60, 0.35)
    camera_yaw_deg: float = 0.0
    max_steps: int = 30
    instruction: str = "pick up the black bowl"

    ee_xy: tuple = field(default=(0.0, 0.0), init=False)
    gripper: float = field(default=0.0, init=False)
    holding: bool = field(default=False, init=False)
    t: int = field(default=0, init=False)

    def reset(self):
        self.ee_xy, self.gripper, self.holding, self.t = (0.0, 0.0), 0.0, False, 0
        return self._obs()

    def _obs(self):
        """Project the world into the camera frame. This is 'rendering'."""
        ox, oy = self.object_xy
        a = math.radians(-self.camera_yaw_deg)            # world -> camera
        c, s = math.cos(a), math.sin(a)
        return Observation(
            instruction=self.instruction,
            image_xy=(c * ox - s * oy, s * ox + c * oy),
            ee_xy=self.ee_xy,
            gripper=self.gripper,
            holding=self.holding,
        )

    def step(self, action):
        self.t += 1
        self.ee_xy = (self.ee_xy[0] + action.dx, self.ee_xy[1] + action.dy)
        self.gripper = action.gripper

        # Grasp succeeds only if the gripper closes while actually on the object
        if action.gripper > 0.5 and not self.holding:
            d = math.dist(self.ee_xy, self.object_xy)
            if d < 0.06:
                self.holding = True
            else:
                return self._obs(), False, True, f"closed on empty air, {d*100:.1f} cm off"

        done = self.holding or self.t >= self.max_steps
        reason = "grasped" if self.holding else ("step limit" if done else "")
        return self._obs(), self.holding, done, reason


# ----------------------------------------------------------------------------
# THE ROLLOUT LOOP -- observe, act, repeat. This is all "running a policy" is.
# ----------------------------------------------------------------------------

def rollout(env, policy, verbose=True):
    obs = env.reset()
    trace = []
    if verbose:
        print(f'    instruction: "{obs.instruction}"')
        print(f"    {'t':>3}  {'ee (world)':>16}  {'sees obj at':>16}  action")

    while True:
        action = policy(obs)                              # <-- THE POLICY CALL
        trace.append((env.t, obs.ee_xy, obs.image_xy, action))
        if verbose and (env.t < 4 or env.t % 4 == 0):
            print(f"    {env.t:>3}  ({obs.ee_xy[0]:+.3f},{obs.ee_xy[1]:+.3f})  "
                  f"({obs.image_xy[0]:+.3f},{obs.image_xy[1]:+.3f})  {action}")
        obs, success, done, reason = env.step(action)
        if done:
            return success, reason, trace, env.ee_xy


def report(title, env, policy):
    print(f"\n  {title}")
    print("  " + "-" * 68)
    success, reason, trace, final_ee = rollout(env, policy)
    verdict = "SUCCESS" if success else "FAILURE"
    err = math.dist(final_ee, env.object_xy)
    print(f"    -> {verdict}  ({reason})  after {len(trace)} steps, "
          f"final error {err*100:.1f} cm")
    return success


if __name__ == "__main__":
    OBJ = (0.60, 0.35)

    print("=" * 72)
    print("  A POLICY IS A FUNCTION: observation -> action")
    print("=" * 72)

    policy = ScriptedReachPolicy(cam_to_world_yaw_deg=0.0)
    print(f"\n  policy: {policy.name}")
    print(f"  it believes the camera sits at yaw 0 deg, because every")
    print(f"  demonstration it learned from was recorded at yaw 0 deg.")

    report("NOMINAL  -- camera at 0 deg, as trained",
           ReachEnv(object_xy=OBJ, camera_yaw_deg=0.0), policy)

    print("\n" + "=" * 72)
    print("  NOW PERTURB THE CAMERA. THE OBJECT DOES NOT MOVE.")
    print("=" * 72)

    for yaw in (5, 15, 30):
        report(f"PERTURBED -- camera yaw +{yaw} deg  (object still at {OBJ})",
               ReachEnv(object_xy=OBJ, camera_yaw_deg=yaw), policy)

    print("\n" + "=" * 72)
    print("  ROBUSTNESS CURVE -- sweep the perturbation, measure the envelope")
    print("=" * 72 + "\n")

    for yaw in range(0, 33, 3):
        ok = rollout(ReachEnv(object_xy=OBJ, camera_yaw_deg=yaw), policy,
                     verbose=False)[0]
        bar = "#" * (28 if ok else 2)
        print(f"    yaw +{yaw:>2} deg  |{bar:<28}| {'success' if ok else 'FAIL'}")

    print("""
  ------------------------------------------------------------------------
  What just happened, and why it matters for the project:

  The task never changed. The object never moved. The instruction never
  changed. The policy is deterministic and has no noise in it. The ONLY
  thing that changed was where the camera was standing -- and the policy
  went from solving the task to closing its gripper on empty air.

  That is a visual grounding failure, and it is invisible to a benchmark
  score. Aggregate success on the nominal task is 100%. The operating
  envelope is roughly +/- 6 degrees of camera yaw. One of those two numbers
  tells you whether to deploy this robot.

  This toy fails sharply because the flaw is a single explicit line
  (`cam_to_world_yaw`). A 7B VLA holds the same assumption diffusely across
  billions of weights, so it degrades over a band rather than a cliff --
  which is exactly why the real boundary has to be MEASURED, with repeated
  episodes and confidence intervals, rather than reasoned about.
  ------------------------------------------------------------------------
""")
