"""L1 -- any LeRobot policy, behind our Policy protocol.

One adapter covers SmolVLA, pi0, pi0.5, GR00T and ACT, because LeRobot already
normalises them behind `select_action()`. That is the point of the L0 contract:
swapping the policy should be a config change, not a code change.

**This is what makes a real VLA rollout MINEABLE.** `lerobot-eval` stores a
success bit and an mp4; driving the policy through our loop stores full
per-step state including privileged simulator truth, which is what the mining
layer consumes.

Requires the LeRobot venv. Import is lazy.
"""
from __future__ import annotations

from ..schema import Observation, Action, derive_id

# LeRobot's LIBERO observation contract (docs: "Policy inputs and outputs").
# The `.images.*` prefix is enforced by LeRobot because the keys are baked into
# the normalisation statistics layer -- renaming them silently denormalises.
IMG_KEYS = ("observation.images.image", "observation.images.image2")
STATE_KEY = "observation.state"


class LeRobotPolicy:
    """Wraps a LeRobot policy checkpoint.

    `state_fn` builds the 8-dim LIBERO state vector (eef pos, axis-angle
    orientation, gripper qpos) from our Observation. It is injectable because a
    different benchmark has a different state layout, and getting it wrong is
    the silent-denormalisation failure mode the proposal warns about.
    """

    def __init__(self, checkpoint: str, device: str = "cuda",
                 n_action_steps: int | None = None, state_fn=None):
        self.checkpoint = checkpoint
        self.device = device
        self.n_action_steps = n_action_steps
        self.state_fn = state_fn or libero_state
        self._policy = None
        self._cfg = {}
        self.action_dims = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
        self.policy_id = derive_id(self.identity())

    def identity(self) -> dict:
        """Checkpoint revision and decoding config are load-bearing.

        `n_action_steps` changes the open-loop horizon and therefore behaviour:
        the published SmolVLA config ships 1 against a chunk size of 50, and
        LeRobot's own pi0.5 reproduction overrides it on the command line. A
        policy evaluated at a different horizon is a different policy, and must
        miss the cache.
        """
        return {"name": f"lerobot:{self.checkpoint}",
                "checkpoint": self.checkpoint,
                "checkpoint_revision": self._cfg.get("revision"),
                "n_action_steps": self.n_action_steps,
                "action_dims": list(self.action_dims),
                "state_fn": getattr(self.state_fn, "__name__", "custom")}

    def _load(self):
        import torch
        from lerobot.policies.factory import get_policy_class
        from lerobot.policies.pretrained import PreTrainedPolicy
        try:
            self._policy = PreTrainedPolicy.from_pretrained(self.checkpoint)
        except Exception:
            # older/newer LeRobot: resolve the concrete class from the config
            from lerobot.configs.policies import PreTrainedConfig
            cfg = PreTrainedConfig.from_pretrained(self.checkpoint)
            self._policy = get_policy_class(cfg.type).from_pretrained(self.checkpoint)
        if self.n_action_steps is not None:
            try:
                self._policy.config.n_action_steps = self.n_action_steps
            except Exception:
                pass
        self._policy.to(self.device).eval()
        self._cfg = {"revision": getattr(self._policy, "revision", None)}
        self.policy_id = derive_id(self.identity())

    def reset(self) -> None:
        if self._policy is None:
            self._load()
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def __call__(self, obs: Observation) -> Action:
        import torch
        import numpy as np
        if self._policy is None:
            self._load()
        batch = {}
        frames = obs.frames or {}
        for i, k in enumerate(IMG_KEYS):
            img = frames.get(f"cam{i}")
            if img is None:
                raise RuntimeError(
                    f"no live frame for {k}. A real VLA cannot be driven from "
                    f"`image_refs` alone -- the env must supply `frames`.")
            t = torch.from_numpy(np.ascontiguousarray(img)).to(self.device)
            batch[k] = t.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        st = np.asarray(self.state_fn(obs), dtype=np.float32)
        batch[STATE_KEY] = torch.from_numpy(st).unsqueeze(0).to(self.device)
        batch["task"] = [obs.instruction]
        with torch.inference_mode():
            a = self._policy.select_action(batch)
        return Action([float(x) for x in a.squeeze(0).cpu().numpy()],
                      self.action_dims)


def libero_state(obs: Observation):
    """LIBERO's 8-dim state: eef position, axis-angle orientation, gripper qpos."""
    import numpy as np
    pos = obs.get("eef_pos") or [0, 0, 0]
    quat = obs.get("eef_quat") or [0, 0, 0, 1]
    grip = obs.get("gripper_qpos") or [0, 0]
    # quaternion (x,y,z,w) -> axis-angle
    q = np.asarray(quat, dtype=np.float64)
    w = float(np.clip(q[3], -1.0, 1.0))
    ang = 2.0 * np.arccos(w)
    sn = np.sqrt(max(1e-12, 1.0 - w * w))
    axis = q[:3] / sn if sn > 1e-6 else np.zeros(3)
    return list(pos) + list(axis * ang) + list(grip)
