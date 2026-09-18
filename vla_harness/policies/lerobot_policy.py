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
# LeRobot's LIBERO camera mapping: agentview -> "image", wrist -> "image2".
# The batch key is the `.images.` form; the frame key is the bare name.
IMG_KEYS = ("observation.images.image", "observation.images.image2")
FRAME_KEYS = ("image", "image2")
STATE_KEY = "observation.state"


class LeRobotPolicy:
    """Wraps a LeRobot policy checkpoint.

    `state_fn` builds the 8-dim LIBERO state vector (eef pos, axis-angle
    orientation, gripper qpos) from our Observation. It is injectable because a
    different benchmark has a different state layout, and getting it wrong is
    the silent-denormalisation failure mode the proposal warns about.
    """

    def __init__(self, checkpoint: str, device: str = "cuda",
                 n_action_steps: int | None = None, state_fn=None,
                 env_cfg=None, policy_overrides: dict | None = None,
                 dtype: str | None = None, rename_map: dict | None = None):
        self.checkpoint = checkpoint
        self.device = device
        self.n_action_steps = n_action_steps
        # Documented eval settings that are not n_action_steps, e.g. MINERVA's
        # temporal_ensemble_coeff=0.01. Without this the harness could not
        # express a policy's published eval config, so a harness-vs-lerobot-eval
        # gap could come from a missing setting rather than a harness bug.
        self.policy_overrides = dict(policy_overrides or {})
        # dtype="bfloat16": load on CPU, cast EVERY parameter, then move to the
        # device. GR00T N1.7 LIBERO checkpoints are stored F32 (3.144B params) and
        # OOM an 8 GB card at 7.2 GiB on every stock path, including
        # model_params_fp32=false; this path fits in 5.87 GiB (RESULTS.md R-021).
        # A departure from the shipped numerics, so it is part of identity().
        self.dtype = dtype
        # Observation key renames applied by LeRobot's rename step, e.g. GR00T's
        # {"observation.images.image2": "observation.images.wrist_image"} -- the
        # same `--rename_map` lerobot-eval takes.
        self.rename_map = dict(rename_map or {})
        self.state_fn = state_fn or libero_state
        self.env_cfg = env_cfg
        self._pre = self._post = self._env_pre = self._env_post = None
        self._policy = None
        self._cfg = {}
        self.action_dims = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
        self.policy_id = derive_id(self.identity())
        self.model_forwards = 0

    def resolved_config(self) -> dict:
        """What the loaded policy ACTUALLY uses -- not what was requested.

        identity() hashes the requested settings into an opaque policy_id, so a
        trace could not be read back to its horizon: vla-81 tried to recover
        n_action_steps for the GR00T runs by reconstructing the hash over a grid
        and could not. Requests and defaults diverge silently (LeRobot's
        GrootConfig auto-migrates an N1.5-era 50 to 40; `--policy.pretrained_path`
        has been seen to drop n_action_steps back to a default), so the RESOLVED
        values are what a later reader needs.
        """
        cfg = getattr(self._policy, "config", None)
        out = {"requested_n_action_steps": self.n_action_steps,
               "dtype": self.dtype, "checkpoint": self.checkpoint,
               "policy_overrides": dict(sorted(self.policy_overrides.items()))}
        for k in ("n_action_steps", "chunk_size", "type", "device"):
            if cfg is not None and hasattr(cfg, k):
                out[k] = getattr(cfg, k)
        return out

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
                "policy_overrides": dict(sorted(self.policy_overrides.items())),
                "dtype": self.dtype,
                "rename_map": dict(sorted(self.rename_map.items())),
                "action_dims": list(self.action_dims),
                "state_fn": getattr(self.state_fn, "__name__", "custom")}

    def _load(self):
        """Load the policy AND LeRobot's own processor pipelines.

        Hand-rolling the batch does not work and should not be attempted:
        SmolVLA expects pre-tokenised language (`observation.language.tokens`),
        and the pre-processor is also where NORMALISATION lives. Building the
        batch ourselves would reproduce, silently, exactly the un-normalisation
        mismatch the proposal warns can "manufacture apparent model failures".

        Two stages, in this order (mirroring `lerobot_eval.py:289-291`):
            env_preprocessor   env-specific (LiberoProcessorStep)
            preprocessor       policy-specific (tokenise, normalise, device)
        and the inverse on the action side.
        """
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class
        from lerobot.policies import make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(self.checkpoint)
        if self.n_action_steps is not None:
            cfg.n_action_steps = self.n_action_steps
        for key, val in self.policy_overrides.items():
            if not hasattr(cfg, key):
                raise ValueError(f"{cfg.type} config has no field {key!r}; refusing "
                                 f"to silently ignore a policy override")
            setattr(cfg, key, val)
        cfg.pretrained_path = self.checkpoint
        cast = None
        if self.dtype:
            import torch
            cast = getattr(torch, self.dtype)
            # from_pretrained places the model on cfg.device; load on CPU so the
            # cast happens before the full-precision weights ever reach the GPU.
            cfg.device = "cpu"
        # config=cfg, as lerobot-eval's make_policy does. Some settings are read
        # at CONSTRUCTION (tinyflow builds its temporal ensembler in __init__),
        # so setting them on the loaded policy afterwards would do nothing.
        self._policy = get_policy_class(cfg.type).from_pretrained(self.checkpoint,
                                                                  config=cfg)
        if cast is not None:
            self._policy = self._policy.to(cast)
            cfg.device = str(self.device)
            try:
                self._policy.config.device = str(self.device)
            except Exception:
                pass
        if self.n_action_steps is not None:
            try:
                self._policy.config.n_action_steps = self.n_action_steps
            except Exception:
                pass
        self._policy.to(self.device).eval()

        self._pre, self._post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=self.checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
                "rename_observations_processor": {"rename_map": dict(self.rename_map)},
            })
        self._env_pre = self._env_post = None
        if self.env_cfg is not None:
            from lerobot.envs.factory import make_env_pre_post_processors
            self._env_pre, self._env_post = make_env_pre_post_processors(
                env_cfg=self.env_cfg, policy_cfg=cfg)
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
        # Hand LeRobot the NESTED robot_state and let LiberoProcessorStep build
        # `observation.state` itself. It concatenates eef_pos + axis-angle +
        # gripper_qpos, and does the quaternion conversion with ITS convention.
        # Computing it ourselves duplicated that conversion and risked a
        # silent mismatch -- the same class of error as un-normalisation.
        #
        # It also FLIPS both image axes (`torch.flip(img, dims=[2,3])`): LIBERO
        # renders in an orientation the policy does not expect. Skipping this
        # stage feeds the policy upside-down images and scores ~0%, which is
        # exactly what happened before `env_cfg` was wired through.
        batch = {}
        frames = obs.frames or {}
        for i, k in enumerate(IMG_KEYS):
            img = frames.get(FRAME_KEYS[i], frames.get(f"cam{i}"))
            if img is None:
                raise RuntimeError(
                    f"no live frame for {k} (looked for {FRAME_KEYS[i]!r}); "
                    f"env supplied {sorted(frames)}. A real VLA cannot be "
                    f"driven from `image_refs` alone.")
            import numpy as _np
            if not isinstance(img, _np.ndarray):
                raise TypeError(
                    f"frame {FRAME_KEYS[i]!r} is {type(img).__name__}, not an "
                    f"array -- the env is passing keys, not pixels.")
            t = torch.from_numpy(np.ascontiguousarray(img)).to(self.device)
            batch[k] = t.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if self._env_pre is not None:
            def T(v):
                return torch.as_tensor(np.asarray(v, dtype=np.float32)
                                       ).unsqueeze(0).to(self.device)
            batch["observation.robot_state"] = {
                "eef": {"pos": T(obs.get("eef_pos")),
                        "quat": T(obs.get("eef_quat"))},
                "gripper": {"qpos": T(obs.get("gripper_qpos"))},
            }
        else:
            st = np.asarray(self.state_fn(obs), dtype=np.float32)
            batch[STATE_KEY] = torch.from_numpy(st).unsqueeze(0).to(self.device)
        batch["task"] = [obs.instruction]

        # A chunked policy only runs the model when its action queue is empty;
        # every other call pops a cached action. Counting calls therefore
        # overstates model invocations by up to n_action_steps (vla-81,
        # 2026-09-18). A policy with no queue runs the model every call, and
        # `q is None` gives exactly that.
        q = getattr(self._policy, "_action_queue", None)
        ran_model = q is None or len(q) == 0

        # LeRobot's own pipelines -- tokenisation and normalisation live here.
        if self._env_pre is not None:
            batch = self._env_pre(batch)
        batch = self._pre(batch)
        with torch.inference_mode():
            a = self._policy.select_action(batch)
        if ran_model:
            self.model_forwards += 1
        if self._post is not None:
            a = self._post(a)
        if self._env_post is not None:
            from lerobot.utils.constants import ACTION
            tr = self._env_post({ACTION: a})
            a = tr[ACTION]
        arr = a.squeeze(0).detach().cpu().numpy() if hasattr(a, "detach") else a
        return Action([float(x) for x in arr], self.action_dims)


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
