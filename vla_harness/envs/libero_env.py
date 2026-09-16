"""L2 -- LIBERO adapter.

Wraps LeRobot's `LiberoEnv` behind our Env protocol. Nothing above L2 changes:
the miner, the manifest and the runner see the same `Rollout` they see from the
toy. That is the whole point of the L0 contract.

Deliberately thin. The one place it is NOT thin is `scene_descriptor()`, which
must read object and camera poses out of MuJoCo in world coordinates -- the
toy's version is three lines because the toy has no simulator.

Requires the LeRobot venv (.venvs/lerobot). Import is lazy so the rest of the
harness stays stdlib-only and testable without MuJoCo.
"""
from __future__ import annotations

import math
from typing import Any

from ..schema import (Observation, Action, PerturbationSpec, derive_id,
                      semantic_runtime)


def _resolve_body(name: str, bodies) -> str | None:
    """Map a BDDL `obj_of_interest` entry onto a MuJoCo body.

    Two shapes occur and only the first is a plain prefix match:

      free object   'cream_cheese_1'               -> 'cream_cheese_1_main'
      REGION on a   'wooden_cabinet_1_middle_region'
      fixture        -> bodies are 'wooden_cabinet_1_cabinet_middle', '..._main'

    The second broke the original matcher and silently cost us six of eight
    LIBERO-Goal tasks: every articulated task ("open the middle drawer",
    "turn on the stove") names a REGION, no body starts with it, the object
    list came back empty, and the miner correctly abstained on 200 traces.

    Region entries are `<fixture>_<part>_region`, so: strip `_region`, then
    prefer the body carrying BOTH the fixture prefix and the part token --
    `..._cabinet_middle` over `..._main` -- because the part is what the task
    acts on. Fall back to progressively shorter prefixes.
    """
    exact = [b for b in bodies if b and b.startswith(name)]
    if exact:
        return sorted(exact, key=len)[0]

    stem = name[:-len("_region")] if name.endswith("_region") else name
    toks = stem.split("_")
    # longest prefix that any body shares, then disambiguate by the remainder
    for cut in range(len(toks) - 1, 0, -1):
        prefix = "_".join(toks[:cut])
        cand = [b for b in bodies if b and b.startswith(prefix)]
        if not cand:
            continue
        rest = toks[cut:]
        if rest:
            part = [b for b in cand if all(t in b for t in rest)]
            if part:
                return sorted(part, key=len)[0]
        return sorted(cand, key=len)[0]
    return None


def _look_at_quat(eye, target):
    """Quaternion (w,x,y,z) aiming a MuJoCo camera from `eye` at `target`."""
    import numpy as np
    f = np.array(target) - np.array(eye)
    n = np.linalg.norm(f)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    f = f / n
    z = -f                                   # MuJoCo cameras look down -z
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.stack([x, y, z], axis=1)
    t = np.trace(m)
    if t > 0:
        s_ = math.sqrt(t + 1.0) * 2
        return np.array([0.25 * s_, (m[2,1]-m[1,2])/s_, (m[0,2]-m[2,0])/s_,
                         (m[1,0]-m[0,1])/s_])
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s_ = math.sqrt(1.0+m[0,0]-m[1,1]-m[2,2])*2
        return np.array([(m[2,1]-m[1,2])/s_, 0.25*s_, (m[0,1]+m[1,0])/s_, (m[0,2]+m[2,0])/s_])
    if i == 1:
        s_ = math.sqrt(1.0+m[1,1]-m[0,0]-m[2,2])*2
        return np.array([(m[0,2]-m[2,0])/s_, (m[0,1]+m[1,0])/s_, 0.25*s_, (m[1,2]+m[2,1])/s_])
    s_ = math.sqrt(1.0+m[2,2]-m[0,0]-m[1,1])*2
    return np.array([(m[1,0]-m[0,1])/s_, (m[0,2]+m[2,0])/s_, (m[1,2]+m[2,1])/s_, 0.25*s_])

# LIBERO: 6-D end-effector delta + gripper, Box(-1, 1, (7,))
ACTION_DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

# Published step caps per suite (LeRobot LIBERO docs). Part of the task
# definition -- "do it within N steps" -- so they belong in identity().
MAX_STEPS = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300,
             "libero_90": 400, "libero_10": 520}

# Knobs this adapter can actually apply. Vanilla LIBERO has no perturbation
# machinery; LIBERO-plus supplies it. Declaring the supported set explicitly
# means an unsupported knob fails loudly instead of being silently ignored --
# which would produce a "robustness result" for a perturbation that never
# happened.
# Applied directly through MuJoCo after reset. LIBERO-plus is not required for
# these: camera extrinsics and the robot's initial configuration are simulator
# state we can set ourselves. That matters because it unblocks sweeps and
# counterfactual attribution without the second container LIBERO-plus needs
# (it uninstalls vanilla LIBERO).
SUPPORTED_KNOBS: set[str] = {
    "camera_yaw_deg", "camera_pitch_deg", "camera_dist_m",
    "ee_offset_x_m", "ee_offset_y_m", "light_intensity",
}
MAIN_CAMERA = "agentview"

# LE-5: termination vocabulary is declared in L0 and adapters map onto it.
# ARCHITECTURE §4 documents it and the manifest prints it, so divergence
# between adapters ("grasped" vs "success") would surface in a client artifact.
from ..schema import TERM_SUCCESS, TERM_TIMEOUT  # noqa: E402


class LiberoEnv:
    """One LIBERO task, wrapped.

    `task_id` indexes the suite. One env instance == one task, matching the
    per-task environment control in PLAN.md §7c discriminator 1 (F2's lesson:
    at suite level a single-task collapse dilutes into noise).
    """

    action_dims = ACTION_DIMS

    def __init__(self, suite: str = "libero_spatial", task_id: int = 0,
                 control_mode: str = "relative", init_states: bool = True,
                 hard_reset: bool = True, num_steps_wait: int = 10,
                 max_steps: int | None = None, image_dir: str | None = None):
        if suite not in MAX_STEPS:
            raise ValueError(f"unknown suite {suite!r}; expected {sorted(MAX_STEPS)}")
        self.suite = suite
        self.task_id_idx = task_id
        self.control_mode = control_mode
        self.init_states = init_states
        self.hard_reset = hard_reset
        self.num_steps_wait = num_steps_wait
        self.max_steps = max_steps or MAX_STEPS[suite]
        self.image_dir = image_dir
        self._env = None
        self.instruction = ""
        self.task_id = f"{suite}/task{task_id}"
        self.env_id = derive_id(self.identity())

    # --- identity ---------------------------------------------------------
    def identity(self) -> dict:
        """Everything that changes what this env DOES.

        `control_mode` is here because LeRobot's docs warn it must match the
        checkpoint's action parameterisation -- a mismatch produces confident,
        plausible, completely wrong motion, i.e. a fake robustness result.
        `init_states` and `hard_reset` are here because soft resets are not
        bit-identical and change camera observations.
        """
        return {"name": f"libero:{self.suite}", "suite": self.suite,
                "task_id": self.task_id_idx, "control_mode": self.control_mode,
                "use_init_states": self.init_states,
                # LE-1: WHICH init states, not merely whether to use them. F2
                # was stored states behaving differently under a new MuJoCo;
                # the converse -- the files changing under a FIXED MuJoCo --
                # is invisible to semantic_deps(), because MuJoCo did not move.
                "init_states_sha": self._init_states_sha(),
                "bddl_sha": self._bddl_sha(),
                "hard_reset": self.hard_reset,
                "num_steps_wait": self.num_steps_wait,
                "max_steps": self.max_steps, "action_dims": list(ACTION_DIMS)}

    def _init_states_sha(self) -> str | None:
        """Content hash of this task's stored initial states."""
        try:
            import hashlib, numpy as np
            from lerobot.envs.libero import get_task_init_states, _get_suite
            arr = get_task_init_states(_get_suite(self.suite), self.task_id_idx)
            return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]
        except Exception:
            return None

    def _bddl_sha(self) -> str | None:
        """Content hash of the BDDL file -- the task definition itself."""
        try:
            import hashlib
            from lerobot.envs.libero import _get_suite
            f = _get_suite(self.suite).get_task(self.task_id_idx).bddl_file
            from libero.libero import get_libero_path
            import os
            for root in (get_libero_path("bddl_files"),):
                for dp, _, fns in os.walk(root):
                    if os.path.basename(f) in fns:
                        with open(os.path.join(dp, os.path.basename(f)), "rb") as fh:
                            return hashlib.sha1(fh.read()).hexdigest()[:12]
        except Exception:
            pass
        return None

    def semantic_deps(self) -> dict:
        """KNOWN-semantic externals -- KEYED (A2/DG-4).

        `mujoco` is here on documented evidence: 3.4.0's box-box collision fix
        broke LIBERO's stored init states, taking SmolVLA 80% -> 28% on
        libero_spatial task 5 (FINDINGS.md F2, lerobot#4390). `MUJOCO_GL` is
        here because the rendering backend changes the pixels the policy sees.
        """
        return semantic_runtime()

    # --- lifecycle --------------------------------------------------------
    def _build(self):
        from lerobot.envs.libero import LiberoEnv as _LeRobotLibero, _get_suite
        suite = _get_suite(self.suite)
        self._env = _LeRobotLibero(
            task_suite=suite, task_id=self.task_id_idx,
            task_suite_name=self.suite,
            obs_type="pixels_agent_pos", control_mode=self.control_mode,
            init_states=self.init_states, hard_reset=self.hard_reset,
            num_steps_wait=self.num_steps_wait,
            episode_length=self.max_steps,
        )
        task = suite.get_task(self.task_id_idx)
        self.base_instruction = task.language
        self.instruction = task.language

    def reset(self, seed: int, spec: PerturbationSpec) -> Observation:
        unsupported = {k for k, v in spec.knobs if v} - SUPPORTED_KNOBS
        if unsupported:
            raise NotImplementedError(
                f"vanilla LIBERO cannot apply {sorted(unsupported)}. Use the "
                f"LIBERO-plus adapter -- silently ignoring a knob would produce "
                f"a robustness curve for a perturbation that never happened.")
        if self._env is None:
            self._build()
        self.spec = spec
        self.t = 0
        raw, _ = self._env.reset(seed=seed)
        self._raw = raw
        if any(v for _, v in spec.knobs):
            self._apply_perturbation(spec)
            raw = self._resettle()
            self._raw = raw
        # Phase G: the language probe substitutes or blanks the instruction.
        # Recorded in identity() is the BASE task; the override is per-rollout
        # and appears in the trace via Observation.instruction.
        ov = getattr(self, "instruction_override", None)
        self.instruction = self.base_instruction if ov is None else ov
        return self._obs()

    def step(self, action: Action) -> tuple[Observation, bool, bool, str]:
        import numpy as np
        self.t += 1
        # LeRobot's LiberoEnv.step asserts action.ndim == 1, so it must be an
        # array, not a list.
        raw, _reward, _terminated, truncated, info = self._env.step(
            np.asarray(action.values, dtype=np.float32))
        self._raw = raw

        # LE-3: success is the ONLY trustworthy signal and the env owns it (I3).
        # An earlier version defaulted to `terminated` when `is_success` was
        # absent -- i.e. treated "the episode ended" as "the task succeeded".
        # That fails silently and in the worst direction: inflated success, a
        # healthy-looking nominal cell, and §6's "nominal below ~90% is a
        # harness bug" heuristic never fires because the number is too HIGH.
        # A missing success signal is a broken adapter, not a default.
        if "is_success" not in info:
            raise RuntimeError(
                "LIBERO env returned no `is_success`. Refusing to infer success "
                "from termination -- that would silently inflate every rate.")
        success = bool(info["is_success"])
        done = success or bool(truncated) or self.t >= self.max_steps
        reason = (TERM_SUCCESS if success
                  else (TERM_TIMEOUT if done else ""))
        return self._obs(), success, done, reason

    # --- perturbation (applied post-reset, in simulator state) ------------
    def _apply_perturbation(self, spec: PerturbationSpec) -> None:
        """Set camera extrinsics / robot offset / lighting directly in MuJoCo.

        Recorded in `scene_descriptor` in physical units, so the perturbed scene
        stays externally reproducible -- a perturbation you cannot describe in
        metres is one nobody can reproduce on a bench.
        """
        import numpy as np
        k = spec.as_dict()
        sim = self._sim()

        yaw, pitch = k.get("camera_yaw_deg", 0.0), k.get("camera_pitch_deg", 0.0)
        dist = k.get("camera_dist_m", 0.0)
        if yaw or pitch or dist:
            cid = sim.model.camera_name2id(MAIN_CAMERA)
            pos = np.array(sim.model.cam_pos[cid], dtype=float)
            look = np.array([0.0, 0.0, pos[2] * 0.5])      # approx table centre
            rel = pos - look
            if yaw:
                a = math.radians(yaw); c, s_ = math.cos(a), math.sin(a)
                rel = np.array([c * rel[0] - s_ * rel[1],
                                s_ * rel[0] + c * rel[1], rel[2]])
            if pitch:
                a = math.radians(pitch)
                r = math.hypot(rel[0], rel[1])
                z = rel[2] * math.cos(a) + r * math.sin(a)
                scale = (r * math.cos(a) - rel[2] * math.sin(a)) / max(r, 1e-9)
                rel = np.array([rel[0] * scale, rel[1] * scale, z])
            if dist:
                n = np.linalg.norm(rel)
                rel = rel * (1.0 + dist / max(n, 1e-9))
            sim.model.cam_pos[cid] = look + rel
            # re-aim at the look-point so the target stays framed; a camera that
            # rotates AND loses the scene confounds viewpoint with occlusion.
            sim.model.cam_quat[cid] = _look_at_quat(look + rel, look)

        dx, dy = k.get("ee_offset_x_m", 0.0), k.get("ee_offset_y_m", 0.0)
        if dx or dy:
            # Nudge the arm's first two joints; a small cartesian offset at the
            # eef without solving IK. Magnitude is approximate by design and the
            # ACHIEVED offset is what scene_descriptor records.
            qpos = sim.data.qpos
            qpos[0] += dy * 2.0
            qpos[1] += dx * 2.0
            sim.forward()

        li = k.get("light_intensity", 0.0)
        if li and sim.model.nlight:
            for i in range(sim.model.nlight):
                sim.model.light_diffuse[i] = np.clip(
                    np.array(sim.model.light_diffuse[i]) * (1.0 + li), 0, 1)

    def _resettle(self):
        """Step the sim so the perturbation takes effect in the rendering."""
        from lerobot.envs.libero import get_libero_dummy_action
        inner = self._env.unwrapped._env
        raw = None
        for _ in range(3):
            raw, _, _, _ = inner.step(get_libero_dummy_action())
        return self._env.unwrapped._to_lerobot_obs(raw) if hasattr(
            self._env.unwrapped, "_to_lerobot_obs") else self._raw

    # --- observation ------------------------------------------------------
    def _sim(self):
        return self._env.unwrapped._env.env.sim

    def _obs(self) -> Observation:
        rs = self._raw["robot_state"]
        eef, grip, joints = rs["eef"], rs["gripper"], rs["joints"]
        state: dict[str, Any] = {
            # policy-visible proprioception (LeRobot's observation.state is
            # eef pos + axis-angle + gripper qpos, 8-dim)
            "eef_pos": list(map(float, eef["pos"])),
            "eef_quat": list(map(float, eef["quat"])),
            "gripper_qpos": list(map(float, grip["qpos"])),
            "gripper_qvel": list(map(float, grip["qvel"])),
            "joint_pos": list(map(float, joints["pos"])),
            "joint_vel": list(map(float, joints["vel"])),
        }
        state.update(self._privileged())
        # LeRobot returns `pixels` as a DICT keyed by its own camera mapping
        # ("image" = agentview, "image2" = wrist), not a list. Enumerating it
        # yields the KEYS, which is how the first version handed the policy
        # strings instead of arrays.
        px = self._raw.get("pixels", {})
        frames = dict(px) if isinstance(px, dict) else {
            f"cam{i}": im for i, im in enumerate(px)}
        return Observation(instruction=self.instruction, t=self.t, state=state,
                           image_refs=self._write_images(), frames=frames)

    def _privileged(self) -> dict:
        """`_gt_*` -- simulator truth, DETECTORS ONLY.

        `policy_view()` strips these before any policy sees them (enforced, not
        conventional). A policy reading one would produce a success rate that
        collapses on hardware.
        """
        try:
            sim = self._sim()
        except Exception as e:
            return {"_gt_unavailable": f"{type(e).__name__}: {e}"}
        out: dict[str, Any] = {}
        eef = self._raw["robot_state"]["eef"]["pos"]
        names, complete = self._object_body_names()
        objs = {}
        for name in names:
            try:
                bid = sim.model.body_name2id(name)
                p = [float(x) for x in sim.data.body_xpos[bid]]
                objs[name] = p
            except Exception:
                continue
        out["_gt_object_pos"] = objs
        # LE-2: detectors must be able to SKIP rather than compute a distance
        # over a subset of the task objects (G8).
        out["_gt_object_pos_complete"] = bool(complete and objs)
        if objs and complete:
            d = {n: math.dist(eef, p) for n, p in objs.items()}
            nearest = min(d, key=d.get)
            out["_gt_eef_to_nearest_object"] = d[nearest]
            out["_gt_nearest_object"] = nearest
            out["_gt_eef_to_object"] = d
        out["_gt_n_contacts"] = int(sim.data.ncon)
        return out

    def _object_body_names(self) -> tuple[list[str], bool]:
        """Task objects, from the BDDL problem definition.

        LE-2: an earlier version substring-filtered every body in the model.
        Two problems. (a) It failed OPEN, not closed: `_gt_object_pos` was
        always present but silently incomplete, so distance detectors got a
        confident wrong number rather than skipping (which is what G8 assumes
        omission produces). (b) `link` was in the skip list, and articulated
        LIBERO objects -- drawers, cabinet doors, microwaves -- carry body
        names containing it. For "open the top drawer" the drawer IS the task
        object and was being filtered out, after which `_gt_nearest_object`
        named something else entirely and the spatial_reasoning rule fired on it.

        `obj_of_interest` is the BDDL object list and enumerates exactly the
        task-relevant objects. Returns (names, complete).
        """
        inner = getattr(self._env.unwrapped, "_env", None)
        names = list(getattr(inner, "obj_of_interest", []) or [])
        if not names:
            return [], False
        try:
            sim = self._sim()
            bodies = {sim.model.body_id2name(i) for i in range(sim.model.nbody)}
        except Exception:
            return names, False
        resolved, complete = [], True
        for n in names:
            b = _resolve_body(n, bodies)
            if b:
                resolved.append(b)
            else:
                complete = False
        return resolved, complete

    def _write_images(self) -> dict[str, str]:
        """G2: pixels go to disk, paths go in the trace."""
        if not self.image_dir:
            return {}
        import os
        os.makedirs(self.image_dir, exist_ok=True)
        refs = {}
        px = self._raw.get("pixels", {})
        items = px.items() if isinstance(px, dict) else enumerate(px)
        for i, img in items:
            # LE-4: PNG, not .npy. Lossless and ~3x smaller -- .npy at ~150 KB
            # x ~300 steps x 2 cameras is ~90 MB/episode, i.e. ~36 GB for a
            # 400-episode screen, with nothing reaping it.
            p = os.path.join(self.image_dir, f"t{self.t:04d}_{i}.png")
            try:
                from PIL import Image
                Image.fromarray(img).save(p, optimize=True)
            except Exception:
                import numpy as np
                p = p[:-4] + ".npy"
                np.save(p, img)
            refs[f"cam{i}"] = p
        return refs

    # --- A3/DG-8 ----------------------------------------------------------
    def scene_descriptor(self) -> dict:
        """The reset scene in PHYSICAL UNITS -- metres, world frame.

        This is what makes a paired real-world trial possible later (PPI):
        a person must be able to rebuild this scene on a bench. `seed` + `spec`
        cannot do that, being sim-internal by construction.
        """
        try:
            sim = self._sim()
        except Exception as e:
            return {"unavailable": f"{type(e).__name__}: {e}"}
        names, complete = self._object_body_names()
        objects = []
        for name in names:
            try:
                bid = sim.model.body_name2id(name)
                objects.append({
                    "name": name,
                    "pos_m": [round(float(x), 5) for x in sim.data.body_xpos[bid]],
                    "quat": [round(float(x), 5) for x in sim.data.body_xquat[bid]],
                })
            except Exception:
                continue
        cameras = {}
        try:
            for i in range(sim.model.ncam):
                cn = sim.model.camera_id2name(i)
                cameras[cn] = {
                    "pos_m": [round(float(x), 5) for x in sim.data.cam_xpos[i]],
                    "xmat": [round(float(x), 5) for x in sim.data.cam_xmat[i]],
                    "fovy_deg": round(float(sim.model.cam_fovy[i]), 3),
                }
        except Exception:
            pass
        return {
            "units": "metres", "frame": "world",
            "suite": self.suite, "task_id": self.task_id_idx,
            "instruction": self.instruction,
            "objects": objects, "objects_complete": complete,
            "robot": {"eef_start_pos_m":
                      [round(float(x), 5)
                       for x in self._raw["robot_state"]["eef"]["pos"]]},
            "cameras": cameras,
        }
