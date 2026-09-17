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
import re
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


def _quat_axis_angle(axis, angle_rad):
    """Quaternion (w,x,y,z) for a rotation of `angle_rad` about unit `axis`."""
    import numpy as np
    a = np.asarray(axis, dtype=float)
    a = a / max(np.linalg.norm(a), 1e-12)
    h = angle_rad / 2.0
    return np.array([math.cos(h), *(math.sin(h) * a)])


def _quat_mul(q1, q2):
    """Hamilton product q1*q2 -- applies q2 first, then q1."""
    import numpy as np
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def _quat_rotate(q, v):
    """Rotate vector v by quaternion q."""
    import numpy as np
    qv = np.array([0.0, *v])
    qc = np.array([q[0], -q[1], -q[2], -q[3]])
    return _quat_mul(_quat_mul(q, qv), qc)[1:]


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
                 max_steps: int | None = None, image_dir: str | None = None,
                 obs_size: int = 256, libero_plus: bool = False,
                 libero_plus_raw_instruction: bool = False):
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
        # Render resolution. 256 matches the LeRobot gym class default and the
        # LIBERO datasets. Some documented evals render elsewhere (MINERVA's fork
        # defaults its env CONFIG to 360), and the harness must be able to match
        # them, or a harness-vs-lerobot-eval comparison measures the render path.
        self.obs_size = int(obs_size)
        # LIBERO-Plus: same scenes and gym interface as LIBERO, but each task_id is
        # one pre-generated PERTURBED variant (e.g. 2,402 for libero_spatial).
        # Needs its fork on PYTHONPATH and its own LIBERO_CONFIG_PATH (see
        # .venvs/libero-plus); LeRobot reads some variants' init states from a
        # different directory, so the flag must reach get_task_init_states too.
        self.libero_plus = bool(libero_plus)
        # Deliberately reproduce LeRobot's contaminated instruction (the
        # variant's perturbation parameters appended). Only for measuring how
        # much that contamination costs; never for a reported result.
        self.libero_plus_raw_instruction = bool(libero_plus_raw_instruction)
        self._env = None
        self.instruction = ""
        self.task_id = (f"libero_plus/{suite}/task{task_id}" if libero_plus
                        else f"{suite}/task{task_id}")
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
        return {"name": f"{'libero_plus' if self.libero_plus else 'libero'}:{self.suite}",
                "benchmark": "libero_plus" if self.libero_plus else "libero",
                # v2: LIBERO-Plus perturbation suffix stripped from the instruction
                "instruction_source": ("task.language_RAW_CONTAMINATED"
                                       if self.libero_plus and self.libero_plus_raw_instruction
                                       else "clean_base_scene_v2" if self.libero_plus
                                       else "task.language"),
                "suite": self.suite,
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
                "obs_size": self.obs_size,
                "max_steps": self.max_steps, "action_dims": list(ACTION_DIMS)}

    def _init_states_sha(self) -> str | None:
        """Content hash of this task's stored initial states."""
        try:
            import hashlib, numpy as np
            from lerobot.envs.libero import get_task_init_states, _get_suite
            arr = get_task_init_states(_get_suite(self.suite), self.task_id_idx,
                                       is_libero_plus=self.libero_plus)
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
            observation_width=self.obs_size, observation_height=self.obs_size,
            is_libero_plus=self.libero_plus,
        )
        task = suite.get_task(self.task_id_idx)
        self.raw_task_language = task.language
        self.base_instruction = self._clean_instruction(task)
        self.instruction = self.base_instruction

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
        # The init state (object layout) is chosen by LeRobot from a COUNTER that
        # advances on every reset (`init_state_id += _reset_stride`,
        # lerobot/envs/libero.py:346) -- the seed does not select it. Two bugs
        # followed: a resumed cell that skipped cached episodes ran every later
        # seed on a DIFFERENT layout than its rollout_id claims, and arms sharing
        # one env object (nominal, then yaw 5, ...) were never on the same layouts
        # as each other. Pin it: seed N -> init state N, independent of history.
        # Matches lerobot-eval, where sub-env i of a batch starts at init state i.
        inner = self._env.unwrapped
        states = getattr(inner, "_init_states", None)       # ndarray: no `or`
        n_init = len(states) if states is not None else 0
        if self.init_states and n_init:
            inner.init_state_id = seed % n_init
            self.init_state_index = seed % n_init
        else:
            self.init_state_index = None
        raw, _ = self._env.reset(seed=seed)
        self._raw = raw
        # Any knob that is PRESENT runs the perturbation path, including a value of
        # 0. The old guard `any(v for ...)` treated 0.0 as falsy, so a "yaw 0"
        # control skipped the camera code AND the resettle that every other arm
        # got, and was not a matched control for them (#22).
        if spec.knobs:
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
        cam_keys = ("camera_yaw_deg", "camera_pitch_deg", "camera_dist_m")
        if any(key in k for key in cam_keys):
            # A camera perturbation MOVES LIBERO's own camera; it never re-aims it.
            #
            # The previous version replaced the orientation with a look-at toward
            # a guessed table centre [0, 0, z*0.5]. That re-aim alone rotated the
            # view ~12 deg and shifted pitch in OPPOSITE directions for floor vs
            # tabletop scenes, so "yaw 1e-6" rendered the same as "yaw 5" and
            # every yaw arm of the 20260916-0015 campaign measured the re-aim,
            # not yaw (#22; found by vla-7f, confirmed by vla-81 from stored
            # extrinsics). Now every knob is a RIGID motion of the original pose
            # about the point that camera actually looks at, so 0 is the trained
            # view exactly and small angles are small.
            cid = sim.model.camera_name2id(MAIN_CAMERA)
            if int(sim.model.cam_bodyid[cid]) != 0:
                # cam_pos/cam_quat are parent-relative; the geometry below
                # assumes the world frame. Fail rather than silently mis-rotate.
                raise NotImplementedError(
                    f"{MAIN_CAMERA} is attached to body "
                    f"{int(sim.model.cam_bodyid[cid])}, not the world")
            pos0 = np.array(sim.model.cam_pos[cid], dtype=float)
            q0 = np.array(sim.model.cam_quat[cid], dtype=float)
            fwd = _quat_rotate(q0, [0.0, 0.0, -1.0])       # MuJoCo cams look down -z
            pivot = self._camera_pivot(sim, pos0, fwd)

            pos, q = pos0.copy(), q0.copy()
            if dist:
                rel = pos - pivot
                pos = pivot + rel * (1.0 + dist / max(np.linalg.norm(rel), 1e-9))
            if pitch:
                # positive pitch raises the camera over the pivot (looks more
                # steeply down), about the horizontal axis across the view
                rel = pos - pivot
                axis = np.cross(rel, [0.0, 0.0, 1.0])
                if np.linalg.norm(axis) > 1e-9:
                    qp = _quat_axis_angle(axis, math.radians(pitch))
                    pos = pivot + _quat_rotate(qp, rel)
                    q = _quat_mul(qp, q)
            if yaw:
                # orbit about the world vertical through the pivot
                qy = _quat_axis_angle([0.0, 0.0, 1.0], math.radians(yaw))
                pos = pivot + _quat_rotate(qy, pos - pivot)
                q = _quat_mul(qy, q)
            sim.model.cam_pos[cid] = pos
            sim.model.cam_quat[cid] = q / np.linalg.norm(q)
            sim.forward()

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

    # Perturbation parameters LIBERO-Plus appends to variant names. Everything
    # from the first of these tokens onward is not part of the instruction.
    # Perturbation markers are a SUFFIX grammar, so they are anchored at the end
    # and stripped repeatedly. Splitting on the first occurrence (what this did
    # until 2026-09-18) truncated every scene whose own name contains a marker
    # word: "pick_up_the_black_bowl_from_table_center_..._table_11" became
    # "pick up the black bowl from" -- 240 of 2,402 libero_spatial variants.
    _LPLUS_SUFFIX_RE = re.compile(
        r"(?:_(?:view|initstate|noise|light|table|tb|add|language|distractor)(?:_-?\d+)+"
        r"|_level\d+(?:_sample\d+)?|_sample\d+|_moved)+$")
    # libero_10 variant names carry the scene prefix ("KITCHEN_SCENE3_..."), which
    # the LeRobot datasets the policies trained on do not.
    _LPLUS_SCENE_PREFIX_RE = re.compile(r"^[A-Z][A-Z0-9_]*SCENE\d+_")

    def _clean_instruction(self, task) -> str:
        """The instruction the policy should receive.

        LIBERO-Plus builds `task.language` from the variant's FILE NAME
        (`benchmark/__init__.py:grab_language_from_filename`). For every
        non-language perturbation the name carries its parameters, so the policy
        was told e.g. "... place it on the plate view 0 0 100 2 352 initstate 0"
        or "... noise 26". LeRobot forwards that string unchanged
        (`envs/libero.py:180`), so every LIBERO-Plus run through it perturbed the
        LANGUAGE as well as the intended factor.

        Fix: for non-language variants, the instruction is the base scene name
        as words, which is exactly vanilla LIBERO's instruction and the text the
        policies were trained on. It is NOT the BDDL's own `language_instruction`
        ("pick the akita black bowl ..."), which is worded differently from the
        training data. Language variants keep `task.language`: their rewrite is
        read from the BDDL and IS the perturbation.
        """
        lang = task.language
        if (not self.libero_plus or "_language_" in task.name
                or self.libero_plus_raw_instruction):
            return lang
        base = self._LPLUS_SCENE_PREFIX_RE.sub("", task.name)
        prev = None
        while prev != base:
            prev, base = base, self._LPLUS_SUFFIX_RE.sub("", base)
        return " ".join(base.split("_"))

    def _libero_plus_variant(self) -> dict | None:
        """Perturbation type and difficulty of this LIBERO-Plus variant.

        From the fork's own task_classification.json, matched by task NAME (not
        position), so a reordered catalogue cannot silently mislabel a variant.
        This is what lets mined failures be grouped by perturbation type.
        """
        if not self.libero_plus:
            return None
        try:
            import json, os
            from lerobot.envs.libero import _get_suite
            from libero.libero import get_libero_path
            name = _get_suite(self.suite).get_task(self.task_id_idx).name
            cache = getattr(LiberoEnv, "_lplus_cls", None)
            if cache is None:
                path = os.path.join(get_libero_path("benchmark_root"), "benchmark",
                                    "task_classification.json")
                cache = LiberoEnv._lplus_cls = json.load(open(path))
            for r in cache.get(self.suite, []):
                if r["name"] == name:
                    return {"variant": name, "category": r["category"],
                            "difficulty_level": r["difficulty_level"],
                            "catalogue_id": r["id"],
                            "raw_task_language": getattr(self, "raw_task_language", None),
                            "instruction_given": getattr(self, "base_instruction", None)}
            return {"variant": name, "category": None, "difficulty_level": None,
                    "note": "not found in task_classification.json"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def _camera_pivot(self, sim, cam_pos, fwd):
        """The point the camera actually looks at.

        The optical axis intersected with the horizontal plane through the task
        objects. Falls back to 1 m along the axis if there are no objects or the
        axis never descends to them.
        """
        import numpy as np
        names, _ = self._object_body_names()
        zs = []
        for n in names:
            try:
                zs.append(float(self._object_pos(sim, n)[2]))
            except Exception:
                continue
        if zs and fwd[2] < -1e-6:
            t = (float(np.mean(zs)) - cam_pos[2]) / fwd[2]
            if t > 0:
                return cam_pos + t * np.asarray(fwd)
        return cam_pos + np.asarray(fwd)

    def _resettle(self):
        """Step the sim so the perturbation takes effect, and RETURN THAT FRAME.

        This used to call `_to_lerobot_obs`, which LeRobot's LiberoEnv does not
        have; a `hasattr` fallback then returned `self._raw` -- the frame rendered
        BEFORE the perturbation. So the policy's first observation of every
        perturbed episode showed the unperturbed scene, with no error (#22).
        The formatter is `_format_raw_obs`, and its absence is now an error.
        """
        from lerobot.envs.libero import get_libero_dummy_action
        outer = self._env.unwrapped
        if not hasattr(outer, "_format_raw_obs"):
            raise RuntimeError(
                "LeRobot LiberoEnv has no _format_raw_obs; cannot return the "
                "post-perturbation frame. Refusing to hand back a stale one.")
        raw = None
        for _ in range(3):
            raw, _, _, _ = outer._env.step(get_libero_dummy_action())
        return outer._format_raw_obs(raw)

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
                p = [float(x) for x in self._object_pos(sim, name)]
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
            sites = {sim.model.site_id2name(i) for i in range(sim.model.nsite)}
        except Exception:
            return names, False
        resolved, complete = [], True
        self._object_kinds = {}
        for n in names:
            b = _resolve_body(n, bodies)
            if b:
                resolved.append(b); self._object_kinds[b] = "body"
            elif n in sites:
                # A table REGION ("main_table_stove_front_region") is a MuJoCo
                # SITE -- a named, massless location marker -- not a body. It was
                # unresolvable before, so libero_goal/task5 ("push the plate to
                # the front of the stove") recorded an incomplete object list
                # and no distances on every trace. The nearest-named body,
                # flat_stove_1_main, sits ~35 cm from the region, so resolving to
                # it by name would have been worse than failing.
                resolved.append(n); self._object_kinds[n] = "site"
            else:
                complete = False
        return resolved, complete

    def _object_pos(self, sim, name):
        """World position of a resolved task object, body or site."""
        if getattr(self, "_object_kinds", {}).get(name) == "site":
            return sim.data.site_xpos[sim.model.site_name2id(name)]
        return sim.data.body_xpos[sim.model.body_name2id(name)]

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
                kind = getattr(self, "_object_kinds", {}).get(name, "body")
                quat = None
                if kind == "body":
                    bid = sim.model.body_name2id(name)
                    quat = [round(float(x), 5) for x in sim.data.body_xquat[bid]]
                objects.append({
                    "name": name, "kind": kind,
                    "pos_m": [round(float(x), 5) for x in self._object_pos(sim, name)],
                    "quat": quat,
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
            # which stored LIBERO init state this episode started from; pinned
            # to seed % n_init_states so it no longer depends on reset history
            "init_state_index": getattr(self, "init_state_index", None),
            "libero_plus": self._libero_plus_variant(),
            "instruction": self.instruction,
            "objects": objects, "objects_complete": complete,
            "robot": {"eef_start_pos_m":
                      [round(float(x), 5)
                       for x in self._raw["robot_state"]["eef"]["pos"]]},
            "cameras": cameras,
        }
