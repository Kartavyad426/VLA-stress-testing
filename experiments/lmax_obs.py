"""Observability logging for LIBERO-MAX arms: the agreed common rollout format (docs/PORTABILITY_LIBERO_MAX.html §5,
agreed 10-14 with amendments [A1]-[A7] and the user's frame policy). Written IN ADDITION to LIBERO-MAX's own case files.

Per arm, under the shard's --output-root:
  rollouts.jsonl                      one schema-3.0 Rollout row per arm, plus "case", "event", "goal" blocks
  npz/<case_id>__<arm>.npz            act_P (F_model x H x 7, normalised, MODEL forwards only), forward_env_steps,
                                      forward_policy_step, query_seed
  npz/<case_id>__<arm>.frames.npz     uint8 frames as the policy saw them (180-degree flip applied): agent + wrist at
                                      every policy query at 256 px; 360 px at t_event-1..t_event+1 and at the
                                      grasp-onset query; per frame t, camera, px, reason

One clock [A4]: steps[t] is env step t counted from the restored initial state, INCLUDING LIBERO-MAX's 10 warm-up
steps ("warmup": true). steps[t] holds the observation at t and the action applied at t (None on the terminal step).
LIBERO-MAX's policy_step = t - warmup.

Ground truth reuses the harness's own code (vla_harness/envs/libero_env.py: _privileged, _scene_object_pos,
_object_body_names, _object_pos) through a shim over the raw LIBERO env, so the _gt_* keys mean exactly what they
mean in every earlier trace. New per step: _gt_goal_predicates (LIBERO's own _eval_predicate on each BDDL goal atom,
never re-implemented) and _gt_held (robosuite _check_grasp per BDDL object).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from vla_harness.envs.libero_env import LiberoEnv
from vla_harness.schema import SCHEMA_VERSION, TERM_SUCCESS, TERM_TIMEOUT, derive_id

FRAME_PX_SMALL, FRAME_PX_LARGE = 256, 360
CAMS = ("agentview_image", "robot0_eye_in_hand_image")
CAM_NAMES = ("agentview", "robot0_eye_in_hand")
ACTION_DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
LIVE_DIMS = 7


class _GTShim:
    """Just enough of LiberoEnv for its _gt_* methods to run on a raw LIBERO OffScreenRenderEnv."""
    _privileged = LiberoEnv._privileged
    _scene_object_pos = LiberoEnv._scene_object_pos
    _object_body_names = LiberoEnv._object_body_names
    _object_pos = LiberoEnv._object_pos

    def __init__(self, raw_env):
        self._env = SimpleNamespace(unwrapped=SimpleNamespace(_env=raw_env))
        self._raw = {"robot_state": {"eef": {"pos": None}}}

    def _sim(self):
        return self._env.unwrapped._env.env.sim

    def privileged(self, raw_obs: dict) -> dict:
        self._raw["robot_state"]["eef"]["pos"] = [float(x) for x in raw_obs["robot0_eef_pos"]]
        return self._privileged()


def _problem(raw_env):
    """The LIBERO problem env (holds parsed_problem, object_states_dict, robots, _check_grasp)."""
    return raw_env.env


def goal_atoms(raw_env) -> list[list[str]]:
    return [list(map(str, s)) for s in _problem(raw_env).parsed_problem["goal_state"]]


def _atom_key(atom) -> str:
    return "%s(%s)" % (atom[0].capitalize(), ",".join(atom[1:]))


def _flip(img: np.ndarray) -> np.ndarray:
    """LeRobot's LiberoProcessorStep rotates both image axes (torch.flip dims=[2,3]); this is that view."""
    return np.ascontiguousarray(np.asarray(img)[::-1, ::-1])


def _resize(img: np.ndarray, px: int) -> np.ndarray:
    if img.shape[0] == px and img.shape[1] == px:
        return img
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((px, px), Image.BILINEAR))


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _num_vec(state: dict) -> dict[str, np.ndarray]:
    """Numeric fields compared between arms at t_event - 1 [A5]."""
    out = {k: np.asarray(state[k], dtype=np.float64)
           for k in ("eef_pos", "eef_quat", "gripper_qpos", "joint_pos", "joint_vel") if k in state}
    for name, p in (state.get("_gt_scene_object_pos") or {}).items():
        out["scene:" + name] = np.asarray(p, dtype=np.float64)
    return out


SNAP_OBSERVABLES = ("robot0_joint_pos",) + CAMS


def _install_obs_snapshot(prob, rec) -> None:
    """Wrap the sensors of SNAP_OBSERVABLES (once per env) so each real sample stores sim.get_state() into
    prob._lmax_rec._snap[name]. Re-pointing prob._lmax_rec lets a reused env serve a new recorder."""
    prob._lmax_rec = rec
    rec._snap = {}
    sim = prob.sim
    # MuJoCo's mj_step runs forward (xpos, site poses, what cameras render) and THEN integrates qpos, so a sample
    # taken right after a substep sees kinematics of the state BEFORE that substep. Keep that pre-step state.
    if getattr(sim, "_lmax_orig_step", None) is None:
        orig_step = sim.step

        def step(*a, _orig=orig_step, _sim=sim, **k):
            _sim._lmax_pre = _sim.get_state().flatten()
            return _orig(*a, **k)
        sim._lmax_orig_step = orig_step
        sim.step = step
    for name in SNAP_OBSERVABLES:
        ob = prob._observables[name]
        if getattr(ob, "_lmax_orig_sensor", None) is not None:
            continue
        orig = ob._sensor

        def sensor(obs_cache, _orig=orig, _name=name, _prob=prob):
            pre = getattr(_prob.sim, "_lmax_pre", None)
            _prob._lmax_rec._snap[_name] = None if pre is None else np.asarray(pre, dtype=np.float64)
            return _orig(obs_cache)
        ob._lmax_orig_sensor = orig
        ob._sensor = sensor


class ArmRecorder:
    """Collects one arm. Call obs(t, raw_obs) for every observation, act(t, action) for every executed action,
    query(...) for every policy query, then finish(...)."""

    def __init__(self, raw_env, arm: str, warmup_steps: int, query_px: int = FRAME_PX_SMALL,
                 every_step_px: int | None = None, event_entities: list[str] | None = None):
        self.raw_env, self.arm, self.warmup = raw_env, arm, int(warmup_steps)
        self.query_px, self.every_step_px = int(query_px), every_step_px
        self.sim_states: list[np.ndarray] = []
        self.act_P_env, self.feat_mean, self.feat_img_mean = [], [], []
        self.release_t = None
        self.gt = _GTShim(raw_env)
        # Bodies named by the case's event (moved, inserted, or reference objects). Inserted obstacles/distractors
        # are parked bodies WITHOUT a free joint, so _gt_scene_object_pos misses them; log their pose every step.
        sim = self.gt._sim()
        bodies = {sim.model.body_id2name(i) for i in range(sim.model.nbody)}
        from vla_harness.envs.libero_env import _resolve_body
        self.event_bodies = {}
        for n in event_entities or []:
            b = _resolve_body(n, bodies)
            if b:
                self.event_bodies[n] = sim.model.body_name2id(b)
        self._last_raw = None
        # robosuite samples observables (proprio AND cameras) on the FIRST sim substep of each control step, not at
        # its end, so what the policy saw is NOT sim_states[t]. obs_sim_states[t] = the state whose forward
        # kinematics the sampled cameras/eef saw (pre-integration state of the sampling substep); restoring it
        # (set_state + forward) should re-render the policy's view exactly. joint_pos/gripper_qpos read qpos, so
        # they are one substep (2 ms) later than that.
        self.obs_sim_states: list[np.ndarray] = []
        self.obs_snap_mismatch = 0                   # steps where proprio and camera were sampled at different states
        _install_obs_snapshot(_problem(raw_env), self)
        self.atoms = goal_atoms(raw_env)
        prob = _problem(raw_env)
        self.held_objects = [n for n in prob.objects_dict]
        self.steps: list[dict] = []
        self.f_small, self.f_large = [], []          # (t, cam_idx, reason, uint8 image)
        self.act_P, self.fwd_t, self.fwd_pstep, self.qseed = [], [], [], []
        self.grasp_onset_t = None
        self._grasp_query_done = False
        self.t_event = None
        self._prev_raw = None
        self._event_large_due = set()
        self.t0 = time.perf_counter()

    # --- per step -------------------------------------------------------------------------------------------
    def _state(self, raw_obs: dict) -> dict:
        s = {"eef_pos": [float(x) for x in raw_obs["robot0_eef_pos"]],
             "eef_quat": [float(x) for x in raw_obs["robot0_eef_quat"]],
             "gripper_qpos": [float(x) for x in raw_obs["robot0_gripper_qpos"]],
             "gripper_qvel": [float(x) for x in raw_obs["robot0_gripper_qvel"]],
             "joint_pos": [float(x) for x in raw_obs["robot0_joint_pos"]],
             "joint_vel": [float(x) for x in raw_obs["robot0_joint_vel"]]}
        s.update(self.gt.privileged(raw_obs))
        prob = _problem(self.raw_env)
        preds = {}
        for atom in self.atoms:
            try:
                preds[_atom_key(atom)] = bool(prob._eval_predicate(atom))
            except Exception as e:                  # recorded, never silently dropped
                preds[_atom_key(atom)] = "error: %s" % type(e).__name__
        s["_gt_goal_predicates"] = preds
        held = []
        gripper = prob.robots[0].gripper
        for name in self.held_objects:
            try:
                if prob._check_grasp(gripper=gripper, object_geoms=prob.objects_dict[name]):
                    held.append(name)
            except Exception:
                continue
        s["_gt_held"] = held
        sim = self.gt._sim()
        # orientation of every free body (knocked-over objects), same body list as _gt_scene_object_pos
        s["_gt_scene_object_quat"] = {n: [float(x) for x in sim.data.body_xquat[b]]
                                      for n, b in getattr(self.gt, "_scene_bodies", []) if n}
        # every active contact as a (body, body) pair: finger pads, object-object, table
        m, pairs = sim.model, set()
        for i in range(int(sim.data.ncon)):
            c = sim.data.contact[i]
            b1 = m.body_id2name(int(m.geom_bodyid[c.geom1])); b2 = m.body_id2name(int(m.geom_bodyid[c.geom2]))
            pairs.add(tuple(sorted((str(b1), str(b2)))))
        s["_gt_contact_pairs"] = sorted(list(p) for p in pairs)
        s["_gt_event_object_pose"] = {n: {"pos": [float(x) for x in sim.data.body_xpos[b]],
                                          "quat": [float(x) for x in sim.data.body_xquat[b]]}
                                      for n, b in self.event_bodies.items()}
        try:
            s["_gt_success"] = bool(prob._check_success())
        except Exception as e:
            s["_gt_success"] = "error: %s" % type(e).__name__
        return s

    def obs(self, t: int, raw_obs: dict) -> None:
        assert t == len(self.steps), (t, len(self.steps))
        st = self._state(raw_obs)
        # FULL restorable MuJoCo state (LIBERO's flattened format: time, qpos, qvel), the same layout as the
        # benchmark init_states, so env.set_init_state(sim_states[t]) restores step t (after re-applying the
        # event's model edits for t >= t_event).
        end = np.asarray(_problem(self.raw_env).sim.get_state().flatten(), dtype=np.float64)
        self.sim_states.append(end)
        snaps = [self._snap.get(n) for n in SNAP_OBSERVABLES]
        if t == 0 or any(x is None for x in snaps):
            # t=0 obs comes from a forced update (reset / set_init_state) at the current state
            self.obs_sim_states.append(end)
        else:
            if any(not np.array_equal(x, snaps[0]) for x in snaps[1:]):
                self.obs_snap_mismatch += 1
            self.obs_sim_states.append(snaps[-1])   # the state the cameras were rendered at
        self._snap.clear()
        self.steps.append({"t": t, "obs_state": st, "action": None, "warmup": t < self.warmup,
                           "policy_query": None})
        if self.grasp_onset_t is None and t >= self.warmup and st["_gt_held"]:
            self.grasp_onset_t = t
        if (self.grasp_onset_t is not None and self.release_t is None and t > self.grasp_onset_t
                and not st["_gt_held"]):
            self.release_t = t
            self._store(t, raw_obs, FRAME_PX_LARGE, "release")
        if self.every_step_px:
            self._store(t, raw_obs, int(self.every_step_px), "step")
        self._last_raw = (t, {k: raw_obs[k] for k in CAMS})
        if t in self._event_large_due:              # t_event + 1
            self._store(t, raw_obs, FRAME_PX_LARGE, "event")
            self._event_large_due.discard(t)
        self._prev_raw = (t, {k: raw_obs[k] for k in CAMS})

    def event(self, t_event: int, raw_obs_at_event: dict) -> None:
        """Called once, right after the step whose observation first reflects the event (steps index t_event)."""
        self.t_event = t_event
        if self._prev_raw is not None and self._prev_raw[0] == t_event - 1:
            self._store(t_event - 1, self._prev_raw[1], FRAME_PX_LARGE, "event")
        self._store(t_event, raw_obs_at_event, FRAME_PX_LARGE, "event")
        self._event_large_due.add(t_event + 1)

    def act(self, t: int, action) -> None:
        self.steps[t]["action"] = [float(x) for x in np.asarray(action, dtype=np.float64).reshape(-1)]

    def query(self, t: int, raw_obs: dict, source: str, instruction: str, forward_index: int,
              act_full_norm: np.ndarray | None, query_seed: int | None, act_full_env: np.ndarray | None = None,
              feats: dict | None = None) -> None:
        views = {CAM_NAMES[i]: _flip(raw_obs[c]) for i, c in enumerate(CAMS)}
        self.steps[t]["policy_query"] = {
            "forward": forward_index, "source": source, "instruction_given": instruction,
            "frame_sha": {k: _sha(v) for k, v in views.items()}}
        if not (self.every_step_px and int(self.every_step_px) == self.query_px):   # else every-step frames cover it
            for i, c in enumerate(CAMS):
                img = _resize(views[CAM_NAMES[i]], self.query_px)
                (self.f_large if self.query_px == FRAME_PX_LARGE else self.f_small).append((t, i, "query", img))
        if (self.grasp_onset_t is not None and not self._grasp_query_done and t >= self.grasp_onset_t):
            self._store(t, raw_obs, FRAME_PX_LARGE, "grasp_onset")
            self._grasp_query_done = True
        if source == "model":
            self.act_P.append(np.asarray(act_full_norm, dtype=np.float32))
            self.fwd_t.append(t)
            self.fwd_pstep.append(t - self.warmup)
            self.qseed.append(-1 if query_seed is None else int(query_seed))
            self.act_P_env.append(np.asarray(act_full_env, dtype=np.float32))
            feats = feats or {}
            self.feat_mean.append(np.asarray(feats.get("mean", np.zeros(0)), dtype=np.float16))
            self.feat_img_mean.append(np.asarray(feats.get("img_mean", np.zeros(0)), dtype=np.float16))

    def _store(self, t: int, raw: dict, px: int, reason: str) -> None:
        for i, c in enumerate(CAMS):
            img = _flip(raw[c])
            assert px in (FRAME_PX_SMALL, FRAME_PX_LARGE), px
            (self.f_large if px == FRAME_PX_LARGE else self.f_small).append((t, i, reason, _resize(img, px)))

    # --- output ---------------------------------------------------------------------------------------------
    def save_npz(self, npz_dir: Path, case_id: str) -> dict:
        npz_dir.mkdir(parents=True, exist_ok=True)
        stem = "%s__%s" % (case_id, self.arm)
        pred = npz_dir / (stem + ".npz")
        H = self.act_P[0].shape[0] if self.act_P else 0
        np.savez_compressed(
            pred,
            act_P=(np.stack(self.act_P) if self.act_P else np.zeros((0, 0, LIVE_DIMS), np.float32)),
            forward_source=np.array(["model"] * len(self.act_P)),
            forward_env_steps=np.asarray(self.fwd_t, np.int64),
            forward_policy_step=np.asarray(self.fwd_pstep, np.int64),
            query_seed=np.asarray(self.qseed, np.int64),
            act_P_env=(np.stack(self.act_P_env) if self.act_P_env else np.zeros((0, 0, LIVE_DIMS), np.float32)),
            backbone_mean=(np.stack(self.feat_mean) if self.feat_mean else np.zeros((0, 0), np.float16)),
            backbone_img_mean=(np.stack(self.feat_img_mean) if self.feat_img_mean else np.zeros((0, 0), np.float16)),
            sim_states=(np.stack(self.sim_states) if self.sim_states else np.zeros((0, 0))),
            sim_state_t=np.arange(len(self.sim_states), dtype=np.int64),
            obs_sim_states=(np.stack(self.obs_sim_states) if self.obs_sim_states else np.zeros((0, 0))),
            obs_snap_mismatch=np.int64(self.obs_snap_mismatch))
        fr = npz_dir / (stem + ".frames.npz")
        payload = {}
        for tag, rows in ((str(FRAME_PX_SMALL), self.f_small), (str(FRAME_PX_LARGE), self.f_large)):
            payload["frames_" + tag] = (np.stack([r[3] for r in rows]) if rows
                                        else np.zeros((0, int(tag), int(tag), 3), np.uint8))
            payload["t_" + tag] = np.asarray([r[0] for r in rows], np.int64)
            payload["camera_" + tag] = np.asarray([CAM_NAMES[r[1]] for r in rows])
            payload["reason_" + tag] = np.asarray([r[2] for r in rows])
            payload["px_" + tag] = np.full(len(rows), int(tag), np.int64)
        np.savez_compressed(fr, **payload)
        return {"predictions": str(pred), "frames": str(fr), "H": int(H)}

    def finish(self) -> None:
        """The frame success was decided on (E7 asks on_plate/done there)."""
        if self._last_raw is not None:
            self._store(self._last_raw[0], self._last_raw[1], FRAME_PX_LARGE, "final")

    def prefix_sha(self, t_end: int) -> str:
        acts = [s["action"] for s in self.steps[:t_end] if s["action"] is not None]
        return _sha(np.asarray(acts, np.float64))


def event_entities(scenario: dict) -> list[str]:
    """Every entity a case's event names: trigger target, moved/inserted object, reference and support entities,
    distractor placements."""
    ch, out = scenario.get("change") or {}, [scenario.get("trigger", {}).get("value")]
    out += [ch.get("object"), ch.get("relative_to"), ch.get("support_entity")]
    for pl in ch.get("placements") or []:
        out += [pl.get("object"), pl.get("relative_to"), pl.get("support_entity")]
    return sorted({x for x in out if isinstance(x, str) and x})


def _scene_descriptor(raw_env, gt: _GTShim, case: dict, instruction: str, eef0) -> dict:
    sim = gt._sim()
    names, complete = gt._object_body_names()
    objects = []
    for n in names:
        try:
            kind = getattr(gt, "_object_kinds", {}).get(n, "body")
            quat = None
            if kind == "body":
                quat = [round(float(x), 5) for x in sim.data.body_xquat[sim.model.body_name2id(n)]]
            objects.append({"name": n, "kind": kind,
                            "pos_m": [round(float(x), 5) for x in gt._object_pos(sim, n)], "quat": quat})
        except Exception:
            continue
    cameras = {}
    for i in range(sim.model.ncam):
        cameras[sim.model.camera_id2name(i)] = {
            "pos_m": [round(float(x), 5) for x in sim.data.cam_xpos[i]],
            "xmat": [round(float(x), 5) for x in sim.data.cam_xmat[i]],
            "fovy_deg": round(float(sim.model.cam_fovy[i]), 3)}
    return {"units": "metres", "frame": "world", "suite": case["task_suite_name"], "task_id": case["task_index"],
            "init_state_index": case["init_state_index"], "libero_max_task_name": case.get("task_name"),
            "instruction": instruction, "objects": objects, "objects_complete": complete,
            "robot": {"eef_start_pos_m": [round(float(x), 5) for x in eef0]}, "cameras": cameras}


def build_rows(case: dict, rec: dict[str, ArmRecorder], trace: dict[str, dict], scenes: dict[str, dict],
               fp: dict, meta: dict, policy_id: str, npz_paths: dict[str, dict], warmup: int) -> list[dict]:
    """One Rollout row per arm (control -> "base", intervention -> "dynamic")."""
    variant = case.get("substrate_variant") or {}
    src = "LIBERO-PRO" if variant.get("benchmark") == "LIBERO-PRO" else "LIBERO-Plus"
    rows = []
    base_rec, dyn_rec = rec["control"], rec["intervention"]
    dyn_tr = trace["intervention"]
    # Event block for the Dynamic arm.
    if True:  # the Dynamic arm's event block (the Base arm's is None)
        events = dyn_tr.get("intervention_events") or []
        trig = dyn_tr.get("trigger_observation")
        t_event = dyn_rec.t_event
        fwd_event = None
        if t_event is not None:
            fwd_event = next((s["t"] for s in dyn_rec.steps if s["t"] >= t_event and s["policy_query"]
                              and s["policy_query"]["source"] == "model"), None)
        state_match = None
        img_match = None
        img_sha = None
        prefix_exact = None
        psha = None
        if t_event is not None and t_event >= 1 and len(base_rec.steps) > t_event - 1:
            a, b = _num_vec(base_rec.steps[t_event - 1]["obs_state"]), _num_vec(dyn_rec.steps[t_event - 1]["obs_state"])
            keys = sorted(set(a) | set(b))
            state_match = max((float(np.max(np.abs(a[k] - b[k]))) if k in a and k in b and a[k].shape == b[k].shape
                               else math.inf) for k in keys)
            q_before = [s for s in dyn_rec.steps[:t_event] if s["policy_query"]]
            if q_before:
                last = q_before[-1]
                img_sha = last["policy_query"]["frame_sha"]
                bq = base_rec.steps[last["t"]]["policy_query"] if len(base_rec.steps) > last["t"] else None
                img_match = bool(bq is not None and bq["frame_sha"] == img_sha)
            psha = dyn_rec.prefix_sha(t_event)
            prefix_exact = bool(psha == base_rec.prefix_sha(t_event))
        resolved = None
        if events:
            br = events[0].get("backend_result") or {}
            if "position_before_m" in br:
                resolved = {"objects": [{"name": br.get("entity"), "pose_before": br.get("position_before_m"),
                                         "pose_after": br.get("position_after_m")}]}
        trigger = case["scenario"]["trigger"]
        ev = {"type": case["scenario"]["change_type"],
              "trigger": {"kind": case.get("dynamic_phase", trigger.get("type")), "entity": trigger.get("value"),
                          "radius_m": trigger.get("distance_m")},
              "trigger_reached": trig is not None,
              "t_event": t_event, "forward_event": fwd_event,
              "stale_actions_after_event": (None if fwd_event is None or t_event is None else fwd_event - t_event),
              "change": case["scenario"]["change"], "change_resolved": resolved,
              "prefix_sha": psha, "prefix_replay_exact": prefix_exact, "prefix_state_match": state_match,
              "prefix_image_sha": img_sha, "prefix_image_match": img_match}
    goal = {"stages": None,   # per-task config (PORTABILITY §4) not built yet; goal_predicates are from the BDDL
            "goal_predicates": base_rec.atoms}
    for arm, label in (("control", "base"), ("intervention", "dynamic")):
        r, tr = rec[arm], trace[arm]
        steps = r.steps
        success = bool(tr["success"])
        nonwarm = sum(1 for s in steps if not s["warmup"] and s["action"] is not None)
        rid = derive_id({"name": "%s:%s" % (case["case_id"], label), "case_id": case["case_id"], "arm": label,
                         "policy_id": policy_id})
        rows.append({
            "rollout_id": rid,
            "task_id": "libero_max/%s/%s" % (case["task_suite_name"], case.get("task_name")),
            "instruction": tr.get("task_description"),
            "policy_id": policy_id,
            "env_id": "libero_max:%s:%s" % (src, case["task_suite_name"]),
            "seed": int(case["policy_seed"]),
            "perturbation": {},
            "action_dims": list(ACTION_DIMS),
            "steps": steps,
            "success": success,
            "termination": TERM_SUCCESS if success else TERM_TIMEOUT,
            "wall_time_s": round(time.perf_counter() - r.t0, 2),
            "env_steps": nonwarm,
            "model_forwards": len(r.act_P),
            "fingerprint": fp,
            "scene_descriptor": scenes[arm],
            "privileged": False,
            "schema_version": SCHEMA_VERSION,
            "meta": dict(meta, npz=npz_paths[arm], grasp_onset_t=r.grasp_onset_t),
            "case": {"case_id": case["case_id"], "pair_id": case["case_id"], "arm": label,
                     "source_variant": case.get("task_name"), "init_state_index": case["init_state_index"],
                     "task_config_ref": "%s/%s" % (case["task_suite_name"], case["task_index"]),
                     "max_steps": meta["max_policy_steps"], "warmup_steps": warmup,
                     "task_name": case.get("task_name"), "substrate_category": case.get("substrate_category"),
                     "source_benchmark": src, "policy_seed": int(case["policy_seed"]),
                     "init_state_sha256": tr.get("init_state_sha256")},
            "event": ev if arm == "intervention" else None,
            "goal": goal,
        })
    return rows


def append_rows(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
