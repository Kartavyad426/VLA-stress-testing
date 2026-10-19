"""B1: restore an R-047 rollout's root state by replaying its logged actions
(docs/DMP_POC_METHOD.html §5.1). CPU.

R-047's logs keep per-step EEF, joints, gripper and object positions, not the
full MuJoCo state. So: rebuild the env exactly as experiments/r047_run.py did
(same LIBERO-Plus task, env seed / init state, perturbation spec), reset, and
step the logged actions for t = 0 .. t_root - 1. After each step the env's own
observation (the code path that wrote the log) is compared with the logged
one: EEF position and every object position. The restore is accepted only if
every step is within TOL_M.

Rendering is not needed but LeRobot's LiberoEnv always renders; run with
MUJOCO_GL=osmesa and CUDA_VISIBLE_DEVICES= so it stays on the CPU.

    CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=<main>/third_party/LIBERO-plus \
    LIBERO_CONFIG_PATH=<main>/third_party/libero-plus-config \
    <main>/.venvs/libero-plus/bin/python experiments/dmp_poc/run.py b1

UNVERIFIED: new numeric code, not yet checked by val.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.dirname(os.path.dirname(HERE))


@lru_cache(maxsize=None)
def main_repo() -> str:
    """The main checkout, where the data, venvs and run logs live (this code
    runs from a worktree that has none of them)."""
    common = subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=HERE, text=True).strip()
    return os.path.dirname(os.path.abspath(os.path.join(HERE, common)))


sys.path.insert(0, main_repo())
from vla_harness.envs.libero_env import LiberoEnv  # noqa: E402
from vla_harness.schema import Action, PerturbationSpec  # noqa: E402

R047 = os.path.join(main_repo(), "runs/r047/rollouts.jsonl")
DT = 0.05            # 20 Hz control
TOL_M = 1e-3         # B1 acceptance: EEF and every object within 1 mm at every step
EEF_SITE = "gripper0_grip_site"


@lru_cache(maxsize=1)
def r047_rows():
    with open(R047) as f:
        return [json.loads(line) for line in f]


def make_env(task_id):
    # identical to experiments/r047_run.py make_env
    return LiberoEnv(suite="libero_spatial", task_id=task_id, libero_plus=True,
                     libero_plus_base_instruction=True, obs_size=360)


def spec_of(perturbation: dict) -> PerturbationSpec:
    # r047_run.spec_for: magnitude 0 (the {} rows) -> empty spec
    return PerturbationSpec.of(**perturbation) if perturbation else PerturbationSpec()


def scene_of(row) -> int:
    return int(row["task_id"].rsplit("task", 1)[1])


def _err(obs_state, logged):
    """Max position error (m) between the env's observation and the log."""
    e = np.linalg.norm(np.subtract(obs_state["eef_pos"], logged["eef_pos"]))
    objs = {}
    for key in ("_gt_object_pos", "_gt_scene_object_pos"):   # task objects, then every scene object
        for k, v in logged.get(key, {}).items():
            got = obs_state.get(key, {}).get(k)
            objs[k] = float("inf") if got is None else float(np.linalg.norm(np.subtract(got, v)))
    return float(e), objs


@dataclass
class Root:
    row: int
    t: int
    env: object = field(repr=False)
    ok: bool
    max_eef_err_m: float
    max_obj_err_m: float
    worst_step: int
    eef_pos: np.ndarray
    eef_rot: Rotation
    eef_vel: np.ndarray            # m/s, from the logged EEF positions (t-1 -> t)
    gripper_qpos: np.ndarray
    target: str                    # object of interest the policy should grasp
    objects: dict                  # body name -> (pos (3,), Rotation)
    per_step: list = field(repr=False, default_factory=list)

    def summary(self):
        return {"row": self.row, "t": self.t, "ok": self.ok, "tol_m": TOL_M,
                "max_eef_err_m": self.max_eef_err_m, "max_obj_err_m": self.max_obj_err_m,
                "worst_step": self.worst_step, "target": self.target,
                "eef_pos": self.eef_pos.tolist(), "eef_vel": self.eef_vel.tolist()}


def sim_of(env):
    return env._sim()


def control_env(env):
    """LIBERO's ControlEnv under LeRobot's wrapper (has obj_of_interest, check_success)."""
    return env._env.unwrapped._env


def eef_pose(sim):
    sid = sim.model.site_name2id(EEF_SITE)
    return np.array(sim.data.site_xpos[sid]), Rotation.from_matrix(np.array(sim.data.site_xmat[sid]).reshape(3, 3))


def body_poses(sim, names):
    out = {}
    for n in names:
        try:
            bid = sim.model.body_name2id(n)
        except Exception:
            continue
        q = np.array(sim.data.body_xquat[bid])           # MuJoCo order w, x, y, z
        out[n] = (np.array(sim.data.body_xpos[bid]), Rotation.from_quat([q[1], q[2], q[3], q[0]]))
    return out


def cameras_off(env, on=False):
    """Disable (or re-enable, on=True) the image observables so ControlEnv.step
    does not render. Physics is untouched (LeRobot's step = ControlEnv.step +
    check_success; all three paths gave identical trajectories, 10-14). They
    must be back on before a LeRobot reset, which formats the camera images."""
    re = control_env(env).env
    for name, ob in list(re._observables.items()):
        if getattr(ob, "modality", "") == "image" or name.endswith("_image") or "depth" in name:
            re.modify_observable(name, "enabled", on)
            re.modify_observable(name, "active", on)


def restore_fast(row_idx: int, t_root: int, env=None, tol: float = TOL_M) -> Root:
    """restore() without rendering after the reset: reset through the harness
    (perturbation + resettle as R-047), cameras off, then step LIBERO's
    ControlEnv with the logged actions. Checked per step against the log with
    the same quantities: robosuite's robot0_eef_pos observable (what the harness
    logged as eef_pos) and the scene body positions. Re-running this is how a
    root is rewound (a flattened-state rewind was not deterministic: 4 cm apart
    after 200 steps, 10-14)."""
    row = r047_rows()[row_idx]
    steps = row["steps"]
    if env is not None and env._env is not None:
        cameras_off(env, on=True)
    env = env or make_env(scene_of(row))
    env.reset(seed=int(row["seed"]), spec=spec_of(row["perturbation"]))
    cameras_off(env)
    ce, sim = control_env(env), sim_of(env)
    names = list(steps[0]["obs_state"].get("_gt_scene_object_pos", {}))
    per = []
    for t in range(t_root):
        obs, *_ = ce.step(np.asarray(steps[t]["action"], dtype=np.float32))
        lg = steps[t + 1]["obs_state"]
        e = float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - lg["eef_pos"]))
        o = max((float(np.linalg.norm(sim.data.body_xpos[sim.model.body_name2id(k)] - np.asarray(lg["_gt_scene_object_pos"][k])))
                 for k in names), default=0.0)
        per.append({"t": t + 1, "eef": e, "obj": o})
    eef_e = max((p["eef"] for p in per), default=0.0)
    obj_e = max((p["obj"] for p in per), default=0.0)
    pos, rot = eef_pose(sim)
    lp = steps[t_root]["obs_state"]["eef_pos"]
    lq = steps[t_root - 1]["obs_state"]["eef_pos"] if t_root > 0 else lp
    objects = body_poses(sim, names)
    target = [o for o in ce.obj_of_interest if "bowl" in o]
    return Root(row=row_idx, t=t_root, env=env, ok=bool(eef_e <= tol and obj_e <= tol),
                max_eef_err_m=eef_e, max_obj_err_m=obj_e,
                worst_step=max(per, key=lambda p: max(p["eef"], p["obj"]))["t"] if per else 0,
                eef_pos=pos, eef_rot=rot, eef_vel=(np.asarray(lp) - np.asarray(lq)) / DT,
                gripper_qpos=np.asarray(steps[t_root]["obs_state"]["gripper_qpos"]),
                target=target[0] if target else "", objects=objects, per_step=per)


def restore(row_idx: int, t_root: int, tol: float = TOL_M, env=None) -> Root:
    """Replay row `row_idx` of R-047 up to (not including) action t_root.

    The returned env is at the state the policy saw at step t_root, before it
    applied action t_root (for a grasp root: before the faulty close command)."""
    row = r047_rows()[row_idx]
    steps = row["steps"]
    if not 0 <= t_root < len(steps):
        raise ValueError(f"t_root {t_root} outside 0..{len(steps) - 1}")
    env = env or make_env(scene_of(row))
    obs = env.reset(seed=int(row["seed"]), spec=spec_of(row["perturbation"]))
    per = []
    e, objs = _err(obs.state, steps[0]["obs_state"])
    per.append({"t": 0, "eef": e, "obj": max(objs.values(), default=0.0)})
    for t in range(t_root):
        obs, _succ, done, _ = env.step(Action(values=list(steps[t]["action"]), dims=list(row["action_dims"])))
        e, objs = _err(obs.state, steps[t + 1]["obs_state"])
        per.append({"t": t + 1, "eef": e, "obj": max(objs.values(), default=0.0)})
        if done and t + 1 < t_root:
            raise RuntimeError(f"env ended at t={t + 1} before the root t={t_root}")
    eef_e = max(p["eef"] for p in per)
    obj_e = max(p["obj"] for p in per)
    worst = max(per, key=lambda p: max(p["eef"], p["obj"]))["t"]
    sim = sim_of(env)
    pos, rot = eef_pose(sim)
    lp = steps[t_root]["obs_state"]["eef_pos"]
    lq = steps[t_root - 1]["obs_state"]["eef_pos"] if t_root > 0 else lp
    vel = (np.asarray(lp) - np.asarray(lq)) / DT
    names = [k[:-5] if k.endswith("_main") else k for k in steps[0]["obs_state"].get("_gt_scene_object_pos", {})]
    names = sorted(set(names) | {k[:-5] if k.endswith("_main") else k for k in steps[0]["obs_state"]["_gt_object_pos"]})
    objects = body_poses(sim, [n + "_main" for n in names]) or body_poses(sim, names)
    target = [o for o in control_env(env).obj_of_interest if "bowl" in o]
    return Root(row=row_idx, t=t_root, env=env, ok=bool(eef_e <= tol and obj_e <= tol),
                max_eef_err_m=eef_e, max_obj_err_m=obj_e, worst_step=worst,
                eef_pos=pos, eef_rot=rot, eef_vel=vel,
                gripper_qpos=np.asarray(steps[t_root]["obs_state"]["gripper_qpos"]),
                target=target[0] if target else "", objects=objects, per_step=per)
