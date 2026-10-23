"""Human-demo statistics that set the scripted expert's waypoints. CPU.

Every number the planner uses for where to grasp, how high to go and where to
put the bowl down comes from the 482 usable LIBERO-Spatial demos (DMP's demo
loader and phase split, demos.py), not from hand tuning:

  grasp[side]       EEF minus bowl at the final grasp (median) and the EEF
                    orientation there (rotation mean), split by the rim side the
                    hand took: +y (341 demos) or -y (141). Humans grip the rim.
  pregrasp_dz_m     EEF height above its grasp height 10 steps before the grasp
                    (median): the descent onto the rim starts this high
  lift_m            how far the bowl rose during the carry (median)
  place_offset_m    bowl minus plate at the release (median)
  tilt_carry_p95    95th percentile over demos of the bowl's largest tilt during
                    the carry; a held bowl tilted more than this is set down
                    and regrasped
  speed_p99         execute.demo_stats(): the keep check's speed limit
  speed_med         median EEF speed over the approach and carry phases
  action_pos_max,   largest |action| over all demo steps, position and rotation
  action_rot_max    dims (the human action range; Expert.action keeps to it)
  speed_p90         its 90th percentile: the human-speed law's cruise cap (10-20; a
                    0.275 m/s cruise saturated the OSC action in D_backup, humans
                    never saturate it: 0 of the demo steps)
  omega_p99         99th percentile EEF rotation rate (rad/s), same phases
  by_task[task]     human-speed law targets (10-20), medians over the task's demos,
                    phases as in phase_timing.py:
                      approach_speed  EEF path / time, start of phase A -> close command
                      carry_speed     EEF path / time, lift-off (bowl +LIFTOFF_M) -> release
                      close_hold      steps from the close command until the EEF has
                                      risen RISE_M (when the human starts lifting)

nearest_grasp() is the "nearest" variant: the grasp of the demo whose approach
passed closest to the root's hand-relative-to-bowl position.

Cached to runs/scripted_expert/demo_params.json.

UNVERIFIED: new numeric code, not yet checked by val.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from dmp_link import WORKTREE, demos as D, execute as E

OUT = os.path.join(WORKTREE, "runs/scripted_expert/demo_params.json")
PRE_STEPS = 10
LIFTOFF_M = 0.01     # as phase_timing.LIFTOFF_M
RISE_M = 0.005       # "started lifting": EEF this far above its height at the close


def side_of(offset) -> str:
    return "+y" if offset[1] > 0 else "-y"


def _tilt_deg(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0))))


def _carry_tilts(task, usable):
    """Largest bowl tilt over each usable demo's carry (needs the sim states)."""
    env = D.control_env(task)
    sim = env.sim
    tgt = [o for o in env.obj_of_interest if "bowl" in o][0]
    bid = sim.model.body_name2id(tgt + "_main")
    out = []
    with h5py.File(D.demo_file(task), "r") as f:
        for k in f["data"]:
            i = int(k.split("_")[1])
            if i not in usable:
                continue
            S = f["data"][k]["states"][:]
            _, B, _ = usable[i].phases
            ts = []
            for t in range(B[0], B[1]):
                env.set_state(S[t])
                sim.forward()
                ts.append(_tilt_deg(sim.data.body_xquat[bid]))
            out.append(max(ts))
    env.close()
    return out


def _path_speed(P):
    n = len(P) - 1
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() / (n * E.DT)) if n > 0 else 0.0


def _task_speeds(demos):
    app, car, hold = [], [], []
    for d in demos:
        A, B, _ = d.phases
        g, rel = B[0], B[1]
        app.append(_path_speed(d.eef_pos[A[0]:g + 1]))
        z = d.target_pos[:, 2]
        lift = next((t for t in range(g, rel) if z[t] - z[g] >= LIFTOFF_M), rel)
        car.append(_path_speed(d.eef_pos[lift:rel + 1]))
        hold.append(next((t - g for t in range(g, rel) if d.eef_pos[t, 2] - d.eef_pos[g, 2] >= RISE_M), rel - g))
    return {"n": len(app), "approach_speed": float(np.median(app)), "carry_speed": float(np.median(car)),
            "close_hold": int(np.median(hold))}


def _action_range():
    acts = []
    for task in D.tasks():
        with h5py.File(D.demo_file(task), "r") as f:
            acts += [f["data"][k]["actions"][:] for k in f["data"]]
    A = np.abs(np.concatenate(acts))
    return {"action_pos_max": float(A[:, :3].max()), "action_rot_max": float(A[:, 3:6].max())}


def build():
    grasp = {"+y": {"off": [], "quat": []}, "-y": {"off": [], "quat": []}}
    pre_dz, lift, place, speeds, omegas, tilts = [], [], [], [], [], []
    by_task = {}
    for task in D.tasks():
        usable = {d.idx: d for d in D.load(task) if d.usable}
        for d in usable.values():
            A, B, _ = d.phases
            g, rel = B[0], B[1]
            off = d.eef_pos[g] - d.target_pos[g]
            grasp[side_of(off)]["off"].append(off)
            grasp[side_of(off)]["quat"].append(d.eef_quat[g])
            pre_dz.append(d.eef_pos[max(A[0], g - PRE_STEPS), 2] - d.eef_pos[g, 2])
            lift.append(d.lift(*B))
            place.append(d.target_pos[rel] - d.plate_pos[rel])
            speeds += list(np.linalg.norm(np.diff(d.eef_pos[A[0]:B[1]], axis=0), axis=1) / E.DT)
            Rs = Rotation.from_quat(d.eef_quat[A[0]:B[1]])
            omegas += list((Rs[1:] * Rs[:-1].inv()).magnitude() / E.DT)
        tilts += _carry_tilts(task, usable)
        by_task[task] = _task_speeds(usable.values())
    p = {
        "tag": "UNVERIFIED",
        "n_demos": int(sum(len(v["off"]) for v in grasp.values())),
        "grasp": {s: {"n": len(v["off"]), "offset_m": np.median(v["off"], axis=0).tolist(),
                      "quat_xyzw": Rotation.from_quat(v["quat"]).mean().as_quat().tolist()}
                  for s, v in grasp.items()},
        "pregrasp_dz_m": float(np.median(pre_dz)),
        "lift_m": float(np.median(lift)),
        "place_offset_m": np.median(place, axis=0).tolist(),
        "tilt_carry_p95_deg": float(np.percentile(tilts, 95)),
        "speed_med": float(np.median(speeds)),
        "speed_p90": float(np.percentile(speeds, 90)),
        "omega_p99": float(np.percentile(omegas, 99)),
        **E.demo_stats(),
        "by_task": by_task,
        **_action_range(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(p, f, indent=1)
    return p


@lru_cache(maxsize=1)
def params() -> dict:
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return build()


def grasp_of(side: str):
    """(EEF-minus-bowl offset, EEF orientation) of the median human grasp on a rim side."""
    g = params()["grasp"][side]
    return np.array(g["offset_m"]), Rotation.from_quat(g["quat_xyzw"])


def nearest_grasp(task: str, eef_minus_bowl):
    """Grasp of the demo whose approach came closest to `eef_minus_bowl`
    (the root's hand position relative to the bowl). Returns (offset, rotation, demo idx)."""
    best = None
    for d in D.load(task):
        if not d.usable:
            continue
        A, B, _ = d.phases
        rel = d.eef_pos[A[0]:B[0]] - d.target_pos[A[0]:B[0]]
        dist = float(np.linalg.norm(rel - eef_minus_bowl, axis=1).min())
        if best is None or dist < best[0]:
            best = (dist, d)
    d = best[1]
    g = d.phases[1][0]
    return d.eef_pos[g] - d.target_pos[g], Rotation.from_quat(d.eef_quat[g]), d.idx


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
