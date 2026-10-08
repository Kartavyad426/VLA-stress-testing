"""PoC toy (mining spec walkthrough): shared helpers, CPU only.

Both pools are put in ONE frame before anything is compared:
  * our rollouts store eef_pos and per-object positions, but their eef_quat is
    in a different convention from MuJoCo's site frame, so gripper orientation
    is recomputed by forward kinematics from joint_pos in the demo's own model;
  * the raw LIBERO demos store the flattened sim state, not object poses, so
    poses are read by setting qpos in the demo's MJCF (no env, no rendering).

The MJCF in each demo file points at the authors' machine; asset paths are
remapped to the local LIBERO-plus and robosuite copies.
"""
import os

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data/libero_raw/libero_spatial")
TASK_FILE = ("pick_up_the_black_bowl_between_the_plate_and_the_ramekin_"
             "and_place_it_on_the_plate_demo.hdf5")
R047 = os.path.join(ROOT, "runs/r047/rollouts.jsonl")
TASK_ID = 984
TARGET = "akita_black_bowl_1_main"
NEIGHBOURS = ["akita_black_bowl_2_main", "cookies_1_main",
              "glazed_rim_porcelain_ramekin_1_main", "plate_1_main"]
SITE = "gripper0_grip_site"
_REMAP = [("/Users/yifengz/workspace/libero-dev/chiliocosm/assets",
           os.path.join(ROOT, "third_party/LIBERO-plus/libero/libero/assets")),
          ("/Users/yifengz/workspace/robosuite-master/robosuite",
           os.path.join(ROOT, ".venvs/libero-plus/lib/python3.12/site-packages/robosuite"))]

# Success criterion (one template, grasp phase): the target is lifted at least
# LIFT_M above its resting height within WINDOW steps of grasp onset.
LIFT_M, WINDOW = 0.05, 40


def load_model(xml):
    for a, b in _REMAP:
        xml = xml.replace(a, b)
    m = mujoco.MjModel.from_xml_string(xml)
    return m, mujoco.MjData(m)


def site_rot(m, D):
    return D.site(SITE).xmat.reshape(3, 3).copy()


def fk_rot(m, D, joint_pos):
    """Gripper site rotation for a 7-joint Panda configuration."""
    D.qpos[:7] = joint_pos
    mujoco.mj_kinematics(m, D)
    return site_rot(m, D), D.site(SITE).xpos.copy()


def angle_deg(R1, R2):
    c = (np.trace(R1.T @ R2) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def tilt_deg(R):
    """Angle of the gripper's approach axis (site z) from straight down."""
    return float(np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1))))


def grasp_events(cmd_closed, target_z, eef, objs, rots, aperture=None):
    """Every open->close command transition, with the success criterion checked.

    cmd_closed[t]: bool; target_z[t]: target height; eef[t]: xyz;
    objs[t]: {name: xyz}; rots[t]: 3x3 gripper rotation.
    """
    z0 = target_z[0]
    ev = []
    for t in range(1, len(cmd_closed)):
        if cmd_closed[t] and not cmd_closed[t - 1]:
            w = target_z[t:t + WINDOW + 1]
            lift = float(max(w) - z0)
            tgt = np.asarray(objs[t][TARGET])
            ev.append({
                "t": t,
                "offset_m": (np.asarray(eef[t]) - tgt).round(4).tolist(),
                "target_xyz": tgt.round(4).tolist(),
                "neighbours_rel_m": {n: (np.asarray(objs[t][n]) - tgt).round(4).tolist()
                                     for n in NEIGHBOURS if n in objs[t]},
                "tilt_deg": round(tilt_deg(rots[t]), 1),
                "_R": rots[t],
                "aperture_m": None if aperture is None else round(float(aperture[t]), 4),
                "max_lift_m": round(lift, 4),
                "success": lift >= LIFT_M,
            })
    return ev


def kin_model():
    """A model to run forward kinematics in (demo 0's scene; robot is identical)."""
    import h5py
    with h5py.File(os.path.join(RAW, TASK_FILE), "r") as f:
        return load_model(f["data/demo_0"].attrs["model_file"])


def rollout_events(row, m, D):
    """Grasp events for one stored rollout (a dict from rollouts.jsonl)."""
    S = row["steps"]
    cmd = [bool(s.get("action")) and s["action"][6] > 0.5 for s in S]
    objs = [s["obs_state"]["_gt_scene_object_pos"] for s in S]
    tz = [o[TARGET][2] for o in objs]
    eef = [s["obs_state"]["eef_pos"] for s in S]
    rots, fkpos = [], []
    for s in S:
        R, p = fk_rot(m, D, s["obs_state"]["joint_pos"])
        rots.append(R); fkpos.append(p)
    ap = [sum(abs(x) for x in s["obs_state"]["gripper_qpos"]) for s in S]
    ev = grasp_events(cmd, tz, eef, objs, rots, ap)
    # FK check: the recomputed site must sit where the trace says the eef is.
    fk_err = max(float(np.linalg.norm(np.asarray(a) - b)) for a, b in zip(eef, fkpos))
    return ev, fk_err
