"""Which object plays which role in a built scene, read from the scene and the recording instead of names (10-26).

Until 10-26 several scripts found roles by name: the replay/scripted-grasp target defaulted to 'orange_block', the
container was the free body whose name contains 'bowl', and the table was the perception mask 'wooden table'. Here:
  table_label(perception_json)   the table's mask label: perception.json 'roles' (perceive.py from 10-26), else the
                                 label every scene before 10-26 used ('wooden table')
  container_bodies(m, fit)       free bodies fitted as containers (scene_fit kind 'bowl', which build_scene keeps for
                                 pack bowls too), not bodies whose name contains 'bowl'
  first_grasp_target(m, fit, episode)  the non-container object nearest the fingertip at the first recorded gripper
                                 close (the same rule as scene_checks #3)
Exploratory, UNVERIFIED.
"""
import os
import numpy as np
import h5py
import mujoco

LEGACY_TABLE = "wooden table"  # the label of every scene built before 10-26


def table_label(per_info):
    return (per_info or {}).get("roles", {}).get("table", LEGACY_TABLE)


def gripper_label(per_info):
    return (per_info or {}).get("roles", {}).get("gripper", "robot gripper")


def _body(lab, o):
    return o.get("body", lab.replace("#", "_").replace(" ", "_"))  # older fits: label-derived body names (composite)


def container_bodies(m, fit):
    """Body ids of the fitted containers present in the model."""
    names = {m.body(i).name for i in range(m.nbody)}
    return [m.body(_body(lab, o)).id for lab, o in fit["objects"].items() if o.get("kind") == "bowl" and _body(lab, o) in names]


def container_fit(fit, body_name):
    return next((o for lab, o in fit["objects"].items() if _body(lab, o) == body_name), None)


def first_grasp_target(m, fit, episode):
    """Name of the non-container free body nearest the fingertip (g_pinch) at the first recorded close; None if the
    episode never closes the gripper."""
    from episode_info import grasp_events
    h = h5py.File(os.path.join(episode, "trajectory.h5"))
    ev = grasp_events(h["observation/robot_state/gripper_position"][:])
    if not ev:
        return None
    J = h["observation/robot_state/joint_positions"][:]
    arm = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    d.qpos[arm] = J[ev[0][0]]
    mujoco.mj_forward(m, d)
    p = d.site_xpos[m.site("g_pinch").id]
    cont = set(container_bodies(m, fit))
    free = [m.jnt_bodyid[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[j] not in cont]
    if not free:
        return None
    return m.body(min(free, key=lambda b: np.linalg.norm(d.xpos[b] - p))).name


def body_kinds(fit):
    """body name -> fitted kind ('box', 'bowl', 'ellipsoid', 'asset'), plus 'cube' for boxes with equal sides."""
    out = {}
    for lab, o in fit["objects"].items():
        k = o.get("kind")
        if k == "box" and len(set(o.get("half", []))) == 1:
            k = "cube"
        out[_body(lab, o)] = k
    return out
