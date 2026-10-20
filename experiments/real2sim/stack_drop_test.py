"""Stacking drop test (10-20): in D2, GR00T released the green block 0.5-2 cm off the orange block's centre but
3-6.6 cm above it, and the block didn't stay. Is that our contact physics or a genuinely hard drop?
Drops the target block (robot parked at its frame-0 pose, out of the way) from a grid of heights above the lower
block's top and horizontal offsets, with 0 / 30 deg yaw, lets it settle 2 s and applies make_tasks.task_success.
Also reports where the scripted expert lets go when it stacks (height of the block's bottom above the lower block's
top at the gripper opening, from its recorded states).

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/stack_drop_test.py \
        --scene runs/real2sim/ep_orange_block_stereo:data/real2sim/droid/ep_orange_block --task "stack the green block on the orange block"
"""
import argparse, glob, json, os, sys
import numpy as np
import h5py
import mujoco
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_tasks import scene_tasks, task_success, free_qadr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--tasks", default="runs/real2sim/tasks_stereo/ep_orange_block_stereo")
    a = ap.parse_args()
    sdir, edir = a.scene.split(":")
    m = mujoco.MjModel.from_binary_path(os.path.join(sdir, "scene.mjb"))
    fit = json.load(open(os.path.join(sdir, "scene_fit.json")))
    tasks = scene_tasks(m, fit)
    ti = [i for i, t in enumerate(tasks) if t["text"] == a.task][0]
    task = tasks[ti]
    tb, rb = m.body(task["target"]).id, m.body(task["ref"]).id
    th = [m.geom_size[g][2] for g in range(m.ngeom) if m.geom_bodyid[g] == tb][0]
    rh = [m.geom_size[g][2] for g in range(m.ngeom) if m.geom_bodyid[g] == rb][0]
    arm_q = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    arm_u = [m.actuator(f"fr3_joint{k}").id for k in range(1, 8)]
    q0 = h5py.File(os.path.join(edir, "trajectory.h5"))["observation/robot_state/joint_positions"][0]
    print(f"target {task['target']} edge {2*th*100:.1f} cm on {task['ref']} edge {2*rh*100:.1f} cm; block friction "
          f"{m.geom_friction[[g for g in range(m.ngeom) if m.geom_bodyid[g] == tb][0]].tolist()}, solref {m.geom_solref[[g for g in range(m.ngeom) if m.geom_bodyid[g] == tb][0]].tolist()}")
    res = []
    for yaw in (0.0, 30.0):
        for off in (0.0, 0.005, 0.01, 0.02):
            for hgt in (0.002, 0.005, 0.01, 0.02, 0.03, 0.045, 0.065):
                d = mujoco.MjData(m)
                d.qpos[arm_q] = q0
                d.ctrl[arm_u] = q0
                mujoco.mj_forward(m, d)
                pr = d.xpos[rb].copy()
                qa = free_qadr(m, task["target"])
                d.qpos[qa:qa + 3] = [pr[0] + off, pr[1], pr[2] + rh + th + hgt]
                d.qpos[qa + 3:qa + 7] = Rot.from_euler("z", yaw, degrees=True).as_quat(scalar_first=True)
                d.qvel[:] = 0
                for _ in range(int(2.0 / m.opt.timestep)):
                    mujoco.mj_step(m, d)
                ok = task_success(m, d, fit, task)
                res.append(dict(yaw_deg=yaw, offset_cm=off * 100, drop_cm=hgt * 100, success=bool(ok)))
    # expert's release heights for this task (recorded states)
    rel = []
    drv = m.joint("g_right_driver_joint")
    for st in sorted(glob.glob(os.path.join(a.tasks, f"{ti:02d}_*", "v*_states.npz"))):
        q = np.load(st)["qpos"]
        dd = mujoco.MjData(m)
        g = q[:, drv.qposadr[0]] / m.jnt_range[drv.id][1]
        closed = np.nonzero(g > 0.3)[0]
        if not len(closed):
            continue
        k = int(closed[-1]) + 1
        dd.qpos[:] = q[min(k, len(q) - 1)]
        mujoco.mj_forward(m, dd)
        rel.append(float((dd.xpos[tb][2] - th) - (dd.xpos[rb][2] + rh)) * 100)
    out = dict(task=a.task, drops=res, expert_release_gap_cm=rel, status="exploratory, UNVERIFIED")
    json.dump(out, open(os.path.join(sdir, f"stack_drop_{task['target']}_on_{task['ref']}.json"), "w"), indent=1)
    for yaw in (0.0, 30.0):
        print(f"yaw {yaw:.0f} deg: rows = offset (cm), cols = drop height above the lower block (cm)")
        hs = sorted({r["drop_cm"] for r in res})
        print("        " + " ".join(f"{h:5.1f}" for h in hs))
        for off in sorted({r["offset_cm"] for r in res}):
            print(f"  {off:4.1f}  " + " ".join("  ok " if [r for r in res if r["yaw_deg"] == yaw and r["offset_cm"] == off and r["drop_cm"] == h][0]["success"] else " FAIL" for h in hs))
    print("expert releases the block with a gap of (cm):", np.round(rel, 2).tolist())


if __name__ == "__main__":
    main()
