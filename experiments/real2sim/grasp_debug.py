"""Why does a closed grasp not lift? (10-20: GR00T closes on the yellow block 4/4 in D2 but never lifts it.)
Re-simulates policy_batch episodes from their recorded actions (as placement_analysis.py: reproduces exactly) and,
at the first closed-gripper step near the target and while the gripper is closed, logs: fingertip point vs the
block (in the block's own frame), gripper driver position (0 open .. 1 closed), which finger pads touch the block
and the total contact normal force, and the block's height. Also the scripted expert's grasp of the same block
(make_tasks states) for comparison.

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/grasp_debug.py \
        --batch runs/real2sim/policy/batch_orange_stereo_v2 --task "put the yellow block in the bowl" --batch_seed_from_json
"""
import argparse, glob, json, os, sys
import numpy as np
import h5py
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_tasks import scene_tasks, randomise  # noqa: E402

FPS = 15


def contacts(m, d, body, pads):
    """(pad names touching `body`, total normal force N)."""
    touch, fsum = set(), 0.0
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if body in (b1, b2):
            other = b2 if b1 == body else b1
            name = m.body(other).name
            if other in pads:
                touch.add(name)
                mujoco.mj_contactForce(m, d, i, f6)
                fsum += abs(f6[0])
    return sorted(touch), fsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--scene", default="runs/real2sim/ep_orange_block_stereo:data/real2sim/droid/ep_orange_block")
    ap.add_argument("--tasks", default="runs/real2sim/tasks_stereo/ep_orange_block_stereo")
    ap.add_argument("--batch_seed_from_json", action="store_true")
    a = ap.parse_args()
    sdir, edir = a.scene.split(":")
    m = mujoco.MjModel.from_binary_path(os.path.join(sdir, "scene.mjb"))
    fit = json.load(open(os.path.join(sdir, "scene_fit.json")))
    tasks = scene_tasks(m, fit)
    ti = [i for i, t in enumerate(tasks) if t["text"] == a.task][0]
    task = tasks[ti]
    tb = m.body(task["target"]).id
    half = [m.geom_size[g] for g in range(m.ngeom) if m.geom_bodyid[g] == tb][0]
    pads = {b for b in range(m.nbody) if "pad" in m.body(b).name}
    pinch = m.site("g_pinch").id
    arm_q = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    arm_u = [m.actuator(f"fr3_joint{k}").id for k in range(1, 8)]
    grip_u = [i for i in range(m.nu) if m.actuator(i).name.startswith("g_")][0]
    g_lo, g_hi = m.actuator_ctrlrange[grip_u]
    drv = m.joint("g_right_driver_joint")
    drv_q, drv_max = drv.qposadr[0], m.jnt_range[drv.id][1]
    free = [m.body(m.jnt_bodyid[j]).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    q_start = h5py.File(os.path.join(edir, "trajectory.h5"))["observation/robot_state/joint_positions"][0]
    n_sub = int(round(1 / FPS / m.opt.timestep))
    lo, hi = m.actuator_ctrlrange[arm_u].T
    print(f"target {task['target']}: cube half-size {np.round(half * 100, 2)} cm; friction {m.geom_friction[[g for g in range(m.ngeom) if m.geom_bodyid[g] == tb][0]]}")
    out = []

    def report(tag, states_iter):
        first = None
        log = []
        for step, (grip_cmd) in states_iter:
            if not grip_cmd:
                continue
            R = d.xmat[tb].reshape(3, 3)
            rel = (d.site_xpos[pinch] - d.xpos[tb]) @ R  # fingertip point in the block frame
            touch, f = contacts(m, d, tb, pads)
            log.append(dict(step=step, rel_cm=np.round(rel * 100, 2).tolist(), driver=round(float(d.qpos[drv_q] / drv_max), 3),
                            pads=touch, normal_force_N=round(f, 2), block_z_cm=round(float(d.xpos[tb][2] * 100), 2)))
            if first is None and np.linalg.norm(rel[:2]) < 0.04:
                first = len(log) - 1
        if first is None:
            print(f"{tag}: gripper never closed near the block")
            return dict(tag=tag, closes=None)
        seg = log[first:first + 20]
        mid = seg[min(8, len(seg) - 1)]
        print(f"{tag}: first close at step {log[first]['step']}: fingertip vs block centre (block frame) {log[first]['rel_cm']} cm; "
              f"0.5 s later: driver {mid['driver']} (1 = fully closed), pads touching {mid['pads']}, normal force {mid['normal_force_N']} N, "
              f"block z {mid['block_z_cm']} cm (start {log[first]['block_z_cm']})")
        return dict(tag=tag, first_close=log[first], after_0p5s=mid)

    for rj in sorted(glob.glob(os.path.join(a.batch, "*", "v*", "rollout.json"))):
        r = json.load(open(rj))
        if r["instruction"] != a.task:
            continue
        z = np.load(rj.replace(".json", ".npz"))
        if "layout_rng_seed" in r:
            rs = r["layout_rng_seed"]
        elif a.batch_seed_from_json:
            vj = json.load(open(os.path.join(a.tasks, os.path.basename(os.path.dirname(os.path.dirname(rj))), f"v{r['variant']}.json")))
            rs = vj["seed"] * 1000 + ti * 100 + r["variant"]
        else:
            rs = ti * 100 + r["variant"]
        d = mujoco.MjData(m)
        d.qpos[arm_q] = q_start
        if r["variant"] > 0:
            randomise(m, d, np.random.default_rng(rs), free)
        d.ctrl[arm_u] = q_start
        d.ctrl[grip_u] = g_lo
        for _ in range(int(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)

        def it():
            for k, act in enumerate(z["action"]):
                d.ctrl[arm_u] = np.clip(act[10:17], lo, hi)
                d.ctrl[grip_u] = g_hi if act[9] > 0.5 else g_lo
                for _ in range(n_sub):
                    mujoco.mj_step(m, d)
                yield k, act[9] > 0.5
        out.append(report(f"GR00T v{r['variant']}", it()))
    # scripted expert, same task, variant 0 (its recorded states: kinematic replay is enough for the grasp geometry)
    st = np.load(glob.glob(os.path.join(a.tasks, f"{ti:02d}_*", "v0_states.npz"))[0])
    d = mujoco.MjData(m)

    def it_exp():
        for k, q in enumerate(st["qpos"]):
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            yield k, d.qpos[drv_q] / drv_max > 0.3
    out.append(report("expert v0", it_exp()))
    json.dump(dict(task=a.task, cube_half_cm=(half * 100).tolist(), results=out, status="exploratory, UNVERIFIED"),
              open(os.path.join(a.batch, f"grasp_debug_{task['target']}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
