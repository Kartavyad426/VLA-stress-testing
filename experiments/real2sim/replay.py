"""Real-to-sim step 3: replay a DROID episode's recorded joint trajectory in the rebuilt scene.

The arm tracks the recorded joint positions (15 Hz) through the FR3 position actuators, the gripper
follows the recorded gripper position, and contacts are simulated, so the block only reaches the bowl
if the rebuilt geometry lets the real motion work. Renders the sim from the calibrated camera next to
the real video.

Writes <out>/replay.mp4 (real | sim | 50/50 blend) and <out>/replay_metrics.json.

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/replay.py \
        --episode data/real2sim/droid/ep_orange_block --cam 22008760 --out runs/real2sim/ep_orange_block
"""
import argparse, json, os
import numpy as np
import cv2
import h5py
import imageio.v2 as imageio
import mujoco

FPS = 15


def read_video(path):
    cap, frames = cv2.VideoCapture(path), []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="orange_block")
    ap.add_argument("--settle_s", type=float, default=0.5)
    a = ap.parse_args()

    m = mujoco.MjModel.from_binary_path(os.path.join(a.out, "scene.mjb"))
    d = mujoco.MjData(m)
    f = h5py.File(os.path.join(a.episode, "trajectory.h5"))
    jp = f["observation/robot_state/joint_positions"][()]
    # gripper: the commanded target. DROID closes to 1.0 and the object stops the fingers; the observed
    # position only says where they stopped, so driving the sim with it never squeezes.
    gp = f["action/target_gripper_position"][()]
    ee = f["observation/robot_state/cartesian_position"][()][:, :3]
    fit = json.load(open(os.path.join(a.out, "scene_fit.json")))

    arm_q = [m.joint(f"fr3_joint{i}").qposadr[0] for i in range(1, 8)]
    arm_u = [m.actuator(f"fr3_joint{i}").id for i in range(1, 8)]
    grip_u = [i for i in range(m.nu) if m.actuator(i).name.startswith("g_")]
    assert len(grip_u) == 1, [m.actuator(i).name for i in range(m.nu)]
    grip_u = grip_u[0]
    g_lo, g_hi = m.actuator_ctrlrange[grip_u]
    pinch = m.site("g_pinch").id
    tgt = m.body(a.target).id
    free_bodies = [m.body(m.jnt_bodyid[j]).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    bowl = m.body([n for n in free_bodies if "bowl" in n][0]).id

    def command(i):
        d.ctrl[arm_u] = jp[i]
        d.ctrl[grip_u] = g_lo + gp[i] * (g_hi - g_lo)

    d.qpos[arm_q] = jp[0]
    command(0)
    mujoco.mj_forward(m, d)
    p0 = {n: d.xpos[m.body(n).id].copy() for n in free_bodies}  # every free object, whatever the scene
    for _ in range(int(a.settle_s / m.opt.timestep)):
        mujoco.mj_step(m, d)
    settle = {n: float(np.linalg.norm(d.xpos[m.body(n).id] - p)) for n, p in p0.items()}

    real = read_video(os.path.join(a.episode, f"{a.cam}.mp4"))
    H, W = real[0].shape[:2]
    rend = mujoco.Renderer(m, H, W)
    n_sub = int(round(1 / FPS / m.opt.timestep))
    writer = imageio.get_writer(os.path.join(a.out, "replay.mp4"), fps=FPS, quality=7, macro_block_size=8)
    pinch_err, tz, tpos, states = [], [], [], []
    T = min(len(jp), len(real))
    for i in range(T):
        command(i)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        states.append(d.qpos.copy())
        pinch_err.append(d.site_xpos[pinch] - ee[i])
        tpos.append(d.xpos[tgt].copy())
        rend.update_scene(d, camera="ext")
        sim = rend.render()
        blend = (0.5 * real[i] + 0.5 * sim).astype(np.uint8)
        row = np.concatenate([real[i], sim, blend], 1)
        row = cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2), interpolation=cv2.INTER_AREA)
        for x, label in ((10, "real"), (W // 2 + 10, "sim (rebuilt)"), (W + 10, "overlay")):
            cv2.putText(row, label, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.append_data(row)
    writer.close()
    np.savez_compressed(os.path.join(a.out, "replay_states.npz"), qpos=np.array(states), real_frames=np.arange(T))

    tpos = np.array(tpos)
    pe = np.array(pinch_err)
    bowl_xy, bowl_z = d.xpos[bowl][:2], d.xpos[bowl][2]
    bowl_fit = [o for o in fit["objects"].values() if o.get("kind", "bowl") == "bowl" and "rim_radius" in o][0]
    r = bowl_fit["rim_radius"]
    final = tpos[-1]
    horiz = float(np.linalg.norm(final[:2] - bowl_xy))
    metrics = dict(
        steps=T, settle_displacement_m=settle,
        target_lifted_m=float(tpos[:, 2].max() - tpos[0, 2]),
        target_final_xyz=final.tolist(), bowl_final_xyz=d.xpos[bowl].tolist(),
        target_to_bowl_horizontal_m=horiz, bowl_rim_radius_m=r,
        target_in_bowl=bool(horiz < 0.8 * r and final[2] < bowl_z + bowl_fit["height_fit"] + 0.02),
        pinch_minus_recorded_ee_mean_m=pe.mean(0).tolist(), pinch_minus_recorded_ee_std_m=pe.std(0).tolist(),
    )
    json.dump(metrics, open(os.path.join(a.out, "replay_metrics.json"), "w"), indent=1)
    print(json.dumps({k: metrics[k] for k in ("target_lifted_m", "target_to_bowl_horizontal_m", "target_in_bowl",
                                              "settle_displacement_m", "pinch_minus_recorded_ee_std_m")}))


if __name__ == "__main__":
    main()
