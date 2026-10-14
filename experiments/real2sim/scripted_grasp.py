"""Real-to-sim check: can the rebuilt scene's physics support a pick-and-place when the actions are made
for the sim scene itself (not replayed from the real robot)?

Scripted expert: inverse kinematics (damped least squares on the Robotiq pinch site) to waypoints computed
from the sim's own block and bowl poses: above block -> down to block centre -> close -> lift -> above bowl
-> lower -> open. Joint targets go through the FR3 position actuators with physics on (contacts, friction).

Writes <out>/scripted_grasp.mp4 (sim, calibrated camera, real first frame alongside) and
<out>/scripted_grasp_metrics.json.

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/scripted_grasp.py \
        --episode data/real2sim/droid/ep_orange_block --cam 22008760 --out runs/real2sim/ep_orange_block
"""
import argparse, json, os, sys
import numpy as np
import cv2
import h5py
import imageio.v2 as imageio
import mujoco
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perceive import read_frame  # noqa: E402

FPS = 15


def ik(m, q0, site, p_tgt, R_tgt, arm_q, arm_dof, iters=300, damp=1e-3):
    """Joint angles putting `site` at (p_tgt, R_tgt); kinematics only, on a scratch MjData."""
    d = mujoco.MjData(m)
    d.qpos[arm_q] = q0
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    lo, hi = m.jnt_range[[m.dof_jntid[k] for k in arm_dof]].T
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        e_p = p_tgt - d.site_xpos[site]
        e_r = Rot.from_matrix(R_tgt @ d.site_xmat[site].reshape(3, 3).T).as_rotvec()
        err = np.r_[e_p, 0.5 * e_r]
        if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-3:
            break
        mujoco.mj_jacSite(m, d, jacp, jacr, site)
        J = np.r_[jacp[:, arm_dof], 0.5 * jacr[:, arm_dof]]
        dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)
        d.qpos[arm_q] = np.clip(d.qpos[arm_q] + dq, lo, hi)
    mujoco.mj_kinematics(m, d)
    return d.qpos[arm_q].copy(), float(np.linalg.norm(p_tgt - d.site_xpos[site]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="orange_block", help="single target body (kept for the orange-block runs)")
    ap.add_argument("--targets", default="", help="comma-separated bodies picked in sequence; 'auto:N' = the N blocks "
                    "with the most clearance from other objects, most isolated first")
    ap.add_argument("--tag", default="scripted_grasp", help="output name prefix")
    a = ap.parse_args()

    m = mujoco.MjModel.from_binary_path(os.path.join(a.out, "scene.mjb"))
    d = mujoco.MjData(m)
    f = h5py.File(os.path.join(a.episode, "trajectory.h5"))
    q_start = f["observation/robot_state/joint_positions"][0]
    arm_q = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    arm_dof = [m.joint(f"fr3_joint{k}").dofadr[0] for k in range(1, 8)]
    arm_u = [m.actuator(f"fr3_joint{k}").id for k in range(1, 8)]
    grip_u = [i for i in range(m.nu) if m.actuator(i).name.startswith("g_")][0]
    g_lo, g_hi = m.actuator_ctrlrange[grip_u]
    pinch = m.site("g_pinch").id
    fit = json.load(open(os.path.join(a.out, "scene_fit.json")))
    free = [m.body(m.jnt_bodyid[j]).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    bowl_name = [n for n in free if "bowl" in n][0]
    bowl = m.body(bowl_name).id
    bowl_fit = [o for o in fit["objects"].values() if o.get("kind", "bowl") == "bowl" and "rim_radius" in o][0]
    rim_top = fit["table"]["z"] + bowl_fit["height_asset_scaled"]

    # settle the scene with the arm at the recorded start pose
    d.qpos[arm_q] = q_start
    d.ctrl[arm_u] = q_start
    d.ctrl[grip_u] = g_lo
    for _ in range(int(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    start_pos = {n: d.xpos[m.body(n).id].copy() for n in free}

    if a.targets.startswith("auto:"):
        n_pick = int(a.targets.split(":")[1])
        blocks = [n for n in free if "block" in n]
        others = {n: d.xpos[m.body(n).id][:2] for n in free}
        clear = {b: min(np.linalg.norm(others[b] - others[o]) for o in free if o != b) for b in blocks}
        targets = sorted(blocks, key=lambda b: -clear[b])[:n_pick]
        print("auto targets (clearance cm):", [(t, round(100 * clear[t], 1)) for t in targets])
    elif a.targets:
        targets = [t.strip() for t in a.targets.split(",")]
    else:
        targets = [a.target]

    H, W = 720, 1280
    rend = mujoco.Renderer(m, H, W)
    real0 = read_frame(os.path.join(a.episode, f"{a.cam}.mp4"), 0)
    writer = imageio.get_writer(os.path.join(a.out, f"{a.tag}.mp4"), fps=FPS, quality=7, macro_block_size=8)
    n_sub = int(round(1 / FPS / m.opt.timestep))
    states, picks = [], []
    z_down = np.array([0, 0, -1.0])
    for ti, tgt in enumerate(targets):
        blk = m.body(tgt).id
        blk_geom = [g for g in range(m.ngeom) if m.geom_bodyid[g] == blk][0]
        p_blk = d.xpos[blk].copy()  # planned from the CURRENT state: earlier picks may have nudged things
        R_blk = d.xmat[blk].reshape(3, 3)
        half = m.geom_size[blk_geom]
        # close across the block's narrower horizontal side: the 2F-85 fingers move along the pinch frame's y axis
        ax = R_blk[:, 0] if half[0] < half[1] else R_blk[:, 1]
        ax = np.array([ax[0], ax[1], 0.0])
        ax /= np.linalg.norm(ax)
        R_grasp = np.c_[np.cross(ax, z_down), ax, z_down]  # columns: x, y (closing), z (approach)
        p_bowl = d.xpos[bowl].copy()
        drop = rim_top + 0.05 + 0.03 * ti  # drop a little higher each time as the bowl fills
        waypoints = [  # (name, pinch position, gripper ctrl, seconds)
            ("above block", p_blk + [0, 0, 0.12], g_lo, 2.0),
            ("down to block", p_blk + [0, 0, 0.0], g_lo, 1.5),
            ("close", p_blk + [0, 0, 0.0], g_hi, 1.0),
            ("lift", p_blk + [0, 0, 0.15], g_hi, 1.5),
            ("above bowl", np.r_[p_bowl[:2], rim_top + 0.15], g_hi, 2.0),
            ("lower", np.r_[p_bowl[:2], drop], g_hi, 1.0),
            ("open", np.r_[p_bowl[:2], drop], g_lo, 1.0),
            ("retreat", np.r_[p_bowl[:2], rim_top + 0.20], g_lo, 1.5),
        ]
        q_prev, ik_err, z_blk = d.qpos[arm_q].copy(), {}, []
        others_before = {n: d.xpos[m.body(n).id].copy() for n in free if n != tgt}
        for name, p, g, secs in waypoints:
            q, e = ik(m, q_prev, pinch, p, R_grasp, arm_q, arm_dof)
            ik_err[name] = e
            n = int(secs * FPS)
            for k in range(n):
                sm = (k + 1) / n
                d.ctrl[arm_u] = q_prev + (q - q_prev) * (0.5 - 0.5 * np.cos(np.pi * sm))  # smooth interpolation
                d.ctrl[grip_u] = g
                for _ in range(n_sub):
                    mujoco.mj_step(m, d)
                z_blk.append(d.xpos[blk][2])
                states.append(d.qpos.copy())
                rend.update_scene(d, camera="ext")
                row = np.concatenate([real0, rend.render()], 1)
                row = cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2), interpolation=cv2.INTER_AREA)
                cv2.putText(row, "real (first frame only)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(row, f"sim pick {ti + 1}/{len(targets)} {tgt}: {name}", (W // 2 + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                writer.append_data(row)
            q_prev = q
        final = d.xpos[blk]
        horiz = float(np.linalg.norm(final[:2] - d.xpos[bowl][:2]))
        moved = {n: float(np.linalg.norm(d.xpos[m.body(n).id] - p0)) for n, p0 in others_before.items()}
        picks.append(dict(target=tgt, block_start=p_blk.tolist(), block_final=final.tolist(), block_size_m=(2 * half).tolist(),
                          lifted_m=float(max(z_blk) - p_blk[2]), block_to_bowl_horizontal_m=horiz,
                          in_bowl=bool(horiz < 0.8 * bowl_fit["rim_radius"] and final[2] < rim_top),
                          max_ik_residual_m=float(max(ik_err.values())),
                          others_moved_m={k: v for k, v in moved.items() if v > 0.005}))
    writer.close()
    # the scripted run has no matching real frames after frame 0: the composite uses the real background only
    np.savez_compressed(os.path.join(a.out, f"{a.tag}_states.npz"), qpos=np.array(states), real_frames=np.zeros(len(states), int))
    metrics = dict(targets=targets, picks=picks, n_in_bowl=sum(p["in_bowl"] for p in picks),
                   bowl_moved_m=float(np.linalg.norm(d.xpos[bowl] - start_pos[bowl_name])))
    if len(picks) == 1:  # keep the single-target fields the orange-block runs report
        metrics.update({k: picks[0][k] for k in ("lifted_m", "block_to_bowl_horizontal_m", "in_bowl")})
    json.dump(metrics, open(os.path.join(a.out, f"{a.tag}_metrics.json"), "w"), indent=1)
    print(json.dumps(dict(n_in_bowl=metrics["n_in_bowl"], of=len(picks), bowl_moved_cm=round(100 * metrics["bowl_moved_m"], 1),
                          picks=[(p["target"], round(100 * p["lifted_m"], 1), p["in_bowl"], {k: round(100 * v, 1) for k, v in p["others_moved_m"].items()}) for p in picks])))


if __name__ == "__main__":
    main()
