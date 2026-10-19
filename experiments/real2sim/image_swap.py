"""Pre-deploy checklist #20: image swap. At recorded states of the real episode's grasp approach, query the policy with
the SAME state and different images, and compare what it plans:
  real_real  real exterior + real wrist frame (as G1)
  sim_sim    our composite exterior + textured wrist render (sim robot posed at the recorded joints)
  sim_real   our exterior + real wrist        real_sim  real exterior + our wrist
  real_fk    real images, state computed in sim by FK of the recorded joints (as policy_rollout does) instead of the
             recorded cartesian_position: rules a state difference in or out
Per query: the planned fingertip point (FK of the joint head) over the executed 15 steps: lowest height above the
rebuilt table, mean tilt from vertical, max gripper. If sim images push the plan lower/more tilted than real images at
the same state, our images cause the in-sim approach difference (grasp_compare.html).

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/image_swap.py \
        --scene runs/real2sim/ep_orange_block_stereo --episode data/real2sim/droid/ep_orange_block \
        --out runs/real2sim/policy/image_swap_orange [--policy hold for a CPU dry run]
"""
import argparse, json, os, sys
import numpy as np
import cv2
import h5py
import mujoco
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from policy_rollout import GrootClient, HoldPolicy, compute_eef_9d, IMG, H, W, DEFAULT_CKPT, REPO  # noqa: E402
from composite import Compositor, WristCompositor, load_plate, wrist_plate  # noqa: E402
from build_scene import FLANGE_IN_LINK7  # noqa: E402

CONDS = ("real_real", "sim_sim", "sim_real", "real_sim", "real_fk")
# G4 (10-19): which part of the sim wrist render lowers the plan? Real exterior + recorded state in all of these;
# only the wrist image changes (masks from the sim segmentation at the recorded pose)
WRIST_CONDS = ("w_simgrip_on_real",   # real wrist frame, sim gripper pixels pasted over it
               "w_realgrip_on_sim",   # sim wrist render, real gripper pixels pasted in
               "w_noshadow",          # sim wrist render without the sim shadow darkening
               "w_recolour",          # sim wrist render, each object scaled to its real median colour (real wrist frame)
               "w_simobj_on_real")    # real wrist frame, sim object pixels drawn over it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="45,48,51,54,57,60,63,66,69,72")
    ap.add_argument("--instruction", default="put the orange block in the bowl")
    ap.add_argument("--ext", default="22008760")
    ap.add_argument("--policy", choices=["groot", "hold"], default="groot")
    ap.add_argument("--conds", default=",".join(CONDS), help=f"comma-separated; G3 default; G4 adds: {','.join(WRIST_CONDS)}")
    a = ap.parse_args()
    conds = a.conds.split(",")
    os.makedirs(a.out, exist_ok=True)
    m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
    fit = json.load(open(os.path.join(a.scene, "scene_fit.json")))
    tz = fit["table"]["z"]
    wserial = fit["wrist_camera"]["serial"]
    arm = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    l7, ps = m.body("fr3_link7").id, m.site("g_pinch").id
    drv_max = m.jnt_range[m.joint("g_right_driver_joint").id][1]
    plate = load_plate(a.scene, "lama", "frame0+plate")
    comp = Compositor(m, a.scene, plate, size=(H, W), verbose=False)
    wp = wrist_plate(m, a.scene, a.episode, comp, wserial)
    wc = WristCompositor(m, plate, comp, (H, W), plate2=wp)
    comp_ns = Compositor(m, a.scene, plate, shadows=False, size=(H, W), verbose=False)
    wc_ns = WristCompositor(m, plate, comp_ns, (H, W), plate2=wp)
    grip_g = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(("fr3_", "g_"))]
    free_b = {m.jnt_bodyid[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
    obj_ids = {b: [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == b] for b in free_b}
    d, dk = mujoco.MjData(m), mujoco.MjData(m)
    h = h5py.File(os.path.join(a.episode, "trajectory.h5"))
    J, G, C = (h[f"observation/robot_state/{k}"][:] for k in ("joint_positions", "gripper_position", "cartesian_position"))

    def real(serial, k):
        cap = cv2.VideoCapture(os.path.join(a.episode, f"{serial}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, b = cap.read()
        return cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA)

    def plan(chunk):
        hs, tl = [], []
        for j in range(15):
            dk.qpos[:] = m.qpos0
            dk.qpos[arm] = chunk[j, 10:17]
            mujoco.mj_kinematics(m, dk)
            hs.append(dk.site_xpos[ps][2] - tz)
            tl.append(np.degrees(np.arccos(np.clip(-dk.xmat[l7].reshape(3, 3)[2, 2], -1, 1))))
        return dict(min_height_cm=float(min(hs) * 100), mean_tilt_deg=float(np.mean(tl)), max_grip=float(chunk[:15, 9].max()))

    pol = HoldPolicy() if a.policy == "hold" else GrootClient(DEFAULT_CKPT, os.path.join(REPO, ".venvs/groot/bin/python"), "cuda", 600)
    rows, thumbs, wthumbs = [], [], []
    try:
        for k in (int(x) for x in a.frames.split(",")):
            d.qpos[:] = m.qpos0
            d.qpos[arm] = J[k]
            for jn in ("g_right_driver_joint", "g_left_driver_joint"):
                d.qpos[m.joint(jn).qposadr[0]] = G[k] * drv_max
            mujoco.mj_forward(m, d)
            sim_wr_f, info = wc(d, return_info=True)
            imgs = dict(real_ext=real(a.ext, k), real_wr=real(wserial, k),
                        sim_ext=comp(d).clip(0, 255).astype(np.uint8), sim_wr=sim_wr_f.clip(0, 255).astype(np.uint8))
            isg = info["seg"][..., 1] == mujoco.mjtObj.mjOBJ_GEOM
            gmask = isg & np.isin(info["seg"][..., 0], grip_g)
            omask = {b: isg & np.isin(info["seg"][..., 0], gs) for b, gs in obj_ids.items()}
            rw, sw = imgs["real_wr"], imgs["sim_wr"]
            v = dict(w_simgrip_on_real=rw.copy(), w_realgrip_on_sim=sw.copy(),
                     w_noshadow=wc_ns(d).clip(0, 255).astype(np.uint8), w_recolour=sw.astype(np.float32), w_simobj_on_real=rw.copy())
            v["w_simgrip_on_real"][gmask] = sw[gmask]
            v["w_realgrip_on_sim"][gmask] = rw[gmask]
            for b, om in omask.items():
                core = cv2.erode(om.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                if core.sum() >= 20 and "bowl" not in m.body(b).name:  # bowl: at the image edge its mask covers real table pixels
                    gain = np.clip(np.median(rw[core].astype(np.float32), 0) / np.maximum(np.median(sw[core].astype(np.float32), 0), 1.0), 1 / 3, 3)
                    v["w_recolour"][om] = sw[om] * gain
                v["w_simobj_on_real"][om] = sw[om]
            v["w_recolour"] = v["w_recolour"].clip(0, 255).astype(np.uint8)
            imgs.update(v)
            R = d.xmat[l7].reshape(3, 3)
            cart_fk = np.r_[d.xpos[l7] + R @ FLANGE_IN_LINK7, Rot.from_matrix(R).as_euler("xyz")]
            st_rec = np.r_[compute_eef_9d(C[k]), G[k], J[k]].astype(np.float32)
            st_fk = np.r_[compute_eef_9d(cart_fk), G[k], J[k]].astype(np.float32)
            obs = dict(real_real=(imgs["real_ext"], imgs["real_wr"], st_rec), sim_sim=(imgs["sim_ext"], imgs["sim_wr"], st_rec),
                       sim_real=(imgs["sim_ext"], imgs["real_wr"], st_rec), real_sim=(imgs["real_ext"], imgs["sim_wr"], st_rec),
                       real_fk=(imgs["real_ext"], imgs["real_wr"], st_fk), **{c: (imgs["real_ext"], imgs[c], st_rec) for c in WRIST_CONDS})
            row = dict(frame=k, state_fk_minus_recorded_max=float(np.abs(st_fk - st_rec).max()))
            for c in conds:
                e, w, s = obs[c]
                row[c] = plan(pol({IMG[0]: e, IMG[1]: w, "state": s, "instruction": a.instruction})["action"])
            rows.append(row)
            thumbs.append(np.concatenate([np.concatenate([imgs["real_ext"], imgs["real_wr"]], 1), np.concatenate([imgs["sim_ext"], imgs["sim_wr"]], 1)], 0))
            wtiles = [imgs[c].copy() for c in ("real_wr", "sim_wr") + WRIST_CONDS]
            for im, lab in zip(wtiles, ("real_wr", "sim_wr") + WRIST_CONDS):
                cv2.putText(im, lab, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            wtiles.append(np.zeros_like(wtiles[0]))  # 7 tiles -> 2 rows of 4
            wthumbs.append(np.concatenate([np.concatenate(wtiles[:4], 1), np.concatenate(wtiles[4:], 1)], 0))
            print(f"f{k}: " + "  ".join(f"{c} h {row[c]['min_height_cm']:+.1f} tilt {row[c]['mean_tilt_deg']:.1f}" for c in conds)
                  + f"  (state fk-rec max {row['state_fk_minus_recorded_max']:.4f})", flush=True)
    finally:
        pol.close()
    summ = {c: dict(min_height_cm_mean=float(np.mean([r[c]["min_height_cm"] for r in rows])),
                    mean_tilt_deg_mean=float(np.mean([r[c]["mean_tilt_deg"] for r in rows]))) for c in conds}
    json.dump(dict(scene=a.scene, episode=a.episode, instruction=a.instruction, policy=a.policy, rows=rows, summary=summ,
                   status="exploratory, UNVERIFIED"), open(os.path.join(a.out, "image_swap.json"), "w"), indent=1)
    cv2.imwrite(os.path.join(a.out, "wrist_variants.jpg"), cv2.cvtColor(np.concatenate(wthumbs[::3], 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(os.path.join(a.out, "inputs.jpg"), cv2.cvtColor(np.concatenate(thumbs[::3], 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    for c, v in summ.items():
        print(f"SUMMARY {c}: lowest planned fingertip {v['min_height_cm_mean']:+.1f} cm, tilt {v['mean_tilt_deg_mean']:.1f} deg")


if __name__ == "__main__":
    main()
