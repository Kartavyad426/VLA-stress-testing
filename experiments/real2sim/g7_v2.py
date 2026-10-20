"""G7 (10-20, step 3a): does the deployable wrist renderer (WristRenderV2: wrist-video table texture, real-gripper
library, camera-style downscale, optional wrist-fitted light) make GR00T-DROID plan its grasp like the real image?
Same 10 recorded approach states, recorded state, real exterior image; only the wrist image changes:
  real_real, real_real_rep   real wrist frame (twice: run-to-run noise)
  sim_old                    the WristCompositor render used in all rollouts so far
  V2_ext                     WristRenderV2 with the exterior-fitted light (light.json)
  V2_wrist                   WristRenderV2 with the wrist-fitted light (light_wrist.json, S4)
  V2_noshadow                WristRenderV2 without sim shadows
  V2_L                       V2_noshadow x the real frame's low-pass brightness ratio on table pixels (diagnostic
                             upper bound, uses the real frame; = G6's L on the V2 base)
Everything except real_* and V2_L uses no real frame of that moment.

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/g7_v2.py \
        --scene runs/real2sim/ep_orange_block_stereo --episode data/real2sim/droid/ep_orange_block \
        --out runs/real2sim/policy/g7_v2 [--policy hold]
"""
import argparse, json, os, sys
import numpy as np
import cv2
import h5py
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from policy_rollout import GrootClient, HoldPolicy, compute_eef_9d, IMG, H, W, DEFAULT_CKPT, REPO  # noqa: E402
from composite import (Compositor, WristCompositor, WristRenderV2, load_plate, wrist_plate, wrist_table_texture,  # noqa: E402
                       load_gripper_library)

CONDS = ("real_real", "real_real_rep", "sim_old", "V2_ext", "V2_wrist", "V2_noshadow", "V2_L")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="45,48,51,54,57,60,63,66,69,72")
    ap.add_argument("--instruction", default="put the orange block in the bowl")
    ap.add_argument("--ext", default="22008760")
    ap.add_argument("--policy", choices=["groot", "hold"], default="groot")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
    fit = json.load(open(os.path.join(a.scene, "scene_fit.json")))
    tz, ws = fit["table"]["z"], fit["wrist_camera"]["serial"]
    arm = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    l7, ps = m.body("fr3_link7").id, m.site("g_pinch").id
    drv_max = m.jnt_range[m.joint("g_right_driver_joint").id][1]
    h = h5py.File(os.path.join(a.episode, "trajectory.h5"))
    J, G, C = (h[f"observation/robot_state/{k}"][:] for k in ("joint_positions", "gripper_position", "cartesian_position"))
    k_close = int(np.argmax(G > 0.5))
    plate = load_plate(a.scene, "lama", "frame0+plate")
    comp = Compositor(m, a.scene, plate, size=(H, W), verbose=False)
    comp_ns = Compositor(m, a.scene, plate, shadows=False, size=(H, W), verbose=False)
    wp = wrist_plate(m, a.scene, a.episode, comp, ws)
    tt = wrist_table_texture(m, a.scene, a.episode, ws, list(range(0, k_close - 2, 2)))
    lib = load_gripper_library(os.path.join(a.scene, "gripper_lib", "gripper_library.npz"))
    lw = json.load(open(os.path.join(a.scene, "light_wrist.json")))
    r_old = WristCompositor(m, plate, comp, (H, W), plate2=wp)
    r_ext = WristRenderV2(m, plate, comp, (H, W), wp, tt, lib)
    r_wr = WristRenderV2(m, plate, comp, (H, W), wp, tt, lib, light=lw)
    r_ns = WristRenderV2(m, plate, comp_ns, (H, W), wp, tt, lib)
    tbl = m.geom("table").id
    d, dk = mujoco.MjData(m), mujoco.MjData(m)

    def real(serial, k):
        cap = cv2.VideoCapture(os.path.join(a.episode, f"{serial}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, b = cap.read()
        return cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA)

    def plan(chunk):
        hts, tl = [], []
        for j in range(15):
            dk.qpos[:] = m.qpos0
            dk.qpos[arm] = chunk[j, 10:17]
            mujoco.mj_kinematics(m, dk)
            hts.append(dk.site_xpos[ps][2] - tz)
            tl.append(np.degrees(np.arccos(np.clip(-dk.xmat[l7].reshape(3, 3)[2, 2], -1, 1))))
        return dict(min_height_cm=float(min(hts) * 100), mean_tilt_deg=float(np.mean(tl)))

    pol = HoldPolicy() if a.policy == "hold" else GrootClient(DEFAULT_CKPT, os.path.join(REPO, ".venvs/groot/bin/python"), "cuda", 600)
    rows, sheet = [], None
    try:
        for k in (int(x) for x in a.frames.split(",")):
            d.qpos[:] = m.qpos0
            d.qpos[arm] = J[k]
            for jn in ("g_right_driver_joint", "g_left_driver_joint"):
                d.qpos[m.joint(jn).qposadr[0]] = G[k] * drv_max
            mujoco.mj_forward(m, d)
            st = np.r_[compute_eef_9d(C[k]), G[k], J[k]].astype(np.float32)
            ext = real(a.ext, k)
            rw = real(ws, k)
            ns, info = r_ns(d, return_info=True)
            table = cv2.resize(((info["seg"][..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (info["seg"][..., 0] == tbl)).astype(np.uint8),
                               (W, H), interpolation=cv2.INTER_NEAREST) > 0
            lr = cv2.GaussianBlur(rw.astype(np.float32).mean(-1), (0, 0), 4)
            ls = cv2.GaussianBlur(ns.astype(np.float32).mean(-1), (0, 0), 4)
            ratio = np.where(table, np.clip(lr / np.maximum(ls, 1), 0.3, 1.6), 1.0)
            imgs = dict(real_real=rw, real_real_rep=rw, sim_old=r_old(d), V2_ext=r_ext(d), V2_wrist=r_wr(d), V2_noshadow=ns,
                        V2_L=ns * ratio[..., None])
            imgs = {c: np.clip(v, 0, 255).astype(np.uint8) for c, v in imgs.items()}
            row = dict(frame=k)
            for c in CONDS:
                row[c] = plan(pol({IMG[0]: ext, IMG[1]: imgs[c], "state": st, "instruction": a.instruction})["action"])
            rows.append(row)
            print(f"f{k}: " + " ".join(f"{c} {row[c]['min_height_cm']:+.1f}" for c in CONDS), flush=True)
            if k == 63 or sheet is None:
                tiles = []
                for c in CONDS:
                    im = imgs[c].copy()
                    cv2.putText(im, c, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                    tiles.append(im)
                tiles.append(np.zeros_like(tiles[0]))
                sheet = np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)], 0)
    finally:
        pol.close()
    summ = {c: dict(min_height_cm_mean=float(np.mean([r[c]["min_height_cm"] for r in rows])),
                    near_grasp_min_height_cm=float(np.mean([r[c]["min_height_cm"] for r in rows if r["frame"] >= 60])),
                    mean_tilt_deg_mean=float(np.mean([r[c]["mean_tilt_deg"] for r in rows])),
                    near_grasp_tilt_deg=float(np.mean([r[c]["mean_tilt_deg"] for r in rows if r["frame"] >= 60]))) for c in CONDS}
    noise = float(np.mean([abs(r["real_real"]["min_height_cm"] - r["real_real_rep"]["min_height_cm"]) for r in rows]))
    json.dump(dict(scene=a.scene, episode=a.episode, policy=a.policy, light_wrist=lw, rows=rows, summary=summ, rerun_noise_cm=noise,
                   status="exploratory, UNVERIFIED"), open(os.path.join(a.out, "g7.json"), "w"), indent=1)
    cv2.imwrite(os.path.join(a.out, "wrist_conditions.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"re-run noise: {noise:.2f} cm")
    for c, v in summ.items():
        print(f"SUMMARY {c:14s} lowest planned fingertip {v['min_height_cm_mean']:+.1f} cm (near grasp {v['near_grasp_min_height_cm']:+.1f}), tilt {v['mean_tilt_deg_mean']:.1f} (near grasp {v['near_grasp_tilt_deg']:.1f})")


if __name__ == "__main__":
    main()
