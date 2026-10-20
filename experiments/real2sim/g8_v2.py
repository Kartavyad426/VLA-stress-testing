"""G8 (10-20, D1 + D3): (a) does the D1 wrist-camera correction (fit_wrist_gripper.py "camera": 8.5 mm, 2.4 deg)
make GR00T-DROID plan more like the real image? (b) what is left between WristRenderV2 + wrist-fitted light (G7:
+1.2 cm vs real +2.4/+2.6) and the real image: live real pixels in a ring around the gripper (its own close shadow
and lighting), or the sim objects?
Same 10 recorded approach states, recorded state, real exterior; only the wrist image changes:
  real_real, real_real_rep    real wrist frame (twice: noise)
  V2w                         WristRenderV2 + light_wrist.json (= G7 V2_wrist)
  V2w_cam                     the same with the D1 camera correction (own table texture through that camera)
  V2w_ring                    V2w + live real pixels in a ring around the real gripper  [diagnostic: real frame]
  V2w_obj                     V2w + real pixels inside the (dilated) sim object masks  [diagnostic]
  V2w_ring_obj                both
  V2w_cam_ring_obj            camera correction + both

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/g8_v2.py \
        --scene runs/real2sim/ep_orange_block_stereo --episode data/real2sim/droid/ep_orange_block \
        --out runs/real2sim/policy/g8_v2 [--policy hold]
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
from composite import Compositor, WristRenderV2, load_plate, wrist_plate, wrist_table_texture, load_gripper_library  # noqa: E402

CONDS = ("real_real", "real_real_rep", "V2w", "V2w_cam", "V2w_ring", "V2w_obj", "V2w_ring_obj", "V2w_cam_ring_obj")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="45,48,51,54,57,60,63,66,69,72")
    ap.add_argument("--instruction", default="put the orange block in the bowl")
    ap.add_argument("--ext", default="22008760")
    ap.add_argument("--ring", type=int, default=9, help="px at 320x180: ring width around the real gripper")
    ap.add_argument("--policy", choices=["groot", "hold"], default="groot")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fit = json.load(open(os.path.join(a.scene, "scene_fit.json")))
    tz, ws = fit["table"]["z"], fit["wrist_camera"]["serial"]
    h = h5py.File(os.path.join(a.episode, "trajectory.h5"))
    J, G, C = (h[f"observation/robot_state/{k}"][:] for k in ("joint_positions", "gripper_position", "cartesian_position"))
    k_close = int(np.argmax(G > 0.5))
    lib = load_gripper_library(os.path.join(a.scene, "gripper_lib", "gripper_library.npz"))
    lw = json.load(open(os.path.join(a.scene, "light_wrist.json")))
    camfix = json.load(open(os.path.join(a.scene, "wrist_gripper_fit.json")))["camera"]

    def setup(cam_correct):
        m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
        if cam_correct:
            c = m.camera("wrist").id
            m.cam_quat[c] = (Rot.from_rotvec(np.radians(camfix["rot_deg"])) * Rot.from_quat(m.cam_quat[c], scalar_first=True)).as_quat(scalar_first=True)
            m.cam_pos[c] = m.cam_pos[c] + np.array(camfix["trans_mm"]) / 1000
        plate = load_plate(a.scene, "lama", "frame0+plate")
        comp = Compositor(m, a.scene, plate, size=(H, W), verbose=False)
        wp = wrist_plate(m, a.scene, a.episode, comp, ws) if not cam_correct else None
        cache = None if not cam_correct else os.path.join(a.out, "table_tex_wrist_camfix.npz")
        tt = wrist_table_texture(m, a.scene, a.episode, ws, list(range(0, k_close - 2, 2)), cache=cache)
        return m, WristRenderV2(m, plate, comp, (H, W), wp, tt, lib, light=lw)

    m, r_v2 = setup(False)
    m_c, r_c = setup(True)
    arm = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    l7, ps = m.body("fr3_link7").id, m.site("g_pinch").id
    drv_max = m.jnt_range[m.joint("g_right_driver_joint").id][1]
    free_b = {m.jnt_bodyid[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
    objs = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] in free_b]
    galpha = cv2.resize(lib["alpha"], (W, H), interpolation=cv2.INTER_AREA) > 0.5
    ring = (cv2.dilate(galpha.astype(np.uint8), np.ones((2 * a.ring + 1,) * 2, np.uint8)) > 0) & ~galpha
    d, dc, dk = mujoco.MjData(m), mujoco.MjData(m_c), mujoco.MjData(m)

    def real(serial, k):
        cap = cv2.VideoCapture(os.path.join(a.episode, f"{serial}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, b = cap.read()
        return cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA)

    def pose(mm, dd, k):
        dd.qpos[:] = mm.qpos0
        dd.qpos[arm] = J[k]
        for jn in ("g_right_driver_joint", "g_left_driver_joint"):
            dd.qpos[mm.joint(jn).qposadr[0]] = G[k] * drv_max
        mujoco.mj_forward(mm, dd)

    def plan(chunk):
        hts, tl = [], []
        for j in range(15):
            dk.qpos[:] = m.qpos0
            dk.qpos[arm] = chunk[j, 10:17]
            mujoco.mj_kinematics(m, dk)
            hts.append(dk.site_xpos[ps][2] - tz)
            tl.append(np.degrees(np.arccos(np.clip(-dk.xmat[l7].reshape(3, 3)[2, 2], -1, 1))))
        return dict(min_height_cm=float(min(hts) * 100), mean_tilt_deg=float(np.mean(tl)))

    def objmask(info):
        isg = info["seg"][..., 1] == mujoco.mjtObj.mjOBJ_GEOM
        om = cv2.resize((isg & np.isin(info["seg"][..., 0], objs)).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        return cv2.dilate(om, np.ones((5, 5), np.uint8)) > 0

    pol = HoldPolicy() if a.policy == "hold" else GrootClient(DEFAULT_CKPT, os.path.join(REPO, ".venvs/groot/bin/python"), "cuda", 600)
    rows, sheet = [], None
    try:
        for k in (int(x) for x in a.frames.split(",")):
            pose(m, d, k)
            pose(m_c, dc, k)
            st = np.r_[compute_eef_9d(C[k]), G[k], J[k]].astype(np.float32)
            ext, rw = real(a.ext, k), real(ws, k)
            v2, info = r_v2(d, return_info=True)
            vc, info_c = r_c(dc, return_info=True)
            v2, vc = np.clip(v2, 0, 255).astype(np.uint8), np.clip(vc, 0, 255).astype(np.uint8)
            om, omc = objmask(info), objmask(info_c)

            def paste(base, mk):
                out = base.copy()
                out[mk] = rw[mk]
                return out
            imgs = dict(real_real=rw, real_real_rep=rw, V2w=v2, V2w_cam=vc, V2w_ring=paste(v2, ring), V2w_obj=paste(v2, om),
                        V2w_ring_obj=paste(v2, ring | om), V2w_cam_ring_obj=paste(vc, ring | omc))
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
                sheet = np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)], 0)
    finally:
        pol.close()
    summ = {c: dict(min_height_cm_mean=float(np.mean([r[c]["min_height_cm"] for r in rows])),
                    near_grasp_min_height_cm=float(np.mean([r[c]["min_height_cm"] for r in rows if r["frame"] >= 60])),
                    mean_tilt_deg_mean=float(np.mean([r[c]["mean_tilt_deg"] for r in rows])),
                    near_grasp_tilt_deg=float(np.mean([r[c]["mean_tilt_deg"] for r in rows if r["frame"] >= 60]))) for c in CONDS}
    noise = float(np.mean([abs(r["real_real"]["min_height_cm"] - r["real_real_rep"]["min_height_cm"]) for r in rows]))
    json.dump(dict(scene=a.scene, episode=a.episode, policy=a.policy, camera_correction=camfix, ring_px=a.ring, rows=rows, summary=summ,
                   rerun_noise_cm=noise, status="exploratory, UNVERIFIED"), open(os.path.join(a.out, "g8.json"), "w"), indent=1)
    cv2.imwrite(os.path.join(a.out, "wrist_conditions.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"re-run noise: {noise:.2f} cm")
    for c, v in summ.items():
        print(f"SUMMARY {c:17s} {v['min_height_cm_mean']:+.1f} cm (near grasp {v['near_grasp_min_height_cm']:+.1f}), tilt {v['mean_tilt_deg_mean']:.1f} (near grasp {v['near_grasp_tilt_deg']:.1f})")


if __name__ == "__main__":
    main()
