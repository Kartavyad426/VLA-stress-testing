"""D1 (10-20): why is our rendered gripper drawn in a different place than the real one in the wrist view
(silhouette IoU 0.76 vs the real open gripper, sim fingers end higher in the image)? Both the wrist camera and the
gripper hang rigidly from fr3_link7, so the open gripper's silhouette depends only on their relative placement.
Two hypotheses, each fitted alone to the real open-gripper mask (gripper_library.py, alpha > 0.5):
  H1 camera: DROID's wrist-camera offset is off -> 6-DoF change of the wrist camera pose on fr3_link7
  H2 gripper: our gripper model / mount is off   -> change of g_base_mount on fr3_link7 (x, y, z, roll, pitch, yaw)
Cost = symmetric chamfer distance (px at 320x180) between sim and real silhouettes; IoU reported. Writes
<scene>/wrist_gripper_fit.json (+ overlay image). Analysis only: nothing is applied to the scene here.

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/fit_wrist_gripper.py --scene runs/real2sim/ep_orange_block_stereo
"""
import argparse, json, os
import numpy as np
import cv2
import mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

H, W = 180, 320


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--lib", default=None, help="default <scene>/gripper_lib/gripper_library.npz")
    a = ap.parse_args()
    m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
    d = mujoco.MjData(m)
    lib = np.load(a.lib or os.path.join(a.scene, "gripper_lib", "gripper_library.npz"))
    real = cv2.resize(lib["alpha"][0], (W, H), interpolation=cv2.INTER_AREA) > 0.5
    dt_real = cv2.distanceTransform((~real).astype(np.uint8), cv2.DIST_L2, 3)
    cid = m.camera("wrist").id
    gb = m.body("g_base_mount").id
    grip = np.array([g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("g_")])
    cam0 = (m.cam_pos[cid].copy(), m.cam_quat[cid].copy())
    body0 = (m.body_pos[gb].copy(), m.body_quat[gb].copy())
    rend = mujoco.Renderer(m, H, W)
    rend.enable_segmentation_rendering()
    d.qpos[:] = m.qpos0

    def apply(hyp, x):
        m.cam_pos[cid], m.cam_quat[cid] = cam0
        m.body_pos[gb], m.body_quat[gb] = body0
        rot = Rot.from_rotvec(np.radians(x[:3]))
        if hyp == "camera":  # deltas in fr3_link7 coordinates (deg, mm)
            m.cam_quat[cid] = (rot * Rot.from_quat(cam0[1], scalar_first=True)).as_quat(scalar_first=True)
            m.cam_pos[cid] = cam0[0] + np.asarray(x[3:6]) / 1000
        elif hyp == "gripper":
            m.body_quat[gb] = (rot * Rot.from_quat(body0[1], scalar_first=True)).as_quat(scalar_first=True)
            m.body_pos[gb] = body0[0] + np.asarray(x[3:6]) / 1000

    def sim_mask(hyp, x):
        apply(hyp, x)
        mujoco.mj_forward(m, d)
        rend.update_scene(d, camera=cid)
        seg = rend.render()
        return (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], grip)

    def cost(hyp, x):
        s = sim_mask(hyp, x)
        if not s.any():
            return 100.0
        dt_sim = cv2.distanceTransform((~s).astype(np.uint8), cv2.DIST_L2, 3)
        return float(dt_real[s].mean() + dt_sim[real].mean())

    def iou(s):
        return float((s & real).sum() / max((s | real).sum(), 1))

    out, masks = {}, {"before": sim_mask("camera", np.zeros(6))}
    out["before"] = dict(chamfer_px=cost("camera", np.zeros(6)), iou=iou(masks["before"]))
    for hyp in ("camera", "gripper"):
        best = None
        for x0 in (np.zeros(6), np.r_[2, 0, 0, 0, 0, 0], np.r_[0, 2, 0, 0, 0, 0], np.r_[0, 0, 3, 0, 0, 0], np.r_[0, 0, 0, 0, 0, 5]):
            r = minimize(lambda x: cost(hyp, x), x0, method="Powell", options=dict(xtol=1e-2, ftol=1e-3, maxfev=1500))
            if best is None or r.fun < best.fun:
                best = r
        masks[hyp] = sim_mask(hyp, best.x)
        out[hyp] = dict(chamfer_px=float(best.fun), iou=iou(masks[hyp]), rot_deg=np.round(best.x[:3], 2).tolist(),
                        trans_mm=np.round(best.x[3:6], 1).tolist(), rot_norm_deg=float(np.linalg.norm(best.x[:3])),
                        trans_norm_mm=float(np.linalg.norm(best.x[3:6])), nfev=int(best.nfev))
        print(hyp, json.dumps(out[hyp]), flush=True)
    apply("camera", np.zeros(6))
    rend.close()
    out["note"] = "deltas in fr3_link7 coordinates; camera = wrist camera pose, gripper = g_base_mount pose"
    out["status"] = "exploratory, UNVERIFIED"
    json.dump(out, open(os.path.join(a.scene, "wrist_gripper_fit.json"), "w"), indent=1)
    lib_rgb = cv2.resize(lib["rgb"][0], (W, H), interpolation=cv2.INTER_AREA)
    tiles = []
    for name in ("before", "camera", "gripper"):
        vis = lib_rgb.copy()
        for mk, col in ((real, (0, 255, 255)), (masks[name], (255, 0, 255))):
            cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cs, -1, col, 1)
        cv2.putText(vis, f"{name}: IoU {out[name]['iou']:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        tiles.append(vis)
    cv2.imwrite(os.path.join(a.scene, "wrist_gripper_fit.jpg"), cv2.cvtColor(cv2.resize(np.concatenate(tiles, 1), (1920, 360)), cv2.COLOR_RGB2BGR))
    print("before", json.dumps(out["before"]))


if __name__ == "__main__":
    main()
