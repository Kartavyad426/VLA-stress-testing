"""Metric 3D object anchors from the DROID wrist camera, independent of the exterior calibration.

The wrist camera's pose comes from the recorded joints (FK) + DROID's gripper offset (build_scene wrist camera);
depth comes from its ZED Mini stereo pair (stereo_depth.py, 63 mm baseline). For each object label of the scene's
exterior perception (table and gripper excluded), the object is detected + segmented in each given wrist frame
(Grounding DINO + SAM, as perceive.py), and the eroded mask's stereo points are taken to the robot frame. Anchor =
median over frames of the per-frame median point (a point on the visible surface, not the object centre).

Use only frames before the object is touched. Writes <scene>/wrist_anchors.json.

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/wrist_anchors.py \
        --scene runs/real2sim/ep_orange_block --episode data/real2sim/droid/ep_orange_block --frames 0,20,40
"""
import argparse, json, os, sys
import numpy as np
import cv2
import mujoco
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stereo_depth import stereo_pair, disparity, BASELINE_M  # noqa: E402
from composite import cam_K, cam_pose  # noqa: E402

SKIP = ("wooden table", "robot gripper")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--frames", default="0,20,40")
    ap.add_argument("--num_disp", type=int, default=320, help="SGBM range; blocks can be ~0.15 m from the wrist camera")
    ap.add_argument("--min_px", type=int, default=150, help="ignore a detection with fewer valid stereo pixels")
    ap.add_argument("--table_z", type=float, default=None, help="drop detections whose point is not within [-3, +15] cm of this "
                    "table height (wrong detections, e.g. the gripper labelled as a block; seen on ep_orange_block frame 0)")
    a = ap.parse_args()
    from perceive import detect_multi, segment
    m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
    d = mujoco.MjData(m)
    fit = json.load(open(os.path.join(a.scene, "scene_fit.json")))
    serial = fit["wrist_camera"]["serial"]
    labels = sorted({k.split("#")[0] for k in fit["objects"]} - set(SKIP))
    J = h5py.File(os.path.join(a.episode, "trajectory.h5"))["observation/robot_state/joint_positions"][:]
    arm = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    cid = m.camera("wrist").id
    per_frame = {}
    for fr in (int(x) for x in a.frames.split(",")):
        q = m.qpos0.copy()
        q[arm] = J[fr]
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        L, R_img = stereo_pair(a.episode, serial, fr)
        H, W = L.shape[:2]
        K = cam_K(m, cid, W, H)
        Rc, tc = cam_pose(d, cid)
        disp = disparity(L, R_img, num_disp=a.num_disp)
        rgb = cv2.cvtColor(L, cv2.COLOR_BGR2RGB)
        dets = detect_multi(rgb, labels)
        masks = segment(rgb, dets)
        for name, mk in masks.items():
            mk = cv2.erode(mk.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            sel = mk & (disp > 1)
            if sel.sum() < a.min_px:
                continue
            v, u = np.nonzero(sel)
            z = K[0, 0] * BASELINE_M["wrist"] / disp[sel]
            pc = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1)
            pw = pc @ Rc.T + tc
            if a.table_z is not None and not (-0.03 <= np.median(pw[:, 2]) - a.table_z <= 0.15):
                print(f"  dropped {name} frame {fr}: z {np.median(pw[:, 2]):.3f} vs table {a.table_z}")
                continue
            per_frame.setdefault(name, []).append(dict(frame=fr, px=int(sel.sum()), mask_px=int(mk.sum()), score=dets[name]["score"],
                                                      point=np.median(pw, 0).tolist(), spread_iqr_m=float(np.mean(np.subtract(*np.percentile(pw, [75, 25], 0))))))
    anchors = {}
    for name, obs in per_frame.items():
        P = np.array([o["point"] for o in obs])
        anchors[name] = dict(point=np.median(P, 0).tolist(), n_frames=len(obs),
                             frame_spread_m=float(np.max(np.linalg.norm(P - np.median(P, 0), axis=1))) if len(obs) > 1 else None, obs=obs)
        print(name, np.round(anchors[name]["point"], 4), "frames", [o["frame"] for o in obs], "spread %.3f" % (anchors[name]["frame_spread_m"] or 0))
    json.dump(dict(episode=a.episode, serial=serial, baseline_m=BASELINE_M["wrist"], frames=a.frames, anchors=anchors),
              open(os.path.join(a.scene, "wrist_anchors.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
