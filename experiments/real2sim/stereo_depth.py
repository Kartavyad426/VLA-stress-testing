"""Metric depth from DROID's ZED stereo videos (<serial>-stereo.mp4: rectified left|right, 2560x720; the left half is
the normal <serial>.mp4 frame, checked: mean abs diff 1-2 grey levels; vertical offset of ORB matches median 0 px).

Baselines are Stereolabs' published specs (checked 10-16): ZED 2 = 120 mm (DROID's two exterior cameras),
ZED Mini = 63 mm (DROID's wrist camera; Stereolabs' DROID case study names the models).

As a check (pre-deploy checklist #6): table-top height in the robot frame from stereo, versus the rebuilt scene.
The wrist camera's pose comes from the robot's joints + DROID's gripper offset, so it is independent of the
exterior calibration.

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/stereo_depth.py \
        --scene runs/real2sim/ep_orange_block --episode data/real2sim/droid/ep_orange_block --frames 0,20,40
"""
import argparse, json, os, sys
import numpy as np
import cv2

BASELINE_M = {"ext": 0.120, "wrist": 0.063}  # ZED 2 / ZED Mini specs


def stereo_pair(episode, serial, frame):
    cap = cv2.VideoCapture(os.path.join(episode, f"{serial}-stereo.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, bgr = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame} of {episode}/{serial}-stereo.mp4")
    w = bgr.shape[1] // 2
    return bgr[:, :w], bgr[:, w:]


def disparity(left_bgr, right_bgr, num_disp=192, block=7):
    """OpenCV SGBM disparity (px, float); <= 0 where invalid."""
    sg = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disp, blockSize=block, P1=8 * 3 * block ** 2,
                               P2=32 * 3 * block ** 2, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                               mode=cv2.STEREO_SGBM_MODE_HH)
    return sg.compute(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16


def stereo_z(episode, serial, frame, fx, baseline, num_disp=192):
    """z-depth (m) of the left image, NaN where the disparity is invalid."""
    d = disparity(*stereo_pair(episode, serial, frame), num_disp=num_disp)
    z = np.full(d.shape, np.nan, np.float32)
    ok = d > 1
    z[ok] = fx * baseline / d[ok]
    return z


MIN_TABLE_PX, MAX_TABLE_IQR_M = 3000, 0.03  # 10-27 (checklist A1): skip frames that don't see the table well


def table_height(m, d, episode, serial, frame, cam):
    """Median table-top z (robot frame) from stereo, over the pixels where the sim sees the table (eroded)."""
    import mujoco
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from composite import cam_K, cam_pose
    cid = m.camera(cam).id
    K = cam_K(m, cid, 1280, 720)
    R, t = cam_pose(d, cid)
    r = mujoco.Renderer(m, 720, 1280)
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=cam)
    seg = r.render()
    r.close()
    table = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 0] == m.geom("table").id)
    table = cv2.erode(table.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    z = stereo_z(episode, serial, frame, K[0, 0], BASELINE_M[cam])
    sel = table & np.isfinite(z)
    if sel.sum() < MIN_TABLE_PX:  # 10-27 guard: c9 crashed here on an empty selection
        return dict(cam=cam, frame=frame, px=int(sel.sum()), table_px=int(table.sum()), z_median=None,
                    skipped=f"only {int(sel.sum())} stereo table pixels (< {MIN_TABLE_PX})")
    v, u = np.nonzero(sel)
    pc = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z[sel], (v + 0.5 - K[1, 2]) / K[1, 1] * z[sel], z[sel]], -1)
    zw = (pc @ R.T + t)[:, 2]
    iqr = [float(np.percentile(zw, 25)), float(np.percentile(zw, 75))]
    res = dict(cam=cam, frame=frame, px=int(sel.sum()), table_px=int(table.sum()), z_median=float(np.median(zw)), z_iqr=iqr)
    if iqr[1] - iqr[0] > MAX_TABLE_IQR_M:  # not a table: the projected sim table covers something else (c11 +34.5 cm)
        res.update(z_median=None, z_median_raw=float(np.median(zw)),
                   skipped=f"table z spread (IQR) {100 * (iqr[1] - iqr[0]):.1f} cm > {100 * MAX_TABLE_IQR_M:.0f} cm")
    return res


def main():
    import mujoco
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--frames", default=None, help="real frames, comma-separated (replay_states.npz poses the sim robot there); derived from the episode (episode_info.py) when omitted")
    ap.add_argument("--ext_serial", default=None, help="default: calib.json 'cam'")
    a = ap.parse_args()
    from episode_info import frames_arg
    frames = frames_arg(a.frames, a.episode, "stereo")
    m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
    d = mujoco.MjData(m)
    fit = json.load(open(os.path.join(a.scene, "scene_fit.json")))
    ext_serial = a.ext_serial or json.load(open(os.path.join(a.scene, "calib.json")))["cam"]
    st = np.load(os.path.join(a.scene, "replay_states.npz"))
    out = dict(sim_table_z=fit["table"]["z"], results=[])
    for fr in frames:
        k = int(np.argmin(np.abs(st["real_frames"] - fr)))
        d.qpos[:] = st["qpos"][k]
        mujoco.mj_forward(m, d)
        for cam, serial in (("ext", ext_serial), ("wrist", fit["wrist_camera"]["serial"])):
            if cam == "ext" and fr != frames[0]:
                continue  # the exterior camera is static: one frame is enough
            res = table_height(m, d, a.episode, serial, fr, cam)
            if res["z_median"] is None:  # skipped frames are kept apart so every consumer of "results" stays valid
                out.setdefault("skipped", []).append(res)
                print("skipped", json.dumps(res))
                continue
            res["minus_sim_cm"] = 100 * (res["z_median"] - fit["table"]["z"])
            out["results"].append(res)
            print(json.dumps({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in res.items()}))
    json.dump(out, open(os.path.join(a.scene, "stereo_table_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
