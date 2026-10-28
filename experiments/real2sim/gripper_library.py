"""Real-gripper library for the wrist view (G6 step 1, 10-20).

The Robotiq gripper is rigidly mounted relative to the DROID wrist camera, so with the gripper OPEN it occupies the
same pixels in every real wrist frame; only the background behind it changes. The per-pixel median over many open
frames is therefore a clean, sharp picture of the real gripper (its real colour, bolts, focus blur), and its dark
pixels (median luminance < --dark_max, the gripper is black anodised; the median background is bright wood) are the
gripper's mask, restricted to a band around the sim gripper silhouette, cleaned and feathered.

Only the open gripper is built (10-20, checked by eye on the median images): partly or fully closed frames hold
objects (an apple, blocks) that get baked into the median, and fingers at different openings within a bin blur
together. For closing/closed the renderer falls back to the sim gripper with focus blur. The approach phase, where
G6 measured the effect, has the gripper open.

Frames from several episodes of the same robot + wrist camera can be pooled (ep_orange_block and ep_c3: robot
FR3-295341-1326595, wrist ZED Mini 18026681). Writes <out>/gripper_library.npz (rgb, alpha at 1280x720, the opening
range) + .json + preview.jpg.

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/gripper_library.py \\
        --scene runs/real2sim/ep_orange_block_stereo \\
        --episodes data/real2sim/droid/ep_orange_block data/real2sim/droid/ep_c3 --out runs/real2sim/gripper_lib_18026681
"""
import argparse, json, os
import numpy as np
import cv2
import h5py
import mujoco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="a scene with the wrist camera (for the sim gripper silhouette)")
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--serial", default=None, help="wrist camera serial; default: the scene's (scene_fit wrist_camera; our rig: 18026681)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--open_max", type=float, default=0.02, help="recorded gripper_position up to which the gripper counts as open")
    ap.add_argument("--max_frames", type=int, default=150)
    ap.add_argument("--dark_max", type=float, default=60.0, help="median luminance below which a pixel is gripper")
    ap.add_argument("--band", type=int, default=81, help="px: only pixels within this dilation of the sim gripper silhouette")
    a = ap.parse_args()
    if a.serial is None:
        a.serial = json.load(open(os.path.join(a.scene, "scene_fit.json")))["wrist_camera"]["serial"]
    os.makedirs(a.out, exist_ok=True)
    m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
    d = mujoco.MjData(m)
    cid = m.camera("wrist").id
    W, H = (int(v) for v in m.cam_sensorsize[cid])
    grip = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("g_")]  # gripper only (camera is on link7)
    rend = mujoco.Renderer(m, H, W)
    rend.enable_segmentation_rendering()
    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)
    rend.update_scene(d, camera=cid)
    seg = rend.render()
    rend.close()
    sim = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], grip)
    band = cv2.dilate(sim.astype(np.uint8), np.ones((a.band, a.band), np.uint8)) > 0

    sel = []
    for ep in a.episodes:
        g = h5py.File(os.path.join(ep, "trajectory.h5"))["observation/robot_state/gripper_position"][:]
        sel += [(ep, k) for k, v in enumerate(g) if v <= a.open_max]
    rng = np.random.default_rng(0)
    if len(sel) > a.max_frames:
        sel = [sel[i] for i in sorted(rng.choice(len(sel), a.max_frames, replace=False))]
    imgs, caps = [], {}
    for ep, k in sel:
        cap = caps.setdefault(ep, cv2.VideoCapture(os.path.join(ep, f"{a.serial}.mp4")))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, bgr = cap.read()
        if ok:
            imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    med = np.median(np.stack(imgs), axis=0).astype(np.uint8)
    dark = (med.astype(np.float32).mean(-1) < a.dark_max) & band
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark)
    keep = np.zeros_like(dark)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 3000:
            keep[lab == i] = 1
    alpha = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.5)
    iou = float(((alpha > 0.5) & sim).sum() / max(((alpha > 0.5) | sim).sum(), 1))
    np.savez_compressed(os.path.join(a.out, "gripper_library.npz"), rgb=med[None], alpha=alpha[None].astype(np.float32),
                        open_max=np.array([a.open_max]))
    prev = np.concatenate([med, (med.astype(np.float32) * alpha[..., None] + np.array([255, 0, 255], np.float32) * (1 - alpha[..., None])).astype(np.uint8)], 1)
    cv2.imwrite(os.path.join(a.out, "preview.jpg"), cv2.cvtColor(cv2.resize(prev, (1280, 360)), cv2.COLOR_RGB2BGR))
    rep = dict(episodes=a.episodes, serial=a.serial, open_max=a.open_max, n_frames=len(imgs),
               frames_per_episode={os.path.basename(e): sum(1 for x in sel if x[0] == e) for e in a.episodes},
               dark_max=a.dark_max, band=a.band, alpha_frac=float((alpha > 0.5).mean()), sim_gripper_frac=float(sim.mean()),
               iou_real_vs_sim_gripper=iou, status="exploratory, UNVERIFIED")
    json.dump(rep, open(os.path.join(a.out, "gripper_library.json"), "w"), indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
