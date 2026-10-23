"""Block size and colour in the wrist view: real wrist frames vs the sim posed at the same recorded joints (10-23).

For each wrist frame before the first real grasp, the sim is posed at the replayed state (replay_states.npz) and its
wrist camera renders a segmentation; the real block is found by colour (HSV range per block, kept to a window around
where the sim draws it, largest connected blob). Pixels covered by the sim robot are left out of both masks, and a
frame is skipped for a block whose sim mask touches the image border or is under --min_px. Per block:
linear size ratio real/sim = sqrt(real px / sim px), median over frames, and the median real RGB in the wrist frames
next to the sim colour (scene_fit rgb, which was taken from the exterior frame 0).

The ratio also absorbs any distance error of the block (a block 1 cm too close at ~25 cm reads ~4% large).
Writes <scene>/block_size_check.json and block_size_check.jpg (contours: white = sim, black = real).
Exploratory, UNVERIFIED.

    MUJOCO_GL=osmesa .venvs/real2sim/bin/python experiments/real2sim/block_size_check.py \
        --scene runs/real2sim/ep_orange_block_stereo --episode data/real2sim/droid/ep_orange_block
"""
import argparse, json, os
import cv2, h5py, mujoco
import numpy as np

# OpenCV HSV (H 0-180): read off the ep_orange_block wrist frames 10-23; the real "green" block is teal, "yellow" is lime
HSV = {"yellow block": [((25, 70, 70), (45, 255, 255))],
       "orange block": [((0, 130, 70), (10, 255, 255)), ((170, 130, 70), (180, 255, 255))],
       "green block": [((75, 70, 40), (100, 255, 255))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--step", type=int, default=2, help="every n-th frame before the first grasp")
    ap.add_argument("--window", type=int, default=60, help="px the real blob may lie outside the sim mask's box")
    ap.add_argument("--min_px", type=int, default=400, help="skip a block drawn smaller than this in the sim")
    a = ap.parse_args()
    m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
    fit = json.load(open(os.path.join(a.scene, "scene_fit.json")))
    st = np.load(os.path.join(a.scene, "replay_states.npz"))
    g = h5py.File(os.path.join(a.episode, "trajectory.h5"))["observation/robot_state/gripper_position"][:]
    k_close = int(np.argmax(g > 0.5))
    cid = m.camera("wrist").id
    W, H = (int(v) for v in m.cam_sensorsize[cid])
    rend = mujoco.Renderer(m, H, W)
    rend.enable_segmentation_rendering()
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1
    blocks = {lab: o for lab, o in fit["objects"].items() if lab in HSV}
    bgeoms = {lab: [i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body(o["body"]).id] for lab, o in blocks.items()}
    robot = [i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name.startswith(("fr3", "g_"))]
    d = mujoco.MjData(m)
    cap = cv2.VideoCapture(os.path.join(a.episode, f"{fit['wrist_camera']['serial']}.mp4"))
    rows, tiles = {lab: [] for lab in blocks}, []
    for k in range(0, k_close - 2, a.step):
        d.qpos[:] = st["qpos"][k]
        mujoco.mj_forward(m, d)
        rend.update_scene(d, camera="wrist", scene_option=opt)
        seg = rend.render()
        geom = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(st["real_frames"][k]))
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read wrist frame {int(st['real_frames'][k])}")
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        occ = np.isin(geom, robot)
        vis = bgr.copy()
        for lab in blocks:
            sim = np.isin(geom, bgeoms[lab])
            if sim.sum() < a.min_px or sim[0].any() or sim[-1].any() or sim[:, 0].any() or sim[:, -1].any():
                continue
            ys, xs = np.nonzero(sim)
            gate = np.zeros_like(sim)
            gate[max(ys.min() - a.window, 0):ys.max() + a.window, max(xs.min() - a.window, 0):xs.max() + a.window] = True
            col = np.zeros(sim.shape, np.uint8)
            for lo, hi in HSV[lab]:
                col |= cv2.inRange(hsv, lo, hi)
            col = cv2.morphologyEx(col, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            col[~gate | occ] = 0
            n, cc, stats, _ = cv2.connectedComponentsWithStats(col)
            if n < 2:
                continue
            real = cc == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if real[0].any() or real[-1].any() or real[:, 0].any() or real[:, -1].any():
                continue
            sim_v = sim & ~occ
            inner = cv2.erode(real.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            rgb = np.median(bgr[inner][:, ::-1], 0) / 255.0 if inner.sum() > 50 else None
            rows[lab].append(dict(frame=int(st["real_frames"][k]), sim_px=int(sim_v.sum()), real_px=int(real.sum()),
                                  lin_ratio=float(np.sqrt(real.sum() / max(sim_v.sum(), 1))),
                                  real_rgb=None if rgb is None else rgb.round(3).tolist()))
            for msk, c in ((sim_v, (255, 255, 255)), (real, (0, 0, 0))):
                cs, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(vis, cs, -1, c, 3)
        if len(tiles) < 12 and k % (3 * a.step) == 0:
            tiles.append(cv2.putText(cv2.resize(vis, (W // 3, H // 3)), str(int(st["real_frames"][k])), (8, 28),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2))
    out = dict(scene=a.scene, episode=a.episode, frames_before_grasp=k_close, step=a.step, hsv=HSV, blocks={},
               status="exploratory, UNVERIFIED")
    for lab, r in rows.items():
        lr = np.array([x["lin_ratio"] for x in r])
        rgbs = np.array([x["real_rgb"] for x in r if x["real_rgb"] is not None])
        edge = 2 * blocks[lab]["half"][0]
        out["blocks"][lab] = dict(n_frames=len(r), sim_edge_cm=round(100 * edge, 2),
                                  lin_ratio_median=float(np.median(lr)) if len(r) else None,
                                  lin_ratio_iqr=np.percentile(lr, [25, 75]).tolist() if len(r) else None,
                                  implied_edge_cm=round(100 * edge * float(np.median(lr)), 2) if len(r) else None,
                                  sim_rgb=blocks[lab]["rgb"], real_wrist_rgb_median=np.median(rgbs, 0).round(3).tolist() if len(rgbs) else None,
                                  frames=r)
        b = out["blocks"][lab]
        print(f"{lab}: {b['n_frames']} frames, sim edge {b['sim_edge_cm']} cm, linear ratio real/sim {b['lin_ratio_median']} "
              f"(IQR {b['lin_ratio_iqr']}) -> {b['implied_edge_cm']} cm; rgb sim {np.round(b['sim_rgb'], 2).tolist()} real wrist {b['real_wrist_rgb_median']}")
    json.dump(out, open(os.path.join(a.scene, "block_size_check.json"), "w"), indent=1)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(os.path.join(a.scene, "block_size_check.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main()
