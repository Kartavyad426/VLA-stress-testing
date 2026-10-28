"""Object size and colour in the wrist view: real wrist frames vs the sim posed at the same recorded joints (10-23;
generalised 10-26).

For each sampled wrist frame before the first real grasp, the sim is posed at the replayed state (replay_states.npz) and
its wrist camera renders a segmentation. The real object is found by one of two methods:
  masks (default, 10-26): Grounding DINO + SAM in the real wrist frame, prompted with the object's own label from the
         scene (as wrist_anchors.py), keeping the instance that best overlaps where the sim draws the object. Works for
         any object and colour, including bowls.
  hsv    (10-23, ep_orange_block only): a colour range per block, read off that episode's wrist frames (the real "green"
         block is teal, "yellow" is lime), largest blob near the sim mask. Kept so the 10-23 result stays reproducible.
Pixels covered by the sim robot are left out of both masks; a frame is skipped for an object whose sim or real mask
touches the image border, or whose sim mask is under --min_px. Per object: linear size ratio real/sim =
sqrt(real px / sim px), median over frames, and the median real RGB in the wrist frames next to the sim colour (scene_fit
rgb, taken from the exterior frame 0). The ratio also absorbs any distance error (1 cm too close at ~25 cm reads ~4%).
Writes <scene>/block_size_check.json and block_size_check.jpg (contours: white = sim, black = real). Exploratory, UNVERIFIED.

    MUJOCO_GL=osmesa .venvs/real2sim/bin/python experiments/real2sim/block_size_check.py \
        --scene runs/real2sim/ep_orange_block_stereo --episode data/real2sim/droid/ep_orange_block
"""
import argparse, json, os, sys
try:  # 10-26: under MUJOCO_GL=osmesa, triton (loaded by transformers with the detector) segfaults if mujoco was loaded first
    import triton  # noqa: F401
except ImportError:
    pass
import cv2, h5py, mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --method hsv only. OpenCV HSV (H 0-180): read off the ep_orange_block wrist frames 10-23
HSV_ORANGE_EPISODE = {"yellow block": [((25, 70, 70), (45, 255, 255))],
                      "orange block": [((0, 130, 70), (10, 255, 255)), ((170, 130, 70), (180, 255, 255))],
                      "green block": [((75, 70, 40), (100, 255, 255))]}


def head_noun(label):
    """'green block#2' -> 'block', 'black bowl' -> 'bowl' (colour/size words as build_scene.COLOURS)."""
    from build_scene import COLOURS
    words = [w for w in label.split("#")[0].lower().split() if w not in COLOURS]
    return words[-1] if words else label.split("#")[0]


def touches_border(mk):
    return mk[0].any() or mk[-1].any() or mk[:, 0].any() or mk[:, -1].any()


def real_by_hsv(bgr, lab, gate, occ):
    col = np.zeros(gate.shape, np.uint8)
    for lo, hi in HSV_ORANGE_EPISODE[lab]:
        col |= cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), lo, hi)
    col = cv2.morphologyEx(col, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    col[~gate | occ] = 0
    n, cc, stats, _ = cv2.connectedComponentsWithStats(col)
    return None if n < 2 else cc == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def real_by_mask(cands, sim, gate, occ):
    """The detected instance (same label) that best overlaps the sim object's mask; None if none lies mostly in the gate."""
    best, score = None, 0.0
    for mk in cands:
        mk = mk & ~occ
        if mk.sum() == 0 or (mk & gate).sum() < 0.5 * mk.sum():
            continue
        iou = (mk & sim).sum() / max((mk | sim).sum(), 1)
        if iou > score:
            best, score = mk, iou
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--method", choices=["masks", "hsv"], default="masks", help="see the docstring; hsv = ep_orange_block's blocks only")
    ap.add_argument("--step", type=int, default=2, help="every n-th frame before the first grasp")
    ap.add_argument("--max_frames", type=int, default=15, help="masks: at most this many frames, evenly spread (DINO + SAM per frame)")
    ap.add_argument("--window", type=int, default=60, help="px the real object may lie outside the sim mask's box")
    ap.add_argument("--min_px", type=int, default=400, help="skip an object drawn smaller than this in the sim")
    a = ap.parse_args()
    m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
    fit = json.load(open(os.path.join(a.scene, "scene_fit.json")))
    st = np.load(os.path.join(a.scene, "replay_states.npz"))
    from episode_info import CLOSED
    g = h5py.File(os.path.join(a.episode, "trajectory.h5"))["observation/robot_state/gripper_position"][:]
    k_close = int(np.argmax(g > CLOSED)) if (g > CLOSED).any() else len(g)
    cid = m.camera("wrist").id
    W, H = (int(v) for v in m.cam_sensorsize[cid])
    rend = mujoco.Renderer(m, H, W)
    rend.enable_segmentation_rendering()
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1
    if a.method == "hsv":
        objs = {lab: o for lab, o in fit["objects"].items() if lab in HSV_ORANGE_EPISODE}
    else:
        from perceive import detect_multi, segment
        objs = dict(fit["objects"])
    root = {lab: m.body(o["body"]).id for lab, o in objs.items()}  # pack meshes sit on child bodies: match by root body
    ogeoms = {lab: [i for i in range(m.ngeom) if m.body_rootid[m.geom_bodyid[i]] == b] for lab, b in root.items()}
    robot = [i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name.startswith(("fr3", "g_"))]
    # masks: one detector call per head noun ("block", "bowl"), colour words dropped (10-26 debug on ep_orange_block: a
    # single query with all four labels returned nothing at frame 62; "orange block" matched the gripper's finger; the
    # real "green" block is teal). Instances are assigned to sim objects by overlap, so colour is measured, not assumed.
    noun = {lab: head_noun(lab) for lab in objs}
    prompts = sorted(set(noun.values()))
    d = mujoco.MjData(m)
    cap = cv2.VideoCapture(os.path.join(a.episode, f"{fit['wrist_camera']['serial']}.mp4"))
    ks = list(range(0, max(k_close - 2, 1), a.step))
    if a.method == "masks" and len(ks) > a.max_frames:
        ks = sorted({ks[int(round(i))] for i in np.linspace(0, len(ks) - 1, a.max_frames)})
    rows, tiles = {lab: [] for lab in objs}, []
    for n_k, k in enumerate(ks):
        d.qpos[:] = st["qpos"][k]
        mujoco.mj_forward(m, d)
        rend.update_scene(d, camera="wrist", scene_option=opt)
        seg = rend.render()
        geom = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(st["real_frames"][k]))
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read wrist frame {int(st['real_frames'][k])}")
        occ = np.isin(geom, robot)
        vis = bgr.copy()
        cands = {}
        if a.method == "masks":
            rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for pr in prompts:
                dets = detect_multi(rgb_img, [pr])
                cands[pr] = [mk.astype(bool) for mk in segment(rgb_img, dets).values()]
        for lab in objs:
            sim = np.isin(geom, ogeoms[lab])
            if sim.sum() < a.min_px or touches_border(sim):
                continue
            ys, xs = np.nonzero(sim)
            gate = np.zeros_like(sim)
            gate[max(ys.min() - a.window, 0):ys.max() + a.window, max(xs.min() - a.window, 0):xs.max() + a.window] = True
            if a.method == "hsv":
                real = real_by_hsv(bgr, lab, gate, occ)
            else:
                real = real_by_mask(cands.get(noun[lab], []), sim & ~occ, gate, occ)
            if real is None or touches_border(real):
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
        keep_tile = (n_k % 2 == 0) if a.method == "masks" else (k % (3 * a.step) == 0)
        if len(tiles) < 12 and keep_tile:
            tiles.append(cv2.putText(cv2.resize(vis, (W // 3, H // 3)), str(int(st["real_frames"][k])), (8, 28),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2))
    out = dict(scene=a.scene, episode=a.episode, method=a.method, frames_before_grasp=k_close, frames_used=[int(st["real_frames"][k]) for k in ks],
               prompts=prompts if a.method == "masks" else None, hsv=HSV_ORANGE_EPISODE if a.method == "hsv" else None,
               objects={}, status="exploratory, UNVERIFIED")
    for lab, r in rows.items():
        o = objs[lab]
        lr = np.array([x["lin_ratio"] for x in r])
        rgbs = np.array([x["real_rgb"] for x in r if x["real_rgb"] is not None])
        med = float(np.median(lr)) if len(r) else None
        res = dict(kind=o.get("kind"), n_frames=len(r), lin_ratio_median=med,
                   lin_ratio_iqr=np.percentile(lr, [25, 75]).tolist() if len(r) else None,
                   sim_rgb=o.get("rgb"), real_wrist_rgb_median=np.median(rgbs, 0).round(3).tolist() if len(rgbs) else None, frames=r)
        if o.get("kind") == "box":  # boxes: the edge the ratio implies
            edge = 2 * o["half"][0]
            res.update(sim_edge_cm=round(100 * edge, 2), implied_edge_cm=round(100 * edge * med, 2) if med else None)
        out["objects"][lab] = res
        print(f"{lab} ({res['kind']}): {res['n_frames']} frames, linear ratio real/sim {med} (IQR {res['lin_ratio_iqr']})"
              + (f", sim edge {res['sim_edge_cm']} -> {res['implied_edge_cm']} cm" if "sim_edge_cm" in res else "")
              + f"; rgb sim {np.round(res['sim_rgb'], 2).tolist() if res['sim_rgb'] else None} real wrist {res['real_wrist_rgb_median']}")
    out["blocks"] = {lab: v for lab, v in out["objects"].items() if v["kind"] == "box"}  # the 10-23 key, boxes only
    json.dump(out, open(os.path.join(a.scene, "block_size_check.json"), "w"), indent=1)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(os.path.join(a.scene, "block_size_check.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main()
