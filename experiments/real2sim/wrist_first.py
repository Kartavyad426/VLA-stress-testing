"""Wrist-first placement pilot (SCENE_V1_MECHANISMS §19; user 10-27: "start a pilot on wrist first thing").

Measurement only, no scene build. Places every object the wrist camera sees well from wrist stereo alone (pose from
the recorded joints), then compares with where the scene's exterior-based build put it.

Step (a), detection cleanup vs wrist_anchors.py: one detector call per head noun ("block", "bowl"; no colour words,
K4), more frames before the first touch, instances matched across frames by 3D distance (not one 3 cm radius on a
surface median), instances seen in fewer than --min_frames frames dropped, detections off the table height dropped.
Step (b), the fit: each instance's stereo points from all its frames are merged and fitted with build_scene's own
fit_box (blocks, cube=True) or fit_ellipsoid (everything else; bowls: footprint + height only).
Output: <out>/wrist_first.json (per instance: frames, fitted centre/size/yaw, nearest exterior-built object of the
same noun and the distances) and <out>/wrist_first.jpg (wrist frames with the matched instances).

    runs/real2sim/v1_dev/py_triton_first.sh experiments/real2sim/wrist_first.py \
        --scene runs/real2sim/ep_orange_block_stereo --episode data/real2sim/droid/ep_orange_block \
        --out runs/real2sim/v1_dev/wrist_first/orange
Exploratory (res31, 10-27), UNVERIFIED.
"""
import argparse, json, os, sys
import numpy as np
import cv2
import h5py
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stereo_depth import stereo_pair, disparity, BASELINE_M  # noqa: E402
from composite import cam_K, cam_pose  # noqa: E402
from episode_info import grasp_events  # noqa: E402
from build_scene import fit_box, fit_ellipsoid  # noqa: E402
from block_size_check import head_noun  # noqa: E402
from scene_roles import table_label, gripper_label  # noqa: E402

COLS = [(255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 220, 60), (255, 80, 255), (60, 255, 255), (255, 150, 60), (180, 120, 255)]


def frames_before_touch(episode, n):
    g = h5py.File(os.path.join(episode, "trajectory.h5"))["observation/robot_state/gripper_position"][:]
    ev = grasp_events(g)
    fc = ev[0][0] if ev else len(g)
    return sorted({int(round(x)) for x in np.linspace(0, 0.7 * fc, n)})


def track(obs, radius):
    """Greedy association of per-frame instances (same noun) by distance between their xy centres."""
    tracks = []
    for o in sorted(obs, key=lambda o: o["frame"]):
        best, bd = None, radius
        for t in tracks:
            if any(x["frame"] == o["frame"] for x in t):
                continue
            dd = float(np.linalg.norm(np.median([x["xy"] for x in t], 0) - o["xy"]))
            if dd < bd:
                best, bd = t, dd
        (best.append(o) if best is not None else tracks.append([o]))
    return tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="built scene (wrist camera model, scene_fit.json to compare with)")
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_frames", type=int, default=10, help="wrist frames spread over [0, 0.7 x first close]")
    ap.add_argument("--min_frames", type=int, default=2, help="drop instances seen in fewer frames")
    ap.add_argument("--radius", type=float, default=0.04, help="max xy distance (m) to join an instance across frames")
    ap.add_argument("--table_z", type=float, default=None, help="default: median wrist z of <scene>/stereo_table_check.json")
    ap.add_argument("--min_px", type=int, default=150)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from perceive import detect_multi, segment

    m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
    d = mujoco.MjData(m)
    fit = json.load(open(os.path.join(a.scene, "scene_fit.json")))
    serial = fit["wrist_camera"]["serial"]
    pj = os.path.join(a.scene, "perception.json")
    pinfo = json.load(open(pj)) if os.path.exists(pj) else {}
    skip = {table_label(pinfo), gripper_label(pinfo), "wooden table", "robot gripper"}
    nouns = sorted({head_noun(k) for k in fit["objects"] if k.split("#")[0] not in skip})
    if a.table_z is None:
        st = json.load(open(os.path.join(a.scene, "stereo_table_check.json")))["results"]
        a.table_z = float(np.median([r["z_median"] for r in st if r["cam"] == "wrist"]))
    frames = frames_before_touch(a.episode, a.n_frames)
    print("nouns", nouns, "frames", frames, "table_z", round(a.table_z, 4), flush=True)
    J = h5py.File(os.path.join(a.episode, "trajectory.h5"))["observation/robot_state/joint_positions"][:]
    arm = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    cid = m.camera("wrist").id

    obs, imgs = {nn: [] for nn in nouns}, {}
    dropped = []
    for fr in frames:
        q = m.qpos0.copy()
        q[arm] = J[fr]
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        L, R_img = stereo_pair(a.episode, serial, fr)
        H, W = L.shape[:2]
        K = cam_K(m, cid, W, H)
        Rc, tc = cam_pose(d, cid)
        disp = disparity(L, R_img, num_disp=320)
        rgb = cv2.cvtColor(L, cv2.COLOR_BGR2RGB)
        imgs[fr] = rgb
        for nn in nouns:  # one call per noun (K4)
            dets = detect_multi(rgb, [nn])
            masks = segment(rgb, dets)
            for name, mk in masks.items():
                er = cv2.erode(mk.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                sel = er & (disp > 1)
                if sel.sum() < a.min_px:
                    dropped.append(dict(frame=fr, det=name, why="few stereo px"))
                    continue
                v, u = np.nonzero(sel)
                z = K[0, 0] * BASELINE_M["wrist"] / disp[sel]
                pc = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1)
                pw = pc @ Rc.T + tc
                zm = float(np.median(pw[:, 2]))
                if not (-0.03 <= zm - a.table_z <= 0.15):
                    dropped.append(dict(frame=fr, det=name, why=f"off table height (z {zm:.3f})"))
                    continue
                edge = bool(mk[0].any() or mk[-1].any() or mk[:, 0].any() or mk[:, -1].any())
                obs[nn].append(dict(frame=fr, det=name, score=dets[name]["score"], px=int(sel.sum()), edge=edge,
                                    xy=np.median(pw[:, :2], 0), pts=pw, mask=mk))
        print(f"frame {fr}: " + ", ".join(f"{nn} {sum(o['frame'] == fr for o in obs[nn])}" for nn in nouns), flush=True)

    # exterior-built objects to compare with
    ext = {}
    for lab, o in fit["objects"].items():
        c = o.get("center") or o.get("pos")
        if c is not None:
            ext.setdefault(head_noun(lab), []).append((lab, np.array(c[:2], float), o))
    results, kept = [], []
    for nn in nouns:
        for t in track(obs[nn], a.radius):
            frs = [o["frame"] for o in t]
            if len(t) < a.min_frames:
                dropped.append(dict(frames=frs, det=nn, why=f"seen in {len(t)} frame(s) < {a.min_frames}"))
                continue
            full = [o for o in t if not o["edge"]] or t  # prefer views where the object isn't cut by the image edge
            P = np.concatenate([o["pts"] for o in full])
            if len(P) > 40000:
                P = P[np.random.default_rng(0).choice(len(P), 40000, replace=False)]
            f_ = fit_box(P, a.table_z, cube=True) if nn in ("block", "cube") else fit_ellipsoid(P, a.table_z)
            r = dict(noun=nn, frames=frs, n_full_view=len(full), score_median=float(np.median([o["score"] for o in t])),
                     center=f_["center"], half=f_["half"], yaw=f_["yaw"],
                     frame_spread_m=float(np.max([np.linalg.norm(o["xy"] - np.median([x["xy"] for x in t], 0)) for o in t])))
            if ext.get(nn):
                lab, c, o = min(ext[nn], key=lambda e: np.linalg.norm(e[1] - np.array(f_["center"][:2])))
                r.update(exterior_match=lab, exterior_center=c.tolist(),
                         horiz_vs_exterior_cm=100 * float(np.linalg.norm(c - np.array(f_["center"][:2]))))
                ext_r = o.get("rim_radius") or (np.max(o["half"][:2]) if o.get("half") else None)
                if ext_r:
                    r["size_ratio_vs_exterior"] = float(np.max(f_["half"][:2]) / ext_r)
            results.append(r)
            kept.append(t)
    unmatched_ext = [lab for nn_, lst in ext.items() for lab, c, o in lst
                     if not any(r.get("exterior_match") == lab for r in results)]
    out = dict(scene=a.scene, episode=a.episode, frames=frames, nouns=nouns, table_z=a.table_z,
               status="pilot, exploratory, UNVERIFIED", instances=results,
               exterior_objects_not_seen_by_wrist=unmatched_ext, dropped=dropped)
    json.dump(out, open(os.path.join(a.out, "wrist_first.json"), "w"), indent=1, default=float)

    # overlay: up to 6 frames, each kept instance in its own colour
    show = frames[:: max(1, len(frames) // 6)][:6]
    tiles = []
    for fr in show:
        im = imgs[fr].copy()
        for k, t in enumerate(kept):
            for o in t:
                if o["frame"] == fr:
                    col = COLS[k % len(COLS)]
                    im[o["mask"]] = (0.55 * im[o["mask"]] + 0.45 * np.array(col)).astype(np.uint8)
                    v, u = np.nonzero(o["mask"])
                    cv2.putText(im, f"{results[k]['noun']}{k}", (int(u.mean()), int(v.mean())), 0, 0.8, col, 2)
        cv2.putText(im, f"frame {fr}", (10, 30), 0, 1.0, (255, 255, 255), 2)
        tiles.append(cv2.resize(im, (640, 360)))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    cv2.imwrite(os.path.join(a.out, "wrist_first.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    for r in results:
        print(r["noun"], "frames", r["frames"], "centre", np.round(r["center"], 3).tolist(), "half",
              np.round(r["half"], 3).tolist(), "vs", r.get("exterior_match"), f"{r.get('horiz_vs_exterior_cm', float('nan')):.1f} cm",
              f"size x{r.get('size_ratio_vs_exterior', float('nan')):.2f}")
    print("exterior objects the wrist never placed:", unmatched_ext)
    print("dropped:", len(dropped))


if __name__ == "__main__":
    main()
