"""Real-to-sim step 2a: calibrate the fixed exterior camera against the robot itself.

DROID's per-episode camera extrinsics are noisy (DROID annotation README, KarlP/droid), and the
replay of ep_orange_block showed the sim arm offset from the real one. This script refines the
camera pose (6 DoF) so the rendered FR3+2F-85 silhouette at the recorded joint angles matches the
real robot's mask in several frames. Intrinsics are the ZED factory values from KarlP/droid
intrinsics.json (fx, cx, fy, cy), not estimated.

Steps
  0. check the MuJoCo camera-intrinsics convention: render a sphere at a known 3D point and
     compare its pixel to the pinhole projection (must agree to < 1.5 px, else abort)
  1. real robot masks: Grounding DINO ("robot arm", "robot gripper") + SAM, per frame (cached)
  2. minimise a symmetric chamfer cost between real and sim silhouettes over the 6 pose parameters
Writes <out>/calib.json and <out>/calib_check.png (sim outline on real frames, before | after).

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/calibrate_cam.py \
        --episode data/real2sim/droid/ep_orange_block --cam 22008760 --out runs/real2sim/ep_orange_block
"""
import argparse, json, os, sys
import numpy as np
import cv2
import h5py
import mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perceive import read_frame, detect, segment  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALIB = os.path.join(REPO, "data/real2sim/droid/calib")
FRAMES = [30, 45, 60, 72, 90, 110, 130]  # frame 0 dropped: robot barely visible, motion mask picks up wall flicker
DOWN = 4  # silhouettes are compared at 1/4 resolution


def episode_id(episode_dir):
    meta = [f for f in os.listdir(episode_dir) if f.startswith("metadata_")][0]
    return json.load(open(os.path.join(episode_dir, meta)))["uuid"]


def zed_K(episode_dir, cam):
    local = os.path.join(episode_dir, "intrinsics.json")  # non-DROID episodes: {cam: [fx, cx, fy, cy]}
    if os.path.exists(local):
        fx, cx, fy, cy = json.load(open(local))[cam]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    fx, cx, fy, cy = json.load(open(os.path.join(CALIB, "intrinsics.json")))[episode_id(episode_dir)][cam]["cameraMatrix"]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def set_camera(m, cid, R, t, K, W, H, principal_mode):
    """R, t: OpenCV camera pose in the base frame. Intrinsics via sensorsize/intrinsic in pixel units."""
    m.cam_pos[cid] = t
    m.cam_quat[cid] = Rot.from_matrix(R @ np.diag([1, -1, -1])).as_quat(scalar_first=True)
    m.cam_sensorsize[cid] = [W, H]
    m.cam_resolution[cid] = [W, H]
    if principal_mode == "offset":  # principal point as offset from the image centre
        px, py = K[0, 2] - W / 2, -(K[1, 2] - H / 2)
    else:
        px, py = K[0, 2], K[1, 2]
    m.cam_intrinsic[cid] = [K[0, 0], K[1, 1], px, py]


def project(K, R, t, p):
    pc = (np.atleast_2d(p) - t) @ R
    return pc[:, :2] / pc[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def check_convention(K, R, t, W, H):
    """Which principal-point convention makes MuJoCo agree with the pinhole projection?"""
    pts = [t + R @ np.array(v) for v in ([0.15, 0.10, 0.8], [-0.2, -0.12, 0.7], [0.0, 0.0, 0.9])]
    errs = {}
    for mode in ("offset", "absolute"):
        worst = 0.0
        for p in pts:
            xml = f"""<mujoco><worldbody><geom type="sphere" size="0.004" pos="{p[0]} {p[1]} {p[2]}" rgba="1 1 1 1"/>
            <camera name="c"/></worldbody><visual><global offwidth="{W}" offheight="{H}"/></visual></mujoco>"""
            m = mujoco.MjModel.from_xml_string(xml)
            d = mujoco.MjData(m)
            set_camera(m, 0, R, t, K, W, H, mode)
            mujoco.mj_forward(m, d)
            r = mujoco.Renderer(m, H, W)
            r.enable_segmentation_rendering()
            r.update_scene(d, "c")
            seg = r.render()[..., 0]
            r.close()
            ys, xs = np.nonzero(seg >= 0)
            got = np.array([xs.mean(), ys.mean()]) if len(xs) else np.array([np.inf, np.inf])
            worst = max(worst, float(np.linalg.norm(got - project(K, R, t, p)[0])))
        errs[mode] = worst
    return errs


def motion_masks(video_path, frames, thr=35):
    """Robot masks by background subtraction: the camera is static and the robot is the main moving thing.
    Plate = per-pixel temporal median. Shadows (darker than the plate, same chromaticity) are rejected."""
    from replay import read_video
    vid = read_video(video_path)
    plate = np.median(np.stack(vid[::max(1, len(vid) // 150)]), axis=0).astype(np.float32)  # <=~150 frames: memory
    fr = {i: vid[i] for i in frames}
    pc = plate / (plate.sum(-1, keepdims=True) + 1e-3)
    out = {}
    for i in frames:
        x = fr[i].astype(np.float32)
        moving = np.abs(x - plate).max(-1) > thr
        ratio = x.sum(-1) / (plate.sum(-1) + 1e-3)
        chroma = np.abs(x / (x.sum(-1, keepdims=True) + 1e-3) - pc).sum(-1)
        shadow = (ratio > 0.2) & (ratio < 0.92) & (chroma < 0.06)
        m = (moving & ~shadow).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] > 2000]  # drop specks
        out[str(i)] = np.isin(lab, keep)
    return out, plate.astype(np.uint8)


def main():
    global FRAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--masks", choices=["motion", "dino"], default="motion",
                    help="motion: background subtraction (default); dino: Grounding DINO + SAM (unreliable on this episode)")
    ap.add_argument("--frames", default=",".join(map(str, FRAMES)), help="robot-visible frames to fit, comma-separated")
    ap.add_argument("--init_sphere", action="store_true", help="coarse pose search when no starting extrinsics exist")
    ap.add_argument("--fit_focal", action="store_true", help="also fit the focal length (unknown intrinsics)")
    ap.add_argument("--init", default=None, help="start from this calib.json instead of DROID's extrinsics")
    ap.add_argument("--level_weight", type=float, default=0.0,
                    help="px of cost per degree of table tilt (assumes robot base and table are level); 0 = off")
    ap.add_argument("--span_deg", type=float, default=15.0)
    ap.add_argument("--span_m", type=float, default=0.25)
    ap.add_argument("--search", type=int, default=2000, help="random camera poses scored before local refinement")
    a = ap.parse_args()
    FRAMES = [int(x) for x in a.frames.split(",")]

    f = h5py.File(os.path.join(a.episode, "trajectory.h5"))
    jp = f["observation/robot_state/joint_positions"][()]
    ex0 = f[f"observation/camera_extrinsics/{a.cam}_left"][0]
    R0, t0 = Rot.from_euler("xyz", ex0[3:]).as_matrix(), np.asarray(ex0[:3])
    if a.init:  # refine from an earlier calibration instead of DROID's recorded extrinsics
        c0 = json.load(open(a.init))
        R0, t0 = np.array(c0["R_cam_to_base"]), np.array(c0["t_cam_in_base"])
    K = zed_K(a.episode, a.cam)
    n_cam = None
    if a.level_weight > 0:
        # table normal in the camera frame (MoGe depth, ZED intrinsics; independent of the depth scale)
        per = np.load(os.path.join(a.out, "perception.npz"))
        Hh, Ww = per["depth"].shape
        uu, vv = np.meshgrid(np.arange(Ww), np.arange(Hh))
        zz = per["depth"]
        P = np.stack([(uu - K[0, 2]) / K[0, 0] * zz, (vv - K[1, 2]) / K[1, 1] * zz, zz], -1)
        tm = cv2.erode(per["mask/wooden table"].astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & per["valid"]
        tp = P[tm]
        n_cam = np.linalg.svd(tp - tp.mean(0), full_matrices=False)[2][-1]

    def tilt_deg(R):
        """angle between the table normal (in the robot frame) and vertical: 0 for a level table"""
        nb = R @ n_cam
        return float(np.degrees(np.arccos(min(1.0, abs(nb[2])))))
    W, H = 1280, 720
    w, h = W // DOWN, H // DOWN
    Kd = K.copy()
    Kd[:2] /= DOWN

    # 0. convention check
    errs = check_convention(Kd, R0, t0, w, h)
    mode = min(errs, key=errs.get)
    print("intrinsics convention check (worst px error):", {k: round(v, 2) for k, v in errs.items()})
    if errs[mode] > 1.5:
        sys.exit(f"ABORT: no principal-point convention matches the pinhole projection ({errs})")

    # 1. real robot masks
    cache = os.path.join(a.out, f"robot_masks_{a.masks}.npz")
    if os.path.exists(cache):
        real = dict(np.load(cache))
    elif a.masks == "motion":
        real, plate = motion_masks(os.path.join(a.episode, f"{a.cam}.mp4"), FRAMES)
        np.savez_compressed(cache, **real)
        cv2.imwrite(os.path.join(a.out, "plate_median.png"), cv2.cvtColor(plate, cv2.COLOR_RGB2BGR))
    else:
        real = {}
        for i in FRAMES:
            img = read_frame(os.path.join(a.episode, f"{a.cam}.mp4"), i)
            masks = segment(img, detect(img, ["robot arm", "robot gripper"]))
            mk = np.zeros(img.shape[:2], bool)
            for v in masks.values():
                mk |= v
            real[str(i)] = mk
        np.savez_compressed(cache, **real)
    real_small = {i: cv2.resize(real[str(i)].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) for i in FRAMES}
    dt_real = {i: cv2.distanceTransform((~real_small[i]).astype(np.uint8), cv2.DIST_L2, 3) for i in FRAMES}

    # 2. sim silhouettes
    m = mujoco.MjModel.from_binary_path(os.path.join(a.out, "scene.mjb"))
    d = mujoco.MjData(m)
    cid = m.camera("ext").id
    arm_q = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    robot_geoms = np.array([g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(("fr3", "g_"))])
    rend = mujoco.Renderer(m, h, w)
    rend.enable_segmentation_rendering()

    def sim_masks(x):
        R = Rot.from_rotvec(x[:3]).as_matrix() @ R0
        t = t0 + x[3:6]
        Kx = Kd.copy()
        if len(x) > 6:  # --fit_focal: x[6] = log focal-length scale
            Kx[0, 0] *= np.exp(x[6])
            Kx[1, 1] *= np.exp(x[6])
        set_camera(m, cid, R, t, Kx, w, h, mode)
        out = {}
        for i in FRAMES:
            d.qpos[arm_q] = jp[i]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
            rend.update_scene(d, camera=cid)
            seg = rend.render()
            out[i] = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], robot_geoms)
        return out

    n_eval = [0]

    def cost(x):
        n_eval[0] += 1
        if n_eval[0] % 100 == 0:
            print(f"eval {n_eval[0]}", flush=True)
        sm = sim_masks(x)
        c = 0.0
        for i in FRAMES:
            if not sm[i].any():  # robot rendered out of view: distance transform of an empty mask is unbounded
                c += 200.0
                continue
            dt_sim = cv2.distanceTransform((~sm[i]).astype(np.uint8), cv2.DIST_L2, 3)
            c += float(dt_real[i][sm[i]].mean()) + float(dt_sim[real_small[i]].mean())
        c /= len(FRAMES)
        if n_cam is not None:  # assumption: the robot base and the table are both level
            c += a.level_weight * tilt_deg(Rot.from_rotvec(x[:3]).as_matrix() @ R0)
        return c

    def iou(x):
        sm = sim_masks(x)
        return float(np.mean([(sm[i] & real_small[i]).sum() / max(1, (sm[i] | real_small[i]).sum()) for i in FRAMES]))

    if a.init_sphere:
        # no usable starting pose (non-DROID data): cameras on spheres around the robot's working area, looking at
        # it with the image's up direction pointing up; the best by IoU becomes the start for the search below
        dk = mujoco.MjData(m)
        pts = []
        for i in FRAMES:
            dk.qpos[arm_q] = jp[i]
            mujoco.mj_kinematics(m, dk)
            pts.append(dk.site_xpos[m.site("g_pinch").id].copy())
        target = np.mean(pts, axis=0)
        best0 = (-1, None, None)
        for az in range(0, 360, 15):
            for el in (15, 30, 45, 60, 75):
                for dist in (0.7, 1.0, 1.3, 1.7, 2.2):
                    c = target + dist * np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)),
                                                  np.cos(np.radians(el)) * np.sin(np.radians(az)), np.sin(np.radians(el))])
                    zc = (target - c) / np.linalg.norm(target - c)  # OpenCV camera: z forward, y down
                    xc = np.cross(zc, [0, 0, 1.0])
                    xc /= np.linalg.norm(xc)
                    yc = np.cross(zc, xc)
                    R0, t0 = np.c_[xc, yc, zc], c
                    sc = iou(np.zeros(6))
                    if sc > best0[0]:
                        best0 = (sc, R0.copy(), t0.copy(), az, el, dist)
        _, R0, t0 = best0[:3]
        print(f"sphere init: IoU {best0[0]:.3f} at azimuth {best0[3]} elevation {best0[4]} distance {best0[5]} m "
              f"around {target.round(3).tolist()}", flush=True)
    nx = 7 if a.fit_focal else 6
    x0 = np.zeros(nx)
    c0, i0 = cost(x0), iou(x0)
    # global stage: DROID's extrinsics can be far off (ep_orange_block: Powell from x0 stalled at IoU 0.28),
    # so score random poses around them by IoU first and refine only the best few
    rng = np.random.default_rng(0)
    span = np.array([np.radians(a.span_deg)] * 3 + [a.span_m] * 3 + ([0.2] if a.fit_focal else []))
    cands = [x0] + [rng.uniform(-span, span) for _ in range(a.search)]
    scored = sorted(((iou(x), k) for k, x in enumerate(cands)), reverse=True)
    print("random search: best IoU", [round(s, 3) for s, _ in scored[:5]], flush=True)
    best = None
    for _, k in scored[:5]:
        res = minimize(cost, cands[k], method="Powell", options=dict(xtol=1e-4, ftol=1e-3, maxfev=1500))
        if best is None or res.fun < best.fun:
            best = res
    res = minimize(cost, best.x, method="Powell", options=dict(xtol=1e-4, ftol=1e-3, maxfev=3000))
    c1, i1 = cost(res.x), iou(res.x)
    R1 = Rot.from_rotvec(res.x[:3]).as_matrix() @ R0
    t1 = t0 + res.x[3:6]
    if a.fit_focal:
        K = K.copy()
        K[0, 0] *= np.exp(res.x[6])
        K[1, 1] *= np.exp(res.x[6])
    tilts = dict(tilt_before_deg=tilt_deg(R0), tilt_after_deg=tilt_deg(R1)) if n_cam is not None else {}
    calib = dict(cam=a.cam, K=K.tolist(), level_weight=a.level_weight, init=a.init, **tilts, R_cam_to_base=R1.tolist(), t_cam_in_base=t1.tolist(),
                 delta_rot_deg=float(np.degrees(np.linalg.norm(res.x[:3]))), delta_trans_m=res.x[3:6].tolist(), focal_scale=float(np.exp(res.x[6])) if a.fit_focal else 1.0,
                 cost_before_px=c0, cost_after_px=c1, iou_before=i0, iou_after=i1, frames=FRAMES,
                 principal_mode=mode, convention_check_px=errs, nfev=int(res.nfev))
    json.dump(calib, open(os.path.join(a.out, "calib.json"), "w"), indent=1)

    # check image: sim silhouette outline (green) on real frames, before | after
    rows = []
    for x in (x0, res.x):
        sm = sim_masks(x)
        tiles = []
        for i in FRAMES[1::2]:
            img = cv2.resize(read_frame(os.path.join(a.episode, f"{a.cam}.mp4"), i), (w, h))
            cnt, _ = cv2.findContours(sm[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.drawContours(img, cnt, -1, (0, 255, 0), 1)
            tiles.append(img)
        rows.append(np.concatenate(tiles, 1))
    cv2.imwrite(os.path.join(a.out, "calib_check.png"), np.concatenate(rows, 0))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in calib.items()
                      if k in ("delta_rot_deg", "delta_trans_m", "cost_before_px", "cost_after_px", "iou_before", "iou_after", "nfev")}))


if __name__ == "__main__":
    main()
