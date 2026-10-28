"""Exterior camera search on a sphere around the grasp points (SCENE_V1_MECHANISMS §21; user 10-28).

The camera sits at distance r from P (the mean of the real grasp points, FK of the recorded joints at each first
gripper close), at azimuth/elevation (az, el) around P, rolled about its viewing axis; its orientation is set so P
lands on the pixel of its real object in the perception frame. A coarse-to-fine grid over (az, el, r, roll) scores:
  outline  robot outline distance (px at 1/4 res) over frames spread over the episode   (as calibrate_cam's cost)
  grasp    grasp points to the centroid of the object mask each lands on (px, full res)
  tilt     the exterior table rebuilt with MoGe scaled by the grasp points: tilt from level (deg)
  dz       its height vs a trusted wrist-stereo table height (cm), only with --table_z
Writes <out>/camera_search.json, <out>/calib_search.json (calib.json format) and <out>/heatmaps.png.

    runs/real2sim/v1_dev/py_triton_first.sh experiments/real2sim/camera_search.py --scene <built scene> \
        --episode <ep> --out runs/real2sim/v1_dev/camera_search/<name> [--table_z -0.0127] [--init calib.json,...]
Exploratory (res31, 10-28), UNVERIFIED.
"""
import argparse, itertools, json, os, sys, time
import numpy as np
import cv2
import h5py
import mujoco
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_cam import set_camera, motion_masks, zed_K, DOWN  # noqa: E402
from build_scene import cam_pose as droid_pose  # noqa: E402
from episode_info import grasp_events  # noqa: E402
from scene_roles import table_label, gripper_label  # noqa: E402

WEIGHTS = dict(outline=1.0, grasp=0.25, tilt=3.0, dz=1.0)  # grasp px are full res: x 0.25 = the outline's 1/4 res


def look_rotation(C, P, uv, K, roll_deg):
    """R (camera -> base, OpenCV axes) with P seen at pixel uv, rolled about the viewing ray; roll 0 = image 'up'
    as close as possible to world +z."""
    d_c = np.linalg.solve(K, [uv[0], uv[1], 1.0])
    d_c /= np.linalg.norm(d_c)
    d_w = (P - C) / np.linalg.norm(P - C)
    # base frame for roll 0: camera z along d_w-ish; build a camera whose optical axis is d_w, y down, then rotate
    z = d_w
    x = np.cross(z, [0, 0, 1.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_axis = np.c_[x, y, z]  # optical axis on d_w
    # rotate inside the camera so the pixel ray d_c (not the axis) points at P
    a = np.cross([0, 0, 1.0], d_c)
    if np.linalg.norm(a) > 1e-9:
        R_off = Rot.from_rotvec(a / np.linalg.norm(a) * np.arccos(np.clip(d_c[2], -1, 1))).as_matrix()
    else:
        R_off = np.eye(3)
    R = R_axis @ R_off.T
    return Rot.from_rotvec(d_w * np.radians(roll_deg)).as_matrix() @ R


def params_of(R, t, P):
    v = t - P
    r = float(np.linalg.norm(v))
    az = float(np.degrees(np.arctan2(v[1], v[0])))
    el = float(np.degrees(np.arcsin(v[2] / r)))
    return az, el, r


def roll_of(R, t, P, uv, K):
    best, br = None, 0.0
    for rl in np.arange(-180, 180, 0.5):
        d = np.linalg.norm(look_rotation(t, P, uv, K, rl) - R)
        if best is None or d < best:
            best, br = d, rl
    return float(br)


class Search:
    def __init__(self, a):
        self.a = a
        self.m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
        self.d = mujoco.MjData(self.m)
        self.per = dict(np.load(os.path.join(a.scene, "perception.npz")))
        self.pinfo = json.load(open(os.path.join(a.scene, "perception.json")))
        self.H, self.W = self.per["image"].shape[:2]
        self.cam = a.cam
        self.K = zed_K(a.episode, self.cam)
        self.Kd = self.K.copy()
        self.Kd[:2] /= DOWN
        self.w, self.h = self.W // DOWN, self.H // DOWN
        f = h5py.File(os.path.join(a.episode, "trajectory.h5"))
        self.J = f["observation/robot_state/joint_positions"][:]
        G = f["observation/robot_state/gripper_position"][:]
        self.f = f
        self.arm = [self.m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
        self.cid = self.m.camera("ext").id
        self.robot_geoms = np.array([g for g in range(self.m.ngeom) if self.m.body(self.m.geom_bodyid[g]).name.startswith(("fr3", "g_"))])
        self.rend = mujoco.Renderer(self.m, self.h, self.w)
        self.rend.enable_segmentation_rendering()
        # grasp points: fingertip at every close. The gripper width can't tell an empty close (c9's thin mallet handle
        # reads 0.96-1.0 while held), so grasps are filtered in targets(): each object's FIRST grasp only (later ones may
        # re-grasp a moved object: c3 has 11), and only grasps that land near an object in the perception frame
        self.all_grasps = []
        for kc, ko in grasp_events(G):
            self.pose(kc)
            self.all_grasps.append((int(kc), self.d.site_xpos[self.m.site("g_pinch").id].copy()))
        if not self.all_grasps:
            raise SystemExit("no real grasp: nothing to anchor the sphere on")
        self.grasps = list(self.all_grasps)
        self.P = np.mean([p for _, p in self.grasps], 0)
        self.objmasks = {k[5:]: self.per[k] for k in self.per if k.startswith("mask/")
                         and k[5:] not in (table_label(self.pinfo), gripper_label(self.pinfo))}
        # frames: spread over the episode, robot visible (motion masks), clamped to the video
        vid = os.path.join(a.episode, f"{self.cam}.mp4")
        cap = cv2.VideoCapture(vid)
        n = min(len(self.J), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
        cand = sorted({int(x) for x in np.linspace(int(0.1 * n), n - 2, a.n_frames)})
        mm, _ = motion_masks(vid, cand)
        self.frames = [fr for fr in cand if mm[str(fr)].sum() > 3000]
        self.real_small, self.dt_real = {}, {}
        for fr in self.frames:
            rs = cv2.resize(mm[str(fr)].astype(np.uint8), (self.w, self.h), interpolation=cv2.INTER_NEAREST).astype(bool)
            self.real_small[fr] = rs
            self.dt_real[fr] = cv2.distanceTransform((~rs).astype(np.uint8), cv2.DIST_L2, 3)
        self.coarse_frames = self.frames[:: max(1, len(self.frames) // 12)]
        # table pixels (eroded) with MoGe depth, for the level / height terms
        tm = cv2.erode(self.per[f"mask/{table_label(self.pinfo)}"].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        tm &= self.per["valid"] & (self.per["depth"] > 0)
        v, u = np.nonzero(tm)
        keep = np.random.default_rng(0).choice(len(u), min(len(u), 20000), replace=False)
        self.tu, self.tv, self.tz = u[keep], v[keep], self.per["depth"][v[keep], u[keep]]

    def pose(self, fr):
        self.d.qpos[:] = self.m.qpos0
        self.d.qpos[self.arm] = self.J[fr]
        mujoco.mj_forward(self.m, self.d)

    def project(self, R, t, p):
        pc = (np.atleast_2d(p) - t) @ R
        return pc[:, :2] / pc[:, 2:3] * [self.K[0, 0], self.K[1, 1]] + [self.K[0, 2], self.K[1, 2]], pc[:, 2]

    def targets(self, R, t, max_px=80):
        """Filter the grasps under (R, t) and give each kept one its object mask and centroid: the mask it lands on,
        else the nearest within max_px; first grasp per mask only. Sets self.grasps and returns the targets."""
        out, kept, used, self.dropped = [], [], set(), []
        for fr, p in self.all_grasps:
            uv, _ = self.project(R, t, p)
            u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))
            best, bd = None, 1e9
            for lab, mk in self.objmasks.items():
                if 0 <= u < self.W and 0 <= v < self.H and mk[v, u]:
                    best, bd = lab, 0
                    break
                ys, xs = np.nonzero(mk)
                dd = np.hypot(xs.mean() - u, ys.mean() - v)
                if dd < bd:
                    best, bd = lab, dd
            if bd > max_px:
                self.dropped.append(dict(frame=fr, why=f"lands {bd:.0f} px from the nearest object"))
                continue
            if best in used:
                self.dropped.append(dict(frame=fr, why=f"second grasp of {best} (object may have moved)"))
                continue
            used.add(best)
            # 10-28: the grasp point is ON or IN the object (rim, handle), not its centre: the target pixel is where it
            # lands if that's inside the mask, else the mask pixel nearest to it; the grasp term is the distance to the
            # mask (0 inside), from a distance transform (was: distance to the mask centroid, which pulled c9's camera off)
            mk = self.objmasks[best]
            if not hasattr(self, "mask_dt"):
                self.mask_dt = {}
            if best not in self.mask_dt:
                self.mask_dt[best] = cv2.distanceTransform((~mk).astype(np.uint8), cv2.DIST_L2, 5)
            if 0 <= u < self.W and 0 <= v < self.H and mk[v, u]:
                tgt_px = [float(u), float(v)]
            else:
                ys, xs = np.nonzero(mk)
                i = int(np.argmin(np.hypot(xs - u, ys - v)))
                tgt_px = [float(xs[i]), float(ys[i])]
            out.append(dict(frame=fr, mask=best, centroid=tgt_px))
            kept.append((fr, p))
        if not kept:  # nothing near an object under this camera: keep the first grasp so the start can still be scored
            fr, p = self.all_grasps[0]
            ys, xs = np.nonzero(next(iter(self.objmasks.values())))
            out, kept = [dict(frame=fr, mask=None, centroid=[float(xs.mean()), float(ys.mean())])], [(fr, p)]
        self.grasps = kept
        return out

    def score(self, R, t, frames):
        set_camera(self.m, self.cid, R, t, self.Kd, self.w, self.h, self.a.principal_mode)
        oc, ious = [], []
        for fr in frames:
            self.pose(fr)
            self.rend.update_scene(self.d, camera=self.cid)
            seg = self.rend.render()
            sm = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], self.robot_geoms)
            rs = self.real_small[fr]
            if not sm.any():
                oc.append(200.0)
                ious.append(0.0)
                continue
            dt_sim = cv2.distanceTransform((~sm).astype(np.uint8), cv2.DIST_L2, 3)
            oc.append(float(self.dt_real[fr][sm].mean()) + float(dt_sim[rs].mean()))
            ious.append(float((sm & rs).sum() / max((sm | rs).sum(), 1)))
        # grasp points to their target centroids
        gp = []
        for (fr, p), tg in zip(self.grasps, self.tg):
            uvp = self.project(R, t, p)[0][0]
            if tg["mask"] is None:
                gp.append(float(np.linalg.norm(uvp - np.array(tg["centroid"]))))
                continue
            ui, vi = int(round(uvp[0])), int(round(uvp[1]))
            inside = 0 <= ui < self.W and 0 <= vi < self.H
            gp.append(float(self.mask_dt[tg["mask"]][vi, ui]) if inside else 200.0)
        # MoGe scaled by the grasp points, table rebuilt through (R, t)
        uv_g = [self.project(R, t, p) for _, p in self.grasps]
        ratios = []
        for (uvp, dp), tg in zip(uv_g, self.tg):
            if tg["mask"] is None:
                continue
            # MoGe read where the grasp point lands on its object (10-28), not the mask's median (centre)
            xu, yv = int(tg["centroid"][0]), int(tg["centroid"][1])
            win = self.per["depth"][max(yv - 2, 0):yv + 3, max(xu - 2, 0):xu + 3]
            win = win[np.isfinite(win) & (win > 0)]
            if len(win):
                ratios.append(dp[0] / float(np.median(win)))
        s = float(np.median(ratios)) if ratios else 1.0
        zz = s * self.tz
        pc = np.stack([(self.tu - self.K[0, 2]) / self.K[0, 0] * zz, (self.tv - self.K[1, 2]) / self.K[1, 1] * zz, zz], -1)
        pb = pc @ R.T + t
        A = np.c_[pb[:, 0], pb[:, 1], np.ones(len(pb))]
        c, *_ = np.linalg.lstsq(A, pb[:, 2], rcond=None)
        tilt = float(np.degrees(np.arctan(np.hypot(c[0], c[1]))))
        dz = abs(100 * (float(np.median(pb[:, 2])) - self.a.table_z)) if self.a.table_z is not None else 0.0
        parts = dict(outline=float(np.mean(oc)), grasp=float(np.mean(gp)), tilt=tilt, dz=dz)
        total = sum(WEIGHTS[k] * v for k, v in parts.items())
        return total, dict(parts, total=total, iou=float(np.mean(ious)), scale=s, table_median_z=float(np.median(pb[:, 2])))

    def cam_from(self, az, el, r, roll):
        C = self.P + r * np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)), np.cos(np.radians(el)) * np.sin(np.radians(az)), np.sin(np.radians(el))])
        R = look_rotation(C, self.P, self.uvP, self.K, roll)
        return R, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="a built scene of this episode (robot model, perception)")
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", default="22008760")
    ap.add_argument("--init", default=None, help="comma list of calib JSONs to start from (default: <scene>/calib.json, "
                    "<scene>/calib_outline_only.json if present); DROID's recorded pose is always tried too")
    ap.add_argument("--table_z", type=float, default=None, help="trusted wrist-stereo table height (m); without it only the level term")
    ap.add_argument("--n_frames", type=int, default=30)
    ap.add_argument("--span_deg", type=float, default=10.0)
    ap.add_argument("--span_r", type=float, default=0.15)
    ap.add_argument("--principal_mode", default="offset")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    S = Search(a)
    # starting poses: given calibs + DROID's recorded pose
    starts = {}
    inits = a.init.split(",") if a.init else [p for p in (os.path.join(a.scene, "calib.json"), os.path.join(a.scene, "calib_outline_only.json")) if os.path.exists(p)]
    for p in inits:
        c = json.load(open(p))
        starts[os.path.basename(p)] = (np.array(c["R_cam_to_base"]), np.array(c["t_cam_in_base"]))
    Rd, td = droid_pose(S.f[f"observation/camera_extrinsics/{S.cam}_left"][S.pinfo["frame"]])
    starts["droid_recorded"] = (Rd, td)
    start_rows = {}
    for name, (R, t) in starts.items():
        S.tg = S.targets(R, t)
        tot, parts = S.score(R, t, S.coarse_frames)
        start_rows[name] = parts
        print("start", name, json.dumps({k: round(v, 3) for k, v in parts.items()}), flush=True)
    best_start = min(start_rows, key=lambda k: start_rows[k]["total"])
    R0, t0_ = starts[best_start]
    S.tg = S.targets(R0, t0_)  # grasp targets fixed from the best start
    S.P = np.mean([p for _, p in S.grasps], 0)
    S.uvP = np.mean([tg["centroid"] for tg in S.tg], 0)
    print("grasps kept", [fr for fr, _ in S.grasps], "dropped", S.dropped, flush=True)
    az0, el0, r0 = params_of(R0, t0_, S.P)
    roll0 = roll_of(R0, t0_, S.P, S.uvP, S.K)
    print(f"start {best_start}: az {az0:.1f} el {el0:.1f} r {r0:.3f} roll {roll0:.1f}; grasp targets {S.tg}", flush=True)
    rows = []

    def evaluate(az, el, r, roll, frames):
        R, C = S.cam_from(az, el, r, roll)
        tot, parts = S.score(R, C, frames)
        rows.append(dict(az=az, el=el, r=r, roll=roll, n_frames=len(frames), **parts))
        return tot

    # coarse: az, el, r at the start's roll
    sd, sr = a.span_deg, a.span_r
    grid = list(itertools.product(np.linspace(az0 - sd, az0 + sd, 7), np.linspace(el0 - sd, el0 + sd, 7), np.linspace(r0 - sr, r0 + sr, 5)))
    for i, (az, el, r) in enumerate(grid):
        evaluate(az, el, r, roll0, S.coarse_frames)
        if i % 25 == 0:
            print(f"coarse {i}/{len(grid)} best {min(x['total'] for x in rows):.2f} ({time.time() - t0:.0f}s)", flush=True)
    b = min(rows, key=lambda x: x["total"])
    # fine: around the best, with roll, on all frames
    fine = list(itertools.product(np.linspace(b["az"] - sd / 3, b["az"] + sd / 3, 5), np.linspace(b["el"] - sd / 3, b["el"] + sd / 3, 5),
                                  np.linspace(b["r"] - sr / 3, b["r"] + sr / 3, 3), np.linspace(roll0 - 5, roll0 + 5, 5)))
    n_coarse = len(rows)
    for i, (az, el, r, rl) in enumerate(fine):
        evaluate(az, el, r, rl, S.frames)
        if i % 50 == 0:
            print(f"fine {i}/{len(fine)} ({time.time() - t0:.0f}s)", flush=True)
    fine_rows = rows[n_coarse:]
    best = min(fine_rows, key=lambda x: x["total"])
    R, C = S.cam_from(best["az"], best["el"], best["r"], best["roll"])
    _, start_full = S.score(R0, t0_, S.frames)
    dR = float(np.degrees(Rot.from_matrix(R @ R0.T).magnitude()))
    res = dict(scene=a.scene, episode=a.episode, status="exploratory, UNVERIFIED", weights=WEIGHTS, table_z=a.table_z,
               P=S.P.tolist(), grasps=[dict(frame=fr, point=p.tolist()) for fr, p in S.grasps], grasp_targets=S.tg, grasps_dropped=S.dropped,
               frames=S.frames, coarse_frames=S.coarse_frames, starts=start_rows, best_start=best_start,
               start=dict(az=az0, el=el0, r=r0, roll=roll0, **start_full), best=best,
               change=dict(position_cm=100 * float(np.linalg.norm(C - t0_)), position_delta_cm=(100 * (C - t0_)).tolist(),
                           rotation_deg=dR), seconds=time.time() - t0, rows=rows)
    json.dump(res, open(os.path.join(a.out, "camera_search.json"), "w"), indent=1, default=float)
    calib = dict(cam=S.cam, K=S.K.tolist(), R_cam_to_base=R.tolist(), t_cam_in_base=C.tolist(), principal_mode=a.principal_mode,
                 frames=S.frames, source="camera_search.py (res31 10-28)", search_best=best)
    json.dump(calib, open(os.path.join(a.out, "calib_search.json"), "w"), indent=1, default=float)
    # heatmaps over az x el at the best r and roll (fine rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = [x for x in fine_rows if abs(x["r"] - best["r"]) < 1e-9 and abs(x["roll"] - best["roll"]) < 1e-9]
    azs, els = sorted({x["az"] for x in sel}), sorted({x["el"] for x in sel})
    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    for axi, key in zip(ax, ("total", "outline", "grasp", "tilt", "dz")):
        Z = np.full((len(els), len(azs)), np.nan)
        for x in sel:
            Z[els.index(x["el"]), azs.index(x["az"])] = x[key]
        im = axi.imshow(Z, origin="lower", extent=[azs[0], azs[-1], els[0], els[-1]], aspect="auto", cmap="viridis")
        axi.plot(best["az"], best["el"], "r+", ms=12)
        axi.plot(az0, el0, "wx", ms=10)
        axi.set_title(key)
        axi.set_xlabel("azimuth (deg)")
        fig.colorbar(im, ax=axi, fraction=0.046)
    ax[0].set_ylabel("elevation (deg)")
    fig.suptitle(f"{os.path.basename(os.path.normpath(a.out))}: score over the sphere (r {best['r']:.3f} m, roll {best['roll']:.1f}°); + best, x start")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "heatmaps.png"), dpi=100)
    print(json.dumps(dict(start={k: round(v, 3) for k, v in start_full.items()}, best={k: round(v, 3) if isinstance(v, float) else v for k, v in best.items()},
                          change=res["change"]), default=float), flush=True)


if __name__ == "__main__":
    main()
