"""Scene-geometry probes (SCENE_V1_MECHANISMS §20; user 10-28: "Plan experiments to isolate the cause and fix", then "yes").

Measurement only. Three probes on a built scene (calib.json, perception.npz/json, scene_fit.json, scene.mjb):
  e6  tilt of the table fitted from the exterior points, and its height by depth band   (H6 tilt vs H1 depth drift)
  e2  front-surface bias of the grasp anchors: what the exterior camera sees along the ray through each real
      fingertip at its grasp, against the fingertip's own depth                           (H2, the user's hypothesis)
  e1  the robot as a depth ruler: on robot pixels over many frames, the true depth (FK + calibrated camera) vs
      exterior stereo and MoGe, by depth band                                           (H1 MoGe drift, H7 stereo)
Writes <out>/geom_probe.json and <out>/geom_probe_e1.png.

    runs/real2sim/v1_dev/py_triton_first.sh experiments/real2sim/geom_probe.py --scene <scene> --episode <ep> \
        --out runs/real2sim/v1_dev/geom_probe/<name> [--probes e6,e2,e1] [--e1_frames 16]
Exploratory (res31, 10-28), UNVERIFIED.
"""
import argparse, json, os, sys
import numpy as np
import cv2
import h5py
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_cam import set_camera, motion_masks  # noqa: E402
from episode_info import grasp_events  # noqa: E402
from stereo_depth import stereo_z, BASELINE_M  # noqa: E402
from scene_roles import table_label  # noqa: E402

BANDS = [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.3, 2.0]


def band_stats(depth_true, value, bands=BANDS):
    """Median and IQR of value per true-depth band, with pixel counts."""
    out = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        k = (depth_true >= lo) & (depth_true < hi) & np.isfinite(value)
        if k.sum() >= 50:
            v = value[k]
            out.append(dict(band_m=[lo, hi], n=int(k.sum()), median=float(np.median(v)),
                            iqr=[float(np.percentile(v, 25)), float(np.percentile(v, 75))]))
    return out


class Scene:
    def __init__(self, a):
        self.a = a
        self.calib = json.load(open(os.path.join(a.scene, "calib.json")))
        self.R, self.t, self.K = (np.array(self.calib[k]) for k in ("R_cam_to_base", "t_cam_in_base", "K"))
        self.cam = self.calib["cam"]
        self.pmode = self.calib.get("principal_mode", "offset")
        self.per = dict(np.load(os.path.join(a.scene, "perception.npz")))
        self.pinfo = json.load(open(os.path.join(a.scene, "perception.json")))
        self.fit = json.load(open(os.path.join(a.scene, "scene_fit.json")))
        self.s = float(self.fit["scale"])
        self.H, self.W = self.per["image"].shape[:2]
        self.m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
        self.d = mujoco.MjData(self.m)
        f = h5py.File(os.path.join(a.episode, "trajectory.h5"))
        self.J = f["observation/robot_state/joint_positions"][:]
        self.G = f["observation/robot_state/gripper_position"][:]
        self.arm = [self.m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
        self.robot_geoms = np.array([g for g in range(self.m.ngeom) if self.m.body(self.m.geom_bodyid[g]).name.startswith(("fr3", "g_"))])
        self.cid = self.m.camera("ext").id
        set_camera(self.m, self.cid, self.R, self.t, self.K, self.W, self.H, self.pmode)
        self.rend = mujoco.Renderer(self.m, self.H, self.W)

    def cam_depth(self, p):
        pc = (np.atleast_2d(p) - self.t) @ self.R
        uv = pc[:, :2] / pc[:, 2:3] * [self.K[0, 0], self.K[1, 1]] + [self.K[0, 2], self.K[1, 2]]
        return uv, pc[:, 2]

    def pose(self, frame):
        self.d.qpos[:] = self.m.qpos0
        self.d.qpos[self.arm] = self.J[frame]
        mujoco.mj_forward(self.m, self.d)

    def render(self, frame):
        """(robot mask, depth) of the sim at the recorded joints of `frame`, from the calibrated exterior camera."""
        self.pose(frame)
        self.rend.enable_segmentation_rendering()
        self.rend.update_scene(self.d, camera=self.cid)
        seg = self.rend.render()
        self.rend.disable_segmentation_rendering()
        self.rend.enable_depth_rendering()
        self.rend.update_scene(self.d, camera=self.cid)
        dep = self.rend.render()  # distance along the optical axis (MuJoCo z-depth)
        self.rend.disable_depth_rendering()
        robot = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], self.robot_geoms)
        return robot, dep

    def moge_z(self):
        return self.per["depth"]

    def table_points(self):
        z = self.moge_z()
        u, v = np.meshgrid(np.arange(self.W), np.arange(self.H))
        mk = cv2.erode(self.per[f"mask/{table_label(self.pinfo)}"].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & self.per["valid"] & (z > 0)
        zz = self.s * z[mk]
        pc = np.stack([(u[mk] - self.K[0, 2]) / self.K[0, 0] * zz, (v[mk] - self.K[1, 2]) / self.K[1, 1] * zz, zz], -1)
        return pc @ self.R.T + self.t, zz


def probe_e6(S):
    pb, zcam = S.table_points()
    A = np.c_[pb[:, 0], pb[:, 1], np.ones(len(pb))]
    coef, *_ = np.linalg.lstsq(A, pb[:, 2], rcond=None)
    tilt = float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))
    used = S.fit.get("table_z_override", {}).get("used", S.fit["table"]["z"])
    st = os.path.join(S.a.scene, "stereo_table_check.json")
    wrist = [r["z_median"] for r in json.load(open(st))["results"] if r["cam"] == "wrist"] if os.path.exists(st) else []
    return dict(n=int(len(pb)), plane=dict(dz_dx=float(coef[0]), dz_dy=float(coef[1]), z0=float(coef[2]), tilt_deg=tilt),
                fitted_median_z=float(np.median(pb[:, 2])), table_z_in_scene=used, wrist_stereo_z=wrist,
                z_by_camera_depth=band_stats(zcam, pb[:, 2]))


def probe_e2(S):
    rows = []
    zs = stereo_z(S.a.episode, S.cam, S.pinfo["frame"], S.K[0, 0], BASELINE_M["ext"])
    robot0, dep0 = S.render(S.pinfo["frame"])  # what the sim robot covers in the perception frame
    masks = {k[5:]: S.per[k] for k in S.per if k.startswith("mask/")}
    objs = {lab: o for lab, o in S.fit["objects"].items()}
    for i, (kc, ko) in enumerate(grasp_events(S.G)):
        S.pose(kc)
        p = S.d.site_xpos[S.m.site("g_pinch").id].copy()
        uv, dtrue = S.cam_depth(p)
        u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))
        inside = 0 <= u < S.W and 0 <= v < S.H
        hit = [lab for lab, mk in masks.items() if inside and mk[v, u]] if inside else []
        r = dict(grasp=i, close_frame=int(kc), fingertip=p.tolist(), px=[u, v], inside_image=inside, cam_depth=float(dtrue[0]),
                 masks_at_px=hit, robot_at_px_in_perception_frame=bool(inside and robot0[v, u]))
        if inside:
            w = 4
            win = lambda img: img[max(v - w, 0):v + w + 1, max(u - w, 0):u + w + 1]
            mz = win(S.moge_z()) * S.s
            sz = win(zs)
            r.update(moge_scaled_at_px=float(np.nanmedian(mz)), stereo_at_px=float(np.nanmedian(sz)) if np.isfinite(sz).any() else None)
            obj = [lab for lab in hit if lab in objs]
            if obj:
                o = objs[obj[0]]
                half = o.get("half") or [o.get("rim_radius", 0.0)] * 3
                mk = cv2.erode(masks[obj[0]].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & S.per["valid"]
                mz_mask = S.moge_z()[mk] * S.s
                # front surface of the object along this ray: about its half-depth closer than its centre
                r.update(object=obj[0], object_half_m=[float(x) for x in half],
                         moge_scaled_mask_median=float(np.median(mz_mask)), moge_scaled_mask_p10=float(np.percentile(mz_mask, 10)),
                         predicted_front_depth=float(dtrue[0] - min(half[:2])),
                         ratio_mask_median_to_fingertip=float(np.median(mz_mask) / dtrue[0]),
                         ratio_predicted_front_to_fingertip=float((dtrue[0] - min(half[:2])) / dtrue[0]))
        rows.append(r)
    return dict(scale_in_scene=S.s, depth_mode=S.fit.get("depth"), grasps=rows)


def probe_e1(S, n_frames, out):
    from perceive import read_frame, geometry
    vid = os.path.join(S.a.episode, f"{S.cam}.mp4")
    cap = cv2.VideoCapture(vid)
    n = min(len(S.J), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))  # the video can be shorter than the trajectory
    cap.release()
    frames = sorted({int(x) for x in np.linspace(0, n - 2, n_frames)} | {S.pinfo["frame"]})
    real_masks, _ = motion_masks(vid, frames)
    per_frame, all_true, all_stereo, all_moge_rel = [], [], [], []
    for fr in frames:
        robot, dep = S.render(fr)
        real = real_masks[str(fr)]
        sel = cv2.erode((robot & real).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & (dep > 0)
        if sel.sum() < 500:
            per_frame.append(dict(frame=fr, robot_px=int(sel.sum()), skipped="robot barely visible"))
            continue
        zs = stereo_z(S.a.episode, S.cam, fr, S.K[0, 0], BASELINE_M["ext"])[sel]
        geo = geometry(read_frame(vid, fr), S.pinfo.get("fov_x_deg"))
        zm = geo["depth"][sel]
        dt = dep[sel]
        ok = np.isfinite(zm) & (zm > 0)
        k_frame = float(np.median(dt[ok] / zm[ok]))  # MoGe's own scale in this image
        row = dict(frame=fr, robot_px=int(sel.sum()), true_depth_range=[float(np.percentile(dt, 5)), float(np.percentile(dt, 95))],
                   stereo_ratio_median=float(np.nanmedian(zs / dt)), moge_scale_this_frame=k_frame)
        if fr == S.pinfo["frame"]:
            row["moge_ratio_with_scene_scale"] = float(np.median(S.s * zm[ok] / dt[ok]))
        per_frame.append(row)
        all_true.append(dt)
        all_stereo.append(zs / dt)
        rel = np.full(dt.shape, np.nan)
        rel[ok] = k_frame * zm[ok] / dt[ok]  # 1 everywhere if MoGe's error is one scale per image
        all_moge_rel.append(rel)
        print(json.dumps(row), flush=True)
    res = dict(frames=frames, per_frame=per_frame)
    if all_true:
        dt, rs, rm = np.concatenate(all_true), np.concatenate(all_stereo), np.concatenate(all_moge_rel)
        res["stereo_ratio_by_depth"] = band_stats(dt, rs)
        res["moge_relative_ratio_by_depth"] = band_stats(dt, rm)
        ok = np.isfinite(rs) & (rs > 0)
        # stereo model: 1/z_stereo = a/z_true + b. a != 1 -> scale (baseline/focal); b != 0 -> disparity offset
        A = np.c_[1 / dt[ok], np.ones(ok.sum())]
        (ca, cb), *_ = np.linalg.lstsq(A, 1 / (rs[ok] * dt[ok]), rcond=None)
        res["stereo_fit_inverse_depth"] = dict(a=float(ca), b=float(cb), note="1/z_stereo = a/z_true + b")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for axi, key, title in ((ax[0], "stereo_ratio_by_depth", "exterior stereo / true depth"),
                                (ax[1], "moge_relative_ratio_by_depth", "MoGe (own scale per image) / true depth")):
            b = res[key]
            x = [np.mean(r_["band_m"]) for r_ in b]
            axi.errorbar(x, [r_["median"] for r_ in b], yerr=np.array([[r_["median"] - r_["iqr"][0], r_["iqr"][1] - r_["median"]] for r_ in b]).T, fmt="o-")
            axi.axhline(1, color="grey", lw=0.8)
            axi.set_xlabel("true depth on robot pixels (m)")
            axi.set_title(title)
        fig.suptitle(f"{os.path.basename(os.path.normpath(S.a.scene))}: robot as a depth ruler ({len(all_true)} frames)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "geom_probe_e1.png"), dpi=110)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probes", default="e6,e2,e1")
    ap.add_argument("--e1_frames", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = Scene(a)
    path = os.path.join(a.out, "geom_probe.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    res.update(scene=a.scene, episode=a.episode, status="exploratory, UNVERIFIED")
    for p in a.probes.split(","):
        res[p] = dict(e6=lambda: probe_e6(S), e2=lambda: probe_e2(S), e1=lambda: probe_e1(S, a.e1_frames, a.out))[p]()
        json.dump(res, open(path, "w"), indent=1, default=float)
        print(p, "done", flush=True)


if __name__ == "__main__":
    main()
