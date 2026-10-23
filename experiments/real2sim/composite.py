"""Real-to-sim visuals: composite a sim run onto the real camera's background (SIMPLER-style green screen).

  1. clean plate: per-pixel temporal median of the real video (the moving robot drops out), with the
     static real objects (bowl, blocks; perception masks, dilated) inpainted away so a moved sim object
     doesn't leave its real twin painted in the background
  2. per frame: render the sim from the calibrated camera; robot + object pixels (segmentation) come from
     the sim render, everything else (wall, real wooden table) from the plate
  3. shadows: render the table twice, with and without robot/objects (hidden via geom group; checked that
     hidden geoms cast no shadow); the brightness ratio on table pixels darkens the plate
  4. colours: sim blocks and bowl get the real median colour of their perception masks

Writes <out>/composite_<run>.mp4 (real | composite).

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/composite.py \
        --episode data/real2sim/droid/ep_orange_block --cam 22008760 --out runs/real2sim/ep_orange_block --run replay
"""
import argparse, os, sys
import numpy as np
import cv2
import imageio.v2 as imageio
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay import read_video  # noqa: E402

FPS = 15
HIDE_GROUP = 5


def scene_objects(out):
    """body name -> perception mask label, from build_scene's scene_fit.json (older fits: label-derived body names)."""
    import json
    objs = json.load(open(os.path.join(out, "scene_fit.json")))["objects"]
    return {o.get("body", lab.replace("#", "_").replace(" ", "_")): o.get("mask", lab) for lab, o in objs.items()}


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAMA_WEIGHTS = os.path.join(REPO, "data/real2sim/tools/lama/big-lama.pt")


def lama_inpaint(img, hole):
    """LaMa (advimman/lama, Apache-2.0) via the TorchScript export from simple-lama-inpainting (Apache-2.0),
    loaded directly so its numpy<2/pillow<10/opencv<5 pins don't touch this venv. Same preprocessing as that
    wrapper: float [0,1] CHW, binary mask, both padded symmetrically to a multiple of 8."""
    import torch
    model = torch.jit.load(LAMA_WEIGHTS, map_location="cpu").eval()
    H, W = hole.shape
    ph, pw = (-H) % 8, (-W) % 8
    x = np.pad(img.astype(np.float32).transpose(2, 0, 1) / 255, ((0, 0), (0, ph), (0, pw)), mode="symmetric")
    mk = np.pad((hole > 0).astype(np.float32)[None], ((0, 0), (0, ph), (0, pw)), mode="symmetric")
    with torch.inference_mode():
        y = model(torch.from_numpy(x)[None], torch.from_numpy(mk)[None])[0]
    return np.clip(y.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)[:H, :W]


def plate_holes(plate, objects):
    """Masks of the objects as they appear in the median plate itself (mid-episode positions, things in the bowl),
    found with the same detector + SAM using the scene's object labels as prompts."""
    from perceive import detect_multi, segment
    prompts = sorted({lab.split("#")[0] for lab in objects.values()})
    dets = detect_multi(plate, prompts)
    hole = np.zeros(plate.shape[:2], np.uint8)
    for mk in segment(plate, dets).values():
        hole |= mk.astype(np.uint8)
    return hole, sorted(dets)


def clean_plate(frames, per, objects, dilate=15, method="telea", holes="frame0"):
    plate = np.median(np.stack(frames[::max(1, len(frames) // 150)]), axis=0).astype(np.uint8)  # <=~150 frames: memory
    hole = np.zeros(plate.shape[:2], np.uint8)
    for name in objects.values():
        if f"mask/{name}" in per:
            hole |= per[f"mask/{name}"].astype(np.uint8)
    if holes == "frame0+plate":
        h2, found = plate_holes(plate, objects)
        print("objects found in the plate:", found)
        hole |= h2
    hole = cv2.dilate(hole, np.ones((dilate, dilate), np.uint8))
    if method == "lama":
        return lama_inpaint(plate, hole)
    return cv2.inpaint(plate, hole, 7, cv2.INPAINT_TELEA)


def load_plate(out, inpaint="telea", holes="frame0", real=None, per=None):
    """Cleaned plate (float32 RGB, full camera resolution), from <out>/plate_cache_<inpaint>_<holes>.png when it
    exists, else computed from the real frames + perception masks and cached."""
    cache = os.path.join(out, f"plate_cache_{inpaint}_{holes.replace('+', '_')}.png")
    if os.path.exists(cache):  # the cleaned plate depends only on the scene, not on the run
        return cv2.cvtColor(cv2.imread(cache), cv2.COLOR_BGR2RGB).astype(np.float32)
    if real is None or per is None:
        raise FileNotFoundError(f"{cache} missing; run composite.py on this scene once to build it")
    plate = clean_plate(real, per, scene_objects(out), method=inpaint, holes=holes).astype(np.float32)
    cv2.imwrite(cache, cv2.cvtColor(plate.astype(np.uint8), cv2.COLOR_RGB2BGR))
    return plate


class Compositor:
    """Per-frame composite for camera 'ext': sim robot + object pixels over the cleaned real plate, the plate darkened
    by the sim's shadow ratio on table pixels. Sets the real median colours on the sim objects and the light from
    light.json (if present) on the model `m` -- render-only fields; physics is unaffected.
    size=(H, W) renders and composites at that size (the plate is downscaled; the camera FOV is resolution-free)."""

    def __init__(self, m, out, plate, per=None, shadows=True, size=None, verbose=True):
        import json
        self.m, self.shadows = m, shadows
        per = per if per is not None else dict(np.load(os.path.join(out, "perception.npz")))
        # real colours on the sim objects
        for body, name in scene_objects(out).items():
            if f"mask/{name}" not in per:
                continue
            rgb = np.median(per["image"][cv2.erode(per[f"mask/{name}"].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0], axis=0) / 255.0
            for g in range(m.ngeom):
                if m.body(m.geom_bodyid[g]).name == body and m.geom_rgba[g, 3] > 0:
                    m.geom_rgba[g, :3] = rgb
        # per-body render colours set by hand (refit_block.py --render_rgb, 10-23), applied over the real medians
        ov = os.path.join(out, "render_colour_overrides.json")
        for body, rgb in (json.load(open(ov)).items() if os.path.exists(ov) else ()):
            for g in range(m.ngeom):
                if m.body(m.geom_bodyid[g]).name == body and m.geom_rgba[g, 3] > 0:
                    m.geom_rgba[g, :3] = rgb
        # light fitted to the real robot's shadow (calibrate_light.py), if available
        self.light = None
        if os.path.exists(os.path.join(out, "light.json")):
            self.light = json.load(open(os.path.join(out, "light.json")))
            m.light_type[0] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            m.light_dir[0] = self.light["dir"]
            m.light_castshadow[0] = 1
            if verbose:
                print("light from light.json: az", self.light["azimuth_deg"], "el", self.light["elevation_deg"],
                      "real shadow ratio", round(self.light["real_shadow_ratio"], 3))
        self.H, self.W = size if size else plate.shape[:2]
        self.plate = plate if plate.shape[:2] == (self.H, self.W) else cv2.resize(plate, (self.W, self.H), interpolation=cv2.INTER_AREA)
        self.table = m.geom("table").id
        self.fg_geoms = np.array([g for g in range(m.ngeom) if g != self.table and m.geom_bodyid[g] != 0])  # robot + objects
        self.rend = mujoco.Renderer(m, self.H, self.W)
        self.opt_all, self.opt_hide = mujoco.MjvOption(), mujoco.MjvOption()
        self.opt_hide.geomgroup[HIDE_GROUP] = 0
        self.group0 = m.geom_group.copy()

    def __call__(self, d):
        """float32 RGB composite of the current state of `d` (kinematics must be up to date)."""
        m, rend = self.m, self.rend
        rend.update_scene(d, camera="ext", scene_option=self.opt_all)
        sim = rend.render().astype(np.float32)
        rend.enable_segmentation_rendering()
        rend.update_scene(d, camera="ext", scene_option=self.opt_all)
        seg = rend.render()
        rend.disable_segmentation_rendering()
        fg = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], self.fg_geoms)
        out = self.plate.copy()
        if self.shadows:
            m.geom_group[self.fg_geoms] = HIDE_GROUP
            rend.update_scene(d, camera="ext", scene_option=self.opt_hide)
            lit = rend.render().astype(np.float32)
            m.geom_group[:] = self.group0
            on_table = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 0] == self.table)
            ratio = np.ones((self.H, self.W), np.float32)
            r = sim[on_table].mean(-1) / np.maximum(lit[on_table].mean(-1), 1.0)
            if self.light:  # sim says where the shadow is; the real video says how dark (measured median ratio)
                inside = np.clip((1.0 - r) / 0.1, 0.0, 1.0)  # same 0.9 threshold as calibrate_light
                ratio[on_table] = 1.0 - inside * (1.0 - self.light["real_shadow_ratio"])
            else:
                ratio[on_table] = np.clip(r, 0.35, 1.0)
            ratio = cv2.GaussianBlur(ratio, (0, 0), 3)
            out *= ratio[..., None]
        out[fg] = sim[fg]
        return out

    def close(self):
        self.rend.close()


def cam_K(m, cid, w, h):
    """OpenCV K of MuJoCo camera `cid` rendered at w x h (inverse of calibrate_cam.set_camera, 'offset' mode)."""
    W0, H0 = m.cam_sensorsize[cid]
    fx, fy, px, py = m.cam_intrinsic[cid]
    s = w / W0
    return np.array([[fx * s, 0, (px + W0 / 2) * s], [0, fy * s, (H0 / 2 - py) * s], [0, 0, 1.0]])


def cam_pose(d, cid):
    """OpenCV (R cam->world, t) of camera `cid` in the current state."""
    return d.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1]), d.cam_xpos[cid].copy()


def redacted_discs(fit, margin=0.0):
    """Footprints (x, y, radius) of objects left out of the scene (asset-pack mode: fit['redacted'], 10-20). The real
    object is still in the real frames, so every real-pixel texture source must mask it (else it reappears as a flat
    ghost on the table)."""
    out = []
    for label, r in fit.get("redacted", {}).items():
        rad = r.get("rim_radius") or r.get("width_m", 0.06) / 2
        out.append((float(r["center"][0]), float(r["center"][1]), float(rad) + margin, label))
    return out


def wrist_plate(m, out, episode, ext_comp, serial, dilate=25):
    """Second texture source for the wrist view: the real wrist frame at the start of <out>/replay_states.npz
    (sim posed at the recorded joints), robot + object pixels (sim wrist segmentation, dilated) inpainted with LaMa.
    Cached as <out>/plate_cache_wrist_frame0.png. Returns dict(img float32 RGB, K, R, t) of that wrist view."""
    import json
    st = np.load(os.path.join(out, "replay_states.npz"))
    d = mujoco.MjData(m)
    d.qpos[:] = st["qpos"][0]
    mujoco.mj_forward(m, d)
    cid = m.camera("wrist").id
    W, H = (int(v) for v in m.cam_sensorsize[cid])
    R, t = cam_pose(d, cid)
    cache = os.path.join(out, "plate_cache_wrist_frame0.png")
    if os.path.exists(cache):
        img = cv2.cvtColor(cv2.imread(cache), cv2.COLOR_BGR2RGB)
    else:
        cap = cv2.VideoCapture(os.path.join(episode, f"{serial}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(st["real_frames"][0]))
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read wrist frame {int(st['real_frames'][0])} of {episode}/{serial}.mp4")
        real = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if real.shape[:2] != (H, W):
            raise ValueError(f"wrist video {real.shape[:2]} != wrist camera sensor size {(H, W)}")
        rend = mujoco.Renderer(m, H, W)
        rend.enable_segmentation_rendering()
        rend.update_scene(d, camera="wrist", scene_option=ext_comp.opt_all)
        seg = rend.render()
        rend.close()
        fg = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], ext_comp.fg_geoms)
        # redacted objects (not in the sim) are still in the real frame: project a cylinder over each footprint
        fitj = json.load(open(os.path.join(out, "scene_fit.json")))
        zt = fitj["table"]["z"]
        Kw0 = cam_K(m, cid, W, H)
        for cx, cy, rad, _ in redacted_discs(fitj, 0.02):
            ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            ring = np.array([[cx + rad * np.cos(t_), cy + rad * np.sin(t_), z_] for t_ in ang for z_ in (zt, zt + 2 * rad)])
            pc = (ring - t) @ R
            ok = pc[:, 2] > 0.02
            if ok.sum() >= 3:
                uv = (pc[ok, :2] / pc[ok, 2:3] * [Kw0[0, 0], Kw0[1, 1]] + [Kw0[0, 2], Kw0[1, 2]]).astype(np.int32)
                fg_u8 = fg.astype(np.uint8)
                cv2.fillConvexPoly(fg_u8, cv2.convexHull(uv), 1)
                fg = fg_u8 > 0
        hole = cv2.dilate(fg.astype(np.uint8), np.ones((dilate, dilate), np.uint8))
        img = lama_inpaint(real, hole)
        # colour gain exterior -> wrist camera (the policy sees the wrist camera's exposure/white balance): per-channel
        # median ratio over background pixels seen by both cameras and outside the inpainted hole
        rend = mujoco.Renderer(m, H, W)
        rend.enable_depth_rendering()
        rend.update_scene(d, camera="wrist", scene_option=ext_comp.opt_all)
        depth = rend.render()
        rend.close()
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        Kw = cam_K(m, cid, W, H)
        pw = np.stack([(u - Kw[0, 2]) / Kw[0, 0], (v - Kw[1, 2]) / Kw[1, 1], np.ones_like(u)], -1) * depth[..., None] @ R.T + t
        Re, te = cam_pose(d, m.camera("ext").id)
        Ke = cam_K(m, m.camera("ext").id, ext_comp.plate.shape[1], ext_comp.plate.shape[0])
        pc = (pw - te) @ Re
        z = np.maximum(pc[..., 2], 1e-6)
        mx = (pc[..., 0] / z * Ke[0, 0] + Ke[0, 2] - 0.5).astype(np.float32)
        my = (pc[..., 1] / z * Ke[1, 1] + Ke[1, 2] - 0.5).astype(np.float32)
        both = (pc[..., 2] > 0) & (mx >= 0) & (mx <= ext_comp.plate.shape[1] - 1) & (my >= 0) & (my <= ext_comp.plate.shape[0] - 1) \
            & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (hole == 0)
        ext_tex = cv2.remap(ext_comp.plate.astype(np.float32), mx, my, cv2.INTER_LINEAR)
        gain = (np.median(real[both].astype(np.float32), 0) / np.maximum(np.median(ext_tex[both], 0), 1.0)).tolist() if both.sum() > 1000 else [1.0, 1.0, 1.0]
        cv2.imwrite(cache, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        json.dump(dict(real_frame=int(st["real_frames"][0]), dilate=dilate, hole_frac=float(hole.mean()),
                       ext_to_wrist_gain=gain, gain_px=int(both.sum())), open(cache.replace(".png", ".json"), "w"))
    meta = json.load(open(cache.replace(".png", ".json")))
    return dict(img=img.astype(np.float32), K=cam_K(m, cid, W, H), R=R, t=t, gain=np.array(meta["ext_to_wrist_gain"], np.float32))


def wrist_table_texture(m, out, episode, serial, frames, res=0.0015, dilate=35, obj_dilate=61, near_frac=0.5, obj_margin=0.015,
                        cache=None):
    """Top-down texture of the table top mosaicked from the REAL wrist video (frames before any object is touched),
    cached as <out>/table_tex_wrist.npz. For each frame the sim is posed at the recorded joints + gripper; every texel
    (x, y, table z) is projected into that wrist frame and sampled if the sim says the camera sees the table there
    (segmentation, with robot/object pixels dilated by `dilate` px so edges and the objects' real twins are skipped).
    Each sample is weighted by closeness and head-on-ness (cos / z^2). Per texel: the median of the samples within
    `near_frac` of its best weight (median: transient things such as the gripper's moving shadow drop out). One
    exposure pass: each frame is scaled to the first mosaic by its median ratio over shared texels (the ZED Mini
    auto-exposes). Unobserved texels are inpainted from the mosaic itself (LaMa).
    Texels within `obj_margin` of an object's footprint (bounding radius at its start pose) are left out: the real
    objects sit there in every frame and their sides stick out beyond the sim masks (10-19: ghosts of the real blocks
    next to the sim ones); per frame, robot pixels are dilated by `dilate` px (the sim gripper's outline differs from the
    real one) and object pixels by `obj_dilate` px (a real block's top face, seen at an angle, lands behind its footprint)."""
    import json
    cache = cache or os.path.join(out, "table_tex_wrist.npz")  # cache=: e.g. a variant camera model (G8)
    if os.path.exists(cache):
        z = np.load(cache)
        return dict(tex=z["tex"].astype(np.float32), valid=z["valid"].astype(np.uint8), x0=float(z["x0"]), y0=float(z["y0"]),
                    res=float(z["res"]), z=float(z["z"]))
    import h5py
    fit = json.load(open(os.path.join(out, "scene_fit.json")))
    zt = fit["table"]["z"]
    (x0, y0), (x1, y1) = fit["table"]["xy_lo"], fit["table"]["xy_hi"]
    nx, ny = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    gx, gy = np.meshgrid(x0 + (np.arange(nx) + 0.5) * res, y0 + (np.arange(ny) + 0.5) * res)
    P = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, zt)], 1)
    d0 = mujoco.MjData(m)
    mujoco.mj_forward(m, d0)
    keep = np.ones(len(P), bool)
    obj_geoms = []
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            b = m.jnt_bodyid[j]
            gs = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == b]
            obj_geoms += gs
            # footprint radius on the table: box half-diagonal in xy (rbound is 3D and ~2x too big for blocks), else rbound
            r = max(np.linalg.norm(d0.geom_xpos[g][:2] - d0.xpos[b][:2])
                    + (np.linalg.norm(m.geom_size[g][:2]) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else m.geom_rbound[g]) for g in gs)
            keep &= np.linalg.norm(P[:, :2] - d0.xpos[b][:2], axis=1) > r + obj_margin
    for cx, cy, rad, _ in redacted_discs(fit, obj_margin + 0.03):  # redacted objects: footprint + extra (height unknown)
        keep &= np.hypot(P[:, 0] - cx, P[:, 1] - cy) > rad
    h = h5py.File(os.path.join(episode, "trajectory.h5"))
    J, G = h["observation/robot_state/joint_positions"][:], h["observation/robot_state/gripper_position"][:]
    arm = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    drv_max = m.jnt_range[m.joint("g_right_driver_joint").id][1]
    cid = m.camera("wrist").id
    W, H = (int(v) for v in m.cam_sensorsize[cid])
    K = cam_K(m, cid, W, H)
    tbl = m.geom("table").id
    d = mujoco.MjData(m)
    rend = mujoco.Renderer(m, H, W)
    rend.enable_segmentation_rendering()
    cap = cv2.VideoCapture(os.path.join(episode, f"{serial}.mp4"))
    cols, wts = [], []
    for k in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, bgr = cap.read()
        if not ok:
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        d.qpos[:] = m.qpos0
        d.qpos[arm] = J[k]
        for jn in ("g_right_driver_joint", "g_left_driver_joint"):
            d.qpos[m.joint(jn).qposadr[0]] = G[k] * drv_max
        mujoco.mj_forward(m, d)
        rend.update_scene(d, camera=cid)
        seg = rend.render()
        isg = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
        table_px = isg & (seg[..., 0] == tbl)
        objpx = isg & np.isin(seg[..., 0], obj_geoms)
        # robot edges (sim and real gripper outlines differ) and a wide band around objects: seen at an angle, a real
        # block's top face lands on table texels behind its footprint (10-19: ghost of the real block)
        table_px &= ~(cv2.dilate((~table_px & ~objpx).astype(np.uint8), np.ones((dilate, dilate), np.uint8)) > 0)
        table_px &= ~(cv2.dilate(objpx.astype(np.uint8), np.ones((obj_dilate, obj_dilate), np.uint8)) > 0)
        R, t = cam_pose(d, cid)
        pc = (P - t) @ R
        zc = pc[:, 2]
        front = zc > 0.03
        u = np.full(len(P), -1, int)
        v = np.full(len(P), -1, int)
        u[front] = np.floor(pc[front, 0] / zc[front] * K[0, 0] + K[0, 2]).astype(int)
        v[front] = np.floor(pc[front, 1] / zc[front] * K[1, 1] + K[1, 2]).astype(int)
        inimg = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        seen = np.zeros(len(P), bool)
        seen[inimg] = table_px[v[inimg], u[inimg]]
        seen &= keep
        col = np.full((len(P), 3), np.nan, np.float32)
        col[seen] = img[v[seen], u[seen]]
        ray = t - P
        cosang = ray[:, 2] / np.linalg.norm(ray, axis=1)
        w = np.where(seen, cosang / np.maximum(zc, 1e-3) ** 2, 0.0).astype(np.float32)
        cols.append(col)
        wts.append(w)
    rend.close()
    C, Wt = np.stack(cols), np.stack(wts)  # (frames, texels, 3), (frames, texels)

    def mosaic(C):
        near = Wt >= near_frac * Wt.max(0, keepdims=True)
        Cn = np.where((near & (Wt > 0))[..., None], C, np.nan)
        return np.nanmedian(Cn, axis=0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN texels (never seen) stay NaN
        tex0 = mosaic(C)
        gains = []
        for f in range(len(C)):
            s = (Wt[f] > 0) & np.isfinite(tex0).all(1)
            gains.append(np.nanmedian(tex0[s] / np.maximum(C[f][s], 1.0), axis=0) if s.sum() > 500 else np.ones(3, np.float32))
        tex = mosaic(C * np.array(gains, np.float32)[:, None, :])
    observed = np.isfinite(tex).all(1).reshape(ny, nx)
    tex = np.nan_to_num(tex).clip(0, 255).reshape(ny, nx, 3).astype(np.uint8)
    # fill the unobserved texels (under/around objects, never-seen spots) from the mosaic itself with LaMa, so the
    # filled wood matches its surroundings (10-19: falling back to the exterior plate left visible colour patches)
    # object zones are widened by ~2 cm first: a few edge texels there carry the real block's colour and LaMa grows
    # them into a ghost (10-19, lime blob next to the yellow block)
    zone = cv2.dilate((~keep).reshape(ny, nx).astype(np.uint8), np.ones((2 * int(0.02 / res) + 1,) * 2, np.uint8)) > 0
    observed &= ~zone
    tex = lama_inpaint(tex, (~observed).astype(np.uint8))
    valid = np.ones_like(observed)
    np.savez_compressed(cache, tex=tex, valid=valid.astype(np.uint8), observed=observed.astype(np.uint8), x0=x0, y0=y0, res=res, z=zt,
                        frames=np.array(frames), gains=np.array(gains), dilate=dilate, obj_dilate=obj_dilate, obj_margin=obj_margin)
    cv2.imwrite(cache.replace(".npz", ".png"), cv2.cvtColor(tex, cv2.COLOR_RGB2BGR))
    return dict(tex=tex.astype(np.float32), valid=valid.astype(np.uint8), x0=x0, y0=y0, res=res, z=zt, observed_frac=float(observed.mean()))


class WristCompositor:
    """Per-frame wrist view with real texture on the static background (projective texturing).
    Robot + object pixels come from the sim render (real colours from Compositor). Every other pixel (table, floor)
    is back-projected with the sim wrist depth to a world point, projected into the calibrated exterior camera and
    coloured from the cleaned exterior plate; points outside the exterior image are coloured from `plate2` (wrist_plate:
    the cleaned real wrist frame 0, if given) and otherwise keep the sim colour. The table is
    darkened by the sim shadow ratio as in Compositor. Needs a Compositor on the same model first (sets colours/light).
    Limits: texture resolution is the exterior camera's; errors of the exterior calibration and of the (approximate)
    wrist camera pose move the texture; the floor is ASSUMED (build_scene --floor_below) and there is no wall."""

    def __init__(self, m, plate, ext_comp, size, cam="wrist", plate2=None, colour_gain=False, table_tex=None):
        self.m, self.ext, self.cam, self.plate2 = m, ext_comp, cam, plate2
        # table_tex (wrist_table_texture, 10-19): table-top texture mosaicked from the REAL WRIST video; used first on
        # table pixels. G3/G4 showed the exterior-plate texture (blurry at this range) makes GR00T plan ~4 cm low
        self.table_tex = table_tex
        # colour_gain: scale the exterior texture by plate2's frame-0 ext->wrist gain. Off by default: the real wrist
        # camera's exposure changes during an episode, and the frame-0 gain made later frames too yellow (res31, 10-15)
        self.gain = plate2["gain"] if (colour_gain and plate2 is not None) else 1.0
        self.H, self.W = size
        self.plate = plate.astype(np.float32)
        self.cid, self.ext_cid = m.camera(cam).id, m.camera("ext").id
        self.Kp = cam_K(m, self.ext_cid, plate.shape[1], plate.shape[0])  # exterior K at the plate's resolution
        self.Kw = cam_K(m, self.cid, self.W, self.H)
        u, v = np.meshgrid(np.arange(self.W) + 0.5, np.arange(self.H) + 0.5)
        self.rays = np.stack([(u - self.Kw[0, 2]) / self.Kw[0, 0], (v - self.Kw[1, 2]) / self.Kw[1, 1], np.ones_like(u)], -1)
        self.rend = mujoco.Renderer(m, self.H, self.W)

    def __call__(self, d, return_info=False):
        m, rend, ext = self.m, self.rend, self.ext
        rend.update_scene(d, camera=self.cam, scene_option=ext.opt_all)
        sim = rend.render().astype(np.float32)
        rend.enable_segmentation_rendering()
        rend.update_scene(d, camera=self.cam, scene_option=ext.opt_all)
        seg = rend.render()
        rend.disable_segmentation_rendering()
        rend.enable_depth_rendering()
        rend.update_scene(d, camera=self.cam, scene_option=ext.opt_all)
        depth = rend.render()
        rend.disable_depth_rendering()
        is_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
        fg = is_geom & np.isin(seg[..., 0], ext.fg_geoms)
        # world points of the wrist pixels -> exterior plate pixels
        R, t = cam_pose(d, self.cid)
        pw = (self.rays * depth[..., None]) @ R.T + t
        Re, te = cam_pose(d, self.ext_cid)
        pc = (pw - te) @ Re
        z = np.maximum(pc[..., 2], 1e-6)
        mx = (pc[..., 0] / z * self.Kp[0, 0] + self.Kp[0, 2] - 0.5).astype(np.float32)
        my = (pc[..., 1] / z * self.Kp[1, 1] + self.Kp[1, 2] - 0.5).astype(np.float32)
        inside = (pc[..., 2] > 0) & (mx >= 0) & (mx <= self.plate.shape[1] - 1) & (my >= 0) & (my <= self.plate.shape[0] - 1)
        tex = cv2.remap(self.plate, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        bg = is_geom & ~fg & inside
        out = sim.copy()
        out[bg] = tex[bg] * self.gain
        bg2 = np.zeros_like(bg)
        if self.plate2 is not None:
            p2 = self.plate2
            q = (pw - p2["t"]) @ p2["R"]
            z2 = np.maximum(q[..., 2], 1e-6)
            nx = (q[..., 0] / z2 * p2["K"][0, 0] + p2["K"][0, 2] - 0.5).astype(np.float32)
            ny = (q[..., 1] / z2 * p2["K"][1, 1] + p2["K"][1, 2] - 0.5).astype(np.float32)
            h2, w2 = p2["img"].shape[:2]
            inside2 = (q[..., 2] > 0) & (nx >= 0) & (nx <= w2 - 1) & (ny >= 0) & (ny <= h2 - 1)
            bg2 = is_geom & ~fg & ~inside & inside2
            tex2 = cv2.remap(p2["img"], nx, ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            out[bg2] = tex2[bg2]
        bg = bg | bg2
        bg3 = np.zeros_like(bg)
        if self.table_tex is not None:  # wrist-camera mosaic of the table top: first choice where it has texture
            tt = self.table_tex
            ix = ((pw[..., 0] - tt["x0"]) / tt["res"] - 0.5).astype(np.float32)
            iy = ((pw[..., 1] - tt["y0"]) / tt["res"] - 0.5).astype(np.float32)
            ok = cv2.remap(tt["valid"], ix, iy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
            bg3 = is_geom & (seg[..., 0] == ext.table) & ok
            tex3 = cv2.remap(tt["tex"], ix, iy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            out[bg3] = tex3[bg3]
            bg = bg | bg3
        if ext.shadows:
            m.geom_group[ext.fg_geoms] = HIDE_GROUP
            rend.update_scene(d, camera=self.cam, scene_option=ext.opt_hide)
            lit = rend.render().astype(np.float32)
            m.geom_group[:] = ext.group0
            on_table = is_geom & (seg[..., 0] == ext.table) & bg
            ratio = np.ones((self.H, self.W), np.float32)
            r = sim[on_table].mean(-1) / np.maximum(lit[on_table].mean(-1), 1.0)
            if ext.light:
                ratio[on_table] = 1.0 - np.clip((1.0 - r) / 0.1, 0.0, 1.0) * (1.0 - ext.light["real_shadow_ratio"])
            else:
                ratio[on_table] = np.clip(r, 0.35, 1.0)
            out *= cv2.GaussianBlur(ratio, (0, 0), 3)[..., None]
        if return_info:
            return out, dict(seg=seg, depth=depth, fg=fg, bg=bg, bg2=bg2, bg3=bg3, mx=mx, my=my, inside=inside)
        return out

    def close(self):
        self.rend.close()


def near_defocus(img, depth, sigma, z_near=0.20):
    """Defocus of the near layer (the gripper, closer than z_near) with its own alpha, so its edges bleed over the
    background as real defocus does (G6, 10-19)."""
    if sigma <= 0:
        return img
    a = (depth < z_near).astype(np.float32)
    f = img.astype(np.float32)
    ab = cv2.GaussianBlur(a, (0, 0), sigma)
    fb = cv2.GaussianBlur(f * a[..., None], (0, 0), sigma) / np.maximum(ab[..., None], 1e-3)
    return ab[..., None] * fb + (1 - ab[..., None]) * f


def load_gripper_library(path):
    z = np.load(path)
    return dict(rgb=z["rgb"][0].astype(np.float32), alpha=z["alpha"][0].astype(np.float32), open_max=float(z["open_max"][0]))


class WristRenderV2:
    """Wrist view v2 (10-20; opt-in, WristCompositor stays the default). G6 showed GR00T plans like on the real image
    only when the wrist view matches the real camera in table texture, gripper look and camera blur. This renders at
    the camera's full 1280x720 with the wrist-video table texture (table_tex), then
      - open gripper (opening <= library open_max + open_tol): the sim gripper is replaced by the REAL gripper image
        (gripper_library.py; rigid w.r.t. the camera); where the sim gripper sticks out beyond the real one, the table
        texture behind it is shown (ray/table-plane intersection);
      - otherwise: the sim gripper with near-layer defocus (blur_sigma px at 1280x720);
    and shrinks to `size` by area averaging, as real frames reach the policy (rendered at work_size first).
    Light/shadow: optional wrist-fitted light (S4, `light=`); the shadow comes from the sim robot with that light."""

    def __init__(self, m, plate, ext_comp, size, plate2, table_tex, grip_lib=None, blur_sigma=8.0, open_tol=0.03, cam="wrist",
                 work_size=(360, 640), light=None):
        """work_size: resolution rendered before the area downscale to `size` (full 1280x720 took 0.5 s/frame, which
        would add ~4 min per 30 s rollout; 640x360 keeps the averaging-down, at ~1/4 the cost). blur_sigma is in
        1280x720 pixels and scaled to work_size; the gripper library is resized to work_size.
        light: light_wrist.json (light_fit_wrist.py, S4): its direction and shadow darkness are used for the wrist
        render only (the model's light is restored afterwards, so the exterior composite keeps its own fit)."""
        self.ext = ext_comp
        self.light = None
        if light is not None:
            self.light = dict(dir=np.asarray(light["dir"], float), real_shadow_ratio=float(light["real_shadow_ratio"]))
        self.m, self.size, self.tt = m, size, table_tex
        self.open_tol = open_tol
        self.cid = m.camera(cam).id
        W0, H0 = (int(v) for v in m.cam_sensorsize[self.cid])
        wh, ww = work_size
        self.blur_sigma = blur_sigma * ww / W0
        self.lib = None
        if grip_lib is not None:
            self.lib = dict(open_max=grip_lib["open_max"],
                            rgb=cv2.resize(grip_lib["rgb"], (ww, wh), interpolation=cv2.INTER_AREA),
                            alpha=cv2.resize(grip_lib["alpha"], (ww, wh), interpolation=cv2.INTER_AREA))
        self.hi = WristCompositor(m, plate, ext_comp, (wh, ww), cam=cam, plate2=plate2, table_tex=table_tex)
        self.grip = np.array([g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(("fr3_", "g_"))])
        drv = m.joint("g_right_driver_joint")
        self.drv_q, self.drv_max = drv.qposadr[0], m.jnt_range[drv.id][1]

    def __call__(self, d, return_info=False):
        if self.light is not None:  # wrist-fitted light for this render only
            saved_dir, saved_light = self.m.light_dir[0].copy(), self.ext.light
            self.m.light_dir[0] = self.light["dir"]
            self.ext.light = dict(saved_light or {}, real_shadow_ratio=self.light["real_shadow_ratio"])
            try:
                out, info = self.hi(d, return_info=True)
            finally:
                self.m.light_dir[0] = saved_dir
                self.ext.light = saved_light
        else:
            out, info = self.hi(d, return_info=True)
        seg = info["seg"]
        gm = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], self.grip)
        opening = float(np.clip(d.qpos[self.drv_q] / self.drv_max, 0, 1))
        use_lib = self.lib is not None and opening <= self.lib["open_max"] + self.open_tol
        if use_lib:
            # table texture behind the sim gripper: intersect each pixel's ray with the table plane
            R, t = cam_pose(d, self.cid)
            dirs = self.hi.rays @ R.T
            s = (self.tt["z"] - t[2]) / np.where(np.abs(dirs[..., 2]) > 1e-6, dirs[..., 2], -1e-6)
            pw = t + dirs * s[..., None]
            ix = ((pw[..., 0] - self.tt["x0"]) / self.tt["res"] - 0.5).astype(np.float32)
            iy = ((pw[..., 1] - self.tt["y0"]) / self.tt["res"] - 0.5).astype(np.float32)
            behind = cv2.remap(self.tt["tex"], ix, iy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            fill = gm & (s > 0)
            out[fill] = behind[fill]
            a = self.lib["alpha"][..., None]
            out = a * self.lib["rgb"] + (1 - a) * out
        else:
            out = near_defocus(out, info["depth"], self.blur_sigma)
        H, W = self.size
        small = cv2.resize(np.clip(out, 0, 255).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        if return_info:
            return small, dict(info, used_library=use_lib, opening=opening)
        return small

    def close(self):
        self.hi.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default="replay", help="<run>_states.npz from replay.py or scripted_grasp.py (--tag)")
    ap.add_argument("--no_shadows", action="store_true")
    ap.add_argument("--states", default=None, help="explicit *_states.npz (e.g. a make_tasks episode) instead of <out>/<run>_states.npz")
    ap.add_argument("--video_out", default=None, help="output mp4 path (default <out>/composite_<run>.mp4)")
    ap.add_argument("--composite_only", action="store_true", help="write the composite at full resolution, no real-frame panel")
    ap.add_argument("--inpaint", choices=["telea", "lama"], default="telea", help="how the real objects are removed from the plate")
    ap.add_argument("--holes", choices=["frame0", "frame0+plate"], default="frame0",
                    help="frame0: perception masks from frame 0; +plate: also objects detected in the median plate")
    a = ap.parse_args()

    m = mujoco.MjModel.from_binary_path(os.path.join(a.out, "scene.mjb"))
    d = mujoco.MjData(m)
    st = np.load(a.states or os.path.join(a.out, f"{a.run}_states.npz"))
    per = dict(np.load(os.path.join(a.out, "perception.npz")))
    real = read_video(os.path.join(a.episode, f"{a.cam}.mp4"))
    H, W = real[0].shape[:2]
    plate = load_plate(a.out, a.inpaint, a.holes, real, per)
    cv2.imwrite(os.path.join(a.out, f"plate_clean_{a.inpaint}.png") if a.inpaint != "telea" else os.path.join(a.out, "plate_clean.png"), cv2.cvtColor(plate.astype(np.uint8), cv2.COLOR_RGB2BGR))
    comp = Compositor(m, a.out, plate, per=per, shadows=not a.no_shadows)

    writer = imageio.get_writer(a.video_out or os.path.join(a.out, f"composite_{a.run}.mp4"), fps=FPS, quality=8, macro_block_size=8)
    for k, q in enumerate(st["qpos"]):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        out = comp(d)
        if a.composite_only:  # full resolution, no side panel, no burned-in text
            writer.append_data(out.clip(0, 255).astype(np.uint8))
            continue
        frame = real[int(st["real_frames"][k])]
        row = np.concatenate([frame, out.clip(0, 255).astype(np.uint8)], 1)
        row = cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2), interpolation=cv2.INTER_AREA)
        cv2.putText(row, "real" if a.run == "replay" else "real (first frame)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(row, f"composite ({a.run})", (W // 2 + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.append_data(row)
    writer.close()
    print("wrote", os.path.join(a.out, f"composite_{a.run}.mp4"), len(st["qpos"]), "frames")


if __name__ == "__main__":
    main()
