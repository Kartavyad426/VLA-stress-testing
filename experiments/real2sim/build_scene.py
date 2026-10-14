"""Real-to-sim step 2: fit a MuJoCo scene to the perception output of one DROID frame.

Inputs: <out>/perception.npz (perceive.py) + the episode's trajectory.h5 (camera extrinsics, robot state).
Robot: Menagerie FR3 + Robotiq 2F-85 (DROID hardware). Objects:
  - table  -> box, top at the fitted table height
  - blocks -> boxes sized and placed from their masked points
  - bowl   -> retrieved LIBERO akita_black_bowl, uniformly scaled to the fitted rim diameter
Metric scale: MoGe-2's metric depth, optionally corrected with the robot gripper as a known-geometry
anchor (--scale anchor, the default) or left raw (--scale raw).

Writes <out>/scene.mjb (+ scene.xml for reading), <out>/scene_fit.json, <out>/projection_check.png.

    .venvs/real2sim/bin/python experiments/real2sim/build_scene.py \
        --episode data/real2sim/droid/ep_orange_block --cam 22008760 --out runs/real2sim/ep_orange_block
"""
import argparse, json, os
import numpy as np
import cv2
import h5py
import mujoco
from scipy.spatial.transform import Rotation as Rot

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MENAGERIE = os.path.join(REPO, "data/real2sim/menagerie")
BOWL_DIR = os.path.join(REPO, "third_party/LIBERO-plus/libero/libero/assets/stable_scanned_objects/akita_black_bowl")
BOWL_ASSET_SCALE = 0.7  # the scale LIBERO's akita_black_bowl.xml applies to its mesh
NOT_OBJECTS = ("table", "gripper", "robot")


def object_kind(label):
    """How a detected label is modelled: bowl -> retrieved LIBERO bowl; block/cube/box -> fitted box;
    other objects (fruit, toys) -> fitted ellipsoid (a physics stand-in until a mesh source exists)."""
    low = label.lower()
    if any(w in low for w in NOT_OBJECTS):
        return None
    if "bowl" in low:
        return "bowl"
    if any(w in low for w in ("block", "cube", "box")):
        return "box"
    return "ellipsoid"


def cam_pose(extr):
    """DROID extrinsics [x y z rx ry rz] -> (R, t) of the OpenCV camera in the robot base frame."""
    return Rot.from_euler("xyz", extr[3:]).as_matrix(), np.asarray(extr[:3])


def project(K, R, t, p_base):
    pc = (np.atleast_2d(p_base) - t) @ R  # base -> camera: R^T (p - t)
    uv = pc[:, :2] / pc[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, pc[:, 2]


def masked_points(per, name, erode=3):
    m = per[f"mask/{name}"] & per["valid"]
    if erode:
        m = cv2.erode(m.astype(np.uint8), np.ones((erode, erode), np.uint8)).astype(bool)
    return per["points"][m]


def fit_box(pb, z_table):
    z_top = np.percentile(pb[:, 2], 95)
    top = pb[pb[:, 2] > z_top - 0.006]
    c = top[:, :2].mean(0)
    evals, evecs = np.linalg.eigh(np.cov((top[:, :2] - c).T))
    ext = [np.ptp((top[:, :2] - c) @ evecs[:, i]) for i in (0, 1)]
    yaw = float(np.arctan2(evecs[1, 1], evecs[0, 1]))
    h = float(z_top - z_table)
    return dict(center=[float(c[0]), float(c[1]), float(z_table + h / 2)], half=[ext[1] / 2, ext[0] / 2, h / 2],
                yaw=yaw, n_points=int(len(pb)), n_top=int(len(top)))


def fit_ellipsoid(pb, z_table):
    z_top = np.percentile(pb[:, 2], 95)
    c = np.median(pb[:, :2], axis=0)
    evals, evecs = np.linalg.eigh(np.cov((pb[:, :2] - c).T))
    ext = [np.percentile((pb[:, :2] - c) @ evecs[:, i], 95) - np.percentile((pb[:, :2] - c) @ evecs[:, i], 5) for i in (0, 1)]
    h = float(z_top - z_table)
    return dict(center=[float(c[0]), float(c[1]), float(z_table + h / 2)], half=[ext[1] / 2, ext[0] / 2, h / 2],
                yaw=float(np.arctan2(evecs[1, 1], evecs[0, 1])), n_points=int(len(pb)))


def fit_circle(xy):
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy ** 2).sum(1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    return sol[:2], float(np.sqrt(sol[2] + sol[:2] @ sol[:2]))


def bowl_asset():
    """Collision boxes and size of the LIBERO bowl, at LIBERO's own scale."""
    spec = mujoco.MjSpec.from_file(os.path.join(BOWL_DIR, "akita_black_bowl.xml"))
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    boxes = []
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and m.geom_contype[g]:
            boxes.append((d.geom_xpos[g].copy(), d.geom_xmat[g].reshape(3, 3).copy(), m.geom_size[g].copy()))
    body0 = d.xpos[m.geom_bodyid[0]].copy()
    pts = np.array([p for p, _, _ in boxes]) - body0
    radius = float(np.max(np.linalg.norm(pts[:, :2], axis=1)))
    zmin = min(p[2] - s.max() for p, _, s in boxes) - body0[2]
    zmax = max(p[2] + s.max() for p, _, s in boxes) - body0[2]
    return boxes, body0, radius, float(zmin), float(zmax)


def robot_spec(grip_yaw_deg):
    spec = mujoco.MjSpec.from_file(os.path.join(MENAGERIE, "franka_fr3/fr3.xml"))
    grip = mujoco.MjSpec.from_file(os.path.join(MENAGERIE, "robotiq_2f85/2f85.xml"))
    site = spec.site("attachment_site")
    site.quat = Rot.from_euler("z", grip_yaw_deg, degrees=True).as_quat(scalar_first=True)
    spec.attach(grip, site=site, prefix="g_")
    return spec


def sim_robot_depth(R, t, K, W, H, q_arm):
    """z-depth of the robot alone, rendered from the calibrated camera; 0 where there is no robot."""
    from calibrate_cam import set_camera
    spec = robot_spec(0.0)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    spec.worldbody.add_camera(name="ext")
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[[m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]] = q_arm
    mujoco.mj_kinematics(m, d)
    set_camera(m, m.camera("ext").id, R, t, K, W, H, "offset")
    mujoco.mj_camlight(m, d)
    r = mujoco.Renderer(m, H, W)
    r.enable_depth_rendering()
    r.update_scene(d, camera="ext")
    dist = r.render()  # MuJoCo depth is distance along the optical axis, i.e. z-depth
    r.disable_depth_rendering()
    r.enable_segmentation_rendering()
    r.update_scene(d, camera="ext")
    seg = r.render()
    r.close()
    return np.where(seg[..., 0] >= 0, dist, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", choices=["anchor", "raw"], default="anchor")
    ap.add_argument("--grip_yaw_deg", type=float, default=0.0, help="Robotiq mount rotation about the flange z axis")
    ap.add_argument("--calib", default=None, help="calib.json from calibrate_cam.py (camera pose + ZED intrinsics); "
                    "default: DROID's recorded extrinsics + MoGe's intrinsics")
    ap.add_argument("--anchor_frame", type=int, default=60, help="frame for the robot-depth scale anchor (with --calib)")
    a = ap.parse_args()
    anchor_info = {}

    per = dict(np.load(os.path.join(a.out, "perception.npz")))
    info = json.load(open(os.path.join(a.out, "perception.json")))
    f = h5py.File(os.path.join(a.episode, "trajectory.h5"))
    ee = f["observation/robot_state/cartesian_position"][()]
    H, W = per["image"].shape[:2]
    if a.calib:
        c = json.load(open(a.calib))
        R, t, K = np.array(c["R_cam_to_base"]), np.array(c["t_cam_in_base"]), np.array(c["K"])
        # re-derive camera-frame points from MoGe's z-depth with the real intrinsics (depth == points[...,2], checked)
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        z = per["depth"]
        per["points"] = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
    else:
        R, t = cam_pose(f[f"observation/camera_extrinsics/{a.cam}_left"][info["frame"]])
        K = per["K"]

    uv_ee, z_ee = project(K, R, t, ee[info["frame"], :3])
    s_anchor = None
    if a.calib:
        # scale anchor at a frame where the calibration was checked (robot clearly visible): sim robot depth vs
        # MoGe depth on pixels where the sim robot and the real motion mask agree; then carried to the perception
        # frame through static table pixels seen in both frames
        from perceive import read_frame, geometry
        n = a.anchor_frame
        zs = sim_robot_depth(R, t, K, W, H, f["observation/robot_state/joint_positions"][n])
        real_n = np.load(os.path.join(a.out, "robot_masks_motion.npz"))[str(n)]
        geo_n = geometry(read_frame(os.path.join(a.episode, f"{a.cam}.mp4"), n))
        both = (zs > 0) & real_n & geo_n["mask"]
        s_n = float(np.median(zs[both] / geo_n["depth"][both]))
        table = cv2.erode(per["mask/wooden table"].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        static = table & ~cv2.dilate(real_n.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool) & per["valid"] & geo_n["mask"]
        ratio = float(np.median(geo_n["depth"][static] / per["depth"][static]))
        s_anchor = s_n * ratio
        anchor_info = dict(anchor_frame=n, robot_px=int(both.sum()), s_at_anchor_frame=s_n, table_px=int(static.sum()),
                           depth_ratio_anchor_over_perception=ratio)
    elif "mask/robot gripper" in per:
        # legacy anchor: recorded gripper position vs MoGe depth on the gripper mask
        zg = per["depth"][per["mask/robot gripper"] & per["valid"]]
        if len(zg):
            s_anchor = float(z_ee[0] / np.median(zg))
    s = s_anchor if (a.scale == "anchor" and s_anchor) else 1.0

    def to_base(pc):
        return (s * pc) @ R.T + t

    fit = dict(scale_mode=a.scale, scale=s, scale_anchor=s_anchor, **anchor_info, ee_pixel=uv_ee[0].tolist(),
               ee_inside_image=bool(0 <= uv_ee[0, 0] < W and 0 <= uv_ee[0, 1] < H), objects={})

    tb = to_base(masked_points(per, "wooden table", erode=9))
    z_table = float(np.median(tb[:, 2]))
    lo, hi = np.percentile(tb[:, :2], 2, axis=0), np.percentile(tb[:, :2], 98, axis=0)
    fit["table"] = dict(z=z_table, xy_lo=lo.tolist(), xy_hi=hi.tolist(), z_spread_iqr=float(np.subtract(*np.percentile(tb[:, 2], [75, 25]))))

    boxes, body0, r_asset, zmin_a, zmax_a = bowl_asset()
    for key in sorted(k for k in per if k.startswith("mask/")):
        label = key[5:]
        kind = object_kind(label)
        if kind is None:
            continue
        pts = to_base(masked_points(per, label))
        if len(pts) < 50:
            continue
        rgb = (np.median(per["image"][cv2.erode(per[key].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0], axis=0) / 255.0).tolist()
        body = label.replace("#", "_").replace(" ", "_")
        if kind == "box":
            o = fit_box(pts, z_table)
        elif kind == "ellipsoid":
            o = fit_ellipsoid(pts, z_table)
        else:  # bowl: LIBERO's akita_black_bowl, uniformly scaled to the fitted rim
            z_top = float(np.percentile(pts[:, 2], 98))
            rim = pts[pts[:, 2] > z_top - 0.012]
            c, r = fit_circle(rim[:, :2])
            k = r / r_asset
            o = dict(center=[float(c[0]), float(c[1])], rim_radius=r, height_fit=z_top - z_table,
                     height_asset_scaled=k * (zmax_a - zmin_a), asset_scale=k, n_rim=int(len(rim)))
        o.update(kind=kind, body=body, mask=label, rgb=rgb)
        fit["objects"][label] = o

    # ---- assemble the MJCF ----
    spec = robot_spec(a.grip_yaw_deg)
    spec.option.timestep = 0.002
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    spec.option.noslip_iterations = 3
    wb = spec.worldbody
    wb.add_light(pos=[0, 0, 2.5], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7])
    th = 0.03
    ctr = (lo + hi) / 2
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[*((hi - lo) / 2), th],
                pos=[*ctr, z_table - th], rgba=[0.80, 0.60, 0.38, 1], friction=[1.0, 0.005, 0.0001])
    n_bowl = 0
    for label, o in fit["objects"].items():
        if o["kind"] == "bowl":
            k = o["asset_scale"]
            b = wb.add_body(name=o["body"], pos=[*o["center"], z_table - k * zmin_a + 0.001])
            b.add_freejoint()
            mesh = f"bowl_vis_{n_bowl}"
            n_bowl += 1
            spec.add_mesh(name=mesh, file=os.path.join(BOWL_DIR, "akita_black_bowl.obj"), scale=[BOWL_ASSET_SCALE * k] * 3)
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, contype=0, conaffinity=0, rgba=[0.08, 0.08, 0.08, 1], mass=0)
            for p, M, sz in boxes:
                b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=k * (p - body0), size=k * sz, density=300,
                           quat=Rot.from_matrix(M).as_quat(scalar_first=True), rgba=[0.1, 0.1, 0.1, 0], friction=[0.95, 0.3, 0.1])
            continue
        b = wb.add_body(name=o["body"], pos=o["center"], quat=Rot.from_euler("z", o.get("yaw", 0.0)).as_quat(scalar_first=True))
        b.add_freejoint()
        gtype = mujoco.mjtGeom.mjGEOM_BOX if o["kind"] == "box" else mujoco.mjtGeom.mjGEOM_ELLIPSOID
        b.add_geom(type=gtype, size=o["half"], rgba=[*o["rgb"], 1], density=600, friction=[1.0, 0.005, 0.0001])
    # camera: MuJoCo looks down -z with +y up; OpenCV looks down +z with +y down
    wb.add_camera(name="ext", pos=t, quat=Rot.from_matrix(R @ np.diag([1, -1, -1])).as_quat(scalar_first=True),
                  fovy=float(np.degrees(2 * np.arctan(H / (2 * K[1, 1])))))
    model = spec.compile()
    if a.calib:  # real ZED intrinsics on the render camera (principal-point convention checked in calibrate_cam.py)
        from calibrate_cam import set_camera
        set_camera(model, model.camera("ext").id, R, t, K, W, H, "offset")
    # the .mjb is what replay loads (asset paths of attached specs don't survive an XML round trip);
    # the .xml is for reading
    mujoco.mj_saveModel(model, os.path.join(a.out, "scene.mjb"), None)
    open(os.path.join(a.out, "scene.xml"), "w").write(spec.to_xml())
    json.dump(fit, open(os.path.join(a.out, "scene_fit.json"), "w"), indent=1)

    # projection check: recorded end-effector path drawn on the real frame
    vis = cv2.cvtColor(per["image"], cv2.COLOR_RGB2BGR)
    uv, z = project(K, R, t, ee[:, :3])
    for (u, v), zz in zip(uv, z):
        if zz > 0:
            cv2.circle(vis, (int(u), int(v)), 3, (0, 0, 255), -1)
    for o in fit["objects"].values():
        p = o["center"] if len(o["center"]) == 3 else [*o["center"], z_table]
        (u, v), = project(K, R, t, np.array(p))[0]
        cv2.drawMarker(vis, (int(u), int(v)), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
    cv2.imwrite(os.path.join(a.out, "projection_check.png"), vis)
    print(json.dumps({"scale": round(s, 3), "scale_anchor": s_anchor, "z_table": round(z_table, 3),
                      "objects": {n: (o["kind"], [round(x, 3) for x in o["center"]]) for n, o in fit["objects"].items()}}))


if __name__ == "__main__":
    main()
