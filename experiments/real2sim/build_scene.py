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
import argparse, json, os, sys
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
COLOURS = {"red", "orange", "yellow", "green", "blue", "black", "white", "brown", "purple", "pink", "grey", "gray", "toy", "small", "large"}
LIBERO_ASSETS = os.path.join(REPO, "third_party/LIBERO-plus/libero/libero/assets")
sys.path.insert(0, os.path.join(REPO, "experiments/asset_pack"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_roles import table_label, gripper_label  # noqa: E402
import catalogue as packs  # noqa: E402  CC BY asset packs (--assets kitchen|gso|ycb or a list): docs/ASSET_PACK_ADOPTION.html


def asset_catalogue(source):
    """name -> MJCF path, one instance per name (sorted). 'libero': LIBERO + LIBERO-Plus objects (research-only
    licence per res32's audit). Other sources (e.g. CC BY mujoco_scanned_objects) plug in here."""
    import glob
    cat = {}
    if source == "libero":
        for sub in ("stable_scanned_objects", "stable_hope_objects", "turbosquid_objects"):
            for x in sorted(glob.glob(f"{LIBERO_ASSETS}/{sub}/*/*.xml")):
                cat.setdefault(os.path.basename(os.path.dirname(x)), x)
        for x in sorted(glob.glob(f"{LIBERO_ASSETS}/new_objects/*/*/usd/MJCF/*.xml")):
            cat.setdefault(x.split("/new_objects/")[1].split("/")[0], x)
    return cat


def match_asset(label, cat):
    """Detected label -> catalogue name: exact noun first, then 'half_<noun>', then names ending in '_<noun>'."""
    words = [w for w in label.split("#")[0].lower().split() if w not in COLOURS]
    for w in reversed(words):  # the head noun is usually last ("red radish" -> "radish")
        for cand in (w, f"half_{w}"):
            if cand in cat:
                return cand
        ends = sorted(n for n in cat if n.endswith("_" + w))
        if ends:
            return ends[0]
    return None


def asset_footprint(xml):
    """(x/y half-extents, z min, z max) of the asset's visual geoms in its body frame, at native scale."""
    m = mujoco.MjSpec.from_file(xml).compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    pts = []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    p = np.concatenate(pts)
    return (p[:, :2].max(0) - p[:, :2].min(0)) / 2, float(p[:, 2].min()), float(p[:, 2].max()), (p[:, :2].max(0) + p[:, :2].min(0)) / 2


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


def robust_extent(xy, axis_vec, lo=5, hi=95):
    proj = xy @ axis_vec
    return float(np.percentile(proj, hi) - np.percentile(proj, lo))


def fit_box(pb, z_table, pb_inner=None, cube=False):
    """Footprint from the top-face points (PCA axes, 5-95th percentile extents, so a few stray points don't
    stretch a side); height from the inner part of the mask (edge pixels smear depth upward in clutter);
    cube=True ("block"/"cube" labels) sets all three sides to the median of the measured ones."""
    inner = pb_inner if pb_inner is not None and len(pb_inner) > 30 else pb
    z_top = np.percentile(inner[:, 2], 90)
    top = pb[(pb[:, 2] > z_top - 0.006) & (pb[:, 2] < z_top + 0.01)]
    if len(top) < 10:
        top = pb
    c = np.median(top[:, :2], axis=0)
    evals, evecs = np.linalg.eigh(np.cov((top[:, :2] - c).T))
    ext = [robust_extent(top[:, :2] - c, evecs[:, i]) for i in (0, 1)]
    yaw = float(np.arctan2(evecs[1, 1], evecs[0, 1]))
    h = float(z_top - z_table)
    half = [ext[1] / 2, ext[0] / 2, h / 2]
    if cube:
        side = float(np.median([ext[0], ext[1], h]))
        half = [side / 2] * 3
        h = side
    return dict(center=[float(c[0]), float(c[1]), float(z_table + h / 2)], half=half, yaw=yaw,
                measured_wdh=[ext[1], ext[0], float(z_top - z_table)], n_points=int(len(pb)), n_top=int(len(top)))


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
    # true lowest/highest corner of each rotated box (10-19: centre -/+ the largest half-size put the bowl ~2 cm too
    # low in zmin, so the bowl was built floating and dropped 2.2-2.6 cm on load: checklist #1)
    corners = [p[2] + (R @ (np.array([i, j, k]) * s))[2] for p, R, s in boxes for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
    zmin = min(corners) - body0[2]
    zmax = max(corners) - body0[2]
    return boxes, body0, radius, float(zmin), float(zmax)


FLANGE_IN_LINK7 = np.array([0, 0, 0.107])  # fr3.xml attachment_site; == DROID's cartesian_position frame (checked:
#                                           0 mm / 0 deg FK error on ep_orange_block frames 0, 60, 120)


def wrist_cam_local(offset):
    """DROID <wrist>_left_gripper_offset [x y z rx ry rz] (camera pose in the end-effector frame, euler 'xyz';
    checked: cartesian_position composed with it reproduces <wrist>_left exactly) -> (R, t) of the OpenCV camera
    in fr3_link7's frame."""
    return Rot.from_euler("xyz", offset[3:]).as_matrix(), FLANGE_IN_LINK7 + np.asarray(offset[:3])


def robot_spec(grip_yaw_deg, robot="fr3_robotiq", wrist_offset=None):
    """fr3_robotiq: DROID's FR3 + Robotiq 2F-85. panda: Menagerie's Panda + Panda hand (LIBERO's robot), renamed to the
    names the other scripts use (fr3_joint1..7 / fr3_link*, g_* gripper bodies, g_pinch site, g_fingers_actuator) and
    with the gripper actuator flipped so ctrl 255 = closed, as on the Robotiq.
    wrist_offset (fr3_robotiq only): DROID's wrist-camera gripper offset; adds camera 'wrist' on fr3_link7 (pose only;
    the caller sets the ZED intrinsics on the compiled model with calibrate_cam.set_camera)."""
    if robot == "panda":
        spec = mujoco.MjSpec.from_file(os.path.join(MENAGERIE, "franka_emika_panda/panda.xml"))
        rename = {f"link{k}": f"fr3_link{k}" for k in range(8)}
        rename.update({b: f"g_{b}" for b in ("hand", "left_finger", "right_finger")})
        for ex in spec.excludes:  # contact-exclude pairs refer to bodies by name
            ex.bodyname1 = rename.get(ex.bodyname1, ex.bodyname1)
            ex.bodyname2 = rename.get(ex.bodyname2, ex.bodyname2)
        for k in range(8):
            spec.body(f"link{k}").name = f"fr3_link{k}"
        for k in range(1, 8):
            spec.joint(f"joint{k}").name = f"fr3_joint{k}"
            act = spec.actuator(f"actuator{k}")
            act.name, act.target = f"fr3_joint{k}", f"fr3_joint{k}"  # actuators refer to their joint by name
        for b in ("hand", "left_finger", "right_finger"):
            spec.body(b).name = f"g_{b}"
        spec.body("g_hand").add_site(name="g_pinch", pos=[0, 0, 0.1034])  # Franka hand TCP offset
        g = spec.actuator("actuator8")
        g.name = "g_fingers_actuator"
        g.gainprm[0] = -0.01568627451  # target opening 0.04 - ctrl*0.04/255: ctrl 255 = closed
        g.biasprm[0], g.biasprm[1], g.biasprm[2] = 4.0, -100.0, -10.0
        for key in list(spec.keys):
            spec.delete(key)  # the 'home' keyframe would carry the old ctrl convention
        return spec
    spec = mujoco.MjSpec.from_file(os.path.join(MENAGERIE, "franka_fr3/fr3.xml"))
    grip = mujoco.MjSpec.from_file(os.path.join(MENAGERIE, "robotiq_2f85/2f85.xml"))
    site = spec.site("attachment_site")
    site.quat = Rot.from_euler("z", grip_yaw_deg, degrees=True).as_quat(scalar_first=True)
    spec.attach(grip, site=site, prefix="g_")
    if wrist_offset is not None:
        Rw, tw = wrist_cam_local(wrist_offset)
        spec.body("fr3_link7").add_camera(name="wrist", pos=tw, quat=Rot.from_matrix(Rw @ np.diag([1, -1, -1])).as_quat(scalar_first=True))
    return spec


def sim_robot_depth(R, t, K, W, H, q_arm, robot="fr3_robotiq"):
    """z-depth of the robot alone, rendered from the calibrated camera; 0 where there is no robot."""
    from calibrate_cam import set_camera
    spec = robot_spec(0.0, robot)
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
    ap.add_argument("--depth", choices=["mono", "stereo", "grasp"], default="mono",
                    help="mono: MoGe-2 depth x robot-anchored scale (scenes built before 10-16; found 6-8 cm too high); "
                         "stereo: MoGe-2 shape, scale from the <cam>-stereo.mp4 pair on the table (needs --calib); "
                         "grasp: MoGe-2 shape, scale from the real grasp points (--grasp_anchors; 10-27, c11)")
    # 90: the sim wrist view (DROID gripper offset + ZED intrinsics) matches the real DROID wrist video only with the
    # Robotiq at +-90 deg about the flange (impl, 10-15, ep_orange_block frame 50; at 0 the gripper is seen side-on,
    # i.e. the fingers close 90 deg rotated vs the real rig). The sign is unresolved (the 2F-85 looks the same).
    # The scale anchor (sim_robot_depth) still renders the robot at 0, so object fits don't depend on this.
    ap.add_argument("--grip_yaw_deg", type=float, default=90.0, help="Robotiq mount rotation about the flange z axis")
    ap.add_argument("--calib", default=None, help="calib.json from calibrate_cam.py (camera pose + ZED intrinsics); "
                    "default: DROID's recorded extrinsics + MoGe's intrinsics")
    ap.add_argument("--assets", default="libero",
                    help="mesh source for named non-block objects: libero | none (fitted ellipsoids only) | a pack "
                    "kitchen|gso|ycb or a comma list (e.g. gso,ycb). Packs also supply the bowl; an object no pack "
                    "has is left out of the scene with a warning (listed under 'redacted' in scene_fit.json)")
    ap.add_argument("--object_props", default=None, help="JSON of per-label / 'category:<c>' overrides "
                    "(mass kg, friction [3], scale_mult); see experiments/asset_pack/catalogue.py")
    ap.add_argument("--robot", choices=["fr3_robotiq", "panda"], default="fr3_robotiq")
    ap.add_argument("--floor_below", type=float, default=0.75, help="ASSUMED floor depth under the table top (m)")
    ap.add_argument("--anchor_frame", type=int, default=None,
                    help="frame for the robot-depth scale anchor (with --calib); default: the middle calibration frame")
    ap.add_argument("--no_wrist", action="store_true", help="don't add the DROID wrist camera")
    ap.add_argument("--table_z", type=float, default=None, help="table-top z (robot frame) from an independent source "
                    "(wrist stereo, stereo_depth.py); replaces the height fitted from the exterior points, objects stand on it. "
                    "10-27, c11: one MoGe scale left the far table ~4.5 cm low")
    ap.add_argument("--grasp_anchors", default=None, help="--depth grasp: JSON {anchors: {label#k: {point}}} of real fingertip "
                    "points at real grasps (FK of the recorded joints); MoGe is scaled so the grasped objects sit at them")
    a = ap.parse_args()
    anchor_info = {}
    if a.depth in ("stereo", "grasp") and not a.calib:
        raise SystemExit(f"--depth {a.depth} needs --calib")
    if a.depth == "grasp" and not a.grasp_anchors:
        raise SystemExit("--depth grasp needs --grasp_anchors")

    per = dict(np.load(os.path.join(a.out, "perception.npz")))
    info = json.load(open(os.path.join(a.out, "perception.json")))
    TABLE, GRIPPER = table_label(info), gripper_label(info)  # roles from perceive.py (10-26); older scenes: "wooden table"
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
        if a.depth == "stereo":
            # MoGe-2 keeps the shape (small blocks); its scale comes from the ZED stereo pair of the perception frame
            # (stereo_depth.py): median stereo/MoGe ratio on the eroded table mask. Not stereo depth directly: SGBM is
            # valid on ~40% of pixels and smears small blocks, and mixing stereo with scaled MoGe gave a bad table
            # (10-16). MoGe's error is not one global scale (ratio 0.91-1.06 by depth band on ep_orange_block), so the
            # scale is taken where the scene is: the table.
            from stereo_depth import stereo_z, BASELINE_M
            zs = stereo_z(a.episode, a.cam, info["frame"], K[0, 0], BASELINE_M["ext"])
            tab = cv2.erode(per[f"mask/{TABLE}"].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            both = tab & np.isfinite(zs) & per["valid"] & (z > 0)
            r = zs[both] / z[both]
            stereo_scale = float(np.median(r))
            anchor_info["stereo"] = dict(baseline_m=BASELINE_M["ext"], table_px=int(both.sum()), table_valid_frac=float(both.sum() / max(tab.sum(), 1)),
                                         scale=stereo_scale, ratio_iqr=[float(np.percentile(r, 25)), float(np.percentile(r, 75))])
        elif a.depth == "grasp":
            # 10-27 (c11: exterior stereo read ~28% deeper than the robot geometry): scale from the real grasp points.
            # Objects don't move before their first grasp, so the fingertip at that grasp (FK, exact) is where the
            # object sat in the perception frame. Per grasp point: its camera depth vs MoGe's depth on the mask it
            # projects into (median of the mask's inner pixels; the fingertip is mid-object, MoGe sees the front
            # surface, so this reads a few % short on small objects)
            rows = []
            for name, an in json.load(open(a.grasp_anchors))["anchors"].items():
                pc = (np.array(an["point"]) - t) @ R
                uu, vv = (pc[:2] / pc[2] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]).round().astype(int)
                # 10-28: the grasp point is ON or IN the object (rim, handle), so MoGe is read where it lands: at its
                # pixel if that's on an object mask, else at the nearest pixel of the nearest object mask within 40 px
                # (was: the mask's median depth, i.e. its centre, which failed on c9's cup gripped at the rim)
                objm = {k: per[k] & per["valid"] & (z > 0) for k in per if k.startswith("mask/") and k not in (f"mask/{TABLE}", f"mask/{GRIPPER}")}
                best = None
                for k, mk in objm.items():
                    ys, xs = np.nonzero(mk)
                    if not len(xs):
                        continue
                    dd = np.hypot(xs - uu, ys - vv)
                    i = int(np.argmin(dd))
                    if dd[i] <= 40 and (best is None or dd[i] < best[0]):
                        best = (float(dd[i]), k, int(xs[i]), int(ys[i]))
                if best is None:
                    rows.append(dict(anchor=name, px=[int(uu), int(vv)], mask=None))
                    continue
                dpx, k, xu, yv = best
                win = z[max(yv - 2, 0):yv + 3, max(xu - 2, 0):xu + 3][objm[k][max(yv - 2, 0):yv + 3, max(xu - 2, 0):xu + 3]]
                rows.append(dict(anchor=name, px=[int(uu), int(vv)], mask=k[5:], px_to_mask=dpx, cam_depth=float(pc[2]),
                                 moge_depth=float(np.median(win)), ratio=float(pc[2] / np.median(win))))
            used = [r_ for r_ in rows if r_.get("ratio")]
            if not used:
                raise SystemExit(f"--depth grasp: no grasp point projects onto an object mask: {rows}")
            stereo_scale = float(np.median([r_["ratio"] for r_ in used]))
            anchor_info["grasp"] = dict(scale=stereo_scale, points=rows)
    else:
        R, t = cam_pose(f[f"observation/camera_extrinsics/{a.cam}_left"][info["frame"]])
        K = per["K"]

    uv_ee, z_ee = project(K, R, t, ee[info["frame"], :3])
    s_anchor = None
    if a.calib and a.depth == "mono":  # stereo/grasp modes: the scale comes from stereo or the grasp points, not the robot anchor
        # scale anchor at a frame where the calibration was checked (robot clearly visible): sim robot depth vs
        # MoGe depth on pixels where the sim robot and the real motion mask agree; then carried to the perception
        # frame through static table pixels seen in both frames
        from perceive import read_frame, geometry
        n = a.anchor_frame if a.anchor_frame is not None else c["frames"][len(c["frames"]) // 2]  # orange: 60
        zs = sim_robot_depth(R, t, K, W, H, f["observation/robot_state/joint_positions"][n], a.robot)
        real_n = np.load(os.path.join(a.out, "robot_masks_motion.npz"))[str(n)]
        geo_n = geometry(read_frame(os.path.join(a.episode, f"{a.cam}.mp4"), n))
        both = (zs > 0) & real_n & geo_n["mask"]
        s_n = float(np.median(zs[both] / geo_n["depth"][both]))
        table = cv2.erode(per[f"mask/{TABLE}"].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        static = table & ~cv2.dilate(real_n.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool) & per["valid"] & geo_n["mask"]
        ratio = float(np.median(geo_n["depth"][static] / per["depth"][static]))
        s_anchor = s_n * ratio
        anchor_info = dict(anchor_frame=n, robot_px=int(both.sum()), s_at_anchor_frame=s_n, table_px=int(static.sum()),
                           depth_ratio_anchor_over_perception=ratio)
    elif f"mask/{GRIPPER}" in per and a.depth == "mono":
        # legacy anchor: recorded gripper position vs MoGe depth on the gripper mask
        zg = per["depth"][per[f"mask/{GRIPPER}"] & per["valid"]]
        if len(zg):
            s_anchor = float(z_ee[0] / np.median(zg))
    if a.depth in ("stereo", "grasp"):
        s = stereo_scale
    else:
        s = s_anchor if (a.scale == "anchor" and s_anchor) else 1.0

    def to_base(pc):
        return (s * pc) @ R.T + t

    fit = dict(depth=a.depth, scale_mode=a.scale if a.depth == "mono" else f"none ({a.depth})", scale=s, scale_anchor=s_anchor, **anchor_info, ee_pixel=uv_ee[0].tolist(),
               ee_inside_image=bool(0 <= uv_ee[0, 0] < W and 0 <= uv_ee[0, 1] < H), objects={})

    tb = to_base(masked_points(per, TABLE, erode=9))
    z_table = float(np.median(tb[:, 2]))
    if a.table_z is not None:
        fit["table_z_override"] = dict(fitted=z_table, used=a.table_z)
        z_table = a.table_z
    lo, hi = np.percentile(tb[:, :2], 2, axis=0), np.percentile(tb[:, :2], 98, axis=0)
    fit["table"] = dict(z=z_table, xy_lo=lo.tolist(), xy_hi=hi.tolist(), z_spread_iqr=float(np.subtract(*np.percentile(tb[:, 2], [75, 25]))))

    boxes, body0, r_asset, zmin_a, zmax_a = bowl_asset()
    if a.assets not in ("libero", "none") and packs.sources(a.assets) is None:
        raise SystemExit(f"--assets {a.assets}: expected libero, none, or packs from {packs.PACK_NAMES}")
    pack = packs.sources(a.assets) is not None
    if pack:  # libero / none keep scene_fit.json unchanged
        fit["redacted"] = {}
    cat = packs.pack_catalogue(a.assets) if pack else asset_catalogue(a.assets) if a.assets != "none" else {}
    props = packs.load_props(a.object_props)

    def pack_pick(label, size_m, open_top=False):
        """First name-matched pack model whose fitted scale (size_m / its largest x/y half-extent) is plausible.
        open_top: skip models whose collision is one convex hull (YCB), which would fill the cavity."""
        for name in packs.match(label, cat, COLOURS)[:20]:
            if open_top and cat[name]["collision"] != "shipped":
                continue
            half_xy, zmin, zmax, cxy = asset_footprint(cat[name]["xml"])
            k = float(size_m / max(half_xy))
            if packs.SCALE_RANGE[0] <= k <= packs.SCALE_RANGE[1]:
                return name, k, zmin, zmax, cxy
        return None

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
            inner = to_base(masked_points(per, label, erode=9))
            o = fit_box(pts, z_table, inner, cube=any(w in label.lower() for w in ("block", "cube")))
        elif kind == "ellipsoid":
            o = fit_ellipsoid(pts, z_table)
            name = None if pack else match_asset(label, cat)
            if pack:
                cols = np.flatnonzero(per[key].any(0))
                z_med = float(np.median(s * per["depth"][per[key] & per["valid"]]))
                width_m = (np.percentile(cols, 95) - np.percentile(cols, 5)) * z_med / K[0, 0]
                hit = pack_pick(label, width_m / 2)
                if hit:
                    name, k, zmin, zmax, cxy = hit
                    o["half"] = [width_m / 2, width_m / 2, o["half"][2]]
                    # asset = category noun (make_tasks words instructions with it); asset_key = the pack model
                    o.update(kind="asset", asset=cat[name]["category"] or label.split("#")[0].split()[-1], asset_key=name,
                             asset_xml=cat[name]["xml"], asset_scale=k, asset_zmin=zmin,
                             asset_height_scaled=k * (zmax - zmin), asset_cxy=cxy.tolist(), asset_pack=cat[name]["pack"],
                             asset_licence=cat[name]["licence"], asset_collision=cat[name]["collision"])
                    kind = "asset"
                else:  # no plausible pack mesh: left out of the scene (user, 10-20), recorded with what was seen
                    fit["redacted"][label] = dict(reason=f"no plausible model in packs {a.assets}", kind="ellipsoid",
                                                  center=o["center"], width_m=width_m, rgb=rgb)
                    print(f"WARNING: '{label}' redacted: no plausible model in packs {a.assets}", file=sys.stderr)
                    continue
                name = None
            if name:  # retrieved mesh, uniformly scaled so its footprint matches the fitted one
                half_xy, zmin, zmax, cxy = asset_footprint(cat[name])
                cols = np.flatnonzero(per[key].any(0))
                z_med = float(np.median(s * per["depth"][per[key] & per["valid"]]))
                width_m = (np.percentile(cols, 95) - np.percentile(cols, 5)) * z_med / K[0, 0]
                k = float(width_m / (2 * max(half_xy)))
                o["half"] = [width_m / 2, width_m / 2, o["half"][2]]  # footprint from the image, not the point spread
                o.update(kind="asset", asset=name, asset_xml=cat[name], asset_scale=k, asset_zmin=zmin,
                         asset_height_scaled=k * (zmax - zmin), asset_cxy=cxy.tolist())
                kind = "asset"
        else:  # bowl: a pack bowl (--assets kitchen|gso|cc) or LIBERO's akita_black_bowl, scaled to the fitted rim
            z_top = float(np.percentile(pts[:, 2], 98))
            rim = pts[pts[:, 2] > z_top - 0.012]
            c, r = fit_circle(rim[:, :2])
            hit = pack_pick(label, r, open_top=True) if pack else None
            if pack and not hit:
                fit["redacted"][label] = dict(reason=f"no open-top bowl model in packs {a.assets}", kind="bowl",
                                              center=[float(c[0]), float(c[1])], rim_radius=r, rgb=rgb)
                print(f"WARNING: '{label}' redacted: no open-top bowl model in packs {a.assets}", file=sys.stderr)
                continue
            if hit:  # a pack bowl, uniformly scaled so its outer radius matches the fitted rim
                name, k, zmin, zmax, cxy = hit
                o = dict(center=[float(c[0]), float(c[1]), z_table + k * (zmax - zmin) / 2], rim_radius=r, height_fit=z_top - z_table,
                         height_asset_scaled=k * (zmax - zmin), n_rim=int(len(rim)),
                         asset="bowl", asset_key=name, asset_xml=cat[name]["xml"], asset_scale=k, asset_zmin=zmin,
                         asset_height_scaled=k * (zmax - zmin), asset_cxy=cxy.tolist(), asset_pack=cat[name]["pack"],
                         asset_licence=cat[name]["licence"])  # kind stays "bowl": replay/make_tasks look for it
            else:
                k = r / r_asset
                o = dict(center=[float(c[0]), float(c[1])], rim_radius=r, height_fit=z_top - z_table,
                         height_asset_scaled=k * (zmax_a - zmin_a), asset_scale=k, n_rim=int(len(rim)))
        words = [w for w in label.split("#")[0].lower().split() if w not in COLOURS]
        o["props"], o["props_source"] = packs.resolve_props(label, words[-1] if words else None, props)
        if "asset_scale" in o and o["props"]["scale_mult"] != 1.0:
            for f_ in ("asset_scale", "asset_height_scaled", "height_asset_scaled"):
                if f_ in o:
                    o[f_] *= o["props"]["scale_mult"]
        o.update(kind=kind, body=body, mask=label, rgb=rgb)
        fit["objects"][label] = o

    groups = {}
    for lab, o in fit["objects"].items():
        if o["kind"] == "box" and o["half"][0] == o["half"][1] == o["half"][2]:
            groups.setdefault(lab.split("#")[0], []).append(lab)
    for base, labs in groups.items():
        if len(labs) < 3:
            continue
        med = float(np.median([fit["objects"][l]["half"][0] for l in labs]))
        for l in labs:
            o = fit["objects"][l]
            h = float(np.clip(o["half"][0], 0.5 * med, 1.5 * med))
            if h != o["half"][0]:
                o["clamped_from_half"] = o["half"][0]
            o["half"] = [h, h, h]
            o["center"][2] = z_table + h

    # ---- assemble the MJCF ----
    # wrist camera from DROID's (constant) gripper offset; found only roughly right in res31's checks, so treat
    # the wrist view as approximate
    wrist_keys = sorted(k for k in f["observation/camera_extrinsics"] if k.endswith("_left_gripper_offset"))
    wrist = None
    if wrist_keys and a.robot == "fr3_robotiq" and not a.no_wrist:
        serial = wrist_keys[0].split("_")[0]
        wrist = dict(serial=serial, offset=f[f"observation/camera_extrinsics/{wrist_keys[0]}"][0])
        fit["wrist_camera"] = dict(serial=serial, gripper_offset=wrist["offset"].tolist(), approximate=True)
    spec = robot_spec(a.grip_yaw_deg, a.robot, wrist_offset=wrist["offset"] if wrist else None)
    spec.option.timestep = 0.002
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    spec.option.noslip_iterations = 3
    wb = spec.worldbody
    wb.add_light(pos=[0, 0, 2.5], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7])
    th = 0.03
    ctr = (lo + hi) / 2
    # ground plane so objects knocked off the table land instead of falling forever (c3 replay: the bowl ended at
    # z = -11780 m). The real floor is not measured; it is ASSUMED --floor_below m under the table top.
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], pos=[0, 0, z_table - a.floor_below],
                rgba=[0.3, 0.3, 0.3, 1])
    fit["floor"] = dict(z=z_table - a.floor_below, assumed=True, below_table_m=a.floor_below)
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[*((hi - lo) / 2), th],
                pos=[*ctr, z_table - th], rgba=[0.80, 0.60, 0.38, 1], friction=[1.0, 0.005, 0.0001])
    n_bowl = 0
    for label, o in fit["objects"].items():
        if o["kind"] == "bowl" and not o.get("asset_pack"):
            k = o["asset_scale"]
            b = wb.add_body(name=o["body"], pos=[*o["center"], z_table - k * zmin_a + 0.001])
            b.add_freejoint()
            mesh = f"bowl_vis_{n_bowl}"
            n_bowl += 1
            spec.add_mesh(name=mesh, file=os.path.join(BOWL_DIR, "akita_black_bowl.obj"), scale=[BOWL_ASSET_SCALE * k] * 3)
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, contype=0, conaffinity=0, rgba=[0.08, 0.08, 0.08, 1], mass=0)
            pr, src = o["props"], o["props_source"]
            dens = 300 if "mass" not in pr else pr["mass"] / sum(8 * np.prod(k * sz) for _, _, sz in boxes)
            fr_ = pr["friction"] if src["friction"] != "default" else [0.95, 0.3, 0.1]
            for p, M, sz in boxes:
                b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=k * (p - body0), size=k * sz, density=dens,
                           quat=Rot.from_matrix(M).as_quat(scalar_first=True), rgba=[0.1, 0.1, 0.1, 0], friction=fr_)
            continue
        b = wb.add_body(name=o["body"], pos=o["center"], quat=Rot.from_euler("z", o.get("yaw", 0.0)).as_quat(scalar_first=True))
        b.add_freejoint()
        if o["kind"] == "asset" or o.get("asset_pack"):
            k = o["asset_scale"]
            b.pos = [o["center"][0], o["center"][1], z_table - k * o["asset_zmin"] + 0.001]
            if o.get("asset_pack"):  # CC BY pack: keep its collision parts; mass/friction from defaults/overrides
                child, info = packs.prepare_child(o["asset_xml"], k, o["props"])
                o.update(info)
                fr = b.add_frame(pos=[-k * o["asset_cxy"][0], -k * o["asset_cxy"][1], 0])
                spec.attach(child, frame=fr, prefix=o["body"] + "_")
                continue
            child = mujoco.MjSpec.from_file(o["asset_xml"])
            for lt in list(child.lights):
                child.delete(lt)
            # LIBERO-Plus objects carry hundreds of convex collision parts (559 for half_radish); resting on the
            # table they made ~7k contacts and overflowed MuJoCo's constraint arena (everything fell through).
            # Collide with the visual mesh's single convex hull instead and drop the parts.
            vis = [g for g in child.geoms if g.meshname == "visuals"]
            if vis:
                for g in list(child.geoms):
                    if g.meshname != "visuals":
                        child.delete(g)
                for ms in list(child.meshes):
                    if ms.name != "visuals":
                        child.delete(ms)
                for g in vis:
                    g.contype, g.conaffinity, g.density = 1, 1, 500
            for ms in child.meshes:
                ms.scale = [k * v for v in ms.scale]
            fr = b.add_frame(pos=[-k * o["asset_cxy"][0], -k * o["asset_cxy"][1], 0])
            spec.attach(child, frame=fr, prefix=o["body"] + "_")
            continue
        gtype = mujoco.mjtGeom.mjGEOM_BOX if o["kind"] == "box" else mujoco.mjtGeom.mjGEOM_ELLIPSOID
        pr, src = o["props"], o["props_source"]
        g = b.add_geom(type=gtype, size=o["half"], rgba=[*o["rgb"], 1], density=600,
                       friction=pr["friction"] if src["friction"] != "default" else [1.0, 0.005, 0.0001])
        if "mass" in pr:
            g.mass = pr["mass"]
    # camera: MuJoCo looks down -z with +y up; OpenCV looks down +z with +y down
    wb.add_camera(name="ext", pos=t, quat=Rot.from_matrix(R @ np.diag([1, -1, -1])).as_quat(scalar_first=True),
                  fovy=float(np.degrees(2 * np.arctan(H / (2 * K[1, 1])))))
    model = spec.compile()
    if a.calib:  # real ZED intrinsics on the render camera (principal-point convention checked in calibrate_cam.py)
        from calibrate_cam import set_camera
        set_camera(model, model.camera("ext").id, R, t, K, W, H, "offset")
    if wrist:  # ZED intrinsics of the wrist camera at its native 1280x720 (renders at other sizes keep the FOV)
        from calibrate_cam import set_camera, zed_K
        Kw = zed_K(a.episode, wrist["serial"])
        Rw, tw = wrist_cam_local(wrist["offset"])
        set_camera(model, model.camera("wrist").id, Rw, tw, Kw, 1280, 720, "offset")
        fit["wrist_camera"]["K"] = Kw.tolist()
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
    for label, r in fit.get("redacted", {}).items():  # repeated last so it is not lost in the log
        print(f"WARNING: '{label}' left out of the scene: {r['reason']}", file=sys.stderr)


if __name__ == "__main__":
    main()
