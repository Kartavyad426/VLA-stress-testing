"""TaskGen MuJoCo layer: a reconstructed scene as an editable MjSpec (docs/TASKGEN_DESIGN.html §3d, §12a).

res31's build_scene.py writes scene.mjb (what everything loads) and scene.xml (for reading). The .mjb can't become an
MjSpec, and scene.xml doesn't compile on its own: the robot meshes are relative to meshdir="assets/" and the
camera intrinsics are set on the compiled model, not in the XML. load_base() reads res31's scene.xml, points every
relative mesh/texture file at the Menagerie folder that holds it, compiles, and re-applies the two cameras the way
build_scene does (calibrate_cam.set_camera with calib.json and scene_fit.json's wrist K). equivalence() compares the
result field by field with res31's scene.mjb, so any difference is reported, not assumed away; replay_equivalence()
runs the recorded episode's joint targets on both models and compares the trajectories, which is the test that
decides (10-28 on orange: field differences are XML float rounding ~1e-7, the fixed base link's mass and a ~0.03%
inertia constant; replay: arm 2e-14, objects <= 0.05 mm).

res31's modules are imported read-only from the main checkout. Exploratory, UNVERIFIED.
"""
import json, os, sys
import numpy as np
import mujoco

MAIN = "/home/imerit/Documents/Code/VLA"  # main checkout: res31's code, data and scenes (read-only)
sys.path.insert(0, os.path.join(MAIN, "experiments/real2sim"))
sys.path.insert(0, os.path.join(MAIN, "experiments/asset_pack"))
MESH_DIRS = [os.path.join(MAIN, "data/real2sim/menagerie", d, "assets") for d in ("franka_fr3", "robotiq_2f85")]


def _resolve(fname):
    """Absolute path of a relative Menagerie asset file; it must exist in exactly one of MESH_DIRS."""
    if os.path.isabs(fname):
        return fname
    hits = [os.path.join(d, fname) for d in MESH_DIRS if os.path.exists(os.path.join(d, fname))]
    if len(hits) != 1:
        raise RuntimeError(f"asset file {fname!r}: found in {len(hits)} Menagerie folders {hits}")
    return hits[0]


def load_spec(scene_dir):
    """res31's scene.xml as an MjSpec with absolute asset paths (not compiled)."""
    spec = mujoco.MjSpec.from_file(os.path.join(scene_dir, "scene.xml"))
    for ms in spec.meshes:
        if ms.file:
            ms.file = _resolve(ms.file)
    for tx in spec.textures:
        if tx.file:
            tx.file = _resolve(tx.file)
    return spec


def set_cameras(model, scene_dir):
    """Re-apply build_scene's post-compile camera settings (exterior from calib.json, wrist from scene_fit.json)."""
    from calibrate_cam import set_camera
    from build_scene import wrist_cam_local
    fit = json.load(open(os.path.join(scene_dir, "scene_fit.json")))
    W, H = int(model.vis.global_.offwidth), int(model.vis.global_.offheight)
    calib = os.path.join(scene_dir, "calib.json")
    if os.path.exists(calib):
        c = json.load(open(calib))
        set_camera(model, model.camera("ext").id, np.array(c["R_cam_to_base"]), np.array(c["t_cam_in_base"]), np.array(c["K"]), W, H, "offset")
    wc = fit.get("wrist_camera")
    if wc and "K" in wc:
        Rw, tw = wrist_cam_local(np.array(wc["gripper_offset"]))
        set_camera(model, model.camera("wrist").id, Rw, tw, np.array(wc["K"]), 1280, 720, "offset")


def compile_scene(spec, scene_dir):
    model = spec.compile()
    set_cameras(model, scene_dir)
    return model


# Fields known to differ after the XML round trip and why (orange, 10-28). They are reported separately, not hidden.
KNOWN = {"mesh_pathadr": "absolute mesh paths are longer strings",
         "body_mass": "fr3_link0 (welded to the world: no dynamic effect)", "body_subtreemass": "follows body_mass",
         "body_inertia": "fr3_link0", "body_ipos": "fr3_link0", "body_iquat": "fr3_link0",
         "cam_poscom0": "follows the subtree COM", "light_poscom0": "follows the subtree COM", "bvh_aabb": "fr3_link0",
         "body_invweight0": "~0.03% on free objects (float rounding of sizes)", "dof_invweight0": "same"}


def equivalence(a, b, rtol=1e-5, atol=1e-6):
    """Every numeric array of two MjModels compared; returns {field: max abs diff} for fields that differ beyond
    rtol/atol or whose shapes differ, plus 'names' if the name tables differ. Empty dict = identical within tolerance.
    Defaults absorb XML float printing (~1e-7); KNOWN lists the fields that differed on orange beyond that."""
    out = {}
    for f in dir(a):
        if f.startswith("_"):
            continue
        try:
            x, y = getattr(a, f), getattr(b, f)
        except Exception:
            continue
        if not isinstance(x, np.ndarray) or x.dtype.kind not in "fiub":
            continue
        if x.shape != y.shape:
            out[f] = f"shape {x.shape} vs {y.shape}"
        elif x.size and not np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True):
            out[f] = float(np.nanmax(np.abs(x.astype(float) - y.astype(float))))
    for grp in ("opt",):
        for f in dir(getattr(a, grp)):
            if f.startswith("_"):
                continue
            x, y = getattr(getattr(a, grp), f), getattr(getattr(b, grp), f)
            if isinstance(x, (int, float, np.ndarray)) and not np.allclose(x, y, rtol=rtol, atol=atol):
                out[f"{grp}.{f}"] = (np.asarray(x).tolist(), np.asarray(y).tolist())
    if bytes(a.names) != bytes(b.names):
        out["names"] = "name tables differ"
    return out


def replay_equivalence(a, b, joint_positions, substeps=33):
    """Drive both models with the same recorded arm joint targets (one per frame, `substeps` physics steps each;
    33 x 2 ms ~ DROID's 15 Hz) from the same start; return max |qpos difference| for the arm and for everything else."""
    out = []
    for m in (a, b):
        d = mujoco.MjData(m)
        arm = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
        u = [m.actuator(f"fr3_joint{k}").id for k in range(1, 8)]
        d.qpos[arm] = joint_positions[0]
        mujoco.mj_forward(m, d)
        traj = []
        for q in joint_positions:
            d.ctrl[u] = q
            for _ in range(substeps):
                mujoco.mj_step(m, d)
            traj.append(d.qpos.copy())
        out.append((np.array(traj), arm))
    (ta, arm), (tb, _) = out
    rest = [i for i in range(ta.shape[1]) if i not in arm]
    return dict(frames=len(joint_positions), arm_max_abs=float(np.abs(ta[:, arm] - tb[:, arm]).max()),
                other_max_abs=float(np.abs(ta[:, rest] - tb[:, rest]).max()) if rest else 0.0)
