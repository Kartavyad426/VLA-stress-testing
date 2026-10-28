"""Equivalence of r2s_common with the code it was copied from (res31 10-28). Each check compares outputs on the orange
scene (or random points), not source text:
  geometry  vs TaskGen core/geometry.py (random point clouds), implres31 common.collision_points / footprint (orange),
            res31 scene_checks.surface_dist (orange grasp point)
  kinds     vs TaskGen core/kinds.py (bounds on a grid of sizes)
  scene     vs implres31 common.SceneFrames (pose, projection, seg + depth render at the pre-touch frames)
  detect    vs implres31 common.Detector (boxes and masks for 'block' on orange frame 0); --fast skips it (two model loads)
Exits 1 on any mismatch.

    runs/real2sim/v1_dev/py_triton_first.sh experiments/real2sim/tests/test_r2s_common.py [--fast]
"""
import importlib.util, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
R2S = os.path.dirname(HERE)
sys.path.insert(0, R2S)
import r2s_common  # noqa: E402,F401  (triton before mujoco)
import numpy as np  # noqa: E402
import mujoco  # noqa: E402
from r2s_common import geometry as G, kinds as Kd, scene as Sc  # noqa: E402

SIB = os.path.dirname(os.path.dirname(os.path.dirname(R2S)))
TASKGEN = os.path.join(SIB, "VLA-exp-taskgen/experiments/taskgen/core")
CHECKS = os.path.join(SIB, "VLA-exp-real2sim-checks/experiments/real2sim_checks")
SCENE = os.path.join(os.path.dirname(os.path.dirname(R2S)), "runs/real2sim/v1/orange")
EP = os.path.join(os.path.dirname(os.path.dirname(R2S)), "data/real2sim/droid/ep_orange_block")
fails = []


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def same(name, a, b, tol=0.0):
    if isinstance(a, (list, tuple)) and a and isinstance(a[0], str):
        ok = list(a) == list(b)
    else:
        ok = (a == b) if not isinstance(a, (np.ndarray, float, list, tuple)) else np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol, equal_nan=True)
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


# geometry vs TaskGen
if os.path.exists(TASKGEN):
    tg = load(os.path.join(TASKGEN, "geometry.py"), "tg_geometry")
    rng = np.random.default_rng(0)
    for i in range(20):
        P = rng.normal(size=(200, 3)) * [0.05, 0.03, 0.04] + rng.normal(size=3)
        Q = rng.normal(size=(150, 3)) * 0.05 + rng.normal(size=3)
        for k in ("lo", "hi", "centre", "size"):
            same(f"aabb {k} #{i}", G.aabb(P)[k], tg.aabb(P)[k])
        same(f"oriented_size #{i}", G.oriented_size(P), tg.oriented_size(P))
        same(f"horizontal_gap #{i}", G.horizontal_gap(G.aabb(P), G.aabb(Q)), tg.horizontal_gap(tg.aabb(P), tg.aabb(Q)))
        a, b = G.cavity(P), tg.cavity(P)
        same(f"cavity #{i}", a is None, b is None)
    tk = load(os.path.join(TASKGEN, "kinds.py"), "tg_kinds")
    for kind in list(tk.BOUNDS) + ["item"]:
        for s in np.linspace(0.01, 0.25, 13):
            size = np.array([s, s * 0.8, s * 1.2])
            same(f"within_bounds {kind} {s:.3f}", Kd.within_bounds(kind, size)[0], tk.within_bounds(kind, size)[0])
            same(f"graspable {s:.3f}", Kd.graspable(size), tk.graspable(size))
else:
    print("skip TaskGen comparisons (worktree not found)")

# MuJoCo geometry, scene vs implres31; surface_dist vs res31 scene_checks
m = mujoco.MjModel.from_binary_path(os.path.join(SCENE, "scene.mjb"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
free = [m.jnt_bodyid[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
import scene_checks as SCK  # noqa: E402
p = np.array([0.6, 0.03, -0.07])
for b in free:
    same(f"surface_dist body {m.body(b).name}", G.surface_dist(m, d, b, p), SCK.surface_dist(m, d, b, p))
if os.path.exists(CHECKS):
    sys.argv += ["--real2sim", R2S]
    ic = load(os.path.join(CHECKS, "common.py"), "ic_common")
    for b in free:
        geoms = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == b]
        same(f"collision_points {m.body(b).name}", G.collision_points(m, d, geoms), ic.collision_points(m, d, geoms))
        a, b_ = G.footprint(m, d, geoms), ic.footprint(m, d, geoms)
        same(f"footprint {m.body(b).name}", np.r_[a[0], a[1], a[2]], np.r_[b_[0], b_[1], b_[2]])
    s1, s2 = Sc.SceneFrames(SCENE, EP), ic.SceneFrames(SCENE, EP)
    same("frames", s1.frames(), s2.frames())
    for fr in s1.frames()[:3]:
        same(f"pose frame {fr}", s1.pose(fr), s2.pose(fr))
        same(f"qpos frame {fr}", s1.d.qpos, s2.d.qpos)
        for cam in ("ext", "wrist"):
            P = s1.d.xpos[free]
            same(f"project {cam} {fr}", s1.project(cam, (180, 320), P)[0], s2.project(cam, (180, 320), P)[0])
            g1, z1 = s1.render(cam, (180, 320))
            g2, z2 = s2.render(cam, (180, 320))
            same(f"render seg {cam} {fr}", g1, g2)
            same(f"render depth {cam} {fr}", z1, z2, tol=1e-6)
    if "--fast" not in sys.argv:
        from r2s_common import detect as Dt
        from perceive import read_frame
        img = read_frame(os.path.join(EP, "22008760.mp4"), 0)
        d1, d2 = Dt.Detector(), ic.Detector()
        b1, b2 = d1.boxes(img, "block"), d2.boxes(img, "block")
        same("detector boxes 'block' count", len(b1), len(b2))
        same("detector boxes 'block'", [x[1] for x in b1], [x[1] for x in b2], tol=1e-4)
        k1, k2 = d1.masks(img, "block"), d2.masks(img, "block")
        same("detector masks 'block'", np.array([mk.sum() for mk in k1]), np.array([mk.sum() for mk in k2]))
        same("head_noun", [Dt.head_noun(x) for x in ("green block#2", "black bowl", "red radish#1")],
             [ic.head_noun(x) for x in ("green block#2", "black bowl", "red radish#1")])
else:
    print("skip implres31 comparisons (worktree not found)")
print(f"\n{'ALL SAME' if not fails else str(len(fails)) + ' MISMATCHES: ' + ', '.join(fails[:10])}")
sys.exit(1 if fails else 0)
