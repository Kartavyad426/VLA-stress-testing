"""TaskGen stage 1: can objects be added to an already built scene? (docs/TASKGEN_DESIGN.html §3d)

Runs on the framework / simulator split: the objects are neutral descriptions (core/objects.py) and everything
simulator-specific goes through core.sim.Sim, here sim_mujoco.MujocoSim. On res31's orange scene
(runs/real2sim/ep_orange_block_stereo, 10-19 build, verified per res31 10-28):
  1. base: res31's scene.xml as an editable MjSpec vs res31's scene.mjb: field comparison (with tolerance) and a
     replay of the episode's recorded joint targets on both;
  2. add a red 3 cm box, a blue 2.5 cm box on it, a Kitchen-pack glass cup at its real size, a primitive cabinet with
     an open drawer, and a purple 3 cm box inside the drawer;
  3. overlap check at the start (+ a positive control scene: a box sunk 1 cm into another), settle 1 s with the arm
     held at frame 0, MuJoCo's runtime warnings;
  4. drawer test: push it closed gently (3 N; damping 10 -> <= 0.3 m/s); the box must ride inside;
  5. renders (exterior, wrist, free), .mjb to open in the viewer, JSON report.
Positions are hand-picked free spots; reach and the camera-covered area are NOT checked here.

    runs/real2sim/v1_dev/py_triton_first.sh ../VLA-exp-taskgen/experiments/taskgen/stage1_add.py   (from the main checkout)
Exploratory, UNVERIFIED.
"""
import argparse, json, os, sys, time
import numpy as np
import h5py
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.objects import Box, PackAsset, Drawer, describe  # noqa: E402
from sim_mujoco.base import MAIN, KNOWN, equivalence, replay_equivalence  # noqa: E402
from sim_mujoco.sim import MujocoSim  # noqa: E402

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the TaskGen worktree


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=os.path.join(MAIN, "runs/real2sim/ep_orange_block_stereo"))
    ap.add_argument("--episode", default=os.path.join(MAIN, "data/real2sim/droid/ep_orange_block"))
    ap.add_argument("--out", default=os.path.join(WT, "runs/taskgen/stage1"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    J = h5py.File(os.path.join(a.episode, "trajectory.h5"))["observation/robot_state/joint_positions"][:]
    import mujoco
    rep = dict(scene=a.scene, episode=a.episode, simulator="mujoco " + mujoco.__version__)

    # 1. base equivalence
    orig = MujocoSim.from_model(mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb")))
    base = MujocoSim()
    rep["base"] = base.load_base(a.scene)
    base.build()
    diffs = equivalence(orig.m, base.m)
    rep["base_equivalence"] = dict(unexplained={k: v for k, v in diffs.items() if k not in KNOWN},
                                   known={k: dict(diff=v, why=KNOWN[k]) for k, v in diffs.items() if k in KNOWN},
                                   replay=replay_equivalence(orig.m, base.m, J))
    zt = rep["base"]["table_z"]

    # 2. modified scene
    drawer = Drawer("drawer_1", (0.72, 0.20))
    centre, _, floor = drawer.inner(zt)
    objs = [Box("red_box_1", (0.03,) * 3, (0.80, 0.10, 0.10, 1), (0.45, 0.10)),
            Box("blue_box_1", (0.025,) * 3, (0.10, 0.25, 0.80, 1), (0.45, 0.10), z_bottom=zt + 0.03, yaw=0.4),
            PackAsset("cup_1", "kitchen/GlassCup002", (0.45, -0.25)),
            drawer,
            Box("purple_box_1", (0.03,) * 3, (0.55, 0.15, 0.65, 1), centre[:2], z_bottom=floor)]
    mod = MujocoSim()
    mod.load_base(a.scene)
    for o in objs:
        mod.add(o)
    info = mod.build()
    rep["added"] = {o.name: dict(describe(o), **info.get(o.name, {})) for o in objs}
    mod.save(os.path.join(a.out, "modified.mjb"))

    # positive control for the overlap check
    ctl = MujocoSim()
    ctl.load_base(a.scene)
    ctl.add(Box("red_box_1", (0.03,) * 3, (0.80, 0.10, 0.10, 1), (0.45, 0.10)))
    ctl.add(Box("control_box", (0.03,) * 3, (0.9, 0.9, 0.9, 1), (0.47, 0.10)))  # centres 2 cm apart: 1 cm overlap
    ctl.build()
    ctl.set_robot(J[0])
    rep["overlap_control"] = dict(expected="control_box x red_box_1, 1.0 cm", found=ctl.overlaps())

    # 3. overlaps, settle, warnings
    for s in (orig, mod):
        s.set_robot(J[0])
    rep["overlaps_at_start"] = dict(original=orig.overlaps(), modified=mod.overlaps())
    rep["settle_1s_moved_m"] = dict(original=orig.settle(1.0), modified=mod.settle(1.0))
    rep["warnings_after_settle"] = dict(original=orig.warnings(), modified=mod.warnings())
    imgs = {"original": {v: orig.render(v) for v in ("ext", "wrist", "free")},
            "modified": {v: mod.render(v) for v in ("ext", "wrist", "free")}}

    # 4. drawer
    nm = drawer.body_names()
    rel0 = mod.position("purple_box_1") - mod.position(nm["drawer"])
    q_start = mod.joint(nm["joint"])
    mod.push_joint(nm["joint"], 3.0, 1.5)
    mod.settle(0.5)
    rel1 = mod.position("purple_box_1") - mod.position(nm["drawer"])
    rep["drawer_close_test"] = dict(qpos_start=q_start, qpos_end=mod.joint(nm["joint"]), closed_qpos=drawer.open_m,
                                    box_shift_in_drawer_m=float(np.linalg.norm(rel1 - rel0)),
                                    box_z_end=float(mod.position("purple_box_1")[2]), drawer_floor_z=floor)
    imgs["modified_closed"] = {v: mod.render(v) for v in ("ext", "free")}

    # 5. outputs
    for sc, vs in imgs.items():
        for v, img in vs.items():
            cv2.imwrite(os.path.join(a.out, f"{sc}_{v}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    rows = [np.concatenate([label(imgs["original"][v], f"original - {v}"), label(imgs["modified"][v], f"modified - {v}")], 1)
            for v in ("ext", "wrist", "free")]
    cv2.imwrite(os.path.join(a.out, "compare.png"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    rep["seconds"] = round(time.time() - t0, 1)
    json.dump(rep, open(os.path.join(a.out, "stage1_report.json"), "w"), indent=1, default=float)
    cm = lambda dd: {k: round(100 * v, 2) for k, v in dd.items()}  # noqa: E731
    print(json.dumps(dict(base_unexplained=rep["base_equivalence"]["unexplained"], replay=rep["base_equivalence"]["replay"],
                          overlaps=rep["overlaps_at_start"], control=rep["overlap_control"]["found"],
                          moved_cm={k: cm(v) for k, v in rep["settle_1s_moved_m"].items()},
                          warnings=rep["warnings_after_settle"], drawer=rep["drawer_close_test"], seconds=rep["seconds"]), indent=1, default=float))


if __name__ == "__main__":
    main()
