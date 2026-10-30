"""Stage 1 tools on orange, without a model: known-answer cases (docs/TASKGEN_DESIGN.html §12b step 1).
Each case prints its result; failures are collected and reported at the end (exit 1 if any). Writes the final scene
config, a .mjb to open in the viewer and renders to runs/taskgen/stage1_tools/ in the worktree.

    runs/real2sim/v1_dev/py_triton_first.sh ../VLA-exp-taskgen/experiments/taskgen/tests/test_stage1_tools.py
Exploratory, UNVERIFIED.
"""
import json, os, sys, time
import h5py
import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from sim_mujoco.base import MAIN  # noqa: E402
from sim_mujoco.sim import MujocoSim  # noqa: E402
from sim_mujoco import assets  # noqa: E402
from core.session import Session  # noqa: E402

WT = os.path.dirname(os.path.dirname(HERE))
SCENE = os.path.join(MAIN, "runs/real2sim/ep_orange_block_stereo")
EPISODE = os.path.join(MAIN, "data/real2sim/droid/ep_orange_block")
OUT = os.path.join(WT, "runs/taskgen/stage1_tools")
fails, results = [], {}


def case(name, got, ok):
    results[name] = got
    print(("PASS " if ok else "FAIL ") + name + ": " + json.dumps(got, default=float)[:400])
    if not ok:
        fails.append(name)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    q0 = h5py.File(os.path.join(EPISODE, "trajectory.h5"))["observation/robot_state/joint_positions"][0]
    S = Session(MujocoSim, assets, SCENE, EPISODE, q0)

    thr = S.vis_thresholds()
    case("visibility thresholds from the base objects", dict(thresholds=thr, base=S.base_visibility), thr["ext"] >= 20 and thr["wrist_above"] >= 20)
    sc = S.list_scene()
    case("list_scene: 4 locked base objects, bowl has an opening", sc,
         len(sc["base"]) == 4 and all(r["locked"] for r in sc["base"].values()) and "opening_cm" in sc["base"]["black_bowl"])

    r = S.list_assets("cup")
    case("list_assets('cup') finds pack cups", r, "assets" in r and len(r["assets"]) > 0)
    r = S.list_assets("zzqx")
    case("list_assets of an unknown noun is an error with a hint", r, "error" in r)

    r = S.add_object("primitive:box", colour="magenta")
    case("unknown colour refused", r, "error" in r)
    r = S.add_object("primitive:box", colour="red", size_cm=9)
    case("block 9 cm refused (bounds 2-6)", r, "error" in r)

    r = S.add_object("primitive:drawer", name="drawer_1")
    case("add drawer", r, r.get("added") == "drawer_1")
    r = S.place("drawer_1", "on_table")
    case("place drawer on the table (reachable, clear of base objects)", r, "placed" in r)

    r = S.add_object("primitive:box", colour="red", size_cm=3)
    red = r.get("added")
    r = S.fits(red, "inside", "drawer_1")
    case("3 cm block fits inside the drawer (geometry)", r, r.get("fits") is True)
    r = S.place(red, "inside", "drawer_1")
    case("place red block inside the drawer", r, "placed" in r)

    r = S.add_object("primitive:box", colour="yellow", size_cm=6, name="big_block")
    S.cfg.added_objects["big_block"].size_m = (0.10, 0.10, 0.10)  # beyond bounds on purpose: the drawer is ~6.7 cm tall inside
    r = S.fits("big_block", "inside", "drawer_1")
    case("10 cm block does not fit inside the drawer, limit named", r, r.get("fits") is False and "height" in (r.get("reason") or ""))
    r = S.place("big_block", "inside", "drawer_1", allow_resize=True)
    case("10 cm block placed inside the drawer after resizing within bounds", r, "placed" in r and "resized" in r)
    S.remove_object("big_block")

    cups = []
    for _ in range(2):
        r = S.add_object("pack:cup")
        cups.append(r.get("added"))
    for c in cups:
        r = S.place(c, "next_to", "black_bowl", seed=len(results))
        case(f"place {c} next to the bowl", r, "placed" in r)

    r = S.add_object("primitive:box", colour="blue", size_cm=2.5)
    blue = r.get("added")
    r = S.place(blue, "on", "green_block")
    case("blue block on the green base block", r, "placed" in r)

    r = S.place("black_bowl", "inside", "drawer_1")
    case("the bowl into the drawer is refused (locked base object)", r, "infeasible" in r and "locked" in r["infeasible"])
    r = S.remove_object("orange_block")
    case("removing a base object is refused", r, "error" in r)

    # rebuild from the config alone: everything placed must come back without overlaps and stay put
    s = S.build()
    ov = s.overlaps()
    moved = s.settle(1.0)
    case("rebuilt from config: no overlaps, nothing moves > 2 cm, no warnings",
         dict(overlaps=ov, max_moved_cm=round(100 * max(moved.values()), 2), warnings=s.warnings()),
         not ov and max(moved.values()) < 0.02 and not s.warnings())
    # every placed object passed the visibility rule; positive control: close the drawer over the red block
    s2 = S.build()
    p = s2.position(red)
    before = S.visibility(s2, red, p)
    s2.push_joint("drawer_1_slide", 3.0, 1.5)
    s2.settle(0.5)
    after = S.visibility(s2, red, p)  # same arm pose as the open case (25 cm above where the block was)
    case("visibility control: block in the open drawer seen from above; hidden once the drawer is closed",
         dict(open=before, closed=after), before["wrist_above"]["ok"] and "why" not in after["wrist_above"] and not after["wrist_above"]["ok"] and not after["ext"]["ok"])
    S.cfg.checks = dict(overlaps=ov, max_moved_m=max(moved.values()))
    S.cfg.save(os.path.join(OUT, "scene_config.json"))
    s.save(os.path.join(OUT, "scene.mjb"))
    row = np.concatenate([s.render(v) for v in ("ext", "wrist")], 1)
    cv2.imwrite(os.path.join(OUT, "views.png"), cv2.cvtColor(np.concatenate([row, np.concatenate([s.render("free"), np.zeros_like(s.render("free"))], 1)], 0), cv2.COLOR_RGB2BGR))
    json.dump(dict(results=results, fails=fails, seconds=round(time.time() - t0, 1), scene=S.list_scene()),
              open(os.path.join(OUT, "test_report.json"), "w"), indent=1, default=float)
    print(f"\n{len(results) - len(fails)}/{len(results)} passed in {time.time() - t0:.0f} s; fails: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
