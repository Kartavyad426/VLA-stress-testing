"""T-cabinet scene setup on res31's v1/orange, without a model: a 3-level cabinet with one drawer open, a block placed
in the open drawer's exposed part, closed drawers refused, the block rides the drawer, renders per open level.
Known-answer cases; exit 1 on any failure. Writes runs/taskgen/cabinet/ in the worktree.

    runs/real2sim/v1_dev/py_triton_first.sh ../VLA-exp-taskgen/experiments/taskgen/tests/test_cabinet.py
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
SCENE = os.path.join(MAIN, "runs/real2sim/v1/orange")
EPISODE = os.path.join(MAIN, "data/real2sim/droid/ep_orange_block")
OUT = os.path.join(WT, "runs/taskgen/cabinet")
fails, results = [], {}


def case(name, got, ok):
    results[name] = got
    print(("PASS " if ok else "FAIL ") + name + ": " + json.dumps(got, default=float)[:400], flush=True)
    if not ok:
        fails.append(name)


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    q0 = h5py.File(os.path.join(EPISODE, "trajectory.h5"))["observation/robot_state/joint_positions"][0]
    rows = []
    for lv in ("bottom", "middle", "top"):
        S = Session(MujocoSim, assets, SCENE, EPISODE, q0)
        r = S.add_object("primitive:cabinet", name="cabinet_1", params=dict(levels=3, open_level=lv))
        case(f"[{lv}] add a 3-level cabinet, {lv} drawer open", r, r.get("added") == "cabinet_1")
        r = S.place("cabinet_1", "on_table")
        case(f"[{lv}] place the cabinet on the table (clear, reachable into the open drawer, visible)", r, "placed" in r)
        if "placed" not in r:
            continue
        sc = S.list_scene()["added"]["cabinet_1"]
        case(f"[{lv}] list_scene: one level open, two closed", sc, list(sc["levels"].values()).count("closed") == 2)
        r = S.add_object("primitive:box", colour="red", size_cm=3, name="red_block_1")
        r = S.place("red_block_1", "inside", f"cabinet_1/{lv}")
        case(f"[{lv}] red 3 cm block into the open {lv} drawer", r, "placed" in r)
        closed = next(x for x in ("bottom", "middle", "top") if x != lv)
        S.add_object("primitive:box", colour="blue", size_cm=3, name="blue_block_1")
        r = S.place("blue_block_1", "inside", f"cabinet_1/{closed}")
        case(f"[{lv}] blue block into the closed {closed} drawer is refused", r, "infeasible" in r and "closed" in r["infeasible"])
        S.remove_object("blue_block_1")
        r = S.place("red_block_1", "inside", "cabinet_1/attic")
        case(f"[{lv}] unknown level is an error", r, "error" in r)
        S.place("red_block_1", "inside", f"cabinet_1/{lv}")
        s = S.build()
        ov, moved = s.overlaps(), s.settle(1.0)
        case(f"[{lv}] rebuilt from the config: no overlaps, nothing moves > 2 cm, no warnings",
             dict(overlaps=ov, max_moved_cm=round(100 * max(moved.values()), 2), warnings=s.warnings()),
             not ov and max(moved.values()) < 0.02 and not s.warnings())
        imgs = [label(s.render(v), f"{lv} open - {v}") for v in ("ext", "wrist", "free")]
        # the block rides the drawer when it is pushed closed
        j = f"cabinet_1_{lv}_slide"
        rel0 = s.position("red_block_1") - s.position(f"cabinet_1_{lv}_drawer")
        s.push_joint(j, 3.0, 1.5)
        s.settle(0.5)
        rel1 = s.position("red_block_1") - s.position(f"cabinet_1_{lv}_drawer")
        case(f"[{lv}] drawer pushed closed: closes, block rides inside", dict(slide_end=round(s.joint(j), 3), slip_cm=round(100 * float(np.linalg.norm(rel1 - rel0)), 2)),
             s.joint(j) > 0.11 and np.linalg.norm(rel1 - rel0) < 0.01)
        rows.append(np.concatenate(imgs, 1))
        S.cfg.save(os.path.join(OUT, f"scene_config_{lv}.json"))
        s2 = S.build()
        s2.save(os.path.join(OUT, f"scene_{lv}.mjb"))
    if rows:
        cv2.imwrite(os.path.join(OUT, "views.png"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    json.dump(dict(results=results, fails=fails, seconds=round(time.time() - t0, 1)), open(os.path.join(OUT, "test_report.json"), "w"), indent=1, default=float)
    print(f"\n{len(results) - len(fails)}/{len(results)} passed in {time.time() - t0:.0f} s; fails: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
