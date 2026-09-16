"""INDICATIVE probe: are libero_goal failures wrong-GOAL or right-goal-but-missed?

READ-ONLY. Reads stored traces; imports nothing from the mining layer.

WHY. Two hypotheses survive for the residual deficit, which is concentrated in
libero_goal (ours 68 vs community repros of the same checkpoint at 81/83/87):

  LANGUAGE-CONDITIONED DISAMBIGUATION -- the goal suite holds ONE FIXED SCENE and
    varies only the instruction, so vision cannot identify the task. A policy that
    under-uses language should execute SOME OTHER task's goal in that scene.
    Predicts WRONG-GOAL COMPLETIONS.
  GEOMETRIC IMPRECISION -- predicts RIGHT-GOAL ATTEMPTS THAT MISS: the manipuland
    is carried toward its own destination and not placed accurately.

The two make opposite predictions on the same corpus. This measures which.

WHAT MAKES IT POSSIBLE, AND WHAT LIMITS IT. The adapter records only each task's
own `obj_of_interest`, so no episode carries the whole scene. But the goal suite's
scene is FIXED ACROSS TASKS, so a destination's position can be recovered from a
task where that object IS recorded, at t=0, and reused for every other task.

Recoverable destinations: the stove (task7, static) and the plate (task5 at t=0,
before it is pushed). NOT recoverable: cabinet top, cabinet drawers, wine rack --
those are the fixture/region bodies absent from every trace, which is the same
gap that blocks BDDL re-evaluation.

So this is INDICATIVE, not conclusive:
  * it covers the bowl-manipulating tasks only;
  * "wrong goal" can only be detected toward the stove or the plate;
  * a bowl carried to the cabinet (an unrecoverable destination) reads as
    "elsewhere", not as a wrong-goal completion.
It is therefore biased AGAINST finding wrong-goal completions. A positive result
is meaningful; a null result is weak.

Usage:  .venvs/lerobot/bin/python experiments/goal_wrong_target_probe.py
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla_harness.schema import TraceStore

RUN = "camp_20260916-0015_libero_goal"

# Which object each task manipulates, and where it is supposed to end up.
# Destination key None = not recoverable from any trace (fixture/region body).
TASKS = {
    "libero_goal/task1": ("akita_black_bowl_1_main", "stove",  "put the bowl on the stove"),
    "libero_goal/task3": ("akita_black_bowl_1_main", None,     "open the top drawer and put the bowl inside"),
    "libero_goal/task4": ("akita_black_bowl_1_main", None,     "put the bowl on top of the cabinet"),
    "libero_goal/task6": ("cream_cheese_1_main",     "bowl",   "put the cream cheese in the bowl"),
    "libero_goal/task5": ("plate_1_main",            None,     "push the plate to the front of the stove"),
}
NEAR_M = 0.10      # within this of a destination centre = "at" it
MOVED_M = 0.05     # displaced at least this far from rest = "engaged"


def scene_map(rollouts):
    """Recover fixed-scene destination positions from whichever task records them.

    The goal suite reuses one scene, so a static object's t=0 position in ANY task
    is its position in EVERY task.
    """
    acc = defaultdict(list)
    for r in rollouts:
        st = r.steps[0].obs_state.get("_gt_object_pos", {})
        if r.task_id == "libero_goal/task7" and "flat_stove_1_main" in st:
            acc["stove"].append(st["flat_stove_1_main"])
        if r.task_id == "libero_goal/task5" and "plate_1_main" in st:
            acc["plate"].append(st["plate_1_main"])          # BEFORE it is pushed
        if "akita_black_bowl_1_main" in st:
            acc["bowl_rest"].append(st["akita_black_bowl_1_main"])
    out = {}
    for k, v in acc.items():
        out[k] = [sum(c[i] for c in v) / len(v) for i in range(3)]
        spread = max(math.dist(p, out[k]) for p in v)
        print(f"  {k:10s} n={len(v):3d}  mean=({out[k][0]:+.3f},{out[k][1]:+.3f},"
              f"{out[k][2]:+.3f})  max spread {spread*100:.1f} cm")
    return out


def main():
    rollouts = TraceStore("runs", RUN).load()
    print(f"{len(rollouts)} goal rollouts\n")
    print("SCENE MAP (recovered from t=0 across tasks; the suite reuses one scene)")
    scene = scene_map(rollouts)
    missing = [k for k in ("stove", "plate", "bowl_rest") if k not in scene]
    if missing:
        sys.exit(f"cannot build scene map, missing {missing}")

    print("\nPER-TASK FAILURE BREAKDOWN")
    print("  own      = ended within 10 cm of its OWN destination  -> right goal, missed")
    print("  OTHER    = ended within 10 cm of a DIFFERENT task's destination -> WRONG GOAL")
    print("  unmoved  = never displaced >5 cm from rest -> never engaged")
    print("  elsewhere= moved, but not near any RECOVERABLE destination\n")

    totals = Counter()
    for task in sorted(TASKS):
        obj, dest, instr = TASKS[task]
        rs = [r for r in rollouts if r.task_id == task
              and r.steps[-1].obs_state.get("_gt_object_pos", {}).get(obj)]
        fails = [r for r in rs if not r.success]
        if not rs:
            continue
        cats = Counter()
        for r in fails:
            p0 = r.steps[0].obs_state["_gt_object_pos"][obj]
            p1 = r.steps[-1].obs_state["_gt_object_pos"][obj]
            if math.dist(p0, p1) < MOVED_M:
                cats["unmoved"] += 1
                continue
            near = [k for k in ("stove", "plate", "bowl_rest")
                    if math.dist(p1, scene[k]) < NEAR_M]
            own = dest in near if dest else False
            other = [k for k in near if k != dest and k != "bowl_rest"]
            if own:
                cats["own"] += 1
            elif other:
                cats[f"OTHER:{'+'.join(other)}"] += 1
            else:
                cats["elsewhere"] += 1
        d = f"dest={dest}" if dest else "dest=UNRECOVERABLE"
        print(f"{task}  ({instr})")
        print(f"   {d}   {len(rs)} eps, {len(fails)} failures")
        for k, v in cats.most_common():
            print(f"     {k:24s} {v:3d}")
        print()
        for k, v in cats.items():
            totals[k] += v

    print("TOTALS OVER ALL EVALUABLE GOAL FAILURES")
    n = sum(totals.values())
    for k, v in totals.most_common():
        print(f"   {k:24s} {v:3d}  ({v/n*100:.1f}%)" if n else k)
    wrong = sum(v for k, v in totals.items() if k.startswith("OTHER"))
    print(f"\n   wrong-goal completions detected: {wrong} of {n}")
    print("   NOTE: only stove and plate are recoverable destinations, so this is a")
    print("   LOWER BOUND. A bowl carried to the cabinet reads as 'elsewhere'.")


if __name__ == "__main__":
    main()
