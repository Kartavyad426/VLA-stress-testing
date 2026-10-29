"""Success rules and the do-nothing check on hand-set states (res31's v1/orange + a 3-level cabinet), without a model
or an expert: objects are moved by set_pose, settled, and every rule must read the state correctly both ways.
Exit 1 on any failure. Writes runs/taskgen/rules/test_report.json in the worktree.

    runs/real2sim/v1_dev/py_triton_first.sh ../VLA-exp-taskgen/experiments/taskgen/tests/test_rules.py
Exploratory, UNVERIFIED.
"""
import json, os, sys, time
import h5py

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from sim_mujoco.base import MAIN  # noqa: E402
from sim_mujoco.sim import MujocoSim  # noqa: E402
from sim_mujoco import assets  # noqa: E402
from core.session import Session  # noqa: E402
from core.templates import PassCondition as PC  # noqa: E402
from core.taskconfig import TaskConfig, do_nothing, rule_known  # noqa: E402
from core.geometry import aabb, cavity  # noqa: E402
from core import rules  # noqa: E402

WT = os.path.dirname(os.path.dirname(HERE))
SCENE = os.path.join(MAIN, "runs/real2sim/v1/orange")
EPISODE = os.path.join(MAIN, "data/real2sim/droid/ep_orange_block")
OUT = os.path.join(WT, "runs/taskgen/rules")
fails, results = [], {}


def case(name, got, ok):
    results[name] = got
    print(("PASS " if ok else "FAIL ") + name + ": " + json.dumps(got, default=float)[:300], flush=True)
    if not ok:
        fails.append(name)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    q0 = h5py.File(os.path.join(EPISODE, "trajectory.h5"))["observation/robot_state/joint_positions"][0]
    S = Session(MujocoSim, assets, SCENE, EPISODE, q0)
    S.add_object("primitive:cabinet", name="cabinet_1", params=dict(levels=3, open_level="middle"))
    S.place("cabinet_1", "on_table")
    S.add_object("primitive:box", colour="red", size_cm=3, name="red_block_1")
    S.place("red_block_1", "inside", "cabinet_1/middle")
    blocks = ["block_1", "block_2", "block_3"]  # v1/orange: yellow, orange, green (scene_fit rgb)

    def fresh():
        s = S.build()
        s.settle(0.5)
        return s

    def ev(s, **kw):
        return rules.evaluate(PC(**kw), S, s)

    s = fresh()
    case("start: red block inside the middle drawer", ev(s, type="inside", object="red_block_1", target="cabinet_1/middle"),
         ev(s, type="inside", object="red_block_1", target="cabinet_1/middle")[0])
    case("start: red block not inside the bowl", ev(s, type="inside", object="red_block_1", target="bowl_1"),
         not ev(s, type="inside", object="red_block_1", target="bowl_1")[0])
    case("start: block_2 not inside the bowl; out_of the bowl true", [ev(s, type="inside", object="block_2", target="bowl_1"),
         ev(s, type="out_of", object="block_2", target="bowl_1")],
         not ev(s, type="inside", object="block_2", target="bowl_1")[0] and ev(s, type="out_of", object="block_2", target="bowl_1")[0])

    # move the red block into the bowl by hand
    cv = cavity(s.collision_points("bowl_1"))
    s.set_pose("red_block_1", cv["centre_xy"][0], cv["centre_xy"][1], cv["floor_z"] + 0.002, 0.0)
    s.settle(1.0)
    r1, r2 = ev(s, type="inside", object="red_block_1", target="bowl_1"), ev(s, type="out_of", object="red_block_1", target="cabinet_1/middle")
    case("red block set into the bowl: inside(bowl) and out_of(drawer)", [r1, r2], r1[0] and r2[0])

    # stack by hand: block_3 on block_1, block_2 on block_3 (order block_1, block_3, block_2)
    s = fresh()
    b1 = aabb(s.collision_points("block_1"))
    s.set_pose("block_3", b1["centre"][0], b1["centre"][1], b1["hi"][2] + 0.002, 0.0)
    s.settle(1.0)
    b3 = aabb(s.collision_points("block_3"))
    s.set_pose("block_2", b3["centre"][0], b3["centre"][1], b3["hi"][2] + 0.002, 0.0)
    s.settle(1.5)
    good = ev(s, type="ordered_stack", order=["block_1", "block_3", "block_2"])
    bad = ev(s, type="ordered_stack", order=["block_1", "block_2", "block_3"])
    case("hand-built stack: the right order passes, a wrong order fails", dict(good=good, bad=bad), good[0] and not bad[0])
    case("on(block_3, block_1) true; on(block_1, block_3) false", [ev(s, type="on", object="block_3", target="block_1"),
         ev(s, type="on", object="block_1", target="block_3")],
         ev(s, type="on", object="block_3", target="block_1")[0] and not ev(s, type="on", object="block_1", target="block_3")[0])

    # next_to
    s = fresh()
    b1 = aabb(s.collision_points("block_1"))
    s.set_pose("block_3", b1["hi"][0] + 0.03, b1["centre"][1], S.table_z, 0.0)
    s.settle(1.0)
    near = ev(s, type="next_to", object="block_3", target="block_1")
    s.set_pose("block_3", b1["hi"][0] + 0.15, b1["centre"][1], S.table_z, 0.0)
    s.settle(1.0)
    far = ev(s, type="next_to", object="block_3", target="block_1")
    case("next_to: 3 cm away true, 15 cm away false", dict(near=near, far=far), near[0] and not far[0])

    # task config: rule known + do-nothing (E2) on the T-cabinet task
    task = TaskConfig(id="T-cabinet_middle_red", source="preset", template="T-cabinet", slot_values=dict(level="middle", colour="red"),
                      scene_config="(in memory)", base_version=S.cfg.base_version,
                      instruction="take the red block out of the middle drawer and put it in the black bowl", targets=["red_block_1"],
                      pass_all=[PC(type="inside", object="red_block_1", target="bowl_1")], time_limit_s=30.0)
    rk = rule_known(task)
    case("task config: rule known (E4)", rk.model_dump(), rk.passed)
    dn = do_nothing(task, S)
    case("do-nothing for 30 s does not solve it (E2)", dict(passed=dn.passed, solved=dn.detail["solved_by_doing_nothing"], warnings=dn.detail["warnings"]), dn.passed)
    trivial = TaskConfig(**{**task.model_dump(), "id": "trivial", "pass_all": [PC(type="inside", object="red_block_1", target="cabinet_1/middle")]})
    dn2 = do_nothing(trivial, S, step_s=5.0)
    case("positive control: a task already true at the start IS caught by do-nothing", dict(passed=dn2.passed), dn2.passed is False)

    json.dump(dict(results=results, fails=fails, seconds=round(time.time() - t0, 1)), open(os.path.join(OUT, "test_report.json"), "w"), indent=1, default=float)
    print(f"\n{len(results) - len(fails)}/{len(results)} passed in {time.time() - t0:.0f} s; fails: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
