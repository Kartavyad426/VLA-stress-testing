"""T-cabinet end to end on res31's v1/orange, without a model: for each drawer level (and two colours), build the
scene from the tools, write the task config, run rule-known (E4), do-nothing (E2) and the sequential expert (E1:
pick from the open drawer -> drop into the bowl), save the configs and a review video per variant.
Exit 1 if any variant fails a check. Writes runs/taskgen/t_cabinet/ in the worktree.

    runs/real2sim/v1_dev/py_triton_first.sh ../VLA-exp-taskgen/experiments/taskgen/tests/test_expert_cabinet.py
Exploratory, UNVERIFIED.
"""
import json, os, sys, time
import h5py

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from sim_mujoco.base import MAIN  # noqa: E402
from sim_mujoco.sim import MujocoSim  # noqa: E402
from sim_mujoco.expert import Expert  # noqa: E402
from sim_mujoco import assets  # noqa: E402
from core.session import Session  # noqa: E402
from core.templates import PassCondition as PC  # noqa: E402
from core.taskconfig import TaskConfig, Check, do_nothing, rule_known, summarise  # noqa: E402
from core import experts  # noqa: E402

WT = os.path.dirname(os.path.dirname(HERE))
SCENE = os.path.join(MAIN, "runs/real2sim/v1/orange")
EPISODE = os.path.join(MAIN, "data/real2sim/droid/ep_orange_block")
OUT = os.path.join(WT, "runs/taskgen/t_cabinet")


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    q0 = h5py.File(os.path.join(EPISODE, "trajectory.h5"))["observation/robot_state/joint_positions"][0]
    rows, fails = [], []
    for level in ("bottom", "middle", "top"):
        for colour in ("red", "blue"):
            vid = f"T-cabinet_{level}_{colour}"
            S = Session(MujocoSim, assets, SCENE, EPISODE, q0)
            setup = [S.add_object("primitive:cabinet", name="cabinet_1", params=dict(levels=3, open_level=level)),
                     S.place("cabinet_1", "on_table"),
                     S.add_object("primitive:box", colour=colour, size_cm=3, name=f"{colour}_block_1"),
                     S.place(f"{colour}_block_1", "inside", f"cabinet_1/{level}")]
            if "placed" not in setup[-1] or "placed" not in setup[1]:
                rows.append(dict(id=vid, setup_failed=setup)); fails.append(vid); continue
            scfg = os.path.join(OUT, f"{vid}_scene_config.json")
            S.cfg.save(scfg)
            task = TaskConfig(id=vid, source="preset", template="T-cabinet", slot_values=dict(level=level, colour=colour),
                              scene_config=scfg, base_version=S.cfg.base_version,
                              instruction=f"take the {colour} block out of the {level} drawer and put it in the black bowl",
                              targets=[f"{colour}_block_1"], pass_all=[PC(type="inside", object=f"{colour}_block_1", target="bowl_1")],
                              start=dict(arm="episode frame 0", articulation=f"cabinet_1 {level} drawer open 12 cm"))
            task.checks["rule_known"] = rule_known(task)
            task.checks["do_nothing"] = do_nothing(task, S)
            ok, detail, ex = experts.run(task, S, Expert, record=True)
            task.checks["expert"] = Check(passed=ok, detail=detail)
            summarise(task)
            ex.save_video(os.path.join(OUT, f"{vid}_expert.mp4"))
            task.save(os.path.join(OUT, f"{vid}_task_config.json"))
            pk = next((x for x in detail["steps"] if x.get("step") == "pick"), {})
            row = dict(id=vid, status=task.status, rule_known=task.checks["rule_known"].passed, do_nothing_ok=task.checks["do_nothing"].passed,
                       expert_ok=ok, lifted_cm=pk.get("lifted_cm"), held=pk.get("held"), aim_err_cm=pk.get("aim_err_cm"),
                       cabinet_at_cm=setup[1].get("at_cm"), block_at_cm=setup[-1].get("at_cm"), warnings=detail["warnings"])
            rows.append(row)
            print(json.dumps(row), flush=True)
            if task.status != "checked":
                fails.append(vid)
    json.dump(dict(rows=rows, fails=fails, seconds=round(time.time() - t0, 1)), open(os.path.join(OUT, "report.json"), "w"), indent=1, default=float)
    print(f"\n{len(rows) - len(fails)}/{len(rows)} variants checked in {time.time() - t0:.0f} s; fails: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
