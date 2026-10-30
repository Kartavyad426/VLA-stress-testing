"""Policy runner on CPU with the hold policy (no GPU): a T-cabinet scene built on res31's ep_orange_block_stereo (the
build that has the composite assets), its task config, a short episode; checks the pipeline end to end (composite
images, state, success by TaskGen rules: hold must NOT succeed, stage scores). Writes runs/taskgen/runner_hold/.

    runs/real2sim/v1_dev/py_triton_first.sh ../VLA-exp-taskgen/experiments/taskgen/tests/test_runner_hold.py
Exploratory, UNVERIFIED.
"""
import json, os, sys, time
import h5py

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from sim_mujoco.base import MAIN  # noqa: E402
from sim_mujoco.sim import MujocoSim  # noqa: E402
from sim_mujoco import assets  # noqa: E402
from sim_mujoco.policy_runner import run_episode, need_assets  # noqa: E402
from core.session import Session  # noqa: E402
from core.templates import PassCondition as PC  # noqa: E402
from core.taskconfig import TaskConfig  # noqa: E402

WT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(WT, "runs/taskgen/runner_hold")
SCENE = os.path.join(MAIN, "runs/real2sim/ep_orange_block_stereo")
EP = os.path.join(MAIN, "data/real2sim/droid/ep_orange_block")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("assets missing on v1/orange:", need_assets(os.path.join(MAIN, "runs/real2sim/v1/orange"), "v2"), flush=True)
    print("assets missing on ep_orange_block_stereo:", need_assets(SCENE, "v2"), flush=True)
    q0 = h5py.File(os.path.join(EP, "trajectory.h5"))["observation/robot_state/joint_positions"][0]
    S = Session(MujocoSim, assets, SCENE, EP, q0)
    print(S.add_object("primitive:cabinet", name="cabinet_1", params=dict(levels=3, open_level="middle")))
    print(S.place("cabinet_1", "on_table"))
    print(S.add_object("primitive:box", colour="red", size_cm=3, name="red_block_1"))
    print(S.place("red_block_1", "inside", "cabinet_1/middle"))
    scfg = os.path.join(OUT, "scene_config.json")
    S.cfg.save(scfg)
    task = TaskConfig(id="T-cabinet_middle_red_stereo", source="preset", template="T-cabinet", scene_config=scfg, base_version=S.cfg.base_version,
                      instruction="take the red block out of the middle drawer and put it in the black bowl", targets=["red_block_1"],
                      pass_all=[PC(type="inside", object="red_block_1", target="black_bowl")])
    task.save(os.path.join(OUT, "task_config.json"))
    t0 = time.time()
    res = run_episode(S, task, OUT, policy="hold", seconds=4.0, log=lambda x: print(x, flush=True))
    print(json.dumps(dict(success=res["success"], steps=res["steps"], policy_calls=res["policy_calls"], stages=res.get("stages"),
                          wall_s=round(res["wall_s"], 1), total_s=round(time.time() - t0, 1)), default=str))
    ok = (res["success"] is False) and res["steps"] == 60
    print("PASS" if ok else "FAIL", "hold policy: pipeline runs, task not solved by holding still")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
