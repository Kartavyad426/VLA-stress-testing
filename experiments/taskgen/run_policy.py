"""Run a policy on a TaskGen task: scene config + task config + policy -> one episode (sim_mujoco/policy_runner.py).

    runs/real2sim/v1_dev/py_triton_first.sh ../VLA-exp-taskgen/experiments/taskgen/run_policy.py \
        --scene_config <scene_config.json> --task_config <task_config.json> --policy hold --out <dir>
--policy hold / replay run on CPU; --policy groot needs the GPU (prim's gate) and must not run under the CPU wrapper's
CUDA_VISIBLE_DEVICES="" (use MUJOCO_GL=egl and .venvs/real2sim/bin/python directly). Exploratory, UNVERIFIED.
"""
import argparse, json, os, sys
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim_mujoco.base import MAIN  # noqa: E402
from sim_mujoco.sim import MujocoSim  # noqa: E402
from sim_mujoco import assets  # noqa: E402
from sim_mujoco.policy_runner import run_episode  # noqa: E402
from core.session import Session  # noqa: E402
from core.config import SceneConfig  # noqa: E402
from core.taskconfig import TaskConfig  # noqa: E402


def session_from(scene_config):
    cfg = SceneConfig.load(scene_config)
    q0 = h5py.File(os.path.join(cfg.episode, "trajectory.h5"))["observation/robot_state/joint_positions"][0]
    S = Session(MujocoSim, assets, cfg.base_scene, cfg.episode, q0)
    if S.cfg.base_version != cfg.base_version:
        print(f"WARNING: base scene changed since the config was made ({cfg.base_version} -> {S.cfg.base_version})", flush=True)
    S.cfg = cfg
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_config", required=True)
    ap.add_argument("--task_config", required=True)
    ap.add_argument("--policy", choices=["hold", "replay", "groot", "expert"], default="hold")
    ap.add_argument("--replay_npz", default=None)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--wrist_render", choices=["v1", "v2"], default="v2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = session_from(a.scene_config)
    task = TaskConfig(**json.load(open(a.task_config)))
    res = run_episode(S, task, a.out, policy=a.policy, seconds=a.seconds, wrist_render=a.wrist_render, replay_npz=a.replay_npz,
                      log=lambda x: print(x, flush=True))
    print(json.dumps({k: res.get(k) for k in ("success", "steps", "policy_calls", "wall_s", "stages", "gpu_used_max_gib")}, default=str), flush=True)


if __name__ == "__main__":
    main()
