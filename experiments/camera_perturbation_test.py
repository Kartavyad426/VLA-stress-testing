"""Render test for LiberoEnv camera perturbations (#22).

The bug this guards against: "yaw 1e-6" rendered like "yaw 5" because the code
re-aimed the camera instead of rotating it, and reset() returned the frame from
BEFORE the perturbation. Needs the LeRobot venv and a GPU (or osmesa).

    MUJOCO_GL=egl .venvs/lerobot/bin/python experiments/camera_perturbation_test.py
"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vla_harness.envs.libero_env import LiberoEnv, MAIN_CAMERA, _quat_rotate
from vla_harness.schema import PerturbationSpec, Action

DUMMY = Action(values=[0, 0, 0, 0, 0, 0, -1],
               dims=["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"])


def pose(env):
    sim = env._sim(); cid = sim.model.camera_name2id(MAIN_CAMERA)
    p = np.array(sim.model.cam_pos[cid], float)
    f = _quat_rotate(np.array(sim.model.cam_quat[cid], float), [0, 0, -1])
    return p, f


def run(suite, **knobs):
    env = LiberoEnv(suite=suite, task_id=0)
    env.reset(seed=0, spec=PerturbationSpec.of(**knobs))
    reset_img = np.asarray(list(env._raw["pixels"].values())[0], float)
    p, f = pose(env)
    env.step(DUMMY)
    step_img = np.asarray(list(env._raw["pixels"].values())[0], float)
    return {"pos": p, "fwd": f, "reset_img": reset_img, "step_img": step_img}


def ang(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1, 1))))


def heading(f):
    return math.degrees(math.atan2(f[1], f[0]))


def pitch_down(f):
    return math.degrees(math.asin(-f[2]))


failed = 0
def check(name, ok, detail):
    global failed
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:52s} {detail}")
    failed += not ok


for suite in ("libero_object", "libero_spatial"):
    print(f"\n=== {suite} task0 ===")
    c = run(suite, camera_yaw_deg=0.0)
    tiny = run(suite, camera_yaw_deg=1e-6)
    y5 = run(suite, camera_yaw_deg=5.0)
    p5 = run(suite, camera_pitch_deg=5.0)
    d10 = run(suite, camera_dist_m=0.10)

    d = np.abs(tiny["step_img"] - c["step_img"]).mean()
    check("yaw 1e-6 renders like yaw 0", d < 1.0, f"mean abs pixel diff {d:.3f} (buggy: 44-53)")
    check("yaw 1e-6 leaves the view axis alone", ang(tiny["fwd"], c["fwd"]) < 0.01,
          f"{ang(tiny['fwd'], c['fwd']):.4f} deg (buggy: ~12)")

    dh = heading(y5["fwd"]) - heading(c["fwd"])
    dp = pitch_down(y5["fwd"]) - pitch_down(c["fwd"])
    check("yaw 5 turns the heading by 5 deg", abs(dh - 5.0) < 0.05, f"heading change {dh:+.3f} deg")
    check("yaw 5 does not change pitch", abs(dp) < 0.05, f"pitch change {dp:+.3f} deg")

    dpp = pitch_down(p5["fwd"]) - pitch_down(c["fwd"])
    check("pitch 5 looks 5 deg more steeply down", abs(dpp - 5.0) < 0.05, f"pitch-down change {dpp:+.3f} deg")
    check("pitch 5 raises the camera", p5["pos"][2] > c["pos"][2],
          f"z {c['pos'][2]:.3f} -> {p5['pos'][2]:.3f}")
    check("pitch 5 keeps the heading", abs(heading(p5["fwd"]) - heading(c["fwd"])) < 0.05,
          f"heading change {heading(p5['fwd']) - heading(c['fwd']):+.3f} deg")

    check("dist +0.10 keeps the view axis", ang(d10["fwd"], c["fwd"]) < 0.01,
          f"{ang(d10['fwd'], c['fwd']):.4f} deg")
    moved = np.dot(d10["pos"] - c["pos"], -c["fwd"])
    check("dist +0.10 backs the camera away 0.10 m", abs(moved - 0.10) < 0.002, f"{moved:+.4f} m along -view")

    stale = np.abs(y5["reset_img"] - y5["step_img"]).mean()
    check("reset() returns the POST-perturbation frame", stale < 5.0,
          f"reset vs step-1 frame diff {stale:.2f} (buggy: 44-53)")

print(f"\n{'ALL PASS' if not failed else f'{failed} FAILED'}")
sys.exit(1 if failed else 0)
