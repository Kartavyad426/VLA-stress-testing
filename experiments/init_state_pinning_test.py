"""Seed N must start from init state N regardless of reset history.

Guards the resume/pairing bug: LeRobot picks the layout from a counter advanced
on every reset, so a resumed cell (or a second arm on the same env object) ran
seeds on different layouts than their rollout ids claimed.

    MUJOCO_GL=egl .venvs/lerobot/bin/python experiments/init_state_pinning_test.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vla_harness.envs.libero_env import LiberoEnv
from vla_harness.schema import PerturbationSpec

NOM = PerturbationSpec.of()

def layout(env):
    sim = env._sim()
    names, _ = env._object_body_names()
    return np.array([sim.data.body_xpos[sim.model.body_name2id(n)] for n in names])

failed = 0
for suite in ("libero_object", "libero_spatial"):
    a = LiberoEnv(suite=suite, task_id=0)
    for sd in (0, 1, 2):
        a.reset(seed=sd, spec=NOM)
    a.reset(seed=3, spec=NOM); after_history = layout(a)
    b = LiberoEnv(suite=suite, task_id=0)
    b.reset(seed=3, spec=NOM); fresh = layout(b)
    b.reset(seed=0, spec=NOM); seed0 = layout(b)
    d_same = float(np.abs(after_history - fresh).max())
    d_diff = float(np.abs(fresh - seed0).max())
    ok1, ok2 = d_same < 1e-4, d_diff > 1e-4
    ok3 = b.scene_descriptor().get("init_state_index") == 0
    print(f"  [{'PASS' if ok1 else 'FAIL'}] {suite}: seed 3 after 3 resets == seed 3 fresh   max diff {d_same:.2e} m")
    print(f"  [{'PASS' if ok2 else 'FAIL'}] {suite}: seed 3 != seed 0 (layouts really differ) max diff {d_diff:.3f} m")
    print(f"  [{'PASS' if ok3 else 'FAIL'}] {suite}: scene_descriptor records init_state_index")
    failed += (not ok1) + (not ok2) + (not ok3)
print("ALL PASS" if not failed else f"{failed} FAILED")
sys.exit(1 if failed else 0)
