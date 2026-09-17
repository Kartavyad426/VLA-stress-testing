"""Every BDDL object of interest must resolve, for every LIBERO task we run.

Guards the instrumentation gaps that silently disabled the miner: articulated
REGIONS on fixtures (fixed earlier in _resolve_body) and table REGIONS that are
MuJoCo sites, not bodies (libero_goal/task5, fixed 2026-09-17).

    MUJOCO_GL=egl .venvs/lerobot/bin/python experiments/object_resolution_test.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vla_harness.envs.libero_env import LiberoEnv
from vla_harness.schema import PerturbationSpec

failed = 0
for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
    for tid in range(10):
        env = LiberoEnv(suite=suite, task_id=tid)
        obs = env.reset(seed=0, spec=PerturbationSpec.of())
        st = obs.state if hasattr(obs, "state") else {}
        names, complete = env._object_body_names()
        kinds = getattr(env, "_object_kinds", {})
        has_dist = "_gt_eef_to_object" in (env._privileged() or {})
        ok = complete and has_dist
        tag = ",".join(f"{n}[{kinds.get(n, '?')}]" for n in names)
        print(f"  [{'PASS' if ok else 'FAIL'}] {suite}/task{tid}: {tag}")
        failed += not ok
print("ALL PASS" if not failed else f"{failed} FAILED")
sys.exit(1 if failed else 0)
