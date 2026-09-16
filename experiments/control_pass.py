"""C9 environment control across all four suites.

Answers a question the smoke test raised: task 1 scored 0/2 nominally while
task 0 scored 100%. Real, or an adapter problem on that task? At n=2 it cannot
be told. This runs 6 tasks x 5 seeds per suite.

A task failing the control is EXCLUDED from policy conclusions and the
exclusion is reported -- per task, never per suite (F2's lesson).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vla_harness.schema import PerturbationSpec, TraceStore
from vla_harness.control import control_per_task
from vla_harness.envs.libero_env import LiberoEnv
from vla_harness.policies.lerobot_policy import LeRobotPolicy
from lerobot.envs.configs import LiberoEnv as Cfg

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
store = TraceStore("runs", "control_pass")
res = {}
for suite in SUITES:
    pol = LeRobotPolicy("HuggingFaceVLA/smolvla_libero", n_action_steps=10,
                        env_cfg=Cfg(task=suite))
    c = control_per_task(lambda t, s=suite: LiberoEnv(suite=s, task_id=t),
                         pol, range(6), list(range(5)),
                         PerturbationSpec.of(), store, threshold=0.30)
    res[suite] = c
    print(f"=== {suite} ===", flush=True)
    for t, v in c["per_task"].items():
        lo, hi = v["ci95"]
        print(f"   task {t}: {v['rate']:5.0%}  CI[{lo:.0%},{hi:.0%}]  "
              f"solvable={v['solvable']}", flush=True)
    print(f"   excluded: {c['excluded_tasks'] or 'none'}", flush=True)
os.makedirs("runs/control_pass", exist_ok=True)
json.dump(res, open("runs/control_pass/control.json", "w"), indent=2, default=str)
print("\nwrote runs/control_pass/control.json")
