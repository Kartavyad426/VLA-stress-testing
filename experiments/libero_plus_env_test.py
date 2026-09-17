"""LIBERO-Plus through our env adapter: every perturbation type, lowest and highest level.

    PYTHONPATH=third_party/LIBERO-plus LIBERO_CONFIG_PATH=third_party/libero-plus-config \
    MUJOCO_GL=egl .venvs/libero-plus/bin/python experiments/libero_plus_env_test.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vla_harness.envs.libero_env import LiberoEnv
from vla_harness.schema import PerturbationSpec

SUITE = "libero_spatial"
cls = json.load(open("third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"))[SUITE]
cases = []
for cat in sorted({r["category"] for r in cls}):
    levels = sorted({r["difficulty_level"] for r in cls if r["category"] == cat})
    for lv in (levels[0], levels[-1]):
        r = next(r for r in cls if r["category"] == cat and r["difficulty_level"] == lv)
        cases.append(r)

failed = 0
for r in cases:
    tid = r["id"] - 1
    env = LiberoEnv(suite=SUITE, task_id=tid, libero_plus=True)
    env.reset(seed=0, spec=PerturbationSpec.of())
    names, complete = env._object_body_names()
    priv = env._privileged() or {}
    sd = env.scene_descriptor()
    lp = sd.get("libero_plus") or {}
    ident = env.identity()
    checks = {
        "objects complete": complete and bool(names),
        "distances recorded": "_gt_eef_to_object" in priv,
        "category matches": lp.get("category") == r["category"],
        "level matches": lp.get("difficulty_level") == r["difficulty_level"],
        "catalogue id = task_id+1": lp.get("catalogue_id") == tid + 1,
        "identity says libero_plus": ident.get("benchmark") == "libero_plus",
        "init_state_index pinned": sd.get("init_state_index") == 0,
    }
    ok = all(checks.values())
    failed += not ok
    bad = [k for k, v in checks.items() if not v]
    print(f"  [{'PASS' if ok else 'FAIL'}] task_id {tid:4d} {r['category']:22s} L{r['difficulty_level']}  "
          f"objects={len(names)}" + (f"  FAILED: {bad}" if bad else ""))
print("ALL PASS" if not failed else f"{failed} FAILED")
sys.exit(1 if failed else 0)
