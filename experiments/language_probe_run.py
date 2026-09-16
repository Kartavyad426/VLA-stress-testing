"""Phase G across suites -- does the policy read the instruction at all?

Gates whether `language_grounding` is an admissible taxonomy family. Decided on
TARGET IDENTITY (which object the end-effector approaches), never on success
rate: a policy that reads a new instruction and fails, and one that ignores it
and executes the original target, both give success -> 0.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vla_harness.schema import PerturbationSpec, TraceStore
from vla_harness.envs.libero_env import LiberoEnv
from vla_harness.policies.lerobot_policy import LeRobotPolicy
from vla_harness.probes.language import language_probe, substitute_from_scene
from lerobot.envs.configs import LiberoEnv as Cfg

out = {}
store = TraceStore("runs", "language_probe")
for suite in ["libero_spatial", "libero_object", "libero_goal"]:
    pol = LeRobotPolicy("HuggingFaceVLA/smolvla_libero", n_action_steps=10,
                        env_cfg=Cfg(task=suite))
    for tid in range(3):
        env = LiberoEnv(suite=suite, task_id=tid)
        env.reset(0, PerturbationSpec.of())      # populate the BDDL object list
        r = language_probe(env, pol, list(range(5)),
                           substitute=lambda s, e=env: substitute_from_scene(e, s),
                           store=store)
        key = f"{suite}/task{tid}"
        out[key] = r
        print(f"{key:28} {r.get('verdict','?'):18} "
              f"nominal={r.get('nominal_rate')} blank={r.get('blank_rate')} "
              f"redirect={r.get('redirect_fraction')}", flush=True)
        print(f"   {r.get('reason','')}", flush=True)
os.makedirs("runs/language_probe", exist_ok=True)
json.dump(out, open("runs/language_probe/probe.json", "w"), indent=2, default=str)
verdicts = [v.get("verdict") for v in out.values()]
adm = [v.get("language_family_admissible") for v in out.values()]
print(f"\nverdicts: {verdicts}")
print(f"language_grounding admissible: {adm}")
