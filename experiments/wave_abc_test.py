"""Verification for Wave A/B/C. Each check maps to a design finding."""
import sys, os, shutil, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vla_harness.schema import (PerturbationSpec, TraceStore, ArmLog,
                                freeze_regression_set, RegressionSet)
from vla_harness.runner import (rollout, run_cell, counterfactual_probe,
                                uniform_frequency, _cell_id)
from vla_harness.envs.toy import ToyReachEnv
from vla_harness.policies.scripted import ScriptedReachPolicy
from vla_harness.policies.privileged import PrivilegedProbePolicy, gt_object_pose
from vla_harness.mining.classify import classify

BAR = "=" * 74
SEEDS = list(range(12))
shutil.rmtree("runs/_wave", ignore_errors=True)
ok = {}

print(BAR, "\n  A2/DG-4 — semantic_runtime is KEYED; runtime is not\n", BAR)
env, pol = ToyReachEnv(), ScriptedReachPolicy()
spec = PerturbationSpec.of()
base = _cell_id(env, pol, 0, spec)
class SimEnv(ToyReachEnv):
    def semantic_deps(self): return {"mujoco": "3.3.7", "MUJOCO_GL": "egl"}
class SimEnv2(ToyReachEnv):
    def semantic_deps(self): return {"mujoco": "3.8.1", "MUJOCO_GL": "egl"}
a, b = _cell_id(SimEnv(), pol, 0, spec), _cell_id(SimEnv2(), pol, 0, spec)
print(f"  toy (no sim deps)       {base}")
print(f"  mujoco 3.3.7            {a}")
print(f"  mujoco 3.8.1            {b}   <- MUST differ (F2)")
ok["A2"] = a != b and base != a
print(f"  -> {'PASS' if ok['A2'] else 'FAIL'}")

print(f"\n{BAR}\n  A3/DG-8 — scene_descriptor in physical units\n{BAR}")
r = rollout(env, pol, 3, PerturbationSpec.of(camera_yaw_deg=12))
sd = r.scene_descriptor
ok["A3"] = sd.get("units") == "metres" and sd["objects"][0]["pos_m"][0] != 0
print(f"  units={sd.get('units')}  target={sd['objects'][0]['pos_m'][:2]}  "
      f"camera_yaw={sd['camera']['yaw_deg']}")
print(f"  -> {'PASS' if ok['A3'] else 'FAIL'}")

print(f"\n{BAR}\n  A1/DG-2 — adaptive-first cell still counts for the uniform arm\n{BAR}")
store, arms = TraceStore("runs", "_wave"), ArmLog("runs", "_wave")
cellspec = PerturbationSpec.of(camera_yaw_deg=6)
run_cell(env, pol, cellspec, SEEDS, store, arms=arms, arm="adaptive")   # adaptive FIRST
run_cell(env, pol, cellspec, SEEDS, store, arms=arms, arm="uniform")    # uniform hits cache
freq = uniform_frequency(store.load(), arms, "uniform")
print(f"  adaptive ran the cell first; uniform then requested the same cell")
print(f"  uniform requested={freq['n_requested']}  resolved={freq['n_resolved']}"
      f"  (would be 0 if we filtered on stored tags)")
ok["A1"] = freq["n_resolved"] == len(SEEDS)
print(f"  -> {'PASS' if ok['A1'] else 'FAIL'}")

print(f"\n{BAR}\n  B4/DG-7 — regression set detects env identity drift\n{BAR}")
rs = freeze_regression_set("regset-001", env, [rollout(env, pol, s, cellspec)
                                               for s in SEEDS[:3]])
same = rs.check_against(ToyReachEnv())
drift = rs.check_against(ToyReachEnv(grasp_radius=0.001))
print(f"  same env       -> drift fields: {sorted(same) or 'none'}")
print(f"  grasp_radius   -> drift fields: {sorted(drift)}")
ok["B4"] = not same and "env.grasp_radius" in drift
print(f"  -> {'PASS' if ok['B4'] else 'FAIL'}")

print(f"\n{BAR}\n  B6/DG-6 — privileged probe runs discriminator 4, and is flagged\n{BAR}")
bugged = ScriptedReachPolicy(bugs=["camera_misalignment"])
yaw = PerturbationSpec.of(camera_yaw_deg=15)
normal = run_cell(ToyReachEnv(), bugged, yaw, SEEDS)
probe_pol = PrivilegedProbePolicy(bugged, gt_object_pose)
probed = run_cell(ToyReachEnv(), probe_pol, yaw, SEEDS)
pr = rollout(ToyReachEnv(), probe_pol, 0, yaw)
print(f"  normal policy, yaw+15        {normal.rate:.0%}")
print(f"  + ground-truth object pose   {probed.rate:.0%}")
print(f"  verdict: {'PERCEPTUAL (upstream of control)' if probed.rate > normal.rate + .5 else 'control gap'}")
print(f"  rollout stamped privileged={pr.privileged}")
ok["B6"] = probed.rate > normal.rate + 0.5 and pr.privileged
print(f"  -> {'PASS' if ok['B6'] else 'FAIL'}")

print(f"\n{BAR}\n  C7 — paired McNemar on the counterfactual probe\n{BAR}")
combo = PerturbationSpec.of(camera_yaw_deg=15, ee_offset_x_m=0.04)
pb = counterfactual_probe(ToyReachEnv(), bugged, combo, SEEDS)
for p in pb["probes"]:
    m = p["paired"]
    print(f"  revert {p['knob']:<18} delta {p['delta_pp']:+6.1f} pp   "
          f"discordant={m['discordant']:<3} p={m['p_value']}")
ok["C7"] = all("paired" in p for p in pb["probes"])
print(f"  -> {'PASS' if ok['C7'] else 'FAIL'}")

print(f"\n{BAR}\n  C8/DG-10 — failure_cost is emitted\n{BAR}")
costs = {}
for s in SEEDS:
    rr = rollout(ToyReachEnv(), bugged, s, yaw)
    d = classify(rr)
    costs[d["failure_cost"]["cost"]] = costs.get(d["failure_cost"]["cost"], 0) + 1
print(f"  cost distribution over {len(SEEDS)} failed rollouts: {costs}")
ok["C8"] = all("failure_cost" in classify(rollout(ToyReachEnv(), bugged, s, yaw))
               for s in SEEDS[:3])
print(f"  -> {'PASS' if ok['C8'] else 'FAIL'}")

shutil.rmtree("runs/_wave", ignore_errors=True)
print(f"\n{BAR}")
print("  " + "  ".join(f"{k}:{'PASS' if v else 'FAIL'}" for k, v in ok.items()))
print(f"  {'ALL PASS' if all(ok.values()) else 'FAILURES PRESENT'}")
print(BAR)
sys.exit(0 if all(ok.values()) else 1)
