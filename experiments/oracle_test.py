"""ACCEPTANCE TEST — can the miner recover a fault we planted?

This is the gate in PLAN.md section 4. We insert a known bug into a scripted
policy, run the full pipeline, and check the diagnosis against ground truth.
If the miner cannot recover a planted trigger, the mining design is wrong and
we find out now rather than after 3,000 GPU-hours.

    python3 experiments/oracle_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla_harness.schema import PerturbationSpec, TraceStore
from vla_harness.runner import (rollout, run_cell, sweep, find_boundary,
                                counterfactual_probe, resume_summary,
                                format_resume_summary)
from vla_harness.envs.toy import ToyReachEnv
from vla_harness.policies.scripted import ScriptedReachPolicy, GROUND_TRUTH
from vla_harness.mining.classify import classify
from vla_harness.mining.cluster import cluster
from vla_harness import manifest

SEEDS = list(range(20))          # 20 episodes/cell — the doc's screening budget
RUNS = "runs"

# Each planted fault, the knob that should expose it, and the sweep range.
# (label, bugs, swept knob, levels, expected family, expected trigger)
#
# The last two scenarios share a trigger ON PURPOSE. `no_recovery` is a
# second-order fault: it cannot fail on its own, only conditional on some
# other failure happening first. So the honest test is discriminative --
# same trigger, same symptom in aggregate, and the miner must still separate
# "searched and missed" (manipulation) from "repeated itself" (recovery).
SCENARIOS = [
    ("camera_misalignment", ["camera_misalignment"], "camera_yaw_deg",
     [0, 3, 6, 9, 12, 15, 20, 25], "visual_grounding", "camera_yaw_deg"),
    ("nearest_object", ["nearest_object"], "distractor_count",
     [0, 1, 2, 3], "spatial_reasoning", "distractor_count"),
    ("unfiltered_grasp", ["unfiltered_grasp"], "pixel_noise_std",
     [0, .01, .02, .03, .05], "manipulation", "pixel_noise_std"),
    ("unfiltered_grasp + no_recovery", ["unfiltered_grasp", "no_recovery"],
     "pixel_noise_std", [0, .01, .02, .03, .05], "recovery", "pixel_noise_std"),
]
BAR = "=" * 78


def nominal_refs(env, policy):
    """One nominal rollout per seed — the reference for divergence detection."""
    nom = PerturbationSpec.of()
    return {s: rollout(env, policy, s, nom) for s in SEEDS}


def run_scenario(label, bugs, knob, levels, exp_family, exp_knob, store):
    env = ToyReachEnv()
    policy = ScriptedReachPolicy(bugs=bugs)
    bug = label

    print(f"\n{BAR}\n  PLANTED FAULT: {bug}\n"
          f"  ground truth -> family={exp_family}  trigger={exp_knob}\n{BAR}")

    # --- 1. SWEEP: vary one axis, everything else nominal ------------------
    print(f"\n  [1] SWEEP  {knob}   ({len(SEEDS)} episodes per cell)\n")
    cells = sweep(env, policy, knob, levels, SEEDS, store=store)
    for c in cells:
        bar = "#" * int(round(c.rate * 40))
        print(f"      {c}  |{bar}")

    boundary = find_boundary(cells, knob)
    if boundary:
        print(f"\n      boundary: {knob} between {boundary['lower']:g} and "
              f"{boundary['upper']:g}   ({boundary['definition']})")
    else:
        print(f"\n      boundary: not crossed in swept range")

    # --- 2. COUNTERFACTUAL PROBE: revert one knob at a time ----------------
    # Build a multi-knob failing cell so the probe has something to isolate.
    failing = next((c for c in cells if c.rate < 0.5), cells[-1])
    combo = {k: v for k, v in failing.spec.as_dict().items() if v}
    # co-perturb other axes so the probe has genuine alternatives to rule out
    for extra, val in (("camera_yaw_deg", 9), ("ee_offset_x_m", .04),
                       ("distractor_count", 2), ("pixel_noise_std", .02)):
        if extra != knob:
            combo.setdefault(extra, val)
    spec = PerturbationSpec.of(**combo)

    print(f"\n  [2] COUNTERFACTUAL PROBE from: {spec.label()}\n")
    probe = counterfactual_probe(env, policy, spec, SEEDS, store=store)
    print(f"      full perturbation          {probe['full_rate']:6.1%}")
    for p in probe["probes"]:
        print(f"      revert {p['knob']:<22} {p['reverted_rate']:6.1%}"
              f"   delta {p['delta_pp']:+6.1f} pp")
    print(f"\n      attributed trigger -> {probe['attributed_knob']}")

    # --- 3. CLASSIFY + CLUSTER --------------------------------------------
    refs = nominal_refs(env, policy)
    rs = []
    for s in SEEDS:
        r = rollout(env, policy, s, spec)
        r.diagnosis = classify(r, nominal=refs.get(s))
        rs.append(r)
    for c in cells:
        for s in SEEDS:
            r = rollout(env, policy, s, c.spec)
            r.diagnosis = classify(r, nominal=refs.get(s))
            rs.append(r)

    clusters = cluster(rs)
    print(f"\n  [3] CLUSTERS  ({sum(c['count'] for c in clusters)} failures)\n")
    for c in clusters[:4]:
        print(f"      {c['family']:<20} n={c['count']:<4} "
              f"knobs={','.join(c['active_knobs']) or '-':<38} "
              f"err={c['mean_final_error_m']*100:.1f}cm")

    # --- 4. CHECK AGAINST GROUND TRUTH ------------------------------------
    got_family = clusters[0]["family"] if clusters else None
    got_knob = probe["attributed_knob"]
    fam_ok = got_family == exp_family
    knob_ok = (got_knob == exp_knob) if exp_knob else True

    print(f"\n  [4] VERDICT")
    print(f"      family   expected {exp_family:<20} got {str(got_family):<20}"
          f" {'PASS' if fam_ok else 'FAIL'}")
    print(f"      trigger  expected {str(exp_knob):<20} got {str(got_knob):<20}"
          f" {'PASS' if knob_ok else 'FAIL'}")

    return {"bug": bug, "family_ok": fam_ok, "trigger_ok": knob_ok,
            "clusters": clusters, "boundary": boundary, "probe": probe,
            "cells": cells, "n": len(SEEDS) * (len(levels) + 1)}


def main():
    store = TraceStore(RUNS, "oracle_test")
    print(f"{BAR}\n  ORACLE ACCEPTANCE TEST — mining layer vs planted faults\n"
          f"{BAR}\n  Gate (PLAN.md §4): recover >=3 of 4 planted triggers.")

    results = [run_scenario(*s, store) for s in SCENARIOS]

    # Resume accounting belongs in the summary, not a log line: a stale cache
    # and a real finding are indistinguishable from the outside.
    print(f"\n{BAR}\n  RESUME / CACHE\n{BAR}\n")
    print(format_resume_summary(
        resume_summary([c for r in results for c in r["cells"]])))

    # --- sanity check: a clean policy must NOT generate manifest rows ------
    print(f"\n{BAR}\n  CONTROL: clean policy under the same perturbations\n{BAR}")
    env, clean = ToyReachEnv(), ScriptedReachPolicy(bugs=[])
    spec = PerturbationSpec.of(camera_yaw_deg=15, ee_offset_x_m=.04,
                               distractor_count=2)
    c = run_cell(env, clean, spec, SEEDS)
    print(f"\n      {c}")
    control_ok = c.rate > 0.9
    print(f"      clean policy survives perturbation: "
          f"{'PASS' if control_ok else 'FAIL'}  "
          f"(a miner that flags a healthy policy is useless)")

    # --- manifest from the strongest scenario ------------------------------
    best = results[0]
    prov = {"schema": manifest.SCHEMA_VERSION, "harness": "prototype",
            "policy": "scripted-reach-v1[camera_misalignment]",
            "env": "toy-reach-v1"}
    total = sum(c["count"] for c in best["clusters"])
    rows = [manifest.make_row(
        f"DGM-{i+1:03d}", cl, best["boundary"] if i == 0 else None,
        best["probe"] if i == 0 else None,
        best["cells"][0].rate, best["n"], prov,
        supply_side=("training demos cluster within +/-8 deg yaw "
                     "(Phase 0 — PLACEHOLDER, not yet measured)")
                     if i == 0 else None,
        total_failures=total)
        for i, cl in enumerate(best["clusters"][:2])]

    md = manifest.render(rows)
    out = os.path.join(RUNS, "oracle_test", "DATA_GAP_MANIFEST.md")
    with open(out, "w") as f:
        f.write(md)

    print(f"\n{BAR}\n  DATA GAP MANIFEST (generated)\n{BAR}\n")
    print(md)

    passed = sum(r["family_ok"] and r["trigger_ok"] for r in results)
    print(BAR)
    print(f"  GATE: {passed}/4 planted faults recovered  |  control "
          f"{'PASS' if control_ok else 'FAIL'}")
    print(f"  {'>>> GATE PASSED' if passed >= 3 and control_ok else '>>> GATE FAILED — redesign the miner before proceeding'}")
    print(f"  manifest written to {out}")
    print(BAR)
    return 0 if (passed >= 3 and control_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
