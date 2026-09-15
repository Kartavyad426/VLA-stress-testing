"""END-TO-END pipeline run on real LIBERO, through OUR harness.

Everything real we have produced so far came through `lerobot-eval`, which
stores a success bit and an mp4 and is therefore unmineable by construction.
This drives the policy through our own loop, so the traces carry privileged
simulator state and the mining layer has something to work on.

Stages, each of which has NEVER run on a real VLA:

    control   C9 per-task environment control      (is the task solvable?)
    capture   LeRobotPolicy + LiberoEnv            (mineable traces)
    sweep     camera-yaw ladder, uniform arm       (robustness curve)
    attribute ddmin over co-perturbed factors      (single/subset/multiple)
    mine      LiberoPhaseSegmenter + classify      (phases, families, cost)
    manifest  clusters -> Data Gap Manifest rows

    python3 experiments/e2e.py --tasks 4 --seeds 5 --suite libero_spatial
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla_harness.schema import (PerturbationSpec, TraceStore, ArmLog,
                                freeze_regression_set)
from vla_harness.runner import (rollout, run_cell, sweep, find_boundary,
                                counterfactual_probe, reproducibility_floor,
                                uniform_frequency)
from vla_harness.envs.libero_env import LiberoEnv
from vla_harness.policies.lerobot_policy import LeRobotPolicy
from vla_harness.control import control_per_task
from vla_harness.mining.phases_libero import LiberoPhaseSegmenter
from vla_harness.mining.classify import classify
from vla_harness.mining.cluster import cluster
from vla_harness import manifest

BAR = "=" * 78


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--checkpoint", default="HuggingFaceVLA/smolvla_libero")
    ap.add_argument("--nas", type=int, default=10)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--yaw-levels", default="0,5,10,15,20")
    ap.add_argument("--skip-sweep", action="store_true")
    a = ap.parse_args()

    run_id = a.run_id or f"e2e_{a.suite}_{int(time.time())}"
    seeds = list(range(a.seeds))
    store, arms = TraceStore("runs", run_id), ArmLog("runs", run_id)
    # env_cfg is REQUIRED: LiberoProcessorStep flips both image axes and builds
    # observation.state. Without it the policy sees upside-down images (~0%).
    from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
    env_cfg = LiberoEnvCfg(task=a.suite)
    pol = LeRobotPolicy(a.checkpoint, n_action_steps=a.nas, env_cfg=env_cfg)
    seg = LiberoPhaseSegmenter()
    mk = lambda t: LiberoEnv(suite=a.suite, task_id=t)
    nominal = PerturbationSpec.of()
    report = {"run_id": run_id, "suite": a.suite, "tasks": a.tasks,
              "seeds": a.seeds, "policy": pol.policy_id}
    t0 = time.time()

    # --- 1. C9 control: is each task solvable at all? ----------------------
    print(f"{BAR}\n  1. PER-TASK ENVIRONMENT CONTROL\n{BAR}")
    ctl = control_per_task(mk, pol, range(a.tasks), seeds, nominal, store,
                           threshold=0.30)
    for tid, v in ctl["per_task"].items():
        print(f"   task {tid}: {v['rate']:5.0%}  CI[{v['ci95'][0]:.0%},"
              f"{v['ci95'][1]:.0%}]  solvable={v['solvable']}")
    print(f"   -> {ctl['verdict']}")
    report["control"] = ctl
    live = [t for t in range(a.tasks) if str(t) not in ctl["excluded_tasks"]]
    if not live:
        print("\n   NO SOLVABLE TASKS -- nothing downstream is interpretable.")
        json.dump(report, open(f"runs/{run_id}/report.json", "w"), indent=2)
        return 1

    # --- 2. reproducibility floor ------------------------------------------
    print(f"\n{BAR}\n  2. REPRODUCIBILITY FLOOR (what does zero look like?)\n{BAR}")
    fl = reproducibility_floor(mk(live[0]), pol, nominal, seeds, repeats=2)
    print(f"   rates {fl['rates']} -> floor {fl['floor_pp']:.1f} pp")
    report["floor"] = fl

    # --- 3. sweep + boundary (uniform arm) ---------------------------------
    if not a.skip_sweep:
        levels = [float(x) for x in a.yaw_levels.split(",")]
        print(f"\n{BAR}\n  3. CAMERA-YAW SWEEP (uniform arm)\n{BAR}")
        allc = {}
        for t in live:
            cells = sweep(mk(t), pol, "camera_yaw_deg", levels, seeds,
                          store=store, arms=arms, arm="uniform")
            allc[t] = [{"yaw": lv, "rate": c.rate, "ci": list(c.ci)}
                       for lv, c in zip(levels, cells)]
            b = find_boundary(cells, "camera_yaw_deg")
            print(f"   task {t}: " + "  ".join(f"{lv:g}deg={c.rate:.0%}"
                                               for lv, c in zip(levels, cells)))
            if b:
                print(f"      boundary between {b['lower']:g} and {b['upper']:g} deg")
            allc[f"boundary_{t}"] = b
        report["sweep"] = allc

    # --- 4. attribution: ddmin over co-perturbed factors -------------------
    print(f"\n{BAR}\n  4. ATTRIBUTION (ddmin)\n{BAR}")
    combo = PerturbationSpec.of(camera_yaw_deg=15, ee_offset_x_m=0.03,
                                light_intensity=0.4)
    pr = counterfactual_probe(mk(live[0]), pol, combo, seeds, store,
                              floor_pp=fl["floor_pp"], arms=arms)
    print(f"   full perturbation: {pr['full_rate']:.0%}")
    for x in pr["probes"]:
        m = x["paired"]
        print(f"   revert {x['knob']:<18} {x['reverted_rate']:5.0%}  "
              f"delta {x['delta_pp']:+6.1f}pp  McNemar p={m['p_value']}")
    att = pr.get("attribution")
    print(f"   -> verdict: {att['verdict'] if att else pr.get('attributed_knob')}")
    report["attribution"] = pr

    # --- 5. MINE: the layer that has never seen a real VLA failure ---------
    print(f"\n{BAR}\n  5. MINING REAL TRACES\n{BAR}")
    rs = store.load()
    ok = skipped = 0
    for r in rs:
        r.diagnosis = classify(r, segmenter=seg)
        if r.diagnosis.get("reason", "").startswith("missing"):
            skipped += 1
        else:
            ok += 1
    print(f"   {len(rs)} traces | classified {ok} | detectors abstained {skipped}")
    fams = {}
    for r in rs:
        f = (r.diagnosis or {}).get("family")
        if f:
            fams[f] = fams.get(f, 0) + 1
    for f, n in sorted(fams.items(), key=lambda x: -x[1]):
        print(f"      {f:20} {n}")
    report["mining"] = {"n_traces": len(rs), "classified": ok,
                        "abstained": skipped, "families": fams}

    # --- 6. cluster + manifest ---------------------------------------------
    print(f"\n{BAR}\n  6. CLUSTERS -> DATA GAP MANIFEST\n{BAR}")
    cl = cluster(rs)
    for c in cl[:5]:
        print(f"   {c['family']:20} n={c['count']:<4} knobs={','.join(c['active_knobs']) or '-'}")
    uf = uniform_frequency(rs, arms, "uniform")
    prov = {"schema": manifest.SCHEMA_VERSION, "run_id": run_id,
            "policy": pol.policy_id, "suite": a.suite}
    rows = [manifest.make_row(f"DGM-{i+1:03d}", c, None, pr,
                              1.0 - (fams.get(c["family"], 0) / max(1, len(rs))),
                              len(rs), prov, uniform_ids=uf["member_ids"],
                              total_failures=sum(fams.values()) or 1)
            for i, c in enumerate(cl[:3])]
    os.makedirs(f"runs/{run_id}", exist_ok=True)
    open(f"runs/{run_id}/DATA_GAP_MANIFEST.md", "w").write(manifest.render(rows))
    report["manifest_rows"] = len(rows)

    # regression set, bound to the env identity it was frozen against
    fails = [r for r in rs if not r.success][:20]
    if fails:
        rset = freeze_regression_set(f"{run_id}-regset", mk(live[0]), fails,
                                     note="frozen from e2e failures")
        rset.save(f"runs/{run_id}/regression_set.json")
        report["regression_set"] = len(fails)

    report["wall_s"] = round(time.time() - t0)
    json.dump(report, open(f"runs/{run_id}/report.json", "w"), indent=2, default=str)
    print(f"\n{BAR}\n  DONE in {report['wall_s']}s -> runs/{run_id}/\n{BAR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
