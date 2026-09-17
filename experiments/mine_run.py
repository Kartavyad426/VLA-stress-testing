"""Mine an existing harness run offline: phases + family classification.

    python3 experiments/mine_run.py runs/lplus_fail_groot [runs/... ...]

Writes <run>/diagnoses.jsonl (one row per rollout: ids, success, LIBERO-Plus
category/level, predicates, families) and prints a family breakdown over the
FAILURES, which is what the taxonomy is judged on. Stdlib only, so it runs in
any interpreter; no GPU, no simulator.
"""
import json, sys, collections
sys.path.insert(0, ".")
from vla_harness.schema import TraceStore
from vla_harness.mining.phases_libero import LiberoPhaseSegmenter
from vla_harness.mining.classify import classify


def mine(run_dir: str) -> dict:
    run_id = run_dir.rstrip("/").split("/")[-1]
    store = TraceStore("runs", run_id)
    seg = LiberoPhaseSegmenter()
    rows, fams, abstain = [], collections.Counter(), collections.Counter()
    for r in store.load():
        d = classify(r, segmenter=seg) or {}
        lp = (r.scene_descriptor or {}).get("libero_plus") or {}
        row = {"rollout_id": r.rollout_id, "task": r.scene_descriptor.get("task_id"),
               "success": bool(r.success), "category": lp.get("category"),
               "level": lp.get("difficulty_level"), "catalogue_id": lp.get("catalogue_id"),
               "family": d.get("family"), "families": d.get("families"),
               "predicates": d.get("predicates"), "reason": d.get("reason"),
               "terminal": d.get("terminal"), "n_steps": len(r.steps or [])}
        rows.append(row)
        if not row["success"]:
            fams[row["family"] or f"UNCLASSIFIED({d.get('reason')})"] += 1
            if str(d.get("reason", "")).startswith("missing"):
                abstain[d["reason"]] += 1
    with open(f"{run_dir}/diagnoses.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    n_fail = sum(1 for r in rows if not r["success"])
    print(f"\n### {run_dir}  n={len(rows)}  failures={n_fail}")
    for fam, k in fams.most_common():
        print(f"   {k:4d}  {fam}")
    if abstain:
        print(f"   detectors abstained: {dict(abstain)}")
    # which predicate fires on what share of failures -- the tightness question
    pred = collections.Counter()
    for r in rows:
        if not r["success"]:
            for p, v in (r["predicates"] or {}).items():
                if v:
                    pred[p] += 1
    if n_fail:
        print("   predicate fire rate over failures:")
        for p, k in pred.most_common():
            print(f"     {p:22s} {k:4d}/{n_fail}  {100*k/n_fail:5.1f}%")
    return {"run": run_dir, "n": len(rows), "failures": n_fail,
            "families": dict(fams), "predicates": dict(pred)}


if __name__ == "__main__":
    out = [mine(d) for d in sys.argv[1:]]
    json.dump(out, open("experiments/repro/mining_summary.json", "w"), indent=1)
