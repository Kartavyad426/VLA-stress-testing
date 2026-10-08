"""PERTURBATION_MAP validity audit (res21, 2026-10-08): CPU-only re-scores of STORED data.

Read-only. Nothing here touches the GPU, the harness or any run directory.

  .venvs/libero-plus/bin/python -I experiments/audit_rescore.py --out runs/audit_pm/rescore.json

Checks (threat numbers follow docs/PERTURBATION_MAP_VALIDITY.html):
  T2  tau at forward 0 recomputed on (a) all 40x7 predicted actions, as stored;
      (b) the 16 executed steps; (c) the first action only.
  T7  tau in ENV units: min-max un-normalised with the checkpoint's own
      postprocessor stats, clipped to [-1, 1] first (clip_normalized_action=True),
      gripper continuous; also translation-only (dims 0-2). Per-dimension-group
      shares of ||P-N||^2 in both spaces.
  T4  seed spread: per-instance tau over noise-seed replicates (base run + seed1/seed2).
  T1  start pose: object layout of the variant vs its control at t = 0 (from traces).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

CKPT = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--gr00t17-lerobot-libero_spatial-640/snapshots/*/"))[0]
ARMS = ("T", "I", "S", "IT", "A", "W")
GROUPS = {"trans": [0, 1, 2], "rot": [3, 4, 5], "grip": [6]}

# (run, label) for every splice run whose npz holds per-arm chunks
RUNS = ["r039", "r042", "r044_seed1", "r044_seed2", "r048_seed1", "r048_seed2",
        "r049", "r049_lang",
        "r051_noise_seed1", "r051_noise_seed2", "r051_light_seed1", "r051_light_seed2",
        "r051_layout_seed1", "r051_layout_seed2", "r051_texture_seed1", "r051_texture_seed2"]
# replicate families for the seed spread: base run first
REPS = {"Robot Initial States": ["r042", "r044_seed1", "r044_seed2"],
        "Camera Viewpoints": ["r042", "r048_seed1", "r048_seed2"],
        "Sensor Noise": ["r042", "r051_noise_seed1", "r051_noise_seed2"],
        "Light Conditions": ["r042", "r051_light_seed1", "r051_light_seed2"],
        "Objects Layout": ["r049", "r051_layout_seed1", "r051_layout_seed2"],
        "Background Textures": ["r049", "r051_texture_seed1", "r051_texture_seed2"]}


def load_stats():
    from safetensors.numpy import load_file
    d = load_file(glob.glob(os.path.join(CKPT, "policy_postprocessor_step_0*.safetensors"))[0])
    return d["action.min"].astype(np.float64), d["action.max"].astype(np.float64)


def unnorm(x, lo, hi):
    x = np.clip(x.astype(np.float64), -1.0, 1.0)
    return (x + 1.0) / 2.0 * (hi - lo) + lo


def tau(p, t, s):
    d = np.linalg.norm((t - s).ravel())
    return float("nan") if d == 0 else 1.0 - np.linalg.norm((p - s).ravel()) / d


def manifest(run):
    p = f"runs/{run}/splice/manifest.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def per_instance(run, lo, hi):
    out = []
    for row in manifest(run):
        p = f"runs/{run}/splice/{row['rollout_id']}.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p)
        if "act_P" not in z.files or "act_N" not in z.files or z["act_P"].shape[0] == 0:
            continue
        P, N = z["act_P"][0].astype(np.float64), z["act_N"][0].astype(np.float64)     # forward 0, 40 x 7
        rec = {"run": run, "task_id": row["task_id"], "category": row["category"], "success": row["success"],
               "anchor_forward": row.get("anchor_forward"), "forwards": row.get("forwards"),
               "d_pn_full": float(np.linalg.norm(P - N)), "d_pn_exec16": float(np.linalg.norm(P[:16] - N[:16])),
               "stored_tf_f0": row.get("tf_at_f0") or {}, "tau": {}}
        Pe, Ne = unnorm(P, lo, hi), unnorm(N, lo, hi)
        d2n = (P - N) ** 2
        d2e = (Pe - Ne) ** 2
        rec["share_norm"] = {g: float(d2n[:, ix].sum() / d2n.sum()) if d2n.sum() else float("nan") for g, ix in GROUPS.items()}
        rec["share_env"] = {g: float(d2e[:, ix].sum() / d2e.sum()) if d2e.sum() else float("nan") for g, ix in GROUPS.items()}
        rec["share_norm_exec16"] = {g: float(d2n[:16, ix].sum() / d2n[:16].sum()) if d2n[:16].sum() else float("nan") for g, ix in GROUPS.items()}
        for arm in ARMS:
            k = f"act_{arm}"
            if k not in z.files:
                continue
            X = z[k][0].astype(np.float64)
            Xe = unnorm(X, lo, hi)
            rec["tau"][arm] = {
                "full": tau(X, P, N),
                "exec16": tau(X[:16], P[:16], N[:16]),
                "first": tau(X[:1], P[:1], N[:1]),
                "env_full": tau(Xe, Pe, Ne),
                "env_exec16": tau(Xe[:16], Pe[:16], Ne[:16]),
                "env_trans_exec16": tau(Xe[:16, :3], Pe[:16, :3], Ne[:16, :3]),
                "nogrip_exec16": tau(X[:16, :6], P[:16, :6], N[:16, :6]),
            }
        # anchor-forward tau, to compare with f0 (threat 3)
        a = row.get("anchor_forward")
        if a is not None and 0 <= a < z["act_P"].shape[0]:
            Pa, Na = z["act_P"][a].astype(np.float64), z["act_N"][a].astype(np.float64)
            rec["tau_anchor_full"] = {arm: tau(z[f"act_{arm}"][a].astype(np.float64), Pa, Na)
                                      for arm in ARMS if f"act_{arm}" in z.files}
        out.append(rec)
    return out


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def boot_ci(xs, n=10000, seed=0):
    xs = np.array([x for x in xs if x is not None and np.isfinite(x)])
    if len(xs) < 3:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    m = np.median(rng.choice(xs, size=(n, len(xs)), replace=True), axis=1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def summarise(recs):
    by = defaultdict(list)
    for r in recs:
        by[(r["run"], r["category"])].append(r)
    out = {}
    for (run, cat), rs in sorted(by.items()):
        arms = sorted({a for r in rs for a in r["tau"]})
        s = {"n": len(rs), "n_fail": sum(not r["success"] for r in rs),
             "median_d_pn_full": med([r["d_pn_full"] for r in rs]),
             "median_share_norm": {g: med([r["share_norm"][g] for r in rs]) for g in GROUPS},
             "median_share_env": {g: med([r["share_env"][g] for r in rs]) for g in GROUPS},
             "arms": {}}
        for a in arms:
            vals = {k: [r["tau"][a][k] for r in rs if a in r["tau"]] for k in
                    ("full", "exec16", "first", "env_full", "env_exec16", "env_trans_exec16", "nogrip_exec16")}
            stored = [r["stored_tf_f0"].get(a) for r in rs]
            s["arms"][a] = {k: med(v) for k, v in vals.items()}
            s["arms"][a]["ci95_full"] = boot_ci(vals["full"])
            s["arms"][a]["ci95_exec16"] = boot_ci(vals["exec16"])
            s["arms"][a]["max_abs_recomputed_vs_stored"] = float(np.nanmax(np.abs(np.array(vals["full"], float) - np.array(stored, float)))) if all(x is not None for x in stored) else None
            s["arms"][a]["anchor_median"] = med([r.get("tau_anchor_full", {}).get(a) for r in rs])
        out[f"{run}|{cat}"] = s
    return out


def seed_spread(recs):
    """Per instance, tau over the noise-seed replicates; spread = max - min."""
    idx = {(r["run"], r["task_id"]): r for r in recs}
    out = {}
    for cat, runs in REPS.items():
        tids = sorted({r["task_id"] for r in recs if r["run"] == runs[0] and r["category"] == cat})
        per_arm = defaultdict(list)
        dpn_rel = []
        for t in tids:
            reps = [idx.get((run, t)) for run in runs]
            reps = [r for r in reps if r is not None]
            if len(reps) < 2:
                continue
            d = [r["d_pn_full"] for r in reps]
            dpn_rel.append((max(d) - min(d)) / np.mean(d) if np.mean(d) else float("nan"))
            for a in ARMS:
                v = [r["tau"].get(a, {}).get("full") for r in reps]
                v = [x for x in v if x is not None and np.isfinite(x)]
                if len(v) >= 2:
                    per_arm[a].append(max(v) - min(v))
        out[cat] = {"replicates": runs, "n_instances": len(dpn_rel),
                    "median_relative_spread_d_pn": med(dpn_rel),
                    "median_tau_range_over_seeds": {a: med(v) for a, v in per_arm.items()},
                    "max_tau_range_over_seeds": {a: float(np.max(v)) for a, v in per_arm.items() if v}}
    return out


def layout_check():
    """Start pose: are the variant and its control the same world at t = 0 (objects), differing only in the arm?"""
    rows = [json.loads(l) for l in open("runs/r042/rollouts.jsonl")]
    by_task = {}
    for r in rows:
        tid = int(str(r["task_id"]).rsplit("task", 1)[-1])
        by_task.setdefault(tid, r)
    sel = json.load(open("experiments/repro/r044_selection_ris.json"))["instances"]
    out = []
    for inst in sel:
        t, c = inst["task_id"], inst["control_task_id"]
        if t not in by_task or c not in by_task:
            out.append({"task_id": t, "control": c, "missing": True}); continue
        a = by_task[t]["steps"][0]["obs_state"]; b = by_task[c]["steps"][0]["obs_state"]
        oa, ob = a.get("_gt_scene_object_pos") or a.get("_gt_object_pos"), b.get("_gt_scene_object_pos") or b.get("_gt_object_pos")
        common = sorted(set(oa) & set(ob))
        dobj = max(float(np.linalg.norm(np.array(oa[k]) - np.array(ob[k]))) for k in common) if common else float("nan")
        deef = float(np.linalg.norm(np.array(a["eef_pos"]) - np.array(b["eef_pos"])))
        out.append({"task_id": t, "control": c, "n_objects": len(common), "max_object_offset_m": dobj, "eef_offset_m": deef,
                    "objects_only_in_one": sorted(set(oa) ^ set(ob))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/audit_pm/rescore.json")
    a = ap.parse_args()
    lo, hi = load_stats()
    recs = []
    for run in RUNS:
        recs += per_instance(run, lo, hi)
    res = {"what": "PERTURBATION_MAP validity audit, CPU re-score of stored splice npz (res21, 2026-10-08)",
           "action_stats": {"min": lo.tolist(), "max": hi.tolist(), "half_range": ((hi - lo) / 2).tolist()},
           "summary": summarise(recs), "seed_spread": seed_spread(recs), "layout_check_startpose": layout_check(),
           "n_instances": len(recs)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    with open(a.out.replace(".json", "_rows.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {a.out}: {len(recs)} instances")


if __name__ == "__main__":
    main()
