"""Training-data distribution vs the eval distribution: where are the gaps?

  .venvs/groot/bin/python .claude/skills/finetune-postmortem/scripts/data_coverage.py \
      --mint-run runs/r054 --export data/r054_train48 \
      --replay ~/.cache/huggingface/lerobot/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
      --eval-run runs/r056_r16 [--arm-run runs/r056_r16=r16 --arm-run runs/r058_r4=r4] \
      --nominal-starts runs/r047/nominal_starts.json [--window 48] [--out ...json]

Answers, from artifacts only:
  1. Scene coverage: kept training demos, distinct directions, noise/env seeds per scene vs the eval.
  2. Offset coverage: the start's gripper offset (cm from the scene's nominal start) of every minting
     ATTEMPT, every KEPT demo and every eval start, binned; the yield per bin (the success filter's
     selection bias) and base/arm eval success per bin. A bin with eval starts but ~no kept demos is a gap.
  3. Nearest-demo distance: for each eval start, the distance from its t=0 eef position to the closest
     kept demo of the same scene; eval success (base, arms) by distance tercile.
  4. Window content: what fraction of kept-demo frames fall in approach / gripper closed / object lifted
     (privileged object positions), vs the replay set's gripper-closed fraction. Shows which phases the
     new data teaches at all.
  5. Replay start-state spread: std of the replay set's frame-0 eef position per task, i.e. how much
     start variation the original training data ever had.
  6. State/action marginals: per-dim mean/std of the export (driven frames, non-driven frames) vs the
     replay set; dims whose mean differs by > 0.5 replay std are flagged (convention or scale mismatch,
     or a real distribution shift).
Minting rollouts are joined to the minting manifest BY POSITION (checked on rollout_id).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np

BINS_CM = [0, 3, 5, 8, 12, 16, 1e9]
LIFT_M = 0.03


def bin_of(cm):
    for lo, hi in zip(BINS_CM[:-1], BINS_CM[1:]):
        if lo <= cm < hi:
            return f"{lo:g}-{hi:g}" if hi < 1e9 else f">{lo:g}"
    return "?"


def jl(path):
    with open(path) as fh:
        for line in fh:
            yield json.loads(line)


def offset_cm(s0, nominal, scene):
    n = nominal.get(str(scene))
    return None if n is None else 100 * float(np.linalg.norm(np.subtract(s0["eef_pos"], n["eef_pos"])))


def mint_attempts(mint_run, exported_ids, nominal):
    man = list(jl(f"{mint_run}/manifest.jsonl"))
    out = []
    for m, r in zip(man, jl(f"{mint_run}/rollouts.jsonl")):
        if m["rollout_id"] != r["rollout_id"]:
            raise SystemExit("minting manifest and rollouts are not aligned by position")
        s0 = r["steps"][0]["obs_state"]
        out.append({"scene": m["scene_task_id"], "dir_seed": m.get("dir_seed"), "noise_seed": m.get("noise_seed"),
                    "env_seed": m.get("env_seed"), "band": m.get("band"), "success": bool(r["success"]),
                    "kept": r["rollout_id"] in exported_ids, "off": offset_cm(s0, nominal, m["scene_task_id"]),
                    "eef0": s0["eef_pos"], "steps": r["steps"]})
    return out


def eval_starts(run, nominal, arms):
    man = list(jl(f"{run}/manifest.jsonl"))
    base = {}
    for m, r in zip(man, jl(f"{run}/rollouts.jsonl")):
        if m["ckpt"] == "base" and m["set"] == "heldout":
            k = (m["scene"], m["dir_seed"], m["radius"], m["noise_seed"], m["env_seed"])
            s0 = r["steps"][0]["obs_state"]
            base[k] = {"scene": m["scene"], "dir_seed": m["dir_seed"], "noise_seed": m["noise_seed"],
                       "env_seed": m["env_seed"], "off": offset_cm(s0, nominal, m["scene"]), "eef0": s0["eef_pos"],
                       "succ": {"base": bool(m["success"])}}
    for arun, label in arms:
        for m in jl(f"{arun}/manifest.jsonl"):
            if m["ckpt"] != "base" and m["set"] == "heldout":
                k = (m["scene"], m["dir_seed"], m["radius"], m["noise_seed"], m["env_seed"])
                if k in base:
                    base[k]["succ"][label] = bool(m["success"])
    return list(base.values())


def phases(steps, window):
    s0 = steps[0]["obs_state"]
    tgt = list(s0["_gt_object_pos"])[0]
    z0 = s0["_gt_object_pos"][tgt][2]
    c = collections.Counter()
    closed = False
    for s in steps[:window]:
        a = s.get("action")
        if a is None:
            break
        closed = closed or a[6] > 0
        lifted = s["obs_state"]["_gt_object_pos"][tgt][2] - z0 > LIFT_M
        c["lifted" if lifted else ("closed" if a[6] > 0 else ("reopened" if closed else "approach"))] += 1
    return c


def parquet_frames(root):
    import pandas as pd
    files = sorted(glob.glob(os.path.join(os.path.expanduser(root), "data", "*", "*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def stack(col):
    return np.stack([np.asarray(x, dtype=np.float64) for x in col])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mint-run", required=True)
    ap.add_argument("--export", required=True, help="exported LeRobot dataset root (meta/provenance.jsonl)")
    ap.add_argument("--replay", default=None, help="replay LeRobot dataset root (parquet)")
    ap.add_argument("--eval-run", required=True, help="run holding the base held-out rollouts")
    ap.add_argument("--arm-run", action="append", default=[], help="RUN_DIR=label")
    ap.add_argument("--nominal-starts", required=True)
    ap.add_argument("--window", type=int, default=None, help="training truncation (frames per demo)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    nominal = json.load(open(a.nominal_starts))
    prov = list(jl(os.path.join(a.export, "meta", "provenance.jsonl")))
    exported = {p["rollout_id"] for p in prov}
    window = a.window or max(p["n_frames"] for p in prov)
    arms = [(x.split("=")[0].rstrip("/"), x.split("=")[1]) for x in a.arm_run]
    labels = ["base"] + [l for _, l in arms]
    att = mint_attempts(a.mint_run, exported, nominal)
    kept = [x for x in att if x["kept"]]
    ev = eval_starts(a.eval_run, nominal, arms)
    out = {}

    print(f"== 1. scene coverage (kept demos {len(kept)} of {len(att)} attempts; eval starts {len(ev)})")
    print(f"{'scene':>6} {'kept':>4} {'dirs':>4} {'noise':>9} {'env':>5} | {'eval':>4} {'dirs':>4} {'noise':>9} {'env':>5}"
          f" | shared dirs")
    out["scenes"] = {}
    for sc in sorted({x['scene'] for x in att} | {x['scene'] for x in ev}):
        K = [x for x in kept if x["scene"] == sc]; E = [x for x in ev if x["scene"] == sc]
        kd, ed = {x["dir_seed"] for x in K}, {x["dir_seed"] for x in E}
        row = {"kept": len(K), "kept_dirs": len(kd), "kept_noise": sorted({x["noise_seed"] for x in K}),
               "kept_env": sorted({x["env_seed"] for x in K}), "eval": len(E), "eval_dirs": len(ed),
               "eval_noise": sorted({x["noise_seed"] for x in E}), "eval_env": sorted({x["env_seed"] for x in E}),
               "shared_dirs": len(kd & ed)}
        out["scenes"][str(sc)] = row
        print(f"{sc:>6} {row['kept']:>4} {row['kept_dirs']:>4} {str(row['kept_noise']):>9} {str(row['kept_env']):>5} | "
              f"{row['eval']:>4} {row['eval_dirs']:>4} {str(row['eval_noise']):>9} {str(row['eval_env']):>5} | {row['shared_dirs']}")

    print("\n== 2. gripper-offset coverage (cm from the scene's nominal start at t=0)")
    names = [bin_of(lo + 1e-9) for lo in BINS_CM[:-1]]
    hdr = f"{'bin':>7} {'attempts':>8} {'kept':>5} {'yield':>6} | {'eval':>4} " + " ".join(f"{l:>7}" for l in labels)
    print(hdr)
    out["offset_bins"] = {}
    for b in names:
        A = [x for x in att if x["off"] is not None and bin_of(x["off"]) == b]
        K = [x for x in A if x["kept"]]
        E = [x for x in ev if x["off"] is not None and bin_of(x["off"]) == b]
        succ = {l: sum(1 for x in E if x["succ"].get(l)) for l in labels}
        yl = sum(x["success"] for x in A) / len(A) if A else float("nan")
        out["offset_bins"][b] = {"attempts": len(A), "kept": len(K), "yield": yl, "eval": len(E), "eval_success": succ}
        flag = "  <- eval starts with <5 kept demos" if E and len(K) < 5 else ""
        print(f"{b:>7} {len(A):>8} {len(K):>5} {yl:>6.2f} | {len(E):>4} " +
              " ".join(f"{succ[l]:>3}/{len(E):<3}" for l in labels) + flag)
    koff = [x["off"] for x in kept if x["off"] is not None]; eoff = [x["off"] for x in ev if x["off"] is not None]
    print(f"  kept demos: median {np.median(koff):.1f} cm, p90 {np.percentile(koff, 90):.1f};"
          f" eval starts: median {np.median(eoff):.1f} cm, p90 {np.percentile(eoff, 90):.1f}")

    print("\n== 3. nearest kept demo (same scene, t=0 eef distance) per eval start")
    for x in ev:
        K = [np.asarray(k["eef0"]) for k in kept if k["scene"] == x["scene"]]
        x["nn_cm"] = 100 * min(np.linalg.norm(np.asarray(x["eef0"]) - k) for k in K) if K else float("inf")
    nn = np.array([x["nn_cm"] for x in ev])
    qs = np.percentile(nn[np.isfinite(nn)], [33.3, 66.7])
    out["nearest"] = {"median_cm": float(np.median(nn)), "terciles_cm": qs.tolist(), "by_tercile": {}}
    print(f"  median {np.median(nn):.1f} cm, terciles at {qs[0]:.1f} / {qs[1]:.1f} cm")
    for name, lo, hi in [("near", -1, qs[0]), ("mid", qs[0], qs[1]), ("far", qs[1], 1e18)]:
        E = [x for x in ev if lo < x["nn_cm"] <= hi]
        s = {l: sum(1 for x in E if x["succ"].get(l)) for l in labels}
        out["nearest"]["by_tercile"][name] = {"n": len(E), **s}
        print(f"  {name:>4} (n={len(E):3d}): " + "  ".join(f"{l} {s[l]}/{len(E)}" for l in labels))

    print(f"\n== 4. what the kept demos' first {window} frames contain (privileged object height)")
    tot = collections.Counter()
    for x in kept:
        tot.update(phases(x["steps"], window))
    n = sum(tot.values())
    out["window_phases"] = {k: v / n for k, v in tot.items()}
    print("  " + "  ".join(f"{k} {v / n:.1%}" for k, v in tot.most_common()))
    full = collections.Counter()
    for x in kept:
        full.update(phases(x["steps"], 10 ** 6))
    nf = sum(full.values())
    print("  (full episodes, for comparison: " + "  ".join(f"{k} {v / nf:.1%}" for k, v in full.most_common()) + ")")

    if a.replay:
        rp = parquet_frames(a.replay)
        ex = parquet_frames(a.export)
        ra, rs = stack(rp["action"]), stack(rp["observation.state"])
        ea, es = stack(ex["action"]), stack(ex["observation.state"])
        drv = ex["frame_driven"].to_numpy().reshape(-1).astype(bool) if "frame_driven" in ex else np.zeros(len(ex), bool)
        print(f"\n  replay gripper-closed fraction (action[6] < 0.5, dataset 1 = open): {np.mean(ra[:, 6] < 0.5):.1%};"
              f" export: {np.mean(ea[:, 6] < 0.5):.1%}")
        out["gripper_closed_fraction"] = {"replay": float(np.mean(ra[:, 6] < 0.5)), "export": float(np.mean(ea[:, 6] < 0.5))}

        print("\n== 5. replay start-state spread (frame 0 eef position, per task)")
        f0 = rp[rp["frame_index"] == 0]
        sd = []
        for t, g in f0.groupby("task_index"):
            s = stack(g["observation.state"])[:, :3]
            sd.append(100 * np.linalg.norm(s.std(0)))
        kd = [x["off"] for x in kept]
        out["replay_start_spread_cm"] = {"median": float(np.median(sd)), "max": float(np.max(sd))}
        print(f"  replay: per-task |std| of frame-0 eef position, median {np.median(sd):.2f} cm, max {np.max(sd):.2f} cm"
              f" (kept demos start {np.median(kd):.1f} cm from nominal, eval {np.median(eoff):.1f} cm)")

        print("\n== 6. state/action marginals: export (driven | non-driven) vs replay; flag |dmean| > 0.5 replay std")
        print("   z_all = vs all replay frames; z_win = non-driven vs replay frames inside the same window"
              f" (frame_index < {window}), which removes phase composition. A dim still flagged on z_win is"
              " a convention/scale suspect; one flagged only on z_all is phase or start-pose composition.")
        win = (rp["frame_index"] < window).to_numpy()
        out["marginals"] = {}
        for name, R, E in [("action", ra, ea), ("state", rs, es)]:
            for d in range(R.shape[1]):
                mu, sdv = R[:, d].mean(), R[:, d].std() + 1e-9
                muw = R[win, d].mean()
                md, mn = E[drv, d].mean() if drv.any() else np.nan, E[~drv, d].mean()
                zd, zn, zw = (md - mu) / sdv, (mn - mu) / sdv, (mn - muw) / sdv
                flag = (" <- convention/scale suspect" if abs(zw) > 0.5 else
                        (" <- composition" if max(abs(np.nan_to_num(zd)), abs(zn)) > 0.5 else ""))
                out["marginals"][f"{name}[{d}]"] = {"replay_mean": mu, "replay_std": sdv, "replay_window_mean": muw,
                                                    "z_driven": zd, "z_nondriven": zn, "z_nondriven_window": zw}
                print(f"  {name}[{d}] replay {mu:+.3f}±{sdv:.3f} | driven z {zd:+.2f}  non-driven z {zn:+.2f}"
                      f"  non-driven z_win {zw:+.2f}{flag}")

    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=float)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
