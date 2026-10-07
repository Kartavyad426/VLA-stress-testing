"""D20(ii), revised (user 2026-10-06): R-047's radius-axis success against the ACHIEVED gripper offset at t=0
(cm from the scene's unperturbed start), next to the requested joint radius it is registered against. CPU only.

  .venvs/groot/bin/python experiments/r056_diag_offset.py [--out runs/r056_diag]

Sources: runs/r047/joint_radius_rad/manifest.jsonl (360 rows: success, requested magnitude, dir, noise seed),
runs/r047/rollouts.jsonl (t=0 obs_state per rollout_id), runs/r047/nominal_starts.json (unperturbed start per
scene, from r056_eval). rollout_id ignores the noise seed, so several manifest rows share one id; the start state
depends only on scene, env seed and spec, and the script checks that every trace line with one id has the same
t=0 state before using it. Magnitude-0 rows are offset 0 by construction and kept as their own band.

Writes <out>/r047_offset_curve.json (per requested radius: achieved joint radius and offset; per offset band:
success with Wilson 95%; per row) and <out>/r047_offset_curve.png (two panels, one y axis each).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np

BANDS_CM = [(0, 0.5), (0.5, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 30), (30, 100)]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/r056_diag")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = [json.loads(l) for l in open("runs/r047/joint_radius_rad/manifest.jsonl") if l.strip()]
    nom = {int(k): v for k, v in json.load(open("runs/r047/nominal_starts.json")).items()}
    rids = {r["rollout_id"] for r in man}
    t0, conflicts = {}, 0
    for line in open("runs/r047/rollouts.jsonl"):
        rid = json.loads(line[: line.index(',"task_id"')] + "}")["rollout_id"]
        if rid not in rids:
            continue
        st = json.loads(line)["steps"][0]["obs_state"]
        s = (tuple(st["eef_pos"]), tuple(st["joint_pos"]))
        if rid in t0 and t0[rid] != s:
            conflicts += 1
        t0.setdefault(rid, s)
    rows = []
    for r in man:
        s = t0.get(r["rollout_id"])
        if s is None:
            continue
        n0 = nom[int(r["task_id"])]
        off = float(np.linalg.norm(np.subtract(s[0], n0["eef_pos"]))) * 100
        rad = float(np.linalg.norm(np.subtract(s[1], n0["joint_pos"])))
        rows.append({"scene": int(r["task_id"]), "requested_rad": float(r["magnitude"]), "dir": r["dir"],
                     "noise_seed": r["noise_seed"], "success": bool(r["success"]), "offset_cm": off,
                     "achieved_rad": rad})
    by_req = {}
    for x in rows:
        by_req.setdefault(x["requested_rad"], []).append(x)
    per_req = {}
    for m, xs in sorted(by_req.items()):
        k = sum(x["success"] for x in xs)
        ach = np.array([x["achieved_rad"] for x in xs]); off = np.array([x["offset_cm"] for x in xs])
        per_req[f"{m:g}"] = {"n": len(xs), "success": k, "rate": k / len(xs), "wilson95": wilson(k, len(xs)),
                             "achieved_rad_mean": float(ach.mean()),
                             "achieved_over_requested": float(ach.mean() / m) if m else None,
                             "offset_cm_median": float(np.median(off)), "offset_cm_p10_p90": [float(np.percentile(off, 10)), float(np.percentile(off, 90))]}
    def bands(rs):
        out = {}
        for lo, hi in BANDS_CM:
            xs = [x for x in rs if lo <= x["offset_cm"] < hi]
            k = sum(x["success"] for x in xs)
            out[f"{lo:g}-{hi:g}"] = {"n": len(xs), "success": k, "rate": (k / len(xs)) if xs else None,
                                     "wilson95": wilson(k, len(xs)), "start_states": len({(x["scene"], x["dir"], x["requested_rad"]) for x in xs}),
                                     "requested_rad_values": sorted({x["requested_rad"] for x in xs})}
        return out
    per_band = bands(rows)
    dirs = sorted({x["dir"] for x in rows})
    per_band_by_dir = {str(d): bands([x for x in rows if x["dir"] == d]) for d in dirs}
    per_req_by_dir = {}
    for d in dirs:
        for m in sorted({x["requested_rad"] for x in rows}):
            xs = [x for x in rows if x["dir"] == d and x["requested_rad"] == m]
            k = sum(x["success"] for x in xs)
            per_req_by_dir.setdefault(str(d), {})[f"{m:g}"] = {
                "n": len(xs), "success": k, "rate": k / len(xs),
                "achieved_over_requested": float(np.mean([x["achieved_rad"] for x in xs]) / m) if m else None,
                "offset_cm_median": float(np.median([x["offset_cm"] for x in xs]))}
    ratio_by_dir = {}
    for x in rows:
        if x["requested_rad"] > 0:
            ratio_by_dir.setdefault(str(x["dir"]), []).append(x["achieved_rad"] / x["requested_rad"])
    out = {"what": "D20(ii) revised: R-047 radius-axis success vs achieved t=0 gripper offset (cm)",
           "n_manifest": len(man), "n_with_start_state": len(rows), "trace_t0_conflicts": conflicts,
           "achieved_over_requested_by_dir": {d: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
                                              for d, v in sorted(ratio_by_dir.items())},
           "offset_definition": "|t=0 obs_state.eef_pos - nominal_starts[scene].eef_pos| x 100, per rollout (not derived from joint radius)",
           "per_requested_radius": per_req, "per_requested_radius_by_dir": per_req_by_dir,
           "per_offset_band_cm": per_band, "per_offset_band_cm_by_dir": per_band_by_dir, "rows": rows}
    json.dump(out, open(os.path.join(a.out, "r047_offset_curve.json"), "w"), indent=1)
    plot(per_req, per_band, os.path.join(a.out, "r047_offset_curve.png"))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


def plot(per_req, per_band, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, col = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor="#fcfcfb")

    def panel(ax, xs, ks, ns, labels, xlabel, title):
        ax.set_facecolor("#fcfcfb")
        r = [k / n for k, n in zip(ks, ns)]
        ci = [wilson(k, n) for k, n in zip(ks, ns)]
        ax.vlines(xs, [c[0] for c in ci], [c[1] for c in ci], color=col, lw=2, alpha=0.35)
        ax.plot(xs, r, color=col, lw=2, marker="o", ms=8, mec="#fcfcfb", mew=2)
        for x, y, n in zip(xs, r, ns):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=8, color=muted)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9, color=muted)
        ax.set_xlabel(xlabel, color=muted); ax.set_title(title, color=ink, fontsize=11, loc="left")
        ax.set_ylim(-0.05, 1.05); ax.grid(axis="y", color=grid, lw=1); ax.tick_params(colors=muted)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)

    rq = list(per_req.items())
    panel(axs[0], list(range(len(rq))), [v["success"] for _, v in rq], [v["n"] for _, v in rq],
          [f"{k}\n({v['offset_cm_median']:.0f} cm)" for k, v in rq], "requested joint radius (rad)  (median offset)",
          "R-047 success vs requested radius")
    bd = [(k, v) for k, v in per_band.items() if v["n"]]
    panel(axs[1], list(range(len(bd))), [v["success"] for _, v in bd], [v["n"] for _, v in bd],
          [k for k, _ in bd], "achieved gripper offset at t=0 (cm)", "R-047 success vs achieved offset")
    axs[0].set_ylabel("success rate (Wilson 95%)", color=muted)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
