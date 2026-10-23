"""Driver for the scripted expert (not run.py: that name is DMP's driver, also on sys.path). CPU only.

    M=<main checkout>
    CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=$M/third_party/LIBERO-plus \
    LIBERO_CONFIG_PATH=$M/third_party/libero-plus-config \
    $M/.venvs/libero-plus/bin/python experiments/scripted_expert/driver.py t1

Subcommands:
  t1     DMP's 5 scenarios (same rows, root steps and restore as DMP's t3) x the
         3 variants; every run starts from a fresh restore_fast
Outputs under runs/scripted_expert/ of this worktree: results.json in DMP's row
format (execute.judge + arm, scenario, plan) and per-run npz (act, eef, rot,
success; same keys as DMP's, not committed).

UNVERIFIED: new numeric code, not yet checked by val.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dmp_link import DMP_PIN, WORKTREE, demos as D, restore as RS, roots as RT  # noqa: E402
import expert as X  # noqa: E402

OUT = os.path.join(WORKTREE, "runs/scripted_expert")

# DMP's §6 scenarios (experiments/dmp_poc/run.py SCENARIOS at dae447a)
SCENARIOS = [
    ("A", 14, 31),
    ("B", 14, 242),
    ("C", 186, 28),
    ("D", 440, 18),
    ("D_backup", 224, 21),
]


def event_of(row, t):
    return next(e for e in RT.roots(row)["events"] if e["t"] == t and e["mode"] not in ("grasp.ok", "place.ok"))


def cmd_t1(a):
    out_dir = os.path.join(OUT, "t1")
    os.makedirs(out_dir, exist_ok=True)
    envs, results = {}, []
    variants = a.variants.split(",") if a.variants else list(X.VARIANTS)
    for name, row, t in SCENARIOS:
        if a.only and name not in a.only.split(","):
            continue
        ev = event_of(row, t)
        r047 = RS.r047_rows()[row]
        sc, task = RS.scene_of(r047), D.task_name(r047["instruction"])
        for v in variants:
            t0 = time.time()
            r = RS.restore_fast(row, ev["restore_t"], env=envs.get(sc))
            envs[sc] = r.env
            if not r.ok:
                raise RuntimeError(f"restore drifted for {name}")
            res, rec = X.run(r, task, v)
            res |= {"scenario": name, "mode": ev["mode"], "phase": ev["phase"], "seconds": round(time.time() - t0, 1)}
            results.append(res)
            np.savez_compressed(os.path.join(out_dir, f"{name}_{v}.npz"), act=np.array(rec.act),
                                eef=np.array(rec.eef), rot=np.array(rec.rot), success=np.array(rec.success))
            c, p = res["checks"], res["plan"]
            print(f"{name:9s} {v:8s} kept {res['kept']!s:5s} succ {c['success']!s:5s} jerk {c['jerk_ok']!s:5s} "
                  f"speed {c['speed_ok']!s:5s} coll {c['collateral_ok']!s:5s} steps {res['steps']:3d} "
                  f"branch {p['branch']} side {p.get('grasp_side')} lift {p.get('lift_m', float('nan')):.3f} "
                  f"retries {p['retries']} jerk {res['jerk_max']:.1f}/{res['jerk_limit']:.1f} "
                  f"speed {res['speed_max']:.3f}/{res['speed_limit']:.3f} moved {max(res['moved_m'].values(), default=0):.3f} "
                  f"{res['seconds']}s", flush=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"tag": "UNVERIFIED", "dmp_pin": DMP_PIN, "rows": results}, f, indent=1, default=float)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t1 = sub.add_parser("t1", help="DMP's 5 scenarios x variants")
    t1.add_argument("--only", default="", help="comma-separated scenario names")
    t1.add_argument("--variants", default="", help="comma-separated variant names (default: all)")
    a = ap.parse_args()
    return {"t1": cmd_t1}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
