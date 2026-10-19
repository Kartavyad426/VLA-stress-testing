"""Driver for the DMP PoC (docs/DMP_POC_METHOD.html §7). CPU only.

    CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=<main>/third_party/LIBERO-plus \
    LIBERO_CONFIG_PATH=<main>/third_party/libero-plus-config \
    <main>/.venvs/libero-plus/bin/python experiments/dmp_poc/run.py b1

Subcommands (added as the files they need land): b1, match, t3.
Outputs under runs/dmp_poc/ of this worktree.

UNVERIFIED: new numeric code, not yet checked by val.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from restore import WORKTREE, restore  # noqa: E402

OUT = os.path.join(WORKTREE, "runs/dmp_poc")

# §6 scenarios: (name, R-047 row, root step). Root steps for C and D are the
# first gripper-close command in the log; B1 confirms them along with the target.
SCENARIOS = [
    ("A", 14, 31),     # grasp.slip, scene 984
    ("B", 14, 242),    # place.off_target, scene 984
    ("C", 186, 28),    # grasp.miss, scene 1062
    ("D", 440, 18),    # far grasp.miss, scene 1327
    ("D_backup", 224, 21),  # far grasp.miss, scene 1090
]


def first_close(row_idx):
    from restore import r047_rows
    for s in r047_rows()[row_idx]["steps"]:
        if s["action"] is not None and s["action"][6] > 0:
            return s["t"]
    return None


def cmd_b1(a):
    os.makedirs(os.path.join(OUT, "b1"), exist_ok=True)
    rows = []
    for name, row, t in SCENARIOS:
        tc = first_close(row)
        t0 = time.time()
        r = restore(row, t)
        s = r.summary() | {"scenario": name, "first_close_t": tc, "wall_s": round(time.time() - t0, 1)}
        rows.append(s)
        with open(os.path.join(OUT, "b1", f"{name}_row{row}_t{t}.json"), "w") as f:
            json.dump(s | {"per_step": r.per_step}, f, indent=1)
        print(f"{name:9s} row {row:3d} t {t:3d}  first_close {tc}  ok {s['ok']}  "
              f"eef {s['max_eef_err_m'] * 1e3:.3f} mm  obj {s['max_obj_err_m'] * 1e3:.3f} mm  "
              f"worst t {s['worst_step']}  target {s['target']}  {s['wall_s']} s", flush=True)
        r.env._env.close()
    with open(os.path.join(OUT, "b1", "summary.json"), "w") as f:
        json.dump({"tag": "UNVERIFIED", "rows": rows}, f, indent=1)
    return 0 if all(r["ok"] for r in rows) else 1


def scenario_root(name, row, t):
    """Restore a §6 scenario and return (Root, task, phase) with the phase from roots.py."""
    from demos import task_name
    from restore import r047_rows
    from roots import roots
    rr = roots(row)
    ev = next(e for e in rr["events"] if e["t"] == t and e["mode"] not in ("grasp.ok", "place.ok"))
    r = restore(row, ev["restore_t"])
    return r, task_name(r047_rows()[row]["instruction"]), ev["phase"], ev


def cmd_match(a):
    os.makedirs(os.path.join(OUT, "match"), exist_ok=True)
    from match import search_root
    rows = []
    for name, row, t in SCENARIOS:
        r, task, phase, ev = scenario_root(name, row, t)
        ms, gap = search_root(r, task, phase)
        rec = {"scenario": name, "row": row, "t": t, "mode": ev["mode"], "restore_t": ev["restore_t"],
               "phase": phase, "restore_ok": r.ok, "d_gap": gap, "d_cover_over_gap": ms[0].d_cover_over_gap,
               "matches": [{"demo": m.demo.idx, "t_star": m.t_star, "d": m.d, "d_over_gap": m.d_over_gap,
                            "tags": m.tags, "phase_range": list(m.demo.phases[0 if phase == "A" else 1])}
                           for m in ms]}
        rows.append(rec)
        print(f"{name:9s} {ev['mode']:22s} restore t {ev['restore_t']:3d} phase {phase}  d_gap {gap:.2f}  "
              f"cover {ms[0].d_cover_over_gap:.2f}  " +
              "  ".join(f"demo {m['demo']:2d} t*{m['t_star']:3d} d/gap {m['d_over_gap']:.2f} {m['tags'] or ''}"
                        for m in rec["matches"]), flush=True)
        r.env._env.close()
    with open(os.path.join(OUT, "match", "scenarios.json"), "w") as f:
        json.dump({"tag": "UNVERIFIED", "rows": rows}, f, indent=1)
    return 0


def cmd_t3(a):
    """§6 scenarios: top-3 matches x 4 arms; every run starts from a fresh fast
    restore (exact, deterministic). Saves results and per-run actions/states."""
    import numpy as np
    from execute import ARMS, run
    from match import search_root
    from demos import task_name
    from restore import r047_rows, restore_fast, scene_of
    from roots import roots
    out_dir = os.path.join(OUT, "t3")
    os.makedirs(out_dir, exist_ok=True)
    envs, results = {}, []
    for name, row, t in SCENARIOS:
        ev = next(e for e in roots(row)["events"] if e["t"] == t and e["mode"] not in ("grasp.ok", "place.ok"))
        sc, task = scene_of(r047_rows()[row]), task_name(r047_rows()[row]["instruction"])
        r = restore_fast(row, ev["restore_t"], env=envs.get(sc))
        envs[sc] = r.env
        ms, gap = search_root(r, task, ev["phase"])
        for mi, m in enumerate(ms):
            for arm in ARMS:
                r = restore_fast(row, ev["restore_t"], env=envs[sc])
                if not r.ok:
                    raise RuntimeError(f"restore drifted for {name}")
                res, rec = run(r, m, ev["phase"], arm=arm)
                res |= {"scenario": name, "mode": ev["mode"], "phase": ev["phase"], "match_rank": mi,
                        "d_over_gap": m.d_over_gap, "d_cover_over_gap": m.d_cover_over_gap, "d_gap": gap}
                results.append(res)
                np.savez_compressed(os.path.join(out_dir, f"{name}_m{mi}_{arm}.npz"), act=np.array(rec.act),
                                    eef=np.array(rec.eef), rot=np.array(rec.rot), success=np.array(rec.success))
                c = res["checks"]
                print(f"{name:9s} m{mi} demo {m.demo.idx:2d} d/gap {m.d_over_gap:.2f} {arm:10s} kept {res['kept']!s:5s} "
                      f"succ {c['success']!s:5s} jerk {c['jerk_ok']!s:5s} speed {c['speed_ok']!s:5s} "
                      f"coll {c['collateral_ok']!s:5s} steps {res['steps']:3d} sat {res['saturated_frac']:.2f}", flush=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"tag": "UNVERIFIED", "rows": results}, f, indent=1, default=float)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("b1", help="root restore check for the §6 scenarios")
    sub.add_parser("match", help="restore each §6 scenario root and search the demos")
    sub.add_parser("t3", help="§6 scenarios: top-3 matches x 4 arms")
    a = ap.parse_args()
    return {"b1": cmd_b1, "match": cmd_match, "t3": cmd_t3}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
