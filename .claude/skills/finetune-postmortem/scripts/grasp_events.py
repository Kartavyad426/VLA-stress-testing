"""Gripper-close geometry and grasp success, for eval runs or for the training demos.

Eval mode -- where and how the arms close the gripper, per scene x arm x outcome:
  .venvs/groot/bin/python .claude/skills/finetune-postmortem/scripts/grasp_events.py \
      eval --run runs/r056_r16=r16 --run runs/r058_r4=r4 [--scenes 1327,1062]

Demo mode -- defects the success filter lets through and truncation hides:
  .venvs/groot/bin/python .claude/skills/finetune-postmortem/scripts/grasp_events.py \
      demos --mint-run runs/r054 --export data/r054_train48 [--window 48]
  Per scene: demos that close / lift inside the window, demos whose FIRST grasp failed (a second
  close before the lift), whether that premature close is inside the window, and whether the recovery
  (reopen + regrasp) is cut by the truncation. The failed-first-grasp list prints close time and height.

A close event is the executed gripper action switching from open (<= 0) to close (> 0), env convention
(-1 open, +1 close). dz = eef z - target z at that step; target = first _gt_object_pos key.
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

LIFT_M = 0.03


def jl(path):
    with open(path) as fh:
        for line in fh:
            yield json.loads(line)


def events(steps):
    s0 = steps[0]["obs_state"]
    tgt = list(s0["_gt_object_pos"])[0]
    z0 = s0["_gt_object_pos"][tgt][2]
    closes, opens, prev = [], [], -1.0
    for s in steps:
        a = s.get("action")
        if a is None:
            continue
        o = s["obs_state"]
        if a[6] > 0 and prev <= 0:
            closes.append((s["t"], *(np.subtract(o["eef_pos"], o["_gt_object_pos"][tgt]))))
        if a[6] <= 0 and prev > 0:
            opens.append(s["t"])
        prev = a[6]
    lift = next((s["t"] for s in steps if s["obs_state"]["_gt_object_pos"][tgt][2] - z0 > LIFT_M), None)
    failed_first = len(closes) > 1 and (lift is None or lift > closes[1][0])
    return closes, opens, lift, failed_first


def cmd_eval(a):
    out = collections.defaultdict(list)
    scenes = {int(x) for x in a.scenes.split(",")} if a.scenes else None
    for spec in a.run:
        run, label = (spec.split("=") + [spec.rstrip("/").split("/")[-1]])[:2]
        man = list(jl(f"{run}/manifest.jsonl"))
        for m, r in zip(man, jl(f"{run}/rollouts.jsonl")):
            if m["set"] not in ("heldout", "nominal") or (scenes and m["scene"] not in scenes):
                continue
            arm = "base" if m["ckpt"] == "base" else label
            if arm == "base" and any(k[1] == "base" for k in out if k[0] == m["scene"]) and spec != a.run[0]:
                continue                                    # base rows only from the first run that has them
            closes, _, lift, _ = events(r["steps"])
            if closes:
                t, dx, dy, dz = closes[0]
                out[(m["scene"], arm, bool(r["success"]))].append((t, dx, dy, dz, len(closes), lift is None))
    print(f"{'scene':>6} {'arm':>5} {'':1} {'n':>3} {'1st close t':>11} {'dx':>6} {'dy':>6} {'dz cm':>6} {'#closes':>7} {'no lift':>7}")
    for k in sorted(out, key=lambda k: (k[0], k[1] != "base", k[1], not k[2])):
        A = np.array(out[k], dtype=float)
        print(f"{k[0]:>6} {k[1]:>5} {'S' if k[2] else 'F'} {len(A):>3} {np.median(A[:, 0]):>11.1f} "
              f"{np.median(A[:, 1]) * 100:>6.1f} {np.median(A[:, 2]) * 100:>6.1f} {np.median(A[:, 3]) * 100:>6.1f} "
              f"{np.median(A[:, 4]):>7.0f} {int(A[:, 5].sum()):>7d}")


def cmd_demos(a):
    prov = {p["rollout_id"]: p for p in jl(f"{a.export}/meta/provenance.jsonl")}
    window = a.window or max(p["n_frames"] for p in prov.values())
    per, bad, good_dz, bad_dz = collections.defaultdict(collections.Counter), [], [], []
    for r in jl(f"{a.mint_run}/rollouts.jsonl"):
        if r["rollout_id"] not in prov:
            continue
        sc = prov[r["rollout_id"]]["provenance"]["scene_task_id"]
        closes, opens, lift, ff = events(r["steps"])
        c = per[sc]
        c["n"] += 1
        c["close_in_window"] += bool(closes and closes[0][0] < window)
        c["lift_in_window"] += bool(lift is not None and lift < window)
        if closes:
            (bad_dz if ff else good_dz).append(closes[0][3])
        if ff:
            c["failed_first"] += 1
            inside = closes[0][0] < window
            recovery_cut = not any(o < window for o in opens)
            c["premature_close_in_window"] += inside
            c["premature_in_window_recovery_cut"] += inside and recovery_cut
            bad.append((sc, prov[r["rollout_id"]]["provenance"].get("band"), closes[0][0], closes[0][3] * 100,
                        len(closes), lift, inside, recovery_cut))
    cols = ["n", "close_in_window", "lift_in_window", "failed_first", "premature_close_in_window",
            "premature_in_window_recovery_cut"]
    print(f"window = {window} frames")
    print(f"{'scene':>6} " + " ".join(f"{c[:14]:>14}" for c in cols))
    tot = collections.Counter()
    for sc in sorted(per):
        tot.update(per[sc])
        print(f"{sc:>6} " + " ".join(f"{per[sc][c]:>14}" for c in cols))
    print(f"{'all':>6} " + " ".join(f"{tot[c]:>14}" for c in cols))
    if good_dz:
        print(f"median dz at first close: clean first grasp {np.median(good_dz) * 100:.1f} cm"
              + (f", failed first grasp {np.median(bad_dz) * 100:.1f} cm" if bad_dz else ""))
    if bad:
        print("\nfailed-first-grasp demos: scene band first_close_t dz_cm n_closes lift_t close_in_window recovery_cut")
        for b in bad:
            print(f"  {b[0]} {b[1]} {b[2]} {b[3]:.1f} {b[4]} {b[5]} {b[6]} {b[7]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval"); e.add_argument("--run", action="append", required=True); e.add_argument("--scenes")
    d = sub.add_parser("demos"); d.add_argument("--mint-run", required=True); d.add_argument("--export", required=True)
    d.add_argument("--window", type=int)
    a = ap.parse_args()
    cmd_eval(a) if a.cmd == "eval" else cmd_demos(a)


if __name__ == "__main__":
    main()
