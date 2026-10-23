"""Where do the steps go? Per-phase durations and speeds: scripted expert vs the
human demos of the same task (and DMP's t3 runs where its npz allows). CPU.

The expert runs DMP's 5 scenarios x 3 variants with a large budget (no 200-step
limit) so every phase completes. Phases, from events read the same way for
both sources:
  approach   start -> close command of the final grasp (expert: root -> its close;
             demos: start of phase A -> close)
  close      close command -> lift-off (bowl >= LIFTOFF_M above its height at the close)
  carry      lift-off -> release command
  finish     release command -> success (expert; 0 when success fires while the
             bowl is still held); release -> demo end (demos are recorded until
             success, so this is an upper bound for them)
For a root with the bowl already held (B) the first closed run starts at 0:
approach is 0 and "close" is the time until the bowl rises.
Speeds: EEF path length / duration per phase.
DMP (t3 npz has act, eef, success; no object poses): approach, close+carry
(close -> release) and finish only.

    ... driver.py-style env ... python experiments/scripted_expert/phase_timing.py

Writes runs/scripted_expert/phase_timing.json.

UNVERIFIED: new numeric code, not yet checked by val.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dmp_link import DMP_WORKTREE, WORKTREE, demos as D, execute as E, restore as RS  # noqa: E402
import expert as X  # noqa: E402
from driver import SCENARIOS, event_of  # noqa: E402

BUDGET = 600
LIFTOFF_M = 0.01
OUT = os.path.join(WORKTREE, "runs/scripted_expert/phase_timing.json")


def path_len(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


def phases(eef, grip_cmd, bowl_z, close, release, end):
    """{phase: (steps, path_m, mean_speed)} from event steps."""
    lift = next((t for t in range(close, release) if bowl_z[t] - bowl_z[close] >= LIFTOFF_M), release)
    spans = {"approach": (0, close), "close": (close, lift), "carry": (lift, release), "finish": (release, end)}
    out = {}
    for k, (a, b) in spans.items():
        n = max(0, b - a)
        L = path_len(eef[a:b + 1])
        out[k] = {"steps": n, "path_m": L, "speed": L / (n * E.DT) if n else 0.0}
    out["total_steps"] = end
    return out


def final_grasp(cmd):
    """(close step, release step) of the last closed run followed by an open, from a command sign array."""
    closed = cmd > 0
    runs, t = [], 0
    while t < len(closed):
        if closed[t]:
            s = t
            while t < len(closed) and closed[t]:
                t += 1
            runs.append((s, t))
        else:
            t += 1
    # a run may end still closed: LIBERO's success often fires with the bowl
    # still in the hand, before the release; then "release" = the last step
    return runs[-1] if runs else (None, None)


def expert_rows():
    rows, envs = [], {}
    for name, row, t in SCENARIOS:
        ev = event_of(row, t)
        r047 = RS.r047_rows()[row]
        sc, task = RS.scene_of(r047), D.task_name(r047["instruction"])
        for v in X.VARIANTS:
            r = RS.restore_fast(row, ev["restore_t"], env=envs.get(sc))
            envs[sc] = r.env
            res, rec = X.run(r, task, v, budget=BUDGET)
            eef, act = np.array(rec.eef), np.array(rec.act)
            bowl = np.array([o[X.goal_of(RS.control_env(r.env))[0]][2] for o in rec.objs])
            close, release = final_grasp(act[:, 6])
            end = res["success_step"] or len(eef)
            out = {"scenario": name, "variant": v, "task": task, "success_step": res["success_step"],
                   "branch": res["plan"]["branch"]}
            if close is not None and release is not None:
                out |= phases(eef, act[:, 6], bowl, close, release, end)
            else:
                out["note"] = "no close->release in the run (bowl held at the root: no approach/close)"
                rel = int(np.argmax(act[:, 6] < 0)) if (act[:, 6] < 0).any() else len(eef)
                out |= {"carry": {"steps": rel, "path_m": path_len(eef[:rel + 1]),
                                  "speed": path_len(eef[:rel + 1]) / max(rel, 1) / E.DT},
                        "finish": {"steps": end - rel}, "total_steps": end}
            rows.append(out)
            print(name, v, {k: (out[k]["steps"] if isinstance(out.get(k), dict) else out.get(k))
                            for k in ("approach", "close", "carry", "finish", "total_steps", "success_step")}, flush=True)
    return rows


def demo_rows(task):
    rows = []
    for d in D.load(task):
        if not d.usable:
            continue
        A, B, _ = d.phases
        eef = d.eef_pos[A[0]:]
        cmd = np.where(d.grip_cmd > 0, 1.0, -1.0)[A[0]:]
        r = phases(eef, cmd, d.target_pos[A[0]:, 2], B[0] - A[0], B[1] - A[0], len(eef))
        rows.append(r)
    return rows


def dmp_rows():
    res = json.load(open(os.path.join(DMP_WORKTREE, "runs/dmp_poc/t3/results.json")))["rows"]
    out = []
    for r in res:
        z = np.load(os.path.join(DMP_WORKTREE, f"runs/dmp_poc/t3/{r['scenario']}_m{r['match_rank']}_{r['arm']}.npz"))
        act, eef = z["act"], z["eef"]
        close, release = final_grasp(act[:, 6])
        if close is None:
            continue
        end = r["success_step"] or len(eef)
        out.append({"scenario": r["scenario"], "arm": r["arm"], "match_rank": r["match_rank"], "kept": r["kept"],
                    "approach": close, "close_carry": release - close, "finish": end - release,
                    "total_steps": end, "success_step": r["success_step"],
                    "approach_speed": path_len(eef[:close + 1]) / max(close, 1) / E.DT,
                    "close_carry_speed": path_len(eef[close:release + 1]) / max(release - close, 1) / E.DT})
    return out


def summarise(rows, keys=("approach", "close", "carry", "finish")):
    s = {}
    for k in keys:
        st = [r[k]["steps"] for r in rows if isinstance(r.get(k), dict)]
        sp = [r[k]["speed"] for r in rows if isinstance(r.get(k), dict) and "speed" in r[k]]
        if st:
            s[k] = {"steps_med": float(np.median(st)), "steps_p10": float(np.percentile(st, 10)),
                    "steps_p90": float(np.percentile(st, 90)),
                    "speed_med": float(np.median(sp)) if sp else None}
    s["total_med"] = float(np.median([r["total_steps"] for r in rows]))
    return s


def main():
    ex = expert_rows()
    tasks = sorted({r["task"] for r in ex})
    humans = {t: summarise(demo_rows(t)) for t in tasks}
    out = {"tag": "UNVERIFIED", "budget": BUDGET, "liftoff_m": LIFTOFF_M, "expert": ex,
           "humans_by_task": humans, "dmp_t3": dmp_rows()}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=float)
    for t in tasks:
        print("HUMAN", t, json.dumps(humans[t]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
