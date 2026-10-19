"""Root events for R-047 failures (docs/DMP_POC_METHOD.html §8, decision C4). No sim.

Rules: MINING_SPEC_V1_DESIGN §3.2 (event vocabulary), §3.3 R1 (recovered flag),
§3.7 (in-gripper check), with the §6 starting thresholds. They supersede
prim's probe (docs/drafts/prim_probes/root_rules_probe_v2.py) where they differ:
lift measured from the height at onset, slip includes rose-then-fell, wrong
object needs the object in the gripper (not a bump), drop needs separation
from the gripper, goal radius 3.5 cm (fitted), not 6 cm.

Deviation from §3.2: grasp.slip's "reached LIFT_OK then fell below LIFT_MIN
while closed" also requires the target to separate from the gripper by SEP_M;
the literal rule fires on a held bowl lowered onto the plate (row 14 t=201).

Addition: a grasp onset while the target already satisfies the place relation
is grasp.post_place (not a failure), e.g. closes on a bowl already on the plate.

Known limitation: GR00T sometimes carries the bowl lower than LIFT_OK (5 cm),
so the grasp is called a slip and no place event follows; such a success shows
an "unrecovered" slip. Fitting LIFT_OK on demos (5th pct 8.7 cm, v1 §6) would
make this worse: humans lift higher than the policy.

Not implemented from §3.2: "or when the held target comes to rest" for
place.ok/off_target (only the release moment is used; V_DESC is a placeholder).

Two roots per failure (user, C4): first failure event in time (option b) and
first unrecovered failure event (option a).

Restore points (where the correction starts, §5.1):
  grasp.miss/slip/wrong_object  the onset step (close commanded, not yet applied)
  place.off_target              the release step (open commanded, bowl still held)
  transport.drop                detection step + SETTLE (the bowl has landed)
  approach.never_reached        the step of the smallest EEF-to-target distance
Phase for matching: place.off_target -> B (held), everything else -> A.

UNVERIFIED: new numeric code, not yet checked by val. Thresholds other than
R_GOAL and DZ are placeholders (v1 §6).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from restore import r047_rows  # noqa: E402

# v1 §6 starting values
LIFT_MIN = 0.015
LIFT_OK = 0.05
W_GRASP = 40
R_GOAL = 0.035
DZ_MIN, DZ_MAX = 0.0, 0.026
SETTLE = 10
SEP_M = 0.03
K = 6
# v1 §6: GRIP_R and FOLLOW_M "fit: demos", the 99th percentile over held windows
# (final grasp, from the bowl first reaching LIFT_OK to the release) of the 482
# usable demos. fit_grip() recomputes them; values from its 10-14 run. The v1
# placeholders (6 cm, 5 mm/step) rejected held bowls (demo median 5.6 cm).
GRIP_R, FOLLOW_M = 0.075, 0.0075


def fit_grip():
    """Recompute GRIP_R and FOLLOW_M from the demos (needs runs/dmp_poc/demos)."""
    import demos as D
    dist, fol = [], []
    for task in D.tasks():
        for d in D.load(task):
            if not d.usable:
                continue
            s, e = d.phases[1]
            z = d.target_pos[s:e, 2] - d.target_pos[s, 2]
            if not (z >= LIFT_OK).any():
                continue
            u = s + int(np.argmax(z >= LIFT_OK))
            dist += list(np.linalg.norm(d.target_pos[u:e] - d.eef_pos[u:e], axis=1))
            fol += list(np.linalg.norm(np.diff(d.target_pos[u:e], axis=0) - np.diff(d.eef_pos[u:e], axis=0), axis=1))
    return {"GRIP_R": float(np.percentile(dist, 99)), "FOLLOW_M": float(np.percentile(fol, 99)),
            "n_steps": len(dist)}
CLOSED = 0.5            # rollouts: action[6] > 0.5 is "commanded closed"

FAIL_MODES = ("grasp.miss", "grasp.slip", "grasp.wrong_object", "transport.drop",
              "place.off_target", "approach.never_reached")


@dataclass
class Event:
    t: int
    mode: str
    recovered_by: int | None = None
    restore_t: int | None = None
    phase: str | None = None
    measures: dict | None = None

    @property
    def failure(self):
        return self.mode in FAIL_MODES


def _arrays(row):
    S = row["steps"]
    names = list(S[0]["obs_state"]["_gt_eef_to_object"])   # BDDL order: target, goal
    main = lambda k: k if k.endswith("_main") else k + "_main"  # noqa: E731
    tgt, goal = main(names[0]), main(names[1])
    objs = list(S[0]["obs_state"]["_gt_scene_object_pos"])
    P = {k: np.array([s["obs_state"]["_gt_scene_object_pos"][k] for s in S]) for k in objs}
    eef = np.array([s["obs_state"]["eef_pos"] for s in S])
    cmd = np.array([s["action"] is not None and s["action"][6] > CLOSED for s in S])
    return S, tgt, goal, P, eef, cmd


def in_gripper(obj_pos, eef, cmd, t0, t1):
    """§3.7 over [t0, t1]: near the grip site, follows the EEF, commanded closed."""
    if t1 - t0 < 1:
        return False
    w = slice(t0, t1 + 1)
    if not cmd[w].all():
        return False
    if (np.linalg.norm(obj_pos[w] - eef[w], axis=1) > GRIP_R).any():
        return False
    follow = np.linalg.norm(np.diff(obj_pos[w], axis=0) - np.diff(eef[w], axis=0), axis=1).mean()
    return bool(follow <= FOLLOW_M)


def _held_somewhere(obj_pos, eef, cmd, a, b):
    """True if the object is in the gripper over some K-step window inside [a, b]."""
    return any(in_gripper(obj_pos, eef, cmd, u, u + K) for u in range(a, max(a, b - K) + 1))


def events(row) -> list[Event]:
    S, tgt, goal, P, eef, cmd = _arrays(row)
    n = len(S)
    tz = P[tgt][:, 2]
    others = [k for k in P if k not in (tgt, goal)]
    onsets = [t for t in range(1, n) if cmd[t] and not cmd[t - 1]]
    if cmd[0]:
        onsets = [0] + onsets
    ev: list[Event] = []
    if not onsets:
        d = np.linalg.norm(eef - P[tgt], axis=1)
        ev.append(Event(n - 1, "approach.never_reached", restore_t=int(d.argmin()), phase="A",
                        measures={"min_eef_to_target_m": float(d.min())}))
    for t in onsets:
        end = min(t + W_GRASP, n - 1)
        base = tz[t]
        rise = tz[t:end + 1] - base
        peak = float(rise.max())
        m = {"peak_lift_m": peak, "eef_to_target_m": float(np.linalg.norm(eef[t] - P[tgt][t]))}
        # a close on a bowl that already sits on the plate is not a grasp failure
        dxy0 = float(np.hypot(*(P[tgt][t, :2] - P[goal][t, :2])))
        dz0 = float(P[tgt][t, 2] - P[goal][t, 2])
        if dxy0 <= R_GOAL and DZ_MIN <= dz0 <= DZ_MAX:
            ev.append(Event(t, "grasp.post_place", measures=m | {"target_goal_xy_m": dxy0}))
            continue
        # a static object next to a still gripper "follows" it trivially, so a
        # non-target also has to rise >= LIFT_MIN in the window (v1: "a lift follows")
        wrong = [k for k in others
                 if P[k][t:end + 1, 2].max() - P[k][t, 2] >= LIFT_MIN and _held_somewhere(P[k], eef, cmd, t, end)]
        if wrong:
            ev.append(Event(t, "grasp.wrong_object", restore_t=t, phase="A", measures=m | {"object": wrong[0]}))
            continue
        if peak < LIFT_MIN:
            ev.append(Event(t, "grasp.miss", restore_t=t, phase="A", measures=m))
            continue
        t_ok = t + int(np.argmax(rise >= LIFT_OK)) if peak >= LIFT_OK else None
        # "rose then fell" counts only if the bowl also left the gripper (separation,
        # as for drop): a held bowl lowered onto the plate inside the window is not
        # a slip. Deviation from the literal v1 §3.2 rule, which misfires on that.
        fell = t_ok is not None and any(
            rise[u - t] < LIFT_MIN and cmd[u]
            and np.linalg.norm(eef[u] - P[tgt][u]) - np.linalg.norm(eef[t_ok] - P[tgt][t_ok]) > SEP_M
            for u in range(t_ok, end + 1))
        if t_ok is None or fell or not in_gripper(P[tgt], eef, cmd, t_ok, min(t_ok + K, n - 1)):
            ev.append(Event(t, "grasp.slip", restore_t=t, phase="A", measures=m))
            continue
        ev.append(Event(t, "grasp.ok", measures=m))
        # held: drop (separation while closed) or the first release
        d0 = np.linalg.norm(eef[t] - P[tgt][t])
        top = tz[t_ok]
        for u in range(t_ok, n):
            top = max(top, tz[u])
            if not cmd[u]:
                v = min(u + SETTLE, n - 1)
                dxy = float(np.hypot(*(P[tgt][v, :2] - P[goal][v, :2])))
                dz = float(P[tgt][v, 2] - P[goal][v, 2])
                ok = dxy <= R_GOAL and DZ_MIN <= dz <= DZ_MAX
                ev.append(Event(u, "place.ok" if ok else "place.off_target", restore_t=u - 1 if not ok else None,
                                phase=None if ok else "B", measures={"target_goal_xy_m": dxy, "dz_m": dz}))
                break
            sep = np.linalg.norm(eef[u] - P[tgt][u]) - d0
            if sep > SEP_M and top - tz[u] >= LIFT_MIN:
                ev.append(Event(u, "transport.drop", restore_t=min(u + SETTLE, n - 1), phase="A",
                                measures={"separation_m": float(sep)}))
                break
    ev.sort(key=lambda e: e.t)
    # R1: grasp.* recovered by a later grasp.ok; drop / off_target by a later place.ok
    for i, e in enumerate(ev):
        if not e.failure or e.mode == "approach.never_reached":
            continue
        fix = "grasp.ok" if e.mode.startswith("grasp.") else "place.ok"
        later = [f.t for f in ev[i + 1:] if f.mode == fix]
        e.recovered_by = later[0] if later else None
    return ev


def roots(row_idx: int) -> dict:
    row = r047_rows()[row_idx]
    ev = events(row)
    fails = [e for e in ev if e.failure]
    first = fails[0] if fails else None
    unrec = next((e for e in fails if e.recovered_by is None), None)
    return {"row": row_idx, "success": bool(row["success"]), "scene": row["task_id"].rsplit("task", 1)[1],
            "perturbation": row["perturbation"],
            "root_first": asdict(first) if first else None,
            "root_unrecovered": asdict(unrec) if unrec else None,
            "events": [asdict(e) for e in ev]}


def all_roots():
    """Every R-047 rollout; failures get roots, successes are the R1 self-check
    (a success with an unrecovered failure event is a rule bug, v1 §3.3)."""
    return [roots(i) for i in range(len(r047_rows()))]


if __name__ == "__main__":
    print("fit_grip (demos, p99):", fit_grip(), " in use:", GRIP_R, FOLLOW_M)
    out = all_roots()
    fails = [r for r in out if not r["success"]]
    bugs = [r["row"] for r in out if r["success"] and r["root_unrecovered"]]
    from collections import Counter
    print("failures", len(fails), " no event:", sum(r["root_first"] is None for r in fails))
    print("root_first      ", Counter(r["root_first"]["mode"] for r in fails if r["root_first"]))
    print("root_unrecovered", Counter(r["root_unrecovered"]["mode"] if r["root_unrecovered"] else None for r in fails))
    print("roots differ", sum((r["root_first"] or {}).get("t") != (r["root_unrecovered"] or {}).get("t") for r in fails))
    print("successes with an unrecovered failure (rule bugs):", len(bugs), bugs[:10])
    r14 = out[14]
    print("row 14:", [(e["t"], e["mode"], e["recovered_by"]) for e in r14["events"]])
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs/dmp_poc/roots.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
