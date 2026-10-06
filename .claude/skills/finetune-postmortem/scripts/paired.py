"""Paired per-start analysis of a fine-tune eval: base vs one or more adapter arms on identical starts.

  .venvs/groot/bin/python .claude/skills/finetune-postmortem/scripts/paired.py \
      --run runs/r056_r16=r16 --run runs/r058_r4=r4 \
      [--nominal-starts runs/r047/nominal_starts.json] [--offset-cm 8] [--group scene] \
      [--out runs/<x>/postmortem_paired.json]

Reads each run's manifest.jsonl + rollouts.jsonl and JOINS THEM BY POSITION (the harness's
rollout_id ignores the noise seed, so an id join silently merges seeds). Base rows come from
whichever run has ckpt == "base"; every other ckpt on the heldout/nominal sets becomes the arm
named after "=". Starts are keyed by (set, scene, dir_seed, radius, noise_seed, env_seed).

Prints, per arm vs base:
  * pooled paired 2x2 (lost = base S / arm F, gained = base F / arm S) with an exact McNemar p;
  * the per-group table (default group = scene) and the leave-one-group-out paired test, which
    shows whether the pooled effect is one group;
  * an offset split (gripper offset at t=0 from the scene's nominal start, needs --nominal-starts)
    with the dir_seed/scene composition of each side, because offset is usually confounded with both;
  * failure mode of lost starts: never lifted the target vs lifted then failed;
  * first-chunk shift: the arm's commanded translation over the first 16 actions, compared with
    base's and with the base NOMINAL rollout's (same scene, noise seed 0), i.e. did the arm move
    toward the nominal chunk and by what fraction;
  * nominal-set failures per arm.
The target object is the FIRST key of steps[0].obs_state._gt_object_pos (LIBERO spatial: the bowl).
"""
from __future__ import annotations

import argparse
import collections
import json
from math import comb

import numpy as np

SETS = ("heldout", "nominal")
LIFT_M = 0.03


def mcnemar(lost: int, gained: int) -> float:
    n = lost + gained
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(lost, gained) + 1)) / 2 ** n)


def load(run: str):
    man = [json.loads(l) for l in open(f"{run}/manifest.jsonl")]
    with open(f"{run}/rollouts.jsonl") as fh:
        for m, line in zip(man, fh):
            r = json.loads(line)
            if r["rollout_id"] != m["rollout_id"] or r["success"] != m["success"]:
                raise SystemExit(f"{run}: manifest and rollouts are not aligned by position")
            yield m, r


def features(r, nominal=None, scene=None):
    st = r["steps"]
    s0 = st[0]["obs_state"]
    target = list(s0["_gt_object_pos"])[0]
    z0 = s0["_gt_object_pos"][target][2]
    zs = [s["obs_state"]["_gt_object_pos"][target][2] for s in st]
    acts = np.array([s["action"] for s in st if s.get("action") is not None])
    f = {"success": bool(r["success"]), "steps": r["env_steps"],
         "lift_t": next((s["t"] for s, z in zip(st, zs) if z - z0 > LIFT_M), None),
         "a0": acts[:16, :3].sum(0).tolist() if len(acts) else None,
         "eef": [s["obs_state"]["eef_pos"] for s in st]}
    if nominal and str(scene) in nominal:
        n = nominal[str(scene)]
        f["offset_m"] = float(np.linalg.norm(np.subtract(s0["eef_pos"], n["eef_pos"])))
        if "joint_pos" in n and "joint_pos" in s0:
            f["joint_radius"] = float(np.linalg.norm(np.subtract(s0["joint_pos"], n["joint_pos"])))
    return f


def build(runs, nominal):
    rows = collections.defaultdict(dict)
    for run, label in runs:
        for m, r in load(run):
            if m["set"] not in SETS:
                continue
            key = (m["set"], m["scene"], m.get("dir_seed"), m.get("radius"), m.get("noise_seed"), m.get("env_seed"))
            arm = "base" if m["ckpt"] == "base" else label
            if arm in rows[key] and arm != "base":
                raise SystemExit(f"two {arm} rollouts for {key}: pass one checkpoint per run")
            rows[key][arm] = features(r, nominal, m["scene"])
    return rows


def paired(rows, arm, keep=lambda k, v: True):
    sub = {k: v for k, v in rows.items() if "base" in v and arm in v and keep(k, v)}
    lost = sum(1 for v in sub.values() if v["base"]["success"] and not v[arm]["success"])
    gained = sum(1 for v in sub.values() if not v["base"]["success"] and v[arm]["success"])
    return {"n": len(sub), "base": sum(v["base"]["success"] for v in sub.values()),
            "arm": sum(v[arm]["success"] for v in sub.values()), "lost": lost, "gained": gained,
            "p": mcnemar(lost, gained)}


def fmt(d):
    return (f"n={d['n']:3d} base {d['base']:3d} arm {d['arm']:3d}  lost {d['lost']:2d} gained {d['gained']:2d}"
            f"  exact McNemar p={d['p']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="RUN_DIR=label (label names the adapter arm)")
    ap.add_argument("--nominal-starts", default=None)
    ap.add_argument("--offset-cm", type=float, default=8.0)
    ap.add_argument("--group", default="scene", choices=["scene", "dir_seed", "radius", "noise_seed"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = [(x.split("=")[0].rstrip("/"), x.split("=")[1] if "=" in x else x.rstrip("/").split("/")[-1]) for x in a.run]
    nominal = json.load(open(a.nominal_starts)) if a.nominal_starts else None
    rows = build(runs, nominal)
    arms = sorted({arm for v in rows.values() for arm in v if arm != "base"})
    gi = {"scene": 1, "dir_seed": 2, "radius": 3, "noise_seed": 4}[a.group]
    H = {k: v for k, v in rows.items() if k[0] == "heldout"}
    out = {"arms": arms, "pooled": {}, "groups": {}, "leave_one_out": {}, "offset": {}, "failure_mode": {},
           "first_chunk": {}, "nominal_failures": {}}

    print("== pooled held-out, paired")
    for arm in arms:
        d = paired(H, arm); out["pooled"][arm] = d; print(f"{arm:>6} {fmt(d)}")

    groups = sorted({k[gi] for k in H}, key=str)
    print(f"\n== per {a.group}: successes (n)   base | " + " | ".join(arms))
    for g in groups:
        sub = {k: v for k, v in H.items() if k[gi] == g}
        cells = [sum(v[x]["success"] for v in sub.values() if x in v) for x in ["base"] + arms]
        out["groups"][str(g)] = dict(zip(["base"] + arms, cells), n=len(sub))
        print(f"{str(g):>8} ({len(sub):3d})  " + " | ".join(f"{c:3d}" for c in cells))

    print(f"\n== per {a.group}: paired test inside the group and with the group left out")
    for arm in arms:
        for g in groups:
            ins = paired(H, arm, lambda k, v, g=g: k[gi] == g)
            if ins["p"] < 0.05 or ins["lost"] + ins["gained"] >= 6:
                loo = paired(H, arm, lambda k, v, g=g: k[gi] != g)
                out["leave_one_out"].setdefault(arm, {})[str(g)] = {"inside": ins, "without": loo}
                print(f"{arm:>6} {a.group} {g}: inside {fmt(ins)}\n{'':>6} {'':>{len(a.group)}} {'':>{len(str(g))}}  without {fmt(loo)}")

    if nominal:
        thr = a.offset_cm / 100
        print(f"\n== offset split at {a.offset_cm:g} cm (composition shows confounds)")
        for side, test in [("le", lambda k, v: v["base"].get("offset_m", 1e9) <= thr),
                           ("gt", lambda k, v: v["base"].get("offset_m", -1) > thr)]:
            sub = {k: v for k, v in H.items() if test(k, v)}
            comp = {"dir_seed": dict(collections.Counter(k[2] for k in sub)),
                    a.group: dict(collections.Counter(str(k[gi]) for k in sub))}
            out["offset"][side] = {"composition": comp}
            print(f"  {side} {a.offset_cm:g} cm: n={len(sub)} dir_seeds {comp['dir_seed']}")
            for arm in arms:
                d = paired(H, arm, test); out["offset"][side][arm] = d; print(f"  {arm:>6} {fmt(d)}")
        by_dir = collections.defaultdict(list)
        for k, v in H.items():
            if "joint_radius" in v["base"] and k[3]:
                by_dir[k[2]].append((v["base"]["joint_radius"] / k[3], v["base"]["offset_m"]))
        if by_dir:
            print("  achieved/requested joint radius and median offset by direction:")
            for d_, L in sorted(by_dir.items(), key=str):
                A = np.array(L); print(f"    dir {d_}: ratio {A[:, 0].mean():.2f} [{A[:, 0].min():.2f},{A[:, 0].max():.2f}]"
                                       f"  offset median {np.median(A[:, 1]) * 100:.1f} cm")

    print("\n== lost starts: failure mode")
    for arm in arms:
        fl = [v[arm] for v in H.values() if arm in v and v["base"]["success"] and not v[arm]["success"]]
        nl = sum(1 for x in fl if x["lift_t"] is None)
        out["failure_mode"][arm] = {"lost": len(fl), "never_lifted": nl, "lifted_then_failed": len(fl) - nl}
        print(f"{arm:>6} lost {len(fl)}: never lifted {nl}, lifted then failed {len(fl) - nl}")

    nomA = {k[1]: v["base"]["a0"] for k, v in rows.items() if k[0] == "nominal" and k[4] == 0 and "base" in v}
    if nomA:
        print("\n== first chunk (commanded translation over 16 steps), held-out, noise seed 0")
        for arm in arms:
            L = []
            for k, v in H.items():
                if k[4] != 0 or arm not in v or k[1] not in nomA:
                    continue
                n0 = np.array(nomA[k[1]]); b = np.array(v["base"]["a0"]); x = np.array(v[arm]["a0"])
                L.append((np.linalg.norm(b - n0), np.linalg.norm(x - n0), np.linalg.norm(x - b)))
            if not L:
                continue
            A = np.array(L)
            d = {"n": len(A), "base_to_nominal": float(np.median(A[:, 0])), "arm_to_nominal": float(np.median(A[:, 1])),
                 "arm_to_base": float(np.median(A[:, 2])), "arm_closer": int((A[:, 1] < A[:, 0]).sum())}
            d["fraction_closed"] = 1 - d["arm_to_nominal"] / d["base_to_nominal"]
            out["first_chunk"][arm] = d
            print(f"{arm:>6} |base-nom| {d['base_to_nominal']:.3f}  |arm-nom| {d['arm_to_nominal']:.3f}  "
                  f"|arm-base| {d['arm_to_base']:.3f}  closer in {d['arm_closer']}/{d['n']}  "
                  f"fraction of distance closed {d['fraction_closed']:.2f}")
        for arm in arms:
            for t in (16, 32, 48):
                dd = [np.linalg.norm(np.subtract(v[arm]["eef"][min(t, len(v[arm]["eef"]) - 1)],
                                                 v["base"]["eef"][min(t, len(v["base"]["eef"]) - 1)]))
                      for v in H.values() if arm in v]
                print(f"{arm:>6} eef |arm-base| at t={t}: median {np.median(dd) * 100:.2f} cm, p90 {np.percentile(dd, 90) * 100:.2f}")

    print("\n== nominal-set failures (scene, noise_seed, arm)")
    for arm in ["base"] + arms:
        f = [(k[1], k[4]) for k, v in rows.items() if k[0] == "nominal" and arm in v and not v[arm]["success"]]
        n = sum(1 for k, v in rows.items() if k[0] == "nominal" and arm in v)
        out["nominal_failures"][arm] = {"n": n, "failures": f}
        print(f"{arm:>6} {n - len(f)}/{n}  failures {f}")

    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=str)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
