"""Run a policy (policy_rollout.py) on every task variant the scripted expert solved (make_tasks.py output) and
aggregate success + stages (checklist #23) per task. Each episode is its own policy_rollout call (fresh policy server);
already-finished episodes (rollout.json present) are skipped, so the batch can be resumed.

Counting rule (pre-deploy checklist E): only expert-solved variants enter; a failure is "unattributed" until the
scene passes the checklist and the policy sanity checks.

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/policy_batch.py \
        --tasks runs/real2sim/tasks_stereo --scene runs/real2sim/ep_orange_block_stereo:data/real2sim/droid/ep_orange_block \
        --out runs/real2sim/policy/batch_orange_stereo
"""
import argparse, glob, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="make_tasks --out dir (summary.json + <scene>/<task>/v<n>.json)")
    ap.add_argument("--scene", required=True, help="<scene out dir>:<episode dir>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="groot")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--exclude", action="append", default=[], help="skip tasks whose instruction contains this (e.g. radish)")
    ap.add_argument("--trivial_from", default=None, help="batch_summary.json of a --policy hold run: skip variants a do-nothing "
                    "policy already 'solves' (task true at the start; found 10-19 on a next-to task)")
    a = ap.parse_args()
    sdir = a.scene.split(":")[0]
    scene = os.path.basename(os.path.normpath(sdir))
    summary = [t for t in json.load(open(os.path.join(a.tasks, "summary.json"))) if t["scene"] == scene]
    if not summary:
        raise SystemExit(f"no tasks for scene {scene} in {a.tasks}")
    os.makedirs(a.out, exist_ok=True)
    trivial = set()
    if a.trivial_from:
        trivial = {(e["task"], e["variant"]) for e in json.load(open(a.trivial_from))["episodes"] if e.get("success")}
    rows = []
    for t in summary:
        if any(x in t["task"] for x in a.exclude):
            continue
        for vf in sorted(glob.glob(os.path.join(a.tasks, t["dir"], "v*.json"))):
            v = json.load(open(vf))
            if not v.get("success"):
                continue  # only variants the expert solves
            if (t["task"], v["variant"]) in trivial:
                print(f"skip {t['task']} v{v['variant']}: already true for a do-nothing policy", flush=True)
                continue
            ep_dir = os.path.join(a.out, os.path.basename(t["dir"]), f"v{v['variant']}")
            rj = os.path.join(ep_dir, "rollout.json")
            if not os.path.exists(rj):
                t0 = time.time()
                cmd = [sys.executable, os.path.join(HERE, "policy_rollout.py"), "--scene", a.scene, "--task", t["task"],
                       "--variant", str(v["variant"]), "--seed", str(v.get("seed", 0)), "--out", ep_dir,
                       "--policy", a.policy, "--seconds", str(a.seconds)]
                p = subprocess.run(cmd, capture_output=True, text=True)
                if p.returncode != 0:
                    print(f"FAILED to run {t['task']} v{v['variant']}: rc {p.returncode}\n{p.stderr[-1500:]}", flush=True)
                    rows.append(dict(task=t["task"], type=t["type"], variant=v["variant"], error=p.returncode))
                    continue
                print(f"{t['task']} v{v['variant']}: done in {time.time() - t0:.0f} s", flush=True)
            r = json.load(open(rj))
            rows.append(dict(task=t["task"], type=t["type"], variant=v["variant"], success=r["success"],
                             stages={k: r["stages"][k] for k in ("reached", "closed_at_target", "lifted", "placed")} if r.get("stages") else None,
                             blocked=r.get("tracking", {}).get("blocked"), gripper_closed_frac=r.get("gripper_closed_frac")))
            print(f"  -> success {r['success']} stages {rows[-1]['stages']} blocked {rows[-1]['blocked']}", flush=True)
    by_task = {}
    for r in rows:
        bt = by_task.setdefault(r["task"], dict(type=r["type"], n=0, success=0, reached=0, closed_at_target=0, lifted=0, placed=0, blocked=0, errors=0))
        bt["n"] += 1
        if "error" in r:
            bt["errors"] += 1
            continue
        bt["success"] += int(r["success"])
        for k in ("reached", "closed_at_target", "lifted", "placed"):
            bt[k] += int(bool(r["stages"] and r["stages"][k]))
        bt["blocked"] += int(bool(r["blocked"]))
    tot = {k: sum(b[k] for b in by_task.values()) for k in ("n", "success", "reached", "closed_at_target", "lifted", "placed", "blocked", "errors")}
    json.dump(dict(scene=a.scene, tasks=a.tasks, policy=a.policy, seconds=a.seconds, trivial_excluded=sorted(trivial), total=tot, by_task=by_task, episodes=rows,
                   status="exploratory, UNVERIFIED"), open(os.path.join(a.out, "batch_summary.json"), "w"), indent=1)
    print("TOTAL", json.dumps(tot))
    for k, b in by_task.items():
        print(f"  {b['success']}/{b['n']} {k}  (reached {b['reached']}, closed {b['closed_at_target']}, lifted {b['lifted']}, placed {b['placed']}, blocked {b['blocked']})")


if __name__ == "__main__":
    main()
