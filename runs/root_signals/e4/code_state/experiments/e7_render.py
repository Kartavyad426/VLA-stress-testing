"""E7 step (b): render the frames the VLM is asked about, and the sim truth it is scored against, by REPLAYING stored
R-047 actions (no policy, no model; experiments/visualise_episode_video.replay, as poc_toy_render.py does).
Approved 10-09 (user via prim). No result travels before a PASS entry in PIPELINE_CHECKS.md.

  ENV="PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl"
  env $ENV .venvs/libero-plus/bin/python experiments/e7_render.py select        # CPU: episode list
  bash experiments/d_step.sh e7_render --gpu .venvs/libero-plus/bin/python experiments/e7_render.py render [--limit N]

Episodes: every R-047 FAILURE on the start-pose and camera-yaw axes (109 + 61 = 170), plus a success calibration
sample of 50 (seeded, stratified by axis and scene: 25 start pose, 25 camera). Rows are selected by LINE INDEX in
runs/r047/rollouts.jsonl (rollout_id is shared across noise seeds) and the (task, perturbation, success, env_steps)
of each is asserted on replay.

Telemetry events (from the STORED executed actions, LIBERO gripper command a[6]: -1 open, +1 close):
  grasp onset  t with a[t][6] > 0 and a[t-1][6] <= 0 (or t = 0);
  onset+10     onset + 10 steps (clipped to the episode);
  release      the first t after an onset with a[t][6] <= 0;
  final        the last frame.
Up to 2 onset/release cycles per episode (call budget; the first cycle holds the PoC root). Frame t is the observation BEFORE action t (replay's own alignment), shown
as the policy saw it (180-degree flip), agent view | wrist view side by side.

Sim truth, recorded during the replay at every frame (aligned with the rendered frame, not with R-047's log):
  object positions (body_xpos) of every movable BDDL object; `held` = objects for which robosuite's
  _check_grasp(gripper, object) is true; `target_on_plate` = LIBERO's own BDDL predicate
  object_states_dict[target].check_ontop(plate). Cross-check: the replayed positions vs R-047's logged
  _gt_scene_object_pos at the same step (invariant, reported).
Outputs runs/root_signals/e7/: episodes.json, render/<ep>/{events.json, truth.jsonl, frames.mp4, ev_<t>.png},
render_summary.json, code_state/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
import numpy as np

OUT = "runs/root_signals/e7"
RUN = "runs/r047/rollouts.jsonl"
N_SUCC = {"joint_radius_rad": 25, "camera_yaw_deg": 25}
MAX_CYCLES = 2


def log(*x, t0=time.time()):
    print(f"[{time.time() - t0:7.0f}s]", *x, flush=True)


def code_state():
    d = os.path.join(OUT, "code_state")
    os.makedirs(d, exist_ok=True)
    os.system(f"git rev-parse HEAD > {d}/HEAD; git diff > {d}/uncommitted.patch; git status --porcelain > {d}/status.txt; "
              f"sha256sum {RUN} > {d}/inputs.sha256; "
              f"git ls-files --others --exclude-standard experiments vla_harness | grep '\\.py$' | xargs -r -I{{}} cp --parents {{}} {d}/")


def axis_of(p):
    return "joint_radius_rad" if "joint_radius_rad" in p else ("camera_yaw_deg" if "camera_yaw_deg" in p else "nominal")


def select(a):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for i, l in enumerate(open(RUN)):
        d = json.loads(l)
        rows.append({"line": i, "rollout_id": d["rollout_id"], "task_id": d["task_id"], "perturbation": d["perturbation"],
                     "success": d["success"], "env_steps": d["env_steps"], "axis": axis_of(d["perturbation"]),
                     "seed": d["seed"]})
    fails = [r for r in rows if not r["success"] and r["axis"] in N_SUCC]
    rng = np.random.default_rng(7)
    succ = []
    for ax, n in N_SUCC.items():
        pool = [r for r in rows if r["success"] and r["axis"] == ax]
        by_scene = {}
        for r in pool:
            by_scene.setdefault(r["task_id"], []).append(r)
        order = []                                     # round-robin over scenes, random within scene
        lists = [list(rng.permutation(v)) for _, v in sorted(by_scene.items())]
        while len(order) < n and any(lists):
            for L in lists:
                if L and len(order) < n:
                    order.append(L.pop())
        succ += order
    eps = [{**r, "role": "failure"} for r in fails] + [{**r, "role": "success_calibration"} for r in succ]
    for e in eps:
        e["ep"] = f"L{e['line']:03d}"
    excluded = [r["line"] for r in rows if not r["success"] and r["axis"] not in N_SUCC]
    json.dump({"n": len(eps), "n_failures": len(fails), "n_success_sample": len(succ),
               "excluded_failures": {"lines": excluded, "why": "magnitude-0 (nominal) failures are outside the start-pose and camera-yaw axes"},
               "by_axis_role": {f"{ax}|{ro}": sum(1 for e in eps if e["axis"] == ax and e["role"] == ro)
                                for ax in N_SUCC for ro in ("failure", "success_calibration")},
               "episodes": eps}, open(os.path.join(OUT, "episodes.json"), "w"), indent=1)
    print(len(fails), "failures,", len(succ), "successes")


def events(actions):
    """Telemetry events from the executed gripper commands."""
    g = [a[6] for a in actions]
    T = len(g)
    ev, cycles, t = [], 0, 0
    while t < T and cycles < MAX_CYCLES:
        on = next((u for u in range(t, T) if g[u] > 0 and (u == 0 or g[u - 1] <= 0)), None)
        if on is None:
            break
        ev.append(("grasp_onset", on))
        ev.append(("onset_plus_10", min(on + 10, T)))
        rel = next((u for u in range(on + 1, T) if g[u] <= 0), None)
        if rel is None:
            break
        ev.append(("release", rel))
        cycles, t = cycles + 1, rel + 1
    ev.append(("final", T))                       # frame T = after the last action
    out, seen = [], set()
    for k, s in ev:
        if (k, s) not in seen:
            out.append({"event": k, "t": int(s)}); seen.add((k, s))
    return sorted(out, key=lambda x: (x["t"], x["event"]))


def names_and_target(env):
    inner = env._env.unwrapped._env.env
    objs = list(inner.objects_dict.keys())
    ooi = list(inner.obj_of_interest)
    target = next(o for o in ooi if "plate" not in o)
    plate = next(o for o in ooi if "plate" in o)
    return inner, objs, target, plate


def truth_step(inner, objs, target, plate):
    gr = inner.robots[0].gripper
    pos = {o: [float(x) for x in inner.sim.data.body_xpos[inner.obj_body_id[o]]] for o in objs}
    held = [o for o in objs if inner._check_grasp(gr, inner.get_object(o))]
    on = bool(inner.object_states_dict[target].check_ontop(inner.object_states_dict[plate]))
    eef = [float(x) for x in inner.sim.data.site_xpos[inner.robots[0].eef_site_id]]
    return {"pos": pos, "held": held, "target_on_plate": on, "eef": eef}


def render(a):
    import imageio.v2 as imageio
    from PIL import Image
    from vla_harness.schema import rollout_from_dict
    from visualise_episode_video import replay
    code_state()
    E = json.load(open(os.path.join(OUT, "episodes.json")))["episodes"]
    if a.limit:
        E = E[: a.limit]
    lines = open(RUN).readlines()
    summ_p = os.path.join(OUT, "render_summary.json")
    summ = json.load(open(summ_p)) if os.path.exists(summ_p) else {}
    for e in E:
        d = os.path.join(OUT, "render", e["ep"])
        if e["ep"] in summ and os.path.exists(os.path.join(d, "events.json")):
            continue                                                     # resumable
        os.makedirs(d, exist_ok=True)
        raw = json.loads(lines[e["line"]])
        got = (raw["rollout_id"], raw["task_id"], raw["perturbation"], raw["success"], raw["env_steps"])
        exp = (e["rollout_id"], e["task_id"], e["perturbation"], e["success"], e["env_steps"])
        if got != exp:
            sys.exit(f"{e['ep']}: rollouts.jsonl line changed: {got} vs {exp}")
        r = rollout_from_dict(raw)
        truth, meta = [], {}
        # G0c c5: deleting an env corrupts a surviving env's renders until its next reset. One env alive at a time:
        # the previous episode's env is released here, before replay() builds (and resets) the next one.
        import gc
        from vla_harness.envs.libero_env import LiberoEnv
        gc.collect()
        from robosuite.environments.base import MujocoEnv
        alive = sum(1 for o in gc.get_objects() if isinstance(o, (LiberoEnv, MujocoEnv)))
        if alive:
            sys.exit(f"{e['ep']}: {alive} LiberoEnv still alive before replay (one env at a time required)")

        def on_step(env):
            if not meta:
                meta["inner"], meta["objs"], meta["target"], meta["plate"] = names_and_target(env)
                meta["instruction"] = env.instruction
            truth.append(truth_step(meta["inner"], meta["objs"], meta["target"], meta["plate"]))
        cams, fid = replay(r, on_step=on_step)
        acts = [s.action for s in r.steps if s.action is not None]
        ev = events(acts)
        # cross-check replayed positions vs the logged _gt_scene_object_pos
        dev = []
        for t, s in enumerate(r.steps[: len(truth)]):
            g = s.obs_state.get("_gt_scene_object_pos") or {}
            for o, p in truth[t]["pos"].items():
                q = g.get(o + "_main") or g.get(o)
                if q is not None:
                    dev.append(float(np.linalg.norm(np.array(p) - np.array(q))))
        frames = [np.concatenate([cams[0][t], cams[1][t]], axis=1) for t in range(len(cams[0]))]
        for x in ev:
            x["t_frame"] = min(x["t"], len(frames) - 1)
            Image.fromarray(frames[x["t_frame"]]).save(os.path.join(d, f"ev_{x['t_frame']:03d}.png"))
        imageio.mimwrite(os.path.join(d, "frames.mp4"), frames, fps=20, codec="libx264", quality=9, macro_block_size=1)
        with open(os.path.join(d, "truth.jsonl"), "w") as fo:
            for t, x in enumerate(truth):
                fo.write(json.dumps({"t": t, **x}) + "\n")
        info = {"envs_alive_before_replay": alive, "frames_source": "env._raw step-returned pixels (replay), no re-render",
                "ep": e["ep"], "line": e["line"], "role": e["role"], "axis": e["axis"], "task_id": e["task_id"],
                "success": e["success"], "instruction": meta.get("instruction"), "objects": meta.get("objs"),
                "target": meta.get("target"), "plate": meta.get("plate"), "events": ev, "n_frames": len(frames),
                "n_truth": len(truth), "fidelity": fid,
                "replay_vs_logged_obj_pos_m": {"max": max(dev) if dev else None, "median": float(np.median(dev)) if dev else None},
                "has_grasp_event": any(x["event"] == "grasp_onset" for x in ev)}
        json.dump(info, open(os.path.join(d, "events.json"), "w"), indent=1)
        summ[e["ep"]] = {k: info[k] for k in ("role", "axis", "success", "n_frames", "has_grasp_event")}
        summ[e["ep"]]["max_dev_m"] = fid.get("max_dev_m")
        json.dump(summ, open(summ_p, "w"), indent=1)
        meta.clear(); del cams, frames                  # drop every reference to this episode's env before the next
        log(e["ep"], e["role"], e["axis"], "events", [(x["event"], x["t"]) for x in ev], "fid", fid.get("max_dev_m"),
            "objdev", info["replay_vs_logged_obj_pos_m"]["max"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["select", "render"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    {"select": select, "render": render}[a.phase](a)


if __name__ == "__main__":
    main()
