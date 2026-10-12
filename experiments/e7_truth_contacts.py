"""E7 truth fix (10-09): the 'approach' truth (object nearest the grip site by BODY CENTRE) is wrong for bowls grasped
at the rim (~12 cm wide): L014 t=31 named the ramekin while the frame shows the target bowl's rim in the fingers.
This pass re-replays each rendered episode (same replay(), stored actions, one env at a time; frames are not saved)
and records, at every EVENT step of events.json, per movable object:
  min_pad_dist_m  minimum surface distance between the Panda fingerpad collision geoms and the object's contact geoms
                  (mujoco.mj_geomDistance, 0 = touching/penetrating, capped at 0.25 m);
  pad_contact     robosuite check_contact(fingerpad geoms, object).
New approach truth = the object with the smallest min_pad_dist_m, provided it is <= 0.03 m, else "no object".
Writes render/<ep>/truth_events.json. No model.
  bash experiments/d_step.sh e7_truth --gpu .venvs/libero-plus/bin/python experiments/e7_truth_contacts.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
import numpy as np

OUT = "runs/root_signals/e7"
RENDER = os.path.join(OUT, "render")
CAP = 0.25
APPROACH_MAX_M = 0.03


def pad_geoms(inner):
    gr = inner.robots[0].gripper
    names = gr.important_geoms["left_fingerpad"] + gr.important_geoms["right_fingerpad"]
    return [inner.sim.model.geom_name2id(n) for n in names]


def obj_geoms(inner, o):
    return [inner.sim.model.geom_name2id(n) for n in inner.get_object(o).contact_geoms]


def measure(inner, objs):
    import mujoco
    m, d = inner.sim.model._model, inner.sim.data._data
    pads = pad_geoms(inner)
    gr = inner.robots[0].gripper
    out = {}
    for o in objs:
        best = CAP
        for p in pads:
            for g in obj_geoms(inner, o):
                best = min(best, float(mujoco.mj_geomDistance(m, d, p, g, CAP, None)))
        pc = bool(inner.check_contact(gr.important_geoms["left_fingerpad"] + gr.important_geoms["right_fingerpad"], inner.get_object(o)))
        out[o] = {"min_pad_dist_m": max(best, 0.0), "pad_contact": pc}
    return out


def main():
    from vla_harness.envs.libero_env import LiberoEnv
    from robosuite.environments.base import MujocoEnv
    from vla_harness.schema import rollout_from_dict
    from visualise_episode_video import replay
    lines = open("runs/r047/rollouts.jsonl").readlines()
    eps = sorted(x for x in os.listdir(RENDER) if os.path.exists(os.path.join(RENDER, x, "events.json")))
    t0 = time.time()
    for ep in eps:
        p = os.path.join(RENDER, ep, "truth_events.json")
        if os.path.exists(p):
            continue
        info = json.load(open(os.path.join(RENDER, ep, "events.json")))
        want = {e["t_frame"] for e in info["events"]}
        r = rollout_from_dict(json.loads(lines[info["line"]]))
        gc.collect()
        alive = sum(1 for o in gc.get_objects() if isinstance(o, (LiberoEnv, MujocoEnv)))
        if alive:
            sys.exit(f"{ep}: {alive} env(s) alive before replay")
        rec, k = {}, {"t": 0}

        def on_step(env):
            t = k["t"]; k["t"] += 1
            if t in want:
                inner = env._env.unwrapped._env.env
                rec[t] = measure(inner, list(inner.objects_dict.keys()))
        _, fid = replay(r, on_step=on_step)
        res = {"ep": ep, "fidelity": fid, "steps": {str(t): v for t, v in sorted(rec.items())}, "approach_max_m": APPROACH_MAX_M,
               "missing_steps": sorted(want - set(rec))}
        json.dump(res, open(p, "w"), indent=1)
        print(f"[{time.time()-t0:6.0f}s] {ep} events {sorted(rec)}", flush=True)


if __name__ == "__main__":
    main()
