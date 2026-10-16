"""Past-row check for the 10-12 nominal_frames pose fix (user-approved, prim 10-12; g0c3/t10i). Report only, no GPU
re-runs of the policy. For every LIBERO-Plus CAMERA-variant row of R-039 (both selections) and R-048, in a fresh env
at the row's reset state, render the clean (nominal) frame two ways from the SAME state:
  old  original LIBERO's agentview pose, nominal.nominal_camera(domain)   (what those runs used)
  new  LIBERO-Plus's canonical pose, _bench_camera_pose(0, 0, 1.0)        (what nominal_frames uses since 10-12)
and the control env's own re-render (the frame the clean condition actually is), then report n_px (pixels differing in
any channel), n_diff (channel elements) and max_abs per row, agent and wrist cameras.
R-047 has no LIBERO-Plus camera variants (its camera axis is the harness knob, restored from a snapshot), so it is
listed as unaffected rather than rendered. EGL render only; no model.
  env $ENV flock /tmp/vla_gpu.lock .venvs/libero-plus/bin/python experiments/r060_pose_pastrows.py
Writes runs/r060/gates/tol8px/pose_pastrows.json.
"""
from __future__ import annotations

import gc
import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
import numpy as np

from r060_gates import OUT_TOL, _cmp_px, _rerender, catalogue, control_index, make_env  # noqa: E402

SOURCES = {"R-039": "experiments/repro/r039_selection.json",
           "R-039 recorded": "experiments/repro/r039_selection_recorded.json",
           "R-048": "experiments/repro/r048_selection_cam.json"}


def render_at_pose(env, pos, quat):
    from vla_harness.envs.libero_env import MAIN_CAMERA
    sim = env._env.unwrapped._env.env.sim
    cid = sim.model.camera_name2id(MAIN_CAMERA)
    p0, q0 = sim.model.cam_pos[cid].copy(), sim.model.cam_quat[cid].copy()
    try:
        sim.model.cam_pos[cid] = np.asarray(pos); sim.model.cam_quat[cid] = np.asarray(quat)
        return _rerender(env)
    finally:
        sim.model.cam_pos[cid] = p0; sim.model.cam_quat[cid] = q0; sim.forward()


def main():
    from vla_harness.envs import nominal
    from vla_harness.schema import PerturbationSpec
    cat = catalogue()
    rows, seen = [], {}
    for run, path in SOURCES.items():
        for x in json.load(open(path))["instances"]:
            if x.get("category") != "Camera Viewpoints":
                continue
            key = (int(x["task_id"]), int(x.get("seed", 0)))
            if key in seen:
                seen[key]["runs"].append(run)
                continue
            seen[key] = {"runs": [run], "task_id": key[0], "seed": key[1], "variant": cat[key[0]]["name"]}
    for key, r in seen.items():
        gc.collect()
        v = make_env(r["task_id"]); c = make_env(control_index(cat, r["variant"]))
        v.reset(r["seed"], PerturbationSpec()); c.reset(r["seed"], PerturbationSpec())
        inner = v._env.unwrapped._env.env
        dom = type(inner).__module__.rsplit(".", 1)[-1].removeprefix("libero_").removesuffix("_manipulation")
        old = render_at_pose(v, *nominal.nominal_camera(dom))
        new = render_at_pose(v, *v._bench_camera_pose(0, 0, 1.0))
        ctl = _rerender(c)
        r.update({"control": control_index(cat, r["variant"]), "domain": dom,
                  "old_vs_new": {k: _cmp_px(old[i], new[i]) for k, i in (("agent", "image"), ("wrist", "image2"))},
                  "new_vs_control_rerender": {k: _cmp_px(new[i], ctl[i]) for k, i in (("agent", "image"), ("wrist", "image2"))},
                  "old_vs_control_rerender": {k: _cmp_px(old[i], ctl[i]) for k, i in (("agent", "image"), ("wrist", "image2"))},
                  "nominal_frames_now_vs_new": _cmp_px(v.nominal_frames()["image"], new["image"])})
        rows.append(r)
        print(r["runs"], r["task_id"], r["variant"].split("_view_")[1], "old-vs-new agent px", r["old_vs_new"]["agent"]["n_px"],
              "max", r["old_vs_new"]["agent"]["max_abs"], "| new-vs-ctl", r["new_vs_control_rerender"]["agent"]["n_px"],
              "| old-vs-ctl", r["old_vs_control_rerender"]["agent"]["n_px"], flush=True)
        del v, c
    out = {"tol8px": True, "what": "past-row pose check: clean frame at the old (LIBERO) vs new (LIBERO-Plus) agentview pose, same reset state",
           "unaffected": {"R-047": "no LIBERO-Plus camera variants (camera axis = harness knob, snapshot restore)"},
           "n_rows": len(rows), "rows": rows}
    os.makedirs(OUT_TOL, exist_ok=True)
    json.dump(out, open(os.path.join(OUT_TOL, "pose_pastrows.json"), "w"), indent=1, default=str)
    print("wrote", os.path.join(OUT_TOL, "pose_pastrows.json"))


if __name__ == "__main__":
    main()
