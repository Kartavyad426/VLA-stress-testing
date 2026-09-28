"""R-047 runner: success probability against magnitude, fixed grid, replicates.

  PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config \
  MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  flock /tmp/vla_gpu.lock .venvs/libero-plus/bin/python experiments/r047_run.py --run-id r047 --axis joint_radius_rad

Env seed is 0 on every rollout (same layout as R-039 onward). Replicates
vary only the fixed denoising noise (noise seeds) and, for the state axis,
the joint direction. Every rollout carries the base splice arms (paired
render for the render axes; recorded magnitude-0 source per (scene, dir,
noise) for the state axis) so ‖P−N‖ at forward 0 is on record.

Writes runs/<run-id>/<axis>/manifest.jsonl (one row per rollout) and the
per-rollout npz; resumable at rollout granularity.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.getcwd())

import numpy as np
import torch

from vla_harness.capture.splice import ARMS, attach_recorder, attach_splice, transfer_fraction
from vla_harness.envs.libero_env import LiberoEnv
from vla_harness.policies.lerobot_policy import LeRobotPolicy
from vla_harness.runner import rollout
from vla_harness.schema import PerturbationSpec, TraceStore, knob_neutral

LIVE_DIMS = 7
ENV_SEED = 0
SCENES = [984, 1030, 1062, 1090, 1132, 1169, 1201, 1247, 1282, 1327]
GRIDS = {  # axis -> (magnitudes, replicate spec)
    "joint_radius_rad": ([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], {"dirs": [0, 1, 2], "noise": [0, 1]}),
    "camera_yaw_deg": ([0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0], {"dirs": [0], "noise": [0, 1, 2, 3]}),
    "camera_dist_m": ([0.0, 0.1, 0.2, 0.4, 0.6, 0.8], {"dirs": [0], "noise": [0, 1, 2, 3]}),
    # LIBERO-Plus's own camera geometry (camera_bench_*). Ranges from the Camera
    # Viewpoints rows of docs/libero_plus_variants/*.csv, identical in all four
    # suites: azimuth -75..75 (stored 285..359 for negative), elevation 0 or 15
    # only, distance 100..200 %. Yaw reuses camera_yaw_deg's magnitudes so the
    # two geometries compare point for point; pitch 5/10 interpolate below the
    # benchmark's one value. Scale's nominal is 1.0, not 0.
    "camera_bench_yaw_deg": ([0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0], {"dirs": [0], "noise": [0, 1, 2, 3]}),
    "camera_bench_pitch_deg": ([0.0, 5.0, 10.0, 15.0], {"dirs": [0], "noise": [0, 1, 2, 3]}),
    "camera_bench_scale": ([1.0, 1.2, 1.4, 1.6, 1.8, 2.0], {"dirs": [0], "noise": [0, 1, 2, 3]}),
}


def build_policy():
    from lerobot.envs.configs import LiberoPlusEnv
    return LeRobotPolicy("nvidia/gr00t17-lerobot-libero_spatial-640", n_action_steps=16,
                         env_cfg=LiberoPlusEnv(task="libero_spatial"),
                         policy_overrides={"base_model_path": "nvidia/GR00T-N1.7-3B", "embodiment_tag": "libero_sim"},
                         dtype="bfloat16", rename_map={"observation.images.image2": "observation.images.wrist_image"})


def make_env(task_id):
    return LiberoEnv(suite="libero_spatial", task_id=task_id, libero_plus=True,
                     libero_plus_base_instruction=True, obs_size=360)


def spec_for(axis, m, d):
    if m == knob_neutral(axis):
        return PerturbationSpec()
    if axis == "joint_radius_rad":
        return PerturbationSpec.of(joint_radius_rad=float(m), joint_dir_seed=int(d))
    return PerturbationSpec.of(**{axis: float(m)})


def closest_to_bowl(r, env):
    try:
        targets = [t for t in env._env.unwrapped._env.obj_of_interest if "bowl" in t]
    except Exception:
        targets = []
    out = []
    for st in r.steps:
        dd = st.obs_state.get("_gt_eef_to_object") or {}
        pool = {k: v for k, v in dd.items() if any(k.startswith(t) or t.startswith(k) for t in targets)} or dd
        out.append(min(pool.values()) if pool else float("nan"))
    return np.array(out, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--axis", required=True, choices=list(GRIDS))
    ap.add_argument("--scenes", default=",".join(map(str, SCENES)))
    ap.add_argument("--max-rollouts", type=int, default=None, help="smoke: stop after N new rollouts")
    a = ap.parse_args()
    n_new = 0
    scenes = [int(s) for s in a.scenes.split(",")]
    mags, rep = GRIDS[a.axis]
    is_state = a.axis == "joint_radius_rad"
    out_dir = os.path.join("runs", a.run_id, a.axis)
    os.makedirs(out_dir, exist_ok=True)
    code_dir = os.path.join("runs", a.run_id, "code_state")
    if not os.path.exists(code_dir):
        os.makedirs(code_dir)
        os.system(f"git rev-parse HEAD > {code_dir}/HEAD; git diff > {code_dir}/uncommitted.patch")
    manifest = os.path.join(out_dir, "manifest.jsonl")
    done = set()
    if os.path.exists(manifest):
        for l in open(manifest):
            r = json.loads(l); done.add((r["task_id"], r["magnitude"], r["dir"], r["noise_seed"]))
    store = TraceStore("runs", a.run_id)
    t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:7.0f}s]", *x, flush=True)
    pol = build_policy(); pol.reset()
    total = len(scenes) * len(mags) * len(rep["dirs"]) * len(rep["noise"])
    log(f"axis {a.axis}: {len(scenes)} scenes x {len(mags)} magnitudes x {len(rep['dirs'])*len(rep['noise'])} replicates = {total}; done {len(done)}")

    recorded = {}      # (scene, dir, noise) -> magnitude-0 features, for the state axis
    for scene in scenes:
        for d in rep["dirs"]:
            for ns in rep["noise"]:
                noise_key = lambda i, s=ns: 1000 * s + i
                for m in mags:                       # magnitude 0 first: it is the recorded source
                    key = (scene, round(float(m), 6), d, ns)
                    if key in done:
                        continue
                    env = make_env(scene)
                    rec = None
                    if m == 0:
                        rec = attach_recorder(pol._policy)
                        source = lambda i, env=env: pol.features_for(env.nominal_observation())
                    elif is_state:
                        recs = recorded.get((scene, d, ns))
                        if recs is None:
                            # resumed run: the magnitude-0 rollout was done in a previous process; redo it silently
                            cenv = make_env(scene); rec0 = attach_recorder(pol._policy)
                            g0 = attach_splice(pol._policy, source=lambda i: None, drive=None, noise_key=noise_key)
                            try:
                                rollout(cenv, pol, ENV_SEED, PerturbationSpec(), video_dir=None)
                            finally:
                                g0.detach(); rec0.detach()
                            recs = recorded[(scene, d, ns)] = rec0.records
                        source = lambda i, recs=recs: recs[i] if i < len(recs) else None
                    else:
                        source = lambda i, env=env: pol.features_for(env.nominal_observation())
                    h = attach_splice(pol._policy, source=source, drive="P", noise_key=noise_key)
                    try:
                        r = rollout(env, pol, ENV_SEED, spec_for(a.axis, m, d), video_dir=None)
                    finally:
                        h.detach()
                        if rec is not None:
                            rec.detach(); recorded[(scene, d, ns)] = rec.records
                    store.append(r)
                    F = len(h.records)
                    if F == 0:
                        raise RuntimeError(f"no forwards recorded: scene {scene} m {m} dir {d} noise {ns}")
                    acts = {arm: np.stack([x["action"][arm][0, :, :LIVE_DIMS].float().cpu().numpy() for x in h.records]).astype(np.float16) for arm in ARMS}
                    tf = {arm: np.array([transfer_fraction(x["action"][arm], x["action"]["P"], x["action"]["N"], LIVE_DIMS) for x in h.records], np.float32) for arm in ("T", "I", "S")}
                    d_pn = np.array([torch.linalg.vector_norm(x["action"]["P"][..., :LIVE_DIMS].float() - x["action"]["N"][..., :LIVE_DIMS].float()).item() for x in h.records], np.float32)
                    ca = closest_to_bowl(r, env)
                    tag = f"{scene}_{m:.6f}_d{d}_n{ns}".replace(".", "p")
                    np.savez_compressed(os.path.join(out_dir, f"{tag}.npz"), env_steps=np.array(pol.forward_env_steps[:F], np.int32),
                                        d_pn=d_pn, closest_approach=ca, **{f"tf_{k}": v for k, v in tf.items()}, **{f"act_{k}": v for k, v in acts.items()})
                    row = {"rollout_id": r.rollout_id, "task_id": scene, "axis": a.axis, "magnitude": round(float(m), 6),
                           "dir": d, "noise_seed": ns, "env_seed": ENV_SEED, "success": r.success, "termination": r.termination,
                           "forwards": F, "env_steps": r.env_steps, "d_pn_f0": float(d_pn[0]), "d_pn_mean": float(d_pn.mean()),
                           "tf_f0": {k: float(v[0]) for k, v in tf.items()},
                           "closest_approach_m": float(np.nanmin(ca)) if np.isfinite(ca).any() else None, "wall_s": round(r.wall_time_s, 1)}
                    with open(manifest, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    done.add(key); n_new += 1
                    if a.max_rollouts is not None and n_new >= a.max_rollouts:
                        log("max-rollouts reached"); return
                    log(f"scene {scene} m={m:<6} dir {d} noise {ns}: success={r.success} F={F} d_pn0={row['d_pn_f0']:.3f} ca={row['closest_approach_m']}")
                    del env
                # the magnitude-0 features are GPU clones used only by this magnitude loop; keeping them OOM'd the first run
                recorded.pop((scene, d, ns), None); torch.cuda.empty_cache()
    log("done")


if __name__ == "__main__":
    main()
