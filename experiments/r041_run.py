"""R-041 runner: single-axis boundary sweeps from the R-029 control, splice on.

  PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config \
  MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  flock /tmp/vla_gpu.lock .venvs/libero-plus/bin/python experiments/r041_run.py \
      --run-id r041 --axis camera_yaw_deg [--tasks 984,985,...] [--limit N]

Per RESULTS.md R-041 ("Bisection allocation and the state axis"):
  phase 0  magnitude-0 rollout per task (the splice recorder captures its
           per-forward features; these are the recorded source for the state
           axis and give tau)
  tau      95th percentile of forward-to-forward chunk distance on the live
           dims across ALL magnitude-0 rollouts, computed before any
           perturbed rollout is analysed and written to tau.json
  phase 1  one rollout at the range maximum; no action boundary in range ->
           stop for that (task, axis)
  phase 2  four bisection steps on the ACTION criterion (d_pn at forward 0
           > tau) in [0, max]
  phase 3  two bisection steps on the OUTCOME in [action bracket, max]
Every rollout runs with all five arms recorded; the driven arm is P.

Sources: paired render for the three render axes (LiberoEnv.nominal_frames
now undoes harness knobs); the magnitude-0 recorded features for the state
axis.

Writes runs/<run-id>/<axis>/<task>/<magnitude>.npz + manifest.jsonl rows,
and runs/<run-id>/<axis>/boundaries.jsonl with one row per task. Resumable
at rollout granularity.
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
from vla_harness.schema import PerturbationSpec, TraceStore

LIVE_DIMS = 7
CONTROL_TASKS = [984, 1030, 1062, 1090, 1132, 1169, 1201, 1247, 1282, 1327]   # R-029 controls, one per scene
AXES = {  # knob -> (max, is_state_axis)
    "camera_yaw_deg": (40.0, False),
    "camera_dist_m": (0.4, False),
    "light_intensity": (3.0, False),
    "joint_radius_rad": (0.5, True),
}
N_ACTION_BISECT, N_OUTCOME_BISECT = 4, 2


def build_policy():
    from lerobot.envs.configs import LiberoPlusEnv
    return LeRobotPolicy(
        "nvidia/gr00t17-lerobot-libero_spatial-640", n_action_steps=16,
        env_cfg=LiberoPlusEnv(task="libero_spatial"),
        policy_overrides={"base_model_path": "nvidia/GR00T-N1.7-3B", "embodiment_tag": "libero_sim"},
        dtype="bfloat16", rename_map={"observation.images.image2": "observation.images.wrist_image"})


def make_env(task_id):
    return LiberoEnv(suite="libero_spatial", task_id=task_id, libero_plus=True,
                     libero_plus_base_instruction=True, obs_size=360)


def spec_for(axis, m, seed):
    if m == 0:
        return PerturbationSpec()
    if axis == "joint_radius_rad":
        return PerturbationSpec.of(joint_radius_rad=float(m), joint_dir_seed=int(seed))
    return PerturbationSpec.of(**{axis: float(m)})


def closest_to_bowl(r, env):
    try:
        targets = [t for t in env._env.unwrapped._env.obj_of_interest if "bowl" in t]
    except Exception:
        targets = []
    out = []
    for st in r.steps:
        d = st.obs_state.get("_gt_eef_to_object") or {}
        pool = {k: v for k, v in d.items() if any(k.startswith(t) or t.startswith(k) for t in targets)} or d
        out.append(min(pool.values()) if pool else float("nan"))
    return np.array(out, dtype=np.float32)


class Runner:
    def __init__(self, run_id, axis, pol, store, log):
        self.axis, self.pol, self.store, self.log = axis, pol, store, log
        self.dir = os.path.join("runs", run_id, axis)
        os.makedirs(self.dir, exist_ok=True)
        self.manifest = os.path.join(self.dir, "manifest.jsonl")
        self.done = {}
        if os.path.exists(self.manifest):
            for l in open(self.manifest):
                row = json.loads(l)
                self.done[(row["task_id"], row["magnitude"])] = row
        self.recorded = {}          # task -> magnitude-0 per-forward features

    def run(self, task, m, seed=0):
        key = (task, round(float(m), 6))
        if key in self.done:
            return self.done[key]
        env = make_env(task)
        noise_key = lambda i, s=seed: 1000 * s + i
        is_state = AXES[self.axis][1]
        rec = None
        if m == 0:
            rec = attach_recorder(self.pol._policy)
            source = lambda i: None
        elif is_state:
            recs = self.recorded[task]
            source = lambda i, recs=recs: recs[i] if i < len(recs) else None
        else:
            source = lambda i, env=env: self.pol.features_for(env.nominal_observation())
        h = attach_splice(self.pol._policy, source=source, drive="P", noise_key=noise_key)
        try:
            r = rollout(env, self.pol, seed, spec_for(self.axis, m, seed), video_dir=None)
        finally:
            h.detach()
            if rec is not None:
                rec.detach()
                self.recorded[task] = rec.records
        self.store.append(r)
        F = len(h.records)
        acts = {a: np.stack([x["action"][a][0, :, :LIVE_DIMS].float().cpu().numpy() for x in h.records]).astype(np.float16)
                for a in ARMS} if F else {}
        tf = {a: np.array([transfer_fraction(x["action"][a], x["action"]["P"], x["action"]["N"], LIVE_DIMS)
                           for x in h.records], np.float32) for a in ("T", "I", "S")}
        d_pn = np.array([torch.linalg.vector_norm(x["action"]["P"][..., :LIVE_DIMS].float()
                                                   - x["action"]["N"][..., :LIVE_DIMS].float()).item()
                         for x in h.records], np.float32)
        # forward-to-forward chunk distance of the EXECUTED arm: tau's ingredient
        p_chunks = acts["P"].astype(np.float32) if F else np.zeros((0, 40, LIVE_DIMS), np.float32)
        step_d = np.linalg.norm((p_chunks[1:] - p_chunks[:-1]).reshape(max(F - 1, 0), -1), axis=1) if F > 1 else np.zeros(0, np.float32)
        ca = closest_to_bowl(r, env)
        tag = f"{task}_{m:.6f}".replace(".", "p")
        np.savez_compressed(os.path.join(self.dir, f"{tag}.npz"), env_steps=np.array(self.pol.forward_env_steps[:F], np.int32),
                            d_pn=d_pn, step_d=step_d, closest_approach=ca,
                            **{f"tf_{k}": v for k, v in tf.items()}, **{f"act_{k}": v for k, v in acts.items()})
        row = {"rollout_id": r.rollout_id, "task_id": task, "axis": self.axis, "magnitude": round(float(m), 6), "seed": seed,
               "success": r.success, "termination": r.termination, "forwards": F, "env_steps": r.env_steps,
               "d_pn_f0": float(d_pn[0]) if F else None, "d_pn_mean": float(d_pn.mean()) if F else None,
               "tf_f0": {k: float(v[0]) for k, v in tf.items()} if F else None,
               "tf_mean": {k: float(np.nanmean(v)) for k, v in tf.items()} if F else None,
               "closest_approach_m": float(np.nanmin(ca)) if np.isfinite(ca).any() else None,
               "wall_s": round(r.wall_time_s, 1)}
        with open(self.manifest, "a") as f:
            f.write(json.dumps(row) + "\n")
        self.done[key] = row
        self.log(f"    {self.axis}={m:.4f} task {task}: success={r.success} F={F} d_pn0={row['d_pn_f0']:.3f} tf0={row['tf_f0']}")
        del env
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--axis", required=True, choices=list(AXES))
    ap.add_argument("--tasks", default=",".join(map(str, CONTROL_TASKS)))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    tasks = [int(t) for t in a.tasks.split(",")][: a.limit] if a.limit else [int(t) for t in a.tasks.split(",")]
    mx, _ = AXES[a.axis]

    root = os.path.join("runs", a.run_id)
    os.makedirs(root, exist_ok=True)
    code_dir = os.path.join(root, "code_state")
    if not os.path.exists(code_dir):
        os.makedirs(code_dir)
        os.system(f"git rev-parse HEAD > {code_dir}/HEAD; git diff > {code_dir}/uncommitted.patch")
    store = TraceStore("runs", a.run_id)
    t0 = time.time()
    log = lambda *x: print(f"[{time.time()-t0:7.0f}s]", *x, flush=True)
    pol = build_policy(); pol.reset()
    log(f"policy loaded; axis {a.axis} max {mx}; tasks {tasks}")
    R = Runner(a.run_id, a.axis, pol, store, log)

    # phase 0: magnitude-0 for every task, then tau (before any perturbed rollout is analysed)
    zeros = [R.run(t, 0.0) for t in tasks]
    if not all(z["success"] for z in zeros):
        log("WARNING magnitude-0 did not reproduce R-029's 100/100: " + str([(z["task_id"], z["success"]) for z in zeros if not z["success"]]))
    tau_path = os.path.join(root, "tau.json")
    if os.path.exists(tau_path):
        tau = json.load(open(tau_path))["tau"]
    else:
        steps = np.concatenate([np.load(os.path.join(R.dir, f"{t}_0p000000.npz"))["step_d"] for t in tasks])
        tau = float(np.percentile(steps, 95))
        json.dump({"tau": tau, "n_step_pairs": int(len(steps)), "tasks": tasks, "axis_used": a.axis,
                   "definition": "95th pct of forward-to-forward P-chunk L2 distance on 7 live dims, magnitude-0 rollouts"},
                  open(tau_path, "w"), indent=1)
    log(f"tau = {tau:.4f}")

    bounds_path = os.path.join(R.dir, "boundaries.jsonl")
    done_tasks = {json.loads(l)["task_id"] for l in open(bounds_path)} if os.path.exists(bounds_path) else set()
    for t in tasks:
        if t in done_tasks:
            continue
        log(f"task {t}: phase 1 at max {mx}")
        top = R.run(t, mx)
        samples = [(0.0, zeros[tasks.index(t)]), (mx, top)]
        row = {"task_id": t, "axis": a.axis, "tau": tau, "max": mx}
        if top["d_pn_f0"] <= tau:
            row.update({"action_boundary": None, "outcome_boundary": None, "note": "no action boundary in range"})
        else:
            lo, hi = 0.0, mx
            for _ in range(N_ACTION_BISECT):
                mid = (lo + hi) / 2
                r = R.run(t, mid); samples.append((mid, r))
                if r["d_pn_f0"] > tau: hi = mid
                else: lo = mid
            row["action_boundary"] = [lo, hi]
            # outcome: search in [hi_action, max]; success at lo side assumed from magnitude 0
            olo, ohi = hi, mx
            o_lo_ok = True
            if top["success"]:
                row["outcome_boundary"] = None; row["note"] = "still succeeds at max"
            else:
                for _ in range(N_OUTCOME_BISECT):
                    mid = (olo + ohi) / 2
                    r = R.run(t, mid); samples.append((mid, r))
                    if r["success"]: olo = mid
                    else: ohi = mid
                row["outcome_boundary"] = [olo, ohi]
        row["samples"] = sorted([(m, r["success"], r["d_pn_f0"], r["tf_f0"]) for m, r in samples], key=lambda x: x[0])
        with open(bounds_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        log(f"task {t}: action {row.get('action_boundary')} outcome {row.get('outcome_boundary')}")
    log("done")


if __name__ == "__main__":
    main()
