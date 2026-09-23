"""R-039 runner: one rollout per instance with the pathway splice attached.

  PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config \
  MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  flock /tmp/vla_gpu.lock .venvs/libero-plus/bin/python experiments/r039_run.py \
      --selection experiments/repro/r039_selection.json --run-id r039 [--drive P] [--limit N]

Per instance (RESULTS.md R-039, revision of 2026-09-23):
  * vision categories: the SOURCE at every forward is a paired render of the
    current state under the nominal condition (LiberoEnv.nominal_observation
    -> LeRobotPolicy.features_for).
  * Robot Initial States: the source is the recorded per-forward features of
    the scene's CONTROL variant (language_1_view_0_0_100_0_0_initstate_0 with
    --base-instruction, i.e. the R-029 control) run first under the same
    seed and the same fixed noise; past its last forward the target passes
    through unspliced.
  * all five arms (P N T I S) are computed at every forward; `--drive`
    (default P) is executed, so the episode's outcome is the real one.
  * noise_key(forward) = 1000*seed + forward, identical across arms.

Writes runs/<run-id>/splice/<rollout_id>.npz with the five chunks per forward
on the 7 live dims (normalised action space), the env step of each forward,
per-step closest approach from the trace, and the transfer fractions; and
appends a row to runs/<run-id>/splice/manifest.jsonl. The trace itself goes
through TraceStore like every other run. Resumable: an instance whose npz
exists is skipped.
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
RIS = "Robot Initial States"


def build_policy():
    from lerobot.envs.configs import LiberoPlusEnv
    return LeRobotPolicy(
        "nvidia/gr00t17-lerobot-libero_spatial-640", n_action_steps=16,
        env_cfg=LiberoPlusEnv(task="libero_spatial"),
        policy_overrides={"base_model_path": "nvidia/GR00T-N1.7-3B",
                          "embodiment_tag": "libero_sim"},
        dtype="bfloat16",
        rename_map={"observation.images.image2": "observation.images.wrist_image"})


def make_env(suite, task_id):
    return LiberoEnv(suite=suite, task_id=task_id, libero_plus=True,
                     libero_plus_base_instruction=True, obs_size=360)


def target_names(env) -> list[str]:
    """The BDDL objects of interest for this task (the target bowl and the
    plate), from the LIBERO-plus wrapper; empty if unavailable."""
    try:
        return list(env._env.unwrapped._env.obj_of_interest)
    except Exception:
        return []


def closest_approach(r, targets: list[str]):
    """Per step: eef distance to the TARGET object (R-038's discriminator),
    the nearest object of interest when several, all task objects as the
    fallback. Which was used is returned alongside."""
    out, used = [], None
    for st in r.steps:
        d = st.obs_state.get("_gt_eef_to_object") or {}
        bowls = [t for t in targets if "bowl" in t] or targets      # the target is the bowl, not the plate
        keys = [k for k in d if any(k.startswith(t) or t.startswith(k) for t in bowls)] if bowls else []
        pool = {k: d[k] for k in keys} if keys else d
        used = used or ("targets" if keys else "all_objects")
        out.append(min(pool.values()) if pool else float("nan"))
    return np.array(out, dtype=np.float32), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--drive", default="P", choices=list(ARMS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()

    sel = json.load(open(a.selection))
    suite = sel["suite"]
    instances = sel["instances"][: a.limit] if a.limit else sel["instances"]
    out_dir = os.path.join("runs", a.run_id, "splice")
    os.makedirs(out_dir, exist_ok=True)
    manifest = os.path.join(out_dir, "manifest.jsonl")
    done = set()
    if os.path.exists(manifest):
        done = {json.loads(l)["task_id"] for l in open(manifest)}
    video_dir = None if a.no_video else os.path.join("runs", a.run_id, "video")
    store = TraceStore("runs", a.run_id)
    code_dir = os.path.join("runs", a.run_id, "code_state")
    if not os.path.exists(code_dir):
        os.makedirs(code_dir)
        os.system(f"git rev-parse HEAD > {code_dir}/HEAD; git diff > {code_dir}/uncommitted.patch")
        json.dump(sel, open(os.path.join(code_dir, "selection.json"), "w"), indent=1)

    pol = build_policy()
    pol.reset()                      # loads: the wrapper is lazy and the splice attaches to the loaded head
    t0 = time.time()
    log = lambda *x: print(f"[{time.time()-t0:7.0f}s]", *x, flush=True)
    log(f"policy loaded; {len(instances)} instances, {len(done)} already done")

    recorded_controls: dict[tuple[int, int], list] = {}

    for k, inst in enumerate(instances):
        tid, seed, cat = inst["task_id"], inst["seed"], inst["category"]
        if tid in done:
            continue
        log(f"[{k+1}/{len(instances)}] task {tid} L{inst['level']} {inst.get('label', cat)} prior={inst['prior_outcome']}")
        noise_key = lambda i, s=seed: 1000 * s + i

        # --- source ----------------------------------------------------------
        env = make_env(suite, tid)
        if cat == RIS:
            ctl = inst["control_task_id"]
            key = (ctl, seed)
            if key not in recorded_controls:
                cenv = make_env(suite, ctl)
                rec = attach_recorder(pol._policy)
                # the control must draw the SAME noise per forward as the target
                gen = attach_splice(pol._policy, source=lambda i: None, drive=None, noise_key=noise_key)
                try:
                    r_ctl = rollout(cenv, pol, seed, PerturbationSpec(), video_dir=None)
                finally:
                    gen.detach(); rec.detach()
                recorded_controls[key] = rec.records
                log(f"    control {ctl}: success={r_ctl.success} forwards={len(rec.records)}")
                store.append(r_ctl)
            recs = recorded_controls[key]
            source = lambda i, recs=recs: recs[i] if i < len(recs) else None
        else:
            source = lambda i, env=env: pol.features_for(env.nominal_observation())

        # --- the spliced rollout ---------------------------------------------
        h = attach_splice(pol._policy, source=source, drive=a.drive, noise_key=noise_key)
        try:
            r = rollout(env, pol, seed, PerturbationSpec(), video_dir=video_dir)
        finally:
            h.detach()
        store.append(r)

        # --- per-forward arrays ----------------------------------------------
        F = len(h.records)
        acts = {arm: np.stack([rec["action"][arm][0, :, :LIVE_DIMS].float().cpu().numpy()
                               for rec in h.records]).astype(np.float16) for arm in ARMS}
        tf = {arm: np.array([transfer_fraction(rec["action"][arm], rec["action"]["P"],
                                               rec["action"]["N"], LIVE_DIMS)
                             for rec in h.records], dtype=np.float32) for arm in ("T", "I", "S")}
        d_pn = np.array([torch.linalg.vector_norm(
            rec["action"]["P"][..., :LIVE_DIMS].float() - rec["action"]["N"][..., :LIVE_DIMS].float()).item()
            for rec in h.records], dtype=np.float32)
        env_steps = np.array(pol.forward_env_steps[:F], dtype=np.int32)
        ca, ca_basis = closest_approach(r, target_names(env))
        np.savez_compressed(os.path.join(out_dir, f"{r.rollout_id}.npz"),
                            env_steps=env_steps, d_pn=d_pn, closest_approach=ca,
                            **{f"tf_{k}": v for k, v in tf.items()},
                            **{f"act_{k}": v for k, v in acts.items()})
        anchor_step = int(np.nanargmin(ca)) if np.isfinite(ca).any() else -1
        anchor_fwd = int(np.searchsorted(env_steps, anchor_step, side="right") - 1) if anchor_step >= 0 and F else -1
        row = {"rollout_id": r.rollout_id, "task_id": tid, "seed": seed, "category": cat,
               "label": inst.get("label"), "scene": inst.get("scene"),
               "variant": inst["variant"], "level": inst["level"], "prior_outcome": inst["prior_outcome"],
               "drive": a.drive, "success": r.success, "termination": r.termination,
               "env_steps": r.env_steps, "forwards": F, "wall_s": round(r.wall_time_s, 1),
               "closest_approach_m": float(np.nanmin(ca)) if np.isfinite(ca).any() else None,
               "closest_approach_basis": ca_basis, "targets": target_names(env),
               "anchor_step": anchor_step, "anchor_forward": anchor_fwd,
               "tf_at_f0": {k: float(v[0]) for k, v in tf.items()} if F else None,
               "tf_at_anchor": {k: float(v[anchor_fwd]) for k, v in tf.items()} if F and anchor_fwd >= 0 else None,
               "tf_mean": {k: float(np.nanmean(v)) for k, v in tf.items()} if F else None,
               "d_pn_f0": float(d_pn[0]) if F else None}
        with open(manifest, "a") as f:
            f.write(json.dumps(row) + "\n")
        log(f"    success={r.success} F={F} d_pn0={row['d_pn_f0']:.3f} "
            f"tf0={row['tf_at_f0']} anchor_fwd={anchor_fwd} tf_anchor={row['tf_at_anchor']}")
        del env
    log("done")


if __name__ == "__main__":
    main()
