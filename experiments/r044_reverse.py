"""R-044 check 1: the REVERSE direction ("noising") for the wrist claim.

R-042 restored the nominal wrist pixels into the perturbed run and the action
moved 88% of the way to nominal (denoising). If the pose were readable from
either camera, that would still hold, but corrupting the wrist alone in the
NOMINAL run would not move the action much toward the perturbed one. So here,
at forward 0 only, no rollouts:

  a_N      = h(V(agent_n, wrist_n), E(state_n); eps)            nominal control
  a_P      = h(V(agent_p, wrist_p), E(state_p); eps)            perturbed instance
  a_revW   = h(V(agent_n, WRIST_P), E(state_n); eps)            wrist corrupted in the nominal run
  a_revA   = h(V(AGENT_P, wrist_n), E(state_n); eps)            agent view corrupted
  a_revS   = h(V(agent_n, wrist_n), E(STATE_P); eps)            state corrupted

  rev_X = 1 - ||a_revX - a_P|| / ||a_N - a_P||     share of the nominal-to-perturbed gap that
                                                    corrupting X alone produces (toward P)

Both directions high -> the wrist is necessary and sufficient. Denoising high,
noising low -> redundancy (pose readable elsewhere too), and the data spec
weakens from "wrist must cover" to "either camera may".

  PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config \
  MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  flock /tmp/vla_gpu.lock .venvs/libero-plus/bin/python experiments/r044_reverse.py \
      --selection experiments/repro/r044_selection_ris.json --out runs/r044_reverse
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

from vla_harness.envs.libero_env import LiberoEnv
from vla_harness.policies.lerobot_policy import LeRobotPolicy
from vla_harness.schema import PerturbationSpec

LIVE = 7


def build_policy(adapter=None, adapter_alpha=1.0):
    from lerobot.envs.configs import LiberoPlusEnv
    return LeRobotPolicy("nvidia/gr00t17-lerobot-libero_spatial-640", n_action_steps=16,
                         env_cfg=LiberoPlusEnv(task="libero_spatial"),
                         policy_overrides={"base_model_path": "nvidia/GR00T-N1.7-3B", "embodiment_tag": "libero_sim"},
                         dtype="bfloat16", rename_map={"observation.images.image2": "observation.images.wrist_image"},
                         adapter=adapter, adapter_alpha=adapter_alpha)


def make_env(task_id):
    return LiberoEnv(suite="libero_spatial", task_id=task_id, libero_plus=True,
                     libero_plus_base_instruction=True, obs_size=360)


def hybrid(obs, frames_from, camera):
    from copy import copy
    o = copy(obs)
    fr = dict(obs.frames or {})
    fr[camera] = frames_from.frames[camera]
    o.frames = fr
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--adapter", default=None, help="R-056: a LoRA adapter dir on top of the base checkpoint")
    ap.add_argument("--adapter-alpha", type=float, default=1.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sel = json.load(open(a.selection))
    pol = build_policy(a.adapter, a.adapter_alpha); pol.reset()
    head = pol._policy._groot_model.action_head
    model = pol._policy._groot_model
    t0 = time.time()
    log = lambda *x: print(f"[{time.time()-t0:6.0f}s]", *x, flush=True)

    def act(obs, state_from=None):
        """Full stack on `obs` (frames + its own state unless overridden) under the fixed noise."""
        feats = pol.features_for(obs)
        st = feats["state_features"] if state_from is None else pol.features_for(state_from)["state_features"]
        batch = pol._build_batch(obs)
        inputs = pol._policy._filter_groot_inputs(batch, include_action=False)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bi, ai = model.prepare_input(inputs)
            bo = model.backbone(bi)
            bo["backbone_features"] = feats["backbone_features"]
            torch.manual_seed(1000 * a.noise_seed + 0)
            out = head.get_action_with_features(backbone_features=feats["backbone_features"], state_features=st,
                                                embodiment_id=ai.embodiment_id, backbone_output=bo,
                                                action_input=ai, options=None)
        return out["action_pred"][..., :LIVE].float()

    rows = []
    for inst in sel["instances"]:
        tid, ctl = inst["task_id"], inst["control_task_id"]
        env_p, env_n = make_env(tid), make_env(ctl)
        obs_p = env_p.reset(inst["seed"], PerturbationSpec())
        obs_n = env_n.reset(inst["seed"], PerturbationSpec())
        aP, aN = act(obs_p), act(obs_n)
        d = lambda x, y: torch.linalg.vector_norm(x - y).item()
        gap = d(aN, aP)
        # denoising (R-042 direction), recomputed here so both directions share one forward
        aW = act(hybrid(obs_p, obs_n, "image2")); aA = act(hybrid(obs_p, obs_n, "image"))
        # noising: corrupt one input of the NOMINAL run with the perturbed one
        aRW = act(hybrid(obs_n, obs_p, "image2")); aRA = act(hybrid(obs_n, obs_p, "image")); aRS = act(obs_n, state_from=obs_p)
        row = {"task_id": tid, "control_task_id": ctl, "label": inst.get("label"), "level": inst.get("level"),
               "gap_PN": gap,
               "denoise_W": 1 - d(aW, aN) / gap, "denoise_A": 1 - d(aA, aN) / gap,
               "noise_W": 1 - d(aRW, aP) / gap, "noise_A": 1 - d(aRA, aP) / gap, "noise_S": 1 - d(aRS, aP) / gap}
        rows.append(row)
        log(f"task {tid} {inst.get('label','')[:34]:34s} gap {gap:.2f}  denoise W {row['denoise_W']:.2f} A {row['denoise_A']:.2f} | "
            f"noise W {row['noise_W']:.2f} A {row['noise_A']:.2f} S {row['noise_S']:.2f}")
        with open(os.path.join(a.out, "rows.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        del env_p, env_n
    med = {k: float(np.median([r[k] for r in rows])) for k in ("denoise_W", "denoise_A", "noise_W", "noise_A", "noise_S")}
    summary = {"n": len(rows), "medians": med,
               "noise_W_gt_A": int(sum(r["noise_W"] > r["noise_A"] for r in rows)),
               "denoise_W_gt_A": int(sum(r["denoise_W"] > r["denoise_A"] for r in rows)),
               "noise_seed": a.noise_seed, "policy_id": pol.policy_id, "adapter": a.adapter,
               "adapter_alpha": a.adapter_alpha}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    log("summary", summary)


if __name__ == "__main__":
    main()
