"""R-039 precondition 1, at the MODEL: is a patched forward a function?

Run with R-037's exact policy configuration (RESULTS.md R-039, "Determinism
gate, restated"):

  PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config \
  MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .venvs/libero-plus/bin/python experiments/r039_determinism_gate.py [out_dir]

Gates, each reported as a number and a verdict, never as an assumption:
  A  source features: `features_for(obs)` twice on the same Observation ->
     bitwise-equal backbone_features / state_features?
  B  head under a fixed seed: `predict_action_chunk` twice with
     torch.manual_seed(0) before each -> bitwise-equal chunk?
  B' control: the same without a seed -> DIFFERENT chunk (else the seed
     proves nothing, the head was already deterministic or dead).
  C  one end-to-end spliced forward on a Camera-Viewpoint variant, all five
     arms recorded from a paired render; P != N expected, transfer fractions
     printed. This is the integration smoke, not a result.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.getcwd())

import numpy as np
import torch

from vla_harness.capture.splice import ARMS, attach_splice, transfer_fraction
from vla_harness.envs.libero_env import LiberoEnv
from vla_harness.policies.lerobot_policy import LeRobotPolicy
from vla_harness.schema import PerturbationSpec

OUT = sys.argv[1] if len(sys.argv) > 1 else "runs/r039_gate"
os.makedirs(OUT, exist_ok=True)
NOMINAL_TASK = 984          # R-037 arm A control id
VIEW_TASK = 616             # ..._view_0_0_120_0_0_initstate_0 (Camera Viewpoints)
LIVE_DIMS = 7

t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


def build_policy():
    from lerobot.envs.configs import LiberoPlusEnv
    return LeRobotPolicy(
        "nvidia/gr00t17-lerobot-libero_spatial-640", n_action_steps=16,
        env_cfg=LiberoPlusEnv(task="libero_spatial"),
        policy_overrides={"base_model_path": "nvidia/GR00T-N1.7-3B",
                          "embodiment_tag": "libero_sim"},
        dtype="bfloat16",
        rename_map={"observation.images.image2": "observation.images.wrist_image"})


def bitwise(a, b):
    return bool(torch.equal(a, b)), float((a.float() - b.float()).abs().max().item())


report = {"torch": torch.__version__, "cudnn_deterministic": torch.backends.cudnn.deterministic,
          "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}

pol = build_policy()
pol.reset()
log("policy loaded")

# ---- A and B on the nominal control ---------------------------------------
env = LiberoEnv(suite="libero_spatial", task_id=NOMINAL_TASK, libero_plus=True,
                libero_plus_base_instruction=True, obs_size=360)
obs = env.reset(0, PerturbationSpec())
log("nominal env reset")

f1 = pol.features_for(obs)
f2 = pol.features_for(obs)
eqA_vl, dA_vl = bitwise(f1["backbone_features"], f2["backbone_features"])
eqA_st, dA_st = bitwise(f1["state_features"], f2["state_features"])
report["A_source_features"] = {"backbone_bitwise": eqA_vl, "backbone_maxabs": dA_vl,
                               "state_bitwise": eqA_st, "state_maxabs": dA_st,
                               "n_tokens": int(f1["backbone_features"].shape[1]),
                               "n_image_tokens": int(f1["image_mask"].sum().item())}
log("A", report["A_source_features"])

batch = pol._build_batch(obs)
with torch.inference_mode():
    torch.manual_seed(0); a1 = pol._policy.predict_action_chunk(batch).clone()
    torch.manual_seed(0); a2 = pol._policy.predict_action_chunk(batch).clone()
    a3 = pol._policy.predict_action_chunk(batch).clone()
    a4 = pol._policy.predict_action_chunk(batch).clone()
eqB, dB = bitwise(a1, a2)
eqBc, dBc = bitwise(a3, a4)
report["B_head_fixed_seed"] = {"bitwise": eqB, "maxabs": dB}
report["B_control_unseeded"] = {"bitwise": eqBc, "maxabs": dBc,
                                "note": "must be False for the seed to mean anything"}
log("B", report["B_head_fixed_seed"], "control", report["B_control_unseeded"])

# ---- C: one spliced forward on a view variant -----------------------------
env_v = LiberoEnv(suite="libero_spatial", task_id=VIEW_TASK, libero_plus=True,
                  libero_plus_base_instruction=True, obs_size=360)
obs_v = env_v.reset(0, PerturbationSpec())
pol.reset()
h = attach_splice(pol._policy, source=lambda i: pol.features_for(env_v.nominal_observation()),
                  drive=None, noise_key=lambda i: 1000 + i)
try:
    act = pol(obs_v)
    rec = h.records[0]
    acts = rec["action"]
    tf = {arm: transfer_fraction(acts[arm], acts["P"], acts["N"], LIVE_DIMS) for arm in ("T", "I", "S")}
    report["C_spliced_forward"] = {
        "arms_recorded": list(acts), "chunk_shape": list(acts["P"].shape),
        "P_vs_N_maxabs_live": float((acts["P"][..., :LIVE_DIMS].float() - acts["N"][..., :LIVE_DIMS].float()).abs().max().item()),
        "transfer_fraction": tf,
        "executed_equals_P": bool(torch.equal(acts["P"][0, 0, :LIVE_DIMS].float().cpu(),
                                              torch.tensor(act.values[:LIVE_DIMS]))) if hasattr(act, "values") else None,
    }
    log("C", report["C_spliced_forward"])
finally:
    h.detach()

report["verdict"] = {
    "A_pass": eqA_vl and eqA_st,
    "B_pass": eqB and not eqBc,
    "C_ran": "C_spliced_forward" in report,
}
json.dump(report, open(os.path.join(OUT, "gate.json"), "w"), indent=1, default=str)
log("verdict", report["verdict"])
