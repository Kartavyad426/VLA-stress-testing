"""R-039 follow-up check for expectation 1: is the state token simply
LOW-INFLUENCE, or is the S splice broken?

On one Robot-Initial-State instance, at forward 0 only, with the fixed noise:
  P      : the target's own features
  S_ctl  : state token from the control rollout (the R-039 S arm)
  S_zero : state token zeroed (what state_dropout does at training time)
  S_rand : state token replaced by N(0, sigma_P) noise
  I_ctl  : image tokens from the control (the R-039 I arm)
Reports ‖a(P) − a(X)‖ on the 7 live dims for each X, against ‖a(P) − a(N)‖.
If zeroing or randomising the state token moves the action about as little
as swapping it for the control's, the token is low-influence and the S
splice is not the problem.

  PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config \
  MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  flock /tmp/vla_gpu.lock .venvs/libero-plus/bin/python experiments/r039_state_token_check.py [task control out]
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.getcwd())

import torch

from vla_harness.capture.splice import attach_recorder, attach_splice
from vla_harness.envs.libero_env import LiberoEnv
from vla_harness.policies.lerobot_policy import LeRobotPolicy
from vla_harness.schema import PerturbationSpec

TASK = int(sys.argv[1]) if len(sys.argv) > 1 else 468        # next-to-cookie-box · initstate 257
CTL = int(sys.argv[2]) if len(sys.argv) > 2 else 1201
OUT = sys.argv[3] if len(sys.argv) > 3 else "runs/r039_state_token_check.json"
LIVE = 7


def build_policy():
    from lerobot.envs.configs import LiberoPlusEnv
    return LeRobotPolicy("nvidia/gr00t17-lerobot-libero_spatial-640", n_action_steps=16,
                         env_cfg=LiberoPlusEnv(task="libero_spatial"),
                         policy_overrides={"base_model_path": "nvidia/GR00T-N1.7-3B", "embodiment_tag": "libero_sim"},
                         dtype="bfloat16", rename_map={"observation.images.image2": "observation.images.wrist_image"})


pol = build_policy(); pol.reset()
env_c = LiberoEnv(suite="libero_spatial", task_id=CTL, libero_plus=True, libero_plus_base_instruction=True, obs_size=360)
env_t = LiberoEnv(suite="libero_spatial", task_id=TASK, libero_plus=True, libero_plus_base_instruction=True, obs_size=360)
obs_c = env_c.reset(0, PerturbationSpec()); obs_t = env_t.reset(0, PerturbationSpec())
src = pol.features_for(obs_c)
tgt = pol.features_for(obs_t)
head = pol._policy._groot_model.action_head
batch = pol._build_batch(obs_t)
model = pol._policy._groot_model


def act(vl, st):
    """One denoise pass from explicit features under the fixed noise."""
    inputs = pol._policy._filter_groot_inputs(batch, include_action=False)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        bi, ai = model.prepare_input(inputs)
        bo = model.backbone(bi)
        bo["backbone_features"] = vl                      # bypass _encode_features: already adapted
        torch.manual_seed(1000)
        out = head.get_action_with_features(backbone_features=vl, state_features=st, embodiment_id=ai.embodiment_id,
                                            backbone_output=bo, action_input=ai, options=None)
    return out["action_pred"][..., :LIVE].float()


vl_t, st_t, vl_s, st_s = tgt["backbone_features"], tgt["state_features"], src["backbone_features"], src["state_features"]
img = tgt["image_mask"]
vl_I = vl_t.clone(); vl_I[img] = vl_s[img]
aP = act(vl_t, st_t); aN = act(vl_s, st_s)
d = lambda x: torch.linalg.vector_norm(aP - x).item()
res = {"task": TASK, "control": CTL, "d_PN": d(aN),
       "d_P_Sctl": d(act(vl_t, st_s)),
       "d_P_Szero": d(act(vl_t, torch.zeros_like(st_t))),
       "d_P_Srand": d(act(vl_t, torch.randn_like(st_t) * st_t.float().std().to(st_t.dtype))),
       "d_P_Ictl": d(act(vl_I, st_t)),
       "state_token_norm_target": st_t.float().norm().item(), "state_token_norm_control": st_s.float().norm().item(),
       "state_token_delta_norm": (st_t.float() - st_s.float()).norm().item()}
res["ratio_Sctl_over_PN"] = res["d_P_Sctl"] / res["d_PN"]
res["ratio_Szero_over_PN"] = res["d_P_Szero"] / res["d_PN"]
res["ratio_Ictl_over_PN"] = res["d_P_Ictl"] / res["d_PN"]
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
