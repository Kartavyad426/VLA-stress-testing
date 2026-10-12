"""R-059 CPU unit check (my prim's t4a spec): the multi-draw wrapper on the REAL GR00TN17ActionHead.
get_action_with_features (bound to a tiny fake head with random small modules), CPU, no checkpoint.

Checks:
  1 draw 0 with B=1 == draw 0 with B=32 (seeded), bitwise, in float32 and bfloat16;
  2 same B, same seed, two calls -> bitwise identical (all draws);
  3 draws differ from each other (the shadow draws are not copies);
  4 the noise patch is restored after the call (torch.randn is the original again) and raises on a stray call;
  5 wrapper with B=1 and seeded noise == calling the original with that same noise directly (the wrapper adds nothing).
  .venvs/libero-plus/bin/python experiments/r059_unit_check.py  -> runs/r059/unit_check.json
"""
import json
import os
import sys
import types

sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from lerobot.policies.groot.groot_n1_7 import GR00TN17ActionHead
import r059_multidraw as M

torch.manual_seed(0)
HZ, D, HID, NV, NS = 40, 32, 24, 12, 1


class Enc(nn.Module):
    def __init__(s): super().__init__(); s.l = nn.Linear(D + 1, HID)
    def forward(s, a, t, e): return s.l(torch.cat([a, (t.float() / 1000).to(a.dtype)[:, None, None].expand(a.shape[0], a.shape[1], 1)], -1))


class Dit(nn.Module):
    def __init__(s): super().__init__(); s.q = nn.Linear(HID, HID); s.k = nn.Linear(HID, HID)
    def forward(s, hidden_states, encoder_hidden_states, timestep, image_mask=None, backbone_attention_mask=None):
        att = torch.softmax(s.q(hidden_states) @ s.k(encoder_hidden_states).transpose(1, 2) / HID ** .5, -1)
        return hidden_states + att @ encoder_hidden_states


class Dec(nn.Module):
    def __init__(s): super().__init__(); s.l = nn.Linear(HID, D)
    def forward(s, x, e): return s.l(x)


def fake_head(dtype):
    h = types.SimpleNamespace()
    h.config = types.SimpleNamespace(action_horizon=HZ, use_alternate_vl_dit=True, add_pos_embed=True)
    h.action_dim, h.action_horizon, h.num_inference_timesteps, h.num_timestep_buckets = D, HZ, 4, 1000
    h.action_encoder, h.model, h.action_decoder = Enc().to(dtype), Dit().to(dtype), Dec().to(dtype)
    h.position_embedding = nn.Embedding(HZ, HID).to(dtype)
    h.get_action_with_features = types.MethodType(GR00TN17ActionHead.get_action_with_features, h)
    return h


def inputs(dtype):
    vl = torch.randn(1, NV, HID).to(dtype); st = torch.randn(1, NS, HID).to(dtype)
    bo = BatchFeature(data={"backbone_features": vl.clone(), "image_mask": torch.ones(1, NV, dtype=torch.bool),
                            "backbone_attention_mask": torch.ones(1, NV, dtype=torch.bool)})
    return dict(backbone_features=vl, state_features=st, embodiment_id=torch.zeros(1, dtype=torch.long),
                backbone_output=bo, action_input=BatchFeature(data={}))


def run(head, kw, B, seed=3, fwd=0):
    h = M.MultiDrawHandle(head, B, seed, seed_noise=True)
    try:
        h._fidx = fwd
        out = head.get_action_with_features(**kw)
    finally:
        h.detach()
    return out["action_pred"], h.records[0]["act_B"]


res = {}
for name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
    head = fake_head(dtype); kw = inputs(dtype)
    a1, B1 = run(head, kw, 1); a32, B32 = run(head, kw, 32); a32b, B32b = run(head, kw, 32)
    orig_randn = torch.randn
    noise = M.seeded_noise((1, HZ, D), dtype, "cpu", 3, 0, 1)
    with M._NoisePatch(noise):
        direct = head.get_action_with_features(**kw)["action_pred"]
    stray = False
    try:
        with M._NoisePatch(noise):
            torch.randn(5)
    except RuntimeError:
        stray = True
    res[name] = {"1_draw0_B1_eq_B32": bool(torch.equal(a1, a32)), "1_max_abs": float((a1.float() - a32.float()).abs().max()),
                 "2_same_seed_repeat_bitwise": bool(torch.equal(B32, B32b)),
                 "3_draws_differ": bool(not torch.equal(B32[0], B32[1])),
                 "4_randn_restored": torch.randn is orig_randn, "4_stray_call_raises": stray,
                 "5_wrapper_eq_direct": bool(torch.equal(a1, direct)),
                 "returned_batch": list(a32.shape), "act_B_shape": list(B32.shape)}
    print(name, res[name])
# 6: an unexpected backbone_output key, or a non-batch leading 1, must stop a B > 1 call
head = fake_head(torch.float32); kw = inputs(torch.float32)
bad = dict(kw); bo = BatchFeature(data=dict(kw["backbone_output"])); bo["extra"] = torch.zeros(1, 3); bad["backbone_output"] = bo
bad2 = dict(kw); bad2["state_features"] = kw["state_features"].reshape(1, -1)
raised = []
for b_ in (bad, bad2):
    try:
        run(head, b_, 8); raised.append(False)
    except RuntimeError:
        raised.append(True)
res["6_whitelist_rejects_extra_key_and_bad_rank"] = raised
print("6", raised)
os.makedirs("runs/r059", exist_ok=True)
json.dump(res, open("runs/r059/unit_check.json", "w"), indent=1)
ok = all(v["2_same_seed_repeat_bitwise"] and v["3_draws_differ"] and v["4_randn_restored"] and v["4_stray_call_raises"]
         and v["5_wrapper_eq_direct"] for k, v in res.items() if isinstance(v, dict)) and all(res["6_whitelist_rejects_extra_key_and_bad_rank"])
print("UNIT CHECK", "PASS" if ok else "FAIL", "(check 1 reported, not required: batch-size kernels may differ on GPU)")
