"""LeRobotPolicy with a PEFT adapter (R-056 WP4), on a tiny CPU module.

Checks: the base identity is untouched by the new fields; the adapter's
CONTENT hash, alpha and merge enter identity(); alpha scales the LoRA delta
linearly (0 = base, 1 = fine-tune: WiSE-FT); merged == unmerged in fp32; the
adapter is injected in place so the policy object keeps its structure.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

peft = pytest.importorskip("peft")

from vla_harness.policies.lerobot_policy import LeRobotPolicy


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head = nn.Module()
        self.action_head.vl_self_attention = nn.Sequential(nn.Linear(6, 6))
        self.other = nn.Linear(6, 6)

    def forward(self, x):
        return self.other(self.action_head.vl_self_attention(x))


def make_adapter(tmp_path, seed=0):
    torch.manual_seed(seed)
    base = Tiny()
    sd = {k: v.clone() for k, v in base.state_dict().items()}
    cfg = peft.LoraConfig(r=2, lora_alpha=2, target_modules=["action_head.vl_self_attention.0"])
    pm = peft.get_peft_model(base, cfg)
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            p.data.normal_()
    pm.save_pretrained(str(tmp_path))
    return sd


def fresh(sd):
    m = Tiny(); m.load_state_dict(sd); return m


def test_identity_unchanged_without_adapter():
    p = LeRobotPolicy("some/ckpt", dtype="bfloat16")
    assert "adapter" not in p.identity() and p.identity()["name"] == "lerobot:some/ckpt"


def test_alpha_scales_delta_and_identity(tmp_path):
    sd = make_adapter(tmp_path)
    x = torch.randn(3, 6)
    base = fresh(sd)(x)
    outs = {}
    for alpha in (0.0, 0.5, 1.0):
        pol = LeRobotPolicy("some/ckpt", adapter=str(tmp_path), adapter_alpha=alpha)
        m = pol._attach_adapter(fresh(sd))
        assert isinstance(m, Tiny)                     # structure kept: injected in place
        with torch.no_grad():
            outs[alpha] = m(x)
    assert torch.allclose(outs[0.0], base, atol=1e-6)
    assert not torch.allclose(outs[1.0], base, atol=1e-3)
    assert torch.allclose(outs[0.5], (outs[0.0] + outs[1.0]) / 2, atol=1e-5)
    merged = LeRobotPolicy("some/ckpt", adapter=str(tmp_path), adapter_merge=True)._attach_adapter(fresh(sd))
    with torch.no_grad():
        assert torch.allclose(merged(x), outs[1.0], atol=1e-5)
    i1 = LeRobotPolicy("some/ckpt", adapter=str(tmp_path)).identity()
    i2 = LeRobotPolicy("some/ckpt", adapter=str(tmp_path), adapter_alpha=0.5).identity()
    assert i1["adapter"]["sha"] == i2["adapter"]["sha"] and i1 != i2
    make_adapter(tmp_path, seed=1)                     # new weights, same path
    assert LeRobotPolicy("some/ckpt", adapter=str(tmp_path)).identity()["adapter"]["sha"] != i1["adapter"]["sha"]


def test_missing_adapter_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        LeRobotPolicy("some/ckpt", adapter=str(tmp_path))
