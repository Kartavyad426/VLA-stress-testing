"""R-056 LR schedule fix: the construction, the guard, and the streaming bf16 load (CPU only).

lerobot_train steps its scheduler per micro-batch; these drive lerobot's own AcceleratorConfig and
update_policy (vla_harness/training/schedule.py) and need the libero-plus venv."""
from __future__ import annotations

import math

import pytest
import torch

S = pytest.importorskip("vla_harness.training.schedule")
pytest.importorskip("lerobot.scripts.lerobot_train")

LR, WR = 1e-4, 0.05


def _micro_count(opt, schedule, n, wr, accum):
    """Diffusers over opt_steps*accum micro-steps: lags the registered LR by (accum-1)/accum of a step."""
    from diffusers.optimization import get_scheduler
    return get_scheduler(schedule, optimizer=opt, num_warmup_steps=S.warmup_steps(n, wr, schedule) * accum,
                         num_training_steps=n * accum)


@pytest.mark.parametrize("n,accum", [(40, 8), (200, 8), (30, 1)])
def test_fixed_construction_applies_registered_lr_through_lerobot(n, accum):
    tr = S.simulate_lerobot(S.build_scheduler, LR, "cosine", n, WR, accum)
    r = S.check(tr, LR, "cosine", n, WR)
    assert r["pass"], r
    assert len(tr) == n and r["max_abs_err"] < 1e-15


def test_constant_schedule_passes():
    r = S.check(S.simulate_lerobot(S.build_scheduler, LR, "constant", 30, WR, 8), LR, "constant", 30, WR)
    assert r["pass"] and r["max_abs_err"] == 0


@pytest.mark.parametrize("builder", [S.legacy_scheduler, _micro_count])
def test_guard_rejects_cycling_and_lagging_constructions(builder):
    r = S.check(S.simulate_lerobot(builder, LR, "cosine", 40, WR, 8), LR, "cosine", 40, WR)
    assert not r["pass"] and r["max_abs_err"] > S.TOL


def test_registered_formula_is_diffusers_cosine_per_optimizer_step():
    from diffusers.optimization import get_scheduler
    n = 2000
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=LR)
    sch = get_scheduler("cosine", optimizer=opt, num_warmup_steps=math.ceil(n * WR), num_training_steps=n)
    for k in range(n):
        assert abs(opt.param_groups[0]["lr"] - S.registered_lr(k, LR, "cosine", n, WR)) < 1e-15
        opt.step(); sch.step()


def test_cached_loop_order_passes():
    assert S.check(S.simulate_cached(S.build_scheduler, LR, "cosine", 100, WR), LR, "cosine", 100, WR)["pass"]


def test_check_catches_short_trace_and_partial_live_trace():
    tr = [S.registered_lr(k, LR, "cosine", 100, WR) for k in range(100)]
    assert not S.check(tr[:99], LR, "cosine", 100, WR)["pass"]                     # final: must reach the end
    assert S.check(tr[20:60], LR, "cosine", 100, WR, start=20, final=False)["pass"]  # resumed, mid-run


# --- streaming bf16 load (vla_harness/policies/lerobot_policy.py) -----------------------------------
class _Toy(torch.nn.Module):
    """Mimics PreTrainedPolicy.from_pretrained: construct, _load_as_safetensor, .to(cpu), eval()."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.emb = torch.nn.Embedding(50, 12)
        self.lin = torch.nn.Linear(12, 7)
        self.head = torch.nn.Linear(12, 50, bias=False)
        self.head.weight = self.emb.weight                      # tied, as the Qwen lm_head is
        self.extra = torch.nn.Linear(3, 3)                       # absent from the file: keeps its init
        self.register_buffer("steps", torch.tensor([3, 4]))      # integer buffer, never cast
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g))

    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):
        from safetensors.torch import load_model
        load_model(model, model_file, strict=strict, device=map_location)
        return model

    @classmethod
    def from_pretrained(cls, path, config=None):
        return cls._load_as_safetensor(cls(), path, "cpu", False).to("cpu").eval()


def test_streaming_cast_load_is_bit_identical(tmp_path):
    from safetensors.torch import save_model
    from vla_harness.policies.lerobot_policy import _from_pretrained_cast
    src = _Toy()
    with torch.no_grad():
        src.lin.weight.mul_(1 + 1e-3)                           # values that do not round-trip through bf16
        for p in src.parameters():
            p.add_(torch.rand(p.shape) * 1e-4)
    del src.extra
    f = str(tmp_path / "model.safetensors")
    save_model(src, f)
    orig = _Toy.__dict__["_load_as_safetensor"]
    old = _Toy.from_pretrained(f).to(torch.bfloat16)
    new = _from_pretrained_cast(_Toy, f, None, torch.bfloat16)
    assert _Toy.__dict__["_load_as_safetensor"] is orig                    # the patch is undone
    so, sn = old.state_dict(), new.state_dict()
    assert so.keys() == sn.keys()
    for k in so:
        assert so[k].dtype == sn[k].dtype, k
        assert torch.equal(so[k], sn[k]), k
    assert new.head.weight is new.emb.weight
    assert sn["steps"].dtype == torch.int64
