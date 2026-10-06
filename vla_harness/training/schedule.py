"""The LR schedule, counted in OPTIMIZER steps, and the guard that proves it (R-056 fix, 2026-09-29).

lerobot_train steps its scheduler once per MICRO-batch: its Accelerator is built with
step_scheduler_with_optimizer=False (configs/accelerator.py:239) and update_policy calls
lr_scheduler.step() after every micro-batch. A diffusers cosine built over the optimizer-step
count therefore ran `accumulation` times too fast: R-056's first run cycled its LR.

`build_scheduler` returns a LambdaLR for that stepping: after s scheduler steps the LR is
lr * L(s // accum), L being diffusers' own lambda over the optimizer-step count. The k-th
optimizer update (0-based) fires at micro-batch k*accum + accum - 1, after exactly that many
scheduler steps, so it applies lr * L(k): the registered schedule, exactly, at every step.
(Building diffusers' scheduler over opt_steps*accum micro-steps instead lags it by
(accum-1)/accum of a step: 8.8e-7 at R-056's scale, but 4e-5 on a 40-step run.)

`simulate_lerobot` runs a dummy one-parameter policy through lerobot's OWN AcceleratorConfig and
update_policy on the CPU and records the LR each real optimizer update applied (an optimizer
step pre-hook fires only when accelerate lets the step through). `check` compares that trace
with `registered_lr`, an independent closed form, and fails beyond TOL.
"""
from __future__ import annotations

import math

TOL = 1e-6


def warmup_steps(opt_steps: int, warmup_ratio: float, schedule: str) -> int:
    return 0 if schedule == "constant" else math.ceil(opt_steps * warmup_ratio)


def registered_lr(k: int, lr: float, schedule: str, opt_steps: int, warmup_ratio: float) -> float:
    """The registered LR at optimizer step k (0-based): linear warmup from 0, then half-cosine to 0.
    Written out, not taken from diffusers, so the guard does not compare diffusers with itself."""
    if schedule == "constant":
        return lr
    w = warmup_steps(opt_steps, warmup_ratio, schedule)
    if k < w:
        return lr * k / max(1, w)
    p = (k - w) / max(1, opt_steps - w)
    return lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))


def _diffusers_lambda(schedule: str, opt_steps: int, warmup_ratio: float):
    import torch
    from diffusers.optimization import get_scheduler
    dummy = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    return get_scheduler(schedule, optimizer=dummy, num_warmup_steps=warmup_steps(opt_steps, warmup_ratio, schedule),
                         num_training_steps=opt_steps).lr_lambdas[0]


def build_scheduler(optimizer, schedule: str, opt_steps: int, warmup_ratio: float, accum: int):
    """LambdaLR stepped once per MICRO-batch whose LR changes only between optimizer updates.
    accum=1 gives the per-optimizer-step scheduler (the cached-features loop)."""
    from torch.optim.lr_scheduler import LambdaLR
    lam = _diffusers_lambda(schedule, opt_steps, warmup_ratio)
    return LambdaLR(optimizer, lambda s: lam(s // accum))


def legacy_scheduler(optimizer, schedule: str, opt_steps: int, warmup_ratio: float, accum: int):
    """R-056's first construction (the bug): diffusers over the OPTIMIZER-step count, stepped per
    micro-batch. Kept only for the guard's negative test."""
    from diffusers.optimization import get_scheduler
    return get_scheduler(schedule, optimizer=optimizer, num_warmup_steps=warmup_steps(opt_steps, warmup_ratio, schedule),
                         num_training_steps=opt_steps)


def record_applied_lr(optimizer, trace: list) -> object:
    """Append the LR of every optimizer update that actually runs (a step pre-hook: accelerate's
    AcceleratedOptimizer skips the inner step on non-sync micro-batches, so they never reach it)."""
    return optimizer.register_step_pre_hook(lambda opt, args, kwargs: trace.append(float(opt.param_groups[0]["lr"])))


def simulate_lerobot(build, lr: float, schedule: str, opt_steps: int, warmup_ratio: float, accum: int,
                     weight_decay: float = 1e-5, grad_clip_norm: float = 1.0) -> list[float]:
    """The LR each optimizer update applies when `build(optimizer, schedule, opt_steps, warmup_ratio, accum)`
    is driven by lerobot_train's own path: AcceleratorConfig(gradient_accumulation=accum).build on the CPU,
    accelerator.prepare, update_policy once per micro-batch, for opt_steps * accum micro-batches.
    Run it in a fresh process (accelerate's state is a per-process singleton)."""
    import torch
    from types import SimpleNamespace
    from lerobot.configs.accelerator import AcceleratorConfig, GradientAccumulationConfig
    from lerobot.configs.parallelism import ParallelismConfig
    from lerobot.optim.optimizers import AdamWConfig
    from lerobot.scripts.lerobot_train import update_policy

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(4))

        def forward(self, batch):
            return (self.w * batch).pow(2).mean(), {}

    par = ParallelismConfig()
    par.resolve(1)
    acc = AcceleratorConfig(gradient_accumulation=GradientAccumulationConfig(steps=accum)).build(par, cpu=True)
    pol = Dummy()
    opt = AdamWConfig(lr=lr, weight_decay=weight_decay, grad_clip_norm=grad_clip_norm).build(
        [p for p in pol.parameters() if p.requires_grad])
    sched = build(opt, schedule, opt_steps, warmup_ratio, accum)
    trace: list[float] = []
    record_applied_lr(opt, trace)
    pol, opt, sched = acc.prepare(pol, opt, sched)
    metrics = SimpleNamespace(update_metrics=lambda d: None)
    x = torch.ones(4)
    for _ in range(opt_steps * accum):
        update_policy(metrics, pol, x, opt, grad_clip_norm, accelerator=acc, lr_scheduler=sched)
    return trace


def simulate_cached(build, lr: float, schedule: str, opt_steps: int, warmup_ratio: float) -> list[float]:
    """The cached-features loop's order: opt.step() then sched.step() once per optimizer update."""
    import torch
    p = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([p], lr=lr)
    sched = build(opt, schedule, opt_steps, warmup_ratio, 1)
    trace: list[float] = []
    record_applied_lr(opt, trace)
    for _ in range(opt_steps):
        p.grad = torch.ones(1)
        opt.step(); sched.step()
    return trace


def probe_steps(opt_steps: int, warmup_ratio: float, schedule: str) -> list[int]:
    w = warmup_steps(opt_steps, warmup_ratio, schedule)
    return sorted({k for k in (0, w, 500, 1000, 1500, opt_steps - 1) if 0 <= k < opt_steps})


def check(trace: list[float], lr: float, schedule: str, opt_steps: int, warmup_ratio: float,
          start: int = 0, final: bool = True) -> dict:
    """Compare `trace` (applied LR at optimizer steps start, start+1, ...) with the registered schedule.
    `pass` needs every step within TOL and, when `final`, the trace to end at step opt_steps - 1."""
    reg = [registered_lr(start + i, lr, schedule, opt_steps, warmup_ratio) for i in range(len(trace))]
    err = [abs(a - r) for a, r in zip(trace, reg)]
    probes = {k: {"applied": trace[k - start], "registered": reg[k - start], "abs_err": err[k - start]}
              for k in probe_steps(opt_steps, warmup_ratio, schedule) if 0 <= k - start < len(trace)}
    complete = (start + len(trace) == opt_steps) if final else (start + len(trace) <= opt_steps)
    ok = bool(trace) and complete and max(err) <= TOL
    return {"pass": ok, "tol": TOL, "n_updates": len(trace), "expected_updates": opt_steps - start, "start": start,
            "max_abs_err": max(err) if err else None, "probes": probes,
            "first_bad_step": next((start + i for i, e in enumerate(err) if e > TOL), None)}
