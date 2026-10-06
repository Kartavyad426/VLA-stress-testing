"""GR00T N1.7 LoRA fine-tuning on an 8 GB card: the port defaults that must change (R-055).

1. `model_params_fp32` defaults True and casts every parameter to fp32
   (modeling_groot.py:116): 12.6 GB, an instant OOM. The policy is instead
   loaded on the CPU, cast to bf16 and moved to the GPU -- the harness path
   (vla_harness/policies/lerobot_policy.py:51, R-021: 5.87 GiB).
2. GR00T's own `lora_*` config fields are read nowhere (configuration_groot.py:358).
   LoRA goes through LeRobot's generic PEFT path (policies/pretrained.py:381),
   with the targets below supplied as the policy's default PEFT targets.
3. Everything but the adapter stays frozen: wrap_with_peft freezes every base
   parameter, and `modules_to_save` is empty, so the projectors (state/action
   encoders, action decoder: 0.34 B, ~5 GB of Adam state) never train.
4. The DiT declares `gradient_checkpointing` and never reads it
   (cross_attention_dit.py:237, 415). `enable_block_checkpointing` wraps each
   transformer block in non-reentrant activation checkpointing instead.

LoRA targets: every nn.Linear under action_head.model.transformer_blocks (the
DiT) and action_head.vl_self_attention; r=16, alpha=16, dropout 0.1 by default
(R-056); rank and alpha are parameters (R-058 sweeps r in {4, 64}, alpha = r). PEFT
keeps adapter weights in fp32 (autocast_adapter_dtype) over bf16 base weights.
"""
from __future__ import annotations

import re

LORA_SCOPES = (r"action_head\.model\.transformer_blocks\.", r"action_head\.vl_self_attention\.")
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 16, 0.1
MAX_TRAINABLE = 25_000_000            # R-055's assertion, at r = 16 (19.1 M expected)
PARAMS_PER_RANK = 19_136_512 // 16    # measured: 248 Linears, sum(in+out) = 1,196,032


def max_trainable(r: int) -> int:
    """The trainable-count ceiling at rank r: R-055's 25 M at r <= 16, else 1.3 x the expected count."""
    return max(MAX_TRAINABLE, int(1.3 * PARAMS_PER_RANK * r))


def lora_target_names(model) -> list[str]:
    """Full module names of every nn.Linear inside the LoRA scopes (named from `model`)."""
    import torch.nn as nn
    scope = re.compile("|".join(LORA_SCOPES))
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and scope.search(n)]


def lora_target_regex(model) -> str:
    """A PEFT `target_modules` regex (re.fullmatch) naming exactly those Linears."""
    names = lora_target_names(model)
    if not names:
        raise RuntimeError("no Linear modules under the LoRA scopes: is this a GR00T N1.7 policy?")
    return "^(?:" + "|".join(re.escape(n) for n in names) + ")$"


def peft_defaults(policy, r: int = LORA_R, alpha: int = LORA_ALPHA) -> dict:
    return {"target_modules": lora_target_regex(policy), "modules_to_save": [],
            "lora_dropout": LORA_DROPOUT, "r": int(r), "lora_alpha": int(alpha)}


def install_peft_defaults(r: int = LORA_R, alpha: int = LORA_ALPHA) -> None:
    """Make GrootPolicy's default PEFT targets the ones above, so `--policy.use_peft`/
    `--peft.*` through lerobot_train builds exactly this adapter."""
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    GrootPolicy._get_default_peft_targets = lambda self: peft_defaults(self, r, alpha)


def cast_bf16_keep_adapter_fp32(policy, device):
    """Base weights bf16, adapter weights fp32, then onto `device`."""
    import torch
    policy = policy.to(torch.bfloat16)
    for n, p in policy.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()
    return policy.to(device)


def install_bf16_make_policy(module) -> None:
    """Patch `module.make_policy` (lerobot_train's import) to load on the CPU and cast to bf16
    before the GPU sees a single fp32 weight. model_params_fp32 is forced False."""
    import torch
    orig = module.make_policy

    def make_policy_bf16(cfg, *a, **kw):
        if getattr(cfg, "model_params_fp32", False):
            cfg.model_params_fp32 = False
        device = cfg.device
        cfg.device = "cpu"
        try:
            policy = orig(cfg, *a, **kw)
        finally:
            cfg.device = device
        policy = cast_bf16_keep_adapter_fp32(policy, torch.device(device))
        try:
            policy.config.device = str(device)
        except Exception:
            pass
        return policy

    module.make_policy = make_policy_bf16


def enable_block_checkpointing(policy) -> int:
    """Wrap every DiT and vl_self_attention transformer block in activation checkpointing
    (non-reentrant, RNG state preserved, active only in train mode). Returns the count."""
    import torch.utils.checkpoint as ckpt
    head = find_head(policy)
    n = 0
    for lst in (head.model.transformer_blocks, head.vl_self_attention.transformer_blocks):
        for block in lst:
            if getattr(block, "_vla_ckpt", False):
                continue
            fwd = block.forward

            def wrapped(*args, _fwd=fwd, _blk=block, **kwargs):
                if _blk.training:
                    return ckpt.checkpoint(_fwd, *args, use_reentrant=False, **kwargs)
                return _fwd(*args, **kwargs)

            block.forward = wrapped
            block._vla_ckpt = True
            n += 1
    return n


def find_head(policy):
    """The action head through a PeftModel/GrootPolicy wrapper."""
    m = policy
    for attr in ("base_model", "model"):
        if hasattr(m, "peft_config") and hasattr(m, attr):
            m = getattr(m, attr)
    from ..capture.groot_features import find_action_head
    h = find_action_head(m)
    if h is None and hasattr(m, "model"):
        h = find_action_head(m.model)
    if h is None:
        raise RuntimeError("no GR00T action head found")
    return h


def trainable_summary(policy) -> dict:
    tr = [(n, p.numel()) for n, p in policy.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in policy.parameters())
    n_tr = sum(k for _, k in tr)
    non_lora = [n for n, _ in tr if "lora_" not in n]
    return {"trainable": n_tr, "total": total, "n_tensors": len(tr), "non_lora_trainable": non_lora[:10],
            "n_non_lora": len(non_lora)}


def assert_trainable(policy, limit: int = MAX_TRAINABLE, expected: int | None = None) -> dict:
    s = trainable_summary(policy)
    if expected is not None and s["trainable"] != expected:
        raise RuntimeError(f"{s['trainable']:,} trainable parameters, expected {expected:,} for this rank")
    print(f"[groot_lora] trainable parameters: {s['trainable']:,} of {s['total']:,} "
          f"({s['n_tensors']} tensors); non-LoRA trainable: {s['n_non_lora']}", flush=True)
    if s["trainable"] >= limit:
        raise RuntimeError(f"{s['trainable']:,} trainable parameters >= {limit:,}")
    if s["n_non_lora"]:
        raise RuntimeError(f"non-adapter parameters are trainable: {s['non_lora_trainable']}")
    if s["trainable"] == 0:
        raise RuntimeError("nothing is trainable")
    return s
