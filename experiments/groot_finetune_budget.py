"""Can we fine-tune GR00T N1.7 on this 8 GB card? Arithmetic, not vibes.

Parameter counts are MEASURED from the cached checkpoint's safetensors header
(nvidia/gr00t17-lerobot-libero_spatial-640), not taken from a model card.

    .venvs/lerobot/bin/python experiments/groot_finetune_budget.py

Assumptions are named and printed so they can be argued with. The one genuinely
uncertain term is ACTIVATION memory, which depends on batch size, image
resolution and action horizon; it is reported as a separate budget line rather
than folded into a single number.
"""
from __future__ import annotations

import glob
import json
import struct
from collections import defaultdict

CARD_GB = 8151 / 1024          # nvidia-smi total, this machine
FREE_GB = 7429 / 1024          # nvidia-smi free with only the display on it
CKPT = ("/home/imerit/.cache/huggingface/hub/"
        "models--nvidia--gr00t17-lerobot-libero_spatial-640/snapshots/*/model.safetensors")

# bytes per parameter, by role
BF16 = 2
NF4 = 0.55                     # 4-bit weights + quant constants, approx
GRAD_BF16 = 2
ADAM_FP32 = 8                  # m + v, fp32
ADAM_8BIT = 2                  # m + v, 8-bit (bitsandbytes)


def measured_params() -> dict[str, int]:
    f = glob.glob(CKPT)
    if not f:
        raise SystemExit("checkpoint not cached; nothing to measure")
    fh = open(f[0], "rb")
    n = struct.unpack("<Q", fh.read(8))[0]
    h = json.loads(fh.read(n))
    g: dict[str, int] = defaultdict(int)
    for k, v in h.items():
        if k == "__metadata__":
            continue
        s = 1
        for d in v["shape"]:
            s *= d
        g[".".join(k.split(".")[:3])] += s
    return dict(g)


def budget(name: str, trainable: int, frozen: int, *,
           frozen_bytes: float = BF16, adam: float = ADAM_FP32) -> None:
    w_f = frozen * frozen_bytes / 1e9
    w_t = trainable * BF16 / 1e9
    grad = trainable * GRAD_BF16 / 1e9
    opt = trainable * adam / 1e9
    total = w_f + w_t + grad + opt
    head = FREE_GB - total
    verdict = f"fits, {head:5.2f} GB left for activations" if head > 0 \
        else f"OVER BY {-head:5.2f} GB before any activations"
    print(f"  {name:52s}")
    print(f"    trainable {trainable/1e9:5.3f}B  frozen {frozen/1e9:5.3f}B"
          f"   weights {w_f+w_t:5.2f}  grads {grad:5.2f}  optim {opt:5.2f}"
          f"  = {total:5.2f} GB")
    print(f"    -> {verdict}")


def main() -> None:
    p = measured_params()
    tot = sum(p.values())
    backbone = p["_groot_model.backbone.model"]
    diffusion = p["_groot_model.action_head.model"]
    small = tot - backbone - diffusion          # encoders, decoder, vl-attn
    action_head = tot - backbone

    print(f"MEASURED from the cached checkpoint [F]")
    print(f"  total          {tot:,} = {tot/1e9:.3f}B   ({tot*BF16/1e9:.2f} GB bf16)")
    print(f"  backbone       {backbone:,} ({backbone/tot*100:.1f}%)  Qwen3-VL")
    print(f"  action head    {action_head:,} ({action_head/tot*100:.1f}%)"
          f"  <- LARGER than the backbone")
    print(f"    diffusion    {diffusion:,}")
    print(f"    encoders etc {small:,}")
    print(f"\nHARDWARE: {CARD_GB:.2f} GB card, {FREE_GB:.2f} GB free\n")

    print("CONFIGURATIONS (AdamW fp32 states unless noted)\n")
    budget("1. full fine-tune", tot, 0)
    budget("2. LeRobot GR00T DEFAULT (tune_diffusion+projector)",
           action_head, backbone)
    budget("3. default + 8-bit Adam (needs bitsandbytes)",
           action_head, backbone, adam=ADAM_8BIT)
    budget("4. encoders/decoder only, diffusion frozen",
           small, backbone + diffusion)
    budget("5. encoders only + 8-bit Adam",
           small, backbone + diffusion, adam=ADAM_8BIT)

    # LoRA: adapters are a small fraction of the wrapped module.
    for r_pct, lbl in ((0.01, "r~16"), (0.005, "r~8")):
        lora = int(action_head * r_pct)
        budget(f"6. LoRA {lbl} on action head, base bf16", lora, tot)
        budget(f"7. QLoRA {lbl}, base 4-bit (NOT SUPPORTED, see notes)",
               lora, tot, frozen_bytes=NF4)
        break

    print("\nNOTES, all checked on this machine")
    print("  * bitsandbytes NOT INSTALLED, and lerobot contains no")
    print("    load_in_4bit / BitsAndBytes / quantization_config anywhere [F].")
    print("    So rows 3, 5 and 7 are NOT reachable without new work.")
    print("  * peft 0.21.0 IS installed and lerobot wires it via cfg.use_peft [F].")
    print("  * GrootConfig has lora_rank / lora_alpha / lora_dropout /")
    print("    lora_full_model, and tune_llm=False, tune_visual=False by")
    print("    default [F] -- selective freezing is already built in.")
    print("  * gradient checkpointing supported (groot_n1_7.py:463, :823) [F];")
    print("    trades ~30-40% step time for a large activation reduction.")
    print("  * ACTIVATIONS are the uncertain term and are NOT included above.")
    print("    At batch 1 with 2x256x256 images they are not negligible;")
    print("    any row with under ~1.5 GB of headroom should be treated as")
    print("    unproven until measured.")


if __name__ == "__main__":
    main()
