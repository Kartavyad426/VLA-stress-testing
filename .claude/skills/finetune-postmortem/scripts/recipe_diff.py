"""Fine-tune recipe vs the base checkpoint's own training recipe, side by side.

  .venvs/groot/bin/python .claude/skills/finetune-postmortem/scripts/recipe_diff.py \
      --base nvidia/gr00t17-lerobot-libero_spatial-640 \
      --finetune runs/r056_r16_train/checkpoints/last/pretrained_model \
      [--report runs/r056_r16_train_report.json]

--base is a local dir or an HF repo id (read from the local HF cache, never downloaded). Prints the
fields that decide training dynamics -- batch (and effective batch under accumulation), steps, LR,
warmup, schedule, weight decay, clip, what is trainable (tune_* flags, PEFT), augmentation (image
transforms, state dropout), dataset, chunk size -- and marks rows that differ. The --report (r056_train
style) supplies what train_config.json cannot: effective batch, the actual augmentation state of the
processors, trainable parameter count, LoRA rank/alpha/dropout.
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def find_config(x):
    if os.path.isdir(x):
        p = os.path.join(x, "train_config.json")
        return p if os.path.exists(p) else None
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    hits = glob.glob(os.path.join(hub, "models--" + x.replace("/", "--"), "snapshots", "*", "train_config.json"))
    return sorted(hits)[-1] if hits else None


def g(d, path, default=None):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


FIELDS = [
    ("batch_size", "batch_size"), ("steps", "steps"), ("seed", "seed"),
    ("grad accumulation", "accelerator.gradient_accumulation.steps"),
    ("optimizer", "optimizer.type"), ("lr", "optimizer.lr"), ("weight_decay", "optimizer.weight_decay"),
    ("grad_clip_norm", "optimizer.grad_clip_norm"), ("scheduler", "scheduler.name"),
    ("warmup steps", "scheduler.num_warmup_steps"),
    ("dataset", "dataset.repo_id"), ("image transforms on", "dataset.image_transforms.enable"),
    ("image transforms", "dataset.image_transforms.tfs"),
    ("tune_llm", "policy.tune_llm"), ("tune_visual", "policy.tune_visual"), ("tune_projector", "policy.tune_projector"),
    ("tune_diffusion_model", "policy.tune_diffusion_model"), ("tune_vlln", "policy.tune_vlln"),
    ("state_dropout_prob", "policy.state_dropout_prob"), ("chunk_size", "policy.chunk_size"),
    ("n_action_steps", "policy.n_action_steps"), ("use_bf16", "policy.use_bf16"),
    ("model_params_fp32", "policy.model_params_fp32"), ("peft", "peft"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--finetune", required=True)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    pb, pf = find_config(a.base), find_config(a.finetune)
    if not pb or not pf:
        raise SystemExit(f"train_config.json not found: base={pb} finetune={pf}")
    B, F = json.load(open(pb)), json.load(open(pf))
    print(f"base:     {pb}\nfinetune: {pf}\n")
    print(f"{'field':<22} {'base':<34} {'finetune':<34}")
    for label, path in FIELDS:
        vb, vf = g(B, path), g(F, path)
        sb, sf = json.dumps(vb)[:34], json.dumps(vf)[:34]
        print(f"{label:<22} {sb:<34} {sf:<34}{'  *' if vb != vf else ''}")
    if a.report:
        R = json.load(open(a.report))
        print("\nfrom the run report:")
        for k in ("effective_batch", "batch_size", "accumulation", "opt_steps", "lora", "augmentation", "trainable", "shares", "parts"):
            if k in R:
                print(f"  {k}: {json.dumps(R[k])[:300]}")
        eb_b = g(B, "batch_size", 0) * (g(B, "accelerator.gradient_accumulation.steps", 1) or 1)
        if R.get("effective_batch") and eb_b:
            print(f"\n  effective batch ratio base/finetune = {eb_b}/{R['effective_batch']} = {eb_b / R['effective_batch']:.0f}x"
                  f" at LR {g(B, 'optimizer.lr')} vs {g(F, 'optimizer.lr')}")


if __name__ == "__main__":
    main()
