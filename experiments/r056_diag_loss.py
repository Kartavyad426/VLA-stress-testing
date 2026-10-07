"""D20(i), revised (user 2026-10-06): fixed-seed training loss per subset, post-mortem §9 item A
(docs/R056_POSTMORTEM.html). No rollouts.

  PYTHONPATH=$PWD/third_party/LIBERO-plus flock /tmp/vla_gpu.lock .venvs/libero-plus/bin/python \
      experiments/r056_diag_loss.py [--limit N] [--out runs/r056_diag]

Models: base, R-056 r16 at opt 500 / 1000 / 2000, R-058 r4 at opt 500 (the selected checkpoints of
R-056 and R-058 are opt1000 and opt500). Checkpoint dirs are picked by vla_train_meta.json's opt_step,
never by directory name (lerobot_train names them by micro-step).

Subsets (the post-mortem's four sample sources), one frame per minted episode, frames drawn with a
fixed RNG so every model sees the same frames:
  minted_f0       frame 0 of each of the 100 minted episodes (the whole 16-step label is the correction)
  minted_f1_15    one frame in 1..15 per episode (label partly the driven nominal chunk)
  minted_f16_47   one frame in 16..47 per episode (the base's own actions)
  ipec            100 frames uniform over IPEC libero_spatial_no_noops (the base's training data)

Each frame is collated --draws times into one batch and evaluated under torch.manual_seed(seed(subset,
frame)), set immediately before every forward, so tau, epsilon and any other draw are identical across
models for the same frame (paired). Eval mode: LoRA and DiT dropout off; the training preprocessor with
its steps in eval mode (no crop / state dropout; R-056 trained with augmentation off).

One GR00T load (the harness's streaming bf16 path, bit-identical to the training path's
from_pretrained + .to(bf16), V3). Adapters are attached to it as named PEFT adapters, switched with
set_adapter; base is the same model under disable_adapter().

Writes <out>/loss_rows.jsonl (one row per model x frame: mean loss over the draws) and
<out>/loss_summary.json (per model x subset: mean, s.e.; paired difference to base with a 95% t
interval and the share of frames below base) plus the checks run on the way.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.getcwd())

import numpy as np

MINTED_ROOT, MINTED_REPO = "data/r054_train48", "local/r054_train48"
MODELS = [("r16_opt500", "runs/r056_r16_train", 500), ("r16_opt1000", "runs/r056_r16_train", 1000),
          ("r16_opt2000", "runs/r056_r16_train", 2000), ("r4_opt500", "runs/r058_r4_train", 500)]
SUBSETS = ["minted_f0", "minted_f1_15", "minted_f16_47", "ipec"]


def adapter_dir(train_dir, opt_step):
    hits = []
    for mp in glob.glob(os.path.join(train_dir, "checkpoints", "[0-9]*", "pretrained_model", "vla_train_meta.json")):
        m = json.load(open(mp))
        if m["opt_step"] == opt_step:
            if m.get("schedule_verified") is not True:
                sys.exit(f"REFUSED: {mp} not schedule_verified")
            hits.append(os.path.dirname(mp))
    if len(hits) != 1:
        sys.exit(f"{train_dir}: expected one checkpoint at opt {opt_step}, found {hits}")
    return hits[0]


def pick_frames(parts, limit, seed):
    """[(subset, part_index, dataset_index, episode, frame)], identical on every call with the same seed."""
    from experiments.r056_train import episode_indices
    rng = np.random.RandomState(seed)
    minted, ipec = parts
    out = []
    eps = list(range(minted.num_episodes))
    f1 = rng.randint(1, 16, size=len(eps))
    f2 = rng.randint(16, 48, size=len(eps))
    for sub, frames in (("minted_f0", [0] * len(eps)), ("minted_f1_15", f1), ("minted_f16_47", f2)):
        for e, f in list(zip(eps, frames))[:limit]:
            idx = episode_indices(minted, e)
            assert len(idx) == 48, (e, len(idx))
            out.append((sub, 0, idx[int(f)], e, int(f)))
    for i in sorted(rng.choice(len(ipec), size=min(limit, 100), replace=False)):
        out.append(("ipec", 1, int(i), None, None))
    return out


def frame_seed(base, sub, ds_index):
    return (base * 1_000_003 + SUBSETS.index(sub) * 100_000_007 + int(ds_index)) % (2**31 - 1)


def tstat_ci(d):
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return None
    from scipy import stats
    se = d.std(ddof=1) / math.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return [float(d.mean() - h), float(d.mean() + h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/r056_diag")
    ap.add_argument("--limit", type=int, default=100, help="frames per subset (smoke: 4)")
    ap.add_argument("--draws", type=int, default=8, help="(tau, eps) draws per frame, one batch")
    ap.add_argument("--seed", type=int, default=2020)
    a = ap.parse_args()
    import torch
    from torch.utils.data import default_collate
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
    from peft import PeftModel
    from experiments.r056_eval import mem_report
    from experiments.r056_train import BASE, CKPT, REPLAY, build_parts, training_preprocessor
    from vla_harness.policies.lerobot_policy import _from_pretrained_cast
    from vla_harness.training.feature_cache import augmentation_report

    os.makedirs(a.out, exist_ok=True)
    rows_p = os.path.join(a.out, "loss_rows.jsonl" if a.limit >= 100 else f"loss_rows_smoke{a.limit}.jsonl")
    if os.path.exists(rows_p):
        os.remove(rows_p)
    t0 = time.time()
    adapters = [(n, adapter_dir(td, st)) for n, td, st in MODELS]
    mem_report("start")

    cfg = PreTrainedConfig.from_pretrained(CKPT)          # as r056_train.load_training_policy
    cfg.pretrained_path = CKPT
    cfg.base_model_path, cfg.embodiment_tag = BASE, "libero_sim"
    cfg.model_params_fp32 = False
    cfg.tune_projector = False
    cfg.device = "cpu"
    pol = _from_pretrained_cast(get_policy_class(cfg.type), CKPT, cfg, torch.bfloat16)
    cfg.device = "cuda"
    pol.config.device = "cuda"
    mem_report("base built (cpu)")
    peft = PeftModel.from_pretrained(pol, adapters[0][1], adapter_name=adapters[0][0], is_trainable=False)
    for n, d in adapters[1:]:
        peft.load_adapter(d, adapter_name=n, is_trainable=False)
    peft.to("cuda").eval()
    mem_report("adapters attached, on cuda")
    checks = {"adapters": {}}
    for n, d in adapters:
        c = json.load(open(os.path.join(d, "adapter_config.json")))
        nmod = sum(1 for _, m in peft.named_modules() if hasattr(m, "lora_A") and n in m.lora_A)
        checks["adapters"][n] = {"dir": d, "r": c["r"], "lora_alpha": c["lora_alpha"], "lora_modules": nmod}
        if nmod == 0:
            sys.exit(f"adapter {n}: no LoRA modules attached")

    pre, _ = training_preprocessor(cfg)
    for s in pre.steps:
        if hasattr(s, "training"):
            s.training = False
    aug = augmentation_report(pre)
    checks["augmentation"] = aug
    if aug.get("random_crop_active"):
        sys.exit(f"augmentation active in the preprocessor: {aug}")
    ns = argparse.Namespace(minted_repo=MINTED_REPO, minted_root=MINTED_ROOT, episodes=None, shares="0.5,0.5",
                            replay=REPLAY)
    parts, names = build_parts(cfg, ns)
    frames = pick_frames(parts, a.limit, a.seed)
    print(f"[diag] {len(frames)} frames: " + ", ".join(f"{s} {sum(f[0] == s for f in frames)}" for s in SUBSETS), flush=True)

    labels = ["base"] + [n for n, _ in adapters]

    def loss_of(label, b, sd):
        torch.manual_seed(sd)
        with torch.no_grad():
            if label == "base":
                with peft.disable_adapter():
                    loss, _ = peft.forward(b)
            else:
                peft.set_adapter(label)
                loss, _ = peft.forward(b)
        return float(loss)

    with open(rows_p, "a") as fo:
        for k, (sub, pi, i, ep, fr) in enumerate(frames):
            ds = parts[pi]
            item = ds[i]
            b = _preprocess_dataset_batch(default_collate([item] * a.draws), ds.meta.camera_keys, {}, pre)
            sd = frame_seed(a.seed, sub, i)
            row = {"subset": sub, "part": names[pi], "index": i, "episode": ep, "frame": fr, "seed": sd,
                   "draws": a.draws, "loss": {lb: loss_of(lb, b, sd) for lb in labels}}
            if k == 0:                                         # determinism and adapter-switch checks
                rep = loss_of("base", b, sd)
                checks["base_repeat_bit_identical"] = rep == row["loss"]["base"]
                checks["after_switch_base_repeat_bit_identical"] = loss_of("base", b, sd) == row["loss"]["base"]
                checks["adapter_differs_from_base_frame0"] = {lb: row["loss"][lb] != row["loss"]["base"] for lb in labels[1:]}
                mem_report("after first frame")
            fo.write(json.dumps(row) + "\n"); fo.flush()
            if k % 20 == 0:
                print(f"  [{time.time()-t0:6.0f}s] {k+1}/{len(frames)} {sub} " +
                      " ".join(f"{lb} {row['loss'][lb]:.4f}" for lb in labels), flush=True)
    mem_report("end")

    R = [json.loads(l) for l in open(rows_p)]
    summ = {}
    for sub in SUBSETS:
        rs = [r for r in R if r["subset"] == sub]
        if not rs:
            continue
        base = np.array([r["loss"]["base"] for r in rs])
        d = {"n": len(rs)}
        for lb in labels:
            x = np.array([r["loss"][lb] for r in rs])
            e = {"mean": float(x.mean()), "se": float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else None,
                 "median": float(np.median(x))}
            if lb != "base":
                diff = x - base
                e.update({"diff_vs_base_mean": float(diff.mean()), "diff_vs_base_ci95": tstat_ci(diff),
                          "ratio_of_means_vs_base": float(x.mean() / base.mean()),
                          "share_frames_below_base": float((diff < 0).mean())})
            d[lb] = e
        summ[sub] = d
    out = {"what": "D20(i) revised: fixed-seed loss per subset (post-mortem §9 A)", "limit": a.limit, "draws": a.draws,
           "seed": a.seed, "models": {"base": CKPT, **{n: d for n, d in adapters}}, "checks": checks,
           "subsets": summ, "rows": rows_p, "wall_s": round(time.time() - t0, 1)}
    sp = os.path.join(a.out, "loss_summary.json" if a.limit >= 100 else f"loss_summary_smoke{a.limit}.json")
    json.dump(out, open(sp, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("checks", "subsets", "wall_s")}, indent=1))


if __name__ == "__main__":
    main()
