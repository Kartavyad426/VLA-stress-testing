"""R-055 gates: zero-step conformance, overfit, memory (and the cached-feature equivalence).
Writes runs/r055/gates.json. Every mode is a GPU job: flock, announced, and only on the user's go.

  ENV="PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8"
  PY=.venvs/libero-plus/bin/python
  flock /tmp/vla_gpu.lock $PY experiments/r055_gates.py conformance
  flock /tmp/vla_gpu.lock $PY experiments/r055_gates.py overfit  --minted-root data/r054_train48 --minted-repo local/r054_train48
  flock /tmp/vla_gpu.lock $PY experiments/r055_gates.py ckpt-equiv --minted-root data/r054_train48 --minted-repo local/r054_train48
  flock /tmp/vla_gpu.lock $PY experiments/r055_gates.py memory   --minted-root data/r054_train48 --minted-repo local/r054_train48
  flock /tmp/vla_gpu.lock $PY experiments/r055_gates.py cache-equiv --minted-root ... --minted-repo ...   # fallback only

conformance  The training-path model (bf16 base + untrained LoRA, B = 0, eval
             mode) and the harness policy compute the forward-0 chunk on the ten
             R-044 instances under the harness's fixed noise (seed 1000*0 + 0).
             The harness path gets the raw env observation through its own
             processors (LiberoProcessorStep flip, 360 render); the training
             path gets that observation in DATASET form -- converted by the
             export adapter exactly as a minted frame would be -- through the
             preprocessor lerobot_train builds. Pass: |a_train - a_harness| <=
             1e-3 on every dim of the normalised chunk, 10/10. This tests the
             model build, the dtype and the converter together.
overfit      r056_train.py on ONE minted episode, 300 optimizer steps at LR 1e-4
             (constant after warmup is not needed: cosine over 300), batch 1,
             no replay; pass if the fixed-seed loss on that episode falls
             below 10% of its step-0 value.
ckpt-equiv   gate 2b (ruling 2026-09-28): on one fixed batch, in train mode, with
             tau/epsilon/dropout seeded identically, the loss and every adapter
             gradient match with and without block checkpointing (bf16 tolerance:
             relative 1e-2). LoRA B is set to small seeded values first, so the
             gradients into A are non-zero and tested too.
memory       r056_train.py for 20 micro-steps at batch 1 and at batch 2, one
             process each, recording torch.cuda.max_memory_allocated and the
             nvidia-smi peak. Expectations: <= 7.4 GiB at batch 1, <= 7.8 at 2.
cache-equiv  the cached-feature path's loss equals the uncached path's on the
             same samples with fixed tau/epsilon (bf16 tolerance), micro-batch 1.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.getcwd())

import numpy as np

OUT = "runs/r055/gates.json"
SEL = "experiments/repro/r044_selection_ris.json"
TOL = 1e-3


def save(key, val):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res[key] = val
    json.dump(res, open(OUT, "w"), indent=1, default=float)


class ChunkTap:
    """Record the head's normalised action chunk per call (outermost wrapper of get_action_with_features)."""

    def __init__(self, head):
        self.head, self.out = head, []
        self._had = "get_action_with_features" in vars(head)
        self._orig = head.get_action_with_features
        head.get_action_with_features = self._w

    def _w(self, *a, **kw):
        r = self._orig(*a, **kw)
        self.out.append(r["action_pred"].detach().float().cpu().numpy())
        return r

    def detach(self):
        if self._had:
            self.head.get_action_with_features = self._orig
        else:
            del self.head.get_action_with_features


def forward0(policy_obj, head_owner, call, noise_seed=0):
    """Run `call()` (one policy forward) under the harness's fixed noise; return the chunk."""
    from vla_harness.capture.splice import attach_splice
    from vla_harness.capture.groot_features import find_action_head
    head = find_action_head(head_owner)
    gen = attach_splice(head_owner, source=lambda i: None, drive=None, noise_key=lambda i, s=noise_seed: 1000 * s + i)
    tap = ChunkTap(head)         # outermost: sees what the splice returns
    try:
        call()
    finally:
        tap.detach(); gen.detach()
    return tap.out[0][0]


def conformance(a):
    import torch
    from torch.utils.data import default_collate
    from experiments.r054_mint import build_policy, make_env
    from experiments.r056_train import load_training_policy, training_preprocessor
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
    from vla_harness.data.export_lerobot import LiberoIpecAdapter
    from vla_harness.schema import PerturbationSpec
    ad = LiberoIpecAdapter()
    insts = json.load(open(SEL))["instances"][: a.limit]
    # 1. the harness policy, on the raw observation
    hp = build_policy(); hp.reset()
    obs_by, a_h = {}, {}
    for inst in insts:
        env = make_env(inst["task_id"])
        obs = env.reset(inst["seed"], PerturbationSpec())
        obs_by[inst["task_id"]] = obs
        hp.reset()
        a_h[inst["task_id"]] = forward0(hp, hp._policy, lambda: hp(obs.policy_view()))
        del env
    del hp; torch.cuda.empty_cache()
    # 2. the training-path model, on the observation in dataset form
    peft, cfg = load_training_policy()
    pre, _ = training_preprocessor(cfg)
    peft.eval()
    inner = peft.base_model.model
    rows = []
    for inst in insts:
        obs = obs_by[inst["task_id"]]
        f = obs.frames
        sample = {"observation.images.image": torch.from_numpy(ad.image(f["image"])).permute(2, 0, 1),
                  "observation.images.wrist_image": torch.from_numpy(ad.image(f["image2"])).permute(2, 0, 1),
                  "observation.state": torch.from_numpy(ad.state({k: obs.state[k] for k in ("eef_pos", "eef_quat", "gripper_qpos")})),
                  "task": obs.instruction}
        b = _preprocess_dataset_batch(default_collate([sample]), ["observation.images.image", "observation.images.wrist_image"], {}, pre)
        inner.reset()
        with torch.no_grad():
            a_t = forward0(inner, inner, lambda: inner.predict_action_chunk(b))
        d = np.abs(a_t[:16, :7] - a_h[inst["task_id"]][:16, :7])
        rows.append({"task_id": inst["task_id"], "max_abs": float(d.max()), "per_dim_max": d.max(axis=0).tolist(),
                     "pass": bool(d.max() <= TOL)})
        print(f"  {inst['task_id']}: max |a_train - a_harness| = {d.max():.2e}", flush=True)
    k = sum(r["pass"] for r in rows)
    res = {"instances": rows, "pass_count": k, "n": len(rows), "tol": TOL, "pass": k == len(rows),
           "note": "normalised action space, first 16 steps x 7 live dims; untrained adapter, eval mode"}
    save("conformance", res)
    print(f"conformance {k}/{len(rows)}")


def run_train(args, report):
    cmd = [sys.executable, "experiments/r056_train.py", *args, "--report", report]
    print(" ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    return rc, (json.load(open(report)) if os.path.exists(report) else None)


def overfit(a):
    rep = "runs/r055/overfit_report.json"
    rc, r = run_train(["--minted-root", a.minted_root, "--minted-repo", a.minted_repo, "--shares", "",
                       "--episodes", str(a.episode), "--opt-steps", "300", "--batch-size", "1", "--effective-batch", "1",
                       "--lr", "1e-4", "--lr-schedule", a.lr_schedule, "--out", "runs/r055/overfit", "--no-save",
                       "--eval-loss-episode", "0"], rep)
    res = {"rc": rc, "episode": a.episode, "lr_schedule": a.lr_schedule}
    if r:
        l0, l1 = r.get("eval_loss_step0"), r.get("eval_loss_final")
        res.update({"loss_step0": l0, "loss_final": l1, "ratio": (l1 / l0) if l0 else None,
                    "pass": bool(l0 and l1 is not None and l1 < 0.1 * l0), "trainable": r.get("trainable")})
    save("overfit", res)
    print(json.dumps(res, indent=1, default=float))


def ckpt_equiv(a):
    import torch
    from torch.utils.data import default_collate
    from experiments.r056_train import build_parts, load_training_policy, training_preprocessor
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
    from vla_harness.training import groot_lora as G
    peft, cfg = load_training_policy(r=a.lora_r, alpha=a.lora_r)
    pre, _ = training_preprocessor(cfg)
    ns = argparse.Namespace(minted_repo=a.minted_repo, minted_root=a.minted_root, episodes=None, shares="", replay=None)
    parts, _ = build_parts(cfg, ns)
    g = torch.Generator().manual_seed(55)
    for n, p in peft.named_parameters():
        if "lora_B" in n:
            p.data = (torch.randn(p.shape, generator=g) * 1e-3).to(p.device, p.dtype)
    b = _preprocess_dataset_batch(default_collate([parts[0][a.episode_frame]]), parts[0].meta.camera_keys, {}, pre)
    params = {n: p for n, p in peft.named_parameters() if p.requires_grad}

    def run():
        for p in params.values():
            p.grad = None
        peft.train()
        torch.manual_seed(2055)
        loss, _ = peft.forward(b)
        loss.backward()
        return float(loss), {n: p.grad.detach().float().clone() for n, p in params.items()}

    l0, g0 = run()
    nblocks = G.enable_block_checkpointing(peft)
    l1, g1 = run()
    rel = lambda x, y: float((x - y).norm() / max(float(y.norm()), 1e-12))
    worst = max(((rel(g1[n], g0[n]), n) for n in g0 if float(g0[n].norm()) > 0), default=(0.0, None))
    zero = [n for n in g0 if float(g0[n].norm()) == 0]
    res = {"loss_plain": l0, "loss_ckpt": l1, "loss_rel": abs(l1 - l0) / max(abs(l0), 1e-12),
           "grad_worst_rel": worst[0], "grad_worst_tensor": worst[1], "n_tensors": len(g0), "n_zero_grad": len(zero),
           "checkpointed_blocks": nblocks, "tol_rel": 1e-2, "lora_r": a.lora_r}
    res["pass"] = res["loss_rel"] <= 1e-2 and worst[0] <= 1e-2 and not zero
    save(f"ckpt_equiv_r{a.lora_r}", res)
    print(json.dumps(res, indent=1, default=float))


def overfit_diag(a):
    """NOT a gate (user, 2026-09-29): the overfit curve over 1500 constant-LR steps on the same episode.
    Written to runs/r055/gates.json["overfit_diag_1500"] and runs/r055/overfit_diag_1500.json; never to "overfit"."""
    curve = "runs/r055/overfit_diag_1500.json"
    rep = "runs/r055/overfit_diag_1500_report.json"
    rc, r = run_train(["--minted-root", a.minted_root, "--minted-repo", a.minted_repo, "--shares", "",
                       "--episodes", str(a.episode), "--opt-steps", "1500", "--batch-size", "1", "--effective-batch", "1",
                       "--lr", "1e-4", "--lr-schedule", "constant", "--out", "runs/r055/overfit_diag", "--no-save",
                       "--eval-loss-episode", "0", "--eval-every", "100", "--curve", curve], rep)
    c = json.load(open(curve)) if os.path.exists(curve) else {}
    ev = c.get("eval", [])
    l0 = ev[0]["fixed_seed_loss"] if ev else None
    res = {"rc": rc, "diagnostic": True, "not_a_gate": True, "episode": a.episode, "steps": 1500, "lr": 1e-4,
           "lr_schedule": "constant", "curve_file": curve,
           "ratios": [{"opt_step": e["opt_step"], "ratio": e["fixed_seed_loss"] / l0} for e in ev] if l0 else None,
           "final_ratio": (c.get("final_fixed_seed_loss") / l0) if l0 and c.get("final_fixed_seed_loss") else None}
    save("overfit_diag_1500", res)
    print(json.dumps(res, indent=1, default=float))


def memory(a):
    out = {}
    for bs in (1, 2):
        rep = f"runs/r055/memory_r{a.lora_r}_b{bs}.json"
        rc, r = run_train(["--minted-root", a.minted_root, "--minted-repo", a.minted_repo,
                           "--batch-size", str(bs), "--effective-batch", str(bs), "--max-micro-steps", "20",
                           "--out", f"runs/r055/memory_r{a.lora_r}_b{bs}", "--no-save", "--lora-r", str(a.lora_r)] +
                          (["--cached-features", a.cached_features] if a.cached_features else []), rep)
        out[f"batch{bs}"] = {"rc": rc, **({k: r.get(k) for k in ("torch_max_memory_allocated_gib", "torch_max_memory_reserved_gib",
                                                                "nvidia_smi_peak_gib", "trainable", "mode")} if r else {})}
    b1, b2 = out["batch1"].get("nvidia_smi_peak_gib"), out["batch2"].get("nvidia_smi_peak_gib")
    out["E3_batch1_le_7.4GiB"] = (b1 <= 7.4) if b1 else None
    out["E4_batch2_le_7.8GiB"] = (b2 <= 7.8) if b2 else None
    out["mode"] = "cached_features" if a.cached_features else "vlm_resident"
    out["lora_r"] = a.lora_r
    out["note"] = "training runs at micro-batch 1 x accumulation 8; batch 2 is a memory readout only"
    save(("memory_cached" if a.cached_features else "memory") + ("" if a.lora_r == 16 else f"_r{a.lora_r}"), out)
    print(json.dumps(out, indent=1, default=float))


def cache_equiv(a):
    """Uncached vs cached loss on the same samples, tau/epsilon/dropout fixed by seeding
    immediately before the head; micro-batch 1 (exact regime)."""
    import torch
    from torch.utils.data import default_collate
    from transformers.feature_extraction_utils import BatchFeature
    from experiments.r056_train import build_parts, load_training_policy, training_preprocessor
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
    from vla_harness.training.feature_cache import (FeatureCache, collate_backbone, precompute, strip_vlm_encode)
    peft, cfg = load_training_policy(r=a.lora_r, alpha=a.lora_r)
    pre, _ = training_preprocessor(cfg)
    ns = argparse.Namespace(minted_repo=a.minted_repo, minted_root=a.minted_root, episodes=None, shares="", replay=None)
    parts, names = build_parts(cfg, ns)
    cdir = a.cached_features or "runs/r055/cache_equiv_features"
    meta = precompute(peft.base_model.model, pre, parts[:1], names[:1], cdir, "nvidia/gr00t17-lerobot-libero_spatial-640",
                      limit=a.n)
    cache = FeatureCache(cdir)
    inner = peft.base_model.model
    model = inner._groot_model
    head = model.action_head
    lite = strip_vlm_encode(pre)
    peft.train()
    rows = []
    for i in range(a.n):
        s = parts[0][i]
        b = _preprocess_dataset_batch(default_collate([s]), parts[0].meta.camera_keys, {}, pre)
        inputs = inner._filter_groot_inputs(b, include_action=True)
        bi, ai = model.prepare_input(inputs)
        with torch.no_grad():
            bo = model.backbone(bi)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            torch.manual_seed(7 + i)
            l_un = float(head(BatchFeature(data=dict(bo)), ai)["loss"])
        bl = _preprocess_dataset_batch(default_collate([s]), parts[0].meta.camera_keys, {}, lite)
        ai2 = head.prepare_input(inner._filter_groot_inputs(bl, include_action=True))
        ai2 = BatchFeature(data={k: (v.to("cuda", dtype=model.dtype) if torch.is_floating_point(v) else v.to("cuda"))
                                 if isinstance(v, torch.Tensor) else v for k, v in ai2.items()})
        bo2 = collate_backbone([cache.get(0, i)], "cuda", dtype=bo["backbone_features"].dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            torch.manual_seed(7 + i)
            l_c = float(head(bo2, ai2)["loss"])
        rows.append({"i": i, "uncached": l_un, "cached": l_c, "rel": abs(l_un - l_c) / max(abs(l_un), 1e-8)})
    worst = max(r["rel"] for r in rows)
    res = {"rows": rows, "worst_rel": worst, "tol_rel": 1e-2, "pass": worst <= 1e-2,
           "bytes_per_frame_measured": meta.get("bytes_per_frame_measured"), "augmentation": meta.get("augmentation")}
    res["lora_r"] = a.lora_r
    save("cache_equiv" + ("" if a.lora_r == 16 else f"_r{a.lora_r}"), res)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["conformance", "overfit", "overfit-diag", "ckpt-equiv", "memory", "cache-equiv"])
    ap.add_argument("--minted-root", default="data/r054_train48")
    ap.add_argument("--minted-repo", default="local/r054_train48")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--cached-features", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16, help="R-058: 4 or 64 (alpha = r)")
    ap.add_argument("--episode-frame", type=int, default=0, help="ckpt-equiv: the fixed sample")
    ap.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine"],
                    help="overfit: constant LR 1e-4 as registered (the 2026-09-29 run used cosine and missed at 12.1%%)")
    ap.add_argument("--limit", type=int, default=None, help="smoke: conformance on the first N instances")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    globals()["OUT"] = a.out
    {"conformance": conformance, "overfit": overfit, "overfit-diag": overfit_diag, "ckpt-equiv": ckpt_equiv, "memory": memory,
     "cache-equiv": cache_equiv}[a.gate](a)


if __name__ == "__main__":
    main()
