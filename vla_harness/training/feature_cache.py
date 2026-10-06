"""Frozen-VLM feature caching for arm (a) (docs/FINE_TUNING_101.html §12). Fallback only:
used if R-055's memory gate fails with the VLM resident.

The cut is GR00TN17.forward's seam (groot_n1_7.py:869-872):

    backbone_outputs = self.backbone(backbone_inputs)      # frozen: cached
    return self.action_head(backbone_outputs, action_inputs)  # vlln, vl_self_attention (LoRA), DiT (LoRA)

What is cached is the RAW backbone output -- `backbone_features` plus its
`backbone_attention_mask` and `image_mask` -- the same tensors the harness's
capture reads, and the same ones the splice feeds the head (R-044 reproduced
R-042 to 0.01 from recorded features). vlln and vl_self_attention sit inside
the head and carry LoRA, so they are never cached. Flow matching's tau and
epsilon are drawn inside the head per step, so caching removes nothing random
from the loss.

Exactness conditions, all logged by `precompute`:
  * image augmentation is frozen by caching. lerobot_train builds the GR00T
    preprocessor with training=False (dataset_meta is passed only for reward
    models, lerobot_train.py:513), so the port's random crop
    (processor_groot.py:2100) and state dropout do not run on the uncached
    path either; ColorJitter is off unless the dataset config enables it.
  * the tokenizer pads on the LEFT (processor_groot.py:1381) and
    vl_self_attention takes no mask, so a padded token's hidden state enters
    every other token's features. Each frame is cached at its own length; at
    micro-batch 1 the cached path is exact, at batch > 1 with unequal prompt
    lengths it pads with zeros where the uncached path pads with the LLM's
    pad-token states (an approximation, reported by the gate).

CUT is a parameter so arm (b) can later cache at the LLM layer-11 -> 12
boundary; only "backbone_output" is implemented.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import torch

CUTS = ("backbone_output",)
VLM_STEP = "groot_n1_7_vlm_encode_v1"
SHARD = 2000


def _step_name(step) -> str:
    return getattr(step, "_registry_name", None) or type(step).__name__


def strip_vlm_encode(preprocessor):
    """The preprocessor without the Qwen3-VL encode step: state/action packing only.
    Cached training never needs input_ids or pixel_values."""
    from copy import copy
    from lerobot.policies.groot.processor_groot import GrootN17VLMEncodeStep  # noqa: F401  (import check)
    p = copy(preprocessor)
    p.steps = [s for s in preprocessor.steps if type(s).__name__ != "GrootN17VLMEncodeStep"]
    if len(p.steps) == len(preprocessor.steps):
        raise RuntimeError("no GrootN17VLMEncodeStep in the preprocessor")
    return p


def _file_sha(path: str) -> str:
    return hashlib.sha1(open(path, "rb").read()).hexdigest()[:12]


def fingerprint(checkpoint: str, parts: list, cut: str) -> dict:
    return {"checkpoint": checkpoint, "cut": cut, "code": _file_sha(__file__),
            "groot_n1_7": _file_sha(__import__("lerobot.policies.groot.groot_n1_7", fromlist=["x"]).__file__),
            "processor_groot": _file_sha(__import__("lerobot.policies.groot.processor_groot", fromlist=["x"]).__file__),
            "parts": [{"repo_id": getattr(p, "repo_id", None), "root": str(getattr(p, "root", "")),
                       "frames": len(p)} for p in parts]}


def augmentation_report(preprocessor, dataset_cfg=None) -> dict:
    out = {}
    for s in preprocessor.steps:
        if hasattr(s, "training"):
            out[type(s).__name__ + ".training"] = bool(s.training)
        for k in ("image_crop_size", "crop_fraction", "use_albumentations", "state_dropout_prob"):
            if hasattr(s, k):
                out[f"{type(s).__name__}.{k}"] = getattr(s, k)
    if dataset_cfg is not None:
        out["dataset.image_transforms.enable"] = bool(dataset_cfg.image_transforms.enable)
    out["random_crop_active"] = any(v for k, v in out.items() if k.endswith(".training"))
    return out


def _single(sample: dict) -> dict:
    """One dataset item -> a batch of 1 (lerobot's default collate for one sample)."""
    from torch.utils.data import default_collate
    return default_collate([sample])


@torch.no_grad()
def backbone_output_for(policy, preprocessor, sample: dict) -> dict:
    """Run the frozen backbone on one sample; the raw output, unpadded (batch of 1)."""
    model = policy._groot_model
    batch = preprocessor(_single(sample))
    inputs = policy._filter_groot_inputs(batch, include_action=True)
    backbone_inputs, _ = model.prepare_input(inputs)
    bo = model.backbone(backbone_inputs)
    return {"backbone_features": bo["backbone_features"][0].detach().to(torch.bfloat16).cpu(),
            "backbone_attention_mask": bo["backbone_attention_mask"][0].detach().cpu(),
            "image_mask": bo["image_mask"][0].detach().cpu()}


def precompute(policy, preprocessor, parts: list, names: list[str], out_dir: str, checkpoint: str,
               cut: str = "backbone_output", dataset_cfg=None, limit: int | None = None) -> dict:
    """Cache the frozen backbone's output for every frame of every part. Resumable by shard."""
    if cut not in CUTS:
        raise NotImplementedError(f"cut {cut!r}: only {CUTS} is implemented")
    os.makedirs(out_dir, exist_ok=True)
    fp = fingerprint(checkpoint, parts, cut)
    meta_p = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_p):
        old = json.load(open(meta_p))
        if old["fingerprint"] != fp:
            raise RuntimeError(f"{out_dir} holds a cache for a different fingerprint; use a new directory")
    policy.eval()
    t0 = time.time()
    sizes = []
    for part, (ds, name) in enumerate(zip(parts, names)):
        n = len(ds) if limit is None else min(limit, len(ds))
        for s0 in range(0, n, SHARD):
            path = os.path.join(out_dir, f"p{part}_{s0 // SHARD:05d}.pt")
            if os.path.exists(path):
                continue
            shard = {}
            for i in range(s0, min(s0 + SHARD, n)):
                e = backbone_output_for(policy, preprocessor, ds[i])
                shard[i] = e
                if len(sizes) < 200:
                    sizes.append(sum(v.numel() * v.element_size() for v in e.values()))
            torch.save(shard, path + ".part")
            os.replace(path + ".part", path)
            print(f"[feature_cache] {name}: {min(s0 + SHARD, n)}/{n} frames ({time.time()-t0:.0f}s)", flush=True)
    meta = {"fingerprint": fp, "names": names, "shard": SHARD,
            "augmentation": augmentation_report(preprocessor, dataset_cfg),
            "bytes_per_frame_measured": (sum(sizes) / len(sizes)) if sizes else None,
            "padding_side": "left", "dtype": "bfloat16"}
    json.dump(meta, open(meta_p, "w"), indent=1)
    return meta


class FeatureCache:
    def __init__(self, out_dir: str):
        self.dir = out_dir
        self.meta = json.load(open(os.path.join(out_dir, "meta.json")))
        self.shard = self.meta["shard"]
        self._loaded: dict[tuple[int, int], dict] = {}

    def get(self, part: int, idx: int) -> dict:
        key = (part, idx // self.shard)
        if key not in self._loaded:
            if len(self._loaded) > 8:
                self._loaded.pop(next(iter(self._loaded)))
            self._loaded[key] = torch.load(os.path.join(self.dir, f"p{part}_{key[1]:05d}.pt"))
        return self._loaded[key][idx]


def collate_backbone(entries: list[dict], device, dtype=torch.bfloat16) -> dict:
    """Left-pad cached per-frame outputs into one backbone_output batch."""
    from transformers.feature_extraction_utils import BatchFeature
    T = max(e["backbone_features"].shape[0] for e in entries)
    D = entries[0]["backbone_features"].shape[1]
    B = len(entries)
    feats = torch.zeros(B, T, D, dtype=dtype)
    attn = torch.zeros(B, T, dtype=torch.bool)
    img = torch.zeros(B, T, dtype=torch.bool)
    for b, e in enumerate(entries):
        t = e["backbone_features"].shape[0]
        feats[b, T - t:] = e["backbone_features"].to(dtype)
        attn[b, T - t:] = e["backbone_attention_mask"]
        img[b, T - t:] = e["image_mask"]
    return BatchFeature(data={"backbone_features": feats.to(device), "backbone_attention_mask": attn.to(device),
                              "image_mask": img.to(device)})
