"""R-059 parts B/C (E1b): B seeded draws per forward from GR00T N1.7's flow-matching head, one batched call.

Runtime wrapper, not a source patch. The head's own `get_action_with_features` (groot_n1_7.py:640-700 in the copy
the libero-plus venv imports, /home/imerit/Documents/Code/VLA/third_party/lerobot/...) is called unchanged with
  - backbone_features, state_features, embodiment_id and backbone_output's masks repeated B x along the batch dim
    (the VLM/backbone has already run once; only the DiT head is batched), and
  - its initial noise `torch.randn(size=(B, action_horizon, action_dim))` (line 657) supplied from seeded per-draw
    CPU generators: draw b uses seed = sha256("r059|<noise_seed>|<forward>|<b>")[:8] (Python's hash() is salted per
    process). torch.randn is replaced only for the duration of that one call, only for that exact size, and any
    other torch.randn call inside the window raises. Per-draw generators make draw 0's noise independent of B by
    construction.
Draw 0's action_pred is returned (it is executed); draws 1..B-1 are shadow draws, recorded only.
Without --seed-noise and with B = 1 the wrapper is never attached (r047_run.py), so behaviour is the current runner's.

`--null-extra` draws a second, independent B-set at every forward (seed tag "null"), recorded, never executed: the
null for MMD² between two B-sets at the same observation (part B).
"""
from __future__ import annotations

import copy
import hashlib
import time
from typing import Callable

import torch

LIVE_DIMS = 7
BO_KEYS = {"backbone_features", "backbone_attention_mask", "image_mask"}


def draw_seed(noise_seed: int, forward: int, b: int, tag: str = "r059") -> int:
    h = hashlib.sha256(f"{tag}|{int(noise_seed)}|{int(forward)}|{int(b)}".encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def seeded_noise(shape_1, dtype, device, noise_seed, forward, draws, tag="r059") -> torch.Tensor:
    """(draws, *shape_1[1:]) standard normal; draw b from its own CPU generator, then cast/moved."""
    out = []
    for b in range(draws):
        g = torch.Generator(device="cpu").manual_seed(draw_seed(noise_seed, forward, b, tag))
        out.append(torch.randn(size=(1, *shape_1[1:]), generator=g, dtype=torch.float32))
    return torch.cat(out).to(dtype=dtype, device=device)


def _repeat(x, B):
    if isinstance(x, torch.Tensor) and x.dim() >= 1 and x.shape[0] == 1:
        return x.repeat(B, *([1] * (x.dim() - 1)))
    return x


class _NoisePatch:
    """Inside the window, torch.randn(size=expected) returns `noise`; any other call raises."""

    def __init__(self, noise: torch.Tensor):
        self.noise, self.used = noise, 0

    def __enter__(self):
        self._orig = torch.randn
        noise = self.noise

        def fake(*args, size=None, dtype=None, device=None, **kw):
            shp = tuple(size if size is not None else (args[0] if len(args) == 1 and not isinstance(args[0], int) else args))
            if shp != tuple(noise.shape) or kw.get("generator") is not None:
                raise RuntimeError(f"unexpected torch.randn{shp} inside the multi-draw window (expected {tuple(noise.shape)})")
            if self.used:
                raise RuntimeError("torch.randn called twice inside the multi-draw window")
            self.used += 1
            return noise.to(dtype=dtype or noise.dtype, device=device or noise.device).clone()
        torch.randn = fake
        return self

    def __exit__(self, *exc):
        torch.randn = self._orig
        return False


class MultiDrawHandle:
    def __init__(self, head, draws: int, noise_seed: int, seed_noise: bool, null_extra: bool = False,
                 forward_key: Callable[[int], int] | None = None):
        if draws < 1:
            raise ValueError("draws must be >= 1")
        if draws > 1 and not seed_noise:
            raise ValueError("B > 1 needs --seed-noise (shadow draws must be reproducible)")
        self._head, self.draws, self.noise_seed = head, draws, int(noise_seed)
        self.seed_noise, self.null_extra = seed_noise, null_extra
        self._fidx = 0
        self.records: list[dict] = []
        self.input_shapes: dict | None = None
        self._had_override = "get_action_with_features" in vars(head)
        self._original = head.get_action_with_features
        head.get_action_with_features = self._wrapped

    def begin_episode(self):
        self._fidx, self.records = 0, []

    def detach(self):
        if self._had_override:
            self._head.get_action_with_features = self._original
        else:
            del self._head.get_action_with_features

    def _call(self, kwargs, B, tag):
        vl = kwargs["backbone_features"]
        H, D = self._head.config.action_horizon, self._head.action_dim
        noise = seeded_noise((1, H, D), vl.dtype, vl.device, self.noise_seed, self._fidx, B, tag)
        kw = dict(kwargs)
        if B > 1:
            # Explicit whitelist, not "anything with a leading 1" (val note #5, 10-09). The real backbone output has
            # exactly these keys, all batch-first (groot_n1_7.py:453-460, :562-565); anything else stops the run.
            st, emb, bo0 = kwargs["state_features"], kwargs["embodiment_id"], kwargs["backbone_output"]
            if set(bo0.keys()) != BO_KEYS:
                raise RuntimeError(f"backbone_output keys {sorted(bo0.keys())} != expected {sorted(BO_KEYS)}")
            seq = vl.shape[1]
            want = {"backbone_features": (vl, 3), "state_features": (st, 3), "embodiment_id": (emb, 1),
                    "bo.backbone_features": (bo0["backbone_features"], 3),
                    "bo.backbone_attention_mask": (bo0["backbone_attention_mask"], 2), "bo.image_mask": (bo0["image_mask"], 2)}
            for name, (x, rank) in want.items():
                if not isinstance(x, torch.Tensor) or x.dim() != rank or x.shape[0] != 1:
                    raise RuntimeError(f"{name}: expected a batch-1 rank-{rank} tensor, got {getattr(x, 'shape', type(x))}")
            for name in ("bo.backbone_attention_mask", "bo.image_mask", "bo.backbone_features"):
                if want[name][0].shape[1] != seq:
                    raise RuntimeError(f"{name}: sequence length {want[name][0].shape[1]} != backbone {seq}")
            kw["backbone_features"] = _repeat(vl, B)
            kw["state_features"] = _repeat(st, B)
            kw["embodiment_id"] = _repeat(emb, B)
            bo = copy.copy(bo0)
            for k in BO_KEYS:
                bo[k] = _repeat(bo0[k], B)
            kw["backbone_output"] = bo
            if self._fidx == 0 and not self.records:
                self.input_shapes = {n: list(x.shape) for n, (x, _) in want.items()}
        if "action" in kwargs.get("action_input", {}):
            raise RuntimeError("RTC inputs are not supported by the multi-draw wrapper")
        with _NoisePatch(noise) as p:
            out = self._original(**kw)
        if p.used != 1:
            raise RuntimeError(f"the head drew its noise {p.used} times, expected once")
        return out, noise

    def _wrapped(self, *args, **kwargs):
        if args:
            raise RuntimeError("get_action_with_features was called positionally")
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        sync(); t0 = time.perf_counter()
        out, noise = self._call(kwargs, self.draws, "r059")
        sync(); head_s = time.perf_counter() - t0
        rec = {"forward_idx": self._fidx, "draw0_seed": draw_seed(self.noise_seed, self._fidx, 0),
               "act_B": out["action_pred"][:, :, :LIVE_DIMS].detach().float().cpu(), "head_s": head_s}
        if self.null_extra:
            o2, _ = self._call(kwargs, self.draws, "null")
            rec["act_B_null"] = o2["action_pred"][:, :, :LIVE_DIMS].detach().float().cpu()
        self.records.append(rec)
        self._fidx += 1
        res = copy.copy(out)
        res["action_pred"] = out["action_pred"][:1]
        res["backbone_features"] = kwargs["backbone_features"]
        res["state_features"] = kwargs["state_features"]
        return res


def attach_multidraw(policy, draws, noise_seed, seed_noise, null_extra=False) -> MultiDrawHandle:
    from vla_harness.capture.splice import find_action_head
    head = find_action_head(policy)
    if head is None:
        raise RuntimeError("no GR00T action head found on the policy")
    return MultiDrawHandle(head, draws, noise_seed, seed_noise, null_extra)
