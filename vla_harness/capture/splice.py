"""R-039 pathway splice for GR00T N1.7 -- attaches from OUTSIDE, like the taps.

Per forward pass the head is run FIVE times on the same inputs and the same
fixed noise, once per arm:

  P  target features as they are (the perturbed run's own)
  N  every feature from the SOURCE (the paired nominal render)
  T  text-token positions of `vl_embeds` from the source, rest target
  I  image-token positions from the source, rest target
  S  the state token from the source, rest target

All five action chunks are recorded; the `drive` arm's output is what the
policy executes. Text/image positions are read off `image_mask`, which
groot_n1_7.py:451 builds as `input_ids == image_token_id` -- never off a
config name (`attend_text_every_n_blocks` lies about its period).

The splice point is `get_action_with_features`, i.e. AFTER `vlln` and the
4-block `vl_self_attention`: the tokens as the DiT cross-attends to them
(RESULTS.md R-039, "where the splice point sits"). A pre-attention splice
is a different experiment and is not what this file does.

Noise: `groot_n1_7.py:657` draws `actions_0` from a bare `torch.randn`. Each
arm's run is preceded by `torch.manual_seed(noise_key(forward_idx))`, so the
five arms share one epsilon and two runs with the same inputs and index are
bitwise equal. The generator is restored afterwards, so nothing outside the
wrapper sees a different stream. This is a SEPARATE path from the capture's
save-draw-rewind, which R-036 depends on; the two compose.
"""
from __future__ import annotations

from typing import Callable

import torch

from .groot_features import _restore_rng, _rng_state, find_action_head

ARMS = ("P", "N", "T", "I", "S")
PAIR_ARMS = {"IT": ("I", "T")}          # R-042: both token blocks restored, state stays perturbed


def transfer_fraction(patched, target, source, live_dims: int) -> float:
    """1 - d(patched, source) / d(target, source), L2 over the live action dims.

    1.0 means the splice moved the action all the way back to the source's;
    0.0 means it did nothing. nan when target == source (nothing to transfer).
    """
    p = patched[..., :live_dims].float()
    t = target[..., :live_dims].float()
    s = source[..., :live_dims].float()
    d_ts = torch.linalg.vector_norm(t - s).item()
    if d_ts == 0.0:
        return float("nan")
    return 1.0 - torch.linalg.vector_norm(p - s).item() / d_ts


class SpliceHandle:
    def __init__(self, head, source: Callable[[int], dict | None],
                 drive: str | None, noise_key: Callable[[int], int],
                 extra: dict[str, Callable[[int], dict | None]] | None = None,
                 drive_until: int | None = None, allow_unaligned: bool = False):
        """`extra` maps an arm name to a callable giving a WHOLE feature set
        (backbone_features, state_features, image_mask) for forward i, or
        None to skip that arm this forward. Used for R-042's per-camera
        arms, where the hybrid observation is run through the full VL stack
        and the resulting features are used as-is."""
        allowed = ARMS + tuple(PAIR_ARMS) + tuple(extra or ())
        if drive is not None and drive not in allowed:
            raise ValueError(f"drive must be one of {allowed} or None, got {drive!r}")
        self._head = head
        self._source = source
        self._extra = dict(extra or {})
        self._drive = drive or "P"
        self._drive_until = drive_until          # forwards >= this execute P; arms still recorded
        self._allow_unaligned = allow_unaligned  # language row: token counts differ; record P, N, S only
        self._noise_key = noise_key
        self._fidx = 0
        self.records: list[dict] = []
        self._had_override = "get_action_with_features" in vars(head)
        self._original = head.get_action_with_features
        head.get_action_with_features = self._wrapped

    def begin_episode(self) -> None:
        self._fidx = 0
        self.records = []

    def detach(self) -> None:
        if self._had_override:            # e.g. the capture wrapper, below us
            self._head.get_action_with_features = self._original
        else:
            del self._head.get_action_with_features

    # ------------------------------------------------------------------
    def _wrapped(self, *args, **kwargs):
        if args:
            raise RuntimeError(
                "get_action_with_features was called positionally; the splice "
                "needs keyword arguments to substitute features by name")
        fidx = self._fidx
        self._fidx += 1
        src = self._source(fidx)
        if src is None:
            # unspliced, but still under common random numbers
            device = kwargs["backbone_features"].device
            outer = _rng_state(device)
            try:
                torch.manual_seed(int(self._noise_key(fidx)))
                return self._original(**kwargs)
            finally:
                _restore_rng(outer)

        vl_t = kwargs["backbone_features"]
        st_t = kwargs["state_features"]
        bo = kwargs["backbone_output"]
        vl_s = src["backbone_features"]
        st_s = src["state_features"]
        if vl_s.shape != vl_t.shape:
            if not self._allow_unaligned:
                raise RuntimeError(
                    f"token count differs: source {tuple(vl_s.shape)} vs target "
                    f"{tuple(vl_t.shape)} -- a paired render must tokenise identically")
            # positional arms are undefined; P, N and the state arm are not
            device = vl_t.device
            outer = _rng_state(device)
            outputs = {}
            # the N arm has the SOURCE's token count, so the DiT must see the
            # source's masks too (image_mask, backbone_attention_mask), not the target's
            bo_src = _with_masks(bo, src)
            try:
                for arm, (vl, st, bo_arm) in {"P": (vl_t, st_t, bo), "N": (vl_s, st_s, bo_src), "S": (vl_t, st_s, bo)}.items():
                    torch.manual_seed(int(self._noise_key(fidx)))
                    outputs[arm] = self._original(**{**kwargs, "backbone_features": vl, "state_features": st, "backbone_output": bo_arm})
            finally:
                _restore_rng(outer)
            self.records.append({"forward_idx": fidx, "aligned": False,
                                 "action": {arm: outputs[arm]["action_pred"].detach().clone() for arm in outputs}})
            if self._drive in outputs and not (self._drive_until is not None and fidx >= self._drive_until):
                return outputs[self._drive]
            return outputs["P"]
        if st_s.shape != st_t.shape:
            raise RuntimeError(
                f"state token differs: source {tuple(st_s.shape)} vs target {tuple(st_t.shape)}")
        image = _get(bo, "image_mask")
        if not torch.equal(image, src["image_mask"]):
            raise RuntimeError("image mask differs between source and target")
        attn = _get(bo, "backbone_attention_mask")
        if attn is not None:
            image = image & attn
            text = (~_get(bo, "image_mask")) & attn
        else:
            text = ~image

        def patched(mask):
            out = vl_t.clone()
            out[mask] = vl_s[mask]
            return out

        feats = {
            "P": (vl_t, st_t),
            "N": (vl_s, st_s),
            "T": (patched(text), st_t),
            "I": (patched(image), st_t),
            "S": (vl_t, st_s),
            "IT": (patched(image | text), st_t),
        }
        for name, fn in self._extra.items():
            ex = fn(fidx)
            if ex is None:
                continue
            if ex["backbone_features"].shape != vl_t.shape or ex["state_features"].shape != st_t.shape:
                raise RuntimeError(f"extra arm {name!r}: feature shapes differ from the target's")
            feats[name] = (ex["backbone_features"], ex["state_features"])

        device = vl_t.device
        outer = _rng_state(device)
        outputs = {}
        try:
            for arm, (vl, st) in feats.items():
                torch.manual_seed(int(self._noise_key(fidx)))
                outputs[arm] = self._original(
                    **{**kwargs, "backbone_features": vl, "state_features": st})
        finally:
            _restore_rng(outer)

        self.records.append({
            "forward_idx": fidx, "aligned": True,
            "action": {arm: outputs[arm]["action_pred"].detach().clone() for arm in outputs},
        })
        if self._drive_until is not None and fidx >= self._drive_until:
            return outputs["P"]
        return outputs[self._drive]


def _with_masks(bo, src):
    """A copy of backbone_output carrying the source's image and attention masks."""
    import copy
    out = copy.copy(bo)
    for k in ("image_mask", "backbone_attention_mask"):
        if k in src:
            try:
                out[k] = src[k]
            except Exception:
                setattr(out, k, src[k])
    return out


def _get(bo, key):
    if isinstance(bo, dict):
        return bo.get(key)
    if key in bo:
        return bo[key]
    return getattr(bo, key, None)


def attach_splice(policy, source: Callable[[int], dict | None], drive: str | None = None,
                  noise_key: Callable[[int], int] | None = None,
                  extra: dict[str, Callable[[int], dict | None]] | None = None,
                  drive_until: int | None = None, allow_unaligned: bool = False) -> SpliceHandle:
    """Wrap the head so every forward runs all five arms and drives `drive`.

    `source(forward_idx)` returns the nominal features for that forward:
    `backbone_features` (post-`vl_self_attention` tokens), `state_features`
    and `image_mask`; or None to pass the forward through unspliced.
    """
    head = find_action_head(policy)
    if head is None:
        raise RuntimeError("no GR00T action head found on the policy")
    return SpliceHandle(head, source, drive, noise_key or (lambda i: i), extra, drive_until, allow_unaligned)


class RecorderHandle:
    """Store the head's encoded features per forward -- the recorded-source
    path, used only where a paired render cannot exist (Robot Initial
    States, where the trigger IS the state). `source(i)` hands them to
    `attach_splice` for forward i and returns None past the end, so a longer
    target episode passes through unspliced rather than reusing a stale
    forward."""

    def __init__(self, head):
        self._head = head
        self.records: list[dict] = []
        self._had_override = "get_action_with_features" in vars(head)
        self._original = head.get_action_with_features
        head.get_action_with_features = self._wrapped

    def _wrapped(self, *args, **kwargs):
        if args:
            raise RuntimeError("recorder needs keyword arguments")
        bo = kwargs["backbone_output"]
        rec = {"backbone_features": kwargs["backbone_features"].detach().clone(),
               "state_features": kwargs["state_features"].detach().clone(),
               "image_mask": _get(bo, "image_mask").detach().clone()}
        attn = _get(bo, "backbone_attention_mask")
        if attn is not None:
            rec["backbone_attention_mask"] = attn.detach().clone()
        self.records.append(rec)
        return self._original(**kwargs)

    def source(self, i: int):
        return self.records[i] if i < len(self.records) else None

    def detach(self) -> None:
        if self._had_override:
            self._head.get_action_with_features = self._original
        else:
            del self._head.get_action_with_features


def attach_recorder(policy) -> RecorderHandle:
    head = find_action_head(policy)
    if head is None:
        raise RuntimeError("no GR00T action head found on the policy")
    return RecorderHandle(head)


# ----------------------------------------------------------------------------
# R-060: the same splice point with a TRANSFORM in place of a source.

class FTM(torch.nn.Module):
    """Feature Token Modulation (arXiv:2512.02902; SPATIAL_ADAPT_MATHS.html eq. 3.1):
    F_hat = (1 + gamma) * F + beta on the IMAGE-token positions only, one
    (gamma, beta) per channel shared across tokens, cameras and tasks. fp32
    parameters, applied in fp32 and cast back to the feature dtype. Zero-init
    is the identity (G0b checks that bitwise on the real forward)."""

    def __init__(self, dim: int = 2048):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.zeros(dim))
        self.beta = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        g = self.gamma.to(x.device)
        b = self.beta.to(x.device)
        out = x.clone()
        out[positions] = (x[positions].float() * (1 + g) + b).to(x.dtype)
        return out

    @classmethod
    def from_moments(cls, mu_s, sd_s, mu_t, sd_t) -> "FTM":
        """E1, eq. 3.2: 1 + gamma = sd_s / sd_t, beta = mu_s - (1 + gamma) mu_t,
        s = source (nominal view), t = target (perturbed view)."""
        m = cls(int(mu_s.numel()))
        with torch.no_grad():
            scale = sd_s.double() / sd_t.double()
            m.gamma.copy_((scale - 1).float())
            m.beta.copy_((mu_s.double() - scale * mu_t.double()).float())
        return m


def image_positions(bo) -> torch.Tensor:
    """image_mask & backbone_attention_mask: the image-token positions the DiT reads."""
    image = _get(bo, "image_mask")
    attn = _get(bo, "backbone_attention_mask")
    return image & attn if attn is not None else image


class TransformHandle:
    """`transform(features, positions) -> features` at one site, every forward,
    under the same common-random-numbers noise as the splice:

      S2  the adapter output, i.e. `backbone_features` as `get_action_with_features`
          receives it (post-`vlln` + `vl_self_attention`) -- the splice point.
      S1  the backbone output BEFORE `vlln`: `process_backbone_output` is wrapped,
          which is the only place both the pre-`vlln` tensor and `image_mask`
          are in hand (`_encode_features` -> `process_backbone_output`).

    `transform=None` is the seeded, untransformed base. `enabled` switches the
    transform off without detaching (paired-source features must be the base's).
    Each forward's `action_pred` is recorded."""

    def __init__(self, head, transform, site: str, noise_key: Callable[[int], int]):
        if site not in ("S1", "S2"):
            raise ValueError(f"site must be S1 or S2, got {site!r}")
        self._head = head
        self.transform = transform
        self.site = site
        self.enabled = True
        self._noise_key = noise_key
        self._fidx = 0
        self.records: list[dict] = []
        self._had_override = {n: n in vars(head) for n in ("get_action_with_features", "process_backbone_output")}
        self._original = head.get_action_with_features
        self._original_pbo = head.process_backbone_output
        head.get_action_with_features = self._wrapped
        if site == "S1":
            head.process_backbone_output = self._wrapped_pbo

    def begin_episode(self) -> None:
        self._fidx = 0
        self.records = []

    def detach(self) -> None:
        for name, orig in (("get_action_with_features", self._original),
                           ("process_backbone_output", self._original_pbo)):
            if self._had_override[name]:
                setattr(self._head, name, orig)
            elif name in vars(self._head):
                delattr(self._head, name)

    def _live(self) -> bool:
        return self.enabled and self.transform is not None

    def _wrapped_pbo(self, backbone_output):
        if self._live():
            backbone_output["backbone_features"] = self.transform(
                backbone_output["backbone_features"], image_positions(backbone_output))
        return self._original_pbo(backbone_output)

    def _wrapped(self, *args, **kwargs):
        if args:
            raise RuntimeError("get_action_with_features was called positionally; the transform "
                               "needs keyword arguments")
        fidx = self._fidx
        self._fidx += 1
        if self.site == "S2" and self._live():
            kwargs = {**kwargs, "backbone_features": self.transform(
                kwargs["backbone_features"], image_positions(kwargs["backbone_output"]))}
        device = kwargs["backbone_features"].device
        outer = _rng_state(device)
        try:
            torch.manual_seed(int(self._noise_key(fidx)))
            out = self._original(**kwargs)
        finally:
            _restore_rng(outer)
        self.records.append({"forward_idx": fidx, "action": out["action_pred"].detach().clone()})
        return out


def attach_transform(policy, transform, site: str = "S2",
                     noise_key: Callable[[int], int] | None = None) -> TransformHandle:
    head = find_action_head(policy)
    if head is None:
        raise RuntimeError("no GR00T action head found on the policy")
    return TransformHandle(head, transform, site, noise_key or (lambda i: i))
