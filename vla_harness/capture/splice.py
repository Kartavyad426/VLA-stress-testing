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
                 drive: str | None, noise_key: Callable[[int], int]):
        if drive is not None and drive not in ARMS:
            raise ValueError(f"drive must be one of {ARMS} or None, got {drive!r}")
        self._head = head
        self._source = source
        self._drive = drive or "P"
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
            return self._original(**kwargs)

        vl_t = kwargs["backbone_features"]
        st_t = kwargs["state_features"]
        bo = kwargs["backbone_output"]
        vl_s = src["backbone_features"]
        st_s = src["state_features"]
        if vl_s.shape != vl_t.shape:
            raise RuntimeError(
                f"token count differs: source {tuple(vl_s.shape)} vs target "
                f"{tuple(vl_t.shape)} -- a paired render must tokenise identically")
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
        }

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
            "forward_idx": fidx,
            "action": {arm: outputs[arm]["action_pred"].detach().clone() for arm in ARMS},
        })
        return outputs[self._drive]


def _get(bo, key):
    if isinstance(bo, dict):
        return bo.get(key)
    if key in bo:
        return bo[key]
    return getattr(bo, key, None)


def attach_splice(policy, source: Callable[[int], dict | None], drive: str | None = None,
                  noise_key: Callable[[int], int] | None = None) -> SpliceHandle:
    """Wrap the head so every forward runs all five arms and drives `drive`.

    `source(forward_idx)` returns the nominal features for that forward:
    `backbone_features` (post-`vl_self_attention` tokens), `state_features`
    and `image_mask`; or None to pass the forward through unspliced.
    """
    head = find_action_head(policy)
    if head is None:
        raise RuntimeError("no GR00T action head found on the policy")
    return SpliceHandle(head, source, drive, noise_key or (lambda i: i))
