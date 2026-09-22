"""Policy-agnostic capture: roles, the per-episode sink.

ROLES, NOT FIELD NAMES. The analyses this feeds contrast an INPUT-side signal
with a BEHAVIOUR signal; they never mention `vlln`. Nearly every VLA is
VLM -> adapter -> action head, so the roles are shared even where the modules
are not, and a second policy implements a tap spec rather than a second sink.

WHAT THIS DOES NOT MAKE COMPARABLE, and it must not be read as doing so: the
CODE is reusable across policies; the NUMBERS are not. Dimensions and
normalisation differ between policies, and `state_encoded` in particular is
produced by a per-embodiment `CategorySpecificLinear` -- its geometry is not
comparable across embodiments, so vectors from different embodiments MUST NOT
be pooled into one reference cloud. Everything in this package is LIBERO-only
today, which is the only reason that constraint is currently satisfied.
"""
from __future__ import annotations

import json
import os
import warnings
from enum import Enum

import numpy as np


def _store(name: str, a: np.ndarray) -> np.ndarray:
    """Cast to fp16, but never at the cost of the value.

    fp16 halves the capture (~65 MB vs ~130 MB over 723 episodes) and is ample
    for everything normalised. It is NOT ample for the PRE-LayerNorm tap: on the
    real checkpoint `vl_encoder_max` peaked at 15,296 against a 65,504 ceiling,
    4.3x of headroom, and those activations are large precisely because `vlln`
    has not run yet. A single brighter-scene excursion over 723 LIBERO-Plus
    episodes would turn that into `inf`, and an `inf` in a max-pooled feature
    puts the episode infinitely far from the reference cloud -- a fabricated OOD
    detection in the exact signal this experiment measures.

    So: fp32 when the value would not survive fp16, fp16 otherwise, and the
    per-key dtype is recorded in the .npz rather than assumed by the reader.
    """
    if not np.isfinite(a).all():
        raise ValueError(
            f"capture: {name} contains non-finite values before storage. A NaN "
            f"or inf reaching the reference cloud poisons every distance "
            f"computed against it -- refusing to write this episode.")
    if np.abs(a).max(initial=0.0) > FP16_MAX:
        return a.astype(np.float32)
    return a.astype(np.float16)

CAPTURE_VERSION = "1.1"
FP16_MAX = 65504.0
MANIFEST = "capture_manifest.jsonl"

# Roles whose tensors are token sequences and are therefore pooled. The others
# are already one vector per forward.
_POOLED = ("vl_encoder", "vl_normed", "vl_adapted")


class Role(str, Enum):
    """What a tapped tensor MEANS, independent of which model produced it."""

    VL_ENCODER = "vl_encoder"          # VLM output, before the action head touches it
    VL_NORMED = "vl_normed"            # after the adapter's norm, before its mixing
    VL_ADAPTED = "vl_adapted"          # what the action head actually receives
    STATE_ENCODED = "state_encoded"    # proprioception after per-embodiment encoding
    ACTION_PRED = "action_pred"        # the intended action chunk
    DECODE_PATH = "decode_path"        # intermediates along the denoise path
    ACTIONS_INIT = "actions_init"      # the noise the denoise path started from
    RESAMPLE_SPREAD = "resample_spread"  # self-consistency over repeated draws


class CaptureSink:
    """Buffers one episode's forwards in memory.

    Small by construction: ~8 forwards x a few thousand dims is kilobytes, so
    buffering costs nothing and lets the writer stamp a `rollout_id` that does
    not exist until the episode ends.
    """

    def __init__(self) -> None:
        self._episode: list[dict] = []
        self._pending: dict = {}
        self._mask = None

    def set_token_mask(self, mask) -> None:
        """Which VL tokens are IMAGE tokens, for the forward being assembled.

        Without it the VL roles are pooled across image and text together --
        two channels the model itself keeps apart (`AlternateVLDiT` drives them
        through separate attention masks). Worse, text token count varies with
        instruction length, so the mixture ratio drifts between episodes for
        reasons unrelated to what the camera saw, injecting variance into the
        feature that dilutes any real signal.
        """
        self._mask = mask

    def begin_episode(self) -> None:
        """Start an episode, refusing to inherit the previous one's forwards.

        The sink is flushed after `rollout()` returns. An episode that RAISES
        part-way never reaches that flush, so without this guard its buffered
        forwards would be carried into the next episode and written under the
        NEXT episode's rollout_id -- the same silent misjoin the explicit
        rollout_id flush exists to prevent, arriving by the exception path
        instead of the ordering path. Mid-run CUDA OOM is a live failure mode
        in this project, not a hypothetical one.

        Discard loudly rather than silently: a dropped episode is a gap in the
        capture, and a gap that nobody was told about is indistinguishable from
        an episode that never ran.
        """
        if self._episode or self._pending:
            warnings.warn(
                f"capture: discarding {len(self._episode)} unflushed forward(s) "
                f"from an episode that never completed -- it did not reach "
                f"flush_capture, most likely because it raised. Those forwards "
                f"are NOT written and must not be counted as captured.",
                UserWarning, stacklevel=2)
        self._episode = []
        self._pending = {}

    def stage(self, role: Role, value) -> None:
        """Record one tensor for the forward currently being assembled."""
        self._pending[role.value] = value

    def note(self, key: str, value) -> None:
        """Record a scalar fact about the current forward (shapes, counts).

        Kept separate from `stage` because these are provenance, not signals:
        they exist so a gate can assert what the tap SAW before any slicing,
        and they are not written as feature arrays.
        """
        self._pending[key] = value

    def commit_forward(self) -> None:
        """Close the current forward and start the next."""
        if self._pending:
            if self._mask is not None:
                self._pending["_image_mask"] = self._mask
            self._episode.append(self._pending)
            self._pending = {}

    def current(self) -> list[dict]:
        return self._episode

    def flush(self, out_dir, rollout_id: str, live_dims: int,
              meta: dict, env_steps: list | None = None) -> str:
        """Write this episode's .npz and append one manifest row.

        Called with a `rollout_id` that does not exist until `rollout()` has
        built the Rollout, which is why the sink buffers rather than streams.
        """
        import torch

        self.commit_forward()
        os.makedirs(out_dir, exist_ok=True)
        rows = self._episode
        arrays: dict[str, np.ndarray] = {}

        def stack(key):
            return torch.stack([r[key][0] for r in rows]).float().cpu().numpy()

        for role in _POOLED:
            if role in rows[0]:
                t = stack(role)                       # (F, tokens, dim)
                # BOTH poolings, decided before any result exists (§6): mean
                # discards spatial structure, max keeps peaks. Storing one would
                # let the choice be made after seeing which separated better.
                arrays[f"{role}_mean"] = _store(f"{role}_mean", t.mean(axis=1))
                arrays[f"{role}_max"] = _store(f"{role}_max", t.max(axis=1))

                # AND separately over image vs text tokens. The combined pool
                # above is kept so results stay comparable with R-036, but it
                # mixes two channels the model keeps apart, with a ratio that
                # drifts with instruction length.
                if "_image_mask" in rows[0]:
                    im = [np.asarray(r["_image_mask"][0].cpu()) for r in rows]
                    # The mask indexes the VL sequence; if its length ever
                    # disagrees with the token axis (padding handled elsewhere,
                    # a future caller), refuse rather than index silently wrong
                    # -- a misaligned mask would pool the wrong tokens into
                    # "image" and the error would look like a weak signal.
                    bad = [(i, len(m), t.shape[1]) for i, m in enumerate(im)
                           if len(m) != t.shape[1]]
                    if bad:
                        raise ValueError(
                            f"capture: image_mask length disagrees with the "
                            f"{role} token axis at forwards {bad} "
                            f"(mask, tokens) -- refusing to pool on a "
                            f"misaligned mask.")
                    for tag, sel in (("img", lambda m: m), ("txt", lambda m: ~m)):
                        mean = np.stack([t[i][sel(m)].mean(axis=0)
                                         for i, m in enumerate(im)])
                        mx = np.stack([t[i][sel(m)].max(axis=0)
                                       for i, m in enumerate(im)])
                        arrays[f"{role}_{tag}_mean"] = _store(f"{role}_{tag}_mean", mean)
                        arrays[f"{role}_{tag}_max"] = _store(f"{role}_{tag}_max", mx)
                    arrays["n_image_tokens"] = np.array(
                        [int(m.sum()) for m in im], dtype=np.int32)
                    arrays["n_text_tokens"] = np.array(
                        [int((~m).sum()) for m in im], dtype=np.int32)

        if Role.STATE_ENCODED.value in rows[0]:
            arrays["state_encoded"] = _store(
                "state_encoded",
                stack(Role.STATE_ENCODED.value).reshape(len(rows), -1))

        # Captured UNSLICED at the tap and sliced here, with `sliced_from`
        # recorded, so the slice cannot quietly become the thing measured.
        sliced_from = None
        for role in (Role.ACTION_PRED.value, Role.ACTIONS_INIT.value,
                     Role.DECODE_PATH.value, Role.RESAMPLE_SPREAD.value):
            if role in rows[0] and rows[0][role] is not None:
                t = torch.stack([r[role] for r in rows]).float().cpu().numpy()
                t = t.squeeze(1) if role != Role.DECODE_PATH.value else t[:, :, 0]
                sliced_from = t.shape[-1]
                arrays[role] = _store(role, t[..., :live_dims])
        if sliced_from is not None:
            arrays["sliced_from"] = np.int32(sliced_from)
            arrays["live_dims"] = np.int32(live_dims)

        # the reader must not have to assume fp16; _store may promote.
        arrays["_dtypes"] = np.array(
            [f"{k}:{v.dtype}" for k, v in sorted(arrays.items())])

        path = os.path.join(out_dir, f"{rollout_id}.npz")
        np.savez_compressed(path, **arrays)

        row = {"rollout_id": rollout_id, "path": os.path.basename(path),
               "n_forwards": len(rows), "capture_version": CAPTURE_VERSION,
               "env_step": list(env_steps) if env_steps is not None else None,
               **meta}
        self._append_manifest(os.path.join(out_dir, MANIFEST), row)
        self._episode = []
        self._pending = {}
        return path

    @staticmethod
    def _append_manifest(path: str, row: dict) -> None:
        """Unique on rollout_id.

        A resume re-enters run_cell's cache-skip path and can re-capture an
        episode already written. A second row for the same rollout_id would
        double-count that episode in every downstream analysis, silently -- so
        the row is REPLACED, not appended.
        """
        rows = {}
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        rows[r["rollout_id"]] = r
        rows[row["rollout_id"]] = row
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            for r in rows.values():
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        os.replace(tmp, path)
