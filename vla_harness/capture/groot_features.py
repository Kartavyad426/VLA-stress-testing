"""GR00T N1.7 tap spec -- the only file in this package that knows what a GR00T is.

Attaches from OUTSIDE. Nothing under `third_party/` is modified: that tree is a
vendored fork whose HEAD is recorded in every run's `code_state/`, so patching
it would retroactively invalidate the provenance of every previous run.
"""
from __future__ import annotations

import torch

from .taps import CaptureSink, Role


def _rng_state(device) -> tuple:
    """Generator state for every device the draw could come from."""
    cpu = torch.get_rng_state()
    cuda = (torch.cuda.get_rng_state(device)
            if device is not None and device.type == "cuda" else None)
    return (cpu, cuda, device)


def _restore_rng(state: tuple) -> None:
    cpu, cuda, device = state
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state(cuda, device)


def attach_groot_capture(policy, sink: CaptureSink, k_resample: int = 1):
    """Register taps on a loaded GR00T policy. Returns a detach callable.

    `k_resample` > 1 re-runs the denoise loop k-1 extra times to measure
    self-consistency (S4). This is the ONLY consistency signal available here:
    RTC is off on this path (`select_action` passes no leftovers, so
    `"action" in action_input` is False and the branch at groot_n1_7.py:665-681
    is dead), which means there is no chunk overlap and no true chunk-to-chunk
    disagreement to measure.

    COST: each extra draw re-runs the DiT denoise loop -- 4 Euler steps -- but
    NOT the 2B backbone, which is already computed. That asymmetry is the main
    reason to intercept at `action_head.get_action` rather than higher up.
    """
    head = policy.action_head
    handles = []
    # the decoder hook also fires during S4 resamples; only the model's own
    # denoise loop is the signal, so the resamples are gated out explicitly.
    live = {"on": True}
    path: list = []

    # S1: pre-`vlln`. It CANNOT come from reading `backbone_features` after
    # `process_backbone_output`, which overwrites that key in place
    # (groot_n1_7.py:563-564) -- a post-call read captures the post-attention
    # tensor twice, and `vlln` between the reads makes the duplicate look
    # plausible rather than raising.
    def _pre_vlln(_module, args):
        sink.stage(Role.VL_ENCODER, args[0].detach())

    # S1.5: free from the same module.
    def _post_vlln(_module, _args, output):
        sink.stage(Role.VL_NORMED, output.detach())

    handles.append(head.vlln.register_forward_pre_hook(_pre_vlln))
    handles.append(head.vlln.register_forward_hook(_post_vlln))

    # S4t: fires ONCE PER EULER STEP (4x per forward). `action_decoder` is
    # applied to `sa_embs = cat(state_features, action_features)`, so its output
    # carries a state-token prefix and the model consumes `pred[:, -H:]`
    # (groot_n1_7.py:709). Slicing anything else makes the V2 reconstruction sum
    # the wrong rows while still looking like a tolerance problem.
    def _post_decoder(_module, _args, output):
        if not live["on"]:
            return
        path.append(output[:, -head.action_horizon:].detach().clone())
        # recorded so V2b can assert the tap saw the state-token prefix and
        # sliced exactly as the model does, rather than trusting that it did.
        sink.note("decode_path_preslice_tokens", int(output.shape[1]))

    handles.append(head.action_decoder.register_forward_hook(_post_decoder))

    # `actions_0` is a bare `torch.randn` INSIDE the denoise loop
    # (groot_n1_7.py:657) -- not a module call, so no hook can observe it.
    # Rather than seed it (which would change which noise the model draws, and
    # therefore the rollout), replay it: save the generator, let the model draw
    # untouched, rewind, redraw the identical tensor, then restore.
    #
    # Deriving actions_0 as `action_pred - dt*sum(pred)` instead would be
    # arithmetically correct and would make the V2 gate pass for ANY hook
    # placement. It has to be observed for that gate to test anything.
    original_gawf = head.get_action_with_features

    def wrapped_gawf(*args, **kwargs):
        path.clear()
        vl = kwargs.get("backbone_features", args[0] if args else None)
        device = vl.device
        pre = _rng_state(device)
        out = original_gawf(*args, **kwargs)
        post = _rng_state(device)

        real_path = list(path)

        _restore_rng(pre)
        actions_0 = torch.randn(
            size=(vl.shape[0], head.action_horizon, head.action_dim),
            dtype=vl.dtype, device=device)

        if k_resample > 1:
            live["on"] = False          # resample decoder calls are not signal
            try:
                draws = [out["action_pred"]] + [
                    original_gawf(*args, **kwargs)["action_pred"]
                    for _ in range(k_resample - 1)]
                sink.stage(Role.RESAMPLE_SPREAD,
                           torch.stack(draws).std(dim=0).detach())
            finally:
                live["on"] = True

        _restore_rng(post)

        sink.stage(Role.ACTIONS_INIT, actions_0.detach())
        sink.stage(Role.DECODE_PATH,
                   torch.stack(real_path) if real_path else None)
        return out

    head.get_action_with_features = wrapped_gawf

    # S2/S0e/S3 come from `get_action`'s return. `get_action` is a plain method,
    # not `forward()`, so no nn.Module hook fires on it -- wrap the bound method.
    original_get_action = head.get_action

    def wrapped_get_action(*args, **kwargs):
        out = original_get_action(*args, **kwargs)
        sink.stage(Role.VL_ADAPTED, out["backbone_features"].detach())
        sink.stage(Role.STATE_ENCODED, out["state_features"].detach())
        sink.stage(Role.ACTION_PRED, out["action_pred"].detach())
        sink.commit_forward()
        return out

    head.get_action = wrapped_get_action

    def detach():
        for h in handles:
            h.remove()
        head.get_action = original_get_action
        head.get_action_with_features = original_gawf

    return detach
