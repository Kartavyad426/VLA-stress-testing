"""A stand-in for GR00T's ActionHead, structurally faithful where it matters.

This is the thing that makes the capture gates runnable in milliseconds on a
CPU. It is NOT a mock: it reproduces the three structural features of
`groot_n1_7.py` that the capture code can get wrong, and it would be useless if
it reproduced anything less.

  1. `process_backbone_output` OVERWRITES `backbone_features` IN PLACE
     (groot_n1_7.py:563-564), so a hook reading the key after the call captures
     the post-attention tensor twice. With `vlln` between the two reads the
     duplicate still looks plausible, which is why V1 asserts the pair.

  2. `action_decoder` emits a STATE-TOKEN PREFIX. The real `sa_embs` is
     `cat(state_features, action_features)` (:696) with
     `state_history_length = 1`, so `pred` is `(B, 1 + action_horizon, dim)` and
     the model consumes `pred[:, -action_horizon:]` (:709). A tap that stores
     `pred` unsliced makes the V2 reconstruction sum one row too many -- and it
     would read as a tolerance failure, not as a wrong-rows failure.

  3. `actions_0` is a bare `torch.randn` inside the denoise loop (:657), not a
     module call, so no hook can observe it.

Dimensions are deliberately small and deliberately DISTINCT from each other, so
a transposed or mis-keyed tensor fails on shape rather than silently working.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

VL_DIM = 16
VL_TOKENS = 7
STATE_DIM = 8
STATE_TOKENS = 1        # mirrors state_history_length = 1
ACTION_DIM = 12         # stands in for max_action_dim = 132
LIVE_DIMS = 5           # stands in for LIBERO's 7 live dims
ACTION_HORIZON = 6      # stands in for action_horizon = 40
N_TIMESTEPS = 4         # mirrors num_inference_timesteps = 4


class FakeActionHead(nn.Module):
    """Same call graph as the real ActionHead, three orders of magnitude smaller."""

    def __init__(self) -> None:
        super().__init__()
        self.vlln = nn.LayerNorm(VL_DIM)
        self.vl_self_attention = nn.Linear(VL_DIM, VL_DIM)
        self.state_encoder = nn.Linear(STATE_DIM, VL_DIM)
        self.action_encoder = nn.Linear(ACTION_DIM, VL_DIM)
        self.action_decoder = nn.Linear(VL_DIM, ACTION_DIM)
        self.num_inference_timesteps = N_TIMESTEPS
        self.action_horizon = ACTION_HORIZON
        self.action_dim = ACTION_DIM
        # make vl_self_attention a genuine mixing op, so S1.5 != S2
        with torch.no_grad():
            self.vl_self_attention.weight.mul_(1.7)
            self.vl_self_attention.bias.add_(0.3)

    # --- mirrors groot_n1_7.py:562-565, including the in-place overwrite ----
    def process_backbone_output(self, backbone_output: dict) -> dict:
        backbone_features = self.vlln(backbone_output["backbone_features"])
        backbone_output["backbone_features"] = self.vl_self_attention(backbone_features)
        return backbone_output

    def _encode_features(self, backbone_output: dict, action_input: dict) -> dict:
        backbone_output = self.process_backbone_output(backbone_output)
        state_features = self.state_encoder(action_input["state"])
        return {"backbone_features": backbone_output["backbone_features"],
                "state_features": state_features}

    # --- mirrors groot_n1_7.py:645-716 --------------------------------------
    def get_action_with_features(self, backbone_features, state_features,
                                 backbone_output, action_input, options=None):
        vl_embeds = backbone_features
        batch_size = vl_embeds.shape[0]
        # the unseeded draw no hook can see (groot_n1_7.py:657)
        actions = torch.randn(size=(batch_size, self.action_horizon, self.action_dim),
                              dtype=vl_embeds.dtype, device=vl_embeds.device)
        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)     # RTC off => identically 1

        for _ in range(self.num_inference_timesteps):
            action_features = self.action_encoder(actions)
            sa_embs = torch.cat((state_features, action_features), dim=1)
            pred = self.action_decoder(sa_embs)     # (B, STATE_TOKENS+H, ACTION_DIM)
            actions = actions + dt * pred[:, -self.action_horizon:] * vel_strength

        return {"action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features}

    @torch.no_grad()
    def get_action(self, backbone_output: dict, action_input: dict, options=None):
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features["backbone_features"],
            state_features=features["state_features"],
            backbone_output=backbone_output,
            action_input=action_input,
            options=options)


class FakePolicy(nn.Module):
    """Stands in for the loaded LeRobot policy: `.action_head` is what we tap."""

    def __init__(self) -> None:
        super().__init__()
        self.action_head = FakeActionHead()

    def one_forward(self, seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
        backbone_output = {"backbone_features": torch.randn(1, VL_TOKENS, VL_DIM)}
        action_input = {"state": torch.randn(1, STATE_TOKENS, STATE_DIM)}
        return self.action_head.get_action(backbone_output, action_input)


@pytest.fixture
def policy() -> FakePolicy:
    torch.manual_seed(0)
    return FakePolicy().eval()
