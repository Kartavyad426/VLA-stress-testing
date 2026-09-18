"""Embedding capture for VLA policies: per-model-forward internal features.

Capture is per MODEL FORWARD (~8 per episode for GR00T at n_action_steps=16),
not per env step (~118). That is why features live in sidecar .npz files with a
manifest rather than on `Step`, where they would be null 93% of the time.
"""
from .taps import CaptureSink, Role

__all__ = ["CaptureSink", "Role"]
