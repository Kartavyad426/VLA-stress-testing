"""The adapter contract: capture is opt-in and never on the default path."""
from __future__ import annotations

from vla_harness.policies.lerobot_policy import LeRobotPolicy


def test_capture_is_off_by_default():
    p = LeRobotPolicy(checkpoint="x", device="cpu")
    assert p.capture_dir is None
    assert not hasattr(p, "flush_capture") or p._sink is None


def test_capture_does_not_change_policy_identity():
    """Capture is an OBSERVATION of a rollout, not a property of the policy.
    If it entered identity() it would change policy_id, miss the trace cache by
    construction, and make captured runs incomparable with every run already on
    disk -- which is the whole point of capturing them."""
    plain = LeRobotPolicy(checkpoint="x", device="cpu")
    capturing = LeRobotPolicy(checkpoint="x", device="cpu", capture_dir="/tmp/cap")

    assert plain.identity() == capturing.identity()
    assert plain.policy_id == capturing.policy_id
