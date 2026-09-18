"""The flush hook: capture must be stamped with the rollout_id, not with order.

Order-based joining was rejected in the design because run_cell's cache-skip
path means capture order does not track trace order across a resume, and a
misjoin attaches one episode's embeddings to another episode's label.
"""
from __future__ import annotations

from vla_harness.runner import rollout
from vla_harness.schema import Action, Observation, PerturbationSpec


class _Env:
    env_id = "fake-env"
    task_id = "fake-task"
    instruction = "do the thing"
    control_mode = "relative"

    def identity(self):
        return {"name": self.env_id}

    def semantic_deps(self):
        return {}

    def scene_descriptor(self):
        return {}

    def reset(self, seed, spec):
        self._t = 0
        return Observation(instruction=self.instruction, state={"eef_pos": [0.0, 0.0, 0.0]}, t=0)

    def step(self, action):
        self._t += 1
        done = self._t >= 2
        return (Observation(instruction=self.instruction, state={"eef_pos": [0.0, 0.0, 0.0]}, t=self._t),
                True, done, "success")


class _Policy:
    policy_id = "fake-policy"
    action_dims = ["dx"]
    model_forwards = 0

    def __init__(self):
        self.flushed = []

    def identity(self):
        return {"name": self.policy_id}

    def reset(self):
        pass

    def __call__(self, obs):
        self.model_forwards += 1
        return Action([0.0], self.action_dims)

    def flush_capture(self, r):
        self.flushed.append(r.rollout_id)


def test_runner_flushes_capture_with_the_rollout_id():
    p = _Policy()
    r = rollout(_Env(), p, seed=0, spec=PerturbationSpec.of())

    assert p.flushed == [r.rollout_id]


def test_runner_does_not_require_a_policy_to_support_capture():
    """Capture stays off the critical path: a policy without flush_capture runs
    exactly as before."""
    class _Plain(_Policy):
        flush_capture = None

    del _Plain.flush_capture
    r = rollout(_Env(), _Plain(), seed=0, spec=PerturbationSpec.of())
    assert r.rollout_id
