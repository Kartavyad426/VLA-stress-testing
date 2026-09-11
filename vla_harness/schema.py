"""L0 -- the canonical rollout contract.

Everything above this layer reads and writes these types and nothing else.
See PLAN.md section 2 for the scalability guarantees these encode (G1-G10).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import dataclass, field, asdict
from typing import Any, Protocol, runtime_checkable

SCHEMA_VERSION = "2.0"


# --- G1: actions are variable-dimension -------------------------------------
# The toy uses 3 DoF, LIBERO uses 7, a bimanual setup uses 14. Never a tuple.

@dataclass
class Observation:
    """One timestep of what the policy can see.

    G2: images live on disk; `image_refs` holds paths, never pixels. A 300-step
    LIBERO episode with two 224x224 camera streams is ~90 MB of raw frames --
    inlining that makes traces unloadable.
    """
    instruction: str
    state: dict[str, Any]                          # proprioception + env-visible state
    image_refs: dict[str, str] = field(default_factory=dict)
    t: int = 0

    def get(self, key, default=None):
        return self.state.get(key, default)

    def policy_view(self) -> "Observation":
        """The observation a POLICY is allowed to see.

        Keys prefixed `_gt_` are privileged simulator truth (object poses,
        exact distances) available only because we are in simulation. A real
        robot does not have them, so a policy that reads one produces an
        inflated success rate that collapses on hardware. `runner.rollout`
        passes this view to the policy; detectors get the full state.
        """
        return Observation(
            instruction=self.instruction, t=self.t,
            image_refs=dict(self.image_refs),
            state={k: v for k, v in self.state.items()
                   if not k.startswith("_gt_")},
        )


@dataclass
class Action:
    values: list[float]
    dims: list[str]                                # e.g. ["dx","dy","gripper"]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.dims, self.values))

    def __repr__(self):
        return "(" + ", ".join(f"{d}={v:+.3f}" for d, v in
                               zip(self.dims, self.values)) + ")"


@dataclass
class Step:
    """One (observation, action) pair.

    `action` is `None` on the TERMINAL step only. The terminal step carries the
    observation in which success was decided -- the state after the last action
    landed -- for which no action was ever requested. Consumers iterating
    `steps[]` for actions must handle `None`; consumers reading state via
    `Rollout.series()` pick it up with no change.
    """
    t: int
    obs_state: dict[str, Any]
    action: list[float] | None
    image_refs: dict[str, str] = field(default_factory=dict)


# --- G4: perturbation names mirror the LIBERO-plus taxonomy from day one -----

LIBERO_PLUS_FACTORS = {
    "camera_pose":   ["camera_yaw_deg", "camera_pitch_deg", "camera_dist_m"],
    "initial_state": ["ee_offset_x_m", "ee_offset_y_m"],
    "object_layout": ["object_shift_m", "distractor_count"],
    "lighting":      ["light_intensity"],
    "texture":       ["texture_id"],
    "sensor_noise":  ["pixel_noise_std"],
    "language":      ["instruction_variant"],
}
_KNOB_TO_FACTOR = {k: f for f, ks in LIBERO_PLUS_FACTORS.items() for k in ks}


@dataclass(frozen=True)
class PerturbationSpec:
    """The knob settings for one episode. Frozen: knobs never change mid-episode."""
    knobs: tuple[tuple[str, float], ...] = ()

    @classmethod
    def of(cls, **knobs) -> "PerturbationSpec":
        for k in knobs:
            if k not in _KNOB_TO_FACTOR:
                raise ValueError(
                    f"unknown knob {k!r}; add it to LIBERO_PLUS_FACTORS so it "
                    f"maps onto the benchmark taxonomy (G4)")
        return cls(tuple(sorted(knobs.items())))

    def as_dict(self) -> dict[str, float]:
        return dict(self.knobs)

    def factors(self) -> set[str]:
        return {_KNOB_TO_FACTOR[k] for k, _ in self.knobs}

    def is_nominal(self) -> bool:
        return all(v == 0 for _, v in self.knobs)

    def revert(self, *knob_names) -> "PerturbationSpec":
        """Counterfactual probe: set these knobs back to nominal, keep the rest."""
        return PerturbationSpec(tuple((k, 0.0 if k in knob_names else v)
                                      for k, v in self.knobs))

    def label(self) -> str:
        active = [f"{k}={v:g}" for k, v in self.knobs if v != 0]
        return ", ".join(active) if active else "nominal"


@dataclass
class Rollout:
    """One episode, start to finish. The atomic unit of evidence."""
    rollout_id: str
    task_id: str
    instruction: str
    policy_id: str
    env_id: str
    seed: int
    perturbation: dict[str, float]
    action_dims: list[str]
    steps: list[Step]
    success: bool
    termination: str                                # grasped | timeout | ...
    wall_time_s: float = 0.0
    forward_passes: int = 0
    # structured identity of the env + policy that produced this rollout.
    # Stored, not merely hashed, so a cache hit can be VERIFIED rather than
    # trusted, and so a mismatch can say which field moved.
    fingerprint: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    meta: dict[str, Any] = field(default_factory=dict)

    # populated by L3; never by the env or policy
    diagnosis: dict[str, Any] | None = None

    def __len__(self):
        return len(self.steps)

    def series(self, key: str) -> list:
        """Pull one state key across the whole trace. Returns [] if absent (G8)."""
        return [s.obs_state[key] for s in self.steps if key in s.obs_state]

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


# --- G3: a loader that refuses unknown majors -------------------------------

def rollout_from_dict(d: dict) -> Rollout:
    got = d.get("schema_version", "0")
    if got.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        raise ValueError(f"incompatible trace schema {got} (expected "
                         f"{SCHEMA_VERSION}); migrate before loading")
    d = dict(d)
    d["steps"] = [Step(**s) for s in d["steps"]]
    return Rollout(**d)


# --- G6: structural protocols, so third-party adapters need no inheritance ---

def identity_hash(identity: dict) -> str:
    """Stable short hash of a semantic identity dict."""
    return hashlib.sha1(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]


def derive_id(identity: dict) -> str:
    """`name@hash` -- readable, and changes when anything load-bearing changes.

    Deliberately NOT a git SHA. A SHA is simultaneously too coarse (a README
    edit invalidates every cached rollout) and too weak (a dirty working tree
    keeps the SHA stable while the code changes underneath it, which
    manufactures confidence in a stale cache during exactly the phase where
    thresholds get nudged constantly). Hash the load-bearing VALUES instead.
    """
    name = identity.get("name", "unnamed")
    return f"{name}@{identity_hash(identity)}"


def fingerprint_diff(a: dict, b: dict, _prefix: str = "") -> dict:
    """Which fingerprint fields differ, and how. Fed to the run summary.

    Recurses so a nested change names the field that moved
    (`runtime.mujoco`), not just the section it lives in.
    """
    out = {}
    for k in sorted(set(a) | set(b)):
        path = f"{_prefix}{k}"
        va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
        if isinstance(va, dict) and isinstance(vb, dict):
            out.update(fingerprint_diff(va, vb, path + "."))
        elif va != vb:
            out[path] = {"cached": va, "current": vb}
    return out


@runtime_checkable
class Policy(Protocol):
    policy_id: str
    action_dims: list[str]
    def identity(self) -> dict:
        """Every semantically load-bearing VALUE: constructor kwargs, decoding
        config, checkpoint revision/hash, action space. Two policies with equal
        identities must be behaviourally interchangeable."""
        ...
    def __call__(self, obs: Observation) -> Action: ...
    def reset(self) -> None: ...


@runtime_checkable
class Env(Protocol):
    env_id: str
    task_id: str
    action_dims: list[str]
    def identity(self) -> dict:
        """Every semantically load-bearing VALUE: constructor kwargs, anything
        affecting the goal predicate, control mode, action space, asset revision.

        Scope note: this covers ENV and POLICY identity only. Mining-layer
        detector config (`PhaseSegmenter` radii, `LOST_TARGET_M`) is deliberately
        outside it, and is safe today only because rollouts are persisted BEFORE
        classification -- see `TraceStore` and ARCHITECTURE.md section 7. If a
        classified rollout is ever persisted, that safety disappears and this
        fingerprint must grow to cover detector config.
        """
        ...
    def reset(self, seed: int, spec: PerturbationSpec) -> Observation: ...
    def step(self, action: Action) -> tuple[Observation, bool, bool, str]: ...


# --- G7 / G10: append-only trace store, resumable by content hash ------------

def cell_hash(policy_id, env_id, task_id, seed, spec: PerturbationSpec) -> str:
    """Content-addressed rollout id.

    `policy_id` and `env_id` are DERIVED from identity() (see `derive_id`), so
    changing a detector threshold or a checkpoint revision changes the id and
    the cache misses instead of silently serving a rollout from different code.
    """
    key = f"{policy_id}|{env_id}|{task_id}|{seed}|{sorted(spec.knobs)}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def runtime_context() -> dict:
    """Runtime facts that shift the DISTRIBUTION but must not key the cache.

    Deliberately outside `cell_hash`. Putting a MuJoCo version or a GPU model
    into the key would invalidate every rollout on every machine, which kills
    the cross-session resume that caching exists for. But a simulator or driver
    change absolutely can move success rates, so it must not pass unnoticed
    either -- hence: stored, compared, and a mismatch re-runs loudly.

    This is what makes the stored fingerprint load-bearing rather than
    decorative. A fingerprint equal to `identity()` could never disagree with
    a hit keyed on `hash(identity())`; it has to be strictly broader.
    """
    ctx = {"python": platform.python_version(),
           "platform": f"{platform.system()}-{platform.machine()}"}
    for mod in ("mujoco", "numpy", "torch", "lerobot", "robosuite"):
        try:
            ctx[mod] = __import__(mod).__version__
        except Exception:
            pass
    return ctx


def make_fingerprint(env, policy) -> dict:
    """Stored on every Rollout; compared on every cache hit.

    Strictly broader than what `cell_hash` keys on: `env`/`policy` identity are
    in the key (so a change misses the cache), `runtime` is not (so a change is
    caught and reported rather than silently invalidating the store).
    """
    return {"env": env.identity(), "policy": policy.identity(),
            "runtime": runtime_context()}


class TraceStore:
    """runs/<run_id>/rollouts.jsonl -- append-only, greppable, parquet later."""

    def __init__(self, root: str, run_id: str):
        self.dir = os.path.join(root, run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "rollouts.jsonl")
        self._seen = self._scan()
        self._cache = None

    def _scan(self) -> set[str]:
        seen = set()
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    if line.strip():
                        seen.add(json.loads(line)["rollout_id"])
        return seen

    def has(self, rollout_id: str) -> bool:
        return rollout_id in self._seen

    def append(self, r: Rollout) -> None:
        with open(self.path, "a") as f:
            f.write(r.to_json() + "\n")
        self._seen.add(r.rollout_id)

    def by_id(self) -> dict[str, Rollout]:
        """Cached rollouts keyed by id, for resumable sweeps."""
        if getattr(self, "_cache", None) is None or len(self._cache) != len(self._seen):
            self._cache = {r.rollout_id: r for r in self.load()}
        return self._cache

    def load(self) -> list[Rollout]:
        out = []
        if not os.path.exists(self.path):
            return out
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    out.append(rollout_from_dict(json.loads(line)))
        return out
