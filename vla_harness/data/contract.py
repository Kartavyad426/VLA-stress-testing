"""The source contract (docs/DATA_MINER_SPEC.html §1): one record shape for every source.

An episode on disk is

    <root>/episodes/<episode_id>.json     the EpisodeRecord, frames included
    <root>/images/<episode_id>/...        per-step PNGs written by the env (G2)
    <root>/manifest.jsonl                 one summary row per episode, append-only

Everything a record holds is env-native: the images as the env rendered them
(raw orientation, native size), the proprioception as the env reported it, the
action the env executed. Converting to a policy's convention (flip, resize,
state vector, gripper encoding) is the exporter's job, so one record serves any
policy trained on that env, and a convention bug is fixed in one place.

Inputs are never borrowed: every frame's images and state are from THIS
episode (spec §4.3, "never feed nominal state or images as inputs"). Nominal
information enters only through `action` on `driven` frames.

`_gt_*` keys never enter a record (I2). The outcome fields that are derived
from privileged state (closest approach) are analysis-only and live in
`outcome`, not in any frame.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

CONTRACT_VERSION = "1.0"

# provenance fields that must be set (not None) on every episode (G1)
REQUIRED_PROVENANCE = ("source", "policy_id", "axis", "magnitude", "scene_task_id",
                       "suite", "env_id", "env_seed", "noise_seed", "privileged")


@dataclass
class Provenance:
    """Stamped per episode. Used for G4 (held-out exclusion) and R-040's budget accounting."""
    source: str                          # "rescue", "rerender", "scripted", ...
    policy_id: str                       # the policy whose outputs label any frame (derive_id)
    axis: str                            # perturbation knob, e.g. "joint_radius_rad"
    magnitude: float
    scene_task_id: int                   # the env's task index (LIBERO-Plus variant id)
    suite: str
    env_id: str
    env_seed: int
    noise_seed: int | None
    privileged: bool                     # True only when a generator read `_gt_` state
    dir_seed: int | None = None          # joint-direction seed (start-pose axis)
    instance_id: str | None = None       # the source's own id for this start (resume key)
    band: str | None = None              # sampling band label, e.g. "0.1-0.2"
    extra: dict = field(default_factory=dict)

    def missing(self) -> list[str]:
        d = asdict(self)
        return [k for k in REQUIRED_PROVENANCE if d.get(k) is None]


@dataclass
class FrameRecord:
    t: int
    images: dict[str, str]               # camera -> PNG path, relative to the record root
    proprio: dict[str, list[float]]      # env-native proprioception (no `_gt_` keys)
    action: list[float] | None           # executed env action; None on the terminal frame
    driven: bool                         # the action came from outside the policy
    segment: str | None = None           # approach / grasp / transport / place, when known


@dataclass
class EpisodeRecord:
    episode_id: str
    provenance: Provenance
    instruction: str
    frames: list[FrameRecord]
    outcome: dict[str, Any]              # success, closest_approach_m, steps, termination, failure_family
    exportable: bool                     # successes only; failures are kept for analysis
    rollout_id: str | None = None        # the harness trace this record was built from
    image_meta: dict[str, Any] = field(default_factory=dict)   # size, orientation, cameras
    baseline: dict[str, Any] | None = None   # same instance and seeds, no intervention (G5)
    fingerprint: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION

    # --- io --------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "EpisodeRecord":
        d = dict(d)
        if str(d.get("contract_version", "0")).split(".")[0] != CONTRACT_VERSION.split(".")[0]:
            raise ValueError(f"record contract {d.get('contract_version')} != {CONTRACT_VERSION}")
        d["provenance"] = Provenance(**d["provenance"])
        d["frames"] = [FrameRecord(**f) for f in d["frames"]]
        return EpisodeRecord(**d)

    def summary(self) -> dict:
        p = self.provenance
        return {"episode_id": self.episode_id, "rollout_id": self.rollout_id,
                "source": p.source, "policy_id": p.policy_id, "axis": p.axis,
                "magnitude": p.magnitude, "band": p.band, "dir_seed": p.dir_seed,
                "scene_task_id": p.scene_task_id, "env_seed": p.env_seed,
                "noise_seed": p.noise_seed, "instance_id": p.instance_id,
                "privileged": p.privileged, "success": bool(self.outcome.get("success")),
                "exportable": self.exportable, "n_frames": len(self.frames),
                "n_driven": sum(f.driven for f in self.frames),
                "closest_approach_m": self.outcome.get("closest_approach_m")}


class EpisodeStore:
    """<root>/manifest.jsonl + <root>/episodes/*.json. Append-only; resumable by instance_id."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(os.path.join(root, "episodes"), exist_ok=True)
        self.manifest = os.path.join(root, "manifest.jsonl")

    def image_dir(self, episode_id: str) -> str:
        return os.path.join(self.root, "images", episode_id)

    def rows(self) -> list[dict]:
        if not os.path.exists(self.manifest):
            return []
        return [json.loads(l) for l in open(self.manifest) if l.strip()]

    def done_instances(self) -> set[str]:
        return {r["instance_id"] for r in self.rows() if r.get("instance_id")}

    def write(self, rec: EpisodeRecord) -> None:
        # the record first, then its manifest row: a crash between the two leaves an
        # orphan json that the next run overwrites, never a row without a record
        p = os.path.join(self.root, "episodes", f"{rec.episode_id}.json")
        tmp = p + ".part"
        with open(tmp, "w") as f:
            json.dump(rec.to_dict(), f, separators=(",", ":"))
        os.replace(tmp, p)
        with open(self.manifest, "a") as f:
            f.write(json.dumps(rec.summary()) + "\n")

    def load(self, episode_id: str) -> EpisodeRecord:
        return EpisodeRecord.from_dict(json.load(open(os.path.join(self.root, "episodes", f"{episode_id}.json"))))

    def records(self, exportable_only: bool = False):
        seen = set()
        for r in self.rows():
            if r["episode_id"] in seen:
                continue
            seen.add(r["episode_id"])
            if exportable_only and not r["exportable"]:
                continue
            yield self.load(r["episode_id"])

    def abspath(self, rel: str) -> str:
        return rel if os.path.isabs(rel) else os.path.join(self.root, rel)


# --- G1 at the record level ------------------------------------------------
def check_record(rec: EpisodeRecord, store: EpisodeStore | None = None, cameras=None,
                 action_dim: int | None = None, action_bound: float = 1.0,
                 proprio_keys=None) -> list[str]:
    """Contract violations for one record; [] when clean. Exporter-level schema
    checks (dtypes, shapes of the converted dataset) are separate (export G1)."""
    errs = []
    miss = rec.provenance.missing()
    if miss:
        errs.append(f"provenance missing {miss}")
    if not rec.instruction:
        errs.append("empty instruction")
    if not rec.frames:
        return errs + ["no frames"]
    if rec.frames[-1].action is not None:
        errs.append("last frame must be the terminal observation (action None)")
    for f in rec.frames:
        if any(k.startswith("_gt_") for k in f.proprio):
            errs.append(f"t={f.t}: privileged key in proprio (I2)")
        if proprio_keys and any(k not in f.proprio for k in proprio_keys):
            errs.append(f"t={f.t}: proprio missing {[k for k in proprio_keys if k not in f.proprio]}")
        if cameras and set(f.images) != set(cameras):
            errs.append(f"t={f.t}: cameras {sorted(f.images)} != {sorted(cameras)}")
        if store is not None:
            for cam, p in f.images.items():
                if not os.path.exists(store.abspath(p)):
                    errs.append(f"t={f.t}: missing image {cam} {p}")
        if f.action is not None:
            if action_dim is not None and len(f.action) != action_dim:
                errs.append(f"t={f.t}: action dim {len(f.action)} != {action_dim}")
            if any(abs(x) > action_bound + 1e-6 for x in f.action):
                errs.append(f"t={f.t}: action outside [-{action_bound}, {action_bound}]")
        if len(errs) > 20:
            errs.append("... truncated")
            break
    ts = [f.t for f in rec.frames]
    if ts != list(range(ts[0], ts[0] + len(ts))):
        errs.append("frame times are not consecutive")
    if rec.exportable and not rec.outcome.get("success"):
        errs.append("exportable but not a success")
    return errs


# --- G4: held-out exclusion ------------------------------------------------
class HeldOutViolation(RuntimeError):
    pass


class HeldOut:
    """Evaluation starts that must never appear in training (spec §5 G4).

    A list file is JSON with an "instances" list; each instance names at least
    `scene_task_id` and `dir_seed` (the start's identity: a direction is a ray,
    so any radius along it is the same start family). Several lists combine.
    """

    def __init__(self, paths=()):
        self.paths = list(paths)
        self.keys: dict[tuple[int, int], str] = {}
        for p in self.paths:
            for inst in json.load(open(p))["instances"]:
                self.keys[(int(inst["scene_task_id"]), int(inst["dir_seed"]))] = os.path.basename(p)

    def refuse(self, scene_task_id: int, dir_seed: int | None) -> None:
        if dir_seed is None:
            return
        hit = self.keys.get((int(scene_task_id), int(dir_seed)))
        if hit:
            raise HeldOutViolation(f"G4: (scene {scene_task_id}, dir seed {dir_seed}) is in {hit}")

    def violations(self, rows) -> list[dict]:
        """Manifest rows (or EpisodeRecord.summary() dicts) that hit a held-out key."""
        out = []
        for r in rows:
            if r.get("dir_seed") is None:
                continue
            hit = self.keys.get((int(r["scene_task_id"]), int(r["dir_seed"])))
            if hit:
                out.append({"episode_id": r["episode_id"], "list": hit})
        return out
