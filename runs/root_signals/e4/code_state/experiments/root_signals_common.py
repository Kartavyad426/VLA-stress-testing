"""Shared R-047 loader for the root-signal CPU analyses (R-059 part A, E2, E3, E6).

runs/r047/rollouts.jsonl holds 720 rows from two axes written in blocks: one
magnitude-0 row (empty `perturbation`, so it looks "nominal") followed by that
block's perturbed rows (5 for joint_radius_rad, 8 for camera_yaw_deg). The
noise seed and dir are NOT in the jsonl row; they are in each axis's
manifest.jsonl. rollout_id is shared across noise seeds, so it is never a key.

join_r047() assigns every row its axis from its block, then walks each axis's
rows in file order against that axis's manifest lines in order, asserting
task, magnitude, success and env_steps agree on every row. The join key is
then (scene, axis, magnitude, dir, noise_seed), and the npz path follows from
it (r047_run.py:189 naming).
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R047 = os.path.join(ROOT, "runs/r047")
AXES = ("joint_radius_rad", "camera_yaw_deg")


def _axis_of(p):
    for a in AXES:
        if a in p:
            return a
    return None


def join_r047(rows=None):
    """Returns (rows, meta): meta[i] = dict(line, scene, axis, magnitude, dir,
    noise_seed, success, env_steps, npz) for jsonl line i."""
    if rows is None:
        rows = [json.loads(l) for l in open(os.path.join(R047, "rollouts.jsonl"))]
    axis = [_axis_of(r["perturbation"] or {}) for r in rows]
    # a magnitude-0 row takes the axis of the block that follows it
    for i in range(len(rows) - 1, -1, -1):
        if axis[i] is None:
            nxt = next((axis[j] for j in range(i + 1, len(rows)) if axis[j] is not None
                        and (rows[j]["perturbation"] or {})), None)
            assert nxt is not None, f"line {i}: magnitude-0 row with no following block"
            axis[i] = nxt
    meta = [None] * len(rows)
    for a in AXES:
        man = [json.loads(l) for l in open(os.path.join(R047, a, "manifest.jsonl"))]
        idx = [i for i in range(len(rows)) if axis[i] == a]
        assert len(idx) == len(man), f"{a}: {len(idx)} jsonl rows vs {len(man)} manifest lines"
        for i, m in zip(idx, man):
            r = rows[i]
            scene = int(r["task_id"].rsplit("task", 1)[1])
            mag = float((r["perturbation"] or {}).get(a, 0.0))
            assert (scene, mag, bool(r["success"]), int(r["env_steps"])) == \
                (int(m["task_id"]), float(m["magnitude"]), bool(m["success"]), int(m["env_steps"])), \
                f"line {i} disagrees with {a} manifest: {(scene, mag, r['success'], r['env_steps'])} vs {m}"
            tag = f"{scene}_{float(m['magnitude']):.6f}_d{m['dir']}_n{m['noise_seed']}".replace(".", "p")
            meta[i] = dict(line=i, scene=scene, axis=a, magnitude=mag, dir=int(m["dir"]),
                           noise_seed=int(m["noise_seed"]), success=bool(r["success"]),
                           env_steps=int(r["env_steps"]), forwards=int(m["forwards"]),
                           npz=os.path.join(R047, a, f"{tag}.npz"))
    keys = [(m["scene"], m["axis"], m["magnitude"], m["dir"], m["noise_seed"]) for m in meta]
    assert len(set(keys)) == len(keys), "join keys not unique"
    return rows, meta
