"""Demo search for a root (docs/DMP_POC_METHOD.html §4). No sim.

Candidates: the usable demos of the root's task (demos.py), steps inside the
root's phase (A approach: bowl not held; B carry: bowl held). Features, in
translation-only object frames (demos.py):
    A: EEF - target (3), EEF rotation (3), EEF velocity (3, weight 0.5), aperture (1)
    B: EEF - plate (3), target - EEF (3), EEF rotation (3), velocity (3, 0.5), aperture (1)
Rotation enters as the rotation vector relative to the mean demo rotation of
that task and phase (small rotations, §1.4). Each feature is divided by its
standard deviation over all candidate steps of the task and phase.

Per demo, the best step t* among steps leaving >= MIN_REMAIN steps of the
phase (so the bent motion still has the demo's approach); the top K demos by
distance are the matches. Coverage (d_cover) uses every step: the nearest
human state at all, which is what the coverage-gap measure asks.
d_gap: the 95th percentile, over candidate steps, of the distance to the
nearest step of any OTHER demo (leave-one-demo-out). A root with
d_best / d_gap > 1 is a coverage gap (§4).

Builds on the toy PoC's success search (experiments/poc_toy_search.py), which
matched with per-axis tolerances at grasp onset only: here every step of the
phase is a candidate and the distance is normalised and continuous.

UNVERIFIED: new numeric code, not yet checked by val.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demos as D  # noqa: E402

DT = 0.05
TOP_K = 3
VEL_W = 0.5
GAP_PCT = 95
GAP_STRIDE = 2          # d_gap uses every 2nd candidate step (speed)
MIN_REMAIN = 20         # a candidate step leaves >= 1 s of its phase (10-14: a match 3 steps
                        # before the demo's grasp turned the approach into a sideways shove)


def features(phase, eef_pos, eef_rot: Rotation, eef_vel, aperture, target_pos, plate_pos, R_ref: Rotation):
    """Raw feature rows (before normalisation), shape (n, F). Inputs (n, ...)."""
    eef_pos, eef_vel = np.atleast_2d(eef_pos), np.atleast_2d(eef_vel)
    target_pos, plate_pos = np.atleast_2d(target_pos), np.atleast_2d(plate_pos)
    rot = (R_ref.inv() * eef_rot).as_rotvec().reshape(-1, 3)
    ap = np.atleast_1d(aperture).reshape(-1, 1)
    if phase == "A":
        return np.hstack([eef_pos - target_pos, rot, eef_vel, ap])
    return np.hstack([eef_pos - plate_pos, target_pos - eef_pos, rot, eef_vel, ap])


def weights(phase):
    w = np.ones(10 if phase == "A" else 13)
    w[-4:-1] = VEL_W
    return w


@dataclass
class Pool:
    task: str
    phase: str
    X: np.ndarray            # (N, F) normalised features
    demo: np.ndarray         # (N,) demo index into demos
    step: np.ndarray         # (N,) step within the demo
    remain: np.ndarray       # (N,) steps left in the phase from this step
    mu: np.ndarray
    sigma: np.ndarray
    R_ref: Rotation
    demos: list
    d_gap: float


def _phase_range(d, phase):
    A, B, _ = d.phases
    return A if phase == "A" else B


@lru_cache(maxsize=None)
def pool(task: str, phase: str) -> Pool:
    ds = [d for d in D.load(task) if d.usable]
    rots = Rotation.concatenate([d.eef_rot[slice(*_phase_range(d, phase))] for d in ds])
    R_ref = rots.mean()
    rows, di, st, rem = [], [], [], []
    for i, d in enumerate(ds):
        s, e = _phase_range(d, phase)
        vel = np.gradient(d.eef_pos, DT, axis=0)
        rows.append(features(phase, d.eef_pos[s:e], d.eef_rot[s:e], vel[s:e], d.aperture[s:e],
                             d.target_pos[s:e], d.plate_pos[s:e], R_ref))
        di += [i] * (e - s)
        st += list(range(s, e))
        rem += list(range(e - s, 0, -1))       # steps left in the phase
    raw = np.vstack(rows)
    mu, sigma = raw.mean(0), raw.std(0) + 1e-9
    X = (raw - mu) / sigma * weights(phase)
    di, st, rem = np.array(di), np.array(st), np.array(rem)
    # leave-one-demo-out nearest-neighbour distances
    idx = np.arange(0, len(X), GAP_STRIDE)
    nn = np.empty(len(idx))
    for j, k in enumerate(idx):
        other = di != di[k]
        nn[j] = np.sqrt(((X[other] - X[k]) ** 2).sum(1)).min()
    return Pool(task, phase, X, di, st, rem, mu, sigma, R_ref, ds, float(np.percentile(nn, GAP_PCT)))


@dataclass
class Match:
    demo: object              # demos.Demo
    t_star: int               # best-matching step in the demo
    d: float                  # normalised distance of the step used for execution
    d_over_gap: float
    d_cover_over_gap: float   # coverage: nearest step of ANY demo, no MIN_REMAIN restriction
    tags: list


def search(task, phase, eef_pos, eef_rot, eef_vel, aperture, target_pos, plate_pos, k=TOP_K):
    """Top-k demos (distinct) for one root state."""
    P = pool(task, phase)
    x = features(phase, eef_pos, eef_rot, eef_vel, aperture, target_pos, plate_pos, P.R_ref)[0]
    x = (x - P.mu) / P.sigma * weights(phase)
    d = np.sqrt(((P.X - x) ** 2).sum(1))
    d_cover = float(d.min())
    best = {}
    for j in np.argsort(d):
        if P.remain[j] < MIN_REMAIN:
            continue
        i = int(P.demo[j])
        if i not in best:
            best[i] = j
            if len(best) == k:
                break
    out = [Match(P.demos[i], int(P.step[j]), float(d[j]), float(d[j] / P.d_gap), d_cover / P.d_gap, P.demos[i].tags)
           for i, j in best.items()]
    return out, P.d_gap


def search_root(root, task, phase):
    """search() from a restore.Root (positions in the R-047 world frame)."""
    plate = [n for n in root.objects if "plate" in n][0]
    tgt = root.target + "_main" if root.target + "_main" in root.objects else root.target
    ap = float(root.gripper_qpos[0] - root.gripper_qpos[1])
    return search(task, phase, root.eef_pos, root.eef_rot, root.eef_vel, ap,
                  root.objects[tgt][0], root.objects[plate][0])
