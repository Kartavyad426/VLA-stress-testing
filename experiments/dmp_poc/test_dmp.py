"""T1: DMP unit tests on synthetic curves (docs/DMP_POC_METHOD.html §7).

    .venvs/libero-plus/bin/python -m pytest -q experiments/dmp_poc/test_dmp.py
"""
import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dmp import DMP, OrientationDMP, align_rotation, linear_bridge, max_rotvec_deg  # noqa: E402

DT = 0.05  # 20 Hz control


def approach_curve(n=80):
    """A demo-like approach: forward and down with a lift-and-descend bump. It
    starts moving (sideways) and ends at rest, like a segment ending at a grasp."""
    t = np.linspace(0.0, 1.0, n)
    smooth = 3 * t ** 2 - 2 * t ** 3
    x = 0.25 * (1 - np.cos(np.pi * t)) / 2
    y = 0.05 * (1 - (1 - t) ** 3)
    z = 0.10 * np.sin(np.pi * t) ** 2 - 0.15 * smooth
    return np.stack([x, y, z], axis=1) + np.array([0.0, 0.1, 1.0])


def test_self_reproduction_under_1mm():
    y = approach_curve()
    dmp = DMP().fit(y, DT, smooth=False)
    Y, _ = dmp.rollout()
    assert Y.shape == y.shape
    assert np.abs(Y - y).max() < 1e-3


def test_start_and_goal_honoured():
    y = approach_curve()
    dmp = DMP().fit(y, DT)
    y0 = y[0] + np.array([0.08, -0.05, 0.03])
    g = y[-1] + np.array([-0.04, 0.02, 0.0])
    n = len(y) - 1
    Y, Yd = dmp.rollout(y0=y0, g=g, n_steps=n + 100)
    assert np.allclose(Y[0], y0)
    # at t = tau the reference is still a few mm short (the (g - y0) s term and
    # the spring lag); execute.py holds g after tau. Settled within 5 s.
    assert np.linalg.norm(Y[n] - g) < 6e-3
    assert np.linalg.norm(Y[-1] - g) < 1e-3
    assert np.linalg.norm(Yd[-1]) < 1e-3


def test_shape_kept_under_pure_start_shift():
    """Hoffmann form: shifting start and goal by the same vector shifts the path."""
    y = approach_curve()
    dmp = DMP().fit(y, DT, smooth=False)
    shift = np.array([0.1, -0.2, 0.05])
    Y, _ = dmp.rollout(y0=y[0] + shift, g=y[-1] + shift)
    assert np.abs(Y - (y + shift)).max() < 1e-3


def test_no_flip_under_goal_rotation():
    """Goal rotated 90 deg about z around the start: the bump in z keeps its sign
    and size (the original Ijspeert form would rescale or flip it)."""
    y = approach_curve()
    dmp = DMP().fit(y, DT, smooth=False)
    Rz = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    g = y[0] + Rz @ (y[-1] - y[0])
    Y, _ = dmp.rollout(g=g)
    bump_demo = (y[:, 2] - np.linspace(y[0, 2], y[-1, 2], len(y))).max()
    bump_new = (Y[:, 2] - np.linspace(Y[0, 2], Y[-1, 2], len(Y))).max()
    assert bump_new > 0
    assert abs(bump_new - bump_demo) < 0.01


def test_vector_scaling_maps_displacement():
    y = approach_curve()
    dmp = DMP().fit(y, DT, smooth=False)
    Y, _ = dmp.rollout(scaling="vector")       # same start/goal: identity
    assert np.abs(Y - y).max() < 1e-3
    g = y[0] + 2.0 * (y[-1] - y[0])
    Y2, _ = dmp.rollout(g=g, scaling="vector", n_steps=len(y) - 1 + 100)
    assert np.linalg.norm(Y2[-1] - g) < 1e-3


def test_start_velocity_honoured_and_decays():
    y = approach_curve()
    dmp = DMP().fit(y, DT)
    v0 = np.array([0.0, 0.3, 0.0])                 # 30 cm/s sideways at the join
    Y, Yd = dmp.rollout(yd0=v0)
    assert np.allclose(Yd[0], v0)
    # position continuous, and the first step moves along v0
    assert np.dot(Y[1] - Y[0], v0) > 0
    # critically damped transient, time constant tau/sqrt(K) (~6.5 steps here):
    # under 15 % of v0 left after ~3 time constants
    _, Yd_demo = dmp.rollout()
    assert np.linalg.norm(Yd[20] - Yd_demo[20]) < 0.15 * np.linalg.norm(v0)


def test_align_rotation():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b = rng.normal(size=3), rng.normal(size=3)
        R = align_rotation(a, b)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.allclose(R @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-9)
    a = np.array([1.0, 0.0, 0.0])
    assert np.allclose(align_rotation(a, -a) @ a, -a)


def _demo_rots(n=80):
    t = np.linspace(0, 1, n)
    smooth = 3 * t ** 2 - 2 * t ** 3                      # ends at rest
    eul = np.stack([0.3 * np.sin(np.pi * t) ** 2, 0.2 * smooth, -0.4 * smooth], axis=1)
    return Rotation.from_euler("xyz", eul) * Rotation.from_euler("x", 180, degrees=True)


def test_orientation_self_reproduction():
    rots = _demo_rots()
    odmp = OrientationDMP().fit(rots, DT, smooth=False)
    R, _, _ = odmp.rollout(rots[0], rots[len(rots) - 1])
    err = np.degrees((R.inv() * rots).magnitude()).max()
    assert err < 0.1


def test_orientation_new_start_goal():
    rots = _demo_rots()
    odmp = OrientationDMP().fit(rots, DT)
    R0 = Rotation.from_euler("z", 25, degrees=True) * rots[0]
    Rg = Rotation.from_euler("y", 10, degrees=True) * rots[len(rots) - 1]
    R, _, _ = odmp.rollout(R0, Rg, n_steps=len(rots) - 1 + 40)
    assert np.degrees((R[0].inv() * R0).magnitude()) < 1e-6
    assert np.degrees((R[len(R) - 1].inv() * Rg).magnitude()) < 0.5
    assert max_rotvec_deg(R, Rg) < 60


def test_linear_bridge():
    p0 = np.array([0.0, 0.0, 1.0])
    R0 = Rotation.identity()
    P = approach_curve(30)
    Rs = Rotation.from_euler("z", np.linspace(20, 30, 30)[:, None], degrees=True)
    p, r = linear_bridge(p0, R0, P, Rs, n_bridge=20)
    assert len(p) == len(r) == 20 + 30
    assert np.allclose(p[0], p0) and np.allclose(p[20], P[0]) and np.allclose(p[-1], P[-1])
    assert np.degrees((r[20].inv() * Rs[0]).magnitude()) < 1e-6
    step = np.linalg.norm(np.diff(p[:21], axis=0), axis=1)
    assert np.allclose(step, step[0])


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        DMP().fit(np.zeros((2, 3)), DT)
    with pytest.raises(RuntimeError):
        DMP().rollout()
