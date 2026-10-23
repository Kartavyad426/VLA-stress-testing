"""Unit tests for paths.py (no sim).

    .venvs/libero-plus/bin/python -m pytest -q experiments/scripted_expert/test_paths.py
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DT, leg, peak_speed, quintic, rise_then_travel, sample  # noqa: E402


def test_rise_then_travel_rises_first_and_ends_at_goal():
    p0, R1 = np.array([0.0, 0.0, 1.0]), Rotation.from_euler("z", 0.3)
    Y, R = rise_then_travel(p0, Rotation.identity(), 1.2, [0.3, 0.0, 1.2], R1, np.zeros(3), vmax=0.27, omega_max=0.5)
    assert np.allclose(Y[-1], [0.3, 0.0, 1.2]) and len(R) == len(Y)
    assert (R[len(R) - 1] * R1.inv()).magnitude() < 1e-9
    assert peak_speed(Y) <= 0.27 + 1e-9
    # horizontal travel only begins after the rise is under way: no sideways motion while below half the rise
    low = Y[:, 2] < 1.1
    assert np.abs(Y[low, 0]).max() < 0.01
    # no stop at the corner: speed never returns to ~0 between start and end
    sp = np.linalg.norm(np.diff(Y, axis=0), axis=1) / DT
    assert sp[2:-2].min() > 0.01


def test_quintic_boundary_conditions():
    p0, p1, v0, T = np.array([0.0, 0.1, 1.0]), np.array([0.2, -0.1, 0.9]), np.array([0.05, 0.0, -0.1]), 1.3
    c = quintic(p0, p1, v0, T)
    Y = sample(c, T, 2600)
    h = T / 2600
    assert np.allclose(Y[0], p0) and np.allclose(Y[-1], p1)
    assert np.allclose((Y[1] - Y[0]) / h, v0, atol=1e-3)
    assert np.allclose((Y[-1] - Y[-2]) / h, 0.0, atol=1e-3)
    acc0 = (Y[2] - 2 * Y[1] + Y[0]) / h ** 2
    acc1 = (Y[-1] - 2 * Y[-2] + Y[-3]) / h ** 2
    assert np.abs(acc0).max() < 1e-2 and np.abs(acc1).max() < 1e-2


def test_leg_respects_speed_cap_and_ends_at_goal():
    R0 = Rotation.identity()
    R1 = Rotation.from_euler("z", 0.4)
    Y, R = leg([0, 0, 1], R0, [0.3, 0.1, 0.9], R1, np.zeros(3), vmax=0.25, omega_max=0.5)
    assert peak_speed(Y) <= 0.25 + 1e-9
    assert np.allclose(Y[-1], [0.3, 0.1, 0.9])
    assert (R[len(R) - 1] * R1.inv()).magnitude() < 1e-9
    rates = (R[1:] * R[:-1].inv()).magnitude() / DT
    assert rates.max() <= 0.5 * 1.0 + 1e-6


def test_leg_starts_at_given_velocity():
    v0 = np.array([0.1, 0.0, 0.0])
    Y, _ = leg([0, 0, 1], Rotation.identity(), [0.0, 0.2, 1.0], Rotation.identity(), v0, vmax=0.3, omega_max=0.5)
    # first sampled step moves roughly along v0 (zero start acceleration)
    first = (Y[1] - Y[0]) / DT
    assert abs(first[0] - 0.1) < 0.02


def test_leg_faster_start_than_cap_is_allowed():
    v0 = np.array([0.4, 0.0, 0.0])
    Y, _ = leg([0, 0, 1], Rotation.identity(), [0.2, 0.0, 1.0], Rotation.identity(), v0, vmax=0.3, omega_max=0.5)
    assert np.allclose(Y[-1], [0.2, 0.0, 1.0]) and peak_speed(Y) <= 0.4 + 1e-9


def _human_law():
    from paths import HumanLaw
    return HumanLaw(vbar=0.2, vmax=0.275, jcap=71.5, omega_max=0.54)


def test_cruise_s_endpoints_and_monotone():
    from paths import cruise_s
    s = cruise_s(40, 10)
    assert abs(s[0]) < 1e-12 and abs(s[-1] - 1) < 1e-9 and (np.diff(s) >= -1e-12).all()
    s = cruise_s(40, 20)                       # pure ramps, no cruise
    assert abs(s[-1] - 1) < 1e-9


def test_human_leg_average_speed_peak_and_jerk():
    from paths import human_leg, SMOOTHSTEP_JERK
    law = _human_law()
    Y, R, s = human_leg([0, 0, 1], Rotation.identity(), [0.4, 0, 1], Rotation.identity(), np.zeros(3), law)
    n = len(Y) - 1
    avg = 0.4 / (n * DT)
    assert 0.18 <= avg <= 0.2 + 1e-9               # human average (rounded up to whole steps)
    assert peak_speed(Y) <= law.vmax + 1e-9
    assert np.allclose(Y[-1], [0.4, 0, 1])
    j = np.linalg.norm(np.diff(Y, n=3, axis=0), axis=1) / DT ** 3
    assert j.max() <= law.jcap * 1.2               # sampled jerk ~ analytic cap


def test_human_leg_short_move_is_stretched_for_jerk():
    from paths import human_leg
    law = _human_law()
    Y, _, _ = human_leg([0, 0, 1], Rotation.identity(), [0, 0, 0.92], Rotation.identity(), np.zeros(3), law)
    j = np.linalg.norm(np.diff(Y, n=3, axis=0), axis=1) / DT ** 3
    assert j.max() <= law.jcap * 1.2 and np.allclose(Y[-1], [0, 0, 0.92])


def test_human_leg_start_velocity():
    from paths import human_leg
    v0 = np.array([0.1, 0.0, -0.05])
    Y, _, _ = human_leg([0, 0, 1], Rotation.identity(), [0, 0.3, 1], Rotation.identity(), v0, _human_law())
    assert np.allclose((Y[1] - Y[0]) / DT, v0, atol=0.01) and np.allclose(Y[-1], [0, 0.3, 1])


def test_rise_then_travel_human():
    law = _human_law()
    Y, R = rise_then_travel(np.array([0.0, 0.0, 1.0]), Rotation.identity(), 1.2, [0.4, 0.0, 1.1],
                            Rotation.from_euler("z", 0.2), np.zeros(3), 0.275, 0.54, 0.5, law=law)
    assert np.allclose(Y[-1], [0.4, 0.0, 1.1]) and len(R) == len(Y)
    assert peak_speed(Y) <= law.vmax + 1e-9
    low = Y[:, 2] < 1.1
    assert np.abs(Y[low & (np.arange(len(Y)) < len(Y) // 2), 0]).max() < 0.01
