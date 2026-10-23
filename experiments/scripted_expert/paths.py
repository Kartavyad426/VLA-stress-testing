"""Smooth point-to-point hand paths for the scripted expert. No sim.

Each leg is a minimum-jerk (quintic) curve per axis from p0 to p1 that starts
at velocity v0 with zero acceleration and ends at rest with zero acceleration.
v0 is the hand's velocity when the leg starts, so the first leg joins the
failed rollout without a velocity jump (the jerk check looks at that join).

Duration: the shortest T (in control steps) whose sampled peak speed is within
vmax, and whose rotation rate is within omega_max. Orientation follows the
same minimum-jerk time law, slerped from R0 to R1.

rise_then_travel: a rise blended into the next leg without stopping at the
corner (10-20, after D ran out of the 200-step budget in the first test).

Human-speed law (HumanLaw, 10-20, after phase_timing showed the minimum-jerk
expert at 60-70% of human speed): a rest-to-rest "speed up, cruise, slow down"
profile whose ramps are quintic smoothsteps of the velocity (acceleration and
jerk zero at both ends of each ramp). Its duration makes the average speed the
human median vbar for the phase; the ramps are as long as the peak-speed cap
vmax allows (smoothest profile with that average), and the leg is stretched
while its peak jerk (V * 5.7735 / Tr^2) exceeds jcap or its rotation rate
exceeds omega_max. A nonzero start velocity v0 is added as a bump
v0 * t (1-x)^3 (1+3x), x = t/Tb, which has velocity v0 and zero acceleration at
t = 0 and vanishes with its first two derivatives at t = Tb. Tb is the
speed-up ramp (so the bump is gone before the cruise; fading it over the
whole leg pushed the cruise over the cap and stretched D's rise 23 -> 45
steps, 10-20), lengthened if needed to keep the join's jerk within jcap.

UNVERIFIED: new numeric code, not yet checked by val.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from scipy.spatial.transform import Rotation, Slerp

DT = 0.05            # 20 Hz control, as execute.DT
MIN_STEPS = 4
MAX_STEPS = 400


def quintic(p0, p1, v0, T):
    """Coefficients (a0, a1, a3, a4, a5) per axis; a2 = 0 (zero start acceleration)."""
    p0, p1, v0 = (np.asarray(x, dtype=float) for x in (p0, p1, v0))
    M = np.array([[T ** 3, T ** 4, T ** 5],
                  [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                  [6 * T, 12 * T ** 2, 20 * T ** 3]])
    rhs = np.stack([p1 - p0 - v0 * T, -v0, np.zeros_like(p0)])
    a3, a4, a5 = np.linalg.solve(M, rhs)
    return p0, v0, a3, a4, a5


def sample(coef, T, n):
    a0, a1, a3, a4, a5 = coef
    t = np.linspace(0.0, T, n + 1)[:, None]
    return a0 + a1 * t + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5


def min_jerk_s(n):
    """Rest-to-rest time law 0 -> 1 over n steps."""
    s = np.linspace(0.0, 1.0, n + 1)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def peak_speed(Y):
    return float((np.linalg.norm(np.diff(Y, axis=0), axis=1) / DT).max()) if len(Y) > 1 else 0.0


@dataclass(frozen=True)
class HumanLaw:
    vbar: float          # target average speed (human median for the phase), m/s
    vmax: float          # peak speed cap, m/s
    jcap: float          # peak jerk cap, m/s^3
    omega_max: float     # rotation rate cap, rad/s


SMOOTHSTEP_JERK = 5.7735   # max |d2/dx2 (6x^5 - 15x^4 + 10x^3)|, at x = 0.2113
SPEED_TOL = 1e-6     # m/s: sampled cruise speeds equal the cap up to float error


def cruise_s(n, r):
    """Speed-up / cruise / slow-down time law 0 -> 1 over n steps with ramps of
    r steps (0 < r <= n/2); velocity ramps are quintic smoothsteps."""
    T, Tr = n * DT, r * DT
    tau = np.arange(n + 1) * DT
    F = lambda x: x ** 6 - 3 * x ** 5 + 2.5 * x ** 4        # integral of the smoothstep  # noqa: E731
    pos = np.where(tau < Tr, Tr * F(np.clip(tau / Tr, 0, 1)),
                   np.where(tau <= T - Tr, 0.5 * Tr + (tau - Tr),
                            (T - Tr) - Tr * F(np.clip((T - tau) / Tr, 0, 1))))
    return pos / (T - Tr)


def bump(v0, n, nb=None):
    """(n+1, 3) displacement that starts at velocity v0 and is gone by step nb (default n)."""
    nb = n if nb is None else max(1, min(nb, n))
    T = nb * DT
    tau = np.minimum(np.arange(n + 1) * DT, T)
    x = tau / T
    return (tau * (1 - x) ** 3 * (1 + 3 * x))[:, None] * np.asarray(v0, dtype=float)[None, :]


BUMP_JERK = 36.0     # |jerk| of the bump at t = 0 is BUMP_JERK |v0| / Tb^2


def bump_steps(v0, r, n, jcap):
    """The bump fades over the speed-up ramp (r steps), or longer if that would
    exceed jcap at the join; never longer than the leg."""
    tb = np.sqrt(BUMP_JERK * float(np.linalg.norm(v0)) / jcap) if jcap > 0 else 0.0
    return int(min(n, max(np.ceil(r), np.ceil(tb / DT), 1)))


def human_steps(dist, ang, law: HumanLaw, n_min=MIN_STEPS):
    """(n, r): the fewest steps >= n_min, starting from the human-average duration,
    whose cruise profile keeps peak speed, jerk and rotation rate within the caps."""
    n = max(n_min, int(np.ceil(dist / law.vbar / DT)))
    while True:
        T = n * DT
        Tr = min(T / 2, T - dist / law.vmax) if dist > 0 else T / 2
        if Tr > 0:
            V = dist / (T - Tr)
            if (V * SMOOTHSTEP_JERK / Tr ** 2 <= law.jcap and ang / (T - Tr) <= law.omega_max) or n >= MAX_STEPS:
                return n, Tr / DT
        n += 1


def human_leg(p0, R0: Rotation, p1, R1: Rotation, v0, law: HumanLaw, n_min=MIN_STEPS):
    """Human-speed leg from (p0, R0) at velocity v0 to (p1, R1) at rest: (Y, R, s)."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    dist = float(np.linalg.norm(p1 - p0))
    ang = float((R1 * R0.inv()).magnitude())
    vcap = max(law.vmax, float(np.linalg.norm(v0)))
    n = n_min
    while True:
        n, r = human_steps(dist, ang, law, n)
        nb = bump_steps(v0, r, n, law.jcap)
        # the start velocity fades while the profile speeds up; if the two add up
        # past the cap, start the profile up to nb steps later rather than
        # stretching the whole leg
        for d in range(0, nb + 1):
            s = np.concatenate([np.zeros(d), cruise_s(n, r)])
            Y = p0 + (p1 - p0)[None, :] * s[:, None] + bump(v0, n + d, nb)
            if peak_speed(Y) <= vcap + SPEED_TOL:
                break
        if peak_speed(Y) <= vcap + SPEED_TOL or n >= MAX_STEPS:
            break
        n += 1
    return Y, Slerp([0.0, 1.0], Rotation.concatenate([R0, R1]))(s), s


def leg(p0, R0: Rotation, p1, R1: Rotation, v0, vmax, omega_max, law: HumanLaw | None = None):
    """(Y (n+1, 3), R (n+1 rotations)) from (p0, R0) at velocity v0 to (p1, R1) at rest.
    With `law`, the human-speed law; otherwise minimum jerk capped at vmax."""
    if law is not None:
        Y, R, _ = human_leg(p0, R0, p1, R1, v0, law)
        return Y, R
    dist = float(np.linalg.norm(np.subtract(p1, p0)))
    ang = float((R1 * R0.inv()).magnitude())
    vmax = max(vmax, float(np.linalg.norm(v0)))    # a leg cannot start slower than the hand already moves
    n = max(MIN_STEPS, int(np.ceil(1.875 * max(dist / vmax, ang / omega_max) / DT)))
    while True:
        Y = sample(quintic(p0, p1, v0, n * DT), n * DT, n)
        if peak_speed(Y) <= vmax or n >= MAX_STEPS:
            break
        n = int(np.ceil(n * 1.1))
    R = Slerp([0.0, 1.0], Rotation.concatenate([R0, R1]))(min_jerk_s(n))
    return Y, R


def rise_then_travel(p0, R0: Rotation, z1, p2, R2: Rotation, v0, vmax, omega_max, overlap=0.5,
                     law: HumanLaw | None = None):
    """A straight-up rise from p0 to height z1, blended into a rest-to-rest leg
    to (p2, R2): the second leg starts once `overlap` of the rise's duration
    has passed, and the two displacements add (C2-smooth, no stop at the
    corner). Durations stretch until the combined peak speed is within vmax.
    With `law`, both parts use the human-speed law and `overlap` is not used:
    the travel starts when the rise starts slowing down (10-20: a fixed
    half-way overlap put two cruises on top of each other, and the stretch
    to the speed cap made the approach ~1.5x slower than planned)."""
    p1 = np.array([p0[0], p0[1], z1], dtype=float)
    if law is not None:
        return _rise_then_travel_human(np.asarray(p0, dtype=float), R0, p1, np.asarray(p2, dtype=float), R2,
                                       v0, overlap, law)
    vmax = max(vmax, float(np.linalg.norm(v0)))
    n1 = max(MIN_STEPS, int(np.ceil(1.875 * abs(z1 - p0[2]) / vmax / DT)))
    n2 = max(MIN_STEPS, int(np.ceil(1.875 * max(np.linalg.norm(np.subtract(p2, p1)) / vmax,
                                                (R2 * R0.inv()).magnitude() / omega_max) / DT)))
    while True:
        Y1 = sample(quintic(p0, p1, v0, n1 * DT), n1 * DT, n1)
        Y2 = sample(quintic(p1, p2, np.zeros(3), n2 * DT), n2 * DT, n2)
        k = int(round(overlap * n1))
        n = max(n1, k + n2)
        t = np.arange(n + 1)
        Y = Y1[np.minimum(t, n1)] + Y2[np.clip(t - k, 0, n2)] - p1
        if peak_speed(Y) <= vmax or n >= MAX_STEPS:
            break
        n1, n2 = int(np.ceil(n1 * 1.1)), int(np.ceil(n2 * 1.1))
    s = np.concatenate([np.zeros(k), min_jerk_s(n2), np.ones(n + 1 - k - (n2 + 1))])
    R = Slerp([0.0, 1.0], Rotation.concatenate([R0, R2]))(s)
    return Y, R


def _rise_then_travel_human(p0, R0, p1, p2, R2, v0, overlap, law: HumanLaw):
    vcap = max(law.vmax, float(np.linalg.norm(v0)))
    n1 = n2 = MIN_STEPS
    while True:
        Y1, _, _ = human_leg(p0, R0, p1, R0, v0, law, n1)
        Y2, _, s2 = human_leg(p1, R0, p2, R2, np.zeros(3), law, n2)
        n1, n2 = len(Y1) - 1, len(Y2) - 1
        sp1 = np.linalg.norm(np.diff(Y1, axis=0), axis=1)
        k = int(np.nonzero(sp1 >= 0.999 * sp1.max())[0][-1])   # travel speeds up while the rise slows down
        n = max(n1, k + n2)
        t = np.arange(n + 1)
        Y = Y1[np.minimum(t, n1)] + Y2[np.clip(t - k, 0, n2)] - p1
        if peak_speed(Y) <= vcap + SPEED_TOL or n >= MAX_STEPS:
            break
        n1, n2 = n1 + 1, n2 + 1
    s = np.concatenate([np.zeros(k), s2, np.ones(n + 1 - k - (n2 + 1))])
    return Y, Slerp([0.0, 1.0], Rotation.concatenate([R0, R2]))(s)
