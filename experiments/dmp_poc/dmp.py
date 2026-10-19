"""DMP core for the corrective-data PoC (docs/DMP_POC_METHOD.html §1-§3). Pure maths, no sim.

Position: discrete DMP in the Hoffmann/Pastor 2009 form, one per axis:

    tau s'  = -alpha_s s                                   s(0) = 1
    tau z'  = K (g - y) - D z - K (g - y0) s + K f(s)
    tau y'  = z
    f(s)    = sum_i w_i psi_i(s) s / sum_i psi_i(s),   psi_i(s) = exp(-h_i (s - c_i)^2)

f is fitted in absolute units, so a new start or goal shifts the demo's shape
instead of scaling it (no flip, no overscale). The vector-wise-scaling arm (C1
extra arm) multiplies the forcing term by S R, where R rotates the demo's
start->goal vector onto the new one and S is the ratio of their lengths.

At t = tau the reference is still short of a moved goal by a few mm (about
0.01 |delta y0| from the (g - y0) s term, plus spring lag; ~4 mm for a 10 cm
start offset in T1), and settles within a few seconds. Callers that need the
goal reached hold y = g after tau and track it.

Orientation: the same DMP on the rotation vector relative to the goal,
r = log(R_g^-1 R), goal 0. Valid while |r| stays well below pi; callers check
max_rotvec_deg() against the 60 deg limit in §1.4.

Rotations are scipy Rotation objects throughout, so no quaternion convention
leaks in here.

UNVERIFIED: new numeric code, not yet checked by val.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

K_DEFAULT = 150.0
ALPHA_S = -math.log(0.01)       # s(tau) = 0.01
N_BASIS = 30
RIDGE = 1e-8
SUBSTEPS = 10                   # Euler substeps per control step


def phase(t, tau, alpha_s=ALPHA_S):
    return np.exp(-alpha_s * np.asarray(t, dtype=float) / tau)


def _basis(n_basis, alpha_s):
    c = np.exp(-alpha_s * np.linspace(0.0, 1.0, n_basis))
    d = np.diff(c)
    h = np.empty(n_basis)
    h[:-1] = 1.0 / d ** 2
    h[-1] = h[-2]
    return c, h


def _features(s, c, h):
    """Normalised basis activations times s, shape (len(s), n_basis)."""
    s = np.atleast_1d(s)[:, None]
    psi = np.exp(-h[None, :] * (s - c[None, :]) ** 2)
    return psi * s / (psi.sum(axis=1, keepdims=True) + 1e-12)


def align_rotation(a, b):
    """Rotation matrix taking direction a onto direction b (Rodrigues)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:  # antiparallel: half turn about any axis normal to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return Rotation.from_rotvec(math.pi * axis / np.linalg.norm(axis)).as_matrix()
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


@dataclass
class DMP:
    """Hoffmann/Pastor DMP over d independent axes."""
    n_basis: int = N_BASIS
    K: float = K_DEFAULT
    alpha_s: float = ALPHA_S

    def __post_init__(self):
        self.D = 2.0 * math.sqrt(self.K)
        self.c, self.h = _basis(self.n_basis, self.alpha_s)
        self.w = None

    # -- fitting -------------------------------------------------------------
    def fit(self, y, dt, smooth=True):
        """Fit to a demo y of shape (T+1, d) sampled every dt seconds."""
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        n = len(y)
        if n < 3:
            raise ValueError("segment needs at least 3 samples")
        ys = savgol_filter(y, 5, 2, axis=0) if (smooth and n >= 5) else y
        self.tau = (n - 1) * dt
        self.dt = dt
        self.n = n
        self.y0_d = y[0].copy()
        self.g_d = y[-1].copy()
        yd = np.gradient(ys, dt, axis=0)
        ydd = np.gradient(yd, dt, axis=0)
        tau = self.tau
        s = phase(np.arange(n) * dt, tau, self.alpha_s)
        f_target = ((tau ** 2) * ydd + self.D * tau * yd) / self.K \
            - (self.g_d - ys) + (self.g_d - self.y0_d) * s[:, None]
        X = _features(s, self.c, self.h)
        A = X.T @ X + RIDGE * np.eye(self.n_basis)
        self.w = np.linalg.solve(A, X.T @ f_target)           # (n_basis, d)
        self.yd0_d = yd[0].copy()
        return self

    def forcing(self, s):
        return _features(s, self.c, self.h) @ self.w

    # -- rollout -------------------------------------------------------------
    def rollout(self, y0=None, g=None, yd0=None, tau=None, n_steps=None, scaling="none"):
        """Integrate from y0 (default demo start) to g (default demo goal).

        yd0: start velocity in units/s (the velocity-matched join, §3: pass the
        policy's velocity at the root). None uses the demo's own start velocity,
        so a plain rollout reproduces the demo; pass zeros for the start-from-rest
        arm. tau: duration (default the demo's). n_steps: control
        steps to return after the start (default round(tau/dt)). scaling:
        "none" (Hoffmann form) or "vector" (vector-wise scaling arm, 3-D only).
        Returns (Y, Yd): positions and velocities, shape (n_steps+1, d).
        """
        if self.w is None:
            raise RuntimeError("fit() first")
        y0 = self.y0_d if y0 is None else np.asarray(y0, dtype=float)
        g = self.g_d if g is None else np.asarray(g, dtype=float)
        tau = self.tau if tau is None else float(tau)
        dt = self.dt
        if n_steps is None:
            n_steps = int(round(tau / dt))
        M = np.eye(len(y0))
        if scaling == "vector":
            dv_d, dv = self.g_d - self.y0_d, g - y0
            if len(y0) == 3 and np.linalg.norm(dv_d) > 1e-3 and np.linalg.norm(dv) > 1e-3:
                M = (np.linalg.norm(dv) / np.linalg.norm(dv_d)) * align_rotation(dv_d, dv)
        elif scaling != "none":
            raise ValueError(scaling)
        y = y0.copy()
        z = tau * (self.yd0_d if yd0 is None else np.asarray(yd0, dtype=float))
        h = dt / SUBSTEPS
        Y, Yd = [y.copy()], [z / tau]
        t = 0.0
        for _ in range(n_steps):
            for _ in range(SUBSTEPS):
                s = math.exp(-self.alpha_s * t / tau)
                f = M @ self.forcing(s)[0]
                zdot = (self.K * (g - y) - self.D * z - self.K * (g - y0) * s + self.K * f) / tau
                y = y + h * z / tau
                z = z + h * zdot
                t += h
            Y.append(y.copy())
            Yd.append(z / tau)
        return np.array(Y), np.array(Yd)


class OrientationDMP:
    """Rotation-vector DMP relative to the goal orientation (§1.4)."""

    def __init__(self, **kw):
        self.dmp = DMP(**kw)

    def fit(self, rots: Rotation, dt, smooth=True):
        self.Rg_d = rots[len(rots) - 1]
        r = (self.Rg_d.inv() * rots).as_rotvec()
        self.max_deg_d = float(np.degrees(np.linalg.norm(r, axis=1).max()))
        self.dmp.fit(r, dt, smooth=smooth)
        return self

    def rollout(self, R0: Rotation, Rg: Rotation, omega0=None, tau=None, n_steps=None):
        """omega0: start rate of the rotation vector r, rad/s (approximately the
        angular velocity in the goal's frame for small r). None = the demo's."""
        r0 = (Rg.inv() * R0).as_rotvec()
        rr, rd = self.dmp.rollout(y0=r0, g=np.zeros(3), yd0=omega0, tau=tau, n_steps=n_steps)
        return Rg * Rotation.from_rotvec(rr), rr, rd


def max_rotvec_deg(rots: Rotation, Rg: Rotation):
    return float(np.degrees(np.linalg.norm((Rg.inv() * rots).as_rotvec(), axis=1).max()))


def linear_bridge(p0, R0: Rotation, P, Rs: Rotation, n_bridge=20):
    """I-Gen-style arm (§2.1): linear EEF interpolation from the root pose to the
    first pose of the (already transformed) demo segment, then the segment.
    Returns positions (n_bridge + len(P), 3) and a Rotation of the same length."""
    a = np.linspace(0.0, 1.0, n_bridge + 1)[1:, None]
    pb = (1 - a) * np.asarray(p0)[None, :] + a * P[0][None, :]
    key = Rotation.concatenate([R0, Rs[0]])
    rb = Slerp([0.0, 1.0], key)(a[:, 0])
    return np.vstack([np.asarray(p0)[None, :], pb, P[1:]]), Rotation.concatenate([R0, rb, Rs[1:]])
