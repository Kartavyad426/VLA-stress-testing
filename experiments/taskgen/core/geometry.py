"""Geometry from an object's collision points (sim truth; docs/TASKGEN_DESIGN.html §3b). Simulator-neutral: the
simulator layer only supplies the points (box corners, collision-mesh vertices) in the world frame.
Exploratory, UNVERIFIED.
"""
import numpy as np


def aabb(pts):
    pts = np.asarray(pts)
    lo, hi = pts.min(0), pts.max(0)
    return dict(lo=lo, hi=hi, centre=(lo + hi) / 2, size=hi - lo)


def cavity(pts):
    """Open-top cavity of a container from its collision points, or None if it is solid.
    floor = highest point near the axis (r < 35% of the outer radius); inner radius = nearest point to the axis in the
    upper 75% of the wall. Approximate: it reads the collision parts' corners/vertices, not a swept surface."""
    pts = np.asarray(pts)
    b = aabb(pts)
    c = b["centre"][:2]
    r = np.linalg.norm(pts[:, :2] - c, axis=1)
    outer = float(r.max())
    zmin, zmax = float(b["lo"][2]), float(b["hi"][2])
    near = r < 0.35 * outer
    floor = float(pts[near, 2].max()) if near.any() else zmin
    if floor > zmax - 0.2 * (zmax - zmin):
        return None
    upper = pts[:, 2] > floor + 0.25 * (zmax - floor)
    inner = float(r[upper].min()) if upper.any() else 0.0
    if inner < 0.2 * outer:
        return None
    return dict(centre_xy=c, floor_z=floor, rim_z=zmax, inner_radius=inner, outer_radius=outer, depth=zmax - floor)


def horizontal_gap(a, b):
    """Gap between two AABBs in the horizontal plane (negative = their footprints overlap)."""
    d = np.maximum(a["lo"][:2] - b["hi"][:2], b["lo"][:2] - a["hi"][:2])
    if (d < 0).all():
        return float(d.max())
    return float(np.linalg.norm(np.maximum(d, 0)))


def oriented_size(pts):
    """Full extents (a, b, height) in the yaw that gives the smallest footprint (1-degree search): sizes of a rotated
    object without the inflation of a world-axis box."""
    pts = np.asarray(pts)
    xy = pts[:, :2] - pts[:, :2].mean(0)
    best = None
    for deg in range(0, 90):
        t = np.radians(deg)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        q = xy @ R
        ext = q.max(0) - q.min(0)
        if best is None or ext[0] * ext[1] < best[0] * best[1]:
            best = ext
    return np.array([best[0], best[1], pts[:, 2].max() - pts[:, 2].min()])


def footprint_box(centre_xy, half_xy, z=(0.0, 0.0)):
    """AABB-like dict for a footprint (used by the cheap pre-filter before any physics)."""
    c = np.asarray(centre_xy, float)
    h = np.asarray(half_xy, float)
    return dict(lo=np.r_[c - h, z[0]], hi=np.r_[c + h, z[1]], centre=np.r_[c, (z[0] + z[1]) / 2], size=np.r_[2 * h, z[1] - z[0]])
