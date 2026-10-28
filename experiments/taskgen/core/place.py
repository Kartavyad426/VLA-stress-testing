"""Placement logic, simulator-neutral (docs/TASKGEN_DESIGN.html §3e): what a relation means, where to try, and whether
it still holds after physics settled. The simulator only builds, settles and reports positions / collision points.
Exploratory, UNVERIFIED.
"""
import numpy as np

from core.geometry import aabb, horizontal_gap

NEXT_TO_GAP = (0.01, 0.06)    # horizontal gap to the target for "next_to" (make_tasks' success rule allows -0.5..8 cm)
EDGE_MARGIN = 0.03            # keep this far inside the fitted table extent (stand-in for the camera-covered area, D2)


def table_ok(box, table, margin=EDGE_MARGIN):
    lo, hi = np.array(table["xy_lo"]) + margin, np.array(table["xy_hi"]) - margin
    return bool((box["lo"][:2] >= lo).all() and (box["hi"][:2] <= hi).all())


def candidates(relation, size, table, target=None, rng=None, n=24, footprint=None):
    """Candidate (x, y, z_bottom, yaw) for an object of full extents `size`. target: dict with 'aabb' and, for
    containers, 'cavity' (bowl-like) or 'inner' (box-shaped: centre, half, floor). footprint: (lo, hi) x/y offsets
    of the object's footprint from its reference point (default: a circle of the size's half diagonal)."""
    rng = rng or np.random.default_rng(0)
    zt = table["z"]
    out = []
    if relation == "on_table":
        if footprint is None:
            r = float(np.linalg.norm(size[:2])) / 2
            footprint = ((-r, -r), (r, r))
        lo = np.array(table["xy_lo"]) + EDGE_MARGIN - np.array(footprint[0])
        hi = np.array(table["xy_hi"]) - EDGE_MARGIN - np.array(footprint[1])
        if (hi < lo).any():
            return []
        for _ in range(n):
            xy = rng.uniform(lo, hi)
            out.append((xy[0], xy[1], zt, rng.uniform(-np.pi, np.pi)))
    elif relation == "next_to":
        tb = target["aabb"]
        rt = float(np.linalg.norm(tb["size"][:2]) / 2)
        ro = float(np.linalg.norm(size[:2]) / 2)
        for k in range(n):
            ang = 2 * np.pi * (k + rng.uniform()) / n
            dist = rt + ro + rng.uniform(*NEXT_TO_GAP)
            out.append((tb["centre"][0] + dist * np.cos(ang), tb["centre"][1] + dist * np.sin(ang), zt, 0.0))
    elif relation == "on":
        tb = target["aabb"]
        for _ in range(n):
            j = rng.uniform(-0.25, 0.25, 2) * np.maximum(tb["size"][:2] - size[:2], 0)
            out.append((tb["centre"][0] + j[0], tb["centre"][1] + j[1], float(tb["hi"][2]), rng.uniform(-0.5, 0.5)))
    elif relation == "inside":
        if "inner" in target:
            c, h, floor = target["inner"]
            room = np.maximum(np.array(h[:2]) - np.array(size[:2]) / 2 - 0.005, 0)
            for _ in range(n):
                j = rng.uniform(-1, 1, 2) * room
                out.append((c[0] + j[0], c[1] + j[1], floor, 0.0))
        else:
            cv = target["cavity"]
            room = max(cv["inner_radius"] - float(np.linalg.norm(size[:2])) / 2 - 0.005, 0)
            for _ in range(n):
                a, r = rng.uniform(0, 2 * np.pi), room * np.sqrt(rng.uniform())
                out.append((cv["centre_xy"][0] + r * np.cos(a), cv["centre_xy"][1] + r * np.sin(a), cv["floor_z"], rng.uniform(-0.5, 0.5)))
    return out


def fits(relation, size, target, table):
    """Geometry-only answer before any physics: (ok, limiting reason or None, max uniform scale that would fit)."""
    if relation == "inside":
        if target is None:
            return False, "inside needs a target", None
        if "inner" in target:
            _, h, _ = target["inner"]
            room = np.array([2 * h[0], 2 * h[1], 2 * h[2]]) - 0.01   # 5 mm clearance on each side
            need = np.array(size)
            names = ("depth", "width", "height")
        elif target.get("cavity"):
            cv = target["cavity"]
            room = np.array([2 * cv["inner_radius"] - 0.01, cv["depth"] + 0.5 * size[2]])  # may stick out half its height
            need = np.array([float(np.linalg.norm(size[:2])), size[2]])
            names = ("opening", "depth")
        else:
            return False, "target has no cavity (it is solid)", None
        ratio = room / need
        k = float(ratio.min())
        if k >= 1:
            return True, None, k
        i = int(ratio.argmin())
        return False, f"{names[i]} {100 * room[i]:.1f} cm < {100 * need[i]:.1f} cm needed", k
    if relation == "on":
        tb = target["aabb"]
        if min(tb["size"][:2]) < 0.5 * min(size[:2]):
            return False, f"target top {100 * min(tb['size'][:2]):.1f} cm is too small to carry {100 * min(size[:2]):.1f} cm", None
        return True, None, None
    span = np.array(table["xy_hi"]) - np.array(table["xy_lo"]) - 2 * EDGE_MARGIN
    if max(size[:2]) > span.min():
        return False, "larger than the usable table", None
    return True, None, None


def holds(relation, obj_box, target, table, others):
    """Does the relation still hold after settling? others: {name: aabb} of every other object (for clearance)."""
    zt = table["z"]
    if not table_ok(obj_box, table, margin=0.0) and relation != "inside":
        return False, "off the table"
    if relation in ("on_table", "next_to") and obj_box["lo"][2] > zt + 0.01:
        return False, "not resting on the table"
    if relation == "next_to":
        g = horizontal_gap(obj_box, target["aabb"])
        if not (NEXT_TO_GAP[0] - 0.005 <= g <= NEXT_TO_GAP[1] + 0.02):
            return False, f"gap to target {100 * g:.1f} cm"
    if relation == "on":
        tb = target["aabb"]
        c = obj_box["centre"]
        if not (tb["lo"][0] <= c[0] <= tb["hi"][0] and tb["lo"][1] <= c[1] <= tb["hi"][1] and obj_box["lo"][2] > tb["hi"][2] - 0.008):
            return False, "not on top of the target"
    if relation == "inside":
        c = obj_box["centre"]
        if "inner" in target:
            ic, h, floor = target["inner"]
            if not (abs(c[0] - ic[0]) <= h[0] and abs(c[1] - ic[1]) <= h[1] and obj_box["lo"][2] < floor + 0.01):
                return False, "not inside the target"
        else:
            cv = target["cavity"]
            if not (np.linalg.norm(c[:2] - cv["centre_xy"]) < cv["inner_radius"] and obj_box["lo"][2] < cv["floor_z"] + 0.02):
                return False, "not inside the target"
    return True, None
