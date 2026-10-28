"""Object kinds, size bounds, colours and the gripper limit. Copied from TaskGen experiments/taskgen/core/kinds.py
(commit 4da2af3, branch exp/taskgen; docs/TASKGEN_DESIGN.html §3a; bounds are Agentic_task_creation §5's first proposal).
Simulator-neutral. Exploratory, UNVERIFIED.
"""
GRIPPER_OPENING_M = 0.085   # Robotiq 2F-85
GRASP_MAX_M = 0.080          # narrowest horizontal side must be below this to count as graspable (5 mm margin)

# kind -> (dimension the bound applies to, min m, max m)
BOUNDS = {
    "block": ("edge", 0.02, 0.06),
    "cup": ("height", 0.06, 0.10),
    "bowl": ("width", 0.10, 0.20),
    "fruit": ("width", 0.05, 0.09),
}

# Kitchen-pack category / name token -> kind (others: "item", no bounds, not resized)
PACK_KIND = {"glass_cup": "cup", "measuring_cup": "cup", "mug": "cup", "cup": "cup", "fruit_bowl": "bowl",
             "bowl": "bowl", "lemon_wedge": "fruit", "apple": "fruit", "orange": "fruit"}

PALETTE = {"red": (0.80, 0.10, 0.10, 1), "blue": (0.10, 0.25, 0.80, 1), "green": (0.10, 0.60, 0.20, 1),
           "yellow": (0.90, 0.85, 0.10, 1), "orange": (0.95, 0.45, 0.05, 1), "purple": (0.55, 0.15, 0.65, 1),
           "white": (0.92, 0.92, 0.92, 1), "black": (0.08, 0.08, 0.08, 1)}


def bound_dim(kind, size):
    """The dimension of `size` (x, y, z full extents) the kind's bound applies to."""
    which = BOUNDS[kind][0]
    if which == "height":
        return size[2]
    if which == "edge":
        return max(size)
    return max(size[0], size[1])


def within_bounds(kind, size):
    if kind not in BOUNDS:
        return True, None
    _, lo, hi = BOUNDS[kind]
    v = bound_dim(kind, size)
    return lo - 1e-9 <= v <= hi + 1e-9, (lo, hi, v)


def graspable(size):
    return min(size[0], size[1]) <= GRASP_MAX_M
