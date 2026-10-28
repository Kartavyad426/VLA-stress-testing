"""Simulator-neutral descriptions of objects added to a scene (docs/TASKGEN_DESIGN.html §2a, §3a).

Units: metres, radians, robot base frame (x forward, y left, z up). z_bottom None = standing on the table top.
Nothing here imports a simulator; a simulator layer (sim_mujoco/) turns these into its own model.
Exploratory, UNVERIFIED.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]
RGBA = Tuple[float, float, float, float]


@dataclass
class Box:
    """Primitive box, free to move."""
    name: str
    size: Vec3                      # full edge lengths
    rgba: RGBA
    xy: Vec2
    z_bottom: Optional[float] = None
    yaw: float = 0.0
    source: str = "primitive:box"


@dataclass
class PackAsset:
    """A model from a CC BY asset pack (experiments/asset_pack/catalogue.py key), free to move, scale k (1 = real size)."""
    name: str
    key: str                        # e.g. "kitchen/GlassCup002"
    xy: Vec2
    scale: float = 1.0
    pack: str = "kitchen"
    source: str = "pack"


@dataclass
class Drawer:
    """Primitive cabinet (fixed) with one sliding drawer, open front facing the robot (-x).
    The drawer is modelled pulled out by open_m; slide joint 0 = that pose, +open_m = closed."""
    name: str
    xy: Vec2
    outer: Vec3 = (0.20, 0.24, 0.10)    # depth (x), width (y), height (z)
    wall: float = 0.01
    open_m: float = 0.10
    gap: float = 0.003                  # clearance drawer ↔ cabinet
    plate: float = 0.008                # drawer plate thickness
    source: str = "primitive:drawer"

    # --- geometry of the parametric drawer, shared by the framework and every simulator layer ---
    def parts(self):
        """Cabinet and drawer boxes in the cabinet frame (origin: footprint centre on the table), drawer at qpos 0.
        Returns dict(cabinet=[(name, centre, half)], drawer=[...], handle=[...])."""
        D, W, H = self.outer
        t, g, td = self.wall, self.gap, self.plate
        cab = [("bottom", (0, 0, t / 2), (D / 2, W / 2, t / 2)),
               ("top", (0, 0, H - t / 2), (D / 2, W / 2, t / 2)),
               ("left", (0, W / 2 - t / 2, H / 2), (D / 2, t / 2, H / 2)),
               ("right", (0, -(W / 2 - t / 2), H / 2), (D / 2, t / 2, H / 2)),
               ("back", (D / 2 - t / 2, 0, H / 2), (t / 2, W / 2, H / 2))]
        x0, x1 = -D / 2, D / 2 - t - g          # drawer front / back faces when closed
        yi = W / 2 - t - g                       # drawer outer half width
        z0, z1 = t + 0.001, H - t - 0.004        # drawer bottom / top of its walls
        xm, hx = (x0 + x1) / 2, (x1 - x0) / 2
        zw, hz = (z0 + td + z1) / 2, (z1 - z0 - td) / 2
        zf = (z0 + H - t - g) / 2                # front panel centre (it closes the opening)
        o = -self.open_m
        drw = [("floor", (o + xm, 0, z0 + td / 2), (hx, yi, td / 2)),
               ("front", (o + x0 + td / 2, 0, zf), (td / 2, yi, (H - t - g - z0) / 2)),
               ("back", (o + x1 - td / 2, 0, zw), (td / 2, yi, hz)),
               ("left", (o + xm, yi - td / 2, zw), (hx, td / 2, hz)),
               ("right", (o + xm, -(yi - td / 2), zw), (hx, td / 2, hz))]
        handle = [("handle", (o + x0 - 0.03, 0, zf), (0.006, 0.05, 0.008)),
                  ("handle_postr", (o + x0 - 0.015, 0.045, zf), (0.015, 0.004, 0.006)),
                  ("handle_postl", (o + x0 - 0.015, -0.045, zf), (0.015, 0.004, 0.006))]
        return dict(cabinet=cab, drawer=drw, handle=handle)

    def inner(self, z_table):
        """The drawer's inner space at qpos 0, world frame: (centre, half sizes, floor z)."""
        D, W, H = self.outer
        t, g, td = self.wall, self.gap, self.plate
        x0, x1 = -D / 2, D / 2 - t - g
        yi = W / 2 - t - g
        z0, z1 = t + 0.001, H - t - 0.004
        xm, hx = (x0 + x1) / 2, (x1 - x0) / 2
        floor = z_table + z0 + td
        centre = (self.xy[0] - self.open_m + xm, self.xy[1], floor + (z1 - z0 - td) / 2)
        return centre, (hx - td, yi - td, (z1 - z0 - td) / 2), floor

    def body_names(self):
        return dict(cabinet=f"{self.name}_cabinet", drawer=f"{self.name}_drawer", joint=f"{self.name}_slide")


def describe(obj):
    """Plain dict for reports / configs."""
    d = asdict(obj)
    d["type"] = type(obj).__name__
    return d
