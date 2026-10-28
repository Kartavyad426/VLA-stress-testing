"""TaskGen MuJoCo layer: neutral objects (core/objects.py) -> MjSpec edits, before compiling (never by patching a
compiled .mjb). Defaults follow res31's build_scene where it has one (fitted boxes: density 600, friction
1.0/0.005/0.0001; pack models attached the way build_scene attaches pack meshes). Exploratory, UNVERIFIED.
"""
import mujoco
from scipy.spatial.transform import Rotation as Rot

from core.objects import Box, PackAsset, Drawer

BOX_DENSITY = 600.0
BOX_FRICTION = [1.0, 0.005, 0.0001]
WOOD, WOOD2, METAL = [0.55, 0.38, 0.22, 1], [0.66, 0.47, 0.29, 1], [0.5, 0.5, 0.52, 1]
BOX = mujoco.mjtGeom.mjGEOM_BOX


def add_box(spec, o: Box, z_table):
    zb = z_table if o.z_bottom is None else o.z_bottom
    b = spec.worldbody.add_body(name=o.name, pos=[o.xy[0], o.xy[1], zb + o.size[2] / 2 + 0.001],
                                quat=Rot.from_euler("z", o.yaw).as_quat(scalar_first=True))
    b.add_freejoint()
    b.add_geom(name=f"{o.name}_geom", type=BOX, size=[s / 2 for s in o.size], rgba=list(o.rgba),
               density=BOX_DENSITY, friction=BOX_FRICTION)
    return {}


def add_pack(spec, o: PackAsset, z_table):
    import catalogue as packs
    from build_scene import asset_footprint
    ent = packs.pack_catalogue(o.pack)[o.key]
    k = o.scale
    half_xy, zmin, zmax, cxy = asset_footprint(ent["xml"])
    child, info = packs.prepare_child(ent["xml"], k, dict(packs.DEFAULTS))
    b = spec.worldbody.add_body(name=o.name, pos=[o.xy[0], o.xy[1], z_table - k * zmin + 0.001])
    b.add_freejoint()
    fr = b.add_frame(pos=[-k * cxy[0], -k * cxy[1], 0])
    spec.attach(child, frame=fr, prefix=o.name + "_")
    info.update(licence=ent["licence"], size_m=[float(2 * k * half_xy[0]), float(2 * k * half_xy[1]), float(k * (zmax - zmin))])
    return info


def add_drawer(spec, o: Drawer, z_table):
    names = o.body_names()
    parts = o.parts()
    cab = spec.worldbody.add_body(name=names["cabinet"], pos=[o.xy[0], o.xy[1], z_table])  # no joint: fixed
    for nm, c, h in parts["cabinet"]:
        cab.add_geom(name=f"{o.name}_cab_{nm}", type=BOX, pos=list(c), size=list(h), rgba=WOOD, density=500)
    dr = cab.add_body(name=names["drawer"])
    dr.add_joint(name=names["joint"], type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], range=[-0.005, o.open_m],
                 damping=10.0, armature=0.01)
    for nm, c, h in parts["drawer"]:
        dr.add_geom(name=f"{o.name}_dr_{nm}", type=BOX, pos=list(c), size=list(h), rgba=WOOD2, density=500)
    for nm, c, h in parts["handle"]:
        dr.add_geom(name=f"{o.name}_{nm}", type=BOX, pos=list(c), size=list(h), rgba=METAL, density=500)
    return dict(closed_qpos=o.open_m, **names)


ADDERS = {Box: add_box, PackAsset: add_pack, Drawer: add_drawer}


def add(spec, obj, z_table):
    return ADDERS[type(obj)](spec, obj, z_table)
