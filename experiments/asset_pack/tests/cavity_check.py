"""Checklist #8 on a built scene: for each bowl body, weld it in place (copy of the scene with the bowl's freejoint
removed is overkill; instead we pin it by zeroing its qvel each step), drop a 1 cm-radius ball at the bowl centre
from 5 cm above the rim, and compare where the ball comes to rest with the inner floor found by a ray cast against
the bowl's VISUAL geoms. open = ball reaches >= 80% of the true depth below the rim; filled = <= 20%.
Also: number of collision geoms on the bowl, its mass, and peak contact count during settle."""
import sys, json
import numpy as np, mujoco
sys.path.insert(0, "/tmp/claude-1000/-home-imerit-Documents-Code-VLA/fadab121-7275-4659-8165-7c7a75e14b9d/scratchpad/patch")
sys.path.insert(0, "/home/imerit/Documents/Code/VLA/experiments/asset_pack")
import build_scene_test as bs, catalogue as packs
from scipy.spatial.transform import Rotation as Rot
scene = sys.argv[1]
fit = json.load(open(f"{scene}/scene_fit.json"))


def bowl_spec(o):
    """Floor + this bowl alone, built exactly as build_scene builds it (pack route or LIBERO boxes)."""
    s = mujoco.MjSpec()
    s.option.timestep = 0.002; s.option.noslip_iterations = 3
    s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    k = o["asset_scale"]
    if o.get("asset_pack"):
        b = s.worldbody.add_body(name="bowl", pos=[0, 0, -k * o["asset_zmin"] + 0.001]); b.add_freejoint()
        child, _ = packs.prepare_child(o["asset_xml"], k, o["props"])
        fr = b.add_frame(pos=[-k * o["asset_cxy"][0], -k * o["asset_cxy"][1], 0])
        s.attach(child, frame=fr, prefix="bowl_")
    else:
        boxes, body0, _, zmin_a, _ = bs.bowl_asset()
        b = s.worldbody.add_body(name="bowl", pos=[0, 0, -k * zmin_a + 0.001]); b.add_freejoint()
        s.add_mesh(name="bv", file=bs.os.path.join(bs.BOWL_DIR, "akita_black_bowl.obj"), scale=[bs.BOWL_ASSET_SCALE * k] * 3)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="bv", contype=0, conaffinity=0, mass=0)
        for p, M, sz in boxes:
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=k * (p - body0), size=k * sz, density=300,
                       quat=Rot.from_matrix(M).as_quat(scalar_first=True), friction=[0.95, 0.3, 0.1])
    return s


out = {}
for label, o in fit["objects"].items():
    if o["kind"] != "bowl":
        continue
    spec = bowl_spec(o)
    m0 = spec.compile(); d0 = mujoco.MjData(m0); mujoco.mj_forward(m0, d0)
    bname = "bowl"
    bid = m0.body(bname).id
    desc = [i for i in range(m0.nbody) if m0.body_rootid[i] == bid]
    geoms = [g for g in range(m0.ngeom) if m0.geom_bodyid[g] in desc]
    col = [g for g in geoms if m0.geom_contype[g] or m0.geom_conaffinity[g]]
    vis = [g for g in geoms if g not in col]
    mass = float(sum(m0.body_mass[i] for i in desc))
    # rim top and inner floor along the vertical through the body centre (visual geoms only)
    c = d0.xpos[bid].copy()
    zs = [d0.geom_xpos[g][2] for g in vis]
    top = c.copy(); top[2] += 0.5
    geomgroup = np.zeros(6, np.uint8); geomgroup[:] = 1
    gid = np.zeros(1, np.int32)
    # cast down from above, hitting only this bowl's visual geoms: iterate ray hits, skipping others
    floor = None
    p = top.copy()
    for _ in range(20):
        dist = mujoco.mj_ray(m0, d0, p, np.array([0, 0, -1.0]), None, 1, -1, gid)
        if dist < 0: break
        hit = p + np.array([0, 0, -dist])
        if gid[0] in vis:
            floor = hit[2]; break
        p = hit + np.array([0, 0, -1e-4])
    # rim height: highest visual-mesh vertex
    rim = -1e9
    for g in vis:
        if m0.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m0.geom_dataid[g]
            v = m0.mesh_vert[m0.mesh_vertadr[mid]:m0.mesh_vertadr[mid] + m0.mesh_vertnum[mid]]
            rim = max(rim, float((v @ d0.geom_xmat[g].reshape(3, 3).T + d0.geom_xpos[g])[:, 2].max()))
    # drop test: add a ball, pin the bowl
    s2 = bowl_spec(o)
    ball = s2.worldbody.add_body(name="probe_ball", pos=[c[0], c[1], rim + 0.05])
    ball.add_freejoint()
    ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], density=1000)
    m = s2.compile(); d = mujoco.MjData(m)
    bj = m.body(bname).jntadr[0]; qa, va = m.jnt_qposadr[bj], m.jnt_dofadr[bj]
    q_bowl = d.qpos[qa:qa + 7].copy()
    peak = 0
    for _ in range(int(1.5 / m.opt.timestep)):
        d.qpos[qa:qa + 7] = q_bowl; d.qvel[va:va + 6] = 0
        mujoco.mj_step(m, d); peak = max(peak, d.ncon)
    bz = d.xpos[m.body("probe_ball").id][2] - 0.01  # ball bottom
    depth_true = rim - floor if floor is not None else None
    reached = rim - bz
    frac = reached / depth_true if depth_true else None
    out[label] = dict(n_collision_geoms=len(col), mass_kg=mass, rim_z=rim, inner_floor_z=floor, depth_true_m=depth_true,
                      ball_bottom_reached_m=reached, frac=frac,
                      verdict=None if frac is None else "open" if frac >= 0.8 else "filled" if frac <= 0.2 else "partial",
                      peak_contacts=peak)
print(json.dumps(out, indent=1))
