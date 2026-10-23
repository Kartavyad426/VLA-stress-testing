"""Copy a built scene with one fitted cube resized (and optionally its render colour overridden), 10-23.

The scene can't be rebuilt from scene.xml (camera intrinsics are set after compile; attached asset paths don't survive
the XML round trip), so the compiled model is edited: geom size, rbound, aabb, the body's BVH box; body mass and inertia (box, the
geom's density 600 as build_scene); the free joint's start height so the cube still rests on the table; then
mj_setConst. scene_fit.json and replay_states.npz get the same size / height change. --render_rgb writes
<out>/render_colour_overrides.json, which composite.Compositor applies after the real median colours.
The source scene is not touched. Exploratory, UNVERIFIED.

    .venvs/real2sim/bin/python experiments/real2sim/refit_block.py --scene runs/real2sim/ep_orange_block_stereo \
        --out runs/real2sim/ep_orange_block_stereo_yfit --object "yellow block" --edge_cm 3.27 --render_rgb 0.6,0.62,0.26
"""
import argparse, json, os, shutil
import mujoco
import numpy as np

DENSITY = 600.0  # build_scene: fitted boxes/ellipsoids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--object", required=True, help="scene_fit label of a fitted cube, e.g. 'yellow block'")
    ap.add_argument("--edge_cm", type=float, required=True)
    ap.add_argument("--render_rgb", default=None, help="r,g,b in 0..1 used for rendering this object instead of the real median")
    ap.add_argument("--note", default="", help="why; stored in scene_fit.json")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    shutil.copytree(a.scene, a.out, symlinks=True)
    m = mujoco.MjModel.from_binary_path(os.path.join(a.out, "scene.mjb"))
    fit = json.load(open(os.path.join(a.out, "scene_fit.json")))
    o = fit["objects"][a.object]
    if o["kind"] != "box" or len(set(o["half"])) != 1:
        raise SystemExit(f"{a.object} is not a fitted cube: {o['kind']} {o['half']}")
    b = m.body(o["body"]).id
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
    if len(gs) != 1:
        raise SystemExit(f"expected one geom on {o['body']}, found {len(gs)}")
    g = gs[0]
    h_old, h = float(m.geom_size[g][0]), a.edge_cm / 200.0
    if abs(h_old - o["half"][0]) > 1e-6:
        raise SystemExit(f"scene.mjb half {h_old} != scene_fit half {o['half'][0]}")
    dz = h - h_old
    m.geom_size[g] = [h, h, h]
    m.geom_rbound[g] = h * np.sqrt(3)
    m.geom_aabb[g] = [0, 0, 0, h, h, h]
    # the body's bounding-volume tree keeps the compiled size: left stale, collisions are dropped (block sank 0.4 cm)
    for n in range(m.body_bvhadr[b], m.body_bvhadr[b] + m.body_bvhnum[b]):
        if m.bvh_nodeid[n] == g:
            m.bvh_aabb[n] = [0, 0, 0, h, h, h]
    mass = DENSITY * (2 * h) ** 3
    dmass = mass - m.body_mass[b]
    m.body_mass[b] = mass
    m.body_inertia[b] = mass * (2 * h) ** 2 / 6
    m.body_subtreemass[b] = mass
    m.body_subtreemass[0] += dmass
    j = [jj for jj in range(m.njnt) if m.jnt_bodyid[jj] == b][0]
    qa = m.jnt_qposadr[j]
    m.qpos0[qa + 2] += dz
    m.body_pos[b][2] += dz
    d = mujoco.MjData(m)
    mujoco.mj_setConst(m, d)
    mujoco.mj_saveModel(m, os.path.join(a.out, "scene.mjb"), None)
    o.update(half=[h, h, h], center=[o["center"][0], o["center"][1], o["center"][2] + dz],
             refit=dict(edge_cm_before=round(200 * h_old, 3), edge_cm=a.edge_cm, mass_kg=mass, note=a.note, source_scene=a.scene))
    if a.render_rgb:
        rgb = [float(x) for x in a.render_rgb.split(",")]
        o["refit"]["render_rgb"] = rgb
        json.dump({o["body"]: rgb}, open(os.path.join(a.out, "render_colour_overrides.json"), "w"), indent=1)
    json.dump(fit, open(os.path.join(a.out, "scene_fit.json"), "w"), indent=1)
    rs = os.path.join(a.out, "replay_states.npz")
    if os.path.exists(rs):
        z = dict(np.load(rs))
        z["qpos"][:, qa + 2] += dz  # the cube never moves in the replay of this episode before the grasp; shift its height
        np.savez(rs, **z)
    print(f"{a.object}: edge {200 * h_old:.2f} -> {a.edge_cm:.2f} cm, mass {mass * 1000:.1f} g, start z +{dz * 100:.2f} cm -> {a.out}")


if __name__ == "__main__":
    main()
