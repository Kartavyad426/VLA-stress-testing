"""Real-to-sim visuals: render a frame of an exported USD run in Blender (Cycles), composited onto the real
clean plate. Runs inside Blender's own Python:

    data/real2sim/tools/blender-5.2.2-linux-x64/blender -b --python experiments/real2sim/blender_render.py -- \
        --out runs/real2sim/ep_orange_block --run scripted_grasp --frame 90 [--device CPU|GPU] [--samples 64] [--check]

- camera rebuilt from calib.json (the USD exporter writes a fixed 12 mm camera): focal length and principal
  point from the ZED intrinsics; the principal-point shift signs are chosen by --check, which renders a
  marker sphere at known 3D points and compares its pixel with the pinhole projection
- sun light from light.json (fitted to the real robot's shadow), soft grey world light for ambient
- the MuJoCo table becomes a Cycles shadow catcher, so only the shadows of robot/objects land on the plate
Writes <out>/blender_<run>_f<frame>.png (composite) and prints render time.
"""
import argparse, json, os, sys, time
import numpy as np
import bpy
from mathutils import Matrix

W, H = 1280, 720


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default="scripted_grasp")
    ap.add_argument("--frame", type=int, default=90)
    ap.add_argument("--device", choices=["CPU", "GPU"], default="CPU")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--check", action="store_true")
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1:])


def make_camera(scene, K, R, t, sx, sy):
    cd = bpy.data.cameras.new("calibrated")
    cd.sensor_fit, cd.sensor_width = "HORIZONTAL", 36.0
    cd.lens = K[0, 0] * 36.0 / W
    cd.shift_x = sx * (K[0, 2] - W / 2) / W  # Blender shifts are in units of the larger image side
    cd.shift_y = sy * (K[1, 2] - H / 2) / W
    cd.clip_start, cd.clip_end = 0.01, 50
    cam = bpy.data.objects.new("calibrated", cd)
    scene.collection.objects.link(cam)
    M = np.eye(4)
    M[:3, :3] = R @ np.diag([1, -1, -1])  # Blender cameras look down -Z with +Y up, like MuJoCo
    M[:3, 3] = t
    cam.matrix_world = Matrix(M.tolist())
    scene.camera = cam
    return cam


def setup_render(scene, device, samples, path):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.device = device
    if device == "GPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for dv in prefs.devices:
            dv.use = dv.type == "OPTIX"
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = W, H, 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = path


def read_png(path):
    img = bpy.data.images.load(path)
    px = np.array(img.pixels[:], dtype=np.float32).reshape(img.size[1], img.size[0], img.channels)[::-1]
    bpy.data.images.remove(img)
    return px


def check_camera(a, K, R, t):
    """Pick principal-point shift signs: marker spheres at known points, compare pixels to the projection."""
    pts = [t + R @ np.array(v) for v in ([0.12, 0.08, 0.7], [-0.15, -0.1, 0.8])]
    best = None
    for sx in (1, -1):
        for sy in (1, -1):
            bpy.ops.wm.read_factory_settings(use_empty=True)
            scene = bpy.context.scene
            make_camera(scene, K, R, t, sx, sy)
            mat = bpy.data.materials.new("e")
            mat.use_nodes = True
            nt = mat.node_tree
            em = nt.nodes.new("ShaderNodeEmission")
            em.inputs["Strength"].default_value = 10
            nt.links.new(em.outputs[0], nt.nodes["Material Output"].inputs[0])
            err = 0.0
            for p in pts:
                for o in list(scene.objects):
                    if o.type == "MESH":
                        bpy.data.objects.remove(o)
                bpy.ops.mesh.primitive_uv_sphere_add(radius=0.004, location=p.tolist())
                bpy.context.object.data.materials.append(mat)
                path = os.path.join(a.out, "_cam_check.png")
                setup_render(scene, "CPU", 4, path)
                bpy.ops.render.render(write_still=True)
                px = read_png(path)
                ys, xs = np.nonzero(px[..., 3] > 0.5)
                pc = R.T @ (p - t)
                want = np.array([K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]])
                got = np.array([xs.mean(), ys.mean()]) if len(xs) else np.array([1e9, 1e9])
                err = max(err, float(np.linalg.norm(got - want)))
            print(f"camera check sx={sx:+d} sy={sy:+d}: worst error {err:.2f} px", flush=True)
            if best is None or err < best[0]:
                best = (err, sx, sy)
    return best


def main():
    a = args()
    c = json.load(open(os.path.join(a.out, "calib.json")))
    K, R, t = np.array(c["K"]), np.array(c["R_cam_to_base"]), np.array(c["t_cam_in_base"])
    signs_path = os.path.join(a.out, "blender_camera_signs.json")
    if a.check:
        err, sx, sy = check_camera(a, K, R, t)
        json.dump(dict(sx=sx, sy=sy, worst_px=err), open(signs_path, "w"))
        print("chosen", sx, sy, "error px", round(err, 2))
        if err > 2.0:
            sys.exit("ABORT: no shift-sign convention reproduces the pinhole projection")
        return
    sg = json.load(open(signs_path))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    usd = sorted(f for f in os.listdir(os.path.join(a.out, f"usd_{a.run}", "frames")) if f.endswith(".usda"))[-1]
    bpy.ops.wm.usd_import(filepath=os.path.join(a.out, f"usd_{a.run}", "frames", usd), import_cameras=False,
                          import_lights=False, import_materials=True, set_frame_range=True, read_mesh_uvs=True)
    scene = bpy.context.scene
    # poses from export_usd.py's world_matrices.npz (Blender's Transform Cache import left the meshes at the
    # origin for this file); each object's constraint names the USD prim it belongs to
    wm = np.load(os.path.join(a.out, f"usd_{a.run}", "world_matrices.npz"))
    index = {p: j for j, p in enumerate(wm["paths"])}
    n_posed = 0
    for o in scene.objects:
        tc = [c for c in o.constraints if c.type == "TRANSFORM_CACHE"]
        if not tc:
            continue
        j = index.get(tc[0].object_path)
        for c in tc:
            o.constraints.remove(c)
        if j is not None:
            o.matrix_world = Matrix(wm["mats"][a.frame, j].tolist())
            n_posed += 1
    bpy.context.view_layer.update()
    n_catch = 0
    for o in scene.objects:
        if o.type == "MESH" and "table" in o.name.lower():
            o.is_shadow_catcher = True
            n_catch += 1
    make_camera(scene, K, R, t, sg["sx"], sg["sy"])
    lj = json.load(open(os.path.join(a.out, "light.json")))
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy, sun.data.angle = 3.0, np.radians(15)  # soft-edged sun
    z = -np.array(lj["dir"])  # a sun shines along its local -Z
    x = np.cross([0, 0, 1], z) if abs(z[2]) < 0.99 else np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    Ms = np.eye(4)
    Ms[:3, :3] = np.c_[x, np.cross(z, x), z]
    sun.matrix_world = Matrix(Ms.tolist())
    scene.collection.objects.link(sun)
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.35, 0.35, 0.35, 1)
    scene.world = world

    path = os.path.join(a.out, f"_blender_{a.run}_f{a.frame}_rgba.png")
    setup_render(scene, a.device, a.samples, path)
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    secs = time.time() - t0

    fg = read_png(path)
    plate = read_png(os.path.abspath(os.path.join(a.out, "plate_clean.png")))[..., :3]
    out = fg[..., :3] + plate * (1 - fg[..., 3:4])  # Cycles writes premultiplied RGBA
    img = bpy.data.images.new("comp", W, H, alpha=False)
    img.pixels[:] = np.concatenate([out[::-1], np.ones((H, W, 1), np.float32)], -1).ravel().tolist()
    img.filepath_raw = os.path.join(a.out, f"blender_{a.run}_f{a.frame}.png")
    img.file_format = "PNG"
    img.save()
    print(json.dumps(dict(frame=a.frame, device=a.device, samples=a.samples, render_seconds=round(secs, 1),
                          shadow_catchers=n_catch, objects=len(scene.objects), posed=n_posed)))


if __name__ == "__main__":
    main()
