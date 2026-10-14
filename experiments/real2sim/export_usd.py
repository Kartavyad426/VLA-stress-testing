"""Real-to-sim visuals: export a sim run (per-frame qpos from replay/scripted_grasp) to one time-sampled USD
stage with MuJoCo's own USDExporter, for rendering in Blender.

Notes (checked in mujoco/usd/*.py, MuJoCo 3.15.0): one stage, poses set per frame (time code = frame index,
starting at 1), Z-up; cameras are written with a fixed 12 mm focal length, so the calibrated camera is
rebuilt in Blender from calib.json instead (blender_render.py).

    .venvs/real2sim/bin/python experiments/real2sim/export_usd.py --out runs/real2sim/ep_orange_block --run scripted_grasp
"""
import argparse, glob, os
import numpy as np
import mujoco
from mujoco.usd import exporter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", choices=["replay", "scripted_grasp"], default="scripted_grasp")
    a = ap.parse_args()

    m = mujoco.MjModel.from_binary_path(os.path.join(a.out, "scene.mjb"))
    d = mujoco.MjData(m)
    q = np.load(os.path.join(a.out, f"{a.run}_states.npz"))["qpos"]
    pkg = f"usd_{a.run}"
    ex = exporter.USDExporter(model=m, height=720, width=1280, output_directory=pkg,
                              output_directory_root=a.out, camera_names=["ext"], verbose=False)
    for k in range(len(q)):
        d.qpos[:] = q[k]
        mujoco.mj_forward(m, d)
        ex.update_scene(data=d)
    ex.save_scene(filetype="usda")
    files = sorted(glob.glob(os.path.join(a.out, pkg, "frames", "*.usda")))
    print("wrote", files[-1], "frames", len(q))

    # Blender 5.2.2's Transform Cache import left every mesh near the origin for this file, although the
    # stage's matrices match MuJoCo's geom poses (checked), so also save world matrices per prim and frame
    # for blender_render.py to apply directly. USD matrices are row-vector (translation in the last row);
    # stored here transposed to column-vector form.
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(files[-1])
    prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Xform) and p.GetName().startswith("Mesh_Xform_")]
    mats = np.zeros((len(q), len(prims), 4, 4))
    for k in range(len(q)):
        for j, p in enumerate(prims):
            mats[k, j] = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode(k + 1))).T
    np.savez_compressed(os.path.join(a.out, pkg, "world_matrices.npz"), paths=np.array([str(p.GetPath()) for p in prims]), mats=mats)
    print("world matrices:", mats.shape)


if __name__ == "__main__":
    main()
