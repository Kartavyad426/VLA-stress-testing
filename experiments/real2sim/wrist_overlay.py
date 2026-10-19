"""Check image (pre-deploy checklist: object positions/sizes from a second viewpoint): outlines of the sim robot
(cyan) and sim objects (magenta, at the scene's own initial poses) over the REAL wrist-camera frames, with the sim arm
at the recorded joints. Use frames before the objects are touched; the wrist pose comes from FK + DROID's gripper
offset, so this view is independent of the exterior calibration (which is blind to errors along its own rays).

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/wrist_overlay.py \
        --scene runs/real2sim/ep_orange_block --episode data/real2sim/droid/ep_orange_block --frames 0,40,60
Writes <scene>/wrist_overlay.jpg (or --out).
"""
import argparse, json, os
import numpy as np
import cv2
import mujoco
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--frames", default="0,40,60")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
    d = mujoco.MjData(m)
    serial = json.load(open(os.path.join(a.scene, "scene_fit.json")))["wrist_camera"]["serial"]
    h = h5py.File(os.path.join(a.episode, "trajectory.h5"))
    J, G = h["observation/robot_state/joint_positions"][:], h["observation/robot_state/gripper_position"][:]
    arm = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    drv_max = m.jnt_range[m.joint("g_right_driver_joint").id][1]
    robot = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(("fr3_", "g_"))]
    free_bodies = {m.jnt_bodyid[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
    objs = [g for g in range(m.ngeom) if m.geom_bodyid[g] in free_bodies]
    cap = cv2.VideoCapture(os.path.join(a.episode, f"{serial}.mp4"))
    rows = []
    r = None
    for k in (int(x) for x in a.frames.split(",")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read wrist frame {k}")
        H, W = bgr.shape[:2]
        if r is None:
            r = mujoco.Renderer(m, H, W)
            r.enable_segmentation_rendering()
        q = m.qpos0.copy()
        q[arm] = J[k]
        for jn in ("g_right_driver_joint", "g_left_driver_joint"):
            q[m.joint(jn).qposadr[0]] = G[k] * drv_max
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera="wrist")
        seg = r.render()
        isg = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
        for geoms, col in ((robot, (255, 255, 0)), (objs, (255, 0, 255))):  # BGR: cyan, magenta
            mk = (isg & np.isin(seg[..., 0], geoms)).astype(np.uint8)
            cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(bgr, cs, -1, col, 3)
        img = cv2.resize(bgr, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        cv2.putText(img, f"wrist frame {k}  cyan = sim robot, magenta = sim objects", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        rows.append(img)
    if r is not None:
        r.close()
    out = a.out or os.path.join(a.scene, "wrist_overlay.jpg")
    cv2.imwrite(out, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", out)


if __name__ == "__main__":
    main()
