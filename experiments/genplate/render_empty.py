"""Generated-plate test, step 1: depth + segmentation of the EMPTY scene from the calibrated exterior camera (CPU).

Robot and movable objects are hidden (geom group trick as in composite.Compositor), so only the static world is
rendered: the floor plane and the table box. Nothing beyond them exists in the sim (no wall), so the depth there is
"far" and the image model invents the room.

Rendered at the camera's own size, 1280x720 (scene.mjb cam_resolution); FLUX takes multiples of 16, so no crop/pad.

Writes to --out:
  depth_m.npy          metric depth (m), float32, 0 where nothing is hit
  depth_control.png    FLUX control image: disparity 1/z normalised to 0..255 (near bright; beyond FAR_M or empty 0), 3 channels
  table_mask.png       table pixels (255) from segmentation
  table_edge.png       1-px outline of table_mask (used by compare.py for edge alignment)
  render_empty.json    sizes, depth range, normalisation, hidden geoms
  render_empty_check.png  LaMa plate | depth control | table outline over the LaMa plate

    MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES= .venvs/real2sim/bin/python experiments/genplate/render_empty.py \
        --scene runs/real2sim/v1/orange --out runs/genplate/orange
"""
import argparse, json, os
import numpy as np
import cv2
import mujoco

HIDE_GROUP = 5
FAR_M = 3.0  # the floor plane runs to the horizon (102 m); without a cap it takes most of the range and the table is flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cid = m.camera("ext").id
    W, H = (int(v) for v in m.cam_resolution[cid])
    hidden = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]  # everything not on the world body
    m.geom_group[hidden] = HIDE_GROUP
    opt = mujoco.MjvOption()
    opt.geomgroup[HIDE_GROUP] = 0

    rend = mujoco.Renderer(m, H, W)
    rend.enable_depth_rendering()
    rend.update_scene(d, camera="ext", scene_option=opt)
    depth = rend.render().astype(np.float32)
    rend.disable_depth_rendering()
    rend.enable_segmentation_rendering()
    rend.update_scene(d, camera="ext", scene_option=opt)
    seg = rend.render()
    rend.close()

    hit = seg[..., 0] >= 0
    depth[~hit] = 0.0
    table = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 0] == m.geom("table").id)
    floor = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 0] == m.geom("floor").id)

    disp = np.zeros_like(depth)
    disp[hit] = 1.0 / np.maximum(depth[hit], 1e-3)
    lo, hi = 1.0 / FAR_M, float(np.percentile(disp[hit], 99.5))  # beyond FAR_M (room scale) -> 0, like the empty pixels
    ctrl = np.clip((disp - lo) / (hi - lo), 0, 1) * 255
    ctrl = np.where(hit, ctrl, 0).astype(np.uint8)

    edge = cv2.morphologyEx(table.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    edge &= table  # inner 1-px outline

    np.save(os.path.join(a.out, "depth_m.npy"), depth)
    cv2.imwrite(os.path.join(a.out, "depth_control.png"), np.repeat(ctrl[..., None], 3, -1))
    cv2.imwrite(os.path.join(a.out, "table_mask.png"), table.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(a.out, "table_edge.png"), edge.astype(np.uint8) * 255)

    lama = cv2.imread(os.path.join(a.scene, "plate_clean_lama.png"))
    over = lama.copy()
    over[cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (0, 255, 0)
    check = np.concatenate([lama, cv2.cvtColor(ctrl, cv2.COLOR_GRAY2BGR), over], 1)
    cv2.imwrite(os.path.join(a.out, "render_empty_check.png"), cv2.resize(check, (check.shape[1] // 2, check.shape[0] // 2), interpolation=cv2.INTER_AREA))

    info = dict(scene=a.scene, size_wh=[W, H], aspect_plan="generate at the camera's 1280x720 (multiples of 16); no crop or pad",
                hidden_geoms=len(hidden), px_table=int(table.sum()), px_floor=int(floor.sum()), px_empty=int((~hit).sum()),
                depth_m_table=[float(depth[table].min()), float(depth[table].max())] if table.any() else None,
                depth_m_hit=[float(depth[hit].min()), float(depth[hit].max())],
                control=f"disparity 1/z, [1/{FAR_M} m, p99.5] -> 0..255; farther or empty -> 0", disp_range=[lo, hi], tag="UNVERIFIED")
    json.dump(info, open(os.path.join(a.out, "render_empty.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
