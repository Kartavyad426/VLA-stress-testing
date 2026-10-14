"""Real-to-sim visuals: fit the sim light to the real robot's shadow.

Real shadow (per frame): table pixels darker than the clean plate with the same chromaticity (same test as
calibrate_cam.motion_masks), robot pixels excluded. Sim shadow: the two-pass ratio (with / without robot and
objects, hidden via geom group) on table pixels, using the replay states (sim arm follows the recorded joints).
A directional light is grid-searched over azimuth x elevation to maximise mean shadow IoU; the real shadow
strength (median real/plate ratio inside real shadows) is measured for the composite.

Writes <out>/light.json and <out>/light_check.png (real shadow red, sim shadow green, both yellow).

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/calibrate_light.py \
        --episode data/real2sim/droid/ep_orange_block --cam 22008760 --out runs/real2sim/ep_orange_block
"""
import argparse, json, os, sys
import numpy as np
import cv2
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay import read_video  # noqa: E402

FRAMES = [30, 45, 60, 72, 90, 110, 130]
DOWN = 4
HIDE_GROUP = 5


def light_dir(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    real = read_video(os.path.join(a.episode, f"{a.cam}.mp4"))
    H, W = real[0].shape[:2]
    h, w = H // DOWN, W // DOWN
    small = [cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) for x in real]
    plate = np.median(np.stack(small), axis=0)
    pc = plate / (plate.sum(-1, keepdims=True) + 1e-3)
    per = np.load(os.path.join(a.out, "perception.npz"))
    table_px = cv2.resize(per["mask/wooden table"].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    real_sh, ratios = {}, []
    for i in FRAMES:
        x = small[i]
        ratio = x.sum(-1) / (plate.sum(-1) + 1e-3)
        chroma = np.abs(x / (x.sum(-1, keepdims=True) + 1e-3) - pc).sum(-1)
        moving = np.abs(x - plate).max(-1) > 35
        shadow = (ratio > 0.2) & (ratio < 0.92) & (chroma < 0.06)
        robot = moving & ~shadow
        sh = shadow & table_px & ~cv2.dilate(robot.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        sh = cv2.morphologyEx(sh.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        real_sh[i] = sh
        ratios.append(ratio[sh])
    real_strength = float(np.median(np.concatenate(ratios)))
    print(f"real shadow: {sum(int(v.sum()) for v in real_sh.values())} px over {len(FRAMES)} frames, median brightness ratio {real_strength:.3f}")

    m = mujoco.MjModel.from_binary_path(os.path.join(a.out, "scene.mjb"))
    d = mujoco.MjData(m)
    st = np.load(os.path.join(a.out, "replay_states.npz"))["qpos"]
    table = m.geom("table").id
    fg = np.array([g for g in range(m.ngeom) if g != table and m.geom_bodyid[g] != 0])
    group0 = m.geom_group.copy()
    m.light_type[0] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    m.light_castshadow[0] = 1
    from calibrate_cam import set_camera
    cid = m.camera("ext").id
    c = json.load(open(os.path.join(a.out, "calib.json")))
    K = np.array(c["K"]); K[:2] /= DOWN
    set_camera(m, cid, np.array(c["R_cam_to_base"]), np.array(c["t_cam_in_base"]), K, w, h, "offset")
    rend = mujoco.Renderer(m, h, w)
    hide = mujoco.MjvOption()
    hide.geomgroup[HIDE_GROUP] = 0

    def sim_shadow(i):
        d.qpos[:] = st[i]
        mujoco.mj_forward(m, d)
        rend.update_scene(d, camera=cid)
        full = rend.render().astype(np.float32).mean(-1)
        rend.enable_segmentation_rendering()
        rend.update_scene(d, camera=cid)
        seg = rend.render()
        rend.disable_segmentation_rendering()
        m.geom_group[fg] = HIDE_GROUP
        rend.update_scene(d, camera=cid, scene_option=hide)
        lit = rend.render().astype(np.float32).mean(-1)
        m.geom_group[:] = group0
        on_table = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 0] == table)
        return on_table & (full / np.maximum(lit, 1) < 0.9)

    results = []
    for az in range(0, 360, 20):
        for el in (30, 40, 50, 60, 70, 80, 89):
            m.light_dir[0] = light_dir(az, el)
            ious = []
            for i in FRAMES:
                s = sim_shadow(i)
                ious.append((s & real_sh[i]).sum() / max(1, (s | real_sh[i]).sum()))
            results.append((float(np.mean(ious)), az, el))
    results.sort(reverse=True)
    best_iou, az, el = results[0]
    m.light_dir[0] = light_dir(az, el)
    print("best light: azimuth", az, "elevation", el, "mean shadow IoU", round(best_iou, 3), "| next:", [tuple(round(v, 3) for v in r) for r in results[1:4]])
    json.dump(dict(type="directional", azimuth_deg=az, elevation_deg=el, dir=light_dir(az, el).tolist(), shadow_iou=best_iou,
                   real_shadow_ratio=real_strength, top5=results[:5], frames=FRAMES), open(os.path.join(a.out, "light.json"), "w"), indent=1)

    tiles = []
    for i in FRAMES[::2]:
        s = sim_shadow(i)
        img = cv2.cvtColor(small[i].astype(np.uint8), cv2.COLOR_RGB2BGR)
        img[real_sh[i] & ~s] = (0.4 * img[real_sh[i] & ~s] + [0, 0, 150]).astype(np.uint8)
        img[s & ~real_sh[i]] = (0.4 * img[s & ~real_sh[i]] + [0, 150, 0]).astype(np.uint8)
        img[s & real_sh[i]] = (0.4 * img[s & real_sh[i]] + [0, 150, 150]).astype(np.uint8)
        tiles.append(img)
    cv2.imwrite(os.path.join(a.out, "light_check.png"), np.concatenate(tiles, 1))


if __name__ == "__main__":
    main()
