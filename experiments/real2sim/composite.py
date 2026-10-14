"""Real-to-sim visuals: composite a sim run onto the real camera's background (SIMPLER-style green screen).

  1. clean plate: per-pixel temporal median of the real video (the moving robot drops out), with the
     static real objects (bowl, blocks; perception masks, dilated) inpainted away so a moved sim object
     doesn't leave its real twin painted in the background
  2. per frame: render the sim from the calibrated camera; robot + object pixels (segmentation) come from
     the sim render, everything else (wall, real wooden table) from the plate
  3. shadows: render the table twice, with and without robot/objects (hidden via geom group; checked that
     hidden geoms cast no shadow); the brightness ratio on table pixels darkens the plate
  4. colours: sim blocks and bowl get the real median colour of their perception masks

Writes <out>/composite_<run>.mp4 (real | composite).

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/composite.py \
        --episode data/real2sim/droid/ep_orange_block --cam 22008760 --out runs/real2sim/ep_orange_block --run replay
"""
import argparse, os, sys
import numpy as np
import cv2
import imageio.v2 as imageio
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay import read_video  # noqa: E402

FPS = 15
HIDE_GROUP = 5


def scene_objects(out):
    """body name -> perception mask label, from build_scene's scene_fit.json (older fits: label-derived body names)."""
    import json
    objs = json.load(open(os.path.join(out, "scene_fit.json")))["objects"]
    return {o.get("body", lab.replace("#", "_").replace(" ", "_")): o.get("mask", lab) for lab, o in objs.items()}


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAMA_WEIGHTS = os.path.join(REPO, "data/real2sim/tools/lama/big-lama.pt")


def lama_inpaint(img, hole):
    """LaMa (advimman/lama, Apache-2.0) via the TorchScript export from simple-lama-inpainting (Apache-2.0),
    loaded directly so its numpy<2/pillow<10/opencv<5 pins don't touch this venv. Same preprocessing as that
    wrapper: float [0,1] CHW, binary mask, both padded symmetrically to a multiple of 8."""
    import torch
    model = torch.jit.load(LAMA_WEIGHTS, map_location="cpu").eval()
    H, W = hole.shape
    ph, pw = (-H) % 8, (-W) % 8
    x = np.pad(img.astype(np.float32).transpose(2, 0, 1) / 255, ((0, 0), (0, ph), (0, pw)), mode="symmetric")
    mk = np.pad((hole > 0).astype(np.float32)[None], ((0, 0), (0, ph), (0, pw)), mode="symmetric")
    with torch.inference_mode():
        y = model(torch.from_numpy(x)[None], torch.from_numpy(mk)[None])[0]
    return np.clip(y.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)[:H, :W]


def clean_plate(frames, per, objects, dilate=15, method="telea"):
    plate = np.median(np.stack(frames[::max(1, len(frames) // 150)]), axis=0).astype(np.uint8)  # <=~150 frames: memory
    hole = np.zeros(plate.shape[:2], np.uint8)
    for name in objects.values():
        if f"mask/{name}" in per:
            hole |= per[f"mask/{name}"].astype(np.uint8)
    hole = cv2.dilate(hole, np.ones((dilate, dilate), np.uint8))
    if method == "lama":
        return lama_inpaint(plate, hole)
    return cv2.inpaint(plate, hole, 7, cv2.INPAINT_TELEA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default="replay", help="<run>_states.npz from replay.py or scripted_grasp.py (--tag)")
    ap.add_argument("--no_shadows", action="store_true")
    ap.add_argument("--inpaint", choices=["telea", "lama"], default="telea", help="how the real objects are removed from the plate")
    a = ap.parse_args()

    m = mujoco.MjModel.from_binary_path(os.path.join(a.out, "scene.mjb"))
    d = mujoco.MjData(m)
    st = np.load(os.path.join(a.out, f"{a.run}_states.npz"))
    per = dict(np.load(os.path.join(a.out, "perception.npz")))
    real = read_video(os.path.join(a.episode, f"{a.cam}.mp4"))
    H, W = real[0].shape[:2]
    objects = scene_objects(a.out)
    plate = clean_plate(real, per, objects, method=a.inpaint).astype(np.float32)
    cv2.imwrite(os.path.join(a.out, f"plate_clean_{a.inpaint}.png") if a.inpaint != "telea" else os.path.join(a.out, "plate_clean.png"), cv2.cvtColor(plate.astype(np.uint8), cv2.COLOR_RGB2BGR))

    # real colours on the sim objects
    for body, name in objects.items():
        if f"mask/{name}" not in per:
            continue
        rgb = np.median(per["image"][cv2.erode(per[f"mask/{name}"].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0], axis=0) / 255.0
        for g in range(m.ngeom):
            if m.body(m.geom_bodyid[g]).name == body and m.geom_rgba[g, 3] > 0:
                m.geom_rgba[g, :3] = rgb

    # light fitted to the real robot's shadow (calibrate_light.py), if available
    light = None
    if os.path.exists(os.path.join(a.out, "light.json")):
        import json
        light = json.load(open(os.path.join(a.out, "light.json")))
        m.light_type[0] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        m.light_dir[0] = light["dir"]
        m.light_castshadow[0] = 1
        print("light from light.json: az", light["azimuth_deg"], "el", light["elevation_deg"], "real shadow ratio", round(light["real_shadow_ratio"], 3))

    table = m.geom("table").id
    fg_geoms = np.array([g for g in range(m.ngeom) if g != table and m.geom_bodyid[g] != 0])  # robot + objects
    rend = mujoco.Renderer(m, H, W)
    opt_all, opt_hide = mujoco.MjvOption(), mujoco.MjvOption()
    opt_hide.geomgroup[HIDE_GROUP] = 0
    group0 = m.geom_group.copy()

    writer = imageio.get_writer(os.path.join(a.out, f"composite_{a.run}.mp4"), fps=FPS, quality=7, macro_block_size=8)
    for k, q in enumerate(st["qpos"]):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        rend.update_scene(d, camera="ext", scene_option=opt_all)
        sim = rend.render().astype(np.float32)
        rend.enable_segmentation_rendering()
        rend.update_scene(d, camera="ext", scene_option=opt_all)
        seg = rend.render()
        rend.disable_segmentation_rendering()
        fg = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], fg_geoms)
        out = plate.copy()
        if not a.no_shadows:
            m.geom_group[fg_geoms] = HIDE_GROUP
            rend.update_scene(d, camera="ext", scene_option=opt_hide)
            lit = rend.render().astype(np.float32)
            m.geom_group[:] = group0
            on_table = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 0] == table)
            ratio = np.ones((H, W), np.float32)
            r = sim[on_table].mean(-1) / np.maximum(lit[on_table].mean(-1), 1.0)
            if light:  # sim says where the shadow is; the real video says how dark (measured median ratio)
                inside = np.clip((1.0 - r) / 0.1, 0.0, 1.0)  # same 0.9 threshold as calibrate_light
                ratio[on_table] = 1.0 - inside * (1.0 - light["real_shadow_ratio"])
            else:
                ratio[on_table] = np.clip(r, 0.35, 1.0)
            ratio = cv2.GaussianBlur(ratio, (0, 0), 3)
            out *= ratio[..., None]
        out[fg] = sim[fg]
        frame = real[int(st["real_frames"][k])]
        row = np.concatenate([frame, out.clip(0, 255).astype(np.uint8)], 1)
        row = cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2), interpolation=cv2.INTER_AREA)
        cv2.putText(row, "real" if a.run == "replay" else "real (first frame)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(row, f"composite ({a.run})", (W // 2 + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.append_data(row)
    writer.close()
    print("wrote", os.path.join(a.out, f"composite_{a.run}.mp4"), len(st["qpos"]), "frames")


if __name__ == "__main__":
    main()
