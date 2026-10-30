"""Generated-plate test, step 3: composite generated plates and the real LaMa plate with the same replay states, and
compare them (CPU). res31's modules (composite, perceive) are imported read-only.

Per plate:
  objects   Grounding DINO (perceive.detect_multi, its default thresholds) for the scene's nouns; a hit whose box
            centre lies on the sim table mask = object baked into the plate -> plate rejected
  edges     table outline from the sim segmentation (render_empty.py, image-border pixels dropped) vs Canny edges of
            the plate: share of outline pixels with a plate edge within 3 / 10 px, median distance (px)
  light     calibrate_light.py fits to the real robot's shadow over video frames; a still has no robot, so it can't
            run on a generated plate. The composites reuse the real scene's light.json (direction AND strength).
  composite composite.Compositor(m, scene, plate) at frames of episode_info "review", same qpos for every plate

Writes <run>/compare.json, <run>/comp_<plate>_f<frame>.png, <run>/compare_grid.jpg, <run>/edges_<plate>.png.

    MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES= .venvs/real2sim/bin/python experiments/genplate/compare.py \
        --scene runs/real2sim/v1/orange --run runs/genplate/orange
UNVERIFIED until the checker passes it.
"""
import argparse, glob, json, os, sys
# 10-30: in .venvs/real2sim, importing triton (pulled in by torchvision.ops when Grounding DINO loads) AFTER mujoco
# segfaults (signal 11); loading it first avoids it. Which native library clashes is not established.
import triton.knobs  # noqa: F401
import torchvision.ops  # noqa: F401
import numpy as np
import cv2
import mujoco

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "experiments/real2sim"))
from composite import Compositor, load_plate  # noqa: E402
from perceive import detect_multi, read_frame  # noqa: E402

GRID_FRAMES = (0, 75, 126)  # before grasp, grasp, release (episode_info grasp_events)


def edge_metrics(plate_rgb, outline):
    g = cv2.GaussianBlur(cv2.cvtColor(plate_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY), (0, 0), 1.5)
    e = cv2.Canny(g, 40, 120) > 0
    dist = cv2.distanceTransform((~e).astype(np.uint8), cv2.DIST_L2, 5)[outline]
    return e, dict(n_outline_px=int(outline.sum()), within_3px=round(float((dist <= 3).mean()), 3),
                   within_10px=round(float((dist <= 10).mean()), 3), median_px=round(float(np.median(dist)), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    S, R = a.scene, a.run

    calib = json.load(open(os.path.join(S, "calib.json")))
    info = json.load(open(os.path.join(S, "episode_info.json")))
    nouns = json.load(open(os.path.join(S, "nouns.json")))["nouns"]
    video = os.path.join(info["episode"], f"{calib['cam']}.mp4")
    st = np.load(os.path.join(S, "replay_states.npz"))
    frames = info["frames"]["review"]
    idx = {int(f): i for i, f in enumerate(st["real_frames"])}

    table = cv2.imread(os.path.join(R, "table_mask.png"), 0) > 0
    outline = cv2.imread(os.path.join(R, "table_edge.png"), 0) > 0
    H, W = table.shape
    border = np.zeros_like(outline)
    border[:3], border[-3:], border[:, :3], border[:, -3:] = True, True, True, True
    outline &= ~border

    plates = {"lama": load_plate(S, "lama", "frame0+plate")}
    for p in sorted(glob.glob(os.path.join(R, "plate_*.png"))):
        plates[os.path.basename(p)[6:-4]] = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32)

    res = dict(scene=S, frames=frames, nouns=nouns, light="light.json of the real scene reused; calibrate_light.py needs the real robot's shadow in video frames, a still has none",
               plates={}, tag="UNVERIFIED")
    for name, plate in plates.items():
        assert plate.shape[:2] == (H, W), (name, plate.shape)
        r = {}
        dets = detect_multi(plate.astype(np.uint8), nouns)
        on_table = {k: v for k, v in dets.items()
                    if table[int(np.clip((v["box"][1] + v["box"][3]) / 2, 0, H - 1)), int(np.clip((v["box"][0] + v["box"][2]) / 2, 0, W - 1))]}
        r["detections"] = dets
        r["objects_on_table"] = sorted(on_table)
        r["rejected"] = bool(on_table) and name != "lama"
        e, r["edges"] = edge_metrics(plate, outline)
        vis = plate.astype(np.uint8).copy()
        vis[e] = (255, 0, 255)
        vis[cv2.dilate(outline.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (0, 255, 0)
        for v in dets.values():
            x0, y0, x1, y1 = (int(t) for t in v["box"])
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 3)
        cv2.imwrite(os.path.join(R, f"edges_{name}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        m = mujoco.MjModel.from_binary_path(os.path.join(S, "scene.mjb"))
        d = mujoco.MjData(m)
        comp = Compositor(m, S, plate, verbose=False)
        for f in frames:
            d.qpos[:] = st["qpos"][idx[f]]
            mujoco.mj_forward(m, d)
            out = np.clip(comp(d), 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(R, f"comp_{name}_f{f}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        comp.close()
        res["plates"][name] = r
        print(name, json.dumps({k: r[k] for k in ("objects_on_table", "rejected", "edges")}), flush=True)

    # grid: rows = frames, columns = real | lama | generated plates (half size)
    names = list(plates)
    rows = []
    for f in GRID_FRAMES:
        cells = [read_frame(video, f)[..., ::-1]] + [cv2.imread(os.path.join(R, f"comp_{n}_f{f}.png")) for n in names]
        rows.append(np.concatenate([cv2.resize(c, (W // 2, H // 2), interpolation=cv2.INTER_AREA) for c in cells], 1))
    cv2.imwrite(os.path.join(R, "compare_grid.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    res["grid"] = dict(frames=GRID_FRAMES, columns=["real"] + names)
    json.dump(res, open(os.path.join(R, "compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
