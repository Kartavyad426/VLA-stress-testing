"""Render one LIBERO-Plus variant per (perturbation type, difficulty level).

Prefers a single base scene so rows differ only by the perturbation. Frames are
what the policy sees (180-degree flip), after one no-op step so sensor noise,
which LIBERO-Plus applies in step(), is present.
"""
import json, os, sys, textwrap
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageDraw
from vla_harness.envs.libero_env import LiberoEnv
from vla_harness.schema import PerturbationSpec, Action

SUITE = sys.argv[1] if len(sys.argv) > 1 else "libero_spatial"
BASE = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
cls = json.load(open("third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"))[SUITE]
D = Action(values=[0, 0, 0, 0, 0, 0, -1], dims=["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"])
CATS = ["Camera Viewpoints", "Robot Initial States", "Light Conditions", "Background Textures",
        "Sensor Noise", "Objects Layout", "Language Instructions"]
T = 256
os.makedirs("viz/libero_plus_levels", exist_ok=True)
picked = {}
rows = []
for cat in CATS:
    tiles = []
    for lv in range(1, 6):
        cand = [r for r in cls if r["category"] == cat and r["difficulty_level"] == lv]
        r = next((r for r in cand if r["name"].startswith(BASE)), cand[0] if cand else None)
        tile = Image.new("RGB", (T, T + 58), (245, 246, 249))
        dr = ImageDraw.Draw(tile)
        if r is None:
            dr.text((8, T // 2), f"no L{lv} variant", fill=(120, 120, 120))
        else:
            tid = r["id"] - 1
            env = LiberoEnv(suite=SUITE, task_id=tid, libero_plus=True)
            env.reset(seed=0, spec=PerturbationSpec.of())
            env.step(D)
            img = np.asarray(list(env._raw["pixels"].values())[0], np.uint8)[::-1, ::-1]
            tile.paste(Image.fromarray(img).resize((T, T)), (0, 58))
            suffix = r["name"][len(BASE):] if r["name"].startswith(BASE) else "  (other scene) " + r["name"][-38:]
            lines = [f"L{lv}  task_id {tid}"] + textwrap.wrap(suffix.strip("_"), 40)[:2]
            if cat == "Language Instructions":
                lines = [f"L{lv}  task_id {tid}"] + textwrap.wrap('"' + env.instruction + '"', 40)[:2]
            for i, line in enumerate(lines):
                dr.text((6, 4 + i * 17), line, fill=(20, 22, 29))
            picked[f"{cat}|L{lv}"] = {"task_id": tid, "name": r["name"], "instruction": env.instruction}
        tiles.append(tile)
    row = Image.new("RGB", (5 * T + 4 * 8 + 190, T + 58), (255, 255, 255))
    ImageDraw.Draw(row).text((8, (T + 58) // 2), cat, fill=(20, 22, 29))
    for i, t in enumerate(tiles):
        row.paste(t, (190 + i * (T + 8), 0))
    row.save(f"viz/libero_plus_levels/{SUITE}_{cat.replace(' ', '_')}.png")
    rows.append(row)
    print("rendered", cat, flush=True)
grid = Image.new("RGB", (rows[0].width, sum(r.height + 10 for r in rows)), (255, 255, 255))
y = 0
for r in rows:
    grid.paste(r, (0, y)); y += r.height + 10
grid.save(f"viz/libero_plus_levels/{SUITE}_all_levels.png")
json.dump(picked, open(f"viz/libero_plus_levels/{SUITE}_picked.json", "w"), indent=1)
print("done")
