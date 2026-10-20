"""One real wrist frame through every wrist-render fix (user request 10-20): what GR00T-DROID was shown at each stage,
at its 320x180 input size, next to the real frame, with the planned lowest fingertip height measured for that stage
in the image-swap tests (mean of the 10 approach states; numbers from the committed G-runs, not recomputed here).
Writes <out>/index.html + one PNG per stage (local page).

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/wrist_progression.py --frame 63 \
        --out runs/real2sim/policy/wrist_progression
"""
import argparse, json, os, sys
import numpy as np
import cv2
import h5py
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from policy_rollout import H, W  # noqa: E402
from composite import (Compositor, WristCompositor, WristRenderV2, load_plate, wrist_plate, wrist_table_texture,  # noqa: E402
                       load_gripper_library)

EP = "data/real2sim/droid/ep_orange_block"
OLD, NEW = "runs/real2sim/ep_orange_block", "runs/real2sim/ep_orange_block_stereo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=63)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    h = h5py.File(os.path.join(EP, "trajectory.h5"))
    J, G = h["observation/robot_state/joint_positions"][:], h["observation/robot_state/gripper_position"][:]
    k = a.frame
    k_close = int(np.argmax(G > 0.5))
    imgs = {}
    cap = cv2.VideoCapture(os.path.join(EP, "18026681.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, b = cap.read()
    imgs["real"] = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA)

    def posed(m):
        d = mujoco.MjData(m)
        arm = [m.joint(f"fr3_joint{j}").qposadr[0] for j in range(1, 8)]
        d.qpos[:] = m.qpos0
        d.qpos[arm] = J[k]
        dm = m.jnt_range[m.joint("g_right_driver_joint").id][1]
        for jn in ("g_right_driver_joint", "g_left_driver_joint"):
            d.qpos[m.joint(jn).qposadr[0]] = G[k] * dm
        mujoco.mj_forward(m, d)
        return d

    # old scene (10-15 geometry)
    m = mujoco.MjModel.from_binary_path(os.path.join(OLD, "scene.mjb"))
    plate = load_plate(OLD, "lama", "frame0+plate")
    comp = Compositor(m, OLD, plate, size=(H, W), verbose=False)  # also sets the real object colours, as in smoke1
    d = posed(m)
    r = mujoco.Renderer(m, H, W)
    r.update_scene(d, camera="wrist")
    imgs["plain_old"] = r.render().copy()
    r.close()
    imgs["tex_old"] = WristCompositor(m, plate, comp, (H, W), plate2=wrist_plate(m, OLD, EP, comp, "18026681"))(d)
    # rebuilt scene
    m = mujoco.MjModel.from_binary_path(os.path.join(NEW, "scene.mjb"))
    plate = load_plate(NEW, "lama", "frame0+plate")
    comp = Compositor(m, NEW, plate, size=(H, W), verbose=False)
    wp = wrist_plate(m, NEW, EP, comp, "18026681")
    tt = wrist_table_texture(m, NEW, EP, "18026681", list(range(0, k_close - 2, 2)))
    lib = load_gripper_library(os.path.join(NEW, "gripper_lib", "gripper_library.npz"))
    lw = json.load(open(os.path.join(NEW, "light_wrist.json")))
    d = posed(m)
    imgs["tex_new"] = WristCompositor(m, plate, comp, (H, W), plate2=wp)(d)
    imgs["T"] = WristCompositor(m, plate, comp, (H, W), plate2=wp, table_tex=tt)(d)
    hi = WristCompositor(m, plate, comp, (360, 640), plate2=wp, table_tex=tt)(d)
    imgs["TS"] = cv2.resize(np.clip(hi, 0, 255).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    imgs["V2"] = WristRenderV2(m, plate, comp, (H, W), wp, tt, lib)(d)
    imgs["V2_wrist"] = WristRenderV2(m, plate, comp, (H, W), wp, tt, lib, light=lw)(d)
    imgs = {n: np.clip(v, 0, 255).astype(np.uint8) for n, v in imgs.items()}

    stages = [
        ("plain_old", "1 · Plain sim render (10-15)", "What GR00T saw in the first smoke test: flat MuJoCo colours, old scene geometry (table ~6 cm too high).",
         "not measured (old scene); smoke1: reached, hovered, never closed", "4c56957"),
        ("tex_old", "2 · + projected table texture (10-15)", "Table textured from the exterior camera's background plate; still the old geometry.",
         "not measured (old scene)", "844466c"),
        ("tex_new", "3 · + correct geometry (10-16/19)", "Same render, rebuilt scene: table height from stereo, camera calibrated with metric anchors, bowl fix.",
         "−1.4 cm (G7; G3–G6: −1.2…−1.5)", "df2cc9f"),
        ("T", "4 · + table texture from the wrist video (G5)", "Table mosaicked from the real wrist frames instead of the exterior camera: sharp real grain and colour.",
         "−0.7 cm (G5, G6)", "250eeae"),
        ("TS", "5 · + rendered larger and shrunk like real frames", "Rendered at 640×360 and area-averaged to 320×180 (no jagged edges).",
         "−0.5 cm (G6, at 1280×720)", "ae72101"),
        ("V2", "6 · + real gripper (WristRenderV2)", "Our gripper replaced by the real gripper image cut from the wrist video (real colour, focus blur, bolts).",
         "+0.8 cm (G7)", "2b8c37e"),
        ("V2_wrist", "7 · + light fitted in the wrist view", "Robot shadow from a light fitted to the real shadows seen by the wrist camera (near overhead, darker).",
         "+1.2 cm (G7) — current best that needs no real frame", "e70feeb"),
        ("real", "Real wrist frame", "The real ZED Mini image at the same moment.", "+2.4 / +2.6 cm (same image twice; noise 0.3 cm)", "—"),
    ]
    cards = []
    for key, title, what, height, commit in stages:
        cv2.imwrite(os.path.join(a.out, f"{key}.png"), cv2.cvtColor(imgs[key], cv2.COLOR_RGB2BGR))
        cards.append(f"""<div class="card"><div class="pair"><figure><img src="{key}.png" alt="{title}"><figcaption>this stage</figcaption></figure>
<figure><img src="real.png" alt="real"><figcaption>real</figcaption></figure></div>
<h3>{title}</h3><p>{what}</p><p class="h">GR00T's planned lowest fingertip: <b>{height}</b></p><p class="meta">commit {commit}</p></div>""")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wrist View Progression</title><style>
:root{{--bg:#fbfbf9;--fg:#1d1d1b;--muted:#5f5f5a;--line:#dddcd5;--card:#f2f1ec}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#161615;--fg:#e8e7e2;--muted:#a3a29b;--line:#34332f;--card:#1f1f1d}}}}
:root[data-theme="dark"]{{--bg:#161615;--fg:#e8e7e2;--muted:#a3a29b;--line:#34332f;--card:#1f1f1d}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}}main{{max-width:1100px;margin:0 auto}}
h1{{font-size:1.5rem;margin:.2em 0}}.meta{{color:var(--muted);font-size:.85rem}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:14px 0}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}figure{{margin:0}}img{{width:100%;image-rendering:pixelated;border-radius:4px;display:block}}
figcaption{{font-size:.8rem;color:var(--muted)}}h3{{margin:.5em 0 .2em;font-size:1.05rem}}.card p{{margin:.2em 0}}.h b{{font-size:1.05rem}}
@media (max-width:640px){{.pair{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>One wrist frame through every fix</h1>
<p class="meta">res31, 10-20, internal. Real episode ep_orange_block, wrist frame {k} (approach, gripper open, ~0.8 s before the operator's grasp). Each stage is shown at GR00T's 320×180 input (pixels enlarged, not smoothed) next to the real frame. "Planned lowest fingertip" = how low GR00T-DROID plans to put its fingertips above the table, averaged over 10 recorded approach states in the image-swap tests (same recorded state, real exterior image, only the wrist image changes); higher = closer to the real image's plan. Exploratory, UNVERIFIED.</p>
{''.join(cards)}
<p class="meta">Remaining gap at stage 7: ~1.3 cm (≈1 cm near the grasp). Candidates: the live real lighting right around the gripper, compression, the sim objects. Script: experiments/real2sim/wrist_progression.py.</p>
</main></body></html>"""
    open(os.path.join(a.out, "index.html"), "w").write(html)
    strip = np.concatenate([np.concatenate([imgs[s[0]] for s in stages[:4]], 1), np.concatenate([imgs[s[0]] for s in stages[4:]], 1)], 0)
    cv2.imwrite(os.path.join(a.out, "strip.jpg"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", os.path.join(a.out, "index.html"))


if __name__ == "__main__":
    main()
