"""G6 (10-19, user-approved): which wrist-image differences make GR00T-DROID plan its grasp too low in sim?
Same 10 recorded approach states, same recorded state, real exterior image; only the wrist image changes.

Ingredients (applied in this order; full-resolution 1280x720 unless noted):
  T  table texture from the real wrist video (composite.wrist_table_texture, G5)
  L  real light/shadow: the real frame's low-frequency brightness relative to our (shadow-free) render, on table
     pixels                                                          [diagnostic: needs the real frame]
  G  real gripper pixels (sim gripper mask, dilated)                  [the gripper is rigid w.r.t. the camera, so a
                                                                       library of real gripper images would do]
  O  real object pixels (sim object masks, dilated)                   [diagnostic]
  F  defocus of the near layer (gripper) with edge bleed, sigma tuned so the gripper outline sharpness at 320x180
     matches the real image (--blur_A; --tune prints the sharpness table)
  C  H.264 round trip (one intra frame, libx264 crf --crf), as DROID's MP4s
  S  1280x720 -> 320x180 by area averaging, as the real frames reach the policy; without S the wrist image is
     rendered directly at 320x180 (what policy_rollout does now)
Real pixels (L/G/O) are pasted after F and C (they are blurred/compressed already).

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/g6_wrist.py \
        --scene runs/real2sim/ep_orange_block_stereo --episode data/real2sim/droid/ep_orange_block \
        --out runs/real2sim/policy/g6_wrist [--policy hold] [--tune]
"""
import argparse, json, os, sys, subprocess, tempfile
import numpy as np
import cv2
import h5py
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from policy_rollout import GrootClient, HoldPolicy, compute_eef_9d, IMG, H, W, DEFAULT_CKPT, REPO  # noqa: E402
from composite import Compositor, WristCompositor, load_plate, wrist_plate, wrist_table_texture  # noqa: E402

HH, WW = 720, 1280
CONDS = {  # name: (ingredients, native render)
    "real_real": None, "real_real_rep": None, "sim_old": None,
    "T": ("T", True), "TS": ("TS", False), "TSF": ("TSF", False), "TSFC": ("TSFC", False),
    "TSFCL": ("TSFCL", False), "TSFCLG": ("TSFCLG", False), "ALL": ("TSFCLGO", False),
    "ALL_noL": ("TSFCGO", False), "ALL_noG": ("TSFCLO", False), "ALL_noSFC": ("TLGO", True),
}


def lap_sharp(im, mk):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.abs(cv2.Laplacian(g, cv2.CV_32F))[mk].mean()) if mk.any() else float("nan")


def h264(img, crf):
    with tempfile.TemporaryDirectory() as td:
        src, dst = os.path.join(td, "in.png"), os.path.join(td, "out.mp4")
        cv2.imwrite(src, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", dst], check=True)
        ok, b = cv2.VideoCapture(dst).read()
    return cv2.cvtColor(b, cv2.COLOR_BGR2RGB)


def focus_blur(img, depth, sigma_near, z_near=0.20):
    """Defocus of the near layer only (the gripper, closer than z_near): the layer is blurred with its own alpha
    (premultiplied), so its edges bleed over the background as real defocus does. The table is left as is (our table
    texture is already softer than the real one). 10-19: a per-pixel layered blur kept the gripper outline ~1.5x too
    sharp because table pixels next to the gripper stayed sharp."""
    if sigma_near <= 0:
        return img
    a = (depth < z_near).astype(np.float32)
    f = img.astype(np.float32)
    ab = cv2.GaussianBlur(a, (0, 0), sigma_near)
    fb = cv2.GaussianBlur(f * a[..., None], (0, 0), sigma_near) / np.maximum(ab[..., None], 1e-3)
    out = ab[..., None] * fb + (1 - ab[..., None]) * f
    return out.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="45,48,51,54,57,60,63,66,69,72")
    ap.add_argument("--instruction", default="put the orange block in the bowl")
    ap.add_argument("--ext", default="22008760")
    ap.add_argument("--policy", choices=["groot", "hold"], default="groot")
    ap.add_argument("--blur_A", type=float, default=6.0, help="defocus sigma of the near layer (gripper), px at 1280x720")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--tune", action="store_true", help="only print the sharpness table for several blur_A values")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
    fit = json.load(open(os.path.join(a.scene, "scene_fit.json")))
    tz, ws = fit["table"]["z"], fit["wrist_camera"]["serial"]
    arm = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    l7, ps = m.body("fr3_link7").id, m.site("g_pinch").id
    drv_max = m.jnt_range[m.joint("g_right_driver_joint").id][1]
    h = h5py.File(os.path.join(a.episode, "trajectory.h5"))
    J, G, Cc = (h[f"observation/robot_state/{k}"][:] for k in ("joint_positions", "gripper_position", "cartesian_position"))
    k_close = int(np.argmax(G > 0.5))
    plate = load_plate(a.scene, "lama", "frame0+plate")
    comp = Compositor(m, a.scene, plate, size=(H, W), verbose=False)
    comp_ns = Compositor(m, a.scene, plate, shadows=False, size=(H, W), verbose=False)
    wp = wrist_plate(m, a.scene, a.episode, comp, ws)
    tt = wrist_table_texture(m, a.scene, a.episode, ws, list(range(0, k_close - 2, 2)))
    w_old = WristCompositor(m, plate, comp, (H, W), plate2=wp)
    w_lo = WristCompositor(m, plate, comp, (H, W), plate2=wp, table_tex=tt)
    w_lo_ns = WristCompositor(m, plate, comp_ns, (H, W), plate2=wp, table_tex=tt)
    w_hi = WristCompositor(m, plate, comp, (HH, WW), plate2=wp, table_tex=tt)
    w_hi_ns = WristCompositor(m, plate, comp_ns, (HH, WW), plate2=wp, table_tex=tt)
    robot = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(("fr3_", "g_"))]
    free_b = {m.jnt_bodyid[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
    objs = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] in free_b]
    tbl = m.geom("table").id
    d, dk = mujoco.MjData(m), mujoco.MjData(m)

    def real(serial, k, size):
        cap = cv2.VideoCapture(os.path.join(a.episode, f"{serial}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, b = cap.read()
        im = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
        return im if size is None else cv2.resize(im, (size[1], size[0]), interpolation=cv2.INTER_AREA)

    def masks(info):
        isg = info["seg"][..., 1] == mujoco.mjtObj.mjOBJ_GEOM
        return isg & np.isin(info["seg"][..., 0], robot), isg & np.isin(info["seg"][..., 0], objs), isg & (info["seg"][..., 0] == tbl)

    def build(ing, native, k, A):
        hs, wsz = (H, W) if native else (HH, WW)
        wc, wc_ns = (w_lo, w_lo_ns) if native else (w_hi, w_hi_ns)
        img, info = (wc_ns if "L" in ing else wc)(d, return_info=True)
        img = img.clip(0, 255).astype(np.uint8)
        gm, om, tm = masks(info)
        rf = real(ws, k, (hs, wsz))
        if "L" in ing:  # real low-frequency brightness relative to our shadow-free render, on the table
            gr, gs_ = (cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), max(2, wsz // 80)) for x in (rf, img))
            ratio = np.clip(gr / np.maximum(gs_, 1.0), 0.3, 1.6)
            ratio = np.where(tm, ratio, 1.0)
            img = (img.astype(np.float32) * ratio[..., None]).clip(0, 255).astype(np.uint8)
        if "F" in ing:
            img = focus_blur(img, info["depth"], A)
        if "C" in ing:
            img = h264(img, a.crf)
        dil = max(3, wsz // 200)
        if "G" in ing:
            gg = cv2.dilate(gm.astype(np.uint8), np.ones((dil, dil), np.uint8)) > 0
            img[gg] = rf[gg]
        if "O" in ing:
            oo = cv2.dilate(om.astype(np.uint8), np.ones((dil, dil), np.uint8)) > 0
            img[oo] = rf[oo]
        if not native:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return img

    def plan(chunk):
        hts, tl = [], []
        for j in range(15):
            dk.qpos[:] = m.qpos0
            dk.qpos[arm] = chunk[j, 10:17]
            mujoco.mj_kinematics(m, dk)
            hts.append(dk.site_xpos[ps][2] - tz)
            tl.append(np.degrees(np.arccos(np.clip(-dk.xmat[l7].reshape(3, 3)[2, 2], -1, 1))))
        return dict(min_height_cm=float(min(hts) * 100), mean_tilt_deg=float(np.mean(tl)))

    frames = [int(x) for x in a.frames.split(",")]

    def pose(k):
        d.qpos[:] = m.qpos0
        d.qpos[arm] = J[k]
        for jn in ("g_right_driver_joint", "g_left_driver_joint"):
            d.qpos[m.joint(jn).qposadr[0]] = G[k] * drv_max
        mujoco.mj_forward(m, d)

    def sharp_table(imgs, k):
        pose(k)
        _, info = w_lo(d, return_info=True)
        gm, om, tm = masks(info)
        grip_in = cv2.erode(gm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        outline = (cv2.dilate(gm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~grip_in
        tab = cv2.erode(tm.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        return {n: dict(outline=lap_sharp(im, outline), inside=lap_sharp(im, grip_in), table=lap_sharp(im, tab)) for n, im in imgs.items()}

    if a.tune:
        for A in (0.0, 4.0, 8.0, 12.0, 16.0):
            acc = {}
            for k in frames[::3]:
                pose(k)
                imgs = {"real": real(ws, k, (H, W)), "T": build("T", True, k, A), "TS": build("TS", False, k, A), "TSF": build("TSF", False, k, A)}
                for n, v in sharp_table(imgs, k).items():
                    for r, x in v.items():
                        acc.setdefault((n, r), []).append(x)
            print(f"A={A}: " + "  ".join(f"{n} outline {np.mean(acc[(n, 'outline')]):.1f} inside {np.mean(acc[(n, 'inside')]):.1f} table {np.mean(acc[(n, 'table')]):.1f}"
                                         for n in ("real", "T", "TS", "TSF")), flush=True)
        return

    pol = HoldPolicy() if a.policy == "hold" else GrootClient(DEFAULT_CKPT, os.path.join(REPO, ".venvs/groot/bin/python"), "cuda", 600)
    rows, sheet, sharp_rows = [], None, []
    try:
        for k in frames:
            pose(k)
            st = np.r_[compute_eef_9d(Cc[k]), G[k], J[k]].astype(np.float32)
            ext = real(a.ext, k, (H, W))
            imgs = {"real_real": real(ws, k, (H, W)), "sim_old": w_old(d).clip(0, 255).astype(np.uint8)}
            imgs["real_real_rep"] = imgs["real_real"]
            for c, spec in CONDS.items():
                if spec is not None:
                    imgs[c] = build(spec[0], spec[1], k, a.blur_A)
            row = dict(frame=k)
            for c in CONDS:
                row[c] = plan(pol({IMG[0]: ext, IMG[1]: imgs[c], "state": st, "instruction": a.instruction})["action"])
            rows.append(row)
            sharp_rows.append(sharp_table(imgs, k))
            print(f"f{k}: " + " ".join(f"{c} {row[c]['min_height_cm']:+.1f}" for c in CONDS), flush=True)
            if k == 63 or sheet is None:
                tiles = []
                for c in CONDS:
                    im = imgs[c].copy()
                    cv2.putText(im, c, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                    tiles.append(im)
                while len(tiles) % 4:
                    tiles.append(np.zeros_like(tiles[0]))
                sheet = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
    finally:
        pol.close()
    summ = {c: dict(min_height_cm_mean=float(np.mean([r[c]["min_height_cm"] for r in rows])),
                    mean_tilt_deg_mean=float(np.mean([r[c]["mean_tilt_deg"] for r in rows])),
                    sharp_outline=float(np.nanmean([s[c]["outline"] for s in sharp_rows])),
                    sharp_table=float(np.nanmean([s[c]["table"] for s in sharp_rows]))) for c in CONDS}
    noise = float(np.mean([abs(r["real_real"]["min_height_cm"] - r["real_real_rep"]["min_height_cm"]) for r in rows]))
    json.dump(dict(scene=a.scene, episode=a.episode, blur_sigma_near_px=a.blur_A, crf=a.crf, policy=a.policy,
                   rows=rows, summary=summ, rerun_noise_cm=noise, status="exploratory, UNVERIFIED"),
              open(os.path.join(a.out, "g6.json"), "w"), indent=1)
    cv2.imwrite(os.path.join(a.out, "wrist_conditions.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"re-run noise (real vs real, same input): {noise:.2f} cm")
    for c, v in summ.items():
        print(f"SUMMARY {c:10s} lowest planned fingertip {v['min_height_cm_mean']:+.1f} cm  tilt {v['mean_tilt_deg_mean']:.1f}  sharpness outline {v['sharp_outline']:.1f} table {v['sharp_table']:.1f}")


if __name__ == "__main__":
    main()
