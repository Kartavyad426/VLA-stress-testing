"""S3 (10-20): what is the "real light/shadow" that G6 found worth ~2.2 cm of GR00T's planned grasp height?

For wrist frames of the real episode (sim posed at the recorded joints), compare the REAL frame's table brightness
with our shadow-free render (wrist-video table texture, no sim shadow) at 640x360, on table pixels away from the
gripper and objects. The log-ratio map log(real/render) (both low-pass filtered) is split, in order, into
  (a) exposure: one gain per frame (median over table pixels)
  (b) vignetting: a fixed radial pattern 1 + a r^2 + b r^4 (r = distance from the image centre / half-diagonal),
      fitted over all frames
  (c) local: what's left; compared with where our sim says the robot's shadow falls (sim shadow ratio < 0.9 with
      the fitted light), and summarised as dark regions (darkness, area, position relative to the gripper)
reporting the share of the log-ratio variance each part explains. Analysis only: renders nothing for the policy.

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/light_analysis.py \
        --scene runs/real2sim/ep_orange_block_stereo --episode data/real2sim/droid/ep_orange_block \
        --out runs/real2sim/ep_orange_block_stereo/light_analysis
"""
import argparse, json, os, sys
import numpy as np
import cv2
import h5py
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from composite import Compositor, WristCompositor, load_plate, wrist_plate, wrist_table_texture  # noqa: E402

H, W = 360, 640


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="0:73:3", help="start:stop:step (default: the approach, before any object moves)")
    ap.add_argument("--margin", type=int, default=61, help="px exclusion around sim objects (real objects sit slightly elsewhere)")
    ap.add_argument("--grip_margin", type=int, default=9, help="px exclusion around the sim robot (narrow: its shadow is next to it)")
    ap.add_argument("--black", type=float, default=50.0, help="real pixels darker than this are objects (black bowl, gripper), not shadow")
    ap.add_argument("--sigma", type=float, default=4.0, help="low-pass (px at 640x360) before the ratio")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
    fit = json.load(open(os.path.join(a.scene, "scene_fit.json")))
    ws = fit["wrist_camera"]["serial"]
    h = h5py.File(os.path.join(a.episode, "trajectory.h5"))
    J, G = h["observation/robot_state/joint_positions"][:], h["observation/robot_state/gripper_position"][:]
    k_close = int(np.argmax(G > 0.5))
    plate = load_plate(a.scene, "lama", "frame0+plate")
    comp = Compositor(m, a.scene, plate, size=(H, W), verbose=False)
    comp_ns = Compositor(m, a.scene, plate, shadows=False, size=(H, W), verbose=False)
    wp = wrist_plate(m, a.scene, a.episode, comp, ws)
    tt = wrist_table_texture(m, a.scene, a.episode, ws, list(range(0, k_close - 2, 2)))
    w_ns = WristCompositor(m, plate, comp_ns, (H, W), plate2=wp, table_tex=tt)
    w_sh = WristCompositor(m, plate, comp, (H, W), plate2=wp, table_tex=tt)
    arm = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    drv_max = m.jnt_range[m.joint("g_right_driver_joint").id][1]
    tbl = m.geom("table").id
    robot_g = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(("fr3_", "g_"))]
    d = mujoco.MjData(m)
    cap = cv2.VideoCapture(os.path.join(a.episode, f"{ws}.mp4"))
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)) / np.sqrt(2)
    lo, hi_, st = (int(x) for x in a.frames.split(":"))
    recs, diag = [], []
    for k in range(lo, min(hi_, len(J)), st):
        d.qpos[:] = m.qpos0
        d.qpos[arm] = J[k]
        for jn in ("g_right_driver_joint", "g_left_driver_joint"):
            d.qpos[m.joint(jn).qposadr[0]] = G[k] * drv_max
        mujoco.mj_forward(m, d)
        sim, info = w_ns(d, return_info=True)
        shaded = w_sh(d)
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, b = cap.read()
        if not ok:
            continue
        real = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        isg = info["seg"][..., 1] == mujoco.mjtObj.mjOBJ_GEOM
        table = isg & (info["seg"][..., 0] == tbl)
        robot_px = isg & np.isin(info["seg"][..., 0], robot_g)
        objs_px = isg & ~table & ~robot_px
        # 10-20: the first version used one wide margin for everything; it hid the gripper's own shadow and still let
        # the real black bowl (bigger in the real view than ours) count as dark table
        other = (cv2.dilate(objs_px.astype(np.uint8), np.ones((a.margin, a.margin), np.uint8)) > 0) | \
                (cv2.dilate(robot_px.astype(np.uint8), np.ones((a.grip_margin, a.grip_margin), np.uint8)) > 0)
        realblack = cv2.dilate((cv2.GaussianBlur(real.mean(-1), (0, 0), 2) < a.black).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        mask = table & ~other & ~realblack
        # gripper-relative position of the dark region: distance from the sim gripper silhouette (px at 640x360)
        dist_to_grip = cv2.distanceTransform((~robot_px).astype(np.uint8), cv2.DIST_L2, 3)
        if mask.sum() < 5000:
            continue
        lr = cv2.GaussianBlur(real.mean(-1), (0, 0), a.sigma)
        ls = cv2.GaussianBlur(sim.astype(np.float32).mean(-1), (0, 0), a.sigma)
        lsh = cv2.GaussianBlur(shaded.astype(np.float32).mean(-1), (0, 0), a.sigma)
        logr = np.log(np.maximum(lr, 1) / np.maximum(ls, 1))
        simshadow = (lsh / np.maximum(ls, 1)) < 0.9  # where our fitted light puts the robot's shadow
        # where are the real dark regions relative to our sim shadow? (frame-level, before the global decomposition)
        gk = float(np.median(logr[mask]))
        darkreal = mask & ((logr - gk) < np.log(0.85))
        simsh_m = mask & simshadow
        cen = lambda mk: np.argwhere(mk).mean(0)[::-1].tolist() if mk.sum() > 200 else None  # noqa: E731
        recs.append(dict(k=k, grip=float(G[k]), logr=logr[mask], rr=rr[mask], sh=simshadow[mask],
                         lr_mean=float(lr[mask].mean()), dark_frac=float(darkreal.sum() / mask.sum()),
                         dark_centroid=cen(darkreal), simshadow_centroid=cen(simsh_m),
                         dark_in_simshadow=float((darkreal & simsh_m).sum() / max(darkreal.sum(), 1)),
                         dark_depth=float(np.exp(np.median((logr - gk)[darkreal]))) if darkreal.sum() > 200 else None,
                         dark_dist_to_gripper_px=float(np.median(dist_to_grip[darkreal])) if darkreal.sum() > 200 else None,
                         near_grip_ratio=float(np.exp(np.median((logr - gk)[mask & (dist_to_grip < 40)]))) if (mask & (dist_to_grip < 40)).sum() > 200 else None,
                         far_grip_ratio=float(np.exp(np.median((logr - gk)[mask & (dist_to_grip > 120)]))) if (mask & (dist_to_grip > 120)).sum() > 200 else None))
        if k in (0, 30, 45, 63, 72):
            diag.append((k, real.astype(np.uint8), sim.clip(0, 255).astype(np.uint8), logr, mask, simshadow))
    # (a) exposure
    tot = np.concatenate([r["logr"] for r in recs])
    var_tot = float(tot.var())
    for r in recs:
        r["gain"] = float(np.median(r["logr"]))
    res_a = np.concatenate([r["logr"] - r["gain"] for r in recs])
    # (b) vignette: least squares on residual vs r^2, r^4 (no constant: the gains absorbed it)
    R2 = np.concatenate([r["rr"] ** 2 for r in recs])
    A = np.stack([R2, R2 ** 2], 1)
    coef, *_ = np.linalg.lstsq(A, res_a, rcond=None)
    res_b = res_a - A @ coef
    # (c) local vs sim shadow
    SH = np.concatenate([r["sh"] for r in recs])
    shadow_mean = float(res_b[SH].mean()) if SH.any() else float("nan")
    noshadow_mean = float(res_b[~SH].mean())
    res_c = res_b - np.where(SH, shadow_mean, noshadow_mean)
    dark = res_b < np.log(0.85)
    out = dict(frames=[r["k"] for r in recs], n_pixels=int(len(tot)),
               var_logratio_total=var_tot,
               explained=dict(exposure=1 - float(res_a.var()) / var_tot,
                              vignette=float(res_a.var() - res_b.var()) / var_tot,
                              sim_shadow_location=float(res_b.var() - res_c.var()) / var_tot,
                              unexplained=float(res_c.var()) / var_tot),
               exposure_gain_per_frame={str(r["k"]): float(np.exp(r["gain"])) for r in recs},
               exposure_gain_range=[float(np.exp(min(r["gain"] for r in recs))), float(np.exp(max(r["gain"] for r in recs)))],
               vignette_coef_r2_r4=coef.tolist(), vignette_at_corner=float(np.exp(coef[0] + coef[1])),
               local=dict(real_vs_render_inside_sim_shadow=float(np.exp(shadow_mean)), real_vs_render_outside=float(np.exp(noshadow_mean)),
                          sim_shadow_frac=float(SH.mean()), real_dark_frac=float(dark.mean()),
                          real_dark_inside_sim_shadow=float((dark & SH).sum() / max(dark.sum(), 1))),
               per_frame_dark=[dict(k=r["k"], dark_frac=r["dark_frac"], dark_depth=r["dark_depth"], dark_centroid_px=r["dark_centroid"],
                                    dark_dist_to_gripper_px=r["dark_dist_to_gripper_px"], near_grip_ratio=r["near_grip_ratio"], far_grip_ratio=r["far_grip_ratio"],
                                    simshadow_centroid_px=r["simshadow_centroid"], dark_in_simshadow=r["dark_in_simshadow"]) for r in recs],
               status="exploratory, UNVERIFIED")
    json.dump(out, open(os.path.join(a.out, "light_analysis.json"), "w"), indent=1)
    tiles = []
    for k, real, sim, logr, mask, simshadow in diag:
        vis = np.zeros((H, W, 3), np.uint8)
        v = np.clip((logr + 0.6) / 1.2, 0, 1)
        vis[..., 0] = (v * 255).astype(np.uint8)
        vis[..., 2] = ((1 - v) * 255).astype(np.uint8)
        vis[~mask] = 40
        cs, _ = cv2.findContours(simshadow.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cs, -1, (255, 255, 255), 1)
        for im, lab in ((real, f"real f{k}"), (sim, "render (no sim shadow)"), (vis, "log(real/render): red brighter, blue darker; white = sim shadow")):
            cv2.putText(im, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([real, sim, vis], 1))
    cv2.imwrite(os.path.join(a.out, "light_analysis.jpg"), cv2.cvtColor(np.concatenate(tiles, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(json.dumps({k: v for k, v in out.items() if k not in ("exposure_gain_per_frame", "frames", "per_frame_dark")}, indent=1))
    for r in out["per_frame_dark"]:
        f = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
        print(f"f{r['k']:3d}: real dark {r['dark_frac']:.2f} of table (depth {f(r['dark_depth'])}), {f(r['dark_dist_to_gripper_px'])} px from the gripper; real/render within 40 px of the gripper {f(r['near_grip_ratio'])}, beyond 120 px {f(r['far_grip_ratio'])}; overlap with sim shadow {r['dark_in_simshadow']:.2f}")


if __name__ == "__main__":
    main()
