"""S4 (10-20): fit the light direction from the WRIST view. S3 found the real arm/gripper shadow on the table falls
on the opposite side of the gripper from our sim shadow (light fitted in the exterior view, IoU 0.445).

For each approach frame (sim posed at the recorded joints): the real dark mask = table pixels >= --dark darker than
our shadow-free wrist render after the per-frame exposure gain (as light_analysis.py); for each candidate directional
light the sim shadow mask = table pixels where (render with robot) / (render without robot) < 0.9. Score = mean IoU
over frames that have real shadow. Grid over azimuth/elevation, then the shadow darkness = median real ratio inside
the best overlap. Writes <scene>/light_wrist.json (dir, az, el, real_shadow_ratio, iou) + a check image.
Convention: dir is the light's travel direction (MuJoCo light_dir) = -(cos el cos az, cos el sin az, sin el).

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/light_fit_wrist.py \
        --scene runs/real2sim/ep_orange_block_stereo --episode data/real2sim/droid/ep_orange_block
"""
import argparse, json, os, sys
import numpy as np
import cv2
import h5py
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from episode_info import frames_arg  # noqa: E402
from composite import Compositor, WristCompositor, load_plate, wrist_plate, wrist_table_texture, HIDE_GROUP  # noqa: E402

H, W = 180, 320


def light_dir(az, el):
    az, el = np.radians(az), np.radians(el)
    return -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--frames", default=None, help="approach frames lo:hi:step or a,b,c (ep_orange_block: 18:73:3); derived from the episode (episode_info.py) when omitted")
    ap.add_argument("--dark", type=float, default=0.85)
    ap.add_argument("--margin", type=int, default=31, help="px at 320x180 around sim objects")
    ap.add_argument("--grip_margin", type=int, default=5, help="px at 320x180 around the sim robot (narrow)")
    ap.add_argument("--black", type=float, default=50.0, help="real pixels darker than this are objects, not shadow")
    a = ap.parse_args()
    m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
    fit = json.load(open(os.path.join(a.scene, "scene_fit.json")))
    ws = fit["wrist_camera"]["serial"]
    h = h5py.File(os.path.join(a.episode, "trajectory.h5"))
    J, G = h["observation/robot_state/joint_positions"][:], h["observation/robot_state/gripper_position"][:]
    k_close = int(np.argmax(G > 0.5))
    plate = load_plate(a.scene, "lama", "frame0+plate")
    comp_ns = Compositor(m, a.scene, plate, shadows=False, size=(H, W), verbose=False)
    wp = wrist_plate(m, a.scene, a.episode, comp_ns, ws)
    tt = wrist_table_texture(m, a.scene, a.episode, ws, list(range(0, k_close - 2, 2)))
    w_ns = WristCompositor(m, plate, comp_ns, (H, W), plate2=wp, table_tex=tt)
    arm = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    drv_max = m.jnt_range[m.joint("g_right_driver_joint").id][1]
    tbl = m.geom("table").id
    robot = np.array([g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(("fr3_", "g_"))])
    cid = m.camera("wrist").id
    m.light_type[0] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    m.light_castshadow[0] = 1
    rend = mujoco.Renderer(m, H, W)
    opt_all, opt_hide = mujoco.MjvOption(), mujoco.MjvOption()
    opt_hide.geomgroup[HIDE_GROUP] = 0
    group0 = m.geom_group.copy()
    d = mujoco.MjData(m)
    cap = cv2.VideoCapture(os.path.join(a.episode, f"{ws}.mp4"))
    frames = []
    for k in frames_arg(a.frames, a.episode, "approach"):
        d.qpos[:] = m.qpos0
        d.qpos[arm] = J[k]
        for jn in ("g_right_driver_joint", "g_left_driver_joint"):
            d.qpos[m.joint(jn).qposadr[0]] = G[k] * drv_max
        mujoco.mj_forward(m, d)
        sim, info = w_ns(d, return_info=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, b = cap.read()
        real = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        isg = info["seg"][..., 1] == mujoco.mjtObj.mjOBJ_GEOM
        table = isg & (info["seg"][..., 0] == tbl)
        rob = isg & np.isin(info["seg"][..., 0], robot)
        objs = isg & ~table & ~rob
        # object-aware masks (10-20, as light_analysis.py): wide margin around objects, narrow around the robot,
        # and real near-black pixels (the real bowl is bigger in the real view) excluded
        realblack = cv2.dilate((cv2.GaussianBlur(real.mean(-1), (0, 0), 1) < a.black).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        mask = table & ~(cv2.dilate(objs.astype(np.uint8), np.ones((a.margin, a.margin), np.uint8)) > 0) \
            & ~(cv2.dilate(rob.astype(np.uint8), np.ones((a.grip_margin, a.grip_margin), np.uint8)) > 0) & ~realblack
        lr = cv2.GaussianBlur(real.mean(-1), (0, 0), 2)
        ls = cv2.GaussianBlur(sim.astype(np.float32).mean(-1), (0, 0), 2)
        logr = np.log(np.maximum(lr, 1) / np.maximum(ls, 1))
        logr -= np.median(logr[mask])
        dark = mask & (logr < np.log(a.dark))
        if dark.sum() < 50:
            continue
        frames.append(dict(k=k, qpos=d.qpos.copy(), mask=mask, dark=dark, ratio=np.exp(logr), real=real.astype(np.uint8)))

    def sim_shadow(qpos, ldir):
        m.light_dir[0] = ldir
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        rend.update_scene(d, camera=cid, scene_option=opt_all)
        lit = rend.render().astype(np.float32).mean(-1)
        m.geom_group[robot] = HIDE_GROUP
        rend.update_scene(d, camera=cid, scene_option=opt_hide)
        bare = rend.render().astype(np.float32).mean(-1)
        m.geom_group[:] = group0
        return lit / np.maximum(bare, 1.0) < 0.9

    def score(az, el):
        ld = light_dir(az, el)
        ious = []
        for f in frames:
            s = sim_shadow(f["qpos"], ld) & f["mask"]
            ious.append((s & f["dark"]).sum() / max((s | f["dark"]).sum(), 1))
        return float(np.mean(ious))

    ext = json.load(open(os.path.join(a.scene, "light.json"))) if os.path.exists(os.path.join(a.scene, "light.json")) else {}
    grid = [(az, el) for az in range(0, 360, 15) for el in range(30, 90, 5)]
    scores = {(az, el): score(az, el) for az, el in grid}
    best = max(scores, key=scores.get)
    # refine around the best
    ref = [(best[0] + da, best[1] + de) for da in (-10, -5, 0, 5, 10) for de in (-4, -2, 0, 2, 4) if 20 <= best[1] + de <= 89]
    for c in ref:
        scores[c] = score(*c)
    best = max(scores, key=scores.get)
    ld = light_dir(*best)
    ins = np.concatenate([f["ratio"][sim_shadow(f["qpos"], ld) & f["mask"] & f["dark"]] for f in frames])
    ratio = float(np.median(ins)) if len(ins) else None
    ext_score = score(ext["azimuth_deg"], ext["elevation_deg"]) if "azimuth_deg" in ext else None
    out = dict(dir=ld.tolist(), azimuth_deg=best[0], elevation_deg=best[1], iou=scores[best], real_shadow_ratio=ratio,
               frames=[f["k"] for f in frames], exterior_fit=dict(azimuth_deg=ext.get("azimuth_deg"), elevation_deg=ext.get("elevation_deg"),
                                                                   iou_in_wrist_view=ext_score),
               top5=[dict(az=k[0], el=k[1], iou=v) for k, v in sorted(scores.items(), key=lambda x: -x[1])[:5]],
               convention="dir = -(cos el cos az, cos el sin az, sin el), MuJoCo light_dir", status="exploratory, UNVERIFIED")
    json.dump(out, open(os.path.join(a.scene, "light_wrist.json"), "w"), indent=1)
    tiles = []
    for f in frames[::max(1, len(frames) // 4)][:4]:
        vis = f["real"].copy()
        for mk, col in ((f["dark"], (0, 255, 255)), (sim_shadow(f["qpos"], ld) & f["mask"], (255, 0, 255))):
            cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cs, -1, col, 1)
        cv2.putText(vis, f"f{f['k']} cyan real dark, magenta sim shadow", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        tiles.append(vis)
    cv2.imwrite(os.path.join(a.scene, "light_wrist_check.jpg"), cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
