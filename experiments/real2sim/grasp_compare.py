"""Grasp-approach comparison (user request 10-19): the real operator vs GR00T-DROID's predictions on the real frames
(G1/test A chunks) vs GR00T-DROID closed loop in the rebuilt sim (slot A) vs the scripted expert, aligned on the
gripper close (or, for an episode that never closes, its first sustained push into the table).

Per series, vs time (s) relative to that event: fingertip height above the table (g_pinch, robot frame; table =
the rebuilt scene's, checked against stereo to 0.4 cm), fingertip speed, gripper tilt (link7 z axis vs straight
down), gripper command. Writes <out>/grasp_compare.json and <out>/grasp_compare.mp4 (2x2 grid: exterior | wrist per
condition, same time window). Everything is computed from saved files: no policy, no GPU.

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/grasp_compare.py --out runs/real2sim/policy/grasp_compare
"""
import argparse, json, os, sys
import numpy as np
import cv2
import h5py
import imageio.v2 as imageio
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from composite import Compositor, WristCompositor, load_plate, wrist_plate, cam_K, cam_pose  # noqa: E402

FPS = 15
SCENE = "runs/real2sim/ep_orange_block_stereo"
EP = "data/real2sim/droid/ep_orange_block"
BATCH = "runs/real2sim/policy/batch_orange_stereo"
EXPERT = "runs/real2sim/tasks_stereo/ep_orange_block_stereo/01_put_in_bowl_orange_block/v0_states.npz"
REALDATA = "runs/real2sim/policy/realdata_orange_full/chunks_0.npz"
PRE, POST = 4.0, 2.0  # seconds before / after the event
TILE = (320, 180)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = mujoco.MjModel.from_binary_path(os.path.join(SCENE, "scene.mjb"))
    fit = json.load(open(os.path.join(SCENE, "scene_fit.json")))
    tz = fit["table"]["z"]
    arm = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    l7, ps = m.body("fr3_link7").id, m.site("g_pinch").id
    drv = m.joint("g_right_driver_joint")
    drv_q, drv_max = drv.qposadr[0], m.jnt_range[drv.id][1]
    dk = mujoco.MjData(m)

    def fk(q_arm, base=None):
        dk.qpos[:] = m.qpos0 if base is None else base
        dk.qpos[arm] = q_arm
        mujoco.mj_kinematics(m, dk)
        zax = dk.xmat[l7].reshape(3, 3)[:, 2]
        tilt = float(np.degrees(np.arccos(np.clip(-zax[2], -1, 1))))  # 0 = gripper pointing straight down
        return dk.site_xpos[ps].copy(), tilt

    def series(name, label, t, pinch, tilt, grip, kind, extra=None):
        p = np.asarray(pinch)
        sp = np.r_[np.nan, np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(np.diff(t), 1e-9)] if len(p) > 1 else np.array([np.nan])
        s = dict(name=name, label=label, kind=kind, t=np.round(t, 4).tolist(), height_cm=np.round((p[:, 2] - tz) * 100, 2).tolist(),
                 speed_cms=np.round(sp * 100, 2).tolist(), tilt_deg=np.round(tilt, 2).tolist(), grip=np.round(grip, 3).tolist())
        if extra:
            s.update(extra)
        win = (np.asarray(t) >= -PRE) & (np.asarray(t) <= POST)
        s["summary"] = dict(min_height_cm=float(np.nanmin(np.asarray(s["height_cm"])[win])) if win.any() else None,
                            height_at_event_cm=float(np.interp(0, t, p[:, 2] - tz) * 100),
                            mean_speed_last_1s_cms=float(np.nanmean(sp[(np.asarray(t) > -1) & (np.asarray(t) <= 0)]) * 100) if ((np.asarray(t) > -1) & (np.asarray(t) <= 0)).any() else None,
                            tilt_at_event_deg=float(np.interp(0, t, tilt)))
        return s

    out_series, events = [], {}
    # 1. real operator
    h = h5py.File(os.path.join(EP, "trajectory.h5"))
    J, G = h["observation/robot_state/joint_positions"][:], h["observation/robot_state/gripper_position"][:]
    kc = int(np.argmax(G > 0.5))
    events["real"] = kc
    P, T = zip(*[fk(J[k]) for k in range(len(J))])
    t = (np.arange(len(J)) - kc) / FPS
    out_series.append(series("real", "Real operator (recorded)", t, P, T, G, "actual", dict(event="gripper close", event_frame=kc)))
    # 2. GR00T predictions on the real frames: each query's next 3 steps (queries are 3 frames apart)
    z = np.load(REALDATA)
    tt, pp, ti, gg = [], [], [], []
    for k, c in zip(z["frames"], z["chunks"]):
        for j in range(3):
            p, tl = fk(c[j, 10:17])
            tt.append((k + 1 + j - kc) / FPS)
            pp.append(p)
            ti.append(tl)
            gg.append(float(c[j, 9]))
    o = np.argsort(tt)
    out_series.append(series("g1", "GR00T on real frames (predicted)", np.array(tt)[o], np.array(pp)[o], np.array(ti)[o], np.array(gg)[o], "predicted",
                             dict(event="operator's gripper close")))
    # 3./4. GR00T closed loop in sim (slot A)
    picks = [("sim_grasp", "GR00T in sim: grasps (orange v0)", "01_put_in_bowl_orange_block/v0", "close"),
             ("sim_push", "GR00T in sim: pushes into table (orange v3)", "01_put_in_bowl_orange_block/v3", "push"),
             ("sim_success", "GR00T in sim: success (green v0)", "00_put_in_bowl_green_block/v0", "close")]
    for name, label, rel, ev in picks:
        r = json.load(open(os.path.join(BATCH, rel, "rollout.json")))
        n = np.load(os.path.join(BATCH, rel, "rollout.npz"))
        k_ev = r["stages"]["first_step"]["closed_at_target"] if ev == "close" else r["tracking"]["blocked_segments"][0]["start"]
        events[name] = k_ev
        st = n["state"]
        tl = [fk(st[k, 10:17])[1] for k in range(len(st))]
        t = (np.arange(len(st)) - k_ev) / FPS
        s = series(name, label, t, n["pinch_ach"], tl, n["grip_cmd"].astype(float), "actual",
                   dict(event="gripper close at the block" if ev == "close" else "first sustained push into the table", event_step=int(k_ev),
                        commanded_height_cm=np.round((n["pinch_cmd"][:, 2] - tz) * 100, 2).tolist(), success=r["success"]))
        out_series.append(s)
    # 5. scripted expert
    e = np.load(EXPERT)
    q = e["qpos"]
    ge = q[:, drv_q] / drv_max
    k_ev = int(np.argmax(ge > 0.3))
    events["expert"] = k_ev
    P, T = zip(*[fk(q[k, arm], q[k]) for k in range(len(q))])
    out_series.append(series("expert", "Scripted expert (orange v0)", (np.arange(len(q)) - k_ev) / FPS, P, T, ge, "actual",
                             dict(event="gripper starts closing", event_step=k_ev)))
    json.dump(dict(scene=SCENE, table_z=tz, window_s=[-PRE, POST], series=out_series, status="exploratory, UNVERIFIED"),
              open(os.path.join(a.out, "grasp_compare.json"), "w"))

    # video: 2x2 grid, each tile = exterior | wrist at 320x180, window [-PRE, POST] around each event
    n_fr = int((PRE + POST) * FPS) + 1
    W, H = TILE
    tiles = {}
    # real: exterior with GR00T's predicted fingertip path (next 15 steps of the latest query) + the operator's fingertip
    cid = m.camera("ext").id
    dd = mujoco.MjData(m)
    mujoco.mj_forward(m, dd)
    K = cam_K(m, cid, W, H)
    Rc, tc = cam_pose(dd, cid)
    proj = lambda p: tuple(np.round(((p - tc) @ Rc)[:2] / ((p - tc) @ Rc)[2] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]).astype(int))  # noqa: E731
    capx, capw = cv2.VideoCapture(os.path.join(EP, "22008760.mp4")), cv2.VideoCapture(os.path.join(EP, "18026681.mp4"))
    rf = []
    for i in range(n_fr):
        k = kc - int(PRE * FPS) + i
        imgs = []
        for cap in (capx, capw):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(k, 0))
            ok, b = cap.read()
            imgs.append(cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA))
        qi = int(np.searchsorted(z["frames"], k, side="right")) - 1
        if qi >= 0:
            path = [proj(fk(z["chunks"][qi][j, 10:17])[0]) for j in range(15)]
            for p0, p1 in zip(path[:-1], path[1:]):
                cv2.line(imgs[0], p0, p1, (235, 104, 52), 2)
        cv2.circle(imgs[0], proj(fk(J[max(k, 0)])[0]), 3, (42, 120, 214), -1)
        rf.append(np.concatenate(imgs, 1))
    tiles["Real operator; orange line = GR00T's predicted fingertip path"] = rf
    for name, label, rel, ev in picks[:2]:
        frames = [f for f in imageio.get_reader(os.path.join(BATCH, rel, "rollout.mp4"))]
        k0 = events[name] - int(PRE * FPS)
        tiles[label] = [frames[min(max(k0 + i, 0), len(frames) - 1)] for i in range(n_fr)]
    # expert: composite exterior + textured wrist from its states
    plate = load_plate(SCENE, "lama", "frame0+plate")
    comp = Compositor(m, SCENE, plate, size=(H, W), verbose=False)
    wc = WristCompositor(m, plate, comp, (H, W), plate2=wrist_plate(m, SCENE, EP, comp, fit["wrist_camera"]["serial"]))
    de = mujoco.MjData(m)
    ef = []
    for i in range(n_fr):
        de.qpos[:] = q[min(max(events["expert"] - int(PRE * FPS) + i, 0), len(q) - 1)]
        mujoco.mj_forward(m, de)
        ef.append(np.concatenate([comp(de), wc(de)], 1).clip(0, 255).astype(np.uint8))
    tiles["Scripted expert (orange v0)"] = ef
    grid = []
    names = list(tiles)
    for i in range(n_fr):
        cells = []
        for nm in names:
            im = np.ascontiguousarray(tiles[nm][i])
            cv2.putText(im, nm[:60], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(im, f"t = {i / FPS - PRE:+.1f} s", (4, H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(im)
        grid.append(np.concatenate([np.concatenate(cells[:2], 1), np.concatenate(cells[2:], 1)], 0))
    w = imageio.get_writer(os.path.join(a.out, "grasp_compare.mp4"), fps=FPS / 2, quality=8, macro_block_size=8)  # half speed
    for g in grid:
        w.append_data(g)
    w.close()
    print("events:", events)
    for s in out_series:
        print(s["name"], json.dumps(s["summary"]))


if __name__ == "__main__":
    main()
