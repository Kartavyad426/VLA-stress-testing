"""Interactive scene inspection (SCENE_V1_MECHANISMS §15/§15a; user 10-27: "have two modes, like have a button for
camera calibration thing or scene changes otherwise user just interacts with it, teleoperate maybe but cant do
anything about the start position").

One window (glfw + MuJoCo's own renderer), three modes picked with the buttons on top (or keys 1/2/3):
  Play (default)  physics runs; teleoperate the gripper; the start position can't be changed (Backspace resets to it).
  Scene edit      physics paused; drag objects with the mouse, move the arm; Enter settles and saves a new start
                  to <scene>/starts/<name>.json (object poses + arm joints).
  Camera calib    the real exterior frame with the sim robot's outline on top; move the camera by keys/mouse while the
                  robot-outline overlap, outline distance and anchor error update live; Enter saves
                  <scene>/calib_manual.json (same format as calib.json) and prints the refine + rebuild commands.

Keys (all modes): 1/2/3 mode · Tab view (free / exterior / wrist; Play and Scene edit) · H help · Esc quit.
Play / Scene edit: W/A/S/D gripper forward/left/back/right relative to the view (hold), Space/Left Shift up/down,
  Q/E rotate the gripper, G open/close (Play only); mouse left-drag orbit, right-drag pan, wheel zoom.
  Scene edit only: left-drag an object slides it across the table under the cursor, wheel while dragging rotates it.
Camera calib: W/S along the line of sight, A/D sideways, R/F up/down, arrow keys yaw/pitch, Q/E roll (hold Ctrl for
  fine steps), [ ] change the shown frame, P DROID's recorded pose, C the scene's calib.json; mouse left-drag
  yaw/pitch, right-drag sideways/up.

Exploratory (res31, 10-27). Needs a display; run without MUJOCO_GL (glfw is used directly):
    .venvs/real2sim/bin/python experiments/real2sim/inspect_scene.py --scene runs/real2sim/ep_orange_block_stereo \
        --episode data/real2sim/droid/ep_orange_block
--selftest renders each mode to <scene>/inspect_selftest_<mode>.png and exits (for checking without a person).
"""
import argparse, json, os, sys, time
import numpy as np
import cv2
import glfw
import h5py
import mujoco
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_cam import set_camera, zed_K, project, DOWN  # noqa: E402
from scripted_grasp import ik  # noqa: E402
from perceive import read_frame  # noqa: E402

MODES = ["play", "scene", "camera"]
LABEL = {"play": "1 Play", "scene": "2 Scene edit", "camera": "3 Camera calib"}
VIEWS = ["free", "ext", "wrist"]
DRAG_GROUP = 5    # unused geom group (scene uses 0, 2, 3)
BAR = 44          # button bar height (framebuffer px)
SPEED = 0.15      # gripper target speed, m/s
TURN = 60.0       # gripper yaw speed, deg/s
CAM_STEP = (0.005, np.deg2rad(0.2))   # camera calib step per key event: 5 mm, 0.2 deg (Ctrl: x0.2)


def cam_pose_droid(f, cam, frame):
    """DROID's recorded exterior pose (same convention as build_scene.cam_pose)."""
    from build_scene import cam_pose
    return cam_pose(f[f"observation/camera_extrinsics/{cam}_left"][frame])


class Inspector:
    def __init__(self, a):
        self.a = a
        self.scene = a.scene
        self.m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.f = h5py.File(os.path.join(a.episode, "trajectory.h5"))
        self.arm_q = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
        self.arm_dof = [m.joint(f"fr3_joint{k}").dofadr[0] for k in range(1, 8)]
        self.arm_u = [m.actuator(f"fr3_joint{k}").id for k in range(1, 8)]
        self.grip_u = [i for i in range(m.nu) if m.actuator(i).name.startswith("g_")][0]
        self.g_lo, self.g_hi = m.actuator_ctrlrange[self.grip_u]
        self.pinch = m.site("g_pinch").id
        self.free_j = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.free_body = {int(m.jnt_bodyid[j]): j for j in self.free_j}
        self.robot_geoms = np.array([g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(("fr3", "g_"))])
        fit = json.load(open(os.path.join(a.scene, "scene_fit.json")))
        self.table_z = fit["table"]["z"]
        self.q_start = self.f["observation/robot_state/joint_positions"][0]
        self.make_start()
        if a.start:
            self.load_start(a.start)
        self.reset()
        self.mode, self.view, self.help = "play", 0, False
        self.msg, self.msg_t = "", 0.0
        self.calib_init()

    # ---------- start state ----------
    def make_start(self):
        """The episode's start: arm at the recorded frame-0 joints, objects where the build put them, settled 0.5 s."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_q] = self.q_start
        d.ctrl[self.arm_u] = self.q_start
        d.ctrl[self.grip_u] = self.g_lo
        for _ in range(int(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        self.start = dict(qpos=d.qpos.copy(), ctrl=d.ctrl.copy(), name="episode start")

    def load_start(self, path):
        s = json.load(open(path))
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.start["qpos"]
        for name, pose in s["objects"].items():
            adr = m.jnt_qposadr[m.body(name).jntadr[0]]
            d.qpos[adr:adr + 7] = pose
        d.qpos[self.arm_q] = s["robot_q"]
        ctrl = self.start["ctrl"].copy()
        ctrl[self.arm_u] = s["robot_q"]
        self.start = dict(qpos=d.qpos.copy(), ctrl=ctrl, name=os.path.basename(path))

    def reset(self):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.start["qpos"]
        d.ctrl[:] = self.start["ctrl"]
        mujoco.mj_forward(m, d)
        self.q_tgt = d.qpos[self.arm_q].copy()
        self.p_tgt = d.site_xpos[self.pinch].copy()
        self.R_tgt = d.site_xmat[self.pinch].reshape(3, 3).copy()
        self.grip_closed = False
        self.ik_err = 0.0
        self.t_wall = time.time()
        self.drag = None

    def object_poses(self):
        return {self.m.body(b).name: self.d.qpos[self.m.jnt_qposadr[j]:self.m.jnt_qposadr[j] + 7].tolist()
                for b, j in self.free_body.items()}

    def save_start(self):
        """Scene edit, Enter: settle 1 s with the arm held, report how far objects moved, save the start."""
        m, d = self.m, self.d
        before = {n: np.array(p[:3]) for n, p in self.object_poses().items()}
        d.ctrl[self.arm_u] = d.qpos[self.arm_q]
        d.ctrl[self.grip_u] = self.g_lo
        d.qvel[:] = 0
        for _ in range(int(1.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        moved = {n: float(np.linalg.norm(np.array(p[:3]) - before[n])) for n, p in self.object_poses().items()}
        os.makedirs(os.path.join(self.scene, "starts"), exist_ok=True)
        name = time.strftime("start_%Y%m%d_%H%M%S")
        path = os.path.join(self.scene, "starts", name + ".json")
        json.dump(dict(objects=self.object_poses(), robot_q=d.qpos[self.arm_q].tolist(), gripper="open",
                       settle_moved_m=moved, source="inspect_scene.py scene edit", episode=self.a.episode,
                       created=time.strftime("%Y-%m-%d %H:%M:%S")), open(path, "w"), indent=1)
        worst = max(moved.items(), key=lambda kv: kv[1]) if moved else ("-", 0)
        warn = f"  WARNING {worst[0]} moved {100 * worst[1]:.1f} cm while settling" if worst[1] > 0.01 else ""
        self.say(f"saved {path}{warn}")
        print("saved start:", path, "settle moved (cm):", {k: round(100 * v, 2) for k, v in moved.items()}, flush=True)
        self.start = dict(qpos=d.qpos.copy(), ctrl=d.ctrl.copy(), name=name)

    def say(self, s):
        self.msg, self.msg_t = s, time.time()

    def safe_forward(self):
        """mj_forward that survives MuJoCo running out of its working memory (huge contact counts)."""
        try:
            mujoco.mj_forward(self.m, self.d)
            return True
        except mujoco.FatalError as e:
            print("mj_forward failed:", str(e).splitlines()[0], flush=True)
            return False

    # ---------- teleop ----------
    def teleop(self, dt, keys, view_forward):
        """Move the pinch target by held keys (relative to the view), IK to joint targets."""
        fwd = np.array([view_forward[0], view_forward[1], 0.0])
        fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
        left = np.array([-fwd[1], fwd[0], 0.0])
        v = np.zeros(3)
        v += fwd * (keys["w"] - keys["s"]) + left * (keys["a"] - keys["d"])
        v[2] += keys["up"] - keys["down"]
        yaw = keys["q"] - keys["e"]
        if not v.any() and not yaw:
            return False
        self.p_tgt = self.p_tgt + v * SPEED * dt
        # the IK knows nothing about contacts: keep the target from sinking into the table (it would keep going
        # down while the real arm is stopped by the table, and take seconds to come back up)
        self.p_tgt[2] = max(self.p_tgt[2], self.table_z + 0.005)
        if yaw:
            self.R_tgt = Rot.from_euler("z", yaw * TURN * dt, degrees=True).as_matrix() @ self.R_tgt
        q, err = ik(self.m, self.q_tgt, self.pinch, self.p_tgt, self.R_tgt, self.arm_q, self.arm_dof, iters=25)
        self.ik_err = err
        if err > 0.01:  # out of reach: hold the last reachable target
            self.p_tgt = self.p_tgt - v * SPEED * dt
        else:
            self.q_tgt = q
        return True

    # ---------- camera calibration ----------
    def calib_init(self):
        a, m = self.a, self.m
        self.cid = m.camera("ext").id
        cpath = os.path.join(self.scene, "calib.json")
        self.calib = json.load(open(cpath)) if os.path.exists(cpath) else None
        self.cam = (self.calib or {}).get("cam") or a.cam
        self.K = np.array(self.calib["K"]) if self.calib and "K" in self.calib else zed_K(a.episode, self.cam)
        self.pmode = (self.calib or {}).get("principal_mode", "offset")
        self.W, self.H = 1280, 720
        fr = (self.calib or {}).get("frames")
        if not fr:
            from episode_info import frames_arg
            fr = frames_arg(None, a.episode, "robot")
        self.frames = list(fr)
        self.R_droid, self.t_droid = cam_pose_droid(self.f, self.cam, 0)
        if self.calib:
            self.R_cal, self.t_cal = np.array(self.calib["R_cam_to_base"]), np.array(self.calib["t_cam_in_base"])
        else:
            self.R_cal, self.t_cal = self.R_droid, self.t_droid
        self.R_c, self.t_c = self.R_cal.copy(), self.t_cal.copy()
        self.fi = len(self.frames) // 2
        self.real = None       # lazy: loaded when camera mode opens
        self.metrics, self.dirty = None, True

    def calib_load_data(self):
        if self.real is not None:
            return
        a = self.a
        vid = os.path.join(a.episode, f"{self.cam}.mp4")
        self.real = {i: read_frame(vid, i) for i in self.frames}
        mp = os.path.join(self.scene, "robot_masks_motion.npz")
        if os.path.exists(mp) and all(str(i) in np.load(mp).files for i in self.frames):
            mm = np.load(mp)
            self.real_mask = {i: mm[str(i)].astype(bool) for i in self.frames}
        else:
            from calibrate_cam import motion_masks
            mm, _ = motion_masks(vid, self.frames)
            self.real_mask = {i: mm[str(i)] for i in self.frames}
        self.real_small = {i: cv2.resize(self.real_mask[i].astype(np.uint8), (self.W // DOWN, self.H // DOWN),
                                         interpolation=cv2.INTER_NEAREST).astype(bool) for i in self.frames}
        self.dt_real = {i: cv2.distanceTransform((~self.real_small[i]).astype(np.uint8), cv2.DIST_L2, 3) for i in self.frames}
        self.jp = self.f["observation/robot_state/joint_positions"]
        # anchors: wrist-stereo 3D object points vs the exterior perception masks (same rule as calibrate_cam)
        self.anchors, self.mask_cent = [], {}
        ap_ = os.path.join(self.scene, "wrist_anchors.json")
        pp_ = os.path.join(self.scene, "perception.npz")
        if os.path.exists(ap_) and os.path.exists(pp_):
            per = np.load(pp_)
            for k in per.files:
                if k.startswith("mask/") and per[k].any():
                    v_, u_ = np.nonzero(per[k])
                    self.mask_cent.setdefault(k[5:].split("#")[0], []).append([u_.mean(), v_.mean()])
            for name, an in json.load(open(ap_))["anchors"].items():
                lab = name.split("#")[0]
                if lab in self.mask_cent:
                    self.anchors.append((lab, np.array(an["point"])))

    def calib_move(self, dx, drot):
        """dx: camera-frame translation (x right, y down, z along the line of sight); drot: camera-frame rotvec."""
        self.t_c = self.t_c + self.R_c @ np.asarray(dx, float)
        self.R_c = self.R_c @ Rot.from_rotvec(drot).as_matrix()
        self.dirty = True

    def calib_metrics(self, render_seg):
        sims, ious, dists = {}, [], []
        for i in self.frames:
            sm = render_seg(i)
            sims[i] = sm
            s_small = cv2.resize(sm.astype(np.uint8), (self.W // DOWN, self.H // DOWN), interpolation=cv2.INTER_NEAREST).astype(bool)
            inter, uni = (sm & self.real_mask[i]).sum(), (sm | self.real_mask[i]).sum()
            ious.append(inter / max(uni, 1))
            if s_small.any():
                dt_sim = cv2.distanceTransform((~s_small).astype(np.uint8), cv2.DIST_L2, 3)
                dists.append(float(self.dt_real[i][s_small].mean()) + float(dt_sim[self.real_small[i]].mean()))
            else:
                dists.append(200.0)
        anc = []
        for lab, p in self.anchors:
            uv = project(self.K, self.R_c, self.t_c, p)[0]
            anc.append(min(np.linalg.norm(uv - np.array(c)) for c in self.mask_cent[lab]))
        dR = np.rad2deg(Rot.from_matrix(self.R_c @ self.R_droid.T).magnitude())
        dRc = np.rad2deg(Rot.from_matrix(self.R_c @ self.R_cal.T).magnitude())
        self.metrics = dict(iou=ious, iou_mean=float(np.mean(ious)), outline_px=float(np.mean(dists)) * DOWN,
                            anchor_px=float(np.mean(anc)) if anc else None, n_anchors=len(anc),
                            d_droid=(dR, 100 * float(np.linalg.norm(self.t_c - self.t_droid))),
                            d_calib=(dRc, 100 * float(np.linalg.norm(self.t_c - self.t_cal))))
        self.sims = sims
        self.dirty = False

    def calib_save(self):
        out = dict(self.calib or {}, cam=self.cam, K=self.K.tolist(), R_cam_to_base=self.R_c.tolist(),
                   t_cam_in_base=self.t_c.tolist(), frames=self.frames, principal_mode=self.pmode,
                   manual=dict(source="inspect_scene.py camera calib", created=time.strftime("%Y-%m-%d %H:%M:%S"),
                               metrics=self.metrics))
        path = os.path.join(self.scene, "calib_manual.json")
        json.dump(out, open(path, "w"), indent=1, default=float)
        ep, sc = self.a.episode, self.scene
        print(f"saved {path}\nrefine from it (CPU ~20 min):\n  MUJOCO_GL=osmesa .venvs/real2sim/bin/python experiments/real2sim/calibrate_cam.py "
              f"--episode {ep} --cam {self.cam} --out {sc} --init {path} [--anchors {sc}/wrist_anchors.json --anchor_weight 1]\n"
              f"then rebuild: build_scene.py --episode {ep} --cam {self.cam} --out {sc} --depth stereo --calib {sc}/calib.json", flush=True)
        self.say(f"saved {path} (refine/rebuild commands printed in the terminal)")


# ---------------- window / rendering ----------------
class App:
    def __init__(self, ins, selftest=False):
        self.ins = ins
        self.selftest = selftest
        if not glfw.init():
            raise SystemExit("glfw init failed (no display?)")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE if selftest else glfw.TRUE)
        self.win = glfw.create_window(1440, 860, f"inspect_scene: {ins.scene}", None, None)
        glfw.make_context_current(self.win)
        glfw.swap_interval(1)
        m = ins.m
        self.scn = mujoco.MjvScene(m, maxgeom=20000)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, self.cam)
        self.cam.lookat[:] = [0.5, 0.0, ins.table_z]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.5, 35.0, -55.0  # behind-left of the robot, above the table
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[DRAG_GROUP] = 1           # the dragged object is moved to this group: still drawn ...
        self.sel_opt = mujoco.MjvOption()
        self.sel_opt.geomgroup[DRAG_GROUP] = 0       # ... but not hit by the cursor, which then lands on the table
        self.pert = mujoco.MjvPerturb()
        self.ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.btn = []
        self.mouse = dict(left=False, right=False, x=0.0, y=0.0)
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_mouse_button_callback(self.win, self.on_button)
        glfw.set_cursor_pos_callback(self.win, self.on_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)

    # --- helpers ---
    def fb_scale(self):
        w, _ = glfw.get_window_size(self.win)
        fw, _ = glfw.get_framebuffer_size(self.win)
        return fw / max(w, 1)

    def viewport(self):
        fw, fh = glfw.get_framebuffer_size(self.win)
        return mujoco.MjrRect(0, 0, fw, fh - BAR)

    def held(self, *keys):
        return float(any(glfw.get_key(self.win, k) == glfw.PRESS for k in keys))

    def set_mode(self, mode):
        ins = self.ins
        if mode == ins.mode:
            return
        if mode == "camera":
            ins.say("loading real frames and robot masks ...")
            ins.calib_load_data()
            ins.dirty = True
        if mode == "play":
            ins.reset()  # Play always starts from the saved start; edits only reach it through Enter in Scene edit
        ins.mode = mode
        ins.say(f"mode: {LABEL[mode]}")

    def view_forward(self):
        ins = self.ins
        if VIEWS[ins.view] == "free":
            az = np.deg2rad(self.cam.azimuth)
            return np.array([np.cos(az), np.sin(az)])
        cid = ins.m.camera(VIEWS[ins.view]).id
        return -ins.d.cam_xmat[cid].reshape(3, 3)[:, 2][:2]

    def select_point(self, x, y, opt):
        """MuJoCo's own picking (matches the drawn pixels): (geom id or -1, 3D point) under window point (x, y)."""
        s = self.fb_scale()
        vp = self.viewport()
        fx, fy = x * s, y * s - BAR
        selpnt = np.zeros((3, 1))
        geomid, flexid, skinid = (np.zeros((1, 1), np.int32) for _ in range(3))
        mujoco.mjv_select(self.ins.m, self.ins.d, opt, vp.width / vp.height, fx / vp.width, 1 - fy / vp.height,
                          self.scn, selpnt, geomid, flexid, skinid)
        return int(geomid[0, 0]), selpnt.ravel()

    def select_body(self, x, y):
        g, pnt = self.select_point(x, y, self.opt)
        if g < 0:
            return None, None
        body = int(self.ins.m.geom_bodyid[g])
        while body and body not in self.ins.free_body and self.ins.m.body_parentid[body] != 0:
            body = int(self.ins.m.body_parentid[body])
        return (body if body in self.ins.free_body else None), pnt

    # --- callbacks ---
    def on_key(self, win, key, sc, action, mods):
        ins = self.ins
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3) and action == glfw.PRESS:
            self.set_mode(MODES[key - glfw.KEY_1])
        elif key == glfw.KEY_H and action == glfw.PRESS:
            ins.help = not ins.help
        elif ins.mode != "camera":
            if key == glfw.KEY_TAB and action == glfw.PRESS:
                ins.view = (ins.view + 1) % len(VIEWS)
            elif key == glfw.KEY_BACKSPACE:
                ins.reset()
                ins.say(f"reset to {ins.start['name']}")
            elif key == glfw.KEY_G and action == glfw.PRESS and ins.mode == "play":
                ins.grip_closed = not ins.grip_closed
                ins.d.ctrl[ins.grip_u] = ins.g_hi if ins.grip_closed else ins.g_lo
            elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER) and action == glfw.PRESS and ins.mode == "scene":
                ins.save_start()
        else:
            k = 0.2 if mods & glfw.MOD_CONTROL else 1.0
            st, rt = CAM_STEP[0] * k, CAM_STEP[1] * k
            mv = {glfw.KEY_W: ([0, 0, st], 0), glfw.KEY_S: ([0, 0, -st], 0), glfw.KEY_D: ([st, 0, 0], 0),
                  glfw.KEY_A: ([-st, 0, 0], 0), glfw.KEY_F: ([0, st, 0], 0), glfw.KEY_R: ([0, -st, 0], 0),
                  glfw.KEY_LEFT: (0, [0, -rt, 0]), glfw.KEY_RIGHT: (0, [0, rt, 0]), glfw.KEY_UP: (0, [-rt, 0, 0]),
                  glfw.KEY_DOWN: (0, [rt, 0, 0]), glfw.KEY_Q: (0, [0, 0, -rt]), glfw.KEY_E: (0, [0, 0, rt])}
            if key in mv:
                dx, dr = mv[key]
                ins.calib_move(dx if dx else [0, 0, 0], dr if dr else [0, 0, 0])
            elif key == glfw.KEY_LEFT_BRACKET:
                ins.fi = (ins.fi - 1) % len(ins.frames)
            elif key == glfw.KEY_RIGHT_BRACKET:
                ins.fi = (ins.fi + 1) % len(ins.frames)
            elif key == glfw.KEY_P and action == glfw.PRESS:
                ins.R_c, ins.t_c, ins.dirty = ins.R_droid.copy(), ins.t_droid.copy(), True
                ins.say("camera: DROID's recorded pose")
            elif key == glfw.KEY_C and action == glfw.PRESS:
                ins.R_c, ins.t_c, ins.dirty = ins.R_cal.copy(), ins.t_cal.copy(), True
                ins.say("camera: the scene's calib.json")
            elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER) and action == glfw.PRESS:
                ins.calib_save()

    def on_button(self, win, button, action, mods):
        x, y = glfw.get_cursor_pos(win)
        self.on_button_at(x, y, button, action)

    def on_button_at(self, x, y, button, action):
        ins, win = self.ins, self.win
        s = self.fb_scale()
        if action == glfw.PRESS and y * s < BAR:
            for rect, mode in self.btn:
                if rect[0] <= x * s <= rect[0] + rect[2]:
                    self.set_mode(mode)
            return
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse["left"] = action == glfw.PRESS
            if action == glfw.PRESS and ins.mode == "scene":  # any view: picking uses the drawn camera
                body, pnt = self.select_body(x, y)
                if body is None:
                    ins.say("no movable object under the cursor" + (" (free view: drag empty space to orbit)" if VIEWS[ins.view] == "free" else ""))
                if body is not None:
                    adr = ins.m.jnt_qposadr[ins.free_body[body]]
                    geoms = np.nonzero(ins.m.geom_bodyid == body)[0]
                    ins.drag = dict(body=body, adr=adr, geoms=geoms, group=ins.m.geom_group[geoms].copy(),
                                    off=None, qpos0=ins.d.qpos[adr:adr + 7].copy())
                    ins.m.geom_group[geoms] = DRAG_GROUP
                    ins.say(f"dragging {ins.m.body(body).name}")
            if action == glfw.RELEASE and ins.drag is not None:
                ins.m.geom_group[ins.drag["geoms"]] = ins.drag["group"]
                if not ins.safe_forward():  # dropped into something MuJoCo can't resolve: put it back
                    adr = ins.drag["adr"]
                    ins.d.qpos[adr:adr + 7] = ins.drag["qpos0"]
                    ins.safe_forward()
                    ins.say("that spot overlaps too much (too many contacts): object put back")
                ins.drag = None
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.mouse["right"] = action == glfw.PRESS
        self.mouse["x"], self.mouse["y"] = x, y

    def on_move(self, win, x, y):
        ins = self.ins
        dx, dy = x - self.mouse["x"], y - self.mouse["y"]
        self.mouse["x"], self.mouse["y"] = x, y
        if not (self.mouse["left"] or self.mouse["right"]):
            return
        if ins.mode == "camera":
            if self.mouse["left"]:
                ins.calib_move([0, 0, 0], [-dy * 0.0006, dx * 0.0006, 0])
            else:
                ins.calib_move([dx * 0.0005, dy * 0.0005, 0], [0, 0, 0])
            return
        if ins.drag is not None:
            g, p = self.select_point(x, y, self.sel_opt)
            if g >= 0:
                adr = ins.drag["adr"]
                if ins.drag["off"] is None:  # first move: keep the grab offset so the object doesn't jump
                    g0, p0 = self.select_point(x - dx, y - dy, self.sel_opt)
                    ins.drag["off"] = ins.d.qpos[adr:adr + 2] - (p0[:2] if g0 >= 0 else p[:2])
                ins.d.qpos[adr:adr + 2] = p[:2] + ins.drag["off"]
                ins.d.qvel[:] = 0
                # kinematics only while dragging: a full forward pass computes contacts, and dragging through a
                # many-part mesh (c1b's cup: 332 contacts) overflowed MuJoCo's arena and crashed (10-28)
                mujoco.mj_kinematics(ins.m, ins.d)
            return
        if VIEWS[ins.view] != "free":
            return
        h = glfw.get_window_size(win)[1]
        act = mujoco.mjtMouse.mjMOUSE_ROTATE_V if self.mouse["left"] else mujoco.mjtMouse.mjMOUSE_MOVE_V
        mujoco.mjv_moveCamera(ins.m, act, dx / h, dy / h, self.cam)

    def on_scroll(self, win, xo, yo):
        ins = self.ins
        if ins.mode == "scene" and ins.drag is not None:
            adr = ins.drag["adr"]
            q = ins.d.qpos[adr + 3:adr + 7]
            r = Rot.from_euler("z", 5 * yo, degrees=True) * Rot.from_quat(q, scalar_first=True)
            ins.d.qpos[adr + 3:adr + 7] = r.as_quat(scalar_first=True)
            mujoco.mj_forward(ins.m, ins.d)
        elif ins.mode != "camera" and VIEWS[ins.view] == "free":
            mujoco.mjv_moveCamera(ins.m, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yo, self.cam)

    # --- per-frame ---
    def step(self):
        ins, m, d = self.ins, self.ins.m, self.ins.d
        now = time.time()
        dt = min(now - ins.t_wall, 0.05)
        ins.t_wall = now
        if ins.mode == "camera":
            return
        keys = dict(w=self.held(glfw.KEY_W), a=self.held(glfw.KEY_A), s=self.held(glfw.KEY_S), d=self.held(glfw.KEY_D),
                    up=self.held(glfw.KEY_SPACE), down=self.held(glfw.KEY_LEFT_SHIFT), q=self.held(glfw.KEY_Q),
                    e=self.held(glfw.KEY_E))
        moved = ins.teleop(dt, keys, self.view_forward())
        if ins.mode == "play":
            d.ctrl[ins.arm_u] = ins.q_tgt
            n = max(1, int(round(dt / m.opt.timestep)))
            try:
                for _ in range(n):
                    mujoco.mj_step(m, d)
            except mujoco.FatalError as e:  # e.g. an object driven deep into a many-part mesh
                ins.reset()
                ins.say("physics overflowed (too many contacts): reset to start")
                print("mj_step failed:", str(e).splitlines()[0], flush=True)
        elif moved:  # scene edit: kinematic arm, no physics; refuse poses that put the arm into the table or objects
            q_old = d.qpos[ins.arm_q].copy()
            d.qpos[ins.arm_q] = ins.q_tgt
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            robot = set(ins.robot_geoms.tolist())
            hit = [c for c in d.contact[:d.ncon] if c.dist < -0.002 and ((c.geom1 in robot) != (c.geom2 in robot))]
            if hit:
                d.qpos[ins.arm_q] = ins.q_tgt = q_old
                mujoco.mj_forward(m, d)
                ins.p_tgt = d.site_xpos[ins.pinch].copy()
                c = hit[0]
                other = c.geom2 if c.geom1 in robot else c.geom1
                ins.say(f"blocked: the arm would go into {m.body(m.geom_bodyid[other]).name or m.geom(other).name}")

    def render_seg_ext(self, frame):
        """Sim robot mask from the exterior camera at the current calib pose and the recorded joints of `frame`."""
        ins, m = self.ins, self.ins.m
        d = mujoco.MjData(m)
        d.qpos[:] = ins.d.qpos
        d.qpos[ins.arm_q] = ins.jp[frame]
        set_camera(m, ins.cid, ins.R_c, ins.t_c, ins.K, ins.W, ins.H, ins.pmode)
        mujoco.mj_forward(m, d)
        cam = mujoco.MjvCamera()
        cam.type, cam.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, ins.cid
        scn = mujoco.MjvScene(m, maxgeom=20000)
        scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
        scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1
        mujoco.mjv_updateScene(m, d, self.opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)
        rect = mujoco.MjrRect(0, 0, ins.W, ins.H)
        mujoco.mjr_render(rect, scn, self.ctx)
        rgb = np.zeros((ins.H, ins.W, 3), np.uint8)
        mujoco.mjr_readPixels(rgb, None, rect, self.ctx)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
        rgb = np.flipud(rgb).astype(np.int32)
        seg = rgb[..., 0] + rgb[..., 1] * 256 + rgb[..., 2] * 65536 - 1
        objid = np.full(scn.ngeom + 1, -1)
        for g in range(scn.ngeom):
            if scn.geoms[g].objtype == mujoco.mjtObj.mjOBJ_GEOM:
                objid[g] = scn.geoms[g].objid
        ids = np.where((seg >= 0) & (seg < scn.ngeom), objid[np.clip(seg, 0, scn.ngeom)], -1)
        return np.isin(ids, ins.robot_geoms)

    def draw_camera_mode(self, vp):
        ins = self.ins
        if ins.dirty:
            ins.calib_metrics(self.render_seg_ext)
        i = ins.frames[ins.fi]
        img = ins.real[i].copy()
        sm, rm = ins.sims[i], ins.real_mask[i]
        cs, _ = cv2.findContours(sm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cr, _ = cv2.findContours(rm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img[sm] = (0.6 * img[sm] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.drawContours(img, cr, -1, (255, 60, 60), 2)
        cv2.drawContours(img, cs, -1, (0, 255, 0), 2)
        for lab, p in ins.anchors:
            uv = project(ins.K, ins.R_c, ins.t_c, p)[0]
            c = min(ins.mask_cent[lab], key=lambda c: np.linalg.norm(uv - np.array(c)))
            cv2.line(img, tuple(int(v) for v in uv), tuple(int(v) for v in c), (255, 255, 0), 2)
            cv2.circle(img, tuple(int(v) for v in uv), 6, (255, 255, 0), 2)
        sc = min(vp.width / ins.W, vp.height / ins.H)
        w, h = int(ins.W * sc), int(ins.H * sc)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        mujoco.mjr_rectangle(vp, 0.1, 0.1, 0.1, 1)
        mujoco.mjr_drawPixels(np.ascontiguousarray(np.flipud(img)).ravel(), None, mujoco.MjrRect(0, vp.height - h, w, h), self.ctx)
        mt = ins.metrics
        left = "\n".join(["frame", "outline overlap (IoU)", "  per frame", "outline distance", "anchor error",
                          "vs DROID recorded", "vs scene calib.json"])
        right = "\n".join([f"{i} ({ins.fi + 1}/{len(ins.frames)})", f"{mt['iou_mean']:.3f}",
                           " ".join(f"{v:.2f}" for v in mt["iou"]), f"{mt['outline_px']:.1f} px",
                           f"{mt['anchor_px']:.1f} px ({mt['n_anchors']})" if mt["anchor_px"] is not None else "no anchors",
                           f"{mt['d_droid'][0]:.1f} deg, {mt['d_droid'][1]:.1f} cm",
                           f"{mt['d_calib'][0]:.1f} deg, {mt['d_calib'][1]:.1f} cm"])
        left += "\ncolours"
        right += "\ngreen sim robot, red real robot, yellow anchor -> mask"
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, vp, left, right, self.ctx)

    def status_text(self):
        ins, m, d = self.ins, self.ins.m, self.ins.d
        p = d.site_xpos[ins.pinch]
        lines = [("start", ins.start["name"]), ("view", VIEWS[ins.view]),
                 ("gripper", f"{'closed' if ins.grip_closed else 'open'}  at {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}")]
        if ins.ik_err > 0.01:
            lines.append(("reach", f"OUT OF REACH ({100 * ins.ik_err:.1f} cm)"))
        st = {n: np.array(v[:3]) for n, v in ins.object_poses().items()}
        s0 = {}
        for b, j in ins.free_body.items():
            adr = m.jnt_qposadr[j]
            s0[m.body(b).name] = ins.start["qpos"][adr:adr + 3]
        for n, pos in st.items():
            h = 100 * (pos[2] - s0[n][2])
            mv = 100 * np.linalg.norm(pos[:2] - s0[n][:2])
            flag = "  LIFTED" if h > 1 else ("  BELOW TABLE" if pos[2] < ins.table_z - 0.005 else "")
            lines.append((n, f"moved {mv:.1f} cm, height {h:+.1f} cm{flag}"))
        pen = [c for c in d.contact[:d.ncon] if c.dist < -0.003]
        if pen:
            names = {tuple(sorted((m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name))) for c in pen}
            lines.append(("overlap", ", ".join(f"{a}/{b}" for a, b in sorted(names))[:90]))
        return "\n".join(k for k, _ in lines), "\n".join(v for _, v in lines)

    def draw_bar(self):
        fw, fh = glfw.get_framebuffer_size(self.win)
        ins = self.ins
        mujoco.mjr_rectangle(mujoco.MjrRect(0, fh - BAR, fw, BAR), 0.15, 0.15, 0.15, 1)
        self.btn, x = [], 8
        for mode in MODES:
            w = 230
            r = mujoco.MjrRect(x, fh - BAR + 6, w, BAR - 12)
            on = ins.mode == mode
            mujoco.mjr_rectangle(r, *(0.55, 0.35, 0.15, 1) if on else (0.3, 0.3, 0.3, 1))
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, r, LABEL[mode], "", self.ctx)
            self.btn.append(((x, 0, w, BAR), mode))
            x += w + 8
        msg = ins.msg if time.time() - ins.msg_t < 6 else "H = help"
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           mujoco.MjrRect(x + 10, fh - BAR + 6, fw - x - 20, BAR - 12), msg, "", self.ctx)

    def draw(self):
        ins = self.ins
        vp = self.viewport()
        if ins.mode == "camera":
            self.draw_camera_mode(vp)
        else:
            cam = self.cam
            if VIEWS[ins.view] != "free":
                cam = mujoco.MjvCamera()
                cam.type, cam.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, ins.m.camera(VIEWS[ins.view]).id
            mujoco.mjv_updateScene(ins.m, ins.d, self.opt, self.pert, cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
            mujoco.mjr_render(vp, self.scn, self.ctx)
            l, r = self.status_text()
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, vp, l, r, self.ctx)
        if ins.help:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, vp, HELP[ins.mode], "", self.ctx)
        self.draw_bar()

    def screenshot(self, path):
        fw, fh = glfw.get_framebuffer_size(self.win)
        rgb = np.zeros((fh, fw, 3), np.uint8)
        mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, fw, fh), self.ctx)
        cv2.imwrite(path, cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR))

    def run(self):
        if self.selftest:
            return self.run_selftest()
        while not glfw.window_should_close(self.win):
            glfw.poll_events()
            self.step()
            self.draw()
            glfw.swap_buffers(self.win)
        glfw.terminate()

    def run_selftest(self):
        """Render each mode once (plus a scripted teleop and object drag) and save screenshots."""
        ins = self.ins
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
        out = {}
        # play: move the gripper 10 cm forward with physics
        for _ in range(60):
            ins.teleop(1 / 60, dict(w=1, a=0, s=0, d=0, up=0, down=0, q=0, e=0), self.view_forward())
            ins.d.ctrl[ins.arm_u] = ins.q_tgt
            for _ in range(int(round(1 / 60 / ins.m.opt.timestep))):
                mujoco.mj_step(ins.m, ins.d)
        ins.help = True
        self.draw()
        p = os.path.join(ins.scene, "inspect_selftest_play.png")
        self.screenshot(p)
        out["play"] = dict(png=p, pinch=ins.d.site_xpos[ins.pinch].tolist(), ik_err=ins.ik_err)
        # the mouse/scroll callbacks as a person would use them: orbit, pan, zoom
        win = self.win
        ww, wh = glfw.get_window_size(win)
        for btn in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT):
            self.mouse["x"], self.mouse["y"] = ww / 2, wh / 2
            self.mouse["left"], self.mouse["right"] = btn == glfw.MOUSE_BUTTON_LEFT, btn == glfw.MOUSE_BUTTON_RIGHT
            self.on_move(win, ww / 2 + 20, wh / 2 + 10)
            self.mouse["left"] = self.mouse["right"] = False
        self.on_scroll(win, 0, 1)
        self.on_scroll(win, 0, -1)
        # scene edit: click the first free object where it is drawn, drag it 40 px, scroll to rotate it
        self.set_mode("scene")
        self.draw()
        b, j = next(iter(ins.free_body.items()))
        adr = ins.m.jnt_qposadr[j]
        p0 = ins.d.qpos[adr:adr + 3].copy()
        ww, wh = glfw.get_window_size(win)
        pts = [(x_, y_) for x_ in range(0, ww, 8) for y_ in range(BAR + 4, wh, 8) if self.select_body(x_, y_)[0] == b]
        cx, cy = pts[len(pts) // 2]
        self.on_button_at(cx, cy, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS)
        picked = ins.drag["body"] if ins.drag else None
        self.on_move(win, cx + 40, cy)
        self.on_scroll(win, 0, 2)
        self.on_button_at(cx + 40, cy, glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE)
        moved_cm = 100 * float(np.linalg.norm(ins.d.qpos[adr:adr + 2] - p0[:2]))
        self.draw()
        p = os.path.join(ins.scene, "inspect_selftest_scene.png")
        self.screenshot(p)
        out["scene"] = dict(png=p, target=ins.m.body(b).name, picked=ins.m.body(picked).name if picked is not None else None,
                            dragged_cm=moved_cm)
        # camera calib: one key step of each kind, then the mouse
        self.set_mode("camera")
        for k in (glfw.KEY_W, glfw.KEY_A, glfw.KEY_R, glfw.KEY_LEFT, glfw.KEY_UP, glfw.KEY_Q, glfw.KEY_RIGHT_BRACKET):
            self.on_key(win, k, 0, glfw.PRESS, 0)
        self.mouse["left"], self.mouse["x"], self.mouse["y"] = True, 100, 100
        self.on_move(win, 110, 105)
        self.mouse["left"] = False
        self.on_key(win, glfw.KEY_C, 0, glfw.PRESS, 0)   # back to the scene calib for the screenshot
        self.on_key(win, glfw.KEY_LEFT_BRACKET, 0, glfw.PRESS, 0)
        self.set_mode("play")
        self.on_key(win, glfw.KEY_TAB, 0, glfw.PRESS, 0)
        self.draw()
        self.on_key(win, glfw.KEY_TAB, 0, glfw.PRESS, 0)
        self.draw()
        self.on_key(win, glfw.KEY_TAB, 0, glfw.PRESS, 0)
        self.on_key(win, glfw.KEY_G, 0, glfw.PRESS, 0)
        self.on_key(win, glfw.KEY_BACKSPACE, 0, glfw.PRESS, 0)
        # camera calib
        self.set_mode("camera")
        self.draw()
        p = os.path.join(ins.scene, "inspect_selftest_camera.png")
        self.screenshot(p)
        out["camera"] = dict(png=p, metrics=ins.metrics)
        print(json.dumps(out, indent=1, default=float))
        glfw.terminate()


HELP = {
    "play": "PLAY  W/A/S/D move gripper (relative to view)  Space/LShift up/down  Q/E rotate  G open/close\n"
            "Tab view  Backspace reset to start  mouse: left orbit, right pan, wheel zoom  1/2/3 modes  Esc quit",
    "scene": "SCENE EDIT  left-drag object = slide on table (free view), wheel while dragging = rotate\n"
             "W/A/S/D Space/LShift Q/E move the arm (no physics)  Enter = settle + save start  Backspace = reset",
    "camera": "CAMERA CALIB  W/S along view  A/D sideways  R/F up/down  arrows yaw/pitch  Q/E roll  Ctrl = fine\n"
              "mouse: left = yaw/pitch, right = slide  [ ] frame  P = DROID pose  C = scene calib  Enter = save calib_manual.json",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="scene dir with scene.mjb, scene_fit.json (calib.json for camera mode)")
    ap.add_argument("--episode", default=None, help="DROID episode dir; default: from <scene>/wrist_anchors.json")
    ap.add_argument("--cam", default="22008760", help="exterior serial when the scene has no calib.json")
    ap.add_argument("--start", default=None, help="a saved start (<scene>/starts/*.json) to open instead of the episode's")
    ap.add_argument("--view", default="free", choices=VIEWS, help="view to open in (Tab cycles)")
    ap.add_argument("--selftest", action="store_true", help="render each mode to PNGs in the scene dir and exit")
    a = ap.parse_args()
    if a.episode is None:
        wa = os.path.join(a.scene, "wrist_anchors.json")
        if not os.path.exists(wa):
            raise SystemExit("--episode is needed (no wrist_anchors.json to read it from)")
        a.episode = json.load(open(wa))["episode"]
    if os.environ.get("MUJOCO_GL") not in (None, "", "glfw"):
        raise SystemExit("run without MUJOCO_GL (this tool opens its own glfw window)")
    ins = Inspector(a)
    ins.view = VIEWS.index(a.view)
    App(ins, selftest=a.selftest).run()


if __name__ == "__main__":
    main()
