"""TaskGen MuJoCo layer: motion primitives for the sequential task expert (docs/TASKGEN_DESIGN.html §7), run in ONE
episode on a built MujocoSim. The framework (core/experts.py) decides the steps and their target points; this file
only moves the arm. Reuses res31's scripted_grasp.ik (damped least squares on the pinch site); copies implres31's
expert_probe ideas (top-down grasp across the narrower side, re-aim on the measured fingertip error, its timings).

  pick(name)              above -> descend -> re-aim -> close -> lift; reports lifted / held
  move_release(point)     up to a safe height -> above point -> lower -> open -> retreat
Frames (exterior camera, 320x180, 15 fps) are kept for a review video when record=True. Exploratory, UNVERIFIED.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot

FPS = 15
T = dict(approach=2.0, descend=1.5, close=1.0, lift=1.5, carry=2.5, lower=1.0, open=0.8, retreat=1.2, correct=0.5)
APPROACH_ABOVE = 0.12
LIFT = 0.15
GRASP_MIN_ABOVE_FLOOR = 0.012
CORRECT_ITERS, CORRECT_TOL = 3, 0.002
Z_DOWN = np.array([0, 0, -1.0])


class Expert:
    def __init__(self, sim, record=False):
        from scripted_grasp import ik
        self.ik, self.s, self.m, self.d = ik, sim, sim.m, sim.d
        m = self.m
        self.arm_q = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
        self.arm_dof = [m.joint(f"fr3_joint{k}").dofadr[0] for k in range(1, 8)]
        self.arm_u = [m.actuator(f"fr3_joint{k}").id for k in range(1, 8)]
        self.grip_u = [i for i in range(m.nu) if m.actuator(i).name.startswith("g_")][0]
        self.g_open, self.g_closed = m.actuator_ctrlrange[self.grip_u]
        self.pinch = m.site("g_pinch").id
        self.n_sub = int(round(1 / FPS / m.opt.timestep))
        self.R = None
        self.frames = [] if record else None
        self.rend = mujoco.Renderer(m, 180, 320) if record else None
        self.q = self.d.qpos[self.arm_q].copy()
        self.grip = self.g_open
        self.on_frame = None  # optional callback after each 1/FPS step (policy_runner: composite frames + stage measurements)

    def _go(self, p, secs, grip=None):
        q, err = self.ik(self.m, self.q, self.pinch, np.asarray(p, float), self.R, self.arm_q, self.arm_dof)
        g0, g1 = self.grip, self.grip if grip is None else grip
        n = max(1, int(secs * FPS))
        for k in range(n):
            a = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / n)
            self.d.ctrl[self.arm_u] = self.q + (q - self.q) * a
            self.d.ctrl[self.grip_u] = g0 + (g1 - g0) * a
            for _ in range(self.n_sub):
                mujoco.mj_step(self.m, self.d)
            if self.frames is not None:
                self.rend.update_scene(self.d, camera="ext")
                self.frames.append(self.rend.render())
            if self.on_frame is not None:
                self.on_frame(self.q + (q - self.q) * a, g0 + (g1 - g0) * a)
        self.q, self.grip = q, g1
        return err

    def _fingertip(self):
        return self.d.site_xpos[self.pinch].copy()

    def _grasp_frame(self, name):
        """Body-frame extents of the object's collision points; close across the narrower horizontal side."""
        b = self.m.body(name).id
        Rb = self.d.xmat[b].reshape(3, 3)
        pts = self.s.collision_points(name)
        loc = (pts - self.d.xpos[b]) @ Rb
        ext = loc.max(0) - loc.min(0)
        ax = Rb[:, 0] if ext[0] <= ext[1] else Rb[:, 1]
        ax = np.array([ax[0], ax[1], 0.0])
        ax /= np.linalg.norm(ax)
        self.R = np.c_[np.cross(ax, Z_DOWN), ax, Z_DOWN]
        lo, hi = pts.min(0), pts.max(0)
        return (lo + hi) / 2, lo, hi, float(min(ext[0], ext[1]))

    def _choose_yaw(self, p_g):
        """Try the grasp rotated by 0/90/180/270 deg about the vertical (same closing line for 0/180); prefer fingers
        closing along world y (sideways, parallel to a cabinet face), keep the first with IK residual < 1 cm above
        and at the grasp point. 10-29: one fixed yaw left IK at 0.82 m on the middle drawer (arm folded)."""
        R0, cands = self.R, []
        for k in range(4):
            a = k * np.pi / 2
            Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            R = Rz @ R0
            cands.append((-abs(R[1, 1]), k, R))  # R[:, 1] = closing axis; prefer it along world y
        tried = []
        for _, k, R in sorted(cands, key=lambda c: (c[0], c[1])):
            _, e1 = self.ik(self.m, self.q, self.pinch, p_g + [0, 0, APPROACH_ABOVE], R, self.arm_q, self.arm_dof)
            _, e2 = self.ik(self.m, self.q, self.pinch, p_g, R, self.arm_q, self.arm_dof)
            tried.append((k * 90, round(max(e1, e2), 4)))
            if max(e1, e2) < 0.01:
                self.R = R
                return k * 90, tried
        self.R = R0
        return None, tried

    def _attempt(self, name, p_g, reaim_z):
        log = {}
        log["ik_above_m"] = self._go(p_g + [0, 0, APPROACH_ABOVE], T["approach"], self.g_open)
        log["ik_descend_m"] = self._go(p_g, T["descend"])
        cmd = p_g.copy()
        for _ in range(CORRECT_ITERS):  # re-aim on the measured fingertip error (implres31's approach_correct)
            e = p_g - self._fingertip()
            if not reaim_z:
                e[2] = 0.0
            if np.linalg.norm(e) < CORRECT_TOL:
                break
            cmd = cmd + e
            self._go(cmd, T["correct"])
        log["aim_err_cm"] = round(100 * float(np.linalg.norm(p_g - self._fingertip())), 2)
        self._go(cmd, T["close"], self.g_closed)
        self._go(cmd + [0, 0, LIFT], T["lift"])
        return log, cmd

    def pick(self, name):
        """Up to two attempts: re-aim horizontally, then (if not held) re-aim in 3D. 10-29: on the middle drawer the 3D
        re-aim pushed the pads into the floor and the close jammed; on the bottom drawer only the 3D re-aim held: the
        grasp is marginal there, so the expert tries both and records which one worked."""
        c, lo, hi, width = self._grasp_frame(name)
        b = self.m.body(name).id
        z0 = float(self.d.xpos[b][2])
        p_g = np.r_[c[:2], max(c[2], lo[2] + GRASP_MIN_ABOVE_FLOOR)]
        log = dict(step="pick", object=name, width_cm=round(100 * width, 1), attempts=[])
        log["yaw_deg"], log["yaw_tried"] = self._choose_yaw(p_g)
        for reaim_z in (False, True):
            a, cmd = self._attempt(name, p_g, reaim_z)
            lifted = float(self.d.xpos[b][2] - z0)
            near = float(np.linalg.norm(self.d.xpos[b][:2] - self._fingertip()[:2]))
            a.update(reaim="3d" if reaim_z else "xy", lifted_cm=round(100 * lifted, 1), held=bool(lifted > 0.08 and near < 0.03))
            log["attempts"].append(a)
            if a["held"]:
                break
            self._go(cmd + [0, 0, LIFT], 0.8, self.g_open)  # let go, back up, settle before the next attempt
            self.settle(0.5)
            z0 = float(self.d.xpos[b][2])
            c, lo, hi, _ = self._grasp_frame(name)
            p_g = np.r_[c[:2], max(c[2], lo[2] + GRASP_MIN_ABOVE_FLOOR)]
        last = log["attempts"][-1]
        log.update(lifted_cm=last["lifted_cm"], held=last["held"], aim_err_cm=last["aim_err_cm"], n_attempts=len(log["attempts"]))
        return log

    def move_release(self, point, label="release"):
        p = np.asarray(point, float)
        safe = max(self._fingertip()[2], p[2]) + 0.10
        self._go(np.r_[self._fingertip()[:2], safe], 1.0)
        log = dict(step=label, at_cm=[round(100 * float(v), 1) for v in p])
        log["ik_above_m"] = self._go(np.r_[p[:2], safe], T["carry"])
        log["ik_lower_m"] = self._go(p, T["lower"])
        self._go(p, T["open"], self.g_open)
        self._go(p + [0, 0, 0.10], T["retreat"])
        return log

    def settle(self, secs=1.0):
        for _ in range(int(secs / self.m.opt.timestep)):
            mujoco.mj_step(self.m, self.d)

    def save_video(self, path):
        if self.frames:
            import imageio.v2 as imageio
            w = imageio.get_writer(path, fps=FPS, quality=7, macro_block_size=8)
            for f in self.frames:
                w.append_data(f)
            w.close()
