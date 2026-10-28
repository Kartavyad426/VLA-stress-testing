"""TaskGen MuJoCo layer: core.sim.Sim implemented on MuJoCo 3 (MjSpec editing, then compile). Exploratory, UNVERIFIED."""
import json, os
import numpy as np
import mujoco

from core.sim import Sim
from sim_mujoco.base import load_spec, compile_scene
from sim_mujoco import build


class MujocoSim(Sim):
    name = "mujoco"

    def __init__(self):
        self.scene_dir = self.spec = self.m = self.d = None
        self.queue, self.info, self._home = [], {}, []

    # --- building ---
    def load_base(self, scene_dir):
        self.scene_dir = scene_dir
        self.spec = load_spec(scene_dir)
        self.fit = json.load(open(os.path.join(scene_dir, "scene_fit.json")))
        self.table_z = float(self.fit["table"]["z"])
        self._home = [(k.name, list(k.qpos), list(k.ctrl)) for k in self.spec.keys]
        self.nq_base = compile_scene(load_spec(scene_dir), scene_dir).nq
        return dict(table_z=self.table_z, base_objects={lab: o["kind"] for lab, o in self.fit["objects"].items()})

    def add(self, obj):
        self.queue.append(obj)

    def build(self):
        if self.queue:  # keyframes must grow with the new bodies' qpos: drop, compile once, re-add extended
            for k in list(self.spec.keys):
                self.spec.delete(k)
        for obj in self.queue:
            self.info[obj.name] = build.add(self.spec, obj, self.table_z)
        if self.queue:
            tail = self.spec.compile().qpos0[self.nq_base:].tolist()
            for name, qpos, ctrl in self._home:
                self.spec.add_key(name=name, qpos=qpos + tail, ctrl=ctrl)
        self.queue = []
        self.m = compile_scene(self.spec, self.scene_dir)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        return self.info

    @classmethod
    def from_model(cls, m):
        """Wrap an already compiled model (e.g. res31's scene.mjb) for checks and renders; no editing."""
        s = cls()
        s.m, s.d = m, mujoco.MjData(m)
        mujoco.mj_forward(m, s.d)
        return s

    # --- state ---
    def set_robot(self, q_arm):
        m, d = self.m, self.d
        arm_q = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
        arm_u = [m.actuator(f"fr3_joint{k}").id for k in range(1, 8)]
        d.qpos[arm_q] = q_arm
        d.ctrl[arm_u] = q_arm
        mujoco.mj_forward(m, d)

    def free_objects(self):
        m = self.m
        return [m.body(m.jnt_bodyid[j]).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]

    def position(self, name):
        return self.d.xpos[self.m.body(name).id].copy()

    def joint(self, name):
        return float(self.d.qpos[self.m.jnt_qposadr[self.m.joint(name).id]])

    # --- dynamics and checks ---
    def settle(self, seconds):
        p0 = {n: self.position(n) for n in self.free_objects()}
        for _ in range(int(seconds / self.m.opt.timestep)):
            mujoco.mj_step(self.m, self.d)
        return {n: float(np.linalg.norm(self.position(n) - p)) for n, p in p0.items()}

    def push_joint(self, name, force, seconds):
        dof = self.m.jnt_dofadr[self.m.joint(name).id]
        for _ in range(int(seconds / self.m.opt.timestep)):
            self.d.qfrc_applied[dof] = force
            mujoco.mj_step(self.m, self.d)
        self.d.qfrc_applied[dof] = 0.0

    def overlaps(self, tol=-0.001):
        """Contacts deeper than -tol between geoms of two different top-level bodies at the current state. Resting
        contacts sit at ~0. Visual-only geoms and parent/child pairs never produce contacts in MuJoCo, so they
        can't show here."""
        m, d = self.m, self.d
        mujoco.mj_forward(m, d)
        rows = {}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.body_rootid[m.geom_bodyid[c.geom1]], m.body_rootid[m.geom_bodyid[c.geom2]]
            if b1 == b2 or c.dist >= tol:
                continue
            key = tuple(sorted((m.body(b1).name or "world", m.body(b2).name or "world")))
            rows[key] = min(rows.get(key, 0.0), float(c.dist))
        return [dict(pair=list(k), depth_cm=round(-100 * v, 3)) for k, v in sorted(rows.items(), key=lambda kv: kv[1])]

    def warnings(self):
        out = {}
        for n, w in mujoco.mjtWarning.__members__.items():
            if n != "mjNWARNING" and self.d.warning[int(w)].number:
                out[n] = int(self.d.warning[int(w)].number)
        return out

    # --- output ---
    def render(self, view, height=360, width=640):
        r = mujoco.Renderer(self.m, height, width)
        if view == "free":
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0.62, 0.0, -0.08]
            cam.distance, cam.azimuth, cam.elevation = 1.25, 205.0, -35.0
            r.update_scene(self.d, camera=cam)
        else:
            r.update_scene(self.d, camera=view)
        img = r.render()
        r.close()
        return img

    def save(self, path):
        mujoco.mj_saveModel(self.m, path, None)
        if self.spec is not None:
            open(os.path.splitext(path)[0] + ".xml", "w").write(self.spec.to_xml())  # for reading (absolute paths)
