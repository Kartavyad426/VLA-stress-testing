"""B3 general scripted-expert test, with the B4 mass and B5 friction sweeps (REAL2SIM_CHECKLIST_V1 §6; implres31, 10-27).

Independent of the episode's task: every graspable object in a built scene is put through the same probes, each from a
fresh start (objects as built, arm at the episode's frame-0 joints, gripper open, 0.5 s settle):
  pick_lift_hold  above -> down to the object -> close -> lift 10 cm -> hold 2 s
  place_upright   pick, lift, carry 10 cm to the free spot farthest from the other objects, lower, open, retreat
  into_container  pick, lift, release above each container (one result per container)
  stack           blocks only (fitted boxes), when there are two or more: onto the nearest other block
Each probe reports the stage it reached (reached / closed / lifted / held / placed) and why it stopped, plus slip and
rotation of the object in the hand during the lift and hold. Objects wider than the gripper's opening along both
horizontal axes are "not graspable by design", not failures. Sweeps (B4/B5, robustness, not truth): the whole probe set
again with mass x0.5 / x2 and friction x0.5 / x2 of the probed object (friction: and the pads), changed on the loaded
model in memory only, followed by mj_setConst (nothing written to the scene; the 10-23 .mjb patch broke collisions, this
does not touch geometry). A probe during which MuJoCo hits NaN/huge accelerations and resets the state is labelled
UNSTABLE, not failed: found 10-27 on orange, where the blocks (9-23 g) flip between held and a reset under a 1% mass
change.

The expert is make_tasks' (res31): damped-least-squares IK on the pinch site (scripted_grasp.ik), cosine-interpolated
joint targets through the position actuators at 15 Hz, physics on. Two changes, both recorded in the output: the grasp
point is the centre of the object's collision bounding box (not the body origin, which a pack mesh may not have at its
centre), clamped to at least grasp_min_above_table_m above the table; the gripper closes across the object's narrower
horizontal axis (make_tasks always uses the object's x axis, which is the same for cubes). All numbers in
expert_probe_config.json. A failure is the scene's or the expert's: the video tells which.

res31's modules are imported read-only from the main checkout (--real2sim; they are uncommitted there).
Writes <out>/expert_probe.json and <out>/expert_probe/<object>__<probe>[__<container>].mp4 (video sweeps only).
Exploratory, UNVERIFIED.

    MUJOCO_GL=osmesa /home/imerit/Documents/Code/VLA/.venvs/real2sim/bin/python experiments/real2sim_checks/expert_probe.py \
        --scene /home/imerit/Documents/Code/VLA/runs/real2sim/ep_orange_block_stereo \
        --episode /home/imerit/Documents/Code/VLA/data/real2sim/droid/ep_orange_block
"""
import argparse, json, os, sys, time
os.environ.setdefault("MUJOCO_GL", "osmesa")  # CPU rendering; the GPU stays free
import numpy as np
import h5py
import mujoco
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = "/home/imerit/Documents/Code/VLA"
R2S = next((sys.argv[i + 1] for i, x in enumerate(sys.argv[:-1]) if x == "--real2sim"), os.path.join(MAIN, "experiments/real2sim"))
sys.path.insert(0, R2S)
from scripted_grasp import ik  # noqa: E402
from scene_roles import container_bodies, container_fit, body_kinds  # noqa: E402
from make_tasks import task_success, FPS  # noqa: E402

DOWN = np.array([0.0, 0.0, -1.0])
SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


class Scene:
    """A built scene loaded fresh for one sweep, with the sweep's mass/friction applied in memory."""

    def __init__(self, scene, q_start, sweep, cfg, target=None):
        """target: body name the sweep applies to (with the pads for friction); None = every non-container free body."""
        self.m = m = mujoco.MjModel.from_binary_path(os.path.join(scene, "scene.mjb"))
        self.fit = json.load(open(os.path.join(scene, "scene_fit.json")))
        self.q_start = q_start
        self.arm_q = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
        self.arm_dof = [m.joint(f"fr3_joint{k}").dofadr[0] for k in range(1, 8)]
        self.arm_u = [m.actuator(f"fr3_joint{k}").id for k in range(1, 8)]
        self.grip_u = [i for i in range(m.nu) if m.actuator(i).name.startswith("g_")][0]
        self.g_open, self.g_closed = m.actuator_ctrlrange[self.grip_u]
        self.pinch = m.site("g_pinch").id
        self.n_sub = int(round(1 / FPS / m.opt.timestep))
        self.free = [m.jnt_bodyid[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.containers = container_bodies(m, self.fit)
        self.kinds = body_kinds(self.fit)
        coll = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])
        self.geoms = {b: [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == b and coll(g)] for b in self.free}
        pad_g = [g for g in range(m.ngeom) if "pad" in m.body(m.geom_bodyid[g]).name and coll(g)]
        self.pads = {s: [g for g in pad_g if s in m.body(m.geom_bodyid[g]).name] for s in ("left", "right")}
        # sweep, in memory only, on the probed object (10-27: scaling every block let one block's instability reset the
        # whole simulation during another block's probe)
        swept = [b for b in self.free if b not in self.containers and (target is None or m.body(b).name == target)]
        if "mass" in sweep:
            k = sweep["mass"]
            for b in range(m.nbody):
                if m.body_rootid[b] in swept:
                    m.body_mass[b] *= k
                    m.body_inertia[b] *= k
        if "friction" in sweep:
            k = sweep["friction"]
            gs = [g for b in swept for g in self.geoms[b]]
            if cfg["friction_scope"] == "object_and_pads":
                gs += pad_g
            for g in gs:
                m.geom_friction[g, 0] *= k
        if sweep:  # derived constants (body_invweight0 etc.) follow the new masses; without this the solver used stale ones
            mujoco.mj_setConst(m, mujoco.MjData(m))
        self.pad_obj_friction = {m.body(b).name: float(max(m.geom_friction[self.geoms[b], 0].max(), m.geom_friction[pad_g, 0].max()))
                                 for b in self.free if b not in self.containers}

    def reset(self, settle_s):
        m, d = self.m, mujoco.MjData(self.m)
        d.qpos[self.arm_q] = self.q_start
        d.ctrl[self.arm_u] = self.q_start
        d.ctrl[self.grip_u] = self.g_open
        for _ in range(int(settle_s / m.opt.timestep)):
            mujoco.mj_step(m, d)
        return d

    def box(self, d, b):
        """Collision bounding box of body b's subtree: world z range, world centre, and widths along the body's own
        x and y axes (for the grasp direction)."""
        pts = []
        for g in self.geoms[b]:
            c, h = self.m.geom_aabb[g][:3], self.m.geom_aabb[g][3:]
            pts.append(d.geom_xpos[g] + (c + h * SIGNS) @ d.geom_xmat[g].reshape(3, 3).T)
        pts = np.concatenate(pts)
        Rb = d.xmat[b].reshape(3, 3)
        loc = (pts - d.xpos[b]) @ Rb
        return dict(zmin=float(pts[:, 2].min()), zmax=float(pts[:, 2].max()), centre=(pts.min(0) + pts.max(0)) / 2,
                    wx=float(np.ptp(loc[:, 0])), wy=float(np.ptp(loc[:, 1])), ax_x=Rb[:, 0], ax_y=Rb[:, 1])

    def closing_axis(self, d):
        """Index of the pinch-site axis along which the fingers close (from the pad positions)."""
        Rs = d.site_xmat[self.pinch].reshape(3, 3)
        cl = lambda s: np.mean([d.geom_xpos[g] for g in self.pads[s]], 0)
        v = Rs.T @ (cl("left") - cl("right"))
        return int(np.argmax(np.abs(v)))

    def grip_rotation(self, d, close_dir):
        """Pinch-site orientation: z down, fingers closing along close_dir (horizontal)."""
        c = np.array([close_dir[0], close_dir[1], 0.0])
        c /= np.linalg.norm(c) + 1e-9
        ci = self.closing_axis(d)
        if ci == 1:
            return np.c_[np.cross(c, DOWN), c, DOWN]
        if ci == 0:
            return np.c_[c, np.cross(DOWN, c), DOWN]
        raise RuntimeError("fingers close along the approach axis: site frame not understood")

    def in_contact(self, d, b):
        """(left pad touches b, right pad touches b)."""
        og, hit = set(self.geoms[b]), {"left": False, "right": False}
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            for s in hit:
                if (g1 in self.pads[s] and g2 in og) or (g2 in self.pads[s] and g1 in og):
                    hit[s] = True
        return hit["left"], hit["right"]


class Runner:
    """Drives waypoints like make_tasks' expert and records the object's pose in the hand."""

    def __init__(self, sc, d, rend, cam):
        self.sc, self.d, self.rend, self.cam = sc, d, rend, cam
        self.frames, self.ik_err = [], 0.0
        self.unstable_at = None  # sim time of the first MuJoCo instability reset (time jumps back to 0)

    def _step(self):
        t = self.d.time
        mujoco.mj_step(self.sc.m, self.d)
        if self.d.time < t and self.unstable_at is None:
            self.unstable_at = round(float(t), 3)

    def go(self, waypoints, R):
        sc, m, d = self.sc, self.sc.m, self.d
        q_prev = d.qpos[sc.arm_q].copy()
        for p, g, secs in waypoints:
            q, err = ik(m, q_prev, sc.pinch, np.asarray(p, float), R, sc.arm_q, sc.arm_dof)
            self.ik_err = max(self.ik_err, err)
            n = max(int(secs * FPS), 1)
            for k in range(n):
                s = (k + 1) / n
                d.ctrl[sc.arm_u] = q_prev + (q - q_prev) * (0.5 - 0.5 * np.cos(np.pi * s))
                d.ctrl[sc.grip_u] = g
                for _ in range(sc.n_sub):
                    self._step()
                if self.rend is not None:
                    self.rend.update_scene(d, camera=self.cam)
                    self.frames.append(self.rend.render().copy())
            q_prev = q

    def settle(self, secs):
        for _ in range(int(secs / self.sc.m.opt.timestep)):
            self._step()
        if self.rend is not None:
            self.rend.update_scene(self.d, camera=self.cam)
            self.frames.append(self.rend.render().copy())

    def in_hand(self, b):
        d = self.d
        Rs, ps = d.site_xmat[self.sc.pinch].reshape(3, 3), d.site_xpos[self.sc.pinch]
        return Rs.T @ (d.xpos[b] - ps), Rs.T @ d.xmat[b].reshape(3, 3)


def free_spot(sc, d, b, centre, cfg):
    """The point place_offset_m from the object's centre, on the table (with margin), farthest from every other object."""
    t = sc.fit["table"]
    lo = np.array(t["xy_lo"]) + cfg["table_margin_m"]
    hi = np.array(t["xy_hi"]) - cfg["table_margin_m"]
    cands = [centre[:2] + cfg["place_offset_m"] * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    inside = [c for c in cands if np.all(c > lo) and np.all(c < hi)]
    others = [sc.box(d, o)["centre"][:2] for o in sc.free if o != b]
    return max(inside or cands, key=lambda c: min([np.linalg.norm(c - o) for o in others] + [1.0])), bool(inside)


def probe(sc, b, kind, cfg, rend, cam, ref=None):
    """One probe on body b. kind: pick_lift_hold | place_upright | into_container | stack. ref: container / lower block."""
    T, P = cfg["timing_s"], cfg["pass"]
    m = sc.m
    d = sc.reset(T["start_settle"])
    r = Runner(sc, d, rend, cam)
    bx = sc.box(d, b)
    table_z = sc.fit["table"]["z"]
    narrow_x = bx["wx"] <= bx["wy"]
    width = min(bx["wx"], bx["wy"])
    R = sc.grip_rotation(d, bx["ax_x"] if narrow_x else bx["ax_y"])
    gz = max(float(bx["centre"][2]), table_z + cfg["grasp_min_above_table_m"])
    p_g = np.r_[bx["centre"][:2], gz]
    up0 = d.xmat[b].reshape(3, 3)[:, 2].copy()
    out = dict(object=m.body(b).name, probe=kind, ref=None if ref is None else m.body(ref).name, width_across_grasp_m=width,
               grasp_point=p_g.tolist(), stage="start", passed=False)
    # pick
    r.go([(p_g + [0, 0, cfg["approach_above_m"]], sc.g_open, T["approach"]), (p_g, sc.g_open, T["descend"])], R)
    # 10-27: the position actuators stop ~0.8 cm short of the IK target (orange: 0.76 cm along the closing axis,
    # 0.68 cm low), so one pad hit the block first and shoved it (popped 3 cm up at nominal friction, squirted 4.5 cm
    # out at x0.5). Re-aim on the measured fingertip error before closing, so B3 tests the scene, not this tracking error.
    out["reach_err_before_correction_m"] = float(np.linalg.norm(d.site_xpos[sc.pinch] - p_g))
    aim = p_g.copy()
    for _ in range(cfg.get("approach_correct_iters", 3)):
        e = p_g - d.site_xpos[sc.pinch]
        if np.linalg.norm(e) < cfg.get("approach_correct_tol_m", 0.002):
            break
        aim = aim + e
        r.go([(aim, sc.g_open, cfg.get("approach_correct_s", 0.5))], R)
    out["reach_err_m"] = float(np.linalg.norm(d.site_xpos[sc.pinch] - p_g))
    out["stage"] = "reached"
    r.go([(p_g, sc.g_closed, T["close"])], R)
    left, right = sc.in_contact(d, b)
    out["pads_on_object_after_close"] = [left, right]
    if not (left and right):
        out["stopped"] = "not both pads on the object after closing"
    else:
        out["stage"] = "closed"
    p0, R0 = r.in_hand(b)
    z0 = float(d.xpos[b][2])
    p_up = p_g + [0, 0, cfg["lift_m"]]
    r.go([(p_up, sc.g_closed, T["lift"])], R)
    rise = float(d.xpos[b][2] - z0)
    out["rise_after_lift_m"] = rise
    if out["stage"] == "closed":
        if rise >= P["lift_min_m"]:
            out["stage"] = "lifted"
        else:
            out["stopped"] = f"rose {rise*100:.1f} cm < {P['lift_min_m']*100:.0f} cm"
    if kind == "pick_lift_hold":
        r.go([(p_up, sc.g_closed, cfg["hold_s"])], R)
    p1, R1 = r.in_hand(b)
    out["slip_in_hand_m"] = float(np.linalg.norm(p1 - p0))
    out["rotation_in_hand_deg"] = float(np.degrees(Rot.from_matrix(R0.T @ R1).magnitude()))
    out["rise_after_hold_m"] = float(d.xpos[b][2] - z0)
    if kind == "pick_lift_hold":
        if out["stage"] == "lifted":
            if out["rise_after_hold_m"] >= P["lift_min_m"] and out["slip_in_hand_m"] <= P["slip_max_m"]:
                out["stage"], out["passed"] = "held", True
            else:
                out["stopped"] = f"during hold: rise {out['rise_after_hold_m']*100:.1f} cm, slip {out['slip_in_hand_m']*100:.1f} cm"
        return finish(out, r)
    # carry and release
    above = gz - bx["zmin"]  # pinch height above the object's bottom
    if kind == "place_upright":
        spot, on_table = free_spot(sc, d, b, bx["centre"], cfg)
        out.update(spot=spot.tolist(), spot_on_table=on_table)
        drop = np.r_[spot, bx["zmin"] + above + 0.005]
    elif kind == "into_container":
        cb = sc.box(d, ref)
        cf = container_fit(sc.fit, m.body(ref).name) or {}
        rim = cf.get("rim_radius") or min(cb["wx"], cb["wy"]) / 2
        out.update(container_top_z=cb["zmax"], rim_radius_m=rim)
        drop = np.r_[cb["centre"][:2], cb["zmax"] + above + 0.02]
    else:  # stack
        rb = sc.box(d, ref)
        drop = np.r_[rb["centre"][:2], rb["zmax"] + above + 0.005]
    r.go([(np.r_[drop[:2], max(drop[2], p_up[2]) + 0.04], sc.g_closed, T["carry"]), (drop, sc.g_closed, T["lower"]),
          (drop, sc.g_open, T["open"]), (drop + [0, 0, 0.15], sc.g_open, T["retreat"])], R)
    r.settle(T["settle"])
    fb = sc.box(d, b)
    up1 = d.xmat[b].reshape(3, 3)[:, 2]
    out["tilt_deg"] = float(np.degrees(np.arccos(np.clip(up0 @ up1, -1, 1))))
    out["final_centre"] = fb["centre"].tolist()
    if out["stage"] != "lifted":
        return finish(out, r)
    if kind == "place_upright":
        err_xy = float(np.linalg.norm(fb["centre"][:2] - drop[:2]))
        err_z = float(fb["zmin"] - bx["zmin"])
        out.update(place_xy_err_m=err_xy, place_z_err_m=err_z)
        ok = err_xy <= P["place_xy_tol_m"] and abs(err_z) <= P["place_z_tol_m"] and out["tilt_deg"] <= P["upright_max_deg"]
        why = f"xy {err_xy*100:.1f} cm, z {err_z*100:+.1f} cm, tilt {out['tilt_deg']:.0f} deg"
    elif kind == "into_container":
        cb1 = sc.box(d, ref)
        horiz = float(np.linalg.norm(fb["centre"][:2] - cb1["centre"][:2]))
        ok = horiz < P["container_rim_frac"] * out["rim_radius_m"] and fb["zmin"] < cb1["zmax"]
        out.update(container_horiz_m=horiz, bottom_below_rim=bool(fb["zmin"] < cb1["zmax"]))
        why = f"{horiz*100:.1f} cm from the container centre (rim {out['rim_radius_m']*100:.1f} cm), bottom {'below' if fb['zmin'] < cb1['zmax'] else 'above'} the rim"
    else:
        ok = task_success(m, d, sc.fit, dict(type="stack", target=m.body(b).name, ref=m.body(ref).name))
        out["stack_horiz_m"] = float(np.linalg.norm(fb["centre"][:2] - sc.box(d, ref)["centre"][:2]))
        why = f"{out['stack_horiz_m']*100:.1f} cm off the lower block's centre"
    out["result"] = why
    if ok:
        out["stage"], out["passed"] = "placed", True
    else:
        out["stopped"] = "after release: " + why
    return finish(out, r)


def finish(out, r):
    out["ik_err_max_m"] = r.ik_err
    out["unstable_at_s"] = r.unstable_at
    if r.unstable_at is not None:  # MuJoCo reset the state mid-probe: nothing after it is a measurement
        out.update(passed=False, stage="UNSTABLE", stopped=f"MuJoCo instability reset at t={r.unstable_at} s (NaN/huge qacc or qvel)")
    out["_frames"] = r.frames
    return out


def write_video(path, frames):
    import imageio.v2 as imageio
    w = imageio.get_writer(path, fps=FPS, quality=7, macro_block_size=4)
    for f in frames:
        w.append_data(f)
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", default=None, help="default: <main checkout>/runs/real2sim_checks/<scene name>")
    ap.add_argument("--config", default=os.path.join(HERE, "expert_probe_config.json"))
    ap.add_argument("--sweeps", default=None, help="comma-separated sweep names from the config (default: all)")
    ap.add_argument("--objects", default=None, help="comma-separated body names (default: every non-container free body)")
    ap.add_argument("--probes", default=None, help="comma-separated probe names (default: the config's list)")
    ap.add_argument("--real2sim", default=R2S, help="res31's experiments/real2sim, imported read-only")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    out_dir = a.out or os.path.join(MAIN, "runs/real2sim_checks", os.path.basename(os.path.normpath(a.scene)))
    vid_dir = os.path.join(out_dir, "expert_probe")
    os.makedirs(vid_dir, exist_ok=True)
    q_start = h5py.File(os.path.join(a.episode, "trajectory.h5"))["observation/robot_state/joint_positions"][0]
    sweeps = a.sweeps.split(",") if a.sweeps else list(cfg["sweeps"])
    probes = a.probes.split(",") if a.probes else cfg["probes"]
    res = dict(scene=a.scene, episode=a.episode, config=cfg, status="exploratory, UNVERIFIED", sweeps={})
    t_all = time.time()
    for sw in sweeps:
        params = cfg["sweeps"][sw]
        sc0 = Scene(a.scene, q_start, {}, cfg)
        d0 = sc0.reset(0.0)
        objs = [b for b in sc0.free if b not in sc0.containers]
        if a.objects:
            objs = [b for b in objs if sc0.m.body(b).name in a.objects.split(",")]
        blocks = [b for b in sc0.free if b not in sc0.containers and sc0.kinds.get(sc0.m.body(b).name) in ("box", "cube")]
        video = sw in cfg["video"]["sweeps"]
        rows = []
        for b in objs:
            name = sc0.m.body(b).name
            sc = Scene(a.scene, q_start, params, cfg, target=name) if params else sc0  # body ids are the same in every load
            bx = sc.box(d0, b)
            if min(bx["wx"], bx["wy"]) > cfg["gripper_max_width_m"]:
                rows.append(dict(object=name, probe="all", passed=None, stage="not graspable by design",
                                 width_across_grasp_m=min(bx["wx"], bx["wy"])))
                continue
            rend = mujoco.Renderer(sc.m, cfg["video"]["height"], cfg["video"]["width"]) if video else None
            jobs = []
            for pr in probes:
                if pr in ("pick_lift_hold", "place_upright"):
                    jobs.append((pr, None))
                elif pr == "into_container":
                    jobs += [(pr, c) for c in sc.containers]
                elif pr == "stack" and b in blocks and len(blocks) >= 2:
                    others = [o for o in blocks if o != b]
                    jobs.append((pr, min(others, key=lambda o: np.linalg.norm(d0.xpos[o][:2] - d0.xpos[b][:2]))))
            for pr, ref in jobs:
                t0 = time.time()
                row = probe(sc, b, pr, cfg, rend, cfg["video"]["camera"], ref)
                frames = row.pop("_frames")
                row["pad_object_friction"] = sc.pad_obj_friction.get(name)
                row["mass_kg"] = float(sum(sc.m.body_mass[x] for x in range(sc.m.nbody) if sc.m.body_rootid[x] == b))
                row["seconds"] = round(time.time() - t0, 1)
                if video and frames:
                    fn = f"{name}__{pr}" + (f"__{sc.m.body(ref).name}" if ref is not None else "") + ".mp4"
                    write_video(os.path.join(vid_dir, fn), frames)
                    row["video"] = os.path.join("expert_probe", fn)
                rows.append(row)
                print(f"[{sw}] {name:>14} {pr:<15} {row['ref'] or '':<14} {'PASS' if row['passed'] else 'fail'}  stage={row['stage']:<8}"
                      f" {row.get('stopped', row.get('result', ''))}  ({row['seconds']} s)", flush=True)
            if rend is not None:
                rend.close()
        res["sweeps"][sw] = dict(params=params, applied_to="the probed object only (+ pads for friction)", rows=rows)
        json.dump(res, open(os.path.join(out_dir, "expert_probe.json"), "w"), indent=1, default=float)
    # one table: object x probe -> pass per sweep
    table = {}
    for sw, v in res["sweeps"].items():
        for r in v["rows"]:
            key = f"{r['object']} | {r['probe']}" + (f" -> {r['ref']}" if r.get("ref") else "")
            table.setdefault(key, {})[sw] = r["stage"] if r["passed"] is None else ("PASS" if r["passed"] else f"fail@{r['stage']}")
    res["table"] = table
    res["seconds_total"] = round(time.time() - t_all, 1)
    json.dump(res, open(os.path.join(out_dir, "expert_probe.json"), "w"), indent=1, default=float)
    print("\n" + " " * 44 + "  ".join(f"{s:>14}" for s in sweeps))
    for k, v in table.items():
        print(f"{k:<44}" + "  ".join(f"{v.get(s, '-'):>14}" for s in sweeps))
    print(f"written {os.path.join(out_dir, 'expert_probe.json')} ({res['seconds_total']} s)")


if __name__ == "__main__":
    main()
