"""Retarget a matched demo from a root, execute it in LIBERO, judge it
(docs/DMP_POC_METHOD.html §2, §3, §5). CPU physics.

Plan from a root in phase A (bowl not held) or B (held), matched to demo d at
step t*:
    segment 1   demo phase (A or B) from t* to its gripper event
    segment 2   phase B (only when starting in A), to the release
    then        open the gripper and hold, until success or the budget
Segment goals (object positions read from the sim when the segment starts):
    A  the demo's EEF at its grasp, moved by the target's offset (demo -> now)
    B  held-object frame: the bowl's goal is the plate now + the demo's
       bowl-minus-plate at release; the EEF goal adds the CURRENT EEF-minus-bowl
       offset (the grasp may differ from the demo's). Fixed 10-14: aiming the
       EEF at the demo's EEF goal drove a differently-held bowl into the plate.
The orientation goal is the demo's world orientation at the segment end.

Arms (C1, §2.1, §3):
    dmp         Hoffmann/Pastor DMP, start velocity = the policy's at the root
    dmp_vector  same, vector-wise scaling of the forcing term
    dmp_rest    same as dmp, starting from rest
    linear      I-Gen style: the demo segment translated by the object offset,
                reached by a 20-step linear bridge (10 between segments)
Orientation starts from rest in every DMP arm (the root's angular velocity is
not matched).

Tracking (§5.1): a = clip((y_ref[i+L] - y) / 0.05), clip(rotvec(R_ref[i+L] R^-1) / 0.5),
the OSC_POSE delta convention (goal = current + delta; world-frame axis-angle).
After a segment's reference ends, its goal is held until the EEF is within
REACH_M (the DMP is a few mm short at t = tau; dmp.py) or HOLD_MAX steps pass.
At each gripper event the arm holds still for the demo's own settle time.

Checks (§5.2): success (env check) within BUDGET steps after the root (C3: not
capped at 280); jerk after the join; peak speed and saturation; collateral
(any non-target object moved > COLLATERAL_M from its root position).

UNVERIFIED: new numeric code, not yet checked by val.
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demos as D  # noqa: E402
from dmp import DMP, OrientationDMP, linear_bridge  # noqa: E402
from restore import control_env, eef_pose, r047_rows, sim_of  # noqa: E402

DT = 0.05
POS_SCALE, ROT_SCALE = 0.05, 0.5     # OSC_POSE output limits (demo controller config)
LOOKAHEAD = 1                        # tuned in T2, then frozen
BUDGET = 200                         # control steps after the root (C3)
REACH_M = 0.005
HOLD_MAX = 30
JERK_STEPS = 16                      # one GR00T chunk after the join
COLLATERAL_M = 0.02
SAT_FRAC = 0.10
BRIDGE, BRIDGE_BETWEEN = 20, 10
CLEAR_DZ = 0.03          # clearance lift if the root is this far below the matched demo point
LIFT_SPEED = 0.15        # m/s for the clearance lift
ARMS = ("dmp", "dmp_vector", "dmp_rest", "linear")
ACTION_DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


# -- demo statistics the checks compare against ------------------------------
def _jerk(p):
    return np.linalg.norm(np.diff(p, n=3, axis=0), axis=1) / DT ** 3


@lru_cache(maxsize=1)
def demo_stats():
    """p99 over the usable demos of: speed (m/s), and the max jerk (m/s^3) over
    a JERK_STEPS window (all windows)."""
    speeds, jmax = [], []
    for task in D.tasks():
        for d in D.load(task):
            if not d.usable:
                continue
            speeds += list(np.linalg.norm(np.diff(d.eef_pos, axis=0), axis=1) / DT)
            j = _jerk(d.eef_pos)
            jmax += [j[i:i + JERK_STEPS].max() for i in range(0, max(1, len(j) - JERK_STEPS))]
    return {"speed_p99": float(np.percentile(speeds, 99)), "jerk_window_p99": float(np.percentile(jmax, 99))}


def settle_steps(d, t_event, cap=20):
    """Steps after a demo's gripper event until its aperture stops changing."""
    ap = d.aperture
    for u in range(t_event + 1, min(t_event + cap, len(ap) - 1)):
        if abs(ap[u + 1] - ap[u]) < 1e-4:
            return max(3, u - t_event)
    return cap


# -- execution ----------------------------------------------------------------
@dataclass
class Rec:
    eef: list = field(default_factory=list)
    rot: list = field(default_factory=list)
    act: list = field(default_factory=list)
    objs: list = field(default_factory=list)
    success: list = field(default_factory=list)


class Runner:
    def __init__(self, root, budget=BUDGET, obj_names=()):
        # steps LIBERO's ControlEnv directly (restore.cameras_off() first): the
        # same call LeRobot's step makes, without rendering
        self.root, self.env, self.sim = root, root.env, sim_of(root.env)
        self.cenv = control_env(root.env)
        self.obj_names = list(obj_names)
        self.budget = budget
        self.rec = Rec()
        self.succeeded_at = None

    @property
    def n(self):
        return len(self.rec.act)

    def pose(self):
        return eef_pose(self.sim)

    def obj(self, name):
        bid = self.sim.model.body_name2id(name)
        return np.array(self.sim.data.body_xpos[bid])

    def step(self, a):
        a = np.asarray(a, dtype=np.float32)     # the dtype the harness hands LeRobot
        self.cenv.step(a)
        success = bool(self.cenv.check_success())
        p, R = self.pose()
        self.rec.eef.append(p)
        self.rec.rot.append(R.as_quat())
        self.rec.act.append(a.astype(float))
        self.rec.objs.append({k: self.obj(k).tolist() for k in self.obj_names})
        self.rec.success.append(bool(success))
        if success and self.succeeded_at is None:
            self.succeeded_at = self.n
        return self.done()

    def done(self):
        return self.succeeded_at is not None or self.n >= self.budget

    def track(self, Y, R: Rotation, grip, hold_goal=True):
        """Follow a reference (positions Y, rotations R) step by step."""
        for i in range(len(Y) - 1):
            j = min(i + LOOKAHEAD, len(Y) - 1)
            if self.step(self.action(Y[j], R[j], grip)):
                return True
        if hold_goal:
            for _ in range(HOLD_MAX):
                p, _ = self.pose()
                if np.linalg.norm(p - Y[-1]) < REACH_M:
                    break
                if self.step(self.action(Y[-1], R[len(R) - 1], grip)):
                    return True
        return False

    def action(self, y_ref, R_ref, grip):
        p, R = self.pose()
        dp = np.clip((y_ref - p) / POS_SCALE, -1, 1)
        dr = np.clip((R_ref * R.inv()).as_rotvec() / ROT_SCALE, -1, 1)
        return np.concatenate([dp, dr, [grip]])

    def hold(self, steps, grip):
        p, R = self.pose()
        for _ in range(steps):
            if self.step(self.action(p, R, grip)):
                return True
        return False

    def velocity(self):
        if self.n >= 2:
            return (self.rec.eef[-1] - self.rec.eef[-2]) / DT
        return self.root.eef_vel


def _segment_plan(d, phase, t_star):
    """[(phase, s, e)] demo step ranges to retarget; e is the gripper-event step."""
    A, B, _ = d.phases
    if phase == "A":
        return [("A", max(t_star, A[0]), A[1]), ("B", B[0], B[1])]
    return [("B", max(t_star, B[0]), B[1])]


def run(root, match, phase, arm="dmp", target="akita_black_bowl_1_main", plate="plate_1_main",
        budget=BUDGET):
    """Execute one corrective rollout from `root` (restore.Root, env at the root)."""
    if arm not in ARMS:
        raise ValueError(arm)
    d = match.demo
    obj_at_root = getattr(root, "obj_at_root", None) or {
        k: np.array(v) for k, v in r047_rows()[root.row]["steps"][root.t]["obs_state"]["_gt_scene_object_pos"].items()}
    run_ = Runner(root, budget, obj_names=obj_at_root.keys())
    plan = _segment_plan(d, phase, match.t_star)
    stats = demo_stats()
    info = {"segments": []}
    first = True
    # clearance lift (deviation from §2, 10-14): a root low beside the bowl cannot
    # be bent into a demo approach from above without sweeping through the bowl
    # (scenario A shoved it 4.5 cm). If the root is > CLEAR_DZ below the matched
    # demo point (heights relative to the target), go straight up to that height
    # first. Applied to every arm, so the arms stay comparable.
    if phase == "A":
        s0 = plan[0][1]
        dz = (d.eef_pos[s0, 2] - d.target_pos[s0, 2]) - (run_.pose()[0][2] - run_.obj(target)[2])
        if dz > CLEAR_DZ:
            p0, R0 = run_.pose()
            n = max(2, int(np.ceil(dz / LIFT_SPEED / DT)))
            Y = p0 + np.linspace(0, 1, n + 1)[:, None] * np.array([0.0, 0.0, dz])
            info["clearance_lift_m"] = float(dz)
            run_.track(Y, Rotation.concatenate([R0] * (n + 1)), -1.0)
    for seg_phase, s, e in plan:
        if run_.done():
            break
        e_idx = min(e, len(d.eef_pos) - 1)
        if e_idx - s < 3:                 # matched at the very end of the segment
            s = max(e_idx - 3, 0)
        Pd, Rd = d.eef_pos[s:e_idx + 1], d.eef_rot[s:e_idx + 1]
        if seg_phase == "A":
            # approach: the demo's grasp pose relative to the target, at the target now
            offset = run_.obj(target) - d.target_pos[e_idx]
            g = Pd[-1] + offset
        else:
            # carry, held-object frame: the bowl must land where the demo's bowl
            # landed relative to the plate; the hand keeps its CURRENT offset to the
            # bowl (it may hold the bowl elsewhere than the demo did)
            bowl_goal = run_.obj(plate) + (d.target_pos[e_idx] - d.plate_pos[e_idx])
            p_now = run_.pose()[0]
            g = bowl_goal + (p_now - run_.obj(target))
            offset = g - Pd[-1]
        Rg = Rd[len(Rd) - 1]
        p0, R0 = run_.pose()
        grip = -1.0 if seg_phase == "A" else 1.0
        if arm == "linear":
            Y, R = linear_bridge(p0, R0, Pd + offset, Rd, n_bridge=BRIDGE if first else BRIDGE_BETWEEN)
            tau = (len(Y) - 1) * DT
        else:
            dmp = DMP().fit(Pd, DT)
            tau = dmp.tau
            yd0 = np.zeros(3) if (arm == "dmp_rest" and first) else run_.velocity()
            scaling = "vector" if arm == "dmp_vector" else "none"
            Y, _ = dmp.rollout(y0=p0, g=g, yd0=yd0, tau=tau, scaling=scaling)
            peak = (np.linalg.norm(np.diff(Y, axis=0), axis=1) / DT).max()
            if peak > stats["speed_p99"]:   # §2: stretch tau so the peak speed is the demos' p99
                tau *= peak / stats["speed_p99"]
                Y, _ = dmp.rollout(y0=p0, g=g, yd0=yd0, tau=tau, scaling=scaling)
            odmp = OrientationDMP().fit(Rd, DT)
            R, _, _ = odmp.rollout(R0, Rg, omega0=np.zeros(3), tau=tau, n_steps=len(Y) - 1)
        info["segments"].append({"phase": seg_phase, "demo_steps": [int(s), int(e_idx)], "tau_s": float(tau),
                                 "goal": g.tolist(), "offset": offset.tolist(), "start_step": run_.n})
        first = False
        if run_.track(Y, R, grip):
            break
        # gripper event at the segment end: switch and hold for the demo's settle time
        if run_.hold(settle_steps(d, e_idx), -grip):
            break
    if not run_.done():                    # released: open and wait for success
        run_.hold(budget - run_.n, -1.0)
    return judge(root, run_, obj_at_root, target, stats) | {"arm": arm, "demo": d.idx, "t_star": match.t_star,
                                                            "tags": list(match.tags), "plan": info}, run_.rec


def judge(root, run_, obj_at_root, target, stats):
    rec = run_.rec
    eef = np.array(rec.eef) if rec.eef else np.zeros((0, 3))
    pre = getattr(root, "pre_eef", None)
    if pre is None:
        pre = np.array([s["obs_state"]["eef_pos"] for s in r047_rows()[root.row]["steps"][max(0, root.t - 3):root.t + 1]])
    joined = np.vstack([pre, eef[:JERK_STEPS]]) if len(eef) else pre
    jerk = float(_jerk(joined).max()) if len(joined) >= 4 else 0.0
    speed = float((np.linalg.norm(np.diff(np.vstack([pre[-1:], eef]), axis=0), axis=1) / DT).max()) if len(eef) else 0.0
    acts = np.array(rec.act) if rec.act else np.zeros((0, 7))
    sat = float((np.abs(acts[:, :6]) >= 1.0 - 1e-9).any(1).mean()) if len(acts) else 0.0
    moved = {}
    if rec.objs:
        last = rec.objs[-1]
        for k, v in last.items():
            if k == target or k not in obj_at_root:
                continue
            moved[k] = float(np.linalg.norm(np.array(v) - obj_at_root[k]))
    checks = {
        "success": run_.succeeded_at is not None,
        "jerk_ok": jerk <= stats["jerk_window_p99"],
        "speed_ok": speed <= stats["speed_p99"] and sat < SAT_FRAC,
        "collateral_ok": all(v <= COLLATERAL_M for v in moved.values()),
    }
    return {"row": root.row, "t_root": root.t, "kept": all(checks.values()), "checks": checks,
            "steps": run_.n, "success_step": run_.succeeded_at,
            "fits_in_280": run_.succeeded_at is not None and root.t + run_.succeeded_at <= 280,
            "jerk_max": jerk, "jerk_limit": stats["jerk_window_p99"], "speed_max": speed,
            "speed_limit": stats["speed_p99"], "saturated_frac": sat, "moved_m": moved}
