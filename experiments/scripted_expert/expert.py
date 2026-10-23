"""Scripted expert: complete a LIBERO pick-and-place from a restored root. CPU physics.

From the root (DMP's restore_fast: the env exactly where the failed rollout
was), read the sim: hand pose, bowl and plate poses, bowl tilt, whether the
bowl is held. Then run primitives whose waypoints come from that state and
from the human-demo statistics in demo_params.py:

  pick            approach: rise straight up to the pre-grasp height (if below
                  it) blended into the move to the pre-grasp point (pregrasp_dz
                  above the grasp, grasp orientation) -> descend to the grasp
                  (median human EEF-minus-bowl offset on the rim side the hand
                  is on, or the nearest demo's) -> close, settle
  carry           rise by lift_m blended into the move to pregrasp_dz above the
                  place point (which may be lower, e.g. a bowl picked off the
                  cabinet: the move comes down on the way, 10-20), keeping the
                  current hand-to-bowl offset; then the lift check: the bowl's
                  highest point during the carry was >= roots.LIFT_OK above
                  its height at the grasp
  place           lower the bowl to support + place_offset -> open, settle
                  -> retreat up pregrasp_dz -> wait open

Branches at the root:
  bowl held, tilt <= tilt_carry_p95   carry (no extra rise), place
  bowl held, tilted more              open, retreat, pick, carry, place
  bowl not held                       pick, carry, place  (one regrasp if the lift
                                      check fails and the variant allows it)

"Held" uses DMP's roots.py definition at a single moment: commanded closed
(action > roots.CLOSED) and the bowl within roots.GRIP_R of the grip site.

Legs are minimum-jerk (paths.leg) capped at the variant's vmax and the demos'
p99 rotation rate; the first leg starts at the root's logged EEF velocity.
Steps go through DMP's execute.Runner (same OSC_POSE action convention,
lookahead and 200-step budget as DMP). Expert.track (10-20) differs from
Runner.track in two ways: actions are kept in the human action range (the
demos' max |action|: position 0.9375, the teleop limit, reached on ~10% of
human steps; never the controller's 1.0) and while the target is out of that
range the reference waits for the hand instead of running ahead; after each
leg the goal is held until within REACH_M or the hand has stopped and are judged by execute.judge (the 4 keep checks).
Gripper settle: hold still until the finger gap changes < SETTLE_TOL per step
(DMP's settle_steps rule, read live), 3..20 steps.

Speed (10-20): variants "default" and "nearest" use the human-speed law
(paths.HumanLaw): approach legs (approach, descend, retreat) average the task's
human approach speed, carry legs (carry, lower) its human carry speed, peak
capped at the human speed_p90 (a faster cruise saturated the action), jerk at the demos' jerk_window_p99, rotation
at omega_p99. After the close the hand holds for the task's human close_hold
steps and then lifts (instead of waiting for the fingers to stop).
"minjerk" is the earlier expert (minimum-jerk legs capped at VMAX_MARGIN x
speed_p99, close settle by the finger-gap rule), kept as an ablation.

The task goal is read from the BDDL (parsed_problem goal_state); only
("on", obj, support) between two bodies is implemented.

Privileged: object poses come from the sim.

UNVERIFIED: new numeric code, not yet checked by val.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

import demo_params as DP
from dmp_link import execute as E, restore as RS, roots as RT
from paths import HumanLaw, leg, rise_then_travel

VMAX_MARGIN = 0.9          # default cap = 0.9 x demo p99 speed, so tracking overshoot stays under the keep limit
SETTLE_TOL, SETTLE_MIN, SETTLE_MAX = 1e-4, 3, 20
STOPPED_M, STOPPED_N = 0.001, 2   # hand moved < 1 mm per step for 2 steps: stopped (10-20)
BLEND = 0.5                       # rise -> travel: travel starts half-way through the rise (10-20)
MIN_RISE_M = 0.01                 # a shorter rise is skipped (one plain leg instead)


@dataclass(frozen=True)
class Variant:
    name: str
    grasp: str = "median"   # "median" | "nearest"
    law: str = "human"      # "human" = HumanLaw per phase, "minjerk" = minimum jerk capped at the vmax
    retry: bool = True


VARIANTS = {v.name: v for v in (Variant("default"), Variant("minjerk", law="minjerk"),
                                Variant("nearest", grasp="nearest"))}


def goal_of(cenv):
    """(object, support) body names from the task's BDDL goal."""
    goals = cenv.env.parsed_problem["goal_state"]
    if len(goals) != 1 or goals[0][0].lower() != "on":
        raise NotImplementedError(f"goal {goals}: only a single ('on', obj, support) is implemented")
    return goals[0][1] + "_main", goals[0][2] + "_main"


def tilt_deg(sim, body) -> float:
    w, x, y, z = sim.data.body_xquat[sim.model.body_name2id(body)]
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0))))


class Expert:
    def __init__(self, root, task, variant: Variant, obj_names, budget=E.BUDGET):
        self.root, self.task, self.v = root, task, variant
        self.P = DP.params()
        self.run = E.Runner(root, budget, obj_names=obj_names)
        self.sim, self.cenv = self.run.sim, self.run.cenv
        self.obj, self.support = goal_of(self.cenv)
        self.vmax = VMAX_MARGIN * self.P["speed_p99"]
        self.omega = self.P["omega_p99"]
        self.law = {"approach": None, "carry": None}
        self.close_hold = None
        if variant.law == "human":
            hp = self.P["by_task"][task]
            self.law = {ph: HumanLaw(vbar=hp[f"{ph}_speed"], vmax=self.P["speed_p90"], jcap=self.P["jerk_window_p99"],
                                     omega_max=self.omega) for ph in ("approach", "carry")}
            self.close_hold = int(hp["close_hold"])
        fj = [self.sim.model.get_joint_qpos_addr(f"gripper0_finger_joint{i}") for i in (1, 2)]
        self._fj = fj
        steps = RS.r047_rows()[root.row]["steps"]
        last = steps[root.t - 1]["action"] if root.t > 0 else None
        self.grip = 1.0 if (last is not None and last[6] > RT.CLOSED) else -1.0
        self.log = {"variant": variant.name, "legs": [], "retries": 0}

    # -- state ---------------------------------------------------------------
    def aperture(self):
        q = self.sim.data.qpos
        return float(q[self._fj[0]] - q[self._fj[1]])

    def held(self):
        p, _ = self.run.pose()
        return self.grip > 0 and np.linalg.norm(self.run.obj(self.obj) - p) <= RT.GRIP_R

    # -- motions -------------------------------------------------------------
    def track(self, name, Y, R, grip):
        """Follow a reference, then hold its goal until the hand is within
        E.REACH_M of it or has stopped (< STOPPED_M per step, STOPPED_N steps
        running; 10-20: a hand resting on the rim never gets within 5 mm),
        at most E.HOLD_MAX steps. True when the episode is over."""
        self.log["legs"].append({"name": name, "start_step": self.run.n, "goal": np.asarray(Y[-1]).tolist(),
                                 "steps": len(Y) - 1})
        self.grip = grip
        i, n, waits = 0, len(Y) - 1, 0
        while i < n:
            j = min(i + E.LOOKAHEAD, n)
            a, limited = self.action(Y[j], R[j], grip)
            if self.run.step(a):
                return True
            if limited:          # the hand is behind: the reference waits for it
                waits += 1
            else:
                i += 1
        self.log["legs"][-1]["waits"] = waits
        still, prev = 0, self.run.pose()[0]
        for _ in range(E.HOLD_MAX):
            if np.linalg.norm(prev - Y[-1]) < E.REACH_M or still >= STOPPED_N:
                break
            if self.run.step(self.action(Y[-1], R[len(R) - 1], grip)[0]):
                return True
            p = self.run.pose()[0]
            still = still + 1 if np.linalg.norm(p - prev) < STOPPED_M else 0
            prev = p
        self.log["legs"][-1]["held_steps"] = self.run.n - self.log["legs"][-1]["start_step"] - (len(Y) - 1) - waits
        return False

    def action(self, y_ref, R_ref, grip):
        """execute.Runner's OSC_POSE action, scaled (direction kept) into the human
        action range (demo_params action_pos_max / action_rot_max). Returns
        (action, limited): limited means the target was out of that range."""
        p, Rc = self.run.pose()
        dp = (np.asarray(y_ref) - p) / E.POS_SCALE
        dr = (R_ref * Rc.inv()).as_rotvec() / E.ROT_SCALE
        limited = False
        for v, cap in ((dp, self.P["action_pos_max"]), (dr, self.P["action_rot_max"])):
            m = np.abs(v).max()
            if m > cap:
                v *= cap / m
                limited = True
        return np.concatenate([dp, dr, [grip]]), limited

    def move(self, name, p1, R1: Rotation, grip, phase):
        """One leg (time law of the variant and phase); True when the episode is over."""
        p0, R0 = self.run.pose()
        Y, R = leg(p0, R0, np.asarray(p1), R1, self.run.velocity(), self.vmax, self.omega, law=self.law[phase])
        return self.track(name, Y, R, grip)

    def rise_travel(self, name, z1, p2, R2: Rotation, grip, phase):
        """Rise straight up to z1 blended into a leg to p2 (no stop at the corner)."""
        p0, R0 = self.run.pose()
        if z1 - p0[2] < MIN_RISE_M:
            return self.move(name, p2, R2, grip, phase)
        Y, R = rise_then_travel(p0, R0, max(z1, p0[2]), np.asarray(p2), R2, self.run.velocity(),
                                self.vmax, self.omega, BLEND, law=self.law[phase])
        return self.track(name, Y, R, grip)

    def close(self):
        """Close on the rim: hold for the human close_hold steps, or (minjerk) settle."""
        if self.close_hold is None:
            return self.settle(1.0)
        self.grip = 1.0
        p, R = self.run.pose()
        for _ in range(self.close_hold):
            if self.run.step(self.action(p, R, 1.0)[0]):
                return True
        return False

    def settle(self, grip):
        self.grip = grip
        p, R = self.run.pose()
        prev = self.aperture()
        for k in range(SETTLE_MAX):
            if self.run.step(self.action(p, R, grip)[0]):
                return True
            ap = self.aperture()
            if k + 1 >= SETTLE_MIN and abs(ap - prev) < SETTLE_TOL:
                break
            prev = ap
        return False

    def rise(self, name, dz, grip, phase):
        p, R = self.run.pose()
        return self.move(name, p + [0.0, 0.0, dz], R, grip, phase)

    # -- primitives ----------------------------------------------------------
    def grasp_pose(self):
        bowl = self.run.obj(self.obj)
        p, _ = self.run.pose()
        if self.v.grasp == "nearest":
            off, Rg, idx = DP.nearest_grasp(self.task, p - bowl)
            self.log["nearest_demo"] = int(idx)
        else:
            off, Rg = DP.grasp_of(DP.side_of(p - bowl))
        self.log["grasp_side"] = DP.side_of(off)
        return bowl + off, Rg

    def pick(self):
        """Approach (rising first if below the pre-grasp height), descend, close.
        True when the episode is over."""
        G, Rg = self.grasp_pose()
        pre = G + [0.0, 0.0, self.P["pregrasp_dz_m"]]
        if (self.rise_travel("approach", pre[2], pre, Rg, -1.0, "approach")
                or self.move("descend", G, Rg, -1.0, "approach")):
            return True
        self.z_grasp = self.run.obj(self.obj)[2]
        return self.close()

    def let_go(self):
        return self.settle(-1.0) or self.rise("retreat", self.P["pregrasp_dz_m"], -1.0, "approach")

    def carry(self, rise_to):
        """Rise to `rise_to` blended into the move to pregrasp_dz above the place
        point, keeping the current hand-to-bowl offset.
        True when the episode is over."""
        p, R = self.run.pose()
        bowl_goal = self.run.obj(self.support) + self.P["place_offset_m"]
        down = bowl_goal + (p - self.run.obj(self.obj))
        above = down + [0.0, 0.0, self.P["pregrasp_dz_m"]]
        self.down, self.R_carry = down, R
        return self.rise_travel("carry", rise_to, above, R, 1.0, "carry")

    def place(self):
        return self.move("lower", self.down, self.R_carry, 1.0, "carry") or self.let_go()

    def solve(self):
        held, tilt = self.held(), tilt_deg(self.sim, self.obj)
        self.log |= {"held_at_root": bool(held), "tilt_at_root_deg": tilt}
        if held and tilt <= self.P["tilt_carry_p95_deg"]:
            self.log["branch"] = "carry"
            if self.carry(self.run.pose()[0][2]):
                return
        else:
            self.log["branch"] = "regrasp_tilted" if held else "pick"
            if held and self.let_go():
                return
            while True:
                if self.pick() or self.carry(self.run.pose()[0][2] + self.P["lift_m"]):
                    return
                # lift check after the carry: the bowl came up with the hand at some point
                s0 = self.log["legs"][-1]["start_step"]
                self.log["lift_m"] = float(max(o[self.obj][2] for o in self.run.rec.objs[s0:]) - self.z_grasp)
                self.log["lift_ok"] = self.log["lift_m"] >= RT.LIFT_OK
                if self.log["lift_ok"] or not self.v.retry or self.log["retries"] >= 1:
                    break
                self.log["retries"] += 1
                if self.let_go():
                    return
        if self.place():
            return
        self.run.hold(self.run.budget - self.run.n, -1.0)     # released: wait for success


def run(root, task, variant="default", budget=E.BUDGET):
    """Execute one scripted correction from `root` (restore.Root, env at the root)."""
    v = VARIANTS[variant]
    obj_at_root = {k: np.array(x) for k, x in
                   RS.r047_rows()[root.row]["steps"][root.t]["obs_state"]["_gt_scene_object_pos"].items()}
    ex = Expert(root, task, v, obj_at_root.keys(), budget)
    ex.solve()
    res = E.judge(root, ex.run, obj_at_root, ex.obj, E.demo_stats())
    res |= {"arm": f"scripted_{v.name}", "plan": ex.log,
            "kept_without_retry": bool(res["kept"] and ex.log["retries"] == 0)}
    return res, ex.run.rec
