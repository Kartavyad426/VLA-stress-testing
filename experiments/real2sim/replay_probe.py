"""CPU re-simulation probes of policy_batch episodes (10-23). No rendering, no policy: each episode's recorded actions
are re-issued open loop from the start policy_rollout used (physics doesn't depend on the renderer). A replay under the
physics the batch ran with must reproduce the recorded fingertip track (checked, reported as rep_err_m).

--mode physics: which part of placement_analysis/policy_rollout's "firm" grip setting makes the open finger pads stick on
  the table (firm batch: blocked 2 -> 18). Each episode is replayed under: firm (as run), torsion (pads condim 4 +
  torsional 0.02 only), solver (elliptic cone + impratio 10 only), elliptic, impratio, scene (as built). Per setting:
  blocked segments (policy_rollout's rule) and steps where an open pad touches the table.
  Open loop: the commands were the policy's reactions to the firm physics, so this shows which setting holds the arm
  back on these commands, not what the policy would have done.
--mode stack: release of the carried block in stacking tasks, for policy batches and for the scripted expert (whose
  make_tasks *_states.npz hold every step's qpos). At the gripper opening that ends the last carry: target bottom above
  the lower block's top, horizontal offset, target speed, spin, tilt, hand speed; during the carry: rotation of the
  block relative to the hand; after: final offset and whether it stayed on top.

    .venvs/real2sim/bin/python experiments/real2sim/replay_probe.py --mode physics \
        --batch runs/real2sim/policy/batch_orange_stereo_v2_firm --grip_physics firm \
        --scene runs/real2sim/ep_orange_block_stereo:data/real2sim/droid/ep_orange_block --only_blocked
Exploratory, UNVERIFIED.
"""
import argparse, glob, json, os, sys
import numpy as np
import h5py
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_tasks import scene_tasks, task_success, randomise  # noqa: E402
from policy_rollout import rollout_diagnostics, FPS  # noqa: E402

SETTINGS = ("firm", "torsion", "solver", "elliptic", "impratio", "scene")


def apply_physics(m, s):
    """firm = torsion + solver, exactly policy_rollout --grip_physics firm."""
    if s in ("firm", "torsion"):
        for g in range(m.ngeom):
            if "pad" in m.body(m.geom_bodyid[g]).name:
                m.geom_condim[g] = 4
                m.geom_friction[g] = [1.0, 0.02, 0.0001]
    if s in ("firm", "solver", "elliptic"):
        m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    if s in ("firm", "solver", "impratio"):
        m.opt.impratio = 10.0


class Scene:
    def __init__(self, scene, setting):
        sdir, edir = scene.split(":")
        self.m = m = mujoco.MjModel.from_binary_path(os.path.join(sdir, "scene.mjb"))
        apply_physics(m, setting)
        self.fit = json.load(open(os.path.join(sdir, "scene_fit.json")))
        self.tasks = scene_tasks(m, self.fit)
        self.arm_q = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
        self.arm_u = [m.actuator(f"fr3_joint{k}").id for k in range(1, 8)]
        self.grip_u = [i for i in range(m.nu) if m.actuator(i).name.startswith("g_")][0]
        self.g_lo, self.g_hi = m.actuator_ctrlrange[self.grip_u]
        self.lo, self.hi = m.actuator_ctrlrange[self.arm_u].T
        self.free = [m.body(m.jnt_bodyid[j]).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.q_start = h5py.File(os.path.join(edir, "trajectory.h5"))["observation/robot_state/joint_positions"][0]
        self.n_sub = int(round(1 / FPS / m.opt.timestep))
        self.pinch = m.site("g_pinch").id
        self.hand = m.site_bodyid[self.pinch]
        self.pads = {b for b in range(m.nbody) if "pad" in m.body(b).name}
        self.table = m.geom("table").id

    def half(self, body):
        g = [i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] == self.m.body(body).id][0]
        return float(self.m.geom_size[g][2])


def layout_seed(r, ti, tasks_dir, ep_dir_name):
    if "layout_rng_seed" in r:
        return r["layout_rng_seed"]
    vj = json.load(open(os.path.join(tasks_dir, ep_dir_name, f"v{r['variant']}.json")))  # pre-fix batch (D2)
    return vj["seed"] * 1000 + ti * 100 + r["variant"]


def replay(S, r, z, seed):
    m = S.m
    ti = [i for i, t in enumerate(S.tasks) if t["text"] == r["instruction"]][0]
    task = S.tasks[ti]
    d, dk = mujoco.MjData(m), mujoco.MjData(m)
    d.qpos[S.arm_q] = S.q_start
    if r["variant"] > 0:
        randomise(m, d, np.random.default_rng(seed), S.free)
    d.ctrl[S.arm_u] = S.q_start
    d.ctrl[S.grip_u] = S.g_lo
    for _ in range(int(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    tb, rb = m.body(task["target"]).id, m.body(task["ref"]).id
    out = {k: [] for k in ("cmd", "ach", "tgt", "ref", "grip", "tq", "hq", "pad_table")}
    for act in z["action"]:
        d.ctrl[S.arm_u] = np.clip(act[10:17], S.lo, S.hi)
        d.ctrl[S.grip_u] = S.g_hi if act[9] > 0.5 else S.g_lo
        for _ in range(S.n_sub):
            mujoco.mj_step(m, d)
        dk.qpos[:] = d.qpos
        dk.qpos[S.arm_q] = d.ctrl[S.arm_u]
        mujoco.mj_kinematics(m, dk)
        out["cmd"].append(dk.site_xpos[S.pinch].copy())
        out["ach"].append(d.site_xpos[S.pinch].copy())
        out["tgt"].append(d.xpos[tb].copy())
        out["ref"].append(d.xpos[rb].copy())
        out["grip"].append(bool(act[9] > 0.5))
        out["tq"].append(d.xquat[tb].copy())
        out["hq"].append(d.xquat[S.hand].copy())
        pt = False
        for ci in range(d.ncon):
            c = d.contact[ci]
            gs = (c.geom1, c.geom2)
            if S.table in gs and (m.geom_bodyid[c.geom1] in S.pads or m.geom_bodyid[c.geom2] in S.pads):
                pt = True
        out["pad_table"].append(pt)
    for _ in range(int(0.8 / m.opt.timestep)):  # settle as policy_rollout before judging
        mujoco.mj_step(m, d)
    out = {k: np.array(v) for k, v in out.items()}
    out["success"] = bool(task_success(m, d, S.fit, task))
    out["final_tgt"], out["final_ref"] = d.xpos[tb].copy(), d.xpos[rb].copy()
    out["task"] = task
    return out


def quat_angle(q1, q2):
    return float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(q1, q2)), 0, 1))))


def tilt_deg(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return float(np.degrees(np.arccos(np.clip(np.abs(R.reshape(3, 3)[2]).max(), 0, 1))))  # cube: nearest axis to world z


def rel_rot(hq, tq):
    """block orientation in the hand frame"""
    hi = np.zeros(4)
    mujoco.mju_negQuat(hi, hq)
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, hi, tq)
    return q


def release_stats(tgt, ref, grip, tq, hq, hand, h_t, h_r, z0):
    """Stats at the gripper opening that ends the last carry (target >= 2 cm above its start), per-step arrays at FPS."""
    lifted = tgt[:, 2] - z0 >= 0.02
    if not lifted.any():
        return dict(outcome="not_lifted")
    last = int(np.nonzero(lifted)[0][-1])
    s = last
    while s > 0 and lifted[s - 1]:
        s -= 1
    opened = [k for k in range(s, len(grip)) if not grip[k]]
    if not opened:
        return dict(outcome="never_released")
    k = opened[0]
    if k > last + 1:
        return dict(outcome="dropped_while_closed", carry_steps=last - s + 1)
    a, b = max(k - 2, 0), min(k + 2, len(tgt) - 1)
    v = np.linalg.norm(tgt[b] - tgt[a]) / ((b - a) / FPS)
    w = quat_angle(tq[a], tq[b]) / ((b - a) / FPS)
    vh = np.linalg.norm(hand[b] - hand[a]) / ((b - a) / FPS)
    c0 = min(s + 5, k)  # hold settled ~0.3 s after the lift starts
    p = max(k - 3, s)  # 0.2 s before the opening command: was the block already moving/turning in the closed gripper?
    pre = dict(pre_speed_cms=100 * float(np.linalg.norm(tgt[k] - tgt[p]) / max((k - p) / FPS, 1e-9)),
               pre_spin_dps=quat_angle(tq[p], tq[k]) / max((k - p) / FPS, 1e-9),
               pre_rot_in_hand_dps=quat_angle(rel_rot(hq[p], tq[p]), rel_rot(hq[k], tq[k])) / max((k - p) / FPS, 1e-9))
    return dict(outcome="released", release_step=k, carry_steps=k - s, **pre,
                bottom_above_ref_top_cm=100 * float((tgt[k, 2] - h_t) - (ref[k, 2] + h_r)),
                offset_cm=100 * float(np.linalg.norm(tgt[k, :2] - ref[k, :2])),
                speed_cms=100 * v, spin_dps=w, tilt_deg=tilt_deg(tq[k]), hand_speed_cms=100 * vh,
                rot_in_hand_deg=quat_angle(rel_rot(hq[c0], tq[c0]), rel_rot(hq[k], tq[k])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["physics", "stack"], required=True)
    ap.add_argument("--batch", action="append", default=[], help="policy_batch out dir (repeatable)")
    ap.add_argument("--grip_physics", action="append", default=[], help="physics each --batch ran with (scene|firm), same order")
    ap.add_argument("--scene", action="append", default=[], help="<scene dir>:<episode dir> per --batch, same order (or one for all)")
    ap.add_argument("--tasks", default="runs/real2sim/tasks_stereo/ep_orange_block_stereo", help="for pre-fix batch seeds")
    ap.add_argument("--expert", action="append", default=[], help="stack mode: make_tasks task dir(s) with v*_states.npz, "
                    "as <task dir>@<scene dir>:<episode dir>")
    ap.add_argument("--only_blocked", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scenes = a.scene if len(a.scene) == len(a.batch) else a.scene * len(a.batch)
    res = dict(mode=a.mode, status="exploratory, UNVERIFIED", rows=[])
    if a.mode == "physics":
        for B, gp, sc in zip(a.batch, a.grip_physics, scenes):
            cache = {s: Scene(sc, s) for s in SETTINGS}
            for rj in sorted(glob.glob(f"{B}/*/v*/rollout.json")):
                r = json.load(open(rj))
                if a.only_blocked and not r["tracking"]["blocked"]:
                    continue
                z = np.load(rj.replace(".json", ".npz"))
                epn = os.path.basename(os.path.dirname(os.path.dirname(rj)))
                row = dict(episode=f"{epn}/v{r['variant']}", recorded_blocked_steps=sum(s["steps"] for s in r["tracking"]["blocked_segments"]))
                for s in SETTINGS:
                    S = cache[s]
                    ti = [i for i, t in enumerate(S.tasks) if t["text"] == r["instruction"]][0]
                    o = replay(S, r, z, layout_seed(r, ti, a.tasks, epn))
                    tr = rollout_diagnostics(o["cmd"], o["ach"], o["tgt"], o["grip"], float(o["tgt"][0][2]), o["success"])["tracking"]
                    row[s] = dict(blocked_steps=sum(x["steps"] for x in tr["blocked_segments"]), n_segments=len(tr["blocked_segments"]),
                                  open_pad_on_table_steps=int((o["pad_table"] & ~o["grip"]).sum()), success=o["success"])
                    if s == gp:
                        row["rep_err_m"] = float(np.abs(o["ach"] - z["pinch_ach"]).max())
                res["rows"].append(row)
                print(row["episode"], "rep_err", f"{row.get('rep_err_m', float('nan')):.1e}", "| blocked steps:",
                      " ".join(f"{s} {row[s]['blocked_steps']}" for s in SETTINGS), "| open pad on table:",
                      " ".join(f"{s} {row[s]['open_pad_on_table_steps']}" for s in SETTINGS), flush=True)
        tot = {s: sum(r_[s]["blocked_steps"] for r_ in res["rows"]) for s in SETTINGS}
        res["total_blocked_steps"] = tot
        print("TOTAL blocked steps", tot)
    else:
        for B, gp, sc in zip(a.batch, a.grip_physics, scenes):
            S = Scene(sc, gp)
            for rj in sorted(glob.glob(f"{B}/*stack*/v*/rollout.json")):
                r = json.load(open(rj))
                z = np.load(rj.replace(".json", ".npz"))
                epn = os.path.basename(os.path.dirname(os.path.dirname(rj)))
                ti = [i for i, t in enumerate(S.tasks) if t["text"] == r["instruction"]][0]
                o = replay(S, r, z, layout_seed(r, ti, a.tasks, epn))
                t = o["task"]
                st = release_stats(o["tgt"], o["ref"], o["grip"], o["tq"], o["hq"], o["ach"], S.half(t["target"]), S.half(t["ref"]), float(o["tgt"][0][2]))
                st.update(source=B, physics=gp, episode=f"{epn}/v{r['variant']}", success=o["success"], rep_err_m=float(np.abs(o["ach"] - z["pinch_ach"]).max()),
                          final_offset_cm=100 * float(np.linalg.norm(o["final_tgt"][:2] - o["final_ref"][:2])),
                          final_on_top=bool(o["final_tgt"][2] > o["final_ref"][2] + 0.5 * S.half(t["ref"])))
                res["rows"].append(st)
        for e in a.expert:
            tdir, sc = e.split("@")
            S = Scene(sc, "scene")
            m = S.m
            d = mujoco.MjData(m)
            for sf in sorted(glob.glob(f"{tdir}/*stack*/v*_states.npz")):
                vj = json.load(open(sf.replace("_states.npz", ".json")))
                t = [x for x in S.tasks if x["text"] == vj["instruction"]][0]
                qs = np.load(sf)["qpos"]
                acts = np.load(sf.replace("_states.npz", ".npz"))["action"]
                tb, rb = m.body(t["target"]).id, m.body(t["ref"]).id
                arr = {k: [] for k in ("tgt", "ref", "tq", "hq", "ach")}
                for q in qs:
                    d.qpos[:] = q
                    mujoco.mj_kinematics(m, d)
                    arr["tgt"].append(d.xpos[tb].copy()); arr["ref"].append(d.xpos[rb].copy())
                    arr["tq"].append(d.xquat[tb].copy()); arr["hq"].append(d.xquat[S.hand].copy()); arr["ach"].append(d.site_xpos[S.pinch].copy())
                arr = {k: np.array(v) for k, v in arr.items()}
                g = acts[:, 7]
                grip = g > 0.5 * (g.max() + g.min()) if g.max() > g.min() else np.zeros(len(g), bool)
                st = release_stats(arr["tgt"], arr["ref"], grip, arr["tq"], arr["hq"], arr["ach"], S.half(t["target"]), S.half(t["ref"]), float(arr["tgt"][0][2]))
                st.update(source="expert:" + tdir, physics="scene", episode=os.path.basename(os.path.dirname(sf)) + "/" + os.path.basename(sf)[:2],
                          success=bool(vj["success"]), final_offset_cm=100 * float(np.linalg.norm(arr["tgt"][-1][:2] - arr["ref"][-1][:2])),
                          final_on_top=bool(arr["tgt"][-1][2] > arr["ref"][-1][2] + 0.5 * S.half(t["ref"])))
                res["rows"].append(st)
        keys = ("bottom_above_ref_top_cm", "offset_cm", "pre_speed_cms", "pre_spin_dps", "pre_rot_in_hand_dps", "speed_cms", "spin_dps", "tilt_deg", "hand_speed_cms", "rot_in_hand_deg", "final_offset_cm")
        for r_ in res["rows"]:
            print(f"{os.path.basename(r_['source'].rstrip('/'))[:28]:28s} {r_['episode'][:28]:28s} {r_['outcome']:14s} succ {int(r_['success'])} on_top {int(r_.get('final_on_top', 0))} "
                  + " ".join(f"{k.split('_')[0]}={r_[k]:.1f}" for k in keys if k in r_), flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
