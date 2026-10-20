"""Separate placement failures (user request 10-20): episodes where the policy LIFTED the target but the task failed.

Each episode of a policy_batch run is re-simulated on CPU from its recorded actions (no rendering, no policy: physics
doesn't depend on the renderer; the episode start replicates policy_rollout exactly, and the replayed target track
must match the recorded one, else the episode is reported as not reproduced). Per step we log the target AND the
reference object (bowl / lower block / neighbour), the gripper command and the fingertip point, then classify the
LAST carry (target held >= 2 cm above its start):
  dropped_while_closed   the target fell (< 1 cm lift) while the gripper was still commanded closed
  released_away          gripper opened with the target NOT over the goal region (per task_success's geometry)
  released_over_goal     gripper opened with the target over the goal region, but the final state failed
                         (bounced off the rim / slid off the lower block / ended outside the next-to band)
  never_released         still held at the end of the episode
  success                (for reference)
Writes <batch>/placement_analysis.json and prints a table.

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/placement_analysis.py \
        --batch runs/real2sim/policy/batch_orange_stereo_v2 \
        --scene runs/real2sim/ep_orange_block_stereo:data/real2sim/droid/ep_orange_block
"""
import argparse, glob, json, os, sys
import numpy as np
import h5py
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_tasks import scene_tasks, task_success, randomise, body_radius, footprint_radius  # noqa: E402

FPS = 15
LIFT_M = 0.02


def goal_region(m, d, fit, task, pt, pr):
    """Is the target (at pt) over the goal region of `task` (horizontal part of task_success)? + distance to it."""
    horiz = float(np.linalg.norm(pt[:2] - pr[:2]))
    if task["type"] == "put_in_bowl":
        lim = 0.8 * [o for o in fit["objects"].values() if o["kind"] == "bowl"][0]["rim_radius"]
        return horiz < lim, horiz, lim
    if task["type"] == "stack":
        lim = body_radius(m, task["ref"])
        return horiz < lim, horiz, lim
    gap = horiz - footprint_radius(m, fit, task["ref"]) - footprint_radius(m, fit, task["target"])
    return (-0.005 < gap < 0.08), gap, 0.08


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_seed_from_json", action="store_true", help="batches run before the 10-20 policy_batch seed fix")
    ap.add_argument("--tasks", default="runs/real2sim/tasks_stereo/ep_orange_block_stereo", help="make_tasks output for the scene")
    a = ap.parse_args()
    sdir, edir = a.scene.split(":")
    m = mujoco.MjModel.from_binary_path(os.path.join(sdir, "scene.mjb"))
    fit = json.load(open(os.path.join(sdir, "scene_fit.json")))
    tasks = scene_tasks(m, fit)
    arm_q = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    arm_u = [m.actuator(f"fr3_joint{k}").id for k in range(1, 8)]
    grip_u = [i for i in range(m.nu) if m.actuator(i).name.startswith("g_")][0]
    g_lo, g_hi = m.actuator_ctrlrange[grip_u]
    free = [m.body(m.jnt_bodyid[j]).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    q_start = h5py.File(os.path.join(edir, "trajectory.h5"))["observation/robot_state/joint_positions"][0]
    n_sub = int(round(1 / FPS / m.opt.timestep))
    lo, hi = m.actuator_ctrlrange[arm_u].T
    rows = []
    for rj in sorted(glob.glob(os.path.join(a.batch, "*", "v*", "rollout.json"))):
        r = json.load(open(rj))
        z = np.load(rj.replace(".json", ".npz"))
        ti = [i for i, t in enumerate(tasks) if t["text"] == r["instruction"]][0]
        task = tasks[ti]
        d = mujoco.MjData(m)
        d.qpos[arm_q] = q_start
        # the layout the rollout actually played: rollout.json records it from 10-20 on; earlier batches passed the expert
        # json's full seed as --seed (policy_batch bug, fixed 10-20), reproduced here with --batch_seed_from_json
        if "layout_rng_seed" in r:
            rs = r["layout_rng_seed"]
        elif a.batch_seed_from_json:
            vj = json.load(open(os.path.join(a.tasks, os.path.basename(os.path.dirname(os.path.dirname(rj))), f"v{r['variant']}.json")))
            rs = vj["seed"] * 1000 + ti * 100 + r["variant"]
        else:
            rs = a.seed * 1000 + ti * 100 + r["variant"]
        rng = np.random.default_rng(rs)
        if r["variant"] > 0:
            randomise(m, d, rng, free)
        d.ctrl[arm_u] = q_start
        d.ctrl[grip_u] = g_lo
        for _ in range(int(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        tb, rb = m.body(task["target"]).id, m.body(task["ref"]).id
        tgt, ref, grip = [], [], []
        for act in z["action"]:
            d.ctrl[arm_u] = np.clip(act[10:17], lo, hi)
            d.ctrl[grip_u] = g_hi if act[9] > 0.5 else g_lo
            for _ in range(n_sub):
                mujoco.mj_step(m, d)
            tgt.append(d.xpos[tb].copy())
            ref.append(d.xpos[rb].copy())
            grip.append(bool(act[9] > 0.5))
        tgt, ref, grip = np.array(tgt), np.array(ref), np.array(grip)
        rep_err = float(np.abs(tgt - z["target_pos"]).max())
        z0 = float(z["target_pos"][0][2])
        lifted = tgt[:, 2] - z0 >= LIFT_M
        row = dict(task=r["instruction"], type=task["type"], variant=r["variant"], success=bool(r["success"]),
                   replay_max_err_m=rep_err, reproduced=rep_err < 1e-6)
        if r["success"]:
            row["outcome"] = "success"
        elif not lifted.any():
            row["outcome"] = "not_lifted"
        else:
            last = int(np.nonzero(lifted)[0][-1])  # last step of the last carry
            s = last
            while s > 0 and lifted[s - 1]:
                s -= 1  # start of that carry
            opened = [k for k in range(s, len(grip)) if not grip[k]]
            k_open = opened[0] if opened else None
            if k_open is None:
                row["outcome"] = "never_released"
            elif k_open > last + 1:  # the target came down before the gripper opened
                row["outcome"] = "dropped_while_closed"
                k = last + 1
                row["drop_step"] = k
                row["drop_height_cm"] = float((tgt[last, 2] - z0) * 100)
            else:
                over, dist, lim = goal_region(m, d, fit, task, tgt[k_open], ref[k_open])
                row["outcome"] = "released_over_goal" if over else "released_away"
                row.update(release_step=int(k_open), release_height_cm=float((tgt[k_open, 2] - z0) * 100),
                           release_dist_cm=float(dist * 100), goal_limit_cm=float(lim * 100))
            row["carry_steps"] = int(last - s + 1)
        rows.append(row)
    out = dict(batch=a.batch, rows=rows, counts={}, status="exploratory, UNVERIFIED")
    for rw in rows:
        out["counts"][rw["outcome"]] = out["counts"].get(rw["outcome"], 0) + 1
    json.dump(out, open(os.path.join(a.batch, "placement_analysis.json"), "w"), indent=1)
    print("reproduced:", sum(r["reproduced"] for r in rows), "/", len(rows), " max replay error (m):", max(r["replay_max_err_m"] for r in rows))
    print("outcomes:", out["counts"])
    for rw in rows:
        if rw["outcome"] in ("success", "not_lifted"):
            continue
        extra = ""
        if "release_dist_cm" in rw:
            extra = f"released {rw['release_dist_cm']:.1f} cm from goal (limit {rw['goal_limit_cm']:.1f}), {rw['release_height_cm']:.1f} cm up"
        if "drop_height_cm" in rw:
            extra = f"dropped from {rw['drop_height_cm']:.1f} cm up after {rw['carry_steps']} steps held"
        print(f"  {rw['task'][:40]:40s} v{rw['variant']}: {rw['outcome']:20s} {extra}")


if __name__ == "__main__":
    main()
