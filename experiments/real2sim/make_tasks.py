"""Real-to-sim: make language-conditioned tasks out of a rebuilt scene, solve them with a scripted expert, record data.

From a scene built by build_scene.py (scene.mjb + scene_fit.json) this
  1. enumerates tasks from the object inventory:
       put_in_bowl   "put the <colour> block in the bowl"          (each uniquely named block, plus radish/apple)
       stack         "stack the <colour> block on the <colour> block"
       next_to       "move the <object> next to the <object>"
     colours are named from each object's real median colour (perception masks);
  2. makes randomised variants ("cousins"): free objects jittered in x/y (+ yaw for blocks) without overlaps;
  3. runs a scripted expert (IK to waypoints from the sim's own object poses, physics on) and checks success from
     sim state (in bowl / stacked / within 3-12 cm on the table);
  4. records each episode: calibrated-camera video, joint states, actions (arm joint targets + gripper), instruction,
     success. Writes <out>/<scene>/<task>/<variant>.{mp4,npz,json} and <out>/summary.json.

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/make_tasks.py \
        --scene runs/real2sim/ep_c3:data/real2sim/droid/ep_c3 --scene runs/real2sim/ep_orange_block:data/real2sim/droid/ep_orange_block \
        --variants 5 --out runs/real2sim/tasks
"""
import argparse, colorsys, json, os, sys
import numpy as np
import cv2
import h5py
import imageio.v2 as imageio
import mujoco
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripted_grasp import ik  # noqa: E402  (damped least squares on the pinch site, verified on both scenes)

FPS = 15
HUES = [(15, "red"), (40, "orange"), (70, "yellow"), (165, "green"), (200, "teal"), (260, "blue"), (300, "purple"), (340, "pink"), (361, "red")]


def colour_name(rgb):
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    if v < 0.2:
        return "black"
    if s < 0.2:
        return "white" if v > 0.7 else "grey"
    return next(n for lim, n in HUES if h * 360 < lim)


def inventory(fit):
    """name every free object; blocks by colour, disambiguated left/right when two share a colour."""
    objs = []
    for lab, o in fit["objects"].items():
        if o["kind"] == "bowl":
            objs.append(dict(body=o["body"], kind="bowl", noun="bowl"))
        elif o["kind"] == "box":
            objs.append(dict(body=o["body"], kind="block", colour=colour_name(o["rgb"]), noun="block"))
        else:
            objs.append(dict(body=o["body"], kind="item", noun=o.get("asset", lab.split("#")[0]).replace("half_", "").replace("_", " ")))
    return objs


def describe(objs, d, m):
    """language phrase per object; same-colour blocks get 'left'/'right' (robot frame: +y is the robot's left)."""
    out = {}
    for o in objs:
        if o["kind"] != "block":
            out[o["body"]] = f"the {o['noun']}"
            continue
        same = [p for p in objs if p.get("colour") == o["colour"]]
        if len(same) == 1:
            out[o["body"]] = f"the {o['colour']} block"
        elif len(same) == 2:
            ys = sorted(same, key=lambda p: d.xpos[m.body(p["body"]).id][1])
            out[o["body"]] = f"the {o['colour']} block on the {'right' if ys[0] is o else 'left'}"
        else:
            out[o["body"]] = None  # three or more look-alikes: not uniquely nameable, skip as a target
    return out


def tasks_for(objs, names, max_stack=3, max_next=3):
    bowl = [o for o in objs if o["kind"] == "bowl"]
    movable = [o for o in objs if o["kind"] in ("block", "item") and names[o["body"]]]
    blocks = [o for o in movable if o["kind"] == "block"]
    t = []
    if bowl:
        for o in movable:
            t.append(dict(type="put_in_bowl", target=o["body"], ref=bowl[0]["body"], text=f"put {names[o['body']]} in the bowl"))
    for a, b in list(zip(blocks, blocks[1:]))[:max_stack]:
        t.append(dict(type="stack", target=a["body"], ref=b["body"], text=f"stack {names[a['body']]} on {names[b['body']]}"))
    others = [o for o in objs if names.get(o["body"])]
    for a, b in list(zip(movable[::-1], others))[:max_next]:
        if a["body"] != b["body"]:
            t.append(dict(type="next_to", target=a["body"], ref=b["body"], text=f"move {names[a['body']]} next to {names[b['body']]}"))
    return t


def free_qadr(m, body):
    j = [j for j in range(m.njnt) if m.jnt_bodyid[j] == m.body(body).id and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
    return m.jnt_qposadr[j]


def body_radius(m, body):
    """largest collision-geom bounding radius under this top-level body (retrieved meshes sit on a child body)"""
    root = m.body_rootid[m.body(body).id]
    return max(float(m.geom_rbound[g]) for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root and m.geom_contype[g])


def randomise(m, d, rng, free, jitter=0.03, yaw=0.5):
    """jitter free objects in x/y (+ yaw), rejecting overlaps (bounding spheres, horizontal)."""
    base = {b: d.qpos[free_qadr(m, b):free_qadr(m, b) + 7].copy() for b in free}
    rad = {b: 0.8 * body_radius(m, b) for b in free}
    for _ in range(200):
        prop = {}
        for b in free:
            q = base[b].copy()
            q[:2] += rng.uniform(-jitter, jitter, 2) * (0.6 if "bowl" in b else 1.0)
            if "block" in b:
                q[3:7] = (Rot.from_quat(q[3:7], scalar_first=True) * Rot.from_euler("z", rng.uniform(-yaw, yaw))).as_quat(scalar_first=True)
            prop[b] = q
        ok = all(np.linalg.norm(prop[a][:2] - prop[b][:2]) > rad[a] + rad[b] for i, a in enumerate(free) for b in free[i + 1:])
        if ok:
            for b, q in prop.items():
                d.qpos[free_qadr(m, b):free_qadr(m, b) + 7] = q
            return True
    return False


def footprint_radius(m, fit, body):
    """horizontal radius: the fitted rim for bowls, half the top-face diagonal for cubes, bounding sphere otherwise"""
    for o in fit["objects"].values():
        if o.get("body") == body and o["kind"] == "bowl":
            return o["rim_radius"]
    r = body_radius(m, body)
    return r * np.sqrt(2) / np.sqrt(3) if "block" in body else r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", action="append", required=True, help="<scene out dir>:<episode dir> (repeatable)")
    ap.add_argument("--variants", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", default="320x180", help="recorded video size WxH")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", action="append", default=[], help="run only tasks whose instruction contains this text (repeatable)")
    ap.add_argument("--variant_list", type=int, nargs="*", default=None, help="run only these variant indices")
    a = ap.parse_args()
    RW, RH = map(int, a.res.split("x"))
    summary = []
    for sc in a.scene:
        sdir, edir = sc.split(":")
        sname = os.path.basename(os.path.normpath(sdir))
        m = mujoco.MjModel.from_binary_path(os.path.join(sdir, "scene.mjb"))
        fit = json.load(open(os.path.join(sdir, "scene_fit.json")))
        q_start = h5py.File(os.path.join(edir, "trajectory.h5"))["observation/robot_state/joint_positions"][0]
        arm_q = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
        arm_dof = [m.joint(f"fr3_joint{k}").dofadr[0] for k in range(1, 8)]
        arm_u = [m.actuator(f"fr3_joint{k}").id for k in range(1, 8)]
        grip_u = [i for i in range(m.nu) if m.actuator(i).name.startswith("g_")][0]
        g_lo, g_hi = m.actuator_ctrlrange[grip_u]
        pinch = m.site("g_pinch").id
        free = [m.body(m.jnt_bodyid[j]).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        bowl_fit = [o for o in fit["objects"].values() if o["kind"] == "bowl"]
        rim_top = fit["table"]["z"] + bowl_fit[0]["height_asset_scaled"] if bowl_fit else None
        objs = inventory(fit)
        rend = mujoco.Renderer(m, RH, RW)
        n_sub = int(round(1 / FPS / m.opt.timestep))

        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        names = describe(objs, d0, m)
        tasks = tasks_for(objs, names)
        print(f"{sname}: {len(tasks)} tasks", [t["text"] for t in tasks], flush=True)

        for ti, task in enumerate(tasks):
            if a.only and not any(x in task["text"] for x in a.only):
                continue
            tdir = os.path.join(a.out, sname, f"{ti:02d}_{task['type']}_{task['target']}")
            os.makedirs(tdir, exist_ok=True)
            n_ok = 0
            for vi in (a.variant_list if a.variant_list else range(a.variants)):
                rng = np.random.default_rng(a.seed * 1000 + ti * 100 + vi)
                d = mujoco.MjData(m)
                d.qpos[arm_q] = q_start
                if vi > 0 and not randomise(m, d, rng, free):  # variant 0 = the reconstructed scene as is
                    continue
                d.ctrl[arm_u] = q_start
                d.ctrl[grip_u] = g_lo
                for _ in range(int(0.5 / m.opt.timestep)):
                    mujoco.mj_step(m, d)
                frames, qs, acts, full = [], [], [], []

                def run(waypoints, R):
                    q_prev = d.qpos[arm_q].copy()
                    for p, g, secs in waypoints:
                        q, _ = ik(m, q_prev, pinch, p, R, arm_q, arm_dof)
                        for k in range(int(secs * FPS)):
                            sm = (k + 1) / int(secs * FPS)
                            d.ctrl[arm_u] = q_prev + (q - q_prev) * (0.5 - 0.5 * np.cos(np.pi * sm))
                            d.ctrl[grip_u] = g
                            for _ in range(n_sub):
                                mujoco.mj_step(m, d)
                            rend.update_scene(d, camera="ext")
                            frames.append(rend.render().copy())
                            qs.append(np.r_[d.qpos[arm_q], d.ctrl[grip_u] / g_hi])
                            acts.append(np.r_[d.ctrl[arm_u], d.ctrl[grip_u] / g_hi])
                            full.append(d.qpos.copy())  # whole sim state, for composite.py
                        q_prev = q

                tb, rb = m.body(task["target"]).id, m.body(task["ref"]).id
                p_t = d.xpos[tb].copy()
                Rt = d.xmat[tb].reshape(3, 3)
                ax = np.array([Rt[0, 0], Rt[1, 0], 0.0])
                ax /= np.linalg.norm(ax) + 1e-9
                R_grasp = np.c_[np.cross(ax, [0, 0, -1.0]), ax, [0, 0, -1.0]]
                h_t = 2 * body_radius(m, task["target"]) / np.sqrt(3)  # cube side from its bounding sphere
                p_r = d.xpos[rb].copy()
                if task["type"] == "put_in_bowl":
                    place, drop = np.r_[p_r[:2], rim_top + 0.05], np.r_[p_r[:2], rim_top + 0.05]
                elif task["type"] == "stack":
                    top = p_r[2] + body_radius(m, task["ref"]) / np.sqrt(3)
                    place, drop = np.r_[p_r[:2], top + h_t / 2 + 0.06], np.r_[p_r[:2], top + h_t / 2 + 0.005]
                else:  # next_to: the free spot 9 cm from the reference that is farthest from everything else
                    others = [d.xpos[m.body(b).id][:2] for b in free if b not in (task["target"], task["ref"])]
                    lo_xy, hi_xy = np.array(fit["table"]["xy_lo"]) + 0.05, np.array(fit["table"]["xy_hi"]) - 0.05
                    off = footprint_radius(m, fit, task["ref"]) + footprint_radius(m, fit, task["target"]) + 0.035
                    cands = [p_r[:2] + off * np.array([np.cos(a_), np.sin(a_)]) for a_ in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
                    cands = [c for c in cands if np.all(c > lo_xy) and np.all(c < hi_xy)] or cands
                    c = max(cands, key=lambda c: min([np.linalg.norm(c - o) for o in others] + [1.0]))
                    place, drop = np.r_[c, p_t[2] + 0.06], np.r_[c, p_t[2] + 0.005]
                run([(p_t + [0, 0, 0.12], g_lo, 2.0), (p_t, g_lo, 1.5), (p_t, g_hi, 1.0), (p_t + [0, 0, 0.15], g_hi, 1.5),
                     (np.r_[place[:2], max(place[2], p_t[2]) + 0.10], g_hi, 2.0), (drop, g_hi, 1.0), (drop, g_lo, 0.8),
                     (np.r_[drop[:2], drop[2] + 0.15], g_lo, 1.2)], R_grasp)
                for _ in range(int(0.8 / m.opt.timestep)):  # let things settle before judging
                    mujoco.mj_step(m, d)
                pt, pr = d.xpos[tb], d.xpos[rb]
                horiz = float(np.linalg.norm(pt[:2] - pr[:2]))
                if task["type"] == "put_in_bowl":
                    ok = horiz < 0.8 * bowl_fit[0]["rim_radius"] and pt[2] < rim_top
                elif task["type"] == "stack":
                    ok = horiz < body_radius(m, task["ref"]) and pt[2] > pr[2] + 0.7 * (h_t / 2 + body_radius(m, task["ref"]) / np.sqrt(3))
                else:
                    gap = horiz - footprint_radius(m, fit, task["ref"]) - footprint_radius(m, fit, task["target"])
                    ok = -0.005 < gap < 0.08 and pt[2] < fit["table"]["z"] + 0.6 * h_t + 0.02  # beside it, on the table
                n_ok += bool(ok)
                base = os.path.join(tdir, f"v{vi}")
                w = imageio.get_writer(base + ".mp4", fps=FPS, quality=7, macro_block_size=4)  # 180 px divides by 4
                for fr in frames:
                    w.append_data(fr)
                w.close()
                np.savez_compressed(base + ".npz", state=np.array(qs), action=np.array(acts))
                np.savez_compressed(base + "_states.npz", qpos=np.array(full), real_frames=np.zeros(len(full), int))
                json.dump(dict(scene=sname, task=task, instruction=task["text"], variant=vi, seed=int(a.seed * 1000 + ti * 100 + vi),
                               success=bool(ok), steps=len(frames), final_target_xyz=pt.tolist(), final_ref_xyz=pr.tolist()),
                          open(base + ".json", "w"), indent=1)
            summary.append(dict(scene=sname, task=task["text"], type=task["type"], dir=os.path.relpath(tdir, a.out),
                                successes=n_ok, variants=a.variants))
            print(f"  [{n_ok}/{a.variants}] {task['text']}", flush=True)
        rend.close()
    if not a.only and not a.variant_list:  # partial reruns don't overwrite the full summary
        json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
