"""TaskGen MuJoCo layer: run a policy closed loop on a TaskGen task (docs/TASKGEN_DESIGN.html §4h).

Input: the newly made scene (scene config -> compiled model, via the Session), the task (task config, TaskGen's
success rules), and the policy (groot | hold | replay). Output: <out>/rollout.mp4 (exterior composite | wrist, as the
policy saw them), rollout.npz, rollout.json (success by TaskGen's rules, stage scores, timings, VRAM).

The episode loop is COPIED from res31's experiments/real2sim/policy_rollout.py main() (10-29; res31: "copy main()'s
episode block ... and import the pieces"), changed only where TaskGen differs: the model is the compiled TaskGen
scene, the composite assets come from its base scene folder, success = core.rules on the final state, the stage
scores are measured on the first pass condition's object. Imported read-only from res31: GrootClient, HoldPolicy,
ReplayPolicy, compute_eef_9d, rollout_diagnostics, gpu_used_gib, FPS/IMG/H/W/TAG/DEFAULT_CKPT; composite.*;
build_scene.FLANGE_IN_LINK7. Needs the base scene's composite assets (cleaned plate; for wrist v2 also gripper_lib/
and light_wrist.json): res31 (10-29): only runs/real2sim/ep_orange_block_stereo has them, the v1 builds don't.
GPU only for --policy groot (prim's gate). Exploratory, UNVERIFIED.
"""
import json, os, time
import numpy as np
import mujoco

REPO = "/home/imerit/Documents/Code/VLA"


def need_assets(base_dir, wrist_render):
    missing = [f for f in ["plate_cache_lama_frame0_plate.png"] if not os.path.exists(os.path.join(base_dir, f))]
    if wrist_render == "v2":
        missing += [f for f in ("gripper_lib/gripper_library.npz", "light_wrist.json") if not os.path.exists(os.path.join(base_dir, f))]
    return missing


def run_episode(session, task, out, policy="hold", seconds=30.0, open_loop=15, wrist_render="v2", wrist_blur="gripper",
                shadows=True, replay_npz=None, device="cuda", ckpt=None, log=print):
    import cv2  # noqa: F401
    import h5py
    import imageio.v2 as imageio
    from scipy.spatial.transform import Rotation as Rot
    import policy_rollout as PR
    from composite import Compositor, WristCompositor, WristRenderV2, load_plate, wrist_plate, wrist_table_texture, load_gripper_library
    from build_scene import FLANGE_IN_LINK7
    from core import rules

    sdir, edir = session.cfg.base_scene, session.cfg.episode
    miss = need_assets(sdir, wrist_render)
    if miss:
        raise RuntimeError(f"base scene {sdir} lacks the composite assets {miss} (res31 10-29: only ep_orange_block_stereo has them)")
    os.makedirs(out, exist_ok=True)
    s = session.build()                     # the TaskGen scene: base + added objects, arm at frame 0
    m, d = s.m, s.d
    fit = session.fit

    # episode start as policy_rollout / make_tasks: DROID frame-0 joints (set by build), 0.5 s settle, gripper open
    arm_q = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    arm_u = [m.actuator(f"fr3_joint{k}").id for k in range(1, 8)]
    grip_u = [i for i in range(m.nu) if m.actuator(i).name.startswith("g_")][0]
    g_lo, g_hi = m.actuator_ctrlrange[grip_u]
    driver = m.joint("g_right_driver_joint")
    drv_q, drv_max = driver.qposadr[0], m.jnt_range[driver.id][1]
    link7 = m.body("fr3_link7").id
    d.ctrl[grip_u] = g_lo
    for _ in range(int(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    n_sub = int(round(1 / PR.FPS / m.opt.timestep))
    lo, hi = m.actuator_ctrlrange[arm_u].T
    pinch = m.site("g_pinch").id
    target = task.pass_all[0].object or (task.pass_all[0].order or [None, None])[1] or task.targets[0]
    tgt_body = m.body(session._body(target)).id
    tgt_z0 = float(d.xpos[tgt_body][2])
    dk = mujoco.MjData(m)
    pinch_cmd, pinch_ach, tgt_pos, grip_cmd, both_pads = [], [], [], [], []
    pad_bodies = {b for b in range(m.nbody) if "pad" in m.body(b).name}
    left_pads = {b for b in pad_bodies if "left" in m.body(b).name}

    plate = load_plate(sdir, "lama", "frame0+plate")
    comp = Compositor(m, sdir, plate, shadows=shadows, size=(PR.H, PR.W), verbose=False)
    # the wrist textures come from the REAL footage: build them with the base model (res31's scene.mjb, which matches
    # its replay_states.npz and has no added objects to mask out); only the rendering uses the TaskGen model.
    # 10-29: with the TaskGen model, wrist_plate failed (replay qpos 43 vs model 53).
    mb = mujoco.MjModel.from_binary_path(os.path.join(sdir, "scene.mjb"))
    comp_b = Compositor(mb, sdir, plate, shadows=shadows, size=(PR.H, PR.W), verbose=False)
    wp = wrist_plate(mb, sdir, edir, comp_b, fit["wrist_camera"]["serial"])
    if wrist_render == "v2":
        g_rec = h5py.File(os.path.join(edir, "trajectory.h5"))["observation/robot_state/gripper_position"][:]
        k_close = int(np.argmax(g_rec > 0.5))
        tt = wrist_table_texture(mb, sdir, edir, fit["wrist_camera"]["serial"], list(range(0, k_close - 2, 2)))
        wcomp = WristRenderV2(m, plate, comp, (PR.H, PR.W), wp, tt, load_gripper_library(os.path.join(sdir, "gripper_lib", "gripper_library.npz")),
                              light=json.load(open(os.path.join(sdir, "light_wrist.json"))), near=wrist_blur)
    else:
        wcomp = WristCompositor(m, plate, comp, (PR.H, PR.W), plate2=wp)

    def observe():
        ext = comp(d).clip(0, 255).astype(np.uint8)
        wr = wcomp(d).clip(0, 255).astype(np.uint8)
        R = d.xmat[link7].reshape(3, 3)
        cart = np.r_[d.xpos[link7] + R @ FLANGE_IN_LINK7, Rot.from_matrix(R).as_euler("xyz")]
        grip = np.clip(d.qpos[drv_q] / drv_max, 0, 1)
        state = np.r_[PR.compute_eef_9d(cart), grip, d.qpos[arm_q]].astype(np.float32)
        return ext, wr, state, cart

    ckpt = ckpt or PR.DEFAULT_CKPT
    if policy == "hold":
        pol = PR.HoldPolicy()
    elif policy == "replay":
        pol = PR.ReplayPolicy(replay_npz, open_loop)
    else:
        pol = PR.GrootClient(ckpt, os.path.join(REPO, ".venvs/groot/bin/python"), device, 600)
    log(f"policy ready: {pol.ready}")
    writer = imageio.get_writer(os.path.join(out, "rollout.mp4"), fps=PR.FPS, quality=8, macro_block_size=4)
    states, actions, carts, infer_s, gpu_used = [], [], [], [], []
    chunk, k_in_chunk, vram = None, 0, {}
    n_steps = int(seconds * PR.FPS)
    t_start = time.time()
    try:
        for step in range(n_steps):
            ext, wr, state, cart = observe()
            writer.append_data(np.concatenate([ext, wr], 1))
            if chunk is None or k_in_chunk >= open_loop:
                r = pol({PR.IMG[0]: ext, PR.IMG[1]: wr, "state": state, "instruction": task.instruction})
                chunk, k_in_chunk = r["action"], 0
                infer_s.append(r["infer_s"])
                vram = {k: v for k, v in r.items() if k.startswith("vram")}
                gpu_used.append(PR.gpu_used_gib())
                if chunk.shape[-1] != 17:
                    raise RuntimeError(f"policy returned action dim {chunk.shape[-1]}, expected 17")
            act = chunk[k_in_chunk]
            k_in_chunk += 1
            d.ctrl[arm_u] = np.clip(act[10:17], lo, hi)
            d.ctrl[grip_u] = g_hi if act[9] > 0.5 else g_lo
            for _ in range(n_sub):
                mujoco.mj_step(m, d)
            dk.qpos[:] = d.qpos
            dk.qpos[arm_q] = d.ctrl[arm_u]
            mujoco.mj_kinematics(m, dk)
            pinch_cmd.append(dk.site_xpos[pinch].copy())
            pinch_ach.append(d.site_xpos[pinch].copy())
            tgt_pos.append(d.xpos[tgt_body].copy())
            grip_cmd.append(bool(act[9] > 0.5))
            touching = set()
            for ci in range(d.ncon):
                b1, b2 = m.geom_bodyid[d.contact[ci].geom1], m.geom_bodyid[d.contact[ci].geom2]
                if tgt_body in (b1, b2):
                    other = b2 if b1 == tgt_body else b1
                    if other in pad_bodies:
                        touching.add("L" if other in left_pads else "R")
            both_pads.append(touching == {"L", "R"})
            states.append(state)
            actions.append(act)
            carts.append(cart)
            if step % 75 == 0:
                log(f"step {step}/{n_steps}")
    finally:
        writer.close()
        pol.close()
    for _ in range(int(0.8 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    ok, rule_rows = rules.check_all(task.pass_all, session, s)
    acts = np.array(actions)
    diag = PR.rollout_diagnostics(np.array(pinch_cmd), np.array(pinch_ach), np.array(tgt_pos), np.array(grip_cmd), tgt_z0, bool(ok),
                                  both_pads=np.array(both_pads))
    res = dict(task_id=task.id, instruction=task.instruction, pass_all=[c.model_dump() for c in task.pass_all], rules=rule_rows,
               success=bool(ok), stage_target=target, scene_config=task.scene_config, base_scene=sdir, base_version=session.cfg.base_version,
               policy=policy, checkpoint=ckpt if policy == "groot" else None, embodiment_tag=PR.TAG, steps=len(actions), seconds=seconds,
               open_loop=open_loop, policy_calls=len(infer_s), load_s=pol.ready.get("load_s"),
               infer_s_median=float(np.median(infer_s)) if infer_s else None, wall_s=time.time() - t_start, vram=vram,
               gpu_used_max_gib=float(np.nanmax(gpu_used)) if gpu_used and not np.all(np.isnan(gpu_used)) else None,
               gripper_closed_frac=float((acts[:, 9] > 0.5).mean()) if len(acts) else None,
               wrist_render=wrist_render, wrist_blur=wrist_blur if wrist_render == "v2" else None, shadows=shadows,
               exterior_view="composite (cleaned real plate of the base scene + sim; added objects drawn with their model colours)",
               status="exploratory, UNVERIFIED", **diag)
    np.savez_compressed(os.path.join(out, "rollout.npz"), state=np.array(states), action=acts, cartesian=np.array(carts), infer_s=np.array(infer_s),
                        pinch_cmd=np.array(pinch_cmd), pinch_ach=np.array(pinch_ach), target_pos=np.array(tgt_pos), grip_cmd=np.array(grip_cmd),
                        both_pads=np.array(both_pads))
    json.dump(res, open(os.path.join(out, "rollout.json"), "w"), indent=1, default=str)
    return res
