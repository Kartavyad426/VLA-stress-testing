"""TaskGen MuJoCo layer: run a policy closed loop on a TaskGen task (docs/TASKGEN_DESIGN.html §4h).

Input: the newly made scene (scene config -> compiled model, via the Session), the task (task config, TaskGen's
success rules), and the policy (groot | hold | replay | expert: TaskGen's scripted expert in this same sim and
cameras, as proof that the task can be solved). Output: <out>/rollout.mp4 (exterior composite | wrist, as the
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
# wrist v2 photo/sim gripper blend: TRIAL from 2026-10-30 ~01:25 until the user's review on 10-31 morning (user via res31:
# "use this for any future rollouts till tom morning, revertible change ofc"). Revert: WRIST_BLEND = {} (the hard switch).
WRIST_BLEND = dict(open_ramp=0.05, w_smooth=0.3)
EXPERT_MAX_S = 120  # --policy expert: runs to the end of its steps (not --seconds), capped here


def need_assets(base_dir, wrist_render):
    missing = [f for f in ["plate_cache_lama_frame0_plate.png"] if not os.path.exists(os.path.join(base_dir, f))]
    if wrist_render == "v2":
        missing += [f for f in ("gripper_lib/gripper_library.npz", "light_wrist.json") if not os.path.exists(os.path.join(base_dir, f))]
    return missing


def compositors(session, m, wrist_render="v2", shadows=True, wrist_blur="gripper"):
    """The exterior + wrist compositors the policy sees (used by run_episode and by the chat's Composite button, so
    the two show the same images). m: the compiled TaskGen model."""
    import h5py
    import policy_rollout as PR
    from composite import Compositor, WristCompositor, WristRenderV2, load_plate, wrist_plate, wrist_table_texture, load_gripper_library
    sdir, edir, fit = session.cfg.base_scene, session.cfg.episode, session.fit
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
                              light=json.load(open(os.path.join(sdir, "light_wrist.json"))), near=wrist_blur,
                              **WRIST_BLEND)  # a fresh WristRenderV2 per run: no reset() needed
    else:
        wcomp = WristCompositor(m, plate, comp, (PR.H, PR.W), plate2=wp)
    # keep the base-model compositor alive as long as wcomp: under MUJOCO_GL=osmesa, freeing a Renderer after a newer one
    # exists corrupts the newer one's segmentation render (10-30, found by res31: IndexError in WristCompositor)
    wcomp._keep_alive = (mb, comp_b)
    return comp, wcomp


def run_episode(session, task, out, policy="hold", seconds=30.0, open_loop=15, wrist_render="v2", wrist_blur="gripper",
                shadows=True, replay_npz=None, device="cuda", ckpt=None, log=print):
    import cv2  # noqa: F401
    import imageio.v2 as imageio
    from scipy.spatial.transform import Rotation as Rot
    import policy_rollout as PR
    from build_scene import FLANGE_IN_LINK7
    from core import rules

    sdir, edir = session.cfg.base_scene, session.cfg.episode
    miss = need_assets(sdir, wrist_render)
    if miss:
        raise RuntimeError(f"base scene {sdir} lacks the composite assets {miss} (res31 10-29: only ep_orange_block_stereo has them)")
    os.makedirs(out, exist_ok=True)
    s = session.build()                     # the TaskGen scene: base + added objects, arm at frame 0
    m, d = s.m, s.d

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

    comp, wcomp = compositors(session, m, wrist_render, shadows, wrist_blur)

    def observe():
        ext = comp(d).clip(0, 255).astype(np.uint8)
        wr = wcomp(d).clip(0, 255).astype(np.uint8)
        R = d.xmat[link7].reshape(3, 3)
        cart = np.r_[d.xpos[link7] + R @ FLANGE_IN_LINK7, Rot.from_matrix(R).as_euler("xyz")]
        grip = np.clip(d.qpos[drv_q] / drv_max, 0, 1)
        state = np.r_[PR.compute_eef_9d(cart), grip, d.qpos[arm_q]].astype(np.float32)
        return ext, wr, state, cart

    def measure(cmd_q, closed):  # after each 1/FPS step: what rollout_diagnostics needs (commanded vs achieved pinch, pads)
        dk.qpos[:] = d.qpos
        dk.qpos[arm_q] = cmd_q
        mujoco.mj_kinematics(m, dk)
        pinch_cmd.append(dk.site_xpos[pinch].copy())
        pinch_ach.append(d.site_xpos[pinch].copy())
        tgt_pos.append(d.xpos[tgt_body].copy())
        grip_cmd.append(bool(closed))
        touching = set()
        for ci in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[ci].geom1], m.geom_bodyid[d.contact[ci].geom2]
            if tgt_body in (b1, b2):
                other = b2 if b1 == tgt_body else b1
                if other in pad_bodies:
                    touching.add("L" if other in left_pads else "R")
        both_pads.append(touching == {"L", "R"})

    ckpt = ckpt or PR.DEFAULT_CKPT
    if policy == "expert":  # TaskGen's scripted expert, in this same sim and cameras: proof the task can be solved
        from sim_mujoco.expert import Expert
        from core import experts as core_experts
        pol = type("ExpertMarker", (), dict(ready=dict(ready=True, load_s=0.0, vram_alloc_gib=0.0), close=lambda self: None))()
    elif policy == "hold":
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
    expert_log = None
    try:
        if policy == "expert":
            ex = Expert(s)

            def frame(cmd_q, grip_ctrl):
                ext, wr, state, cart = observe()
                writer.append_data(np.concatenate([ext, wr], 1))
                closed = grip_ctrl > (g_lo + g_hi) / 2
                measure(cmd_q, closed)
                act = np.zeros(17, np.float32)
                act[9], act[10:17] = float(closed), cmd_q
                states.append(state)
                actions.append(act)
                carts.append(cart)
                if len(states) % 75 == 0:
                    log(f"expert frame {len(states)}")
                if len(states) >= EXPERT_MAX_S * PR.FPS:
                    raise TimeoutError(f"expert ran past {EXPERT_MAX_S} s")
            ex.on_frame = frame
            expert_log = []
            try:
                for kind, obj, tg in core_experts.steps_for(task.pass_all):
                    if kind == "pick":
                        lg = ex.pick(obj)
                        expert_log.append(lg)
                        if not lg["held"]:
                            break
                    else:
                        expert_log.append(ex.move_release(core_experts.release_point(session, s, kind, obj, tg), label=f"{kind} {tg}"))
                ex._go(ex._fingertip(), 1.5)  # 1.5 s of still frames at the end (the settle, on video)
            except (ValueError, TimeoutError) as e:
                expert_log.append(dict(error=str(e)))
        else:
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
                measure(d.ctrl[arm_u].copy(), act[9] > 0.5)
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
               policy=policy, checkpoint=ckpt if policy == "groot" else None, embodiment_tag=PR.TAG, steps=len(actions),
               seconds=round(len(actions) / PR.FPS, 1) if policy == "expert" else seconds,
               open_loop=open_loop, policy_calls=len(infer_s), load_s=pol.ready.get("load_s"),
               infer_s_median=float(np.median(infer_s)) if infer_s else None, wall_s=time.time() - t_start, vram=vram,
               gpu_used_max_gib=float(np.nanmax(gpu_used)) if gpu_used and not np.all(np.isnan(gpu_used)) else None,
               gripper_closed_frac=float((acts[:, 9] > 0.5).mean()) if len(acts) else None,
               wrist_render=wrist_render, wrist_blur=wrist_blur if wrist_render == "v2" else None, shadows=shadows,
               wrist_blend=dict(WRIST_BLEND) if wrist_render == "v2" else None,
               exterior_view="composite (cleaned real plate of the base scene + sim; added objects drawn with their model colours)",
               expert_steps=expert_log, status="exploratory, UNVERIFIED", **diag)
    np.savez_compressed(os.path.join(out, "rollout.npz"), state=np.array(states), action=acts, cartesian=np.array(carts), infer_s=np.array(infer_s),
                        pinch_cmd=np.array(pinch_cmd), pinch_ach=np.array(pinch_ach), target_pos=np.array(tgt_pos), grip_cmd=np.array(grip_cmd),
                        both_pads=np.array(both_pads))
    json.dump(res, open(os.path.join(out, "rollout.json"), "w"), indent=1, default=str)
    return res
