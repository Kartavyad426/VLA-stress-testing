"""Real-to-sim: run a pretrained DROID VLA closed loop on a make_tasks task in a rebuilt scene.

Policy: nvidia/GR00T-N1.7-DROID (embodiment oxe_droid_relative_eef_relative_joint), run the way NVIDIA's own DROID
client runs it (Isaac-GR00T examples/DROID/main_gr00t.py, read 10-15): 15 Hz; images resized (with pad) to 180x320;
one frame per camera (the checkpoint's video delta_indices are [0]); state = eef_9d + gripper_position +
joint_position (17-D); the 40-step chunk is decoded to absolute joint targets, the first --open_loop (15) steps are
executed, gripper binarised at 0.5.

  - exterior camera: composite.Compositor per frame (cleaned real plate + sim robot/objects + sim shadow ratio),
    rendered directly at 180x320, so the policy sees the real background, not a flat render;
  - wrist camera: composite.WristCompositor (sim robot/objects; background textured from the cleaned exterior plate,
    else from the cleaned real wrist frame 0 via wrist_plate);
  - eef_9d: FK of fr3_link7 + flange offset (== DROID's cartesian_position frame, checked 0 mm / 0 deg) -> DROID euler
    ('xyz', checked against DROID's own extrinsics) -> NVIDIA's compute_eef_9d ('XYZ' + DROID_EEF_ROTATION_CORRECT,
    copied from gr00t/data/state_action/droid_frame.py; checked: our episodes fall inside the checkpoint's q01-q99);
  - gripper state: Robotiq driver joint / 0.8 (0 open .. 1 closed, DROID's convention);
  - success: make_tasks.task_success after a 0.8 s settle (same predicate as the scripted expert).

Two processes: the sim side runs in .venvs/real2sim (the scenes are MuJoCo 3.15 .mjb; .venvs/groot has 3.3.2), the
policy runs in .venvs/groot as a child (`--serve`) and talks over stdin/stdout (pickled dicts).
`--policy hold` replaces GR00T with "hold the current joints" to test the sim side without a GPU.

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/policy_rollout.py \
        --scene runs/real2sim/ep_orange_block:data/real2sim/droid/ep_orange_block \
        --task "put the green block in the bowl" --variant 0 --out runs/real2sim/policy/smoke1

Writes <out>/rollout.mp4 (exterior composite | wrist, as the policy saw them), <out>/rollout.npz (state, action,
chunk timings), <out>/rollout.json (success, VRAM, settings). Exploratory; numbers UNVERIFIED.
"""
import argparse, json, os, pickle, struct, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FPS = 15
TAG = "oxe_droid_relative_eef_relative_joint"
DEFAULT_CKPT = os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-DROID/snapshots/05e7cc97e40dbd33b0890c35cc0214fcb0547ab5")
IMG = ("observation.images.exterior_image_1_left", "observation.images.wrist_image_left")
H, W = 180, 320  # NVIDIA's DROID client RESOLUTION

# copied from Isaac-GR00T gr00t/data/state_action/droid_frame.py (Apache-2.0); must stay identical to it
DROID_EEF_ROTATION_CORRECT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)


def compute_eef_9d(cartesian_position):
    from scipy.spatial.transform import Rotation
    c = np.asarray(cartesian_position, dtype=np.float64).reshape(6)
    rot = Rotation.from_euler("XYZ", c[3:6]).as_matrix() @ DROID_EEF_ROTATION_CORRECT
    return np.concatenate([c[:3], rot[:2, :].reshape(6)]).astype(np.float32)


# ---------------- pipe protocol: 4-byte length + pickle ----------------
def send(f, obj):
    b = pickle.dumps(obj, protocol=4)
    f.write(struct.pack("<I", len(b)) + b)
    f.flush()


def recv(f):
    n = f.read(4)
    if len(n) < 4:
        return None
    return pickle.loads(f.read(struct.unpack("<I", n)[0]))


# ---------------- policy side (.venvs/groot) ----------------
def serve(ckpt, device):
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # anything the libraries print goes to stderr, the pipe stays clean
    inp = sys.stdin.buffer
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors_from_pretrained

    feats = {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, H, W)) for k in IMG}
    feats["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(17,))
    # model_params_fp32=False + bf16 cast: fp32 parameters would need ~12 GB (cf. vla_harness _from_pretrained_cast)
    cfg = GrootConfig(base_model_path=ckpt, embodiment_tag=TAG, input_features=feats,
                      output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(17,))},
                      model_params_fp32=False, device="cpu")
    t0 = time.time()
    with torch.no_grad():
        policy = GrootPolicy(cfg)
    policy = policy.to(torch.bfloat16).to(device).eval()
    cfg.device = device
    pre, post = make_groot_pre_post_processors_from_pretrained(
        cfg, ckpt, preprocessor_overrides={"device_processor": {"device": device}})
    send(out, dict(ready=True, load_s=time.time() - t0, vram_alloc_gib=torch.cuda.max_memory_allocated() / 2**30))
    while True:
        req = recv(inp)
        if req is None:
            break
        t0 = time.time()
        batch = {k: torch.from_numpy(np.ascontiguousarray(req[k])).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0 for k in IMG}
        batch["observation.state"] = torch.from_numpy(req["state"].astype(np.float32)).unsqueeze(0).to(device)
        batch["task"] = [req["instruction"]]
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(pre(batch))
            act = post(chunk)  # full chunk while the pack step's raw state is cached: relative -> absolute
        act = act.detach().float().cpu().numpy()[0]
        send(out, dict(action=act, infer_s=time.time() - t0,
                       vram_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
                       vram_reserved_gib=torch.cuda.max_memory_reserved() / 2**30))


class GrootClient:
    def __init__(self, ckpt, python, device, timeout_s):
        self.p = subprocess.Popen([python, os.path.abspath(__file__), "--serve", "--ckpt", ckpt, "--device", device],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=REPO)
        self.ready = recv(self.p.stdout)
        if not self.ready:
            raise RuntimeError(f"policy server exited during load (rc {self.p.wait()})")

    def __call__(self, obs):
        send(self.p.stdin, obs)
        r = recv(self.p.stdout)
        if r is None:
            raise RuntimeError(f"policy server died (rc {self.p.wait()})")
        return r

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=60)


class HoldPolicy:  # sim-side test without a GPU: every chunk = hold the current joints, gripper open
    ready = dict(ready=True, load_s=0.0, vram_alloc_gib=0.0)

    def __call__(self, obs):
        s = obs["state"]
        return dict(action=np.tile(s, (40, 1)) * np.r_[np.ones(9), 0, np.ones(7)], infer_s=0.0)

    def close(self):
        pass


class ReplayPolicy:  # CPU regression: re-issue a recorded rollout's actions (rollout.npz) open loop, chunk by chunk
    ready = dict(ready=True, load_s=0.0, vram_alloc_gib=0.0)

    def __init__(self, npz, open_loop):
        self.acts, self.k, self.step = np.load(npz)["action"], 0, open_loop

    def __call__(self, obs):
        ch = self.acts[self.k:self.k + 40]
        if len(ch) < 40:
            ch = np.concatenate([ch, np.repeat(self.acts[-1:], 40 - len(ch), 0)])
        self.k += self.step
        return dict(action=ch, infer_s=0.0)

    def close(self):
        pass


def gpu_used_gib():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return float(r.stdout.split()[0]) / 1024
    except Exception:
        return float("nan")


# ---------------- sim side (.venvs/real2sim) ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="internal: run the policy server (in .venvs/groot)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scene", help="<scene out dir>:<episode dir>")
    ap.add_argument("--task", help="instruction (or a unique part of it) of a make_tasks task in this scene")
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="make_tasks --seed (variant seeds match make_tasks)")
    ap.add_argument("--out")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--open_loop", type=int, default=15, help="chunk steps executed before re-planning (NVIDIA client: 15)")
    ap.add_argument("--policy", choices=["groot", "hold", "replay"], default="groot")
    ap.add_argument("--replay_npz", default=None, help="--policy replay: rollout.npz whose actions are re-issued")
    ap.add_argument("--groot_python", default=os.path.join(REPO, ".venvs/groot/bin/python"))
    ap.add_argument("--wrist_render", choices=["v1", "v2"], default="v1",
                    help="v1: WristCompositor (default, as all runs before 10-20); v2: WristRenderV2 = wrist-video table texture "
                         "+ real-gripper library + render-and-shrink + wrist-fitted light (needs <scene>/gripper_lib, light_wrist.json)")
    a = ap.parse_args()
    if a.serve:
        return serve(a.ckpt, a.device)

    import cv2, h5py, imageio.v2 as imageio, mujoco
    from scipy.spatial.transform import Rotation as Rot
    sys.path.insert(0, HERE)
    from composite import Compositor, WristCompositor, WristRenderV2, load_plate, wrist_plate, wrist_table_texture, load_gripper_library
    from make_tasks import scene_tasks, task_success, randomise
    from build_scene import FLANGE_IN_LINK7

    sdir, edir = a.scene.split(":")
    m = mujoco.MjModel.from_binary_path(os.path.join(sdir, "scene.mjb"))
    fit = json.load(open(os.path.join(sdir, "scene_fit.json")))
    tasks = scene_tasks(m, fit)
    hits = [i for i, t in enumerate(tasks) if a.task in t["text"]]
    if len(hits) != 1:
        raise SystemExit(f"--task {a.task!r} matches {len(hits)} of {[t['text'] for t in tasks]}")
    ti = hits[0]
    task = tasks[ti]
    os.makedirs(a.out, exist_ok=True)

    # episode start exactly as make_tasks: DROID frame-0 joints, variant jitter, 0.5 s settle, gripper open
    arm_q = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    arm_u = [m.actuator(f"fr3_joint{k}").id for k in range(1, 8)]
    grip_u = [i for i in range(m.nu) if m.actuator(i).name.startswith("g_")][0]
    g_lo, g_hi = m.actuator_ctrlrange[grip_u]
    driver = m.joint("g_right_driver_joint")
    drv_q, drv_max = driver.qposadr[0], m.jnt_range[driver.id][1]
    link7 = m.body("fr3_link7").id  # DROID's EE frame = fr3_link7 + FLANGE_IN_LINK7, no rotation; NOT the
    # attachment_site, which build_scene rotates by --grip_yaw_deg (90) to mount the Robotiq
    free = [m.body(m.jnt_bodyid[j]).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    q_start = h5py.File(os.path.join(edir, "trajectory.h5"))["observation/robot_state/joint_positions"][0]
    d = mujoco.MjData(m)
    d.qpos[arm_q] = q_start
    rng = np.random.default_rng(a.seed * 1000 + ti * 100 + a.variant)
    if a.variant > 0 and not randomise(m, d, rng, free):
        raise SystemExit("could not place the variant's objects")
    d.ctrl[arm_u] = q_start
    d.ctrl[grip_u] = g_lo
    for _ in range(int(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    n_sub = int(round(1 / FPS / m.opt.timestep))
    lo, hi = m.actuator_ctrlrange[arm_u].T
    # checklist #17 / #23 logging (10-19): commanded vs achieved fingertip point and the target, per step
    pinch = m.site("g_pinch").id
    tgt_body = m.body(task["target"]).id
    tgt_z0 = float(d.xpos[tgt_body][2])
    dk = mujoco.MjData(m)  # FK of the commanded joints only
    pinch_cmd, pinch_ach, tgt_pos, grip_cmd = [], [], [], []

    plate = load_plate(sdir, "lama", "frame0+plate")
    comp = Compositor(m, sdir, plate, size=(H, W), verbose=False)  # first: sets the real object colours + light
    wp = wrist_plate(m, sdir, edir, comp, fit["wrist_camera"]["serial"])
    if a.wrist_render == "v2":
        g_rec = h5py.File(os.path.join(edir, "trajectory.h5"))["observation/robot_state/gripper_position"][:]
        k_close = int(np.argmax(g_rec > 0.5))  # mosaic from the frames before the first real grasp (as built in G5)
        tt = wrist_table_texture(m, sdir, edir, fit["wrist_camera"]["serial"], list(range(0, k_close - 2, 2)))
        wcomp = WristRenderV2(m, plate, comp, (H, W), wp, tt, load_gripper_library(os.path.join(sdir, "gripper_lib", "gripper_library.npz")),
                              light=json.load(open(os.path.join(sdir, "light_wrist.json"))))
    else:
        wcomp = WristCompositor(m, plate, comp, (H, W), plate2=wp)

    def observe():
        ext = comp(d).clip(0, 255).astype(np.uint8)
        wr = wcomp(d).clip(0, 255).astype(np.uint8)
        R = d.xmat[link7].reshape(3, 3)
        cart = np.r_[d.xpos[link7] + R @ FLANGE_IN_LINK7, Rot.from_matrix(R).as_euler("xyz")]  # DROID cartesian_position
        grip = np.clip(d.qpos[drv_q] / drv_max, 0, 1)
        state = np.r_[compute_eef_9d(cart), grip, d.qpos[arm_q]].astype(np.float32)
        return ext, wr, state, cart

    if a.policy == "hold":
        pol = HoldPolicy()
    elif a.policy == "replay":
        pol = ReplayPolicy(a.replay_npz, a.open_loop)
    else:
        pol = GrootClient(a.ckpt, a.groot_python, a.device, 600)
    print("policy ready:", pol.ready, flush=True)
    writer = imageio.get_writer(os.path.join(a.out, "rollout.mp4"), fps=FPS, quality=8, macro_block_size=4)
    states, actions, carts, infer_s, gpu_used = [], [], [], [], []
    chunk, k_in_chunk, vram = None, 0, {}
    n_steps = int(a.seconds * FPS)
    t_start = time.time()
    try:
        for step in range(n_steps):
            ext, wr, state, cart = observe()
            writer.append_data(np.concatenate([ext, wr], 1))
            if chunk is None or k_in_chunk >= a.open_loop:
                r = pol({IMG[0]: ext, IMG[1]: wr, "state": state, "instruction": task["text"]})
                chunk, k_in_chunk = r["action"], 0
                infer_s.append(r["infer_s"])
                vram = {k: v for k, v in r.items() if k.startswith("vram")}
                gpu_used.append(gpu_used_gib())
                if chunk.shape[-1] != 17:
                    raise RuntimeError(f"policy returned action dim {chunk.shape[-1]}, expected 17")
            act = chunk[k_in_chunk]
            k_in_chunk += 1
            q_tgt = act[10:17]  # [eef_9d(9), gripper(1), joint_position(7)], decoded to absolute
            d.ctrl[arm_u] = np.clip(q_tgt, lo, hi)
            d.ctrl[grip_u] = g_hi if act[9] > 0.5 else g_lo  # binarised as in NVIDIA's client
            for _ in range(n_sub):
                mujoco.mj_step(m, d)
            dk.qpos[:] = d.qpos
            dk.qpos[arm_q] = d.ctrl[arm_u]
            mujoco.mj_kinematics(m, dk)
            pinch_cmd.append(dk.site_xpos[pinch].copy())
            pinch_ach.append(d.site_xpos[pinch].copy())
            tgt_pos.append(d.xpos[tgt_body].copy())
            grip_cmd.append(bool(act[9] > 0.5))
            states.append(state)
            actions.append(act)
            carts.append(cart)
    finally:
        writer.close()
        pol.close()
    for _ in range(int(0.8 / m.opt.timestep)):  # let things settle before judging (as make_tasks)
        mujoco.mj_step(m, d)
    ok = task_success(m, d, fit, task)
    acts = np.array(actions)
    diag = rollout_diagnostics(np.array(pinch_cmd), np.array(pinch_ach), np.array(tgt_pos), np.array(grip_cmd), tgt_z0, bool(ok))
    res = dict(scene=os.path.basename(os.path.normpath(sdir)), task=task, instruction=task["text"], variant=a.variant,
               policy=a.policy, checkpoint=a.ckpt if a.policy == "groot" else None, replay_npz=a.replay_npz, seed=a.seed,
               layout_rng_seed=int(a.seed * 1000 + ti * 100 + a.variant), embodiment_tag=TAG, success=bool(ok),
               steps=len(actions), seconds=a.seconds, open_loop=a.open_loop, policy_calls=len(infer_s),
               load_s=pol.ready.get("load_s"), infer_s_median=float(np.median(infer_s)) if infer_s else None,
               wall_s=time.time() - t_start, vram=vram, gpu_used_max_gib=float(np.nanmax(gpu_used)) if gpu_used else None,
               joint_target_minus_ctrlrange_max=float(np.max(np.maximum(acts[:, 10:17] - hi, lo - acts[:, 10:17]))) if len(acts) else None,
               gripper_closed_frac=float((acts[:, 9] > 0.5).mean()) if len(acts) else None,
               final_target_xyz=d.xpos[m.body(task["target"]).id].tolist(), final_ref_xyz=d.xpos[m.body(task["ref"]).id].tolist(),
               wrist_view=("WristRenderV2 (wrist-video table, real-gripper library, render-and-shrink, wrist-fitted light)" if a.wrist_render == "v2"
                           else "textured (exterior plate + real wrist frame 0, composite.WristCompositor)"), wrist_render=a.wrist_render, exterior_view="composite (cleaned real plate + sim)",
               status="exploratory, UNVERIFIED", **diag)
    np.savez_compressed(os.path.join(a.out, "rollout.npz"), state=np.array(states), action=acts, cartesian=np.array(carts), infer_s=np.array(infer_s),
                        pinch_cmd=np.array(pinch_cmd), pinch_ach=np.array(pinch_ach), target_pos=np.array(tgt_pos), grip_cmd=np.array(grip_cmd))
    json.dump(res, open(os.path.join(a.out, "rollout.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("success", "steps", "policy_calls", "infer_s_median", "vram", "gpu_used_max_gib", "wall_s", "stages", "tracking")}), flush=True)


# checklist #17 (command tracking) and #23 (stage scoring); thresholds res31's, user-accepted 10-19 (reversible)
REACH_M = 0.03        # fingertip point within 3 cm of the target's centre
LIFT_M = 0.02         # target raised >= 2 cm above its start height
TRACK_M = 0.01        # commanded - achieved fingertip point
TRACK_SUSTAIN = FPS   # ... for >= 1 s of consecutive steps = "blocked"
STILL_MPS = 0.01      # ... while the achieved fingertip moves < 1 cm/s (10-19: without this, controller lag in fast
#                       moves and holding contacts were flagged; G2 medians 1.3-2.0 cm while moving)


def rollout_diagnostics(cmd, ach, tgt, grip_cmd, tgt_z0, success):
    """Stages reached -> closed -> lifted -> placed, and commanded-vs-achieved tracking (blocked arm = sim/geometry flag)."""
    if len(cmd) == 0:
        return dict(stages={}, tracking={})
    dist = np.linalg.norm(ach - tgt, axis=1)
    reached = dist < REACH_M
    closed = reached & grip_cmd
    lifted = (tgt[:, 2] - tgt_z0) >= LIFT_M
    first = lambda mask: int(np.argmax(mask)) if mask.any() else None  # noqa: E731
    stages = dict(reached=bool(reached.any()), closed_at_target=bool(closed.any()), lifted=bool(lifted.any()), placed=bool(success),
                  first_step=dict(reached=first(reached), closed_at_target=first(closed), lifted=first(lifted)),
                  min_dist_to_target_m=float(dist.min()), thresholds=dict(reach_m=REACH_M, lift_m=LIFT_M))
    err = cmd - ach
    en = np.linalg.norm(err, axis=1)
    speed = np.r_[0.0, np.linalg.norm(np.diff(ach, axis=0), axis=1) * FPS]
    over = (en > TRACK_M) & (speed < STILL_MPS)
    runs, cur, start = [], 0, 0
    for i, o in enumerate(np.r_[over, False]):
        if o:
            if cur == 0:
                start = i
            cur += 1
        elif cur:
            if cur >= TRACK_SUSTAIN:
                seg = slice(start, start + cur)
                runs.append(dict(start=start, steps=cur, mean_err_m=float(en[seg].mean()), mean_dz_cmd_minus_ach_m=float(err[seg, 2].mean())))
            cur = 0
    tracking = dict(max_err_m=float(en.max()), median_err_m=float(np.median(en)), blocked=bool(runs), blocked_segments=runs,
                    thresholds=dict(err_m=TRACK_M, sustain_steps=TRACK_SUSTAIN, still_mps=STILL_MPS))
    return dict(stages=stages, tracking=tracking)


if __name__ == "__main__":
    main()
