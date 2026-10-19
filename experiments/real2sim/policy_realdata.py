"""Pre-deploy checklist #19: GR00T-N1.7-DROID on a REAL DROID episode (no sim): real exterior + wrist frames and the
recorded robot state in, predicted action chunks out, compared with what the operator did next.

Answers: does the policy close the gripper around the operator's grasp time, and does it move where the operator
moved? If yes on real data but not in our sim, the sim (visuals/geometry) is the suspect; if no here too, the
interface (state/gripper conventions) or the policy itself.

Inputs per query frame k (as policy_rollout.py builds them in sim): images = <ext>.mp4 / <wrist>.mp4 frame k resized
1280x720 -> 320x180 (INTER_AREA; same aspect, so no padding); state = compute_eef_9d(cartesian_position[k]) +
gripper_position[k] + joint_positions[k]. The chunk is decoded to absolute by the policy's post-processor (as in
policy_rollout). Compared over the first --horizon steps of the chunk with the recorded frames k+1..k+horizon.

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/policy_realdata.py \
        --episode data/real2sim/droid/ep_orange_block --out runs/real2sim/policy/realdata_orange \
        --instruction "put the orange block in the bowl"
"""
import argparse, json, os, sys, time
import numpy as np
import cv2
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from policy_rollout import GrootClient, compute_eef_9d, IMG, H, W, DEFAULT_CKPT, REPO  # noqa: E402


def frames(path):
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, bgr = cap.read()
        if not ok:
            return out
        out.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--instruction", action="append", required=True, help="repeatable")
    ap.add_argument("--ext", default="22008760")
    ap.add_argument("--wrist", default="18026681")
    ap.add_argument("--every", type=int, default=5, help="query every N frames")
    ap.add_argument("--horizon", type=int, default=15, help="chunk steps compared (NVIDIA client executes 15)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--groot_python", default=os.path.join(REPO, ".venvs/groot/bin/python"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    f = h5py.File(os.path.join(a.episode, "trajectory.h5"))
    cart = f["observation/robot_state/cartesian_position"][:]
    grip = f["observation/robot_state/gripper_position"][:]
    joints = f["observation/robot_state/joint_positions"][:]
    ext, wr = frames(os.path.join(a.episode, f"{a.ext}.mp4")), frames(os.path.join(a.episode, f"{a.wrist}.mp4"))
    n = min(len(ext), len(wr), len(cart))
    real_close = int(np.argmax(grip > 0.5)) if (grip > 0.5).any() else None
    print(f"{n} frames; recorded gripper first > 0.5 at frame {real_close}", flush=True)
    pol = GrootClient(a.ckpt, a.groot_python, "cuda", 600)
    print("policy ready:", pol.ready, flush=True)
    res = dict(episode=a.episode, n_frames=n, real_close_frame=real_close, every=a.every, horizon=a.horizon,
               image_prep="cv2 INTER_AREA 1280x720 -> 320x180", ready=pol.ready, runs={})
    try:
        for instr in a.instruction:
            rows, chunks = [], {}
            for k in range(0, n - 1, a.every):
                state = np.r_[compute_eef_9d(cart[k]), grip[k], joints[k]].astype(np.float32)
                r = pol({IMG[0]: ext[k], IMG[1]: wr[k], "state": state, "instruction": instr})
                act = r["action"]
                chunks[k] = act
                hz = min(a.horizon, n - 1 - k)
                fut = slice(k + 1, k + 1 + hz)
                pg = act[:hz, 9]  # predicted gripper (0 open .. 1 closed)
                ee_pred, ee_real = act[:hz, :3], cart[fut, :3]
                rows.append(dict(frame=k, infer_s=r["infer_s"], grip_state=float(grip[k]),
                                 pred_grip_max=float(pg.max()), pred_grip_first_close_step=int(np.argmax(pg > 0.5)) if (pg > 0.5).any() else None,
                                 real_grip_max_next=float(grip[fut].max()),
                                 ee_err_mean_m=float(np.linalg.norm(ee_pred - ee_real, axis=1).mean()),
                                 ee_err_end_m=float(np.linalg.norm(ee_pred[-1] - ee_real[-1])),
                                 joint_err_mean_rad=float(np.abs(act[:hz, 10:17] - joints[fut]).mean()),
                                 pred_dz_m=float(ee_pred[-1, 2] - cart[k, 2]), real_dz_m=float(ee_real[-1, 2] - cart[k, 2])))
                rw = rows[-1]
                print(f"[{instr[:28]}] f{k:3d} grip {rw['grip_state']:.2f} pred_grip_max {rw['pred_grip_max']:.2f} "
                      f"(close@{rw['pred_grip_first_close_step']}) real_next_max {rw['real_grip_max_next']:.2f} "
                      f"ee_err {rw['ee_err_mean_m']*100:.1f} cm dz pred {rw['pred_dz_m']*100:+.1f} real {rw['real_dz_m']*100:+.1f}", flush=True)
            pred_close = [rw["frame"] + rw["pred_grip_first_close_step"] for rw in rows if rw["pred_grip_first_close_step"] is not None]
            np.savez_compressed(os.path.join(a.out, f"chunks_{len(res['runs'])}.npz"), frames=np.array(sorted(chunks)), chunks=np.array([chunks[k] for k in sorted(chunks)]))
            res["runs"][instr] = dict(chunks_npz=f"chunks_{len(res['runs'])}.npz", rows=rows, first_pred_close_frame=min(pred_close) if pred_close else None,
                                      ee_err_mean_m=float(np.mean([rw["ee_err_mean_m"] for rw in rows])))
    finally:
        pol.close()
    json.dump(res, open(os.path.join(a.out, "realdata.json"), "w"), indent=1)
    for instr, rr in res["runs"].items():
        print(f"SUMMARY [{instr}]: first predicted close at frame {rr['first_pred_close_frame']} (real {real_close}); "
              f"mean EE error over {a.horizon} steps {rr['ee_err_mean_m']*100:.1f} cm")


if __name__ == "__main__":
    main()
