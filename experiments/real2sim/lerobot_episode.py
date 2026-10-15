"""Real-to-sim: turn one episode of a LeRobot v3 dataset (e.g. Open X-Embodiment iamlab_cmu_pickup_insert, a Franka
Panda with a fixed third-person camera) into the episode folder the real2sim scripts read (DROID-style keys).

Writes <out>/<cam>.mp4 (the episode's frames, re-encoded at the dataset fps), <out>/trajectory.h5 and
<out>/metadata_<uuid>.json. Joint angles = observation.state[:7]. Gripper: the dataset's 1 = open / 0 = closed
(checked on iamlab_cmu episode 0) is flipped to the scripts' 1 = closed. There are no camera extrinsics: the
placeholder is all zeros and calibrate_cam.py --init_sphere finds the pose; intrinsics come from intrinsics.json
(written from MoGe's estimate by perceive.py's output, see --intrinsics_from).

    .venvs/lerobot/bin/python experiments/real2sim/lerobot_episode.py --dataset data/real2sim/oxe/iamlab_cmu_pickup_insert \
        --episode 0 --cam image --out data/real2sim/oxe/iamlab_cmu_pickup_insert/ep_000
"""
import argparse, json, os, subprocess
import numpy as np
import pandas as pd
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--cam", default="image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gripper_open_is", type=float, default=1.0, help="dataset value meaning 'open' (iamlab_cmu: 1)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    info = json.load(open(os.path.join(a.dataset, "meta/info.json")))
    fps = info["fps"]
    eps = pd.read_parquet(os.path.join(a.dataset, "meta/episodes/chunk-000/file-000.parquet"))
    e = eps[eps.episode_index == a.episode].iloc[0]
    key = f"videos/observation.images.{a.cam}"
    video = os.path.join(a.dataset, info["video_path"].format(video_key=f"observation.images.{a.cam}",
                                                              chunk_index=int(e[f"{key}/chunk_index"]),
                                                              file_index=int(e[f"{key}/file_index"])))
    n = int(e["length"])
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{float(e[f'{key}/from_timestamp']):.4f}", "-i", video,
                    "-frames:v", str(n), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-r", str(fps),
                    os.path.join(a.out, f"{a.cam}.mp4")], check=True)

    d = pd.read_parquet(os.path.join(a.dataset, "data/chunk-000/file-000.parquet"))
    d = d[d.episode_index == a.episode].sort_values("frame_index")
    s = np.stack(d["observation.state"].values).astype(np.float64)
    act = np.stack(d["action"].values).astype(np.float64)
    closed = lambda g: np.where(a.gripper_open_is > 0.5, 1.0 - g, g)  # -> 1 = closed
    uuid = f"{os.path.basename(os.path.normpath(a.dataset))}+ep{a.episode:06d}"
    with h5py.File(os.path.join(a.out, "trajectory.h5"), "w") as f:
        f["observation/robot_state/joint_positions"] = s[:, :7]
        f["observation/robot_state/gripper_position"] = closed(s[:, 7])
        f["action/target_gripper_position"] = closed(act[:, -1])
        f["observation/robot_state/cartesian_position"] = np.zeros((len(s), 6))  # not in the dataset
        f[f"observation/camera_extrinsics/{a.cam}_left"] = np.zeros((len(s), 6))  # unknown: found by calibration
    tasks = pd.read_parquet(os.path.join(a.dataset, "meta/tasks.parquet"))
    task = [t for t, row in tasks.iterrows() if int(row["task_index"]) == int(d["task_index"].iloc[0])]
    json.dump(dict(uuid=uuid, robot="panda", fps=fps, steps=len(s), task=task[0] if task else None, source=a.dataset,
                   episode=a.episode, ext1_cam_serial=a.cam),
              open(os.path.join(a.out, f"metadata_{uuid}.json"), "w"), indent=1)
    print(json.dumps(dict(uuid=uuid, steps=len(s), video_frames=n, fps=fps, task=task[0] if task else None,
                          gripper_closed_steps=int((closed(s[:, 7]) > 0.5).sum()))))


if __name__ == "__main__":
    main()
