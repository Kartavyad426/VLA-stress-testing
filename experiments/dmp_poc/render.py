"""Videos for the DMP PoC showcase page (docs/DMP_POC_METHOD.html §9a, file 10).

    GPU part (failure rollouts and corrective rollouts, by replaying recorded
    actions; run under the GPU lock and announce to prim):
      MUJOCO_GL=egl PYTHONPATH=<main>/third_party/LIBERO-plus \
      LIBERO_CONFIG_PATH=<main>/third_party/libero-plus-config \
      flock /tmp/vla_gpu.lock <main>/.venvs/libero-plus/bin/python experiments/dmp_poc/render.py sim
    CPU part (human demos, from the frames stored in the demo HDF5s):
      <main>/.venvs/libero-plus/bin/python experiments/dmp_poc/render.py demos

Every video is agentview | wrist side by side at 20 fps, rotated 180 degrees as
LeRobot's LiberoProcessorStep does, so it shows what the policy sees. Pixels
are never stored, so sim videos are REPLAYS: each one is checked against the
recorded EEF path and the deviation is written next to it (fidelity.json).

UNVERIFIED: new code, not yet checked by val.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from restore import WORKTREE, main_repo  # noqa: E402

sys.path.insert(0, os.path.join(main_repo(), "experiments"))
from visualise_episode_video import encode_video  # noqa: E402  (reused, not copied)

VID = os.path.join(WORKTREE, "runs/dmp_poc/videos")
T3 = os.path.join(WORKTREE, "runs/dmp_poc/t3")
OBS_SIZE = 256        # render size only; physics does not depend on it
FPS = 20
RENDER_ARMS = ("dmp", "linear")   # the page shows these as video; all four arms in its table


def _frames(env):
    px = env._raw.get("pixels", {})
    imgs = list(px.values()) if isinstance(px, dict) else list(px)
    return [np.asarray(im, dtype=np.uint8)[::-1, ::-1] for im in imgs]


def _replay(env, row, actions_after=None, t_root=None):
    """Reset as R-047, replay the logged actions (all, or up to t_root), then
    optional corrective actions. Returns per-camera frames and EEF path."""
    from restore import Action, spec_of
    steps = row["steps"]
    env.reset(seed=int(row["seed"]), spec=spec_of(row["perturbation"]))
    cams, eef = None, []

    def grab():
        nonlocal cams
        f = _frames(env)
        if cams is None:
            cams = [[] for _ in f]
        for i, im in enumerate(f):
            cams[i].append(im)
        eef.append(np.asarray(env._raw["robot_state"]["eef"]["pos"], dtype=float))

    grab()
    n_log = (len(steps) - 1) if t_root is None else t_root
    for t in range(n_log):
        env.step(Action(values=list(steps[t]["action"]), dims=list(row["action_dims"])))
        grab()
    for a in (actions_after if actions_after is not None else []):
        env.step(Action(values=[float(x) for x in a], dims=list(row["action_dims"])))
        grab()
    return cams, np.array(eef)


def render_sim(which=None):
    from restore import LiberoEnv, r047_rows, scene_of
    os.makedirs(VID, exist_ok=True)
    res = json.load(open(os.path.join(T3, "results.json")))["rows"]
    fid_path = os.path.join(VID, "fidelity.json")
    fid = json.load(open(fid_path)) if os.path.exists(fid_path) else {}
    envs = {}
    scen = sorted({(r["scenario"], r["row"], r["t_root"]) for r in res})
    for name, row_idx, t_root in scen:
        if which and name not in which:
            continue
        row = r047_rows()[row_idx]
        sc = scene_of(row)
        env = envs.setdefault(sc, LiberoEnv(suite="libero_spatial", task_id=sc, libero_plus=True,
                                            libero_plus_base_instruction=True, obs_size=OBS_SIZE))
        # 1) the failure rollout, whole episode
        out = os.path.join(VID, f"fail_{name}.mp4")
        if not os.path.exists(out):
            cams, eef = _replay(env, row)
            logged = np.array([s["obs_state"]["eef_pos"] for s in row["steps"]])
            n = min(len(eef), len(logged))
            fid[f"fail_{name}"] = {"max_dev_m": float(np.linalg.norm(eef[:n] - logged[:n], axis=1).max()), "frames": n}
            encode_video(cams, out)
            print("wrote", out, fid[f"fail_{name}"], flush=True)
        # 2) every corrective run of this scenario: logged prefix to the root, then its actions
        for r in [r for r in res if r["scenario"] == name and r["arm"] in RENDER_ARMS]:
            key = f"{name}_m{r['match_rank']}_{r['arm']}"
            out = os.path.join(VID, f"fix_{key}.mp4")
            if os.path.exists(out):
                continue
            z = np.load(os.path.join(T3, key + ".npz"))
            cams, eef = _replay(env, row, actions_after=z["act"], t_root=t_root)
            rec = z["eef"]
            # the recorded path is the grip site; the replay reads the eef observable (same site,
            # sampled one substep apart: ~0.5 mm, 10-14), so allow for that offset
            dev = float(np.linalg.norm(eef[t_root + 1:t_root + 1 + len(rec)] - rec, axis=1).max()) if len(rec) else 0.0
            fid[f"fix_{key}"] = {"max_dev_m": dev, "frames": len(eef), "root_frame": t_root}
            # keep only the root-onward part plus 20 frames of lead-in
            s = max(0, t_root - 20)
            encode_video([c[s:] for c in cams], out)
            fid[f"fix_{key}"]["first_frame"] = s
            print("wrote", out, fid[f"fix_{key}"], flush=True)
        with open(fid_path, "w") as f:
            json.dump(fid, f, indent=1)


def render_demos():
    """Matched demos, from the demo files' own 128x128 frames (no sim)."""
    import h5py
    from demos import demo_file
    os.makedirs(VID, exist_ok=True)
    match = json.load(open(os.path.join(WORKTREE, "runs/dmp_poc/match/scenarios.json")))["rows"]
    from restore import r047_rows
    from demos import task_name
    for m in match:
        task = task_name(r047_rows()[m["row"]]["instruction"])
        with h5py.File(demo_file(task), "r") as f:
            for mm in m["matches"]:
                out = os.path.join(VID, f"demo_{task[:40]}_{mm['demo']}.mp4")
                if os.path.exists(out):
                    continue
                g = f["data"][f"demo_{mm['demo']}"]["obs"]
                cams = [[im[::-1, ::-1] for im in g["agentview_rgb"][:]], [im[::-1, ::-1] for im in g["eye_in_hand_rgb"][:]]]
                encode_video(cams, out)
                print("wrote", out, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demos"
    if cmd == "sim":
        render_sim(sys.argv[2:] or None)
    else:
        render_demos()
