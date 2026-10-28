"""Per-episode frame sets and facts, derived from the recording instead of hand-picked (scene creation v1, 10-26).

Until 10-26 every step had frame lists read off ep_orange_block (153 frames, first close at 75): 0,20,40 for stereo and
anchors, 30..130 for the exterior calibration and light, 18:73:3 for the wrist light, 0,40,63,75,100,126 for the review
page. On ep_c3 (1183 frames, 11 grasps) those lists cover only its first grasp. Here each set is a rule on the episode's
own length and recorded gripper events (gripper_position > 0.5 = closed, as scene_checks), so a new episode needs no
hand-picked frames. Explicit --frames flags in each script still override (old runs stay reproducible).

Rules (fc = first close frame, n = number of frames; checked against the hand-picked orange lists 10-26):
  before_touch  6 frames evenly over [0, 0.7 fc]      wrist anchors (orange hand list 0..50, c3 0..60)
  stereo        3 of those: [0, 0.3 fc, 0.55 fc]      stereo table / scene_checks #6 (orange hand list 0,20,40)
  approach      every 3rd frame over [0.25 fc, fc-3]  wrist light fit (orange hand list 18..72: identical)
  robot         7 frames evenly over [0.2 n, 0.85 n]  exterior calibration + light (orange hand list 30..130)
  review        0, 0.5 fc, fc-12, fc, mid-carry, first open   scene_review (orange hand list 0,40,63,75,100,126)
Writes <scene>/episode_info.json when called with a scene dir (a record of what was used). Exploratory, UNVERIFIED.

    .venvs/real2sim/bin/python experiments/real2sim/episode_info.py --episode data/real2sim/droid/ep_c3 [--scene runs/real2sim/ep_c3_stereo]
"""
import argparse, glob, json, os
import numpy as np
import h5py

CLOSED = 0.5  # recorded gripper_position above this = closed (same threshold as scene_checks)


def grasp_events(g, thr=CLOSED):
    """(close_frame, open_frame) pairs from the recorded gripper position."""
    ev, k = [], 0
    while k < len(g):
        if g[k] > thr:
            o = k + int(np.argmax(g[k:] < thr)) if (g[k:] < thr).any() else len(g) - 1
            ev.append((k, o))
            k = o + 1
        else:
            k += 1
    return ev


def _even(lo, hi, k):
    return sorted({int(round(x)) for x in np.linspace(lo, hi, k)})


def episode_info(episode):
    h = h5py.File(os.path.join(episode, "trajectory.h5"))
    g = h["observation/robot_state/gripper_position"][:]
    n = len(g)
    ev = grasp_events(g)
    fc = ev[0][0] if ev else n - 1  # no grasp: the whole episode counts as approach
    fo = ev[0][1] if ev else n - 1
    meta = sorted(glob.glob(os.path.join(episode, "metadata_*.json")))
    md = json.load(open(meta[0])) if meta else {}
    frames = dict(
        before_touch=_even(0, 0.7 * fc, 6),
        stereo=_even(0, 0.55 * fc, 3) if fc >= 6 else [0],
        approach=list(range(int(0.25 * fc), max(int(0.25 * fc) + 1, fc - 2), 3)),
        robot=_even(0.2 * n, 0.85 * n, 7),
        review=sorted({0, int(0.5 * fc), max(fc - 12, 0), fc, (fc + fo) // 2, fo}),
    )
    return dict(episode=episode, n_frames=n, grasp_events=ev, first_close=fc, first_open=fo, frames=frames,
                uuid=md.get("uuid"), robot_serial=md.get("robot_serial"), wrist_cam_serial=md.get("wrist_cam_serial"),
                ext1_cam_serial=md.get("ext1_cam_serial"), ext2_cam_serial=md.get("ext2_cam_serial"))


def frames_arg(value, episode, key):
    """A script's --frames value: explicit 'a,b,c' (or 'a:b:s') as given, else the derived set `key`."""
    if value:
        if ":" in value:
            lo, hi, st = (int(x) for x in value.split(":"))
            return list(range(lo, hi, st))
        return [int(x) for x in value.split(",")]
    fr = episode_info(episode)["frames"][key]
    print(f"frames ({key}, derived from the episode by episode_info.py): {fr}")
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--scene", default=None, help="write <scene>/episode_info.json")
    a = ap.parse_args()
    info = episode_info(a.episode)
    print(json.dumps(info, indent=1))
    if a.scene:
        json.dump(info, open(os.path.join(a.scene, "episode_info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
