"""Exterior camera pose for an episode from published calibrations (SceneGen_v1_gaps_1028 §7, §7b; user 10-28: use the
published calibrations now, adapt the method later).

Order, first match wins:
  1. DROID refined  data/real2sim/droid/calib/cam2base_extrinsics.json, then cam2base_extrinsic_superset.json
                    (DROID's post-release calibrations, ~36k / ~24k episodes; [x y z rx ry rz], cam -> base, euler xyz)
  2. PointWorld     data/real2sim/droid/calib/pointworld/cameras/<uuid>_cameras.json (NVIDIA PointWorld-DROID, ~43k
                    episodes; optimized_extrinsics stored base -> camera, inverted here; NVIDIA License, NON-COMMERCIAL
                    research/evaluation only: --no_pointworld excludes it)
  3. none           the caller falls back to calibrate_cam / camera_search / manual placement in inspect_scene.
Writes <out>/calib.json in calibrate_cam's format (R_cam_to_base, t_cam_in_base, K from intrinsics.json, frames,
principal_mode 'offset') with the provenance in 'source' and 'licence'.

    .venvs/real2sim/bin/python experiments/real2sim/camera_source.py --episode data/real2sim/droid/ep_c9 --cam 22008760 --out <scene>
Exploratory (res31, 10-28).
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_scene import cam_pose  # noqa: E402  [x y z rx ry rz] -> (R, t), cam -> base (DROID convention)
from calibrate_cam import zed_K, episode_id  # noqa: E402

CALIB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/real2sim/droid/calib")


def from_droid_refined(uuid, serial):
    for fn in ("cam2base_extrinsics.json", "cam2base_extrinsic_superset.json"):
        p = os.path.join(CALIB, fn)
        if not os.path.exists(p):
            continue
        e = json.load(open(p)).get(uuid)
        if e and serial in e:
            R, t = cam_pose(np.array(e[serial], float))
            return dict(R=R, t=t, source=f"DROID refined ({fn}; {e.get('source')} {e.get('metric_type')} {e.get('quality_metric')})",
                        quality=e.get("quality_metric"), licence="DROID dataset licence")
    return None


def from_pointworld(uuid, serial):
    p = os.path.join(CALIB, "pointworld", "cameras", f"{uuid}_cameras.json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    if serial not in j or "optimized_extrinsics" not in j[serial]:
        return None
    T = np.linalg.inv(np.array(j[serial]["optimized_extrinsics"], float))  # stored base -> camera (res32, 10-28)
    return dict(R=T[:3, :3], t=T[:3, 3], source="PointWorld-DROID optimized_extrinsics (inverted)", quality=None,
                licence="NVIDIA License: non-commercial research/evaluation only")


def camera_for(episode, serial, use_pointworld=True):
    uuid = episode_id(episode)
    return from_droid_refined(uuid, serial) or (from_pointworld(uuid, serial) if use_pointworld else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_pointworld", action="store_true", help="exclude PointWorld (non-commercial licence)")
    a = ap.parse_args()
    c = camera_for(a.episode, a.cam, not a.no_pointworld)
    if c is None:
        print(json.dumps(dict(found=False, episode=a.episode, cam=a.cam)))
        sys.exit(2)
    from episode_info import frames_arg
    calib = dict(cam=a.cam, K=zed_K(a.episode, a.cam).tolist(), R_cam_to_base=c["R"].tolist(), t_cam_in_base=c["t"].tolist(),
                 principal_mode="offset", frames=frames_arg(None, a.episode, "robot"), source=c["source"],
                 quality=c["quality"], licence=c["licence"])
    os.makedirs(a.out, exist_ok=True)
    json.dump(calib, open(os.path.join(a.out, "calib.json"), "w"), indent=1)
    print(json.dumps(dict(found=True, source=c["source"], licence=c["licence"])))


if __name__ == "__main__":
    main()
