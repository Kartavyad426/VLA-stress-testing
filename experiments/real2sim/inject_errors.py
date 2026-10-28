"""Error injection for the image-physics checks (res31H's IMAGE_PHYSICS_CHECKS §5; prepared 10-26, NOT run: Q1 is open).

Builds copies of a scene with one known error each and runs the existing checks on every copy, so each check's
response to a known error can be compared with its noise floor (the unperturbed copy). Today only exterior-camera pose
errors are injected: the calibration is perturbed and the scene rebuilt from the same perception, which moves the
table and objects the way a bad calibration would. Table height, object shift and object size need build-time
overrides (scene_overrides.json, B8): patching the compiled .mjb broke collisions on 10-23 (stale BVH), so those wait.

Per copy: build_scene --depth stereo with the perturbed calib -> replay -> scene_checks -> block_size_check (masks).
Summary: <out>/injection_summary.json, one row per copy (table error, grasp/release offsets, scale, size ratios).
Default is a dry run that only lists the copies; --run executes (CPU, ~10-15 min per copy on orange).

    MUJOCO_GL=osmesa .venvs/real2sim/bin/python experiments/real2sim/inject_errors.py \
        --scene runs/real2sim/ep_orange_block_stereo --episode data/real2sim/droid/ep_orange_block --out runs/real2sim/v1_dev/injection [--run]
"""
import argparse, json, os, shutil, subprocess, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
# (name, kind, amount): ray = along the camera's optical axis (m), side = along its image x axis (m), pitch = rotation
# about its image x axis (deg). Sizes from IMAGE_PHYSICS_CHECKS §5.
INJECTIONS = [("none", "none", 0.0)] + \
    [(f"ray_{s}{v}cm", "ray", sgn * v / 100) for v in (1, 2, 5) for s, sgn in (("p", 1), ("m", -1))] + \
    [(f"side_{v}cm", "side", v / 100) for v in (1, 2, 5)] + \
    [(f"pitch_{v}deg", "pitch", float(v)) for v in (1, 3, 5)]
# runs a script with triton loaded before mujoco (MUJOCO_GL=osmesa segfault, SCENE_V1_MECHANISMS K5)
WRAP = ("import sys, os, runpy, triton; sys.argv = sys.argv[1:]; "
        "sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0]))); runpy.run_path(sys.argv[0], run_name='__main__')")


def perturb(calib, kind, amount):
    c = dict(calib)
    R, t = np.array(c["R_cam_to_base"]), np.array(c["t_cam_in_base"])
    if kind == "ray":
        t = t + R[:, 2] * amount
    elif kind == "side":
        t = t + R[:, 0] * amount
    elif kind == "pitch":
        R = R @ Rot.from_euler("x", amount, degrees=True).as_matrix()
    c.update(R_cam_to_base=R.tolist(), t_cam_in_base=t.tolist(), injected=dict(kind=kind, amount=amount))
    return c


def run(args, log):
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "osmesa"))
    with open(log, "a") as f:
        return subprocess.run([sys.executable, "-c", WRAP] + args, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def summarise(d):
    row = {}
    sc = os.path.join(d, "scene_checks.json")
    if os.path.exists(sc):
        j = json.load(open(sc))
        ev = [e for e in j["checks"].get("3_4_5_grasp_events", []) if "skipped" not in e]
        row.update(stereo_minus_built_m=j["checks"]["6_stereo_table"]["stereo_minus_built_m"],
                   grasp_horiz_m=[e["horiz_m"] for e in ev], grasp_vert_m=[e["vert_m"] for e in ev],
                   release_horiz_m=[e.get("release_horiz_m") for e in ev], scale_ratio=[e.get("scale_ratio") for e in ev],
                   settle_max_m=max(j["checks"]["1_settle"]["moved_m"].values()))
    bs = os.path.join(d, "block_size_check.json")
    if os.path.exists(bs):
        row["size_ratio"] = {k: v["lin_ratio_median"] for k, v in json.load(open(bs))["objects"].items()}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="source scene dir: perception.npz/.json and calib.json are reused")
    ap.add_argument("--episode", required=True)
    ap.add_argument("--cam", default=None, help="exterior serial; default: calib.json 'cam'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", action="append", default=[], help="run only copies whose name contains this (repeatable)")
    ap.add_argument("--run", action="store_true", help="execute (default: list the copies only)")
    a = ap.parse_args()
    calib = json.load(open(os.path.join(a.scene, "calib.json")))
    cam = a.cam or calib["cam"]
    todo = [x for x in INJECTIONS if not a.only or any(o in x[0] for o in a.only)]
    print(f"{len(todo)} copies:", [n for n, _, _ in todo])
    if not a.run:
        return
    os.makedirs(a.out, exist_ok=True)
    summary = {}
    for name, kind, amount in todo:
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        for fn in ("perception.npz", "perception.json"):
            if not os.path.exists(os.path.join(d, fn)):
                shutil.copy(os.path.join(a.scene, fn), d)
        json.dump(perturb(calib, kind, amount), open(os.path.join(d, "calib.json"), "w"), indent=1)
        log = os.path.join(d, "inject.log")
        if os.path.exists(os.path.join(d, "block_size_check.json")):  # finished on an earlier run: resumable after a stop
            summary[name] = dict(kind=kind, amount=amount, **summarise(d))
            print(name, "already done", flush=True)
            continue
        steps = [
            [os.path.join(HERE, "build_scene.py"), "--episode", a.episode, "--cam", cam, "--out", d, "--depth", "stereo", "--calib", os.path.join(d, "calib.json")],
            [os.path.join(HERE, "replay.py"), "--episode", a.episode, "--cam", cam, "--out", d],
            [os.path.join(HERE, "scene_checks.py"), "--scene", d, "--episode", a.episode, "--no_overlay"],
            [os.path.join(HERE, "block_size_check.py"), "--scene", d, "--episode", a.episode],
        ]
        for s in steps:
            rc = run(s, log)
            if rc != 0:
                print(f"{name}: {os.path.basename(s[0])} failed rc={rc} (see {log})")
                break
        summary[name] = dict(kind=kind, amount=amount, **summarise(d))
        json.dump(dict(scene=a.scene, episode=a.episode, status="exploratory, UNVERIFIED", copies=summary),
                  open(os.path.join(a.out, "injection_summary.json"), "w"), indent=1)
        print(name, json.dumps(summary[name])[:300], flush=True)


if __name__ == "__main__":
    main()
