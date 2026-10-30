"""One command: build every file the composite renders need on a built scene (res31 10-30; user: "can you build the 1
command builder for composite assets, talk to TaskGen for more context"). For TaskGen's chat Build button and
rollouts: Compositor (exterior) + WristRenderV2 (wrist v2). Steps, in dependency order, each skipped when its outputs
exist (idempotent; --force rebuilds):

  1 plate        cleaned exterior plate, LaMa ("lama", "frame0+plate")  plate_cache_lama_frame0_plate.png, plate_clean_lama.png
  2 light        exterior light fit to the robot's shadow (calibrate_light.py)  light.json, light_check.png
  3 wrist_plate  wrist frame-0 plate (composite.wrist_plate; reads light.json)  plate_cache_wrist_frame0.png/.json
  4 table_tex    wrist table texture, frames 0..first-grasp-2 step 2 (as policy_rollout / TaskGen)  table_tex_wrist.npz/.png
  5 gripper_lib  real open-gripper library from this episode (gripper_library.py, --band 161 as the orange build)
                 gripper_lib/gripper_library.npz/.json, preview.jpg
  6 light_wrist  wrist-view light fit (light_fit_wrist.py)  light_wrist.json, light_wrist_check.jpg

Needs a scene built by build_v1 (scene.mjb, scene_fit.json, perception.npz, calib.json, replay_states.npz) and its
episode. All steps run on CPU (MUJOCO_GL=osmesa, no CUDA; LaMa runs on CPU in composite.lama_inpaint). Checks (exit 1
if any fails) use PROVISIONAL thresholds set from one reference scene (ep_orange_block_stereo: light IoU 0.44, wrist
light IoU 0.21, gripper IoU 0.76, 88 open frames): files present, light IoU >= 0.15, wrist light IoU >= 0.10, gripper
library IoU vs the sim gripper >= 0.60 with >= 20 open frames, table texture observed >= 30 % of its area. A human
still has to look at the check images. Writes <scene>/composite_assets.json (per step: status, seconds, cpu/gpu, files
written; checks; check images; peak RAM; disk written; ok) and prints its path as the last line.

    runs/real2sim/v1_dev/py_triton_first.sh experiments/real2sim/build_composite_assets.py <scene_dir> <episode_dir> [--force]
Exploratory, UNVERIFIED.
"""
import argparse, json, os, resource, subprocess, sys, time
import triton  # noqa: F401  (before mujoco, as py_triton_first.sh)
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
PY = [os.path.join(REPO, ".venvs/real2sim/bin/python"), "-c",
      "import sys, os, runpy, triton; sys.argv = sys.argv[1:]; sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0]))); "
      "runpy.run_path(sys.argv[0], run_name='__main__')"]
ENV = dict(os.environ, MUJOCO_GL="osmesa", CUDA_VISIBLE_DEVICES="")
# provisional thresholds (one reference scene, see the docstring)
LIGHT_IOU, WRIST_LIGHT_IOU, GRIP_IOU, GRIP_FRAMES, TEX_OBSERVED = 0.15, 0.10, 0.60, 20, 0.30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("episode")
    ap.add_argument("--force", action="store_true", help="rebuild every step even if its outputs exist")
    a = ap.parse_args()
    S, E = os.path.abspath(a.scene), os.path.abspath(a.episode)
    os.environ.update(MUJOCO_GL="osmesa", CUDA_VISIBLE_DEVICES="")
    rep = dict(scene=S, episode=E, created=time.strftime("%Y-%m-%d %H:%M"), status="exploratory, UNVERIFIED", steps=[], checks={},
               check_images=[], thresholds=dict(light_iou=LIGHT_IOU, wrist_light_iou=WRIST_LIGHT_IOU, gripper_iou=GRIP_IOU,
                                                gripper_frames=GRIP_FRAMES, table_tex_observed=TEX_OBSERVED, provisional=True))
    out_json = os.path.join(S, "composite_assets.json")
    need = ["scene.mjb", "scene_fit.json", "perception.npz", "calib.json", "replay_states.npz"]
    missing = [f for f in need if not os.path.exists(os.path.join(S, f))]
    if missing or not os.path.exists(os.path.join(E, "trajectory.h5")):
        rep.update(ok=False, error=f"not a built scene/episode: missing {missing or ['trajectory.h5 in the episode']}")
        json.dump(rep, open(out_json, "w"), indent=1)
        print(out_json)
        sys.exit(1)
    calib, fit = json.load(open(os.path.join(S, "calib.json"))), json.load(open(os.path.join(S, "scene_fit.json")))
    cam, ws = calib["cam"], fit["wrist_camera"]["serial"]
    G = h5py.File(os.path.join(E, "trajectory.h5"))["observation/robot_state/gripper_position"][:]
    k_close = int(np.argmax(G > 0.5))
    p = lambda *f: os.path.join(S, *f)

    def step(name, outputs, fn, images=()):
        have = all(os.path.exists(p(f)) for f in outputs)
        r = dict(step=name, device="cpu", outputs=outputs)
        if have and not a.force:
            r.update(status="skipped (outputs exist)", seconds=0.0)
        else:
            t0 = time.time()
            try:
                fn()
                r["status"] = "done" if all(os.path.exists(p(f)) for f in outputs) else "failed: outputs missing"
            except Exception as e:  # noqa: BLE001  (reported, then exit 1)
                r["status"] = f"failed: {type(e).__name__}: {e}"
            r["seconds"] = round(time.time() - t0, 1)
        rep["steps"].append(r)
        rep["check_images"] += [p(f) for f in images if os.path.exists(p(f))]
        print(f"{name}: {r['status']} ({r['seconds']} s)", flush=True)
        json.dump(rep, open(out_json, "w"), indent=1)
        return r["status"] in ("done", "skipped (outputs exist)")

    def sub(script, *args):
        log = open(p("composite_assets.log"), "a")
        log.write(f"\n=== {time.strftime('%H:%M:%S')} {script} {' '.join(args)}\n")
        log.flush()
        subprocess.run(PY + [os.path.join(HERE, script), *args], check=True, env=ENV, stdout=log, stderr=subprocess.STDOUT)

    # 1 cleaned exterior plate (LaMa on CPU)
    def plate():
        import cv2
        from composite import load_plate
        from replay import read_video
        pl = load_plate(S, "lama", "frame0+plate", read_video(os.path.join(E, f"{cam}.mp4")), dict(np.load(p("perception.npz"))))
        cv2.imwrite(p("plate_clean_lama.png"), cv2.cvtColor(pl.astype(np.uint8), cv2.COLOR_RGB2BGR))
    ok = step("plate", ["plate_cache_lama_frame0_plate.png", "plate_clean_lama.png"], plate, ["plate_clean_lama.png"])
    # 2 exterior light
    ok = ok and step("light", ["light.json", "light_check.png"], lambda: sub("calibrate_light.py", "--episode", E, "--cam", cam, "--out", S),
                     ["light_check.png"])

    # 3 + 4 wrist plate and table texture (one model; the Compositor's renderer stays alive until both are done:
    # freeing an older mujoco.Renderer while a newer one exists breaks the newer one under osmesa, 10-30)
    def wrist_assets(what):
        import mujoco
        from composite import Compositor, load_plate, wrist_plate, wrist_table_texture
        m = mujoco.MjModel.from_binary_path(p("scene.mjb"))
        comp = Compositor(m, S, load_plate(S, "lama", "frame0+plate"), size=(180, 320), verbose=False)
        if what == "plate":
            wrist_plate(m, S, E, comp, ws)
        else:
            wrist_table_texture(m, S, E, ws, list(range(0, k_close - 2, 2)))
        del comp
    if a.force:
        for f in ("plate_cache_wrist_frame0.png", "plate_cache_wrist_frame0.json", "table_tex_wrist.npz"):
            if os.path.exists(p(f)):
                os.remove(p(f))  # both functions return their cache when it exists
    ok = ok and step("wrist_plate", ["plate_cache_wrist_frame0.png"], lambda: wrist_assets("plate"), ["plate_cache_wrist_frame0.png"])
    ok = ok and step("table_tex", ["table_tex_wrist.npz", "table_tex_wrist.png"], lambda: wrist_assets("tex"), ["table_tex_wrist.png"])
    # 5 gripper library from this episode's open frames
    ok = ok and step("gripper_lib", ["gripper_lib/gripper_library.npz", "gripper_lib/gripper_library.json"],
                     lambda: sub("gripper_library.py", "--scene", S, "--episodes", E, "--band", "161", "--out", p("gripper_lib")),
                     ["gripper_lib/preview.jpg"])
    # 6 wrist light
    ok = ok and step("light_wrist", ["light_wrist.json", "light_wrist_check.jpg"], lambda: sub("light_fit_wrist.py", "--scene", S, "--episode", E),
                     ["light_wrist_check.jpg"])

    # checks
    J = lambda f: json.load(open(p(f))) if os.path.exists(p(f)) else {}
    lj, lw, gl = J("light.json"), J("light_wrist.json"), J("gripper_lib/gripper_library.json")
    C = rep["checks"]
    C["light_iou"] = dict(value=lj.get("shadow_iou"), pass_=(lj.get("shadow_iou") or 0) >= LIGHT_IOU)
    C["wrist_light_iou"] = dict(value=lw.get("iou"), pass_=(lw.get("iou") or 0) >= WRIST_LIGHT_IOU)
    C["gripper_iou"] = dict(value=gl.get("iou_real_vs_sim_gripper"), n_frames=gl.get("n_frames"),
                            pass_=(gl.get("iou_real_vs_sim_gripper") or 0) >= GRIP_IOU and (gl.get("n_frames") or 0) >= GRIP_FRAMES)
    if os.path.exists(p("table_tex_wrist.npz")):
        z = np.load(p("table_tex_wrist.npz"))
        frac = float(z["observed"].mean()) if "observed" in z.files else None
        C["table_tex_observed"] = dict(value=frac, pass_=frac is not None and frac >= TEX_OBSERVED)
    else:
        C["table_tex_observed"] = dict(value=None, pass_=False)
    files = [o for s in rep["steps"] for o in s["outputs"]]
    C["files_present"] = dict(missing=[f for f in files if not os.path.exists(p(f))], pass_=all(os.path.exists(p(f)) for f in files))
    rep["files"] = files
    rep["disk_mb"] = round(sum(os.path.getsize(p(f)) for f in files if os.path.exists(p(f))) / 1e6, 1)
    rep["peak_ram_gib"] = round(max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                    resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1048576, 2)
    rep["seconds"] = round(sum(s["seconds"] for s in rep["steps"]), 1)
    rep["ok"] = bool(ok and all(c["pass_"] for c in C.values()))
    rep["note"] = "ok = every step ran and every provisional check passed; a human still accepts the check images"
    json.dump(rep, open(out_json, "w"), indent=1)
    for k, c in C.items():
        print(f"check {k}: {'PASS' if c['pass_'] else 'FAIL'} ({c.get('value')})", flush=True)
    print(out_json)
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
