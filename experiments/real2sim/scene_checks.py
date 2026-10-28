"""Build-time self-check of a rebuilt scene against the REAL episode (pre-deploy checklist A: #1, #3-#6).

Lesson of 10-16: a scene validated only from the camera it was estimated from is blind to errors along that
camera's rays (a 6 cm table offset looked perfect in every composite). So every check here uses something
independent of the exterior calibration: the recorded robot joints (exact, via FK) and the stereo depth.

  #1 settle      every free body moves < SETTLE_M in 0.5 s (robot held at its frame-0 joints)
  #3 real grasp  fingertip point (g_pinch, FK of the recorded joints) at the operator's gripper close vs the
                 nearest object: on or in it, within GRASP_M of its collision boxes (10-28; was: |horizontal| and
                 |vertical| offset from its centre <= GRASP_M, which failed correct scenes on rim/handle grasps)
  #4 release     fingertip point at the operator's release vs the nearest container (bowl): inside its rim
  #5 scale       real grasp->release distance vs sim object->container distance: ratio within 1 +- SCALE_TOL
  #6 stereo      table-top z from the exterior and wrist stereo (stereo_depth.py) vs the built table: <= TABLE_M

Episodes with several grasps (c3): #3/#4 are checked for every close/open pair. Writes <scene>/scene_checks.json and
<scene>/wrist_overlay.jpg (wrist_overlay.py; look at it: object positions and sizes from a second viewpoint).

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/scene_checks.py \
        --scene runs/real2sim/ep_orange_block_stereo --episode data/real2sim/droid/ep_orange_block
"""
import argparse, json, os, subprocess, sys
import numpy as np
import mujoco
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from composite import redacted_discs  # noqa: E402
from scene_roles import container_bodies  # noqa: E402
from episode_info import frames_arg, grasp_events  # noqa: E402,F401  (grasp_events moved there 10-26)

# thresholds: res31's, user-accepted 10-19 (reversible)
SETTLE_M, GRASP_M, SCALE_TOL, TABLE_M = 0.02, 0.015, 0.10, 0.01


def surface_dist(m, d, body, p):
    """Distance (m) from point p to the nearest collision box of `body` (all geoms whose root is body; each as its local
    bounding box), 0 inside. 10-28: a grasp point is ON or IN the object (rim, handle), not at its centre (c9's cup)."""
    best = np.inf
    for g in range(m.ngeom):
        if m.body_rootid[m.geom_bodyid[g]] != body or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        q = d.geom_xmat[g].reshape(3, 3).T @ (p - d.geom_xpos[g]) - c
        best = min(best, float(np.linalg.norm(np.maximum(np.abs(q) - h, 0.0))))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--stereo_frames", default=None, help="wrist frames for #6 (before the objects are touched); derived from the episode (episode_info.py) when omitted")
    ap.add_argument("--no_overlay", action="store_true")
    a = ap.parse_args()
    a.stereo_frames = ",".join(map(str, frames_arg(a.stereo_frames, a.episode, "stereo")))
    m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
    fit = json.load(open(os.path.join(a.scene, "scene_fit.json")))
    h = h5py.File(os.path.join(a.episode, "trajectory.h5"))
    J, G = h["observation/robot_state/joint_positions"][:], h["observation/robot_state/gripper_position"][:]
    arm = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    au = [m.actuator(f"fr3_joint{k}").id for k in range(1, 8)]
    pinch = m.site("g_pinch").id
    free = [m.jnt_bodyid[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    containers = container_bodies(m, fit)  # fitted containers (10-26; was: free bodies named *bowl*)
    out = dict(scene=a.scene, episode=a.episode, thresholds=dict(settle_m=SETTLE_M, grasp_m=GRASP_M, scale_tol=SCALE_TOL, table_m=TABLE_M), checks={})

    # #1 settle
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    d.qpos[arm] = J[0]
    d.ctrl[au] = J[0]
    mujoco.mj_forward(m, d)
    x0 = {b: d.xpos[b].copy() for b in free}
    for _ in range(int(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    mv = {m.body(b).name: float(np.linalg.norm(d.xpos[b] - x0[b])) for b in free}
    out["checks"]["1_settle"] = dict(pass_=max(mv.values()) < SETTLE_M, moved_m=mv)

    # #3/#4/#5 per grasp event, objects at the scene's initial poses
    def pinch_at(k):
        dk = mujoco.MjData(m)
        dk.qpos[:] = m.qpos0
        dk.qpos[arm] = J[k]
        mujoco.mj_forward(m, dk)
        return dk.site_xpos[pinch].copy(), dk
    events = []
    for kc, ko in grasp_events(G):
        p, dk = pinch_at(kc)
        objs = [b for b in free if b not in containers]
        tgt = min(objs, key=lambda b: surface_dist(m, dk, b, p))  # nearest by surface, not centre (10-28)
        # a grasp of an object left out of the scene (asset-pack redaction, 10-20) has nothing to compare with
        red = [(np.hypot(p[0] - cx, p[1] - cy), lab) for cx, cy, _, lab in redacted_discs(fit)]
        if red and min(red)[0] <= np.linalg.norm((dk.xpos[tgt] - p)[:2]):
            events.append(dict(close_frame=kc, open_frame=ko, object=min(red)[1], skipped="object redacted (not in the scene)"))
            continue
        dp = p - dk.xpos[tgt]
        ev = dict(close_frame=kc, open_frame=ko, object=m.body(tgt).name, surface_m=surface_dist(m, dk, tgt, p),
                  horiz_m=float(np.linalg.norm(dp[:2])), vert_m=float(dp[2]),
                  pinch_above_table_m=float(p[2] - fit["table"]["z"]))
        ev["grasp_pass"] = ev["surface_m"] <= GRASP_M  # 10-28: on/in the object within 1.5 cm (was: centre offsets)
        if containers:
            p2, _ = pinch_at(ko)
            c = min(containers, key=lambda b: np.linalg.norm(dk.xpos[b][:2] - p2[:2]))
            rim = next((o.get("rim_radius") for o in fit["objects"].values() if o.get("body") == m.body(c).name), None)
            ev.update(container=m.body(c).name, release_horiz_m=float(np.linalg.norm((p2 - dk.xpos[c])[:2])), rim_radius_m=rim)
            ev["release_pass"] = bool(rim and ev["release_horiz_m"] < rim)
            real_d = float(np.linalg.norm((p2 - p)[:2]))
            sim_d = float(np.linalg.norm((dk.xpos[c] - dk.xpos[tgt])[:2]))
            ev.update(scale_ratio=real_d / sim_d if sim_d > 0 else None)
            ev["scale_pass"] = bool(ev["scale_ratio"] and abs(ev["scale_ratio"] - 1) <= SCALE_TOL)
        events.append(ev)
    out["checks"]["3_4_5_grasp_events"] = events
    used = [e for e in events if "skipped" not in e]
    na = None if not used else False  # no usable grasp event (none recorded, or all on redacted objects): n/a, not FAIL
    out["checks"]["3_grasp"] = dict(pass_=all(e["grasp_pass"] for e in used) if used else na, n_events=len(used),
                                    n_skipped_redacted=len(events) - len(used))
    out["checks"]["4_release"] = dict(pass_=all(e.get("release_pass", False) for e in used) if used else na)
    out["checks"]["5_scale"] = dict(pass_=all(e.get("scale_pass", False) for e in used) if used else na)

    # #6 stereo table height (stereo_depth.py writes <scene>/stereo_table_check.json)
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"))
    subprocess.run([sys.executable, os.path.join(HERE, "stereo_depth.py"), "--scene", a.scene, "--episode", a.episode,
                    "--frames", a.stereo_frames], check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    st = json.load(open(os.path.join(a.scene, "stereo_table_check.json")))
    diffs = {f"{r['cam']}_{r['frame']}": r["minus_sim_cm"] / 100 for r in st["results"]}
    out["checks"]["6_stereo_table"] = dict(pass_=all(abs(v) <= TABLE_M for v in diffs.values()), stereo_minus_built_m=diffs)

    if not a.no_overlay:
        subprocess.run([sys.executable, os.path.join(HERE, "wrist_overlay.py"), "--scene", a.scene, "--episode", a.episode,
                        "--frames", a.stereo_frames], check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out["wrist_overlay"] = os.path.join(a.scene, "wrist_overlay.jpg")
    out["all_pass"] = all(v["pass_"] for k, v in out["checks"].items() if isinstance(v, dict) and v.get("pass_") is not None)
    json.dump(out, open(os.path.join(a.scene, "scene_checks.json"), "w"), indent=1, default=float)
    for k, v in out["checks"].items():
        if isinstance(v, dict) and "pass_" in v:
            print(f"{k}: {'n/a' if v['pass_'] is None else 'PASS' if v['pass_'] else 'FAIL'}")
    for e in events:
        if "skipped" in e:
            print(f"  grasp f{e['close_frame']}->{e['open_frame']} {e['object']}: skipped ({e['skipped']})")
            continue
        print(f"  grasp f{e['close_frame']}->{e['open_frame']} {e['object']}: surface {e['surface_m']*100:.1f} cm (centre: horiz {e['horiz_m']*100:.1f} vert {e['vert_m']*100:+.1f} cm)"
              + (f"; release {e['release_horiz_m']*100:.1f} cm from {e['container']} (rim {e['rim_radius_m'] and e['rim_radius_m']*100:.1f}); scale {e['scale_ratio']:.3f}" if "container" in e else ""))
    print("ALL PASS" if out["all_pass"] else "NOT ALL PASS")


if __name__ == "__main__":
    main()
