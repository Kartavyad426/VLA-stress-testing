"""SceneGen v1 driver: one episode -> a built, checked scene + a Tier 0 / Tier 1 report (SceneGen_v1_gaps_1028 §9;
REAL2SIM_CHECKLIST_V1 §3b; user 10-28: "close v1 asap").

Steps (each skipped when its output exists; --force reruns all):
  1 camera      camera_source.py: DROID refined -> PointWorld (--no_pointworld excludes it) -> stop (no fallback in v1:
                pick covered episodes, or place the camera by hand in inspect_scene.py and pass --calib)
  2 nouns       <out>/nouns.json, the object nouns the detector is prompted with: from --nouns, or written by --vlm_nouns
                (NIM llama-3.2-11b, noun-only prompt) and then STOP so the user reviews it (checklist A6)
  3 perceive    perceive.py, every noun as a multi-instance prompt (one detector call per noun; counts come from the
                detector, not the VLM)
  4 build       build_scene.py --depth stereo with the camera; off-table detections listed, not built
  5 table       stereo_depth.py --wrist_plane (mask-free wrist-stereo plane) -> rebuild with --table_z
  6 scale check at the real grasp points, exterior stereo vs the camera's geometry; off by > 5 % -> rebuild with
                --depth grasp (+ --table_z)
  7 replay + scene_checks (+ episode_info.json)
  8 doable      implres31's expert_probe.py (B3), nominal sweep, if its worktree is present
  9 report      <out>/v1_report.json + v1_report.html: Tier 0 / Tier 1 rows with pass / fail / not run and the numbers
  --inspect     then opens inspect_scene.py on the scene

    .venvs/real2sim/bin/python experiments/real2sim/build_v1.py --episode data/real2sim/droid/ep_c11 \
        --out runs/real2sim/v1/c11 --nouns block,bowl [--inspect]
CPU (MUJOCO_GL=osmesa, triton first, nice 19 via py_triton_first.sh). Exploratory (res31, 10-28), UNVERIFIED.
"""
import argparse, glob, html, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PYW = os.path.join(REPO, "runs/real2sim/v1_dev/py_triton_first.sh")
CHECKS = os.path.join(os.path.dirname(REPO), "VLA-exp-real2sim-checks/experiments/real2sim_checks")
NEAR_BUILT_M = 0.05  # 10-29: an unbuilt grasp this close to a built object's surface is ambiguous (A8 'needs review')


def run(args, log, name):
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"\n===== {name}: {' '.join(args)}\n")
        f.flush()
        rc = subprocess.run([PYW] + args, stdout=f, stderr=subprocess.STDOUT, cwd=REPO).returncode
    print(f"  {name}: rc {rc} ({time.time() - t0:.0f} s)", flush=True)
    return rc


def ext_serial(episode):
    md = json.load(open(glob.glob(os.path.join(episode, "metadata_*.json"))[0]))
    return md.get("ext1_cam_serial", "22008760")


def vlm_nouns(episode, cam, out):
    sys.path.insert(0, HERE)
    import vlm_nouns_test as V
    from perceive import read_frame
    key = os.environ.get("NVIDIA_NIM_API_KEY") or sys.exit("set NVIDIA_NIM_API_KEY (robotics_agentic/.env) for --vlm_nouns")
    text, secs = V.ask(read_frame(os.path.join(episode, f"{cam}.mp4"), 0), "meta/llama-3.2-11b-vision-instruct", key)
    objs = V.parse(text) or []
    nouns = sorted({str(o.get("noun", "")).strip().lower() for o in objs if o.get("noun")})
    json.dump(dict(nouns=nouns, source="vlm llama-3.2-11b (noun-only prompt)", reviewed=False, raw=objs),
              open(os.path.join(out, "nouns.json"), "w"), indent=1)
    return nouns


def grasp_scale_ratios(out, episode):
    """Exterior stereo depth / camera-geometry depth at each real grasp point (geom_probe e2)."""
    p = os.path.join(out, "probe", "geom_probe.json")
    if not os.path.exists(p):
        return []
    rows = json.load(open(p))["e2"]["grasps"]
    return [g["stereo_at_px"] / g["cam_depth"] for g in rows if g.get("stereo_at_px") and g.get("inside_image")]


def redact_unbuilt_grasps(out, episode):
    """10-29 (user: "redact them, make json, have a check before scripted to not check for these"). A real grasp whose
    point, projected into the perception frame (frame 0, before anything moved), lands on no built or redacted object's
    mask (dilated 12 px) grabbed something that was never built: not detected, or never named (c24, c26, c3 scored 7-14 cm
    'misplacements' against the nearest built object). It goes into scene_fit.json 'unbuilt' at the grasp point (width
    unknown: 6 cm); composite.redacted_discs reads 'redacted' and 'unbuilt', so scene_checks skips that grasp and the
    composite masks it from the real textures. (A separate key, not 'redacted': implres31's image_checks C2 prompts the
    detector with each redacted label's noun.) A grasp within
    3 cm of an earlier release is a re-grasp of a moved object and is not added. One view only: an unbuilt object in
    front of or behind a built one in the image is missed (c06). Re-run after every rebuild (it rewrites scene_fit.json).
    An unbuilt grasp within NEAR_BUILT_M of a built object's surface may be a rim/edge grasp of that object (c9's cup,
    3.4 cm): it stays redacted but is marked near_built, and the report's A8 reads 'needs review'. UNVERIFIED."""
    import cv2, mujoco, h5py
    sys.path.insert(0, HERE)
    from episode_info import grasp_events
    from scene_checks import surface_dist
    from scene_roles import container_bodies
    J = lambda f: json.load(open(os.path.join(out, f)))
    c, fit = J("calib.json"), J("scene_fit.json")
    z = np.load(os.path.join(out, "perception.npz"))
    H, W = z["image"].shape[:2]
    R, t, K = (np.array(c[k], float) for k in ("R_cam_to_base", "t_cam_in_base", "K"))

    def pixel(p):
        x = R.T @ (np.asarray(p, float) - t)
        if x[2] <= 0:
            return None
        u = K @ (x / x[2])
        u, v = int(round(u[0])), int(round(u[1]))
        return (u, v) if 0 <= u < W and 0 <= v < H else None
    unb = fit["unbuilt"] = {}  # rebuilt from scratch each run
    labels = [o.get("mask", k) for k, o in fit["objects"].items() if o.get("body")] + list(fit.get("redacted", {}))
    ker = np.ones((25, 25), np.uint8)
    masks = [cv2.dilate(z["mask/" + l].astype(np.uint8), ker) > 0 for l in labels if "mask/" + l in z.files]
    m = mujoco.MjModel.from_binary_path(os.path.join(out, "scene.mjb"))
    d = mujoco.MjData(m)
    f = h5py.File(os.path.join(episode, "trajectory.h5"))
    Jq, G = f["observation/robot_state/joint_positions"][:], f["observation/robot_state/gripper_position"][:]
    arm = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    pinch = m.site("g_pinch").id
    cont = container_bodies(m, fit)
    objs = [m.jnt_bodyid[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[j] not in cont]

    def fk(k):
        d.qpos[:] = m.qpos0
        d.qpos[arm] = Jq[k]
        mujoco.mj_forward(m, d)
        return d.site_xpos[pinch].copy()
    releases, added, rows = [], [], []
    for kc, ko in grasp_events(G):
        p = fk(kc)
        row = dict(close_frame=int(kc), point=[round(float(x), 4) for x in p])
        q = pixel(p)
        if any(np.linalg.norm((p - r)[:2]) < 0.03 for r in releases):
            row["class"] = "re-grasp of a moved object (not judged)"
        elif q is None:
            row["class"] = "outside the camera image (not judged)"
        elif any(mk[q[1], q[0]] for mk in masks):
            row["class"] = "on a built or redacted object"
        elif any(np.linalg.norm((p - a)[:2]) < 0.03 for a in added):
            row["class"] = "same unbuilt object as an earlier grasp"
        else:
            unb[f"unbuilt@f{kc}"] = dict(reason="real grasp lands on no built object in the camera image (not detected, or never named)",
                                         kind="ellipsoid", center=[float(p[0]), float(p[1]), float(max(p[2], fit["table"]["z"]))],
                                         width_m=0.06, close_frame=int(kc), source="build_v1 unbuilt grasp")
            added.append(p)
            row["class"] = f"unbuilt: redacted as unbuilt@f{kc}"
            if objs:
                dist, b = min((surface_dist(m, d, b, p), b) for b in objs)
                if dist <= NEAR_BUILT_M:
                    unb[f"unbuilt@f{kc}"]["near_built"] = row["near_built"] = dict(body=m.body(b).name, surface_m=round(dist, 4))
                    row["class"] += f" (needs review: {100 * dist:.1f} cm from built {m.body(b).name}, may be a rim/edge grasp of it)"
        releases.append(fk(ko))
        rows.append(row)
    json.dump(fit, open(os.path.join(out, "scene_fit.json"), "w"), indent=1)
    return rows


def redaction_gate(out, grasp_rows):
    """Check before the scripted expert (B3; user 10-29): no redacted object may exist in the model, so B3 never probes
    it and no check scores it. Writes <out>/redacted.json: every redacted object with its reason and source, the nouns
    with no detection (names only, no position), how each real grasp was judged, and the gate result. Covers both
    scene_fit.json 'redacted' (no matching model) and 'unbuilt' (grasped, never built)."""
    import mujoco
    fit = json.load(open(os.path.join(out, "scene_fit.json")))
    m = mujoco.MjModel.from_binary_path(os.path.join(out, "scene.mjb"))
    bodies = {m.body(i).name for i in range(m.nbody)}
    red = {**fit.get("redacted", {}), **fit.get("unbuilt", {})}
    present = [l for l in red if l.replace("#", "_").replace(" ", "_") in bodies]
    inv_p = os.path.join(out, "inventory.json")
    inv = json.load(open(inv_p)) if os.path.exists(inv_p) else {}
    rec = dict(note="objects left out of the scene and out of every check; add later (inspect_scene / TaskGen agent / ObjGen). "
                    "build_scene --redact_unmatched: detected, no matching model. build_v1 unbuilt grasp: the real robot grasped "
                    "something no built object covers (position = the real grasp point, width unknown). UNVERIFIED.",
               objects=red, missing_names=inv.get("missing", []), grasps=grasp_rows,
               gate=dict(passed=not present, redacted_but_in_model=present))
    json.dump(rec, open(os.path.join(out, "redacted.json"), "w"), indent=1)
    return not present


def report(out, episode, info, label=None):
    J = lambda f: json.load(open(os.path.join(out, f))) if os.path.exists(os.path.join(out, f)) else None
    calib, fit, sc, wp = J("calib.json"), J("scene_fit.json"), J("scene_checks.json"), J("wrist_table_plane.json")
    rows = []

    def row(tier, cid, name, state, detail):
        rows.append(dict(tier=tier, id=cid, check=name, state=state, detail=detail))
    lic = (calib or {}).get("licence", "")
    row(0, "A7", "camera from a published calibration", "pass" if calib and "source" in calib else "fail",
        f"{(calib or {}).get('source', 'none')}; licence: {lic}")
    if wp and wp.get("z") is not None and fit:
        dz = 100 * abs(fit["table"]["z"] - wp["z"])
        row(0, "A1", "table height = wrist plane", "pass" if dz <= 1.0 or "table_z_override" in fit else "fail",
            f"built {fit['table']['z']:+.4f} m, wrist plane {wp['z']:+.4f} m (tilt {wp['tilt_deg']:.1f}°, {wp['n_inliers']} pts)"
            + ("; built table set from the wrist plane" if "table_z_override" in fit else ""))
    else:
        row(0, "A1", "table height = wrist plane", "not run", (wp or {}).get("why", "no wrist plane"))
    if sc:
        ch = sc["checks"]
        ev = [e for e in ch.get("3_4_5_grasp_events", []) if "skipped" not in e]
        row(0, "B1", "settle", "pass" if ch["1_settle"]["pass_"] else "fail",
            f"max move {100 * max(ch['1_settle']['moved_m'].values(), default=0.0):.2f} cm")
        row(0, "A3", "real grasps on/in the objects", "pass" if ch["3_grasp"]["pass_"] else ("n/a" if ch["3_grasp"]["pass_"] is None else "fail"),
            "; ".join(f"f{e['close_frame']} {e['object']} {100 * e.get('surface_m', np.nan):.1f} cm" for e in ev))
        row(0, "A3", "real drops inside the containers", "pass" if ch["4_release"]["pass_"] else ("n/a" if ch["4_release"]["pass_"] is None else "fail"),
            "; ".join(f"f{e['open_frame']} {100 * e['release_horiz_m']:.1f} / rim {100 * (e.get('rim_radius_m') or 0):.1f} cm" for e in ev if "release_horiz_m" in e))
        row(1, "Z3", "grasp→drop distance, real vs sim (report only)", "report", "; ".join(f"{e.get('scale_ratio') or 0:.2f}" for e in ev))
    else:
        row(0, "A3", "real grasps / drops", "not run", "")
    off = (fit or {}).get("off_table", {})
    red = (fit or {}).get("redacted", {})
    nouns = J("nouns.json") or {}
    inv = J("inventory.json") or {}
    miss = inv.get("missing") or []
    row(0, "A6", "object inventory reviewed; off-table / redacted / missing listed",
        "needs review" if (miss or not nouns.get("reviewed")) else "pass",
        f"nouns {nouns.get('nouns')} ({'reviewed' if nouns.get('reviewed') else 'NOT reviewed'}); built {sorted((fit or {}).get('objects', {}))}; "
        f"off-table {sorted(off)}; redacted {sorted(red)}; unbuilt (grasped, never built) {sorted((fit or {}).get('unbuilt', {}))}; "
        f"MISSING (noun given, nothing detected; add later) {miss}")
    rj = J("redacted.json")
    if rj:
        n_skip = sum(1 for e in ((sc or {}).get("checks", {}).get("3_4_5_grasp_events", [])) if "redacted" in str(e.get("skipped", "")))
        near = {k: v["near_built"] for k, v in rj["objects"].items() if v.get("near_built")}
        row(0, "A8", "redacted objects left out of the model and the checks (gate before B3)",
            "fail" if not rj["gate"]["passed"] else ("needs review" if near else "pass"),
            "; ".join(f"{k}: {v.get('source', 'asset pack')}" for k, v in sorted(rj["objects"].items())) +
            f" | real grasps skipped as redacted: {n_skip}" + (f" | IN THE MODEL: {rj['gate']['redacted_but_in_model']}" if not rj["gate"]["passed"] else "") +
            "".join(f" | {k} is {100 * v['surface_m']:.1f} cm from built {v['body']}: may be a rim/edge grasp of it, so that grasp's "
                    "A3 placement/drop is skipped, not passed (look in inspect_scene)" for k, v in sorted(near.items())))
    ep = J(os.path.join("checks", "expert_probe.json"))
    if rj and not rj["gate"]["passed"]:
        row(0, "B3", "doable: scripted expert per object (implres31)", "not run", "redaction gate failed (A8): a redacted object is in the model")
    elif ep:
        t = ep["table"]
        n_pass = sum(1 for v in t.values() if v.get("nominal") == "PASS")
        fails = [k for k, v in t.items() if v.get("nominal") != "PASS"]
        if not t:  # 10-29: no graspable object built (c16, c20, c22): nothing was probed, so not a pass
            row(0, "B3", "doable: scripted expert per object (implres31)", "n/a", "0 probes: no built graspable object to probe")
        else:
            row(0, "B3", "doable: scripted expert per object (implres31)", "pass" if not fails else "fail",
                f"{n_pass}/{len(t)} probes pass" + (f"; failing: {fails[:6]}" if fails else ""))
    else:
        row(0, "B3", "doable: scripted expert per object (implres31)", "not run", "expert_probe.py not available or not run")
    row(0, "E2", "do-nothing policy doesn't succeed", "not run", "make_tasks + policy_batch --policy hold (task generation not in build_v1 yet)")
    row(0, "E4", "success rule for the task type", "not run", "bowl tasks only today")
    ratios = info.get("scale_ratios") or []
    row(1, "A1/A7", "exterior stereo vs geometry at grasp points", "pass" if ratios and all(abs(r - 1) <= 0.05 for r in ratios) else ("report" if ratios else "not run"),
        ", ".join(f"{r:.3f}" for r in ratios) + (f"; built with --depth {info.get('depth')}" if info.get("depth") else ""))
    for cid, name in (("A4", "object sizes"), ("A5", "shapes"), ("A2", "table extent"), ("D1", "wrist-view offsets"),
                      ("C1-C5", "appearance"), ("D6", "MoGe check")):
        row(1, cid, name, "not run", "implres31 image_checks.py / moge_check.py (rerun on this scene)")
    # 10-29 (user: "dont remove the prev checklist result tho, append into a new section while saying what changes"):
    # the previous report's rows go into 'history' (oldest first) unless they are identical to the new rows
    prev = J("v1_report.json")
    history, prev_label = (prev or {}).get("history", []), (prev or {}).get("label")
    if prev and prev["rows"] != rows:
        history = history + [dict(created=prev["created"], label=prev.get("label") or "earlier run", rows=prev["rows"])]
        prev_label = None
    rep = dict(episode=episode, out=out, status="exploratory, UNVERIFIED", created=time.strftime("%Y-%m-%d %H:%M"),
               label=label or prev_label or "build_v1 run",
               tier0_open=[r for r in rows if r["tier"] == 0 and r["state"] not in ("pass", "n/a")], rows=rows, info=info, history=history)
    key = lambda r: (r["id"], r["check"])
    last = {key(r): r for r in history[-1]["rows"]} if history else {}
    rep["changes"] = [dict(id=r["id"], check=r["check"], before=last.get(key(r), {}).get("state", "(no row)"), after=r["state"])
                      for r in rows if history and last.get(key(r), {}).get("state") != r["state"]]
    json.dump(rep, open(os.path.join(out, "v1_report.json"), "w"), indent=1, default=float)
    col = {"pass": "#2c6e3a", "fail": "#a3321f", "needs review": "#8a6d12", "not run": "#777", "report": "#555", "n/a": "#555"}
    st = lambda s: f"<span style='color:{col.get(s, '#888')};font-weight:600'>{html.escape(s)}</span>"
    table = lambda rs: "<table><tr><th>Tier</th><th>Check</th><th>What</th><th>State</th><th>Detail</th></tr>" + "".join(
        f"<tr><td>{r['tier']}</td><td>{r['id']}</td><td>{html.escape(r['check'])}</td><td>{st(r['state'])}</td><td>{html.escape(str(r['detail']))}</td></tr>" for r in rs) + "</table>"
    trs = table(rows)
    if history:
        trs += (f"<h2>What changed since the previous result</h2><p>Previous: {html.escape(history[-1]['label'])} ({html.escape(str(history[-1]['created']))}). "
                f"This run: {html.escape(rep['label'])}.</p>" +
                ("<table><tr><th>Check</th><th>What</th><th>Before</th><th>Now</th></tr>" + "".join(
                    f"<tr><td>{c['id']}</td><td>{html.escape(c['check'])}</td><td>{st(c['before'])}</td><td>{st(c['after'])}</td></tr>" for c in rep["changes"]) + "</table>"
                 if rep["changes"] else "<p>No state changed (details may differ).</p>") +
                "<h2>Earlier results (kept, newest first)</h2>" + "".join(
                    f"<details><summary>{html.escape(h['label'])} ({html.escape(str(h['created']))})</summary>{table(h['rows'])}</details>" for h in reversed(history)))
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SceneGen v1 report</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;margin:0;padding:20px 16px;background:#fbfbf9;color:#1d1d1b}}main{{max-width:1100px;margin:0 auto}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:5px 7px;text-align:left;vertical-align:top}}th{{background:#f2f1ec}}
@media (prefers-color-scheme:dark){{body{{background:#161615;color:#e8e7e2}}th{{background:#1f1f1d}}td,th{{border-color:#34332f}}}}</style></head>
<body><main><h1>SceneGen v1 report: {html.escape(os.path.basename(os.path.normpath(out)))}</h1>
<p>Episode {html.escape(episode)} · {rep['created']} · exploratory, UNVERIFIED · Tier 0 open: {len(rep['tier0_open'])}</p>
{trs}</main></body></html>"""
    open(os.path.join(out, "v1_report.html"), "w").write(page)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", default=None, help="exterior serial; default: the episode's ext1 camera")
    ap.add_argument("--nouns", default=None, help="comma-separated object nouns (the user-reviewed list)")
    ap.add_argument("--vlm_nouns", action="store_true", help="ask the VLM for nouns, write <out>/nouns.json and stop for review")
    ap.add_argument("--auto_nouns", action="store_true", help="unattended runs: VLM nouns WITHOUT review, then continue "
                    "(nouns.json reviewed=false, so the report's A6 reads 'needs review')")
    ap.add_argument("--calib", default=None, help="a calib.json to use instead of camera_source (e.g. from inspect_scene)")
    ap.add_argument("--no_pointworld", action="store_true", help="exclude PointWorld (non-commercial licence)")
    ap.add_argument("--no_probe", action="store_true", help="skip implres31's expert_probe (B3)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--report_only", action="store_true", help="10-29: on a built scene, redo the unbuilt-grasp redaction, "
                    "scene_checks, the gate and the report (keeps the model and the B3 results; no rebuild, no expert probe)")
    ap.add_argument("--label", default=None, help="what this run is / what changed; shown in the report next to the earlier results it keeps")
    a = ap.parse_args()
    out, ep = os.path.abspath(a.out), os.path.abspath(a.episode)
    os.makedirs(out, exist_ok=True)
    log = os.path.join(out, "build_v1.log")
    cam = a.cam or ext_serial(ep)
    R = os.path.join(HERE, "{}.py")
    if a.report_only:
        old = json.load(open(os.path.join(out, "v1_report.json")))
        grasp_rows = redact_unbuilt_grasps(out, ep)
        run([R.format("scene_checks"), "--scene", out, "--episode", ep, "--no_overlay"], log, "scene_checks (report_only)")
        gate_ok = redaction_gate(out, grasp_rows)
        rep = report(out, ep, old["info"], a.label)
        print(f"report (report_only, gate {'pass' if gate_ok else 'FAIL'}): Tier 0 open: {[r['id'] + ' ' + r['state'] for r in rep['tier0_open']]}", flush=True)
        return
    have = lambda f: os.path.exists(os.path.join(out, f)) and not a.force
    info = dict(cam=cam)
    print(f"build_v1 {ep} -> {out} (camera {cam})", flush=True)
    # 1 camera
    if a.calib:
        json.dump(json.load(open(a.calib)), open(os.path.join(out, "calib.json"), "w"), indent=1)
    elif not have("calib.json"):
        if run([R.format("camera_source"), "--episode", ep, "--cam", cam, "--out", out] + (["--no_pointworld"] if a.no_pointworld else []), log, "camera") != 0:
            sys.exit("no published calibration for this episode: place the camera in inspect_scene.py (camera mode) and pass --calib")
    # 2 nouns
    npath = os.path.join(out, "nouns.json")
    if a.nouns:
        json.dump(dict(nouns=[x.strip() for x in a.nouns.split(",") if x.strip()], source="--nouns", reviewed=True), open(npath, "w"), indent=1)
    elif a.vlm_nouns:
        print("VLM nouns (review, then rerun with --nouns):", vlm_nouns(ep, cam, out))
        return
    elif a.auto_nouns and not os.path.exists(npath):
        print("  VLM nouns (unreviewed):", vlm_nouns(ep, cam, out), flush=True)
    if not os.path.exists(npath):
        sys.exit("no object nouns: pass --nouns, or --vlm_nouns to get a list to review")
    nouns = json.load(open(npath))["nouns"]
    # 3 perceive: every noun multi-instance
    if not have("perception.json"):
        if run([R.format("perceive"), "--episode", ep, "--cam", cam, "--out", out, "--prompts", "", "--multi", ",".join(nouns)], log, "perceive") != 0:
            sys.exit(f"perceive failed: see {log}")
    # inventory (10-29; user: "We can put all this in redacted as a workaround … so manual mode or agent mode can add it
    # later with ObjGen"): a reviewed noun the detector found nothing for (c3: "radish") is listed as missing, like a
    # redacted object; inspect_scene / the TaskGen agent / ObjGen add it later. The real object stays in the background
    # images until then (redaction gap a)
    det = json.load(open(os.path.join(out, "perception.json")))["detections"]
    found = {k.split("#")[0] for k in det}
    missing = [n for n in nouns if n not in found]
    json.dump(dict(nouns=nouns, detected=sorted(found - {"table", "robot gripper"}), missing=missing,
                   note="missing = noun given, nothing detected; not built; add later (inspect_scene / TaskGen agent / ObjGen)"),
              open(os.path.join(out, "inventory.json"), "w"), indent=1)
    if missing:
        print(f"  WARNING nouns with no detection (not built, listed as missing): {missing}", flush=True)
    # 4 build (stereo scale)
    calib = os.path.join(out, "calib.json")
    build = lambda extra, tag: run([R.format("build_scene"), "--episode", ep, "--cam", cam, "--out", out, "--calib", calib,
                                    "--redact_unmatched"] + extra, log, tag)  # 10-29: no stand-in ellipsoids
    if not have("scene.mjb"):
        if build(["--depth", "stereo"], "build (stereo)") != 0:
            sys.exit(f"build failed: see {log}")
    # 5 table from the wrist plane
    if not have("wrist_table_plane.json"):
        run([R.format("stereo_depth"), "--scene", out, "--episode", ep, "--wrist_plane"], log, "wrist plane")
    wp = json.load(open(os.path.join(out, "wrist_table_plane.json")))
    tz = ["--table_z", f"{wp['z']:.4f}"] if wp.get("z") is not None else []
    # grasp anchors (FK at each real close) for the scale check and a possible grasp-scale build
    if not have("grasp_anchors.json"):
        import mujoco, h5py
        sys.path.insert(0, HERE)
        from episode_info import grasp_events
        m = mujoco.MjModel.from_binary_path(os.path.join(out, "scene.mjb"))
        d = mujoco.MjData(m)
        f = h5py.File(os.path.join(ep, "trajectory.h5"))
        J, G = f["observation/robot_state/joint_positions"][:], f["observation/robot_state/gripper_position"][:]
        arm = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
        an = {}
        for i, (kc, ko) in enumerate(grasp_events(G)):
            d.qpos[:] = m.qpos0
            d.qpos[arm] = J[kc]
            mujoco.mj_forward(m, d)
            an[f"g{i}"] = dict(point=d.site_xpos[m.site("g_pinch").id].tolist(), frame=int(kc))
        json.dump(dict(anchors=an), open(os.path.join(out, "grasp_anchors.json"), "w"), indent=1)
    # 6 scale check at the grasp points; rebuild with the wrist table (and grasp scale if stereo disagrees)
    if not have(os.path.join("probe", "geom_probe.json")):
        run([R.format("geom_probe"), "--scene", out, "--episode", ep, "--out", os.path.join(out, "probe"), "--probes", "e6,e2"], log, "scale check")
    ratios = grasp_scale_ratios(out, ep)
    # 10-29: with a published camera, exterior stereo agreed with the robot geometry on every scene (0.95-1.02), while a
    # grasp point can sit on a rim or handle (c9: the cup gripped at its rim read 1.17), so the published-camera build
    # keeps stereo and only reports the ratio; grasp scale only for cameras we fitted ourselves (--calib)
    published = "source" in json.load(open(calib)) and not a.calib
    depth = "grasp" if (not published and ratios and np.median(np.abs(np.array(ratios) - 1)) > 0.05) else "stereo"
    info.update(scale_ratios=ratios, depth=depth, wrist_plane=wp)
    extra = ["--depth", depth] + tz + (["--grasp_anchors", os.path.join(out, "grasp_anchors.json")] if depth == "grasp" else [])
    if build(extra, f"rebuild ({depth} scale, wrist table)") != 0:
        sys.exit(f"rebuild failed: see {log}")
    # 6b grasps of objects that were never built -> redacted at the grasp point (after the rebuild, which rewrites the fit)
    grasp_rows = redact_unbuilt_grasps(out, ep)
    n_unbuilt = sum(1 for r in grasp_rows if r["class"].startswith("unbuilt"))
    print(f"  unbuilt grasps redacted: {n_unbuilt} of {len(grasp_rows)} grasps", flush=True)
    # 7 replay + checks
    run([R.format("replay"), "--episode", ep, "--cam", cam, "--out", out], log, "replay")
    run([R.format("scene_checks"), "--scene", out, "--episode", ep, "--no_overlay"], log, "scene_checks")
    run([R.format("episode_info"), "--episode", ep, "--scene", out], log, "episode_info")
    # 7b gate before the scripted expert: no redacted object in the model (writes redacted.json)
    gate_ok = redaction_gate(out, grasp_rows)
    print(f"  redaction gate: {'pass' if gate_ok else 'FAIL (a redacted object is in the model): B3 skipped'}", flush=True)
    # 8 doable (implres31's B3), nominal sweep only
    probe = os.path.join(CHECKS, "expert_probe.py")
    if gate_ok and not a.no_probe and os.path.exists(probe):
        run([probe, "--scene", out, "--episode", ep, "--out", os.path.join(out, "checks"), "--sweeps", "nominal"], log, "B3 expert probe")
    # 9 report
    rep = report(out, ep, info, a.label)
    print(f"report: {os.path.join(out, 'v1_report.html')}  Tier 0 open: {[r['id'] + ' ' + r['state'] for r in rep['tier0_open']]}", flush=True)
    if a.inspect:
        subprocess.Popen([os.path.join(REPO, ".venvs/real2sim/bin/python"), os.path.join(HERE, "inspect_scene.py"), "--scene", out,
                          "--episode", ep, "--view", "ext"], env={k: v for k, v in os.environ.items() if k != "MUJOCO_GL"})


if __name__ == "__main__":
    main()
