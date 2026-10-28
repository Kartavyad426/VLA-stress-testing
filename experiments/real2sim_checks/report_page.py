"""Results page for implres31's REAL2SIM_CHECKLIST_V1 checks: reads whatever result files exist under
runs/real2sim_checks/ (expert_probe.json, image_checks.json, moge_check.json, encoder_distance.json per scene; tests/ for
the known-bad cases) and writes runs/real2sim_checks/index.html (local page; images and videos linked relatively).
Re-run after every check run. Numbers are exploratory, UNVERIFIED; nothing here gates (checklist rule 1).

    python3 experiments/real2sim_checks/report_page.py
"""
import datetime, glob, html, json, os

ROOT = "/home/imerit/Documents/Code/VLA/runs/real2sim_checks"
CHECKLIST = "../../docs/REAL2SIM_CHECKLIST_V1.html"
SCENE_ORDER = ["ep_orange_block_stereo", "c11", "c1b", "c3h", "c9"]
SCENE_NOTE = {
    "ep_orange_block_stereo": "reference scene (orange booth, rebuilt 10-20); the only one with composite caches",
    "c11": "10-27 calibration test, built from the anchored calibration that K13 found wrong (objects 13–21 cm from the real grasps)",
    "c1b": "10-27 calibration test, broken build (K13: rotated 39°, cups placed far away and fallen to the floor)",
    "c3h": "10-27 calibration test with hand prompts; anchored calibration (K13: overlap 0.27 → 0.15)",
    "c9": "10-27 calibration test; stopped at the stereo table step (K14), built from the outline-only fit",
}

CSS = """:root{--bg:#fbfaf7;--fg:#1f1c17;--muted:#6b645a;--line:#e2ddd3;--card:#f3efe7;--accent:#7a5c2e;--bad:#a3321f;--ok:#2f6b3a;--warn:#8a6a12}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#17150f;--fg:#ebe6dc;--muted:#a69e90;--line:#363127;--card:#211e17;--accent:#c9a46a;--bad:#ee8a76;--ok:#86c994;--warn:#e0c060}}
:root[data-theme="dark"]{--bg:#17150f;--fg:#ebe6dc;--muted:#a69e90;--line:#363127;--card:#211e17;--accent:#c9a46a;--bad:#ee8a76;--ok:#86c994;--warn:#e0c060}
*{box-sizing:border-box}body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:28px 16px}
main{max-width:1180px;margin:0 auto}h1{font-size:1.6rem;margin:.1em 0 .2em}h2{font-size:1.2rem;margin:2em 0 .5em;padding-bottom:.25em;border-bottom:2px solid var(--accent)}
h3{font-size:1.02rem;margin:1.4em 0 .4em;color:var(--accent)}a{color:var(--accent)}.meta{color:var(--muted);font-size:.86rem}
.card{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:6px;padding:12px 16px;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:.86rem;margin:.4em 0}th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{background:var(--card);font-weight:600}td.n{font-variant-numeric:tabular-nums;white-space:nowrap}.wrap{overflow-x:auto}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}.warn{color:var(--warn);font-weight:600}.mut{color:var(--muted)}
img{max-width:100%;border-radius:4px;display:block;margin:6px 0}figure{margin:8px 0}figcaption{font-size:.8rem;color:var(--muted)}
nav a{margin-right:12px;font-size:.88rem}details{margin:.4em 0}summary{cursor:pointer;color:var(--accent)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}@media (max-width:760px){.grid2{grid-template-columns:1fr}}
video{max-width:320px;width:100%;border-radius:4px}"""


def e(x):
    return html.escape(str(x))


def f(x, nd=2, unit=""):
    if x is None:
        return '<span class="mut">—</span>'
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}{unit}"
    return e(x)


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def cls_pass(s):
    s = str(s)
    return "ok" if s == "PASS" else "mut" if "not graspable" in s else "bad"


def table(head, rows, raw=False):
    h = ['<div class="wrap"><table><tr>' + "".join(f"<th>{e(c)}</th>" for c in head) + "</tr>"]
    for r in rows:
        h.append("<tr>" + "".join(f"<td>{c if raw else e(c)}</td>" for c in r) + "</tr>")
    h.append("</table></div>")
    return "\n".join(h)


def img(rel, cap):
    if not os.path.exists(os.path.join(ROOT, rel)):
        return ""
    return f'<figure><a href="{e(rel)}"><img src="{e(rel)}" alt="{e(cap)}" loading="lazy"></a><figcaption>{e(cap)}</figcaption></figure>'


# ------------------------------------------------------------------ per-scene summaries
def b3_summary(ep):
    if not ep:
        return None
    out = {}
    for sw, v in ep["sweeps"].items():
        rows = [r for r in v["rows"] if r.get("passed") is not None]
        out[sw] = (sum(bool(r["passed"]) for r in rows), len(rows), sum(r.get("stage") == "UNSTABLE" for r in rows))
    return out


def scene_row(name, d):
    ep, ic = d.get("expert_probe"), d.get("image_checks")
    b3 = b3_summary(ep)
    nom = b3.get("nominal") if b3 else None
    sweeps = [k for k in (b3 or {}) if k != "nominal"]
    sw_txt = ", ".join(f"{k} {p}/{n}" for k, (p, n, _) in ((k, b3[k]) for k in sweeps)) if sweeps else "—"
    c = (ic or {}).get("checks", {})
    d3 = c.get("d3", {})
    d3_err = [o for o, v in d3.items() if v.get("error")]
    d1 = c.get("d1", {})
    wr = [v["wrist"]["offset_cm_median"] for v in d1.values() if v.get("wrist", {}).get("offset_cm_median") is not None]
    # misplaced = the real object was found in some frame but never near the sim one (no measured frame in that view)
    far = [f"{o} ({cm})" for o, v in d1.items() for cm in ("wrist", "ext")
           if v.get(cm, {}).get("far_frames") and not v.get(cm, {}).get("n_frames")]
    unw = [o for o, v in d1.items() if v.get("wrist", {}).get("label") == "unwitnessed"]
    a2 = c.get("a2", {}).get("views", {}).get("wrist", {})
    d5 = c.get("d5", {})
    d5_err = [f"{o} ({v['rests_on']})" for o, v in d5.items() if v.get("error")]
    return [f"<b>{e(name)}</b><br><span class='mut'>{e(SCENE_NOTE.get(name, ''))}</span>",
            (f"<span class='{'ok' if nom[0] == nom[1] else 'bad'}'>{nom[0]}/{nom[1]}</span>" if nom else "—") + f"<br><span class='mut'>{e(sw_txt)}</span>",
            (f"<span class='bad'>{len(d3_err)}: {e(', '.join(d3_err))}</span>" if d3_err else ("<span class='ok'>all drawn</span>" if d3 else "—")),
            (f"{min(wr):.1f}–{max(wr):.1f} cm" if wr else "—") + (f"<br><span class='mut'>unwitnessed: {e(', '.join(unw))}</span>" if unw else ""),
            (f"<span class='bad'>{e(', '.join(far))}</span>" if far else ("none" if d1 else "—")),
            f(a2.get("edge_dist_cm_median"), 1, " cm") if a2 else "—",
            (f"<span class='bad'>{e('; '.join(d5_err))}</span>" if d5_err else ("<span class='ok'>all on the table</span>" if d5 else "—"))]


# ------------------------------------------------------------------ sections
def sec_b3(name, ep):
    if not ep:
        return "<p class='mut'>B3 not run on this scene.</p>"
    sweeps = list(ep["sweeps"])
    vids = {}
    for r in ep["sweeps"].get("nominal", {}).get("rows", []):
        if r.get("video"):
            key = f"{r['object']} | {r['probe']}" + (f" -> {r['ref']}" if r.get("ref") else "")
            vids[key] = f"{name}/{r['video']}"
    stops, fails = {}, []
    for sw, v in ep["sweeps"].items():
        for r in v["rows"]:
            key = f"{r['object']} | {r['probe']}" + (f" -> {r['ref']}" if r.get("ref") else "")
            stops[(key, sw)] = r.get("stopped") or r.get("result") or ""
            if r.get("passed") is False:
                fails.append((key, sw, r.get("stopped") or r.get("stage", "")))
    rows = []
    for key, v in ep["table"].items():
        cells = [e(key)]
        for sw in sweeps:
            s = v.get(sw, "—")
            cells.append(f"<span class='{cls_pass(s)}' title='{e(stops.get((key, sw), ''))}'>{e(s)}</span>")
        cells.append(f"<a href='{e(vids[key])}'>video</a>" if key in vids else "")
        rows.append(cells)
    h = [table(["object | probe"] + sweeps + ["nominal video"], rows, raw=True)]
    if fails:
        h.append("<details><summary>Why each failure stopped (" + str(len(fails)) + ")</summary>" +
                 table(["object | probe", "sweep", "stopped"], [[a, b, c] for a, b, c in fails]) + "</details>")
    return "\n".join(h)


def sec_image(name, ic):
    if not ic:
        return "<p class='mut'>Image checks not run on this scene.</p>"
    c = ic["checks"]
    h = [f"<p class='meta'>Frames (before anything is touched): {e(ic['frames'])}. Composite: {e(ic.get('composite'))}.</p>"]
    if "d3" in c:
        h.append("<h3>D3 · nothing physical that's invisible</h3>")
        h.append(table(["object", "exterior", "wrist", "error"],
                       [[e(o), e(f"{v['ext']['label']} ({v['ext']['max_px']} px)"), e(f"{v['wrist']['label']} ({v['wrist']['max_px']} px)"),
                         "<span class='bad'>yes</span>" if v["error"] else "no"] for o, v in c["d3"].items()], raw=True))
    if "d1" in c:
        h.append("<h3>D1 · objects drawn where physics has them</h3><p class='meta'>Offset real − sim centroid in cm at the object's depth, "
                 "median over frames; wrist = independent witness (pose from the joints); exterior = the camera the scene was built from "
                 "(compares the build with its own input). FAR = the real object was found, but nowhere near the sim one.</p>")
        rows = []
        for o, v in c["d1"].items():
            for cam in ("wrist", "ext"):
                x = v[cam]
                lab = x["label"]
                rows.append([o, cam, x["n_frames"], f(x.get("offset_cm_median"), 2), f(x.get("dx_cm_median"), 2), f(x.get("dy_cm_median"), 2),
                             f(x.get("iou_median"), 2),
                             f"<span class='{'bad' if lab.startswith('FAR') else 'mut' if lab == 'unwitnessed' else ''}'>{e(lab)}</span>",
                             e(", ".join(f"{k}: {n}" for k, n in x.get("skipped", {}).items()))])
        h.append(table(["object", "view", "frames", "offset cm", "dx", "dy", "IoU", "label", "frames skipped"], rows, raw=True))
        h.append("<div class='grid2'>" + img(f"{name}/image_checks/d1_wrist.jpg", "wrist: white = sim, black = real (red = nearest real, FAR)")
                 + img(f"{name}/image_checks/d1_ext.jpg", "exterior: white = sim, black = real") + "</div>")
    if "a2" in c:
        a = c["a2"]
        h.append("<h3>A2 · table extent and edges</h3>")
        h.append(table(["view", "frames", "table IoU", "edge distance cm (median)", "independent", "label"],
                       [[k, v["n_frames"], f(v["iou_median"]), f(v["edge_dist_cm_median"], 1), v.get("independent"), v["label"]] for k, v in a["views"].items()], raw=True))
        h.append(f"<p class='meta'>Each object's distance to the nearest built table edge (cm): {e(a['objects_to_edge_cm'])}</p>")
        h.append("<div class='grid2'>" + img(f"{name}/image_checks/a2_wrist.jpg", "wrist: white = sim table, black = real table")
                 + img(f"{name}/image_checks/a2_ext.jpg", "exterior") + "</div>")
    if "d5" in c:
        h.append("<h3>D5 · contacts</h3>")
        rows = [[o, f"<span class='{'bad' if v['error'] else ''}'>{e(v['rests_on'])}</span>", f(v["bottom_minus_table_cm"], 2), f(v["deepest_contact_cm"], 3),
                 e(v.get("contact_shadow_ratio_ext", "—")), e(v.get("contact_shadow_ratio_wrist", "—"))] for o, v in c["d5"].items()]
        h.append(table(["object", "rests on", "bottom − table cm", "deepest contact cm", "contact shadow ext (ours/real)", "contact shadow wrist"], rows, raw=True))
    if "c1" in c and "label" not in c["c1"]:
        h.append("<h3>C1 · similarity per region (policy's 320×180)</h3><p class='meta'>Colour error: mean |ours − real| per channel (0–1); "
                 "edge F1: Canny edges within 1 px; sharpness: Laplacian variance ours/real (&gt;1 = ours sharper). "
                 + e(c["c1"].get("note", "")) + "</p>")
        rows = [[cam, n, f(v["colour_err_mean"], 3), f(v["colour_err_exposure_matched_mean"], 3), f(v["edge_f1"], 2), f(v["sharpness_ratio"], 2), v["n_frames"]]
                for cam in ("ext", "wrist") for n, v in c["c1"][cam].items()]
        h.append(table(["view", "region", "colour err", "colour err (exposure-matched)", "edge F1", "sharpness ratio", "frames"], rows, raw=True))
        h.append(img(f"{name}/image_checks/c1_wrist.jpg", "wrist: real | ours, first frames") + img(f"{name}/image_checks/c1_ext.jpg", "exterior: real | ours"))
    elif "c1" in c:
        h.append(f"<h3>C1–C5</h3><p class='mut'>Not run: {e(c['c1'].get('reason'))}</p>")
    if "c2" in c and "label" not in c["c2"]:
        x = c["c2"]
        h.append("<h3>C2 · background ghosts</h3><p class='meta'>" + e(x["note"]) + "</p>")
        h.append(table(["view", "residue share (median)", "ghost-suspect frames", "unbuilt-or-false-positive frames"],
                       [[cam, f(x[f"residue_share_median_{cam}"], 3), e(x[f"ghost_suspect_frames_{cam}"]), e(x.get(f"unbuilt_or_false_positive_frames_{cam}", "—"))]
                        for cam in ("ext", "wrist")], raw=True))
        h.append(img(f"{name}/image_checks/c2_residue_wrist.jpg", "wrist: real with residue in magenta | ours")
                 + img(f"{name}/image_checks/c2_residue_ext.jpg", "exterior"))
    if "c3" in c and "label" not in c["c3"]:
        x = c["c3"]
        h.append("<h3>C3 · wrist table-texture coverage</h3>")
        h.append(table(["measure", "share observed (real texture)"],
                       [["wrist view, table pixels (median over frames)", f(x["observed_share_wrist_view_median"], 2)],
                        [f"reachable table (≤ {x['reach_m']} m from the base)", f(x["observed_share_reachable_table"], 2)],
                        ["whole table", f(x["observed_share_whole_table"], 2)]] +
                       [[f"around {o} ({k})", f(vv, 2)] for o, v in x["observed_share_around_objects"].items()
                        for k, vv in (v.items() if isinstance(v, dict) else [("ring", v)])], raw=True))
    if "c4" in c:
        x = c["c4"]
        h.append("<h3>C4 · light per view</h3><p class='meta'>" + e(x.get("note", "")) + "</p>")
        rows = []
        for cam in ("ext", "wrist"):
            v = x.get(cam, {})
            ss = v.get("shadow_share", {})
            rows.append([cam, f(v.get("fit_iou"), 3), f(v.get("best_minus_second"), 4), f(v.get("real_shadow_ratio"), 2),
                         f"{f(ss.get('ours_median'), 3)} / {f(ss.get('real_median'), 3)}", f"{f(ss.get('broad_ours_median'), 3)} / {f(ss.get('broad_real_median'), 3)}"])
        h.append(table(["view", "fit overlap", "gap to runner-up", "real shadow darkness", "sharp shadow share ours / real", "broad shadow share ours / real"], rows, raw=True))
        h.append(f"<p class='meta'>Angle between the exterior and wrist light fits: {f(x.get('angle_between_fits_deg'), 1, '°')}</p>")
    if "c5" in c and "label" not in c["c5"]:
        rows = []
        for o, v in c["c5"].items():
            for cam in ("ext", "wrist"):
                x = v.get(cam, {})
                if not x.get("n_frames"):
                    rows.append([o, cam, "unwitnessed", "", "", "", ""])
                    continue
                clip = x.get("ours_clipped") or 0
                rows.append([o, cam, e(x["ours_rgb"]), e(x["real_rgb"]), e(x["err_per_channel"]),
                             f"<span class='{'bad' if clip > 0.1 else ''}'>{clip:.2f}</span>", x["n_frames"]])
        h.append("<h3>C5 · objects as rendered</h3><p class='meta'>Median colour (0–1) inside the object; clipped = share of its pixels with a channel ≥ 250.</p>")
        h.append(table(["object", "view", "ours RGB", "real RGB", "|error| per channel", "ours clipped", "frames"], rows, raw=True))
    return "\n".join(h)


def sec_moge(name, mg):
    if not mg or "views" not in mg:
        return ""
    rows = []
    for v in mg["views"]:
        b = v["ours_vs_physics"]
        rows.append([v["cam"], v["frame"], v.get("variant") or "", e(b["object_residual_cm"]), f(b["table_patch_spread_cm_p90"], 2),
                     e(v.get("real_vs_ours", {}).get("object_residual_cm", "—"))])
    h = ["<h3>D6 · MoGe relative check</h3><p class='meta'>" + e(mg.get("note", "")) + "</p>",
         table(["view", "frame", "variant", "ours vs physics: object residual cm", "table-patch spread cm (p90)", "real vs ours: object residual cm"], rows, raw=True)]
    for v in mg["views"][:2]:
        fn = f"{v['cam']}_f{v['frame']}.jpg"
        h.append(img(f"{name}/moge_check/{fn}", f"{v['cam']} frame {v['frame']}: composite | residual vs physics | real | residual real vs ours"))
    return "\n".join(h)


def sec_c7(name, ed):
    if not ed:
        return ""
    rows = [[cam, f(v["ours_vs_real_median"], 4), f(v["real_motion_median"], 4), f(v["real_cross_episode_median"], 4), e(v["ours_nn_percentile_among_reference"])]
            for cam, v in ed["cameras"].items()]
    return ("<h3>C7 · encoder distance (GR00T backbone)</h3>" +
            table(["view", "ours vs real (same moment)", "real motion", "real cross-episode", "ours NN percentile among real"], rows, raw=True))


def sec_tests():
    T = os.path.join(ROOT, "tests")
    rows = []
    bad = load(os.path.join(T, "b3_orange_bad", "expert_probe.json"))
    if bad:
        s = b3_summary(bad)
        for sw, (p, n, u) in s.items():
            rows.append(["B3", f"orange, {sw}", "every probe fails", f"{p}/{n} pass", "<span class='ok'>caught</span>" if p == 0 else "<span class='bad'>missed</span>"])
    hide = load(os.path.join(T, "ic_hide_yellow", "image_checks.json"))
    if hide:
        v = hide["checks"]["d3"]["yellow_block"]
        rows.append(["D3", "orange, yellow block drawn nowhere (alpha 0), physics kept", "error",
                     e(f"exterior: {v['ext']['label']}; wrist: {v['wrist']['label']}"), "<span class='ok'>caught</span>" if v["error"] else "<span class='bad'>missed</span>"])
    base = load(os.path.join(ROOT, "ep_orange_block_stereo", "image_checks.json"))
    for cp, what in (("ray_p2cm", "exterior camera +2 cm along its line of sight"), ("pitch_3deg", "exterior camera tilted 3°"), ("side_2cm", "exterior camera 2 cm sideways")):
        x = load(os.path.join(T, f"inj_{cp}", "image_checks.json"))
        if not (x and base) or "d1" not in base["checks"] or "a2" not in base["checks"]:
            continue  # results still being written (a run in progress)
        d1, d1b = x["checks"]["d1"], base["checks"]["d1"]
        o = [(k, d1b[k]["wrist"]["offset_cm_median"], v["wrist"]["offset_cm_median"]) for k, v in d1.items()
             if v["wrist"]["offset_cm_median"] is not None and d1b[k]["wrist"]["offset_cm_median"] is not None]
        if o:
            grew = sum(b > a + 0.5 for _, a, b in o)
            rows.append(["D1 wrist", f"orange copy: {what}", "offsets grow", e("; ".join(f"{k} {a:.1f}→{b:.1f} cm" for k, a, b in o)),
                         "<span class='ok'>caught</span>" if grew == len(o) else "<span class='warn'>partly</span>" if grew else "<span class='bad'>missed</span>"])
        a2, a2b = x["checks"]["a2"]["views"]["wrist"]["edge_dist_cm_median"], base["checks"]["a2"]["views"]["wrist"]["edge_dist_cm_median"]
        if a2 is not None and a2b is not None:
            rows.append(["A2 wrist", f"orange copy: {what}", "edge distance grows", f"{a2b:.2f} → {a2:.2f} cm",
                         "<span class='ok'>caught</span>" if a2 > a2b + 0.5 else "<span class='warn'>small change</span>"])
    mi = load(os.path.join(T, "moge_inject", "moge_check.json"))
    mb = load(os.path.join(ROOT, "ep_orange_block_stereo", "moge_check.json"))
    if mi and mb:
        for vi, vb in zip(mi["views"], mb["views"]):
            if vi["cam"] != vb["cam"] or vi["frame"] != vb["frame"]:
                continue
            a = vb["ours_vs_physics"]["object_residual_cm"].get("orange_block")
            b = vi["ours_vs_physics"]["object_residual_cm"].get("orange_block")
            if a is None or b is None:
                continue
            ok = abs(b - a) > 1.0
            rows.append(["D6", f"orange block 2 cm higher in the sim depth only ({vi['cam']} frame {vi['frame']})", "its residual moves ~2 cm",
                         f"{a:+.2f} → {b:+.2f} cm (table spread {vb['ours_vs_physics']['table_patch_spread_cm_p90']} cm)",
                         "<span class='ok'>caught</span>" if ok and vi["cam"] == "ext" else "<span class='warn'>too noisy to read</span>" if vi["cam"] == "wrist" else "<span class='bad'>missed</span>"])
    return table(["check", "injected error", "expected", "observed", "result"], rows, raw=True) if rows else "<p class='mut'>No test results yet.</p>"


DECISIONS = [
    ("10-27", "prim (user's onboarding)", "implres31 builds the unbuilt checklist items next to res31; outline first, no code until the user's go",
     "outline shown in implres31's terminal; res31 listed 7 items it wants built (REAL2SIM_CHECKLIST_V1 §15) and the ones it keeps (calibration, A3, inspect_scene, overrides, build_v1)"),
    ("10-27", "user", "\"start B3, A2, all C and D … I was thinking of using config for task generation … F6 for now\"",
     "built B3, A2, C1–C5, C7, D1, D3, D5, D6; B3 is driven by expert_probe_config.json (the config idea; the same for make_tasks would be a suggestion to res31 via prim, not sent yet); F6 read as parked"),
    ("10-27", "user", "\"can you work in a separate worktree maybe\"", "worktree ../VLA-exp-real2sim-checks, branch exp/real2sim-checks; res31's uncommitted modules imported read-only from the main checkout"),
    ("10-27", "implres31", "B3 sweeps: every block failed at 'reached' under mass x0.5", "own bug: masses changed without mj_setConst, and every block swept at once; fixed (per object, setConst, MuJoCo resets labelled UNSTABLE)"),
    ("10-28", "implres31", "orange block still failed under small sweeps", "cause: the expert's arm stops ~1 cm short of the IK target, so one pad shoved the block; fixed with a closed-loop re-aim before closing (make_tasks' expert has the same open-loop approach: to tell res31)"),
    ("10-28", "implres31", "known-bad B3 cases", "friction x0.05 kept (every probe fails); mass x20 dropped as a bad case (0.2–0.46 kg blocks are liftable)"),
    ("10-28", "implres31", "c11's exterior D1 read 'unwitnessed' while its blocks are 13–21 cm off", "D1 now reports the nearest same-noun detection as FAR instead of hiding it; D5 flags objects on the floor; D3's wording no longer claims 'invisible' when the object may be occluded"),
    ("10-28", "user", "\"none of these are gating as of now right?\"", "confirmed: every check only writes numbers and labels; nothing stops a build or drops a scene (rule 1)"),
    ("10-28", "user", "\"make the results page\"", "this page, generated by report_page.py from the result files"),
]
NOT_TAKEN = [
    ("Rewriting C6 (image swap)", "already exists in res31's image_swap.py / g7_v2.py; only per-scene GPU runs are missing"),
    ("Running C7's GR00T stage", "needs the GPU (prim's go + disk line) or the user's OK for a slow CPU run"),
    ("Building composite caches for the c-scenes", "would write into res31's scene directories; build_v1 / scene_review make them"),
    ("Labels (consistent / check / likely error)", "wait for the bars (IMAGE_PHYSICS Q1, checklist rule 2)"),
]


def main():
    scenes = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
        n = os.path.basename(d)
        if not os.path.isdir(d) or n in ("logs", "tests"):
            continue
        scenes[n] = {k: load(os.path.join(d, f"{k}.json")) for k in ("expert_probe", "image_checks", "moge_check", "encoder_distance")}
    names = [n for n in SCENE_ORDER if n in scenes] + [n for n in scenes if n not in SCENE_ORDER]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    h = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
         f"<title>Scene Checks v1</title><style>{CSS}</style></head><body><main>",
         "<h1>Scene checks v1: results</h1>",
         f"<p class='meta'>implres31 · generated {now} by <code>experiments/real2sim_checks/report_page.py</code> (worktree "
         f"<code>../VLA-exp-real2sim-checks</code>, branch <code>exp/real2sim-checks</code>) · checks from <a href='{CHECKLIST}'>REAL2SIM_CHECKLIST_V1</a> · "
         "internal, exploratory, numbers UNVERIFIED (the user is the checker) · <b>nothing here gates</b>: every check writes numbers, no build stops and no scene is dropped.</p>",
         "<nav>" + "".join(f"<a href='#{e(n)}'>{e(n)}</a>" for n in names) + "<a href='#tests'>known-bad tests</a><a href='#notrun'>not run</a><a href='#log'>decision log</a></nav>"]
    h.append("<h2>Short answers</h2><div class='card'><ul>"
             "<li><b>Orange (reference):</b> the scripted expert picks, lifts, places, puts in the bowl and stacks every block, at nominal and at mass/friction ×0.5 and ×2. "
             "Its weak points are visual: in the wrist view our blocks are drawn 1–1.7 cm off the real ones (K3 reproduced), the sim table's edges sit ~3 cm inside the real ones, "
             "the wrist composite has ~1.6× the real shadow area (30% vs 19% of the table), and objects render darker than real in the exterior view.</li>"
             "<li><b>The checks catch known errors:</b> a camera moved 2 cm along its line of sight or tilted 3° makes the wrist offsets grow ~1.5–2.5 cm; a hidden object is flagged; "
             "low friction fails every expert probe; a 2 cm depth shift shows in MoGe's exterior residual. See the known-bad table.</li>"
             "<li><b>The 10-27 calibration-test scenes</b> were built from calibrations already known to be wrong (K13/K14); the checks are shown on them as a second, harder test, not as scene verdicts.</li>"
             "<li><b>Not run:</b> C6/C7's GPU parts, and every composite-based check (C1–C5, D6, contact shadows) on scenes without composite caches.</li></ul></div>")
    h.append("<h2>Across scenes</h2>")
    h.append(table(["scene", "B3 expert (nominal; sweeps)", "D3 objects drawn", "D1 wrist offset (median per object)", "D1 FAR (misplaced)", "A2 wrist table edge", "D5 contacts"],
                   [scene_row(n, scenes[n]) for n in names], raw=True))
    h.append("<h2 id='tests'>Known-bad tests (does each check notice an error we put in?)</h2>" + sec_tests())
    for n in names:
        d = scenes[n]
        h.append(f"<h2 id='{e(n)}'>{e(n)}</h2><p class='meta'>{e(SCENE_NOTE.get(n, ''))}</p>")
        h.append("<h3>B3 · general scripted-expert test (+ B4 mass, B5 friction sweeps)</h3><p class='meta'>Hover a cell for why it stopped. "
                 "Stages: reached → closed → lifted → held / placed. UNSTABLE = MuJoCo reset the state (not a fail).</p>" + sec_b3(n, d["expert_probe"]))
        h.append(sec_image(n, d["image_checks"]))
        h.append(sec_moge(n, d["moge_check"]))
        h.append(sec_c7(n, d["encoder_distance"]))
    h.append("<h2 id='notrun'>Not run, and why</h2>" + table(["what", "why"], NOT_TAKEN))
    h.append("<h2 id='log'>Decision log</h2>" + table(["when", "who", "asked / found", "decision / outcome"], DECISIONS))
    h.append("</main></body></html>")
    open(os.path.join(ROOT, "index.html"), "w").write("\n".join(h))
    print("wrote", os.path.join(ROOT, "index.html"))


if __name__ == "__main__":
    main()
