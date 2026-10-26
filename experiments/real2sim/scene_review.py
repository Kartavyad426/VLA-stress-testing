"""Scene review page (user request 10-23): for chosen real frames, the exterior and wrist views as real | ours | real
with our robot (cyan) and objects (white) outlined, plus absolute and relative distance checks, so the user can judge
the scene by eye and say what looks wrong. Sim posed at the replayed recorded joints (<scene>/replay_states.npz).

Numbers (all in the robot base frame, from files the checks already wrote; nothing is fitted here):
  - table top: built z vs stereo (stereo_table_check.json), i.e. the robot base <-> table vertical distance;
  - each object: built centre vs its wrist-stereo anchor (wrist_anchors.json; an anchor is a point on the visible
    surface, not the centre, so ~half an object's size of difference is expected);
  - each object pair: distance between the built centres vs between the anchors (relative layout);
  - real grasp / release vs the sim (scene_checks.json).

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/scene_review.py \
        --scene runs/real2sim/ep_orange_block_stereo:data/real2sim/droid/ep_orange_block --wrist v2 \
        --frames 0,40,63,75,100,126 --out runs/real2sim/scene_review/ep_orange_block_stereo
Exploratory, UNVERIFIED. Local page.
"""
import argparse, html, itertools, json, os, sys
import numpy as np
import cv2
import h5py
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from composite import (Compositor, WristCompositor, WristRenderV2, load_plate, wrist_plate, wrist_table_texture,  # noqa: E402
                       load_gripper_library)

H, W = 720, 1280  # review size = the real cameras’ resolution (GR00T sees 180x320); user 10-23: 640x360 too coarse to judge
CSS = """:root{--bg:#fbfbf9;--fg:#1d1d1b;--muted:#5f5f5a;--line:#dddcd5;--card:#f2f1ec;--bad:#a3321f;--ok:#2c6e3a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#e8e7e2;--muted:#a3a29b;--line:#34332f;--card:#1f1f1d;--bad:#ee8a76;--ok:#7fc98d}}
:root[data-theme="dark"]{--bg:#161615;--fg:#e8e7e2;--muted:#a3a29b;--line:#34332f;--card:#1f1f1d;--bad:#ee8a76;--ok:#7fc98d}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}main{max-width:1300px;margin:0 auto}
h1{font-size:1.5rem;margin:.2em 0}h2{font-size:1.15rem;margin-top:1.6em}.meta{color:var(--muted);font-size:.85rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:14px 0}
.tri{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}figure{margin:0}img{width:100%;border-radius:4px;display:block}
figcaption{font-size:.8rem;color:var(--muted)}table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left}th{background:var(--card)}
.bad{color:var(--bad);font-weight:600}.ok{color:var(--ok);font-weight:600}.wrap{overflow-x:auto}
@media (max-width:760px){.tri{grid-template-columns:1fr}}"""


def outlines(img, geom, groups):
    out = img.copy()
    for ids, col in groups:
        msk = np.isin(geom, ids).astype(np.uint8)
        cs, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, col, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="<scene dir>:<episode dir>")
    ap.add_argument("--frames", default="0,40,63,75,100,126")
    ap.add_argument("--wrist", choices=["v1", "v2"], default="v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--wrist_blur", choices=["depth", "gripper"], default="gripper", help="WristRenderV2 near blur (see policy_rollout)")
    a = ap.parse_args()
    sdir, edir = a.scene.split(":")
    os.makedirs(a.out, exist_ok=True)
    m = mujoco.MjModel.from_binary_path(os.path.join(sdir, "scene.mjb"))
    fit = json.load(open(os.path.join(sdir, "scene_fit.json")))
    per = json.load(open(os.path.join(sdir, "perception.json")))
    st = np.load(os.path.join(sdir, "replay_states.npz"))
    plate = load_plate(sdir, "lama", "frame0+plate")
    comp = Compositor(m, sdir, plate, size=(H, W), verbose=False)
    wser = fit["wrist_camera"]["serial"]
    wp = wrist_plate(m, sdir, edir, comp, wser)
    wrist_mode = a.wrist
    if a.wrist == "v2" and os.path.exists(os.path.join(sdir, "gripper_lib", "gripper_library.npz")) and os.path.exists(os.path.join(sdir, "light_wrist.json")):
        g = h5py.File(os.path.join(edir, "trajectory.h5"))["observation/robot_state/gripper_position"][:]
        kc = int(np.argmax(g > 0.5)) if (g > 0.5).any() else len(g)
        tt = wrist_table_texture(m, sdir, edir, wser, list(range(0, max(kc - 2, 2), 2)))
        wc = WristRenderV2(m, plate, comp, (H, W), wp, tt, load_gripper_library(os.path.join(sdir, "gripper_lib", "gripper_library.npz")),
                           light=json.load(open(os.path.join(sdir, "light_wrist.json"))), near=a.wrist_blur)
    else:
        wrist_mode = "v1"
        wc = WristCompositor(m, plate, comp, (H, W), plate2=wp)
    robot = [i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name.startswith(("fr3", "g_"))]
    free = [m.jnt_bodyid[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    objs = [i for i in range(m.ngeom) if m.geom_bodyid[i] in free]
    rend = mujoco.Renderer(m, H, W)
    rend.enable_segmentation_rendering()
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1
    caps = {s: cv2.VideoCapture(os.path.join(edir, f"{s}.mp4")) for s in (per["cam"], wser)}
    d = mujoco.MjData(m)
    cards = []
    for f in [int(x) for x in a.frames.split(",")]:
        k = int(np.argmin(np.abs(st["real_frames"] - f)))
        d.qpos[:] = st["qpos"][k]
        mujoco.mj_forward(m, d)
        row = []
        for cam, ser, sim in (("ext", per["cam"], comp(d)), ("wrist", wser, wc(d))):
            caps[ser].set(cv2.CAP_PROP_POS_FRAMES, int(st["real_frames"][k]))
            ok, bgr = caps[ser].read()
            real = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
            rend.update_scene(d, camera=cam, scene_option=opt)
            seg = rend.render()
            geom = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
            ov = outlines(real, geom, [(robot, (255, 255, 0)), (objs, (255, 255, 255))])
            names = []
            for tag, im in (("real", real), ("ours", cv2.cvtColor(sim.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)), ("overlay", ov)):
                fn = f"f{f:03d}_{cam}_{tag}.jpg"
                cv2.imwrite(os.path.join(a.out, fn), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
                names.append(fn)
            row.append((cam, names))
        cards.append((f, row))
    # numbers
    rows_abs, rows_rel = [], []
    stc = os.path.join(sdir, "stereo_table_check.json")
    if os.path.exists(stc):
        j = json.load(open(stc))
        for r in j["results"]:
            rows_abs.append((f"table top ({r['cam']} stereo, frame {r['frame']})", f"{100 * j['sim_table_z']:.1f}", f"{100 * r['z_median']:.1f}",
                             r["minus_sim_cm"], 1.0))
    else:
        rows_abs.append(("table top", f"{100 * fit['table']['z']:.1f}", "not measured (no stereo_table_check.json)", None, 1.0))
    anc = os.path.join(sdir, "wrist_anchors.json")
    built = {lab: np.array(o["center"][:2] if len(o["center"]) == 2 else o["center"]) for lab, o in fit["objects"].items()}
    real_pts = {}
    if os.path.exists(anc):
        for lab, an in json.load(open(anc))["anchors"].items():
            base = lab.split("#")[0] if lab.split("#")[0] in built else lab
            if base in built:
                real_pts[base] = np.array(an["point"])
        for lab, p in real_pts.items():
            b = built[lab]
            dxy = 100 * float(np.linalg.norm(b[:2] - p[:2]))
            rows_abs.append((f"{lab}: horizontal, built centre vs wrist-stereo anchor", f"({100 * b[0]:.1f}, {100 * b[1]:.1f})",
                             f"({100 * p[0]:.1f}, {100 * p[1]:.1f})", dxy, 1.5))
        for x, y in itertools.combinations(sorted(real_pts), 2):
            ds = 100 * float(np.linalg.norm(built[x][:2] - built[y][:2]))
            dr = 100 * float(np.linalg.norm(real_pts[x][:2] - real_pts[y][:2]))
            rows_rel.append((f"{x} ↔ {y}", ds, dr, ds - dr))
    sc = os.path.join(sdir, "scene_checks.json")
    if os.path.exists(sc):
        for e in json.load(open(sc))["checks"].get("3_4_5_grasp_events", []):
            rows_abs.append((f"real grasp (frame {e['close_frame']}) vs sim {e['object']}: horizontal", "", "", 100 * e["horiz_m"], 1.5))
            rows_abs.append((f"real grasp (frame {e['close_frame']}) vs sim {e['object']}: vertical", "", "", 100 * e["vert_m"], 1.5))
    T = a.title or f"Scene review: {os.path.basename(os.path.normpath(sdir))}"
    h = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Scene Review</title><style>{CSS}</style></head><body><main>',
         f"<h1>{html.escape(T)}</h1>",
         f'<p class="meta">res31, internal. Scene <code>{html.escape(sdir)}</code>, episode <code>{html.escape(edir)}</code>. Each row: real camera frame | our render of the same moment (sim posed at the recorded joints; exterior = composite on the cleaned real plate, wrist = {"WristRenderV2" if wrist_mode == "v2" else "old WristCompositor (v1)"}) | the real frame with our robot outlined in cyan and our objects in white. If the scene is right, the outlines sit on the real robot and objects. Shown at 1280×720, the real cameras’ resolution (GR00T sees 320×180). Exploratory, UNVERIFIED.</p>',
         "<h2>Distances</h2><p class=\"meta\">Robot base frame, cm. Absolute = where things are relative to the robot; relative = object to object. Wrist-stereo anchors are points on the visible surface, so up to ~half an object's size is expected there.</p>",
         '<div class="wrap"><table><tr><th>Absolute</th><th>ours</th><th>real</th><th>difference (cm)</th><th>limit</th></tr>']
    for name, ours, real, diff, lim in rows_abs:
        cls = "" if diff is None else ("ok" if abs(diff) <= lim else "bad")
        h.append(f"<tr><td>{html.escape(name)}</td><td>{ours}</td><td>{html.escape(str(real))}</td><td class='{cls}'>{'' if diff is None else f'{diff:+.1f}'}</td><td>{lim}</td></tr>")
    h.append("</table></div>")
    if rows_rel:
        h.append('<div class="wrap"><table><tr><th>Relative (object pair)</th><th>ours</th><th>real (anchors)</th><th>difference (cm)</th></tr>')
        for name, ds, dr, df in rows_rel:
            h.append(f"<tr><td>{html.escape(name)}</td><td>{ds:.1f}</td><td>{dr:.1f}</td><td class='{'ok' if abs(df) <= 1.5 else 'bad'}'>{df:+.1f}</td></tr>")
        h.append("</table></div>")
    else:
        h.append('<p class="meta">Relative distances: no wrist-stereo anchors for this scene.</p>')
    iss = os.path.join(a.out, "issues.json")  # the user's per-frame notes and their status (10-23), kept by hand
    issues = json.load(open(iss))["issues"] if os.path.exists(iss) else []
    if issues:
        h.append('<h2>Issues</h2><p class="meta">What looks wrong, frame by frame: raised → cause found → fix agreed → fixed (checked by the user). Kept in <code>issues.json</code> next to this page.</p>'
                 '<div class="wrap"><table><tr><th>#</th><th>Frame</th><th>Camera</th><th>Seen</th><th>Cause</th><th>Fix</th><th>Status</th></tr>')
        for i in issues:
            h.append("<tr>" + "".join(f"<td>{html.escape(str(i.get(k, '')))}</td>" for k in ("id", "frame", "camera", "seen", "cause", "fix", "status")) + "</tr>")
        h.append("</table></div>")
    h.append("<h2>Frames</h2>")
    for f, row in cards:
        h.append(f'<div class="card"><b>Frame {f}</b>')
        mine = [i for i in issues if str(f) in [x.strip() for x in str(i["frame"]).split(",")] or i["frame"] == "all"]
        if mine:
            h.append('<p class="meta">Issues here: ' + "; ".join(f"#{i['id']} ({i['camera']}) {html.escape(i['seen'])} [{i['status']}]" for i in mine) + "</p>")
        for cam, names in row:
            h.append(f'<p class="meta">{"exterior camera" if cam == "ext" else "wrist camera"}</p><div class="tri">')
            for fn, cap in zip(names, ("real", "ours", "real + our outlines")):
                h.append(f'<figure><a href="{fn}"><img src="{fn}" alt="{cap}" loading="lazy"></a><figcaption>{cap}</figcaption></figure>')
            h.append("</div>")
        h.append("</div>")
    h.append("</main></body></html>")
    open(os.path.join(a.out, "index.html"), "w").write("\n".join(h))
    print("wrote", os.path.join(a.out, "index.html"))


if __name__ == "__main__":
    main()
