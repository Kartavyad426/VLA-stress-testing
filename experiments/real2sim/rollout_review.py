"""Rollout review pages + a hub index (user request 10-23: "see how a successful rollout looks first, make multiple
indexes"). Per rollout (policy_rollout output dir): what the policy saw (rollout.mp4 = exterior | wrist at its input
size) at the key moments - start, reached, closed on the target, lifted, gripper opened after the lift, end - plus
every --every_s seconds, and a timeline of the target's height and the gripper command. The hub (<out>/index.html)
links every rollout page and any scene review pages given with --scene_page.

    .venvs/real2sim/bin/python experiments/real2sim/rollout_review.py --out runs/real2sim/review \
        --rollout runs/real2sim/policy/batch_yfit_v2/02_put_in_bowl_yellow_block/v0 \
        --scene_page runs/real2sim/scene_review/ep_orange_block_stereo/index.html
Exploratory, UNVERIFIED. Local pages.
"""
import argparse, html, json, os, sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

FPS = 15
CSS = """:root{--bg:#fbfbf9;--fg:#1d1d1b;--muted:#5f5f5a;--line:#dddcd5;--card:#f2f1ec;--a:#2a6fb0;--b:#c0661d;--ok:#2c6e3a;--bad:#a3321f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#e8e7e2;--muted:#a3a29b;--line:#34332f;--card:#1f1f1d;--a:#6aa7e0;--b:#e29a5a;--ok:#7fc98d;--bad:#ee8a76}}
:root[data-theme="dark"]{--bg:#161615;--fg:#e8e7e2;--muted:#a3a29b;--line:#34332f;--card:#1f1f1d;--a:#6aa7e0;--b:#e29a5a;--ok:#7fc98d;--bad:#ee8a76}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}main{max-width:1100px;margin:0 auto}
h1{font-size:1.5rem;margin:.2em 0}h2{font-size:1.15rem;margin-top:1.6em}.meta{color:var(--muted);font-size:.85rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:12px 0}
img{width:100%;border-radius:4px;display:block}figure{margin:0}figcaption{font-size:.8rem;color:var(--muted)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:6px 0}.small{max-width:420px}
@media (max-width:760px){.two{grid-template-columns:1fr}}a{color:var(--a)}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left}th{background:var(--card)}
svg text{fill:var(--muted);font-size:11px}"""


def head(title):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>")


def timeline(z_cm, grip, marks):
    """inline SVG: target height above its start (cm) and gripper closed (band)."""
    n, Wd, Hd, pad = len(z_cm), 1000, 160, 30
    zmax = max(5.0, float(np.max(z_cm)) * 1.1)
    x = lambda i: pad + (Wd - 2 * pad) * i / max(n - 1, 1)  # noqa: E731
    y = lambda v: Hd - pad - (Hd - 2 * pad) * v / zmax  # noqa: E731
    s = [f'<svg viewBox="0 0 {Wd} {Hd}" width="100%" role="img" aria-label="target height and gripper over time">']
    i = 0
    while i < n:  # gripper-closed bands
        if grip[i]:
            j = i
            while j < n and grip[j]:
                j += 1
            s.append(f'<rect x="{x(i):.1f}" y="{pad}" width="{x(j - 1) - x(i) + 1:.1f}" height="{Hd - 2 * pad}" fill="var(--b)" opacity="0.18"/>')
            i = j
        else:
            i += 1
    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(z_cm))
    s.append(f'<polyline points="{pts}" fill="none" stroke="var(--a)" stroke-width="2"/>')
    s.append(f'<line x1="{pad}" y1="{y(0):.1f}" x2="{Wd - pad}" y2="{y(0):.1f}" stroke="var(--line)"/>')
    for name, k in marks:
        s.append(f'<line x1="{x(k):.1f}" y1="{pad}" x2="{x(k):.1f}" y2="{Hd - pad}" stroke="var(--muted)" stroke-dasharray="3 3"/>'
                 f'<text x="{x(k) + 3:.1f}" y="{pad - 6}">{html.escape(name)}</text>')
    s.append(f'<text x="{pad}" y="{Hd - 8}">0 s</text><text x="{Wd - pad - 30}" y="{Hd - 8}">{n / FPS:.0f} s</text>'
             f'<text x="{pad + 4}" y="{y(zmax * 0.9):.1f}">target height above start (cm, blue); gripper closed (shaded)</text></svg>')
    return "".join(s)


def hires(r, z, steps, episode, size=(720, 1280)):
    """Re-simulate the rollout from its recorded actions (replay_probe; reproduces the recorded fingertip track) and
    render the given steps at the real cameras' resolution with the same renderers the rollout used. For people to look
    at: the policy itself saw the 320x180 frames in rollout.mp4. Returns {step: (exterior BGR, wrist BGR, replay err m)}."""
    import mujoco, h5py
    sys.path.insert(0, HERE)
    import replay_probe as rp
    from composite import Compositor, WristCompositor, WristRenderV2, load_plate, wrist_plate, wrist_table_texture, load_gripper_library
    sdir = os.path.join(REPO, "runs/real2sim", r["scene"])
    S = rp.Scene(f"{sdir}:{episode}", r.get("grip_physics", "scene"))
    m = S.m
    ti = [i for i, t in enumerate(S.tasks) if t["text"] == r["instruction"]][0]
    seed = rp.layout_seed(r, ti, os.path.join(REPO, "runs/real2sim/tasks_stereo/ep_orange_block_stereo"), r["_ep_dir"])
    plate = load_plate(sdir, "lama", "frame0+plate")
    comp = Compositor(m, sdir, plate, size=size, verbose=False)
    ser = S.fit["wrist_camera"]["serial"]
    wp = wrist_plate(m, sdir, episode, comp, ser)
    if r.get("wrist_render") == "v2":
        g = h5py.File(os.path.join(episode, "trajectory.h5"))["observation/robot_state/gripper_position"][:]
        kc = int(np.argmax(g > 0.5))
        tt = wrist_table_texture(m, sdir, episode, ser, list(range(0, kc - 2, 2)))
        wc = WristRenderV2(m, plate, comp, size, wp, tt, load_gripper_library(os.path.join(sdir, "gripper_lib", "gripper_library.npz")),
                           light=json.load(open(os.path.join(sdir, "light_wrist.json"))))
    else:
        wc = WristCompositor(m, plate, comp, size, plate2=wp)
    d = mujoco.MjData(m)
    d.qpos[S.arm_q] = S.q_start
    if r["variant"] > 0:
        rp.randomise(m, d, np.random.default_rng(seed), S.free)
    d.ctrl[S.arm_u] = S.q_start
    d.ctrl[S.grip_u] = S.g_lo
    for _ in range(int(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    out, want = {}, set(steps)
    for k, act in enumerate(z["action"]):
        d.ctrl[S.arm_u] = np.clip(act[10:17], S.lo, S.hi)
        d.ctrl[S.grip_u] = S.g_hi if act[9] > 0.5 else S.g_lo
        for _ in range(S.n_sub):
            mujoco.mj_step(m, d)
        if k in want:
            err = float(np.abs(d.site_xpos[S.pinch] - z["pinch_ach"][k]).max())
            ext = cv2.cvtColor(comp(d).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            wr = cv2.cvtColor(wc(d).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            out[k] = (ext, wr, err)
        if k >= max(want):
            break
    return out


def rollout_page(rdir, out, name, episode):
    r = json.load(open(os.path.join(rdir, "rollout.json")))
    z = np.load(os.path.join(rdir, "rollout.npz"))
    tgt, grip = z["target_pos"], z["grip_cmd"].astype(bool)
    zc = 100 * (tgt[:, 2] - tgt[0, 2])
    fs = r["stages"].get("first_step", {})
    marks = [("start", 0)] + [(k.replace("_at_target", ""), fs[k]) for k in ("reached", "closed_at_target", "lifted") if fs.get(k) is not None]
    if fs.get("lifted") is not None:
        op = [k for k in range(fs["lifted"], len(grip)) if not grip[k]]
        if op:
            marks.append(("opened", op[0]))
    marks.append(("end", len(grip) - 1))
    marks = sorted(set(marks), key=lambda t: t[1])
    cap = cv2.VideoCapture(os.path.join(rdir, "rollout.mp4"))
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    r["_ep_dir"] = os.path.basename(os.path.dirname(os.path.normpath(rdir)))
    hi = hires(r, z, [k for _, k in marks], episode)
    cards = []
    for lab, k in marks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, im = cap.read()
        if not ok:
            continue
        fn = f"step{k:03d}_policy.png"
        cv2.imwrite(os.path.join(d, fn), im)
        ext, wr, err = hi[k]
        fe, fw = f"step{k:03d}_ext.jpg", f"step{k:03d}_wrist.jpg"
        cv2.imwrite(os.path.join(d, fe), ext, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(os.path.join(d, fw), wr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cards.append((lab, k, fn, fe, fw, err))
    ok = r["success"]
    st = r.get("stages", {})
    tr = r.get("tracking", {})
    h = [head(f"Rollout {name}"), f'<p class="meta"><a href="../index.html">← all review pages</a></p>',
         f"<h1>{html.escape(r['instruction'])} · v{r['variant']}</h1>",
         f'<p class="meta">{html.escape(rdir)} · policy {html.escape(str(r.get("policy")))} · wrist render {html.escape(str(r.get("wrist_render")))} · '
         f'grip physics {html.escape(str(r.get("grip_physics", "scene")))} · scene {html.escape(r["scene"])}. Large images: the episode re-simulated from its '
         f'recorded actions and rendered at 1280×720 (the real cameras\' resolution) with the same renderers, for looking at. Small strip: exactly what the '
         f'policy saw (exterior | wrist at its 320×180 input). Exploratory, UNVERIFIED.</p>',
         f'<p>Result: <span class="{"ok" if ok else "bad"}">{"success" if ok else "failed"}</span> · reached {st.get("reached")} · '
         f'closed on the block {st.get("closed_at_target")} · lifted {st.get("lifted")} · placed {st.get("placed")} · arm blocked {tr.get("blocked")}</p>',
         '<div class="card">' + timeline(zc, grip, marks) + "</div>"]
    for lab, k, fn, fe, fw, err in cards:
        h.append(f'<div class="card"><b>{html.escape(lab)}</b> <span class="meta">step {k} · {k / FPS:.1f} s · target {zc[k]:+.1f} cm · gripper {"closed" if grip[k] else "open"}'
                 f' · re-simulation matches the recorded run to {err * 1000:.2f} mm</span>'
                 f'<div class="two"><figure><a href="{fe}"><img class="hi" src="{fe}" alt="exterior" loading="lazy"></a><figcaption>exterior</figcaption></figure>'
                 f'<figure><a href="{fw}"><img class="hi" src="{fw}" alt="wrist" loading="lazy"></a><figcaption>wrist</figcaption></figure></div>'
                 f'<figure class="small"><a href="{fn}"><img src="{fn}" alt="policy input" loading="lazy"></a><figcaption>what the policy saw (320×180 each)</figcaption></figure></div>')
    h.append("</main></body></html>")
    open(os.path.join(d, "index.html"), "w").write("\n".join(h))
    return dict(name=name, instruction=r["instruction"], variant=r["variant"], success=ok, lifted=st.get("lifted"),
                batch=os.path.basename(os.path.dirname(os.path.dirname(os.path.normpath(rdir)))), wrist=r.get("wrist_render"),
                grip=r.get("grip_physics", "scene"), scene=r["scene"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rollout", action="append", default=[])
    ap.add_argument("--episode", default=os.path.join(REPO, "data/real2sim/droid/ep_orange_block"))
    ap.add_argument("--scene_page", action="append", default=[], help="scene_review.py index.html to link from the hub")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for rd in a.rollout:
        rd = os.path.normpath(rd)
        name = f"{os.path.basename(os.path.dirname(os.path.dirname(rd)))}__{os.path.basename(os.path.dirname(rd))}__{os.path.basename(rd)}"
        rows.append(rollout_page(rd, a.out, name, a.episode))
    h = [head("Real2sim Review"), "<h1>Real-to-sim review pages</h1>",
         '<p class="meta">res31, internal. Scene pages: the real episode replayed in our scene, real vs ours with outlines, distances and the issue list. '
         "Rollout pages: what the policy saw at each key moment of one episode. Exploratory, UNVERIFIED.</p><h2>Scenes</h2><ul>"]
    for p in a.scene_page:
        h.append(f'<li><a href="{html.escape(os.path.relpath(p, a.out))}">{html.escape(os.path.dirname(p))}</a></li>')
    h.append('</ul><h2>Rollouts</h2><div class="card" style="overflow-x:auto"><table><tr><th>Result</th><th>Task</th><th>Batch</th><th>Wrist render</th><th>Grip physics</th><th>Scene</th></tr>')
    for r in sorted(rows, key=lambda x: (not x["success"], x["instruction"])):
        h.append(f'<tr><td class="{"ok" if r["success"] else "bad"}">{"success" if r["success"] else "failed"}</td>'
                 f'<td><a href="{r["name"]}/index.html">{html.escape(r["instruction"])} · v{r["variant"]}</a></td><td>{html.escape(r["batch"])}</td>'
                 f'<td>{html.escape(str(r["wrist"]))}</td><td>{html.escape(str(r["grip"]))}</td><td>{html.escape(r["scene"])}</td></tr>')
    h.append("</table></div></main></body></html>")
    open(os.path.join(a.out, "index.html"), "w").write("\n".join(h))
    print("wrote", os.path.join(a.out, "index.html"), len(rows), "rollouts")


if __name__ == "__main__":
    main()
