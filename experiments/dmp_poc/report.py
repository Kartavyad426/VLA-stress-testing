"""Self-contained showcase page for the DMP PoC (docs/DMP_POC_METHOD.html §9a, file 9).

Per scenario: the failure rollout (video, events on a clickable timeline, the
root marked), the demo search (3 matched human demos as video, the matched
moment marked, distances), and the corrective rollouts (DMP and linear-bridge
videos, all four arms in a table with the four checks). Every video is
embedded as base64, so the page is one file.

    <main>/.venvs/libero-plus/bin/python experiments/dmp_poc/report.py [out.html]
    default out: <main>/docs/DMP_POC_SHOWCASE.html

UNVERIFIED: numbers from new code, not yet checked by val; the page says so.
"""
from __future__ import annotations

import base64
import html
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from restore import WORKTREE, main_repo, r047_rows, scene_of  # noqa: E402
from demos import task_name  # noqa: E402

RUNS = os.path.join(WORKTREE, "runs/dmp_poc")
VID = os.path.join(RUNS, "videos/web")   # small re-encodes (crf 30) of runs/dmp_poc/videos
FPS = 20
ARM_LABEL = {"dmp": "DMP (velocity-matched)", "dmp_vector": "DMP, vector-wise scaling",
             "dmp_rest": "DMP from rest", "linear": "Linear bridge (I-Gen style)"}
SCEN_NOTE = {
    "A": "Near root: the toy case. The gripper closes 7.9 cm from the bowl; it lifts 2 cm and slips.",
    "B": "Same rollout, the root kept by the 'first unrecovered' rule: the bowl is released 5.2 cm off the plate centre.",
    "C": "A clean miss: one attempt, closes 13.9 cm from the bowl, nothing moves.",
    "D": "Far root: closes 40.8 cm from the target bowl (13.9 cm from the other bowl).",
    "D_backup": "Far root, clean miss: closes 28.6 cm from the target bowl.",
}


def demo_len(task, idx):
    from demos import load
    return next(len(d.eef_pos) for d in load(task) if d.idx == idx)


def b64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return "data:video/mp4;base64," + base64.b64encode(f.read()).decode()


def esc(x):
    return html.escape(str(x))


def video(path, vid_id, markers, n_frames, first_frame=0, caption=""):
    """A video with a timeline of clickable markers: [(frame, label, kind)]."""
    src = b64(path)
    if src is None:
        return f'<div class="missing">video not rendered yet ({esc(os.path.basename(path))})</div>'
    ticks = []
    span = max(1, n_frames - 1)
    for fr, label, kind in markers:
        x = 100.0 * (fr - first_frame) / span
        if 0 <= x <= 100:
            ticks.append(f'<button class="tick {kind}" style="left:{x:.2f}%" title="{esc(label)} (t={fr})" '
                         f'onclick="seek(\'{vid_id}\',{(fr - first_frame) / FPS:.3f})"></button>')
    jump = "".join(f'<button class="jump {kind}" onclick="seek(\'{vid_id}\',{(fr - first_frame) / FPS:.3f})">'
                   f'{esc(label)} · t={fr}</button>' for fr, label, kind in markers if kind in ("root", "match"))
    return (f'<figure><video id="{vid_id}" src="{src}" controls muted playsinline preload="metadata" '
            f'data-first="{first_frame}"></video>'
            f'<div class="timeline">{"".join(ticks)}</div>'
            f'<div class="tlabel"><span class="fr" id="{vid_id}_t">t = {first_frame}</span>{jump}</div>'
            f'<figcaption>{caption}</figcaption></figure>')


def section(name, rows, match_rows, roots_by_row, fid):
    rs = [r for r in rows if r["scenario"] == name]
    if not rs:
        return ""
    r0 = rs[0]
    row_idx, t_root = r0["row"], r0["t_root"]
    row = r047_rows()[row_idx]
    rr = roots_by_row[row_idx]
    task = task_name(row["instruction"])
    m = next(x for x in match_rows if x["scenario"] == name)
    pert = ", ".join(f"{k} {v}" for k, v in row["perturbation"].items()) or "none"
    ev = next(e for e in rr["events"] if e["restore_t"] == t_root and e["mode"] not in ("grasp.ok", "place.ok"))
    # 1) failure
    marks = []
    for e in rr["events"]:
        kind = "ok" if e["mode"] in ("grasp.ok", "place.ok", "grasp.post_place") else "fail"
        marks.append((e["t"], e["mode"], kind))
    marks.append((t_root, f"ROOT ({ev['mode']})", "root"))
    n_fail = len(row["steps"])
    fidf = fid.get(f"fail_{name}", {})
    cap = (f"R-047 row {row_idx} · scene {scene_of(row)} · perturbation: {esc(pert)} · "
           f"{len(row['steps']) - 1} steps, {'success' if row['success'] else 'failed (timeout)'}. "
           f"Ticks: red = failure events, green = grasp/place ok, the tall marker = the root. "
           f"Replay deviation from the log: {fidf.get('max_dev_m', float('nan')) * 1000:.2f} mm.")
    fail_html = video(os.path.join(VID, f"fail_{name}.mp4"), f"fail_{name}", marks, n_fail, 0, cap)
    roots_line = (f"<b>First failure:</b> {esc(rr['root_first']['mode'])} at t = {rr['root_first']['t']} · "
                  f"<b>First unrecovered:</b> {esc(rr['root_unrecovered']['mode'])} at t = {rr['root_unrecovered']['t']}")
    # 2) search
    cards = []
    for k, mm in enumerate(m["matches"]):
        vp = os.path.join(VID, f"demo_{task[:40]}_{mm['demo']}.mp4")
        tags = ", ".join(mm["tags"]) if mm["tags"] else "clean"
        gap = "covered" if mm["d_over_gap"] <= 1 else "coverage gap"
        ps, pe = mm["phase_range"]
        cap = (f"<b>#{k + 1}: demo {mm['demo']}</b> ({esc(tags)}) · matched at t* = {mm['t_star']} in phase "
               f"{m['phase']} (steps {ps}–{pe}) · d/d_gap = <b>{mm['d_over_gap']:.2f}</b> ({gap})")
        cards.append(video(vp, f"demo_{name}_{k}", [(mm["t_star"], "matched moment", "match"),
                                                    (pe, "gripper event", "ok")],
                           demo_len(task, mm["demo"]), 0, cap).replace('preload="metadata"', 'preload="metadata" class="small"'))
    # 3) corrections
    trs = []
    for r in sorted(rs, key=lambda x: (x["match_rank"], list(ARM_LABEL).index(x["arm"]))):
        c = r["checks"]
        mark = lambda b: '<span class="y">✓</span>' if b else '<span class="n">✗</span>'  # noqa: E731
        trs.append(f"<tr><td>#{r['match_rank'] + 1} demo {r['demo']}</td><td>{esc(ARM_LABEL[r['arm']])}</td>"
                   f"<td>{'<b class=y>kept</b>' if r['kept'] else '<span class=n>rejected</span>'}</td>"
                   f"<td>{mark(c['success'])} {'' if r['success_step'] is None else 'at +' + str(r['success_step'])}</td>"
                   f"<td>{mark(c['jerk_ok'])} {r['jerk_max']:.0f}/{r['jerk_limit']:.0f}</td>"
                   f"<td>{mark(c['speed_ok'])} {r['speed_max']:.2f}/{r['speed_limit']:.2f} m/s, sat {r['saturated_frac']:.0%}</td>"
                   f"<td>{mark(c['collateral_ok'])} {max(r['moved_m'].values(), default=0) * 100:.1f} cm</td>"
                   f"<td>{'yes' if r['fits_in_280'] else 'no'}</td></tr>")
    table = ("<div class=tw><table><tr><th>Match</th><th>Arm</th><th>Verdict</th><th>Success</th>"
             "<th>Jerk (m/s³ vs demo p99)</th><th>Speed</th><th>Other objects moved</th><th>Fits in 280</th></tr>"
             + "".join(trs) + "</table></div>")
    fixes = []
    for r in sorted(rs, key=lambda x: x["match_rank"]):
        if r["arm"] not in ("dmp", "linear"):
            continue
        key = f"{name}_m{r['match_rank']}_{r['arm']}"
        ff = fid.get(f"fix_{key}", {})
        first = ff.get("first_frame", max(0, t_root - 20))
        n = ff.get("frames", t_root + 1 + r["steps"])
        mk = [(t_root, "ROOT: correction starts", "root")]
        for sgm in r["plan"]["segments"]:
            mk.append((t_root + sgm["start_step"], f"segment {sgm['phase']}", "ok"))
        if r["success_step"] is not None:
            mk.append((t_root + r["success_step"], "success", "ok"))
        cap = (f"#{r['match_rank'] + 1} demo {r['demo']} · {esc(ARM_LABEL[r['arm']])} · "
               f"<b>{'KEPT' if r['kept'] else 'rejected'}</b> · 20 frames of the failure before the root, then the "
               f"correction{'; clearance lift ' + format(r['plan'].get('clearance_lift_m', 0) * 100, '.1f') + ' cm first' if r['plan'].get('clearance_lift_m') else ''}.")
        fixes.append(video(os.path.join(VID, f"fix_{key}.mp4"), f"fix_{key}", mk, n, first, cap))
    kept = sum(r["kept"] for r in rs)
    return f"""
<section id="s_{name}">
<h2>Scenario {esc(name)} · {esc(ev['mode'])} at t = {ev['t']} <span class="pill">{kept}/{len(rs)} kept</span></h2>
<p class="note">{esc(SCEN_NOTE.get(name, ''))}</p>
<p class="meta">{esc(row['instruction'])}</p>
<h3>1 · The failure</h3>
{fail_html}
<p>{roots_line}. The correction restarts at step <b>{t_root}</b> (phase {esc(m['phase'])}: {'bowl not held' if m['phase'] == 'A' else 'bowl held'}).
The restore replays the logged actions and matches the log to 0.000 mm.</p>
<h3>2 · Search: the 3 nearest human demos</h3>
<p>Search over the {esc(task.replace('_', ' '))} demos, phase {esc(m['phase'])}, by hand position relative to the
{'bowl' if m['phase'] == 'A' else 'plate'}, orientation, velocity and gripper opening. d_gap = {m['d_gap']:.2f}:
how far the demos are from each other (95th percentile); d/d_gap &gt; 1 is a coverage gap.</p>
<p>Coverage: the nearest human demo state at all is <b>{m['d_cover_over_gap']:.2f}</b> × d_gap
({'covered' if m['d_cover_over_gap'] <= 1 else 'a coverage gap'}). The matches used for execution must leave ≥ 1 s of the
phase, so their distances can be larger.</p>
<div class="grid3">{''.join(cards)}</div>
<h3>3 · Corrections: the matched demo bent from the root to the goal, executed in the sim</h3>
{table}
<div class="grid2">{''.join(fixes)}</div>
</section>"""


def build(out):
    rows = json.load(open(os.path.join(RUNS, "t3/results.json")))["rows"]
    match_rows = json.load(open(os.path.join(RUNS, "match/scenarios.json")))["rows"]
    roots_by_row = {}
    with open(os.path.join(RUNS, "roots.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            roots_by_row[r["row"]] = r
    fp = os.path.join(RUNS, "videos/fidelity.json")
    fid = json.load(open(fp)) if os.path.exists(fp) else {}
    names = []
    for r in rows:
        if r["scenario"] not in names:
            names.append(r["scenario"])
    arms = list(ARM_LABEL)
    summ = "".join(
        f"<tr><td><a href='#s_{n}'>{esc(n)}</a></td>"
        + "".join(f"<td>{sum(r['kept'] for r in rows if r['scenario'] == n and r['arm'] == a)}/"
                  f"{sum(1 for r in rows if r['scenario'] == n and r['arm'] == a)}</td>" for a in arms)
        + f"<td>{next(m for m in match_rows if m['scenario'] == n)['d_cover_over_gap']:.2f}</td></tr>"
        for n in names)
    body = "".join(section(n, rows, match_rows, roots_by_row, fid) for n in names)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DMP PoC Showcase</title>
<style>
:root{{--bg:#fbfbf9;--fg:#1d1d1b;--muted:#5f5f5a;--line:#dddcd5;--accent:#2f5d8a;--warn:#8a4b12;--card:#f2f1ec;--ok:#2e7d3a;--bad:#b23a2e}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#161615;--fg:#e8e7e2;--muted:#a3a29b;--line:#34332f;--accent:#8fb5dc;--warn:#e0a26a;--card:#1f1f1d;--ok:#7fcf8c;--bad:#ef8a7e}}}}
:root[data-theme="dark"]{{--bg:#161615;--fg:#e8e7e2;--muted:#a3a29b;--line:#34332f;--accent:#8fb5dc;--warn:#e0a26a;--card:#1f1f1d;--ok:#7fcf8c;--bad:#ef8a7e}}
body{{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:1100px;margin:0 auto}}
h1{{font-size:1.6rem;margin:.2em 0}}h2{{font-size:1.25rem;margin-top:2.2em;border-bottom:1px solid var(--line);padding-bottom:.2em}}h3{{font-size:1rem;margin:1.2em 0 .3em}}
.meta,.note{{color:var(--muted);font-size:.9rem}} .note{{color:var(--fg)}}
.status{{background:var(--card);border-left:3px solid var(--warn);padding:8px 12px;margin:12px 0}}
.pill{{font-size:.8rem;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:1px 8px;vertical-align:middle}}
table{{border-collapse:collapse;width:100%;font-size:.85rem;margin:8px 0}} .tw{{overflow-x:auto}}
th,td{{border:1px solid var(--line);padding:5px 7px;text-align:left;vertical-align:top}} th{{background:var(--card)}}
.y{{color:var(--ok)}} .n{{color:var(--bad)}} a{{color:var(--accent)}}
figure{{margin:8px 0 16px}} video{{width:100%;background:#000;border-radius:4px;display:block}}
figcaption{{font-size:.82rem;color:var(--muted);margin-top:4px}}
.timeline{{position:relative;height:16px;background:var(--card);border:1px solid var(--line);border-radius:3px;margin-top:4px}}
.tick{{position:absolute;top:2px;width:3px;height:10px;padding:0;border:0;margin-left:-1px;cursor:pointer;background:var(--muted)}}
.tick.fail{{background:var(--bad)}} .tick.ok{{background:var(--ok)}} .tick.root{{top:-4px;height:22px;width:4px;background:var(--warn)}} .tick.match{{top:-4px;height:22px;width:4px;background:var(--accent)}}
.tlabel{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:4px;font-size:.8rem}}
.fr{{font-variant-numeric:tabular-nums;color:var(--muted);min-width:4.5em}}
.jump{{font:inherit;font-size:.78rem;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:3px;padding:1px 6px;cursor:pointer}}
.jump.root{{border-color:var(--warn)}} .jump.match{{border-color:var(--accent)}}
.grid3{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}}
@media (max-width:520px){{.grid2{{grid-template-columns:1fr}}}}
.missing{{background:var(--card);border:1px dashed var(--line);padding:24px;text-align:center;color:var(--muted);font-size:.85rem}}
</style></head><body><main>
<h1>DMP corrective-data PoC: failure → root → search → correction</h1>
<p class="meta">2026-10-14 · session impl21 · worktree VLA-exp-dmp-poc (branch exp/dmp-poc) · method: <a href="DMP_POC_METHOD.html">DMP_POC_METHOD</a></p>
<div class="status"><b>UNVERIFIED.</b> Every number here comes from new code that val has not checked. Videos are replays of recorded
actions (pixels are never stored); each replay is checked against the recorded path.</div>
<p>For each scenario: a GR00T N1.7 failure from R-047, its root (where it went wrong), the 3 nearest human demos found by the
search, and the corrections: the matched demo's remaining motion bent from the root to the goal (DMP, or the I-Gen-style
linear bridge as the baseline), executed in LIBERO through the normal controller. A correction is <b>kept</b> only if the task
succeeds and it passes the jerk, speed and collateral checks.</p>
<h2>Summary</h2>
<div class="tw"><table><tr><th>Scenario</th>{''.join(f'<th>{esc(ARM_LABEL[a])}</th>' for a in arms)}<th>Coverage: nearest demo state, d/d_gap</th></tr>{summ}</table></div>
<p class="meta">Kept / runs (3 matched demos per arm). Deviations from the method page found while building (10-14):
matches must leave ≥ 1 s of their phase; the carry goal is in the held-object frame; a clearance lift precedes the approach
when the root is &gt; 3 cm below the matched demo point.</p>
{body}
</main>
<script>
function seek(id,t){{const v=document.getElementById(id);if(!v)return;v.currentTime=t;v.pause();}}
document.querySelectorAll('video').forEach(v=>{{const l=document.getElementById(v.id+'_t');const f=+v.dataset.first||0;
v.addEventListener('timeupdate',()=>{{if(l)l.textContent='t = '+(f+Math.round(v.currentTime*{FPS}));}});}});
</script></body></html>"""
    with open(out, "w") as f:
        f.write(page)
    return out


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(main_repo(), "docs/DMP_POC_SHOWCASE.html")
    print("wrote", build(out), os.path.getsize(out) // 1024, "KB")
