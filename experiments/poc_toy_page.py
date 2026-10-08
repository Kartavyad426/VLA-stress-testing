"""PoC toy: build the stakeholder walkthrough docs/MINING_SPEC_TOY.html. CPU.

Reads only runs/poc_toy/ (instance.json, traces.json, search.json, render/,
demo_frames/) so every number on the page is traceable to a file. Frames are
embedded as JPEG data URIs; the page has no other dependency than an optional
Google Fonts link.

    .venvs/libero-plus/bin/python experiments/poc_toy_page.py
"""
import base64, html, io, json, math, os

import imageio.v2 as imageio
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "runs/poc_toy")
OUT = os.path.join(ROOT, "docs/MINING_SPEC_TOY.html")
HZ = 20

inst = json.load(open(os.path.join(P, "instance.json")))["instance"]
tr = json.load(open(os.path.join(P, "traces.json")))
srch = json.load(open(os.path.join(P, "search.json")))
fid = json.load(open(os.path.join(P, "render/fidelity.json")))
R047 = [json.loads(l) for l in open(os.path.join(ROOT, "runs/r047/rollouts.jsonl"))]


def img(path, w=None):
    im = Image.open(path).convert("RGB")
    if w and im.width != w:
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def frame(row, cam, t):
    return img(os.path.join(P, f"render/row{row}_{cam}_t{t:03d}.png"), 300)


def sec(t):
    return f"{t / HZ:.2f}".rstrip("0").rstrip(".") + " s"


def strip(row, items, cls=""):
    """items: (t, caption). Agent view above wrist view per column."""
    cols = []
    for t, cap in items:
        cols.append(f'<figure class="fr {cls}"><img src="{frame(row, "agent", t)}" alt="scene view at {sec(t)}">'
                    f'<img src="{frame(row, "wrist", t)}" alt="wrist camera at {sec(t)}">'
                    f'<figcaption><b>{sec(t)}</b> {cap}</figcaption></figure>')
    return '<div class="strip">' + "".join(cols) + "</div>"


E = html.escape
root = inst["root"]; sit = inst["situation"]; term = inst["terminal"]
off = [round(x * 100, 1) for x in sit["derived"]["eef_to_target_offset_m"]]
t_root = root["t"]
ok_ev = next(e for e in tr["20"]["events"] if e["success"])
ok_off = [round(x * 100, 1) for x in ok_ev["offset_m"]]
s0 = R047[14]["steps"][0]["obs_state"]["eef_pos"]; n0 = R047[0]["steps"][0]["obs_state"]["eef_pos"]
start_shift_cm = round(math.dist(s0, n0) * 100, 1)
div = inst["divergence_from_same_start_success"]
drop = term["lifted_then_dropped"][0]
fun = srch["funnel"]; hd, our = fun["human demo"], fun["our rollout"]
hit = srch["hits"][0]
hit_row = int(hit["ref"].split()[-1])
hit_pert = R047[hit_row]["perturbation"]
hit_off = [round(x * 100, 1) for x in hit["offset_m"]]
demo_off = srch["human_demo_successful_onset_offsets_cm"]
sweep = srch["offset_tolerance_sweep_cm"]
tol = srch["tolerances"]
max_fid = max(v["max_dev_m"] for v in fid.values())
# R-047 cell for this start: scene 984 at 0.2 rad, 6 runs
MAN = [json.loads(l) for l in open(os.path.join(ROOT, "runs/r047/joint_radius_rad/manifest.jsonl"))]
cell = [r for r in MAN if r["task_id"] == 984 and r["magnitude"] == 0.2]
cell_ok = sum(r["success"] for r in cell)
pooled = {m: sum(r["success"] for r in MAN if r["magnitude"] == m) for m in (0.0, 0.2, 0.5)}
assert all(sum(1 for r in MAN if r["magnitude"] == m) == 60 for m in pooled)

# ---------- chart 1: bowl height over time --------------------------------
def lift_chart():
    W, H, l, r, t, b = 760, 260, 30, 16, 14, 34
    xmax, ymax = 14.0, 12.0
    X = lambda s: l + (W - l - r) * s / xmax
    Y = lambda c: t + (H - t - b) * (1 - c / ymax)
    g = []
    for c in range(0, 13, 3):
        g.append(f'<line x1="{l}" x2="{W-r}" y1="{Y(c):.1f}" y2="{Y(c):.1f}" class="grid"/>'
                 f'<text x="{l-6}" y="{Y(c)+4:.1f}" class="ax" text-anchor="end">{c}</text>')
    for s in range(0, 15, 2):
        g.append(f'<text x="{X(s):.1f}" y="{H-b+16}" class="ax" text-anchor="middle">{s} s</text>')
    g.append(f'<line x1="{l}" x2="{W-r}" y1="{Y(5):.1f}" y2="{Y(5):.1f}" class="thr"/>'
             f'<text x="{X(5):.1f}" y="{Y(5)-5:.1f}" class="ax">success line: lifted 5 cm</text>')

    def path(key, cls, label):
        L = tr[key]["lift_cm"]
        pts = " ".join(f"{X(i / HZ):.1f},{Y(max(0, v)):.1f}" for i, v in enumerate(L))
        hov = "".join(f'<rect x="{X(i / HZ)-3:.1f}" y="{t}" width="6" height="{H-t-b}" class="hit">'
                      f'<title>{label} · {i / HZ:.2f} s · bowl {max(0, L[i]):.1f} cm up</title></rect>'
                      for i in range(0, len(L), 4))
        return f'<polyline points="{pts}" class="ln {cls}"/>' + hov

    g.append(path("20", "ok", "Same start, run 2 (succeeded)"))
    g.append(path("14", "fail", "Same start, run 1 (failed)"))
    # annotations
    g.append(f'<circle cx="{X(t_root / HZ):.1f}" cy="{Y(0):.1f}" r="5" class="pt fail"/>'
             f'<text x="{X(0)+2:.1f}" y="{Y(8):.1f}" class="lab">① grip closes, {sec(t_root)}</text>'
             f'<line x1="{X(t_root / HZ):.1f}" x2="{X(t_root / HZ):.1f}" y1="{Y(7.4):.1f}" y2="{Y(0)-6:.1f}" class="lead"/>')
    pk = max(range(len(tr["14"]["lift_cm"][:80])), key=tr["14"]["lift_cm"][:80].__getitem__)
    g.append(f'<line x1="{X(pk / HZ)+3:.1f}" y1="{Y(tr["14"]["lift_cm"][pk])-3:.1f}" x2="{X(4.1):.1f}" y2="{Y(3.6):.1f}" class="lead"/>'
             f'<text x="{X(4.15):.1f}" y="{Y(3.6)+4:.1f}" class="lab">② bowl rises 2 cm, slips out</text>')
    pk2 = max(range(len(tr["14"]["lift_cm"])), key=tr["14"]["lift_cm"].__getitem__)
    g.append(f'<text x="{X(pk2 / HZ)-6:.1f}" y="{Y(tr["14"]["lift_cm"][pk2])-8:.1f}" class="lab" text-anchor="end">③ 6th try lifts it, then drops it</text>')
    g.append(f'<text x="{X(len(tr["20"]["lift_cm"]) / HZ)+6:.1f}" y="{Y(tr["20"]["lift_cm"][-1])+14:.1f}" class="lab">run 2 done, {sec(len(tr["20"]["lift_cm"]) - 1)}</text>')
    g.append(f'<text x="{X(14)-2:.1f}" y="{Y(3.6):.1f}" class="lab" text-anchor="end">run 1 times out, 14 s</text>')
    leg = ('<div class="legend"><span><i class="sw fail"></i>Run 1 · failed</span>'
           '<span><i class="sw ok"></i>Run 2 · same start, succeeded</span>'
           '<span>Height of the bowl above the table, cm</span></div>')
    return leg + f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Bowl height over time for the two runs">{"".join(g)}</svg>'


# ---------- chart 2: where grasps start ------------------------------------
def scatter():
    W, H, l, r, t, b = 560, 380, 48, 16, 14, 40
    x0, x1, y0, y1 = -9.0, 8.0, -9.0, 2.0
    X = lambda v: l + (W - l - r) * (v - x0) / (x1 - x0)
    Y = lambda v: t + (H - t - b) * (1 - (v - y0) / (y1 - y0))
    g = []
    for v in range(-8, 9, 2):
        g.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{t}" y2="{H-b}" class="grid"/>'
                 f'<text x="{X(v):.1f}" y="{H-b+16}" class="ax" text-anchor="middle">{v}</text>')
    for v in range(-8, 3, 2):
        g.append(f'<line x1="{l}" x2="{W-r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                 f'<text x="{l-6}" y="{Y(v)+4:.1f}" class="ax" text-anchor="end">{v}</text>')
    g.append(f'<text x="{(l+W-r)/2:.0f}" y="{H-6}" class="ax" text-anchor="middle">← towards the robot · distance from bowl centre, front–back (cm) · away →</text>')
    g.append(f'<text x="12" y="{(t+H-b)/2:.0f}" class="ax" transform="rotate(-90 12 {(t+H-b)/2:.0f})" text-anchor="middle">side to side (cm)</text>')
    tc = tol["offset_per_axis_m"] * 100
    g.append(f'<rect x="{X(off[0]-tc):.1f}" y="{Y(off[1]+tc):.1f}" width="{X(off[0]+tc)-X(off[0]-tc):.1f}" '
             f'height="{Y(off[1]-tc)-Y(off[1]+tc):.1f}" class="box"/>')
    for d in demo_off["all"]:
        if x0 <= d[0] <= x1 and y0 <= d[1] <= y1:
            g.append(f'<circle cx="{X(d[0]):.1f}" cy="{Y(d[1]):.1f}" r="4.5" class="pt demo">'
                     f'<title>Human demo grasp · {d[0]} cm, {d[1]} cm</title></circle>')
    for d in srch["our_successful_onset_offsets_cm"]:
        if x0 <= d[0] <= x1 and y0 <= d[1] <= y1:
            g.append(f'<rect x="{X(d[0])-4:.1f}" y="{Y(d[1])-4:.1f}" width="8" height="8" rx="1.5" class="pt ours">'
                     f'<title>Our successful run ({E(d[3])}) · {d[0]} cm, {d[1]} cm</title></rect>')
    g.append(f'<circle cx="{X(hit_off[0]):.1f}" cy="{Y(hit_off[1]):.1f}" r="9" class="ring"/>'
             f'<text x="{X(hit_off[0])-8:.1f}" y="{Y(hit_off[1])+26:.1f}" class="lab" text-anchor="end">the one match</text>')
    fx, fy = X(off[0]), Y(off[1])
    g.append(f'<path d="M{fx-6:.1f},{fy-6:.1f}L{fx+6:.1f},{fy+6:.1f}M{fx-6:.1f},{fy+6:.1f}L{fx+6:.1f},{fy-6:.1f}" class="x fail">'
             f'<title>The failure · {off[0]} cm, {off[1]} cm</title></path>'
             f'<text x="{fx-6:.1f}" y="{fy-14:.1f}" class="lab" text-anchor="middle">the failure</text>')
    leg = ('<div class="legend"><span><i class="mk x"></i>The failure</span>'
           '<span><i class="mk sq"></i>Our successful runs, this scene</span>'
           '<span><i class="mk ci"></i>Human demos, this task</span>'
           '<span><i class="mk bx"></i>Search window</span></div>')
    return leg + f'<svg viewBox="0 0 {W} {H}" class="chart sc" role="img" aria-label="Where each successful grasp started, relative to the bowl">{"".join(g)}</svg>'


def demo_pair(name, t):
    a = img(os.path.join(P, f"demo_frames/{name}_t{t:03d}_agent.png"), 300)
    w = img(os.path.join(P, f"demo_frames/{name}_t{t:03d}_wrist.png"), 300)
    return a, w


near_demo = next(e for e in srch["nearest_by_offset"] if e["source"] == "human demo"
                 and os.path.exists(os.path.join(P, f"demo_frames/{e['ref']}_t{e['t']:03d}_agent.png")))
nd_a, nd_w = demo_pair(near_demo["ref"], near_demo["t"])

CSS = """
:root{color-scheme:light dark;
 --bg:#F3F5F8;--surface:#FFFFFF;--surface-2:#EDF0F4;--ink:#14181D;--muted:#5B636E;--rule:#D7DDE5;
 --accent:#2F4A9E;--ok:#1B6B4F;--warn:#8A6A12;--bad:#A8341B;
 --c-fail:#B83A1A;--c-ok:#0F7A55;--c-ours:#3552B8;--c-demo:#8A929C;
 --sans:'IBM Plex Sans',system-ui,-apple-system,Segoe UI,sans-serif;--mono:'IBM Plex Mono',ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --bg:#0E1116;--surface:#151A21;--surface-2:#1B212A;--ink:#E5EAF1;--muted:#9AA3AF;--rule:#262D37;
 --accent:#93A9F0;--ok:#5DC79B;--warn:#D9B85C;--bad:#E5905E;--c-fail:#D4683A;--c-ok:#2EA173;--c-ours:#6A86E0;--c-demo:#6F7782}}
:root[data-theme="dark"]{--bg:#0E1116;--surface:#151A21;--surface-2:#1B212A;--ink:#E5EAF1;--muted:#9AA3AF;--rule:#262D37;
 --accent:#93A9F0;--ok:#5DC79B;--warn:#D9B85C;--bad:#E5905E;--c-fail:#D4683A;--c-ok:#2EA173;--c-ours:#6A86E0;--c-demo:#6F7782}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 var(--sans)}
main{max-width:980px;margin:0 auto;padding:36px 16px 72px}
h1{font-size:30px;line-height:1.2;margin:0 0 6px}
h2{font-size:20px;margin:44px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--rule)}
h2 .n{font:500 13px var(--mono);color:var(--accent);margin-right:8px}
.sub{color:var(--muted);margin:0 0 22px}
p{max-width:72ch}
.lede{font-size:18px}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:16px 18px;margin:14px 0}
.key{border-left:4px solid var(--accent)}
.mock{border:2px dashed var(--warn);background:var(--surface)}
.badge{display:inline-block;font:600 11px var(--mono);letter-spacing:.04em;padding:2px 8px;border-radius:10px;vertical-align:2px}
.badge.m{background:var(--ok);color:var(--surface)} .badge.i{background:var(--warn);color:var(--surface)}
.badge.mu{background:var(--warn);color:var(--surface)}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin:12px 0}
.fr{margin:0;background:var(--surface);border:1px solid var(--rule);border-radius:6px;overflow:hidden}
.fr img{display:block;width:100%;height:auto}
.fr img+img{border-top:1px solid var(--rule)}
.fr figcaption{font-size:13px;line-height:1.4;padding:6px 8px;color:var(--muted)}
.fr figcaption b{color:var(--ink);display:block}
.fr.hl{outline:3px solid var(--c-fail);outline-offset:-3px}
.fr.hlok{outline:3px solid var(--c-ok);outline-offset:-3px}
.row2{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
.entry{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin:12px 0}
.field{background:var(--surface);border:1px solid var(--rule);border-top:3px solid var(--accent);border-radius:6px;padding:12px 14px}
.field h4{margin:0 0 2px;font-size:15px} .field .q{color:var(--muted);font-size:13px;margin:0 0 8px}
.field .v{font-size:15px}
.chart{width:100%;height:auto;display:block;background:var(--surface);border:1px solid var(--rule);border-radius:6px}
.chart.sc{max-width:620px}
.chart .grid{stroke:var(--rule);stroke-width:1}
.chart .ax{fill:var(--muted);font:12px var(--sans)}
.chart .lab{fill:var(--ink);font:500 12.5px var(--sans)}
.chart .thr{stroke:var(--muted);stroke-dasharray:5 4;stroke-width:1.5}
.chart .lead{stroke:var(--muted);stroke-width:1}
.chart .ln{fill:none;stroke-width:2;stroke-linejoin:round}
.chart .ln.fail{stroke:var(--c-fail)} .chart .ln.ok{stroke:var(--c-ok)}
.chart .pt.fail{fill:var(--c-fail);stroke:var(--surface);stroke-width:2}
.chart .pt.demo{fill:none;stroke:var(--c-demo);stroke-width:1.6}
.chart .pt.ours{fill:var(--c-ours);stroke:var(--surface);stroke-width:1.5}
.chart .x{stroke-width:3;stroke-linecap:round} .chart .x.fail{stroke:var(--c-fail)}
.chart .ring{fill:none;stroke:var(--c-ok);stroke-width:2.5}
.chart .box{fill:var(--c-fail);fill-opacity:.07;stroke:var(--c-fail);stroke-dasharray:4 3;stroke-width:1.2}
.chart .hit{fill:transparent}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--muted);margin:8px 0}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{display:inline-block;width:18px;height:3px;border-radius:2px} .sw.fail{background:var(--c-fail)} .sw.ok{background:var(--c-ok)}
.mk{display:inline-block;width:11px;height:11px}
.mk.x{background:linear-gradient(45deg,transparent 40%,var(--c-fail) 40% 60%,transparent 60%),linear-gradient(-45deg,transparent 40%,var(--c-fail) 40% 60%,transparent 60%)}
.mk.sq{background:var(--c-ours);border-radius:2px} .mk.ci{border:1.6px solid var(--c-demo);border-radius:50%}
.mk.bx{border:1.2px dashed var(--c-fail);background:color-mix(in srgb,var(--c-fail) 8%,transparent)}
.tw{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--surface);margin:12px 0}
table{border-collapse:collapse;width:100%;min-width:560px}
th,td{text-align:left;vertical-align:top;padding:8px 12px;border-bottom:1px solid var(--rule);font-size:14px}
th{background:var(--surface-2);font-weight:600} tr:last-child td{border-bottom:0}
th.n,td.n{font-variant-numeric:tabular-nums;text-align:right}
.big{font-size:22px;font-weight:600}
.flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:8px;margin:14px 0}
.flow div{flex:1 1 150px;background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:10px 12px;font-size:14px}
.flow div b{display:block;font-size:15px}
.flow .on{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}
.muted{color:var(--muted)} .small{font-size:13px}
details{margin:10px 0} summary{cursor:pointer;color:var(--accent)}
pre{font:12.5px/1.5 var(--mono);background:var(--surface-2);border:1px solid var(--rule);border-radius:6px;padding:12px;overflow-x:auto}
footer{color:var(--muted);font-size:13px;margin-top:48px;border-top:1px solid var(--rule);padding-top:12px}
"""

entry_json = json.dumps({k: inst[k] for k in ("id", "scenario_ids", "root", "situation", "success", "terminal")},
                        indent=1)

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Failure Mining Walkthrough</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style></head><body><main>

<h1>From one robot failure to the data that fixes it</h1>
<p class="sub">A worked example of the failure-scenario mining spec, built from one real failure in our simulator.
Draft for stakeholder review · 2026-10-08.</p>

<div class="card key">
<p class="lede" style="margin-top:0">A robot policy fails at a task. Generic "more training data" rarely fixes it. What does fix it is
<b>examples of the exact moment where it went wrong, done right</b>. The mining spec is how we write that moment down,
so that we can search existing data for it, brief a person to demonstrate it, or recreate it in simulation.</p>
<p style="margin-bottom:0">This page follows <b>one</b> real failure through the first two steps: turning it into a spec entry, and searching
for successful examples of it.
Labels: <span class="badge m">MEASURED</span> means taken from our run files.
<span class="badge i">ILLUSTRATIVE</span> means a setting we chose for this example.
<span class="badge mu">MOCK-UP</span> means not built yet.</p>
</div>

<div class="flow">
<div class="on"><b>1 · The failure</b>what happened</div>
<div class="on"><b>2 · Find the root</b>the first thing that went wrong</div>
<div class="on"><b>3 · Write the entry</b>a readable, searchable record</div>
<div class="on"><b>4 · Search for successes</b>in human demos and our runs</div>
<div><b>5 · Fill the gap</b>produce what the search can't find <span class="badge mu">MOCK-UP</span></div>
</div>

<h2><span class="n">1</span>The failure</h2>
<p>The robot is a standard research arm, controlled by <b>GR00T N1.7</b>, NVIDIA's open robot model, in the version
NVIDIA tuned for this set of tabletop tasks. The instruction is: <i>"{E(inst['task']['instruction'])}"</i>.</p>
<p>Normally every episode starts with the arm in the same pose. Here we started it <b>slightly off that pose</b>: the hand began
{start_shift_cm} cm from its usual spot. <span class="badge m">MEASURED</span> Real robots rarely start in exactly the same place, so this is a
realistic stress. In our test of this scene from this offset, <b>{cell_ok} of {len(cell)}</b> runs succeeded.
Across all ten scenes, success fell from {pooled[0.0]}/60 at the normal start to {pooled[0.2]}/60 at this offset and
{pooled[0.5]}/60 at a larger one. <span class="badge m">MEASURED</span></p>
<p>Here is the failed run. The top row of each frame is the scene camera, the bottom row is the camera on the robot's wrist.
These are the two images the policy sees.</p>
{strip(14, [(0, "Starts off its usual pose"), (16, "Reaches towards the bowl"),
            (t_root, "Closes its fingers on the bowl's near rim"), (44, "Fingers shut, holding only the edge"),
            (50, "Bowl hangs tilted and starts to slip"), (54, "Bowl falls back to the table")], "")}
<p>It then keeps trying. On the sixth try it lifts the bowl {drop['max_lift_m']*100:.0f} cm, drops it again, and runs out of time
after {term['grasp_attempts']} attempts and {term['env_steps'] / HZ:.0f} s. <span class="badge m">MEASURED</span></p>
{strip(14, [(100, "Still re-trying"), (220, "Sixth try: bowl lifted"), (279, "Time runs out, bowl not on the plate")])}

<h2><span class="n">2</span>Find where it went wrong: the root</h2>
<p>A failure like this is a chain: miss, retry, retry, lift, drop, time out. Fixing the end of the chain (better retries) is not the point.
The <b>root</b> is the <b>first</b> thing that went wrong, and everything after it is a consequence.</p>
<p>We compare against a second run that started from <b>exactly the same pose</b> and succeeded. The only difference is the random
variation inside the policy. The two runs leave the start together. At {sec(div['t'])} their hands are {div['distance_m']*100:.0f} cm apart, and at
<b>{sec(t_root)}</b> the failing run closes its grip in the wrong place. <span class="badge m">MEASURED</span></p>
{lift_chart()}
<div class="row2">
<div>{strip(14, [(t_root, "Run 1: closes {:.1f} cm short of the bowl's centre".format(abs(off[0])))], "hl")}</div>
<div>{strip(20, [(31, "Run 2: closes {:.1f} cm short, on the bowl wall".format(abs(ok_off[0])))], "hlok")}</div>
</div>
<p>So the root is <b>the grasp</b>, and what went wrong is a <b>slip</b>: the fingers caught only the edge of the rim, the bowl rose
{root['evidence']['max_lift_m']*100:.0f} cm and fell out. Both runs started in the same place, so the start pose alone did not decide the outcome.
What decided it is <b>where the hand was when it closed</b>. That moment, not the whole episode, is what we need successful examples of.</p>

<h2><span class="n">3</span>Write it down: the spec entry</h2>
<p>The root becomes one entry with four parts. Each part answers one question, and each value is read from the simulator,
not typed by hand. <span class="badge m">MEASURED</span></p>
<div class="entry">
<div class="field"><h4>Root</h4><p class="q">Where did it first go wrong?</p>
<div class="v">The <b>grasp</b> · failure mode <b>slip</b><br><span class="muted small">at {sec(t_root)}, the first time the fingers close</span></div></div>
<div class="field"><h4>Situation</h4><p class="q">What did the robot see and where was it, at that moment?</p>
<div class="v">Hand <b>{abs(off[0]):.1f} cm short</b> of the bowl's centre, {abs(off[1]):.1f} cm to the side, {off[2]:.1f} cm above it; gripper
{sit['derived']['gripper_tilt_deg']:.0f}° from vertical; plate, ramekin and the other objects where they were<br>
<span class="muted small">plus both camera images at that moment</span></div></div>
<div class="field"><h4>Success</h4><p class="q">What does "got past it" mean?</p>
<div class="v">The bowl is <b>lifted at least 5 cm</b> within <b>2 seconds</b> of the fingers closing <span class="badge i">ILLUSTRATIVE</span></div></div>
<div class="field"><h4>Outcome</h4><p class="q">How did the episode end?</p>
<div class="v">Ran out of time after {term['grasp_attempts']} grasp attempts; bowl lifted once and dropped; ended {term['target_dist_to_plate_m']*100:.0f} cm from the plate</div></div>
</div>
<p>One more thing the entry records is the <b>cause</b>. Here that's the off-pose start, but it's kept as a <i>condition</i>, not as
part of what the failure <i>is</i>. A grasp slip is the same physical failure whether it was caused by an odd start, a moved camera
or a bumped table. Keeping causes separate means the same kind of failure can be counted and fixed across tasks, robots
and customers.</p>
<details><summary>The same entry as the machine reads it (excerpt)</summary><pre>{E(entry_json)}</pre>
<p class="small muted">Field names and layout are a draft. <span class="badge i">ILLUSTRATIVE</span></p></details>

<h2><span class="n">4</span>Search for successes</h2>
<p>We then run the entry as a query: <b>find moments where a robot or a person was in this same situation, and the grasp worked</b>.
We searched two pools for this task:</p>
<ul>
<li><b>Human demonstrations:</b> the {hd['episodes']} demonstrations people recorded for this exact task, which is the data this policy was tuned on.
They contain {hd['meets_success_criterion']} successful grasps.</li>
<li><b>Our own successful runs</b> of this scene: {our['episodes']} episodes from our robustness tests, with {our['meets_success_criterion']} successful grasps.</li>
</ul>
<p>A grasp counts as a match only if, at the moment the fingers start closing, the hand is within {tol['offset_per_axis_m']*100:.1f} cm
of the failure's position relative to the bowl in every direction, the gripper is angled within {tol['gripper_rot_deg']:.0f}° of it, and
the objects on the table are laid out within {tol['layout_xy_m']*100:.0f} cm of the same arrangement.
<span class="badge i">ILLUSTRATIVE</span> These tolerances are our choice for this example.</p>
{scatter()}
<p class="small muted">Each mark is one successful grasp, placed by where the hand was relative to the bowl when the fingers started closing.
The chart shows two of the three directions. The search also checks height, gripper angle and table layout. Hover a mark for its values.
<span class="badge m">MEASURED</span></p>

<div class="tw"><table>
<tr><th></th><th class="n">Human demos</th><th class="n">Our runs</th></tr>
<tr><td>Successful grasps in the pool</td><td class="n">{hd['meets_success_criterion']}</td><td class="n">{our['meets_success_criterion']}</td></tr>
<tr><td>… hand in the same place relative to the bowl</td><td class="n">{hd['cumulative']['offset']}</td><td class="n">{our['cumulative']['offset']}</td></tr>
<tr><td>… and gripper at the same angle</td><td class="n">{hd['cumulative']['offset+gripper']}</td><td class="n">{our['cumulative']['offset+gripper']}</td></tr>
<tr><td>… and the table laid out the same way</td><td class="n big">{hd['cumulative']['offset+gripper+layout']}</td><td class="n big">{our['cumulative']['offset+gripper+layout']}</td></tr>
</table></div>

<h3>What came back</h3>
<div class="card">
<p style="margin-top:0"><b>From the human demonstrations: nothing.</b> The people who recorded them always closed the gripper nearer the bowl's centre.
The furthest-back successful human grasp started {abs(demo_off['min'][0]):.1f} cm short of centre; the failure was {abs(off[0]):.1f} cm short.
Widening the window to ±4 cm still finds no human demo that also has the same table layout, because the demonstrations were recorded
with the objects placed slightly differently. <span class="badge m">MEASURED</span></p>
<p style="margin-bottom:0"><b>This empty result is the finding.</b> The data the policy learned from never shows it how to finish a grasp from
this position. When the hand ends up there, the policy has nothing to imitate.</p>
</div>
<div class="card">
<p style="margin-top:0"><b>From our own runs: one match.</b> In another test run the arm started from a <i>different</i> off-pose start
({hit_pert['joint_radius_rad']} instead of 0.2 in our units) but reached almost the same spot at the moment of grasping, and that time
the grasp held. That's exactly what the search is designed to find: the same situation, reached by a different path.
It's also a borderline match, within the window by a few millimetres on each axis, so a stricter window of ±1 cm returns nothing.
<span class="badge m">MEASURED</span></p>
{strip(hit_row, [(hit['t'], "Fingers close on the rim"), (44, "Bowl lifted, tilted"), (54, "Still held, on its way to the plate")], "hlok")}
<p class="small muted" style="margin-bottom:0">The search returns this segment, from the fingers closing to the lift, ready to use as a training example
for this exact situation.</p>
</div>
<details><summary>The nearest human demonstration, for comparison</summary>
<div class="row2" style="max-width:640px">
<figure class="fr"><img src="{nd_a}" alt="human demo scene view"><img src="{nd_w}" alt="human demo wrist view">
<figcaption><b>{E(near_demo['ref'].replace('_', ' '))}, fingers closing</b>The hand position differs from the failure's by {near_demo['match']['offset_dist_cm']} cm,
and the objects sit up to {near_demo['match']['worst_neighbour_dev_cm']} cm from their positions in our scene. Lower-resolution images, as recorded.</figcaption></figure>
</div></details>

<h2><span class="n">5</span>Filling the gap <span class="badge mu">MOCK-UP</span></h2>
<div class="card mock">
<p style="margin-top:0"><b>Not built yet.</b> This section shows what would happen next. Nothing below has been run.</p>
<p>The search found one example where we would want many. Because the failure happened in simulation, we can recreate the exact
moment it went wrong and produce more successes from it:</p>
<ul>
<li><b>Recreate the moment:</b> put the arm, bowl and objects back exactly as they were at {sec(t_root)}.</li>
<li><b>Finish it correctly:</b> let a scripted expert or a human operator complete the grasp from there, varying the start of the
moment slightly each time, so the examples cover the area around it rather than a single point.</li>
<li><b>Check each one</b> against the entry's own success test: the bowl is lifted 5 cm within 2 seconds.</li>
</ul>
<p style="margin-bottom:0">The same entry would also brief a person to set up the situation on a real robot, or tell a sim pipeline what to
generate. Those are the entry's two other uses. They are not shown here.</p>
</div>

<h2><span class="n">6</span>What a full spec gives you</h2>
<p>This page followed one entry. A spec covers a policy in one deployment setting, with an entry for each distinct failure found.
Each entry is readable by a person and searchable by a machine. For each one, it says:</p>
<ul>
<li>where the failure starts, and what the robot saw and where it was at that moment;</li>
<li>what success at that moment looks like, as a test any new example can be checked against;</li>
<li>whether existing data already covers it (search results attached) or it's a <b>gap</b> that needs new data, and what kind.</li>
</ul>
<p>Failure types use shared names (here, <i>grasp · slip</i>). That means counts can be compared across policies and datasets
without exposing one customer's details to another.</p>

<footer>
Sources (all in this repository): the failed run and the same-start success are runs/r047 rollouts rows 14 and 20 (R-047 in RESULTS.md),
and the match is row {hit_row}. Human demos: data/libero_raw (LIBERO-Spatial, 50 demos for this task).
Spec entry: runs/poc_toy/instance.json. Search: runs/poc_toy/search.json.
Frames re-rendered by replaying each run's recorded actions in the simulator; they follow the recorded hand path to within
{max_fid*1000:.1f} mm. One scene, one failure, one policy: this is an example, not a measurement of how often anything happens.
</footer>
</main></body></html>"""

open(OUT, "w").write(page)
print(f"wrote {OUT} ({len(page)//1024} KB)")
