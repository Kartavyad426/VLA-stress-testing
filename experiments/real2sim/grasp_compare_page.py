"""HTML page for grasp_compare.py output: runs/real2sim/policy/grasp_compare/index.html (local; video next to it)."""
import json, os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "runs/real2sim/policy/grasp_compare"
data = json.load(open(os.path.join(D, "grasp_compare.json")))
for s in data["series"]:
    if s["name"] == "g1":  # stitched from separate 3-step predictions: the jumps between queries are not motion
        s["speed_cms"] = [None] * len(s["speed_cms"])
S = {s["name"]: s for s in data["series"]}


def mean_in(s, key, lo, hi):
    t, v = np.array(s["t"]), np.array([np.nan if x is None else x for x in s[key]], float)
    w = (t > lo) & (t <= hi) & np.isfinite(v)
    return float(np.mean(v[w])) if w.any() else float("nan")


rows = []
for s in data["series"]:
    rows.append(dict(name=s["label"], event=s.get("event", ""),
                     h_event=s["summary"]["height_at_event_cm"], h_min=s["summary"]["min_height_cm"],
                     h_cmd_min=(min(np.array(s["commanded_height_cm"])[(np.array(s["t"]) >= -4) & (np.array(s["t"]) <= 2)]) if "commanded_height_cm" in s else None),
                     tilt_event=s["summary"]["tilt_at_event_deg"], tilt_1s=mean_in(s, "tilt_deg", -1, 0),
                     speed_1s=mean_in(s, "speed_cms", -1, 0), speed_3s=mean_in(s, "speed_cms", -3, -1)))


def f(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


table_rows = "\n".join(
    f"<tr><td>{r['name']}</td><td>{r['event']}</td><td>{f(r['h_event'])}</td><td>{f(r['h_min'])}</td><td>{f(r['h_cmd_min'])}</td>"
    f"<td>{f(r['tilt_event'])}</td><td>{f(r['tilt_1s'])}</td><td>{f(r['speed_3s'])}</td><td>{f(r['speed_1s'])}</td></tr>" for r in rows)

R = {r["name"]: r for r in rows}
sg, sp, ss, rl, g1, ex = (R[S[k]["label"]] for k in ("sim_grasp", "sim_push", "sim_success", "real", "g1", "expert"))

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grasp Approach Comparison</title>
<style>
:root{{--bg:#fcfcfb;--fg:#1d1d1b;--muted:#5f5f5a;--line:#dddcd5;--card:#f2f1ec;--grid:#e6e5df;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#34332f;--card:#232321;--grid:#2c2c29;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#34332f;--card:#232321;--grid:#2c2c29;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:1180px;margin:0 auto}} h1{{font-size:1.5rem;margin:.2em 0}} h2{{font-size:1.15rem;margin-top:1.8em;border-bottom:1px solid var(--line);padding-bottom:.2em}}
.meta{{color:var(--muted);font-size:.9rem}} .note{{background:var(--card);border-left:4px solid var(--s2);padding:10px 14px;margin:12px 0}}
video{{width:100%;background:#000;border-radius:6px}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:.88rem}} th,td{{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}} th{{background:var(--card)}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 18px;margin:8px 0 4px;font-size:.88rem}} .legend span{{display:inline-flex;align-items:center;gap:6px}}
.legend i{{display:inline-block;width:18px;height:3px;border-radius:2px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px}}
.chart{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 10px;position:relative}}
.chart h3{{font-size:.95rem;margin:.1em 0 .2em}} svg{{width:100%;height:auto;display:block}}
.tip{{position:absolute;pointer-events:none;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-size:.8rem;display:none;white-space:nowrap;z-index:2}}
</style></head><body><main>
<h1>Grasp approach: real operator vs GR00T (real frames, sim) vs scripted expert</h1>
<p class="meta">res31, 10-19, internal. Scene <code>{data['scene']}</code> (rebuilt with stereo scale + anchored camera). All curves aligned on the gripper close (t = 0); the pushing episode is aligned on its first sustained push into the table. Heights are of the fingertip point (between the finger pads, near their tips) above the rebuilt table top (checked against stereo to 0.4 cm). Exploratory, numbers UNVERIFIED.</p>

<div class="note"><b>What differs, in plain words</b>
<ul>
<li><b>GR00T in sim goes ~2–4 cm lower than everyone else.</b> At the close its fingertip point is at {f(sg['h_event'])} / {f(ss['h_event'])} cm (grasp / success episodes) and {f(sp['h_event'])} cm in the pushing one: at or just below the table top, so the finger ends press into the table. Its <i>commanded</i> height goes down to {f(sg['h_cmd_min'])} / {f(sp['h_cmd_min'])} cm. The real operator closed at {f(rl['h_event'])} cm, the expert at {f(ex['h_event'])} cm, and GR00T's own predictions on the real frames never go below {f(g1['h_min'])} cm.</li>
<li><b>GR00T in sim approaches tilted.</b> Gripper tilt from straight down at the close: {f(sg['tilt_event'])}° / {f(ss['tilt_event'])}° / {f(sp['tilt_event'])}° in sim vs {f(rl['tilt_event'])}° for the real operator, {f(g1['tilt_event'])}° for GR00T on real frames and {f(ex['tilt_event'])}° for the expert. A tilted gripper puts one finger lower than the other, which fits the finger-into-table pushing.</li>
<li><b>Speed is not the difference.</b> In the last second before the close the real operator moves at {f(rl['speed_1s'])} cm/s; GR00T in sim at {f(sg['speed_1s'])} (grasp) and {f(sp['speed_1s'])}–{f(ss['speed_1s'])} cm/s (push/success: it has already stopped against the table); the expert at {f(ex['speed_1s'])} cm/s. (No speed for GR00T-on-real-frames: its curve is stitched from separate predictions.)</li>
<li><b>So:</b> on real frames GR00T plans the approach like the operator (upright, stops ~2–5 cm above the table); in our sim the same policy comes in tilted and too low. Same weights and same action decoding; what differs is its input: the images (sim-rendered robot and objects, projected textures) and the state (FK in sim vs recorded on real frames; the two agree on recorded poses, checked 10-15). So the suspect is the images, or a state difference we have not caught. This is the sim-side suspect to test next (image swap at the approach, checklist #20).</li>
</ul></div>

<h2>Video (half speed, t = −4 s … +2 s)</h2>
<p class="meta">Top left: the real operator (orange line = GR00T's predicted fingertip path from the latest real-frame query, blue dot = the operator's fingertip). Top right: GR00T in sim, an episode that grasps (orange v0). Bottom left: GR00T in sim pushing into the table (orange v3; it heads for the yellow block in this variant). Bottom right: the scripted expert (orange v0). Each tile: exterior | wrist, as the policy sees them.</p>
<video controls preload="metadata" src="grasp_compare.mp4"></video>

<h2>Curves</h2>
<div class="legend" id="legend"></div>
<div class="grid">
<div class="chart" data-key="height_cm" data-title="Fingertip height above the table (cm)"></div>
<div class="chart" data-key="tilt_deg" data-title="Gripper tilt from straight down (°)"></div>
<div class="chart" data-key="speed_cms" data-title="Fingertip speed (cm/s)"></div>
<div class="chart" data-key="grip" data-title="Gripper command / position (0 open … 1 closed)"></div>
</div>
<p class="meta">Dashed = GR00T's <i>commanded</i> fingertip height in sim (achieved is solid). GR00T-on-real-frames is a prediction (the next 3 steps of each query, queries every 3 frames), not executed motion.</p>

<h2>Numbers</h2>
<div class="wrap"><table>
<tr><th>Series</th><th>Aligned on</th><th>Height at t=0 (cm)</th><th>Min height −4…+2 s (cm)</th><th>Min commanded (cm)</th><th>Tilt at t=0 (°)</th><th>Tilt, last 1 s (°)</th><th>Speed −3…−1 s (cm/s)</th><th>Speed, last 1 s (cm/s)</th></tr>
{table_rows}
</table></div>
<p class="meta">Source: <code>grasp_compare.json</code> (experiments/real2sim/grasp_compare.py). Real operator: recorded joints → FK. GR00T real frames: <code>runs/real2sim/policy/realdata_orange_full</code>. GR00T sim: <code>runs/real2sim/policy/batch_orange_stereo</code>. Expert: <code>runs/real2sim/tasks_stereo</code>.</p>
</main>
<div class="tip" id="tip"></div>
<script>
const DATA = {json.dumps(data["series"])};
const COL = {{real:"--s1", g1:"--s2", sim_grasp:"--s3", sim_push:"--s4", expert:"--s5", sim_success:"--s6"}};
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const leg = document.getElementById("legend");
DATA.forEach(s => {{ leg.insertAdjacentHTML("beforeend", `<span><i style="background:var(${{COL[s.name]}})"></i>${{s.label}}</span>`); }});
const tip = document.getElementById("tip");
function draw(el) {{
  const key = el.dataset.key, W = 560, H = 260, m = {{l: 44, r: 12, t: 8, b: 30}};
  const x0 = -4, x1 = 2;
  let ys = [];
  DATA.forEach(s => s.t.forEach((t, i) => {{ if (t >= x0 && t <= x1 && s[key][i] !== null && isFinite(s[key][i])) ys.push(s[key][i]); }}));
  if (key === "height_cm") DATA.forEach(s => (s.commanded_height_cm || []).forEach((v, i) => {{ if (s.t[i] >= x0 && s.t[i] <= x1) ys.push(v); }}));
  let y0 = Math.min(...ys), y1 = Math.max(...ys); if (key === "grip") {{ y0 = 0; y1 = 1; }}
  if (key === "speed_cms") {{ y0 = 0; y1 = Math.min(y1, 40); }}
  const pad = (y1 - y0) * 0.06 || 1; y0 -= key === "grip" || key === "speed_cms" ? 0 : pad; y1 += pad;
  const X = t => m.l + (t - x0) / (x1 - x0) * (W - m.l - m.r), Y = v => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);
  let g = "";
  const nt = 5; for (let k = 0; k <= nt; k++) {{ const v = y0 + (y1 - y0) * k / nt; g += `<line x1="${{m.l}}" x2="${{W - m.r}}" y1="${{Y(v)}}" y2="${{Y(v)}}" stroke="var(--grid)"/><text x="${{m.l - 6}}" y="${{Y(v) + 4}}" font-size="11" text-anchor="end" fill="var(--muted)">${{v.toFixed(key === "grip" ? 1 : 0)}}</text>`; }}
  for (let t = x0; t <= x1; t++) g += `<text x="${{X(t)}}" y="${{H - 10}}" font-size="11" text-anchor="middle" fill="var(--muted)">${{t > 0 ? "+" : ""}}${{t}} s</text>`;
  g += `<line x1="${{X(0)}}" x2="${{X(0)}}" y1="${{m.t}}" y2="${{H - m.b}}" stroke="var(--muted)" stroke-dasharray="3 3"/>`;
  if (key === "height_cm" && y0 < 0) g += `<line x1="${{m.l}}" x2="${{W - m.r}}" y1="${{Y(0)}}" y2="${{Y(0)}}" stroke="var(--muted)"/><text x="${{W - m.r - 4}}" y="${{Y(0) - 4}}" font-size="11" text-anchor="end" fill="var(--muted)">table top</text>`;
  const path = (t, v) => {{ let d = "", pen = false; t.forEach((tt, i) => {{ if (tt < x0 || tt > x1 || v[i] === null || !isFinite(v[i])) {{ pen = false; return; }} const yy = Math.max(m.t, Math.min(H - m.b, Y(v[i]))); d += (pen ? "L" : "M") + X(tt).toFixed(1) + " " + yy.toFixed(1); pen = true; }}); return d; }};
  DATA.forEach(s => {{
    g += `<path d="${{path(s.t, s[key])}}" fill="none" stroke="var(${{COL[s.name]}})" stroke-width="2" stroke-linejoin="round"/>`;
    if (key === "height_cm" && s.commanded_height_cm) g += `<path d="${{path(s.t, s.commanded_height_cm)}}" fill="none" stroke="var(${{COL[s.name]}})" stroke-width="2" stroke-dasharray="5 4" opacity=".8"/>`;
  }});
  g += `<line id="cx" x1="0" x2="0" y1="${{m.t}}" y2="${{H - m.b}}" stroke="var(--fg)" opacity="0"/><rect x="${{m.l}}" y="${{m.t}}" width="${{W - m.l - m.r}}" height="${{H - m.t - m.b}}" fill="transparent"/>`;
  el.innerHTML = `<h3>${{el.dataset.title}}</h3><svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="${{el.dataset.title}}">${{g}}</svg>`;
  const svg = el.querySelector("svg"), cx = svg.querySelector("#cx");
  svg.addEventListener("mousemove", e => {{
    const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) / r.width * W;
    const t = x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0); if (t < x0 || t > x1) {{ tip.style.display = "none"; cx.setAttribute("opacity", 0); return; }}
    cx.setAttribute("x1", px); cx.setAttribute("x2", px); cx.setAttribute("opacity", .35);
    let h = `<b>t = ${{t.toFixed(2)}} s</b>`;
    DATA.forEach(s => {{ let bi = -1, bd = 1e9; s.t.forEach((tt, i) => {{ const d = Math.abs(tt - t); if (d < bd) {{ bd = d; bi = i; }} }});
      if (bi >= 0 && bd < 0.15 && s[key][bi] !== null && isFinite(s[key][bi])) h += `<br><span style="color:var(${{COL[s.name]}})">■</span> ${{s.label}}: ${{s[key][bi].toFixed(key === "grip" ? 2 : 1)}}` + (key === "height_cm" && s.commanded_height_cm ? ` (cmd ${{s.commanded_height_cm[bi].toFixed(1)}})` : ""); }});
    tip.innerHTML = h; tip.style.display = "block";
    tip.style.left = Math.min(window.innerWidth - 330, e.pageX + 14) + "px"; tip.style.top = (e.pageY + 14) + "px";
  }});
  svg.addEventListener("mouseleave", () => {{ tip.style.display = "none"; cx.setAttribute("opacity", 0); }});
}}
document.querySelectorAll(".chart").forEach(draw);
</script></body></html>"""
open(os.path.join(D, "index.html"), "w").write(html)
print("wrote", os.path.join(D, "index.html"))
for r in rows:
    print(r)
