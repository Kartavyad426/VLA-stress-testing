"""R-041 report: runs/<run>/<axis>/boundaries.jsonl + manifest.jsonl -> docs/R041_RESULTS.html.

  .venvs/groot/bin/python experiments/r041_report.py runs/r041 [--out docs/R041_RESULTS.html]

Per axis: one row per task with the action boundary bracket, the outcome
boundary bracket, and every sampled magnitude; one inline-SVG plot per axis
of ‖P−N‖ at forward 0 against magnitude (all tasks, one line each, failures
marked), with tau drawn; and the image-arm transfer at forward 0 against
magnitude for expectation 4.
"""
from __future__ import annotations

import argparse
import html
import json
import os

SCENE = {984: "between-plate-ramekin", 1030: "next-to-ramekin", 1062: "table-center", 1090: "on-cookie-box",
         1132: "top-drawer", 1169: "on-ramekin", 1201: "next-to-cookie-box", 1247: "on-stove",
         1282: "next-to-plate", 1327: "on-wooden-cabinet"}
R038_NULL_FAIL = {1247, 1327, 1132}      # 0/10 with an empty prompt (R-038)
AXIS_LABEL = {"camera_yaw_deg": "camera yaw (deg)", "camera_dist_m": "camera distance (m)",
              "light_intensity": "light intensity (×)", "joint_radius_rad": "start-pose radius (rad)"}

CSS = """
:root{color-scheme:light dark;--bg:#F3F5F8;--surface:#FFFFFF;--surface-2:#EDF0F4;--ink:#14181D;--muted:#5B636E;
 --rule:#D7DDE5;--rule-soft:#E6EAEF;--accent:#2F4A9E;--fail:#A8481B;--ok:#1B6B4F;--tau:#eda100;
 --sans:'IBM Plex Sans',system-ui,sans-serif;--mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0E1116;--surface:#151A21;--surface-2:#1B212A;
 --ink:#E5EAF1;--muted:#9AA3AF;--rule:#262D37;--rule-soft:#1F2630;--accent:#93A9F0;--fail:#E5905E;--ok:#5DC79B;--tau:#c98500;}}
:root[data-theme="dark"]{--bg:#0E1116;--surface:#151A21;--surface-2:#1B212A;--ink:#E5EAF1;--muted:#9AA3AF;--rule:#262D37;
 --rule-soft:#1F2630;--accent:#93A9F0;--fail:#E5905E;--ok:#5DC79B;--tau:#c98500;}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.55}
.wrap{max-width:1100px;margin:0 auto;padding:32px 16px 80px}
h1{font-size:1.9rem;margin:0 0 6px;letter-spacing:-.02em}h2{font-size:1.25rem;margin:2.2em 0 .6em}h3{font-size:1rem;margin:1.4em 0 .4em}
.meta{color:var(--muted);font-size:.85rem;margin-bottom:22px}
table{border-collapse:collapse;width:100%;font-size:.84rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 10px 6px 0;border-bottom:1px solid var(--rule-soft);vertical-align:top}
th{font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--rule)}
td.num{text-align:right;padding-right:14px}.ok{color:var(--ok);font-weight:600}.fail{color:var(--fail);font-weight:600}
figure{margin:14px 0;background:var(--surface);border:1px solid var(--rule-soft);border-radius:4px;padding:12px}
figure svg{display:block;width:100%;height:auto;color:var(--ink)}
figcaption{font-size:.8rem;color:var(--muted);margin-top:8px;line-height:1.4}
.note{background:var(--surface);border:1px solid var(--rule);border-left:3px solid var(--accent);border-radius:4px;padding:12px 14px;margin:14px 0;font-size:.9rem}
code{font-family:var(--mono);font-size:.85em;background:var(--surface-2);padding:.05em .3em;border-radius:3px}
.sw{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:6px;border-radius:2px}
"""


def load_axis(run, axis):
    d = os.path.join(run, axis)
    rows = [json.loads(l) for l in open(os.path.join(d, "manifest.jsonl"))] if os.path.exists(os.path.join(d, "manifest.jsonl")) else []
    bounds = [json.loads(l) for l in open(os.path.join(d, "boundaries.jsonl"))] if os.path.exists(os.path.join(d, "boundaries.jsonl")) else []
    return rows, bounds


def fmt_b(b):
    return "—" if b is None else f"{b[0]:.3g} – {b[1]:.3g}"


def plot(rows, axis, mx, tau, key, ylabel, ymax=None, tau_line=True):
    W, H, pl, pr, pt, pb = 900, 300, 50, 16, 12, 34
    by_task = {}
    for r in rows:
        v = r.get(key) if key != "tf_I" else (r["tf_f0"]["I"] if r.get("tf_f0") else None)
        if v is None:
            continue
        by_task.setdefault(r["task_id"], []).append((r["magnitude"], float(v), r["success"]))
    vals = [v for pts in by_task.values() for _, v, _ in pts]
    if not vals:
        return ""
    lo, hi = (min(vals + [0.0]), max(vals)) if ymax is None else (-0.3, ymax)
    hi = hi if hi > lo else lo + 1
    x = lambda m: pl + (W - pl - pr) * (m / mx)
    y = lambda v: pt + (H - pt - pb) * (1 - (v - lo) / (hi - lo))
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(ylabel)} against {html.escape(AXIS_LABEL.get(axis, axis))}, one line per task; failures marked">']
    for g in [lo + (hi - lo) * k / 4 for k in range(5)]:
        out.append(f'<line x1="{pl}" y1="{y(g):.1f}" x2="{W-pr}" y2="{y(g):.1f}" stroke="currentColor" opacity=".12"/>'
                   f'<text x="{pl-6}" y="{y(g)+3.5:.1f}" font-size="10" text-anchor="end" fill="currentColor" opacity=".6">{g:.2g}</text>')
    for k in range(5):
        m = mx * k / 4
        out.append(f'<text x="{x(m):.1f}" y="{H-pb+14}" font-size="10" text-anchor="middle" fill="currentColor" opacity=".6">{m:.3g}</text>')
    out.append(f'<text x="{(pl+W-pr)/2:.0f}" y="{H-4}" font-size="11" text-anchor="middle" fill="currentColor" opacity=".7">{html.escape(AXIS_LABEL.get(axis, axis))}</text>')
    if tau_line and tau is not None and lo <= tau <= hi:
        out.append(f'<line x1="{pl}" y1="{y(tau):.1f}" x2="{W-pr}" y2="{y(tau):.1f}" stroke="var(--tau)" stroke-width="1.5" stroke-dasharray="6 4"/>'
                   f'<text x="{W-pr-4}" y="{y(tau)-4:.1f}" font-size="10" text-anchor="end" fill="var(--tau)">τ = {tau:.3g}</text>')
    for t, pts in sorted(by_task.items()):
        pts = sorted(pts)
        poly = " ".join(f"{x(m):.1f},{y(v):.1f}" for m, v, _ in pts)
        col = "var(--fail)" if t in R038_NULL_FAIL else "var(--accent)"
        out.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="1.6" opacity=".85"/>')
        for m, v, ok in pts:
            if ok:
                out.append(f'<circle cx="{x(m):.1f}" cy="{y(v):.1f}" r="3" fill="var(--surface)" stroke="{col}" stroke-width="1.5"/>')
            else:
                out.append(f'<rect x="{x(m)-3.5:.1f}" y="{y(v)-3.5:.1f}" width="7" height="7" fill="{col}"/>')
        m, v, _ = pts[-1]
        out.append(f'<text x="{x(m)+5:.1f}" y="{y(v)+3:.1f}" font-size="9" fill="{col}">{SCENE.get(t, t)}</text>')
    out.append("</svg>")
    return "".join(out)


def render(run, axes, out):
    tau = json.load(open(os.path.join(run, "tau.json")))["tau"] if os.path.exists(os.path.join(run, "tau.json")) else None
    parts = ["<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>",
             "<title>R-041 Results</title>",
             "<link rel=stylesheet href='https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap'>",
             f"<style>{CSS}</style></head><body><div class=wrap>",
             "<h1>R-041 · where is the boundary</h1>",
             f"<div class=meta>GR00T N1.7 · libero_spatial controls (R-029, 100/100) · one seed and one fixed noise per task · τ = {tau if tau is None else f'{tau:.3g}'} "
             "(95th pct of forward-to-forward chunk distance at magnitude 0) · pre-registration in <code>RESULTS.md</code> R-041</div>",
             "<div class=note><b>How to read.</b> Action boundary: smallest magnitude at which the first chunk departs from the nominal chunk by more than τ. "
             "Outcome boundary: smallest magnitude at which the episode fails, searched above the action boundary. Brackets are bisection intervals. "
             "Squares are failures, circles successes. Lines in red are the three scenes that failed R-038's null prompt (stove, wooden cabinet, top drawer).</div>",
             "<div class=meta><i class=sw style='background:var(--accent)'></i>scene <i class=sw style='background:var(--fail)'></i>R-038 null-prompt failure scene <i class=sw style='background:var(--tau)'></i>τ</div>"]
    for axis in axes:
        rows, bounds = load_axis(run, axis)
        if not rows:
            continue
        mx = max(r["magnitude"] for r in rows)
        parts.append(f"<h2>{html.escape(AXIS_LABEL.get(axis, axis))} · {len(rows)} rollouts · {len(bounds)} tasks bracketed</h2>")
        trs = []
        for b in sorted(bounds, key=lambda b: b["task_id"]):
            samples = "".join(f"<div>{m:.3g}: {'<span class=fail>fail</span>' if not ok else '<span class=ok>ok</span>'} · ‖P−N‖ {d:.2f} · I {tf['I']:.2f}</div>"
                              for m, ok, d, tf in b["samples"])
            trs.append(f"<tr><td>{SCENE.get(b['task_id'], b['task_id'])}{' <span class=fail>*</span>' if b['task_id'] in R038_NULL_FAIL else ''}</td>"
                       f"<td class=num>{fmt_b(b.get('action_boundary'))}</td><td class=num>{fmt_b(b.get('outcome_boundary'))}</td>"
                       f"<td>{html.escape(b.get('note', ''))}</td><td style='font-size:.78rem'>{samples}</td></tr>")
        parts.append("<table><thead><tr><th>scene</th><th>action boundary</th><th>outcome boundary</th><th>note</th><th>samples (magnitude: outcome · ‖P−N‖ f0 · I f0)</th></tr></thead><tbody>"
                     + "".join(trs) + "</tbody></table>")
        parts.append(f"<figure>{plot(rows, axis, mx, tau, 'd_pn_f0', '‖P−N‖ at forward 0')}<figcaption>‖P−N‖ at forward 0 against magnitude, one line per scene. Expectation 1: monotone. Where a line crosses τ is the action boundary.</figcaption></figure>")
        parts.append(f"<figure>{plot(rows, axis, mx, None, 'tf_I', 'image-arm transfer at forward 0', ymax=1.05, tau_line=False)}<figcaption>Image-arm transfer at forward 0 against magnitude. Expectation 4: ≥ 0.8 below the action boundary on the render axes.</figcaption></figure>")
    parts.append("</div></body></html>")
    open(out, "w").write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--axes", default="camera_yaw_deg,camera_dist_m,light_intensity,joint_radius_rad")
    ap.add_argument("--out", default="docs/R041_RESULTS.html")
    a = ap.parse_args()
    axes = [x for x in a.axes.split(",") if os.path.isdir(os.path.join(a.run, x))]
    render(a.run, axes, a.out)
    for axis in axes:
        rows, bounds = load_axis(a.run, axis)
        print(axis, len(rows), "rollouts;", len(bounds), "tasks")
        for b in sorted(bounds, key=lambda b: b["task_id"]):
            print(f"  {SCENE.get(b['task_id'], b['task_id']):24s} action {fmt_b(b.get('action_boundary')):>16s}  outcome {fmt_b(b.get('outcome_boundary')):>16s}  {b.get('note', '')}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
