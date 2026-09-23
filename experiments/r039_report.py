"""R-039 report: runs/<run>/splice -> runs/<run>/analysis.json + docs/R039_RESULTS.html.

  python3 experiments/r039_report.py runs/r039 [runs/r039_recorded] [--out docs/R039_RESULTS.html]

Standalone HTML, no external scripts (the project ships write-ups as local
files). Inline SVG: one transfer-fraction curve per instance, three arms.
Level is colour-coded on every instance row, as asked.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from vla_harness.analysis.r039 import ARMS as BASE_ARMS, aggregate, load_instances  # noqa: E402

ARM_NAME = {"T": "text tokens", "I": "image tokens", "S": "state token",
            "IT": "image + text tokens", "A": "agent-view pixels", "W": "wrist pixels"}
ARMS = list(BASE_ARMS)          # extended per run below
CATS = ("Camera Viewpoints", "Light Conditions", "Sensor Noise", "Robot Initial States")

CSS = """
:root{color-scheme:light dark;--bg:#F3F5F8;--surface:#FFFFFF;--surface-2:#EDF0F4;--ink:#14181D;--muted:#5B636E;
 --rule:#D7DDE5;--rule-soft:#E6EAEF;--accent:#2F4A9E;
 --arm-T:#2a78d6;--arm-I:#eb6834;--arm-S:#1baf7a;--arm-IT:#eda100;--arm-A:#e87ba4;--arm-W:#4a3aa7;
 --L1:#e8f1fb;--L2:#cfe0f6;--L3:#a9c8ee;--L4:#7ea9e0;--L5:#4f86cf;--Lfg5:#ffffff;
 --sans:'IBM Plex Sans',system-ui,sans-serif;--mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0E1116;--surface:#151A21;--surface-2:#1B212A;
 --ink:#E5EAF1;--muted:#9AA3AF;--rule:#262D37;--rule-soft:#1F2630;--accent:#93A9F0;
 --arm-T:#3987e5;--arm-I:#d95926;--arm-S:#199e70;--arm-IT:#c98500;--arm-A:#d55181;--arm-W:#9085e9;
 --L1:#1a2230;--L2:#1f2f47;--L3:#25405f;--L4:#2c5280;--L5:#3567a5;}}
:root[data-theme="dark"]{--bg:#0E1116;--surface:#151A21;--surface-2:#1B212A;--ink:#E5EAF1;--muted:#9AA3AF;
 --rule:#262D37;--rule-soft:#1F2630;--accent:#93A9F0;--arm-T:#3987e5;--arm-I:#d95926;--arm-S:#199e70;--arm-IT:#c98500;--arm-A:#d55181;--arm-W:#9085e9;
 --L1:#1a2230;--L2:#1f2f47;--L3:#25405f;--L4:#2c5280;--L5:#3567a5;}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.55}
.wrap{max-width:1100px;margin:0 auto;padding:32px 16px 80px}
h1{font-size:1.9rem;margin:0 0 6px;letter-spacing:-.02em}h2{font-size:1.25rem;margin:2.2em 0 .6em}h3{font-size:1rem;margin:1.4em 0 .4em}
.meta{color:var(--muted);font-size:.85rem;margin-bottom:22px}
table{border-collapse:collapse;width:100%;font-size:.84rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 10px 6px 0;border-bottom:1px solid var(--rule-soft);vertical-align:top}
th{font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--rule)}
td.num{text-align:right;padding-right:14px}
.lvl{display:inline-block;min-width:2.2em;text-align:center;border-radius:3px;padding:1px 6px;font-family:var(--mono);font-size:.78rem}
.L1{background:var(--L1)}.L2{background:var(--L2)}.L3{background:var(--L3)}.L4{background:var(--L4)}.L5{background:var(--L5);color:var(--Lfg5)}
.ok{color:#1B6B4F;font-weight:600}.fail{color:#A8481B;font-weight:600}
.legend{display:flex;gap:18px;font-size:.82rem;color:var(--muted);margin:8px 0 14px;flex-wrap:wrap}
.sw{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:6px;border-radius:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px}
figure{margin:0;background:var(--surface);border:1px solid var(--rule-soft);border-radius:4px;padding:10px}
figure svg{display:block;width:100%;height:auto;color:var(--ink)}
figcaption{font-size:.78rem;color:var(--muted);margin-top:6px;line-height:1.4}
.note{background:var(--surface);border:1px solid var(--rule);border-left:3px solid var(--accent);border-radius:4px;padding:12px 14px;margin:14px 0;font-size:.9rem}
code{font-family:var(--mono);font-size:.85em;background:var(--surface-2);padding:.05em .3em;border-radius:3px}
"""


def lvl(n):
    return f'<span class="lvl L{n}">L{n}</span>'


def curve_svg(inst: dict, aid: str) -> str:
    F = len(inst["tf"]["T"])
    W, H, pl, pr, pt, pb = 250, 130, 30, 8, 8, 22
    x = lambda i: pl + (W - pl - pr) * (i / max(F - 1, 1))
    y = lambda v: pt + (H - pt - pb) * (1 - min(max(v, -0.25), 1.05) + 0.05) / 1.3
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="transfer fraction per forward for {html.escape(str(inst.get("label")))}">']
    for g in (0.0, 0.5, 1.0):
        parts.append(f'<line x1="{pl}" y1="{y(g):.1f}" x2="{W-pr}" y2="{y(g):.1f}" stroke="currentColor" opacity=".15"/>'
                     f'<text x="{pl-4}" y="{y(g)+3.5:.1f}" font-size="9" text-anchor="end" fill="currentColor" opacity=".6">{g:g}</text>')
    for arm in inst["tf"]:
        pts = " ".join(f"{x(i):.1f},{y(float(v)):.1f}" for i, v in enumerate(inst["tf"][arm]))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--arm-{arm})" stroke-width="2" stroke-linejoin="round"/>')
    a = inst.get("anchor_forward")
    if a is not None and 0 <= a < F:
        parts.append(f'<line x1="{x(a):.1f}" y1="{pt}" x2="{x(a):.1f}" y2="{H-pb}" stroke="currentColor" stroke-dasharray="3 3" opacity=".5"/>')
    parts.append(f'<text x="{pl}" y="{H-6}" font-size="9" fill="currentColor" opacity=".6">forward 0</text>'
                 f'<text x="{W-pr}" y="{H-6}" font-size="9" text-anchor="end" fill="currentColor" opacity=".6">{F-1}</text></svg>')
    return "".join(parts)


def _arms_of(agg: dict) -> list:
    for c in agg["by_category"].values():
        return list(c["median_tf_f0"])
    return list(BASE_ARMS)


def category_table(agg: dict, title: str) -> str:
    ARMS = _arms_of(agg)
    rows = []
    for cat in CATS:
        c = agg["by_category"].get(cat)
        if not c:
            continue
        dom = ", ".join(f"{k} {v}" for k, v in c["dominant"].items())
        rows.append(f"<tr><td>{cat}</td><td class=num>{c['n']}</td><td class=num>{c['n_fail']}</td>"
                    f"<td class=num>{c['median_d_pn_f0']:.2f}</td>"
                    + "".join(f"<td class=num>{c['median_tf_f0'][a]:.2f}</td>" for a in ARMS)
                    + "".join(f"<td class=num>{c['median_tf_anchor'][a]:.2f}</td>" for a in ARMS)
                    + f"<td class=num>{c['S_slope_per_forward']['median']:+.3f} ({c['S_slope_per_forward']['n_positive']}↑)</td><td>{dom}</td></tr>")
    return (f"<h3>{title}</h3><table><thead><tr><th>category</th><th>n</th><th>fail</th><th>‖P−N‖ f0</th>"
            + "".join(f"<th>{a} f0</th>" for a in ARMS) + "".join(f"<th>{a} anchor</th>" for a in ARMS)
            + "<th>S slope</th><th>dominant @anchor</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def instance_table(agg: dict) -> str:
    ARMS = _arms_of(agg)
    rows = []
    for p in sorted(agg["instances"], key=lambda p: (CATS.index(p["category"]) if p["category"] in CATS else 9, -(p["level"] or 0), p["task_id"])):
        out = '<span class=ok>success</span>' if p["success"] else '<span class=fail>fail</span>'
        rows.append(f"<tr><td>{lvl(p['level'])}</td><td>{html.escape(str(p['label']))}</td><td class=num>{p['task_id']}</td><td>{out}</td>"
                    f"<td class=num>{p['forwards']}</td><td class=num>{p['anchor_forward']}</td><td class=num>{(p['d_pn_f0'] or 0):.2f}</td>"
                    + "".join(f"<td class=num>{p['tf_f0'][a]:.2f}</td>" for a in ARMS)
                    + "".join(f"<td class=num>{p['tf_anchor'][a]:.2f}</td>" for a in ARMS)
                    + f"<td>{p['dominant'] or '<span class=fail>none</span>'}</td></tr>")
    return ("<table><thead><tr><th>level</th><th>instance</th><th>task</th><th>outcome</th><th>fwds</th><th>anchor</th><th>‖P−N‖ f0</th>"
            + "".join(f"<th>{a} f0</th>" for a in ARMS) + "".join(f"<th>{a} anc</th>" for a in ARMS)
            + "<th>dominant</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def render(runs: list[tuple[str, list, dict]], out: str):
    parts = ["<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>",
             "<title>R-039 Results</title>",
             "<link rel=stylesheet href='https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap'>",
             f"<style>{CSS}</style></head><body><div class=wrap>",
             "<h1>R-039 · which pathway carries a perturbation to the action</h1>",
             "<div class=meta>GR00T N1.7 · LIBERO-Plus libero_spatial · transfer fraction = 1 − ‖patched − N‖ / ‖P − N‖ on the 7 live action dims · "
             "anchor = forward containing the closest approach to the target bowl · pre-registration in <code>RESULTS.md</code> R-039</div>",
             "<div class=legend>" + "".join(f'<span><i class=sw style="background:var(--arm-{a})"></i>{a} · {ARM_NAME[a]}</span>' for a in ARM_NAME)
             + "<span>level: " + " ".join(lvl(i) for i in range(1, 6)) + "</span></div>"]
    parts.append("<div class=note><b>How to read.</b> P is the perturbed run's own chunk, N the chunk from the nominal source. "
                 "Each arm puts ONE input back to nominal; its transfer says how much of the P→N gap that alone closes. "
                 "Paired-render pass: the source is the same sim state re-rendered under the trained condition, so the S arm is 0 by construction there. "
                 "Recorded pass: the source is the scene's control rollout under the same noise, off-manifold after forward 0, reported as a bound.</div>")
    for name, inst, agg in runs:
        parts.append(f"<h2>{html.escape(name)} · {agg['n']} instances · residual (no arm ≥ 0.9 at anchor): {agg['residual']['n']} ({agg['residual']['fraction']:.0%})</h2>")
        parts.append(category_table(agg, "By category (medians)"))
        parts.append("<h3>Per instance</h3>" + instance_table(agg))
        parts.append("<h3>Transfer curves</h3><div class=grid>")
        for i in sorted(inst, key=lambda p: (CATS.index(p["category"]) if p["category"] in CATS else 9, p["task_id"])):
            parts.append(f"<figure>{curve_svg(i, i['rollout_id'])}<figcaption>{lvl(i.get('level'))} {html.escape(str(i.get('label')))} · "
                         f"{'success' if i.get('success') else 'fail'} · dashed = anchor</figcaption></figure>")
        parts.append("</div>")
    parts.append("</div></body></html>")
    with open(out, "w") as f:
        f.write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="docs/R039_RESULTS.html")
    a = ap.parse_args()
    bundle = []
    for r in a.runs:
        inst = load_instances(r)
        agg = aggregate(inst)
        json.dump(agg, open(os.path.join(r, "analysis.json"), "w"), indent=1, default=float)
        bundle.append((os.path.basename(r.rstrip("/")), inst, agg))
        print(r, "n =", agg["n"], "residual =", agg["residual"]["n"])
        for cat, c in agg["by_category"].items():
            print(f"  {cat:22s} n={c['n']:2d} fail={c['n_fail']:2d} f0 " + " ".join(f"{k}={v:.2f}" for k, v in c["median_tf_f0"].items())
                  + " | anchor " + " ".join(f"{k}={v:.2f}" for k, v in c["median_tf_anchor"].items()) + f" | dom {c['dominant']}")
    render(bundle, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
