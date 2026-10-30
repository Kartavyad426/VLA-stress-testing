"""Generate docs/TASKGEN_PRESETS.html (main checkout) from the preset JSON files and the agent-eval results.
User 2026-10-30: "also can you make a doc with preset if not already made". Format (user 2026-10-29 00:53): one row per
preset = what scene + what task + what variation (+ state); JSON shown colour-coded (user 2026-10-29).
Re-run after a test to refresh the result columns:
    python3 ../VLA-exp-taskgen/experiments/taskgen/presets_doc.py      (from the main checkout; no simulator needed)
Exploratory, UNVERIFIED.
"""
import glob, html, itertools, json, math, os, re, time

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.dirname(os.path.dirname(HERE))
MAIN = "/home/imerit/Documents/Code/VLA"
OUT = os.path.join(MAIN, "docs/TASKGEN_PRESETS.html")
EVAL = os.path.join(WT, "runs/taskgen/agent_eval_v3")
FILES = [  # (file, results folder tag, state note)
    ("v1_orange.json", "", "written by TaskGen 10-29; no user review recorded"),
    ("v1_orange_x.json", "x", "written by TaskGen 10-30 at the user's request; NOT verified by the user (user's words)"),
]
MODELS = ["nvidia/nemotron-3-ultra-550b-a55b", "openai/gpt-oss-20b", "nvidia/nemotron-3.5-lightning-30b-a3b"]
SHORT = {"nvidia/nemotron-3-ultra-550b-a55b": "550b", "openai/gpt-oss-20b": "gpt-oss", "nvidia/nemotron-3.5-lightning-30b-a3b": "lightning"}
esc = html.escape


def hl(obj):
    """Colour-coded JSON (keys / strings / numbers / booleans / null)."""
    s = esc(json.dumps(obj, indent=1))
    pat = r'(&quot;(?:[^&]|&(?!quot;))*?&quot;(\s*:)?|\b(true|false)\b|\bnull\b|-?\d+(?:\.\d+)?)'

    def f(m):
        t = m.group(0)
        c = "k" if t.endswith(":") else "s" if t.startswith("&quot;") else "b" if t in ("true", "false") else "z" if t == "null" else "n"
        return f'<span class="{c}">{t}</span>'
    return re.sub(pat, f, s)


def count(slots):
    n = 1
    for s in slots.values():
        k, v = s["kind"], s.get("values") or []
        n *= len(v) if k == "choice" else math.comb(len(v), s["k"]) if k == "choose" else math.perm(len(v), s["k"]) if k == "permute" else 1
    return n


def val_str(v):
    return v["colour"] if isinstance(v, dict) and "colour" in v else str(v)


def variation(slots):
    if not slots:
        return "none (one fixed case)"
    parts = []
    for n, s in slots.items():
        vals = ", ".join(val_str(v) for v in s.get("values") or [])
        how = {"choice": "one of", "choose": f"{s.get('k')} of (unordered)", "permute": f"{s.get('k')} of, in order"}.get(s["kind"], s["kind"])
        parts.append(f"<code>{esc(n)}</code>: {how} {{{esc(vals)}}}")
    return "; ".join(parts)


def scene_str(req, base):
    add = (req.get("scene") or {}).get("add") or []
    if not add:
        return f"{esc(base)} as reconstructed (nothing added)"
    out = []
    for o in add:
        what = o.get("colour", "") + " block" if o["source"] == "box" else o.get("noun") if o["source"] == "pack" else \
            f"{o.get('levels', 3)}-drawer cabinet, {o.get('open_level')} open" if o["source"] == "cabinet" else o["source"]
        rel = o["relation"].replace("_", " ") + ("" if not o.get("target") else f" {o['target']}")
        out.append(f"+ {esc(what)} ({esc(rel)})")
    return f"{esc(base)} " + "<br>".join(out)


def task_str(req):
    t = req.get("task")
    refused = req.get("infeasible") or []
    out = []
    if t:
        rules = "; ".join(f"{c['type']}({', '.join(x for x in [c.get('object'), c.get('target')] if x) or ' → '.join(c.get('order') or [])})"
                          for c in t.get("pass_all") or [])
        out.append(f"“{esc(t['instruction'])}”<br><span class='note'>pass: {esc(rules)}</span>")
    elif not refused:
        out.append("<i>none (scene only)</i>")
    for r in refused:
        out.append(f"<span class='warn'>refuse {esc(r['part'])}</span>: {esc(r['reason'])}" +
                   (f" → {esc(r['alternative'])}" if r.get("alternative") else ""))
    return "<br>".join(out)


def kind(req):
    k = ("scene" if (req.get("scene") or {}).get("add") else "") + ("+" if (req.get("scene") or {}).get("add") and req.get("task") else "") + \
        ("task" if req.get("task") else "")
    if req.get("infeasible"):
        k = (k + " + " if k else "") + "refusal"
    return k


def results(tag, pid):
    """Per model: suite and variants match / calls for this preset (from the per-call JSON files)."""
    cells = []
    for m in MODELS:
        parts = []
        for test in ("suite", "variants"):
            d = os.path.join(EVAL, test + (f"_{tag}" if tag else ""), m.replace("/", "__"))
            rows = [json.load(open(f)) for f in glob.glob(os.path.join(d, f"{pid}_v*_r*.json"))]
            if rows:
                ok = sum(bool(r["match"]) for r in rows)
                bad = sum(not r["valid"] for r in rows)
                c = "ok" if ok == len(rows) else "warn" if ok else "bad"
                parts.append(f"<span class='{c}'>{test[0]} {ok}/{len(rows)}</span>" + (f" <span class='note'>({bad} err)</span>" if bad else ""))
        cells.append("<br>".join(parts) if parts else "<span class='note'>—</span>")
    return cells


def mismatches(tag, pid):
    out = []
    for m in MODELS:
        for test in ("suite", "variants"):
            d = os.path.join(EVAL, test + (f"_{tag}" if tag else ""), m.replace("/", "__"))
            for f in sorted(glob.glob(os.path.join(d, f"{pid}_v*_r*.json"))):
                r = json.load(open(f))
                if r["match"]:
                    continue
                c = r.get("compare") or {}
                why = r.get("reason") if not r["valid"] else "; ".join(f"{k}: {json.dumps(v)}" for k, v in c.items() if k.endswith("diff"))
                out.append(f"<tr><td>{SHORT[m]}</td><td>{esc(os.path.basename(f)[:-5])}</td><td>{esc(r['prompt'])}</td><td class='note'>{esc(str(why))[:400]}</td></tr>")
    return out


def main():
    stamp = time.strftime("%Y-%m-%d %H:%M IST")
    rows, cards, n_p, n_v = [], [], 0, 0
    for fname, tag, state in FILES:
        P = json.load(open(os.path.join(HERE, "presets", fname)))
        base = P["scene"].split("runs/real2sim/")[-1]
        for p in P["presets"]:
            td = p["template_def"]
            n = count(td["slots"])
            n_p += 1
            n_v += n
            req = td["request"]
            res = results(tag, p["id"])
            rows.append(f"<tr><td><b>{esc(p['id'])}</b><br><span class='note'>{esc(fname)}</span></td><td><span class='lv'>{esc(p.get('level', ''))}</span> "
                        f"{esc(p.get('template', ''))}<br><span class='note'>{esc(kind(req))}</span></td><td>{scene_str(req, base)}</td><td>{task_str(req)}</td>"
                        f"<td>{variation(td['slots'])}<br><span class='note'>{n} variant(s)</span></td>" + "".join(f"<td>{c}</td>" for c in res) + "</tr>")
            mm = mismatches(tag, p["id"])
            cards.append(f"<details><summary><b>{esc(p['id'])}</b> · {esc(p.get('template', ''))} · {esc(p.get('note', '') or kind(req))}"
                         f"{' · <span class=warn>' + str(len(mm)) + ' mismatch(es)</span>' if mm else ''}</summary>"
                         f"<p class='meta'>State: {esc(state)}.</p>"
                         "<p><b>Prompts</b> (slots in braces):</p><ul>" + "".join(f"<li>{esc(x)}</li>" for x in p["prompts"]) + "</ul>"
                         f"<p><b>Expected answer</b> (template, before the slots are filled):</p><pre>{hl(td)}</pre>" +
                         (("<p><b>Calls that didn't match</b>:</p><div class='wrap'><table><tr><th>Model</th><th>Call</th><th>Prompt</th><th>Difference</th></tr>"
                           + "".join(mm) + "</table></div>") if mm else "") + "</details>")
    rob = json.load(open(os.path.join(HERE, "presets/v1_orange_robustness.json")))["cases"]
    rob_rows = "".join(f"<tr><td><b>{esc(c['id'])}</b></td><td>{esc(c['prompt'])}</td><td><pre class='inline'>{hl(c['expect'])}</pre></td></tr>" for c in rob)
    legacy = json.load(open(os.path.join(HERE, "presets/orange.json")))["presets"]
    leg_rows = "".join(f"<tr><td><b>{esc(p['id'])}</b></td><td>{esc(p['prompts'][0])}</td><td class='note'>{len(p['prompts'])} prompts</td></tr>" for p in legacy)
    queue = open(os.path.join(EVAL, "queue_x.log")).read().strip().splitlines() if os.path.exists(os.path.join(EVAL, "queue_x.log")) else []
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TaskGen presets</title>
<style>
:root{{--bg:#fbfbf9;--fg:#1d1d1b;--muted:#5f5f5a;--line:#dddcd5;--card:#f2f1ec;--accent:#7a5c2e;--ok:#2c6e3a;--warn:#8a6d12;--bad:#a3321f}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--fg:#e8e7e2;--muted:#a3a29b;--line:#34332f;--card:#1f1f1d;--accent:#c79a5e;--ok:#7fc98d;--warn:#d9c46a;--bad:#ee8a76}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:1250px;margin:0 auto}} h1{{font-size:1.5rem;margin:.2em 0}}
h2{{font-size:1.15rem;margin-top:1.9em;border-bottom:2px solid var(--accent);padding-bottom:.2em}}
.meta,.note{{color:var(--muted);font-size:.86rem}} .box{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:6px;padding:10px 14px;margin:12px 0}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:.85rem}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}} th{{background:var(--card)}}
pre{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 10px;overflow-x:auto;font-size:.8rem}} pre.inline{{margin:0}}
details{{border:1px solid var(--line);border-radius:6px;padding:6px 10px;margin:6px 0}} summary{{cursor:pointer}}
.lv{{display:inline-block;min-width:2.2em;text-align:center;font-weight:700;color:#fff;background:var(--accent);border-radius:4px;padding:0 4px}}
a{{color:var(--accent)}} .ok{{color:var(--ok);font-weight:600}}.warn{{color:var(--warn);font-weight:600}}.bad{{color:var(--bad);font-weight:600}}
.k{{color:#8a5a2b}}.s{{color:#2c6e3a}}.n{{color:#2a5d8a}}.b{{color:#8a2b6e}}.z{{color:var(--muted)}}
@media (prefers-color-scheme:dark){{.k{{color:#d29a62}}.s{{color:#7fc98d}}.n{{color:#8ab4e0}}.b{{color:#d88ac4}}}}
</style></head><body><main>
<h1>TaskGen presets</h1>
<p class="meta"><b>Generated {stamp} by experiments/taskgen/presets_doc.py</b> (clock time; re-run it to refresh the results). Started at the user's request
(2026-10-30: <i>"also can you make a doc with preset if not already made"</i>). Format (user, 2026-10-29 00:53): <i>"what scene, what task, what variation"</i>.
Templates and difficulty levels: <a href="TASKGEN_TEMPLATES.html">TASKGEN_TEMPLATES.html</a>. Design: <a href="TASKGEN_DESIGN.html">TASKGEN_DESIGN.html</a>.</p>
<div class="box"><b>Count.</b> {n_p} presets on res31's v1/orange scene ({n_v} variants), plus {len(rob)} robustness cases and {len(legacy)} older presets on the pre-v1 build (§3).
<b>Base objects</b> (locked for the setup, movable by the task): block_1 yellow, block_2 orange, block_3 green, bowl_1 black bowl. The real episode's task: “Put the orange block inside the black bowl”.<br>
<b>Results columns</b> (agent v3 via NIM): <b>s</b> = suite (first variant × each prompt × 2 runs), <b>v</b> = variants (every variant, first prompt, 1 run); x/y = answers matching the expected answer.
A mismatch is not always the model's fault: the X presets' expected answers are TaskGen's judgement, not reviewed by the user. Agent answers only, no policy runs here. UNVERIFIED.</div>
<h2>1 · Presets: scene, task, variation</h2>
<div class="wrap"><table><tr><th>Preset</th><th>Template</th><th>Scene</th><th>Task</th><th>Variation</th>{''.join(f'<th>{SHORT[m]}</th>' for m in MODELS)}</tr>
{''.join(rows)}</table></div>
<h2>2 · Each preset in full (prompts, expected answer, mismatches)</h2>
{''.join(cards)}
<h2>3 · Robustness cases and older presets</h2>
<p class="meta">Robustness: prompts checked against an expectation, not a full answer (presets/v1_orange_robustness.json; the older orange_robustness.json has the same 8 ids).</p>
<div class="wrap"><table><tr><th>Case</th><th>Prompt</th><th>Expectation</th></tr>{rob_rows}</table></div>
<p class="meta">Older presets on runs/real2sim/ep_orange_block_stereo (presets/orange.json, v1/v2 answer format, tested 10-28/29 with the v1/v2 agent):</p>
<div class="wrap"><table><tr><th>Preset</th><th>First prompt</th><th></th></tr>{leg_rows}</table></div>
<h2>4 · Test runs</h2>
<pre>{esc(chr(10).join(queue)) or 'no queue log'}</pre>
</main></body></html>"""
    open(OUT, "w").write(doc)
    print(OUT, n_p, "presets", n_v, "variants")


if __name__ == "__main__":
    main()
