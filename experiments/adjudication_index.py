"""Build the hand-adjudication index: every rendered failure, with what the
classifier said about it, in one sortable page.

    python3 experiments/adjudication_index.py runs/lplus_fail_groot viz/lplus_fail_groot

The point is to make disagreement cheap to spot. Each row links to the episode
page (both cameras, replayed) and states the family and predicates the mining
layer assigned. You watch the video, decide what actually went wrong, and
compare. Rows are grouped by perturbation type, because that is where the
suspicion is: a robot-initial-state failure labelled `visual_grounding` is the
pattern to look for.

Verdicts are recorded in a CSV you edit (`--verdicts`), not in the page: a file
survives a re-render, browser state does not.
"""
import json, os, sys, html, collections

def build(run_dir: str, viz_dir: str, out: str, verdicts_csv: str) -> None:
    diag = {r["rollout_id"]: r for r in
            (json.loads(l) for l in open(f"{run_dir}/diagnoses.jsonl"))}
    pages = {}
    for f in sorted(os.listdir(viz_dir)):
        if f.endswith(".html") and f != os.path.basename(out):
            rid = f.rsplit("_", 1)[-1][:-len(".html")]
            pages[rid] = f
    rows = [d for rid, d in diag.items() if not d["success"]]
    rows.sort(key=lambda d: (d["category"] or "", -(d["level"] or 0)))

    prior = {}
    if os.path.exists(verdicts_csv):
        import csv
        for r in csv.DictReader(open(verdicts_csv)):
            prior[r["rollout_id"]] = r

    by_cat = collections.Counter(d["category"] for d in rows)
    rendered = sum(1 for d in rows if d["rollout_id"] in pages)
    parts = [f"""<!doctype html><meta charset=utf-8><title>Failure adjudication — {html.escape(run_dir)}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:24px;max-width:1200px;color:#111}}
 h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#666;margin-bottom:20px}}
 table{{border-collapse:collapse;width:100%}} th,td{{padding:6px 8px;border-bottom:1px solid #e5e5e5;vertical-align:top;text-align:left}}
 th{{background:#fafafa;position:sticky;top:0;font-weight:600}}
 tr.cat td{{background:#f3f4f6;font-weight:600}}
 .fam{{font-family:ui-monospace,monospace;font-size:12px}}
 .pred{{color:#555;font-size:12px}} .no{{color:#b00}} .yes{{color:#060}}
 a{{color:#0645ad}} .term{{font-size:12px;color:#444}}
</style>
<h1>Failure adjudication — {html.escape(run_dir)}</h1>
<div class=sub>{len(rows)} failures, {rendered} with a rendered episode page.
Watch the episode, decide what actually went wrong, and record it in
<code>{html.escape(verdicts_csv)}</code> (columns: rollout_id, verdict, note).
The classifier's label is shown so you can disagree with it, not so you can agree.</div>
<table><tr><th>type / level</th><th>episode</th><th>classifier family</th>
<th>predicates</th><th>terminal</th><th>steps</th><th>your verdict</th></tr>"""]
    cur = None
    for d in rows:
        if d["category"] != cur:
            cur = d["category"]
            parts.append(f'<tr class=cat><td colspan=7>{html.escape(str(cur))} '
                         f'— {by_cat[cur]} failures</td></tr>')
        rid = d["rollout_id"]
        page = pages.get(rid)
        link = (f'<a href="{html.escape(page)}">{rid[:12]}</a>' if page
                else f'<span class=no>{rid[:12]} (not rendered)</span>')
        preds = " ".join(f'<span class="{"yes" if v else "pred"}">{html.escape(k)}</span>'
                         for k, v in (d["predicates"] or {}).items() if v)
        v = prior.get(rid, {})
        verdict = html.escape(f'{v.get("verdict","")} {v.get("note","")}'.strip())
        parts.append(
            f'<tr><td>L{d["level"]}</td><td>{link}</td>'
            f'<td class=fam>{html.escape(str(d["family"]))}</td>'
            f'<td>{preds}</td><td class=term>{html.escape(str(d["terminal"]))}</td>'
            f'<td>{d["n_steps"]}</td><td>{verdict}</td></tr>')
    parts.append("</table>")
    open(out, "w").write("\n".join(parts))

    if not os.path.exists(verdicts_csv):
        with open(verdicts_csv, "w") as f:
            f.write("rollout_id,category,level,classifier_family,verdict,note\n")
            for d in rows:
                if d["rollout_id"] in pages:
                    f.write(f'{d["rollout_id"]},{d["category"]},{d["level"]},'
                            f'{d["family"]},,\n')
    print(f"wrote {out} ({len(rows)} failures, {rendered} rendered)")
    print(f"verdict sheet: {verdicts_csv}")


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/lplus_fail_groot"
    viz = sys.argv[2] if len(sys.argv) > 2 else "viz/lplus_fail_groot"
    build(run, viz, f"{viz}/index.html", f"{run}/adjudication.csv")
