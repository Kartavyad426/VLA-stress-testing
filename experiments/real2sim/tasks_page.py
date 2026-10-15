"""Real-to-sim: presentation page for a make_tasks.py run. Writes <tasks>/index.html (per scene: task table with
instruction, type and expert success rate; every variant's video inline, failures marked) and <tasks>/montage.mp4
(grid of variant-0 videos, one per task, for slides).

    .venvs/real2sim/bin/python experiments/real2sim/tasks_page.py --tasks runs/real2sim/tasks
"""
import argparse, glob, html, json, os, subprocess

CSS = """:root{--bg:#fbfbf9;--fg:#1d1d1b;--muted:#5f5f5a;--line:#dddcd5;--card:#f2f1ec;--ok:#2d6a3e;--bad:#a33a2b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#e8e7e2;--muted:#a3a29b;--line:#34332f;--card:#1f1f1d;--ok:#86c79a;--bad:#e8907f}}
:root[data-theme="dark"]{--bg:#161615;--fg:#e8e7e2;--muted:#a3a29b;--line:#34332f;--card:#1f1f1d;--ok:#86c79a;--bad:#e8907f}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}main{max-width:1200px;margin:0 auto}
h1{font-size:1.5rem}h2{font-size:1.15rem;margin-top:2em;border-bottom:1px solid var(--line)}.meta{color:var(--muted);font-size:.9rem}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin:8px 0}th,td{border:1px solid var(--line);padding:5px 8px;text-align:left}th{background:var(--card)}
.ok{color:var(--ok);font-weight:700}.bad{color:var(--bad);font-weight:700}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:8px;margin:6px 0 18px}
.cell{background:var(--card);border:1px solid var(--line);border-radius:5px;padding:4px}.cell video{width:100%;display:block}
.cell div{font-size:.8rem;color:var(--muted)}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    a = ap.parse_args()
    summ = json.load(open(os.path.join(a.tasks, "summary.json")))
    tot_ok, tot = sum(t["successes"] for t in summ), sum(t["variants"] for t in summ)
    parts = [f"<h1>Tasks made from rebuilt real scenes</h1><p class='meta'>Scenes rebuilt from one real DROID video frame; "
             f"tasks generated automatically from the object inventory; each task solved by a scripted expert (IK + physics) on the "
             f"rebuilt scene (variant 0) and on randomised variants (objects jittered ±3 cm, blocks rotated). Success is judged from "
             f"sim state. Overall expert success: <b>{tot_ok}/{tot}</b> episodes, {len(summ)} tasks. Exploratory, not val-checked.</p>"]
    for scene in sorted({t["scene"] for t in summ}):
        rows = [t for t in summ if t["scene"] == scene]
        parts.append(f"<h2>{html.escape(scene)}: {len(rows)} tasks, expert success {sum(r['successes'] for r in rows)}/{sum(r['variants'] for r in rows)}</h2>")
        parts.append("<table><tr><th>#</th><th>Instruction</th><th>Type</th><th>Expert success</th></tr>")
        for i, r in enumerate(rows):
            parts.append(f"<tr><td>{i}</td><td>{html.escape(r['task'])}</td><td>{r['type']}</td><td>{r['successes']}/{r['variants']}</td></tr>")
        parts.append("</table>")
        for r in rows:
            parts.append(f"<h3>{html.escape(r['task'])}</h3><div class='grid'>")
            for js in sorted(glob.glob(os.path.join(a.tasks, r["dir"], "v*.json"))):
                ep = json.load(open(js))
                vid = os.path.relpath(js[:-5] + ".mp4", a.tasks)
                tag = "<span class='ok'>success</span>" if ep["success"] else "<span class='bad'>fail</span>"
                kind = "reconstructed scene" if ep["variant"] == 0 else f"variant {ep['variant']} (randomised)"
                parts.append(f"<div class='cell'><video controls muted preload='metadata' src='{vid}'></video><div>{kind}: {tag}</div></div>")
            parts.append("</div>")
    open(os.path.join(a.tasks, "index.html"), "w").write(
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Real-to-Sim Tasks</title><style>{CSS}</style></head><body><main>{''.join(parts)}</main></body></html>")

    # montage: variant-0 video of up to 9 tasks in a 3x3 grid, each labelled with its instruction
    vids = [(os.path.join(a.tasks, t["dir"], "v0.mp4"), t["task"]) for t in summ if os.path.exists(os.path.join(a.tasks, t["dir"], "v0.mp4"))][:9]
    while len(vids) % 3:
        vids.append(vids[-1])
    inputs, filt = [], []
    for k, (v, text) in enumerate(vids):
        inputs += ["-i", v]
        safe = text.replace(":", r"\:").replace("'", "")
        filt.append(f"[{k}:v]scale=320:180,tpad=stop_mode=clone:stop_duration=4,drawbox=x=0:y=0:w=320:h=18:color=black@0.7:t=fill,"
                    f"drawtext=text='{safe}':x=4:y=3:fontsize=11:fontcolor=white[v{k}]")
    rows = [f"{''.join(f'[v{r * 3 + c}]' for c in range(3))}hstack=inputs=3[r{r}]" for r in range(len(vids) // 3)]
    fc = ";".join(filt + rows) + f";{''.join(f'[r{r}]' for r in range(len(vids) // 3))}vstack=inputs={len(vids) // 3}[out]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", fc, "-map", "[out]", "-shortest",
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", os.path.join(a.tasks, "montage.mp4")], check=True)
    print("wrote", os.path.join(a.tasks, "index.html"), "and montage.mp4;", f"{tot_ok}/{tot} episodes succeeded over {len(summ)} tasks")


if __name__ == "__main__":
    main()
