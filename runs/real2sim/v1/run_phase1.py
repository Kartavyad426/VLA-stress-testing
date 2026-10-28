"""res31 10-29: unattended SceneGen v1 batch over res31H's phase-1 list (docs/SCENE_SOURCES_V1.html, 14 scenes / 34
candidates). User 10-29: "lets make all the scenes? everything is automated right, I am going away so wont be able to
allow anything".

Order: one calibrated candidate per scene first (picks before reserves), then the remaining calibrated candidates.
Calibrated = a published exterior camera (DROID refined or PointWorld) for camera 22008760, else 24400334 (that camera
is then used). Per candidate: download (trajectory.h5, metadata, six MP4s; no SVO) -> build_v1 --auto_nouns (VLM nouns,
unreviewed: A6 reads "needs review") -> v1_report. Resumable: a candidate with v1_report.json is skipped; a download
with all files present is skipped. CPU only (build_v1's steps run via py_triton_first.sh). Waits while MemAvailable
< 10 GiB. Writes runs/real2sim/v1/phase1/index.html (one row per scene/candidate) after every candidate.
Exploratory, UNVERIFIED.
"""
import html, json, os, re, subprocess, sys, time, urllib.request
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "experiments/real2sim"))
os.chdir(REPO)
import camera_source as CS  # noqa: E402

OUT = os.path.join(REPO, "runs/real2sim/v1/phase1")
DATA = os.path.join(REPO, "data/real2sim/droid/phase1")
BASE = "https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1/"
PATHS = json.load(open(os.path.join(REPO, "data/real2sim/droid/scouting/ann/episode_id_to_path.json")))
LOG = os.path.join(OUT, "run.log")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)


def say(s):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + s
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")


def candidates():
    h = open(os.path.join(REPO, "docs/SCENE_SOURCES_V1.html")).read()
    parts = re.split(r'(<h3 id="p1s\d+">[^<]*</h3>|<div class="cand" id="c\d+">)', h)
    scene, rows = None, []
    for i, p in enumerate(parts):
        m = re.match(r'<h3 id="p1s\d+">([^<]*)</h3>', p)
        if m:
            scene = m.group(1)
            continue
        if p.startswith('<div class="cand"'):
            body = parts[i + 1]
            t = re.search(r'<h4>#(\d+) · (.*?) <span class="tag (\w+)', body)
            u = re.search(r'(AUTOLab\+[0-9a-z]+\+20\d\d-\d\d-\d\d-\d\dh-\d\dm-\d\ds)(?!_)', body)
            if t and u:
                rows.append(dict(scene=scene, n=int(t.group(1)), title=html.unescape(t.group(2)), pick=t.group(3) == "pick", uuid=u.group(1)))
    return rows


def calibrated_cam(uuid):
    for cam in ("22008760", "24400334"):
        c = CS.from_droid_refined(uuid, cam) or CS.from_pointworld(uuid, cam)
        if c:
            return cam, c["source"].split(" (")[0]
    return None, None


def download(uuid, dest):
    rel = PATHS[uuid]
    os.makedirs(dest, exist_ok=True)
    files = ["trajectory.h5", f"metadata_{uuid}.json"] + [f"recordings/MP4/{s}{x}.mp4" for s in ("18026681", "22008760", "24400334") for x in ("", "-stereo")]
    for f in files:
        local = os.path.join(dest, os.path.basename(f))
        if os.path.exists(local) and os.path.getsize(local) > 0:
            continue
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(BASE + rel + "/" + f, local + ".part")
                os.replace(local + ".part", local)
                break
            except Exception as e:  # noqa: BLE001
                say(f"  download {uuid} {f} attempt {attempt + 1}: {e!r}")
                time.sleep(10)
        else:
            return False
    return True


def mem_ok():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1]) / 1048576 >= 10
    return True


def index(rows):
    trs = []
    for r in rows:
        rep = os.path.join(OUT, r["name"], "v1_report.json")
        if os.path.exists(rep):
            j = json.load(open(rep))
            t0 = [x for x in j["rows"] if x["tier"] == 0]
            ok = sum(x["state"] == "pass" for x in t0)
            open0 = ", ".join(f"{x['id']} {x['state']}" for x in j["tier0_open"])
            state = f"Tier 0: {ok}/{len(t0)} pass; open: {open0}"
            link = f'<a href="{r["name"]}/v1_report.html">report</a>'
        else:
            state, link = r.get("status", "not run"), ""
        trs.append(f"<tr><td>{html.escape(r['scene'])}</td><td>#{r['n']} {'pick' if r['pick'] else 'reserve'}</td><td>{html.escape(r['title'][:90])}</td>"
                   f"<td>{r['cam']} ({html.escape(r['camsrc'])})</td><td>{html.escape(state)}</td><td>{link}</td></tr>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SceneGen v1 Phase 1</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;margin:0;padding:20px 16px;background:#fbfbf9;color:#1d1d1b}}main{{max-width:1300px;margin:0 auto}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:5px 7px;text-align:left;vertical-align:top}}th{{background:#f2f1ec}}
a{{color:#8a5a2b}}@media (prefers-color-scheme:dark){{body{{background:#161615;color:#e8e7e2}}th{{background:#1f1f1d}}td,th{{border-color:#34332f}}a{{color:#d29a62}}}}</style></head>
<body><main><h1>SceneGen v1, phase 1 (res31H's 14 scenes)</h1><p>Unattended batch (res31, user 10-29). Nouns from the VLM, NOT reviewed: every A6 reads "needs review".
Updated {time.strftime('%Y-%m-%d %H:%M')}. Exploratory, UNVERIFIED.</p>
<table><tr><th>Scene</th><th>Candidate</th><th>Task (annotation)</th><th>Camera</th><th>State</th><th></th></tr>{''.join(trs)}</table></main></body></html>"""
    open(os.path.join(OUT, "index.html"), "w").write(page)


def main():
    rows = candidates()
    for r in rows:
        r["cam"], r["camsrc"] = calibrated_cam(r["uuid"])
        r["camsrc"] = r["camsrc"] or "none"
        r["name"] = f"c{r['n']:02d}_" + r["uuid"].split("+")[-1]
    cal = [r for r in rows if r["cam"]]
    say(f"{len(rows)} candidates in {len({r['scene'] for r in rows})} scenes; calibrated {len(cal)}")
    first, rest, seen = [], [], set()
    for r in sorted(cal, key=lambda r: (not r["pick"], r["n"])):
        (rest if r["scene"] in seen else first).append(r)
        seen.add(r["scene"])
    order = sorted(first, key=lambda r: r["n"]) + sorted(rest, key=lambda r: r["n"])
    for r in rows:
        if not r["cam"]:
            r["status"] = "skipped: no published calibration"
    index(rows)
    for k, r in enumerate(order):
        out = os.path.join(OUT, r["name"])
        if os.path.exists(os.path.join(out, "v1_report.json")):
            say(f"[{k + 1}/{len(order)}] {r['name']}: done before, skip")
            continue
        while not mem_ok():
            say("  MemAvailable < 10 GiB: waiting")
            time.sleep(60)
        say(f"[{k + 1}/{len(order)}] {r['name']} ({r['scene']}; camera {r['cam']} {r['camsrc']}): download")
        ep = os.path.join(DATA, r["name"])
        if not download(r["uuid"], ep):
            r["status"] = "download failed"
            say(f"  download failed: skip")
            index(rows)
            continue
        say("  build_v1")
        t0 = time.time()
        rc = subprocess.run([os.path.join(REPO, ".venvs/real2sim/bin/python"), "-c",
                             "import triton, runpy, sys; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')",
                             os.path.join(REPO, "experiments/real2sim/build_v1.py"), "--episode", ep, "--out", out, "--cam", r["cam"], "--auto_nouns"],
                            stdout=open(os.path.join(OUT, r["name"] + ".log"), "w"), stderr=subprocess.STDOUT,
                            env=dict(os.environ, MUJOCO_GL="osmesa", CUDA_VISIBLE_DEVICES="")).returncode
        r["status"] = f"build_v1 rc {rc}"
        say(f"  build_v1 rc {rc} ({time.time() - t0:.0f} s)")
        index(rows)
    say("all done")
    index(rows)


if __name__ == "__main__":
    main()
