#!/usr/bin/env python3
"""A reference episode and any number of perturbed variants, as ONE page.

Every episode is replayed from its stored actions (no policy inference) and
rendered, and all of them share one time cursor: scrub once and every video and
every plot moves to the same env step. Generalises `init_state_compare.py`, which
is hard-wired to three robot-initial-state episodes.

    MUJOCO_GL=egl PYTHONPATH=third_party/LIBERO-plus \\
    LIBERO_CONFIG_PATH=third_party/libero-plus-config \\
    .venvs/libero-plus/bin/python experiments/visualise_set.py \\
        lplus_fail_groot:5115b970e766 lplus_fail_groot:550e57403a39 \\
        [--reference groot_control_lplus_stack:384eaa6974c4] [--out viz/x.html]

Episodes are `RUN:ROLLOUT_ID` (a rollout id can exist in several runs). Without
`--reference`, one is found automatically: same base scene, same checkpoint,
unperturbed variant (camera 0, init state 0, base instruction, no other knob),
successful. Take the GPU lock (`flock /tmp/vla_gpu.lock`); rendering uses EGL.

WHY EVERY OBJECT IS TRACKED. The trace's `_gt_object_pos` holds only the BDDL
task objects (target + destination). On 5115b970e766 it reads "nothing moved"
while the arm had grasped and lifted the ramekin by 6 cm -- a distractor the
trace never recorded. So the replay reads every `*_main` body from the simulator
itself, via `replay(on_step=...)`.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

FPS = 20
KNOBS = ("language", "view", "initstate", "noise", "light", "table", "add")
_KNOB_RE = re.compile(r"_(%s)_([0-9_]+?)(?=_(?:%s)_|$)" % ("|".join(KNOBS), "|".join(KNOBS)))
CANON = {"view": "0_0_100_0_0", "initstate": "0"}


# --- loading and matching ----------------------------------------------------

def load(spec: str) -> dict:
    run, rid = spec.split(":", 1)
    path = os.path.join(ROOT, "runs", run, "rollouts.jsonl")
    with open(path) as fh:
        for line in fh:
            if f'"rollout_id":"{rid}"' in line[:80] or f'"rollout_id": "{rid}"' in line[:80]:
                d = json.loads(line)
                d["_run"] = run
                return d
    raise SystemExit(f"{rid} not in {path}")


def variant(d: dict) -> str:
    return ((d.get("scene_descriptor") or {}).get("libero_plus") or {}).get("variant", "")


def knobs(d: dict) -> dict:
    """LIBERO-Plus variant suffix -> {knob: value}. Absent view/initstate are canonical."""
    k = dict(CANON)
    k.update({m.group(1): m.group(2) for m in _KNOB_RE.finditer(variant(d))})
    return k


def base_scene(d: dict) -> str:
    v = variant(d)
    cut = min([m.start() for m in _KNOB_RE.finditer(v)] or [len(v)])
    return v[:cut]


def identity(d: dict) -> dict:
    """What a fair comparison must hold fixed, beside the variant knobs."""
    fp = d.get("fingerprint") or {}
    pol, env = fp.get("policy") or {}, fp.get("env") or {}
    return {"base scene": base_scene(d),
            "instruction given": d.get("instruction"),
            "checkpoint": pol.get("checkpoint"), "dtype": pol.get("dtype"),
            "obs size": env.get("obs_size")}


def differences(ref: dict, d: dict) -> dict:
    kr, kd = knobs(ref), knobs(d)
    out = {k: (kr.get(k, "-"), kd.get(k, "-")) for k in KNOBS
           if kr.get(k) != kd.get(k)}
    ir, idd = identity(ref), identity(d)
    if ir["instruction given"] == idd["instruction given"]:
        # the variant NAME's language token is not what the policy got: control
        # runs pass --base-instruction. Only the text actually sent counts.
        out.pop("language", None)
    out.update({k: (ir[k], idd[k]) for k in ir if ir[k] != idd[k]})
    return out


def is_unperturbed(d: dict) -> bool:
    """Canonical camera and init state, and no noise/light/table/add knob.

    A `language_N` token is allowed: our control runs pass --base-instruction,
    so the policy was given the base text whatever the variant name says --
    which is checked separately against the instruction actually sent.
    """
    k = knobs(d)
    return (k.get("view") == CANON["view"] and k.get("initstate") == CANON["initstate"]
            and not any(n in k for n in ("noise", "light", "table", "add")))


def find_reference(d: dict) -> dict:
    """Unperturbed, successful, same scene / checkpoint / dtype -- else refuse."""
    want = identity(d)
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "rollouts.jsonl"))):
        with open(path) as fh:
            for line in fh:
                if want["base scene"] not in line:
                    continue
                c = json.loads(line)
                ic = identity(c)
                if (c.get("success") and is_unperturbed(c)
                        and all(ic[k] == want[k] for k in
                                ("base scene", "checkpoint", "dtype", "obs size"))):
                    c["_run"] = os.path.basename(os.path.dirname(path))
                    return c
    raise SystemExit("no unperturbed successful reference found; pass --reference RUN:ID")


# --- replay ------------------------------------------------------------------

def replay_all(d: dict):
    """Frames, fidelity, and per-step simulator truth for EVERY object body."""
    from vla_harness.schema import rollout_from_dict
    from visualise_episode_video import replay

    r = rollout_from_dict({k: v for k, v in d.items() if not k.startswith("_")})
    per_step: list[dict] = []

    def on_step(env):
        sim = env._sim()
        m = sim.model
        pos = {}
        for i in range(m.nbody):
            n = m.body_id2name(i)
            if n and n.endswith("_main"):
                pos[n] = [float(x) for x in sim.data.body_xpos[i]]
        per_step.append({"obj": pos,
                         "eef": [float(x) for x in env._raw["robot_state"]["eef"]["pos"]],
                         "grip": float(env._raw["robot_state"]["gripper"]["qpos"][0]
                                       - env._raw["robot_state"]["gripper"]["qpos"][1])})

    cams, fidelity = replay(r, on_step=on_step)
    return r, cams, fidelity, per_step


def signals(d: dict, per_step: list[dict]) -> dict:
    target = next(iter(d["steps"][0]["obs_state"]["_gt_object_pos"]))
    p0 = per_step[0]["obj"]
    dist, grip, lift, other = [], [], [], []
    summary = {n: {"moved": 0.0, "lift": 0.0, "closest": 9.0, "closest_t": 0} for n in p0}
    for t, s in enumerate(per_step):
        dist.append(math.dist(s["eef"], s["obj"][target]) * 100)
        grip.append(s["grip"] * 1000)
        lift.append((s["obj"][target][2] - p0[target][2]) * 100)
        worst = 0.0
        for n, p in s["obj"].items():
            mv = math.dist(p, p0[n])
            a = summary[n]
            a["moved"] = max(a["moved"], mv)
            a["lift"] = max(a["lift"], p[2] - p0[n][2])
            e = math.dist(p, s["eef"])
            if e < a["closest"]:
                a["closest"], a["closest_t"] = e, t
            if n != target:
                worst = max(worst, mv)
        other.append(worst * 100)
    ct = min(range(len(dist)), key=dist.__getitem__)
    objs = sorted(({"name": n.replace("_main", ""), "target": n == target,
                    "moved_cm": round(a["moved"] * 100, 1), "lift_cm": round(a["lift"] * 100, 1),
                    "closest_cm": round(a["closest"] * 100, 1), "closest_t": a["closest_t"]}
                   for n, a in summary.items()), key=lambda o: (not o["target"], -o["moved_cm"]))
    r1 = lambda xs: [round(x, 2) for x in xs]
    return {"target": target.replace("_main", ""), "dist": r1(dist), "grip": r1(grip),
            "lift": r1(lift), "other": r1(other), "objects": objs,
            "closest": {"t": ct, "cm": round(dist[ct], 1)}}


def capture(d: dict) -> dict | None:
    """Stored per-forward activations for this episode, if its run captured them.

    None when there is no sidecar -- activations cannot be recovered from a
    replay (the policy does not run), so this never fabricates them.
    """
    import numpy as np
    cdir = os.path.join(ROOT, "runs", d["_run"], "capture")
    npz = os.path.join(cdir, d["rollout_id"] + ".npz")
    if not os.path.exists(npz):
        return None
    steps = None
    with open(os.path.join(cdir, "capture_manifest.jsonl")) as fh:
        for line in fh:
            row = json.loads(line)
            if row["rollout_id"] == d["rollout_id"]:
                steps = row.get("env_step")
    z = np.load(npz)
    f = lambda k: z[k].astype(np.float32)
    n = f("vl_encoder_mean").shape[0]
    return {"step": list(steps) if steps else [i * 16 for i in range(n)],
            "enc": f("vl_encoder_mean"), "adp": f("vl_adapted_mean"),
            "spread": [round(float(x), 4) for x in
                       np.abs(f("resample_spread")).mean(axis=(1, 2))]
            if "resample_spread" in z.files else None}


def nn_dist(q, ref, self_ref: bool) -> list[float]:
    """Distance from each query forward to its NEAREST reference forward.

    For the reference itself the same forward is excluded (leave-one-out);
    otherwise every distance is 0 by construction -- the same-run trap that
    once produced a 4.9e7x "separation" (HANDOFF advice #4).
    """
    import numpy as np
    d = np.linalg.norm(q[:, None, :] - ref[None, :, :], axis=-1)
    if self_ref:
        np.fill_diagonal(d, np.inf)
    return [round(float(x), 3) for x in d.min(axis=1)]


def encode(cams) -> str:
    import numpy as np
    import imageio.v2 as imageio
    n = min(len(c) for c in cams)
    tmp = tempfile.mktemp(suffix=".mp4")
    # -g 10: a keyframe every 0.5 s. The default GOP (~250 frames) made every
    # seek decode up to 12 s of video, x every tile -- scrubbing stuttered.
    w = imageio.get_writer(tmp, fps=FPS, codec="libx264", macro_block_size=1, quality=7,
                           ffmpeg_params=["-g", "10"])
    for t in range(n):
        w.append_data(np.hstack([c[t] for c in cams]))
    w.close()
    b = base64.b64encode(open(tmp, "rb").read()).decode()
    os.remove(tmp)
    return b


# --- page --------------------------------------------------------------------

NICE = {"initstate": "init state", "view": "camera", "language": "instruction",
        "noise": "noise", "light": "light", "table": "table", "add": "added objects"}


def label(diffs: dict, ref: bool, category: str | None) -> str:
    """Name an episode by WHAT DIFFERS from the reference, not by its category:
    three 'Robot Initial States L5' tiles are indistinguishable; 'init state
    475' / '395' / '225' are not."""
    if ref:
        return "reference"
    if not diffs:
        return category or "perturbed"
    return " · ".join(f"{NICE.get(k, k)} {b}" for k, (a, b) in diffs.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="*", help="RUN:ROLLOUT_ID, one or more")
    ap.add_argument("--reference", help="RUN:ROLLOUT_ID; found automatically if omitted")
    ap.add_argument("--out")
    ap.add_argument("--rerender", metavar="PAGE",
                    help="rebuild an existing page with the current template and "
                         "encoder from its own embedded data -- CPU only, no replay")
    ap.add_argument("--reencode", action="store_true",
                    help="with --rerender: also re-encode the videos (lossy; only "
                         "needed for pages built before the -g 10 encoder)")
    a = ap.parse_args()
    if a.rerender:
        return rerender(a.rerender, a.out or a.rerender, a.reencode)
    if not a.episodes:
        ap.error("give at least one RUN:ROLLOUT_ID, or --rerender PAGE")

    eps = [load(s) for s in a.episodes]
    ref = load(a.reference) if a.reference else find_reference(eps[0])
    out = a.out or os.path.join(ROOT, "viz", "set_" + "_".join(
        d["rollout_id"][:6] for d in eps) + ".html")

    rows, caps = [], []
    for i, d in enumerate([ref] + eps):
        is_ref = i == 0
        print(f"replaying {d['_run']}:{d['rollout_id']} ...", flush=True)
        r, cams, fid, per_step = replay_all(d)
        if fid.get("max_dev_m") is None or fid["max_dev_m"] > 1e-3:
            print(f"  WARNING: replay diverged from the trace: {fid}", flush=True)
        lp = (d.get("scene_descriptor") or {}).get("libero_plus") or {}
        rows.append({"id": d["rollout_id"], "run": d["_run"], "ref": is_ref,
                     "success": bool(d.get("success")),
                     "n": len(d["steps"]), "category": lp.get("category"),
                     "level": lp.get("difficulty_level"),
                     "diffs": {} if is_ref else {k: list(v) for k, v in differences(ref, d).items()},
                     "fidelity": fid, "video": encode(cams), **signals(d, per_step)})
        rows[-1]["label"] = label(rows[-1]["diffs"], is_ref, rows[-1]["category"])
        caps.append(capture(d))

    # activations: only where the reference itself was captured
    cref = caps[0]
    for row, c in zip(rows, caps):
        row["act"] = None if (c is None or cref is None) else {
            "step": c["step"], "spread": c["spread"],
            "enc": nn_dist(c["enc"], cref["enc"], row["ref"]),
            "adp": nn_dist(c["adp"], cref["adp"], row["ref"])}

    write(out, rows, ref.get("instruction", ""), base_scene(ref), len(eps))


def write(out, rows, instr, scene, n_eps):
    html = PAGE.replace("__DATA__", json.dumps(rows)).replace(
        "__INSTR__", instr).replace("__SCENE__", scene).replace("__N__", str(n_eps))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(html)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


def rerender(page: str, out: str, reencode: bool = False):
    import imageio.v3 as iio
    src = open(page).read()
    rows = json.loads(re.search(r"const D=(\[.*?\]);\nconst col", src, re.S).group(1))
    instr = re.search(r'class="instr">&ldquo;(.*?)&rdquo;', src).group(1)
    scene = re.search(r"base scene: (\S+)</span>", src).group(1)
    for r in rows if reencode else []:
        tmp = tempfile.mktemp(suffix=".mp4")
        open(tmp, "wb").write(base64.b64decode(r["video"]))
        frames = list(iio.imread(tmp, plugin="pyav"))
        os.remove(tmp)
        r["video"] = encode([frames])
    for r in rows:                              # pages built with older labels
        r["label"] = label(r["diffs"], r["ref"], r.get("category"))
    write(out, rows, instr, scene, len(rows) - 1)


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Episode set comparison</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{color-scheme:light;--bg:#f4f6f9;--sf:#fff;--sf2:#eef1f6;--ink:#12161d;--ink2:#555f6e;--ink3:#8a93a3;
--rule:#dfe4ec;--rule2:#c3ccda;--acc:#2a78d6;--crit:#e34948;--good:#1baf7a;
--ref:#52514e;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#4a3aa7;
--fd:"IBM Plex Sans Condensed",system-ui,sans-serif;--fb:"IBM Plex Sans",system-ui,sans-serif;--fm:"IBM Plex Mono",monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--bg:#0e1116;--sf:#161b23;--sf2:#1d232c;--ink:#eef1f6;--ink2:#a3adbd;--ink3:#6d7787;--rule:#262d38;--rule2:#39424f;
--acc:#3987e5;--crit:#e66767;--good:#199e70;
--ref:#c3c2b7;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9}}
:root[data-theme="dark"]{color-scheme:dark;
--bg:#0e1116;--sf:#161b23;--sf2:#1d232c;--ink:#eef1f6;--ink2:#a3adbd;--ink3:#6d7787;--rule:#262d38;--rule2:#39424f;
--acc:#3987e5;--crit:#e66767;--good:#199e70;
--ref:#c3c2b7;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--fb);font-size:14px;line-height:1.55}
.w{max-width:1280px;margin:0 auto;padding:30px 16px 70px}
h1{font-family:var(--fd);font-size:27px;margin:0;letter-spacing:-.01em}
h2{font-family:var(--fd);font-size:16px;margin:0 0 3px}
.eyebrow{font-family:var(--fm);font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3)}
.mono{font-family:var(--fm);font-variant-numeric:tabular-nums}
header{border-bottom:1px solid var(--rule);padding-bottom:16px;display:flex;flex-direction:column;gap:7px}
.instr{font-size:16px;color:var(--ink2);font-style:italic;margin:0}
section{margin-top:22px}
.panel{background:var(--sf);border:1px solid var(--rule);padding:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,400px),1fr));gap:12px}
.tile{background:var(--sf);border:1px solid var(--rule);border-top:3px solid var(--c);padding:10px}
.tile h3{font-family:var(--fd);font-size:15px;margin:0;display:flex;align-items:center;gap:8px}
.sw{width:14px;height:3px;border-radius:2px;background:var(--c);flex:none}
.out{font-family:var(--fm);font-size:11px;margin-left:auto;color:var(--ink2)}
.sub{font-family:var(--fm);font-size:11px;color:var(--ink3);margin:2px 0 8px;overflow-wrap:anywhere}
.diff{font-family:var(--fm);font-size:11.5px;color:var(--ink2);margin:0 0 8px}
.diff b{color:var(--ink)}
video{width:100%;background:#000;display:block}
.vw{position:relative}
.ended{position:absolute;inset:auto 0 0 0;font-family:var(--fm);font-size:11px;color:#fff;background:rgba(0,0,0,.6);padding:3px 8px}
.ctrl{display:flex;align-items:center;gap:12px;flex-wrap:wrap;position:sticky;top:0;z-index:5;
background:var(--bg);padding:10px 0;border-bottom:1px solid var(--rule)}
button{font-family:var(--fm);font-size:12px;padding:5px 12px;border:1px solid var(--rule2);background:var(--sf2);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--acc)}
input[type=range]{flex:1;min-width:160px}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:4px 0 8px;font-size:12.5px;color:var(--ink2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
svg{display:block;width:100%;height:auto}
svg text{fill:var(--ink3);font-family:var(--fm);font-size:10px}
.grid line{stroke:var(--rule)}
.cursor{stroke:var(--ink);stroke-width:1}
.tip{position:fixed;pointer-events:none;background:var(--sf);border:1px solid var(--rule2);padding:6px 9px;
font-family:var(--fm);font-size:11px;color:var(--ink);display:none;z-index:9;box-shadow:0 2px 8px rgba(0,0,0,.15)}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-family:var(--fm);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);font-weight:500}
td.n{font-family:var(--fm);text-align:right;font-variant-numeric:tabular-nums}
tr.hot td{background:color-mix(in srgb,var(--crit) 16%,transparent);color:var(--ink)}
tr.hot td:first-child{box-shadow:inset 3px 0 0 var(--crit)}
.badge{font-family:var(--fm);font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:#fff;background:var(--crit);padding:1px 6px;margin-left:6px;white-space:nowrap}
.scroll{overflow-x:auto}
.note{font-size:12.5px;color:var(--ink2);margin:8px 0 0}
figcaption{font-family:var(--fm);font-size:11px;color:var(--ink3);margin-top:6px}
.pipe{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:10px 0 14px;font-size:11px;color:var(--ink2)}
.pipe span,.pipe b{border:1px solid var(--rule2);padding:4px 8px;background:var(--sf2);line-height:1.3}
.pipe b{border-color:var(--acc);color:var(--ink);font-weight:600}
.pipe i{font-style:normal;color:var(--ink3)}
.pipe small{color:var(--ink3);font-size:10px}
</style></head><body><div class="w">
<header>
  <span class="eyebrow">Episode set &middot; reference + __N__ variant(s) &middot; replayed from stored actions</span>
  <h1>Same scene, different conditions</h1>
  <p class="instr">&ldquo;__INSTR__&rdquo;</p>
  <span class="sub">base scene: __SCENE__</span>
</header>

<div class="ctrl">
  <button id="play">play</button><button id="bk">&larr;</button><button id="fw">&rarr;</button>
  <input type="range" id="scrub" min="0" value="0" step="1" aria-label="env step">
  <span class="mono" id="tread">t=0</span>
</div>

<section><div class="tiles" id="tiles"></div>
<p class="note">Each tile: agent view (left) and wrist camera (right), rotated to the policy's view. A
shorter episode holds its last frame once it has ended.</p></section>

<section class="panel"><h2>Signals over time</h2>
  <p class="eyebrow">hover for values &middot; click to seek every video there</p>
  <div class="legend" id="lg"></div>
  <div id="plots"></div>
  <figcaption>Distances are to body origins, not surfaces. "Other objects moved" is the largest
  displacement of any non-target object, read from the simulator for EVERY body, not just the task objects
  in the trace.</figcaption>
</section>

<section class="panel" id="actsec" hidden><h2>Inside the policy</h2>
  <p class="eyebrow">stored activations, one point per policy call (every 16 steps) &middot; hover for values</p>
  <div class="pipe mono">
    <span>agent + wrist images, instruction</span><i>&rarr;</i>
    <span>Qwen3-VL: vision encoder + LM layers 1&ndash;16<br><small>(16 of 28 kept; select_layer=16)</small></span><i>&rarr;</i>
    <b>&#9312; vl_encoder</b><i>&rarr;</i>
    <span>vlln (LayerNorm)</span><i>&rarr;</i>
    <span>VL self-attention<br><small>4 blocks</small></span><i>&rarr;</i>
    <b>&#9313; vl_adapted</b><i>&rarr;</i>
    <span>DiT action head<br><small>32 blocks &times; 4 denoise steps</small></span><i>&rarr;</i>
    <b>&#9314; action chunk</b>
  </div>
  <div id="actplots"></div>
  <figcaption id="actnote"></figcaption>
</section>

<section class="panel"><h2>Per episode</h2><div class="scroll"><table id="sum"></table></div></section>
<section class="panel"><h2>Every object, every episode</h2>
  <p class="eyebrow">red rows: a non-target object moved more than 1 cm &mdash; the arm handled the wrong object</p>
  <div class="scroll"><table id="obj"></table></div></section>
<div class="tip" id="tip"></div>
</div>
<script>
const D=__DATA__;
const col=(i)=>i===0?'var(--ref)':`var(--s${((i-1)%7)+1})`;
const N=Math.max(...D.map(d=>d.n));
const $=(id)=>document.getElementById(id);
const esc=(s)=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let t=0;
const scrub=$('scrub');scrub.max=N-1;

// tiles; videos go through Blob URLs so seeking does not re-parse a data: URI
const vids=[];
D.forEach((d,i)=>{
  const bin=atob(d.video),u=new Uint8Array(bin.length);for(let k=0;k<bin.length;k++)u[k]=bin.charCodeAt(k);
  const url=URL.createObjectURL(new Blob([u],{type:'video/mp4'}));
  const diffs=Object.entries(d.diffs).map(([k,[a,b]])=>`<b>${esc(k)}</b> ${esc(a)} &rarr; ${esc(b)}`).join(' &middot; ');
  const el=document.createElement('div');el.className='tile';el.style.setProperty('--c',col(i));
  el.innerHTML=`<h3><span class="sw"></span>${esc(d.label)}<span class="out">${d.success?'&#10003; success':'&#10007; failed'} &middot; ${d.n} steps</span></h3>
  <div class="sub">${d.ref?'':esc(d.category||'')+(d.level?' L'+d.level:'')+' &middot; '}${esc(d.run)}:${esc(d.id)}</div>
  <p class="diff">${d.ref?'the unperturbed baseline':(diffs||'no recorded difference from the reference')}</p>
  <div class="vw"><video muted playsinline preload="auto" src="${url}"></video><span class="ended" hidden></span></div>`;
  $('tiles').appendChild(el);vids.push([el.querySelector('video'),el.querySelector('.ended'),d]);
});
function show(nt){  // move cursor + labels, no video seeking
  t=Math.max(0,Math.min(N-1,nt));scrub.value=t;$('tread').textContent='t='+t;
  vids.forEach(([v,e,d])=>{e.hidden=t<d.n;e.textContent=`ended at t=${d.n-1}: ${d.success?'success':'failed'}`;});
  cursors.forEach(c=>{const x=X(t);c.setAttribute('x1',x);c.setAttribute('x2',x);});
}
function seek(nt){  // explicit jump: scrub, step, click
  show(nt);vids.forEach(([v,e,d])=>{v.currentTime=(Math.min(t,d.n-1)+0.5)/20;});
}
// Playback is NATIVE: every video plays itself and the longest one drives the
// cursor. Seeking every tick (the first version) forced a keyframe decode per
// video 20x a second, which is what stalled playback.
const lead=vids.reduce((a,b)=>b[2].n>a[2].n?b:a)[0];
let raf=null;
function tick(){
  const lt=lead.currentTime;show(Math.floor(lt*20));
  vids.forEach(([v,e,d])=>{if(v===lead)return;
    const want=Math.min(lt,(d.n-0.5)/20);
    if(lt<d.n/20){if(v.paused)v.play().catch(()=>{});
      if(Math.abs(v.currentTime-want)>0.12)v.currentTime=want;}
    else if(!v.paused)v.pause();});
  if(lead.ended||lead.paused){stop();return;}
  raf=requestAnimationFrame(tick);
}
function stop(){if(raf)cancelAnimationFrame(raf);raf=null;vids.forEach(([v])=>v.pause());$('play').textContent='play';}
scrub.oninput=()=>{stop();seek(+scrub.value);};
$('bk').onclick=()=>{stop();seek(t-1);};$('fw').onclick=()=>{stop();seek(t+1);};
$('play').onclick=()=>{if(raf){stop();return;}
  if(t>=N-1)seek(0);$('play').textContent='pause';
  vids.forEach(([v,e,d])=>{if(t<d.n)v.play().catch(()=>{});});raf=requestAnimationFrame(tick);};

// legend (always present for >=2 series; identity is never colour alone -- labels carry it)
$('lg').innerHTML=D.map((d,i)=>`<span><span class="sw" style="--c:${col(i)};background:${col(i)}"></span>${esc(d.label)} <span class="mono">${esc(d.id.slice(0,6))}</span></span>`).join('');

// small multiples sharing the step axis; one y-scale each
const W=1200,H=164,L=52,R=12,T=24,B=22;
const X=(s)=>L+(W-L-R)*s/Math.max(1,N-1);
const P=[['dist','distance to target (cm)'],['grip','gripper opening (mm)'],['lift','target height change (cm)'],['other','other objects moved (cm)']];
const cursors=[];
P.forEach(([key,title])=>{
  let lo=0,hi=0;D.forEach(d=>d[key].forEach(v=>{lo=Math.min(lo,v);hi=Math.max(hi,v);}));if(hi-lo<1)hi=lo+1;
  const Y=(v)=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}"><text x="${L}" y="4" dominant-baseline="hanging" style="fill:var(--ink2)">${title}</text><g class="grid">`;
  for(let k=0;k<=3;k++){const v=lo+(hi-lo)*k/3,y=Y(v);s+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}"/><text x="${L-6}" y="${y}" text-anchor="end" dominant-baseline="middle">${v.toFixed(1)}</text>`;}
  for(let k=0;k<=N;k+=40)s+=`<text x="${X(Math.min(k,N-1))}" y="${H-6}" text-anchor="middle">${k}</text>`;
  s+='</g>';
  D.forEach((d,i)=>{s+=`<polyline fill="none" stroke="${col(i)}" stroke-width="2" stroke-linejoin="round" points="${d[key].map((v,k)=>X(k).toFixed(1)+','+Y(v).toFixed(1)).join(' ')}"/>`;});
  s+=`<line class="cursor" y1="${T}" y2="${H-B}" x1="${L}" x2="${L}"/><rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent" data-k="${key}"/></svg>`;
  const div=document.createElement('div');div.innerHTML=s;$('plots').appendChild(div);
  const svg=div.firstChild;cursors.push(svg.querySelector('.cursor'));
  const hit=svg.querySelector('rect'),tip=$('tip');
  const stepAt=(ev)=>{const r=svg.getBoundingClientRect();const x=(ev.clientX-r.left)*W/r.width;return Math.round((x-L)/(W-L-R)*(N-1));};
  hit.onmousemove=(ev)=>{const k=Math.max(0,Math.min(N-1,stepAt(ev)));tip.style.display='block';
    tip.style.left=(ev.clientX+14)+'px';tip.style.top=(ev.clientY+10)+'px';
    tip.innerHTML=`<b>t=${k}</b> &middot; ${title}<br>`+D.map((d,i)=>`<span style="color:${col(i)}">&#9632;</span> ${esc(d.label)}: ${k<d.n?d[key][k].toFixed(1):'(ended)'}`).join('<br>');};
  hit.onmouseleave=()=>{tip.style.display='none';};
  hit.onclick=(ev)=>{stop();seek(stepAt(ev));};
});

// activations: point series at the policy-call steps, same x axis and cursor
const AP=[['enc','\u2460 backbone output \u2014 Qwen3-VL LM layer 16 (last kept), before vlln \u00b7 distance to nearest reference call'],
          ['adp','\u2461 action-head input \u2014 after vlln + 4 VL self-attention blocks \u00b7 distance to nearest reference call'],
          ['spread','\u2462 action-head output \u2014 std of the 40-step action chunk over 4 DiT draws']];
if(D.some(d=>d.act)){
  $('actsec').hidden=false;
  const missing=D.filter(d=>!d.act).map(d=>d.label);
  $('actnote').innerHTML='Distances are Euclidean between token-averaged vectors, to the nearest of the reference episode&rsquo;s own calls; '+
   'the reference line excludes each call&rsquo;s match with itself. Taps: &#9312; hook on the last kept LM layer (groot_n1_7.py:434), '+
   '&#9313; the tensor the DiT cross-attends to (groot_n1_7.py:563-564), &#9314; spread of action_pred over 4 draws. '+
   'Per-DiT-block activations are not stored in these runs. R-037 found camera perturbations visible at &#9312; and gone by &#9313;.'+
   (missing.length?` <b>No stored activations for:</b> ${missing.map(esc).join(', ')}.`:'');
  AP.forEach(([key,title])=>{
    const S=D.map((d,i)=>[i,d.act&&d.act[key]?d.act.step.map((x,k)=>[x,d.act[key][k]]):[]]);
    let lo=Infinity,hi=-Infinity;S.forEach(([i,pts])=>pts.forEach(([x,y])=>{lo=Math.min(lo,y);hi=Math.max(hi,y);}));
    if(!isFinite(lo))return;lo=Math.min(0,lo);if(hi-lo<1e-6)hi=lo+1;
    const Y=(v)=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
    let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}"><text x="${L}" y="4" dominant-baseline="hanging" style="fill:var(--ink2)">${title}</text><g class="grid">`;
    for(let k=0;k<=3;k++){const v=lo+(hi-lo)*k/3,y=Y(v);s+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}"/><text x="${L-6}" y="${y}" text-anchor="end" dominant-baseline="middle">${v.toFixed(v<10?2:0)}</text>`;}
    for(let k=0;k<=N;k+=40)s+=`<text x="${X(Math.min(k,N-1))}" y="${H-6}" text-anchor="middle">${k}</text>`;
    s+='</g>';
    S.forEach(([i,pts])=>{if(!pts.length)return;
      s+=`<polyline fill="none" stroke="${col(i)}" stroke-width="2" points="${pts.map(([x,y])=>X(x).toFixed(1)+','+Y(y).toFixed(1)).join(' ')}"/>`;
      pts.forEach(([x,y])=>{s+=`<circle cx="${X(x).toFixed(1)}" cy="${Y(y).toFixed(1)}" r="4" fill="${col(i)}" stroke="var(--sf)" stroke-width="2"/>`;});});
    s+=`<line class="cursor" y1="${T}" y2="${H-B}" x1="${L}" x2="${L}"/><rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent"/></svg>`;
    const div=document.createElement('div');div.innerHTML=s;$('actplots').appendChild(div);
    const svg=div.firstChild;cursors.push(svg.querySelector('.cursor'));
    const hit=svg.querySelector('rect'),tip=$('tip');
    const stepAt=(ev)=>{const r=svg.getBoundingClientRect();const x=(ev.clientX-r.left)*W/r.width;return Math.round((x-L)/(W-L-R)*(N-1));};
    hit.onmousemove=(ev)=>{const k=Math.max(0,Math.min(N-1,stepAt(ev)));tip.style.display='block';
      tip.style.left=(ev.clientX+14)+'px';tip.style.top=(ev.clientY+10)+'px';
      tip.innerHTML=`<b>t=${k}</b> &middot; ${title}<br>`+S.map(([i,pts])=>{const d=D[i];
        if(!pts.length)return `<span style="color:${col(i)}">&#9632;</span> ${esc(d.label)}: (not captured)`;
        const p=pts.filter(([x])=>x<=k).pop();
        return `<span style="color:${col(i)}">&#9632;</span> ${esc(d.label)}: ${p?p[1].toFixed(3)+' (call @ t='+p[0]+')':'&mdash;'}`;}).join('<br>');};
    hit.onmouseleave=()=>{tip.style.display='none';};
    hit.onclick=(ev)=>{stop();seek(stepAt(ev));};
  });
}

// tables -- the non-visual view of everything plotted
$('sum').innerHTML='<tr><th>episode</th><th>outcome</th><th>steps</th><th>differs from reference</th><th>closest to target</th><th>replay fidelity</th></tr>'+
 D.map((d,i)=>`<tr><td><span class="sw" style="display:inline-block;background:${col(i)}"></span> ${esc(d.label)}<br><span class="mono" style="font-size:11px;color:var(--ink3)">${esc(d.run)}:${esc(d.id)}</span></td>
 <td>${d.success?'&#10003; success':'&#10007; failed'}</td><td class="n">${d.n}</td>
 <td class="mono" style="font-size:11.5px">${d.ref?'&mdash;':Object.entries(d.diffs).map(([k,[a,b]])=>`${esc(k)}: ${esc(a)} &rarr; ${esc(b)}`).join('<br>')}</td>
 <td class="n">${d.closest.cm} cm @ t=${d.closest.t}</td>
 <td class="n">${d.fidelity.max_dev_m===null?'n/a':(d.fidelity.max_dev_m*1000).toFixed(2)+' mm max'}</td></tr>`).join('');
$('obj').innerHTML='<tr><th>episode</th><th>object</th><th>moved (cm)</th><th>lifted (cm)</th><th>closest gripper approach</th></tr>'+
 D.flatMap((d,i)=>d.objects.map(o=>`<tr class="${!o.target&&o.moved_cm>1?'hot':''}"><td>${esc(d.label)} <span class="mono" style="color:var(--ink3)">${esc(d.id.slice(0,6))}</span></td>
 <td>${esc(o.name)}${o.target?' <span class="eyebrow">target</span>':''}${!o.target&&o.moved_cm>1?'<span class="badge">&#9888; wrong object moved</span>':''}</td><td class="n">${o.moved_cm}</td><td class="n">${o.lift_cm}</td>
 <td class="n">${o.closest_cm} cm @ t=${o.closest_t}</td></tr>`)).join('');
seek(0);
</script></body></html>
"""

if __name__ == "__main__":
    main()
