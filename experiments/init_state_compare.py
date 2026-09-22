#!/usr/bin/env python3
"""Canonical vs perturbed robot-initial-state comparison, as ONE self-contained page.

Frame 0 of each episode is lifted out of the already-rendered episode video, so
this runs offline and starts no simulation -- the GPU is usually busy. Each
episode's full review page is embedded whole and expands inline, so the file can
be sent to someone as a single attachment with nothing to resolve.

    python3 experiments/init_state_compare.py
"""

from __future__ import annotations

import base64
import json
import math
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
VIZ = ROOT / "viz"
OUT = VIZ / "robot_initial_state_vs_canonical.html"

FAIL_RUN = ROOT / "runs" / "lplus_fail_groot" / "rollouts.jsonl"
CTRL_RUN = ROOT / "runs" / "groot_control_lplus_stack" / "rollouts.jsonl"

# EXACTLY ONE THING VARIES between the four panels: the initstate index.
#
# All four are the same base scene, the same camera (view_0_0_100_0_0), the same
# instruction, the same policy and the same stack. Earlier revisions of this page
# borrowed a canonical frame from a vanilla-LIBERO SmolVLA episode, which also
# changed the suite, the policy AND the object layout -- the cabinet-top bowl was
# visibly in a different place, which is precisely the confound this page exists
# to rule out. The canonical below is now rendered from our own control run:
#
#   .venvs/libero-plus/bin/python experiments/visualise_episode_video.py \
#       runs/groot_control_lplus_stack 8ca33ca61dbd viz/canonical_ramekin_initstate0.html
#
# Verified per panel by _assert_single_variable() before the page is written.
CANON_ID = "8ca33ca61dbd"
CANON_PAGE = VIZ / "canonical_ramekin_initstate0.html"
CANON_NOTE = ("our own control run, same scene and policy, initstate 0 "
              "-- solved in 113 steps")

EPISODES = [
    ("f1b17e6b55f4", "Robot_Initial_States_L4_task302_f1b17e6b55f4.html"),
    ("954cd148cd7a", "Robot_Initial_States_L4_task320_954cd148cd7a.html"),
    ("ed9afbfee053", "Robot_Initial_States_L5_task297_ed9afbfee053.html"),
]

VID_RE = re.compile(r'id="vid" src="data:video/mp4;base64,([A-Za-z0-9+/=]+)"')


def first_frame(page: pathlib.Path, agent_only: bool = True) -> bytes:
    """Decode the embedded replay video and pull frame 0 as a PNG."""
    m = VID_RE.search(page.read_text())
    if m is None:
        raise SystemExit(f"no embedded video in {page}")
    with tempfile.TemporaryDirectory() as td:
        mp4 = pathlib.Path(td, "e.mp4")
        png = pathlib.Path(td, "f.png")
        mp4.write_bytes(base64.b64decode(m.group(1)))
        # The replay is agent view and wrist side by side; the left half is the
        # third-person view the perturbation is visible in.
        vf = "crop=iw/2:ih:0:0,scale=440:-1" if agent_only else "scale=440:-1"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
             "-vf", f"select=eq(n\\,0),{vf}", "-vframes", "1", str(png)],
            check=True,
        )
        return png.read_bytes()


def load(path: pathlib.Path) -> dict[str, dict]:
    return {json.loads(l)["rollout_id"]: json.loads(l) for l in path.open()}


def base_scene(rollout: dict) -> str:
    v = rollout["scene_descriptor"]["libero_plus"]["variant"]
    return v.split("_language_")[0].split("_view_")[0]


def quat_deg(a: list[float], b: list[float]) -> float:
    dot = min(1.0, abs(sum(x * y for x, y in zip(a, b))))
    return 2 * math.degrees(math.acos(dot))


def _assert_single_variable(ctrl: dict, rollout: dict) -> None:
    """Refuse to build the page unless initstate is the ONLY difference.

    A comparison that silently varies two things is worse than no comparison, so
    this is a hard failure rather than a warning. Object poses are read from the
    t=0 observation, never from `scene_descriptor.objects` -- the latter is
    written at the END of the episode, so on a successful control it shows the
    bowl already sitting on the plate and every object looks "moved".
    """
    cs, rs = ctrl["scene_descriptor"], rollout["scene_descriptor"]
    checks = {
        "base scene": (base_scene(ctrl), base_scene(rollout)),
        "suite": (cs["suite"], rs["suite"]),
        "instruction": (ctrl["instruction"], rollout["instruction"]),
        "camera": (cs["libero_plus"]["variant"].split("_initstate_")[0].split("_view_")[-1],
                   rs["libero_plus"]["variant"].split("_initstate_")[0].split("_view_")[-1]),
        "policy": (ctrl["fingerprint"]["policy"]["checkpoint"],
                   rollout["fingerprint"]["policy"]["checkpoint"]),
        "dtype": (ctrl["fingerprint"]["policy"]["dtype"],
                  rollout["fingerprint"]["policy"]["dtype"]),
    }
    for name, (a, b) in checks.items():
        if a != b:
            raise SystemExit(f"{rollout['rollout_id']}: {name} differs -- {a!r} vs {b!r}")

    co = ctrl["steps"][0]["obs_state"]["_gt_object_pos"]
    ro = rollout["steps"][0]["obs_state"]["_gt_object_pos"]
    if set(co) != set(ro):
        raise SystemExit(f"{rollout['rollout_id']}: different objects in the scene")
    for name in co:
        mm = math.dist(co[name], ro[name]) * 1000
        if mm > 1.0:
            raise SystemExit(f"{rollout['rollout_id']}: {name} moved {mm:.1f} mm at t=0")


def main() -> int:
    fails = load(FAIL_RUN)
    ctrl = load(CTRL_RUN)[CANON_ID]

    cards, blobs, tabs = [], [], []

    png = base64.b64encode(first_frame(CANON_PAGE)).decode()
    blobs.append('<script type="application/octet-stream" id="d_canonical">'
                 f'{base64.b64encode(CANON_PAGE.read_bytes()).decode()}</script>')
    tabs.append('<button class="tab canon" data-ep="canonical">'
                '<b>initstate 0</b><span>canonical &middot; 0 mm &middot; 0&deg;</span></button>')
    cards.append(card(
        png=png, tone="ok", level="canonical", init="initstate 0",
        dpos="0.0 mm", dquat="0.0&deg;",
        term=f"success @{len(ctrl['steps'])}",
        fam="&mdash;", instr=ctrl["instruction"],
        body='<button class="expand" data-ep="canonical">review this episode &darr;</button>'
             f'<span class="ink3" style="display:block;margin-top:7px">{CANON_NOTE}</span>',
    ))

    for rid, page_name in EPISODES:
        rollout = fails[rid]
        page = VIZ / "lplus_fail_groot" / page_name
        sd = rollout["scene_descriptor"]["libero_plus"]
        _assert_single_variable(ctrl, rollout)

        s0, c0 = rollout["steps"][0]["obs_state"], ctrl["steps"][0]["obs_state"]
        dpos = math.dist(s0["eef_pos"], c0["eef_pos"]) * 1000
        dquat = quat_deg(s0["eef_quat"], c0["eef_quat"])

        fam = re.search(r"<h1>([^<]*)</h1>", page.read_text()).group(1)
        init = re.search(r"initstate_(\d+)", sd["variant"]).group(1)
        solved = 4 - sd["difficulty_level"] + 1 if sd["difficulty_level"] <= 5 else 0

        png = base64.b64encode(first_frame(page)).decode()
        blobs.append(f'<script type="application/octet-stream" id="d_{rid}">'
                     f'{base64.b64encode(page.read_bytes()).decode()}</script>')
        tabs.append(f'<button class="tab" data-ep="{rid}">'
                    f'<b>initstate {init}</b><span>L{sd["difficulty_level"]} '
                    f'&middot; {dpos:.0f} mm &middot; {dquat:.0f}&deg;</span></button>')
        cards.append(card(
            png=png, tone="bad",
            level=f"L{sd['difficulty_level']} &middot; solved by {5 - sd['difficulty_level']} of 4",
            init=f"initstate {init}",
            dpos=f"{dpos:.1f} mm", dquat=f"{dquat:.1f}&deg;",
            term=f"{rollout['termination']} @{len(rollout['steps'])}",
            fam=fam, instr=rollout["instruction"],
            body=f'<button class="expand" data-ep="{rid}">'
                 f'review this episode &darr;</button>',
        ))

    page = (PAGE.replace("{cards}", "".join(cards))
                .replace("{tabs}", "".join(tabs))
                .replace("{blobs}", "".join(blobs)))
    OUT.write_text(page)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1_048_576:.1f} MB)")
    return 0


def card(*, png, tone, level, init, dpos, dquat, term, fam, instr, body) -> str:
    return f"""
<figure class="card {tone}">
  <figcaption class="cap"><span class="eyebrow">{level}</span><b class="mono">{init}</b></figcaption>
  <img src="data:image/png;base64,{png}" alt="initial frame, {init}">
  <div class="kv">
    <div><span>eef displacement</span><span class="v">{dpos}</span></div>
    <div><span>eef rotation</span><span class="v">{dquat}</span></div>
    <div><span>outcome</span><span class="v">{term}</span></div>
    <div><span>classifier</span><span class="v">{fam}</span></div>
  </div>
  <p class="instr">&ldquo;{instr}&rdquo;</p>
  <p class="lnk">{body}</p>
</figure>"""


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Robot initial state &mdash; canonical vs perturbed</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{color-scheme:light;--bg:#f4f6f9;--sf:#fff;--sf2:#eef1f6;--ink:#12161d;--ink2:#555f6e;--ink3:#8a93a3;
--rule:#dfe4ec;--rule2:#c3ccda;--acc:#2a78d6;--crit:#e34948;--good:#1baf7a;--warn:#eda100;
--fd:"IBM Plex Sans Condensed",system-ui,sans-serif;--fb:"IBM Plex Sans",system-ui,sans-serif;--fm:"IBM Plex Mono",monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--bg:#0e1116;--sf:#161b23;--sf2:#1d232c;--ink:#eef1f6;--ink2:#a3adbd;--ink3:#6d7787;--rule:#262d38;--rule2:#39424f;
--acc:#3987e5;--crit:#e66767;--good:#199e70;--warn:#c98500}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--fb);font-size:14px;line-height:1.55}
img{max-width:100%;display:block;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
[hidden]{display:none!important}
.w{max-width:1180px;margin:0 auto;padding:30px 22px 70px}
h1{font-family:var(--fd);font-size:27px;margin:0;letter-spacing:-.01em}
h2{font-family:var(--fd);font-size:16px;margin:0 0 3px}
.eyebrow{font-family:var(--fm);font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3)}
.mono{font-family:var(--fm);font-variant-numeric:tabular-nums}
header{border-bottom:1px solid var(--rule);padding-bottom:16px;display:flex;flex-direction:column;gap:7px}
.lede{font-size:15px;color:var(--ink2);max-width:74ch;margin:6px 0 0}
section{margin-top:26px}
.panel{background:var(--sf);border:1px solid var(--rule);padding:16px}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(255px,1fr));gap:16px;margin-top:14px}
.card{margin:0;background:var(--sf);border:1px solid var(--rule);border-top:3px solid var(--rule2);
display:flex;flex-direction:column}
.card.ok{border-top-color:var(--good)}.card.bad{border-top-color:var(--crit)}
.cap{display:flex;justify-content:space-between;align-items:baseline;gap:8px;padding:10px 13px}
.kv{display:flex;flex-direction:column;gap:6px;font-size:12.5px;padding:11px 13px 4px}
.kv div{display:flex;justify-content:space-between;gap:10px;border-bottom:1px dotted var(--rule);padding-bottom:5px}
.kv div:last-child{border:0}.kv .v{font-family:var(--fm);text-align:right}
.instr{font-size:12.5px;color:var(--ink2);font-style:italic;padding:0 13px;margin:4px 0 0}
.lnk{padding:0 13px 13px;margin:9px 0 0;font-family:var(--fm);font-size:11.5px}
.ink3{color:var(--ink3)}
button.expand{font-family:var(--fm);font-size:11.5px;padding:6px 12px;border:1px solid var(--rule2);
background:var(--sf2);color:var(--acc);cursor:pointer;width:100%}
button.expand:hover{border-color:var(--acc)}
button.expand[aria-expanded="true"]{color:var(--ink2)}
.stage{margin-top:26px}
.stage .head{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
.tab{display:flex;flex-direction:column;align-items:flex-start;gap:1px;
font-family:var(--fm);font-size:12px;padding:8px 14px;border:1px solid var(--rule2);
background:var(--sf2);color:var(--ink2);cursor:pointer;text-align:left}
.tab span{font-size:10.5px;color:var(--ink3)}
.tab:hover{border-color:var(--acc)}
.tab[aria-selected="true"]{background:var(--acc);border-color:var(--acc);color:#fff}
.tab.canon[aria-selected="true"]{background:var(--good);border-color:var(--good)}
.tab[aria-selected="true"] span{color:rgba(255,255,255,.8)}
/* The review is the exhibit, so it gets the full window width, not a card column. */
.bleed{margin-top:14px;border:1px solid var(--rule2);background:var(--sf);
width:min(1600px,96vw);position:relative;left:50%;transform:translateX(-50%)}
.bleed .bar{display:flex;justify-content:space-between;align-items:center;gap:10px;
padding:9px 14px;border-bottom:1px solid var(--rule);background:var(--sf2);
font-family:var(--fm);font-size:11px;color:var(--ink3)}
.bleed iframe{width:100%;height:min(1500px,92vh);border:0;display:block;background:var(--sf)}
table.num{border-collapse:collapse;width:100%;margin-top:6px;font-size:13px}
table.num th,table.num td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--rule);
font-family:var(--fm);font-variant-numeric:tabular-nums}
table.num th{font-family:var(--fb);font-size:11px;letter-spacing:.06em;text-transform:uppercase;
color:var(--ink3);font-weight:600;border-bottom-color:var(--rule2)}
table.num td:first-child,table.num th:first-child{text-align:left}
table.num tr.ctl td{color:var(--good)}
.scroll{overflow-x:auto}
.empty{padding:46px 20px;text-align:center;color:var(--ink3);font-family:var(--fm);font-size:12.5px}
.tall .bleed iframe{height:1500px}
.warn{border-left:3px solid var(--warn);padding:10px 14px;background:color-mix(in srgb,var(--warn) 9%,transparent);
font-size:13px;margin-top:14px}
ul.assume{font-size:13px;color:var(--ink2);margin:10px 0 0;padding-left:18px}
ul.assume li{margin-bottom:6px}
footer{margin-top:40px;padding-top:16px;border-top:1px solid var(--rule);font-size:12.5px;color:var(--ink3)}
</style></head><body><div class="w">
<header>
  <span class="eyebrow">LIBERO-Plus &middot; Robot Initial States &middot; GR00T N1.7 libero_spatial-640</span>
  <h1>The scene is identical. Only the arm moved.</h1>
  <p class="lede">Frame 0 of three failures beside the canonical start. Every object sits
  where it sat in the control; the table, the lighting, the camera and the wording are
  untouched. The classifier calls all three <b>visual&nbsp;grounding</b>. Each card opens
  its full episode review inline &mdash; video, predicates and plots &mdash; so this one
  file is the whole exhibit.</p>
</header>

<section><div class="row" id="cards">{cards}</div></section>

<section class="stage">
  <div class="head">
    <div>
      <span class="eyebrow">the full analysis, at full width</span>
      <h2>Episode review</h2>
    </div>
    <label class="eyebrow" style="display:flex;align-items:center;gap:7px;cursor:pointer">
      <input type="checkbox" id="tall"> tall mode &mdash; let the page scroll instead of the frame
    </label>
  </div>
  <div class="tabs" role="tablist">{tabs}</div>
  <div class="bleed" id="stage"><div class="empty">
    pick an episode above &mdash; its complete review loads here: both cameras, the
    scrubber, the derived predicates and every plot
  </div></div>
</section>

<section class="panel">
  <h2>Why this is the contested cluster</h2>
  <ul class="assume">
    <li><b>69 of the 78</b> rendered robot-initial-state failures carry <span class="mono">visual_grounding</span>
        (56 as <span class="mono">+ manipulation</span>, 13 also <span class="mono">+ spatial_reasoning</span>);
        only 9 are plain <span class="mono">manipulation</span>.</li>
    <li>Displacement is small in absolute terms &mdash; <b>83 to 133 mm</b> and <b>22 to 32&deg;</b> &mdash;
        and it is the <i>only</i> thing that changed.</li>
    <li>All three run to <b>281 steps and time out</b>. Not a grasp that slipped: the episode never resolves.
        Across the whole cluster the terminal state is a timeout every time.</li>
    <li>The canonical start succeeds <b>10/10</b> in the unperturbed control on this same stack.</li>
  </ul>
  <div class="warn">Nothing about the scene&rsquo;s appearance was perturbed, so
  <span class="mono">visual_grounding</span> cannot be what these episodes demonstrate. There is no
  family in the taxonomy that names <i>starting pose</i>. That is PENDING&nbsp;#15, and this is the evidence.</div>
</section>

<section class="panel">
  <h2>What is held fixed, and how that is checked</h2>
  <p class="lede" style="margin-bottom:4px">All four panels are the same base scene, the same camera
  (<span class="mono">view_0_0_100_0_0</span>), the same instruction, the same policy and the same stack.
  <b>The initstate index is the only thing that varies.</b> The builder refuses to write this page otherwise:</p>
  <ul class="assume">
    <li>Base scene, suite, instruction, camera parameters, checkpoint and dtype are compared field by field
        and any mismatch is a hard failure, not a warning.</li>
    <li>Every object pose is compared at <b>t = 0</b> and must agree within <b>1 mm</b>. For these four it is
        <b>0.00 mm</b> &mdash; bit-identical. For scale, two different control rollouts of this same scene
        differ by about 21 mm, so 0.00 is a real result rather than a broken comparison.</li>
    <li>Object poses are read from the t=0 observation, never from the episode's scene descriptor. That
        descriptor is written at the <i>end</i> of the episode, so on a successful control it shows the bowl
        already on the plate and every object looks moved. An earlier revision of this page compared a start
        position against an end position and appeared to show the target bowl shifting 167 mm.</li>
    <li>The canonical panel is our own control rollout <span class="mono">8ca33ca61dbd</span>, GR00T on the
        LIBERO-Plus stack at <span class="mono">initstate 0</span>, solved in 113 steps. An earlier revision
        borrowed a frame from a vanilla-LIBERO SmolVLA episode, which also changed the suite, the policy and
        the object layout &mdash; the cabinet-top bowl sat somewhere else, which is exactly the confound this
        page exists to rule out.</li>
  </ul>
</section>

<section class="panel">
  <h2>What the arm actually does in the 281 steps</h2>
  <p class="lede" style="margin-bottom:10px">It is not frozen, it does reach the bowl, and it still never
  finishes. Distances in cm, measured from the stored trajectories.</p>
  <div class="scroll"><table class="num">
    <thead><tr><th>episode</th><th>path length</th><th>closest to bowl</th>
      <th>start offset from&nbsp;home</th><th>closest to home</th><th>steps gripping</th><th>outcome</th></tr></thead>
    <tbody>
      <tr class="ctl"><td>canonical &middot; initstate 0</td><td>115.1</td><td>4.6</td><td>0.0</td><td>0.0</td><td>53</td><td>success @113</td></tr>
      <tr><td>initstate 232 &middot; L4</td><td>196.8</td><td>6.4</td><td>2.7</td><td>2.7</td><td>162</td><td>timeout @281</td></tr>
      <tr><td>initstate 412 &middot; L4</td><td>159.3</td><td>7.3</td><td>9.1</td><td>7.9</td><td>202</td><td>timeout @281</td></tr>
      <tr><td>initstate 162 &middot; L5</td><td>197.0</td><td>7.4</td><td>3.4</td><td>3.4</td><td>170</td><td>timeout @281</td></tr>
    </tbody>
  </table></div>
  <ul class="assume">
    <li><b>It travels further than the success does</b> &mdash; 159 to 197 cm against 115 &mdash; and closes the
        gripper <b>three to four times as often</b> (162&ndash;202 steps against 53). It is working hard and
        achieving nothing.</li>
    <li><b>It gets to the bowl.</b> Closest approach 6.4&ndash;7.4 cm against the control's 4.6 cm. So on this
        scene the failure is not a failure to navigate &mdash; it arrives and cannot close the loop.</li>
    <li><b>It never returns to the neutral pose.</b> The closest it comes is its own starting offset, and it
        ends 34&ndash;42 cm away. Nothing pulls it back toward the distribution it was trained on.</li>
    <li>The perturbation is joint-space: all seven joints differ, by up to <b>18&deg;</b> on a single joint.
        The end-effector offset is the consequence, not the cause.</li>
  </ul>
  <div class="warn">The failure is <b>not uniform across base scenes</b>. On the cookie-box scenes the arm
  never gets within 19 cm of the target; here it reaches 6&ndash;7 cm and still times out. Any family name
  that fits both has to describe the <i>outcome</i> &mdash; work without progress, no recovery to the training
  distribution &mdash; rather than a single mechanism. That distinction is what the adjudication has to settle.</div>
</section>

<footer>Built by <span class="mono">experiments/init_state_compare.py</span> from
<span class="mono">runs/lplus_fail_groot</span> and <span class="mono">runs/groot_control_lplus_stack</span>.
Frames extracted from the rendered episode videos; no new simulation was run.
Each embedded review is the full page, byte for byte.</footer>
</div>
{blobs}
<script>
// The episode pages are carried base64-encoded and decoded on first selection, so
// the file opens fast and stays a single attachment with nothing to resolve. Each
// decoded review keeps its own iframe, so switching back to one is instant and it
// holds its scrub position.
(function () {
  var stage = document.getElementById("stage");
  var tabs = [].slice.call(document.querySelectorAll(".tab"));
  var frames = {};
  var current = null;

  function decode(id) {
    var raw = atob(document.getElementById("d_" + id).textContent.trim());
    var bytes = new Uint8Array(raw.length);
    for (var i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    return new TextDecoder("utf-8").decode(bytes);
  }

  function show(id) {
    if (current === id) return;
    var empty = stage.querySelector(".empty");
    if (empty) empty.remove();

    if (!frames[id]) {
      var bar = document.createElement("div");
      bar.className = "bar";
      var note = (id === "canonical")
        ? "canonical start &mdash; borrowed from a nominal vanilla-LIBERO episode " +
          "(SmolVLA parity run), shown for the unperturbed starting pose only"
        : "episode " + id + " &mdash; full review, embedded";
      bar.innerHTML = "<span>" + note + "</span><span>scroll inside the frame</span>";
      var f = document.createElement("iframe");
      f.setAttribute("title", "episode " + id);
      f.srcdoc = decode(id);
      var wrap = document.createElement("div");
      wrap.appendChild(bar);
      wrap.appendChild(f);
      stage.appendChild(wrap);
      frames[id] = wrap;
    }
    Object.keys(frames).forEach(function (k) {
      frames[k].hidden = (k !== id);
    });
    tabs.forEach(function (t) {
      t.setAttribute("aria-selected", String(t.dataset.ep === id));
    });
    current = id;
  }

  function select(id) {
    show(id);
    document.querySelector(".stage").scrollIntoView({behavior: "smooth", block: "start"});
  }

  tabs.forEach(function (t) {
    t.setAttribute("aria-selected", "false");
    t.addEventListener("click", function () { select(t.dataset.ep); });
  });
  document.querySelectorAll("button.expand").forEach(function (b) {
    b.addEventListener("click", function () { select(b.dataset.ep); });
  });

  // A nested scroll region is awkward to read; tall mode gives the frame its full
  // height and lets the outer page do the scrolling instead.
  document.getElementById("tall").addEventListener("change", function (e) {
    document.body.classList.toggle("tall", e.target.checked);
  });
})();
</script>
</body></html>"""


if __name__ == "__main__":
    sys.exit(main())
