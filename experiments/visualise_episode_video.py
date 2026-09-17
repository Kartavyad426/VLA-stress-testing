"""Render one rollout as a self-contained HTML page WITH VIDEO.

Sibling of `visualise_episode.py`, which is left untouched. That file plots the
signals; this one replays the episode in the simulator, renders both cameras,
and locks the video to the plots on one shared time cursor.

Why it exists: the mining layer's labels were argued about for days because the
plots showed what the classifier BELIEVED and nothing showed what HAPPENED. The
analysis worth doing lives exactly where those two disagree, so they have to be
on screen together, scrubbing in lockstep.

    MUJOCO_GL=egl .venvs/lerobot/bin/python experiments/visualise_episode_video.py \
        <run_dir> [rollout_id] [out.html]

Needs the LeRobot venv and a GPU for offscreen rendering. It does NOT run the
policy: the stored actions are replayed, so it is seconds per episode, not
minutes, and it costs no inference.

REPLAY FIDELITY. Pixels are never persisted (G2), so the video is a
RECONSTRUCTION, not a recording. Determinism is assumed (same seed, same
init-state hash, same actions) and then CHECKED: the replayed end-effector path
is compared against the stored one and the divergence is reported in the page.
A video that silently drifted from the trace it is captioned with would be worse
than no video at all.
"""
import base64, json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vla_harness.schema import rollout_from_dict, PerturbationSpec, Action
from vla_harness.mining.phases_libero import LiberoPhaseSegmenter
# Imported, NOT copied. The same expression living in two modules is exactly how
# O1 happened (IMPLEMENTATION_OVERSIGHTS.md); one source of truth for the signal
# extraction, a different HTML template on top of it.
from visualise_episode import extract

FPS = 20                      # LIBERO control frequency; playback speed only


# --- replay -----------------------------------------------------------------

def replay(r, cameras=2):
    """Re-execute the stored actions and collect frames from every camera.

    Returns (frames_per_camera, fidelity). `frames_per_camera` is a list of
    per-camera lists of HxWx3 uint8 arrays.
    """
    import numpy as np
    from vla_harness.envs.libero_env import LiberoEnv

    suite = r.fingerprint.get("env", {}).get("suite") or r.task_id.split("/")[0]
    task_id = r.fingerprint.get("env", {}).get("task_id")
    if task_id is None:
        task_id = int(r.task_id.rsplit("task", 1)[-1])

    env = LiberoEnv(suite=suite, task_id=task_id)
    spec = PerturbationSpec.of(**(r.perturbation or {}))
    env.reset(seed=r.seed, spec=spec)

    cams, eef_replay = None, []
    for st in r.steps:
        px = env._raw.get("pixels", {})
        imgs = list(px.values()) if isinstance(px, dict) else list(px)
        if cams is None:
            cams = [[] for _ in imgs]
        for i, im in enumerate(imgs):
            if i < len(cams):
                # 180-degree rotation, matching LeRobot's LiberoProcessorStep
                # (processor/env_processor.py:59, `torch.flip(img, dims=[2,3])`),
                # which is applied to every OBS_IMAGES key before the policy sees
                # it -- "accounts for the HuggingFaceVLA/libero camera
                # orientation convention".
                #
                # `env._raw["pixels"]` is the RAW observation, taken BEFORE that
                # processor runs, so it is upside down relative to the policy's
                # view. Rendering it unflipped shows a world the policy never
                # saw -- which defeats the point of putting the video next to the
                # detector flags. e2e.py makes the same note from the other
                # direction: without the processor the policy scores ~0%.
                cams[i].append(np.asarray(im, dtype=np.uint8)[::-1, ::-1])
        eef_replay.append(list(env._raw["robot_state"]["eef"]["pos"]))
        if st.action is None:
            break
        env.step(Action(values=list(st.action), dims=r.action_dims))

    # --- fidelity: does the replay follow the stored trace? -----------------
    import math
    stored = [s.obs_state.get("eef_pos") for s in r.steps]
    devs = [math.dist(a, b) for a, b in zip(eef_replay, stored)
            if a and b]
    fidelity = {"n_compared": len(devs),
                "max_dev_m": round(max(devs), 5) if devs else None,
                "median_dev_m": round(sorted(devs)[len(devs) // 2], 5) if devs else None}
    return cams or [], fidelity


def encode_video(cams, out_path):
    """Both cameras side by side in ONE file, so a single currentTime syncs all."""
    import numpy as np, imageio.v2 as imageio
    if not cams or not cams[0]:
        return None
    n = min(len(c) for c in cams)
    h = max(c[0].shape[0] for c in cams)

    def fit(img):
        if img.shape[0] == h:
            return img
        import cv2
        w = int(img.shape[1] * h / img.shape[0])
        return cv2.resize(img, (w, h))

    w = imageio.get_writer(out_path, fps=FPS, codec="libx264",
                           macro_block_size=1, quality=7)
    for t in range(n):
        w.append_data(np.hstack([fit(c[t]) for c in cams]))
    w.close()
    return out_path


# --- per-step detector state, the thing the video is here to adjudicate -----

# Empty-air closure settles near 0.006 m (median 0.0059 over 4599 commanded-close
# steps). A stall clearly above that but below closed_m is the signature of a
# THIN held object -- which by_gripper cannot see by construction (O4).
EMPTY_CLOSED_M = 0.010


def detector_trace(r):
    """Per-step detector flags. None means NOT COMPUTED, never 'no'.

    The first version returned False when the segmenter abstained, so an episode
    missing `_gt_eef_to_object` showed `by_lift: no` on a successful pick-and-place.
    That is the O6 anti-pattern again -- a detector with no data answering anyway.
    """
    seg = LiberoPhaseSegmenter()
    n = len(r.steps)
    missing = seg.missing(r)
    tgt = seg._target(r.steps[0].obs_state) if not missing else None
    g = [bool(seg._holding_by_gripper(r, i)) for i in range(n)]
    if tgt is None:
        l = [None] * n
        abstain = (f"missing state keys: {missing}" if missing
                   else "no BDDL target object in trace")
    else:
        z0 = (r.steps[0].obs_state.get("_gt_object_pos", {}).get(tgt) or [0, 0, 0])[2]
        l = [bool(seg._holding_by_lift(r, i, tgt, z0)) for i in range(n)]
        abstain = None
    holding = [True if (a or b) else (None if b is None else False)
               for a, b in zip(g, l)]

    # thin-object stall: fingers stopped, commanded closed, aperture between
    # empty-closure and closed_m -- by_gripper's blind spot, surfaced not hidden
    thin = None
    for i in range(n - seg.hold_steps):
        ap = []
        for k in range(i, i + seg.hold_steps):
            q = r.steps[k].obs_state.get("gripper_qpos") or []
            if not q or not seg._commanded_closed(r.steps[k]):
                break
            ap.append(sum(abs(x) for x in q))
        if (len(ap) == seg.hold_steps and max(ap) - min(ap) < seg.stall_m
                and EMPTY_CLOSED_M < min(ap) < seg.closed_m):
            thin = {"t": i, "aperture_m": round(min(ap), 4)}
            break
    return {"by_gripper": g, "by_lift": l, "holding": holding, "target": tgt,
            "abstain": abstain, "thin_stall": thin}


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Episode __ID__ (video)</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{color-scheme:light;--bg:#f4f6f9;--sf:#fff;--sf2:#eef1f6;--ink:#12161d;--ink2:#555f6e;--ink3:#8a93a3;
--rule:#dfe4ec;--rule2:#c3ccda;--acc:#2a78d6;--crit:#e34948;--good:#1baf7a;--warn:#eda100;
--seq-2:#9ec5f4;--seq-3:#5598e7;--seq-5:#1c5cab;
--fd:"IBM Plex Sans Condensed",system-ui,sans-serif;--fb:"IBM Plex Sans",system-ui,sans-serif;--fm:"IBM Plex Mono",monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--bg:#0e1116;--sf:#161b23;--sf2:#1d232c;--ink:#eef1f6;--ink2:#a3adbd;--ink3:#6d7787;--rule:#262d38;--rule2:#39424f;
--acc:#3987e5;--crit:#e66767;--good:#199e70;--warn:#c98500;--seq-2:#184f95;--seq-3:#256abf;--seq-5:#6da7ec}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--fb);font-size:14px;line-height:1.55}
img{max-width:100%}[hidden]{display:none!important}
.w{max-width:1180px;margin:0 auto;padding:30px 22px 70px}
h1{font-family:var(--fd);font-size:27px;margin:0;letter-spacing:-.01em}
h2{font-family:var(--fd);font-size:16px;margin:0 0 3px}
.eyebrow{font-family:var(--fm);font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3)}
.mono{font-family:var(--fm);font-variant-numeric:tabular-nums}
header{border-bottom:1px solid var(--rule);padding-bottom:16px;display:flex;flex-direction:column;gap:7px}
.instr{font-size:16px;color:var(--ink2);font-style:italic}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;font-family:var(--fm);font-size:11.5px;color:var(--ink3)}
section{margin-top:26px}
.panel{background:var(--sf);border:1px solid var(--rule);padding:16px}
.grid{display:grid;grid-template-columns:1fr 330px;gap:18px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.kv{display:flex;flex-direction:column;gap:7px;font-size:13px}
.kv div{display:flex;justify-content:space-between;gap:10px;border-bottom:1px dotted var(--rule);padding-bottom:6px}
.kv div:last-child{border:0}.kv .v{font-family:var(--fm);text-align:right}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:10px;font-size:12.5px;color:var(--ink2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:13px;height:3px;border-radius:2px;flex:none}
svg{display:block;max-width:100%}.scroll{overflow-x:auto}
figcaption{font-family:var(--fm);font-size:11.5px;color:var(--ink3);margin-top:8px;line-height:1.5}
video{width:100%;background:#000;display:block}
.vidwrap{position:relative}
.camtag{position:absolute;top:8px;font-family:var(--fm);font-size:10px;letter-spacing:.1em;
text-transform:uppercase;color:#fff;background:rgba(0,0,0,.55);padding:2px 7px}
.ctrl{display:flex;align-items:center;gap:12px;margin-top:10px;flex-wrap:wrap}
button{font-family:var(--fm);font-size:12px;padding:5px 12px;border:1px solid var(--rule2);
background:var(--sf2);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--acc)}
input[type=range]{flex:1;min-width:160px}
.flags{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
.flag{font-family:var(--fm);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
padding:3px 9px;border:1px solid var(--rule2);color:var(--ink3)}
.flag.on{color:#fff;border-color:transparent}
.flag.na{border-style:dashed;color:var(--ink3);text-decoration:line-through}
.flag.on.g{background:var(--acc)}.flag.on.l{background:var(--good)}.flag.on.h{background:var(--warn)}
.warn{border-left:3px solid var(--warn);padding:10px 14px;background:color-mix(in srgb,var(--warn) 9%,transparent);
font-size:13px;margin-top:14px}
.bad{border-left-color:var(--crit);background:color-mix(in srgb,var(--crit) 10%,transparent)}
.trust{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin-top:12px}
.tier{border:1px solid var(--rule);border-top:3px solid var(--rule2);padding:12px 14px;background:var(--sf2)}
.tier h3{font-family:var(--fd);font-size:14px;margin:0 0 2px}
.tier .tag{font-family:var(--fm);font-size:10px;letter-spacing:.1em;text-transform:uppercase}
.tier ul{margin:8px 0 0;padding-left:18px;font-size:13px;color:var(--ink2)}
.tier li{margin-bottom:5px}
.t-ok{border-top-color:var(--good)}.t-ok .tag{color:var(--good)}
.t-mid{border-top-color:var(--warn)}.t-mid .tag{color:var(--warn)}
.t-bad{border-top-color:var(--crit)}.t-bad .tag{color:var(--crit)}
.assume{font-size:13px;color:var(--ink2);margin:14px 0 0;padding-left:18px}
.assume li{margin-bottom:5px}
footer{margin-top:40px;padding-top:16px;border-top:1px solid var(--rule);font-size:12.5px;color:var(--ink3)}
</style></head><body><div class="w">
<header>
  <span class="eyebrow">Episode replay &middot; __TASKID__</span>
  <h1>__FAMILY__</h1>
  <p class="instr">&ldquo;__INSTR__&rdquo;</p>
  <div class="meta"><span>__ID__</span><span>__N__ steps</span><span>__TERM__</span>
  <span>__PERT__</span><span>__POLICY__</span></div>
</header>

<section><div class="grid">
  <div class="panel">
    <h2>What actually happened</h2>
    <p class="eyebrow" style="margin-bottom:12px">left: agent view &middot; right: wrist &middot; scrub to move every plot below</p>
    <div class="vidwrap">
      <video id="vid" __VIDATTR__ playsinline muted preload="auto"></video>
      <span class="camtag" style="left:8px">agent view</span>
      <span class="camtag" style="left:50.5%">wrist</span>
    </div>
    <div class="ctrl">
      <button id="play">play</button>
      <button id="bk">&larr; step</button>
      <button id="fw">step &rarr;</button>
      <input type="range" id="scrub" min="0" max="__NMAX__" value="0" step="1">
      <span class="mono" id="tread">t=0</span>
    </div>
    <div class="flags">
      <span class="flag g" id="f_g">by_gripper</span>
      <span class="flag l" id="f_l">by_lift</span>
      <span class="flag h" id="f_h">holding</span>
      <span class="flag" id="f_p">phase</span>
    </div>
    <figcaption>The flags are the DERIVED predicates the segmenter assigns at this
    frame. If one is lit while the picture shows an empty gripper, the detector is
    wrong at that frame &mdash; which is the whole reason this page exists.</figcaption>
  </div>
  <div>
    <div class="panel"><h2>At this frame</h2><div class="kv" id="live" style="margin-top:10px"></div></div>
    <div class="panel" style="margin-top:16px"><h2>Diagnosis</h2>
      <p class="eyebrow" style="margin-bottom:8px">for the whole episode</p>
      <div class="kv" id="diag"></div></div>
  </div>
</div>
<div id="fid"></div>
</section>

<section class="panel"><h2>What the detectors saw</h2>
  <p class="eyebrow" style="margin-bottom:12px">click anywhere to seek the video there</p>
  <div class="legend" id="lg"></div>
  <div class="scroll"><svg id="main" viewBox="0 0 900 470" width="900" height="470"></svg></div>
  <figcaption>Top: phase segmentation. Then the two holding routes, scored separately
  &mdash; they are ORed into `holding`, and an OR is validated by whichever branch fires,
  so they are drawn apart. Middle: end-effector distance to each BDDL object. Bottom:
  gripper aperture; shaded spans are commanded closures.</figcaption>
</section>

<section class="panel"><h2>Actions emitted</h2>
  <p class="eyebrow" style="margin-bottom:10px">7-D: 6 end-effector deltas + gripper</p>
  <div class="scroll"><svg id="act" viewBox="0 0 900 200" width="900" height="200"></svg></div>
</section>

<section class="panel" id="howtoread"><h2>How to read this page &mdash; what is accurate and what is indicative</h2>
  <p class="eyebrow">applies to every page this tool generates &middot; state of the pipeline 2026-09-16</p>
  <div class="trust">
    <div class="tier t-ok"><span class="tag">accurate &mdash; recorded</span><h3>Values read straight from the trace</h3>
      <ul>
        <li>End-effector position, gripper aperture, joint state</li>
        <li>The 7-D actions the policy emitted, every step</li>
        <li>Object positions and eef-to-object distances
          <br><em>measured to each object's body origin, not its surface &mdash; touching a large object can still read a few cm</em></li>
        <li>Success / failure, termination, step count</li>
      </ul></div>
    <div class="tier t-mid"><span class="tag">computed correctly &mdash; rule may be wrong</span><h3>Derived by our detectors</h3>
      <ul>
        <li><b>by_lift / holding</b> &mdash; object rose above its rest height. Reasonably reliable.</li>
        <li><b>by_gripper</b> &mdash; fingers stalled while open. <b>Misses thin objects</b> (0/79 on libero_spatial successes) and misreads a closing transient ~1.5% of the time.</li>
        <li><b>Phase bands</b> &mdash; built on the holding signal, so they inherit its errors.</li>
        <li><b>Grasp attempts</b> &mdash; counts commanded closures, not physical grasps.</li>
        <li>Check these against the video before relying on them.</li>
      </ul></div>
    <div class="tier t-bad"><span class="tag">not reliable yet</span><h3>The Diagnosis panel</h3>
      <ul>
        <li><b>Families</b> &mdash; <b>every</b> rule that matched is listed, unranked (since 2026-09-17; previously only the first match, which made labels depend on rule order). Listing all matches removes that artefact but <b>the rules are still weak</b>: <i>manipulation</i> fires on ~99% of failures, <i>spatial reasoning</i> counts the task&rsquo;s own destination as a distractor, and <i>visual grounding</i> is really a distance test. Unvalidated.</li>
        <li><b>Reason text</b> &mdash; states which measured value tripped each rule; the measurement is accurate, the family name attached to it is not.</li>
        <li><b>Failure cost</b> &mdash; always &ldquo;benign&rdquo; on LIBERO: the rule reads a toy-only <code>holding</code> key LIBERO never emits.</li>
      </ul></div>
    <div class="tier t-mid"><span class="tag">indicative &mdash; reconstruction</span><h3>The video</h3>
      <ul>
        <li>A <b>replay</b> of the stored actions from the same seed and init state, not a recording (pixels are never stored).</li>
        <li id="fidline">Checked against the stored trace every step; see the fidelity banner above.</li>
        <li>Frames are rotated 180&deg; to match what the policy saw (LeRobot&rsquo;s <code>LiberoProcessorStep</code>).</li>
        <li>When scrubbing, the picture is the replay and the numbers are the original recording. If the fidelity above is 0.0 mm they are the same episode frame for frame; otherwise use the <b>numbers</b> for exact positions and the <b>video</b> to judge what happened.</li>
      </ul></div>
  </div>
  <h3 style="font-family:var(--fd);font-size:14px;margin:18px 0 0">Assumptions behind this page</h3>
  <p style="font-size:13px;color:var(--ink2);margin:6px 0 0">Threshold values below are read from the code
  when the page is generated, so they match what produced this page.</p>
  <ul class="assume">
    <li><b>Target object</b> = the <i>first</i> object in the task&rsquo;s BDDL <code>obj_of_interest</code> list,
      which is the thing being moved; the destination (plate, basket, cabinet) is listed after it. If a task
      listed its destination first, every distance labelled &ldquo;target&rdquo; would be to the wrong object.</li>
    <li><b>Object positions</b> are a physical body&rsquo;s origin, or &mdash; for a table <i>region</i> such as
      &ldquo;front of the stove&rdquo; &mdash; a MuJoCo <i>site</i> (a massless location marker). Distances are from
      the end-effector point, not the fingertips, to that origin, so a gripper touching a large object can
      read several cm. Some regions resolve imperfectly: goal task 2&rsquo;s &ldquo;top of the cabinet&rdquo; maps to
      the cabinet <i>base</i> body.</li>
    <li><b>Thresholds carried over from a toy environment, never fitted on LIBERO:</b>
      closed gripper &lt; __TH_CLOSED__ m &middot; object lifted &gt; __TH_LIFT__ m &middot;
      fingers stalled &lt; __TH_STALL__ m over __TH_HOLD__ steps &middot; empty closure &asymp; __TH_EMPTY__ m &middot;
      pre-grasp &lt; __TH_PREGRASP__ m &middot; far from target &gt; __TH_LOST__ m &middot;
      near another object &lt; __TH_WRONG__ m &middot; repeated attempts within __TH_REPEAT__ m.</li>
    <li><b>Family rules</b> &mdash; every rule is evaluated and <b>all matches are shown, unranked</b>:
      <ul>
        <li><i>planning</i>: never entered pre-grasp</li>
        <li><i>spatial reasoning</i>: ended within __TH_WRONG__ m of <b>any other task object</b> and nearer to it
          than 0.6&times; the target distance &mdash; this includes the task&rsquo;s own destination</li>
        <li><i>visual grounding</i>: ended more than __TH_LOST__ m from the target &mdash; a <b>distance</b> test,
          not a test of which object the policy chose</li>
        <li><i>recovery</i>: two or more grasp attempts within __TH_REPEAT__ m of each other</li>
        <li><i>manipulation</i>: at least one grasp attempt &mdash; true of almost every failure</li>
      </ul></li>
    <li><b>No ground truth for failure families exists</b> on LIBERO. Demonstrations cover only successes;
      nothing validates a family label yet.</li>
    <li><b>Holding</b> has no direct signal. <code>by_lift</code> needs simulator state; <code>by_gripper</code>
      infers it from finger aperture and cannot see thin objects. Finger&ndash;object <b>contact is not recorded</b>
      in traces, so &ldquo;just missed&rdquo; and &ldquo;touched and slipped&rdquo; look the same.</li>
    <li><b>A grasp attempt</b> = the gripper action switching to close (&gt; 0.5). Counts commands, not physical
      grasps. Demonstrations are binary, so intermediate values are themselves out of distribution.</li>
    <li><b>Replay reproduces the episode</b> when the replay uses the trace&rsquo;s seed, init state, actions and
      <b>the same MuJoCo version</b>. Since 2026-09-17 seed N always starts from init state N. Older traces replay
      exactly only if their recorded reset order happened to match their seeds. This is not assumed &mdash; the
      fidelity check above measures it for this page.</li>
    <li><b>The policy&rsquo;s eval settings</b> are whatever the trace recorded (see its fingerprint). No checkpoint we
      use declares its control mode, so <code>relative</code> is unverified for every policy; SmolVLA has no documented
      eval settings at all.</li>
  </ul>
</section>

<footer>Generated from a stored rollout by <code>experiments/visualise_episode_video.py</code>.
See <code>IMPLEMENTATION_OVERSIGHTS.md</code> and <code>PENDING_DECISIONS.md</code> for the state of each detector.</footer>
</div>
<script>const D=__DATA__;const FPS=__FPS__;</script>
<script>
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const NS="http://www.w3.org/2000/svg";
const el=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e};
const tx=(x,y,s,a)=>{const e=document.createElementNS(NS,"text");
  for(const k in a)e.setAttribute(k,a[k]);e.setAttribute("x",x);e.setAttribute("y",y);e.textContent=s;return e};
const PC={approach:"--seq-2",pre_grasp:"--seq-3",grasp:"--seq-5",transport:"--good",retry:"--crit",idle:"--rule"};
const SER=["--acc","--warn","--good","--crit"];
const W=900,L=64,R=14,n=D.n,X=i=>L+(W-L-R)*i/Math.max(1,n-1);
const vid=document.getElementById("vid"),scrub=document.getElementById("scrub");
let cur=0;

function band(s,y,h,flags,colour,label){
  s.appendChild(tx(L-8,y+h-1,label,{fill:css("--ink3"),"font-size":10,"font-family":css("--fm"),"text-anchor":"end"}));
  s.appendChild(el("rect",{x:L,y,width:W-L-R,height:h,fill:css("--rule"),"fill-opacity":.4}));
  if(flags.every(v=>v===null)){
    s.appendChild(el("rect",{x:L,y,width:W-L-R,height:h,fill:"url(#hatch)"}));
    s.appendChild(tx(L+6,y+h-2,"NOT COMPUTED",{fill:css("--ink3"),"font-size":8.5,"font-family":css("--fm")}));
    return}
  let run=null;
  flags.forEach((v,i)=>{if(v&&run===null)run=i;
    if((!v||i===n-1)&&run!==null){
      s.appendChild(el("rect",{x:X(run),y,width:Math.max(1.5,X(i)-X(run)),height:h,fill:css(colour)}));run=null}});
}

function main(){
  const s=document.getElementById("main");s.textContent="";
  const defs=el("defs",{}),pat=el("pattern",{id:"hatch",width:6,height:6,patternUnits:"userSpaceOnUse",patternTransform:"rotate(45)"});
  pat.appendChild(el("line",{x1:0,y1:0,x2:0,y2:6,stroke:css("--ink3"),"stroke-width":1.2,"stroke-opacity":.5}));
  defs.appendChild(pat);s.appendChild(defs);
  const py=12,ph=20;
  D.phases.forEach(p=>{const x=X(p.a),w=Math.max(1.5,X(p.b)-X(p.a));
    s.appendChild(el("rect",{x,y:py,width:w,height:ph,fill:css(PC[p.p]||"--rule")}));
    if(w>54)s.appendChild(tx(x+w/2,py+14,p.p,{fill:"#fff","font-size":10,"font-family":css("--fm"),"text-anchor":"middle"}))});
  s.appendChild(tx(L-8,py+14,"phase",{fill:css("--ink3"),"font-size":10.5,"font-family":css("--fm"),"text-anchor":"end"}));
  band(s,42,11,D.by_gripper,"--acc","by_gripper");
  band(s,57,11,D.by_lift,"--good","by_lift");
  band(s,72,11,D.holding,"--warn","holding");
  const dy=104,dh=210;
  let mx=0;D.objects.forEach(o=>D.dist[o].forEach(v=>{if(v!=null&&v>mx)mx=v}));
  mx=Math.ceil(mx*10)/10||1;const Y=v=>dy+dh-dh*v/mx;
  [0,mx/2,mx].forEach(g=>{s.appendChild(el("line",{x1:L,x2:W-R,y1:Y(g),y2:Y(g),stroke:css("--rule")}));
    s.appendChild(tx(L-8,Y(g)+4,g.toFixed(2),{fill:css("--ink3"),"font-size":10.5,"font-family":css("--fm"),"text-anchor":"end"}))});
  s.appendChild(tx(L-8,dy-6,"dist (m)",{fill:css("--ink3"),"font-size":10.5,"font-family":css("--fm"),"text-anchor":"end"}));
  D.objects.forEach((o,k)=>{let d="",st=false;
    D.dist[o].forEach((v,i)=>{if(v==null)return;d+=(st?" L":"M")+X(i)+" "+Y(v);st=true});
    s.appendChild(el("path",{d,fill:"none",stroke:css(SER[k%SER.length]),"stroke-width":k===0?2.2:1.5,
      "stroke-dasharray":k===0?"":"4 3"}))});
  let run=null;D.cmd.forEach((c,i)=>{if(c&&run===null)run=i;
    if((!c||i===n-1)&&run!==null){s.appendChild(el("rect",{x:X(run),y:dy,width:Math.max(1,X(i)-X(run)),
      height:dh,fill:css("--crit"),"fill-opacity":.08}));run=null}});
  const gy=340,gh=90;let gmx=Math.max(...D.grip,0.001);
  s.appendChild(tx(L-8,gy-4,"aperture",{fill:css("--ink3"),"font-size":10.5,"font-family":css("--fm"),"text-anchor":"end"}));
  let gd="";D.grip.forEach((v,i)=>{gd+=(i?" L":"M")+X(i)+" "+(gy+gh-gh*v/gmx)});
  s.appendChild(el("path",{d:gd,fill:"none",stroke:css("--ink2"),"stroke-width":1.6}));
  const cl=gy+gh-gh*D.closed_m/gmx;
  s.appendChild(el("line",{x1:L,x2:W-R,y1:cl,y2:cl,stroke:css("--crit"),"stroke-dasharray":"3 3","stroke-opacity":.7}));
  s.appendChild(tx(W-R,cl-4,"closed_m",{fill:css("--crit"),"font-size":9.5,"font-family":css("--fm"),"text-anchor":"end"}));
  [0,n-1].forEach(i=>s.appendChild(tx(X(i),460,"t="+i,{fill:css("--ink3"),"font-size":10.5,"font-family":css("--fm"),"text-anchor":i?"end":"start"})));
  s.appendChild(el("line",{id:"ph1",x1:L,x2:L,y1:8,y2:440,stroke:css("--ink"),"stroke-width":1.3,"stroke-opacity":.85}));
  const lg=document.getElementById("lg");
  D.objects.forEach((o,k)=>{const sp=document.createElement("span");
    sp.innerHTML='<i class="sw" style="background:'+css(SER[k%SER.length])+'"></i>'+o.replace(/_main$/,"")+(k===0?" (target)":"");
    lg.appendChild(sp)});
}
function acts(){
  const s=document.getElementById("act");s.textContent="";
  const names=["dx","dy","dz","droll","dpitch","dyaw","grip"],h=22;
  names.forEach((nm,k)=>{const y0=10+k*h+h/2;
    s.appendChild(tx(L-8,y0+4,nm,{fill:css("--ink3"),"font-size":10.5,"font-family":css("--fm"),"text-anchor":"end"}));
    s.appendChild(el("line",{x1:L,x2:W-R,y1:y0,y2:y0,stroke:css("--rule")}));
    let d="";D.act.forEach((a,i)=>{if(!a)return;const v=Math.max(-1,Math.min(1,a[k]||0));
      d+=(d?" L":"M")+X(i)+" "+(y0-v*(h/2-2))});
    s.appendChild(el("path",{d,fill:"none",stroke:css(k===6?"--crit":"--acc"),"stroke-width":1.2}))});
  s.appendChild(el("line",{id:"ph2",x1:L,x2:L,y1:4,y2:196,stroke:css("--ink"),"stroke-width":1.3,"stroke-opacity":.85}));
}
const tri=v=>v===null?"n/a — not computed":(v?"yes":"no");
function setCur(i,seek){
  cur=Math.max(0,Math.min(n-1,Math.round(i)));
  const x=X(cur);
  ["ph1","ph2"].forEach(id=>{const e=document.getElementById(id);if(e){e.setAttribute("x1",x);e.setAttribute("x2",x)}});
  scrub.value=cur;
  document.getElementById("tread").textContent="t="+cur;
  const ph=(D.phase_at||[])[cur]||"-";
  [["f_g",D.by_gripper[cur]],["f_l",D.by_lift[cur]],["f_h",D.holding[cur]]].forEach(([id,v])=>{
    const e=document.getElementById(id);e.classList.toggle("on",v===true);e.classList.toggle("na",v===null)});
  document.getElementById("f_p").textContent=ph;
  const live=document.getElementById("live");live.textContent="";
  const rows=[["step",cur+"  /  "+(n-1)],["phase",ph],
    ["aperture",D.grip[cur]!=null?D.grip[cur].toFixed(4)+" m":"n/a"],
    ["commanded",D.cmd[cur]?"CLOSE":"open"],
    ["by_gripper",tri(D.by_gripper[cur])],["by_lift",tri(D.by_lift[cur])]];
  D.objects.forEach((o,k)=>{const v=D.dist[o][cur];
    rows.push([(k===0?"→ ":"   ")+o.replace(/_main$/,""),v==null?"n/a":(v*100).toFixed(1)+" cm"])});
  rows.forEach(([k,v])=>{const r=document.createElement("div");
    r.innerHTML='<span>'+k+'</span><span class="v">'+v+'</span>';live.appendChild(r)});
  if(seek&&vid.src){const t=cur/FPS;if(Math.abs(vid.currentTime-t)>1/FPS)vid.currentTime=t}
}
function diag(){
  const d=D.diag,c=document.getElementById("diag");
  const cost=d.failure_cost&&d.failure_cost.cost;
  const fams=(d.families&&d.families.length)?d.families:null;
  [["families matched",fams?fams.length+" of 5 rules":(d.family?d.family:"succeeded")],
   ["confident",String(d.confident)],["terminal",d.terminal],
   ["final error",d.final_error_m==null?"n/a":(d.final_error_m*100).toFixed(1)+" cm"],
   ["grasp attempts",d.grasp_attempts],
   ["attempt spread",d.attempt_spread_m==null?"n/a":(d.attempt_spread_m*100).toFixed(1)+" cm"],
   ["failure cost",cost||"n/a"]].forEach(([k,v])=>{const r=document.createElement("div");
    r.innerHTML='<span>'+k+'</span><span class="v">'+v+'</span>';c.appendChild(r)});
  if(fams){
    const h=document.createElement("div");h.style.borderBottom="0";
    h.innerHTML='<span class="eyebrow" style="margin-top:6px">every rule that matched &mdash; not ranked</span>';c.appendChild(h);
    fams.forEach(f=>{const r=document.createElement("div");
      r.style.flexDirection="column";r.style.alignItems="flex-start";
      r.innerHTML='<span class="flag on h" style="text-transform:none">'+f.replace(/_/g," ")+'</span>'+
        '<span style="color:var(--ink2);font-size:12.5px;margin-top:3px">'+((d.reasons||{})[f]||"")+'</span>';
      c.appendChild(r)});
    const P=d.predicates||{};
    const q=document.createElement("div");q.style.borderBottom="0";
    q.innerHTML='<span class="mono" style="font-size:11px;color:var(--ink3)">'+
      Object.keys(P).map(k=>(P[k]?"✓ ":"✗ ")+k).join(" &nbsp; ")+'</span>';c.appendChild(q);
  }else{
    const r=document.createElement("div");r.style.borderBottom="0";
    r.innerHTML='<span style="color:var(--ink2);font-style:italic">'+(d.reason||"")+'</span>';c.appendChild(r);
  }
}
function fid(){
  const f=D.fidelity,box=document.getElementById("fid");
  let pre="";
  if(D.abstain)pre+='<div class="warn bad"><b>Detectors abstained for this episode.</b> '+
    'The phase segmenter and <code>by_lift</code> could not run ('+D.abstain+'), so they show '+
    '<b>NOT COMPUTED</b> — not &ldquo;no&rdquo;. The Diagnosis panel is empty for the same reason. '+
    'On libero_goal this is the known instrumentation gap: tasks 0&ndash;5 were captured before the '+
    'BDDL region fix and never recorded object distances. Needs the goal re-run.</div>';
  if(D.thin_stall)pre+='<div class="warn"><b>Possible thin-object grasp that <code>by_gripper</code> cannot see.</b> '+
    'From t='+D.thin_stall.t+' the fingers were commanded closed and <b>stopped</b> at '+
    (D.thin_stall.aperture_m*1000).toFixed(1)+' mm — well above an empty closure (~6 mm) but below '+
    'the '+(D.closed_m*1000).toFixed(0)+' mm threshold, so <code>by_gripper</code> reads it as &ldquo;closed on '+
    'nothing&rdquo;. Known limitation O4: a single aperture threshold cannot serve thin and thick objects.</div>';
  if(!f||f.max_dev_m==null){box.innerHTML='<div class="warn bad"><b>Replay not verified.</b> '+
    'The end-effector path could not be compared against the stored trace, so the video '+
    'is not confirmed to be this episode.</div>';box.insertAdjacentHTML("afterbegin",pre);return}
  const bad=f.max_dev_m>0.005;
  box.innerHTML='<div class="warn'+(bad?' bad':'')+'"><b>Replay fidelity:</b> '+
    'max deviation <span class="mono">'+(f.max_dev_m*1000).toFixed(2)+' mm</span>, '+
    'median <span class="mono">'+(f.median_dev_m*1000).toFixed(2)+' mm</span> over '+
    f.n_compared+' steps'+(bad?' &mdash; ABOVE 5 mm. The replay diverged from the stored '+
    'trace; treat the video as indicative only.':' &mdash; the replay follows the stored trace.')+'</div>';
  box.insertAdjacentHTML("afterbegin",pre);
}
main();acts();diag();fid();
(function(){const f=D.fidelity,li=document.getElementById("fidline");if(!li)return;
  if(!f||f.max_dev_m==null){li.innerHTML="<b>Not verified</b> for this episode: the replay could not be compared to the trace.";return}
  const ok=f.max_dev_m<=0.005;
  li.innerHTML="This episode: median <b>"+(f.median_dev_m*1000).toFixed(1)+" mm</b>, max <b>"+(f.max_dev_m*1000).toFixed(1)+
    " mm</b> from the stored path &mdash; "+(ok?"within 5 mm, trust frame by frame.":
    "<b>above 5 mm</b>: right episode and motion, not millimetre-exact.")})();
scrub.addEventListener("input",e=>setCur(+e.target.value,true));
document.getElementById("bk").onclick=()=>setCur(cur-1,true);
document.getElementById("fw").onclick=()=>setCur(cur+1,true);
document.getElementById("play").onclick=()=>{if(vid.paused){vid.play();document.getElementById("play").textContent="pause"}
  else{vid.pause();document.getElementById("play").textContent="play"}};
vid.addEventListener("timeupdate",()=>{if(!vid.paused)setCur(vid.currentTime*FPS,false)});
vid.addEventListener("ended",()=>{document.getElementById("play").textContent="play"});
["main","act"].forEach(id=>document.getElementById(id).addEventListener("click",ev=>{
  const sv=ev.currentTarget,bb=sv.getBoundingClientRect();
  const vx=(ev.clientX-bb.left)/bb.width*W;
  setCur((vx-L)/(W-L-R)*(n-1),true)}));
document.addEventListener("keydown",e=>{
  if(e.key==="ArrowLeft"){setCur(cur-1,true);e.preventDefault()}
  if(e.key==="ArrowRight"){setCur(cur+1,true);e.preventDefault()}
  if(e.key===" "){document.getElementById("play").click();e.preventDefault()}});
setCur(0,false);
</script></body></html>"""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    run_dir = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 else None
    out = sys.argv[3] if len(sys.argv) > 3 else "episode_video.html"

    chosen = None
    for line in open(os.path.join(run_dir, "rollouts.jsonl")):
        r = rollout_from_dict(json.loads(line))
        if want is None or r.rollout_id == want:
            chosen = r
            break
    if chosen is None:
        sys.exit(f"rollout {want} not found in {run_dir}")

    D = extract(chosen)
    # extract() (visualise_episode.py, deliberately left untouched) copies a fixed
    # set of diagnosis keys. The classifier now reports EVERY matching family, so
    # add those fields here rather than showing only the joined grouping key.
    from vla_harness.mining.classify import classify
    _d = classify(chosen, segmenter=LiberoPhaseSegmenter())
    D["diag"].update(families=_d.get("families", []), predicates=_d.get("predicates", {}),
                     reasons=_d.get("reasons", {}), family=_d.get("family"),
                     reason=_d.get("reason"))
    det = detector_trace(chosen)
    seg = LiberoPhaseSegmenter()
    segs, _ = seg(chosen)
    phase_at = [None] * len(chosen.steps)
    for sg in segs:
        for i in range(sg.t_start, sg.t_end + 1):
            if i < len(phase_at):
                phase_at[i] = sg.phase
    D.update(by_gripper=det["by_gripper"], by_lift=det["by_lift"],
             holding=det["holding"], phase_at=phase_at,
             abstain=det["abstain"], thin_stall=det["thin_stall"],
             cmd=[bool(seg._commanded_closed(s)) for s in chosen.steps],
             closed_m=seg.closed_m)

    print(f"replaying {chosen.rollout_id} ({D['n']} steps) ...")
    cams, fidelity = replay(chosen)
    D["fidelity"] = fidelity
    print(f"  fidelity: max {fidelity['max_dev_m']} m, median {fidelity['median_dev_m']} m "
          f"over {fidelity['n_compared']} steps")

    vid_attr = ""
    if cams:
        tmp = tempfile.mktemp(suffix=".mp4")
        if encode_video(cams, tmp):
            b = base64.b64encode(open(tmp, "rb").read()).decode()
            os.unlink(tmp)
            vid_attr = f'src="data:video/mp4;base64,{b}"'
            print(f"  video: {len(cams)} cameras, {len(cams[0])} frames, "
                  f"{len(b)//1024} KB base64")
    if not vid_attr:
        print("  WARNING: no frames captured; page will render without video")

    from vla_harness.mining import classify as _cls, phases_libero as _pl
    _seg = LiberoPhaseSegmenter()
    thresholds = {"__TH_CLOSED__": _seg.closed_m, "__TH_LIFT__": _seg.lift_m,
                  "__TH_STALL__": _seg.stall_m, "__TH_HOLD__": _seg.hold_steps,
                  "__TH_EMPTY__": EMPTY_CLOSED_M, "__TH_PREGRASP__": _seg.pregrasp_m,
                  "__TH_LOST__": _cls.LOST_TARGET_M, "__TH_WRONG__": _cls.WRONG_OBJECT_M,
                  "__TH_REPEAT__": _cls.REPEAT_SPREAD_M}
    html = HTML
    for k, v in thresholds.items():
        html = html.replace(k, f"{v:g}")
    html = (html.replace("__DATA__", json.dumps(D))
                .replace("__VIDATTR__", vid_attr)
                .replace("__FPS__", str(FPS))
                .replace("__NMAX__", str(D["n"] - 1))
                .replace("__ID__", D["id"]).replace("__TASKID__", D["task"])
                .replace("__INSTR__", D["instruction"])
                .replace("__FAMILY__", (" + ".join(D["diag"].get("families") or []) or
                                        D["diag"]["family"] or "success").replace("_", " "))
                .replace("__N__", str(D["n"]))
                .replace("__TERM__", D["termination"])
                .replace("__PERT__", json.dumps(D["perturbation"]) if D["perturbation"] else "nominal")
                .replace("__POLICY__", D["policy"].split("@")[0]))
    open(out, "w").write(html)
    print(f"wrote {out}  ({os.path.getsize(out)//1024} KB, family={D['diag']['family']})")
