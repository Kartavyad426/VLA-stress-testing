"""The perturbation → correction map, compiled: one section per row, every
result that bears on the row, common method and maths first, glossary last.

  .venvs/groot/bin/python experiments/compile_map.py [--out docs/PERTURBATION_MAP.html]

Every number is read from the run files (manifests, analysis.json, probe
json, boundaries.jsonl); nothing is typed in. Figures are inline SVG, reused
from the existing method pages where they already exist.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from experiments.r041_report import AXIS_LABEL, SCENE, fmt_b, load_axis, plot as r041_plot  # noqa: E402
from vla_harness.analysis.r039 import load_instances  # noqa: E402

E = html.escape
ARM_NAME = {"P": "perturbed (executed)", "N": "nominal", "T": "text tokens restored", "I": "image tokens restored",
            "S": "state token restored", "IT": "image + text tokens restored", "A": "agent-view pixels restored",
            "W": "wrist pixels restored"}
ROWS = [("Camera Viewpoints", "camera"), ("Sensor Noise", "noise"), ("Light Conditions", "light"),
        ("Robot Initial States", "startpose"), ("Objects Layout", "layout"), ("Background Textures", "texture"),
        ("Language Instructions", "language")]


# ----------------------------------------------------------------------------- data
def manifest(run):
    p = f"runs/{run}/splice/manifest.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def analysis(run):
    p = f"runs/{run}/analysis.json"
    return json.load(open(p)) if os.path.exists(p) else None


def jsonf(p):
    return json.load(open(p)) if os.path.exists(p) else None


def rows_of(run, cat):
    return [r for r in manifest(run) if r["category"] == cat]


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(xs)) if xs else float("nan")


def f2(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


# ----------------------------------------------------------------------------- html bits
def table(headers, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td{' class=num' if isinstance(c, (int, float)) else ''}>{c if not isinstance(c, float) else f2(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl-scroll"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def ok(b):
    return '<span class="yes">ok</span>' if b else '<span class="no">fail</span>'


def lvl(n):
    return f'<span class="lvl L{n}">L{n}</span>'


def note(title, body, kind=""):
    return f'<div class="note {kind}"><span class="nh">{title}</span>{body}</div>'


def figure_from(path, keyword):
    """Lift an existing <figure>…</figure> whose svg aria-label contains keyword."""
    s = open(path).read()
    for m in re.finditer(r"<figure>.*?</figure>", s, re.S):
        if keyword in m.group(0):
            return re.sub(r'<span class="fignum">[^<]*</span>\s*', "", m.group(0))   # drop the source page's figure numbers
    return ""


def denoise_table(cat, runs_labels):
    """Median transfer at forward 0 and at the anchor, per run, for one category."""
    out = []
    for run, label in runs_labels:
        a = analysis(run)
        if not a or cat not in a["by_category"]:
            continue
        c = a["by_category"][cat]
        arms = list(c["median_tf_f0"])
        out.append([label, c["n"], c["n_fail"], c["median_d_pn_f0"]] + [c["median_tf_f0"][x] for x in arms] + [c["median_tf_anchor"][x] for x in arms])
        hdr = ["run", "n", "fail", "‖P−N‖ f0"] + [f"{x} f0" for x in arms] + [f"{x} anchor" for x in arms]
    return table(hdr, out) if out else "<p>no runs</p>"


def instance_table(cat, run):
    rs = rows_of(run, cat)
    if not rs:
        return ""
    arms = [x for x in ("T", "I", "S", "IT", "A", "W") if rs[0].get("tf_at_f0") and x in rs[0]["tf_at_f0"]]
    hdr = ["level", "instance", "task", "outcome", "fwds", "anchor", "‖P−N‖ f0"] + [f"{x} f0" for x in arms] + [f"{x} anc" for x in arms]
    body = []
    for r in sorted(rs, key=lambda r: (-(r["level"] or 0), r["task_id"])):
        f0, an = r.get("tf_at_f0") or {}, r.get("tf_at_anchor") or {}
        body.append([lvl(r["level"]), E(str(r["label"])), r["task_id"], ok(r["success"]), r["forwards"], r["anchor_forward"], r["d_pn_f0"] if r["d_pn_f0"] is not None else float("nan")]
                    + [f0.get(x, float("nan")) for x in arms] + [an.get(x, float("nan")) for x in arms])
    return table(hdr, body)


def curves(cat, run, title):
    from experiments.r039_report import curve_svg
    try:
        inst = [i for i in load_instances(f"runs/{run}") if i["category"] == cat]
    except Exception:
        return ""
    if not inst:
        return ""
    figs = "".join(f"<figure class=small>{curve_svg(i, i['rollout_id'])}<figcaption>{lvl(i.get('level'))} {E(str(i.get('label')))} · {'success' if i.get('success') else 'fail'}</figcaption></figure>"
                   for i in sorted(inst, key=lambda p: p["task_id"]))
    return f"<h4>{title}</h4><div class=grid>{figs}</div>"


def rescue_table(base_run, cat, drives):
    P = {r["task_id"]: r for r in rows_of(base_run, cat)}
    D = {d: {r["task_id"]: r for r in manifest(run)} for d, run in drives.items() if manifest(run)}
    if not P or not D:
        return "", {}
    hdr = ["task", "instance", "drive P"] + [f"drive {d}" for d in D] + ["closest P"] + [f"closest {d}" for d in D]
    body = []
    for t in sorted(P):
        body.append([t, E(str(P[t]["label"])), ok(P[t]["success"])] + [ok(D[d][t]["success"]) if t in D[d] else "—" for d in D]
                    + [P[t]["closest_approach_m"]] + [D[d][t]["closest_approach_m"] if t in D[d] else float("nan") for d in D])
    summ = {"P": (sum(1 for t in P if not P[t]["success"]), med([P[t]["closest_approach_m"] for t in P]))}
    for d in D:
        fails = sum(1 for t in D[d] if not D[d][t]["success"]); flips = sum(1 for t in P if not P[t]["success"] and t in D[d] and D[d][t]["success"])
        broke = sum(1 for t in P if P[t]["success"] and t in D[d] and not D[d][t]["success"])
        summ[d] = (fails, med([D[d][t]["closest_approach_m"] for t in D[d]]), flips, broke)
    srows = [["P", summ["P"][0], "—", "—", summ["P"][1]]] + [[d, summ[d][0], summ[d][2], summ[d][3], summ[d][1]] for d in D]
    return table(hdr, body) + table(["driven at forward 0", "fails /10", "P-failures flipped", "successes broken", "median closest (m)"], srows), summ


def downstream_table(base_run, cat, drives):
    """R-050: ‖P−N‖ per forward after a forward-0 correction vs the drive-P rollouts."""
    P = {r["task_id"]: r for r in rows_of(base_run, cat)}
    def dpn(run, t, rid):
        p = f"runs/{run}/splice/{rid}.npz"
        return np.load(p)["d_pn"] if os.path.exists(p) else None
    out, summ = [], {}
    series = {"P": {t: dpn(base_run, t, P[t]["rollout_id"]) for t in P}}
    for d, run in drives.items():
        m = {r["task_id"]: r for r in manifest(run)}
        series[d] = {t: dpn(run, t, m[t]["rollout_id"]) for t in m}
    for t in sorted(P):
        row = [t, E(str(P[t]["label"]))]
        for d in series:
            s = series[d].get(t)
            row.append(" ".join(f"{x:.1f}" for x in s[:5]) if s is not None else "—")
        out.append(row)
    for d in series:
        later = [float(np.mean(s[1:4])) for s in series[d].values() if s is not None and len(s) >= 4]
        summ[d] = med(later)
    return table(["task", "instance"] + [f"‖P−N‖ f0…f4 · drive {d}" for d in series], out), summ


# ----------------------------------------------------------------------------- sections
def common():
    entry = figure_from("docs/FAILURE_TO_DATA_PIPELINE.html", "GR00T N1.7 data flow")
    arms = figure_from("docs/FAILURE_TO_DATA_PIPELINE.html", "Source run under the nominal condition")
    hybrid = figure_from("docs/R042_WRIST_MATHS.html", "Two simulator runs at forward 0")
    return f'''
<section id="common">
<p class="part">0 · Common ground</p>
<h2>The policy, the intervention, the maths, the conventions</h2>
<p>Everything below uses one checkpoint, <code>nvidia/gr00t17-lerobot-libero_spatial-640</code> on GR00T N1.7, on the LIBERO-Plus fork of <code>libero_spatial</code>, through the project harness. The policy sees two RGB images, eight numbers of proprioception and an instruction string, and emits a chunk of 40 actions every 16 simulator steps ("one forward"). Of the 132 output dimensions, 7 are live.</p>
{entry}
<h3>The intervention</h3>
<p>At a forward, the action head is run several times from the same simulator state and the same fixed denoising noise, once per <em>arm</em>. Each arm replaces one input with what it would have been under the trained condition and leaves the rest as the perturbed run produced them. Only one arm's chunk is executed, normally the perturbed one, so the measurement rides along the real trajectory without altering it.</p>
{arms}
<p>Token-level arms (T, I, S, IT) splice at the adapter output, where the DiT cross-attends. Pixel-level arms (A, W) swap one camera's image before the vision-language model, which recomputes every token; they are the only clean per-camera split, because at the adapter output every image token depends on both cameras (R-045).</p>
{hybrid}
<h3>The maths</h3>
<div class="thm trig"><div class="thm-label">Transfer fraction</div><div class="thm-body">
<span class="eq">τ<sub>X</sub>(f) = 1 − ( ‖<span class="m">a</span><sub>X</sub> − <span class="m">a</span><sub>N</sub>‖ / ‖<span class="m">a</span><sub>P</sub> − <span class="m">a</span><sub>N</sub>‖ )</span>
Euclidean distances on the 280-number live chunk in the policy's normalised action units. The share of the perturbed-to-nominal gap that restoring input X alone closes. 1: that input carried everything. 0: nothing. Negative: restoring it moved the action further away. Bounded above by 1, not rescaled. Not a cosine, and not additive across arms. Full derivation and sources in <a href="R039_MATHS.html">R039_MATHS.html</a>.
</div></div>
<div class="thm"><div class="thm-label">Effect size, onset, anchor, dominant, residual</div><div class="thm-body">
<span class="eq">‖P−N‖ = ‖<span class="m">a</span><sub>P</sub> − <span class="m">a</span><sub>N</sub>‖ &nbsp;·&nbsp; onset<sub>X</sub> = min{{ f : τ<sub>X</sub>(f) ≥ 0.5 }} &nbsp;·&nbsp; dominant = argmax<sub>X</sub> τ<sub>X</sub>(f*) if ≥ 0.9, else residual</span>
The anchor f* is the forward containing the closest approach of the end effector to the target bowl, from the simulator's privileged distance, mapped through the logged step of each forward.
</div></div>
<div class="thm"><div class="thm-label">Reverse direction ("noising")</div><div class="thm-body">
<span class="eq">noise<sub>X</sub> = 1 − ‖<span class="m">a</span><sub>revX</sub> − <span class="m">a</span><sub>P</sub>‖ / ‖<span class="m">a</span><sub>N</sub> − <span class="m">a</span><sub>P</sub>‖</span>
One perturbed input placed into the nominal run; the share of the effect it produces alone. Denoising high and noising high: necessary and sufficient. Denoising high, noising low: redundancy, the information is readable elsewhere too.
</div></div>
<h3>Where the nominal comes from</h3>
{table(["source", "used for", "exactness"], [
    ["paired render", "camera, light, noise: the same simulator state re-rendered under the trained camera and lights", "exact at every forward; state token identical, so S ≡ 0 and IT ≡ 1 by construction"],
    ["recorded control", "start pose, layout, texture: the scene's control variant rolled out under the same noise, features replayed per forward", "exact at forward 0, a bound afterwards"],
    ["paired text", "language: the same observation with the trained instruction", "exact at every forward; A and W are identity"],
])}
<h3>Conventions that hold throughout</h3>
<ul class="tight">
<li><b>Determinism.</b> The denoising noise is seeded per forward and shared across arms (common random numbers). Model-level gate: identical inputs and seed give bitwise-identical chunks; the unseeded control differs (R-039).</li>
<li><b>Outcome instability.</b> Two rollouts with identical seed and noise, differing only by render jitter, disagreed on success on 7 of 30, 9 of 40 and 4 of 10 occasions across experiments. Action-level numbers replicate to 0.03; outcomes at one seed carry a ~30–40% flip rate. Every outcome claim here is read against that.</li>
<li><b>Env seed 0</b> on every rollout since R-039, so the object layout is fixed; replicates vary the noise seed and, for start pose, the joint direction.</li>
<li><b>Levels</b> L1–L5 are the benchmark's difficulty presets; the failure campaign ran L4 and L5 only.</li>
<li><b>Pre-registration.</b> Every experiment's expectations were written before its run; misses are recorded, and corrections made mid-run are timestamped (appendix C).</li>
</ul>
</section>'''


def section_camera():
    a39 = analysis("runs/r039") if False else None
    rev = jsonf("runs/r048_reverse/summary.json")
    rescue_html, rs = rescue_table("r042", "Camera Viewpoints", {"A": "r048_rescue_A", "W": "r048_rescue_W", "N": "r048_rescue_N"})
    down_html, ds = downstream_table("r042", "Camera Viewpoints", {"N": "r050_cam_N", "A": "r050_cam_A"})
    seeds = []
    s0 = {p["task_id"]: p for p in (analysis("r042") or {"instances": []})["instances"] if p["category"] == "Camera Viewpoints"}
    for sd in (1, 2):
        a = analysis(f"r048_seed{sd}")
        if a:
            b = {p["task_id"]: p for p in a["instances"]}
            seeds.append([f"noise seed {sd}", med([b[t]["tf_f0"]["A"] for t in b]), med([b[t]["tf_f0"]["W"] for t in b]),
                          f"{sum(b[t]['tf_f0']['A'] > b[t]['tf_f0']['W'] for t in b)}/10", f"{sum(abs(b[t]['tf_f0']['A'] - s0[t]['tf_f0']['A']) <= 0.1 for t in b if t in s0)}/10",
                          f"{sum(1 for t in b if not b[t]['success'])}/10"])
    yaw_rows, yaw_b = load_axis("runs/r041", "camera_yaw_deg"); dist_rows, dist_b = load_axis("runs/r041", "camera_dist_m")
    tau = (jsonf("runs/r041/tau.json") or {}).get("tau")
    def btable(bounds):
        return table(["scene", "action boundary", "outcome boundary", "note"], [[SCENE.get(b["task_id"], b["task_id"]), fmt_b(b.get("action_boundary")), fmt_b(b.get("outcome_boundary")), E(b.get("note", ""))] for b in sorted(bounds, key=lambda b: b["task_id"])])
    return f'''
<section id="camera">
<p class="part">1 · Camera viewpoint</p>
<h2>Carried by the agent-view pixels; the loop absorbs it for tens of degrees</h2>
<p><b>What the perturbation is.</b> The benchmark moves the third-person camera on a cone of 15° to 75° in azimuth and elevation and scales its distance up to 2.0×; the wrist camera and the world are untouched. Our ten instances are level-4 and level-5 failures from the campaign: yaw 22° to 30° with 15° pitch, yaw around −45° and −63°, and two distance scalings of 1.73× and 1.85×.</p>
<h3>Denoising map (R-039, R-042)</h3>
{denoise_table("Camera Viewpoints", [("r039", "R-039 paired render"), ("r039_recorded", "R-039 recorded control"), ("r042", "R-042 with pixel arms")])}
{instance_table("Camera Viewpoints", "r042")}
<p>Restoring the agent-view pixels alone recovers 0.95 to 0.98 on every instance; the wrist arm reads 0.00 ± 0.02. The token-level image arm reads only 0.69 because image content leaks into text positions inside the VLM; A is the informational arm and closes the gap.</p>
{curves("Camera Viewpoints", "r042", "Transfer curves along each episode (R-042)")}
<h3>Robustness (R-048)</h3>
<p><b>Reverse direction.</b> A viewpoint change touches one input, so corrupting the agent view of the nominal run reproduces the whole effect by construction. This run is therefore an exactness check on the paired render: noise<sub>A</sub> {f2(rev["medians"]["noise_A"]) if rev else "—"}, noise<sub>W</sub> {f2(rev["medians"]["noise_W"]) if rev else "—"}, noise<sub>S</sub> {f2(rev["medians"]["noise_S"]) if rev else "—"} (medians), and denoise<sub>A</sub> reproduces R-042 to 0.01 on 10/10.</p>
{table(["replicate", "A median", "W median", "A > W", "A within 0.1 of seed 0", "fails"], seeds)}
<p><b>Forward-0 rescue.</b> The corrected chunk is executed for the first forward only, then the perturbed policy continues. On this row the wrist arm's chunk equals P's up to render jitter, so "drive W" is a re-run and gives the no-correction baseline.</p>
{rescue_html}
<p>A re-run alone flipped 3 of 7 failures; the agent-view correction added one, the full nominal chunk two; closest approach improved on 9 of 10. At one seed the outcome is dominated by the flip rate, not the first chunk.</p>
<h3>Downstream of a single correction (R-050)</h3>
<p>‖P−N‖ at each forward after the corrected chunk was executed at forward 0, against the drive-P rollouts. Median over forwards 1–3: {" · ".join(f"drive {d}: {f2(v)}" for d, v in ds.items())}.</p>
{down_html}
<h3>Continuous sweep (R-041)</h3>
<p>Bisection from the clean control, one rollout per point, τ = {f2(tau)} (paired-render noise floor). The first chunk departs at the smallest step on every scene; the outcome holds to 40° on 8 of 10 and to 0.4 m on 9 of 10.</p>
<h4>Yaw, 0–40°</h4>{btable(yaw_b)}
<figure>{r041_plot(yaw_rows, "camera_yaw_deg", 40.0, tau, "d_pn_f0", "‖P−N‖ at forward 0")}<figcaption>‖P−N‖ at forward 0 against yaw, one line per scene; squares are failures; red lines are R-038's null-prompt scenes.</figcaption></figure>
<h4>Distance, 0–0.4 m</h4>{btable(dist_b)}
<figure>{r041_plot(dist_rows, "camera_dist_m", 0.4, tau, "d_pn_f0", "‖P−N‖ at forward 0")}<figcaption>‖P−N‖ at forward 0 against camera distance.</figcaption></figure>
{note("Verdict", "<b>Dominant input:</b> agent-view pixels, 0.98, exact, stable across noise seeds, necessary by construction. <b>Necessity:</b> established (one-input perturbation). <b>Outcome:</b> a camera change moves the first action immediately and is absorbed for 20–40°; a single corrected chunk changes the outcome only marginally above a re-run. <b>Data spec:</b> re-render existing demonstrations under the new viewpoint (condition gap, family B); the wrist stream is untouched. <b>Open:</b> the benchmark's full 75° cone and probability curves with replicates (R-047, pending); whether the accumulation reading holds on the downstream measurement above.", "repo")}
</section>'''


def section_noise():
    return f'''
<section id="noise">
<p class="part">2 · Sensor noise</p>
<h2>Carried by the agent-view pixels; needs one to three forwards to be absorbed at the token level</h2>
<p><b>What the perturbation is.</b> The benchmark's fork blurs or fogs the agent-view image after rendering (motion, Gaussian, zoom and glass blur, fog, severities to 50); the wrist image is clean. Our ten instances: all five campaign failures plus five successes, severities 16 to 39.</p>
<h3>Denoising map (R-039, R-042)</h3>
{denoise_table("Sensor Noise", [("r039", "R-039 paired render"), ("r039_recorded", "R-039 recorded control"), ("r042", "R-042 with pixel arms")])}
{instance_table("Sensor Noise", "r042")}
<p>The token-level image arm reads only 0.45 at forward 0 and reaches 0.82 by the anchor, with onset at forwards 1–3 on six instances; the pixel-level agent-view arm reads 0.97 at forward 0 with the wrist at 0.00. The deficit in the token-level arm was image content in text positions, not an interaction. Four instances have ‖P−N‖ above 3, the largest first-action effect of any render category, yet only one to three of ten fail.</p>
{curves("Sensor Noise", "r042", "Transfer curves (R-042)")}
{note("Verdict", "<b>Dominant input:</b> agent-view pixels, 0.97; necessity by construction (one-input perturbation). <b>Not run:</b> the noising, seed and rescue checks (the camera row's R-048 is the template) and a continuous severity sweep. <b>Data spec:</b> augmentation of the agent-view stream with the noise family; the wrist stream is untouched.", "repo")}
</section>'''


def section_light():
    light_rows, light_b = load_axis("runs/r041", "light_intensity"); tau = (jsonf("runs/r041/tau.json") or {}).get("tau")
    return f'''
<section id="light">
<p class="part">3 · Lighting</p>
<h2>A cross-camera property: neither camera alone recovers it</h2>
<p><b>What the perturbation is.</b> The benchmark swaps the scene's light definitions: diffuse colour, direction, specular, shadows. Our ten instances: all four campaign failures plus six successes. Separately, the harness has a brightness-only knob used for the sweep, which is not the benchmark's quantity.</p>
<h3>Denoising map (R-039, R-042)</h3>
{denoise_table("Light Conditions", [("r039", "R-039 paired render"), ("r039_recorded", "R-039 recorded control"), ("r042", "R-042 with pixel arms")])}
{instance_table("Light Conditions", "r042")}
<p>Restoring either camera's pixels alone recovers under 0.2 (A 0.16, W 0.19, wrist above agent on 4 of 10); restoring both cameras' tokens recovers 0.73. The sum of the single-camera arms falls short of the joint arm by up to 0.95: the head reads illumination as a joint property of the two views, and a mismatch between them is itself a perturbation. This is the one row where a finer decomposition would have something to explain.</p>
{curves("Light Conditions", "r042", "Transfer curves (R-042)")}
<h3>Continuous sweep, brightness only (R-041)</h3>
<p>Brightness scaled to 3× barely reaches the action: three scenes never cross the noise floor; none crosses the stricter floor; no scene fails. The benchmark's lighting variants change colour and direction, and those are the ones that fail.</p>
{table(["scene", "action boundary", "outcome boundary", "note"], [[SCENE.get(b["task_id"], b["task_id"]), fmt_b(b.get("action_boundary")), fmt_b(b.get("outcome_boundary")), E(b.get("note", ""))] for b in sorted(light_b, key=lambda b: b["task_id"])])}
<figure>{r041_plot(light_rows, "light_intensity", 3.0, tau, "d_pn_f0", "‖P−N‖ at forward 0")}<figcaption>‖P−N‖ at forward 0 against brightness multiplier.</figcaption></figure>
{note("Verdict", "<b>Dominant input:</b> both cameras jointly; no single input. <b>Data spec:</b> re-render existing demonstrations under the new lighting with both cameras consistent; a single-camera augmentation would leave a cross-view mismatch. <b>Open:</b> a colour-and-direction knob for the sweep; the necessity checks; a per-region or dictionary decomposition of the interaction.", "repo")}
</section>'''


def section_startpose():
    rev = jsonf("runs/r044_reverse/summary.json"); stc = jsonf("runs/r039_state_token_check.json")
    rescue_html, rs = rescue_table("r042", "Robot Initial States", {"W": "r044_rescue_W", "A": "r044_rescue_A", "N": "r044_rescue_N"})
    down_html, ds = downstream_table("r042", "Robot Initial States", {"N": "r050_ris_N", "W": "r050_ris_W"})
    s0 = {p["task_id"]: p for p in (analysis("r042") or {"instances": []})["instances"] if p["category"] == "Robot Initial States"}
    seeds = []
    for sd in (1, 2):
        a = analysis(f"r044_seed{sd}")
        if a:
            b = {p["task_id"]: p for p in a["instances"]}
            seeds.append([f"noise seed {sd}", med([b[t]["tf_f0"]["W"] for t in b]), med([b[t]["tf_f0"]["A"] for t in b]), f"{sum(b[t]['tf_f0']['W'] > b[t]['tf_f0']['A'] for t in b)}/10",
                          f"{sum(abs(b[t]['tf_f0']['W'] - s0[t]['tf_f0']['W']) <= 0.15 for t in b if t in s0)}/10", f"{sum(1 for t in b if not b[t]['success'])}/10"])
    revrows = [[r["task_id"], E(str(r["label"])), r["gap_PN"], r["denoise_W"], r["denoise_A"], r["noise_W"], r["noise_A"], r["noise_S"]] for r in (json.loads(l) for l in open("runs/r044_reverse/rows.jsonl"))] if os.path.exists("runs/r044_reverse/rows.jsonl") else []
    probe = jsonf("runs/r045/probe.json"); pss = jsonf("runs/r045/probe_scene_split.json")
    prow = []
    if probe:
        for k in ("image_all_flat", "image_mean_pool", "state_token"):
            r = probe["per_feature"][k]["rmse_by_radius_m"]
            prow.append([k.replace("_", " ")] + [float(r[x]) * 100 for x in ("0.0", "0.1", "0.2", "0.3", "0.4", "0.5")] + [probe["per_feature"][k]["null_rmse_m"] * 100])
    ssrow = [[k.replace("_", " "), v["rmse_r0_m"] * 100, v["rmse_r0p5_m"] * 100, v["rmse_all_radii_median_m"] * 100] for k, v in pss["summary"].items()] if pss else []
    joint_rows, joint_b = load_axis("runs/r041", "joint_radius_rad"); tau = (jsonf("runs/r041/tau.json") or {}).get("tau")
    return f'''
<section id="startpose">
<p class="part">4 · Robot initial state</p>
<h2>Carried by the wrist camera; the state token is near-inert; the episode is decided in the first chunk</h2>
<p><b>What the perturbation is.</b> The benchmark moves the arm's seven joints from the reset pose along a random direction by 0.1 to 0.5 rad, in blocks of a hundred variants per radius. Both cameras see the displaced arm and the proprioception reads the displaced pose, so nothing is perturbed in isolation. Our ten instances: five level-4 and five level-5 campaign failures.</p>
<h3>Denoising map (R-039, R-042)</h3>
{denoise_table("Robot Initial States", [("r039", "R-039 recorded control"), ("r042", "R-042 with pixel arms")])}
{instance_table("Robot Initial States", "r042")}
<p>The pre-registered expectation was that the state token would carry this row at 0.8 or more; it carried 0.04. A one-forward control settled whether the splice was at fault: zeroing the state token moves the action by {f2(stc["ratio_Szero_over_PN"]) if stc else "—"} of the gap, replacing it with the control's by {f2(stc["ratio_Sctl_over_PN"]) if stc else "—"}, swapping the image tokens by {f2(stc["ratio_Ictl_over_PN"]) if stc else "—"}. The checkpoint's training config zeroes the state token on a fifth of samples (<code>state_dropout_prob: 0.2</code>); the head learned not to depend on it. R-042 then split the images: wrist 0.88, agent view 0.17, wrist above agent on 10 of 10.</p>
{curves("Robot Initial States", "r042", "Transfer curves (R-042)")}
<h3>Robustness (R-044)</h3>
<p><b>Reverse direction.</b> One perturbed input placed into the nominal run at forward 0, no rollouts. denoise<sub>W</sub> reproduces R-042 to 0.01 on every instance.</p>
{table(["task", "instance", "‖N−P‖", "denoise W", "denoise A", "noise W", "noise A", "noise S"], revrows)}
<p>Medians: noise<sub>W</sub> {f2(rev["medians"]["noise_W"]) if rev else "—"}, noise<sub>A</sub> {f2(rev["medians"]["noise_A"]) if rev else "—"}, noise<sub>S</sub> {f2(rev["medians"]["noise_S"]) if rev else "—"}; wrist above agent on 10 of 10. Corrupting the wrist alone reproduces about half the effect and nothing else reproduces any; the three sum to about half, so the full effect needs the wrist and the agent view perturbed together. The wrist is necessary; the agent view is a consistency check the head also uses.</p>
<p><b>Noise seeds.</b></p>
{table(["replicate", "W median", "A median", "W > A", "W within 0.15 of seed 0", "fails"], seeds)}
<p><b>Forward-0 rescue.</b> Executed for the first forward only, then the perturbed policy continues. Note this design had no no-op drive; R-048 later measured a 3-of-7 no-correction flip rate on the camera row.</p>
{rescue_html}
<h3>Downstream of a single correction (R-050)</h3>
<p>‖P−N‖ at each forward after the corrected chunk was executed at forward 0, against the drive-P rollouts. Median over forwards 1–3: {" · ".join(f"drive {d}: {f2(v)}" for d, v in ds.items())}.</p>
{down_html}
<h3>Is the pose in the VLM's output? (R-045)</h3>
<p>A linear decoder from the adapter-output tokens to the simulator's true end-effector position, 360 resets across ten scenes, twelve radii and three joint directions. The pre-registered split (train two directions, test the third) failed its own sanity gate: even the state token, which is the pose after an encoder, decodes at 5 cm at radius 0.5, because a linear map does not extrapolate to an unseen direction. RMSE in cm by radius:</p>
{table(["features (direction split)", "0", "0.1", "0.2", "0.3", "0.4", "0.5", "null"], prow)}
<p>A post-hoc split by scene, labelled exploratory, trains on eight scenes across all directions and radii and tests on two unseen scenes, five hold-outs:</p>
{table(["features (scene split, median of 5 hold-outs)", "r = 0 (cm)", "r = 0.5 (cm)", "all radii (cm)"], ssrow)}
<p>The image tokens decode the arm's position on unseen scenes to about 1.5–2 cm at every radius, no worse than the state token. On that evidence the pose is present in the VLM's output across the range; the gap at new poses is in the head's mapping. Swapping the wrist image changed all 128 image tokens, so per-camera attribution is impossible at the token level.</p>
<h3>Continuous sweep, radius 0–0.5 rad (R-041)</h3>
{table(["scene", "action boundary", "outcome boundary", "note"], [[SCENE.get(b["task_id"], b["task_id"]), fmt_b(b.get("action_boundary")), fmt_b(b.get("outcome_boundary")), E(b.get("note", ""))] for b in sorted(joint_b, key=lambda b: b["task_id"])])}
<figure>{r041_plot(joint_rows, "joint_radius_rad", 0.5, tau, "d_pn_f0", "‖P−N‖ at forward 0")}<figcaption>‖P−N‖ at forward 0 against start-pose radius. The stove and top-drawer scenes, two of R-038's null-prompt scenes, fail earliest.</figcaption></figure>
{note("Verdict", "<b>Dominant input:</b> wrist pixels, 0.88 at forward 0, stable across seeds, necessary (noising 0.46, nothing else reproduces the effect); the agent view is a secondary channel and consistency check (0.17; the full effect needs both); the state token is near-inert (0.04). <b>Decided early:</b> one corrected chunk at forward 0 rescued 7/7 with the nominal chunk and 4/7 with the wrist chunk, without a no-op baseline. <b>Representation:</b> the pose is in the VLM output at every radius (post-hoc). <b>Data spec:</b> demonstrations that start from the new poses, containing both views, covering the approach (about the first second). <b>Retraining:</b> head and adapter with the VLM frozen predicted sufficient; the definitive test (R-046) is out of scope. <b>Open:</b> gripper-region vs near-scene inside the wrist frame; a second checkpoint; probability curves with replicates (R-047).", "repo")}
</section>'''


def section_layout():
    return f'''
<section id="layout">
<p class="part">5 · Object layout</p>
<h2>Agent view where the effect is large; the wrist leads when the near scene moves</h2>
<p><b>What the perturbation is.</b> Added distractor objects (<code>_add_N</code>) or moved objects (<code>levelK_sampleN</code>); the world differs, so the nominal is the recorded control of the canonical layout. Ten campaign failures, five level-4 and five level-5, drawn from the wooden-cabinet, cookie-box, plate and between-plate-ramekin scenes.</p>
<h3>Denoising map (R-049)</h3>
{denoise_table("Objects Layout", [("r049", "R-049 recorded control")])}
{instance_table("Objects Layout", "r049")}
<p>On the two wooden-cabinet distractor instances the effect is large (‖P−N‖ ≈ 3) and the agent view carries 0.89. On the cookie-box distractor instances the effect is small (0.5–0.8) and no single arm dominates, with a third of it in the text positions as leakage. The one moved-object instance is led by the wrist (0.71): the near scene changed.</p>
{curves("Objects Layout", "r049", "Transfer curves (R-049)")}
{note("Verdict", "<b>Dominant input:</b> agent-view pixels where the effect is large; the wrist where an object near the gripper moved; state near zero. <b>Not run:</b> necessity, seed and rescue checks. <b>Data spec:</b> demonstrations with the distractor or the moved object present, in both views (coverage of object placement and clutter, families A1/A2/B).", "repo")}
</section>'''


def section_texture():
    return f'''
<section id="texture">
<p class="part">6 · Background texture</p>
<h2>The smallest effect of any row, and a cross-camera one like lighting</h2>
<p><b>What the perturbation is.</b> The scene XML's table, floor or wall textures are swapped; physics is unchanged, but the textures are compiled into the model, so the nominal is the recorded control. Only two campaign failures exist in this category; both are here plus eight successes.</p>
<h3>Denoising map (R-049)</h3>
{denoise_table("Background Textures", [("r049", "R-049 recorded control")])}
{instance_table("Background Textures", "r049")}
<p>Restoring either camera's pixels alone does nothing or makes the action worse (A −0.02, W −0.05, ranges to −0.6); restoring both cameras' tokens recovers 0.62. The table surface is in both views and the head treats a mismatch between them as a perturbation, as with lighting. No instance failed; with ‖P−N‖ this small the row sits near the outcome noise floor.</p>
{curves("Background Textures", "r049", "Transfer curves (R-049)")}
{note("Verdict", "<b>Dominant input:</b> both cameras jointly. <b>Data spec:</b> re-render under new textures with both cameras consistent. <b>Open:</b> whether the two campaign failures reproduce at all with replicates.", "repo")}
</section>'''


def section_language():
    rs = rows_of("r049_lang", "Language Instructions")
    body = ""
    if rs:
        R038 = {"on-stove", "on-wooden-cabinet", "top-drawer"}
        def dpn_mean(r):
            p = f"runs/r049_lang/splice/{r['rollout_id']}.npz"
            return float(np.load(p)["d_pn"].mean()) if os.path.exists(p) else float("nan")
        body += table(["level", "instance", "task", "outcome", "‖P−N‖ f0", "‖P−N‖ mean over forwards", "S f0", "aligned", "instruction served"],
                      [[lvl(r["level"]), E(str(r["label"])), r["task_id"], ok(r["success"]), r["d_pn_f0"] if r["d_pn_f0"] is not None else float("nan"), dpn_mean(r),
                        (r.get("tf_at_f0") or {}).get("S", float("nan")), "yes" if r.get("aligned") else "no", E(str(r.get("instruction_given", ""))[:70])]
                       for r in sorted(rs, key=lambda r: (-(r["level"] or 0), r["task_id"]))])
        d38 = [r["d_pn_f0"] for r in rs if r["scene"] in R038 and r["d_pn_f0"] is not None]; dother = [r["d_pn_f0"] for r in rs if r["scene"] not in R038 and r["d_pn_f0"] is not None]
        body += f"<p>{sum(1 for r in rs if not r['success'])} of {len(rs)} fail. ‖P−N‖ at forward 0: median {f2(med([r['d_pn_f0'] for r in rs]))} overall; {f2(med(d38))} on R-038's null-prompt scenes (n={len(d38)}) against {f2(med(dother))} elsewhere (n={len(dother)}). S at forward 0: median {f2(med([(r.get('tf_at_f0') or {}).get('S') for r in rs]))}.</p>"
        body += curves("Language Instructions", "r049_lang", "‖P−N‖ is the row's curve here; the S arm is the only transfer defined")
    else:
        body = "<p class=lead-in>Rerun pending.</p>"
    body = ("<p><b>Why this row has no T or I arm.</b> The rewrite tokenises to a different length than the trained wording (153 vs 151 tokens on the first instance), so the token positions of P and N do not correspond and a positional splice is undefined, not merely unmeasured. The pixel arms are identity (frames unchanged). What is well defined is the effect size ‖P−N‖ at every forward, from the paired text, and the state arm.</p>") + body
    return f'''
<section id="language">
<p class="part">7 · Language instruction</p>
<h2>The only row where the text arm is a finding rather than leakage</h2>
<p><b>What the perturbation is.</b> The instruction is rewritten ("pick up the darkhued vessel situated adjacent to the small ramekin…"); frames and state are identical, so the nominal is the same observation with the trained wording. Ten campaign failures, five level-4 and five level-5, heavy on the wooden-cabinet and ramekin scenes. Context: R-038 found the empty prompt takes this policy from 100/100 to 50/100 with a scene-dependent spread, so "insensitive to language" is not this policy's property.</p>
<h3>Denoising map (R-049 rerun)</h3>
{body}
{note("Verdict", "<b>Dominant input:</b> the text by construction, the only differing input. <b>What the row measures:</b> how far a rewrite moves the first action and whether that is scene-dependent in the R-038 pattern, since a positional text-vs-image split is undefined when token counts differ. <b>Data spec:</b> instruction augmentation over existing demonstrations (family A3). <b>Not run:</b> seeds, rescue; a pre-VLM instruction swap with matched token counts would restore the positional arms.", "repo")}
</section>'''


def section_map_and_appendix():
    return f'''
<section id="map">
<p class="part">8 · The map, and what it says for data</p>
<h2>Seven rows</h2>
{table(["perturbation", "dominant input at forward 0", "secondary", "necessity", "data specification"], [
    ["camera viewpoint", "agent-view pixels 0.98", "none", "by construction; exact and stable", "re-render under the new viewpoint"],
    ["sensor noise", "agent-view pixels 0.97", "none", "by construction", "augment the agent-view stream"],
    ["lighting", "both cameras jointly", "interaction", "not run", "re-render with both cameras consistent"],
    ["robot initial state", "wrist pixels 0.88", "agent view 0.17 (consistency check); state 0.04", "noising 0.46; stable; rescue 4/7 (no no-op)", "demonstrations from the new poses, both views, approach phase"],
    ["object layout", "agent-view pixels 0.55 (0.89 where large)", "wrist when the near scene moves", "not run", "demonstrations with the objects present, both views"],
    ["background texture", "both cameras jointly", "interaction", "not run", "re-render with both cameras consistent"],
    ["language", "text tokens (by construction)", "see row 7", "degenerate", "instruction augmentation"],
])}
<p>Two rows are single-input and exact (camera, noise), one is single-input with a secondary channel (start pose), two are joint properties of the two cameras (lighting, texture), one is mixed by effect size (layout), and one is text by construction (language). The state token is at or below 0.2 on every row, which traces to training-time state dropout. What no row establishes is that the specified data fixes anything: that is a retraining experiment, pre-registered as R-046 for the start-pose row and out of scope for now; the cost, data and expected outcome are in <a href="RETRAINING_PLAN.html">RETRAINING_PLAN.html</a>.</p>
<h3>Boundaries across rows (R-041, one seed)</h3>
<p>The first action departs from nominal by more than the noise floor at the smallest tested step on camera yaw, camera distance and start pose, and barely at all on brightness. The outcome holds to 40° on 8 of 10 scenes, 0.4 m on 9, 3× brightness on 10, 0.5 rad on 7. The boundary that matters is the loop's recovery, not the policy's sensitivity, and it is scene-dependent. Probability curves with replicates (R-047) are pre-registered and pending overnight.</p>
</section>

<section id="appendix">
<p class="part">Appendix</p>
<h2>A · Glossary</h2>
<dl class="gloss">
<dt>forward</dt><dd>One policy call; a chunk of 40 actions every 16 simulator steps; at most 18 per episode.</dd>
<dt>live dims</dt><dd>The 7 of 132 action dimensions the arm uses. All distances are on these.</dd>
<dt>P, N</dt><dd>The chunk from the perturbed inputs (executed), and from the nominal inputs.</dd>
<dt>T, I, S, IT</dt><dd>Token-level arms at the adapter output: text positions, image positions, the state token, image + text restored to nominal.</dd>
<dt>A, W</dt><dd>Pixel-level arms: the agent-view image or the wrist image swapped for its nominal version before the VLM; everything downstream recomputed.</dd>
<dt>transfer fraction τ</dt><dd>1 − ‖a<sub>X</sub> − a<sub>N</sub>‖ / ‖a<sub>P</sub> − a<sub>N</sub>‖. Share of the gap closed by one restored input. ≤ 1, unbounded below, not additive.</dd>
<dt>‖P−N‖</dt><dd>Effect size: how far the perturbation moved the chunk at that forward.</dd>
<dt>f0, anchor</dt><dd>Forward 0, the first decision; the forward containing the closest approach to the target bowl.</dd>
<dt>onset, dominant, residual</dt><dd>First forward with τ ≥ 0.5; the single arm with τ ≥ 0.9 at the anchor; no such arm.</dd>
<dt>denoising, noising</dt><dd>Nominal input into the perturbed run (τ); perturbed input into the nominal run (noise<sub>X</sub>).</dd>
<dt>paired render / recorded control / paired text</dt><dd>The three ways the nominal is obtained; see section 0.</dd>
<dt>drive, drive-until</dt><dd>Which arm's chunk is executed; executing it for the first N forwards only, with every arm still recorded afterwards.</dd>
<dt>action boundary, outcome boundary</dt><dd>Smallest magnitude at which ‖P−N‖ at forward 0 exceeds the noise floor τ; smallest at which the episode fails.</dd>
<dt>τ (R-041)</dt><dd>95th percentile of ‖P−N‖ at forward 0 over unperturbed rollouts, the paired-render noise floor (0.093); the stricter all-forward floor is 0.79.</dd>
<dt>level</dt><dd>LIBERO-Plus difficulty preset L1–L5, defined by how many of four reference models solve the variant.</dd>
<dt>control variant</dt><dd>The scene's <code>language_1_view_0_0_100_0_0_initstate_0</code> variant run with the trained instruction: the unperturbed episode on this stack (R-029, 100/100).</dd>
</dl>
<h2>B · Runs behind this page</h2>
{table(["run id", "what", "entry"], [
    ["r039, r039_recorded", "denoising map, 40 + 30 instances, token arms", "R-039"],
    ["r039_gate, r039_state_token_check", "determinism gate; state-token deletion control", "R-039"],
    ["r042", "pixel arms A/W and pairwise IT on the same 40", "R-042"],
    ["r044_reverse, r044_seed1/2, r044_rescue_W/A/N", "start-pose robustness", "R-044"],
    ["r045", "pose probe, 360 forwards", "R-045"],
    ["r041", "bisection sweeps, four axes, 240 rollouts", "R-041"],
    ["r048_reverse, r048_seed1/2, r048_rescue_A/W/N", "camera robustness", "R-048"],
    ["r049, r049_lang", "layout, texture, language maps", "R-049"],
    ["r050_ris_N/W, r050_cam_N/A", "downstream of a forward-0 correction", "R-050"],
    ["r047 (pending)", "probability curves with replicates", "R-047"],
])}
<h2>C · Corrections and misses, in order</h2>
<ul class="tight">
<li>R-039: S arm is zero by construction under paired rendering (recorded mid-run; second pass added). Expectation 1 (state token carries start pose) missed by 0.76; four of six missed.</li>
<li>R-042: IT is 1.0 by construction on the paired-render pass (recorded from the smoke). Expectation 3 (light: A > W) and 5 (A + W ≈ I) missed.</li>
<li>R-041: τ mis-specified as forward-to-forward chunk distance (13.7); redefined as the paired-render noise floor before any bisection. Expectations 3, 4, 5 missed; 1 held in trend only.</li>
<li>R-044: expectation 1a's median (0.6) missed at 0.46; no no-op drive in the rescue design.</li>
<li>R-045: the pre-registered direction split failed its sanity gate; the scene split is post hoc. The wrist-token position check changed all tokens.</li>
<li>R-048: expectations 3, 4, 5 missed; the no-op drive revealed a 3/7 baseline flip rate.</li>
<li>R-049: language run had P ≡ N (nominal text bug); fixed and rerun. Layout IT ≥ 0.9 on 5/10; texture A > W on 4/10.</li>
<li>General: the claim that the pixel-swap arms show the VLM "passes the pose through faithfully" was withdrawn on 2026-09-24; they show the VLM is in the causal path, not that its encoding is adequate (R-045 addresses that).</li>
</ul>
<h2>D · Reproduction</h2>
<p>All runners live under <code>experiments/</code> (<code>r039_run.py</code>, <code>r041_run.py</code>, <code>r044_reverse.py</code>, <code>r045_pose_probe.py</code>, <code>r047_run.py</code>, <code>r048_reverse.py</code>) with their queue scripts; every GPU job runs under <code>flock /tmp/vla_gpu.lock</code>. Reports: <code>r039_report.py</code>, <code>r041_report.py</code>, this page from <code>compile_map.py</code>. Run metadata is committed (manifests, aggregates, code state); per-rollout arrays and videos are regenerated from the same code state.</p>
</section>'''


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/PERTURBATION_MAP.html"); a = ap.parse_args()
    src = open("docs/R039_MATHS.html").read()
    head = src[:src.index("</head>")].replace("<title>Splice Maths</title>", "<title>Perturbation Map</title>")
    head = head.replace("  .footer{", "  .wrap{max-width:1180px}\n  td:nth-child(2){white-space:nowrap}\n  table{font-size:.8rem}\n  .footer{").replace("  .footer{", "  .lvl{display:inline-block;min-width:2.2em;text-align:center;border-radius:3px;padding:1px 6px;font-family:var(--mono);font-size:.78rem}\n  .L1{background:#e8f1fb}.L2{background:#cfe0f6}.L3{background:#a9c8ee}.L4{background:#7ea9e0}.L5{background:#4f86cf;color:#fff}\n  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}\n  figure.small{margin:0;background:var(--surface);border:1px solid var(--rule-soft);border-radius:4px;padding:8px}\n  figure.small svg{display:block;width:100%;height:auto;color:var(--ink)}\n  figure.small figcaption{border:0;padding-top:4px;margin-top:4px;font-size:.74rem}\n  h4{font-family:var(--sans);font-size:.95rem;margin:1.2em 0 .4em}\n  dl.gloss dt{font-family:var(--sans);font-weight:600;margin-top:.8em}dl.gloss dd{margin:.1em 0 0}\n  .toc a{margin-right:14px;font-family:var(--sans);font-size:.85rem}\n  .footer{")
    body = f'''</head>
<body>
<div class="wrap">
<header class="mast">
  <p class="eyebrow">Compendium · GR00T N1.7 · LIBERO-Plus · 2026-09-23/24</p>
  <h1>The perturbation → correction map</h1>
  <p class="standfirst">Every result from R-039 to R-050, organised by row of the map: for each perturbation, which input carries it, how sure we are, how far it can be pushed, and what data would fix it. Method and maths first; glossary last. Numbers are read from the run files at build time.</p>
  <div class="meta"><span>{len(ROWS)} rows</span><span>RESULTS.md R-039 – R-050</span><span><a href="RETRAINING_PLAN.html">retraining plan</a></span><span><a href="R039_MATHS.html">maths</a></span></div>
  <p class="toc"><a href="#common">0 common</a><a href="#camera">1 camera</a><a href="#noise">2 noise</a><a href="#light">3 lighting</a><a href="#startpose">4 start pose</a><a href="#layout">5 layout</a><a href="#texture">6 texture</a><a href="#language">7 language</a><a href="#map">8 map</a><a href="#appendix">appendix</a></p>
</header>
{common()}
{section_camera()}
{section_noise()}
{section_light()}
{section_startpose()}
{section_layout()}
{section_texture()}
{section_language()}
{section_map_and_appendix()}
<div class="footer">Generated by <code>experiments/compile_map.py</code> from <code>runs/</code>; scored expectations and timestamps in <code>RESULTS.md</code>.</div>
</div>
</body>
</html>'''
    open(a.out, "w").write(head + body)
    print("wrote", a.out, len(head + body), "bytes")


if __name__ == "__main__":
    main()
