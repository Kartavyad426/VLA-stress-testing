"""Build the retraining visualisation page from a runs config.

  python3 .claude/skills/retraining-viz/scripts/build.py \
      --config .claude/skills/retraining-viz/scripts/configs/start_pose_pilot.json \
      --out docs/RETRAINING_VIZ.html [--read-rollouts] [--compare docs/RETRAINING_VIZ.html]

CPU only, stdlib only. Runs from any cwd (chdirs to the repo root). Adding a run means
adding a config entry and re-running: the template has no run names in it.

--read-rollouts  allow the one panel that needs rollouts.jsonl (achieved t=0 gripper offset).
                 Off by default (metadata-only rule).
--compare FILE   diff the embedded data blob of the new page against FILE's and print every
                 difference (path: old -> new). Exit code 1 if they differ.
"""
from __future__ import annotations

import argparse
import re
import datetime
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import extract  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))


def f2(x):
    return "–" if x is None else f"{x:.2f}"


def f3(x):
    return "–" if x is None else f"{x:.3f}"


def jb(v):  # JS-style bool / null
    return {True: "true", False: "false", None: "null"}.get(v, str(v))


def code(p):
    return f"<code>{p}</code>"


# ------------------------------------------------------------------ inventory rows
def inventory_rows(cfg, blob):
    inv, ck, loss, gates = blob.get("inventory", {}), blob.get("checkpoints", {}), blob.get("loss", {}), blob.get("gates", {})
    rows = []
    for r in cfg.get("train_runs", []):
        k = r["key"]
        ckname = os.path.basename(os.path.dirname(r["ckpt_dir"].rstrip("/"))) if r.get("ckpt_dir") else None
        cks = ck.get(ckname, [])
        ck_txt = ", ".join(f"{c['opt_step']}←{c['dir']}" for c in cks) or "not saved"
        sel = None
        if r.get("eval"):
            sp = os.path.join(r["eval"], "selection.json")
            if os.path.exists(sp):
                sel = json.load(open(sp))
        status = r.get("status", "")
        if sel:
            status = f"{status} (selected {sel['label']}, val {sel['success']}/{sel['n']})"
        g = gates.get(r.get("gate", ""), {})
        if g:
            status = (f"ratio {f3(g.get('ratio', g.get('final_ratio')))}; pass={jb(g.get('pass'))}" if "ratio" in g
                      else f"diagnostic; final ratio {f3(g.get('final_ratio'))}") + (f" · {r['note']}" if r.get("note") else "")
        if k in inv and r.get("inventory") == "report":
            d = inv[k]
            rows.append([r["row"], r["arm_label"], f"{d['r']} / {d['alpha']}", f"{d['batch']} × {d['accum']} = {d['eff']}",
                         r.get("lr_text") or f"{d['lr']}, {d['sched']} (warmup {d['warmup'] * 100:g}%), verified={jb(d['verified'])}",
                         f"{d['opt_steps']} → {cks[-1]['opt_step'] if cks else '–'}", f"{d['trainable']:,}",
                         f"{f2(d['alloc'])} / {f2(d['smi'])}", ck_txt, status])
        elif k in inv and r.get("inventory") == "ckpt_meta":
            d = inv[k]
            meta = json.load(open(os.path.join(r["ckpt_dir"], cks[-1]["dir"], "pretrained_model", "vla_train_meta.json")))
            s = meta.get("schedule", {})
            last = loss.get(k, {}).get("rows", [[None]])[-1][0]
            trainable = "–"
            if r.get("log") and os.path.exists(r["log"]):  # trainable count: the run's own [groot_lora] line
                m = re.search(r"trainable parameters: ([\d,]+) of", open(r["log"], errors="replace").read())
                trainable = m.group(1) if m else "–"
            rows.append([r["row"], r["arm_label"], f"{d['r']} / {d['alpha']}", f"{d['batch']} × {d['accum']} = {d['eff']}",
                         f"{s.get('lr', '–')}, {s.get('type', '–')} (warmup {s.get('warmup_ratio', 0) * 100:g}%), verified={jb(d['verified'])}",
                         f"{d['opt_total']} → {last} (last log), ckpt {cks[-1]['opt_step'] if cks else '–'}", trainable, "–", ck_txt, status])
        elif r.get("report") and os.path.exists(r["report"]):
            d = json.load(open(r["report"]))
            a = d.get("args", {})
            rows.append([r["row"], r["arm_label"], f"{d['lora']['r']} / {d['lora']['alpha']}" if "lora" in d else "16 / 16 (default)",
                         f"{d['batch_size']} × {d['accumulation']} = {d['effective_batch']}",
                         f"{a.get('lr')}, {a.get('lr_schedule', '–')}", f"{d['opt_steps']}", f"{d['trainable']['trainable']:,}",
                         f"{f2(d.get('torch_max_memory_allocated_gib'))} / {f2(d.get('nvidia_smi_peak_gib'))}", ck_txt, status])
        elif g:
            rows.append([r["row"], r["arm_label"], "–", "–", g.get("lr_schedule", "–"), str(r.get("opt_steps", "–")), "–", "–", ck_txt, status])
    for f in cfg.get("failed_runs", []):
        d = blob.get(f["key"])
        if d:
            rows.append([f["row"], f["arm_label"], "planned:", f.get("planned", "–"), "", f"0 (micro {d['last_micro']})",
                         d["trainable"][0] if d["trainable"] else "–", "–", "none", "CUDA OOM" if d["oom"] else "failed"])
    head = ["Run", "Arm", "LoRA r / α", "micro-batch × accum = eff. batch", "LR, schedule", "Opt steps (planned → done)",
            "Trainable params", "GPU peak (torch / smi, GiB)", "Checkpoints (opt step ← dir)", "Status"]
    return head, rows


def gate_rows(blob):
    G = blob.get("gates") or {}
    rows = []
    if "conformance" in G:
        c = G["conformance"]; rows.append(["Zero-step conformance (untrained adapter = base)", f"{c['pass_count']}/{c['n']} instances within tol", f"tol {c['tol']}", jb(c["pass"])])
    if "ckpt_equiv_r16" in G:
        c = G["ckpt_equiv_r16"]; rows.append(["Gradient checkpointing equivalence (r=16)", f"loss {f3(c['loss_plain'])} = {f3(c['loss_ckpt'])}; worst grad rel {c['grad_worst_rel']:.4f}", f"rel {c['tol_rel']}", jb(c["pass"])])
    if "memory" in G:
        m = G["memory"]
        rows.append(["Memory, batch 1 (VLM resident)", f"{f2(m['b1_alloc'])} GiB torch / {f2(m['b1_smi'])} GiB smi", "≤ 7.4 GiB", "true"])
        rows.append(["Memory, batch 2 (readout)", f"{f2(m['b2_alloc'])} GiB torch / {f2(m['b2_smi'])} GiB smi", "≤ 7.8 GiB", "true"])
    for k, lab in (("overfit", "Overfit, 300 steps, constant LR"), ("overfit_cosine", "Overfit, 300 steps, cosine LR")):
        if k in G:
            o = G[k]; rows.append([lab, f"fixed-seed loss {f3(o['loss_step0'])} → {f3(o['loss_final'])} (ratio {f3(o['ratio'])})", "ratio < 0.10", jb(o["pass"])])
    if "overfit_diag" in G:
        rows.append([f"Overfit diagnostic, {G['overfit_diag']['steps']} steps", f"final ratio {f3(G['overfit_diag']['final_ratio'])}", "not a gate", "–"])
    if "cache_equiv_r64" in G:
        c = G["cache_equiv_r64"]; rows.append(["Cached-feature equivalence (r=64)", f"worst rel {c['worst_rel']}", f"rel {c['tol_rel']}", jb(c["pass"])])
    return rows


# ------------------------------------------------------------------ meta + sources
def build_meta(cfg, blob, skipped, cfg_path):
    evals = cfg.get("evals", [])
    base_name = os.path.basename(cfg["base_eval"].rstrip("/")) if cfg.get("base_eval") else None
    base_arm = next((e.get("base_arm") for e in evals if e.get("base_arm")), "base")
    ev_dirs = [e["dir"] for e in evals]
    train_meta, src_list = [], []
    for r in cfg.get("train_runs", []):
        ckname = os.path.basename(os.path.dirname(r["ckpt_dir"].rstrip("/"))) if r.get("ckpt_dir") else None
        sel = None
        if r.get("eval") and os.path.exists(os.path.join(r["eval"], "selection.json")):
            sel = json.load(open(os.path.join(r["eval"], "selection.json")))["step"]
        n_lines = len(blob.get("loss", {}).get(r["key"], {}).get("rows", []))
        fmt = '"[cached] opt step N loss X lr Y"' if r.get("log_format") == "cached" else "fields <code>loss:</code> <code>grdn:</code> <code>lr:</code>"
        src = f"Loss{'' if r.get('no_grad') else ', grad norm'} and logged LR: {code(r.get('log', '–'))} ({n_lines} log lines, {fmt})."
        if r.get("lr_trace_from_report"):
            src += f" Applied LR per opt step: {code(r['report'])} → <code>lr_trace.applied</code>."
        if ckname:
            src += f" Checkpoint markers: {code(r['ckpt_dir'] + '/*/pretrained_model/vla_train_meta.json')} → <code>opt_step</code>."
        if r.get("curve_json"):
            src += f" Train-time eval curve: {code(r['curve_json'])} → <code>eval[]</code>."
        if r.get("gate"):
            src += f" Fixed-seed loss before/after: {code(cfg.get('gates'))} → <code>{extract.GATE_KEYS.get(r['gate'], (r['gate'],))[0]}</code>."
        if r.get("kind") == "smoke":
            continue
        train_meta.append({"key": r["key"], "arm": r["arm"], "title": r["title"], "ck": ckname, "sel": sel,
                           "noGrad": bool(r.get("no_grad")), "short": bool(r.get("short")), "overlay": bool(r.get("overlay")),
                           "curve_key": r.get("curve_blob_key"), "endpoints": r.get("gate") if r.get("gate") in ("overfit", "overfit_cosine") else None,
                           "opt_steps": r.get("opt_steps"), "has_log": bool(r.get("log")), "src_html": src})
        for p, what in ((r.get("log"), f"{r['row']}: training log, loss/grdn/lr ({n_lines} lines)"),
                        (r.get("report"), f"{r['row']}: train report (recipe, lr_trace, parts/shares, memory)"),
                        (r.get("curve_json"), f"{r['row']}: train-time eval curve"),
                        ((r["ckpt_dir"] + "/*/pretrained_model/vla_train_meta.json") if r.get("ckpt_dir") else None, f"{r['row']}: opt_step / micro_step per checkpoint")):
            if p:
                src_list.append([p, what])
    smoke = [{"key": r["key"], "title": r["title"], "src_html": f"Source: {code(r['report'])} (<code>eval_loss_step0</code>, <code>eval_loss_final</code>, <code>opt_steps</code>, <code>effective_batch</code>, <code>parts</code>)"}
             for r in cfg.get("train_runs", []) if r.get("kind") == "smoke"]
    failed = []
    for f in cfg.get("failed_runs", []):
        failed.append({"key": f["key"], "title": f["title"], "note": f.get("note", ""), "total_micro": f.get("total_micro"),
                       "src_html": "Source: " + ", ".join(code(p) for p in [f["log"]] + f.get("extra_sources", []))})
        src_list.append([f["log"], f"{f['row']}: last tqdm position, OOM traceback, trainable-parameter line"])
        for p in f.get("extra_sources", []):
            src_list.append([p, "run status"])
    head, rows = inventory_rows(cfg, blob)
    if cfg.get("gates"):
        src_list.append([cfg["gates"], "gate table; overfit endpoints loss_step0/loss_final"])
    for d in ev_dirs:
        src_list.append([f"{d}/score.json", "heldout, nominal, per_band, paired, gap_closed, transfer, expectations, uninterpretable_checks, wise, exploratory_gripper_offset"])
        src_list.append([f"{d}/selection.json", "validation success/n/median_ca per checkpoint; selected label; rule"])
        src_list.append([f"{d}/manifest.jsonl", "per-scene held-out rows (set, scene, success, closest_approach_m); start keys checked unique"])
        if os.path.exists(f"{d}/transfer/rows.jsonl") and os.path.getsize(f"{d}/transfer/rows.jsonl"):
            src_list.append([f"{d}/transfer/rows.jsonl", "denoise_W per transfer instance"])
    for d in cfg.get("manifest_counts", []):
        if d not in ev_dirs:
            src_list.append([f"{d}/manifest.jsonl", "(ckpt, set) success counts"])
    read_ro = "offset" not in skipped
    if read_ro:
        src_list.append([", ".join(f"{d}/rollouts.jsonl" for d in ev_dirs), "steps[0].obs_state.eef_pos for the achieved-offset chart (read because --read-rollouts / read_rollouts was set)"])
        src_list.append([cfg["nominal_starts"], "nominal eef_pos per scene (offset reference)"])
    dg = cfg.get("diag") or {}
    for k, what in (("loss_summary", "fixed-seed loss per subset × model (mean, se, median, diff_vs_base_ci95, share_frames_below_base)"),
                    ("fit", "first-chunk distance: to_label/to_nominal by_cm and all, heldout_matched.all"),
                    ("fit_rows", "per-start first-chunk rows (kind=train)"), ("offset_curve", "base success per achieved-offset band, overall and by direction")):
        if dg.get(k):
            src_list.append([dg[k], what])
    dc = cfg.get("data") or {}
    if dc:
        src_list += [[f"{dc['dataset']}/meta/provenance.jsonl", "kept demos: scene_task_id, band, noise_seed, source, n_frames, n_driven, outcome.steps"],
                     [f"{dc['dataset']}/meta/export.json", "episodes, frames, truncate_steps"],
                     [", ".join(f"{d}/meta/info.json" for d in dc.get("datasets", [])), "total_episodes, total_frames, fps"],
                     [dc["mint_manifest"], "mint attempts per scene (scene_task_id, success)"],
                     [dc["yield"], "attempts / successes / yield per band and pooled"],
                     [dc["coverage"], "scenes, offset_bins, window_phases, gripper_closed_fraction"]]
    for k, p in (cfg.get("upstream") or {}).items():
        src_list.append([p, f"{k.upper()} upstream table/chart"])
    src_list += cfg.get("extra_sources", [])

    def exists(spec):  # "a, b" lists; "x/*/y" globs; "file:12-34" line refs; "file (note)" annotations
        for part in re.sub(r" \(.*$", "", spec).split(", "):
            part = re.sub(r":\d[\d,-]*$", "", part.strip())
            if not (glob.glob(part) if "*" in part else os.path.exists(part)):
                return False
        return True
    src_list = [[p, w if exists(p) else f"{w} — MISSING on disk, panel skipped"] for p, w in src_list]

    evs = ", ".join(code(f"{d}/score.json") for d in ev_dirs)
    S = {
        "inventory": "Sources: train reports (<code>lora</code>, <code>batch_size</code>, <code>accumulation</code>, <code>effective_batch</code>, <code>opt_steps</code>, <code>args.lr/lr_schedule/warmup_ratio</code>, <code>trainable</code>, memory peaks, <code>lr_live.schedule_verified</code>); checkpoint <code>vla_train_meta.json</code>; selection from <code>&lt;eval&gt;/selection.json</code>; status text from the config "
                     + code(os.path.relpath(cfg_path, REPO)) + ". Files: " + ", ".join(code(r["report"]) for r in cfg.get("train_runs", []) if r.get("report")),
        "gates": f"Source: {code(cfg.get('gates'))}",
        "over": "Sources: " + ", ".join(code(r["log"]) for r in cfg.get("train_runs", []) if r.get("overlay") and r.get("log")) + ". Medians computed on this page.",
        "val": "Sources: " + ", ".join(code(f"{d}/selection.json") for d in ev_dirs) + " (<code>all[].success/n/median_ca/step</code>)" +
               "".join(f", {code(x['mc'])} manifest ({x['key']})" for x in cfg.get("val_extra", [])),
        "ckl": f"Source: {code(dg.get('loss_summary'))} → <code>subsets.&lt;subset&gt;.&lt;model&gt;.{{mean,se}}</code>, pooled on this page (mean of subset means; SE = √Σse² / n_subsets). Model → checkpoint map from the config.",
        "rates": f"Source: {evs} → <code>heldout/nominal.{{base,retrained}}</code>; WiSE from <code>wise</code>.",
        "scene": "Computed from " + ", ".join(code(f"{d}/manifest.jsonl") for d in ev_dirs) + " rows with <code>set = heldout</code> (WiSE: <code>heldout_wise</code>), grouped by <code>scene</code>. Cell shade = success rate.",
        "band": f"Source: {evs} → <code>per_band.&lt;band&gt;.{{base,retrained}}</code>",
        "off": f"Computed per rollout as |t=0 <code>obs_state.eef_pos</code> − {code(cfg.get('nominal_starts'))}[scene].eef_pos| × 100, from <code>rollouts.jsonl</code> joined to the manifest by line position. The split at score.json's threshold is asserted equal to <code>exploratory_gripper_offset</code> at build time.",
        "pair": f"Source: {evs} → <code>paired.mcnemar.{{b,c,p_value}}</code>, <code>paired.ca_improved</code>, <code>gap_closed</code> (computed by the eval scorer). b = base ✓ arm ✗, c = the reverse.",
        "exp": f"Source: {evs} → <code>expectations</code>, <code>failure_criterion_gap_lt_one_third</code>, <code>uninterpretable_checks.base_heldout_vs_r047_curve.consistent</code>",
        "tr": "Source: " + ", ".join(code(f"{d}/transfer/rows.jsonl") for d in ev_dirs) + " (field <code>denoise_W</code>)",
        "d20l": f"Source: {code(dg.get('loss_summary'))} → <code>subsets.&lt;subset&gt;.&lt;model&gt;.{{mean,se,median,diff_vs_base_ci95}}</code>",
        "fit": f"Source: {code(dg.get('fit'))} → <code>to_label.by_cm.&lt;bin&gt;.&lt;model&gt;.median_to_ref</code>. Metric: <code>||a[:16,:3].sum(0) − ref||</code>, env action units.",
        "fitsc": f"Computed from {code(dg.get('fit_rows'))} (<code>kind = train</code>) as ‖<code>disp.own.&lt;model&gt;</code> − <code>label_disp</code>‖ against <code>offset_cm</code>.",
        "fittbl": f"Source: {code(dg.get('fit'))} → <code>to_label.all</code>, <code>to_nominal.all</code>, <code>heldout_matched.all</code>",
        "r047": f"Source: {code(dg.get('offset_curve'))} → <code>per_offset_band_cm</code>, <code>per_offset_band_cm_by_dir</code>.",
        "mix": f"Source: train report of <code>{dc.get('mix_run')}</code> → <code>parts[]</code>, <code>shares</code>, <code>opt_steps</code>, <code>effective_batch</code>",
        "mintscene": f"Kept: {code(dc.get('dataset', '') + '/meta/provenance.jsonl')}; attempted: {code(dc.get('mint_manifest'))}",
        "yield": f"Source: {code(dc.get('yield'))}",
        "offbins": f"Source: {code(dc.get('coverage'))} → <code>offset_bins.&lt;bin&gt;.{{attempts,kept,eval,eval_success}}</code>",
        "phase": f"Source: {code(dc.get('coverage'))} → <code>window_phases</code>, <code>gripper_closed_fraction</code>",
        "noise": f"Source: {code(dc.get('dataset', '') + '/meta/provenance.jsonl')} → <code>provenance.noise_seed</code>",
        "ds": "Source: " + ", ".join(code(d + "/meta/info.json") for d in dc.get("datasets", [])),
        "cov": f"Source: {code(dc.get('coverage'))} → <code>scenes</code>; held-out base successes from the base manifest",
        "r053": f"Source: {code((cfg.get('upstream') or {}).get('r053'))} → <code>per_seed</code>",
        "r057": f"Source: {code((cfg.get('upstream') or {}).get('r057'))} → <code>rows.{{ris,cam}}.by_k</code>",
    }
    built = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return {
        "title": cfg["title"], "subtitle": cfg.get("subtitle", "") + f" Built {built} from files on disk; every chart cites its source and the <a href=\"#sources\">Sources</a> section lists them all.",
        "config": os.path.relpath(cfg_path, REPO), "arms": cfg["arms"], "train_runs": train_meta, "smoke_runs": smoke, "failed_runs": failed,
        "inventory_head": head, "inventory_rows": rows, "gate_rows": gate_rows(blob),
        "evals": [{"score": os.path.basename(e["dir"].rstrip("/")), "arm": e["arm"]} for e in evals],
        "base": {"score": base_name, "pr": base_arm} if base_name else None,
        "wise": {"score": os.path.basename(cfg["wise"]["eval"].rstrip("/")), "arm": cfg["wise"]["arm"], "pr": cfg["wise"]["pr"]} if cfg.get("wise") else None,
        "pr_arm": cfg.get("pr_arm", {}), "val_extra": cfg.get("val_extra", []), "d20_models": dg.get("models", {}),
        "mix_run": dc.get("mix_run"), "outcomes_note": cfg.get("outcomes_note", ""), "transfer_note": cfg.get("transfer_note", ""),
        "src": S, "sources": src_list, "skipped": skipped,
    }


def blob_of(html):
    i = html.index('<script id="vizdata"')
    return json.loads(html[html.index(">", i) + 1: html.index("</script>", i)])


def diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a:
                out.append(f"{path}/{k}: <absent> -> present")
            elif k not in b:
                out.append(f"{path}/{k}: present -> <absent>")
            else:
                out += diff(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: list length {len(a)} -> {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append(f"{path}: {json.dumps(a)[:80]} -> {json.dumps(b)[:80]}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--read-rollouts", action="store_true")
    ap.add_argument("--compare")
    a = ap.parse_args()
    cfg_path = os.path.abspath(a.config)
    out_path = os.path.abspath(a.out)
    cmp_path = os.path.abspath(a.compare) if a.compare else None
    os.chdir(REPO)
    cfg = json.load(open(cfg_path))
    blob, skipped = extract.extract(cfg, read_rollouts=a.read_rollouts)
    blob = json.loads(json.dumps(blob))  # tuples -> lists, exactly as embedded
    meta = build_meta(cfg, blob, skipped, cfg_path)
    tpl = open(os.path.join(HERE, "template.html")).read()
    esc = lambda o: json.dumps(o, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = tpl.replace("__TITLE__", cfg["title"].split(":")[0]).replace("__DATA__", esc(blob)).replace("__META__", esc(meta))
    open(out_path, "w").write(html)
    print(f"wrote {os.path.relpath(out_path, REPO)} ({len(html):,} bytes); panels skipped: {len(skipped)}")
    for k, v in skipped.items():
        print(f"  skipped {k}: {v}")
    if cmp_path:
        d = diff(blob_of(open(cmp_path).read()), blob)
        print(f"compare vs {os.path.relpath(cmp_path, REPO)}: {'IDENTICAL' if not d else f'{len(d)} differences'}")
        for line in d[:200]:
            print("  " + line)
        sys.exit(1 if d else 0)


if __name__ == "__main__":
    main()
