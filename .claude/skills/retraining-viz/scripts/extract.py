"""Panel extractors for the retraining visualisation page.

Each panel is a function `panel_<name>(cfg, ctx) -> dict` that reads files named in the
config and returns the blob keys it owns. If a source the panel needs is missing, it
raises `Missing(path_or_reason)`. build.py records that as a visible skip note, and the
page renders the note in place of the chart. Nothing is ever filled in or guessed.

Metadata-only rule: no panel reads rollouts.jsonl or *.npz unless the config sets
"read_rollouts": true or build.py gets --read-rollouts. Only the achieved-offset field
(`offset_cm` in per_rollout rows) needs it.

All paths are relative to the repo root (build.py chdirs there).
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os
import re


class Missing(Exception):
    """A source the panel needs is absent; the panel is skipped with this note."""


def R(p):
    if not os.path.exists(p):
        raise Missing(f"missing {p}")
    with open(p) as f:
        return json.load(f)


def J(p):
    if not os.path.exists(p):
        raise Missing(f"missing {p}")
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def need(*paths):
    for p in paths:
        if not os.path.exists(p):
            raise Missing(f"missing {p}")


# ---------------------------------------------------------------- log parsing
# lerobot_train logs every log_freq=10 MICRO-steps; `loss:` is the mean over those 10
# micro-batches (AverageMeter reset at each log: lerobot_train.py:698,792). The tqdm
# counter "N/TOTAL" gives the micro-step exactly ("step:1K" in the INFO part is rounded).
# Opt step = micro / accumulation, so with accum 8 a point lands every 1.25 opt steps.
TQDM = re.compile(r"(\d+)/(\d+) \[[^\]]*\]\s*INFO .*? loss:([\d.]+) grdn:([\d.]+) lr:([\d.e+-]+)")
# cached-feature mode (experiments/r056_train.py:478): every 10 opt steps, mean of the last
# accum*10 micro-batch losses.
CACHED = re.compile(r"\[cached\] opt step (\d+) loss ([\d.]+) lr ([\d.e+-]+)")


def parse_lerobot_log(path, accum):
    need(path)
    rows = []
    for seg in open(path, errors="replace").read().split("\r"):
        for line in seg.split("\n"):
            m = TQDM.search(line)
            if m:
                rows.append([int(m.group(1)) / accum, float(m.group(3)), float(m.group(4)), float(m.group(5))])
    return rows  # [opt_step, loss, grad_norm, lr]


def parse_cached_log(path):
    need(path)
    return [[int(m.group(1)), float(m.group(2)), None, float(m.group(3))]
            for m in CACHED.finditer(open(path, errors="replace").read())]


# ---------------------------------------------------------------- training panels
def panel_loss(cfg, ctx):
    out = {}
    for r in cfg.get("train_runs", []):
        if not r.get("log"):
            continue
        try:
            rows = parse_cached_log(r["log"]) if r.get("log_format") == "cached" else parse_lerobot_log(r["log"], r.get("accum", 1))
            if not rows:
                raise Missing(f"no loss lines in {r['log']}")
            ent = {"rows": rows, "src": r["log"]}
            if r.get("lr_trace_from_report"):
                rep = R(r["report"])
                if "lr_trace" in rep:
                    ent["lr_trace"] = rep["lr_trace"]["applied"]
            out[r["key"]] = ent
        except Missing as e:
            ctx["skipped"][f"loss:{r['key']}"] = str(e)
    if not out:
        raise Missing("no training log could be parsed")
    return {"loss": out}


def panel_failed_runs(cfg, ctx):
    """Runs that died before logging a loss (e.g. OOM). Reads only the log."""
    out = {}
    for r in cfg.get("failed_runs", []):
        try:
            need(r["log"])
            txt = open(r["log"], errors="replace").read()
            m = re.findall(r"(\d+)/(\d+) \[", txt)
            t = re.search(r"trainable parameters: ([\d,]+) of ([\d,]+)", txt)
            out[r["key"]] = {"last_micro": int(m[-1][0]) if m else None,
                             "oom": "torch.OutOfMemoryError" in txt,
                             "trainable": t.groups() if t else None}
        except Missing as e:
            ctx["skipped"][f"failed:{r['key']}"] = str(e)
    return out


def panel_curves(cfg, ctx):
    """Train-time eval curves written by r056_train.py --curve (e.g. the overfit diagnostic)."""
    out = {}
    for r in cfg.get("train_runs", []):
        if r.get("curve_json"):
            od = R(r["curve_json"])
            out[r["curve_blob_key"]] = {"train": [[x["opt_step"], x["loss_mean10"]] for x in od["train"]],
                                        "eval": [[x["opt_step"], x["fixed_seed_loss"]] for x in od["eval"]]}
    return out


GATE_KEYS = {
    "conformance": ("conformance", ("pass_count", "n", "tol", "pass", "note")),
    "ckpt_equiv_r16": ("ckpt_equiv_r16", ("loss_plain", "loss_ckpt", "grad_worst_rel", "tol_rel", "pass")),
    "overfit": ("overfit", ("lr_schedule", "loss_step0", "loss_final", "ratio", "pass")),
    "overfit_cosine": ("overfit_cosine_20260929", ("lr_schedule", "loss_step0", "loss_final", "ratio", "pass")),
    "overfit_diag": ("overfit_diag_1500", ("steps", "lr_schedule", "final_ratio")),
    "cache_equiv_r64": ("cache_equiv_r64", ("worst_rel", "tol_rel", "pass")),
}


def panel_gates(cfg, ctx):
    if not cfg.get("gates"):
        raise Missing("no gates file in config")
    g = R(cfg["gates"])
    out = {}
    for name, (src, keys) in GATE_KEYS.items():
        if src in g:
            out[name] = {k: g[src][k] for k in keys}
    if "memory" in g:
        m = g["memory"]
        out["memory"] = {"b1_alloc": m["batch1"]["torch_max_memory_allocated_gib"], "b1_smi": m["batch1"]["nvidia_smi_peak_gib"],
                         "b2_alloc": m["batch2"]["torch_max_memory_allocated_gib"], "b2_smi": m["batch2"]["nvidia_smi_peak_gib"]}
    return {"gates": out}


def panel_smoke(cfg, ctx):
    out = {}
    for r in cfg.get("train_runs", []):
        if r.get("kind") == "smoke":
            sm = R(r["report"])
            out[r["key"]] = {"eval_loss_step0": sm["eval_loss_step0"], "eval_loss_final": sm["eval_loss_final"],
                             "opt_steps": sm["opt_steps"], "effective_batch": sm["effective_batch"], "parts": sm["parts"]}
    return out


def panel_inventory(cfg, ctx):
    inv, ck = {}, {}
    for r in cfg.get("train_runs", []):
        src = r.get("inventory")
        try:
            if src == "report":
                d = R(r["report"])
                inv[r["key"]] = {"r": d["lora"]["r"], "alpha": d["lora"]["alpha"], "dropout": d["lora"]["dropout"],
                                 "batch": d["batch_size"], "accum": d["accumulation"], "eff": d["effective_batch"],
                                 "opt_steps": d["opt_steps"], "lr": d["args"]["lr"], "sched": d["args"]["lr_schedule"],
                                 "warmup": d["args"]["warmup_ratio"], "trainable": d["trainable"]["trainable"],
                                 "alloc": d["torch_max_memory_allocated_gib"], "smi": d["nvidia_smi_peak_gib"],
                                 "verified": d.get("lr_live", {}).get("schedule_verified"), "parts": d["parts"],
                                 "shares": d["shares"], "seed": d["args"]["seed"], "aug": d["augmentation"]}
            elif src == "ckpt_meta":
                metas = sorted(glob.glob(os.path.join(r["ckpt_dir"], "0*", "pretrained_model", "vla_train_meta.json")))
                if not metas:
                    raise Missing(f"no vla_train_meta.json under {r['ckpt_dir']}")
                m = R(metas[-1])
                inv[r["key"]] = {"r": m["lora_r"], "alpha": m["lora_alpha"], "eff": m["effective_batch"],
                                 "accum": m["accumulation"], "batch": m["batch_size"], "opt_total": m["opt_steps_total"],
                                 "mode": m["mode"], "verified": m["schedule_verified"]}
        except Missing as e:
            ctx["skipped"][f"inventory:{r['key']}"] = str(e)
        if r.get("ckpt_dir"):
            name = os.path.basename(os.path.dirname(r["ckpt_dir"].rstrip("/")))
            rows = []
            for d in sorted(glob.glob(os.path.join(r["ckpt_dir"], "0*"))):
                p = os.path.join(d, "pretrained_model", "vla_train_meta.json")
                if not os.path.exists(p):
                    continue  # e.g. the smoke run's checkpoints carry no meta
                mt = R(p)
                rows.append({"dir": os.path.basename(d), "opt_step": mt["opt_step"], "micro_step": mt["micro_step"],
                             "lr_last": mt.get("lr_applied_last_update"), "verified": mt.get("schedule_verified")})
            if rows:
                ck[name] = rows
    return {"inventory": inv, "checkpoints": ck}


# ---------------------------------------------------------------- eval panels
def panel_manifest_counts(cfg, ctx):
    out = {}
    for d in cfg.get("manifest_counts", []):
        try:
            c = collections.defaultdict(lambda: [0, 0])
            for m in J(os.path.join(d, "manifest.jsonl")):
                k = f'{m["ckpt"]}|{m["set"]}'
                c[k][0] += m["success"]; c[k][1] += 1
            out[os.path.basename(d.rstrip("/"))] = dict(c)
        except Missing as e:
            ctx["skipped"][f"manifest_counts:{d}"] = str(e)
    return {"manifest_counts": out}


SCORE_KEYS = ("selected", "heldout", "nominal", "per_band", "paired", "gap_closed", "transfer", "expectations",
              "failure_criterion_gap_lt_one_third", "nominal_on_ramekin", "wise", "uninterpretable_checks")


def panel_score(cfg, ctx):
    sc = {}
    for e in cfg.get("evals", []):
        try:
            d = R(os.path.join(e["dir"], "score.json"))
            sc[os.path.basename(e["dir"].rstrip("/"))] = {k: d[k] for k in SCORE_KEYS if k in d} | (
                {"offset": {k: d["exploratory_gripper_offset"][k] for k in ("offset_le_8cm", "offset_gt_8cm")}}
                if "exploratory_gripper_offset" in d else {})
        except Missing as e2:
            ctx["skipped"][f"score:{e['dir']}"] = str(e2)
    if not sc:
        raise Missing("no score.json found for any eval")
    return {"score": sc}


def panel_per_rollout(cfg, ctx):
    """Held-out/nominal rows per arm from manifest.jsonl. With read_rollouts, adds offset_cm from
    rollouts.jsonl line i (joined by POSITION; rollout_id ignores the noise seed, so it is only
    asserted equal, never used as a key)."""
    read_ro = ctx["read_rollouts"]
    nom = R(cfg["nominal_starts"]) if read_ro else None
    sets = ("heldout", "nominal", "heldout_wise", "nominal_wise")
    pr = {}
    for e in cfg.get("evals", []):
        mp = os.path.join(e["dir"], "manifest.jsonl")
        if not os.path.exists(mp):
            ctx["skipped"][f"per_rollout:{e['dir']}"] = f"missing {mp}"
            continue
        man = J(mp)
        keys = [(m["ckpt"], m["set"], m["scene"], m["dir_seed"], m["radius"], m["noise_seed"], m["env_seed"]) for m in man]
        if len(keys) != len(set(keys)):
            raise Missing(f"duplicate start keys in {e['dir']}/manifest.jsonl")
        ro_lines = None
        if read_ro:
            ro_path = os.path.join(e["dir"], "rollouts.jsonl")
            need(ro_path)
            ro_lines = open(ro_path)
        for i, m in enumerate(man):
            line = next(ro_lines) if ro_lines else None
            if m["set"] not in sets:
                continue
            row = {"ckpt": m["ckpt"], "set": m["set"], "scene": m["scene"], "dir_seed": m["dir_seed"],
                   "radius": m["radius"], "noise_seed": m["noise_seed"], "success": m["success"],
                   "ca": m["closest_approach_m"]}
            if line is not None:
                ro = json.loads(line)
                assert ro["rollout_id"] == m["rollout_id"], (e["dir"], i)
                row["offset_cm"] = 100 * math.dist(ro["steps"][0]["obs_state"]["eef_pos"], nom[str(m["scene"])]["eef_pos"])
            if m["ckpt"] == "base":
                if e.get("base_arm"):
                    pr.setdefault(e["base_arm"], []).append(row)
            elif m["set"].endswith("_wise"):
                if e.get("wise_arm"):
                    pr.setdefault(e["wise_arm"], []).append(row)
            else:
                pr.setdefault(e["arm"], []).append(row)
        if ro_lines:
            ro_lines.close()
    if read_ro:  # cross-check the offset split against score.json's own exploratory block
        for e in cfg.get("evals", []):
            sp = os.path.join(e["dir"], "score.json")
            if not os.path.exists(sp) or e["arm"] not in pr:
                continue
            ex = R(sp).get("exploratory_gripper_offset")
            if not ex:
                continue
            thr = 100 * ex["offset_threshold_m"]
            for arm, side in ((e["arm"], "retrained"), (e.get("base_arm"), "base")):
                if not arm:
                    continue
                le = [x for x in pr[arm] if x["set"] == "heldout" and x["offset_cm"] <= thr]
                want = ex["offset_le_8cm"][side]
                assert (sum(x["success"] for x in le), len(le)) == (want["success"], want["n"]), (e["dir"], side)
    else:
        ctx["skipped"]["offset"] = ("achieved-offset chart needs rollouts.jsonl (t=0 eef_pos); metadata-only by default. "
                                    "Re-run build.py with --read-rollouts to include it.")
    return {"per_rollout": pr}


def panel_transfer(cfg, ctx):
    out = {}
    for e in cfg.get("evals", []):
        p = os.path.join(e["dir"], "transfer", "rows.jsonl")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            out[e["arm"]] = J(p)
        else:
            ctx["skipped"][f"transfer:{e['dir']}"] = f"no transfer rows at {p}"
    return {"transfer": out}


# ---------------------------------------------------------------- diagnostics
def panel_d20(cfg, ctx):
    dg = cfg.get("diag")
    if not dg:
        raise Missing("no diag block in config")
    out = {}

    def part(name, fn):
        try:
            fn()
        except Missing as e:
            ctx["skipped"][name] = str(e)

    def _loss():
        ls = R(dg["loss_summary"])
        out["d20_loss"] = {"subsets": {s: {m: {k: v[k] for k in ("mean", "se", "median") if k in v} |
                                           ({"ci": v["diff_vs_base_ci95"], "below": v["share_frames_below_base"]} if "diff_vs_base_ci95" in v else {})
                                           for m, v in sub.items() if isinstance(v, dict)} | {"n": sub["n"]}
                                       for s, sub in ls["subsets"].items()},
                           "draws": ls["draws"], "seed": ls["seed"], "models": ls["models"]}

    def _fit():
        fc = R(dg["fit"])

        def strip(block):
            return {k: {"n": v["n"], **{m: v["own"][m] for m in v.get("own", {})}} for k, v in block.items()}
        out["d20_fit"] = {"metric": fc["metric"],
                          "to_label_by_cm": strip(fc["to_label"]["by_cm"]), "to_label_all": strip({"all": fc["to_label"]["all"]})["all"],
                          "to_nominal_by_cm": strip(fc["to_nominal"]["by_cm"]), "to_nominal_all": strip({"all": fc["to_nominal"]["all"]})["all"],
                          "heldout_all": strip({"all": fc["heldout_matched"]["all"]})["all"]}

    def _rows():
        fr = []
        for r in J(dg["fit_rows"]):
            if r["kind"] != "train":
                continue
            lab = r["label_disp"]
            fr.append({"offset_cm": r["offset_cm"], "scene": r["scene"],
                       **{m: math.dist(v, lab) for m, v in r["disp"]["own"].items()}})
        out["d20_fit_rows"] = fr

    def _curve():
        oc = R(dg["offset_curve"])
        out["r047_curve"] = {"bands": oc["per_offset_band_cm"], "bands_by_dir": oc["per_offset_band_cm_by_dir"],
                             "definition": oc["offset_definition"], "n": oc["n_manifest"]}

    for key, name, fn in (("loss_summary", "d20", _loss), ("fit", "d20_fit", _fit), ("fit_rows", "d20_fit_rows", _rows),
                          ("offset_curve", "r047_curve", _curve)):
        if dg.get(key):
            part(name, fn)
    return out


# ---------------------------------------------------------------- training data
def panel_data(cfg, ctx):
    dc = cfg.get("data")
    if not dc:
        raise Missing("no data block in config")
    root = dc["dataset"]
    prov = J(os.path.join(root, "meta", "provenance.jsonl"))
    P = lambda r: r["provenance"]
    exp = R(os.path.join(root, "meta", "export.json"))
    out = {
        "scene": dict(collections.Counter(str(P(r)["scene_task_id"]) for r in prov)),
        "band": dict(collections.Counter(P(r)["band"] for r in prov)),
        "noise": dict(collections.Counter(str(P(r)["noise_seed"]) for r in prov)),
        "source": dict(collections.Counter(P(r)["source"] for r in prov)),
        "n_frames": dict(collections.Counter(str(r["n_frames"]) for r in prov)),
        "n_driven": dict(collections.Counter(str(r["n_driven"]) for r in prov)),
        "full_steps": [r["outcome"]["steps"] for r in prov],
        "export": {k: exp[k] for k in ("episodes", "frames", "truncate_steps")},
        "yield": R(dc["yield"]),
        "datasets": {os.path.basename(d): {k: R(os.path.join(d, "meta", "info.json"))[k] for k in ("total_episodes", "total_frames", "fps")}
                     for d in dc.get("datasets", [])},
    }
    cov = R(dc["coverage"])
    out["coverage"] = {k: cov[k] for k in ("scenes", "offset_bins", "nearest", "window_phases", "gripper_closed_fraction", "replay_start_spread_cm")}
    out["mint_attempts_by_scene"] = {}
    for m in J(dc["mint_manifest"]):
        s = out["mint_attempts_by_scene"].setdefault(str(m["scene_task_id"]), [0, 0])
        s[0] += m["success"]; s[1] += 1
    return {"data": out}


def panel_upstream(cfg, ctx):
    up = cfg.get("upstream") or {}
    out = {}
    if up.get("r053"):
        s53 = R(up["r053"])
        out["r053"] = {k: {kk: v[kk] for kk in ("n", "P_fail", "N_success", "N_flips", "rerun_flips", "rerun_breaks", "median_ca_P", "median_ca_N")}
                       for k, v in s53["per_seed"].items()}
    if up.get("r057"):
        s57 = R(up["r057"])
        out["r057"] = {ax: {k: {kk: v[kk] for kk in ("n", "success", "rate", "wilson95", "median_closest_approach")}
                            for k, v in s57["rows"][ax]["by_k"].items()} for ax in s57["rows"]}
    if not out:
        raise Missing("no upstream sources in config")
    return out


PANELS = [("loss", panel_loss), ("failed_runs", panel_failed_runs), ("curves", panel_curves), ("gates", panel_gates),
          ("smoke", panel_smoke), ("inventory", panel_inventory), ("manifest_counts", panel_manifest_counts),
          ("score", panel_score), ("per_rollout", panel_per_rollout), ("transfer", panel_transfer),
          ("d20", panel_d20), ("data", panel_data), ("upstream", panel_upstream)]


def extract(cfg, read_rollouts=False):
    ctx = {"skipped": {}, "read_rollouts": read_rollouts or bool(cfg.get("read_rollouts"))}
    blob = {}
    for name, fn in PANELS:
        if name in cfg.get("disable_panels", []):
            ctx["skipped"][name] = "disabled in config"
            continue
        try:
            blob.update(fn(cfg, ctx))
        except Missing as e:
            ctx["skipped"][name] = str(e)
    return blob, ctx["skipped"]
