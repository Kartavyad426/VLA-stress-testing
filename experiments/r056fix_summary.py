"""Append one step's headline to runs/overnight/SUMMARY_r056fix.txt (the file the user reads first).

  python experiments/r056fix_summary.py <step> <kind> <rc> <minutes> [--run DIR] [--report JSON] [--peaks JSON] [--note TEXT]

kinds: train (report + last checkpoint meta), eval (DIR/score.json: held-out base vs retrained with Wilson
intervals, gap closed, nominal, W transfer, exploratory offset split), wise (DIR/score.json "wise"),
r052 (per-axis rollout counts), mint (store successes / attempts), other. Never raises: a missing file
becomes a line saying so. Outcomes are SIGNALS: nothing downstream reads this file.
"""
import argparse
import glob
import json
import os
import time

SUMMARY = "runs/overnight/SUMMARY_r056fix.txt"


def J(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def pct(s):
    if not s or not s.get("n"):
        return "n/a"
    lo, hi = s.get("wilson95") or (None, None)
    ci = f" [{lo:.2f},{hi:.2f}]" if lo is not None else ""
    return f"{s['success']}/{s['n']} = {s['rate']:.2f}{ci}"


def train_lines(a):
    out = []
    r = J(a.report) if a.report else None
    if r:
        lv = r.get("lr_live", {})
        out.append(f"  schedule_verified={lv.get('schedule_verified')} lr max err={lv.get('lr_live_max_abs_err')} "
                   f"updates={lv.get('lr_live_updates')} (from opt step {lv.get('lr_live_first_opt_step')}); "
                   f"guard max err={(r.get('schedule_guard') or {}).get('max_abs_err')}")
        out.append(f"  torch peak alloc {r.get('torch_max_memory_allocated_gib', 0):.2f} GiB, nvidia-smi peak "
                   f"{r.get('nvidia_smi_peak_gib', 0):.2f} GiB; trainable {(r.get('trainable') or {}).get('trainable')}")
    else:
        out.append(f"  no report at {a.report}")
    if a.run:
        ms = [(json.load(open(p)), p) for p in glob.glob(os.path.join(a.run, "checkpoints", "*", "pretrained_model",
                                                                      "vla_train_meta.json"))]
        if ms:
            m, p = max(ms, key=lambda x: x[0]["opt_step"])
            out.append(f"  last checkpoint opt {m['opt_step']}/{m['opt_steps_total']} schedule_verified="
                       f"{m.get('schedule_verified')} lr_last={m.get('lr_applied_last_update')} ({os.path.dirname(p)})")
        else:
            out.append(f"  no checkpoints under {a.run}")
    return out


def eval_lines(a):
    s = J(os.path.join(a.run, "score.json"))
    if not s:
        return [f"  no score.json in {a.run}"]
    sel = s.get("selected", {})
    h, n = s.get("heldout", {}), s.get("nominal", {})
    g = s.get("gap_closed")
    gs = f"{g:.2f}" if g is not None else "n/a"
    out = [f"  selected {sel.get('label')} (val {sel.get('success')}/{sel.get('n')})",
           f"  HELD-OUT base {pct(h.get('base'))}  retrained {pct(h.get('retrained'))}  gap closed {gs}",
           f"  nominal base {pct(n.get('base'))}  retrained {pct(n.get('retrained'))}"]
    t = s.get("transfer", {})
    out.append(f"  W transfer at forward 0: retrained median {t.get('retrained_median_W')} vs base {t.get('base_median_W')}")
    ex = s.get("exploratory_gripper_offset", {})
    if "offset_le_8cm" in ex:
        out.append(f"  offset <=8cm: base {pct(ex['offset_le_8cm']['base'])} retrained {pct(ex['offset_le_8cm']['retrained'])}; "
                   f">8cm: base {pct(ex['offset_gt_8cm']['base'])} retrained {pct(ex['offset_gt_8cm']['retrained'])} (exploratory)")
    elif ex:
        out.append(f"  offset split: {ex.get('error')}")
    e = s.get("expectations", {})
    out.append("  expectations: " + ", ".join(f"{k.split('_')[0]}={v}" for k, v in e.items()))
    u = s.get("uninterpretable_checks", {})
    out.append(f"  checks: base vs R-047 curve consistent={u.get('base_heldout_vs_r047_curve', {}).get('consistent')}, "
               f"base nominal>=26/27={u.get('base_nominal_ge_26_of_27')}, G4 violations={len(u.get('g4_violations', []))}")
    if s.get("nominal_on_ramekin"):
        r = s["nominal_on_ramekin"]
        out.append(f"  on-ramekin (apart): base {pct(r.get('base'))} retrained {pct(r.get('retrained'))}")
    return out


def wise_lines(a):
    s = J(os.path.join(a.run, "score.json"))
    w = (s or {}).get("wise")
    if not w:
        return [f"  no wise block in {a.run}/score.json"]
    h = (s.get("heldout") or {})
    return [f"  WiSE alpha={w['alpha']}: held-out {pct(w['heldout'])} (base {pct(h.get('base'))}, alpha=1 "
            f"{pct(h.get('retrained'))}); nominal {pct(w['nominal'])} (exploratory)"]


def r052_lines(a):
    out = []
    for ax in ("camera_bench_yaw_deg", "camera_bench_scale", "camera_bench_pitch_deg"):
        p = f"runs/r052/{ax}/manifest.jsonl"
        n = sum(1 for _ in open(p)) if os.path.exists(p) else 0
        out.append(f"  {ax}: {n} rollouts on record")
    out.append(f"  DONE marker: {os.path.exists('runs/r052/DONE')}")
    return out


def mint_lines(a):
    p = os.path.join(a.run, "manifest.jsonl")
    if not os.path.exists(p):
        return [f"  no manifest in {a.run}"]
    rows = [json.loads(l) for l in open(p) if l.strip()]
    k = sum(bool(r.get("success")) for r in rows)
    return [f"  {a.run}: {k} successes / {len(rows)} attempts (+100 in the pilot runs/r054 = {k + 100} total)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step"); ap.add_argument("kind"); ap.add_argument("rc"); ap.add_argument("minutes")
    ap.add_argument("--run"); ap.add_argument("--report"); ap.add_argument("--peaks"); ap.add_argument("--note", default="")
    a = ap.parse_args()
    status = {"0": "OK"}.get(a.rc, "SKIPPED" if a.rc == "skip" else f"FAILED")
    lines = [f"[{time.strftime('%F %T')}] {a.step}: {status} (rc {a.rc}, {a.minutes} min){(' - ' + a.note) if a.note else ''}"]
    try:
        lines += {"train": train_lines, "eval": eval_lines, "wise": wise_lines, "r052": r052_lines,
                  "mint": mint_lines}.get(a.kind, lambda a: [])(a)
    except Exception as e:
        lines.append(f"  (summary error: {type(e).__name__}: {e})")
    pk = J(a.peaks) if a.peaks else None
    if pk:
        lines.append(f"  memory: RSS tree max {pk.get('rss_tree_max_gib')} GiB (anon {pk.get('rss_anon_tree_max_gib')}), "
                     f"largest process {pk.get('rss_proc_max_gib')} GiB, GPU used max {pk.get('gpu_used_max_gib')} GiB")
    os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
    with open(SUMMARY, "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
