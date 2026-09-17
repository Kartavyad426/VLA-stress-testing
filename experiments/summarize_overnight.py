"""Morning summary for experiments/overnight_20260917.sh. Stdlib only.

    python3 experiments/summarize_overnight.py

Reads whatever finished. Missing pieces are reported as NOT RUN, never as zero.
"""
import json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

PUBLISHED = {"libero_spatial": 96.8, "libero_object": 99.6,
             "libero_goal": 97.4, "libero_10": 89.2}
SUITES = list(PUBLISHED)


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def successes_by_episode(eval_info):
    """Per-episode success list from a single-task lerobot-eval eval_info.json."""
    d = json.load(open(eval_info))
    for t in d.get("per_task", []):
        s = (t.get("metrics") or {}).get("successes")
        if s is not None:
            return list(s)
    return None


print("=" * 78)
print("OVERNIGHT 2026-09-17 SUMMARY")
print("=" * 78)
log = "experiments/repro/logs/overnight_20260917.log"
if os.path.exists(log):
    lines = open(log).read().splitlines()
    tests = [l for l in lines if "test " in l]
    print("\nHarness self-tests:")
    for l in tests:
        print("  " + l)
    acs = sorted({l.split("ac=")[1][0] for l in lines if "ac=" in l})
    print(f"  power states seen during run: ac={','.join(acs)}  (0 = battery: timings suspect)")

# --- step 4: MINERVA reproduction ---------------------------------------------
R = "experiments/repro/runs/minerva_repro_full"
print("\n[4] MINERVA reproduction (authors' lerobot-eval command) vs published")
print(f"  {'suite':15s} {'tasks':>6s} {'eps':>5s} {'ours':>7s} {'95% CI':>15s} {'published':>9s}  verdict")
repro_eps = {}
tot_k = tot_n = 0
for s in SUITES:
    k = n = tasks = 0
    for t in range(10):
        f = f"{R}/{s}/task{t}/eval_info.json"
        if not os.path.exists(f):
            continue
        ep = successes_by_episode(f)
        if ep is None:
            continue
        repro_eps[(s, t)] = ep
        k += sum(ep); n += len(ep); tasks += 1
    if not n:
        print(f"  {s:15s} NOT RUN")
        continue
    tot_k += k; tot_n += n
    lo, hi = wilson(k, n)
    pub = PUBLISHED[s]
    verdict = "consistent" if lo <= pub <= hi else ("BELOW published" if hi < pub else "above published")
    part = "" if tasks == 10 else f"  (partial: {tasks}/10 tasks)"
    print(f"  {s:15s} {tasks:>6d} {n:>5d} {100*k/n:6.1f}% [{lo:5.1f},{hi:5.1f}] {pub:8.1f}%  {verdict}{part}")
if tot_n:
    print(f"  {'OVERALL':15s} {'':>6s} {tot_n:>5d} {100*tot_k/tot_n:6.1f}%  published avg 95.75%")

# --- step 2: harness parity ------------------------------------------------------
print("\n[2] Harness parity: our harness vs lerobot-eval, SAME init states 0-9 per task")
cells = "runs/minerva_harness_parity/cells.jsonl"
if not os.path.exists(cells):
    print("  NOT RUN")
else:
    rows = [json.loads(l) for l in open(cells) if not json.loads(l)["spec"]]
    print(f"  {'suite':15s} {'harness':>12s} {'lerobot-eval (eps 0-9)':>24s}  diff    verdict")
    for s in ("libero_object", "libero_spatial"):
        hk = hn = lk = ln = 0
        for r in rows:
            if r["suite"] != s:
                continue
            hk += r["successes"]; hn += r["n"]
            ep = repro_eps.get((s, r["task"]))
            if ep is not None:
                lk += sum(ep[:10]); ln += len(ep[:10])
        if not hn:
            print(f"  {s:15s} NOT RUN"); continue
        if not ln:
            print(f"  {s:15s} {100*hk/hn:6.1f}% (n={hn})   lerobot-eval side not run yet")
            continue
        p1, p2 = hk / hn, lk / ln
        se = math.sqrt(max(p1*(1-p1)/hn + p2*(1-p2)/ln, 1e-12))
        d = 100 * (p1 - p2)
        z = (p1 - p2) / se
        v = "no detectable harness bias" if abs(z) < 1.96 else "HARNESS GAP -- investigate"
        print(f"  {s:15s} {100*p1:6.1f}% n={hn:<4d} {100*p2:14.1f}% n={ln:<4d} {d:+6.1f}  {v} (z={z:+.2f})")
    print("  Note: MINERVA samples noise, so episodes are not expected to match 1:1;")
    print("  n=100 per suite only detects gaps of roughly 10 pp or more.")

# --- step 3: camera sweep ---------------------------------------------------------
print("\n[3] Camera sweep with FIXED code (MINERVA, libero_object, 5 eps x 10 tasks)")
cells = "runs/minerva_harness_camera/cells.jsonl"
if not os.path.exists(cells):
    print("  NOT RUN")
else:
    agg = {}
    for l in open(cells):
        r = json.loads(l)
        key = json.dumps(r["spec"], sort_keys=True)
        a = agg.setdefault(key, [0, 0, 0])
        a[0] += r["successes"]; a[1] += r["n"]; a[2] += 1
    order = ['{}', '{"camera_yaw_deg": 0}', '{"camera_yaw_deg": 2}', '{"camera_yaw_deg": 5}',
             '{"camera_yaw_deg": 10}', '{"camera_yaw_deg": 20}', '{"camera_pitch_deg": 5}']
    for key in order:
        if key not in agg:
            print(f"  {key:26s} NOT RUN"); continue
        k, n, tasks = agg[key]
        lo, hi = wilson(k, n)
        print(f"  {key:26s} {100*k/n:6.1f}%  [{lo:5.1f},{hi:5.1f}]  n={n} ({tasks} tasks)")
    if '{}' in agg and '{"camera_yaw_deg": 0}' in agg:
        a, b = agg['{}'], agg['{"camera_yaw_deg": 0}']
        print(f"  check: yaw 0 vs nominal = {100*b[0]/b[1]:.1f}% vs {100*a[0]/a[1]:.1f}% "
              f"(should be equal within noise; they are the same camera)")

# --- step 5: SmolVLA harness parity -------------------------------------------------
print("\n[5] SmolVLA harness parity: our adapter vs lerobot-eval res256_nas10_seed1000, spatial")
cells = "runs/smolvla_harness_parity/cells.jsonl"
ref = "experiments/repro/runs/res256_nas10_seed1000/eval_info.json"
if not os.path.exists(cells):
    print("  NOT RUN")
else:
    rows = [json.loads(l) for l in open(cells)]
    hk = sum(r["successes"] for r in rows); hn = sum(r["n"] for r in rows)
    d = json.load(open(ref))
    lk = ln = 0
    pairs = []
    for t in d["per_task"]:
        if t.get("task_group") != "libero_spatial" and "spatial" not in json.dumps(t)[:300]:
            continue
        tid = t.get("task_id")
        ep = (t.get("metrics") or {}).get("successes") or []
        mine = [r for r in rows if r["task"] == tid]
        if not mine:
            continue
        lk += sum(ep); ln += len(ep)
    if hn and ln:
        p1, p2 = hk / hn, lk / ln
        se = math.sqrt(max(p1*(1-p1)/hn + p2*(1-p2)/ln, 1e-12)); z = (p1-p2)/se
        v = "no detectable harness bias" if abs(z) < 1.96 else "HARNESS GAP -- investigate"
        print(f"  harness {100*p1:.1f}% (n={hn})  vs  lerobot-eval {100*p2:.1f}% (n={ln})  "
              f"diff {100*(p1-p2):+.1f} pp  {v} (z={z:+.2f})")
    else:
        print(f"  partial: harness n={hn}, matched reference n={ln}")
print()
