import json, os
def J(p):
    try: return json.load(open(p))
    except Exception: return None
print("Overnight queue summary\n")
print(f"{'step':26s} {'status':34s} {'rc':>4s} {'duration':>9s}")
for line in open("runs/overnight/steps_all.tsv"):
    n, st, rc, dt = line.rstrip("\n").split("\t")
    dt = int(dt); print(f"{n:26s} {st:34s} {rc:>4s} {dt//3600}h{dt%3600//60:02d}m")
print("\nHeadlines")
s = J("runs/r053/score.json")
if s:
    p = s["pooled"]; print(f"R-053  N success {p['N_success']}/{p['n']}; N flip rate {p['N_flip_rate']}, re-run flip rate {p['rerun_flip_rate']}; "
                          f"expectations {s['expectations']}; interpretable {s['uninterpretable_checks']['interpretable']}")
y = J("runs/r054/yield.json")
if y: print("R-054  yield " + ", ".join(f"{b} {v['successes']}/{v['attempts']}" for b, v in y.items()))
g = J("runs/r054/gates.json")
if g: print("R-054  gates " + ", ".join(f"{k}={v.get('pass')}" for k, v in g.items() if isinstance(v, dict)))
g = J("runs/r055/gates.json")
if g: print("R-055  gates " + ", ".join(f"{k}={v.get('pass', v.get('E3_batch1_le_7.4GiB'))}" for k, v in g.items() if isinstance(v, dict)))
for m in ("memory",):
    if g and m in g: print(f"R-055  memory b1 {g[m]['batch1'].get('nvidia_smi_peak_gib')} GiB, b2 {g[m]['batch2'].get('nvidia_smi_peak_gib')} GiB (nvidia-smi)")
r = J("runs/r056_r16_train_report.json")
if r: print(f"R-056  training {r.get('mode')}: trainable {r.get('trainable', {}).get('trainable')}, peak {r.get('nvidia_smi_peak_gib')} GiB")
s = J("runs/r056_r16/score.json")
if s:
    print(f"R-056  held-out base {s['heldout']['base']['success']}/{s['heldout']['base']['n']} -> retrained "
          f"{s['heldout']['retrained']['success']}/{s['heldout']['retrained']['n']}; gap closed {s['gap_closed']}; "
          f"nominal base {s['nominal']['base']['success']}/{s['nominal']['base']['n']}, retrained {s['nominal']['retrained']['success']}/{s['nominal']['retrained']['n']}")
    print(f"R-056  expectations {s['expectations']}")
    if "wise" in s: print(f"R-056  WiSE-FT {s['wise']}")
s = J("runs/r057/score.json")
if s:
    for row, v in s["rows"].items():
        print(f"R-057  {row}: k* = {v['k_star']}; " + ", ".join(f"{k} {x['success']}/{x['n']}" for k, x in v["by_k"].items()))
r = J("runs/r058_r4_train_report.json")
if r: print(f"R-058  r=4 training {r.get('mode')}: trainable {r.get('trainable', {}).get('trainable')}, peak {r.get('nvidia_smi_peak_gib')} GiB")
for ax in ("camera_bench_yaw_deg", "camera_bench_scale", "camera_bench_pitch_deg"):
    p = f"runs/r052/{ax}/manifest.jsonl"
    if os.path.exists(p):
        rows = [json.loads(l) for l in open(p) if l.strip()]
        print(f"R-052  {ax}: {len(rows)} rollouts, success {sum(r['success'] for r in rows)}")
print("\nR-052 status: " + ("DONE" if os.path.exists("runs/r052/DONE") else "PARTIAL" if os.path.exists("runs/r052/PARTIAL") else "not run"))
