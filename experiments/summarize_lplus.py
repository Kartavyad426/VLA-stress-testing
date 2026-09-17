"""Summarise LIBERO-Plus harness runs: success by perturbation type and level.

    python3 experiments/summarize_lplus.py runs/lplus_sweep_groot [runs/lplus_sweep_smolvla ...]
"""
import json, math, sys
from collections import defaultdict

def wilson(k, n, z=1.96):
    if not n: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (100*(c-h), 100*(c+h))

for run in sys.argv[1:]:
    cell = defaultdict(lambda: [0, 0]); cat = defaultdict(lambda: [0, 0]); lvl = defaultdict(lambda: [0, 0])
    inst, n = set(), 0
    try:
        f = open(f"{run}/rollouts.jsonl")
    except FileNotFoundError:
        print(f"\n### {run}: NOT RUN"); continue
    for line in f:
        r = json.loads(line); lp = r["scene_descriptor"].get("libero_plus") or {}
        c, l = lp.get("category"), lp.get("difficulty_level")
        s = bool(r["success"]); n += 1
        inst.add(r["fingerprint"]["env"].get("instruction_source"))
        for d, key in ((cell, (c, l)), (cat, c), (lvl, l)):
            d[key][0] += s; d[key][1] += 1
    print(f"\n### {run}  n={n}  instruction_source={sorted(x for x in inst if x)}")
    print(f"  {'perturbation type':22s} " + "".join(f"  L{l}      " for l in (2,3,4,5)) + "   overall")
    for c in sorted(x for x in cat if x):
        row = ""
        for l in (2,3,4,5):
            k, m = cell[(c, l)]
            row += f"  {100*k/m:5.1f}%({m:2d})" if m else "     -    "
        k, m = cat[c]; lo, hi = wilson(k, m)
        print(f"  {c:22s}{row}   {100*k/m:5.1f}% [{lo:4.0f},{hi:4.0f}] n={m}")
    print(f"  {'ALL':22s} " + "".join(
        f"  {100*lvl[l][0]/lvl[l][1]:5.1f}({lvl[l][1]:3d})" if lvl[l][1] else "     -    " for l in (2,3,4,5)))
