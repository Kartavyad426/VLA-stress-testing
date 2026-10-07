import json, sys, os, numpy as np
sys.path.insert(0, os.getcwd())
from experiments.r041_report import SCENE
from scipy.optimize import minimize
def fit(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    def nll(p):
        a, b = p; z = np.clip(a + b*xs, -30, 30); pr = np.clip(1/(1+np.exp(-z)), 1e-6, 1-1e-6)
        return -np.sum(ys*np.log(pr)+(1-ys)*np.log(1-pr)) + 1e-3*(a*a+b*b)
    return minimize(nll, [3.0, -1.0], method="Nelder-Mead", options={"maxiter": 4000}).x
def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return (c-h, c+h)
OUT = {}
for axis, base, mx in (("camera_bench_yaw_deg", 0.0, 75.0), ("camera_bench_scale", 1.0, 2.0), ("camera_bench_pitch_deg", 0.0, 15.0)):
    rs = [json.loads(l) for l in open(f"runs/r052/{axis}/manifest.jsonl")]
    keys = {(r["task_id"], r["magnitude"], r["dir"], r["noise_seed"], r["env_seed"]) for r in rs}
    mags = sorted({r["magnitude"] for r in rs}); scenes = sorted({r["task_id"] for r in rs})
    print(f"\n==== {axis} n={len(rs)} unique keys={len(keys)} scenes={len(scenes)} mags={mags}")
    term = {}
    for r in rs: term[r["termination"]] = term.get(r["termination"], 0)+1
    print("terminations", term)
    pooled = {}
    for m in mags:
        g = [r["success"] for r in rs if r["magnitude"] == m]; pooled[m] = (sum(g), len(g))
    print("pooled", {m: f"{k}/{n}" for m, (k, n) in pooled.items()})
    res = {}
    for t in scenes:
        p = [np.mean([r["success"] for r in rs if r["task_id"] == t and r["magnitude"] == m]) for m in mags]
        n = [sum(1 for r in rs if r["task_id"] == t and r["magnitude"] == m) for m in mags]
        viol = [(mags[i], mags[j]) for i in range(len(mags)) for j in range(i+1, len(mags)) if p[j]-p[i] > 1.0/n[i]+1e-9]
        xs = [(r["magnitude"]-base)/(mx-base) for r in rs if r["task_id"] == t]; ys = [r["success"] for r in rs if r["task_id"] == t]
        a, b = fit(xs, ys)
        if b < -1e-6: x50 = base + (-a/b)*(mx-base); w = (2*np.log(4))/(-b)*(mx-base)
        else: x50 = w = float("inf")
        res[t] = dict(p=p, n=n, viol=viol, x50=x50, width=w, p_last=p[-1], below05_last=p[-1] < 0.5, reach02=min(p) <= 0.2)
        print(f"{SCENE.get(t,t):22s} " + " ".join(f"{x:.2f}" for x in p) + f" | x50 {x50:7.3g} w {w:7.3g} viol {viol[:3]}")
    rr = []
    for m in mags:
        if m == base: continue
        for t in scenes:
            g = [r for r in rs if r["task_id"] == t and r["magnitude"] == m]
            s = np.array([r["success"] for r in g], float); d = np.array([r["d_pn_f0"] for r in g], float)
            if s.std() > 0 and d.std() > 0: rr.append(((d-d.mean())/d.std(), (s-s.mean())/s.std()))
    r_within = float(np.mean(np.concatenate([a*b for a, b in rr]))) if rr else None
    print(f"within-cell r = {r_within} over {len(rr)} mixed cells")
    # pooled logistic crossing
    xs = [(r["magnitude"]-base)/(mx-base) for r in rs]; ys = [r["success"] for r in rs]; a, b = fit(xs, ys)
    px50 = base + (-a/b)*(mx-base) if b < -1e-6 else float("inf")
    print("pooled x50", px50)
    OUT[axis] = dict(n=len(rs), unique=len(keys), mags=mags, pooled={str(m): v for m, v in pooled.items()},
                     pooled_wilson={str(m): wilson(*v) for m, v in pooled.items()}, pooled_x50=px50,
                     r_within=r_within, ncells=len(rr), terminations=term,
                     res={str(t): {k: (x if not isinstance(x, float) or np.isfinite(x) else None) for k, x in d.items()} for t, d in res.items()})
# R-047 harness yaw comparison
h = [json.loads(l) for l in open("runs/r047/camera_yaw_deg/manifest.jsonl")]
b = [json.loads(l) for l in open("runs/r052/camera_bench_yaw_deg/manifest.jsonl")]
hk = {(r["task_id"], r["magnitude"], r["noise_seed"]): r["success"] for r in h}
bk = {(r["task_id"], r["magnitude"], r["noise_seed"]): r["success"] for r in b}
common = sorted(set(hk) & set(bk))
print(f"\n==== bench vs harness yaw: {len(common)} matched (scene,mag,seed); harness n={len(h)} scenes={sorted({r['task_id'] for r in h})}")
cmp = {}
for m in sorted({k[1] for k in common}):
    c = [k for k in common if k[1] == m]
    hs = sum(hk[k] for k in c); bs = sum(bk[k] for k in c)
    b01 = sum(1 for k in c if hk[k] and not bk[k]); c10 = sum(1 for k in c if bk[k] and not hk[k])
    cmp[str(m)] = dict(n=len(c), harness=hs, bench=bs, harness_only=b01, bench_only=c10)
    print(f"yaw {m:5}: harness {hs}/{len(c)} bench {bs}/{len(c)}  (harness-only {b01}, bench-only {c10})")
per_scene75 = {str(t): dict(harness=sum(hk[k] for k in common if k[0]==t and k[1]==75.0), bench=sum(bk[k] for k in common if k[0]==t and k[1]==75.0), n=sum(1 for k in common if k[0]==t and k[1]==75.0)) for t in sorted({k[0] for k in common})}
print("per scene @75 (harness/bench):", {SCENE.get(int(t),t): f"{v['harness']}/{v['bench']} of {v['n']}" for t, v in per_scene75.items()})
OUT["bench_vs_harness_yaw"] = dict(per_mag=cmp, per_scene_75=per_scene75)
json.dump(OUT, open(sys.argv[1], "w"), indent=1, default=float)
