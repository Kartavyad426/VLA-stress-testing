"""R-047 scoring: per-scene success curves, logistic 50% point and 0.8→0.2 width, within-cell r(‖P−N‖ f0, success). Writes runs/r047/score.json.

  .venvs/groot/bin/python experiments/r047_score.py
"""
import json, sys, os, numpy as np
sys.path.insert(0, os.getcwd())
from experiments.r041_report import SCENE, load_axis
from scipy.optimize import minimize
def fit(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    def nll(p):
        a, b = p; z = np.clip(a + b * xs, -30, 30); pr = 1/(1+np.exp(-z)); pr = np.clip(pr, 1e-6, 1-1e-6)
        return -np.sum(ys*np.log(pr) + (1-ys)*np.log(1-pr)) + 1e-3*(a*a+b*b)
    r = minimize(nll, [3.0, -1.0], method="Nelder-Mead", options={"maxiter": 4000}); return r.x
out = {}
for axis, mx in (("joint_radius_rad", 0.5), ("camera_yaw_deg", 75.0)):
    rs = [json.loads(l) for l in open(f"runs/r047/{axis}/manifest.jsonl")]
    mags = sorted({r["magnitude"] for r in rs}); scenes = sorted({r["task_id"] for r in rs})
    print(f"\n==== {axis}  n={len(rs)}")
    m0 = [r["success"] for r in rs if r["magnitude"] == 0]; print(f"mag-0 success {sum(m0)}/{len(m0)}")
    res = {}
    for t in scenes:
        p = [np.mean([r["success"] for r in rs if r["task_id"] == t and r["magnitude"] == m]) for m in mags]
        n = [sum(1 for r in rs if r["task_id"] == t and r["magnitude"] == m) for m in mags]
        # monotone allowing single-replicate flips: any increase > 1/n is a violation
        viol = [(mags[i], mags[j]) for i in range(len(mags)) for j in range(i+1, len(mags)) if p[j] - p[i] > 1.0/n[i] + 1e-9]
        xs = [r["magnitude"] / mx for r in rs if r["task_id"] == t]; ys = [r["success"] for r in rs if r["task_id"] == t]
        a, b = fit(xs, ys)
        if b < -1e-6:
            x50 = -a / b * mx; w = (np.log(0.8/0.2) - np.log(0.2/0.8)) / (-b) * mx
        else:
            x50, w = float("inf"), float("inf")
        reach = min(p) <= 0.2
        res[t] = dict(p=p, viol=viol, x50=x50, width=w, reach=reach, p_max=p[-1])
        print(f"{SCENE.get(t,t):22s} " + " ".join(f"{x:.2f}" for x in p) + f" | x50 {x50:7.3g} width {w:7.3g} reach<=.2 {reach} viol {viol[:3]}")
    # corr within magnitude
    rr = []
    for m in mags:
        if m == 0: continue
        for t in scenes:
            g = [r for r in rs if r["task_id"] == t and r["magnitude"] == m]
            s = np.array([r["success"] for r in g], float); d = np.array([r["d_pn_f0"] for r in g], float)
            if s.std() > 0 and d.std() > 0:
                rr.append((d - d.mean()) / d.std()); rr[-1] = (rr[-1], (s - s.mean()) / s.std())
    zd = np.concatenate([a for a, b in rr]); zs = np.concatenate([b for a, b in rr])
    print(f"pooled within-cell r(d_pn_f0, success) = {np.mean(zd*zs):.3f} over {len(rr)} mixed cells, {len(zd)} rollouts")
    out[axis] = dict(res=res, r=float(np.mean(zd*zs)), ncells=len(rr), mags=mags)
json.dump({a: {"r": v["r"], "ncells": v["ncells"], "mags": v["mags"], "res": {str(t): {k: (x if not isinstance(x, float) or np.isfinite(x) else None) for k, x in d.items()} for t, d in v["res"].items()}} for a, v in out.items()},
          open("runs/r047/score.json", "w"), indent=1, default=float)
