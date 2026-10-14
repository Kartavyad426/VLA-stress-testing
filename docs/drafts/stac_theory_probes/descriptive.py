"""res 10-14: descriptive checks on R-059 C1 for docs/STAC_GROOT_THEORY.html. CPU, read-only. UNVERIFIED (no val PASS).

  cd ../VLA-exp-rootsignals && .venvs/groot/bin/python -I ../VLA/docs/drafts/stac_theory_probes/descriptive.py

Reads runs/r059/c1/{joint_radius_rad/manifest.jsonl + npz, rollouts.jsonl, score_c.json}; prints JSON to stdout.
"""
import json
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

C = "runs/r059/c1/"


def au(p, n):
    r = rankdata(np.r_[p, n])
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


man = [json.loads(l) for l in open(C + "joint_radius_rad/manifest.jsonl")]
rol = [json.loads(l) for l in open(C + "rollouts.jsonl")]
E = json.load(open(C + "score_c.json"))["per_episode"]
rows, pairs = [], []
for m, r, e in zip(man, rol, E):
    assert m["rollout_id"] == r["rollout_id"]
    z = np.load(C + f"joint_radius_rad/{e['tag']}.npz")
    P, B = z["act_P"].astype(np.float64), z["act_B"].astype(np.float64)
    pos = np.array([s["obs_state"]["eef_pos"] for s in r["steps"]])
    d2o = np.array([s["obs_state"]["_gt_eef_to_nearest_object"] for s in r["steps"]])
    k = min(64, len(pos) - 1)                      # 4 forwards x 16 executed steps
    row = {"fail": not m["success"], "radius": m["magnitude"],
           "d2o_drop_64": float(d2o[0] - d2o[k]), "eef_disp_64": float(np.linalg.norm(pos[k] - pos[0]))}
    for f in range(2):
        mu = B[f, :, :16, :3].mean(0)
        row[f"cmd_xyz_f{f}"] = float(np.linalg.norm(P[f, :16, :3], axis=1).mean())
        row[f"rel_spread_f{f}"] = float(np.linalg.norm(B[f, :, :16, :3] - mu, axis=2).mean() / np.linalg.norm(mu, axis=1).mean())
        pairs.append({"fail": not m["success"], "stac": e["stac"][f],
                      "head_std": float(B[f, :, :16, :6].std(0).mean()), "tail_std": float(B[f, :, 16:40, :6].std(0).mean()),
                      "mag": float(np.abs(P[f, 16:40, :6]).mean() + np.abs(P[f + 1, :24, :6]).mean())})
    rows.append(row)

out = {"tag": "UNVERIFIED (res descriptive, no val PASS)", "n": len(rows), "auroc_fail_high": {}}
F = [x for x in rows if x["fail"]]; S = [x for x in rows if not x["fail"]]
for key in ("d2o_drop_64", "eef_disp_64", "cmd_xyz_f0", "cmd_xyz_f1", "rel_spread_f0", "rel_spread_f1", "radius"):
    out["auroc_fail_high"][key] = {"auroc": au([x[key] for x in F], [x[key] for x in S]),
                                   "median_fail": float(np.median([x[key] for x in F])), "median_succ": float(np.median([x[key] for x in S]))}
st = np.array([p["stac"] for p in pairs]); mag = np.array([p["mag"] for p in pairs]); fl = np.array([p["fail"] for p in pairs])
out["head_vs_tail_draw_std_median"] = {"head_0_16": float(np.median([p["head_std"] for p in pairs])),
                                       "tail_16_40": float(np.median([p["tail_std"] for p in pairs]))}
rho = spearmanr(st, mag)
out["stac_vs_motion_spearman"] = {"rho": float(rho.statistic), "p": float(rho.pvalue), "n_pairs": len(pairs), "note": "f0 and f1 rows pooled"}
q = np.quantile(mag, [1 / 3, 2 / 3]); b = np.digitize(mag, q)
out["stac_auroc_within_motion_tertiles"] = [{"tertile": t, "n": int((b == t).sum()), "n_fail": int((fl & (b == t)).sum()),
                                             "auroc": au(st[fl & (b == t)], st[~fl & (b == t)])} for t in range(3)]
prog = {}
for k in (16, 32, 64):                              # executed env steps = 1, 2, 4 forwards
    v = []
    for m, r in zip(man, rol):
        d = np.array([s["obs_state"]["_gt_eef_to_nearest_object"] for s in r["steps"]])
        v.append((m["magnitude"], not m["success"], float(d[0] - d[min(k, len(d) - 1)])))
    prog[k] = {"all": au([x[2] for x in v if x[1]], [x[2] for x in v if not x[1]]),
               "within_radius": {rad: au([x[2] for x in v if x[0] == rad and x[1]], [x[2] for x in v if x[0] == rad and not x[1]])
                                 for rad in (0.2, 0.3, 0.4)}}
out["progress_drop_auroc_fail_high_by_steps"] = prog
json.dump(out, sys.stdout, indent=1)
