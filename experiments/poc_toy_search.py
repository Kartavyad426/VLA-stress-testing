"""PoC toy, stage 2: run the instance as a success search (D14). CPU.

Pool: the 50 human demos for task 984 in data/libero_raw, plus every
successful scene-984 rollout in runs/r047 (both axes). Every grasp onset in the
pool is a candidate. Filter: it meets the success criterion. Match, at that
onset, on every situation dimension (the mechanism is unknown, so all are held):

    offset   eef -> target, each axis within OFFSET_TOL_M
    gripper  rotation within ROT_TOL_DEG of the failure's
    layout   target xy within LAYOUT_TOL_M, and each neighbour's xy relative
             to the target within LAYOUT_TOL_M

Tolerances are ILLUSTRATIVE (P2 is pending). The funnel is reported per filter
alone and cumulatively, with the nearest candidates by offset, so an empty
result says which dimension emptied it.

    .venvs/libero-plus/bin/python experiments/poc_toy_search.py
    -> runs/poc_toy/search.json, runs/poc_toy/demo_frames/*.png
"""
import json, os, sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poc_toy_common import (RAW, TASK_FILE, R047, ROOT, TARGET, NEIGHBOURS, TASK_ID,
                            WINDOW, LIFT_M, load_model, site_rot, angle_deg,
                            grasp_events, kin_model, rollout_events)
import mujoco

OUT = os.path.join(ROOT, "runs/poc_toy")
OFFSET_TOL_M, ROT_TOL_DEG, LAYOUT_TOL_M = 0.015, 20.0, 0.03
N_NEAREST = 5


def demo_events(f):
    out = []
    for name in sorted(f["data"], key=lambda s: int(s.split("_")[1])):
        d = f["data"][name]
        m, D = load_model(d.attrs["model_file"])
        st = d["states"][:]
        objs, eef, rots = [], [], []
        for t in range(len(st)):
            D.qpos[:] = st[t, 1:1 + m.nq]
            mujoco.mj_forward(m, D)
            objs.append({n: D.body(n).xpos.copy() for n in [TARGET] + NEIGHBOURS})
            eef.append(D.site("gripper0_grip_site").xpos.copy())
            rots.append(site_rot(m, D))
        # stored ee_pos must agree with the restored site (same frame as rollouts).
        # obs/ee_pos[t] is recorded AFTER step t, i.e. it is the site at states[t+1].
        err = float(np.max(np.linalg.norm(np.asarray(eef[1:]) - d["obs/ee_pos"][:-1], axis=1)))
        cmd = list(d["actions"][:, 6] > 0)
        ap = np.abs(d["obs/gripper_states"][:]).sum(1)
        ev = grasp_events(cmd, [o[TARGET][2] for o in objs], eef, objs, rots, ap)
        for e in ev:
            e.update(source="human demo", ref=name, n_steps=len(st), restore_err_m=round(err, 5))
        out += ev
    return out


def match(e, q):
    off = np.abs(np.asarray(e["offset_m"]) - np.asarray(q["offset_m"]))
    rot = angle_deg(e["_R"], q["_R"])
    lay_t = float(np.linalg.norm(np.asarray(e["target_xyz"][:2]) - np.asarray(q["target_xyz"][:2])))
    lay_n = max(float(np.linalg.norm(np.asarray(e["neighbours_rel_m"][n][:2]) -
                                     np.asarray(q["neighbours_rel_m"][n][:2]))) for n in NEIGHBOURS)
    return {"offset_ok": bool((off <= OFFSET_TOL_M).all()), "offset_dev_cm": (off * 100).round(1).tolist(),
            "offset_dist_cm": round(float(np.linalg.norm(off)) * 100, 1),
            "rot_ok": rot <= ROT_TOL_DEG, "rot_dev_deg": round(rot, 1),
            "layout_ok": lay_t <= LAYOUT_TOL_M and lay_n <= LAYOUT_TOL_M,
            "target_xy_dev_cm": round(lay_t * 100, 1), "worst_neighbour_dev_cm": round(lay_n * 100, 1)}


def main():
    m, D = kin_model()
    rows = [json.loads(l) for l in open(R047)]
    q = rollout_events(rows[14], m, D)[0][0]          # the root grasp onset (t=31)
    assert q["t"] == 31 and not q["success"]

    with h5py.File(os.path.join(RAW, TASK_FILE), "r") as f:
        assert f["data"].attrs["problem_info"].find(rows[14]["instruction"]) >= 0
        pool = demo_events(f)
    n_demo_ev = len(pool)
    n_ours = 0
    for i, r in enumerate(rows):
        if r["scene_descriptor"]["task_id"] != TASK_ID or not r["success"]:
            continue
        n_ours += 1
        ev, _ = rollout_events(r, m, D)
        for e in ev:
            e.update(source="our rollout", ref=f"r047 row {i}",
                     perturbation=r["perturbation"])
        pool += ev

    for e in pool:
        e["match"] = match(e, q)
    srcs = ["human demo", "our rollout"]
    funnel = {}
    for s in srcs:
        P = [e for e in pool if e["source"] == s]
        ok = [e for e in P if e["success"]]
        funnel[s] = {
            "episodes": 50 if s == "human demo" else n_ours,
            "grasp_onsets": len(P),
            "meets_success_criterion": len(ok),
            "alone": {k: sum(e["match"][k] for e in ok) for k in ("offset_ok", "rot_ok", "layout_ok")},
            "cumulative": {
                "offset": sum(e["match"]["offset_ok"] for e in ok),
                "offset+gripper": sum(e["match"]["offset_ok"] and e["match"]["rot_ok"] for e in ok),
                "offset+gripper+layout": sum(e["match"]["offset_ok"] and e["match"]["rot_ok"]
                                             and e["match"]["layout_ok"] for e in ok)},
        }
    ok_all = [e for e in pool if e["success"]]
    nearest = sorted(ok_all, key=lambda e: e["match"]["offset_dist_cm"])[:N_NEAREST]
    hits = [e for e in ok_all if all(e["match"][k] for k in ("offset_ok", "rot_ok", "layout_ok"))]
    # sensitivity: full matches as the per-axis offset tolerance varies
    sweep = {}
    for tol_cm in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        sweep[tol_cm] = {s: sum(1 for e in ok_all if e["source"] == s and max(e["match"]["offset_dev_cm"]) <= tol_cm
                                and e["match"]["rot_ok"] and e["match"]["layout_ok"]) for s in srcs}
        sweep[tol_cm]["human demo, offset only"] = sum(1 for e in ok_all if e["source"] == "human demo"
                                                       and max(e["match"]["offset_dev_cm"]) <= tol_cm)
    offs = np.asarray([e["offset_m"] for e in ok_all if e["source"] == "human demo"]) * 100

    # demo frames at onset for the nearest human-demo candidates (policy orientation)
    import imageio.v2 as imageio
    fd = os.path.join(OUT, "demo_frames"); os.makedirs(fd, exist_ok=True)
    with h5py.File(os.path.join(RAW, TASK_FILE), "r") as f:
        for e in [e for e in nearest if e["source"] == "human demo"][:3] + \
                 [e for e in sorted(ok_all, key=lambda e: e["match"]["offset_dist_cm"])
                  if e["source"] == "human demo"][3:4]:
            d = f["data"][e["ref"]]
            for t in (e["t"], min(e["t"] + 20, e["n_steps"] - 1)):
                for cam, key in (("agent", "obs/agentview_rgb"), ("wrist", "obs/eye_in_hand_rgb")):
                    imageio.imwrite(os.path.join(fd, f"{e['ref']}_t{t:03d}_{cam}.png"), d[key][t][::-1, ::-1])

    strip = lambda e: {k: v for k, v in e.items() if k != "_R"}
    res = {"query": strip(q), "tolerances": {"offset_per_axis_m": OFFSET_TOL_M, "gripper_rot_deg": ROT_TOL_DEG,
                                             "layout_xy_m": LAYOUT_TOL_M, "status": "illustrative (P2 pending)"},
           "success_criterion": f"target lifted >= {LIFT_M} m within {WINDOW} steps",
           "funnel": funnel, "offset_tolerance_sweep_cm": sweep, "n_hits": len(hits), "hits": [strip(e) for e in hits],
           "nearest_by_offset": [strip(e) for e in nearest],
           "human_demo_successful_onset_offsets_cm": {
               "n": int(len(offs)), "median": np.median(offs, 0).round(1).tolist(),
               "min": offs.min(0).round(1).tolist(), "max": offs.max(0).round(1).tolist(),
               "all": offs.round(1).tolist()},
           "our_successful_onset_offsets_cm": [[round(x * 100, 1) for x in e["offset_m"]] + [e["ref"]]
                                               for e in ok_all if e["source"] == "our rollout"],
           "demo_restore_err_max_m": max(e["restore_err_m"] for e in pool if e["source"] == "human demo"),
           "n_demo_grasp_onsets": n_demo_ev}
    json.dump(res, open(os.path.join(OUT, "search.json"), "w"), indent=1)
    print(json.dumps(funnel, indent=1)); print("hits", len(hits)); print("sweep", sweep)
    for e in nearest:
        print(e["source"], e["ref"], e["t"], e["offset_m"], e["match"])
    print("demo offsets median/min/max", res["human_demo_successful_onset_offsets_cm"]["median"],
          res["human_demo_successful_onset_offsets_cm"]["min"], res["human_demo_successful_onset_offsets_cm"]["max"],
          "restore err", res["demo_restore_err_max_m"])


if __name__ == "__main__":
    main()
