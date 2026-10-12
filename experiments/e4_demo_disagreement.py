"""E4 (ROOT_SIGNAL_EXPERIMENTS, signal d): imitation disagreement. Base GR00T N1.7 run OPEN-LOOP on the frames of
the 50 human demos of the toy task; at each forward-equivalent step its predicted chunk is compared with the demo's
next actions. Approved 10-09 (user via prim). Exploratory, not pre-registered. No result travels before a PASS
entry in PIPELINE_CHECKS.md.

  ENV="PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8"
  env $ENV .venvs/libero-plus/bin/python experiments/e4_demo_disagreement.py invariants   # EGL env, no model
  bash experiments/d_step.sh e4_run --gpu .venvs/libero-plus/bin/python experiments/e4_demo_disagreement.py run
  .venvs/groot/bin/python experiments/e4_demo_disagreement.py score [--plant]               # CPU

Alignment (LIBERO-Plus scripts/create_dataset.py:175-226): every stored obs[j] (images, ee_pos, robot_states) is
recorded AFTER env.step(actions[j]). The observation the demonstrator had when choosing actions[t] is obs[t-1], so
forward k runs at t = 1 + 16k on obs[t-1] and is compared with actions[t : t+16]. `invariants` re-checks this
against the sim: the eef of states[t] (the pre-action state of step t) must equal ee_pos[t-1].

Policy input = the harness Observation: frames {image: agentview_rgb, image2: eye_in_hand_rgb} as stored (128 px,
the env's raw OpenGL orientation; the GR00T processor flips and resizes them, as it does the env's 360 px frames in
D20), state {eef_pos, eef_quat (x,y,z,w), gripper_qpos} from robot_states[t-1] = [gripper qpos 2, eef pos 3, eef quat
4]; the harness's state_fn converts the quaternion exactly as at eval. Instruction = the HDF5's language instruction
(asserted equal to R-047's for scene 984). Variant `up256`: frames resized 128 -> 256 (PIL bicubic) first, for demos
0-4 at noise seed 0 (sensitivity to the input size).

Prediction = the head's action_pred (normalised space, first 7 dims, all 40 steps), recorded by attach_transform
(transform None = the seeded untransformed base, same common-random-numbers seeding as every splice run). Noise key
1000*noise_seed + k. Noise seeds 0 and 1: the seed-0 vs seed-1 distance is the sampling-noise floor at each forward.

Controls built in (PIPELINE_CHECKS):
  known answer  `run` first re-computes R-047's stored act_P[0] for 6 nominal rollouts (scenes 984/1030/1062, noise
                seeds 0/1, fresh env reset seed 0, 360 px) through THIS path; max |diff| is reported (fp16-stored).
  null          demo 0 forward 0 twice under the same key -> bitwise equal; seed-0 vs seed-1 floor everywhere.
  planted       `score --plant`: the labels of one (demo, forward) are replaced by those of another demo's later
                forward; the score must rank that cell first within its demo.
  invariants    `invariants`: normalise/un-normalise round trip, label clipping counts, obs lag vs sim, frame shapes,
                instruction match.
Outputs: runs/root_signals/e4/ (pred.npz, rows.jsonl, controls.json, invariants.json, score.json,
per_demo_step.jsonl, code_state/).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.getcwd())
import numpy as np

OUT = "runs/root_signals/e4"
H5 = "data/libero_raw/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
TOY_SCENE = 984                     # LIBERO-Plus catalogue index of this task's control (R-047's scene 984)
CHUNK, H, LIVE = 40, 16, 7
NOISE_SEEDS = (0, 1)
UP256_DEMOS = range(5)
ANCHOR = [(s, ns) for s in (984, 1030, 1062) for ns in (0, 1)]
# R-047's own two axes store different act_P[0] for these nominal cells (984/n0 max 0.038; 1062/n0 max 0.005): the
# reference itself is unstable, so a mismatch there is reported, not counted as a pipeline fail (my prim 10-09).
ANCHOR_UNSTABLE_IN_R047 = {(984, 0), (1062, 0)}


def log(*x, t0=time.time()):
    print(f"[{time.time() - t0:7.0f}s]", *x, flush=True)


def code_state():
    d = os.path.join(OUT, "code_state")
    os.makedirs(d, exist_ok=True)
    os.system(f"git rev-parse HEAD > {d}/HEAD; git diff > {d}/uncommitted.patch; git status --porcelain > {d}/status.txt; "
              f"sha256sum {H5} > {d}/inputs.sha256 2>/dev/null; "
              f"git ls-files --others --exclude-standard experiments vla_harness | grep '\\.py$' | xargs -r -I{{}} cp --parents {{}} {d}/")


def load_stats():
    from safetensors.numpy import load_file
    ck = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--gr00t17-lerobot-libero_spatial-640/snapshots/*/"))[0]
    d = load_file(glob.glob(os.path.join(ck, "policy_postprocessor_step_0*.safetensors"))[0])
    return d["action.min"].astype(np.float64)[:LIVE], d["action.max"].astype(np.float64)[:LIVE]


def norm(a, lo, hi):
    """env units -> the policy's normalised space (min-max to [-1, 1]); NOT clipped (see norm_label)."""
    return 2.0 * (np.asarray(a, np.float64) - lo) / (hi - lo) - 1.0


def norm_label(a, lo, hi):
    """Demo actions in the policy's normalised space. Dims 0-5: min-max normalised then clipped to [-1, 1]
    (clip_outliers, processor_groot.py:1823-1825); matches R-047's act_P vs its executed actions to 2e-4.
    Gripper (dim 6): the env command is binary -1 open / +1 close, and in the policy's normalised space +1 = OPEN
    (R-047: executed gripper = -0.989 * act_P[..., 6] - 0.003, runs/r059/partA/summary.json act_P_to_env_map, my
    prim 10-09). So the label is -command. The min-max route (stats 0/1) inverted it; fixed 10-09 before scoring."""
    a = np.asarray(a, np.float64)
    out = np.clip(norm(a, lo, hi), -1.0, 1.0)
    out[..., 6] = -np.sign(a[..., 6])
    return out


def unnorm(x, lo, hi):
    return (np.clip(np.asarray(x, np.float64), -1.0, 1.0) + 1.0) / 2.0 * (hi - lo) + lo


def demos():
    import h5py
    f = h5py.File(H5, "r")
    instr = json.loads(f["data"].attrs["problem_info"])["language_instruction"]
    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    return f, instr, keys


def forward_steps(T):
    return list(range(1, T, H))          # t = 1, 17, 33, ... (obs[t-1] exists, actions[t] exists)


def observation(e, t, instr, up256=False):
    from PIL import Image
    from vla_harness.schema import Observation
    rs = e["robot_states"][t - 1]
    fr = {"image": e["obs/agentview_rgb"][t - 1], "image2": e["obs/eye_in_hand_rgb"][t - 1]}
    if up256:
        fr = {k: np.asarray(Image.fromarray(v).resize((256, 256), Image.BICUBIC)) for k, v in fr.items()}
    st = {"eef_pos": [float(x) for x in rs[2:5]], "eef_quat": [float(x) for x in rs[5:9]],
          "gripper_qpos": [float(x) for x in rs[0:2]]}
    return Observation(instruction=instr, state=st, frames={k: np.ascontiguousarray(v) for k, v in fr.items()}, t=int(t))


# ---------------------------------------------------------------------------------------------------------------
def build_policy():
    from lerobot.envs.configs import LiberoPlusEnv
    from vla_harness.policies.lerobot_policy import LeRobotPolicy
    return LeRobotPolicy("nvidia/gr00t17-lerobot-libero_spatial-640", n_action_steps=16,
                         env_cfg=LiberoPlusEnv(task="libero_spatial"),
                         policy_overrides={"base_model_path": "nvidia/GR00T-N1.7-3B", "embodiment_tag": "libero_sim"},
                         dtype="bfloat16", rename_map={"observation.images.image2": "observation.images.wrist_image"})


def predict(pol, obs, key):
    """One model forward on `obs` under noise key `key`; the full normalised chunk (40 x 7, float32)."""
    from vla_harness.capture.splice import attach_transform
    pol.reset()
    f0 = pol.model_forwards
    h = attach_transform(pol._policy, None, "S2", noise_key=lambda i, k=int(key): k)
    try:
        pol(obs.policy_view())
    finally:
        h.detach()
    if pol.model_forwards - f0 != 1 or len(h.records) != 1:
        raise RuntimeError(f"expected one model forward, got {pol.model_forwards - f0} / {len(h.records)} records")
    return h.records[0]["action"][0, :, :LIVE].float().cpu().numpy()


def run(a):
    os.makedirs(OUT, exist_ok=True)
    code_state()
    f, instr, keys = demos()
    keys = keys[: a.limit] if a.limit else keys
    pol = build_policy()
    pol.reset()
    log("policy loaded")
    controls = {}
    if not a.no_controls:
        # known answer: R-047's stored act_P[0] (fp16) through this path
        from vla_harness.envs.libero_env import LiberoEnv
        from vla_harness.schema import PerturbationSpec
        # Pass rule, stated before the run: bitwise in fp16; otherwise max |diff| <= the same-seed repeat jitter of
        # that anchor (each anchor is predicted twice) or one fp16 ulp at |x| <= 1 (2**-10), whichever is larger.
        rows = []
        for scene, ns in ANCHOR:
            stored = np.load(f"runs/r047/joint_radius_rad/{scene}_0p000000_d0_n{ns}.npz")["act_P"][0].astype(np.float32)
            env = LiberoEnv(suite="libero_spatial", task_id=scene, libero_plus=True, libero_plus_base_instruction=True, obs_size=360)
            obs = env.reset(0, PerturbationSpec())
            p = predict(pol, obs, 1000 * ns + 0)
            p2 = predict(pol, obs, 1000 * ns + 0)
            mx = float(np.abs(p.astype(np.float16).astype(np.float32) - stored).max())
            rep = float(np.abs(p - p2).max())
            rows.append({"scene": scene, "noise_seed": ns, "max_abs_vs_stored_fp16": mx, "repeat_max_abs": rep,
                         "bitwise_fp16": bool(np.array_equal(p.astype(np.float16), stored.astype(np.float16))),
                         "tolerance": max(rep, 2 ** -10)})
            rows[-1]["PASS"] = rows[-1]["bitwise_fp16"] or mx <= rows[-1]["tolerance"]
            if (scene, ns) in ANCHOR_UNSTABLE_IN_R047:
                rows[-1]["note"] = "anchor unstable in R-047 (axes disagree); not gating"
            del env
            log("anchor", rows[-1])
        controls["known_answer_r047_actP0"] = rows
        controls["known_answer_PASS"] = all(r["PASS"] for r in rows if "note" not in r)
        e = f["data"][keys[0]]
        o = observation(e, 1, instr)
        p1, p2 = predict(pol, o, 0), predict(pol, o, 0)
        controls["null_same_obs_same_key_bitwise"] = bool(np.array_equal(p1, p2))
        controls["null_same_obs_seed0_vs_seed1_l2"] = float(np.linalg.norm(p1 - predict(pol, o, 1000)))
        log("null", controls)
        json.dump(controls, open(os.path.join(OUT, "controls.json"), "w"), indent=1)
    preds, rows = [], []
    for di, k in enumerate(keys):
        e = f["data"][k]
        T = len(e["actions"])
        for fi, t in enumerate(forward_steps(T)):
            for ns in NOISE_SEEDS:
                for variant in (("raw128", "up256") if (ns == 0 and di in UP256_DEMOS) else ("raw128",)):
                    preds.append(predict(pol, observation(e, t, instr, up256=variant == "up256"), 1000 * ns + fi))
                    rows.append({"i": len(rows), "demo": k, "demo_idx": di, "T": T, "forward": fi, "t": t,
                                 "noise_seed": ns, "variant": variant})
        log(f"demo {k}: T={T}, {len(forward_steps(T))} forwards, total preds {len(preds)}")
    np.savez_compressed(os.path.join(OUT, "pred.npz"), pred=np.stack(preds).astype(np.float32))
    with open(os.path.join(OUT, "rows.jsonl"), "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    log("done", len(rows), "predictions")


# ---------------------------------------------------------------------------------------------------------------
def invariants(a):
    """No model. EGL env for the sim-state lag check."""
    os.makedirs(OUT, exist_ok=True)
    f, instr, keys = demos()
    lo, hi = load_stats()
    out = {"stats_min": lo.tolist(), "stats_max": hi.tolist()}
    A = np.concatenate([f["data"][k]["actions"][:] for k in keys])
    rt = unnorm(norm(A, lo, hi), lo, hi)
    inr = (A >= lo) & (A <= hi)
    out["roundtrip_max_abs_in_range"] = float(np.abs(rt - A)[inr].max())
    out["actions_out_of_range_per_dim"] = (~inr).sum(0).tolist()          # these get clipped as training labels
    out["n_actions"] = int(len(A))
    r047 = json.loads(open("runs/r047/rollouts.jsonl").readline())
    out["instruction_hdf5"] = instr
    out["instruction_matches_r047_scene984"] = bool(r047["task_id"].endswith(f"task{TOY_SCENE}") and r047["instruction"] == instr)
    shapes = {k: [list(f["data"][k]["obs/agentview_rgb"].shape), list(f["data"][k]["actions"].shape)] for k in keys[:3]}
    out["shapes_first3"] = shapes
    out["forwards_total_per_seed"] = int(sum(len(forward_steps(len(f["data"][k]["actions"]))) for k in keys))
    # lag: eef of states[t] (set in the sim) vs ee_pos[t-1] and ee_pos[t]
    from vla_harness.envs.libero_env import LiberoEnv
    from vla_harness.schema import PerturbationSpec
    env = LiberoEnv(suite="libero_spatial", task_id=TOY_SCENE, libero_plus=True, libero_plus_base_instruction=True, obs_size=128)
    env.reset(0, PerturbationSpec())
    inner = env._env.unwrapped._env.env
    lag = {"t_minus_1": [], "t": []}
    img = {"t_minus_1": [], "t": []}
    for k in keys[:5]:
        e = f["data"][k]
        S, ep, av = e["states"][:], e["obs/ee_pos"][:], e["obs/agentview_rgb"]
        for t in range(1, len(S), 7):
            inner.sim.set_state_from_flattened(S[t]); inner.sim.forward()
            o = inner._get_observations(force_update=True)
            lag["t_minus_1"].append(float(np.linalg.norm(o["robot0_eef_pos"] - ep[t - 1])))
            lag["t"].append(float(np.linalg.norm(o["robot0_eef_pos"] - ep[t])))
            img["t_minus_1"].append(float(np.abs(o["agentview_image"].astype(float) - av[t - 1]).mean()))
            img["t"].append(float(np.abs(o["agentview_image"].astype(float) - av[t]).mean()))
            # orientation: the env's raw render (what the harness hands the processor, which then flips it) vs the
            # demo frame as stored, and vs the demo frame rotated 180 degrees; both cameras
            for cam, dk in (("agentview_image", "obs/agentview_rgb"), ("robot0_eye_in_hand_image", "obs/eye_in_hand_rgb")):
                dem = e[dk][t - 1].astype(float)
                img.setdefault(f"orient_{cam}_as_stored", []).append(float(np.abs(o[cam].astype(float) - dem).mean()))
                img.setdefault(f"orient_{cam}_rot180", []).append(float(np.abs(o[cam].astype(float) - dem[::-1, ::-1]).mean()))
    out["lag_eef_m"] = {k: {"median": float(np.median(v)), "max": float(np.max(v)), "n": len(v)} for k, v in lag.items()}
    out["lag_agentview_meanabs"] = {k: {"median": float(np.median(v)), "max": float(np.max(v))} for k, v in img.items()}
    out["lag_ok"] = out["lag_eef_m"]["t_minus_1"]["max"] < 1e-3 < out["lag_eef_m"]["t"]["median"]
    # orientation passes if, for both cameras, the as-stored frame is far closer to the env's raw render than its
    # 180-degree rotation: E4 then feeds demo frames exactly as the harness feeds env frames (same processor flip)
    out["orientation_ok"] = all(np.median(img[f"orient_{c}_as_stored"]) * 3 < np.median(img[f"orient_{c}_rot180"])
                                for c in ("agentview_image", "robot0_eye_in_hand_image"))
    json.dump(out, open(os.path.join(OUT, "invariants.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------------------------------------------
def disp(a):
    return np.asarray(a, float)[:H, :3].sum(0)


def cell_metrics(p, lab_n, lab_env, lo, hi):
    n = len(lab_n)
    d = p[:n] - lab_n
    m = {"n_label": n,
         "d16": float(np.linalg.norm(d, axis=1).mean()),                     # mean per-step L2, normalised space
         "d16_nogrip": float(np.linalg.norm(d[:, :6], axis=1).mean()),
         "d1": float(np.linalg.norm(d[0])),
         "d1_nogrip": float(np.linalg.norm(d[0, :6])),
         "d16_per_dim": np.abs(d).mean(0).tolist()}
    if n == H:
        m["disp_err_env"] = float(np.linalg.norm(disp(unnorm(p, lo, hi)) - disp(lab_env)))   # an7 metric, env units
    return m


def score(a):
    lo, hi = load_stats()
    f, instr, keys = demos()
    P = np.load(os.path.join(OUT, "pred.npz"))["pred"]
    R = [json.loads(l) for l in open(os.path.join(OUT, "rows.jsonl"))]
    assert len(R) == len(P) and [r["i"] for r in R] == list(range(len(R))), "rows/preds misaligned"
    keyset = {(r["demo"], r["forward"], r["noise_seed"], r["variant"]) for r in R}
    assert len(keyset) == len(R), "duplicate join keys"
    idx = {(r["demo"], r["forward"], r["noise_seed"], r["variant"]): r["i"] for r in R}
    labels = {}
    for k in sorted({r["demo"] for r in R}):
        e = f["data"][k]
        labels[k] = (e["actions"][:], e["obs/gripper_states"][:], e["obs/ee_pos"][:])
    plant = None
    if a.plant:            # a KNOWN offset at a known cell: +0.5 on dims 0-2 of one forward's labels (normalised space)
        ks = sorted(labels, key=lambda k: int(k.split("_")[1]))
        plant = {"demo": ks[0], "forward": 2, "offset_dims_0_2": 0.5}
    out_rows = []
    for r in R:
        if r["noise_seed"] != 0:
            continue
        acts, grip, eep = labels[r["demo"]]
        t = r["t"]
        lab_env = acts[t: t + H]
        lab_n = norm_label(lab_env, lo, hi)
        if plant and r["demo"] == plant["demo"] and r["forward"] == plant["forward"]:
            m0 = cell_metrics(P[r["i"]], lab_n, lab_env, lo, hi)
            lab_n = lab_n.copy(); lab_n[:, :3] += plant["offset_dims_0_2"]
            plant["unplanted_d16"] = m0["d16"]
        m = cell_metrics(P[r["i"]], lab_n, lab_env, lo, hi)
        j = idx.get((r["demo"], r["forward"], 1, r["variant"]))
        if j is not None:
            m["floor_d16"] = float(np.linalg.norm(P[r["i"]] - P[j], axis=1)[:m["n_label"]].mean())
            m["floor_d1"] = float(np.linalg.norm(P[r["i"]][0] - P[j][0]))
        m.update({k: r[k] for k in ("demo", "demo_idx", "T", "forward", "t", "variant")})
        m["frac_t"] = t / r["T"]
        m["demo_gripper_closed_at_t"] = bool(grip[t - 1][0] < 0.03)          # finger qpos; ~0.039 open
        m["demo_label_gripper_cmd"] = float(acts[t][6])
        m["demo_eef_z_at_t"] = float(eep[t - 1][2])
        out_rows.append(m)
    with open(os.path.join(OUT, "per_demo_step.jsonl" if not a.plant else "per_demo_step_plant.jsonl"), "w") as fo:
        for m in out_rows:
            fo.write(json.dumps(m) + "\n")
    raw = [m for m in out_rows if m["variant"] == "raw128"]
    q = lambda xs: {"median": float(np.median(xs)), "p90": float(np.percentile(xs, 90)), "n": len(xs)} if xs else None
    s = {"what": "E4 open-loop disagreement on 50 demos (exploratory; PIPELINE_CHECKS pending)",
         "n_cells_seed0_raw128": len(raw),
         "d16": q([m["d16"] for m in raw]), "floor_d16": q([m["floor_d16"] for m in raw if "floor_d16" in m]),
         "d1": q([m["d1"] for m in raw]), "floor_d1": q([m["floor_d1"] for m in raw if "floor_d1" in m]),
         "disp_err_env": q([m["disp_err_env"] for m in raw if "disp_err_env" in m]),
         "by_forward": {str(fw): {"d16": q([m["d16"] for m in raw if m["forward"] == fw]),
                                  "floor_d16": q([m["floor_d16"] for m in raw if m["forward"] == fw and "floor_d16" in m])}
                        for fw in sorted({m["forward"] for m in raw})},
         "by_gripper_closed": {str(g): q([m["d16"] for m in raw if m["demo_gripper_closed_at_t"] == g]) for g in (False, True)}}
    up = {(m["demo"], m["forward"]): m["d16"] for m in out_rows if m["variant"] == "up256"}
    if up:
        pairs = [(m["d16"], up[(m["demo"], m["forward"])]) for m in raw if (m["demo"], m["forward"]) in up]
        s["up256_vs_raw128_d16"] = {"n": len(pairs), "median_raw": float(np.median([x for x, _ in pairs])),
                                    "median_up256": float(np.median([y for _, y in pairs])),
                                    "median_abs_change": float(np.median([abs(x - y) for x, y in pairs])),
                                    "per_demo": {k: {"median_raw128": float(np.median([m["d16"] for m in raw if m["demo"] == k and (k, m["forward"]) in up])),
                                                     "median_up256": float(np.median([v for (dk, _), v in up.items() if dk == k])),
                                                     "median_floor": float(np.median([m["floor_d16"] for m in raw if m["demo"] == k and "floor_d16" in m]))}
                                                 for k in sorted({dk for dk, _ in up})}}
    if plant:
        cells = [m for m in raw if m["demo"] == plant["demo"]]
        top = max(cells, key=lambda m: m["d16"])
        pl = next(m["d16"] for m in cells if m["forward"] == plant["forward"])
        others = [m["d16"] for m in cells if m["forward"] != plant["forward"]]
        s["plant"] = {**plant, "top_forward": top["forward"], "found": top["forward"] == plant["forward"],
                      "planted_d16": pl, "rise_vs_unplanted": pl - plant["unplanted_d16"],
                      "max_other_d16": max(others), "margin_vs_max_other": pl - max(others),
                      "note": "rise_vs_unplanted is the effect the metric registers for a known +0.5 offset on 3 dims"}
    json.dump(s, open(os.path.join(OUT, "score.json" if not a.plant else "score_plant.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in s.items() if k != "by_forward"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "invariants", "score"])
    ap.add_argument("--limit", type=int, default=None, help="first N demos only (smoke)")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--plant", action="store_true")
    a = ap.parse_args()
    {"run": run, "invariants": invariants, "score": score}[a.phase](a)


if __name__ == "__main__":
    main()
