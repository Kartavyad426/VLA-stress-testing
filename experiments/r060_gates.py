"""R-060 part 0 gates (RESULTS.md R-060, "Part 0"). Writes runs/r060/gates/<gate>.json.
All three run on the CPU: no GR00T on the GPU. G0b loads GR00T (bf16) on the CPU and is
still run under flock /tmp/vla_gpu.lock, so it can never overlap a GPU GR00T load in RAM.
Renders use MUJOCO_GL=egl, the eval backend, so G0c checks the renders the eval sees.

  ENV="PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8"
  PY=.venvs/libero-plus/bin/python
  env $ENV $PY experiments/r060_gates.py g0a
  env $ENV $PY experiments/r060_gates.py g0c
  env $ENV flock /tmp/vla_gpu.lock $PY experiments/r060_gates.py g0b

g0a  Camera order. Builds the checkpoint's own pre-processors (no model) and runs one
     observation through them twice: agent view black / wrist white, then swapped.
     Records the pack step's video_modality_keys, `_ordered_image_keys`, the image-token
     runs in input_ids, and which camera fills the first image block of pixel_values.
     Qwen3-VL scatters image i's merged tokens into the i-th block of image positions
     (transformers 5.5.4 modeling_qwen3_vl.py:1161 split by grid_thw, :1297
     masked_scatter), so the first pixel_values block is the first block of image tokens.
     PASS = the order is determined consistently by all three readings. The order itself
     decides A2-S1 (agent first -> A2-S1 runs).
g0c  Training-pair exactness, render only (no model).
     camera: for each variant, the CANONICAL env of its scene (the R-029 control) at the
       same init state, rendered with `perturbed_frames(variant)`, must equal the variant
       env's own render bitwise; the two envs' physics state must be identical; and the
       variant env's `nominal_frames()` must equal the canonical env's own render (the
       pair's N side).
     noise: the canonical env's `perturbed_frames(variant, seed)` must equal LIBERO-Plus's
       own wrapper dispatch (env_wrapper.py step()) applied to the same clean frame under
       the same np.random seed, bitwise; deterministic families (gaussian, zoom) are also
       compared against the variant env's own reset frame end to end.
g0b  Zero-init conformance. GR00T on the CPU, ten forward-0 observations (the R-048
     camera instances, seed 0), noise_key = 1000*seed + forward. Per observation:
     base, base again (determinism), FTM gamma=beta=0 at S2, at S1 -> all must equal base
     bitwise (torch.equal on action_pred, all dims). Liveness: a non-zero FTM at each
     site must change the action (else the identity is vacuous).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.getcwd())

import numpy as np

OUT = "runs/r060/gates"
CKPT = "nvidia/gr00t17-lerobot-libero_spatial-640"
CLS = "third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"
R048 = "experiments/repro/r048_selection_cam.json"


def catalogue():
    """task index (LiberoEnv task_id) -> catalogue row; index = catalogue id - 1
    (checked: index 635 is id 636, the R-048 view_22_15 variant)."""
    rows = json.load(open(CLS))["libero_spatial"]
    return {r["id"] - 1: r for r in rows}


def control_index(cat: dict, variant_name: str) -> int:
    """The R-029 control of a variant's scene: '<scene>_language_1_view_0_0_100_0_0_initstate_0'
    (index 1030 for next-to-ramekin, as in r048_selection_cam.json)."""
    scene = variant_name.split("_view_")[0]
    want = f"{scene}_language_1_view_0_0_100_0_0_initstate_0"
    hits = [i for i, r in cat.items() if r["name"] == want]
    if len(hits) != 1:
        raise RuntimeError(f"control for {scene!r}: {len(hits)} matches")
    return hits[0]


def make_env(task_id):
    from vla_harness.envs.libero_env import LiberoEnv
    return LiberoEnv(suite="libero_spatial", task_id=task_id, libero_plus=True,
                     libero_plus_base_instruction=True, obs_size=360)


def build_policy(device):
    from lerobot.envs.configs import LiberoPlusEnv
    from vla_harness.policies.lerobot_policy import LeRobotPolicy
    over = {"groot_n1_7_vlm_encode_v1": {"device": device}} if device == "cpu" else None
    return LeRobotPolicy(CKPT, device=device, n_action_steps=16,
                         env_cfg=LiberoPlusEnv(task="libero_spatial"),
                         policy_overrides={"base_model_path": "nvidia/GR00T-N1.7-3B", "embodiment_tag": "libero_sim"},
                         dtype="bfloat16", rename_map={"observation.images.image2": "observation.images.wrist_image"},
                         preprocessor_overrides=over)


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{name}.json")
    json.dump(obj, open(p, "w"), indent=1, default=str)
    print(json.dumps({k: obj[k] for k in obj if k not in ("rows",)}, indent=1, default=str))
    print("wrote", p)


def code_state():
    os.makedirs(OUT, exist_ok=True)
    os.system(f"git rev-parse HEAD > {OUT}/HEAD; git diff > {OUT}/uncommitted.patch; "
              f"git status --porcelain > {OUT}/status.txt")


# --- G0a ---------------------------------------------------------------------------
def g0a(_a):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies import make_pre_post_processors

    pol = build_policy("cpu")
    cfg = PreTrainedConfig.from_pretrained(CKPT)
    cfg.n_action_steps = 16
    for k, v in pol.policy_overrides.items():
        setattr(cfg, k, v)
    cfg.pretrained_path = CKPT
    cfg.device = "cpu"
    pre_over = {"device_processor": {"device": "cpu"},
                "rename_observations_processor": {"rename_map": dict(pol.rename_map)}}
    for step, kv in pol.preprocessor_overrides.items():
        pre_over.setdefault(step, {}).update(kv)
    pol._pre, pol._post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=CKPT,
                                                   preprocessor_overrides=pre_over)
    pol._env_pre, pol._env_post = make_env_pre_post_processors(env_cfg=pol.env_cfg, policy_cfg=cfg)

    pack = [s for s in pol._pre.steps if hasattr(s, "_ordered_image_keys")]
    if len(pack) != 1:
        raise RuntimeError(f"expected one pack step with _ordered_image_keys, got {len(pack)}")
    pack = pack[0]

    env = make_env(1030)
    from vla_harness.schema import PerturbationSpec
    obs = env.reset(0, PerturbationSpec())
    H, W, _ = obs.frames["image"].shape

    def run(agent_val, wrist_val):
        from copy import copy
        o = copy(obs)
        o.frames = {"image": np.full((H, W, 3), agent_val, np.uint8),
                    "image2": np.full((H, W, 3), wrist_val, np.uint8)}
        return pol._build_batch(o)

    seen_keys = {}
    orig = pack._ordered_image_keys

    def spy(o):
        out = orig(o)
        seen_keys["available"] = sorted(k for k in o if k.startswith("observation.images"))
        seen_keys["ordered"] = list(out)
        return out
    pack._ordered_image_keys = spy
    b1 = run(0, 255)        # agent black, wrist white
    b2 = run(255, 0)        # swapped
    pack._ordered_image_keys = orig

    def find(batch, name):
        for k, v in batch.items():
            if k.endswith(name):
                return v
            if isinstance(v, dict):
                r = find(v, name)
                if r is not None:
                    return r
        return None

    pv1, pv2 = find(b1, "pixel_values"), find(b2, "pixel_values")
    thw = find(b1, "image_grid_thw")
    ids = find(b1, "input_ids")
    if pv1 is None or thw is None or ids is None:
        raise RuntimeError(f"batch keys: {sorted(b1)}")
    from transformers import AutoConfig
    image_token_id = None
    try:
        bcfg = AutoConfig.from_pretrained("nvidia/Cosmos-Reason2-2B")
        image_token_id = bcfg.image_token_id
    except Exception as e:
        print("config read failed:", e)
    n1 = int(thw[0].prod())
    blk1 = [float(pv1[:n1].float().mean()), float(pv1[n1:].float().mean())]
    blk2 = [float(pv2[:n1].float().mean()), float(pv2[n1:].float().mean())]
    first_is_agent_pv = blk1[0] < blk1[1] and blk2[0] > blk2[1]
    runs = []
    if image_token_id is not None:
        m = (ids[0] == image_token_id).int().tolist()
        i = 0
        while i < len(m):
            if m[i]:
                j = i
                while j < len(m) and m[j]:
                    j += 1
                runs.append([i, j - i])
                i = j
            else:
                i += 1
    first_key = seen_keys.get("ordered", [None])[0]
    agent_key_first = first_key == "observation.images.image"
    consistent = agent_key_first == first_is_agent_pv
    res = {"gate": "G0a camera order",
           "video_modality_keys": pack.video_modality_keys,
           "available_keys": seen_keys.get("available"), "ordered_keys": seen_keys.get("ordered"),
           "image_grid_thw": thw.tolist(), "pixel_values_shape": list(pv1.shape),
           "pixel_block_means_agent_black": blk1, "pixel_block_means_agent_white": blk2,
           "image_token_id": image_token_id, "image_token_runs_[start,len]": runs,
           "seq_len": int(ids.shape[-1]),
           "first_camera": "agentview" if first_is_agent_pv else "wrist",
           "consistent_across_readings": consistent,
           "PASS": bool(consistent and len(runs) == 2),
           "A2_S1_runs": bool(consistent and first_is_agent_pv)}
    dump("g0a", res)


# --- G0c ---------------------------------------------------------------------------
def _phys(env):
    sim = env._sim()
    return np.concatenate([sim.data.qpos.ravel(), sim.data.qvel.ravel()])


def _cmp(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return {"equal": False, "shape": [list(a.shape), list(b.shape)]}
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"equal": bool(np.array_equal(a, b)), "max_abs": float(d.max()), "n_diff": int((d > 0).sum())}


def g0c(a):
    from vla_harness.envs.libero_env import lplus_sensor_noise
    from vla_harness.schema import PerturbationSpec
    cat = catalogue()
    rx = re.compile(r"_view_(-?\d+)_(-?\d+)_(\d+)_(-?\d+)_(-?\d+)")
    cam = sorted(i for i, r in cat.items() if r["category"] == "Camera Viewpoints")
    rng = np.random.default_rng(60)
    r048 = [x["task_id"] for x in json.load(open(R048))["instances"]]
    endpt = [i for i in cam if rx.search(cat[i]["name"]).group(4, 5) != ("0", "0")]
    scaled = [i for i in cam if rx.search(cat[i]["name"]).group(3) != "100"]
    by_lvl = {L: [i for i in cam if cat[i]["difficulty_level"] == L] for L in range(1, 6)}
    pick = list(r048[:4]) + list(rng.choice(endpt, 3, replace=False)) + list(rng.choice(scaled, 3, replace=False))
    pick += [int(rng.choice(by_lvl[L])) for L in range(1, 6)]
    pick = list(dict.fromkeys(int(x) for x in pick))[: a.n_camera]
    rows = []
    controls = {}

    def control(ci):
        if ci not in controls:
            controls[ci] = make_env(ci)
        return controls[ci]

    for tid in pick:
        name = cat[tid]["name"]
        ci = control_index(cat, name)
        v = make_env(tid); c = control(ci)
        ov = v.reset(0, PerturbationSpec()); oc = c.reset(0, PerturbationSpec())
        row = {"row": "camera", "task_id": tid, "control": ci, "variant": name, "level": cat[tid]["difficulty_level"],
               "phys_equal": bool(np.array_equal(_phys(v), _phys(c))),
               "init_state_index": [v.init_state_index, c.init_state_index]}
        pf = c.perturbed_frames(name)
        row["P_agent_vs_variant_own"] = _cmp(pf["image"], ov.frames["image"])
        row["P_wrist_vs_variant_own"] = _cmp(pf["image2"], ov.frames["image2"])
        row["control_unchanged_after"] = _cmp(c.nominal_frames()["image"], oc.frames["image"])["equal"]
        nf = v.nominal_frames()
        row["N_agent_vs_control_own"] = _cmp(nf["image"], oc.frames["image"])
        row["PASS"] = bool(row["phys_equal"] and row["P_agent_vs_variant_own"]["equal"]
                           and row["P_wrist_vs_variant_own"]["equal"] and row["control_unchanged_after"]
                           and row["N_agent_vs_control_own"]["equal"])
        print(f"camera {tid} L{row['level']} {name.split('_view_')[1]}: phys={row['phys_equal']} "
              f"P {row['P_agent_vs_variant_own']} N {row['N_agent_vs_control_own']} -> {row['PASS']}", flush=True)
        rows.append(row)
        del v

    # noise: every family, incl. motion (1-10) and glass (41-50)
    noi = sorted(i for i, r in cat.items() if r["category"] == "Sensor Noise")
    nval = lambda i: int(re.search(r"_noise_(\d+)", cat[i]["name"]).group(1))
    fams = {"motion": (1, 10), "gaussian": (11, 20), "zoom": (21, 30), "fog": (31, 40), "glass": (41, 50)}
    npick = []
    for f, (lo, hi) in fams.items():
        c_ = [i for i in noi if lo <= nval(i) <= hi]
        npick += [int(x) for x in rng.choice(c_, min(a.n_noise_per_family, len(c_)), replace=False)]
    for tid in npick:
        name = cat[tid]["name"]; n = nval(tid)
        fam = next(f for f, (lo, hi) in fams.items() if lo <= n <= hi)
        ci = control_index(cat, name)
        v = make_env(tid); c = control(ci)
        ov = v.reset(0, PerturbationSpec()); oc = c.reset(0, PerturbationSpec())
        row = {"row": "noise", "task_id": tid, "control": ci, "variant": name, "noise": n, "family": fam,
               "level": cat[tid]["difficulty_level"], "phys_equal": bool(np.array_equal(_phys(v), _phys(c)))}
        seed = 1000 * 0 + 0
        ours = c.perturbed_frames(name, noise_seed=seed)
        # LIBERO-Plus's own dispatch: the variant wrapper's step() on the canonical env's clean frame
        wrapper = v._env.unwrapped._env
        clean = c._env.unwrapped._env.env._get_observations(force_update=True)
        real_step = wrapper.env.step
        st = np.random.get_state()
        try:
            wrapper.env.step = lambda _a, clean=clean: ({k: (np.array(x, copy=True) if isinstance(x, np.ndarray) else x)
                                                        for k, x in clean.items()}, 0.0, False, {})
            np.random.seed(seed)
            theirs_raw = wrapper.step(None)[0]
        finally:
            wrapper.env.step = real_step
            np.random.set_state(st)
        theirs = c._env.unwrapped._format_raw_obs(theirs_raw)["pixels"]
        row["P_agent_vs_lplus_dispatch"] = _cmp(ours["image"], theirs["image"])
        row["P_wrist_vs_clean"] = _cmp(ours["image2"], oc.frames["image2"])
        again = c.perturbed_frames(name, noise_seed=seed)
        row["seeded_repeatable"] = _cmp(again["image"], ours["image"])["equal"]
        other = c.perturbed_frames(name, noise_seed=seed + 1)
        row["seed_changes_frame"] = not _cmp(other["image"], ours["image"])["equal"]
        if fam in ("gaussian", "zoom"):            # no randomness: the variant env's own frame must match end to end
            row["P_agent_vs_variant_own"] = _cmp(ours["image"], ov.frames["image"])
        row["N_agent_vs_control_own"] = _cmp(v.nominal_frames()["image"], oc.frames["image"])
        row["PASS"] = bool(row["phys_equal"] and row["P_agent_vs_lplus_dispatch"]["equal"]
                           and row["P_wrist_vs_clean"]["equal"] and row["seeded_repeatable"]
                           and row.get("P_agent_vs_variant_own", {"equal": True})["equal"]
                           and row["N_agent_vs_control_own"]["equal"])
        print(f"noise {tid} {fam} {n}: phys={row['phys_equal']} P {row['P_agent_vs_lplus_dispatch']} "
              f"own {row.get('P_agent_vs_variant_own')} N {row['N_agent_vs_control_own']} "
              f"seed-varies={row['seed_changes_frame']} -> {row['PASS']}", flush=True)
        rows.append(row)
        del v
    cr = [r for r in rows if r["row"] == "camera"]; nr = [r for r in rows if r["row"] == "noise"]
    dump("g0c", {"gate": "G0c training-pair exactness",
                 "camera_pass": f"{sum(r['PASS'] for r in cr)}/{len(cr)}",
                 "noise_pass": f"{sum(r['PASS'] for r in nr)}/{len(nr)}",
                 "noise_families": sorted({r['family'] for r in nr}),
                 "PASS": all(r["PASS"] for r in rows), "rows": rows})


# --- G0c, amended (tol8px; amendment A1-A4 of runs/r060/gates/g0c_amendment_draft.md, user-approved 10-12) -----
TOL_PX = 8          # A3: at most 8 differing PIXELS per camera (any channel, any magnitude); test run at this tolerance
OUT_TOL = os.path.join(OUT, "tol8px")


def _cmp_px(a, b):
    """n_diff (channel elements, as _cmp), n_px (pixels differing in any channel), max_abs; ok = n_px <= TOL_PX."""
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return {"ok": False, "shape": [list(a.shape), list(b.shape)]}
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    n_px = int((d > 0).any(-1).sum()) if d.ndim == 3 else int((d > 0).sum())
    return {"ok": n_px <= TOL_PX, "equal": bool(np.array_equal(a, b)), "n_px": n_px, "n_diff": int((d > 0).sum()),
            "max_abs": float(d.max()) if d.size else 0.0}


def _both(fa, fb):
    return {"agent": _cmp_px(fa["image"], fb["image"]), "wrist": _cmp_px(fa["image2"], fb["image2"])}


def _ok(x):
    return all(v["ok"] for v in x.values()) if "agent" in x else x["ok"]


def _rerender(env, raw_hook=None):
    """A1: the env's frames re-rendered at its CURRENT state (sim.forward + force_update), the same path
    perturbed_frames / nominal_frames use; never a step/reset frame (robosuite samples those on a timer)."""
    outer = env._env.unwrapped
    inner = outer._env.env
    inner.sim.forward()
    raw = dict(inner._get_observations(force_update=True))
    if raw_hook is not None:
        raw = raw_hook(raw)
    return dict(outer._format_raw_obs(raw)["pixels"])


def g0c_selftest(_a):
    """CPU unit checks for the tol8px comparison (no env, no model)."""
    z = np.zeros((10, 10, 3), np.uint8)
    one = z.copy(); one[0, 0, :] = 255                       # 1 pixel, 3 channels, max 255
    eight = z.copy(); eight[:8, 0, 0] = 1                    # 8 pixels
    nine = z.copy(); nine[:9, 0, 1] = 1                      # 9 pixels
    res = {"identical": _cmp_px(z, z), "one_pixel_255": _cmp_px(z, one), "eight_px": _cmp_px(z, eight),
           "nine_px": _cmp_px(z, nine), "shape_mismatch": _cmp_px(z, z[:5])}
    ok = (res["identical"]["ok"] and res["identical"]["equal"] and res["one_pixel_255"]["ok"]
          and res["one_pixel_255"]["n_px"] == 1 and res["one_pixel_255"]["n_diff"] == 3 and res["one_pixel_255"]["max_abs"] == 255
          and res["eight_px"]["ok"] and res["eight_px"]["n_px"] == 8 and not res["nine_px"]["ok"] and res["nine_px"]["n_px"] == 9
          and not res["shape_mismatch"]["ok"])
    res["PASS"] = bool(ok)
    os.makedirs(OUT_TOL, exist_ok=True)
    json.dump({"tol8px": True, "what": "G0c tol8px comparison self-test (CPU)", **res}, open(os.path.join(OUT_TOL, "selftest.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    if not ok:
        sys.exit(1)


def g0c_tol8(a):
    """G0c with amendment A1-A4 (user-approved 10-12), tolerance A3 = TOL_PX pixels per camera.
    A1 every comparison is re-render vs re-render;  A2 a FRESH control env and a FRESH variant env per row, each
    reset once, nothing deleted between that reset and the row's renders (both are deleted together after it);
    A3 pass = n_px <= 8 per camera, n_diff / n_px / max_abs reported for every comparison;  A4 noise end to end:
    control perturbed_frames vs LIBERO-Plus's noise applied to the VARIANT env's own clean re-render, same seed.
    Row selection is identical to g0c (same rng and order). N side uses nominal_frames with the 10-12 pose fix."""
    import gc
    from vla_harness.envs.libero_env import lplus_sensor_noise
    from vla_harness.schema import PerturbationSpec
    global OUT
    out_main = OUT
    OUT = OUT_TOL
    try:
        os.makedirs(OUT_TOL, exist_ok=True)
        code_state()
    finally:
        OUT = out_main
    cat = catalogue()
    rx = re.compile(r"_view_(-?\d+)_(-?\d+)_(\d+)_(-?\d+)_(-?\d+)")
    cam = sorted(i for i, r in cat.items() if r["category"] == "Camera Viewpoints")
    rng = np.random.default_rng(60)
    r048 = [x["task_id"] for x in json.load(open(R048))["instances"]]
    endpt = [i for i in cam if rx.search(cat[i]["name"]).group(4, 5) != ("0", "0")]
    scaled = [i for i in cam if rx.search(cat[i]["name"]).group(3) != "100"]
    by_lvl = {L: [i for i in cam if cat[i]["difficulty_level"] == L] for L in range(1, 6)}
    pick = list(r048[:4]) + list(rng.choice(endpt, 3, replace=False)) + list(rng.choice(scaled, 3, replace=False))
    pick += [int(rng.choice(by_lvl[L])) for L in range(1, 6)]
    pick = list(dict.fromkeys(int(x) for x in pick))[: a.n_camera]
    rows = []

    def fresh_pair(tid, ci):
        gc.collect()
        v = make_env(tid); c = make_env(ci)
        ov = v.reset(0, PerturbationSpec()); oc = c.reset(0, PerturbationSpec())
        return v, c, ov, oc

    for tid in pick:
        name = cat[tid]["name"]
        ci = control_index(cat, name)
        v, c, ov, oc = fresh_pair(tid, ci)
        row = {"row": "camera", "task_id": tid, "control": ci, "variant": name, "level": cat[tid]["difficulty_level"],
               "phys_equal": bool(np.array_equal(_phys(v), _phys(c)))}
        c_before = _rerender(c)
        row["P_vs_variant_rerender"] = _both(c.perturbed_frames(name), _rerender(v))
        row["control_restored"] = _both(_rerender(c), c_before)
        row["N_vs_control_rerender"] = _both(v.nominal_frames(), _rerender(c))
        row["PASS"] = bool(row["phys_equal"] and _ok(row["P_vs_variant_rerender"]) and _ok(row["control_restored"])
                           and _ok(row["N_vs_control_rerender"]))
        print(f"camera {tid} L{row['level']} {name.split('_view_')[1]}: phys={row['phys_equal']} "
              f"P {[row['P_vs_variant_rerender'][k]['n_px'] for k in ('agent', 'wrist')]} "
              f"N {[row['N_vs_control_rerender'][k]['n_px'] for k in ('agent', 'wrist')]} -> {row['PASS']}", flush=True)
        rows.append(row)
        del v, c

    noi = sorted(i for i, r in cat.items() if r["category"] == "Sensor Noise")
    nval = lambda i: int(re.search(r"_noise_(\d+)", cat[i]["name"]).group(1))
    fams = {"motion": (1, 10), "gaussian": (11, 20), "zoom": (21, 30), "fog": (31, 40), "glass": (41, 50)}
    npick = []
    for f, (lo, hi) in fams.items():
        c_ = [i for i in noi if lo <= nval(i) <= hi]
        npick += [int(x) for x in rng.choice(c_, min(a.n_noise_per_family, len(c_)), replace=False)]
    for tid in npick:
        name = cat[tid]["name"]; n = nval(tid)
        fam = next(f for f, (lo, hi) in fams.items() if lo <= n <= hi)
        ci = control_index(cat, name)
        v, c, ov, oc = fresh_pair(tid, ci)
        row = {"row": "noise", "task_id": tid, "control": ci, "variant": name, "noise": n, "family": fam,
               "level": cat[tid]["difficulty_level"], "phys_equal": bool(np.array_equal(_phys(v), _phys(c)))}
        seed = 0
        ours = c.perturbed_frames(name, noise_seed=seed)
        # LIBERO-Plus's own dispatch on the control's clean re-render (unchanged check, now under the tolerance)
        wrapper = v._env.unwrapped._env
        c._env.unwrapped._env.env.sim.forward()
        clean = c._env.unwrapped._env.env._get_observations(force_update=True)
        real_step = wrapper.env.step
        st = np.random.get_state()
        try:
            wrapper.env.step = lambda _a, clean=clean: ({k: (np.array(x, copy=True) if isinstance(x, np.ndarray) else x)
                                                        for k, x in clean.items()}, 0.0, False, {})
            np.random.seed(seed)
            theirs_raw = wrapper.step(None)[0]
        finally:
            wrapper.env.step = real_step
            np.random.set_state(st)
        theirs = c._env.unwrapped._format_raw_obs(theirs_raw)["pixels"]
        row["P_agent_vs_lplus_dispatch"] = _cmp_px(ours["image"], theirs["image"])
        row["P_wrist_vs_control_rerender"] = _cmp_px(ours["image2"], _rerender(c)["image2"])
        row["seeded_repeatable"] = _cmp_px(c.perturbed_frames(name, noise_seed=seed)["image"], ours["image"])
        row["seed_changes_frame"] = not _cmp_px(c.perturbed_frames(name, noise_seed=seed + 1)["image"], ours["image"])["equal"]

        def noisy(raw, n=n):                         # A4: LIBERO-Plus noise on the VARIANT's own clean re-render
            raw["agentview_image"] = lplus_sensor_noise(raw["agentview_image"], n, seed)
            return raw
        row["P_agent_vs_noise_of_variant_rerender"] = _cmp_px(ours["image"], _rerender(v, noisy)["image"])
        row["N_vs_control_rerender"] = _both(v.nominal_frames(), _rerender(c))
        row["PASS"] = bool(row["phys_equal"] and row["P_agent_vs_lplus_dispatch"]["ok"] and row["P_wrist_vs_control_rerender"]["ok"]
                           and row["seeded_repeatable"]["ok"] and row["P_agent_vs_noise_of_variant_rerender"]["ok"]
                           and _ok(row["N_vs_control_rerender"]))
        print(f"noise {tid} {fam} {n}: phys={row['phys_equal']} dispatch {row['P_agent_vs_lplus_dispatch']['n_px']} "
              f"e2e {row['P_agent_vs_noise_of_variant_rerender']['n_px']} N {[row['N_vs_control_rerender'][k]['n_px'] for k in ('agent', 'wrist')]} "
              f"-> {row['PASS']}", flush=True)
        rows.append(row)
        del v, c
    cr = [r for r in rows if r["row"] == "camera"]; nr = [r for r in rows if r["row"] == "noise"]
    res = {"tol8px": True, "gate": "G0c training-pair exactness, AMENDED A1-A4 (test run at tolerance 8 px per camera)",
           "tolerance_px_per_camera": TOL_PX, "amendment": "runs/r060/gates/g0c_amendment_draft.md A1-A4; user 10-12",
           "pose_fix": "nominal_frames uses _bench_camera_pose(0, 0, 1.0) for LIBERO-Plus camera variants (user 10-12)",
           "camera_pass": f"{sum(r['PASS'] for r in cr)}/{len(cr)}", "noise_pass": f"{sum(r['PASS'] for r in nr)}/{len(nr)}",
           "noise_families": sorted({r['family'] for r in nr}), "PASS": all(r["PASS"] for r in rows), "rows": rows}
    p = os.path.join(OUT_TOL, "g0c.json")
    json.dump(res, open(p, "w"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in res if k != "rows"}, indent=1))
    print("wrote", p)


# --- G0b ---------------------------------------------------------------------------
def g0b(a):
    import torch
    from vla_harness.capture.splice import FTM, attach_transform
    from vla_harness.schema import PerturbationSpec
    torch.manual_seed(0)
    pol = build_policy("cpu")
    pol.reset()
    t0 = time.time()
    log = lambda *x: print(f"[{time.time()-t0:6.0f}s]", *x, flush=True)
    log("policy loaded on cpu")
    sel = json.load(open(R048))["instances"][: a.n]
    noise_key = lambda i: 1000 * 0 + i

    def forward(obs, transform, site):
        pol.reset()
        h = attach_transform(pol._policy, transform, site=site, noise_key=noise_key)
        try:
            pol(obs)
        finally:
            h.detach()
        if len(h.records) != 1:
            raise RuntimeError(f"expected one model forward, got {len(h.records)}")
        return h.records[0]["action"]

    live = FTM(2048)
    with torch.no_grad():
        g = torch.Generator().manual_seed(1)
        live.gamma.copy_(0.05 * torch.randn(2048, generator=g))
        live.beta.copy_(0.05 * torch.randn(2048, generator=g))
    # the module's own identity on a real S2 tensor (sign of zero reported, not gated)
    feats_probe = {}
    rows = []
    for inst in sel:
        env = make_env(inst["task_id"])
        obs = env.reset(inst["seed"], PerturbationSpec())
        base = forward(obs, None, "S2")
        base2 = forward(obs, None, "S2")
        z2 = forward(obs, FTM(2048), "S2")
        z1 = forward(obs, FTM(2048), "S1")
        l2 = forward(obs, live, "S2")
        l1 = forward(obs, live, "S1")
        if not feats_probe:
            f = pol.features_for(obs)
            from vla_harness.capture.splice import image_positions
            x = f["backbone_features"]
            y = FTM(2048)(x, f["image_mask"] & f["backbone_attention_mask"])
            feats_probe = {"S2_equal": bool(torch.equal(x, y)),
                           "S2_bits_equal": bool(torch.equal(x.view(torch.int16), y.view(torch.int16))),
                           "neg_zeros_in_image_positions": int(((x == 0) & torch.signbit(x)).sum()),
                           "image_positions": int((f["image_mask"] & f["backbone_attention_mask"]).sum()),
                           "seq_len": int(x.shape[1]), "dtype": str(x.dtype)}
        eq = lambda u, w: bool(torch.equal(u, w))
        row = {"task_id": inst["task_id"], "label": inst.get("label"),
               "det_base_eq_base": eq(base, base2), "S2_zero_eq_base": eq(z2, base), "S1_zero_eq_base": eq(z1, base),
               "S2_live_changes": not eq(l2, base), "S1_live_changes": not eq(l1, base),
               "S2_live_maxabs": float((l2 - base).abs().max()), "S1_live_maxabs": float((l1 - base).abs().max()),
               "dtype": str(base.dtype), "shape": list(base.shape)}
        row["PASS"] = all(row[k] for k in ("det_base_eq_base", "S2_zero_eq_base", "S1_zero_eq_base",
                                            "S2_live_changes", "S1_live_changes"))
        log(row)
        rows.append(row)
        del env
    dump("g0b", {"gate": "G0b zero-init conformance (CPU, bf16)", "device": "cpu",
                 "n": len(rows), "pass": f"{sum(r['PASS'] for r in rows)}/{len(rows)}",
                 "module_identity_on_S2": feats_probe, "PASS": all(r["PASS"] for r in rows), "rows": rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["g0a", "g0b", "g0c", "g0c_tol8", "g0c_selftest"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--n-camera", type=int, default=15)
    ap.add_argument("--n-noise-per-family", type=int, default=2)
    a = ap.parse_args()
    if a.gate not in ("g0c_tol8", "g0c_selftest"):      # the amended runs write their own code_state under tol8px/
        code_state()
    {"g0a": g0a, "g0b": g0b, "g0c": g0c, "g0c_tol8": g0c_tol8, "g0c_selftest": g0c_selftest}[a.gate](a)


if __name__ == "__main__":
    main()
