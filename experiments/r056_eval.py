"""R-056 evaluation: base vs retrained GR00T on the held-out start poses, validation for
checkpoint selection, the nominal control, W/A transfer at forward 0. Resumable.

  ENV="PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8"
  PY=.venvs/libero-plus/bin/python      # every phase but `score` is a GPU job: flock, announced
  flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py select   [--lora-r 16]   # train dir runs/r056_r16_train
  # R-058: --run-prefix r058 --lora-r 4 --base-manifest runs/r056_r16/manifest.jsonl (reuses R-056's base rollouts)
  flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py heldout
  flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py nominal
  flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py transfer
  flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py wise          # exploratory, alpha 0.5
  .venvs/groot/bin/python experiments/r056_eval.py score

Phases (runs/r056/manifest.jsonl, one row per rollout, keyed so a rerun skips it):
  select    the validation list (experiments/repro/r056_val.json) on every saved
            checkpoint of --train-dir; picks the best by success, then median
            closest approach, then the later checkpoint; writes selection.json.
            Loss is never used (R-056).
  heldout   the 108 held-out starts (r056_heldout.json) on base and on the selected checkpoint.
  nominal   the R-029 control (canonical variant, PerturbationSpec()) on the NINE scored
            scenes (on-ramekin 1169 dropped, ruling 2026-09-28) x env seeds 0-2, noise
            seed = env seed: 27 per checkpoint; on-ramekin runs separately as set
            "nominal_ramekin" (--with-ramekin) and is reported apart.
  transfer  experiments/r044_reverse.py with the selected adapter: W/A transfer at
            forward 0 on the ten R-044 instances (no rollouts).
  wise      held-out + nominal with the selected adapter at alpha = --wise-alpha.
  loadcheck load base and each --adapters dir in turn, freeing each before the next; logs RSS and
            GPU memory after every load and free. No rollouts, writes nothing (memory check, F3).
  score     success, closest approach, per-radius success, gap closed =
            (retrained - base) / (nominal - base) on the held-out set; the six
            pre-registered expectations and the uninterpretability checks.

Every rollout runs P only, under the harness's fixed noise (noise_key = 1000*noise_seed + forward).

An adapter is refused unless its vla_train_meta.json has schedule_verified: true (r056_train's LR
schedule guard; R-056's first run cycled its LR). Policies load through the harness's streaming bf16
path, one at a time: the previous one is freed (del, gc.collect, empty_cache) before the next loads,
and process RSS and GPU memory are logged after each load.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.getcwd())

import numpy as np

OUT = "runs/r056_r16"
HELD = "experiments/repro/r056_heldout.json"
VAL = "experiments/repro/r056_val.json"
R029_SCENES = [984, 1030, 1062, 1090, 1132, 1169, 1201, 1247, 1282, 1327]
ON_RAMEKIN = 1169
NOMINAL_SCENES = [s for s in R029_SCENES if s != ON_RAMEKIN]
NOMINAL_MIN = 26                     # base below 26/27 makes R-056 uninterpretable (ruling 2026-09-28)
BANDS = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5 + 1e-9)]


def manifest_path():
    return os.path.join(OUT, "manifest.jsonl")


BASE_MANIFEST = None                 # another run's manifest whose "base" rows stand in for this run's


def _read(p):
    return [json.loads(l) for l in open(p) if l.strip()] if p and os.path.exists(p) else []


def rows():
    own = _read(manifest_path())
    if BASE_MANIFEST:
        own = [r for r in own if r["ckpt"] != "base"] + [r for r in _read(BASE_MANIFEST) if r["ckpt"] == "base"]
    return own


def key(r):
    return (r["ckpt"], r["set"], r["scene"], r.get("dir_seed"), r.get("radius"), r["noise_seed"], r["env_seed"])


def checkpoints(train_dir) -> list[tuple[str, str]]:
    """(label, adapter dir) for every saved checkpoint, in OPTIMIZER-step order. The label uses
    vla_train_meta.json's opt_step, never the directory name (lerobot_train names by micro-step)."""
    out = []
    for d in glob.glob(os.path.join(train_dir, "checkpoints", "[0-9]*", "pretrained_model")):
        if not os.path.exists(os.path.join(d, "adapter_config.json")):
            continue
        mp = os.path.join(d, "vla_train_meta.json")
        if not os.path.exists(mp):
            sys.exit(f"{d}: no vla_train_meta.json, cannot tell its optimizer step")
        require_verified(d)
        out.append((int(json.load(open(mp))["opt_step"]), d))
    return [(f"opt{st}", d) for st, d in sorted(out)]


def require_verified(adapter):
    """Refuse an adapter whose training did not pass r056_train's LR-schedule guard."""
    mp = os.path.join(adapter, "vla_train_meta.json")
    m = json.load(open(mp)) if os.path.exists(mp) else {}
    if m.get("schedule_verified") is not True:
        sys.exit(f"REFUSED: {adapter} has no schedule_verified in {mp} (trained before the LR-schedule fix, "
                 f"or its schedule check failed); not evaluating it")


def mem_report(tag):
    """Process RSS (current, peak) and GPU memory, one log line."""
    import torch
    st = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/self/status")
          if l.startswith(("VmRSS", "VmHWM", "RssAnon", "RssFile"))}
    try:
        smi = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.split()[0])
    except Exception:
        smi = -1
    out = {"tag": tag, "rss_gib": st["VmRSS"] / 2**20, "rss_peak_gib": st["VmHWM"] / 2**20,
           "rss_anon_gib": st["RssAnon"] / 2**20, "rss_file_gib": st["RssFile"] / 2**20,
           "cuda_alloc_gib": torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else None,
           "cuda_reserved_gib": torch.cuda.memory_reserved() / 2**30 if torch.cuda.is_available() else None,
           "nvidia_smi_used_gib": smi / 1024 if smi >= 0 else None}
    print(f"[mem] {tag}: RSS {out['rss_gib']:.2f} GiB (anon {out['rss_anon_gib']:.2f}, file {out['rss_file_gib']:.2f}; "
          f"peak {out['rss_peak_gib']:.2f}), cuda alloc "
          f"{out['cuda_alloc_gib']:.2f} / reserved {out['cuda_reserved_gib']:.2f} GiB, nvidia-smi "
          f"{out['nvidia_smi_used_gib']} GiB", flush=True)
    return out


def load_policy(label, adapter=None, alpha=1.0):
    """Build and load (reset() loads) one policy, then log memory."""
    if adapter:
        require_verified(adapter)
    pol = build_policy(adapter, alpha)
    pol.reset()
    mem_report(f"loaded {label} (alpha {alpha})")
    return pol


def free_policy(pol, label):
    """Drop every reference this module holds, collect, and return the CUDA cache before the next load."""
    import gc
    import torch
    pol._policy = pol._peft = pol._pre = pol._post = pol._env_pre = pol._env_post = None
    del pol
    gc.collect()
    torch.cuda.empty_cache()
    try:                                          # hand freed heap back to the OS (glibc keeps it otherwise)
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    mem_report(f"freed {label}")


def build_policy(adapter=None, alpha=1.0):
    from lerobot.envs.configs import LiberoPlusEnv
    from vla_harness.policies.lerobot_policy import LeRobotPolicy
    return LeRobotPolicy("nvidia/gr00t17-lerobot-libero_spatial-640", n_action_steps=16,
                         env_cfg=LiberoPlusEnv(task="libero_spatial"),
                         policy_overrides={"base_model_path": "nvidia/GR00T-N1.7-3B", "embodiment_tag": "libero_sim"},
                         dtype="bfloat16", rename_map={"observation.images.image2": "observation.images.wrist_image"},
                         adapter=adapter, adapter_alpha=alpha)


def run_set(label, adapter, alpha, set_name, starts):
    """Roll out `starts` ([{scene_task_id, dir_seed, radius, noise_seed, env_seed}]) on one checkpoint."""
    import torch
    from experiments.r054_mint import make_env
    from vla_harness.capture.splice import attach_splice
    from vla_harness.data.sources.rescue import closest_to_target, noise_key_for
    from vla_harness.runner import rollout
    from vla_harness.schema import PerturbationSpec, TraceStore
    os.makedirs(OUT, exist_ok=True)
    done = {key(r) for r in rows()}
    todo = [s for s in starts if (label, set_name, s["scene_task_id"], s.get("dir_seed"), s.get("radius"),
                                  s["noise_seed"], s["env_seed"]) not in done]
    print(f"[{label} {set_name}] {len(starts) - len(todo)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    if label == "base" and BASE_MANIFEST:
        sys.exit(f"base rows for {set_name} are missing from {BASE_MANIFEST}; run them there first")
    pol = load_policy(label, adapter, alpha)
    store = TraceStore(os.path.dirname(OUT) or ".", os.path.basename(OUT))
    t0 = time.time()
    for s in todo:
        env = make_env(s["scene_task_id"])
        spec = (PerturbationSpec.of(joint_radius_rad=float(s["radius"]), joint_dir_seed=int(s["dir_seed"]))
                if s.get("radius") else PerturbationSpec())
        g = attach_splice(pol._policy, source=lambda i: None, drive=None, noise_key=noise_key_for(s["noise_seed"]))
        try:
            r = rollout(env, pol, s["env_seed"], spec, video_dir=None)
        finally:
            g.detach()
        store.append(r)
        row = {"ckpt": label, "adapter": adapter, "alpha": alpha, "policy_id": pol.policy_id, "set": set_name,
               "scene": s["scene_task_id"], "dir_seed": s.get("dir_seed"), "radius": s.get("radius"),
               "noise_seed": s["noise_seed"], "env_seed": s["env_seed"], "rollout_id": r.rollout_id,
               "success": bool(r.success), "steps": r.env_steps, "closest_approach_m": closest_to_target(r, env),
               "wall_s": round(r.wall_time_s, 1)}
        with open(manifest_path(), "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  [{time.time()-t0:6.0f}s] {label} {set_name} scene {row['scene']} r={row['radius']} dir {row['dir_seed']} "
              f"noise {row['noise_seed']}: success={row['success']} ca={row['closest_approach_m']}", flush=True)
        del env
    g = None                                      # the splice handle holds the action head
    free_policy(pol, label)


def load_list(p):
    return [{**s, "env_seed": s.get("env_seed", 0)} for s in json.load(open(p))["instances"]]


def nominal_starts(a, scenes=None):
    scenes = scenes or [int(x) for x in a.nominal_scenes.split(",")]
    return [{"scene_task_id": sc, "dir_seed": None, "radius": None, "noise_seed": sd, "env_seed": sd}
            for sc in scenes for sd in (int(x) for x in a.nominal_seeds.split(","))]


def selected():
    p = os.path.join(OUT, "selection.json")
    if not os.path.exists(p):
        sys.exit("no selection.json: run the `select` phase first")
    return json.load(open(p))


# --- scoring ----------------------------------------------------------------------
def wilson(k, n, z=1.96):
    from vla_harness.runner import wilson_ci
    return list(wilson_ci(k, n, z))


def r047_expected(held_rows):
    """R-047's logistic curve per scene at each held-out radius (runs/r047/score.json)."""
    res = json.load(open("runs/r047/score.json"))["joint_radius_rad"]["res"]
    ps = []
    for r in held_rows:
        v = res[str(r["scene"])]
        x50, w = v.get("x50"), v.get("width")
        if x50 is None or w is None or not math.isfinite(x50) or not math.isfinite(w):
            ps.append(float(np.mean(v["p"][-1:]))); continue
        b = -2 * math.log(4) * 0.5 / w          # r047_score.py: width = 2 ln4 / (-b) * mx, mx = 0.5
        a_ = -b * x50 / 0.5
        ps.append(1 / (1 + math.exp(-(a_ + b * r["radius"] / 0.5))))
    return float(np.mean(ps)) if ps else None


def band_of(radius):
    for lo, hi in BANDS:
        if lo <= radius < hi:
            return f"{lo:.1f}-{min(hi, 0.5):.1f}"
    return "other"


def summarize(rs):
    k, n = sum(r["success"] for r in rs), len(rs)
    ca = [r["closest_approach_m"] for r in rs if r["closest_approach_m"] is not None]
    return {"success": k, "n": n, "rate": (k / n) if n else None, "wilson95": wilson(k, n),
            "median_ca": float(np.median(ca)) if ca else None}


# --- exploratory (user, 2026-09-29): gripper offset at t=0 and achieved joint radius -----------------
OFFSET_M = 0.08
NOMINAL_STARTS = "runs/r047/nominal_starts.json"      # derived once from R-047's unperturbed traces


def nominal_starts_by_scene() -> dict:
    """{scene: {eef_pos, joint_pos}} at t=0 of the UNPERTURBED start (R-047's '{}'-perturbation traces)."""
    if os.path.exists(NOMINAL_STARTS):
        return {int(k): v for k, v in json.load(open(NOMINAL_STARTS)).items()}
    out = {}
    for line in open("runs/r047/rollouts.jsonl"):
        if '"perturbation":{}' not in line:
            continue
        r = json.loads(line)
        sc = int(r["task_id"].rsplit("task", 1)[1])
        if sc not in out and r["steps"]:
            st = r["steps"][0]["obs_state"]
            out[sc] = {"eef_pos": st["eef_pos"], "joint_pos": st["joint_pos"], "rollout_id": r["rollout_id"]}
    json.dump(out, open(NOMINAL_STARTS, "w"), indent=1)
    return out


def start_states(rids: set) -> dict:
    """{rollout_id: steps[0].obs_state (eef_pos, joint_pos)} from this run's and the base run's trace stores."""
    stores = [os.path.join(OUT, "rollouts.jsonl")]
    if BASE_MANIFEST:
        stores.append(os.path.join(os.path.dirname(BASE_MANIFEST), "rollouts.jsonl"))
    out = {}
    for p in stores:
        if not os.path.exists(p):
            continue
        for line in open(p):
            rid = line[15:27] if line.startswith('{"rollout_id":"') else None
            if rid in rids and rid not in out:
                st = json.loads(line)["steps"][0]["obs_state"]
                out[rid] = {"eef_pos": st["eef_pos"], "joint_pos": st["joint_pos"]}
    return out


def gripper_offset_breakdown(hb, hr) -> dict:
    """EXPLORATORY, not pre-registered: held-out success split by the gripper's offset from the scene's
    unperturbed start at t=0 (<= 8 cm vs > 8 cm), and the achieved joint-space radius next to the requested
    one. Changes no registered expectation and not the gap-closed computation."""
    nom = nominal_starts_by_scene()
    ss = start_states({r["rollout_id"] for r in hb + hr if r.get("rollout_id")})
    def enrich(rs):
        out = []
        for r in rs:
            s0, n0 = ss.get(r.get("rollout_id")), nom.get(r["scene"])
            if not s0 or not n0:
                continue
            off = float(np.linalg.norm(np.subtract(s0["eef_pos"], n0["eef_pos"])))
            rad = float(np.linalg.norm(np.subtract(s0["joint_pos"], n0["joint_pos"])))
            out.append({**r, "offset_m": off, "achieved_radius": rad})
        return out
    eb, er = enrich(hb), enrich(hr)
    res = {"label": "EXPLORATORY (post-hoc, not pre-registered); no registered expectation depends on it",
           "offset_threshold_m": OFFSET_M, "nominal_start_source": NOMINAL_STARTS,
           "n_with_start_state": {"base": len(eb), "retrained": len(er)}}
    for name, cond in (("offset_le_8cm", lambda x: x["offset_m"] <= OFFSET_M), ("offset_gt_8cm", lambda x: x["offset_m"] > OFFSET_M)):
        res[name] = {"base": summarize([x for x in eb if cond(x)]), "retrained": summarize([x for x in er if cond(x)])}
    rad = {}
    for x in eb + er:
        rad.setdefault(x["radius"], []).append(x["achieved_radius"])
    res["achieved_radius_by_requested"] = {str(k): {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)),
                                                    "n": len(v)} for k, v in sorted(rad.items())}
    return res


def score(a):
    R = rows()
    sel = selected()
    ret = sel["label"]
    by = lambda ck, st, alpha=None: [r for r in R if r["ckpt"] == ck and r["set"] == st and (alpha is None or r["alpha"] == alpha)]
    out = {"selected": sel}
    hb, hr = by("base", "heldout"), by(ret, "heldout", 1.0)
    nb, nr = by("base", "nominal"), by(ret, "nominal", 1.0)
    out["heldout"] = {"base": summarize(hb), "retrained": summarize(hr)}
    out["nominal"] = {"base": summarize(nb), "retrained": summarize(nr)}
    # per-radius (per held-out radius and per minting band)
    per = {}
    for ck, rs in (("base", hb), ("retrained", hr)):
        for r in rs:
            per.setdefault(band_of(r["radius"]), {}).setdefault(ck, []).append(r["success"])
    out["per_band"] = {b: {ck: {"success": int(sum(v)), "n": len(v), "rate": float(np.mean(v))} for ck, v in d.items()}
                       for b, d in sorted(per.items())}
    kb = lambda r: (r["scene"], r["dir_seed"], r["radius"], r["noise_seed"])
    pb = {kb(r): r for r in hb}
    pairs = [(pb[kb(r)], r) for r in hr if kb(r) in pb]
    ca_pairs = [(x, y) for x, y in pairs if x["closest_approach_m"] is not None and y["closest_approach_m"] is not None]
    ca_better = sum(y["closest_approach_m"] < x["closest_approach_m"] for x, y in ca_pairs)
    from vla_harness.runner import mcnemar
    out["paired"] = {"n": len(pairs), "mcnemar": mcnemar([(bool(x["success"]), bool(y["success"])) for x, y in pairs]),
                     "ca_improved": ca_better, "ca_pairs": len(ca_pairs)}
    rb, rr, nom = out["heldout"]["base"]["rate"], out["heldout"]["retrained"]["rate"], out["nominal"]["base"]["rate"]
    gap = ((rr - rb) / (nom - rb)) if None not in (rb, rr, nom) and nom != rb else None
    out["gap_closed"] = gap
    tr = os.path.join(OUT, "transfer", "summary.json")
    base_w = 0.88                                             # R-042 median W transfer at forward 0
    tw = json.load(open(tr))["medians"]["denoise_W"] if os.path.exists(tr) else None
    out["transfer"] = {"retrained_median_W": tw, "base_median_W": base_w,
                       "retrained": json.load(open(tr)) if os.path.exists(tr) else None}
    bands = [b for b in out["per_band"] if "base" in out["per_band"][b] and "retrained" in out["per_band"][b]]
    gain = {b: out["per_band"][b]["retrained"]["rate"] - out["per_band"][b]["base"]["rate"] for b in bands}
    ok = lambda *x: all(v is not None for v in x)
    out["expectations"] = {
        "E1_heldout_gain_ge_15pp": (rr - rb >= 0.15) if ok(rr, rb) else None,
        "E2_gap_closed_ge_40pct": (gap >= 0.40) if gap is not None else None,
        "E3_nominal_regress_lt_5pp": (out["nominal"]["base"]["rate"] - out["nominal"]["retrained"]["rate"] < 0.05)
        if ok(out["nominal"]["base"]["rate"], out["nominal"]["retrained"]["rate"]) else None,
        "E4_W_transfer_falls_ge_0.2": (base_w - tw >= 0.2) if tw is not None else None,
        "E5_low_band_gain_gt_high_band": (gain[min(bands)] > gain[max(bands)]) if len(bands) >= 2 else None,
        "E6_ca_improves_ge_60pct": (ca_better >= 0.6 * len(ca_pairs)) if ca_pairs else None,
    }
    out["failure_criterion_gap_lt_one_third"] = (gap < 1 / 3) if gap is not None else None
    exp = r047_expected(hb)
    lo, hi = out["heldout"]["base"]["wilson95"]
    gates = {}
    for p in ("runs/r055/gates.json",):
        if os.path.exists(p):
            g = json.load(open(p))
            gates = {k: v.get("pass") for k, v in g.items() if isinstance(v, dict)}
    from vla_harness.data.contract import HeldOut
    g4 = []
    for pr in glob.glob("data/*/meta/provenance.jsonl"):
        g4 += HeldOut([HELD, VAL]).violations([{"episode_id": x["episode_id"], "scene_task_id": x["provenance"]["scene_task_id"],
                                                "dir_seed": x["provenance"]["dir_seed"]} for x in map(json.loads, open(pr))])
    out["uninterpretable_checks"] = {
        "base_heldout_vs_r047_curve": {"r047_expected": exp, "base_wilson95": [lo, hi],
                                       "consistent": (lo <= exp <= hi) if exp is not None else None},
        "base_nominal_ge_26_of_27": (out["nominal"]["base"]["success"] >= NOMINAL_MIN) if out["nominal"]["base"]["n"] == 27 else None,
        "g4_violations": g4, "r055_gates": gates,
    }
    ram = {"base": summarize(by("base", "nominal_ramekin")), "retrained": summarize(by(ret, "nominal_ramekin", 1.0))}
    if ram["base"]["n"] or ram["retrained"]["n"]:
        out["nominal_on_ramekin"] = {**ram, "note": "reported apart; not in the nominal readout or its threshold"}
    out["run"] = {"out": OUT, "base_manifest": BASE_MANIFEST}
    try:
        out["exploratory_gripper_offset"] = gripper_offset_breakdown(hb, hr)
    except Exception as e:                         # exploratory: never blocks the registered readout
        out["exploratory_gripper_offset"] = {"error": f"{type(e).__name__}: {e}"}
    if a.wise_alpha is not None:
        out["wise"] = {"alpha": a.wise_alpha, "heldout": summarize(by(ret, "heldout_wise", a.wise_alpha)),
                       "nominal": summarize(by(ret, "nominal_wise", a.wise_alpha)), "exploratory": True}
    json.dump(out, open(os.path.join(OUT, "score.json"), "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("heldout", "nominal", "gap_closed", "expectations", "uninterpretable_checks")},
                     indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["select", "heldout", "nominal", "transfer", "wise", "score", "loadcheck"])
    ap.add_argument("--adapters", default="", help="loadcheck: comma-separated adapter dirs (not checked for "
                                                    "schedule_verified: nothing is evaluated)")
    ap.add_argument("--run-prefix", default="r056")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--train-dir", default=None, help="default runs/<prefix>_r<r>_train")
    ap.add_argument("--base-manifest", default=None, help="reuse another run's base rollouts (R-058)")
    ap.add_argument("--with-ramekin", action="store_true", help="also run on-ramekin's nominal, reported apart")
    ap.add_argument("--nominal-scenes", default=",".join(map(str, NOMINAL_SCENES)))
    ap.add_argument("--nominal-seeds", default="0,1,2")
    ap.add_argument("--wise-alpha", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="default runs/<prefix>_r<r>")
    a = ap.parse_args()
    globals()["OUT"] = a.out or f"runs/{a.run_prefix}_r{a.lora_r}"
    globals()["BASE_MANIFEST"] = a.base_manifest
    a.train_dir = a.train_dir or f"runs/{a.run_prefix}_r{a.lora_r}_train"
    if a.phase != "loadcheck":
        os.makedirs(OUT, exist_ok=True)
    if a.phase == "select":
        val = load_list(VAL)
        cks = checkpoints(a.train_dir)
        if not cks:
            sys.exit(f"no adapter checkpoints under {a.train_dir}/checkpoints")
        for label, d in cks:
            run_set(label, d, 1.0, "val", val)
        R = rows()
        scored = []
        for label, d in cks:
            rs = [r for r in R if r["ckpt"] == label and r["set"] == "val"]
            ca = [r["closest_approach_m"] for r in rs if r["closest_approach_m"] is not None]
            scored.append({"label": label, "adapter": d, "success": sum(r["success"] for r in rs), "n": len(rs),
                           "median_ca": float(np.median(ca)) if ca else float("inf"), "step": int(label[3:])})
        best = sorted(scored, key=lambda s: (-s["success"], s["median_ca"], -s["step"]))[0]
        json.dump({**best, "all": scored, "rule": "max val success, then min median closest approach, then later step"},
                  open(os.path.join(OUT, "selection.json"), "w"), indent=1)
        print("selected", best)
    elif a.phase == "heldout":
        s = selected(); held = load_list(HELD)
        run_set("base", None, 1.0, "heldout", held)
        run_set(s["label"], s["adapter"], 1.0, "heldout", held)
    elif a.phase == "nominal":
        s = selected(); nom = nominal_starts(a)
        run_set("base", None, 1.0, "nominal", nom)
        run_set(s["label"], s["adapter"], 1.0, "nominal", nom)
        if a.with_ramekin:
            ram = nominal_starts(a, [ON_RAMEKIN])
            run_set("base", None, 1.0, "nominal_ramekin", ram)
            run_set(s["label"], s["adapter"], 1.0, "nominal_ramekin", ram)
    elif a.phase == "loadcheck":
        mem_report("start")
        free_policy(load_policy("base"), "base")
        for d in filter(None, a.adapters.split(",")):
            pol = build_policy(d, 1.0); pol.reset(); mem_report(f"loaded {d}")
            free_policy(pol, d)
        return
    elif a.phase == "transfer":
        s = selected()
        require_verified(s["adapter"])
        d = os.path.join(OUT, "transfer")
        if os.path.exists(os.path.join(d, "rows.jsonl")):
            os.remove(os.path.join(d, "rows.jsonl"))              # r044_reverse appends
        rc = subprocess.run([sys.executable, "experiments/r044_reverse.py", "--selection",
                             "experiments/repro/r044_selection_ris.json", "--out", d, "--adapter", s["adapter"]]).returncode
        sys.exit(rc)
    elif a.phase == "wise":
        s = selected()
        run_set(s["label"], s["adapter"], a.wise_alpha, "heldout_wise", load_list(HELD))
        run_set(s["label"], s["adapter"], a.wise_alpha, "nominal_wise", nominal_starts(a))
    else:
        score(a)


if __name__ == "__main__":
    main()
