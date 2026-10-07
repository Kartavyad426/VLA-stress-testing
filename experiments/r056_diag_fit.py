"""D20(i) follow-up (prim 2026-10-06, within the user's D20 approval): first-chunk fit on the R-054 TRAINING starts.
No rollouts: reset to each start, one chunk-0 forward per model, compare with the minted label.

  .venvs/groot/bin/python experiments/r056_diag_fit.py anchor          # CPU: reproduce the post-mortem's 4.51 / 3.98 / 4.12
  PYTHONPATH=$PWD/third_party/LIBERO-plus MUJOCO_GL=egl flock /tmp/vla_gpu.lock \
      .venvs/libero-plus/bin/python experiments/r056_diag_fit.py run [--limit N]
  .venvs/groot/bin/python experiments/r056_diag_fit.py score

Metric (experiments/analysis_r056_postmortem/an7.py): disp(a) = a[:16, :3].sum(0), the commanded translation over
chunk 0 in env action units (the rollouts' stored actions); distance = ||disp(x) - disp(ref)||; medians over starts.

Training starts: the 100 episodes of data/r054_train48. Input (SFT, 10-06) = each episode's RECORDED frame-0
observation (the env's lossless PNGs at 360 px and the recorded proprio), not a fresh reset; noise = the episode's
minting noise seed, identical for the three models. Label = the minted record's actions at frames 0-15 (env units).
The label was driven by the base head on the scene's NOMINAL features under that noise seed, so for noise seed 0 its
disp should equal the an7 nominal reference (base, seed 0, R-056 nominal set); `score` reports the discrepancy and
gives distances to BOTH references.

Models: base, r16 opt1000 (R-056 selected), r4 opt500 (R-058 selected). One GR00T load (the harness's streaming
bf16 path); adapters are injected in place as named PEFT adapters and switched; base = adapter layers disabled.
Checked in the run: base chunk before attaching adapters == base chunk after (bit-identical), each adapter's chunk
differs from base, chunk 0 is exactly one model forward. Info only: recorded vs fresh-reset observation at start 0.

The held-out anchor (`anchor`): a0 = the first 16 EXECUTED actions of each rollout trace (manifest and trace joined by
position within a run, as they must be, since rollout_id ignores the noise seed); arms paired with base by start key
(scene, dir, radius, noise seed, env seed); reference = base's executed chunk 0 of the scene's nominal seed-0 rollout.
Writes runs/r056_diag/fit_firstchunk_rows.jsonl and runs/r056_diag/fit_firstchunk.json (split by joint direction x
achieved t=0 offset band in cm, as D20(ii)).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.getcwd())
import numpy as np

OUT = "runs/r056_diag"
ROWS = os.path.join(OUT, "fit_firstchunk_rows.jsonl")
MODELS = [("r16_opt1000", "runs/r056_r16_train", 1000), ("r4_opt500", "runs/r058_r4_train", 500)]
BANDS_CM = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 100)]


def disp(a):
    return np.asarray(a, float)[:16, :3].sum(0)


def load_run(run):
    """(manifest row, rollout) pairs joined BY POSITION (rollout_id ignores the noise seed)."""
    man = [json.loads(l) for l in open(f"{run}/manifest.jsonl")]
    with open(f"{run}/rollouts.jsonl") as fh:
        for m, line in zip(man, fh):
            r = json.loads(line)
            if r["rollout_id"] != m["rollout_id"] or r["success"] != m["success"]:
                sys.exit(f"{run}: manifest and rollouts not aligned by position")
            yield m, r


def actions(r):
    return np.array([s["action"] for s in r["steps"] if s.get("action") is not None])


def nominal_disp() -> dict:
    """{scene: disp of base's seed-0 chunk 0 at the nominal start} from R-056's nominal set (an7's 'nominal')."""
    out = {}
    for m, r in load_run("runs/r056_r16"):
        if m["set"] == "nominal" and m["ckpt"] == "base" and m["noise_seed"] == 0:
            out[int(m["scene"])] = disp(actions(r)).tolist()
    return out


def anchor():
    """Recompute the post-mortem's held-out seed-0 first-chunk numbers from the stored rollouts."""
    nom = nominal_disp()
    rows = {}
    for run, arm in (("runs/r056_r16", "r16"), ("runs/r058_r4", "r4")):
        for m, r in load_run(run):
            if m["set"] != "heldout":
                continue
            k = (m["scene"], m["dir_seed"], m["radius"], m["noise_seed"], m["env_seed"])
            rows.setdefault(k, {})["base" if m["ckpt"] == "base" else arm] = disp(actions(r))
    out = {}
    for arm in ("r16", "r4"):
        L = []
        for k, v in rows.items():
            if k[3] != 0 or "base" not in v or arm not in v:
                continue
            n0 = np.array(nom[int(k[0])])
            L.append((np.linalg.norm(v["base"] - n0), np.linalg.norm(v[arm] - n0), np.linalg.norm(v[arm] - v["base"])))
        A = np.array(L)
        out[arm] = {"n": len(A), "median_base_to_nominal": float(np.median(A[:, 0])),
                    "median_arm_to_nominal": float(np.median(A[:, 1])), "median_arm_to_base": float(np.median(A[:, 2])),
                    "arm_closer": int((A[:, 1] < A[:, 0]).sum())}
    print(json.dumps(out, indent=1))
    ok = (out["r16"]["n"] == 36 and round(out["r16"]["median_base_to_nominal"], 2) == 4.51
          and round(out["r16"]["median_arm_to_nominal"], 2) == 3.98 and round(out["r4"]["median_arm_to_nominal"], 2) == 4.12)
    print("ANCHOR", "REPRODUCED" if ok else "NOT REPRODUCED")
    return out, ok


def starts(limit=None):
    out = []
    for l in open("data/r054_train48/meta/provenance.jsonl"):
        p = json.loads(l)
        pv = p["provenance"]
        out.append({"episode_id": p["episode_id"], "scene": int(pv["scene_task_id"]), "radius": float(pv["magnitude"]),
                    "dir_seed": int(pv["dir_seed"]), "env_seed": int(pv["env_seed"]), "noise_seed": int(pv["noise_seed"]),
                    "band": pv.get("band"), "records_root": p["records_root"]})
    return out[:limit] if limit else out


def recorded_obs(s, rec):
    """The episode's RECORDED frame-0 observation (lossless PNGs the env wrote from the arrays the policy saw, and the
    recorded proprio), as the harness Observation the policy consumes."""
    from PIL import Image
    from vla_harness.schema import Observation
    f0 = rec["frames"][0]
    frames = {cam: np.asarray(Image.open(os.path.join(s["records_root"], rel)).convert("RGB")) for cam, rel in f0["images"].items()}
    return Observation(instruction=rec["instruction"], state=dict(f0["proprio"]), frames=frames, t=0)


def run(a):
    import torch
    from peft import PeftModel
    from experiments.r054_mint import make_env
    from experiments.r056_eval import build_policy, mem_report
    from experiments.r056_diag_loss import adapter_dir
    from vla_harness.capture.splice import attach_splice
    from vla_harness.data.sources.rescue import noise_key_for
    from vla_harness.schema import PerturbationSpec
    os.makedirs(OUT, exist_ok=True)
    rows_p = ROWS if not a.limit else os.path.join(OUT, f"fit_firstchunk_rows_smoke{a.limit}.jsonl")
    if os.path.exists(rows_p):
        os.remove(rows_p)
    t0 = time.time()
    adapters = [(n, adapter_dir(td, st)) for n, td, st in MODELS]
    pol = build_policy()                       # base, streaming bf16, n_action_steps 16
    pol.reset()
    mem_report("base loaded")
    S = starts(a.limit)
    nom = json.load(open("runs/r047/nominal_starts.json"))
    recs = {s["episode_id"]: json.load(open(os.path.join(s["records_root"], "episodes", s["episode_id"] + ".json"))) for s in S}

    def chunk0(obs, seed):
        pol.reset()
        f0 = pol.model_forwards
        g = attach_splice(pol._policy, source=lambda i: None, drive=None, noise_key=noise_key_for(seed))
        try:
            acts = [pol(obs.policy_view()).values for _ in range(16)]
        finally:
            g.detach()
        if pol.model_forwards - f0 != 1:
            raise RuntimeError(f"chunk 0 took {pol.model_forwards - f0} model forwards, expected 1")
        return np.array(acts)

    checks = {"adapters": {n: d for n, d in adapters}, "input": "recorded frame-0 observation, minting noise seed"}
    # info only: the recorded observation vs a fresh env reset of start 0 (render nondeterminism exists at some scenes)
    s0 = S[0]
    env = make_env(s0["scene"])
    fresh = env.reset(s0["env_seed"], PerturbationSpec.of(joint_radius_rad=s0["radius"], joint_dir_seed=s0["dir_seed"]))
    ro = recorded_obs(s0, recs[s0["episode_id"]])
    checks["info_fresh_vs_recorded_start0"] = {
        "pixels_equal": {c: bool(np.array_equal(np.asarray(fresh.frames[c]), ro.frames[c])) for c in ro.frames},
        "eef_max_abs_diff_m": float(np.abs(np.asarray(fresh.state["eef_pos"]) - np.asarray(ro.state["eef_pos"])).max())}
    pre_attach = chunk0(ro, s0["noise_seed"])
    checks["info_fresh_vs_recorded_start0"]["chunk_max_abs_diff"] = float(np.abs(chunk0(fresh, s0["noise_seed"]) - pre_attach).max())
    del env, fresh
    peft = PeftModel.from_pretrained(pol._policy, adapters[0][1], adapter_name=adapters[0][0], is_trainable=False)
    for n, d in adapters[1:]:
        peft.load_adapter(d, adapter_name=n, is_trainable=False)
    pol._policy.to(pol.device).eval()
    lm = peft.base_model
    mem_report("adapters attached")

    def use(label):
        if label == "base":
            lm.disable_adapter_layers()
        else:
            lm.enable_adapter_layers()
            peft.set_adapter(label)

    labels = ["base"] + [n for n, _ in adapters]
    with open(rows_p, "a") as fo:
        for k, s in enumerate(S):
            rec = recs[s["episode_id"]]
            assert all(f["driven"] for f in rec["frames"][:16]), s["episode_id"]
            label = np.array([f["action"] for f in rec["frames"][:16]])
            obs = recorded_obs(s, rec)
            row = {**s, "kind": "train", "offset_cm": float(np.linalg.norm(np.asarray(obs.state["eef_pos"]) - np.asarray(nom[str(s["scene"])]["eef_pos"])) * 100),
                   "label_disp": disp(label).tolist(), "disp": {"own": {}}}
            for lb in labels:
                use(lb)
                c = chunk0(obs, s["noise_seed"])
                row["disp"]["own"][lb] = disp(c).tolist()
                if k == 0:
                    if lb == "base":
                        checks["base_after_attach_bit_identical"] = bool(np.array_equal(c, pre_attach))
                    else:
                        checks.setdefault("adapter_chunk_differs_from_base", {})[lb] = bool(
                            not np.array_equal(disp(c), np.array(row["disp"]["own"]["base"])))
            fo.write(json.dumps(row) + "\n"); fo.flush()
            if k % 10 == 0:
                print(f"  [{time.time()-t0:6.0f}s] {k+1}/{len(S)} {s['episode_id']} scene {s['scene']}", flush=True)
        # matched held-out arm (my prim, 10-06): the 36 held-out seed-0 starts through this SAME path (fresh reset:
        # no images were recorded for them), noise seed 0, scored against the an7 nominal reference in `score`
        held = [x for x in json.load(open("experiments/repro/r056_heldout.json"))["instances"] if x["noise_seed"] == 0]
        for k, h in enumerate(held[: a.limit] if a.limit else held):
            env = make_env(h["scene_task_id"])
            obs = env.reset(h.get("env_seed", 0), PerturbationSpec.of(joint_radius_rad=float(h["radius"]), joint_dir_seed=int(h["dir_seed"])))
            row = {"kind": "heldout", "scene": int(h["scene_task_id"]), "dir_seed": int(h["dir_seed"]), "radius": float(h["radius"]),
                   "noise_seed": 0, "env_seed": int(h.get("env_seed", 0)),
                   "offset_cm": float(np.linalg.norm(np.asarray(obs.state["eef_pos"]) - np.asarray(nom[str(h["scene_task_id"])]["eef_pos"])) * 100),
                   "disp": {"own": {}}}
            for lb in labels:
                use(lb)
                row["disp"]["own"][lb] = disp(chunk0(obs, 0)).tolist()
            del env
            fo.write(json.dumps(row) + "\n"); fo.flush()
        print(f"  [{time.time()-t0:6.0f}s] held-out arm done", flush=True)
    use("base")
    mem_report("end")
    json.dump(checks, open(rows_p.replace(".jsonl", "_checks.json"), "w"), indent=1)
    print(json.dumps(checks, indent=1))
    if not checks.get("base_after_attach_bit_identical") or not all(checks.get("adapter_chunk_differs_from_base", {}).values()):
        sys.exit("CHECK FAILED")


def summarize(rs, ref_key):
    """Median distances to the reference disp, per model; share closer than base; |model - base|."""
    out = {"n": len(rs)}
    if not rs:
        return out
    labels = list(rs[0]["disp"]["own"].keys())
    for noise in ("own",):
        d = {}
        ref = lambda r: np.array(r[ref_key] if ref_key == "label_disp" else r["nominal_disp"])
        base = np.array([np.linalg.norm(np.array(r["disp"][noise]["base"]) - ref(r)) for r in rs])
        for lb in labels:
            x = np.array([np.linalg.norm(np.array(r["disp"][noise][lb]) - ref(r)) for r in rs])
            e = {"median_to_ref": float(np.median(x))}
            if lb != "base":
                xb = np.array([np.linalg.norm(np.array(r["disp"][noise][lb]) - np.array(r["disp"][noise]["base"])) for r in rs])
                e.update({"share_closer_than_base": float((x < base).mean()), "median_to_base": float(np.median(xb)),
                          "fraction_closed_of_median": float(1 - np.median(x) / np.median(base)) if np.median(base) else None})
            d[lb] = e
        out[noise] = d
    return out


def score(a):
    rows_p = ROWS if not a.limit else os.path.join(OUT, f"fit_firstchunk_rows_smoke{a.limit}.jsonl")
    ALL = [json.loads(l) for l in open(rows_p)]
    R = [r for r in ALL if r.get("kind", "train") == "train"]
    HR = [r for r in ALL if r.get("kind") == "heldout"]
    nom = nominal_disp()
    for r in HR:
        r["nominal_disp"] = nom[r["scene"]]
        r["dir"] = r["dir_seed"]
    for r in R:
        r["nominal_disp"] = nom[r["scene"]]
        r["dir"] = r["dir_seed"]
    seed0 = [r for r in R if r["noise_seed"] == 0]
    disc = [float(np.linalg.norm(np.array(r["label_disp"]) - np.array(r["nominal_disp"]))) for r in seed0]
    disc_by_seed = {str(ns): float(np.median([np.linalg.norm(np.array(r["label_disp"]) - np.array(r["nominal_disp"]))
                                              for r in R if r["noise_seed"] == ns])) for ns in sorted({r["noise_seed"] for r in R})}
    band = lambda cm: next(f"{lo}-{hi}" for lo, hi in BANDS_CM if lo <= cm < hi)
    out = {"what": "D20(i) follow-up: chunk-0 commanded-translation distance (an7 metric) on the R-054 training starts",
           "metric": "||a[:16,:3].sum(0) - ref||, env action units, medians over starts",
           "anchor": anchor()[0],
           "label_vs_nominal_seed0": {"n": len(disc), "max": max(disc) if disc else None,
                                      "median": float(np.median(disc)) if disc else None,
                                      "median_by_noise_seed": disc_by_seed},
           "to_label": {"all": summarize(R, "label_disp")},
           "to_nominal": {"all": summarize(R, "nominal")}}
    # matched held-out arm and its consistency with the stored rollouts' executed chunk 0 (the anchor's path)
    stored = {}
    for run_, arm in (("runs/r056_r16", "r16_opt1000"), ("runs/r058_r4", "r4_opt500")):
        for m, rr in load_run(run_):
            if m["set"] == "heldout" and m["noise_seed"] == 0:
                stored.setdefault((m["scene"], m["dir_seed"], m["radius"]), {})["base" if m["ckpt"] == "base" else arm] = disp(actions(rr))
    cons = {}
    for r in HR:
        st = stored.get((r["scene"], r["dir_seed"], r["radius"]), {})
        for lb, v in r["disp"]["own"].items():
            if lb in st:
                cons.setdefault(lb, []).append(float(np.linalg.norm(np.array(v) - st[lb])))
    out["heldout_matched"] = {"all": summarize(HR, "nominal"),
                              "by_dir_x_cm": {k: summarize([r for r in HR if f"dir{r['dir']}|{band(r['offset_cm'])}" == k], "nominal")
                                              for k in sorted({f"dir{r['dir']}|{band(r['offset_cm'])}" for r in HR})},
                              "path_vs_stored_rollout_disp": {lb: {"n": len(v), "median": float(np.median(v)), "max": float(np.max(v))}
                                                              for lb, v in cons.items()}}
    for key in ("to_label", "to_nominal"):
        ref = "label_disp" if key == "to_label" else "nominal"
        by = {}
        for r in R:
            by.setdefault(f"dir{r['dir']}|{band(r['offset_cm'])}", []).append(r)
        out[key]["by_dir_x_cm"] = {k: summarize(v, ref) for k, v in sorted(by.items())}
        out[key]["by_dir"] = {d: summarize([r for r in R if str(r["dir"]) == d], ref) for d in sorted({str(r["dir"]) for r in R})}
        out[key]["by_cm"] = {b: summarize([r for r in R if band(r["offset_cm"]) == b], ref) for b in [f"{lo}-{hi}" for lo, hi in BANDS_CM]}
    p = os.path.join(OUT, "fit_firstchunk.json" if not a.limit else f"fit_firstchunk_smoke{a.limit}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out["label_vs_nominal_seed0"], indent=1))
    print(json.dumps(out["to_label"]["all"], indent=1))
    print(json.dumps({k: out["heldout_matched"][k] for k in ("all", "path_vs_stored_rollout_disp")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["anchor", "run", "score"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    {"anchor": lambda: anchor(), "run": lambda: run(a), "score": lambda: score(a)}[a.phase]()


if __name__ == "__main__":
    main()
