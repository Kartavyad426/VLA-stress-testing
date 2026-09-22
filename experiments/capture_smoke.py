"""Tier-2 gates: the embedding capture against the REAL GR00T checkpoint.

Tier 1 (`tests/test_capture_taps.py`) runs every structural gate on CPU in
milliseconds against a stand-in. This is what proves the stand-in was not lying.
Neither tier substitutes for the other.

    .venvs/groot/bin/python experiments/capture_smoke.py \
        --checkpoint <ckpt> --episodes 2 --run-id capture_smoke

Gates, numbered as in docs/superpowers/specs/2026-09-18-embedding-capture-design.md:

  V8   forward-count join   rows per episode == Rollout.model_forwards, and
                            model_forwards > 0 FIRST -- see below
  V9   non-degenerate       per-signal variance across forwards > 0
  V10  scene separation     two different episodes farther apart than two
                            forwards of one episode
  V11  provenance           third_party/ unmodified
  V12  conformance          the existing gate still passes
  F5   cost                 wall-clock per forward at k=1 and k=4

WHY V8 CHECKS `> 0` BEFORE IT CHECKS EQUALITY. `model_forwards` landed in
commit 58eb285. EVERY trace written before it stores `forward_passes`, which
`rollout_from_dict` migrates into `env_steps`, leaving `model_forwards` at its
default of 0 (schema.py:216-218). Verified: the newest run on disk,
`groot_control_lplus_stack`, still stores `forward_passes` and has no
`model_forwards` key at all. So this gate run against a REPLAYED trace would
compare N captured rows against 0 -- and the natural guard, `if
model_forwards:`, would skip it and report green. A gate that can pass by being
skipped is worse than no gate, so this script runs FRESH episodes and asserts
the field is populated before it asserts anything about it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla_harness.capture.taps import MANIFEST  # noqa: E402
from vla_harness.envs.libero_env import LiberoEnv  # noqa: E402
from vla_harness.policies.lerobot_policy import LeRobotPolicy  # noqa: E402
from vla_harness.runner import rollout  # noqa: E402
from vla_harness.schema import PerturbationSpec  # noqa: E402

FAILURES: list[str] = []


def parse_override(kv: str):
    """Same --override contract as harness_eval.py.

    NOT optional for GR00T: the shipped checkpoint's `base_model_path` is a
    stale absolute path from NVIDIA's own build machine
    (/cache/huggingface/models--nvidia--GR00T-N1.7-3B/snapshots/...), which does
    not exist here and makes transformers reject it as a repo id. Every GR00T
    run in this repo overrides it, so the smoke must be able to as well.
    """
    k, v = kv.split("=", 1)
    try:
        v = json.loads(v)
    except json.JSONDecodeError:
        pass
    return k, v


def gate(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="0,1",
                    help="comma-separated task ids. TWO ARE REQUIRED for V10: "
                         "the scene-separation control needs two different "
                         "scenes, and two seeds of one task is the same scene.")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--obs-size", type=int, default=256)
    ap.add_argument("--rename-map", default='{"observation.images.image2": '
                                            '"observation.images.wrist_image"}')
    ap.add_argument("--run-id", default="capture_smoke")
    ap.add_argument("--k-resample", type=int, default=4)
    ap.add_argument("--override", action="append", default=[],
                    help="policy config override, e.g. "
                         "base_model_path=nvidia/GR00T-N1.7-3B (required for "
                         "GR00T: see parse_override)")
    a = ap.parse_args()

    # LIBERO renders through EGL; without this the env fails to make a context.
    os.environ.setdefault("MUJOCO_GL", "egl")

    cap_dir = os.path.join("runs", a.run_id, "capture")

    # --- V11: provenance, checked BEFORE anything runs ----------------------
    dirty = subprocess.run(["git", "status", "--porcelain", "third_party/"],
                           capture_output=True, text=True).stdout.strip()
    gate("V11 third_party/ unmodified", dirty == "",
         "clean" if not dirty else f"MODIFIED:\n{dirty}")

    from lerobot.envs.configs import LiberoEnv as EnvCfg

    print(f"\ncapture smoke: {a.episodes} episodes x tasks {a.tasks}, {a.suite}",
          flush=True)
    pol = LeRobotPolicy(a.checkpoint, n_action_steps=a.n_action_steps,
                        env_cfg=EnvCfg(task=a.suite), dtype=a.dtype,
                        rename_map=json.loads(a.rename_map),
                        policy_overrides=dict(parse_override(kv) for kv in a.override),
                        capture_dir=cap_dir, capture_k_resample=a.k_resample)
    task_ids = [int(t) for t in a.tasks.split(",")]
    rollouts, task_of = [], {}

    # Load the model BEFORE the clock starts. F5 is a per-forward marginal cost
    # used to budget a multi-hour capture; folding a ~20s one-time load into it
    # inflates the number and makes it drift with episode count (measured: 4.344
    # s/forward over 10 forwards vs 3.111 over 25, same k). Same run, same work.
    t_load = time.perf_counter()
    pol.reset()
    load_s = time.perf_counter() - t_load

    t_start = time.perf_counter()
    for tid in task_ids:
        env = LiberoEnv(suite=a.suite, task_id=tid, obs_size=a.obs_size)
        for seed in range(a.episodes):
            r = rollout(env, pol, seed=seed, spec=PerturbationSpec.of())
            rollouts.append(r)
            task_of[r.rollout_id] = tid
            print(f"  task{tid} episode {seed}: success={r.success} "
                  f"env_steps={r.env_steps} model_forwards={r.model_forwards}",
                  flush=True)
    wall = time.perf_counter() - t_start

    manifest = {}
    with open(os.path.join(cap_dir, MANIFEST)) as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                manifest[row["rollout_id"]] = row

    print("\ngates:", flush=True)

    # --- V8: forward-count join ---------------------------------------------
    populated = all(r.model_forwards > 0 for r in rollouts)
    gate("V8a model_forwards is populated (not a pre-58eb285 default of 0)",
         populated, f"{[r.model_forwards for r in rollouts]}")
    if populated:
        joined = all(r.rollout_id in manifest for r in rollouts)
        gate("V8b every rollout_id joins the manifest", joined,
             f"{len(manifest)} rows / {len(rollouts)} rollouts")
        if joined:
            mism = [(r.rollout_id, manifest[r.rollout_id]["n_forwards"],
                     r.model_forwards)
                    for r in rollouts
                    if manifest[r.rollout_id]["n_forwards"] != r.model_forwards]
            gate("V8c captured rows == model_forwards exactly", not mism,
                 "" if not mism else f"mismatches (id, captured, expected): {mism}")
            steps_ok = all(len(manifest[r.rollout_id]["env_step"] or []) ==
                           r.model_forwards for r in rollouts)
            gate("V8d forward-index -> env-step mapping is complete", steps_ok)

    # --- V9 / V10 -----------------------------------------------------------
    sigs = ["vl_encoder_mean", "vl_normed_mean", "vl_adapted_mean", "state_encoded"]
    per_ep = {}
    for r in rollouts:
        path = os.path.join(cap_dir, f"{r.rollout_id}.npz")
        if os.path.exists(path):
            with np.load(path) as z:
                per_ep[r.rollout_id] = {k: z[k].astype(np.float32)
                                        for k in sigs if k in z}

    for sig in sigs:
        vals = [v[sig] for v in per_ep.values() if sig in v and len(v[sig]) > 1]
        if not vals:
            gate(f"V9 {sig} non-degenerate", False, "absent or single-forward episode")
            continue
        var = float(np.mean([v.var(axis=0).mean() for v in vals]))
        gate(f"V9 {sig} non-degenerate across forwards", var > 0,
             f"mean variance {var:.3e}")

    # A hook that captures a constant is the most likely bug and would look
    # exactly like "no signal" downstream.
    #
    # V10 MEASURES ACROSS-TASK vs ACROSS-EPISODE-SAME-TASK, at matched forward
    # index. An earlier version compared each episode's MEAN against the
    # distance between one episode's FIRST and LAST forward, for two seeds of
    # ONE task -- and failed on all four signals against a capture that is
    # correct. Two seeds of one task is the same scene, not the "two visibly
    # different scenes" EXP_EMBEDDING_OOD.md §6 asks for; and an episode's
    # endpoints are its maximally-distant pair, so the comparison was biased
    # twice over. Measured on the real checkpoint, across-task exceeded
    # across-episode by 1.35x-2.71x while the old form reported FAIL.
    #
    # Matched forward index matters: within an episode the arm moves and the
    # scene changes with it, so consecutive forwards are genuinely far apart.
    # That is the encoder working, not a defect, and a gate must not read it as
    # one.
    by_task: dict[int, list] = {}
    for r in rollouts:
        if r.rollout_id in per_ep:
            by_task.setdefault(task_of[r.rollout_id], []).append(per_ep[r.rollout_id])

    if len(by_task) < 2:
        gate("V10 scene separation", False,
             f"needs >=2 tasks, got {sorted(by_task)} -- pass --tasks 0,1")
    else:
        (ta, ea), (tb, eb) = sorted(by_task.items())[:2]
        for sig in sigs:
            if any(sig not in e for e in ea + eb):
                continue
            def at(eps, i):
                return [e[sig][i] for e in eps if len(e[sig]) > i]

            same, cross = [], []
            for i in range(min(min(len(e[sig]) for e in ea),
                               min(len(e[sig]) for e in eb))):
                a_i, b_i = at(ea, i), at(eb, i)
                if len(a_i) > 1:
                    same.append(np.linalg.norm(a_i[0] - a_i[1]))
                cross.append(np.linalg.norm(a_i[0] - b_i[0]))
            if not same or not cross:
                continue
            s_m, c_m = float(np.mean(same)), float(np.mean(cross))
            gate(f"V10 {sig} separates task{ta} from task{tb}", c_m > s_m,
                 f"across-task {c_m:.3f} vs across-episode-same-task {s_m:.3f} "
                 f"({c_m / max(s_m, 1e-9):.2f}x)")

    # --- V1r / V2r: tier-1 structural gates, RE-ASSERTED ON THE REAL MODEL ---
    # Tier 1 proves these against a stand-in whose layout we wrote ourselves.
    # That is exactly how the `_groot_model.action_head` bug survived to the
    # first real run, so the structural claims are re-checked here against the
    # actual checkpoint's tensors.
    for r in rollouts[:1]:
        f = os.path.join(cap_dir, f"{r.rollout_id}.npz")
        if not os.path.exists(f):
            continue
        with np.load(f) as z:
            # V2r: with RTC off (vel_strength == 1) the Euler loop is exactly
            # actions_0 + dt*sum(pred[:, -H:]). If the decoder hook were on the
            # wrong module, or actions_0 inferred rather than observed, this
            # would not close. Tolerance is fp16 STORAGE rounding, not model
            # error -- the tensors are cast on write.
            a0 = z["actions_init"].astype(np.float64)
            dp = z["decode_path"].astype(np.float64)
            ap = z["action_pred"].astype(np.float64)
            rel = (np.abs(a0 + 0.25 * dp.sum(axis=1) - ap).max()
                   / max(np.abs(ap).max(), 1e-9))
            gate("V2r Euler path reconstructs action_pred on the real model",
                 rel < 2e-2, f"relative error {rel:.2e}")

            # V1r: vlln is a LayerNorm, so a genuine PRE-vlln tap must be on a
            # visibly larger scale than the post-vlln one. A hook that captured
            # the post-attention tensor twice would show no such drop -- which
            # is the failure mode V1 exists for, seen here in real numbers.
            s1 = z["vl_encoder_mean"].astype(np.float64)
            s15 = z["vl_normed_mean"].astype(np.float64)
            s2 = z["vl_adapted_mean"].astype(np.float64)
            gate("V1r pre-vlln tap is on a larger scale than post-vlln",
                 s1.std() > 2 * s15.std(),
                 f"S1 std {s1.std():.3f} vs S1.5 std {s15.std():.3f} "
                 f"({s1.std() / max(s15.std(), 1e-9):.1f}x)")
            gate("V1r the three VL taps are distinct tensors",
                 not np.allclose(s1, s2) and not np.allclose(s15, s2),
                 f"||S1-S2|| {np.linalg.norm(s1 - s2):.1f}, "
                 f"||S1.5-S2|| {np.linalg.norm(s15 - s2):.1f}")

            # V5r: the real dims, which tier 1 cannot assert -- the stand-in
            # uses small fake ones.
            shapes = {"vl_encoder_mean": 2048, "vl_normed_mean": 2048,
                      "vl_adapted_mean": 2048, "state_encoded": 1536}
            bad = {k: z[k].shape[-1] for k, want in shapes.items()
                   if k in z and z[k].shape[-1] != want}
            gate("V5r real embedding widths are 2048/2048/2048/1536", not bad,
                 "" if not bad else f"wrong: {bad}")
            gate("V5r action tensors sliced to 7 live dims from 132",
                 z["action_pred"].shape[-1] == 7 and int(z["sliced_from"]) == 132,
                 f"action_pred {z['action_pred'].shape}, "
                 f"sliced_from {int(z['sliced_from'])}")

            # fp16 headroom: vl_encoder is PRE-LayerNorm and runs hot.
            for k in z.files:
                if z[k].dtype == np.float16 and z[k].ndim:
                    peak = float(np.abs(z[k].astype(np.float64)).max())
                    if peak > 65504 / 4:
                        gate(f"fp16 headroom on {k}", False,
                             f"peak {peak:.0f}, only {65504/peak:.1f}x below "
                             f"the fp16 ceiling -- promote to fp32")

    # --- F5: cost, which primary needs before scheduling the full run -------
    total_fwd = sum(r.model_forwards for r in rollouts)
    if total_fwd:
        print(f"\nF5 cost: {wall:.1f}s for {total_fwd} forwards at "
              f"k={a.k_resample}  ->  {wall / total_fwd:.3f} s/forward "
              f"(model load {load_s:.1f}s, EXCLUDED)", flush=True)
        print("     re-run with --k-resample 1 for the baseline; the delta is "
              "the S4 budget for the full capture.", flush=True)

    # --- V12: conformance ---------------------------------------------------
    cf = subprocess.run([sys.executable, "-m", "vla_harness.conformance",
                         "--checkpoint", a.checkpoint,
                         "--obs-size", str(a.obs_size)],
                        capture_output=True, text=True)
    gate("V12 conformance gate still passes", cf.returncode == 0,
         (cf.stdout or cf.stderr).strip().splitlines()[-1] if (cf.stdout or cf.stderr) else "")

    print()
    if FAILURES:
        print(f"SMOKE FAILED -- {len(FAILURES)} gate(s):", flush=True)
        for f in FAILURES:
            print(f"  - {f}", flush=True)
        sys.exit(1)
    print("SMOKE PASSED -- all tier-2 gates green.", flush=True)


if __name__ == "__main__":
    main()
