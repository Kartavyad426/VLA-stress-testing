#!/usr/bin/env python3
"""LIBERO-MAX shard runner for GR00T N1.7 (LeRobot -640 ports), one suite, one resident policy.

A copy of LIBERO-MAX's reference X-VLA shard (third_party/LIBERO-MAX/scripts/run_xvla_persistent_shard.py
@01625e2) with only the policy swapped. Their event wrapper (CosmosInterventionEnv), env factory, PRO wrapper,
prefix replay and pair classifier are used unchanged, so the output under --output-root is exactly what their
aggregator (scripts/aggregate_cosmos_benchmark.py) reads: cases/<case_id>/{scenario.json, control/trace.jsonl,
intervention/trace.jsonl, paired_summary.json, status.json, DONE}.

GR00T is loaded the way every GR00T run in this repo is (vla_harness LeRobotPolicy: bf16 streamed cast, the
checkpoint's processors, LeRobot's LIBERO env processor, which builds observation.state and flips both image
axes). Differences from the X-VLA shard, all forced by the policy:
  * relative (delta) control throughout: GR00T's LIBERO checkpoints predict deltas, so there is no switch to
    absolute control after warmup;
  * query interval Q (default 8, the paper's GR00T serving, Table 9) is GR00T's n_action_steps; one model
    forward per query, its first Q actions executed;
  * render resolution defaults to 360, the setting of our GR00T runs since R-039 (the paper does not state
    GR00T's); recorded in run_config.json.
Everything else (MAX_STEPS, 10 warmup steps, policy seed 195, query seed = seed + policy_step, fresh env per
arm, Dynamic replays Base's recorded chunks until the event) is theirs.

Run (one suite, LIBERO-Plus cases), from the repo root:
  PYTHONPATH=$PWD/third_party/LIBERO-plus \
  LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  .venvs/libero-plus/bin/python experiments/lmax_groot_shard.py \
      third_party/LIBERO-MAX/benchmark/lite/libero_max_lite.json --suite libero_spatial --source plus \
      --output-root runs/lmax/<run>/plus --resume [--case-id ID ...] [--max-cases N]
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# LIBERO-Plus and its MuJoCo assets still reference NumPy's pre-2.0 scalar alias (as in the X-VLA shard).
if not hasattr(np, "float_"):
    np.float_ = np.float64

REPO = Path(__file__).resolve().parents[1]
LMAX = REPO / "third_party" / "LIBERO-MAX"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(LMAX / "src"))
sys.path.insert(0, str(LMAX / "scripts"))
sys.path.insert(0, str(REPO / "experiments"))

from libero_max.cosmos_integration import CosmosInterventionEnv  # noqa: E402
from libero_max.env_factory import create_libero_env_with_retry  # noqa: E402
from libero_max.manifest import load_manifest  # noqa: E402
from libero_max.pro_runtime import wrap_case_env  # noqa: E402
from libero_max.substrate import load_case_task  # noqa: E402
from summarize_cosmos_paired_smoke import classify_persistent_pair  # noqa: E402

from vla_harness.policies.lerobot_policy import LeRobotPolicy  # noqa: E402
import lmax_obs  # noqa: E402  (experiments/, the agreed rollout format)
from vla_harness.schema import Observation  # noqa: E402

# Identical to the X-VLA shard's table.
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

CHECKPOINTS = {s: "nvidia/gr00t17-lerobot-%s-640" % s for s in MAX_STEPS}
# The settings every GR00T run in this repo uses (e.g. runs/r053_N_s1 fingerprint).
GROOT_OVERRIDES = {"base_model_path": "nvidia/GR00T-N1.7-3B", "embodiment_tag": "libero_sim"}
GROOT_RENAME = {"observation.images.image2": "observation.images.wrist_image"}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.%d" % os.getpid())
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _create_libero_env(task: Any, resolution: int, seed: int, *, bddl_root: str, env_type: Any):
    bddl_path = Path(bddl_root) / task.problem_folder / task.bddl_file
    env = env_type(bddl_file_name=str(bddl_path), camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env, task.language


def _observation(raw_obs: dict, instruction: str, t: int) -> Observation:
    """Raw robosuite obs -> the Observation LeRobotPolicy._build_batch reads: un-flipped frames under LeRobot's
    LIBERO camera names, plus eef pos/quat and gripper qpos (LeRobot's env processor builds the state and flips)."""
    return Observation(
        instruction=instruction,
        state={"eef_pos": np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64),
               "eef_quat": np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float64),
               "gripper_qpos": np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float64)},
        t=t,
        frames={"image": raw_obs["agentview_image"], "image2": raw_obs["robot0_eye_in_hand_image"]})


class _ModelTap:
    """Read-only taps on the loaded GR00T N1.7 model: the head's FULL action_pred (all H steps, normalised, before
    LeRobot truncates it to chunk_size) and the backbone output, mean-pooled over attended tokens and over image
    tokens. Neither changes what the policy computes."""

    def __init__(self, policy):
        model = policy._groot_model
        self.last_pred, self.last_feats = None, None
        orig = model.get_action

        def get_action(*a, **k):
            out = orig(*a, **k)
            self.last_pred = out["action_pred"][0].detach().float().cpu().numpy()
            return out
        model.get_action = get_action

        def bb_hook(_m, _inp, out):
            f = out["backbone_features"][0].detach().float()
            att = out["backbone_attention_mask"][0].bool()
            img = out["image_mask"][0].bool() & att
            self.last_feats = {"mean": f[att].mean(0).cpu().numpy(),
                               "img_mean": (f[img].mean(0) if bool(img.any()) else f[att].mean(0)).cpu().numpy()}
        model.backbone.register_forward_hook(bb_hook)


def _predict_action_chunk(pol: LeRobotPolicy, raw_obs: dict, instruction: str, t: int,
                          query_interval: int, query_seed: int, tap: "_ModelTap" = None):
    """One GR00T forward under a query-local seed; its first `query_interval` actions in env units.
    Same structure as the X-VLA shard: build the batch once, then pop Q actions from a fresh queue."""
    from lerobot.utils.constants import ACTION
    torch.manual_seed(query_seed)
    torch.cuda.manual_seed_all(query_seed)
    pol._policy.reset()
    batch = pol._build_batch(_observation(raw_obs, instruction, t))
    actions = []
    with torch.inference_mode():
        for _ in range(query_interval):
            a = pol._post(pol._policy.select_action(batch))
            a = pol._env_post({ACTION: a})[ACTION]
            actions.append(a.squeeze(0).detach().float().cpu().numpy())
        full_norm = full_env = feats = None
        if tap is not None:
            full_norm = tap.last_pred[:, :lmax_obs.LIVE_DIMS]
            env_rows = []
            for i in range(full_norm.shape[0]):   # every step of the full chunk through the same postprocessors
                x = torch.as_tensor(full_norm[i:i + 1], device=pol.device)
                x = pol._env_post({ACTION: pol._post(x)})[ACTION]
                env_rows.append(x.squeeze(0).detach().float().cpu().numpy())
            full_env = np.asarray(env_rows, dtype=np.float32)
            feats = tap.last_feats
    pol.model_forwards += 1
    return np.asarray(actions, dtype=np.float32), full_norm, full_env, feats


def _git_head(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--suite", required=True, choices=sorted(MAX_STEPS))
    ap.add_argument("--source", required=True, choices=("plus", "pro"))
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--checkpoint", default=None, help="default: nvidia/gr00t17-lerobot-<suite>-640")
    ap.add_argument("--query-interval", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=360)
    ap.add_argument("--case-id", action="append", default=[])
    ap.add_argument("--max-cases", type=int)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-obs", action="store_true", help="skip the observability output (LIBERO-MAX files only)")
    ap.add_argument("--query-px", type=int, default=256, choices=(256, 360))
    ap.add_argument("--every-step-px", type=int, default=None, choices=(256,),
                    help="also store both cameras at EVERY env step at this size")
    args = ap.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        ap.error("invalid shard")

    from lerobot.envs.configs import LiberoPlusEnv
    from lerobot.envs.libero import get_libero_dummy_action
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    manifest = load_manifest(args.manifest)
    cases = [c for c in manifest["cases"]
             if c["task_suite_name"] == args.suite
             and ((c.get("substrate_variant", {}) or {}).get("benchmark") == "LIBERO-PRO") == (args.source == "pro")]
    if args.case_id:
        wanted = set(args.case_id)
        cases = [c for c in cases if c["case_id"] in wanted]
        missing = wanted - {c["case_id"] for c in cases}
        if missing:
            ap.error("case ids not in this suite/source: %s" % sorted(missing))
    selected = cases[args.shard_index::args.num_shards]
    if args.max_cases is not None:
        selected = selected[: args.max_cases]
    policy_seeds = {int(c["policy_seed"]) for c in selected}
    if len(policy_seeds) > 1:
        raise ValueError("one persistent shard requires one policy seed")
    policy_seed = next(iter(policy_seeds), 195)

    checkpoint = args.checkpoint or CHECKPOINTS[args.suite]
    torch.manual_seed(policy_seed)
    torch.cuda.manual_seed_all(policy_seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: refusing the silent CPU fallback")
    pol = LeRobotPolicy(checkpoint, n_action_steps=args.query_interval,
                        env_cfg=LiberoPlusEnv(task=args.suite), policy_overrides=dict(GROOT_OVERRIDES),
                        dtype="bfloat16", rename_map=dict(GROOT_RENAME))
    pol.reset()  # loads
    tap = None if args.no_obs else _ModelTap(pol._policy)
    resolved = pol.resolved_config()
    if int(resolved.get("n_action_steps", -1)) != args.query_interval:
        raise SystemExit("resolved n_action_steps %r != query interval %d" % (resolved.get("n_action_steps"),
                                                                             args.query_interval))
    if args.query_interval > int(resolved.get("chunk_size", 0)):
        raise SystemExit("query interval exceeds GR00T chunk_size %r" % resolved.get("chunk_size"))

    import mujoco
    import robosuite
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "run_config.json", {
        "runner": "experiments/lmax_groot_shard.py", "manifest": str(args.manifest),
        "suite": args.suite, "source": args.source, "checkpoint": checkpoint,
        "checkpoint_revision": _snapshot_revision(checkpoint),
        "query_interval": args.query_interval, "resolution": args.resolution,
        "max_policy_steps": MAX_STEPS[args.suite], "warmup_steps": 10, "policy_seed": policy_seed,
        "control": "relative (delta) throughout", "resolved_policy_config": resolved,
        "policy_id": pol.policy_id, "n_cases": len(selected),
        "libero_max_commit": _git_head(LMAX), "repo_commit": _git_head(REPO),
        "libero_pro_commit": _git_head(REPO / "third_party" / "LIBERO-PRO"),
        "observability": None if args.no_obs else {"format": "PORTABILITY_LIBERO_MAX.html §5 + A1-A7",
                                                   "query_px": args.query_px, "every_step_px": args.every_step_px},
        "not_in_this_run": "multi-draw B=32 / STAC: it changes the serving noise path, so it gets its own run",
        "libero_source": os.environ.get("LIBERO_SOURCE_PACKAGE_ROOT"),
        "mujoco": mujoco.__version__, "robosuite": robosuite.__version__, "torch": torch.__version__,
        "started": time.strftime("%F %T")})

    rc = json.loads((args.output_root / "run_config.json").read_text())
    fingerprint = {"env": {"name": "libero_max", "suite": args.suite, "source": args.source,
                           "obs_size": args.resolution, "max_steps": MAX_STEPS[args.suite], "warmup_steps": 10,
                           "control_mode": "relative"},
                   "policy": pol.identity(),
                   "mujoco": rc["mujoco"], "robosuite": rc["robosuite"], "libero_max_commit": rc["libero_max_commit"],
                   "libero_pro_commit": rc["libero_pro_commit"], "repo_commit": rc["repo_commit"],
                   "checkpoint": checkpoint, "checkpoint_revision": rc["checkpoint_revision"],
                   "H": None, "Q": args.query_interval}
    suite_cache: Dict[str, Any] = {}
    failures = []
    for ordinal, case in enumerate(selected, start=1):
        case_id = case["case_id"]
        case_dir = args.output_root / "cases" / case_id
        summary_path, done_path = case_dir / "paired_summary.json", case_dir / "DONE"
        if args.resume and done_path.exists() and (summary_path.exists() or (case_dir / "status.json").exists()):
            print("[%d/%d] %s skipped-complete" % (ordinal, len(selected), case_id), flush=True)
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        for stale in (done_path, case_dir / "FAILED", summary_path):
            if stale.exists():
                stale.unlink()
        _write_json(case_dir / "scenario.json", case["scenario"])
        t_case = time.time()
        try:
            task, initial_states = load_case_task(case, benchmark, suite_cache)
            initial_state = initial_states[case["init_state_index"]]
            seed = int(case["policy_seed"])
            rows, recs, scenes = {}, {}, {}
            for arm in ("control", "intervention"):
                arm_dir = case_dir / arm
                arm_dir.mkdir(parents=True, exist_ok=True)
                trace_path = arm_dir / "trace.jsonl"
                trace_path.write_text("", encoding="utf-8")

                def reseed(value: int) -> None:
                    np.random.seed(value)
                    torch.manual_seed(value)
                    torch.cuda.manual_seed_all(value)

                env, task_description = create_libero_env_with_retry(
                    lambda: _create_libero_env(task, args.resolution, seed,
                                               bddl_root=get_libero_path("bddl_files"),
                                               env_type=OffScreenRenderEnv),
                    policy_seed=seed, reseed=reseed)
                env = wrap_case_env(env, case)
                env.seed(seed)
                wrapped = CosmosInterventionEnv(
                    env=env, task_description=task_description, scenario=case["scenario"], arm=arm,
                    trace_path=trace_path, original_task_index=case["task_index"],
                    init_state_index=case["init_state_index"])
                wrapped.configure_episode(task_suite_name=case["task_suite_name"], policy_seed=seed,
                                          query_interval=args.query_interval,
                                          max_policy_steps=MAX_STEPS[case["task_suite_name"]])
                control_queries = {q["policy_step"]: q["actions"]
                                   for q in rows.get("control", {}).get("policy_queries", [])}
                success = False
                rec = None
                try:
                    wrapped.reset()
                    observation = wrapped.set_init_state(initial_state)
                    raw_env = env   # the OffScreenRenderEnv whose .env is the LIBERO problem env (holds .sim)
                    while not (hasattr(raw_env, "env") and hasattr(raw_env.env, "sim")) and hasattr(raw_env, "_env"):
                        raw_env = raw_env._env
                    if not args.no_obs:
                        rec = lmax_obs.ArmRecorder(raw_env, arm, wrapped.warmup_steps, query_px=args.query_px,
                                                   every_step_px=args.every_step_px,
                                                   event_entities=lmax_obs.event_entities(case["scenario"]))
                        rec.obs(0, observation)
                        scenes[arm] = lmax_obs._scene_descriptor(raw_env, rec.gt, case, task_description,
                                                                 observation["robot0_eef_pos"])
                    n_forward = 0
                    action_plan = collections.deque()
                    total_limit = MAX_STEPS[case["task_suite_name"]] + wrapped.warmup_steps
                    for total_step in range(total_limit):
                        if total_step < wrapped.warmup_steps:
                            dummy = get_libero_dummy_action()
                            if rec:
                                rec.act(wrapped.total_env_steps, dummy)
                            n_ev = len(wrapped.events)
                            observation, _, _, _ = wrapped.step(dummy)
                            if rec:
                                rec.obs(wrapped.total_env_steps, observation)
                            continue
                        if not action_plan:
                            instruction = wrapped.runtime.current_instruction
                            policy_step = max(0, wrapped.total_env_steps - wrapped.warmup_steps)
                            replay = (arm == "intervention" and not wrapped.runtime.applied
                                      and bool(control_queries))
                            if replay:
                                if policy_step not in control_queries:
                                    raise RuntimeError("control trace missing query step %d" % policy_step)
                                actions = np.asarray(control_queries[policy_step], dtype=np.float32)
                                source = "control_replay"
                                full_norm = full_env = feats = None
                            else:
                                actions, full_norm, full_env, feats = _predict_action_chunk(
                                    pol, observation, instruction, wrapped.total_env_steps, args.query_interval,
                                    seed + policy_step, tap)
                                source = "model"
                            if actions.ndim != 2 or len(actions) < args.query_interval:
                                raise ValueError("GR00T returned an invalid action chunk: %s" % (actions.shape,))
                            actions = actions[: args.query_interval]
                            wrapped.record_policy_query(actions, instruction=instruction, source=source)
                            if rec:
                                rec.query(wrapped.total_env_steps, observation, source, instruction, n_forward,
                                          full_norm, seed + policy_step if source == "model" else None,
                                          full_env, feats)
                            n_forward += 1
                            action_plan.extend(actions)
                        a_t = np.asarray(action_plan.popleft()).tolist()
                        if rec:
                            rec.act(wrapped.total_env_steps, a_t)
                        n_ev = len(wrapped.events)
                        observation, _, done, _ = wrapped.step(a_t)
                        if rec:
                            if len(wrapped.events) > n_ev:
                                rec.event(wrapped.total_env_steps, observation)
                            rec.obs(wrapped.total_env_steps, observation)
                        if done:
                            success = True
                            break
                    rows[arm] = wrapped.record_outcome(success)
                    if rec:
                        rec.finish()
                        recs[arm] = rec
                finally:
                    env.close()
            summary, terminal = classify_persistent_pair(rows["control"], rows["intervention"])
            if not args.no_obs:
                npz_paths = {arm: recs[arm].save_npz(args.output_root / "npz", case_id) for arm in recs}
                fingerprint["H"] = max(v["H"] for v in npz_paths.values()) or fingerprint["H"]
                meta = {"resolved_policy_config": resolved, "control_mode": "relative",
                        "max_policy_steps": MAX_STEPS[case["task_suite_name"]], "obs_size": args.resolution,
                        "warmup_steps": 10, "query_interval": args.query_interval}
                obs_rows = lmax_obs.build_rows(case, recs, rows, scenes, fingerprint, meta, pol.policy_id,
                                               npz_paths, 10)
                lmax_obs.append_rows(args.output_root / "rollouts.jsonl", obs_rows)
            if summary is not None:
                _write_json(summary_path, summary)
            status = {"case_id": case_id, "shard_index": args.shard_index,
                      "wall_s": round(time.time() - t_case, 1),
                      "control_success": bool(rows["control"]["success"]),
                      "intervention_success": bool(rows["intervention"]["success"])}
            if terminal is not None:
                status.update(terminal)
            _write_json(case_dir / "status.json", status)
            done_path.touch()
            print("[%d/%d] %s completed base=%d dynamic=%d events=%d (%.0fs)" % (
                ordinal, len(selected), case_id, rows["control"]["success"], rows["intervention"]["success"],
                rows["intervention"]["intervention_event_count"], time.time() - t_case), flush=True)
        except Exception as exc:
            failures.append(case_id)
            _write_json(case_dir / "status.json", {
                "case_id": case_id, "shard_index": args.shard_index,
                "error": "%s: %s" % (type(exc).__name__, exc), "traceback": traceback.format_exc()})
            (case_dir / "FAILED").touch()
            print("[%d/%d] %s failed: %s" % (ordinal, len(selected), case_id, exc), flush=True)
    return 1 if failures else 0


def _snapshot_revision(repo_id: str):
    """The HF cache's resolved revision for repo_id (refs/main), or None for a local path."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        ref = Path(HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--")) / "refs" / "main"
        return ref.read_text().strip() if ref.exists() else None
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
