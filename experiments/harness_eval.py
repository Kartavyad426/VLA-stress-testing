"""Evaluate a policy through OUR harness (env adapter + policy adapter + trace store).

The counterpart to running `lerobot-eval` directly. Same checkpoint, seeds and
init states; any systematic gap between the two paths is a harness defect,
because the policy and the simulator are identical.

Resumable: traces go to runs/<run_id>/rollouts.jsonl via run_cell (G10), and
seed N always starts from init state N, so a resumed cell is on the same
layouts as an uninterrupted one. One summary line per cell is appended to
runs/<run_id>/cells.jsonl.

    MUJOCO_GL=egl .venvs/minerva/bin/python experiments/harness_eval.py \
        --checkpoint third_party/MINERVA/ckpt/t05_l1_0.54M --n-action-steps 1 \
        --override temporal_ensemble_coeff=0.01 --obs-size 360 \
        --suites libero_object,libero_spatial --episodes 10 --run-id harness_minerva
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla_harness.envs.libero_env import LiberoEnv
from vla_harness.policies.lerobot_policy import LeRobotPolicy
from vla_harness.runner import run_cell
from vla_harness.schema import PerturbationSpec, TraceStore, ArmLog


def parse_override(kv: str):
    k, v = kv.split("=", 1)
    try:
        v = json.loads(v)
    except json.JSONDecodeError:
        pass
    return k, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--override", action="append", default=[],
                    help="policy config override, key=value (repeatable)")
    ap.add_argument("--preprocessor-override", action="append", default=[],
                    help="shipped-preprocessor override, step.key=value (repeatable). "
                         "A setting can live in both the policy config and the "
                         "preprocessor; --override only reaches the former.")
    ap.add_argument("--obs-size", type=int, default=256)
    ap.add_argument("--dtype", default=None, help="e.g. bfloat16: load on CPU, cast, move (GR00T on 8 GB)")
    ap.add_argument("--rename-map", default="{}", help='JSON, e.g. {"observation.images.image2": "observation.images.wrist_image"}')
    ap.add_argument("--libero-plus", action="store_true",
                    help="task ids index LIBERO-Plus perturbed variants (needs .venvs/libero-plus, "
                         "PYTHONPATH=third_party/LIBERO-plus, LIBERO_CONFIG_PATH=third_party/libero-plus-config)")
    ap.add_argument("--base-instruction", action="store_true",
                    help="force the base-scene instruction even on a LANGUAGE "
                         "variant; with a canonical-camera initstate-0 variant "
                         "this is an UNPERTURBED control on the LIBERO-Plus stack")
    ap.add_argument("--raw-instruction", action="store_true",
                    help="LIBERO-Plus: keep LeRobot's contaminated instruction (measurement only)")
    ap.add_argument("--instruction-override", default=None,
                    help="Replace the instruction for EVERY episode. Pass an empty "
                         "string for the null-prompt arm. The env already supports "
                         "this via instruction_override (libero_env.py:305-309); this "
                         "exposes it to a campaign. Recorded in the trace, and the "
                         "per-step Observation.instruction shows what was actually sent.")
    ap.add_argument("--suites", required=True)
    ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--specs", default="[{}]",
                    help='JSON list of perturbation knob dicts, e.g. [{}, {"camera_yaw_deg": 5}]')
    ap.add_argument("--capture-dir", default=None,
                    help="write per-episode embedding sidecars here (GR00T "
                         "only). Off by default and absent from the policy "
                         "identity, so policy_id and the trace cache are "
                         "unchanged and a captured run stays comparable with "
                         "every run already on disk.")
    ap.add_argument("--capture-k-resample", type=int, default=1,
                    help="k>1 measures S4 self-consistency by re-running the "
                         "denoise loop k-1 extra times. Measured cost: +17%% "
                         "wall at k=4 (1.815 -> 2.208 s/forward).")
    ap.add_argument("--no-video", action="store_true",
                    help="skip the per-episode mp4 (runs/<run-id>/video/<rollout_id>.mp4, "
                         "~1 MB/episode, CPU-encoded). On by default so every failure "
                         "can be watched or VLM-labelled without a sim replay. Not part "
                         "of the policy identity: trace cache and rollout ids unchanged.")
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()

    overrides = dict(parse_override(kv) for kv in a.override)
    pre_overrides: dict[str, dict] = {}
    for item in a.preprocessor_override:
        path, val = parse_override(item)
        if "." not in path:
            raise SystemExit(f"--preprocessor-override needs step.key=value, got {item!r}")
        step, key = path.split(".", 1)
        pre_overrides.setdefault(step, {})[key] = val
    specs = json.loads(a.specs)
    video_dir = None if a.no_video else os.path.join("runs", a.run_id, "video")
    if video_dir:
        # fail at launch, not at the first frame of the first episode after a
        # multi-minute model load (.venvs/minerva-lplus ships without it).
        try:
            import imageio_ffmpeg  # noqa: F401
        except ImportError:
            raise SystemExit("video is on by default but imageio-ffmpeg is not "
                             "installed in this venv: pip install imageio-ffmpeg, "
                             "or pass --no-video")
    store, arms = TraceStore("runs", a.run_id), ArmLog("runs", a.run_id)
    cells_path = os.path.join("runs", a.run_id, "cells.jsonl")
    done = set()
    if os.path.exists(cells_path):
        for line in open(cells_path):
            c = json.loads(line)
            done.add((c["suite"], c["task"], json.dumps(c["spec"], sort_keys=True)))

    if a.libero_plus:
        from lerobot.envs.configs import LiberoPlusEnv as EnvCfg
    else:
        from lerobot.envs.configs import LiberoEnv as EnvCfg
    seeds = list(range(a.episodes))
    for suite in a.suites.split(","):
        pol = LeRobotPolicy(a.checkpoint, n_action_steps=a.n_action_steps,
                            env_cfg=EnvCfg(task=suite), policy_overrides=overrides,
                            dtype=a.dtype, rename_map=json.loads(a.rename_map),
                            preprocessor_overrides=pre_overrides,
                            capture_dir=a.capture_dir,
                            capture_k_resample=a.capture_k_resample)
        for tid in [int(t) for t in a.tasks.split(",")]:
            env = LiberoEnv(suite=suite, task_id=tid, obs_size=a.obs_size,
                            libero_plus=a.libero_plus,
                            libero_plus_raw_instruction=a.raw_instruction,
                            libero_plus_base_instruction=a.base_instruction)
            if a.instruction_override is not None:
                # reset() reads this every episode (libero_env.py:308).
                env.instruction_override = a.instruction_override
            for knobs in specs:
                key = (suite, tid, json.dumps(knobs, sort_keys=True))
                if key in done:
                    print(f"{suite} task{tid} {knobs}: done, skipping", flush=True)
                    continue
                t0 = time.time()
                cell = run_cell(env, pol, PerturbationSpec.of(**knobs), seeds,
                                store=store, arms=arms, arm="uniform",
                                video_dir=video_dir)
                row = {"suite": suite, "task": tid, "spec": knobs,
                       "n": cell.n, "successes": cell.successes,
                       "rate": cell.rate, "ci95": list(cell.ci),
                       "wall_s": round(time.time() - t0, 1),
                       "checkpoint": a.checkpoint, "n_action_steps": a.n_action_steps,
                       "overrides": overrides, "obs_size": a.obs_size,
                       "dtype": a.dtype, "rename_map": json.loads(a.rename_map),
                       "libero_plus": a.libero_plus,
                       "capture_dir": a.capture_dir,
                       "video_dir": video_dir,
                       "capture_k_resample": a.capture_k_resample,
                       "raw_instruction": a.raw_instruction,
                       "base_instruction": a.base_instruction}
                with open(cells_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{suite} task{tid} {knobs}: {cell.successes}/{cell.n} "
                      f"({row['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
