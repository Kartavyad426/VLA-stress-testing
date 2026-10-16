#!/usr/bin/env python3
"""Check that LIBERO-MAX rollouts' saved sim states (npz sim_states, written by lmax_obs.py) RESTORE the recorded
state, and that stepping the recorded action from a restored state reproduces the next recorded state.

For each sampled step t of a Base arm (or a Dynamic arm before its event, where no model edit is involved):
  1. rebuild the case env exactly as the shard does, reset, set_init_state(initial state of the case);
  2. env.set_init_state(sim_states[t]); compare the resulting obs (eef pos/quat, joints, gripper) and every free
     body's position against steps[t].obs_state  -> restore error;
  3. step steps[t].action once; compare against steps[t+1].obs_state            -> one-step replay error.
Writes <run_root>/restore_check.json. No policy, no model: simulator + offscreen renderer only (EGL by default).

  PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 .venvs/libero-plus/bin/python experiments/lmax_restore_check.py \
      third_party/LIBERO-MAX/benchmark/lite/libero_max_lite.json runs/lmax/<run>/<suite_source> [--per-arm 6]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "float_"):
    np.float_ = np.float64

REPO = Path(__file__).resolve().parents[1]
LMAX = REPO / "third_party" / "LIBERO-MAX"
for p in (REPO, LMAX / "src", LMAX / "scripts", REPO / "experiments"):
    sys.path.insert(0, str(p))

from libero_max.env_factory import create_libero_env_with_retry  # noqa: E402
from libero_max.manifest import load_manifest  # noqa: E402
from libero_max.pro_runtime import wrap_case_env  # noqa: E402
from libero_max.substrate import load_case_task  # noqa: E402
import lmax_obs  # noqa: E402
from lmax_groot_shard import _create_libero_env  # noqa: E402

KEYS = ("eef_pos", "eef_quat", "gripper_qpos", "joint_pos")


def _diff(a: dict, b: dict) -> dict:
    out = {k: float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k])))) for k in KEYS}
    sa, sb = a.get("_gt_scene_object_pos") or {}, b.get("_gt_scene_object_pos") or {}
    common = set(sa) & set(sb)
    out["scene_pos"] = max((float(np.max(np.abs(np.asarray(sa[n]) - np.asarray(sb[n])))) for n in common),
                           default=0.0)
    out["scene_missing"] = sorted(set(sa) ^ set(sb))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--per-arm", type=int, default=6, help="steps sampled per arm")
    ap.add_argument("--max-rows", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=360)
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    cases = {c["case_id"]: c for c in load_manifest(args.manifest)["cases"]}
    rows = [json.loads(line) for line in (args.run_root / "rollouts.jsonl").open()][: args.max_rows]
    suite_cache, report = {}, []
    for row in rows:
        case = cases[row["case"]["case_id"]]
        steps = row["steps"]
        z = np.load(row["meta"]["npz"]["predictions"])
        states = z["sim_states"]
        t_hi = len(steps) - 1
        if row["event"] and row["event"].get("t_event") is not None:
            t_hi = min(t_hi, int(row["event"]["t_event"]) - 1)    # restoring post-event needs the model edits
        if t_hi < 1:
            continue
        ts = sorted({int(x) for x in np.linspace(0, t_hi - 1, num=min(args.per_arm, t_hi))})
        task, init_states = load_case_task(case, benchmark, suite_cache)
        seed = int(case["policy_seed"])
        env, _ = create_libero_env_with_retry(
            lambda: _create_libero_env(task, args.resolution, seed, bddl_root=get_libero_path("bddl_files"),
                                       env_type=OffScreenRenderEnv),
            policy_seed=seed, reseed=lambda v: np.random.seed(v))
        env = wrap_case_env(env, case)
        raw = env
        while not (hasattr(raw, "env") and hasattr(raw.env, "sim")) and hasattr(raw, "_env"):
            raw = raw._env
        rec = lmax_obs.ArmRecorder(raw, "check", 10, event_entities=[])
        try:
            env.reset()
            env.set_init_state(init_states[case["init_state_index"]])
            per = []
            for t in ts:
                obs = env.set_init_state(states[t])
                restored = rec._state(obs)
                r_err = _diff(restored, steps[t]["obs_state"])
                s_err = None
                if steps[t]["action"] is not None and t + 1 < len(steps) and t + 1 <= t_hi:
                    obs2, _, _, _ = env.step(steps[t]["action"])
                    s_err = _diff(rec._state(obs2), steps[t + 1]["obs_state"])
                per.append({"t": t, "restore": r_err, "one_step": s_err})
        finally:
            env.close()
        worst_r = max(max(v for k, v in p["restore"].items() if k != "scene_missing") for p in per)
        worst_s = max((max(v for k, v in p["one_step"].items() if k != "scene_missing")
                       for p in per if p["one_step"]), default=None)
        report.append({"case_id": case["case_id"], "arm": row["case"]["arm"], "t_checked": ts,
                       "worst_restore_abs": worst_r, "worst_one_step_abs": worst_s, "per_t": per})
        print("%s %s restore max|d| %.3g  one-step max|d| %s" % (case["case_id"], row["case"]["arm"], worst_r,
                                                                 "n/a" if worst_s is None else "%.3g" % worst_s))
    out = {"run_root": str(args.run_root), "rows_checked": len(report), "per_row": report,
           "worst_restore_abs": max((r["worst_restore_abs"] for r in report), default=None),
           "worst_one_step_abs": max((r["worst_one_step_abs"] for r in report
                                      if r["worst_one_step_abs"] is not None), default=None)}
    (args.run_root / "restore_check.json").write_text(json.dumps(out, indent=2) + "\n")
    print("worst restore %s, worst one-step %s" % (out["worst_restore_abs"], out["worst_one_step_abs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
