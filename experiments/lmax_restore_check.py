#!/usr/bin/env python3
"""Check that LIBERO-MAX rollouts' saved sim states (npz, written by lmax_obs.py) are restorable.

robosuite samples observations on the first substep of each control step, so two states are saved per step:
  sim_states[t]      end-of-step physics state;
  obs_sim_states[t]  the state whose kinematics the policy's observation (cameras, eef) saw.
For sampled steps t of a Base arm (or a Dynamic arm before its event; post-event restores need the model edits):
  A. set_init_state(sim_states[t]) -> flattened state must equal sim_states[t]; step steps[t].action once ->
     compare with sim_states[t+1] (qpos/qvel max |d|)                          -> physics restore + replay;
  B. set_init_state(obs_sim_states[t]) -> eef_pos/eef_quat and every free body vs steps[t].obs_state, and, at
     policy queries, sha256 of each camera frame vs steps[t].policy_query.frame_sha  -> view re-render.
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

VIEW_KEYS = ("eef_pos", "eef_quat")


def _view_diff(a: dict, b: dict) -> dict:
    out = {k: float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k])))) for k in VIEW_KEYS}
    sa, sb = a.get("_gt_scene_object_pos") or {}, b.get("_gt_scene_object_pos") or {}
    worst, name = 0.0, None
    for n in set(sa) & set(sb):
        d = float(np.max(np.abs(np.asarray(sa[n]) - np.asarray(sb[n]))))
        if d > worst:
            worst, name = d, n
    out["scene_pos"], out["scene_worst_body"] = worst, name
    out["scene_missing"] = sorted(set(sa) ^ set(sb))
    return out


def _num(d: dict) -> float:
    return max(v for k, v in d.items() if isinstance(v, float))


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
        states, obs_states = z["sim_states"], z["obs_sim_states"]
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
        sim = raw.env.sim
        nq = sim.model.nq
        try:
            env.reset()
            env.set_init_state(init_states[case["init_state_index"]])
            per = []
            for t in ts:
                # A. physics state + one-step replay
                env.set_init_state(states[t])
                got = np.asarray(sim.get_state().flatten())
                phys = float(np.max(np.abs(got - states[t])))
                step = None
                if steps[t]["action"] is not None and t + 1 < len(states) and t + 1 <= t_hi:
                    env.step(steps[t]["action"])         # rec.act(t, a) is logged before the step from state t
                    nxt = np.asarray(sim.get_state().flatten())
                    step = {"qpos": float(np.max(np.abs(nxt[1:1 + nq] - states[t + 1][1:1 + nq]))),
                            "qvel": float(np.max(np.abs(nxt[1 + nq:] - states[t + 1][1 + nq:])))}
                # B. the policy's view
                obs = env.set_init_state(obs_states[t])
                view = _view_diff(rec._state(obs), steps[t]["obs_state"])
                q = steps[t].get("policy_query")
                if q and q.get("frame_sha"):
                    view["frame_sha_match"] = {cn: lmax_obs._sha(lmax_obs._flip(obs[c])) == q["frame_sha"].get(cn)
                                               for c, cn in zip(lmax_obs.CAMS, lmax_obs.CAM_NAMES)}
                per.append({"t": t, "phys_restore": phys, "one_step": step, "view": view})
        finally:
            env.close()
        worst_p = max(p["phys_restore"] for p in per)
        worst_s = max((max(p["one_step"].values()) for p in per if p["one_step"]), default=None)
        worst_v = max(_num(p["view"]) for p in per)
        sha = [v for p in per for v in (p["view"].get("frame_sha_match") or {}).values()]
        report.append({"case_id": case["case_id"], "arm": row["case"]["arm"], "t_checked": ts,
                       "worst_phys_restore": worst_p, "worst_one_step": worst_s, "worst_view": worst_v,
                       "frames_sha_matched": "%d/%d" % (sum(sha), len(sha)), "per_t": per})
        print("%s %s phys %.3g  one-step %s  view %.3g  frame sha %d/%d" % (
            case["case_id"], row["case"]["arm"], worst_p, "n/a" if worst_s is None else "%.3g" % worst_s,
            worst_v, sum(sha), len(sha)))
    def worst(k):
        return max((r[k] for r in report if r[k] is not None), default=None)
    out = {"run_root": str(args.run_root), "rows_checked": len(report), "per_row": report,
           "worst_phys_restore": worst("worst_phys_restore"), "worst_one_step": worst("worst_one_step"),
           "worst_view": worst("worst_view"),
           "frames_sha_all_match": all(r["frames_sha_matched"].split("/")[0] == r["frames_sha_matched"].split("/")[1]
                                       for r in report)}
    (args.run_root / "restore_check.json").write_text(json.dumps(out, indent=2) + "\n")
    print("worst phys %s, one-step %s, view %s, all frame sha match %s" % (
        out["worst_phys_restore"], out["worst_one_step"], out["worst_view"], out["frames_sha_all_match"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
