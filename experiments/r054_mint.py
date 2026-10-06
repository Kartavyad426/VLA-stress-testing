"""R-054 runner: rescue minting pilot (start pose), resumable, G4-guarded.

  # 1. the held-out and validation lists (CPU; from runs/r047/score.json)
  python experiments/r054_mint.py --write-lists
  # 2. minting (GPU, under the flock, announced)
  PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config \
  MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  flock /tmp/vla_gpu.lock .venvs/libero-plus/bin/python experiments/r054_mint.py --out runs/r054 [--target 100]

Sampling (RESULTS.md R-054), a deterministic schedule over attempt index i:
  scene     = SCENES[i % 9]                (R-047's ten minus on-ramekin 1169)
  band      = BANDS[(i // 9) % 4]          (0.1-0.2, 0.2-0.3, 0.3-0.4, 0.4-0.5 rad)
  noise     = NOISE[(i // 36) % len(NOISE)]
  dir seed  = 1000 + i                     (unique per attempt)
  radius    = band_lo + 0.1 * U,  U ~ Uniform[0,1) from default_rng([54, i])
so every 36 attempts cover scene x band once (balanced attempts, not balanced
successes). Env seed 0. Runs until --target successes exist in the manifest;
a rerun skips attempts already recorded. Any (scene, dir seed) in the held-out
or validation list is refused (G4) -- the schedule never produces one, the
guard makes sure of it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.getcwd())

import numpy as np

SCENES_ALL = [984, 1030, 1062, 1090, 1132, 1169, 1201, 1247, 1282, 1327]
ON_RAMEKIN = 1169                         # fails unperturbed (R-047)
SCENES = [s for s in SCENES_ALL if s != ON_RAMEKIN]
BANDS = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
AXIS = "joint_radius_rad"
ENV_SEED = 0
DIR_SEED_BASE = 1000
HELDOUT = "experiments/repro/r056_heldout.json"
VAL = "experiments/repro/r056_val.json"
R_MAX = 0.5


def attempt(i: int, noise_seeds) -> dict:
    scene = SCENES[i % len(SCENES)]
    b = (i // len(SCENES)) % len(BANDS)
    lo, hi = BANDS[b]
    u = float(np.random.default_rng([54, i]).random())
    return {"i": i, "scene": scene, "band": f"{lo:.1f}-{hi:.1f}", "radius": round(lo + (hi - lo) * u, 6),
            "dir_seed": DIR_SEED_BASE + i,
            "noise_seed": noise_seeds[(i // (len(SCENES) * len(BANDS))) % len(noise_seeds)],
            "instance_id": f"r054_a{i:05d}"}


# --- held-out / validation lists -------------------------------------------
def heldout_radii(x50: float | None) -> list[float]:
    """Two radii just beyond the scene's R-047 50% point, clipped to 0.5.

    r1 = x50 + 0.025, r2 = x50 + 0.075 (a quarter and three quarters of R-047's
    0.1 grid step past the midpoint), each clipped to R_MAX. When clipping
    collapses both onto 0.5 (x50 >= 0.475), the pair becomes (0.45, 0.5) so the
    two radii stay distinct; that case is flagged in the list."""
    if x50 is None or not math.isfinite(x50) or x50 + 0.025 >= R_MAX:
        return [R_MAX - 0.05, R_MAX]
    return [round(min(x50 + 0.025, R_MAX), 4), round(min(x50 + 0.075, R_MAX), 4)]


def write_lists(score_path="runs/r047/score.json"):
    res = json.load(open(score_path))[AXIS]["res"]
    held, radii = [], {}
    for s in SCENES:
        x50 = res[str(s)]["x50"]
        radii[s] = heldout_radii(x50)
        for d in (100, 101):
            for r in radii[s]:
                for ns in (0, 1, 2):
                    held.append({"scene_task_id": s, "dir_seed": d, "radius": r, "noise_seed": ns,
                                 "env_seed": ENV_SEED, "r047_x50": x50,
                                 "clipped": bool(x50 is None or x50 + 0.025 >= R_MAX)})
    val = []
    for k in range(10):
        s = SCENES[k % len(SCENES)]
        val.append({"scene_task_id": s, "dir_seed": 200 + k, "radius": radii[s][k % 2], "noise_seed": 0,
                    "env_seed": ENV_SEED})
    common = {"suite": "libero_spatial", "axis": AXIS, "source": score_path,
              "rule": "radii = x50+0.025, x50+0.075 clipped to 0.5; (0.45, 0.5) when both clip"}
    json.dump({**common, "note": "R-056 held-out: 9 scenes x 2 radii x dir seeds 100,101 x noise 0-2 = 108",
               "n": len(held), "instances": held}, open(HELDOUT, "w"), indent=1)
    json.dump({**common, "note": "R-056 validation: 10 starts, dir seeds 200-209, noise 0 (checkpoint selection)",
               "n": len(val), "instances": val}, open(VAL, "w"), indent=1)
    print(f"wrote {HELDOUT} ({len(held)}) and {VAL} ({len(val)})")
    for s in SCENES:
        print(f"  scene {s}: x50 {res[str(s)]['x50']} -> radii {radii[s]}")


# --- minting -------------------------------------------------------------------
def build_policy():
    from lerobot.envs.configs import LiberoPlusEnv
    from vla_harness.policies.lerobot_policy import LeRobotPolicy
    return LeRobotPolicy("nvidia/gr00t17-lerobot-libero_spatial-640", n_action_steps=16,
                         env_cfg=LiberoPlusEnv(task="libero_spatial"),
                         policy_overrides={"base_model_path": "nvidia/GR00T-N1.7-3B", "embodiment_tag": "libero_sim"},
                         dtype="bfloat16", rename_map={"observation.images.image2": "observation.images.wrist_image"})


def make_env(task_id):
    from vla_harness.envs.libero_env import LiberoEnv
    return LiberoEnv(suite="libero_spatial", task_id=task_id, libero_plus=True,
                     libero_plus_base_instruction=True, obs_size=360)


def yield_table(rows) -> dict:
    out = {}
    for lo, hi in BANDS:
        b = f"{lo:.1f}-{hi:.1f}"
        g = [r for r in rows if r["band"] == b]
        k = sum(r["success"] for r in g)
        out[b] = {"attempts": len(g), "successes": k, "yield": (k / len(g)) if g else None}
    k = sum(r["success"] for r in rows)
    out["pooled"] = {"attempts": len(rows), "successes": k, "yield": (k / len(rows)) if rows else None}
    return out


def print_yield(rows):
    for b, v in yield_table(rows).items():
        y = "-" if v["yield"] is None else f"{v['yield']:.2f}"
        print(f"  {b:8s} {v['successes']:4d}/{v['attempts']:<4d} yield {y}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-lists", action="store_true")
    ap.add_argument("--out", default="runs/r054")
    ap.add_argument("--target", type=int, default=100, help="stop at this many successes")
    ap.add_argument("--max-attempts", type=int, default=400)
    ap.add_argument("--start-attempt", type=int, default=0,
                    help="first schedule index; an extension store continues after the pilot's attempts "
                         "(same schedule, continuing dir seeds 1000 + i)")
    ap.add_argument("--noise-seeds", default="0,1,2")
    ap.add_argument("--max-new", type=int, default=None, help="smoke: stop after N new attempts")
    ap.add_argument("--scenes", default=None, help="smoke: restrict to these scenes (comma list)")
    a = ap.parse_args()
    if a.write_lists:
        write_lists(); return

    from vla_harness.data.contract import EpisodeStore, HeldOut, Provenance
    from vla_harness.data.sources.rescue import RescueMinter
    from vla_harness.schema import PerturbationSpec

    for p in (HELDOUT, VAL):
        if not os.path.exists(p):
            sys.exit(f"{p} missing: run --write-lists first (G4 needs it)")
    noise = [int(x) for x in a.noise_seeds.split(",")]
    only = {int(s) for s in a.scenes.split(",")} if a.scenes else None
    store = EpisodeStore(a.out)
    code_dir = os.path.join(a.out, "code_state")
    if not os.path.exists(code_dir):
        os.makedirs(code_dir)
        os.system(f"git rev-parse HEAD > {code_dir}/HEAD; git diff > {code_dir}/uncommitted.patch")
        json.dump(vars(a), open(os.path.join(code_dir, "args.json"), "w"), indent=1)
    heldout = HeldOut([HELDOUT, VAL])
    done = store.done_instances()
    rows = store.rows()
    t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:7.0f}s]", *x, flush=True)
    succ = sum(r["success"] for r in rows)
    log(f"{len(rows)} attempts on record, {succ} successes; target {a.target}")
    if succ >= a.target:
        print_yield(rows); return

    pol = build_policy(); pol.reset()
    minter = RescueMinter(pol, make_env, store, heldout=heldout, drive="N", drive_until=1)
    n_new = 0
    for i in range(a.start_attempt, a.start_attempt + a.max_attempts):
        at = attempt(i, noise)
        if at["instance_id"] in done or (only and at["scene"] not in only):
            continue
        prov = Provenance(source="rescue", policy_id=pol.policy_id, axis=AXIS, magnitude=at["radius"],
                          scene_task_id=at["scene"], suite="libero_spatial", env_id="",
                          env_seed=ENV_SEED, noise_seed=at["noise_seed"], privileged=False,
                          dir_seed=at["dir_seed"], instance_id=at["instance_id"], band=at["band"],
                          extra={"drive": "N", "drive_until": 1, "attempt": i})
        spec = PerturbationSpec.of(joint_radius_rad=at["radius"], joint_dir_seed=at["dir_seed"])
        env_id = make_env(at["scene"]).env_id
        prov.env_id = env_id
        rec = minter.mint(at["instance_id"], at["scene"], spec, prov)
        n_new += 1; succ += rec.exportable
        log(f"a{i:04d} scene {at['scene']} band {at['band']} r={at['radius']:.3f} dir {at['dir_seed']} "
            f"noise {at['noise_seed']}: success={rec.outcome['success']} ca={rec.outcome['closest_approach_m']} "
            f"driven={sum(f.driven for f in rec.frames)} [{succ}/{a.target}]")
        if succ >= a.target or (a.max_new is not None and n_new >= a.max_new):
            break
    rows = store.rows()
    print_yield(rows)
    json.dump(yield_table(rows), open(os.path.join(a.out, "yield.json"), "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
