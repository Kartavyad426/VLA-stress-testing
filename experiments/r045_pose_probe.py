"""R-045: linear pose probe on the VLM output across start-pose radius.

  PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config \
  MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  flock /tmp/vla_gpu.lock .venvs/libero-plus/bin/python experiments/r045_pose_probe.py --out runs/r045

Phase 1 (GPU): 10 scenes x 12 radii x 3 directions = 360 resets, one
backbone forward each; saves the adapter-output image tokens, the state
token, the true eef pose, and the wrist-block position check.
Phase 2 (CPU): ridge decoders per feature set, split by direction, RMSE per
radius. Writes runs/r045/features.npz, probe.json.
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
import torch

from vla_harness.envs.libero_env import LiberoEnv
from vla_harness.policies.lerobot_policy import LeRobotPolicy
from vla_harness.schema import PerturbationSpec

SCENES = [984, 1030, 1062, 1090, 1132, 1169, 1201, 1247, 1282, 1327]
RADII = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
DIRS = [0, 1, 2]


def build_policy():
    from lerobot.envs.configs import LiberoPlusEnv
    return LeRobotPolicy("nvidia/gr00t17-lerobot-libero_spatial-640", n_action_steps=16,
                         env_cfg=LiberoPlusEnv(task="libero_spatial"),
                         policy_overrides={"base_model_path": "nvidia/GR00T-N1.7-3B", "embodiment_tag": "libero_sim"},
                         dtype="bfloat16", rename_map={"observation.images.image2": "observation.images.wrist_image"})


def capture(a):
    pol = build_policy(); pol.reset()
    t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:6.0f}s]", *x, flush=True)
    X_img, X_state, Y_pos, Y_quat, Y_joint, meta = [], [], [], [], [], []
    wrist_positions = None
    for scene in SCENES:
        env = LiberoEnv(suite="libero_spatial", task_id=scene, libero_plus=True,
                        libero_plus_base_instruction=True, obs_size=360)
        for d in DIRS:
            for r in RADII:
                spec = PerturbationSpec() if r == 0 else PerturbationSpec.of(joint_radius_rad=r, joint_dir_seed=d)
                obs = env.reset(0, spec)
                f = pol.features_for(obs)
                img = f["backbone_features"][0][f["image_mask"][0]].float().cpu().numpy()      # (128, 2048)
                X_img.append(img.astype(np.float32))
                X_state.append(f["state_features"][0, 0].float().cpu().numpy().astype(np.float32))
                Y_pos.append(np.array(obs.state["eef_pos"], np.float64))
                Y_quat.append(np.array(obs.state["eef_quat"], np.float64))
                Y_joint.append(np.array(obs.state["joint_pos"], np.float64))
                meta.append({"scene": scene, "radius": r, "dir": d})
                if wrist_positions is None:
                    # which image positions belong to the wrist? swap the wrist frame for the
                    # agent frame and see which token rows change (intervention, not assumption)
                    from copy import copy
                    o2 = copy(obs); fr = dict(obs.frames); fr["image2"] = fr["image"]; o2.frames = fr
                    g = pol.features_for(o2)["backbone_features"][0][f["image_mask"][0]].float().cpu().numpy()
                    changed = np.abs(g - img).max(axis=1) > 1e-3
                    wrist_positions = np.flatnonzero(changed).tolist()
                    log(f"wrist token positions by intervention: n={len(wrist_positions)} "
                        f"range {min(wrist_positions)}-{max(wrist_positions)} (of {img.shape[0]})")
            log(f"scene {scene} dir {d} done ({len(meta)} samples)")
        del env
    np.savez_compressed(os.path.join(a.out, "features.npz"), X_img=np.stack(X_img), X_state=np.stack(X_state),
                        Y_pos=np.stack(Y_pos), Y_quat=np.stack(Y_quat), Y_joint=np.stack(Y_joint),
                        wrist_positions=np.array(wrist_positions), meta=json.dumps(meta))
    log("features saved")


def ridge_fit_predict(Xtr, Ytr, Xte, lam):
    """Dual-form ridge (n << d): W = X^T (X X^T + lam I)^-1 Y, with centring."""
    mx, my = Xtr.mean(0), Ytr.mean(0)
    A, B = Xtr - mx, Ytr - my
    K = A @ A.T
    alpha = np.linalg.solve(K + lam * np.eye(len(K)), B)
    return my + (Xte - mx) @ (A.T @ alpha)


def probe(a):
    z = np.load(os.path.join(a.out, "features.npz"))
    meta = json.loads(str(z["meta"]))
    rad = np.array([m["radius"] for m in meta]); dr = np.array([m["dir"] for m in meta])
    wp = set(z["wrist_positions"].tolist())
    img = z["X_img"]                                  # (n, 128, 2048)
    agent_idx = [i for i in range(img.shape[1]) if i not in wp]; wrist_idx = sorted(wp)
    feats = {
        "image_all_flat": img.reshape(len(img), -1),
        "agent_block_flat": img[:, agent_idx].reshape(len(img), -1),
        "wrist_block_flat": img[:, wrist_idx].reshape(len(img), -1),
        "image_mean_pool": img.mean(1),
        "state_token": z["X_state"],
    }
    Y = z["Y_pos"]
    tr, te = dr != 2, dr == 2
    rng = np.random.default_rng(0)
    out = {"n": int(len(Y)), "n_train": int(tr.sum()), "n_test": int(te.sum()), "wrist_positions": sorted(wp),
           "label_spread_rmse": float(np.sqrt(((Y[te] - Y[tr].mean(0)) ** 2).sum(1).mean())), "per_feature": {}}
    for name, X in feats.items():
        best = None
        for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4):
            # lambda chosen on direction 1 held out from direction 0 (never on the test direction)
            tr0, va = dr == 0, dr == 1
            p = ridge_fit_predict(X[tr0], Y[tr0], X[va], lam)
            err = np.sqrt(((p - Y[va]) ** 2).sum(1).mean())
            if best is None or err < best[1]:
                best = (lam, err)
        lam = best[0]
        pred = ridge_fit_predict(X[tr], Y[tr], X[te], lam)
        Yte, rad_te = Y[te], rad[te]
        per_r = {}
        for r in RADII:
            m = rad_te == r
            per_r[str(r)] = float(np.sqrt(((pred[m] - Yte[m]) ** 2).sum(1).mean()))
        Ysh = Y[tr][rng.permutation(tr.sum())]
        pnull = ridge_fit_predict(X[tr], Ysh, X[te], lam)
        out["per_feature"][name] = {"lambda": lam, "rmse_by_radius_m": per_r,
                                    "rmse_all_m": float(np.sqrt(((pred - Yte) ** 2).sum(1).mean())),
                                    "rmse_r0_m": per_r["0.0"], "rmse_r0p5_m": per_r["0.5"],
                                    "ratio_r0p5_over_r0": per_r["0.5"] / max(per_r["0.0"], 1e-9),
                                    "null_rmse_m": float(np.sqrt(((pnull - Yte) ** 2).sum(1).mean()))}
        print(f"{name:18s} lam={lam:<7g} rmse r0 {per_r['0.0']*100:5.2f} cm  r0.5 {per_r['0.5']*100:5.2f} cm  "
              f"ratio {out['per_feature'][name]['ratio_r0p5_over_r0']:.2f}  all {out['per_feature'][name]['rmse_all_m']*100:5.2f} cm  "
              f"null {out['per_feature'][name]['null_rmse_m']*100:5.2f} cm")
    json.dump(out, open(os.path.join(a.out, "probe.json"), "w"), indent=1)
    print("label spread rmse %.2f cm" % (out["label_spread_rmse"] * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/r045")
    ap.add_argument("--phase", default="both", choices=["capture", "probe", "both"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.phase in ("capture", "both"):
        capture(a)
    if a.phase in ("probe", "both"):
        probe(a)


if __name__ == "__main__" and "--scene-split" not in sys.argv:
    main()


def probe_scene_split(a):
    """POST-HOC analysis added 2026-09-24 after the pre-registered direction
    split failed its own sanity gate (the state token does not linearly
    extrapolate to an unseen joint direction any better than the image
    tokens do). Train on 8 scenes with every direction and radius, test on
    2 held-out scenes: does the pose read transfer across scenes, and is it
    flat in radius when the pose range is covered?"""
    z = np.load(os.path.join(a.out, "features.npz"))
    meta = json.loads(str(z["meta"]))
    rad = np.array([m["radius"] for m in meta]); sc = np.array([m["scene"] for m in meta])
    img = z["X_img"]; Y = z["Y_pos"]
    feats = {"image_all_flat": img.reshape(len(img), -1), "image_mean_pool": img.mean(1), "state_token": z["X_state"]}
    out = {"design": "train 8 scenes x 3 directions x 12 radii, test 2 held-out scenes; ridge lam=1; eef position RMSE (m) per radius",
           "holds": {}}
    for hold in ((1062, 1247), (984, 1327), (1030, 1201), (1090, 1132), (1169, 1282)):
        tr, te = ~np.isin(sc, hold), np.isin(sc, hold)
        res = {}
        for n, X in feats.items():
            p = ridge_fit_predict(X[tr], Y[tr], X[te], 1.0)
            res[n] = {str(r): float(np.sqrt(((p[rad[te] == r] - Y[te][rad[te] == r]) ** 2).sum(1).mean())) for r in RADII}
        out["holds"][f"{hold[0]}_{hold[1]}"] = res
    # summary: median over holds of RMSE at r=0 and r=0.5, per feature
    out["summary"] = {n: {"rmse_r0_m": float(np.median([h[n]["0.0"] for h in out["holds"].values()])),
                          "rmse_r0p5_m": float(np.median([h[n]["0.5"] for h in out["holds"].values()])),
                          "rmse_all_radii_median_m": float(np.median([v for h in out["holds"].values() for v in h[n].values()]))}
                      for n in feats}
    json.dump(out, open(os.path.join(a.out, "probe_scene_split.json"), "w"), indent=1)
    for n, s in out["summary"].items():
        print(f"scene-split {n:16s} r0 {s['rmse_r0_m']*100:4.1f} cm  r0.5 {s['rmse_r0p5_m']*100:4.1f} cm  all-radii median {s['rmse_all_radii_median_m']*100:4.1f} cm")


if __name__ == "__main__" and "--scene-split" in sys.argv:
    class _A: out = "runs/r045"
    probe_scene_split(_A())
