"""R-054 dataset gates (docs/DATA_MINER_SPEC.html §5). Writes <records>/gates.json.

  # CPU: G1 (records + exported dataset), G4, spot check, G7 videos
  .venvs/groot/bin/python experiments/r054_gates.py --records runs/r054 --dataset data/r054_full
  # GPU (flock, announced): G6 open-loop replay, and the forward-0 re-splice check
  PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl \
  flock /tmp/vla_gpu.lock .venvs/libero-plus/bin/python experiments/r054_gates.py --records runs/r054 \
      --dataset data/r054_full --g6 --f0-check

--dataset must be an UNTRUNCATED export (G6 replays whole episodes to their
outcome); the 48-step training export is checked by G1 only (--dataset-train).

  G1   records: check_record on every episode (exportable or not); dataset:
       feature keys/dtypes/shapes equal the reference's, fps, task text in the
       reference's task set, state finite, action dims 0-5 in [-1, 1],
       gripper in {0, 1}, images 256x256x3 uint8, provenance complete and
       privileged=False, stats.json equal to the checkpoint's.
  G4   no exported episode's (scene, dir seed) is in the held-out/val lists.
  spot exported state/action bit-equal to the adapter's mapping of the ROLLOUT
       TRACE (rollouts.jsonl, independent of the record json) on 3 episodes.
  G6   10 exported episodes: reset the scene with the recorded seed and spec,
       check the initial joint state equals the record's, execute the exported
       actions open-loop (inverse adapter), compare the outcome.
  G7   10 random successes as mp4 (vla_harness/video.py) for a person to watch.
  f0   R-054's uninterpretability rule: the recorded forward-0 N chunk vs a
       fresh splice forward from the same start, on 3 episodes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.getcwd())

import numpy as np

from vla_harness.data.contract import EpisodeStore, HeldOut, check_record
from vla_harness.data.export_lerobot import LiberoIpecAdapter, checkpoint_stats, reference_info

REF = "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot"
CKPT = "nvidia/gr00t17-lerobot-libero_spatial-640"
LISTS = ["experiments/repro/r056_heldout.json", "experiments/repro/r056_val.json"]


def load_dataset(root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    meta = json.load(open(os.path.join(root, "meta", "export.json")))
    return LeRobotDataset(meta["repo_id"], root=root, video_backend="pyav"), meta


def provenance_rows(root):
    return [json.loads(l) for l in open(os.path.join(root, "meta", "provenance.jsonl")) if l.strip()]


# --- G1 ------------------------------------------------------------------------
def g1_records(store: EpisodeStore) -> dict:
    bad = {}
    n = 0
    for rec in store.records():
        n += 1
        e = check_record(rec, store, cameras=("image", "image2"), action_dim=7,
                         proprio_keys=("eef_pos", "eef_quat", "gripper_qpos"))
        if e:
            bad[rec.episode_id] = e[:5]
    return {"n": n, "failed": len(bad), "examples": dict(list(bad.items())[:5]), "pass": n > 0 and not bad}


def g1_dataset(root: str, reference=REF, stats_from=CKPT, decode_images=True) -> dict:
    ds, meta = load_dataset(root)
    ref = reference_info(reference)
    errs = []
    ad = LiberoIpecAdapter()
    for k in list(ad.camera_map.values()) + [ad.state_key, ad.action_key]:
        a, b = dict(ds.meta.features[k]), dict(ref["features"][k])
        for x in (a, b):
            x.pop("info", None); x["shape"] = list(x["shape"])
        if a != b:
            errs.append(f"feature {k}: {a} != reference {b}")
    if ds.fps != ref["fps"]:
        errs.append(f"fps {ds.fps} != {ref['fps']}")
    hf = ds.hf_dataset.with_format(None)
    st = np.asarray(hf["observation.state"], np.float32); ac = np.asarray(hf["action"], np.float32)
    if st.shape[1:] != (8,) or st.dtype != np.float32 or not np.isfinite(st).all():
        errs.append(f"state {st.shape} {st.dtype} finite={np.isfinite(st).all()}")
    if ac.shape[1:] != (7,) or np.abs(ac[:, :6]).max() > 1.0 or not set(np.unique(ac[:, 6])) <= {0.0, 1.0}:
        errs.append(f"action {ac.shape}: |dims0-5| max {np.abs(ac[:, :6]).max():.3f}, gripper {sorted(set(np.unique(ac[:, 6])))[:5]}")
    tasks = set(ds.meta.tasks.index)
    unknown = sorted(tasks - set(ref["_tasks"]))
    if unknown:
        errs.append(f"task text not in reference: {unknown}")
    prov = provenance_rows(root)
    if len(prov) != ds.meta.total_episodes:
        errs.append(f"provenance rows {len(prov)} != episodes {ds.meta.total_episodes}")
    from vla_harness.data.contract import Provenance
    for p in prov:
        pv = Provenance(**p["provenance"])
        if pv.missing() or pv.privileged:
            errs.append(f"episode {p['episode_index']}: provenance missing {pv.missing()} privileged={pv.privileged}")
    if decode_images:
        # every frame decodes to 256x256x3 in [0,1]; sampled if large
        idx = range(len(ds)) if len(ds) <= 6000 else sorted(random.Random(0).sample(range(len(ds)), 6000))
        for i in idx:
            x = ds[i]
            for k in ad.camera_map.values():
                im = x[k]
                if tuple(im.shape) != (3, 256, 256) or float(im.min()) < 0 or float(im.max()) > 1:
                    errs.append(f"frame {i} {k}: shape {tuple(im.shape)} range {float(im.min()):.2f}..{float(im.max()):.2f}")
                    break
    stats = json.load(open(os.path.join(root, "meta", "stats.json")))
    ck, _ = checkpoint_stats(stats_from)
    if stats != ck:
        errs.append("meta/stats.json differs from the checkpoint's processor statistics")
    return {"dataset": root, "episodes": ds.meta.total_episodes, "frames": len(ds),
            "truncate_steps": meta.get("truncate_steps"), "errors": errs[:20], "pass": not errs}


# --- G4 ------------------------------------------------------------------------
def g4(store: EpisodeStore, dataset_roots) -> dict:
    h = HeldOut([p for p in LISTS if os.path.exists(p)])
    rec_v = h.violations(store.rows())
    ds_v = []
    for root in dataset_roots:
        for p in provenance_rows(root):
            ds_v += h.violations([{"episode_id": p["episode_id"], "scene_task_id": p["provenance"]["scene_task_id"],
                                   "dir_seed": p["provenance"]["dir_seed"]}])
    return {"lists": h.paths, "keys": len(h.keys), "record_violations": rec_v, "dataset_violations": ds_v,
            "pass": bool(h.keys) and not rec_v and not ds_v}


# --- spot check against the rollout trace ------------------------------------
def spot_check(store: EpisodeStore, root: str, n=3, seed=0) -> dict:
    from vla_harness.schema import TraceStore
    ds, meta = load_dataset(root)
    trunc = meta.get("truncate_steps")
    traces = TraceStore(os.path.dirname(os.path.abspath(store.root)), os.path.basename(os.path.abspath(store.root))).by_id()
    prov = provenance_rows(root)
    pick = random.Random(seed).sample(prov, min(n, len(prov)))
    ad = LiberoIpecAdapter()
    hf = ds.hf_dataset.with_format(None)
    out = []
    for p in pick:
        r = traces.get(p["rollout_id"])
        if r is None:
            out.append({"episode_id": p["episode_id"], "error": "rollout trace not found"}); continue
        ep = ds.meta.episodes[p["episode_index"]]
        rows = range(ep["dataset_from_index"], ep["dataset_to_index"])
        steps = [s for s in r.steps if s.action is not None]
        if trunc:
            steps = steps[:trunc]
        ok_s = ok_a = len(steps) == len(rows)
        for i, s in zip(rows, steps):
            prop = {k: s.obs_state[k] for k in ("eef_pos", "eef_quat", "gripper_qpos")}
            ok_s &= np.array_equal(np.asarray(hf[i]["observation.state"], np.float32), ad.state(prop))
            ok_a &= np.array_equal(np.asarray(hf[i]["action"], np.float32), ad.action(s.action))
        out.append({"episode_id": p["episode_id"], "frames": len(rows), "state_bit_equal": bool(ok_s),
                    "action_bit_equal": bool(ok_a)})
    return {"episodes": out, "pass": bool(out) and all(o.get("state_bit_equal") and o.get("action_bit_equal") for o in out)}


# --- G7 --------------------------------------------------------------------------
def g7_videos(store: EpisodeStore, out_dir: str, n=10, seed=0) -> dict:
    from PIL import Image
    from vla_harness.video import EpisodeVideo
    recs = [r["episode_id"] for r in store.rows() if r["exportable"]]
    pick = random.Random(seed).sample(recs, min(n, len(recs)))
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for eid in pick:
        rec = store.load(eid)
        v = EpisodeVideo(os.path.join(out_dir, f"{eid}.mp4"))
        for f in rec.frames:
            v.add({cam: np.asarray(Image.open(store.abspath(p)).convert("RGB")) for cam, p in f.images.items()})
        m = v.close()
        made.append({"episode_id": eid, "video": m and m["path"], "n_frames": m and m["n_frames"],
                     "scene": rec.provenance.scene_task_id, "radius": rec.provenance.magnitude})
    return {"videos": made, "note": "a person watches these; the gate is signed off by hand", "pass": None}


# --- G6 and the forward-0 check (GPU) -------------------------------------------
def make_env(task_id):
    from vla_harness.envs.libero_env import LiberoEnv
    return LiberoEnv(suite="libero_spatial", task_id=task_id, libero_plus=True,
                     libero_plus_base_instruction=True, obs_size=360)


def spec_of(prov):
    from vla_harness.schema import PerturbationSpec
    return PerturbationSpec.of(joint_radius_rad=float(prov["magnitude"]), joint_dir_seed=int(prov["dir_seed"]))


def g6_replay(store: EpisodeStore, root: str, n=10, seed=0) -> dict:
    from vla_harness.schema import Action
    ds, meta = load_dataset(root)
    if meta.get("truncate_steps"):
        return {"pass": False, "error": "G6 needs an untruncated export"}
    hf = ds.hf_dataset.with_format(None)
    prov = provenance_rows(root)
    ad = LiberoIpecAdapter()
    out = []
    for p in random.Random(seed).sample(prov, min(n, len(prov))):
        rec = store.load(p["episode_id"])
        pv = p["provenance"]
        env = make_env(pv["scene_task_id"])
        obs = env.reset(pv["env_seed"], spec_of(pv))
        init_ok = np.allclose(obs.state["joint_pos"], rec.frames[0].proprio["joint_pos"], atol=1e-9)
        ep = ds.meta.episodes[p["episode_index"]]
        success, t = False, 0
        for i in range(ep["dataset_from_index"], ep["dataset_to_index"]):
            obs, success, done, _ = env.step(Action([float(x) for x in ad.inverse_action(hf[i]["action"])],
                                                    list(env.action_dims)))
            t += 1
            if done:
                break
        out.append({"episode_id": p["episode_id"], "init_state_equal": bool(init_ok), "source_success": rec.outcome["success"],
                    "replay_success": bool(success), "replay_steps": t, "source_steps": rec.outcome.get("steps"),
                    "same_outcome": bool(success) == bool(rec.outcome["success"])})
        del env
    k = sum(o["same_outcome"] for o in out)
    return {"episodes": out, "same_outcome": k, "n": len(out), "pass": k >= 9,
            "expectation": "R-054 E3: >= 9/10 reach the same outcome"}


def f0_check(store: EpisodeStore, n=3, seed=0, tol=1e-3) -> dict:
    """Re-run forward 0 of `n` minted episodes with a fresh splice and compare the N chunk."""
    import torch
    from experiments.r054_mint import build_policy
    from vla_harness.capture.splice import attach_splice
    from vla_harness.data.sources.rescue import noise_key_for
    import vla_harness.data.sources.rescue as rescue
    pol = build_policy(); pol.reset()
    minter = rescue.RescueMinter(pol, make_env, store, trace_store=False)
    rows = [r for r in store.rows()]
    out = []
    for r in random.Random(seed).sample(rows, min(n, len(rows))):
        rec = store.load(r["episode_id"])
        pv = rec.provenance
        stored = np.load(os.path.join(store.root, "splice", f"{rec.episode_id}.npz"))
        recs = minter.nominal_features(pv.scene_task_id, pv.noise_seed, pv.env_seed)
        env = make_env(pv.scene_task_id)
        h = attach_splice(pol._policy, source=lambda i, recs=recs: recs[i] if i < len(recs) else None,
                          drive="N", noise_key=noise_key_for(pv.noise_seed), drive_until=1)
        try:
            obs = env.reset(pv.env_seed, spec_of({"magnitude": pv.magnitude, "dir_seed": pv.dir_seed}))
            pol.reset()
            pol(obs.policy_view())
        finally:
            h.detach()
        fresh = h.records[0]["action"]["N"][0, :, :7].float().cpu().numpy()
        d = float(np.abs(fresh - stored["act_N_f0"]).max())
        out.append({"episode_id": rec.episode_id, "max_abs_diff": d, "pass": d <= tol})
        del env; torch.cuda.empty_cache()
    return {"episodes": out, "tol": tol, "pass": bool(out) and all(o["pass"] for o in out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="runs/r054")
    ap.add_argument("--dataset", required=True, help="untruncated export (G1, G4, spot, G6)")
    ap.add_argument("--dataset-train", default=None, help="the truncated training export (G1, G4, spot)")
    ap.add_argument("--g6", action="store_true", help="GPU/EGL: open-loop replay")
    ap.add_argument("--f0-check", action="store_true", help="GPU: forward-0 re-splice on 3 episodes")
    ap.add_argument("--no-videos", action="store_true")
    ap.add_argument("--no-decode", action="store_true", help="G1 without decoding every video frame")
    a = ap.parse_args()
    store = EpisodeStore(a.records)
    out_path = os.path.join(a.records, "gates.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    if not (a.g6 or a.f0_check):
        res["G1_records"] = g1_records(store)
        res["G1_dataset"] = g1_dataset(a.dataset, decode_images=not a.no_decode)
        roots = [a.dataset]
        res["spot"] = spot_check(store, a.dataset)
        if a.dataset_train:
            res["G1_dataset_train"] = g1_dataset(a.dataset_train, decode_images=not a.no_decode)
            res["spot_train"] = spot_check(store, a.dataset_train)
            roots.append(a.dataset_train)
        res["G4"] = g4(store, roots)
        if not a.no_videos:
            res["G7"] = g7_videos(store, os.path.join(a.records, "g7_videos"))
    if a.g6:
        res["G6"] = g6_replay(store, a.dataset)
    if a.f0_check:
        res["f0_check"] = f0_check(store)
    json.dump(res, open(out_path, "w"), indent=1, default=float)
    for k, v in res.items():
        print(f"{k:18s} pass={v.get('pass')}  " + json.dumps({x: y for x, y in v.items() if x in ("n", "failed", "errors", "same_outcome", "episodes") and x != "episodes" or x == "errors"})[:300])
    print("wrote", out_path)


if __name__ == "__main__":
    main()
