"""Harness episode records -> a LeRobot (v3.0) dataset in a reference dataset's schema.

  .venvs/groot/bin/python -m vla_harness.data.export_lerobot \
      --records runs/r054 --out data/r054_lerobot --repo-id local/r054_rescue \
      --reference IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
      --stats-from nvidia/gr00t17-lerobot-libero_spatial-640 [--truncate-steps 48]

Only records with exportable=True (successes) are written. The schema comes
from the reference dataset's meta/info.json: the same feature keys, dtypes,
shapes, names and fps, so a policy trained on the reference reads the export
unchanged and the two can be mixed in one batch.

Convention mapping lives in an ADAPTER (records are env-native, contract.py).
`LiberoIpecAdapter` is the LIBERO -> IPEC-COMMUNITY/libero_*_no_noops mapping
that GR00T N1.7's LIBERO checkpoints were trained on, each rule checked
against a real sample of that dataset:

  images  env raw render, flipped 180 degrees (LeRobot's LiberoProcessorStep,
          processor/env_processor.py:59 -- the dataset is stored upright), THEN
          resized to 256x256 with cv2 INTER_AREA. That is the order and the
          interpolation the checkpoint's own preprocessor applies to a 360
          render at eval (albumentations path, shortest edge 256), so the
          export matches what the policy saw pixel for pixel before encoding.
          camera "image" (agent view) -> observation.images.image,
          camera "image2" (wrist)     -> observation.images.wrist_image.
  state   8-d: eef pos (3) + axis-angle of the eef quaternion (3), computed with
          LiberoProcessorStep's own _quat2axisangle, + gripper qpos (2).
  action  7-d: dims 0-5 are the executed end-effector delta, unchanged.
          Gripper: the env executes -1 = open, +1 = close; the dataset stores
          1 = open, 0 = close (GR00T's postprocessor maps x -> -sign(2x-1),
          processor_groot.py:2512). Export writes (1 - g) / 2.

Normalisation statistics are NOT recomputed: meta/stats.json is overwritten
with the statistics the checkpoint's processor carries, and the checkpoint's
pipeline is what training loads anyway.

Extra columns: per frame `frame_driven` (int64: the action came from outside
the policy) and `frame_source_t` (int64: the env step in the source episode);
per episode meta/provenance.jsonl (episode_index -> the record's provenance
and outcome). Policies read neither: they are not observation.*/action keys.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from .contract import EpisodeRecord, EpisodeStore, HeldOut

EXTRA_FRAME_FEATURES = {
    "frame_driven": {"dtype": "int64", "shape": [1], "names": None},
    "frame_source_t": {"dtype": "int64", "shape": [1], "names": None},
}


class LiberoIpecAdapter:
    """LIBERO env-native record -> IPEC libero_*_no_noops frame."""

    name = "libero->ipec_no_noops/v1"
    camera_map = {"image": "observation.images.image", "image2": "observation.images.wrist_image"}
    state_key = "observation.state"
    action_key = "action"
    size = (256, 256)

    def describe(self) -> dict:
        return {"name": self.name, "camera_map": self.camera_map, "image": "flip180 then cv2.INTER_AREA to 256",
                "state": "eef_pos + quat2axisangle(eef_quat) [LiberoProcessorStep] + gripper_qpos",
                "action": "dims0-5 unchanged; gripper (1-g)/2 (env -1 open -> 1)"}

    def image(self, raw: np.ndarray) -> np.ndarray:
        import cv2
        img = np.ascontiguousarray(np.asarray(raw, dtype=np.uint8)[::-1, ::-1])
        h, w = self.size
        if img.shape[:2] != (h, w):
            # shortest-edge resize, as processor_groot's albumentations path does (square here)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return img

    @staticmethod
    def quat2axisangle(quat) -> np.ndarray:
        """LiberoProcessorStep._quat2axisangle, on CPU float32 -- the function the policy's
        state went through. Falls back to the same formula in numpy without lerobot."""
        q = np.asarray(quat, dtype=np.float32).reshape(1, 4)
        try:
            import torch
            from lerobot.processor.env_processor import LiberoProcessorStep
            return LiberoProcessorStep()._quat2axisangle(torch.from_numpy(q)).numpy()[0].astype(np.float32)
        except ImportError:
            w = np.clip(q[:, 3], -1.0, 1.0)
            den = np.sqrt(np.clip(1.0 - w * w, 0.0, None))
            angle = 2.0 * np.arccos(w)
            out = np.where(den[:, None] > 1e-10, q[:, :3] * (angle / np.where(den > 1e-10, den, 1.0))[:, None], 0.0)
            return out[0].astype(np.float32)

    def state(self, proprio: dict) -> np.ndarray:
        return np.concatenate([np.asarray(proprio["eef_pos"], np.float32),
                               self.quat2axisangle(proprio["eef_quat"]),
                               np.asarray(proprio["gripper_qpos"], np.float32)]).astype(np.float32)

    def action(self, env_action) -> np.ndarray:
        a = np.asarray(env_action, dtype=np.float32).copy()
        g = float(a[6])
        if g not in (-1.0, 1.0):
            raise ValueError(f"env gripper action {g} is not binarised (-1/+1); cannot map to the dataset's 0/1")
        a[6] = (1.0 - g) / 2.0
        return a

    def inverse_action(self, ds_action) -> np.ndarray:
        """Dataset action -> env action (G6 replay)."""
        a = np.asarray(ds_action, dtype=np.float32).copy()
        a[6] = -np.sign(2.0 * a[6] - 1.0)
        return a


ADAPTERS = {LiberoIpecAdapter.name: LiberoIpecAdapter}


# --- reference schema and checkpoint stats -----------------------------------
def reference_info(reference: str) -> dict:
    """meta/info.json of the reference dataset (a local root, or a repo id under HF_LEROBOT_HOME)."""
    p = Path(reference)
    if not (p / "meta" / "info.json").exists():
        from lerobot.utils.constants import HF_LEROBOT_HOME
        p = Path(HF_LEROBOT_HOME) / reference
    info = json.load(open(p / "meta" / "info.json"))
    tasks = []
    if (p / "meta" / "tasks.parquet").exists():
        import pandas as pd
        tasks = list(pd.read_parquet(p / "meta" / "tasks.parquet").index)     # v3.0: indexed by the task text
    info["_root"] = str(p)
    info["_tasks"] = [str(x) for x in tasks]
    return info


def checkpoint_stats(checkpoint: str) -> tuple[dict, str]:
    """The normalisation statistics the checkpoint's preprocessor carries, as {feature: {stat: list}}."""
    from huggingface_hub import snapshot_download
    from safetensors.numpy import load_file
    root = checkpoint if os.path.isdir(checkpoint) else snapshot_download(checkpoint, allow_patterns=["*.json", "*processor*"])
    cfg = json.load(open(os.path.join(root, "policy_preprocessor.json")))
    files = [s["state_file"] for s in cfg["steps"] if s.get("state_file")]
    if not files:
        raise RuntimeError(f"{checkpoint}: preprocessor carries no statistics")
    flat = load_file(os.path.join(root, files[0]))
    out: dict[str, dict] = {}
    for k, v in flat.items():
        feat, stat = k.rsplit(".", 1)
        out.setdefault(feat, {})[stat] = v.tolist()
    return out, os.path.join(root, files[0])


def features_from_reference(info: dict, adapter) -> dict:
    keys = list(adapter.camera_map.values()) + [adapter.state_key, adapter.action_key]
    feats = {}
    for k in keys:
        if k not in info["features"]:
            raise KeyError(f"reference dataset has no feature {k!r}")
        f = dict(info["features"][k])
        f.pop("info", None)                     # encoder info is regenerated by the writer
        feats[k] = f
    return {**feats, **EXTRA_FRAME_FEATURES}


# --- export ----------------------------------------------------------------------
def iter_records(roots) -> list[tuple[EpisodeStore, EpisodeRecord]]:
    out = []
    for root in roots:
        st = EpisodeStore(root)
        for rec in st.records(exportable_only=True):
            out.append((st, rec))
    return out


def frame_rows(store: EpisodeStore, rec: EpisodeRecord, adapter, truncate_steps: int | None, load_images=True):
    """Yield (frame dict for LeRobot, source frame) for every frame with an action, up to truncate."""
    from PIL import Image
    for f in rec.frames:
        if f.action is None:
            break                                       # terminal observation: no action, not a sample
        if truncate_steps is not None and f.t >= truncate_steps:
            break
        row = {adapter.state_key: adapter.state(f.proprio), adapter.action_key: adapter.action(f.action),
               "frame_driven": np.array([int(f.driven)], np.int64),
               "frame_source_t": np.array([int(f.t)], np.int64), "task": rec.instruction}
        if load_images:
            for cam, key in adapter.camera_map.items():
                row[key] = adapter.image(np.asarray(Image.open(store.abspath(f.images[cam])).convert("RGB")))
        yield row, f


def export(records_roots, out: str, repo_id: str, reference: str, stats_from: str,
           truncate_steps: int | None = None, adapter_name: str = LiberoIpecAdapter.name,
           heldout_lists=(), limit: int | None = None, overwrite: bool = False) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    adapter = ADAPTERS[adapter_name]()
    info = reference_info(reference)
    feats = features_from_reference(info, adapter)
    pairs = iter_records(records_roots)
    if limit:
        pairs = pairs[:limit]
    viol = HeldOut(heldout_lists).violations([r.summary() for _, r in pairs])
    if viol:
        raise RuntimeError(f"G4: {len(viol)} exportable episodes hit a held-out list, e.g. {viol[:3]}")
    if os.path.exists(out):
        if not overwrite:
            raise FileExistsError(f"{out} exists (pass --overwrite)")
        shutil.rmtree(out)
    ds = LeRobotDataset.create(repo_id=repo_id, fps=int(info["fps"]), features=feats, root=out,
                               robot_type=info.get("robot_type"), use_videos=True)
    prov_rows, unknown_tasks = [], set()
    for ep_idx, (store, rec) in enumerate(pairs):
        n = 0
        for row, _f in frame_rows(store, rec, adapter, truncate_steps):
            ds.add_frame(row); n += 1
        if n == 0:
            raise RuntimeError(f"{rec.episode_id}: no frames to export")
        ds.save_episode()
        if info["_tasks"] and rec.instruction not in info["_tasks"]:
            unknown_tasks.add(rec.instruction)
        prov_rows.append({"episode_index": ep_idx, "episode_id": rec.episode_id, "records_root": store.root,
                          "rollout_id": rec.rollout_id, "n_frames": n,
                          "n_driven": sum(1 for f in rec.frames[:n] if f.driven),
                          "provenance": rec.to_dict()["provenance"], "outcome": rec.outcome})
    ds.finalize()
    meta = Path(out) / "meta"
    stats, stats_file = checkpoint_stats(stats_from)
    json.dump(stats, open(meta / "stats.json", "w"), indent=1)
    with open(meta / "provenance.jsonl", "w") as fh:
        for r in prov_rows:
            fh.write(json.dumps(r) + "\n")
    summary = {"repo_id": repo_id, "episodes": len(prov_rows), "frames": sum(r["n_frames"] for r in prov_rows),
               "truncate_steps": truncate_steps, "adapter": adapter.describe(),
               "reference": {"dataset": reference, "root": info["_root"], "codebase_version": info.get("codebase_version")},
               "stats_from": {"checkpoint": stats_from, "file": stats_file},
               "records_roots": list(records_roots), "unknown_tasks": sorted(unknown_tasks),
               "heldout_lists": list(heldout_lists)}
    json.dump(summary, open(meta / "export.json", "w"), indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True, help="EpisodeStore roots")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--reference", default="IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot")
    ap.add_argument("--stats-from", default="nvidia/gr00t17-lerobot-libero_spatial-640")
    ap.add_argument("--truncate-steps", type=int, default=None)
    ap.add_argument("--adapter", default=LiberoIpecAdapter.name, choices=list(ADAPTERS))
    ap.add_argument("--heldout", nargs="*", default=["experiments/repro/r056_heldout.json", "experiments/repro/r056_val.json"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    s = export(a.records, a.out, a.repo_id, a.reference, a.stats_from, a.truncate_steps, a.adapter,
               [p for p in a.heldout if os.path.exists(p)], a.limit, a.overwrite)
    print(json.dumps({k: v for k, v in s.items() if k != "adapter"}, indent=1))


if __name__ == "__main__":
    main()
