"""Does the loss mask exclude padded action steps at the end of truncated episodes? (CPU, no weights)

  .venvs/libero-plus/bin/python .claude/skills/finetune-postmortem/scripts/padcheck.py \
      --root data/r054_train48 --repo local/r054_train48 \
      [--ckpt nvidia/gr00t17-lerobot-libero_spatial-640 --base-model nvidia/GR00T-N1.7-3B --embodiment libero_sim] \
      [--episode 0] [--frames 0,15,32,40,47]

Builds the dataset with the policy's delta_timestamps and the checkpoint's real training preprocessor
(on CPU), then prints, per frame: the task string, action_is_pad count, and how many horizon steps
the preprocessed action_mask leaves valid. Near an episode's end the valid count must shrink
(frame L-1 -> 1). If it stays at the full chunk, padded steps (LeRobot repeats the last action) are
being trained on as real labels. Run from the repo root.
"""
from __future__ import annotations

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", default="nvidia/gr00t17-lerobot-libero_spatial-640")
    ap.add_argument("--base-model", default="nvidia/GR00T-N1.7-3B")
    ap.add_argument("--embodiment", default="libero_sim")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frames", default=None, help="comma list; default 0, mid, and the last 8 frames")
    a = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    from torch.utils.data import default_collate
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    cfg = PreTrainedConfig.from_pretrained(a.ckpt)
    for k, v in (("base_model_path", a.base_model), ("embodiment_tag", a.embodiment), ("device", "cpu")):
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    meta = LeRobotDatasetMetadata(a.repo, root=a.root)
    delta = resolve_delta_timestamps(cfg, meta, {})
    ds = LeRobotDataset(a.repo, root=a.root, delta_timestamps=delta, video_backend="pyav", return_uint8=True,
                        episodes=[a.episode])
    n = len(ds)
    frames = [int(x) for x in a.frames.split(",")] if a.frames else sorted({0, n // 2, *range(max(0, n - 8), n)})
    pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=a.ckpt,
                                      preprocessor_overrides={"device_processor": {"device": "cpu"}})
    print(f"chunk (action delta indices) {len({k: v for k, v in delta.items()}.get('action', []))}; episode {a.episode} has {n} frames")
    ok = True
    for i in frames:
        it = ds[i]
        pad = int(it["action_is_pad"].sum()) if "action_is_pad" in it else None
        b = _preprocess_dataset_batch(default_collate([it]), ds.meta.camera_keys, {}, pre)
        am = b.get("action_mask")
        valid = int((am[0].reshape(am.shape[1], -1).sum(-1) > 0).sum()) if am is not None else None
        expect = min(len(delta["action"]), n - i)
        ok &= valid == expect
        print(f"frame {i:3d}: task {str(it.get('task'))[:50]!r}  is_pad {pad}  valid mask steps {valid} (expected {expect})")
    print("PASS: padded steps are masked" if ok else "FAIL: the mask does not match the padding")


if __name__ == "__main__":
    main()
