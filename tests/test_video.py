"""Per-episode video: named by rollout_id, one frame per recorded step, and
oriented as the POLICY saw it (raw env pixels are 180 degrees off).

The frames carry markers that survive lossy encoding -- a bright quadrant per
camera -- so a flip or a panel swap fails on content, not on a count.
"""
from __future__ import annotations

import os

import numpy as np

from vla_harness.runner import rollout
from vla_harness.schema import Action, Observation, PerturbationSpec

H = W = 64
N_STEPS = 5


def _frames(t):
    # raw (unflipped) frames: bright TOP-LEFT quadrant on cam "image",
    # bright BOTTOM-RIGHT on "image2". After the 180-degree flip these swap:
    # image -> bottom-right, image2 -> top-left.
    a = np.zeros((H, W, 3), np.uint8)
    a[: H // 2, : W // 2] = 250
    b = np.zeros((H, W, 3), np.uint8)
    b[H // 2:, W // 2:] = 250
    return {"image2": b, "image": a}          # deliberately NOT sorted


class _Env:
    env_id = "fake-env"
    task_id = "fake-task"
    instruction = "do the thing"
    control_mode = "relative"

    def identity(self):
        return {"name": self.env_id}

    def semantic_deps(self):
        return {}

    def scene_descriptor(self):
        return {}

    def _obs(self):
        return Observation(instruction=self.instruction, t=self._t,
                           state={"eef_pos": [0.0, 0.0, 0.0]},
                           frames=_frames(self._t))

    def reset(self, seed, spec):
        self._t = 0
        return self._obs()

    def step(self, action):
        self._t += 1
        return self._obs(), False, self._t >= N_STEPS, "timeout"


class _Policy:
    policy_id = "fake-policy"
    action_dims = ["dx"]
    model_forwards = 0

    def identity(self):
        return {"name": self.policy_id}

    def reset(self):
        pass

    def __call__(self, obs):
        return Action([0.0], self.action_dims)


def test_video_named_counted_and_policy_oriented(tmp_path):
    import imageio.v3 as iio
    r = rollout(_Env(), _Policy(), 0, PerturbationSpec.of(),
                video_dir=str(tmp_path))
    meta = r.meta["video"]
    assert meta["path"] == os.path.join(str(tmp_path), r.rollout_id + ".mp4")
    assert os.listdir(tmp_path) == [r.rollout_id + ".mp4"]    # no .part left
    # N_STEPS actions + the terminal observation, same as len(r.steps)
    assert meta["n_frames"] == len(r.steps) == N_STEPS + 1
    assert meta["cameras"] == ["image", "image2"]              # sorted, fixed

    frames = iio.imread(meta["path"], plugin="pyav")
    assert len(frames) == N_STEPS + 1
    f = frames[0].astype(int)
    assert f.shape[:2] == (H, 2 * W)
    left, right = f[:, :W], f[:, W:]                           # image | image2
    q = lambda im, r0, c0: im[r0 * H // 2:(r0 + 1) * H // 2,
                              c0 * W // 2:(c0 + 1) * W // 2].mean()
    # flipped: cam "image" bright bottom-right, "image2" bright top-left
    assert q(left, 1, 1) > 200 and q(left, 0, 0) < 50
    assert q(right, 0, 0) > 200 and q(right, 1, 1) < 50


def test_video_off_changes_nothing(tmp_path):
    on = rollout(_Env(), _Policy(), 0, PerturbationSpec.of(),
                 video_dir=str(tmp_path))
    off = rollout(_Env(), _Policy(), 0, PerturbationSpec.of())
    assert off.meta["video"] is None
    assert on.rollout_id == off.rollout_id     # not part of the identity
