"""Per-episode video, recorded live from the frames the policy was shown.

G2 keeps pixels out of traces; this writes one mp4 per rollout next to them and
puts the PATH in `Rollout.meta["video"]`. It exists because failures have to be
watched -- by a person or by a VLM (Cosmos-Reason) -- and the replay route in
`experiments/visualise_episode_video.py` is a reconstruction that needs the sim
and a GPU per episode. A recording needs neither.

Frames are streamed to the encoder, never buffered: ~280 steps x 2 cameras at
360x360 is ~200 MB of RAM per episode if held.

ORIENTATION. `obs.frames` are the RAW env pixels, taken before LeRobot's
LiberoProcessorStep flips every image 180 degrees (processor/env_processor.py:59).
Unflipped they show a world the policy never saw, so they are flipped here,
matching visualise_episode_video.py.
"""
from __future__ import annotations

import os

FPS = 20                      # LIBERO control frequency; playback speed only


class EpisodeVideo:
    """Both cameras side by side in one file, so one timeline covers both."""

    def __init__(self, path: str):
        self.path = path
        self._tmp = path + ".part.mp4"
        self._w = None
        self.n_frames = 0
        self._keys = None

    def add(self, frames: dict) -> None:
        import numpy as np
        if not frames:
            return
        if self._keys is None:
            # fix the camera order on the first frame; a dict reordering
            # mid-episode would swap the panels silently.
            self._keys = sorted(frames)
        imgs = [np.asarray(frames[k], dtype=np.uint8)[::-1, ::-1]
                for k in self._keys]
        h = max(im.shape[0] for im in imgs)
        if any(im.shape[0] != h for im in imgs):
            import cv2
            imgs = [im if im.shape[0] == h else
                    cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
                    for im in imgs]
        if self._w is None:
            import imageio.v2 as imageio
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self._w = imageio.get_writer(self._tmp, fps=FPS, codec="libx264",
                                         macro_block_size=1, quality=7)
        self._w.append_data(np.ascontiguousarray(np.hstack(imgs)))
        self.n_frames += 1

    def close(self) -> dict | None:
        """Finalise; returns the meta record, or None if nothing was written.

        Written to a temp name and renamed only on success, so a crash mid-
        episode leaves a `.part.mp4`, never a truncated file under the real name.
        """
        if self._w is None:
            return None
        self._w.close()
        os.replace(self._tmp, self.path)
        return {"path": self.path, "n_frames": self.n_frames, "fps": FPS,
                "cameras": list(self._keys), "orientation": "policy_view"}

    def abort(self) -> None:
        if self._w is not None:
            try:
                self._w.close()
            finally:
                if os.path.exists(self._tmp):
                    os.remove(self._tmp)
