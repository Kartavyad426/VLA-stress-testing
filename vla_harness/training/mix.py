"""A weighted mixture of LeRobot datasets that share one schema.

lerobot_train refuses MultiLeRobotDataset (datasets/factory.py) and its
EpisodeAwareSampler has no weights, so "mix the new data 1:1 with the
original demos by sampling weight" (R-056) needs both a dataset and a
sampler. `MixedDataset` concatenates; `mixture_weights` gives every sample of
part k the weight share_k / len(part_k), so each draw comes from part k with
probability share_k whatever the parts' sizes.

Only the keys every part has are returned, so a part's extra columns (the
export's `frame_driven`) never break collation. `meta` is the FIRST part's
metadata: the parts must agree on features and fps, which is checked.
"""
from __future__ import annotations

import bisect

import torch


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, parts: list, names: list[str] | None = None):
        if not parts:
            raise ValueError("no datasets")
        self.parts = list(parts)
        self.names = names or [getattr(p, "repo_id", f"part{i}") for i, p in enumerate(parts)]
        ref = parts[0].meta
        for p in parts[1:]:
            for k in {k for k in set(ref.features) & set(p.meta.features) if k.startswith(("observation.", "action"))}:
                a, b = dict(ref.features[k]), dict(p.meta.features[k])
                for x in (a, b):
                    x.pop("info", None); x["shape"] = list(x["shape"])
                if a != b:
                    raise ValueError(f"feature {k!r} differs between parts: {a} vs {b}")
            if p.meta.fps != ref.fps:
                raise ValueError(f"fps differs: {ref.fps} vs {p.meta.fps}")
            missing = {k for k in ref.features if k.startswith(("observation.", "action"))} - set(p.meta.features)
            if missing:
                raise ValueError(f"{self.names[parts.index(p)]} lacks {sorted(missing)}")
        self.meta = ref
        self.cum = []
        n = 0
        for p in parts:
            n += len(p)
            self.cum.append(n)
        self._keys = None

    def __len__(self):
        return self.cum[-1]

    @property
    def num_frames(self):
        return len(self)

    @property
    def num_episodes(self):
        return sum(p.num_episodes for p in self.parts)

    @property
    def episodes(self):
        return None

    def locate(self, idx: int) -> tuple[int, int]:
        k = bisect.bisect_right(self.cum, idx)
        return k, idx - (self.cum[k - 1] if k else 0)

    def __getitem__(self, idx):
        k, j = self.locate(int(idx))
        item = self.parts[k][j]
        if self._keys is None:
            common = None
            for p in self.parts:
                ks = set(p[0].keys())
                common = ks if common is None else common & ks
            self._keys = common
        return {key: v for key, v in item.items() if key in self._keys}


def mixture_weights(ds: MixedDataset, shares: list[float]) -> torch.Tensor:
    if len(shares) != len(ds.parts):
        raise ValueError("one share per part")
    s = torch.tensor(shares, dtype=torch.double)
    s = s / s.sum()
    w = []
    for p, share in zip(ds.parts, s):
        w.append(torch.full((len(p),), float(share) / len(p), dtype=torch.double))
    return torch.cat(w)


def weighted_loader(ds: MixedDataset, shares, batch_size: int, num_samples: int, seed: int,
                    num_workers: int = 2, start_step: int = 0, **kw) -> torch.utils.data.DataLoader:
    """Draws with replacement. Deterministic in (seed, start_step): a resumed run draws a
    fresh but reproducible stream rather than replaying the first steps' samples."""
    g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(start_step))
    sampler = torch.utils.data.WeightedRandomSampler(mixture_weights(ds, shares), num_samples=num_samples,
                                                     replacement=True, generator=g)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                                       drop_last=True, pin_memory=torch.cuda.is_available(),
                                       persistent_workers=num_workers > 0, **kw)
