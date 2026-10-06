"""Training-loss trace from a lerobot_train log: plateau, gradient norm vs clip, LR, per phase.

  python3 .claude/skills/finetune-postmortem/scripts/loss_trace.py \
      runs/overnight/R056fix_train.log [more logs...] --accum 8 [--blocks 10] [--clip 1.0]

lerobot_train logs `step:<micro> ... loss:<mean since last log> grdn:<..> lr:<..>` every log_freq
MICRO-steps; steps >= 1000 are printed rounded ("1.2K"), so lines are taken in FILE ORDER and never
sorted or de-duplicated by step. Progress bars use '\r', which is normalised. Each logged loss is a
mean over log_freq micro-batches, so single values are noisy: read block means/medians, not lines.
Reports, per block of optimizer steps: mean/median/p90 loss, median/max grad norm, LR at block end,
and the optimizer step where the block-median loss first comes within 10% of its final value (plateau).
"""
from __future__ import annotations

import argparse
import re
import statistics as st

PAT = re.compile(r"step:([0-9.]+K?) .*?loss:([0-9.e+-]+) grdn:([0-9.e+-]+) lr:([0-9.e+-]+)")


def parse(path):
    txt = open(path, errors="ignore").read().replace("\r", "\n")
    rows = []
    for line in txt.split("\n"):
        m = PAT.search(line)
        if m:
            rows.append((float(m.group(2)), float(m.group(3)), float(m.group(4))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--log-freq", type=int, default=10, help="micro-steps between log lines")
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--clip", type=float, default=1.0)
    a = ap.parse_args()
    for path in a.logs:
        rows = parse(path)
        if not rows:
            print(f"== {path}: no loss lines"); continue
        n = len(rows)
        opt_total = n * a.log_freq // a.accum
        print(f"== {path}: {n} log lines = {n * a.log_freq} micro-steps = {opt_total} optimizer steps")
        per = max(1, n // a.blocks)
        meds = []
        for b in range(0, n, per):
            blk = rows[b:b + per]
            L = [x[0] for x in blk]; G = [x[1] for x in blk]
            o0, o1 = b * a.log_freq // a.accum, (b + len(blk)) * a.log_freq // a.accum
            meds.append((o1, st.median(L)))
            print(f"  opt {o0:6d}-{o1:6d}  loss mean {st.mean(L):.4f} median {st.median(L):.4f} "
                  f"p90 {sorted(L)[int(.9 * (len(L) - 1))]:.3f}  grad median {st.median(G):.3f} max {max(G):.2f} "
                  f"(>{a.clip:g}: {sum(g > a.clip for g in G)})  lr {blk[-1][2]:.1e}")
        final = st.median([m for _, m in meds[-3:]])
        plateau = next((o for o, m in meds if m <= 1.1 * final), None)
        first = meds[0][1]
        print(f"  first-block median {first:.4f} -> final {final:.4f} ({final / first:.0%});"
              f" block median within 10% of final by opt step {plateau}")


if __name__ == "__main__":
    main()
