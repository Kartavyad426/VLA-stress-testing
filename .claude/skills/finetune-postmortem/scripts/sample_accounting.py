"""Dose accounting: how often did training actually see the frames that carry the new signal?

  python3 .claude/skills/finetune-postmortem/scripts/sample_accounting.py \
      --report runs/r056_r16_train_report.json [--episode-len 48 --driven 16] [--checkpoints 500,1000,1500,2000]
  # or without a report:
  python3 ... --part minted:4800:0.5 --part replay:52970:0.5 --effective-batch 8 --opt-steps 2000

The FIRST part is the new data. Its frames are split into frame 0 (the only frame whose whole
chunk label is the correction when the policy is queried at forward 0), frames 1..driven-1 (partly
corrected labels) and frames driven..episode_len-1 (the policy's own actions). Prints each category's
share of draws and the expected number of views per frame at each checkpoint:
views = share x effective_batch x opt_step / frames_in_category. Under 1 view per correction frame at the
selected checkpoint means the correction was barely trained on.
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", help="r056_train report json (parts, shares, effective_batch, opt_steps)")
    ap.add_argument("--part", action="append", default=[], help="name:frames:share (first = new data)")
    ap.add_argument("--effective-batch", type=int)
    ap.add_argument("--opt-steps", type=int)
    ap.add_argument("--episode-len", type=int, default=48)
    ap.add_argument("--driven", type=int, default=16)
    ap.add_argument("--checkpoints", default=None)
    a = ap.parse_args()
    if a.report:
        r = json.load(open(a.report))
        parts = [(p["name"], p["frames"], s) for p, s in zip(r["parts"], r["shares"])]
        eb, steps = r["effective_batch"], r["opt_steps"]
    else:
        parts = [(n, int(f), float(s)) for n, f, s in (x.split(":") for x in a.part)]
        eb, steps = a.effective_batch, a.opt_steps
    tot = sum(s for _, _, s in parts)
    parts = [(n, f, s / tot) for n, f, s in parts]
    ckpts = [int(x) for x in a.checkpoints.split(",")] if a.checkpoints else sorted({steps // 4, steps // 2, steps})
    name0, f0, s0 = parts[0]
    eps = f0 // a.episode_len
    L, D = a.episode_len, a.driven
    cats = [(f"{name0} frame 0 (correction)", s0 / L, eps),
            (f"{name0} frames 1-{D - 1} (partly corrected)", s0 * (D - 1) / L, eps * (D - 1)),
            (f"{name0} frames {D}-{L - 1} (policy's own actions)", s0 * (L - D) / L, eps * (L - D))]
    cats += [(f"{n} (replay)", s, f) for n, f, s in parts[1:]]
    print(f"effective batch {eb}, {steps} optimizer steps, {eb * steps} samples drawn; {eps} new episodes x {L} frames")
    print(f"{'category':<48} {'share':>7} " + " ".join(f"{'@' + str(c):>8}" for c in ckpts))
    for name, share, frames in cats:
        print(f"{name:<48} {share:>7.2%} " + " ".join(f"{share * eb * c / frames:>8.2f}" for c in ckpts))
    print("(columns: expected views per frame by that optimizer step)")
    no_signal = sum(sh for n, sh, _ in cats[2:])
    print(f"share of draws whose labels the base already produces (own actions + replay): {no_signal:.1%}")


if __name__ == "__main__":
    main()
