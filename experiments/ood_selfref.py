"""Self-referenced OOD: is a rollout outside the policy's OWN known-good behaviour?

Needs NO training corpus. The reference is a set of rollouts where the policy
succeeded; the threshold is calibrated leave-one-out on that reference, so the
false-positive rate is a design parameter rather than an outcome.

    .venvs/lerobot/bin/python experiments/ood_selfref.py \
        --ref groot_harness_parity --query lplus_fail_groot

Feature space: the 5 dimensions of the policy's observation state that can be
matched without assuming a rotation convention -- eef xyz + both gripper
fingers -- normalised by the LIBERO training corpus's own stats so distances
are comparable across dimensions with different units.

Caveats this tool cannot fix, and which belong with any number it prints:
  * 5 proprioceptive dims only; says nothing about appearance coverage.
  * Direction of causation is not established. A failing episode ends up in odd
    states almost by definition, so OOD may accompany failure rather than cause
    it. The discriminator is onset timing, which this does not measure.
    R-036 sharpened this: the state signal separated failures at a similar rate
    in EVERY perturbation category, which is what "unusual because it failed"
    looks like, and not what a perturbation-specific mechanism looks like.

!! --ref AND --query MUST BE DIFFERENT RUNS. !!

    This is assumed everywhere below and enforced nowhere. Point both at the
    same run and every success episode is scored against a reference cloud
    CONTAINING ITS OWN POINTS: nearest-neighbour distance 0, success OOD rate
    0%, and a separation ratio of x/1e-9. R-036's first pass reported 4.9e7x
    this way and it would have been published as a spectacular result.

    The tell is an invariant worth applying anywhere a reference cloud is
    built: TWO NUMBERS THAT DESCRIBE THE SAME EPISODES MUST AGREE. Here that is
    the reference rate against the success rate -- 5.1% against 0.0% is not a
    strong result, it is a bug.

    Same-run references are not perverse; they are REQUIRED when the policy is
    non-deterministic. GR00T's denoising loop draws an unseeded `torch.randn`
    (groot_n1_7.py:657), so a re-run does not reproduce its own successes and
    labels must come from the run being scored. If you need that, score the
    reference leave-one-out as `_calibrate` already does internally -- do not
    pass the same run to both flags.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vla_harness.schema import TraceStore  # noqa: E402

DIMS = [0, 1, 2, 6, 7]
STATS = ("/home/imerit/.cache/huggingface/hub/datasets--lerobot--libero/"
         "snapshots/*/meta/stats.json")


def norm_params():
    st = json.load(open(glob.glob(STATS)[0]))["observation.state"]
    return np.array(st["mean"])[DIMS], np.array(st["std"])[DIMS]


def traj(r, mu, sd, stride=5):
    q = []
    for s in r.steps[::stride]:
        o = s.obs_state
        if "eef_pos" in o and "gripper_qpos" in o:
            q.append(list(o["eef_pos"]) + list(o["gripper_qpos"]))
    return (np.array(q) - mu) / sd if len(q) >= 3 else None


def nn(q, c, chunk=3000):
    out = np.full(len(q), np.inf)
    for i in range(0, len(c), chunk):
        d = np.linalg.norm(q[:, None, :] - c[None, i:i + chunk, :], axis=2).min(1)
        out = np.minimum(out, d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="run id providing known-good rollouts")
    ap.add_argument("--query", required=True, help="run id to score")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="target false-positive rate on the reference")
    ap.add_argument("--suite", default=None,
                    help="restrict BOTH runs to this suite; without it a "
                         "reference spanning suites the query does not cover "
                         "inflates the cloud and understates OOD")
    a = ap.parse_args()

    mu, sd = norm_params()
    def keep(r):
        return a.suite is None or a.suite in r.task_id
    ref = [r for r in TraceStore("runs", a.ref).load() if r.success and keep(r)]
    T = [t for t in (traj(r, mu, sd) for r in ref) if t is not None]
    if len(T) < 5:
        sys.exit(f"reference {a.ref}: only {len(T)} usable successful rollouts")
    cloud = np.concatenate(T)

    # leave-one-out calibration on the reference itself
    loo = []
    for i, t in enumerate(T):
        others = np.concatenate([x for j, x in enumerate(T) if j != i])
        loo.append(nn(t, others))
    thr = float(np.percentile(np.concatenate(loo), 100 * (1 - a.alpha)))
    ref_frac = float(np.mean([(d > thr).mean() for d in loo]))

    print(f"REFERENCE  {a.ref}: {len(T)} successful rollouts, "
          f"{len(cloud)} frames")
    print(f"  threshold (leave-one-out p{100*(1-a.alpha):.0f}) = {thr:.3f}"
          f"   reference itself scores {ref_frac*100:.1f}% steps OOD")

    q = [r for r in TraceStore("runs", a.query).load() if keep(r)]
    rows = {True: [], False: []}
    for r in q:
        t = traj(r, mu, sd)
        if t is None:
            continue
        d = nn(t, cloud)
        rows[r.success].append(((d > thr).mean(), float(np.median(d))))
    print(f"\nQUERY  {a.query}")
    print(f"  {'outcome':9s}{'n':>5s}{'median dist':>13s}{'mean % steps OOD':>19s}")
    for k, lab in ((True, "SUCCESS"), (False, "fail")):
        if rows[k]:
            f = np.array([x[0] for x in rows[k]])
            m = np.array([x[1] for x in rows[k]])
            print(f"  {lab:9s}{len(f):5d}{np.median(m):13.3f}{f.mean()*100:18.1f}%")
    if rows[True] and rows[False]:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu([x[0] for x in rows[True]], [x[0] for x in rows[False]])
        sep = np.mean([x[0] for x in rows[False]]) / max(np.mean([x[0] for x in rows[True]]), 1e-9)
        print(f"  separation {sep:.1f}x, Mann-Whitney p = {p:.2e}"
              f"{'' if p < 0.05 else '   NOT SIGNIFICANT'}")


if __name__ == "__main__":
    main()
