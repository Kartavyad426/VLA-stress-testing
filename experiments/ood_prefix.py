"""Prefix-restricted re-analysis of a capture -- Corollary 7.1.

    .venvs/groot/bin/python experiments/ood_prefix.py \
        --capture runs/embed_derisk/capture

Pre-registered in docs/superpowers/specs/2026-09-22-prefix-restricted-reanalysis.md.

R-036 scores EVERY forward of an episode. Every forward after the first
divergence step is a descendant of the failure, so a failing episode is
displaced almost by definition -- whatever moved it. This tool asks how much of
the separation is available EARLY, by recomputing the whole R-036 pipeline on
the first k forwards only.

A prefix is a PROXY for "before divergence", not a measurement of it: h* needs a
matched control per episode, which this unpaired draw does not have. Read the
reference-rate column first at every k, exactly as in R-036 -- small k means one
point per episode and a noisy leave-one-out threshold.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla_harness.capture.analysis import (  # noqa: E402
    calibrate_threshold, load_capture, nn_distance)

SIGNALS = ["state_encoded", "vl_encoder_mean", "vl_encoder_max",
           "vl_normed_mean", "vl_adapted_mean", "vl_adapted_max"]


def _score(ref: list, queries: list, alpha: float):
    """R-036's scoring, on whatever slices are handed in.

    `ref` are the success slices (also scored, leave-one-out); `queries` the
    failures. Returns (ref_rate, success_mean, fail_mean, separation, p).
    """
    thr, rate = calibrate_threshold(ref, alpha)
    cloud = np.concatenate(ref)
    s = []
    for i, t in enumerate(ref):
        others = np.concatenate(ref[:i] + ref[i + 1:])
        s.append(float((nn_distance(t, others) > thr).mean()))
    f = [float((nn_distance(t, cloud) > thr).mean()) for t in queries]
    p = float(mannwhitneyu(s, f).pvalue) if s and f else float("nan")
    sm, fm = float(np.mean(s)), float(np.mean(f))
    # A zero success rate makes the RATIO undefined, not enormous. Dividing by
    # an epsilon here is the same defect R-036 caught on its first pass (4.9e7x
    # from a division by epsilon) and it fabricates a result rather than
    # crashing: this tool printed 2.8e8x for a cell where the reference simply
    # scored 0% -- which also means the threshold was mis-calibrated there.
    sep = float("nan") if sm == 0.0 else fm / sm
    return rate, sm, fm, sep, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--signals", default=None)
    a = ap.parse_args()
    signals = a.signals.split(",") if a.signals else SIGNALS

    eps = load_capture(a.capture, signals[0])
    ns = [len(e.features) for e in eps if e.success]
    nf = [len(e.features) for e in eps if not e.success]
    print(f"CAPTURE {a.capture}")
    print(f"  {len(eps)} episodes: {len(ns)} success / {len(nf)} fail")
    print("\nCONFOUND CHECK -- forwards per episode by outcome")
    print(f"  success  n={len(ns):2d}  mean {np.mean(ns):5.2f}  median "
          f"{np.median(ns):4.1f}  min {min(ns)}  max {max(ns)}")
    print(f"  fail     n={len(nf):2d}  mean {np.mean(nf):5.2f}  median "
          f"{np.median(nf):4.1f}  min {min(nf)}  max {max(nf)}")
    print(f"  Mann-Whitney p = {mannwhitneyu(ns, nf).pvalue:.3g}")
    kmax = max(max(ns), max(nf))

    for sig in signals:
        eps_s = load_capture(a.capture, sig)
        ref_all = [e.features for e in eps_s if e.success]
        q_all = [e.features for e in eps_s if not e.success]

        print(f"\n=== {sig} -- CUMULATIVE PREFIX (first k forwards) ===")
        print(f"{'k':>3s}{'ref pts':>9s}{'ref rate':>10s}{'success':>10s}"
              f"{'fail':>9s}{'separation':>12s}{'p':>11s}")
        print("-" * 64)
        for k in range(1, kmax + 1):
            ref = [t[:k] for t in ref_all]
            q = [t[:k] for t in q_all]
            try:
                rate, sm, fm, sep, p = _score(ref, q, a.alpha)
            except Exception as e:                      # noqa: BLE001
                print(f"{k:3d}  SKIPPED: {e}")
                continue
            flag = "" if abs(rate - a.alpha) < 0.5 * a.alpha else "  <-- MIS-CAL"
            seps = "     undef" if np.isnan(sep) else f"{sep:10.2f}x"
            print(f"{k:3d}{sum(len(t) for t in ref):9d}{rate*100:9.1f}%"
                  f"{sm*100:9.1f}%{fm*100:8.1f}%{seps}{p:11.2e}{flag}")

        print(f"\n--- {sig} -- PER-INDEX (forward j alone, matched length) ---")
        print(f"{'j':>3s}{'n_s':>5s}{'n_f':>5s}{'ref rate':>10s}{'success':>10s}"
              f"{'fail':>9s}{'separation':>12s}{'p':>11s}")
        print("-" * 66)
        for j in range(0, kmax):
            ref = [t[j:j + 1] for t in ref_all if len(t) > j]
            q = [t[j:j + 1] for t in q_all if len(t) > j]
            if len(ref) < 5 or len(q) < 3:
                continue
            try:
                rate, sm, fm, sep, p = _score(ref, q, a.alpha)
            except Exception as e:                      # noqa: BLE001
                print(f"{j:3d}  SKIPPED: {e}")
                continue
            flag = "" if abs(rate - a.alpha) < 0.5 * a.alpha else "  <-- MIS-CAL"
            seps = "     undef" if np.isnan(sep) else f"{sep:10.2f}x"
            print(f"{j:3d}{len(ref):5d}{len(q):5d}{rate*100:9.1f}%{sm*100:9.1f}%"
                  f"{fm*100:8.1f}%{seps}{p:11.2e}{flag}")

    print("\nA PREFIX IS NOT h*. Estimating the true first-divergence step needs "
          "a matched\ncontrol per episode; this draw is unpaired. Read this as "
          "'how much of the\nseparation is available early', not 'how much is "
          "available before divergence'.")


if __name__ == "__main__":
    main()
