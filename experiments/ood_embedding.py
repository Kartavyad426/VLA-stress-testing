"""§4 of docs/EXP_EMBEDDING_OOD.md -- embedding-space OOD over captured features.

    .venvs/lerobot/bin/python experiments/ood_embedding.py \
        --capture runs/embed_derisk/capture

Currently implements **§4A** (swap the feature space, keep the method).
§4B (probe AUROC), §4C (signal x perturbation type) and §4D (onset timing) are
not implemented yet and are NOT stubbed as passing -- see the footer this prints.

READ THE REFERENCE-RATE COLUMN FIRST. It is the calibration check, not a
result: the reference must score ~alpha against itself leave-one-out. If it does
not, the threshold is mis-set and every separation ratio on the row is against a
wrong bar.

Reference and labels both come from the run being scored. GR00T's denoise loop
starts from an unseeded randn, so a re-run does not reproduce its own
success/failure split; borrowing a previous run's successes would silently
calibrate against a different policy state.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla_harness.capture.analysis import load_capture, score_run  # noqa: E402

# Ordered so the proprioceptive analogue sits beside the VL stream it is being
# compared against. S0's 6.9x on 5 raw proprioceptive dims is the number to beat.
SIGNALS = ["vl_encoder_mean", "vl_encoder_max",
           "vl_normed_mean", "vl_adapted_mean", "vl_adapted_max",
           "state_encoded"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, help="a run's capture/ directory")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="target false-positive rate ON THE REFERENCE")
    ap.add_argument("--signals", default=None, help="comma-separated subset")
    a = ap.parse_args()

    signals = a.signals.split(",") if a.signals else SIGNALS
    eps = load_capture(a.capture, signals[0])
    n_s = sum(1 for e in eps if e.success)
    print(f"CAPTURE {a.capture}")
    print(f"  {len(eps)} episodes: {n_s} success / {len(eps) - n_s} fail, "
          f"{sum(len(e.features) for e in eps)} model forwards total")
    if len(eps) - n_s == 0:
        print("\n  NO FAILURES in this capture -- separation is undefined. "
              "Nothing to report.")
        return

    print(f"\n{'signal':22s}{'ref rate':>10s}{'success':>10s}{'fail':>10s}"
          f"{'separation':>12s}{'p':>12s}")
    print("-" * 76)
    rows = []
    for sig in signals:
        try:
            r = score_run(a.capture, sig, alpha=a.alpha)
        except (ValueError, KeyError) as e:
            print(f"{sig:22s}  SKIPPED: {e}")
            continue
        rows.append(r)
        flag = "" if abs(r.reference_rate - a.alpha) < 0.5 * a.alpha else "  <-- MIS-CALIBRATED"
        print(f"{sig:22s}{r.reference_rate*100:9.1f}%{r.mean_ood_success*100:9.1f}%"
              f"{r.mean_ood_fail*100:9.1f}%{r.separation:11.2f}x{r.p_value:12.2e}{flag}")

    best = max(rows, key=lambda r: r.separation) if rows else None
    if best:
        sig = "significant" if best.p_value < 0.05 else "NOT SIGNIFICANT"
        print(f"\nbest signal here: {best.signal} at {best.separation:.2f}x ({sig})")
    print("\nS0's 6.9x is NOT a like-for-like bar and this tool does not claim "
          "to beat it:")
    print("  * S0 is per ENV STEP over whole episodes; these are per MODEL "
          "FORWARD (~5-8/episode).")
    print("  * S0 was measured on a different run and reference set.")
    print("  * state_encoded is proprioception AFTER the per-embodiment "
          "encoder, so it beating")
    print("    the VL signals says the state pathway carries the signal -- not "
          "that vision does.")
    print("  A like-for-like comparison needs S0 recomputed on THIS capture's "
          "episodes and forwards.")
    print("\nNOT YET IMPLEMENTED: §4B probe AUROC, §4C signal x perturbation "
          "type, §4D onset timing.")
    print("Distances are per MODEL FORWARD (~5-8/episode at n_action_steps=16), "
          "not per env step.")


if __name__ == "__main__":
    main()
