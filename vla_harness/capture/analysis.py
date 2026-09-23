"""§4A: self-referenced OOD over captured embeddings.

Same method as `experiments/ood_selfref.py` -- reference = the policy's OWN
successes, threshold calibrated leave-one-out so the false-positive rate is a
design parameter rather than an outcome -- with the 5-dim proprioceptive feature
vector swapped for a captured embedding. **The number to beat is S0's 6.9x.**

TWO CONSTRAINTS THAT ARE NOT STYLE CHOICES:

  * **The reference must come from the SAME run being scored.** GR00T's denoise
    loop starts from an unseeded `torch.randn` (groot_n1_7.py:657), so a re-run
    does not reproduce its own success/failure split. Calibrating against an
    earlier run's successes means "the policy's own successes" silently refers
    to a different policy state than the one being scored, and the leave-one-out
    threshold inherits that mismatch.

  * **Distances are per MODEL FORWARD, not per env step.** GR00T at
    n_action_steps=16 runs the model ~5-8 times in a ~100-step episode and acts
    open-loop in between, so an episode contributes ~7 points, not ~100. Any
    "% steps OOD" phrasing inherited from the proprioceptive tool would be
    wrong here; this module says forwards and means forwards.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

MANIFEST = "capture_manifest.jsonl"


@dataclass
class Episode:
    rollout_id: str
    features: np.ndarray          # (n_forwards, dim)
    success: bool
    task_id: str
    env_step: list
    meta: dict


@dataclass
class Result:
    signal: str
    threshold: float
    reference_rate: float         # must land at ~alpha; this is the calibration check
    n_reference: int
    n_fail: int
    mean_ood_success: float
    mean_ood_fail: float
    separation: float
    p_value: float


def load_capture(cap_dir, signal: str) -> list[Episode]:
    """Join sidecars to labels on rollout_id, refusing any that do not join.

    An unjoined .npz has no label. Dropping it silently would shrink the
    reference; scoring it as a success would corrupt the reference cloud. Both
    are worse than stopping.
    """
    cap_dir = str(cap_dir)
    rows = {}
    mpath = os.path.join(cap_dir, MANIFEST)
    if os.path.exists(mpath):
        with open(mpath) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    rows[r["rollout_id"]] = r

    out = []
    for fn in sorted(os.listdir(cap_dir)):
        if not fn.endswith(".npz"):
            continue
        rid = fn[:-4]
        if rid not in rows:
            raise ValueError(
                f"capture: {fn} has no manifest row (rollout_id {rid!r}) -- its "
                f"label is unknown, and both dropping it and assuming an outcome "
                f"would corrupt the reference. Refusing to load this capture.")
        with np.load(os.path.join(cap_dir, fn)) as z:
            if signal not in z:
                raise ValueError(f"{fn} has no signal {signal!r}; "
                                 f"available: {sorted(k for k in z.files)}")
            feats = z[signal].astype(np.float64)
        r = rows[rid]
        out.append(Episode(rid, feats, bool(r["success"]), r.get("task_id", ""),
                           r.get("env_step") or [], r))
    return out


def nn_distance(q: np.ndarray, cloud: np.ndarray, chunk: int = 4000) -> np.ndarray:
    """1-nearest-neighbour distance from each query point to the cloud."""
    out = np.full(len(q), np.inf)
    for i in range(0, len(cloud), chunk):
        d = np.linalg.norm(q[:, None, :] - cloud[None, i:i + chunk, :], axis=2).min(1)
        out = np.minimum(out, d)
    return out


def calibrate_threshold(reference: list[np.ndarray], alpha: float = 0.05):
    """Leave-one-out threshold, so alpha is a design parameter not an outcome.

    Returns (threshold, achieved_rate). The achieved rate landing at ~alpha is
    the CALIBRATION CHECK, not a result -- if it does not, every ratio computed
    downstream is against a mis-set bar.
    """
    loo = []
    for i, t in enumerate(reference):
        others = np.concatenate([x for j, x in enumerate(reference) if j != i])
        loo.append(nn_distance(t, others))
    thr = float(np.percentile(np.concatenate(loo), 100 * (1 - alpha)))
    rate = float(np.mean([(d > thr).mean() for d in loo]))
    return thr, rate


def score_run(cap_dir, signal: str, alpha: float = 0.05,
              min_reference: int = 5) -> Result:
    eps = load_capture(cap_dir, signal)
    ref = [e.features for e in eps if e.success]
    if len(ref) < min_reference:
        raise ValueError(f"only {len(ref)} successful episodes in {cap_dir}; "
                         f"need >= {min_reference} to calibrate a threshold")
    thr, rate = calibrate_threshold(ref, alpha)
    cloud = np.concatenate(ref)

    # A SUCCESS EPISODE IS PART OF THE REFERENCE, so it must be scored against
    # the cloud MINUS ITSELF. Scoring it against the full cloud finds its own
    # points at distance 0, collapses the success rate to 0%, and turns the
    # separation ratio into a division by epsilon -- 4.9e7x on the real stage-0
    # capture, which is an artefact, not a finding.
    #
    # `ood_selfref.py` never hit this because its --ref and --query are
    # DIFFERENT runs. Here they cannot be: the unseeded randn at
    # groot_n1_7.py:657 means a re-run does not reproduce its own split, so the
    # reference has to come from the run being scored. The constraint that makes
    # the experiment valid is what breaks the inherited method.
    #
    # The invariant to check in any output: reference_rate and mean_ood_success
    # describe the same episodes and must agree.
    frac = {True: [], False: []}
    ref_i = 0
    for e in eps:
        if e.success:
            others = np.concatenate(ref[:ref_i] + ref[ref_i + 1:])
            ref_i += 1
            d = nn_distance(e.features, others)
        else:
            d = nn_distance(e.features, cloud)
        frac[e.success].append(float((d > thr).mean()))

    s = np.array(frac[True]) if frac[True] else np.array([0.0])
    f = np.array(frac[False]) if frac[False] else np.array([0.0])
    p = float("nan")
    if frac[True] and frac[False]:
        from scipy.stats import mannwhitneyu
        p = float(mannwhitneyu(frac[True], frac[False]).pvalue)
    return Result(signal, thr, rate, len(frac[True]), len(frac[False]),
                  float(s.mean()), float(f.mean()),
                  float(f.mean() / max(s.mean(), 1e-9)), p)


def detection_rate(arm: list[np.ndarray], reference: list[np.ndarray],
                   threshold: float, leave_one_out: bool = False) -> float:
    """Q1: what fraction of an arm's forwards fall outside the nominal cloud?

    This is the question R-036 never asked. It gates Q2: if a perturbation does
    not move a signal at all, a pass/fail null within that perturbation says
    the instrument is deaf, not that the model is blind.

    `leave_one_out` is REQUIRED when scoring the nominal arm against itself --
    otherwise each episode finds its own points at distance 0 and reads 0%,
    making every ratio against it a division by epsilon (O8).
    """
    out = []
    for i, ep in enumerate(arm):
        cloud = (np.concatenate(reference[:i] + reference[i + 1:])
                 if leave_one_out else np.concatenate(reference))
        out.append(float((nn_distance(ep, cloud) > threshold).mean()))
    return float(np.mean(out)) if out else float("nan")
