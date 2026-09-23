# Pre-registration — prefix-restricted re-analysis of R-036's capture

**Written before the numbers were computed.** Analysis only: no new GPU time,
no new episodes, the 40 captures in `runs/embed_derisk/capture` as they stand.

## Why

R-036 scores every model forward of an episode and reports one separation per
signal. `state_encoded` separates at 12.95x. Corollary 7.1 of
`docs/TRIGGER_MECHANISM_CALCULUS.html` says that number cannot be read as
mechanistic: every activation after the first divergence step is a DESCENDANT
of the failure, so a failing episode is displaced almost by definition,
whatever moved it. The per-category table already hints at this — the state
signal is broad across every category (27.8%–60.4%), which disconfirmed the
hypothesis that it was carried by the robot-init and layout episodes.

## What is computed

For prefix length k = 1..18, restrict every episode to its FIRST k forwards and
recompute the whole R-036 pipeline on the restriction: reference cloud from the
successes' first k forwards, leave-one-out threshold at alpha=0.05, per-episode
fraction of forwards beyond threshold, separation = fail/success, Mann-Whitney p.

Also computed, as the confound check that motivates the whole thing:

  * n_forwards by outcome. Failures run to the step cap; successes terminate on
    success. If failures are systematically longer, the all-forwards number is
    partly "long episodes drift", which is Corollary 7.1 in the crudest form.
  * per-INDEX separation at forward j (not cumulative), over episodes that have
    at least j+1 forwards — matched length, so episode duration cannot carry it.

## Predictions

1. **Failures have more forwards than successes**, p < 0.05.
2. `state_encoded` separation **declines monotonically as k decreases** and lands
   near 1x at k = 1–2. That is Corollary 7.1: the signal is post-failure
   displacement.
3. Per-index separation at j = 0 is **not significant**.
4. The VL signals stay flat at every k — they are flat for the reasons in
   Theorems 1, 6 and 6b, and prefix restriction does not touch any of them.

## Falsifier

If `state_encoded` separation survives at k = 1–2 at comparable magnitude, and
per-index j = 0 separates, the 12.95x is a genuine PRE-failure signal and the
state pathway is the strongest lead in the project. That would be the good
outcome, and it is the one this analysis is set up to be able to find.

## What this CANNOT do

It cannot estimate h*, the true first-divergence step: that needs a matched
control per episode, which this unpaired 40-episode draw does not have. A
prefix is a crude proxy — it bounds how much of the signal is available EARLY,
not how much is available BEFORE divergence. Small k also costs power: one
point per episode makes the per-episode fraction 0 or 1 and the leave-one-out
threshold noisy. The calibration column (reference rate ~ alpha) must be read
first at every k, exactly as in R-036.
