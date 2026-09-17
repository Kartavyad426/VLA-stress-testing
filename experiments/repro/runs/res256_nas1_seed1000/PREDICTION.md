# Prediction, recorded BEFORE the result — 2026-09-16

Written while the run was in flight, so the result cannot be read into it.

**Source:** vla-7f, from lerobot#4614 (issue body read in full) quoting the
SmolVLA paper Table 13 (paper itself not yet checked): LIBERO average by
n_action_steps — 50 → 51.8%, **10 → 82.8%, 1 → 80.3%**. Non-monotonic: 1 ≈ 10 >> 50.

**Prediction:** libero_spatial at nas=1 lands **near 73.0%, or a few pp below**
(baseline res256_nas10_seed1000).

**How to read it:**
- Near 73% or slightly below → nas=1 vs 10 is NOT the remaining ~20-pp gap.
  Falsifies it as a cause; consistent with the paper.
- A large jump (e.g. +10 pp or more) → surprising, contradicts Table 13, and
  would need a second seed before believing it.
- n=100 per arm: the 95% Wilson interval is ~±8 pp, so differences under ~10 pp
  are not distinguishable from noise in a single run.

## Update, still before the result — Table 13 read directly (vla-7f, [F])

| nas | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| 1 | 89 | 94 | 85 | 53 | 80.3 |
| 10 | 89 | 94 | 91 | 57 | 82.8 |
| 50 | 54 | 70 | 58 | 25 | 51.8 |

**Spatial is identical at nas=1 and nas=10.** Prediction sharpened: ~73%.

**Caveat:** paper §4.7 — ablations are "trained from scratch without any
pretraining on robotics data". Table 13 is evidence about the ARCHITECTURE's
sensitivity to nas, not a spec for the released checkpoint.

## RESULT — 2026-09-16, run completed rc=0, 4274 s

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nas=1 | 6 | 9 | 8 | 7 | 7 | 10 | 5 | 8 | 10 | 7 | **77** |
| nas=10 | 7 | 8 | 8 | 6 | 7 | 8 | 8 | 6 | 7 | 8 | **73** |

**+4.0 pp, 95% CI [−8.0, 16.0].** Paired by episode (same seed and init state):
18 episodes succeed only at nas=1, 14 only at nas=10, exact McNemar **p = 0.597**.

**Verdict: consistent with the prediction.** nas=1 vs nas=10 makes no detectable
difference on spatial, which matches Table 13 (89 at both). `n_action_steps`
between 1 and 10 is **not** the residual gap. F9 hypothesis #1 is falsified as a cause.

The 32 discordant episodes (18 vs 14) are a large amount of per-episode
disagreement for no net effect. Outcomes at this n are noisy episode to episode,
and single-arm differences under ~10 pp should not be read as signal.
