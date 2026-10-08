"""R-056 D20 diagnosis + WiSE, R-058 r=64 completion (SFT, 2026-10-07). One-shot edit."""
from pathlib import Path

P = Path("/home/imerit/Documents/Code/VLA/RESULTS.md")
t = P.read_text()

def sub1(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:100])
    t = t.replace(old, new)

# ---------------------------------------------------------------- R-056 heading, status, index
sub1("## R-056 — DONE, UNINTERPRETABLE BY ITS OWN RULE (interpretation provisional, pending D20): start-pose retraining pilot, arm (a) — held-out 59/108 vs base 67/108; the whole drop is scene 1327 at the grasp",
     "## R-056 — DONE, UNINTERPRETABLE BY ITS OWN RULE (post-hoc diagnosis D20 added 10-07): start-pose retraining pilot, arm (a) — held-out 59/108 vs base 67/108; the whole drop is scene 1327 at the grasp; the adapter fitted its training starts and did not generalise to the held-out directions")
sub1("**Status** DONE — UNINTERPRETABLE BY ITS OWN RULE; interpretation provisional pending D20 (scored 2026-10-06)",
     "**Status** DONE — UNINTERPRETABLE BY ITS OWN RULE (scored 2026-10-06; post-hoc D20 diagnosis and WiSE added 2026-10-07)")
sub1("| DONE — UNINTERPRETABLE BY ITS OWN RULE; interpretation provisional (D20) | Held-out 59/108 vs base 67/108 (p = 0.24), gap −0.20; whole drop is scene 1327 at the grasp (12/12 → 4/12); excluding 1327, 55 vs 55 |",
     "| DONE — UNINTERPRETABLE BY ITS OWN RULE; post-hoc D20 diagnosis (10-07) | Held-out 59/108 vs base 67/108 (p = 0.24), gap −0.20; whole drop is scene 1327 at the grasp (12/12 → 4/12); excluding 1327, 55 vs 55. D20: training starts closed 30% toward their labels, held-out 12%, so it is a generalisation failure, not a fit failure. WiSE α 0.5: 69/108 |")

# ---------------------------------------------------------------- R-056 corrected radius bullet
sub1("""  - The post-mortem's account (§7): achieved radius is 0.74–0.78 × requested, and the two held-out directions per scene give very different gripper offsets from R-047's average direction.
  - D20's radius check bears on this.""",
"""  - *Corrected 2026-10-07.* The 10-06 text here quoted the post-mortem's "achieved radius 0.74–0.78 × requested" as if it were one factor. D20(ii) shows the factor depends on direction: about 0.72 for R-047's direction 0, and about 0.95 / 0.93 for directions 1 and 2. The account of the mismatch is in the D20 addendum below.""")

# ---------------------------------------------------------------- R-056 interpretation -> D20 addendum + WiSE
i = t.index("**Interpretation — PROVISIONAL, pending D20.**")
j = t.index("**Not run.** WiSE-FT at α = 0.5 (exploratory), D19.\n", i) + len("**Not run.** WiSE-FT at α = 0.5 (exploratory), D19.\n")

D20 = r"""### RESULT addendum, 2026-10-07 — post-hoc diagnosis (D20) and WiSE-FT. Not pre-registered.

**D20 was approved by the user on 2026-10-06.** impl ran it and my prim read it; SFT co-read it. All runs are committed in a3dcb3d. Everything in this addendum is post hoc: diagnosis, not a test of the registered expectations. For each item, **verified** means SFT re-derived the number from the artifact. **Read** means the number was taken from the summary file without being re-derived.

**(i) Fixed-seed flow loss per subset** (`runs/r056_diag/loss_summary.json`, `loss_rows.jsonl`; post-mortem §9 A).
- *Design:* n = 100 frames per subset, 8 draws, seed 2020, paired with base. Augmentation was off, as in training. Base repeats were bit-identical.
- *Training frame 0 was fitted* (read). Mean loss on the minted frame-0s, which are all 100 training starts: base 0.159; r = 16 opt500 / 1000 / 2000 0.078 / 0.059 / 0.055; r = 4 opt500 0.081. Between 91% and 96% of frames fell below base, and every 95% CI on the difference excludes 0. Frames 1–15 and 16–47 also fell (ratio of means 0.36–0.83).
- *The original-data subset (IPEC) is mixed* (read). Median loss is slightly worse: base 0.0077, adapters 0.0085–0.0097. Only 24–42% of frames fell below base. The mean fell, but only because of a few outliers in base's heavy tail, and every CI on the mean difference includes 0. This is consistent with nominal 26/27.
- *Reading:* the "dilution means the adapter could not fit" hypothesis does not hold. Dilution stays a candidate for poor *generalisation*.

**(ii) R-047's success curve against the achieved gripper offset** (`runs/r056_diag/r047_offset_curve.json`, 360/360 R-047 rollouts with a start state).
- *Achieved / requested joint radius depends on direction* (read): direction 0 about 0.72; directions 1 and 2 about 0.95 and 0.93.
- *Success against offset in cm* (read):
  | offset | ≤ 8 cm | 8–12 cm | 12–16 cm | > 16 cm |
  |---|---|---|---|---|
  | success | ≈ 0.93 | 0.65 | 0.22 | 0.15 |
- *Direction matters inside the same cm band* (read): at 8–12 cm, direction 0 scores 33/40 and direction 1 scores 7/20.
- *Applied to R-056's held-out starts, the cm curve predicts base 55.7/108; base scored 67* (verified: SFT recomputed each start's t = 0 offset against `runs/r047/nominal_starts.json`). R-047's radius curve predicted 0.46 (≈ 50/108). The cm re-plot therefore closes **about a third** of the 0.46 → 0.62 gap.
  - The rest is almost entirely one cell: held-out direction 101 at 16–20 cm, which scored **16/33 against 5.0 predicted** (+11 of the +11.3 total).
  - R-047's > 16 cm points come only from its directions 1 and 2, and R-056's held-out directions (100, 101) are a different set.
  - The residual is a direction effect, not a radius-labelling error.
- **Consequence:** evals and training data should be stratified by direction × offset in cm, not by requested radius.

**(iii) The first-chunk fit test on the training starts** (`runs/r056_diag/fit_firstchunk.json`, read). The metric is post-mortem an7's (`experiments/analysis_r056_postmortem/an7.py`): ‖Σ_{16 steps}(dx, dy, dz) − reference‖ over chunk 0, as a median over starts.
- *Pipeline anchor:* the held-out re-computation reproduces an7 exactly (base 4.51, r = 16 3.98, r = 4 4.12, 30/36 closer). The one-forward path matches the stored rollouts' chunk-0 displacement at median 0.0 (max 0.13). The minted labels equal the nominal chunk at seed 0 (median 0.0). **The pipeline is cleared as a suspect.**
- *Training starts* (n = 100, reference = each start's minted label):
  - r = 16 moved from 3.40 to 2.37, closing **30%**, with 74% of starts closer; r = 4 closed 27%.
  - Against the an7-style nominal reference, which is comparable with the held-out figure, r = 16 closed 24%.
  - By offset, r = 16: 0–4 cm 23% (n = 24), 4–8 cm 19% (49), 8–12 cm 48% (18), 12–16 cm 42% (7), 16–20 cm 61% (2).
- *Held-out starts* (n = 36, seed 0): r = 16 closed **12%** overall. By direction × cm: dir 100 4–8 cm **0.5%** (n = 9), dir 100 8–12 cm 12% (9), dir 101 8–12 cm 11% (3), dir 101 12–16 cm 15% (4), dir 101 16–20 cm 10% (11).
- *The cleanest matched cell* is 4–8 cm: training starts closed 19–22% there, held-out direction 100 closed about 0%.

**Post-hoc reading** (SFT and my prim agree; not pre-registered).
- **Fit:** the adapter fitted its training starts, measured both by loss and by the sampled first chunk. Even there it moved only about 30% of the way, so the recipe is also under-dosed.
- **Generalisation:** on held-out starts it moved 0–15%. That gap is **consistent with a failure to generalise across directions, not shown.** Every training start has its own random direction (one start per direction), so direction and offset are confounded within the training set. The held-out directions 100 and 101 occur nowhere in training.
- **Suspects, in order:**
  1. the data: direction diversity and dose per region of start space (coverage, post-mortem §10);
  2. an under-dosed recipe: effective batch 8, each frame 0 seen fewer than once;
  3. the adapter: ruled down by R-058, where ranks 4 / 16 / 64 give 65 / 59 / 64.
- **Not explained by this:** the scene-1327 grasp regression (post-mortem §4). It appears at every rank and needs its own control (the replay-only arm, D23).

**WiSE-FT at α = 0.5** (exploratory, as registered; on r = 16 opt1000; run 2026-10-06 by user decision D19; `runs/r056_r16/manifest.jsonl` sets `heldout_wise` and `nominal_wise`). Verified: paired counts were re-derived by SFT, joined on the start key.

| readout | base | r = 16 | WiSE α = 0.5 |
|---|---|---|---|
| held-out (108) | 67 | 59 | **69 (0.64 [0.54, 0.72])** |
| paired vs base | | 14 gained / 22 lost, p = 0.24 | 11 gained / 9 lost, p = 0.82 |
| paired vs r = 16 | | | 19 gained / 9 lost, p = 0.087 |
| scene 1327 (12) | 12 | 4 | 8 (0 gained / 4 lost vs base, p = 0.125) |
| excluding 1327 (96) | 55 | 55 | 61 (11 / 5 vs base, p = 0.21) |
| scene 984 (12) | 6 | 5 | 10 |
| nominal (27) | 27 | 26 | 26 (the failure is 1327 seed 0) |

Halfway interpolation recovers half of the 1327 loss and is level with base pooled. No comparison is significant at n = 108. Exploratory only.
"""
t = t[:i] + D20 + t[j:]

# ---------------------------------------------------------------- R-058 heading, status, index
sub1("## R-058 — PARTIAL (r = 4 done; r = 64 pending D18): LoRA rank sweep — r = 4 scores 65/108, level with base and r = 16 within noise",
     "## R-058 — DONE (r = 64 truncated at 1500 steps; its transfer not run): LoRA rank sweep — ranks 4 / 16 / 64 give 65 / 59 / 64 of 108 against base 67; capacity is not the lever")
sub1("**Run** r = 4 2026-09-29 23:32 → 09-30 02:36; r = 64 killed 09-30 10:57 · **Status** PARTIAL — r = 64 PENDING (D18) (scored 2026-10-06)",
     "**Run** r = 4 2026-09-29 23:32 → 09-30 02:36; r = 64 trained to opt 1500 (killed 09-30 10:57), evaluated 2026-10-06 · **Status** DONE, with a deviation for r = 64 (r = 4 scored 2026-10-06; r = 64 scored 2026-10-07)")
sub1("| PARTIAL — r = 64 PENDING (D18) | r = 4: 65/108 vs base 67 and r = 16 59 (paired p 0.86 and 0.38). r = 64 OOM-killed at opt ~1900; checkpoints 500–1500 not evaluated |",
     "| DONE (r = 64 deviates: 1500/2000 steps, transfer not run) | Ranks 4 / 16 / 64: 65 / 59 / 64 of 108 vs base 67 (paired p 0.86 / 0.24 / 0.74); scene 1327 2 / 4 / 3 of 12 at every rank; capacity is not the lever |")

# ---------------------------------------------------------------- R-058 result block -> replace
i = t.index("### RESULT (partial), 2026-10-06 (r = 4 run")
j = t.index("\n---\n", i)

R058 = r"""### RESULT, 2026-10-06 / 10-07 (r = 4 run 2026-09-29 23:32 → 09-30 02:36; r = 64 evaluated 2026-10-06; scored by SFT) — no rank beats base; ranks 4 / 16 / 64 give 65 / 59 / 64; the scene-1327 regression appears at every rank

Paired counts and per-scene numbers were re-derived by SFT from the manifests, joined on the start key (verified). Training-log figures are read from `runs/overnight/SUMMARY_r056fix.txt` and `vla_train_meta.json`.

**r = 4.**
- Training: `runs/r058_r4_train`, 113 min. Schedule verified (max LR error 1.4e-20). Peak 6.08 GiB allocated / 6.46 GiB `nvidia-smi`. 4,784,128 trainable parameters.
- Evaluation: `runs/r058_r4/`.
- Selection: validation 5, 5, 4 and 5 of 10, so opt500 was selected on the closest-approach tie-break.

**r = 64: deviation from the registered design.**
- *Resident training ran out of CUDA memory at start* (09-30 02:36). The cache-equivalence gate passed exactly (relative difference 0.0), and training resumed from cached VLM features. As registered, that switch is not a deviation.
- *The truncation is a deviation.* The kernel OOM-killed the cached run on 09-30 at 10:57, at optimizer step ~1900/2000 (unit peak 19.9 GB). The last checkpoint is **opt1500, not annealed:** LR 1.6e-5, against 6.8e-11 at opt2000 for r = 4 and r = 16. The registered "same optimizer steps" does not hold, so r = 64 is a shorter, unannealed arm.
- *Selection:* among the three surviving checkpoints, validation scored 4, 4 and 6 of 10 at opt 500 / 1000 / 1500, so opt1500 was selected. Selection is near random at n = 10.
- *Evaluation* (2026-10-06, user decision D18; `runs/r058_r64/`): held-out 108 and nominal 27 completed.
- **Transfer was not run.** The transfer phase hit a CUDA OOM on its first forward (2026-10-06 16:28, 7.39 of 7.53 GiB), and **by the user's decision (~16:40) it was skipped with no code fix.** impl's hypothesis, two LiberoEnv EGL contexts held by `r044_reverse.py`, is **unverified** and was not patched. So `transfer.retrained` and `retrained_median_W` are null for r = 64.

| readout | base | r = 4 | r = 16 (R-056) | r = 64 (opt1500) |
|---|---|---|---|---|
| held-out success (108) | 67 (0.62) | 65 (0.60) | 59 (0.55) | **64 (0.59 [0.50, 0.68])** |
| paired vs base (gained / lost, p) | | 14 / 16, 0.86 | 14 / 22, 0.24 | 17 / 20, 0.74 |
| paired vs r = 16 | | 19 / 13, 0.38 | | 16 / 11, 0.44 |
| gap closed | | −0.05 | −0.20 | −0.07 |
| band 0.2–0.3 / 0.3–0.4 / 0.4–0.5 | 14 / 13 / 40 | 10 / 8 / 47 | 14 / 8 / 37 | 13 / 7 / 44 |
| scene 1327 (12) | 12 | 2 | 4 | **3** (0 / 9, p = 0.004) |
| excluding 1327 (96) | 55 | 63 | 55 | 61 (17 / 11, p = 0.35) |
| nominal control (27) | 27 | 25 | 26 | 26 |
| W transfer at forward 0, median | 0.88 | 0.82 | 0.84 | **not run** |
| held-out closest approach, median; improved vs base | 5.65 cm | 5.30 cm; 54/108 | 5.72 cm; 45/108 | 5.41 cm; 45/108 |

Per scene (held-out, 12 each; scenes 984, 1030, 1062, 1090, 1132, 1201, 1247, 1282, 1327): r = 64 scores 7, 0, 3, 6, 11, 12, 11, 11, 3. The other arms are in R-056's table. Nominal failures: r = 64 fails scene 1327 seed 0; r = 4 fails 1327 seed 0 and 1030 seed 0.

1. **r = 4 within 5 pp of r = 16 on held-out success: MISSED, narrowly, with r = 4 the better.** +5.6 pp (65 vs 59; paired p = 0.38). A tie within noise.
2. **r = 64 does not beat r = 16 by ≥ 10 pp: HELD.** +4.6 pp (64 vs 59; paired p = 0.44). Caveat: r = 64 is the truncated, unannealed arm (see the deviation above).
3. **r = 64's nominal drop is ≥ r = 4's: MISSED.** r = 64 dropped 1/27 and r = 4 2/27. That is a one-rollout difference at n = 27, so the registered "only a large difference is visible" applies.
4. **W-transfer drop within 0.1 across the three ranks: NOT RUN for r = 64**, by the user's decision of 2026-10-06 ~16:40 (above). On the two ranks measured it holds: r = 16 −0.04, r = 4 −0.06. Neither moves W much.
5. **Closest approach agrees with 1 and 2 in direction: HELD.** r = 4 beats r = 16 in both success and closest approach (5.30 vs 5.72 cm). r = 64 beats r = 16 in both, by a small margin (5.41 cm, +4.6 pp).

**Uninterpretability checks.**
- Step-0 training-path conformance was measured at r = 16 only (R-055 gate 1, exact) and **not repeated at r = 4 or r = 64.** With LoRA's B = 0 at initialisation the adapter is a no-op at any rank, so this is a gap in the record, not a known defect.
- The held-out and nominal sets are R-056's; the base arm is reused from `runs/r056_r16/manifest.jsonl`.
- Cache equivalence passed before r = 64 used the cache.
- None fired. The r = 64 truncation is a recorded deviation, not one of the registered conditions.

**What it decides.** The registered branch **"every rank flat and R-056 flat: the limit is the data, not the adapter; the next comparison is a different source (S3), not more parameters"** applies.
- No rank differs from base pooled (p ≥ 0.24).
- No rank differs from another.
- The scene-1327 grasp regression appears at every rank (2, 4 and 3 of 12).
- The 500-demo follow-up's rank choice is moot until the data source changes (D21–D23).
- D7 (renting a GPU for a full-head fine-tune) gains nothing from this entry.

The caveat on this branch is that r = 64 is a truncated, unannealed arm. A fully trained r = 64 is not expected to change the reading: by R-056's selections, the best checkpoints are early (opt500 and opt1000) anyway.
"""
t = t[:i] + R058 + t[j:]

P.write_text(t)
print("ok")
