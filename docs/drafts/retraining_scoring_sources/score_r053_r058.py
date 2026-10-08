"""Score R-053..R-058 into RESULTS.md (SFT, D17 approved 2026-10-06). One-shot edit."""
from pathlib import Path

P = Path("/home/imerit/Documents/Code/VLA/RESULTS.md")
t = P.read_text()

def sub1(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:90])
    t = t.replace(old, new)

# ---------------------------------------------------------------- index rows
ROWS = {
 "| R-053 | 09-28 | EVAL | Start-pose no-op baseline: how many R-044 rescues are render jitter | PRE-REGISTERED, NOT RUN | 60 rollouts, ~50 min |":
 "| R-053 | 09-28 | EVAL | Start-pose no-op baseline: how many R-044 rescues are render jitter | DONE (scored 10-06) | 4/4 held: N 24/30; N flips 20/24 P-failures, a plain re-run 0/22. The forward-0 rescue is real |",
 "| R-054 | 09-28 | DATA | Rescue-minting pilot: 100 start-pose demos, LeRobot converter, dataset gates | PRE-REGISTERED, NOT RUN | ~1.7 h GPU; converter and runner by impl |":
 "| R-054 | 09-28 | DATA | Rescue-minting pilot: 100 start-pose demos, LeRobot converter, dataset gates | DONE (scored 10-06) | Yield 100/146 = 68.5% (E1 missed narrowly); G1, G4, G6 pass; G7 never signed off; the post-mortem later found 16/100 failed-first-grasp demos |",
 "| R-055 | 09-28 | GATE | GR00T LoRA fine-tune gates: zero-step conformance, overfit, memory on 8 GB | PRE-REGISTERED, NOT RUN | bf16 load, LoRA r16 on head, projectors and VLM frozen |":
 "| R-055 | 09-28 | GATE | GR00T LoRA fine-tune gates: zero-step conformance, overfit, memory on 8 GB | DONE (scored 10-06) | Conformance exact (10/10), 2b and memory pass (6.73 GiB); overfit missed a miscalibrated bar (13.9%) and is informational (user, 09-29) |",
 "| R-056 | 09-28 | EVAL | Start-pose retraining pilot, arm (a): LoRA head, frozen VLM, 100 minted demos + 1:1 replay | PRE-REGISTERED, NOT RUN | reinstates R-046 (a) as a pilot; ~290 eval rollouts |":
 "| R-056 | 09-28 | EVAL | Start-pose retraining pilot, arm (a): LoRA head, frozen VLM, 100 minted demos + 1:1 replay | DONE — UNINTERPRETABLE BY ITS OWN RULE; interpretation provisional (D20) | Held-out 59/108 vs base 67/108 (p = 0.24), gap −0.20; whole drop is scene 1327 at the grasp (12/12 → 4/12); excluding 1327, 55 vs 55 |",
 "| R-057 | 09-28 | EVAL | When is the episode decided: drive-until sweep k ∈ {1,2,4,all} on start pose and camera | PRE-REGISTERED, NOT RUN | 240–300 rollouts, ~3–3.8 h; layout excluded (needs a state-conditional source) |":
 "| R-057 | 09-28 | EVAL | When is the episode decided: drive-until sweep k ∈ {1,2,4,all} on start pose and camera | DONE (scored 10-06) | Start pose k* = 1 (P 6, k1 24, all 19 of 30); camera throughout (k4 23 vs all 30); all six held, E1 at its boundary |",
 "| R-058 | 09-28 | EVAL | LoRA rank sweep on the start-pose pilot: r ∈ {4, 64} vs R-056's r = 16 | PRE-REGISTERED, NOT RUN | ~350 rollouts + 2 trainings; decides rank for the 500-demo run and whether D7 (rent) is needed |":
 "| R-058 | 09-28 | EVAL | LoRA rank sweep on the start-pose pilot: r ∈ {4, 64} vs R-056's r = 16 | PARTIAL — r = 64 PENDING (D18) | r = 4: 65/108 vs base 67 and r = 16 59 (paired p 0.86 and 0.38). r = 64 OOM-killed at opt ~1900; checkpoints 500–1500 not evaluated |",
}
for a, b in ROWS.items():
    sub1(a, b)

# ---------------------------------------------------------------- headings
HEAD = {
 '## R-053 — PRE-REGISTERED, NOT YET RUN: the start-pose no-op baseline (how many R-044 "rescues" are render jitter)':
 '## R-053 — DONE: the forward-0 rescue is real — the nominal first chunk flips 20/24 start-pose failures, a plain re-run 0/22',
 '## R-054 — PRE-REGISTERED, NOT YET RUN: rescue minting pilot, 100 start-pose demos, and the dataset gates':
 '## R-054 — DONE: 100 minted demos at 68.5% yield; the contract, exclusion and round-trip gates pass; G7 never signed off',
 '## R-055 — PRE-REGISTERED, NOT YET RUN: GR00T LoRA fine-tune gates — zero-step conformance, overfit, memory on 8 GB':
 '## R-055 — DONE: GR00T LoRA gates — conformance exact, checkpointing equivalent, 6.73 GiB at batch 1; the overfit bar was miscalibrated (missed at 13.9%, informational)',
 '## R-056 — PRE-REGISTERED, NOT YET RUN: start-pose retraining pilot, arm (a) — LoRA on the action head, VLM frozen, 100 minted demos':
 '## R-056 — DONE, UNINTERPRETABLE BY ITS OWN RULE (interpretation provisional, pending D20): start-pose retraining pilot, arm (a) — held-out 59/108 vs base 67/108; the whole drop is scene 1327 at the grasp',
 '## R-057 — PRE-REGISTERED, NOT YET RUN: when is the episode decided — a drive-until sweep k ∈ {1, 2, 4, all} on start pose and camera':
 '## R-057 — DONE: start pose is decided at forward 0 (k* = 1); camera is decided throughout (k = 4 still 23 vs 30)',
 '## R-058 — PRE-REGISTERED, NOT YET RUN: LoRA rank sweep on the start-pose pilot — r ∈ {4, 64} against R-056\'s r = 16':
 '## R-058 — PARTIAL (r = 4 done; r = 64 pending D18): LoRA rank sweep — r = 4 scores 65/108, level with base and r = 16 within noise',
}
for a, b in HEAD.items():
    sub1(a, b)

# ---------------------------------------------------------------- status lines
def status(new_heading, run, status_word):
    global t
    i = t.index(new_heading)
    j = t.index("**Status** PRE-REGISTERED", i)
    assert j - i < 400
    t = t[:j] + f"**Run** {run} · **Status** {status_word}" + t[j + len("**Status** PRE-REGISTERED"):]

H = list(HEAD.values())
status(H[0], "2026-09-29 00:20–01:04", "DONE (scored 2026-10-06)")
status(H[1], "2026-09-29 01:04–04:48", "DONE (scored 2026-10-06)")
status(H[2], "2026-09-29 04:48–04:55; rerun 11:53; diagnostic 12:24", "DONE (scored 2026-10-06)")
status(H[3], "2026-09-29 19:37–23:32 (fixed-schedule retrain)", "DONE — UNINTERPRETABLE BY ITS OWN RULE; interpretation provisional pending D20 (scored 2026-10-06)")
status(H[4], "2026-09-29 04:55–07:21", "DONE (scored 2026-10-06)")
status(H[5], "r = 4 2026-09-29 23:32 → 09-30 02:36; r = 64 killed 09-30 10:57", "PARTIAL — r = 64 PENDING (D18) (scored 2026-10-06)")

# ---------------------------------------------------------------- result blocks
R053 = r"""
### RESULT, 2026-10-06 (run 2026-09-29 00:20–01:04, scored by SFT) — all four held; the forward-0 rescue is real, not render jitter

`runs/r053_{P,N}_s{0,1,2}`, `runs/r053/score.json`; six jobs, rc = 0.

| noise seed | P failures | N successes | N flips (of P failures) | re-run flips (of the original run's failures) | re-run breaks |
|---|---|---|---|---|---|
| 0 | 8/10 | 8/10 | 6/8 | 0/7 (R-042) | 1 (548) |
| 1 | 9/10 | 8/10 | 8/9 | 0/9 (R-044 seed 1) | 0 |
| 2 | 7/10 | 8/10 | 6/7 | 0/6 (R-044 seed 2) | 1 (313) |
| **pooled** | **24/30** | **24/30** | **20/24 (83%)** | **0/22 (0%)** | 2 |

Median closest approach: P 7.3 cm, N 5.1 cm.

1. **N succeeds on ≥ 21/30: HELD.** 24/30.
2. **N flips more P failures than the re-run, by ≥ 30 pp: HELD.** 83% against 0%.
3. **The seed-0 re-run flips ≤ 3 of R-042's 7 failures: HELD.** 0 of 7.
4. **Closest approach under N beats P on ≥ 20/30: HELD.** 24/30 (7, 9 and 8 per seed).

**Uninterpretability check:** ‖P−N‖ at forward 0 matches R-044 within 0.05 on 30/30 (10/10 at seed 0); every rc is 0. Interpretable.

**What it decides.** Expectation 2 held, so R-054 ran unchanged. A plain re-run flips no start-pose failure (0/22, against 3/7 on camera in R-048), so a minted success at these instances is a real rescue. The re-run did break two successes (548 at seed 0, 313 at seed 2), so outcome noise runs both ways. The claim is limited to these ten instances, which sit at ≤ 10 cm gripper offset except 281 (16.3 cm, not rescued, 0/3; see the note above and R-054's offset curve).
"""

R054 = r"""
### RESULT, 2026-10-06 (run 2026-09-29 01:04–04:48, scored by SFT) — 100 demos at 68.5% yield; registered gates pass except G7, which was never signed off; the post-mortem later found defects no gate tested

`runs/r054/{yield.json,gates.json,manifest.jsonl}`. Datasets: `data/r054_full` (100 episodes, 11,567 frames) and `data/r054_train48` (100 × 48 = 4,800 frames).

| band (requested, rad) | attempts | kept | yield |
|---|---|---|---|
| 0.1–0.2 | 38 | 34 | 0.89 |
| 0.2–0.3 | 36 | 28 | 0.78 |
| 0.3–0.4 | 36 | 18 | 0.50 |
| 0.4–0.5 | 36 | 20 | 0.56 |
| **pooled** | **146** | **100** | **0.685** |

1. **Pooled yield ≥ 70%: MISSED, narrowly.** 100/146 = 68.5%.
2. **The 0.4–0.5 band's yield is ≥ 15 pp below the 0.1–0.2 band's: HELD.** 89% → 56% (−34 pp). It is not monotone: 0.3–0.4 (50%) sits below 0.4–0.5 (56%). Those two bands overlap in gripper offset (median 11.2 vs 11.1 cm; see the 09-29 note).
3. **G6 round trip ≥ 9/10 same outcome: HELD.** 10/10, with equal initial states and identical step counts.
4. **G1 on 100%: HELD.** 146/146 records; both datasets have 0 errors.

**Uninterpretability check:** recorded forward-0 N chunk vs a fresh splice, max |Δ| = 0.0 on 3/3; exported state and action bit-equal to the rollout trace on 3/3, in both the full and the 48-step datasets. Interpretable.

**Other gates.** G4 (held-out exclusion): pass, 28 keys, 0 violations in records and dataset. G7 (eyes on): 10 videos rendered to `runs/r054/g7_videos/` and **never signed off by a person** (`pass: null`).

**Found later, not by any gate of this entry** (`docs/R056_POSTMORTEM.html` §4, §5, §10):
- 16 of 100 kept demos contain a failed first grasp. They close at a median 5.7 cm, against 3.4 cm in clean demos. The 48-step truncation keeps 7 of those premature closes and cuts every recovery.
- The 48-frame windows are 91% approach.
- The kept demos have a median 6.1 cm gripper offset, so they concentrate where the base already succeeds.

The converter is sound. The defects come from the success filter and the truncation. G7, had it been signed off, is the one registered gate that could have caught the failed grasps.
"""

R055 = r"""
### RESULT, 2026-10-06 (run 2026-09-29, scored by SFT) — conformance exact, checkpointing equivalent, memory fits; the overfit check missed its miscalibrated bar and is informational

`runs/r055/{gates.json,memory_r16_b1.json,memory_r16_b2.json,overfit_report.json,overfit_diag_1500.json}`.

1. **Conformance ≤ 1e-3 per dimension on 10/10: HELD.** max |Δa| = 0.0 on 10/10, every dimension.
2. **Overfit below 10% of the step-0 loss: MISSED.**
   - As registered (constant LR 1e-4): 13.9% (0.0523 → 0.0073).
   - Under the overnight cosine deviation: 12.1%.
   - The 1500-step diagnostic reached 8.6% at step 500 and 5.3% at step 1500.
   - Informational by the user's decision (amendment 15:15).
3. **Memory ≤ 7.4 GiB at batch 1: HELD.** 6.30 GiB `max_memory_allocated`, 6.73 GiB `nvidia-smi`.
4. **Batch 2 ≤ 7.8 GiB: HELD.** 6.31 / 6.93 GiB. This is a memory readout only (19:11 amendment).

**Gate 2b (checkpointing equivalence): HELD.** The loss is identical. The worst gradient relative error is 0.0083 < 0.01 over 496 tensors, with 0 zero gradients.

The adapter has 19,136,512 trainable parameters, all LoRA, with no non-LoRA tensor trainable.

**What it decided.** Gate 1 passed and memory fits with the VLM resident. So R-056 trained resident at r = 16, with no cached features, and D7 (renting a GPU) did not become blocking. R-056 and R-058 r = 4 training peaks matched this gate: 6.30 / 6.73 and 6.08 / 6.46 GiB. At r = 64, resident training ran out of CUDA memory (see R-058).
"""

R056 = r"""
### RESULT, 2026-10-06 (fixed-schedule retrain, run 2026-09-29 19:37–23:32, scored by SFT) — every gain expectation missed; uninterpretable by the entry's own rule; the whole held-out drop is scene 1327, at the grasp

**The numbers below are final. The interpretation at the end is PROVISIONAL until D20 reports** (approved 2026-10-06, with impl): a fit test of the adapter on its own training starts, plus an achieved-radius check.

**Runs.**
- Training: `runs/r056_r16_train`. Schedule verified: max LR error 1.4e-20 over 2,000 optimizer steps. Peak 6.30 GiB allocated / 6.73 GiB `nvidia-smi`.
- Evaluation: `runs/r056_r16/{manifest.jsonl,score.json,selection.json,transfer/}`. Headline: `runs/overnight/SUMMARY_r056fix.txt`.
- The cycling first training (amendment 17:31) is not this result.

**Selection.** Validation success 5, 6, 3 and 3 of 10 at optimizer steps 500, 1000, 1500 and 2000, so opt1000 was selected. At n = 10 these differences are within noise; the selection does not show a peak.

| readout | base | retrained (opt1000) |
|---|---|---|
| held-out success (108) | 67/108 = 0.62 [0.53, 0.71] | **59/108 = 0.55 [0.45, 0.64]** |
| paired against base | | 14 gained, 22 lost; exact McNemar p = 0.24 |
| gap closed | | **−0.20** |
| band 0.2–0.3 (18) | 14 | 14 |
| band 0.3–0.4 (18) | 13 | 8 |
| band 0.4–0.5 (72) | 40 | 37 |
| nominal control, 9 scenes (27) | 27/27 | 26/27 |
| on-ramekin, reported apart (3) | 3/3 | 3/3 |
| W transfer at forward 0, median (10) | 0.88 | 0.84 |
| held-out closest approach, median | 5.65 cm | 5.72 cm; improved on 45/108 |

1. **Held-out success +15 pp or more: MISSED.** −7.4 pp.
2. **Gap closed ≥ 40%: MISSED.** −20%.
3. **Nominal regression < 5 pp: HELD.** −3.7 pp (26/27). The one failure is scene 1327, seed 0.
4. **W transfer falls by ≥ 0.2: MISSED.** −0.04.
5. **Lowest band gains more than the highest: HELD only formally.** The 0.2–0.3 band gained 0 and the 0.4–0.5 band lost 3/72 (−4 pp). No band gained.
6. **Closest approach improves on ≥ 60%: MISSED.** 45/108 = 42%.

**Registered failure criterion** (less than a third of the gap closed, with R-055's gates passed): **met.** Its registered next step, scaling to 500 demos, is superseded by the user's decision of 2026-09-30: robust data sourcing comes before any further training (D21–D23).

**Uninterpretability checks.**
- **Base held-out vs R-047's curve at the same radii: FIRED.** R-047 predicts 0.46; base's 95% interval is [0.53, 0.71]. **By its own registered rule, R-056 is uninterpretable.** The base and retrained arms ran on identical starts and are paired, so the comparison between them stands as a measurement. What the check removes is the link to R-047's difficulty scale: the held-out set is easier than designed.
  - The post-mortem's account (§7): achieved radius is 0.74–0.78 × requested, and the two held-out directions per scene give very different gripper offsets from R-047's average direction.
  - D20's radius check bears on this.
- **Base nominal ≥ 26/27:** passed (27/27).
- **G4 held-out exclusion:** passed (0 violations).
- **R-055 gates resolved:** conformance, 2b and memory passed; overfit is informational.

**Per scene** (held-out, 12 starts each). Paired counts re-derived by SFT on 10-06 from `manifest.jsonl`, joined on the start key (scene, direction, radius, noise seed), so the rollout_id hygiene bug does not affect them. They match the post-mortem §1. The r = 4 column is R-058's.

| scene | 984 | 1030 | 1062 | 1090 | 1132 | 1201 | 1247 | 1282 | **1327** |
|---|---|---|---|---|---|---|---|---|---|
| base | 6 | 0 | 6 | 7 | 7 | 12 | 8 | 9 | **12** |
| r = 16 | 5 | 0 | 3 | 5 | 8 | 11 | 11 | 12 | **4** |
| r = 4 | 6 | 1 | 5 | 9 | 10 | 11 | 9 | 12 | **2** |

- **Scene 1327:** base 12/12; r = 16 4/12 (0 gained, 8 lost, exact p = 0.008); r = 4 2/12 (p = 0.002).
- **Excluding 1327 (96 starts):** r = 16 55/96 vs base 55/96 (14 gained, 14 lost, p = 1.0); r = 4 63/96 (14 gained, 6 lost, p = 0.12).
- **1327 fails at the grasp, not in the first chunk** (post-mortem §4; analyzer's measurements, not re-measured by SFT). The adapters close the gripper 5.5–5.8 cm above the bowl, against base's 4.2 cm, about 1.5 cm higher. They then cycle through 5–7 regrasps and time out. Nominal 1327 seed 0 also fails under both adapters, so this is a regression on the task, not one specific to start pose.
- **The exploratory "≤ 8 cm offset" drop (27/27 → 21/27) is the 1327 drop plus three 1062 losses.** Every ≤ 8 cm start is joint direction 100, so offset is confounded with scene and direction (post-mortem §1.3). It carries no offset effect.

**Interpretation — PROVISIONAL, pending D20.**

> *[D20 placeholder: impl's fit test (retrained adapter on its own training starts) and the achieved-radius check. To be written when they report. If the adapter fails on its own training starts, the reading below changes from "the data was wrong" to "the pipeline or capacity could not fit it".]*

The current reading follows the post-mortem (§1–§9 analyzer, §10 SFT).
- **Effect.** The pilot did not improve start-pose robustness. Pooled, it did not measurably hurt it either. The one significant effect is a grasp regression on a single task.
- **Cause.** The likely cause is the data, not the adapter. The post-mortem lists five faults:
  - *coverage:* kept demos sit at a median 6.1 cm offset, while 42% of held-out starts are beyond 12 cm, where there are 9 demos;
  - *dilution:* about 83% of the gradient went to labels the base already produces, and each frame 0 was seen fewer than once by the selected checkpoint;
  - *pollution:* 16/100 failed-first-grasp demos with their recoveries truncated, which is the 1327 signature;
  - *phase skew:* 91% approach frames;
  - *borrowed labels:* "act as if at the canonical pose".
- **What R-055 contributes.** Its 1500-step diagnostic shows the pipeline can fit one episode. It does not show that the adapter fit these 100.

**Not run.** WiSE-FT at α = 0.5 (exploratory), D19.
"""

R057 = r"""
### RESULT, 2026-10-06 (run 2026-09-29 04:55–07:21, scored by SFT) — start pose is decided at k = 1; camera throughout; all six held, expectation 1 exactly at its boundary

`runs/r057_{ris,cam}_*`, `runs/r057/score.json`. The start-pose P and k = 1 arms are reused from R-053, as the design allows: the code state matched (`runs/r057/reuse_r053`). All rc = 0.

| row | drive | success /30 | flips vs P | breaks vs P | closest approach (median) | post-hand-back gap / P's |
|---|---|---|---|---|---|---|
| start pose | P | 6 | — | — | 7.3 cm | — |
| start pose | k = 1 | **24** | 20 | 2 | 5.1 cm | **0.53** |
| start pose | k = 2 | 22 | 17 | 1 | 5.2 cm | 0.59 |
| start pose | k = 4 | 21 | 18 | 3 | 5.6 cm | 0.38 |
| start pose | all | 19 | 16 | 3 | 5.6 cm | — |
| camera | P | 7 | — | — | 4.7 cm | — |
| camera | k = 1 | 24 | 17 | 0 | 3.9 cm | 1.38 |
| camera | k = 2 | 23 | 16 | 0 | 4.5 cm | 1.24 |
| camera | k = 4 | 23 | 17 | 1 | 4.5 cm | 0.99 |
| camera | all | **30** | 23 | 0 | 4.4 cm | — |

Decision point k\*: **start pose k\* = 1**; **camera "throughout (> 4 forwards)"**.

1. **Start pose k = 1 within 10 pp of k = 4: HELD, exactly at the boundary.** 24/30 against 21/30 is 3/30 = 10.0 pp, with k = 1 the higher. `score.json` prints `False`. The cause is floating point: `experiments/r057_score.py:167` tests `abs(0.8 − 0.7) <= 0.10`, and 0.8 − 0.7 evaluates to 0.10000000000000009. In counts the difference is exactly the registered 10 pp, which "within" includes. Recorded as held, with this note; the k\* rule, computed separately, also gives 1.
2. **Start pose post-hand-back gap at k = 1 ≤ 0.6 × P's: HELD.** 0.53.
3. **Camera N-all ≥ 27/30: HELD.** 30/30.
4. **Camera k = 4 ≥ 15 pp below N-all: HELD.** 23 vs 30 (−23 pp).
5. **Camera gap ≥ 0.8 × P's after every finite k: HELD.** 1.38, 1.24, 0.99.
6. **Camera monotone in k within one-episode ties: HELD.** 7, 24, 23, 23, 30.

**Uninterpretability checks:** forward-0 arms match R-044 (start pose) and R-048 (camera) on 10/10 for every drive; camera N-all 30/30 ≥ 24; every rc is 0. Interpretable.

**Beyond the expectations.** Start-pose success *falls* as k grows: 24, 22, 21, 19. Following the control episode's time-aligned chunks past forward 0 is worse than handing back. This is the caveat in "What N means": past forward 0, N is not a correction at the current state. The start-pose correction lives at forward 0.

**What it decides.**
- **Start pose k\* = 1:** R-056's 48-step truncation stands, and the 16 driven steps are the whole correction. The post-mortem's objection to the 48-step window, that it cuts grasps in half, concerns what the window contains after the correction, not k\*, and this result does not reopen it.
- **Camera "throughout":** re-render (spec S1) is camera's only source. Rescue minting with drive-until k is not a camera source.
"""

R058 = r"""
### RESULT (partial), 2026-10-06 (r = 4 run 2026-09-29 23:32 → 09-30 02:36, scored by SFT) — r = 4 is level with base and with r = 16 within noise; r = 64 was never evaluated (D18)

**r = 4.**
- Training: `runs/r058_r4_train`, 113 min. Schedule verified (max LR error 1.4e-20). Peak 6.08 GiB allocated / 6.46 GiB `nvidia-smi`. 4,784,128 trainable parameters.
- Evaluation: `runs/r058_r4/{manifest.jsonl,score.json,selection.json,transfer/}`. The base arm is R-056's, on the identical sets (`--base-manifest runs/r056_r16/manifest.jsonl`), as registered.
- Selection: validation 5, 5, 4 and 5 of 10, so opt500 was selected on the closest-approach tie-break.

| readout | base | r = 16 (R-056) | r = 4 |
|---|---|---|---|
| held-out success (108) | 67 (0.62) | 59 (0.55) | **65 (0.60 [0.51, 0.69])** |
| paired against base | | 14 gained / 22 lost, p = 0.24 | 14 gained / 16 lost, p = 0.86 |
| gap closed | | −0.20 | −0.05 |
| band 0.2–0.3 / 0.3–0.4 / 0.4–0.5 | 14 / 13 / 40 | 14 / 8 / 37 | 10 / 8 / 47 |
| held-out excluding scene 1327 (96) | 55 | 55 | 63 |
| scene 1327 (12) | 12 | 4 | 2 |
| nominal control (27) | 27 | 26 | 25 |
| W transfer at forward 0, median | 0.88 | 0.84 | 0.82 |
| held-out closest approach, median; improved vs base | 5.65 cm | 5.72 cm; 45/108 | 5.30 cm; 54/108 |

r = 4 against r = 16, paired on the same 108 starts: 19 succeed only under r = 4 and 13 only under r = 16 (exact p = 0.38). r = 4's two nominal failures are scene 1327 seed 0 and scene 1030 seed 0.

**r = 64: not evaluated.**
- Resident training ran out of CUDA memory at start (09-30 02:36; marker `runs/overnight/r058_r64_resident_oom`).
- The cache-equivalence gate then passed exactly (relative difference 0.0).
- Training from cached features started at 02:37. The kernel OOM-killed it on 09-30 at 10:57, at optimizer step ~1900/2000; the unit's memory peak was 19.9 GB.
- Checkpoints at 500, 1000 and 1500 survive in `runs/r058_r64_cached_train/checkpoints/`, each `schedule_verified`. They are not evaluated.
- **Pending D18:** evaluate opt1500 as is, resume, or retrain with a memory fix.

1. **r = 4 within 5 pp of r = 16 on held-out success: MISSED, narrowly, with r = 4 the better.** +5.6 pp (65 vs 59; paired p = 0.38). Read as a tie within noise; it is not evidence about capacity in either direction.
2. **r = 64 does not beat r = 16 by ≥ 10 pp: PENDING (D18).**
3. **r = 64's nominal drop ≥ r = 4's: PENDING.** For reference, r = 4 dropped 2/27 and r = 16 1/27.
4. **W-transfer drop within 0.1 across the three ranks: HELD so far, on two ranks.** r = 16 −0.04, r = 4 −0.06. r = 64 is pending. Neither rank moves W much, so neither learned the forward-0 invariance (R-056 expectation 4).
5. **Closest approach agrees with 1 and 2 in direction: HELD for 1.** It favours r = 4 (5.30 vs 5.72 cm), as success does. The half that concerns 2 is pending.

**Uninterpretability checks.**
- Step-0 training-path conformance was measured at r = 16 only (R-055 gate 1, exact) and **not repeated at r = 4.** With LoRA's B = 0 at initialisation the adapter is a no-op at any rank, so this is a gap in the record, not a known defect.
- The held-out and nominal sets are R-056's.
- Cache equivalence passed before r = 64 used the cache.

**What it decides so far.** Two of three ranks are flat against base pooled. Rank does not remove the scene-1327 grasp regression: r = 4 has the larger one (2/12) and the better remainder (63/96). This points to the registered branch "every rank flat and R-056 flat: the limit is the data, not the adapter", which is consistent with the post-mortem and the user's 09-30 decision. That branch is not concluded until r = 64 is evaluated or dropped (D18).
"""

def insert_before_sep(heading, block):
    global t
    i = t.index(heading)
    j = t.index("\n---\n", i)
    t = t[:j].rstrip("\n") + "\n" + block + t[j:]

for h, blk in zip(H, [R053, R054, R055, R056, R057, R058]):
    insert_before_sep(h, blk)

P.write_text(t)
print("ok")
