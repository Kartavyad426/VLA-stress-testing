# RESULTS

**The main record of every experiment in this project.** Each entry says why we
ran the experiment, how, what we expected, what happened, and whether it can
still be trusted. Someone reading only this document should be able to follow
the whole line of experimentation.

- Started 2026-09-17.
- Entries R-001 to R-020 were reconstructed from run artifacts, `FINDINGS.md`,
  `PENDING_DECISIONS.md` and session records.
- From R-021 onward, entries are written as the work happens: expectations
  **before** the run, results **after**.

---

## How this document works

### Appending
- Add new entries at the **bottom** of *Entries*, with the next `R-NNN` id.
- Add a row to the **Index**.
- Update **Current state of knowledge** if the result changes it.
- Never edit an old entry's result. If a later experiment invalidates it,
  change only its **Status**, add a line to the **Invalidation log**, and point
  to the entry that invalidated it.

### Status vocabulary
| Status | Meaning |
|---|---|
| `PLANNED` | Designed, expectation written, not yet run |
| `RUNNING` | In progress |
| `DONE` | Finished; result stands |
| `DONE — PARTIALLY INVALID` | Some arms or conclusions no longer hold; see the Invalidation log |
| `INVALID` | Result must not be used |
| `SUPERSEDED` | Still correct, but a later entry answers the same question better |

### Entry types
- **EVAL:** new rollouts on the GPU.
- **ANALYSIS:** re-analysis of stored traces, CPU only.
- **TEST:** a deterministic check of harness correctness.

### Expected result: pre-registration
Every entry records its expectation and **says when it was written**:
- **pre-registered, file:** written in a file before the result existed. Strongest.
- **stated before run:** said in the session before the result, not in a file.
- **reconstructed:** written after the fact. Treat as context, not as a test.

### Statistics conventions
- Success rates use **Wilson 95% CIs**.
- Unpaired comparisons use a **two-proportion z-test**.
- Comparisons on the same init states use the **exact McNemar test** on
  discordant pairs.
- **Noise floor:** at n=100 per arm, single-run differences below ~10 pp are not
  distinguishable from noise. R-009 showed 32/100 episodes flipping between two
  arms with no net effect.
- MINERVA and SmolVLA **sample noise at inference**, so identical settings do
  not give identical episodes. "Paired" means the same init state, not the same
  trajectory.

### Configuration fields (every EVAL entry fills these)
`policy / checkpoint + revision` · `suites / tasks` · `episodes per task` ·
`seed` · `init states` · `render size` · `n_action_steps` ·
`other policy overrides` · `control mode` · `fps` · `MuJoCo` · `venv` ·
`LeRobot sha or fork` · `runner (lerobot-eval | our harness)` ·
`perturbations` · `harness code state`

---

## Current state of knowledge — as of 2026-09-17

1. **The setup can reproduce a published LIBERO number.** MINERVA scores 95.3%
   against 95.75% published, and all four suites fall inside their CIs (R-016).
2. **Our harness adds no detectable bias** beyond ~10 pp: MINERVA shows
   0.0 / +2.0 pp against lerobot-eval (R-014), SmolVLA −6.0 pp, which is
   within noise (R-017). A smaller bias is not ruled out, especially for SmolVLA.
3. **SmolVLA `HuggingFaceVLA/smolvla_libero` has no published LIBERO target.** It
   is a 32-layer (~605M) build, not the paper's 16-layer headline model (R-011).
   Our best config scores 67% (R-003). Against the paper's closest ablation, the
   residual is ~16–23 pp on spatial / goal / long and ~0 on object.
4. **Not the cause of the SmolVLA gap:** render resolution beyond F9's fix,
   `n_action_steps` 1 vs 10 (R-009), MuJoCo 3.3.7 vs 3.3.2 on goal (R-018, not
   significant), eef-state normalisation (R-011 notes).
5. **Every campaign camera-perturbation result before 2026-09-17 is invalid**
   (R-010). With the fix, MINERVA degrades gradually: flat to 10° yaw, 88% at
   20°, 82% at 5° pitch (R-015).
6. **Failure-family labels are unvalidated.** They were order-dependent (76.6%
   of failures fire more than one rule, R-008). **Since 2026-09-17 the classifier
   reports every matching family, not the first.** That removes the ordering
   artefact but not the weak predicates: `manipulation` fires on ~99% of
   failures, and `spatial_reasoning` counts the destination as a distractor. No
   family count is reportable yet. **Diagnoses produced before 2026-09-17 use
   first-match labels; re-mine before comparing.**
7. **libero_goal is now fully instrumented** for mining (R-019, R-020).

---

## Environments registry

| venv | MuJoCo | LeRobot | Python | Used for |
|---|---|---|---|---|
| `.venvs/lerobot` | 3.3.7 | 0.6.2 (`third_party/lerobot` @ `b6ec006`) | 3.12 | SmolVLA, campaign, harness tests |
| `.venvs/lerobot-mj332` | **3.3.2** | same as above; freeze diff = the MuJoCo line only | 3.12 | SmolVLA MuJoCo comparison |
| `.venvs/minerva` | **3.3.2** | MINERVA fork 0.6.1 (`third_party/MINERVA` @ `64c3cc8`, authors' `uv.lock`) | 3.13 | MINERVA |

---

## Invalidation log

| Date | Invalidated | By | What remains valid |
|---|---|---|---|
| 2026-09-16 | R-005 camera-yaw sweep arms, compound / attribution arms, **all 12 manifest rows** | R-010 camera re-aim bug + stale first frame | R-005 nominal arms |
| 2026-09-17 | R-005 cross-arm pairing: arms on one env object ran on different init states | R-013 init-state counter bug | Per-arm rates; not paired comparisons |
| 2026-09-16 | R-005 family counts as originally reported | R-006 segmenter fixes, then R-008 precedence | Raw rollouts (re-mineable) |
| 2026-09-16 | R-002 framing "SmolVLA does not reproduce 87.3%" | R-011 architecture mismatch | The measured 61.3% itself |
| 2026-09-17 | **R-023** (all non-language arms) and the 4-variant harness check: policy received the instruction with the perturbation suffix appended ("…view 0 0 100 2 352 initstate 0") | LIBERO-Plus builds instructions from file names; LeRobot forwards them (docs/LIBERO_PLUS_LEVELS.md §0) | Language-perturbation arm; the fact that the pipeline runs |

---

## Index

| ID | Date | Type | Title | Status | Headline |
|---|---|---|---|---|---|
| R-001 | 09-15 | EVAL | π0.5 fit on 8 GB | DONE | Does not fit: OOM loading weights alone |
| R-002 | 09-15 | EVAL | SmolVLA baseline, lerobot-eval | DONE — PARTIALLY INVALID | 61.3% (framing invalid, number stands) |
| R-003 | 09-15 | EVAL | SmolVLA render 360→256 | DONE | 67.0%, +17 pp on spatial |
| R-004 | 09-16 | EVAL | Language probe | DONE | Object follows language; spatial ignores it |
| R-005 | 09-16 | EVAL | Stress campaign, 965 episodes | DONE — PARTIALLY INVALID | Nominal valid; perturbed arms invalid |
| R-006 | 09-16 | ANALYSIS | TRANSPORT over-firing / segmenter | DONE | Object 260→87; three defects fixed |
| R-007 | 09-16 | ANALYSIS | Where visual_grounding failures ended | DONE | 14 of 333 near a different object |
| R-008 | 09-16 | ANALYSIS | Family precedence sensitivity | DONE | 76.6% multi-rule; families order-dependent |
| R-009 | 09-16 | EVAL | SmolVLA nas=1 vs 10, spatial | DONE | 77 vs 73, p=0.60: no effect |
| R-010 | 09-16 | TEST | Camera perturbation verification | DONE | yaw 1e-6 ≡ yaw 5: bug confirmed |
| R-011 | 09-16 | ANALYSIS | SmolVLA checkpoint architecture | DONE | 32-layer build, no published target |
| R-012 | 09-16 | EVAL | MINERVA smoke | SUPERSEDED by R-016 | 5/5 |
| R-013 | 09-17 | TEST | Harness self-tests (overnight) | DONE | All pass |
| R-014 | 09-17 | EVAL | MINERVA harness parity | DONE | Δ 0.0 / +2.0 pp |
| R-015 | 09-17 | EVAL | MINERVA camera sweep, fixed code | DONE | Gradual: 100 → 88 (yaw 20) |
| R-016 | 09-17 | EVAL | MINERVA full reproduction | DONE | 95.3% vs 95.75% published |
| R-017 | 09-17 | EVAL | SmolVLA harness parity | DONE | 67 vs 73, z=−0.93 |
| R-018 | 09-17 | EVAL | SmolVLA goal, MuJoCo 3.3.2 | DONE | 74% vs 68%, paired p=0.26 |
| R-019 | 09-17 | EVAL | Goal instrumentation re-run | DONE | 235/265 instrumented |
| R-020 | 09-17 | EVAL | Goal task 5 site-lookup fix | DONE | 30/30 instrumented |
| R-021 | 09-17 | EVAL | GR00T N1.7 smoke test | DONE | Fits in bf16 (5.87 GiB); 5/5 on spatial t0 |
| R-022 | 09-17 | EVAL | GR00T harness parity | DONE | 98 vs 97, McNemar p=1.00 |
| R-023 | 09-17 | EVAL | GR00T on basic LIBERO-Plus perturbations | DONE — PARTIALLY INVALID | 53/54 at level 1; instructions contaminated |
| R-024 | 09-17 | EVAL | GR00T on hard LIBERO-Plus variants, through harness | DONE | 33/41; camera and robot-init worst |
| R-025 | 09-18 | EVAL | Contamination A/B: clean vs LeRobot instruction | DONE | 34 vs 33 of 42; no detectable effect |
| R-026 | 09-18 | EVAL | GR00T failure hunt: 623 LIBERO-Plus L5+L4 variants | DONE | 155 failures; robot init state 32.8% |
| R-027 | 09-18 | EVAL | MINERVA on the same variants (non-language) | RUNNING | — |
| R-028 | 09-18 | BUG | Clean-instruction stripper truncated 10% of variants | FIXED | validated on all 8,493 non-language variants |

---

# Entries

## R-001 — π0.5 fit on the 8 GB card

**Date** 2026-09-15 · **Status** DONE · **Type** EVAL · **Source** `FINDINGS.md` F4

**Question (why).** π0.5 is the one checkpoint with a documented LeRobot LIBERO
reproduction (97.5%), so it would be the ideal reference for "is our setup
right". Can we run it here?

**Expected** *(reconstructed).* Marginal. It is a ~3B model on 8 GB.

**Configuration.** `lerobot/pi05_libero_finetuned` · `.venvs/lerobot` · three
tests: lerobot-eval with EGL, lerobot-eval with osmesa, and weights only with no
simulator.

**Result.**
| Test | Outcome |
|---|---|
| lerobot-eval, EGL | OOM: 7.40 GiB in use, 18.75 MiB free |
| lerobot-eval, osmesa | died at OpenGL init, never loaded the model. **Not evidence about VRAM** |
| weights only | OOM during `.to("cuda")`, 7.34 GiB free at start |

**What happened vs expected.** Worse than marginal. The weights alone do not fit.

**Interpretation.** A π0.5 reference needs a ≥16 GB card. This led to choosing
MINERVA as the reproduction anchor.

**Artifacts.** `FINDINGS.md` F4.

---

## R-002 — SmolVLA baseline through lerobot-eval

**Date** 2026-09-15 · **Status** DONE — PARTIALLY INVALID · **Type** EVAL ·
**Source** `FINDINGS.md` F8

**Question.** Does `HuggingFaceVLA/smolvla_libero` reproduce its published LIBERO
numbers (~87.3%) on our machine?

**Expected** *(pre-registered, file: `experiments/repro/TARGET.md`).* Published
~87.3%. Two upstream reproductions got 73.25% and ~67%. A 10 eps/task run is an
ORIENTING screen, not a reported result.

**Configuration.**
| Field | Value |
|---|---|
| checkpoint | `HuggingFaceVLA/smolvla_libero` (snapshot `6721902b`) |
| suites / tasks | all 4 × 10 |
| episodes / seed | 10 per task, seed 1000, init states 0–9 |
| render | **360×360** (LeRobot `LiberoEnv` config default) |
| n_action_steps | **10** (`run_repro.sh` default) |
| control / fps | relative / 20 |
| MuJoCo / venv | 3.3.7 / `.venvs/lerobot` |
| runner | lerobot-eval, batch 1 |
| wall | 8,078 s |

**Result.**
| Suite | Ours | 95% CI | Published |
|---|---|---|---|
| spatial | 56.0 | [46.2, 65.3] | ~90 |
| object | 85.0 | [76.7, 90.7] | ~96 |
| goal | 69.0 | [59.4, 77.2] | ~92 |
| long | 35.0 | [26.4, 44.7] | ~71 |
| **overall** | **61.3** | [56.4, 65.9] | ~87.3 |

**What happened vs expected.** Below published on every suite, and below both
upstream reproductions.

**Validity & caveats.**
- The measured 61.3% stands.
- **The framing "does not reproduce 87.3%" is invalid (R-011).** That published
  number belongs to a different architecture.
- Not mineable: lerobot-eval stores only success and video.

**Artifacts.** `experiments/repro/runs/nas10_seed1000/`

---

## R-003 — SmolVLA render resolution 360 → 256

**Date** 2026-09-15 · **Status** DONE · **Type** EVAL · **Source** `FINDINGS.md` F9

**Question.** The checkpoint declares `[3,256,256]` inputs but LeRobot renders
360×360. Is that mismatch part of R-002's gap?

**Expected** *(pre-registered, file: provenance `purpose` line).* The mismatch
is real. Effect size unknown.

**Configuration.** As R-002, with **one change: render 256×256**. Seed 1000,
10 eps/task, nas=10, MuJoCo 3.3.7, lerobot-eval.

**Result.**
| Suite | 256 | 360 (R-002) | Δ | p |
|---|---|---|---|---|
| spatial | **73.0** | 56.0 | **+17.0** | **0.011** |
| object | 90.0 | 85.0 | +5.0 | 0.284 |
| goal | 68.0 | 69.0 | −1.0 | 0.879 |
| long | 37.0 | 35.0 | +2.0 | 0.768 |
| **overall** | **67.0** | 61.3 | +5.7 | 0.092 |

**What happened vs expected.** A real but partial cause. The only individually
significant gain is on spatial, the suite most dependent on fine visual
discrimination.

**Interpretation.** Fixed going forward. **67% is the current SmolVLA reference
number.** The conformance gate now refuses a render / declared-shape mismatch for
policies without an internal resize.

**Artifacts.** `experiments/repro/runs/res256_nas10_seed1000/`

---

## R-004 — Language probe: does SmolVLA use the instruction?

**Date** 2026-09-16 · **Status** DONE · **Type** EVAL · **Source** `FINDINGS.md` F7,
`runs/language_probe/probe.json`

**Question.** Does SmolVLA condition on language, or ignore it the way published
LIBERO models are reported to?

**Expected** *(reconstructed).* Likely language-insensitive.

**Configuration.** SmolVLA through our harness · 3 tasks each from spatial,
object and goal · 5 seeds per cell · arms: nominal, blanked instruction, swapped
instruction. **Instrument: target identity** — which object the eef approached —
not success rate.

**Result.**
| Suite / tasks | nominal | blank | swapped | redirect fraction | verdict |
|---|---|---|---|---|---|
| object t0–2 | 1.0 | 0.0 | 0.0 | **1.0** | comprehending |
| spatial t2 | 0.8 | 0.0 | 0.0 | 0.8 | comprehending |
| spatial t0, t1 | 0.6 / 1.0 | 0.0 | 0.0 | **0.0** | partial grounding |
| goal t0–2 | 0.8 | 0.0 | 0.0 | — | inconclusive |

**What happened vs expected.** Not language-insensitive. Blanking the instruction
drops success to 0 everywhere.
- On **object**, a swapped instruction redirects the arm to the new target.
- On **spatial t0 and t1**, it still goes to the original object. Success falls
  and it looks like comprehension in aggregate, but it is not.

**Interpretation.** A success-rate design would have misread spatial.
It supports the language branch of F10 (see `FINDINGS_DRAFT_F8_F10.md`).

**Validity & caveats.**
- n=5 per cell.
- Goal is inconclusive.
- The run predates the init-state fix (R-013), so arms within a cell may not have
  shared layouts.

**Artifacts.** `runs/language_probe/`

---

## R-005 — Stress campaign 20260916-0015

**Date** 2026-09-16 · **Status** DONE — PARTIALLY INVALID · **Type** EVAL

**Question.** End to end: per-task control, reproducibility floor, camera-yaw
robustness curve, ddmin attribution over co-perturbed factors, then mining into a
Data Gap Manifest.

**Expected** *(reconstructed).* Degradation with yaw; attribution isolating the
causal knob.

**Configuration.**
| Field | Value |
|---|---|
| checkpoint | `HuggingFaceVLA/smolvla_libero`, nas=10 |
| suites | spatial, object, goal (8 tasks each); long |
| per suite | control 8 tasks × 5 seeds nominal · floor 2×5 on task 0 · yaw sweep 0/5/10/15/20 × 8 tasks × 5 seeds · attribution: {yaw 15, ee_offset_x 0.03, light 0.4} plus three single-knob reverts × 5 |
| episodes | spatial 260, object 260, goal 260, long 185 = **965** |
| render / MuJoCo | adapter default / 3.3.7 |
| runner | our harness (`experiments/e2e.py`), `.venvs/lerobot` |

**Result — nominal (valid).** Pooled nominal + yaw-0 arms, which are the same
camera:
| spatial | object | goal | long |
|---|---|---|---|
| 72.5 | 88.8 | 75.0 | 56.9 |

**Result — yaw sweep (INVALID).**
| suite | nominal | yaw 0 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|
| spatial | 65 | 80 | 10 | 12 | 8 | 12 |
| object | 90 | 88 | 0 | 2 | 5 | 5 |
| goal | 70 | 80 | 12 | 18 | 18 | 8 |
| long | 52 | 64 | 8 | 12 | 8 | 4 |

**What happened vs expected.** A cliff at 5°, then flat. That is not a dose-response.

**Validity & caveats.**
- **All perturbed arms and all 12 manifest rows are invalid (R-010).** The yaw
  knob re-aimed the camera ~12° at any non-zero value.
- **Arms were not paired on layouts (R-013).**
- Families as reported are invalid (R-006, R-008).
- 200/260 goal traces were uninstrumented; fixed in R-019.
- The raw rollouts can still be re-mined.

**Artifacts.** `runs/camp_20260916-0015_*/`

---

## R-006 — TRANSPORT over-firing and the phase segmenter

**Date** 2026-09-16 · **Status** DONE · **Type** ANALYSIS · **Source**
`IMPLEMENTATION_OVERSIGHTS.md` O1–O4

**Question.** Why did the TRANSPORT (holding) phase fire on 260/260 libero_object
episodes, including 161 "went to the wrong place" failures?

**Expected** *(stated before analysis).* `hold_steps` too short, or an
object-suite gripper behaviour.

**Configuration.** CPU re-mine of R-005's 705 instrumented traces with
`LiberoPhaseSegmenter`.

**Result — three defects plus one structural finding.**
| | Defect | Measured |
|---|---|---|
| O1 | target chosen alphabetically, not in BDDL order | wrong object on 210/260 object and 110/185 long episodes |
| O2 | lift route gated by "fingers closed", which holding prevents | lift reached 1/79 object successes |
| O3 | `hold_steps=6` shorter than finger travel (~8 steps) | closing transient read as a grasp |
| O4 | `CLOSED_M=0.030` not portable across object sizes | by_gripper 0/79 on spatial successes; lift 79/79 |

After fixes: TRANSPORT 260→87 on object, with every success still showing it.

**What happened vs expected.** `hold_steps` was one of four problems. The check
"every success shows TRANSPORT" passed throughout while both routes were broken,
because an OR is validated by whichever branch fires.

**Artifacts.** `experiments/phase_segmenter_test.py`, `IMPLEMENTATION_OVERSIGHTS.md`

---

## R-007 — Where `visual_grounding` failures actually ended

**Date** 2026-09-16 · **Status** DONE · **Type** ANALYSIS

**Question.** The rule is "final eef-to-target > 0.10 m", labelled "went
confidently to the wrong place". Did those episodes go to a different object?

**Expected** *(stated before analysis, vla-81).* Mostly positional, not grounding.

**Configuration.** CPU, post-R-006 segmenter, spatial / object / long.

**Result.**
| suite | n | ended near a different object (<0.10 m) | ended nearest its own target | median grasp attempts (own-target group) |
|---|---|---|---|---|
| spatial | 99 | 12 | 56 | 7 |
| object | 161 | **0** | 147 | **23** |
| long | 73 | 2 | 26 | 5 |

**What happened vs expected.** Stronger than expected. 14/333 (4.2%) match the
label. Every own-target episode attempted grasps.

**Interpretation.** A positional test carrying a semantic name. Recorded as O5 /
taxonomy anti-pattern §6.1; the rename is deferred until the taxonomy revision.

---

## R-008 — Family precedence sensitivity

**Date** 2026-09-16 · **Status** DONE · **Type** ANALYSIS · **By** vla-81, verified
in-session

**Question.** Are family counts properties of the data, or of rule order?

**Configuration.** CPU. `experiments/family_precedence_test.py` computes each
failure's full predicate vector, then applies all 120 rule orderings. 534
featurised failures.

**Result.**
- 409/534 (**76.6%**) fire more than one rule.
- `lost_target + any_attempt` alone covers 336.

| family | shipped | min | max |
|---|---|---|---|
| visual_grounding | 350 | **0** | 397 |
| manipulation | 125 | 125 | **528** |
| spatial_reasoning | 43 | 0 | 44 |
| recovery | 10 | 0 | 24 |

- `manipulation`'s rule fires on 98.9% of failures.
- `ambiguous` is unreachable under any ordering.

**Interpretation.** No family count is reportable. Families must be disjoint by
construction, or reported as predicate conjunctions. See
`docs/TAXONOMY_FAMILY_GUIDELINES.md`.

---

## R-009 — SmolVLA `n_action_steps` 1 vs 10 on spatial

**Date** 2026-09-16 · **Status** DONE · **Type** EVAL

**Question.** The checkpoint ships nas=1; R-003 ran 10. Is nas the residual gap?

**Expected** *(pre-registered, file: `PREDICTION.md`).* Near 73%. SmolVLA paper
Table 13 has spatial at 89 for both nas=1 and nas=10.

**Configuration.** As R-003 spatial, **one change: nas 10→1**. 256, seed 1000,
10 eps/task, init states 0–9, MuJoCo 3.3.7, lerobot-eval batch 1, 4,274 s.

**Result.**
| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nas=1 | 6 | 9 | 8 | 7 | 7 | 10 | 5 | 8 | 10 | 7 | **77** |
| nas=10 | 7 | 8 | 8 | 6 | 7 | 8 | 8 | 6 | 7 | 8 | **73** |

- +4.0 pp, 95% CI [−8.0, 16.0].
- McNemar 18 vs 14 discordant, p=0.597.

**What happened vs expected.** As predicted. nas is not the gap.

**Interpretation.** Falsifies F8 hypothesis #1. Also established the noise floor:
32 of 100 episodes flipped with no net effect.

**Artifacts.** `experiments/repro/runs/res256_nas1_seed1000/` (`PREDICTION.md`)

---

## R-010 — Camera perturbation verification

**Date** 2026-09-16 · **Status** DONE · **Type** TEST · **Found by** vla-7f (code) and
vla-81 (stored extrinsics)

**Question.** Does `camera_yaw_deg` rotate the camera?

**Expected** *(stated before test, vla-7f).* No. The code re-aims the camera at a
guessed point.

**Configuration.** Renders of libero_object and libero_spatial task 0 at yaw 0,
1e-6 and 5, comparing view axis and mean absolute pixel difference after 1 step.

**Result.**
| scene | yaw 1e-6 vs yaw 0 | yaw 5 vs yaw 0 |
|---|---|---|
| object | rotated 12.0°, pitch 31.9→19.9, pixel diff 53.1 | 12.8°, 52.3 |
| spatial | rotated 11.8°, pitch 38.9→50.7, pixel diff 44.1 | 12.3°, 45.8 |

- Also found: `reset()` returned the **pre-perturbation** frame. It called a
  non-existent `_to_lerobot_obs` and fell back silently.
- vla-81, from stored extrinsics: "yaw 5" was a ~13° rotation plus 6–8 cm of
  translation.

**Interpretation.** Invalidates R-005's perturbed arms. Fixed 2026-09-16:
- rigid motion of the original pose about its look point
- guards run on a present knob, including 0
- `_format_raw_obs` for the post-perturbation frame

Verified in R-013.

**Artifacts.** `viz/perturbation_libero_*.png` (before), `viz/perturbation_fixed_*.png` (after)

---

## R-011 — SmolVLA checkpoint architecture

**Date** 2026-09-16 · **Status** DONE · **Type** ANALYSIS · **By** vla-7f, verified
in-session

**Question.** Is `HuggingFaceVLA/smolvla_libero` the model the paper's 87.3%
describes?

**Result.**
- **Layers.** `config.json` has `num_vlm_layers: 0`. The truncation guard runs
  only when >0, so all **32** VLM layers are kept, and the expert is built to
  match. ~605M parameters (safetensors header, parsed independently by vla-7f and
  vla-81).
- **Paper model.** The headline model uses the first **16** layers, ~0.45B
  (§4.3).
- **Normaliser.** Its stats are byte-identical to `lerobot/libero`; the paper
  trained on `physical-intelligence/libero`.
- **State check.** Per-task first-frame eef state matches the dataset within
  ~1 cm, so the state frame and normalisation are ruled out as a cause.

**Interpretation.** No published LIBERO number exists for this checkpoint. The
nearest reference is the paper's full-depth ablation (nas=10: 89/94/91/57),
trained without robotics pretraining.

| | spatial | object | goal | long |
|---|---|---|---|---|
| deficit vs that ablation | 16 [5.4, 26.6] | 4 n.s. | 23 [12.3, 33.7] | 20 [6.4, 33.6] |

Drafted into `FINDINGS_DRAFT_F8_F10.md`; awaiting approval (#20).

---

## R-012 — MINERVA smoke test

**Date** 2026-09-16 · **Status** SUPERSEDED by R-016 · **Type** EVAL

**Question.** Does the MINERVA box run end to end?

**Configuration.** `k1000dai/MINERVA/t05_l1_0.54M` @ revision `1b4fb174` ·
`.venvs/minerva` (MuJoCo 3.3.2) · authors' command on libero_object task 0,
5 eps, batch 5.

**Result.** 5/5, 7.5 s/episode. Timing on libero_10 task 0: 5 episodes in 40 s.

---

## R-013 — Harness self-tests before the overnight queue

**Date** 2026-09-17 03:00 · **Status** DONE · **Type** TEST

**Question.** Are the fixes correct before we spend a night of GPU on them?

**Expected** *(stated before run).* All pass. A failure skips the harness runs.

**Configuration.** `.venvs/lerobot`. Code state saved in
`experiments/repro/runs/overnight_20260917/code_state/` (HEAD `6f039dd3` + a
680-line uncommitted patch).

**Result.**
| Test | Checks | Result |
|---|---|---|
| `camera_perturbation_test.py` | yaw 1e-6 pixel-identical to 0; yaw 5 = exactly 5.000° heading; pitch 5 = 5.000° steeper and camera raised; dist +0.10 = 0.1000 m on-axis; reset frame is post-perturbation — on object and spatial | 20/20 PASS |
| `init_state_pinning_test.py` | seed 3 after three resets == seed 3 fresh (bit-identical); records `init_state_index` | PASS |
| `phase_segmenter_test.py` | O1–O3 regressions | 4/4 PASS |

**Found while preparing this run (fixed before launch).**
- **Init state chosen by a reset counter, not the seed.**
  - Effect on resumed cells: later seeds ran on different layouts than their ids
    claimed.
  - Effect on multi-arm runs: arms on one env object were never on the same
    layouts.
  - Fix: pin `init_state_id = seed % n_init_states`.
- **Temporal ensembling never switched on through the harness.** The policy
  adapter could not pass `temporal_ensemble_coeff`, and tinyflow reads it at
  construction. Added `policy_overrides`, passed as `config=` at load.
- **Adapter could not match MINERVA's 360 render.** Added `obs_size`.

---

## R-014 — MINERVA harness parity

**Date** 2026-09-17 · **Status** DONE · **Type** EVAL

**Question.** Does our harness introduce bias? Same checkpoint, simulator and
init states as lerobot-eval; any systematic gap would be our code.

**Expected** *(stated before run).* No gap beyond noise.

**Configuration.**
| Field | Our harness | Reference |
|---|---|---|
| runner | `experiments/harness_eval.py` | lerobot-eval (R-016 episodes 0–9 per task) |
| checkpoint | MINERVA `t05_l1_0.54M` @ `1b4fb174` | same |
| suites / eps | object + spatial, 10/task, seeds 0–9 → init states 0–9 | same init states (batch 5, sub-env i → init i) |
| settings | 360 render, nas=1, `temporal_ensemble_coeff=0.01`, relative, fps 20 | same |
| MuJoCo / venv | 3.3.2 / `.venvs/minerva` | same |

**Result.**
| suite | harness | lerobot-eval | Δ | z |
|---|---|---|---|---|
| object | 100/100 | 100/100 | 0.0 | 0.00 |
| spatial | 99/100 | 97/100 | +2.0 | +1.01 |

**What happened vs expected.** As expected. No detectable harness bias.

**Validity & caveats.** MINERVA is near ceiling, so this cannot detect a bias
that lowers success by a few points. See R-017.

**Artifacts.** `runs/minerva_harness_parity/`

---

## R-015 — MINERVA camera sweep with fixed perturbation code

**Date** 2026-09-17 · **Status** DONE · **Type** EVAL

**Question.** With the R-010 fix, is yaw 0 identical to nominal, and does
performance degrade gradually rather than cliff?

**Expected** *(stated before run).* yaw 0 == nominal; gradual decline.

**Configuration.** Our harness · MINERVA (as R-014) · libero_object · 10 tasks ×
5 seeds · specs: nominal, yaw 0/2/5/10/20, pitch 5 · MuJoCo 3.3.2.

**Result.**
| nominal | yaw 0 | yaw 2 | yaw 5 | yaw 10 | yaw 20 | pitch 5 |
|---|---|---|---|---|---|---|
| 100 | 100 | 98 | 100 | 100 | **88** [76.2, 94.4] | **82** [69.2, 90.2] |

**What happened vs expected.** As expected. Flat to 10°, decline at 20°. Pitch 5
already costs more than yaw 20.

**Validity & caveats.** n=50 per arm. MINERVA is a scratch CNN without a
pretrained backbone, so its sensitivity is not SmolVLA's or GR00T's.

**Artifacts.** `runs/minerva_harness_camera/`

---

## R-016 — MINERVA full reproduction

**Date** 2026-09-17 03:36–05:46 · **Status** DONE · **Type** EVAL

**Question.** Can our machine and setup reproduce a published LIBERO number?
This is gate 1 of the route to GR00T.

**Expected** *(pre-registered, file: model card).* 96.8 / 99.6 / 97.4 / 89.2,
average 95.75.

**Configuration.**
| Field | Value |
|---|---|
| checkpoint | `k1000dai/MINERVA/t05_l1_0.54M` @ `1b4fb1743f00` |
| command | authors' `lerobot-eval`, one task per invocation (resumable) |
| suites / eps | 4 × 10 tasks × **50** = 2,000 · seed 1000 · init states 0–49 · batch 5 |
| settings | 360 render (fork default), nas=1, `temporal_ensemble_coeff=0.01`, relative, hard reset |
| MuJoCo / venv | **3.3.2** / `.venvs/minerva` (authors' `uv.lock` verbatim, plus cmake as a build tool) |
| power | AC throughout, P4 1,620 MHz |

**Result.**
| suite | ours | 95% CI | published | verdict |
|---|---|---|---|---|
| spatial | 96.6 | [94.6, 97.9] | 96.8 | consistent |
| object | 99.6 | [98.6, 99.9] | 99.6 | consistent |
| goal | 97.4 | [95.6, 98.5] | 97.4 | consistent |
| long | 87.8 | [84.6, 90.4] | 89.2 | consistent |
| **overall** | **95.3** | | **95.75** | |

Weakest tasks: libero_10 task 6 at 58% and task 4 at 78%.

**What happened vs expected.** Reproduced.

**Interpretation.** Gate 1 is met: the machine, simulator, renderer and
evaluation protocol can hit a published number. The SmolVLA gap is therefore not
a general setup failure.

**Artifacts.** `experiments/repro/runs/minerva_repro_full/`

---

## R-017 — SmolVLA harness parity

**Date** 2026-09-17 05:46 · **Status** DONE · **Type** EVAL

**Question.** Is the SmolVLA adapter — used by every campaign — faithful? R-014
cannot detect small bias because MINERVA is at ceiling; SmolVLA at ~70% can.

**Expected** *(stated before run).* ~73%, no gap beyond noise.

**Configuration.**
| Field | Our harness | Reference |
|---|---|---|
| runner | `harness_eval.py` | lerobot-eval `res256_nas10_seed1000` (R-003) |
| suite / eps | spatial, 10/task, seeds 0–9 → init states 0–9 | same init states (batch 1) |
| settings | 256, nas=10, relative, fps 20 | same |
| MuJoCo / venv | 3.3.7 / `.venvs/lerobot` | same |
| gate | PASS | — |

**Result.**
| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| harness | 6 | 9 | 8 | 4 | 8 | 7 | 5 | 6 | 8 | 6 | **67** |
| lerobot-eval | 7 | 8 | 8 | 6 | 7 | 8 | 8 | 6 | 7 | 8 | **73** |

−6.0 pp, z=−0.93.

**What happened vs expected.** Within noise, but in the unfavourable direction.

**Validity & caveats.** Not significant, but not strong evidence of no bias. A
real −5 pp harness bias would look exactly like this. It needs more episodes per
side to resolve (see Open questions).

**Artifacts.** `runs/smolvla_harness_parity/`

---

## R-018 — SmolVLA goal suite under MuJoCo 3.3.2

**Date** 2026-09-17 05:59 · **Status** DONE · **Type** EVAL

**Question.** Four community reruns of this checkpoint score goal at 81 / 83 / 87,
on MuJoCo 3.3.2; we score 68 on 3.3.7. Is the version the cause?

**Expected** *(stated before run, vla-7f).* A lead, not an established defect.
Needs 20+ eps/task to rise above noise.

**Configuration.** As R-003 goal, **one change: MuJoCo 3.3.7→3.3.2**
(`.venvs/lerobot-mj332`; freeze diff is one line). 256, nas=10, seed 1000,
**20 eps/task** (first 10 paired with R-003), lerobot-eval batch 1, one task per
invocation.

**Result.**
| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.3.2, 20 eps | 12 | 20 | 15 | 9 | 19 | 14 | 16 | 20 | 19 | 4 | **148/200 = 74.0** |
| 3.3.2, first 10 | 7 | 10 | 9 | 4 | 10 | 6 | 8 | 10 | 9 | 3 | 76/100 |
| 3.3.7 (R-003) | 6 | 7 | 6 | 7 | 8 | 8 | 5 | 10 | 6 | 5 | 68/100 |

Paired first 10/task: 23 succeed only on 3.3.2, 15 only on 3.3.7, **McNemar p=0.256**.

**What happened vs expected.** Consistent with "a lead, weakly". +6 to +8 pp, not
significant, and still well below the community reruns.

**Interpretation.** The MuJoCo version explains little of goal's gap, if any.
Tasks 9 (4/20) and 3 (9/20) carry most of the deficit.

**Artifacts.** `experiments/repro/runs/res256_nas10_seed1000_mj332_goal20/`

---

## R-019 — libero_goal instrumentation re-run

**Date** 2026-09-17 06:59–07:33 · **Status** DONE · **Type** EVAL

**Question.** R-005 goal had 200/260 traces with no object distances, so the goal
suite could not be mined. Does the region→body resolver fix recover them?

**Expected** *(stated before run).* Articulated-region tasks become instrumented.
Verify `_gt_object_pos_complete`.

**Configuration.** `experiments/e2e.py --suite libero_goal --tasks 8 --seeds 5`,
yaw sweep 0/5/10/15/20 · SmolVLA nas=10 · `.venvs/lerobot-mj332` (3.3.2) · with the
camera and init-state fixes.

**Result.** 235/265 traces instrumented, up from 60/260. The 30 remaining were all
**task 5**, whose target `main_table_stove_front_region` is a MuJoCo **site**, not
a body. Fixed in R-020.

**Validity & caveats.** First run of the fixed camera code on SmolVLA. Its yaw
arms have not been analysed yet.

**Artifacts.** `runs/goal_rerun_mj332/`

---

## R-020 — Goal task 5 site-lookup fix

**Date** 2026-09-17 · **Status** DONE · **Type** EVAL + TEST

**Question.** Does resolving table regions as MuJoCo sites complete goal
instrumentation?

**Expected** *(stated before run).* Task 5 is complete on all traces.

**Configuration.**
- **Code change:** `LiberoEnv._object_body_names` tries bodies first, then an
  exact-name **site**. `_object_pos` reads `site_xpos` for sites.
- **Test:** `experiments/object_resolution_test.py` over all 40 tasks.
- **Re-run:** `harness_eval.py` · SmolVLA nas=10, 256 · libero_goal task 5 ·
  nominal + yaw 0/5/10/15/20 × 5 seeds · `.venvs/lerobot-mj332`. **Laptop on
  battery.**

**Result.**
- **Test: 40/40 tasks resolve every object.** goal task 5 is
  `plate_1_main[body], main_table_stove_front_region[site]`.
- **Re-run: 30/30 traces with complete distances.** The site sits 35 cm from the
  nearest-named body, so resolving it by name to the stove would have been wrong.

Success (n=5, not interpretable): nominal 4, yaw 0: 2, yaw 5: 3, yaw 10: 4, yaw 15: 4, yaw 20: 5.

**Interpretation.** libero_goal is fully instrumented across R-019 + R-020. That
unblocks the language-vs-geometry discriminator (`PENDING_DECISIONS.md` #3).

**Validity & caveats.**
- Timings are unreliable (on battery).
- Note: goal task 2 ("top of the cabinet") resolves to `wooden_cabinet_1_base`,
  not a top surface. The distance reference may be offset. Not yet examined.

**Artifacts.** `runs/goal_task5_site_fix_mj332/`

---

## R-021 — GR00T N1.7 smoke test

**Date** 2026-09-17 · **Status** DONE · **Type** EVAL

**Question (why).** GR00T N1.7 is the project's priority policy. Before any
experiment: does it load on the 8 GB card, is it correctly wired to our LIBERO
environment, and does it produce sensible behaviour?

**Expected** *(stated before run, written here before running).*
1. **As shipped, it will not fit.** The checkpoint config sets
   `model_params_fp32: true`, which casts all 3.144B parameters to 32-bit
   (~11.7 GiB) after loading. Expect out-of-memory.
2. **With `model_params_fp32=false`** the weights stay in the bf16 the backbone
   is built in (~5.9 GiB). **Uncertain** whether it fits alongside activations
   and the renderer on ~7.9 GiB free.
3. **If it runs:** high success on libero_spatial task 0 (published suite
   average 95%). But at n=5 a near-0% result would be the more informative
   outcome, pointing at the known orientation risk: GR00T was trained on
   IPEC-COMMUNITY data, whose image orientation is unverified against LeRobot's
   180° flip.

**Configuration (planned).**
| Field | Value |
|---|---|
| checkpoint | `nvidia/gr00t17-lerobot-libero_spatial-640` @ `32a6ec78` (3.144B params, F32: backbone 1.524B + action head 1.621B — self-contained) |
| base model sidecars | `nvidia/GR00T-N1.7-3B` @ `2fc962b` — statistics / processor / embodiment files only, **no weights needed** |
| backbone sidecars | `nvidia/Cosmos-Reason2-2B` (gated) — config, tokenizer, image processor only (12 MB); weights skipped because LeRobot builds the backbone from config (`load_backbone_weights=False`, `groot_n1_7.py:897`) |
| overrides | `--policy.base_model_path=nvidia/GR00T-N1.7-3B` (checkpoint bakes in NVIDIA's local path, harmless) · `--policy.model_params_fp32=false` (**departure from shipped config**, needed to fit 8 GB) |
| embodiment tag | `libero_sim` (declared in checkpoint) |
| declared inputs | `observation.images.image` [256,256,3], `observation.images.wrist_image` [256,256,3], state [8]; n_action_steps 16, chunk 16 |
| env | libero_spatial task 0, 5 episodes, seed 1000, LeRobot default render |
| MuJoCo / venv | 3.3.2 / `.venvs/groot` (= `.venvs/lerobot-mj332` + `decord`) |
| runner | lerobot-eval, batch 1 |

**Procedure.**
- (a) Load weights only, as shipped and with the fp32 override, and record
  peak GPU memory.
- (b) Run the conformance gate.
- (c) lerobot-eval, 5 episodes.

**Result.**

(a) Memory, weights only, no simulator. GPU 7.53 GiB total, 7.34 GiB free at start:
| Load path | Result |
|---|---|
| as shipped (`model_params_fp32=true`) | **OOM** at 7.2 GiB peak |
| `model_params_fp32=false` | **OOM** at 7.2 GiB — identical: the override does not change the stored F32 weights or `load_bf16=False` |
| cast to bf16 after `from_pretrained` on cuda | **OOM** — `from_pretrained` already places the model on `cfg.device` |
| **load on CPU → cast bf16 → move to GPU** | **LOADED: 3.144B params bf16, 5.87 GiB, 1.38 GiB free** |

(b) Conformance gate: **PASS**, after fixing two gate bugs it exposed.
- It read GR00T's HWC image shape `[256,256,3]` as (256, 3).
- It didn't know GR00T's verified `image2 → wrist_image` alias
  (`processor_groot.py:1579`).
- Warns only on render size (GR00T resizes internally) and on unverified control mode.

(c) lerobot-eval via `experiments/groot_eval_bf16.py`:
- Stock lerobot-eval plus two changes: CPU→bf16→GPU policy load, and
  `--rename_map='{"observation.images.image2": "observation.images.wrist_image"}'`.
  LeRobot's pre-eval key check rejects the env/policy key mismatch before
  GR00T's own alias can apply; passing a rename map also disables that check,
  which our gate covers.
- libero_spatial task 0, 5 episodes, seed 1000, default render (360),
  MuJoCo 3.3.2, AC power.
- **5/5 success**, 94.4 s eval (~19 s/episode), rc=0.

**What happened vs expected.**
1. As-shipped OOM: as predicted.
2. The `model_params_fp32=false` override was expected to fit and did **not**.
   The weights have to be explicitly cast on CPU.
3. It runs and succeeds, so the orientation risk did not show up on this task.

**Interpretation.**
- GR00T N1.7 is usable on this laptop, but **only in full bf16**, which departs
  from the shipped F32-weights config. Every GR00T result from this path must be
  labelled bf16.
- The priority policy is unblocked.
- 5/5 is a smoke result, not an accuracy number.

**Validity & caveats.**
- n=5, one task.
- bf16 numerics are not verified equivalent to the shipped F32 config.
- Render at LeRobot default 360, not the declared 256; GR00T resizes internally.
- **Not yet run through our harness** — GR00T needs its own harness-parity check
  (as R-014 / R-017) before any mining.

**Artifacts.** `experiments/repro/runs/groot_smoke_spatial_t0/`, `experiments/groot_vram_probe.py`, `experiments/groot_eval_bf16.py`

---

## R-022 — GR00T harness parity

**Date** 2026-09-17 · **Status** DONE · **Type** EVAL

**Question (why).** Before mining any GR00T failures: does our harness drive GR00T
the same way lerobot-eval does? Same checkpoint, simulator, settings and init
states; a systematic gap would be our code. GR00T needed two new adapter
features — bf16 CPU-cast loading, and a camera rename map — both first exercised
here.

**Expected** *(stated before run, written here before running).*
- No gap beyond noise.
- **Low power, stated up front:** GR00T is reported at 95% on spatial, so like
  MINERVA (R-014) this run sits near ceiling. It can catch a large wiring defect
  (for example an image, state or rename error collapsing success) but not a bias
  of a few points.

**Configuration (planned).**
| Field | Our harness | Reference |
|---|---|---|
| runner | `experiments/harness_eval.py` | `experiments/groot_eval_bf16.py` = stock lerobot-eval + CPU→bf16 load |
| checkpoint | `nvidia/gr00t17-lerobot-libero_spatial-640` @ `32a6ec78` | same |
| overrides | `base_model_path=nvidia/GR00T-N1.7-3B`, `embodiment_tag=libero_sim`, `n_action_steps=16`, **bf16**, rename `image2→wrist_image` | same (`--rename_map`) |
| suite / eps | libero_spatial, 10 tasks × 10, seeds 0–9 → init states 0–9 (pinned) | 10 tasks × 10, batch 1 → episode i on init state i |
| render | 360 (lerobot-eval default; GR00T resizes internally) | 360 |
| MuJoCo / venv | 3.3.2 / `.venvs/groot` | same |
| code state | uncommitted; saved to run dir | same |

**Procedure.**
- Reference first, one task per invocation (resumable).
- Then the harness.
- Compare per-suite rates with a two-proportion z-test and McNemar on
  init-state pairs.
- Preflight done: the harness ran 2/2 on task 0. Its trace fingerprint records
  dtype, rename_map, overrides, obs_size, init_state_index and MuJoCo.

**Result.** Ran 12:41–13:40, AC throughout. Reference ~4 min/task; harness ~10–14 s/episode.

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| our harness | 10 | 10 | 10 | 10 | 10 | 9 | 10 | 10 | 10 | 9 | **98/100** |
| lerobot-eval | 10 | 9 | 10 | 9 | 10 | 10 | 10 | 10 | 10 | 9 | **97/100** |

- +1.0 pp, z=+0.45.
- **Paired on init states 0–9 per task:** 2 episodes succeed only in the harness,
  1 only in lerobot-eval, exact McNemar **p=1.00**.
- Every harness trace records `init_state_index` 0–9, and its fingerprint includes
  dtype, rename_map and overrides.

**What happened vs expected.** As expected: no gap. Only 3 of 100 paired episodes
disagree.

**Interpretation.**
- Our harness drives GR00T equivalently to lerobot-eval, including the two new
  adapter paths (bf16 load and camera rename). GR00T traces from the harness can
  be mined.
- With MINERVA (R-014) and SmolVLA (R-017), all three roster policies now have a
  harness-parity check.
- GR00T scores 97–98% on spatial through both paths, consistent with the reported
  95%, in bf16.

**Validity & caveats.**
- **Near ceiling.** This excludes a wiring defect (images, state, rename, loading)
  but cannot detect a bias of a few points.
- One suite.
- **bf16 throughout** — not compared against the shipped F32 numerics, which do
  not fit on this GPU.

**Artifacts.** `experiments/repro/runs/groot_parity_spatial/` (reference per task, code state), `runs/groot_harness_parity/`

---

## R-023 — GR00T on basic LIBERO-Plus perturbations (first look)

**Date** 2026-09-17 · **Status** DONE · **Type** EVAL

**Question (why).** First run on LIBERO-Plus, the recommended next benchmark
(`docs/DATA_AND_BENCHMARKS.md`). It replaces our home-grown perturbations with 7
published dimensions. Does GR00T hold up at the easiest level, and does the
pipeline run?

**Expected** *(stated before run, written here before running).*
- High success at difficulty level 1 overall.
- LIBERO-Plus's published ordering for ten other models puts camera viewpoint and
  robot initial state as most damaging, language least. If GR00T follows it,
  camera and robot-init should be the lowest here.
- **This run cannot establish an ordering:** 8 variants per type is a CI of
  roughly ±30 pp. GR00T is not in LIBERO-Plus's published table.

**Configuration.**
| Field | Value |
|---|---|
| checkpoint | `nvidia/gr00t17-lerobot-libero_spatial-640` (bf16 via `groot_eval_bf16.py`, image2→wrist_image) |
| benchmark | LIBERO-Plus `sylvestf/LIBERO-plus` @ `4976dc3`, libero_spatial (2,402 variants; task_id i = catalogue id i+1, all 2,402 matched by name) |
| selection | difficulty **level 1** only; 8 per perturbation type, one per distinct scene where possible; Objects Layout has only 6 at level 1 → **54 variants** (`selection.json`) |
| episodes | 1 per variant (each variant is its own perturbed instance) |
| MuJoCo / venv | **3.3.2** (not the recipe's 3.7.0, inside the task-5 physics break) / `.venvs/libero-plus` = `.venvs/groot` with hf-libero → LIBERO-Plus fork, robosuite 1.4.0 → 1.4.1, + gym, scikit-image, wand |
| isolation | own `LIBERO_CONFIG_PATH` (global `~/.libero` untouched); fork on `PYTHONPATH` |
| runner | lerobot-eval, one invocation per type — **not mineable** (harness lacks LIBERO-Plus support yet) |
| smoke | 3/3 on camera, sensor-noise and light level-1 variants; ~26 s/variant + ~45 s startup |

**Result.** Ran 14:28–14:58.

| Perturbation type (level 1) | Success |
|---|---|
| Camera Viewpoints | 8/8 |
| Robot Initial States | 8/8 |
| Light Conditions | 8/8 |
| Background Textures | 8/8 |
| Sensor Noise | **7/8** |
| Objects Layout | 6/6 |
| Language Instructions | 8/8 |
| **Total** | **53/54 (98.1%)** |

Failure: Sensor Noise task_id 1563 (`pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_26`).

**What happened vs expected.**
- High success, as expected.
- The ordering prediction is **not testable here**: 53 of 54 succeed, so there
  is no spread to rank. Level 1 is too easy for GR00T.

**Second correction, 2026-09-17: instructions were contaminated.** Every non-language
variant's instruction reached GR00T with its perturbation parameters appended
(e.g. "… on the plate view 0 0 100 2 352 initstate 0"), because LIBERO-Plus
derives instructions from file names and LeRobot forwards them. All six
non-language arms therefore also perturbed language. The language arm is
unaffected. See docs/LIBERO_PLUS_LEVELS.md §0.

**Correction added 2026-09-17, after the run.** LIBERO-Plus "difficulty level" is
the number of four reference models (OpenVLA-OFT, π0, π0-fast, UniVLA) that
solved a variant (paper §C.3), not perturbation strength. Level 1 = solved by
all four. So this run tested variants those models already solve; 53/54 was
expected, and is **not** evidence that GR00T is robust to mild perturbations. See
`docs/LIBERO_PLUS_LEVELS.md`.

**Interpretation.**
- The pipeline works end to end on LIBERO-Plus: isolated venv, own config,
  MuJoCo 3.3.2, GR00T bf16.
- GR00T is essentially unaffected by the easiest level of every perturbation
  type. Sensitivity has to be looked for at higher difficulty levels (2–5).

**Validity & caveats.**
- n=6–8 per type.
- Level 1 only.
- One episode per variant.
- **Laptop went on battery at 14:51** for the last two batches (Objects Layout,
  Language). Outcomes are unaffected; timings for those are not comparable.
- Not mineable: lerobot-eval.

**Artifacts.** `experiments/repro/runs/lplus_basic_groot_spatial/` (selection, per-type eval_info, progress, code state)

---

## R-024 — GR00T on hard LIBERO-Plus variants, through our harness

**Date** 2026-09-17 · **Status** RUNNING · **Type** EVAL

**Question (why).**
- R-023 showed only level-1 variants, which the reference models solve, and it
  produced no mineable traces.
- This run collects GR00T **failures with full traces** under perturbation, so
  they can be inspected (episode pages) and mined.

**Expected** *(stated before run).*
- Materially lower success than R-023, since these are L4–L5 variants that one
  or none of the four reference models solved.
- How much lower is unknown: levels encode other models' weaknesses, not GR00T's.
- n=6 per type, so no ordering claim.

**Restarted 2026-09-17 as run `lplus_hard_groot_v2`** after finding the instruction
contamination. The first attempt (16 variants, `runs/lplus_hard_groot_CONTAMINATED_instructions`)
fed perturbation parameters into the instruction and is discarded. v2 uses the
cleaned instruction (env identity `instruction_source: clean_base_scene_v2`).

**Configuration.**
| Field | Value |
|---|---|
| runner | `experiments/harness_eval.py --libero-plus` (**mineable**) |
| checkpoint | `nvidia/gr00t17-lerobot-libero_spatial-640`, bf16, rename image2→wrist_image, nas=16, render 360 |
| selection | libero_spatial, 6 per perturbation type, **L5 first then L4**, one per distinct scene where possible → 42 variants (`experiments/repro/lplus_hard_selection.json`) |
| episodes | 1 per variant, seed 0 → init state 0 |
| MuJoCo / venv | 3.3.2 / `.venvs/libero-plus` |

**Result.** 41 of 42 variants (the fog variant crashed upstream — see below).

| Perturbation type (L4–L5) | Success |
|---|---|
| Background Textures | 6/6 |
| Language Instructions | 6/6 |
| Light Conditions | 6/6 |
| Sensor Noise | 5/5 |
| Objects Layout | 5/6 |
| **Camera Viewpoints** | **3/6** |
| **Robot Initial States** | **2/6** |
| **Total** | **33/41 (80%)** |

All 8 failures are level 5. Episode pages: `viz/lplus_failures/` (8 pages, every
replay exact at 0.0 mm).

**What happened vs expected.** Lower than R-023's 53/54, as expected. The two
worst types — camera viewpoint and robot initial state — are the two the
LIBERO-Plus paper reports as most damaging across ten other models. With n=6 per
type this is consistent with that ordering, **not evidence for it**.

**Two upstream bugs found, both patched locally**
(`experiments/repro/patches/libero_plus_numpy2_fog.patch`):
1. `np.float_` was removed in NumPy 2.0, so **every fog variant crashed**
   (`env_wrapper.py:105`).
2. Fog generates a 256×256 pattern, so it **crashes at any render above 256**
   ("operands could not be broadcast together with shapes (360,360,3)
   (256,256,1)"). LeRobot's own LIBERO-Plus default render is 360.

**Validity & caveats.** n=6 per type, one episode each, level 4–5 only, single
suite, bf16. Levels encode four other models' weaknesses, not GR00T's.

**Artifacts.** `runs/lplus_hard_groot_v2/`, `viz/lplus_failures/`

---

## R-025 — Contamination A/B: clean vs LeRobot's instruction

**Date** 2026-09-18 · **Status** PLANNED · **Type** EVAL

**Question.** LeRobot feeds LIBERO-Plus variants an instruction with the
perturbation parameters appended (`docs/LIBERO_PLUS_LEVELS.md` §0). What did that
cost? Every published LIBERO-Plus number obtained through LeRobot carries it.

**Expected** *(stated before run).* Some drop, since the suffix is out of
distribution for the language encoder. GR00T scored **33/41 clean**. If the
contaminated arm is much lower, LeRobot-based LIBERO-Plus results are
systematically pessimistic; if it is the same, GR00T ignores the junk tokens and
the bug costs little. Either is worth knowing. n=41, so only a large effect is
detectable.

**Configuration.** Identical to R-024 v2 (same 42 variants, GR00T bf16, 360
render) except `--raw-instruction`, which restores LeRobot's string. Env identity
records `instruction_source: task.language_RAW_CONTAMINATED`.

**Result.** **No detectable effect.** Paired over the same 42 variants:

| Arm | Successes |
|---|---|
| clean instruction (R-024 v2) | 34 / 42 |
| LeRobot's contaminated instruction | 33 / 42 |

7 variants disagreed, and they disagreed in *both directions* — 4 that only the
clean arm solved, 3 that only the contaminated arm solved. Exact McNemar on
(4, 3) gives p = 1.0. The difference is noise.

**What happened vs expected.** We expected some drop. There is none we can see.
The most likely reading is that GR00T's language encoder ignores the appended
parameter tokens ("… on the plate view 0 0 100 2 352 initstate 0"): the
instruction is still a complete, correct sentence with junk after it.

**Validity & caveats.** n=42 detects only a large effect — the 95% CI on a
1/42 difference spans roughly ±20 pp, so a real 10 pp penalty would be invisible
here. This does **not** clear LeRobot's LIBERO-Plus path for other policies:
a smaller or more brittle language encoder could be hurt where GR00T is not, and
R-028 showed that a *truncated* instruction is a far more damaging corruption
than a suffix of junk. It clears this bug only for this policy, at this size.

**Artifacts.** `runs/lplus_hard_groot_rawinstr/`, summary in
`experiments/repro/runs/overnight_20260918/summary.txt`.

---

## R-026 — GR00T failure hunt: 623 LIBERO-Plus L5+L4 variants

**Date** 2026-09-18 · **Status** RUNNING · **Type** EVAL

**Question.** Not a rate question. The mining layer's family rules
(`vla_harness/mining/classify.py`) were written against the toy environment and
have never been judged against a large body of real VLA failures. This run
exists to **produce those failures** — many, of many kinds — so the taxonomy can
be adjudicated by hand against rendered episodes (user directive, 2026-09-18:
*"I need a lot of failure cases … we'll see if our rules for taxonomies are
accurate or maybe need more tightness"*).

**Expected** *(stated before run).*
- **Failure yield is the metric.** R-024 got 8 failures from 41 hard variants
  (20%). At that rate 623 variants give ~120 failures; if GR00T is weaker on the
  full L5 set the yield is higher. Either is a usable corpus.
- Camera viewpoint and robot initial state worst, per R-024 and the paper's
  ordering for other models.
- **`manipulation` will fire on nearly every failure** and `any_attempt` on all
  of them by construction (PENDING #15). That is the finding to confirm and then
  act on, not a surprise.
- The design is **deliberately unbalanced** — L5 first, then L4, interleaved by
  type — so a run cut short by the deadline is still type-balanced, but no
  cross-level comparison from it is valid.

**Configuration.** 623 variants = every libero_spatial level-5 (191) and level-4
(432) variant, all 7 types, order in `experiments/repro/lplus_fail_selection.json`
(seed 7). GR00T bf16, nas=16, render 360, clean instructions, 1 episode each,
MuJoCo 3.3.2, mineable. Wall-clock deadline 05:40; resumable.

**Result.** All 623 ran. **468 successes, 155 failures (24.9%)** — the corpus
this run existed to produce.

| Perturbation type | L4 | L5 | Overall (95% CI) |
|---|---|---|---|
| **Robot Initial States** | 47.8% (46) | **22.9%** (70) | **32.8%** [25, 42] n=116 |
| **Camera Viewpoints** | 62.5% (40) | 68.2% (22) | **64.5%** [52, 75] n=62 |
| Objects Layout | 89.5% (153) | 73.3% (45) | 85.9% [80, 90] n=198 |
| Language Instructions | 87.8% (82) | 82.9% (35) | 86.3% [79, 91] n=117 |
| Background Textures | 87.5% (16) | 100% (1) | 88.2% [66, 97] n=17 |
| Light Conditions | 89.5% (38) | 100% (6) | 90.9% [79, 96] n=44 |
| Sensor Noise | 91.2% (57) | 100% (12) | 92.8% [84, 97] n=69 |
| **All** | 82.4% (432) | 58.6% (191) | 75.1% n=623 |

Where the failures come from: robot initial state 78, object layout 28, camera
22, language 16, sensor noise 5, light 4, background 2. Split evenly by level
(79 at L5, 76 at L4) — L4 has more variants, L5 a higher failure rate.

**What happened vs expected.**
- **Robot initial state is the dominant failure mode, by a wide margin** —
  32.8% success, half of it at L5's 22.9%. This is the perturbation that just
  moves the arm's starting joint configuration; the scene and the instruction are
  untouched. It confirms R-024's n=6 hint at n=116.
- Camera viewpoint second (64.5%), also as expected.
- **Sensor noise, lighting and background barely register** (≥88%). GR00T is
  robust to appearance perturbation and fragile to *proprioceptive* perturbation.
- Level ordering holds within every type except camera viewpoint, where L5
  (68.2%) scored *above* L4 (62.5%) — consistent with §1 of
  `docs/LIBERO_PLUS_LEVELS.md`: level is four other models' failure rate, not
  perturbation strength, and the largest camera angles are not at L5.

**What the mining layer said, and why it is the actual point.**

| Family | Count |
|---|---|
| visual_grounding + manipulation | 90 |
| manipulation | 41 |
| visual_grounding + spatial_reasoning + manipulation | 18 |
| spatial_reasoning + manipulation | 6 |

| Predicate | Fires on |
|---|---|
| `any_attempt` | 155/155 (100%) |
| `lost_target` | 108/155 (69.7%) |
| `wrong_object` | 24/155 (15.5%) |

Terminal states: `retry_loop` 146, `failed_grasp_no_retry` 9.

**Four families over 155 failures, and `manipulation` in every one of them.**
`any_attempt` fires on 100% by construction, so it carries no information; the
taxonomy is effectively a two-bit code (`lost_target`, `wrong_object`). 78 of the
failures are robot-initial-state variants and the classifier has no family that
names *starting pose*, so it files them under visual grounding — a label the
evidence does not support, since nothing about the scene's appearance changed.
This is PENDING #15 with data behind it, and it is what tomorrow's hand
adjudication is for.

**Validity & caveats.** One episode per variant, one suite, level 4–5 only, bf16.
Per-type CIs are ±6–12 pp, wide enough to rank types but not to separate the
three appearance types from each other. The L4/L5 restriction means these are
**not** GR00T's success rates on LIBERO-Plus — they are its rates on the subset
four other models mostly failed.

**Artifacts.** `runs/lplus_fail_groot/` (rollouts + `diagnoses.jsonl`),
`experiments/repro/runs/overnight_20260918/{mining,summary}.txt`, 80 rendered
failure episodes in `viz/lplus_fail_groot/`.

---

## R-027 — MINERVA on the same variants: a policy that cannot read

**Date** 2026-09-18 · **Status** RUNNING · **Type** EVAL

**Question.** Do two policies fail on the same variants, and do they fail the
same *way*? The second half is what the taxonomy work needs: if the family
distribution is identical across two very different policies, the families are
describing the perturbation, not the policy — or they are too coarse to tell
them apart.

**Why MINERVA and not SmolVLA.** SmolVLA was dropped at the user's direction
(2026-09-18: *"no smol VLA, not reproducible"*) — our SmolVLA number has never
matched its published one. MINERVA reproduces exactly (R-016: 95.3% vs a
published 95.75%), which is the property that makes a comparison mean anything.

**MINERVA is not language-conditioned.** `TinyflowTaskToIndexStep` maps the
instruction through a fixed 40-entry table recorded at training time and raises
`KeyError` on anything else. Two consequences, one useful and one limiting:
- **Useful:** it is a control for *visual* sensitivity. Where GR00T and MINERVA
  both fail on a camera or noise variant, language cannot be the cause.
- **Limiting:** the 117 language variants **cannot be given to it at all**. Its
  arm is the 506 non-language variants
  (`experiments/repro/lplus_fail_ids_nolang.txt`).

**Expected** *(stated before run).* MINERVA is a 0.54 M-parameter policy trained
on these scenes and is near-ceiling nominally, but has far less visual capacity
than GR00T's 3 B; on strong visual perturbation it should collapse harder. If it
instead tracks GR00T variant-for-variant, the variants are hard for reasons
intrinsic to the scene (an unreachable pose, say), which would be a finding about
LIBERO-Plus rather than about either policy.

**Configuration.** 506 non-language L5+L4 variants, same harness and order.
MINERVA `t05_l1_0.54M`, nas=1, `temporal_ensemble_coeff=0.01`, render 360, MuJoCo
3.3.2, new venv `.venvs/minerva-lplus`. Wall-clock deadline 07:50; resumable.

**Result.** *(to fill after running)*

---

## R-028 — The clean-instruction stripper truncated 10% of variants

**Date** 2026-09-18 · **Status** FIXED · **Type** BUG

**What was wrong.** The fix for LIBERO-Plus instruction contamination (R-023,
`docs/LIBERO_PLUS_LEVELS.md` §0) stripped perturbation suffixes by splitting the
variant name on the **first** occurrence of a marker token. Marker words also
occur inside scene names, so those scenes were cut short:

| Variant name | Instruction sent | Should have been |
|---|---|---|
| `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_table_11` | "pick up the black bowl from" | "pick up the black bowl from table center and place it on the plate" |

**240 of 2,402 libero_spatial variants (10%)** were affected, and the equivalent
share in the other suites. The policy received a truncated command — a *worse*
language perturbation than the contamination the fix was written to remove.

**Fix.** The markers are a suffix grammar, so anchor them at the end and strip
repeatedly; also drop the `KITCHEN_SCENE3_`-style prefix that libero_10 variant
names carry but the LeRobot training datasets do not.

**Validation.** Every one of the **8,493 non-language variants across all four
suites** now strips to a string that is exactly one of the 40 vanilla LIBERO
instructions, checked against the task table shipped inside MINERVA's checkpoint
(recorded from the training datasets themselves). Before the fix, 2,321 did not
match — 240 truncated, the rest from the libero_10 prefix and a `_moved` marker.

**What it invalidates.** R-023 and R-024 ran with the old stripper, so a minority
of their variants carried a truncated instruction; their numbers stand only as
provisional. `docs/libero_plus_variants/*.csv` still carries the old
`instruction_clean` column and needs regenerating. Tonight's R-025/026/027 use
the fixed path.

---

## R-029 — Unperturbed control on the LIBERO-Plus stack: 100/100

**Date** 2026-09-18 · **Status** DONE · **Run** `runs/groot_control_lplus_stack`

**Expectation stated before the run:** GR00T should be at or near its parity
ceiling (R-022: 98/100) when nothing is perturbed.

**Result: 100/100.** Ten base scenes x ten episodes, GR00T N1.7 bf16, canonical
camera `view_0_0_100_0_0`, `initstate 0`, `--base-instruction`.

**Why this run had to exist.** The LIBERO-Plus fork replaces the `libero`
package, so there is no vanilla suite in that venv at all. A canonical-camera,
initstate-0 variant run with `--base-instruction` is the only unperturbed
control obtainable on the same stack the perturbed campaigns ran on. Without it
every perturbation number was being compared against a baseline measured on a
*different* software stack.

It also gives every perturbed episode a matched reference: same scene, same
camera, same instruction, same checkpoint, differing in exactly one parameter.
R-030 depends on that.

---

## R-030 — Robot initial state is a joint-space perturbation, and nothing else moves

**Date** 2026-09-18 · **Status** DONE · **Type** ANALYSIS
**Runs** `runs/lplus_fail_groot` vs `runs/groot_control_lplus_stack`
**Page** `viz/robot_initial_state_vs_canonical.html` (built by
`experiments/init_state_compare.py`)

**What the perturbation actually changes.** All seven joints, dominated by
**J6 (wrist pitch, -15 to -18 deg)** and **J2 (shoulder, +6 to +11 deg)**. The
end-effector offset (83-133 mm, 22-32 deg) is the consequence, not the cause.

**What it does not change: anything else.** Every object pose at t=0 is
**0.00 mm** from the control's. For scale, two different control rollouts of the
same base scene differ by ~21 mm, so 0.00 is a real result rather than a broken
comparison. Base scene, suite, instruction, camera parameters, checkpoint and
dtype were compared field by field and are identical.

**A measurement error worth recording, because it nearly became a finding.**
The first version of this comparison read object poses from
`scene_descriptor.objects` and appeared to show the target bowl displaced by
167-273 mm. That field is written at the **end** of the episode: on a successful
control it shows the bowl already sitting on the plate, so every object looks
"moved" when compared against a failed episode that never touched it. Object
poses must be read from the t=0 observation (`_gt_object_pos`).
`experiments/init_state_compare.py` now refuses to build the page unless every
held-fixed field matches and every object agrees within 1 mm at t=0.

**What the arm does with its 281 steps** (`next to the ramekin`, cm):

| episode | path | closest to bowl | closest to home | steps gripping | outcome |
|---|---|---|---|---|---|
| canonical, initstate 0 | 115.1 | 4.6 | 0.0 | 53 | success @113 |
| initstate 232, L4 | 196.8 | 6.4 | 2.7 | 162 | timeout @281 |
| initstate 412, L4 | 159.3 | 7.3 | 7.9 | 202 | timeout @281 |
| initstate 162, L5 | 197.0 | 7.4 | 3.4 | 170 | timeout @281 |

It is not frozen: it travels further than the success does and closes the
gripper three to four times as often. **It never returns toward the neutral
pose** — the closest it comes is its own starting offset, and it ends 34-42 cm
away.

**The failure is not uniform across base scenes.** On the cookie-box scenes the
arm never gets within 19 cm of the target; on the ramekin scenes it reaches
6-7 cm and still times out. Any family name that covers both has to describe the
*outcome* — work without progress, no recovery to the training distribution —
rather than a single mechanism.

**Bearing on the taxonomy (PENDING #15).** 69 of the 78 rendered
robot-initial-state failures are labelled `visual_grounding`. Nothing visual
changed, to 0.00 mm. That label is wrong, and `manipulation` does not fit either
where the arm never arrives. The predicate that separates these cases —
*never approached the target* — is already computable from `_gt_eef_to_object`.

**Open, and cheap:** a scripted homing prefix before handing control to the
policy would make the perturbation a no-op by construction, so it is not a
benchmark fix. As a **diagnostic** it is worth running: if homing recovers most
of the 78, the deficiency is precisely "out of distribution at t=0, with no
mechanism to get back in". `vla_harness/policies/scripted.py` already exists.

---

## R-031 — Our per-category rates against the LIBERO-Plus paper

**Date** 2026-09-18 · **Status** DONE · **Type** COMPARISON
**Source** LIBERO-Plus, arXiv:2510.13626, Table 1 (read verbatim from the PDF)

**Expectation:** the paper's Finding 2 is that camera viewpoint and robot
initial state are the two damaging perturbations while lighting, background and
noise are superficial. If our harness measures what theirs does, GR00T should
show the same ordering despite being a model they never evaluated.

**It does.** Ours, on the L4+L5 subset:

| category | GR00T | MINERVA |
|---|---|---|
| Robot Initial States | **32.8%** | 37.9% |
| Camera Viewpoints | **64.5%** | **6.5%** |
| Objects Layout | 85.9% | 69.2% |
| Language Instructions | 86.3% | — |
| Background Textures | 88.2% | **0.0%** |
| Light Conditions | 90.9% | 27.3% |
| Sensor Noise | 92.8% | 40.6% |

The paper's headline is "95% to below 30% under modest perturbations". Our
control is **100/100 to 32.8%** on robot initial state.

**A split in Table 1 that our two policies reproduce.** The strong models
(OpenVLA-OFT 59.7 camera / 37.2 robot, pi0 15.8/6.6, pi0-fast 66.4/24.8,
RIPT-VLA 58.3/36.7) are hurt more by robot initial state than by camera. The
weak ones (OpenVLA 1.1, Nora 4.0, WorldVLA 0.3, UniVLA 4.3 on camera) are
annihilated by camera specifically. **GR00T sits with the strong group**
(robot 32.8 < camera 64.5); **MINERVA sits with the weak group** (camera 6.5,
background 0.0). That is a better-grounded version of R-027's claim that visual
robustness is what the 3 B stack buys.

**Turning the difficulty prior into a baseline.** Table 1's columns are over all
1,680 variants per category; ours are the hardest two levels only, so the two
cannot be compared column to column. But L4 means "solved by 1 of 4 reference
models" and L5 "0 of 4", so on *our exact variants* the four reference models
(OpenVLA-OFT, pi0, pi0-fast, UniVLA) average **17.3%** by construction:

| | GR00T | ref avg, same variants | gap |
|---|---|---|---|
| Robot Initial States | 32.8% | 9.9% | +22.8 pp |
| Camera Viewpoints | 64.5% | 16.1% | +48.4 pp |
| Sensor Noise | 92.8% | 20.7% | +72.1 pp |
| **overall** | **75.1%** | **17.3%** | **+57.8 pp** |

MINERVA is +27.2 pp overall but **negative** on background (-23.5) and camera
(-9.7).

**Caveats, and the derived baseline is the weakest part.** The 17.3% treats a
per-instance binary — probably a single trial per model — as a success *rate*,
so it is a point estimate of something noisy. It is still the only comparison
that is matched variant-for-variant, which Table 1 is not. Separately, the paper
does not state whether its evaluations used the contaminated file-name
instructions (`docs/LIBERO_PLUS_LEVELS.md` §0); that would matter most for its
Finding 3 on language. And some of our cells are thin — Background Textures is
n=17.

---

## R-032 — pi0 family: pi0.5 does not fit, pi0-FAST is blocked on its action tokenizer

**Date** 2026-09-18 · **Status** pi0.5 ANSWERED / pi0-FAST OPEN

**pi0.5 does not fit, and this time the measurement counts.** OOM at
**7.40 GiB in use of 7.53 GiB available**, twice, on a card verified idle
(<400 MiB) before each probe. The 14:53 probe that produced the same verdict
does **not** count — an orphaned eval held 6.9 GiB at the time (handoff advice #5). Two clean measurements now agree with the contaminated one, which is
the outcome that made checking it feel unnecessary and was exactly why it needed
checking.

**pi0-FAST never reached the GPU.** `RuntimeError: Failed to load required
tokenizers for PI0FastPolicy initialization`. Two separate causes, one cleared:

1. **`google/paligemma-3b-pt-224` is a gated repo.** `model_info` resolved (a
   gated repo serves metadata) while file downloads returned 403, so the token
   looked valid. Cleared 2026-09-18 by accepting the licence.
2. **The checkpoint's action tokenizer still fails.** `config.json` overrides
   `action_tokenizer_name` to `jadechoghari/fast-libero-tokenizer-mean-std`; the
   LeRobot default `lerobot/fast-action-tokenizer` is a different repo and
   **loads fine**, which is why a standalone check passed and the probe did not.

Ruled out so far: `sentencepiece` is installed (0.2.2), so the reported
"You need to have sentencepiece or tiktoken installed" is a red herring; both
repos' `tokenizer.json` load standalone via `PreTrainedTokenizerFast` (1024 and
2048 vocab, both BPE); `processing_action_tokenizer.py` is byte-identical
between them. The one structural difference is a **`bpe_tokenizer/`
subdirectory present only in the working repo**.

**The tokenizers are not interchangeable.** Different vocab (1024 vs 2048),
different `min_token` (-203 vs -354), and the checkpoint was trained against the
mean-std one. Substituting the default would silently corrupt action
detokenisation and produce a plausible wrong number.

**Resolved 2026-09-18.** `UniversalActionProcessor.attributes == ["bpe_tokenizer"]`
and transformers loads a processor attribute from a **subfolder of that name**.
The checkpoint's tokenizer repo ships those files at the repo root only. Fix:
`assets/pi0fast_action_tokenizer/`, a local copy of the checkpoint's own
tokenizer with the subfolder materialised. Verified to load with the correct
parameters (vocab 1024, `min_token` -203, scale 10.0). No substitution.

**pi0-FAST then loaded, and still does not run here.** Weights fit; the rollout
does not:

| | PyTorch allocated |
|---|---|
| `--dtype bfloat16` (cast every parameter) | 6.44 GiB |
| native, no cast | 6.40 GiB |

**The bf16 cast buys ~40 MiB, not headroom.** The checkpoint is already bf16
(563 BF16 tensors against 40 deliberately F32) and declares `dtype: bfloat16`,
so the cast only touches the small F32 tensors. It was never a fit-vs-numerics
trade-off here; it is pure downside.

OOM during the first rollout at **7.36-7.38 GiB of 7.53**. The gap over the
weights is autoregressive generation — pi0-FAST emits up to
`max_action_tokens=256` text tokens per step, so there is a KV cache — plus the
MuJoCo EGL renderer sharing the card. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
cut fragmentation from 302 MiB to 10.41 MiB and **did not fix it**, which rules
fragmentation out as the cause.

**A hypothesis this run did NOT test.** Under the blanket bf16 cast, generation
produced garbage — `AssertionError: Token sequence does not start with
['Action', ':']`, followed by ~200 junk tokens. Flattening the 40 F32 tensors is
the obvious suspect, but the native-dtype arm OOMed before emitting an action,
so **native generation was never observed working**. The cast is not established
as the cause; both remain open.

**Renderer moved to CPU; it still does not fit. 2026-09-21.** `libosmesa6`
installed, `MUJOCO_GL=osmesa`. Measured on an env-only loop (no policy, so the
number is the renderer's own footprint):

| backend | GPU held | per env step |
|---|---|---|
| `egl` | **633 MiB** | 16.7 ms |
| `osmesa` | **0 MiB** | 287.2 ms (**17.2x slower**) |

With all 633 MiB freed, the smoke **still OOMed** — and PyTorch's own allocation
had grown from 6.66 GiB (EGL) to **7.29 GiB**, of 7.53 total. The freed memory
was consumed entirely by generation and was still not enough.

**A framing error of mine, corrected.** I had read the earlier "64 MiB" from
`Tried to allocate 64.00 MiB` as the size of the remaining deficit. It is not —
it is only the size of the *next* allocation attempt at the moment of death. The
real deficit is **at least 633 MiB and still unmeasured**, because the process
dies before generation peaks. A failed allocation tells you where it stopped,
not how far it had to go.

**Also measured wrong the first time:** `nvidia-smi --query-compute-apps` lists
only CUDA processes, so a MuJoCo EGL graphics context reports as 0 MiB. Total
device-memory delta is the correct metric.

**Status: pi0-FAST does not fit this card as published.** `max_action_tokens`
is tunable and would shrink the KV cache, but it is part of the published eval
config, and changing it to fit our GPU means no longer running the policy as
published — which is the whole point of the gate. It joins pi0.5 as a
rented-GPU candidate (PENDING #25). Every local lever that preserves the
published config has now been tried: native dtype, `expandable_segments`, and a
CPU renderer. Four configurations, four failures.

**Still untested, and now probably untestable here:** whether the blanket bf16
cast is what produced the garbage tokens. No native-dtype arm has ever reached
generation, so the hypothesis in this entry remains open rather than refuted.

**`osmesa` is a useful tool regardless**, now installed and measured: it frees
633 MiB for any job that is VRAM-bound rather than time-bound. At 17.2x slower
rendering it is not a default — on the 623-variant campaign it would add roughly
14 hours.

---

## R-033 — The precision A/B reported rc=0 and produced nothing

**Date** 2026-09-18 · **Status** BUG · **Run** `runs/lplus_fail_minerva_bf16`

`precision_ab.sh` logged `precision A/B done rc=0` **19 seconds** after starting
and wrote zero rollouts. `arms.jsonl` holds the same `rollout_id` three times —
three attempts — and `diagnoses.jsonl` is empty.

The real error:

```
RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same
```

MINERVA's image path never casts observations to bf16, so `conv2d` receives
float input against a bf16 bias. **MINERVA bf16 is broken, not merely untested.**

**Two defects, and the second is the dangerous one.** The cast is a small fix.
The wrapper exiting 0 after three crashed attempts is the harness telling us a
run succeeded when it never produced a row — handoff advice #4 with a clean exit
code on top. A failing eval must fail loudly.

**What stays open.** R-022's question — whether bf16 changes anything under
perturbation — is still unanswered and cannot be answered this way until the
cast is fixed.

---

## R-034 — Three harness bugs found by trying to run a drop-in checkpoint

**Date** 2026-09-18 · **Status** FIXED · **Type** BUG

Each of these was invisible until a checkpoint that was not GR00T came through,
and each is the same disease: **what the code consumes is not what it declares.**

**1. The adapter silently discarded a checkpoint's shipped rename map.**
`_load()` passed `"rename_observations_processor": {"rename_map": dict(self.rename_map)}`
unconditionally, so omitting `--rename-map` overwrote the checkpoint's own map
with `{}`. pi0-FAST ships
`observation.images.image -> observation.images.base_0_rgb` and
`observation.images.image2 -> observation.images.left_wrist_0_rgb`; without them
it raises "All image features are missing from the batch" at the first rollout.
GR00T masked this because it needs an explicit map anyway. **Blast radius
checked before fixing:** of the checkpoints in use, only pi0-FAST ships a
non-empty map (SmolVLA, pi0.5 and GR00T all ship `{}`), so no earlier run is
affected. Fixed: override only when the caller supplied a map.

**2. A setting can live in both the policy config and the shipped preprocessor.**
`action_tokenizer_name` is read by both, by different code. `--override` reached
only the policy config; the preprocessor built its own tokenizer from the
checkpoint's processor config and failed at `runner.py:39`, at the first
rollout rather than at load. Fixed: `--preprocessor-override step.key=value`,
plumbed to `LeRobotPolicy(preprocessor_overrides=...)` and recorded in **both**
`identity()` and `resolved_config()` — a different action tokenizer decodes
different actions, so a run under one must miss the cache of a run under
another. Deliberately generic rather than auto-propagating that one key: other
checkpoints will declare other settings twice.

**3. `setup_pi0.sh`'s smoke targets the vanilla suite from the LIBERO-Plus venv.**
The fork replaces the `libero` package, so `--tasks 0,1,2` resolves to
LIBERO-Plus variant names whose `.pruned_init` files do not exist. It would have
hit pi0.5 too had its probe ever passed. The smoke should run
`--libero-plus --base-instruction` against the canonical variants R-029 used, so
its number is directly comparable to GR00T's on the same scenes.

---

## R-035 — PRE-REGISTERED, NOT YET RUN: balanced radius sweep on robot initial state

**Date registered** 2026-09-22 · **Status** PRE-REGISTERED · **Spec** `res`,
`docs/FRAME_LABEL_METHODOLOGY.md` §8.10

**Registered before the run, because the analysis is about SHAPE and shapes are
easy to see after the fact.**

### Why it exists

LIBERO-Plus robot-initial-state variants are generated by perturbing the
canonical neutral pose along an **isotropic random unit direction in joint
space** at a **fixed radius** — 0.1 to 0.5 rad in blocks of 100, seed 42
(`third_party/LIBERO-plus/libero/libero/envs/robots/new_init.py`, verified by
recomputing `||qpos - original||` over all 500 generated classes: exact, zero
spread).

**The radius is therefore a benchmark-native measure of perturbation MAGNITUDE**
— precisely what the difficulty level is not (`LIBERO_PLUS_LEVELS.md` §1). This
is the first dose-response question we can ask.

The existing 116 robot-init episodes from R-026 cannot answer it. Radius and
level are confounded in that subset (radius predicts level: 26.7% L5 at r=0.2
rising to 67.4% at r=0.5, Mann-Whitney p=0.0076), so **level sits downstream of
radius** and stratifying on it blocks part of the effect being measured.
Marginally, 0.2 against pooled 0.3-0.5 gives Fisher p=0.246, OR 1.98, with
Wilson intervals of [21,72] and [17,44] — heavily overlapping. **Neither
"saturating" nor "flat" is supported on that subset.** Hence a balanced design.

### Design

| | |
|---|---|
| scenes | **2** — `next_to_the_ramekin` and a cookie-box scene |
| radii | 0 (canonical anchor), 0.1, 0.2, 0.3, 0.4, 0.5 |
| episodes | 20 per radius per scene = **240** |
| init states | 20 distinct indices per radius block, fixed recorded seed |
| held fixed | camera `view_0_0_100_0_0`, `--base-instruction`, GR00T bf16, nas=16, obs 360, MuJoCo 3.3.2 |

Two scenes rather than one because R-030 found **the mechanism is not uniform
between them** — the arm reaches 6.4-7.4 cm on ramekin scenes and never gets
within 19 cm on cookie-box scenes. One scene would measure one mechanism and
generalise wrongly, which is exactly the error recorded in R-030.

Radius 0.1 is **absent from all 116 existing episodes** and is where the
marginal shape suggests the action is.

### Pre-registered hypotheses

Exactly three, committed before the run:

1. **Monotone decline** — success falls steadily with radius across the range.
2. **Saturating** — a drop by 0.2-0.3 rad, flat thereafter at ~30%.
3. **Flat above zero** — canonical succeeds, everything perturbed fails at a
   rate independent of magnitude.

**Analysis is a trend test, not pairwise.** At n=40 per radius pooled over
scenes the per-cell CI is about +/-15 pp, so detecting a 16 pp drop pairwise
would need ~150 per arm. Cochran-Armitage across the six ordered radii has far
more power against a monotone or saturating alternative, and the question is
about shape anyway. Report per scene first, then pool; **expect the two scenes
to differ.**

### Why it is worth the GPU

The two live hypotheses prescribe **opposite** post-training data:

- **Saturating** → beyond ~0.2 rad the policy is equally lost everywhere, so
  corrective data must cover the whole ball.
- **Monotone decline** → near-neutral corrective data is worth most.

This is the cheapest experiment that separates them.

---

## R-037 — PRE-REGISTERED: GR00T with a NULL PROMPT on libero_spatial

**Date registered** 2026-09-22 · **Status** PRE-REGISTERED, launching now

**Registered before the run because both outcomes are interpretable and it
would be easy to claim either was expected.**

### The question

LIBERO-Plus Finding 3 reports models are "largely insensitive to language
variations", and our own R-025 contamination A/B was null (34 vs 33 of 42,
exact McNemar p=1.0). Both are consistent with the policy **ignoring the
instruction**. Neither tested the limit: what happens with **no instruction at
all**.

**`libero_spatial` is the right suite for this and it is not an accident.** The
suite is built so that language is the *only* disambiguator — several visually
identical black bowls in one scene, distinguished solely by spatial referent
("the black bowl **next to the ramekin**" vs "**on the stove**" vs "**between
the plate and the ramekin**"). A policy that cannot read the instruction has no
way to know which bowl is meant.

### Design

- **Null arm:** 20 canonical LIBERO-Plus variants (`view_0_0_100_0_0`,
  `initstate 0`), 2 per base scene across all 10 base scenes, 5 episodes each =
  **100 episodes**, `--instruction-override ""`.
- **Control arm:** already measured. `runs/groot_control_lplus_stack`, 100/100,
  identical stack and settings, and its 10 task ids are the first of each pair
  here. **No control GPU time is needed.**
- Held fixed: GR00T N1.7 bf16, nas=16, obs 360, canonical camera,
  `--base-instruction` on the control arm, MuJoCo 3.3.2.

### Pre-registered outcomes, with what each would mean

| result | reading |
|---|---|
| **near 100%** | Language is not merely de-emphasised, it is **unused**. The policy is solving `libero_spatial` from vision and proprioception alone, which means it must be resolving "which bowl" by something other than the instruction — position prior, saliency, or a scene-to-target association memorised in training. Strongest possible version of Finding 3. |
| **drops toward chance among candidates** | Language **is** being read and used. That would contradict Finding 3 and R-025 — and would mean our null A/B was underpowered rather than correct. |
| **collapses to ~0** | Neither. An empty string is off-distribution for the tokenizer, and the failure would be about degenerate conditioning rather than about language. **This is the outcome that proves nothing**, and it must not be reported as "language matters". |

**Expectation, stated plainly: near 100%.** R-025, LIBERO-Plus Finding 3, and
`res`'s survey (VLM4VLA: general VLM competence poorly predicts control) all
point the same way. Recording it so that if the score drops, the miss is on
record.

### The trap this design has to survive

The third row above is the one to guard. If the score collapses, the discriminator
is **which object the arm approaches**, not the success rate —
`experiments/language_probe_run.py` already makes this argument for substituted
instructions and it applies here. A policy that ignores language and goes to its
usual target, and a policy whose conditioning is degenerate and goes nowhere,
both score 0. `_gt_eef_to_object` and `_gt_nearest_object` are recorded per step
and separate them.

---

## Open questions carried forward

- **SmolVLA harness bias under ~10 pp** (R-017): needs ~50 eps/task per side on
  one suite to resolve.
- **The SmolVLA residual gap:** language disambiguation vs geometric imprecision
  (F10 draft). The discriminator is now runnable on R-019 + R-020 traces.
- **Goal tasks 3 and 9 carry most of goal's deficit** (R-018). Why?
- **Goal task 2's target resolves to the cabinet base** (R-020).
- **Failure taxonomy** is to be redesigned with consumer input
  (`docs/TAXONOMY_FAMILY_GUIDELINES.md`). R-030 gives it a concrete starting
  point: `visual_grounding` is measurably wrong on 69 of 78 robot-initial-state
  failures, and *never approached the target* is already computable.
- **Does a scripted homing prefix recover the robot-initial-state failures?**
  (R-030). Diagnostic, not a benchmark fix.
- **pi0-FAST's action tokenizer** (R-032): the `bpe_tokenizer/` subdirectory is
  the remaining lead.
- **MINERVA bf16 cast**, and a harness that fails loudly (R-033).
- **R-005's perturbed arms** need a re-run with fixed code before any manifest
  row ships.
- **GR00T** (the priority): VRAM fit (#16), then a harness-parity check for GR00T
  itself.

---

## R-036 — NEGATIVE: GR00T's VL embeddings do not separate its own failures; the state pathway does

**Date** 2026-09-22 · **Status** RUN, 40 episodes · **Run** `runs/embed_derisk`
· **Spec** `docs/EXP_EMBEDDING_OOD.md` §4A, capture design
`docs/superpowers/specs/2026-09-18-embedding-capture-design.md`

**This contradicts pre-registered expectation #1, which was held at HIGH
confidence.** §5 forbids editing expectations after the fact, so it is recorded
as a miss rather than revised.

### What was run

The §2.3 de-risk gate: 40 LIBERO-Plus `libero_spatial` episodes, GR00T N1.7
(`gr00t17-lerobot-libero_spatial-640`, `n_action_steps=16`, bf16, obs 360),
k=4 resampling, embeddings captured per model forward. Task ids were sampled 20
from ids that failed and 20 from ids that succeeded in `lplus_fail_groot`
(`experiments/repro/lplus_derisk_ids.txt`, seed 0) — that biases SELECTION
toward a balanced split and borrows no labels. **Labels come from this run**,
as required: the denoise loop starts from an unseeded `torch.randn`
(`groot_n1_7.py:657`), so a re-run does not reproduce its own split.

Outcome: **22 success / 18 fail, 495 model forwards.**

### Result

| signal | ref rate | success | fail | separation | p |
|---|---|---|---|---|---|
| `vl_encoder_mean` (S1) | 5.1% | 5.1% | 4.9% | **0.97x** | 0.70 |
| `vl_encoder_max` (S1) | 5.0% | 5.0% | 1.9% | **0.37x** | 0.83 |
| `vl_normed_mean` (S1.5) | 5.1% | 5.1% | 2.2% | **0.42x** | 0.70 |
| `vl_adapted_mean` (S2) | 4.9% | 4.9% | 5.2% | **1.08x** | 0.46 |
| `vl_adapted_max` (S2) | 5.6% | 5.6% | 0.0% | **0.00x** | 0.035 |
| `state_encoded` (S0e) | 3.7% | 3.7% | **47.8%** | **12.95x** | **4.1e-06** |

**Every vision-language signal is flat.** The only signal that separates is the
proprioceptive one, after the per-embodiment encoder.

`vl_adapted_max` at p=0.035 runs the WRONG way — failures score *less* OOD than
successes. Across six signals at n=40 that is multiplicity, not a finding, and
it is recorded here so it is not later quoted as one.

### ⚠ CORRECTION, 2026-09-22 — THE SAMPLE COULD NOT HAVE TESTED THE HYPOTHESIS

Raised by `primary`, verified independently here against
`third_party/LIBERO-plus/.../task_classification.json` (2,402 libero_spatial
variants, task ids index it directly). **The conclusion originally written below
was broader than the sample supports.**

| category | n | failures |
|---|---|---|
| Robot Initial States | 12 | 8 |
| Objects Layout | 11 | 4 |
| Language Instructions | 7 | 1 |
| Sensor Noise | 4 | 2 |
| Light Conditions | 4 | 2 |
| **Camera Viewpoints** | **2** | **1** |

**Camera Viewpoints has n=2.** That is the category the capture was specifically
aimed at, and the one where `res` (§8.9) holds the upstream-vs-action-head
question to be well posed. The sample contains essentially none of it.

**23 of 40 episodes are Robot Initial States + Objects Layout**, carrying 12 of
the 18 failures — the two categories where R-030 and `res` give independent
reason NOT to expect a VL-pathway signal (family-C skill gap: the arm reaches
the object and cannot close the loop).

**So what was actually measured is: the VL pathway does not separate failures we
already believed were not VL failures.** That is a much weaker claim than "S2
does not separate", and it means **the §2.3 gate was neither met nor failed — it
was never tested.** The seed-0 draw was taken from a corpus that is only ~10%
camera, and stratifying 40 episodes cannot fix it: the decisive follow-up is a
CATEGORY-BALANCED de-risk at the same cost (~20 camera-viewpoint against 20
matched controls), not the 3.5 h full capture.

### Per-category rates, computed from this capture

Cells with n<5 are listed for completeness and carry no weight.

| category | `state_encoded` fail% | `vl_adapted_mean` fail% |
|---|---|---|
| Robot Initial States (8 fail) | 60.4% | 7.6% |
| Objects Layout (4 fail) | 34.7% | 5.6% |
| Sensor Noise (2 fail) | 52.8% | 2.8% |
| Light Conditions (2 fail) | 52.8% | 0.0% |
| Camera Viewpoints (1 fail) | 27.8% | 0.0% |

**This disconfirms a hypothesis `primary` offered when raising the correction** —
that `state_encoded`'s 12.95x would prove to be carried by the robot-init and
layout episodes, corroborating the skill-gap reading. It is not: the state signal
is **broad across every category**, 27.8%–60.4%. That is more consistent with
the standing causation caveat — a failing episode ends up in an unusual state
almost by definition — than with a perturbation-specific mechanism. Recorded
because it was predicted before it was computed and came out the other way.

### Three things this result is NOT

1. It is not "the embedding space beats S0's 6.9x".** Different unit (per
MODEL FORWARD, ~5-8 per episode, against S0's per env step over whole
episodes), different run, different reference set. A like-for-like number needs
S0 recomputed on these episodes and forwards. The analysis tool was changed to
stop printing that comparison.

**2. It is not evidence that vision carries no information about failure.**
`state_encoded` is the STATE pathway. Its winning says proprioception carries
the signal — it says nothing about vision beyond the VL rows being flat.

**3. ⚠ IT IS BOUNDED BY TOKEN POOLING — TWO SEPARATE DEFECTS.**

**3a. The pooled vector is MODALITY-MIXED.** `taps.py` pools with
`t.mean(axis=1)` over the whole VL sequence, but that sequence is image tokens
AND text tokens interleaved: `groot_n1_7.py:451` builds
`image_mask = input_ids == image_token_id` and `AlternateVLDiT` uses it to drive
the two streams through separate attention masks. The model treats them as
separate channels; the capture averages them into one vector. Worse, the
MIXTURE RATIO VARIES BY EPISODE — instruction length ran 14 to 23 words across
these 40 episodes, so the text share of the mean drifts for reasons unrelated to
what the camera saw. That injects episode-to-episode variance into the feature
which is uncorrelated with success, **diluting a real signal rather than
creating a false one**. Raised by `primary`, verified here at
`groot_n1_7.py:451/457` and `taps.py`. The fix is cheap and was not done before
this run: pool separately over image and text tokens using a mask already in
scope at the tap.

**3b. The pooling discards WHERE.**
The VL roles are stored mean- and max-pooled across the token axis
(`taps.py`). Spatial information in a ViT-style encoder is carried by WHICH
tokens are active, not by the magnitude of a token-averaged vector. **A null on
pooled S2 is equally consistent with "the representation does not encode the
difference" and with "it does, and pooling discarded it."** This measurement
cannot separate those. Raised by `primary` against `res`'s probe design (§8.12)
and it applies with full force here. Storing both poolings before any result
existed was the right call under §6; the limitation is that neither pooling
preserves *where*.

### Bugs found while producing it, both of which would have faked a result

**A separation of 4.9e7x on the first pass — a division by epsilon.**
`ood_selfref.py` takes `--ref` and `--query` as different runs. Here they cannot
be, because the reference must come from the run being scored (unseeded randn,
above). So every success episode was scored against a cloud containing its own
points: nearest-neighbour distance 0, success rate 0%, ratio = x/1e-9. **The
tell was `ref rate` 5.1% against `success` 0.0% — two numbers describing the
same episodes.** Successes are now scored leave-one-out, and that invariant is
asserted in `tests/test_ood_embedding.py`. **The constraint that makes this
experiment valid is what broke the method inherited from the proprioceptive
tool.**

**An fp16 overflow that would have manufactured OOD hits in a labelled
category.** `vl_encoder` is PRE-LayerNorm and its activations are large by
construction; measured peak 15,296 against the fp16 ceiling of 65,504. Light
Conditions is one of the seven LIBERO-Plus perturbation categories (44 GR00T
instances, R-026), so brighter scenes are generated deliberately, not
hypothetically. An `inf` in a max-pooled feature puts that episode infinitely
far from the reference cloud — in the exact signal being measured, on episodes
from a category that would then have been reported as strongly OOD. Now
promotes to fp32, refuses non-finite values, records per-key dtype.

### What would change the conclusion

**§4C, the signal x perturbation-type table, and it needs no new GPU time** —
the 40 captured episodes carry LIBERO-Plus category labels. `res` found camera
and robot-initial-state robustness are uncorrelated across ten published models
(Spearman +0.09, p=0.803), so a mixed sample can dilute a category-specific VL
signal to nothing. This null is over a mixed sample and does not rule that out.

A per-token or coarse-spatial summary would be needed to close the pooling
caveat, and that is a capture change, not an analysis one.

### Status of the full capture

**NOT RUN.** The §2.3 gate was "check that S2 separates at all before spending
3.5 h". It does not, on a mixed sample. Budget if it is later run, measured not
estimated: 26.3 KB/forward, 2.208 s/forward at k=4 -> 145 MB and 3.46 h for 723
episodes.

---

## R-037 — PRE-REGISTERED, NOT YET RUN: does the perturbation show up at all, and does it separate?

**Date registered** 2026-09-22 · **Status** PRE-REGISTERED · **Spec** this entry

**Registered before the run. §5's convention: expectations written first, and a
wrong one is the finding.**

### Why it exists — R-036 asked only half the question

R-036 asked, of failures: **does the signal separate pass from fail?** It never
asked the prior question: **does the perturbation move the signal at all?**

Those are different, and the second gates the first. If a lighting change does
not move the VL embedding, a null on pass/fail separation in lighting episodes
says nothing about failure prediction — it says the tap, or the pooling, is
deaf. R-036 cannot distinguish "VL is blind to appearance" from "VL saw it and
the modality-mixed mean averaged it away."

### Design — three arms, two questions

| arm | n | composition |
|---|---|---|
| **A** nominal | 20 | control ids, canonical camera, initstate 0 |
| **B** vision | 30 | Camera Viewpoints 16, Light Conditions 8, Sensor Noise 6 |
| **C** telemetry | 20 | Robot Initial States |

Within B and C, ids are drawn balanced pass/fail from `lplus_fail_groot`
outcomes. **That biases SELECTION only; labels come from this run**, because the
unseeded randn means outcomes do not reproduce.

**`--base-instruction` on ALL THREE ARMS.** Not only the control. Two reasons:
it holds text tokens constant so any VL difference is genuinely visual rather
than instruction-length drift; and without it LeRobot passes the variant
FILENAME as the instruction, so the text would encode which perturbation was
applied (O7, R-025). Language is deliberately held fixed and is therefore NOT
tested here.

**Q1 — detection.** Distance from arm A's nominal cloud, regardless of outcome.
**Q2 — discrimination.** Within each arm, pass vs fail, leave-one-out as O8
requires.

The signal x arm matrix is the point, including the OFF-DIAGONAL: the taps are
complementary by construction — S1 cannot see robot state, S0e cannot see
pixels.

### Pre-registered expectations

1. **Arm B will move the VL signals away from nominal (Q1).** *High
   confidence.* Camera, lighting and sensor noise change pixels. **If this
   fails, the capture is deaf and R-036's null is uninterpretable rather than
   negative** — this is the load-bearing prediction of the experiment.
2. **Arm C will move `state_encoded` away from nominal (Q1).** *High.* Robot
   initial state is a joint-space perturbation (R-030).
3. **The off-diagonals will be weak: arm C will barely move the VL signals, and
   arm B will barely move `state_encoded` at t=0.** *Medium.* Complementary by
   construction. Arm B may move state LATER in the episode via behaviour, which
   is a different claim from moving it at the start.
4. **Image-pooled VL will show a larger arm-B effect than the combined pool.**
   *Medium-high.* If modality mixing dilutes, removing it should help. **If the
   two are identical, the mixing was not a material problem and R-036's caveat
   3a should be downgraded.**
5. **Q2 within arm B will be weaker than Q1 within arm B.** *Medium.* Detecting
   that the scene changed is easier than predicting whether the policy will
   cope.
6. **`state_encoded` will separate pass/fail in EVERY arm (Q2), including B.**
   *Medium-high*, and it is the uncomfortable one: R-036's per-category table
   already shows 27.8%-60.4% across all categories. If it holds, `state_encoded`
   is measuring "this episode went wrong" rather than any mechanism, and the one
   positive result in R-036 is a restatement of S0 with the same causation
   caveat.

### What would make this a failed experiment rather than a negative result

Expectation 1 failing. If a camera-viewpoint change does not move a
vision-language embedding, the problem is in the instrument, not the model.
