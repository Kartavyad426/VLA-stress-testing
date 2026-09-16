# FAILURE FAMILY AUDIT — what our taxonomy actually is, how it compares, and how to find out whether it is useful

*Written 2026-09-16. Companion to `docs/FAILURE_MINING_METHODS.md` §3–§4 (the field survey) and
`docs/OUR_MINING_APPROACH.md` (what we do). This document does one thing those do not: it reads
`vla_harness/mining/classify.py` as the definition of our taxonomy — not the docs — compares that to
published family sets, and proposes the experiment that decides whether the families earn their place.*

> **Provenance.** §1–§3 are read from source at `vla_harness/mining/classify.py` (167 lines) and are
> checkable. The campaign counts in §4 are **as reported by the mining session**, not independently
> re-run here. §5–§6 are proposals, not results.

---

## 0. The question this document exists to answer

The client-facing claim is not "we can name failures". It is *"we can tell you what data to collect."*
A family earns its place if and only if **partitioning by it changes which data you would gather, and
gathering that data helps more than gathering the same volume uniformly.** Everything else — κ,
cluster purity, how sensible the names sound — is instrumentation, not evidence.

The field survey reached the same conclusion from the other direction (§3.5): of four ways people
validate taxonomies, only downstream utility can falsify one. §5 below is that experiment.

---

## 1. What our families actually are

Declared in `classify.py:20`:

```python
FAMILIES = ["visual_grounding", "language_grounding", "spatial_reasoning",
            "planning", "manipulation", "recovery", "distribution_shift",
            "ambiguous"]
```

Eight names. The decision rules, in the order they are evaluated (`classify.py:130-160`):

| # | Condition | Family assigned |
|---|---|---|
| 0 | segmenter skipped, or no signal set matches | `ambiguous` |
| 1 | `terminal == "never_reached"` | `planning` |
| 2 | `_wrong_object(r, S)` — ended within 9 cm of a distractor **and** < 0.6× the distance to the target | **`spatial_reasoning`** |
| 3 | `final_err > 0.10 m` | **`visual_grounding`** |
| 4 | `attempts ≥ 2` **and** `spread < 0.015 m` | `recovery` |
| 5 | `attempts ≥ 1` | `manipulation` |
| 6 | otherwise | `ambiguous` |

Plus an orthogonal **cost** axis (`failure_cost`, `classify.py:76-99`): `benign`, `disruptive`,
`safety`.

### Finding 1 — two of the eight families are unreachable, and so is one cost level

**No rule assigns `language_grounding`. No rule assigns `distribution_shift`.** They exist in the
list and nothing can produce them. Six families are reachable, not eight.

`language_grounding` being dead is *correct and unavoidable* on nominal LIBERO, and the survey now
states why in the strongest available form (§5.3): MINERVA scores **95.05% with no language encoder
at all**, using a 40-entry task-ID lookup table, and permuting that table collapses it to chance. The
benchmark never asks one scene to support two goals, so there is nothing for a grounding pathway to
do — the family is not under-populated, it is **not measurable on nominal LIBERO in principle**.
Keeping the name in the list is fine; reporting it as a family we *cover* is not.

`distribution_shift` being dead is a different matter and is not explained anywhere. It is a
declared capability with no implementation.

Same pattern on the cost axis: `COST_SAFETY` is defined at `classify.py:99` and **never assigned by
any branch of `failure_cost`**. The function can only return `benign`, `disruptive`, or `None`. Since
cost is flagged in the code as *"rated our most novel contribution by the landscape survey"*, a
permanently-empty severity level is worth knowing about before it appears in a client manifest.

### Finding 2 — `visual_grounding` and `spatial_reasoning` are swapped

This is the substantive one.

**Rule 2** fires when the end-effector finishes *near a different object than the one named*. The
function's own docstring (`classify.py:33-40`) says exactly what it is:

> *"Did it end up at a distractor rather than the named target? … Selection has its own signal —
> proximity to a distractor — and it must be checked directly."*

Ending at a **different named object** is an object-**selection** failure. Selection from a named set
is *grounding*. The rule assigns `spatial_reasoning`.

**Rule 3** fires when the end-effector finishes >10 cm from the target *with no distractor nearby* —
the episode went to a place, and the place was not near anything in particular. That is a
**positional / spatial** failure. The rule assigns `visual_grounding`, with the reason string *"went
confidently to the wrong place"*.

The two labels are the wrong way round relative to both the code's own comments and ordinary usage.
This is not cosmetic: it is the §3.5 failure mode observed live — **a family quietly carrying
something other than its name, which κ cannot detect**, because two annotators applying the same
written rubric will agree with each other perfectly while both label the phenomenon wrongly.

It also explains the distribution. `visual_grounding` dominating at ~67% is not a claim that the
policy cannot bind language to objects; it is the share of failures that ended somewhere unremarkable
— which for a policy at 61% overall is exactly what you would expect the modal failure to be.

**Independent corroboration that the grounding reading is wrong.** LIBERO-Plus decomposed object
layout into *adding confounding objects* versus *displacing the target*: most models barely moved
under the first and dropped hard under the second, and the authors conclude models *"may have merely
learned the positional information of the target objects"*. Object discrimination is not the weak
point on this benchmark. Position is. A taxonomy reporting 67% grounding failures contradicts the
published characterisation of the same benchmark — and the contradiction resolves entirely once the
two labels are swapped back.

### Finding 3 — rule precedence makes family sizes non-comparable

The rules are evaluated most-specific-first and the first match wins, so each family's catchment is
its own condition **minus every earlier condition**. The conditions are wildly unequal in size:

| family | conditions that must hold |
|---|---|
| `visual_grounding` | **one**: `final_err > 0.10` |
| `recovery` | **four**: `attempts ≥ 2` **and** `spread < 0.015` **and** `final_err ≤ 0.10` **and** not near a distractor |

**`recovery` cannot fire on any episode that ends more than 10 cm from the target**, because rule 3
takes it first. An episode that retries five times in one spot and finishes 12 cm away — a textbook
non-recovering retry loop — is labelled `visual_grounding`.

So `recovery = 8` across the campaign is **partly a precedence artifact and not a prevalence
measurement.** This is a better explanation than the two on offer ("genuinely rare" vs "detector
broken"), and it is testable in minutes by permuting rule order (§5, Test 2).

The published expectation is that recovery *is* rare — VLAs are trained on failure-free
demonstrations and a whole 2026 subfield (FLARE, RedFlow, RePO-VLA, B2FF, FAR) exists to add recovery
behaviour they lack. But "rare for the published reason" and "structurally suppressed by rule order"
predict the same count, and we currently cannot tell them apart.

---

## 2. How our families compare to published sets

| Source | Size | Cut | Categories |
|---|---|---|---|
| **Ours** | 6 reachable (8 declared) | **mixed** — see below | visual_grounding, spatial_reasoning, planning, manipulation, recovery, ambiguous |
| RoboFAC | 6 leaves, 3 levels | pipeline stage | Task Planning {Step Omission, Wrong Object}; Motion Planning {Position Deviation, Orientation Deviation}; Execution Control {Grasping Error, Timing Error} |
| AHA / FailGen | 7 | perturbation geometry | No_Grasp, Slip, Translation, Rotation, No_Rotation, Wrong Action Sequence, Wrong Target Object |
| SO-101 | 4 | symptom | Grasp Instability, Repetition Loop, State Mismatch, Precision Misalignment |
| LIBERO-Plus | 7 → 21 | environmental cause | Objects Layout, Camera, Robot Initial State, Language, Light, Background, Sensor Noise |
| LIBERO-Safety | 5 suites × L0–L2 | consequence / safety | Affordance-Aware Grasping, Human-Robot Interaction, Tabletop Spatial Avoidance, Free-Space Hand-Object Avoidance, Semantic Safety Reasoning |
| Honig & Oron-Gilad | 2 top-level | origin | technical vs social/interaction (HRI framing; not applicable to us) |

### Our taxonomy has no single cut, and that is the root problem

Run our six against the survey's four axes (§3.1):

- `planning` — **pipeline stage** (which module is at fault)
- `manipulation` — **pipeline stage**
- `visual_grounding` / `spatial_reasoning` — nominally **capability**, actually **symptom** (where the
  end-effector ended up)
- `recovery` — **behaviour pattern** (did it retry, and how)
- `ambiguous` — not a category; an abstention

Four different cuts in one list. That is why the categories are not mutually exclusive in principle —
an episode can be a grasping failure (`manipulation`) *that occurred at a distractor*
(`spatial_reasoning`) *after three retries* (`recovery`) — and the rule ordering, rather than the
taxonomy, silently decides which one is reported. Every published set above commits to **one** cut.
That is the single clearest structural difference between ours and theirs.

**Closest published relative:** RoboFAC, whose three-level pipeline-stage cut maps onto our
`planning` / `manipulation` split, and whose *Wrong Object* leaf is our rule 2 under the correct name.
**Most useful complement:** LIBERO-Plus, because it is a *cause* taxonomy and therefore the only one
that is directly actionable — but it requires you to have controlled the cause, which needs the
perturbation machinery we do not yet have.

### What we have that they do not

The **cost axis** (`benign` / `disruptive` / `safety`) is genuinely uncommon: of the sets above only
LIBERO-Safety is consequence-oriented, and it is a whole benchmark rather than a label on ordinary
rollouts. Orthogonal cost-by-family is a defensible contribution — **once `safety` is reachable.**
LIBERO-Safety is also the obvious thing to populate it against.

---

## 3. Campaign distribution, as reported

Post-fix counts from the mining session, three suites (goal not included in what was reported):

| family | spatial | object | libero_10 | total | share |
|---|---|---|---|---|---|
| visual_grounding | 99 | 161 | 73 | **333** | **67%** |
| manipulation | 57 | 20 | 32 | 109 | 22% |
| spatial_reasoning | 19 | 0 | 23 | 42 | 8% |
| recovery | 4 | 0 | 4 | 8 | 1.6% |
| planning | 2 | 0 | 1 | 3 | 0.6% |
| ambiguous | 0 | — | — | 0 | 0% |
| | | | | **495** | |

Three things to read off it, all of which are consequences of §1 rather than facts about the policy:

1. **One family holds two-thirds.** A partition where the modal class is 67% carries little
   information — and per Finding 2 it is the catch-all "ended somewhere unremarkable" bucket, so the
   concentration is a property of the rule, not a discovery about the policy.
2. **`ambiguous = 0` is suspicious, not reassuring.** A classifier that never abstains on 495 real
   failures, having been threshold-tuned on a toy, is more likely to be over-assigning than to be
   complete. G8 abstention is a design principle here and it is firing zero times.
3. **Two families are near-empty for structural reasons** (`recovery`, `planning`) and two are empty
   for definitional ones (`language_grounding`, `distribution_shift`).

Effective taxonomy size is therefore **three** (`visual_grounding`, `manipulation`,
`spatial_reasoning`), not eight.

---

## 4. What the field says "useful" means, and the one experiment that measures it

The only published controlled test of whether a discovered failure taxonomy is *useful*
([arXiv:2506.06570](https://arxiv.org/abs/2506.06570), and see the correction in survey §3.5):

| arm | failure rate |
|---|---|
| before | 46% |
| **failure-guided data collection** | **18%** |
| uniform random collection, **same budget** (40K samples) | 34% |

Guided beat uniform 28 points to 12 at equal budget. Nine clusters were discovered, carrying
prevalence (Thin–Protruding Objects 42%, Uniform/Featureless Surface 23%, Narrow-Gap 18%), and
several had been independently identified by HJ-reachability analysis.

**Caveat that matters for us:** this was indoor navigation with a CNN policy, not manipulation with a
VLA. No manipulation equivalent exists. That is simultaneously the gap and the opportunity.

**This is the design to copy**, and note what makes it work: it has a **control arm**. Without the
uniform arm, "we collected data in failure zones and the failure rate halved" is unfalsifiable —
more data helps regardless. The uniform arm at equal budget is the entire experiment.

---

## 5. The test programme — which families are useful, further clusterable, and actionable

Ordered by cost. The first four are CPU-only over the existing 965-trace campaign.

### Test 1 — Reachability and concentration *(minutes, CPU)*
Report, per family: reachable or dead; count; share; and the abstention rate. **Pass condition:** no
declared family is unreachable, and `ambiguous` is non-zero. Currently fails on both counts.

### Test 2 — Precedence sensitivity *(minutes, CPU)* — **run this first**
Re-classify the full campaign under permuted rule orders, and again with each rule ablated. Report
how far each family's count moves. **Interpretation:** a family whose count swings widely under
reordering is measuring the rule order, not the phenomenon. This directly settles whether
`recovery = 8` is rarity or suppression, and it is the cheapest informative thing in this document.

*Nobody in the surveyed literature does this.* Every published rule-based taxonomy has an evaluation
order and none report sensitivity to it.

### Test 3 — Separability and further clusterability *(hours, CPU)*
Per the survey's §4.1 finding — **cluster the discrepancy, not the episode** — build per-episode
feature vectors from what differs between the failure and its nominal counterpart, then:

- **Within-family clustering.** If a family splits into coherent sub-clusters with distinguishable
  triggers, it is **under-resolved** and should be split. Expect this for `visual_grounding` at 67%.
- **Cross-family separation.** Train a simple classifier to predict the assigned family from the
  feature vector. If two families are not separable above chance, they are **one family wearing two
  names** and should be merged.
- **Null-corpus control** (survey §4.3): run the identical pipeline on a corpus with no mode
  structure. If the null produces comparable cluster statistics, the structure is the pipeline's.

**This is the "further clusterable" question asked properly**, and its output is a *revised* family
set derived from our data rather than carried from the toy.

### Test 4 — Name-based re-assignment *(one annotator pass)*
Survey §8.5. Give a held-out judge only each family's **name and reason string**, have them assign
held-out episodes, and measure agreement with the classifier. Unlike κ on the rubric itself, this
tests whether the *name* reproduces the partition. **Finding 2 predicts this test fails specifically
on `visual_grounding` and `spatial_reasoning`** — a human given those two names will assign
distractor-proximate episodes to grounding, which is the opposite of what the code does. That makes
this a sharp, falsifiable prediction and a good first use of the instrument.

### Test 5 — The utility experiment *(GPU, the real gate)*
The 46→18-vs-34 design, adapted:

1. Partition the campaign by (revised) family.
2. For the largest families, generate remediation data targeting that family specifically — for the
   positional family, demonstrations at perturbed target placements; for the grasp family,
   demonstrations at the failing approach geometries.
3. **Arm A:** fine-tune on family-targeted data. **Arm B:** fine-tune on the *same number* of
   uniformly-sampled additional demonstrations. **Arm C:** no fine-tuning.
4. Measure success on held-out instances.

**A family is useful iff Arm A beats Arm B at equal budget, for that family's own failures.** A
family where guided ≈ uniform is decoration however clean its clusters look. This is the only test
here that can retire a family, and it is the only claim a client should be asked to pay for.

**Sequencing note:** Tests 1–4 should precede Test 5, because Test 5 is expensive and its result is
meaningless if run against a taxonomy with swapped labels and precedence artifacts.

---

## 5b. TEST 2 RESULT — precedence sensitivity, run 2026-09-16

`experiments/family_precedence_test.py` (read-only; imports `classify.py`, does not modify it).
965 rollouts, 667 failures, **534 featurised**, 133 skipped. Rather than permute-and-re-run, it
computes each failure's full **predicate vector** — which rules *would* fire, independent of order —
then applies orderings to that. One segmentation pass, and the overlap is exposed directly.

### The headline

> **409 of 534 failures (76.6%) fire more than one rule.** Their family is decided by **rule order,
> not by the taxonomy.**

| rules firing | failures | share |
|---|---|---|
| 1 | 125 | 23.4% |
| 2 | 353 | 66.1% |
| 3 | 56 | 10.5% |

Most common co-firing sets: `lost_target + any_attempt` **336**; `any_attempt` alone 125;
`wrong_object + lost_target + any_attempt` 41; `lost_target + repeat_attempts + any_attempt` 14.

### Family counts swing enormously across the 120 possible orderings

| family | shipped | min | max | swing | swing ÷ shipped |
|---|---|---|---|---|---|
| visual_grounding | 350 | 0 | 397 | 397 | 1.1× |
| manipulation | 125 | 125 | **528** | 403 | **3.2×** |
| spatial_reasoning | 43 | 0 | 44 | 44 | 1.0× |
| recovery | 10 | 0 | 24 | 24 | **2.4×** |
| planning | 6 | 0 | 6 | 6 | 1.0× |
| ambiguous | 0 | 0 | 0 | 0 | — |

**Every family except `manipulation` can be driven to zero by reordering alone.** Not one of these
counts is a property of the rollouts.

### Unconditioned rule sizes — how much each family is suppressed

| rule | family | fires alone | shipped | suppressed |
|---|---|---|---|---|
| `any_attempt` | manipulation | **528** | 125 | **403** |
| `lost_target` | visual_grounding | 397 | 350 | 47 |
| `wrong_object` | spatial_reasoning | 44 | 43 | 1 |
| `repeat_attempts` | recovery | **24** | 10 | **14** |
| `never_reached` | planning | 6 | 6 | 0 |

### Four findings

**1. The recovery question is answered: suppression, not rarity.** 24 failures exhibit
repeat-attempt behaviour; **10 survive precedence**. More than half are relabelled — almost all as
`visual_grounding`, because a retry loop that ends >10 cm away is taken by rule 3 first. Recovery is
**~2.4× more prevalent than reported**. The published expectation that recovery is rare is still
true; it is just not what our count was measuring.

**2. `manipulation`'s predicate is near-vacuous.** `attempts >= 1` fires on **528 of 534 failures
(98.9%)**. A condition satisfied by 99% of the population carries essentially no information, and
under a different ordering it would swallow the entire taxonomy.

**3. `ambiguous = 0` is structural, and abstention is dead.** `any_attempt` covers 528 and
`never_reached` the remaining 6 — **disjoint and exhaustive, 528 + 6 = 534**. No failure can reach
the fallthrough under *any* ordering. G8 abstention is unreachable on LIBERO by construction, which
means §3's *"ambiguous = 0 is suspicious, not reassuring"* was right for a more specific reason than
suspected: the classifier is not confident, it is merely unable to decline.

**4. `visual_grounding`'s dominance is a precedence artifact layered on a naming error.** Rule 3
fires on 397 of 534 (74%) before rule 5 ever runs. Combined with Finding 2 of §1 (the label is
swapped), the reported "65.5% visual grounding" is the share of failures that *ended more than 10 cm
from the target and were not claimed by an earlier rule* — which is neither grounding nor a
discovery.

### A data-quality finding that fell out of the run

133 of 667 failures were skipped for missing `_gt_eef_to_object`, and they are **not evenly
distributed**:

| suite | rollouts | failures | missing the key |
|---|---|---|---|
| libero_spatial | 260 | 181 | 0 |
| libero_object | 260 | 181 | 0 |
| libero_10 | 185 | 133 | 0 |
| **libero_goal** | 260 | 172 | **133 (77%)** |

The goal suite is **inconsistently instrumented within itself** — 77% of its failures carry no
object-distance key at all and silently fall through to the toy segmenter, which then abstains.
Every `libero_goal` family count published so far is computed over the 39 failures that happened to
be instrumented, not over 172. That is not a sampling caveat, it is a different denominator, and it
is worth finding before anything else in this document is acted on.

---

## 5c. PERTURBATION-ARM ANALYSIS — measured 2026-09-16, **largely withdrawn same day**

> **CORRECTION.** The first version of this section reported that **48% of all failures** end with the
> object unmoved and the arm never near it, and read that as "what a SmolVLA failure normally looks
> like". **That was confounded and the claim is withdrawn.** `vla-7f` pointed out that the campaign is
> a **perturbation sweep**, not a nominal corpus: most failures are camera-yaw episodes and most
> successes are nominal, so the separation I validated was partly nominal-vs-perturbed. I had seen the
> `perturbation` field in these traces and did not split on it. The corrected analysis is below and it
> says something different — and more useful.

### The campaign design

Per suite: 40 nominal, 40 at `camera_yaw_deg = 0.0` (*intended* as a control for the perturbation
machinery — it is not one; see below), 40 each at yaw 5/10/15/20°, 20 compound (yaw + ee offset +
light). Nominal and yaw-0 are pooled as *nominal*, which `vla-7f` confirmed is legitimate: both skip
the camera code entirely and they agree within noise.

### The raw table — ⚠ THE PERTURBED ROWS ARE WITHDRAWN, see below before using any of them

Retained only so the withdrawal is checkable. **The nominal rows are sound; the yaw and compound
rows measure something other than their label.**

"never approached" = object displaced <5 cm **and** end-effector never within 5 cm of any recorded object.

| suite | arm | n | success % | failures never approaching |
|---|---|---|---|---|
| **object** | nominal | 80 | **88.8** | 11.1% |
| | yaw 5° | 40 | **0.0** | **92.5%** |
| | yaw 10° | 40 | 2.5 | 87.2% |
| | yaw 15° | 40 | 5.0 | 86.8% |
| | yaw 20° | 40 | 5.0 | 84.2% |
| **goal** | nominal | 80 | **75.0** | 23.5% |
| | yaw 5° | 40 | 12.5 | **70.0%** |
| | yaw 10° | 40 | 17.5 | 64.3% |
| | yaw 15° | 40 | 17.5 | 72.4% |
| | yaw 20° | 40 | 7.5 | 65.6% |
| **spatial** | nominal | 80 | **72.5** | 18.2% |
| | yaw 5° | 40 | 10.0 | 38.9% |
| | yaw 10° | 40 | 12.5 | 31.4% |
| | yaw 15° | 40 | 7.5 | 27.0% |
| | yaw 20° | 40 | 12.5 | 25.7% |
| **long** | nominal | 65 | **56.9** | 7.1% |
| | yaw 5° | 25 | 8.0 | 8.7% |
| | yaw 10° | 25 | 12.0 | 13.6% |
| | yaw 15° | 25 | 8.0 | 8.7% |
| | yaw 20° | 25 | 4.0 | 16.7% |

### ⚠ THE YAW ARMS ARE NOT A YAW SWEEP — measured from recorded extrinsics

**Raised by `vla-7f` from the code; confirmed here from the traces.** Do not cite the dose-response
below as a yaw dose-response.

*The bug* (`vla_harness/envs/libero_env.py:291-313`), verified by reading it:
1. The guard is `if yaw or pitch or dist:` — **`yaw = 0.0` is falsy, so the yaw-0 arm skips the entire
   camera block.** It is byte-identical to nominal `{}`, and is therefore **not a control for the yaw
   arms**; it is a second copy of nominal.
2. In the non-zero branch, after rotating the camera position the code does
   `sim.model.cam_quat[cid] = _look_at_quat(look + rel, look)` — it **replaces** the orientation with a
   look-at toward an approximated table centre, **discarding LIBERO's original camera aim**. The
   intent is documented and sound ("a camera that rotates AND loses the scene confounds viewpoint with
   occlusion"), but because the control skips it, the re-aim lands entirely inside the first step.

*The magnitude*, computed from `scene_descriptor.cameras.agentview.xmat` in the stored traces — no
simulator, no frame assumptions, just the recorded rotation matrices. Angle between each arm and the
yaw-0 arm:

| suite | yaw 5° | yaw 10° | yaw 15° | yaw 20° | camera translation at yaw 20° |
|---|---|---|---|---|---|
| object | **12.99°** | 15.61° | 19.19° | 23.29° | 31.1 cm |
| spatial | **12.81°** | 15.45° | 19.06° | 23.19° | 22.9 cm |
| goal | **12.81°** | 15.45° | 19.06° | 23.19° | 22.9 cm |
| long | **8.14°** | 11.88° | 16.31° | 21.00° | 21.1 cm |

**A "5° yaw" is a ~13° camera rotation plus ~6–8 cm of translation.** The marginal cost of each
further 5° of nominal yaw is only 2.6–4.1° of actual orientation change, while the *first* 5° buys
13°. The constant re-aim dominates the first step and is absent from the control.

*Consequences:*
- The headline **"5° yaw → 0.0%"** is wrong as stated. What was measured is "**a ~13° camera
  orientation change plus ~8 cm translation takes libero_object to 0/40**". Still a real and striking
  fragility result — but not a 5° one, and not a dose-response.
- **It is a threshold effect, not a dose.** Success is 0.0 / 2.5 / 5.0 / 5.0 across 13→23° of actual
  change: flat once past the first step. The sweep has **no samples between 0° and 13°**, which is
  exactly where the transition happens. **The campaign cannot locate the robustness boundary, because
  its smallest non-zero perturbation is already past it.** That is what `PLAN.md`'s adaptive-sweep
  machinery (D10) exists for, and this is the concrete case for it.
- The "spatial fails differently under yaw" reading (below) is **unverified**: the re-aim changes
  pitch in opposite directions for floor-type and tabletop-type scenes, which could produce that
  difference on its own.
- Compound arms carry the same confound (their orientation delta equals yaw-15's exactly).
- **Any `camera_misalignment` manifest row from this campaign is suspect.**
- **The nominal-arm results are unaffected** — `{}` and yaw-0 both skip the block, so the goal-gap
  corroboration below stands.

The fix belongs to `vla_harness` (`my primary`): either rotate the *original* quaternion about world
z, or run the re-aim at yaw = 0 as well so the control matches the treatment.

### WITHDRAWN — the perturbed-arm claims

**Confirmed by render** (`my primary`, 2026-09-16): at `yaw = 1e-6` the agentview image changes **as
much as at `yaw = 5`** — mean absolute pixel difference **53.1 vs 52.3** on object, **44.1 vs 45.8**
on spatial. A negligible yaw produces the same image change as a 5° yaw, because the change is the
re-aim, not the yaw. `vla-7f`'s geometry reproduced exactly.

The following claims from the first version of this section are **withdrawn, not caveated**:

- ~~"A 5° camera yaw takes libero_object from 88.8% to 0.0%"~~ — the arms are not a yaw sweep. The
  independent variable was a fixed ~12–13° re-aim plus a translation, identical in kind across all
  four yaw levels.
- ~~"dose-response"~~ — there is no dose. The nominal yaw contributes 2.6–4.1° per 5° step against a
  ~12° constant present in every treated arm and absent from the control.
- ~~"spatial fails differently under yaw"~~ — the re-aim shifts pitch in **opposite directions** for
  floor-type and tabletop-type scenes, which is sufficient to produce that difference on its own.

**A second defect in the same function**, found by `my primary`: `reset()` returns a **stale frame** —
`_resettle` only refreshes `_raw` when `_to_lerobot_obs` exists, so the policy's first observation of
**every perturbed episode** shows the *unperturbed* scene. This affects the ee-offset and lighting
arms too, not just yaw.

**Net: the entire perturbed half of this campaign is unusable**, and all 12 `camera_misalignment`
manifest rows (3 per suite) are invalid. Both fixes are with the owner as `PENDING_DECISIONS.md` #22.

**"Lost under viewpoint shift" may well be a real family** — the render check shows the camera really
did move, and success really did collapse. But this campaign cannot *measure* it, because the
treatment was not the labelled variable and the first frame was stale. It needs a re-run after the
fix.

### What survives, and why

**The nominal arms are unaffected.** `{}` and `yaw = 0.0` both fail the `if yaw or pitch or dist:`
guard and skip the camera block entirely, so neither carries the re-aim or the stale frame. `vla-7f`
checked they agree within noise (spatial 65% vs 80%, n=40 each; z = 1.52, not significant), so
pooling them as *nominal* is legitimate.

Surviving results: the nominal success-rate table below, the goal-gap corroboration, and the
nominal-only wrong-goal null (0/15).

### The methodological finding, which outlives the bug

**The sweep has no samples between 0° and ~13° of actual camera change, and that is where the
transition happens.** Success is 0.0 / 2.5 / 5.0 / 5.0 across 13→23°: flat once past the first step.
Even after the bug is fixed, **a fixed grid at 5/10/15/20 cannot locate this policy's robustness
boundary, because the boundary lies below its first sample.**

That is the concrete case for the adaptive-sweep machinery (`PLAN.md` D10), which has been deferred
on the grounds that it is "not needed until we know roughly where a boundary is". We now know: it is
below the grid.

### Nominal success rates — a better estimate than the pooled campaign number

| suite | ours, nominal arm | community repros of this checkpoint |
|---|---|---|
| object | 88.8 | 91 / 96 / 93 |
| goal | **75.0** | **83 / 87 / 81** |
| spatial | 72.5 | 73 / 83 / 82 / 63 |
| long | 56.9 | 43 / 38 / 56 |

On nominal episodes, spatial is in range, long is at or **above** the community range, object is
~4 pp low, and **goal is the only meaningful gap**. That independently corroborates `vla-7f`'s
community-repro finding that the residual is concentrated in libero_goal.

### The wrong-goal probe, re-run on nominal only

**0 wrong-goal completions in 15 evaluable nominal goal failures** (8 "elsewhere", 7 unmoved). By the
rule of three, that bounds wrong-goal execution at **<20% of nominal goal failures at 95%
confidence**. Weak, but not vacuous — and it is evidence against language-conditioned wrong-goal
execution as the explanation for the goal deficit. The pooled version (4/101, 2 of them artifacts)
was measured mostly on yaw-perturbed episodes where the policy may not resolve the scene at all, and
should not be cited.

### What survives for the taxonomy — the point that holds under either reading

`classify.py` has **no family for "never approached the object"**, under nominal *or* perturbed
conditions. The nearest rule is `planning` via `terminal == "never_reached"`, which §5b measured
firing **6 times in 534 failures**. A direct geometric measure finds ~301 pooled, and ~11 of 76
evaluable nominal failures. The nominal mismatch is smaller than the pooled one and still a mismatch.

More importantly, the corrected analysis surfaces **two families the taxonomy does not have**, both
defined by a controlled intervention rather than by a threshold carried from the toy:

- **lost under viewpoint shift** — object unmoved, arm never approaches; dominant in object and goal
  under yaw
- **engages but fails under viewpoint shift** — arm approaches, task still fails; dominant in spatial
  under yaw

That is what a *cause*-cut taxonomy looks like (`docs/TAXONOMY_FAMILY_GUIDELINES.md` §3.2), and it is
only visible because the campaign assigned the treatment. It is the strongest available argument for
the cause cut over the symptom cut we currently have.

### Method note for the revision

**Any family count over this campaign must be reported per perturbation arm.** Pooling across a
designed sweep mixes populations that differ by construction, which is exactly the error corrected
here. Scripts: `experiments/goal_wrong_target_probe.py` and the split pass in this section.

---

## 6. Recommendations

**Before any further mining runs:**

1. ~~**Swap `visual_grounding` and `spatial_reasoning`,** or rename both to what they measure.~~
   **DECIDED 2026-09-16 — leave as-is, document.** The project owner's reasoning: the family set is
   about to be revised with input from the people who will consume the manifest, so renaming rules
   that are likely to be replaced is churn. **The finding stands and is recorded** — rule 2 measures
   object *selection* and is labelled `spatial_reasoning`; rule 3 measures *position* and is labelled
   `visual_grounding`. Anyone reading a family count in the interim must read §1 Finding 2 first.
   Carried into `docs/TAXONOMY_FAMILY_GUIDELINES.md` §6.1 as a named anti-pattern so the revised
   taxonomy does not reproduce it.
2. **Remove `language_grounding` and `distribution_shift` from `FAMILIES`,** or mark them explicitly
   unreachable. A declared-but-impossible family in a client-facing schema is a liability. Note in
   the same place that `language_grounding` is unreachable *by property of the benchmark* (MINERVA),
   not by our omission — that is a defensible statement and worth making.
3. **Make `COST_SAFETY` reachable or remove it.** LIBERO-Safety is the benchmark that would populate
   it.
4. **Commit to one cut.** Given what we can actually control, the **cause** cut (LIBERO-Plus-style)
   is the one that is both actionable and defensible, because we assign the treatment. Symptom
   families should then be evidence *for* a cause, not the reported label.

**Then, in order:** ~~Test 2~~ **(done — §5b)**, Test 1 (minutes), Test 4 (one pass), Test 3 (hours),
Test 5 (GPU).

> **DECISIONS, 2026-09-16.**
> - **Test 5 (utility experiment): documented as a next step, not scheduled.** Until it runs, our
>   families are **unvalidated in the only sense that matters** (guidelines §5, rung 5) and every
>   statement about them should be hedged accordingly. This is a deliberate, recorded choice — not
>   an oversight.
> - **The family set will be revised with input from manifest consumers** before the next mining
>   pass, rather than repaired in place. Requirements given: families must be *relevant to those
>   consumers* and *meaningful when viewed separately*. The second requirement has teeth — see
>   guidelines §1, since a family produced by first-match-wins precedence is by construction **not**
>   standalone-meaningful, which connects §5b's result directly to a stated requirement.
> - **Naming left as-is and documented** (see recommendation 1 above).
>
> **`docs/TAXONOMY_FAMILY_GUIDELINES.md` (2026-09-16) is now the governing document** for how that
> revision should be done, tested and reported. Read it before any taxonomy decision. This audit
> becomes the worked example of what it is guarding against.

**The single most important thing in this document** is that our current family counts cannot be
reported to anyone, because three separate mechanisms — swapped labels, rule precedence, and
unreachable families — each independently distort them. All three are cheap to fix and none require
the GPU.

**Test 2 has now measured the second of those** (§5b) and it is larger than expected: **76.6% of
failures have their family decided by rule order**, every family but one can be driven to zero by
reordering, and `recovery` is suppressed by more than half. A fourth mechanism turned up in the same
run — the `libero_goal` traces are 77% missing the key the whole pipeline depends on.

### Fixing precedence properly

Reordering the rules does not fix this; it relabels the same 76.6%. The overlap is the problem, and
there are three honest responses:

1. **Make the predicates disjoint by construction.** Partition on a single variable per level — e.g.
   first on terminal behaviour, then on *where* it ended, then on *what it did there* — so that no
   failure satisfies two leaves. This is what committing to one cut (§6.4) means operationally.
2. **Emit the predicate vector, not the label.** Report which rules fired, and let the count be
   over predicates rather than over families. `lost_target ∧ any_attempt` (336 episodes) is a
   perfectly good manifest row and is exactly what the data supports; collapsing it to
   `visual_grounding` throws away the conjunction and adds a claim the evidence does not carry.
   **This is the cheapest correct option and it requires no new thresholds.**
3. **Keep first-match-wins but publish the sensitivity.** Report each family's count *and* its range
   across orderings. Honest, but it concedes that the number is not well-defined.

Option 2 is what I would do. It also composes with `ddmin` (survey §7.1): a conjunction of predicates
is exactly the object a minimal-failure-inducing-subset search consumes.
