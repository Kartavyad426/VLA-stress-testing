# Findings

> ## ⭐ REFERENCE — re-derive these when the ground moves
>
> Several findings below are **measurements of things that can change under
> us**. They are true of a specific corpus, checkpoint, library version or
> machine, and they are the kind of fact that is expensive to rediscover and
> dangerous to assume still holds.
>
> **Re-run the analysis behind a finding when any of these change:**
>
> | If this changes | Re-derive | Why |
> |---|---|---|
> | LIBERO-plus corpus revision | **F5** | Category counts, the multi-factor fraction (3,279/10,030) and single-factor coverage are all properties of *this* corpus. `benchmark_scripts/` may also gain generation tooling, which would restore a real knob API. |
> | The policy under test | **F1, F3, F4** | VRAM footprint, reproduction status and transfer behaviour are per-checkpoint. π0.5 not fitting says nothing about the next model. |
> | Dataset revision (`lerobot/libero` SHA) | **Phase 0 predictions** | `HuggingFaceVLA/libero` has been silently re-uploaded before. Coverage histograms and every prediction derived from them are pinned to `a1aaacb7…`. |
> | MuJoCo, robosuite, or LeRobot version | **F2, F3** | F2 *is* a version-dependent physics change. A bump can fix it or introduce a new one. |
> | The machine | **F4** and all timings | Power state alone moves wall-clock ~25×. |
>
> **How to re-derive F5 specifically** (the most likely to move):
> ```bash
> curl -sL https://raw.githubusercontent.com/sylvestf/LIBERO-plus/main/\
> libero/libero/benchmark/task_classification.json -o tc.json
> # then: count by category and difficulty_level; and count names encoding
> # >1 of (_table_, _view_, _light_, _language_, _noise, _add_)
> ```
> Takes about a minute and settles whether the attribution design still holds.

Running record of things we've established that aren't in the original proposal.
Newest first. Each entry says what we know, how we know it, and what it changes.

---

## F9 — The 360→256 render mismatch is a REAL defect, and a PARTIAL cause

**Date:** 2026-09-15 · **Status:** MEASURED · `res256_nas10_seed1000`, 400 episodes
· one variable changed from F8: render resolution 360→256

| Suite | 256×256 | 360×360 | Δ | p | vs published |
|---|---|---|---|---|---|
| **libero_spatial** | **73.0%** | 56.0% | **+17.0** | **0.011** | −17 |
| libero_object | 90.0% | 85.0% | +5.0 | 0.284 | −6 |
| libero_goal | 68.0% | 69.0% | −1.0 | 0.879 | −24 |
| libero_10 | 37.0% | 35.0% | +2.0 | 0.768 | −34 |
| **OVERALL** | **67.0%** | 61.3% | **+5.7** | 0.092 | **−20.3** |

### The hypothesis was right and insufficient

`LiberoEnvConfig` renders at **360×360**; the checkpoint declares **256×256**
inputs and was trained on a 256×256 dataset. The policy pads to 512 either way,
so training was a clean 2.00× upscale and our eval was a 1.42× resample.

Fixing it recovered **+17 pp on `libero_spatial`** — the only individually
significant effect. **It did not fix the problem: ~20 points remain.**

### The pattern supports the mechanism and bounds it

Spatial gained hugely; **goal and long gained nothing** (−1, +2). That is what a
resampling artifact should do: it degrades *fine visual discrimination*, and
`libero_spatial` is the suite defined by spatial relations among similar-looking
objects. Goal (−24) and long (−34) fail for reasons resolution does not touch.

### An upstream defect worth reporting

**`LiberoEnvConfig` defaults to 360 while `LiberoEnv`'s own class default is
256, and the SmolVLA checkpoint declares 256.** Nothing warns. Anyone running
`lerobot-eval` on this checkpoint at defaults is evaluating out of distribution
on every frame — plausibly a contributor to the nine open reproduction issues.

### Statistical caveat

Tests above are **unpaired**, but both runs used `seed=1000` with
`init_states=true`, so the episodes **are** paired. McNemar would be tighter and
might push the overall effect under 0.05. Per-episode outcomes are not in
`eval_info.json` — the same limitation that makes these runs unmineable.

### Next

`n_action_steps=1` (shipped default; we ran 10), ~9 h. Now the leading candidate
for the residual, and notably `libero_10` — the largest remaining gap — is
exactly where open-loop horizon should matter most.

---

## F8 — FIRST REAL RESULT: SmolVLA does not reproduce, and we are lower than anyone

**Date:** 2026-09-15 · **Status:** MEASURED · run `nas10_seed1000`, 400 episodes,
134 min · mujoco **3.3.7 (healthy)** · `n_action_steps=10` · seed 1000 · 10 eps/task

| Suite | Ours | 95% Wilson CI | Published | [#3264](https://github.com/huggingface/lerobot/issues/3264) | Verdict |
|---|---|---|---|---|---|
| libero_spatial | **56.0%** | [46.2, 65.3] | ~90% | 63.0% | published **outside** CI |
| libero_object | **85.0%** | [76.7, 90.7] | ~96% | 93.0% | published **outside** CI |
| libero_goal | **69.0%** | [59.4, 77.2] | ~92% | 81.0% | published **outside** CI |
| libero_10 | **35.0%** | [26.4, 44.7] | ~71% | 56.0% | published **outside** CI |
| **OVERALL** | **61.3%** | [56.4, 65.9] | **~87.3%** | 73.25% | **outside** |

**Row 2 of the §5.3 gate on every suite: genuine non-reproduction, not a screen
ambiguity.** And not multiplicity — 18.5% predicts *one* spurious flag across
four suites, not four.

**We are ~12 points below #3264 and ~6 below #3287.** Three independent parties,
three different numbers, all far below published, and ours is the lowest.

### What makes it hard to dismiss

- **We ran healthy physics.** mujoco 3.3.7; F2's broken range (3.4.0–3.8.1)
  takes one spatial task 80% → 28%. Reporters on broken versions should have
  scored *worse* than us, not better.
- **`libero_10` has the largest gap** (−36). The long-horizon suite is where
  compounding error predicts the most damage (F1 §1.4), and it is also where
  language reportedly matters most even for models that ignore it elsewhere.
- Only `libero_object` is close-ish (−11), which is the suite the literature
  calls least ambiguous.

### Caveats that must travel with this number

1. **It is a SCREEN.** 10 episodes/task, **one seed**. LeRobot recommends
   averaging three. The survey warns 10 vs 50 can swing spatial/long by 15–20 pp.
2. **`n_action_steps=10`, and nobody knows what the published numbers used.**
   The shipped config is **1**. This is the single most testable explanation and
   it is a **~9 h run** to check (3.95 vs 17 steps/s).
3. `#3264` at 63% on spatial sits **inside our CI** — our disagreement with
   *them* is not established; our disagreement with *published* is.

### Ranked hypotheses

| # | Hypothesis | Test | Cost |
|---|---|---|---|
| 1 | `n_action_steps` mismatch | re-run at 1 | ~9 h |
| 2 | Seed/sample variance | 3 seeds at 50 eps/task | ~15–24 h |
| 3 | Published numbers are not reproducible by anyone | nothing we can run — the accumulating evidence *is* the finding | — |
| 4 | Our environment is wrong | **the π0.5 anchor** — the only test that separates this from 1–3 | rented GPU |

### What it means for the project

**This is exactly the state DG-5 predicted when we deferred the anchor.** A
subject policy 26 points below its own published numbers makes every downstream
robustness claim hard to interpret: we cannot tell a policy weakness from a
setup problem without a reference we trust.

**Two readings, and we cannot currently choose between them:**
- our setup is wrong ⇒ every subsequent number is about us, not the policy
- SmolVLA genuinely does not reproduce ⇒ a **reportable finding** corroborating
  nine open issues that no maintainer has answered

**The anchor is what separates them**, and it needs a ≥16 GB card (F4).
Alternatively, switch the subject policy to VLA-Adapter, whose known gap has an
identified cause.

> **This run is NOT mineable** — it went through `lerobot-eval`, which stores a
> success bit and an mp4. Deliberate (see `EXPERIMENT_PROCEDURE.md` §0), but it
> means we cannot ask *why* these failed without re-running through our own loop.

---

## F7 — AS-3 closed; ddmin is validated, not speculative; the language probe must measure target identity

**Date:** 2026-09-15 · Source: `vla-81`, survey §3.3/§5.3/§5.5 · marked [A] where README-level

### 7a. Generation code does NOT ship — but the ingredients do

Definitively closing AS-3's second half. The LIBERO-plus repo ships:
the **10,030 pre-generated instances**, `assets.zip` (articulated objects, new
objects, scenes, textures, `.xml`/`.stl`), `task_classification.json` mapping
task ID → category + difficulty, and RLDS/LeRobot training datasets.

**No generation script, knob spec or perturbation config.** *(README-level read;
a definitive answer means cloning and listing the tree — cheap, not yet done.)*

**Wave E forks toward instance selection — but a better version than feared.**
The *ingredients* ship even though the generator does not: asset library,
scenes, textures, and a per-instance label saying which knob produced each one.

**So revert-one-knob becomes: diff a perturbed instance against its nominal
parent and undo the difference.** We build the reversion from released
artefacts rather than calling theirs. Recoverable, and it keeps the
interventional claim intact.

### 7b. ddmin is validated by the benchmark itself

LIBERO-Plus §5 already tested **combined** perturbations — six dimensions, 2,000
independent trials, OpenVLA-OFT — and found a consistent **negative
compositionality gap**: combined perturbations are worse than independent
effects predict, because "co-occurring shifts act as coupled noise sources".

| Pair | Success |
|---|---|
| Camera + Robot state | 19.05% |
| Robot state + Noise | 22.15% |
| Layout + Camera | 35.95% |

χ² significant. **On the benchmark closest to ours, main-effects-only
attribution is a DEMONSTRATED inadequacy, not a theoretical one.** ddmin is not
insurance against a hypothetical; it addresses a measured property of this
benchmark. *(Marked [A] — HTML render, not the PDF text layer.)*

### 7c. The shipped language instances are the WRONG probe

**Correction to our Phase G plan.** LIBERO-plus's 1,537 "Language Instructions"
instances are **LLM-based instruction rewriting to increase linguistic richness**
— *paraphrase*. Same goal, reworded.

**Running against them measures robustness to surface form, and it would read
as a grounding result.** Three distinct probes, only one of which answers our
question:

| Probe | Shipped? | Tests |
|---|---|---|
| Paraphrase (1,537 instances) | **yes** | surface-form robustness |
| Blank instruction | no | whether language is consumed at all — but the input is itself OOD, so an unchanged rate is **ambiguous** |
| **Directed substitution** | no | **the only one separating insensitivity from comprehension** |

We generate the latter two. Substitution needs one other object known to be in
the scene — available from the BDDL object list.

### 7d. Success rate cannot run this experiment. Target identity can.

**A success-rate delta cannot separate comprehension from partial grounding.**
A policy that reads a new instruction and fails, and a policy that ignores it
and executes the original target, **both give success → 0**. In aggregate they
are identical.

The distinguishing measurement is **which object the end-effector actually
approaches** — target identity, not the outcome bit. We already derive it from
object poses and eef-to-object distance.

`probes/language.py` now decides on `redirect_fraction` over paired episodes.
*Endpoint displacement, which the first version used, is better than success
rate but still indirect — an endpoint can move for reasons unrelated to target
choice.*

> **Caveat carried into the code:** target identity runs through the same object
> resolution path that failed in LE-2. **Gate it on the oracle before trusting a
> verdict built on it.**

### 7e. The scale hypothesis is unsettled, and worth stating as such

Not a design, a live disagreement, recorded because either outcome is a result:

- **Small models may need language MORE** — if SmolVLA lacks capacity to
  memorise 40 scene→action mappings, that escape hatch is unavailable.
- **Small models may ignore language MORE** — ignoring language is a
  *training-incentive* outcome, not a capability failure: in LIBERO the scene
  layout already identifies the task, so the instruction carries no information
  the images do not, and no model of any size gets gradient pressure to build a
  language pathway. Less capacity makes memorising scene→action *cheaper*.

Weak supporting evidence *[A, secondary]*: language sensitivity tracks task
**ambiguity**, not scale or architecture — 60–100% success with null/wrong
prompts on the unambiguous object suite, versus 94% → 10% collapse on the goal
suite, same models.

**SmolVLA is not among the ten checkpoints LIBERO-Plus tested**, so there is no
prior on it. And `libero_10` — in our running baseline — is where language
mattered even for models that ignored it elsewhere.

---

## F6 — Three consequences from the methods survey

**Date:** 2026-09-15 · Source: `docs/FAILURE_MINING_METHODS.md` §5.3, §7.1, §8.1, §8.5

### 6a. Nobody validates a taxonomy as correct. κ is the wrong instrument.

Our open problem #2 has a **negative** answer in the literature. Four substitutes
exist — planted-label recovery (internal correctness only), κ (self-consistency
only), resemblance to a prior taxonomy (unavailable: manipulation has no
canonical one), and downstream utility (**the only one that can falsify a
taxonomy, and nobody runs it**).

Two things that hit our design directly:

- **Raising the κ threshold does not fix it.** κ cannot distinguish a good
  taxonomy from a merely consistently-applicable one:
  `{failure on a Tuesday, failure not on a Tuesday}` scores **κ = 1.0**. This is
  construct validity, not reliability, and 0.5 → 0.7 addresses the wrong axis.
- **κ is depressed by exactly the class imbalance failure taxonomies always
  have** (the kappa paradox). Report **percentage agreement and Gwet's AC1**
  beside it.

**Partial substitute to adopt — name-based re-assignment (§8.5):** give a
held-out judge *only* the cluster names and descriptions, have them assign
held-out episodes, and measure agreement with the induced partition. Unlike κ on
a hand-designed taxonomy, this genuinely tests the partition. It establishes
that a cluster has a **communicable common property** — not that the property is
the *right* one. Partial, and must be reported as partial.

### 6b. Delta debugging closes our interaction blind spot

**ddmin** finds the **minimal failure-inducing knob subset** in O(n²) same-seed
re-runs rather than 2ⁿ. Every re-run is a rollout we can already do.

Revert-one-knob **structurally cannot see interactions**: a failure requiring
viewpoint AND initial state *jointly* shows a small effect on every single
reversion, and we currently report "no factor responsible" — which in our output
is **indistinguishable from a genuinely uninformative case and from multiple
sufficient causes**. Three different situations, one rendering.

Highest-value import from the survey. Sequence it with Wave E.

### 6c. Test language insensitivity BEFORE a language family reaches a client

LIBERO-Plus's blank-instruction experiment: OpenVLA-OFT on the object suite is
**"largely unchanged with no language input at all"** — the authors say it
"degenerates into a form that disregards language, behaving more like a
Vision-Action model". Their goal-replacement probe then shows success **dropping
nearly to zero while the model still executes the ORIGINAL target's
trajectory**.

**So apparent language robustness is insensitivity, not comprehension.** If our
policy behaves the same, a `language_grounding` family is near-unpopulated on
nominal LIBERO, and a classifier carrying that family will either abstain or
**quietly label something else with its name** — which, per 6a, κ will not catch.

The directed-substitution probe is **one extra rollout per instance with one
word changed**. Cheap, and it should run before any language claim.

> Note this cuts against DG-11, where we declared `language_grounding` out of
> scope for lack of a detector. The corpus has 1,537 language instances and this
> probe is nearly free — the constraint was ours, and it is smaller than we said.

---

## F5 — LIBERO-plus: a GENERATOR whose RELEASED CORPUS is difficulty-filtered

> ### ⚠ CORRECTED 2026-09-15 — the original conclusion was wrong
>
> I concluded from LeRobot's loader that LIBERO-plus is "a fixed corpus, not a
> knob API". **That is wrong.** Session `vla-81` read the paper: it is a
> **generator** — 40 LIBERO tasks × 500 instances per sub-task across 7
> dimensions = **14,000 candidates**, and its unit of operation is explicitly
> the **single-dimension perturbation**. So revert-one-knob is *structurally
> constructible* after all, and Wave E's premise needs revisiting.
>
> I inferred a generator's absence from a *consumer's* code path — the same
> class of error as the interface lesson we had just circulated. Reading the
> loader told me how LeRobot **consumes** the artifact, not how the artifact is
> **made**.
>
> **Still unverified:** whether the generation code is actually distributed.
> That is a repo question and the remaining half of AS-3.
>
> ### The new finding, which is more consequential than the old one
>
> The **released artifact is not the generated population.** The authors deleted
> every task "solved by all models, or by a large majority", balanced the
> remainder, and released **10,030 of 14,000** — difficulty-filtered against four
> reference models (OpenVLA-OFT, π0, π0-fast, UniVLA).
>
> **The released corpus is therefore a fine stress corpus and a fine relative
> comparator, but NOT an unbiased robustness sample.** A policy-vs-policy
> difference measured on it is confounded with similarity to those four models.
>
> **Direct consequence for Phase B:** a frequency estimate over this corpus is
> not an estimate of "how often this policy fails" — it is an estimate over a
> population selected to be hard for four specific models. This independently
> re-justifies DG-5b: it is a second reason we must not ship a cardinal severity
> and must leave prevalence to the client.
>
> The category/difficulty counts below remain accurate **as a description of the
> released 10,030**; they are not a description of the perturbation space.



**Date:** 2026-09-15 · **Status:** VERIFIED from `task_classification.json` (10,030 rows)
· Settles `critical`'s AS-3, the highest-value open unknown

### What it is

Not a generator. Perturbations are **baked into pre-generated task instances**,
with the perturbation encoded in the task *filename*. LeRobot's own loader gives
it away:

```python
_LIBERO_PERTURBATION_SUFFIX_RE = re.compile(
    r"_(?:language|view|light)_[^.]*|_(?:table|tb)_\d+")
# "LIBERO-plus perturbation variants encode the perturbation in the filename
#  but on disk only the base `.pruned_init` exists"
```

**There is no API to say "yaw +15°, everything else nominal".** You select an
instance that already exists.

### The corpus

10,030 instances, ~2,400–2,600 per suite. Each carries **one** primary
`category` and a difficulty level:

| Category | n | | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|
| Sensor Noise | 1601 | | 177 | 500 | 300 | 383 | 241 |
| Camera Viewpoints | 1599 | | 199 | 360 | 325 | 228 | 487 |
| Robot Initial States | 1550 | | 248 | 292 | 374 | 285 | 351 |
| Language Instructions | 1537 | | 328 | 280 | 259 | 328 | 342 |
| Objects Layout | 1525 | | 348 | 302 | 329 | 257 | 289 |
| Light Conditions | 1142 | | 111 | 197 | 244 | 242 | 227 (+121 unlabelled) |
| Background Textures | 1076 | | 233 | 271 | 263 | 163 | 146 |

**But the single `category` label is not the whole truth.** 3,279 of 10,030
names encode **more than one** perturbation token, e.g.

```
..._view_0_0_100_0_0_initstate_13    <- viewpoint AND initial state
```

So a third of the corpus is multi-factor while being labelled with one primary
category. Reporting per-category success rates without accounting for that would
attribute a combined effect to one factor.

### What this does to counterfactual attribution — less than feared

`PLAN.md` §5.1's probe re-runs the same seed with one knob reverted. That was
specified against a knob API we do not have. But the mechanism survives, in a
different form:

1. **The ~6,750 single-factor instances are already interventions.** For those,
   *nominal vs perturbed* **is** the counterfactual, and the `category` label
   names the factor. No probe needed — the corpus did the intervention for us.
2. **The parameters are in the filename.** `view_0_0_100_0_0` is parseable, so
   multi-factor instances can be matched to single-factor ones by string
   structure. Revert-one-knob becomes a **lookup**, not a generation.
3. **Coverage is not guaranteed.** Whether the matching single-factor instance
   exists is an empirical property of the corpus, per base task. Where it does
   not, that row is **correlational** and must be labelled so — which the
   manifest already supports.

### Consequences

- The LIBERO-plus adapter is a **task-selection** layer, not a knob-setting one.
  `PerturbationSpec` maps to *"which instance"*, not *"which parameters"*.
- `SUPPORTED_KNOBS` for that adapter is the set of factors for which a matching
  instance can be resolved — determined by parsing the corpus, not declared.
- **Difficulty level is not magnitude on a single axis** and must not be plotted
  as though it were. It is a stratification the benchmark authors chose.
- **`Language Instructions` has 1,537 instances**, so the axis exists in the
  corpus even though we declared `language_grounding` out of scope (DG-11) for
  lack of a detector. The constraint is ours, not the benchmark's — worth
  restating honestly in the readout.
- Still unchecked: `benchmark_scripts/` in the LIBERO-plus repo may contain
  generation tooling. If it does, the knob API might be recoverable. Worth one
  look before building the adapter.

---

## F4 — π0.5 does not fit on this GPU. The anchor exists, but not here.

**Date:** 2026-09-15 · **Status:** VERIFIED by direct test · settles DG-5

### The test

Three runs, narrowing the cause:

| Test | Result |
|---|---|
| `lerobot-eval`, `MUJOCO_GL=egl` | `torch.OutOfMemoryError` — 7.40 GiB in use, 18.75 MiB free |
| `lerobot-eval`, `MUJOCO_GL=osmesa` + `expandable_segments` | **NOT EVIDENCE ABOUT VRAM.** Died at OpenGL init (`AttributeError: 'NoneType' object has no attribute 'glGetError'`) and never reached the model. It shows osmesa is broken in this venv, nothing more. |
| **Weights only, no simulator, no renderer** | **`WEIGHTS DO NOT FIT ON 8 GB`** — 7.34 GiB free at start, OOM during `.to("cuda")` |

**The finding rests on tests 1 and 3 only.** Test 2 must not be read as
corroboration — it never loaded the model, so it says nothing about memory. Two
failures are not two pieces of evidence when one of them failed for an unrelated
reason.

Tests 1 and 3 are sufficient on their own. **This is not the lerobot#3098
CUDA/EGL contention both sessions assumed:** PyTorch held 7.01 GiB of 7.53
*before any render context existed*, and test 3 reproduced the OOM with no
simulator at all. Freeing the rendering context cannot buy back 7 GiB, so
`osmesa` was never going to rescue it regardless of whether it worked.

### What this settles

`PLAN.md` §5.2 offered π0.5 as a deferred *reproduction anchor* — the one LIBERO
checkpoint LeRobot maintainers confirm reproduces (97.5%), whose job is to
**exonerate our environment** rather than grade a policy. DG-5 flagged that a
contingency nobody has smoke-tested is not a contingency. It was right.

**On this machine the anchor does not exist.**

### But the anchor is not dead — it is a different machine

π0.5 needs >7.34 GiB. **Colab and Kaggle free-tier T4s have 16 GB** — roughly
double this laptop (`MODELS_AND_COMPUTE.md` §6). The anchor is:

- **inference only** — no fine-tuning, so no LoRA memory
- **one-off** — run once to clear the environment, never repeated
- **small** — 400 episodes at most, well inside a 9–12 h session cap

So it fits a free tier almost perfectly. Two caveats: `MUJOCO_GL=egl` in hosted
notebooks is a known pain point and must be verified before planning around it;
and an environment cleared *on a T4* is not strictly the same environment as
this laptop — different GPU, different driver, therefore different `runtime`
fingerprint. That weakens the exoneration but does not void it: the things the
gate is actually checking (control mode, un-normalisation, action chunking,
dataset revision, simulator version) are machine-independent.

### Consequences

1. **`PLAN.md` §5.2 must stop offering the anchor as a local contingency.**
   Rewritten to say: not available on this hardware; available on a 16 GB free
   tier; verify EGL there first.
2. **`MODELS_AND_COMPUTE.md` §1 is wrong.** It lists π0.5 inference as "tight"
   on 8 GB. It is impossible. Corrected.
3. VLA-Adapter's gate now stands alone locally, with exactly the ambiguity DG-5
   warned about — unless we spend a free-tier session on the anchor first.

### Process note

The first smoke attempt (Sep 11) reported `exit 0` and I nearly recorded it as a
pass. That was the **launcher wrapper's** exit code; the real result went to a
session-scoped scratchpad that a restart cleared, and the checkpoint cache was
28 KB — it had never downloaded. **Logs for anything whose result matters now go
to `experiments/repro/logs/` inside the repo.** A result that does not survive a
session restart is not a result.

---

## F3 — Why VLA LIBERO numbers don't reproduce: a catalogue of causes

**Date:** 2026-09-11 · **Status:** compiled from `docs/LANDSCAPE.md`; issue states
as of the survey. Several items marked where we verified them ourselves.

This is deliverable content, not just background. Nearly every benchmark in this
field has an open issue where published numbers do not reproduce — RoboCasa,
CALVIN, RLBench, ManiSkill, SimplerEnv, LIBERO; Meta-World's are unreproducible
*by construction* after a v1→v2 reward rewrite. **That reframes our reproduction
gate from pedantry into the thing that separates a measurement from a number**,
and it is worth saying in the readout.

### The distinction that decides which risks apply to us

| | Affects | Applies to us? |
|---|---|---|
| **Evaluation reproduction** — run a *published checkpoint*, get a different number | our Phase 2 | **YES — directly** |
| **Training reproduction** — retrain from scratch, get a different checkpoint | Phase 5 only | Only when we fine-tune |

Most of the angry GitHub issues conflate these. Sorting them matters: a
multi-GPU batch-size mismatch cannot affect us while we are only *evaluating*
someone else's weights.

### Causes affecting EVALUATION — these are our risks

**1. Simulator version changes physics.** MuJoCo 3.4.0's box-box collision fix
broke LIBERO's stored init states. `libero_spatial` task 5: SmolVLA **80% →
28%**, OpenVLA-OFT+GRPO 98% → 12–19%; π0.5 comparatively robust (90% → 86%).
A fresh install resolves to 3.8.1 — the broken side. LeRobot pins
`mujoco<3.9.0`, which guards API breaks and **not** behavioural ones.
[lerobot#4390](https://github.com/huggingface/lerobot/issues/4390);
[PR#4465](https://github.com/huggingface/lerobot/pull/4465) pinning
`>=3.2.7,<3.4.0` is **open, approved by one reviewer, unmerged**.
*We verified this ourselves and are now on 3.3.7 — see F2.*

**2. Batch size silently changes the evaluation *sample*, not just its speed.**
[lerobot#2850](https://github.com/huggingface/lerobot/issues/2850) — **closed,
genuinely fixed** — `--eval.batch_size` was capping how many unique initial
states got sampled. Long-horizon `libero_10` was hit hardest. This is the
subtle one: batch size reads like a throughput knob, and it was determining
*which episodes ran*. LeRobot's docs now advise setting `--eval.batch_size`
equal to episodes-per-task and keeping `--env.init_states=true` when comparing
two policies on the same episodes.

**3. Action-chunk mismatch.** `--policy.n_action_steps` differing from what the
checkpoint was trained with. Community-attributed as a likely cause of the
`libero_10`-specific gap. The published `smolvla_libero` config ships
`n_action_steps: 1` against `chunk_size: 50`, and LeRobot's own π0.5
reproduction passes `--policy.n_action_steps=10` explicitly on the command line
— an override that is **not in any config file**.
*We measured the throughput consequence ourselves: 3.95 vs 17.0 steps/s.*

**4. The dataset moved under people.** `HuggingFaceVLA/libero` has been
re-uploaded. In [#1369](https://github.com/huggingface/lerobot/issues/1369) a
maintainer redirected users to re-download a **silently modified** dataset.
LeRobot's docs now warn: *"Pin `--dataset.revision=<commit-sha>` when reporting
results."* **Make this policy on every run.** Our `PREDICTIONS.md` already pins
`a1aaacb7…`.

**5. Too few episodes.** 10 vs the recommended 50 per task can swing
Spatial/Long by **15–20 points**. At 10 eps/task, n=100 per suite gives ±6–9 pp
confidence intervals — wide enough to "fail to reproduce" from sampling noise
alone. LeRobot additionally recommends averaging **3 seeds**. *Decided 2026-09-11: 10/task, with
the ±5 pp gate replaced by a CI-overlap test — `PLAN.md` §5.3.*

**6. 8 GB VRAM contention — our exact hardware class.**
[lerobot#3098](https://github.com/huggingface/lerobot/issues/3098) was filed on
an **RTX 3070 Laptop (8 GB)**: the PyTorch CUDA context and MuJoCo's EGL
rendering context contend for VRAM inside one process and evaluation can fail
outright. [PR#3235](https://github.com/huggingface/lerobot/pull/3235) adds
`--eval.process_isolated=true` (subprocess with `MUJOCO_GL=osmesa`) — **open,
unmerged**. Workaround today: force CPU rendering via `MUJOCO_GL=osmesa` or
`glfw`, at a speed cost.

**7. Power state.** Not in any issue, and ours: a laptop on battery throttles
the GPU to a 15 W cap and ~180 MHz, ~25× slower, while `nvidia-smi` still
reports `utilization.gpu 100%`. Affects timing, not accuracy — but it caused a
peer session on this machine to log a run as *"system_failure, cause unknown"*.
See F2 and the provenance block in `run_repro.sh`.

### Causes affecting TRAINING reproduction — only bite us in Phase 5

**8. Multi-GPU recipes do not transpose to one GPU.** VLA-Adapter
[#46](https://github.com/OpenHelix-Team/VLA-Adapter/issues/46) reports 94% vs
~98%, traced to a **batch-size mismatch against the official multi-GPU recipe**.
Effective batch size is a function of per-device batch × devices × accumulation
steps, and published recipes usually state only the first. Reproducing a
multi-GPU recipe on one 8 GB card requires deriving the *effective* batch and
matching it with gradient accumulation — plus LR scaling, which most papers do
not specify.

This is instructive rather than damning: **it has an identified cause**, which
is exactly the class of environment bug our gate is designed to catch. Contrast
SmolVLA, where nine issues have produced no diagnosis.

**9. Unstated training config generally.** [#3287](https://github.com/huggingface/lerobot/issues/3287)
asks for learning rate, scheduler, batch size, GPU count and commit hash to
replicate SmolVLA on LIBERO. **No maintainer answer visible.** The requester's
numbers: 83.0 / 70.0 / 70.0 / 44.8 against ~90 / ~96 / ~92 / ~71.

### What the maintainers actually say — and don't

| Issue | State | Maintainer response |
|---|---|---|
| [#3264](https://github.com/huggingface/lerobot/issues/3264) | **open** since 2026-04-02 | **none** |
| [#2107](https://github.com/huggingface/lerobot/issues/2107) | **open** | none |
| [#2354](https://github.com/huggingface/lerobot/issues/2354) | **open**, assigned | none |
| [#3628](https://github.com/huggingface/lerobot/issues/3628) | **open** | none — A100, official checkpoint, official CLI |
| [#1316](https://github.com/huggingface/lerobot/issues/1316) | closed "completed" | 55 comments; users still reported 0.672 Spatial in Apr 2026. **Closed, not resolved.** |
| [#1369](https://github.com/huggingface/lerobot/issues/1369) | closed "completed" | redirected users to a re-uploaded dataset |
| [#2850](https://github.com/huggingface/lerobot/issues/2850) | closed | **genuinely fixed** (batch-size/init-states) |

**The signal that should change a decision:** LeRobot's LIBERO documentation has
a *"Reproducing published results"* section written **entirely about π0.5** —
`lerobot/pi05_libero_finetuned`, 97.5% average, reproducing OpenPI's 96.85%.
There is **no equivalent SmolVLA claim anywhere in the documentation.** With
nine open reproduction issues, that absence reads as deliberate.

### What we do about it

- **Subject under test: VLA-Adapter** (1B, MIT, ~2 GB, LoRA-able in 8 GB).
  Known gap has a diagnosis; SmolVLA's does not.
- **Deferred but available: π0.5 as a reproduction *anchor*.** See `PLAN.md` §5.2.
- Every cause above that we can control is now either pinned, recorded in
  provenance, or an explicit open decision.

---

## F2 — MuJoCo ≥3.4.0 silently breaks a LIBERO task. We were on the broken side.

**Date:** 2026-09-11 · **Status:** VERIFIED against the upstream issue, and FIXED here
**Source:** [lerobot#4390](https://github.com/huggingface/lerobot/issues/4390)

### What it is

MuJoCo 3.4.0 shipped a *correct* bugfix to box-box collision distance. LIBERO's
stored initial states depended on the old behaviour. In `libero_spatial` task 5
("pick up the black bowl on the ramekin and place it on the plate") the bowl no
longer settles onto the ramekin — it ends up tilted on the rim, roughly half
overhanging. The bowl's collision mesh is 40 box geoms and the ramekin's is 25,
so this task is maximally exposed to exactly that change.

### Measured impact (reporter, 50 episodes on task 5)

| Policy | healthy MuJoCo | broken MuJoCo |
|---|---|---|
| **SmolVLA 0.45B finetune** | **80% @ 3.2.7** | **28% @ 3.8.1** |
| OpenVLA-OFT GRPO | 98% @ 3.2.7 | 12% @ 3.8.1 |
| OpenVLA-OFT full SFT | 96% @ 3.2.7 | 52% @ 3.9.0 |
| π0.5 (LeRobot) | ~86–90% | ~86–90% (stable) |

Broken: 3.4.0–3.8.1. Healthy: 3.2.7, 3.3.0, 3.3.7.

### Why it matters more than a one-task bug

A fresh `pip install` of `lerobot[libero]` resolves to **3.8.1** — the broken
side. LeRobot pins `mujoco<3.9.0`, which guards against API breaks and **does
not guard against behavioural ones**. We installed today and got 3.8.1.

Left uncaught this is the worst class of error for this project: **a physics
change masquerading as a model failure.** Task 5 would have appeared in our
stress results as a genuine weakness, been clustered, attributed to whichever
perturbation happened to co-occur, and written into a Data Gap Manifest row
recommending that a client spend money collecting bowl-on-ramekin
demonstrations. Every downstream artifact would have looked sound.

It is also a concrete instance of the risk `PLAN.md` §7c discriminator #1 exists
to catch — *can a competent controller still solve this task?* — and evidence
that the discriminator needs to run per *task*, not per suite.

### Action taken

Pinned and reinstalled: **mujoco 3.8.1 → 3.3.7**. EGL rendering re-verified.

### Consequences for what we already ran

- The `n_action_steps` calibration was run under 3.8.1. Its **timing** numbers
  stand — physics does not affect throughput — but any success/failure outcome
  from it is suspect. It used `libero_spatial` task 0, not task 5, so probably
  unaffected; not worth re-running for timing alone.
- Partially relevant to [#3264](https://github.com/huggingface/lerobot/issues/3264):
  a 52 pp drop on 1 of 10 tasks is ~5 pp of suite-level Spatial, against that
  report's 27 pp shortfall. Contributory, not the whole explanation.
- **`mujoco` version now belongs in the reuse-relevant part of provenance**, not
  just the runtime record. It changes results, not merely timings.

---

## F1 — Policy transfer, and how much of the brittleness is behaviour cloning

**Date:** 2026-09-11 · **Status:** analysis, partly testable by our own Phase 2

### 1.1 What has to line up for a policy to transfer

Three things, in increasing order of subtlety — the last is the dangerous one.

| | What must match | Failure mode if it doesn't |
|---|---|---|
| **Action space** | shape *and* semantics: `Box(-1,1,(7,))` for LIBERO (6-D eef delta + gripper); delta vs absolute; eef vs joint | wrong shape = immediate error; wrong semantics = smooth, confident, wrong motion |
| **Observation space** | camera count/placement/resolution, state dim, **and the feature keys** | LeRobot enforces `.images.*` naming because the keys are baked into the normalisation layer |
| **Normalisation statistics** | per-dataset action mean/std (`unnorm_key`) | **confident, plausible, completely wrong motion** — indistinguishable from a weak model |

The third is why the proposal says environment mismatch "can manufacture apparent
model failures." A mis-set `unnorm_key` does not crash. It produces a robustness
curve.

### 1.2 Why transfer works at all

A VLA is two pieces with very different transfer properties:

- **the vision-language backbone** — essentially a VLM; transfers well, and is
  what Open X-Embodiment pretraining across 22 embodiments buys
- **the action head** — embodiment-specific; the part that must be adapted

Hence the standard recipe, and hence `smolvla_base` → `smolvla_libero` being a
fine-tune rather than a from-scratch train.

Roughly, easiest to hardest:

| Shift | Outcome |
|---|---|
| New instruction, same embodiment and scene | often zero-shot |
| New objects, same embodiment | partial |
| **Camera moved** | **often fails** — our prediction P1 |
| Different embodiment | new head at minimum |

**A viewpoint shift is a mild form of "different dataset."** The observation
distribution moves while the task does not. Our entire stress test is transfer
failure measured at small radius.

---

### 1.3 Is it all behaviour cloning? — No, and the distinction decides the fix

Worth separating carefully, because "it's a BC problem" and "it's a coverage
problem" imply different remediations and we sell remediations.

**Genuinely attributable to behaviour cloning:**

- **No corrective data.** Training on expert demonstrations means the policy has
  never seen how to recover from its *own* errors. It only knows the expert
  manifold. Step off it and there is no training signal at all — not degraded
  signal, none.
- **Compounding error / covariate shift.** The classic BC result (Ross &
  Bagnell, DAgger): a small per-step error moves the state distribution, which
  produces a larger error, and the cost compounds with horizon length rather
  than staying linear in it. This is a property of learning from a fixed
  distribution you then depart from.
- **No notion of progress.** No value function, no reward, no self-assessment.
  It cannot know it is failing, so it cannot change strategy.

**NOT attributable to BC — any paradigm trained on this data would share it:**

- **Viewpoint brittleness.** This is a *coverage* problem. Phase 0 measured zero
  camera-pose variation in the fine-tuning set. An RL agent trained in that same
  single-viewpoint environment would be just as brittle. Nothing about BC causes
  this; the data causes it.
- **Action-space and normalisation mismatch.** Engineering, not learning theory.

**Architectural, not BC:**

- **Memorylessness.** SmolVLA's config has `n_obs_steps: 1` — a single frame.
  The policy cannot remember that it already tried this. A recurrent or
  history-conditioned model could, trained identically by BC. Our toy's
  `retry_loop` failure is architecture and BC *together*: no memory to notice the
  repetition, no corrective data to know what else to do.

### 1.4 Evidence for the compounding-error account, from published numbers

SmolVLA's own published results:

| Suite | Published | Horizon |
|---|---|---|
| LIBERO-Object | ~96 | 280 steps |
| LIBERO-Goal | ~92 | 300 |
| LIBERO-Spatial | ~90 | 280 |
| **LIBERO-Long** | **~71** | **520** |

The long-horizon suite is ~20 points worse than the others, and it is the one
with roughly double the step cap. That is the shape compounding error predicts:
cost growing with horizon, not with task difficulty per se. It is consistent
across the other policies too — π0.5 scores 92.4–98.8 with Long also lowest.

Not proof. Long-horizon tasks are also multi-stage and semantically harder, and
the two explanations are confounded in this data. But it is the cheapest
available evidence and it points the right way. **A testable discriminator:**
if it is compounding error, failure probability should rise with *elapsed steps*
within an episode roughly independently of which subtask is in progress; if it
is task difficulty, failures should cluster at specific subtask boundaries. Our
phase localisation can separate these — it is exactly what stage localisation is
for, and it costs nothing beyond the runs we are already doing.

### 1.5 Why this matters commercially

It maps directly onto the fixability classes in `PLAN.md` §7b.1, and picking the
wrong one wastes the client's money:

| Cause | Correct remediation | Wrong remediation |
|---|---|---|
| Coverage (viewpoint) | augmentation or targeted collection | RL fine-tuning |
| BC / no recovery data | **corrective** demonstrations — failed attempt + recovery — or interactive collection (DAgger-shaped) | more clean expert demos, which is what a vendor defaults to selling |
| Architecture (memorylessness) | history-conditioned model; escalate to the model team | any amount of data |

The middle row is the sharpest: **more clean demonstrations cannot fix a
recovery gap**, because clean demonstrations are precisely what created it.
Recommending "collect 5,000 more demos" for a recovery failure is the most
expensive way to not fix the problem, and it is the default recommendation of
anyone whose business is collecting demonstrations. Being the party that says
"not this one — you need corrective data, and less of it" is the differentiated
position.

---

### 1.6 What the proposal says, and what we should add

It addresses transfer in three places, all as *warnings* rather than tasks:

- §7.1 — pin the action un-normalization config; mismatch "can manufacture
  apparent model failures"
- §7.1 — for GR00T, "match the policy action representation/control mode"
- §7.3 — policies differ in backbone, action representation and control mode, so
  cross-model work yields "**two robustness profiles rather than a universal
  model ranking**"

The third is the proposal at its best and should be elevated into the readout.
A client will ask "so which model is better?" and the honest answer is that we
cannot say — only which is more robust *along the axes we measured, in this
environment*. That refusal belongs in the document, not in the meeting.

**Three things to add:**

1. **A pre-flight equivalence gate.** The proposal says match the control mode
   but never says how you would know you had. Before any cross-policy
   comparison, each policy must reproduce **its own** published nominal
   baseline. If a second policy lands at 40% where published says 90%, the
   config is wrong and every robustness number downstream is noise about our
   setup, not a finding about the model. Make this a hard precondition for
   comparison, not a one-off Phase-1 check.
2. **Action config goes in `identity()`.** `control_mode`, `unnorm_key`, action
   space and checkpoint revision belong in the fingerprint, so a mismatch
   changes the policy id and misses the cache rather than silently mixing
   incompatible rollouts. Mechanism already built (see `ARCHITECTURE.md` §4.1);
   this is the structural version of the proposal's warning, and the difference
   between "remember to check" and "cannot get this wrong."
3. **Variable-dimension actions** — already covered by G1. The proposal's
   scale-out path contemplates additional embodiments but its rollout contract
   does not address the dimensionality change; ours does.
