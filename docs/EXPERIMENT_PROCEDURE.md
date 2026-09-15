# Experiment procedure — phases, volume, GPU time

**Written 2026-09-15.** What we actually run, in what order, how much of it, and
what each phase costs.

**A–F are CAMPAIGN PHASES, not per-episode steps.** Each is a batch of many
episodes. One *episode* is: reset → ~150–520 policy steps → success or timeout.

---

## 0. The single most important planning fact

**Only rollouts captured through OUR harness can be mined later.**

| Captured by | Stores | Mineable afterwards? |
|---|---|---|
| `lerobot-eval` | success bit, reward, mp4 | **NO** — object poses, contacts, gripper state and phase signals were never recorded |
| our harness | full per-step state incl. `_gt_*`, `scene_descriptor`, fingerprint | **YES** — mining re-runs over stored traces without re-simulating |

This is a *designed* property: `run_cell` persists a rollout **before** anything
classifies it, so every load re-classifies from scratch. Change a detector
threshold, re-mine the same traces, get a different answer — no GPU needed.

**The consequence for planning:** deciding to use `lerobot-eval` for a phase is
deciding that phase can never be mined. It is not a reversible optimisation.

- Phase A (nominal baseline) via `lerobot-eval` — **accepted**: its job is a
  success number to compare against published, and the reproduction question is
  better answered by the standard tool.
- Phases B and D **must** go through our harness, or the mining layer has
  nothing to work on.

> **Open risk:** trace volume. Full per-step state for 10,030 episodes at ~300
> steps is large. Frames dominate and are optional (`image_dir=None` writes no
> images). Measure the per-episode JSONL size on the first real batch before
> committing to a full-corpus run.

---

## 1. The reframe: sampling, not searching

LIBERO-plus is a **fixed corpus of 10,030 pre-generated instances** (F5), not a
knob API. So this is **survey sampling from a known finite population**, not
optimisation over a continuous space.

That is a better position than it sounds: we know the sampling frame, we can
stratify by (category × difficulty), and per-stratum rates are unbiased by
construction.

**And the population is small enough to run whole:**

```
10,030 instances × ~9 s/episode  ≈  25 GPU-hours
```

at the measured 17 steps/s with `n_action_steps=10`. **The entire corpus, once
through, is about a long weekend.** That materially weakens the case for
adaptive sampling as a *discovery* mechanism — you cannot outsmart exhaustive
when exhaustive is affordable. Adaptive becomes a question of **where to spend
repeat seeds**, not where to look.

---

## 2. The phases

| Phase | What | Episodes | GPU time | Harness? |
|---|---|---|---|---|
| **A** Nominal baseline | vanilla LIBERO, 4 suites, 10/task | 400 | **1–2 h** | `lerobot-eval` |
| **B** Stratified screen | LIBERO-plus, uniform arm | 2,000–10,030 | **5–25 h** | **ours** |
| **C** Attribution | mostly free — see below | 0–500 | 0–1 h | ours |
| **D** Targeted repeats | seeds at measured boundaries | ~1,000 | **2.5 h** | **ours** |
| **E** Mining | phases, families, clusters, manifest | — | **0 (CPU)** | ours |
| **F** Remediation | 5 fine-tunes + 5 regression re-runs | ~1,500 | **10–15 h** | ours |

**Total ≈ 20–45 GPU-hours**, i.e. two to three overnight runs — not a month.

### A — Nominal baseline *(running now)*
The reference every perturbed rate is measured against, and the reproduction
question. **A screen, not a gate**: at n=100/suite a 5 pp break goes unflagged
66% of the time, so a pass means *"screen passed, gate not yet run"*.

### B — Stratified screen
Sample the corpus by (category × difficulty). This **is** the robustness
profile — with an ordinal x-axis, which E4 forces us to label as such rather
than plotting it like a dose-response curve.

**The uniform arm must stay uniform.** Every frequency claim in the manifest
rests on it; that is what `arms.jsonl` protects, and why the adaptive arm's
output can never feed a frequency estimate.

Start at ~2,000 (≈200/stratum, 5 h). Escalate toward the full corpus only if
stratum intervals are too wide to act on.

### C — Attribution, and it is mostly already done
**~6,750 of 10,030 instances are single-factor**, so *nominal vs instance* **is**
the intervention and needs no additional runs at all — B already collected it.
For the 3,279 multi-factor instances, look up a single-factor counterpart;
where none exists the row is **correlational** and must say so.

### D — Targeted repeats
Where B shows a boundary, spend seeds to tighten the interval. **Allocation, not
search.** This is the adaptive arm on a finite frame.

### E — Mining — CPU only, and re-runnable
Runs over stored traces. Costs no GPU and can be repeated as detectors improve,
which is exactly why §0 matters.

### F — Remediation
Three arms at **one** budget (targeted data / re-rendered augmentation /
frozen-policy adapter), plus a data-response curve on arm A for the top manifest
row. **5 fine-tunes, not 9** (DG-9). LoRA on a 1B policy fits 8 GB.

---

## 3. Is this brute force?

Partly, and deliberately.

**B is close to exhaustive by design** — the frame is finite and affordable, and
exhaustive sampling has a property no clever method has: the frequency estimate
is unbiased without argument.

**The intelligence is not in the search.** It is in C (interventional
attribution rather than correlation), E (why it failed, not just that it did),
and F (which remediation actually recovers success per unit cost). Those are the
parts nobody else does.

**Where genuine method is required is D** — but on a finite frame that is
allocation, and the honest framing is optimal-allocation-under-budget, not
search.

---

## 4. Sequencing

1. **A finishes** → per-suite rates vs published, with intervals
2. **LIBERO `PhaseSegmenter`** → unblocks mining on real traces *(cheapest
   high-value item; does not need LIBERO-plus)*
3. **Policy adapter through our loop** → lets B capture traces
4. **LIBERO-plus in a second container** → it uninstalls vanilla LIBERO
5. **B** → the robustness profile
6. **C, E** → attribution and mining, CPU-cheap, iterate freely
7. **D, F** → tighten and validate

Steps 2 and 3 gate everything; neither needs a GPU to write.

---

## 5. Revisions from the methods survey (2026-09-15)

`docs/FAILURE_MINING_METHODS.md` completed after this document was written. Four
changes, three of which add phases.

### 5.1 Attribution becomes ddmin, not revert-one-knob

**Revert-one-knob structurally cannot see interactions.** A failure requiring
viewpoint AND initial state *jointly* shows a small effect on every single
reversion, and we report "no factor responsible" — which today renders
**identically** to a genuine null and to multiple sufficient causes. Three
distinct situations, one output string.

**ddmin** (delta debugging) finds the **minimal failure-inducing subset** in
O(n²) same-seed re-runs rather than 2ⁿ. Every re-run is a rollout we can already
do, so the cost is episodes, not new machinery.

**Output vocabulary must grow to match:** `single_factor`, `minimal_subset`
(with the subset), `multiple_sufficient`, `no_factor_identified`. Collapsing
these is the bug, not the arithmetic.

### 5.2 NEW PHASE G — language sensitivity probe *(run EARLY)*

| | | Episodes | GPU |
|---|---|---|---|
| **G** | blank-instruction + directed-substitution | ~2 per instance probed | ~1 h for 1,500 |

**Not optional, and it gates a taxonomy family.** LIBERO-Plus reports
OpenVLA-OFT "largely unchanged" with **no language input at all** — it
"degenerates into … a Vision-Action model". Their goal-replacement probe then
shows success dropping nearly to zero **while the model still executes the
ORIGINAL target's trajectory**.

Three outcomes, and the third is the one a success-rate-only design misses:

| Probe result | Reading |
|---|---|
| blank instruction ⇒ success unchanged | **insensitivity** — the policy is a VA model |
| goal replaced ⇒ trajectory follows the NEW target | **comprehension** |
| goal replaced ⇒ success collapses, trajectory follows the **ORIGINAL** target | **partial grounding** — looks like comprehension in aggregate, is not |

**Why it gates:** if the policy is insensitive, a `language_grounding` family is
near-unpopulated, and a classifier carrying it will either abstain or *quietly
label something else with its name* — which κ cannot catch. So this probe
decides whether that family is admissible, and must run **before** any
language claim reaches a client.

This reverses part of DG-11: we declared the axis out of scope for lack of a
detector, but the corpus has **1,537 language instances** and the probe is one
rollout with one word changed. The constraint was ours and smaller than claimed.

### 5.3 NEW PHASE H — taxonomy validation by name-based re-assignment

| | | Cost |
|---|---|---|
| **H** | held-out judge assigns episodes from cluster NAMES only | CPU + human time |

**Nobody validates a taxonomy as correct** — that is the survey's negative
answer to our open problem #2. κ is the wrong instrument *regardless of
threshold*: `{failure on a Tuesday, failure not on a Tuesday}` scores **κ = 1.0**.
That is construct validity, not reliability.

So: **supplement, do not raise.**

- report **percentage agreement and Gwet's AC1** beside κ — κ is depressed by
  exactly the class imbalance failure taxonomies always have (kappa paradox)
- **name-based re-assignment**: give a held-out judge *only* cluster names and
  descriptions; have them assign held-out episodes; measure agreement with the
  induced partition

This genuinely tests the partition rather than the raters. It establishes a
cluster has a **communicable common property** — *not* that the property is the
right one. **Report it as partial.**

### 5.4 Every frequency claim carries the selection caveat

The released 10,030 were filtered from 14,000 by **deleting tasks solved by all
or most of four reference models** (OpenVLA-OFT, π0, π0-fast, UniVLA). A rate
over that corpus is **not a robustness rate** — it is a rate over a population
selected to be hard for four specific models, and cross-policy comparison on it
is confounded with similarity to them.

This is a second, independent reason not to ship a cardinal severity.

### 5.5 Wave E forks on one unanswered question

LIBERO-plus **is a generator** (14,000 candidates; single-dimension perturbation
is its unit of operation), so revert-one-knob is constructible *in principle*.

**Unverified: whether the generation code is distributed.**

- **If yes** — we get a real knob API, the original probe design works, and
  Wave E's instance-selection machinery (E1) is largely unnecessary.
- **If no** — instance selection stands, and attribution is limited to what the
  released corpus happens to contain.

**This is the highest-value outstanding check and it is a repo question.**
Answer it before building either version.

### Revised phase table

| Phase | Episodes | GPU | Notes |
|---|---|---|---|
| A nominal baseline | 400 | 1–2 h | running |
| **G language probe** | ~3,000 | **~1 h** | **early — gates a taxonomy family** |
| B stratified screen | 2,000–10,030 | 5–25 h | ours, mineable |
| C attribution + **ddmin** | 500–2,000 | 1–5 h | grew; ddmin is O(n²) |
| D targeted repeats | ~1,000 | 2.5 h | |
| E mining | — | 0 | CPU |
| **H taxonomy validation** | — | **0** | CPU + human |
| F remediation | ~1,500 | 10–15 h | 5 fine-tunes |

**Total ≈ 21–51 GPU-hours.** Still two to three overnight runs.
