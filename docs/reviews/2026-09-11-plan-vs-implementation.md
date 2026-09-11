# Review — `PLAN.md` against the implementation, and a design for controls

**Date:** 2026-09-11 · **Scope:** every claim `PLAN.md` makes about what is built,
checked against `vla_harness/` and `experiments/` as they stand; plus §2, a proposed
new section on experimental control that the plan currently lacks.
**Method:** read + executed + measured. Companion to the architecture-flow,
methodology and models-and-compute reviews.

**Status key:** `open` · `discussing` · `accepted` · `wontfix` · `fixed`

---

## Part 1 — `PLAN.md` against what exists

### Summary table

| # | Finding | Class | Severity | Status |
|---|---|---|---|---|
| P1 | Every manifest row the code emits is, by §9's own rule, a hypothesis | **contract** | **high** | open |
| P2 | G5 is declared and unwired — `reproducibility_floor()` is still never called | guarantee | **high** | open |
| P3 | §8's cost model has no power-state term; the repro scaffold just measured a 25× swing | **cost** | **high** | open |
| P4 | §0's "supply-side first" decision has been inverted in practice and not recorded | planning | medium | open |
| P5 | G2 is the only guarantee never exercised, and it fails open at the worst moment | guarantee | medium | open |
| P6 | §4's "GATE PASSED" is reported without the caveats now known | reporting | medium | open |
| P7 | The κ gate at week 5 has no annotation tooling and none is scheduled before it | planning | medium | open |
| P8 | Mining-layer config sits outside `identity()` — safe today, by accident | contract | medium | open |

**Verified as built and honest** — the guarantees that hold up under inspection:

- **G1** variable-dimension actions: `list[float]` + `action_dims` throughout, and
  the new nullable terminal `Step.action` is documented where it lands.
- **G3** schema versioning: `SCHEMA_VERSION` was correctly bumped to `2.0` when
  `Rollout.fingerprint` and the terminal step changed the format, and
  `rollout_from_dict` refuses a foreign major. The discipline held under its first
  real test, which is the only evidence that matters for a versioning scheme.
- **G4** knob names validated against `LIBERO_PLUS_FACTORS` at construction, not by
  convention. (See M17 for why the *granularity* is still wrong.)
- **G6** `identity()` is declared on both Protocols, so a third-party adapter is told
  what to implement rather than crashing in `make_fingerprint`.
- **G8/G9** detectors declare `requires` and skip cleanly; `mining/` imports nothing
  from `envs/` or `policies/`.
- **G10** resume is now *verified* rather than trusted, and the fingerprint is
  deliberately broader than the cache key. The reasoning in `run_cell`'s docstring
  about why mismatch re-runs rather than raises is correct.
- **`reproducibility_floor` takes no store.** The comment explaining why — threading
  one through would serve the second repeat from the first repeat's cache and report
  `floor_pp = 0.0` by construction — is the single best defensive comment in the
  codebase.
- **`experiments/repro/` is correctly quarantined from `vla_harness`.** "If the
  number comes out wrong we must be able to tell 'the checkpoint doesn't reproduce'
  from 'our adapter is buggy'" is exactly right, and most teams get this wrong.

---

### P1. Every manifest row the code emits is, by §9's own rule, a hypothesis — *[verified]*

§9 defines the Data Gap Manifest schema and states the rule plainly: "Fields marked
✱ are mandatory; **a row missing one is a hypothesis, not a manifest row.**"

`manifest.make_row()` emits 14 top-level keys:

```
row_id · severity · evidence_strength · task_scope · failure_family · symptom
trigger · boundary · evidence · supply_side · coverage_required
validation_plan · validation_result · provenance
```

Four schema fields are absent, **three of them mandatory**:

| Field | §9 status | Emitted? |
|---|---|---|
| `fixability` | ✱ mandatory | **no** |
| `discriminators` | ✱ mandatory | **no** |
| `failure_cost` | ✱ mandatory | **no** |
| `envelope` | optional | no |

These are not incidental fields. They are §7b and §7c — the fixability class, the
discriminator results, the failure-cost weighting and the operating envelope. Which
is to say: **the entire section of the plan that distinguishes this project from a
robustness report is specified in the schema and absent from the artifact.** The
manifest the code produces today is the proposal's original table plus an evidence
strength flag.

That is a reasonable state for a prototype. It is not a reasonable thing to leave
undocumented in a plan going to reviewers, because §9 reads as a description of what
the artifact contains.

**Direction.** Two options, and the choice is a real one. Either emit the fields with
explicit `null` + a `not_yet_determined` reason (consistent with how
`validation_result: null` is already handled, and it makes the gap visible in the
rendered manifest), or mark them in §9 as Phase 3/4 fields not present in the Phase 1
artifact. The first is better: it puts the missing work in front of the reader
instead of in a plan they may not have read.

---

### P2. G5 is declared and unwired — *[verified]*

G5 promises "Reproducibility, measured not assumed", and §2's assumptions table
schedules `reproducibility_floor()` for "**Wk 3, day 1** — before any stress
campaign", calling its absence a state in which "every causal claim is void."

`reproducibility_floor` is defined in `runner.py:213`. **Nothing calls it.**
`grep` across the tree returns only its own definition and the `floor_pp` parameter
it was built to supply. `counterfactual_probe(floor_pp=0.0)` therefore gates on a
permanent `max(20, 0) = 20`.

This was finding #5 in the architecture review and is still open, so this is a
restatement rather than a discovery — but it belongs here because `PLAN.md` §2 lists
G5 in the "no fundamental flaw in month 2" table, and a guarantee that exists only as
an uncalled function is not a guarantee. On the toy the floor genuinely is 0, so this
cannot bite until the moment it matters most.

**Direction.** Beyond wiring it: the toy cannot test the machinery, because a
deterministic fixture makes `floor_pp = 0.0` the correct answer and an unwired floor
indistinguishable from a wired one. Add a scripted policy with a seeded noise draw in
its action, so the floor is known-nonzero and the gate can be shown to move. That is
a ten-line fixture and it is the only way to know the apparatus works before a
stochastic action head arrives.

---

### P3. §8's cost model has no power-state term, and the swing is 25× — *[verified in-repo]*

`experiments/repro/run_repro.sh` records, in a comment beside the provenance block:

> Power state is part of the result, not trivia. A wall-clock number taken at a 15 W
> battery cap and one taken at 35 W differ by ~25x on identical work (independently
> confirmed: 601.7 s vs 24.0 s on the same input). `nvidia-smi`'s `utilization.gpu`
> does NOT reveal this — it read 100% throughout the throttled window. `pstate` and
> `clocks.sm` do.

This is an excellent catch and it has not propagated. `PLAN.md` §8 builds a timing
table, an episode-cost model and the headline claim — "**the full campaign is an
overnight run on this laptop**" — on a single assumed throughput. A 25× swing turns
an eight-hour overnight run into eight days, and the tell that it is happening is
invisible in the metric everyone checks.

It compounds with the models-and-compute finding C2 (the campaign belongs on this
laptop's 16 cores rather than a 2–4 core free tier): if the laptop is the primary
runner, its power profile is now a first-class experimental variable, not an
environment detail.

**Direction.** Three changes. (1) Add a power-state row to §8 and state the assumed
profile beside the throughput numbers. (2) Put `enforced.power.limit`, `pstate` and
`clocks.sm` in the runtime fingerprint — `run_repro.sh` already collects them, so
this is plumbing, and a throttled overnight run is exactly the "config changed under
me" case the fingerprint exists to catch. (3) Assert AC power and a fixed profile at
campaign start, and fail loudly rather than silently taking 25× longer.

---

### P4. §0's "supply-side first" decision has been inverted in practice — *[verified]*

§0's decisions table:

> **First step — Supply-side data analysis.** Measure the training distribution
> before designing experiments. *Consequence: perturbation axes are chosen from
> evidence, not from the paper's priors.*

§10 schedules Phase 0 for weeks 1–2. What exists on disk is `third_party/lerobot`, an
8.7 GB `.venvs/lerobot`, and `experiments/repro/` — Phase 2 reproduction work. Phase 0
has not started.

**I think the inversion is correct and the document is what is wrong.** Given two
independent open non-reproductions of the chosen checkpoint (#3264 and #3287, both
documented in `TARGET.md`), settling the reproduction gate before committing to
SmolVLA dominates measuring a training distribution for a policy we might not use.
That is also what `MODELS_AND_COMPUTE.md` §4 instructs, and it resolves the M11
scheduling contradiction in the sensible direction.

But §0 calls it a "Decision taken" with a stated consequence, and the plan now
disagrees with the repository. For a document going to seniors that reads as drift
rather than as judgement.

**Direction.** Record the change as a dated amendment in §0 with the reason — two
open non-reproductions make the checkpoint choice the binding decision — and re-order
§10. Keep the original rationale for Phase 0; it is still right, it is just no longer
first.

---

### P5. G2 is the only guarantee never exercised, and it fails open at the worst moment — *[by inspection]*

G2 — "Observations carry optional image refs, not pixels" — is declared on
`Observation.image_refs` and threaded through `Step`. The toy has no images, so
**every value it has ever held is `{}`**. The guarantee is structurally present and
has never been executed.

That matters because of *where* it first executes: the first LIBERO rollout, which is
also the first time trace size becomes a real constraint (300 steps × two 224×224
streams). The failure mode if the adapter inlines an array instead of a path is not
an exception — `obs_state` is `dict[str, Any]` and `to_json` will happily serialise a
nested list — it is a `rollouts.jsonl` that grows by ~90 MB per episode and an OOM
several hundred episodes into an overnight sweep. Compounded by the `TraceStore`
O(N²) issue (architecture review #11), which is still open.

**Direction.** A cheap assertion with real teeth: in `rollout()`, reject any
`obs_state` value that is a list/array above some small length, or any non-JSON-scalar
that is not a declared ref. Ten lines, and it converts a silent 90 MB/episode leak
into an immediate, legible failure on the first LIBERO episode.

---

### P6. §4's "GATE PASSED" is reported without the caveats now known — *[verified]*

§4 records: "**Result (prototype, `experiments/oracle_test.py`): 3/4 — GATE
PASSED.**" It then honestly documents the `no_recovery` fixture limitation.

Three things established in the methodology review are not reflected:

- The fixture's maximum achievable score is 3, and the gate passes at 3 (M4) — so
  the threshold equals the ceiling.
- Two of the three passing attributions are separated from every alternative by
  **+95 pp** (M5); the one realistic effect is +35 pp.
- The gate measures recall only. On a one-fault policy the pipeline emits a two-row
  manifest whose second row is a false positive recommending the wrong data (M2/M3).

§4 is the section a reviewer will read as "the design is validated." It should carry
what the gate does and does not establish.

**Direction.** Replace "GATE PASSED" with the recall/precision pair and a one-line
scope note. The honest version is still a good result — it is the first known-answer
test of a failure miner I have seen in this space — and it survives being stated
precisely.

---

### P7. The κ gate has no tooling and none is scheduled before it — *[by inspection]*

§10 gates week 5–6 on κ ≥ 0.5, and §2 makes κ < 0.5 a kill criterion. §7 defers the
annotation workflow — "Build once Tier 1 is producing real ambiguity worth reviewing
(Phase 3), not before" — which is sound sequencing, but Phase 3 *is* weeks 5–6. The
gate and the tooling that produces its input land in the same fortnight, with the
tooling first and no slack.

Add M13: at 30 traces across 7 families, κ's confidence interval cannot separate 0.4
from 0.7, so the kill criterion can fire on sampling noise.

**Direction.** Either move a minimal double-labelling harness into week 3 (it needs
only a trace viewer and a CSV), or move the κ gate to week 7 and say what happens to
the manifest if it fails that late.

---

### P8. Mining-layer config sits outside `identity()` — safe today, by accident — *[verified]*

`schema.py`'s `Env.identity()` docstring is admirably explicit:

> Scope note: this covers ENV and POLICY identity only. Mining-layer detector config
> (`PhaseSegmenter` radii, `LOST_TARGET_M`) is deliberately outside it, and is safe
> today only because rollouts are persisted BEFORE mining.

Correct, and the safety is conditional on a property nothing enforces. The moment any
code path persists a `Rollout` with `diagnosis` populated — which `ARCHITECTURE.md`
§7 explicitly anticipates as the normal way to re-mine stored traces — a cached trace
carries a diagnosis computed under detector settings the fingerprint cannot see. The
architecture review's finding #12 (`grasp_radius` duplicated across the L2/L3
boundary) is the same coupling from the other side.

This is the precise failure the fingerprint work was built to prevent, one layer up.

**Direction.** A separate `mining_fingerprint` on `diagnosis`, compared whenever a
stored diagnosis is read. Mining is re-runnable over stored traces without
re-simulating — that is the point of storing them — so the cheap rule is: **trace
identity gates re-simulation; mining identity gates re-mining.** Two keys, two
decisions, neither contaminating the other.

---

## Part 2 — Proposed new section: control and treatment

`PLAN.md` has §7c (data gap vs policy gap) and I8 (the clean-policy control), both
good. What it does not have is a statement of **experimental design**: what is the
treatment, what is the control, what varies that we did not intend, and how a
measured difference is licensed as real. This is the section to add, because it is
the question a senior reviewer will ask first and the docs currently answer it in
scattered pieces.

### 2.1 Three sources of variation, and only one is the treatment

Every success rate this project reports is a proportion measured under three
independent sources of variation. They are routinely conflated and they need
completely different treatment.

| # | Source | What it is | Size, measured | Shrinks with | Detected by |
|---|---|---|---|---|---|
| **S1** | **Sampling error** | finite episodes; a fair coin measured 20 times | ±20 pp at n=20 | more episodes (1/√n) | Wilson CI — already computed |
| **S2** | **Execution nondeterminism** | same config, same seed, different outcome: GPU float nondeterminism, sampling action heads (π0, GR00T), contact chaos | ~1 pp on LIBERO | repeats + more episodes | **A/A repeat** — `reproducibility_floor()` |
| **S3** | **Configuration bias** | a different `control_mode`, `n_action_steps`, MuJoCo version, power profile, detector threshold | **10–20 pp** | **nothing** | **external reference only** |

The measured sizes are not guesses. S1 comes from the harness's own `wilson_ci`.
S2 is the published LIBERO seed spread — OpenVLA reports 88.4 **± 0.8**, 84.7 ± 0.9,
79.2 ± 1.0, 53.7 ± 1.3 over 3 seeds × 500 rollouts. S3 is openvla#282 (68% against a
published 88.4%) and lerobot#3264 (73.25% against a published ~87.3%).

**The asymmetry is the whole point.** S3 is the largest term by an order of magnitude,
and it is the *only* one that is not variance. A wrong `control_mode` gives the same
wrong answer every time. In openvla#282 the reporter changed the seed from 3 to 15
and the number **stayed at 68%**.

So: repeating a run cannot detect S3, because repeating holds constant the thing that
is wrong. `reproducibility_floor()` measures S2 and will report a reassuringly small
number in exactly the situation where the result is least trustworthy.

### 2.2 Is a reproducibility test needed? Yes — three of them, and they answer different questions

"Reproducibility test" is one phrase covering three distinct experiments. Running one
and believing you have done all three is the trap.

**Test A — the A/A (null) experiment.** Run the identical configuration twice. Any
difference observed is, by construction, not a treatment effect. This is the null
distribution every later claim is judged against, and it comes in two variants that
are *not* interchangeable:

- **Same seed set, repeated** → isolates **S2 alone**. The episode set is held fixed,
  so what moves is execution nondeterminism. Call this the **execution floor**.
- **Different seed sets, same size** → gives **S1 + S2 together**. This is the total
  noise on a number you will later compare. Call this the **sampling floor**.

`reproducibility_floor()` as written does the first. Both are needed: the execution
floor tells you whether paired comparison is even meaningful; the sampling floor is
what an unpaired delta must beat.

Two defects in the current implementation. `repeats=2` and `floor_pp = max − min`
makes the estimator the **range of two samples**, which is a very noisy statistic and
biased low — with two draws you will usually underestimate the spread, and the floor
is used as a *safety threshold*, so underestimating is the dangerous direction. Use
`repeats ≥ 5` and report the standard deviation, or better an upper quantile.

**Test B — the golden reference cell (S3 detection).** One frozen
(task, seed set, nominal spec) cell, established once at large n with a tight CI, and
re-run after **any** environment change: library upgrade, driver change, power profile,
machine. If it moves by more than the floor from Test A, the environment moved, and
every cross-version comparison is void until reconciled.

This is statistical process control: the reference cell is a control chart and the
floor sets the limits. It is the only instrument that sees S3, because it compares
against an *external* fixed point rather than against a repeat of itself.

The fingerprint work already tells you *that* config changed. The reference cell tells
you whether the change **mattered**. Those are different questions and the project
currently answers only the first.

**Test C — the published-number gate (S3 calibration).** `MODELS_AND_COMPUTE.md` §4.
This is the only test that can catch a configuration that has been wrong *from the
beginning* — where there is no "before" to compare against. Tests A and B are both
self-referential; if you start wrong, they certify you as stably wrong. That is the
precise reason the reproduction gate is, as `PLAN.md` §1 says, the project's only
defence against "your environment was broken, not the model."

### 2.3 How to tell "our change" from "the model's nondeterminism"

The direct answer: **you cannot, from a pair of numbers.** You can only do it by
design, and the design is four rules.

**Rule 1 — measure the null before the treatment.** Test A at the same n you will use
later. Without a floor, no delta is interpretable, and a floor measured at a different
n does not transfer.

**Rule 2 — change exactly one thing.** With the fingerprint proving it. If two things
changed, there is no attribution — not weaker attribution, none. This is what the
counterfactual probe already does for perturbation knobs; it should be stated as the
rule for *code and config* changes too, where nothing currently enforces it.

**Rule 3 — the delta must clear the floor, and the floor is not zero.** Current gate:
`max(20 pp, 3 × floor_pp)`. The 20 pp constant is approximately 1.3 standard errors
at n=20 (architecture review #8), so per-knob it is roughly a 10% false-positive rate
before the max-over-knobs inflation. Either raise n or replace the constant with a
test.

**Rule 4 — block on seed, then test paired.** Run both arms on the identical seed set,
keep per-seed outcomes, and test discordant pairs (McNemar). Concordant pairs carry no
information about which arm is better; discarding the pairing to compare two scalar
rates throws away most of the power you paid for. This is standard practice in robot
policy comparison and it is what the probe currently forfeits at the last step.

**The important caveat on Rule 4:** pairing is only valid while S2 is small relative
to the effect. With a stochastic action head, the "same seed" pair is not a repeated
measurement of one thing — it is two draws. Measure the execution floor (Test A,
variant 1) **first**; if it is large, pairing buys less than it appears to and the
honest design is repeated measures per seed.

### 2.4 Sample sizes, computed

From the harness's own `wilson_ci`. Half-width of the 95% interval, in percentage
points:

| n | p=0.50 | p=0.70 | p=0.90 | p=0.95 |
|---|---|---|---|---|
| 10 | 26.3 | 24.8 | 19.3 | 13.9 |
| **20** | **20.1** | 18.7 | 13.7 | 11.4 |
| **50** | **13.4** | 12.3 | 8.5 | 6.2 |
| 100 | 9.6 | 8.8 | 6.0 | 4.5 |
| 150 | 7.9 | 7.3 | 4.8 | 3.7 |
| 500 | 4.4 | 4.0 | 2.6 | 1.9 |
| 2000 | 2.2 | 2.0 | 1.3 | 1.0 |

Smallest n for a half-width ≤ 5 pp: **n = 141** at p = 0.9, **n = 320** at p = 0.7,
**n = 381** at p = 0.5.

**This has an immediate consequence for `experiments/repro/TARGET.md`.** The protocol
there is 10 tasks × 10 episodes = **n = 100 per suite**, against a gate of ±5 pp:

| Suite | Published | CI at n=100 | Half-width |
|---|---|---|---|
| Spatial | ~90 | [82.6, 94.5] | **±6.0 pp** |
| Object | ~96 | [90.2, 98.4] | ±4.1 pp |
| Long | ~71 | [61.5, 79.0] | **±8.8 pp** |

**Two of four suites cannot decide a ±5 pp gate at this budget** — the measurement is
less precise than the tolerance. The published protocol is 50 episodes/task (n = 500,
±2.6 pp at p = 0.9), which is what the tolerance was written for.

This does not make the planned run useless — as a *screen* it is exactly right, and
the gaps at issue are 14 pp, comfortably outside even the n=100 interval. But the
distinction has to be stated: **n = 100 can detect the 14 pp failure; it cannot certify
a pass.** Confirming a pass requires the full protocol. Worth writing into `TARGET.md`
before a number gets quoted as "reproduced."

### 2.5 What to build, in order

Each is small and each blocks something expensive.

| | Build | Answers | Cost |
|---|---|---|---|
| 1 | **Wire `reproducibility_floor()`** into the oracle test and every campaign; `repeats ≥ 5`, report SD not range | S2, and closes G5 | hours |
| 2 | **A stochastic fixture policy** (seeded noise in the action) | proves the floor machinery works while the toy is still deterministic | ten lines |
| 3 | **The golden reference cell**, re-run at session start and after any config change | S3 — the term nothing currently sees | half a day |
| 4 | **Power state + `pstate` + `clocks.sm` into the runtime fingerprint** | P3's 25× swing | plumbing; already collected |
| 5 | **Keep per-seed outcomes on `Cell`; McNemar in the probe** | Rule 4; recovers the power pairing already pays for | half a day |
| 6 | **`mining_fingerprint` on `diagnosis`** | P8 | half a day |

Items 1–3 should land before the first LIBERO campaign, not after. They are the
difference between a campaign that produces evidence and one that produces numbers.
