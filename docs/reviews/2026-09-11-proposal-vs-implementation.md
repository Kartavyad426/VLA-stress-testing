# Review — the proposal vs what we actually built

**Date:** 2026-09-11 · **Scope:** `VLA Scenario Testing.md` (the proposal) checked
item by item against `vla_harness/`, `experiments/` and `PLAN.md` as they stand.
**Question asked:** where have we deviated, and is each deviation an improvement or a
loss?

Three tables. **A** — deviations that are improvements, keep them. **B** — things the
proposal specifies that we have dropped or weakened and should reclaim. **C** —
deliberate deferrals that are fine but need tracking so they do not become silent
omissions.

---

## A. Where we are ahead of the proposal

These are additions with no counterpart in the proposal. Worth stating explicitly,
because a reviewer holding the proposal will otherwise read them as scope creep.

| | Addition | Why it matters |
|---|---|---|
| A1 | **`_gt_` privileged-state separation, structurally enforced** via `policy_view()` | The proposal never distinguishes what a policy may see from what a detector may see. Without it a policy reading simulator ground truth produces a success rate that collapses on hardware. We enforce it in code, after the convention was violated once. |
| A2 | **The oracle test** — plant a known fault, require the miner to recover it | The proposal has no validation strategy for the mining layer at all. Validating a failure miner normally has no ground truth; planting the fault creates one. This is the single biggest methodological addition. |
| A3 | **Determinism treated as measurable, not assumed** | The proposal specifies counterfactual probes ("re-run the same seed reverting one dimension") as though rollouts were reproducible. They are not — stochastic action heads and contact chaos. We compare distributions over N seeds and measure a floor. The proposal's probe design would have failed silently. |
| A4 | **Fingerprinted, verified resume** | Not in the proposal. Prevents a cached rollout from a different configuration being served as current — a bug that reports a number, not an exception. |
| A5 | **§7b fixability / operating envelope / failure cost** | The proposal's remediation is always "collect more demonstrations". Four fixability classes, an envelope spec and cost-weighted severity are genuine additions and are the strongest commercial material we have. |
| A6 | **§7c six discriminators** (data gap vs policy gap) | The proposal asserts behaviour gap ⇒ data gap. We test it. |
| A7 | **Evidence strength downgrade** — a `trigger` without a counterfactual is marked `correlational` | The proposal's example manifest states triggers flatly. |
| A8 | **Boundary as a bracket, Wilson intervals, environment control (I8)** | The proposal asks for CIs (good) but not for a solvability control, and would have permitted interpolated point boundaries. |
| A9 | **Provenance beyond the proposal's list** | §7.3 step 1 asks for repo SHA, checkpoint revision, GPU, library versions. `run_repro.sh` records all of those **plus GPU power limit, pstate and clock** — after measuring a 25× wall-clock swing from power capping that `utilization.gpu` did not reveal. |
| A10 | **Reproduction harness quarantined from our own code** | `run_repro.sh` deliberately uses `lerobot-eval` as shipped, so "the checkpoint doesn't reproduce" stays separable from "our adapter is buggy". Not in the proposal, and the kind of thing usually learned the hard way. |

---

## B. Where the proposal is ahead of us — reclaim these

Ranked by how expensive they are to add late.

### B1. The regression set — a named build item that exists nowhere — **high**

The proposal lists it as **build item 7 of 7**:

> **7. Regression Set** — Promote representative failures into a reusable evaluation
> set for future model versions.

and again in §7.3 step 6: "freeze representative failures as a reusable regression set."

There is no regression-set code, no schema, no `run_id` convention for a frozen set,
and no week in `PLAN.md` §10 that builds one.

Worse, it is *depended upon*. Every manifest row emitted today contains:

> "…re-run the frozen regression set; report the data-response curve."

`grep` finds the phrase "regression set" exactly once in the codebase — inside that
template string. So every row promises validation against an artifact that does not
exist, and `PLAN.md` §9 makes `validation_plan` a mandatory field.

It also breaks Phase 5. "Re-run the frozen regression set and report the delta"
requires the set to have been frozen *before* fine-tuning. Freeze it late and it is
contaminated by the very data you are testing.

**Reclaim:** a regression set is a frozen list of `(task_id, seed, PerturbationSpec)`
plus the baseline outcome for each — perhaps 40 lines given `TraceStore`. Freeze it at
the end of Phase 3, before any remediation data is generated. This is cheap now and
unrecoverable later.

### B2. Confidence intervals are required in every manifest row and are not emitted — **high**

The proposal, §5:

> Every row must carry the evidence behind it — rollouts run, **observed success rate
> with confidence interval**, and the perturbation level at which the boundary was
> crossed.

`manifest.make_row()`'s `evidence` block carries `n_episodes`, `nominal_success`,
`failures_in_cluster` and example ids. **No interval anywhere in the row**, and none
on the boundary's `lower_rate` / `upper_rate` either.

The machinery exists — `Cell.ci` computes Wilson intervals and the oracle test prints
them. They are dropped at the last step, in the one artifact a client reads. This is
the same shape as the probe discarding its seed pairing: computed, then thrown away
before the output.

**Reclaim:** thread `Cell.ci` into `evidence` and onto both boundary rates. An hour,
and it satisfies an explicit proposal requirement.

### B3. Phase detectors are missing `place` and `release`, and ignore BDDL — **medium**

The proposal, build item 4, is specific about the phase vocabulary:

> approach (end-effector-to-target distance), pre-grasp alignment, grasp
> (gripper closure/contact/object lift), transport, **placement or target relation
> achieved**, **release**, and final LIBERO goal satisfaction. **Use BDDL predicates
> where they represent genuine semantic milestones, especially final goal relations**,
> but do not assume every task exposes reach/grasp/transport/place predicates.

Ours: `approach · pre_grasp · grasp · transport · retry · idle`. No `place`, no
`release`, no BDDL.

Defensible today — the toy task ends at grasp, so there is nothing to place. But the
consequence is that **the entire second half of a LIBERO task is currently invisible
to the miner**, and LIBERO-Long is exactly where success rates are lowest (~71%
published, 56% in the failed reproduction). A failure during placement would classify
as `transport` or fall through to `ambiguous`.

The BDDL point is the sharper one: the proposal identifies goal predicates as the
*semantically trustworthy* milestone signal, and warns against assuming every task
exposes the full set. That nuance is not reflected anywhere in our detector design,
and `PLAN.md` §11 question 2 asks it as an open question without an owner.

**Reclaim:** add `place`/`release` to the phase vocabulary now (they are labels, and
`PhaseSegmenter` thresholds are already constructor args), and make BDDL predicate
extraction part of the LIBERO adapter checklist in `ARCHITECTURE.md` §8 rather than
an open question.

### B4. Clustering is one of four specified signals — **medium**

The proposal, build item 5:

> Group failures by **behaviour, trigger, visual/semantic similarity and trajectory
> pattern**, then attribute cause with counterfactual probes.

`cluster()` groups by `(family, active_knobs)` — behaviour and trigger. Visual/semantic
similarity and trajectory pattern are absent.

`cluster.py` documents this as deliberate ("deferred until there is evidence it
separates anything this does not"), and I think that judgement is right for embedding
models. But **trajectory pattern is not an embedding model** — it is cheap, we already
store full `ee_xy` series, and it is the signal most likely to separate two failures
that share a family and a knob but differ in mechanism. Given M3 (the family label
currently tracks perturbation magnitude rather than mechanism), a trajectory-shape
signal is the natural corrective.

**Reclaim:** trajectory pattern, cheaply — divergence direction and consistency over
steps. Keep deferring the embedding work.

### B5. The baseline protocol is 50 episodes/task; we are running 10 — **medium**

The proposal, §7.3 step 1: "using **50 episodes per task** and the canonical evaluation
configuration. Acceptance gate: per-suite success should reproduce the canonical
published result within +/-5 percentage points."

`experiments/repro/TARGET.md`: "10 episodes/task × 10 tasks × 4 suites = 400 episodes."

At n=100 per suite the 95% interval half-width is **±6.0 pp at p=0.9 and ±8.8 pp at
p=0.71** — wider than the ±5 pp tolerance the gate is written to. The proposal's 50
episodes/task gives n=500 and ±2.6 pp, which is what the tolerance assumes.

**This does not invalidate the planned run.** The gaps at issue are ~14 pp, far outside
even the n=100 interval, so the screen will detect a failure decisively. But the
asymmetry needs writing down: **n=100 can detect the failure; it cannot certify a
pass.** Confirming a pass needs the full protocol.

**Reclaim:** state that in `TARGET.md` before any number is quoted as "reproduced", and
budget the full 50/task for whichever configuration ends up being the reported baseline.

### B6. Occlusion has no knob anywhere — **low**

The proposal's perturbation table has five dimensions; ours (via `LIBERO_PLUS_FACTORS`)
has seven. Mostly a superset — but the proposal's **Visibility: partial occlusion,
illumination** maps only half. We have `lighting`; there is no occlusion knob, and
LIBERO-plus's seven factors do not provide one either.

Minor, but occlusion is the perturbation most often asked about by people who deploy
robots, and its absence should be a stated scope decision rather than an accident of
inheriting LIBERO-plus's taxonomy.

### B7. Human adjudication moved off Ango Hub — **low, but strategically loaded**

The proposal, §7.2, assigns human adjudication to **Ango Hub** — review ambiguous root
causes, confirm taxonomy labels, double-label a subset, report inter-annotator
agreement, preserve gold cases. `PLAN.md` §7 replaces this with a Claude Code skill
doing the same workflow.

Operationally ours is probably faster to build. But Ango Hub is an iMerit product, and
§8 of the proposal makes "iMerit can contribute multimodal trace review, failure
taxonomies, human adjudication" a core part of the strategic pitch. Dropping it removes
the most concrete iMerit hook in the document.

**Reclaim or decide:** either route the κ exercise through Ango Hub, or state in the
readout why not. This is a positioning call, not a technical one — flagging it because
it will be noticed by exactly the audience §8 is written for.

---

## C. Deferred with justification — track, do not lose

| | Proposal item | Status | Comment |
|---|---|---|---|
| C1 | OpenVLA-OFT as first policy, native evaluator | replaced by SmolVLA + LeRobot | Documented in `PLAN.md` §1 and correct given 8 GB. Note it also drops the proposal's "do not force OpenVLA-OFT through LeRobot" warning as moot. |
| C2 | Second policy (GR00T N1.7) | deferred | Proposal already marks it "future direction". Free-tier T4 is the right home (VRAM-bound), but see C7 in the compute review: T4 is `sm_75`, no bf16, no FlashAttention-2. |
| C3 | Parquet + MP4 rollout format | JSONL only | `ARCHITECTURE.md` §7 correctly calls Parquet a projection. **MP4 is different** — it is an input to human adjudication (B7) and to any embedding work, and no frames are captured at all today. `image_refs` has never held a value. |
| C4 | MLflow tracking / model registry | absent | Proposal §7.3 step 6. Provenance is currently a text file per run, which is honest and greppable; MLflow is the scale-out answer. Fine to defer, worth naming as deferred. |
| C5 | Cosmos Embed1 for failure clustering | deferred | Proposal itself marks it optional and conditional ("only if it materially improves clustering"). Correctly deferred. |
| C6 | Separate pinned environments for vanilla LIBERO and LIBERO-plus | one venv exists | `MODELS_AND_COMPUTE.md` §3 confirms LIBERO-plus *uninstalls* vanilla LIBERO, so this is required, not optional. Only `.venvs/lerobot` exists today. Build the second before Phase 2 stress work. |
| C7 | Object storage with immutable experiment prefixes | local `runs/` | Fine at this scale. Interacts with the trace-size problem: `rollouts.jsonl` is gitignored for good reason and needs a home before LIBERO volumes arrive. |
| C8 | Four difficulty levels (§7.3 step 2) | not yet used | And when used, see M17 — LIBERO-plus L1–L5 are stratified by *other models' accuracy*, not physical magnitude, so they cannot carry a boundary claim. |

---

## Summary

**Net position: we are ahead on method, behind on artifacts.**

Everything in table A is about making claims trustworthy — privileged-state
enforcement, a validated miner, measured determinism, verified caching, evidence
grading. That is the right place to have spent the effort, and it is material the
proposal simply does not contain.

Table B is mostly *outputs*: the regression set, confidence intervals in rows, the
second half of the phase vocabulary, trajectory-based clustering. These are the parts
a client sees, and three of the six are explicit proposal requirements rather than
nice-to-haves.

**Two are urgent for ordering reasons rather than size.** The regression set (B1) must
be frozen before remediation data exists or Phase 5's comparison is contaminated. And
`place`/`release` phases (B3) must exist before the first LIBERO-Long campaign, or the
half of the task where most failures occur is unclassifiable.

The rest can follow the plan.
