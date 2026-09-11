# Review — methodology and claims, for the go/no-go

**Date:** 2026-09-11 · **Scope:** the thesis, the acceptance gate, the attribution
mechanism, and the inference chain from simulator failure to data requirement.
Code defects are out of scope — `2026-09-11-architecture-flow.md` covers those and
its findings are taken as established.
**Method:** read + executed. *[verified]* = reproduced by running the harness from
a clean trace store; *[by inspection]* = read, not executed; *[unverified]* = an
external fact that must be checked before it can be relied on.

**Status key:** `open` · `discussing` · `accepted` · `wontfix` · `fixed`

---

## Verdict

**Revised 2026-09-11** after reading arXiv:2510.13626 (LIBERO-Plus) in full and
checking the VLA evaluation-reproducibility literature. The original verdict turned
on M7, which is now resolved favourably. What replaces it is a novelty problem and a
measurement problem, both correctable, neither cheap to discover late.

**Conditional go.** The thesis holds, the harness design is sound, and §7b–7c is the
strongest material in the workspace. Three things should be settled before the
commitment rather than inside it:

1. **M16 — the closed loop is already published.** LIBERO-Plus §6.2 collected 20k
   remediation trajectories, mixed-fine-tuned OpenVLA-OFT and took camera-viewpoint
   robustness from 55.6% to **92.8%**. "Targeted data closes the gap" has a published
   answer of yes on the headline axis. Phase 5 as written reproduces it with a smaller
   model. The defensible claim is that *targeted beats broad at equal budget* — which
   needs a control arm the plan does not have. This is a scoping decision, and it is
   the one that most changes what the project is worth.

2. **M15 — the floor measures the wrong variance component.** Seed noise on LIBERO
   is ~1 pp; configuration bias is ~20 pp and is *perfectly reproducible*, so
   re-running a cell cannot see it. `reproducibility_floor()` will certify the cheap
   risk and stay silent on the expensive one.

3. **M1–M4 — the Phase-1 gate does not currently license proceeding.** It was tuned
   on its own answer key, its pass threshold equals the maximum achievable score, and
   it measures recall while ignoring precision — emitting, on a one-fault policy, a
   two-row manifest whose second row is wrong.

**M7 is no longer a blocker but still sets an integration decision.** Factor-level
counterfactuals are expressible; the shipped task-list API is too coarse for them.
Two independent findings (M7, M17) converge on the same instruction: **drive the
generation specifications, not `task_classification.json`.** Decide that in week 1,
because `boundary` and the operating-envelope deliverable both depend on it.

Everything else is correctable inside the plan.

---

## Summary table

| # | Finding | Class | Severity | Status |
|---|---|---|---|---|
| M1 | The gate is tuned on its own answer key | validation | high | open |
| M2 | The gate measures recall, ignores precision — and emits a false manifest row | validation | **high** | open |
| M3 | Family label tracks perturbation magnitude, not mechanism — wrong exactly at the boundary | taxonomy | **high** | open |
| M4 | Pass threshold equals the maximum achievable score | validation | high | open |
| M5 | Attribution demonstrated only at 95 pp effects; the realistic band is untested | statistics | high | open |
| M6 | `find_boundary` exercised only on step functions; already fails on the graded one | statistics | high | open |
| M7 | Tier-2 counterfactual on LIBERO-plus: mechanism exists, but coarser and unpaired | inference | high | **revised 2026-09-11** |
| M8 | The environment control is vacuous for the headline perturbation axis | inference | medium | open |
| M9 | Phase 0's prediction may have no measurable quantity; supply-side is not independent | inference | medium | open |
| M10 | "The definitive test" cannot identify a shape from three noisy points | statistics | medium | open |
| M11 | The highest-risk gate is scheduled at week 1 and week 3 simultaneously | planning | medium | open |
| M12 | The differentiator is scheduled last | planning | medium | open |
| M13 | The κ gate is underpowered against its own threshold | statistics | low | open |
| M14 | `failure_cost` is unfalsifiable on this testbed | scoping | low | open |
| M15 | `reproducibility_floor()` measures the small variance component and misses the large one | **statistics** | **high** | open |
| M16 | LIBERO-Plus already published the closed loop; Phase 5 has no control arm | **novelty** | **high** | open |
| M17 | L1–L5 difficulty is calibrated on other models' accuracy, not on physical magnitude | inference | high | open |
| M18 | The benchmark authors used n=2000 per cell for interaction claims; the plan budgets n=20 | statistics | medium | open |

**Not at fault** — stated so silence is not read as approval:

- The `_gt_` / `policy_view()` enforcement. Turning a convention into a structural
  impossibility after it had already been violated once is the right instinct and
  is rare in evaluation code.
- The sweep-versus-probe separation. These are routinely conflated; the runner
  docstring gets it exactly right.
- **§7b.1 fixability, §7b.2 operating envelope, §7b.3 failure cost.** This is the
  strongest thinking in the workspace and the real differentiator — more so than
  the mining layer. A vendor with a collection business writing down the four ways
  the answer is *not* "collect more data" is what makes the manifest credible.
- Going small to close the loop, for the reason given. Correct trade, correctly argued.
- Having kill criteria at all.

---

## M1. The gate is tuned on its own answer key — *[by inspection]*

`PLAN.md` §4 and `classify.py`'s module docstring both state it plainly: two
decision rules were added *because the oracle test caught the miner getting them
wrong*. `_wrong_object` and `_attempt_spread` were authored with the ground truth
in hand.

That is fitting the detector to the fixture. Four fixtures, roughly five rules, no
held-out fault, no pre-registration of the rule set. The gate then reports that the
miner recovers the faults the rules were written to recover.

This matters more here than in ordinary software because the gate is doing
load-bearing epistemic work: `PLAN.md` §2 names it as the week-1 test that decides
whether the mining design is sound, and §4 calls it "the core methodological
trick." A gate that is revised until it passes cannot discharge that duty.

**Direction.** Plant a fifth fault *after* freezing the rule set and never look at
its diagnosis until the rules are committed. A single genuinely held-out fixture
converts the gate from a demonstration into a test. Cheap — it is one more entry in
`GROUND_TRUTH` and one more `bugs=` branch.

---

## M2. The gate measures recall and ignores precision — and empirically emits a false manifest row — *[verified]*

The acceptance criterion is "recovers ≥ 3 of 4 planted triggers." Nothing anywhere
asks the converse question: *does a policy with one fault produce one manifest row?*

The `CONTROL` block is not that check. It verifies that a **clean** policy under
perturbation yields no rows — necessary, and much weaker than it looks, because a
clean policy produces no failures to misclassify in the first place.

Run from a clean store, the `camera_misalignment` scenario — **one** planted fault —
produces three clusters and a two-row manifest:

```
visual_grounding   n=100   camera_yaw_deg    err=19.6cm
manipulation       n=20    camera_yaw_deg    err= 7.3cm
manipulation       n=19    (4 knobs)         err= 4.7cm
```

```
## DGM-002 — manipulation [MEDIUM]
- Trigger: camera_yaw_deg — co-occurs with failure; NOT counterfactual-confirmed
- Coverage required: Trajectory variants around contact and alignment across the
  failing camera_yaw_deg band; include successful corrections.
```

`DGM-002` is a false positive. The policy has no manipulation fault. The row is
labelled MEDIUM severity, carries evidence, and instructs the reader to buy
contact-and-alignment demonstrations for a camera-calibration problem.

**Why this outranks a missed detection.** A manifest that omits a gap costs the
client a failure they already had. A manifest that invents one costs them a data
purchase, and costs us the credibility that is the entire product. The proposal's
own framing — "every row is a falsifiable claim with its evidence attached" — makes
precision the metric that matters, and it is the one not measured.

**Direction.** Add a precision criterion to the gate: for a single-fault policy,
rows whose family is not the planted family must be zero, or the extra clusters must
fall below a stated reporting floor. Report precision and recall side by side.

---

## M3. The family label tracks perturbation magnitude, not mechanism — and it is wrong exactly at the boundary — *[verified]*

This is the mechanism behind M2, and it is worse than a stray cluster.

One policy, one planted fault (`camera_misalignment`), swept across yaw, 20 seeds
per level, family assigned by `classify()`:

```
yaw=  3   {None: 20}                  (all succeed)
yaw=  6   {'manipulation': 20}
yaw=  9   {'visual_grounding': 20}
yaw= 12   {'visual_grounding': 20}
yaw= 15   {'visual_grounding': 20}
yaw= 20   {'visual_grounding': 20}
yaw= 25   {'visual_grounding': 20}
```

Same fault, same mechanism, unanimous labels — and the label flips between yaw 6
and yaw 9. The cause is structural: `classify` routes on `final_error_m` against a
fixed `LOST_TARGET_M = 0.10`. A misalignment fault produces an error proportional to
the misalignment, so the *intensity* of the perturbation, not the nature of the
failure, decides the family.

**The damaging part is where it is wrong.** `find_boundary` brackets this fault
between 3° and 6°. The manifest is built around the boundary — that is the whole
point of the artifact, and `coverage_required` instructs the client to collect
across the measured band. So the taxonomy is least reliable at precisely the
operating point every downstream claim is anchored to, and the wrong label there
carries a different remediation from `REMEDIATION` and a different `fixability`
class.

Note also that at yaw=6 the policy is at 0% success. This is not a marginal or
ambiguous cell. It is a total failure, confidently and uniformly mislabelled.

**Direction.** A magnitude threshold cannot separate mechanisms. Grounding failures
have a signature independent of scale — the error direction is *consistent* across
steps and consistent with a frame rotation, whereas a control failure is
unbiased around the target. Test direction and bias, not distance. Until then, any
family assigned on `final_error_m` alone should be flagged low-confidence and
routed to Tier 3 rather than reported.

---

## M4. The pass threshold equals the maximum achievable score — *[verified]*

The gate is "≥ 3 of 4." From a clean store, fixture 4 (`unfiltered_grasp +
no_recovery`) produces **zero failures at every swept level**:

```
nominal              100.0%
pixel_noise_std=0.01 100.0%
pixel_noise_std=0.02 100.0%
pixel_noise_std=0.03 100.0%
pixel_noise_std=0.05 100.0%
attributed trigger -> None
```

`PLAN.md` §4 explains why honestly (noise re-rolls, so repetition is a winning
strategy in the toy) and marks recovery diagnosis unvalidated. Good. But the
consequence is not drawn: **the fixture can score at most 3, and the gate passes at
3.** A threshold set at the achievable maximum cannot fail, which makes "GATE
PASSED" uninformative about the thing it was built to test.

There is a second, quieter version of the same problem: fixtures 3 and 4 share a
trigger knob and fixture 3 already passes, so the "discriminative test" framing in
the `SCENARIOS` comment is doing less work than it claims.

**Direction.** Either remove fixture 4 and state the gate as 3 of 3 with recovery
explicitly out of scope until LIBERO, or — better — build a toy where the world is
deterministic given state, so repetition genuinely cannot succeed. The second also
removes the reason recovery is unvalidated at all.

---

## M5. Attribution is demonstrated only at 95 pp effect sizes — *[verified]*

The three passing attributions:

| fixture | top delta | second delta |
|---|---|---|
| camera_misalignment | **+95.0 pp** | +0.0 |
| nearest_object | **+95.0 pp** | +0.0 |
| unfiltered_grasp | +35.0 pp | +0.0 |

Two of three are separated from every alternative by 95 percentage points. The
finding in the prior review (#8) puts the unpaired SE at n=20 near 15.8 pp, so the
demonstrated regime is roughly 6σ.

Published LIBERO-plus degradations are large in aggregate but individual
factor-level contributions will land in the 10–30 pp band — where the mechanism is
untested, and where by the prior review's own arithmetic the 20 pp threshold is
about 1.3 SE. The oracle test therefore validates that the probe *plumbs* correctly;
it does not validate that it *discriminates*.

**Direction.** Add a fixture with a deliberately weak fault — one that costs 15–25 pp
— and require the probe to either attribute it or correctly decline. Declining is a
pass. That fixture is what would have surfaced the McNemar point from the other
direction.

---

## M6. `find_boundary` has been exercised only on step functions, and already fails on the graded one — *[verified]*

The two fixtures whose boundaries are reported are cliffs:

```
camera_yaw_deg:    3 -> 100%,  6 -> 0%
distractor_count:  0 -> 100%,  1 -> 0%
```

A one-level fall from 100% to 0% makes bracketing trivial and tells us nothing about
behaviour under noise. The one fixture that produces a graded curve is `pixel_noise_std`:

```
nominal 100%   0.01 100%   0.02 95%   0.03 100%   0.05 60%
boundary: not crossed in swept range
```

Non-monotone, and no boundary found. Real VLA degradation is graded by construction —
LIBERO-plus is stratified L1–L5 precisely to produce a curve. So the detector has
been validated on the shape it will not see and fails on the shape it will.

This upgrades prior-review #9 from "honest about interpolation, silent about noise"
to "empirically failing on the only realistic fixture available." It should be
treated as a Phase-1 blocker rather than a statistics refinement.

---

## M7. Tier-2 on LIBERO-plus: the mechanism exists, but coarser and unpaired — *[revised, verified against the paper]*

**Revised 2026-09-11** after reading arXiv:2510.13626 in full. The original finding
asked whether LIBERO-plus permits factor-level counterfactuals at all. It does. The
concern was right to raise and wrong in its severity; what replaces it is narrower
and still consequential.

**What the paper establishes.** §5.1 defines per-perturbation indicator variables
`D_i ∈ {0,1}` and estimates `s(D_i=a, D_j=b)` for all four combinations, including
double-perturbation instances. The corpus is therefore factorially structured: for a
doubly-perturbed group there exist singly-perturbed and unperturbed counterparts.
Appendix A confirms the underlying perturbations are genuinely parametric — Table 5
gives key parameters per component and their L1–L5 settings (Gaussian blur σ = 1 → 10,
motion blur radius r = 5 → 35, and so on), across 21 low-level components under the
seven factors.

So `spec.revert()` has a referent, and `G4` turns out to be better founded than I
credited. Three caveats survive, and they change what the manifest can claim.

**Caveat 1 — the shipped granularity is factor-presence, not factor-magnitude.**
The public organisation is task ID → category → level (`task_classification.json`).
At that API the counterfactual available is "camera viewpoint perturbed: yes/no",
not "yaw 15° → 12°". Recovering the continuous parameter means going under the
benchmark's task-selection interface to its generation specifications (Appendix D).
That is feasible and is the right move, but it is integration work the plan has not
scoped, and it is the difference between shipping a `boundary` field and not.

**Caveat 2 — the comparison is between groups of scenes, not within one scene.**
`s(·)` is a success rate over an instance set. Moving from the (1,1) group to the
(0,1) group changes the scenes as well as the factor. That is an unpaired contrast,
which forfeits exactly the power the prior review's #8 wanted to recover with
McNemar — pairing needs the same scene under two conditions. Regenerating instances
ourselves from a fixed base episode restores it; selecting from the shipped corpus
does not.

**Caveat 3 — `boundary` is the field most at risk.** `PLAN.md` §9 marks `boundary`
mandatory, and §7b.2's operating-envelope deliverable is quoted in physical units
("certified within ±10° camera yaw"). Both require the continuous axis from caveat 1.
If the project ends up working at factor-presence granularity, the envelope spec
degrades from "±10° of yaw" to "viewpoint perturbations of the LIBERO-plus kind",
which is dramatically less useful to a buyer and should not be discovered in week 7.

**Direction.** Unchanged in urgency, narrowed in scope: in week 1, confirm that the
generation specifications can be driven directly to emit a base episode plus a
one-factor-varied sibling at a chosen parameter value. Build the adapter against
*that* interface, not against `task_classification.json`. If it cannot be driven,
decide then — before the policy choice hardens — whether to generate perturbations
on vanilla LIBERO ourselves and use LIBERO-plus only as a published reference.

---

## M8. The environment control is vacuous for the headline perturbation axis — *[by inspection]*

§7c presents discriminator #1 — can a scripted expert still solve the perturbed
task? — as "a control we already have and the proposal does not mention," and
instructs running it for every perturbation cell.

On the toy it is genuine, because the scripted policy closes the loop through the
simulated camera. On LIBERO it mostly will not be. The available expert is a
recorded demonstration, which is open-loop with respect to pixels. Camera yaw,
lighting, texture and sensor noise change *only* the observation — the physics and
the goal predicate are untouched — so a replayed demo succeeds by construction,
regardless of how severe the visual perturbation is.

So for four of the seven LIBERO-plus factors, including the headline viewpoint axis,
the control returns "solvable" every time and discriminates nothing. It retains real
force for `initial_state` and `object_layout`, where the demo genuinely can become
infeasible — which is worth keeping, and is narrower than §7c claims.

**Direction.** State the scope limit in §7c. For the observation-only factors the
control has to be a closed-loop privileged-state expert (discriminator #4's
apparatus), not a replay — which means #1 and #4 collapse into one test there, and
the cheap-tests-first ordering changes.

---

## M9. Phase 0's headline prediction may have no measurable quantity, and its corroboration is not independent — *[by inspection]*

Two problems with the supply-side step as specified.

**The quantity may be degenerate.** The promised deliverable is a timestamped
prediction of the form "yaw coverage is ±X°, therefore we predict the boundary near
Y°." LIBERO demonstrations are recorded from fixed camera placements; the agentview
extrinsics are essentially constant across the suite. The coverage histogram for the
project's primary axis is then a spike at zero, not a distribution with a width. There
is no X to measure, so no Y to predict. The same holds for lighting and texture.
The axes where a genuine density *does* exist — initial end-effector pose, object
placement, trajectory length, gripper-event timing — are the ones the proposal
deprioritises.

**The corroboration is not independent.** `supply_side` is described in §9 as "the
independent corroboration" for a manifest row. SmolVLA's LIBERO checkpoint is a
fine-tune of a base model pretrained on community robot data; LIBERO demonstrations
are a subset of what the policy saw, not its training distribution. A sparse region
in the LIBERO histogram is therefore weak evidence about what the policy was exposed
to, and the two "independent lines" in §2's assumptions table are closer to one and
a half.

**Direction.** Re-scope Phase 0 honestly: it measures *benchmark* coverage, which
bounds what fine-tuning could have taught, and say that. Move the falsifiable
prediction onto the axes that actually have a measurable spread. And check whether
SmolVLA's pretraining mix is enumerable at all — if it is not, `supply_side` should
be labelled as benchmark coverage, never as training coverage.

---

## M10. "The definitive test" cannot identify a shape from three noisy points — *[by inspection]*

§7c calls discriminator #6 definitive and says **the shape of the data-response
curve is the diagnostic** — rising steadily versus plateauing below acceptable.
`validation_plan` operationalises this as three escalating LoRA budgets re-measured
on the frozen regression set.

Three points, each a proportion measured on a few hundred episodes, each carrying a
CI of several points, and the discrimination required is between a modest positive
slope and a flattening one. Those two hypotheses are not separable at that
resolution. The honest output of the planned experiment is "it moved" or "it did
not move" — a point, which §7c itself says cannot distinguish "needs more data" from
"cannot learn this."

This is load-bearing because #6 is the only discriminator that can assign
`fixability: architecture` — the class whose presence is what stops the manifest
reading as a collection vendor recommending collection.

**Direction.** Either accept that `architecture` will be assigned on discriminators
2+3+4 (coverage, in-distribution replay, privileged-input ablation) and drop the
claim that #6 is definitive, or spend the budget on more points at fewer seeds and
say up front which alternatives the design can actually separate. Do the power
calculation before the fine-tunes, not after.

---

## M11. The highest-risk gate is scheduled at week 1 and week 3 simultaneously — *[by inspection]*

- `MODELS_AND_COMPUTE.md` §4: "**Action — do this before committing to SmolVLA:** run
  the suites, compare against published… Budget a week."
- `PLAN.md` §2 assumptions table: "Published checkpoint numbers are reproducible" —
  **Wk 1**.
- `PLAN.md` §10 schedule: week 1 is "checkpoint survey; harness skeleton; oracle
  test"; **LIBERO integration is weeks 3–4**.

The reproduction gate cannot run before the LIBERO integration exists. So the item
identified as the top Phase-2 risk, and the one whose answer should gate the choice
of primary policy, is in fact scheduled after that choice has been made and two
weeks of work have been built on it.

**Direction.** Reconcile to one date. Given lerobot#3264 is open and unanswered, the
integration spike should move to week 1 in a reduced form — one suite, one seed,
enough to see whether the numbers are in the neighbourhood — even if the full
harness swap waits.

---

## M12. The differentiator is scheduled last — *[by inspection]*

Phase 5 — fine-tune on remediation data and re-measure — sits at weeks 8–9 of 10.
`PLAN.md` §1 justifies the entire headline-model downgrade on the ability to run it:
"a complete cycle on SmolVLA is a stronger artifact than a partial cycle on
OpenVLA." The plan then places that cycle in the last slot, behind every dependency,
with no slack.

§10 claims "stopping after any week leaves something presentable," and that is true —
but the thing presentable at week 7 is a robustness report plus a data plan, which
is close to what §2 of the proposal defines the project *against*. The argument for
the trade and the schedule that realises it are pulling in opposite directions.

**Direction.** Pull a minimal closed loop forward. A deliberately tiny remediation
fine-tune on the toy's successor, or on a single LIBERO task with a small budget, run
at week 5 as a rehearsal, converts the riskiest step into a known quantity and keeps
the option to report *a* closed loop even if the good one does not land.

---

## M13. The κ gate is underpowered against its own threshold — *[by inspection]*

"Double-label 30 traces, compute κ", gate at κ ≥ 0.5, kill criterion below it. Seven
families across 30 traces is roughly four instances each, and κ at n=30 has a
confidence interval wide enough that 0.4 and 0.7 are not distinguishable. The gate
can pass or fail on sampling noise, and it is a kill criterion.

**Direction.** Either raise n, or stratify so each family has a stated minimum, or
report κ with its CI and gate on the lower bound. Also worth deciding in advance
whether κ is computed over all seven families or over the subset the harness can
actually populate — the toy validated three.

---

## M14. `failure_cost` is unfalsifiable on this testbed — *[by inspection]*

§7b.3 calls failure cost "the largest omission in the proposal," and as product
thinking that is right — buyers do care what a failure *does*. But LIBERO is
tabletop pick-and-place with no fragile parts, no line to stop and no human in the
cell. The benign/disruptive/safety classes have no measurable referent there, so the
mandatory `failure_cost` field will be populated by assertion while every neighbouring
field carries evidence.

**Direction.** Keep it — it is the right idea and it is what makes the envelope spec
saleable. Mark it in the manifest as a *designed* field rather than a measured one,
the way `supply_side` currently carries its PLACEHOLDER note, so a reader can see
which columns are evidence and which are judgement.

---

## M15. `reproducibility_floor()` measures the small variance component and misses the large one — *[verified against literature]*

`ARCHITECTURE.md` §5.1 and `runner.reproducibility_floor()` treat run-to-run
variation as the thing to measure and beat: repeat a cell at fixed configuration,
take `max − min`, require an attribution delta to clear `3 × floor`. The reasoning
is sound and the literature says it is aimed at the wrong term.

**Evaluation error on LIBERO decomposes into two components of very different size.**

*Stochastic (seed) variance — small.* Published LIBERO protocol is 3 random seeds ×
500 rollouts; OpenVLA reports 88.4% **± 0.8** on Object, 84.7% ± 0.9 on Spatial,
79.2% ± 1.0 on Goal, 53.7% ± 1.3 on Long. Robustness studies report seed std in the
0.2–0.6% range. On a policy with a deterministic decode this is roughly a
percentage point.

*Configuration (environment) bias — large.* openvla#282 reports **68%** on
LIBERO-Object against a published 88.4% — a 20 pp gap — while the same setup
reproduced Spatial, Goal and Long within ~1–3 pp. lerobot#3264 reports SmolVLA
LIBERO at 73.25% overall against published numbers. `vla-eval` (arXiv:2603.13966)
documents the mechanisms: SimplerEnv's termination flag signals *transient* success
so stopping early inflates scores, and CALVIN requires hardcoded observation
normalisation statistics over 15 robot-state and 24 scene-state dimensions that are
"absent from the official evaluation documentation." Its own careful reproduction
landed within ±3 pp across the four LIBERO suites — i.e. getting it right is
possible, and the gap when it is wrong is an order of magnitude above seed noise.

**The finding.** From openvla#282, the decisive detail: the reporter changed the
random seed from 3 to 15 and the success rate **stayed at 68%**. Redownloading the
dataset, raising `max_steps` to 300, and disabling flash-attention moved it by
about a point in total.

That is what a bias looks like. It is not variance, it is **perfectly reproducible
and perfectly wrong** — and `reproducibility_floor()`, which re-runs the same cell
under the same configuration, will report a reassuringly tiny floor in exactly the
situation where the numbers are least trustworthy. It cannot see the term that
matters, because repeating a run holds constant the thing that is broken.

The practical risk is specific and it is not hypothetical for this plan: a
`PhaseSegmenter` threshold change, a `control_mode` flip, a soft-versus-hard reset,
a LeRobot point release between week 4 and week 8. Each shifts the success rate by
more than the floor and none of them register as variance.

**Direction — three changes, all cheap.**

1. **Rename the concept and add the missing half.** Keep `reproducibility_floor()`
   as the *stochastic* floor; add a **configuration floor** measured by re-running a
   fixed reference cell after any environment change, and gate campaign comparisons
   on the larger of the two.
2. **Make the reference cell a standing fixture.** A single frozen
   (task, seed set, nominal spec) cell, re-run at the start of every session and
   logged. A drift in it is the tell that the environment moved under you, and it
   costs a couple of minutes. This is the sim analogue of the nominal-cell check
   §6 already recommends.
3. **Note the consequence for the fingerprint work.** This strengthens the prior
   review's #2: the fingerprint must cover anything that shifts the *distribution* —
   library versions, control mode, reset mode, normalisation statistics, step caps —
   not merely anything that changes a trace. Those are precisely the fields openvla#282
   and vla-eval identify, and they belong in the fingerprint by name.

**What this does not change.** The seed-averaging design is still right, and the
decision to compare distributions rather than trajectories is still right. The point
is only that the floor as implemented certifies the cheaper of the two risks.

---

## M16. LIBERO-Plus already published the closed loop — Phase 5 needs a control arm — *[verified against the paper]*

This is the finding with the most bearing on the go/no-go, because it is about what
the project can claim rather than whether it works.

§6.2 of arXiv:2510.13626 does what `PLAN.md` Phase 5 proposes to do. The authors
built a remediation dataset of **over 20,000 successful trajectories** by expanding
LIBERO along the perturbation factors, mixed-fine-tuned from official OpenVLA-OFT
weights, and re-measured on LIBERO-Plus. Table 2:

```
camera viewpoint   55.6%  ->  92.8%   (+37.2 pp, best model by 37.2 pp)
noise                     ->  89.3%
layout                    ->  77.6%
overall                   ->  79.6%
```

"Collect data in the region where the policy is weak, retrain, and the weakness
closes" is therefore **already published, on the headline axis, with a large
effect.** The proposal's §5 question 5 — "did that data close the gap when the
policy was retrained and the same conditions were re-run?" — has a published answer
of yes.

**This does not sink the project, but it relocates the contribution.** LIBERO-Plus
closed the gap by collecting broadly: 20k trajectories across all seven factors.
That is the brute-force answer. The thesis worth defending is the one `PLAN.md` §3
already gestures at — *targeted* beats *broad* at equal budget, and the mining layer
is what tells you where to aim. That claim is genuinely open and genuinely valuable.
It is also not what Phase 5 currently measures.

**As written, Phase 5 cannot support it.** Fine-tuning on the top manifest row and
reporting an improvement reproduces a known result with a smaller model. A reviewer
who knows this paper will say so.

**Direction — add a control arm, and change the success criterion.** Phase 5 should
be a comparison, not a demonstration:

| Arm | Data | Budget |
|---|---|---|
| A | targeted at the top manifest row | N |
| B | **uniformly sampled across all factors** | N |
| C | no fine-tune | — |

The result that matters is A vs B at matched N. If targeted wins, the manifest has
demonstrated value and the mining layer is justified as more than description. If it
ties, the honest finding is that broad collection is sufficient and the diagnosis
layer is a cost saving at best — which is worth knowing, is publishable, and is
exactly the kind of negative result §5 already says is reportable.

This also happens to be the cheapest version of §7c discriminator #6 that answers a
real question, and it partly rescues M10: A-vs-B at one budget is a sharper test
than a three-point curve fit through noise.

**Second-order point for the readout.** §6.2 fine-tuned *from OpenVLA-OFT*, so the
published closed loop is on the model the plan deferred. A SmolVLA result is
genuinely new as a data point — but "new checkpoint, same finding" is a weak claim
to build a ten-week project on, and should not be the headline.

---

## M17. L1–L5 difficulty is calibrated on other models' accuracy, not on physical magnitude — *[verified against the paper]*

`VLA Scenario Testing.md` §7.3 step 2 instructs stressing "across camera viewpoint,
robot initial state and object layout **at four difficulty levels**", treating L1–L5
as the intensity axis.

It is not one. §6.1 of the paper: the benchmark was constructed by applying the
perturbations, then "evaluating the resulting tasks using four representative models,
then stratifying them into five difficulty levels (Level-1–Level-5) **according to
the accuracy distribution observed across these models**."

L3 is therefore a bucket of instances that four reference VLAs found comparably hard.
It is a *model-calibrated* scale, not a physical one, and three things follow:

1. **A boundary expressed in levels is circular for a robustness claim.** "SmolVLA
   fails at L3" says it fails where OpenVLA, π0 and the others failed. That is a
   statement about agreement with the reference models, not about the operating
   envelope of the policy under test.
2. **Monotonicity is not guaranteed for a policy outside the calibration set.**
   SmolVLA was not among the four. Its success-versus-level curve may be flat or
   non-monotone, and `find_boundary` — already fragile on non-monotone input (M6) —
   would return nothing or something arbitrary.
3. **It cannot produce an operating envelope.** §7b.2's deliverable is in physical
   units. Levels do not convert into degrees.

**Direction.** Use L1–L5 for *sampling* and for comparability with published
results, never as the axis a boundary is measured on. The boundary axis must be the
underlying component parameter from Appendix A / Table 5 — which is the same
requirement M7 caveat 1 arrives at from the other direction. Two independent lines
now point at the same integration decision: **drive the generation specs, not the
task list.**

---

## M18. The benchmark authors used n=2000 per cell for interaction claims — *[verified against the paper]*

§5.2: "we perform independent tests for each type of single-dimension perturbation
and pairwise perturbations, recording the success rate over **2000 repeated trials**",
with a chi-square test for independence to establish that the compositionality gap
is real "rather than sampling noise arising from finite trials."

The plan budgets 20 episodes per screening cell and 50 near the boundary. The gap is
two orders of magnitude, and the reason is instructive rather than damning: they were
making *interaction* claims, and interactions are second-order effects that need far
more data than main effects. The prior review's #10 flags that the probe finds main
effects only. This is the price tag on fixing it.

Supporting figures from the evaluation-statistics literature: a 90% success rate
measured over 70 rollouts carries a 95% Clopper-Pearson interval of 80.5–95.9% —
15.4 pp wide — and tightening that to ±2 pp requires ~1,030 rollouts, roughly 15×
more. Sequential testing recovers some of this (reported savings of 16–25% on
hardware, and much larger in simulation), and **paired designs recover more**: with
both policies run on identical tasks, layouts and seeds, only discordant pairs carry
information. That is the same McNemar move the prior review's #8 recommends, now with
external support — and it is a further argument for generating our own paired
instances (M7 caveat 2) rather than selecting from the shipped corpus.

**Direction.** Decide explicitly which claims the project is funded to make. Main
effects and boundaries at n=50 with honest CIs: affordable. Interaction claims:
not affordable at this budget, and should be dropped from scope rather than made
weakly. Say which in the readout.

---

## Appendix — what would change the verdict

In rough order of how much each moves it:

1. **A Phase-5 design with a control arm** (M16). Targeted-vs-uniform at matched
   budget converts a reproduction into a result. Without it the project's headline
   finding is already in the literature.
2. **A configuration floor alongside the stochastic one** (M15), and a standing
   reference cell re-run every session. This is what makes any before/after number
   in weeks 4–9 mean anything.
3. **A held-out fixture passes** (M1) with precision reported (M2) and a graded,
   weak-effect fault included (M5, M6). That is a gate that could have failed.
4. **M3 addressed by a direction-based grounding signal.** Without it every
   boundary-region row is suspect, and boundary-region rows are the manifest.
5. **The adapter targets the generation specs** (M7, M17), so `boundary` and the
   envelope spec can be stated in physical units.
6. **§10 reconciled with `MODELS_AND_COMPUTE.md` §4** (M11) and a rehearsal loop
   pulled forward (M12).

---

## Sources consulted

- [LIBERO-Plus: In-depth Robustness Analysis of VLA Models (arXiv:2510.13626)](https://arxiv.org/abs/2510.13626) — §5.1 compositionality, §6.1 difficulty stratification, §6.2 remediation fine-tune, Appendix A/Table 5 perturbation parameters
- [vla-eval: A Unified Evaluation Harness for VLA Models (arXiv:2603.13966)](https://arxiv.org/html/2603.13966v1) — hidden normalisation, termination semantics, ±3 pp reproduction
- [openvla#282 — Significantly Lower Success Rate on LIBERO-Object](https://github.com/openvla/openvla/issues/282) — 68% vs 88.4% ± 0.8, unchanged across seeds
- [openvla#335 — Low success rate on LIBERO-Object](https://github.com/openvla/openvla/issues/335)
- [lerobot#3264 — SmolVLA LIBERO reproduction](https://github.com/huggingface/lerobot/issues/3264)
- [Is Your Imitation Learning Policy Better than Mine? (arXiv:2503.10966)](https://arxiv.org/pdf/2503.10966) — Clopper-Pearson intervals, trial counts
- [Beyond Binary Success: Statistically Rigorous Robot Policy Comparison (arXiv:2603.13616)](https://arxiv.org/html/2603.13616) — sequential testing savings
- [TRI, Statistical Thinking for Robot Policy Evaluation](https://medium.com/toyotaresearch/statistical-thinking-for-robot-policy-evaluation-from-rigorous-a-b-testing-to-effective-0ae886fbd68d) — paired-outcome design, discordant pairs
- [OpenVLA-OFT project page](https://openvla-oft.github.io/) — 3 seeds × 500 rollouts protocol
