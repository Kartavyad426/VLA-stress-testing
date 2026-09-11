# Design review — PLAN/ARCHITECTURE after the landscape pass

**Date:** 2026-09-11 · **Reviewer:** session `critical` · **Requested by:** `primary`
**Scope:** the DESIGN, before ARCHITECTURE §0's nine changes are implemented.
Judged against `VLA Scenario Testing.md` (the proposal) and `docs/LANDSCAPE.md`.
Harness findings #1–#19 are a separate document and are not revisited here.

**Verdict in one line:** the positioning work is right and the adoption ledger is
honest, but **three design decisions do not do what the document says they do**
(DG-1, DG-2, DG-5), and **three proposal deliverables have no owner in the
architecture** (DG-6, DG-7, DG-8). DG-1 and DG-2 would change a decision.

| # | Finding | Bears on | Severity |
|---|---|---|---|
| DG-1 | The CI-overlap gate's third branch fires for exactly one outcome in 41; row 1 is the permissive one | §5.3 / D2 | **high** |
| DG-2 | `sampling_arm` on the `Rollout` cannot work — the arm is a property of the REQUEST | §5.1 / §0 item 1 | **high** |
| DG-3 | Adaptive-discovered clusters get severity ≈ 0 — the manifest under-ranks what the adaptive arm exists to find | §5.1 | **high** |
| DG-4 | "Answer vs timing" does not partition; it collapses on torch/numpy | §0 item 2 / I10 | medium |
| DG-5 | Deferring the anchor contradicts the argument used to justify deferring it | D1 / §5.2 | **high** |
| DG-6 | Discriminator 4 is unrunnable — it violates the `_gt_` contract the architecture enforces | §7c | medium |
| DG-7 | The regression set is a proposal deliverable with no owner, and it silently expires | §6.1, proposal §4.7 | **high** |
| DG-8 | "PPI-consumable form" is the one retrofit-expensive item left unspecified | §9.2 / §0 item 9 | medium |
| DG-9 | Three-arm remediation is 9 fine-tunes as written; arms A and B are not the contrast that matters | §6.1 | medium |
| DG-10 | `failure_cost` is mandatory in §9 and unimplemented in §0 — no row today is a valid row | §9 | low |
| DG-11 | `language_grounding` can never fire: knob exists, no env implements it, no detector detects it | §6 / taxonomy | low |

---

## DG-1 — The gate's third branch is unreachable; row 1 is where the permissiveness is

`primary` asked whether row 3 is reachable often enough to matter. Computed, at
n=100/suite, published 87%, failed-repro 73%:

```
row1 "consistent with published"   k in 82..93   12 of 41 outcomes
row2 "genuine non-reproduction"    k in 60..100  28 of 41 outcomes
row3 "measurement too weak"        k = 81        1  of 41 outcomes
```

**Row 3 fires for exactly one observed value.** The CI half-width at p̂≈0.81 is
~7.6 pp, barely more than half the 14 pp gap between the two hypotheses, so the
window where the interval spans both is one count wide. The worry was
well-placed but pointed at the wrong row.

**The permissiveness is in row 1.** P(gate says "consistent") by true rate:

| true rate | P(passes as "consistent") |
|---|---|
| 0.87 (correct env) | 94.9% ← correct, 5% false alarm by construction |
| 0.82 (5 pp broken) | **66.0%** |
| 0.80 (7 pp broken) | **46.0%** |
| 0.77 | 20.5% |
| 0.73 (the known failed-repro value) | 4.2% |

A 5 pp break — exactly the deviation the original ±5 pp tolerance existed to
catch — passes this gate two times in three. The gate did not become more
honest than the tolerance; it became **less sensitive, while reading as more
rigorous.** "Consistent with published" is a true statement about a test with
almost no power at that effect size, and it is the sentence a reader will quote.

**What actually functions as a gate is the 50/task escalation:**

```
n=500:  true 0.87 -> 94.7% pass    true 0.82 -> 11.0% pass    true 0.80 -> 1.0% pass
```

That discriminates. n=100 does not.

**Recommendation — invert the framing rather than change the test.** 10/task is a
*screen*, and its published operating characteristic is the table above. 50/task
is the *gate*. §5.3 already says "our first full run is an orienting run, not a
reported result" and then offers escalation only for rows 2 and 3 — follow
through: escalation is unconditional for anything reported, and row 1 at n=100
means "screen passed, gate not yet run," never "consistent with published."

Two smaller things in the same section:

- **Multiple comparisons.** Four suites at 5% each ⇒ **19%** chance of at least
  one spurious non-reproduction. §9.2 adopts multiple-comparison correction;
  §5.3 does not apply it. Two sections of one document disagree.
- **73% is a single anecdote used as a fixed hypothesis.** Row 3's definition
  depends on it. Define row 3 as "CI half-width exceeds half the deviation we
  care about" — a property of the measurement, not of one reported number.

---

## DG-2 — `sampling_arm` cannot live on the `Rollout`

§0 item 1 makes `sampling_arm` a required field on every `Rollout`. It cannot
work, because `rollout_id` is `hash(env identity, policy identity, seed, spec)`
and **does not include the arm** — correctly, since the physics is identical.

So the store holds one rollout per (identity, seed, spec), tagged with whichever
arm requested it *first*:

- Adaptive arm runs cell X. Rollout stored, tagged `adaptive`.
- Uniform arm later requests cell X. Cache hit. It gets a rollout tagged
  `adaptive`. The frequency filter (`must filter to uniform`) **drops it**.
- The uniform arm's effective n is now silently smaller than requested, and
  which cells are missing depends on adaptive search order — i.e. the uniform
  arm's sample is no longer uniform. **The exact corruption the two-arm split
  exists to prevent, introduced by the mechanism meant to prevent it.**

Adding the arm to the hash is worse: it duplicates identical physics, doubles
the campaign cost, and breaks the pairing that §9.2 depends on.

**The arm is a property of the REQUEST, not of the episode.** Record it as a
per-arm set of requested cells — `runs/<id>/arms.jsonl` mapping
`(arm, rollout_id)` many-to-many — and compute uniform frequency over the cells
the uniform arm *asked for*, resolving each through the cache. One rollout can
then legitimately serve both arms, which is the efficient and correct behaviour.

This is worth fixing before item 1 ships, since it is a schema decision.

---

## DG-3 — Adaptive-discovered clusters get severity ≈ 0

`severity = cluster.count / total_failures`. If frequency statistics filter to
the uniform arm, then a failure mode that only the adaptive arm found has a
uniform-arm count of 0 or 1 — so it is ranked `low` and sorts to the bottom of
the manifest.

The adaptive arm exists to find failures the grid misses (+29.7% more failures
at equal budget). Under the current severity definition, **everything it
uniquely finds is ranked as negligible.** The manifest systematically demotes
the output of its own most novel component.

This is not the bias §5.1 warns about — it is the opposite, and it is not
mentioned anywhere.

**Recommendation:** severity needs two numbers, not one. The adaptive arm
establishes *that a failure region exists and where its boundary is*; the
uniform arm estimates *how often you land in it*. A region discovered
adaptively and then **not** sampled by the uniform arm has an unknown frequency,
not a low one — it must render as `frequency: not estimated` rather than
`severity: low`. Silent demotion and honest abstention look identical in a
ranked table and are completely different claims.

See also DG-5b below: this is the same structural problem as the sim-to-real
severity question `primary` says it has no answer to, and the same fix resolves
both.

---

## DG-4 — "Anything that changes the answer" does not partition

New rule I10: *changes the ANSWER ⇒ `identity()`; changes only TIMING ⇒
`runtime`.* The MuJoCo call is right — a documented physics change must miss the
cache. The rule generalising it is not decidable.

`torch` and `numpy` change answers. Different BLAS, different reduction order,
different kernel selection ⇒ different floats ⇒ different actions ⇒ different
contacts. `ARCHITECTURE.md` §5.1 says so already ("contact chaos… a 1e-6
command difference flips whether a grasp catches an edge"). Applied honestly,
I10 puts torch and numpy in `identity()`, every pip upgrade invalidates the
whole store, and you are back at the git-SHA position you correctly rejected.

**The real distinction is epistemic, not causal: KNOWN-semantic vs
POSSIBLY-semantic.** MuJoCo is in `identity()` because a specific documented
change is known to move results. `torch` sits in `runtime` because it *might*,
and reporting a mismatch is the proportionate response to "might."

**Recommendation:** restate I10 as a promotion policy, not a taxonomy.
`identity()` holds values with a *documented, specific* semantic effect;
`runtime` holds the rest, compared and reported. Learning that a runtime field
changes answers *promotes* it — a deliberate, rare, store-invalidating event,
recorded like a schema bump. That keeps the rule decidable and preserves
cross-session resume.

**Layering note:** MuJoCo is not a property of an env's *design*, so
`ToyReachEnv.identity()` reporting a MuJoCo version it never uses is incoherent.
A third namespaced bucket — `semantic_runtime`, inside the key, explicitly
enumerated per adapter — keeps the cache correct without pretending a simulator
build is part of the task definition.

---

## DG-5 — Deferring the anchor contradicts its own justification

§5.2's argument is good: *the gate's job is to exonerate our environment, not to
grade a policy.* Follow it one step further and it argues against the deferral.

If the gate exists to clear the environment, running it on a policy whose own
reproduction is uncertain **cannot clear anything** — a miss is ambiguous
between our setup and the checkpoint. VLA-Adapter is chosen *because* its gap
has an identified cause (batch-size mismatch vs a multi-GPU recipe), which is
another way of saying it is known not to reproduce cleanly out of the box.

So the plan is: spend weeks of setup, run an ambiguous gate, land in §5.2's
"take this option if…" state, then spend the day on the anchor anyway. The
anchor is ~1 day, inference-only. Running it *first* costs the same day and
removes the ambiguity from every subsequent number — which is exactly what §5.2
claims it does. DG-1 compounds this: at n=100 the gate is a coin flip at 5 pp,
so "lands far from published" is a likely state, not an edge case.

**Blocker that must be checked before the deferral can even be honest:** §5.4
cites lerobot#3098, filed on this exact hardware class, where the PyTorch CUDA
context and MuJoCo's EGL context contend on 8 GB. π0.5 is 6–7 GB. If that
contention makes π0.5 unrunnable here, the anchor is not a deferred option —
it does not exist, and §5.2 is promising a fallback that cannot be taken. If
osmesa makes it runnable but slow, the one-day estimate is wrong. **Verify the
anchor runs before relying on it as a contingency.** A contingency nobody has
smoke-tested is not a contingency.

### DG-5b — an answer to the sim-to-real severity question

`primary` states this as the thing it has no answer to: sim-to-real rank
correlation is contested (Spearman 0.4–0.7) and **severity ordering transfers
worst**, yet severity is what the manifest prioritises by.

The reason severity transfers badly is that it conflates two quantities with
very different transfer properties:

- **P(fail | condition)** — a mechanism claim. Measured in sim, transfers
  moderately, and is what the counterfactual probe actually establishes.
- **P(condition)** — how often that condition arises. Measured in sim, this is
  an artifact of *our own perturbation grid*, not of the world. It has no reason
  to transfer at all, because we chose it.

Current severity = frequency × task coverage, i.e. it is dominated by the half
that cannot transfer, and we chose that half ourselves.

**Recommendation: stop shipping a cardinal severity and ship a conditional
one.** The manifest reports `failure_conditional` (ours, measured, defensible)
and leaves `condition_prevalence` as a **client-supplied column** filled from
their deployment logs. Severity is computed at delivery as conditional ×
prevalence × `failure_cost`.

This is honest — we never claim to know their prevalence — and it is *stronger*
commercially, not weaker: the artifact becomes something the client's own data
completes, and the ranking becomes theirs to defend. It is also already the
design used for `failure_cost` (§7b.3), which is client-specific for the same
reason. Severity and cost have identical structure; only one of them currently
admits it.

It also resolves DG-3 for free: a region the uniform arm never sampled simply
has no prevalence estimate, which is the truthful rendering.

---

## DG-6 — Discriminator 4 is unrunnable as designed

§7c discriminator 4: *"Privileged-input ablation. Give it ground-truth object
pose instead of pixels."*

`Observation.policy_view()` strips every `_gt_` key before the policy sees it,
and `ARCHITECTURE.md` §3 states this is enforced rather than conventional —
correctly, and it is the architecture's best feature. **The architecture makes
discriminator 4 impossible to run.** Two documents in direct collision; it will
be discovered at implementation time.

**Recommendation:** an explicit, audited escape hatch rather than a quiet
exception — a `PrivilegedProbePolicy` wrapper that receives the full state,
stamps `privileged: true` on every rollout it produces, and is excluded from
every headline number by the same filter that excludes `tier3` diagnoses. The
pattern already exists in the codebase; reuse it rather than weakening
`policy_view()`.

---

## DG-7 — The regression set has no owner, and it expires silently

Proposal §4 deliverable 7: *"Promote representative failures into a reusable
evaluation set for future model versions."* §6.1 and §9 both depend on a
"frozen regression set" — it is the comparison basis for all of Phase 5, and
`validation_plan` cites a `regression set id`.

It appears in no module map row, has no schema type, and no fingerprint.

Worse, after the #2 fix it **expires invisibly**. A regression set is a list of
(seed, spec) pairs whose meaning depends entirely on env identity — and env
identity now changes whenever `grasp_radius`, `max_steps`, MuJoCo, or the
rendering backend moves. A regression set frozen in Phase 3 and re-run in
Phase 5 after any of those changes is measuring a different thing, and nothing
detects it. That is the #2 bug again, one level up, against the artifact that
carries the project's headline before/after claim.

**Recommendation:** make it a first-class L0 artifact — `RegressionSet` with its
own id, the env identity it was frozen against, and the rollout ids of its
members. Re-running it against a different env identity must be a loud,
explicit override, not a silent pass. Cheap now, and it is the comparison basis
for the one result the project is most defined by.

---

## DG-8 — "PPI-consumable form" is the one retrofit-expensive item left unspecified

§9.2 and §0 item 9 both say a PPI-ready schema "costs nothing today and is
expensive to retrofit," which is the correct reason to do it now. Neither says
what the form *is*.

What SureSim-style PPI needs is paired records: a sim outcome and a real outcome
**for the same scene configuration**. The binding constraint is therefore a
stable, externally-reproducible identifier for the scene — something a person
could set up on a physical bench. What exists today is `seed` + `spec`, which is
sim-internal and cannot be reproduced on hardware by construction.

So the single field that makes PPI possible later is precisely the one not
specified: a `scene_descriptor` — object poses, camera pose, lighting, in
physical units, resolved from the seed at reset time. That is cheap to emit now
and genuinely expensive to reconstruct from a trace store later.

Recording an unspecified intention as "adopted" is how retrofit-expensive items
get missed. Either specify the field or move E4b to Deferred.

---

## DG-9 — Three-arm remediation: executable, but not as currently scoped

Asked directly: is it 8 GB-feasible or scope creep?

**Feasible per arm.** Arm C (FTM, ~4K params) is trivially feasible and is the
cheapest defensible result in the plan. Arms A and B need a LoRA fine-tune of a
1B policy — fits.

**Not feasible as combined with discriminator 6.** §7c-6 requires a
data-response curve — three escalating budgets — and §6.1 requires three arms.
As written that is **9 fine-tunes plus 9 closed-loop re-evaluations**, each
re-evaluation on a regression set, at 1–2 h minimum and 5–8 h if escalated per
DG-1. That is the whole Phase-5 budget several times over.

**Two problems with the comparison itself, independent of cost:**

1. **Arms A and B are not the contrast that matters.** Re-rendered augmentation
   *is* targeted data, generated differently. A-vs-B answers "collected vs
   synthesised" — a fine question, but it is not the `fixability` question. The
   contrast that converts `fixability` from judgement into measurement is
   **(A or B) vs C**: data versus not-data.
2. **C has no budget axis**, so "cost per point recovered" compares a curve
   against a point. Legitimate, but it must be stated that way or the headline
   number is not comparable across arms.

**Recommendation:** three arms at **one** budget for the fixability comparison,
plus the data-response curve on arm A only, for the top manifest row. Five
fine-tunes, both results preserved, and the novel claim is unaffected.

---

## DG-10 — `failure_cost` is mandatory and unimplemented

§9 marks `failure_cost` ✱ mandatory, and states "a row missing one is a
hypothesis, not a manifest row." §0 item 6 records it as not yet built. By the
schema's own rule, no row the harness can currently emit is a manifest row.
Same for `sampling_arm`, `non_data_fix`, `fixability`, `discriminators`.

Not a design error — a status error that will read as one. Mark the mandatory
fields with their implementation status, or the schema section is describing a
document nobody can produce yet.

---

## DG-11 — `language_grounding` can never fire

`instruction_variant` is in `LIBERO_PLUS_FACTORS`; no env implements it; no
detector in `classify()` can produce `language_grounding`. The landscape's
finding that LIBERO models largely ignore language was adopted "as a reporting
rule" — but the rule as stated ("if the family never fires, say so") cannot
distinguish *the policy is insensitive to language* from *we never tested
language and have no detector for it.* Those are very different sentences to put
in front of a client.

Either implement the axis and the detector, or state in the taxonomy that the
family is out of scope for this benchmark — and note that κ and the E5
discovered-taxonomy comparison are then computed over **6** families, not 7.

---

## What I am not flagging

Stated so silence is not read as agreement-by-omission:

- **§0b positioning.** Leading with "integration and delivery, not research
  novelty" is correct and unusually honest for a document that also has to sell.
- **The MuJoCo pin (F2).** Verified, urgent, right, and caught before it put a
  physics artifact into a manifest row.
- **Discriminator 7 promoted to mandatory and first.** The single highest-value
  change in the adoption ledger.
- **Refusing an LLM in the measurement path.** Correct, and the reason given
  (reproducibility of the audit trail, not cost) is the right reason.
- **The L0–L4 contract.** Unchanged by the survey, and rightly so.
