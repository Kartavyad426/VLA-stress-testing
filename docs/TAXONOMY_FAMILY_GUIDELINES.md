# TAXONOMY FAMILY GUIDELINES

> # ⚠ READ THIS BEFORE ANY TAXONOMY DECISION
>
> **Written 2026-09-16.** Any change to the failure family set — adding a family, removing one,
> renaming one, changing a rule, changing how families are assigned or reported — should be made
> against this document. It exists because we have already made four taxonomy mistakes that were
> invisible until measured (§6), and because the family set is about to be revised with input from
> the people who will consume the manifest.
>
> **Status of the current taxonomy: not settled, and not reportable.** `docs/FAILURE_FAMILY_AUDIT.md`
> §5b measured that **76.6% of failures have their family decided by rule order rather than by the
> taxonomy**, and that every family but one can be driven to zero by reordering alone. No family
> count should go in front of anyone until that is resolved.
>
> **Companion documents.** `docs/FAILURE_MINING_METHODS.md` §3–§4 (what the field does, and the
> evidence behind every claim here) · `docs/FAILURE_FAMILY_AUDIT.md` (what our current taxonomy
> actually is, measured) · `docs/OUR_MINING_APPROACH.md` (the pipeline).

---

## 0. The one thing to take away

**A failure taxonomy is not a description of reality. It is an interface to a decision.**

There is no fact of the matter about what the "real" failure categories are — §5 of the methods
survey establishes that four published groups carved the same phenomenon four incompatible ways and
none is more correct. What there *is* a fact of the matter about is whether a category **changes
what someone does, and whether doing that thing helps.**

So the design question is never *"what are the failure modes?"* It is:

> **Who reads this row, what decision do they make on it, and would a different label have made them
> decide differently?**

A family that cannot answer those three is decoration, however clean its clusters and however
sensible its name.

---

## 1. The two requirements that drive this design

Stated by the project owner, 2026-09-16, and they are more restrictive than they look:

1. **Relevant to the manifest's consumers.** The taxonomy is reported to people who will act on it.
   Families must correspond to actions *those particular people* can take.
2. **Meaningful when viewed separately.** Each family must be interpretable **standalone** — a reader
   who sees one row, without the rest of the taxonomy in front of them, must understand what it
   means.

### Requirement 2 has a consequence that is easy to miss and expensive to get wrong

**A family produced by first-match-wins precedence is *not* meaningful standalone**, because its
actual definition is *"its own condition, minus every condition evaluated before it."* You cannot
state what it means without reciting the rules above it.

Our current `manipulation` family is the worked example. Its rule is `attempts >= 1`, which fires on
**528 of 534 failures (98.9%)**. What it actually means is *"had a grasp attempt, and did not end far
from the target, and did not end at a distractor, and did not repeat in place, and did reach
pre-grasp"* — five conditions, four of which are invisible and live in other rules. A reader seeing
`manipulation: 125` cannot recover that.

**This is not an argument for reordering the rules.** Reordering relabels the same 76.6% of episodes.
It is an argument for one of:

- **disjoint predicates by construction** — partition on one variable per level so no failure
  satisfies two leaves; or
- **reporting the predicate conjunction** rather than a collapsed label — `lost_target ∧ any_attempt`
  (336 episodes) is self-describing and standalone in a way `visual_grounding` is not.

Either satisfies requirement 2. First-match-wins over overlapping predicates cannot.

---

## 2. What makes a taxonomy good — seven properties

Ordered by how much trouble their absence has caused, here and in the literature.

| # | Property | Test | Our current status |
|---|---|---|---|
| 1 | **Actionable** — each family maps to a distinct intervention | Two families prescribing the same action should be merged | untested |
| 2 | **Standalone-meaningful** — interpretable without the rest of the set | Show one row to a naive reader; can they say what happened? | **fails** (§1) |
| 3 | **Mutually exclusive** — no failure belongs in two | Count multi-firing episodes | **fails: 76.6%** |
| 4 | **Exhaustive, with an honest residual** | An `other` bucket that is reachable and non-trivial | **fails: abstention unreachable** |
| 5 | **Reachable** — every declared family can actually be produced | Run the classifier; look for permanent zeros | **fails: 2 of 8 dead** |
| 6 | **Informative** — no family is near-universal or near-empty | Report the share; flag anything >60% or <2% | **fails: 65.5% / 98.9% predicate** |
| 7 | **Named for what is measured**, not for an inferred cause | Does the name assert something the rule does not observe? | **fails** (§6.1) |

### On property 7, because it is the subtlest

A rule over terminal state observes **where the arm ended up**. It does not observe *why*.
`visual_grounding` asserts a perceptual cause; the rule that produces it measures a distance. The
name is a conclusion smuggled into a detector's output.

**Capability claims should be the output of an attribution step (survey §5), not the name of a
detector.** Prefer names that describe the observation — `ended_off_target`, `wrong_object_selected`,
`repeated_in_place` — and let the causal language appear only where an intervention supports it. This
costs nothing and removes an entire class of silent error, because a wrongly-named family is
precisely the thing inter-rater agreement **cannot** detect: two annotators applying the same written
rubric will agree perfectly while both label the phenomenon wrongly.

### On property 6, the thresholds

>60% in one family means the partition carries little information. <2% means the family is either an
artifact or genuinely rare — and those look identical from the count alone, which is why §4's
precedence test exists.

---

## 3. How to decide a taxonomy — the procedure

Five steps, in this order. The order matters: steps 1–2 constrain step 3, and doing 3 first is how
you end up with a taxonomy that carves neatly and helps nobody.

### Step 1 — Enumerate the consumers and their decisions

Before proposing a single category, write down who reads the manifest and what they decide. Likely
roles, each of which wants a **different cut**:

| Consumer | The decision they make | The cut that serves it |
|---|---|---|
| Whoever commissions data collection | which demonstrations to pay for | **cause** — what to vary when collecting |
| The engineer who fine-tunes | what to change in training | **cause** or **pipeline stage** |
| Whoever decides deploy / don't deploy | is this safe enough, where | **consequence / cost** |
| A safety or compliance reviewer | which failures are unacceptable at any rate | **consequence** |
| Whoever debugs the policy | which component is at fault | **pipeline stage** |

**If the consumers want different cuts, do not average them into one taxonomy.** Pick one **primary
cut** and express the others as orthogonal axes on the same row. We already do this correctly once:
`failure_cost` (benign / disruptive / safety) is a consequence axis orthogonal to family. That
pattern generalises and is the right answer to "different readers want different things."

### Step 2 — Commit to one primary cut, and say which it is

The four cuts the field actually uses (methods survey §3.1):

| Cut | Names | Example | Requires |
|---|---|---|---|
| **Pipeline stage** | the module at fault | RoboFAC: planning / motion / execution | the consumer's system to have those stages |
| **Perturbation geometry** | the DoF that was wrong | FailGen: slip, translation, no-rotation | a generator that injects them |
| **Symptom** | what an observer sees | SO-101: grasp instability, repetition loop | nothing — cheapest to annotate |
| **Environmental cause** | what was different about the world | LIBERO-Plus: camera, layout, lighting | **that you controlled the cause** |
| *(orthogonal)* **Consequence** | what the failure costs | LIBERO-Safety; our `failure_cost` | a cost model |

**The cause cut is the only one that is directly actionable, and the only one that requires an
experiment.** In simulation we assign the treatment, so we can use it — that is the project's
structural advantage and it should not be given up lightly. A symptom cut is cheap and honest but
tells a buyer nothing about what to collect.

**Write the chosen cut into the schema.** A taxonomy whose cut is undeclared will drift back to mixed,
which is how ours got to four cuts in one list.

### Step 3 — Propose candidates from three sources, not one

- **Published sets** (survey §3.1) — RoboFAC 6, FailGen 7, SO-101 4, LIBERO-Plus 7→21,
  LIBERO-Safety 5. Start here for coverage and vocabulary; do not adopt wholesale, since each was
  built for its own corpus.
- **The data** — let the corpus propose categories. The strongest published evidence in this whole
  area is that **discovered taxonomies beat imposed ones by a wide margin**: clustering VLM-written
  failure *explanations* gave 85.53% trajectory-assignment F1 against 32.41% for embedding similarity
  on the same corpus. The transferable rule is **cluster the discrepancy, not the episode** — what
  differs between a failure and its nominal counterpart, which for counterfactual pairs we have by
  construction.
- **The consumers** — §4.

Then reconcile. Where a discovered cluster has no published name, that is a candidate finding. Where
a published family never appears in our data, that is a coverage gap or a benchmark limitation — and
it matters which (see `language_grounding`, §6.4).

### Step 4 — Check every candidate against §2's seven properties

Cheap, and it is where most candidates should die.

### Step 5 — Validate (§5) before reporting anything

---

## 4. Eliciting the taxonomy from manifest consumers

Since the family set is to be revised with consumer input, the elicitation itself needs designing.
**Do not ask people what the failure categories should be.** They will produce plausible category
names, which is the least reliable thing they can give you and the hardest to falsify later.

**Ask about decisions and money instead.** Four questions that produce usable answers:

1. **"When a row says a failure happened, what would you do differently depending on what kind it
   was?"** Produces the action set. Categories that do not appear in any answer should not exist.
   *This is the single most informative question.*
2. **"Which two of these failures would you handle the same way?"** Run over real episode pairs, not
   labels. Two episodes handled identically belong in one family regardless of how differently they
   look — **the consumer's action set is the ground truth for the partition**, and this is the only
   place ground truth exists at all.
3. **"Which failures would you pay to eliminate, and roughly in what order?"** Produces the
   consequence axis and supplies prevalence/severity weighting, which the survey establishes we
   should **not** be inventing ourselves (DG-5b: prevalence is client-supplied).
4. **"What would make you distrust this row?"** Produces the evidence requirements — whether they
   need the video, the counterfactual, the confidence, the count.

**Run it on real episodes, not on a list of names.** Show 20 actual failure videos and ask for
sorting into piles and a name per pile. This gives a partition you can measure agreement against,
and it surfaces distinctions the names would have hidden.

**Expect consumers to disagree with each other.** That is a finding, not noise — it means you need
either orthogonal axes (§3.1) or per-consumer views over a shared partition. Resolving it by
averaging produces a taxonomy that serves nobody.

**Record which families came from which consumer.** When a family later fails validation you need to
know whose decision it was serving, so you can go back and ask what they would want instead.

---

## 5. How to test that a taxonomy is good — the validation ladder

Five rungs, weakest to strongest. **Only the top rung can falsify a taxonomy.** Each states what it
does and does not establish, because conflating those is the standard error in this area.

### Rung 1 — Structural checks *(minutes, CPU)*
Reachability, concentration, abstention rate, and **precedence sensitivity** — re-assign under
permuted rule orders and report each family's range. *Establishes:* the counts are properties of the
data, not of the code. *Does not establish:* anything about correctness.

**Nobody in the published literature reports precedence sensitivity.** Every rule-based taxonomy has
an evaluation order and none report sensitivity to it. Ours swings 0–397 on one family. Run this
first, always; it is the cheapest informative test that exists here.

### Rung 2 — Inter-rater agreement *(one annotation pass)*
Two annotators, shared rubric, report **κ alongside percentage agreement and Gwet's AC₁** — failure
categories are always heavily imbalanced, which is the kappa-paradox regime. *Establishes:* the
rubric can be applied consistently. *Does not establish:* that it carves correctly. A taxonomy of
{*failure on a Tuesday*, *failure not on a Tuesday*} scores κ = 1.0.

> **κ cannot carry the load our design assigns it, and raising the threshold does not help.** It is
> the wrong instrument for a construct-validity question, at any value.

### Rung 3 — Name-based re-assignment *(one annotation pass)*
Give a held-out judge **only the family names and descriptions** and have them assign held-out
episodes; measure agreement with the classifier. *Establishes:* the name communicates the partition —
i.e. requirement 2, standalone meaning. *Does not establish:* that the property is the right one.

This is the direct test for property 2 and for the hazard that **an LLM (or a person) will name any
set fluently, including a set with no common property**, so a plausible name gets read as evidence of
a coherent cluster.

### Rung 4 — Planted-fault recovery *(needs a fixture)*
Inject known faults, check the classifier recovers them. *Establishes:* implementation correctness.
*Does not establish:* that the planted categories are the ones that occur.

Two hard limits. **(a)** The ceiling is lower than intuition suggests — the most careful published
evaluation of this kind recovers **36% of 1,235 planted structures**. **(b)** A procedurally-planted
taxonomy is structurally blind to closed-loop pathologies: `repetition_loop` cannot be produced by
perturbing a demonstration keyframe, so it is absent from every generator-derived family set. Our
oracle gate has exactly this shape, so **a passing oracle gate is evidence of implementation
correctness, not of taxonomy coverage.**

### Rung 5 — Downstream utility *(GPU; the only falsifier)*
Partition the corpus by family. For each family, gather remediation data targeting it. Fine-tune.
Compare against **the same volume** of uniformly-sampled data.

| arm | published result |
|---|---|
| before | 46% failure |
| **family-guided collection** | **18%** |
| uniform, same budget | 34% |

*Establishes:* that the partition changes what you do, usefully. **The control arm is the entire
experiment** — without it, "we collected data where it failed and the failure rate halved" is
unfalsifiable, because more data helps regardless.

**A family is useful iff guided beats uniform at equal budget, for that family's own failures.** A
family where guided ≈ uniform is decoration however clean its clusters.

*Caveat:* the published instance is indoor navigation with a CNN policy. **No manipulation equivalent
exists.** That is both the gap and the opportunity — running it would be a genuine contribution, not
an adoption.

---

## 6. Anti-patterns — the four mistakes we actually made

Each was invisible until measured. Each is cheap to check for.

### 6.1 Names that assert a cause the rule does not measure
Our rule 2 fires when the arm ends near a **different named object** — its own docstring calls this
*"object-selection failures"* — and labels it `spatial_reasoning`. Rule 3 fires on distance-with-no-
distractor, which is positional, and labels it `visual_grounding`. The names are the wrong way round
relative to the code's own comments.

It also contradicts the literature: LIBERO-Plus found models barely move when you **add distractors**
but collapse when you **displace the target**, concluding they *"may have merely learned the
positional information of the target objects"*. Discrimination is not the weak point on this
benchmark; position is. A 65% "grounding" share contradicts that, and the contradiction dissolves once
the labels are read as what they measure.

**Check:** for each family, read the rule and ask whether the name asserts anything the rule does not
observe.

### 6.2 Overlapping predicates resolved by evaluation order
76.6% of failures fire ≥2 rules. Family counts swing 0–397 across orderings. **Check:** compute the
predicate vector per episode and count multi-firing.

### 6.3 Unreachable declared families
`language_grounding` and `distribution_shift` are declared and no rule can produce them.
`COST_SAFETY` is defined and never assigned. **Check:** run the classifier over a real corpus and
look for permanent zeros. A declared-but-impossible family in a client-facing schema is a liability.

### 6.4 Unreachable abstention
`any_attempt` covers 528 and `never_reached` the other 6 — disjoint and exhaustive, so **no failure
can reach the fallthrough under any ordering**. `ambiguous = 0` looked like confidence and was
inability to decline. **Check:** confirm the residual bucket is reachable *in principle*, not just
empty in practice.

> **A distinction worth preserving in the schema.** `language_grounding` being empty is **not** our
> bug: MINERVA scores 95.05% on LIBERO with *no language encoder at all* — a 40-entry task-ID lookup
> table — and permuting that table collapses it to chance. Standard LIBERO never asks one scene to
> support two goals, so a grounding family is **not measurable there in principle**. That is a
> property of the benchmark and it is defensible to state. `distribution_shift` being empty is an
> unimplemented capability. **Same symptom, opposite meaning — the schema should distinguish
> "unreachable by benchmark property" from "not implemented".**

---

## 7. Checklist — before any taxonomy change ships

**Design**
- [ ] Consumers and their decisions written down (§3.1)
- [ ] One primary cut chosen **and declared in the schema** (§3.2)
- [ ] Other consumers' needs expressed as orthogonal axes, not folded in
- [ ] Candidates drawn from published sets, the data, *and* consumers (§3.3)

**Properties** (§2)
- [ ] Each family maps to a distinct intervention
- [ ] Each family is interpretable standalone
- [ ] No episode satisfies two families — measured, not assumed
- [ ] Residual bucket reachable in principle
- [ ] No declared family permanently unreachable — or marked, with which kind (§6.4)
- [ ] No family >60% or <2% without an explanation
- [ ] No name asserts a cause its rule does not observe

**Validation** (§5)
- [ ] Rung 1 structural checks pass, precedence sensitivity reported
- [ ] Rung 2 agreement reported as κ **+ percentage + AC₁**
- [ ] Rung 3 name-based re-assignment run
- [ ] Rung 4 planted-fault recovery, with its ceiling stated
- [ ] Rung 5 planned, scheduled, or **explicitly declared out of scope in writing**

**Reporting**
- [ ] Every count carries the corpus it is over and any excluded suite
      *(e.g. `libero_goal` is currently excluded: 200/260 traces lack `_gt_eef_to_object` — absent
      data, needs a GPU re-run, not a re-mine)*
- [ ] Severity/prevalence weighting is client-supplied, not invented
- [ ] Families validated only at rungs 1–4 are labelled as such

---

## 8. Open questions this document does not settle

1. **Which primary cut we adopt.** §3.2 argues for cause on the grounds that we assign the treatment,
   but that needs perturbation machinery we do not yet have working.
2. **Whether the current families survive the revision at all**, or whether the set is rebuilt from
   consumer input plus discovery. Decision deferred pending consumer interviews.
3. **Whether rung 5 is ever run.** Currently documented as a next step, not scheduled. Until it is,
   **our families are unvalidated in the only sense that matters**, and anything we say about them
   should be hedged accordingly.
4. **How to report a taxonomy in flux to a client** without either overclaiming or being useless.
