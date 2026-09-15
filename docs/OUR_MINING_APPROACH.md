# Our failure-mining approach

**Written 2026-09-15.** What we do between *"the episode failed"* and *"here is
a named cluster with a cause and a data requirement."*

Companion to `docs/FAILURE_MINING_METHODS.md` (how the field does it) and
`docs/COMPONENTS.md` (where each piece lives).

---

## For the non-specialist: why this is hard at all

A robot policy is evaluated by running it and checking whether the task got
done. The output is one bit per attempt. Run it a thousand times and you get a
percentage.

That percentage is nearly useless for improving the robot, because it does not
say *what went wrong*. And unlike most software, you cannot simply read the
code: the policy is a few hundred million learned numbers, and there is no line
to point at.

Worse, **there is no "correct answer" to compare against.** In ordinary testing
you know what the output should have been. Here, a robot can reach the same cup
by twenty different arm paths, all correct. So you cannot score a robot by
comparing its movements to a reference — you would flag every valid alternative
as a failure.

So the question becomes: given a video, a stream of joint angles, and a single
bit saying *it failed*, can we work out **where** it went wrong, **what kind** of
mistake it was, **which mistakes keep recurring**, and **what caused them**?

That is failure mining. Our claim is not that we detect failures better —
the simulator already tells us. It is that we turn a pile of failures into a
**prioritised, evidence-backed list of what data to collect next**.

---

## The pipeline

```
  detect      did it fail?          <- the simulator's goal predicate
  localise    WHERE did it break?   <- phase segmentation from state
  classify    WHAT kind?            <- deterministic rules, 7 families
  cluster     which recur?          <- group by (family, active knobs)
  attribute   WHY?                  <- counterfactual probes, paired
     |
  Data Gap Manifest
```

Each stage is separately fallible, and each is designed to **abstain rather than
guess**. That is the single most important design property here, and §7 explains
why.

---

## Stage 1 — Detection

**We do not build this, deliberately.** LIBERO ships a goal predicate per task
(a BDDL expression such as `On(bowl, plate)`); the simulator evaluates it and we
read the answer.

`success` is the **only** trustworthy signal in the trace, and the environment
owns it (invariant I3). The adapter **raises** if the environment fails to
supply it rather than inferring success from termination. An earlier version did
infer it, and that bug fails in the worst possible direction: inflated success,
a healthy-looking nominal cell, and the "nominal below 90% means a harness bug"
heuristic never fires because the number is too *high*.

**Limit:** a goal predicate can be satisfied by accident, and it says nothing
about *how well* the task was done. We inherit whatever the benchmark authors
encoded.

## Stage 2 — Localisation

**The stage where most of the information comes from**, and the one with the
most assumptions.

We segment each rollout into manipulation phases from simulator state — one
label per timestep, then collapsed into contiguous segments:

```python
if   holding:                 TRANSPORT     # object acquired and moving with the gripper
elif gripper_closed:          GRASP         # closure attempt in progress
elif dist_to_object < r:      PREGRASP      # within reach of the target
else:                         APPROACH      # still travelling
```

Then a second pass over the segments applies the **retry rule**: any approach or
pre-grasp occurring *after* a grasp is relabelled `RETRY`. This is what separates
"reached and missed once" from "looping".

We also record:

- **first meaningful divergence** from a nominal run at the same seed —
  reported, but explicitly a **secondary** signal. Taking a different path is
  not the same as taking a wrong path, and ranking failures by trajectory
  distance would flag every valid alternative solution.
- **terminal behaviour** — `success`, `retry_loop`, `failed_grasp_no_retry`,
  `never_reached`.
- **grasp-attempt spread** — how far apart the closure attempts were.

**Every detector declares the state keys it needs and skips with a recorded
reason when they are absent** (guarantee G8). This is not defensive
programming; it is the mechanism that lets one miner run over both a toy and
LIBERO without silently computing a number from partial state.

> **Current status on LIBERO: the segmenter abstains.** First real rollout
> returned `missing state keys: ['_gt_ee_to_obj', 'gripper', 'holding']`. The
> thresholds and key names are toy-specific; LIBERO exposes
> `_gt_eef_to_object` (a dict per object) and `gripper_qpos` (finger positions
> in metres), and has **no `holding` at all** — it must be derived from gripper
> closure plus object lift. That derivation is itself a detector, and if it is
> wrong every downstream phase is wrong.

## Stage 3 — Classification

Seven families: visual grounding, spatial reasoning, planning, manipulation,
recovery, distribution shift, and `language_grounding` — which is **declared out
of scope** for this campaign, because no environment we use implements the
language axis and no detector can produce it. κ is therefore computed over
**six** families, not seven.

Rules run **most-specific-first**, and each keys on a distinct signal rather
than on a threshold of the same signal:

| Signal | Family |
|---|---|
| never entered pre-grasp | `planning` |
| ended nearer a **distractor** than the target | `spatial_reasoning` |
| ended > 10 cm from the target | `visual_grounding` |
| ≥ 2 attempts within a 1.5 cm radius | `recovery` — *repeating, not searching* |
| reached the target but the grasp failed | `manipulation` |

Two of these rules exist because the acceptance gate caught the earlier version
getting them wrong:

**Object selection needs its own signal.** The first version inferred
"went to the wrong object" from error *magnitude*. That is wrong — a large error
equally indicates a grounding failure with no distractor present. Proximity to a
distractor is the signal; magnitude is not.

**Recovery is about spread, not count.** Every retried failure was originally
labelled `recovery`, which masked the manipulation fault underneath. Measuring
the *spread* of grasp attempts separates a policy that searched and missed from
one that repeated itself.

**Failure cost** is assigned alongside the family, from terminal state: benign
(timed out with nothing acquired), disruptive (acquired the object then dropped
it), or safety. The count of failures is not the thing a buyer cares about; the
consequence is.

**Tiering.** Deterministic rules decide everything they can. An LLM is consulted
*only* for what tier 1 leaves ambiguous, is tagged `tier3`, and is **excluded
from every headline number**. The reason is reproducibility of the audit trail,
not cost: the manifest's claims are quantitative, and a non-reproducible
component in the measurement path destroys exactly the credibility we are
selling.

## Stage 4 — Clustering

Deliberately simple: group by **(family, active perturbation knobs)**.

Embedding-based clustering of failure video is a crowded area and we have no
evidence it separates anything this does not. If the discovered-taxonomy check
(below) shows our families are arbitrary, that is when a learned clustering
earns its place.

Each cluster carries its member count, tasks affected, example rollout ids, mean
terminal error, and the set of terminal behaviours observed.

## Stage 5 — Attribution

**The stage we think is genuinely underexploited in robotics**, and the one that
turns a descriptive cluster into an actionable one.

A stress cell perturbs several things at once. To establish *which* one caused
the failure, we re-run **the same seeds** with **one knob reverted at a time**:

```
full perturbation      (yaw+15, offset+4cm, 2 distractors)   15%
revert yaw only                                              73%   <- +58 pp
revert offset only                                           11%
revert distractors only                                      18%
```

Reverting yaw recovers the loss; reverting the others recovers nothing. Yaw is
the trigger — and this is an **interventional** claim, not a correlational one.
We changed one thing and measured the effect.

Three properties make it defensible:

1. **Randomised treatment.** We assign the perturbation; it is not observed.
   This is a stronger setting than the observational counterfactual work that
   formalised this idea for LLM agents.
2. **Paired statistics.** The same seeds run in both arms, so the outcomes are
   paired and we use McNemar's exact test on the discordant pairs rather than
   comparing two rates. Free variance reduction.
3. **A measured noise floor.** `reproducibility_floor()` runs one cell twice at
   fixed settings; a knob is attributed only when its effect clears
   `max(20 pp, 3 × floor)`. You cannot call an effect real until you know what
   zero looks like.

A row whose trigger has no counterfactual behind it is downgraded to
**"correlational"** in the manifest, explicitly.

---

## What the manifest does *not* claim

**We do not ship a severity ranking.** Severity conflates two things with
opposite transfer properties: `P(fail | condition)`, which we measure and which
is a claim about the policy, and `P(condition)`, which in simulation is an
artifact of the perturbation grid *we chose*. The literature reports that
severity *ordering* transfers worst from sim to real — and it is the half we
invented that does the damage.

So the manifest reports `failure_conditional` with a confidence interval, and
`condition_prevalence` is a **client-supplied column** from their deployment
logs. Severity is computed at delivery.

This also fixes a silent bug: a failure mode found only by the adaptive
(failure-seeking) sampling arm has no uniform-arm frequency, so under the old
definition it ranked *low* — the manifest systematically buried the output of
its own most novel component. It now renders `prevalence: not estimated`, which
is the truthful statement. **Silent demotion and honest abstention look
identical in a ranked table and are completely different claims.**

---

## How we know any of this works

**The oracle.** We plant a known fault in a scripted policy, run the full
pipeline, and check the diagnosis against ground truth. If the miner cannot
recover a fault we inserted ourselves, the design is wrong and we find out for
free rather than after thousands of GPU-hours.

Current: **3 of 4 planted faults recovered, plus a control** — a *clean* policy
under the same perturbations must produce no manifest rows. A miner that flags a
healthy policy is worse than useless.

The fourth is `recovery`, and it is **unvalidated**. The toy could not isolate
it: perception noise re-rolls every step, so repeating an identical grasp
eventually succeeds — repetition is a *winning* strategy there and would not be
on real hardware. Recovery is also inherently second-order; it cannot fail
alone, only conditional on a prior failure. We flagged this rather than tuning
the fixture until it agreed with us.

---

## Open problems — stated, not solved

**1. Thresholds have no principled source on LIBERO.** The proposal is to derive
them from the ~50 human demonstrations per task: the eef-to-object distance at
the moment the gripper closes *is* the grasp radius, measured rather than
guessed. But that fits a discriminative boundary using only the success class.
It is defensible **only if framed as coverage** — "the region from which
successful grasps are observed to occur, 95th percentile" — and not as "the
distance separating manipulation failures from grounding failures", which
demonstrations cannot speak to.

**2. Family assignment has no fixture on LIBERO.** On the toy we validate the
classifier because we planted the fault. On LIBERO there is none. Demonstrations
cover only the success path. Degrading a policy gives a known *trigger* — we
chose the perturbation — but **not a known family**: a yaw shift may produce a
grounding failure, a grasp failure, or none, and the policy is under no
obligation to fail in the family we expect.

So inter-annotator agreement becomes the *sole* remaining validation of family
assignment, which promotes it from a nice-to-have to load-bearing — and κ ≥ 0.5
(moderate agreement) is thin for that role.

**3. Phase grammar is not universal.** The approach → pre-grasp → grasp →
transport → place sequence assumes pick-and-place. LIBERO includes opening
drawers, operating a stove, and multi-stage long-horizon tasks. The original
proposal warned about exactly this. The grammar must be task-conditional, and a
task without a declared grammar should skip the ordering check with a recorded
reason — not have the check loosened until it passes, which would destroy it.

**4. The whole layer has never seen a real VLA failure.** Everything above is
validated on a toy with planted faults. On first contact with LIBERO it
abstained — correctly, but it abstained.
