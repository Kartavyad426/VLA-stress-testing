# THE RETRAINING-DEFAULT TAXONOMY

*Written 2026-09-16. An instantiation of `docs/TAXONOMY_FAMILY_GUIDELINES.md` for one assumed
consumer action. Pairs with `docs/CONSUMER_ELICITATION_DESIGN.md`, which turns out to be the escape
hatch for this rather than a separate exercise.*

---

## 0. The premise, and what it collapses

**Assume the consumer's default response to a failure is: collect more data and fine-tune.**

That assumption is doing more work than it looks. In the elicitation design, Q1 — *"what would you do
about it?"* — is the question whose answer defines the families. If the answer is always "retrain",
**Q1 collapses and can no longer carry the partition.**

What carries it instead is Q1's *follow-up*: **"demonstrations of what?"**

> **Under the retraining default, a family is not a description of a failure. It is a data
> specification.** Every family must answer: *what do I add to the training set, and how will I know
> it worked?*

This is the sharpest constraint anyone has put on this taxonomy, and it is a good one, because it
makes families **falsifiable against an artefact we already have** — the training set. A family that
cannot name a hole in the data is not a family under this cut.

It also explains something the guidelines asserted without justifying: that the **cause cut** is the
only directly actionable one (§3.2). Now we can say why. *A cause is a data specification.* The axes
you vary when collecting demonstrations are the same axes a cause taxonomy names.

---

## 1. The family set

Four actionable families and one router. Each actionable family carries a **checkable claim about the
training corpus** — that is what distinguishes this cut from every symptom cut.

### A — COVERAGE GAP
*The situation is in-distribution in kind, but under-sampled.*

| sub | the hole | spec | how it is checked |
|---|---|---|---|
| A1 **placement** | target pose/position rare in demos | N more demos of this task, sampling that region | count demos with the target in that region |
| A2 **object** | this object or class is rare or absent | demos using that object | count demos containing it |
| A3 **phrasing** | this referring expression is rare | demos with that instruction form | count matching instructions |

**Cheapest family to fix and the most common.** The claim — *"the training set contains N demos in
this region, and N is small"* — is a count, and either true or false.

### B — CONDITION GAP
*A nuisance variable is outside the demonstrated range.*

Sub-families: **viewpoint**, **illumination/appearance**, **clutter/distractors**.

*Spec:* re-render or re-collect existing demos under the condition — not new task data, the same
tasks under different conditions.
*Check:* measure the condition's range across the training set and show the failure sits outside it.

> **B is the family where "collect data" is often the wrong answer, and the manifest must say so.**
> Moving a camera back, adding a lamp, or fixturing the part is usually cheaper than collecting a
> viewpoint-robust dataset. So every B row should carry **both** options and their relative cost. This
> is the one family where the retraining default should actively be questioned, and it is a large
> fraction of what we can generate in simulation — which biases us toward proposing data where a
> bracket would do.

### C — SKILL GAP
*The required behaviour is absent from the demonstration set entirely.*

Sub-families: **recovery/regrasp after a failed attempt**, **mid-motion correction**.

*Spec:* demonstrations that **contain the behaviour** — which means demos containing a failure and a
recovery, not more successes.
*Check:* does any demo contain a failed-then-recovered segment? For LIBERO the answer is no, and the
published position is general: VLAs are trained on failure-free demonstrations, which is why an entire
2026 subfield (FLARE, RedFlow, RePO-VLA, B2FF, FAR) exists to add recovery they lack.

> **C is the family where more of the same data cannot help, and that must be visible on the row.**
> A client who responds to a C row by collecting another 500 nominal demonstrations has spent money
> to no effect. Of the four, C is the one whose misclassification is most expensive — which is exactly
> why the audit's finding that `recovery` is suppressed by rule precedence (24 instances reported as
> 10) matters commercially and not just methodologically.

### D — PRECISION GAP
*The skill is present, but the tolerance required is tighter than what was demonstrated.*

*Spec:* demos emphasising the fine phase, or more demos at the tight end of the tolerance
distribution.
*Check:* compare the clearance/tolerance distribution in demos against what the failing instances
demand.

### E — NOT A DATA PROBLEM → route to elicitation

| sub | meaning |
|---|---|
| E1 **environment** | fix the setup, not the policy |
| E2 **task specification** | the task or instruction is the problem |
| E3 **unsolvable / defect** | the instance cannot be done, or our harness broke it |
| E4 **acceptable** | this failure is tolerable at its rate |

**E is not a residual bucket.** It is a *router*, and §3 is about where it routes to.

---

## 2. Why this cut is better than what we have — one property

Every family in A–D makes a claim of the form *"the training distribution lacks X"*, and **X is
countable in a dataset we possess.**

Nothing else in this project has that property. Our current families make claims about what the policy
*perceived* or *reasoned* — claims with no artefact to check them against, which is why the audit
found labels that had drifted from their names with nothing to catch it. A data-gap claim cannot drift
silently: someone counts, and it is right or wrong.

Three things follow:

1. **We get a ground truth that is not a human opinion.** The guidelines' validation ladder (§5) tops
   out at "downstream utility", which needs a GPU and a fine-tuning loop. **Corpus-checkable claims sit
   below that and need neither.** A family whose claimed hole isn't in the data is refuted on a laptop.
2. **The utility experiment becomes natural rather than bolted on.** If you can *measure* the hole, you
   can fill it and re-measure. The guided-vs-uniform design (46→18% vs 46→34%) stops being an
   experiment we would have to invent and becomes the obvious next step.
3. **A row that cannot name a countable hole is not shippable.** That is a concrete admission rule for
   the manifest, and we have not had one.

---

## 3. How the two modes connect — E is the bridge

The elicitation questions are not a separate exercise running in parallel. **They are what E routes
to.**

```
     failure
        │
        ├─ can it be specified as data?  ── yes ──▶  A / B / C / D
        │                                             manifest row, shipped
        │
        └─ no ──▶ E ──▶ the elicitation cards
                        Q1 options 2–5 (setup / task / acceptable / escalate)
                        are exactly E1 / E2 / E4 / E3
                                    │
                                    └──▶ answers refine the taxonomy,
                                         and may move rows back into A–D
```

The mapping is exact, and it was not designed to be — the elicitation's non-data actions and E's
sub-families are the same set arrived at twice from different directions. That is mild evidence the
cut is natural rather than invented.

**Consequences worth stating:**

- **We can ship before anyone answers anything.** Default mode produces A–D rows immediately. E rows
  are marked *pending consumer input* rather than guessed at.
- **The questions get cheaper and better targeted.** Instead of 20–40 cards sampled across everything,
  show the consumer the E cases specifically — the ones we genuinely cannot decide. That is a shorter
  session and a higher information rate per card.
- **Q2 (cost) still applies to every family, not just E.** Severity is client-supplied regardless of
  whether the fix is data, and A–D rows need it to be prioritised.
- **Disagreement about E4 is the commercially important one.** If we route something to "not a data
  problem, probably acceptable" and the consumer says it must never happen, that is the most valuable
  single answer the elicitation can produce.

---

## 4. What a manifest row looks like under this cut

| field | example |
|---|---|
| failure mode *(observed)* | target object never moved; arm ended 14 cm away |
| count / prevalence | 31 of 181 failures on this task |
| **family** | **A1 — placement coverage** |
| **the claimed hole** | **target in the left third of the workspace** |
| **evidence for the hole** | **4 of 377 training demos have the target in that region** |
| **spec** | **~50 demos, target sampled across that region** |
| **how we would verify** | **re-run this task set; compare against 50 uniformly-sampled demos** |
| cost | *client-supplied* (elicitation Q2) |
| confidence | tier-1 rule, deterministic |

The three bold middle rows are what this cut buys. The verification row is what makes it a claim
rather than advice.

---

## 5. The sequencing consequence — and it de-risks the project

**Mining is separable from simulation, and that means the taxonomy does not block the campaign.**

- Simulation produces traces: GPU, slow, scheduled, hard to repeat.
- Mining produces manifests: CPU, fast, offline, **re-runnable over stored traces any number of
  times**.
- Consumer answers arrive on human time.

So the order is not *decide the taxonomy → run the campaign → mine*. It is: **run the campaign once,
mine repeatedly.** Every time the taxonomy changes — when elicitation answers land, when a family is
refuted against the training corpus — we re-mine the same traces and get a new manifest for free.

Two things this protects against, both of which have already bitten:

- The audit's precedence result (76.6% of labels decided by rule order) would have been a catastrophe
  if the taxonomy were baked in at capture time. It wasn't, so it cost a re-mine.
- The invalidated camera arms need re-*capture*, not just re-mining — which is precisely the
  expensive case, and the contrast shows why keeping the boundary in the right place matters.

**One real dependency runs the other way**, and it should be designed for now rather than discovered
later: mining can only produce specs for variables the traces **recorded**. `libero_goal`'s missing
fixture bodies are the worked example — no amount of re-mining recovers a state key that was never
captured. So capture should record the full scene descriptor even where no current detector reads it.
Storage is cheap; a re-run is 42 minutes of GPU and rising.

---

## 5b. HOW THE CHECK ACTUALLY RUNS — three levels, and only one is stochastic

**A/B/C/D are not per-episode labels.** A "placement coverage gap" is not a property of an episode; it
is a property of the *relationship between a group of episodes and the training corpus*. So the
assignment happens at three levels, at different granularities.

### Level 1 — per episode, deterministic

Compute an **observable feature vector**. Assign **no family**. Rules over numbers, fully
reproducible, no thresholds that encode a conclusion.

This is also the fix for the precedence problem (`FAILURE_FAMILY_AUDIT.md` §5b): emit the vector, not
a collapsed label, so nothing is decided by rule order.

> ### The constraint that determines the feature set
>
> **Every feature must be computable for BOTH our rollouts AND the training demonstrations**, or the
> corpus comparison is not like-for-like.
>
> Our rollouts carry privileged simulator state. **The training set does not.** `lerobot/libero`
> (1,693 episodes, 273,465 frames, 40 tasks) has exactly: `observation.images.image`/`image2`
> (256×256 video), `observation.state` (8-dim), `action` (7-dim), and `task_index`. **No object
> poses. No contacts. No BDDL state.**
>
> So the shared space is **the policy's own observation space**: eef pose, gripper, actions, images,
> instruction. Privileged state can inform *clustering* but can never appear in a *hole claim*,
> because there is nothing to compare it against.
>
> This is a constraint that turns out to be the right design anyway. The hole should be defined in the
> space the policy actually learns from — and a feature set that avoids privileged state is one that
> transfers to a real robot, where privileged state does not exist.

Per-episode features, all from the shared space:

| feature | from |
|---|---|
| eef pose at the grasp moment | `state[0:3]` at the first frame where `action[6] > 0` |
| terminal eef pose | `state[0:3]` at the last frame |
| gripper open/close sequence | `action[6]` sign changes |
| episode length | frame count |
| image statistics | frame pixel moments |
| instruction | task string |

### Level 2 — group episodes, **stochastic, and the part needing validation**

Cluster on those features. This is the only step with choices in it — embedding, algorithm, k — and
therefore the only one carrying the survey's §4 warnings: cluster the discrepancy rather than the
episode, prefer a method that can say *noise*, don't pick k by silhouette, and **run the null-corpus
control**.

### Level 3 — per cluster, deterministic: query the corpus

For each cluster, ask whether the region it occupies is represented in the training set. **A count
against a threshold.** Deterministic given the cluster.

**Worked example, run against the real training set:**

```
grasp-point coverage per task (the A1 proxy), from 1,617 of 1,693 demos
  task    n   gx mean   gx sd   gy mean   gy sd
     0   38    -0.110   0.018    -0.115   0.016
     3   41    -0.205   0.017     0.199   0.011
     7   49     0.105   0.019    -0.200   0.017

hole query — task 0, is the band y ∈ [0.15, 0.25] covered?
  demos with a grasp point in that band: 0 of 38     ->  HOLE
  grasp-y range the demos actually span: [-0.147, -0.090]
```

**And the measurement itself is a finding.** Grasp points in the demonstrations have a standard
deviation of **1.1–2.6 cm per task**, spanning ranges of ~5 cm. The training set contains almost no
variation in where the object is grasped. That is a direct, countable explanation for LIBERO-Plus's
conclusion that these models *"merely learned the positional information of the target objects"* — the
demonstrations give them no reason to learn anything else, and **any target displacement beyond about
±5 cm is outside the demonstrated distribution.**

### Which families are computable this way, and which are not

| family | corpus check | status |
|---|---|---|
| **A1** placement | grasp-point distribution per task | **demonstrated above** |
| **A2** object | `task_index` → instruction; count episodes | direct |
| **A3** phrasing | instruction strings | **degenerate on LIBERO — 40 tasks, one instruction each, no paraphrases.** Unmeasurable here, like `language_grounding`, and for the same reason |
| **B** condition | image statistics, demos vs our renders | direct — and it is exactly the pixel-statistics check `vla-7f` recommends for conformance. **One measurement, two uses.** |
| **C** skill gap | search demos for close→open→close without progress | computable; needs a detector, and the expected answer is zero |
| **D** precision | spread of terminal eef pose per task | proxy only — true tolerance needs object poses the corpus lacks |

### A discrepancy worth chasing separately

**The training set declares `fps = 10`. Our evaluation ran at `--env.fps=20`** (`baseline.log:61`).
`vla-7f` found `lerobot#4614` documenting that a non-20 Hz `--env.fps` silently rescales delta
actions; this is the mirror image — a 10 Hz corpus against a 20 Hz env. Whether LIBERO's native
control rate reconciles them, or the dataset is subsampled, I have not established. It belongs on the
Q2 mismatch list either way.

---

## 6. What this cut does not solve

1. **It assumes data can be collected.** For a client without a teleop rig, an A1 row is not
   actionable. The spec should say *how many* and *of what*, and let them price it.
2. **Counting a hole does not prove filling it helps.** The corpus check refutes bad families cheaply;
   it does not confirm good ones. Only the utility experiment does that, and it is still unscheduled.
3. **Our holes are simulated holes.** A gap in LIBERO's demo distribution is evidence about LIBERO. The
   claim that it transfers to a client's corpus is separate and currently unsupported.
4. **B is where we are structurally biased.** Simulation makes condition variation cheap to *generate*,
   which will make us over-propose condition data relative to fixing the environment. The
   both-options-with-costs rule in §1 is the guard, and it needs enforcing rather than remembering.
5. **C depends on a family the audit found suppressed.** Recovery is currently under-counted by rule
   precedence, and it is the family whose misclassification wastes the most client money. Fixing the
   precedence problem is a prerequisite for shipping C rows, not an independent cleanup.
