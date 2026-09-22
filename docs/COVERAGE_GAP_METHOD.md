# HOW WE TEST WHETHER A FAILURE IS A COVERAGE GAP

*Written 2026-09-18. Method, results, weaknesses and sources for the question: **is this failure
caused by a hole in the training data, or by something else?***

> Why it matters: `docs/RETRAINING_DEFAULT_TAXONOMY.md` §1 makes **family A (coverage gap)** the
> cheapest and most common prescription — *"collect N demonstrations of X"*. That prescription is only
> honest if the hole is real.
>
> **Headline, and it took two attempts to get right:** on nominal LIBERO, failures **do** start from
> covered positions (§3) but **leave the demonstrated manifold during execution** (§3b) — spending
> **35–52% of their timesteps out of distribution against 4–6% for successes**. So the hole is real,
> but it is **not** where family A says it is. It is the off-manifold region a successful
> demonstration never visits, which makes these **family C (skill gap)**, and the fix on-policy
> corrective data rather than more nominal demonstrations.
>
> **A confound remains open:** whether drift causes failure or merely accompanies it. §3b names the
> measurement that would settle it and it has not been run.

---

## 1. The question, stated so it can be answered

A "coverage gap" claim is: *the failing situation lies outside the region of input space the training
demonstrations occupy.* That is a **support** question, not an accuracy question, and it has a
standard shape in the literature: given a training set and a query point, **how far is the query from
the training distribution's support?**

Two possible answers and their consequences:

| finding | meaning | prescription |
|---|---|---|
| failures sit **far** from the training data | genuine coverage gap | collect data there — family A applies |
| failures sit **inside** the training data | not a coverage gap | collecting more there adds density where density already exists. **Family A does not apply** |

---

## 2. The method

### Step 0 — the constraint that dictates everything

**The training corpus contains no privileged state.** `lerobot/libero` ships exactly two video
streams, `observation.state` (8-dim), `action` (7-dim) and `task_index` **[F]**. No object poses, no
contacts.

So any coverage claim must be expressed in **the policy's own observation space** — the only space
both corpora share. That is a limitation and also the right call: the hole should be defined in the
space the policy learns from, and a feature set that avoids privileged state is one that transfers to
a real robot.

### Step 1 — build a per-task reference cloud from the demonstrations

For each of the 1,693 demo episodes:

1. Find the first frame where the gripper is **commanded closed** — `action[6] > 0`.
2. Take the end-effector position at that frame — `observation.state[0:3]`, a world-frame xyz.
3. Group by `task_index`.

This yields a point cloud per task: **where, in space, the demonstrations grasp.** 1,617 of 1,693
episodes have a detectable grasp moment.

*Why this quantity:* it is a physically interpretable proxy for **where the manipulated object is**,
it is computable identically on both sides, and it is the exact variable a *placement*-coverage claim
is about. The demos have no object poses, so the grasp point is the closest available stand-in.

### Step 2 — join our rollouts to the demo tasks

On the **instruction string**, exact match, verified: our
`"pick up the black bowl between the plate and the ramekin and place it on the plate"` matches a
`tasks.parquet` index entry exactly **[F]**. There are 40 tasks and one instruction each, so the join
is unambiguous.

### Step 3 — score each of our episodes

Distance = **minimum Euclidean distance from the query point to that task's demo cloud** — a 1-nearest-
neighbour distance to the training support.

Then compare the distance **distribution for failures against the distribution for successes**. This
comparison is the actual test: an absolute distance means little, but *failures being systematically
further out than successes* is what a coverage-gap story predicts.

### Step 4 — two choices of query point, and why the second is the real one

**Version A — the policy's own grasp point.** Same definition as the demos: eef position at the first
commanded close. Falls back to terminal eef where the gripper never closes.

**Version A has a circularity problem that makes it close to unfalsifiable.** The policy *learned*
where to grasp from these demonstrations. So it will grasp near demo grasp points whether it succeeds
or fails, and "failures are in-distribution" is partly a statement about its training rather than
about the scene. **This is the deepest weakness of the first measurement and it is why there is a
version B.**

**Version B — the object's initial position** (`_gt_object_pos` at t=0). This is set by the
environment's init state and is **independent of the policy**. It cannot be tautological. The
comparison is now slightly mismatched — our object position against the demos' grasp position — but
that mismatch is bounded (a grasp point is near the object it grasps) and the policy-independence is
worth more than the tidiness.

**Version B is the measurement to trust.**

---

## 3. Results

### Version A — policy grasp point vs demo cloud

| suite | | n | median | >5 cm |
|---|---|---|---|---|
| spatial | success | 58 | 0.9 cm | 0 |
| spatial | fail | 22 | 1.3 cm | 0 |
| object | success | 71 | 0.6 cm | 0 |
| object | fail | 9 | 1.3 cm | 0 |
| goal | success | 46 | 1.0 cm | 0 |
| goal | fail | 14 | 1.7 cm | 0 |
| libero_10 | success | 37 | 1.1 cm | 0 |
| libero_10 | fail | 28 | 0.9 cm | 2 |

### Version B — object initial position vs demo cloud *(policy-independent)*

| suite | | n | median | p90 | >5 cm |
|---|---|---|---|---|---|
| spatial | success | 58 | 3.4 cm | 4.4 | 1 |
| spatial | **fail** | 22 | **2.9 cm** | 4.4 | 1 |
| object | success | 71 | 1.4 cm | 4.0 | 0 |
| object | **fail** | 9 | **0.7 cm** | 1.6 | 0 |
| goal | success | 46 | 2.5 cm | 10.1 | 8 |
| goal | **fail** | 14 | **1.9 cm** | 8.4 | 2 |
| libero_10 | success | 37 | 3.5 cm | 10.3 | 16 |
| libero_10 | **fail** | 28 | **3.0 cm** | 8.8 | 9 |

### What it says

**1. Failures and successes start at INDISTINGUISHABLE distances from the training data.** Medians
differ by 0.5–0.7 cm with failures nominally nearer, but a Mann-Whitney U test says that direction is
**not significant in any suite** — p = 0.416 (spatial), 0.167 (object), 0.070 (goal), 0.292
(libero_10).

> **CORRECTION.** An earlier version of this line said failures are "slightly nearer" in all four
> suites and treated the direction as evidence. **It is noise.** The finding is that both groups sit
> *inside* the demonstrated distribution and are not separable there — not that failures are nearer.

**2. Where genuine out-of-distribution instances exist, the policy does *better* on them, not worse.**
`libero_10` has 25 nominal episodes beyond 5 cm — **16 succeeded and 9 failed, a 64% success rate
against the suite's overall nominal 56.9%.** The instances furthest outside the demonstrated
distribution are not the ones it fails.

**3. The reference clouds are extremely tight.** Per-task standard deviation **1.1–2.6 cm**, spanning
~5 cm. That is the underlying fact: the demonstrations contain almost no positional variation, which
is a direct countable explanation for LIBERO-Plus's conclusion that these models *"merely learned the
positional information of the target objects"*.

**Conclusion, on the initial state only: failures do not start from uncovered positions.**

> ### ⚠ AND THAT IS THE WRONG OBJECT TO MEASURE — see §3b
>
> Prompted by the question *"shouldn't we match failures against the training data to see if they are
> OOD?"*, I re-ran this against the **visited state trajectory** rather than the initial state. **The
> conclusion reverses.** Coverage is not a property of where an episode *starts*; it is a property of
> the states the policy *visits*. §3b is the measurement that matters and this section is retained
> only because its negative result is still true of initial states.

### 3b. The measurement that matters — trajectory-level, calibrated

Same 1-NN idea, three changes, all of which address weaknesses listed in §5:

- **Feature space:** the policy's own `observation.state`, restricted to the 5 dimensions matchable
  without a rotation-convention assumption — **eef xyz + both gripper fingers** — and **normalised by
  the training corpus's own `meta/stats.json` mean/std**, i.e. the same normalisation the policy sees.
- **Query object:** **every timestep of the rollout**, not one point.
- **Threshold: calibrated, not invented.** Leave-one-out demo-vs-demo 1-NN distance, n = 470:
  **median 0.070, p95 = 0.227 normalised units.** That p95 is the threshold, derived from
  in-distribution data exactly as Sun et al. prescribe.

| suite | | n | median dist | p90 of per-episode max | **% of steps OOD** |
|---|---|---|---|---|---|
| spatial | success | 58 | 0.062 | 0.307 | **4.2%** |
| spatial | **fail** | 22 | **0.161** | **0.927** | **34.8%** |
| object | success | 71 | 0.071 | 0.375 | **5.7%** |
| object | **fail** | 9 | **0.267** | **1.191** | **52.4%** |
| goal | success | 60 | 0.060 | 0.342 | **3.8%** |
| goal | **fail** | 20 | **0.179** | **0.800** | **35.5%** |
| libero_10 | success | 37 | 0.071 | 0.457 | **5.8%** |
| libero_10 | **fail** | 28 | **0.179** | **2.088** | **41.8%** |

**Failures spend 35–52% of their timesteps outside the demonstrated manifold. Successes spend
4–6%.** A 6–9× separation, consistent across all four suites and in the same direction every time.

Note also that **successes are statistically indistinguishable from the demonstrations themselves** —
median 0.060–0.071 against a demo-vs-demo median of 0.070. A successful rollout stays on the manifold;
a failing one leaves it.

### So failures ARE out-of-distribution — but not where I first looked

Both measurements are correct and they answer different questions:

| | finding |
|---|---|
| **initial state** | covered. Failures start from positions the demos cover densely (§3) |
| **visited states** | **not covered.** Failures leave the manifold during execution (§3b) |

This is **covariate shift under closed-loop execution** — the canonical behaviour-cloning failure
(Ross & Bagnell; the premise of DAgger). The policy begins on-distribution, accumulates error, and
ends up in states no demonstration covers, where it has no training signal.

**The consequence for the taxonomy is a re-assignment, not a retirement:**

- **Not family A (placement coverage).** More demonstrations at the same start positions add density
  where density already exists — §3 stands on that.
- **Family C (skill gap).** The uncovered region is the **off-manifold recovery region**, and it is
  absent from the corpus by construction, because the demos are successful trajectories that never go
  there. This is the same structural fact as *"VLAs are trained on failure-free demonstrations"*, now
  measured on our own data rather than cited.
- **So the fix is on-policy corrective data, not more nominal demos** — DAgger-style correction, or
  recovery segments of the kind `samwald/robotwin-failure-recovery`'s `expert_corrected` branch
  contains and we have none of.

### ⚠ The confound this does not resolve

**Direction of causation is not established.** A failing episode ends up somewhere odd almost by
definition, so "failures are OOD" may be a *consequence* of failing rather than a cause.

**The discriminator is onset timing**, and it is measurable from data already computed: does the OOD
excursion *precede* the point of failure, or only appear at the end? If OOD onset leads the failure,
drift is plausibly causal; if it coincides with the terminal state, it is a symptom. **That
measurement has not been run**, and no causal claim should be made until it has.


---

## 3c. WE DO NOT NEED THE TRAINING CORPUS — self-referenced OOD

Everything above used `lerobot/libero` as the reference. **For a client's policy, or GR00T, we will
not have the training data** — it may be proprietary, or 107k episodes across many corpora.

So: re-run §3b with the reference replaced by **the policy's own successful rollouts**, per task,
leave-one-out. No training corpus touched at any point.

| suite | reference eps | | n | median | **% steps OOD** |
|---|---|---|---|---|---|
| spatial | 58 | success | 58 | 0.100 | **4.9%** |
| | | **fail** | 22 | **0.405** | **54.9%** |
| object | 71 | success | 71 | 0.103 | **5.1%** |
| | | **fail** | 9 | **0.406** | **48.7%** |
| goal | 60 | success | 60 | 0.110 | **5.0%** |
| | | **fail** | 20 | **0.458** | **47.9%** |
| libero_10 | 36 | success | 36 | 0.107 | **4.6%** |
| | | **fail** | 14 | **0.278** | **32.5%** |

**It works, and it separates at least as well as the training corpus did** — failures at 32–55% of
steps OOD against ~5% for successes, a 6.5–11× margin, versus 35–52% vs 4–6% when referenced against
the demos.

Two things to notice:

1. **The success rows land at ~5% by construction, and that is the check that the calibration is
   working.** The threshold is the 95th percentile of leave-one-out success distances, so successes
   *must* come out near 5%. They do, in all four suites. The false-positive rate is a design
   parameter, not an outcome.
2. **Absolute distances shift but conclusions do not.** Self-referenced medians (0.100–0.110) are
   higher than demo-referenced (0.060–0.071) simply because 58 episodes make a sparser cloud than
   ~6,800 demo frames. The calibration absorbs it — which is the property that makes the method
   portable.

### The design this implies: three independent choices, none needing the training data

| choice | what we used | the general answer | needs |
|---|---|---|---|
| **reference set** | demos, then **successful rollouts** | any distribution of **known-good behaviour** — always generable by running the policy | rollouts |
| **feature space** | 5 proprioceptive dims, normalised | the **policy's own embedding** (Sun et al.), or **policy-internal signals** like action-chunk disagreement | model access, *not* data |
| **threshold** | leave-one-out p95 | **conformal prediction** — distribution-free, black-box | a calibration split |

**Only the middle row needs anything from the model, and none of the three needs the training
corpus.** That is the answer to "how do we build a space to analyse OOD for a policy whose training
distribution we do not have".

This also converges with the survey's independent conclusion
(`FAILURE_MINING_METHODS.md` §2.11 item 2): *"Policy-internal signals beat everything else, and the
margins are large … the policy's own prediction inconsistency is self-referential — no reference
trajectory, no demo corpus, no environment state."* We reached the same place from the data side.

### Caveats specific to §3c

- **Reference size.** 3–7 successful episodes per task here. Sparse, and it inflates absolute
  distances. Calibration handles the threshold but not the variance.
- **It requires the policy to succeed sometimes.** On a task at 0% success there is no reference
  set — and that is exactly the regime a client most wants analysed.
- **Same open confound as §3b:** drift may accompany failure rather than cause it.

---

## 4. What this method is, formally

**It is the deep-kNN OOD test with a hand-chosen feature space instead of a learned one.**

The canonical version is Sun et al., *Out-of-Distribution Detection with Deep Nearest Neighbors*
(ICML 2022): extract an embedding for a test input, compute the distance to its k nearest neighbours
in the training set, threshold. The authors describe it as **non-parametric level-set estimation**,
partitioning inputs into in- and out-of-distribution by k-th nearest-neighbour distance, with the
threshold set on **in-distribution data only** — typically so 95% of training data falls inside.

Ours differs in three ways, each of which is a weakness:

| | canonical | ours |
|---|---|---|
| feature space | learned penultimate embedding | **hand-chosen 3-vector** (world-frame position) |
| k | k > 1, tuned | **k = 1** |
| threshold | calibrated so 95% of training data is inside | **5 cm, picked because the clouds span ~5 cm** — not calibrated |

---

## 5. Weaknesses, in order of how much they should worry you

1. **One axis.** Position only. It says nothing about coverage in **appearance** (lighting, texture),
   **object identity**, **phrasing** (degenerate here — one instruction per task), or **dynamics**. A
   failure could be an appearance-coverage gap and this test would call it in-distribution.
2. **Hand-chosen feature space.** The policy's own notion of similarity lives in its embedding, and
   two scenes 1 cm apart in eef space can be far apart in its features. Sun et al. use penultimate
   features precisely for this reason. **The principled version of this measurement runs in the
   policy's embedding, and we have not done it.**
3. **Threshold not calibrated.** 5 cm came from the spread of the clouds, not from a leave-one-out
   estimate on the demos. The fix is standard and cheap: hold out demos, compute their own 1-NN
   distances, set the threshold at the 95th percentile.
4. **k = 1 ignores density.** A 1-NN distance cannot distinguish "one demo nearby" from "forty demos
   nearby". A coverage claim is really about density, and a k-NN or kernel-density estimate would say
   more.
5. **Version A's circularity** (§2 Step 4) — addressed by Version B, but Version A's numbers should
   not be cited alone.
6. **Version A's fallback is a different quantity.** Episodes that never close the gripper use
   terminal eef, and **I did not record how many did.** That should be counted before the Version A
   table is reused.
7. **Small n for failures** — 9 to 28 per suite, so per-suite medians are soft. The consistency of
   direction across four suites is the load-bearing part, not any single row.
8. **Version B compares object position to grasp position.** Bounded mismatch, but a mismatch.

---

## 6. How to do it properly, in cost order

1. **Calibrate the threshold** by leave-one-out on the demos (95th percentile of in-distribution 1-NN
   distance). Minutes, and it converts an arbitrary 5 cm into a defensible number.
2. **Report k-NN density, not 1-NN distance** — how many demos within r, for a few r.
3. **Add axes.** Image statistics against the demo frames gives appearance coverage, and that is the
   same pixel-statistics measurement `docs/POLICY_SIM_COUPLING.md` recommends for conformance
   checking. **One measurement, two uses.**
4. **Run it in the policy's embedding**, which is the canonical method and the only version that
   tests coverage as the *policy* experiences it.
5. **Then test the prescription, not the diagnosis.** Even a correctly-found hole does not prove
   filling it helps — that needs the guided-vs-uniform experiment
   (`TAXONOMY_FAMILY_GUIDELINES.md` §5 rung 5).

---

## 7. Sources

### The method we are using, properly done

- **Sun, Ming, Zhu, Li — *Out-of-Distribution Detection with Deep Nearest Neighbors*.** ICML 2022.
  [arXiv:2204.06507](https://arxiv.org/abs/2204.06507) ·
  [PMLR](https://proceedings.mlr.press/v162/sun22d/sun22d.pdf).
  The canonical k-NN-distance OOD test; non-parametric, no distributional assumption, threshold from
  ID data only. Reports a 24.77% FPR reduction over a Mahalanobis-based baseline on ImageNet-1k.
  **Read this one first** — it is the method our measurement approximates.
- **Yang et al. — *A Unified Survey on Anomaly, Novelty, Open-Set and Out-of-Distribution
  Detection*.** [arXiv:2110.14051](https://arxiv.org/abs/2110.14051). Orientation on how these four
  literatures relate; useful because "is this a coverage gap" gets asked in all four vocabularies.
- **Galesso et al. — *Far Away in the Deep Space: Dense Nearest-Neighbor-Based OOD Detection*.**
  [arXiv:2211.06660](https://arxiv.org/abs/2211.06660). The dense/per-pixel variant, relevant if we
  ever score appearance coverage rather than a single pose.

### Coverage and data quality specifically for robot demonstrations

- **Belkhale, Cui, Sadigh — *Data Quality in Imitation Learning*.** NeurIPS 2023.
  [arXiv:2306.02437](https://arxiv.org/abs/2306.02437) ·
  [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2023/file/fe692980c5d9732cf153ce27947653a7-Paper-Conference.pdf).
  Decomposes data quality into **action consistency** and **state diversity**, and argues the
  *required* diversity is a function of policy mismatch and transition noise — i.e. **how much
  coverage you need is not a property of the dataset alone.** The most directly relevant paper to
  what we are doing.
- **Hejna et al. — *Robot Data Curation with Mutual Information Estimators*.**
  [arXiv:2502.08623](https://arxiv.org/abs/2502.08623). Curation by an information-theoretic
  criterion rather than a distance one.
- ***An Efficient Metric for Data Quality Measurement in Imitation Learning*.**
  [arXiv:2605.01544](https://arxiv.org/abs/2605.01544). Source of the **bin coverage ratio** — divide
  each dimension into equal-width bins, report the fraction of bins containing at least one sample.
  A cheap, interpretable alternative to 1-NN distance and worth computing alongside it.

### Why coverage is the right frame at all — the theory

- **Zhan et al. — *Offline RL with Realizability and Single-Policy Concentrability*.** COLT 2022.
  [arXiv:2202.04634](https://arxiv.org/abs/2202.04634) ·
  [PMLR](https://proceedings.mlr.press/v178/zhan22a/zhan22a.pdf).
  **Concentrability** is the formal version of "does the data cover what the good policy does". Where
  the coefficient is large, there are state-action pairs the optimal policy visits that barely appear
  in the data, and efficient learning is provably hard. This is the theory behind family A.
- **Chen & Jiang (2019)**, and the discussion in
  [arXiv:2111.10919](https://arxiv.org/abs/2111.10919) (*Fundamental Barriers for Value Function
  Approximation*): coverage plus realizability may **not** suffice for sample-efficient offline
  learning. Worth knowing before promising that filling a hole fixes a policy.
- **Offline Constrained RL under Partial Data Coverage.**
  [arXiv:2505.17506](https://arxiv.org/abs/2505.17506). Without the coverage assumption the problem
  reduces to an MDP restricted to the **support of the data** — which is the formal statement of why
  a policy cannot be expected to work outside it.

### Our own evidence, for cross-reference

- `docs/RETRAINING_DEFAULT_TAXONOMY.md` §5b — the three-level check the corpus query belongs to.
- `docs/DATA_AND_BENCHMARKS.md` §1.2 — the demo corpus measurements (no privileged state; 1.1–2.6 cm
  grasp SD; one instruction per task).
- LIBERO-Plus [arXiv:2510.13626](https://arxiv.org/abs/2510.13626) — *"merely learned the positional
  information of the target objects"*, the published claim our 5 cm spread explains.
- LIBERO-PRO [arXiv:2510.03827](https://arxiv.org/abs/2510.03827) — >90% → 0.0% under generalised
  settings, attributed to rote memorisation. The reason a same-distribution fine-tune would be hard
  to interpret.
