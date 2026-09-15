# LANDSCAPE — prior and concurrent work around "VLA Stress-Test with Failure Mining"

> **[UNSOURCED — do not cite]** the sim-to-real rank-correlation figure (Spearman 0.4–0.7) and the claim that *severity ordering transfers worst* were carried forward from an early survey and **no primary source has been located** (flagged 2026-09-15). The design decisions they motivated stand on their own reasoning; the numbers must not appear in a readout until sourced.

**Compiled:** 2026-09-11 · **Companion to:** `VLA Scenario Testing.md`, `PLAN.md`,
`experiments/phase0/PREDICTIONS.md`
**Method:** web search + primary-source fetch. Everything below carries a URL. Claims that
could not be checked against a primary source are marked **[unverified]** inline and
collected in §6.

**How to read this.** The document is deliberately unflattering. The brief asked for the
single most valuable thing a survey can say — *where we are redundant* — so redundancy is
stated first and bluntly in §1.8, and the parts of the project that survive the survey
intact are named there too. There are three of them.

---

## 0. Executive summary — the five things we should change or adopt

### E1. Replace the grid sweep with surrogate-guided boundary search — but keep a uniform arm

Our stress matrix (PLAN §8: ~1,200–2,400 episodes, "~20/cell screening, ~50 near the
boundary") is a hand-rolled version of a method the field has already formalised.
**FATE-VLA** ([arXiv:2606.02307](https://arxiv.org/abs/2606.02307),
[repo](https://github.com/pablovalle/FATE-VLA)) fits a surrogate over scene parameters and
steers sampling toward diverse, failure-prone regions — **+29.7% more failures at equal
budget**. **Liao et al.** ([arXiv:2607.14439](https://arxiv.org/abs/2607.14439)) do the same
on real hardware over 2,331 evaluations with an information-gain criterion, reporting 20–40%
trial savings and explicitly framing the output as "identification of failure-prone regions"
— our operating envelope, obtained more cheaply.

A logistic or GP surrogate over (perturbation magnitude → success) is a few hundred lines of
CPU code. It gives the manifest's `boundary` field a proper posterior interval instead of a
Wilson interval on whichever grid cell happened to straddle the transition, and it supplies
a principled stopping rule.

**The trap, which the source papers do not have to care about and we do:** active sampling
optimises for *finding* failures, which biases the discovered failure population. Our
manifest's `severity` is frequency-weighted. **Run a uniform-sampling arm for frequency
estimation and the adaptive arm for boundary estimation, and never mix them.** Getting this
wrong silently corrupts every severity score in the manifest.

### E2. Manifest rows should specify diversity axes, not demonstration counts

PLAN already resists inventing sample counts. The literature says that instinct is right and
gives us the citation to justify it. **Lin et al., "Data Scaling Laws in Imitation Learning
for Robotic Manipulation"** ([arXiv:2410.18647](https://arxiv.org/abs/2410.18647), ICLR 2025;
40,000+ demos, 15,000+ real rollouts) find that **generalization follows a power law in the
number of distinct environments and objects, not in raw demonstration count**, with sharp
diminishing returns past a per-environment threshold.

So a manifest row reading "collect 500 more demos in the failing yaw band" is, by this
result, close to the *worst* possible recommendation. The right row specifies coverage over
*distinct conditions*. Our schema's `coverage_required` field ("ranges, step sizes, counts of
conditions") already has the correct shape — this finding turns it from a stylistic
preference into an evidence-backed requirement, and it should be cited in the readout.

Reinforcing this from the opposite direction is the most alarming single data point in the
survey: an industry report
([Pebblous, Jul 2026](https://blog.pebblous.ai/report/robot-data-curation-closed-loop-gap/en/),
**[unverified — vendor blog, not peer-reviewed]**) reports adding 93 targeted "prescription"
demonstrations aimed at known weaknesses **dropped closed-loop success from 73% → 43% while
offline loss showed no change at all.** If even directionally true, it means (a) targeted
data can actively harm, and (b) **offline metrics are blind to it.** Our Phase 5 — fine-tune
and re-measure *closed-loop* on a frozen regression set — is exactly the procedure that
catches this. That is an argument for Phase 5 being non-negotiable, and for reporting
regression on *non-targeted* cells alongside improvement on targeted ones.

### E3. Add a discriminator: "is there a published non-data fix for this failure family?"

Phase 0 predicts (P1) that camera viewpoint is our most brittle axis. The field agrees — and
has already largely solved it **without collecting data**:

- **Feature Token Modulation** takes viewpoint success **48.5% → 87.1% with 4,000 parameters**;
  Feature Linear Adaptation reaches 90.8% with 4.7M, matching LoRA-scale fine-tuning at a
  fraction of the cost ([arXiv:2512.02902](https://arxiv.org/abs/2512.02902)).
- **GS-VLA** canonicalises the viewpoint in front of a **frozen** policy — zero policy changes
  ([arXiv:2608.19066](https://arxiv.org/html/2608.19066)).
- **The Moving Eye** ([arXiv:2607.02322](https://arxiv.org/pdf/2607.02322)) takes the data
  route but warns that **naively increasing viewpoint diversity induces shortcut learning.**

If our top manifest row is "camera viewpoint" and we classify it `fixability: data`, a
competent buyer can produce these three papers and ask why we are selling collection. PLAN
§7b.1 already anticipates exactly this failure mode of a data vendor; this survey supplies
the ammunition. **Make the literature check a formal, mandatory discriminator in §7c — the
cheapest of the seven, and the one that most protects our credibility.**

The opportunity on the other side is larger than the risk. Phase 5 could compare **three**
remediations on one frozen regression set — targeted data, re-rendered augmentation, and a
frozen-policy canonicalisation wrapper — and report cost per point of recovered success.
Nothing in this survey does that. It converts `fixability` from a judgement call into a
measurement, and it is the most defensible novel result available to us.

### E4. Adopt the evaluation statistics the good labs use, and make the schema PPI-ready

Two moves, both cheap:

**Now.** Follow **Kress-Gazit et al., "Robot Learning as an Empirical Science: Best Practices
for Policy Evaluation"** (TRI, [arXiv:2409.09491](https://arxiv.org/abs/2409.09491)) and the
LBM paper's practice ([*Science Robotics*](https://www.science.org/doi/10.1126/scirobotics.aea6201)):
paired statistical tests, Bayesian credible regions, multiple-comparison correction. We
control the seed, so **paired** tests across perturbation conditions are free variance
reduction — which directly relieves the counterfactual-probe budget problem in §1.4. Context
worth quoting in the readout: an audit in **PhAIL** ([arXiv:2605.29710](https://arxiv.org/pdf/2605.29710))
finds the modal per-condition N across 13 recent real-robot VLA papers is **10–20, and none
of the 13 reported confidence intervals or paired tests** **[unverified — read via search
summary, not the paper]**. Our promised Wilson intervals already beat the modal published
paper; say so.

**Later, for free if we plan now.** **SureSim**
([arXiv:2510.04354](https://arxiv.org/abs/2510.04354),
[site](https://suresim-robot-eval.github.io/)) uses prediction-powered inference to convert
many imperfect-sim evaluations plus a *small* number of paired real trials into
**statistically valid confidence intervals on real-world performance**. We have no hardware,
but if per-cell sim results are recorded in a PPI-consumable form, then "give us 30 real
trials and we will upgrade this entire sim campaign into real-world CIs" becomes a
commercial offer. That is a schema decision costing nothing today and expensive to retrofit.

### E5. Validate the taxonomy against a discovered one, not only against κ

Our taxonomy gate is κ ≥ 0.5 on 30 double-labelled traces. That is a weak gate and it only
measures whether two humans agree — not whether the taxonomy carves the failure space at its
joints. **Gupta, Ciftci & Bansal** ([arXiv:2506.06570](https://arxiv.org/abs/2506.06570),
[site](https://mllm-failure-clustering.github.io/)) discover failure taxonomies *without* a
predefined label set, by clustering VLM-generated structured failure explanations in semantic
reasoning space. The general-ML ancestor is **Domino**
([arXiv:2203.14960](https://arxiv.org/abs/2203.14960),
[code](https://github.com/HazyResearch/domino)), which is the canonical slice-discovery method
and is the closest methodological cousin to "failure mining" anywhere in ML.

**Run the discovered taxonomy as a second arm over the same traces and publish the confusion
matrix against our fixed 7 families.** It is a clustering job, not a rollout job, so it costs
almost nothing. If our families recover their clusters, that is a far stronger validation
than κ. If they don't, we find out what we're missing while it is still cheap to change.

Two specific taxonomy problems the survey exposes, both fixable now:
- **"Language grounding" may be an empty family on LIBERO.** LIBERO-Plus reports models are
  largely insensitive to language and "tend to ignore language instructions completely"
  ([arXiv:2510.13626](https://arxiv.org/abs/2510.13626)). Either drop the family for this
  benchmark or reframe it as a measured absence — which is itself a reportable finding.
- **"Distribution shift" is not a sibling of the other six families.** It overlaps all of
  them, which will depress κ for reasons that have nothing to do with reviewer skill.
  Consider the two-level cut used by the SO-101 study
  ([arXiv:2606.08881](https://arxiv.org/abs/2606.08881)) — semantic vs. execution level —
  or Sentinel's erratic vs. task-progression split
  ([arXiv:2410.04640](https://arxiv.org/abs/2410.04640)), with distribution shift as an
  orthogonal tag.

### Bottom line on novelty

Three things survive the survey as genuinely ours:
1. **The costed, evidence-backed Data Gap Manifest as an artifact** with severity, fixability,
   discriminators and validation status per row. The *ideas* exist; the artifact does not.
2. **Counterfactual attribution of robot policy failures to environment factors.** Formalised
   for LLM agents ([CAR, arXiv:2606.08275](https://arxiv.org/abs/2606.08275)), essentially
   unexploited in robot policy evaluation — and our randomised-treatment setting is
   *methodologically stronger* than theirs. We currently undersell this.
3. **Failure cost weighting** (PLAN §7b.3). No perturbation benchmark found in this survey
   weights failures by consequence. PLAN calls it the proposal's largest omission; the survey
   agrees and upgrades it to a differentiator.

Everything else has prior art, and the honest positioning is *integration and delivery*, not
research novelty.

---
## 1. Who else is doing this, and how do they differ?

The honest headline: **almost every individual component of this project exists in the
literature already.** Perturbation benchmarks exist. Failure taxonomies exist. Semantic
failure clustering exists. Counterfactual attribution exists. Influence-function-based
data curation exists. Active failure-region search exists. What does *not* exist, as far
as this survey could find, is a single pipeline that runs all of them end-to-end and
terminates in a costed, evidence-backed data plan that is then validated by retraining.
That is a real gap, but it is an *integration* gap, not a research gap — and the project
should be positioned and defended on exactly those terms.

Below, each cluster of prior work, and specifically **what they do that we don't.**

---

### 1.1 Systematic perturbation benchmarks (the "stress" half)

These are the direct substrates. We already plan to use LIBERO-plus; the others are
either alternatives or sources of design ideas we are missing.

**LIBERO-Plus** — Fei, Wang et al.
[arXiv:2510.13626](https://arxiv.org/abs/2510.13626) ·
[repo](https://github.com/sylvestf/LIBERO-plus) ·
[CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/38735) ·
[OpenReview](https://openreview.net/forum?id=6mEfYoMRpF)

7 perturbation dimensions, 21 sub-dimensions, 10,030 task instances, L1–L5 difficulty.
Headline: success drops 95% → <30% under camera-viewpoint and initial-state perturbation;
models are nearly *insensitive* to language, and the paper argues models largely **ignore
the instruction altogether**. Note the version history: v3 (Dec 2025) and a CVPR 2026
camera-ready retitled *"LIBERO-Plus: A Progressive Robustness Benchmark"*
([CVF open access](https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html)).
**Check the camera-ready, not the v1 arXiv** — our proposal cites v1 numbers and the
difficulty-level definitions may have moved.

*What they do that we don't:* nothing on the diagnosis side — which is precisely our
claimed contribution, and that remains true. But their "models ignore language" finding
is a **direct threat to one of our seven taxonomy families**. If language grounding is not
used by the policy at all, a "language grounding" failure family will be empty or
meaningless on LIBERO. We should either drop it for this benchmark or reframe it as a
*measured absence* ("the policy is instruction-insensitive; this is itself a finding").

**THE COLOSSEUM** — Pumacay, Singh, Duan, Krishna, Thomason, Fox (RSS 2024)
[arXiv:2402.08191](https://arxiv.org/abs/2402.08191) ·
[RSS paper](https://www.roboticsproceedings.org/rss20/p133.pdf) ·
[OpenReview](https://openreview.net/forum?id=aiSZqS5wkB)

20 tasks × 14 perturbation axes on RLBench. 30–50% degradation per axis; ≥75% when
composed. Most damaging factors: distractor count, target object colour, lighting.

*What they do that we don't — three things, all important:*
1. **Sim–real perturbation correlation, measured: R² = 0.614.** This is the single most
   useful number in the survey for us. We are a sim-only project selling conclusions about
   real deployment. We should cite this explicitly as the *quantified* limit of that claim,
   rather than hand-waving "sim is a proxy". It is also a warning: R²=0.614 means ~40% of
   real variance is unexplained by sim perturbation results.
2. **Composed perturbations.** We plan one-axis-at-a-time sweeps plus one-axis-reverting
   counterfactuals. COLOSSEUM shows interaction effects dominate (≥75% vs 30–50%). A purely
   univariate stress matrix will *systematically understate* the failure surface, and our
   counterfactual probe design (revert one dimension) implicitly assumes additivity.
   **We should run at least one 2-factor cell to measure whether additivity holds**, and
   report the interaction term. If effects are strongly super-additive, single-axis
   boundaries in the manifest are optimistic and must be labelled as such.
3. **3D-printable objects released** so the sim perturbations can be physically replicated.
   A cheap credibility move we cannot match, but worth noting as the bar for "real".

**VLATest** — Wang, Duan, Chen, Zhou, Ray, Jin, Chen, Johnson-Roberson (FSE 2025)
[arXiv:2409.12894](https://arxiv.org/abs/2409.12894) ·
[ACM](https://dl.acm.org/doi/10.1145/3729343)

**This is the closest prior work to our Phase 2 and we did not know about it.** A fuzzing
framework that generates test scenes for VLA manipulation, varying confounding-object
count, lighting, camera pose, unseen objects, and instruction mutations, evaluated across
seven VLA models. Conclusion: current VLAs lack deployment robustness.

*What they do that we don't:* they come from the **software-engineering testing** community
and bring its vocabulary — fuzzing, test oracles, test adequacy, replication packages,
mutation operators. That framing is more mature than ours in one specific way: it treats
scene generation as a *search* problem rather than a grid sweep (see FATE-VLA below).
It is also a reminder that an SE-venue reviewer would ask us "what is your test oracle,
and is it sound?" — our answer is the Tier-1 deterministic detectors, which is a good
answer, but we should use the word.

**RobustVLA** [arXiv:2510.00037](https://arxiv.org/abs/2510.00037) — 17 perturbations across
four modalities. *Unverified in depth* — I read only the abstract. Worth 20 minutes because
it may partition the perturbation space differently from LIBERO-plus (by *modality* rather
than by *scene factor*), which is a taxonomy alternative for our axis design.

---

### 1.2 Active / search-based failure discovery — the biggest methodological gap

This is where our design is weakest relative to the state of the art. **We plan a grid
sweep. The field has moved to adaptive search.**

**FATE-VLA: Failure-Aware Test Generation for VLA Models** — Kanwal, Valle, Ali, Arrieta
(June 2026) [arXiv:2606.02307](https://arxiv.org/abs/2606.02307) ·
[repo](https://github.com/pablovalle/FATE-VLA)

Frames VLA evaluation explicitly as **an active failure-discovery problem**. Combines
diversity-driven adaptive random testing (FSCS-ART) with an iteratively-retrained surrogate
(decision tree / random forest) over scene parameters, then steers sampling toward
predicted-failure-prone-but-diverse regions. Finds up to **29.7% more failures than
baselines** for the same budget; drove GR00T-N1.6 from 64.4% → 34.7%. Tested on
SimplerEnv "pick up" with OpenVLA, π0, GR00T-N1.6, EO-1. Full replication package public.

**Active Real-World Factor-Based Evaluation for Generalist Robot Policies** — Liao, Cui,
Desingh, Deshwal (July 2026) [arXiv:2607.14439](https://arxiv.org/abs/2607.14439)

Same idea, real robots. Probabilistic surrogate over a structured factor space (object
poses, camera viewpoints), adaptive configuration selection maximising information gain.
**2,331 real-world evaluations.** Reports 20–40% trial savings vs random, and explicitly
frames the output as "systematic identification of failure-prone regions" — i.e. our
"operating envelope", obtained more cheaply.

**Efficient Evaluation of Multi-Task Robot Policies With Active Experiment Selection**
[arXiv:2502.09829](https://arxiv.org/abs/2502.09829) — active experiment selection for
multi-task policy capability estimation.

*What they do that we should adopt — this is recommendation #1 of the whole survey:*
Our PLAN §8 budget is "~1,200 episodes for targeted stress, ~2,400 for the full grid".
A surrogate-guided sampler over the perturbation-parameter space would locate the **failure
boundary** — which is literally what our manifest's `boundary` field records — in a fraction
of those episodes, and would place samples *densely at the boundary* where the CI matters
and sparsely in the saturated regions where it does not. Our current plan ("~20 episodes
per cell for screening, ~50 near the boundary") is a hand-rolled, coarser version of this.
Fitting a simple GP or logistic surrogate over (perturbation magnitude → success) is a
few hundred lines, runs on CPU, and converts a grid sweep into a **boundary-estimation**
procedure with a principled stopping rule. It also gives the boundary a proper posterior
CI rather than a Wilson interval on the nearest grid cell.

**Caveat we should state:** these methods optimise for *finding failures*, which biases
the discovered failure population. Our manifest claims frequency-weighted severity. If we
adopt active sampling we must **keep a separate uniform-sample estimate for frequency**,
or reweight by the sampling density. Getting this wrong would silently corrupt every
severity score.

---

### 1.3 Failure taxonomy discovery and semantic failure clustering (the "mine" half)

**Unsupervised Discovery of Failure Taxonomies from Deployment Logs** — Gupta, Ciftci,
Bansal (USC) [arXiv:2506.06570](https://arxiv.org/abs/2506.06570) ·
[project site](https://mllm-failure-clustering.github.io/)

**This is the most direct prior art for our L3 mining layer, and we should read it before
building anything else.** Pipeline: VLM reasoning produces a *structured failure
explanation* per episode → embed the explanations → cluster in that **semantic reasoning
space** (not in pixel or trajectory space) → recover an interpretable failure taxonomy
*without a predefined label set*. Validated across robotic manipulation, indoor navigation,
and autonomous driving. Explicitly states the downstream uses are (a) targeted data
collection for offline policy refinement and (b) runtime failure monitoring — i.e. both our
§7b.4 and our Data Gap Manifest. Note the retitling history (v1 "Enhancing Robot Safety via
MLLM-Based Semantic Interpretation of Failure Data" → v2 "From Perception Logs to Failure
Modes" → v4 current, July 2026); cite v4.

*What they do that we don't, and should consider:*
- **Cluster in explanation space, not feature space.** Our PLAN clusters on "trajectory and
  visual features" with an LLM only for "semantic residue". Theirs inverts this: the LLM
  produces the representation, and clustering is downstream. Their argument is that
  trajectory-space clusters are not *semantically coherent* — two failures with identical
  trajectories can have different causes. That is a real objection to our design.
- **Taxonomy is discovered, not imposed.** We fix a 7-family taxonomy up front (from the
  proposal, itself partly inherited). Ours is auditable and deterministic; theirs is
  data-driven and may find families we didn't anticipate. **The strongest move is to run
  both and report the confusion matrix between our fixed taxonomy and their discovered one.**
  If our 7 families recover their clusters, that validates the taxonomy far more convincingly
  than a κ on 30 double-labelled traces. If they don't, we learn what we're missing.
  This is cheap — it's a clustering run over traces we already have — and it turns our
  weakest gate (κ ≥ 0.5) into a much stronger one.

**RoboFAC: A Comprehensive Framework for Robotic Failure Analysis and Correction**
[arXiv:2505.12224](https://arxiv.org/pdf/2505.12224) — failure analysis *and* correction,
with a dataset and a fine-tuned model for failure reasoning. *Partially verified* (abstract
and PDF skim only).

**AHA: A VLM for Detecting and Reasoning Over Failures in Robotic Manipulation** — NVIDIA /
UW / MIT [arXiv:2410.00371](https://arxiv.org/abs/2410.00371) ·
[repo](https://github.com/NVlabs/AHA) · [site](https://aha-vlm.github.io/)

*What they do that we should seriously consider adopting:* **FailGen** — a procedural
pipeline that generates failure trajectories by *systematically perturbing successful
demonstrations*. This is the first large-scale robot failure dataset and it was built
exactly the way our PLAN §11 open-question 4 asks about ("is remediation data generated by
scripted demonstration in sim, or by re-rendering existing demos under perturbation?").
**FailGen is the answer to that open question and it is open source.** It is also directly
reusable for the *recovery* family that our PLAN §4 admits is unvalidated: FailGen produces
the failed-attempt half of the failed-attempt-plus-recovery demonstrations our taxonomy
asks for. AHA itself (LLaVA-1.5-13B fine-tune) beats GPT-4o ICL by 10.3% on failure
reasoning — relevant if we use an LLM judge at Tier 3: **there is a specialised open model
for this task and using a generic LLM is a weaker choice we'd have to defend.**

**Benchmarking VLA Models on SO-101: Failure and Recovery Analysis** — Yu, Qiu (June 2026)
[arXiv:2606.08881](https://arxiv.org/abs/2606.08881)

Real hardware, low-cost SO-101 arm, fine-tunes and evaluates **π0.5, SmolVLA, Wall-X and
ACT** with a structured failure taxonomy, semantic- vs execution-level failure decomposition,
and recovery-aware metrics. Finding: after task-specific fine-tuning, **most failures are
execution-level (imprecise grasp pose, insufficient trajectory refinement), not high-level
task misunderstanding.**

*Why this one matters to us specifically:* it evaluates **SmolVLA**, our exact policy, with
a failure taxonomy, on real hardware. Its finding (execution-level dominates) is a *prior*
we can test against in sim, and if we reproduce it that is a sim–real corroboration we get
for free. It also splits the taxonomy along a **semantic / execution** axis, which is a
cleaner top-level cut than our flat 7 families and may be worth adopting as a *two-level*
taxonomy (semantic{grounding, spatial, planning} / execution{manipulation, recovery} /
distribution-shift as an orthogonal tag rather than a sibling family). Our current 7
families are not mutually exclusive — "distribution shift" overlaps every other family —
and that is a labelling-reliability problem that will show up in our κ.

**VISOR: A VLM-based Test Oracle for Testing Robots** — Valle et al.
[arXiv:2605.10408](https://arxiv.org/html/2605.10408) ·
[ACM AIware](https://doi.org/10.1145/3805760.3814910)
**MANGO: Automated Multi-Agent Test Oracle Generation for VLA Models**
[arXiv:2606.24815](https://arxiv.org/html/2606.24815)

Both attack the *test oracle* problem: how do you decide, automatically and soundly, that a
rollout failed and why, without hand-written symbolic predicates. VISOR's explicit motivation
is that "existing testing approaches rely on manually constructed symbolic test oracles" —
which is exactly our Tier-1 design. **We should read VISOR as the steelman of the opposing
position and have an answer.** Our answer is reproducibility (PLAN §7's "why not LLM-primary"),
and it is a good answer, but we should cite VISOR when making it rather than arguing against
a strawman.

---

### 1.4 Counterfactual / causal attribution

Our Tier-2 counterfactual probe is the mechanism we lean on hardest for causal claims. The
closest formalisation is **not** in robotics:

**Causal Agent Replay: Counterfactual Attribution for LLM-Agent Failures** (June 2026)
[arXiv:2606.08275](https://arxiv.org/abs/2606.08275)

Models an agent run as a structural causal model, applies **do-operations** to individual
steps, and re-executes forward under the same stochastic policy, measuring the shift in the
*outcome distribution*. Reports that **simple attribution heuristics are unreliable —
state-of-the-art step-level attribution accuracy around 14%.**

**CausalFlow: Causal Attribution and Counterfactual Repair for LLM Agent Failures**
[arXiv:2605.25338](https://arxiv.org/pdf/2605.25338)

*What we should take from these:*
1. **The vocabulary.** Our "re-run the same seed reverting one perturbation dimension" *is*
   a do-operation on an environment variable. Saying so, and writing the estimand down
   (`E[success | do(yaw = 0)] − E[success | do(yaw = 15°)]`), makes the claim precise and
   makes its assumptions auditable. It costs nothing and materially raises the credibility
   of the manifest's `trigger` field.
2. **The warning.** CAR's 14% figure is about *step*-level attribution in a long causal
   chain — harder than our problem, which is attribution to an *exogenous* environment
   factor we ourselves set. Ours is genuinely easier: we randomise the treatment, so it's
   closer to an RCT than to observational attribution. **That is a real advantage and we
   should say it explicitly** — it is the strongest methodological claim in the project and
   the proposal currently undersells it.
3. **Distributional, not per-episode.** CAR measures shifts in outcome *distributions* under
   re-execution, matching our G5 (compare distributions over N seeds, not bit-exact replays).
   Good — our design is already right here. But note the implication: a counterfactual probe
   needs enough seeds to beat `reproducibility_floor()`, and **PLAN §8 budgets only ~600
   episodes for all probes.** If the floor is wide (contact chaos in MuJoCo is not gentle),
   600 episodes across multiple axes and cells may not resolve a single effect. Measure the
   floor first (already planned, Wk 3 day 1) and **re-derive the probe budget from it**
   rather than from a guess.

---

### 1.5 Independent / third-party and statistically-honest evaluation

This is where the proposal's commercial pitch ("independent evaluation of somebody's policy")
has the most established competition, and where our *methods* are weakest.

**RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies** — Atreya,
Pertsch, Lee, Kim et al. (CoRL 2025) [arXiv:2506.18123](https://arxiv.org/abs/2506.18123) ·
[PMLR](https://proceedings.mlr.press/v305/atreya25a.html)

Crowd-sourced, decentralised, **double-blind pairwise** comparisons on DROID across seven
academic sites, 600+ real episodes, seven policies. Argues pairwise crowd evaluation ranks
policies more accurately than centralised fixed-task evaluation.

*What they do that we don't:* **double-blind pairwise comparison** and an explicit argument
that *absolute* success rates are the wrong unit. Their diversity comes from evaluator
heterogeneity; ours comes from parameterised perturbation. These are complementary, and the
honest framing is that RoboArena answers "which policy is better" while we answer "where
does *this* policy break" — a different and, for a single-client engagement, more useful
question. **Say that explicitly in the readout**, because "we built an eval harness" invites
the response "RoboArena already exists".

**AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real
World** — Zhou, Atreya, Tan, Pertsch, Levine (CoRL 2025)
[arXiv:2503.24278](https://arxiv.org/abs/2503.24278)

Automatic success detection + automatic scene reset, 24/7 unattended real evaluation, >99%
reduction in human supervision, public evaluation cells on BridgeData/WidowX. Reports that
its results **match human evaluation more closely than SIMPLER (photorealistic sim) or
offline validation MSE do.** That is a direct, quantified criticism of sim-based evaluation
as a proxy — including ours. We should cite it in the limitations rather than let a reader
find it.

**Reliable and Scalable Robot Policy Evaluation with Imperfect Simulators (SureSim)** —
Badithela, Snyder, Zha, Mikhail, O'Kelly, Dixit, Majumdar (Oct 2025)
[arXiv:2510.04354](https://arxiv.org/abs/2510.04354) ·
[site](https://suresim-robot-eval.github.io/)

**The single most adoptable idea in this section.** Uses **prediction-powered inference**:
a large number of imperfect-simulator evaluations plus a *small* number of paired real
evaluations, combined with non-asymptotic mean estimation, yields **statistically valid
confidence intervals on real-world performance**. ~20–25% reduction in hardware trials at
equal guarantees.

*Why this matters to us:* it is the principled answer to "your numbers are sim numbers".
We cannot run it now (no hardware), but **the manifest schema should be designed so that
adding a handful of paired real trials later upgrades every sim CI into a real CI** — i.e.
record the per-cell sim success and the cell identity in a form PPI can consume. That is a
schema decision, free today, expensive to retrofit. It is also a strong commercial story:
"give us 30 real trials and we will convert the whole sim campaign into real-world
confidence intervals."

**A careful examination of large behavior models for multitask dexterous manipulation** —
Toyota Research Institute, *Science Robotics*
[DOI](https://www.science.org/doi/10.1126/scirobotics.aea6201) ·
[TRI blog: "Statistical Thinking for Robot Policy Evaluation"](https://medium.com/toyotaresearch/statistical-thinking-for-robot-policy-evaluation-from-rigorous-a-b-testing-to-effective-0ae886fbd68d)

**This is the methodological gold standard and should be our template for the readout.**
Blind A/B evaluation, N=50 real / 200 sim per condition, Bayesian posterior credible
regions, paired Barnard's exact and Welch's t-tests, Bonferroni-corrected significance
grouping. The blog is the readable version and is on the read-first list.

**Beyond Binary Success: Sample-Efficient and Statistically Rigorous Robot Policy Comparison**
[arXiv:2603.13616](https://arxiv.org/html/2603.13616)
**PhAIL: A Real-Robot VLA Benchmark and Distributional Methodology**
[arXiv:2605.29710](https://arxiv.org/pdf/2605.29710) — contains the useful audit that the
**modal per-condition N across 13 recent real-robot VLA papers (2023–2025) is 10–20, and
none of the 13 reported confidence intervals or paired tests.**

*What we should adopt:* that PhAIL audit number is a gift. Our proposal already promises
Wilson CIs, which puts us above the modal published paper. **Say so, with the citation.**
It is the cheapest credibility we will ever buy, and it reframes "we only ran 20 episodes
per screening cell" from an apology into a comparison we win. Separately: adopt **paired**
tests (same seeds across conditions) rather than unpaired ones — we control the seed, so we
get the variance reduction for free, and it directly shrinks the counterfactual-probe budget
problem raised in §1.4.

**XPolicyLab: A Unified Standard and Open Ecosystem for Robot Policy Evaluation and
Deployment** [arXiv:2608.09892](https://arxiv.org/html/2608.09892) — Aug 2026, *abstract only,
unverified.* Appears to standardise the policy/environment interface across heterogeneous
model architectures and simulators — i.e. our L0 "canonical rollout contract" and G6
Protocols. **Worth 30 minutes before we freeze our schema**, purely to avoid inventing a
private standard when a public one is emerging.

---

### 1.6 Runtime monitoring (our §7b.4 "second product" — it is a crowded field)

PLAN §7b.4 proposes a runtime failure monitor as a follow-on product. It is worth knowing
that this is an established and competitive area, not open ground:

- **Sentinel / Unpacking Failure Modes of Generative Policies** — Agia, Sinha, Yang, Cao,
  Antonova, Pavone, Bohg (CoRL 2024) [arXiv:2410.04640](https://arxiv.org/abs/2410.04640) ·
  [PMLR](https://proceedings.mlr.press/v270/agia25a.html). Splits monitoring into *erratic*
  failures (statistical temporal action consistency, "STAC") and *task-progression* failures
  (VLM-detected). **The erratic/progression split is a better-motivated two-way cut than
  parts of our 7-family taxonomy and is computed from signals we already record.**
- **FIPER: Failure Prediction at Runtime for Generative Robot Policies** (NeurIPS 2025)
  [arXiv:2510.09459](https://arxiv.org/abs/2510.09459) ·
  [proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/file/0b7cb3b8cc44e652761245537027db44-Paper-Conference.pdf).
  Needs **no failure data**: OOD detection via random network distillation in the policy
  embedding space + action-chunk entropy, calibrated by **conformal prediction** on a small
  set of successful rollouts.
- **Hide-and-Seek in Trajectories: Discovering Failure Signals for VLA Runtime Monitoring**
  [arXiv:2605.30834](https://arxiv.org/pdf/2605.30834).
- **Deployment-Time Reliability of Learned Robot Policies** — Christopher Agia, Stanford PhD
  thesis, 2026 [arXiv:2603.11400](https://arxiv.org/abs/2603.11400). 182 pages tying together
  runtime monitoring, influence-function interpretability **and dataset curation**, and policy
  coordination. **This thesis is the closest single document to our entire project's
  intellectual programme and is the best one-stop read in this survey.**

*What we should adopt now, cheaply:* conformal prediction for calibration. Our manifest
promises boundaries "with a confidence interval". Conformal gives a *distribution-free,
finite-sample* guarantee, which is a stronger and more defensible statement than a Wilson
interval on a binomial, and it is a small amount of code. FIPER's action-chunk entropy is
also computable from traces we already store, at zero extra rollout cost, and would give
every manifest row a per-episode uncertainty signal — useful evidence for the
`fixability: architecture` class (high entropy *inside* the training distribution suggests
capacity, not coverage).

---

### 1.7 Where camera viewpoint — our headline prediction P1 — has already been worked

This deserves its own subsection because Phase 0's P1 ("camera viewpoint is the most brittle
axis") is our flagship falsifiable claim, and the field has essentially already confirmed it
*and* moved on to fixing it. That has a direct consequence for our `fixability` classes.

- **VLA Models Are More Generalizable Than You Think: Revisiting Physical and Spatial
  Modeling** [arXiv:2512.02902](https://arxiv.org/abs/2512.02902). Argues viewpoint
  brittleness comes from **spatial-modelling misalignment, not physical modelling**, and
  that a *one-shot* adaptation fixes most of it: Feature Token Modulation takes viewpoint
  success **48.5% → 87.1% with 4K parameters**; Feature Linear Adaptation reaches 90.8% with
  4.7M — **matching LoRA-scale fine-tuning at far lower cost.**
- **GS-VLA: Plug-and-Play Viewpoint Canonicalization for Frozen VLA Policies via Gaussian
  Splatting** [arXiv:2608.19066](https://arxiv.org/html/2608.19066). Fixes viewpoint
  brittleness **without touching the policy at all** — canonicalise the view first.
- **The Moving Eye: Enhancing VLA Spatial Generalization via Hybrid Dynamic Data Collection**
  [arXiv:2607.02322](https://arxiv.org/pdf/2607.02322). The data-side answer. Important
  caveat reported: **naively increasing viewpoint diversity can induce shortcut learning**,
  so they explicitly reshape the view distribution rather than just adding more views.
- **Don't Blind Your VLA: Aligning Visual Representations for OOD Generalization**
  [arXiv:2510.25616](https://arxiv.org/pdf/2510.25616).

*This is the most commercially dangerous finding in the survey for us.* If our top manifest
row is "camera viewpoint" — which Phase 0 predicts it will be — then a competent buyer can
reply: *"the literature says this is fixable with a 4K-parameter adapter or a free
canonicalisation wrapper; why are you selling me data collection?"* Three consequences:

1. **PLAN §7b.1's `fixability` field is vindicated and becomes the most important field in
   the schema.** A viewpoint row should very likely be classified `architecture` (or
   `augmentation`), **not** `data`. Getting this right is the difference between a credible
   consultancy and a vendor talking its own book — which §7b.1 already worries about.
2. **Add a fifth discriminator to §7c: "is there a published non-data fix for this failure
   family?"** It is free, it is a literature check, and it should gate every `data`
   classification.
3. **This is also an opportunity.** Our Phase-5 validation could compare *three* remediations
   on the same regression set — targeted data, re-rendered augmentation, and a frozen-policy
   canonicalisation wrapper — and report which is cheapest per point of recovered success.
   That is a far stronger deliverable than "we fine-tuned and it got better", it directly
   populates `fixability` with evidence rather than judgement, and it is the thing no
   existing paper in this survey does.

---

### 1.8 Verdict on redundancy

Plainly stated, since this is what was asked for:

| Our component | Redundant with | Status |
|---|---|---|
| Perturbation generation | LIBERO-Plus, COLOSSEUM, VLATest, RobustVLA | **Fully redundant** — and we already planned to consume, not build, this. Correct call. |
| Grid stress sweep | FATE-VLA, Liao et al. 2607.14439 | **Redundant and worse.** Adaptive search dominates grid sweep. Adopt a surrogate. |
| Failure clustering into a taxonomy | Gupta/Ciftci/Bansal 2506.06570; RoboFAC; AHA | **Substantially redundant.** Our differentiator is determinism/auditability, not novelty. Defend it on that basis or adopt theirs as a comparison arm. |
| Failure→data-collection recommendation | 2506.06570 states it as a use case; CUPID does it rigorously from the training side | **Partially redundant in concept, not in execution.** Nobody has published a costed, evidence-backed *manifest artifact* with severity, fixability and validation status per row. This is our real contribution. |
| Counterfactual attribution of failures to environment factors | CAR / CausalFlow (LLM agents, not robots) | **Not redundant in robotics.** Genuinely underexploited; and our randomised-treatment setting is *stronger* than theirs. Our best methodological claim. |
| Validate by retraining and re-measuring | The Moving Eye (viewpoint data); LIBERO-plus does *not* do this | **Not redundant, and rare.** Keep this. It is the part most likely to be cut and the part that most distinguishes us. |
| Runtime monitor (§7b.4) | Sentinel, FIPER, Hide-and-Seek, Agia thesis | **Fully redundant as a research idea.** Only viable as productisation of existing methods. Do not pitch it as novel. |
| Operating envelope spec (§7b.2) | Liao et al. call it "failure-prone regions"; framing is common | Redundant as an idea, **not** as a deliverable. The *artifact* is the value. |
| Failure cost weighting (§7b.3) | Nothing found | **Genuinely novel in this literature.** PLAN calls it "the largest omission in the proposal" and the survey agrees: no perturbation benchmark found weights failures by consequence. Push this harder. |

---

## 2. Data-centric robotics and data curation — the "Data Gap Manifest" half

This was the suspected location of the real prior art, and the suspicion was correct.
There is a substantial and fast-moving literature on *which robot data to keep*. There is
much less on *which robot data to collect next*, and almost nothing that closes the full
loop. The distinction matters and should be the spine of how we position the manifest.

**The framing that clarifies everything here:** existing methods are **subtractive** — they
score, filter, prune, reweight or retrieve from a pool that already exists. The Data Gap
Manifest is **generative** — it specifies conditions that are *not* in any pool and must be
created. That is a real difference, and it is the one sentence to have ready when someone
says "isn't this just CUPID?".

### 2.1 The closest prior art — read these three first

**So You Think You Can Scale Up Autonomous Robot Data Collection?** — Mirchandani, Belkhale,
Hejna, Choi, Islam, Sadigh (Stanford, CoRL 2024)
[arXiv:2411.01813](https://arxiv.org/abs/2411.01813)

**This is the strongest empirical precedent for our entire thesis and we should cite it in
the readout's first page.** Across 7 sim and real tasks at multiple data scales, they show
that **human data collected at states sampled from *failed autonomous rollouts* helps far
more than equivalent amounts of random human data, random autonomous data, or targeted
autonomous data.** That is, in one sentence: *failure-targeted collection beats volume.* It
is the premise our manifest rests on, independently validated, by a credible group, before
we started.

*What they do that we don't:* they identify *where* (which states) to collect, per task. They
do not produce a taxonomy-level prioritisation across failure *categories*, do not cost it,
and do not produce a portable artifact. That gap is ours. But note the direction of the
asymmetry — **they have the empirical validation and we have the artifact.** We should lean
on their result rather than re-deriving it, and spend our episodes on the parts they skipped.

**CUPID: Curating Data your Robot Loves with Influence Functions** — Agia, Sinha, Yang,
Antonova, Pavone, Nishimura, Itkina, Bohg (Stanford / TRI, CoRL 2025)
[arXiv:2506.19121](https://arxiv.org/abs/2506.19121) ·
[repo](https://github.com/agiachris/cupid) · [site](https://cupid-curation.github.io/) ·
[PMLR](https://proceedings.mlr.press/v305/agia25a.html)

An influence-function formulation for imitation learning that estimates the causal effect of
each *training demonstration* on the policy's expected return, from a set of evaluation
rollouts. Ranks and selects demonstrations by impact on **closed-loop** success. Reaches SOTA
diffusion-policy performance on RoboMimic with **<33% of the training data**, and is used on
hardware to debug spurious correlations.

*What they do that we don't, and it is significant:* CUPID attributes failures **backwards
into the training set**. We attribute failures backwards into the **environment**
(counterfactual probes on perturbation dimensions). These are complementary halves of the
same question and CUPID's half is the more rigorous. Two consequences:

1. **CUPID is the natural downstream consumer of our manifest**, and saying so strengthens
   both. We say *what conditions to collect*; CUPID checks whether the data actually
   collected is the data that helps. That is a coherent joint pitch and it is honest.
2. **Our §7c discriminator #2 ("supply-side density — is the failing region represented in
   training?") is a crude proxy for what CUPID measures properly.** Our Phase 0 histogram
   answers "is this region covered?"; CUPID answers "do the demos covering it actually help?"
   Those can differ. We should state that limitation rather than let it be found.

Note also the rest of that group's output: **Sentinel** ([arXiv:2410.04640](https://arxiv.org/abs/2410.04640)),
and the umbrella thesis **Deployment-Time Reliability of Learned Robot Policies**
([arXiv:2603.11400](https://arxiv.org/abs/2603.11400)), which stitches runtime monitoring,
influence-based interpretability, dataset curation and policy coordination into one
programme. **That thesis is the closest single document to our project's intellectual
agenda and is the single best read in this entire survey.**

**Demo-SCORE: Curating Demonstrations using Online Experience** — Chen, Lessing, Liu, Finn
(Stanford) [arXiv:2503.03707](https://arxiv.org/abs/2503.03707)

Trains a classifier on **closed-loop rollout outcomes** to score which demonstrations in a
heterogeneous dataset are reliable, filters, and retrains. Reports **+15–35 percentage points**
absolute success over unfiltered training.

*Overlap and difference:* it closes measure → filter → retrain, but only *removes* data. It
never specifies new data. Together with CUPID and the Pebblous report (§2.4), these three are
the works a knowledgeable reviewer will name at us, and the answer in each case is the same:
subtractive vs. generative.

### 2.2 Demonstration quality metrics — vocabulary we should be using

**Data Quality in Imitation Learning** — Belkhale, Cui, Sadigh (NeurIPS 2023)
[arXiv:2306.02437](https://arxiv.org/abs/2306.02437)

The theoretical foundation most later work builds on. Formalises quality through distribution
shift via two axes: **action divergence** (mismatch between expert and learned-policy actions
at a state) and **transition diversity** (environment stochasticity at a state). Contains the
counter-intuitive result that **state diversity is not always beneficial.**

*Adopt the vocabulary, immediately and at zero cost.* A manifest row that justifies itself as
"high action divergence in this perturbation regime, low transition diversity" is a far
stronger claim than "the policy failed a lot here", and it connects our evidence to an
established formalism a technical buyer will recognise. This is the cheapest credibility
upgrade in §2.

**DemInf: Robot Data Curation with Mutual Information Estimators** — Hejna, Mirchandani,
Balakrishna, Xie, Wahid, Tompson, Sanketi, Shah, Devin, Sadigh (Stanford / Google DeepMind)
[arXiv:2502.08623](https://arxiv.org/abs/2502.08623) ·
[site](https://joeyhejna.com/demonstration-info/) — per-demonstration quality from k-NN mutual
information over VAE embeddings. **Requires no policy training**, which makes it usable on our
hardware. 5–10% gains on RoboMimic; validated on ALOHA and Franka.

**QoQ — Demonstration Curation via Influence Functions** — Lee, Min, Kim, Kang, Liu, Pinto,
Lee (ICRA 2026) [arXiv:2603.09056](https://arxiv.org/abs/2603.09056). Influence functions with
robotics-specific adaptations (max-influence over validation samples; trajectory-level
aggregation). Up to 99.4% curation accuracy in sim. Near-identical territory to CUPID.

**ATHENA — Accelerated Multi-Task Heterogeneous Influence Functions**
[arXiv:2606.16208](https://arxiv.org/abs/2606.16208). Scales influence-based curation to
billion-parameter VLAs (~313× speedup via Kronecker-structured gradients and low-rank Hessian
approximation); matches full-dataset fine-tuning with 50–67% of the data on RoboTwin 2.0.
Relevant if we ever want influence scoring at SmolVLA scale on our hardware.

**Cheap offline quality scores** (a fragmented cluster, cite collectively):
PSD/smoothness-based ranking [arXiv:2605.01544](https://arxiv.org/abs/2605.01544);
"Learning from the Best: Smoothness-Driven Metrics" [arXiv:2604.23000](https://arxiv.org/abs/2604.23000);
"Consistency Matters" [arXiv:2412.14309](https://arxiv.org/abs/2412.14309).
All fully offline, no rollouts. **[unverified — repo maintenance status not checked.]** Useful
as a pre-filter on any remediation data we generate in Phase 5, before spending fine-tuning
compute on it.

### 2.3 Dataset pruning, mixture optimisation and retrieval

**Re-Mix: Optimizing Data Mixtures for Large Scale Imitation Learning** — Hejna, Bhateja,
Jian, Pertsch, Sadigh (NeurIPS 2024) [arXiv:2408.14037](https://arxiv.org/abs/2408.14037).
Distributionally-robust optimisation to reweight OXE *domains* for pretraining. +38%/+32% over
naive/expert-curated mixtures on WidowX/Franka; competitive at 25% of the data. Directly
applicable if a manifest row turns out to be about **re-weighting existing data** rather than
collecting new data — which is a `fixability` outcome our schema currently has no name for.
**Consider adding `reweighting` as a fifth fixability class**; it is cheaper than both
`augmentation` and `data` and Re-Mix gives it machinery.

**SCIZOR: Self-Supervised Data Curation for Large-Scale Imitation Learning**
[arXiv:2505.22626](https://arxiv.org/abs/2505.22626) ·
[site](https://ut-austin-rpl.github.io/SCIZOR/). Removes suboptimal state-action pairs via a
task-progress predictor and deduplicates in joint state-action space.

**STRAP: Robot Sub-Trajectory Retrieval for Augmented Policy Learning** — Memmel et al. (UW,
ICLR 2025) [arXiv:2412.15182](https://arxiv.org/abs/2412.15182) ·
[site](https://weirdlabuw.github.io/strap/). Retrieves relevant **sub-trajectories** from a
large prior dataset using vision-foundation-model features plus subsequence DTW.

*Why STRAP matters to us specifically:* PLAN §11 open question 4 asks whether remediation data
should be scripted in sim or re-rendered from existing demos. **STRAP is a third option we did
not consider — retrieve the relevant sub-trajectories that already exist in LIBERO's broader
pool.** It is the cheapest of the three by a wide margin and should be tried before either.
Combined with AHA's **FailGen** ([repo](https://github.com/NVlabs/AHA)), which procedurally
generates *failure* trajectories by perturbing successful demonstrations, we have open-source
answers to both halves of that open question — including the failed-attempt half of the
recovery demonstrations PLAN §4 admits it cannot currently validate.

General-ML coreset and pruning work (non-robotics, background only): non-uniform class-wise
coreset selection [arXiv:2504.13234](https://arxiv.org/abs/2504.13234); coresets from
trajectories [arXiv:2508.20230](https://arxiv.org/abs/2508.20230); multimodal-guided dynamic
pruning [arXiv:2507.12750](https://arxiv.org/abs/2507.12750). Low priority.

### 2.4 Active learning and the DAgger lineage

**DAgger** — Ross, Gordon, Bagnell (AISTATS 2011) is the direct intellectual ancestor of this
whole project: roll out the learned policy, find where it leaves the expert's distribution,
get supervision *there*. We should acknowledge it explicitly; a reviewer who spots that we
reinvented DAgger's premise without naming it will not be generous.

**Interactive Imitation Learning for Dexterous Robotic Manipulation: Challenges and
Perspectives — A Survey** (*Frontiers in Robotics and AI*, Dec 2025)
[arXiv:2506.00098](https://arxiv.org/abs/2506.00098). One citation covering the modern DAgger
family — HG-DAgger, LazyDAgger, Intervention-Weighted Regression, and hierarchical variants.

*The deliberate trade-off to state in our writeup:* DAgger-family methods query a live expert
**per timestep at the exact off-distribution state**. That is far finer-grained than our
per-cluster manifest. We trade granularity for **actionability by a data-operations team with
no expert sitting at the simulator** — which is precisely the iMerit/EXL delivery model. Saying
this out loud converts an apparent weakness into a design rationale.

Narrower active-collection work: **Active Robot Curriculum Learning from Online Human
Demonstrations** (HRI 2025) [arXiv:2503.02277](https://arxiv.org/abs/2503.02277);
**ActivePusher** [arXiv:2506.04646](https://arxiv.org/abs/2506.04646);
**Active Constraint Learning in High Dimensions** [arXiv:2512.22757](https://arxiv.org/abs/2512.22757).

### 2.5 Scaling laws and mixture ablations — how much data closes a gap

**Data Scaling Laws in Imitation Learning for Robotic Manipulation** — Lin, Hu, Sheng, Wen,
You, Gao (Tsinghua / Shanghai Qi Zhi, ICLR 2025)
[arXiv:2410.18647](https://arxiv.org/abs/2410.18647). 40,000+ demos, 15,000+ real rollouts.
**Generalization follows a power law in the number of distinct environments and objects, not
in demonstration count**, with sharp diminishing returns past a per-environment/object
threshold. See E2 — this is the most consequential single result in §2 for how manifest rows
should be written.

**Open X-Embodiment / RT-X** — [arXiv:2310.08864](https://arxiv.org/abs/2310.08864) ·
[site](https://robotics-transformer-x.github.io/). The canonical "which subset matters"
ablation: co-training helps small-data domains substantially, helps large-data domains only
with sufficient model capacity, and removing Bridge specifically collapses RT-2-X's emergent
skills.

**π0.5** — Physical Intelligence, [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) ·
[blog](https://www.pi.website/blog/pi05). Explicit mixture ablations across mobile-manipulator,
static-manipulator, cross-embodiment, high-level-semantic and web-VQA sources; removing either
cross-embodiment source significantly degrades performance. A modern template for the ablation
methodology we would need if a manifest row ever recommends changing a *pretraining* mix
rather than fine-tuning data.

### 2.6 Continuous / self-improving loops

**Self-Evolving Learning for Embodied AI with Criticality Model**
[arXiv:2607.28251](https://arxiv.org/abs/2607.28251). Learns a criticality model from the
policy's own execution outcomes to predict future failure, then importance-samples training
data toward failure-prone states. Reports **51–67% failure-rate reduction** across 5 sim and
real domains. Conceptually an automated, continuous version of our manifest — but it
reweights an existing pool and produces no human-readable artifact.
*Adopt:* its criticality score is a principled ranking function for our failure clusters,
complementing hand-assigned `severity`.

**ARMADA** — Yu, Lv, Ying, Jin, Wen, Lu (SJTU, RA-L 2026)
[arXiv:2510.02298](https://arxiv.org/abs/2510.02298) ·
[repo](https://github.com/Virlus/armada). Online failure detection (FLOAT, ~95% accuracy)
routes to human tele-intervention only on detected failures, feeding corrective data back.
The real-time, reactive counterpart to our offline stress-and-prioritise workflow.

**Data and Evaluation Closed-Loop for Model Capability Enhancement** — Li, Yuan, Xu (Jun 2026)
[arXiv:2606.28471](https://arxiv.org/abs/2606.28471). **LLMs, not robotics** — but the
structure is our structure, and it is worth reading for exactly that reason. Introduces the
**"capability slice"** (a grouping of evaluation samples sharing background conditions, task
type, solving operation and output constraints) as the bridge from benchmark failure to data
intervention, plus explicit mapping rules from an evaluation taxonomy to a data taxonomy. Two
validated case studies, one of which lifts AIME 6.67% → 26.67% through targeted sampling; the
other is a cautionary tale where a 46.82% benchmark drop turned out to be a masked `<EOS>`
loss bug, **not** a capability gap at all.

*Two things to take:* the **"capability slice"** is a better name than "failure cluster" and
carries the right connotation (a *region of capability*, not an incident). And their second
case study is the LLM-world version of our §7c discriminator #1 — **always check the
measurement before believing the gap.**

### 2.7 Verdict on §2

| Our component | Prior art | Verdict |
|---|---|---|
| Failure-targeted collection beats random collection | Mirchandani et al. 2411.01813 | **Already established.** Cite it; don't re-prove it. |
| Scoring/ranking existing demonstrations | CUPID, Demo-SCORE, DemInf, QoQ, ATHENA, SCIZOR, Re-Mix | **Crowded and more rigorous than us.** We should not enter this space; we should consume it. |
| Specifying *new* conditions to collect, prioritised and costed | Nothing found that does this as a deliverable | **Our contribution.** Generative, not subtractive. |
| Validating by fine-tuning and re-measuring closed-loop | Demo-SCORE (filtering); The Moving Eye (viewpoint) | **Rare and valuable.** Keep. The Pebblous negative result is the argument for why. |
| Sample-count recommendations | Lin et al. scaling laws contradict the naive version | **Change the schema's emphasis** from counts to distinct-condition coverage. |

---

## 3. Datasets and policies we could expand to

**Read this section before the next Phase-2 work session.** It contains the only finding in
the survey that changes something we are about to do: our reproduction gate is currently
aimed at a checkpoint that the community has repeatedly failed to reproduce, on a stack with
a known, unpinned, silently-breaking dependency.

### 3.1 The SmolVLA / LIBERO reproducibility situation is worse than PLAN assumed

PLAN §1 flags [lerobot#3264](https://github.com/huggingface/lerobot/issues/3264) as "the
largest risk" and budgets a week. The picture is broader than one issue.

**#3264 is still open** (filed 2026-04-02, last activity 2026-08-09). The reporter got
Spatial 63% / Object 93% / Goal 81% / Long 56%, averaging **73.25%** against the paper's
87.3%, using the official checkpoint. **No LeRobot maintainer has commented on it.**

It is not isolated. At least **nine** distinct SmolVLA/LIBERO reproduction issues exist:

| Issue | State | Note |
|---|---|---|
| [#1316](https://github.com/huggingface/lerobot/issues/1316) | closed (completed) | 55 comments; as late as 2026-04-21 users still report 0.672 on Spatial. Closed, not resolved. |
| [#1369](https://github.com/huggingface/lerobot/issues/1369) | closed (completed) | 27 comments; maintainer redirected users to re-download a **silently modified** dataset |
| [#2107](https://github.com/huggingface/lerobot/issues/2107) | **open** | 7.5% success training SmolVLA-0.24B; no maintainer response |
| [#2354](https://github.com/huggingface/lerobot/issues/2354) | **open** | 0.73/0.91/0.83/0.43 vs paper 0.90/0.96/0.92/0.71; assigned, no reply |
| [#2850](https://github.com/huggingface/lerobot/issues/2850) | closed — **genuinely fixed** | `--eval.batch_size` was capping the number of unique initial states sampled, hitting long-horizon `libero_10` hardest |
| [#3287](https://github.com/huggingface/lerobot/issues/3287) | closed | user got 44.8% vs ~71% on `libero_10`; no maintainer answer visible |
| [#3628](https://github.com/huggingface/lerobot/issues/3628) | **open** | A100, official checkpoint, official CLI, still short |
| [#4390](https://github.com/huggingface/lerobot/issues/4390) | **open** | the MuJoCo bug — see below |
| [#2418](https://github.com/huggingface/lerobot/issues/2418), [#2624](https://github.com/huggingface/lerobot/issues/2624) | **open** | shape mismatch on fine-tune; dataset simulation errors |

**The most actionable single item in this entire document:**
[lerobot#4390](https://github.com/huggingface/lerobot/issues/4390) — **MuJoCo ≥ 3.4.0 changed
box-box collision handling, so in `libero_spatial` task 5 the bowl no longer settles onto the
ramekin.** Measured impact: SmolVLA 0.45B **80% → 28% on that task alone** (up to ~9 pp on the
whole suite); OpenVLA-OFT+GRPO 98% → 12–19%; π0.5 comparatively robust (90% → 86%). **A fresh
`pip install` today resolves MuJoCo to 3.8.1 — the broken side.**
[PR #4465](https://github.com/huggingface/lerobot/pull/4465), pinning
`mujoco>=3.2.7,<3.4.0`, is **open, approved by one reviewer, not merged.**

**Pin `mujoco<3.4.0` now, before running anything.** Without it, a physics change masquerades
as a model failure — which is the single failure mode our whole methodology exists to prevent,
and it would have poisoned Phase 2 and every manifest row derived from it. Note honestly that
this does *not* explain #3264, whose reporter was on MuJoCo 3.3.2 (the healthy side). Several
causes are in play.

Other contributing causes identified across the issues:
- **Action-chunk mismatch** — `--policy.n_action_steps` differing from what the checkpoint was
  trained with; community-attributed as a likely cause of the `libero_10`-specific gap.
- **Dataset drift** — `HuggingFaceVLA/libero` has been re-uploaded. LeRobot's docs now warn:
  *"Pin `--dataset.revision=<commit-sha>` when reporting results."* **Our PREDICTIONS.md
  already pins `a1aaacb7...` — that was the right instinct and should extend to every run.**
- **Too few episodes** — 10 vs. the recommended 50 can swing Spatial/Long by 15–20 points.
- **8 GB specifically**: [#3098](https://github.com/huggingface/lerobot/issues/3098) was filed
  on an **RTX 3070 Laptop (8 GB)** — the PyTorch CUDA context and MuJoCo's EGL rendering
  context contend for VRAM in one process and evaluation can fail outright.
  [PR #3235](https://github.com/huggingface/lerobot/pull/3235) adds
  `--eval.process_isolated=true` (subprocess env with `MUJOCO_GL=osmesa`) — **open, unmerged.**
  Workaround available today: set `MUJOCO_GL=osmesa` or `glfw` to force CPU rendering, at a
  speed cost. **This is our exact hardware class** and it reinforces PLAN §8's expectation that
  rendering, not inference, is the bottleneck.

**The finding that should change a decision.** LeRobot's
[LIBERO docs](https://huggingface.co/docs/lerobot/libero) contain a "Reproducing published
results" section written **entirely about π0.5** — `lerobot/pi05_libero_finetuned` at
**97.5% average** (97/99/98/96), reproducing OpenPI's own 96.85%. **There is no equivalent
"we reproduced SmolVLA" claim anywhere in the documentation.** Given nine open reproduction
issues, that absence is not an oversight.

The consequence for PLAN §5: **our Phase-2 reproduction gate on SmolVLA may simply be
unachievable**, and PLAN already contemplates the fallback ("establish our own baseline over
3 seeds, and say plainly this is a weaker gate"). The survey says: take the fallback
seriously and consider changing the policy instead. Three candidates below give a gate that
can actually pass, at 8 GB.

### 3.2 Small policies with a LIBERO checkpoint that fits 8 GB

| Policy | Params | bf16 VRAM | LoRA in 8 GB | Licence | LIBERO checkpoint | Reproducible? |
|---|---|---|---|---|---|---|
| **SmolVLA** | 450M | ~1–2 GB | **yes** | Apache-2.0 | [`lerobot/smolvla_libero`](https://huggingface.co/lerobot/smolvla_libero) | **No** — §3.1 |
| **VLA-Adapter** | **1B** (0.5B Qwen2.5-0.5B backbone) | ~2 GB | **yes** | MIT | [`VLA-Adapter/LIBERO-Goal`](https://huggingface.co/VLA-Adapter/LIBERO-Goal) family; **97.3% claimed** | Partial gap: [#46](https://github.com/OpenHelix-Team/VLA-Adapter/issues/46) reports 94% vs ~98%, traced to a **batch-size mismatch vs the official multi-GPU recipe** |
| **MiniVLA** | **1B** (Qwen2-0.5B + SigLIP + DINOv2) | ~2 GB | **yes** — explicit LoRA support | MIT | [`Stanford-ILIAD/minivla-libero90-prismatic`](https://huggingface.co/Stanford-ILIAD/minivla-libero90-prismatic) + VQ variant | No failed-repro issue found **[unverified — absence of evidence]** |
| **TurboVLA** | **0.2B**, no LLM at all | <1 GB (0.9 GB on a 4090) | trivially | Apache-2.0 code (DINOv3 licence on weights) | [`H-EmbodVis/TurboVLA`](https://huggingface.co/H-EmbodVis/TurboVLA); **97.7% claimed at 31 ms/step** | **[unverified — mid-2026, no issue history yet]** |
| **π0.5** | ~3B (PaliGemma-class) | ~6–7 GB | **no** | Apache-2.0 | [`lerobot/pi05_libero_finetuned`](https://huggingface.co/lerobot/pi05_libero_finetuned) | **Yes — the only one LeRobot maintainers confirm** |
| Octo-small / base | 27M / 93M | <1 GB | trivially | MIT | **no official LIBERO checkpoint** from rail-berkeley; the 75.1% figure is a baseline other papers report | — |
| π0 / π0-FAST | ~3.3B | >8 GB per openpi docs | **no** (openpi: LoRA needs >22.5 GB) | Apache-2.0 | not the primary LIBERO target | — |
| TinyVLA S/B/H | 400M/700M/1.3B | 0.8–2.6 GB | **yes** (LoRA *is* the method) | MIT | **no LIBERO checkpoint** (ALOHA/MetaWorld) | — |
| GR00T N1 / N1.5 / N1.7 | 2.2B / 3B / 3B | ~4.4–6 GB | community reports need 16 GB | N1.5 **non-commercial**; N1.7 permits commercial | N1.7 has a **SimplerEnv-Bridge** checkpoint, not LIBERO | — |
| SpatialVLA | 4B | ~8 GB | no | MIT | via IPEC-COMMUNITY | — |
| NORA / NORA-1.5 | 4B | ~8 GB | no | MIT | checkpoints pending at research time — **verify** | — |
| RynnVLA-002 | large (Chameleon) | — | no | Apache-2.0 | claims **97.4% LIBERO** | **[unverified]** |
| RoboVLMs | ~1.8–2B (KosMos-2) | ~3.5–4 GB | plausible | Apache-2.0 | **no LIBERO** — CALVIN and SimplerEnv only | — |
| Seer / EdgeVLA | — | — | — | MIT (EdgeVLA) | **no LIBERO results** | — |
| Diffusion Policy / ACT | ~263M / ~27–84M | <1 GB | trivial | MIT | DP appears as a 72.4% re-implementation baseline in the OpenVLA paper; ACT is ALOHA-oriented | — |

**The recommendation this table produces.** PLAN §1's argument for going small was *"give up
the prestige of the headline model, gain the ability to close the loop."* That argument is
sound and this survey strengthens it — but it does not uniquely select SmolVLA, and SmolVLA is
the one small policy in the table with a documented, unresolved, nine-issue reproduction
problem.

**Consider evaluating VLA-Adapter or MiniVLA alongside SmolVLA in week 1**, at ~2 GB each, both
MIT, both with LIBERO checkpoints, both LoRA-able in 8 GB. Cost: a day or two. Benefit: the
Phase-2 reproduction gate — *the project's only defence against "your environment was broken,
not the model"* — might actually pass. VLA-Adapter's known gap is even instructive: it is
attributed to a **batch-size mismatch against the official multi-GPU recipe**, which is exactly
the class of environment bug the gate is designed to catch, and it has an identified cause
rather than nine years of silence.

There is also a second-policy argument. PLAN §7c discriminator #5 is *"do other policies fail
in the same region?"* — currently unscoped. Two 1–2 GB policies make that discriminator
affordable, and it is the one that distinguishes "this policy is weak" from "this regime is
genuinely hard or the benchmark is broken." Given §3.1, the ability to tell those apart is
worth more to us than usual.

### 3.3 Benchmarks beyond LIBERO

| Benchmark | Engine | 8 GB? | Checkpoints | Licence | Red flags |
|---|---|---|---|---|---|
| **LIBERO-plus** | MuJoCo/robosuite — identical stack | **yes** | same baselines as LIBERO | — | **Replaces** vanilla LIBERO (uninstalls `hf-libero`); cannot co-exist — matches the proposal's separate-container guidance. First-class LeRobot CLI: `--env.type=libero_plus`, [docs](https://huggingface.co/docs/lerobot/libero_plus), dataset [`lerobot/libero_plus`](https://huggingface.co/datasets/lerobot/libero_plus) |
| **RoboCasa** | MuJoCo/robosuite | **yes** — same stack | **[`lerobot/smolvla_robocasa`](https://huggingface.co/lerobot/smolvla_robocasa)** — a direct SmolVLA baseline | MIT code; asset licences vary (Objaverse) | DP-via-robomimic reproduction far below paper ([openpi#612](https://github.com/Physical-Intelligence/openpi/issues/612)); hang in `dataset_states_to_obs.py` ([robocasa#84](https://github.com/robocasa/robocasa/issues/84)) |
| **CALVIN** | PyBullet | **yes — lightest option** | HULC/HULC++, 3D Diffuser Actor, RoboFlamingo, GR-1/Seer/MDT, RL-tuned π0/π0.5 | MIT | Evaluator can score valid ground-truth trajectories as failures ([calvin#32](https://github.com/mees/calvin/issues/32)) **[unverified whether fixed]** |
| **THE COLOSSEUM** | PyRep/CoppeliaSim | likely | [rvt_colosseum](https://github.com/robot-colosseum/rvt_colosseum) runs RVT/PerAct; **V2 (2026) adds VLA support** ([arXiv:2605.27759](https://arxiv.org/pdf/2605.27759)) | **[unverified — likely inherits RLBench's academic-only terms]** | PerAct 34.5% → 6.4% under perturbation is the *result*, not a bug |
| **SimplerEnv** | SAPIEN (ManiSkill) | likely | harness only, not a checkpoint source | **[unverified]** | Sim-to-real gripper-frame mismatch causing OpenVLA underperformance ([#78](https://github.com/simpler-env/SimplerEnv/issues/78)); headless friction ([#6](https://github.com/simpler-env/SimplerEnv/issues/6), [#7](https://github.com/simpler-env/SimplerEnv/issues/7)). See also §4.2 on its contested sim-real correlation |
| **VLABench** | MuJoCo + dm_control | likely | π0, π0-FAST, π0.5 fine-tunes on HF; OpenVLA support | MIT | none found |
| **ManiSkill3** | SAPIEN, GPU-parallel PhysX | **yes** — explicitly memory-efficient, runs on free Colab | PPO/SAC/TD-MPC2, BC, DP, Octo, RDT-1B, RT-X | Apache-2.0 code; **assets CC BY-NC 4.0 — non-commercial** | Trajectory-replay determinism issues ([#892](https://github.com/haosulab/ManiSkill/issues/892), [#422](https://github.com/haosulab/ManiSkill/issues/422)) — **directly threatens our G5 determinism guarantee if we ever move here** |
| **RLBench** | CoppeliaSim + PyRep | yes, but headless X/EGL is finicky ([#93](https://github.com/stepjam/RLBench/issues/93)) | PerAct, RVT/RVT-2, 3D Diffuser Actor | **Academic / non-commercial only** — flag for any EXL/iMerit commercial use | Below-paper numbers reported ([3d_diffuser_actor#71](https://github.com/nickgkan/3d_diffuser_actor/issues/71)); RVT authors average 5 seeds because the motion planner is stochastic |
| **RoboCasa365** | MuJoCo, mobile manip, 20 Hz | **[unverified]** | ICLR 2026; 2,500 kitchens / 365 tasks | **[unverified]** | [paper](https://arxiv.org/pdf/2603.04356), [leaderboard](https://robocasa.ai/leaderboard.html) |
| **RoboTwin 2.0** | SAPIEN + CuRobo | **[unverified]** | ACT/DP baselines in progress | MIT | project's own docs call leaderboard numbers "provisional" |
| **AGNOSTOS** | RLBench | likely | zero-shot VLA baselines, not a checkpoint release | **[unverified]** | [site](https://jiaming-zhou.github.io/AGNOSTOS/) |
| **GemBench** | RLBench | likely | 3D-LOTUS ([repo](https://github.com/vlc-robot/robot-3dlotus)) | **[unverified]** | — |
| **VIMA-Bench** | PyBullet | yes | 7 checkpoints, 2M–200M, on [HF](https://huggingface.co/VIMA/VIMA) | MIT | none found |
| **Meta-World** | MuJoCo | yes, but **RL only — no VLA checkpoints** | — | MIT | v1→v2 reward rewrite means original paper numbers are **structurally** unreproducible |
| **BEHAVIOR-1K** | Isaac Sim / OmniGibson | **no** — needs RT cores; Stanford recommends 3090/4090 | π0 task-specific checkpoints | **[unverified]** | setup friction ([#362](https://github.com/StanfordVL/BEHAVIOR-1K/issues/362), [#899](https://github.com/StanfordVL/BEHAVIOR-1K/issues/899)) |
| **Isaac Lab / Lab-Arena** | Isaac Sim / PhysX | **borderline — Isaac Lab recommends ≥16 GB** | most SmolVLA-native heavy sim: first-class GR00T N1x, π0/π0.5, SmolVLA, ACT, DP | Apache-2.0 + Omniverse EULA | ecosystem too new for issue history. Confirms the proposal's §7.2 placement as scale-out, not now |
| **RoboArena** | **real robots, not sim** | n/a | n/a | n/a | Not a sim benchmark — see §1.5. Distinct from **RobotArena∞** ([arXiv:2510.23571](https://arxiv.org/abs/2510.23571)) |

**Three observations worth acting on.**

1. **`lerobot/smolvla_robocasa` exists.** A second benchmark with the *same policy*, the *same
   MuJoCo/robosuite stack*, and a published checkpoint is the cheapest possible generality
   claim available to us. If a manifest row reproduces on RoboCasa, its scope upgrades from
   "one benchmark" to "two benchmarks, one embodiment" — meaningfully stronger for one extra
   integration, and far cheaper than the cross-embodiment claim §4 warns us off.
2. **Licence discipline matters here more than in a research project.** RLBench (and probably
   COLOSSEUM, which builds on it) is **academic/non-commercial**, and ManiSkill's *assets* are
   CC BY-NC. For an iMerit/EXL commercial artifact, **LIBERO/LIBERO-plus/RoboCasa (MIT,
   MuJoCo) and VLABench (MIT) are the clean expansion path**, and COLOSSEUM — attractive on
   methodology — may not be usable commercially. Check before building on it.
3. **The reproducibility red-flag column is the real content of this table.** Nearly every
   benchmark has an open issue where published numbers do not reproduce: RoboCasa, CALVIN,
   RLBench, ManiSkill, SimplerEnv, LIBERO. Meta-World's are unreproducible by construction.
   This is worth saying in the readout, because it reframes our reproduction gate from
   pedantry into the thing that distinguishes a measurement from a number — and it makes
   §3.1's MuJoCo finding a *demonstration of the method working*, not an embarrassment.

---

## 4. Transfer and cross-embodiment — how far our findings can travel

The commercially important question this section answers is narrow and specific:
**when we show a client a Data Gap Manifest derived from SmolVLA on LIBERO, what is the
defensible scope of that claim?** The literature gives a clear, and usefully asymmetric,
answer.

### 4.1 The distinction that must not be blurred

Two different generalisations get casually conflated. Keeping them apart is the single most
important thing in this section.

**(a) Cross-policy, same embodiment** — does a failure mode found on SmolVLA/LIBERO hold for
OpenVLA, π0, UniVLA on the same benchmark?
**Evidence: strong yes.** LIBERO-Plus ([arXiv:2510.13626](https://arxiv.org/abs/2510.13626))
tests **10 architecturally diverse models** — OpenVLA, OpenVLA-OFT and two variants, π0,
π0-FAST, NORA, WorldVLA, UniVLA, RIPT-VLA — spanning autoregressive, diffusion,
FAST-tokenized and world-model designs, and finds the brittleness pattern is essentially
**uniform across all of them**: catastrophic sensitivity to camera viewpoint and robot initial
state, moderate to lighting/background/noise/layout, near-insensitivity to language, with
evidence that models rely on **memorised spatial positions rather than semantic grounding**.

Note that **SmolVLA is not among the ten.** That is both a gap we fill and a caveat: our
policy's inclusion in the pattern is an inference, not a measurement, until we measure it.

**(b) Cross-embodiment / sim-to-real** — does it hold on a different robot, or a real one?
**Evidence: mixed to cautionary.** Details below.

Our manifest should label every row with which of these two scopes its evidence supports.
Most of our rows will be (a)-scoped. Presenting an (a)-scoped finding as if it were
(b)-scoped is the specific over-claim this literature punishes.

### 4.2 Sim-to-real correlation is contested, and the numbers have moved against us

This is the most important quantitative finding in §4 and it is not what the proposal assumes.

- **SIMPLER / SimplerEnv** (CoRL 2024, [arXiv:2405.05941](https://arxiv.org/abs/2405.05941),
  [repo](https://github.com/simpler-env/SimplerEnv)) — 1,500+ paired sim/real evaluations,
  2 embodiments, 8 task families. Reports **Pearson r ≈ 0.92** average on Google Robot tasks
  (range 0.855–0.976), MMRV ≈ 0.056. Notably, SIMPLER *predicted* Octo-Base's sensitivity to
  arm-texture changes, later confirmed on real hardware. **[moderate confidence — figures read
  via secondary summary; check the paper's tables before quoting.]**
- **A Practical Recipe Towards Improving Sim-and-Real Correlation for VLA Evaluation**
  (Jun 2026, Tsinghua / Shanghai Qi Zhi, [arXiv:2606.10366](https://arxiv.org/abs/2606.10366))
  — an independent systematic replication across VLA-Arena, SIMPLER and REALM reports
  **Spearman ranking correlations of REALM ≈ 0.700, VLA-Arena ≈ 0.575, SIMPLER ≈ 0.400**
  (MMRV 0.030 / 0.060 / 0.128). It further finds SIMPLER **omits behaviour perturbations
  entirely** and that its **perturbation-severity ordering does not match the real world's.**
  Simulator-specific fine-tuning improved REALM's correlation 0.700 → 0.875, but more data was
  **non-monotonic** — 20 demos/task was worse than 10. **[moderate confidence — figures read
  via fetch summary.]**
- **THE COLOSSEUM** ([arXiv:2402.08191](https://arxiv.org/abs/2402.08191)) independently
  measures sim–real perturbation correlation at **R² = 0.614.**
- **AutoEval** ([arXiv:2503.24278](https://arxiv.org/abs/2503.24278)) reports that automated
  *real-world* evaluation matches human evaluation **more closely than SIMPLER or offline
  validation MSE do** — a direct, quantified criticism of photorealistic sim as a proxy.
- **Robot Policy Evaluation for Sim-to-Real Transfer: A Benchmarking Perspective**
  ([arXiv:2508.11117](https://arxiv.org/abs/2508.11117)) names **LIBERO specifically**, with
  RLBench, Meta-World, ManipBench and COLOSSEUM, as benchmarks that may not capture the
  distribution shifts that matter in deployment. **[unverified beyond abstract.]**

**What we should do about it.** Do not write "sim is an imperfect proxy" and move on. State a
number and a range: *sim-to-real rank correlation for VLA evaluation is measured between
roughly 0.4 and 0.7 Spearman depending on the simulator, and COLOSSEUM puts perturbation
correlation at R² = 0.614.* Then note that **severity ordering specifically is the part that
transfers worst** — which matters more to us than to most, because our manifest *prioritises
by severity.* A prioritisation derived purely in sim may reorder in the real world. Say so in
the manifest's provenance, and make it the argument for the SureSim/PPI upgrade path in E4.

### 4.3 Mechanisms — which of our findings are likely to transfer, and which are not

The literature is now specific enough to *predict* this, which is more useful than a blanket
hedge. The dividing line is **where in the stack the failure lives.**

**Likely embodiment-agnostic** (upstream of the action decoder — perception, semantics,
spatial grounding):
- Latent-action work — **LAPA** ([arXiv:2410.11758](https://arxiv.org/abs/2410.11758),
  [repo](https://github.com/LatentActionPretraining/LAPA)) and **Moto**
  ([arXiv:2412.04445](https://arxiv.org/abs/2412.04445),
  [repo](https://github.com/TencentARC/Moto)) define actions on *pixels*, making them
  embodiment-agnostic by construction; **GR-2** ([arXiv:2410.06158](https://arxiv.org/abs/2410.06158))
  and **GR00T N1** ([arXiv:2503.14734](https://arxiv.org/abs/2503.14734)) both treat
  action-free video as a pseudo-embodiment unified by a latent-action codebook.
- **AGNOSTOS** ([arXiv:2505.15660](https://arxiv.org/abs/2505.15660)) shows cross-*task*
  generalization is broken *within* a fixed embodiment: on 23 unseen RLBench tasks, π0 reaches
  **17.5% overall (21.7% L1 / 12.0% L2)**, OpenVLA **13.6%**, VoxPoser 15.6%, 3D-LOTUS++ 14.4%;
  the paper's own X-ICM method reaches only 23.5–30.1%. **[moderate-high confidence — table
  read via HTML fetch.]** Conclusion: brittleness to novel objects and spatial configurations
  is a field-wide pattern, not an embodiment artifact.

So: our **visual grounding, spatial reasoning and planning** family findings are the ones most
likely to travel. Good — they are also the ones a client cares most about.

**Likely embodiment-specific** (at or below the action decoder):
- **FAST** ([arXiv:2501.09747](https://arxiv.org/abs/2501.09747)) exists because **naive
  per-dimension action binning fails outright on high-frequency, dexterous data.** Any failure
  of ours tied to action encoding is a property of SmolVLA's action head at LIBERO's 10 Hz,
  not a property of VLAs.
- **Demystifying Action Space Design for Robotic Manipulation Policies**
  ([arXiv:2602.23408](https://arxiv.org/abs/2602.23408); >13,000 real rollouts, 500+ trained
  models) finds **joint-space control is better in-distribution, but the ordering *reverses*
  under cross-embodiment fine-tuning**, where end-effector/task-space wins because it abstracts
  away kinematics. Chunk-wise delta encodings beat step-wise by ~10%. **[figures approximate.]**
  This is the sharpest single caution in §4: **a conclusion can flip sign between the
  in-distribution and cross-embodiment settings.**
- **RDT-1B** ([arXiv:2410.07864](https://arxiv.org/abs/2410.07864),
  [repo](https://github.com/thu-ml/RoboticsDiffusionTransformer)) building an explicitly
  *physically interpretable* unified action space is itself evidence that a leading lab
  considers naive action pooling insufficient.

**Consequence for the manifest, concretely:** add a `scope` annotation to each row —
`cross-policy` / `same-embodiment` / `speculative-cross-embodiment` — justified by *where in
the stack the failure sits*. Our `manipulation/control` and `recovery` rows are almost
certainly embodiment-specific. Our `visual grounding` and `spatial reasoning` rows have real
claim to breadth. That is a more sophisticated and more defensible statement than a uniform
disclaimer, and it costs one field.

### 4.4 Negative transfer — it is real, and conditional

**X-Diffusion: Training Diffusion Policies on Cross-Embodiment Human Demonstrations** (Cornell,
2025) [arXiv:2511.04671](https://arxiv.org/abs/2511.04671). Documents that naive co-training of
human plus robot demonstrations **degrades performance versus robot-only training**, because
human execution strategies (side-grasping, for instance) are kinematically infeasible for the
robot. Their fix treats human actions as high-noise counterparts in the diffusion forward
process, discarding embodiment-specific execution detail while preserving task intent.

**Cross-Embodiment Offline RL for Heterogeneous Robot Datasets**
[arXiv:2602.18025](https://arxiv.org/abs/2602.18025) — reports that embodiments with little
native data degrade under cross-embodiment training, especially when the donor pool is
dominated by suboptimal trajectories. **[unverified — read from a search snippet only; check
before quoting.]**

**ET-VLA: Embodiment Transfer Learning for VLA Models**
[arXiv:2511.01224](https://arxiv.org/abs/2511.01224). Frames the problem as: SOTA
autoregressive VLAs do **not** transfer to new embodiments without dedicated adaptation
machinery; reports beating OpenVLA by >53% relative on 6 real tasks after adding synthetic
continued pretraining and an embodied graph-of-thought. **A >53% deficit requiring explicit
machinery to close is itself the evidence** that single-embodiment competence does not
generalise for free.

**The counter-signal, and why it does not rescue us.** None of the three major
"one policy, many embodiments" papers report negative transfer in their own ablations —
**Octo** ([arXiv:2405.12213](https://arxiv.org/abs/2405.12213),
[repo](https://github.com/octo-models/octo), 800k trajectories / 9 platforms),
**CrossFormer** ([arXiv:2408.11812](https://arxiv.org/abs/2408.11812),
[repo](https://github.com/rail-berkeley/crossformer), 900k / 20 embodiments incl. quadrupeds
and quadcopters), **HPT** ([arXiv:2409.20537](https://arxiv.org/abs/2409.20537),
[repo](https://github.com/liruiw/HPT), 52 datasets, >20% gain on unseen tasks). The resolution
that fits all the evidence: **negative transfer appears when pooling is naive (raw actions, no
embodiment conditioning) and is largely avoided by explicit embodiment tokens, stems and
adaptation layers.** Publication bias toward positive scaling results is also a live
possibility and should not be dismissed.

Either way, SmolVLA-on-LIBERO is a **single-embodiment** setting with none of that machinery
— so it has no exposure to either the benefit or the safeguard, and informal extrapolation
from it is unprotected.

Finally, **Open X-Embodiment / RT-X** ([arXiv:2310.08864](https://arxiv.org/abs/2310.08864))
established the founding claim that transfer works, but its own per-robot breakdown shows the
effect is **uneven** — strongest for small or low-quality single-robot datasets, weakest or
occasionally negative for large well-curated ones **[unverified at the per-robot level]**. Even
the paper that made the field believe in transfer shows transfer is heterogeneous.

### 4.5 One more benchmark-integrity caution

**LIBERO-PRO: Towards Robust and Fair Evaluation**
[arXiv:2510.03827](https://arxiv.org/abs/2510.03827) **[unverified — title and context only]**
appears to be a contemporaneous critique of LIBERO evaluation fairness, published alongside
LIBERO-Plus. If it holds up, it means **raw LIBERO success rate is considered unreliable even
for judging performance on the training embodiment** — which bears directly on our Phase-2
reproduction gate. Worth 30 minutes before we commit to that gate's tolerance.

---

## 5. Prioritised reading list

Tiered. **Read first** is roughly five evenings. URLs were fetch-checked on 2026-09-11 except
where marked.

### Tier 1 — read first

**Evening 1 · The error-analysis mindset (this is the intellectual core of "failure mining")**

1. **Domino: Discovering Systematic Errors with Cross-Modal Embeddings** — Eyuboglu, Varma,
   Saab, Delbrouck, Lee-Messer, Dunnmon, Zou, Ré (ICLR 2022)
   [arXiv:2203.14960](https://arxiv.org/abs/2203.14960) ·
   [code](https://github.com/HazyResearch/domino)
   *The canonical slice-discovery paper and the closest methodological cousin to failure
   mining anywhere in ML. Automatically finds coherent, describable error slices using
   cross-modal embeddings. If we present failure mining to an ML audience without knowing
   this paper, it will be the first thing we are asked about.*
2. **A Recipe for Training Neural Networks** — Karpathy (2019)
   [karpathy.github.io](https://karpathy.github.io/2019/04/25/recipe/)
   *"Neural net training fails silently." The discipline — visualise inputs immediately before
   the model, overfit a tiny batch, ramp complexity slowly — is precisely what separates a
   real failure from a broken environment. Cf. §3.1.*
3. **Full Stack Deep Learning — Troubleshooting & Testing** (Josh Tobin)
   [fullstackdeeplearning.com](https://fullstackdeeplearning.com/course/2022/lecture-3-troubleshooting-and-testing/)
   *The practitioner-grade version of the same discipline: bias/variance decomposition, error
   analysis workflow, "can it memorise a tiny subset".*

**Evening 2 · Evaluation rigour**

4. **Robot Learning as an Empirical Science: Best Practices for Policy Evaluation** —
   Kress-Gazit, Hashimoto, Kuppuswamy, Shah, Horgan, Richardson, Feng, Burchfiel (TRI)
   [arXiv:2409.09491](https://arxiv.org/abs/2409.09491)
   ***The canonical reference for this project's evaluation half.*** Argues success-rate-only
   reporting is unrigorous and prescribes what to report instead — N and initial conditions,
   complementary metrics, statistical tests, qualitative failure-mode descriptions. Read it as
   a template for the manifest's evidence fields.
5. **A careful examination of large behavior models for multitask dexterous manipulation** —
   TRI, *Science Robotics*
   [DOI](https://www.science.org/doi/10.1126/scirobotics.aea6201) · companion blog
   [Statistical Thinking for Robot Policy Evaluation](https://medium.com/toyotaresearch/statistical-thinking-for-robot-policy-evaluation-from-rigorous-a-b-testing-to-effective-0ae886fbd68d)
   **[blog unverified — Medium returned HTTP 403 to automated fetch; likely bot-blocking, not
   a dead link]**
   *The gold-standard worked example: blind A/B, N=50 real / 200 sim per condition, Bayesian
   credible regions, paired Barnard's exact and Welch's tests, Bonferroni correction.*
6. **Robot Policy Evaluation: Why 90% vs 92% Proves Little** — Tidiane Stano
   [dev.to](https://dev.to/tidiane_stano_c6b88f8b685/robot-policy-evaluation-why-90-vs-92-proves-little-59p3)
   *Short and concrete: Clopper–Pearson/Wilson intervals, McNemar's paired test, and the
   hierarchical episode-within-task structure, applied directly to robot success rates.*

**Evening 3 · Know the model and its family**

7. **A Survey on Vision-Language-Action Models: An Action Tokenization Perspective**
   [arXiv:2507.01925](https://arxiv.org/abs/2507.01925)
   *The most conceptually useful VLA survey: frames every VLA as one pipeline emitting
   "action tokens" of differing types. Gives the vocabulary to reason about* why *SmolVLA
   fails where it does, rather than only where.*
8. **SmolVLA** — [HF blog](https://huggingface.co/blog/smolvla) ·
   [arXiv:2506.01844](https://arxiv.org/html/2506.01844v1)
   *The model under test, from the source. Read the training-data description before
   perturbing it — it is the qualitative companion to our Phase-0 histogram.*
9. **π0.5: a VLA with Open-World Generalization** — Physical Intelligence
   [blog](https://www.pi.website/blog/pi05) · [arXiv:2504.16054](https://arxiv.org/abs/2504.16054)
   *The clearest public account of* why *VLAs fail to generalise — co-training mixture
   composition and data heterogeneity. Also the contrast case: π0.5 is the one LIBERO
   checkpoint LeRobot confirms reproduces (§3.1).*

**Evening 4 · The two papers that most directly precede us**

10. **Deployment-Time Reliability of Learned Robot Policies** — Christopher Agia, Stanford PhD
    thesis (2026) [arXiv:2603.11400](https://arxiv.org/abs/2603.11400)
    ***The single best read in this survey.*** 182 pages joining runtime monitoring,
    influence-function interpretability for diagnosis *and dataset curation*, and policy
    coordination — the closest existing document to our whole intellectual programme. If only
    one thing is read from this list, read this.
11. **LIBERO-Plus: In-depth Robustness Analysis of VLA Models**
    [arXiv:2510.13626](https://arxiv.org/abs/2510.13626) ·
    [CVPR 2026 camera-ready](https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html) ·
    [repo](https://github.com/sylvestf/LIBERO-plus)
    *The substrate we are building on, in full — not the abstract. Read the camera-ready, not
    the v1 arXiv (§1.1). Pay attention to the "models ignore language" result and to which 10
    models were tested (SmolVLA is not among them).*

**Evening 5 · The data half**

12. **Data Scaling Laws in Imitation Learning for Robotic Manipulation** — Lin, Hu, Sheng,
    Wen, You, Gao (ICLR 2025) [arXiv:2410.18647](https://arxiv.org/abs/2410.18647)
    *Generalization is a power law in distinct environments and objects, not in demonstration
    count. This single result determines how every manifest row should be written (E2).*
13. **So You Think You Can Scale Up Autonomous Robot Data Collection?** — Mirchandani et al.
    (CoRL 2024) [arXiv:2411.01813](https://arxiv.org/abs/2411.01813)
    *The empirical precedent for our whole thesis: data collected at failed-rollout states
    beats random collection. Know it before someone else cites it at us.*
14. **CUPID: Curating Data your Robot Loves with Influence Functions** — Agia et al.
    (CoRL 2025) [arXiv:2506.19121](https://arxiv.org/abs/2506.19121) ·
    [repo](https://github.com/agiachris/cupid) · [site](https://cupid-curation.github.io/)
    *The rigorous version of the question we ask informally. The named competitor to have an
    answer for.*

### Tier 2 — read soon

**Evaluation and testing**
- **Unsupervised Discovery of Failure Taxonomies from Deployment Logs** — Gupta, Ciftci, Bansal
  [arXiv:2506.06570](https://arxiv.org/abs/2506.06570) ·
  [site](https://mllm-failure-clustering.github.io/) — the robotics-specific Domino; the
  comparison arm proposed in E5.
- **Reliable and Scalable Robot Policy Evaluation with Imperfect Simulators (SureSim)**
  [arXiv:2510.04354](https://arxiv.org/abs/2510.04354) ·
  [site](https://suresim-robot-eval.github.io/) — prediction-powered inference; our upgrade
  path from sim CIs to real CIs.
- **FATE-VLA: Failure-Aware Test Generation for VLA Models**
  [arXiv:2606.02307](https://arxiv.org/abs/2606.02307) ·
  [repo](https://github.com/pablovalle/FATE-VLA) — the surrogate-guided sampler of E1, with a
  working replication package.
- **VLATest** (FSE 2025) [arXiv:2409.12894](https://arxiv.org/abs/2409.12894) ·
  [ACM](https://dl.acm.org/doi/10.1145/3729343) — the SE community's fuzzing-and-oracles
  framing of exactly our Phase 2.
- **THE COLOSSEUM** (RSS 2024) [arXiv:2402.08191](https://arxiv.org/abs/2402.08191) — for the
  R²=0.614 sim-real number and the composed-perturbation result.
- **AutoEval** (CoRL 2025) [arXiv:2503.24278](https://arxiv.org/abs/2503.24278) and
  **RoboArena** (CoRL 2025) [arXiv:2506.18123](https://arxiv.org/abs/2506.18123) — the
  competing evaluation paradigms we must position against.
- **A Practical Recipe Towards Improving Sim-and-Real Correlation for VLA Evaluation**
  [arXiv:2606.10366](https://arxiv.org/abs/2606.10366) — the contested sim-real numbers of §4.2.
- **Data Quality in Imitation Learning** — Belkhale, Cui, Sadigh (NeurIPS 2023)
  [arXiv:2306.02437](https://arxiv.org/abs/2306.02437) — action divergence and transition
  diversity; the vocabulary to justify manifest rows.
- **Sentinel** [arXiv:2410.04640](https://arxiv.org/abs/2410.04640) and **FIPER**
  [arXiv:2510.09459](https://arxiv.org/abs/2510.09459) — the erratic/progression taxonomy split
  and conformal calibration.
- **AHA + FailGen** [arXiv:2410.00371](https://arxiv.org/abs/2410.00371) ·
  [repo](https://github.com/NVlabs/AHA) — open-source failure-trajectory generation; the answer
  to PLAN §11 open question 4.
- **Error Slice Discovery via Manifold Compactness**
  [arXiv:2501.19032](https://arxiv.org/pdf/2501.19032) — newer than Domino; check whether it
  wins on our data.

**Practitioner writing**
- **Chris Paxton — What are Robot World Models?**
  [itcanthink.substack.com](https://itcanthink.substack.com/p/what-are-robot-world-models) —
  the clearest practitioner framing of robot data scarcity (10K hours for π0 vs internet-scale
  for LLMs). The "why a Data Gap Manifest at all" argument in plain language.
- **LeRobot v0.4.0 release**
  [HF blog](https://huggingface.co/blog/lerobot-release-v040) — official LIBERO support,
  π0.5/GR00T N1.5 integration, and a new open robot-learning course. Load-bearing for our stack.
- **Eric Jang** [blog.evjang.com](https://blog.evjang.com/) — older, but the "how do you
  evaluate a system with 50% success across thousands of conditions" framing is our problem
  statement, still cited.
- **Figure — Helix** [figure.ai/news/helix](https://www.figure.ai/news/helix) — the industry
  framing of the VLM-universal-but-slow vs policy-fast-but-narrow tension.
- **NVIDIA — Develop Humanoid Robot Policies End-to-End with Isaac GR00T**
  [developer blog](https://developer.nvidia.com/blog/develop-humanoid-robot-policies-end-to-end-with-nvidia-isaac-gr00t/) —
  the heavyweight pipeline view, useful as contrast and for the §7.2 scale-out path.

**Courses and notes**
- **Russ Tedrake — Robotic Manipulation** (MIT 6.4210/6.4212)
  [manipulation.csail.mit.edu](https://manipulation.csail.mit.edu/) — verified current through
  Fall 2026. The perception and "anatomy of a manipulation system" chapters give the vocabulary
  to say whether a failure was perception, planning or control — which is our taxonomy's job.
- **Yunzhu Li — CS231n Lecture 17: Robot Learning** (Stanford, May 2025)
  [slides](https://cs231n.stanford.edu/slides/2025/lecture_17.pdf) — 103 slides, fast current
  overview.

**Data-centric AI**
- **DataComp** (NeurIPS 2023 D&B) [arXiv:2304.14108](https://arxiv.org/abs/2304.14108) —
  *the* template: hold the model fixed, iterate the dataset, measure downstream. The general
  form of what we are doing, and a good structural analogy for the readout.

**Workshops** (all confirmed on the [CoRL 2025 programme](https://2025.corl.org/program/workshops))
- **Eval&Deploy: Evaluation and Deployment in the Robot Learning Lifecycle**
  [eval-deploy.github.io](https://eval-deploy.github.io/) — the workshop-length version of our
  core question. Check the accepted-papers list; it is the highest-yield single page here.
- **Making Sense of Data in Robotics: Composition, Curation, and Interpretability at Scale**
  [site](https://sites.google.com/stanford.edu/corldata25/home) — the closest workshop analogue
  to the Data Gap Manifest.
- **SAFE-ROL: Safe and Robust Robot Learning for Operation in the Real World**
  [site](https://sites.google.com/view/corl-2025-safe-rol-workshop).

### Tier 3 — reference

- **A Survey on VLA Models for Embodied AI** — Ma, Song, Zhuang, Hao, King
  [arXiv:2405.14093](https://arxiv.org/abs/2405.14093) — earliest and most-cited; best for the
  datasets/simulators/benchmarks taxonomy.
- **Pure Vision Language Action Models: A Comprehensive Survey**
  [arXiv:2509.19012](https://arxiv.org/abs/2509.19012) — 300+ papers, organised by
  autoregressive/diffusion/RL/hybrid; most current.
- **A Survey on Efficient VLA Models** [arXiv:2510.24795](https://arxiv.org/pdf/2510.24795) —
  relevant because SmolVLA is explicitly efficiency-driven; useful if we want to argue a
  failure is a capacity trade-off rather than a data gap.
- **Interactive Imitation Learning for Dexterous Robotic Manipulation: A Survey**
  [arXiv:2506.00098](https://arxiv.org/abs/2506.00098) — one citation covering the whole
  DAgger lineage.
- **Robot Policy Evaluation for Sim-to-Real Transfer: A Benchmarking Perspective**
  [arXiv:2508.11117](https://arxiv.org/abs/2508.11117) — names LIBERO among benchmarks under
  external-validity scrutiny.
- **Andrew Ng — The Data-Centric AI Approach**
  [deck](https://go.snorkel.ai/rs/979-SZB-034/images/The%20Data-Centric%20AI%20Approach%20-%20Andrew%20Ng%20-%20Future%20of%20Data-Centric%20AI.pdf)
  **[unverified]** — the most accessible form of the argument, useful for a commercial audience.
- **Diagnosing and Rectifying Vision Models using Language**
  [arXiv:2302.04269](https://arxiv.org/abs/2302.04269) — language-guided slice discovery;
  relevant since SmolVLA has its own VLM backbone that could describe its failures.
- **Open X-Embodiment / RT-X** [arXiv:2310.08864](https://arxiv.org/abs/2310.08864) — the
  canonical mixture-ablation study.
- **Causal Agent Replay** [arXiv:2606.08275](https://arxiv.org/abs/2606.08275) — the
  do-operator vocabulary for our Tier-2 probes.
- **XPolicyLab** [arXiv:2608.09892](https://arxiv.org/html/2608.09892) **[unverified]** — skim
  before freezing our rollout schema, in case a public standard is emerging.

---

## 6. What we could not verify

Listed so that nothing above is read as more solid than it is. Anything in this section should
be checked against a primary source before it appears in a client-facing document.

### 6.1 Numbers read through a summarisation step, not from the source table

Several figures reached this document via a fetch-and-summarise step rather than a direct read
of the paper's tables. They are directionally reliable and should not be quoted precisely.

| Claim | Source | Status |
|---|---|---|
| SIMPLER sim-real Pearson r ≈ 0.92 (range 0.855–0.976), MMRV ≈ 0.056 | [arXiv:2405.05941](https://arxiv.org/abs/2405.05941) | **Check the paper's tables before quoting.** |
| Spearman correlations REALM 0.700 / VLA-Arena 0.575 / SIMPLER 0.400 | [arXiv:2606.10366](https://arxiv.org/abs/2606.10366) | Moderate confidence; read the primary tables. This number carries a lot of weight in §4.2. |
| AGNOSTOS unseen-task rates (π0 17.5%, OpenVLA 13.6%, …) | [arXiv:2505.15660](https://arxiv.org/abs/2505.15660) | Moderate-high (HTML table fetch) but re-read before use. |
| Joint-space ~88.0% vs EE ~89.6%; chunk-wise beats step-wise by ~10% | [arXiv:2602.23408](https://arxiv.org/abs/2602.23408) | Precision uncertain; the *direction* and the *reversal under cross-embodiment* are the load-bearing claims and are stated in the abstract. |
| "Modal per-condition N is 10–20 across 13 VLA papers; none reported CIs" | PhAIL [arXiv:2605.29710](https://arxiv.org/pdf/2605.29710) | **Read from a search summary only.** This is a rhetorically powerful number we plan to quote — verify it properly first. |

### 6.2 Sources read only at abstract or search-snippet level

- **RobustVLA** [arXiv:2510.00037](https://arxiv.org/abs/2510.00037) — abstract only. Its
  modality-based partition of the perturbation space may be a useful alternative to
  LIBERO-plus's factor-based one; unassessed.
- **XPolicyLab** [arXiv:2608.09892](https://arxiv.org/html/2608.09892) — abstract only. Whether
  it genuinely standardises a policy/environment interface (and whether we should conform to
  it) is unknown.
- **LIBERO-PRO** [arXiv:2510.03827](https://arxiv.org/abs/2510.03827) — **title and context
  only, not read.** If it substantively criticises LIBERO evaluation fairness, it bears
  directly on our Phase-2 gate. Highest-value unread item in this list.
- **Cross-Embodiment Offline RL for Heterogeneous Robot Datasets**
  [arXiv:2602.18025](https://arxiv.org/abs/2602.18025) — search snippet only. The negative-
  transfer conditions attributed to it in §4.4 are unconfirmed.
- **RoboFAC** [arXiv:2505.12224](https://arxiv.org/pdf/2505.12224) — skim only.
- **Robot Policy Evaluation for Sim-to-Real Transfer** [arXiv:2508.11117](https://arxiv.org/abs/2508.11117)
  — abstract only; the PDF is image-embedded and no figures were extracted.
- **Open X-Embodiment per-robot transfer breakdown** — the claim in §4.4 that transfer is
  uneven (strongest for small/low-quality donors, weakest or negative for large curated ones)
  is **widely repeated in the field but was not confirmed against RT-X's own tables here.**
- **RynnVLA-002's 97.4% LIBERO claim**, **TurboVLA's 97.7%**, **NanoVLA**, **GigaBrain-O-Small**
  — vendor/author claims with no independent reproduction found. TurboVLA and NanoVLA are too
  recent to have issue history, which is *not* evidence they reproduce.

### 6.3 Non-peer-reviewed sources used

- **Pebblous, "Robot Data Curation & the Closed-Loop Gap" (Jul 2026)**
  [link](https://blog.pebblous.ai/report/robot-data-curation-closed-loop-gap/en/) — a **vendor
  blog**. Its headline negative result (93 targeted demos dropping closed-loop success 73% →
  43% with no change in offline loss) is quoted twice in this document because it is exactly
  the risk our Phase 5 guards against, but **it is unverified and comes from a party with a
  commercial interest in the framing.** Do not put it in a client deck without either
  reproducing the effect ourselves or finding a peer-reviewed equivalent. Its claim that no
  peer-reviewed work closes the full loop is consistent with what this survey found
  independently, but consistency is not confirmation.
- **TRI "Statistical Thinking" Medium post** — returned HTTP 403 to automated fetch. Almost
  certainly bot-blocking rather than a dead link (the companion *Science Robotics* paper is
  solid), but not confirmed.
- **Andrew Ng data-centric AI deck**, **TechRxiv VLA review**, **1X World Model Lab page** —
  surfaced consistently in search, not independently fetched.

### 6.4 Things this survey did not establish and that matter

1. **Whether anyone has published the full closed loop.** The claim in §2.7 that nobody
   produces a costed, prioritised, evidence-backed manifest *and* validates it by retraining
   is an **absence-of-evidence claim** from a few hours of searching. It is the project's
   central novelty claim and it deserves a proper systematic search — at minimum a forward
   citation sweep on CUPID, Demo-SCORE and Mirchandani et al. 2411.01813 — before being
   asserted to a client.
2. **Whether our seven failure families are separable in practice.** §1.3 and E5 argue
   "distribution shift" overlaps the other six. That is an analytical objection, not a
   measurement. The κ study will reveal it; the discovered-taxonomy comparison would reveal
   *why*.
3. **Whether LIBERO-plus's difficulty-level definitions changed between arXiv v1 and the CVPR
   2026 camera-ready.** Our proposal cites v1 numbers. Unchecked, and it affects how L1–L5
   map onto our boundary definitions.
4. **Whether MiniVLA or VLA-Adapter actually reproduce.** §3.2 recommends evaluating them
   partly because no failed-reproduction issue was found — which is weak evidence, since both
   have far less community traffic than SmolVLA. The recommendation is to *test* them, not to
   assume they are better.
5. **The reproducibility of our own counterfactual probes.** Nothing external bears on this.
   `reproducibility_floor()` (PLAN G5, Wk 3 day 1) remains the gate, and §1.4 argues the
   ~600-episode probe budget should be re-derived from its result rather than assumed.
6. **Licence status of COLOSSEUM, SimplerEnv, AGNOSTOS, GemBench, BEHAVIOR-1K and
   RoboCasa365.** Marked unverified in §3.3. For a commercial deliverable this must be
   settled before any of them is built on — RLBench's academic-only terms likely propagate to
   COLOSSEUM, which is otherwise one of the most methodologically attractive options here.

---

# ADOPTION LEDGER — what we took, what we didn't, and why

**Added 2026-09-11 by `primary`, after folding this survey into `PLAN.md`.**
Status of every recommendation. `PLAN.md` is the authority on what we build;
this table is the audit trail for *why*.

| # | Recommendation | Status | Landed in | Rationale |
|---|---|---|---|---|
| **E1** | Surrogate-guided boundary search, with a uniform arm | **ADOPTED** | `PLAN.md` §5.1 | Grid sweep is "redundant and worse". Two arms, never mixed: uniform (~40%) feeds frequency-weighted `severity`, adaptive (~60%) feeds `boundary`. The bias trap is the reason for the split. |
| **E2** | Diversity axes, not demonstration counts | **ADOPTED** | §9.1, schema | Lin et al. ICLR 2025. Turns an instinct we already had into an evidence-backed requirement. The proposal's own example row ("increasing targeted sample budgets") is now flagged as the wrong axis. |
| **E3** | "Published non-data fix?" as a discriminator | **ADOPTED, promoted** | §7c row 7 | Made **mandatory and run FIRST** — cheapest of the seven. Our headline prediction (viewpoint) has a 4K-parameter published fix; marking it `fixability: data` would be indefensible. |
| **E4a** | Paired tests, Bayesian intervals, multiple-comparison correction | **ADOPTED** | §9.2 | We control the seed, so paired tests are free variance reduction, and they relieve the counterfactual-probe budget. |
| **E4b** | PPI-ready schema for SureSim | **ADOPTED as schema-only** | §9.2 | Costs nothing now, expensive to retrofit. We have no hardware, so this is a *future commercial option* ("give us 30 real trials…"), not current work. |
| **E5** | Validate taxonomy against a discovered one, not only κ | **ADOPTED** | §6 | κ only measures human agreement, not whether the taxonomy carves at the joints. Added as a second gate. |
| — | `mujoco<3.4.0` pin | **ADOPTED, urgent** | §5, `FINDINGS.md` F2 | Verified independently against lerobot#4390. We were on the broken side (3.8.1); now 3.3.7. Would have put a physics artifact into a manifest row. |
| — | Policy risk: nine SmolVLA repro issues | **ADOPTED as a risk + fallbacks** | §5, open Q8 | VLA-Adapter and MiniVLA named as 8 GB-compatible fallbacks. Not switching yet — one day of testing budgeted before committing the campaign. |
| — | Three-arm remediation comparison | **ADOPTED — best novel result available** | §6.1 | Data vs augmentation vs frozen-policy wrapper, on one regression set, reporting cost per point recovered. Converts `fixability` from judgement into measurement. |
| — | Report regression on non-targeted cells | **ADOPTED** | §6.1 | Driven by the *[unverified]* vendor claim that 93 targeted demos dropped closed-loop success 73%→43% with no offline signal. Cheap insurance; makes Phase 5 non-negotiable. |
| — | Honest positioning: integration, not research novelty | **ADOPTED** | §0b | Three things survive as ours. Stated up front rather than left for a reviewer to discover. |
| — | `language_grounding` may be empty by construction | **ADOPTED as a reporting rule** | §6 | LIBERO-Plus reports models largely ignore language. If the family never fires, say so rather than implying we looked and found nothing. |
| — | Runtime monitor (§7b.4) is not novel | **ADOPTED as a demotion** | §7b.4 | Sentinel, FIPER, Hide-and-Seek occupy it. Kept as possible productisation; explicitly not pitched as research. |
| — | Severity ordering transfers worst to real | **LOGGED, unresolved** | open Q9 | Sim-to-real rank correlation is contested (Spearman 0.4–0.7 **[UNSOURCED]**) and severity — the thing we prioritise by — transfers worst. We do not yet know how to caveat this in a client-facing manifest. |

## Deferred — not rejected

| Item | Why deferred | Revisit when |
|---|---|---|
| Sampled cache audit (~2% of hits) | Unavailable once traces stop being bit-reproducible on a real VLA | If we ever need to verify a store we cannot re-run |
| Adopting a published clustering method as a comparison arm | Our differentiator is determinism and auditability, not clustering novelty. Worth a comparison arm, not worth replacing tier 1. | Phase 3, if the discovered-taxonomy check (E5) shows our families are arbitrary |
| `lerobot/smolvla_robocasa` second benchmark | Cheapest possible generality claim, but a distraction before Phase 2 reproduces *anything* | After the LIBERO gate resolves |
| SureSim PPI in anger | Requires real hardware trials | If a client brings a robot |

## Rejected

| Item | Why |
|---|---|
| Pitching the runtime monitor as novel research | Fully redundant. Would not survive a knowledgeable reviewer. |
| Replacing our deterministic tier-1 classifier with an LLM-based one | The manifest's claims are quantitative. A non-reproducible component in the measurement path destroys the credibility we are selling. `PLAN.md` §7 already settles this; the survey does not change it. |

## What this survey did NOT change

Worth recording, so silence is not read as agreement-by-omission:

- **The L0–L4 architecture and the model-agnostic rollout contract.** Nothing in
  the survey argues against it, and the "evaluate *your* model" positioning
  depends on it.
- **Phase 0 supply-side analysis first.** Rated as correct ordering.
- **Treating stress generation as integration, not contribution.** Explicitly
  confirmed as the right call.
- **The deterministic-first diagnosis hierarchy** (§7).
- **Phase 5 as the thing that makes the manifest evidence rather than opinion.**
  The survey independently reaches this and calls it "rare. Keep this."
