# FAILURE MINING METHODS — how people actually detect, localise, classify, cluster and attribute failures in robot policy rollouts

> ## ⚠ INCOMPLETE — stopped 2026-09-15 at 1,403 lines
>
> Research was halted mid-document to conserve budget. **Stages 1 (Detection)
> and 2 (Localisation) are complete and usable** — 113 URLs, eleven
> localisation subsections. The agent had absorbed its source material and was
> beginning the classification section when stopped.
>
> **STILL TO WRITE:**
> - §3 CLASSIFICATION — published failure taxonomies (category sets, sizes,
>   hand-designed vs discovered), rule/learned/VLM classifiers, inter-annotator
>   agreement figures, **and how anyone validates a taxonomy is correct rather
>   than merely self-consistent** — this is our open problem #2
> - §4 CLUSTERING — embedding spaces, choosing cluster count, naming clusters,
>   evaluating cluster quality without ground truth
> - §5 ATTRIBUTION — counterfactual/interventional vs correlational, input
>   ablation, saliency unreliability in VLAs
> - Cross-cutting: real-robot vs sim (partially covered in §2.10)
> - Adjacent fields worth stealing from: software fault localisation, driving
>   scenario analysis, LLM agent trajectory analysis
> - "What we could adopt" and "what could not be verified"
>
> To resume: re-dispatch with the original brief (recorded in `HANDOFF.md`),
> scoped to §3–§5 only, telling it §0–§2 already exist here.


*A methods survey. Compiled 2026-09-15.*

> **What this document is, and what it is not.** `docs/LANDSCAPE.md` answers *"who else
> does this and are we redundant?"* — it is a competitive survey. This document answers a
> different question: **how do these methods actually work?** For every method below the
> aim is the *mechanism*: what signal it keys on, what it computes, what it needs to be
> available, and where it breaks. Where a method has a published negative result, that is
> reported in preference to its headline number. Negative results are the most valuable
> content here.
>
> There is deliberate overlap in *names* with LANDSCAPE §1.3/§1.4 (RoboFAC, AHA, Sentinel,
> FIPER, CAR, Gupta/Ciftci/Bansal). There should be no overlap in *content*: LANDSCAPE says
> what they are, this says how they work.

**Verification marks used throughout.** Research at this volume cannot all be read at full
text. Every claim carries one of:

| Mark | Meaning |
|---|---|
| **[F]** | Full text (HTML/PDF) retrieved; numbers read off the source |
| **[A]** | Abstract or landing page only |
| **[S]** | Search-snippet level only — re-check before citing |
| **[?]** | Could **not** be verified; listed because the lead is worth chasing |

A standing caution: a large fraction of the most on-point work is 2026 arXiv preprints.
Several are single- or two-author, and none of the 2606–2609 IDs below are known to be
peer-reviewed. Weight them accordingly. §9 lists everything that could not be verified.

---

## 0. Why identifying a failure is hard at all

Suppose a robot arm is told *"put the black bowl on the plate"*, and 300 timesteps later
the episode ends. Did it work?

The naive answer is that you look. The practical answer is that at the scale this project
operates — thousands of rollouts across a perturbation sweep — you cannot look, so you need
a *checker*, and every checker is a classifier with an error rate that nobody has measured.

**There is no ground truth. There are only checkers.** In simulation the checker is a
symbolic predicate: LIBERO's BDDL goal is a conjunction of ground literals like
`On(bowl, plate)`, and each literal is a Python function over MuJoCo state — bounding-box
containment plus a contact query, or a joint-angle threshold **[F]**. That feels like ground
truth because it is exact and deterministic. It is not: it is *one person's operationalisation*
of "on". MANGO built an independent symbolic oracle for the same LIBERO_10 tasks using LLM
agents and compared the two. They agree at **F1 0.841** — meaning roughly **11–16% of
episodes are labelled differently depending on which symbolic checker you run** **[F]**. Two
exact, deterministic, privileged-state oracles for the same tasks disagree on one episode in
seven.

The disagreements are not random. MANGO documents a LIBERO_10 case where the object is
accidentally dropped outside the cabinet, and *closing the cabinet door pushes it in* — the
end-state predicate fires and the episode is scored a clean success **[F]**. Symbolic goals
encode *what outcome was reached*, never *how*, so a violent, absurd or lucky trajectory that
terminates in the right pose is indistinguishable from competent behaviour.

**Moving to a learned or VLM checker makes this worse, not better.** FailBench (Sept 2026)
pooled 2,197 manipulation attempts from 14 sources — 12 real, 2 sim, with **75% of failures
occurring naturally** rather than being synthetically induced — and scored 13 VLM judges by
macro balanced accuracy. The best, Gemini 3 Flash, reaches **0.77**. Purpose-built failure
detectors do worse: RoboFAC-7B scores **0.51**, FailSense-Calvin-3B **0.50** — chance. On
contact-rich assembly **no model exceeds 0.60** **[F]**. And the errors are asymmetric: VLM
judges show *"a systematic bias toward predicting success under ambiguous evidence, which
persists even with increased reasoning effort"* **[F]**. The reference point is that two
human experts labelling robot rollouts under a shared rubric agree on **97.0%** of outcomes
**[S]**. There is a twenty-point gap between the best automatic checker and the human floor,
and it points in the direction that silently inflates every success rate you report.

**This compounds.** A detector with sensitivity 0.85 and specificity 0.69 applied to a policy
whose true success rate is 50% *estimates* 58%. That eight-point inflation is larger than
most claimed method improvements in the VLA literature.

**And detection is the easy stage.** The same two human annotators who agreed on 97% of
outcomes agreed on only **91% of primary failure phases** **[S]** — *where* it broke is
already harder than *whether* it broke. By the time you reach *why*, the numbers collapse:
the best step-level failure attribution on the Who&When LLM-agent benchmark is **14.2%
accuracy**, with some methods **below random** **[A]**; on TRAIL the best long-context model
scores **11%** **[S]**.

This is the shape of the problem. Each of the five stages below is strictly harder than the
one before it, the ground truth for each stage is itself manufactured by a fallible process,
and the errors at every stage flow downstream into the next. Failure mining is not a pipeline
of solved components; it is a pipeline of estimators, each of which needs its own error bar.

The one structural advantage available in simulation — and it is a large one — is that **we
assign the treatment**. When the experimenter sets the camera yaw, the object pose or the
lighting, there is no confounding by construction, and the resulting causal claim needs no
identification assumptions beyond "the rest of the simulator was held fixed" (§5.6). Almost
every defensible causal claim in this survey comes from that fact, and almost every
undefensible one comes from forgetting it.

---

## 1. DETECTION — "did it fail?"

Detection splits along three axes that matter more than the method families do:

1. **Signal**: simulator state (predicates) → reward scalar → human eyes → learned classifier
   on pixels → VLM judge → policy-internal signals (actions, features, embeddings).
2. **Timing**: *post-hoc* ("was this a failure?") vs *runtime/early* ("will this fail?").
   These are different problems with different metrics — accuracy/F1 versus an
   **F1-versus-detection-time Pareto frontier**. Citing an early-detection paper as evidence
   about post-hoc accuracy (or the reverse) is a common and invisible error.
3. **Cost of being wrong**: a false success silently poisons the numerator of every reported
   success rate, and if rollouts are recycled as training data it poisons the dataset too.

### 1.1 Symbolic goal predicates (BDDL / PDDL / simulator state)

**LIBERO** — [arXiv:2306.03310](https://arxiv.org/abs/2306.03310) ·
[code](https://github.com/Lifelong-Robot-Learning/LIBERO)

*Mechanism* **[F]**. Each task ships a `.bddl` file with `(:objects …)`, `(:init …)` and a
`(:goal …)` expression. The goal is a conjunction (occasionally a disjunction) of ground
literals over unary predicates (`Open(X)`, `TurnOn(X)`) and binary spatial predicates
(`On(A,B)`, `In(A,B)`). Each predicate is a Python function evaluated against live MuJoCo
state. Read directly from
[`base_predicates.py`](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/predicates/base_predicates.py) **[F]**:

| Predicate | Actual test |
|---|---|
| `On(a,b)` | `arg2.check_ontop(arg1)` |
| `In(a,b)` | `arg2.check_contact(arg1) and arg2.check_contain(arg1)` |
| `Stack(a,b)` | contact **and** containment **and** `z(arg1) > z(arg2)` |
| `Up(a)` | `arg1.get_geom_state()["pos"][2] >= 1.0` — an **absolute world-frame z threshold** |
| `InContact(a,b)` | `arg1.check_contact(arg2)` |
| `Open/Close/TurnOn/TurnOff` | joint `qpos` compared against per-object `default_open_ranges` / `default_close_ranges` (e.g. microwave `[-2.094, -1.3]` / `[-0.005, 0.0]`; short cabinet `[0.10, 0.16]` / `[-0.005, 0.0]`) — **and the comparison direction flips per object** |

Two things in that table deserve attention for our purposes. `Up` is a **hardcoded absolute
world z**, so any perturbation that moves the table or the robot base invalidates it silently.
And the articulated-object thresholds are **per-object hand-tuned ranges with object-dependent
comparison direction** — a per-asset config file, not a physical law.

*Requires*: privileged simulator state (body poses, joint positions, contact list). Nothing
learned, no human, no language grounding beyond what the BDDL author wrote.

*The "hold for ≥10 steps" detail* **[?]**. Multiple secondary sources state LIBERO requires
the goal predicate to hold continuously for ≥10 timesteps to avoid transient credit. This
could **not** be confirmed against `libero/libero/envs/bddl_base_domain.py` and should be
verified by direct source read. Note separately that eval harnesses commonly insert a
`num_steps_wait ≈ 10` no-op at episode start for physics settling — including our own
`vla_harness/envs/libero_env.py` — which is a *different* 10 and trivially conflated.

**Known pathologies of predicate checking**

| Pathology | Evidence |
|---|---|
| **Goal satisfied by accident / physics side-effect** | MANGO on LIBERO_10: *"in many successful executions, the object was accidentally dropped outside the cabinet next to the door; therefore, closing the door pushed it inside, resulting in a successful task execution."* The end-state oracle scores this a clean success. **[F]** [arXiv:2606.24815](https://arxiv.org/abs/2606.24815) |
| **Predicate presupposition false at reset** | LIBERO [issue #131](https://github.com/Lifelong-Robot-Learning/LIBERO/issues/131): in `pick_up_the_black_bowl_between_the_plate_and_the_ramekin…`, some sampled init states do not place the bowl between them, so the *instruction* is unsatisfiable-as-written while the *predicate* remains checkable and grantable. Instruction and predicate have drifted apart. **[S]** |
| **Outcome-only — says nothing about *how*** | BDDL encodes only what outcome is achieved. Violent, unsafe or absurd trajectories ending in the right pose are successes. This is the explicit motivation for VISOR, MANGO and Eval-Actions. **[S]** |
| **Transient satisfaction then undone** | Mitigated by the hold window where one exists; where the check is "ever true during the episode", a bowl that passes through the goal region and rolls off counts. **No paper measures this rate on LIBERO.** **[?]** |
| **Success for the wrong reason** | LIBERO-Plus ([2510.13626](https://arxiv.org/abs/2510.13626)): models *"tend to ignore language instructions completely"*, while 95% success collapses below 30% under modest viewpoint or init-state perturbation **[A]**. LIBERO-PRO ([2510.03827](https://arxiv.org/abs/2510.03827)): >90% under standard LIBERO → **0.0%** under their generalised setting; models *"persist in executing grasping actions when the target object is replaced with irrelevant items"* **[A]**. Neither claims the checker is broken — they show the checker cannot distinguish competence from memorisation. |
| **Reproducibility spread under a fixed checker** | OpenVLA LIBERO-Object: reported 88.4%, independently reproduced at **69.4%** (−19.0 pts), with seeds and flash-attention accounting for ~1 pt **[S]** ([openvla#282](https://github.com/openvla/openvla/issues/282)). VLA-0: reported 94.7, reproduced 92.2–92.7 **[S]**. The checker is deterministic, so this spread is *all* harness/policy/seed — it is the noise floor you must clear before attributing anything to a detector change. |

**RLBench** — [arXiv:1909.12271](https://arxiv.org/abs/1909.12271) ·
[code](https://github.com/stepjam/RLBench). Each task registers `success_conditions`
evaluated every physics step: `DetectedCondition(obj, proximity_sensor)` (an invisible
proximity volume placed at the goal — a virtual wall inside a drawer, a detector box over a
target square), `GraspedCondition`, `NothingGrasped`, `JointCondition(joint, Δ)`. Success =
AND of conditions at any timestep **[S/F-tutorial]**. *Breaks*: the proximity volume is a
hand-tuned box; too loose grants near-misses, too tight denies correct outcomes.
`NothingGrasped` conjuncts make "did the right thing but never released" a failure. No
published audit of RLBench condition correctness was found **[?]**.

**BEHAVIOR-1K / OmniGibson** — [arXiv:2403.09227](https://arxiv.org/abs/2403.09227) ·
[challenge rules](https://behavior.stanford.edu/challenge/evaluation.html). Same BDDL
formalism over a richer state vocabulary (`Cooked`, `Soaked`, `Sliced`, `Frozen`, `Covered`,
`Filled`) backed by particle systems and heat-source proximity integrators. **Partial success
= (# goal predicates satisfied at end) / (total # goal predicates)** **[S/A]**. *Breaks*:
partial credit is sensitive to *how the author decomposed the goal into literals* — one
conjunct of five gives a different curve than five of one; end-of-episode-only evaluation
discards transiently-achieved subgoals; `Cooked` is a proximity integrator, not physics, so it
can be satisfied by holding an object near a burner.

**The general lesson.** The failure surface of symbolic checking is always the *lifting* from
continuous state to ground literals, never the logic: threshold choice, frame choice
(world vs parent-relative), AABB vs mesh containment, and whether contact is required. A
predicate written as AABB z-overlap accepts an object hovering 1 mm above a plate in a
gripper; one written with a contact requirement rejects a correct placement on a soft surface.
**Every symbolic checker is a hand-written classifier with unmeasured FP/FN rates.**

### 1.2 Reward thresholds and the truncation confound

A sparse success flag `r_t = 1[goal predicate]` with episode success `any(r_t)` or `r_T` is
identical to §1.1 with the same pathologies, *plus* one that is almost never stated in papers:
`any(r_t)` explicitly **licenses** transient success and `r_T` explicitly **discards** it.
Which one a harness uses changes reported rates on tasks with unstable end states.

**Truncation is not failure.** Pardo et al., *Time Limits in Reinforcement Learning*, ICML
2018 — [arXiv:1712.00378](https://arxiv.org/abs/1712.00378). An episode ends by *termination*
(absorbing state) or *truncation* (external step cap). Treating truncation as termination
substitutes 0 for a non-zero bootstrap value at every horizon boundary, corrupting Bellman
updates; the fix is partial-episode bootstrapping plus feeding remaining time into the
observation. The reported degradation from conflating them is "20–40% on standard MuJoCo
benchmarks" **[S]** — verify before quoting.

*Why this matters for detection, not just training*: our LIBERO wrapper's per-suite step caps
(280/280/300/400/520) make "failure by timeout" a **definitional** category. A slow-but-correct
policy is recorded identically to a wrong one. Any Stage-1 taxonomy must keep `TERM_TIMEOUT`
and `TERM_FAILURE` distinct, and note that the caps are part of the task identity: changing
them changes the reported success rate with no change to the policy.

### 1.3 Human labelling

*Cost, measured.* AutoEval ([arXiv:2503.24278](https://arxiv.org/abs/2503.24278)) **[F]**: a
human doing manual resets needs ≈**16 hours** to run the trial count AutoEval completes in a
day; AutoEval needed **3 human interventions across 24 continuous hours** (~3 minutes of
operator time), a **>99% reduction**. OpenVLA's real evaluation reportedly cost ~**100 human
hours for 2,500 rollouts** (~2.4 min/rollout) **[A]**. RoboDojo: 180 physical trials across 18
tasks in **202 min** **[A]**.

*Agreement, measured.* The only inter-annotator numbers found for robot rollouts are
**RoboLineage** ([arXiv:2606.22142](https://arxiv.org/abs/2606.22142), **[S]** — snippet only):
two experts independently labelling under a common rubric agreed on **97.0% of task
outcomes, 91.0% of primary failure phases, 98.0% of admission decisions**. Note the structure —
**outcome agreement (97%) is far higher than cause agreement (91%)**. Stage 1 is the easy part.

*Where it breaks.* (a) 3% outcome disagreement is a hard floor on any detector's measurable
accuracy, and is the same order as differences papers routinely claim; (b) rubric drift;
(c) **ambiguous rollouts — partial placement, object dropped then recovered, task done but
something knocked over — are exactly the ones where both humans and VLMs fail**, so agreement
measured on random rollouts *overstates* agreement on the rollouts that matter; (d) **no
published Cohen's κ or Krippendorff's α for robot-rollout success labelling was found** **[?]** —
the RoboLineage figures are raw percent agreement, which overstates κ.

### 1.4 Learned success classifiers and video reward models

**SuccessVQA / "VLMs as Success Detectors"** (DeepMind, CoLLAs 2023) —
[arXiv:2303.07280](https://arxiv.org/abs/2303.07280). *Mechanism*: reframe success detection
as VQA. Feed a pretrained **Flamingo** a short clip (or first+last frame) plus *"Did the agent
successfully {task}?"*; fine-tune on human reward annotations; read the yes/no token. Uniform
interface across domains — no per-task reward head. *Requires*: a large pretrained VLM and
human binary reward annotations. *Reported*: beats bespoke per-task reward models on OOD
splits with visual or language variation; authors concede unseen real video is *"an even more
challenging generalisation task"* **[A]**. Numeric tables could **not** be extracted **[?]**;
the one external datapoint is GVL's comparison table listing SuccessVQA at **0.62 accuracy /
0.33 precision / 0.73 recall** — note the very low precision, i.e. heavy false-success.

**GVL — Generative Value Learning** (ICLR 2025) —
[arXiv:2411.04549](https://arxiv.org/abs/2411.04549) ·
[site](https://generative-value-learning.github.io/). *Mechanism* **[F]**, and the trick is
the whole paper: naively asking a VLM for a per-frame value fails, because **temporal
correlation between successive frames** makes the model emit a monotone ramp. GVL therefore
**shuffles the video frames**, keeps the first frame as an unshuffled anchor, and asks the VLM
to autoregressively predict task-completion percentage over the shuffled sequence. Making the
task *harder* forces semantic and temporal grounding rather than visual continuity. Unshuffle
to recover the value curve. Zero training. *Requires*: a long-context VLM, video frames, a task
string. No privileged state, no reward, no goal image. *Metric*: **Value-Order Correlation
(VOC)** = rank correlation between predicted values and true frame order. *Numbers* **[F]**:
success detection **zero-shot 0.71 acc / 0.71 prec / 0.71 rec**; **one-shot 0.75 / 0.85 /
0.70**; SuccessVQA 0.62 / 0.33 / 0.73. Dataset-quality ranking by VOC: RT-1 **0.74**, Dobb-E
0.53, Bridge 0.51, QT-OPT 0.19, RoboNet **−0.85**. ALOHA bimanual: zero-shot **median VOC
0.12**, one-shot 0.37 — **zero-shot GVL is near-useless on long-horizon bimanual**.
*Breaks* (authors'): periodic/repetitive tasks (wiping, stirring) where frame order is
genuinely ambiguous; camera dependency — poor top-down resolution produces noisy values.
**Read: 0.71–0.75 is the state of the art for training-free success detection, and it is not
close to the ~97% human floor.**

**VIP / LIV / R3M / RoboCLIP** — the embedding-distance family.
**VIP** ([arXiv:2210.00030](https://arxiv.org/abs/2210.00030)) learns an embedding φ on
action-free human video (Ego4D) such that `−‖φ(o_t) − φ(g)‖₂` is an implicit value; success
detection thresholds that distance. Requires a **goal image**. *Breaks*: the distance is not
calibrated across tasks or viewpoints so a single threshold does not transfer, and
visually-similar-but-wrong end states (right object, wrong receptacle) sit close in embedding
space → false successes. **LIV** ([arXiv:2306.00958](https://arxiv.org/abs/2306.00958))
generalises VIP to vision-*language*, so the goal can be a string — a major practical
advantage — at the cost of needing in-domain fine-tuning for reward quality **[S]**. **R3M**
([arXiv:2203.12601](https://arxiv.org/abs/2203.12601)) is a feature extractor, consistently
beaten by VIP as a reward; include for lineage only **[S]**. **RoboCLIP**
([arXiv:2310.07899](https://arxiv.org/abs/2310.07899)) encodes the *whole episode* and the task
specifier (one demo video or a text string) and emits their cosine similarity as a single
episode-end reward. *Breaks*: episode-level similarity is dominated by scene appearance, is
high for visually-similar failures, and has **no temporal resolution at all** **[S]**.

**RoboReward** (2026) — [arXiv:2601.00675](https://arxiv.org/abs/2601.00675). Notable for the
*mechanism by which it manufactures negatives*, since **OXE contains essentially no failures**:
**counterfactual relabeling** (re-caption a successful episode against a task it did not
accomplish) and **temporal clipping** (truncate a success to create calibrated partial-progress
near-misses) **[A]**. Cross-reference that matters: FailBench scores **RoboReward-8B at 0.62**
macro balanced accuracy on out-of-domain failures — *below* general-purpose Gemini 3 Flash at
0.77. Specialisation did not generalise.

**AutoEval's PaliGemma classifier** — [arXiv:2503.24278](https://arxiv.org/abs/2503.24278) ·
[site](https://auto-eval.github.io/). This is the best-documented example of a learned success
detector good enough to *run an evaluation*, and the price is instructive. *Mechanism* **[F]**:
per-scene **PaliGemma-3B** fine-tuned with quantised LoRA (lr 2e-5, batch 4, 80 iterations) on
≈**1,000 manually labelled images per scene**, into **three** classes — success / failure /
**invalid** (the third catches unresettable states). Deployed only after clearing a **>95%
accuracy** bar reached by iteratively mining misclassified observations. A separately
LoRA-finetuned reset policy (50–100 teleop demos, >95% reliability) restores the scene.
*Reported*: **Pearson r = 0.942** against human-run evaluation, **MMRV = 0.015** (rankings
essentially never inverted); ~500–850 episodes per 24 h; WidowX cell **<$3,500**; setup 1–3 h.
*Stated limits*: binary success only (no partial credit — which kills fine-grained mining);
does not support varying camera angle, lighting or distractors; the >95% bar is **in-scene** —
a policy failing in a novel visual way is precisely what the classifier never saw.

**Fine-tuned VLM classifiers** — *Assessing VLMs for Failure Detection in Robotic
Manipulation*, [IEEE Pulse](https://www.embs.org/pulse/articles/assessing-vision-language-models-for-failure-detection-in-robotic-manipulation/)
**[S]**: MiniGPT-4 **89.33%**, Florence-2 **88.86%** overall — but **successes detected at
82–99% and failures at only 71–84%**; models "favor success prediction". With human escalation
they automate **>59% of decisions at 95% system accuracy**. The expensive error (missed
failure) is the one they make most.

### 1.5 VLM-as-judge for success detection

**FailBench** (Sept 2026) — [arXiv:2609.03611](https://arxiv.org/abs/2609.03611). The cleanest
cross-domain measurement available.

*Construction* **[F]**: 2,197 manipulation attempts from **14 public sources** (12 real, 2 sim:
rh20t, armnetbench, robometersim, simplerenv, botfails, ur5fail, reassemble, robometer,
bdv2fail, roboarena, roboreward, phail, robofac, reflect). **75% of failures occur naturally.**
Detectors get video + task string; scored by **macro balanced accuracy** (mean of TPR and TNR,
averaged over sources), deliberately robust to the class imbalance that inflates plain accuracy.

*Results, all 13 detectors* **[F]**:

| Class | Detector | Macro balanced acc |
|---|---|---|
| General-purpose | Gemini 3 Flash | **0.77** |
| | Gemma-4-31B-it | 0.75 |
| | Qwen3-VL-8B-Thinking | 0.69 |
| | GPT-4o | 0.69 |
| | Qwen3-VL-2B-Thinking | 0.53 |
| Embodied | Hy-Embodied-VLM-1.0 | 0.61 |
| Purpose-built failure detectors | Guardian | 0.63 |
| | RoboReward-8B | 0.62 |
| | ViFailback-8B | 0.59 |
| | **RoboFAC-7B** | **0.51** |
| | **FailSense-Calvin-3B** | **0.50** (chance) |

Three findings matter more than the table:
- **Contact-rich assembly (`reassemble`): no model exceeds 0.60.**
- **Specialisation hurts.** *"All fine-tuned specialists that allow a direct comparison perform
  worse than their own base models."* RoboFAC-7B **0.514 vs its own base model's 0.555**.
- **Optimism bias and confabulation.** A systematic bias toward predicting success under
  ambiguity that *persists with more reasoning effort*; judges "frequently confabulate
  observations unsupported by video"; reasoning traces are **longer when wrong** (1,868 chars
  vs 1,464 when correct) — so response length is a usable, weak wrongness signal.
- One cheap fix: **evidence localisation** (crop the input to outcome-relevant regions) gives
  **+2.3 pts** without retraining (0.773 → 0.797).

**VISOR** — [arXiv:2605.10408](https://arxiv.org/abs/2605.10408). A VLM watches the rollout and
emits a correctness verdict plus a 3-way quality grade, with self-reported uncertainty.
**1,032 videos** (516/516 balanced) across 4 tasks from 3 VLA models. *Numbers* **[F]**:
GPT-4.1 precision **0.775–0.980**, recall **0.251–0.746**, F1 0.386–0.842; Gemini-2.5-Flash
precision 0.561–0.782, recall 0.730–0.869, F1 0.635–0.759. **GPT is precision-biased, Gemini
is recall-biased — the same rollouts, judged by two frontier VLMs, give materially different
success rates.** Quality grading is far worse (best F1-micro 0.483). **Uncertainty is useless**:
DeepGini 0.0003–0.0733, MSP 0.92–1.0 — "low uncertainty does not correlate with correctness."
Cost ≈$0.03 and ~7 s per video.

**MANGO** — [arXiv:2606.24815](https://arxiv.org/abs/2606.24815). Generates *symbolic* oracles
rather than judging directly, which is why its agreement numbers are the most useful thing in
this section. *Mechanism* **[F]**: three LLM-agent modules — (I) decompose instructions into a
reusable library of **atomic tasks** (Open, Pick, Place, Close), with a Generator proposing, an
Assessor checking atomicity/redundancy, and a Judge accepting or refining; (II) ground each
atomic task in **simulator API calls** to yield executable oracle definitions; (III) compose
per-step oracles into an ordered chain for a specific instruction. The output is **code**, so
it is cheap, deterministic and re-runnable — but it needs simulator access.
*Numbers* **[F]**: agreement with LIBERO's own oracle — LIBERO_10 **acc 0.825 / prec 0.818 /
rec 0.864 / F1 0.841**; RoboCasa **F1 0.906**. Failure *localisation* exact match **78.95%**
(LIBERO_10) / **83.33%** (RoboCasa). **Executability 0.790 / 0.933** — the residual is LLM
hallucination of non-existent simulator functions.

**Metamorphic testing** — Valle, Segura, Ali, Arrieta,
[arXiv:2602.22579](https://arxiv.org/abs/2602.22579) **[A]**. A structurally different answer
to the oracle problem: define **metamorphic relations** over input transformations and check
whether the transformation *should* change the trajectory. Two relation patterns and five
relations across 5 VLA models, 2 simulated robots, 4 tasks. Explicit motivation is that
current oracles "capture symbolic representations of the world… [but] fail to assess task
quality." Specific numeric results were not in the abstract **[?]**. *Why this is interesting
for us*: metamorphic relations need **no ground-truth success label at all** — they test
consistency, which is exactly the property a perturbation sweep already measures.

**R2S-Eval** — [arXiv:2609.03276](https://arxiv.org/abs/2609.03276) **[F]**. Sidesteps
calibration by judging **pairwise**: (1) a per-video *behaviour description* covering motion
smoothness, temporal continuity, progress and visible contact/effort; (2) **pairwise comparison
of the anonymised descriptions**; (3) Bradley-Terry aggregation. Describing-then-comparing
anonymised text is an explicit mitigation for verbosity and identity bias. *Numbers*: mean
human agreement **82.9% on LIBERO**, **91.9% real-to-sim**, over 8 VLMs; sim-vs-hardware mean
success-rate difference **2.13 pp** with identical ranking; Spearman **ρ = 0.823**.
*Breaks*: produces a ranking, not an absolute rate.

**Code-as-Monitor** (CVPR 2025) — [arXiv:2412.04455](https://arxiv.org/abs/2412.04455). The VLM
writes the checker instead of being the checker. *Mechanism* **[F]**: (1) **ConSeg**, a
constraint-aware open-vocab segmentation model, produces instance- and part-level masks given
constraint text; (2) depth lifts masks to point clouds across multi-view RGB-D; voxelisation +
clustering abstracts each into a compact **constraint element** — a point, line or surface —
numbered and overlaid on the images as visual prompts; (3) **GPT-4o** sees the annotated images
plus textual constraints and generates **executable Python-like monitor code** returning a
boolean flag plus an explanation string. **CoTracker** tracks elements across frames so the
code runs every frame at no further VLM cost. Unifies *reactive* and *proactive* detection as
one spatio-temporal constraint-satisfaction problem. *Numbers*: +17.5% success under severe
disturbance (CLIPort); −38.7% execution time vs Inner Monologue; +28.7% (OmniGibson) with
−52.2% tokens vs DoReMi; +20.4% real pick-and-place. *Breaks*: ConSeg's training data is
largely pick-and-place; depends on good depth; **geometric constraints cannot express
non-geometric goals** (`Cooked`, "is it clean?"); generated code can be wrong and nothing
checks it.

**I-FailSense** (ICRA 2026) — [arXiv:2509.16072](https://arxiv.org/abs/2509.16072) ·
[site](https://clemgris.github.io/I-FailSense/) **[S]**. *Mechanism*: LoRA post-training of a
base VLM, then lightweight **classification heads ("FS blocks") attached at multiple internal
representation depths**, combined by an ensembling/arbitration mechanism. Targets **semantic
misalignment** — the robot does something coherent but not what was asked — a failure mode
symbolic checkers miss when the predicate is loose and VLM judges miss when they judge
plausibility rather than instruction-following. Trained on DROID/CALVIN/AHA; generalises to
broader failure categories and transfers zero-shot or with minimal post-training. Cross-check:
FailSense-Calvin-3B scores **0.50** on FailBench — chance **[F]**.

**Known VLM-judge biases** (general, transfer directly):
optimism under ambiguity **[F]**; confabulation, with verbosity as a weak wrongness signal
**[F]**; **positional bias** (verdicts flip when option order reverses) **[S]**; **verbosity
bias** — "MLLMs are even more vulnerable to verbosity bias than to position bias" **[S]**;
**self-preference** ([arXiv:2410.21819](https://arxiv.org/abs/2410.21819)) **[S]**;
**miscalibration** (VISOR, **[F]**); **informativeness bias** — *When Vision-Language Models
Judge Without Seeing* ([arXiv:2604.17768](https://arxiv.org/abs/2604.17768)) **[S]**, judges
reaching verdicts the visual evidence does not support.

**RoboProcessBench** — [arXiv:2606.13040](https://arxiv.org/abs/2606.13040) **[F]** bounds *why*
VLM judges fail: ~58K QA pairs, 260 tasks, 19 VLMs. **Primitive-local progress 34.4%** (chance
33.3%); **temporal ordering best 22.3%, below chance**; contact detection 41.9–53.5%.
Post-training helps static perception (phase recognition 26.6 → 58.5%, contact 82.7%) but **not
temporal reasoning (17.0% on ordering)**. *If a judge cannot tell which frame came first or
whether contact occurred, its success verdict on a contact-rich task is not grounded in the
video.* This single result explains the FailBench assembly number.

### 1.6 When the success signal itself is unreliable

This is the question with the most decision-relevant content, and it has three parts.

**(a) Silent failures are measurable, and exteroception is what catches them.**
*How Visible Are Silent Manipulation Failures?* —
[arXiv:2606.03134](https://arxiv.org/abs/2606.03134) (single author, UC Berkeley, June 2026;
weight accordingly) **[F, qualitative]**. *Design*: two bimanual ALOHA tasks in sim; failures
induced by **environment perturbation** rather than by flipping labels; then, restricted to
episodes the robot **itself logged as successes**, compare a proprioception-only detector
against a vision detector at recovering the false successes. Crucially, **episodes are labelled
by privileged simulator state (object pose relative to target) which the detectors never
observe** — the ground truth is itself privileged. *Findings*: on **cube transfer**, false
successes are "almost fully recoverable from joint data alone"; on **peg insertion**,
proprioception recovers only part and **vision closes most of the gap**. Conclusion: catching
silent failures generally requires exteroception. *Author's own caveat*: clean-sim results are
an **optimistic upper bound** — the discriminating signals were partly in velocity channels
**below realistic sensor noise thresholds**. Exact per-task percentages could not be extracted
**[?]**.

**(b) The LLM-agent field has the hard numbers, and they are brutal.**
*From Confident Closing to Silent Failure* —
[arXiv:2606.09863](https://arxiv.org/abs/2606.09863) **[F]**. 9,876 tau2-bench trajectories
across 8 model families plus 1,879 AppWorld trajectories across 4. *False-success rates*:
**45–48% of failures in single-control tau2-bench domains, but only 3% in dual-control
telecom** — where an independent process can verify the agent's actions. That is roughly an
**order-of-magnitude suppression from independent verification**, and it is the cleanest
published statement of the underlying principle: *a checker that shares the actor's view
inherits the actor's blind spots.* On AppWorld, **75.8%** of self-assessing coding-agent
trajectories with explicit status claims were false successes. *Detection results, and this is
the part to internalise*: **no LLM-judge configuration across 5 judges, 5 prompt strategies
and full task specifications exceeds AUROC 0.65** on tau2-bench, and the same judges reach
**0.54 on AppWorld**. A plain **TF-IDF detector** on surface features reaches **0.83 and 0.95**
respectively, at vastly lower cost. Judges "rely on surface completion proxies — confident
closing language… and coarse action-sequence volume."

Related: *Beyond Task Completion: Revealing Corrupt Success in LLM Agents through
Procedure-Aware Evaluation* ([arXiv:2603.03116](https://arxiv.org/abs/2603.03116)) **[S]**
formalises **"corrupt success"** — outcome check passes, procedure was invalid — and gates on
four non-redundant axes (Utility, Efficiency, Interaction Quality, Procedural Integrity). The
robotics analogue is MANGO's door-pushed-the-object-in case.

**(c) Reported numbers demonstrably move when you change the checker.**
- Two *symbolic* oracles: MANGO vs LIBERO's own, **F1 0.841** — ~11–16% of episodes labelled
  differently **[F]**.
- Two frontier *VLM* judges on the same 1,032 videos: GPT-4.1 precision-high/recall-low (recall
  to 0.251), Gemini recall-high/precision-low **[F]**.
- Thirteen judges on the same 2,197 attempts: **0.50 → 0.77** spread **[F]**.
- Under a *fixed* checker: OpenVLA LIBERO-Object 88.4 reported vs 69.4 reproduced **[S]**.

**The statistical consequence, and the fix.** A judge with sensitivity/specificity < 1 gives a
**biased** estimate of the true success rate — the naive mean is wrong, not merely noisy. Two
correction families, both borrowed from the LLM-as-judge literature which is ahead of robotics
here **[S]**:
1. **Measurement-error correction** — Rogan–Gladen-style estimators that invert the
   misclassification matrix.
2. **Prediction-powered inference (PPI)** — run the judge on everything, calibrate residuals on
   a small gold human-labelled subset, get an unbiased estimate with tightened CIs.

**Both require human-labelled calibration data**, and model *comparison* is more fragile than
single-model estimation because the calibration assumptions may not hold equally across the
models compared. Sources: [2605.06939](https://arxiv.org/abs/2605.06939),
[2601.05420](https://arxiv.org/abs/2601.05420), [2511.21140](https://arxiv.org/abs/2511.21140),
[2603.00039](https://arxiv.org/abs/2603.00039) — all **[S]**.

> **The single most actionable recommendation in this section**: if you use a VLM judge
> anywhere in the measurement path, report a **PPI- or Rogan–Gladen-corrected** rate against a
> human-labelled calibration slice, together with the judge's measured sensitivity and
> specificity — never the raw judge mean.

### 1.7 OOD / anomaly detection — a different problem, often miscited as the same one

§1.1–1.6 answer *"did it fail?"* from the outcome. This family answers *"is something going
wrong?"* from the policy's own behaviour, often **before** the outcome exists. These detectors
never see success; they see distribution shift, and therefore **cannot distinguish benign OOD
from actual failure**. The correct metric is not accuracy but the **F1-versus-detection-time
Pareto frontier**; a paper reporting only one axis is uninterpretable.

**Sentinel** (CoRL 2024) — [arXiv:2410.04640](https://arxiv.org/abs/2410.04640) ·
[code](https://github.com/agiachris/sentinel). The reference two-detector monitor, and its
*taxonomy* is more valuable than its numbers.
*Mechanism* **[F]**, two complementary detectors:
1. **Erratic failures → STAC (Statistical Temporal Action Consistency).** A chunked generative
   policy re-plans every *k* steps, so successive plans **overlap**. Sample action distributions
   at consecutive inference steps, take the empirical statistical distance between the
   overlapping segments, and accumulate: `η_t := Σ_i D̂(π̄_ik, π̃_(i+1)k)`. `D̂` is instantiated
   as **MMD with RBF kernels**, **forward KL via KDE**, or **reverse KL via KDE**. Intuition: a
   confident policy's re-plans agree; a policy in trouble thrashes. Negligible compute; uses only
   actions, so it is policy-agnostic and needs no vision model.
2. **Task-progression failures → a VLM monitor.** GPT-4o gets frames `I_{0:t}` sampled at
   frequency ν, plus task description, elapsed time and time limit, with chain-of-thought;
   outputs binary {ok, failure}. Catches what STAC *structurally cannot*: a policy that is
   **confidently and consistently doing the wrong thing**.
*Calibration*: threshold γ set to the (1−δ) quantile of STAC scores on **successful** rollouts →
conformal guarantee FPR ≤ δ. Real deployment used only **10 calibration rollouts**.
*Numbers* **[F]**: Close Box (sim) — STAC-MMD TPR 0.99 / TNR 0.93 / det. time 14.72 s / acc
0.96; GPT-4o Video QA TPR 0.69 / TNR 0.95 / 21.02 s / acc 0.87; combined TPR 1.00 / TNR 0.90 /
14.25 s / **acc 0.91**. Real Push Chair — STAC rev-KL 0.80/0.90/9.83 s; GPT-4o 0.90/1.00/12.89 s;
combined 1.00/0.90/9.60 s. Baselines beaten: CLIP/ResNet embedding similarity 0.68–0.79,
diffusion reconstruction error 0.76, output variance 0.72. **Headline: 18% more failures
detected than either detector alone.**
*The trade to notice*: combining **raises TPR to 1.00 but lowers accuracy** (0.96 → 0.91),
because the union of two detectors unions their false positives.
*Requires*: a **generative/chunked** policy you can sample from — it does **not** apply to a
deterministic single-step regressor.

**FAIL-Detect** (RSS 2025) — [arXiv:2503.08558](https://arxiv.org/abs/2503.08558). Failure
detection **without any failure data**. Two stages: (1) distil policy inputs and outputs into
scalar scores capturing epistemic uncertainty (best is `logpZO`, a normalising-flow density on
the observation embedding); (2) frame it as **sequential OOD detection** with **time-varying**
conformal thresholds — the threshold at step *t* is calibrated separately, because nominal score
distributions drift over an episode. *Requires*: successful rollouts only **[S]**.

**FIPER** (NeurIPS 2025) — [arXiv:2510.09459](https://arxiv.org/abs/2510.09459) ·
[code](https://github.com/utiasDSL/fiper). Two indicators: (i) **OOD observations via random
network distillation** in the policy's embedding space (train a predictor to match a frozen
random network on nominal embeddings; prediction error = novelty); (ii) **action-chunk
entropy**. Both conformally calibrated on a small set of **successful** rollouts. **An alarm
fires only when *both*, aggregated over short windows, exceed threshold** — and *that
conjunction is the contribution*: **OOD alone is not failure**; a novel-but-fine object pose is
OOD. Requiring conjunction with action uncertainty is the fix **[A]**.

**SAFE** (NeurIPS 2025, TRI) — [arXiv:2506.09937](https://arxiv.org/abs/2506.09937) ·
[site](https://vla-safe.github.io/). The standard baseline everything else compares to, and its
real-world numbers are the most decision-relevant thing in this subsection.
*Mechanism* **[F]**: the finding is that **VLA internal features already encode task-generic
success/failure information**. Take the latent from the **last layer** of the VLA, feed a 1–2
layer **MLP or LSTM** (kept tiny to limit overfitting), emit a scalar failure score per
timestep; threshold with a **time-varying band from functional conformal prediction** on a
held-out calibration set, trading accuracy against detection time via α.
*Requires*: **white-box access to policy internals** (rules out API-only policies), and **both
successful and failed rollouts** for training — unlike Sentinel/FAIL-Detect/FIPER.
*Numbers, ROC-AUC (seen / unseen tasks)* **[F]**:

| | OpenVLA LIBERO | π₀-FAST LIBERO | π₀ LIBERO | π₀ SimplerEnv | **Real Franka π₀-FAST** | **Real WidowX OpenVLA** |
|---|---|---|---|---|---|---|
| SAFE-LSTM | 70.2 / 72.5 | 93.0 / 84.5 | 77.0 / 71.1 | 88.9 / 80.1 | 77.3 / **58.7** | 84.3 / 71.8 |
| SAFE-MLP | 72.7 / 73.5 | 90.1 / 80.4 | 73.5 / 73.3 | 89.5 / 84.8 | 86.8 / **64.2** | 89.1 / 88.4 |
| Best baseline | 67.1 / 69.5 | 92.1 / 84.6 | 77.2 / 75.2 | 90.2 / 71.3 | 80.4 / 60.3 | 82.4 / 70.0 |

Data collected: LIBERO-10 **500 rollouts** (210 train / 140 eval-seen / 150 eval-unseen); real
Franka **780 rollouts across 13 tasks**; real WidowX 532 rollouts.
**The story is the last two columns.** On **unseen real tasks** SAFE falls to **58.7–64.2
ROC-AUC** — barely above chance. Authors' stated limitations: only last-layer features are used
and multi-layer aggregation is open; generalisation across embodiments and sim2real is
untested. Conformal exchangeability is violated cross-task, so the true negative rate deviates
from 1−α.

**ActProbe** (2026) — [arXiv:2606.08508](https://arxiv.org/abs/2606.08508). Fully **black-box**:
no policy internals, no resampling, single forward pass. *Mechanism* **[F]**: two action-only
signals — **TCE (Temporal Consistency Error)** = MSE between overlapping regions of consecutive
action chunks (the single-sample analogue of STAC, which needs resampled distributions); and
**ACM (Action Chunk Magnitude)** = L2 norm of the current chunk, separating small smooth motion
from the oversized corrections that precede failure. These feed a task-conditioned **bridged
LSTM–MLP of ~24K parameters**: the LSTM path captures integrated behavioural drift, a skip
connection preserves sensitivity to instantaneous anomalies, and task-embedding initialisation
from a frozen language encoder enables multi-task generalisation. Split-conformal calibration.
*Verified empirical finding worth stealing*: **sliding-window aggregation beats both cumulative
accumulation (detects too late) and single-timestep decisions (too many false alarms)**.
*Numbers*: **+12.7% average hypervolume gain on the F1–detection-time Pareto frontier**;
**+9.0% early-stage ROC-AUC (q=0.25)** on unseen tasks vs SAFE-MLP; **75.8% average
early-detection AUC**; **2.9× fewer environment interactions** for RL-finetuning parity.

**Black-box action monitoring as a cheap baseline** — *How VLAs Fail Differently*,
[arXiv:2605.28726](https://arxiv.org/abs/2605.28726) **[A]**. Three motor-command signals —
**direction reversal rate**, **jerk**, **velocity violations** — over 450 episodes:

| Architecture | Direction reversal AUROC | Jerk AUROC | Velocity AUROC |
|---|---|---|---|
| VQ-BeT (discrete) | **0.93** | 0.88 | — |
| Diffusion Policy | 0.79 | 0.69 | — |
| ACT | **0.91** | 0.41 | 0.41–0.52 |

**Direction reversal rate is a universal failure predictor across all three architectures**,
while **velocity monitoring — "the most common safety mechanism in VLA deployment code" —
performs at 0.41–0.52 for continuous architectures, i.e. at or below chance.** This is the
cheapest implementable detector in the entire survey and it beats the thing people actually
deploy.

**KnowNo** (CoRL 2023) — [arXiv:2307.01928](https://arxiv.org/abs/2307.01928) ·
[site](https://robot-help.github.io/). Included to scope it out: KnowNo poses the next-step
decision as **MCQA**, reads per-option token likelihoods, computes a nonconformity threshold on
a calibration set, and triggers human help when the prediction set has **>1 element**. The
conformal guarantee is on *coverage of the correct option*. **This detects ambiguity in the
plan, not failure in the execution** — citing it as a failure detector is a category error.

> **Standing caveat on every conformal method in §1.7.** The coverage guarantee is **marginal
> and conditional on exchangeability with the calibration set**. Deployment distribution shift —
> exactly the condition under which you want a failure detector — voids it. Every paper here
> advertising a "formal guarantee" inherits this.

### 1.8 Detection: synthesis

1. **There is no ground truth, only checkers.** Two independently-built symbolic oracles for
   LIBERO_10 agree at F1 0.841 **[F]**.
2. **Human-human outcome agreement (~97% **[S]**) is the ceiling**, and it is the only number
   in this section above 0.9 describing agreement on outcome.
3. **Best general VLM judge: 0.77 macro balanced accuracy; <0.60 on contact-rich; biased toward
   success** **[F]**. Best training-free value method: **0.71–0.75** **[F]**.
4. **Fine-tuning a VLM for failure detection made every directly-comparable model worse than
   its own base** **[F]**. Purpose-built detectors overfit their own failure distribution.
5. **In the adjacent LLM-agent field, a TF-IDF surface detector (AUROC 0.83/0.95) beats every
   LLM judge configuration (≤0.65/0.54) at detecting false success** **[F]**. Before building a
   sophisticated judge, build the dumb baseline.
6. **Detector errors inflate success rates by more than most claimed improvements.** Correct with
   PPI or Rogan–Gladen against a human-labelled slice.
7. **Independent verification is the only order-of-magnitude effect anyone has measured**
   (45–48% → 3% false success) **[F]**. In robotics the analogue is exteroception.
8. **Keep post-hoc detection and early prediction separate**, with metrics to match.
9. **Truncation is not failure.**

---

## 2. LOCALISATION — "where/when did it break?"

Localisation answers *"at which timestep t\* did this rollout go wrong?"* Methods divide by
the signal they key on:

| Family | Signal | Needs a reference? | Transfers to a real robot? |
|---|---|---|---|
| **(A) Segmentation / phase** | structure of the *nominal* trajectory | trained on demos | mostly (proprioceptive); vision segmenters degrade |
| **(B) Change-point / dynamics** | a statistical break in the signal itself | no | yes |
| **(C) Progress / value** | a scalar that should increase monotonically | goal image or task text | yes |
| **(D) Irrecoverability** | forward dynamics / reachability | a model or world model | usually **no** — needs privileged state or a learned world model |

> **The single most important finding of this section.** Family (A) is trained on curated
> *successful* demonstrations and is therefore out-of-distribution on exactly the rollouts you
> want to localise. Every 2026 method that works well — FailureSpot, Hide-and-Seek, Foresight,
> Rewind-IL, ActProbe — has **abandoned demo-referenced segmentation** in favour of (B)/(C)-style
> self-referential signals computed from the *policy's own* internals. If our L3 phase
> segmenter is the primary localiser, we are building the thing the field moved away from.

### 2.1 Temporal action segmentation (TAS) — the wrong default tool, with reasons

**MS-TCN / MS-TCN++** — [arXiv:1903.01945](https://arxiv.org/abs/1903.01945) ·
[arXiv:2006.09220](https://arxiv.org/abs/2006.09220). *Mechanism* **[F]**: a stack of *S*
stages, each a sequence of dilated residual layers with dilation doubling per layer —
`Ĥ_l = ReLU(W₁ * H_{l-1} + b₁)` with `W₁ ∈ ℝ^{3×D×D}` (kernel 3, dilation 2^l), then
`H_l = H_{l-1} + W₂ * Ĥ_l + b₂`. Stage 1 predicts per-frame class posteriors; stages 2..S take
the previous stage's softmax and refine it. *Losses*: cross-entropy plus the **truncated MSE
over-segmentation penalty** — `Δ_{t,c} = |log y_{t,c} − log y_{t−1,c}|`, clipped at τ, squared
and averaged, with **λ = 0.15, τ = 4**.

**That loss is the problem.** T-MSE is a *smoothness prior on the boundary set* — it explicitly
discourages the model from emitting boundaries. A failed rollout by construction contains an
*unexpected* boundary. The architecture's central regulariser works against the thing you want.

*Numbers* (F1@10/25/50 | Edit | Acc): 50Salads 76.3/74.0/64.5 | 67.9 | 80.7; GTEA
85.8/83.4/69.8 | 79.0 | 76.3; Breakfast(I3D) 52.6/48.1/37.9 | 61.7 | 66.3. **Breakfast F1@50 of
~38 means the majority of predicted segments do not reach 50% IoU with ground truth on
in-distribution data.**

**ASFormer** — [arXiv:2110.08568](https://arxiv.org/pdf/2110.08568). Encoder-decoder transformer
with **local windowed attention whose window grows hierarchically with depth**, plus MS-TCN's
multi-stage refinement re-expressed with cross-attention. 50Salads F1@50 **76.0**. *Published
negative result* **[F]**: *How Much Temporal Long-Term Context is Needed for Action
Segmentation?* ([arXiv:2308.11358](https://arxiv.org/pdf/2308.11358)) shows ASFormer is *"limited
by [its] inability to capture the full context of a video"* — restricting context to 50% of the
average video length drops F1@50 from **82.0 to 76.0**. *Implication*: these models are
**offline and non-causal**. They cannot be run online to say "it broke at t" without re-running
on a growing prefix, and their accuracy *on a prefix* is not what the reported numbers measure.

**DiffAct** ([ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Diffusion_Action_Segmentation_ICCV_2023_paper.pdf))
**[A]** frames segmentation as conditional denoising diffusion with three explicit priors —
**position** (when in the video an action tends to occur), boundary ambiguity, relational
dependency. *The position prior is directly harmful here*: a model that has learned "the pour
happens ~60% into the video" will hallucinate structure onto a rollout that aborted at 30%.
Results tables could not be retrieved **[?]**.

**Unsupervised TAS** carries the same disease in sharper form:
- **CTE** ([arXiv:1904.04189](https://arxiv.org/abs/1904.04189)) **[A]** learns an embedding by
  regressing **relative timestamp** from frame features, then clusters. That regression head
  *is* a progress estimator fitted to successful executions; on a failed rollout relative
  timestamp ≠ progress and the embedding collapses.
- **TOT** ([arXiv:2105.13353](https://arxiv.org/abs/2105.13353)) **[A]** solves an
  entropy-regularised optimal-transport problem per minibatch for frame-to-cluster
  pseudo-labels, with a **temporal regularisation term penalising transport mass far from the
  diagonal** of the frame×cluster matrix. **The diagonal prior is the assumption that time ≈
  progress. Retries, stalls and regressions — the defining features of a failing rollout — are
  exactly what it forbids.** This is the deepest reason unsupervised TAS is the wrong tool here.
- **ASOT** (CVPR 2024, [arXiv:2404.01518](https://arxiv.org/abs/2404.01518)) **[F]** is the one
  worth trying. It solves a **fused, unbalanced Gromov-Wasserstein** problem over the
  frame×action coupling `T`:
  `min_{T ∈ 𝒯_p} F_FGW(C,T) + λ·D_KL(Tᵀ1_n ‖ q) − ε·H(T)`, with
  `F_FGW = α·F_GW(C^v, C^a, T) + (1−α)·F_KOT(C^k, T)`, `C^k` the frame-to-action cosine
  distance, the GW term encoding temporal consistency via a radius parameter `r` **without
  requiring the action order a priori** (its advance over TOT), and — the part that matters —
  the `λ·D_KL` **unbalanced** term permitting unequal durations and **missing actions**. Solved
  by projected mirror descent (~25 iterations, 26.1 ms for a 16k-frame video on an RTX 4090).
  Decoding is `j*(i) = argmax_j T*_{ij}`. *A failed rollout is a video where some actions never
  happen*, so the unbalanced term is genuinely the right property. *Author-acknowledged
  limitations*: underperforms per-video on short under-represented actions; **heavy
  hyperparameter sensitivity requiring per-dataset tuning**; **random initialisation degrades
  results — requires k-means pre-init**; cannot train across activity categories simultaneously.
- **ABD** (action-boundary detection) **[A]**: detect boundaries directly from frame-to-frame
  feature dissimilarity peaks, then refine by clustering. **Boundary-only methods are more
  robust on failures than clustering methods, because they never have to name the action** —
  they only claim "something changed here", which makes them functionally change-point detectors
  on visual features (§2.4).

**TAS metrics do not measure what localisation needs** **[F/A]**:
- **MoF / frame accuracy** is dominated by long segments; for unsupervised methods it is computed
  **after Hungarian matching**, either per-video or globally, and the two differ substantially.
- **Segmental F1@{10,25,50}** requires IoU ≥ k% between matched segments — **insensitive to
  small temporal offsets**, exactly the wrong sensitivity profile for finding a timestep.
- **Edit score** is normalised Levenshtein over the *segment label sequence* — an **order**
  metric, tolerant of temporal misalignment.

**None of F1@k, edit or MoF measures boundary-timing error.** If a Stage-2 output is "the
failure began at t = 137", the TAS metric suite cannot score it. §2.8 gives the metrics that can.

**No paper was found measuring MS-TCN/ASFormer/ASOT on failed robot rollouts** **[?]**. Treat
"TAS works on failures" as an untested assumption and say so.

**The one domain that HAS done this: surgical error detection.** JIGSAWS carries both gesture
segmentation labels *and* error labels. *Analysis of executional and procedural errors in dry-lab
robotic surgery* ([PMC9285717](https://pmc.ncbi.nlm.nih.gov/articles/PMC9285717/)) **[A]**
establishes a distinction that should be imported wholesale:
- **Procedural error** — a gesture is omitted, or the sequence deviates from the task grammar.
  *Localisable by segmentation plus a grammar check.*
- **Executional error** — the gesture is performed, but badly. **Not localisable by segmentation
  at all**; needs a within-segment quality signal.

> **Segmentation-based localisation can only ever catch procedural failures.** That single
> sentence should govern how much weight we put on our phase segmenter. Related: SEDMamba
> ([arXiv:2406.15920](https://arxiv.org/html/2406.15920)), Chain-of-Gesture prompting
> ([arXiv:2406.19217](https://arxiv.org/pdf/2406.19217)).

### 2.2 Phase / subgoal / keyframe detection — cheap, transferable, and blind in one place

**The PerAct/RVT keyframe heuristic — the exact rule** **[F]**.
[PerAct](https://arxiv.org/pdf/2209.05451) §3.2, verbatim: *"an action is a keyframe if (1) the
joint-velocities are near zero and (2) the gripper open state has not changed"*. **PerAct gives
no numerical threshold for "near zero" — verified absence.** The origin is Q-attention
([James & Davison](https://arxiv.org/pdf/2105.14829)), where it is stated as a **disjunction**:
*"performing a disjunction over two simple conditions worked well: (1) change in gripper state…
and (2) velocities approaching near zero (a common occurrence when entering pre-grasp poses or
entering a new phase of a task)"*. **Note the sign flip between conjunctive and disjunctive
phrasings — implementations differ and it matters if you reuse the rule.**

*Requires*: joint velocities + binary gripper state. **Proprioception only** — no privileged
simulator state, no object segmentations, no poses. This is why it transfers to real robots.

*Published limitation* (PerAct's own): *"as tasks get more complex, keyframe functions will
inevitably need to become more sophisticated… such as sudden changes in direction or joint
velocity, or large changes in pixel values."* Independent 2026 critique (SkiP,
[arXiv:2605.15536](https://arxiv.org/html/2605.15536)): *"Prior methods rely on heuristic
keyframes such as gripper-state changes or velocity zero-crossings, which capture pick-and-place
events but **miss sustained high-precision motion like sweeping or dragging**"* **[F]**.

> *Verdict*: **this is the cheapest, most transferable and most robust phase segmenter
> available, and it works on failed rollouts as well as successful ones because it makes no
> claim about *what* the phase is.** Its blind spot is continuous-contact tasks (wiping,
> dragging, insertion-with-search) where velocity never zeroes and the gripper never changes —
> precisely the tasks where localisation matters most.

**AWE — Automatic Waypoint Extraction** — [arXiv:2307.14326](https://arxiv.org/abs/2307.14326) ·
[code](https://github.com/lucys0/awe). *Mechanism* **[F]**: decompose a trajectory into the
**minimal set of waypoints such that linear interpolation between consecutive waypoints
approximates the original within η at every timestep**, chosen by dynamic programming. *The
important detail, verified*: the reconstruction loss is the **maximum projection error over all
proprioceptive states, not the mean** — author rationale: *"The success of a trajectory often
relies on reaching key states, and the mean error can be low while having a high projection
error for those key states."* Thresholds used in the repo: RoboMimic `err_threshold = 0.005`,
Bimanual Suite `0.01`. *Requires*: **proprioception only**; no vision, no reference demo — it is
a geometric decomposition of whatever trajectory you hand it, so it applies unchanged to failed
rollouts. *Numbers*: up to +25% success in sim, +4–28% real bimanual, up to 10× horizon
reduction. The exact DP recursion was **not** verified **[?]**.

> **An under-exploited repurposing**: AWE's per-timestep projection error, computed against the
> rollout's *own* linear reconstruction, is a cheap, **reference-free, proprioception-only
> erraticness signal**. A thrashing rollout produces a dense spray of waypoints. Nobody appears
> to have used waypoint density as a localisation feature.

**Subgoal discovery via LLM/VLM** — the standard and sensible division of labour is that the
*temporal cut is proprioceptive and the VLM only labels*. V2GP
([arXiv:2603.13433](https://arxiv.org/html/2603.13433)) **[A]** segments by **gripper-state
signals**, then a VLM identifies the manipulated object per segment. SeqVLA
([arXiv:2509.14138](https://arxiv.org/html/2509.14138)) learns an explicit subtask-completion
detector but needs **dense per-frame human annotation**. *Break*: a VLM asked "which subgoal is
the robot on?" during a failing rollout will typically answer with the *nearest plausible*
subgoal rather than "none of them" — a silent failure mode.

**SBD — Skill Boundary Detection** ([arXiv:2503.10684](https://arxiv.org/abs/2503.10684)) **[A]**
deserves separate attention. *Mechanism*: annotation-free segmentation that detects skill
boundaries via **prediction errors of a pretrained unconditional action-prediction model** — a
significant increase in prediction error indicates a shift in the skill being executed. (Exact
threshold not stated in the abstract **[?]**; domain is Minecraft, not robotics.)

> **The collision nobody has addressed.** SBD localises *skill boundaries* by prediction-error
> spikes. FailureSpot and Rewind-IL/TIDE (§2.8) localise *failures* by prediction-inconsistency
> spikes of the policy. **These are the same statistic.** No published work disambiguates "the
> policy is switching phases" from "the policy is confused". Legitimate subgoal transitions will
> produce false positives in any TIDE-style detector. This is a real, citable gap and a good
> place to contribute.

**Corpus-level segmentation methods that cannot triage a single rollout.** Two classic robotics
methods, included because they are the obvious things to reach for and both fail for the *same
structural reason*:
- **BP-AR-HMM** (Fox et al.; applied by Niekum, [RSS 2013](https://www.roboticsproceedings.org/rss09/p48.pdf)) **[A]**.
  A **beta-process (IBP) prior over an infinite mode library** means each demonstration exhibits
  a *subset* of globally discovered modes with its own transition dynamics — it does not force
  every demo to contain every skill or to order them identically. **Autoregressive emissions**
  `y_t = Σ_{i=1}^{r} A_{i,z_t} y_{t−i} + e_t(z_t)` make a "mode" a *dynamical regime* rather than
  a static cluster. Inference is MCMC with IBP birth/death moves. *Requires*: multiple
  demonstrations of the same task; kinematics only, real-robot compatible (PR2).
- **TSC — Transition State Clustering** (Krishnan, Garg, Patil, Lea, Hager, Abbeel, Goldberg,
  [ISRR 2015](https://people.eecs.berkeley.edu/~pabbeel/papers/2015-ISRR-TSC.pdf)) **[A]**. Three
  stages: (1) model demos as a switched linear dynamical system, proposing candidate transition
  states via a DP-GMM over local linear dynamics; (2) **cluster candidate transitions by
  kinematic similarity across demonstrations**, keeping only clusters supported by enough
  demonstrations — explicitly to **prune spurious transitions from inconsistent motion**;
  (3) correlate surviving clusters with sensory and temporal features.

*Why both fail here*: **consistency across demonstrations is their signal, and a failure is by
definition inconsistent.** TSC's pruning stage *actively deletes the event you are hunting for*;
BP-AR-HMM either absorbs a one-off failure into the nearest mode or spawns a singleton.

> **But TSC inverted is a usable localiser that nobody has published.** Run TSC on the
> *successful* corpus to obtain the canonical transition-state set, then ask *at which canonical
> transition did this rollout fail to produce a matching transition state?* That is a well-posed,
> reference-based, proprioception-only Stage-2 method and it is cheap for us because we already
> have a scripted-oracle corpus.

### 2.3 Contact-event segmentation

**Contact mode as a discrete hybrid mode** ([survey, arXiv:2112.01942](https://arxiv.org/pdf/2112.01942))
**[A]**. Manipulation dynamics are hybrid: the contact configuration — which patches are active
and, for each, **stick / slip / separation** — indexes a discrete mode; within a mode the
dynamics are smooth. **A mode transition *is* a segmentation boundary with a physical rather than
statistical definition.** *In simulation this is free (privileged). On a real robot you must
infer it.*

**ContactNets-style learning without a contact oracle** — Bianchini, Halm, Posa, CoRL 2023,
[arXiv:2310.12054](https://arxiv.org/abs/2310.12054) **[A]**. Learns contact + continuous
dynamics jointly using a **violation-based loss** that *infers unmeasured contact forces* and
penalises their violation of physical constraints given current model parameters — no forward
simulation during training and **no contact-detection oracle**. Their stated motivation is
exactly our sim-vs-real problem: *"Detecting contact events is extremely difficult in many
practical scenarios, and many model building works utilize simpler variations such as using a
contact detection oracle or assuming knowledge of contact distances."*

**F/T-derivative segmentation with BOCPD** **[A]**: compute time-derivatives of 6-axis F/T
measurements; declare candidate boundaries at derivative spikes; run **Bayesian online
changepoint detection over the haptic signal to suppress over-segmentation from sensor noise**.
A concrete published recipe ([arXiv:2605.17601](https://arxiv.org/pdf/2605.17601)) is
**Total-Variation-denoise the force signal, segment at rising/falling edges above a 10 N
threshold, fused with gripper-state changes**. *Requires wrist F/T sensing* — real but **absent
from standard LIBERO/ALOHA/UMI setups**, which is the main practical blocker.

**Tactile event detection — Evetac** (T-RO 2024, [arXiv:2312.01236](https://arxiv.org/abs/2312.01236))
**[F]**. Replaces the RGB camera in a GelSight-style sensor with a **DVXplorer Mini event
camera**; events accumulated every **1 ms** ⇒ **1000 Hz**, verified to detect vibrations up to
**498 Hz**. Slip detection: per-dot features (displacement, event count) → 2 conv + 2 FC layers →
slip probability.

> **Evetac's annotation trick is the most transferable idea in this subsection.** Ground-truth
> slip labels are obtained **automatically via optical flow through a transparent window cut in
> the gel**, which eliminates external-sensor delay. It is a genuinely clever solution to
> exactly the problem we face — *"when exactly did this begin?"* is hard to annotate by hand,
> and the answer is to instrument a second, independent, higher-bandwidth channel that observes
> the same event.

*Author-stated limitation*: event cameras report only *changes*, so *"it is impossible to recover
the gel's global configuration given only a single measurement"* — you can ask "did contact just
change", never "are we in contact now". The literature also distinguishes an early **micro-slip**
phase from **bulk slip**, a two-stage onset structure that maps directly onto "degradation began
/ point of no return".

### 2.4 Change-point detection — the reference-free family

**CUSUM** (Page 1954) **[F]**. With `ℓ(u)` the per-sample log-likelihood ratio between post- and
pre-change models, `Y(n) = max_{0≤t≤n} Σ_{u=t+1}^{n} ℓ(u)`, computed by the recursion
**`Y(n) = (Y(n−1) + ℓ(n))⁺`, `Y(0) = 0`**, alarming when `Y(n) > b`. *The catch*: you need a
post-change model, and *"in the problem of fault detection, the post-change distribution is
likely to be unknown."* *Where it breaks*: the `(·)⁺` accumulator detects **persistent drift
excellently and transient spikes poorly**, and the alarm time is **systematically later than the
true change point** — detection *delay* is what the theory optimises, not localisation accuracy.
**If you use CUSUM, estimate the change point post-hoc as `argmax_t` of the running sum, not as
the alarm time.** Direct empirical corroboration from robotics: ActProbe reports that
*"cumulative accumulation leads to late detection"* while single-timestep detection *"produces
excessive false alarms"*, and that **sliding-window aggregation beats both** **[F]**.

**BOCPD — Bayesian Online Changepoint Detection** (Adams & MacKay 2007,
[arXiv:0710.3742](https://arxiv.org/abs/0710.3742)) **[F]**. Maintain an online posterior over
the **run length** `r_t` = observations since the most recent change point. The joint
`P(r_t, x_{1:t})` is propagated by either *growing* the run (`r_t = r_{t−1}+1`, weight
`1−H(r_{t−1})`) or *resetting* (`r_t = 0`, weight `H(r_{t−1})`), each multiplied by the UPM
predictive `P(x_t | r_{t−1}, x^{(r)})`, then normalised. `H` is the **hazard function**; constant
hazard `H = 1/λ` is a geometric run-length prior. Conjugate exponential-family UPMs give
closed-form updates. *Requires*: a likelihood model and a hazard rate. **Fully online and causal
— its main advantage over every TAS method in §2.1.** No reference trajectory, no privileged
state, no training on successes. *Break*: *"The update formula highly relies on the pre-defined,
fixed hazard rate"*; Gaussian UPMs on robot joint signals fire on every velocity reversal.

**`ruptures` — PELT, BinSeg, kernel CPD** ([arXiv:1801.00826](https://arxiv.org/abs/1801.00826) ·
[docs](https://centre-borelli.github.io/ruptures-docs/)) **[A]**. Offline CPD as minimisation of
`Σ_k c(y_{t_k : t_{k+1}})` over segmentations. **Dynp**: exact DP when K is known, `O(K n²)`.
**PELT**: minimises the penalised criterion `Σ c(·) + β·K` with a pruning rule discarding
candidate last-change points that can never be optimal — exact and near-linear. **BinSeg**:
greedy, `O(n log n)`, approximate. **KernelCPD**: costs in an RKHS, so non-parametric. *Break*:
β selection is the entire game with no principled default; the `l2` cost detects **mean** shifts
and misses variance/dynamics changes — use `rbf` or `ar` for robot signals.

> **Verdict**: `ruptures` with an **RBF kernel cost + PELT** on end-effector speed ‖v‖ and
> gripper width is the single cheapest credible baseline for "where did this rollout's phase
> structure differ from normal", and it needs nothing but proprioception.

**rSLDS** (Linderman, Johnson, Miller, Adams, Blei, Paninski, AISTATS 2017,
[arXiv:1610.08466](https://arxiv.org/abs/1610.08466)) **[F]**. Discrete mode `z_t`, continuous
latent `x_t`, observation `y_t`: `x_t = A_{z_t} x_{t−1} + b_{z_t} + ε`, `y_t = C x_t + d + δ`.
**The innovation**: the mode transition depends on the *previous continuous state*,
`p(z_t | z_{t−1}, x_{t−1}) ∝ softmax(R_{z_{t−1}} x_{t−1} + r_{z_{t−1}})`, which partitions the
continuous state space into regions each governed by its own linear dynamics — so the model
**explains *why* the switch happened**, not merely that it did. Inference uses a **Pólya-gamma
augmentation** making the logistic evidence potentials conditionally Gaussian, enabling conjugate
block Gibbs updates. *Break*: slow; mode identity not identifiable across runs; and the switching
rule is fit on *nominal* data, so a failing rollout visits regions where it was never fit.

> **An open opportunity**: because rSLDS learns a *map* from continuous state to mode, a rollout
> entering a region with high switching-rule entropy or low likelihood under every mode is
> directly flaggable — a principled "the dynamics stopped being explainable at t\*" statistic.
> **No published work does this for robot failure localisation.**

**Sticky HDP-HMM / HDP-SLDS** (Fox, Sudderth, Jordan, Willsky,
[arXiv:0905.2592](https://arxiv.org/pdf/0905.2592)) **[A]**: an HDP prior over an **unbounded**
number of modes, with a "sticky" parameter κ adding mass to self-transitions to stop the model
fragmenting one mode into rapidly-alternating pseudo-modes. *Break*: **κ does the same job as
MS-TCN's λ and BOCPD's hazard — a tunable over-segmentation knob with no principled setting.**
Notice how many methods in this section reduce to one such knob.

### 2.5 Point of no return / irrecoverability

**Hamilton–Jacobi reachability** (Bansal, Chen, Herbert, Tomlin,
[arXiv:1709.07523](https://arxiv.org/abs/1709.07523)) **[A]** gives the mathematically exact
definition. Define a failure set ℒ via `l(x) ≤ 0`. The safety value `V(x,t)` is the viscosity
solution of a Hamilton–Jacobi–Isaacs variational inequality; the **Backward Reachable Tube** —
states from which failure is unavoidable under worst-case disturbance — is the **zero sublevel
set of V**, and the optimal safe control is `u* = argmax_u ∇V·f(x,u,d)`. Then:

> **t\* = min{ t : x_t ∈ BRT }.** That is the point of no return, stated exactly.

*Requires*: a dynamics model `f`, control and disturbance bounds, a failure set `l(x)`, and a
state grid. *Breaks*: **exponential complexity in state dimension** (grid methods impractical
above ~5–6 dims); worst-case conservatism; **and you must be able to write `l(x)` down, which for
"the block fell over" means object pose — i.e. privileged state.**

**Latent safety filters — the key transfer result** **[A]**. Nakamura, Peters, Bajcsy,
[arXiv:2502.00935](https://arxiv.org/pdf/2502.00935); **UNISafe** (CoRL 2025,
[site](https://cmu-intentlab.github.io/UNISafe/)). Learn a world model from vision, run
**approximate HJ reachability in the latent state space**, so the failure set can be defined on
*learned* features rather than analytic object poses — "can detect, predict, and mitigate
failures that are hard to model, such as those encountered in vision-based manipulation."
**This is the only credible route to a real-robot point-of-no-return estimate.** UNISafe's
uncertainty-awareness exists because the latent dynamics model is itself unreliable OOD — which
is precisely the states you care about. **That circularity is the central unsolved problem.**

**Dead-end detection** — Fatemi et al., ICML 2019
([PMLR](https://proceedings.mlr.press/v97/fatemi19a.html)) **[A]**. *Definition, verified*: a
state is a **dead-end** if all trajectories from it reach an undesired terminal with probability
1 in finite steps. And the sentence that is the whole motivation for Stage 2: *"unlike undesired
terminal states which are signaled when entered, no such assumption can be made for dead-ends,
which may exist far before undesired terminals."* Mechanism: a **security** condition on
exploration, translated into an auxiliary value function that **caps** any exploration policy;
the auxiliary MDP has reward −1 on undesired terminals and 0 elsewhere. **Exact Bellman equations
and theorem statements could not be verified** **[?]**. *The obstacle for robotics*: you need
many episodes that actually *reach* dead-ends to learn the value function — effectively
simulation-only.

**Reversibility** — Grinsztajn et al., *There Is No Turning Back*, NeurIPS 2021
([arXiv:2106.04480](https://arxiv.org/abs/2106.04480)) **[A]**. *Mechanism, and it is elegant*:
approximate reversibility is learned by a **surrogate task — ranking randomly sampled pairs of
trajectory events in chronological order**. A classifier `ψ(s_i, s_j)` predicts `P(t_i < t_j)`.
Intuition, verified: *"pairs of events that are always observed in the same order are likely to
be separated by an irreversible sequence of actions."* So `ψ ≈ 1` (perfectly predictable order)
⇒ irreversible; `ψ ≈ 0.5` ⇒ reversible. Two uses: **RAE** penalises irreversible transitions via
reward; **RAC** rejects actions above an irreversibility threshold. *Requires*: only trajectory
data — fully self-supervised, no model, no reward, no privileged state.

> ***Break, and it is fatal for demonstration data.*** It measures **statistical**
> irreversibility (what the behaviour policy never undoes), not **physical** irreversibility.
> Anything the data-collection policy simply never tried to undo — which for curated demos is
> nearly every action — is scored irreversible. **On demo data this estimator is close to
> degenerate: demos never go backwards, so everything looks irreversible.** You would have to
> train it on exploratory or scripted-perturbation data. A 2025 successor
> ([arXiv:2510.14503](https://arxiv.org/html/2510.14503)) proposes **an online per-state-action
> estimator of the probability of returning to a prior state within a fixed horizon** — a much
> better-posed, robot-applicable definition **[A]**.

**Robotics work estimating the earliest non-recoverable timestep.** Thin but real, and all
2024–2026:

- **Foresight** (2026, [arXiv:2606.23085](https://arxiv.org/html/2606.23085)) **[F]**. *Mechanism*:
  **V-JEPA 2-AC action-conditioned world-model latents** — a frozen visual encoder over the
  observation context plus a trained action-conditioned predictor that rolls the latent forward
  **under the policy's own predicted action chunk**; a causal Transformer over the latent tokens
  emits a **per-timestep failure score ∈ [0,1]**; **functional conformal prediction** on held-out
  *successful* rollouts gives a time-varying threshold δ_t. Inputs: 256×256 images (8-frame
  sliding window) and policy action chunks; **trajectory-level labels only — no timestep
  annotations**. *Numbers* (ROC-AUC / balanced acc): LIBERO-Long **0.89±0.02 / 0.94±0.06**;
  ManiSkill-Long 0.84 / 0.80; BEHAVIOR-1K (**8,557 steps**) 0.76 / 0.78; real ReactorX/ACT
  0.93±0.01, Franka/GR00T 0.89±0.10. *Author limitations*: world-model latency rules out highly
  reactive tasks; conformal guarantees need calibration data matching deployment; **asymmetric
  cross-policy transfer**. **Verified: Foresight does *not* identify a discrete point of no
  return — it does continuous monitoring.**

- **Rewind-IL** (2026, [arXiv:2604.16683](https://arxiv.org/html/2604.16683)) **[F]** — the
  closest thing to an operational point-of-no-return system, and the mechanism is worth stating
  in full. *Detection signal* **TIDE (Temporal Inter-chunk Discrepancy Estimate)**:
  `TIDE_t = (1/BDT) Σ (Â^{(b)}_{t−1,τ,d} − Ã^{(b)}_{t,τ,d})²` — the squared disagreement between
  the policy's **previous** action-chunk prediction and its **current** prediction over the
  overlapping horizon, i.e. *"the policy is reconsidering its near-future plan after an
  unexpected state."* Threshold by **split conformal prediction** on successful rollouts:
  `q̂ = Quantile(S_cal, ⌈(n+1)(1−α)⌉/n)` with **α = 0.001**. *Where to rewind to*: offline, a VLM
  inspects training demos to mark K semantically meaningful recovery timestamps per episode
  (*"immediately after a grasp closes, or at the transition between sub-goals"*); online, cosine
  similarity between the current observation and each checkpoint embedding is tracked, a slot is
  declared **"peaked"** when its max similarity fails to improve for Δ_peak steps, and on alarm
  the system respawns to **the latest peaked slot** — the furthest confirmed-safe waypoint
  traversed. Checkpoint templates chosen by **KDE max log-density in PCA space**, not averaging.
  *Numbers*: TIDE balanced accuracy **0.95** average over six real tasks (1.00 on four) vs
  FAIL-Detect 0.83, RND 0.59, clustering-OOD 0.60; real ACT 66.7% → 80.0% undisturbed and
  **18.3% → 76.7% (+58.4 pp) under adversarial disturbance**; overhead <0.2 ms/cycle.
  *Author-stated limitations, all instructive*: on deformable tasks with few demos, the policy's
  intrinsic prediction inconsistency **suppresses** the TIDE signal (Folding Towel drops to 55%);
  flow-matching policies' tighter action distributions *"leave less margin for recovery on
  geometrically demanding tasks"*; **no recovery is possible unless at least one slot has peaked
  — early-episode failures have no verified checkpoint.**

- **PREFAIL** ([arXiv:2607.16921](https://arxiv.org/abs/2607.16921)) **[A]**. Non-prehensile
  lift-and-place; analyses **the relative motion of the target object with respect to the
  carrier** to predict failures, and claims to *"precisely identify the latest intervention time
  for risky manipulations"* — i.e. `t_stop`, the latest emergency stop that still prevents
  failure, versus `t_fail`. **The backtracking algorithm itself could not be retrieved** **[?]**,
  but the *framing* — report `t_stop` and `t_fail` as two separate quantities — is directly
  adoptable and is exactly what a "point of no return" field in a manifest should contain.

- **AEGIS** ([arXiv:2606.06660](https://arxiv.org/pdf/2606.06660)) **[A]**. A lightweight probe
  on a weak policy's **frozen activations** detects high-risk steps *while there is still time to
  act*, then hands control to a stronger policy for those steps only. LIBERO-Spatial: recovers
  **10.1%** of the weak policy's failed trajectories using the strong policy for **38% of steps**;
  **early-window AUROC 0.764 (95% CI [0.70, 0.84])**; vs 4.6% for budget-matched blind escalation
  and 5.1% for a random-trigger placebo. **Note the placebo arm — that design should be copied.**

- **RecoveryChaining** ([arXiv:2410.13979](https://arxiv.org/pdf/2410.13979)) **[A]** defines
  recoverability *operationally*: "can a nominal model-based controller take over from here?"
  For us that has an exact analogue — **can the scripted oracle finish from this state?** — which
  we can evaluate directly and cheaply, since we already have the oracle.

### 2.6 Trajectory divergence from a reference — and why it should be dropped, not caveated

**DTW and soft-DTW** ([Cuturi & Blondel, ICML 2017](https://arxiv.org/abs/1703.01541)) **[F]**.
DTW finds the minimum-cost monotone alignment through the n×m cost matrix by
`r_{i,j} = c_{i,j} + min(r_{i−1,j}, r_{i,j−1}, r_{i−1,j−1})`; soft-DTW replaces `min` with
`min_γ(a) = −γ log Σ exp(−a_i/γ)`, making it differentiable. *Requires a reference trajectory —
the fatal requirement.*

*Published critique, and it is devastating and exactly on point* **[F]** (from *Imitation
Learning from a Single Temporally Misaligned Video*,
[arXiv:2502.05397](https://arxiv.org/pdf/2502.05397)):

> *"Although DTW constrains the order of assignment, it does not limit the number of learner
> frames matched with each subgoal, and **a trajectory stuck in an intermediate subgoal can
> achieve the same DTW reward as a trajectory that completes all subgoals**."*

**A stalled rollout can score identically to a successful one. DTW distance cannot even detect
the failure, let alone localise it.**

**Fréchet distance** **[A]** — `F(P,Q) = inf_{α,β} max_t ‖P(α(t)) − Q(β(t))‖`, order-preserving
and **parameterisation independent**, which is why people reach for it to compare end-effector
paths. *Break*: it is a **max**, so one instantaneous excursion dominates the score, making it
hypersensitive to a single outlier and uninformative about *where* things went wrong; and being
speed-invariant it is **blind to the most common VLA failure mode — stalling.**

**The conceptual objection.** Behaviour-cloned and especially diffusion/flow-matching policies
model a *multimodal* action distribution; most manipulation tasks admit many valid trajectories.
So: a large distance-to-nearest-demo is not evidence of failure (it may be a valid mode the demo
set undersampled); a small distance is not evidence of success; and the **argmin over demos is
unstable**, so mode-switching mid-rollout produces spurious divergence spikes precisely at the
multimodal decision points.

**The published evidence is strong enough to state this flatly:**

| Source | Finding |
|---|---|
| **RoboEval** (2025, [arXiv:2507.00435](https://arxiv.org/pdf/2507.00435)) **[F]** | Behavioural metrics (joint/Cartesian path length, joint/Cartesian jerk, self- and environment collisions, object slips, inter-arm height discrepancy, EE velocity divergence) **correlate significantly with success in only 59.4% of task-metric combinations** — in ~40% of cases the trajectory metric tells you nothing. On *Lift Tray (Rotation)* policies achieve identical success rates while behavioural metrics show ACT has superior motion quality. |
| **MetaFine** (2026, [arXiv:2605.19986](https://arxiv.org/pdf/2605.19986)) **[A]** | The cleanest negative result: on *Rotate Along*, **π₀.₅ achieves a stability score of 0.90 with a 10% success rate** — *"the vast majority of its rollouts produce smooth, controlled rotational motions that nonetheless fail to satisfy the directional constraint."* A trajectory-quality metric rates this rollout excellent. Trajectory length and slip count have consistently low coefficient of variation across policies — not discriminative. |
| **Geometric Entropy** (IROS 2026, [arXiv:2606.20871](https://arxiv.org/abs/2606.20871)) **[A]** | A task-agnostic measure of **intrinsic** trajectory diversity after normalising away extrinsic variation (goal pose, workspace scale) by target-frame alignment. Finds a **consistent inverted-U between success and H_G**: diversity helps in low-diversity regimes and *hurts* once it induces strategy ambiguity. Formula and numbers not retrieved **[?]**. Relevant as the constructive counterpart — diversity is a property of the *demo set*, not a per-rollout failure signal. |

> **Verdict**: trajectory-distance-to-demo should be used, if at all, only as a *feature* feeding
> a learned detector — never as a standalone localisation signal, and never as an evaluation
> metric.

### 2.7 Learned progress / value functions as a localisation signal

This is the family that most directly produces a timestep, because **the derivative of progress
is the localisation quantity**.

**VIP** ([arXiv:2210.00030](https://arxiv.org/abs/2210.00030)) **[F]**. The KL-regularised dual
of goal-conditioned RL via Fenchel duality yields, with `V*` taken as the negative L2 embedding
distance, the implicit time-contrastive loss
`L(ϕ) = E_{p(g)}[(1−γ)E_{μ₀}‖ϕ(o)−ϕ(g)‖₂ + log E_D exp(‖ϕ(o)−ϕ(g)‖₂ − δ̃_g(o) − γ‖ϕ(o′)−ϕ(g)‖₂)]`,
which **attracts initial and goal frames while implicitly repelling intermediate frames** through
recursive TD minimisation. The deployment reward is
**`R(o_t, o_{t+1}) = S_ϕ(o_{t+1}, g) − S_ϕ(o_t, g)` with `S_ϕ(o,g) := −‖ϕ(o)−ϕ(g)‖₂`** — a
per-timestep progress *derivative*, so **`t* = argmin_t R_t` is "the moment of greatest
regression"**. Pretrained on Ego4D (~72k clips, 4.3M frames), **action-free and label-free**.
*Requires a goal image at deployment.* *Limitation, and it is the family's shared one*: the
*"implicit assumption that visual progress correlates with task success"* — which is what breaks
on occluded or visually-ambiguous failures.

**LIV** ([arXiv:2306.00958](https://arxiv.org/abs/2306.00958)) **[A]** generalises VIP to
vision-*language*, so you can localise against **the task instruction itself** rather than a goal
image — a major practical advantage for a harness. Same load-bearing assumption.

**GVL** ([arXiv:2411.04549](https://arxiv.org/abs/2411.04549)) **[F]** — mechanism in §1.4. For
localisation: compute the value curve `v_t`, then `t* = argmin_t (v_{t+k} − v_t)` (largest
progress regression over a window), or the last `t` with `v_t` above its running max. **Its VOC
metric doubles as a per-rollout health score.** No privileged state, no reference trajectory, no
training. This is the most directly usable off-the-shelf localiser in this section.

**PROGRESSOR** (ICCV 2025, [arXiv:2411.17764](https://arxiv.org/abs/2411.17764)) **[A]** predicts
**a distribution over task progress** from the (current, initial, goal) observation triple — so
you get uncertainty on the progress signal — and, during online RL, **adversarially refines the
reward by pushing back predictions for out-of-distribution observations**.

> **This is the single most important design property in §2.7 for our use.** Every other progress
> model here is fitted on experts and silently extrapolates on failures. PROGRESSOR's OOD
> push-back is the only mechanism explicitly designed for the fact that **failed rollouts are OOD
> for the progress estimator**.

**Value-function collapse as the localisation signal** **[A]** — this mechanism is published and
named. **ValueFormer** (2026, [arXiv:2608.02958](https://arxiv.org/html/2608.02958)), verbatim:
*"VLA policies trained by behavior cloning **fail silently**: from the action stream alone, a
collapsing rollout looks much like one making clean progress, because imitation supplies no
notion of progress."* And: *"A value function cleanly separates success and failure trajectories,
and **reproduces canonical rollout signatures including a rise-then-drop on early collapse**."*
*To Err is Robotic* ([Stanford](https://cicl.stanford.edu/papers/du2024robotic.pdf)) adds the
clause that matters: a value function *"is able to capture both progress regression **and
subsequent recovery**"* — **a value curve distinguishes a recovered stumble from a terminal
collapse, which no segmentation method can.**

**Two warnings that apply to the whole family:**

1. **The circularity, documented verbatim**: *"failure detection models can detect mistake states
   effectively, but **the same out-of-distribution observations that degrade performance for the
   policy can also degrade performance for the failure detection models**."* This is the central
   epistemic problem of Stage 2 and should be stated in any writeup.
2. **Retries are not regressions.** *Beyond Monotonic Progress: Retry-Supervised Value Learning
   for Robot Imitation* (2026, [arXiv:2606.24633](https://arxiv.org/pdf/2606.24633)) **[A]**
   attacks the monotone-progress assumption baked into VIP/LIV/GVL/PROGRESSOR: a retry is not a
   regression in task terms, and supervising a value function to be monotone in time mislabels
   legitimate retries as failures. **Any derivative-of-progress localiser will fire on every
   retry** — and "recovery" is one of our own taxonomy families, so this is directly load-bearing.

### 2.8 Policy-internal and VLM localisers — where the field actually is in 2026

**REFLECT** (CoRL 2023, [arXiv:2306.15724](https://arxiv.org/abs/2306.15724)) **[F]** — the
original hierarchical localiser, and the source of the most quotable privileged-state number in
this survey.

*Mechanism, in detail*: a three-level **hierarchical summary** of multisensory experience,
queried by an LLM.
- *Sensory level*: RGB-D → object detection → projection to a **3D semantic point cloud** →
  object states via **CLIP embeddings** → a **task-informed scene graph** over **8 spatial
  relations** (inside, on top of, on left, on right, above, below, occluding, near), computed by
  heuristics with a **5 cm contact threshold** and **40 cm distance threshold**; point clouds
  aggregated across frames with add/update/replace/delete operations. Audio via AudioCLIP.
- *Event level*: **key event frames are selected by three conditions** — (1) the scene graph of
  the current frame **differs from the previous frame**; (2) the frame is the **start or end of
  an audio event**; (3) the frame marks the **end of a subgoal execution**. (**No numerical
  threshold is given for the scene-graph change — verified absence.**)
- *Subgoal level*: the environment observation at the end of each subgoal.
- **Two-stage localisation**: iterate through subgoals checking success; **if a subgoal fails,
  retrieve the event summaries leading up to that point and prompt the LLM to explain**
  (execution-level failure). **If all subgoals succeeded but the task failed, compare the
  original plan against the final environment state** (planning-level failure).

*Numbers* **[F]**, on RoboFail (100 simulated failures across 10 AI2THOR tasks + 30 real UR5
failures across 11 tasks): **simulation — execution failures 88.4% explanation accuracy, 96.0%
localisation accuracy**; planning 84.2% / 80.7%. **Real world — 68.8% / 93.8%** for execution
failures.

> **The privileged-state number.** Verified: *"In simulation, it assumes access to **ground-truth
> object detection and state detection** — this represents privileged simulator information not
> available in real-world deployment."* The real-world variant substitutes MDETR + CLIP +
> AudioCLIP, and explanation accuracy drops **88.4% → 68.8%**. That is a direct, published
> measurement of the privileged-state gap for a localisation method.

*Break*: scene-graph-change-driven keyframing means **failures that do not change the symbolic
scene graph — slip without drop, wrong-force insertion, near-miss grasp — produce no key event
frame and are invisible.**

**KITE** (2026, [arXiv:2604.07034](https://arxiv.org/html/2604.07034)) **[F]** — the most
Stage-2-shaped VLM pipeline found, and **training-free**.
*Mechanism*: convert a long execution video into compact tokenised evidence.
- **Keyframe selection**: dense **optical flow**; score each frame by **average flow magnitude**;
  keyframes are **local peaks of this score under temporal non-maximum suppression**; if fewer
  than the budget **M = 8** motion-salient frames are found, top up with uniform sampling. *This
  is a pure motion-energy rule requiring nothing but the video, and it works identically on
  failed and successful rollouts.*
- **Evidence bundle**: robot profile (morphology, gripper type, workspace); **timestamped
  keyframe tags with the frame index and timestamp burned into the RGB** — the mechanism by which
  the VLM can *name a time*; **contact-transition tokens** with coarse states **Gain / Loss /
  Stable** from IoU and gripper–object distance changes; scene relations; **pseudo-BEV
  schematics** (non-metric top-down layouts with confidence-scaled circles).
*Numbers* on the RoboFAC benchmark vs vanilla Qwen2.5-VL-7B in simulation: failure **detection
+36 points**, **identification +18 points**, **localization +33 points**.

> **The +33 points on failure localization from a purely training-free front-end is strong
> evidence that VLMs cannot temporally localise robot failures without explicit timestamp
> scaffolding — and that giving them motion-salient keyframes with burned-in timestamps mostly
> fixes it.** This is the practical recipe for Stage 2 with a VLM.

*Author limitations*: relies on open-vocabulary detection and monocular relative depth, which
struggle with small, occluded, reflective or ambiguous objects; the contact proxy captures
*"coarse interaction trends rather than precise force events"*; the pseudo-BEV *"flattens
vertical structure."*

**FailureSpot** (Sept 2026, [arXiv:2609.04277](https://arxiv.org/abs/2609.04277)) **[F]** — the
current SOTA framing for **timestamp-level** localisation, and the paper whose diagnosis we should
adopt.

*Problem*: per-timestep binary classification `p_{i,t} = f_θ(h_{i,t}) ∈ [0,1]` where `h_{i,t}` is
the VLA's internal representation **before action execution**. The motivating diagnosis, verified:
trajectory-level supervision *"causes normal pre-failure behavior in unsuccessful trajectories to
be incorrectly labeled as failure, which introduces label noise and limits precise
timestamp-level failure localization."* **This is exactly what is wrong with SAFE.**

*Stage 1 — action-derived weak supervision, no human labels*: three signals from unlabelled
action chunks —
1. **short-term action consistency** `c_{i,t}` = ℓ2 distance between overlapping predictions from
   consecutive chunks;
2. **full-chunk magnitude** `r_{i,t}` = average ℓ2 norm over predicted actions;
3. **executed magnitude** `r^exec_{i,t}` = ℓ2 norm of actions actually executed.

EMA-smoothed with **β = 0.8** and combined as
`s_{i,t} = [c + E(c)] + η₁[r + E(r)] + η₂[r^exec + E(r^exec)]`, then normalised by **1st/99th
percentile** statistics into soft pseudo-labels in [0,1].

*Stage 2 — active learning*: per-timestep entropy `H(p) = −p log p − (1−p) log(1−p)`, aggregated
per trajectory as `U_i = (1/T_i) Σ_t H(p_{i,t})`; **only the highest-uncertainty trajectories get
dense human annotation, under a 15% annotation budget**; fine-tune with BCE.

*Requires*: VLA internal representations + action chunks. **No vision features, no proprioception,
no privileged information at deployment.**

*Numbers — timestamp-level AUROC on unseen LIBERO-10 tasks (500 trajectories)*:

| Method | π₀ | π₀-FAST | OpenVLA |
|---|---|---|---|
| SAFE-MLP | 76.6 | 81.9 | 64.7 |
| SAFE-LSTM | 70.0 | 65.2 | 67.7 |
| ActProbe | 55.4 | – | 64.6 |
| **FailureSpot-MLP** | **90.5** | **85.8** | 63.0 |
| FailureSpot-LSTM | 84.3 | 84.4 | **66.5** |

**Detected onset is typically 1–2 timesteps early** relative to ground truth.
*Author-stated limitations, and the first is structural*: **OpenVLA is autoregressive with
H = K = 1, so there is no chunk overlap and therefore no consistency signal** — the method's core
statistic does not exist for that policy class (AUROC 63.0 timestamp, 50.3 trajectory). Failures
without an action-magnitude signature are missed. **LIBERO simulation only.**

**Hide-and-Seek in Trajectories** (2026, [arXiv:2605.30834](https://arxiv.org/html/2605.30834))
**[F]** — multiple-instance learning to manufacture temporal structure from trajectory labels.
*Mechanism*: extract VLA action embeddings `h_t`, sliding-window aggregation, LSTM emits per-step
failure scores, with two contrastive objectives:
- **Inter-trajectory** `L_inter`: the **maximum** score within any failed trajectory must exceed
  the **maximum** within successful trajectories by margin `m_r` (classic MIL max-pooling).
- **Intra-trajectory** `L_intra`: define a **proxy failure onset as the point of sharpest increase
  in the failure score**, and encourage a higher average score after that onset than before.
  *This is what manufactures temporal structure from trajectory-level labels alone.*
Runtime uses functional conformal prediction for time-varying thresholds `ζ_t`.
*Numbers*: LIBERO-10/OpenVLA bACC 85.2 seen / 83.4 unseen vs SAFE-MLP 82.3 / 77.5; real xArm-6
unseen KITCHEN bACC 97.2 vs LogpZO 83.7. **vs a VLM monitor: bACC 84.3 vs 71.2 (+13.1) at 0.001 s
vs 2.343 s — about 2,000× faster.** Compared against 12 baseline families.
*Limitations*: the sharpest-gradient onset proxy **may not align with annotated failure times**;
window size must be tuned per architecture.

> **The 2,000× latency figure is the decisive practical argument**: VLMs belong in offline
> post-hoc analysis, not in the online localisation loop.

**Code-as-Monitor** (§1.5) deserves a second mention here for a different reason: **a violated
constraint carries its own timestamp *and* its own semantic identity.** It is the most
interpretable localisation output of any method surveyed — the output is not "t=137 looks
anomalous" but "at t=137 the constraint `distance(gripper_surface, mug_handle) < 2cm` was
violated."

**Tri-Info** ([arXiv:2606.19998](https://arxiv.org/abs/2606.19998)) **[A]** formalises VLA control
as a **closed-loop information pipeline** and derives three information-theoretic signals capturing
whether actions remain **diverse, temporally consistent, and coupled to state transitions**. It
matches the strongest baselines in-domain but **transfers across architectures, environments and
the sim-to-real gap without retraining, reaching 83% accuracy on real-world tasks where prior
detectors collapse to chance.** *That "collapse to chance" claim about competitors on real
hardware is the sharpest sim-to-real warning in this section*, and it is consistent with SAFE's
own 58.7–64.2 unseen-real AUROC.

### 2.9 Repurposable video temporal grounding

The natural-video community solves "find the [t_start, t_end] matching this sentence", which is
formally identical to "find the moment the robot dropped the block". Leads **[A]**:
[Moment Quantization for VTG](https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_Moment_Quantization_for_Video_Temporal_Grounding_ICCV_2025_paper.pdf)
(ICCV 2025); Keyword-DETR (AAAI'25, QVHighlights **54.89 mAP** VGG / 61.08 SlowFast+CLIP);
**VTG-LLM** ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/download/32341/34496)), which
**injects explicit timestamp knowledge into video LLMs** — the same diagnosis KITE reaches
independently, that VLMs lack native time grounding.

**The metrics are the transferable part**: **R@1 at IoU ∈ {0.5, 0.7}**, mAP, HIT@1 — IoU against
a ground-truth failure *interval*, which is a much more honest target than a single timestep given
that failure onset is genuinely fuzzy. *Break on transfer*: these models train on natural, edited,
semantically diverse web video, whereas robot rollouts are static-camera, visually monotonous, and
the "moment" is often a few-pixel sub-second event. **Direct transfer is untested** **[?]**.

### 2.10 Which localisation methods need privileged simulator state

**Hard requirement — do not transfer without substitution:**

| Method | Privileged element |
|---|---|
| Classical HJ reachability / BRT | analytic dynamics `f` **and** a failure set `l(x)` over object poses, plus a state grid |
| REFLECT in simulation | **verified**: ground-truth object detection *and* object state. Substituting MDETR+CLIP costs **88.4% → 68.8%** explanation accuracy |
| Explicit contact-mode enumeration | object geometry + contact Jacobians; ContactNets exists specifically to remove the **contact-detection oracle** |
| **Any ground-truth failure-onset annotation** | **verified**: outcomes in the silent-failures study are read from **privileged simulator state that the detectors never observe**. On a real robot you often cannot construct the Stage-2 ground truth, let alone the detector. This is the deepest issue in the whole stage. |
| AHA / FailGen *training data* | needs a simulator to procedurally perturb successful demos. The model deploys without it; the data does not exist without it |
| Dead-end value functions | not privileged in principle, but need many episodes that actually reach undesired terminals — effectively simulation-only |

**Proprioception-only — transfers cleanly**: PerAct/RVT keyframes · AWE · ruptures/PELT/BOCPD/CUSUM
on joint or EE signals · BP-AR-HMM · TSC · the Grinsztajn reversibility classifier · gripper-state
segmentation.

**Policy-internals-only — transfers cleanly, and is where the field has landed**: SAFE ·
FailureSpot · Hide-and-Seek · FIPER · FAIL-Detect · Rewind-IL/TIDE · Sentinel's consistency half ·
Tri-Info · ActProbe. **None of these touch environment state at all.** Their common requirement is
instead a calibration set of *successful* rollouts and, for some, trajectory-level success labels
— which on a real robot means a human or an error-prone VLM judge.

**Vision-only — transfers with a measured perception tax**: VIP/LIV/GVL/PROGRESSOR · KITE ·
Code-as-Monitor · Foresight · video temporal grounding.

**Requires special hardware absent from most VLA platforms**: F/T-derivative + BOCPD contact
segmentation; Evetac and all tactile event detection.

### 2.11 Localisation: synthesis

1. **TAS is the wrong default and the field has moved on.** Every TAS method embeds a prior fitted
   to nominal executions — MS-TCN's T-MSE (λ=0.15, τ=4), DiffAct's position prior, TOT's diagonal
   prior, CTE's relative-time regression. A failed rollout violates exactly these priors. And its
   metrics cannot score boundary *timing*. The one exception worth trying is **ASOT**, whose
   unbalanced-OT term explicitly permits missing actions — but it is hyperparameter-fragile and
   k-means-init-dependent by the authors' own admission.
2. **Policy-internal signals beat everything else, and the margins are large.** FailureSpot 90.5
   timestamp AUROC on π₀ vs SAFE-MLP 76.6; Hide-and-Seek +13.1 bACC over a VLM monitor at
   1/2000th the latency; TIDE 0.95 bACC vs FAIL-Detect 0.83. The reason is structural: the
   policy's own prediction *inconsistency* is self-referential — no reference trajectory, no demo
   corpus, no environment state.
3. **Skill boundaries and failures are currently the same statistic** (§2.2). Nobody has
   disambiguated them. Real gap.
4. **Drop trajectory-distance-to-demo.** RoboEval 59.4%; MetaFine 0.90-stability/10%-success; the
   DTW stall-equivalence.
5. **Use complementary detectors or be structurally blind to a failure class.** Sentinel's +18%
   from consistency ∪ progress, and FIPER's requirement that OOD *and* action uncertainty both
   fire, say the same thing. A minimum viable stack is: (a) a **consistency** signal (TIDE /
   action-chunk disagreement), (b) a **progress** signal (GVL or a learned value with
   PROGRESSOR-style OOD push-back), (c) a **phase** signal (PerAct-style proprioceptive
   keyframes — free, and works on failures), (d) conformal calibration on successful rollouts.
6. **Adopt the surgical executional/procedural distinction.** Segmentation can only catch
   procedural errors.
7. **Stage-2 metrics should be**: timestamp-level AUROC; balanced accuracy swept over conformal α;
   **failure-onset time error**; **Time-Weighted Accuracy** (which penalises lateness); and, if
   you annotate *intervals*, **R@1 at IoU ∈ {0.5, 0.7}**. Explicitly **not** F1@{10,25,50}, edit,
   or MoF.
8. **State the ground-truth problem honestly.** Failure-onset labels are themselves privileged,
   and the circularity — *the same OOD observations that degrade the policy degrade the detector*
   — means Stage-2 accuracy from simulation should be read as an **upper bound**.

---
