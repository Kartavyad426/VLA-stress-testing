# FAILURE MINING METHODS — how people actually detect, localise, classify, cluster and attribute failures in robot policy rollouts

> ## STATUS — all five stages written
>
> **§0–§2 written 2026-09-15** (detection, localisation; 113 URLs, eleven localisation
> subsections). **§3–§9 written 2026-09-15 in a second pass** (classification, clustering,
> attribution, cross-cutting sim-vs-real, adjacent fields, what we could adopt, what could not be
> verified). 122 distinct URLs.
>
> The two passes are not of equal depth. §1–§2 drew on 113 sources; §3–§9 cite 12 further URLs from
> ~20 sources consulted, leaning on full-text readings of four central papers (LIBERO-Plus, RoboFAC,
> Causal Agent Replay, and the unsupervised-taxonomy paper) rather than on breadth. **§9 lists
> everything that could not be verified**, and it is longer for the second pass than the first.
>
> **Three results in §3–§9 bear on decisions currently open in `PLAN.md` / `HANDOFF.md`:**
> - **§3.5** — nobody validates a failure taxonomy as correct. κ measures self-consistency, not
>   correctness, and cannot do the job our design assigns it. §8.5 proposes the partial substitute.
> - **§3.3** — LIBERO-Plus is a **generator** (14,000 candidates from 40 tasks × 7 dimensions)
>   whose **released 10,030-task corpus is difficulty-filtered**: instances solved by all-or-most of
>   four reference models were deleted. Revert-one-knob is constructible; the released corpus is not
>   an unbiased robustness sample. This bears on the open AS-3 question.
> - **§7.1 / §8.1** — `ddmin` over the perturbation set finds **minimal failure-inducing knob
>   subsets** in O(n²) re-runs, closing the interaction blind spot that revert-one-knob structurally
>   cannot see. The highest-value import in the document.
>
> **§2.6's argument against trajectory-distance-to-demo remains unresolved in code** —
> `vla_harness/mining/phases.py` still keeps it as a caveated secondary signal. Flagged in §8, not
> changed in this pass.

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

## 3. CLASSIFICATION — "what kind of failure was it?"

Detection asks a binary question against a checker. Localisation asks *when*, and at least has a
metric space to be wrong in. Classification asks **which of a set of named categories this failure
belongs to** — and the set is invented by the people doing the asking. That is the whole
difficulty. There is no measurement error to bound, because there is nothing being measured; there
is a *construct*, and constructs are validated differently from estimators.

Three things follow, and the rest of this section is about them:

1. **The taxonomy is a design artefact, not a discovery.** Published taxonomies range from 4 to
   ~21 categories over the same underlying phenomenon (manipulation failure), and they are not
   refinements of each other — they cut on different axes.
2. **Where the label comes from determines what it can be used for.** A label injected by a
   generator is ground truth about the *injection*, not about the failure. A label applied by a
   human is an opinion with an agreement rate. A label emitted by a VLM is a prediction with an
   error rate nobody propagated.
3. **Almost nobody validates a taxonomy.** They validate *agreement* with it. §3.5 is the
   important part of this section and the answer it reaches is mostly negative.

### 3.1 Hand-designed taxonomies — the published category sets

The literature's category sets, side by side. Sizes are the leaf count.

| Source | Levels / size | Categories | How labels were produced |
|---|---|---|---|
| **RoboFAC** ([arXiv:2505.12224](https://arxiv.org/abs/2505.12224)) | 3 levels, **6 leaves** | Task Planning Error → *Step Omission*, *Wrong Object*; Motion Planning Error → *Position Deviation*, *Orientation Deviation*; Execution Control Error → *Grasping Error*, *Timing Error* | Injected by the generator for the deterministic fields; GPT-4o for the semantic fields, then manually reviewed **[F]** |
| **AHA / FailGen** ([arXiv:2410.00371](https://arxiv.org/abs/2410.00371)) | flat, **7** | No_Grasp, Slip, Translation (misaligned keyframe), Rotation (incorrect), No_Rotation, Wrong Action Sequence, Wrong Target Object | **Injected** — the label *is* the perturbation applied to a successful demo **[A]** |
| **SO-101 benchmark** ([arXiv:2606.08881](https://arxiv.org/abs/2606.08881)) | flat, **4** | Grasp Instability, Repetition Loop, State Mismatch, Precision Misalignment | Human, replaying logs; **annotators and agreement not reported** **[F]** |
| **LIBERO-Plus** ([arXiv:2510.13626](https://arxiv.org/abs/2510.13626)) | 2 levels, **7 → 21** | Objects Layout, Camera Viewpoints, Robot Initial States, Language Instructions (**paraphrase only** — see §5.3), Light Conditions, Background Textures, Sensor Noise | **Not a failure taxonomy at all** — a taxonomy of *causes applied*. See §3.3 |
| **REFLECT / RoboFail** ([arXiv:2306.15724](https://arxiv.org/abs/2306.15724)) | flat | execution / planning failures over a symbolic scene graph | Human, over scripted faults **[A]** |

**Read the first column against the second.** RoboFAC's top level (*planning / motion / execution*)
is a **pipeline-stage** cut: it names the module you would go fix. FailGen's seven are a
**geometric-perturbation** cut: they name the degree of freedom that was wrong. SO-101's four are a
**symptom** cut: they name what an observer sees. LIBERO-Plus's seven are an **environmental-cause**
cut: they name what the experimenter changed.

These are four different questions, all called "failure classification". A single episode — the arm
closes early on a bowl, the bowl slips, the arm continues its scripted approach and knocks the plate
— is `Execution Control / Grasping Error` to RoboFAC, `Slip` to FailGen, `Grasp Instability` to
SO-101, and quite possibly `Objects Layout` to LIBERO-Plus. **All four are correct.** None of them
is more correct. There is no experiment that distinguishes them, because they are answers to
different questions.

*Why this matters to a client manifest*: the cut you choose determines what the label can be acted
on. A symptom cut is cheap to annotate and useless as a remediation instruction. A pipeline-stage
cut is an instruction but presumes the client's system has those stages. A cause cut is the only one
that is directly actionable *and* the only one that requires you to have controlled the cause — which
in simulation you did, and on a client's real robot you did not.

### 3.2 Procedurally-generated taxonomies — where the label is the injection

**FailGen** ([arXiv:2410.00371](https://arxiv.org/abs/2410.00371)) is the reference implementation of
the trick our own toy harness uses. *Mechanism* **[A]**: take a successful demonstration, represented
as a keyframe sequence; pick one of seven perturbation operators; apply it; re-execute. The operators
are geometric and trivially parameterisable — offset a keyframe's translation, delete its rotation
component, widen the gripper at the grasp keyframe so the grasp does not close (`No_Grasp`), weaken
it so contact is lost mid-transport (`Slip`), permute two keyframes (`Wrong Action Sequence`),
retarget the grasp keyframe to a distractor (`Wrong Target Object`). The resulting trajectory is a
failure **whose category is known by construction**, because you chose it before running anything.

This buys the one thing the whole stage otherwise lacks: **ground truth with zero annotation cost and
zero annotator disagreement.** AHA is trained on it and transfers — beating GPT-4o in-context by
10.3% and six-model average by 35.3% on real failure data **[A]**.

**And here is the cost, which is structural and not fixable by scale.** The taxonomy is now *the set
of perturbations somebody was able to write*, not the set of failures that occur. Every category is,
by construction, a failure a scripted operator can produce from a success — which biases the entire
category set toward **single-keyframe, geometric, discrete** faults, and away from everything that is
distributed over time, emergent from closed-loop interaction, or a consequence of the policy's own
compounding error. `Repetition Loop` (SO-101's second category, and one of the most commonly observed
VLA failures in practice) **cannot be produced by perturbing a demonstration keyframe at all.** It is
a closed-loop pathology. It is absent from FailGen's seven for exactly that reason.

This is the same structural blindness §2 identified for demo-referenced localisers (§2.1, and the
header finding of §2), arriving one stage later by a different route: *methods built by perturbing
nominal executions inherit the nominal execution's structure, and are blind wherever real failure
does not respect it.*

The counterweight number is from FailBench (§1): **75% of its 2,197 pooled failures occurred
naturally** rather than being synthetically induced **[F]**. A synthetic taxonomy has never been
checked against that 75% — no published work maps FailGen's seven onto a corpus of natural failures
and reports the residual. **That residual is the single most useful unpublished number in this
section**, and it is cheap for us to produce (§8).

> ⚠ **Direct consequence for us.** Our oracle gate plants faults and checks the classifier recovers
> them, which is precisely FailGen's construction and inherits precisely its limitation: it can only
> ever validate the classifier on the fault families we were able to plant. **A passing oracle gate
> is evidence of implementation correctness, not of taxonomy coverage.** Those are different claims
> and `ARCHITECTURE.md` §8 should not be read as establishing the second.

### 3.3 Cause taxonomies vs failure taxonomies — and LIBERO-Plus as the worked example

LIBERO-Plus is worth a subsection because it is the closest published thing to what we are building,
and because reading it carefully answers a question that has been open in our own planning.

*Mechanism* **[F]**, read from the paper directly: start from LIBERO's 40 evaluation tasks; for each
of the four suites (Spatial, Object, Goal, Long), generate **500 instances per generalization
dimension** across seven dimensions — **14,000 candidate tasks**. Then filter: *"Tasks that were
solved by all models, or by a large majority, were removed to avoid ceiling effects"*, and the
remainder balanced across sub-dimensions. **The released benchmark is 10,030 tasks over 7 dimensions
and 21 sub-dimensions.**

Two things fall out of that paragraph, and both matter more than the headline results.

**(a) It is a generator with a corpus released from it — but the generator itself does not ship.**
The seven dimensions are applied as *single-dimension perturbations to an existing task* — exactly
the revert-one-knob structure that counterfactual attribution requires (§5.5). The paper lists
"Automation: automated task generation" as a stated contribution, so the knob API exists upstream of
the release.

It is **not distributed**. The repository ([sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus))
ships the 10,030 pre-generated task instances, an `assets.zip` of *"hundreds of new objects, textures,
and other required assets"* (`articulated_objects/`, `new_objects/`, `scenes/`, `textures/`, plus
`.xml`/`.stl` files), a `task_classification.json` mapping task IDs to perturbation category and
difficulty level, and RLDS/LeRobot training datasets. It installs as a **drop-in replacement for the
LIBERO repository**. What the README does *not* document is any script, knob specification or config
for generating new perturbed instances **[A — README-level read; a definitive answer requires cloning
and listing the tree, which is cheap and has not been done]**.

**A different group's generator does ship.** **LIBERO-PRO**
([github.com/Zxy-MLlab/LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO)) releases
`perturbation.py` plus an `evaluation_config.yaml` exposing its dimensions as boolean knobs
(`use_swap`, `use_object`, `use_language`, `use_task`) with spatial-displacement **intensity levels
`x0.1` through `x0.5`**, described as *"combinable and configurable via YAML for scalable and
controlled generalization studies"* **[A — README-level read; not cloned or run]**. That is a knob
API with a magnitude axis, which is what revert-one-knob and `ddmin` (§7.1) both need, and it covers
the **object-swap and language dimensions that LIBERO-Plus's released corpus does not** (§5.3).
Its README nonetheless steers users toward pre-built files from Hugging Face, so the generation path
may carry undocumented dependencies — **verify by running it before planning around it.**

**And LIBERO-Plus's own ingredients ship even though its generator does not:** the asset library, the scene and texture sets, and a per-instance label saying
which knob produced it. Reverting one knob is then a matter of diffing a perturbed instance against
its nominal parent and undoing the difference — recoverable from released artefacts, at the cost of
building the reversion ourselves rather than calling theirs.

**(b) The released 10,030 are a difficulty-filtered sample, and that breaks one specific use.** Tasks
solved by all-or-most of four reference models were **deliberately deleted**. Any success rate
computed over the released corpus is therefore an estimate over a population that was selected, after
the fact, for being hard for OpenVLA-OFT, π₀, π₀-fast and UniVLA. It is a perfectly good *stress*
corpus and a perfectly good relative comparator. It is **not** an unbiased estimate of robustness
under perturbation, and a difference between two policies measured on it is confounded with how
similar each policy is to those four. Nothing in the paper misuses it this way; a downstream user
easily could.

*Numbers* **[F]** — Table 1 in full, success rate (%) per dimension with absolute drop beneath.
This is the per-dimension sensitivity profile for ten checkpoints and it is the most directly usable
table in this document:

| Model | Original | Camera | Robot | Language | Light | Background | Noise | Layout |
|---|---|---|---|---|---|---|---|---|
| OpenVLA | 76.5 | 1.1 ↓75.4 | 4.1 ↓72.4 | 26.8 ↓49.7 | 4.4 ↓72.1 | 25.3 ↓51.2 | 19.3 ↓57.2 | 31.6 ↓44.9 |
| OpenVLA-OFT | 97.1 | 59.7 ↓37.4 | 37.2 ↓59.9 | 81.5 ↓15.6 | 85.8 ↓11.3 | 92.4 ↓4.7 | 76.7 ↓20.4 | 77.1 ↓20.0 |
| OpenVLA-OFT_w *(3rd-person only)* | 95.3 | 16.8 ↓78.5 | 43.7 ↓51.6 | 73.2 ↓22.1 | 68.2 ↓27.1 | 92.5 ↓2.8 | 51.4 ↓43.9 | 72.3 ↓23.0 |
| OpenVLA-OFT_m *(mix-sft)* | 97.6 | 57.9 ↓39.7 | 30.6 ↓67.0 | 83.6 ↓14.0 | 91.6 ↓6.0 | 83.6 ↓14.0 | 76.3 ↓21.3 | 73.2 ↓24.4 |
| π₀ | 94.2 | 15.8 ↓78.4 | 6.6 ↓87.6 | 61.0 ↓33.2 | 79.6 ↓14.6 | 78.5 ↓15.7 | 79.4 ↓14.8 | 70.4 ↓23.8 |
| π₀-fast | 85.5 | 66.4 ↓19.1 | 24.8 ↓60.7 | 63.3 ↓22.2 | 73.0 ↓12.5 | 67.7 ↓17.8 | 75.8 ↓9.7 | 70.3 ↓15.2 |
| Nora | 87.9 | 4.0 ↓83.9 | 41.1 ↓46.8 | 67.0 ↓20.9 | 31.0 ↓56.9 | 50.5 ↓37.4 | 17.6 ↓70.3 | 63.9 ↓24.0 |
| WorldVLA | 79.1 | 0.3 ↓78.8 | 30.2 ↓48.9 | 44.2 ↓34.9 | 29.4 ↓49.7 | 14.5 ↓64.6 | 12.2 ↓66.9 | 39.4 ↓39.7 |
| UniVLA | 95.2 | 4.3 ↓90.9 | 50.3 ↓44.9 | 71.8 ↓23.4 | 59.1 ↓36.1 | 80.0 ↓15.2 | 25.3 ↓69.9 | 34.3 ↓60.9 |
| RIPT-VLA | 97.5 | 58.3 ↓39.2 | 36.7 ↓60.8 | 80.1 ↓17.4 | 87.9 ↓9.6 | 90.4 ↓7.1 | 73.8 ↓23.7 | 76.5 ↓21.0 |

Finding 1 — fragility across all seven. Finding 2 — worst on **camera viewpoint** and **robot initial
state** ("require a high-level understanding of spatial geometry and proprioception"), most resilient
to **lighting and background** ("superficial, low-level visual changes"). Finding 4 — a **first-person
wrist camera** is the single biggest architectural protection against viewpoint shift: OpenVLA-OFT
59.7 on Camera versus its third-person-only variant's 16.8, same training otherwise.

**The ordering is stable across ten checkpoints spanning four architectures**, which is what makes it
usable as an external reference: Camera and Robot dominate; Language is consistently among the
smallest drops; Background and Light are smallest. A miner whose output inverts that ordering on a
new policy has either found a genuinely different policy or has a bug, and those are worth
distinguishing.

**Finding 3 is the one to read.** Language perturbation produces *the second smallest* average drop
(−25.3). The authors do not accept this as linguistic robustness and go looking, which produces the
material in §5.4 below. Note also the decomposition of Objects Layout into **confounding objects**
(adding distractors — most models barely move) versus **target displacement** (moving the target —
large drops), and the authors' reading: the models *"may have merely learned the positional
information of the target objects"* **[F]**.

> That decomposition is a methodological lesson independent of its result. A single "objects layout"
> factor would have shown a moderate drop and supported a bland conclusion. Splitting it into two
> sub-factors that move in opposite directions is what produced the finding. **The granularity of
> your cause taxonomy is not a presentational choice; it is the resolution limit of every conclusion
> you can draw from the sweep.** Cf. F2, where suite-level aggregation hid an 80% → 28% collapse on
> a single task.

### 3.4 Discovered taxonomies — letting the corpus name its own categories

The alternative to designing a category set is inducing one. Two mechanisms, from different fields.

**Unsupervised Discovery of Failure Taxonomies from Deployment Logs**
([arXiv:2506.06570](https://arxiv.org/abs/2506.06570)). *Mechanism* **[F]**, four stages: (i)
**semantic downsampling** of the rollout video; (ii) **failure reasoning** — a VLM (Gemini 2.5 Pro)
produces a free-text chain-of-thought explanation per episode; (iii) **taxonomy discovery** — and
here is the departure from convention: *the clustering is not done in an embedding space at all*. An
LLM is used as the optimiser over the set of explanations, generating several independent candidate
taxonomies and then reconciling them ("ensemble-and-refine") to damp prompt sensitivity. **The
cluster count L is never specified**; it is implicitly optimised against stated criteria — semantic
coherence, minimal inter-cluster overlap, coverage. (iv) each cluster gets a name, a description and
keywords; then every trajectory is assigned to one.

*Numbers* **[F]**, against RoboFail's expert annotations: Cluster Precision **0.920** (vs 0.875 for a
BERTopic+LLM baseline), Taxonomy Coverage **1.0**, Semantic Alignment Score **0.958** (harmonic mean
of the two). Trajectory assignment F1 **85.53%** versus **32.41%** for an embedding-similarity
baseline — a 2.6× gap that is the strongest published argument against naive embedding clustering of
failures. The BERTopic clusters were *"broad and overlapping, merging conceptually distinct
categories"* **[F]**.

*Where it breaks*: the authors state plainly that **"there is no single canonical failure
taxonomy"**, and that discovered structures may be sensitive to the clustering strategy **[F]**. For
the two domains without expert annotations (driving, navigation) the evaluation is *qualitative
coherence* — the driving taxonomy "aligned with the U.S. DoT Volpe Center's pre-crash typology", the
navigation clusters "matched failure types previously identified manually". That is validation by
resemblance to a prior human taxonomy, which is §3.5's problem restated.

**TnT-LLM** ([arXiv:2403.12173](https://arxiv.org/abs/2403.12173), KDD'24). *Mechanism* **[A]**: a
two-phase pipeline — (1) summarise each document, then build the taxonomy by a *pseudo-gradient*
loop: show the LLM a batch of summaries plus the current taxonomy, ask for an update, iterate over
batches; (2) use the LLM as a **labeller** to produce training data for a *lightweight supervised
classifier* that is what actually ships. The second phase is the transferable idea: the expensive
model builds the category set and labels a sample, and a cheap deterministic model does the
production labelling. That is auditable and re-runnable in a way an LLM call is not — which is
directly relevant to our own open tier-3-LLM decision.

### 3.5 **How anyone validates that a taxonomy is correct** — the honest answer

This is the section the rest of §3 exists to set up, and the finding is largely negative: **no
published work in this space validates that its taxonomy is correct. Four weaker things are done in
its place, and it is worth being precise about what each one actually establishes.**

**(1) Recovery of planted labels.** Inject a known fault, check the classifier returns it. FailGen,
our own oracle gate, and — in the adjacent slice-discovery literature — Domino's evaluation framework
([arXiv:2203.14960](https://arxiv.org/abs/2203.14960)) do this. Domino is the most rigorous instance
anyone has built: **1,235 slice-discovery settings across three input domains with ground-truth
planted slices**, on which the best method (Domino itself) *"accurately identifies 36% of the 1,235
slices"* — a 12-point improvement on prior work **[S]**.

> **Sit with that number.** In the field that has built the most careful planted-ground-truth
> evaluation of "find the coherent subpopulation where the model fails", the state of the art
> recovers **roughly one in three** planted structures. Our own stage-3/4 machinery is doing the same
> job on a harder input domain with a far smaller evaluation. Any implied accuracy above Domino's 36%
> is a claim that needs its own evidence.
>
> *What this establishes*: internal correctness — the pipeline can find what is there.
> *What it does not establish*: that the planted categories are the categories that occur.

**(2) Inter-annotator agreement.** Two humans label independently under a shared rubric; report
Cohen's κ (mutually exclusive categories) or Jaccard (multi-label); resolve disagreements by
discussion **[S]**. Landis–Koch bands: 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial,
0.81–1.00 almost perfect **[S]**. The reference points already in this document: **97.0%** human–human
agreement on *outcome*, **91%** on *primary failure phase* **[S]** — and those are the same annotators
getting monotonically worse as the question gets deeper. No robot-failure paper surveyed here reports
κ on *category* at all; SO-101 does not report who annotated **[F]**, RoboFAC reports none **[F]**.

Two cautions. First, **κ measures whether the rubric can be applied consistently, not whether it
carves the phenomenon correctly.** A taxonomy of {*failure on a Tuesday*, *failure not on a Tuesday*}
achieves κ = 1.0. High agreement is necessary and nowhere near sufficient. Second, the **kappa
paradox**: under heavy class imbalance κ is depressed even at high raw agreement, so κ should always
be reported alongside percentage agreement and a paradox-resistant statistic such as Gwet's AC₁ **[S]**.
Failure categories are *always* heavily imbalanced.

> ⚠ **Directly on our open problem.** Our design makes κ ≥ 0.5 the sole validation of family
> assignment, because LIBERO offers no family fixture. Two things this section establishes about
> that: (a) κ ≥ 0.5 is "moderate" — and it is being asked to carry a load that *no surveyed paper
> asks κ to carry at all*, since they mostly do not measure it; (b) κ is the wrong instrument for
> the question regardless of threshold, because it cannot distinguish a good taxonomy from a
> consistently-applicable one. Raising the threshold does not fix a construct-validity problem.
> §8 proposes what to do instead.

**(3) Agreement with an independently-constructed taxonomy.** The strongest available design, and it
appears once in this document — MANGO building a second symbolic oracle for the same LIBERO_10 tasks
by a different process and reporting F1 0.841 against the first (§0) **[F]**. The equivalent for
categories has not been done for robot failures. The unsupervised-taxonomy paper's appeal to the DoT
Volpe pre-crash typology is a weak version: resemblance to a prior taxonomy, assessed qualitatively,
not agreement measured per-episode **[F]**.

**(4) Downstream utility.** Does the label change what anyone does, and does acting on it help? Two
instances: TnT-LLM explicitly frames taxonomy quality in terms of *downstream classification utility*
rather than intrinsic quality **[A]**; the unsupervised-taxonomy paper reports that the discovered
taxonomies *"guide targeted data collection for offline policy refinement and enhance runtime failure
monitoring"* **[F]**.

**This fourth one is the only validation that survives contact with the objection.** A taxonomy is a
means to an intervention. If two taxonomies prescribe the same intervention they are equivalent for
our purposes however differently they carve; if a category never changes an action it is decoration
regardless of its κ. The test is: **partition the corpus by category, apply the remediation each
category prescribes, and measure whether the per-category improvement is larger than the improvement
from the same remediation budget spent uniformly.** That is an experiment, it needs no ground-truth
labels, and it is the only thing in this section that would falsify a taxonomy.

It is also expensive, and nobody in the surveyed literature has run it.

### 3.6 Classifiers: what actually assigns the label

Three mechanisms, in increasing order of cost and decreasing order of auditability.

**Rules over privileged state.** A decision procedure over `gripper`, `holding`, contact flags and
eef-to-object distances. Deterministic, auditable, free, and re-runnable — the properties that matter
for an evidence trail. Breaks in exactly one way, and it is the way that bites: **the rule is bound
to state-key names and threshold values calibrated on one environment**, and silently mislabels when
either changes. Our own LE-2 is the canonical instance — a `link` substring filter dropped drawers
and cabinet doors, `_gt_nearest_object` named the wrong thing, and the spatial-reasoning rule fired
on it. Note that nothing *errored*: the pipeline produced a confident, wrong category. This is the
failure mode of rule classifiers generally, and it is invisible without an oracle gate.

**Learned classifiers over trajectory features.** Cheap at inference, need labels to train, and
inherit the label source's biases wholesale. Nothing surveyed here reports one beating a VLM on
category assignment for manipulation, but note §1.8's item 5 — in the adjacent LLM-agent field a
TF-IDF surface detector beat every LLM judge configuration at detecting false success (AUROC
0.83/0.95 vs ≤0.65/0.54) **[F]**. **Build the dumb baseline before the sophisticated one** applies to
stage 3 at least as strongly as to stage 1.

**VLM classifiers.** RoboFAC-7B is the best-documented **[F]**: 7B, fine-tuned on 9,440 erroneous
trajectories (8,960 sim + 480 real) plus 1,282 successes, 78,623 video-QA pairs over 16 tasks and 53
scenes, with eight question types spanning Task Identification, Task Planning, Failure Detection,
Failure Identification, Failure Locating, Failure Explanation, and High-/Low-level Correction.
RoboFAC-Bench results: short-horizon **82.74**, medium **84.92**, long **81.78**, dynamic **83.28**,
**real-world 68.94**, overall **79.10** — and +34.1% over GPT-4o.

**The real-world row is the finding.** A ~14-point drop from the sim average to the real split,
within a single model, on its own benchmark, with 480 real trajectories in training. Whatever
category accuracy we measure in simulation, the corresponding real-robot number is materially lower —
and note that this is the *optimistic* direction of the comparison, since the model saw real data.

Set this against §1's FailBench result that **RoboFAC-7B scores 0.51 macro balanced accuracy at
detection** — chance **[F]**. The same model is strong on its own benchmark and at chance on a pooled
external one. That is not a contradiction; it is the definition of overfitting to a failure
distribution, and it is §1.8's item 4 recurring at stage 3. **Category accuracy reported on the
benchmark a model was built around carries almost no information about category accuracy on ours.**

### 3.7 Classification: synthesis

1. **The taxonomy is a design decision with no correct answer.** Four published sets cut on four
   different axes — pipeline stage (RoboFAC), perturbation geometry (FailGen), symptom (SO-101),
   environmental cause (LIBERO-Plus). One episode maps to all four. Choose the cut that matches the
   *intervention* you intend to prescribe, and say which cut it is.
2. **Nobody validates a taxonomy as correct.** They report planted-label recovery (internal
   correctness), κ (self-consistency), resemblance to a prior taxonomy (weak), or downstream utility.
   **Only downstream utility can falsify a taxonomy**, and no surveyed work runs it.
3. **κ cannot do the job we have assigned it.** It cannot distinguish a good taxonomy from a
   consistently-applicable one, and it is depressed by the class imbalance that failure categories
   always have. Report percentage agreement and Gwet's AC₁ alongside it, and stop treating the
   threshold as the load-bearing check.
4. **Planted-fault validation tops out lower than intuition suggests.** Domino: 36% recovery over
   1,235 settings, and that is the careful state of the art **[S]**.
5. **Procedurally-generated taxonomies are structurally blind to closed-loop pathologies.**
   `Repetition Loop` cannot be produced by perturbing a demonstration keyframe; it is therefore
   absent from FailGen's seven. 75% of FailBench's failures were natural **[F]**; the residual
   against a synthetic taxonomy has never been measured.
6. **Discovered taxonomies beat embedding clustering by a wide margin** — 85.53% vs 32.41%
   trajectory-assignment F1, and BERTopic's clusters were broad and overlapping **[F]**. If we
   cluster, do not cluster raw trajectory embeddings. §4.
7. **Sub-factor granularity is the resolution limit of every conclusion.** LIBERO-Plus's split of
   objects-layout into confounding-vs-displacement is what produced its finding; F2 is our own
   version of the same lesson.
8. **Category accuracy does not transfer.** RoboFAC-7B: 79.10 on its own benchmark, 68.94 on its own
   real-world split, **0.51 balanced accuracy on FailBench** **[F]**.
9. **LIBERO-Plus is a generator whose released corpus is difficulty-filtered.** Single-dimension
   perturbations are revert-one-knob compatible; the 10,030 released tasks had all-model successes
   deleted and are not an unbiased robustness sample **[F]**.

---

## 4. CLUSTERING — "which of these are the same failure?"

Classification assigns each episode to a pre-existing category. Clustering asks the corpus to
produce the categories, and — more importantly for anything shipped to a client — to produce
**recurring modes with counts**. One failure is an anecdote; forty instances of the same failure
with a shared trigger is a manifest row. The stage's job is the word *same*.

Everything hinges on a decision made before any clustering algorithm runs: **what object is being
clustered.** The algorithm choice is nearly irrelevant by comparison, and the published evidence on
that is unusually clear.

### 4.1 What to embed — the decision that determines the outcome

Five choices appear in the literature, in ascending order of how well they work.

**(a) Raw trajectory embeddings.** Encode the state/action sequence, cluster the vectors. *Where it
breaks*: the dominant variance in a trajectory corpus is **task identity and object position**, not
failure mode. Two episodes failing the same way on different tasks are far apart; two episodes on the
same task failing differently are close. You recover a partition of the *task distribution* and read
it as a partition of the *failure distribution*. The measured cost is large: **32.41%
trajectory-assignment F1 for an embedding-similarity baseline against 85.53% for reasoning-space
clustering on the same corpus** **[F]** ([arXiv:2506.06570](https://arxiv.org/abs/2506.06570)).

**(b) Perceptual embeddings with a distance-to-centroid rule.** Reduce a visual world model's image
embeddings by PCA, k-means the successes, flag by distance to the nearest centroid **[S]**. This is a
detector wearing a clustering algorithm's clothes — it produces a scalar novelty score, not a set of
named modes — and it inherits (a)'s problem that scene appearance dominates.

**(c) Cross-modal embeddings.** Domino ([arXiv:2203.14960](https://arxiv.org/abs/2203.14960))
embeds inputs in a joint image–text space (CLIP-family), so that a discovered slice can be
*described* by finding text that lands near its centroid. This is the mechanism that makes a cluster
nameable rather than merely indexed **[A]**.

**(d) Failure-aware embeddings.** Embed not the input but the **evidence of failure** — the model
output together with the verifier's outcome, or a description derived from the mismatch between
output and ground truth — then cluster those, with HDBSCAN marking outliers as noise **[S]**. The
principle generalises cleanly: *embed the discrepancy, not the episode.*

**(e) Semantic reasoning space.** Have a VLM write a free-text explanation of each failure and
cluster the **explanations** **[F]**. This is (d) taken to its conclusion, and it is the design that
produced 0.920 cluster precision and 1.0 coverage against RoboFail's expert annotations.

> **The ordering (a) → (e) is a single idea repeated: cluster the thing that differs between a
> failure and its success, never the episode.** An episode's representation is dominated by the task;
> a discrepancy's representation is dominated by the failure. Every method that works has found this
> route, and the two that have been measured head-to-head on one corpus differ by 2.6×.

There is a cost, and it should be stated plainly: (e) puts a VLM — the component §1 measured at 0.77
macro balanced accuracy at its *easiest* task **[F]** — upstream of every cluster boundary. The
clustering is then no better than the explanations, and explanation quality on contact-rich
manipulation is where VLMs are worst (<0.60 **[F]**). Nobody has propagated that error into a cluster
purity bound. **A published cluster precision of 0.920 is conditional on explanations the same
literature elsewhere measures as unreliable**, and the two numbers have never been reconciled.

### 4.2 Algorithms, and the one property that matters

Given a good embedding, the algorithm is close to a free choice — with one exception.

**k-means** partitions exhaustively: every point lands in a cluster. On a failure corpus this is
actively wrong, because failure corpora have **heavy singleton tails** — one-off, genuinely
idiosyncratic events — and forcing them into the nearest mode inflates that mode's count. The count
is the number the client acts on.

**HDBSCAN** is density-based and **labels low-density points as noise** rather than assigning them
**[S]**. That noise label is not a defect; it is the honest output for a singleton, and it is the
reason HDBSCAN is the default in the slice-discovery and error-analysis literature. Its cost is that
"noise" can absorb a real but small mode, so the noise set must itself be inspected, not discarded.

**Error-aware mixture models.** Domino's contribution: fit a mixture over the embedding **jointly
conditioned on the model's predictions and its errors**, so that components are encouraged to be
error-homogeneous rather than merely input-homogeneous **[A]**. Reported effect: mean precision@10
**0.639** on noisy and rare slices in natural images, *"a 105% improvement over the next-best
method"* **[S]**.

**LLM-as-optimiser.** No embedding space and no distance function at all: show the model the
explanations and ask for a partition, generate several candidate taxonomies independently, reconcile
them **[F]**. This is the 0.958-SAS method. It scales badly with corpus size (the whole set must pass
through context, or be batched with a reconciliation step), and it is not reproducible in the strict
sense — which for an audit trail is a real cost and is the substance of our open tier-3-LLM decision.

### 4.3 Choosing the number of clusters — where the standard advice is wrong

Three approaches and their failure modes.

**Internal validity indices** (silhouette, Dunn, Calinski–Harabasz). Compute a ratio of within- to
between-cluster dispersion and pick the k that optimises it. *Where it breaks*: these indices encode
a **geometric prior — compact, convex, roughly equal-sized clusters** — that failure-mode structure
does not satisfy. Failure modes are unequal by construction (that is what prevalence means) and often
non-convex in an embedding. Silhouette is explicitly less reliable for non-convex or intricately
shaped clusters **[S]**, and Dunn shows very large variance at its optimum **[S]**. Worse, high
silhouette is achievable by assignments that are **not reproducible between runs** **[S]** — the index
and the stability of the thing it scores are decoupled.

**Bootstrap stability.** Resample, re-cluster, measure assignment agreement; prefer the k that is
stable. This is the check most often presented as the rigorous one, and the objection to it is
sharp: **stability cannot distinguish real clusters from noise.** A k-means run on structureless data
is highly stable, because k-means is close to deterministic given its initialisation — so high
stability is *"evidence that your k-means run is deterministic, not that your clusters exist"*
**[S — secondary source, a practitioner blog; the argument is sound and reproducible in ten lines of
code, but it is not a peer-reviewed citation and should not be cited as one]**.

**Let the method choose.** The LLM-as-optimiser approach never specifies L; it is implicitly
optimised against coherence, non-overlap and coverage criteria **[F]**. Honest about what it is — a
qualitative criterion applied by a stochastic judge — and it sidesteps the geometric prior entirely.

> **The load-bearing check none of the three performs is a null comparison.** Run the identical
> pipeline on a corpus where no mode structure exists by construction — failures sampled uniformly,
> or labels permuted — and report the same statistic. If the null corpus yields clusters of
> comparable silhouette and stability, the statistic is measuring your algorithm, not your data.
> This is the clustering analogue of the control arm that our oracle gate already runs for stage 3
> (`ARCHITECTURE.md` §8), and we do not currently run it for stage 4. **Cheap, and it is the
> difference between a mode and an artefact.**

### 4.4 Naming clusters — and how naming launders incoherence

A cluster with a count is not yet a manifest row; it needs a name a human can act on. Mechanisms:
nearest-text-neighbour in a cross-modal space (Domino) **[A]**; LLM summarisation of cluster members
into a name, description and keywords (the unsupervised-taxonomy paper generates exactly this triple
per cluster) **[F]**; interactive LLM-generated cluster descriptions
([LangLasso, arXiv:2601.10458](https://arxiv.org/abs/2601.10458)) **[S]**.

**The hazard is specific and under-discussed.** An LLM asked to name a set of items will *always*
return a fluent name, including for a set with no common property — and the name's fluency is then
read as evidence that the cluster is coherent. The naming step is the point at which an
under-determined clustering acquires a false appearance of meaning, and it is downstream of every
check in §4.3, so no check catches it.

The counter is cheap and nobody reports it: **name-based re-assignment.** Give a held-out annotator
(or a second model) only the *names and descriptions*, ask them to assign held-out episodes, and
measure agreement with the cluster assignment. If the name does not reproduce the partition, the
name is a summary of noise. This is the same instrument as §3.5(2), pointed at the right target —
and unlike κ on a hand-designed taxonomy it *is* a test of the partition, because the partition was
induced rather than given.

### 4.5 Evaluating cluster quality without ground truth

Ranked by how much they establish.

| Method | What it establishes | Cost |
|---|---|---|
| Internal indices (silhouette, Dunn) | that the geometry matches a convex-compact prior | free |
| Bootstrap stability | that the algorithm is deterministic | cheap |
| **Null-corpus comparison** | that the structure is not an artefact of the pipeline | one extra run |
| **Name-based re-assignment** (§4.4) | that the cluster has a communicable common property | one annotator pass |
| **Planted slices** (Domino-style) | that the pipeline recovers structure that is there | needs a fixture |
| **Downstream utility** | that the partition changes what you do, usefully | an experiment |

Domino's framework is the only large-scale planted-ground-truth evaluation of this stage in any
field: **1,235 settings, 36% recovery for the best method** **[S]**. And the metric it established —
**precision@10**, how many of the top-10 items in a discovered slice belong to the ground-truth slice
**[S]** — is the right shape for our use too, because a manifest row is read top-down: what matters
is whether the *examples shown to the client* are actually the same failure, not whether the cluster
tail is pure.

**Two further cautions from the adjacent literature.** BERTopic — the obvious off-the-shelf choice —
underperformed on failure explanations specifically, with clusters *"broad and overlapping, merging
conceptually distinct categories"* **[F]**. And the same source that reports 0.920 cluster precision
against expert annotations falls back to **qualitative coherence assessment** for its two domains
without annotations **[F]**; the honest reading is that the quantitative result exists only where a
fixture existed, which is the situation we are in on LIBERO.

### 4.6 Clustering: synthesis

1. **Cluster the discrepancy, not the episode.** Embedding-similarity on trajectories: 32.41% F1.
   Clustering VLM-written failure explanations: 85.53% on the same corpus **[F]**. This is the
   largest single-decision effect in the stage.
2. **Prefer density-based methods that can say "noise".** Failure corpora have heavy singleton tails
   and k-means inflates mode counts by forcing them in — and the count is what the client acts on.
3. **Do not choose k by silhouette.** It encodes a convex-equal-size prior that failure modes
   violate, and high silhouette is achievable by non-reproducible assignments **[S]**.
4. **Bootstrap stability is not evidence of structure.** It largely measures algorithmic determinism
   **[S, secondary]**.
5. **Run a null corpus.** The one check that separates a mode from an artefact, and the one nobody
   reports. We already do the equivalent at stage 3 and should extend it.
6. **Naming launders incoherence.** An LLM names any set fluently. Validate by name-based
   re-assignment on held-out episodes.
7. **Planted-slice recovery is the strongest available fixture and it tops out at 36%** **[S]**.
   Report precision@10 rather than cluster purity — it matches how a manifest row is read.
8. **A VLM sits upstream of every cluster boundary in the best-performing designs**, at 0.77 macro
   balanced accuracy on its easiest task and <0.60 on contact-rich manipulation **[F]**. That error
   has never been propagated into a cluster-purity bound. Ours should be.

---

## 5. ATTRIBUTION — "what caused it?"

This is the stage where the published numbers collapse. Best step-level attribution on the
Who&When LLM-agent benchmark: **14.2% accuracy, with some methods below random** **[A]**. Best
long-context model on TRAIL: **11%** **[S]**. Those are not hard benchmarks being approached; they
are benchmarks nobody is close to.

It is worth being clear about why, because the reason is not "the models are not good enough yet".
**Attribution is a causal question, and almost every deployed method answers it with correlational
evidence.** A judge reading a trajectory and naming a cause is performing post-hoc narration: it
observes the failure and produces a story consistent with it, and there is no mechanism in that
procedure by which a *wrong but consistent* story would be penalised. Consistency is what it
optimises. Causation is what was asked.

The stage divides cleanly on exactly that line.

### 5.1 Correlational attribution — reading the trace and naming a cause

*Mechanism*: give a model the trajectory (frames, actions, tool calls, logs), ask which step or
factor was responsible. Everything from `RoboFAC`'s Failure Explanation question type **[F]** to the
LLM-judge baselines on Who&When sits here.

*The benchmark*: **Who&When** ([search-level](https://arxiv.org/abs/2505.00212)) — 184 annotated
failure tasks from algorithm-generated (CaptainAgent) and hand-crafted (Magnetic-One) multi-agent
systems, each labelled with the responsible agent, the decisive error step, and a natural-language
explanation **[S]**. Extended by **Who&When Pro** ([arXiv:2607.09996](https://arxiv.org/abs/2607.09996))
to 12,326 traces over 26 source benchmarks, 9 task categories, 3 modalities **[S]**.

*The number*: ~**14%** step-level accuracy for correlational baselines **[F, as reported in
[arXiv:2606.08275](https://arxiv.org/abs/2606.08275)]**.

*Where it breaks*, and this is the structural point: **the benchmark labels one deterministic
failure-inducing step per trace**. Real failures frequently have no such step — the outcome is
overdetermined, or committed gradually. So part of the 86% gap is method weakness and part is the
ground truth being an idealisation. Both parts should make us cautious about promising step-level
attribution on robot rollouts, where commitment is *continuous* and the notion of "the step that
caused it" is even less well posed than in a discrete agent trace.

### 5.2 Saliency and attention — why not to use them

*Mechanism*: compute a gradient- or attention-based map over the input, present the high-mass regions
as the cause.

*Why not*: **Sanity Checks for Saliency Maps** ([arXiv:1810.03292](https://arxiv.org/abs/1810.03292),
NeurIPS'18) introduced two tests — the **model-parameter randomisation test** (progressively
randomise the trained weights; a valid attribution must change) and the **data randomisation test**
(retrain on permuted labels; a valid attribution must change). **Several widely used saliency methods
are invariant to both** — they are functions of the input's edge structure, not of the model or the
data **[A]**. They produce maps that look plausible for a model that has learned nothing. Follow-up
work finds that the *metrics* used to rank saliency methods are themselves statistically unreliable,
so comparative rankings between methods are untrustworthy **[S]**.

The paper's own conclusion is the operative sentence: methods failing these tests are inadequate for
*"explaining the relationship between inputs and outputs that the model learned, and debugging the
model"* **[A]** — which is precisely and only what stage 5 wants them for.

Attention weights inherit the problem without the benefit of having been tested this carefully.
**Neither belongs in an attribution claim we would defend to a client**, and the alternative in §5.3
is strictly better, cheap, and interventional.

### 5.3 Input ablation — the cheapest interventional method

*Mechanism*: remove, blank, shuffle or replace one input channel; re-run; measure the change in
success rate. This is an intervention — you set the value rather than observing it — so the
resulting claim is causal with respect to that channel, with no identification assumptions beyond
"nothing else changed".

*Worked example* **[F]**, from LIBERO-Plus:

- **Blank instruction.** Replace the language input with an empty value. OpenVLA-OFT on the object
  suite: *"remained largely unchanged"*, with significant degradation only on the long suite. The
  authors' conclusion is unusually blunt for a paper: the model *"degenerates into a form that
  disregards language, behaving more like a Vision-Action (VA) model"*.
- **Goal replacement.** Keep the scene, change the instruction to name a *different* object present
  in it. Success rates *"dropping nearly to zero"* — and the rollouts show the model executing the
  **original** target's trajectory: instructed to pick up butter, it picks up alphabet soup;
  instructed to pick up tomato sauce, it executes the butter action. The authors' reading: VLAs here
  behave as *"visual pattern matchers mapping scene configurations to predetermined action
  sequences"*.

The pairing is what makes the argument. Blanking language alone would have supported "the model is
robust to language perturbation" (Finding 3's −25.3, the second *smallest* drop). The replacement
probe shows that the robustness is insensitivity, not comprehension. **One ablation gives an
ambiguous number; an ablation plus a directed substitution identifies which way it points.**

*Corroboration from mechanistic work* **[A]**
([alphaXiv:2603.19233](https://www.alphaxiv.org/abs/2603.19233), a summary source, not the primary
paper — treat the figures as indicative): injecting visual-backbone activations alone, with no
language prompt, yields **73–77% task success**; given task A's instruction and task B's visual
activations in a shared scene, the robot follows the **injected visual signal 93.3%** of the time;
language sensitivity tracks task *ambiguity* rather than architecture (60–100% success with null or
wrong prompts on the LIBERO object suite; X-VLA on LIBERO goal collapsing **94% → 10%** under wrong
prompts); causal ablation shows **28–92% zero-effect rates** across architectures, and **~70% of
critical features encode object identity** rather than motion commands.

*Where input ablation breaks.* Three ways, all of which matter to us:

1. **The ablated input is itself out of distribution.** A blank instruction is not a neutral control;
   it is a novel input. A drop could be caused by the OOD-ness rather than by the loss of
   information. The directed substitution (goal replacement) avoids this — the replacement
   instruction is perfectly in-distribution — which is a second reason to prefer it.
2. **Channel-level necessity is not episode-level causation.** "Language contributes little on
   average" does not license "this episode failed because the language was misread". Ablation
   attributes to a *channel over a population*; a manifest row attributes to an *instance*. Do not
   silently convert one into the other.
3. **Zero effect is not absence of the pathway.** A 28–92% zero-effect rate under ablation is
   consistent with redundant encoding, where removing one route leaves an equivalent one intact.
   Ablation establishes sufficiency-of-removal, not the absence of a mechanism.

**Two further results, and together they are stronger than anything above.**

**LIBERO-PRO** ([arXiv:2510.03827](https://arxiv.org/abs/2510.03827)) perturbs four dimensions —
manipulated objects, initial states, task instructions, environments. *"Although existing models
achieve over 90% accuracy under the standard LIBERO evaluation, their performance collapses to
**0.0%** under our generalized setting."* Zero. The authors attribute it to *"rote memorization of
action sequences and environment layouts"*, and report that models *"persist in executing grasping
actions when the target object is replaced with irrelevant items, and their outputs remain unchanged
even when given corrupted instructions or even messy tokens"* **[S]**. That is LIBERO-Plus's goal-
replacement result reproduced independently, by a different group, at a more extreme magnitude.

**MINERVA** ([arXiv:2609.03715](https://arxiv.org/abs/2609.03715)) is the one that settles the
question, and it does so without perturbing anything. It asks how small a policy can be and still
solve LIBERO. *Mechanism* **[A]**: a from-scratch CNN encoder, **no language encoder at all** —
instead a **learned embedding table of 40 entries**, one per task — and a flow-matching action-chunk
head whose token mixer is an MLP. *Numbers*: **0.54M parameters → 95.05% average** (Spatial 94.4,
Object 99.6, Goal 96.4, Long 89.8); 0.99M → 96.75%. That is 2.4 points below the reported LeRobot
π₀.₅ result with **~7,700× fewer parameters**.

The authors' own probe is the decisive part: **permuting the task-ID mapping collapses success from
96.75% to near chance**, establishing that the embedding drives task selection while visual context
disambiguates. Their conclusion is that standard LIBERO is *"satisfiable by ~0.5M parameters of
task-indexed visuomotor memorization"*, and that performance above ~97% is *"unlikely to measure the
capabilities that motivate large models"* **[A]**.

> **Read the three together.** LIBERO-Plus: blanking the instruction barely hurts. LIBERO-PRO:
> substituting the target drops success to 0.0% while the model keeps executing the original action.
> MINERVA: a policy with **no language pathway whatsoever** — a 40-entry lookup table indexed by task
> — scores 95%. The third is not more evidence that models ignore language; it is evidence that
> **on standard LIBERO there is nothing for a language pathway to do.** A task ID suffices, because
> the benchmark never asks the same scene to support two different goals.
>
> This is a fact about the *benchmark*, not about the *models*, and that distinction is the whole
> point. It means a language-grounding failure family is not merely under-populated on nominal
> LIBERO — it is **not measurable there in principle**, because the benchmark contains no instance
> where reading the instruction is necessary. Populating that family requires an environment that
> asks one scene to support two goals, which is exactly what LIBERO-PRO's object-swap and
> instruction dimensions construct. **No amount of rollout volume on nominal LIBERO substitutes for
> it.**
>
> One caution against over-reading MINERVA in the other direction: under LIBERO-Plus perturbations
> its accuracy falls to **46–56%**, and the authors report *"photometric robustness remains near
> zero"* across all scales tested. Small-and-memorising solves the nominal benchmark; it does not
> solve the perturbed one. The capacity floor is a statement about what LIBERO measures, not a
> recommendation.

> **Three probes, not one, and the released corpus supplies none of them.** LIBERO-Plus's *Language
> Instructions* dimension — 1,537 of the released 10,030 instances — is **LLM-based instruction
> rewriting for linguistic diversity**: paraphrase. That is a third, weaker probe, testing robustness
> to *surface form*. The blank-instruction and goal-replacement experiments quoted above are a
> **separate analysis in the paper's §4, not part of the seven dimensions and not in the corpus**.
> Anyone planning a grounding experiment off the released instances would get paraphrase robustness
> and mistake it for grounding. The three probes answer different questions:
>
> | Probe | Manipulation | Distinguishes |
> |---|---|---|
> | **Paraphrase** (shipped, 1,537) | reword the same goal | robustness to surface form |
> | **Blank** (not shipped) | empty instruction | whether language is consumed at all — but the input is itself OOD |
> | **Directed substitution** (not shipped) | name a *different object present in the scene* | **insensitivity vs comprehension** — the only one that does |
>
> **The instrument matters as much as the probe.** Success rate alone cannot separate *comprehension*
> from *partial grounding*: a policy that reads the new instruction and fails, and a policy that
> ignores it and executes the original target, both show success → 0. The distinguishing measurement
> is **which object the end-effector actually approaches** — available to us from object poses and
> eef-to-object distance, i.e. from state we already derive. Report target identity, not just the
> outcome bit.

> ⚠ **Consequence for our family set.** If contemporary VLAs largely do not consume language, then a
> **language-grounding failure family is close to unpopulated on nominal LIBERO** — not because
> grounding is solved but because grounding is barely attempted. A classifier with such a family
> will therefore either abstain on it or capture something else under its name, and κ will not
> detect the latter (§3.5). The directed-substitution probe is the instrument that tells the two
> apart, and it is cheap: it is one extra rollout per instance with one word changed.

### 5.4 Step-level counterfactual replay

The most developed interventional method for trace-shaped failures, and worth describing in full
because its structure transfers to rollouts.

**Causal Agent Replay** ([arXiv:2606.08275](https://arxiv.org/abs/2606.08275)) **[F]**. *Mechanism*:
model the trajectory as a structural causal model
`τ = [s₀, (a₁,o₁), …, (aₙ,oₙ), y]` — decision state, stochastic action from policy π, observation,
outcome `y ∈ [0,1]`. Attribution is then a `do`-operation at step *k* followed by **re-execution of
the remainder under the unchanged policy**, measuring the shift in the outcome distribution over *K*
rollouts. Five intervention types: `do_resample` (re-draw the action from the same policy),
`do_action` (force a specific action), `do_observation` (replace the tool result), `do_context`
(edit history), `do_policy` (swap the model).

Four design details that are the transferable content:

- **`do_resample` is the null intervention and the foundation.** It changes nothing except re-drawing
  step *k* from the unchanged policy, which measures *"the intrinsic causal sensitivity of the
  outcome to that step"*. It is the correct baseline because it holds the policy fixed — any other
  intervention confounds "this step mattered" with "the thing I substituted was better".
- **The point-of-commitment rule.** Resampling step *k* also re-rolls every downstream stochastic
  decision, which confounds the estimate. The resolution: identify *"the latest step whose effect's
  confidence interval still excludes zero — the last point at which re-deciding still rescues the
  run; beyond it, the outcome is committed."* This is §2.5's point-of-no-return arriving as an
  estimator rather than a predictor, and it is a better-posed target than "the step that caused it".
- **Shapley over steps**, by Monte-Carlo permutation sampling with antithetic reverse-pairing,
  because single-step methods *structurally cannot* split credit across interacting steps.
  Deliberately **no caching of coalition values across permutations** — caching would suppress
  per-step marginal variance and produce *"false confidence"*. Budget-bounded with circuit breakers.
- **Validation against synthetic SCMs with analytic ground truth.** On a two-step interaction the
  estimator recovered φ₀=0.44, φ₁=0.45, φ₂≈0, efficiency sum **0.909** against an analytic **0.91**.

*Stated limitations* **[F]**, all of which apply to us verbatim: contrastive effects measure **total**
effects through stochastic paths, and isolating direct effects needs **common random numbers**, which
is unimplemented; judge-based outcome functions inject noise, so rule-based scoring is preferred;
Shapley is worst-case exponential; and — the one to internalise — *"even at temperature 0, hosted
inference varies because of floating-point non-associativity"*, so faithful replay requires recording
every nondeterministic input.

> **We are better placed than this method is.** Common random numbers, which CAR lists as unimplemented,
> is *free* for us: it is the same seed. Rule-based outcome scoring, which CAR prefers, is what a BDDL
> predicate already is. Deterministic replay, which CAR cannot guarantee, is what a pinned simulator
> gives (subject to F2 — the pin *is* the guarantee). The methods we would be adopting were developed
> under constraints we do not have, which is an argument for taking their **structure** and not their
> **workarounds**.

### 5.5 Interventional attribution in simulation — revert-one-knob

The strongest attribution available to this project, and it is strong for a boring reason: **we
assign the treatment** (§0).

*Mechanism*: fix the instance and the seed. Run the perturbed configuration; observe failure. Revert
exactly one factor to its nominal value; re-run at the same seed; observe the outcome. The
**paired** difference over instances is an estimate of that factor's contribution, and because the
assignment was ours, there is no confounding to adjust for. The identification assumption is the
entire content of the claim and it is checkable rather than assumed: *the rest of the simulator was
held fixed*.

This is why LIBERO-Plus's construction matters to us beyond its results. Its seven dimensions are
applied as **single-dimension perturbations to an existing task** **[F]** — which is the revert-one-knob
structure already in place. The reversion is the inverse of an operator the generator already applies.

*What to report*: the paired difference with a paired interval, not two independent success rates
with overlapping CIs. Pairing at a fixed seed removes instance variance, which is the dominant
variance component, and it is the difference between needing tens and needing hundreds of episodes
per cell.

*Where it breaks*, and these are not small:

1. **Interaction, and it is measured rather than hypothetical.** Reverting one knob at a time
   estimates main effects. If a failure requires viewpoint *and* initial-state jointly, every single
   reversion may show a small effect and the conclusion "no single factor is responsible" is correct
   but useless. **LIBERO-Plus §5 tested exactly this** over six dimensions and 2,000 independent
   trials on OpenVLA-OFT, and found a **consistent *negative* compositionality gap** — combined
   perturbations are worse than the independent effects predict, because *"co-occurring shifts act
   as coupled noise sources"*. Pairwise success rates: **Camera+Robot 19.05%, Robot+Noise 22.15%,
   Layout+Camera 35.95%**, with chi-square testing confirming the interactions are significant
   **[A]**. So on the benchmark closest to ours, main-effects-only attribution is not a theoretical
   weakness but a demonstrated one. Detecting interaction needs a factorial or a Shapley-style
   coalition estimator (§5.4) and costs combinatorially more — or `ddmin` (§7.1), which does not.
2. **Multiple sufficient causes.** If either of two factors alone would have caused the failure,
   reverting either one individually shows no effect, and revert-one-knob reports **no cause at all**.
   This is the standard counterexample to but-for causation, it is not exotic, and it is invisible in
   the output — the row simply looks uninformative rather than wrong.
3. **The factor is not the mechanism.** "Camera viewpoint caused it" names the knob we turned, not
   what broke. The step from knob to mechanism is not interventional and is where the
   correlational methods of §5.1 quietly re-enter.

### 5.6 Attribution: synthesis

1. **The published numbers are ~11–14% and the gap is partly the ground truth.** Who&When labels one
   deterministic failure-inducing step; real failures are frequently overdetermined or committed
   gradually. Be correspondingly careful about promising step-level attribution on rollouts.
2. **Correlational attribution is post-hoc narration.** Nothing in the procedure penalises a wrong
   but consistent story, and consistency is what it optimises.
3. **Do not use saliency or attention.** Several standard methods are invariant to randomising the
   model's weights *and* to retraining on permuted labels **[A]**, and the metrics that rank saliency
   methods are themselves unreliable **[S]**.
4. **Input ablation is the cheapest real intervention** — but pair it with a **directed substitution**,
   because a blank input is itself OOD and an unchanged success rate is ambiguous between robustness
   and insensitivity. LIBERO-Plus's blank-instruction / goal-replacement pair is the model to copy
   **[F]**.
5. **Contemporary VLAs largely do not consume language.** Blank instruction: little change on the
   object suite; goal replacement: success *"nearly to zero"* with the original target's trajectory
   still executed **[F]**. A language-grounding family is therefore near-unpopulated on nominal
   LIBERO, and κ will not tell us if something else is being labelled with its name.
6. **Take CAR's structure, not its workarounds.** Null intervention by resampling under the unchanged
   policy; the point-of-commitment rule as a better-posed target than "the causal step"; Shapley for
   interacting steps with no coalition caching; validation against an analytic SCM. Common random
   numbers, rule-based outcomes and deterministic replay — CAR's three stated gaps — are free for us.
7. **Revert-one-knob at a fixed seed is our strongest instrument**, and its two blind spots are
   **interaction** and **multiple sufficient causes**. Both present as *"no factor responsible"*,
   which is indistinguishable in the output from a genuinely uninformative case. Any manifest row
   reporting no attributable factor should say which of the three it could be.
8. **Naming the knob is not naming the mechanism**, and the step between them is where correlational
   reasoning re-enters an otherwise interventional pipeline. Mark it in the output.

---

## 6. CROSS-CUTTING — what survives the move to a real robot

§2.10 tabulated this for localisation. The pattern generalises, and it is not the pattern one would
guess: **the stages do not degrade uniformly, and the stage that degrades worst is the one we are
relying on most.**

| Stage | In simulation | On a real robot | Transfer |
|---|---|---|---|
| **1 Detection** | BDDL predicate over privileged state — exact, deterministic, free | no predicate exists; a human or a VLM judge at ≤0.77 macro bACC **[F]** | **poor.** The instrument changes entirely |
| **2 Localisation** | phase segmentation over `_gt_` state; onset labels readable from state | policy-internal signals transfer cleanly; **onset ground truth often cannot be constructed at all** (§2.10) | **mixed.** Methods transfer, evaluation does not |
| **3 Classification** | rules over privileged state, or a fine-tuned VLM | VLM only. RoboFAC-7B: 79.10 sim-average → **68.94 real**, on its own benchmark, having trained on real data **[F]** | **degrades ~14 points, optimistically measured** |
| **4 Clustering** | any route | the explanation-clustering route (§4.1e) needs **no state at all** | **good — the best-performing design is also the most portable** |
| **5 Attribution** | revert-one-knob at a fixed seed; common random numbers free | **cannot re-run the same instance**; no CRN; no reversion | **poor to impossible.** The mechanism does not exist |

Three observations follow.

**The privileged-state stages are the ones that vanish.** Detection and attribution — stages 1 and 5,
the two ends of the pipeline — are the ones most completely dependent on being able to read and set
simulator state. A client deploying on real hardware inherits stages 2, 3 and 4 and must reconstruct
1 and 5 by other means. It is worth saying this out loud in any readout, because the natural client
reading of "we found the causes of your failures" is that the method comes with them.

**Clustering is the exception, and for an instructive reason.** The design that performs best
(cluster VLM-written explanations) is the one that touches no environment state, and is therefore the
one that ports unchanged. That is not a coincidence: the reason it works — that a natural-language
explanation abstracts away scene and task specifics — is the same reason it is portable. **Where a
method's mechanism is abstraction rather than privilege, it transfers.**

**The transfer coefficient is itself unmeasured for what we plan to report.** Published Spearman
correlations for sim-to-real transfer of failure-conditional quantities run **0.4–0.7**, with
*severity ordering transferring worst* **[? — carried forward from project notes; the primary source
was not re-verified in this pass and should be before it is cited to a client]**. A rank correlation
of 0.4 on the quantity a manifest is ordered by is not a caveat, it is a different document. Note
that this is a further, independent reason not to ship a cardinal severity (DG-5b) — the ordering is
the part that transfers worst, and a cardinal number implies an ordering.

---

## 7. ADJACENT FIELDS WORTH STEALING FROM

Three fields have been doing versions of this for longer, against harder ground truth, and two of
them have an algorithm we should take directly.

### 7.1 Software fault localisation

**Spectrum-based fault localisation (SBFL).** *Mechanism* **[S]**: run a test suite; record, per
test, which program statements it executed (the *spectrum*); for each statement compute a
**suspiciousness score** from the four counts (executed-and-failed, executed-and-passed,
not-executed-and-failed, not-executed-and-passed). Tarantula and Ochiai are two such formulas;
Ochiai is `a_ef / sqrt((a_ef + a_nf) · (a_ef + a_ep))`. Rank statements by suspiciousness; the
developer reads down the list.

*The mapping to us is exact and the vocabulary is the only thing that changes*: statements → phases
(or knob settings); tests → episodes; pass/fail → the BDDL predicate. A phase executed by many
failing episodes and few passing ones is suspicious in precisely Ochiai's sense. **This gives a
principled, twenty-year-old, zero-parameter ranking for "which phase is implicated", computed from
data we already have** — and it is a far better default than any threshold we would invent.

*Published failure modes, which transfer too* **[S]**:
- SBFL **"loses discrimination when passing and failing tests execute the same statements"**. Our
  analogue: if every episode traverses the same phase sequence, phase-level suspiciousness is
  uninformative — and on a short manipulation task that is the common case, not the edge case.
- **Coincidental correctness** — tests that execute the faulty statement but pass — degrades Ochiai.
  Our analogue is the lucky success, which §0 already documents concretely (MANGO's cabinet door
  pushing the dropped object in). It is the same statistical nuisance under a different name.
- Tarantula and Ochiai **"poorly performed as the number of faults increased"**. Our analogue is the
  multi-cause episode, i.e. §5.5's interaction and multiple-sufficient-cause blind spots, showing up
  as degradation rather than as an error.

**Delta debugging (`ddmin`).** *Mechanism* **[S]**: given a failing input, systematically partition
it and re-test subsets to find a **1-minimal failure-inducing subset** — one where removing any
single element makes the failure go away. `DDMIN-LOC` extends it by feeding the passing and failing
inputs generated *during* the minimisation into an SBFL formula.

> **This is the direct answer to §5.5's interaction blind spot, and it is the single most valuable
> import in this section.** Revert-one-knob estimates main effects and reports "no cause" when a
> failure needs two factors jointly. `ddmin` over the *perturbation set* finds the minimal subset of
> knobs that still produces the failure — which is exactly the right object — in **O(n²) re-runs
> worst case, typically far fewer**, rather than the 2ⁿ of a full factorial. Every re-run is a
> same-seed rollout, which we can already do. A manifest row reading *"fails under {viewpoint yaw
> +15°, initial state B} together, and under neither alone"* is strictly more actionable than
> anything main effects can produce, and this is how to compute it.
>
> One caveat carried over from the source: `ddmin` needs inputs that **decompose**. The one Siemens
> program unsuitable for delta debugging was TCAS, whose input is a fixed-size integer vector that
> *"cannot be easily decomposed"* **[S]**. Our perturbation set is a set of independent knobs, which
> is the decomposable case — but a single continuous knob's *magnitude* is not, and mixing the two
> needs care.

### 7.2 Driving scenario analysis

*Mechanism* **[S]**: mine fleet or crash logs for scenarios; cluster them (k-medoids on junction
crash data, then association-rule mining within each cluster to specify the scenario); score them by
**criticality metrics** (time-to-collision and relatives); retain the critical tail as a test suite.
One reported result: new selection methods increase the proportion of critical scenarios by **17.4%
and 13.6%** over traditional methods **[S]**.

Two things this field has that robotics does not, and both are about the surrounding infrastructure
rather than the algorithms:

1. **An external, authoritative taxonomy.** The US DoT Volpe Center pre-crash typology exists, is
   agreed, and predates any of the methods — which is why the unsupervised-taxonomy paper could use
   alignment with it as evidence its driving clusters were meaningful **[F]**. Robot manipulation has
   no equivalent, and §3.1 is a picture of what its absence looks like: four groups, four
   incompatible category sets, no arbiter. **Alignment-with-a-standard is a validation mechanism that
   is simply unavailable to us**, and it is the one that would most cheaply close §3.5.
2. **Externally-supplied prevalence.** Crash databases say how often each scenario occurs in the
   world. No simulation can supply that, and the field does not pretend otherwise. This is exactly
   DG-5b's position — prevalence is client-supplied, not measured by us — arriving from an
   independent direction, which is reassuring about the decision.

### 7.3 LLM agent trajectory analysis

The closest adjacent field methodologically, because a trace is trace-shaped whether the actions are
tool calls or end-effector deltas, and it is roughly two years ahead on stage 5. Its content is
distributed through §5 above (Who&When, Who&When Pro, TRAIL, CAR) and §1.8 item 5. The three
importable lessons, none of which are about robots:

- **Build the dumb baseline first.** A TF-IDF surface detector beat every LLM-judge configuration at
  detecting false success (AUROC 0.83/0.95 vs ≤0.65/0.54) **[F]**. Nothing in that result is specific
  to text.
- **Benchmark the attribution, not the narrative.** This field constructed traces with annotated
  responsible steps and discovered its methods were at 11–14%. Robotics has not done the equivalent,
  which is why robot-failure-explanation papers report fluency-adjacent metrics and this one reports
  accuracy.
- **Model the trace as an SCM and intervene** (§5.4) rather than asking a model to narrate it.

---

## 8. WHAT WE COULD ADOPT

Ranked by (value ÷ cost), with the concrete change named. Nothing here requires the GPU except where
stated.

**1. `ddmin` over the perturbation set, to find minimal failure-inducing knob subsets.** (§7.1)
Closes the interaction blind spot that revert-one-knob cannot see, in O(n²) same-seed re-runs instead
of 2ⁿ. It is the only method surveyed that turns *"no single factor is responsible"* from a dead end
into a manifest row. **Highest value in this document.** Needs the counterfactual-probe path, so it
sequences after a perturbable environment exists.

**2. Ochiai suspiciousness over phases.** (§7.1) A zero-parameter, twenty-year-old ranking for "which
phase is implicated", computable from the pass/fail and phase-traversal data our manifest already
carries. Replaces an invented threshold with a formula that has a literature and known failure modes.
Cheap: it is a function over existing columns. Report alongside it the fraction of episodes sharing a
phase sequence — that fraction is the statistic's own validity check, since SBFL loses discrimination
exactly when it approaches 1.

**3. The directed-substitution probe.** (§5.3) For every ablation we run, pair it with an
in-distribution substitution — not just "blank the instruction" but "name a different object that is
present". It costs one extra rollout per instance and it is the difference between measuring
*robustness* and measuring *insensitivity*. LIBERO-Plus's two experiments are the template.
**Sequence this early**: if it shows our policy does not consume language, that fact should be known
before a language-grounding family is put in front of a client.

**4. A null corpus at stage 4.** (§4.3) Run the clustering pipeline on a corpus with no mode
structure by construction and report the same statistics. One extra run; it is the only check that
distinguishes a mode from an artefact of the pipeline, and we already run the analogous control at
stage 3.

**5. Name-based re-assignment, replacing κ's load-bearing role.** (§3.5, §4.4) Give a held-out judge
only the cluster *names and descriptions* and have them assign held-out episodes; measure agreement
with the induced partition. Unlike κ on a hand-designed taxonomy this genuinely tests the partition,
because the partition was induced rather than given — and it catches the specific hazard that an LLM
names any set fluently. This is the closest thing this survey found to a fix for our open problem,
and it is worth being precise that it is a *partial* fix: it establishes that a cluster has a
communicable common property, not that the property is the right one.

**6. Report percentage agreement and Gwet's AC₁ next to κ.** (§3.5) Failure categories are always
heavily imbalanced, which is the kappa paradox's regime. Nearly free.

**7. Cluster the discrepancy, not the episode.** (§4.1) If `mining/cluster.py` clusters trajectory
features, the measured penalty is 32.41% vs 85.53% assignment F1. The portable form is to cluster a
structured *description* of what differs between the failure and its nominal counterpart — which, for
counterfactual pairs, we have by construction and without needing a VLM in the loop at all. **We are
better placed here than the published methods**: they reach for a VLM explanation because they have
no nominal counterpart; we generate ours.

**8. Take CAR's structure for probe design.** (§5.4) Specifically: `do_resample` as the null
intervention; the **point-of-commitment** rule (the latest step whose effect interval excludes zero)
as a better-posed target than "the causal step"; no coalition caching if we ever compute Shapley;
and validation against an analytic ground truth before trusting the estimator. CAR's three stated
gaps — common random numbers, rule-based outcomes, deterministic replay — are all free for us.

**9. Distil the LLM tier into a deterministic classifier.** (§3.4, TnT-LLM) If an LLM is used to
build the category set or label a sample, let it train a cheap deterministic classifier that does the
production labelling. This is a direct input to our open tier-3 decision and it reframes it:
the question is not local-vs-API but *which component ships* — an auditable artefact re-runnable
years later, or a model call.

**10. Measure the residual of a synthetic taxonomy against natural failures.** (§3.2) Take our
planted-fault family set, apply it to a corpus of *natural* LIBERO failures, and report the fraction
that fits no family. **No published work reports this number**, it is the direct measure of the
blind spot that procedural generation creates, and it is a genuine contribution rather than an
adoption. Cost: one annotation pass over a modest sample.

**11. Report precision@10 rather than cluster purity.** (§4.5) It matches how a manifest row is
actually read — top-down, by the examples shown — and it is the metric the one rigorous
planted-ground-truth evaluation in this space established.

**Explicitly not adopted**, with reasons: saliency and attention maps (§5.2 — they fail
model-randomisation sanity checks); silhouette for choosing k (§4.3); bootstrap stability as evidence
of structure (§4.3); BERTopic off the shelf for failure explanations (§4.5); and — carried from
§2 — TAS as a primary localiser (§2.1) and trajectory-distance-to-demo (§2.6, which remains a
**caveated secondary signal in `vla_harness/mining/phases.py` and is therefore still an open
inconsistency between this document and the code**; flagged, not resolved, in this pass).

---

## 9. WHAT COULD NOT BE VERIFIED IN THIS PASS

Listed so that nothing above is mistaken for having been read at full text.

| Claim | Mark | Why, and what would settle it |
|---|---|---|
| Sim-to-real Spearman 0.4–0.7 for failure-conditional quantities; severity ordering transfers worst | **[?]** | Carried forward from project notes; the primary source was **not located or re-verified in this pass**. It is load-bearing for §6 and for any client readout. **Do not cite until found.** |
| AHA / FailGen mechanism and the seven operators | **[A]** | The ICLR PDF exceeded the fetch size limit; the seven mode names and headline numbers come from the abstract and from search results. The per-operator injection details in §3.2 are a reconstruction from those names and should be checked against the paper before being relied on |
| Domino: 36% of 1,235 settings; precision@10 0.639; 105% over George | **[S]** | Search-snippet level. Numbers are consistent across snippets but were not read off the paper |
| Who&When: 184 tasks, composition, ~14% step accuracy | **[S]/[F]** | The benchmark description is snippet-level; the ~14% figure is **[F]** only as *reported by CAR*, not from the Who&When paper itself. arXiv ID given in §5.1 is unconfirmed |
| TnT-LLM two-phase mechanism | **[A]** | Abstract only — the ACM full text returned HTTP 403. The "pseudo-gradient over batches" description is from secondary coverage and should be confirmed |
| Mechanistic VLA figures (73–77%, 93.3%, 28–92%, ~70%) | **[A, secondary]** | From an alphaXiv summary page, **not the primary paper**. Indicative only. The LIBERO-Plus ablation results in §5.3, which make the same point, **are [F]** and should be preferred as the citation |
| Bootstrap stability cannot distinguish clusters from noise | **[S, secondary]** | A practitioner blog. The argument is sound and reproducible in a few lines, but it is not a peer-reviewed citation. Cite the demonstration, not the source |
| SBFL / `ddmin` details, Ochiai formula, Siemens/TCAS note | **[S]** | Search-snippet level across several sources, mutually consistent. The Ochiai formula as given is standard but was written from memory of the standard form, not copied from a source — **verify before implementing** |
| Driving scenario mining: +17.4% / +13.6% critical scenario proportion | **[S]** | Snippet-level, single source, method not inspected |
| Human–human 97.0% outcome / 91% phase agreement | **[S]** | Carried from §1; unchanged in this pass |
| LIBERO-Plus per-model per-factor table | **[F]** | ~~Text layer did not extract~~ — **resolved 2026-09-16** by re-extracting with layout preservation (`pdftotext -layout`). Table 1 is now quoted in full in §3.3, all ten models, all seven dimensions. The findings text, generation pipeline, 14,000 → 10,030 filtering and language experiments were already [F] |
| Whether LIBERO-Plus's generation code is **released** | **unverified** | §3.3 establishes from the paper that a generator exists and that single-dimension perturbations are its unit of operation. Whether it is distributed is a question about the repository, answerable in minutes, and **not answered here** |

**One methodological note on this pass.** Sections 3–9 cite 12 further URLs (122 distinct in the
document) from roughly 20 sources consulted, against §1–§2's 113, and lean more heavily on **[F]** readings of a few central papers (LIBERO-Plus, RoboFAC, CAR,
the unsupervised-taxonomy paper) than on breadth. The synthesis items are correspondingly more
exposed to any one of those four being wrong. The compensating fact is that the four were chosen
because they are the ones whose *mechanisms* are closest to what we are building, which is what this
document is for.

---
