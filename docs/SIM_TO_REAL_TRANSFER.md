# SIM-TO-REAL TRANSFER — does our loop work on a real robot?

*Written 2026-09-17. Answers the revised brief: take a generalist policy trained on **real** data,
evaluate it in **simulation**, mine its failures, and prescribe data. Does that improve the policy on a
**real** robot?*

> Marks: **[F]** full text read · **[A]** abstract/landing page · **[S]** search-snippet · **[?]**
> unverified. **Peer-reviewed results are named as such; 2026 arXiv preprints and company demos are
> marked separately.** This area is full of demo videos and they are not evidence.

---

## 0. The answer in one page

**Partly supported, and the supported half is the half we sell.**

| our output | transfers? | evidence |
|---|---|---|
| **Which perturbation hurts *most*** (the top of the ordering) | **YES — the best-evidenced claim here** | SIMPLER Table VI pairs sim and real |Δsuccess| per factor; camera pose and table texture hold the top two places in both **[F]** |
| **The full ordering, including the tail** | **NO — 4 of 5, one swap** | lighting and distractors exchange places; near-tied and far below the top two, so MMRV stays low (0.000/0.041) **[F]** |
| **Relative policy ranking** | **YES** | MMRV 0.000–0.111, Pearson r 0.855–0.976 **[F]**, peer-reviewed (CoRL 2024) |
| **Per-factor magnitude** | **NO — and the error is signed** | real÷sim: camera **0.61×**, background 2.15×, lighting 2.08×, table texture 3.44×, distractors **4.11×**. Sim over-estimates camera and under-estimates everything else **[F]** |
| **Absolute success rate** | **NO** | same tables: per-task sim and real rates diverge widely even where ranks hold |
| **Prevalence of a condition** | **NO — and not for the reason we claim.** See §5 | |
| **Failure *families*** | **UNKNOWN. Nobody has measured this.** | §1.3 |

**The single most important finding for this project** is SIMPLER's Table VI, because it is the only
published thing that pairs sim and real outcomes **for the same perturbation condition** rather than
for the same policy's overall rate. That is precisely our product's claim, and it holds — for
ordering.

**The gap in the literature that is precisely our gap:** nobody has closed the loop. No published work
mines failures in simulation, prescribes data from that mining, fine-tunes, and measures a real-robot
gain. §1.2.

---

## 1. PART A — sim-mined failures → real improvement

### 1.1 Does a real-trained policy fail the same way in sim as on a real robot?

**SIMPLER** ([arXiv:2405.05941](https://arxiv.org/abs/2405.05941), CoRL 2024, **peer-reviewed**) is the
anchor. It evaluates real-trained generalists (RT-1 at three training checkpoints, RT-1-X, RT-2-X,
Octo-Base, Octo-Small) in SAPIEN reconstructions of the Google Robot and WidowX+Bridge setups.

**Policy-ranking correlation [F]**, read from the paper:

| | MMRV ↓ | Pearson r ↑ |
|---|---|---|
| reported across setups | 0.000, 0.014, 0.031, 0.055, 0.111 | 0.890, 0.915, 0.969, 0.976, 0.855 |

**Table VI is the one that matters to us [F].** It measures |Δ success| under five distribution
shifts, in simulation *and* on a real tabletop, for RT-1 trained with and without image augmentation:

| shift | RT-1 w/o Aug: **sim** | **real** | RT-1 +Aug: **sim** | **real** |
|---|---|---|---|---|
| Background | 0.013 | 0.028 | 0.153 | 0.167 |
| Lighting | 0.040 | 0.083 | 0.033 | 0.042 |
| Distractors | 0.027 | 0.111 | 0.033 | 0.083 |
| Table texture | 0.113 | **0.389** | 0.220 | 0.167 |
| **Camera pose** | **0.753** | **0.458** | **0.613** | **0.375** |

Read three things off it:

1. **The ordering *mostly* transfers — 4 of 5 positions, with one swap.** Computed rather than eyeballed
   (`experiments/stats.py`):

   ```
   sim : camera pose > table texture > lighting > distractors > background
   real: camera pose > table texture > distractors > lighting > background
   ```

   > **CORRECTION.** I first wrote "the ordering transfers" flatly. **It does not.** `lighting` and
   > `distractors` swap. I had read the table rather than sorted it, which is the same error class as
   > everything else I got wrong this week.

   **What survives is still the useful part**, and the swap is benign for a specific, checkable reason:
   the two factors that exchange places are near-tied in both columns (sim 0.040 vs 0.027; real 0.083
   vs 0.111), and **both sit far below the top two**. The rank violation is small-margin — which is
   precisely what MMRV is built to discount, and why the paper reports **MMRV 0.000/0.041** alongside
   r 0.831/0.970 **[F]** rather than claiming exact rank agreement. Its own wording is the careful
   one: *"SIMPLER evaluations accurately track the policies' robustness to distribution shifts"*.

   **So the defensible claim is narrower than the one I made:** *the dominant factor and the top of
   the ordering transfer; the tail does not reliably order.* For a manifest that ranks conditions, that
   means the top row is trustworthy and the bottom rows should not be presented as ranked.
2. **Camera pose dominates, in reality as well as in sim.** Independent of LIBERO-Plus's identical
   finding, on a different simulator, different robot, different policies. Two benchmarks, one
   conclusion — and it is the factor our own (currently defective) perturbation code targets.
3. **Magnitudes are unreliable.** Table texture: sim 0.113 vs real 0.389 for the un-augmented policy —
   sim underestimated the real effect by **3.4×** — and `distractors` is worse at **4.1×**
   (0.027 vs 0.111). So "which factor" transfers; "how much" does not.

   **The direction of the error is systematic, which I missed on the first pass and only saw once the
   arithmetic was scripted** (`experiments/stats.py`). Real ÷ sim, per factor: camera pose **0.61×**,
   background 2.15×, lighting 2.08×, table texture 3.44×, distractors 4.11×. **Simulation
   over-estimates camera sensitivity and under-estimates every other factor.** So the one factor sim
   ranks first is also the one it exaggerates — which flatters exactly the conclusion a
   perturbation-sweep project most wants to draw, and is worth knowing before we draw it.

**Caveats that must travel with these numbers, and they are not small:**

- **N is tiny.** The Google Robot correlation is over **6 policies**, WidowX over **3** **[F]**. A
  Pearson r computed over 3 points carries almost no information, and the paper's own §III discussion
  concedes that r *"does not reflect the range of values it is computed over"*.
- **Rigid objects only.** *"we focus our evaluations on rigid-object [manipulation]"* **[F]**.
- **Known rendering gaps** — *"does not accurately capture object shadows"* **[F]**.
- Environment parameters have a **≤15% impact** on success rates **[F]** — i.e. the reconstruction
  itself is a meaningful error source.

**A second metric worth adopting.** *A Practical Recipe Towards Improving Sim-and-Real Correlation for
VLA Evaluation* ([arXiv:2606.10366](https://arxiv.org/abs/2606.10366), **preprint**) distinguishes
Spearman (ordering), Pearson (score-level), MMRV (severity of high-margin rank errors) and
**"sensitivity MAE — the absolute discrepancy between simulated and real-world vulnerability
profiles"** **[S]**. That last one is *exactly* the quantity our manifest implicitly claims, and we
should report it by name if we ever get paired data. **[?]** on its numbers: the summariser
mis-expanded MMRV, so treat that source's specifics as unverified until read directly.

**One older datapoint on feature-ranking transfer:** Spearman **ρ = 0.77** between feature rankings
found by real-world versus simulated testing **[S]**, from the mid-level-visual-representations line
([arXiv:2011.06698](https://arxiv.org/abs/2011.06698)). Moderate, and in the same direction.

### 1.2 Has anyone closed the loop? **No.**

**This is the clearest "nobody has shown this" in the document, and it is our gap.**

Searching for work that (i) mines failures in simulation, (ii) prescribes data from that mining, (iii)
fine-tunes, and (iv) measures a **real-robot** gain returns nothing that does all four. The adjacent
pieces exist and stop short:

- **Failure-guided data collection works — in simulation.** The unsupervised-taxonomy paper
  ([arXiv:2506.06570](https://arxiv.org/abs/2506.06570)) fine-tunes on 40K samples gathered inside
  discovered failure zones: **46% → 18% failure rate, versus 46% → 34% for uniformly-collected data at
  equal budget** **[F]**. That is the loop, closed, with a control arm — but in **simulated indoor
  navigation with a CNN policy**, not manipulation, and never measured on a real robot.
- **Real-to-sim-to-real robustification exists** (RialTo; *Reconciling Reality through Simulation*,
  [arXiv:2403.03949](https://arxiv.org/abs/2403.03949)) — build a digital twin from a little real
  data, RL in sim, deploy. But the sim is used to *train*, not to *diagnose*, and the weaknesses it
  addresses are chosen by the experimenter, not mined.
- **AutoEval** ([arXiv:2503.24278](https://arxiv.org/abs/2503.24278)) goes the other way: autonomous
  evaluation of generalist policies **in the real world**, which is a substitute for our sim loop
  rather than a validation of it.

⇒ **Our loop is unvalidated end-to-end by anyone, including us.** That is a genuine contribution
opportunity and a genuine risk, and the honest framing is that we would be the first to test it, not
the first to exploit a known result.

### 1.3 Which kinds of sim-found weakness plausibly transfer

Evidence-backed where possible; the gaps are marked.

| weakness | transfers? | basis |
|---|---|---|
| **Camera / viewpoint sensitivity** | **yes** | SIMPLER Table VI, both policies, sim and real **[F]**; corroborated by LIBERO-Plus on a different stack |
| **Lighting, background** | **yes, small in both** | Table VI **[F]** |
| **Distractors** | **yes, direction; sim under-estimates** | 0.027 vs 0.111 **[F]** |
| **Table texture / appearance** | **direction yes, magnitude badly wrong** | 3.4× underestimate **[F]** |
| **Distractors — magnitude** | **worst of the five** | 4.1× underestimate **[F]** |
| **Spatial-relation / object-pose errors** | **[?] untested** | no paired sim-real study found |
| **Contact physics, grasp slip** | **likely artefact** | SIMPLER's rigid-object restriction; contact modelling is the canonical sim-to-real failure **[S]** |
| **Rendering / asset-quality effects** | **artefact** | shadow limitation stated in-paper **[F]** |
| **Failure *taxonomies*** | **[?] nobody has measured it** | §0 |

The pattern is consistent and mechanistically sensible: **perturbations of the policy's *input
distribution* transfer; perturbations mediated by *physics* do not.** Camera pose, lighting and
background change pixels, and a real-trained policy's pixel sensitivity is a property of the policy.
Contact and friction change dynamics, and those are exactly what a simulator approximates worst.

> **Direct consequence for our family set.** Our current families split roughly into pixel-mediated
> (`visual_grounding`, `spatial_reasoning` as actually implemented) and physics-mediated
> (`manipulation`, `recovery`, and the whole grasp-detection layer). **The first group has a transfer
> argument; the second does not.** That is a sharper prioritisation criterion than anything in the
> taxonomy guidelines, and it did not come from us.

### 1.4 Can sim findings direct a small real-data budget?

**Not demonstrated. Supported by analogy only.**

The strongest analogy is §1.2's 46→18 vs 46→34 result, which shows guided beats uniform *within one
domain*. The cross-domain version — mine in sim, collect in real — has not been run. Our manifest's
central claim therefore rests on: sensitivity *ordering* transfers (§1.1, evidenced) + guided
collection beats uniform (§1.2, evidenced in-domain) + the composition of the two (**unevidenced**).

State it that way rather than as a single claim.

---

## 2. PART B — policies trained on simulated data, transferred to real

Secondary to the brief, and the evidence is more promising here than in Part A.

**DreamGen / GR00T-Dreams** ([arXiv:2505.12705](https://arxiv.org/abs/2505.12705), **preprint**;
NVIDIA) — synthetic "neural trajectories" from a video world model **[S]**: real-robot success rose
**37% → 46.4%** (4 GR1 humanoid tasks), **23% → 37%** (3 Franka tasks), **21% → 45.5%** (2 SO-100
tasks). Reported to need only **10–13 real trajectories per task** for the novel-behaviour setting,
and to add up to **8.8%** even where real demos already exist. Company-affiliated preprint; treat the
headline as promotional until the protocol is read.

**Sim-and-real co-training** ([arXiv:2503.24361](https://arxiv.org/abs/2503.24361)) **[S]**: the
mixing ratio α can go as high as **0.99 simulation**, but **beyond ~0.995 performance degrades** — a
small amount of real data is load-bearing. Reported **average +38%** real success over real-only
training, with as few as **5–15 real demonstrations** producing 2–3.5× gains on some tasks, and
co-trained policies still ahead at 400 real demos. **Visual alignment is called out as critical, with
misaligned camera poses causing significant drops** — the same variable that dominates §1.1.

**Sim-pretrain then fine-tune:** **500–2,000 real demonstrations recovers 80–95% of simulation
performance** for moderate visual complexity **[S]**, and a sim-pretrained policy needs **3–5× less
real data** than training from scratch **[S]**. Both snippet-level; useful as an order of magnitude,
not as a citation.

**RL fine-tuning in sim → real.** SimpleVLA-RL ([arXiv:2509.09674](https://arxiv.org/abs/2509.09674),
ICLR 2026) claims policies trained **entirely in simulation** show *"strong sim-to-real transfer …
substantial performance gains in real-world robotic tasks on the Agilex Piper arm without any real
robot data"* **[S]**. If it holds at full text, it is the strongest single result against the
"toyish" worry. **I have not read it [A/S], and the claim is strong enough that it should be verified
before being repeated.** The RLinf-Co line ([arXiv:2602.12628](https://arxiv.org/abs/2602.12628))
addresses the same question via sim-real co-training.

**Nothing found showing sim-RL gains that fail to transfer** — but absence here is weak evidence,
because negative transfer results are under-published.

---

## 3. PART C — how close can simulation get?

### 3.1 The fidelity ladder, and what each step buys

| approach | what it does | measured effect on sim-real correlation |
|---|---|---|
| **Visual matching** (SIMPLER) | overlay real images, match textures | **the ablation is the evidence** — visual matching plus variant aggregation achieves the MMRV/r in §1.1 **[F]** |
| **Gaussian-splatting digital twins** | reconstruct the real scene photorealistically | *"simulated rollouts correlate strongly with real-world execution"* on deformables ([arXiv:2511.04665](https://arxiv.org/abs/2511.04665)) **[A]** — first real result on **soft bodies**, which SIMPLER excludes |
| **RoboGSim** ([arXiv:2411.11839](https://arxiv.org/abs/2411.11839)) | real2sim2real splatting simulator | *"high consistency in texture and physics"* **[A]** — qualitative, no correlation metric found |
| **Digital cousins** ([arXiv:2604.15805](https://arxiv.org/abs/2604.15805)) | generative high-fidelity scenes | **[A]** |
| **Generative world models** (Cosmos, DreamGen) | video model as simulator | trains well (§2); **no evaluation-correlation number found [?]** |

**The honest summary of Part C.3:** higher visual fidelity plausibly raises correlation, and SIMPLER's
design is built on that premise — but **I found no study that varies fidelity and reports the
resulting change in sim-real correlation as a controlled measurement.** The closest is SIMPLER's own
≤15% environment-parameter sensitivity **[F]**. That is a real hole in the literature.

### 3.2 Practicality for us

| option | 8 GB? | MuJoCo/robosuite (our mining carries over)? | verdict |
|---|---|---|---|
| **LIBERO-Plus** | ✅ | ✅ **identical env class** | cheapest fidelity gain available: published perturbations, zero adapter work |
| **RoboCasa365** | ⚠ ~30 GB assets | ✅ robosuite | breadth + second embodiment; real adapter cost |
| **SIMPLER** | **[?]** | ✗ SAPIEN | the only stack with *published real correlation*; new simulator |
| **Gaussian-splat twins** | ✗ likely | ✗ | needs real scans of a real cell — a different project |
| **Isaac / Omniverse RT** | ✗ | ✗ | out of scope on this hardware |

**The cheapest genuine de-toy-ing available to us is not better rendering. It is
`scene_descriptor`-level realism: perturbation ranges chosen to match a real deployment**, which costs
nothing and is the variable SIMPLER shows matters most (camera pose).

---

## 4. What this means for the project

**Is the loop supported?** **Partly.** The diagnostic half — *which conditions break this policy, and
in what order* — is the best-evidenced claim in this document, on peer-reviewed paired sim/real data.
The prescriptive half — *therefore collect this data and the real robot improves* — **has never been
demonstrated by anyone**, in sim or real.

**What to claim, and what to stop claiming:**

| claim | status |
|---|---|
| "These are the conditions your policy is most sensitive to, in order" | **defensible now** |
| "Policy A is more robust than policy B" | **defensible** (MMRV/Pearson) |
| "This condition costs you X% success" | **not defensible** — magnitudes off by up to **4.1×**, and the error is signed differently for camera (0.61×) than for everything else |
| "This condition occurs Y% of the time" | **not defensible**, and we already refuse it |
| "Collect this data and your real robot improves" | **unvalidated** — say so |

**Direction change?** No — but a **re-emphasis**. The product is stronger sold as *ranked sensitivity
diagnosis* than as *prescriptive data shopping list*, because the first is evidenced and the second is
not. That is a framing change, not a rebuild.

**Cheapest real-world check for a small team with one arm**, and it is genuinely cheap:

1. Take a real-trained generalist that runs on the arm.
2. Pick **three** perturbation factors our sim says are ordered — e.g. camera pose ≫ lighting ≈
   background.
3. Run ~20 real trials per factor per level. **~120–180 trials, one arm, a few days.**
4. Compare the **ordering**, not the rates. Report Spearman ρ and sensitivity MAE.

That is a direct replication of SIMPLER Table VI on our stack. If the ordering holds, the diagnostic
claim is validated on our own hardware; if it inverts, the loop is dead and we learned it for the cost
of a week rather than a product launch. **No published work does this for a LIBERO-trained
policy** — so it is also publishable.

---

## 5. The manifest's unsourced claim — resolved

`vla_harness/manifest.py:58-59` says prevalence *"is the half of severity the sim-to-real literature
says transfers worst"*.

**I could not find a primary source for that claim, and I looked for it twice** (also §9 of
`FAILURE_MINING_METHODS.md`, where a related "Spearman 0.4–0.7, severity ordering transfers worst"
note is marked **[?]** and remains unsourced). **Treat it as unsupported.**

**The refusal it justifies is still right, on a better argument that needs no citation:** prevalence
measured in simulation is a property of **the perturbation grid we chose**. We set the camera angles;
the rate at which those angles occur in a client's plant is not something our grid can estimate, at
any fidelity. That argument is self-sufficient and checkable.

**Recommended edit** (`my primary`'s call — I have not touched `vla_harness/`): keep the refusal,
delete the appeal to literature, and replace it with the grid argument. Add that what the literature
*does* support is the narrower claim established in §1.1 — **the top of the ordering transfers, the
tail does not, and magnitudes do not at all** — which is an argument for shipping *the leading
condition* and refusing rates, rather than for shipping a full ranked table.

---

## 6. What could not be verified

| claim | mark | to settle |
|---|---|---|
| SimpleVLA-RL's "no real data, strong real gains" | **[S]** | read the paper; it is the strongest anti-"toyish" claim here |
| Practical-Recipe correlation numbers | **[?]** | the summariser mis-expanded MMRV; read directly |
| DreamGen real-robot figures | **[S]** | company-affiliated preprint |
| Co-training ratios, 5–15 demos, +38% | **[S]** | snippet-level |
| "500–2,000 real demos recovers 80–95%" | **[S]** | snippet-level, no primary source located |
| Whether failure *families* transfer | **[?] — nobody has measured it** | would need paired sim/real failure annotation |
| Whether fidelity increases measurably raise correlation | **[?] — no controlled study found** | a real hole in the literature |
| Spearman ρ = 0.77 feature-ranking transfer | **[S]** | 2011 work, different setting |
