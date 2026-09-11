# Phase 0 — Supply-side coverage, and predictions committed before measurement

**Written:** 2026-09-11, **before any LIBERO stress run has been executed.**
**Dataset:** `lerobot/libero` @ `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
— 1,693 episodes · 273,465 frames · 40 tasks · Panda · 10 fps
**Method:** `experiments/phase0/coverage.py`. CPU only; no policy, no simulator.

The point of committing these first: a prediction made before the experiment is
evidence. The same story told afterwards is not. If these are wrong, that is a
result and stays in the record.

---

## 1. What the demonstrations actually cover

### The aggregate numbers are misleading — check within-task

| dim | aggregate std | median **within-task** std | ratio |
|---|---|---|---|
| eef_x | 0.0537 | **0.0062 m** | 0.116 |
| eef_y | 0.0083 | **0.0076 m** | 0.913 |
| eef_z | 0.4001 | **0.0064 m** | **0.016** |
| rot_ax | 0.00054 | **0.00052 rad** (0.03°) | 0.950 |
| rot_ay | 0.00189 | **0.00185 rad** (0.11°) | 0.976 |
| rot_az | 0.00361 | **0.00294 rad** (0.17°) | 0.815 |
| grip | 0.0000 | **0.0000** | — |

`eef_z` looks like it has 0.40 m of spread. It does not. It is **trimodal** —
three discrete clusters at z ≈ 0.24, 0.60, 1.08 with **zero mass between them**:

```
z~ 0.238  ██████████████████████████ 454
z~ 0.357   0
z~ 0.477   0
z~ 0.597  ███████████ 199
z~ 0.716   0
z~ 0.836   0
z~ 0.955   0
z~ 1.075  ████████████████████████████████████████████████████████████ 1040
```

That is across-task scene structure, not variation the policy can learn from.
Per-task mean z ranges 0.260–1.178 with std 0.393, while within any one task z
varies by **6.4 mm**.

**Taking the ratio column seriously is the whole analysis.** A policy is
conditioned on the instruction, so what governs its robustness on a given task
is within-task coverage. Reading the aggregate std as "good coverage" would have
inverted the conclusion.

### Within a task, the robot starts in essentially the same pose every time

- translation: σ ≈ **6–8 mm**
- orientation: σ ≈ **0.03°–0.17°**
- gripper: **exactly constant** (0.0388, σ = 0.0 across all 1,693 episodes)

### Camera viewpoint: zero coverage, structurally

There is no camera-pose feature in the dataset at all. LIBERO renders from
**fixed cameras** — `agentview` and `robot0_eye_in_hand`. All 273,465 frames in
all 1,693 demonstrations come from the same viewpoint. This is not sparse
coverage; it is a single point.

---

## 2. Predictions

Stated as falsifiable claims with numbers. Ranked by confidence.

**P1 — Camera viewpoint is the most brittle axis, by a wide margin.**
Zero coverage in fine-tuning. Expect the steepest degradation of any axis and a
failure boundary at a *small* angle. Concretely: success should fall below half
its nominal value within the first two LIBERO-plus difficulty levels of the
camera-pose factor.

**P2 — Initial-state sensitivity begins at a few centimetres.**
Within-task translational σ is 6–8 mm, so a 2 cm offset is already ~3σ and 4 cm
is ~5σ outside anything demonstrated. Expect measurable degradation by ~2 cm and
a crossed boundary by ~4–5 cm.

**P3 — Initial *orientation* perturbation is worse than initial *translation*.**
Orientation coverage is 0.03°–0.17°, i.e. effectively none, versus millimetres
of genuine translational jitter. This is the sharpest prediction here and the one
I have not seen stated elsewhere — LIBERO-plus's "robot initial state" factor
bundles both, so if it is reported as one number this needs separating to test.

**P4 — Gripper-state perturbation, if available, breaks the policy immediately.**
The initial gripper value is *identical* in every demonstration. Any non-zero
perturbation is out-of-distribution at step 0.

**P5 — Object layout is the most robust of the axes we can reason about.**
Weakest prediction here — see the limits below.

---

## 3. Limits — what this analysis cannot see

Stated so the predictions are not over-read.

1. **Proprioception only.** `observation.state` is 8-dim end-effector state.
   Object positions are not in it, so **object-layout coverage cannot be measured
   from this dataset.** P5 rests on the 40 tasks spanning varied scenes, which is
   an argument, not a measurement. To do better we would have to parse the BDDL
   init files or the scene XML.
2. **Fine-tuning data only.** SmolVLA was *pretrained* on community robot data
   before being fine-tuned on LIBERO. "Zero camera coverage" is true of the
   fine-tuning set, not of the model's entire training history. The base model
   has seen other viewpoints. Fine-tuning on a single viewpoint tends to collapse
   such invariance, but it does not guarantee it — P1 is a prediction about a
   mechanism, not a certainty.
3. **Coverage is not the only cause of brittleness.** A gap can be
   architectural rather than a data gap; that is the whole of `PLAN.md` §7c.
   These predictions say where to look, not what the fix is.
4. **Demos per task is uneven** — min 29, median 44, max 50, not a uniform 50.
   Tasks with 29 demos are a thinner base than tasks with 50.

---

## 4. Other observations

- **Episode lengths:** mean 162 frames, p50 140, p95 289, max 505 (at 10 fps ⇒
  ~16 s mean). Relevant to the §8 cost model — the earlier 300-step assumption
  was roughly right on average but the tail reaches 505.
- **The gripper action is fully binary.** 100% of gripper commands are at ±1;
  it is never intermediate. A policy emitting a partial gripper value would be
  producing something never demonstrated.
- **Translation actions are well inside their rails** (|a| max 0.938 of 1.0,
  0% at rail), so action clipping is not shaping the demonstrations.

---

## 5. How these get tested

Against the stress sweep, once the GPU is free. P1 and P2 fall out of the
planned viewpoint and initial-state sweeps directly. **P3 needs the LIBERO-plus
initial-state factor decomposed into translation and rotation**, which may
require going below its published difficulty levels. P4 needs a gripper-state
perturbation, which may not exist in LIBERO-plus and might have to be added.

A correct prediction here is worth considerably more than the same observation
made after the fact: it demonstrates the supply-side analysis has predictive
power, which is what would let a client act on a Data Gap Manifest row *before*
paying for the stress campaign that confirms it.
