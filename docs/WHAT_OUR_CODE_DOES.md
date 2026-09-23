# What our code does, and what it deliberately doesn't

**Written 2026-09-15; updated 2026-09-23.** Answers one question: *if HuggingFace
already ships a LIBERO evaluator, what is our code for?*

---

## 1. What `lerobot-eval` records

This is a real `eval_info.json` from our own calibration run:

```json
{"per_task": [{"task_group": "libero_spatial", "task_id": 0,
               "metrics": {"successes": [true],
                           "sum_rewards": [1.0], "max_rewards": [1.0],
                           "video_paths": ["....mp4"]}}],
 "per_group": {"libero_spatial": {"pc_success": 100.0, "n_episodes": 1}},
 "overall":   {"pc_success": 100.0, "eval_s": 8.22}}
```

**One bit per episode, plus an mp4.** That is the correct output for an
evaluator and it is not a criticism — it is what a benchmark score *is*.

(Our harness, `experiments/harness_eval.py`, now also records a per-episode mp4
by default; `--no-video` turns it off.)

But you cannot localise a failure in it. There is no per-step state, no object
pose, no contact signal, no gripper aperture, and no way to recover *which* of
280 steps went wrong or why.

## 2. The division we use

| Need | Use | Why |
|---|---|---|
| Success rates — the screen, the gate, per-task controls | ~~`lerobot-eval`, as shipped~~ **our harness (`experiments/harness_eval.py`), after proving parity with `lerobot-eval`** | Parity is measured, not assumed: MINERVA Δ 0.0 / +2.0 pp and 95.3% vs 95.75% published (R-014, R-016); GR00T 98 vs 97, McNemar p=1.00 (R-022). `lerobot-eval` remains the reference the harness is checked against. |
| Per-step traces with privileged state | **our harness** | Nothing else emits `_gt_` object poses, contacts and phase signals |
| Failure mining, attribution, the manifest | **our harness** | Nothing else does this at all |

**This was a deliberate narrowing** *(2026-09-15; since relaxed — rates now come
from our harness, which is parity-checked against `lerobot-eval` per policy)*.
Every number someone could dispute was produced by somebody else's evaluator. Our code only touches the part nobody
else does — which is also the only part we claim as a contribution
(`PLAN.md` §0b).

## 3. What our code uniquely does

**Trace capture with privileged state.** Every step records eef pose, gripper
state, joint state (policy-visible) plus object poses, contacts and distances
(`_gt_`, detectors only, stripped by `policy_view()` before any policy sees
them). Plus a `scene_descriptor` in metres so the scene could be rebuilt on a
physical bench. Two object-pose keys matter: `_gt_object_pos` holds only the
BDDL task objects (it missed a wrong-object grasp of a ramekin);
`_gt_scene_object_pos` holds every free-joint object, including distractors.

**Activation capture and splicing** (`vla_harness/capture/`). Taps on GR00T's
backbone and action head, and a splice that copies one pathway's tokens (text,
image or state) from the paired nominal run into the perturbed one, under fixed
noise. This is how we found that a vision
perturbation vanishes after the VL self-attention (R-037) and that every
LIBERO-Plus category reaches the action through the image tokens (R-039).

**Phase localisation.** Which phase the rollout was in when it broke —
approach, pre-grasp, grasp, transport, retry — derived from simulator state,
not from a success bit.

**Failure classification and clustering.** Seven families, deterministic rules
first, LLM adjudication only for the residue, never overriding a deterministic
verdict. **The family labels are still unvalidated** (RESULTS.md state-of-knowledge
item 6). So are the VLM labels from `experiments/vlm_label.py` (local Qwen3-VL-8B
4-bit or NVIDIA NIM): its first test named the wrong grasped object.

**Counterfactual attribution.** *This one is structurally impossible on
`eval_info.json`.* It re-runs **the same seed** with **one perturbation knob
reverted** and compares **paired per-seed outcomes** (McNemar). The aggregate
percentage in `eval_info.json` has already discarded the pairing; it cannot be
recovered afterwards.

**Failure cost.** Benign / disruptive / safety, from terminal state. Buyers care
what a failure *does*, not only how often it happens.

**The Data Gap Manifest.** Evidence-backed data requirements, with `severity`
deliberately *not* shipped as a cardinal number (`PLAN.md` §9.3).

**Provenance and caching discipline.** Content-addressed rollout ids, a
three-bucket fingerprint, an arms request log, regression sets bound to the env
identity they were frozen against.

## 4. The honest proportions

5,784 lines in `vla_harness/` (re-counted 2026-09-23 with
`find vla_harness -name '*.py' | xargs wc -l`; it was 2,014 on 2026-09-15).

```
mining + manifest                 1,038 lines   18%   <- the differentiator
capture + analysis + probes       1,160 lines   20%   <- mechanism work (R-037, R-039)
schema (contract/prov.)             544 lines    9%
runner + control                    577 lines   10%
adapters, conformance, fixtures   2,465 lines   43%
```

**Roughly two-fifths of the code is the contribution** (mining plus capture). The
rest exists to make that part trustworthy, and most of it is the LIBERO and
LeRobot adapters plus the conformance gate.

That ratio is defensible but should be stated rather than discovered. The design
reviews found real defects in the infrastructure — a cache that served episodes from a
different environment, a trace missing the frame in which success was decided, a
sampling design that would have de-uniformed its own control arm. Each would
have produced a confident false finding, so the infrastructure work was not
wasted. But it *is* infrastructure.

**One genuine duplication:** `runner.rollout()` re-implements a rollout loop
`lerobot-eval` already has. It exists because we need per-step `_gt_` state that
LeRobot's loop does not emit. Real cost, defensible, and worth re-examining if a
way appears to instrument LeRobot's loop instead of replacing it.

## 5. The thing to keep in view

~~**The mining layer has never seen a real VLA failure.**~~ *(True on 2026-09-15;
no longer.)* The mining layer has now run on real failures: 155 GR00T failures out
of 623 LIBERO-Plus L4+L5 variants, with robot initial state and camera the worst
categories (R-026, R-031). The experiments have run too: the language ablation
(R-038), and the pathway work (R-037, R-039).

**What is still not in view:** whether the family labels are *correct*. They have
no fixture on LIBERO and the adjudication CSV is 0 of 80 verdicts (`HANDOFF.md` §7).
Treat any family count as unvalidated until that lands.
