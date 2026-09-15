# What our code does, and what it deliberately doesn't

**Written 2026-09-15.** Answers one question: *if HuggingFace already ships a
LIBERO evaluator, what is our code for?*

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

But you cannot localise a failure in it. There is no per-step state, no object
pose, no contact signal, no gripper aperture, and no way to recover *which* of
280 steps went wrong or why.

## 2. The division we use

| Need | Use | Why |
|---|---|---|
| Success rates — the screen, the gate, per-task controls | **`lerobot-eval`, as shipped** | Maintained by HF, already the published protocol, **and not our code to defend**. Any number a reader might challenge comes from the standard tool. |
| Per-step traces with privileged state | **our harness** | Nothing else emits `_gt_` object poses, contacts and phase signals |
| Failure mining, attribution, the manifest | **our harness** | Nothing else does this at all |

**This is a deliberate narrowing.** Every number someone could dispute is
produced by somebody else's evaluator. Our code only touches the part nobody
else does — which is also the only part we claim as a contribution
(`PLAN.md` §0b).

## 3. What our code uniquely does

**Trace capture with privileged state.** Every step records eef pose, gripper
state, joint state (policy-visible) plus object poses, contacts and distances
(`_gt_`, detectors only, stripped by `policy_view()` before any policy sees
them). Plus a `scene_descriptor` in metres so the scene could be rebuilt on a
physical bench.

**Phase localisation.** Which phase the rollout was in when it broke —
approach, pre-grasp, grasp, transport, retry — derived from simulator state,
not from a success bit.

**Failure classification and clustering.** Seven families, deterministic rules
first, LLM adjudication only for the residue, never overriding a deterministic
verdict.

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

2,014 lines total.

```
mining + manifest        461 lines   23%   <- the differentiator
schema (contract/prov.)  517 lines   26%
runner                   344 lines   17%
adapters + fixtures      692 lines   34%
```

**Roughly a quarter of the code is the contribution.** The other three quarters
exist to make that quarter trustworthy.

That ratio is defensible but should be stated rather than discovered. The design
reviews found real defects in the 77% — a cache that served episodes from a
different environment, a trace missing the frame in which success was decided, a
sampling design that would have de-uniformed its own control arm. Each would
have produced a confident false finding, so the infrastructure work was not
wasted. But it *is* infrastructure.

**One genuine duplication:** `runner.rollout()` re-implements a rollout loop
`lerobot-eval` already has. It exists because we need per-step `_gt_` state that
LeRobot's loop does not emit. Real cost, defensible, and worth re-examining if a
way appears to instrument LeRobot's loop instead of replacing it.

## 5. The thing to keep in view

**The mining layer has never seen a real VLA failure.** It has been validated
against a toy with planted faults, and on first contact with LIBERO it correctly
*abstained* (§2 of `COMPONENTS.md`) because its detectors are tuned to toy state
keys.

Everything built so far is scaffolding for an experiment not yet run. That is a
normal place to be; it stops being normal if it continues.
