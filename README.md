# VLA Stress-Test — with Failure Mining

Systematically stress a Vision-Language-Action policy in simulation, localise
*where* and *why* it fails, cluster the failures, and convert them into a costed,
evidence-carrying **Data Gap Manifest** — then validate at least one row by
retraining and re-measuring.

The output is not a leaderboard score. It is a repeatable method for turning model
behaviour into a data strategy: what failed, under what conditions, how often, what it
would cost, and whether more data would fix it at all.

> **Status: prototype.** Phase 0 (supply-side coverage) and Phase 1 (harness + oracle
> gate) are complete. Phase 2 (real policy on LIBERO) is in progress. Nothing here has
> yet produced a client-facing result, and several findings below are open risks rather
> than achievements. See [`docs/REVIEW_SUMMARY.md`](docs/REVIEW_SUMMARY.md).

---

## Honest positioning

A 238-source literature survey ([`docs/LANDSCAPE.md`](docs/LANDSCAPE.md)) found the
core loop — perturb, cluster, recommend — is **substantially prior art**. Perturbation
generation (LIBERO-Plus, COLOSSEUM, VLATest), failure clustering (RoboFAC, AHA),
runtime monitoring (Sentinel, FIPER) and demonstration scoring (CUPID, Demo-SCORE) are
all published. LIBERO-Plus §6.2 has already run a version of the closed loop, taking
camera-viewpoint robustness from 55.6% to 92.8% with 20k remediation trajectories.

Three things survive the survey as genuinely ours:

1. **The Data Gap Manifest as a costed artifact** — severity, fixability,
   discriminators and validation status per row. The ideas exist; the artifact does not.
2. **Counterfactual attribution of policy failures to environment factors** — formalised
   for LLM agents, essentially unexploited in robot policy evaluation.
3. **Failure-cost weighting** — no perturbation benchmark found weights failures by
   consequence.

The open claim worth testing is narrower than the original proposal's: not *"targeted
data closes the gap"* (published), but **"targeted collection beats broad collection at
equal budget."** That is what the mining layer is for, and it needs a control arm.

---

## Architecture

Five layers. The model is a plugin at L1; four of the five need no GPU.

```
L4  manifest.py    Data Gap Manifest generation
L3  mining/        phase detection · classification · clustering
L2  envs/          ToyReachEnv → LIBERO → LIBERO-plus     (Env protocol)
L1  policies/      ScriptedPolicy → SmolVLA → π0          (Policy protocol)
L0  schema.py      canonical Rollout contract + trace store
```

**The contract is the product.** Everything above L0 reads and writes the same
`Rollout`. Swapping the policy or the simulator is a one-file change. The mining layer
imports nothing from `envs/` or `policies/` — if a detector needs a signal, it belongs
in `Observation.state`.

Two invariants worth knowing before reading the code:

- **`_gt_` keys are privileged.** They are simulator ground truth a real robot does not
  have. `runner.rollout` passes `obs.policy_view()` to the policy, which strips them, so
  a policy *cannot* read one. Detectors get the full state.
- **Cache hits are verified, not trusted.** Every `Rollout` carries a fingerprint
  (`env` + `policy` identity + runtime). A hit whose fingerprint disagrees is re-run and
  reported, because a stale cache reports a *number*, not an exception.

---

## Layout

```
vla_harness/          the harness (L0–L4)
experiments/
  oracle_test.py      the Phase-1 acceptance gate
  phase0/             supply-side coverage + predictions committed before measurement
  repro/              SmolVLA/LIBERO reproduction, deliberately outside vla_harness
docs/
  REVIEW_SUMMARY.md   ← start here
  LANDSCAPE.md        prior-art survey
  reviews/            five detailed design reviews
```

---

## Running it

The oracle gate needs no GPU, no simulator and no model — pure Python:

```bash
python3 experiments/oracle_test.py
```

It plants known faults in a scripted policy, runs the full
sweep → probe → classify → cluster → manifest pipeline, and checks the diagnosis
against the fault that was planted. Validating a failure miner normally has no ground
truth; planting the fault creates one.

The LIBERO reproduction path needs the pinned environment (see
[`MODELS_AND_COMPUTE.md`](MODELS_AND_COMPUTE.md) §3):

```bash
./experiments/repro/run_repro.sh <n_action_steps> <seed> [suites]
```

It calls `lerobot-eval` as shipped rather than our adapters, so that *"the checkpoint
doesn't reproduce"* stays distinguishable from *"our adapter is buggy."*

---

## What we have established so far

Full record in [`FINDINGS.md`](FINDINGS.md). The three that most change the plan:

**MuJoCo ≥ 3.4.0 silently breaks a LIBERO task, and a fresh install lands on the broken
side.** A correct box-box collision fix invalidated LIBERO's stored initial states; on
`libero_spatial` task 5 a SmolVLA fine-tune drops from 80% to 28%. LeRobot pins
`mujoco<3.9.0`, which guards against API breaks and not behavioural ones. Left uncaught,
this is a physics change masquerading as a model failure — it would have been clustered,
attributed to a co-occurring perturbation, and written into a manifest row recommending
a client buy bowl-on-ramekin demonstrations. Pinned to 3.3.7.

**The fine-tuning set contains zero camera-pose variation.** Phase 0 measured
within-task end-effector spread of ~6 mm and rotational spread under 0.2°. Aggregate
statistics are misleading — `eef_z` looks like it spans 0.40 m and is actually trimodal
with zero mass between clusters. Predictions were committed before any stress run.

**Published LIBERO numbers may not measure capability.** LIBERO-PRO reports models above
90% collapsing to 0.0% under fair perturbation, with success going to zero once object
displacement exceeds 0.2 units. If the baseline has memorised rather than generalised,
more demonstrations in the failing region teach more memorisation — which makes
`fixability: architecture` potentially the *modal* answer, not the rare one.

---

## Known open risks

Carried honestly because the reviews exist to surface them:

| | Risk |
|---|---|
| Reproduction gate | SmolVLA publishes ~87.3% on LIBERO; two independent open reports get 73.25% and ~67%. Against a ±5 pp gate that is a 14 pp gap, and there is no budget for a rented fallback reference. |
| Oracle gate | Reports 3/4, but one fixture cannot fire, so the pass threshold equals the ceiling. It measures recall only — on a one-fault policy the pipeline emits a false second manifest row. |
| Manifest completeness | `fixability`, `discriminators` and `failure_cost` are mandatory in the schema and not yet emitted. Confidence intervals are computed and dropped before the artifact. |
| Regression set | Named as build item 7 of 7 in the proposal, referenced by every manifest row, and does not exist. Must be frozen *before* remediation data or Phase 5's comparison is contaminated. |
| Phase coverage | Detectors stop at `transport` — no `place`, no `release`, no BDDL predicates. The second half of a LIBERO task is currently invisible to the miner. |

---

## Documents

| | |
|---|---|
| [`VLA Scenario Testing.md`](VLA%20Scenario%20Testing.md) | the original proposal |
| [`PLAN.md`](PLAN.md) | how we build it, on the hardware we have |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | who produces which data, and where to look when a number is wrong |
| [`MODELS_AND_COMPUTE.md`](MODELS_AND_COMPUTE.md) | what we can run, on what, for how much |
| [`FINDINGS.md`](FINDINGS.md) | running record of what we've established |
| [`docs/LANDSCAPE.md`](docs/LANDSCAPE.md) | prior-art survey |
| [`docs/REVIEW_SUMMARY.md`](docs/REVIEW_SUMMARY.md) | design critique summary |

`.venvs/` and `third_party/` are excluded from the repository; rebuild from the pinned
versions recorded in each run's `provenance.txt`.
