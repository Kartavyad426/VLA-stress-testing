# VLA Stress-Test — with Failure Mining

Systematically stress a Vision-Language-Action policy in simulation, localise
*where* and *why* it fails, cluster the failures, and convert them into a costed,
evidence-carrying **Data Gap Manifest** — then validate at least one row by
retraining and re-measuring.

The output is not a leaderboard score. It is a repeatable method for turning model
behaviour into a data strategy: what failed, under what conditions, how often, what it
would cost, and whether more data would fix it at all.

> **Status (2026-09-23): research prototype.** The harness reproduces published LIBERO
> numbers and runs GR00T N1.7 on LIBERO-Plus. The work has moved from *measuring*
> failures to *locating* them inside the policy (which pathway carries a perturbation
> to the action) and designing the data that would fix them. Nothing here is
> client-facing yet. New readers: start with [Reading order](#reading-order).

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
L1  policies/      ScriptedPolicy → GR00T N1.7 · MINERVA  (Policy protocol)
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
  capture/            activation taps on GR00T (backbone, adapter, action head)
  video.py            per-episode mp4, recorded by default
experiments/
  harness_eval.py     run a policy over LIBERO / LIBERO-Plus tasks through the harness
  visualise_set.py    reference-vs-variants page: synced videos, signals, activations
  vlm_label.py        VLM description of an episode (unvalidated; see RESULTS.md)
  oracle_test.py      the Phase-1 acceptance gate
  phase0/             supply-side coverage + predictions committed before measurement
  repro/              LIBERO reproduction via lerobot-eval, outside vla_harness
runs/                 one folder per run: metadata in git, traces and video not
viz/                  generated pages (local; large ones are not committed)
docs/                 see Reading order below
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

A GR00T run on LIBERO-Plus through the harness (announce and take the GPU lock first;
other sessions share the card):

```bash
flock /tmp/vla_gpu.lock env MUJOCO_GL=egl PYTHONPATH=third_party/LIBERO-plus \
  LIBERO_CONFIG_PATH=third_party/libero-plus-config \
  .venvs/libero-plus/bin/python experiments/harness_eval.py \
  --checkpoint nvidia/gr00t17-lerobot-libero_spatial-640 --n-action-steps 16 \
  --override base_model_path=nvidia/GR00T-N1.7-3B --override embodiment_tag=libero_sim \
  --dtype bfloat16 --obs-size 360 --libero-plus --base-instruction \
  --rename-map '{"observation.images.image2": "observation.images.wrist_image"}' \
  --suites libero_spatial --tasks 0,1 --episodes 1 --run-id my_run
```

To watch episodes side by side against an unperturbed reference:
`experiments/visualise_set.py RUN:ROLLOUT_ID [RUN:ROLLOUT_ID ...]` (see its docstring).

The LIBERO reproduction path needs the pinned environment (see
[`MODELS_AND_COMPUTE.md`](MODELS_AND_COMPUTE.md) §3):

```bash
./experiments/repro/run_repro.sh <n_action_steps> <seed> [suites]
```

It calls `lerobot-eval` as shipped rather than our adapters, so that *"the checkpoint
doesn't reproduce"* stays distinguishable from *"our adapter is buggy."*

---

## What we have established so far

The running record is [`RESULTS.md`](RESULTS.md): its **Current state of knowledge**
section and its **Index** of every experiment (R-001 onward). The headlines:

- **The measuring instrument is sound.** MINERVA reproduces its published LIBERO score
  (95.3% vs 95.75%). GR00T N1.7 runs the same through our harness as through
  `lerobot-eval` (98 vs 97). Unperturbed LIBERO-Plus scenes score 100/100. SmolVLA was
  dropped because it never reproduced a published number.
- **Every policy here is behaviour-cloned from demonstrations**, so a failure under
  perturbation is first a question about what those demonstrations covered.
- **GR00T breaks mainly under robot-initial-state and camera perturbations.** That's
  155 failures in 623 hard LIBERO-Plus variants, the same ordering as the LIBERO-Plus paper.
- **Language is not inert.** With the instruction removed, GR00T falls from 100/100 to
  50/100, from 0/10 to 10/10 depending on the scene. That overturns the published
  "insensitive to language" reading.
- **Inside the policy:** a visual perturbation is clearly visible in the vision-language
  backbone's output and has almost vanished by the time the action head reads it. For
  every perturbation type, the change reaches the action through the image tokens, not
  the robot-state token.
- **Some failures are wrong-object grasps**, which the original traces could not see.
  The environment now records every object's position, not just the task objects.

Early findings that still stand: MuJoCo is pinned to 3.3.7, because ≥ 3.4.0 silently
changes a LIBERO task; the fine-tuning data contains no camera-pose variation (Phase
0); published LIBERO numbers can reflect memorisation (LIBERO-PRO). Details are in
[`FINDINGS.md`](FINDINGS.md).

---

## Known open risks

| | Risk |
|---|---|
| Failure labels | Failure-family labels are unvalidated, and the human adjudication sheet is at 0 of 80 verdicts. VLM labels are unvalidated too: the first test named the wrong grasped object. |
| Correlation, not cause | The "signal vanishes in the adapter" localisation is correlational. The pathway result (R-039) is an intervention, but on 40 instances. |
| Compute | One 8 GB GPU is shared by several sessions. The π0 family does not fit, so a second modern policy needs a rented GPU (PENDING #25). |
| External claims | The action-atlas paper's GR00T layer ordering (arXiv:2603.19233) could not be reproduced: its code does not load under its own pinned versions. Treat it as unverified. |
| Manifest | `fixability`, `discriminators` and `failure_cost` are still not emitted. The regression set named in the proposal does not exist yet and must be frozen before any remediation data. |

---

## Reading order

For someone new to the project who wants a high-level picture of what has been done
and what is happening now. About an hour, in this order:

| | Read | What you get |
|---|---|---|
| 1 | [`docs/VLA_101.md`](docs/VLA_101.md) | What a VLA policy is, how it is trained (imitation of demonstrations), and the setups used here |
| 2 | This README | What the project is for and what it has found |
| 3 | [`docs/WHAT_OUR_CODE_DOES.md`](docs/WHAT_OUR_CODE_DOES.md) | What the harness does, and deliberately does not do |
| 4 | [`RESULTS.md`](RESULTS.md): *Current state of knowledge* and *Index* only | Every experiment in one table; the current conclusions |
| 5 | [`HANDOFF.md`](HANDOFF.md) | What is running now, what is open, and the working conventions |
| 6 | [`docs/FAILURE_TO_DATA_PIPELINE.html`](docs/FAILURE_TO_DATA_PIPELINE.html) | Where the work is heading: from a failure to a data specification |
| 7 | [`docs/R039_RESULTS.html`](docs/R039_RESULTS.html) | The latest headline experiment: which pathway carries a perturbation to the action |

### Extended reading

**Project design:** [`VLA Scenario Testing.md`](VLA%20Scenario%20Testing.md) (the
original proposal) · [`PLAN.md`](PLAN.md) · [`ARCHITECTURE.md`](ARCHITECTURE.md) ·
[`docs/FLOW.md`](docs/FLOW.md) · [`docs/COMPONENTS.md`](docs/COMPONENTS.md) ·
[`IMPLEMENTATION.md`](IMPLEMENTATION.md)

**Setup and benchmarks:** [`MODELS_AND_COMPUTE.md`](MODELS_AND_COMPUTE.md) ·
[`docs/DATA_AND_BENCHMARKS.md`](docs/DATA_AND_BENCHMARKS.md) ·
[`docs/LIBERO_PLUS_LEVELS.md`](docs/LIBERO_PLUS_LEVELS.md) ·
[`docs/EXPERIMENT_PROCEDURE.md`](docs/EXPERIMENT_PROCEDURE.md) ·
[`docs/BASELINE_FORENSICS.md`](docs/BASELINE_FORENSICS.md)

**Decisions and history:** [`PENDING_DECISIONS.md`](PENDING_DECISIONS.md) ·
[`FINDINGS.md`](FINDINGS.md) · [`NEXT_STEPS.md`](NEXT_STEPS.md) ·
[`docs/REVIEW_SUMMARY.md`](docs/REVIEW_SUMMARY.md) and [`docs/reviews/`](docs/reviews/)

**Failure mining and taxonomy:** [`docs/OUR_MINING_APPROACH.md`](docs/OUR_MINING_APPROACH.md) ·
[`docs/FAILURE_MINING_METHODS.md`](docs/FAILURE_MINING_METHODS.md) ·
[`docs/TAXONOMY_FAMILY_GUIDELINES.md`](docs/TAXONOMY_FAMILY_GUIDELINES.md) ·
[`docs/FAILURE_FAMILY_AUDIT.md`](docs/FAILURE_FAMILY_AUDIT.md) ·
[`docs/RETRAINING_DEFAULT_TAXONOMY.md`](docs/RETRAINING_DEFAULT_TAXONOMY.md) ·
[`docs/COVERAGE_GAP_METHOD.md`](docs/COVERAGE_GAP_METHOD.md)

**Inside the policy (advanced):** [`docs/FRAME_LABEL_METHODOLOGY.md`](docs/FRAME_LABEL_METHODOLOGY.md) ·
[`docs/EXP_EMBEDDING_OOD.md`](docs/EXP_EMBEDDING_OOD.md) ·
[`docs/R039_MATHS.html`](docs/R039_MATHS.html) ·
[`docs/EPISODE_RECORD_SCHEMA.md`](docs/EPISODE_RECORD_SCHEMA.md) ·
[`docs/TRIGGER_MECHANISM_CALCULUS.html`](docs/TRIGGER_MECHANISM_CALCULUS.html) ·
[`docs/VLA_JOIN_TOPOLOGY.html`](docs/VLA_JOIN_TOPOLOGY.html)

**Prior art and related work:** [`docs/LANDSCAPE.md`](docs/LANDSCAPE.md) (238-source
survey) · [`docs/SENTINEL_METHODOLOGY.html`](docs/SENTINEL_METHODOLOGY.html) ·
[`docs/MIND_ROBOTICS_METHODOLOGY.html`](docs/MIND_ROBOTICS_METHODOLOGY.html) ·
[`docs/POLICY_SIM_COUPLING.md`](docs/POLICY_SIM_COUPLING.md) ·
[`docs/SIM_TO_REAL_TRANSFER.md`](docs/SIM_TO_REAL_TRANSFER.md) ·
[`docs/REAL_SIM_PAIRED_DATA.md`](docs/REAL_SIM_PAIRED_DATA.md) ·
[`docs/CONSUMER_ELICITATION_DESIGN.md`](docs/CONSUMER_ELICITATION_DESIGN.md)

**Why "behaviour-cloned":** VLA policies are trained by imitating demonstrations, and
fail out of distribution for exactly that reason. See VLA-RL (Lu et al., 2025,
[arXiv:2505.18719](https://arxiv.org/abs/2505.18719)) and SimpleVLA-RL (Li et al., 2025,
[arXiv:2509.09674](https://arxiv.org/abs/2509.09674)).

`.venvs/` and `third_party/` are excluded from the repository; rebuild from the pinned
versions recorded in each run's `provenance.txt`.
