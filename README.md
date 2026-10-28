# VLA Stress-Test — with Failure Mining

Systematically stress a Vision-Language-Action policy in simulation, localise
*where* and *why* it fails, cluster the failures, and convert them into a costed,
evidence-carrying **Data Gap Manifest** — then validate at least one row by
retraining and re-measuring.

The output is not a leaderboard score. It is a repeatable method for turning model
behaviour into a data strategy: what failed, under what conditions, how often, what it
would cost, and whether more data would fix it at all.

> **Status (2026-10-19): research prototype.** The harness reproduces published LIBERO
> numbers and runs GR00T N1.7 on LIBERO-Plus. A perturbation → correction map covers
> all seven LIBERO-Plus perturbation types. The first retraining pilot (R-053 to R-058)
> did not close the gap. Since 10-06 the work has moved to **finding the root of each
> failure** and to a **mining spec** (the format in which failures, their roots and
> matching data are delivered). Since 10-14 it has grown into an **end-to-end loop**:
> scene from a real recording → task set → policy runs → failure root → mechanism →
> corrected demo → retrain → held-out re-test
> ([`docs/END_TO_END_PIPELINE.html`](docs/END_TO_END_PIPELINE.html), draft). Steps 1–3
> and 6 have working proofs of concept; the rest are design only or open. Nothing here
> is client-facing yet. New readers: start with [Reading order](#reading-order). For
> what is running and open right now, see
> [`docs/handoffs/HANDOFF_2026-10-15.html`](docs/handoffs/HANDOFF_2026-10-15.html).

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

## Pipeline

The end-to-end loop, from a robot failure to a better policy
([`docs/END_TO_END_PIPELINE.html`](docs/END_TO_END_PIPELINE.html)):

```
1 Scene            a simulated workspace (LIBERO, or rebuilt from a real recording)
2 Task set         what to do in it: instruction + success check + time limit
3 Policy runs      many rollouts
4 Failures         the success check says no
5 Root             where it went wrong
6 Mechanism        why: which input carries the failure, and when it is decided
7 Corrected demo   from the root to the goal, in sim
8 Retrain          add the new data
9 Held-out re-test on unseen episodes; the new policy goes back to step 3
```

The one design choice that differs from a naive "fail → correct → retrain" loop is
step 6: we classify the mechanism *before* correcting, because the mechanism decides
which fix can work. The loop is not built end to end yet.

---

## Layout

```
vla_harness/          the harness (L0–L4)
  capture/            activation taps on GR00T (backbone, adapter, action head)
  training/           GR00T LoRA fine-tuning: replay mix, cached features, verified LR schedule
  data/               demo record contract, rescue source, LeRobot export
  video.py            per-episode mp4, recorded by default
experiments/
  harness_eval.py     run a policy over LIBERO / LIBERO-Plus tasks through the harness
  visualise_set.py    reference-vs-variants page: synced videos, signals, activations
  vlm_label.py        VLM description of an episode (unvalidated; see RESULTS.md)
  oracle_test.py      the Phase-1 acceptance gate
  phase0/             supply-side coverage + predictions committed before measurement
  repro/              LIBERO reproduction via lerobot-eval, outside vla_harness
  real2sim/           real recording → MuJoCo scene, tasks, composites, policy rollouts
  lmax_*.py           LIBERO-MAX GR00T runner and full-state recording
runs/                 one folder per run: metadata in git, traces, video, weights not
data/                 minted training demos (LeRobot format)
.claude/skills/       project skills (e.g. finetune-postmortem)
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
- **The perturbation → correction map is complete** for all seven rows
  ([`docs/PERTURBATION_MAP.html`](docs/PERTURBATION_MAP.html)). For start pose, the
  wrist camera carries the effect, and one corrected first action chunk rescues 20 of 24
  failures, where a plain re-run rescues 0 of 22 (R-053). The outcome is decided at the
  first chunk (R-057).
- **Success falls with start-pose distance** (R-047): about 0.93 at a gripper offset of
  8 cm or less, 0.65 at 8–12 cm, 0.22 at 12–16 cm and 0.15 beyond 16 cm, and it depends
  on the direction of the offset, not only its size.
- **The first retraining pilot did not help.** A LoRA fine-tune on 100 minted start-pose
  demos scored 59/108 on held-out starts against the base model's 67/108 (R-056, p = 0.24).
  Rank 4 scored 65/108 (R-058). The whole drop is one scene (1327) at the grasp. The model
  did fit its training frames, and why that did not carry over to held-out starts is
  being tested. The demos stop at 8 cm while failures sit beyond 12 cm, so the next round
  needs a different data source
  ([`docs/R056_POSTMORTEM.html`](docs/R056_POSTMORTEM.html),
  [`docs/DATA_SOURCING_RESEARCH.html`](docs/DATA_SOURCING_RESEARCH.html)).

Since 10-06 (all exploratory: produced by new code that the pipeline checker,
[`PIPELINE_CHECKS.md`](PIPELINE_CHECKS.md), has not yet passed):

- **Mining spec.** Design decisions D1–D15 are agreed
  ([`docs/MINING_SPEC_DESIGN.html`](docs/MINING_SPEC_DESIGN.html)) and a toy end-to-end
  example exists ([`docs/MINING_SPEC_TOY.html`](docs/MINING_SPEC_TOY.html)). The v1
  design is written and awaits approval; nothing of v1 is coded.
- **Root identification** ([`docs/ROOT_IDENTIFICATION.html`](docs/ROOT_IDENTIFICATION.html)).
  Root-signal experiments E1–E7 and R-059 run in the worktree branch `exp/root-signals`.
  **R-059 (STAC-style sampling disagreement as an early warning) did not work** on our
  failures: AUROC 0.18–0.43 at the first four model calls, with failures showing *lower*
  disagreement; the as-published score (0.97) is explained by episode length. STAC
  tracks motion size, and our failures are smooth timeouts. A simple progress signal
  (gripper-to-nearest-object distance) reached AUROC 0.91 by the fourth call
  ([`docs/STAC_GROOT_THEORY.html`](docs/STAC_GROOT_THEORY.html)). E7, VLM answers at
  event frames, is not usable yet: on a 40-frame hand-labelled set, Cosmos-Reason2-2B
  matched the human on 10 of 26 decided questions.
- **Corrective demos from the root (DMP proof of concept)**
  ([`docs/DMP_POC_METHOD.html`](docs/DMP_POC_METHOD.html),
  [`docs/DMP_POC_SHOWCASE.html`](docs/DMP_POC_SHOWCASE.html); code on branch
  `exp/dmp-poc`). Restoring the failure moment by action replay is exact (0.0 mm) in
  all five test scenarios; bending the nearest human demos with a DMP kept a correction
  in every main scenario (10 of 15 attempts, against 7 of 15 for a straight-line
  bridge). The sweep over all failures has not run.
- **Real-to-sim scenes** ([`docs/REAL_TO_SIM_SCENES.html`](docs/REAL_TO_SIM_SCENES.html),
  [`docs/REAL_TO_SIM_LESSONS.html`](docs/REAL_TO_SIM_LESSONS.html),
  [`docs/REAL_TO_SIM_PREDEPLOY_CHECKLIST.html`](docs/REAL_TO_SIM_PREDEPLOY_CHECKLIST.html);
  code in `experiments/real2sim/`). Two DROID episodes are rebuilt as MuJoCo scenes, with
  generated language tasks (a scripted arm solves 91 of 105). GR00T-N1.7-DROID runs in
  them end to end and reaches the right object but never grasps; stereo depth showed the
  rebuilt table sits about 6 cm too high, and the scenes are being re-calibrated against
  a 23-item pre-deploy checklist. Only 1 of 181 public LeRobot datasets ships camera
  calibration.
- **LIBERO-MAX** (`third_party/LIBERO-MAX`, a benchmark with a mid-task change in each
  paired episode). A Spatial-only pilot (133 pairs) gave GR00T 86.5% without the change
  and 61.7% with it. The full Lite reproduction (800 pairs), recording full sim state for
  later failure analysis, is about to run.
- **R-060 (feature-level perturbation pairs):** the camera-pose bug in the paired clean
  frame is fixed (LIBERO-Plus's own camera), and the amended exactness gate G0c passes
  at an 8-pixel tolerance. Whether the fix changes past camera rows (R-039, R-047,
  R-048) is being checked.
- The design behind the corrective demos is in
  [`docs/ROOT_TO_GOAL_CORRECTIVE_DEMOS.html`](docs/ROOT_TO_GOAL_CORRECTIVE_DEMOS.html).
  Papers adopted or rejected since 24 Sep:
  [`docs/PAPERS_SINCE_24SEP.html`](docs/PAPERS_SINCE_24SEP.html).

Early findings that still stand: MuJoCo is pinned to 3.3.7, because ≥ 3.4.0 silently
changes a LIBERO task; the fine-tuning data contains no camera-pose variation (Phase
0); published LIBERO numbers can reflect memorisation (LIBERO-PRO). Details are in
[`FINDINGS.md`](FINDINGS.md).

---

## Known open risks

| | Risk |
|---|---|
| Failure labels | Failure-family labels are unvalidated, and the human adjudication sheet is at 0 of 80 verdicts. No VLM tested so far labels event frames reliably (E7, 10-12); a hand-labelled set comes first. |
| Unchecked numbers | Results from code written since 10-09 are exploratory until the pipeline checker passes them ([`PIPELINE_CHECKS.md`](PIPELINE_CHECKS.md)). |
| Render nondeterminism | Some reset frames render differently between processes (e.g. scene 984), so bitwise repeat checks fail on those cells; see R-047, E4 and R-059. |
| Correlation, not cause | The "signal vanishes in the adapter" localisation is correlational. The pathway result (R-039) is an intervention, but on 40 instances. |
| Compute | One 8 GB GPU and 30 GB RAM are shared by several sessions. A GR00T load peaks around 16–18 GB of RAM; the R-058 rank-64 run was OOM-killed. The π0 family does not fit, so a second modern policy needs a rented GPU (PENDING #25). |
| Retraining data | Rescue-minted demos cover start-pose offsets of 8 cm or less, but failures sit beyond 12 cm. Minting with the current method yields about 20% beyond 16 cm. A new source (bridge-and-join or a scripted expert) is undecided (D21). |
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
| 5 | [`docs/END_TO_END_PIPELINE.html`](docs/END_TO_END_PIPELINE.html) (draft) | The whole loop on one page, with diagrams, the state of each step and the gaps |
| 6 | [`docs/handoffs/HANDOFF_2026-10-15.html`](docs/handoffs/HANDOFF_2026-10-15.html) and [`docs/Decision_28Sep.html`](docs/Decision_28Sep.html) | What each working session is doing, what is running now, and every decision still open |
| 7 | [`docs/PERTURBATION_MAP.html`](docs/PERTURBATION_MAP.html) | Which input carries each perturbation to the action, and what corrects it |
| 8 | [`docs/R056_POSTMORTEM.html`](docs/R056_POSTMORTEM.html) | The first retraining pilot and why it did not help |
| 9 | [`docs/MINING_SPEC_DESIGN.html`](docs/MINING_SPEC_DESIGN.html) and [`docs/ROOT_IDENTIFICATION.html`](docs/ROOT_IDENTIFICATION.html) | The deliverable format, and how the root of a failure is found |

Earlier day summary: [`docs/EXPERIMENTS_2026-10-09.html`](docs/EXPERIMENTS_2026-10-09.html).
All handoffs: [`docs/handoffs/index.html`](docs/handoffs/index.html).

Older handoff notes and working conventions are in [`HANDOFF.md`](HANDOFF.md) and
[`docs/HANDOFF_6Oct.html`](docs/HANDOFF_6Oct.html).

### Extended reading

**Project design:** [`VLA Scenario Testing.md`](VLA%20Scenario%20Testing.md) (the
original proposal) · [`PLAN.md`](PLAN.md) ·
[`docs/FLOW.md`](docs/FLOW.md) · [`docs/COMPONENTS.md`](docs/COMPONENTS.md) ·
[`IMPLEMENTATION.md`](IMPLEMENTATION.md)

**Setup and benchmarks:** [`MODELS_AND_COMPUTE.md`](MODELS_AND_COMPUTE.md) ·
[`docs/DATA_AND_BENCHMARKS.md`](docs/DATA_AND_BENCHMARKS.md) ·
[`docs/LIBERO_PLUS_LEVELS.md`](docs/LIBERO_PLUS_LEVELS.md) ·
[`docs/EXPERIMENT_PROCEDURE.md`](docs/EXPERIMENT_PROCEDURE.md) ·
[`docs/BASELINE_FORENSICS.md`](docs/BASELINE_FORENSICS.md)

**Retraining:** [`docs/FINE_TUNING_101.html`](docs/FINE_TUNING_101.html) ·
[`docs/RETRAINING_PLAN.html`](docs/RETRAINING_PLAN.html) ·
[`docs/DATA_MINER_SPEC.html`](docs/DATA_MINER_SPEC.html) ·
[`docs/DATA_SOURCING_RESEARCH.html`](docs/DATA_SOURCING_RESEARCH.html) ·
[`docs/FAILURE_TO_DATA_PIPELINE.html`](docs/FAILURE_TO_DATA_PIPELINE.html)

**Decisions and history:** [`PENDING_DECISIONS.md`](PENDING_DECISIONS.md) ·
[`FINDINGS.md`](FINDINGS.md) · [`NEXT_STEPS.md`](NEXT_STEPS.md) ·
[`docs/REVIEW_SUMMARY.md`](docs/REVIEW_SUMMARY.md) and [`docs/reviews/`](docs/reviews/)

**Failure mining and taxonomy:** [`docs/OUR_MINING_APPROACH.md`](docs/OUR_MINING_APPROACH.md) ·
[`docs/FAILURE_MINING_METHODS.md`](docs/FAILURE_MINING_METHODS.md) ·
[`docs/TAXONOMY_FAMILY_GUIDELINES.md`](docs/TAXONOMY_FAMILY_GUIDELINES.md) ·
[`docs/FAILURE_FAMILY_AUDIT.md`](docs/FAILURE_FAMILY_AUDIT.md) ·
[`docs/RETRAINING_DEFAULT_TAXONOMY.md`](docs/RETRAINING_DEFAULT_TAXONOMY.md) ·
[`docs/COVERAGE_GAP_METHOD.md`](docs/COVERAGE_GAP_METHOD.md) ·
[`docs/FAILURE_LABEL_DATASETS.html`](docs/FAILURE_LABEL_DATASETS.html) ·
[`docs/IDM_VLA_HORIZONS.html`](docs/IDM_VLA_HORIZONS.html)

**Inside the policy (advanced):** [`docs/R039_RESULTS.html`](docs/R039_RESULTS.html) ·
[`docs/FRAME_LABEL_METHODOLOGY.md`](docs/FRAME_LABEL_METHODOLOGY.md) ·
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
Exception: `third_party/LIBERO-MAX` is a git submodule pinned to 01625e2 (added 2026-10-14); after cloning, run
`git submodule update --init` (or clone with `--recurse-submodules`). Keep any local patches to it as `.patch` files in this repo.
