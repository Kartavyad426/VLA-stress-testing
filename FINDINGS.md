# Findings

Running record of things we've established that aren't in the original proposal.
Newest first. Each entry says what we know, how we know it, and what it changes.

---

## F3 — Why VLA LIBERO numbers don't reproduce: a catalogue of causes

**Date:** 2026-09-11 · **Status:** compiled from `docs/LANDSCAPE.md`; issue states
as of the survey. Several items marked where we verified them ourselves.

This is deliverable content, not just background. Nearly every benchmark in this
field has an open issue where published numbers do not reproduce — RoboCasa,
CALVIN, RLBench, ManiSkill, SimplerEnv, LIBERO; Meta-World's are unreproducible
*by construction* after a v1→v2 reward rewrite. **That reframes our reproduction
gate from pedantry into the thing that separates a measurement from a number**,
and it is worth saying in the readout.

### The distinction that decides which risks apply to us

| | Affects | Applies to us? |
|---|---|---|
| **Evaluation reproduction** — run a *published checkpoint*, get a different number | our Phase 2 | **YES — directly** |
| **Training reproduction** — retrain from scratch, get a different checkpoint | Phase 5 only | Only when we fine-tune |

Most of the angry GitHub issues conflate these. Sorting them matters: a
multi-GPU batch-size mismatch cannot affect us while we are only *evaluating*
someone else's weights.

### Causes affecting EVALUATION — these are our risks

**1. Simulator version changes physics.** MuJoCo 3.4.0's box-box collision fix
broke LIBERO's stored init states. `libero_spatial` task 5: SmolVLA **80% →
28%**, OpenVLA-OFT+GRPO 98% → 12–19%; π0.5 comparatively robust (90% → 86%).
A fresh install resolves to 3.8.1 — the broken side. LeRobot pins
`mujoco<3.9.0`, which guards API breaks and **not** behavioural ones.
[lerobot#4390](https://github.com/huggingface/lerobot/issues/4390);
[PR#4465](https://github.com/huggingface/lerobot/pull/4465) pinning
`>=3.2.7,<3.4.0` is **open, approved by one reviewer, unmerged**.
*We verified this ourselves and are now on 3.3.7 — see F2.*

**2. Batch size silently changes the evaluation *sample*, not just its speed.**
[lerobot#2850](https://github.com/huggingface/lerobot/issues/2850) — **closed,
genuinely fixed** — `--eval.batch_size` was capping how many unique initial
states got sampled. Long-horizon `libero_10` was hit hardest. This is the
subtle one: batch size reads like a throughput knob, and it was determining
*which episodes ran*. LeRobot's docs now advise setting `--eval.batch_size`
equal to episodes-per-task and keeping `--env.init_states=true` when comparing
two policies on the same episodes.

**3. Action-chunk mismatch.** `--policy.n_action_steps` differing from what the
checkpoint was trained with. Community-attributed as a likely cause of the
`libero_10`-specific gap. The published `smolvla_libero` config ships
`n_action_steps: 1` against `chunk_size: 50`, and LeRobot's own π0.5
reproduction passes `--policy.n_action_steps=10` explicitly on the command line
— an override that is **not in any config file**.
*We measured the throughput consequence ourselves: 3.95 vs 17.0 steps/s.*

**4. The dataset moved under people.** `HuggingFaceVLA/libero` has been
re-uploaded. In [#1369](https://github.com/huggingface/lerobot/issues/1369) a
maintainer redirected users to re-download a **silently modified** dataset.
LeRobot's docs now warn: *"Pin `--dataset.revision=<commit-sha>` when reporting
results."* **Make this policy on every run.** Our `PREDICTIONS.md` already pins
`a1aaacb7…`.

**5. Too few episodes.** 10 vs the recommended 50 per task can swing
Spatial/Long by **15–20 points**. At 10 eps/task, n=100 per suite gives ±6–9 pp
confidence intervals — wide enough to "fail to reproduce" from sampling noise
alone. LeRobot additionally recommends averaging **3 seeds**. *Decided 2026-09-11: 10/task, with
the ±5 pp gate replaced by a CI-overlap test — `PLAN.md` §5.3.*

**6. 8 GB VRAM contention — our exact hardware class.**
[lerobot#3098](https://github.com/huggingface/lerobot/issues/3098) was filed on
an **RTX 3070 Laptop (8 GB)**: the PyTorch CUDA context and MuJoCo's EGL
rendering context contend for VRAM inside one process and evaluation can fail
outright. [PR#3235](https://github.com/huggingface/lerobot/pull/3235) adds
`--eval.process_isolated=true` (subprocess with `MUJOCO_GL=osmesa`) — **open,
unmerged**. Workaround today: force CPU rendering via `MUJOCO_GL=osmesa` or
`glfw`, at a speed cost.

**7. Power state.** Not in any issue, and ours: a laptop on battery throttles
the GPU to a 15 W cap and ~180 MHz, ~25× slower, while `nvidia-smi` still
reports `utilization.gpu 100%`. Affects timing, not accuracy — but it caused a
peer session on this machine to log a run as *"system_failure, cause unknown"*.
See F2 and the provenance block in `run_repro.sh`.

### Causes affecting TRAINING reproduction — only bite us in Phase 5

**8. Multi-GPU recipes do not transpose to one GPU.** VLA-Adapter
[#46](https://github.com/OpenHelix-Team/VLA-Adapter/issues/46) reports 94% vs
~98%, traced to a **batch-size mismatch against the official multi-GPU recipe**.
Effective batch size is a function of per-device batch × devices × accumulation
steps, and published recipes usually state only the first. Reproducing a
multi-GPU recipe on one 8 GB card requires deriving the *effective* batch and
matching it with gradient accumulation — plus LR scaling, which most papers do
not specify.

This is instructive rather than damning: **it has an identified cause**, which
is exactly the class of environment bug our gate is designed to catch. Contrast
SmolVLA, where nine issues have produced no diagnosis.

**9. Unstated training config generally.** [#3287](https://github.com/huggingface/lerobot/issues/3287)
asks for learning rate, scheduler, batch size, GPU count and commit hash to
replicate SmolVLA on LIBERO. **No maintainer answer visible.** The requester's
numbers: 83.0 / 70.0 / 70.0 / 44.8 against ~90 / ~96 / ~92 / ~71.

### What the maintainers actually say — and don't

| Issue | State | Maintainer response |
|---|---|---|
| [#3264](https://github.com/huggingface/lerobot/issues/3264) | **open** since 2026-04-02 | **none** |
| [#2107](https://github.com/huggingface/lerobot/issues/2107) | **open** | none |
| [#2354](https://github.com/huggingface/lerobot/issues/2354) | **open**, assigned | none |
| [#3628](https://github.com/huggingface/lerobot/issues/3628) | **open** | none — A100, official checkpoint, official CLI |
| [#1316](https://github.com/huggingface/lerobot/issues/1316) | closed "completed" | 55 comments; users still reported 0.672 Spatial in Apr 2026. **Closed, not resolved.** |
| [#1369](https://github.com/huggingface/lerobot/issues/1369) | closed "completed" | redirected users to a re-uploaded dataset |
| [#2850](https://github.com/huggingface/lerobot/issues/2850) | closed | **genuinely fixed** (batch-size/init-states) |

**The signal that should change a decision:** LeRobot's LIBERO documentation has
a *"Reproducing published results"* section written **entirely about π0.5** —
`lerobot/pi05_libero_finetuned`, 97.5% average, reproducing OpenPI's 96.85%.
There is **no equivalent SmolVLA claim anywhere in the documentation.** With
nine open reproduction issues, that absence reads as deliberate.

### What we do about it

- **Subject under test: VLA-Adapter** (1B, MIT, ~2 GB, LoRA-able in 8 GB).
  Known gap has a diagnosis; SmolVLA's does not.
- **Deferred but available: π0.5 as a reproduction *anchor*.** See `PLAN.md` §5.2.
- Every cause above that we can control is now either pinned, recorded in
  provenance, or an explicit open decision.

---

## F2 — MuJoCo ≥3.4.0 silently breaks a LIBERO task. We were on the broken side.

**Date:** 2026-09-11 · **Status:** VERIFIED against the upstream issue, and FIXED here
**Source:** [lerobot#4390](https://github.com/huggingface/lerobot/issues/4390)

### What it is

MuJoCo 3.4.0 shipped a *correct* bugfix to box-box collision distance. LIBERO's
stored initial states depended on the old behaviour. In `libero_spatial` task 5
("pick up the black bowl on the ramekin and place it on the plate") the bowl no
longer settles onto the ramekin — it ends up tilted on the rim, roughly half
overhanging. The bowl's collision mesh is 40 box geoms and the ramekin's is 25,
so this task is maximally exposed to exactly that change.

### Measured impact (reporter, 50 episodes on task 5)

| Policy | healthy MuJoCo | broken MuJoCo |
|---|---|---|
| **SmolVLA 0.45B finetune** | **80% @ 3.2.7** | **28% @ 3.8.1** |
| OpenVLA-OFT GRPO | 98% @ 3.2.7 | 12% @ 3.8.1 |
| OpenVLA-OFT full SFT | 96% @ 3.2.7 | 52% @ 3.9.0 |
| π0.5 (LeRobot) | ~86–90% | ~86–90% (stable) |

Broken: 3.4.0–3.8.1. Healthy: 3.2.7, 3.3.0, 3.3.7.

### Why it matters more than a one-task bug

A fresh `pip install` of `lerobot[libero]` resolves to **3.8.1** — the broken
side. LeRobot pins `mujoco<3.9.0`, which guards against API breaks and **does
not guard against behavioural ones**. We installed today and got 3.8.1.

Left uncaught this is the worst class of error for this project: **a physics
change masquerading as a model failure.** Task 5 would have appeared in our
stress results as a genuine weakness, been clustered, attributed to whichever
perturbation happened to co-occur, and written into a Data Gap Manifest row
recommending that a client spend money collecting bowl-on-ramekin
demonstrations. Every downstream artifact would have looked sound.

It is also a concrete instance of the risk `PLAN.md` §7c discriminator #1 exists
to catch — *can a competent controller still solve this task?* — and evidence
that the discriminator needs to run per *task*, not per suite.

### Action taken

Pinned and reinstalled: **mujoco 3.8.1 → 3.3.7**. EGL rendering re-verified.

### Consequences for what we already ran

- The `n_action_steps` calibration was run under 3.8.1. Its **timing** numbers
  stand — physics does not affect throughput — but any success/failure outcome
  from it is suspect. It used `libero_spatial` task 0, not task 5, so probably
  unaffected; not worth re-running for timing alone.
- Partially relevant to [#3264](https://github.com/huggingface/lerobot/issues/3264):
  a 52 pp drop on 1 of 10 tasks is ~5 pp of suite-level Spatial, against that
  report's 27 pp shortfall. Contributory, not the whole explanation.
- **`mujoco` version now belongs in the reuse-relevant part of provenance**, not
  just the runtime record. It changes results, not merely timings.

---

## F1 — Policy transfer, and how much of the brittleness is behaviour cloning

**Date:** 2026-09-11 · **Status:** analysis, partly testable by our own Phase 2

### 1.1 What has to line up for a policy to transfer

Three things, in increasing order of subtlety — the last is the dangerous one.

| | What must match | Failure mode if it doesn't |
|---|---|---|
| **Action space** | shape *and* semantics: `Box(-1,1,(7,))` for LIBERO (6-D eef delta + gripper); delta vs absolute; eef vs joint | wrong shape = immediate error; wrong semantics = smooth, confident, wrong motion |
| **Observation space** | camera count/placement/resolution, state dim, **and the feature keys** | LeRobot enforces `.images.*` naming because the keys are baked into the normalisation layer |
| **Normalisation statistics** | per-dataset action mean/std (`unnorm_key`) | **confident, plausible, completely wrong motion** — indistinguishable from a weak model |

The third is why the proposal says environment mismatch "can manufacture apparent
model failures." A mis-set `unnorm_key` does not crash. It produces a robustness
curve.

### 1.2 Why transfer works at all

A VLA is two pieces with very different transfer properties:

- **the vision-language backbone** — essentially a VLM; transfers well, and is
  what Open X-Embodiment pretraining across 22 embodiments buys
- **the action head** — embodiment-specific; the part that must be adapted

Hence the standard recipe, and hence `smolvla_base` → `smolvla_libero` being a
fine-tune rather than a from-scratch train.

Roughly, easiest to hardest:

| Shift | Outcome |
|---|---|
| New instruction, same embodiment and scene | often zero-shot |
| New objects, same embodiment | partial |
| **Camera moved** | **often fails** — our prediction P1 |
| Different embodiment | new head at minimum |

**A viewpoint shift is a mild form of "different dataset."** The observation
distribution moves while the task does not. Our entire stress test is transfer
failure measured at small radius.

---

### 1.3 Is it all behaviour cloning? — No, and the distinction decides the fix

Worth separating carefully, because "it's a BC problem" and "it's a coverage
problem" imply different remediations and we sell remediations.

**Genuinely attributable to behaviour cloning:**

- **No corrective data.** Training on expert demonstrations means the policy has
  never seen how to recover from its *own* errors. It only knows the expert
  manifold. Step off it and there is no training signal at all — not degraded
  signal, none.
- **Compounding error / covariate shift.** The classic BC result (Ross &
  Bagnell, DAgger): a small per-step error moves the state distribution, which
  produces a larger error, and the cost compounds with horizon length rather
  than staying linear in it. This is a property of learning from a fixed
  distribution you then depart from.
- **No notion of progress.** No value function, no reward, no self-assessment.
  It cannot know it is failing, so it cannot change strategy.

**NOT attributable to BC — any paradigm trained on this data would share it:**

- **Viewpoint brittleness.** This is a *coverage* problem. Phase 0 measured zero
  camera-pose variation in the fine-tuning set. An RL agent trained in that same
  single-viewpoint environment would be just as brittle. Nothing about BC causes
  this; the data causes it.
- **Action-space and normalisation mismatch.** Engineering, not learning theory.

**Architectural, not BC:**

- **Memorylessness.** SmolVLA's config has `n_obs_steps: 1` — a single frame.
  The policy cannot remember that it already tried this. A recurrent or
  history-conditioned model could, trained identically by BC. Our toy's
  `retry_loop` failure is architecture and BC *together*: no memory to notice the
  repetition, no corrective data to know what else to do.

### 1.4 Evidence for the compounding-error account, from published numbers

SmolVLA's own published results:

| Suite | Published | Horizon |
|---|---|---|
| LIBERO-Object | ~96 | 280 steps |
| LIBERO-Goal | ~92 | 300 |
| LIBERO-Spatial | ~90 | 280 |
| **LIBERO-Long** | **~71** | **520** |

The long-horizon suite is ~20 points worse than the others, and it is the one
with roughly double the step cap. That is the shape compounding error predicts:
cost growing with horizon, not with task difficulty per se. It is consistent
across the other policies too — π0.5 scores 92.4–98.8 with Long also lowest.

Not proof. Long-horizon tasks are also multi-stage and semantically harder, and
the two explanations are confounded in this data. But it is the cheapest
available evidence and it points the right way. **A testable discriminator:**
if it is compounding error, failure probability should rise with *elapsed steps*
within an episode roughly independently of which subtask is in progress; if it
is task difficulty, failures should cluster at specific subtask boundaries. Our
phase localisation can separate these — it is exactly what stage localisation is
for, and it costs nothing beyond the runs we are already doing.

### 1.5 Why this matters commercially

It maps directly onto the fixability classes in `PLAN.md` §7b.1, and picking the
wrong one wastes the client's money:

| Cause | Correct remediation | Wrong remediation |
|---|---|---|
| Coverage (viewpoint) | augmentation or targeted collection | RL fine-tuning |
| BC / no recovery data | **corrective** demonstrations — failed attempt + recovery — or interactive collection (DAgger-shaped) | more clean expert demos, which is what a vendor defaults to selling |
| Architecture (memorylessness) | history-conditioned model; escalate to the model team | any amount of data |

The middle row is the sharpest: **more clean demonstrations cannot fix a
recovery gap**, because clean demonstrations are precisely what created it.
Recommending "collect 5,000 more demos" for a recovery failure is the most
expensive way to not fix the problem, and it is the default recommendation of
anyone whose business is collecting demonstrations. Being the party that says
"not this one — you need corrective data, and less of it" is the differentiated
position.

---

### 1.6 What the proposal says, and what we should add

It addresses transfer in three places, all as *warnings* rather than tasks:

- §7.1 — pin the action un-normalization config; mismatch "can manufacture
  apparent model failures"
- §7.1 — for GR00T, "match the policy action representation/control mode"
- §7.3 — policies differ in backbone, action representation and control mode, so
  cross-model work yields "**two robustness profiles rather than a universal
  model ranking**"

The third is the proposal at its best and should be elevated into the readout.
A client will ask "so which model is better?" and the honest answer is that we
cannot say — only which is more robust *along the axes we measured, in this
environment*. That refusal belongs in the document, not in the meeting.

**Three things to add:**

1. **A pre-flight equivalence gate.** The proposal says match the control mode
   but never says how you would know you had. Before any cross-policy
   comparison, each policy must reproduce **its own** published nominal
   baseline. If a second policy lands at 40% where published says 90%, the
   config is wrong and every robustness number downstream is noise about our
   setup, not a finding about the model. Make this a hard precondition for
   comparison, not a one-off Phase-1 check.
2. **Action config goes in `identity()`.** `control_mode`, `unnorm_key`, action
   space and checkpoint revision belong in the fingerprint, so a mismatch
   changes the policy id and misses the cache rather than silently mixing
   incompatible rollouts. Mechanism already built (see `ARCHITECTURE.md` §4.1);
   this is the structural version of the proposal's warning, and the difference
   between "remember to check" and "cannot get this wrong."
3. **Variable-dimension actions** — already covered by G1. The proposal's
   scale-out path contemplates additional embodiments but its rollout contract
   does not address the dimensionality change; ours does.
