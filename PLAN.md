# VLA Stress-Test — Implementation Plan

> **[UNSOURCED — do not cite]** the sim-to-real rank-correlation figure (Spearman 0.4–0.7) and the claim that *severity ordering transfers worst* were carried forward from an early survey and **no primary source has been located** (flagged 2026-09-15). The design decisions they motivated stand on their own reasoning; the numbers must not appear in a readout until sourced.

**Status:** draft v1 · 2026-09-10
**Companion to:** `VLA Scenario Testing.md` (the proposal)
**This document:** how we actually build it, on the hardware we actually have.

---

## 0. Decisions taken

| Decision | Choice | Consequence |
|---|---|---|
| Compute | Laptop only (RTX PRO 1000, **8 GB VRAM**) | OpenVLA-OFT 7B is out of reach at full precision. Headline policy changes. |
| Diagnosis | Hybrid — deterministic detectors primary, LLM judge for semantic residue only | Measurement path stays reproducible; LLM never overrides a detector. |
| First step | **Supply-side data analysis** — measure the training distribution before designing experiments | Perturbation axes are chosen from evidence, not from the paper's priors. |
| Prototype | Full L0–L4 harness against a broken scripted policy | Design is validated against a known answer before any model is involved. |
| **Subject policy** *(D1, 2026-09-11)* | **VLA-Adapter** (1B, MIT, ~2 GB, LoRA-able) — **not** SmolVLA | SmolVLA has nine unresolved LIBERO reproduction issues and no maintainer diagnosis. VLA-Adapter's known gap has an identified cause. See `FINDINGS.md` F3. |
| **Reproduction anchor** *(D1)* | **π0.5 — documented, deferred** | See §5.2. Available if the subject's gate cannot be made to pass. |
| **Episodes per task** *(D2, decided 2026-09-11)* | **10**, per LeRobot docs | n=100/suite. **Consequence: the ±5 pp gate is replaced by a CI-overlap test** — see §5.3. Escalate to 50 only where a result is ambiguous or reported. |
| **Rendering** *(D3)* | EGL, with `MUJOCO_GL=osmesa` fallback | VRAM contention on 8 GB — see §5.4. |
| **Second benchmark** *(D4)* | **Deferred** — RoboCasa after the LIBERO gate resolves | Cheapest generality claim available, but a distraction before Phase 2 reproduces anything. |
| **Sampling split** *(D5)* | 40% uniform / 60% adaptive | §5.1. Revisit once the surrogate's convergence behaviour is known. |

---

## 0b. Positioning — what is actually ours *(revised 2026-09-11 after `docs/LANDSCAPE.md`)*

A literature survey (238 sources, `docs/LANDSCAPE.md`) found the core loop —
perturb, cluster, recommend — is **substantially prior art**. Three things
survive as genuinely ours:

1. **The Data Gap Manifest as a costed artifact**, with severity, fixability,
   discriminators and validation status per row. The ideas exist; the artifact
   does not.
2. **Counterfactual attribution of policy failures to environment factors.**
   Formalised for LLM agents (CAR, [arXiv:2606.08275](https://arxiv.org/abs/2606.08275)),
   essentially unexploited in robot policy evaluation — and our
   randomised-treatment setting is *methodologically stronger* than theirs.
   **We currently undersell this.**
3. **Failure-cost weighting** (§7b.3). No perturbation benchmark found weights
   failures by consequence.

**Honest positioning is integration and delivery, not research novelty.** Say
this in the readout rather than letting a reviewer discover it. Specifically,
do not pitch as novel: perturbation generation (LIBERO-Plus, COLOSSEUM, VLATest),
failure clustering (2506.06570, RoboFAC, AHA), runtime monitoring (Sentinel,
FIPER), or demonstration scoring (CUPID, Demo-SCORE, Re-Mix).

Phase 5 — retrain and re-measure — is rated "not redundant, and rare. Keep this."
It remains the part most likely to be cut and the part that most distinguishes us.

---

## 1. The constraint, and why it is an opportunity

8 GB of VRAM cannot hold OpenVLA-OFT (7B ≈ 15 GB at bf16). Two options: 4-bit quantize it, or use a smaller policy.

**Quantizing is the wrong move.** The proposal's Phase-1 acceptance gate is reproducing published LIBERO numbers within ±5 pp. A quantized checkpoint has no published number to reproduce, so the gate — the project's only defence against "your environment was broken, not the model" — evaporates.

**Going small is the right move, and it buys something the 7B path never had.**

The hardest requirement in the proposal is §7.3 step 5: fine-tune on remediation data and re-measure. That is the step that converts the Data Gap Manifest from opinion into evidence, and it is the step most likely to be cut for cost. Fine-tuning a 7B model is a serious GPU job. **Fine-tuning a 450M model with LoRA fits in 8 GB.**

So the trade is: give up the prestige of the headline model, gain the ability to close the loop. For a project whose entire thesis is *the loop*, that is the correct trade. A complete cycle on SmolVLA is a stronger artifact than a partial cycle on OpenVLA.

### Candidate policies, sized to the hardware

| Policy | Params | bf16 VRAM | Inference | LoRA fine-tune | Notes |
|---|---|---|---|---|---|
| Scripted oracle | — | 0 | trivial | n/a | **Test fixture.** Known planted bug = ground truth for the miner. |
| Octo-small | ~27 M | <1 GB | comfortable | yes | Diffusion head. Fast, weak, good smoke test. |
| **SmolVLA** | ~450 M | ~1 GB | comfortable | **yes** | *Primary candidate.* LeRobot-native. |
| π0 | ~3 B | ~6 GB | tight | marginal | Flow-matching head — architecturally distinct, good 2nd policy. |
| OpenVLA-OFT | 7 B | ~15 GB | 4-bit only | no | Deferred. See §8. |

> **RESOLVED 2026-09-11 — see `MODELS_AND_COMPUTE.md`.** A SmolVLA LIBERO checkpoint exists (`HuggingFaceVLA/smolvla_libero`), and LeRobot ships a first-class LIBERO-plus CLI. Phase 2 is unblocked and simulator integration is much smaller than budgeted.
>
> **The largest risk is now reproducibility, not availability.** [lerobot#3264](https://github.com/huggingface/lerobot/issues/3264) — open, unanswered — reports being unable to reproduce published SmolVLA LIBERO numbers with the official checkpoints (obtained 73.25% overall). Our Phase-2 reproduction gate is the project's only defence against "your environment was broken, not the model", and this attacks it directly. Budget a week to settle it before committing to SmolVLA.

---

## 2. Architecture

Five layers. The model is a plugin at L1. Four of the five need no GPU.

```
L4  manifest/      Data Gap Manifest generation
L3  mining/        phase detection · classification · counterfactuals · clustering
L2  envs/          ToyReachEnv → LIBERO → LIBERO-plus      (Env protocol)
L1  policies/      ScriptedPolicy → SmolVLA → π0           (Policy protocol)
L0  schema.py      canonical Rollout contract + trace store
```

**The contract is the product.** Everything above L0 reads and writes the same `Rollout`. A client's real question is "evaluate *my* model", so swapping the policy must be a one-file change. Same for the simulator.

### 2.1 Three provenance buckets, and a promotion policy *(revised after DG-4)*

The earlier rule — *changes the ANSWER ⇒ keyed; changes only TIMING ⇒ reported* —
**is not decidable.** `torch` and `numpy` change answers too: different BLAS,
reduction order or kernel selection ⇒ different floats ⇒ different contacts.
`ARCHITECTURE.md` §5.1 says exactly this. Applied honestly the rule keys torch
and numpy, every pip upgrade invalidates the store, and we are back at the
git-SHA position we rejected.

**The real distinction is epistemic, not causal: KNOWN-semantic vs
POSSIBLY-semantic.**

| Bucket | Holds | In the cache key? |
|---|---|---|
| `identity()` | the env's / policy's **design**: constructor kwargs, thresholds affecting the goal predicate, checkpoint revision, control mode, action space | **yes** |
| `semantic_runtime` | external components with a **documented, specific** semantic effect — `mujoco`, `MUJOCO_GL`. Enumerated per adapter. | **yes** |
| `runtime` | everything else — `torch`, `numpy`, `python`, platform | no — compared and **reported** |

`semantic_runtime` exists because a simulator build is not part of an env's task
*design*: `ToyReachEnv.identity()` declaring a MuJoCo version it never loads
would be incoherent.

**Promotion policy.** Learning that a `runtime` field moves results *promotes*
it to `semantic_runtime`. That is a deliberate, rare, store-invalidating event,
recorded like a schema bump. MuJoCo is there because F2 documented a specific
change moving one task 80% → 28%. `torch` stays in `runtime` because it *might*,
and reporting a mismatch is the proportionate response to "might."

### Scalability guarantees — the "no fundamental flaw in month 2" list

These are the design choices that exist specifically to prevent expensive rework. Each is cheap now and very expensive later.

| # | Guarantee | How | What it prevents |
|---|---|---|---|
| G1 | **Actions are variable-dimension** | `action: list[float]` + `action_dims: list[str]` metadata. Never a fixed 3- or 7-tuple. | Toy uses 3 DoF, LIBERO uses 7, bimanual uses 14. Hardcoding 3 means rewriting every consumer. |
| G2 | **Observations carry optional image refs, not pixels** | `image_refs: dict[str, str]` pointing at files on disk. | Toy has no images; LIBERO has two camera streams at 224×224×N steps. Inlining pixels makes traces unloadable. |
| G3 | **Schema is versioned** | `schema_version` on every rollout; a loader that refuses unknown majors. | Silent format drift across a 3-month project. |
| G4 | **Perturbations use LIBERO-plus factor names from day one** | `PerturbationSpec` keys mirror the benchmark taxonomy even in the toy. | Renaming every axis when the real benchmark arrives. |
| G5 | **Reproducibility, measured not assumed** | Seed recorded per rollout. Toy is bit-exact; a real VLA is not (GPU nondeterminism, stochastic action heads, contact chaos). Probes compare *distributions* over N seeds, and `reproducibility_floor()` measures the noise a delta must beat. | Counterfactual attribution is impossible without this. Assuming bit-exactness and discovering otherwise mid-project would invalidate every probe already run. |
| G6 | **Env/Policy are Protocols, not base classes** | `typing.Protocol` structural typing. | Adapters that must inherit from our tree — third-party policies never will. |
| G7 | **Traces are append-only JSONL, one dir per run** | `runs/<run_id>/rollouts.jsonl` + `meta.json`. Parquet is a projection, not the source. | Needing a database on day one; losing the ability to grep. |
| G8 | **Detectors declare their inputs** | Each detector states which state keys it needs; missing keys ⇒ skip with a recorded reason, never a crash. | Toy state ≠ LIBERO state. Hard-coupling the miner to one env. |
| G9 | **The miner never sees the env** | It consumes `Rollout` objects only. | The miner silently becoming LIBERO-specific. |
| G10 | **Sweeps are resumable** | Completed cells are skipped on re-run by content hash. | A crash 80% through an overnight sweep. |

### Assumptions that could invalidate the work — and their early, cheap tests

The user's stated fear. Each row is a way this project could be worthless, with a test that runs in week 1–2 rather than month 2.

| Assumption | If false | Cheap early test | When |
|---|---|---|---|
| A phase can be localized from simulator state | The trace layer's value collapses to pass/fail | Build detectors on the toy where phases are known; then check LIBERO exposes contact + EE + gripper signals | Wk 1–2 |
| Counterfactual probes isolate a trigger | Attribution rests on clustering alone — much weaker claim | Run probes on the scripted oracle where the trigger is *planted*. Miner must recover it. | **Wk 1** |
| Rollouts are reproducible enough to probe | Attribution deltas are noise; every causal claim is void | `reproducibility_floor()` on the real policy — run one cell twice, compare. Do this **before** any stress campaign. | **Wk 3, day 1** |
| A behaviour gap implies a data gap | The manifest is a hypothesis generator, not a finding | Cross-check against the supply-side histogram (Phase 0). Two independent lines or one weak one. | Wk 1–3 |
| Targeted data closes the gap | The loop does not close; deliverable is diagnosis only | LoRA fine-tune on the smallest model, top manifest row only | Wk 6–8 |
| ~~A LIBERO-capable small policy exists~~ | ~~Must train our own~~ | **RESOLVED — `HuggingFaceVLA/smolvla_libero` exists** | done 2026-09-11 |
| Published checkpoint numbers are reproducible | The Phase-2 gate is unavailable; every later claim rests on an unverified environment | Run the four suites, compare to published. See `MODELS_AND_COMPUTE.md` §4. | **Wk 1** |
| The taxonomy is human-reproducible | Cluster labels are not defensible to a client | Double-label 30 traces, compute κ | Wk 5 |

**Kill criteria.** If the miner cannot recover a *planted* trigger on the scripted oracle (week 1), the mining design is wrong and we stop and redesign rather than proceed. If κ < 0.5 on the double-labelled subset, the taxonomy is not a product and must be simplified before any client sees it.

---

## 3. Phase 0 — Supply-side analysis *(first, per decision)*

**Rationale.** The proposal picks perturbation axes from LIBERO-Plus's published sensitivity results. That is borrowed evidence. Measuring the training distribution ourselves gives us our own, and lets us *predict* where the boundary falls before we measure it — a falsifiable claim rather than a post-hoc story.

LIBERO demos are public HDF5, single-digit GB, and need no GPU.

**Tasks**
1. Pull LIBERO demonstration datasets (Spatial, Object, Goal, Long — 10 tasks × 50 demos each).
2. Histogram the coverage: camera extrinsics, initial EE pose, object placements, trajectory length, gripper-event timing, instruction phrasing variety.
3. Identify low-density regions — the axes where demonstrations cluster tightest.
4. **Write down predictions before running anything.** "Yaw coverage is ±X°, therefore we predict the boundary near Y°."
5. Verify which small policies have LIBERO checkpoints (the §1 open question).

**Deliverable** — `phase0_coverage_report.md` + plots, and a **written, timestamped prediction** of which axes are weak and roughly where.

**Effort** — 3–5 days. No GPU.

**Why this ordering is better:** it converts the stress matrix from an exhaustive sweep into a targeted test of a stated hypothesis. Cheaper to run and far more convincing to present.

---

## 4. Phase 1 — Harness on the scripted oracle *(this session)*

Build L0–L4 end-to-end against a policy whose bug we planted.

**The core methodological trick.** Validating a failure miner normally has no ground truth — when it says "visual grounding, triggered by yaw", nothing independently confirms that. With a scripted policy we *inserted* the fault, so we know the answer. If the miner recovers it, the miner works. If not, we learn that now, for free.

Faults to plant (each maps to a real taxonomy family):

| Planted fault | Should be diagnosed as | Trigger the miner must recover |
|---|---|---|
| Camera-frame misalignment | Visual grounding | camera yaw |
| Grasp tolerance too tight | Manipulation / control | initial-state offset |
| Nearest-object selection | Spatial reasoning | distractor count |
| No re-plan after miss | Recovery | any grasp failure |

**Acceptance:** miner recovers ≥ 3 of 4 planted triggers.

**Result (prototype, `experiments/oracle_test.py`): 3/4 — GATE PASSED.** Control passed: a clean policy under the same perturbations produces zero manifest rows.

Two real defects were caught by this gate, both of which would have been expensive to find later:
- **Resumable sweeps silently reported 0% success.** Cached rollouts skipped the compute *and* the result, so every resumed cell read as a total failure. This would have corrupted every overnight run (G10).
- **The classifier inferred object-selection failures from error magnitude.** Wrong: a large error equally indicates a grounding failure with no distractor present. Selection needs its own signal — proximity to a distractor. A second rule was added distinguishing "searched and missed" (manipulation) from "repeated itself" (recovery) via grasp-attempt spread; without it every retried failure was mislabelled `recovery`, masking the manipulation fault underneath.

**Known fixture limitation.** `no_recovery` cannot be isolated in the toy. Perception noise re-rolls every step, so repeating an identical grasp eventually succeeds — repetition is a *winning* strategy here, which it is not in a real system where the world is deterministic given the same state. Recovery is also inherently second-order: it cannot fail alone, only conditional on a prior failure. **Recovery diagnosis is therefore unvalidated and must be re-tested against LIBERO before any recovery row appears in a client manifest.**

**Effort** — 1 session for the skeleton, ~1 week to harden.

---

## 5. Phase 2 — Real policy, targeted experiments

Swap the scripted policy for a real small VLA; swap the toy env for LIBERO. Run the perturbations **Phase 0 predicted**, not the full grid.

### 5.1 Two-arm sampling — replaces the grid sweep *(revised 2026-09-11)*

The original "~20 episodes/cell screening, ~50 near the boundary" grid is a
hand-rolled version of a formalised method, and it is **redundant and worse**:
**FATE-VLA** ([arXiv:2606.02307](https://arxiv.org/abs/2606.02307)) fits a
surrogate over scene parameters and steers sampling toward failure-prone
regions for **+29.7% more failures at equal budget**; **Liao et al.**
([arXiv:2607.14439](https://arxiv.org/abs/2607.14439)) report 20–40% trial
savings on real hardware over 2,331 evaluations.

**Run two arms, and never mix them:**

| Arm | Sampling | Feeds | Budget |
|---|---|---|---|
| **Uniform** | fixed grid, unbiased | `severity` (frequency-weighted), cluster sizes | ~40% |
| **Adaptive** | surrogate-guided (logistic/GP over magnitude → success) | `boundary` — with a posterior interval, not a Wilson interval on whichever cell straddled the transition | ~60% |

**The trap the source papers do not have to care about and we do:** active
sampling optimises for *finding* failures, so it biases the discovered failure
population. Anything frequency-weighted must therefore be computed over the
uniform arm only.

**The arm is a property of the REQUEST, not of the episode** *(DG-2)*. Tagging
the `Rollout` cannot work: `rollout_id` hashes (identity, seed, spec) and
correctly excludes the arm, since the physics is identical — so the store holds
one rollout per cell tagged with whichever arm asked *first*. The uniform arm
then loses cells to cache hits tagged `adaptive`, and *which* cells it loses
depends on adaptive search order. The uniform sample stops being uniform: the
exact corruption this split exists to prevent, introduced by the mechanism meant
to prevent it. Hashing the arm is worse — it duplicates identical physics,
doubles the campaign, and breaks the seed pairing §9.2 needs.

**Design:** `runs/<id>/arms.jsonl` maps `(arm, rollout_id)` many-to-many.
Uniform frequency is computed over the cells the uniform arm **asked for**, each
resolved through the cache. One rollout legitimately serves both arms.
`Rollout` gains no field, so this needs no schema major bump.

The surrogate is a few hundred lines of CPU code and additionally supplies a
principled stopping rule, which the grid does not have.

Gate: reproduce the chosen checkpoint's published task success within a stated tolerance. If there is no published number (likely for SmolVLA on LIBERO), establish and document our own baseline over 3 seeds and treat *that* as the reference — and say plainly in the readout that this is a weaker gate than the proposal assumed.

**Effort** — 2–3 weeks, dominated by MuJoCo/EGL setup pain, not by science.

> **PIN `mujoco<3.4.0`.** Non-negotiable. MuJoCo 3.4.0's box-box collision fix
> breaks LIBERO's stored init states: SmolVLA drops **80% → 28%** on
> `libero_spatial` task 5, OpenVLA-OFT+GRPO 98% → 12%. A fresh install resolves
> to 3.8.1 — the broken side. See `FINDINGS.md` F2. Uncaught, this is a physics
> change masquerading as a model failure, straight into a manifest row.

> **Policy risk is worse than §1 assumed.** The survey found **nine** open or
> unresolved SmolVLA/LIBERO reproduction issues, and LeRobot's "Reproducing
> published results" section is written entirely about π0.5 with no SmolVLA
> equivalent. Fallbacks that fit 8 GB with LIBERO checkpoints: **VLA-Adapter
> (1B, MIT, 97.3%)** and **MiniVLA (1B, MIT, LoRA-native)**. Budget one day to
> test a fallback before committing the campaign. `lerobot/smolvla_robocasa`
> also exists — same policy, same stack, the cheapest possible second-benchmark
> generality claim.

---

### 5.2 The reproduction anchor — deferred, not discarded

**The gate's job is to exonerate *our environment*, not to grade a policy.** Once
that is clear, it does not have to be the same policy we study.

`lerobot/pi05_libero_finetuned` is **the only LIBERO checkpoint LeRobot
maintainers confirm reproduces** — 97.5% average against OpenPI's own 96.85%.
A failure to reproduce *it* is unambiguously our fault. That makes it an ideal
anchor: run it once, and every subsequent number from this harness is measured
against a setup already cleared of suspicion.

**STATUS: DEFERRED BY DECISION, 2026-09-15.** The user chose to proceed to the
VLA-Adapter gate and pay for the anchor only if that result lands ambiguous.
This is a decision, not an oversight, and the residual cost is stated below.

**The anchor is UNAVAILABLE on this hardware.** π0.5's weights alone exceed
8 GB: PyTorch held 7.01 GiB of 7.53 before any render context existed, and a
weights-only load with no simulator running reproduced the OOM. This is *not*
the lerobot#3098 CUDA/EGL contention we assumed. See `FINDINGS.md` F4.

**Taking it later costs RENTED COMPUTE, not a day of local work.** Any framing
of the anchor as a cheap in-house fallback is now false. It needs a ≥16 GB card
— a free-tier T4 or a rented hour — and it is inference-only, one-off, ≤400
episodes.

**The residual this deferral buys, stated so it is not discovered later:** if
the VLA-Adapter gate lands ambiguous, **we cannot attribute the miss between our
setup and the checkpoint without spending money.** At n=100 a 5 pp break goes
unflagged 66% of the time (§5.3), so an ambiguous landing is likely rather than
exceptional. **This belongs in the readout**, not in a footnote.

**A rented anchor is stronger than "we couldn't run it locally" implies.** It
still exonerates almost everything that matters — harness code, the LIBERO
adapter, detector configuration, init states, the MuJoCo pin, control mode,
un-normalisation, dataset revision. Only genuinely machine-specific effects stay
unexonerated, which is a far smaller residue than the original ambiguity. The
`runtime` fingerprint bucket exists precisely to make that residue visible.

**(superseded) STATUS: undecided, blocked on a smoke test (DG-5).** The review argues the
deferral contradicts its own justification: if the gate exists to clear the
*environment*, running it on a policy whose own reproduction is uncertain
**cannot clear anything** — a miss is ambiguous between our setup and the
checkpoint. VLA-Adapter was chosen *because* its gap has an identified cause,
which is another way of saying it is known not to reproduce cleanly. DG-1
compounds it: at n=100 the screen is a coin flip at 5 pp, so "lands far from
published" is a *likely* state, not an edge case.

**Blocking action:** π0.5 is 6–7 GB and lerobot#3098 documents CUDA/EGL VRAM
contention on this exact hardware class. **Smoke-test whether π0.5 runs here at
all before relying on it.** If it does not, the anchor is not deferred — it does
not exist, and this section is promising a fallback that cannot be taken. A
contingency nobody has smoke-tested is not a contingency.

**Take this option if** the subject's baseline lands far from published and we
cannot tell whether the cause is ours or the checkpoint's. It converts an
ambiguous result into a decidable one for roughly a day of work, inference-only,
no fine-tuning required.

### 5.3 Episodes per task — DECIDED (D2): 10/task

The guidance genuinely conflicts:

| Source | Episodes/task | Implication |
|---|---|---|
| LeRobot LIBERO docs | 10 | "matches the protocol used for published results"; 400 episodes total |
| Original proposal §7.3 / OpenVLA-OFT canonical | **50** | 500 trials per suite, averaged over 3 seeds |
| `docs/LANDSCAPE.md` §3.1 | — | 10 vs 50 **can swing Spatial/Long by 15–20 points** |

At 10 eps/task, n=100 per suite gives **±6–9 pp** confidence intervals — wide
enough that a genuine reproduction could read as a failure, and a genuine
failure as success. The reproduction gate is ±5 pp. **A gate narrower than its
own measurement error is not a gate.**

Cost, from our calibration (~17 steps/s): 10 eps/task ≈ 1–2 h; 50 eps/task ≈
5–8 h; 50 eps × 3 seeds ≈ 15–24 h. All are feasible on this hardware; none are
free of the GPU-sharing arrangement with the peer session.

### DECIDED 2026-09-11: 10 episodes/task, and the gate changes shape

We take LeRobot's protocol: **10 episodes/task, 4 suites, 400 episodes**, which
is what published numbers were produced under and therefore the like-for-like
comparison.

**But the ±5 pp gate cannot survive that n, so it is replaced.** At n=100 per
suite the 95% Wilson interval is roughly ±6–9 pp. A ±5 pp tolerance is narrower
than the measurement error — it would reject genuine reproductions and accept
genuine failures, at a rate we could not even quantify.

**10/task is a SCREEN. 50/task is the GATE.** *(revised after DG-1)*

The CI-overlap test at n=100 has almost no power at the effect size we care
about. Measured operating characteristic, published = 87%:

| True success rate | P(test says "consistent") @ n=100 | @ n=500 |
|---|---|---|
| 0.87 — correct environment | 92.4% | 94.7% |
| **0.82 — a 5 pp break** | **56.2%** (66.0% incl. escalate) | **11.0%** |
| 0.80 — a 7 pp break | 36.2% | 1.0% |
| 0.73 — the known failed-repro value | 2.4% | 0.0% |

**A 5 pp break — exactly what the original ±5 pp tolerance existed to catch —
passes at n=100 **56.2%** of the time as "consistent", and is **not flagged as a
non-reproduction 66.0%** of the time. The gap between those two figures is the
single outcome k=81, which escalates rather than passing; it carries ~10% of the
mass at p=0.82. **66.0% is the operationally important number** — two times in
three, a 5 pp break is not caught.** The CI-overlap framing did not make
the gate more honest than the tolerance; it made it *less sensitive while
reading as more rigorous*. "Consistent with published" is a true statement about
a test with no power, and it is the sentence a reader will quote.

**Therefore:**

- **10/task is a screen.** Its job is orienting and deciding where to spend
  effort. **Publish the table above beside every screen result.** A screen pass
  means *"screen passed, gate not yet run"* — **never** "consistent with
  published".
- **50/task is the gate**, and escalation is **unconditional for anything
  reported**, not only for ambiguous outcomes.
- Row 3 ("too weak") fires for exactly one outcome at n=100 (k=81) — effectively
  unreachable. Define it as a property of the measurement — *CI half-width
  exceeds half the deviation we care about* — not by reference to the 73%
  anecdote, which is a single report being used as a fixed hypothesis.
- **Apply multiple-comparison correction.** Four suites at 5% each ⇒ **18.5%**
  chance of at least one spurious non-reproduction. §9.2 mandates correction;
  this section must actually apply it.

**The test itself, used at either n:**

| Outcome | Verdict | What we do |
|---|---|---|
| Published value **inside** our 95% CI | **Consistent with published.** Not "reproduced" — we cannot distinguish. | Proceed. State the interval, never a point estimate. |
| Published **outside** the CI | **Genuine non-reproduction.** A real finding. | Investigate via `FINDINGS.md` F3 causes; escalate that suite to 50 eps/task before reporting. |
| CI so wide it contains both published and the failed-repro value (~73%) | **Measurement too weak to say anything.** | Escalate to 50 eps/task. Do not report either way. |

The third row is the one that matters and the reason this framing is better than
a tolerance: at n=100 the interval can be wide enough to contain *both*
hypotheses at once, and a point-tolerance gate would silently pick one.

**Escalation policy:** 50 eps/task (≈5–8 h) for any suite that lands in row 2 or
3, and for anything that goes in a client-facing readout. 10 is for orienting
and for deciding where to spend the 50.

**This means our first full run is an orienting run, not a reported result.**
Label it as such in the output so it cannot be quoted as a baseline later.

### 5.4 Rendering backend (D3)

**EGL, with `MUJOCO_GL=osmesa` as the documented fallback.**

[lerobot#3098](https://github.com/huggingface/lerobot/issues/3098) was filed on
an **RTX 3070 Laptop, 8 GB — our exact hardware class**: the PyTorch CUDA
context and MuJoCo's EGL rendering context contend for VRAM in one process and
evaluation can fail outright.
[PR#3235](https://github.com/huggingface/lerobot/pull/3235) adds
`--eval.process_isolated=true` — open, unmerged.

EGL is verified working here at ~2.8 GB peak. It is the faster path and we keep
it. If a larger policy (π0.5, §5.2) or a bigger batch triggers OOM, switch to
`osmesa` — CPU rendering, slower, no VRAM contention. **The backend goes in
`semantic_runtime`** (§2.1), since it can change rendered observations and
therefore results.

---

## 6. Phase 3 — Mining · Phase 4 — Manifest · Phase 5 — Validate

**Mining** runs the L3 stack over real traces: phase localization, deterministic classification, LLM adjudication of the residue, clustering, counterfactual attribution. Double-label 30 traces and report κ.

> **κ alone is too weak a gate.** It measures whether two humans agree, not
> whether the taxonomy carves the failure space at its joints. Add a second
> check: run an unsupervised clustering of the failure traces and compare the
> discovered partition against our seven families. Families that no discovered
> cluster respects are probably ours, not the data's.
>
> **`language_grounding` is OUT OF SCOPE for this campaign** *(DG-11)*.
> `instruction_variant` is in `LIBERO_PLUS_FACTORS`, but **no env implements the
> axis and no detector in `classify()` can produce the family** — so it cannot
> fire. The earlier "reporting rule" could not distinguish *"the policy is
> insensitive to language"* from *"we never tested language"*, which are very
> different sentences to put in front of a client.
>
> State it as out of scope in the taxonomy, and compute **κ and the E5
> discovered-taxonomy comparison over 6 families, not 7.**

**Manifest** (see §9 for the schema) — every row carries its evidence.

**Validate** — LoRA fine-tune on remediation data for the single top row; re-run the frozen regression set; report the delta. **On this hardware this is achievable, which is the whole argument of §1.** Negative and neutral results are reportable.

### 6.1 Three-arm remediation comparison — the best novel result available *(added 2026-09-11)*

Rather than validating one remediation, compare **three** on one frozen
regression set and report **cost per point of recovered success** — at **one
budget**, not three *(rescoped after DG-9)*:

| Arm | What | Cost proxy |
|---|---|---|
| A | targeted data collection | demos × collection cost |
| B | re-rendered augmentation of existing demos | GPU-hours |
| C | frozen-policy wrapper / tiny adapter (e.g. FTM, ~4K params) | GPU-hours, no collection |

Nothing in the survey does this. **It converts `fixability` from a judgement
call into a measurement**, and it is the most defensible novel result available
to us. It also directly answers the buyer's question in §7c discriminator 7.

**Scope, corrected.** Three arms × discriminator 6's three budgets would be
**9 fine-tunes plus 9 closed-loop re-evaluations** — several times the whole
Phase-5 budget. Instead:

- **Three arms at ONE budget** → the fixability comparison.
- **Data-response curve on arm A only**, for the top manifest row → discriminator 6.
- **Five fine-tunes, not nine.** Both results preserved.

Two framing corrections that hold regardless of cost:

1. **A vs B is not the contrast that matters.** Re-rendered augmentation *is*
   targeted data, generated differently — A-vs-B answers "collected vs
   synthesised". The contrast that converts `fixability` into a measurement is
   **(A or B) vs C**: data versus not-data.
2. **Arm C has no budget axis**, so "cost per point recovered" compares a curve
   against a point. Legitimate — but it must be **stated that way**, or the
   headline number reads as comparable across arms when it is not.

**Report regression on NON-targeted cells alongside improvement on targeted
ones.** An industry report *[unverified — vendor blog]* claims 93 targeted
"prescription" demonstrations dropped closed-loop success **73% → 43% while
offline loss showed no change at all**. If even directionally true: targeted
data can actively harm, and offline metrics are blind to it. Our closed-loop
re-measurement is precisely the procedure that catches this — which makes
Phase 5 non-negotiable rather than desirable.

---

## 7. Diagnosis design (hybrid)

Three tiers. Strictly ordered — a later tier never overrides an earlier one.

**Tier 1 — deterministic detectors.** Reproducible, auditable, cheap. From simulator state: EE-to-target distance, gripper aperture and contact flags, object pose delta, velocity, BDDL goal predicate. They answer *where* and *what*:
- phase segmentation (approach / pre-grasp / grasp / transport / place / release)
- first meaningful divergence from the nominal envelope
- point of no return
- terminal behaviour (timeout, loop, premature stop)

**Tier 2 — counterfactual probes.** Also deterministic. Answers *which knob*, by re-running the same seed reverting one dimension at a time. **This is the attribution mechanism.** It depends entirely on G5 (determinism).

**Tier 3 — LLM judge, narrow scope.** Only for the semantic residue Tier 1 cannot express: "is this a recovery attempt or the same action repeated?", "did it approach the wrong object or the right object badly?". Constrained to a fixed label set, always records its reasoning, always flagged as LLM-sourced in the trace, and **never used to compute a headline number.** Sampled for human audit.

**Claude Code skill** — for the human-review workflow, not classification: package ambiguous traces for a reviewer, capture adjudications, compute inter-annotator agreement, promote gold cases to the regression set. Build once Tier 1 is producing real ambiguity worth reviewing (Phase 3), not before.

> **Why not LLM-primary:** the manifest's claims are quantitative ("34.0% ± 4.4 at L3"). A non-reproducible component in the measurement path means those numbers cannot be regenerated, which is precisely the credibility the project is selling.

---

## 7b. Beyond "collect more demonstrations"

The proposal's remediation is almost always "more demonstrations in the failing region". On its own that is thin, and a sophisticated buyer will notice that a vendor with a collection business always recommends collection. Four additions, all supported by traces we already capture.

### 7b.1 Fixability class — is this even a data gap?

Every manifest row gets a `fixability` field, because "more data" is only one of four answers:

| Class | Meaning | Typical remediation |
|---|---|---|
| `augmentation` | Existing demos re-rendered under the perturbation would suffice | Cheap. Often 100× cheaper than collection; frequently sufficient for viewpoint. |
| `data` | The regime is genuinely unrepresented | Collect. |
| `architecture` | The policy lacks the invariance; more data teaches memorisation, not generalisation | Representation change (camera-relative action space, calibration as input). Escalate to the model team. |
| `operational` | Cheaper to constrain the environment than to fix the model | Ship an envelope spec (7b.2). |

**Recommending the cheapest sufficient intervention is the consulting value.** See §7c for how the class is actually determined.

### 7b.2 Operating envelope specification

A second deliverable that is not a data plan at all: *"certified within ±10° camera yaw and ≤2 distractors; outside that, success falls below 50%."* Immediately actionable, costs nothing to act on, and for a buyer deciding whether to deploy **today** it is worth more than a retraining roadmap.

### 7b.3 Failure cost, not just failure rate

The proposal counts failures. Buyers care what a failure *does*. A timeout with the gripper open is free; a dropped part stops a line; a collision is a safety event. Same success rate, very different risk. Terminal state is already in the trace — classify it by consequence and weight severity by cost, not frequency alone. **This is the largest omission in the proposal as written.**

### 7b.4 Runtime failure monitor *(scope separately)*

Having mapped which conditions predict failure, train a detector that flags **at inference time**: "current conditions resemble the region where this policy is unreliable." The robot abstains or asks for help instead of confidently closing on empty air.

This turns a post-hoc report into a live safety component, needs no additional data collection, and keeps earning after the report is filed. **It is a second product, not a phase of this one** — scope it separately so it does not inflate the current project.

> **IN SCOPE as a follow-on (decided 2026-09-15)** — but **do NOT pitch it as
> novel.** The survey rates it "fully redundant as a research idea": Sentinel,
> FIPER, Hide-and-Seek and the Agia thesis all occupy it. It is viable as
> *productisation* of existing methods, which is a real offering and an honest
> one. Scoped separately from this project so it does not inflate it, and the
> traces this project produces are its input — no additional collection.

---

## 7c. Data gap or policy gap?

The manifest's core claim is "collect this data and the failure goes away." That is false whenever the weakness is capacity or representation rather than coverage. Six discriminators, cheapest first.

| # | Test | Reads as **data gap** | Reads as **policy gap** |
|---|---|---|---|
| 1 | **Environment control.** Can a scripted expert still solve the perturbed task? | expert succeeds | expert also fails ⇒ the perturbation broke the *task*, not the policy — no gap at all |
| 2 | **Supply-side density** (Phase 0). Is the failing region represented in training? | sparse there | densely covered and still failing ⇒ capacity/representation |
| 3 | **In-distribution replay.** Does it succeed on its own training conditions? | yes | fails inside its own training distribution ⇒ underfitting |
| 4 | **Privileged-input ablation.** Give it ground-truth object pose instead of pixels — via `PrivilegedProbePolicy` (see below). | still fails ⇒ control/action gap | succeeds ⇒ failure is perceptual, upstream of control |
| 5 | **Cross-policy control.** Do other policies fail in the same region? | only this one fails | all fail ⇒ benchmark artifact or genuinely hard regime |
| 6 | **Data-response curve.** Fine-tune at escalating budgets. | rises steadily — slope tells you how much more | plateaus below acceptable, or does not move ⇒ capacity limit or wrong trigger hypothesis |
| **7** | **Literature check: is there a published NON-DATA fix for this failure family?** Mandatory, cheapest of the seven, run it FIRST. | nothing published ⇒ data is plausibly the lever | a published cheap fix exists ⇒ `fixability` is **not** `data` |

**Discriminator 7 is the one that protects our credibility**, and our headline
prediction is the case in point. P1 says camera viewpoint is the most brittle
axis. The field has largely solved it *without collecting data*:

- **Feature Token Modulation**: viewpoint success **48.5% → 87.1% with 4,000
  parameters**; Feature Linear Adaptation reaches 90.8% with 4.7M
  ([arXiv:2512.02902](https://arxiv.org/abs/2512.02902)).
- **GS-VLA** canonicalises viewpoint in front of a **frozen** policy — zero
  policy changes ([arXiv:2608.19066](https://arxiv.org/html/2608.19066)).
- **The Moving Eye** takes the data route but warns that naively increasing
  viewpoint diversity **induces shortcut learning**
  ([arXiv:2607.02322](https://arxiv.org/pdf/2607.02322)).

If our top manifest row is viewpoint and we mark it `fixability: data`, a
competent buyer produces these three papers and asks why we are selling
collection. §7b.1 anticipated this failure mode of a data vendor; the survey
supplies the ammunition.

Notes on the two that carry most of the weight:

**Discriminator 4 needs an audited escape hatch** *(DG-6)*. `policy_view()`
strips every `_gt_` key and `ARCHITECTURE.md` §3 makes that *enforced*, not
conventional — which is the architecture's best feature and must not be
weakened. So the ablation runs through an explicit **`PrivilegedProbePolicy`**
wrapper that receives full state, stamps `privileged: true` on every rollout it
produces, and is **excluded from every headline number** by the same filter that
excludes `tier3` diagnoses. An opt-in audited exception, never a flag on the
strict path.

**#1 is a control we already have and the proposal does not mention.** Our scripted policy solves the perturbed task at 100% while the faulty one fails — that is proof the perturbation is *solvable*, so a failure is attributable to the policy. Without it, an over-aggressive perturbation that makes the object unreachable looks exactly like a model weakness. Run it for every perturbation cell before drawing any conclusion.

**#6 is the definitive test, and the SHAPE of the curve is the diagnostic** — not its endpoint. This is why the proposal's "staged remediation budgets" matter and why collapsing Phase 5 to a single fine-tune destroys the information: one budget gives a point, and a point cannot distinguish "needs more data" from "cannot learn this."

Practical ordering: run 1, 2 and 4 *before* writing a remediation recommendation — all three are cheap and none require training. Reserve 6 for the top row.

---

## 8. Timing and cost

### Forward-pass estimates

**These are estimates, not measurements.** Measuring them is a Phase-1 task; treat everything here as ±3×, and expect rendering — not inference — to dominate on small models.

| Policy | Hardware | Fwd pass | Chunk | Effective |
|---|---|---|---|---|
| Scripted | CPU | ~µs | 1 | ~10⁵ Hz |
| Octo-small | 8 GB laptop | ~10–20 ms | 1 | ~50–100 Hz |
| SmolVLA 450M | 8 GB laptop | ~30–60 ms | ~8 | ~130–260 Hz |
| π0 3B | 8 GB laptop | ~150–300 ms | ~8 | ~25–50 Hz |
| OpenVLA (vanilla) | A100 | ~160 ms | 1 | ~6 Hz (published) |
| OpenVLA-OFT | A100 | ~30 ms | 8 | ~75–100 Hz |
| OpenVLA 4-bit | 8 GB laptop | ~300–1000 ms | 1 | ~1–3 Hz |

### Episode and campaign cost

Assumptions: 300 steps/episode; chunk 8 ⇒ ~38 forward passes; MuJoCo step + offscreen render ≈ 5 ms/step ⇒ **~1.5 s/episode of pure simulation**.

| Campaign | Episodes | SmolVLA @ laptop | Notes |
|---|---|---|---|
| Smoke test | 10 | ~1 min | |
| Baseline, 10 tasks × 50 | 500 | **~30–45 min** | render-bound |
| Targeted stress (Ph. 0-directed) | ~1,200 | ~1.5–2 h | vs ~2,400 for the full grid |
| Counterfactual probes | ~600 | ~45 min | |
| Regression set | ~300 | ~20 min | |

**The headline number: the full campaign is an overnight run on this laptop.** Rendering is the bottleneck, it is CPU-bound, and it parallelizes across cores — so a multi-process runner is worth building early (and G10, resumability, matters because overnight runs crash).

### The cheap unlock, stated once

If the OpenVLA-OFT path is ever wanted: the whole campaign is roughly **10–20 A100-hours ≈ $20–50** at current spot rates. The compute objection is smaller than it looks; the real costs are setup time and the 7B fine-tune in Phase 5. Noting it because the number is small enough to change the decision. Not pursuing it unless asked.

---

## 9. Data Gap Manifest — schema

Every row is a falsifiable claim with its evidence attached. Fields marked ✱ are mandatory; a row missing one is a hypothesis, not a manifest row.

> **Status (DG-10).** ✱ marks a field mandatory, and the rule below says a row
> missing one is a hypothesis, not a manifest row. Several ✱ fields are **not
> yet implemented** — `failure_cost`, `non_data_fix`, `fixability`,
> `discriminators`. **By the schema's own rule, no row the harness can currently
> emit is a valid manifest row.** That is a status fact, not a design error, and
> the implementation status column below says which is which.

| Field | Type | Status | Notes |
|---|---|---|---|
| `row_id` ✱ | str | built | `DGM-001` |
| `failure_conditional` ✱ | float + CI | **P(fail \| condition)** — ours, measured, what the counterfactual probe establishes. Transfers moderately. |
| `condition_prevalence` | obj \| `"not estimated"` | **CLIENT-SUPPLIED** from deployment logs. In sim this is an artifact of *our own grid*, so we do not ship a number for it. A region the uniform arm never sampled renders `not estimated` — **never** a low severity. |
| `severity` | computed **at delivery** | conditional × prevalence × `failure_cost`. **We do not ship a cardinal severity.** See §9.3. |
| `task_scope` ✱ | list[str] | which tasks, and how many of the suite |
| `failure_family` ✱ | enum | the 7-family taxonomy |
| `symptom` ✱ | str | observable behaviour, not inferred cause |
| `trigger` ✱ | str | **must** cite the counterfactual probe result |
| `boundary` ✱ | obj | axis, units, value, CI, and the boundary *definition* used |
| `evidence` ✱ | obj | n episodes, success at L0 and at boundary, seeds, run_ids |
| `supply_side` | obj | training-set coverage on this axis (Phase 0) — the independent corroboration |
| `coverage_required` ✱ | str | **diversity axes, NEVER demonstration counts** — see below |
| `sampling_arm` ✱ | enum | uniform / adaptive — frequency stats must filter to `uniform` (§5.1) |
| `non_data_fix` ✱ | obj | §7c discriminator 7: published cheap fixes for this family, with citations |
| `validation_plan` ✱ | str | budgets, regression set id, metric |
| `validation_result` | obj | populated by Phase 5. **Null = untested claim.** |
| `fixability` ✱ | enum | augmentation / data / architecture / operational — see §7b.1 |
| `discriminators` ✱ | obj | results of the §7c tests that were run |
| `failure_cost` ✱ | enum | benign / disruptive / safety — see §7b.3 |
| `envelope` | obj | certified operating range, for the envelope spec (§7b.2) |
| `provenance` ✱ | obj | schema version, harness commit, checkpoint revision, date |

Two properties the proposal implies but does not state: a row whose `trigger` lacks a counterfactual is downgraded to "correlational" in the readout; and `validation_result: null` must be visible to the reader, not hidden. Overstating an untested row is the fastest way to lose a client's trust.

### 9.1 `coverage_required`: diversity, not counts *(revised 2026-09-11)*

**Lin et al., ICLR 2025** ([arXiv:2410.18647](https://arxiv.org/abs/2410.18647);
40,000+ demos, 15,000+ real rollouts) find generalization follows a **power law
in the number of distinct environments and objects, not in raw demonstration
count**, with sharp diminishing returns past a per-environment threshold.

So a row reading *"collect 500 more demos in the failing yaw band"* is close to
the **worst** possible recommendation — it buys depth on an axis with
diminishing returns instead of breadth on the axis that scales. The proposal's
own example table says "validate increasing targeted sample budgets": wrong
axis, and it should be corrected.

Our schema's existing wording already had roughly the right shape. This finding
turns that from a stylistic preference into an **evidence-backed requirement**,
and the citation belongs in the readout.

### 9.3 Severity is not ours to compute *(added after DG-3 / DG-5b)*

Severity conflates two quantities with opposite transfer properties:

| | What it is | Transfers? |
|---|---|---|
| **P(fail \| condition)** | a mechanism claim, measured, established by the counterfactual probe | moderately — it is a property of the policy |
| **P(condition)** | how often the condition arises | **no** — in sim it is an artifact of the perturbation grid *we chose* |

The old `severity = frequency × task coverage` was dominated by the half that
cannot transfer and that we invented. That is why the literature reports
severity *ordering* transferring worst, and it is why the manifest must not ship
a cardinal severity.

**We ship `failure_conditional`. The client supplies `condition_prevalence` from
their deployment logs. Severity is computed at delivery.**

Honest — we never claim to know their prevalence — and commercially *stronger*:
the artifact becomes something the client's own data completes, and the ranking
becomes theirs to defend. It is structurally identical to `failure_cost`
(§7b.3), which is already client-specific for exactly this reason. Severity and
cost have the same shape; only one of them previously admitted it.

**This also fixes a silent demotion.** With frequency filtered to the uniform
arm, a failure mode found *only* by the adaptive arm had count ≈ 0 and sorted to
the bottom — so the manifest systematically buried the output of its own most
novel component. An adaptively-discovered region has an **unknown** prevalence,
not a low one. Silent demotion and honest abstention look identical in a ranked
table and are entirely different claims.

### 9.2 Evaluation statistics *(added 2026-09-11)*

**Now — cheap, do it.** Follow Kress-Gazit et al., *Robot Learning as an
Empirical Science* (TRI, [arXiv:2409.09491](https://arxiv.org/abs/2409.09491)):
**paired** statistical tests across perturbation conditions, Bayesian credible
regions, multiple-comparison correction. We control the seed, so paired tests
are **free variance reduction** — and they directly relieve the
counterfactual-probe budget problem.

Context worth quoting: an audit of 13 recent real-robot VLA papers found the
modal per-condition N is **10–20, and none reported confidence intervals or
paired tests** *[unverified — read via search summary]*. Our Wilson intervals
already beat the modal published paper. Say so.

**Later, free if we plan now.** **SureSim**
([arXiv:2510.04354](https://arxiv.org/abs/2510.04354)) uses prediction-powered
inference to turn many imperfect-sim evaluations plus a *small* number of paired
real trials into **statistically valid CIs on real-world performance**. We have
no hardware — but if per-cell results are recorded in a PPI-consumable form,
then *"give us 30 real trials and we upgrade this entire sim campaign into
real-world confidence intervals"* becomes a commercial offer. **A schema
decision that costs nothing today and is expensive to retrofit.**

---

## 10. Schedule

| Wk | Phase | Deliverable | Gate |
|---|---|---|---|
| 1 | 0 + 1 | Checkpoint survey; harness skeleton; oracle test | **Miner recovers planted triggers** |
| 2 | 0 | Coverage report + written predictions | Predictions committed before measurement |
| 3–4 | 2 | LIBERO integration; baseline | Baseline reproduced or documented |
| 5–6 | 2 + 3 | Two-arm stress (§5.1); mining; κ + discovered-taxonomy check | κ ≥ 0.5 **and** families survive comparison |
| 7 | 4 | Data Gap Manifest | Top row actionable by an outsider |
| 8–9 | 5 | **Three-arm** remediation (§6.1); re-measure incl. non-targeted cells | Loop closed (any sign of result) |
| 10 | — | Readout + demo | |

Every phase produces a standalone deliverable. Stopping after any week leaves something presentable.

---

## 11. Open questions

1. ~~Does a LIBERO-capable small VLA checkpoint exist?~~ **Resolved: yes.** Now: can we reproduce its published numbers, given lerobot#3264? See `MODELS_AND_COMPUTE.md` §4.
2. Does LIBERO expose per-task contact/EE signals uniformly enough for generic phase detectors, or is per-task work needed?
3. Which LLM for Tier 3, and does it run locally or via API? (Cost is trivial at this volume; the question is reproducibility of the audit trail.)
4. Is remediation data generated by scripted demonstration in sim, or by re-rendering existing demos under perturbation? The second is far cheaper and probably sufficient.
5. Who is the readout audience — technical, or EXL/iMerit commercial? Changes the demo, not the work. **OPEN.**
6. ~~Is the runtime monitor in scope as a follow-on?~~ **Decided 2026-09-15: YES, as a follow-on.** Positioned as *productisation of existing methods* (Sentinel, FIPER, Hide-and-Seek), never as novel research — see §7b.4. Scoped separately so it does not inflate this project.
7. Recovery diagnosis is unvalidated (see §4). Do we exclude recovery rows from the first client manifest, or validate it against LIBERO first?
8. ~~Do we switch policy?~~ **Decided 2026-09-11: VLA-Adapter as subject; π0.5 anchor deferred (§5.2).** Remaining: does VLA-Adapter's own gate pass?
8b. ~~D2 — episodes per task.~~ **Decided 2026-09-11: 10/task, and the ±5 pp gate is replaced by a CI-overlap test (§5.3).**
9. Sim-to-real rank correlation is contested (Spearman 0.4–0.7 **[UNSOURCED]**) and **severity ordering transfers worst** — which is what we prioritise by. How do we caveat severity in a client-facing manifest?
