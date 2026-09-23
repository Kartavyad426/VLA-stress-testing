# Implementation & Data Flow

**Purpose:** who produces which data, who consumes it, and where to look when a
number is wrong. Companion to `PLAN.md` (what we are building and why).

> **This document describes the code AS BUILT.** Rows and boxes marked
> **`[PLANNED]`** are agreed in `PLAN.md` but **not yet implemented** — do not
> debug against them. See §0 for the pending list.

---

## 0. Pending changes — agreed 2026-09-11; status as of 2026-09-23

> **Status (2026-09-23):** all but two are now built (Waves A–C, see
> `IMPLEMENTATION.md`). **Still open: #5 `adaptive_sweep()` and #7 the
> `coverage_required` validator.** #2 was built in the form #12 specifies. The
> "why" column is kept as written on 2026-09-11.

From the 2026-09-11 landscape review (`docs/LANDSCAPE.md` adoption ledger) and
the `FINDINGS.md` entries. Ordered by when they get more expensive to retrofit.

| # | Change | Where | Why it cannot wait |
|---|---|---|---|
| 1 | **BUILT** — ~~`sampling_arm` on the `Rollout`~~ → **`runs/<id>/arms.jsonl`, a request log** mapping `(arm, rollout_id)` many-to-many | `runner.py`, store | **Revised after DG-2.** Tagging the rollout de-uniforms the uniform arm: `rollout_id` excludes the arm (correctly — same physics), so one stored rollout gets whichever arm asked first, and the uniform arm silently loses cells to cache hits in adaptive search order. The arm belongs to the **request**. `Rollout` unchanged ⇒ **no schema major bump for this item.** |
| 2 | **BUILT** — **`mujoco` version moves from `runtime` into Env `identity()`** | `schema.py`, env adapters | F2: mujoco does not shift timings, it **changes results** (80% → 28% on one task). A version bump must MISS the cache, not warn. My original placement was wrong. |
| 3 | **BUILT** — **Rendering backend into Env `identity()`** | env adapters | EGL vs osmesa can change rendered observations ⇒ changes results (`PLAN.md` §5.4). |
| 4 | **BUILT** — **Paired statistics in `counterfactual_probe`** | `runner.py` | We run the same seeds in both arms and currently discard the pairing. McNemar on paired per-seed outcomes is free variance reduction. |
| 5 | **OPEN** — `adaptive_sweep()` with a surrogate; `boundary` gains a posterior interval | `runner.py` | `PLAN.md` §5.1. Largest piece; only pays off once there is a real boundary. |
| 6 | **BUILT** — `failure_cost` computed in `classify()` | `mining/classify.py` | `PLAN.md` §7b.3 — rated our most novel contribution, and currently unimplemented. |
| 7 | **OPEN** — `coverage_required` validator rejecting count-shaped strings | `manifest.py` | `PLAN.md` §9.1 — demo counts are the wrong axis (Lin et al.). |
| 8 | **BUILT** — Environment control runs **per task**, not per suite | `vla_harness/control.py` | F2's lesson: at suite level a task-5 collapse dilutes into noise. |
| 9 | **BUILT** — **`scene_descriptor`** on every rollout — object poses, camera pose, lighting **in physical units**, resolved from the seed at reset | env adapters, `schema.py` | `PLAN.md` §9.2. PPI needs a scene a *person could rebuild on a bench*; `seed`+`spec` is sim-internal by construction and cannot supply it (DG-8). Cheap now, unreconstructable later. |
| 10 | **BUILT** — **`RegressionSet`** as a first-class L0 artifact: own id, `frozen_env` identity, member rollout ids | `schema.py`, new module | It is the comparison basis for all of Phase 5 and currently has no type, no owner and no fingerprint — so it expires invisibly whenever env identity moves (DG-7). Finding #2 one level up, against our headline claim. |
| 11 | **BUILT** — **`PrivilegedProbePolicy`** — full state, stamps `privileged: true`, excluded like `tier3` | `policies/` | §7c discriminator 4 is unrunnable today: `policy_view()` strips `_gt_` and that is enforced. Audited escape hatch, not a weakening (DG-6). |
| 12 | **BUILT** — `semantic_runtime` bucket + promotion policy | `schema.py` | Supersedes item 2's naive "mujoco into identity()" (DG-4). |

**Sequencing:** 1–3 first (correctness + schema, worse to retrofit once real
traces exist), then 8–9 before Phase 2 accumulates traces, then 4 and 6, then 5.

---

## 1. Module map

| File | Layer | Owns | Must NOT know about |
|---|---|---|---|
| `vla_harness/schema.py` | L0 | `Observation` `Action` `Step` `Rollout` `PerturbationSpec` `TraceStore` | anything above it |
| `vla_harness/policies/scripted.py` | L1 | the oracle + its plantable faults | envs, mining, manifest |
| `vla_harness/envs/toy.py` | L2 | world state, rendering, goal predicate | policies, mining, manifest |
| `vla_harness/runner.py` | — | `rollout` `run_cell` `sweep` `counterfactual_probe` `find_boundary` | which env/policy it holds |
| `vla_harness/mining/phases.py` | L3 | phase segmentation, divergence, terminal behaviour | envs (G9) |
| `vla_harness/mining/classify.py` | L3 | taxonomy rules, tier-3 seam | envs (G9) |
| `vla_harness/mining/cluster.py` | L3 | grouping failures | envs (G9) |
| `vla_harness/manifest.py` | L4 | manifest rows + rendering | envs, policies |
| `experiments/oracle_test.py` | — | the acceptance gate | — |
| `vla_harness/envs/libero_env.py` | L2 | LIBERO / LIBERO-Plus adapter over LeRobot's `LiberoEnv`; `scene_descriptor`, the LIBERO `_gt_*` keys (§3) | policies, mining, manifest |
| `vla_harness/envs/nominal.py` | L2 | what "nominal" means for a LIBERO-Plus vision variant (R-039's paired source) | policies, mining |
| `vla_harness/policies/lerobot_policy.py` | L1 | any LeRobot policy (GR00T, MINERVA, SmolVLA, π0 family) behind the Policy protocol | envs, mining, manifest |
| `vla_harness/policies/privileged.py` | L1 | `PrivilegedProbePolicy`, the audited `_gt_` escape hatch (§0 #11) | mining, manifest |
| `vla_harness/mining/phases_libero.py` | L3 | LIBERO phase segmentation | envs (G9) |
| `vla_harness/mining/signals.py` | L3 | per-env signal extraction feeding one `classify()` | envs (G9) |
| `vla_harness/mining/agreement.py` | L3 | taxonomy validation beyond κ (F6a) | envs (G9) |
| `vla_harness/control.py` | — | per-task environment control (C9) | — |
| `vla_harness/conformance.py` | — | checkpoint ↔ env wiring gate, run before a campaign | mining, manifest |
| `vla_harness/video.py` | — | per-episode mp4 of the frames the policy saw; path in `Rollout.meta["video"]` | mining |
| `vla_harness/probes/language.py` | — | language sensitivity probe | — |
| `vla_harness/capture/` | — | activation taps (`taps.py`, `groot_features.py`), OOD analysis (`analysis.py`), R-039 pathway splice (`splice.py`). Attaches from outside; `third_party/` unmodified | mining, manifest |
| `vla_harness/analysis/r039.py` | — | R-039 aggregation from `runs/<run>/splice/` | model, simulator |
| `experiments/harness_eval.py` | — | the eval entry point for every reported rate; records mp4 per episode by default (`--no-video` to disable) | — |
| `experiments/visualise_set.py` | — | reference-vs-variants comparison page, with activations | — |
| `experiments/vlm_label.py` | — | VLM labeller (local Qwen3-VL-8B 4-bit, or NVIDIA NIM). **Labels unvalidated** — first test named the wrong grasped object | — |

**The one rule that keeps this honest:** the mining layer imports nothing from
`envs/` or `policies/`. It consumes `Rollout` objects. If you ever need an env
import in `mining/`, the signal you want belongs in `Observation.state` instead.

---

## 2. Flow

```
   seed ──┐
          ├──► Env.reset(seed, spec) ──► Observation ──┬─► .policy_view() ─► Policy
   spec ──┘                ▲                           │   (_gt_ stripped)    │
                           │                           └─► FULL state ────►  │
                           │                               PrivilegedProbe    │
                           │                               [PLANNED, audited] │
                           │                                               Action
                           └───────────── Env.step(action) ◄──────────────────┘
                                              │
                              (loop until done: success | timeout)
                                              │
                                              ▼
                                         Rollout                    ──► TraceStore
                                    + scene_descriptor [PLANNED]         (jsonl)
                                    + fingerprint{identity,              + arms.jsonl
                                       semantic_runtime, runtime}          [PLANNED]
                                              │
                        nominal reference ────┤
                                              ▼
                                     classify()  ──► rollout.diagnosis
                                              │
                                              ▼
                                        cluster()  ──► [cluster]
                                              │
   sweep()          ──► [Cell] ──► find_boundary() ──► boundary ──┐
   (uniform arm)         │                                        │
                         └──► frequency ──► failure_conditional ──┤
                                                                  ├─► manifest
   adaptive_sweep()  ──► [Cell] ──► surrogate ──► boundary ───────┤   .make_row()
   [PLANNED §5.1]                            (posterior interval) │        │
                                                                  │        ▼
   counterfactual_probe() ─────────────────► probe ───────────────┘  DATA_GAP_
        └─ paired per-seed outcomes [PLANNED]                        MANIFEST.md

   RegressionSet [PLANNED] ──► frozen (seed, spec) members + frozen_env identity
        └─► Phase 5 before/after.  Re-run under a DIFFERENT env identity
            requires a LOUD explicit override — never a silent pass.

   arms.jsonl [PLANNED] ──► (arm, rollout_id) many-to-many
        └─► uniform frequency computed over cells the UNIFORM arm ASKED for,
            each resolved through the cache.  One rollout serves both arms.
```

### What enters a rollout

**seed**, **PerturbationSpec**, **policy identity** — nothing else. Same three in
⇒ same *outcome distribution* out. Note this is **distributional, not
byte-identical**: bit-exactness holds on the toy and will not survive a real VLA
(GPU nondeterminism, sampling action heads, contact chaos — §5.1). Probes
therefore compare distributions over N seeds against a measured noise floor,
never single trajectories.

### Two sampling arms, never mixed

The **uniform** arm feeds anything frequency-weighted; the **adaptive** arm
feeds boundary estimation. Active sampling optimises for *finding* failures, so
it biases the discovered failure population.

**The arm belongs to the REQUEST, not the episode (DG-2).** `rollout_id` hashes
(identity, seed, spec) and correctly excludes the arm — the physics is identical
either way. Tagging the stored rollout would mean one rollout per cell carrying
whichever arm asked *first*, so the uniform arm would silently lose cells to
cache hits in adaptive search order, and its sample would stop being uniform.
Hence `arms.jsonl`: frequency is computed over the cells the uniform arm
**asked for**, each resolved through the cache. One rollout legitimately serves
both arms.

### The manifest does not ship a severity

`cluster()` and the uniform arm produce **`failure_conditional`** —
P(fail | condition), which we measure and the counterfactual probe establishes.
**`condition_prevalence` is client-supplied** from their deployment logs;
severity is computed at delivery as conditional × prevalence × `failure_cost`
(DG-5b, `PLAN.md` §9.3).

Two consequences visible in the flow: a region the **adaptive** arm found and
the uniform arm never sampled renders `prevalence: not estimated` — never a low
severity (I11); and `privileged: true` rollouts from the probe path never enter
a reported number (I13).

---

## 3. Data dictionary — `Observation.state`

This is where most debugging ends up. Two classes of key, and the distinction is
enforced in code, not by convention alone.

| Key | Written by | Read by | Privileged |
|---|---|---|---|
| `obj_cam_xy` | env `_obs` | **policy** | no — "the pixels" |
| `distractors_cam_xy` | env `_obs` | **policy** | no |
| `ee_xy` | env `_obs` | policy + miner | no — proprioception |
| `gripper` | env `_obs` | policy + miner | no |
| `holding` | env `_obs` | policy + miner | no |
| `grasp_attempts` | env `_obs` | policy + miner | no — the robot's own action history |
| `_env_camera_yaw_deg` | env `_obs` | policy | no — extrinsics from calibration |
| `_gt_obj_xy` | env `_obs` | **miner only** | **YES** |
| `_gt_ee_to_obj` | env `_obs` | **miner only** | **YES** |
| `_gt_dist_to_distractor` | env `_obs` | **miner only** | **YES** |

The rows above are the **toy** env's keys. The **LIBERO** env (`envs/libero_env.py`)
writes its own `_gt_` keys; the two object-pose keys differ in scope and it matters:

| Key | Written by | Read by | Privileged |
|---|---|---|---|
| `_gt_object_pos` | LIBERO env `_obs` | **miner only** | **YES** — **BDDL task objects only**. It missed a wrong-object grasp of a ramekin; do not use it to ask "what did the gripper touch?" |
| `_gt_scene_object_pos` | LIBERO env `_obs` | **miner only** | **YES** — **every free-joint object**, including distractors |

Also written: `_gt_object_pos_complete`, `_gt_eef_to_object`,
`_gt_eef_to_nearest_object`, `_gt_nearest_object` (all over the BDDL task objects
only) and `_gt_n_contacts`.

### The `_gt_` contract

`_gt_` = ground truth available *only because we are in simulation*. A real robot
does not have exact object poses. A policy that reads one produces a success rate
that collapses on hardware.

**This is enforced, not documented.** `runner.rollout` calls
`policy(obs.policy_view())`, which strips every `_gt_` key. A policy that reaches
for one gets `None`/`KeyError` immediately rather than a quietly inflated score.
Detectors receive the full state via `Step.obs_state`.

> Caught during this build: `grasp_attempts` was originally named `_gt_grasp_attempts`
> and read by the policy. It was simply misnamed — a robot does know its own
> action history — but under the old, unenforced convention it read as a leak.
> Adding `policy_view()` turned "remember not to do this" into "cannot do this."

**Adding a signal for LIBERO:** decide first whether a real robot has it. Camera
extrinsics — yes, calibration. Object pose — no, `_gt_`. When in doubt, `_gt_` it:
a detector losing a signal is cheap, a policy gaining one invalidates the run.

---

## 4. Data dictionary — the artifacts

### `Rollout` — the atomic unit of evidence

| Field | Written by | Read by | Notes |
|---|---|---|---|
| `rollout_id` | `runner.rollout` via `cell_hash` | `TraceStore` | hash of (policy_id, env_id, task, seed, spec), where the ids are **derived from `identity()`** — see §4.1 |
| `steps[]` | `runner.rollout` | all of `mining/` | full `obs_state` per step, privileged keys included. **The last Step is terminal: `action is None`** — it carries the observation in which success was decided, for which no action was ever requested. Consumers iterating for actions must handle `None`; `series()` picks it up unchanged. |
| `fingerprint` | `runner.rollout` via `make_fingerprint` | `run_cell` on every cache hit | `{env, policy, runtime}` — **strictly broader than the cache key** (§4.1) |
| *(no arm field — see `arms.jsonl`)* | `runner.run_cell` writes the request log | every frequency statistic | **DG-2:** the arm is a property of the request, not the episode. Frequency is computed over cells the uniform arm **asked for**, resolved through the cache. |
| `success` | **env only** | everything | the goal predicate. The only trustworthy signal. |
| `termination` | env | `mining/phases` | `grasped` / `timeout` |
| `perturbation` | `PerturbationSpec` | `cluster`, `manifest` | flat knob→value |
| `forward_passes` | `runner.rollout` | cost reporting | for the §8 timing table |
| `diagnosis` | **`mining/classify` only** | `cluster`, `manifest` | `None` until mined. Never written by env or policy. |

### 4.1 Identity, the cache key, and the fingerprint

Three layers, and the distinction between them is the whole of finding #2.

| | Contents | Purpose |
|---|---|---|
| `identity()` | every load-bearing **value**: constructor kwargs, thresholds affecting the goal predicate, checkpoint revision, control mode, action space | defines what "the same env/policy" means |
| `env_id` / `policy_id` | `name@hash(identity())` | readable, and changes when identity does |
| `cell_hash` | `(policy_id, env_id, task_id, seed, spec)` | the cache key |
| `Rollout.fingerprint` | `{env: identity, policy: identity, runtime: {python, platform, mujoco, torch, …}}` | **stored and compared on every hit** |

**Why the fingerprint must be broader than the key.** A fingerprint equal to
`identity()` can never disagree with a hit keyed on `hash(identity())`: change
identity and the key misses; leave it alone and the fingerprint matches. The
comparison would be dead code. `runtime` is what makes it live — a MuJoCo point
release or a different GPU can move success rates, but putting either in the key
would invalidate every rollout on every machine and destroy the cross-session
resume that caching exists for. So: **in the fingerprint, out of the key. A
change is caught and reported, not silently ignored and not silently fatal.**

**No git SHA, in any form.** Simultaneously too coarse (a README edit invalidates
the store) and too weak (a dirty tree keeps the SHA stable while code changes
underneath it, manufacturing confidence in a stale cache during exactly the phase
where env config gets nudged most). Hash the values.

> **Revised twice — final form `[PLANNED]` (DG-4).** First I put all library
> versions in `runtime` (reported, never fatal). Then F2 showed MuJoCo changing
> a task 80% → 28%, and I proposed the rule *"anything that changes the ANSWER
> goes in `identity()`"*. **That rule is not decidable**: torch and numpy change
> answers too, via reduction order → contact chaos (§5.1 says so). Applied
> honestly it keys torch, every pip upgrade invalidates the store, and we are
> back at the git-SHA failure mode.
>
> **Three buckets, and a promotion policy.** The distinction is epistemic:
>
> | Bucket | Holds | Keyed? |
> |---|---|---|
> | `identity()` | env/policy **design** | **yes** |
> | `semantic_runtime` | external components with a **documented, specific** semantic effect: `mujoco`, `MUJOCO_GL`. Enumerated per adapter. | **yes** |
> | `runtime` | `torch`, `numpy`, `python`, platform | no — reported |
>
> `semantic_runtime` is separate from `identity()` because a simulator build is
> not part of an env's task *design* — `ToyReachEnv.identity()` declaring a
> MuJoCo version it never loads would be incoherent.
>
> **Promotion** from `runtime` → `semantic_runtime` requires documented
> evidence, and is a deliberate, rare, store-invalidating event logged like a
> schema bump. MuJoCo qualified (F2). torch has not.

**On mismatch: re-run loudly, don't raise.** Raising kills an overnight sweep at
hour 8 over a moved threshold; under session caps someone then disables the check,
and a disabled check is worse than none. The diff of changed fields goes in the
run summary. `strict=True` raises, for client-facing runs.

**Scope limit — detector config is NOT covered.** `PhaseSegmenter` radii,
`LOST_TARGET_M` and friends are outside the fingerprint. This is safe *only*
because rollouts are persisted **before** classification (§7), so every load
re-classifies from scratch and a detector change correctly shows a changed
result. **That property is load-bearing.** The moment a classified rollout is
persisted, detector churn joins this problem and the fingerprint must grow.

*Deferred, not rejected:* sampled cache audit (re-run ~2% of hits, check
agreement). Sound on the toy; unavailable on a real VLA — see §5.1.

### `diagnosis` — written by `classify()`

| Field | Meaning | Debug value |
|---|---|---|
| `source` | `tier1` (deterministic) or `tier3` (LLM) | **filter headline numbers to `tier1`** |
| `confident` | did a rule match | `False` ⇒ routed to review |
| `family` | taxonomy label, or `None` if succeeded | |
| `reason` | human-readable justification | first thing to read on a misclassification |
| `phases[]` | per-step phase labels | wrong ⇒ look at `PhaseSegmenter` thresholds |
| `terminal` | `success` `retry_loop` `failed_grasp_no_retry` `never_reached` | |
| `final_error_m` | distance to target at end | drives the `visual_grounding` rule |
| `attempt_spread_m` | spread of grasp attempts | separates recovery from manipulation |
| `failure_cost` **`[PLANNED]`** | benign / disruptive / safety, from terminal state | `PLAN.md` §7b.3 — severity should weight by consequence, not only frequency |
| `divergence` | first step outside the nominal envelope | **secondary signal only** — many valid trajectories exist |

### `Cell` / `boundary` / `probe`

| Artifact | From | Key fields | Watch for |
|---|---|---|---|
| `Cell` | `run_cell` | `rate`, `ci` (Wilson), `n` | at n=20 the CI is ±20 pp. Do not report a point estimate. |
| `boundary` | `find_boundary` | `lower`, `upper`, `definition` | a **bracket**, never interpolated — false precision otherwise |
| `probe` | `counterfactual_probe` | `probes[]` sorted by `delta_pp`, `attributed_knob` | `attributed_knob` is `None` unless top delta > 20 pp |

---

## 5. Invariants

Violations are bugs, not surprises. Worth asserting in CI.

| # | Invariant | Symptom if broken |
|---|---|---|
| I1 | Same (seed, spec, policy) ⇒ same **success-rate distribution** over N seeds, within the measured floor (§5.1) | probe deltas become noise; attribution is void |
| I2 | Policies never see `_gt_*` | success rate inflated in sim, collapses on hardware |
| I3 | `success` written only by the env | you are measuring your detector, not the policy |
| I4 | `diagnosis` written only by `mining/` | circular reasoning |
| I5 | A cached rollout still contributes its outcome | **resumed cells read 0% — see §6** |
| I6 | Knob names exist in `LIBERO_PLUS_FACTORS` | axis cannot be mapped to the real benchmark |
| I7 | A detector missing a state key skips, never crashes | miner dies on LIBERO traces |
| I8 | Clean policy + perturbation ⇒ near-100% success, checked **per task** | perturbation broke the task, not the policy — at suite level a single-task collapse dilutes into noise (F2) |
| I9 **`[PLANNED]`** | Frequency statistics are computed over cells the **uniform arm requested** (via `arms.jsonl`), never over stored-rollout tags | the uniform sample silently stops being uniform (DG-2) |
| I11 **`[PLANNED]`** | A region the uniform arm never sampled renders `prevalence: not estimated`, never a low severity | the manifest silently buries everything the adaptive arm uniquely found (DG-3) |
| I10 **`[PLANNED]`** | `identity()` = design; `semantic_runtime` = documented semantic effect (keyed); `runtime` = everything else (reported). Promotion requires evidence. | undecidable rule ⇒ either a stale simulator is served from cache, or every pip upgrade invalidates the store (DG-4) |
| I12 **`[PLANNED]`** | A `RegressionSet` re-run under an env identity different from `frozen_env` requires a loud explicit override | Phase 5's before/after silently compares two different things (DG-7) |
| I13 **`[PLANNED]`** | Rollouts stamped `privileged: true` are excluded from every headline number | discriminator 4's privileged ablation leaks into reported success rates (DG-6) |

I8 is the environment control from `PLAN.md` §7c and runs in the oracle test as
`CONTROL`. Run it for every new perturbation axis **before** interpreting any
failure on it.

### 5.1 Reproducibility — I1 is weaker than it looks

The toy is bit-exact. **A real VLA will not be**, and the counterfactual probe
rests entirely on this, so it needs stating plainly.

| Source | Effect | Mitigation |
|---|---|---|
| GPU float nondeterminism (non-deterministic kernels, cuDNN algo selection, TF32) | logits differ in the last bits | `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off — costs speed |
| **Stochastic action heads** — π0 (flow matching), GR00T / Octo (diffusion) sample | genuinely different actions per call | seed the noise draw explicitly; OpenVLA's argmax decode is the exception |
| **Contact chaos** | a 1e-6 command difference flips whether a grasp catches an edge | **cannot be eliminated** — manipulation is numerically stiff |
| Driver / CUDA / library drift | different numbers on a different machine | pin everything; cross-machine reproduction is effectively unachievable |

Tell from the literature: the OpenVLA-OFT paper averages over three random
seeds. You do not do that unless you see variance.

**So we do not depend on exact determinism.** `counterfactual_probe` compares
*distributions* over N seeds. `reproducibility_floor()` measures run-to-run
variance at a fixed knob setting, and a knob is attributed only when its delta
clears `max(20 pp, 3 x floor)`.

**What this means for the cache.** Once traces stop being bit-reproducible, the
cache stops being "avoiding redundant deterministic work" and becomes **reusing
samples from a distribution**. Two consequences:

1. The fingerprint must cover anything that shifts the **distribution**, not
   merely anything that changes a trace — which is why `runtime` is in it (§4.1).
2. **A cache entry can never be verified by re-running it.** Any audit degrades
   to comparing success-rate distributions at n ≥ 20, which catches gross
   corruption and nothing subtle. Do not assume the store is checkable.

Note `reproducibility_floor()` takes **no store, deliberately**. Threading one
through would serve the second repeat from the first repeat's cache and report
`floor_pp = 0.0` by construction — a plausible-looking optimisation that
silently destroys the only measurement telling us whether a probe delta is real.

**Run `reproducibility_floor()` once per (policy, env) before trusting any
probe, and report the floor alongside every boundary.** A 58 pp delta survives
a 5 pp floor comfortably; a 7 pp delta does not, and reporting it as causal
would be the single easiest way to put a false claim in a client manifest.

---

## 6. Debugging playbook

| Symptom | Most likely cause | Look at |
|---|---|---|
| Every cell reads 0% or 100% | fault fires always/never — the fixture, not the miner | sweep the knob alone; check nominal ≈ 100% |
| Resumed sweep reads 0% | I5 — cached rollouts skipped their result too | `run_cell` must count `cached.get(rid)` |
| `attributed_knob` is `None` | probe spec lacks the failing knob, or delta < 20 pp | print `probe["probes"]` — the deltas |
| Boundary "not crossed" | swept range too narrow, or fault is gradual | widen levels; check nominal rate |
| Everything labelled `recovery` | grasp-attempt spread rule not firing | `attempt_spread_m` in the diagnosis |
| Everything labelled `visual_grounding` | `LOST_TARGET_M` too low, or a genuine grounding fault | `final_error_m` distribution |
| Family right, trigger wrong | co-perturbed knobs are confounded | probe from a spec with the suspect knob **active** |
| Miner flags a clean policy | I8 violated — the perturbation broke the task | the `CONTROL` block |
| CI absurdly wide | n too small | n=20 screens; go to n=50 near the boundary |
| Trace won't load | I6/G3 — schema major changed | `schema_version` in the jsonl |
| One task collapses, suite looks fine | simulator version, or a genuine per-task weakness | run the I8 control **on that task**; check `fingerprint.env.mujoco` (F2) |
| Severity ordering looks odd | I9 — frequency computed over adaptive samples | filter to `sampling_arm == "uniform"` |

### Worked example — the resumability bug

*Symptom:* second run of the oracle test showed 0% success in every cell,
including nominal, on a policy that had just scored 100%.

*Wrong reading:* "the policy collapsed."
*Actual:* `run_cell` skipped cached rollouts with `continue`, so `ok` never
incremented while `n` stayed at `len(seeds)`. Every resumed cell read 0/20.

*Tell:* nominal was 0% too. **A nominal cell below ~90% is nearly always a
harness bug, not a finding** — that is the cheapest sanity check in the system,
and it is why the oracle test prints the nominal row first.

---

## 7. On-disk layout

```
runs/<run_id>/
  rollouts.jsonl          append-only, one Rollout per line, greppable
  DATA_GAP_MANIFEST.md    the L4 artifact
```

`rollouts.jsonl` is the **source of truth**. Parquet, Delta and S3 are
projections of it (`PLAN.md` §7.2) — never the other way round. It stays
greppable on purpose:

```bash
# how many failures at yaw >= 12?
grep -c '"camera_yaw_deg":1[2-9]' runs/oracle_test/rollouts.jsonl

# every diagnosed family, counted
python3 -c "
import json,collections
c=collections.Counter()
for l in open('runs/oracle_test/rollouts.jsonl'):
    d=json.loads(l).get('diagnosis') or {}
    c[d.get('family')]+=1
print(c)"
```

**Rollouts are persisted BEFORE classification, and this is load-bearing.**
`run_cell` calls `store.append(r)` while `r.diagnosis` is still `None`; nothing
in the codebase ever persists a classified rollout. Two things follow:

1. Mining is re-runnable over stored traces without re-simulating — the point of
   storing them.
2. **A detector-config change correctly produces a changed result**, because
   every load re-classifies from scratch. This is why detector thresholds can sit
   outside the fingerprint (§4.1) without reintroducing finding #2.

If anyone ever persists a classified rollout, property 2 disappears silently and
the fingerprint must be extended to cover detector config. Verify with:

```bash
python3 -c "
import json,collections
c=collections.Counter(type(json.loads(l).get('diagnosis')).__name__
                      for l in open('runs/oracle_test/rollouts.jsonl'))
print(c)   # expect: Counter({'NoneType': N})"
```

---

## 8. Adding a real policy or env

**Policy** — implement `policy_id`, `action_dims`, `reset()`, `__call__(obs) -> Action`.
Read only non-`_gt_` keys. Nothing else in the codebase changes.

**Env** — implement `env_id`, `task_id`, `action_dims`, `reset(seed, spec)`,
`step(action)`. Map `PerturbationSpec` knobs onto the simulator's parameters,
and expose whatever detectors need in `state` under the `_gt_` rules.

**LIBERO checklist**
1. Map LIBERO-plus factors to `LIBERO_PLUS_FACTORS` — names already match (G4).
2. Expose contacts, EE pose, object poses, BDDL predicate as `_gt_*`.
3. Write frames to disk; put paths in `image_refs`, never pixels in state (G2).
4. Re-tune `PhaseSegmenter` radii — they are constructor args for this reason.
5. **Re-run the oracle gate against the new `PhaseSegmenter` config.** Detector
   thresholds are the part most likely to silently break on a new env.
