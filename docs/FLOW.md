# FLOW — how one episode becomes a manifest row

**Purpose:** an explanatory walkthrough of the pipeline, end to end, for someone
who needs to understand *what each stage is for* before reading the code.

**Companions:** `ARCHITECTURE.md` (who produces which data, and where to look
when a number is wrong) · `PLAN.md` (what we are building and why) ·
`docs/reviews/` (open review findings — **deliberately not included here**;
this document describes the design as intended, and will be revised as those
findings are settled).

---

## What the machine is for

A robot policy gets a score — "87% on LIBERO." That number hides the *shape* of
the failure. This pipeline's job is to replace one number with a map: under what
conditions does the policy fail, why, how often, and what data would fix it.

Every design choice below exists to make one of those four claims **defensible
rather than plausible**.

---

## The diagram

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

---

## Stage 1 — Three things enter an episode

```
seed ──┐
       ├──► Env.reset(seed, spec)   + policy identity
spec ──┘
```

An episode is fully determined by three inputs and nothing else:

| Input | Role | Example |
|---|---|---|
| **seed** | *nuisance* variation — sampling the task distribution | where the object happens to sit this time |
| **`PerturbationSpec`** | the *treatment* — what we are testing | camera yawed 9°, two distractors |
| **policy identity** | the subject under test | `vla-adapter@<hash>` |

Same three in ⇒ same **outcome distribution** out. Note *distributional*, not
byte-identical: bit-exactness holds on the toy and will not survive a real VLA
(GPU nondeterminism, sampling action heads, contact chaos — `ARCHITECTURE.md`
§5.1). Probes therefore compare distributions over N seeds against a measured
noise floor, never single trajectories.

### The seed/spec split is the core experimental-design move

The **seed** is what you vary to get a *sample*. The **spec** is what you vary to
get an *effect*. Confusing the two is how robustness evaluation usually goes
wrong: "we ran 100 random scenes" yields an average, not a boundary.

### Two kinds of knob

Undocumented elsewhere, but behaviourally distinct:

| | Applied | Examples |
|---|---|---|
| **Initial-state** | once, at `reset` | `ee_offset_x_m`, `object_shift_m`, `distractor_count` |
| **Observation-channel** | every `_obs()` call, i.e. every step | `camera_yaw_deg`, `pixel_noise_std` |

Knobs are **frozen for the episode** — `PerturbationSpec` is a frozen dataclass.
There is no mid-episode intervention.

### The rule that licenses every downstream claim

Perturbations may touch the **observation** and the **initial state**. They may
never touch the object's true position, the goal predicate, or the dynamics.
That is what makes "the rate dropped, so the perturbation caused it" a valid
inference. If a knob could move the goalposts, a rate drop would be
uninterpretable.

---

## Stage 2 — The episode loop

```
Observation ──┬─► .policy_view()  ─► Policy ─► Action ─► Env.step ─┐
              └─► FULL state (detectors only)                      │
                              ▲                                    │
                              └────────────────────────────────────┘
```

Closed loop: the env shows an observation, the policy acts, the world moves,
repeat until the goal predicate fires or the step budget runs out.

### The fork is the important part

Env state contains two classes of thing:

- **What a real robot could know** — joint positions, gripper state, its own
  action history, camera extrinsics from calibration. Legitimate policy input.
- **What only a simulator knows** — the object's exact pose, exact distances.
  Prefixed `_gt_`.

`policy_view()` strips every `_gt_` key before the policy sees it. Detectors
receive the full state via `Step.obs_state`.

**Why this matters more than it sounds.** A policy that peeks at the true object
pose scores beautifully in simulation and collapses on hardware — and you will
not know, because the number looks fine. This is the most common way simulated
evaluation lies. The architecture makes it *impossible* rather than
discouraged.

> Caught during the build: `grasp_attempts` was originally `_gt_grasp_attempts`
> and read by the policy. It was simply misnamed — a robot does know its own
> action history — but under the old unenforced convention it read as a leak.
> Adding `policy_view()` turned "remember not to do this" into "cannot do this."

### Closed-loop compounding

Actions are sequentially dependent. A perturbed observation at t=0 changes
action 0, which changes the state at t=1, which changes the next observation. A
1 mm perceptual error can become a 10 cm terminal error.

This is why **single trajectories are never compared**, and why trajectory
divergence is a *secondary* signal — successful manipulation admits many valid
trajectories, so separation is not failure.

---

## Stage 3 — The `Rollout` lands in the store

One episode becomes one `Rollout`: every step's observation and action, the
outcome, the seed, the spec, and a **fingerprint** of what produced it. Appended
to `runs/<run_id>/rollouts.jsonl` — append-only, greppable, content-addressed.

### Why content-addressing is load-bearing

A full campaign is thousands of episodes over many hours. It will crash. Resume
has to be safe — and "safe" means a cache hit must be **the same experiment**,
not merely the same label.

Identity is derived from the load-bearing *values* (constructor kwargs,
thresholds, checkpoint revision, control mode, action space), never from a git
SHA — a SHA is simultaneously too coarse (a README edit invalidates everything)
and too weak (a dirty tree keeps the SHA stable while code changes underneath
it). The fingerprint is **stored as well as hashed**, so a cache hit can be
*verified* rather than trusted, and a mismatch can name the field that moved.

### What the cache actually is

Once traces stop being bit-reproducible, the cache is no longer "avoiding
redundant deterministic work" — it is **reusing samples from a distribution**.
Two consequences: the fingerprint must cover anything that shifts the
*distribution*, not merely anything that changes a trace; and a cache entry can
never be verified by re-running it.

Hence the split: `identity` (documented semantic effect — inside the key),
`semantic_runtime` (e.g. MuJoCo version — inside the key), `runtime`
(python/torch/numpy — compared and *reported*, never fatal, because keying on
them destroys cross-machine resume).

---

## Stage 4 — `classify()` — the *why*

The env has already said pass/fail. This layer says **what kind of failure**.

It reads privileged state and applies deterministic rules, most-specific first:

| Observation | Family |
|---|---|
| never entered pre-grasp | `planning` |
| terminated at a distractor, not the named target | `spatial_reasoning` |
| ended far from everything | `visual_grounding` |
| repeated attempts in the same spot | `recovery` |
| reached the target but the grasp failed | `manipulation` |
| no rule matched | `ambiguous` (`confident: False`) |

### The separation that keeps it honest

**The env owns `success`. The miner owns `family`. The miner can never influence
`success`.** If it could, you would be measuring your detector rather than the
policy.

### Tiering

- **Tier 1** — deterministic detectors. Reproducible. Headline numbers filter to
  this.
- **Tier 3** — an LLM (or human) judge, consulted *only* where tier 1 is not
  confident, and it **never overrides** a tier-1 verdict.

Every label records its `source`, so a reader can tell which numbers are
reproducible. Keeping the LLM out of the measurement path is a deliberate
decision, made for audit-trail reproducibility rather than cost.

---

## Stage 5 — `cluster()`

Group failures by **(family, active knobs)**. Deliberately simple — no
embeddings — because the differentiator is *auditability*, not clustering
novelty. A client can check a cluster by hand.

Each cluster carries its count, task scope, example rollout ids, mean final
error and observed terminal behaviours.

---

## Stage 6 — The two experiments

These are constantly conflated and they answer different questions.

### `sweep()` — *"where does it break?"*

Hold everything at nominal, vary **one** axis across a ladder of levels, N seeds
per level. Output: a dose-response curve.

```
pixel_noise_std=0.02   95.0%  [76.4%, 99.1%]  n=20
pixel_noise_std=0.04   70.0%  [48.1%, 85.5%]  n=20
pixel_noise_std=0.06   45.0%  [25.8%, 65.8%]  n=20
pixel_noise_std=0.08   35.0%  [18.1%, 56.7%]  n=20
```

The ladder is hand-chosen rather than randomly sampled *because the deliverable
is a boundary*. Random knob values give a scatter with one episode per value: no
interval anywhere, no boundary, no ability to say a cell differs from nominal.
This is a designed dose-response experiment, not a Monte Carlo.

### `counterfactual_probe()` — *"which of these broke it?"*

Start from a **failing, fully-perturbed** cell. Revert **one** knob to nominal.
Re-run **the same seeds**. If success jumps, that knob was the cause.

### Why both are needed

A sweep gives a boundary on one axis *in isolation*. A probe says which axis
mattered when several move at once — which is what deployment looks like. A
sweep cannot do attribution; a probe cannot draw a curve.

### Why "the same seeds" is the clever part

Both cells run the **identical task instances**, so the comparison is not
polluted by which scenes happened to be drawn. That is a *paired* experiment,
and pairing is free variance reduction — the strongest statistical property
available here, and the reason `PLAN.md` §9.2 can promise paired tests at no
extra episode cost.

### The noise floor

A knob is attributed only when its delta clears both a fixed threshold and
`3 × floor`, where the floor is measured by `reproducibility_floor()` — run-to-run
variance at a *fixed* knob setting. On the toy the floor is 0; on a real VLA it
will not be, and a delta that cannot beat the noise it sits in is not evidence.

---

## Stage 7 — `find_boundary()`

Scan the curve for the knob value where success first falls below a stated
fraction of nominal. Report a **bracket** — "between 0.04 and 0.06" — never an
interpolated point.

**Why refuse to interpolate:** with n=20 per cell you can honestly say "between
L2 and L3." You cannot say "at 13.4 degrees." Interpolating a precise number
from noisy cells is false precision, and the boundary is the field most likely
to be quoted back at you.

The planned adaptive arm replaces the grid scan with a surrogate fit
(logistic/GP over magnitude → success), which yields a **posterior interval** on
the boundary rather than a bracket, plus a principled stopping rule.

---

## Stage 8 — The manifest row

Each row is a **falsifiable claim with its evidence attached**: family, trigger
(citing the probe result), boundary with the definition used, n, interval,
supply-side corroboration, the coverage required to remediate, and the
validation plan.

### Three properties that make this an artifact rather than an opinion

1. **A trigger without a counterfactual is downgraded to `correlational` in the
   output.** The system says out loud when it is guessing.
2. **`validation_result: null` renders as `UNTESTED CLAIM`** — visible to the
   reader, never hidden. Overstating an untested row is the fastest way to lose
   a client's trust.
3. **`coverage_required` specifies diversity axes, never demonstration counts.**
   Generalization follows a power law in the number of *distinct environments
   and objects*, not in raw demo count (Lin et al., ICLR 2025), so "collect 500
   more demos in the failing yaw band" buys depth on the axis with diminishing
   returns.

### Severity is not shipped as a number

The manifest reports **`failure_conditional`** — P(fail | condition), which we
measure and the counterfactual probe establishes. **`condition_prevalence` is
client-supplied** from their own deployment logs. Severity is computed at
delivery as `conditional × prevalence × failure_cost`.

The reason: P(fail | condition) is a mechanism claim and transfers moderately;
P(condition) is an artifact of *the perturbation grid we chose* and has no
reason to transfer at all. Shipping their product as one sim-measured number
would be ranking by the half that cannot transfer.

Two consequences visible in the flow: a region the **adaptive** arm found and
the uniform arm never sampled renders `prevalence: not estimated` — never a low
severity; and `privileged: true` rollouts never enter a reported number.

---

## Two sampling arms, never mixed

| Arm | Sampling | Feeds | Budget |
|---|---|---|---|
| **Uniform** | fixed grid, unbiased | anything frequency-weighted | ~40% |
| **Adaptive** | surrogate-guided | `boundary`, with a posterior interval | ~60% |

Active sampling optimises for *finding* failures, so it biases the discovered
failure population. Anything frequency-weighted must therefore come from the
uniform arm alone.

**The arm belongs to the REQUEST, not the episode.** `rollout_id` hashes
(identity, seed, spec) and correctly excludes the arm — the physics is identical
either way. Tagging the stored rollout would mean one rollout per cell carrying
whichever arm asked *first*, so the uniform arm would silently lose cells to
cache hits in adaptive search order and its sample would stop being uniform.
Hence `arms.jsonl`: frequency is computed over the cells the uniform arm **asked
for**, each resolved through the cache. One rollout legitimately serves both.

---

## The shape of the whole thing

```
inputs you control → closed loop → evidence → why → grouping → two experiments → claims
   (seed, spec)      (env owns      (Rollout)  (miner    (cluster)   (sweep +      (manifest)
                      success)                  owns why)             probe)
```

Read it as a chain of custody. Each stage may only add information the previous
stage licensed:

- the env may not know what the miner will conclude,
- the miner may not change what the env decided,
- the manifest may not claim more than the probe established,
- and every number carries the provenance needed to re-derive it.

Where that chain is broken, a weak inference arrives at the manifest formatted
identically to a strong one — which is why `correlational` vs `causal`, tier-1
vs tier-3, and `UNTESTED CLAIM` are *printed* rather than merely tracked.
