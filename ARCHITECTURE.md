# Implementation & Data Flow

**Purpose:** who produces which data, who consumes it, and where to look when a
number is wrong. Companion to `PLAN.md` (what we are building and why).

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

**The one rule that keeps this honest:** the mining layer imports nothing from
`envs/` or `policies/`. It consumes `Rollout` objects. If you ever need an env
import in `mining/`, the signal you want belongs in `Observation.state` instead.

---

## 2. Flow

```
   seed ──┐
          ├──► Env.reset(seed, spec) ──► Observation ──► .policy_view() ──► Policy
   spec ──┘                ▲                                                  │
                           │                                               Action
                           └───────────── Env.step(action) ◄──────────────────┘
                                              │
                              (loop until done: success | timeout)
                                              │
                                              ▼
                                         Rollout                    ──► TraceStore
                                              │                          (jsonl)
                        nominal reference ────┤
                                              ▼
                                     classify()  ──► rollout.diagnosis
                                              │
                                              ▼
                                        cluster()  ──► [cluster]
                                              │
   sweep()  ──► [Cell] ──► find_boundary() ──► boundary ──┐
                                                          ├──► manifest.make_row()
   counterfactual_probe() ──────────────► probe ──────────┘         │
                                                                    ▼
                                                       DATA_GAP_MANIFEST.md
```

Three things enter a rollout: **seed**, **PerturbationSpec**, **policy**. Nothing
else. Same three in ⇒ byte-identical trace out (G5). If that ever stops being
true, counterfactual probes become noise and every attribution in the manifest
is void. It is the load-bearing invariant.

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
| I8 | Clean policy + perturbation ⇒ near-100% success | perturbation broke the task, not the policy |

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
