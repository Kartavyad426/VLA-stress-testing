# Components — how each piece works and where it lives

**Written 2026-09-15.** Companion to `ARCHITECTURE.md` (data flow) and
`WHAT_OUR_CODE_DOES.md` (why we built any of it).
Every row states its **verification status**, because several pieces are built
and unproven outside the toy.

---

## Status at a glance

| | Component | File | Works on toy | Works on LIBERO |
|---|---|---|---|---|
| L0 | Rollout contract | `schema.py` | ✅ | ✅ |
| L0 | Identity / fingerprint / cache | `schema.py` | ✅ | ✅ |
| L0 | `ArmLog` | `schema.py` | ✅ | untested |
| L0 | `RegressionSet` | `schema.py` | ✅ | untested |
| L1 | Scripted oracle | `policies/scripted.py` | ✅ | n/a |
| L1 | `PrivilegedProbePolicy` | `policies/privileged.py` | ✅ | untested |
| L1 | **Real VLA policy adapter** | — | — | **NOT BUILT** |
| L2 | Toy env | `envs/toy.py` | ✅ | n/a |
| L2 | LIBERO env | `envs/libero_env.py` | n/a | ✅ capture only |
| L2 | **LIBERO-plus env** | — | — | **NOT BUILT** |
| — | Runner: rollout / sweep / probe | `runner.py` | ✅ | ✅ rollout |
| L3 | Phase segmentation | `mining/phases.py` | ✅ | ❌ **abstains** |
| L3 | Classification + `failure_cost` | `mining/classify.py` | ✅ | ❌ **abstains** |
| L3 | Clustering | `mining/cluster.py` | ✅ | untested |
| L4 | Manifest | `manifest.py` | ✅ | untested |

---

## L0 — the contract (`schema.py`, 517 lines)

**`Observation` / `Action` / `Step` / `Rollout`.** Actions are
`list[float]` + named dims, never a fixed tuple, so 3-DoF toy and 7-DoF LIBERO
and 14-DoF bimanual all work unchanged (G1). Images are paths, never pixels
(G2). The **last `Step` is terminal with `action=None`** — it carries the
observation in which success was decided, which nothing requested an action for.

**`Observation.policy_view()`** strips every `_gt_` key. This is *enforced*, not
conventional: `runner.rollout()` passes the stripped view unless the policy
declares `privileged = True`. A policy that read `_gt_` state would post a
success rate that collapses on hardware.

**Identity and the three provenance buckets.**

```
identity()         env/policy DESIGN                     -> KEYED
semantic_runtime   documented-semantic externals         -> KEYED
                   (mujoco, robosuite, MUJOCO_GL)
runtime            torch, numpy, python, platform        -> reported only
```

`rollout_id = hash(policy_id, env_id, task, seed, spec, semantic)`. A MuJoCo
version bump therefore **misses the cache** rather than warning — because F2
showed 3.4.0 taking one task 80% → 28%. torch stays reported: it *might* change
answers, and reporting is the proportionate response to "might". Promotion
between buckets requires documented evidence.

**`ArmLog`** (`arms.jsonl`). Which sampling arm *requested* which cell, many-to-
many. Not a field on the Rollout: `rollout_id` correctly excludes the arm since
the physics is identical, so tagging the stored episode would let the uniform
arm lose cells to cache hits in adaptive search order.

**`RegressionSet`.** Members plus the `frozen_env` and `frozen_semantic` it was
frozen against, and `check_against()`. Re-running under a different identity is
a loud override, never a silent pass.

**`TraceStore`.** Append-only JSONL, greppable, content-addressed, resumable —
and a cached rollout still *contributes its outcome* (skipping the compute is
correct; skipping the result made every resumed cell read 0%).

## L1 — policies

**`ScriptedReachPolicy`** — the oracle. Four plantable faults, each with a known
family and trigger, so the miner can be checked against ground truth.

**`PrivilegedProbePolicy`** — §7c discriminator 4. Receives full state, stamps
`privileged: True`, excluded from headline numbers like `tier3`. Verified on the
toy: a camera-misaligned policy goes **0% → 100%** with ground-truth object
pose, correctly reading *perceptual, upstream of control*.

**Not built: a real VLA policy adapter.** SmolVLA has been run through
`lerobot-eval`, never through our loop.

## L2 — environments

**`ToyReachEnv`** — 3-DoF reaching, LIBERO-plus-shaped knobs, no simulator.

**`LiberoEnv`** (`envs/libero_env.py`, 345 lines) — wraps LeRobot's `LiberoEnv`.

- `identity()` carries suite, task, `control_mode`, `hard_reset`, `max_steps`,
  and content hashes of **the init-state array and the BDDL file** — because the
  files changing under a *fixed* MuJoCo is invisible to `semantic_deps()`.
- Task objects come from **`obj_of_interest`, the BDDL object list**, not from
  substring-filtering body names. The earlier heuristic filtered out anything
  containing `link`, which is exactly what articulated LIBERO objects — drawers,
  cabinet doors, microwaves — are named. `_gt_object_pos_complete` lets
  detectors skip rather than compute over a subset.
- `success` is read from the env and **raises if absent**. It never falls back
  to "the episode ended", which would inflate every rate silently and upward —
  past the reach of the "nominal below 90% is a harness bug" heuristic.
- `scene_descriptor()` emits object poses and camera extrinsics in **metres**.
- Unsupported perturbation knobs **raise**. Vanilla LIBERO has no perturbation
  machinery, and a silently-ignored knob would produce a robustness curve for a
  perturbation that never happened.

**Verified:** `libero_spatial` task 0 — resets, steps, 2 BDDL objects resolved,
7 cameras, no `_gt_` leakage, full rollout through our runner (26 steps,
terminal action `None`, all four fingerprint buckets present).

## The runner (`runner.py`, 344 lines)

`rollout()` — one episode; emits the terminal observation; stamps fingerprint,
scene descriptor and privileged flag.

`run_cell()` / `sweep()` — the **uniform** arm. Verified cache behaviour,
fingerprint comparison, loud stale-hit reporting, `strict` mode.

`counterfactual_probe()` — re-runs the **same seeds** with one knob reverted and
reports **paired McNemar**. Verified on the toy: p = 0.00049 on 12 discordant
pairs, where a rate comparison said only "+100 pp".

`reproducibility_floor()` — takes **no store, deliberately**; a cache would
serve the second repeat from the first and report a floor of 0 by construction.

`uniform_frequency()` — frequency over cells the uniform arm *requested*,
resolved through the cache.

## L3 — mining (295 lines) — **the differentiator, and the part not yet working on LIBERO**

**`phases.py`** — segments a rollout into approach / pre-grasp / grasp /
transport / retry from simulator state. Each detector **declares the state keys
it needs** and skips with a recorded reason if they are absent (G8).

**`classify.py`** — deterministic rules, most specific first: never reached →
planning; ended at a distractor → spatial_reasoning; went confidently to the
wrong place → visual_grounding; repeated attempts within a small radius →
recovery; reached but grasp failed → manipulation. Plus `failure_cost()` and a
narrow tier-3 LLM seam that never overrides a deterministic verdict.

**`cluster.py`** — groups by (family, active knobs).

> ### Current LIBERO status: ABSTAINS, correctly
>
> First real rollout produced:
> `missing state keys: ['_gt_ee_to_obj', 'gripper', 'holding']`
>
> The detectors are tuned to toy key names; LIBERO emits `_gt_eef_to_object`,
> `gripper_qpos`, and has no `holding` at all (it needs deriving from gripper
> closure plus object lift). **It abstained rather than crashing or computing a
> wrong number** — G8/I7 working on first contact with a real environment.
>
> Making this work is the next real task, and per `ARCHITECTURE.md` §8 the
> oracle gate must be re-run against the new configuration.

## L4 — manifest (`manifest.py`, 166 lines)

Rows carry `failure_conditional` (ours, measured, + Wilson CI),
`condition_prevalence` (**client-supplied**, or `not estimated`), `failure_cost`,
`fixability`, `non_data_fix`, `discriminators`, boundary, evidence and
provenance. **No cardinal severity is emitted** — it is computed at delivery as
conditional × prevalence × cost.

## Tests

| File | Covers |
|---|---|
| `experiments/oracle_test.py` | The acceptance gate. 3/4 planted faults recovered + control PASS. |
| `experiments/wave_abc_test.py` | Provenance and artifacts. 7/7. |

---

## Are we in a complete running state?

**No — and precisely which part is missing is now known.**

**Running today:** the toy end-to-end (stress → mine → cluster → manifest);
`lerobot-eval` standalone on SmolVLA; the LIBERO adapter's capture path through
our runner.

**Missing before a real experiment:**

1. **A LIBERO `PhaseSegmenter`** — key mapping plus a derived `holding`. Without
   it the mining layer abstains on every LIBERO rollout.
2. **A real VLA policy adapter** — SmolVLA or VLA-Adapter through our loop.
3. **C9** per-task environment control.
4. **The LIBERO-plus adapter** — nothing perturbable exists yet, so no sweep,
   no boundary, no counterfactual probe on real data.
5. **D10** the surrogate (only needed once a boundary exists).

Items 1 and 2 are the gate on producing any real result. Item 4 gates everything
about *robustness* specifically — today we could only measure nominal success.
