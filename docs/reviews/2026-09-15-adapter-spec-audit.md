# Adapter spec audit — before the LIBERO env / VLA-Adapter policy are built

**Date:** 2026-09-15 · **Reviewer:** session `critical`
**Audited:** `ARCHITECTURE.md` §8 ("Adding a real policy or env") and
`NEXT_STEPS.md` step 1, against the `Env` / `Policy` protocols as they stand
after Waves A/B/C (commit `ccb0fab`).
**Why now:** reviewing the spec is cheaper than reviewing the adapter.

| # | Finding | Severity |
|---|---|---|
| AS-1 | §8's checklist predates Waves A/B — it never mentions the three methods the protocol now requires | **high** |
| AS-2 | §8 says to mark EE pose `_gt_`. That is wrong and contradicts §3, the toy, and the policy's needs | **high** |
| AS-3 | LIBERO-plus may be a fixed corpus, not a knob API — which would break `sweep()` and the counterfactual probe | **high, unverified** |
| AS-4 | `identity()` must hash init-state file CONTENTS, not a path — F2's exact failure mode | **high** |
| AS-5 | Re-tuning `PhaseSegmenter` has no fixture to tune against, and step 2 as written is incoherent | **high** |
| AS-6 | Action chunking breaks the one-Action-per-call assumption and corrupts `forward_passes` | medium |
| AS-7 | `scene_descriptor()` needs a physical datum or the field is decorative; frames have no lifecycle or ignore rule | medium |

---

## AS-1 — §8 is stale and is the document that looks canonical

The `Env` protocol now requires **`identity()`**, **`semantic_deps()`** and
**`scene_descriptor()`**. §8 — the section titled *"Adding a real policy or
env"* — mentions none of them. It still describes the pre-Wave-A protocol:
`env_id`, `task_id`, `action_dims`, `reset`, `step`.

`NEXT_STEPS.md` step 1 does list them. So there are two specs, they disagree,
and the stale one has the canonical-sounding title. Whoever writes the adapter
will read §8 first.

Also missing from §8: nothing says `Policy.identity()` is now required, so the
VLA-Adapter policy is exposed to the same omission — and for a policy, a missing
`identity()` field means the checkpoint revision is not in the cache key, which
is finding #2 aimed directly at the Phase-5 before/after claim.

**Fix:** §8 becomes a pointer to one authoritative checklist, or is rewritten
against the current protocol. Not both.

---

## AS-2 — "Expose … EE pose … as `_gt_*`" is wrong

§8 item 2: *"Expose contacts, EE pose, object poses, BDDL predicate as `_gt_*`."*

`ARCHITECTURE.md` §3 states the rule correctly: *decide first whether a real
robot has it.* A real robot **has its own end-effector pose** — it is
proprioception, the most basic thing a manipulator knows. `ToyReachEnv`
implements this correctly (`ee_xy`, unprefixed).

Marking EE pose `_gt_` denies it to the **policy**, not to detectors. Every VLA
policy in scope consumes proprioceptive state; the adapter would either starve
it or quietly bypass `policy_view()`. The second is far more likely, and it
reintroduces exactly the convention-vs-enforcement weakness that caused the
original `_gt_` leak.

Same paragraph, second problem: **the BDDL predicate**. Exposing it to detectors
is fine. But `success` must continue to come from the env's return value (I3),
never from a detector reading the predicate — otherwise the miner is scoring
itself. Worth stating, because on LIBERO the predicate is right there and it
will be tempting.

**Correct partition for LIBERO:**

| Signal | `_gt_`? | Why |
|---|---|---|
| EE pose, joint states, gripper state | **no** | proprioception |
| Camera extrinsics | **no** | calibration — a real robot knows these |
| Object poses | **yes** | a real robot does not have them |
| Contact flags | **yes** | simulator truth |
| BDDL predicate value | **yes** | detectors only; never the source of `success` |

---

## AS-3 — Is LIBERO-plus a generator or a corpus? *(unverified, highest priority)*

**I cannot settle this from here and it should be checked before any adapter
code is written.**

Everything downstream assumes **continuous knobs we set**:

- `sweep()` chooses its own ladder — `camera_yaw_deg` at `[0,3,6,9,12,15,20,25]`
- `find_boundary()` brackets between adjacent levels we picked
- `counterfactual_probe()` **reverts one knob while holding the others fixed**

But LIBERO-plus is described everywhere in our own documents as *"10,030 task
instances across seven perturbation factors and 21 sub-dimensions, stratified
into L1–L5"*. That is the vocabulary of a **fixed corpus of pre-generated
instances**, not a parameterised API.

If it is a corpus, then:

- `sweep()` cannot choose levels — resolution is capped at whatever L1–L5 gives,
  and `find_boundary()` inherits that granularity.
- **`counterfactual_probe()` may not be implementable at all.** "Revert
  `camera_yaw` while holding `distractor_count`" requires the corpus to contain
  the exact instance with one factor reverted and the rest identical. A
  stratified sample almost certainly does not guarantee that.
- The seed pairing that C7's McNemar depends on requires the *same task
  instance* under two conditions. Same problem.

That matters more than a resolution limit: **counterfactual attribution is one
of the three things `docs/LANDSCAPE.md` says is genuinely ours.** If the
benchmark cannot support it, either we generate perturbations ourselves against
vanilla LIBERO — which changes "stress generation is an integration task rather
than a contribution" (proposal §4) — or the differentiator is unavailable on
this benchmark.

**Action:** read `github.com/sylvestf/LIBERO-plus` and answer one question — can
a caller construct an instance with specified factor values, or only draw from
the 10,030? A day of checking now versus discovering it after the adapter exists.

---

## AS-4 — `identity()` must hash init-state CONTENTS

`NEXT_STEPS` step 1 lists `init_states` among the `identity()` fields, which is
right. *How* it is recorded decides whether it works.

F2 is the worked example: MuJoCo 3.4.0 did not change LIBERO's init-state
**files** — it changed how those **stored states** behave, 80% → 28% on one
task. A path or a suite name is a *label*, and serving a cached rollout because
two labels match is finding #2 verbatim.

**Record a hash of the init-state file contents**, plus the demo-dataset
revision already pinned (`a1aaacb7…`). The MuJoCo half is separately handled by
`semantic_deps()` — both are needed, and they catch different things:
`semantic_deps` catches the simulator moving under fixed states; the content
hash catches the states moving under a fixed simulator.

---

## AS-5 — There is no fixture to re-tune `PhaseSegmenter` against

§8 item 5 and `NEXT_STEPS` step 2 both call this non-negotiable:
*"Re-run the oracle gate against the new `PhaseSegmenter` config."*

As written this cannot be done. The oracle gate is `ToyReachEnv` +
`ScriptedReachPolicy`. Its geometry is a 0.7 m reach with a 0.06 m grasp radius;
LIBERO radii would **fail** on it, and toy radii are meaningless in LIBERO.
Running the toy gate with LIBERO thresholds tests nothing.

What the step actually requires is a **LIBERO fixture**: a scripted expert that
solves a LIBERO task, plus plantable faults with known families and known
trigger factors — the LIBERO equivalent of `GROUND_TRUTH` in
`policies/scripted.py`. **Nobody has scoped that.** It is real work, it is on the
critical path, and it is the only thing that can tell you whether the detectors
mean anything in the new env.

This is the deepest gap in the spec. Without it, LIBERO detector thresholds are
tuned against nothing, which is precisely the failure §8 item 5 exists to
prevent — *"detector thresholds are the part most likely to silently break on a
new env."*

**Options, in the order I would consider them:** (a) a scripted/waypoint expert
on one or two LIBERO tasks with 2–3 planted faults — real work, but it is the
only honest version; (b) validate thresholds against human-labelled traces and
report κ, weaker and slower; (c) declare detectors provisional and mark every
tier-1 family `confident=False` until a fixture exists — which at least does not
lie.

---

## AS-6 — Action chunking breaks the per-call assumption

The `Policy` protocol returns **one `Action` per `__call__`**, and
`runner.rollout` increments `forward_passes` once per call.

Modern VLA policies emit action **chunks** — OpenVLA-OFT and π0 both do; whether
VLA-Adapter does is unchecked and should be, before the adapter is written. If
it chunks, the adapter must buffer and return one action per call (correct), and
then `forward_passes` silently becomes *actions consumed*, not forward passes.
It feeds the §8 cost table, so the campaign estimate would be wrong by the chunk
factor.

**Fix:** let the adapter report forward passes explicitly rather than having the
runner infer them by counting calls.

---

## AS-7 — Two smaller things that are cheap now

**`scene_descriptor()` needs a physical datum.** Its purpose (A3/DG-8) is that a
person can rebuild the scene on a bench. MuJoCo gives poses in the model frame.
Without a declared origin — robot base, table corner — metres are not
reproducible, and the field becomes decorative precisely when it is finally
needed. Declare the datum in the descriptor itself.

**Frames have no lifecycle and no ignore rule.** §8 item 3 requires writing
frames to disk with paths in `image_refs` (G2). Nothing writes or reaps them,
and `.gitignore` covers `runs/**/rollouts.jsonl` and
`experiments/repro/runs/**/videos/` but **not** frames written under `runs/`. At
400 episodes × ~300 steps × 2 cameras this is substantial, and the first
accident is committing it.

---

# Review — `libero_env.py` as built, and the demonstration-fixture proposal

**Added 2026-09-15**, after `primary` built the adapter and proposed using
LIBERO demonstrations as the detector-tuning fixture.

**AS-2 is correctly taken:** `eef_pos`, `eef_quat`, `gripper_qpos`,
`joint_pos/vel` are policy-visible; object poses and contact counts are `_gt_`.
The partition matches §3's rule.

| # | Finding | Severity |
|---|---|---|
| LE-1 | `identity()` records `init_states` as a BOOLEAN — AS-4 is not addressed | **high** |
| LE-2 | `_object_body_names()` fails silently in the direction its docstring says is safe | **high** |
| LE-3 | `success` falls back to `terminated` — conflates "episode ended" with "task succeeded" | **high** |
| LE-4 | `.npy` frames: ~36 GB per 400-episode campaign, no reaping, not gitignored | medium |
| LE-5 | `termination` vocabulary diverges from the toy with nothing normalising it | low |

---

## LE-1 — `init_states` is a flag, not a fingerprint

```python
"init_states": self.init_states,     # bool: whether to use them
```

AS-4 asked for a **hash of the init-state file contents**. This records whether
init states are enabled, not which ones. F2 is the exact reason that matters:
MuJoCo 3.4.0 did not change the init-state *files*, it changed how those *stored
states behave* — 80% → 28%. The converse is equally live: the files changing
under a fixed MuJoCo is invisible here, and `semantic_deps()` cannot catch it
because MuJoCo did not move.

Also absent: the **task-suite / asset revision**. The `Env` protocol docstring
names "asset revision" explicitly, and the BDDL files and meshes define the task.
The demo dataset is pinned (`a1aaacb7…`); the suite is not.

**Fix:** hash the init-state array bytes for the task, plus the LIBERO suite
revision, into `identity()`.

---

## LE-2 — the conservative direction is not the safe direction

```python
skip = ("world", "robot", "gripper", "table", "mount", "base",
        "controller", "link", "finger")
```

The docstring argues this is safe because *"a missing object costs a detector a
signal (it skips, per G8); a wrongly-included robot link would silently corrupt
every distance-based detector."*

**The first half does not hold.** G8 skipping requires the *key* to be absent.
Here `_gt_object_pos` is always present — just incomplete. So nothing skips.
Instead `_gt_eef_to_nearest_object` and `_gt_eef_to_object` are computed over a
**subset**, and every distance-based detector gets a confident wrong number.
Omission produces silence in the design G8 assumes and a wrong value here.

**And `"link"` is likely to exclude real task objects.** Articulated LIBERO
objects — drawers, cabinet doors, the microwave — commonly carry MuJoCo body
names containing `link`. For "open the top drawer", the drawer is the task
object and would be filtered out. `_gt_nearest_object` then names something
else entirely, and the miner's `spatial_reasoning` rule fires on it.

**Fix:** derive object names from the task's **BDDL object list** — the task
definition already enumerates them — rather than substring-filtering every body
in the model. Where that is unavailable, emit `_gt_object_pos_incomplete: true`
so detectors can skip as G8 intends rather than silently computing on a subset.

---

## LE-3 — `success` must never be inferred

```python
success = bool(info.get("is_success", terminated))
```

If `is_success` is absent, this treats **`terminated`** as success. `terminated`
means the episode ended — which includes ending badly. Invariant I3 says the env
owns `success` and it is "the only trustworthy signal"; a fallback that guesses
turns the most load-bearing field in the schema into an inference.

The failure mode is silent and maximally damaging: inflated success, a nominal
cell that looks healthy, and §6's "nominal below ~90% is a harness bug"
heuristic never triggers because the number is *too high*.

**Fix:** raise if `is_success` is absent. A missing success signal is a broken
adapter, not a default.

---

## LE-4 — frames

`.npy` per camera per step: 224×224×3 uint8 ≈ 150 KB raw, ×~300 steps ×2
cameras ≈ **90 MB/episode**, ≈ **36 GB** for a 400-episode screen. Nothing reaps
them, and `.gitignore` covers `runs/**/rollouts.jsonl` but **not** frames under
`runs/`. Encode as PNG (~3× smaller, lossless), add an ignore rule, and decide a
retention policy before the first campaign rather than during it.

---

## LE-5 — termination vocabulary

Toy emits `grasped`/`timeout`; this emits `success`/`timeout`/`""`.
`ARCHITECTURE.md` §4 documents the vocabulary, `cluster()` reports the set and
the manifest prints it. Two envs, two vocabularies, nothing normalising. Declare
the allowed values in L0 and have adapters map onto them.

---

## On "the demonstrations are the fixture"

**The instinct is right and this should be built.** ~50 human demos per task is
real ground truth that is already on disk, and it is free. Three qualifications,
one of which changes what the gate can claim.

### It is not a weaker oracle gate — it is only the CONTROL arm

The toy gate has two halves: *recover the planted trigger* (sensitivity) and
*CONTROL: a clean policy under perturbation must not generate rows*
(specificity). "No demonstration is classified as a failure" is the **second
half only**, and it is passed trivially by a classifier that labels everything
`success`. Necessary, not sufficient, and it supplies no evidence at all on the
failure path. Worth naming precisely so the gate is not read as equivalent.

### Thresholds from the success path are defensible — if framed as coverage

Fitting `grasp_radius` to eef-to-object distance at gripper closure uses **one
class** to set a **discriminative** boundary. That is fine *if* the definition is
explicitly "pre-grasp = the region from which successful grasps are observed to
occur" — a coverage statement, e.g. the 95th percentile of the demo
distribution. It is not fine if read as "the distance that separates
manipulation failures from grounding failures", which the demos cannot speak to.
State which one it is in the code, or the next reader will assume the second.

Suggestion that costs nothing: record the demo-derived distribution **in the
diagnosis**, so a reader sees where an episode fell relative to the
demonstrations. That turns a tuned threshold into evidence.

### Phase-order validation assumes pick-place, and LIBERO is not all pick-place

*"Demos must segment as approach → pre-grasp → grasp → transport → place →
release in that order"* holds for pick-and-place. LIBERO contains tasks that are
not — opening drawers, operating a stove, multi-stage `libero_10` tasks. The
proposal itself warns against this (`VLA Scenario Testing.md` §4 item 4: *"do not
assume every task exposes reach/grasp/transport/place predicates"*).

As written, the gate would **fail on legitimate demonstrations** of those tasks,
and the likely response is to loosen it until it passes — which destroys it.
Make the expected phase grammar task-conditional, and let a task with no
declared grammar skip the ordering check with a recorded reason (G8).

### The degraded-policy fixture recovers `trigger_ok`, but NOT `family_ok`

This is the important one. The toy's oracle gate checks two things per scenario:
the **family** the miner assigns, and the **trigger** the probe attributes.

A degraded-policy fixture in LIBERO gives a **known trigger** — we chose the
perturbation — so `trigger_ok` transfers. It does **not** give a known family. In
the toy, `nearest_object → spatial_reasoning` was ground truth because the bug
*was* "pick the nearest blob". Perturbing a real VLA's input tells you what you
changed, not which failure mode it will exhibit: a yaw perturbation may produce a
grounding failure, a grasp failure, or no failure at all, and the policy is under
no obligation to fail in the family we expect.

**So on LIBERO, `family_ok` has no fixture — not from demos (success path only),
not from degradation (trigger known, family not).** That is a genuine residual
and it should be recorded as a known limitation rather than closed by assumption.
It bears directly on the κ gate: κ measures whether two humans agree, which is
the *only* remaining check on family assignment once the planted-fault fixture is
gone. That raises κ from a nice-to-have to the load-bearing validation of the
classifier, and PLAN's κ ≥ 0.5 threshold should be reconsidered in that light.
