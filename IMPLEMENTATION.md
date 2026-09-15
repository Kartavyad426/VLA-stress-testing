# Pending Implementation

**What changes in the code, why, and what is genuinely new.**
Written 2026-09-11 after the design review (`docs/reviews/2026-09-11-design-review.md`,
findings DG-1..DG-11, all adjudicated). `PLAN.md` is the authority on *what we
build*; `ARCHITECTURE.md` describes the code **as built**; this document is the
bridge between them.

## Status — 2026-09-15

**Waves A, B and C7/C8 are BUILT and VERIFIED** (`experiments/wave_abc_test.py`,
7/7; oracle gate 3/4 + control PASS). Committed as `ccb0fab` + `5b0b8e5`.

| Item | State |
|---|---|
| A1 `arms.jsonl` · A2 `semantic_runtime` · A3 `scene_descriptor` | **DONE** |
| B4 `RegressionSet` · B5 severity split · B6 `PrivilegedProbePolicy` | **DONE** |
| C7 paired McNemar · C8 `failure_cost` | **DONE** |
| **C9 per-task environment control** | **NOT DONE** |
| **D10 surrogate `adaptive_sweep()`** | **NOT DONE** |

*(Numbering note: an earlier status message said "items 1-8 and 10-12" while
also listing C9 as outstanding. The prose was right and the numbering was
wrong. This table is authoritative.)*

Also built since: **`vla_harness/envs/libero_env.py`** — the LIBERO adapter
(NEXT_STEPS step 1, first half). Verified against `libero_spatial` task 0:
resets, emits 13 object poses and 7 camera extrinsics in metres, privileged
keys stripped from `policy_view()`.

---

## 0. Order of work

Sequenced by *retrofit cost* — how much more expensive each item becomes once
real traces exist — not by size.

| Wave | Items | Why this wave |
|---|---|---|
| **A — before any real run** | 1 `arms.jsonl` · 2 `semantic_runtime` + promotion · 3 `scene_descriptor` | All three are provenance/schema. Once a trace store exists they cannot be added retroactively — the information was never captured. |
| **B — before Phase 3** | 4 `RegressionSet` · 5 severity split · 6 `PrivilegedProbePolicy` | Needed by the artifacts Phase 3 produces; cheap now, structural later. |
| **C — Phase 3** | 7 paired statistics · 8 `failure_cost` · 9 per-task env control | Improve claims we are already making. |
| **D — Phase 3+** | 10 `adaptive_sweep()` + surrogate | Largest piece; only pays off once there is a real boundary to find. |

**Schema version.** Waves A and B together are one break: **2.0 → 3.0**. Ship
them as a unit; do not bump twice. Note `arms.jsonl` alone does *not* require a
bump — it adds no `Rollout` field (that is the whole point of DG-2).

---

## Wave A — provenance and schema

### A1. `arms.jsonl` — the sampling arm is a property of the REQUEST

**Problem.** The two-arm design (`PLAN.md` §5.1) needs frequency statistics
computed over the uniform arm only. The obvious implementation — a
`sampling_arm` field on each `Rollout` — **de-uniforms the uniform arm.**
`rollout_id` hashes (identity, seed, spec) and correctly excludes the arm, since
the physics is identical. So the store holds one rollout per cell tagged with
whichever arm asked *first*; the uniform arm then loses cells to cache hits
tagged `adaptive`, and *which* cells it loses depends on adaptive search order.
That is precisely the corruption the split exists to prevent, introduced by the
mechanism meant to prevent it. Hashing the arm is worse — duplicate identical
physics, double the campaign, and it breaks the seed pairing A7 needs.

**New.**
```
runs/<run_id>/arms.jsonl     {"arm": "uniform"|"adaptive", "rollout_id": "...", "cell": {...}}
```
Many-to-many. One rollout can legitimately serve both arms.

**Changes.** `run_cell(..., arm=...)` appends a request record per requested
cell. A `frequency_over(arm)` helper resolves requested cells through the cache.
`Rollout` **unchanged**.

**Invariant (I9).** Frequency statistics are computed over cells the uniform arm
*requested*, never over stored-rollout tags.

### A2. `semantic_runtime` + a promotion policy

**Problem.** Provenance had two buckets: `identity()` (keyed) and `runtime`
(reported). F2 showed MuJoCo 3.4.0 moving one task 80% → 28% — so a simulator
version must **miss the cache**, not warn. The rule proposed for that —
*"anything that changes the ANSWER is keyed"* — **is not decidable**: torch and
numpy change answers too, via reduction order → contact chaos
(`ARCHITECTURE.md` §5.1 already says so). Applied honestly it keys torch, every
pip upgrade invalidates the store, and we are back at the git-SHA failure mode
we rejected.

**New — three buckets. The distinction is epistemic, not causal.**

| Bucket | Holds | Keyed |
|---|---|---|
| `identity()` | env/policy **design** | yes |
| `semantic_runtime` | **documented, specific** semantic effect: `mujoco`, `MUJOCO_GL`. Enumerated per adapter. | **yes** |
| `runtime` | `torch`, `numpy`, `python`, platform | no — reported |

`semantic_runtime` is separate from `identity()` because a simulator build is
not part of an env's task *design* — `ToyReachEnv.identity()` declaring a MuJoCo
version it never loads would be incoherent.

**Promotion** `runtime` → `semantic_runtime` requires documented evidence and is
a deliberate, rare, store-invalidating event, logged like a schema bump. MuJoCo
qualified via F2; torch has not.

### A3. `scene_descriptor` — the field that makes PPI possible later

**Problem.** `PLAN.md` §9.2 records a PPI-ready schema as adopted "because it is
expensive to retrofit," and never says what the form *is*. PPI pairs a sim
outcome with a real outcome **for the same scene**, so it needs a scene
identifier a person could rebuild on a physical bench. `seed` + `spec` is
sim-internal *by construction* and cannot supply that.

**New — emitted per rollout at reset, in physical units:**
```
scene_descriptor:
  objects:  [{name, pos_m: [x,y,z], quat: [w,x,y,z]}]
  camera:   {pos_m, quat, fov_deg}
  lighting: {intensity, direction}
```

Cheap while writing the env adapter. Genuinely unreconstructable from a trace
store afterwards — which is the definition of retrofit-expensive.

---

## Wave B — artifacts Phase 3 depends on

### B4. `RegressionSet` as a first-class L0 artifact

**Problem.** The frozen regression set is the comparison basis for **all of
Phase 5** and carries the project's headline before/after claim. It has no
module, no schema type, no fingerprint. And after the identity fix it **expires
invisibly**: it is a list of (seed, spec) pairs whose meaning depends on env
identity, which now changes whenever `grasp_radius`, `max_steps`, MuJoCo or the
rendering backend moves. Frozen in Phase 3, re-run in Phase 5, measuring a
different thing, undetected. That is the rollout-identity bug one level up.

**New.**
```
RegressionSet: {id, frozen_env: <env identity>, frozen_at, members: [rollout_id]}
```
**Invariant (I12).** Re-running against a different `frozen_env` requires a loud
explicit override — never a silent pass.

### B5. Severity split — `failure_conditional` vs client-supplied prevalence

**Problem, two-in-one.** (a) With frequency filtered to the uniform arm, a
failure mode found *only* by the adaptive arm has count ≈ 0 → ranked low → the
manifest systematically buries the output of its own most novel component.
(b) Severity ordering is the property the sim-to-real literature says transfers
*worst*, and severity is what the manifest prioritises by.

Both have one cause: severity conflates **P(fail | condition)** — a mechanism
claim we measure, which the counterfactual probe establishes, and which
transfers moderately — with **P(condition)**, which in sim is an artifact of
*the perturbation grid we chose* and has no reason to transfer at all.

**New.** Ship `failure_conditional` (+ CI). `condition_prevalence` becomes a
**client-supplied** column from their deployment logs. Severity is computed **at
delivery** as conditional × prevalence × `failure_cost`.

Honest — we never claim to know their prevalence — and commercially *stronger*:
the artifact becomes something the client's data completes. Structurally
identical to `failure_cost`, which is already client-specific for the same
reason.

**Invariant (I11).** A region the uniform arm never sampled renders
`prevalence: not estimated` — **never** a low severity. Silent demotion and
honest abstention look identical in a ranked table and are different claims.

### B6. `PrivilegedProbePolicy`

**Problem.** §7c discriminator 4 ("give it ground-truth object pose") is
**unrunnable**: `policy_view()` strips every `_gt_` key, and that is *enforced*,
not conventional. Two documents in direct collision.

**New.** An opt-in wrapper receiving full state, stamping `privileged: true` on
every rollout, excluded from headline numbers by the same filter that excludes
`tier3`. `policy_view()` is **unchanged** — it is the architecture's best
feature and a flag on the strict path would reintroduce exactly the
convention-vs-enforcement weakness that caused the original `_gt_` leak.

**Invariant (I13).** `privileged: true` rollouts never enter a reported number.

---

## Wave C — claims we are already making

### C7. Paired statistics in `counterfactual_probe`
We run the **same seeds** in both arms and currently discard the pairing,
comparing rate to rate. McNemar on paired per-seed outcomes is strictly more
powerful — free variance reduction, which directly relieves the probe budget.

### C8. `failure_cost` in `classify()`
Marked ✱ mandatory in the schema and **not implemented**, so by the schema's own
rule **no row the harness can emit today is a valid manifest row** (DG-10). It
is also rated our most novel contribution. Terminal state is already in the
trace: benign timeout vs dropped object vs collision.

### C9. Environment control runs per task
F2's lesson: at suite level, task 5's 80% → 28% collapse dilutes into noise.
§7c discriminator 1 must run **per task** or it does not catch the thing it
exists to catch.

---

## Wave D

### D10. `adaptive_sweep()` + surrogate
Logistic or GP over (perturbation magnitude → success). Gives `boundary` a
posterior interval instead of a Wilson interval on whichever grid cell happened
to straddle the transition, plus a principled stopping rule. A few hundred lines
of CPU code.

---

## Not code — decisions that changed the plan

| | Change | Where |
|---|---|---|
| DG-1 | **10/task is a SCREEN, 50/task is the GATE.** A 5 pp break passes at n=100 **56%** of the time. Screen results are never quotable as "consistent with published". Escalation unconditional for anything reported. Multiple-comparison correction applied (4 suites ⇒ 18.5% spurious-failure risk). | `PLAN.md` §5.3 |
| DG-9 | Three arms at **one** budget; data-response curve on arm A only. **5 fine-tunes, not 9.** The fixability contrast is (A or B) vs C — data vs not-data — and arm C is a **point, not a curve**; say so. | §6.1 |
| DG-11 | `language_grounding` is **out of scope** — no env implements the axis, no detector produces it. κ and the E5 comparison over **6** families, not 7. | §6 |
| DG-10 | ✱ fields now carry implementation status. | §9 |
| DG-5 | **Anchor decision BLOCKED on a smoke test.** π0.5 is 6–7 GB; lerobot#3098 documents CUDA/EGL contention on this hardware class. If it will not run here the anchor does not exist and §5.2 promises an impossible fallback. **A contingency nobody has smoke-tested is not a contingency.** | §5.2 |

---

## Open, not blocking

- Which LLM for tier-3 adjudication; local vs API (audit-trail reproducibility,
  not cost)
- Remediation data source for arm A: scripted demonstration vs re-rendering
- Do recovery rows ship in a first client manifest? The family is
  **unvalidated** — the toy fixture could not isolate it (noise re-rolls per
  step, so repeating a grasp wins there and would not on real hardware)
- Readout audience: technical vs commercial
