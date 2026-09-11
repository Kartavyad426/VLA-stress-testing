# Review — `ARCHITECTURE.md` §2 Flow, and the code it describes

**Date:** 2026-09-11 · **Scope:** the rollout → classify → cluster → probe → manifest
path, checked against `vla_harness/` as built. Thesis and PLAN §0 treated as fixed.
**Method:** read + executed. Findings marked *[verified]* were reproduced by running
the harness; *[by inspection]* were read but not executed.

**Status key**

| Status | Meaning |
|---|---|
| `OPEN` | not yet discussed |
| `DISCUSSING` | under discussion, no decision yet |
| `ACCEPTED` | fix agreed and specified below; **not yet implemented** |
| `ASSIGNED` | agreed fix handed to an implementer; **not yet verified** |
| `FIXED` | implemented **and** re-verified by re-running the repro in this doc |
| `WONTFIX` | deliberately declined; reason recorded |

**#1 and #2 are `FIXED` as of 2026-09-11 (repros re-run, see the implementation notes under each). Everything else is not.** `ACCEPTED` and `ASSIGNED` mean the
argument is settled, not that the code changed. A finding moves to `FIXED` only
after the repro that produced it is re-run and no longer reproduces.

---

## Summary table

| # | Finding | Class | Severity | Status |
|---|---|---|---|---|
| 1 | Terminal observation never recorded | wrong now | high | **FIXED** — `primary` 2026-09-11 |
| 2 | `rollout_id` is not a content hash; cache serves stale traces | wrong now | high | **FIXED** — `primary` 2026-09-11 |
| 3 | `never_reached` unreachable; `grasp_radius` is a dead branch | wrong now | medium | **DISCUSSING** |
| 4 | `divergence_point` silently truncates against a shorter reference | wrong now | low | OPEN |
| 5 | `reproducibility_floor` never called; `floor_pp` always 0 | missing wire | **high** | **OPEN** — blocked on Q5.1 |
| 6 | "nominal reference" arrow has no producer outside the oracle test | missing wire | low | OPEN |
| 7 | `find_boundary` mislabels `cells[0]` as nominal | missing wire | medium | OPEN |
| 8 | Probe discards seed pairing; 20 pp threshold ≈ 1 SE at n=20 | statistics | high | OPEN |
| 9 | Boundary bracket ignores CIs and non-monotonicity | statistics | medium | OPEN |
| 10 | Probe finds main effects only; interactions fail silently | statistics | medium | OPEN |
| 11 | `TraceStore` is O(N²) and unbounded in memory | scales badly | high | OPEN |
| 12 | `grasp_radius` duplicated across the L2/L3 boundary | scales badly | medium | OPEN |
| 13 | "byte-identical trace" invariant is unfalsifiable | scales badly | low | OPEN |
| 14 | `rollout()` has no runner-side step cap | scales badly | low | OPEN |
| 15 | Goal predicate is weaker than the instruction; `TRANSPORT` unreachable | fixture gap | medium | OPEN |
| 16 | No mid-episode intervention: perception vs control cannot be separated | design limit | medium | OPEN |
| 17 | Shared RNG stream — `distractor_count` re-rolls the target's pixel noise | **confound** | **high** | OPEN |
| 18 | Seed jitter is 1-D: object slides on a diagonal, bearing varies 1.5° | fixture gap | medium | OPEN |
| 19 | Distractor layout has no seed dependence — Wilson CIs invalid on that axis | **statistics** | **high** | OPEN |

**Not at fault** (stated so silence isn't read as approval): the `_gt_` /
`policy_view()` enforcement, and the sweep-vs-probe conceptual separation.

**Suggested order:** 2 → 1 → 5 → 3 → 8. #2 and #1 corrupt data silently; #2
specifically threatens the retrain-and-re-measure result that PLAN §0 justifies
the small-policy trade with. #8 is where the methodological upside is.

---

## 1. The terminal observation is never recorded — *[verified]*

`runner.py:33-37` appends the `Step` *before* calling `env.step`, then breaks on
`done`. The observation in which the episode actually ended is discarded.

Consequences, measured on the fixture:

- `r.series("holding")` is `False` for every step of a **successful** rollout.
  The state transition the task is defined by is absent from the trace.
- `grasp_attempts` undercounts: recorded `0` vs true `1` on every successful seed.
- `final_error_m` is the distance one action *before* the end. With `step=0.15 m`
  that is a large displacement against `LOST_TARGET_M = 0.10`.

25 seeds, `unfiltered_grasp` @ `pixel_noise_std=0.15`, recorded vs true terminal error:

```
seed2   rec 0.0642 -> manipulation      true 0.1005 -> visual_grounding
seed4   rec 0.1280 -> visual_grounding  true 0.0287 -> manipulation
seed5   rec 0.0717 -> manipulation      true 0.2204 -> visual_grounding
seed17  rec 0.1933 -> visual_grounding  true 0.0877 -> manipulation
seed24  rec 0.0217 -> manipulation      true 0.1534 -> visual_grounding
```

10 of 23 failures change family. This is the fixture whose job is to give the
miner a known answer, and the answer is computed from a stale observation.

The 40% figure is inflated by the toy's large step size; the structural fact is
not toy-specific. On LIBERO the discarded frame is the most diagnostic one —
contact, object dropped, gripper closed on air.

**Direction:** append the post-step observation as a final `Step` before returning.

---

## 2. `rollout_id` is not a content hash — the cache serves stale traces — *[verified]*

`cell_hash` covers `(policy_id, env_id, task_id, seed, spec)`: identifiers, not
the code or config behind them. `ARCHITECTURE.md` §4 calls this
"content-addressed, so re-running is idempotent." It is addressed by *names*.

```
ToyReachEnv(grasp_radius=0.06)  -> hash 44960de79400, rate 1.00
ToyReachEnv(grasp_radius=0.001) -> hash 44960de79400, rate 1.00  (served from cache)
ToyReachEnv(grasp_radius=0.001) -> fresh store,       rate 0.00  (truth)
```

An env whose true success rate is 0% reports 100%, silently, because a constructor
argument outside the hash changed. Same applies to any edit to `toy.py`,
`scripted.py`, or a policy checkpoint.

**Why this is the worst one:** PLAN §0 justifies the small-policy trade by the
ability to fine-tune and re-measure. If the LoRA'd SmolVLA keeps
`policy_id = "smolvla"`, resuming into an existing store returns the *pre*-fine-tune
rollouts and the remediation step reports no change. The failure mode is a number,
not an exception.

**Direction:** fold a fingerprint into the hash — git SHA + hash of env/policy
constructor kwargs + checkpoint hash — or store it per-rollout and have `run_cell`
refuse a cached hit whose fingerprint differs. Same move already made for `_gt_`:
enforce in code rather than document.

### Discussion — #2 (2026-09-11)

Second opinion from session `primary`, which authored this code — treat as the
author conceding the bug, not as independent confirmation. Bug and repro accepted.

**Options 2 and 3 are not alternatives; take both.** #3 defines *what* the
fingerprint contains; #2 defines *when it is checked and what happens on mismatch*.
#3 alone means trusting a derived key that is never verified. #2 alone means
hand-maintaining the fingerprint at every call site, which rots. So: `identity()`
on the `Env`/`Policy` protocols returns the structured semantic fingerprint,
`env_id`/`policy_id` derive from it, and the fingerprint is stored on the `Rollout`
and compared on every cache hit.

**Option 4 (one store per fingerprint) rejected.** It trades away the thing that
motivated caching. Under session caps (Kaggle 9-12 h) resume across interruptions
is the point; killing cross-commit reuse means a threshold tweak at hour 8 costs
the whole campaign.

**Git SHA rejected outright — worse than no fingerprint.** Too coarse *and* too
weak at once. Too coarse: a README edit invalidates tens of thousands of rollouts.
Too weak, and this is the disqualifier: during Phase 3 the tree is dirty
constantly, so the SHA is *stable while the code changes underneath it*. That
manufactures confidence in a stale cache exactly when iteration is fastest. Keeping
a git component at all requires SHA + hash-of-diff, which is a slow content hash —
at which point hash the semantic values directly: constructor kwargs, detector
thresholds, checkpoint revision, control_mode, action space. Not source text.

**Correction (2026-09-11, after checking the code): the detector-churn half of the
argument below does not hold.** `store.append(r)` runs in `run_cell` *before*
anything classifies, so every persisted rollout carries `diagnosis=None` and is
re-classified on load; nothing in the codebase appends a classified rollout. A
`PhaseSegmenter` or `LOST_TARGET_M` change therefore shows a changed result today,
and identical rollouts after such a change are the cache working, not failing. The
Phase-3 framing survives for **env and policy configuration only** — which is what
the repro in this finding actually exercises. *Latent version:* `Rollout.diagnosis`
is a serialized field, so the moment anyone persists a classified rollout, detector
churn joins the problem, and the fingerprint covers env/policy identity only.

**Risk framing corrected — the fine-tune case is the *safer* one.** Phase 5 happens
once, under maximum scrutiny, with someone explicitly looking for a before/after
delta; a suspicious null gets investigated, and a fresh `run_id` is the natural
convention anyway. The live danger is **detector-threshold churn in Phase 3**:
someone widens `PhaseSegmenter(pregrasp_radius=...)` or nudges `grasp_radius`,
re-runs, sees identical numbers, and concludes their change had no effect. That
happens constantly, under no scrutiny, and nobody suspects the cache — they suspect
themselves. Note the repro in this finding is an *env constructor kwarg*, not a
policy swap, which supports this reading. Keep the fine-tune case in the writeup as
the one with commercial consequences; lead the fix's justification with config churn.

**On mismatch: re-run loudly, do not raise (by default).** Raising kills an
overnight sweep at hour 8 over a moved threshold; under session caps that is
expensive enough that someone disables the check, and a disabled check is worse
than none. Default to re-run, record a diff of *which fingerprint fields changed*,
and surface it in the run summary — not a log line. Add a strict mode that raises,
for client-facing runs where a silent 10x slowdown is the lesser problem.

**Fifth option — sampled audit, with a caveat that mostly kills it.** On cache hit,
re-run ~2% at random and check agreement. No fingerprint is ever complete (a MuJoCo
point release, a numpy change, an asset repack), and sampling catches gaps nobody
enumerated. **But this only works while rollouts are bit-reproducible, i.e. on the
toy.** Per §5.1, a real VLA is not: GPU nondeterminism, sampling action heads
(π0, GR00T), contact chaos.

**That caveat reframes the whole caching question, and belongs in the review:**
once traces are not reproducible, the cache is no longer "avoiding redundant
deterministic work" — it is "reusing samples from a distribution." Statistically
fine, but it means (a) the fingerprint must cover anything that shifts the
*distribution*, not merely anything that changes a trace, and (b) a cache entry can
never be verified by re-running it — the audit degrades to comparing success-rate
distributions at n>=20, which catches gross corruption and nothing subtle. State
this explicitly so nobody later assumes the cache is verifiable.

**Make resume visible.** A stale cache and a real finding look identical from
outside. §6 already carries the heuristic that a nominal cell below ~90% is nearly
always a harness bug; a resumed-vs-fresh split in the run summary is what makes
that check usable.

**Precedent to cite in the fix:** the `_gt_` prefix was a documented convention, the
policy violated it anyway, and the fix was structural — `policy_view()` makes
violation impossible rather than discouraged. "Remember to bump the id" is the same
convention that already failed once in this codebase.


---

## 3. `never_reached` is unreachable for the common case — *[by inspection]*

`phases.py:53-57`:

```python
elif d < self.grasp_radius:      labels.append(PREGRASP)
elif d < self.pregrasp_radius:   labels.append(PREGRASP)
```

Both branches emit `PREGRASP`. `grasp_radius` is a dead constructor arg; one
branch was presumably meant to be `GRASP`.

Downstream, `terminal_behaviour` returns `never_reached` only when
`segs[-1].phase == APPROACH` (`phases.py:107`). A policy that arrives at pre-grasp
and times out without closing returns `"unknown"`, so `classify`'s first and
most-specific rule (`never_reached` → `planning`) never fires for it. The
`planning` family is systematically under-populated.

---

## 4. `divergence_point` silently truncates — *[by inspection]*

`phases.py:90` zips the perturbed trace against the nominal one. Nominal rollouts
succeed in ~7 steps; perturbed ones run to the 40-step cap, so `zip` stops at 7.
Divergence is only evaluated over the prefix where the reference was still alive,
and `"trace stayed inside nominal envelope"` can mean "the reference ended."

Damage is limited because the doc already calls divergence a secondary signal —
but the `note` string asserts something the code did not check.

---

## 5. `floor_pp` is decorative — *[verified]*

> **Severity raised medium → high (2026-09-11).** `ARCHITECTURE.md` §5.1 does not
> merely suggest this step, it mandates it: *"Run `reproducibility_floor()` once per
> (policy, env) before trusting any probe, and report the floor alongside every
> boundary."* The codebase performs it nowhere. This is a documented precondition of
> every attribution claim, unexecuted.

`counterfactual_probe` gates attribution on `max(min_delta_pp, 3 * floor_pp)`, and
`reproducibility_floor` exists to supply that number. **Nothing calls it** — not
`oracle_test.py`, not anything. `floor_pp` defaults to `0.0` (`runner.py:149`), so
the guard is permanently `max(20, 0) = 20`.

On the toy the floor genuinely is 0, so this never bites. It starts mattering the
moment a stochastic action head (flow/diffusion) is swapped in — exactly the moment
nobody will think to check. Same class as #2: an invariant left in a docstring
instead of a precondition.

**Direction:** make `floor_pp` required, or have the probe refuse to attribute
without a measured floor. The second is preferable — it fails toward "we cannot
say," which is the correct epistemic state.

### Open question Q5.1 — how many repeats give a stable floor? (blocks #5)

Deferred 2026-09-11 by the user: the number of runs needed for a stable floor is
not known, and the fix should not be specified until it is.

What is known:

- `floor_pp = (max(rates) - min(rates)) * 100` at `repeats=2` is the gap between
  two draws — a *lower* bound on variability, and a noisy one. At n=20 per run, two
  runs can agree by luck and report a floor of 0 for a genuinely stochastic policy.
  The range statistic also grows with `repeats`, so the number is not comparable
  across different `repeats` values. If the floor gates every attribution it
  probably wants the SD of the rates, or an interval on it, rather than a range.
- **The floor cannot be measured on the toy.** `random.Random(seed)` makes rollouts
  bit-exact, so the true floor is 0.0 by construction and any `repeats` "works."
  Q5.1 is only answerable against a policy with a stochastic action head, i.e. after
  the first real policy lands. Deciding it now would be guessing.
- Cost is `repeats x len(seeds)` rollouts per (policy, env), **paid once** — not per
  probe. That is likely cheap enough that the answer is simply "more than 2," but the
  number should come from a measurement, not this document.

**Landmine to guard while this is open:** `reproducibility_floor` calls
`run_cell(env, policy, spec, seeds)` with **no store**, which is correct *by
accident*. Threading a store through would serve cached rollouts and report a floor
of exactly 0 every time, by construction — a plausible-looking "optimisation" that
silently destroys the measurement. Worth an explicit comment or assertion now, even
though the rest of #5 is deferred.

---

## 6. The "nominal reference" arrow has no producer — *[verified]*

`classify()` takes `nominal=` and the §2 diagram shows it feeding in. Only
`experiments/oracle_test.py:46` builds one. Neither `run_cell`, `sweep`, nor
anything in `mining/` produces or threads it, so any use of the pipeline outside
the oracle test gets `divergence: {"skipped": ...}`. The diagram documents a wire
only the test harness soldered.

---

## 7. `find_boundary` mislabels its baseline as nominal — *[verified]*

`runner.py:122`: `nominal = cells[0].rate`, and the emitted `definition` reads
`"success < 0.5 x nominal (95.0%)"`. `cells[0]` is the first *swept level*, equal
to nominal only if `levels[0] == 0` and `base` is empty.

Sweeping `[0.02, 0.04, 0.06, 0.08]` produced `nominal: 0.95` — a rate measured
under perturbation. That number reaches the manifest as `nominal_success`.

---

## 8. The probe discards its own pairing, then uses a threshold ≈ 1 SE — *[by inspection]*

`counterfactual_probe` re-runs the **same seed set** for the full and reverted
cells, so the two cells are paired — the strong part of the design. It then reduces
each to a scalar `rate` and subtracts, discarding the pairing entirely.

At n=20 per cell around p≈0.5, the SE of an *unpaired* difference of proportions is
~15.8 pp. `min_delta_pp=20` is ~1.3 SE: roughly a 10% one-sided false-positive rate
per knob, and the probe takes the **max** over ~5 knobs before thresholding, which
inflates that considerably. `attributed_knob` is doing less work than §4 implies.

**Direction:** keep per-seed outcomes on `Cell` (`rollout_ids` already makes them
recoverable) and run McNemar on the discordant pairs. Paired testing on the same
seeds is far more powerful than comparing rates and yields a p-value instead of a
magic 20.

---

## 9. The boundary bracket is honest about interpolation, silent about noise — *[verified]*

The docstring's argument — "between L2 and L3, not at 13.4 degrees" — is right
about interpolation and says nothing about sampling error. The scan takes the
*first* crossing of `cur.rate < thresh <= prev.rate` over point estimates each
carrying ±20 pp.

Measured sweep: `95 / 70 / 45 / 35%` with CIs `[76,99] / [48,86] / [26,66] / [18,57]`.
Adjacent intervals overlap heavily, so a noise dip at any level yields a bracket
formatted with the same confidence as a real one. Every other artifact in §4
reports a CI; this is the one that drops it, and the one that gets quoted.

**Direction:** bracket on CI bounds (first level whose *upper* bound falls below
threshold); return `None` + reason when the curve is not monotone within CI.

---

## 10. Single-knob reverts find main effects only, and fail silently — *[by inspection]*

The probe reverts one knob at a time. If two knobs break the policy only jointly,
every single revert yields a small delta, the max falls under threshold,
`attributed_knob` is `None`, and the manifest row loses the cause with no signal
that an interaction was the reason. Negative deltas — reverting a knob making
things *worse*, the clearest interaction tell — sort to the bottom and are dropped.

A real limit of the design, not a bug. But §2 presents `probe` as *the* attribution
input to `make_row`, so it should be stated as main-effects-only, and
`attributed_knob: None` should distinguish "no knob mattered" from "no *single*
knob mattered."

---

## 11. `TraceStore` is O(N²) and unbounded in memory — *[by inspection]*

`run_cell:92` calls `store.by_id()` per cell. `by_id` rebuilds when
`len(_cache) != len(_seen)`, true after every append — so each cell re-parses and
re-materializes the entire jsonl. `_scan()` also `json.loads` every full trace to
read one field. G2 moved images out but left full `obs_state` per step inline.

A LIBERO sweep — 300-step episodes, tens of thousands of rollouts — hits this
first, and it arrives as an OOM mid-sweep rather than a slowdown.

**Direction:** id-only index (parse `rollout_id` without full decode, or a sidecar
index file) and a `has()`-based miss path that loads one rollout by offset.

---

## 12. `grasp_radius` is duplicated across the L2/L3 boundary — *[by inspection]*

`PhaseSegmenter(grasp_radius=0.06)` restates `ToyReachEnv(grasp_radius=0.06)`.
G9 says the miner never imports an env, and it doesn't — but the *number* crossed
by hand. Change it in one place and segmentation silently disagrees with the goal
predicate. The rule kept the import out; it did not keep the coupling out.

**Direction:** the env publishes its thresholds into `Observation.state`
(unprefixed — a real robot knows its own calibration) and the segmenter reads them.

---

## 13. "Byte-identical trace out" is unfalsifiable as written — *[by inspection]*

`Rollout.wall_time_s` is wall-clock and is serialized by `to_json()`, so two runs
never produce identical bytes. For the project's stated load-bearing invariant,
there should be a canonical form excluding timing/host fields and a test asserting
equality over it. As written the invariant can only be checked by eye.

---

## 14. `rollout()` has no runner-side step cap — *[by inspection]*

`while True` with the env owning termination is the right architecture, but a buggy
adapter or a policy that never triggers the goal predicate hangs an unattended
overnight sweep with no diagnostic. A hard `max_steps * safety_factor` guard in the
runner that raises is cheap insurance.

---

## 15. Goal predicate is weaker than the instruction — *[verified]*

Instruction: *"pick up the black bowl."* Predicate: gripper closed within
`grasp_radius` of the object. No lift, no transport, no hold-for-N-steps.
`holding = True` returns `done` in the same call, so:

- `TRANSPORT` can never occur in any trace, on any policy.
- A whole class of real failure is unrepresentable — grasped then dropped, grasped
  the rim and it rolled, lifted and collided.

Acceptable in a fixture, but it means the taxonomy has never been exercised against
the failures LIBERO will actually produce, and `FAMILIES` was designed for failures
the toy cannot generate. Interacts with #1 (which makes `holding` observable) and #3
(which depends on `TRANSPORT` segment counts).

---

## 16. No mid-episode intervention — *[by inspection]*

`PerturbationSpec` is frozen and knobs are applied at `reset` only. Two kinds,
undocumented but behaviourally distinct:

| | Applied | Examples |
|---|---|---|
| Initial-state | once, at reset | `ee_offset_x_m`, `object_shift_m`, `distractor_count` |
| Observation-channel | every `_obs()`, i.e. per step | `camera_yaw_deg`, `pixel_noise_std` |

Nothing can perturb *partway through* an episode. That rules out the standard
technique for separating a perception failure from a control failure — run nominal
for k steps, then perturb, and see whether the policy was already committed. Also
rules out time-varying disturbances (lighting shift mid-episode, object nudged by
contact), which LIBERO-plus does not currently require but a real deployment does.

Not a bug; a limit of the design that should be stated rather than discovered. The
`t`-indexed `Step` record already carries what a future `PerturbationSchedule`
would need.

---

## 17. Shared RNG stream — `distractor_count` re-rolls the target's noise — *[verified]*

`_to_cam` draws from `self.rng` when `pixel_noise_std > 0`, and `_obs()` calls it
once per object **plus once per distractor**. So `distractor_count` changes how many
draws are consumed per step, which changes the noise realized on the **target**.

Verified with a CLEAN policy that never reads distractors:

```
seed0  obj_cam_xy@t1  d=0: +0.62353,+0.43040   d=2: +0.60818,+0.42440   same=False
seed1  obj_cam_xy@t1  d=0: +0.58242,+0.38652   d=2: +0.58963,+0.30652   same=False
seed2  obj_cam_xy@t1  d=0: +0.71539,+0.42565   d=2: +0.60920,+0.44110   same=False
no-noise control (pixel_noise_std=0): identical trajectory?  True
```

The no-noise control isolates the mechanism as the shared RNG stream, not the
distractors.

**Why this is rated high:** it breaks the premise of `counterfactual_probe`.
"Revert ONE knob" does not hold whenever `pixel_noise_std > 0` and
`distractor_count` is among the reverted knobs — reverting one silently re-rolls
the other. `experiments/oracle_test.py` co-perturbs exactly that pair when building
its probe spec (`("distractor_count", 2), ("pixel_noise_std", .02)`), so the
acceptance gate is measuring a confounded delta.

**Direction:** one independent `random.Random` per noise source (layout jitter,
target pixel noise, distractor pixel noise), each seeded deterministically from the
episode seed — e.g. `Random(hash((seed, "target_noise")))`. Then the number of draws
consumed by one source cannot shift another.

**Generalises beyond the toy:** any env where one knob changes how much randomness
is consumed has this bug. Worth an invariant — *reverting knob A must not change the
realized values of knob B* — and a test: clean policy, two specs differing in one
knob the policy provably ignores, assert identical traces.

---

## 18. Seed jitter is one-dimensional — *[verified]*

`ToyReachEnv.reset` draws **one** scalar and adds it to both coordinates:

```python
jitter = self.rng.uniform(-0.03, 0.03)
self.object_xy = (base_r * math.cos(base_th) + jitter,
                  base_r * math.sin(base_th) + jitter)
```

So across seeds the object slides along a 45° diagonal rather than varying in a
region. Measured over 20 seeds:

```
x spread 0.0515   y spread 0.0515   (y - x) constant across seeds: True
radius 0.6678..0.7382     bearing 29.26..30.79 deg
```

Radius varies 7 cm; **bearing varies 1.5°**. The policy is never asked to reach in a
materially different direction. Almost certainly meant to be two independent draws.

Consequence: the seed axis samples a far narrower task distribution than "20
episodes per cell" suggests, and any directional failure mode (a policy that is fine
reaching right and bad reaching left) is invisible to the entire harness.

---

## 19. Distractor layout has no seed dependence — CIs invalid on that axis — *[verified]*

Distractor placement is a pure function of `distractor_count`:

```python
a  = base_th + math.radians(18 * (i + 1))
rr = base_r * (0.72 - 0.06 * i)
```

No `self.rng` call. Verified: placement is byte-identical across all 20 seeds.

Every cell on the `distractor_count` axis is therefore ~20 near-identical episodes —
the only remaining variation is the 1-D object jitter of #18. Observed directly
earlier in this review: 25 consecutive seeds all classified `spatial_reasoning` with
final errors clustered in 0.436–0.469.

**Why this is rated high — it makes a reported statistic wrong, not merely narrow.**
`wilson_ci` assumes n independent Bernoulli draws. On this axis the episodes are
effectively one episode counted 20 times: the whole cell flips together under a small
threshold change. The printed interval understates uncertainty by roughly sqrt(n_eff)
— and the manifest carries those intervals as evidence.

It also weakens the acceptance gate: a miner that recovers the planted
`nearest_object` fault across 20 copies of one episode has demonstrated less than
`PASS` implies.

**Direction:** seed the distractor layout (angle and radius jitter per distractor)
from the episode RNG — coordinated with #17, so each source draws from its own
stream. Then re-check whether the `nearest_object` scenario still passes; if it
becomes marginal, that is information, not a regression.

---

## Agreed fixes — handed to `primary` 2026-09-11

Both are schema changes and interact, so they ship together under a single
`SCHEMA_VERSION` major bump. Traces already in `runs/` are untrustworthy while #2
stands, so there is nothing to preserve — do not write a migration.

### #2 — rollout identity (status: FIXED 2026-09-11)

1. **`identity()` on the protocols.** `Env` and `Policy` (`schema.py`, G6) grow a
   required `identity() -> dict` returning every semantically load-bearing value:
   constructor kwargs, detector thresholds, checkpoint revision/hash, control mode,
   action space. **Values, not source text.** `env_id` / `policy_id` derive from it
   rather than being typed by hand.
2. **No git SHA, in any form.** Rejected: too coarse (a README edit invalidates the
   store) *and* too weak (a dirty tree keeps the SHA stable while code changes
   underneath it, manufacturing confidence in a stale cache exactly during Phase-3
   iteration). SHA + hash-of-diff is just a slow content hash — hash the values.
3. **Fingerprint stored, not only hashed.** Persist the structured fingerprint on
   each `Rollout`. `run_cell` compares it on every cache hit.
4. **On mismatch: re-run loudly; do not raise by default.** Record a diff of *which
   fingerprint fields changed* and surface it in the run summary, not a log line.
   Raising kills an overnight sweep at hour 8 over a moved threshold, and under
   session caps someone will then disable the check — a disabled check is worse than
   none. Provide a strict mode that raises, for client-facing runs.
5. **Make resume visible.** Report the resumed-vs-fresh split in the run summary. A
   stale cache and a real finding are indistinguishable from outside; §6's "nominal
   below ~90% is nearly always a harness bug" heuristic is unusable without it.
6. **Document what the cache *is*.** Add to §5.1: once traces stop being
   bit-reproducible, the cache stops being "avoiding redundant deterministic work"
   and becomes "reusing samples from a distribution." Therefore the fingerprint must
   cover anything that shifts the **distribution**, not merely anything that changes
   a trace; and a cache entry can never be verified by re-running it — any audit
   degrades to comparing success-rate distributions at n>=20, catching gross
   corruption and nothing subtle.

*Deferred, not rejected:* sampled cache audit (re-run ~2% of hits, check agreement).
Sound on the toy, unavailable on a real VLA for the reason in item 6.

*Justification order for the commit message:* lead with **detector-config churn in
Phase 3** — nudge `pregrasp_radius`, re-run, see identical numbers, blame your own
change. Constant and unscrutinised. The Phase-5 fine-tune case is the one with
commercial consequences but it is the *safer* one: it happens once, under scrutiny,
with someone hunting a before/after delta.

### #1 — terminal observation (status: FIXED 2026-09-11)

1. **Append a terminal `Step` after the loop exits**, carrying the observation
   returned by the final `env.step`, with `action=None`.
2. **`action=None`, not a separate `Rollout.final_obs` field.** `r.series()` is how
   all of `mining/` reads state; a terminal `Step` is picked up automatically and no
   detector changes. A `final_obs` field is cleaner semantically but requires every
   detector to remember to look in two places, and `r.series()` would silently keep
   missing it. Consumers iterating `steps[]` for actions must handle `None`.
3. **Keep the existing obs/action pairing.** Pairing `obs_t` with `action_t` inside
   one `Step` is correct — that is the decision the policy made given that input.
   The bug is the missing final emission, not the pairing.
4. **Re-run the repro before claiming this fixed:** `holding` must be `True` in the
   final step of every successful rollout; recorded `grasp_attempts` must equal the
   env's; `final_error_m` must equal `math.dist(env.ee_xy, env.object_xy)` at
   termination. Expect ~10/23 failure labels in the `unfiltered_grasp` @
   `pixel_noise_std=0.15` fixture to change — that is the bug being removed, not a
   regression.
5. **Check the knock-on:** `PhaseSegmenter.requires` includes `holding`, so
   `TRANSPORT` becomes emittable for the first time. Confirm that does not perturb
   `terminal_behaviour`, and re-run `experiments/oracle_test.py` — some ground-truth
   expectations may legitimately shift.


---

## Appendix — smaller items found while tracing §2, outside its scope

These sit just off the §2 flow (in `classify`, `cluster`, `oracle_test`). Recorded
so they are not lost; not triaged into the table above.

- **A1.** `Step.obs_state["gripper"]` lags by one step — the env sets
  `self.gripper = a["gripper"]` inside `step()`, but `rollout` records the
  *pre-action* obs. `_attempt_spread` selects attempt points by that lagged
  gripper value. Currently benign only because a toy grasp action has
  `dx=dy=0` so the ee has not moved; it is correct by accident. Reading
  `Step.action` (which carries the commanded gripper) would be correct by design.
- **A2.** `cluster.mean_final_error_m` coerces a missing `final_error_m` to `0`,
  silently dragging the mean toward zero. Should skip `None`s and report the count
  it averaged over.
- **A3.** `oracle_test` takes its verdict from `clusters[0]["family"]` — the modal
  family over a pool mixing the multi-knob probe spec with every sweep level. The
  gate can pass for the wrong reason (a family dominant because of a co-perturbed
  axis the test itself injected, not because of the planted bug).
- **A4.** `oracle_test` step 3 re-runs rollouts via `rollout()` directly rather than
  through the store, so the traces the manifest's evidence is drawn from are never
  persisted, and work already done by `sweep` is repeated.


---

## Implementation notes — #1 and #2, `primary` 2026-09-11

Shipped together under `SCHEMA_VERSION` 1.0 → 2.0. No migration; prior traces in
`runs/` were discarded as specified.

### #1 — verified

`runner.rollout` now appends a terminal `Step` carrying the post-`env.step`
observation with `action=None`. Repro re-run over 25 rollouts:

```
holding True at final step on 25/25 successes
grasp_attempts matches env on every seed
final_error_m == math.dist(env.ee_xy, env.object_xy) at termination
terminal step action is None
violations: 0
```

Knock-on checked as instructed. `TRANSPORT` is emittable for the first time
(6/25 rollouts at `pixel_noise_std=0.15` now contain one; previously 0).
`terminal_behaviour` is unperturbed — it short-circuits on `r.success` before
inspecting phases, so all 6 remain `success`; the other 19 remain `retry_loop`.
No consumer iterates `Step.action`, so `None` is safe (`grep` confirmed).

`experiments/oracle_test.py` still passes 3/4 with control PASS — no ground-truth
expectation shifted.

### #2 — verified, with one correction to the handed-down spec

Repro re-run at `pixel_noise_std=0.03`, where `grasp_radius` is actually
load-bearing:

```
grasp_radius=0.06   shared=1.00  fresh=1.00  id=toy-reach-v1@e1b9543d  cached=0
grasp_radius=0.001  shared=0.00  fresh=0.00  id=toy-reach-v1@17176c1a  cached=0
```

Shared-store rates now match fresh-store truth and differ from each other. (Note
for anyone re-running: with **no** perturbation the clean policy lands exactly on
the target, so `grasp_radius=0.001` still succeeds and the repro shows 1.00/1.00
for the wrong reason. The original repro's 0.00 needs a config where the radius
binds.)

**Correction — spec item 3 as written produced dead code.** "Persist the
fingerprint and compare it on every hit" cannot fire when the fingerprint *is*
`identity()` and the key is `hash(identity())`: change identity and the key
misses; leave it and the fingerprint matches. It also cannot catch an
`identity()` that forgot a field, because the fingerprint forgot it too —
demonstrated with a deliberately-lying env, which sailed through with
`cached=20, stale_hits=0`.

Fixed by making the fingerprint **strictly broader than the key**:
`{env: identity, policy: identity, runtime: {python, platform, mujoco, torch,
numpy, lerobot, robosuite}}`. `runtime` is deliberately *not* in the key — a
MuJoCo point release or a different GPU can move success rates, but keying on
them would invalidate every rollout on every machine and destroy the
cross-session resume caching exists for. In the fingerprint, out of the key:
caught and reported, neither silently ignored nor silently fatal.

Verified by simulating a simulator upgrade between sessions:

```
after simulated MuJoCo upgrade: cached=0 fresh=10 stale_hits=10
  diff: {'runtime.mujoco': {'cached': '<absent>', 'current': '3.3.2-UPGRADED'}}
```

`fingerprint_diff` recurses, so a nested change names the field that moved
(`runtime.mujoco`), not the section containing it.

Everything else per spec: no git SHA anywhere; ids derived via `derive_id`;
mismatch re-runs loudly with a field-level diff in the run summary;
`strict=True` raises; `resume_summary()` / `format_resume_summary()` report the
cached-vs-fresh split.

### Your correction on detector churn — confirmed, and acted on

You were right and I was wrong. Verified directly:

```
$ grep -rn "store.append\|\.diagnosis *=" vla_harness/ experiments/
vla_harness/runner.py:142:                store.append(r)      # r.diagnosis is None here
experiments/oracle_test.py:99,104:  r.diagnosis = classify(...)   # outside the store path

$ 840/840 persisted rollouts have diagnosis: NoneType
```

The Phase-3 justification is now scoped to **env and policy configuration only**.
I also removed the "detector thresholds" claim from the `Env.identity()`
docstring, where I had written the same error into the code.

The latent version is documented rather than designed for, as you preferred:
`ARCHITECTURE.md` §7 now states that rollouts are persisted pre-diagnosis, that
this is what keeps detector config safely outside the fingerprint, and that the
property disappears the moment a classified rollout is persisted — with a
one-liner to check it.

`reproducibility_floor`'s storeless call is now an explicit comment explaining
that threading a store through would report `floor_pp = 0.0` by construction.
Nothing else in #5 touched.

### Not done

- Sampled cache audit — recorded as deferred in `ARCHITECTURE.md` §4.1.
- Appendix A1–A4 — untouched, not assigned.
- Findings #3–#14 — untouched.
