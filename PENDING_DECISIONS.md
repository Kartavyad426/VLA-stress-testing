# Pending decisions

Everything waiting on a human call, as of 2026-09-23 (first written 2026-09-16).
Each item says what it is, what it costs, what it blocks, and what I'd do — so a
decision can be made without re-reading the transcript.

**Status 2026-09-23:** items marked ✅ / ❌ are closed, and #10, #20, #21 and #25
carry a dated status line where later results overtook them. Current roster:
GR00T N1.7 (workhorse), MINERVA (fp32 control), SmolVLA vetoed, π0.5 and π0-FAST
do not fit (R-032). `HANDOFF.md` §7 lists what is still open.

*(2026-09-16 note, now stale: "Nothing below has been actioned. Nothing is
committed since `b198613`." The code files in the working-tree list that followed
have since been committed; the `viz/*_video.html` pages are untracked.)*

---

## Suggested order

**Project priority: GR00T N1.7**, reached via MINERVA reproduction → harness verified correct → GR00T (see #10).

**15 → 13(b) → 3 → 16**, then reassess. #15 first: until the taxonomy is settled, no family count or manifest row is reportable, and #4, #5 and #6 all depend on its answer. Item 11 is cheap and may explain the
headline gap; item 4 is free and the taxonomy is actively wrong until it lands;
item 13(b) is the abstention contract that stops O1/O2/O6 recurring and is
worth landing before more detectors are written; item 3 unblocks a whole suite.
Items 2 and 6 are the expensive ones and both get
*better* if 11 moves the numbers — there is no point re-capturing 965 traces at
67% if a version pin puts us at 85%.

---

## 1. Which holding route is the label?

**Decision:** make `by_lift` the phase label and report `by_gripper`'s agreement
with it as a measured statistic, rather than OR-ing the two.

**Why:** `holding = by_gripper or by_lift` means the most permissive detector
wins. Post-fix branch coverage of successes:

| suite | by_gripper alone | by_lift alone |
|---|---|---|
| libero_spatial | 0/79 | 79/79 |
| libero_object | 78/79 | 79/79 |
| libero_10 | 43/52 | 45/52 |

`by_lift` is at least as good everywhere, so the OR adds only risk. Reporting
agreement turns the privileged-free route from an assumption into a number.

**Cost:** free. **Blocks:** nothing. **Partly superseded by #2.**

---

## 2. GPU re-run to capture `_check_grasp` contact data

**Decision:** re-run the campaign capturing a proper grasp signal.

**Why:** robosuite already has the standard grasp test —
`ManipulationEnv._check_grasp` (manipulation_env.py:201) — a bilateral
fingerpad-contact check used by every robosuite manipulation task. It has **no
aperture threshold**, so it is immune to the object-geometry problem (O4) that
makes a single `CLOSED_M` impossible, and immune to the 8-step travel transient
(O3). Verified reachable from our adapter:

```python
robo = self._env.unwrapped._env.env          # same path as our _sim()
obj  = robo.get_object("alphabet_soup_1")    # returns the MujocoObject
held = robo._check_grasp(gripper=robo.robots[0].gripper, object_geoms=obj)
```

`BDDLBaseDomain` inherits `_check_grasp` from `ManipulationEnv` — confirmed by
MRO inspection.

**Why it needs a re-run:** `_gt_n_contacts` is `int(sim.data.ncon)` — a scalar
count of every contact in the whole simulation. It cannot distinguish "both pads
on the object" from "one pad on the object, one on the table", so the contact
method **cannot be mined from the 965 stored traces**. Second time we have hit
this: capture privileged state generously.

**Cost:** ~3h21m GPU. **Blocks:** the proper fix for O3/O4.

---

## 3. ✅ DONE 2026-09-17 — `libero_goal` re-run

**Done** (`RESULTS.md` R-019, R-020). Goal is fully instrumented: 235/265 traces
from the re-run, and the last task (task 5, a table region that is a MuJoCo
*site*) fixed and re-run at 30/30. `experiments/object_resolution_test.py`
passes on all 40 tasks. **Next:** run the language-vs-geometry discriminator on
these traces.

### Earlier status: PRIORITY RAISED 2026-09-16

**Now the input to the only experiment that separates F10's two branches.**
Within goal: language failure predicts wrong-goal completions, geometric
failure predicts right-goal attempts that miss. The missing fixture/region
bodies are exactly what the goal predicates refer to. On re-run, verify
`_gt_object_pos_complete` is True for those bodies before trusting the traces.

### Original entry

**Decision:** re-run the goal suite (~42 min GPU). Outstanding since the
original handoff.

**Why:** 200/260 goal traces never recorded `_gt_eef_to_object`, so the miner
abstains on them and the suite is unusable. The BDDL region→body resolver fix
cannot be applied retroactively — the signal does not exist in those traces.

```bash
MUJOCO_GL=egl .venvs/lerobot/bin/python experiments/e2e.py \
  --suite libero_goal --tasks 8 --seeds 5 --yaw-levels 0,5,10,15,20 \
  --run-id goal_rerun
```

**Cost:** ~42 min GPU. **Blocks:** the entire goal suite.
**Combine with #2 if #2 is approved.**

---

## 4. The `visual_grounding` family is mislabelled

**Decision:** rename it to what the rule measures, reorder `classify()`, and
reclassify the affected episodes.

**Why:** the rule is `final eef-to-target distance > 0.10 m` — a **positional**
test carrying a **semantic** name. Where those episodes actually ended:

| suite | n | ended near a DIFFERENT named object (<0.10 m) | ended nearest its OWN target |
|---|---|---|---|
| libero_spatial | 99 | 12 (12%) | 56 (57%) |
| libero_object | 161 | **0** | 147 (91%) |
| libero_10 | 73 | 2 (3%) | 26 (36%) |

Across all 333 labels, **14 (4.2%)** match the only pattern the name describes.
On libero_object it is zero out of 161.

And the ones that ended nearest their own target all *tried*:

| suite | median final distance | median grasp attempts | zero-attempt episodes |
|---|---|---|---|
| libero_spatial | 0.126 m | 7 | 0/56 |
| libero_object | 0.207 m | **23** | 0/147 |
| libero_10 | 0.145 m | 5 | 0/26 |

Our reason string is *"went confidently to the wrong place"*. A policy making 23
grasp attempts 0.2 m from the correct object is not doing that.

**Corroboration:** MINERVA (arXiv:2609.03715) scores 95.05% on all four suites
with **no language encoder** — a 40-entry task-ID embedding table. So nominal
LIBERO never requires binding an instruction to an object, and a grounding
family is not measurable on it *in principle*. LIBERO-PRO (arXiv:2510.03827)
and LIBERO-Plus (arXiv:2510.13626) point the same way: position, not
discrimination, is the weak axis.

**Sub-decisions:**
- (a) rename `visual_grounding` → `positional_error` (or similar)
- (b) reorder `classify()` so `wrong_object()` runs BEFORE the distance rule —
  it already exists in `signals.py` but is ordered after, so the positional rule
  always wins
- (c) reclassify repeated-attempt-near-target cases as `manipulation`
  (~147 episodes on libero_object alone)

**Cost:** free, CPU re-mine. **Blocks:** every family count is wrong until done.

---

## 5. Write the above into `IMPLEMENTATION_OVERSIGHTS.md` as O5

**Decision:** whether to document #4 as O5. It is the most consequential entry
so far and belongs with O1–O4, but it changes the taxonomy, so the naming in #4
should probably settle first.

**Cost:** free. **Blocks:** nothing.

---

## 6. Fit the toy-derived thresholds on LIBERO

**Decision:** run `fit_from_demos()`, which exists and has never been run.

**Why:** `LOST_TARGET_M = 0.10`, `WRONG_OBJECT_M = 0.09`, the `0.6` ratio,
`REPEAT_SPREAD_M = 0.015` and `pregrasp_m = 0.10` were all chosen in the toy
environment to make a planted fault reproducible. They are metric and LIBERO is
metric, so they *transfer* — but none has been *validated* here.

An outside review called this a bigger risk than O1–O4, and I agree: a bug
produces a wrong answer loudly once you look, an unfitted threshold produces a
confident plausible answer forever.

**Caveat already in the code:** `fit_from_demos` is framed as COVERAGE, not a
discriminative boundary — demos contain only successes and cannot speak to where
failures begin. Fitting on one class and reporting it as discriminative is the
error its docstring exists to prevent.

**Cost:** ~1h. **Blocks:** trusting any family count quantitatively.

---

## 7. Fix replay fidelity in the video visualiser

**Decision:** capture and restore the exact init state so the replay is
bit-exact.

**Why:** `experiments/visualise_episode_video.py` works, but self-reports
**9.5 mm median / 17 mm max** deviation from the stored end-effector path,
failing its own 5 mm bar. The page shows an amber warning rather than presenting
the video as ground truth.

Diagnosis so far: deviation is *largest at reset and decays* (15.7 mm → 5.4 mm),
and offset 0 is the best time alignment — so it is a small initial-state
difference the relative controller washes out, not chaotic divergence and not a
frame-offset bug.

Usable now for "did the gripper have anything in it". Not yet usable for
frame-exact claims.

**Cost:** ~1h. **Blocks:** nothing.

---

## 8. Evaluate LIBERO-PRO's perturbation API

**Decision:** read it before building more perturbation machinery ourselves.

**Why:** LIBERO-PRO (arXiv:2510.03827) ships `perturbation.py` +
`evaluation_config.yaml` with `use_swap` / `use_object` / `use_language` /
`use_task` boolean knobs and intensity levels x0.1–x0.5. That is a working knob
API, which is more than LIBERO-Plus ships. It is also the kind of environment
where a *real* grounding family could exist (one scene supporting two goals),
which nominal LIBERO cannot provide — see #4.

**Cost:** reading. **Blocks:** nothing.

---

## 9. Commit

**Decision:** what to commit and under what message.

**Note:** commit `b198613` ("Correct GR00T entry…") swept up in-progress work
from this session — `IMPLEMENTATION_OVERSIGHTS.md`, the phases_libero fixes and
`experiments/phase_segmenter_test.py` are all inside a commit whose message
describes none of them. Worth a follow-up commit message or a note.

**Cost:** free.

---

## 10. ✅ DECIDED 2026-09-16 — Policy roster and architecture

> **Status 2026-09-23 — roster overtaken.** GR00T N1.7 is the workhorse (bf16;
> parity 98 vs 97, R-022). MINERVA is the fp32 control (95.3% vs 95.75%, R-016;
> bf16 broken, R-033). SmolVLA: vetoed 2026-09-18 (R-011, R-017). The MINERVA and
> GR00T venvs are built. The architecture and one-venv-per-policy decisions stand.

**Decided by user:**
- Roster: ~~**SmolVLA** (primary) / **MINERVA** (alternate) / **GR00T N1.7** (third)~~ — see status above.
- Architecture: **their policy, our simulator, our observability layer**, plus a **spare simulator** for policies whose pins conflict (MINERVA's MuJoCo 3.3.2). Policies are swappable boxes; switching must be low-friction.
- Every policy in its **own venv** (`MODELS_AND_COMPUTE.md` §R7).

**Sequencing (user directive, 2026-09-16): GR00T N1.7 is the project priority.**
The path to it is gated, in order:
1. **MINERVA reproduces** in our harness (confirms the harness can hit a
   published number with a known-good checkpoint).
2. **The harness is verified correct** — the open defects that would corrupt
   any policy's results are fixed: camera perturbation (#22), stale first frame
   (#22), `failure_cost` abstention (#13), goal instrumentation (#3).
3. **Then GR00T** becomes the policy for further experimentation.

SmolVLA work continues only where it serves steps 1–2. Note #16 (GR00T VRAM fit
on 8 GB) is the first thing to check once step 3 starts, and vla-7f flagged a
candidate silent failure: GR00T was trained on
`IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot`, whose image orientation
convention is unverified against `LiberoProcessorStep`'s 180° flip.

**Built:** the conformance gate, `vla_harness/conformance.py` — see #14.

**MINERVA box: BUILT 2026-09-16.** `.venvs/minerva` from the authors' `uv.lock`
verbatim (repo `third_party/MINERVA` @ `64c3cc8`, their LeRobot fork 0.6.1,
MuJoCo 3.3.2, Python 3.13). Only non-lockfile inputs: `cmake` as a uv tool and
`CMAKE_POLICY_VERSION_MINIMUM=3.5`, both needed to build `egl-probe` under CMake 4.
Checkpoint `t05_l1_0.54M` at the card's pinned revision `1b4fb174`. Gate PASSES the
documented command. **Smoke test: 5/5 on libero_object task 0**, 7.5 s/episode at
batch 5. **Not yet a reproduction** — the published protocol is 2,000 episodes
(~4 h at that rate).

**Still to build:** the box registry and per-box interpreter dispatch (`e2e.py` still hardcodes `.venvs/lerobot`); MINERVA's venv; GR00T's venv.

### Original entry

**Decision:** which model to add as a control.

**Why a control at all:** our SmolVLA scores ~67% against ~87.3% published. A
second, independent checkpoint separates "our harness is wrong" from "this
checkpoint's integration is wrong".

| candidate | size | LeRobot-native? | notes |
|---|---|---|---|
| **MINERVA** `k1000dai/MINERVA` | 0.54M | **yes, safetensors** | 95.75% avg. `t05_l1_0.54M` + a `t3C_2.75M` teacher. Apache-2.0. No language encoder, so nothing to misconfigure — the strongest control available. |
| **GR00T N1.7** | 3B | **yes** — see #12 | `nvidia/gr00t17-lerobot-libero_*-640`, 96.5% avg. VRAM marginal on 8 GB. |
| VLA-Adapter | 0.5B | unknown | claims 97.3%. Integration cost unknown. |
| MiniVLA | 1B | no (Prismatic) | a whole second stack. |
| TurboVLA | 0.2B | unknown | noted as unverified. |

**Recommendation:** MINERVA first. It is a download, not a training run,
LeRobot-native, and removes checkpoint provenance as a variable entirely.

**Constraint (user directive, 2026-09-16):** each model gets its OWN virtual
environment, and the harness stays plug-and-play — see `MODELS_AND_COMPUTE.md`
§R7. This is load-bearing, not hygiene: MINERVA pins MuJoCo 3.3.2 against our
3.3.7, and GR00T N1.5 needs `lerobot==0.5.1` against N1.7's current. A shared
env would silently resolve to one set of pins and misattribute every number.

**Prerequisite, not yet built:** `experiments/e2e.py` hardcodes
`.venvs/lerobot`. A model registry with per-model interpreter dispatch is
needed BEFORE the second model lands, plus MuJoCo version in
`semantic_deps()` so a renderer difference shows up in the fingerprint rather
than being discovered later.

**Cost:** ~1h for MINERVA + ~1-2h for the registry. **Blocks:** nothing, but it
is the cleanest read on #11.

---

## 11. ✅ nas RESULT IN / MuJoCo still open — 2026-09-16

**`n_action_steps` 1 vs 10 on spatial: no effect.** 77% vs 73%, +4.0 pp, 95% CI
[−8.0, 16.0]. Paired McNemar p=0.597 (18 vs 14 discordant). Matches the
pre-registered prediction (SmolVLA Table 13: 89 at both). **nas is not the
residual gap.** Details: `experiments/repro/runs/res256_nas1_seed1000/PREDICTION.md`.

MuJoCo part: see #21 (goal-only A/B proposal).

### Earlier status

**Running now — `n_action_steps` 10→1** (not MuJoCo). While building the gate, resolution turned out to be ALREADY fixed (F9; the 67% is the 256×256 number), and the untested variable is `n_action_steps`: the baseline ran 10, SmolVLA ships 1. One-variable A/B against `res256_nas10_seed1000` spatial = 73.0%. Output: `experiments/repro/runs/res256_nas1_seed1000/`.

**Caveat:** SmolVLA has NO documented eval anywhere (vla-7f). nas=1 is its shipped DEFAULT, not a known-correct eval setting — a clean test, not a test of the right config.

**MuJoCo 3.3.2 A/B: still pending.** User chose the full documented protocol (400 episodes/arm, ~6h). Needs its own venv per §R7. Note F2 (`control.py`): a MuJoCo point release already took one LIBERO task from 80% to 28% — direct evidence of version sensitivity in our own findings.

### Original entry

**Decision:** downgrade to MuJoCo 3.3.2 in a scratch venv and re-measure.

**Why:** MINERVA's model card pins **MuJoCo 3.3.2 "due to renderer
sensitivity"**. We are on 3.3.7.

| | MuJoCo | LIBERO score |
|---|---|---|
| MINERVA authors (pinned) | 3.3.2 | 95.75% |
| lerobot#3264 reporter | 3.3.2 | 73.25% |
| **us** | **3.3.7** | **67%** |

We already established (F9) that render configuration was worth +17pp on
spatial alone — LeRobot renders 360x360 while the checkpoint declares 256x256.
A version-sensitive renderer is the same class of cause. Three independent
reports of under-reproduction now exist in this ecosystem (lerobot#3264,
lerobot#2114, openvla#282), so the gap is probably not mainly ours.

**Cost:** ~30 min. **Blocks:** nothing, but it changes the value of #2 and #6.

---

## 12. ✅ CLOSED 2026-09-16 — Correct the GR00T entry

**Resolved:** corrected in `MODELS_AND_COMPUTE.md` §R1. LeRobot has a first-class `groot` policy type (in our checkout) and NVIDIA publishes LeRobot-format LIBERO checkpoints at 96.5% avg. VRAM fit remains unverified — carried into #16.

### Original entry

**Decision:** fix `MODELS_AND_COMPUTE.md` and note the correction against
`b198613`.

**What is wrong:** `b198613` concluded GR00T LIBERO checkpoints "are not
LeRobot-loadable" because "LeRobot expects its own GrootConfig, and no
LeRobot-hosted GR00T checkpoint exists". Both halves are false as of now:

- LeRobot has a first-class `groot` policy type. Present in OUR installed
  checkout at `third_party/lerobot/src/lerobot/policies/groot/`
  (`configuration_groot.py` defining `GrootConfig`, `modeling_groot.py`,
  `groot_n1_7.py`, `processor_groot.py`), registered at `factory.py:50`.
- NVIDIA publishes LeRobot-format LIBERO checkpoints:
  `nvidia/gr00t17-lerobot-libero_{spatial,object,goal,10}-640`, built against
  LeRobot 0.6.1 (we run 0.6.2), Apache-2.0, reported 95/100/98/93 = **96.5%
  avg** at `eval.n_episodes >= 50`.

Usage is `--policy.type=groot --policy.base_model_path=<id>
--policy.embodiment_tag=libero_sim --env.type=libero`.

**The real remaining blocker is VRAM, not format.** 3B params; the HF page lists
F32 tensors, so bf16 loading is required. `b198613` measured 6.44 GiB inference
weights against ~7.34 GiB free on an 8 GB card — feasible but marginal, and the
`-640` checkpoints suggest large activations. **Unverified: whether it actually
fits.** That is a ~20-minute test.

Note also the checkpoint declares 256x256 observation images — the same
resolution at the heart of F9.

**Cost:** doc fix free; VRAM test ~20 min. **Blocks:** nothing, but the repo
currently records a false conclusion.

---

## 13. Fix `failure_cost` — the manifest's severity column is uninformative

**Decision:** give `failure_cost` a real holding signal and an abstention
contract. Recorded as **O6** in `IMPLEMENTATION_OVERSIGHTS.md`.

**What is wrong:** it reads `r.series("holding")`, a **toy-only** state key.
LIBERO never emits it — `holding` is derived in the segmenter and never written
back to `obs_state`. Confirmed: **0 of 705 traces** contain it. So the `dropped`
branch is always False and `COST_DISRUPTIVE` — "grasped it, then dropped it",
the failure a client cares about most — can never fire:

```
failure_cost over 495 LIBERO failures:   benign 492   None 3   disruptive 0
```

**Why it matters:** a manifest row's severity is
`CONDITIONAL x PREVALENCE x COST`. `condition_prevalence` is deliberately
`NOT_ESTIMATED` (client-supplied). That leaves cost as one of two computed
factors — and it is a constant. **The severity column is uninformative by
construction**, and nothing in the output says so.

**Two parts to the fix:**

- (a) **Feed it a real signal.** Either the derived `holding` from the
  segmenter, or — better, and the reason this is worth sequencing after #2 —
  the `_check_grasp` contact signal. A drop is "was in contact, then was not",
  which contact answers directly and aperture does not.
- (b) **Give it an abstention contract.** A declared `requires` list and
  `cost: None, needs_review: True` when unmet, the way
  `LiberoPhaseSegmenter.missing()` already works. This is the actual fix:
  without it the next missing key produces another silent constant.

**Note the pattern.** O1, O2 and O6 are the same shape — a detector reads a key
or condition its environment does not supply and degrades into a constant
instead of abstaining. (b) is what stops this recurring; (a) only fixes this
instance.

**Cost:** (a) free if using derived `holding`, otherwise gated on #2.
(b) ~1h. **Blocks:** any severity claim in the manifest.

---

## 14. ✅ BUILT 2026-09-16 — Conformance gate

`vla_harness/conformance.py`. Reads a checkpoint's `config.json` and compares it
against the run: image keys, **render resolution vs declared shape**, state and
action dims, `n_action_steps` (vs documented overrides), `chunk_size`, control
mode, normalisation, and prose-only pins (MuJoCo). Three verdicts —
MATCH / MISMATCH / **UNDECLARED**, where undeclared is explicitly not a pass (O6).
Refuses on ERROR-severity mismatches, and prints recommended settings with the
source of each.

Tested against real configs:

| config | gate | why |
|---|---|---|
| original campaign (360, nas=10) | **REFUSE** | resolution — the F9 defect. Would have stopped the 400-episode run. |
| F9 baseline (256, nas=10) | PASS, 1 warn | nas=10 is an undocumented departure |
| current rerun (256, nas=1) | PASS | only control_mode unverified |
| pi0.5 + documented nas=10 | PASS | override recognised, source cited |
| MINERVA under MuJoCo 3.3.7 | **REFUSE** | prose MuJoCo 3.3.2 pin |
| MINERVA + documented eval | PASS, 8/0/1 | nas=1 + temporal_ensemble_coeff=0.01 |

Root cause it closes (vla-7f, primary source): LeRobot's own pre-eval check
compares visual key **names only, never shapes** (`policies/factory.py:426-428`,
`policies/utils.py:249-257`), which is why 360 vs 256 passed silently.

Bugs found in the gate during testing, fixed: it refused pi0.5 over
policy-synthesised `empty_camera_*` placeholder keys; it could not load
subfolder checkpoints; it recommended MINERVA's shipped nas=8 instead of the
documented 1.

**Known gaps:** no rule yet for **image orientation** or **camera→key
assignment** (neither is declared in any config), and GR00T's keys only match
via a hardcoded `image2 → wrist_image` alias
(`policies/groot/processor_groot.py:1579`).

---

## 15. 🟡 PARTLY ADDRESSED 2026-09-17 — family taxonomy (was order-dependent)

**Done (user decision):** `classify()` no longer uses first-match-wins. Every rule
is evaluated and the diagnosis reports all matches: `families` (list),
`predicates` (booleans) and `reasons` (per family). `family` remains only as an
order-independent grouping key (e.g. `visual_grounding+manipulation`) so clusters
and the manifest keep working; manifest coverage concatenates each member's
prescription. Oracle gate unchanged: 3/4 + control PASS; the oracle now checks
that the planted family is AMONG the matches.

**Still open:** the predicates themselves. `manipulation` fires on ~99% of
failures, `spatial_reasoning` treats the task's own DESTINATION as a distractor,
and `visual_grounding` is positional (R-007). Reporting all matches removes the
order dependence; it does not make the families meaningful. That needs the
taxonomy redesign with consumer input.

### Original entry

**Verified this session** by running vla-81's read-only
`experiments/family_precedence_test.py`; its per-suite counts match an
independent re-mine exactly.

**409 of 534 featurised failures (76.6%) fire more than one rule.** Their family
is decided by rule ORDER, not by the taxonomy.

| family | shipped | min over 120 orderings | max |
|---|---|---|---|
| visual_grounding | 350 | **0** | 397 |
| manipulation | 125 | 125 | **528** |
| spatial_reasoning | 43 | **0** | 44 |
| recovery | 10 | **0** | 24 |
| planning | 6 | **0** | 6 |

**Every family except manipulation can be driven to zero by reordering alone.**
Most common co-firing set: `lost_target + any_attempt`, 336 episodes.

Also verified:
- **`manipulation` is near-vacuous.** Its predicate `attempts >= 1` fires on
  528/534 (98.9%).
- **`ambiguous` is unreachable.** `any_attempt` (528) and `never_reached` (6)
  partition all 534 exactly, so no failure can reach the fallthrough under ANY
  ordering. G8 abstention is dead on LIBERO — the classifier is not confident,
  it is *unable to decline*. Same shape as O1/O2/O6.
- **`recovery` is suppressed, not rare.** 24 failures show repeat attempts; 14 are
  relabelled (almost all to visual_grounding). Recovery is ~2.4× the shipped count.

**Sub-decisions — all yours:**

- **(a) Labels or predicate vectors?** vla-81 recommends emitting the predicate
  vector instead of a single label: `lost_target AND any_attempt` is exactly what
  the data supports, and collapsing it to one family adds a claim the evidence
  does not carry. Needs no new thresholds, and composes with ddmin. **vla-81's
  Test 3 is blocked on this.**
- **(b) Are visual_grounding and spatial_reasoning swapped?** vla-81 argues rule 2
  (ended near a *different named object*) is grounding and rule 3 (ended far with
  nothing nearby) is positional, but the code assigns them the other way round.
  Overlaps with #4's rename — settle together.
- **(c) Is the downstream-utility experiment in scope?** The only test that can
  falsify a taxonomy: fine-tune on family-targeted vs uniform data at equal
  budget. Needs a fine-tuning loop we have never run, plus GPU. If out of scope,
  vla-81 will write the audit's conclusion as "unvalidated, and here is the
  experiment that would settle it" — an honest deliverable.
- **(d) Editing boundary.** vla-81 has stayed read-only on `vla_harness/`. Keep
  that, or let it fix `mining/` directly? Two sessions editing `classify.py`
  uncoordinated is worse than either.

Also flagged, minor: `COST_SAFETY` is defined but never assigned;
`language_grounding` and `distribution_shift` are listed families no rule can
produce.

**Supersedes #4 in scope** — #4's rename is one piece of this.

---

## 16. ✅ ANSWERED 2026-09-17 — GR00T VRAM fit (RESULTS.md R-021)

**Fits, only in full bf16:** load on CPU, cast to bf16, then move to GPU. That
takes 5.87 GiB, leaving 1.38 GiB free. Every stock path OOMs at 7.2 GiB, including
`model_params_fp32=false`. Smoke test 5/5 on libero_spatial task 0. Next: GR00T
harness parity, and whether bf16 changes behaviour vs the shipped F32 config.

### Original entry

3B params, F32 tensors listed, bf16 loading required. `b198613` measured 6.44 GiB
weights against ~7.34 GiB free on our 8 GB card. ~20 min test before committing
GR00T as the third policy. Also a candidate silent failure (vla-7f): trained on
`IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot`, and whether that shares
the 180° orientation convention `LiberoProcessorStep` assumes is unverified.

---

## 17. Decisions REPORTED via vla-81 — please confirm they are yours

vla-81 relayed these as owner decisions. They were not made in this session,
so they are recorded here as reported, not assumed.

- **#15(b) naming / label swap → LEAVE AS-IS, DOCUMENT.** The family set is to be
  revised with input from manifest consumers, so renaming rules likely to be
  replaced is churn. vla-81 phrased it as one question covering its swap finding
  AND my #4 rename, and said not to over-read it as covering #4 as I phrased it.
  **Confirm this also closes #4.**
- **#15(c) utility experiment → DOCUMENTED AS NEXT STEP, NOT SCHEDULED.**
  Consequence: no "taxonomy is validated" claim anywhere until it runs.
- **#15(a) labels vs predicate vectors → WITHDRAWN.** Reporting format belongs to
  the revised design. Test 3 held pending consumer input.
- **New governing doc:** `docs/TAXONOMY_FAMILY_GUIDELINES.md` (vla-81), marked
  read-before-any-taxonomy-decision. Two requirements for the revised set:
  families must be *relevant to consumers* and *meaningful viewed separately*.
  The second rules out first-match-wins over overlapping predicates by
  construction — `manipulation`'s real definition is five conditions, four of
  them living in other rules.

---

## 18. MuJoCo: a shared pinned simulator IS viable (vla-7f research)

No primary source shows two LIBERO checkpoints needing conflicting versions.
Two documented effects; a pin must sit below both:

| band | effect | evidence |
|---|---|---|
| ≥ 3.10.0 | robosuite 1.4.0 crashes (mj_fullM signature) | GR00T setup pins 3.3.1 for this |
| ≥ 3.4.0 | box-box contact fix corrupts spatial task-5 init state, **every policy** — SmolVLA 80→28 | lerobot#4390, LIBERO#141; causal: re-settled init file restores 84% |
| ≥ 3.3.3 | render/lighting shift, pixel gap 1.9→34.5 on Object | MINERVA README, LIBERO#88 |

Pins in the wild: openpi 3.2.3, GR00T 3.3.1, MINERVA 3.3.2. LeRobot's own
uv.lock resolves 3.8.1 — on the broken side.

**We run 3.3.7:** below the physics break, past the lighting change.

**Consequence for #10's architecture:** one simulator per box is NOT required. A
single canonical pin at **3.3.2** satisfies every documented good setting, which
also makes MINERVA's "spare simulator" unnecessary. Each box should declare the
MuJoCo version its training data was rendered at. Now encoded in the gate as
version bands.

**Decision:** adopt 3.3.2 as the canonical simulator pin? That replaces #11's
full MuJoCo A/B with a single-arm move — though our own Object score (85, closest
to published) argues lighting is not our main gap.

---

## 19. ❌ WITHDRAWN 2026-09-16 by vla-81 — `num_steps_wait=50` arm

**Do not schedule.** Two reasons, both verified by vla-7f:
1. **Not runnable as a flag.** `num_steps_wait` is not a field of the LeRobot
   `LiberoEnv` config and is not passed through `create_libero_envs`
   (only `libero.py:127,162,351`). It would need a code change.
2. **The hypothesis is weak.** pi0.5 scores 97.5% through this same env at the
   default 10, and every reference harness (openvla, openvla-oft, openpi,
   VLA-Adapter, MINERVA) uses 10. The MolmoAct2 doc sentence has no numbers
   behind it. At most a SmolVLA-specific fragility, not an env defect.

**Replacement, no GPU:** vla-7f may propose a CPU-only probe — reset each
spatial task, step no-ops, log object-pose drift at step 10 vs 50, with
libero_object as control. Reads `sim.data`, needs no rendering (can run under
`MUJOCO_GL=osmesa`, so no clash with the running GPU arm). vla-7f will ask
before running it.

### Original proposal (kept for the record)

LeRobot's own docs (`docs/source/molmoact2.mdx:318-321`): "`num_steps_wait=10`
does not reliably let the LIBERO scene stabilize and can degrade measured
success. All LIBERO evaluation results reported here use `num_steps_wait=50`."
Default is 10; **all five of our stored configs used 10.**

Why it fits: a settling defect hits every reset uniformly — matching the **flat**
libero_spatial per-task profile (6 4 5 5 6 6 7 6 6 5) versus Object's real
competence profile — and spatial tasks are *defined* by inter-object geometry.
Same physical phenomenon as #18's task-5 finding.

Proposal: spatial, 256, seed 1000, **nas=10**, `num_steps_wait=50`, vs 73.0%.
One variable. Caveat: the doc note is about MolmoAct2, not SmolVLA; mechanism is
policy-independent. **GPU — needs your go-ahead, and a ping to solver first.**

**Sequencing:** the nas=1 arm now running is predicted by SmolVLA Table 13 to
land ~73% (spatial 89 at both nas=1 and 10). If it does, this is the natural
next arm. Prediction pre-registered in
`experiments/repro/runs/res256_nas1_seed1000/PREDICTION.md`.

---

## 20. Reframe F8/F9's headline gap — our checkpoint is not the published model

> **Status 2026-09-23 — moot.** The architecture finding is recorded as R-011, and
> SmolVLA was vetoed on 2026-09-18. `FINDINGS_DRAFT_F8_F10.md` was never applied and
> carries a banner saying so. No decision is needed unless SmolVLA returns.

**Escalated from vla-7f and vla-81**, who both think this is yours to decide:
F8 is a headline project claim and `FINDINGS.md` has no live owner.

**Verified this session:** `HuggingFaceVLA/smolvla_libero` ships
`num_vlm_layers: 0`. The class default is 16, and
`smolvlm_with_expert.py:102` only truncates when >0 — so 0 keeps **all 32**
VLM layers, and the action expert is built to match (:112). vla-7f's weight
count: ~605M, vs the paper's 16-layer ~0.45B headline. No provenance in the
checkpoint (empty safetensors metadata, card says "datasets: unknown").

**Consequence:** the paper's 90/96/92/71 (87.3%) is for a *different
architecture*. There is **no published LIBERO number for our checkpoint**.
"67% vs 87.3%" compares different models.

Closest match: the paper's full-depth (N=32) ablation rows — trained WITHOUT
robotics pretraining, n unverified:

| | Spatial | Object | Goal | Long |
|---|---|---|---|---|
| ours (256, nas=10) | 73 | 90 | 68 | 37 |
| N=32 ablation, nas=10 | 89 | 94 | 91 | 57 |
| deficit | 16 | 4 (n.s.) | 23 | 20 |

So the "36-pp long-horizon gap" is mostly an artefact of the wrong reference;
the real pattern is Object ≈ matched, other three ~16–23 pp down. Even post-F9,
the spatial gap's 95% CI is [6.5, 27.5] at n=100 per side.

**Still valid:** every paired within-checkpoint result (F9's +17 pp, the nas=1
arm vs nas=10).

**Decision:** reword F8/F9 to "no published target for this checkpoint;
nearest reference is the N=32 ablation" and re-state the gap against it?

**Drafted, not applied:** `FINDINGS_DRAFT_F8_F10.md` holds the F8 amendment
plus a new **F10 — residual deficit is geometric, not photometric**, which vla-81
relayed as an owner request. The two must land together. All CIs re-derived
and verified this session. Approve → I apply both to `FINDINGS.md`.

**Gate:** now surfaces `num_vlm_layers` as INFO so architecture reaches
provenance.

---

## 21. Goal-suite MuJoCo 3.3.2 vs 3.3.7 A/B — cheapest discriminating GPU arm

> **Status 2026-09-23 — run, and moot.** R-018 ran it: 74% vs 68%, paired p=0.26,
> not significant. SmolVLA has since been vetoed, so nothing waits on it.

From vla-7f's `docs/POLICY_SIM_COUPLING.md` §7.2 (uncommitted). There are four
community reproductions of **this exact checkpoint** (lerobot#2354 ×3,
lerobot#3264). Our spatial, object and long scores fall within their range.
**Goal is the outlier: ours 68 vs theirs 81 / 83 / 87.** All four ran MuJoCo
3.3.2; we run 3.3.7. openvla#282 documents +17 pp on Object for OpenVLA from that
version change alone.

So a **goal-only** paired A/B at 3.3.2 vs 3.3.7 is far cheaper than #11's full
400-episode protocol and targets the one suite that doesn't fit. It needs a
separate venv (per §R7) or a reinstall.

**Interaction with F10:** goal is also where F10's language-vs-geometry
discriminator runs (#3). If 3.3.2 closes goal's gap, part of F10's residual was a
version effect after all. That is worth knowing before landing F10's claim.

**Size it above the noise (vla-7f).** The nas=1 run showed 32/100 discordant
episodes with no net effect. Our goal gap to the community repros is 13–19 pp,
which is outside the ~10 pp single-arm noise band but not by much. So goal is a
**lead, not an established defect**. At 10 eps/task the A/B risks landing inside
the noise; **use 20+ eps/task** (goal only: ~200 episodes per arm, roughly
2 × ~70 min ≈ 2h20m GPU at the observed ~43 s/episode).

**Decision:** approve the goal-only A/B at 20 eps/task instead of #11's full protocol?

---

## 22. ✅ FIXED 2026-09-16 — Camera perturbation (was: every yaw arm invalid)

**Fixed with option (i):** every camera knob is now a rigid motion of LIBERO's
ORIGINAL camera pose about the point it looks at (optical axis ∩ the task
objects' height plane). No re-aim. Both guards now run on a present knob,
including 0. `_resettle` returns the post-perturbation frame via
`_format_raw_obs` and raises instead of falling back.

**Verified:** `experiments/camera_perturbation_test.py`, 20/20 on object and
spatial scenes — yaw 1e-6 is pixel-identical to yaw 0 (was 44–53); yaw 5 turns
the heading exactly 5.000° with pitch unchanged; pitch 5 looks exactly 5.000°
more steeply down and raises the camera; dist +0.10 backs away exactly 0.1000 m
on-axis; reset frame now matches the perturbed render (diff 0.01–0.02, was
44–53). The test caught an inverted pitch sign on first run.

**Still required:** every perturbed arm of the 20260916-0015 campaign must be
re-run before any camera result or manifest row ships. Nominal arms unaffected.

### Original report

Found by vla-7f from the code. **Verified this session with renders.**

`_apply_perturbation` (`envs/libero_env.py:290-313`) has two defects:
1. `if yaw or pitch or dist:` skips the block at yaw=0.0, so the "yaw 0" control
   never runs the camera code.
2. For any nonzero yaw it **replaces** the camera orientation with a re-aim at a
   guessed look-point `[0, 0, z*0.5]`, which is not where LIBERO's camera points.

Measured, task 0:

| scene | yaw=1e-6 (≈ no rotation) | yaw=5 |
|---|---|---|
| libero_object | view rotated **12.0°**, pitch-down 31.9→19.9°, pixel diff **53.1** | 12.8°, pixel diff 52.3 |
| libero_spatial | view rotated **11.8°**, pitch-down 38.9→50.7°, pixel diff **44.1** | 12.3°, pixel diff 45.8 |

**A millionth of a degree of "yaw" changes the image as much as 5°.** The knob
measures the re-aim, not yaw. Pitch moves in opposite directions per scene.

Campaign success by arm — a cliff at 5 and flat after, the signature of a
constant confound rather than a dose-response:

| suite | nominal | yaw 0 | yaw 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|
| spatial | 65 | 80 | **10** | 12 | 8 | 12 |
| object | 90 | 88 | **0** | 2 | 5 | 5 |
| goal | 70 | 80 | **12** | 18 | 18 | 8 |
| long | 52 | 64 | **8** | 12 | 8 | 4 |

**Invalidated:** every yaw-sweep result, every compound/attribution arm, and
**all 12 manifest rows** (3 per suite, every one attributes its trigger to
`camera_yaw_deg`). Nominal-arm results are unaffected.

**Second defect found while verifying:** the frame returned by `reset()` is
**stale** — rendered before the perturbation. `_resettle` only refreshes
`_raw` if `_to_lerobot_obs` exists, and here it doesn't. The policy's first
observation of every perturbed episode shows the unperturbed scene.

**Fix (not applied):** compose the rotation about world z (through the camera's
own look point) onto the ORIGINAL `cam_quat` instead of re-aiming; run the
block whenever the knob is present, including 0; refresh the observation after
resettling. Then a render test: yaw=0 and yaw=1e-6 must be pixel-identical.

**Second, independent confirmation (vla-81), from stored traces.** Angle between
each arm's recorded agentview rotation and the yaw-0 arm's:

| suite | "yaw 5" | 10 | 15 | 20 | translation @ 20 |
|---|---|---|---|---|---|
| object | 12.99° | 15.61° | 19.19° | 23.29° | 31.1 cm |
| spatial / goal | 12.81° | 15.45° | 19.06° | 23.19° | 22.9 cm |
| long | 8.14° | 11.88° | 16.31° | 21.00° | 21.1 cm |

The first "5°" buys ~13° of real rotation plus 6–8 cm of translation; each later
5° buys only 2.6–4.1°. **Threshold, not dose-response.** "5° yaw takes object to
0%" should read "~13° rotation + ~8 cm translation takes object to 0/40".

**Methodological finding:** the grid has **no samples between 0° and ~13°**, and
that is where the transition sits. A fixed 5/10/15/20 grid cannot locate this
policy's boundary — the concrete case for the adaptive sweep (PLAN D10).

**Fix options — they disagree:**
- **(i) Rotate the ORIGINAL camera orientation about world z** (my
  recommendation). Yaw 0 is then genuinely LIBERO's camera, so the control has
  the policy's full ~88% dynamic range and small angles are actually small.
- **(ii) Run the re-aim at yaw=0 too** (vla-81's preference: a one-line guard
  change that keeps the documented framing intent). But the control would then
  itself be the ~12° re-aimed camera, which this data suggests the policy
  already fails on (~0–12%). The sweep would measure degradation from an
  already-broken baseline, and it could no longer tell us anything about small
  real-world misalignments. It also stops matching the published LIBERO view.

Either way: rename arms to their **measured** orientation delta, and add a render
test.

**Blocks the goal run (#3):** `run_goal_mj332.sh` job B calls `e2e.py`, which
runs the yaw sweep. Launched before the fix, it reproduces this bug for 42 min.
Job A (plain `lerobot-eval`) is unaffected.

---

## 23. ✅ FIXED 2026-09-17 — Starting layout depended on reset history, not seed

LeRobot picks each episode's init state from a COUNTER advanced on every reset
(`init_state_id += _reset_stride`, `lerobot/envs/libero.py:346`); the seed does
not select it. In our harness that caused two silent defects:
- **Resume:** `run_cell` skips cached episodes without resetting, so after a
  resume every later "seed N" ran on a different layout than its rollout id
  claims.
- **Pairing:** arms sharing one env object (nominal, then yaw 5, ...) were never
  on the same layouts, so paired comparisons across perturbation levels in the
  20260916-0015 campaign were not actually paired.

**Fix:** `LiberoEnv.reset` pins `init_state_id = seed % n_init_states` before
each reset, matching lerobot-eval (sub-env i starts at init state i), and records
`init_state_index` in `scene_descriptor`. **Test:**
`experiments/init_state_pinning_test.py` — seed 3 after three resets is
bit-identical to seed 3 fresh, on object and spatial.

Also added: `LiberoEnv(obs_size=...)` render size (in `identity()`, so env ids
change), and `LeRobotPolicy(policy_overrides=...)`, applied at CONSTRUCTION —
tinyflow builds its temporal ensembler in `__init__`, so setting it afterwards
did nothing. The earlier MINERVA harness probe had run without ensembling.

---

## 24. ✅ DONE — overnight queue 2026-09-17 (results: `RESULTS.md` R-013 to R-019)

Chosen without user input (user asleep): harness validation first, then the
MINERVA reproduction. **Not committed** — code state is recorded instead in
`experiments/repro/runs/overnight_20260917/code_state/` (HEAD + uncommitted patch
+ freezes).

1. Harness self-tests (camera, init-state pinning, segmenter). A failure skips 2–3.
2. **Harness parity:** MINERVA through our harness, object+spatial, 10 eps/task,
   compared to lerobot-eval's first 10 eps/task on the same init states.
3. **Camera sweep, fixed code:** MINERVA, object, yaw 0/2/5/10/20 + pitch 5.
4. **MINERVA reproduction:** authors' exact command, 2,000 episodes.

Morning: `python3 experiments/summarize_overnight.py`. Log:
`experiments/repro/logs/overnight_20260917.log`. Resume by re-running
`experiments/overnight_20260917.sh`. Not in tonight's window: the SmolVLA goal
runs (#3, #21).

---

## Not decisions — things already settled this session

- O1 (alphabetical target in the segmenter), O2 (lift gated by closure) and O3
  (`hold_steps` < travel time) are **fixed**, with a regression test at
  `experiments/phase_segmenter_test.py` (4/4 passing).
- TRANSPORT over-firing is **resolved**: libero_object 260/260 → 87/260, with
  every success still showing TRANSPORT in all three suites.
- The video visualiser is **built and working** —
  `experiments/visualise_episode_video.py`, output in `viz/`. See #7 for its one
  caveat.

---

## 25. Which fourth policy — and whether to rent a GPU for π0.5 / OpenVLA-OFT

**Open. Needs your call.** Full survey with measured VRAM in
`MODELS_AND_COMPUTE.md` §R9.

> **Status 2026-09-23 — now a budget question only.** Options 1 and 2 below were
> tried: **neither π0-FAST nor π0.5 fits this card** (R-032). π0-FAST needed three
> harness fixes to get far enough to show that (R-034), and π0.5 hit OOM at
> 7.40 / 7.53 GiB. So the choice is whether to rent ≥16 GB for π0.5 / π0-FAST /
> OpenVLA-OFT, or add X-VLA locally. The "fits locally" and "VRAM probe?" text
> below is overtaken.

The roster today is GR00T N1.7 (priority), MINERVA (control), SmolVLA
(vetoed 2026-09-18 — never reproduced). A fourth policy is wanted that is
frontier, general and robust.

**The recommendation, to accept or override:**

1. **π0-FAST** (`lerobot/pi0fast-libero-v044`, 2.92 B, 5.44 GiB) — **fits
   locally with more headroom than GR00T.** Scores only 82.5% on LIBERO, but it
   is one of the four reference models LIBERO-Plus used to assign difficulty
   levels. Running it on R-026's 623 variants tests whether those levels mean
   anything on our stack — currently an unverified prior we caveat in every
   entry. Different architecture family too (autoregressive FAST tokens).
2. **π0.5** (`lerobot/pi05_libero_finetuned_v044`, 3.62 B, 6.74 GiB) — the
   frontier choice: 97.5%, the only documented LeRobot reproduction. **Borderline
   on 8 GB** (0.8 GiB headroom vs GR00T's 1.4). *Decision needed: spend ~10 min
   on a VRAM probe to settle it?*
3. **X-VLA** (`lerobot/xvla-libero`, 0.88 B) — cheap third architecture.

**The compute question.** *Decision needed:* if rented compute is available,
≥16 GB unlocks OpenVLA-OFT 7 B — the strongest LIBERO-Plus reference model, and
the only pairing that answers "does robustness training buy robustness", since
the LIBERO-Plus authors released the same model fine-tuned **on** LIBERO-Plus
(`Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata`). That experiment cannot
be run on any model that fits this card.

**Not blocking anything.** The failure corpus and taxonomy work proceed on GR00T
and MINERVA regardless.
