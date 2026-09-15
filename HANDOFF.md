# HANDOFF — state of the VLA Stress-Test project

> **[UNSOURCED — do not cite]** the sim-to-real rank-correlation figure (Spearman 0.4–0.7) and the claim that *severity ordering transfers worst* were carried forward from an early survey and **no primary source has been located** (flagged 2026-09-15). The design decisions they motivated stand on their own reasoning; the numbers must not appear in a readout until sourced.

**Written 2026-09-15.** For resuming after a context break. Read this first,
then the doc it points you at for whatever you are doing.

---

## 🔴 IN FLIGHT RIGHT NOW — baseline screen

Started 2026-09-15 14:03. SmolVLA on vanilla LIBERO, 4 suites, 10 episodes/task,
**400 episodes**, `n_action_steps=10`, seed 1000, mujoco 3.3.7 (healthy — F2).
**First run of this checkpoint on unbroken physics.**

**Check on it:**
```bash
cd /home/imerit/Documents/Code/VLA
D=experiments/repro/runs/nas10_seed1000
[ -f experiments/repro/logs/baseline.done ] \
  && echo "DONE rc=$(cat experiments/repro/logs/baseline.done)" || echo RUNNING
find $D/videos -name '*.mp4' | wc -l          # progress out of 400
cat $D/eval_info.json                          # results (written ONLY at the end)
```

**Survives a session close.** SIGHUP is ignored (signal mask `0x1001001`) and it
already outlived its launching shell. Only an explicit SIGKILL to the process
group would take it.

**The MONITOR does not survive** — monitors are session-scoped. After a session
close there is no notification on completion, stall or crash; check manually
with the commands above.

**If it dies mid-run:** `eval_info.json` is written **once, at the end**, so the
aggregate is lost — but per-episode videos and `eval.log` survive and the run is
restartable from scratch (~1–2 h). Nothing else depends on it.

**How to read the result:** it is a **SCREEN, not a gate**. At n=100/suite a
5 pp break goes unflagged 66% of the time. A pass means *"screen passed, gate
not yet run"* — never "consistent with published". Compare against ~90 / 96 /
92 / 71 (spatial/object/goal/long), and note the two upstream non-reproductions
at 73.25% and ~67% (F3).

**This run is NOT mineable** — it goes through `lerobot-eval`, which stores a
success bit and an mp4. Accepted deliberately: its job is the reproduction
question. See the next section.

---

## 💡 PROPOSED — a recording wrapper, before Phase B

**Problem:** `lerobot-eval` output cannot be mined (no per-step state), so the
plan had us choosing between *their evaluator* and *our traces*.

**The choice is not forced.** LeRobot registers environments:

```python
@EnvConfig.register_subclass("libero")
```

Register a **`libero_traced`** subclass that records as it runs, then
`lerobot-eval --env.type=libero_traced` gives **their loop and protocol AND our
traces**.

Three wins:
1. **Removes the one genuine duplication** — `runner.rollout()` currently
   re-implements LeRobot's loop.
2. **Captures what LeRobot discards** — the subclass sees robosuite's RAW
   observation before the 7-key filter, which is where per-object state lives,
   including `<object>_to_robot0_eef_pos` computed by the simulator. That would
   replace our fragile `sim.data.body_xpos` name-matching outright.
3. **Every phase becomes mineable**, including a re-run of the baseline.

**Caveats:** counterfactual probes need same-seed re-runs of a chosen instance,
which is our access pattern rather than `lerobot-eval`'s — keep a thin direct
path for those. And it couples us to LeRobot internals, so the LeRobot version
belongs in `semantic_runtime` (keyed), not `runtime`.

**Sequence it BEFORE Phase B.** Phase B is 5–25 GPU-hours whose traces we want
to keep; running it first and wrapping later would waste the lot.

---

## Read in this order

| Doc | What it is |
|---|---|
| `VLA Scenario Testing.md` | The original proposal (not ours; the client brief) |
| **`PLAN.md`** | What we build and why. Decisions D1–D5 in §0, positioning in §0b |
| `ARCHITECTURE.md` | Data flow, invariants, debugging playbook. Describes code **as built** |
| `docs/COMPONENTS.md` | Every component, where it lives, **verification status** |
| `docs/WHAT_OUR_CODE_DOES.md` | Why we wrote any code when `lerobot-eval` exists |
| `docs/OUR_MINING_APPROACH.md` | The mining layer end to end — the differentiator |
| `docs/FAILURE_MINING_METHODS.md` | Field survey. **COMPLETE — §0–§9, all five stages.** See its status banner: §3.5 (taxonomy validation), §3.3 (LIBERO-Plus answers AS-3), §7.1/§8.1 (`ddmin` for knob interactions) |
| `docs/LANDSCAPE.md` | Competitive survey, 238 sources, + adoption ledger |
| `FINDINGS.md` | F1–F4. Things established that are not in the proposal |
| `IMPLEMENTATION.md` | Build plan, wave by wave, with status |
| `NEXT_STEPS.md` | Experiment order + blockers |
| `docs/reviews/` | Peer reviews from session `critical` (findings #1–#19, DG-1–DG-11, LE-1–LE-5) |

---

## Where we actually are

**Working:** the toy harness end-to-end (stress → mine → cluster → manifest),
oracle gate **3/4 + control PASS**, wave verification **7/7**.
`lerobot-eval` installed and working on SmolVLA. LIBERO adapter **captures**
correctly through our runner.

**Not working:** the mining layer on LIBERO. It **abstains** —
`missing state keys: ['_gt_ee_to_obj', 'gripper', 'holding']`. Detectors are
bound to toy key names. This is correct behaviour (G8) but it means we can
record real LIBERO rollouts and learn nothing beyond pass/fail.

**Never done:** any real VLA failure has never been mined. Everything is
validated on a toy with planted faults.

### The five things between here and a real result

1. **LIBERO `PhaseSegmenter`** — key mapping (`_gt_eef_to_object` is a *dict*,
   `gripper_qpos` is finger positions in metres) plus **derive `holding`** from
   gripper closure + object lift. That derivation is itself a detector.
2. **A real VLA policy adapter** — SmolVLA or VLA-Adapter through *our* loop.
   SmolVLA has only ever run through `lerobot-eval`.
3. **C9** — per-task environment control (not built).
4. **LIBERO-plus adapter** — nothing perturbable exists. No sweep, no boundary,
   no counterfactual probe on real data. Today we could measure nominal success
   only, which `lerobot-eval` already does.
5. **D10** — the surrogate for adaptive sweeps (only once a boundary exists).

(1) and (2) gate any result. **(4) gates everything about robustness.**

---

## ⚠ UNVERIFIED — check before trusting `_gt_` object state

**Raised 2026-09-15**, prompted by a peer session's finding that *the dangerous
interfaces are not the ones you never touch, but the ones you touch **partially***.

**LeRobot extracts exactly 7 keys from robosuite's raw observation**
(`envs/libero.py:291-302`): two camera images, `robot0_eef_pos`,
`robot0_eef_quat`, `robot0_gripper_qpos`, `robot0_gripper_qvel`,
`robot0_joint_pos`, `robot0_joint_vel`. **Everything else is discarded**, and we
never see the raw dict — our adapter receives LeRobot's processed observation.

robosuite normally also publishes **per-object state** in that raw observation,
including `<object>_pos`, `<object>_quat` and **`<object>_to_robot0_eef_pos`**
computed natively.

We instead re-derive object poses and eef-to-object distances from
`sim.data.body_xpos` with BDDL name matching. **That is the code path that
already failed once** (LE-2: a `link` substring filter removed drawers and
cabinet doors, so `_gt_nearest_object` named the wrong thing and the
spatial_reasoning rule fired on it).

**To check** (needs a live env — do NOT run while a campaign is using the GPU):

```python
inner = env._env.unwrapped._env          # OffScreenRenderEnv
raw = inner._get_observations()          # robosuite's RAW obs, pre-LeRobot
print(sorted(raw))                       # <- the whole interface, ~4 minutes
```

**If `<object>_to_robot0_eef_pos` is present**, prefer it over our derivation:
it is computed by the simulator, needs no name matching, and removes an entire
class of silent error from every distance-based detector.

**Status: UNVERIFIED.** Deliberately not run while the baseline campaign holds
the GPU — a second MuJoCo env could OOM an in-flight 400-episode run.

---

## Highest-value unknown

**Is LIBERO-plus a knob API or a fixed corpus of 10,030 pre-generated
instances?** If it is a corpus, *revert-one-knob* may not be constructible — and
counterfactual attribution is the mechanism the whole project rests on. Check
this before building the LIBERO-plus adapter. Raised by `critical` as AS-3,
still unanswered.

---

## Hard-won facts that are expensive to rediscover

- **Plug the laptop in.** On battery the GPU throttles to 180 MHz / 15 W — about
  **25× slower**. `nvidia-smi` reports `utilization.gpu` at 100% throughout.
  Check `pstate` and `clocks.sm`, never utilisation. Cost us and a peer session
  hours; they logged a run as "system_failure, cause unknown".
- **`mujoco<3.4.0` is pinned and must stay pinned.** 3.4.0's collision fix takes
  SmolVLA 80% → 28% on `libero_spatial` task 5. A fresh install gets 3.8.1, the
  broken side. (F2)
- **π0.5 does not fit on 8 GB** — weights alone, no simulator. The reproduction
  anchor needs a ≥16 GB card. (F4)
- **Logs go in the repo, not the scratchpad.** A scratchpad result did not
  survive a session restart, and the "exit 0" we saw was the launcher wrapper,
  not the eval.
- **`n_action_steps=10`** for SmolVLA — 17 steps/s vs 3.95 at the shipped
  default of 1. Chunking plateaus after that; the simulator becomes the
  bottleneck.
- **Per-episode wall time is not a throughput metric.** Successful episodes end
  early, so it conflates speed with success. Use steps/s.

---

## Open decisions

**Yours:** readout audience (technical vs commercial). Anchor compute is being
arranged by you via a free-tier channel.

**Mine unless you object:** tier-3 LLM local (audit-trail reproducibility);
remediation arm A by re-rendering existing demos rather than scripting new ones.

**Unresolved by anyone:**
- **Family assignment has no fixture on LIBERO.** Demos cover only the success
  path; degradation gives a known *trigger* but not a known *family*. That makes
  κ the sole validation of the classifier, and **κ ≥ 0.5 is thin for that role**.
- How well `failure_conditional` transfers sim-to-real (Spearman 0.4–0.7 **[UNSOURCED]**
  published; severity ordering transfers worst).
- Do recovery rows ship in a first client manifest? The family is unvalidated.

---

## Peer sessions

`critical` — reviews the work; has produced five review docs under
`docs/reviews/`. Do not ask it to implement; the separation is deliberate.
`solver` — different project (action segmentation), **shares this GPU**.
Standing agreement: ping before either of us starts heavy GPU work.

---

## If you resume the research doc

Original brief is in the conversation; the short version: a **methods** survey
(mechanisms, not competitive positioning), five stages, readable opening,
real URLs, mark unverified claims, prefer primary sources, negative results
especially valuable. Scope a resumed pass to **§3 classification, §4 clustering,
§5 attribution** plus cross-cutting and "what we could adopt" — and tell it
§0–§2 already exist.

Two headings in the finished part worth reading before defending our design:
- §2.1 "Temporal action segmentation — **the wrong default tool**, with reasons"
- §2.6 "Trajectory divergence from a reference — and why it should be
  **dropped, not caveated**" — we currently keep it as a caveated secondary
  signal in `mining/phases.py`. If the argument holds, that is a code change.
