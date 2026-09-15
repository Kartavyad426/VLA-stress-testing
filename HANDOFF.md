# HANDOFF — state of the VLA Stress-Test project

**Written 2026-09-15.** For resuming after a context break. Read this first,
then the doc it points you at for whatever you are doing.

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
| `docs/FAILURE_MINING_METHODS.md` | Field survey. **INCOMPLETE — §3–§5 missing** |
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
- How well `failure_conditional` transfers sim-to-real (Spearman 0.4–0.7
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
