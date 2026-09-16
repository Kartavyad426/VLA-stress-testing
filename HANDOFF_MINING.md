# Handoff — mining layer, 2026-09-16

For whoever picks this up. `HANDOFF.md` has overall project state; this covers
the mining layer specifically, which is where the live work is.

---

## Where things stand

**The campaign ran.** `runs/camp_20260916-0015_*`, 965 mineable traces across
four suites, 3h21m. 298 successes, 667 failures — **every failure is a timeout**,
because LIBERO only ends an episode on success or step cap.

**Three suites mine cleanly** (spatial, object, long). **`libero_goal` abstained
on 200/260** — see "must re-run" below.

**Mining is re-runnable on CPU** over stored traces. Change a threshold, re-mine
965 traces in seconds. That property is why the fixes below could be validated
without touching the GPU.

---

## What changed today, and what is still wrong

### Fixed — BDDL region→body resolution (`envs/libero_env.py`)
Articulated tasks name a **region**, not a body:
`'wooden_cabinet_1_middle_region'` vs body `'wooden_cabinet_1_cabinet_middle'`.
Prefix matching failed, six of eight goal tasks resolved **zero** objects, and
the miner abstained. `_resolve_body()` strips `_region` and prefers the body
carrying both the fixture prefix and the part token.

**Benefit:** unblocks ~200 goal traces, and any future task with drawers,
cabinets or stoves. Without it the whole articulated-manipulation half of LIBERO
is invisible to us.

### Fixed — target selection was alphabetical (`mining/signals.py`)
`_target()` did `sorted(d)[0]`, destroying BDDL order. `obj_of_interest` lists
the **manipuland first**; on `['cream_cheese_1', 'akita_black_bowl_1']` sorting
picks the *bowl* — the destination — so **every distance was measured to the
wrong object**. Now takes dict insertion order.

### Fixed — gripper state uses INTENT, not just response
`action[6]` is what the policy commanded (+1 close, −1 open, binary in the
demonstrations); `gripper_qpos` is how the hardware responded. Measured on 60
real rollouts, two steps after a close command:

```
fully closed  n=4599  median aperture 0.0059   (nothing between the fingers)
blocked open  n= 812  median aperture 0.0597   (an object is)
```

Cleanly bimodal. So `holding` can be inferred **without privileged object
state** — which is what a real robot has. Grasp attempts now count rising edges
of *commanded* closure: no lag, no threshold. (Review finding A1 noted the old
version was "correct by accident".)

### ⚠ STILL WRONG — TRANSPORT over-fires on failures

Persistence (`hold_steps=6`) was added after a first version fired on 258/260.
Current state:

| suite | TRANSPORT | of successes |
|---|---|---|
| libero_spatial | 224/260 | **79/79** ✓ |
| libero_object | **260/260** ⚠ | 79/79 ✓ |
| libero_10 | 170/185 | 52/52 ✓ |

**Good:** every success shows TRANSPORT — correct by construction, a real
validity check.

**Bad:** `libero_object` fires on **all 260**, including all 161
`visual_grounding` failures. A policy that went to the wrong place should not be
holding anything. Either `hold_steps` is still too short, or the object suite
has a gripper behaviour the test misreads.

**Next step:** take a known `visual_grounding` failure from libero_object, plot
it with `experiments/visualise_episode.py`, and look at where `by_gripper`
fires. The visualiser exists for exactly this.

---

## MUST RE-RUN: libero_goal (~42 min GPU)

The resolver fix **cannot** be applied retroactively: those 200 traces never
recorded `_gt_eef_to_object`, so the signal does not exist in them.

**"Capture once, mine many" only holds for signals that were captured.** Capture
privileged state generously — you can drop fields later, you cannot invent them.

```bash
MUJOCO_GL=egl .venvs/lerobot/bin/python experiments/e2e.py \
  --suite libero_goal --tasks 8 --seeds 5 --yaw-levels 0,5,10,15,20 \
  --run-id goal_rerun
```

---

## Other cheap deterministic parts, audited

Asked for explicitly. Ranked by how likely each is to produce a wrong answer.

| # | Shortcut | Where | Risk | Better |
|---|---|---|---|---|
| 1 | **`hold_steps=6`** | `phases_libero` | **HIGH — actively wrong now** | fit from demonstrations; or require object displacement to correlate with eef displacement |
| 2 | `pregrasp_m = 0.10` | `phases_libero` | HIGH — a guess in metres | `fit_from_demos()` exists and **has never been run** |
| 3 | `LOST_TARGET_M = 0.10`, `WRONG_OBJECT_M = 0.09`, `0.6` ratio | `classify` | HIGH — toy-derived, decide family boundaries | fit per suite; report sensitivity |
| 4 | `REPEAT_SPREAD_M = 0.015` | `classify` | MED — separates recovery from manipulation | fit from observed retry spreads |
| 5 | `lift_m = 0.015` | `phases_libero` | MED — now partly redundant given gripper test | keep as the corroborating route |
| 6 | eef offset via `qpos[0] += dy*2.0` | `libero_env` | MED — "approximate by design"; achieved offset is recorded, so honest but crude | solve IK, or sweep the achieved offset |
| 7 | camera look-point `[0, 0, z*0.5]` | `libero_env` | MED — a guessed table centre; wrong look-point couples yaw with framing | derive from the BDDL table region |
| 8 | `divergence tol = 0.03` | `phases` | LOW — secondary signal; survey argues **drop it entirely** | §2.6 says dropped, not caveated. **Open inconsistency.** |
| 9 | control threshold `0.30` | `control.py` | LOW but arbitrary | justify against per-task baselines |
| 10 | cluster = `(family, active knobs)` | `cluster` | LOW — deliberately simple | embedding clustering only if the name-based check (F6a) shows families are arbitrary |

**Pattern worth noting:** every one of items 1–5 is a distance or count
threshold carried over from the toy, where they were chosen to make a planted
fault reproducible. They are physically meaningful in metres and LIBERO is
metric, so they *transfer* — but none has been *validated* on LIBERO.

---

## Validity checks that exist

- **Oracle gate** — 3/4 planted faults + control. Toy only.
- **Every success shows TRANSPORT** — the best real-data check we have.
- **Abstention on missing keys** (G8) — caught the goal-suite gap honestly
  rather than producing 200 confident wrong diagnoses.
- **`experiments/visualise_episode.py`** — renders one rollout with every
  signal the classifier uses on one axis. Use it before arguing about a label.

## What has NO validity check

- **Family assignment on LIBERO.** No fixture: demos cover only the success
  path, and degrading a policy gives a known *trigger*, not a known *family*.
  κ is therefore the sole check — and κ cannot distinguish a good taxonomy from
  a merely consistent one at any threshold (F6a).
- **`recovery`** fires 4–8 times in ~700 failures. Consistent with "rare" and
  equally with "the detector does not work".
