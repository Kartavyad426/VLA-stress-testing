# Next Steps — experimentation order, blockers, requirements

**Written 2026-09-11**, after Waves A/B/C were implemented and verified
(`experiments/wave_abc_test.py`, 7/7 pass; oracle gate 3/4 + control PASS).

Read with: `PLAN.md` (what and why) · `IMPLEMENTATION.md` (code changes) ·
`ARCHITECTURE.md` (as-built map).

---

## 0. BLOCKERS — read before planning anything

### B-1. The laptop is on battery again. **Nothing timed is valid right now.**

Checked at the time of writing:

```
AC online      : 0          <- ON BATTERY
clocks.sm      : 180 MHz    <- idle clock; ~1400-2600 on AC
```

At this state the GPU runs roughly **25x slower** — independently confirmed by
the peer session `solver` on identical work (601.7 s vs 24.0 s) and by us (22.8
s/step vs ~0.06 s/step). It caused `solver` to log a run as *"system_failure,
cause unknown"*.

**`nvidia-smi` will report `utilization.gpu` at 100% throughout.** Utilisation
means a kernel is resident, not that it is running at speed. **Check `pstate`
and `clocks.sm`, never utilisation.**

> **ACTION: plug the laptop in before any GPU work.** Nothing below is worth
> starting until `AC online: 1`.

### B-2. π0.5 smoke test needs the GPU, and needs you logged in

**Yes — the π0.5 test is a GPU job.** π0.5 is ~3B parameters, ~6–7 GB in bf16.
It cannot be usefully tested on CPU.

Waiting on: you being logged in (your note), and B-1.

### B-3. GPU is shared with `solver`

Currently **free** (only a 86 MiB display process). `solver` is running a
three-seed reproducibility study and we have a standing agreement: **ping before
either of us starts heavy GPU work, in both directions.** They will stop between
seeds if we ask; each seed's results land on disk as it completes, so a boundary
stop costs them nothing.

### B-4. 8 GB VRAM contention is documented on this exact hardware class

[lerobot#3098](https://github.com/huggingface/lerobot/issues/3098) was filed on
an RTX 3070 Laptop (8 GB): the PyTorch CUDA context and MuJoCo's EGL rendering
context contend inside one process and evaluation can fail outright. π0.5 at
6–7 GB plus an EGL context is exactly that scenario. Fallback is
`MUJOCO_GL=osmesa` (CPU rendering, slower).
**This is the thing step 0 is testing.**

---

## Requirements checklist

| | Requirement | Status |
|---|---|---|
| Laptop on AC power | **NOT MET** — on battery | blocker B-1 |
| GPU free / coordinated with `solver` | free now; ping before starting | ok |
| `mujoco < 3.4.0` pinned | **MET** — 3.3.7 (F2) | ok |
| LeRobot env installed + EGL verified | **MET** — `.venvs/lerobot` | ok |
| `~/.libero/config.yaml` written | **MET** | ok |
| LIBERO demos downloaded | **MET** — pinned `a1aaacb7…` | ok |
| Harness Waves A/B/C | **MET** — 7/7 verified | ok |
| LIBERO env adapter | **NOT BUILT** | step 1 |
| VLA-Adapter policy adapter | **NOT BUILT** | step 1 |
| Surrogate / `adaptive_sweep()` (D10) | **NOT BUILT** — not needed until step 6 | deferred |

---

## The order

### Step 0 — ~~π0.5 smoke test~~ **DONE 2026-09-15: IT DOES NOT FIT (F4)**

**Decides DG-5**, which is still open. Does π0.5 load and run one LIBERO episode
on 8 GB alongside the EGL context?

- **Runs** ⇒ the anchor is real. Decide anchor-first on evidence.
- **Does not run** ⇒ **the anchor does not exist**, and `PLAN.md` §5.2 is
  promising a fallback that cannot be taken — it must be rewritten.

**Result: weights alone exceed 8 GB.** Not contention — the model. DG-5's
warning that *a contingency nobody has smoke-tested is not a contingency* was
correct.

**Replacement step 0: run the anchor on a 16 GB free-tier T4** (Colab/Kaggle),
inference-only, one-off. Verify `MUJOCO_GL=egl` works in a hosted notebook
first — it is a known pain point. Until then, step 1 proceeds with an
acknowledged ambiguity in its gate.

### Step 1 — LIBERO env adapter + VLA-Adapter policy adapter *(CPU to write)*

Two thin files against the `Env` / `Policy` protocols. Nothing else changes —
that is the point of the L0 contract.

Must implement:
- `identity()` — task suite, control mode, action space, init_states, step cap
- `semantic_deps()` — **`mujoco` and `MUJOCO_GL`** (keyed; F2)
- `scene_descriptor()` — object poses, camera extrinsics, lighting **in metres**.
  Harder than the toy: needs poses out of MuJoCo in world coordinates.
- `_gt_*` state for the detectors; everything else policy-visible

### Step 2 — Re-run the oracle gate against the LIBERO detector config *(CPU)*

**Non-negotiable** (`ARCHITECTURE.md` §8). `PhaseSegmenter` radii are tuned to
the toy and are meaningless in LIBERO. Detector thresholds are the part most
likely to silently break on a new env.

### Step 3 — Per-task environment control *(GPU, small)* — **C9, not yet built**

Confirm each task is solvable before attributing any failure to the policy.
**Per task, not per suite** — F2's lesson: at suite level, task 5's 80% → 28%
collapse dilutes into noise.

### Step 4 — `reproducibility_floor()` on the real policy *(GPU, small)*

Run one cell twice; compare. Everything downstream depends on knowing what zero
looks like. A large floor is itself a finding, and tells us how many seeds per
cell we actually need.

> Note `reproducibility_floor()` takes **no store, deliberately**. Threading one
> through would serve the second repeat from the first's cache and report
> `floor_pp = 0.0` by construction.

### Step 5 — The SCREEN *(GPU, ~1–2 h at full clocks)*

10 eps/task × 4 suites = 400 episodes. **Orienting only.** Label it so in the
output: at n=100 a 5 pp break passes 56% of the time, so a screen pass means
*"screen passed, gate not yet run"* — **never** "consistent with published"
(DG-1).

### Step 6 — Test the Phase 0 predictions *(GPU)*

Viewpoint and initial-state sweeps, **uniform arm only** — the surrogate is not
needed until we know roughly where a boundary is. P1–P4 either hold or they do
not; both outcomes are results, and they were committed in writing beforehand.

### Step 7 — Mining + manifest on real traces *(CPU)*

Phase localisation, classification, clustering, counterfactual probes with
paired statistics, `failure_conditional`. Remember: **we do not ship a cardinal
severity** — prevalence is client-supplied (DG-5b).

### Step 8 — Escalate to 50 eps/task *(GPU, ~5–8 h)*

Only where the screen says it matters, and **unconditionally for anything
reported**. This is the actual gate.

---

## What is most likely to get skipped, and should not be

**Steps 3 and 4.** Both are small, both are unglamorous, and both are the only
defence against reporting a harness artifact as a model finding. F2 — where a
MuJoCo point release would have produced a manifest row telling a client to buy
bowl-on-ramekin demonstrations — is what step 3 exists to catch.

## Open decisions that do not block

- Tier-3 LLM: local vs API (audit-trail reproducibility, not cost)
- Remediation data source for arm A: scripted demo vs re-rendering
- Do recovery rows ship in a first client manifest? The family is
  **unvalidated** — the toy fixture could not isolate it
- Readout audience: technical vs commercial
