# Experiment procedure — phases, volume, GPU time

**Written 2026-09-15.** What we actually run, in what order, how much of it, and
what each phase costs.

**A–F are CAMPAIGN PHASES, not per-episode steps.** Each is a batch of many
episodes. One *episode* is: reset → ~150–520 policy steps → success or timeout.

---

## 0. The single most important planning fact

**Only rollouts captured through OUR harness can be mined later.**

| Captured by | Stores | Mineable afterwards? |
|---|---|---|
| `lerobot-eval` | success bit, reward, mp4 | **NO** — object poses, contacts, gripper state and phase signals were never recorded |
| our harness | full per-step state incl. `_gt_*`, `scene_descriptor`, fingerprint | **YES** — mining re-runs over stored traces without re-simulating |

This is a *designed* property: `run_cell` persists a rollout **before** anything
classifies it, so every load re-classifies from scratch. Change a detector
threshold, re-mine the same traces, get a different answer — no GPU needed.

**The consequence for planning:** deciding to use `lerobot-eval` for a phase is
deciding that phase can never be mined. It is not a reversible optimisation.

- Phase A (nominal baseline) via `lerobot-eval` — **accepted**: its job is a
  success number to compare against published, and the reproduction question is
  better answered by the standard tool.
- Phases B and D **must** go through our harness, or the mining layer has
  nothing to work on.

> **Open risk:** trace volume. Full per-step state for 10,030 episodes at ~300
> steps is large. Frames dominate and are optional (`image_dir=None` writes no
> images). Measure the per-episode JSONL size on the first real batch before
> committing to a full-corpus run.

---

## 1. The reframe: sampling, not searching

LIBERO-plus is a **fixed corpus of 10,030 pre-generated instances** (F5), not a
knob API. So this is **survey sampling from a known finite population**, not
optimisation over a continuous space.

That is a better position than it sounds: we know the sampling frame, we can
stratify by (category × difficulty), and per-stratum rates are unbiased by
construction.

**And the population is small enough to run whole:**

```
10,030 instances × ~9 s/episode  ≈  25 GPU-hours
```

at the measured 17 steps/s with `n_action_steps=10`. **The entire corpus, once
through, is about a long weekend.** That materially weakens the case for
adaptive sampling as a *discovery* mechanism — you cannot outsmart exhaustive
when exhaustive is affordable. Adaptive becomes a question of **where to spend
repeat seeds**, not where to look.

---

## 2. The phases

| Phase | What | Episodes | GPU time | Harness? |
|---|---|---|---|---|
| **A** Nominal baseline | vanilla LIBERO, 4 suites, 10/task | 400 | **1–2 h** | `lerobot-eval` |
| **B** Stratified screen | LIBERO-plus, uniform arm | 2,000–10,030 | **5–25 h** | **ours** |
| **C** Attribution | mostly free — see below | 0–500 | 0–1 h | ours |
| **D** Targeted repeats | seeds at measured boundaries | ~1,000 | **2.5 h** | **ours** |
| **E** Mining | phases, families, clusters, manifest | — | **0 (CPU)** | ours |
| **F** Remediation | 5 fine-tunes + 5 regression re-runs | ~1,500 | **10–15 h** | ours |

**Total ≈ 20–45 GPU-hours**, i.e. two to three overnight runs — not a month.

### A — Nominal baseline *(running now)*
The reference every perturbed rate is measured against, and the reproduction
question. **A screen, not a gate**: at n=100/suite a 5 pp break goes unflagged
66% of the time, so a pass means *"screen passed, gate not yet run"*.

### B — Stratified screen
Sample the corpus by (category × difficulty). This **is** the robustness
profile — with an ordinal x-axis, which E4 forces us to label as such rather
than plotting it like a dose-response curve.

**The uniform arm must stay uniform.** Every frequency claim in the manifest
rests on it; that is what `arms.jsonl` protects, and why the adaptive arm's
output can never feed a frequency estimate.

Start at ~2,000 (≈200/stratum, 5 h). Escalate toward the full corpus only if
stratum intervals are too wide to act on.

### C — Attribution, and it is mostly already done
**~6,750 of 10,030 instances are single-factor**, so *nominal vs instance* **is**
the intervention and needs no additional runs at all — B already collected it.
For the 3,279 multi-factor instances, look up a single-factor counterpart;
where none exists the row is **correlational** and must say so.

### D — Targeted repeats
Where B shows a boundary, spend seeds to tighten the interval. **Allocation, not
search.** This is the adaptive arm on a finite frame.

### E — Mining — CPU only, and re-runnable
Runs over stored traces. Costs no GPU and can be repeated as detectors improve,
which is exactly why §0 matters.

### F — Remediation
Three arms at **one** budget (targeted data / re-rendered augmentation /
frozen-policy adapter), plus a data-response curve on arm A for the top manifest
row. **5 fine-tunes, not 9** (DG-9). LoRA on a 1B policy fits 8 GB.

---

## 3. Is this brute force?

Partly, and deliberately.

**B is close to exhaustive by design** — the frame is finite and affordable, and
exhaustive sampling has a property no clever method has: the frequency estimate
is unbiased without argument.

**The intelligence is not in the search.** It is in C (interventional
attribution rather than correlation), E (why it failed, not just that it did),
and F (which remediation actually recovers success per unit cost). Those are the
parts nobody else does.

**Where genuine method is required is D** — but on a finite frame that is
allocation, and the honest framing is optimal-allocation-under-budget, not
search.

---

## 4. Sequencing

1. **A finishes** → per-suite rates vs published, with intervals
2. **LIBERO `PhaseSegmenter`** → unblocks mining on real traces *(cheapest
   high-value item; does not need LIBERO-plus)*
3. **Policy adapter through our loop** → lets B capture traces
4. **LIBERO-plus in a second container** → it uninstalls vanilla LIBERO
5. **B** → the robustness profile
6. **C, E** → attribution and mining, CPU-cheap, iterate freely
7. **D, F** → tighten and validate

Steps 2 and 3 gate everything; neither needs a GPU to write.
