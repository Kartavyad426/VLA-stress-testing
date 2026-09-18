# Handoff — 2026-09-18, 15:20 IST

For whoever picks this up next, agent or human. `RESULTS.md` is the experiment
record and `PENDING_DECISIONS.md` holds what needs the user's call; this file is
the state of the machine right now and the one job waiting.

---

## 1. The job waiting: adjudicate the failure taxonomy

**436 real failures are on disk.** The family rules in
`vla_harness/mining/classify.py` were written against a toy environment and have
never been checked against real VLA failures. The user's instruction
(2026-09-18): *"we'll see if our rules for taxonomies are accurate or maybe need
more tightness, I'll maybe even adjudicate or eyeball some of the failure
scenarios, then maybe we can generalise that insight into a skill"*.

**Start here:** `viz/lplus_fail_groot/index.html` — every failure grouped by
perturbation type, each row showing the classifier's family, the predicates that
fired, the terminal state, and a link to the episode video (both cameras).
Record verdicts in `runs/lplus_fail_groot/adjudication.csv` (pre-filled with
rollout ids and the classifier's label; add `verdict` and `note`). A CSV, not
browser state, so it survives a re-render.

**What the data already says the problem is.** GR00T's 155 failures collapse
into four families, `manipulation` in every one, `any_attempt` firing on 100% by
construction — effectively a two-bit code. And 78 of those failures are
robot-initial-state variants filed under **visual_grounding**, a label the
evidence contradicts: nothing about the scene's appearance changed. There is no
family that names *starting pose*. That is PENDING #15 with data behind it.

MINERVA's 281 failures are richer — seven families, including `never_reached`
(16%) and `ambiguous`, which GR00T never triggered. Adjudicate both: the
disagreement between them is the signal.

| Corpus | Failures | Rendered pages |
|---|---|---|
| `runs/lplus_fail_groot` | 155 of 623 | 156 in `viz/lplus_fail_groot/` |
| `runs/lplus_fail_minerva` | 281 of 506 | 25 in `viz/lplus_fail_minerva/` (more rendering) |

---

## 2. What is running right now

Serialised on a `flock` mutex at `/tmp/vla_gpu.lock` (see
`experiments/gpu_lock.sh` for why — a `pgrep` race cost 25 minutes today).

| Job | Script | State |
|---|---|---|
| Unperturbed control | `baseline_same_stack.sh` | RUNNING, 65/100 rollouts, **65/65 so far** |
| Precision A/B | `precision_ab.sh` | queued behind the lock |
| π0 VRAM probes + smoke | `setup_pi0.sh` | queued behind the lock |

Everything is resumable: re-running a script continues from its `cells.jsonl`.

**Two process hazards, both now guarded, both of which wasted time today:**
- Killing a wrapper script **orphans its python child**, which keeps the GPU.
  Kill the eval too (`pkill -f harness_eval`), or the next job measures
  contention. This invalidated the first π0.5 VRAM probe.
- Holding the lock is **not** the same as the card being free: CUDA memory is
  released asynchronously and EGL render contexts linger. MINERVA (0.54 M
  parameters) hit `CUDA error: out of memory` seconds after a render job exited.
  The GPU jobs now wait for `<400 MiB` used before starting.

---

## 3. Results in, since the last handoff

Full entries in `RESULTS.md`; the short version:

- **R-026 — GR00T on 623 L5+L4 variants: 468/623.** Robot initial state is the
  dominant failure mode at **32.8%** (n=116), camera viewpoint second at 64.5%,
  and every appearance perturbation ≥88%. **GR00T is robust to how the scene
  looks and fragile to where the arm starts.**
- **R-027 — MINERVA on the same 506 non-language variants: 225/506.** It
  collapses where GR00T does not (background 0/12, camera 1/33) but matches
  GR00T almost exactly on robot initial state (~31% vs ~33%). Suggestive: visual
  robustness is what the 3 B stack buys; the starting-pose failure may be
  something neither model's capacity fixes.
- **R-025 — contamination A/B: no detectable effect.** 34 vs 33 of 42, the 7
  disagreements running in both directions, exact McNemar p=1.0. Clears the bug
  for this policy at this n, not for smaller language encoders.
- **R-028 — the clean-instruction stripper truncated 10% of variants** and is
  fixed and validated against all 8,493 non-language variants. R-023 and R-024
  ran with the bug; their numbers are provisional.

---

## 4. Policy roster

| # | Policy | State |
|---|---|---|
| 1 | GR00T N1.7 | priority, in use, bf16 (fp32 does not fit) |
| 2 | MINERVA | control, in use, fp32, no language encoder |
| 3 | SmolVLA | **vetoed** 2026-09-18 — never reproduced its published score |
| 4 | **π0.5** `lerobot/pi05_libero_finetuned_v044` | downloaded, gate PASS, **VRAM unresolved** |
| 5 | **π0-FAST** `lerobot/pi0fast-libero-v044` | downloaded, gate PASS, fits (5.44 GiB) |

Both π0 checkpoints are drop-ins: they ship their own rename map and 8-D
normalisation statistics, so neither needs a `--rename-map` or a state shim.
π0.5's only question is VRAM — 6.74 GiB of weights against 7.53 usable. **The
first probe's OOM does not count**: an orphaned eval held the card at the time.
The re-probe is queued and refuses to run unless the GPU is idle.

Rationale for both choices is `MODELS_AND_COMPUTE.md` §R9; the open questions
(probe π0.5? rent a GPU for OpenVLA-OFT?) are PENDING #25.

---

## 5. Harness changes today

- **Instruction stripper** anchored at the end and validated against the 40
  vanilla instructions (R-028 / O7).
- **`--base-instruction`**: forces the base-scene text even on a language
  variant. With a canonical-camera, initstate-0 variant this is the only
  unperturbed control obtainable on the LIBERO-Plus stack — the fork replaces
  the libero package, so there is no vanilla suite in that venv at all.
- **Conformance gate** now reads the checkpoint's shipped preprocessor (rename
  map + normaliser statistics) instead of trusting `config.json`, which for the
  pi0 family describes something the pipeline never sees. It was REFUSING a
  drop-in checkpoint.
- **`forward_passes` → `env_steps`, plus a real `model_forwards`** (reported by
  vla-81). The old field counted environment steps; a chunked policy runs the
  model once per `n_action_steps`. `Rollout.meta` now records the **resolved**
  policy config, since requests and defaults diverge silently.
- **`experiments/mine_run.py`** mines any run offline; **`adjudication_index.py`**
  builds the review page.

---

## 6. Open, needs the user

`PENDING_DECISIONS.md`: #25 (fourth policy / rented GPU), #20 (apply F8+F10 to
FINDINGS.md), #15 (taxonomy redesign — tomorrow's work bears on this), #13, #6,
#2, #21, #8. R-005's perturbed arms still need re-running.

**Not blocking:** the laptop is on AC now (it was on battery overnight).
