# Handoff — failure-hunt night, 2026-09-18

*Written 2026-09-18 00:40 IST, while the queue is running.* For whoever (agent
or human) picks this up next. Read `RESULTS.md` for the experiment record and
`PENDING_DECISIONS.md` for what is still open; this file is only about the state
of the night and the one job waiting in the morning.

## The job tomorrow

**Adjudicate the failure taxonomy by hand.** The family rules in
`vla_harness/mining/classify.py` were written against a toy environment and have
never been checked against a large set of real VLA failures. Tomorrow's task,
in the user's words: *"we'll see if our rules for taxonomies are accurate or
maybe need more tightness, I'll maybe even adjudicate or eyeball some of the
failure scenarios, then maybe we can generalise that insight into a skill"*.

So this night is deliberately **not** a balanced design. It is a failure
harvest: LIBERO-Plus difficulty levels 5 then 4 (the variants four reference
models mostly failed), all seven perturbation types, interleaved by type so any
prefix of the run is type-balanced.

**Morning sequence:**

1. `cat experiments/repro/runs/overnight_20260918/progress.log` — what ran.
2. `cat experiments/repro/runs/overnight_20260918/mining.txt` — family counts
   and, more importantly, the **predicate fire rate over failures**. A predicate
   that fires on ~100% of failures (`any_attempt` does, by construction) carries
   no information; that is the tightness problem to fix.
3. `viz/lplus_fail_groot/*.html` — up to 80 rendered failure episodes, both
   cameras, for eyeballing. Each page states its replay fidelity in mm; anything
   above ~1 mm means the replay diverged and the video is not what the policy saw.
4. Compare the hand verdict with `runs/<run>/diagnoses.jsonl` (one row per
   rollout: family, families, predicates, terminal, level, category).

## What is running

`experiments/overnight_20260918.sh`, launched 00:36, under `systemd-inhibit`.
Resumable — re-running it continues; every step is capped by a wall-clock
deadline so the morning always has mined and summarised output.

| Step | Run id | What |
|---|---|---|
| contamination A/B (R-025) | `lplus_hard_groot_rawinstr` | 41 hard variants with LeRobot's contaminated instruction |
| GR00T failure hunt (R-026) | `lplus_fail_groot` | 623 L5+L4 variants, until 05:40 |
| MINERVA (R-027) | `lplus_fail_minerva` | the same variants minus language, until 07:50 |
| mine + summarise + render | — | `mining.txt`, `summary.txt`, `viz/lplus_fail_groot/` |

## Two things found tonight that change earlier results

**1. Instruction truncation (harness bug, fixed).**
`_clean_instruction()` stripped LIBERO-Plus perturbation suffixes by splitting on
the *first* occurrence of a marker token. Scenes whose own name contains a marker
word were truncated: `pick_up_the_black_bowl_from_table_center_..._table_11`
became **"pick up the black bowl from"**. That hit **240 of 2,402 libero_spatial
variants (10%)**, and the equivalent in the other suites.

Fixed by anchoring the suffix grammar at the end and stripping repeatedly, plus
dropping the `KITCHEN_SCENE3_`-style prefix that libero_10 names carry but the
training datasets do not. **Validated**: all 8,493 non-language variants across
all four suites now strip to a string that is exactly one of the 40 vanilla
LIBERO instructions (checked against the task table shipped in MINERVA's
checkpoint, which was recorded from the training datasets).

*Consequence:* R-023 and R-024 ran with the old stripper, so a minority of their
variants got a truncated instruction. R-024's numbers are the ones quoted in
`RESULTS.md`; they need re-running before they are cited outside this repo.
`docs/libero_plus_variants/*.csv` also still carries the truncated
`instruction_clean` column and should be regenerated.

**2. MINERVA cannot be given LIBERO-Plus language variants.**
It is not language-conditioned at all: `TinyflowTaskToIndexStep` looks the
instruction up in a fixed 40-entry table and raises `KeyError` on anything else.
Its arm therefore runs the 506 non-language variants. This is a feature for the
comparison — MINERVA is a control for how much of GR00T's perturbation
sensitivity is visual rather than linguistic — but it is a hard limit, not a
choice.

## Environment notes

- New venv `.venvs/minerva-lplus` (py3.12): MINERVA's lerobot fork + the
  LIBERO-Plus fork, MuJoCo 3.3.2, robosuite 1.4.1. Built with
  `--no-deps` for robosuite because its `pynput`→`evdev` chain needs Python
  headers that are not installed; the working `.venvs/libero-plus` has the same
  shape, so this is consistent, not a shortcut.
- LIBERO-Plus needs `PYTHONPATH=third_party/LIBERO-plus` and
  `LIBERO_CONFIG_PATH=third_party/libero-plus-config`.
- **The laptop is on battery (`ac=0`).** Every log line records it. If the run
  ends early with no error, check this first.
- Two upstream LIBERO-Plus patches are local and unpushed:
  `experiments/repro/patches/libero_plus_numpy2_fog.patch`.

## Uncommitted

The instruction-truncation fix in `vla_harness/envs/libero_env.py`, the new
`experiments/mine_run.py`, the rewritten queue, and the L4/L5 selection files
are uncommitted. Commit them in layers (harness fix / tooling / experiment
config) once the night's runs confirm the fix behaves in the simulator, not only
in the offline validation above.
