# PIPELINE_CHECKS: code-check ledger

A companion to `RESULTS.md`. `RESULTS.md` records what experiments found; this file records **whether the code that produced them was fit to produce them**. Started 2026-10-09 at the user's request.

## The rule (user, 10-09: a necessary condition; scope finalised 10-09)

**What val is for.** `val` makes sure the **pipeline isn't wrong**. It doesn't judge whether results are interesting. That covers experiment code and also training recipes (e.g. LR schedule and recipe settings for a fine-tune), and it may reproduce a model's published or earlier numbers to check our pipeline.

**The gate is on code, not on results** (user, 10-09).
- Output of new or changed code **may be reported and discussed**, but it must carry the tag **UNVERIFIED (code not yet checked)** until its pipeline has a PASS here (or a PASS with only non-numeric notes).
- Three things still wait for a PASS: writing into `RESULTS.md`, committing as a run result, and justifying or launching a GPU step.
- A **FAIL** blocks downstream use and any launch that depends on it, immediately.
- Entries already in `RESULTS.md` from before 10-09 stand.

**The flow.**
1. The author (impl, my prim, PoC, res*, prim for its own pipelines) says "done" and hands over to `val` and prim: the command, inputs, output path, the plan section it implements, a proposed anchor for check 1, and `code_state/` in the outputs. Anything missing comes back as "not ready".
2. **Before checking**, `val` asks the code owner about open known issues and which behaviours are deliberate design choices. Its report says what it learned from the owner and what it checked for itself.
3. `val` runs the checklist. It may run authors' CPU scripts read-only, with its outputs only under `runs/checks/<PC-id>/`. For GPU pipelines it checks the CPU parts. The GPU controls it thinks a check needs (same-seed bit-identity, a nominal-vs-nominal null, a planted corruption) go to **prim or the user as proposals**, not to impl directly; once approved they run through the normal gate.
4. **Judgement:** `val` records each approve/reject with its reasoning, acts on it immediately (a FAIL blocks downstream dependents at once), then asks the user to approve the judgement. The user has precedence: only the user can override a block, and a block becomes permanent only with the user's approval.
5. `val` shows each entry (its judgement plus the ledger text) to the user **once**; one approval covers both. prim commits.

**Strictness.**
- Counts and labels must match exactly.
- Deterministic CPU floats must agree to relative 1e-6.
- GPU or bf16 re-runs must agree within a tolerance the author states **up front** from a same-seed repeat.
- A null test must be exactly zero for deterministic code, otherwise at or below the measured noise floor.
- "PASS with notes" may travel only if every note is non-numeric, and the notes go wherever the result is quoted.

**Pinning.** Each entry records the sha256 of every checked script and input. Any edit after a PASS invalidates it, and a delta re-check (PC-NNNa) covers what the edit touches.

**val's own limits.**
- **Standing approval for its own check scripts**, confined to `experiments/check_*.py` and `runs/checks/` (in the worktree for worktree pipelines), read-only on everyone else's code. Any other code or doc edit needs approval: it proposes, waits, then acts. It may create standalone HTML check reports in the workspace's format.
- **Upstream caveats:** each entry lists the known caveats of its input pipelines. No anchors come from runs flagged in the G0c follow-ups (g0c3: the env-deletion render corruption, R-039's camera pose and frame phase) until res21's audit clears them.
- val may **ask** code owners questions directly (known issues, design choices). **Change requests** go to prim or the user first, never straight to an author.
- NIM calls are allowed, mindful of rate limits and quotas.
- No commits, no subagents.

**Turnaround:** about 2 h for a CPU pipeline after a complete handover.

The purpose is to separate **pipeline faults from data effects**. If a pipeline passes its controls and the result is still surprising, the surprise is in the data. If it fails, it's the pipeline. (R-056 took days to settle this; D20's anchor reproduction finally did.)

**Scope:** any code that turns data into a number someone will act on: experiment runners, scorers, analysis scripts, harness changes. **Out of scope:** docs, plotting-only changes (unless the plot is the result), and one-off inspection snippets nobody reports from.

**Later:** after val's first 2–3 real entries, val drafts a `pipeline-check` skill (`.claude/skills/pipeline-check/`) with the checklist, recipes for null, planted and known-answer checks per pipeline type, the repo's trap list and helpers. It's built with approval; prim commits it. Authors can use it as a pre-flight; it doesn't replace val's independent check.

## The checklist (every pipeline, every time)

| # | Check | Rules out | How |
|---|---|---|---|
| 1 | **Known answer.** Reproduce an already-verified number through the new pipeline. | The pipeline computes something other than intended. | Pick the closest verified anchor (e.g. R-047 counts 720 / 544, toy root t = 31 for row 14, D20 anchor 4.514 / 3.979 / 4.120). |
| 2 | **Null test.** Identical inputs must give zero (or noise-floor) signal. | Spurious signal created by the pipeline. | Same input on both sides; same observation twice; a nominal-vs-nominal comparison. |
| 3 | **Planted effect.** Inject a known effect at a known place; the method must find it there. | A detector or score blind to what it claims to measure. | Corrupt the actions at step t in a success, swap a label, offset a pose. |
| 4 | **Invariants.** Rows in = rows out, unique join keys, units and normalisation round-trip, time alignment, array shapes. | Join, lag and normalisation bugs. | Never join on `rollout_id` (shared across noise seeds). Check demo `obs[t]` is after `step(t)`. Normalise → un-normalise equals the original. |
| 5 | **Independent recompute.** One headline number recomputed with the checker's own small script. | Logic errors in the main script. | A separate implementation, not a re-run of the same code. |
| 6 | **Provenance.** Code state, input hashes and config recorded with the outputs. | Results that can't be traced back. | `code_state/` (HEAD, status, patch) in the output folder. |
| 7 | **Matches the plan.** Pre-registration or scoping doc. | Quiet scope drift. | Compare definitions, windows, metrics and thresholds line by line. |

A check that genuinely doesn't apply is marked **N/A with a reason**; it is never skipped silently.

## Entry template

```
### PC-NNN: <pipeline name>, <PASS | FAIL | PASS with notes>
- Date / checker: YYYY-MM-DD HH:MM, <session>
- Author / commit or worktree path: <session>, <path @ commit>
- Feeds: <R-0xx / E-n / doc>
- Inputs: <paths, row counts, sha256>
- From the code owner: <open known issues, design choices>
- Upstream caveats: <known issues of input pipelines>
- 1 Known answer: <anchor → reproduced value> PASS/FAIL
- 2 Null: <what → value> PASS/FAIL
- 3 Planted: <what, where → found where> PASS/FAIL
- 4 Invariants: <list> PASS/FAIL
- 5 Recompute: <number: main vs independent> PASS/FAIL
- 6 Provenance: <present?> PASS/FAIL
- 7 Plan: <conforms / deviations> PASS/FAIL
- Verdict and plain-language summary (2–3 lines, readable without the code):
- Check scripts: experiments/check_*.py, outputs runs/checks/<id>/
```

## Queue (awaiting `val`)

| Pipeline | Author | Where | Feeds |
|---|---|---|---|
| R-059 part A (A1–A3 = E1) | my prim | worktree `VLA-exp-rootsignals` | R-059, ROOT_SIGNAL_EXPERIMENTS |
| E2 d_pn signal | my prim | worktree | ROOT_SIGNAL_EXPERIMENTS |
| E3 commanded vs achieved | my prim | worktree | ROOT_SIGNAL_EXPERIMENTS |
| E6 FAST action tokens | my prim | worktree | ROOT_SIGNAL_EXPERIMENTS, MINING_SPEC D16 |
| E4 open-loop on demos | impl | worktree | ROOT_SIGNAL_EXPERIMENTS |
| R-059 multi-draw runner (E1b) | impl | worktree | R-059 |
| E7 VLM closed questions + bisection | impl | worktree | ROOT_SIGNAL_EXPERIMENTS |

## Entries

_None yet._
