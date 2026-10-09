# PIPELINE_CHECKS: gate ledger for new code

A companion to `RESULTS.md`. `RESULTS.md` records what experiments found; this file records **whether the code that produced them was fit to produce them**. Started 2026-10-09 at the user's request.

## The rule (user, 10-09: a necessary condition)

No output of **new or changed experiment / analysis / pipeline code** may be:

- reported to prim or the user as a result,
- interpreted, or written into `RESULTS.md`,
- committed as a run result, or
- used to justify or launch a further GPU step,

until that pipeline has a **PASS** entry in this file.

The flow: the author (impl, my prim, PoC, res*, …) says "done" and notifies the **checker session** and prim. The checker runs the checklist below and appends an entry here. Only then does the result travel. A **FAIL** goes back to the author with the failing check named, and nobody interprets the data until a later entry passes.

The purpose is to separate **pipeline faults from data effects**. If a pipeline passes its controls and the result is still surprising, the surprise is in the data. If it fails, it's the pipeline. (R-056 took days to settle this; D20's anchor reproduction finally did.)

**Scope:** any code that turns data into a number someone will act on: experiment runners, scorers, analysis scripts, harness changes. **Out of scope:** docs, plotting-only changes (unless the plot is the result), and one-off inspection snippets nobody reports from.

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
- Inputs: <paths, row counts, hashes>
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

## Queue (awaiting `val`; scope being finalised 10-09)

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
