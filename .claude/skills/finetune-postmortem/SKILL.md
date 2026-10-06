---
name: finetune-postmortem
description: Critical post-mortem of a policy fine-tuning / retraining experiment whose results were negative, flat, surprising or disputed - find what actually went wrong (eval artefacts, data gaps and defects, too little signal, recipe drift, selection noise) from existing artifacts, separate data from hypothesis, and propose the cheapest discriminating checks. Use this whenever the user asks why a fine-tune, LoRA, adapter, retrain or behaviour-cloning run did not help, got worse, regressed, "hurt", or wants results "critically analysed", a "post-mortem", "what could have gone wrong", or a sanity check before scaling a training run - even if they never say post-mortem. Covers VLA / robot policies especially (rollout evals, minted or rescued demos, replay mixes).
---

# Fine-tune post-mortem

A negative fine-tuning result has many possible causes, and most of them are not "the idea is wrong":
- a broken or underpowered eval;
- one bad sub-group hiding inside a pooled number;
- training data that never covered the eval distribution;
- defects the data filter let through;
- a correction signal too diluted to learn;
- a recipe that drifted from how the base was trained;
- checkpoint selection by noise.

The job is to find out which of these hold, with numbers, before anyone reacts to the headline.

This file is the method. Two companions:
- `scripts/` holds tested analysis code: prefer running it over rewriting it.
- `references/vla-harness.md` holds the layout, gotchas and house rules of the VLA repo at `~/Documents/Code/VLA`. Read it first when working there, since it maps every phase below to a script and a path.

## Ground rules

- **Artifacts first, CPU first.** Everything below runs on existing logs, rollouts and datasets. A GPU re-run is a proposal at the end, not a step, and in pre-registered projects it needs registration and the owner's approval.
- **Don't edit the pipeline being judged.** Analysis code goes in a scratch or analysis folder.
- **Tag every claim.** Use one of three tags:
  - **data**: measured here, with n and the source path;
  - **hypothesis**: consistent with the data but not isolated;
  - **unknown**: these artifacts can't decide it.

  Readers act on the tags. A hypothesis written as a finding sends the next experiment the wrong way.
- **Verify rule-outs by running something.** Reading code and deciding it's fine is not enough. Examples of real checks: a masking bug shows up in the preprocessed batch; a join bug shows up as duplicate keys; a convention bug survives a like-for-like comparison.
- **Correct yourself in the open.** When a later check overturns an earlier number, fix the report and say so.

## Phases, in order

The order matters. Each phase can invalidate the ones after it: there is no point diagnosing the recipe of a result the eval can't support.

### 0. Frame the question
- Read the pre-registration or plan: expectations, "what would make this uninterpretable", deviations and amendments, which checkpoint was selected and how.
- Read the run summary and the headline numbers.
- List what the owner already suspects, and test those claims rather than inheriting them.
- Settle scope with the user when it is ambiguous: which runs, and which question (bug vs genuine negative, data, eval, recipe). Also settle where the report goes.

### 1. Can the eval be trusted?
- **Integrity.** Row counts per (arm, set) must match the design. Check that every join key is unique; IDs that ignore a seed will silently merge rollouts. Check that the scorer's inputs agree with the raw rollouts.
- **Uninterpretability checks.** Did any pre-registered check fire? Explain why before going further. Common causes: base not reproducing an earlier curve, a label measuring the requested value rather than the achieved one, a different direction or seed sample.
- **Noise floor.** Find the run-to-run flip rate of a *plain re-run*, e.g. from an earlier no-op arm or two base runs. Every later comparison is read against it.

### 2. Paired outcome analysis
- When base and arm ran on identical starts and seeds, use the paired 2×2 with an exact McNemar test, not pooled rates.
- Break down by the natural group (scene, task). For any group that moved, **rerun the test with that group left out**. A pooled "regression" is often one group.
- Any subgroup split (offset, radius, difficulty) must be printed with its **composition** (which scenes, directions and seeds fall on each side). Splits are usually confounded, and ceiling blocks (base at n/n) can only lose.
- State the power: with this discordance rate and n, what size of effect can the eval resolve?

### 3. Did the intended mechanism move?
- Measure the targeted change directly, not just outcomes. Examples: first-chunk actions vs the corrected chunk, a transfer or sensitivity metric, a representation probe.
- Report **direction and magnitude**, e.g. "moved the right way in 30 of 36 starts, closing 10% of the distance". Right direction with small magnitude means a dose problem. No movement means the signal never landed. The wrong direction is a sign or convention bug.

### 4. Localise the failures
- Find the phase of each lost episode: the first action chunk, approach, grasp, transport or place. Use privileged state (object heights, contacts) where the logs have it.
- Measure the mechanism at that phase. Examples: gripper height at close, number of regrasps, divergence from base at t = 16, 32 and 48.
- Check the nominal / unperturbed control per group. A regression there too is task-level damage, not a failure of the targeted perturbation.

### 5. Data distribution and coverage gaps
The question here is whether the training data covers what the eval asks, along the axes that physically matter.
- **Group coverage.** Demos per scene or task; distinct perturbation directions; seeds; env/layout variation. Compare with the eval. A zero overlap of directions is fine by design; a zero of env or layout variation is a generalisation gap.
- **Physical-axis coverage.** Bin training attempts, kept demos and eval starts on a physical axis (e.g. gripper offset in cm, not a requested-radius label). Two things to read:
  - the **yield per bin**: a success filter removes exactly the hard region;
  - bins with many eval starts but few kept demos: those are the gaps.
- **Nearest-demo distance.** For each eval start, find the distance to the closest kept demo of the same group, then look at success (base vs arms) by distance tercile. This separates "didn't learn" from "never saw anything like it".
- **Phase composition.** What fraction of training frames fall in each behaviour phase? Truncated demos often teach approach only, leaving grasp and transport entirely to the replay set.
- **Original-data spread.** How much variation along the target axis did the base's own training data have? That is the gap the fine-tune is meant to fill.
- **Marginals.** Compare state and action per-dimension statistics against replay, twice:
  1. against all replay frames;
  2. against replay frames from the **same phase or window**.

  A shift that survives the like-for-like comparison is a convention or scale suspect. A shift that disappears was composition.

### 6. Data defects the filter let through
- **Outcome filters keep bad segments.** "Successful" episodes can contain failed sub-behaviours, such as a failed first grasp followed by a regrasp. Truncation can then keep the error and cut the recovery.
- Count such episodes, check whether the defect falls inside the training window, and compare its signature with the failure signature from phase 4.
- Check label provenance per frame: which labels came from the correction source and which from the policy itself.

### 7. Dose accounting
- For each data category (correction frames, partly corrected, policy's own, replay), compute:
  - its share of draws;
  - the expected **views per frame** at each evaluated checkpoint.
- Under one view per correction frame at the selected checkpoint means the correction was barely trained on. Also report the share of gradient spent on labels the base already produces.

### 8. Recipe vs the base's own recipe
- Diff the fine-tune config against the checkpoint's original training config:
  - effective batch (under accumulation) at the same LR;
  - steps and warmup;
  - which modules are trainable;
  - augmentation, including state or observation dropout;
  - precision.
- Read the loss trace in blocks, not single lines:
  - where it plateaus;
  - how many steps ran after the plateau, and at what LR;
  - how often the gradient clip fired.

  Many post-plateau steps at a small batch make random-walk drift a plausible cause of scattered, task-level regressions.
- Note what widens the adapter's reach beyond the intended mapping, e.g. adapters on shared attention blocks, or augmentation turned off.

### 9. Checkpoint selection
- Compare the validation n and the per-checkpoint spread with the noise floor from phase 1.
- Was base run on validation? Was only the selected checkpoint evaluated on held-out? If so, checkpoint-vs-checkpoint claims are unknown.

### 10. Rule-outs, each with a real check
Typical checks: padded action steps masked in the loss at truncation boundaries; the LR schedule actually applied; task or instruction strings resolved per dataset part; gripper, image and state conventions; scorer joins. Record each one as ruled out, with the evidence.

### 11. Unknowns and the cheapest discriminating experiments
- List what the artifacts can't decide, and why.
- Name the **missing control**, often a replay-only arm or a base re-run at more seeds.
- Propose diagnostics ranked by cost. Examples: a fixed-seed, per-subset loss on saved checkpoints (minutes of GPU); targeted re-runs of the regressed group at more seeds.
- Then list recipe or data changes for the next arm, each tied to a finding.

## Report

Write a standalone report. In the VLA repo, it is a local HTML file in `docs/`; see the reference file. Structure:
1. **Bottom line.** 3–5 numbered findings, each tagged, with the numbers inline.
2. **The paired data.** The per-group table, with the regressed group highlighted.
3. **Mechanism and dose.** Did the fix move, and how much signal was there?
4. **Failure localisation and data** (coverage gaps, defects).
5. **Recipe and selection.**
6. **Eval design limits.**
7. **Ruled out.**
8. **Unknown.**
9. **What to change, in order.** Diagnostics first, then recipe and data changes.
10. **Sources.** Every path used.

Keep "what the data shows" visibly separate from "what might explain it". If another session commissioned the analysis, send it a compact summary: numbers, paths, and the data/hypothesis split.
