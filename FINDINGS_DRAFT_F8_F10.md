# DRAFT — not yet applied to FINDINGS.md

Prepared 2026-09-16 for user approval (`PENDING_DECISIONS.md` #20). vla-81
relayed that the owner wants F10 pinned. The F8 reframe it depends on was still
an open decision in this session, so both are drafted here rather than written.
They should land together: F10 contradicts F8 as currently worded.

All numbers were re-derived in this session. The confidence intervals match
vla-81's to 0.1 pp, and "ours" matches F9's 256 column exactly.

---

## F8 — AMENDMENT (prepend to the existing entry)

**Status change 2026-09-16: the comparison target was wrong.** The published
90/96/92/71 (87.3%) is for SmolVLA's **16-layer 0.45B headline build**.
`HuggingFaceVLA/smolvla_libero` ships `num_vlm_layers: 0`, and
`smolvlm_with_expert.py:102` truncates only when that value is >0. So the checkpoint
keeps **all 32 VLM layers**, and the expert is built to match (:112).
The architecture argument is vla-7f's: `num_vlm_layers: 0`, the >0
truncation guard, 32 VLM text layers **and** 32 expert layers present in the
weights, and expert depth fixed at build time, so the model was *trained* at
full depth. Separately, the parameter count (~605M; 787 tensors,
604,934,176 params) was parsed from the safetensors header by both vla-81 and
vla-7f. The architecture argument stands on its own if the count is ever disputed. The checkpoint carries no provenance: its
safetensors metadata is empty and the card says "datasets: unknown". A third,
independent strand (vla-7f): its normaliser stats are byte-identical to
`lerobot/libero` `meta/stats.json`, while the paper trained on
`physical-intelligence/libero`.

**There is no published LIBERO number for this checkpoint.** "61% vs 87%"
compared different architectures. Paired within-checkpoint results (F9's +17 pp)
are unaffected.

---

## F10 — The residual is concentrated where the scene alone does not determine the task

**Date:** 2026-09-16 · **Status:** HYPOTHESIS CLASS, supported by two
independent measurements · **Depends on:** the F8 amendment above

### Claim

The residual (~16–23 pp) is in **spatial, goal and long**, and **absent in
object** (the CI spans zero). Two candidate explanations fit that profile:
**geometric imprecision** and **language-conditioned disambiguation**.
Photometric causes are disfavoured (see the inverted correlation below).
Eef-state frame and normalisation are ruled out (per-task first-frame
comparison).

*Amended before landing.* The first draft claimed "geometric, not photometric".
It did not explain goal, which has the largest deficit. Goal holds ONE fixed
scene and varies only the instruction, so vision cannot identify the task (this
is why MINERVA needs a task-ID embedding there). Spatial tasks share object types
in different arrangements, so there too the disambiguating information is in the
instruction. Object is the one suite where the scene largely identifies the task.

### Evidence 1 — against the architecture-matched reference

Ours is render 256 with nas=10 (F9). The reference is the SmolVLA paper's
full-depth ablation row at nas=10. n=100 per suite on our side; n on the
reference side is **unverified**, assumed 100.

| suite | ref | ours | deficit | 95% CI | z |
|---|---|---|---|---|---|
| spatial | 89 | 73 | 16.0 | [5.4, 26.6] | 2.95 |
| object | 94 | 90 | 4.0 | **[−3.5, 11.5] n.s.** | 1.05 |
| goal | 91 | 68 | 23.0 | [12.3, 33.7] | 4.20 |
| long | 57 | 37 | 20.0 | [6.4, 33.6] | 2.89 |

Against the nas=1 row (89/94/85/53) the deficits are 16 / 4 n.s. / 17 / 16.
Object is not significant either way.

### Evidence 2 — F9's profile, measured independently, has the same shape

The 360→256 resolution fix gave **+17 pp on spatial** (p=0.011) and **+5 on
object** (n.s.). Resolution helps fine spatial discrimination and barely touches
identification.

### Evidence 3 — the language probe (weaker, n=5 per cell)

`runs/language_probe/probe.json`: libero_object tasks 0–2 are at
redirect_fraction **1.0** (the policy follows a substituted instruction).
libero_spatial tasks 0/1 are at **0.0** (it ignores it), and that is exactly the
suite where the instruction carries the disambiguating information. It supports
the language branch. It does not discriminate it from the geometric branch.

### Inference (not measured)

libero_object varies *which object* over a roughly fixed layout. Spatial varies
the *relation* among fixed objects, goal varies the *goal*, and long composes
them. We are down on the three suites where the scene does NOT determine the task.
We match on the one where it largely does. That fits both geometric precision
and language disambiguation.

### Candidates

**Language-conditioned disambiguation.** The policy fails to use the
instruction where the scene underdetermines the task.

**Geometric imprecision.** These degrade precision and leave recognition intact:
- The resize/padding path. The checkpoint declares `[3,256,256]` inputs and
  `resize_imgs_with_padding=[512,512]`.
- Control mode and delta-action scaling.
- Coordinate-frame or unit mismatches.
- ~~Normalisation of the 8-dim state vector~~ — **checked and ruled out**
  (vla-7f). The checkpoint's normaliser stats are identical to
  `lerobot/libero` `meta/stats.json`. Per-task first-frame `eef_pos` matches
  across all 377 dataset files within ~1 cm, with a uniform ~5 mm x offset.
  So there is no frame or unit bug in state. Note that the normaliser came
  from `lerobot/libero`, not the `physical-intelligence/libero` the paper names.

### Excluded — half the value of the finding

Lighting, textures, renderer version, anything photometric. This is
independently consistent with vla-7f's inverted correlation: object has by far
the largest dataset-vs-env pixel gap (34.3) and our **smallest** deficit, while
the three suites with negligible gaps — goal 1.6, spatial 2.5, long 0.7 —
carry deficits of 23, 16 and 20.

### Retires

**"libero_10 is catastrophically bad."** Against the matched reference it is
16–20 pp, in line with spatial and goal. The 36 pp figure came from comparing
against the 0.45B headline, and it has been driving hypothesis generation.

**This does not make the deficit uniform.** Object remains statistically
indistinguishable from matched, and that *differential* is what F10 rests on.
Do not read the retirement as "look for one global cause".

### What would falsify it, and how to narrow it

**Falsifier:** if a purely **photometric** intervention closes a material part
of the gap, F10 is wrong, because both surviving branches predict it will not.

**Discriminators within the class**, cheapest first, all CPU:
- **Control mode.** Confirm the runs used the action parameterisation the
  checkpoint was trained for (relative vs absolute). No config we have checked
  declares it, and a mismatch is silent — the conformance gate reports it as
  UNDECLARED.
- **Resize/padding path.** Compare the pixel geometry the policy receives
  against what the training dataset produced: 256→512 padded vs the dataset's
  path.
- **State normalisation.** Already checked and ruled out (see above).

**The discriminator between the two branches (vla-7f).** Within libero_goal,
per task:
- **Language failure** predicts **wrong-goal completions**: the policy achieves
  a *different* task's BDDL goal in the same scene.
- **Geometric failure** predicts **right-goal attempts that miss**.

The predictions are opposite, and they run on one corpus. Goal is the ideal
suite because its fixed scene means every task's goal predicate can be evaluated
against any episode's final state.

**Blocked** by the libero_goal instrumentation gap, and on exactly the wrong
bodies: the missing fixture/region bodies ARE the drawers, cabinets and stove
the goal predicates refer to. It needs the ~42 min goal re-run
(`PENDING_DECISIONS.md` #3). Before trusting those traces, verify
`_gt_object_pos_complete` is True for the fixture/region bodies. An indicative
version on the 39 currently-instrumented goal failures is possible on CPU, but
too thin to conclude from.

Each check either moves F10 from a class to a cause or eliminates a branch.

### Caveats that travel with this entry

1. **Every reference row is training-mismatched.** The ablation models were
   trained without robotics pretraining (paper §4.7). No like-for-like
   published number exists.
2. n for the ablation rows is unverified.
3. Our object 90 comes from an F9 arm whose +5 was not significant, so object's
   "matched" status rests on a soft number.
4. The suite-semantics step is inference.
5. The language-probe evidence below is n=5 seeds per cell, and goal was
   inconclusive.
