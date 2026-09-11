# VLA Stress-Test — design review summary

**Date:** 2026-09-11 · **For:** design critique before further build
**Detail:** four review documents under `docs/reviews/`. This is the summary; the
reviews carry the evidence.

---

## Position

The engineering is disciplined — versioned schema, structurally enforced privileged-state
separation, verified resume, and a reproduction harness deliberately kept outside our own
code so that "the checkpoint doesn't reproduce" stays distinguishable from "our adapter is
buggy". The thesis is sound. Two things should change **before** more is built on top,
and neither is a code problem.

1. **The claimed contribution overlaps a published result.** It needs restating — narrower,
   and in a form that is still worth doing.
2. **The measurement apparatus certifies the wrong class of error.** It is built to catch
   noise, and the dominant error in VLA evaluation is not noise.

A third item is a consequence of the no-budget constraint and needs a decision.

---

## 1. The contribution needs restating

**Finding.** LIBERO-Plus (arXiv:2510.13626) §6.2 already ran the loop this project
proposes. They built 20,000+ remediation trajectories, mixed-fine-tuned OpenVLA-OFT, and
re-measured: camera-viewpoint robustness **55.6% → 92.8%** (+37.2 pp), overall 79.6%.

So the proposal's §5 question 5 — *"did that data close the gap when the policy was
retrained?"* — has a published answer of **yes**, on our headline axis. Phase 5 as written
reproduces it with a smaller model.

**What is still open, and is the better claim.** They closed the gap by collecting
*broadly* — 20k trajectories across all seven factors. That is the brute-force answer.
Nobody has shown that **targeted collection beats broad collection at equal budget**. That
is precisely what the mining layer is for, it is the commercial proposition ("we tell you
*which* data"), and it is currently untested.

**What it costs to fix: one control arm.**

| Arm | Data | Budget |
|---|---|---|
| A | targeted at the top manifest row | N |
| B | uniformly sampled across factors | N |
| C | no fine-tune | — |

A vs B at matched N. If targeted wins, the manifest has demonstrated value. If it ties,
the honest finding is that broad collection suffices and diagnosis is a cost saving — worth
knowing, reportable, and far better than discovering it after the fact.

**A second, larger question underneath.** LIBERO-PRO (arXiv:2510.03827) finds that models
scoring above 90% on standard LIBERO **collapse to 0.0%** under fair perturbation, that
success goes to zero once object displacement exceeds 0.2 units, and that policies "reach
toward memorized locations" and "ignore meaningless instructions". Reported accuracies
"largely reflect rote memorization of fixed mappings from the training set".

If the baseline has memorised rather than generalised, then in the failing region there is
**no generalisation mechanism to feed**, and more demonstrations teach more memorisation.
That is `PLAN.md` §7b.1's `architecture` class — and it may be the **modal** answer on
LIBERO rather than the rare one. This inverts the pitch from "here is the data to buy" to
"data will not fix this one". Not a problem to hide: §7b.1 already has the vocabulary, and
a vendor who can say *"do not buy data for this"* is more credible, not less. But it
should be a decision, not a week-7 surprise.

**Ask:** endorse or reject the reframing to targeting-efficiency, and confirm that
"data will not fix this" is an acceptable finding to deliver.

---

## 2. The measurement apparatus certifies the wrong error

Every success rate we report varies from three independent sources. They need completely
different treatment and the plan currently addresses one.

| | Source | Size | Shrinks with | Caught by |
|---|---|---|---|---|
| **S1** | Sampling error (finite episodes) | ±20 pp at n=20 | more episodes | Wilson CI — already computed |
| **S2** | Execution nondeterminism (GPU, sampling heads, contact chaos) | **~1 pp** | repeats | `reproducibility_floor()` — built, **never called** |
| **S3** | **Configuration bias** (control mode, action-chunk setting, library version, power profile) | **10–20 pp** | **nothing** | external reference only |

Sizes are measured, not assumed: S2 from published LIBERO seed spreads (OpenVLA reports
88.4 **± 0.8** over 3 seeds × 500 rollouts); S3 from openvla#282, where a reporter got
**68%** against a published 88.4%.

**The decisive detail.** In openvla#282 the reporter changed the random seed from 3 to 15
and the number **stayed at 68%**. Redownloading data, raising step caps and disabling
flash-attention moved it by about a point in total.

That is a bias, not variance. It is *perfectly reproducible and perfectly wrong*. Repeating
a run cannot detect it, because repeating holds constant the thing that is broken — so
`reproducibility_floor()` will report a reassuringly small number in exactly the situation
where the result is least trustworthy.

**What is needed — three tests, not one:**

- **A/A repeat** — identical config run twice. Gives the noise floor (S2). Built, unwired;
  also `repeats=2` with `max − min` is the range of two samples, which is biased low, and
  low is the dangerous direction for a safety threshold.
- **Golden reference cell** — one frozen cell re-run after *any* environment change. This
  is the only instrument that sees S3, because it compares against an external fixed point
  rather than a repeat of itself. Not built.
- **Published-number gate** — the only test that catches a configuration wrong *from the
  beginning*, where there is no "before". Item 3 below.

Detail, with computed sample sizes, in `docs/reviews/2026-09-11-plan-vs-implementation.md` §2.

---

## 3. The reproduction gate is at risk, and there is now no paid fallback

`PLAN.md` §1 calls the reproduction gate "the project's only defence against *your
environment was broken, not the model*".

**It is likely to fail.** SmolVLA publishes ~87.3% average on LIBERO. Two independent,
open, unanswered reproduction reports exist — lerobot#3264 (same checkpoint, eval only)
at **73.25%**, and #3287 at ~67%. Against a ±5 pp gate that is a 14 pp gap. On present
evidence failure is the expected outcome, not a coin flip.

With no GPU budget, the previously obvious fallback — rent an A100 for ~$10 and use
OpenVLA-OFT as the gated reference — is unavailable. So the gate needs a different defence.

**The useful move is to split one question into two, because only one of them is essential:**

| Question | Essential? | Answerable free? |
|---|---|---|
| **Is our harness correct?** | **yes** — everything downstream depends on it | **yes** |
| Are the published numbers reproducible? | no — nice to have | possibly not |

The first is answerable by **differential testing**: run `lerobot-eval` as shipped and our
`vla_harness` adapter on the identical checkpoint, suite and seeds, and require them to
agree. Agreement validates our adapter *regardless of whether either matches the published
number*. `experiments/repro/run_repro.sh` is already built for exactly this — the comment
says so — and it costs nothing.

**And if the published number does not reproduce, that is a result, not a failure.** Three
independent non-reproductions of a headline VLA benchmark — two public, one ours, with
controlled provenance — is a legitimate finding, and a directly commercial one for an
evaluation-services proposition: it is the argument for why systematic evaluation
infrastructure is needed at all. It should be written up as a deliverable rather than
absorbed as a setback.

**Ask:** confirm that the no-budget constraint is firm, and that "published numbers do not
reproduce" is acceptable as a reportable finding.

---

## 4. Two smaller items that change plans

**Compute strategy inverts.** `MODELS_AND_COMPUTE.md` §6 designates free-tier notebooks as
batch runners on the strength of 16 GB VRAM. But `PLAN.md` §8 already established that the
bottleneck is **rendering** — CPU-bound and parallel across cores. Measured: this laptop
has **16 cores**; Kaggle gives 4, Colab free gives 2. For a render-bound campaign the
laptop is **4–8× the throughput** of the designated runners, and SmolVLA needs only 1–2 GB
so the extra VRAM buys nothing. Free tiers are for the GR00T 3B path where VRAM *is*
binding. This also lifts session caps and notebook-EGL problems off the critical path.

Related and unpropagated: the repro scaffold measured a **25× wall-clock swing** from GPU
power capping (601.7 s vs 24.0 s on identical work), invisible in `nvidia-smi`'s
utilization figure. §8's "the full campaign is an overnight run" assumes a fixed
throughput. Power state belongs in the fingerprint.

**The oracle gate cannot currently fail.** It reports 3/4 and passes at 3 — but one fixture
produces zero failures at every level, so the maximum achievable score *is* 3. Two of the
three passing attributions are separated from all alternatives by +95 pp, roughly 6σ, while
real effects will land in the 10–30 pp band. And it measures recall only: on a policy with
**one** planted fault the pipeline emits a **two-row manifest whose second row is a false
positive**, recommending contact-and-alignment demonstrations for a camera-calibration
problem. The cause is that the family label routes on error magnitude, so one fault is
labelled `manipulation` at yaw 6° and `visual_grounding` at yaw 9°+ — wrong precisely at
the boundary, which is what the manifest is built around.

Fixes are small: a held-out fixture, a precision criterion, a weak-effect fixture, and a
direction-based rather than magnitude-based grounding signal.

---

## Recommended sequence

| | Action | Blocks |
|---|---|---|
| 1 | Finish the SmolVLA reproduction run; report against per-suite targets with CIs | the model decision |
| 2 | Differential-test `vla_harness` against `lerobot-eval` on the same seeds | everything downstream |
| 3 | Decide the contribution framing (§1) | what Phase 5 measures |
| 4 | Wire the A/A floor; build the golden reference cell; power state into the fingerprint | any cross-run comparison |
| 5 | Make the oracle gate able to fail; add the precision criterion | the Phase-1 claim |
| 6 | Phase 0 supply-side, re-scoped | manifest corroboration |

Items 2, 4 and 5 are days of work and all three gate expensive things.

---

## Open questions for reviewers

1. Is targeting-efficiency (A vs B at matched budget) an acceptable restatement of the
   contribution, or should the project aim elsewhere?
2. Is "data will not fix this — it is an architecture limit" an acceptable manifest
   finding for a client?
3. Is "published VLA numbers do not reproduce" an acceptable deliverable?
4. Is the no-GPU-budget constraint firm? It removes the only fallback if the gate fails.
5. Who owns licence clearance across four chains (LIBERO, LIBERO-plus assets, checkpoints,
   generated data)? It is a precondition on anything client-facing.

---

## Reference

| Document | Covers |
|---|---|
| `reviews/2026-09-11-architecture-flow.md` | code defects in the rollout→manifest path (14) |
| `reviews/2026-09-11-methodology-and-claims.md` | thesis, gate validity, attribution, novelty (18) |
| `reviews/2026-09-11-models-and-compute.md` | hardware, model choice, reproduction gate (10) |
| `reviews/2026-09-11-plan-vs-implementation.md` | plan vs built; experimental control design (8 + §2) |
