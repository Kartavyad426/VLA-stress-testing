# Review — `MODELS_AND_COMPUTE.md`

**Date:** 2026-09-11 · **Scope:** the hardware argument, the SmolVLA bet, the
reproduction gate, and the free-tier compute strategy.
**Method:** read + measured on this machine + checked against primary sources.
*[measured]* = run on the laptop; *[verified]* = checked against paper/issue/docs;
*[by inspection]* = read, not checked.

**Status key:** `open` · `discussing` · `accepted` · `wontfix` · `fixed`

---

## Verdict

The document is the most disciplined artifact in the workspace — it separates
verified fact from assumption, dates itself, tells you to re-verify, and §5's
rejection of hosted APIs is correct and correctly argued. Three things change the
decision it supports:

- **C1** — the reproduction gate has no target number written down, and when you
  write it down the gate is *expected to fail*, not 50/50.
- **C2** — the free-tier strategy optimises VRAM, which is not the binding
  constraint. On a render-bound campaign this laptop is 4–8× the throughput of the
  proposed batch runners.
- **C5** — published LIBERO numbers may not measure what the gate assumes they
  measure, which reaches past this document into `PLAN.md` §7b.1.

Taken together they argue for a **hybrid** the document dismisses in one line: rent
OpenVLA-OFT for the gated reference, keep SmolVLA local for the loop. See C10.

---

## Summary table

| # | Finding | Class | Severity | Status |
|---|---|---|---|---|
| C1 | The reproduction gate has no target number — and the real one fails it by 9 pp | **gate** | **high** | open |
| C2 | The free-tier strategy optimises the resource that is not the bottleneck | **compute** | **high** | open |
| C3 | The VRAM budget is 7.3 GiB, not 8, and has no activation accounting | compute | medium | open |
| C4 | `control_mode` correctly flagged as a trap, then left unresolved | config | **high** | open |
| C5 | Published LIBERO scores may reflect memorisation — this inverts §7b.1 | **inference** | **high** | open |
| C6 | The Octo row is unverified but sits in a verified table | rigour | low | open |
| C7 | Free-tier T4 is Turing — no bf16, no FlashAttention-2 for the GR00T path | compute | medium | open |
| C8 | EGL risk is flagged for Kaggle and not for the new Blackwell laptop | risk | medium | open |
| C9 | Licences are a gate for a client deliverable, not a footnote | commercial | medium | open |
| C10 | The rented-GPU estimate is ~5–10× conservative, which changes the §1 decision | **strategy** | **high** | open |

**Not at fault:**

- **§5, hosted APIs.** The argument is structural rather than about price — a VLA is
  a closed loop with a simulator, step *t+1* depends on step *t*, so calls are
  serial and unbatchable. 38 chunked forward passes against 40 req/min is ~1
  episode/minute, ~50 h for 3,000 episodes. The arithmetic checks out and the
  conclusion is right. Correctly marked settled.
- **The verified/assumed discipline.** §3's "this was an assumption in `PLAN.md`
  §7.1; it is now verified fact" and §7's was/now table are exactly right, and rarer
  than they should be.
- **The dual-environment finding.** LIBERO-plus uninstalling vanilla LIBERO is the
  kind of thing that costs a day when discovered late.
- **Naming the reproducibility risk at all**, and replacing the resolved risk with it.

---

## C1. The reproduction gate has no target number — and the real one fails it by 9 pp — *[verified]*

§4 instructs: "run the suites, compare against published, and treat the outcome as a
finding either way." The document never states **what the published number is.** A
gate without a threshold cannot be run.

Writing it down changes the picture materially. SmolVLA publishes an average LIBERO
success rate of **87.3%** (0.45B, against Octo 75.1%, OpenVLA 76.5%, π0 86.0%).
lerobot#3264 reports **73.25%** overall — Object 93.0 / Goal 81.0 / Spatial 63.0 /
Long 56.0.

That is a **14 pp gap** against a gate specified at ±5 pp. The gate does not merely
have a chance of failing; on the only public evidence available it **fails by 9 pp
beyond tolerance**, and the shortfall is concentrated in Spatial (63.0) and Long
(56.0) rather than spread evenly — which looks like a systematic problem with
specific suites, not noise.

§4 frames three readings as if equally likely (reporter's environment / genuine
regression / published numbers unreproducible). They are not equally likely. Reading
1 remains possible, but the plan should treat **failure as the default branch** and
price it now.

**Direction.** Write the target numbers into §4 as a table, per suite, with the
source. Then state the decision rule *before* measuring: what result continues with
SmolVLA, what result triggers the fallback, and what the fallback is. Right now the
fallback is "consider Octo or a rented OpenVLA-OFT run" — which is C10, and which
deserves to be a costed plan rather than an aside.

---

## C2. The free-tier strategy optimises the resource that is not the bottleneck — *[measured]*

§6 builds to: "**The important asymmetry: free-tier T4s have 16 GB — double the
laptop.** The laptop stays the dev box; free tiers become the batch runners."

The asymmetry is real and it is not the binding constraint. `PLAN.md` §8 already
established what is: "**Rendering is the bottleneck, it is CPU-bound, and it
parallelizes across cores** — so a multi-process runner is worth building early."

Measured on this machine and checked against the platforms:

| | Cores | VRAM | Render throughput |
|---|---|---|---|
| **This laptop** | **16** (Intel Ultra 7 265H) | 7.3 GiB usable | baseline |
| Kaggle T4 ×2 | 4 | 16 GB | **~¼** |
| Colab free | 2 vCPU | 16 GB | **~⅛** |

For a render-bound campaign the laptop is **4–8× the throughput** of the machines
§6 designates as batch runners. SmolVLA needs 1–2 GB, so the extra VRAM buys nothing
on the primary path — the free tiers are GPU-rich and CPU-poor, which is precisely
the wrong shape for this workload.

**This inverts the conclusion, not the facts.** §6's three consequences (resumability,
trace persistence, EGL verification) all stand. What changes is the role:

**Direction.** Free tiers are for the path where VRAM *is* binding — GR00T N1.7 3B
inference as the second policy (§2 already puts it there) and any LoRA that outgrows
7.3 GiB. The overnight campaign belongs on the laptop with a multi-process runner
across 16 cores. That also removes the session-cap and EGL-in-notebooks risks from
the critical path, where they currently sit for no gain.

---

## C3. The VRAM budget is 7.3 GiB, not 8, and has no activation accounting — *[measured]*

```
memory.total 8151 MiB   memory.used 271 MiB   memory.free 7437 MiB
```

8151 MiB total = 7.96 GiB, and with the desktop session running the usable figure is
**~7.3 GiB**. The whole of §1 is a budget against "8 GB", and §1 is what justifies
the headline-model decision, so the number should be the measured one.

Separately, §1's `LoRA on 8 GB?` column is asserted without a breakdown. Weights are
the small term in fine-tuning; activations and optimizer state are what OOM. SmolVLA
is SigLIP-400M vision + SmolLM2-135M with action chunking — the activation cost
depends on image resolution, chunk size and batch size, none of which appear. The
claim is probably right and is currently unfalsifiable.

**Direction.** One measured row: batch size, resolution, chunk, peak allocated. It is
an afternoon and it converts the central premise of `PLAN.md` §1 from argument into
fact.

---

## C4. `control_mode` is correctly flagged as a trap, then left unresolved — *[by inspection]*

§3 is emphatic and right:

> **`--env.control_mode`** is `relative` or `absolute`, and **must match the
> checkpoint's action parameterisation.** Wrong mode ⇒ confident, plausible,
> completely wrong motion.

The document then never says which mode `HuggingFaceVLA/smolvla_libero` requires.

This is the highest-value unknown in the document, and it connects directly to C1.
The evaluation-reproducibility literature finds that configuration mismatches of
exactly this kind produce 10–20 pp gaps that are *perfectly reproducible* — openvla#282
reports 68% vs a published 88.4% ± 0.8 and the number does not move when the seed
changes; `vla-eval` documents hidden normalisation statistics and ambiguous
termination semantics as the mechanisms. A wrong `control_mode` is a candidate
explanation for lerobot#3264 itself.

**Direction.** Resolve it before anything else in Phase 2 — it is one line of config
and it may be the whole of C1. Then record it in the fingerprint by name (see the
architecture review's finding #2 and M15).

---

## C5. Published LIBERO scores may reflect memorisation — and this inverts §7b.1 — *[verified]*

Not in the document, and it reaches past it into the plan.

LIBERO-PRO (arXiv:2510.03827) argues that standard LIBERO evaluation measures rote
recall of training configurations rather than task competence. Their findings:

- Models achieving **over 90%** under standard LIBERO evaluation see performance
  **collapse to 0.0%** under fair perturbation.
- For OpenVLA and π0, success collapses once object displacement exceeds **0.2
  units**, "dropping sharply to zero thereafter"; π0.5 also degrades to zero.
- Models "reach toward memorized locations under positional perturbations" and
  "ignore meaningless instructions" — executing a task when given an instruction
  that specifies nothing coherent.
- "Reported accuracies above 90% largely reflect rote memorization of fixed mappings
  from the training set."

**Two consequences, of very different size.**

*The smaller one.* The ±5 pp gate certifies that our environment matches the
reference environment. That is exactly what `PLAN.md` §1 says it is for — "the
project's only defence against 'your environment was broken, not the model'" — and
it remains valid for that. But it certifies environment match, **not policy
competence**, and the docs occasionally lean on it as though it did both.

*The larger one.* `PLAN.md` §7b.1 offers four fixability classes and the project's
commercial proposition assumes `data` is the usual answer — the manifest's job being
to say *which* data. If the baseline policy has memorised fixed mappings and goes to
zero on a 0.2-unit displacement, then in the failing region it has **no
generalisation mechanism to feed**, and more demonstrations in that region teach
more memorisation. That is the definition of §7b.1's `architecture` class.

So `architecture` may be the **modal** answer on LIBERO rather than the rare one.
That is not fatal — it is arguably a more interesting finding, and §7b.1 exists
precisely so the project can say it — but it inverts the pitch, and it should be
decided deliberately rather than discovered in week 7.

It also raises a question about M16: LIBERO-Plus closed the camera gap to 92.8% with
20k trajectories, and it is worth asking whether that too is memorisation of the
perturbation set rather than acquired invariance. A held-out perturbation regime
would tell you, and it is the sharpest version of the targeted-vs-broad experiment.

**Direction.** Add LIBERO-PRO to the §7c discriminator battery as a *prior*, not just
a test: run the in-distribution replay (#3) and a small displacement sweep early, and
let the result set expectations for the whole manifest. And say plainly in the
readout what the gate does and does not certify.

---

## C6. The Octo row is unverified but sits in a verified table — *[by inspection]*

§1 lists Octo-small with `LIBERO ckpt: via LeRobot` — no checkpoint id, where
SmolVLA's row names two. §2 then assigns Octo the smoke-test role, which is load-
bearing: it is how the LeRobot path gets shaken out cheaply before SmolVLA.

The document's main virtue is separating verified from assumed. This row reads as
verified and is not.

**Direction.** Name the checkpoint or mark the row assumed. If no Octo LIBERO
checkpoint exists, the smoke-test role falls to SmolVLA itself and §2's "shakes out
the LeRobot path cheaply" is lost.

---

## C7. Free-tier T4 is Turing — no bf16, no FlashAttention-2 — *[by inspection]*

§6 assigns GR00T N1.7 inference to the free tiers on the strength of 16 GB. Correct
on memory (C2 agrees this is the right home for it), but T4 is `sm_75`:

- **No bf16.** GR00T checkpoints distributed in bf16 need fp16 or fp32 conversion;
  fp16 has a much narrower exponent range and flow-matching sampling is where
  numerical trouble shows up.
- **FlashAttention-2 requires Ampere or newer.** Expect eager or SDPA attention and
  a slower, more memory-hungry run than the 6 GB weight figure suggests.
- Kaggle's P100 option is `sm_60` — worse on both counts.

**Direction.** One footnote in §6's table and a note in §1's GR00T row. Verify
inference actually runs before the second-policy path is counted as available.

---

## C8. EGL risk is flagged for Kaggle and not for the new Blackwell laptop — *[measured]*

§6 consequence 3 correctly calls headless `MUJOCO_GL=egl` a known pain point in
hosted notebooks and budgets half a day, "have the laptop as fallback."

The laptop is an RTX PRO 1000 **Blackwell** on driver 595.84 — new silicon, and the
fallback for everything else. If EGL rendering does not work there, the fallback has
no fallback, and under C2 it is no longer the fallback but the primary runner.

**Direction.** Move "verify `MUJOCO_GL=egl` renders on the laptop" to the first day
of Phase 2, ahead of the Kaggle check. It is a ten-minute test that gates the entire
compute plan.

---

## C9. Licences are a gate, not a footnote — *[by inspection]*

§1 carries: "**Not yet verified: licences.** Confirm commercial-use terms per
checkpoint before anything client-facing. Do not assume Apache-2.0." Right call,
wrong weight. `PLAN.md` §8 and the proposal's §8 make this an iMerit/EXL commercial
artifact, so terms are a precondition on the deliverable, not a caveat inside it.

The surface is wider than checkpoints: LIBERO derives from robosuite/MuJoCo, LIBERO-plus
ships assets separately as an `assets.zip` from an HF dataset, and any remediation
dataset we generate inherits from those. Four separate licence chains.

**Direction.** An owner and a date, before week 3. Cheap now, potentially
project-ending later.

---

## C10. The rented-GPU estimate is ~5–10× conservative — and that changes the §1 decision — *[by inspection]*

§8 of `PLAN.md` states the unlock as "roughly **10–20 A100-hours ≈ $20–50**", noting
"the number is small enough to change the decision. Not pursuing it unless asked."

Using the plan's own cost model: ~3,000 episodes × (38 chunked forward passes ×
~30 ms for OpenVLA-OFT + ~1.5 s simulation/render) ≈ **2–3 GPU-hours**, not 10–20.
At spot rates that is **under $10** for the entire stress campaign. The document
undersells its own argument by roughly an order of magnitude.

**Why this matters rather than being a footnote.** `PLAN.md` §1 frames the choice as
binary — quantize OpenVLA (voids the gate) or go small (loses the headline model) —
and resolves it on the grounds that fine-tuning is what closes the loop. That
reasoning is sound and the conclusion for *Phase 5* is right. But it silently
assumes the same model must serve both roles, and the two roles have completely
different costs:

| Role | Requirement | Cost |
|---|---|---|
| **Gated reference** — reproduce a published number, run the stress campaign | inference only | **~$10 rented** |
| **The loop** — fine-tune on remediation data, re-measure | LoRA on 7.3 GiB | free, local, SmolVLA |

**The hybrid is strictly better than either pure option, and C1 makes it urgent.**
If the SmolVLA reproduction gate fails — which on present evidence is the expected
outcome — the project loses the defence that `PLAN.md` §1 calls its only one. A
rented OpenVLA-OFT baseline restores it for about the price of lunch, while SmolVLA
still carries Phase 5 locally. The two policies also give §7c discriminator #5
(cross-policy control) for free, which is otherwise the discriminator most likely to
be cut.

**Direction.** Promote the hybrid from an aside to the recommended plan, contingent
on C1. Concretely: rent for the OpenVLA-OFT ±5 pp gate and the headline stress
campaign; run SmolVLA locally for the loop; report both robustness profiles. Price
it properly — the current $20–50 figure is conservative enough to be arguing against
itself.

---

## Sources consulted

- [SmolVLA (arXiv:2506.01844)](https://arxiv.org/html/2506.01844v1) — 87.3% LIBERO average at 0.45B
- [LIBERO-PRO (arXiv:2510.03827)](https://arxiv.org/pdf/2510.03827) — memorisation, collapse to 0.0%, 0.2-unit displacement threshold
- [lerobot#3264](https://github.com/huggingface/lerobot/issues/3264) — SmolVLA LIBERO 73.25%
- [openvla#282](https://github.com/openvla/openvla/issues/282) — 68% vs 88.4% ± 0.8, invariant to seed
- [vla-eval (arXiv:2603.13966)](https://arxiv.org/html/2603.13966v1) — hidden normalisation, termination semantics
- [LeRobot LIBERO docs](https://huggingface.co/docs/lerobot/en/libero)
- [Kaggle notebook hardware](https://www.kaggle.com/docs/notebooks) · [Colab FAQ](https://research.google.com/colaboratory/faq.html) — 4 cores / 2 vCPU
- Local measurements: `nproc`, `lscpu`, `nvidia-smi` on the target laptop
