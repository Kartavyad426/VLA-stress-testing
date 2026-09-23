# Pre-registration — reproduction gate on arXiv:2603.19233 (GR00T)

**Written before the run.** Bands and decision rules fixed here; §5 forbids
editing them afterwards.

## Why a gate at all

The paper has been treated as a premise for several days of reasoning: its
GR00T pathway-ablation ordering (DiT 40–80% success drop, Eagle moderate, VL-SA
most resilient) is the prior behind "intervene at the action head", and its
per-token SAE recipe is the plan for feature work. No number in it has been
reproduced here. Our own bar — a policy stays on the roster only if it
reproduces its published number, which is why SmolVLA was vetoed — has never
been applied to this paper.

## Two things already established by reading their code, before any GPU time

1. **N1.5's DiT has 16 blocks; N1.7's has 32**, and the hook paths differ
   (`action_head.diffusion_model.transformer_blocks` vs
   `action_head.model.transformer_blocks`) — `third_party/action-atlas/
   experiments/groot_common.py:1-12`. Any "layer k" result in the paper is over
   a different depth than ours. Layer indices do NOT transfer; at best
   proportions do. This weakens the transfer of the ablation ordering
   independently of whether the gate passes.
2. **Their GR00T LIBERO checkpoints are community finetunes**, not NVIDIA
   releases: `aractingi/libero-groot-goal`, `liorbenhorin-nv/groot-libero_
   spatial-128_20000` (`model_adapters.py:421-428`). So the paper's GR00T rows
   describe a third-party finetune of N1.5. Reproducing on THEIR checkpoint is
   the right gate; it is not evidence about `nvidia/GR00T-N1.7-LIBERO`.

## What is run

Their harness, their checkpoint, our hardware. `experiments/baseline.py
--model groot --suite libero_goal`, no activation collection (8 GB VRAM; N1.7
measured 7.53 GiB at bf16, essentially weights only — hooks storing per-token
tensors will not fit and are not needed for the gate).

  * **Arm A — baseline**, n = 50. Published: **97%**.
  * **Arm B — null prompt**, n = 50. Published: **0%**.

## Bands, fixed now

**Tier 1 — absolute rate (Arm A vs 97%).** Governs whether absolute numbers
from the paper may be compared to ours at all.

| outcome | band | reading |
|---|---|---|
| PASS | >= 90% | reproduces on our hardware |
| AMBIGUOUS | 80–90% | likely harness/render/checkpoint difference, not a failed claim |
| FAIL | < 80% | at n=50 the Wilson interval for 80% excludes 97% |

**Tier 2 — the null-prompt contrast (Arm B).** This is the REAL gate, because
the claim being imported is about language dependence and pathway
specialisation, not about an absolute success rate. It is also robust to
harness differences: a baseline of 85% with a null-prompt of 0% still
reproduces the mechanism.

| outcome | band | reading |
|---|---|---|
| PASS | <= 10% | the mechanism claim reproduces |
| AMBIGUOUS | 10–40% | underpowered or partially reproducing; escalate n |
| FAIL | >= 40% | the headline language-dependence claim does not reproduce here |

## Decision rule

  * **Tier 2 PASS** — the paper's GR00T mechanism claims may be used as priors,
    with the N1.5→N1.7 depth caveat above attached to every one of them. The
    SAE recipe is worth adapting.
  * **Tier 2 FAIL** — the paper drops to a source of hypotheses and vocabulary.
    Localisation proceeds on our own measurements only, and the "intervene at
    the action head" prior loses its external support (it would still have
    weak internal support from `state_encoded`).
  * **Tier 1 FAIL with Tier 2 PASS** — treat all absolute numbers as
    incomparable and use contrasts only. Given F9 (render 360 vs 256 moved
    libero_spatial by 17pp), this is a live outcome, not a hedge.

## What we expect

Honest prior, recorded so it can be wrong: **Tier 1 AMBIGUOUS, Tier 2 PASS.**
Absolute LIBERO rates are harness-sensitive and we are running a community
checkpoint through their vendored lerobot on different hardware; the
null-prompt collapse is a large, sharp effect that should survive all of that.

## What this gate CANNOT establish

That any of it transfers to N1.7. Different depth, different conditioning
schedule, different checkpoint lineage. A pass licenses using their results as
a PRIOR for where to look on N1.7 — never as a measurement of it.
