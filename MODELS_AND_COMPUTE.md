# Runnable VLAs & Compute Options

**Purpose:** what we can actually run, on what, for how much.
**Verified:** 2026-09-11 by web search. Re-verify before acting — this field moves fast.
**Hardware baseline:** RTX PRO 1000 Blackwell laptop, **8 GB VRAM**, Linux (MuJoCo needs Linux — we qualify).

---

## 0. Headlines

**1. The blocking unknown from `PLAN.md` §11.1 is RESOLVED.** A SmolVLA LIBERO
checkpoint exists: [`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero)
(also `lerobot/smolvla_libero`). We do not have to fine-tune our own. **Phase 2 is unblocked.**

**2. LeRobot ships first-class LIBERO-plus support with a CLI.** The perturbation
benchmark is one command, not an integration project:

```bash
lerobot-eval --policy.path=HuggingFaceVLA/smolvla_libero \
  --env.type=libero_plus --env.task=libero_spatial \
  --eval.n_episodes=10 --eval.batch_size=1
```

This removes most of the 2–3 weeks `PLAN.md` §5 budgeted for simulator integration.

**3. There is a known, OPEN reproducibility problem with exactly the checkpoint
we want.** [lerobot#3264](https://github.com/huggingface/lerobot/issues/3264)
(Apr 2026, unresolved, no maintainer response): a user on LeRobot v0.5.1 /
MuJoCo 3.3.2 / LIBERO v3.0 could not reproduce published SmolVLA LIBERO numbers,
obtaining Object 93.0 / Goal 81.0 / Spatial 63.0 / Long 56.0, overall **73.25%**.

This is the single most important finding in this document. See §4.

**4. Free tiers give 16 GB, double our laptop.** Colab and Kaggle T4s unlock
models the laptop cannot hold. Session limits make resumable sweeps mandatory —
which we already built (G10).

**5. Hosted VLA APIs are not a viable path.** See §5.

---

## 1. Models vs 8 GB

| Model | Params | Inference VRAM | On our 8 GB? | LoRA on 8 GB? | LIBERO ckpt | Notes |
|---|---|---|---|---|---|---|
| **Octo-small** | 27 M | <1 GB | **yes** | yes | via LeRobot | 93 M variant also exists. Runs in ~4 GB. Fast, weak. Good smoke test. |
| **SmolVLA** | 450 M (some sources 535 M) | ~1–2 GB | **yes** | **yes** | **`HuggingFaceVLA/smolvla_libero`** | *Primary candidate.* ~25 Hz on A100; runs on Jetson Orin NX. |
| **GR00T N1.7** | 3 B | ~6 GB weights | **tight** — inference maybe, training no | no | `nvidia/GR00T-N1.7-LIBERO` | Flow-matching action head ⇒ **stochastic, must seed** |
| **π0 / π0.5** | ~3 B | **>8 GB stated** | **no** | no (>22.5 GB LoRA) | openpi LIBERO expert ckpts | OOMs on Jetson Orin NX. Full FT >70 GB. |
| **OpenVLA** | 7.4 B | ~15 GB bf16 | **no** (4-bit only) | no (8×A100 for full FT) | yes, published | Quantizing voids the ±5 pp gate — see `PLAN.md` §1 |
| **OpenVLA-OFT** | 7.4 B | ~15 GB | **no** | no | yes, per-suite | The proposal's headline. Needs rented GPU. |

Architecture notes that matter for us:

- **SmolVLA** = SigLIP 400M vision + SmolLM2 135M backbone.
- **OpenVLA** = SigLIP 400M + Llama-2 7B; ~5–8 Hz on an A100 (why OFT exists).
- **GR00T N1.7** = NVIDIA Eagle encoder + flow-matching action transformer.
- **π0** = flow matching. **GR00T and π0 both sample**, so `I1`/§5.1 in
  `ARCHITECTURE.md` applies hard: seed the noise draw or probes are noise.

> **Not yet verified: licences.** Confirm commercial-use terms per checkpoint
> before anything client-facing. Do not assume Apache-2.0.

---

## 2. Recommended stack

| Role | Choice | Why |
|---|---|---|
| Oracle / CI | `ScriptedReachPolicy` | Known answer. Already gating the miner. |
| Smoke test | Octo-small | Trivial VRAM; shakes out the LeRobot path cheaply |
| **Primary** | **SmolVLA + `HuggingFaceVLA/smolvla_libero`** | Only model that fits inference **and** LoRA on 8 GB ⇒ **Phase 5 closes the loop locally** |
| Second policy | GR00T N1.7 (inference only, free tier) | Different architecture ⇒ genuinely informative comparison |
| Stretch | OpenVLA-OFT | Rented GPU, ~$20–50. Only if a published-number gate is needed. |

SmolVLA remains the right primary for the reason in `PLAN.md` §1: it is the
largest model that can be **fine-tuned** on this hardware, and the retrain step
is what converts the manifest from opinion into evidence.

---

## 3. Environment setup

Verified specifics from the LeRobot LIBERO-plus docs:

- **Linux only.** MuJoCo. `export MUJOCO_GL=egl` for headless.
- **LIBERO-plus REPLACES vanilla LIBERO** — it uninstalls `hf-libero` so `import
  libero` resolves to the fork. **Two separate environments/containers, confirmed**
  (this was an assumption in `PLAN.md` §7.1; it is now verified fact).
- Assets ship separately: `assets.zip` from the HF dataset, extracted into the package.
- System deps: `libexpat1 libfontconfig1-dev libmagickwand-dev`.

**Observation/action contract** (maps onto our L0 schema):

| LeRobot key | Shape | Our schema |
|---|---|---|
| `observation.state` | 8-dim (eef pos, axis-angle, gripper qpos) | `state` |
| `observation.images.image` | agentview HWC uint8 | `image_refs["agentview"]` (G2) |
| `observation.images.image2` | wrist HWC uint8 | `image_refs["wrist"]` |
| action | `Box(-1,1,(7,))` 6-D eef delta + gripper | `action_dims` (G1) |

**Two traps, both capable of manufacturing fake "model failures":**

- **`--env.control_mode`** is `relative` or `absolute`, and **must match the
  checkpoint's action parameterisation.** Wrong mode ⇒ confident, plausible,
  completely wrong motion.
- **Soft resets are not bit-identical to hard resets.** `--env.hard_reset=false`
  is faster but changes camera observations. **Use hard resets** — soft resets
  would contaminate the reproducibility floor (`ARCHITECTURE.md` §5.1).

Suite step limits: Spatial/Object 280, Goal 300, Long 520, LIBERO-90 400.
Feeds the §8 cost model in `PLAN.md` — Long episodes cost ~2× Spatial.

---

## 4. The reproducibility problem — read before Phase 2

[lerobot#3264](https://github.com/huggingface/lerobot/issues/3264) reports being
unable to reproduce SmolVLA's published LIBERO numbers with official checkpoints
and current dependencies. Open, no maintainer response.

**Why it matters to us specifically.** `PLAN.md` §5 sets a reproduction gate:
match published success within a tolerance before trusting any perturbation
result. That gate is the only defence against "your environment was broken, not
the model." **A publicly unresolved reproduction failure on our chosen checkpoint
attacks that gate directly.**

Three possible readings, and we cannot tell which without measuring:
1. an environment/version mismatch on the reporter's side (our gate would catch it);
2. a genuine regression in LeRobot or the checkpoint (our gate would fail, correctly);
3. the published numbers are not reproducible (the gate is unavailable to us at all).

**Action — do this before committing to SmolVLA:** run the suites, compare against
published, and treat the outcome as a finding either way. If we cannot reproduce,
we document our own baseline over 3 seeds, state plainly in the readout that the
gate is weaker than intended, and consider Octo or a rented OpenVLA-OFT run as the
gated reference instead.

Budget a week. **This is now the top Phase-2 risk**, replacing the
checkpoint-availability risk it resolved.

---

## 5. Hosted APIs — why this doesn't work

NVIDIA NIM offers a free OpenAI-compatible endpoint (`integrate.api.nvidia.com/v1`,
~40 req/min, free Developer Program signup) across 100+ models — but those are
**LLMs, not VLA policies**. No robot-action endpoint was found.

**Even if one existed, it would be unusable for this workload**, and the reason is
structural rather than about pricing:

A VLA is a *closed loop with a simulator*. Each step needs an action before the
sim can advance, so calls are strictly serial and latency is per-step. At ~38
forward passes per episode (chunked) and a 40 req/min limit, that is **roughly
one episode per minute** — our ~3,000-episode campaign becomes ~50 hours of
wall-clock, dominated by network round-trips, with no batching available because
step *t+1* depends on step *t*.

Local inference on a small model is faster, free, reproducible, and doesn't put a
third party's rate limiter inside our measurement path.

**Conclusion: no API path. Local weights only.** This is settled, not a trade-off
to revisit.

---

## 6. Compute options

| Option | GPU / VRAM | Free allowance | Session cap | Best for |
|---|---|---|---|---|
| **This laptop** | 8 GB Blackwell | unlimited | none | dev, oracle CI, SmolVLA/Octo, **LoRA** |
| **Kaggle** | T4 / P100, **16 GB** | ~30 h/week | 9–12 h | overnight sweeps; most reliable free tier |
| **Colab** | T4 / P100, **16 GB** | ~15–30 h/week | 12 h | ad-hoc; less predictable |
| **Lightning AI** | T4/L4/A10G/L40S | ~80 GPU-h/mo (sources vary; some say 15 credits ≈ 22 h T4) | persistent workspace | closest to a real dev box |
| Rented A100/H100 | 40–80 GB | — | — | OpenVLA-OFT, ~$20–50 for the whole campaign |
| Research credits | cluster | large | — | NSF ACCESS, Google TRC, AWS Research — worth applying if the timeline allows |

Colab + Kaggle together ≈ **up to 60 free GPU-hours/week**, no credit card.

**The important asymmetry: free-tier T4s have 16 GB — double the laptop.** That
brings GR00T N1.7 inference into range and gives SmolVLA LoRA far more headroom.
The laptop stays the dev box; free tiers become the batch runners.

**Three consequences for how we build:**

1. **Session caps make resumability mandatory.** A 9–12 h cap will cut an
   overnight sweep. G10 already covers this — and `run_cell` now counts cached
   outcomes correctly, which is exactly the bug that would have silently
   corrupted every resumed run.
2. **Traces must survive the session.** Notebook storage is ephemeral: push
   `rollouts.jsonl` to HF Hub or object storage as it is written, not at the end.
3. **`MUJOCO_GL=egl` headless rendering in hosted notebooks is a known pain
   point.** Verify it renders on Kaggle *before* planning a campaign around it.
   Budget half a day; have the laptop as fallback.

---

## 7. Changes to `PLAN.md`

| Was | Now |
|---|---|
| §11.1 "does a LIBERO-capable small checkpoint exist?" — top unknown | **Resolved: yes.** Replaced by the §4 reproducibility risk. |
| §5 "2–3 weeks, dominated by MuJoCo/EGL setup pain" | Likely shorter — LeRobot ships the CLI. Still budget for the dual-env split and EGL. |
| §7.1 "keep LIBERO-plus in a separate container" (assumption) | **Verified fact** — it uninstalls vanilla LIBERO. |
| §8 cost model: 300 steps/episode assumed | Use real caps: 280 Spatial/Object, 300 Goal, 520 Long. |
| Laptop-only | Laptop + ~60 free GPU-h/week at 16 GB. GR00T second-policy path viable without spending. |

---

## Sources

- [SmolVLA LIBERO checkpoint (HuggingFaceVLA)](https://huggingface.co/HuggingFaceVLA/smolvla_libero) · [lerobot/smolvla_libero](https://huggingface.co/lerobot/smolvla_libero) · [smolvla_base](https://huggingface.co/lerobot/smolvla_base)
- [SmolVLA announcement](https://huggingface.co/blog/smolvla)
- [LeRobot LIBERO-plus docs](https://huggingface.co/docs/lerobot/libero_plus) · [LeRobot LIBERO docs](https://huggingface.co/docs/lerobot/en/libero)
- [lerobot#3264 — SmolVLA LIBERO reproduction failure](https://github.com/huggingface/lerobot/issues/3264)
- [LIBERO-Plus paper (arXiv:2510.13626)](https://arxiv.org/abs/2510.13626) · [LIBERO-plus repo](https://github.com/sylvestf/LIBERO-plus)
- [OpenVLA paper (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246) · [OpenVLA repo](https://github.com/openvla/openvla)
- [Isaac GR00T repo](https://github.com/NVIDIA/Isaac-GR00T) · [GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) · [GR00T LIBERO eval issue #136](https://github.com/NVIDIA/Isaac-GR00T/issues/136)
- [VLA models comparison (SVRC)](https://www.roboticscenter.ai/tools/vla-models-comparison) · [OpenELAB VLA comparison](https://openelab.io/blogs/learn/vla-models-for-robot-arms)
- [awesome-vision-language-action](https://github.com/ai4s-research/awesome-vision-language-action)
- [NVIDIA NIM free tier](https://freellm.net/providers/nvidia-nim) · [NIM pricing/limits](https://decodethefuture.org/en/nvidia-nim-api-pricing-limits-guide/)
- [Colab vs Kaggle free GPU (Clusy)](https://www.clusy.io/compare/colab-vs-kaggle) · [Colab alternatives (Thunder Compute)](https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025) · [Free GPU credits for researchers](https://www.aquanode.io/blog/free-gpu-credits-for-students-and-researchers)
