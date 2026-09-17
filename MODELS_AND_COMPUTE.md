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
| **GR00T N1.7** | 3 B | 6.44 GiB weights (measured) | no | NVIDIA non-comm. N1.5 / N1.7 commercial | **4 LIBERO checkpoints EXIST** (`nvidia/GR00T-N1.7-LIBERO`, one per suite) | ~~**NOT LOADABLE BY LEROBOT — see below** ~~ **CORRECTED 2026-09-16: loadable — see §R1 below** |
| **π0 / π0.5** | ~3 B | **>8 GB stated** | **no** | no (>22.5 GB LoRA) | openpi LIBERO expert ckpts | OOMs on Jetson Orin NX. Full FT >70 GB. |
| **OpenVLA** | 7.4 B | ~15 GB bf16 | **no** (4-bit only) | no (8×A100 for full FT) | yes, published | Quantizing voids the ±5 pp gate — see `PLAN.md` §1 |
| **OpenVLA-OFT** | 7.4 B | ~15 GB | **no** | no | yes, per-suite | The proposal's headline. Needs rented GPU. |

Architecture notes that matter for us:

- **SmolVLA** = SigLIP 400M vision + SmolLM2 135M backbone.
- **OpenVLA** = SigLIP 400M + Llama-2 7B; ~5–8 Hz on an A100 (why OFT exists).
- **GR00T N1.7** = NVIDIA Eagle encoder + flow-matching action transformer.
- **π0** = flow matching. **GR00T and π0 both sample**, so `I1`/§5.1 in
  `ARCHITECTURE.md` applies hard: seed the noise draw or probes are noise.

### GR00T N1.7 — checkpoints exist, but not in a form we can load

**Correction (2026-09-16).** The survey recorded GR00T as having "a SimplerEnv-Bridge
checkpoint, not LIBERO". **Wrong: four LIBERO checkpoints exist**, one fine-tune per
suite, at `nvidia/GR00T-N1.7-LIBERO`.

But the blocker is not memory, and not licensing:

| | |
|---|---|
| inference weights | **6.44 GiB** (4.65 + 1.79 safetensors) vs 7.34 GiB free — marginal |
| the 133 GiB repo total | misleading: DeepSpeed optimiser shards (16 × 1.13 GiB/suite), **training artifacts** |
| **config format** | **NVIDIA-native**: `"architectures": ["Gr00tN1d7"]`, `diffusion_model_cfg`, `backbone_embedding_dim` |
| what LeRobot expects | `GrootConfig(type="groot", chunk_size, normalization_mapping, …)` |
| LeRobot-hosted GR00T checkpoint | **none exists** |

LeRobot's `model_type = "Gr00tN1d7"` matches, so the *model class* is right — the
*config* is not. Using it therefore means either **adopting NVIDIA's Isaac-GR00T
stack** (separate install, inference path and observation contract, plus a third
adapter for us) or **hand-translating the config**, which means guessing
normalisation statistics. That second option is precisely the un-normalisation
failure the proposal warns "can manufacture apparent model failures", and we would
have no way to distinguish a bad translation from a weak model.

**Verdict: not now.** `peft`, `diffusers`, `timm`, `dm-tree` are installed, so the
door is open if a LeRobot-format checkpoint appears.

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

---

# Revision — 2026-09-16

**Source of truth for this section: the official LeRobot docs**
(`huggingface.co/docs/lerobot/main/en/`), cross-checked against our own
installed checkout at `third_party/lerobot/src/lerobot` (v0.6.2). Where this
section and anything above it disagree, **this section wins** — §1 above was
verified 2026-09-11 by web search and has since gone stale in at least one
material way (see R1).

Every claim below is tagged **[verified]** (read from the docs or our installed
source) or **[unverified]** (stated but not yet tested here).

---

## R1. The GR00T entry above is WRONG — corrected

`MODELS_AND_COMPUTE.md` §1 and commit `b198613` both conclude that GR00T LIBERO
checkpoints exist but "are not LeRobot-loadable", because "LeRobot expects its
own `GrootConfig`, and no LeRobot-hosted GR00T checkpoint exists".

**Both halves are false.** [verified]

LeRobot has a first-class `groot` policy type, present in OUR installed
checkout:

```
third_party/lerobot/src/lerobot/policies/groot/
    configuration_groot.py      defines GrootConfig
    modeling_groot.py
    groot_n1_7.py
    processor_groot.py
factory.py:50                   from .groot.configuration_groot import GrootConfig
```

And NVIDIA publishes **LeRobot-format** LIBERO checkpoints, built against
LeRobot 0.6.1 (we run 0.6.2), Apache-2.0:

| Suite | Success | Checkpoint |
|---|---:|---|
| Spatial | 95% | `nvidia/gr00t17-lerobot-libero_spatial-640` |
| Object | 100% | `nvidia/gr00t17-lerobot-libero_object-640` |
| Goal | 98% | `nvidia/gr00t17-lerobot-libero_goal-640` |
| Long | 93% | `nvidia/gr00t17-lerobot-libero_10-640` |
| **Average** | **96.5%** | at `eval.n_episodes >= 50` |

```bash
lerobot-eval --policy.type=groot --policy.base_model_path=$MODEL_ID \
  --policy.embodiment_tag=libero_sim --env.type=libero \
  --env.task=libero_spatial --eval.n_episodes=50
```

**What the old entry got right:** GR00T N1.5 support WAS removed — current
LeRobot supports N1.7 only, and N1.5 checkpoints are rejected with a migration
note. Pin `lerobot==0.5.1` for N1.5. The reasoning was correct for NVIDIA's
*native* Isaac-GR00T format; it missed that LeRobot added a `groot` policy type
and that NVIDIA published LeRobot-format conversions.

**The real blocker is VRAM, not format.** [unverified] 3B params, HF lists F32
tensors so bf16 loading is required. `b198613` measured 6.44 GiB inference
weights against ~7.34 GiB free on our 8 GB card — feasible but marginal, and
the `-640` checkpoints imply large activations. Needs a ~20 min test.

---

## R2. Correction to the reproducibility story

An earlier reading of this project's evidence — mine included — was that LIBERO
reproduction is broken ecosystem-wide. **That is too strong.** [verified]

LeRobot **does** reproduce published LIBERO numbers, for at least one model:

| Model | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| Pi0.5 (LeRobot) | 97.0 | 99.0 | 98.0 | 96.0 | **97.5** |
| Pi0.5 (OpenPI, original) | 98.8 | 98.2 | 98.0 | 92.4 | **96.85** |

Checkpoint: `lerobot/pi05_libero_finetuned`, evaluated with
`--policy.n_action_steps=10` (matching OpenPI).

So the harness is capable of reproducing. That makes our ~67% vs ~87.3%
**more** likely to be a configuration problem on our side, not less — which is
good news, because configuration is fixable. The open reproduction reports
(lerobot#3264 SmolVLA 73.25%, lerobot#2114 pi0/pi0.5, openvla#282) are real but
do not amount to "the benchmark cannot be reproduced here".

---

## R3. Three concrete config leads on our ~20-point gap

All from the official LIBERO page. None tested yet. [unverified]

**(a) Control mode.** The docs are explicit:

> LIBERO supports two control modes — `relative` (default) and `absolute`.
> **Different VLA checkpoints are trained with different action
> parameterizations, so make sure the mode matches your policy.**

We run `relative` (see any `fingerprint.env.control_mode` in our traces). We
have never verified that `HuggingFaceVLA/smolvla_libero` was trained with
relative actions. A mismatch here would degrade everything silently and
uniformly — exactly the shape of our gap.

**(b) Image resolution.** Both official LIBERO datasets ship **2× 256×256×3**
cameras. This corroborates F9: LeRobot renders 360×360 while the checkpoint
declares 256×256, already measured as worth +17pp on spatial alone. The
datasets confirm 256 is the trained-on resolution.

**(c) `n_action_steps`.** LeRobot's own pi0.5 reproduction sets
`n_action_steps=10` to match the original implementation. We pass `--nas`;
whether our value matches SmolVLA's training-time chunking is unverified.

Also worth noting for protocol: the docs recommend **10 episodes per task
across all four suites (400 episodes), averaged over 3 seeds**, with
`--env.init_states=true` and **hard resets** ("use hard resets when reproducing
benchmark results"). We already use hard resets.

---

## R4. What LeRobot actually ships, as installed here

19 registered policy types in our checkout [verified]:

```
act  diffusion  eo1  evo1  fastwam  gaussian_actor  groot  lingbot_va
molmoact2  multi_task_dit  pi0  pi05  pi0_fast  smolvla  tdmpc
vla_jepa  vqbet  wall_x  xvla
```

(Directories also exist for `rtc` and `pi_gemma`.)

**With known LIBERO checkpoints:**

| Policy | Checkpoint | LIBERO avg | Native? |
|---|---|---|---|
| `pi05` | `lerobot/pi05_libero_finetuned` | **97.5%** | yes [verified] |
| `groot` | `nvidia/gr00t17-lerobot-libero_*-640` | **96.5%** | yes [verified] |
| `smolvla` | `HuggingFaceVLA/smolvla_libero` | 87.3% published / **67% ours** | yes, in use |
| — | `k1000dai/MINERVA` | 95.75% | LeRobot-format safetensors [verified] |

The remaining 15 policy types have no LIBERO checkpoint we have found. `act`,
`diffusion`, `vqbet`, `tdmpc` are imitation-learning baselines intended to be
trained on your own data, not pretrained VLAs.

---

## R5. VRAM — the guide above conflates training with inference

LeRobot's Compute & Hardware Guide gives this table, and it is a **training**
table (batch size 8, AdamW, which "adds ~30–100% over a forward+backward pass
alone"):

| Group | Policies | Peak VRAM (BS 8, AdamW) |
|---|---|---|
| Light BC | `act`, `vqbet`, `tdmpc` | ~2–6 GB |
| Diffusion | `diffusion`, `multi_task_dit` | ~8–14 GB |
| Small VLA | `smolvla` | ~10–16 GB |
| Large VLA | `pi0`, `pi0_fast`, `pi05`, `xvla`, `wall_x` | ~24–40 GB |
| Multimodal | `groot`, `eo1` | ~24–40 GB |

**We do inference only.** Every number above overstates our requirement, in
some cases by an order of magnitude — SmolVLA's "~10–16 GB" training envelope
runs inference in ~1–2 GB on our 8 GB card today. Do not use this table to rule
a policy out for *evaluation*; use it to rule one out for *fine-tuning*.

On fine-tuning: the guide puts our 8 GB card below its lowest consumer tier
(24 GB), so any training we do needs rented or free-tier compute. That part of
§1 above stands.

---

## R6. What this changes about model selection

Superseding the recommendation in `PENDING_DECISIONS.md` §10:

1. **MINERVA** (`k1000dai/MINERVA`, 0.54M) — still the best *control*.
   LeRobot-format safetensors, trivially fits, and having no language encoder
   means there is nothing to misconfigure. Its card pins **MuJoCo 3.3.2 "due to
   renderer sensitivity"**; we run 3.3.7. [verified]
2. **pi0.5** (`lerobot/pi05_libero_finetuned`) — the only checkpoint with a
   *documented LeRobot reproduction* (97.5%). That makes it the best reference
   for "is our harness configured correctly", though at ~3B it may not fit our
   card for inference. [unverified]
3. **GR00T N1.7** — now known to be loadable (R1). Worth the VRAM test.
4. **VLA-Adapter / MiniVLA / TurboVLA** — deprioritised. None is LeRobot-native
   as far as we have checked, and R3 suggests our gap is a config problem that a
   new model would not diagnose.

**Cheapest first move is still not a model.** R3(a) — checking whether
SmolVLA expects `relative` or `absolute` actions — costs minutes and could
account for a large share of the gap on its own.

---

## R7. Environment isolation — a standing constraint (user directive, 2026-09-16)

**Every additional policy gets its own virtual environment. No exceptions.**

The existing layout already follows this — `.venvs/lerobot` is separate from the
stdlib-only harness core — and it extends to every model added from here:

```
.venvs/lerobot      SmolVLA, pi0/pi05, groot  (shared LeRobot stack)
.venvs/minerva      MINERVA         — pins MuJoCo 3.3.2, NOT 3.3.7
.venvs/<model>      one per model with conflicting pins
```

**Why this is not optional here.** MINERVA pins MuJoCo 3.3.2 "due to renderer
sensitivity"; we run 3.3.7 for everything else. GR00T N1.5 needs
`lerobot==0.5.1` while N1.7 needs current. LeRobot's LIBERO extra pins its own
`robosuite`/`libero` versions. These conflict *directly* — a single shared
environment would silently resolve to one set of pins and every downstream
number would be attributable to the wrong cause. That is the un-normalisation
failure mode the proposal warns manufactures apparent model failures, arriving
by a different route.

**Plug and play is the other half of the requirement.** Isolation must not leak
into the harness. The contract:

- **L0 stays stdlib-only.** The harness core imports nothing from any model env.
  Adapters import lazily, as `envs/libero_env.py` already does.
- **One adapter per model, behind the existing policy protocol.** Adding a model
  means adding an adapter and a venv, never editing the miner, the classifier or
  the runner.
- **The venv is selected by the runner, not hardcoded.** A model is named; its
  interpreter path is resolved from a registry, so `--policy minerva` is the
  whole user-facing change.
- **Traces stay comparable.** Every rollout already records `policy_id` and a
  `fingerprint` including env + library identity. Those must keep resolving
  correctly across venvs — that is what makes cross-model comparison honest
  rather than assumed. A version difference must show up in the fingerprint, not
  be discovered later.
- **`semantic_deps()` must include the rendering backend and MuJoCo version.**
  R3(b) and the MINERVA pin both say the renderer changes the pixels the policy
  sees. If MuJoCo version is not in the fingerprint, two runs that differ only
  by renderer will look identical in the trace store.

**Not yet built.** The registry and per-model interpreter dispatch do not exist;
today `experiments/e2e.py` hardcodes `.venvs/lerobot`. This is a prerequisite
for adding the second model, not an afterthought.

---

## R8. Priority and sequencing (user directive, 2026-09-16)

**GR00T N1.7 is the project's priority policy.** The route there is gated:

1. MINERVA reproduces in our harness — a known-good checkpoint with a
   documented eval proves the harness can hit a published number.
2. The harness is verified correct (camera perturbation, stale first frame,
   detector abstention, goal instrumentation — `PENDING_DECISIONS.md`).
3. Then experimentation moves to GR00T N1.7.

Rationale for the order: GR00T is the most expensive policy to run on our 8 GB
card and the least documented (no published eval command for its LeRobot
LIBERO checkpoints). Any result on it is only interpretable once the harness has
reproduced a documented number, so harness defects are not misattributed to
GR00T.
