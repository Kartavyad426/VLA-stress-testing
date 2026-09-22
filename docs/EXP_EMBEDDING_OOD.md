# EXPERIMENT PLAN — embedding-space OOD for GR00T

*Written 2026-09-18. **PLANNED, not run.** Expectations in §5 are pre-registered: they are written
here before any result exists, per `RESULTS.md`'s convention. Do not edit them after the fact — if
they are wrong, that is the finding.*

---

## 1. The question

Our current OOD detector works in **5 proprioceptive dimensions** (eef xyz + two gripper fingers) and
separates GR00T's LIBERO-Plus failures from its successes at **6.9×**, p = 6e-66. It is blind to
everything that does not move the arm: lighting, texture, background, distractors, instruction
wording.

**Does moving to the policy's own embedding space see what proprioception cannot — and does it
predict failure better?**

Secondary, and the reason this is worth GPU time rather than being a curiosity: **which signal
detects which kind of perturbation?** LIBERO-Plus labels every instance with the dimension that was
perturbed, so this experiment can produce a **signal × perturbation-type** table that nothing in our
pipeline currently has.

---

## 2. What we capture

### 2.1 The N1.7 architecture, and why it dictates the tap points

The published figure is from **GR00T N1** (`arXiv:2503.14734`). **N1.7 differs in the contents of both
boxes**, so this is the version we are actually instrumenting — all of it read from the installed
code and the cached checkpoint **[F]**:

```
  Image 256x256 --.                    SYSTEM 2                         SYSTEM 1
                   >- Qwen3VLProcessor -> [ nvidia/Cosmos-Reason2-2B ]
  Text ----------'                       (Qwen3-VL, flash-attn-2)
                                          1.523 B params (48.5%)
                                                   |
                                      backbone_features  [S1: pre-vlln]
                                                   |
                                        vlln -> vl_self_attention (4 blk, 2048)
                                                   |
                                             vl_embeds  [S2, 2048]
                                                   |
                                          CROSS-ATTENTION  (encoder_hidden_states)
                                                   |
  state (132 x 1) -> state_encoder -> state_features [S0e, 1536] --.        v
                     per-embodiment                                 >-- cat --> [ DiT ]
  actions (40 x 132) -> action_encoder -> action_features [1536] --'  sa_embs   32 layers
       ^                                                                        1.092 B
       |                                                                          |
       '----- actions += dt * pred ------- action_decoder (1024->132) <-----------'
              x4 Euler steps  [S4 / S4t]
                     |
                     v
              action_pred (40 x 132)  [S3]  -> motor action (7 used)
```

**Corrections to the N1 figure, from the code [F]:**

- **VL does not merge into the token stream — it enters by CROSS-ATTENTION.** `self.model(
  hidden_states=sa_embs, encoder_hidden_states=vl_embeds, ...)` (`groot_n1_7.py:599-613, 695-706`).
  What is *concatenated* is state + action: `sa_embs = torch.cat((state_features, action_features),
  dim=1)`. The N1 diagram's single merge point is three different mechanisms.
- **The action chunk is 132-wide, not 7.** `self.action_dim = config.max_action_dim` = **132**
  (`:484`); LIBERO uses 7 of them and the rest are padding. Any distance computed on S3 must slice the
  live dimensions or the padding will dominate.
- **`state_history_length = 1`** (`:108`) — S0e is one token per step, not a window.
- **`input_embedding_dim = 1536`** (`:107`) — both `state_encoder` and `action_encoder` emit 1536.

### 2.1b What is actually new in N1.7 — from NVIDIA's own repo

Primary source: [github.com/NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) **[A]**.
Everything above was read from LeRobot's *port*; these are NVIDIA's claims about the original.

| change | N1.6 → N1.7 | relevance to us |
|---|---|---|
| **VLM backbone** | Eagle → **`nvidia/Cosmos-Reason2-2B`** (Qwen3-VL) | confirmed in our config |
| **Native aspect ratio** | padded → *"encodes images in their native aspect ratio without padding"*, flexible resolution | **S1's geometry is resolution-dependent**; a perturbation changing image shape changes tokenisation |
| **Relative EEF action space** | absolute → *"relative end-effector action space shared across robot and human embodiments"* | **OFF in our checkpoint** — `use_relative_actions: False` **[F]**. So we run the absolute variant, not the headline N1.7 feature |
| **Human video pretraining** | — → *"20K hours of EgoScale human video data"* | the backbone has priors from human video, not only robot demos. Relevant to what S1 considers in-distribution |
| **Action horizon** | 16 → **40** | matches `action_horizon: 40` |
| **State/action dims** | 29 → **132** | matches; and confirms the padding point |
| **DiT depth** | *"updated from 32 to 16 diffusion layers"* | **⚠ CONTRADICTS our config — see below** |

> **⚠ TWO DISCREPANCIES, neither resolved.**
>
> **1. DiT depth.** NVIDIA's README says N1.7 went from 32 to 16 diffusion layers. Our checkpoint's
> `diffusion_model_cfg` says **`num_layers: 32`** **[F]**. A plausible reconciliation is that the
> README counts text-attending blocks — the config has `attend_text_every_n_blocks: 2` and
> `interleave_self_attention: True`, so 32 layers would contain 16 text-attending ones — **but this is
> a guess and has not been verified.** It does not change the tap points; it does change any claim
> about model depth.
>
> **2. `n_action_steps` is 16 in the checkpoint, not the class default of 40.**
> `config.json: n_action_steps: 16` **[F]**. See §2.1c.

### 2.1c How often the model actually runs — corrected

An earlier draft of this plan said "~3 forward passes per episode" from the class default of 40.
**Wrong on two counts.**

The mechanism is right: `select_action` only calls `predict_action_chunk` when the action queue
empties (`modeling_groot.py:526-530`), so the model runs **once per chunk**, not once per frame. For
the intervening steps the policy executes queued actions **open-loop, without observing**.

But the number is set by **`n_action_steps`, which our checkpoint declares as 16** — so at ~118 steps
per episode that is **≈ 7–8 forward passes**, not 3.

**Still unresolved [?]:** the harness can override the checkpoint
(`lerobot_policy.py:105, 134`), our stored `meta` is `None`, and every GR00T run shares one
`policy_id` (`@0e9ffa56`) — so all runs used the *same* setting, but which one is not readable from
the traces. It **is** recoverable: `n_action_steps` is part of `identity()` (`:79`), so recomputing
`derive_id` for candidate values and matching the hash settles it. **Do that before sizing the
capture.**

**Why this matters beyond bookkeeping:** at `n_action_steps = 16`, GR00T is not looking at the world
for **15 of every 16 steps**. That bounds the granularity of any embedding-derived signal — an
input-side detector can fire at most ~8 times an episode — and it is plausibly part of *why* the
policy drifts off-manifold at all, since it cannot see itself drifting.

**Also worth noting:** `normalization_mapping` is `IDENTITY` for VISUAL, STATE and ACTION **[F]** —
normalisation lives in the processor, not the policy. Any distance computed on S0e or S3 is in
**raw**, not normalised, units.

**Three N1.7-specific facts that change the experiment:**

1. **Only 4 denoising steps.** `num_inference_timesteps = 4`, integrated as
   `dt = 1/num_inference_timesteps` over `for t_step in range(...)` (`groot_n1_7.py:662, 683`). So S4
   is **cheap** — four Euler steps per resample — and we can additionally capture **all four
   intermediate states for free**, giving a denoising *trajectory* rather than only its spread.
2. **The state and action encoders are `CategorySpecificLinear`, per-embodiment**, with
   `max_num_embodiments: 32`; LIBERO occupies `libero_sim: 2`. The S0e tensor we tap is therefore
   **embodiment-specific**, so its geometry is not comparable across embodiments — fine here, since
   everything is LIBERO, but it must not be pooled across embodiments later.
3. **The backbone is a general VLM, not a robotics-trained encoder.** Cosmos-Reason2-2B via Qwen3-VL,
   with the image processor owned by the backbone rather than by the policy config. That is the
   reason to expect S1 to carry *scene appearance* well and *task-relevant distinctions* less well.

**The topology point is unchanged and still governs the design:** the robot state **bypasses System 2
entirely** and meets the vision-language stream only inside the DiT. So there is **no single embedding
space holding vision, language and proprioception** upstream of System 1 — hence several taps — and
**S1 cannot see the state while S0 cannot see the pixels**, making them complementary by construction
rather than by observation.

**[?]** `nvidia/Cosmos-Reason2-2B` is nominally 2B but the backbone measures **1.52B** in our
checkpoint. Not reconciled; likely a naming convention rather than a different model, but the measured
number is the one used for all sizing here.

### 2.2 Tap table — verified against the code

**Three of the five taps need no hook at all.** `get_action_with_features` already returns them
(`groot_n1_7.py:710-716`):

```python
return BatchFeature(data={
    "action_pred":       actions,          # S3
    "backbone_features": vl_embeds,        # S2  (post vlln + vl_self_attention)
    "state_features":    state_features,   # S0e
})
```

| # | signal | how to get it | dim (per step) | detects |
|---|---|---|---|---|
| **S0** | raw proprioception | **already in traces** — no rerun needed | 5 | arm/gripper motion. **The 6.9× baseline** |
| **S0e** | `state_features` | **free** — returned key | **1536** | what the DiT conditions on, after per-embodiment encoding |
| **S2** | `vl_embeds` | **free** — returned as `backbone_features` | tokens × **2048** | VL stream as System 1 receives it |
| **S1** | pre-`vlln` backbone output | **needs a hook** — see below | tokens × **2048** | the VLM's own encoding, before the action head touches it |
| **S3** | `action_pred` | **free** — returned key | **40 × 132**, slice to 7 | intended action |
| **S4** | spread over k resamples | rerun `get_action_with_features` k=4 | scalar | self-consistency |
| **S4t** | the 4 Euler intermediates | **needs a hook** on the loop at `:683` | 4 × 40 × 132 | does the *path* to the action differ, not just its spread? |

> **⚠ The S1/S2 gotcha, and it would silently produce the wrong result.**
> `process_backbone_output` **overwrites the key in place**:
> ```python
> backbone_features = self.vlln(backbone_output["backbone_features"])
> backbone_output["backbone_features"] = self.vl_self_attention(backbone_features)   # :563-564
> ```
> So after that call, `backbone_features` **is** S2. A hook that reads the key post-call captures S2
> twice and reports it as S1 — and the two would look plausibly different because of `vlln`, so the
> bug would not be obvious. **S1 must be captured by a forward pre-hook on `self.vlln`**, or by
> copying the key before `process_backbone_output` runs.

> **A capability the code reveals that we should exploit:** there is an **RTC (real-time chunking)
> path** — `rtc_overlap_steps`, `rtc_frozen_steps`, `rtc_ramp_rate` (`:665-681`). When enabled,
> consecutive chunks **overlap**, which makes true chunk-to-chunk disagreement available — the
> Sentinel/TIDE signal proper, rather than the resampling proxy S4. Worth checking whether our eval
> config enables it before settling for S4. **[?]** not yet checked.

**Storage.** S1+S2 at 2048-dim fp16 (pooled over tokens), S0e at 1536-dim, ~120 steps/episode over
723 episodes ≈ **950 MB**. S3 at 40×132 fp16 ≈ 900 MB unless sliced to the 7 live dims first — **slice
it**, which drops it to ~50 MB. S4 negligible; S4t is 4× S3, so slice that too. Store `.npy` per episode beside the trace. **Do not store frames** —
RGB would be ~100× that and we need embeddings, not pixels.

## 2.3 IMPLEMENTATION — reviewed by `primary`, 2026-09-18

Their review corrected four things in §2.2. All verified against the code by them; recorded here so
the plan is buildable rather than merely sketched.

**A. The three "free" tensors are NOT free at the adapter level.** `predict_action_chunk` does
`actions = outputs.get("action_pred")` (`modeling_groot.py:502`) and **drops `backbone_features` and
`state_features` on the floor**, then slices to `prediction_horizon` and `original_action_dim`;
`select_action` then queues the chunk and pops **one** action. By the time
`lerobot_policy.py` returns, all that survives is a single 7-D vector. A wrapper there would have to
re-run the model — double cost.

**⇒ Intercept at `_groot_model.action_head.get_action`**, which returns the BatchFeature with all
three, **unsliced** (`action_pred` still 40 × 132, before truncation).

**B. `get_action` / `get_action_with_features` are plain methods, not `forward()`** — so
`nn.Module` forward hooks **do not fire on them**. Wrap the bound method at load time. Reserve real
hooks for genuine module calls:

| signal | mechanism |
|---|---|
| **S1** | `register_forward_pre_hook` on `action_head.vlln` |
| *(middle)* | `register_forward_hook` on the same module — **free**, see C |
| **S4t** | `register_forward_hook` on `action_head.action_decoder` — fires **once per Euler step**, 4× per `get_action` |

**C. There are THREE tensors around `:563-564`, not two.** Pre-`vlln` (**S1**), post-`vlln`/pre-attention
(**S1.5**, free from a post-hook on the same module), and post-attention (**S2** — which is what the
returned `backbone_features` / `vl_embeds` actually is). If the middle one is interesting it costs
nothing.

**D. S4t needs no patch, but needs the noise.** The `action_decoder` hook yields `pred`, which is the
**velocity**, not the action state. With RTC off `vel_strength` is identically 1, so the intermediates
reconstruct exactly as `actions₀ + dt · cumsum(pred[:, -action_horizon:])` — **provided `actions₀` is
captured**. It is a bare `torch.randn` (`:657`), so capture it in the method wrap or seed it,
otherwise S4t is unreconstructable.

### RTC is OFF — confirmed end to end

`select_action` calls `predict_action_chunk(batch)` with no kwargs → `_prepare_n1_7_rtc_inputs`
receives `prev_chunk_left_over=None` → returns `(inputs, None)` (`:354-355`) → `get_action` with no
options → `"action" in action_input` is False → **the whole RTC branch (`:665-681`) is dead and
`vel_strength` stays all-ones.** `configuration_groot.py` carries only
`rtc_ramp_rate: float | None = None`.

**So there is no chunk overlap and no true chunk-to-chunk disagreement.** S4's k-resample proxy is the
only available consistency signal. Enabling RTC means feeding leftovers back through the rollout loop,
which **changes the policy's behaviour** and therefore cannot be bolted onto an observational run.

### ⚠ The wrinkle that could invalidate the whole experiment

**The denoising loop starts from a bare `torch.randn` with no seed control on this path (`:657`), so a
re-run will NOT reproduce the current 468/623 success split.**

⇒ **The reference set must be the successes *of the capture run*, not the 98 from
`groot_harness_parity`.** Otherwise "the policy's own successes" silently refers to a different policy
state than the one being scored, and the leave-one-out calibration inherits that mismatch. The same
applies to which query episodes count as failures. **Label from the run you capture.**

### Corrected sizing

My §2.2 estimate of 25 MB was light. 5,632 dims (2048 + 2048 + 1536) at fp16 × ~8 forwards × 723
episodes ≈ **65 MB for embeddings alone**; adding `action_pred` (40 × 132 fp16 = 10.6 KB/forward) and
S4t's four intermediates gives **~600 MB unsliced, ~100 MB if sliced to the 7 live dims first**.
**Slice first.**

### Storage format

**Sidecar `.npz` per episode**, keyed by signal, plus a manifest mapping `rollout_id → path` and
**which forward index maps to which env step**. The reason is stronger than greppability: capture is
per *model forward* (~8) not per *step* (~118), so a field on `Step` would be **null 93% of the
time**. `rollouts.jsonl` is already 130 MB for the MINERVA run.

### Estimate and sequencing

**~4 h implementation (CPU, no GPU) + ~3.5 h GPU.** One of the four hours is verification and it is
**not optional**: a 2-episode smoke that asserts `vlln(S1)` matches the post-`vlln` tensor within
tolerance **and** that `S1 ≠ S2` — which fails loudly if a hook landed on the wrong side of the
in-place overwrite.

**De-risk before committing the GPU:** build and verify the capture (no GPU), then run **40 episodes
(20 success / 20 fail, ~10 min)** and check that S2 separates *at all* before spending 3.5 h.

### Why it is worth doing — sharper than my §1

`primary` supplied the motivation I did not have: **78 of GR00T's failures are robot-initial-state
variants currently filed under `visual_grounding`**, and a 5-D proprioceptive detector is blind by
construction to exactly the axis that would settle whether that label means anything. That is a
concrete mislabelling this experiment can adjudicate, not a general curiosity.

---

## 3. Runs

Reference and query, mirroring the existing proprioceptive analysis so results are directly
comparable:

| run | purpose | n | measured cost |
|---|---|---|---|
| `groot_embed_parity` | **reference** — nominal libero_spatial | 100 | 10.1 s/ep → **~17 min** |
| `groot_embed_lplus` | **query** — LIBERO-Plus libero_spatial | 623 | 16.0 s/ep → **~2.8 h** |

Costs are measured from the existing runs, not estimated. **Add ~20–30% for k=4 resampling on S4 and
for embedding writeback → budget ~4 hours GPU**, one arm, plus a short parity check first.

**Sequencing:** run the 100-episode reference first and confirm the embeddings are non-degenerate
(§6) before committing the 2.8-hour query.

---

## 4. Analysis

**A. Swap the feature space, keep the method.** Run `experiments/ood_selfref.py` unchanged except for
the feature vector, once per signal. Reference = nominal successes, threshold = leave-one-out p95,
report mean % steps OOD for success vs failure, separation ratio, Mann-Whitney p.
**The number to beat is S0's 6.9×.**

**B. Frame-level separability.** Per signal, fit a linear probe on frames from passing vs failing
episodes and report **AUROC with a held-out split by episode** (never by frame — frames within an
episode are not independent, and splitting by frame would leak). Report AUROC, not a UMAP picture; a
projection can look separated when a probe cannot separate.

**C. Signal × perturbation type.** LIBERO-Plus labels each instance's perturbed dimension. Cross
that against per-signal OOD rate. **This is the table that would be genuinely new**, and it is the
one that tells a client *which detector catches which failure cause*.

**D. Onset timing — resolves the open confound.** Our standing caveat is that drift may accompany
failure rather than cause it. With per-frame signals, compute for each failing episode the first
frame each signal crosses threshold, and compare against the failure point. **If S1 (input-side)
crosses before S0 (behaviour), the causal ordering is input-shift → behaviour-drift → failure.** If
everything crosses together at the end, all these signals are symptoms and should be described that
way.

---

## 5. Pre-registered expectations

Written before any result. **Confidence stated, so being wrong is informative.**

1. **S1/S2 will detect appearance perturbations (lighting, texture, background) that S0 misses.**
   *High confidence.* These change pixels and not arm position, so proprioception cannot see them by
   construction.

2. **S1/S2 will MISS the language-perturbation dimension** — instances where only the instruction was
   rewritten will look in-distribution in the VL embedding. *Medium-high confidence.*
   **This is the sharp falsifiable prediction and the main reason to run the experiment.** The
   reasoning: embedding-space OOD inherits the encoder's invariances, and these models are documented
   to under-use language (LIBERO-PRO: instruction swapped, output unchanged; our own probe: spatial
   tasks at `redirect_fraction 0.0`). If a language rewrite barely moves the embedding, the detector
   is blind exactly where the model is blind. **If S1 *does* flag language perturbations, my stated
   limitation on embedding-space OOD is wrong and should be retracted.**

3. **S4 (chunk consistency) will beat S1/S2 at predicting failure.** *Medium confidence*, from the
   survey's margins. If it holds, it is also the cheapest signal to ship — no image pipeline.

4. **S1 will cross threshold earlier than S0 in failing episodes.** *Medium confidence.* Input shift
   should precede behavioural drift.

5. **S0 will remain competitive on camera and robot-initial-state perturbations.** *Medium.* Those
   move the arm, which is what S0 measures.

**What would make this experiment a failure rather than a negative result:** if no signal separates
better than chance on any perturbation type, that would more likely indicate a capture bug than a
finding — see §6.

---

## 6. Controls and failure modes to check first

- **Non-degenerate embeddings.** Confirm variance across frames is non-zero and that two visibly
  different scenes are further apart than two frames of the same scene. A hook capturing a constant
  is the most likely bug and would silently produce "no signal".
- **Pooling.** Mean-pooling over tokens discards spatial structure. Record both mean and max pooled;
  if they disagree, say so rather than picking the better one after seeing results.
- **Reference sufficiency.** 98 successful reference episodes worked for S0. Higher-dimensional
  spaces need more reference data for the same nearest-neighbour reliability — **2048 dimensions with
  ~2,000 reference frames is sparse.** Mitigate by reporting results after PCA to 32/64 dims as well
  as raw, and treat a raw-2048 result that disagrees with the reduced one as unresolved.
- **The success rows must land at ~5%.** That is the calibration check, not a result.
- **GPU coordination.** The card is shared. Ping before starting, and check `pstate`/`clocks.sm`
  rather than utilisation — on battery this runs ~25× slower and would silently look like a hang.

---

## 7. What this does not do

- **Does not establish causation.** §4D gets ordering, which is necessary and not sufficient.
- **Does not transfer to a policy we cannot instrument.** S1–S4 all need model internals. For a
  black-box client policy only S0 and behavioural signals remain.
- **Does not measure whether any of this predicts *real-robot* failure.** That needs paired real/sim
  data we do not have (`docs/REAL_SIM_PAIRED_DATA.md`).
- **One suite.** `libero_spatial` only, because that is where both the reference and the LIBERO-Plus
  query already exist. Generalisation across suites is a follow-up, not a claim.
