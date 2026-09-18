# DESIGN — embedding feature capture for GR00T N1.7

*Written 2026-09-18 by `implementor`. Scope: **capture, verification and smoke test only.** The §4
A–D analysis of `docs/EXP_EMBEDDING_OOD.md` is NOT in this spec and currently has no owner.*

Implements the capture half of `docs/EXP_EMBEDDING_OOD.md`, including the four corrections `primary`
made in its §2.3.

---

## 0. Scope boundary, stated first because it was contested

`primary` scoped this session to capture **and** the §4 analysis. The user scoped it to capture,
verification and smoke test. **The user's scope governs this spec.**

| in scope | out of scope |
|---|---|
| tap framework + GR00T tap spec | `experiments/ood_embedding.py` (§4 A–D) |
| `.npz` sidecar format + manifest | running the 40-ep de-risk or the 3.5 h capture |
| CPU unit tests against a stand-in model | any MINERVA / π0 tap spec |
| a 2-episode real-model smoke, for `primary` to run | enabling RTC |

**The §4 analysis needs an owner and it is not this session.** Recorded here so the gap is visible
rather than assumed away.

---

## 1. Constraints inherited, not chosen

These come from `primary`'s handoff and from the code. They are listed because each one rules out an
implementation that would otherwise look reasonable.

1. **`third_party/lerobot/` must not be patched.** It is a vendored fork whose HEAD is recorded in
   every run's `code_state/`. Editing it retroactively invalidates the provenance of every previous
   run and makes this capture unreproducible. **Everything attaches from outside.**
2. **RTC is dead on this path**, verified end to end: `select_action` → `predict_action_chunk` with no
   kwargs → `prev_chunk_left_over=None` → `_prepare_n1_7_rtc_inputs` returns `options=None` → the
   branch at `groot_n1_7.py:665-681` never runs and `vel_strength` is identically 1. Do not add an RTC
   path; it would change the policy's behaviour and this is an observational run.
3. **`get_action` and `get_action_with_features` are plain methods, not `forward()`.** `nn.Module`
   forward hooks do not fire on them. Real hooks are reserved for genuine module calls.
4. **`process_backbone_output` overwrites its key in place** (`groot_n1_7.py:563-564`), so after that
   call `backbone_features` *is* S2. See §5.1.
5. **The adapter drops two of the three tensors.** `predict_action_chunk` does
   `actions = outputs.get("action_pred")` (`modeling_groot.py:502`) and discards
   `backbone_features` and `state_features`, then slices. Capture must intercept at
   `action_head.get_action`, above that loss.
6. **No `Step` or `Rollout` schema change.** Capture is per model forward (~8/episode), not per env
   step (~118), so a `Step` field would be null 93% of the time; and `rollouts.jsonl` is already
   130 MB for the MINERVA run.

---

## 2. Architecture

Three layers. Only the middle one knows what a GR00T is.

```
  vla_harness/capture/taps.py         policy-agnostic: roles, TapSpec, FeatureSink, pooling
            ^
  vla_harness/capture/groot_features.py   the ONLY GR00T-specific file: hooks + method wrap
            ^
  vla_harness/policies/lerobot_policy.py  `capture=` flag; attach at load, mark episode at reset
  vla_harness/runner.py                   3 lines: flush the sink once rollout_id exists
```

### 2.1 Why a role taxonomy rather than GR00T field names

§4's analyses never mention `vlln`. They contrast an **input-side** signal with a **behaviour** signal.
Nearly every VLA is VLM → adapter → action head, so the roles are shared even where the modules are
not. Naming the roles now means the eventual analysis code, and any second policy, target a stable
contract.

| role | GR00T N1.7 | doc's label | what §4 uses it for |
|---|---|---|---|
| `vl_encoder` | pre-`vlln` backbone output | S1 | appearance/language shift; onset timing |
| `vl_adapted` | `vl_embeds`, post self-attention | S2 | what the action head actually receives |
| `vl_normed` | post-`vlln`, pre-attention | S1.5 | free from the same module; kept because it costs nothing |
| `state_encoded` | `state_features` | S0e | the proprio analogue of the 6.9× S0 baseline |
| `action_pred` | `action_pred`, sliced to 7 | S3 | intended behaviour |
| `decode_path` | 4 Euler intermediates | S4t | the path to the action, not only its endpoint |
| `resample_spread` | spread over k draws | S4 | self-consistency |

**Deliberately not general:** the *numbers* are not comparable across policies — dims and
normalisation differ, and `EXP_EMBEDDING_OOD.md` §2.1 warns `state_encoded` is embodiment-specific and
must never be pooled across embodiments. This taxonomy makes the **code** reusable, not the
**results** commensurable. Only the GR00T spec is implemented here; the roles are not validated
against a second policy and may prove GR00T-shaped.

---

## 3. Tap mechanisms

| signal | role | mechanism | fires |
|---|---|---|---|
| S1 | `vl_encoder` | `register_forward_pre_hook` on `action_head.vlln` | 1× / forward |
| S1.5 | `vl_normed` | `register_forward_hook` on `action_head.vlln` | 1× / forward |
| S2 | `vl_adapted` | bound-method wrap of `action_head.get_action`, reading `backbone_features` from its return | 1× |
| S0e | `state_encoded` | same return, `state_features` | 1× |
| S3 | `action_pred` | same return, `action_pred`, **unsliced 40×132** | 1× |
| S4t | `decode_path` | `register_forward_hook` on `action_head.action_decoder` | **4× / forward** |
| S4 | `resample_spread` | k−1 extra `get_action_with_features` calls inside the wrap | k−1 |

`action_head.get_action` (`groot_n1_7.py:719`) is the correct intercept: it returns the BatchFeature
with all three tensors, unsliced, before `modeling_groot.py` truncates to `prediction_horizon` and
`original_action_dim`.

### 3.1 `actions₀` and S4: RNG save/restore

Both need the generator, and both must leave the rollout bit-identical to an uninstrumented run.

`actions₀` is a bare `torch.randn` inside `get_action_with_features` (`:657`) — no module call, so no
hook can see it. The wrap does:

1. save generator state → `S_pre`
2. call the original method (the model draws its own `actions₀`, untouched)
3. save generator state → `S_post`
4. restore `S_pre`; replay `torch.randn(shape, dtype, device)` → **exactly the `actions₀` the model
   used**
5. run the k−1 S4 resamples
6. restore `S_post`

The model's draw is never intercepted or reseeded, and the generator ends where the unwrapped run
left it. State is saved for CPU and, when the policy is on CUDA, for that device.

**Why `actions₀` is observed rather than derived, and this is the whole point.** With
`vel_strength ≡ 1`, the loop is exactly `action_pred = actions₀ + dt·Σᵢ pred_i[:, -40:]`. So `actions₀`
*could* be computed by subtracting the captured `pred`s from the returned `action_pred` — and doing
that would make §6's S4t gate **pass by construction**, testing arithmetic instead of testing whether
the hook is on the right module. An earlier draft of this design proposed exactly that subtraction.
It is wrong, and the gate is the reason.

**Residual risk, stated rather than hidden:** step 4 assumes the model's first RNG consumption inside
`get_action_with_features` is that `randn` and that nothing else draws before it. True on the read
code in eval/`inference_mode` with RTC off. §6's S4t reconstruction check is what detects it if that
assumption ever breaks — a wrong `actions₀` fails the reconstruction loudly.

---

## 4. Storage

**One `.npz` per episode** beside the trace, plus `capture_manifest.jsonl`.

Per-episode arrays, `F` = number of model forwards (~8):

| key | shape | dtype |
|---|---|---|
| `vl_encoder_mean`, `vl_encoder_max` | `F × 2048` | fp16 |
| `vl_normed_mean`, `vl_normed_max` | `F × 2048` | fp16 |
| `vl_adapted_mean`, `vl_adapted_max` | `F × 2048` | fp16 |
| `state_encoded` | `F × 1536` | fp16 |
| `action_pred` | `F × 40 × 7` | fp16 |
| `decode_path` | `F × 4 × 40 × 7` | fp16 |
| `actions_init` | `F × 40 × 7` | fp16 |
| `resample_std` | `F × 40 × 7` | fp16 |
| `env_step` | `F` | int32 |

Attributes: `rollout_id`, `policy_id`, `task_id`, `success`, `n_forwards`, `sliced_from=132`,
`live_dims=[0..6]`, `pool=("mean","max")`, `k_resamples`, `capture_version`.

**Pooling.** Mean **and** max are both stored, per `EXP_EMBEDDING_OOD.md` §6 — the decision is made
here, before any result exists, so that a later disagreement between them is reported rather than
resolved by picking the better one.

**Slicing.** Captured unsliced at 40×132, sliced to the 7 live dims **after** capture, with
`sliced_from` recorded. §6 asserts the unsliced shape at the tap, so the slice cannot silently become
the thing that was measured.

**Sizing** (from `primary`'s corrected figures, not the doc's light 25 MB): three embeddings alone
≈ 65 MB over 723 episodes; ≈ 600 MB unsliced with `action_pred` and `decode_path`, ≈ 100 MB sliced.
**Slice first.** No frames are stored, ever (harness invariant G2).

**Manifest** — one JSON line per episode: `rollout_id`, `run_id`, `path`, `n_forwards`, `success`,
`task_id`, and the `forward_index → env_step` mapping. Joins on `rollout_id`, completely.

---

## 5. The two traps this design exists to avoid

### 5.1 The S1/S2 trap

```python
backbone_features = self.vlln(backbone_output["backbone_features"])
backbone_output["backbone_features"] = self.vl_self_attention(backbone_features)  # :563-564
```

A hook reading the key **after** this captures S2 twice. The two copies would still look different —
`vlln` sits between the reads — so the bug produces plausible numbers and no error. **S1 is therefore
captured by a forward pre-hook on `vlln`, never by reading the dict.** §6's paired assertion is the
only thing that catches a regression here.

### 5.2 The tautological S4t check

Covered in §3.1. Derive `actions₀` by subtraction and the verification gate becomes arithmetic.

---

## 6. Verification

Two tiers, so the gates do not need a GPU in order to fail.

**Tier 1 — CPU unit tests** against a miniature stand-in `nn.Module` reproducing the real structure:
a `vlln`, a `vl_self_attention`, the **in-place key overwrite**, a 4-step Euler loop with an
`action_decoder`, and a `get_action` returning the three-key BatchFeature. Small dims, runs in
milliseconds.

| # | gate | assertion |
|---|---|---|
| V1 | **S1/S2 trap** | `vlln(S1) ≈ vl_normed` within tol **AND** `S1 != S2` |
| V2 | **S4t reconstruction** | `actions_init + dt·Σ decode_path` == returned `action_pred` within tol |
| V3 | **no behaviour change** | wrap returns the original object — `is`, not `==` |
| V4 | **RNG neutrality** | generator state after a wrapped call == state after an unwrapped call, same seed |
| V5 | **shapes/dtypes** | 2048 / 2048 / 1536; `action_pred` **40×132 at the tap**, 40×7 stored |
| V6 | **forward count** | one capture row per `get_action` call |
| V7 | **off is off** | with `capture=None`, zero hooks registered and the sink is never constructed |

**Tier 2 — 2-episode real-model smoke**, for `primary` to run on GPU. Re-asserts V1–V6 against the
real checkpoint, plus:

| # | gate | assertion |
|---|---|---|
| V8 | **forward-count join** | rows per episode == `Rollout.model_forwards` exactly |
| V9 | **non-degenerate** | per-signal variance across forwards > 0 |
| V10 | **scene separation** | two visibly different scenes farther apart than two forwards of one scene |
| V11 | **provenance** | `git status --porcelain third_party/` is empty |
| V12 | **conformance** | the existing conformance gate still passes |

V4 and V7 are mine, not `primary`'s: V4 is what makes "no behaviour change" checkable rather than
asserted, given §3.1 touches the generator; V7 is what keeps this off the critical path of every other
run in the repo.

**Tier 1 is what makes this buildable without the card. Tier 2 is what proves the stand-in was not
lying.** Neither substitutes for the other.

---

## 7. Deliverables

| file | status |
|---|---|
| `vla_harness/capture/taps.py` | new |
| `vla_harness/capture/groot_features.py` | new |
| `vla_harness/capture/__init__.py` | new |
| `tests/test_capture_taps.py` | new — V1–V7 |
| `experiments/capture_smoke.py` | new — V8–V12, one command for `primary` |
| `vla_harness/policies/lerobot_policy.py` | `capture=` arg; attach in `_load`, mark in `reset` |
| `vla_harness/runner.py` | 3 lines: flush once `rollout_id` exists |

### 7.1 The one shared-harness edit, and why order-based joining was rejected

The sink buffers an episode, but `rollout_id` does not exist until `rollout()` builds the `Rollout`
at the end (`runner.py:67`). So after it is built:

```python
if hasattr(policy, "flush_capture"):
    policy.flush_capture(r)
```

The alternative — join by episode order, touching nothing — fails because `run_cell`'s cache-skip
path means capture order does not track trace order across a resume, and resumable runs are
load-bearing in this project. A silent misjoin would attach one episode's embeddings to another
episode's label, which is exactly the class of error this experiment cannot survive. An explicit
flush is the honest version.

---

## 8. What this does not do

- **Does not run anything on the GPU.** `primary` holds the lock and runs both the 40-episode
  de-risk and the full capture.
- **Does not analyse.** No `experiments/ood_embedding.py`. §4 A–D is unowned.
- **Does not generalise beyond GR00T.** The role taxonomy is designed for reuse and is validated
  against exactly one policy.
- **Does not settle the `n_action_steps` confound** (GR00T 16, MINERVA 1). `primary` owns that. This
  capture records `model_forwards` and the resolved config so the capture is readable *after* it is
  settled, and assumes no resolution.
