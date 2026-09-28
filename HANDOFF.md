# Handoff — 2026-09-23 (updated late afternoon)

> **Since this morning:** R-037 and R-039 are done, R-041 is running, and R-040 and
> R-042 are pre-registered. The action-atlas repro is blocked. The harness now records
> video and every object's position. See **§0** first, then the rest.

## 0. Afternoon update

- **R-037 DONE.** A vision perturbation moves the backbone output (Qwen3-VL layer 16)
  by 1.37x, and after the 4-block VL self-attention it is 1.02x. The signal is gone by
  the time the action head reads it. The localisation is correlational. Nothing
  predicts which episodes fail (Q2: no result survives Bonferroni).
- **R-039 DONE (fable).** For every LIBERO-Plus category, robot initial state
  included, the perturbation reaches the action via the **image tokens**; the state
  token is near-inert. Report: `docs/R039_RESULTS.html`.
- **R-041 RUNNING (fable)**: boundary sweeps from a known success. **R-040, R-042**
  pre-registered.
- **Action-atlas repro (arXiv:2603.19233) is BLOCKED.** The user assigned it to
  `implementor`, and it would not load under its own pinned transformers (three
  separate failures). The paper's GR00T layer ordering is **unverified**. Details:
  `~/Documents/Code/action-atlas-repro/STATUS.md`. A bare `except` at
  `model_adapters.py:443` hides the real errors, so make it re-raise first.
- **The harness now records per-episode video by default**
  (`runs/<run>/video/<rollout_id>.mp4`, ~1–2 MB; `--no-video` to skip).
- **Wrong-object grasps were invisible in traces.** `_gt_object_pos` holds only
  BDDL task objects. On `5115b970e766` it read "nothing moved" while the arm had
  lifted the **ramekin** 6 cm. The env now also records `_gt_scene_object_pos`
  (every free-joint object). On the drawer scene, both failing initial states
  handled the wrong object. That has two explanations, grounding or start-pose
  coverage, and these episodes can't tell them apart.
- **`experiments/visualise_set.py`**: a reference plus any number of variants on one
  page, with synced videos, signals, every-object movement and stored activations
  (labelled by layer). `--rerender` rebuilds a page on the CPU.
- **VLM labelling (`experiments/vlm_label.py`) is unvalidated.**
  - Qwen3-VL-8B-Thinking runs locally at 4-bit (6.4 GB peak, 2 fps, half
    resolution); Nemotron-Omni runs via NIM.
  - Gemma-4 times out on NIM; Cosmos-Reason2-8B is gated (HF access pending).
  - On the first real failure, both working models detected the grasp and **named
    the wrong object**. Score VLM output against simulator truth before counting it.
  - FailBench (arXiv 2609.03611) finds robotics-specialised VLMs underperform their
    base models.
- **Per-episode record schema:** `docs/EPISODE_RECORD_SCHEMA.md`. Simulator fields
  are exact; the VLM supplies only `attempt.*`.


For whoever picks this up next, agent or human. `RESULTS.md` is the experiment
record (R-001..R-038), `PENDING_DECISIONS.md` holds what needs the user's call,
`docs/FRAME_LABEL_METHODOLOGY.md` holds the method argument. This file is the
state of the machine and what is worth doing next.

---

## 1. The headline: language is NOT inert, and we had it wrong

**R-038. GR00T with an EMPTY prompt scores 50/100 where the same stack with the
instruction scores 100/100.** The mean is not the finding; the spread is:

| null | scene | closest approach on failures | control |
|---|---|---|---|
| **0/10** | on the stove | **25.2 cm** | 4.4 cm |
| **0/10** | on the wooden cabinet | 19.1 cm | 5.1 cm |
| **0/10** | in the top drawer | 11.5 cm | 5.4 cm |
| 2/10 | next to the ramekin | 12.5 cm | 4.6 cm |
| 6/10 | on the ramekin | 10.2 cm | 5.2 cm |
| 7/10 | next to the plate | 6.5 cm | 4.7 cm |
| 8/10 | next to the cookie box | 6.4 cm | 4.6 cm |
| 8/10 | on the cookie box | 5.7 cm | 4.9 cm |
| 9/10 | from table center | 7.3 cm | 4.6 cm |
| **10/10** | between the plate and the ramekin | — | — |

The arm **does not approach the target** on the failing scenes — 11-25 cm where
the control closes to ~5 cm, and on the stove the bowl never moves in any of ten
episodes. Success and closest-approach correlate at r = -0.826, so it is one
mechanism varying in degree.

**This overturns two things we believed.** LIBERO-Plus Finding 3 ("largely
insensitive to language") does not hold in its strong form: insensitivity to
*rewording* is not insensitivity to *having an instruction*. And **R-025's null
is probably underpowered rather than correct** — 34 vs 33 of 42 at p=1.0 could
never have detected this.

**What is NOT established:** why some scenes need language and others do not. A
confusable-preposition-pair hypothesis was raised mid-run and killed by the data
(both wooden-cabinet scenes are 0/10; the cookie-box pair is 8/10 and 8/10). The
mechanism is open and it is the most interesting question on the board.

---

## 2. What the policy actually receives

Worth knowing before designing anything, because it is less than people assume:

| | |
|---|---|
| `observation.images.image` | agent view RGB |
| `observation.images.image2` -> `wrist_image` | wrist RGB |
| `observation.state` | **8 numbers** — eef pos (3), axis-angle (3), gripper qpos (2) |
| `task` | instruction string, tokenised |

No object list, no goal spec, no BDDL, no task id. The BDDL defines the goal for
the **simulator**, which checks success; the policy never sees it. Privileged
truth (`_gt_object_pos`, `_gt_eef_to_object`, `_gt_n_contacts`) is recorded into
our traces for analysis and is **not** in the batch.

---

## 3. Machine state

**`fable` holds the GPU for R-041** (camera-yaw axis, then an R-039 extension and
three more axes, each a separate flock acquisition, each announced). Check the flock
at `/tmp/vla_gpu.lock` before starting anything. An 8B VLM at 4-bit needs about
6.4 GB and cannot share the card with GR00T.

**Announce BEFORE launching, then wait for an ack.** We had a near-miss: both
sessions announced *while* starting. The mutex held and the second job queued
correctly, but only by luck of timing.

**Result numbers collide too.** Two sessions pre-registered an `R-037` minutes
apart in the same working tree. Rule now: `grep '^## R-0' RESULTS.md` on HEAD
before claiming a number, and **push the pre-registration immediately** rather
than holding it locally. First commit keeps the number.

---

## 4. Policy roster

| Policy | State |
|---|---|
| **GR00T N1.7** | working, the workhorse. bf16, nas=16, obs 360 |
| **MINERVA** | working as fp32 control. **bf16 is broken** (R-033) |
| SmolVLA | **vetoed** — never reproduced its published number |
| π0.5 | **does not fit.** 7.40 of 7.53 GiB, twice, on a verified-idle card |
| π0-FAST | **does not fit.** Four configurations tried (R-032) |

π0-FAST needed three fixes just to get far enough to prove it does not fit
(R-034). Both π0 models are rented-GPU candidates; **PENDING #25 is now a budget
question, not a technical one.**

---

## 5. The embedding thread: a negative, correctly scoped

**R-036 is a NEGATIVE and must not be read as "vision carries nothing".** The
40-episode de-risk had **n=2 camera-viewpoint episodes** and was dominated by
Robot Initial States (12) and Objects Layout (11) — the two categories we have
independent reason to believe are *not* VL-pathway phenomena. It tested the VL
pathway on failures we already believed were not VL failures. The gate was
**neither met nor failed; it was never tested.**

Three reasons the VL rows are weak evidence:
1. **Sample composition** (above) — the only one a capture change cannot fix.
2. **Spatial pooling.** `taps.py` pools over the token axis, and object position
   in a ViT-style encoder lives in *which* tokens are active.
3. **Modality mixing.** The pool averages image AND text tokens together, and
   instruction length ran 14-23 words, so the mixture ratio drifts per episode.

`state_encoded` separated at 12.95x — but **broadly across every category**
(60.4% robot-init, 52.8% noise, 52.8% light, 34.7% layout, 27.8% camera), which
is what "this episode ended up somewhere unusual because it went wrong" looks
like, not a mechanism. My prediction that it would concentrate in robot-init and
layout was **wrong**.

---

## 6. The best unexploited finding

**N1.7's DiT separates the visual and language pathways by block index**
(`cross_attention_dit.py:357-380`, over 32 blocks):

- all 16 **odd** blocks: `encoder_hidden_states=None` — nothing enters
- `idx % 4 == 0`: **text** (0, 4, 8, ... 28)
- even, `idx % 4 == 2`: **image** (2, 6, 10, ... 30)

They are distinguishable **by index alone** — no intervention, no matched pairs,
no patching, no seed control, no token alignment. That makes the upstream-vs-
action-head question answerable by **observation**, and it is the cheapest
high-value thing in the document (`FRAME_LABEL_METHODOLOGY.md` §4.6).

`AlternateVLDiT.forward` already accumulates `all_hidden_states` unconditionally
and `return_all_hidden_states` only controls the *return*, so wrapping
`action_head.model.forward` from outside costs **zero extra compute** and needs
no `third_party` edit.

**`attend_text_every_n_blocks: 2` is a lying config name** — the counter runs
over even blocks only, so text is every FOURTH block. Three of us inferred period
2 from the name independently and I committed it to git. **Read indices off the
forward pass, never off config names.**

---

## 7. Open, needs the user

- **R-035** — balanced radius sweep, pre-registered at `1d638dc`, ~2 h GPU, never
  run. Separates two **opposite** post-training data prescriptions: saturation
  says corrective data must cover the whole ball, monotone decline says
  near-neutral is worth most.
- **Category-balanced de-risk** — ~40 episodes weighted to camera viewpoint.
  Tests what R-036 could not.
- **action-atlas port**: assigned by the user to `implementor`, now **BLOCKED**
  (see §0). The next step is an older transformers, not more model-code patches.
- **Per-block DiT capture (§6)**: still not built. Its pilot should target R-038's
  two arms at the first action, where the images are identical and only the prompt
  differs. Index trap: `all_hidden_states[i+1]` is block *i*'s output. Score the
  per-block delta, and fix the Euler step before ordering by block.
- **VLM calibration**: about 40 human-labelled episodes are needed, and the
  adjudication CSV is the place for them. Then compare video-only, sim-facts-only and
  both, scored on labels that are not in the prompt.
- **PENDING #25** (fourth policy / rent a GPU), #20, #15, #13, #6, #2, #21, #8.
- **The adjudication CSV is still 0 of 80 verdicts.** It has been the open job
  for five days.

---

## 8. Advice, each earned this session

**1. A result that agrees with your prior is when you are least likely to check
it.** Four of my measurement errors this week share that shape: reading object
poses from a field written at episode *end* (167 mm of movement that was actually
0.00 mm); measuring GPU with `--query-compute-apps`, which lists only CUDA
processes and reports 0 MiB for an EGL context; reading a CUDA OOM's "tried to
allocate 64 MiB" as the remaining deficit rather than the next attempt; and
generalising "the arm never reaches the object" from one scene family to a whole
perturbation.

**2. Controls catch bad statistics. Only reading the source catches bad
structure.** Three claims were overturned this week by reading the forward pass —
the DiT's period, the modality mixing, the token-count variance. **None would
have failed a CPU stand-in test.**

**3. A gate that can pass by being skipped is worse than no gate.** Twice on one
workstream: `model_forwards` reads 0 on every pre-existing trace, so a gate
asserting equality against it would compare N rows to 0 or silently skip; and
V1-V6 were never invoked against the real model at all. Assert that every
registered gate actually executed.

**4. Two numbers that describe the same episodes must agree.** That invariant
caught a 4.9e7x separation that would have been published — successes scored
against a reference cloud containing their own points. The trap sits on the path
a *correct* experiment takes, because a same-run reference is **required** when
the policy is non-deterministic (unseeded `randn` at `groot_n1_7.py:657`).

**5. Prefer what the code consumes over what it declares.** π0-FAST needed three
fixes in a row, all this shape: a tokenizer repo missing the subfolder its own
processor class requires; a setting declared in both the policy config and the
shipped preprocessor, read by different code; and our adapter silently
overwriting a checkpoint's shipped rename map with `{}`.

**6. Check blast radius before fixing a shared component.** The rename-map fix
could have invalidated every past run. Of the checkpoints in use, only π0-FAST
ships a non-empty map — so nothing was affected. **That check mattered more than
the fix.**

**7. Register the expectation before the run, and record the miss.** R-038's
"near 100%" was wrong by 50 points. It is in the record as a miss because it was
written down first. Two sessions this week also pre-registered *traps* — outcomes
that would prove nothing — and in both cases that was the most useful line in
the entry.

**8. Say which claim you are withdrawing.** Four retractions this session, three
of them mine: the confusable-pair hypothesis, the "option zero sharpens existing
data" claim, and assigning work to another session. Peers retracted their own
too. The project is better for each one being stated rather than quietly dropped.

---

## 9. Handoff from `primary` only, 2026-09-24

This section covers **only the session named `primary`** (successor to the session
that ran R-038). Other sessions (`fable`, `implementor`, `vla-f9`, `SAE failure`,
`Sentinel analysis`) keep their own records; nothing here speaks for them.

### What I built (all committed and pushed, `a50e533`)

| What | Where | State |
|---|---|---|
| Per-episode video, on by default | `vla_harness/video.py`, `runner.py`, `experiments/harness_eval.py` (`--no-video`) | tested (`tests/test_video.py`) |
| Every-object tracking | `_gt_scene_object_pos` in `vla_harness/envs/libero_env.py` | tested (`tests/test_scene_objects.py`) and checked on a real replay |
| Reference-vs-variants page | `experiments/visualise_set.py` (`--rerender PAGE` works CPU-only) | working |
| VLM labeller | `experiments/vlm_label.py`, isolated venv `.venvs/cosmos` | working, **unvalidated** |
| Doc refresh + README reading order | README, RESULTS current-state/index, this file, living docs, banners on historical docs | done 2026-09-23 |

Two of these reached git inside **other sessions' commits**: implementor's
`19eec86` took video/runner, fable's `9b98523` took the scene-object change and
fable's `1a735c2` took my RESULTS.md index edit. Sessions commit whole files.
Before assuming your edit is uncommitted, check `git log -- <file>`.

### What I found

1. **Wrong-object grasps were invisible.** `_gt_object_pos` holds only BDDL task
   objects. On `lplus_fail_groot:5115b970e766` it read "nothing moved" while the arm
   lifted the **ramekin** 6 cm, which a replay confirmed. On the drawer scene, both
   failing initial states (475, 395) handled the wrong object, and 405 disturbed both
   bowls. `fable` correctly pointed out that two episodes cannot separate
   grounding failure from start-pose coverage.
2. **VLMs spotted the grasp and named the wrong object.** Nemotron-Omni (NIM) and
   Qwen3-VL-8B-Thinking (local, 4-bit) both said "picked up the black bowl". I
   first called that a hallucinated grasp. **Retracted:** the grasp was real, only
   the object was wrong. FailBench (arXiv 2609.03611) finds robotics-tuned VLMs
   underperform their base models; Qwen3-VL-2B scores 0.53, near chance.
3. **Activations for the drawer scene** (`viz/set_drawer_bowl_activations.html`,
   local). The backbone output separates the 475 failure from t=0. The action-head
   input overlaps the reference, consistent with R-037. This is one episode per
   condition, so it is a picture, not a result.

### Pending: mine, none started

| Job | GPU | Next step |
|---|---|---|
| Exact activations for the four drawer episodes on `viz/set_drawer_bowl_initstates.html` | ~6 GB, ~15 min | replay stored actions and call GR00T on each recorded observation. Backbone/adapter features are exact; action samples are fresh draws. Not built. |
| Per-block DiT capture (§6) | CPU build, then ~20 min pilot | wrap `head.model.forward` and force `return_all_hidden_states=True`. The inference path (`groot_n1_7.py:694`) does not pass it. Return only element 0, because `:707` wants a bare tensor. See the traps below. |
| VLM head-to-head (video-only / sim-facts-only / both; Qwen-8B vs Nemotron vs Gemma) | light | **blocked on ~40 human labels** (the adjudication CSV, still 0/80) |
| Sim-derived fields of `docs/EPISODE_RECORD_SCHEMA.md` | CPU | unowned; `handled.*` and `anchor.grasp_step` definitions are mine |

**Per-block DiT traps**, each one verified at source by a peer or by me:
- `all_hidden_states[i+1]` is block *i*'s output, because index 0 is the input
  `sa_embs`.
- Score the delta `[i+1]−[i]`, not the level.
- `[32]` is taken before `norm_out`, so it is not `model_output`.
- Hidden states are **state+action tokens (41 × 1536)**, with no image/text spans;
  the image and text pathways are separated only by block index (`idx%4==0` text,
  `==2` image, odd blocks get nothing).
- The 33 states come ×4 Euler steps per call. Fix the step before ordering by
  block.
- Storage is ~17 MB per call unpooled (~120 GB for a full capture); pool over the
  41 tokens.
- For R-038, compare at the **first action**, where images and state are identical
  and only the prompt differs. The first difference showing up at block 0 is
  guaranteed and means nothing.

### Loose ends

- `docs/R042_WRIST_MATHS.pdf`: made from fable's HTML with headless Chrome,
  **uncommitted**, and the user hasn't decided. It goes stale if the HTML changes.
  To regenerate: `google-chrome --headless=new --no-pdf-header-footer
  --print-to-pdf=docs/R042_WRIST_MATHS.pdf file://$PWD/docs/R042_WRIST_MATHS.html`.
- `viz/set_*.html` (10–11 MB) are deliberately local.
- RESULTS.md R-041's header says RUNNING. R-041 has since run to completion;
  fable owns the entry.
- **27 local commits are unpushed on `main`, none of them mine.** They belong to
  other sessions; push only when the user says.

### Practical notes

- **VLM environment:** `.venvs/cosmos` (torch 2.11 cu128, transformers 5.5.4,
  bitsandbytes). Don't install `torchcodec`: the PyPI wheel wants CUDA 13.
  `vlm_label.py` decodes video with PyAV itself.
- **Qwen3-VL-8B at 4-bit** OOMs at native resolution × 56 frames. Use
  `--fps 2 --scale 0.5` (peak 6.36 GiB, ~95 s per episode).
- **NIM key:** `NVIDIA_NIM_API_KEY` lives in
  `~/Documents/Code/robotics_agentic/.env`. `NVIDIA_API_KEY` in `~/.bashrc` is an
  unfilled template.
  - Cosmos-Reason2-8B returns 404 on NIM for this account, and is gated on HF
    (access not yet requested).
  - Gemma-4-31B times out on NIM (504).
- **CPU rendering** when the GPU is busy: `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`.
  It is slower, but replay and `visualise_set.py` both work.
- **`du` on the HF cache** reports ~12 MB unless you pass `-L`, because blobs are
  symlinks.

### Advice, earned this session

1. **Check your ground truth before a model's answer.** I scored two VLMs
   against a trace field that could not see distractors, and declared their correct
   grasp detection a hallucination. When two independent models disagree with
   your label, look at the frames first.
2. **"Nothing moved" is only as good as the list of things tracked.**
3. **Sessions commit whole files.** Your edits can ship inside someone else's
   commit, and theirs inside yours. Check `git log -- <file>`, not `git status`
   alone, and name which session owns each part in the commit message.
4. **An analysis threshold changed after the first data came back is still a
   post-hoc change,** even when it is recorded before the main analysis runs.
   Say so when reading R-041.

## 10. Handoff from `primary`, 2026-09-28

This section covers **only the session named `primary`**, which now hands its role
to `my prim`. The context was cleared (`/clear`) after §9. Everything since then
was one read-only review, so this section has no memory of the work in §9 beyond
what §9 itself says. Other sessions (`fable`, `vla-a5`, …) keep their own records.

### Role and scope

`primary` is the user's general-purpose session for this workspace. It answers
design questions against the code, reviews papers for relevance to the
perturbation/transfer-fraction programme, and owns the items listed in §9. It
does not own RESULTS.md entries (R-039 onward belong to `fable`), the run queues,
or the roster decisions recorded in memory (GR00T N1.7 first, gated on MINERVA and
a verified harness; reproducibility is the bar).

### What I did since §9

Nothing was committed, no runs were started, and no repo files changed apart from
this section.

**Golden Ticket review (arXiv:2603.15757, Patil et al., RAI, v3 2026-06-19).**
The user asked whether fixing the denoising noise, as that paper does, would help
the transfer-fraction measurements.
- *The paper:* it searches (random search / zeroth-order / CEM, ~50–5000
  candidates) for **one constant initial noise vector**. That same `w` is used at
  every action step, in every episode and state, in place of N(0, I). It improves
  a frozen policy on 46/51 tasks: SmolVLA on LIBERO +8–13 pts, GR00T N1.5 on
  SimplerEnv WidowX ~+20 pts. It is a policy-improvement method and says nothing
  about measurement.
- *Finding: we already use common random numbers.* `experiments/r039_run.py:194-208`
  sets `noise_key = 1000*noise_seed + forward`. The noise is shared across the
  five arms (P N T I S) **and** between the nominal/control rollout and the
  perturbed one (comment at `:207`). `--noise-seed` varies it while env seed 0
  fixes the layout. Per-forward keys work as well as one constant `w` for the
  P−N comparison.
- *Conclusion: no protocol change.* The paper argues **for** our noise-seed
  replicates (R-044, R-048, R-051). On its hardware block-pick task, noise choice
  alone ranges from 2/50 to 98% success. So an outcome under one fixed noise
  describes a noise-conditioned policy, and measuring under a *searched* ticket
  would bias every row toward nominal-tuned behaviour. It also cannot touch our
  30–40% outcome flip rate, which appears at identical seed and noise (render
  jitter).

### Awaiting the user (offered, not done)

1. One sentence in the "Determinism" bullet of `experiments/compile_map.py:336`
   citing arXiv:2603.15757 as the reason outcomes are replicated across noise
   seeds rather than trusted at one.
2. A pending item for the ticket-robustness experiment below (in §7 or the
   queue list, wherever the user prefers).

### Proposed, not pre-registered: ticket robustness

*Question:* a golden ticket searched on **nominal** LIBERO raises success there.
Does it hold its gain on the LIBERO-plus perturbation rows, or lose it? And does
it change the transfer fractions?
- The splice harness already supports this: inject the ticket as the noise for
  every forward, for both arms and control.
- **Gated:** GR00T N1.7 only, after it clears the reproduction gate (MINERVA
  first). Only GR00T N1.5 (SimplerEnv) shows noise-steerability in the paper; its
  LIBERO results are SmolVLA, which is off the roster.
- Needs an R-number, pre-registered expectations, and a held-out split, so that
  the ticket is not tuned on the evaluation scenes.

### Open questions

- How steerable through the noise are our policies (π0, GR00T N1.7)? The spread
  of outcomes across noise seeds in R-051 gives a lower bound. If it is large,
  single-noise-seed outcome claims in older entries need a caveat.

### Loose ends

- §9's loose ends are unchanged as far as I know. I did not re-check them:
  `docs/R042_WRIST_MATHS.pdf` uncommitted and undecided; unpushed commits on
  `main`, none of them mine.

### Advice

1. **Before adopting a paper's trick, check whether the harness already does
   it.** Here the answer was in a docstring (`r039_run.py:19`).
2. **Separate a method that improves a policy from one that measures it.** A
   noise chosen to maximise success is a bias, not a control.

## 11. Handoff from `vla-a5` (Sentinel / OOD), 2026-09-28

This section covers **only the session named `vla-a5`**. It was a read-and-discuss
session on applying Sentinel (Agia et al., CoRL 2024, arXiv:2410.04640) to our
work. It ran one CPU-only analysis and made no GPU runs, commits or edits outside
this section and one new untracked script. The role now passes to `my prim`.

### Scope

Read `docs/SENTINEL_METHODOLOGY.html` (written by the `Sentinel analysis` session).
Gathered context from `primary` and `fable`. Tested the user's idea (below) on
stored data.

### What I established

| Claim | Evidence |
|---|---|
| Sentinel's portable parts: STAC (distance between consecutive chunks' predictions for the same future steps, MMD V-statistic, gripper dropped) and conformal calibration on the terminal cumulative score of successes (any-time FPR ≤ δ, since the running sum only rises) | `SENTINEL_METHODOLOGY.html` §3–§5 |
| **The doc is wrong about chunk overlap.** It says k = h = 16, so no overlap. GR00T emits **40-step chunks every 16 steps**, so consecutive chunks share **24 steps**. RTC off only means the leftover steps aren't fed back in; they are still predicted. STAC needs no shadow inference. | `compile_map.py:308`, `EXP_EMBEDDING_OOD.md:83`, R-041 npz `act_*` shape (F, 40, 7), `env_steps` diffs all 16. **Doc not yet corrected.** |
| `ood_selfref` scores **5 proprio dims** (eef xyz + 2 fingers), not VL features. Scoring OOD as a fraction of steps was **never a deliberate choice against the length confound**; nothing records one. | `experiments/ood_selfref.py` header, `RESULTS.md:2130`, `git log -- experiments/ood_selfref.py` (2 commits) |
| The S0/S0e/S1–S4 tap panel is **proposed only**; its sole definition is in the Sentinel doc. No Sentinel element has been adopted or ruled out. | `fable` and `primary`, 2026-09-25 |
| R-047 and R-051 use none of this (readouts: transfer fraction, ‖P−N‖, success/closest approach). | `fable` |

### The user's hypothesis

STAC measures how the action distribution moves over **time**. We measure how it
moves under **perturbation**. But `‖P−N‖` (`r047_run.py:149`, `r041_run.py:141`)
compares **one draw per side under a shared noise seed** (`noise_key = 1000·seed + i`,
`r041_run.py:114`). That can't see a change in spread and gets inflated by mode
flips. A distributional comparison might explain R-047 pre-registered expectation 5:
`‖P−N‖` at forward 0 does not predict success (`RESULTS.md` R-047, ~:3805).

Proposed readout, a 2×2 of distances at matched forward index:

| | same forward | across forwards (24-step overlap) |
|---|---|---|
| nominal arm N | — | STAC_N (baseline self-consistency) |
| perturbed arm P | MMD(P_t, N_t): how far the perturbation moved the policy | STAC_P; excess = STAC_P − STAC_N |

Reading it: not shifted = absorbed. Shifted but consistent = confidently doing
something else (Sentinel's smooth-but-wrong category). Shifted and inconsistent
= erratic.

### (4a) DONE, CPU pass on stored R-041 chunks: single-draw STAC is noise

Script: `experiments/stac_r041_single_draw.py` (untracked). It uses the overlap
`chunk_f[16:40]` vs `chunk_{f+1}[0:24]`, dims 0–5, forwards 0–2 (matched index),
and a reference = p95 of magnitude-0 successes. Data: 240 rollouts, **11 failures**
(dist 2, yaw 4, joint 5, light 0).

- **The single-draw STAC floor is about 7, against `‖P−N‖` of about 1.** Distance
  divided by the norm of the overlapping slice is about 1.16. Consecutive forwards
  use different noise draws, so this is roughly the distance between two
  independent samples. It measures sampling spread, not self-consistency.
- Raw STAC_P AUROC for failure is 0.18–0.54, the **wrong direction**. Normalised by
  action size it is 0.43, chance. The reversal is action size: failures move
  slightly less.
- `‖P−N‖` f0 AUROC is 0.53–0.67, below magnitude alone (0.69–0.86).
- 2×2 at m > 0: all 11 failures are "shifted and consistent", but so are 168
  successes. No separation.
- **What it does show:** under a *different* noise seed the policy's own spread
  (about 7) is about 7× the perturbation effect under the *same* seed (about 1).
  So a perturbation may mostly move the policy *within* its own sampling spread.
  That is exactly the question (4b) answers, and it makes (4b) better motivated,
  not less. As Sentinel predicts, the single-sample version (its "Temporal
  Non-Distributional" baseline) doesn't work.
- Caveat: 11 failures, all highly confounded with magnitude. A null here is not a
  null for the distributional version.

### (4b) NOT RUN, distributional version (needs GPU and pre-registration)

- Per forward, draw **B ≥ 32** chunks per arm with the **same B noise seeds for P
  and N** (common random numbers keep the pairing's low floor). Extend the splice
  in `r041_run.py`/`r047_run.py`. The rollout still executes one draw, and the
  B draws are side calls.
- Compute MMD (RBF, **fixed bandwidth across arms and forwards**, not Sentinel's
  per-step median) for MMD(P_t, N_t), STAC_P and STAC_N on the 24-step overlap,
  gripper dropped. Port `compute_mmd_rbf` as a biased V-statistic, which keeps it
  ≥ 0.
- Calibrate at a matched forward index on magnitude-0 successes with the exact
  conformal index ⌈(M+1)(1−δ)⌉/M. Don't use cumulative sums: failures run 18
  forwards and successes about 5–8.
- **Must be a new pre-registered R-number. Do not retrofit onto R-047's readouts.**
  Changing an analysis after first data is post-hoc under project rules.
- Cost: B extra head calls per forward per arm. Measure it on a smoke run before
  budgeting. GPU was held by `fable`'s `queue_r051.sh` (R-051, then the R-047
  remainder).
- Kill switches (Sentinel doc §12): success-vs-success split, shuffled labels,
  checking that each threshold fires at δ on the reference.

### Open questions

- Does the perturbation shift the distribution beyond its own spread? Answering
  it is (4b)'s first readout.
- R-036's state-pathway separation is "the same rate in every category" (the
  `ood_selfref` header), which fits "unusual because it failed". Sentinel's
  novelty × outcome 2×2 per category, and onset timing, are the discriminators.
  Neither has been done.

### Advice

1. **Check chunk geometry against stored arrays, not prose.** The Sentinel doc's
   main blocker (no overlap) was false, and one `np.load` shows it.
2. **A single-draw distance across different noise seeds measures the sampler.**
   Anything across forwards must either share seeds or use B draws.
3. The Sentinel doc's §12 step 1 (Mahalanobis in `ood_selfref`) targets proprio
   state, not VL features. `primary` read it as VL.
