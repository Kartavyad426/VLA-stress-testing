# Handoff — 2026-09-23

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

**Nothing of mine is running. GPU free at 106 MiB.** `implementor` announced it
was taking the card for R-037 (~70 episodes, ~25 min) — check the flock at
`/tmp/vla_gpu.lock` before starting anything.

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
- **action-atlas port** — **unowned and disputed.** I told `implementor` it was
  theirs; I had no standing to assign it, and they correctly declined. Only a
  user can assign it.
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
