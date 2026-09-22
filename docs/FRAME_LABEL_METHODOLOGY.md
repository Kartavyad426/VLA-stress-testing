# Frame-level analysis without frame-level labels

*Written 2026-09-21 by `primary`. Proposal, not a result. §8 is reserved for
`res` and is not mine to fill.*

Companion to `EXP_EMBEDDING_OOD.md` (what we capture) and
`COVERAGE_GAP_METHOD.md` (what we prescribe). This doc is about the step
between: **given per-forward embeddings and only per-episode labels, what can we
legitimately conclude?**

---

## 1. The problem, stated precisely

The embedding capture produces roughly 8 vectors per episode (one per model
forward, not per env step — GR00T executes 16 actions per forward). We want to
know which of those forwards are anomalous, and in which signal.

We do not have a label for any of them. We have a **success bit per episode**.

The obvious move — assign the episode's label to each of its forwards — is
wrong, and wrong in a specific, predictable direction. A failed episode is
normal for its first stretch: the arm reaches, the gripper closes, and only then
does nothing further happen. Those early forwards are behaviourally
indistinguishable from a success, and labelling them `fail` injects noise into
one class only. The `success` class is comparatively cleaner but not clean — a
successful episode may contain a recovered-from moment that is genuinely
anomalous and is labelled normal.

**This is not a gap in our data collection. It is a category error.** A frame is
not a failure. Failure is a property of a *trajectory* — of a temporal process
that did not reach its goal. Asking "is this frame a failure" is close to asking
"is this pixel a video".

**Reformulation is one route, not the only one — corrected after `res`'s survey.**
FailureSpot agrees with the diagnosis above and rejects this conclusion: it
treats the *supervision* as malformed rather than the question, and repairs it
with action-derived weak labels plus a small active-learning annotation budget.
Timestamp-level failure localisation is therefore an established goal, not an
ill-posed one. §4 below is the reformulation route; a weak-supervision route
exists beside it and we have not costed it.

### 1.1 The reformulations that are well posed

| malformed | well posed |
|---|---|
| is this forward a failure? | when did this episode leave the reference manifold? |
| which frames are anomalous? | how far from the manifold did it get, and for how long? |
| is the VLM failing? | **at a given forward, which signals are outside the manifold and which are inside?** |

The third row is the one that matters for `res`'s remit, and §5 argues it needs
no episode label at all.

---

## 2. What the literature says

Our problem is not ours alone; it is stated almost verbatim in the recent VLA
failure-detection work.

**The label-noise problem is named.** *FailureSpot*
([arXiv:2609.04277](https://arxiv.org/abs/2609.04277)) — "existing proactive
methods are often supervised with trajectory-level labels, causing normal
pre-failure behavior in unsuccessful trajectories to be incorrectly labeled as
failure. This supervision mismatch introduces label noise and limits both
trajectory-level detection accuracy and precise timestamp-level failure
localization." That is exactly §1. Their answer is action-derived weak
supervision (inconsistent consecutive chunks, frozen/idle actions, aggressive
random motion) plus active learning to spend a small annotation budget on the
most uncertain trajectories. **[R]** — read from the abstract/summary, not the
full paper.

**Reference-from-successes is the established pattern.** *Sentinel*
([Agia et al.](https://proceedings.mlr.press/v270/agia25a.html),
[arXiv:2410.04640](https://arxiv.org/pdf/2410.04640)) requires "only a set of
successful policy rollouts and a description of the task", splitting detection
into erratic failures (STAC — statistical temporal action consistency, comparing
action distributions of *overlapping consecutive chunks*) and task-progression
failures (VLM video QA). *FIPER* predicts failures **without failure data** by
combining random network distillation on observation embeddings with
action-chunk entropy over multiple samples. Both avoid the labelling problem by
never needing a negative class.

**Others in the same space:** *SAFE*
([arXiv:2506.09937](https://arxiv.org/html/2506.09937v2)) on multitask failure
detection; *ActProbe* ([arXiv:2606.08508](https://arxiv.org/pdf/2606.08508)) on
action-space probing for early detection; perturbation-based epistemic
uncertainty ([arXiv:2606.20754](https://arxiv.org/html/2606.20754)).

**For the mechanism question, the relevant tradition is activation patching**
(causal tracing / interchange interventions —
[best-practices survey, ICLR 2024](https://arxiv.org/pdf/2309.16042)): run the
model on a clean input caching activations, run it on a corrupted input, then
re-run the corrupted input with one component's activation replaced by its clean
cache. If behaviour is repaired, that component is causally implicated. Work on
VLMs uses it to localise where text-vision integration fails, and carries a
warning we would have walked into: **patching only the last token does not work
for VLMs**, because visual information is distributed across many image tokens
rather than concentrated at one position
([causal tracing in LVLMs](https://arxiv.org/html/2511.05923v3)).

### 2.1 What does not transfer

**STAC's core signal is unavailable to us.** It compares *overlapping*
consecutive action chunks. Our eval runs with RTC off — verified end to end,
`groot_n1_7.py:665-681` is dead code on our path and `vel_strength` is
identically 1 — so consecutive chunks do not overlap. We have only a resampling
proxy (S4: call the head k times from the same conditioning). That is a
different quantity: **spread across draws at one moment, not disagreement
between successive commitments.** It should not be reported as if it were STAC.

**We are better off than the runtime-monitoring literature in one respect.**
Those methods are built for deployment, where the future is unknown and only
past observations are available. We are doing *post-hoc forensics on stored
traces*, with privileged simulator truth (§4.3) and matched controls (§4.2) that
no runtime monitor can have. We should not inherit their constraints along with
their methods.

---

## 3. The discipline

**Score per forward, aggregate per episode, test at the level where a label was
actually measured.** Never assert a label at a level you did not measure one.

`experiments/ood_selfref.py` already does this correctly: `:88` computes a
per-episode OOD *fraction*, and `:112` runs Mann-Whitney over per-episode rows.
The 6.9x separation is an episode-level claim built from per-frame scores, which
is the right shape.

**Preserve the effective n.** Forwards within an episode are heavily
autocorrelated — same scene, same task, adjacent in time. The effective sample
size is ~723 episodes, not ~5,784 forwards. A statistic computed over forwards
as if they were independent is pseudo-replication, and its p-value is
meaningless however small it looks. This is easy to do by accident when the
array is already flat.

---

## 4. Methods that need no frame labels

Ordered by what they cost.

### 4.1 Reference from successes only

Build the manifold from successful episodes — the cleanest available set — and
score everything against it, leave-one-out for the reference itself. No negative
class is ever required. This is `ood_selfref`'s design and it extends unchanged
to the embedding taps.

**Caveat inherited from that script's own header:** direction of causation is
not established. A failing episode ends up in unusual states *because* it is
failing, so "these embeddings are unusual" and "these embeddings caused the
failure" are not the same claim. §4.5 is the only method here that separates
them.

### 4.2 Matched-pair differencing — our strongest unused asset

We have canonical and perturbed episodes of the **same scene**, differing in
exactly one parameter. R-030 established this holds to 0.00 mm on object poses
at t=0, against ~21 mm between two different control rollouts of the same scene,
so the zero is a real measurement rather than a broken comparison.

So: compare forward *t* of a perturbed episode against forward *t* of **its own
canonical counterpart**. This is a paired difference. Pass/fail never enters.

`runs/groot_control_lplus_stack` (100/100) supplies a canonical episode for each
of the ten `libero_spatial` base scenes at `initstate 0`.

**Bounded validity, stated up front.** Once trajectories diverge, forward *t* in
one run and forward *t* in the other describe different situations, and the
difference stops being interpretable. It is valid early and degrades late —
which is acceptable, because onset is early by definition. Report the divergence
point alongside any paired statistic; do not average over the whole episode and
call it a distance.

### 4.3 Privileged simulator truth as genuine frame labels

Every step carries `_gt_object_pos`, `_gt_eef_to_object`, `_gt_n_contacts`,
`joint_pos`, `eef_pos`/`eef_quat`, `gripper_qpos`, plus the phase segmentation.

These are **real per-frame facts, not inherited labels**: is the gripper
approaching the target, is there contact, has the object left its start pose,
which phase is this. They convert the weak-label problem into a supervised one
on a different axis — you cannot ask "is this frame a failure", but you can ask
"does this embedding predict that contact is about to occur" or "does it encode
distance-to-target at all".

This is the most underused thing we have, and it is the reason our traces are
worth more than `lerobot-eval`'s success bit and mp4.

**It also gives a linear-probe programme.** Train a probe from each tapped
signal to each privileged quantity, on successful episodes, and ask what each
representation *contains*. If S2 (what the DiT receives) does not linearly
encode distance-to-target under a robot-initial-state perturbation, the action
head was never given the information. If it does, the action head had it and
failed anyway. That is a mechanism claim from correlational data, and it is much
cheaper than §4.5.

### 4.4 Onset and cross-signal ordering

Replace classification with changepoint detection: per episode, per signal, the
forward index at which it departs the reference manifold. No labels needed, and
strictly more informative than a per-frame verdict.

Then the payoff: **the ordering of departure across taps within one episode.**
If S1/S2 (VLM side) depart at forward 1 while S3/S4 (action side) depart at
forward 5, the perturbation entered through the visual pathway. If S1/S2 stay
inside while S3/S4 leave, it did not.

**Hazard, raised by `res` and not previously in this doc.** This argument has no
published precedent, and layer-wise probing studies find decodability peaks at
*intermediate* layers. So **tap depth alone changes when a signal becomes
visible**, independent of when anything went wrong: an ordering could be
measuring the architecture rather than the failure. Mandatory negative control —
run the ordering on **successful** episodes and confirm no consistent order
appears. If it does, the ordering is a property of where the taps sit and this
method is dead.

**And in N1.7 that hazard has a known period — which turns out to be an
opportunity. See §4.6.** Briefly: the DiT's conditioning is not uniform across
depth, so any depth-wise onset analysis inherits an architectural pattern that
looks exactly like an "information arrives here" finding. The control must be
*structural*, not statistical: detrending a periodic component out of an onset
statistic would also remove real periodic signal. The null is not "no
periodicity" but **"periodicity at exactly the architectural phase"**.

### 4.6 The DiT separates the visual and language pathways by block index

N1.7 instantiates `AlternateVLDiT` (`use_alternate_vl_dit: True`,
`groot_n1_7.py:113`, wired at `:472-476`) over 32 blocks. From
`action_head/cross_attention_dit.py:357-380`:

| blocks | what enters |
|---|---|
| all 16 **odd** | `encoder_hidden_states=None` — **nothing**, pure self-attention |
| 0, 4, 8, 12, 16, 20, 24, 28 (`idx % 4 == 0`) | **text** (`non_image_attention_mask`) |
| 2, 6, 10, 14, 18, 22, 26, 30 (even, `idx % 4 == 2`) | **image** (`image_attention_mask`) |

**The config name is misleading and cost us a wrong commit.**
`attend_text_every_n_blocks: 2` reads as period 2; text is actually every
*fourth* block, because the counter runs over even blocks only
(`idx % (2 * attend_text_every_n_blocks)`). An earlier revision of this section
said period 2, inferred from the name. `res` and `vla-dd` made the same
inference independently. **A phase error here silently corrupts any layer-wise
readout** — take the period from the forward pass, never from the config.

**Why this outranks §4.5.** The architecture has pre-separated the pathways.
Image-attending and text-attending blocks are distinguishable **by index alone**
— no intervention, no matched pairs, no patching, no seed control, no token
alignment. All four of §4.5's problems are bypassed because nothing is being
intervened on, and the multiple-mediators hazard does not apply because there is
no patching estimand. **This is §5's question answered by observation rather
than experiment**, and it should be tried before knockout and long before
patching.

The sharp version: does the observed onset phase align with the **text**
indices or the **image** indices? Those are different mechanistic claims and the
architecture separates them for us.

Aim it at **camera viewpoint**, the failure mode where §5 stays well posed (§8.9:
camera and robot-initial-state robustness are uncorrelated across ten published
models, so no single mechanism story should be expected to cover both).

**Capture requirement.** The current taps do not record per-block DiT outputs.
`AlternateVLDiT.forward` already accepts `return_all_hidden_states` (`:338`) and
returns all 33 (`:385-386`); the caller never passes it, so wrapping
`action_head.model.forward` from outside gets them with no hooks and no
`third_party` edit. **This must be priced before the full capture runs** —
re-running to add taps costs more than adding them now.

**N1.7-specific.** N1.5's plain DiT has no alternation, so published layer-wise
results on N1.5 carry no equivalent artefact and cannot calibrate ours.

### 4.5 Activation patching — the only causal method here

Our matched pairs (§4.2) are exactly the clean/corrupted structure that
activation patching requires, which is unusual; most work has to construct such
pairs synthetically.

Procedure: run the canonical episode caching activations at a tap; run the
perturbed episode; re-run the perturbed episode with that tap's activation
replaced by the canonical cache. **If the failure is repaired, the fault is at
or upstream of that tap. If it persists, the fault is downstream.**

Patching S2 — the VL stream as the DiT receives it — is the single most
informative intervention we could run. It splits the question exactly along the
line `res` was asked about: *was the action head given a good representation?*

**Four things that make this harder than it sounds.**

1. **Unseeded `randn`.** The denoising loop draws `torch.randn` unseeded at
   `groot_n1_7.py:657`, so patched and unpatched runs differ by noise as well as
   by the patch. It needs seed control, which `implementor`'s capture already
   has the mechanism for (§3.1 of the capture design: save/restore generator
   state), but the patching path would need it deliberately rather than
   incidentally.
2. **Token misalignment.** Canonical and perturbed observations produce
   different numbers of image tokens if image geometry differs — and
   `EXP_EMBEDDING_OOD.md` §2 notes N1.7 encodes images in their native aspect
   ratio. Patching across a token-count mismatch is not defined. This is the
   VLM-specific warning from §2 arriving in our own stack.
3. **It requires the GPU and a modified forward path**, so it is neither `res`'s
   to run nor covered by the current capture.
4. **A repaired episode is not proof of a single cause.** Patching a rich
   representation can repair behaviour by supplying information the model would
   have derived elsewhere. Report repair *rates* across many pairs, not
   anecdotes.
5. **Multiple mediators — never produce a ranked list of taps.** "The Curse of
   Multiple Mediators" ([arXiv:2606.27510](https://arxiv.org/abs/2606.27510),
   via `res`) shows the patching estimand contains interaction terms that
   **scale with the distance between the clean and patched activations**. Our
   robot-initial-state pairs are precisely the far-apart case, so interactions
   will be large and greedy per-tap ranking will misattribute. Report per-tap
   effects **as a set, with interactions acknowledged** — never a leaderboard
   with a winner.

**Run knockout before patching — `res`'s recommendation, adopted.** Attention
knockout and input masking need no matched pairs, no seed control and no token
alignment, which removes problems 1, 2 and half of 5 above. There is released
code (`action-atlas`, §8) and it supports GR00T **N1.5**, one minor version off
our policy. Patching is the expensive follow-up, not the first move.

---

## 5. Why the mechanism question needs no labels at all

This is the part worth internalising before anyone spends GPU time.

**"Which module is failing" is a comparison across signals within the same
forward.** At forward *t*, if the VLM-side taps sit inside the reference
manifold while the action-side taps sit outside it, the action head received a
good representation and produced bad actions. Reverse it and the fault is
upstream.

That statement is *relative* — it compares S1/S2 against S3/S4 at one moment. It
needs no pass/fail label, no episode label, and no threshold that has to be
calibrated against a negative class. The per-episode label is then used only
where it is valid: to check that the pattern differs between episodes that
succeeded and episodes that did not.

And it prescribes different data. **Upstream fault → source more diverse
scenes. Downstream fault → source more diverse trajectories within the scenes we
already have.** That is the decision this whole programme exists to inform, and
we currently cannot make it.

---

## 6. Negative controls, non-negotiable

Three failures this week looked like findings until they were checked. Before
believing any separation:

1. **Success-vs-success split.** Run the entire pipeline on two halves of the
   *successful* episodes. If they separate, the method is detecting nuisance
   structure — scene identity, task, episode length, base-scene appearance — and
   not failure. This is the single cheapest way to invalidate the whole thing,
   so run it first.
2. **Prefix-matched comparison.** First *k* forwards of failures against first
   *k* of successes. If indistinguishable early, that confirms the §1
   contamination is real *and quantifies* when signal appears. It also bounds
   how early any of this could work as a predictor.
3. **Scene-stratified evaluation.** Ten base scenes are not ten independent
   samples of "a manipulation task". Any aggregate that pools them can be driven
   by one scene. Report per-scene, then pool.
4. **A shuffled-label run.** Permute episode labels and confirm the separation
   disappears. Cheap, and it catches leakage through episode length — failures
   run 281 steps and successes ~113, so *any* statistic correlated with duration
   will separate the classes for reasons that have nothing to do with
   representations.

**Prefer "fix the timestep, then compare" over normalising by length.** `res`
reports the frozen-VLA probing literature uses exactly that control. Comparing
forward *i* against forward *i* removes the episode-length confound outright,
where a length-normalised aggregate only dilutes it. Normalisation is the
fallback when a matched timestep is unavailable.

Control 4 is not hypothetical. Our failures time out at 281 steps and our
successes finish around 113. **Episode length alone is a near-perfect
classifier.** Any method that sees more forwards from failures than successes
inherits this, and an aggregate like "fraction of steps OOD" is only safe
because it normalises by length. Anything that sums rather than averages is
contaminated by construction.

---

## 7. What we do not know

- Whether any embedding signal separates at all. The 40-episode de-risk exists
  to answer this before the ~3.5 h capture is spent.
- Whether the resampling proxy (S4) carries usable information without RTC
  overlap. It is a different quantity from STAC and may simply be weaker.
- Whether departures are ordered across taps, or simultaneous. If everything
  leaves the manifold at once, §4.4 yields nothing and §4.5 becomes the only
  route.
- **The `n_action_steps` confound is live** (GR00T at 16, MINERVA at 1). It sits
  underneath any GR00T-vs-MINERVA architectural claim. `primary` owns settling
  it; nothing here should assume a resolution.
- Whether `state_encoded` (S0e) is comparable at all across embodiments — it is
  produced by a per-embodiment `CategorySpecificLinear`, so vectors from
  different embodiments must not be pooled into one reference cloud.

---

## 8. `res` — findings from the literature

*Written 2026-09-21 by `res`. §1-§7 left untouched; disagreements are named by
section below. Ratings are my triage estimate of usefulness **to this project**,
not paper quality. `[R]` = read from search-result summaries only. `[V]` =
verified against the primary source. `[D]` = derived from our own data by me.*

### 8.0 Headline, four claims

1. **The submodule-localisation gap `primary` assumed we were in is mostly
   closed.** Two 2026 papers do exactly this, cross-architecture, and one ships a
   toolkit whose README declares GR00T N1.5 support. §4.5's activation-patching
   plan is probably not a first-of-its-kind build. (§8.2)

   **Caveat, `primary`'s — and the recon has now settled it against me.**
   §8.11 read the source: action-atlas supports N1.5 **and N1.6**, and its
   action-head accessors (VL-SA, 32-block DiT) will likely transfer to N1.7 —
   but its **input path is pinned to the Eagle processor**, which N1.7 replaced
   with `Qwen3VLProcessor`, so its episode runner cannot feed our checkpoint.
   **The "released code we could run" part of this claim is withdrawn.** What
   survives: the accessors are worth reading as a reference, the *methods* are
   documented well enough to reimplement, and the *finding* that the visual
   pathway dominates stands on the papers' own models. We would be
   reimplementing published method, not adopting published code.
2. **`primary`'s OpenVLA / OpenVLA-OFT natural experiment survives verification
   and is stronger than claimed** — OFT's vision encoders are not merely shared,
   they are *frozen*. (§8.3)
3. **The linear-probe programme in §4.3 has been done**, repeatedly, and it
   works better than §4.3 expects — but the published versions probe for
   *outcome*, not for the privileged quantities we hold, which is where our
   version would be novel. (§8.4)
4. **My own hypothesis (B) is false in its simple form, by our own data.**
   Robot-initial-state t=0 states are displaced far beyond any demonstrated
   *start*, but about half sit inside the demonstrated manifold overall, and the
   displacement **does not predict which variants fail**. (§8.6)

---

### 8.1 Broad survey, rated for triage

**Tier A — read these, they change what we should build.**

| Source | What it is | Relevance |
|---|---|---|
| [VLA-Trace, arXiv:2605.30117](https://arxiv.org/abs/2605.30117) `[R]` | CKA representation tracing + **attention-knockout** interventions + behavioural probes, on π0.5 and OpenVLA. Localises which modality pathway carries control. | **10/10** — closest published thing to our whole remit |
| [Not All Features Are Created Equal, arXiv:2603.19233](https://arxiv.org/abs/2603.19233) (ICLR 2026) `[R]` | First cross-architecture mechanistic study: activation injection + SAEs + linear probes, 6 models 80M–7B, 394k rollouts. Finds the **visual pathway dominates action generation**; causal ablation zero-effect rates 28–92%. | **10/10** — the method catalogue we were about to reinvent |
| [action-atlas](https://github.com/CWRU-AISM/action-atlas) `[V]` | That paper's toolkit. Grid ablation, vision perturbation, counterfactual prompting, cross-task injection, SAE training, concept ablation/steering. Supports GR00T **N1.5 and N1.6**; action-head accessors likely transfer to N1.7, **input path pinned to the Eagle processor does not** — §8.11. | **6/10** — reference for tap paths; not runnable as-is |
| [VLM4VLA, arXiv:2601.03309](https://arxiv.org/abs/2601.03309) (ICLR 2026) `[R]` | 17 VLMs → VLA under a fixed minimal adapter. **Vision encoder, not the language model, is the bottleneck**; general VLM competence poorly predicts control. | **9/10** — directly answers the user's "would a better VLM help" |
| [OpenVLA-OFT, arXiv:2502.19645](https://arxiv.org/abs/2502.19645) `[V]` | Action-side-only changes at a frozen vision backbone. | **9/10** — the natural experiment, verified (§8.3) |
| [FailureSpot, arXiv:2609.04277](https://arxiv.org/abs/2609.04277) `[R]` | States §1's label problem verbatim; fixes it with action-derived weak supervision + active learning. | **8/10** — confirms §1 is a real, named problem |
| [Curse of Multiple Mediators, arXiv:2606.27510](https://arxiv.org/abs/2606.27510) `[R]` | Activation patching's NIE contains **interaction effects** that scale with clean/patched activation distance; greedy per-component ranking misses mechanisms. | **8/10** — a direct hazard for §4.5, see §8.5 |

**Tier B — relevant, read if the top tier leaves gaps.**

| Source | What it is | Relevance |
|---|---|---|
| [What Frozen VLAs Already Know About Success, arXiv:2605.28527](https://arxiv.org/abs/2605.28527) `[R]` | Linear probes on frozen π0.5/OpenVLA/DINOv2/CLIP recover outcome; ~92% pairwise ordering with task+timestep fixed; probe-reranking lifts push-plate 26.7→44.3%. | 8/10 — §4.3 precedent, and the timestep control matters for §6 |
| [Beyond Task Success, arXiv:2606.01095](https://arxiv.org/abs/2606.01095) `[R]` | Behavioural + SAE diagnostics over 7 policies on LIBERO; classifies representations **memorized / reactive / predictive**. | 7/10 — a vocabulary for what a tap contains |
| [Sparse Autoencoders Reveal … Steerable Features in VLA, arXiv:2603.19183](https://arxiv.org/abs/2603.19183) + [drvla](https://github.com/swannaiden/drvla) `[R]` | SAE features are causally steerable; ablating them destroys performance. | 7/10 — causal, and packaged |
| [LIBERO-Plus, arXiv:2510.13626](https://arxiv.org/abs/2510.13626) `[R]` | Our benchmark. 95%→<30% under camera/robot-init; models **ignore language**. | 7/10 — already in use; §8.3 leans on its Table 1 |
| [Sentinel, arXiv:2410.04640](https://arxiv.org/abs/2410.04640) | STAC + VLM QA, successes only. | 6/10 — §2.1 correct that STAC does not transfer (§8.5) |
| [Causal tracing in LVLMs, arXiv:2511.05923](https://arxiv.org/abs/2511.05923) `[R]` | Last-token patching fails for VLMs; visual info is distributed across image tokens. | 6/10 — confirms §2's warning |
| [VLA-Scope, arXiv:2609.21246](https://arxiv.org/abs/2609.21246) · [FARM, arXiv:2609.11445](https://arxiv.org/abs/2609.11445) · [ProbeAct, arXiv:2606.09740](https://arxiv.org/abs/2606.09740) · [SAFE, arXiv:2506.09937](https://arxiv.org/abs/2506.09937) `[R]` | Failure *detection* family. | 5/10 — detection, not localisation; §8.2 |
| [Hide-and-Seek in Trajectories, arXiv:2605.30834](https://arxiv.org/abs/2605.30834) `[R]` | Discovering failure signals for runtime monitoring. | 5/10 |

**Tier C — context only.** [Knowledge Insulating VLA](https://openreview.net/forum?id=cb0xbZ3APM) (action experts harm knowledge transfer); [Decoupled Action Expert, arXiv:2511.12101](https://arxiv.org/abs/2511.12101); [Characterizing VLA Models, arXiv:2603.02271](https://arxiv.org/abs/2603.02271) (edge-AI bottleneck, not failure); [Event-Grounded SAEs, arXiv:2605.17204](https://arxiv.org/abs/2605.17204); [Prisma, arXiv:2504.19475](https://arxiv.org/abs/2504.19475) (vision/video mech-interp toolkit).

---

### 8.2 Q1 — has anyone localised VLA failure to a submodule?

**Yes. This is the answer to the question `primary` framed as our gap, and the
gap is narrower than assumed.**

*Not All Features Are Created Equal* is the direct precedent: activation
injection, SAEs and linear probes across six VLAs from 80M to 7B, on 394k+
rollout episodes. Its headline — **the visual pathway dominates action
generation** — is a submodule-attribution claim of exactly the shape we want.
It reports causal-ablation zero-effect rates of 28–92%, i.e. *most components,
most of the time, do not matter causally*, which is a warning about how much
signal to expect from any single tap.

*VLA-Trace* is the closer methodological match because it is explicitly
diagnostic rather than descriptive: CKA to trace representation evolution,
**attention knockout** to test whether a representation is functionally required
for action decoding, and behavioural probes for grounding vs shortcut
dependence. Its masking battery (target-object, gripper, robot-body, background)
is a cheaper causal instrument than activation patching and would answer a
different slice of our question.

**What this changes for us.** §4.5 should be re-scoped from "the only causal
method here" to "the expensive causal method, to be run after the cheap ones
that already have published protocols and, in action-atlas, released code
supporting GR00T N1.5". Attention knockout and input masking need no matched
pairs, no seed control, and no token alignment — they sidestep three of §4.5's
four stated problems.

**The genuine gap is narrower and we still own it:** nobody in this set
localises failure to a submodule **crossed with a labelled perturbation
dimension**. LIBERO-Plus gives us the perturbation label per instance; the
mechanistic papers work on nominal rollouts. The *signal × perturbation-type*
table `primary` wants remains, as far as I can find, unpublished.

---

### 8.3 The natural experiment — verified, and stronger than claimed

`primary` asked me to verify the same-backbone claim because the argument rests
on it. **It holds. [V]**

From the OFT paper: OpenVLA "combines a fused vision backbone (with both SigLIP
and DINOv2 vision transformers), a Llama-2 7B language model". OFT's changes are
all action-side or conditioning-side: multi-image input through the *shared*
backbone, proprioceptive state via MLP, causal→**bidirectional attention** for
parallel decoding, the LM decoder output layer replaced by a **4-layer MLP** for
continuous actions, action chunks instead of single timesteps, and FiLM
modules **in OFT+ only**. All experiments use LoRA; **the vision encoders remain
frozen**.

So the LIBERO-Plus Table 1 swing `primary` quotes — camera 1.1 → 59.7, robot
init 4.1 → 37.2 — is obtained with *frozen, identical* vision encoders. That is
a stronger statement than "same backbone": it is *the same visual features*.

**Two caveats that must travel with the claim.** Causal→bidirectional attention
changes how the language model *reads* those features even though the weights
are shared, and OFT+ adds FiLM, which is backbone conditioning. So "action-side
only" is right about the *encoders* and slightly loose about the *pathway*. If
the Table 1 row is OFT+ rather than OFT, say so.

**Verdict on the user's question — would a stronger VLM help GR00T?** The
literature agrees with `primary`'s answer but for a sharper reason. VLM4VLA
found that general VLM competence **poorly predicts** downstream control and
that the *vision encoder* is the bottleneck, not the language model — so
"stronger VLM" in the usual sense (a better reasoner) is the wrong axis.
Combined with our own numbers (the categories a VLM owns already sit at 86–93%
on the hardest variants), the honest answer is: **a better backbone buys little
except possibly on camera viewpoint, and the leverage is action-side.** I could
not break the argument; I tried.

**One cheap thing nobody appears to have done** (Q2 of `primary`'s list): regress
LIBERO-Plus Table 1 robustness against backbone properties across its eight
models. The numbers are published, it is CPU-only, and it would turn this
argument from two anecdotes into a trend. I can run it.

---

### 8.4 Q2 — has the §4.3 linear-probe programme been done for action heads?

**Yes, and it works better than §4.3 anticipates — but not the version we would
run, and that is the opportunity.**

*What Frozen VLAs Already Know About Success* probes frozen π0.5/OpenVLA
features for Monte-Carlo outcome and reaches ~92% pairwise ordering accuracy
**with task and timestep held fixed** — and then uses the probe to rerank
candidate action prefixes, lifting a real task from 26.7% to 44.3%. That is a
probe with demonstrated causal purchase, not just decodability. *Beyond Task
Success* runs SAE feature analysis over 7 LIBERO policies and sorts
representations into memorized / reactive / predictive.

**Two things transfer directly into §4.3 and §6.** First, their task-and-timestep
control is the published form of §6's control 4: fixing timestep is precisely
what defeats episode-length leakage, and it is a stronger control than
normalising by length because it removes the confound rather than diluting it.
**§6 should adopt "fix timestep, then compare" as its primary form.** Second,
the standard caution applies — linear decodability is not causal relevance, as
the mech-interp literature states plainly and as §4.3 already concedes.

**Where our version would be new.** Everyone probes for *outcome*. We hold
privileged simulator truth — `_gt_eef_to_object`, `_gt_n_contacts`, object pose,
phase — so we can probe for **what the representation contains about the world**
rather than **how it expects things to turn out**. "Does S2 linearly encode
distance-to-target under a robot-initial-state perturbation" is a question the
published probes cannot ask, because their environments do not expose it. §4.3's
instinct that this is our most underused asset is, as far as I can tell, correct.

---

### 8.5 Q3/Q4 — the ordering argument, and what is already broken

**§4.4's cross-tap ordering argument: no published precedent found, and it has a
hazard.** I found no VLA work that dates manifold-departure per tap and reads
the mechanism off the ordering. The nearest analogues are layer-wise probing
studies in LLMs/retrieval, where decodability peaks at intermediate layers — a
result that should make us cautious, because it means **tap depth alone changes
when a signal becomes visible**, independent of when anything went wrong.
Ordering across taps at different depths may measure the architecture rather
than the failure. §7 already lists "departures may be simultaneous" as a risk;
add this one: *departures may be ordered for reasons that have nothing to do
with causation*. A negative control exists — run the ordering on **successful**
episodes and confirm no consistent order appears.

**§4.5 has a named hazard we should record.** *The Curse of Multiple Mediators*
shows activation patching's estimand contains interaction effects that **scale
with the distance between clean and patched activations** and are negligible
only when the model is locally affine. Our matched pairs under a
robot-initial-state perturbation are exactly the far-apart case, so interaction
effects will be *large*, and greedy per-tap ranking will misattribute. The paper
argues INT is a diagnostic rather than a nuisance — its sign and magnitude tell
you when conclusions are prompt-dependent. This does not sink §4.5; it means
**report per-tap effects as a set with their interactions, never a ranked list
with a winner.** Add it as problem 5 alongside §4.5's four.

**§2.1's STAC judgement: `primary` is right. [V-ish]** STAC compares distributions
of *overlapping* consecutive chunks; with RTC off there is no overlap, and
resampling k times from fixed conditioning measures posterior spread at one
moment rather than disagreement between successive commitments. These are
different quantities and should not be reported interchangeably. I found **no
paper that quantifies how much weaker the resampling proxy is** — so we cannot
cite a discount factor, and any claim about its strength has to be measured on
our own data. Worth noting FIPER independently uses action-chunk entropy over
multiple samples as a signal in its own right, so the proxy is not worthless —
it is simply not STAC.

**§1's category-error claim: the literature agrees with the diagnosis and
disagrees with the conclusion.** FailureSpot states our problem verbatim, which
vindicates §1. But it does not conclude the question is malformed — it concludes
the *supervision* is, and fixes it with action-derived weak labels plus a small
active-learning annotation budget. So "reformulate, do not answer" is one valid
route; "get better labels cheaply" is another that a published method already
takes. §1 should acknowledge the second exists rather than presenting
reformulation as the only option.

---

### 8.6 `[D]` My hypothesis (B), tested on our data — it is false in its simple form

I claimed the robot-initial-state failures might be explained by the perturbed
t=0 state lying outside the demonstrated corpus. **Run CPU-only, read-only. It
does not hold up, and the negative control is what killed it.**

**Method.** Reference = the 432 `libero_spatial` demonstrations (task_index
30–39, 52,970 frames) from `lerobot/libero` @ `a1aaacb7`, the corpus GR00T's
`gr00t17-lerobot-libero_spatial-640` checkpoint was finetuned on. Query = t=0
state of every episode in `runs/lplus_fail_groot` and
`runs/groot_control_lplus_stack`, rebuilt with the harness's own
`libero_state()` (`vla_harness/policies/lerobot_policy.py:334`) so the
comparison is in the 8-D vector the policy actually consumes. Statistic =
Euclidean distance to the **nearest demonstrated state anywhere in the corpus**,
per-dimension-standardised by the all-frame sd.

**A trap I fell into and the check that caught it.** My first pass used
Mahalanobis against the demonstrated *t=0* covariance and returned distances of
332–790 for robot-init against ~3.6 for controls. That is an artefact: at t=0
the demonstrated `aa_x` has sd ~1e-3, so the covariance is near-singular and
inflates everything by ~100x. It also silently pooled all four suites because my
task-text match failed. The number was enormous and agreed with my prior, which
per `primary`'s own rule is exactly when to distrust it. Reported here because
the corrected analysis is the one below.

**Result 1 — the perturbation is real and it is the only one that moves the
state.** Distance to nearest demonstrated state, t=0:

| arm | n | median | p90 | max |
|---|---|---|---|---|
| control (unperturbed) | 100 | 0.024 | 0.041 | 0.060 |
| Objects Layout | 198 | 0.012 | 0.012 | 0.012 |
| Light Conditions | 44 | 0.018 | 0.019 | 0.037 |
| Sensor Noise | 69 | 0.020 | 0.044 | 0.047 |
| Language Instructions | 117 | 0.023 | 0.044 | 0.047 |
| Background Textures | 17 | 0.037 | 0.040 | 0.047 |
| Camera Viewpoints | 62 | 0.044 | 0.047 | 0.047 |
| **Robot Initial States** | **116** | **1.358** | **2.465** | **3.713** |

Every non-proprioceptive perturbation leaves t=0 indistinguishable from a
demonstrated start — as it must, and it is a clean positive control for the
pipeline. Robot-init is ~30–90x further out. The dominant dimensions are
**orientation, not position** (`aa_z` −4.01, `aa_x` −3.78, `aa_y` −2.32 sd at
the extreme, against ≤1.15 for eef xyz), which independently corroborates
R-030's finding that J6 wrist pitch dominates.

**Result 2 — the negative control, which is where the hypothesis dies.** Holding
out 40 whole demonstrations and scoring their frames against the rest:
**NN median 0.372, max 1.445**. Genuinely demonstrated mid-trajectory states are
*much* further from their neighbours than demonstrated *starts* are. Against
that yardstick only **48.3% (56/116)** of robot-init t=0 states are outside the
demonstrated manifold at all. "These states appear nowhere in the corpus" is
**too strong and I withdraw it.**

**Result 3 — the distance does not predict failure.**

| t=0 position | n | GR00T success |
|---|---|---|
| inside demonstrated manifold | 60 | 36.7% |
| outside it | 56 | 28.6% |

8 pp on n≈58 per bin; Mann-Whitney on the raw distances, failed vs succeeded,
gives **p = 0.083 — not significant**. Both bins fail roughly two times in
three. **Being off-manifold at t=0 is not what makes these episodes fail.**

**What it means, and it matters for §5.** The failure is not explained by
coverage of the *starting state*. Combined with R-030 — the arm travels further
than a success, grips 3–4x as often, and never returns toward neutral — this
supports the framing in my pushback (C) and `COVERAGE_GAP_METHOD.md`'s existing
family-C headline: what is missing is not demonstrated *states* but demonstrated
*behaviour from those states*. No demonstration ever shows recovery toward
neutral, because every demonstration starts there. **That is a skill gap, not a
coverage gap, and the prescription is on-policy corrective data rather than more
nominal demonstrations.**

**Bearing on §5, stated plainly as requested.** For this failure mode, §5's
upstream-vs-action-head cut looks like **the wrong question**. Neither horn fits
a policy whose representation is fine, whose individual actions are unremarkable,
and which fails by compounding into a region where no demonstrated behaviour
exists. §5 remains well posed for camera-viewpoint failures, where a perceptual
cause is credible. I would not spend the 3.5 h capture on robot-init expecting
§5 to resolve it.

**Caveats.** `observation.state` in the corpus is 8-D — eef pos, axis-angle,
gripper — with **no joint angles**, so the joint-space histogram `primary` asked
for is not obtainable from the training corpus; this analysis is in the policy's
input space instead, which I would argue is the more relevant one but is not the
same claim. Single suite, one checkpoint. `steps[0]` is assumed post-settle;
`num_steps_wait` was not verified. And `primary`'s caution stands and is now
load-bearing: I did **not** establish whether LIBERO-Plus draws its initial
states independently of the demonstrated distribution, so Result 1 may partly
describe the benchmark's design rather than a discovered gap.

---

### 8.7 What I would do next, ranked

1. **Re-scope §4.5.** Run attention knockout / input masking (VLA-Trace's
   battery, action-atlas's grid ablation) *before* activation patching. No
   matched pairs, no seed control, no token alignment — three of §4.5's four
   problems vanish, and there is released code for GR00T N1.5.
2. **Check action-atlas against GR00T N1.7.** It supports N1.5. If the tap
   points survive the minor-version gap, a large part of `implementor`'s capture
   work has a reference implementation to check against. CPU-inspectable.
3. **Regress LIBERO-Plus Table 1 robustness on backbone properties** across its
   eight models. CPU-only, published numbers, and it converts §8.3 from anecdote
   to trend.
4. **Probe for privileged quantities, not outcome** (§4.3, §8.4). This is our
   genuinely novel slice.
5. **Settle whether LIBERO-Plus initial states are drawn from the demonstrated
   distribution.** It decides whether §8.6 Result 1 is a finding or a
   description of the benchmark's design, and it is a documentation question,
   not a GPU one.

### 8.8 `[D][V]` How LIBERO-Plus generates robot initial states — the blocker, settled

`primary` made this a blocker because §8.6's headline means opposite things
depending on the answer. **It is now answered, from the fork's own source.**

**The generator** is `third_party/LIBERO-plus/libero/libero/envs/robots/new_init.py`,
a code-generation script that writes out the variant classes. Its whole method:

```python
np.random.seed(42)
original_qpos = np.array([0.0, -1.61037389e-01, 0.00, -2.44459747e00,
                          0.00, 2.22675220e00, np.pi / 4])   # canonical neutral pose
perturbation  = np.random.randn(7)
perturbation /= np.linalg.norm(perturbation)                 # isotropic unit direction
perturbed_qpos = original_qpos + perturbation * RADIUS
```

Five blocks of 100, `RADIUS` stepping 0.1 → 0.5 rad. **Verified by parsing the
500 generated `MountedPanda` classes** in `mounted_panda.py` and recomputing
‖qpos − original‖: initstate 1–100 = 0.1000, 101–200 = 0.2000, 201–300 = 0.3000,
301–400 = 0.4000, 401–500 = 0.5000 rad, exactly, no spread.

**The answer to the blocking question: the initial states are NOT drawn from the
demonstrated distribution.** They are an isotropic random direction in 7-D joint
space at a fixed radius from the canonical start. Demonstration data plays no
part. **So §8.6's Result 1 partly describes the benchmark's generation procedure
rather than discovering a hole** — robot-init states are off-manifold *by
construction*, and `primary`'s caution was correct. §8.6 Result 1 should be
reported as *"the perturbation does what it says, and no other perturbation
touches proprioception"* — a positive control for our measurement — **not** as a
discovered coverage gap. Results 2 and 3 are unaffected: they compare perturbed
states against each other and against held-out demonstrations, and neither
depends on how the states were generated.

**The bonus, and it is worth more than the blocker.** The radius is a genuine,
benchmark-native measure of **perturbation magnitude** — exactly what
`docs/LIBERO_PLUS_LEVELS.md` §1 says the difficulty *level* is not. We can now
ask a dose-response question that was previously unaskable.

**Result — corrected after `primary`'s mediator objection, which was right.**
My first reading of this was "magnitude carries nothing, level carries
everything". That over-corrected, for a reason worth recording: **difficulty
level is a mediator of radius, not an independent covariate.** Level is how many
of four reference models solved the variant, and whether they solved it depends
on how far the arm was displaced — so the causal path runs
`radius → task difficulty → level`, with level *downstream* of the effect being
estimated. Conditioning on a mediator blocks part of that effect and makes a real
one look null. **Confirmed in our own data:** radius strongly predicts level
(26.7% L5 at r=0.2 rising to 72.4% at r=0.4; mean radius 0.413 at L5 vs 0.359 at
L4, Mann-Whitney **p = 0.0076**). So the stratified rows below do not say radius
is inert; they say **radius adds nothing once level is known**, which is a
different and much weaker claim.

GR00T's 116 robot-init variants from R-026:

| radius (rad) | n | success |
|---|---|---|
| 0.2 | 15 | 46.7% |
| 0.3 | 26 | 30.8% |
| 0.4 | 29 | 31.0% |
| 0.5 | 46 | 30.4% |

**The marginal table is the total effect**, which is what a dose-response
question actually asks for: a 16-point drop from 0.2 to 0.3, then flat. That
shape — **an effect that saturates by ~0.3 rad** — is substantive, not a null.
But it is *not* separable at this n: 0.2 against pooled 0.3–0.5 gives Fisher
**p = 0.246, OR 1.98**, and the per-radius Wilson intervals ([21,72] at r=0.2
against [17,44] at r=0.5) overlap heavily. **The honest statement is that the
shape is suggestive of saturation and this subset cannot separate it from flat.**
Neither "composition, not dose" (mine) nor "a real effect then saturation"
(`primary`'s) is supported by these numbers alone. §8.10 specifies the experiment
that would settle it.

Within difficulty level — i.e. conditioning on the mediator, so read this as
"radius adds nothing *beyond level*", not "radius is inert":

| stratum | n | mean radius, failed | mean radius, succeeded | Mann-Whitney |
|---|---|---|---|---|
| level 4 | 46 | 0.358 | 0.359 | p = 0.514 |
| level 5 | 70 | 0.415 | 0.406 | p = 0.346 |

Level, by contrast, is strongly predictive: **L4 47.8% vs L5 22.9%, Fisher exact
p = 0.0081, OR 3.09.**

**Why this matters more than the blocker it was raised to settle.** Difficulty
level is defined as *how many of four other VLAs (OpenVLA-OFT, π0, π0-FAST,
UniVLA) solved that variant* — a property of other models, not of the physics.
So what predicts GR00T's failure on a robot-init variant is **not how far the
arm was displaced, but whether other architectures also failed on it.** Beyond
about 0.2 rad the response saturates at ~30% and stops caring about magnitude.

**Held one notch looser than I first put it, at `primary`'s insistence and
correctly so.** Co-failure across architectures *constrains*; it does not
*identify*. It is consistent with the family-C skill gap of §8.6, and equally
consistent with a shared property of the LIBERO demonstration corpus all five
models trained on, and with some variants simply being kinematically harder in a
way no current method handles. That is the same discipline applied to the
MINERVA claim in `primary`'s (A) correction, and it applies here too. What it
does do is make a **single-module fault in GR00T specifically** hard to sustain.

**Caveats.** These 116 are the L4+L5 subset of a deliberately unbalanced failure
hunt (R-026 states no cross-level comparison from it is valid), so radius and
level are confounded in the marginal table — which is exactly why the stratified
rows are the ones to read. No radius-0.1 variants are present at all. Per-cell
n is 15–46. The clean version of this experiment is a **balanced radius sweep at
fixed scene**, which would need the GPU and is `primary`'s to schedule; it would
turn a null result on a biased subset into a real dose-response curve.

---

### 8.9 `[D][V]` Does the backbone predict robustness? — LIBERO-Plus Table 1

`primary`'s task (3). **Descriptive only**: n=10 models with correlated design
choices, so no fitted model and no inference. Table 1 read from the PDF
(arXiv:2510.13626v3) with `pdftotext`, not from a search summary.

**The decisive fact needs no statistics.** Camera-viewpoint robustness across
all ten models spans 0.3 → 66.4. The five models sharing **one identical
Prismatic backbone** (Llama-2 7B, DINOv2+SigLIP: OpenVLA, OFT, OFT_w, OFT_m,
RIPT-VLA) span **1.1 → 59.7 — 89% of that entire range.** The two PaliGemma-3B
models span 15.8 → 66.4, another 50.6 pp. **Backbone identity explains
essentially none of the variance in robustness**, because the within-backbone
spread is as large as the across-model spread.

**Three fixed-backbone natural experiments — the shape `primary` asked for.**

| pair | what changes | camera | robot init | clean |
|---|---|---|---|---|
| OpenVLA → OpenVLA-OFT | action side; encoders **frozen** | 1.1 → 59.7 (**+58.6**) | 4.1 → 37.2 (**+33.1**) | +20.6 |
| π0 → π0-fast | **action representation only** | 15.8 → 66.4 (**+50.6**) | 6.6 → 24.8 (**+18.2**) | **−8.7** |
| OpenVLA-OFT → OFT_w | **input channel only** (wrist camera removed) | 59.7 → 16.8 (**−42.9**) | 37.2 → 43.7 (+6.5) | −1.8 |

**π0 → π0-fast is the cleaner of the two `primary` has been working from, and it
is new to this thread.** Verified: π0-fast "uses the same model backbone and
training dataset" as π0, differing only in action representation — flow matching
replaced by FAST (DCT-based discrete tokenisation). So **backbone *and* training
data are both held fixed**, which the OpenVLA/OFT pair cannot claim, and there is
no bidirectional-attention or FiLM confound. +50.6 pp on camera from the action
representation alone.

**And it rules out the obvious story.** OpenVLA→OFT goes discrete → continuous
and gains; π0→π0-fast goes continuous → discrete and gains. **The two point in
opposite directions on that axis**, so "continuous actions are more robust" is
not the lesson. What both share is only that the action side was redesigned.

**π0-fast also gets *more robust while getting less accurate*** (clean 94.2 →
85.5, camera 15.8 → 66.4). Robustness and clean success rate dissociate — which
is the whole premise of this project stated in someone else's numbers.

**The finding I did not expect, and it changes what we should do.** Across the
ten models, camera robustness and robot-init robustness are **uncorrelated**:
Spearman **ρ = +0.09, p = 0.803**. Clean success predicts neither strongly
(ρ = +0.59, p = 0.074 for camera; ρ = +0.49, p = 0.150 for robot init). The
OFT → OFT_w row shows the dissociation directly: removing the wrist camera costs
**42.9 pp of camera robustness and *gains* 6.5 pp on robot init**.

**So "camera" and "robot initial state" are not one axis of difficulty.**
Whatever buys viewpoint robustness does not buy proprioceptive robustness, and
can cost it. This independently supports `primary`'s decision to aim the
embedding capture at **camera viewpoint**, where §5's upstream-vs-head cut stays
well posed, and to stop expecting one mechanism story to cover both.

**Scope caveat, raised by `vla-dd` and important enough to state before the
verdict.** Everything above is a claim about **variance in robustness *across*
models**. It does not establish that the backbone is causally uninvolved
*within* one model. A component can be necessary to a behaviour while not being
what differentiates ten checkpoints — necessity and explanatory variance are
different properties. So §8.9 is a **strong reason to demand a specific question
before spending ablation or SAE compute on backbone interpretability**, and a
**weak reason to conclude backbone interpretability is uninformative**. Do not
read it as the latter.

**Verdict on "would a stronger VLM help GR00T".** Unchanged and now better
grounded: **no, and the leverage is action-side.** Three independent lines —
Table 1's within-backbone spread, the π0/π0-fast pair, and VLM4VLA's finding that
general VLM competence poorly predicts control — all say the backbone is not
where robustness lives. **Caveat on scope:** the paper's own Finding 4 credits
OFT's wrist camera, and the OFT_w row confirms it is worth 42.9 pp on camera. So
*input channels* matter enormously even though *backbone identity* does not.
"Better VLM" is the wrong axis; "more/better-placed cameras" is a live one.

**Variant labelling, per `primary`'s caution.** Table 1's rows are OpenVLA-OFT
(wrist + third-person), OFT_w (third-person only), OFT_m (mix-sft, all four
suites). The paper does not state which carry FiLM; FiLM is an OFT+ feature and
none of these rows is labelled OFT+. **So "frozen encoders" is safe for the
OpenVLA → OFT row as regards the *encoders*, and the bidirectional-attention
caveat from §8.3 still applies to all three.** RIPT-VLA is an RL post-training
stage over the same stack, which is why it lands beside OFT.

---

### 8.10 The balanced radius sweep — spec, as requested

`primary` approved this and asked me to spec it. It exists because §8.8 is stuck
for a structural reason no post-hoc adjustment fixes: radius and level are
confounded, and level is a *mediator*, so it can be neither ignored nor
conditioned on. **A balanced design breaks the confound by construction.**

| field | value | why |
|---|---|---|
| **Scenes** | **2**: `next_to_the_ramekin` and a cookie-box scene | R-030 found the mechanism is *not uniform* — the arm reaches 6.4–7.4 cm on ramekin scenes and never gets within 19 cm on cookie-box ones. One scene would measure one mechanism and generalise wrongly, which is `primary`'s own measurement error #4. |
| **Radii** | **0, 0.1, 0.2, 0.3, 0.4, 0.5** | radius 0 = canonical, the anchor `primary` asked about — **include it**. 0.1 is absent from all 116 existing episodes and is exactly where the marginal table suggests the action is. |
| **Episodes** | **20 per radius per scene = 240 total** | fits the couple-of-hours budget; R-026 ran 623 overnight. |
| **Init states** | 20 distinct `initstate` indices drawn from each radius block (1–100, 101–200, …) with a **fixed seed, recorded** | direction is isotropic, so sampling 20 of 100 gives direction diversity within a fixed radius. |
| **Held fixed** | canonical camera `view_0_0_100_0_0`, `--base-instruction`, GR00T bf16, nas=16, render 360, MuJoCo 3.3.2 | only `initstate` varies. Same stack as R-029/R-026 so results are comparable. |

**The analysis must be a trend test, not pairwise.** At n=40 per radius (pooled
over scenes) the per-cell CI is about ±15 pp, so detecting the ~16 pp drop
pairwise would need roughly 150 per arm — unaffordable. **Cochran–Armitage across
all six ordered radii** has far more power against a monotone or saturating
alternative than any pairwise comparison, and the question is about *shape*.
Pre-register the three shapes before running: **monotone decline**, **saturating
by 0.2–0.3** (what §8.8 hints at), or **flat above 0**.

**Report per scene, then pool** (§6 control 3), and expect the two scenes to
differ — R-030 says they should.

**Why it matters beyond settling an argument.** `primary` is right that if
saturation is real it has a direct post-training consequence: **corrective data
would need to cover the whole ball rather than concentrate near neutral**,
because past ~0.2 rad the policy is equally lost everywhere. If instead the
decline is monotone, near-neutral corrective data is worth most. Those are
opposite data-sourcing prescriptions and this is the cheapest experiment that
distinguishes them.

---

### 8.11 `[V]` action-atlas against N1.7 — recon, from the source

`primary`'s task (2), including its flash-attention question. **Recon only;
nothing in `vla_harness/` was touched.** I read the repository source rather than
the README, and it changes the answer in both directions — the version gap is
*narrower* than the README implies, and there is a **hard blocker the README does
not mention.**

**First, a structural point about the repo.** `action_atlas/api/*` is a **Flask
backend that serves pre-baked JSON** (`experiment_results_<model>.json`, cached
ablation indices). It is a results browser over the paper's precomputed output,
not instrumentation. The code that actually runs models is `experiments/` —
`groot_common.py`, `hooks.py`, `model_adapters.py`. Judge usability from those.

**What `experiments/groot_common.py` supports, in its own words:**

```
N1.5 Eagle: model.backbone.eagle_model.language_model.model.layers[i]
N1.5 DiT:   model.action_head.diffusion_model.transformer_blocks[i] (16 blocks)
N1.6 Eagle: model.backbone.model.language_model.model.layers[i]
N1.6 DiT:   model.action_head.model.transformer_blocks[i] (32 blocks)
```

**N1.6 is supported, and that matters** — it is one version closer than the
README's "N1.5" suggests, and its DiT is **32 blocks, exactly our N1.7's depth**.
My earlier claim that the DiT gap kills layer-indexed taps was **wrong** and is
withdrawn.

| component | their accessor | our N1.7 | verdict |
|---|---|---|---|
| VL-SA | `action_head.vl_self_attention.transformer_blocks` | 4 blocks, 2048 | **likely works unchanged** |
| DiT | `action_head.model.transformer_blocks[i]` (N1.6 path, 32) | 32 layers, 1.092 B | **likely works unchanged** |
| VLM backbone | `backbone.{eagle_model,model}.language_model.model.layers` | Cosmos-Reason2-2B (Qwen3-VL) | **unverified** — Qwen3-VL does expose `.language_model`, so it may partially resolve, but nothing in the repo has tested it |
| input building | `build_eagle_processor()` | `Qwen3VLProcessor` | **hard blocker** |

**Correction to my first pass, after `vla-dd`'s port assessment and my own
source check.** I called the Eagle processor a hard blocker on the whole path.
**That is wrong for model loading.** `ModelAdapter.load_model`
(`experiments/model_adapters.py:437-446`) tries
`GrootPolicy.from_pretrained(checkpoint, strict=False)` and takes
`policy._groot_model` **first**, falling back to their own `load_groot_n15` only
on exception. That primary path is *the same entry point our harness already
uses* — `vla_harness/capture/groot_features.py` resolves
`policy._groot_model.action_head`. So loading an N1.7 checkpoint should work.

**But it is 3 of 5 adapter methods clean, not 4 of 5.** `build_eagle_processor()`
is called **unconditionally at `model_adapters.py:448`**, outside the try/except,
and `run_episode` passes `self.eagle_processor` straight into
`run_groot_episode`, which builds `eagle_*`-prefixed inputs. `run_episode` also
defaults `action_horizon=16`, against N1.7's 40.

| adapter method | transfers to N1.7? |
|---|---|
| `load_model` | **yes** — LeRobot `GrootPolicy` path |
| `setup_suite`, `create_env` | **yes** — LIBERO, not model-specific |
| `get_layer_groups` | **no** — three accessors, below |
| `run_episode` | **no** — pinned to the Eagle processor; `action_horizon` 16 vs 40 |

**The three accessors, verified line-by-line against our vendored
`third_party/lerobot/src/lerobot/policies/groot/groot_n1_7.py`** (I checked these
rather than accept them):

1. **Backbone LM** — theirs `backbone.eagle_model.language_model.model.layers`;
   N1.7 has **no `eagle_model` at all**. Ours is the `language_model` property at
   **`groot_n1_7.py:339-340`**, `getattr(self.model,"model",self.model).language_model`.
2. **VL self-attention** — `action_head.vl_self_attention.transformer_blocks`,
   **identical, 4 blocks in both** (`vl_self_attention_cfg.num_layers: 4`,
   line 128). **Guard before indexing:** it is `nn.Identity()` when
   `num_layers == 0` (**lines 506-509**).
3. **DiT** — theirs `action_head.diffusion_model.transformer_blocks` (16);
   ours `action_head.model.transformer_blocks`, **32** (`num_layers: 32`,
   line 117).

**The DiT finding that matters beyond the port, and it is `vla-dd`'s.** N1.7
instantiates **`AlternateVLDiT`** when `use_alternate_vl_dit` — **which defaults
True** (line 113) — with **`attend_text_every_n_blocks: 2`** (line 114), verified
at lines 472-476. So the DiT attends text only every second block. **Two
consequences.** First, DiT block index is *not semantically comparable* to an
N1.5 block index, so no layer-wise plot inherited from their baked N1.5 results
is a like-for-like axis. Second — and this is new for **§4.4** — N1.7's DiT has a
**built-in period-2 structure in when VL information can arrive**. Any cross-tap
ordering or onset analysis on the DiT must not read that periodicity as evidence
about the failure; it is architecture. That strengthens the §4.4 hazard already
recorded and gives it a concrete, checkable form.

**Reading `AlternateVLDiT.forward` directly — it is period 4, not period 2, and
it is an instrument rather than only a hazard.** `vla-dd` was right that the
text-attending indices must be read off the construction rather than assumed to
be the even blocks. Doing so
(`action_head/cross_attention_dit.py:358-372`) gives a **three-way** structure:

```python
for idx, block in enumerate(self.transformer_blocks):
    if idx % 2 == 1:                      # SELF-ATTENTION ONLY - no VL enters at all
        ...encoder_hidden_states=None...
    else:
        curr_encoder_attention_mask = (
            non_image_attention_mask      # TEXT   <- idx % (2 * attend_text_every_n_blocks) == 0
            if idx % (2 * self.attend_text_every_n_blocks) == 0
            else image_attention_mask     # IMAGE
        )
```

With `attend_text_every_n_blocks = 2`, over N1.7's 32 blocks:

| block indices | what enters |
|---|---|
| 0, 4, 8, 12, 16, 20, 24, 28 (`idx % 4 == 0`) | **TEXT tokens** (8 blocks) |
| 2, 6, 10, 14, 18, 22, 26, 30 | **IMAGE tokens** (8 blocks) |
| all 16 odd indices | **nothing** — pure self-attention |

**The config name is misleading:** text is attended every **fourth** block, not
every second, because the counter runs over even blocks only (`2 * n`). Anyone
reading `attend_text_every_n_blocks: 2` and inferring period 2 — as both
`vla-dd` and I initially did — gets the phase wrong.

**What this gives — an entry schedule, not a pathway decomposition. I
over-claimed and `vla-dd` corrected it.** I wrote that the visual and language
pathways are "pre-separated by architecture" and distinguishable by block index
alone. **That is true of where each modality *enters* and false of what any
block's output *contains*.** `hidden_states` is a **single residual stream**
threaded sequentially through all 32 blocks
(`hidden_states = block(hidden_states, ...)`). Text enters at block 0, image
first at block 2, and **from block 2 onward every block output is a mixture of
both**, which the 16 self-attention blocks then propagate. **Block 0 is the only
point in the network where the stream has seen text and not image.**

So what the architecture supplies free is a **known schedule of indices at which
each modality can *first* influence the stream**. The analysis it supports
cleanly is **first-onset attribution**: does a departure from the reference
manifold first appear immediately after a **text-entry** block or an
**image-entry** block? That is real, cheap and intervention-free, and for that
one question it is the cheapest instrument in this document. **Its power decays
with depth** as mixing accumulates, and it says nothing about attribution for a
departure that first appears at, say, block 19.

**Correct statement of scope:** the entry schedule **replaces patching for
first-onset attribution, and constrains but does not replace it otherwise.**
General "which pathway carries this failure" still needs §4.5.

**And it dictates the statistic.** The phase test must be run on the **phase of
the first departure per episode**, not a phase fitted over the whole depth
profile — a whole-profile fit would pick up the period-4 architectural rhythm in
*both* conditions and look like a result. This is the same trap as §6's
episode-length leakage, one level down.

**And why it is still a hazard, per `vla-dd`, whose framing is right.** The
control must be **structural, not statistical**. Detrending or smoothing a
periodic component out of an onset statistic would also remove any *real*
periodic signal. The null is not "no periodicity" but **"periodicity at exactly
the architectural phase"** — so the test is whether observed onset phase aligns
with the text-attending indices specifically (0, 4, 8, …) as against the
image-attending ones (2, 6, 10, …), which is a far sharper question than
"is there an ordering". **This hazard is N1.7-specific:** N1.5's plain `DiT` has
no such alternation, so the papers' published layer-wise results contain no
equivalent artefact and **cannot be used to calibrate for it**.

**Pricing the capture change this would need (`vla-dd`, verified).** Our taps
record no per-block DiT output. **The return value is not the route:**
`AlternateVLDiT` does build `all_hidden_states` unconditionally, but the
**inference** path inside the denoise loop calls `self.model(...)` *without*
`return_all_hidden_states` (**`groot_n1_7.py:694`**, single return value), while
only the training/forward path passes it (**line 600**, `model_output, _ =`).
Using the return would mean patching `third_party/`, which the capture module
exists specifically not to do — the fork's HEAD is recorded in every run's
`code_state/`, so patching it retroactively invalidates prior runs' provenance.
**The route is per-block forward hooks on
`action_head.model.transformer_blocks[i]`** — the same accessor the port needs —
which attach from outside and fit the existing `attach_groot_capture` pattern.
Small change, **two traps that will silently corrupt the data if missed**:

1. **The blocks fire once per Euler step — 4× per `select_action`**
   (`num_inference_timesteps: 4`, line 134; loop at 683). A naive hook yields
   32 × 4 = 128 tensors per action, **interleaved by step, not by block**. Step
   disambiguation needs the same append-order discipline the existing
   `_post_decoder` tap uses, or block 4 of step 2 is indistinguishable from
   block 4 of step 1.
2. **The `live["on"]` gate must cover these hooks**, or `k_resample > 1` draws
   contaminate every block series exactly as they would the decode path.

Volume at bf16 is roughly 128 × 41 tokens × 1024 ≈ **11 MB per `select_action`**,
tractable if only the 16 VL-attending blocks are kept. **This should be priced
before the capture runs**, since re-running it to add taps costs more than adding
them now. Implementation is `implementor`'s.

**What cannot be reused, and it is the real cost.** Every baked artifact
(`groot_ablation_index.json`, `groot_baseline_index.json`,
`groot_concept_list.json`, `groot_concept_ablation_baked.json`,
`groot_injection_baked.json`, `layer_connections/groot_libero_*.json`) and every
SAE dictionary is N1.5 activations over a 16-block DiT and an Eagle LM. Against
N1.7 these are not stale, they are **meaningless**, and must be regenerated.
**The ablation and SAE-training compute is the expense; the port is not.**
(Correcting myself again: their adapter *does* list `libero_spatial`
— `liorbenhorin-nv/groot-libero_spatial-128_20000`, line 423. My earlier
"their suites exclude spatial" applied to the baked viz data, not the adapter.)

**Net verdict.** Portable as a *vendored analysis layer over our own loader*,
not runnable as-is and not worth installing their stack. Their install would be a
fourth copy of LeRobot that is **not our vendored fork**, so activations from it
would not be attributable to the `code_state` our runs record, and would bypass
the conformance gate. `vla-dd`'s recommended shape is right: lift the
`ModelAdapter` interface and the ablation/perturbation/injection scripts, write a
`GR00TN17Adapter` supplying the three accessors above plus a Qwen3-VL input path,
and feed it policies loaded the way we already load them.

**Caveat on the cheap first step.** A CPU stand-in dry-run of
`get_layer_groups()` is necessary but **not sufficient** — the caveat already
recorded in `groot_features.py` applies: a stand-in defines its own layout, so a
path error can survive a green CPU gate and only surface against the real
checkpoint. Treat a passing dry-run as a precondition, not a verification.

**Ownership:** this is `implementor`'s to build, not mine. `vla-dd` notes the
same assessment reached `implementor` before being routed here, so the port must
not be picked up twice.

---

**Reproduction.** §8.6 is
`/tmp/claude-1000/-home-imerit-Documents-Code-VLA/7213b0d3-eee4-4061-aae9-83476f85ac5e/scratchpad/demo_support2.py`
plus the two follow-up snippets; nothing was written outside scratch and no GPU
was used. It should move into `experiments/` before anyone relies on it.

---

## Provenance

`[R]` marks claims read from search-result summaries rather than full papers;
they need checking before being relied on. Everything about our own stack
(R-030's 0.00 mm, RTC being off, the `ood_selfref` aggregation level, the
unseeded `randn`, episode lengths) was read from the code or the traces and is
cited to the file and line where it lives.
