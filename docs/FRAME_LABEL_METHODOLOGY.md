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

1. **The submodule-localisation gap `primary` assumed we were in is closed.**
   Two 2026 papers do exactly this, cross-architecture, with released code — and
   one of them supports **GR00T N1.5**. §4.5's activation-patching plan is no
   longer a first-of-its-kind build; it is a re-run of published method. (§8.2)
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
| [action-atlas](https://github.com/CWRU-AISM/action-atlas) `[R]` | That paper's toolkit. Grid ablation, vision perturbation, counterfactual prompting, cross-task injection, SAE training, concept ablation/steering. **Supports GR00T N1.5** (plus π0.5, OpenVLA-OFT, SmolVLA, X-VLA). | **10/10** — running code, one minor version off our policy |
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
