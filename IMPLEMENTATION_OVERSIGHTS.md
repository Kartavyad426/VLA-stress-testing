# Implementation oversights — mining layer

Written 2026-09-16, while diagnosing "TRANSPORT over-fires on `libero_object`"
(`HANDOFF_MINING.md`). That investigation found three distinct defects, not one,
and a structural mistake underneath them. The headline bug was the least
important of the four.

This document is separate from `FINDINGS.md` on purpose. `FINDINGS.md` records
things learned about *the policy and the environment*. This records things we
got wrong *in our own code*, and the reasoning error each one came from — those
are worth keeping because the same error is likely to recur elsewhere in the
harness.

---

## O1 — The alphabetical-target bug was fixed in one of the two places it lived

`obj_of_interest` in a BDDL file lists the **manipuland first**. Taking
`sorted(d)[0]` destroys that order. On `['cream_cheese_1', 'basket_1']` it
returns `basket_1` — the *destination*.

This was found and fixed in `mining/signals.py::LiberoSignals._target()`. The
identical line in `mining/phases_libero.py::LiberoPhaseSegmenter._target()` was
left in place, and is still live.

Episodes where alphabetical order disagrees with BDDL order:

| suite | disagree |
|---|---|
| libero_spatial | 0/260 |
| libero_object | **210/260** |
| libero_10 | **110/185** |

**Consequence.** On most `libero_object` episodes the segmenter takes its rest
height `z0`, its lift measurement and its pre-grasp distance against a basket
that never moves. Every phase downstream of those is measured against the wrong
body.

**Why it survived.** `libero_spatial` is the one suite where alphabetical order
happens to be correct (0/260 disagreements), and spatial was the suite that
looked healthiest in every summary table. Its health was a coincidence of
object naming, and it was read as evidence the segmenter was sound.

**The generalisable error:** the same expression was duplicated in two modules.
Fixing the copy that the failing test exercised left the other copy wrong, and
nothing pointed at it because the two modules are never compared.

---

## O2 — The lift route is gated by a condition that lifting prevents

```python
by_lift = closed and (zs - z0) > self.lift_m
```

`closed` is `aperture < CLOSED_M` (0.030 m). But an object held between the
fingers is *exactly what stops the aperture reaching 0.030*. So on any object
thicker than the threshold, the lift route is disabled precisely in the case it
exists to detect.

With the conjunct removed, lift fires on 79/79 `libero_spatial` successes.

**The generalisable error:** two conditions were ANDed because both sounded like
part of "holding". They are not independent — one is anti-correlated with the
other by construction.

---

## O3 — `hold_steps` is shorter than the gripper's own travel time

The `holding` inference has a privileged-state-free route: the policy commanded
the fingers shut but they did not shut, so something is between them. "Did not
shut" is ambiguous — it is true when an object blocks the fingers, and equally
true while the fingers are still *travelling*.

A persistence requirement (`hold_steps=6`) was added to filter the transient,
without measuring how long the transient lasts. Measured, from a close command
to fingers actually closed on **empty air**:

```
steps-to-close, libero_object:  0:1417   5:102   7:145   8:547  ← mode
```

Free travel takes ~8 steps. A 6-step window fits entirely inside it, so the
filter excludes nothing. Sweeping the real segmenter over stored traces shows
the cliff exactly at the travel time:

| hold_steps | libero_object | successes |
|---|---|---|
| 6 (current) | **260/260** | 79/79 |
| 8 | 258/260 | 79/79 |
| 10 | **87/260** | 78/79 |
| 25 | 80/260 | 78/79 |

Flat after 10 — the whole effect, not a tuning slope.

**The generalisable error:** a threshold was introduced to suppress a transient
whose duration was never measured, although the traces needed to measure it were
already on disk.

---

## O4 (structural) — `CLOSED_M` is not portable across suites, so the two routes cover disjoint regimes

Minimum gripper aperture during **successful** episodes:

| suite | median min aperture | meaning |
|---|---|---|
| libero_object | **0.0388** | fingers never reach 0.030 while holding — chunky objects |
| libero_10 | 0.0201 | mixed |
| libero_spatial | **0.0046** | fingers close almost fully while holding — thin objects |

So "commanded closed but fingers still open" is **false while genuinely holding
a thin object**. The gripper-intent route is not universally valid; its validity
depends on whether the grasped object is thicker than `CLOSED_M`.

The two routes were written as redundant backups. They are not. Each is broken
in the regime the other covers:

| suite | by_gripper on successes | by_lift on successes |
|---|---|---|
| libero_spatial | 78/79 | **79/79** ← carries it |
| libero_object | **79/79** ← carries it (but fires on all 260) | 1/79 |
| libero_10 | 52/52 | 16/52 |

### The part worth remembering

`holding = by_gripper or by_lift` is an OR, so the most permissive detector
wins. The stated validity check — **"every success shows TRANSPORT"** — passed
at 79/79, 79/79, 52/52 the entire time, for a *different reason in each suite*,
while both detectors were individually broken. Their failures were
complementary, so the OR concealed them.

**A disjunction of detectors cannot be validated by a check that only asks
whether the disjunction fired.** The check confirmed the OR, and was read as
confirming both branches. To validate a branch you must score that branch alone.

---

## O5 — RESERVED

Reserved for the `visual_grounding` mislabelling finding, pending the taxonomy
decision in `PENDING_DECISIONS.md` #4/#5. Short version: the rule is
`final eef-to-target > 0.10 m`, a POSITIONAL test carrying a SEMANTIC name, and
only 14 of 333 labelled episodes (4.2%) ended near a different named object —
0 of 161 on libero_object. Not written up here yet because the fix renames the
family, and the entry should describe the final name.

---

## O6 — `failure_cost` is bound to a toy-only state key, and is dead on LIBERO

```python
held = r.series("holding")
dropped = bool(held) and any(held) and not held[-1]
```

`holding` is a **toy** state key. The toy env decides it and writes it into
`obs_state`. LIBERO never emits it — it is DERIVED in the segmenter and never
written back. Confirmed: **0 of 705 traces** contain a literal `holding` key.

So `dropped` is always `False`, and `COST_DISRUPTIVE` — the branch for "the
policy grasped the object and then dropped it", the single most consequential
failure a client cares about — **can never fire on LIBERO**:

```
failure_cost over 495 LIBERO failures:   benign 492   None 3   disruptive 0
```

**Why this matters beyond one wrong field.** A manifest row's severity is
defined as `CONDITIONAL x PREVALENCE x COST`. `condition_prevalence` is
deliberately `NOT_ESTIMATED` (it is an artifact of the perturbation grid we
chose, so the client supplies it). That leaves cost as one of only two factors
we compute — and it is a **constant**. A constant times anything is a constant,
so **the severity column is currently uninformative by construction**, and
nothing in the output says so.

**Why it survived.** It fails CLOSED into a plausible value rather than
abstaining. "benign" is a reasonable-looking label; 492 benign failures reads as
a finding rather than as a detector that never ran. Compare G8/LE-2, where
abstention on missing keys is the designed behaviour and did catch the goal-suite
gap honestly.

**The generalisable error — and this is the third time in this document.**
O1 (`sorted()` in one of two modules), O2 (`closed` gating `by_lift`), and now
O6 are all the same shape: **a detector reads a key or condition its environment
does not supply, and degrades into a constant instead of abstaining.** The
original signals-layer bug was this too — the classifier read the toy's
`_gt_ee_to_obj`, `gripper` and `holding` directly, every rule fell through, and
every trace came back `ambiguous` with zero reported abstentions.

The lesson is not "check for this key". It is that **every detector needs a
declared `requires` list and must abstain when it is unmet**, the way
`LiberoPhaseSegmenter.missing()` already does. `failure_cost` has no such
contract, which is why it could silently answer a question it had no data for.

**Fix:** feed it the derived `holding` (or, better, the `_check_grasp` contact
signal from `PENDING_DECISIONS.md` #2), and give it the same abstention contract
as the segmenter — return `cost: None, needs_review: True` when the holding
history is unavailable, rather than defaulting to "not dropped".

---

## Related, not yet acted on

- **`_gt_n_contacts` is captured in every trace and unused.** A direct contact
  count is probably the cleanest privileged holding signal available.
- **The existing mined output is a mix.** `classify()` reads the *fixed*
  `signals.py`, so its distances were right; the phase segments fed into it came
  from the *unfixed* segmenter. Family counts will shift once O1 and O2 land —
  re-mine all 965 and diff rather than assuming the change is small.

---

## Audit note

Every number above was measured on the 965 stored traces from
`runs/camp_20260916-0015_*`, on CPU, without re-running the policy. Mining being
re-runnable over stored traces is what made this diagnosis possible at all;
the only reason O1–O4 were arguable before is that nobody had scored the
branches separately, not that the data was missing.

---

## O7 — The contamination fix itself truncated 10% of instructions

**Where.** `vla_harness/envs/libero_env.py:_clean_instruction()` (as written
2026-09-17, fixed 2026-09-18).

**What it did.** LIBERO-Plus encodes a variant's perturbation in its file name,
and LeRobot passes that name to the policy as the instruction. The fix stripped
the perturbation suffix with `base.split(tok)[0]` for each marker token — the
**first** occurrence, not the last.

Marker words also appear inside scene names. `..._from_table_center_..._table_11`
split at the first `_table_` and the policy was told **"pick up the black bowl
from"**. 240 of 2,402 libero_spatial variants (10%).

**Why it is the interesting kind of bug.** The fix was written to stop the
language being perturbed by accident, and it perturbed the language by accident,
more severely — a truncated command is further out of distribution than a
suffix of junk tokens. Neither a run nor a rendered episode shows it: success
rates just come out lower, and the failures look like ordinary manipulation
failures. It surfaced only when the stripped strings were checked against an
independent list of what the instructions must be.

**Lesson, and the fix's shape.** A string transform whose output has a known
closed set of legal values should be **validated against that set**, not
eyeballed on examples. Here every non-language variant must strip to one of the
40 vanilla LIBERO instructions; that list is shipped inside MINERVA's checkpoint
(`policy_preprocessor.json:task_to_index`). Checking all 8,493 non-language
variants across four suites took one script and caught three distinct defects:
the first-occurrence split, libero_10's `KITCHEN_SCENE3_` prefix, and a `_moved`
marker in libero_goal. See R-028.
