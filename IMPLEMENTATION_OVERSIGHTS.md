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
