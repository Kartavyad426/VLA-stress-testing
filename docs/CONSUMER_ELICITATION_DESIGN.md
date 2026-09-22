# CONSUMER ELICITATION — the question set

*Written 2026-09-16. Design only; implementation belongs to `my primary`.
Expands `docs/TAXONOMY_FAMILY_GUIDELINES.md` §4. Card and interaction mechanics take after
`Tau-harness/scripts/adjudicate_tool.py`, which solves the same structural problem for a different
task.*

---

## 0. The goal, and the one rule

**The consumer produces the taxonomy without ever naming a category.**

They look at real failures and answer questions about what they would *do*. The families are then
**induced** as the equivalence classes of their answers: two failures the consumer would handle the
same way are in the same family, whatever they look like and whatever we would have called them.

> **The rule: never ask "what kind of failure is this?"**
>
> People asked to name categories produce fluent, plausible names — which is the least reliable thing
> they can give us and the hardest to falsify later. Worse, once someone has said "grounding failure"
> out loud, every subsequent answer is anchored to it. Ask about **decisions and money**, which people
> are reliable about because they have consequences.

The one structural borrowing from `adjudicate_tool.py` is the most important: **the card supplies the
"should"**. That tool exists because asking a reader to label an episode cold requires them to hold
the domain in their head. Same problem here. The consumer should never need to know what LIBERO is,
what the policy architecture is, or what we think happened.

---

## 1. What the card shows — and what it must not

**Shows:**

| element | why |
|---|---|
| The instruction, verbatim | *"put the black bowl on the plate"* — this is the "should" |
| Video of the rollout | the evidence; autoplay, loop, scrubbable |
| A still of the **start** state | so they can see what was there to begin with |
| Outcome: **did not complete the task** | flat statement, no elaboration |
| Nothing else | |

**Must NOT show, and each of these is a contamination path we can name:**

- **Our family label.** Anchors everything downstream. This is the whole point.
- **Our phase segmentation, detector outputs, or any `_gt_` state.** Our vocabulary leaks through it.
- **The task's success rate, or how often this failure occurs.** Anchors severity: people rate
  frequent things as more serious regardless of consequence.
- **Any other consumer's answers.** Independence is what makes cross-consumer agreement meaningful.
- **Whether *we* think it is a bug in the harness.** Q4 exists to let them tell us.

---

## 2. The six questions

Ordered deliberately. **Action first, words last** — most elicitation tools do the reverse and
contaminate the answers they most need.

### Q1 — the action question *(the primary signal)*

> **"If your robot did this on your line, what would you do about it?"**

| # | option | latent axis |
|---|---|---|
| 1 | Get more demonstration data → *of what?* (follow-up, §3) | **data gap** |
| 2 | Change the setup — lighting, camera, fixture, how parts arrive | **environment** |
| 3 | Change the task or how it's specified | **task definition** |
| 4 | Nothing — this is acceptable at some rate | **tolerable** |
| 5 | Stop and escalate — this must not happen | **unacceptable** |
| 6 | Something else → free text | **escape hatch** |
| 7 | Can't tell from this | **abstention, and a real answer** |

**This is the taxonomy.** Families are the equivalence classes under this answer. If two failures
both get "get more data, of the bowl-on-ramekin case", they are one family regardless of how
differently they look on video.

*Why this wording:* "on your line" forces a production frame rather than a research one. "What would
you do" is answerable; "what caused it" is not — they cannot know, and asking invites confabulation
(the §5.1 problem in the methods survey, transplanted into a human).

*What would be wrong to ask:* "Is this a perception or a control failure?" — that is our decomposition,
not theirs, and it presumes the answer.

### Q2 — the consequence question

> **"If this happened in production, what does it cost you?"**

| # | option |
|---|---|
| 1 | Nothing — the cell retries and carries on |
| 2 | A wasted cycle — someone has to reset it |
| 3 | Damaged product or tooling |
| 4 | A safety event |
| 5 | Depends on something I can't see here |

Yields the **cost axis**, orthogonal to family — the same shape as our existing
`benign / disruptive / safety`, but *supplied rather than invented*. Two things follow:

- It tells us whether our three levels are the right three. If consumers never pick 3 without also
  meaning 4, we have two levels, not three.
- It is the source of severity weighting. **We do not invent severity** (DG-5b), and this is where it
  legitimately comes from.

*Deliberately not asked:* "rate the severity 1–10". Cardinal severity implies an ordering we cannot
support, and the methods survey found ordering is the quantity that transfers worst from sim to real.

### Q3 — the pairing question *(the partition inducer)*

Two failures side by side, both videos playing.

> **"Would you handle these two the same way?"**
> `Same` · `Different` · `Not sure` — plus, if Different: *"what's the difference that matters?"*

**This is the strongest signal in the whole instrument**, because it asks for a *partition* directly
without anyone naming anything. It is also the most expensive, so pairs are sampled, not exhausted
(§5).

*The free-text on "Different" is the single most valuable field in the tool.* It is where the real
vocabulary comes from — a consumer saying "the first one never even got near it, the second one
dropped it" has just handed us two families and their names, without being asked for either.

### Q4 — the fairness question

> **"Should the robot have been able to do this?"**
> `Yes, that's reasonable` · `No — that setup isn't fair` · `Can't tell`

Catches three things nothing else does:

- **Unsolvable instances** — the C9 concern, and F2's worked example, where a physics change made a
  task impossible and a naive pipeline would have sold a client demonstrations to fix it.
- **Scope disagreements** — cases the consumer considers out of bounds for the robot entirely.
- **Our own harness bugs.** A consumer saying "the camera is pointing at the wrong thing" would have
  caught the yaw re-aim defect from the outside.

### Q5 — the evidence question *(asked once per session, not per card)*

> **"Suppose a report told you this happens in 12% of attempts. What would you need to see before you
> acted on it?"**
> Multi-select: the video · how many times it happened · what conditions trigger it · a suggested fix
> · a comparison against another policy · how confident you are · nothing more

Yields the **manifest's evidence requirements** — what a row must carry to be actionable. Cheap, and
it tells us what to build rather than guessing.

### Q6 — the words question *(last, and optional)*

> **"In your own words — what went wrong here?"** *(free text)*

Asked **last on each card**, after the action is committed, so the wording cannot anchor Q1–Q4.

This is the input to cluster *naming* — never to the partition. The partition comes from Q1 and Q3.
That separation is deliberate: the methods survey found that an LLM (or a person) will name any set
fluently, including a set with no common property, so naming must never be load-bearing for grouping.

---

## 3. Follow-ups that fire conditionally

Keep the main path to four keystrokes; put the depth behind a branch.

- **Q1 → option 1 ("more data"):** *"Demonstrations of what?"* — free text, plus quick-picks lifted
  from the card's own scene (the named objects, "this lighting", "this camera angle", "this starting
  position"). This turns an action into a **data specification**, which is the manifest's actual
  output.
- **Q1 → option 5 ("stop and escalate"):** *"What's the worst case if it keeps doing this?"*
- **Q2 → option 5 ("depends"):** *"On what?"* — reveals missing context in the card design itself.
- **Q4 → "not fair":** *"What's wrong with the setup?"*

---

## 4. Quality control — built in, not bolted on

Without these the output is uninterpretable, and none of them cost the consumer anything they'd notice.

- **Repeat cards.** Show ~10% of cards twice, well separated, with the video re-encoded so it isn't
  visually identical. Yields **intra-rater consistency** — a per-consumer reliability number. This is
  the measurement κ was being asked for and could not give, because it answers "is this person
  self-consistent?" rather than "do two people agree on a rubric?".
- **Planted successes.** A few episodes that actually succeeded, presented identically. Anyone who
  marks them as failures with confident consequences is rubber-stamping.
- **Planted unsolvables.** If we have a MANGO-style case — goal predicate satisfied by accident, or a
  task made impossible by a physics change — it belongs here. Q4 should catch it. If it doesn't, Q4
  isn't working.
- **Randomised card order per consumer**, so order effects don't correlate across people.
- **Distinct storage key per consumer.** `adjudicate_tool.py` carries a scar worth inheriting: two
  pages sharing a `localStorage` key silently overwrote each other's verdicts. One key per
  `(consumer, card set)`.

---

## 5. Sampling — which failures to show

Budget realistically: **20–40 cards per consumer**, 10–15 minutes.

- **Stratify across our current partition** so every family gets seen — but never tell them that is
  what we are doing, and never in blocks.
- **Include the structural cases** the audit found: object never approached; object moved but goal
  not achieved; repeated attempts in one spot. These are the distinctions we most need adjudicated.
- **Nominal episodes only, for now.** The perturbed arms of the current campaign are invalid
  (§5c of the audit) and would show consumers a camera defect rather than a policy failure.
- **Pairs for Q3, sampled two ways:**
  - pairs our clustering calls *same* → tests whether we over-merge
  - pairs our clustering calls *different* → tests whether we over-split

  That second use is worth stating plainly: **Q3 doubles as a validation of our existing partition**,
  not only as a way to induce a new one.

---

## 6. What comes out, and what we compute

Per card, per consumer: Q1 action, Q2 cost, Q4 fairness, Q6 free text. Per pair: Q3 verdict and
difference text. Per session: Q5 evidence needs, plus consistency from the repeats.

Then:

1. **Induced partition** — group cards by (Q1 action, Q1 follow-up). That is the candidate family set,
   derived from decisions, named by nobody.
2. **Cross-consumer agreement on the induced partition.** Consumers disagreeing is a **finding**, not
   noise: it means either orthogonal axes are needed or different consumers need different views. Do
   not average it away.
3. **Agreement between the induced partition and ours.** The concrete test: do our families predict
   their Q1 answers better than chance? A family that does not is decoration, and this is the
   cheapest available version of the utility test — it needs no GPU and no fine-tuning.
4. **The cost distribution per induced family** — client-supplied severity, in the form we can ship.
5. **Vocabulary** from Q6 and the Q3 difference text, used to *name* the induced groups after they
   exist.
6. **Per-consumer reliability** from the repeat cards, reported alongside everything above.

---

## 7. Questions we deliberately do not ask

| not asked | why |
|---|---|
| "What kind of failure is this?" | The whole point. Produces names, not partitions. |
| "Why did it fail?" | They cannot know, and asking invites confabulation. Attribution is stage 5's job and needs intervention, not opinion. |
| "Is this a perception / planning / control failure?" | Our decomposition, presumes the answer, and assumes their system has those stages. |
| "Rate severity 1–10" | Cardinal severity implies an ordering we cannot support and that transfers worst to real robots. |
| "How often do you think this happens?" | We measure prevalence; they supply consequence. Mixing the two loses both. |
| "Would you buy a fix for this?" | A commercial question wearing a technical one's clothes. It will bias toward whatever they think we are selling. |

---

## 8. Mechanics worth inheriting from `adjudicate_tool.py`

- One self-contained card, everything needed to answer on screen, no scrolling between evidence and
  question.
- **"Not sure" is a real answer**, given equal visual weight. An abstention rate is data; a forced
  choice is noise.
- Keyboard-first: `1`/`2`/`3`… to answer, `←`/`→` to move, `next unanswered`.
- A progress strip of per-card pips, colour-coded by answer, clickable.
- Free-text note on every card, placeholder wording that tells the truth: *"why? (optional, but the
  most useful part)"*.
- `localStorage` persistence keyed by a slug, plus **Copy JSON** export — no server, no accounts, and
  it works if you email someone an HTML file.
- Static HTML generated from a template. A consumer should be able to open it offline and send back
  one JSON blob.

**One deliberate divergence.** `adjudicate_tool.py` asks a single question per card; this asks four
plus an optional fifth. That is a real cost to throughput and the reason the budget is 20–40 cards
rather than hundreds. If it proves too slow in practice, cut **Q4** first (it is diagnostic rather
than taxonomic) and keep Q1, Q2 and Q3 — those three carry the partition, the cost axis and the
validation respectively.

---

## 9. Open design questions

1. **How many consumers?** Cross-consumer agreement is the point, so at least three. Fewer than three
   and disagreement is uninterpretable.
2. **Do they see the same cards?** They must, for agreement to be computable — but then card count
   limits total coverage. Suggest a shared core (~20 cards, everyone) plus a rotating tail.
3. **Real consumers or proxies?** If actual buyers are not available, say so in the output. A
   taxonomy elicited from proxies is a hypothesis about buyers, not a measurement of them.
4. **Video length.** Full episodes are up to 300 steps. A consumer will not watch that 40 times.
   Probably needs a trimmed clip around the decisive moment — but *we* choose that moment, which
   re-introduces our localisation as a contamination path. Worth thinking about before building.
