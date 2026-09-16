"""Rollout, sweep and counterfactual probe.

The three operations the whole project rests on:

  rollout(env, policy, seed, spec)     one episode -> one Rollout
  sweep(...)                           vary ONE axis  -> robustness curve
  counterfactual_probe(...)            revert ONE knob -> attribution

Note the difference between the last two. The sweep holds everything at nominal
and varies one axis, and answers "where does it break". The probe starts from a
FAILING fully-perturbed cell and reverts one knob, and answers "which of these
things broke it". Both are needed; they are not the same operation.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .schema import (Rollout, Step, PerturbationSpec, TraceStore, cell_hash,
                     make_fingerprint, fingerprint_diff, semantic_runtime,
                     keyed_part)


def _sem(env) -> dict:
    """KNOWN-semantic externals this env depends on -- part of the cache key."""
    return getattr(env, "semantic_deps", lambda: semantic_runtime())()


def _cell_id(env, policy, seed, spec) -> str:
    return cell_hash(policy.policy_id, env.env_id, env.task_id, seed, spec,
                     _sem(env))


def rollout(env, policy, seed: int, spec: PerturbationSpec) -> Rollout:
    """One episode. Knobs are set at reset and frozen for its duration."""
    t0 = time.perf_counter()
    obs = env.reset(seed, spec)
    policy.reset()
    steps, fwd = [], 0

    while True:
        # B6/DG-6: a PrivilegedProbePolicy opts in to full state for the
        # §7c discriminator-4 ablation. Everything else gets the stripped view,
        # and policy_view() itself is unchanged -- a flag on the strict path
        # would reintroduce the convention-vs-enforcement weakness that caused
        # the original `_gt_` leak.
        action = policy(obs if getattr(policy, "privileged", False)
                        else obs.policy_view())
        fwd += 1
        steps.append(Step(t=obs.t, obs_state=dict(obs.state),
                          action=list(action.values),
                          image_refs=dict(obs.image_refs)))
        obs, success, done, reason = env.step(action)
        if done:
            break

    # The terminal observation -- the state in which success was DECIDED. No
    # action was ever requested for it, hence action=None. Without this the
    # trace ends one step early and `holding` is never True in any rollout,
    # so the very moment the task was completed is the one moment not recorded.
    steps.append(Step(t=obs.t, obs_state=dict(obs.state), action=None,
                      image_refs=dict(obs.image_refs)))

    return Rollout(
        rollout_id=_cell_id(env, policy, seed, spec),
        task_id=env.task_id, instruction=env.instruction,
        policy_id=policy.policy_id, env_id=env.env_id,
        seed=seed, perturbation=spec.as_dict(),
        action_dims=list(policy.action_dims), steps=steps,
        success=success, termination=reason,
        wall_time_s=time.perf_counter() - t0, forward_passes=fwd,
        fingerprint=make_fingerprint(env, policy),
        scene_descriptor=(env.scene_descriptor()
                          if hasattr(env, "scene_descriptor") else {}),
        privileged=getattr(policy, "privileged", False),
    )


# --- statistics --------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI for a proportion. Wilson, not normal-approx: at n=20 with k near
    0 or n the normal interval is badly wrong, and screening cells are n=20."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


@dataclass
class Cell:
    """One point on a robustness curve: N episodes at one knob setting."""
    spec: PerturbationSpec
    n: int
    successes: int
    rollout_ids: list[str]
    # resume accounting -- a stale cache and a real finding look identical from
    # outside, so the split has to be visible in the summary, not a log line.
    n_cached: int = 0
    n_fresh: int = 0
    stale_hits: list[dict] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.successes / self.n if self.n else 0.0

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_ci(self.successes, self.n)

    def __repr__(self):
        lo, hi = self.ci
        return (f"{self.spec.label():<34} {self.rate:6.1%}  "
                f"[{lo:5.1%},{hi:5.1%}]  n={self.n}")


def run_cell(env, policy, spec, seeds, store: TraceStore | None = None,
             strict: bool = False, arms=None, arm: str = "uniform") -> Cell:
    """G10: resumable, and the resume is VERIFIED rather than trusted.

    Two distinct failure modes are guarded here:

    * A cached rollout must still CONTRIBUTE its outcome -- skipping the compute
      is correct, skipping the result made every resumed cell read 0% success.
    * A cache hit must be PROVEN reusable. Ids derive from `identity()`, so a
      changed threshold or checkpoint normally misses. The stored fingerprint is
      the belt to that braces: it catches an identity() that forgot a field,
      which is the failure mode that survives everything else.

    On mismatch: re-run, and record it. Raising by default would kill an
    overnight sweep at hour 8 over a moved threshold, and under session caps
    someone would then disable the check -- a disabled check is worse than none.
    `strict=True` raises instead, for client-facing runs.
    """
    ok, ids, cached_n, fresh_n, stale = 0, [], 0, 0, []
    cached = store.by_id() if store else {}
    current_fp = make_fingerprint(env, policy)

    for sd in seeds:
        rid = _cell_id(env, policy, sd, spec)
        if arms is not None:
            arms.record(arm, rid, spec, sd)          # A1: request, not episode
        r = cached.get(rid)

        if r is not None:
            # only the REPORTED (non-keyed) part can differ on a hit: a keyed
            # change alters rollout_id and misses the cache by construction.
            diff = fingerprint_diff(r.fingerprint or {}, current_fp)
            if diff:
                rec = {"rollout_id": rid, "seed": sd, "diff": diff}
                if strict:
                    raise RuntimeError(
                        f"stale cache hit {rid}: fingerprint mismatch on "
                        f"{sorted(diff)} (strict mode)")
                stale.append(rec)
                r = None                       # fall through and re-run

        if r is None:
            r = rollout(env, policy, sd, spec)
            fresh_n += 1
            if store:
                store.append(r)
        else:
            cached_n += 1

        ok += r.success
        ids.append(r.rollout_id)

    return Cell(spec, len(seeds), ok, ids, n_cached=cached_n, n_fresh=fresh_n,
                stale_hits=stale)


def sweep(env, policy, knob: str, levels, seeds, base=None,
          store=None, strict=False, arms=None, arm="uniform") -> list[Cell]:
    """Vary ONE axis, hold everything else at its base value.

    This is the UNIFORM arm: a fixed grid, unbiased, and therefore the only
    source frequency statistics may be computed from (PLAN.md 5.1).
    """
    base = base or {}
    return [run_cell(env, policy, PerturbationSpec.of(**{**base, knob: lv}),
                     seeds, store, strict, arms, arm) for lv in levels]


def uniform_frequency(rollouts, arm_log, arm: str = "uniform") -> dict:
    """A1/DG-2 -- frequency computed over cells the UNIFORM ARM REQUESTED.

    NOT over stored-rollout tags. A rollout the adaptive arm happened to run
    first is still a legitimate member of the uniform sample if the uniform arm
    asked for that cell -- the physics is identical. Filtering on who stored it
    would silently shrink the uniform sample in adaptive search order, which is
    the corruption this whole mechanism exists to prevent (I9).
    """
    requested = arm_log.requested_by(arm)
    members = [r for r in rollouts if r.rollout_id in requested]
    fails = [r for r in members if not r.success]
    return {"arm": arm, "n_requested": len(requested), "n_resolved": len(members),
            "n_failures": len(fails),
            "rate": (len(fails) / len(members)) if members else None,
            "member_ids": {r.rollout_id for r in members}}


def resume_summary(cells) -> dict:
    """Resumed-vs-fresh accounting for the run summary.

    Belongs in the summary, not the log: §6's "a nominal cell below ~90% is
    nearly always a harness bug" heuristic is unusable if you cannot see
    whether the numbers came from this run or a previous one.
    """
    cached = sum(c.n_cached for c in cells)
    fresh = sum(c.n_fresh for c in cells)
    stale = [h for c in cells for h in c.stale_hits]
    fields = sorted({k for h in stale for k in h["diff"]})
    return {"cached": cached, "fresh": fresh, "total": cached + fresh,
            "stale_hits": len(stale), "stale_fields": fields,
            "examples": stale[:3]}


def format_resume_summary(s: dict) -> str:
    pct = (100 * s["cached"] / s["total"]) if s["total"] else 0
    lines = [f"  episodes: {s['total']}  ({s['fresh']} fresh, "
             f"{s['cached']} from cache = {pct:.0f}%)"]
    if s["stale_hits"]:
        lines.append(f"  STALE CACHE HITS: {s['stale_hits']} re-run — "
                     f"fingerprint changed on {', '.join(s['stale_fields'])}")
        for e in s["examples"]:
            for k, v in e["diff"].items():
                lines.append(f"      {k}: cached={v['cached']!r} "
                             f"current={v['current']!r}")
    return "\n".join(lines)


def find_boundary(cells: list[Cell], knob: str, frac: float = 0.5):
    """The knob value where success first falls below `frac` of nominal.

    Reported as a bracket, not a point: with n=20 per cell we can honestly say
    'between L2 and L3', not 'at 13.4 degrees'. Interpolating a precise number
    from noisy cells would be false precision.
    """
    if not cells:
        return None
    nominal = cells[0].rate
    thresh = nominal * frac
    for prev, cur in zip(cells, cells[1:]):
        if cur.rate < thresh <= prev.rate:
            return {"knob": knob, "threshold": thresh, "nominal": nominal,
                    "lower": prev.spec.as_dict().get(knob, 0.0),
                    "upper": cur.spec.as_dict().get(knob, 0.0),
                    "lower_rate": prev.rate, "upper_rate": cur.rate,
                    "definition": f"success < {frac:g} x nominal ({nominal:.1%})"}
    return None


def reproducibility_floor(env, policy, spec, seeds, repeats=2) -> dict:
    """Measure run-to-run variance at a FIXED knob setting.

    On the toy this is exactly 0. On a real VLA it will not be: GPU float
    nondeterminism, stochastic action heads (flow/diffusion) and contact chaos
    all mean identical inputs can give different outcomes. This measures how
    much, so an attribution delta can be compared against the noise it must
    beat. Run it once per (policy, env) before trusting any probe.
    """
    # NO STORE, DELIBERATELY. Threading a store through here would serve the
    # second repeat from the first repeat's cache and report floor_pp = 0.0 by
    # construction -- a plausible-looking optimisation that silently destroys
    # the only measurement that tells us whether a probe delta is real.
    rates = [run_cell(env, policy, spec, seeds, store=None).rate
             for _ in range(repeats)]
    return {"rates": rates, "floor_pp": (max(rates) - min(rates)) * 100,
            "n_per_run": len(seeds), "repeats": repeats}


def mcnemar(pairs) -> dict:
    """C7 -- paired test on per-seed outcomes.

    We run the SAME seeds in both arms, so outcomes are paired and comparing
    rate-to-rate discards that. Only the discordant pairs carry information:
    seeds that flip. Exact binomial two-sided test on b vs c.

    Free variance reduction, and it relieves the probe's episode budget.
    """
    b = sum(1 for f, r in pairs if not f and r)      # failed full, passed revert
    c = sum(1 for f, r in pairs if f and not r)      # passed full, failed revert
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "discordant": 0, "p_value": 1.0,
                "note": "no seed changed outcome"}
    from math import comb
    k = min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"b": b, "c": c, "discordant": n, "p_value": round(p, 5)}


# --- attribution verdicts (5.1) -------------------------------------------
# These were ONE string before. A genuine null, an interaction and multiple
# sufficient causes are three different claims and were rendering identically,
# which is the bug -- not the arithmetic.
ATTR_SINGLE = "single_factor"          # one knob, reverting it recovers
ATTR_SUBSET = "minimal_subset"         # irreducible set; no member suffices alone
ATTR_MULTIPLE = "multiple_sufficient"  # several knobs each independently cause it
ATTR_NONE = "no_factor_identified"     # nothing reverted recovers -- genuinely null


def ddmin_factors(run_fn, factors, floor_pp=0.0, min_delta_pp=20.0) -> dict:
    """Delta debugging -- the MINIMAL failure-inducing subset.

    `run_fn(active_factors) -> success_rate`.

    The distinction that matters, and that revert-one-knob collapses:

      SUFFICIENCY  does this factor ALONE cause failure?   -> run_fn([f])
      NECESSITY    does removing it fix things?            -> run_fn(rest)

    Reverting one at a time answers necessity only. In a conjunctive failure
    (A and B jointly required) BOTH are necessary, so both reversions recover
    and it looks like two independent causes -- when in fact neither is
    sufficient alone. Testing each factor ALONE separates the cases.

    ddmin then narrows to an irreducible set in O(n^2) runs rather than 2^n.
    Every run is a same-seed rollout we can already do.
    """
    thresh = max(min_delta_pp, 3 * floor_pp)
    nominal = run_fn([])
    base = run_fn(list(factors))

    def fails(active):
        """Does this subset drop success by more than the noise floor allows?"""
        return (nominal - run_fn(list(active))) * 100 >= thresh

    if not fails(list(factors)):
        return {"verdict": ATTR_NONE, "nominal_rate": nominal,
                "all_active_rate": base, "threshold_pp": thresh,
                "reason": "activating every factor does not drop success -- "
                          "the failure is not attributable to these factors"}

    # SUFFICIENCY: each factor alone
    alone = {f: run_fn([f]) for f in factors}
    sufficient = [f for f in factors if (nominal - alone[f]) * 100 >= thresh]
    deltas = {f: round((nominal - r) * 100, 1) for f, r in alone.items()}

    if len(sufficient) == 1:
        return {"verdict": ATTR_SINGLE, "factor": sufficient[0],
                "alone_deltas_pp": deltas, "threshold_pp": thresh}
    if len(sufficient) > 1:
        return {"verdict": ATTR_MULTIPLE, "factors": sufficient,
                "alone_deltas_pp": deltas, "threshold_pp": thresh,
                "note": "each independently sufficient -- these are separate "
                        "causes, not a set; remediating one leaves the others"}

    # No factor suffices alone, yet together they fail => INTERACTION. Narrow it.
    cur, n = list(factors), 2
    while len(cur) > 1:
        chunks = [cur[i::n] for i in range(n) if cur[i::n]]
        moved = False
        for c in chunks:
            if fails(c):
                cur, n, moved = c, 2, True
                break
        if not moved:
            for c in chunks:
                comp = [x for x in cur if x not in c]
                if comp and fails(comp):
                    cur, n, moved = comp, max(n - 1, 2), True
                    break
        if not moved:
            if n >= len(cur):
                break
            n = min(len(cur), 2 * n)
    return {"verdict": ATTR_SUBSET, "subset": cur, "alone_deltas_pp": deltas,
            "threshold_pp": thresh,
            "note": "irreducible -- no member causes failure alone. This is an "
                    "INTERACTION, which revert-one-knob reports as 'no factor "
                    "responsible'. Remediating any single member will not fix it."}


def counterfactual_probe(env, policy, spec: PerturbationSpec, seeds,
                         store=None, floor_pp: float = 0.0,
                         min_delta_pp: float = 20.0, arms=None) -> dict:
    """Attribution. Re-run the SAME seeds reverting one knob at a time.

    Compares DISTRIBUTIONS over `seeds`, not single trajectories -- exact
    trace-level determinism does not survive a real VLA (see ARCHITECTURE.md
    section 5.1). A knob is attributed only when its delta clears both a fixed
    threshold and the measured reproducibility floor, so run-to-run jitter
    cannot be mistaken for a cause.
    """
    def outcomes(sp):
        """Per-seed success, in seed order -- the pairing C7 needs."""
        cached = store.by_id() if store else {}
        out = []
        for sd in seeds:
            rid = _cell_id(env, policy, sd, sp)
            r = cached.get(rid)
            if r is None:
                r = rollout(env, policy, sd, sp)
                if store:
                    store.append(r)
            if arms is not None:
                arms.record("probe", rid, sp, sd)
            out.append(bool(r.success))
        return out

    full_out = outcomes(spec)
    full = Cell(spec, len(seeds), sum(full_out), [])
    results = []
    for knob, val in spec.knobs:
        if val == 0:
            continue
        rev_spec = spec.revert(knob)
        rev_out = outcomes(rev_spec)
        rev = Cell(rev_spec, len(seeds), sum(rev_out), [])
        results.append({"knob": knob, "reverted_rate": rev.rate,
                        "delta_pp": (rev.rate - full.rate) * 100,
                        "paired": mcnemar(list(zip(full_out, rev_out)))})
    results.sort(key=lambda r: -r["delta_pp"])
    threshold = max(min_delta_pp, 3 * floor_pp)

    # --- attribution -------------------------------------------------------
    # ddmin now runs WHENEVER there are >=2 active factors.
    #
    # It used to run only when no single reversion cleared the threshold, on
    # the reasoning that a clear winner needed no further work. That was wrong,
    # and the campaign showed it: reverting camera yaw recovered +65 pp, so the
    # short-circuit fired and reported `single_factor` -- while `ee_offset` was
    # never tested ALONE. Its measured "+0.0 pp" was taken with camera yaw
    # still active and success already at the floor, where no reversion except
    # the dominant one can move anything. That is MASKING, not absence.
    #
    # The two tests answer different questions and only one of them is the
    # question we report:
    #     revert one  -> NECESSITY  (does removing it help?)  -- masked by a
    #                    dominant co-factor
    #     alone       -> SUFFICIENCY (does it cause failure by itself?) -- not
    #                    masked, and this is what `single_factor` claims
    #
    # LIBERO-Plus reports robot initial state as the second most damaging axis
    # across all ten checkpoints they tested; our short-circuited probe scored
    # it at zero. That disagreement is what surfaced this.
    attribution = None
    active = [k for k, v in spec.knobs if v]
    if len(active) >= 2:
        vals = dict(spec.knobs)

        def run_fn(on):
            sp = PerturbationSpec.of(**{k: (vals[k] if k in on else 0.0)
                                        for k in active})
            return run_cell(env, policy, sp, seeds, store).rate

        attribution = ddmin_factors(run_fn, active, floor_pp, min_delta_pp)
        # Reversion deltas are still reported, but they measure necessity and
        # are unreliable when the full perturbation has already bottomed out.
        attribution["reversion_deltas_pp"] = {
            r["knob"]: round(r["delta_pp"], 1) for r in results}
        attribution["reversions_masked"] = full.rate <= 0.05
        if attribution["reversions_masked"]:
            attribution["masking_note"] = (
                f"full perturbation sits at {full.rate:.0%}; reversion deltas "
                f"for non-dominant factors are uninformative at this floor. "
                f"The verdict above comes from ALONE tests, which are not "
                f"masked.")
    top = results[0] if results else None
    return {"full_rate": full.rate, "full_spec": spec.label(),
            "probes": results, "n_per_cell": len(seeds),
            "threshold_pp": threshold, "floor_pp": floor_pp,
            "attributed_knob": top["knob"] if top and
                               top["delta_pp"] > threshold else None,
            # No fallback verdict. With <2 active factors there is nothing to
            # attribute BETWEEN, and inventing `single_factor` from a reversion
            # delta is the bug this section documents.
            "attribution": attribution}
