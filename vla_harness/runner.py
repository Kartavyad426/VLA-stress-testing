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
                     make_fingerprint, fingerprint_diff)


def rollout(env, policy, seed: int, spec: PerturbationSpec) -> Rollout:
    """One episode. Knobs are set at reset and frozen for its duration."""
    t0 = time.perf_counter()
    obs = env.reset(seed, spec)
    policy.reset()
    steps, fwd = [], 0

    while True:
        action = policy(obs.policy_view())        # privileged keys stripped
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
        rollout_id=cell_hash(policy.policy_id, env.env_id, env.task_id, seed, spec),
        task_id=env.task_id, instruction=env.instruction,
        policy_id=policy.policy_id, env_id=env.env_id,
        seed=seed, perturbation=spec.as_dict(),
        action_dims=list(policy.action_dims), steps=steps,
        success=success, termination=reason,
        wall_time_s=time.perf_counter() - t0, forward_passes=fwd,
        fingerprint=make_fingerprint(env, policy),
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
             strict: bool = False) -> Cell:
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
        rid = cell_hash(policy.policy_id, env.env_id, env.task_id, sd, spec)
        r = cached.get(rid)

        if r is not None:
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
          store=None, strict=False) -> list[Cell]:
    """Vary ONE axis, hold everything else at its base value."""
    base = base or {}
    return [run_cell(env, policy, PerturbationSpec.of(**{**base, knob: lv}),
                     seeds, store, strict) for lv in levels]


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


def counterfactual_probe(env, policy, spec: PerturbationSpec, seeds,
                         store=None, floor_pp: float = 0.0,
                         min_delta_pp: float = 20.0) -> dict:
    """Attribution. Re-run the SAME seeds reverting one knob at a time.

    Compares DISTRIBUTIONS over `seeds`, not single trajectories -- exact
    trace-level determinism does not survive a real VLA (see ARCHITECTURE.md
    section 5.1). A knob is attributed only when its delta clears both a fixed
    threshold and the measured reproducibility floor, so run-to-run jitter
    cannot be mistaken for a cause.
    """
    full = run_cell(env, policy, spec, seeds, store)
    results = []
    for knob, val in spec.knobs:
        if val == 0:
            continue
        rev = run_cell(env, policy, spec.revert(knob), seeds, store)
        results.append({"knob": knob, "reverted_rate": rev.rate,
                        "delta_pp": (rev.rate - full.rate) * 100})
    results.sort(key=lambda r: -r["delta_pp"])
    threshold = max(min_delta_pp, 3 * floor_pp)
    top = results[0] if results else None
    return {"full_rate": full.rate, "full_spec": spec.label(),
            "probes": results, "n_per_cell": len(seeds),
            "threshold_pp": threshold, "floor_pp": floor_pp,
            "attributed_knob": top["knob"] if top and
                               top["delta_pp"] > threshold else None}
