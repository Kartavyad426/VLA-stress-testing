"""Phase G -- language sensitivity probe.

**This gates a taxonomy family, which is why it runs early.**

LIBERO-Plus reports OpenVLA-OFT "largely unchanged" with *no language input at
all* -- it "degenerates into a form that disregards language, behaving more like
a Vision-Action model". Their goal-replacement probe then shows success dropping
nearly to zero **while the model still executes the ORIGINAL target's
trajectory**.

So apparent language robustness can be INSENSITIVITY rather than comprehension.
If our policy is insensitive, a `language_grounding` family is near-unpopulated,
and a classifier carrying it will either abstain or **quietly label something
else with its name** -- which kappa cannot catch (FINDINGS F6a). The probe
therefore decides whether that family is admissible at all.

Three outcomes. The third is the one a success-rate-only design misses, because
in aggregate it looks like the second.
"""
from __future__ import annotations

from ..runner import rollout
from ..schema import PerturbationSpec

INSENSITIVE = "insensitive"        # blank instruction changes nothing -> a VA model
COMPREHENDING = "comprehending"    # goal swap redirects the trajectory
PARTIAL = "partial_grounding"      # success collapses, trajectory unchanged
INCONCLUSIVE = "inconclusive"


def approached_object(r, window: int = 20) -> str | None:
    """WHICH OBJECT the end-effector went to. The instrument that matters.

    A success-rate delta CANNOT separate comprehension from partial grounding:
    a policy that reads a new instruction and fails, and a policy that ignores
    it and executes the ORIGINAL target, both give success -> 0. In aggregate
    they are identical. Endpoint *displacement* is better but still indirect --
    the endpoint can move for reasons unrelated to target choice.

    Target IDENTITY is the measurement that can land in cell (3) at all.

    Taken as the object closest to the end-effector, averaged over the final
    `window` steps, so a single noisy frame cannot decide it.

    CAVEAT: this runs through the same object-resolution path that failed in
    LE-2, where a substring filter removed articulated objects and
    `_gt_nearest_object` then named the wrong thing. Gate target identity on
    the oracle before trusting a verdict built on it.
    """
    tally = {}
    for s in r.steps[-window:]:
        d = s.obs_state.get("_gt_eef_to_object") or {}
        if s.obs_state.get("_gt_object_pos_complete") is False:
            return None                      # incomplete object list -> abstain
        if d:
            near = min(d, key=d.get)
            tally[near] = tally.get(near, 0) + 1
    return max(tally, key=tally.get) if tally else None


def language_probe(env, policy, seeds, spec: PerturbationSpec | None = None,
                   blank: str = "", substitute=None, unchanged_pp: float = 10.0,
                   store=None) -> dict:
    """Run nominal / blank-instruction / goal-substituted and compare.

    `substitute(instruction) -> str` swaps the target with ONE word changed --
    e.g. "put the black bowl on the plate" -> "put the black bowl in the basket".
    A directed substitution, not a paraphrase: a paraphrase tests robustness,
    a substitution tests whether the words are read at all.
    """
    spec = spec or PerturbationSpec.of()
    orig = env.instruction if hasattr(env, "instruction") else None

    def run(instr):
        rs = []
        for sd in seeds:
            if instr is not None:
                env.instruction_override = instr
            r = rollout(env, policy, sd, spec)
            if store:
                store.append(r)
            rs.append(r)
        return rs

    nominal = run(None)
    blanked = run(blank)
    swapped = run(substitute(orig)) if (substitute and orig) else []
    if hasattr(env, "instruction_override"):
        env.instruction_override = None

    def rate(rs):
        return sum(r.success for r in rs) / len(rs) if rs else None

    r_nom, r_blank = rate(nominal), rate(blanked)
    blank_drop_pp = (r_nom - r_blank) * 100 if None not in (r_nom, r_blank) else None

    out = {"nominal_rate": r_nom, "blank_rate": r_blank,
           "blank_drop_pp": blank_drop_pp, "n_seeds": len(seeds)}

    # Does removing language change anything at all?
    if blank_drop_pp is not None and blank_drop_pp < unchanged_pp:
        out["verdict"] = INSENSITIVE
        out["reason"] = (f"removing the instruction entirely costs "
                         f"{blank_drop_pp:.1f} pp -- the policy is not reading it. "
                         f"A language_grounding family would be unpopulated, and "
                         f"a classifier carrying it may mislabel instead.")
        out["language_family_admissible"] = False
        return out

    if not swapped:
        out["verdict"] = INCONCLUSIVE
        out["reason"] = "blank instruction matters, but no substitution probe run"
        return out

    # It reads the instruction -- but does it FOLLOW it, or merely need one?
    # Decided on TARGET IDENTITY, not on the success bit.
    r_swap = rate(swapped)
    pairs = [(approached_object(a), approached_object(b))
             for a, b in zip(nominal, swapped)]
    usable = [(x, y) for x, y in pairs if x and y]
    if not usable:
        out.update({"swapped_rate": r_swap, "verdict": INCONCLUSIVE,
                    "reason": "could not resolve target identity (incomplete "
                              "object list, or no privileged object state)"})
        return out
    redirected = sum(1 for x, y in usable if x != y) / len(usable)
    out.update({"swapped_rate": r_swap, "n_identity_resolved": len(usable),
                "redirect_fraction": round(redirected, 3),
                "instrument": "target identity (object approached), not success rate"})

    if redirected >= 0.5:
        out["verdict"] = COMPREHENDING
        out["reason"] = (f"substituting the goal redirected the end-effector to a "
                         f"different object in {redirected:.0%} of paired episodes "
                         f"-- the policy followed the new instruction")
        out["language_family_admissible"] = True
    else:
        out["verdict"] = PARTIAL
        out["reason"] = (f"success fell to {r_swap:.0%} but the end-effector still "
                         f"approached the ORIGINAL object in {1-redirected:.0%} of "
                         f"paired episodes. Looks like comprehension in aggregate "
                         f"and is not -- a success-rate design would misread this.")
        out["language_family_admissible"] = False
    return out


def substitute_from_scene(env, instruction: str) -> str | None:
    """Directed substitution: name a DIFFERENT object that IS in the scene.

    Not a paraphrase. LIBERO-Plus's shipped 1,537 "Language Instructions"
    instances are LLM rewrites of the SAME goal -- they test surface-form
    robustness, which is a different question and would read as a grounding
    result if used here. The probes that separate insensitivity from
    comprehension are not in the corpus; we generate them.

    Uses the BDDL object list, so the named alternative is guaranteed present
    and reachable rather than a hallucinated distractor.
    """
    names, complete = (env._object_body_names() if hasattr(env, "_object_body_names")
                       else ([], False))
    if len(names) < 2:
        return None
    def pretty(n):
        return n.replace("_main", "").replace("_1", "").replace("_", " ").strip()
    alt = pretty(names[-1])
    if alt and alt.split()[0].lower() not in instruction.lower():
        return f"{instruction.rstrip('.')} -- instead, pick up the {alt}"
    return None
