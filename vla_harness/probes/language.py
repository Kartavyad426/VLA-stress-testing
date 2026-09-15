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


def _endpoint(r):
    """Where the end-effector finished -- the trajectory's verdict."""
    s = r.steps[-1].obs_state
    return s.get("eef_pos") or s.get("ee_xy")


def _dist(a, b):
    if not a or not b:
        return None
    n = min(len(a), len(b))
    return sum((a[i] - b[i]) ** 2 for i in range(n)) ** 0.5


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

    # It reads the instruction -- but does it FOLLOW it, or just need one present?
    r_swap = rate(swapped)
    moved = [d for d in (_dist(_endpoint(a), _endpoint(b))
                         for a, b in zip(nominal, swapped)) if d is not None]
    mean_move = sum(moved) / len(moved) if moved else None
    out.update({"swapped_rate": r_swap, "mean_endpoint_shift_m": mean_move})

    if mean_move is not None and mean_move > 0.05:
        out["verdict"] = COMPREHENDING
        out["reason"] = (f"substituting the goal moved the endpoint "
                         f"{mean_move:.3f} m -- the policy followed the new target")
        out["language_family_admissible"] = True
    else:
        out["verdict"] = PARTIAL
        out["reason"] = (f"success fell to {r_swap:.0%} but the endpoint moved only "
                         f"{mean_move if mean_move is None else round(mean_move,3)} m "
                         f"-- the policy still went to the ORIGINAL target. This "
                         f"looks like comprehension in aggregate and is not.")
        out["language_family_admissible"] = False
    return out
