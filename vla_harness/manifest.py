"""L4 -- Data Gap Manifest generation.

Every row is a falsifiable claim with its evidence attached. A row whose
`trigger` has no counterfactual behind it is downgraded to "correlational";
a row with `validation_result: null` is an untested claim and says so. Both
rules exist because overstating an unvalidated row is the fastest way to lose
a client's trust. See PLAN.md section 9.
"""
from __future__ import annotations

import json
from datetime import date

SCHEMA_VERSION = "1.0"

REMEDIATION = {
    "visual_grounding": ("Demonstrations spanning the measured {axis} band in "
                         "even steps, across multiple object layouts and both "
                         "target-object classes."),
    "spatial_reasoning": ("Demonstrations varying target position against "
                          "distractor count and placement; include hard "
                          "negatives where the nearest object is not the target."),
    "manipulation": ("Trajectory variants around contact and alignment across "
                     "the failing {axis} band; include successful corrections."),
    "recovery": ("Failed-attempt-plus-successful-recovery trajectories with "
                 "varied miss direction and contact state."),
    "planning": ("Longer-horizon and subgoal-rich demonstrations covering the "
                 "phase where the policy terminates early."),
    "language_grounding": ("Instruction paraphrases and semantic contrast pairs "
                           "for the affected tasks."),
    "distribution_shift": ("Targeted domain variation sampled around the "
                           "measured failure boundary."),
}


def severity(cluster, total_failures, boundary) -> str:
    share = cluster["count"] / max(1, total_failures)
    if share > 0.30 or (boundary and boundary["upper_rate"] < 0.20):
        return "high"
    return "medium" if share > 0.10 else "low"


def make_row(row_id, cluster, boundary, probe, nominal_rate, n_episodes,
             provenance, supply_side=None, total_failures=1) -> dict:
    axis = (boundary or {}).get("knob") or (
        cluster["active_knobs"][0] if cluster["active_knobs"] else "unknown")

    attributed = probe.get("attributed_knob") if probe else None
    if attributed:
        top = probe["probes"][0]
        trigger = (f"{attributed} — counterfactual probe: reverting this knob "
                   f"alone recovers {top['delta_pp']:+.0f} pp "
                   f"({probe['full_rate']:.0%} → {top['reverted_rate']:.0%})")
        strength = "causal (counterfactual-confirmed)"
    else:
        trigger = f"{axis} — co-occurs with failure; NOT counterfactual-confirmed"
        strength = "correlational"

    return {
        "row_id": row_id,
        "severity": severity(cluster, total_failures, boundary),
        "evidence_strength": strength,
        "task_scope": cluster["tasks"],
        "failure_family": cluster["family"],
        "symptom": (f"{cluster['count']} failures terminating as "
                    f"{'/'.join(t for t in cluster['terminals'] if t)}; "
                    f"mean final error {cluster['mean_final_error_m']*100:.1f} cm"),
        "trigger": trigger,
        "boundary": boundary,
        "evidence": {
            "n_episodes": n_episodes,
            "nominal_success": round(nominal_rate, 4),
            "failures_in_cluster": cluster["count"],
            "example_rollout_ids": cluster["example_rollout_ids"],
        },
        "supply_side": supply_side,
        "coverage_required": REMEDIATION.get(
            cluster["family"], "Targeted coverage of the failing region."
        ).format(axis=axis),
        "validation_plan": (
            "LoRA fine-tune at three escalating data budgets on the coverage "
            "above; re-run the frozen regression set; report the data-response "
            "curve. Neutral and negative results are reportable."),
        "validation_result": None,
        "provenance": provenance,
    }


def render(rows) -> str:
    """Human-readable manifest. The client-facing artifact."""
    L = ["# Data Gap Manifest", "",
         f"Generated {date.today().isoformat()} · schema v{SCHEMA_VERSION}", ""]
    if not rows:
        return "\n".join(L + ["_No failure clusters found._"])
    for r in rows:
        L += [f"## {r['row_id']} — {r['failure_family']} "
              f"[{r['severity'].upper()}]", ""]
        L += [f"- **Evidence strength:** {r['evidence_strength']}"]
        L += [f"- **Tasks:** {', '.join(r['task_scope'])}"]
        L += [f"- **Symptom:** {r['symptom']}"]
        L += [f"- **Trigger:** {r['trigger']}"]
        b = r["boundary"]
        if b:
            L += [f"- **Boundary:** {b['knob']} between {b['lower']:g} and "
                  f"{b['upper']:g} "
                  f"(success {b['lower_rate']:.0%} → {b['upper_rate']:.0%}; "
                  f"definition: {b['definition']})"]
        else:
            L += ["- **Boundary:** not crossed in the swept range"]
        e = r["evidence"]
        L += [f"- **Evidence:** {e['n_episodes']} episodes · nominal "
              f"{e['nominal_success']:.0%} · {e['failures_in_cluster']} failures "
              f"in cluster"]
        if r.get("supply_side"):
            L += [f"- **Supply-side corroboration:** {r['supply_side']}"]
        L += [f"- **Coverage required:** {r['coverage_required']}"]
        L += [f"- **Validation:** {r['validation_plan']}"]
        vr = r["validation_result"]
        L += [f"- **Validation result:** " +
              ("`null` — UNTESTED CLAIM" if vr is None else json.dumps(vr))]
        L += [""]
    return "\n".join(L)
