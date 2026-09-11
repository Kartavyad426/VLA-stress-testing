"""L3 -- grouping failures into actionable clusters.

Deliberately simple: group by (family, dominant active factor). Embedding-based
clustering (Cosmos Embed1 etc.) is deferred until there is evidence it separates
anything this does not -- see PLAN.md.
"""
from __future__ import annotations

from collections import defaultdict


def cluster(rollouts) -> list[dict]:
    groups = defaultdict(list)
    for r in rollouts:
        if r.success or not r.diagnosis:
            continue
        fam = r.diagnosis.get("family") or "ambiguous"
        active = sorted([k for k, v in r.perturbation.items() if v != 0])
        groups[(fam, tuple(active))].append(r)

    out = []
    for (fam, knobs), rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        out.append({
            "family": fam,
            "active_knobs": list(knobs),
            "count": len(rs),
            "tasks": sorted({r.task_id for r in rs}),
            "example_rollout_ids": [r.rollout_id for r in rs[:3]],
            "mean_final_error_m": round(sum(
                r.diagnosis.get("final_error_m") or 0 for r in rs) / len(rs), 4),
            "terminals": sorted({r.diagnosis.get("terminal") for r in rs}),
        })
    return out
