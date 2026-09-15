"""C9 -- the environment control, run PER TASK.

PLAN.md §7c discriminator 1: *can a competent controller still solve this task?*
If not, the perturbation broke the TASK, not the policy, and any failure
attributed to the policy is an artifact.

**Per task, not per suite.** F2 is the worked example: a MuJoCo point release
took one LIBERO task from 80% to 28%. At suite level that is ~5 pp diluted
across ten tasks -- invisible. At task level it is unmissable. A control that
aggregates is a control that cannot catch the thing it exists for.
"""
from __future__ import annotations

from .runner import run_cell
from .schema import PerturbationSpec

# Below this, a task is not reliably solvable and results on it are suspect.
SOLVABLE_RATE = 0.60


def control_per_task(make_env, reference_policy, task_ids, seeds,
                     spec: PerturbationSpec | None = None, store=None,
                     threshold: float = SOLVABLE_RATE) -> dict:
    """Run a reference (known-competent) policy on each task.

    `make_env(task_id) -> Env`. `reference_policy` is whatever we trust to solve
    the nominal task -- a scripted expert, a replayed demonstration, or a policy
    with an established baseline.

    Returns a per-task verdict. A task that fails the control is **excluded from
    policy conclusions and reported as excluded** -- not silently dropped, and
    not quietly averaged in.
    """
    spec = spec or PerturbationSpec.of()
    out, excluded = {}, []
    for tid in task_ids:
        env = make_env(tid)
        cell = run_cell(env, reference_policy, spec, seeds, store)
        ok = cell.rate >= threshold
        lo, hi = cell.ci
        out[str(tid)] = {"rate": round(cell.rate, 4),
                         "ci95": [round(lo, 4), round(hi, 4)],
                         "n": cell.n, "solvable": ok}
        if not ok:
            excluded.append(str(tid))
    return {"threshold": threshold, "spec": spec.label(),
            "per_task": out, "excluded_tasks": excluded,
            "n_excluded": len(excluded), "n_tasks": len(task_ids),
            "verdict": ("all tasks solvable" if not excluded else
                        f"{len(excluded)} task(s) NOT solvable by the reference "
                        f"policy -- exclude from policy conclusions, and report "
                        f"the exclusion")}


def assert_control_before_conclusions(control: dict, task_id) -> None:
    """Raise rather than let an excluded task reach a manifest row."""
    t = control.get("per_task", {}).get(str(task_id))
    if t is None:
        raise RuntimeError(f"task {task_id} has no environment control -- "
                           f"run control_per_task before drawing conclusions")
    if not t["solvable"]:
        raise RuntimeError(
            f"task {task_id} failed the environment control "
            f"(reference policy {t['rate']:.0%} < {control['threshold']:.0%}). "
            f"A failure here is an artifact of the environment, not the policy.")
