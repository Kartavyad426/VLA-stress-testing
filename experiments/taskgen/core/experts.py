"""The task expert (checklist E1), simulator-neutral side: turns a task's pass conditions into steps, resolves each
step's target point from scene geometry, and judges the result with the same success rules (docs/TASKGEN_DESIGN.html
§7). The simulator layer executes the motion primitives (sim_mujoco/expert.py): pick(name), move_release(point).

  inside(x, container)   pick x -> release above the container's floor (bowl cavity, or drawer / cabinet level)
  on / stacked(a, b)     pick a -> release just above b's top
  ordered_stack(order)   for each next one: pick it -> release on the previous
  next_to(a, b)          pick a -> release on the table beside b, on the side facing the robot
  out_of(x, container)   pick x -> release on the table in front of the container
One expert per task type, run on every variant. Exploratory, UNVERIFIED.
"""
import numpy as np

from core.geometry import aabb
from core import rules


def steps_for(pass_all):
    out = []
    for c in pass_all:
        if c.type == "inside":
            out.append(("pick", c.object, None)); out.append(("into", c.object, c.target))
        elif c.type in ("on", "stacked"):
            out.append(("pick", c.object, None)); out.append(("onto", c.object, c.target))
        elif c.type == "ordered_stack":
            for lo, hi in zip(c.order, c.order[1:]):
                out.append(("pick", hi, None)); out.append(("onto", hi, lo))
        elif c.type == "next_to":
            out.append(("pick", c.object, None)); out.append(("beside", c.object, c.target))
        elif c.type == "out_of":
            out.append(("pick", c.object, None)); out.append(("beside", c.object, c.target))
        else:
            raise ValueError(f"no expert for rule '{c.type}'")
    return out


def release_point(session, s, kind, obj, target):
    ob = aabb(s.collision_points(session._body(obj)))
    h = float(ob["size"][2])
    if kind == "into":
        t = session._target(s, target)
        if "cavity" in t:
            cv = t["cavity"]
            return np.r_[cv["centre_xy"], cv["floor_z"] + h / 2 + 0.03]
        if "inner" in t:
            c, _, floor = t["inner"]
            return np.r_[c[0], c[1], floor + h / 2 + 0.02]
        raise ValueError(f"'{target}' has no cavity to release into")
    tb = aabb(s.collision_points(session._body(target)))
    if kind == "onto":
        return np.r_[tb["centre"][:2], tb["hi"][2] + h / 2 + 0.005]
    # beside: on the table, towards the robot (-x) from the target, 3 cm gap
    return np.r_[tb["lo"][0] - 0.03 - ob["size"][0] / 2, tb["centre"][1], session.table_z + h / 2 + 0.005]


def run(task, session, expert_cls, record=False, settle_s=1.5):
    """Run the expert for `task` on a fresh build; returns (passed, detail, expert) — detail has per-step logs and rules."""
    s = session.build()
    s.settle(0.5)
    ex = expert_cls(s, record=record)
    logs = []
    try:
        for kind, obj, target in steps_for(task.pass_all):
            if kind == "pick":
                lg = ex.pick(obj)
                logs.append(lg)
                if not lg["held"]:
                    break
            else:
                logs.append(ex.move_release(release_point(session, s, kind, obj, target), label=f"{kind} {target}"))
        ex.settle(settle_s)
    except ValueError as e:
        logs.append(dict(error=str(e)))
    ok, rows = rules.check_all(task.pass_all, session, s)
    return ok, dict(steps=logs, rules=rows, warnings=s.warnings()), ex
