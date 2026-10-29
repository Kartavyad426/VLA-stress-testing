"""Success rules on the final state (docs/TASKGEN_DESIGN.html §5; checklist E4), simulator-neutral: every rule reads
object geometry through the Session (collision points from the simulator) and the placement logic's relation checks.

  inside(object, target)      target: a bowl-like container (cavity), a drawer, or a cabinet level "cabinet_1/middle"
  out_of(object, target)      not inside
  on(object, target)          resting on top of the target (centre over its footprint, bottom at its top)
  stacked(object, target)     same as on
  ordered_stack(order)        order[0] on the table, each next one on the previous (bottom to top)
  next_to(object, target)     on the table, horizontal gap 0.5-8 cm (make_tasks' rule)
evaluate() returns (ok, detail) for one condition; check_all() for a list (all must hold).
Exploratory, UNVERIFIED.
"""
from core.geometry import aabb, horizontal_gap
from core import place as P

NEXT_TO = (-0.005, 0.08)
ON_TABLE = 0.01


def _box(session, s, name):
    return aabb(s.collision_points(session._body(name)))


def _on(session, s, a, b):
    ba, bb = _box(session, s, a), _box(session, s, b)
    c = ba["centre"]
    over = bb["lo"][0] <= c[0] <= bb["hi"][0] and bb["lo"][1] <= c[1] <= bb["hi"][1]
    resting = ba["lo"][2] > bb["hi"][2] - 0.008
    return bool(over and resting), dict(centre_over_target=bool(over), bottom_cm=round(100 * float(ba["lo"][2]), 1),
                                        target_top_cm=round(100 * float(bb["hi"][2]), 1))


def evaluate(cond, session, s):
    t = cond.type
    if t in ("inside", "out_of"):
        tgt = session._target(s, cond.target)
        if tgt.get("closed"):  # a closed drawer: inside it = within its full inner box
            from core.objects import Cabinet  # noqa: F401
            a = session.cfg.added_objects[cond.target.split("/")[0]]
            cab = session._neutral(a)
            lv = cond.target.split("/")[1]
            tgt["inner"] = cab.inner(lv, session.table_z, shift=s.joint(cab.body_names()["joints"][lv]), exposed=False)
        ok, why = P.holds("inside", _box(session, s, cond.object), tgt, session.table, None)
        if t == "out_of":
            return (not ok), dict(inside=ok)
        return ok, dict(reason=why)
    if t in ("on", "stacked"):
        return _on(session, s, cond.object, cond.target)
    if t == "ordered_stack":
        order = cond.order or []
        if len(order) < 2:
            return False, dict(reason="ordered_stack needs at least 2 objects")
        b0 = _box(session, s, order[0])
        steps = [dict(object=order[0], on_table=bool(b0["lo"][2] < session.table_z + ON_TABLE))]
        ok = steps[0]["on_table"]
        for lo, hi in zip(order, order[1:]):
            o, d = _on(session, s, hi, lo)
            steps.append(dict(object=hi, on=lo, ok=o, **d))
            ok = ok and o
        return bool(ok), dict(steps=steps)
    if t == "next_to":
        ba, bb = _box(session, s, cond.object), _box(session, s, cond.target)
        g = horizontal_gap(ba, bb)
        on_table = ba["lo"][2] < session.table_z + ON_TABLE
        return bool(NEXT_TO[0] < g < NEXT_TO[1] and on_table), dict(gap_cm=round(100 * g, 1), on_table=bool(on_table))
    return False, dict(reason=f"unknown rule '{t}'")


def check_all(conds, session, s):
    rows = []
    for c in conds:
        ok, d = evaluate(c, session, s)
        rows.append(dict(rule=c.model_dump(), ok=bool(ok), detail=d))
    return all(r["ok"] for r in rows), rows
