"""Mode A executor: a filled template (SceneRequest) -> Session tool calls, in order (docs/TASKGEN_DESIGN.html §4a).
Plain code, no model. Stops at the first object that fails and reports why; the task is checked against the
scene's objects (its rule is evaluated in stage 3). Exploratory, UNVERIFIED.
"""
from core.templates import SceneRequest

SOURCE = {"box": "primitive:box", "drawer": "primitive:drawer"}


def run(session, req: SceneRequest, allow_resize=True):
    rep = dict(steps=[], ok=True)
    if req.infeasible:
        rep.update(ok=True, infeasible=req.infeasible, note="the request was declared infeasible; nothing built")
        return rep
    for o in req.add:
        src = SOURCE.get(o.source) or f"pack:{o.noun or ''}"
        r = session.add_object(src, colour=o.colour, size_cm=o.size_cm, name=o.name)
        rep["steps"].append(dict(call="add_object", name=o.name, result=r))
        if "error" in r:
            rep.update(ok=False, failed=o.name, reason=r["error"])
            return rep
        r = session.place(o.name, o.relation, o.target, allow_resize=allow_resize)
        rep["steps"].append(dict(call="place", name=o.name, result=r))
        if "placed" not in r:
            rep.update(ok=False, failed=o.name, reason=r.get("infeasible") or r.get("error"))
            return rep
    if req.task:
        known = set(session.base_bodies) | {n for n, a in session.cfg.added_objects.items() if a.pose is not None}
        names = set(req.task.targets) | {x for x in (req.task.pass_condition.object, req.task.pass_condition.target) if x}
        names |= set(req.task.pass_condition.order or [])
        missing = sorted(names - known)
        if missing:
            rep.update(ok=False, reason=f"task names objects not in the scene: {missing}")
            return rep
        rep["task"] = req.task.model_dump()
    return rep
