"""Executor for the v3 two-part answer (core/request.py): the scene part through the Session tools (job A), then the
task part checked against the scene as built and turned into a task config (job B). Reports what the scene step
changed (resizes, refusals), which decides whether the agent is asked again about the task. Plain code, no model.
Exploratory, UNVERIFIED.
"""
from core.request import Request
from core.taskconfig import TaskConfig, rule_known

SOURCE = {"box": "primitive:box", "drawer": "primitive:drawer", "cabinet": "primitive:cabinet"}


def _emit(on_event, **kw):
    if on_event:
        on_event(kw)


def run_scene(session, req: Request, allow_resize=True, on_event=None):
    """on_event(dict): called before and after every tool call (type tool_start / tool_end) for live views."""
    rep = dict(steps=[], ok=True, changed=[])
    for o in (req.scene.add if req.scene else []):
        src = SOURCE.get(o.source) or f"pack:{o.noun or ''}"
        params = dict(levels=o.levels or 3, open_level=o.open_level or "middle") if o.source == "cabinet" else None
        args = dict(source=src, colour=o.colour, size_cm=o.size_cm, name=o.name, params=params)
        _emit(on_event, type="tool_start", tool="add_object", args=args)
        r = session.add_object(src, colour=o.colour, size_cm=o.size_cm, name=o.name, params=params)
        _emit(on_event, type="tool_end", tool="add_object", args=args, result=r, ok="error" not in r)
        rep["steps"].append(dict(call="add_object", name=o.name, result=r))
        if "error" in r:
            rep.update(ok=False, failed=o.name, reason=r["error"])
            return rep
        args = dict(name=o.name, relation=o.relation, target=o.target, allow_resize=allow_resize)
        _emit(on_event, type="tool_start", tool="place", args=args)
        r = session.place(o.name, o.relation, o.target, allow_resize=allow_resize)
        _emit(on_event, type="tool_end", tool="place", args=args, result=r, ok="placed" in r)
        rep["steps"].append(dict(call="place", name=o.name, result=r))
        if "placed" not in r:
            rep.update(ok=False, failed=o.name, reason=r.get("infeasible") or r.get("error"))
            return rep
        if "resized" in r:
            rep["changed"].append(f"{o.name} resized: {r['resized']}")
    return rep


def task_names_ok(session, task):
    known = set(session.base_bodies) | {n for n, a in session.cfg.added_objects.items() if a.pose is not None}
    names = set(task.targets)
    for c in task.pass_all:
        names |= {x for x in (c.object, c.target) if x} | set(c.order or [])
    missing = sorted(n for n in names if not session._exists(n) and n not in known)
    return missing


def make_task(session, req: Request, task_id, source="agent", scene_config_path="(in memory)"):
    """TaskConfig from the task part, with the rule check (E4) filled in; None if there is no task part."""
    if not req.task:
        return None
    t = TaskConfig(id=task_id, source=source, scene_config=scene_config_path, base_version=session.cfg.base_version,
                   instruction=req.task.instruction, targets=req.task.targets, pass_all=req.task.pass_all)
    t.checks["rule_known"] = rule_known(t)
    return t
