"""Compare a SceneRequest (e.g. the agent's) with a preset's (docs/TASKGEN_DESIGN.html §4d). Must match: added
objects' kinds / colours / nouns / relations / targets (as a multiset, names ignored), pass-condition type, object,
target and order, and feasible vs infeasible. May differ: names, sizes, wording. Exploratory, UNVERIFIED.
"""
from collections import Counter


def _sig(req):
    names = {o.name: (o.source, o.colour or o.noun or "") for o in req.add}

    def canon(n):
        return names.get(n, n)  # added objects by what they are, base objects by name
    objs = Counter((o.source, o.colour or (o.noun or "").lower(), o.relation, canon(o.target) if o.target else None) for o in req.add)
    task = None
    if req.task:
        pc = req.task.pass_condition
        task = (pc.type, canon(pc.object) if pc.object else None, canon(pc.target) if pc.target else None,
                tuple(canon(x) for x in (pc.order or [])))
    return objs, task


def compare(got, want):
    g_obj, g_task = _sig(got)
    w_obj, w_task = _sig(want)
    out = dict(feasible_match=bool(got.infeasible) == bool(want.infeasible))
    if want.infeasible:
        out.update(objects_match=True, task_match=True)
    else:
        out.update(objects_match=g_obj == w_obj, task_match=g_task == w_task)
        if g_obj != w_obj:
            out["objects_diff"] = dict(missing=list((w_obj - g_obj).elements()), extra=list((g_obj - w_obj).elements()))
        if g_task != w_task:
            out["task_diff"] = dict(got=g_task, want=w_task)
    out["match"] = out["feasible_match"] and out["objects_match"] and out["task_match"]
    return out
