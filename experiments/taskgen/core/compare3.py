"""Compare a v3 two-part answer (core/request.py) with a preset's, part by part (docs/TASKGEN_DESIGN.html §4d, D20:
each job scored separately). Scene part: added objects as a multiset of (source, colour or noun, levels/open_level,
relation, target), names ignored (an added object as target is replaced by what it is; a cabinet drawer keeps its
level). Task part: the pass conditions as a multiset. Refusals: which parts were refused. Exploratory, UNVERIFIED.
"""
from collections import Counter


def _canon_map(req):
    m = {}
    for o in (req.scene.add if req.scene else []):
        what = o.colour or (o.noun or "").lower()
        if o.source == "cabinet":
            what = f"{o.levels or 3}L/{o.open_level or 'middle'}"
        m[o.name] = (o.source, what)
    return m


def _canon(m, n):
    if not n:
        return None
    base, _, lv = n.partition("/")
    c = m.get(base, base)
    return (c, lv) if lv else c


def sig(req):
    m = _canon_map(req)
    scene = Counter((m[o.name], o.relation, _canon(m, o.target)) for o in (req.scene.add if req.scene else []))
    task = Counter((c.type, _canon(m, c.object), _canon(m, c.target), tuple(_canon(m, x) for x in (c.order or [])))
                   for c in (req.task.pass_all if req.task else []))
    refused = frozenset(r.part for r in req.infeasible)
    return scene, task, refused


def compare(got, want):
    gs, gt, gr = sig(got)
    ws, wt, wr = sig(want)
    out = dict(scene_match=gs == ws, task_match=gt == wt, refusal_match=gr == wr)
    if gs != ws:
        out["scene_diff"] = dict(missing=[str(x) for x in (ws - gs).elements()], extra=[str(x) for x in (gs - ws).elements()])
    if gt != wt:
        out["task_diff"] = dict(missing=[str(x) for x in (wt - gt).elements()], extra=[str(x) for x in (gt - wt).elements()])
    if gr != wr:
        out["refusal_diff"] = dict(got=sorted(gr), want=sorted(wr))
    out["match"] = out["scene_match"] and out["task_match"] and out["refusal_match"]
    return out
