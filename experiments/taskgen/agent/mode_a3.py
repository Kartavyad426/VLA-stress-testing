"""Mode A, v3 (D20: one general agent; docs/TASKGEN_DESIGN.html §1a, §1b): one structured answer with two optional
parts, scene (job A: add objects) and task (job B: what the robot must do). Our side validates (schema + names +
known pack nouns, with hints the model can act on), runs the scene part through the tools, checks the task against
the scene as built, and asks ONE more time about the task only if the scene step resized or refused something.
New file beside agent/mode_a.py (v1/v2 kept for the runs still using them). Exploratory, UNVERIFIED.
"""
import json
from pydantic import ValidationError

from core.request import Request, schema_json
from core import executor3
from agent.mode_a import parse

PROMPT_VERSION = "v3"
RULES = """You set up robot manipulation tasks in a simulated scene reconstructed from a real video. There are TWO separate jobs;
a request may need either one or both:
A. SCENE (field "scene"): objects to ADD to the scene. The base objects listed are LOCKED for the setup: you can't add, move,
   resize or remove them. Each added object: name, source, and where it goes: relation (on_table | next_to | on | inside)
   + target (a base object, an object added earlier in the list, or a cabinet drawer written "cabinet_1/middle").
   Sources: box (a block; needs colour; size_cm 2-6), cabinet (a cabinet with stacked drawers: levels 1-3, open_level =
   which drawer starts open: bottom | middle | top; objects can only go INTO the open drawer), drawer (a single drawer),
   pack (an everyday object from the asset packs; needs a noun such as cup, bowl, jar). A cabinet or a drawer is never "pack".
   Never give coordinates: the simulator finds the place and may resize added objects to fit.
B. TASK (field "task"): what the robot must do at the end. The robot MAY move any object, base or added ("put the orange
   block in the bowl" is a task on base objects with nothing to add). Give the instruction, the target objects, and
   pass_all: conditions that must ALL hold at the end, each one of inside | on | stacked | ordered_stack | out_of | next_to.
If a part can't be done under these rules (e.g. "the bowl starts inside a drawer": the setup can't move a base object), add an
entry to "infeasible" for that part with the reason, and the closest alternative you used instead if there is one
(e.g. an added object in place of a base one). Leave a part out (null) when the request doesn't ask for it.
Answer with ONE JSON object matching the schema, nothing else."""


def scene_summary(session):
    sc = session.list_scene()
    lines = ["Base objects (LOCKED for the setup; the robot may move them in a task):"]
    for n, r in sc["base"].items():
        extra = f", opening {r['opening_cm']} cm" if "opening_cm" in r else ""
        lines.append(f"  {n}: {r['kind']}, colour {session.colour_of(n)}, size {r['size_cm']} cm{extra}")
    if sc["added"]:
        lines.append("Added objects:")
        for n, r in sc["added"].items():
            lv = f", drawers {r['levels']}" if "levels" in r else ""
            lines.append(f"  {n}: {r['kind']}, size {r['size_cm']} cm{lv}")
    lines.append(f"Gripper opening: {sc['gripper_opening_cm']} cm")
    return "\n".join(lines)


def semantic_errors(req, session, assets):
    errs = []
    base = set(session.base_bodies)
    seen = set(base) | {n for n, a in session.cfg.added_objects.items() if a.pose is not None}
    cabinets = {}
    for o in (req.scene.add if req.scene else []):
        if o.name in base:
            errs.append(f"'{o.name}' is a locked base object: the scene part can only ADD objects (the task part may move it)")
        if o.source == "box" and not o.colour:
            errs.append(f"{o.name}: source box needs a colour")
        if o.source == "pack":
            if not o.noun:
                errs.append(f"{o.name}: source pack needs a noun")
            elif o.noun.lower() in ("cabinet", "drawer", "dresser", "chest of drawers"):
                errs.append(f"{o.name}: a {o.noun} is source cabinet (stacked drawers) or drawer, not pack")
            elif not assets.search(o.noun, 1):
                errs.append(f"{o.name}: no pack model matches '{o.noun}'; use another everyday noun (e.g. cup, bowl, jar, pot) or a box")
        if o.source == "cabinet":
            lv = o.levels or 3
            names = {1: ["only"], 2: ["bottom", "top"], 3: ["bottom", "middle", "top"]}.get(lv)
            if names is None:
                errs.append(f"{o.name}: levels must be 1, 2 or 3")
            else:
                cabinets[o.name] = (names, o.open_level or "middle")
                if (o.open_level or "middle") not in names:
                    errs.append(f"{o.name}: open_level must be one of {names}")
        if o.relation != "on_table":
            tb, _, lv = (o.target or "").partition("/")
            if tb not in seen:
                errs.append(f"{o.name}: target '{o.target}' doesn't exist (base objects: {sorted(base)}; add targets earlier in the list)")
            elif lv:
                if tb not in cabinets:
                    errs.append(f"{o.name}: '{o.target}' names a drawer level, but {tb} is not a cabinet")
                elif lv not in cabinets[tb][0]:
                    errs.append(f"{o.name}: {tb} has drawers {cabinets[tb][0]}")
                elif o.relation == "inside" and lv != cabinets[tb][1]:
                    errs.append(f"{o.name}: the {lv} drawer of {tb} is closed; objects can only go into the open one ({cabinets[tb][1]}), or set open_level={lv}")
        seen.add(o.name)
    if req.task and not req.task.pass_all:  # 10-29: "close the drawer" came back with an empty pass_all (no rule for it)
        errs.append("the task has no pass_all conditions: if what is asked can't be written with inside / on / stacked / ordered_stack / "
                    "out_of / next_to (e.g. opening or closing a drawer), leave task null and add an infeasible entry for part 'task' saying so")
    if req.task:
        names = set(req.task.targets)
        for c in req.task.pass_all:
            names |= {x for x in (c.object, c.target) if x} | set(c.order or [])
        for n in sorted(names):
            tb = n.split("/")[0]
            if tb not in seen:
                errs.append(f"the task refers to '{n}', which is neither a base object nor one you add")
    return errs


def _emit(on_event, **kw):
    if on_event:
        on_event(kw)


def _ask(model, msgs, session, assets, retries, log, on_event=None, purpose="answer"):
    import time
    for attempt in range(retries + 1):
        _emit(on_event, type="model_start", purpose=purpose, attempt=attempt, model=getattr(model, "model", type(model).__name__),
              messages=[dict(m) for m in msgs])  # exactly what the model is sent (observability O2)
        t0 = time.time()
        text = model.complete(msgs, schema_json())
        _emit(on_event, type="model_end", purpose=purpose, attempt=attempt, seconds=round(time.time() - t0, 1), answer=text)
        try:
            req = Request.model_validate(parse(text))
            errs = semantic_errors(req, session, assets)
        except (ValueError, ValidationError) as e:
            req = None
            errs = ["; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())] if isinstance(e, ValidationError) else [str(e).splitlines()[0]]
        log.append(dict(attempt=len(log), answer=text[:3000], errors=errs))
        _emit(on_event, type="validate", attempt=attempt, ok=not errs, errors=errs,
              summary=None if errs else dict(n_add=len(req.scene.add) if req.scene else 0, task=bool(req.task),
                                             refusals=[r.model_dump() for r in req.infeasible]))
        if not errs:
            return req
        msgs += [dict(role="assistant", content=text),
                 dict(role="user", content="That answer is invalid: " + " | ".join(errs) + ". Answer again with one corrected JSON object only.")]
    return None


def run(model, session, assets, prompt, retries=2, task_id="agent_task", on_event=None):
    """on_event(dict) is called at every step (model_start/end, validate, tool_start/end, second_call, task, done) for live
    views such as the chat UI; the result is the same with or without it."""
    msgs = [dict(role="system", content=RULES + "\n\nSchema:\n" + schema_json()),
            dict(role="user", content=scene_summary(session) + "\n\nRequest: " + prompt)]
    log = []
    req = _ask(model, msgs, session, assets, retries, log, on_event)
    if req is None:
        _emit(on_event, type="done", ok=False, reason="no valid answer after retries")
        return dict(ok=False, attempts=len(log), log=log, reason="no valid answer after retries")
    scene_rep = executor3.run_scene(session, req, on_event=on_event)
    second = None
    if req.task and (scene_rep["changed"] or not scene_rep["ok"]):  # the scene came out differently: ask about the task once more
        note = "; ".join(scene_rep["changed"]) or f"failed: {scene_rep.get('reason')}"
        msgs2 = [msgs[0], dict(role="user", content=scene_summary(session) + "\n\nRequest: " + prompt +
                               f"\n\nThe scene part was applied and came out differently: {note}. Give the TASK part again for the scene as it is now "
                               "(set scene to null; add an infeasible entry if the task can no longer be done).")]
        _emit(on_event, type="second_call", why=note)
        req2 = _ask(model, msgs2, session, assets, retries, log, on_event, purpose="task again")
        if req2 is not None:
            second = req2.model_dump()
            req = Request(scene=req.scene, task=req2.task, infeasible=req.infeasible + req2.infeasible)
    task = executor3.make_task(session, req, task_id) if (req.task and scene_rep["ok"]) else None
    missing = executor3.task_names_ok(session, req.task) if task else []
    if task is not None:
        _emit(on_event, type="task", task=json.loads(task.model_dump_json()), missing=missing)
    _emit(on_event, type="done", ok=True, scene_ok=scene_rep["ok"], scene_reason=scene_rep.get("reason"),
          changed=scene_rep["changed"], refusals=[r.model_dump() for r in req.infeasible])
    return dict(ok=True, prompt_version=PROMPT_VERSION, request=req.model_dump(), attempts=len(log), log=log, scene=scene_rep,
                second_call=second, task=None if task is None else json.loads(task.model_dump_json()), task_missing_names=missing)
