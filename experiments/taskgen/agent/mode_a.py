"""Mode A agent (docs/TASKGEN_DESIGN.html §4a): one structured answer per prompt. The model fills a SceneRequest
(JSON); our side validates it (schema + names) and sends any error back for a retry (<= 2); plain code then executes
it. The model never does geometry: the tools place, resize and refuse. Exploratory, UNVERIFIED.
"""
import json, re
from pydantic import ValidationError

from core.templates import SceneRequest, schema_json
from core import executor

PROMPT_VERSION = "v2"  # v1 -> v2 (10-29, user OK): "locked" means setup only (L26); drawer/cabinet synonym (L30)
RULES = """You set up robot manipulation tasks in a simulated scene that was reconstructed from a real video.
Rules:
- The base objects listed are LOCKED for the scene setup: you can't add, move, resize or remove them when building the scene.
  Tasks may (and usually do) ask the ROBOT to move base objects: "put the orange block in the bowl" is a valid task on base
  objects with nothing to add.
- To add an object give: name, source (box | drawer | pack with a noun; a drawer or cabinet is source drawer, not pack), colour for boxes, optional size_cm (blocks 2-6 cm),
  and where it goes: relation (on_table | next_to | on | inside) and target (an existing object or one you add earlier in the list).
- Never give coordinates; the simulator finds the exact place and may resize added objects to fit.
- The task: an instruction, the target objects, and a pass_condition (inside | on | stacked | ordered_stack | out_of | next_to).
- If the request can't be done under these rules (e.g. the scene would have to START with a base object somewhere else,
  like "the bowl starts inside a drawer"), set "infeasible" to the reason and leave add empty.
Answer with ONE JSON object matching the schema, nothing else."""


def scene_summary(session):
    sc = session.list_scene()
    lines = ["Base objects (locked):"]
    for n, r in sc["base"].items():
        extra = f", opening {r['opening_cm']} cm" if "opening_cm" in r else ""
        lines.append(f"  {n}: {r['kind']}, size {r['size_cm']} cm{extra}")
    if sc["added"]:
        lines.append("Added objects:")
        for n, r in sc["added"].items():
            lines.append(f"  {n}: {r['kind']}, size {r['size_cm']} cm")
    lines.append(f"Gripper opening: {sc['gripper_opening_cm']} cm")
    return "\n".join(lines)


def parse(text):
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()  # reasoning models: drop the trace
    t = re.sub(r"^```(json)?|```$", "", t, flags=re.M).strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    return json.loads(m.group(0) if m else t)


def semantic_errors(req, session):
    """Name errors the schema can't catch, phrased so a model can fix them."""
    errs = []
    base = set(session.base_bodies)
    seen = set(base) | {n for n, a in session.cfg.added_objects.items() if a.pose is not None}
    for o in req.add:
        if o.name in base:
            errs.append(f"'{o.name}' is a locked base object; it can't be added or moved (answer infeasible if the request needs that)")
        if o.relation != "on_table" and o.target not in seen:
            errs.append(f"{o.name}: target '{o.target}' doesn't exist (base objects: {sorted(base)}; add targets earlier in the list)")
        if o.source == "pack" and not o.noun:
            errs.append(f"{o.name}: source pack needs a noun")
        if o.source == "box" and not o.colour:
            errs.append(f"{o.name}: source box needs a colour")
        seen.add(o.name)
    if req.task and not req.infeasible:
        for n in set(req.task.targets) | {x for x in (req.task.pass_condition.object, req.task.pass_condition.target) if x}:
            if n not in seen:
                errs.append(f"task refers to '{n}', which is neither a base object nor one you add")
    return errs


def run(model, session, prompt, retries=2):
    msgs = [dict(role="system", content=RULES + "\n\nSchema:\n" + schema_json()),
            dict(role="user", content=scene_summary(session) + "\n\nRequest: " + prompt)]
    log = []
    for attempt in range(retries + 1):
        text = model.complete(msgs, schema_json())
        try:
            req = SceneRequest.model_validate(parse(text))
            errs = semantic_errors(req, session)
        except (ValueError, ValidationError) as e:
            req, errs = None, [str(e).splitlines()[0] if isinstance(e, ValueError) and not isinstance(e, ValidationError) else
                               "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())]
        log.append(dict(attempt=attempt, answer=text[:2000], errors=errs))
        if not errs:
            return dict(ok=True, request=req.model_dump(), attempts=attempt + 1, log=log, exec=executor.run(session, req))
        msgs += [dict(role="assistant", content=text),
                 dict(role="user", content="That answer is invalid: " + " | ".join(errs) + ". Answer again with one corrected JSON object only.")]
    return dict(ok=False, attempts=retries + 1, log=log, reason="no valid answer after retries")
