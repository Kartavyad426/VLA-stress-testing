"""Agent v3 (two-part answer, agent/mode_a3.py) against presets or robustness cases on res31's v1/orange, via NIM.
Tests: robustness (presets/v1_orange_robustness.json x runs, checked against each case's expectation), suite (each
preset's first variant x its prompts x runs, compared part by part: core/compare3.py), variants (every variant,
first prompt, 1 run). Writes runs/taskgen/agent_eval_v3/<test>/<model>/ (per-call JSON + report.json).

    runs/real2sim/v1_dev/py_triton_first.sh ../VLA-exp-taskgen/experiments/taskgen/eval_agent3.py --test robustness --model openai/gpt-oss-20b
Exploratory, UNVERIFIED.
"""
import argparse, json, os, re, sys, time
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim_mujoco.base import MAIN  # noqa: E402
from sim_mujoco.sim import MujocoSim  # noqa: E402
from sim_mujoco import assets  # noqa: E402
from core.session import Session  # noqa: E402
from core.templates import Template  # noqa: E402
from core.request import Request  # noqa: E402
from core.compare3 import compare  # noqa: E402
from agent.models import Nim  # noqa: E402
from agent import mode_a3  # noqa: E402

WT = os.path.dirname(os.path.dirname(HERE))


def fill_prompt(p, values):
    def val(m):
        v = values[m.group(1)]
        if m.group(2) is not None:
            v = v[int(m.group(2))]
        return str(v[m.group(3)] if m.group(3) else v)
    return re.sub(r"\{(\w+)(?:\[(\d+)\])?(?:\.(\w+))?\}", val, p)


def check_expect(req, exp):
    if "either" in exp:
        res = [check_expect(req, e) for e in exp["either"]]
        return any(r[0] for r in res), [n for r in res for n in r[1]]
    refused = bool(req.infeasible)
    if "refused" in exp and refused != exp["refused"]:
        return False, [f"refused={refused}, expected {exp['refused']}"]
    if refused and exp.get("refused"):
        return True, []
    add = req.scene.add if req.scene else []
    pc = req.task.pass_all[0] if (req.task and req.task.pass_all) else None
    notes = []
    if "n_add" in exp and len(add) != exp["n_add"]:
        notes.append(f"{len(add)} added, expected {exp['n_add']}")
    if "sources" in exp and sorted(o.source for o in add) != sorted(exp["sources"]):
        notes.append(f"sources {[o.source for o in add]}")
    if "relations" in exp and sorted(o.relation for o in add) != sorted(exp["relations"]):
        notes.append(f"relations {[o.relation for o in add]}")
    if "size_cm_max" in exp and any((o.size_cm or 0) > exp["size_cm_max"] for o in add):
        notes.append("size above the bound")
    if "targets" in exp and any(o.target not in exp["targets"] for o in add if o.relation != "on_table"):
        notes.append(f"targets {[o.target for o in add]}")
    if "adds_noun" in exp and not any((o.noun or "").lower() == exp["adds_noun"] for o in add):
        notes.append(f"no added {exp['adds_noun']}")
    if "pass_type" in exp and (pc is None or pc.type != exp["pass_type"]):
        notes.append(f"pass type {pc and pc.type}")
    if "pass_type_in" in exp and (pc is None or pc.type not in exp["pass_type_in"]):
        notes.append(f"pass type {pc and pc.type}")
    if "pass_object" in exp and (pc is None or pc.object != exp["pass_object"]):
        notes.append(f"pass object {pc and pc.object}")
    if "pass_target" in exp and (pc is None or pc.target != exp["pass_target"]):
        notes.append(f"pass target {pc and pc.target}")
    return not notes, notes


def cases(a, P):
    if a.test == "robustness":
        return [(c["id"], c["prompt"], None, c["expect"], {}) for c in json.load(open(a.robustness))["cases"]]
    out = []
    for p in P["presets"]:
        variants = Template.model_validate(p["template_def"]).expand(model=Request)[1]
        variants = variants[:1] if a.test == "suite" else variants
        prompts = p["prompts"] if a.test == "suite" else p["prompts"][:1]
        for vi, (values, want) in enumerate(variants):
            for pi, pr in enumerate(prompts):
                out.append((f"{p['id']}_v{vi}_p{pi}", fill_prompt(pr, values), want, None, values))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["robustness", "suite", "variants"], default="robustness")
    ap.add_argument("--presets", default=os.path.join(HERE, "presets/v1_orange.json"))
    ap.add_argument("--robustness", default=os.path.join(HERE, "presets/v1_orange_robustness.json"))
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tag", default="", help="suffix for the output folder (e.g. x for presets/v1_orange_x.json), so runs on other preset files don't overwrite each other")
    a = ap.parse_args()
    if a.test == "variants":
        a.runs = 1
    P = json.load(open(a.presets))
    out = os.path.join(WT, "runs/taskgen/agent_eval_v3", a.test + (f"_{a.tag}" if a.tag else ""), a.model.replace("/", "__"))
    os.makedirs(out, exist_ok=True)
    q0 = h5py.File(os.path.join(MAIN, P["episode"], "trajectory.h5"))["observation/robot_state/joint_positions"][0]
    model = Nim(a.model)
    rows = []
    for cid, text, want, exp, values in cases(a, P):
        for run in range(a.runs):
            S = Session(MujocoSim, assets, os.path.join(MAIN, P["scene"]), os.path.join(MAIN, P["episode"]), q0)
            t0 = time.time()
            try:
                r = mode_a3.run(model, S, assets, text, task_id=cid)
            except Exception as e:  # server / network errors are recorded, not hidden
                r = dict(ok=False, reason=f"{type(e).__name__}: {e}", log=[], attempts=0)
            dt = time.time() - t0
            req = Request.model_validate(r["request"]) if r["ok"] else None
            cmp_ = compare(req, want) if (req is not None and want is not None) else None
            ok_e, notes = check_expect(req, exp) if (req is not None and exp is not None) else (None, [])
            row = dict(test=a.test, case=cid.rsplit("_v", 1)[0] if want is not None else cid, case_id=cid, run=run, prompt=text, valid=r["ok"],
                       attempts=r.get("attempts"), seconds=round(dt, 1), match=bool(cmp_["match"]) if cmp_ else bool(ok_e), compare=cmp_,
                       expect_notes=notes, scene_ok=(r.get("scene") or {}).get("ok"), scene_reason=(r.get("scene") or {}).get("reason"),
                       second_call=bool(r.get("second_call")), refusals=(r.get("request") or {}).get("infeasible"), reason=r.get("reason"))
            rows.append(row)
            json.dump(dict(row, log=r.get("log"), result=r), open(os.path.join(out, f"{cid}_r{run}.json"), "w"), indent=1, default=str)
            print(json.dumps({k: row[k] for k in ("case_id", "run", "valid", "attempts", "match", "scene_ok", "seconds")}), flush=True)
    summ = {}
    for r in rows:
        s = summ.setdefault(r["case"], dict(calls=0, valid=0, match=0, scene_ok=0, seconds=0.0))
        s["calls"] += 1; s["valid"] += r["valid"]; s["match"] += r["match"]; s["scene_ok"] += bool(r["scene_ok"]); s["seconds"] += r["seconds"]
    json.dump(dict(model=a.model, test=a.test, presets=a.presets, tag=a.tag, prompt_version=mode_a3.PROMPT_VERSION, runs=a.runs, summary=summ, rows=rows),
              open(os.path.join(out, "report.json"), "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
