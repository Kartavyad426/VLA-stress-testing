"""Stage 2: the agent (mode A) against the orange presets (docs/TASKGEN_DESIGN.html §4d).

For each preset, variant 0 of its template is the expected answer; every prompt paraphrase (filled with the same slot
values) is sent --runs times to the model, each on a fresh session. Per call: valid answer, attempts, time, field
match with the preset (core/compare.py), executor result. Needs a running llama-server (prim's GPU gate):
    ~/Documents/Code/Tau-harness/scripts/serve_examiner_9b.sh        # Qwen3.5-9B on :8099
    runs/real2sim/v1_dev/py_triton_first.sh ../VLA-exp-taskgen/experiments/taskgen/eval_agent.py --model qwen3.5-9b
Tests: smoke (1 call; exit 2 if invalid), suite (presets x paraphrases x runs), variants (every slot value, 1st
paraphrase, 1 run), robustness (presets/orange_robustness.json x runs, checked against each case's expectation).
Writes runs/taskgen/agent_eval/<test>/<model>/report.json (+ per-call logs). Exploratory, UNVERIFIED.
"""
import argparse, json, os, re, sys, time
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim_mujoco.base import MAIN  # noqa: E402
from sim_mujoco.sim import MujocoSim  # noqa: E402
from sim_mujoco import assets  # noqa: E402
from core.session import Session  # noqa: E402
from core.templates import Template, SceneRequest  # noqa: E402
from core.compare import compare  # noqa: E402
from agent.models import LlamaServer  # noqa: E402
from agent import mode_a  # noqa: E402

WT = os.path.dirname(os.path.dirname(HERE))


def fill_prompt(p, values):
    return re.sub(r"\{(\w+)(\[(\d+)\])?\}", lambda m: str(values[m.group(1)][int(m.group(3))] if m.group(3) else values[m.group(1)]), p)


def check_expect(req, exp):
    """Does a valid answer satisfy a robustness case's expectation? Returns (ok, notes)."""
    if "either" in exp:
        res = [check_expect(req, e) for e in exp["either"]]
        return any(r[0] for r in res), [n for r in res for n in r[1]]
    notes = []
    feas = not req.infeasible
    if exp.get("feasible") is not None and feas != exp["feasible"]:
        return False, [f"feasible={feas}, expected {exp['feasible']}"]
    if not feas:
        return True, notes
    if "n_add" in exp and len(req.add) != exp["n_add"]:
        notes.append(f"{len(req.add)} objects added, expected {exp['n_add']}")
    if "sources" in exp and sorted(o.source for o in req.add) != sorted(exp["sources"]):
        notes.append(f"sources {[o.source for o in req.add]}")
    if "relations" in exp and sorted(o.relation for o in req.add) != sorted(exp["relations"]):
        notes.append(f"relations {[o.relation for o in req.add]}")
    if "size_cm_max" in exp and any((o.size_cm or 0) > exp["size_cm_max"] for o in req.add):
        notes.append("size above the bound")
    if "targets" in exp and any(o.target not in exp["targets"] for o in req.add if o.relation != "on_table"):
        notes.append(f"targets {[o.target for o in req.add]}")
    pc = req.task.pass_condition if req.task else None
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
    """(case id, prompt, expected request or None, expectation or None) per test."""
    out = []
    if a.test == "robustness":
        for c in json.load(open(a.robustness))["cases"]:
            out.append((c["id"], c["prompt"], None, c["expect"], {}))
        return out
    for p in P["presets"]:
        if a.only and p["id"] not in a.only.split(","):
            continue
        variants = Template.model_validate(p["template"]).expand()[1]
        if a.test in ("suite", "smoke"):
            variants = variants[:1]
        prompts = p["prompts"] if a.test == "suite" else p["prompts"][:1]
        for vi, (values, want) in enumerate(variants):
            for pi, prompt in enumerate(prompts):
                out.append((f"{p['id']}_v{vi}_p{pi}", fill_prompt(prompt, values), want, None, values))
    return out[:1] if a.test == "smoke" else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["smoke", "suite", "variants", "robustness"], default="suite")
    ap.add_argument("--robustness", default=os.path.join(HERE, "presets/orange_robustness.json"))
    ap.add_argument("--presets", default=os.path.join(HERE, "presets/orange.json"))
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--only", default=None, help="comma-separated preset ids")
    a = ap.parse_args()
    P = json.load(open(a.presets))
    if a.test == "smoke":
        a.only = a.only or "P3_added_block_into_bowl"
        a.runs = 1
    if a.test == "variants":
        a.runs = 1
    out = os.path.join(WT, "runs/taskgen/agent_eval", a.test, a.model)
    os.makedirs(out, exist_ok=True)
    q0 = h5py.File(os.path.join(MAIN, P["episode"], "trajectory.h5"))["observation/robot_state/joint_positions"][0]
    model = LlamaServer(a.url, a.model)
    rows = []
    for cid, text, want, exp, values in cases(a, P):
        for run in range(a.runs):
            S = Session(MujocoSim, assets, os.path.join(MAIN, P["scene"]), os.path.join(MAIN, P["episode"]), q0)
            t0 = time.time()
            try:
                r = mode_a.run(model, S, text)
            except Exception as e:  # server down / timeout: recorded, not hidden
                r = dict(ok=False, reason=f"{type(e).__name__}: {e}", log=[], attempts=0)
            dt = time.time() - t0
            req = SceneRequest.model_validate(r["request"]) if r["ok"] else None
            cmp_ = compare(req, want) if (req is not None and want is not None) else None
            ok_exp, notes = check_expect(req, exp) if (req is not None and exp is not None) else (None, [])
            match = bool(cmp_["match"]) if cmp_ else bool(ok_exp)
            row = dict(test=a.test, case=cid.rsplit("_v", 1)[0] if want is not None else cid, case_id=cid, values=values, run=run,
                       prompt=text, valid=r["ok"], attempts=r.get("attempts"), seconds=round(dt, 1), match=match, compare=cmp_,
                       expect_notes=notes, exec_ok=(r.get("exec") or {}).get("ok"), exec_reason=(r.get("exec") or {}).get("reason"),
                       infeasible=(r.get("request") or {}).get("infeasible"), reason=r.get("reason"))
            rows.append(row)
            json.dump(dict(row, log=r.get("log")), open(os.path.join(out, f"{cid}_r{run}.json"), "w"), indent=1, default=str)
            print(json.dumps({k: row[k] for k in ("case_id", "run", "valid", "attempts", "match", "exec_ok", "seconds")}), flush=True)
            if a.test == "smoke" and not r["ok"]:
                json.dump(dict(model=a.model, test=a.test, rows=rows), open(os.path.join(out, "report.json"), "w"), indent=1, default=str)
                sys.exit(2)  # the queue stops on a failed smoke test
    summ = {}
    for r in rows:
        s = summ.setdefault(r["case"], dict(calls=0, valid=0, match=0, exec_ok=0, attempts=0, seconds=0.0))
        s["calls"] += 1
        s["valid"] += r["valid"]
        s["match"] += r["match"]
        s["exec_ok"] += bool(r["exec_ok"])
        s["attempts"] += r["attempts"] or 0
        s["seconds"] += r["seconds"]
    json.dump(dict(model=a.model, test=a.test, url=a.url, runs=a.runs, summary=summ, rows=rows), open(os.path.join(out, "report.json"), "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
