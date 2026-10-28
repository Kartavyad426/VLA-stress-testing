"""Stage 2: the agent (mode A) against the orange presets (docs/TASKGEN_DESIGN.html §4d).

For each preset, variant 0 of its template is the expected answer; every prompt paraphrase (filled with the same slot
values) is sent --runs times to the model, each on a fresh session. Per call: valid answer, attempts, time, field
match with the preset (core/compare.py), executor result. Needs a running llama-server (prim's GPU gate):
    ~/Documents/Code/Tau-harness/scripts/serve_examiner_9b.sh        # Qwen3.5-9B on :8099
    runs/real2sim/v1_dev/py_triton_first.sh ../VLA-exp-taskgen/experiments/taskgen/eval_agent.py --model qwen3.5-9b
Writes runs/taskgen/agent_eval/<model>/report.json (+ per-call logs). Exploratory, UNVERIFIED.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default=os.path.join(HERE, "presets/orange.json"))
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--only", default=None, help="comma-separated preset ids")
    a = ap.parse_args()
    P = json.load(open(a.presets))
    out = os.path.join(WT, "runs/taskgen/agent_eval", a.model)
    os.makedirs(out, exist_ok=True)
    q0 = h5py.File(os.path.join(MAIN, P["episode"], "trajectory.h5"))["observation/robot_state/joint_positions"][0]
    model = LlamaServer(a.url, a.model)
    rows = []
    for p in P["presets"]:
        if a.only and p["id"] not in a.only.split(","):
            continue
        values, want = Template.model_validate(p["template"]).expand()[1][0]
        for pi, prompt in enumerate(p["prompts"]):
            text = fill_prompt(prompt, values)
            for run in range(a.runs):
                S = Session(MujocoSim, assets, os.path.join(MAIN, P["scene"]), os.path.join(MAIN, P["episode"]), q0)
                t0 = time.time()
                try:
                    r = mode_a.run(model, S, text)
                except Exception as e:  # server down / timeout: recorded, not hidden
                    r = dict(ok=False, reason=f"{type(e).__name__}: {e}", log=[], attempts=0)
                dt = time.time() - t0
                cmp_ = compare(SceneRequest.model_validate(r["request"]), want) if r["ok"] else None
                row = dict(preset=p["id"], values=values, paraphrase=pi, run=run, prompt=text, valid=r["ok"], attempts=r.get("attempts"),
                           seconds=round(dt, 1), match=bool(cmp_ and cmp_["match"]), compare=cmp_,
                           exec_ok=(r.get("exec") or {}).get("ok"), exec_reason=(r.get("exec") or {}).get("reason"), reason=r.get("reason"))
                rows.append(row)
                json.dump(dict(row, log=r.get("log")), open(os.path.join(out, f"{p['id']}_p{pi}_r{run}.json"), "w"), indent=1, default=str)
                print(json.dumps({k: row[k] for k in ("preset", "paraphrase", "run", "valid", "attempts", "match", "exec_ok", "seconds")}))
    summ = {}
    for r in rows:
        s = summ.setdefault(r["preset"], dict(calls=0, valid=0, match=0, exec_ok=0, attempts=0, seconds=0.0))
        s["calls"] += 1
        s["valid"] += r["valid"]
        s["match"] += r["match"]
        s["exec_ok"] += bool(r["exec_ok"])
        s["attempts"] += r["attempts"] or 0
        s["seconds"] += r["seconds"]
    json.dump(dict(model=a.model, url=a.url, runs=a.runs, summary=summ, rows=rows), open(os.path.join(out, "report.json"), "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
