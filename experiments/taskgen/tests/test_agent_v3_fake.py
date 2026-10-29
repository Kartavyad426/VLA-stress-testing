"""Agent v3 (two-part answer) without a real model, on res31's v1/orange: preset counts, the executor on every
preset's first variant, and the mode A v3 loop with a fake model: a scene-and-task answer, a task-only answer, the
"pack cabinet" mistake corrected through the hint, a closed-drawer target corrected, a partial refusal with an
alternative, and the second call after a resize. Exit 1 on any failure. Writes runs/taskgen/agent_v3_fake/.

    runs/real2sim/v1_dev/py_triton_first.sh ../VLA-exp-taskgen/experiments/taskgen/tests/test_agent_v3_fake.py
Exploratory, UNVERIFIED.
"""
import copy, json, os, sys, time
import h5py

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from sim_mujoco.base import MAIN  # noqa: E402
from sim_mujoco.sim import MujocoSim  # noqa: E402
from sim_mujoco import assets  # noqa: E402
from core.session import Session  # noqa: E402
from core.templates import Template  # noqa: E402
from core.request import Request  # noqa: E402
from core.compare3 import compare  # noqa: E402
from core import executor3  # noqa: E402
from agent.models import FakeModel  # noqa: E402
from agent import mode_a3  # noqa: E402

WT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(WT, "runs/taskgen/agent_v3_fake")
P = json.load(open(os.path.join(HERE, "presets/v1_orange.json")))
fails, results = [], {}


def case(name, got, ok):
    results[name] = got
    print(("PASS " if ok else "FAIL ") + name + ": " + json.dumps(got, default=str)[:330], flush=True)
    if not ok:
        fails.append(name)


def session():
    q0 = h5py.File(os.path.join(MAIN, P["episode"], "trajectory.h5"))["observation/robot_state/joint_positions"][0]
    return Session(MujocoSim, assets, os.path.join(MAIN, P["scene"]), os.path.join(MAIN, P["episode"]), q0)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    tpl = {p["id"]: Template.model_validate(p["template_def"]) for p in P["presets"]}
    counts = {k: t.count() for k, t in tpl.items()}
    case("preset counts", counts, counts == {"P-cabinet": 9, "P-into": 3, "P-stack": 6, "P-alt-start": 1, "P-infeasible": 1})
    first = {k: t.expand(model=Request)[1][0] for k, t in tpl.items()}

    for pid, (values, req) in first.items():
        S = session()
        rep = executor3.run_scene(S, req)
        missing = executor3.task_names_ok(S, req.task) if req.task else []
        case(f"executor v3: {pid} {values if pid != 'P-into' else values['block']['colour']}",
             dict(ok=rep["ok"], placed=[st["result"].get("placed") for st in rep["steps"] if st["call"] == "place"], missing=missing, reason=rep.get("reason")),
             rep["ok"] and not missing)

    want = first["P-cabinet"][1]
    good = json.loads(want.model_dump_json())
    # 1. scene + task, correct, other names
    ren = copy.deepcopy(good)
    ren["scene"]["add"][1]["name"] = "my_cube"
    ren["task"]["targets"] = ["my_cube"]
    ren["task"]["pass_all"][0]["object"] = "my_cube"
    r = mode_a3.run(FakeModel([json.dumps(ren)]), session(), assets, "add a three-drawer cabinet ...")
    c = compare(Request.model_validate(r["request"]), want) if r["ok"] else None
    case("v3 fake: scene + task answer, other names -> built, task config made, matches", dict(attempts=r.get("attempts"), scene_ok=r.get("scene", {}).get("ok"), compare=c,
         task_status=(r.get("task") or {}).get("status")), r["ok"] and r["scene"]["ok"] and c["match"] and r["task"] is not None)
    # 2. the 'pack cabinet' mistake, then corrected
    bad = copy.deepcopy(good)
    bad["scene"]["add"][0] = {"name": "cabinet_1", "source": "pack", "noun": "cabinet", "relation": "on_table"}
    r = mode_a3.run(FakeModel([json.dumps(bad), json.dumps(good)]), session(), assets, "a small cabinet ...")
    case("v3 fake: 'pack cabinet' -> hint -> corrected", dict(attempts=r.get("attempts"), first_errors=r["log"][0]["errors"]),
         r["ok"] and r["attempts"] == 2 and "source cabinet" in r["log"][0]["errors"][0])
    # 3. closed drawer target, then corrected
    bad = copy.deepcopy(good)
    bad["scene"]["add"][1]["target"] = "cabinet_1/top"
    r = mode_a3.run(FakeModel([json.dumps(bad), json.dumps(good)]), session(), assets, "...")
    case("v3 fake: block into a closed drawer -> told which one is open -> corrected", dict(first_errors=r["log"][0]["errors"]),
         r["ok"] and r["attempts"] == 2 and "closed" in r["log"][0]["errors"][0])
    # 4. task only
    w_into = first["P-into"][1]
    r = mode_a3.run(FakeModel([w_into.model_dump_json()]), session(), assets, "put the orange block in the black bowl")
    c = compare(Request.model_validate(r["request"]), w_into)
    case("v3 fake: task only (scene null) -> no scene steps, task config made, matches", dict(steps=len(r["scene"]["steps"]), compare=c),
         r["ok"] and not r["scene"]["steps"] and c["match"] and r["task"] is not None)
    # 5. partial refusal with an alternative
    w_alt = first["P-alt-start"][1]
    r = mode_a3.run(FakeModel([w_alt.model_dump_json()]), session(), assets, "the orange block should start inside the middle drawer ...")
    c = compare(Request.model_validate(r["request"]), w_alt)
    case("v3 fake: partial refusal (scene) + alternative + task -> built, matches", dict(compare=c, refusals=r["request"]["infeasible"]),
         r["ok"] and r["scene"]["ok"] and c["match"])
    # 6. a block too tall for the drawer: the scene step resizes it -> the agent is asked once more about the task
    big = copy.deepcopy(good)
    big["scene"]["add"][1]["size_cm"] = 6
    r = mode_a3.run(FakeModel([json.dumps(big), json.dumps({"scene": None, "task": good["task"], "infeasible": []})]), session(), assets, "...")
    case("v3 fake: scene step resized the block -> second call for the task", dict(changed=r["scene"]["changed"], second=bool(r["second_call"]), attempts=r["attempts"]),
         r["ok"] and r["scene"]["changed"] and r["second_call"] is not None)
    # 7. task refers to an object that isn't there
    bad = copy.deepcopy(good)
    bad["task"]["pass_all"][0]["target"] = "blue_bowl"
    r = mode_a3.run(FakeModel([json.dumps(bad)] * 3), session(), assets, "...")
    case("v3 fake: task names a missing object 3 times -> fails cleanly with the reason", dict(ok=r["ok"], errors=r["log"][-1]["errors"]),
         not r["ok"] and "blue_bowl" in r["log"][-1]["errors"][0])

    json.dump(dict(results=results, fails=fails, seconds=round(time.time() - t0, 1)), open(os.path.join(OUT, "report.json"), "w"), indent=1, default=str)
    print(f"\n{len(results) - len(fails)}/{len(results)} passed in {time.time() - t0:.0f} s; fails: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
