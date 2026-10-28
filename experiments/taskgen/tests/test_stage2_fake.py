"""Stage 2 without a GPU: templates, PnC counts, the executor on every orange preset, and the mode A loop with a fake
model (good, malformed, invalid, unfixable answers). Known-answer cases; exit 1 on any failure. Writes
runs/taskgen/stage2_fake/report.json and per-preset scene configs in the worktree.

    runs/real2sim/v1_dev/py_triton_first.sh ../VLA-exp-taskgen/experiments/taskgen/tests/test_stage2_fake.py
Exploratory, UNVERIFIED.
"""
import json, os, sys, time
import h5py

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from sim_mujoco.base import MAIN  # noqa: E402
from sim_mujoco.sim import MujocoSim  # noqa: E402
from sim_mujoco import assets  # noqa: E402
from core.session import Session  # noqa: E402
from core.templates import Template  # noqa: E402
from core.compare import compare  # noqa: E402
from core import executor  # noqa: E402
from agent.models import FakeModel  # noqa: E402
from agent import mode_a  # noqa: E402

WT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(WT, "runs/taskgen/stage2_fake")
P = json.load(open(os.path.join(HERE, "presets/orange.json")))
fails, results = [], {}


def case(name, got, ok):
    results[name] = got
    print(("PASS " if ok else "FAIL ") + name + ": " + json.dumps(got, default=str)[:350])
    if not ok:
        fails.append(name)


def session():
    q0 = h5py.File(os.path.join(MAIN, P["episode"], "trajectory.h5"))["observation/robot_state/joint_positions"][0]
    return Session(MujocoSim, assets, os.path.join(MAIN, P["scene"]), os.path.join(MAIN, P["episode"]), q0)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    tpl = {p["id"]: Template.model_validate(p["template"]) for p in P["presets"]}
    want_counts = {"P1_real_task": 3, "P2_stack_order": 6, "P3_added_block_into_bowl": 3, "P4_drawer_out_of": 3, "P5_infeasible_bowl_in_drawer": 1}
    counts = {k: t.count() for k, t in tpl.items()}
    case("PnC counts (3 blocks, 3! orders, 3 colours, 3 colours, 1)", counts, counts == want_counts)
    total, sample = tpl["P2_stack_order"].expand(cap=4, seed=0)
    case("expand with a cap reports the total and samples", dict(total=total, sampled=len(sample), first=sample[0][0]), total == 6 and len(sample) == 4)

    # executor on variant 0 of every preset, each on a fresh session
    for pid, t in tpl.items():
        values, req = t.expand()[1][0]
        S = session()
        r = executor.run(S, req)
        ok = r["ok"] and (("infeasible" in r) == pid.startswith("P5"))
        case(f"executor {pid} {values}", dict(ok=r["ok"], reason=r.get("reason"), infeasible=r.get("infeasible"),
                                              placed=[s["result"].get("placed") for s in r["steps"] if s["call"] == "place"]), ok)
        S.cfg.save(os.path.join(OUT, f"{pid}_v0_scene_config.json"))

    # mode A with a fake model, on P3 variant 0 (red block next to the bowl, task into the bowl)
    _, want = tpl["P3_added_block_into_bowl"].expand()[1][0]
    good = json.dumps(want.model_dump())
    renamed = json.loads(good)
    renamed["add"][0]["name"] = "my_red_cube"
    renamed["task"]["targets"] = ["my_red_cube"]
    renamed["task"]["pass_condition"]["object"] = "my_red_cube"

    S = session()
    r = mode_a.run(FakeModel(["```json\n" + json.dumps(renamed) + "\n```"]), S, "add a red block next to the bowl; the task is to put it in the bowl")
    cmp_ = compare(type(want).model_validate(r["request"]), want) if r["ok"] else None
    case("fake model: good answer (in a code fence, other names) -> built, matches the preset", dict(attempts=r["attempts"], exec_ok=r["exec"]["ok"], compare=cmp_),
         r["ok"] and r["exec"]["ok"] and cmp_["match"])

    S = session()
    bad_rel = json.loads(good)
    bad_rel["add"][0]["relation"] = "beside"
    r = mode_a.run(FakeModel(["{not json", json.dumps(bad_rel), good]), S, "add a red block next to the bowl")
    case("fake model: malformed JSON, then a bad relation, then good -> 3 attempts, errors sent back",
         dict(attempts=r["attempts"], errors=[l["errors"] for l in r["log"]]), r["ok"] and r["attempts"] == 3 and bool(r["log"][0]["errors"]) and "relation" in json.dumps(r["log"][1]["errors"]))

    S = session()
    bad_target = json.loads(good)
    bad_target["add"][0]["target"] = "blue_bowl"
    r = mode_a.run(FakeModel([json.dumps(bad_target)] * 3), S, "add a red block next to the blue bowl")
    case("fake model: unknown target three times -> fails cleanly with the reason", dict(ok=r["ok"], errors=r["log"][-1]["errors"]),
         not r["ok"] and "blue_bowl" in r["log"][-1]["errors"][0])

    S = session()
    move_base = {"add": [{"name": "black_bowl", "source": "pack", "noun": "bowl", "relation": "inside", "target": "drawer_1"}]}
    infeasible = {"add": [], "infeasible": "black_bowl is a locked base object"}
    r = mode_a.run(FakeModel([json.dumps(move_base), json.dumps(infeasible)]), S, "put the black bowl inside a drawer")
    cmp_ = compare(type(want).model_validate(r["request"]), tpl["P5_infeasible_bowl_in_drawer"].expand()[1][0][1]) if r["ok"] else None
    case("fake model: tries to move a base object -> told it's locked -> answers infeasible (matches P5)",
         dict(attempts=r["attempts"], first_errors=r["log"][0]["errors"], compare=cmp_), r["ok"] and r["attempts"] == 2 and cmp_["match"])

    S = session()
    wrong_colour = json.loads(good)
    wrong_colour["add"][0]["colour"] = "blue"
    wrong_colour["add"][0]["name"] = "blue_block_1"
    wrong_colour["task"]["targets"] = ["blue_block_1"]
    wrong_colour["task"]["pass_condition"]["object"] = "blue_block_1"
    r = mode_a.run(FakeModel([json.dumps(wrong_colour)]), S, "add a red block next to the bowl")
    cmp_ = compare(type(want).model_validate(r["request"]), want)
    case("compare catches a wrong colour (valid answer, wrong content)", cmp_, r["ok"] and not cmp_["match"] and not cmp_["objects_match"])

    json.dump(dict(results=results, fails=fails, seconds=round(time.time() - t0, 1)), open(os.path.join(OUT, "report.json"), "w"), indent=1, default=str)
    print(f"\n{len(results) - len(fails)}/{len(results)} passed in {time.time() - t0:.0f} s; fails: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
