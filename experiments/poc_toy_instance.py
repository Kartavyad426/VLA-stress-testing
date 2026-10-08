"""PoC toy, stage 1: one failed rollout -> one mining-spec instance (D13). CPU.

The case (user, 10-08): base GR00T N1.7, LIBERO-Spatial scene 984, start pose
0.2 rad off (joint direction 1), noise seed 0 -- runs/r047/rollouts.jsonl row
14, which failed. Row 20 is the SAME start under noise seed 1 and succeeded;
row 0 is the nominal start. Rows by index: rollout_id is shared by 14 and 20.

    .venvs/libero-plus/bin/python experiments/poc_toy_instance.py
    -> runs/poc_toy/instance.json, runs/poc_toy/traces.json (plot data)

Root = the first grasp onset that does not meet the success criterion. The
divergence from the same-start success is reported next to it, using
divergence_point's rule (end-effector xy apart by more than 3 cm), computed
inline because that function reads an `ee_xy` series LIBERO traces don't carry.
"""
import json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poc_toy_common import (R047, ROOT, TARGET, LIFT_M, WINDOW, kin_model,
                            rollout_events)
from vla_harness.schema import rollout_from_dict
from vla_harness.mining.phases_libero import LiberoPhaseSegmenter, DEFAULT_LIFT_M
from vla_harness.mining.phases import terminal_behaviour

OUT = os.path.join(ROOT, "runs/poc_toy")
FAIL, SAME_START_OK, NOMINAL = 14, 20, 0
DIV_TOL_M = 0.03


def main():
    rows = [json.loads(l) for l in open(R047)]
    assert rows[FAIL]["rollout_id"] == rows[SAME_START_OK]["rollout_id"] == "d46c2c057325"
    assert not rows[FAIL]["success"] and rows[SAME_START_OK]["success"] and rows[NOMINAL]["success"]
    m, D = kin_model()
    ev = {i: rollout_events(rows[i], m, D) for i in (FAIL, SAME_START_OK, NOMINAL)}
    for i, (_, err) in ev.items():
        assert err < 0.002, f"row {i}: FK disagrees with stored eef by {err:.4f} m"

    fail = rows[FAIL]
    r = rollout_from_dict(fail)
    segs, info = LiberoPhaseSegmenter()(r)
    term = terminal_behaviour(r, segs)

    # root: first grasp onset that fails the criterion
    root_ev = next(e for e in ev[FAIL][0] if not e["success"])
    mode = "slip" if root_ev["max_lift_m"] > DEFAULT_LIFT_M else "miss"

    # divergence from the same-start success
    a = [s["obs_state"]["eef_pos"] for s in fail["steps"]]
    b = [s["obs_state"]["eef_pos"] for s in rows[SAME_START_OK]["steps"]]
    div = None
    for t, (p, q) in enumerate(zip(a, b)):
        d = math.hypot(p[0] - q[0], p[1] - q[1])
        if d > DIV_TOL_M:
            div = {"t": t, "distance_m": round(d, 4), "tolerance_m": DIV_TOL_M}
            break

    last = fail["steps"][-1]["obs_state"]
    objs_end = last["_gt_scene_object_pos"]
    n_attempts = len(ev[FAIL][0])
    st_root = fail["steps"][root_ev["t"]]["obs_state"]
    inst = {
        "spec": {"policy": fail["policy_id"], "checkpoint": "nvidia/gr00t17-lerobot-libero_spatial-640",
                 "deployment": "LIBERO-Spatial via LIBERO-Plus harness (sim)", "phase": 0,
                 "schema_version": "toy-0 (illustrative; P6 pending)"},
        "instance": {
            "id": "toy-984-startpose-01",
            "unit": "segment",
            "scenario_ids": ["pick-and-place.bowl-to-plate", "grasp.slip"],
            "competence_domain": "in-domain + condition",
            "task": {"suite": "libero_spatial", "task_id": 984, "instruction": fail["instruction"]},
            "root": {"phase": "grasp", "failure_mode": mode, "t": root_ev["t"],
                     "policy_forward": root_ev["t"] // 16,
                     "seconds": round(root_ev["t"] / 20, 2),
                     "evidence": {"max_lift_m": root_ev["max_lift_m"], "window_steps": WINDOW,
                                  "slip_rule": f"lift > {DEFAULT_LIFT_M} m then < {LIFT_M} m"}},
            "divergence_from_same_start_success": div,
            "situation": {
                "at": "grasp onset (gripper commanded closed)",
                "observation_window": "1 frame (GR00T N1.7: state_horizon 1, no video history)",
                "raw": {"cameras": ["agentview", "wrist"], "frames": f"t={root_ev['t']} (re-render)",
                        "eef_pos": [round(x, 4) for x in st_root["eef_pos"]],
                        "joint_pos": [round(x, 4) for x in st_root["joint_pos"]],
                        "gripper_aperture_m": root_ev["aperture_m"]},
                "derived": {"eef_to_target_offset_m": root_ev["offset_m"],
                            "gripper_tilt_deg": root_ev["tilt_deg"],
                            "target_xyz": root_ev["target_xyz"],
                            "neighbours_rel_m": root_ev["neighbours_rel_m"]},
                "held_fixed": "all (mechanism unknown)",
                "condition": {"start_pose_radius_rad": 0.2, "joint_direction": 1,
                              "note": "the cause under test; not part of the identity (D12)"},
            },
            "success": {"template": "grasp.lift", "predicate":
                        f"target rises >= {LIFT_M*100:.0f} cm above rest within {WINDOW} steps of grasp onset"},
            "terminal": {"termination": fail["termination"], "env_steps": fail["env_steps"],
                         "behaviour": term, "grasp_attempts": n_attempts,
                         # classify.failure_cost reads a `holding` series LIBERO traces lack,
                         # so it cannot see a drop; measured here from the target's height.
                         "lifted_then_dropped": [{"grasp_t": e["t"], "max_lift_m": e["max_lift_m"]}
                                                 for e in ev[FAIL][0] if e["success"]],
                         "target_dist_to_plate_m": round(math.dist(objs_end[TARGET], objs_end["plate_1_main"]), 4),
                         "gripper_open_at_end": sum(abs(x) for x in last["gripper_qpos"]) > 0.03},
            "evidence": {"run": "runs/r047 (R-047)", "rows": {"failure": FAIL, "same_start_success": SAME_START_OK,
                         "nominal": NOMINAL}, "rollout_id": fail["rollout_id"], "env_seed": fail["seed"],
                         "noise_seeds": {"failure": 0, "same_start_success": 1},
                         "same_start_outcomes": "1 of 2 noise seeds failed at this start (R-047 cell)"},
            "mechanism": {"status": "untested",
                          "context": "R-044: on start-pose failures the wrist view carries the pose at forward 0; "
                                     "not localised for this instance"},
        },
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(inst, open(os.path.join(OUT, "instance.json"), "w"), indent=1)

    # plot data: offset, bowl lift, aperture per step, plus the grasp events
    traces = {}
    for i in (FAIL, SAME_START_OK, NOMINAL):
        S = rows[i]["steps"]
        z0 = S[0]["obs_state"]["_gt_scene_object_pos"][TARGET][2]
        traces[i] = {"success": rows[i]["success"],
                     "lift_cm": [round((s["obs_state"]["_gt_scene_object_pos"][TARGET][2] - z0) * 100, 2) for s in S],
                     "dist_cm": [round(math.dist(s["obs_state"]["eef_pos"], s["obs_state"]["_gt_scene_object_pos"][TARGET]) * 100, 2) for s in S],
                     "cmd_closed": [bool(s.get("action")) and s["action"][6] > 0.5 for s in S],
                     "events": [{k: v for k, v in e.items() if k != "_R"} for e in ev[i][0]]}
    traces["phases_failure"] = [(s.phase, s.t_start, s.t_end) for s in segs]
    json.dump(traces, open(os.path.join(OUT, "traces.json"), "w"))
    print(json.dumps(inst["instance"]["root"]), "\n", json.dumps(inst["instance"]["terminal"]),
          "\n div", div)


if __name__ == "__main__":
    main()
