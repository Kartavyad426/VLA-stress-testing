"""R-060 G0c diagnosis (approved 10-08 via prim): why does G0c fail? CPU and render only,
no model. Writes runs/r060/gates/g0c_diag.json (+ code state under g0c_diag_code_state/).
Does NOT change the gate definition in r060_gates.py; it only measures.

  ENV="PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8"
  env $ENV flock /tmp/vla_gpu.lock .venvs/libero-plus/bin/python experiments/r060_g0c_diag.py

a  Render timing vs restore, in a control env with no knob. robosuite's step loop is
   forward -> pre_action -> sim.step -> render (robosuite/environments/base.py:391-395), so
   the frame it returns is drawn from world poses (xpos) that lag qpos by one substep.
   `nominal_frames` / `perturbed_frames` call sim.forward() first, so they draw the CURRENT
   qpos. Checks: reset frame vs a re-render without / with forward; nominal_frames before
   and after a perturbed_frames call (restore); after one env step, the step frame vs a
   re-render from the state saved before the last substep (the lag, reproduced); and
   whether calling the paired renders every step changes the physics trajectory.
b  Re-render vs re-render on both sides, for the 25 G0c rows: P = control's
   perturbed_frames(variant) vs the variant env re-rendered (forward) at its own pose;
   N = variant's nominal_frames() vs the control re-rendered (forward), plus N with the
   variant's camera set to LIBERO-Plus's canonical pose instead of nominal._CAMERA.
c  Init states of the phys-mismatch variants (768, 1522, 1534) and a passing one (745)
   against their control 1132, by joint.
d  Motion blur (noise 6): repeatability of the ImageMagick call on one fixed input.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))

import numpy as np

from r060_gates import OUT, _cmp, catalogue, control_index, make_env  # noqa: E402

T0 = time.time()


def log(*x):
    print(f"[{time.time() - T0:6.0f}s]", *x, flush=True)


def parts(env):
    outer = env._env.unwrapped
    return outer, outer._env, outer._env.env, outer._env.env.sim


def raw_render(env, forward: bool):
    outer, _, inner, sim = parts(env)
    if forward:
        sim.forward()
    return inner._get_observations(force_update=True)


def px(env, forward: bool, raw=None):
    outer = parts(env)[0]
    raw = raw_render(env, forward) if raw is None else raw
    return dict(outer._format_raw_obs(raw)["pixels"])


def phys(env):
    sim = parts(env)[3]
    return np.concatenate([[sim.data.time], sim.data.qpos.ravel(), sim.data.qvel.ravel()])


def cam(env):
    from vla_harness.envs.libero_env import MAIN_CAMERA
    sim = parts(env)[3]
    cid = sim.model.camera_name2id(MAIN_CAMERA)
    return cid, np.array(sim.model.cam_pos[cid]), np.array(sim.model.cam_quat[cid])


def both(a, b):
    return {"agent": _cmp(a["image"], b["image"]), "wrist": _cmp(a["image2"], b["image2"])}


def code_state():
    d = os.path.join(OUT, "g0c_diag_code_state")
    os.makedirs(d, exist_ok=True)
    os.system(f"git rev-parse HEAD > {d}/HEAD; git diff > {d}/uncommitted.patch; "
              f"git status --porcelain > {d}/status.txt")


def step_a(cat, g0c_rows):
    from lerobot.envs.libero import get_libero_dummy_action
    from vla_harness.schema import PerturbationSpec
    out = []
    first_variant = {}
    for r in g0c_rows:
        if r["row"] == "camera":
            first_variant.setdefault(r["control"], r["variant"])
    for ci in (1030, 1132, 1327):
        c = make_env(ci)
        oc = c.reset(0, PerturbationSpec())
        outer, wrapper, inner, sim = parts(c)
        F0 = oc.frames
        row = {"control": ci, "variant_used": first_variant[ci]}
        xpos0 = sim.data.xpos.copy()
        r_nf = px(c, False)
        row["reset_vs_rerender_noforward"] = both(r_nf, F0)
        row["rerender_noforward_repeat"] = both(px(c, False), r_nf)
        p0 = phys(c)
        sim.forward()
        row["xpos_change_on_forward_max_m"] = float(np.abs(sim.data.xpos - xpos0).max())
        row["phys_unchanged_by_forward"] = bool(np.array_equal(phys(c), p0))
        r_f = px(c, False)
        row["reset_vs_rerender_forward"] = both(r_f, F0)
        n1 = c.nominal_frames()
        row["nominal_frames_vs_rerender_forward"] = both(n1, r_f)
        cam0 = cam(c)
        c.perturbed_frames(first_variant[ci])
        cam1 = cam(c)
        row["cam_restored_after_perturbed_frames"] = bool(np.array_equal(cam0[1], cam1[1]) and np.array_equal(cam0[2], cam1[2]))
        row["phys_unchanged_by_perturbed_frames"] = bool(np.array_equal(phys(c), p0))
        row["nominal_frames_before_vs_after_perturbed"] = both(c.nominal_frames(), n1)

        # one env step: reproduce the step frame from the state before its last substep
        snaps = []
        real_step = sim.step

        def spy(*a, **k):
            snaps.append(sim.get_state())
            return real_step(*a, **k)
        sim.step = spy
        try:
            raw_s, *_ = wrapper.step(get_libero_dummy_action())
        finally:
            sim.step = real_step
        S = dict(outer._format_raw_obs(raw_s)["pixels"])
        row["substeps_per_env_step"] = len(snaps)
        row["step_vs_rerender_noforward"] = both(px(c, False), S)
        cur = sim.get_state()
        sim.set_state(snaps[-1]); sim.forward()
        row["step_vs_forward_render_of_state_before_last_substep"] = both(px(c, False), S)
        sim.set_state(cur); sim.forward()
        row["step_vs_rerender_forward"] = both(px(c, False), S)
        log("a", ci, {k: v for k, v in row.items() if not isinstance(v, dict)},
            {k: (v["agent"]["n_diff"], v["wrist"]["n_diff"]) for k, v in row.items() if isinstance(v, dict) and "agent" in v})
        out.append(row)
        del c

    # do the paired renders, called every step, change the physics? (R-039/R-042-style use)
    ci, var = 1030, first_variant[1030]
    A, B = make_env(ci), make_env(ci)
    A.reset(0, PerturbationSpec()); B.reset(0, PerturbationSpec())
    wa, wb = parts(A)[1], parts(B)[1]
    frames_equal = []
    for k in range(40):
        act = np.array([0.4 * np.sin(k / 5), 0.4 * np.cos(k / 5), 0.2 * np.sin(k / 9), 0, 0, 0, -1 if k < 25 else 1],
                       dtype=np.float64)
        A.nominal_frames(); A.perturbed_frames(var)
        ra, *_ = wa.step(act)
        rb, *_ = wb.step(act)
        frames_equal.append(bool(np.array_equal(ra["agentview_image"], rb["agentview_image"])
                                 and np.array_equal(ra["robot0_eye_in_hand_image"], rb["robot0_eye_in_hand_image"])))
    pa, pb = phys(A), phys(B)
    traj = {"control": ci, "steps": 40, "phys_equal_after": bool(np.array_equal(pa, pb)),
            "phys_max_abs": float(np.abs(pa - pb).max()), "step_frames_equal_all": all(frames_equal),
            "first_unequal_step": next((i for i, e in enumerate(frames_equal) if not e), None)}
    log("a traj", traj)
    del A, B
    return out, traj


def step_b(cat, g0c_rows):
    from vla_harness.envs import nominal
    from vla_harness.envs.libero_env import lplus_sensor_noise
    from vla_harness.schema import PerturbationSpec
    controls, out = {}, []
    for r in g0c_rows:
        tid, ci, name = r["task_id"], r["control"], r["variant"]
        if ci not in controls:
            controls[ci] = make_env(ci)
        c, v = controls[ci], make_env(tid)
        ov = v.reset(0, PerturbationSpec()); c.reset(0, PerturbationSpec())
        row = {"row": r["row"], "task_id": tid, "control": ci, "variant": name,
               "phys_equal": bool(np.array_equal(phys(v), phys(c)))}
        ref_N = px(c, True)                                  # control at its current qpos
        if r["row"] == "camera":
            P = c.perturbed_frames(name)
            row["P_vs_variant_rerender"] = both(P, px(v, True))
            row["variant_reset_vs_variant_rerender_noforward"] = both(px(v, False), ov.frames)
            row["N_vs_control_rerender"] = both(v.nominal_frames(), ref_N)
            # N with the canonical LIBERO-Plus pose instead of nominal._CAMERA
            cid, p0, q0 = cam(v)
            sim = parts(v)[3]
            cp, cq = v._bench_camera_pose(0, 0, 1.0)
            dom = type(parts(v)[2]).__module__.rsplit(".", 1)[-1].removeprefix("libero_").removesuffix("_manipulation")
            npos, nq = nominal.nominal_camera(dom)
            _, ccp, ccq = cam(c)
            row["cam"] = {"control_model_pos": ccp.tolist(), "control_model_quat": ccq.tolist(),
                          "bench_000_pos": cp.tolist(), "bench_000_quat": cq.tolist(),
                          "nominal_CAMERA_pos": npos, "nominal_CAMERA_quat": nq,
                          "nominal_vs_control_pos_max": float(np.abs(np.array(npos) - ccp).max()),
                          "bench_vs_control_pos_max": float(np.abs(cp - ccp).max()),
                          "nominal_vs_control_quat_max": float(min(np.abs(np.array(nq) - ccq).max(), np.abs(np.array(nq) + ccq).max())),
                          "bench_vs_control_quat_max": float(min(np.abs(cq - ccq).max(), np.abs(cq + ccq).max()))}
            try:
                sim.model.cam_pos[cid] = cp; sim.model.cam_quat[cid] = cq
                row["N_canonical_pose_vs_control_rerender"] = both(px(v, True), ref_N)
            finally:
                sim.model.cam_pos[cid] = p0; sim.model.cam_quat[cid] = q0; sim.forward()
        else:
            n = r["noise"]
            P = c.perturbed_frames(name, noise_seed=0)
            raw_v = dict(raw_render(v, True))
            raw_v["agentview_image"] = lplus_sensor_noise(raw_v["agentview_image"], n, 0)
            row["P_vs_noise_of_variant_rerender_seed0"] = both(P, px(v, False, raw=raw_v))
            if r["family"] in ("gaussian", "zoom"):          # deterministic: rebuild the reset frame from the stale render
                raw_s = dict(raw_render(v, False))
                raw_s["agentview_image"] = lplus_sensor_noise(raw_s["agentview_image"], n)
                row["variant_reset_vs_noise_of_rerender_noforward"] = both(px(v, False, raw=raw_s), ov.frames)
            row["N_vs_control_rerender"] = both(v.nominal_frames(), ref_N)
        log("b", tid, ci, row["phys_equal"],
            {k: (x["agent"]["n_diff"], x["wrist"]["n_diff"]) for k, x in row.items() if isinstance(x, dict) and "agent" in x})
        out.append(row)
        del v
    del controls
    return out


def step_c(cat):
    from lerobot.envs.libero import _get_suite, get_task_init_states
    suite = _get_suite("libero_spatial")
    ci = 1132
    c = make_env(ci)
    from vla_harness.schema import PerturbationSpec
    c.reset(0, PerturbationSpec())
    m = parts(c)[3].model
    nq = m.nq
    qidx = {}
    for j in range(m.njnt):
        a = int(m.jnt_qposadr[j]); w = {0: 7, 1: 4}.get(int(m.jnt_type[j]), 1)
        for k in range(w):
            qidx[a + k] = m.joint_id2name(j)
    S_c = get_task_init_states(suite, ci, is_libero_plus=True)
    out = {"control": ci, "control_name": cat[ci]["name"], "control_init_file": suite.get_task(ci).init_states_file,
           "control_bddl": suite.get_task(ci).bddl_file, "control_shape": list(S_c.shape), "nq": nq, "rows": []}
    for tid in (768, 1522, 1534, 745):
        S_v = get_task_init_states(suite, tid, is_libero_plus=True)
        row = {"task_id": tid, "name": cat[tid]["name"], "init_file": suite.get_task(tid).init_states_file,
               "bddl": suite.get_task(tid).bddl_file, "shape": list(S_v.shape)}
        if S_v.shape[1:] == S_c.shape[1:]:
            d = np.abs(S_v[0] - S_c[0])
            row["row0_equal"] = bool(np.array_equal(S_v[0], S_c[0]))
            row["row0_max_abs"] = float(d.max())
            diff = [int(i) for i in np.nonzero(d > 0)[0]]
            row["row0_diff_by_joint"] = {}
            for i in diff:
                q = i - 1
                key = "time" if i == 0 else (qidx.get(q, f"qpos[{q}]") if q < nq else f"qvel[{q - nq}]")
                cur = row["row0_diff_by_joint"].setdefault(key, 0.0)
                row["row0_diff_by_joint"][key] = max(cur, float(d[i]))
        log("c", tid, {k: v for k, v in row.items() if k != "row0_diff_by_joint"}, row.get("row0_diff_by_joint"))
        out["rows"].append(row)
    del c
    return out


def step_d():
    from vla_harness.envs.libero_env import lplus_sensor_noise
    from vla_harness.schema import PerturbationSpec
    c = make_env(984)
    c.reset(0, PerturbationSpec())
    img = np.array(raw_render(c, True)["agentview_image"])
    del c
    outs = [lplus_sensor_noise(img, 6, 0) for _ in range(20)]
    distinct = len({o.tobytes() for o in outs})
    maxdiff = max(_cmp(o, outs[0])["n_diff"] for o in outs)
    p = os.path.join(OUT, "g0c_diag_motion_input.npy")
    np.save(p, img)
    probe = ("import sys,numpy as np; sys.path.insert(0,'.');"
             "from vla_harness.envs.libero_env import lplus_sensor_noise as f;"
             f"i=np.load('{p}'); o=[f(i,6,0).tobytes() for _ in range(20)]; print(len(set(o)))")
    res = {"input": p, "in_process_distinct_of_20": distinct, "in_process_max_ndiff": maxdiff,
           "MAGICK_THREAD_LIMIT_env": os.environ.get("MAGICK_THREAD_LIMIT")}
    for lim in ("1", None):
        e = dict(os.environ)
        if lim:
            e["MAGICK_THREAD_LIMIT"] = lim
        r = subprocess.run([sys.executable, "-c", probe], env=e, capture_output=True, text=True)
        res[f"subprocess_distinct_of_20_threads_{lim or 'default'}"] = (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1]
    log("d", res)
    return res


def main():
    code_state()
    cat = catalogue()
    g0c_rows = json.load(open(os.path.join(OUT, "g0c.json")))["rows"]
    res = {"what": "R-060 G0c diagnosis, CPU/render only, no gate change", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    res["d_motion"] = step_d()
    res["a_render_timing"], res["a_trajectory"] = step_a(cat, g0c_rows)
    res["c_init_states"] = step_c(cat)
    res["b_rerender_pairs"] = step_b(cat, g0c_rows)
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p = os.path.join(OUT, "g0c_diag.json")
    json.dump(res, open(p, "w"), indent=1, default=str)
    log("wrote", p)


if __name__ == "__main__":
    main()
