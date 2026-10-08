"""R-060 G0c diagnosis, follow-ups (CPU/render only):
c2  The 1132-scene variants 768/1522/1534 share control 1132's init state bitwise (g0c_diag.json),
    yet their physics differs after reset. Where does it come in? Compares the compiled models
    (sizes, body names, body pos/quat, geom sizes, a hash of the model XML) and qpos by joint,
    both right after set_init_state (before the reset's settle steps) and after reset. 745
    (passes) is the reference.
d2  Motion 6 (variant 1377, control 984): is the off-by-1 the blur of a few-pixel render jitter?
    Two forward re-renders of the same state are compared clean and after the same seeded blur.
Writes runs/r060/gates/g0c_diag_c2.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))

import numpy as np

from r060_g0c_diag import log, parts, raw_render  # noqa: E402
from r060_gates import OUT, _cmp, catalogue, make_env  # noqa: E402


def qnames(m):
    out = {}
    for j in range(m.njnt):
        a = int(m.jnt_qposadr[j]); w = {0: 7, 1: 4}.get(int(m.jnt_type[j]), 1)
        for k in range(w):
            out[a + k] = m.joint_id2name(j)
    return out


def model_digest(env):
    sim = parts(env)[3]; m = sim.model
    xml = sim.model.get_xml() if hasattr(sim.model, "get_xml") else ""
    return {"nq": m.nq, "nbody": m.nbody, "ngeom": m.ngeom, "nlight": m.nlight,
            "bodies": [m.body_id2name(i) for i in range(m.nbody)],
            "xml_sha": hashlib.sha1(xml.encode()).hexdigest()[:12] if xml else None}


def by_joint(a, b, names):
    d = np.abs(np.asarray(a) - np.asarray(b)); out = {}
    for i in np.nonzero(d > 0)[0]:
        out[names.get(int(i), f"q{i}")] = max(out.get(names.get(int(i), f"q{i}"), 0.0), float(d[i]))
    return out


def main():
    from lerobot.envs.libero import _get_suite, get_task_init_states
    from vla_harness.schema import PerturbationSpec
    cat = catalogue()
    suite = _get_suite("libero_spatial")
    res = {"c2": [], "d2": None}
    c = make_env(1132); c.reset(0, PerturbationSpec())
    mc, sc = model_digest(c), parts(c)[3]
    names = qnames(sc.model)
    init = get_task_init_states(suite, 1132, is_libero_plus=True)[0]

    def at_init(env):
        w = parts(env)[1]
        w.set_init_state(init)
        return parts(env)[3].data.qpos.copy()

    q_c_reset = sc.data.qpos.copy()
    q_c_init = at_init(c)
    for tid in (768, 1522, 1534, 745):
        v = make_env(tid); v.reset(0, PerturbationSpec())
        mv, sv = model_digest(v), parts(v)[3]
        row = {"task_id": tid, "name": cat[tid]["name"],
               "same_sizes": all(mv[k] == mc[k] for k in ("nq", "nbody", "ngeom", "nlight")),
               "same_body_names": mv["bodies"] == mc["bodies"], "xml_sha": [mv["xml_sha"], mc["xml_sha"]],
               "body_pos_max_abs": float(np.abs(sv.model.body_pos - sc.model.body_pos).max()) if mv["nbody"] == mc["nbody"] else None,
               "body_quat_max_abs": float(np.abs(sv.model.body_quat - sc.model.body_quat).max()) if mv["nbody"] == mc["nbody"] else None,
               "geom_size_max_abs": float(np.abs(sv.model.geom_size - sc.model.geom_size).max()) if mv["ngeom"] == mc["ngeom"] else None,
               "qpos_after_reset_diff_by_joint": by_joint(sv.data.qpos, q_c_reset, names)}
        if mv["nbody"] == mc["nbody"]:
            bp = np.abs(sv.model.body_pos - sc.model.body_pos).max(1)
            row["bodies_pos_differ"] = {mc["bodies"][i]: float(bp[i]) for i in np.nonzero(bp > 0)[0]}
        row["qpos_at_set_init_state_diff_by_joint"] = by_joint(at_init(v), q_c_init, names)
        log("c2", tid, row)
        res["c2"].append(row)
        del v
    del c

    from vla_harness.envs.libero_env import lplus_sensor_noise
    c = make_env(984); c.reset(0, PerturbationSpec())
    r1 = np.array(raw_render(c, True)["agentview_image"]); r2 = np.array(raw_render(c, True)["agentview_image"])
    reps = [np.array(raw_render(c, True)["agentview_image"]) for _ in range(10)]
    res["d2"] = {"clean_rerender_vs_rerender": _cmp(r1, r2),
                 "clean_distinct_of_10": len({x.tobytes() for x in reps}),
                 "clean_max_ndiff_of_10": max(_cmp(x, reps[0])["n_diff"] for x in reps),
                 "blur6_of_rerenders": _cmp(lplus_sensor_noise(r1, 6, 0), lplus_sensor_noise(r2, 6, 0)),
                 "blur6_of_10_distinct": len({lplus_sensor_noise(x, 6, 0).tobytes() for x in reps})}
    log("d2", res["d2"])
    p = os.path.join(OUT, "g0c_diag_c2.json")
    json.dump(res, open(p, "w"), indent=1)
    log("wrote", p)


if __name__ == "__main__":
    main()
