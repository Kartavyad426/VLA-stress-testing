"""R-060 G0c diagnosis, follow-up c4 (CPU/render only): does the REUSED control env 1132
also RENDER differently from a fresh one (beyond its ~1e-9 qpos drift)? Same order as G0c.
Writes runs/r060/gates/g0c_diag_c4.json."""
import json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
from r060_g0c_diag import both, log, parts, px
from r060_gates import OUT, make_env
from vla_harness.schema import PerturbationSpec
import numpy as np

c = make_env(1132); rows = []
for tid in (745, 768, 1522, 1534):
    v = make_env(tid)
    v.reset(0, PerturbationSpec()); c.reset(0, PerturbationSpec())
    fresh = make_env(1132); fresh.reset(0, PerturbationSpec())
    rc, rf, rv = px(c, True), px(fresh, True), px(v, True)
    sim_c, sim_f = parts(c)[3], parts(fresh)[3]
    r = {"task_id": tid, "reused_vs_fresh_render": both(rc, rf), "variant_vs_fresh_render": both(rv, rf),
         "light_equal": all(np.array_equal(getattr(sim_c.model, k), getattr(sim_f.model, k))
                            for k in ("light_pos", "light_dir", "light_diffuse", "light_specular")),
         "cam_pos_equal": bool(np.array_equal(sim_c.model.cam_pos, sim_f.model.cam_pos)),
         "geom_rgba_equal": bool(np.array_equal(sim_c.model.geom_rgba, sim_f.model.geom_rgba)),
         "qpos_max": float(np.abs(sim_c.data.qpos - sim_f.data.qpos).max())}
    log("c4", r); rows.append(r); del v, fresh
json.dump(rows, open(os.path.join(OUT, "g0c_diag_c4.json"), "w"), indent=1)
