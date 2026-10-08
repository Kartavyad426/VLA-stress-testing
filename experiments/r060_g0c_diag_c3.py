"""R-060 G0c diagnosis, follow-up c3 (CPU/render only): the 3/25 phys mismatches are all
control 1132 rows AFTER its first use. Replays G0c's order (one cached control env, reset
per row) and splits the mismatch into time / qpos / qvel. Writes runs/r060/gates/g0c_diag_c3.json."""
import json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
import numpy as np
from r060_g0c_diag import log, parts
from r060_gates import OUT, make_env
from vla_harness.schema import PerturbationSpec

def split(v, c):
    a, b = parts(v)[3].data, parts(c)[3].data
    return {"time": [float(a.time), float(b.time)],
            "qpos_max": float(np.abs(a.qpos - b.qpos).max()), "qvel_max": float(np.abs(a.qvel - b.qvel).max()),
            "qvel_nonzero_v_c": [int((a.qvel != 0).sum()), int((b.qvel != 0).sum())]}

c = make_env(1132); rows = []
for tid in (745, 768, 1522, 1534, 745):
    v = make_env(tid)
    v.reset(0, PerturbationSpec()); c.reset(0, PerturbationSpec())
    r = {"task_id": tid, "control_resets_so_far": len(rows) + 1, **split(v, c)}
    fresh = make_env(1132); fresh.reset(0, PerturbationSpec())
    r["fresh_control_vs_reused_control"] = split(fresh, c)
    r["variant_vs_fresh_control"] = split(v, fresh)
    log("c3", r); rows.append(r); del v, fresh
json.dump(rows, open(os.path.join(OUT, "g0c_diag_c3.json"), "w"), indent=1)
