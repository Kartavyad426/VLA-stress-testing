"""R-060 G0c diagnosis, follow-up c5 (CPU/render only): the reused control 1132 renders
differently from a fresh one after its 2nd reset. Isolate: (i) delete another env, no reset;
(ii) reset again, no other env touched. Each render vs the env's own first render.
Writes runs/r060/gates/g0c_diag_c5.json."""
import gc, json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
from r060_g0c_diag import both, log, px
from r060_gates import OUT, make_env
from vla_harness.schema import PerturbationSpec

out = {}
for ci in (1132, 1327):
    c = make_env(ci); c.reset(0, PerturbationSpec()); R0 = px(c, True)
    r = {}
    o = make_env(745 if ci == 1132 else 952); o.reset(0, PerturbationSpec())
    r["after_other_env_created"] = both(px(c, True), R0)
    del o; gc.collect()
    r["after_other_env_deleted"] = both(px(c, True), R0)
    c.reset(0, PerturbationSpec())
    r["after_second_reset"] = both(px(c, True), R0)
    c2 = make_env(ci); c2.reset(0, PerturbationSpec()); c2.reset(0, PerturbationSpec())
    r["fresh_env_reset_twice_no_deletes_vs_R0"] = both(px(c2, True), R0)
    log("c5", ci, {k: (v["agent"]["n_diff"], v["wrist"]["n_diff"]) for k, v in r.items()})
    out[ci] = r; del c, c2; gc.collect()
json.dump(out, open(os.path.join(OUT, "g0c_diag_c5.json"), "w"), indent=1)
