"""E4 known-answer diagnosis (render only, no model): R-047's 984/n0 anchor differed by 0.049 in E4-1 while the other 5
were bitwise. 984/n0 was the FIRST env created in both processes (R-047's first rollout; E4's first anchor). Does the
first LiberoEnv in a process render its reset frame differently from later ones? Writes runs/root_signals/e4/anchor_firstenv.json."""
import json, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
from vla_harness.envs.libero_env import LiberoEnv
from vla_harness.schema import PerturbationSpec
fr = []
for k in range(4):
    for scene in (984, 1030):
        env = LiberoEnv(suite="libero_spatial", task_id=scene, libero_plus=True, libero_plus_base_instruction=True, obs_size=360)
        o = env.reset(0, PerturbationSpec())
        fr.append((k, scene, {c: np.array(v) for c, v in o.frames.items()}))
        del env
out = {}
for scene in (984, 1030):
    F = [f for k, s, f in fr if s == scene]
    out[scene] = [{c: int((F[i][c] != F[-1][c]).any(-1).sum()) for c in F[0]} for i in range(len(F))]
print(out)
json.dump({"px_diff_vs_last_creation, order=984,1030 x4": out}, open("runs/root_signals/e4/anchor_firstenv.json", "w"), indent=1)
