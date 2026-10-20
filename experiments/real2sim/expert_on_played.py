"""Scripted expert on the layouts GR00T actually played in slot A / D2 (pre-fix policy_batch: rollout --seed = the
expert json's full seed, so layout rng = json_seed*1000 + task*100 + variant). make_tasks --seed <json_seed> reproduces it."""
import glob, json, os, subprocess
T = "runs/real2sim/tasks_stereo/ep_orange_block_stereo"; OUT = "runs/real2sim/tasks_stereo_played"
res = []
for vj in sorted(glob.glob(f"{T}/*/v[1-4].json")):
    j = json.load(open(vj))
    o = f"{OUT}/seed{j['seed']}"
    subprocess.run([".venvs/real2sim/bin/python", "experiments/real2sim/make_tasks.py", "--scene",
                    "runs/real2sim/ep_orange_block_stereo:data/real2sim/droid/ep_orange_block", "--out", o, "--seed", str(j["seed"]),
                    "--only", j["instruction"], "--variant_list", str(j["variant"])], env=dict(os.environ, MUJOCO_GL="egl"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
    got = glob.glob(f"{o}/ep_orange_block_stereo/{os.path.basename(os.path.dirname(vj))}/v{j['variant']}.json")
    ok = json.load(open(got[0]))["success"] if got else None
    res.append(dict(task=j["instruction"], variant=j["variant"], json_seed=j["seed"], expert_success_on_played_layout=ok))
    print(j["instruction"], "v", j["variant"], "->", ok, flush=True)
json.dump(res, open(f"{OUT}/expert_on_played.json", "w"), indent=1)
print("expert solves", sum(1 for r in res if r["expert_success_on_played_layout"]), "of", len(res))
