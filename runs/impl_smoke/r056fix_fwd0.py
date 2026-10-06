"""V3/V5 (R-056 fix verification, 2026-09-29): forward-0 action chunks on the ten R-044 instances under fixed
noise, through the harness load path. One policy per process so peak RSS is the load's own.

  --load old   the pre-fix load (from_pretrained with the full fp32 state dict, then .to(bf16))
  --load new   the streaming bf16 load now in lerobot_policy.py
  --adapter D --alpha A   a LoRA adapter at WiSE alpha A
Writes <out>.npz (chunk per instance: raw predict_action_chunk and the post-processed first action) and <out>.json.
"""
import argparse, json, os, sys, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.getcwd())
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--load", choices=["old", "new"], default="new")
ap.add_argument("--adapter", default=None)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--out", required=True)
ap.add_argument("--obs-dump", default=None, help="pickle the rendered observations here")
ap.add_argument("--obs-load", default=None, help="use these pickled observations instead of rendering")
a = ap.parse_args()

import vla_harness.policies.lerobot_policy as LP
if a.load == "old":
    def _old(policy_cls, checkpoint, cfg, cast):          # the code before the fix, verbatim in effect
        p = policy_cls.from_pretrained(checkpoint, config=cfg)
        return p.to(cast) if cast is not None else p
    LP._from_pretrained_cast = _old


def vm():
    return {l.split(":")[0]: int(l.split()[1]) / 2**20 for l in open("/proc/self/status") if l.startswith(("VmRSS", "VmHWM"))}


sys.argv = [sys.argv[0]]
from experiments.r044_reverse import build_policy, make_env
from vla_harness.schema import PerturbationSpec
m0 = vm(); t0 = time.time()
pol = build_policy(a.adapter, a.alpha); pol.reset()
m1 = vm(); load_s = time.time() - t0
print(f"[fwd0] load={a.load} adapter={a.adapter} alpha={a.alpha}: RSS before {m0['VmRSS']:.2f} GiB, after load "
      f"{m1['VmRSS']:.2f} GiB, PEAK during load {m1['VmHWM']:.2f} GiB, {load_s:.0f}s; cuda alloc "
      f"{torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
dtypes = sorted({str(p.dtype) for p in pol._policy.parameters()})
import hashlib
sd_hash = {}
for k, v in [("P:" + n, t) for n, t in pol._policy.named_parameters(remove_duplicate=False)] + \
            [("B:" + n, t) for n, t in pol._policy.named_buffers(remove_duplicate=False)]:
    t = v.detach().contiguous().cpu()
    sd_hash[k] = hashlib.sha1(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"").hexdigest()[:16] + ":" + str(t.dtype)
all_hash = hashlib.sha1(json.dumps(sd_hash, sort_keys=True).encode()).hexdigest()
print(f"[fwd0] params+buffers: {len(sd_hash)} tensors, combined sha1 {all_hash}", flush=True)
if os.environ.get("HASH_ONLY"):
    json.dump({"load": a.load, "rss_peak_load_gib": m1["VmHWM"], "rss_after_load_gib": m1["VmRSS"], "load_s": load_s,
               "state_dict_sha1": all_hash, "tensor_hashes": sd_hash, "policy_id": pol.policy_id},
              open(a.out + ".json", "w"), indent=1)
    sys.exit(0)
sel = json.load(open("experiments/repro/r044_selection_ris.json"))
chunks, firsts, ids, obs_h = [], [], [], []
import pickle
frozen = pickle.load(open(a.obs_load, "rb")) if a.obs_load else None
dumped = []
for i, inst in enumerate(sel["instances"]):
    env = None if frozen else make_env(inst["task_id"])
    obs = frozen[i] if frozen else env.reset(inst["seed"], PerturbationSpec())
    dumped.append(obs)
    pol.reset()
    batch = pol._build_batch(obs)
    obs_h.append(hashlib.sha1(b"".join(np.ascontiguousarray(obs.frames[k]).tobytes() for k in sorted(obs.frames))
                              + np.asarray(list(obs.get("eef_pos")) + list(obs.get("eef_quat")), dtype=np.float64).tobytes()).hexdigest()[:16])
    torch.manual_seed(0)
    with torch.inference_mode():
        ch = pol._policy.predict_action_chunk(batch)
    chunks.append(ch.float().cpu().numpy())
    pol.reset()
    torch.manual_seed(0)
    firsts.append(np.asarray(pol(obs).values, dtype=np.float64))
    ids.append(inst["task_id"])
    del env
if a.obs_dump:
    pickle.dump(dumped, open(a.obs_dump, "wb"))
np.savez(a.out + ".npz", chunks=np.stack(chunks), firsts=np.stack(firsts), task_ids=np.array(ids), obs_hash=np.array(obs_h))
json.dump({"load": a.load, "adapter": a.adapter, "alpha": a.alpha, "rss_before_gib": m0["VmRSS"],
           "rss_after_load_gib": m1["VmRSS"], "rss_peak_load_gib": m1["VmHWM"], "load_s": load_s,
           "param_dtypes": dtypes, "state_dict_sha1": all_hash, "tensor_hashes": sd_hash, "policy_id": pol.policy_id, "chunk_shape": list(np.stack(chunks).shape)},
          open(a.out + ".json", "w"), indent=1)
print("[fwd0] done", a.out, np.stack(chunks).shape, dtypes, flush=True)
