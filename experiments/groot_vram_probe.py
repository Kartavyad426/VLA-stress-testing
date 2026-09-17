"""R-021(a): does GR00T N1.7 fit on this GPU? Weights only, no simulator.

    .venvs/groot/bin/python experiments/groot_vram_probe.py [--fp32 true|false]
"""
import argparse, json, sys, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="nvidia/gr00t17-lerobot-libero_spatial-640")
ap.add_argument("--fp32", default="true", help="model_params_fp32 as shipped (true) or overridden (false)")
ap.add_argument("--cast-bf16", action="store_true",
                help="cast every floating parameter to bfloat16 on CPU BEFORE moving to GPU. "
                     "model_params_fp32=false alone does not do this: the checkpoint is stored "
                     "F32 and the backbone is built with load_bf16=False (groot_n1_7.py:89).")
a = ap.parse_args()
fp32 = a.fp32.lower() == "true"

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

free0, total = torch.cuda.mem_get_info()
out = {"model_params_fp32": fp32, "cast_bf16": a.cast_bf16, "gpu_total_gib": round(total / 2**30, 2),
       "free_before_gib": round(free0 / 2**30, 2)}
t0 = time.time()
try:
    cfg = PreTrainedConfig.from_pretrained(a.checkpoint)
    cfg.base_model_path = "nvidia/GR00T-N1.7-3B"
    cfg.model_params_fp32 = fp32
    cfg.device = "cpu" if a.cast_bf16 else "cuda"   # cuda here loads straight onto the GPU, before any cast
    pol = get_policy_class(cfg.type).from_pretrained(a.checkpoint, config=cfg)
    if a.cast_bf16:
        pol = pol.to(torch.bfloat16)
    pol.to("cuda").eval()
    torch.cuda.synchronize()
    dt = {}
    for p in pol.parameters():
        dt[str(p.dtype)] = dt.get(str(p.dtype), 0) + p.numel()
    out.update(status="LOADED",
               param_dtypes={k: f"{v/1e9:.3f}B" for k, v in dt.items()},
               allocated_gib=round(torch.cuda.memory_allocated() / 2**30, 2),
               peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               free_after_gib=round(torch.cuda.mem_get_info()[0] / 2**30, 2))
except torch.OutOfMemoryError as e:
    out.update(status="OOM", error=str(e).splitlines()[0][:240],
               peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
except Exception as e:
    out.update(status="ERROR", error=f"{type(e).__name__}: {str(e)[:400]}")
out["load_s"] = round(time.time() - t0, 1)
print("VRAM_PROBE " + json.dumps(out))
