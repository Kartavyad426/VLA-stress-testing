"""lerobot-eval, unchanged except that the policy is loaded on CPU, cast to bf16,
then moved to the GPU. R-021.

Why a wrapper: GR00T N1.7 LIBERO checkpoints are stored F32 (3.144B params) and
the backbone is built with load_bf16=False, so every stock load path puts
~11.7 GiB on the GPU and OOMs on an 8 GB card at 7.2 GiB. Neither
`model_params_fp32=false` nor a cast after `from_pretrained` helps, because
`from_pretrained` already places the model on `cfg.device`. Loading on CPU,
casting, then moving gives 5.87 GiB (measured).

THIS IS A DEPARTURE FROM THE SHIPPED CONFIG: all parameters, including norms
and the action head, run in bf16 rather than F32 under bf16 autocast. Results
from this path must be labelled as such. Everything else is stock lerobot-eval.

    MUJOCO_GL=egl .venvs/groot/bin/python experiments/groot_eval_bf16.py <lerobot-eval args>
"""
import torch
import lerobot.scripts.lerobot_eval as le

_orig_make_policy = le.make_policy


def make_policy_bf16(cfg, *args, **kwargs):
    target = cfg.device or "cuda"
    cfg.device = "cpu"
    policy = _orig_make_policy(cfg, *args, **kwargs)
    policy = policy.to(torch.bfloat16).to(target)
    cfg.device = target
    free, total = torch.cuda.mem_get_info()
    print(f"[groot_eval_bf16] policy on {target} in bf16: "
          f"{torch.cuda.memory_allocated()/2**30:.2f} GiB allocated, "
          f"{free/2**30:.2f} GiB free of {total/2**30:.2f}", flush=True)
    return policy


le.make_policy = make_policy_bf16

if __name__ == "__main__":
    le.main()
