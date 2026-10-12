"""Does building GR00T under default dtype bf16 give the same params+buffers (sha) with a lower peak? CPU only."""
import sys, os, json, hashlib, contextlib
sys.path.insert(0, os.getcwd())
import torch
import vla_harness.policies.lerobot_policy as LP
mode = sys.argv[1]
if mode == "bf16build":
    orig = LP._from_pretrained_cast
    def patched(policy_cls, checkpoint, cfg, cast):
        prev = torch.get_default_dtype(); torch.set_default_dtype(cast)
        try:
            return orig(policy_cls, checkpoint, cfg, cast)
        finally:
            torch.set_default_dtype(prev)
    LP._from_pretrained_cast = patched
sys.argv = [sys.argv[0]]
from experiments.r044_reverse import build_policy
pol = build_policy(); pol.device = "cpu"; pol._load()
vm = {l.split(":")[0]: round(int(l.split()[1]) / 2**20, 2) for l in open("/proc/self/status") if l.startswith(("VmRSS", "VmHWM"))}
h = {}
for k, v in [("P:" + n, t) for n, t in pol._policy.named_parameters(remove_duplicate=False)] + \
            [("B:" + n, t) for n, t in pol._policy.named_buffers(remove_duplicate=False)]:
    t = v.detach().contiguous().cpu()
    h[k] = hashlib.sha1(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"").hexdigest()[:16] + ":" + str(t.dtype)
print(mode, vm, "sha1", hashlib.sha1(json.dumps(h, sort_keys=True).encode()).hexdigest(), len(h), "default dtype now", torch.get_default_dtype())
json.dump(h, open(f"runs/impl_smoke/bf16build_{mode}.json", "w"))
