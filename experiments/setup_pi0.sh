#!/usr/bin/env bash
# Fourth policy: pi0-FAST and pi0.5, probed then smoke-tested.
#
# Both pass the conformance gate as drop-ins after the 2026-09-18 gate fix: the
# checkpoints ship their own rename map (pi0-FAST: image -> base_0_rgb) and 8-D
# normalisation statistics, so neither needs a --rename-map or a state shim.
#
# The open question is VRAM, and it is measured here rather than argued:
#   pi0-FAST 2.92 B -> 5.44 GiB of weights   (GR00T, which fits, is 5.86)
#   pi0.5    3.62 B -> 6.74 GiB of weights   (0.8 GiB headroom vs GR00T's 1.4)
# Each probe loads on CPU, casts to bf16, moves to GPU, and reports peak usage --
# the same route that made GR00T fit (R-020).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl
OUT=experiments/repro/runs/pi0_setup; mkdir -p "$OUT"
say() { echo "[$(date +%T)] $*" | tee -a "$OUT/progress.log"; }

while pgrep -f "[h]arness_eval.py" >/dev/null || pgrep -f "[f]ollowon_20260918.sh" >/dev/null \
   || pgrep -f "[b]aseline_same_stack.sh" >/dev/null || pgrep -f "[p]recision_ab.sh" >/dev/null \
   || pgrep -f "[v]isualise_episode_video.py" >/dev/null; do sleep 60; done
say "=== pi0 setup start (GPU free) ==="

probe () {  # name, checkpoint
  say "VRAM probe $1"
  .venvs/libero-plus/bin/python - "$2" > "$OUT/vram_$1.log" 2>&1 <<'PY'
import sys, torch
from lerobot.policies.factory import make_policy_config          # noqa
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.factory import get_policy_class
from lerobot.configs.policies import PreTrainedConfig
ckpt = sys.argv[1]
cfg = PreTrainedConfig.from_pretrained(ckpt)
cfg.device = "cpu"
cls = get_policy_class(cfg.type)
p = cls.from_pretrained(ckpt, config=cfg)
p = p.to(dtype=torch.bfloat16).to("cuda").eval()
torch.cuda.synchronize()
free, total = torch.cuda.mem_get_info()
print(f"RESULT loaded_ok alloc={torch.cuda.memory_allocated()/2**30:.2f}GiB "
      f"free={free/2**30:.2f}GiB of {total/2**30:.2f}GiB")
PY
  grep -E "RESULT|OutOfMemory|Error" "$OUT/vram_$1.log" | tail -3 | tee -a "$OUT/progress.log"
}

smoke () {  # name, checkpoint, obs_size, nas
  say "smoke $1 (3 tasks x 2 eps, vanilla libero_spatial)"
  .venvs/libero-plus/bin/python experiments/harness_eval.py \
    --checkpoint "$2" --obs-size "$3" --n-action-steps "$4" --dtype bfloat16 \
    --suites libero_spatial --tasks 0,1,2 --episodes 2 --run-id "pi0_smoke_$1" \
    > "$OUT/smoke_$1.log" 2>&1
  say "smoke $1 rc=$? -> $(grep -c . runs/pi0_smoke_$1/rollouts.jsonl 2>/dev/null || echo 0) rollouts"
  tail -4 "$OUT/smoke_$1.log" | tee -a "$OUT/progress.log"
}

probe pi0fast lerobot/pi0fast-libero-v044
probe pi05    lerobot/pi05_libero_finetuned_v044

grep -q "RESULT loaded_ok" "$OUT/vram_pi0fast.log" && smoke pi0fast lerobot/pi0fast-libero-v044 224 10 \
  || say "pi0-FAST: probe failed, smoke skipped"
grep -q "RESULT loaded_ok" "$OUT/vram_pi05.log" && smoke pi05 lerobot/pi05_libero_finetuned_v044 256 50 \
  || say "pi0.5: probe failed, smoke skipped"
say "=== pi0 setup complete ==="
notify-send -u critical "VLA: pi0 setup done" "$(tail -2 "$OUT/progress.log" | head -1)"
