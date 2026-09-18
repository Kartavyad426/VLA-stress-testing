#!/usr/bin/env bash
# Does half precision change how a policy responds to PERTURBATION?
#
# Our GR00T runs are bf16 because fp32 (12.6 GiB) does not fit the card; MINERVA
# runs in fp32 because it fits either way. So the roster already contains the
# controlled comparison we need, on MINERVA: same policy, same 506 variants, the
# only difference being the cast. R-022's parity check cannot answer this -- both
# of its arms were bf16, so it tested wiring, not precision, and it sat at
# ceiling, the regime where a precision effect is least visible.
#
# Paired by variant, so McNemar applies.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl
OUT=experiments/repro/runs/overnight_20260918
IDS=$(cat experiments/repro/lplus_fail_ids_nolang.txt)
exec 9>/tmp/vla_gpu.lock; flock 9   # GPU mutex (see experiments/gpu_lock.sh)
# Holding the lock is not the same as the card being free: CUDA memory is
# released asynchronously, and the previous holder's EGL render contexts can
# still be resident. MINERVA is 0.54 M parameters and still hit
# "CUDA error: out of memory" at 15:12 seconds after the render job exited.
for _ in $(seq 60); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$USED" -lt 400 ] && break
  sleep 10
done

echo "[$(date +%T)] precision A/B (MINERVA bf16) start" | tee -a "$OUT/progress.log"
.venvs/minerva-lplus/bin/python experiments/harness_eval.py --libero-plus \
  --checkpoint third_party/MINERVA/ckpt/t05_l1_0.54M --n-action-steps 1 \
  --override temporal_ensemble_coeff=0.01 --obs-size 360 --dtype bfloat16 \
  --suites libero_spatial --episodes 1 --tasks "$IDS" \
  --run-id lplus_fail_minerva_bf16 >> "$OUT/precision_ab.log" 2>&1
echo "[$(date +%T)] precision A/B done rc=$?" | tee -a "$OUT/progress.log"
python3 experiments/mine_run.py runs/lplus_fail_minerva runs/lplus_fail_minerva_bf16 >> "$OUT/mining.txt" 2>&1
