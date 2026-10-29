#!/usr/bin/env bash
# One GR00T episode on a TaskGen task (user 10-29: "I want to test groot on orange stereo scene and a task I am making
# rn on TaskGen"). GPU: holds /tmp/vla_gpu.lock for its whole life (prim's condition); refuses to start if the GPU is busy.
#   run_groot.sh <scene_config.json> <task_config.json> <out_dir> [seconds]
# Run from the main checkout. EGL rendering (not the CPU wrapper: that hides CUDA). Exploratory, UNVERIFIED.
set -uo pipefail
cd /home/imerit/Documents/Code/VLA
SC=$1; TC=$2; OUT=$3; SECS=${4:-30}
if [ "${TASKGEN_LOCKED:-0}" != "1" ]; then
    exec env TASKGEN_LOCKED=1 flock -n /tmp/vla_gpu.lock "$0" "$@" || { echo "GPU lock held by another job; not started" >&2; exit 1; }
fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$used" -gt 500 ]; then echo "GPU busy ($used MiB used); not started" >&2; exit 1; fi
mkdir -p "$OUT"
echo "$(date '+%F %T') start: scene $SC task $TC seconds $SECS; disk free $(df -h / | awk 'NR==2{print $4}')" | tee -a "$OUT/run.log"
# monitor: time, GPU MiB, MemAvailable MiB; prim's condition (c): abort if MemAvailable < 6 GiB
( while sleep 5; do
    ram=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    echo "$(date '+%T'),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1),$ram" >> "$OUT/gpu_monitor.csv"
    if [ "$ram" -lt 6144 ]; then
        echo "$(date '+%T') ABORT: MemAvailable $ram MiB < 6 GiB" >> "$OUT/run.log"
        pkill -f "taskgen/run_policy.py --scene_config"; pkill -f "policy_rollout.py --serve"; break
    fi
  done ) &
MON=$!
MUJOCO_GL=egl .venvs/real2sim/bin/python -c 'import triton' 2>/dev/null
MUJOCO_GL=egl .venvs/real2sim/bin/python ../VLA-exp-taskgen/experiments/taskgen/run_policy.py \
    --scene_config "$SC" --task_config "$TC" --policy groot --seconds "$SECS" --out "$OUT" 2>&1 | grep -v -i "warning" | tee -a "$OUT/run.log"
rc=${PIPESTATUS[0]}
kill $MON 2>/dev/null
echo "$(date '+%F %T') end rc $rc; peak GPU $(awk -F, '$2>m{m=$2}END{print m}' "$OUT/gpu_monitor.csv") MiB; min MemAvailable $(awk -F, 'NR==1||$3<m{m=$3}END{print m}' "$OUT/gpu_monitor.csv") MiB" | tee -a "$OUT/run.log"
# the policy server is a child of run_policy.py (GrootClient.close()); make sure none is left
pgrep -f "policy_rollout.py --serve" > /dev/null && { echo "policy server left running: killing" | tee -a "$OUT/run.log"; pkill -f "policy_rollout.py --serve"; }
exit $rc
