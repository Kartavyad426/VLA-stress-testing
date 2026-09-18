#!/usr/bin/env bash
# Runs after the MINERVA arm finishes. Renders the failures the overnight cap
# left out, mines both runs, and rebuilds the adjudication index.
#
# The first 80 renders were the first 80 failures in run order, which skewed to
# robot initial state (55 of 80) -- the class under suspicion, but it left camera
# viewpoint at 7 of 22 and sensor noise, lighting and background at zero. Hand
# adjudication needs the rare classes too: a family rule that is wrong only on
# 2 background failures is still wrong.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl
OUT=experiments/repro/runs/overnight_20260918
say() { echo "[$(date +%T)] $*" | tee -a "$OUT/progress.log"; }

while [ ! -f runs/groot_control_lplus_stack/DONE ]; do sleep 30; done
# serialise on a real lock, not on a pgrep pattern that also matches
# the launching shell (that deadlocked the control run twice today)
exec 9>/tmp/vla_gpu.lock; flock 9
# Holding the lock is not the same as the card being free: CUDA memory is
# released asynchronously, and the previous holder's EGL render contexts can
# still be resident. MINERVA is 0.54 M parameters and still hit
# "CUDA error: out of memory" at 15:12 seconds after the render job exited.
for _ in $(seq 60); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$USED" -lt 400 ] && break
  sleep 10
done

say "follow-on start"

say "render remaining GR00T failures"
timeout --foreground 7200 ./experiments/render_lplus_failures.sh runs/lplus_fail_groot viz/lplus_fail_groot 155 >> "$OUT/render.log" 2>&1
say "render MINERVA failures"
timeout --foreground 7200 ./experiments/render_lplus_failures.sh runs/lplus_fail_minerva viz/lplus_fail_minerva 120 >> "$OUT/render_minerva.log" 2>&1

say "mine + summarise"
python3 experiments/mine_run.py runs/lplus_fail_groot runs/lplus_fail_minerva \
  runs/lplus_hard_groot_v2 runs/lplus_hard_groot_rawinstr > "$OUT/mining.txt" 2>&1
python3 experiments/summarize_lplus.py runs/lplus_fail_groot runs/lplus_fail_minerva \
  runs/lplus_hard_groot_v2 runs/lplus_hard_groot_rawinstr > "$OUT/summary.txt" 2>&1
python3 experiments/adjudication_index.py runs/lplus_fail_groot viz/lplus_fail_groot >> "$OUT/progress.log" 2>&1
python3 experiments/adjudication_index.py runs/lplus_fail_minerva viz/lplus_fail_minerva >> "$OUT/progress.log" 2>&1
say "follow-on complete"
