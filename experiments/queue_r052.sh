#!/usr/bin/env bash
# R-052: the benchmark's own camera cone (camera_bench_* knobs), r047_run.py pattern.
# Axes in the registered order: yaw, then scale, then pitch; grids and 4 noise seeds are GRIDS in r047_run.py.
# Resumable: every axis resumes from runs/r052/<axis>/manifest.jsonl. Scenes whose camera setup ignores
# scale are dropped by r047_run.py and listed in runs/r052/camera_bench_scale/dropped_scenes.json.
#
#   bash experiments/queue_r052.sh                    # run to completion
#   R052_UNTIL=<epoch s> bash experiments/queue_r052.sh
#       stop 5 min before that time: SIGTERM, which r047_run.py honours at a rollout boundary
#       (the rollout in flight finishes and is recorded); SIGKILL only 15 min after that.
# Per-job flock (the countdown starts once the lock is held), rc logged, DONE marker only when every axis finished.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/libero-plus/bin/python; Q=runs/r052/queue.log; mkdir -p runs/r052; rm -f runs/r052/DONE runs/r052/PARTIAL
say() { echo "[$(date +%T)] $*" >> "$Q"; }
UNTIL=${R052_UNTIL:-}
axis_job() {   # axis_job <axis>
  local ax=$1 rc
  say "$ax start"
  if [ -n "$UNTIL" ]; then
    flock /tmp/vla_gpu.lock bash -c '
      left=$(( '"$UNTIL"' - $(date +%s) - 300 ))
      [ "$left" -gt 60 ] || exit 99
      exec timeout --preserve-status --signal=TERM --kill-after=900 "$left" "$@"' _ \
      $PY experiments/r047_run.py --run-id r052 --axis "$ax" >> runs/r052/$ax.log 2>&1
  else
    flock /tmp/vla_gpu.lock $PY experiments/r047_run.py --run-id r052 --axis "$ax" >> runs/r052/$ax.log 2>&1
  fi
  rc=$?
  if [ $rc = 99 ]; then say "$ax skipped: deadline"; return 99; fi
  if tail -n 3 runs/r052/$ax.log | grep -q "] done$"; then say "$ax rc=$rc complete"
  else say "$ax rc=$rc INCOMPLETE (resumable)"; return 1; fi
}
NDONE=0
for AX in camera_bench_yaw_deg camera_bench_scale camera_bench_pitch_deg; do
  axis_job $AX; r=$?
  [ $r = 0 ] && NDONE=$((NDONE + 1))
  [ $r = 99 ] && break                  # past the deadline: later axes cannot start either
  if [ -n "$UNTIL" ] && [ "$(date +%s)" -ge $((UNTIL - 300)) ]; then say "deadline reached"; break; fi
done
[ -f runs/r052/camera_bench_scale/dropped_scenes.json ] && say "scale axis dropped: $(cat runs/r052/camera_bench_scale/dropped_scenes.json | tr -d '\n' | cut -c1-300)"
if [ $NDONE = 3 ]; then say DONE; touch runs/r052/DONE; else say PARTIAL; touch runs/r052/PARTIAL; fi
