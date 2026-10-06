#!/usr/bin/env bash
# R-057: when is the episode decided -- drive-until sweep k in {1, 2, 4, all} on start pose and camera.
#   bash experiments/queue_r057.sh [--reuse-r053]
# --reuse-r053: take start-pose P and k=1 from runs/r053_{P,N}_s*, but only if their code_state matches
# this checkout on the code that shapes a rollout (vla_harness/, experiments/r039_run.py); otherwise they
# are rerun here as r057_ris_{P,k1}_s*. Per-job flock, rc logged, later jobs run on a failure, DONE marker.
# Announce to "my prim" before launching; do not launch without the user's go.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/libero-plus/bin/python; Q=runs/r057/queue.log
RIS=experiments/repro/r044_selection_ris.json; CAM=experiments/repro/r048_selection_cam.json
mkdir -p runs/r057; rm -f runs/r057/DONE runs/r057/BLOCKED
say() { echo "[$(date +%T)] $*" >> "$Q"; }
gpu() { local name="$1"; shift; say "$name start"; flock /tmp/vla_gpu.lock "$@"; local rc=$?; say "$name rc=$rc"; [ $rc = 0 ] || FAILED="$FAILED $name"; return $rc; }
FAILED=""
REUSE=0
if [ "${1:-}" = "--reuse-r053" ]; then
  if .venvs/groot/bin/python experiments/r057_score.py --check-reuse >> "$Q" 2>&1; then REUSE=1; say "reusing R-053 for start-pose P and k1"
  else say "R-053 code_state does not match (or runs missing): rerunning start-pose P and k1"; fi
fi
echo $REUSE > runs/r057/reuse_r053

# run <row> <selection> <label> <seed> <r039_run drive args...>
run() { local row=$1 sel=$2 lab=$3 ns=$4; shift 4
  gpu ${row}_${lab}_s$ns $PY experiments/r039_run.py --selection $sel --run-id r057_${row}_${lab}_s$ns --arms extended "$@" --noise-seed $ns --no-video >> runs/r057/${row}_${lab}_s$ns.log 2>&1; }

for NS in 0 1 2; do
  # start pose
  if [ $REUSE = 0 ]; then
    run ris $RIS P $NS --drive P
    run ris $RIS k1 $NS --drive N --drive-until 1
  fi
  run ris $RIS k2 $NS --drive N --drive-until 2
  run ris $RIS k4 $NS --drive N --drive-until 4
  run ris $RIS all $NS --drive N
  # camera
  run cam $CAM P $NS --drive P
  for K in 1 2 4; do run cam $CAM k$K $NS --drive N --drive-until $K; done
  run cam $CAM all $NS --drive N
done
.venvs/groot/bin/python experiments/r057_score.py >> "$Q" 2>&1 || say "SCORE FAILED"
if [ -z "$FAILED" ]; then say DONE; touch runs/r057/DONE; else say "FAILED:$FAILED"; echo "$FAILED" > runs/r057/BLOCKED; fi
