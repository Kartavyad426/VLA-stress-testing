#!/usr/bin/env bash
# R-053: the start-pose no-op baseline. 10 R-044 instances x noise seeds 0,1,2 x drives P, N at forward 0.
# Per-job flock, rc logged, later jobs run even if one fails, scored at the end, DONE marker.
# Announce to "my prim" before launching; do not launch without the user's go.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/libero-plus/bin/python; Q=runs/r053/queue.log; SEL=experiments/repro/r044_selection_ris.json
mkdir -p runs/r053; rm -f runs/r053/DONE runs/r053/BLOCKED
say() { echo "[$(date +%T)] $*" >> "$Q"; }
gpu() { local name="$1"; shift; say "$name start"; flock /tmp/vla_gpu.lock "$@"; local rc=$?; say "$name rc=$rc"; [ $rc = 0 ] || FAILED="$FAILED $name"; return $rc; }
FAILED=""
for NS in 0 1 2; do
  for D in P N; do
    gpu ${D}_s$NS $PY experiments/r039_run.py --selection $SEL --run-id r053_${D}_s$NS --arms extended --drive $D --drive-until 1 --noise-seed $NS --no-video >> runs/r053/${D}_s$NS.log 2>&1
  done
done
.venvs/groot/bin/python experiments/r053_score.py >> "$Q" 2>&1 || say "SCORE FAILED"
if [ -z "$FAILED" ]; then say DONE; touch runs/r053/DONE; else say "FAILED:$FAILED"; echo "$FAILED" > runs/r053/BLOCKED; fi
