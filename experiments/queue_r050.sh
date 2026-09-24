#!/usr/bin/env bash
# R-050: rescue at forward 0 with every arm recorded downstream. Waits for R-049. Per-job flock, rc logged, DONE marker.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/libero-plus/bin/python; Q=runs/r050/queue.log; mkdir -p runs/r050
say() { echo "[$(date +%T)] $*" >> "$Q"; }
gpu() { local name="$1"; shift; say "$name start"; flock /tmp/vla_gpu.lock "$@"; local rc=$?; say "$name rc=$rc"; return $rc; }
say "waiting for R-049"; until [ -f runs/r049/DONE ] || [ -f runs/r049/BLOCKED ]; do sleep 60; done
for D in N W; do gpu ris_$D $PY experiments/r039_run.py --selection experiments/repro/r044_selection_ris.json --run-id r050_ris_$D --arms extended --drive $D --drive-until 1 --noise-seed 0 --no-video > runs/r050/ris_$D.log 2>&1; done
for D in N A; do gpu cam_$D $PY experiments/r039_run.py --selection experiments/repro/r048_selection_cam.json --run-id r050_cam_$D --arms extended --drive $D --drive-until 1 --noise-seed 0 --no-video > runs/r050/cam_$D.log 2>&1; done
say DONE; touch runs/r050/DONE
