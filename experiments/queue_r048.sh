#!/usr/bin/env bash
# R-048: camera-viewpoint robustness checks. Per-job flock, rc logged, DONE marker.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/libero-plus/bin/python; Q=runs/r048/queue.log; SEL=experiments/repro/r048_selection_cam.json; mkdir -p runs/r048
say() { echo "[$(date +%T)] $*" >> "$Q"; }
gpu() { local name="$1"; shift; say "$name start"; flock /tmp/vla_gpu.lock "$@"; local rc=$?; say "$name rc=$rc"; return $rc; }
gpu reverse $PY experiments/r048_reverse.py --selection $SEL --out runs/r048_reverse --noise-seed 0 > runs/r048_reverse.log 2>&1
for NS in 1 2; do
  gpu seed$NS $PY experiments/r039_run.py --selection $SEL --run-id r048_seed$NS --arms extended --noise-seed $NS --no-video > runs/r048_seed$NS.log 2>&1
done
for D in A W N; do
  gpu rescue_$D $PY experiments/r039_run.py --selection $SEL --run-id r048_rescue_$D --arms extended --drive $D --drive-until 1 --noise-seed 0 --no-video > runs/r048_rescue_$D.log 2>&1
done
for R in r048_seed1 r048_seed2 r048_rescue_A r048_rescue_W r048_rescue_N; do
  [ -f runs/$R/splice/manifest.jsonl ] && .venvs/groot/bin/python experiments/r039_report.py runs/$R --out /dev/null >> "$Q" 2>&1
done
say DONE; touch runs/r048/DONE
