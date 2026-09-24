#!/usr/bin/env bash
# R-044 check 3 only (forward-0 rescue), rerun after the --drive choices fix.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/libero-plus/bin/python; Q=runs/r044/queue.log; SEL=experiments/repro/r044_selection_ris.json
say() { echo "[$(date +%T)] $*" >> "$Q"; }
gpu() { local name="$1"; shift; say "$name start"; flock /tmp/vla_gpu.lock "$@"; local rc=$?; say "$name rc=$rc"; return $rc; }
rm -rf runs/r044_smoke
gpu smoke2 $PY experiments/r039_run.py --selection $SEL --run-id r044_smoke --arms extended --drive W --drive-until 1 --limit 1 --no-video > runs/r044_smoke.log 2>&1
if grep -q '"drive": "W"' runs/r044_smoke/splice/manifest.jsonl 2>/dev/null; then
  for D in W A N; do
    gpu rescue_$D $PY experiments/r039_run.py --selection $SEL --run-id r044_rescue_$D --arms extended --drive $D --drive-until 1 --noise-seed 0 --no-video > runs/r044_rescue_$D.log 2>&1
  done
  for R in r044_rescue_W r044_rescue_A r044_rescue_N; do .venvs/groot/bin/python experiments/r039_report.py runs/$R --out /dev/null >> "$Q" 2>&1; done
else
  say "SMOKE2 FAILED"; echo smoke2 > runs/r044/BLOCKED
fi
say "RESCUE DONE"; touch runs/r044/RESCUE_DONE
