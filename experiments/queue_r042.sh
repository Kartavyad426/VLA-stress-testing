#!/usr/bin/env bash
# R-042 smoke then full run, each under its own flock acquisition.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p runs/r042_smoke runs/r042
rm -rf runs/r042_smoke
flock /tmp/vla_gpu.lock .venvs/libero-plus/bin/python experiments/r039_run.py --selection experiments/repro/r039_selection.json --run-id r042_smoke --arms extended --limit 1 --no-video > runs/r042_smoke.log 2>&1
if [ -s runs/r042_smoke/splice/manifest.jsonl ] && grep -q '"arms": \["P", "N", "T", "I", "S", "IT", "A", "W"\]' runs/r042_smoke/splice/manifest.jsonl; then
  flock /tmp/vla_gpu.lock .venvs/libero-plus/bin/python experiments/r039_run.py --selection experiments/repro/r039_selection.json --run-id r042 --arms extended --no-video > runs/r042/run.log 2>&1
else
  echo "SMOKE FAILED: not launching the full run" > runs/r042/BLOCKED
fi
