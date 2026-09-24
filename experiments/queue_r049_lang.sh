#!/usr/bin/env bash
# R-049 language row rerun (nominal text fixed), behind R-050.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/libero-plus/bin/python; Q=runs/r049/queue.log
say() { echo "[$(date +%T)] $*" >> "$Q"; }
say "lang rerun: waiting for R-050"; until [ -f runs/r050/DONE ]; do sleep 60; done
say "lang start"; flock /tmp/vla_gpu.lock $PY experiments/r039_run.py --selection experiments/repro/r049_selection_lang.json --run-id r049_lang --arms base --no-video > runs/r049/lang.log 2>&1; say "lang rc=$?"
.venvs/groot/bin/python experiments/r039_report.py runs/r049_lang --out /dev/null >> "$Q" 2>&1
say "LANG DONE"; touch runs/r049/LANG_DONE
