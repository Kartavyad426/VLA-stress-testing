#!/usr/bin/env bash
# R-047: smoke, then start-pose axis, then camera yaw at the benchmark range. Per-job flock, rc logged, DONE marker.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/libero-plus/bin/python; Q=runs/r047/queue.log; mkdir -p runs/r047
say() { echo "[$(date +%T)] $*" >> "$Q"; }
gpu() { local name="$1"; shift; say "$name start"; flock /tmp/vla_gpu.lock "$@"; local rc=$?; say "$name rc=$rc"; return $rc; }
rm -rf runs/r047_smoke; rm -f runs/r047/DONE runs/r047/BLOCKED
gpu smoke $PY experiments/r047_run.py --run-id r047_smoke --axis joint_radius_rad --scenes 984 --max-rollouts 2 > runs/r047_smoke.log 2>&1
if [ "$(wc -l < runs/r047_smoke/joint_radius_rad/manifest.jsonl 2>/dev/null || echo 0)" -lt 2 ]; then say "SMOKE FAILED"; echo smoke > runs/r047/BLOCKED; exit 1; fi
FAILED=0
for AXIS in joint_radius_rad camera_yaw_deg; do
  gpu $AXIS $PY experiments/r047_run.py --run-id r047 --axis $AXIS >> runs/r047/$AXIS.log 2>&1 || FAILED=1   # append: the runner resumes from the manifest
done
if [ $FAILED = 0 ]; then say DONE; touch runs/r047/DONE; else say "FAILED (see axis logs)"; echo axis > runs/r047/BLOCKED; exit 1; fi
