#!/usr/bin/env bash
# R-041, all four axes in sequence, each under its own flock acquisition.
# Waits for the R-042 queue (pid in $1, optional) to exit first so the
# announced order holds.
set -u
cd "$(dirname "$0")/.."
if [ "${1:-}" != "" ]; then while kill -0 "$1" 2>/dev/null; do sleep 30; done; fi
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p runs/r041
for AXIS in camera_yaw_deg camera_dist_m light_intensity joint_radius_rad; do
  echo "[$(date +%T)] $AXIS start" >> runs/r041/queue.log
  flock /tmp/vla_gpu.lock .venvs/libero-plus/bin/python experiments/r041_run.py --run-id r041 --axis "$AXIS" > "runs/r041/$AXIS.log" 2>&1
  echo "[$(date +%T)] $AXIS rc=$?" >> runs/r041/queue.log
done
