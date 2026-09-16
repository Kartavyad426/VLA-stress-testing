#!/usr/bin/env bash
# 10-HOUR CAMPAIGN -- end-to-end pipeline on real LIBERO, through OUR harness.
#
# Goal is to exercise the FULL pipeline across DIVERSE tasks, not to produce a
# client-grade robustness claim -- SmolVLA sits ~20 pp below its published
# numbers (F8/F9), so the failures we mine are partly setup artifacts. This
# tests the machinery; interpretation waits on the reproduction question.
#
# Per suite: control -> reproducibility floor -> camera-yaw sweep (uniform arm)
#            -> ddmin attribution -> mining -> clusters -> manifest -> regset
#
# Sizing (~20 s/episode measured):
#   control  8 tasks x 5 seeds                =  40
#   floor    2 repeats x 5 seeds              =  10
#   sweep    8 tasks x 5 levels x 5 seeds     = 200
#   ddmin    ~4 specs x 5 seeds               =  20
#   -> ~270 episodes/suite, ~1.5 h; x4 suites ~= 6 h
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
STAMP=$(date +%Y%m%d-%H%M)
LOG=experiments/repro/logs/campaign_$STAMP.log

{
  echo "CAMPAIGN $STAMP"
  echo "policy:  HuggingFaceVLA/smolvla_libero  n_action_steps=10"
  echo "render:  256x256 (F9 -- LiberoEnv class default; NOT LiberoEnvConfig's 360)"
  echo "mujoco:  $(.venvs/lerobot/bin/python -c 'import mujoco;print(mujoco.__version__)')"
  echo "ac:      $(cat /sys/class/power_supply/A*/online 2>/dev/null|head -1)"
  echo "gpu:     $(nvidia-smi --query-gpu=name,enforced.power.limit --format=csv,noheader)"
  echo
} | tee "$LOG"

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  echo "################ $SUITE ################" | tee -a "$LOG"
  timeout 9000 .venvs/lerobot/bin/python experiments/e2e.py \
    --suite "$SUITE" --tasks 8 --seeds 5 \
    --yaw-levels 0,5,10,15,20 \
    --run-id "camp_${STAMP}_${SUITE}" 2>&1 \
    | grep -vE "EGL|eglMake|err =|baseOp|cArgu|robosuite|WARNING|OpenGL|Loading weights|torch_dtype|HF_TOKEN" \
    | tee -a "$LOG"
  echo "[$SUITE done $(date +%H:%M)]" | tee -a "$LOG"
done

# Phase G -- language sensitivity. Runs LAST because it gates a taxonomy family
# rather than feeding the sweep, and because a partial result is still useful.
echo "################ LANGUAGE PROBE ################" | tee -a "$LOG"
timeout 4000 .venvs/lerobot/bin/python experiments/language_probe_run.py 2>&1 \
  | grep -vE "EGL|eglMake|robosuite|WARNING|OpenGL|Loading weights" | tee -a "$LOG"

echo "CAMPAIGN COMPLETE $(date +%H:%M)" | tee -a "$LOG"
