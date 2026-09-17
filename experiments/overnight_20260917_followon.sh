#!/usr/bin/env bash
# Follow-on queue: starts only after overnight_20260917.sh has exited. Resumable.
#   5. SmolVLA harness parity  -- our adapter vs lerobot-eval, the adapter every
#      existing campaign used. Reference: res256_nas10_seed1000 libero_spatial
#      (73%, lerobot-eval batch 1 => episode i on init state i, same as ours).
#   6. goal suite, MuJoCo 3.3.2 (#21) and goal pipeline re-run (#3):
#      experiments/repro/run_goal_mj332.sh, itself resumable per task.
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
LOG=experiments/repro/logs/overnight_20260917.log
OUT=experiments/repro/runs/overnight_20260917
say() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
power() { echo "ac=$(cat /sys/class/power_supply/AC/online) $(nvidia-smi --query-gpu=pstate,clocks.sm,power.draw --format=csv,noheader)"; }

while pgrep -f "experiments/overnight_20260917.sh" > /dev/null; do sleep 30; done
say "=== follow-on queue start  $(power) ==="

if grep -q "test init_state_pinning_test: PASS" "$LOG" && grep -q "test camera_perturbation_test: PASS" "$LOG"; then
  say "step 5 SmolVLA harness parity start  $(power)"
  .venvs/lerobot/bin/python -m vla_harness.conformance --checkpoint HuggingFaceVLA/smolvla_libero \
      --obs-size 256 --n-action-steps 10 --control-mode relative --fps 20 --max-parallel-tasks 1 \
      --mujoco 3.3.7 > "$OUT/step5_gate.log" 2>&1
  .venvs/lerobot/bin/python experiments/harness_eval.py \
      --checkpoint HuggingFaceVLA/smolvla_libero --n-action-steps 10 --obs-size 256 \
      --suites libero_spatial --episodes 10 --run-id smolvla_harness_parity \
      >> "$OUT/step5_smolvla_parity.log" 2>&1
  say "step 5 done rc=$?"
else
  say "step 5 skipped: harness self-tests did not pass"
fi

say "step 6 goal suite (mj332) start  $(power)"
./experiments/repro/run_goal_mj332.sh >> "$OUT/step6_goal.log" 2>&1
say "step 6 done rc=$?"
say "=== follow-on queue complete ==="
