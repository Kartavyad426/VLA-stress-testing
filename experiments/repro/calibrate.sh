#!/usr/bin/env bash
# Time ONE episode at each n_action_steps. Fast configs first, so we learn
# something even if the slow one has to be abandoned.
#
# Clocks are sampled per-episode, not once per session: `solver` measured
# 1380 MHz and 2610 MHz minutes apart on the same 35 W limit, so "on AC" is
# not a single operating point and a per-session clock reading would let us
# compare across clock states while believing we compare across configs.
set -uo pipefail
cd "$(dirname "$0")/../.."
export MUJOCO_GL=egl
OUT=experiments/repro/calib; mkdir -p "$OUT"
RES="$OUT/results.tsv"
[ -f "$RES" ] || printf "nas\twall_s\tclk_start\tclk_end\tpstate\tpower_lim\tsuccess\n" > "$RES"

for NAS in 50 10 1; do
  echo "=== n_action_steps=$NAS ==="
  CLK0=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits)
  T0=$(date +%s)
  timeout 2400 .venvs/lerobot/bin/lerobot-eval \
    --policy.path=HuggingFaceVLA/smolvla_libero \
    --policy.n_action_steps="$NAS" \
    --env.type=libero --env.task=libero_spatial --env.task_ids="[0]" \
    --env.init_states=true \
    --eval.batch_size=1 --eval.n_episodes=1 --env.max_parallel_tasks=1 \
    --seed=1000 --output_dir="$OUT/nas$NAS" > "$OUT/nas$NAS.log" 2>&1
  RC=$?
  W=$(( $(date +%s) - T0 ))
  CLK1=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits)
  PS=$(nvidia-smi --query-gpu=pstate --format=csv,noheader)
  PL=$(nvidia-smi --query-gpu=enforced.power.limit --format=csv,noheader,nounits)
  SR=$(grep -o 'running_success_rate=[0-9.]*%' "$OUT/nas$NAS.log" | tail -1)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s(rc=%s)\n" "$NAS" "$W" "$CLK0" "$CLK1" "$PS" "$PL" "${SR:-?}" "$RC" >> "$RES"
  echo "  wall=${W}s clocks ${CLK0}->${CLK1} MHz  ${SR:-?}  rc=$RC"
done
echo "=== results ==="; column -t "$RES"
