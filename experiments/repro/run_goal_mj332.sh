#!/usr/bin/env bash
# Goal suite under MuJoCo 3.3.2 (PENDING_DECISIONS #21 + #3, user-approved 2026-09-16).
#   A) lerobot-eval, ONE variable vs res256_nas10_seed1000 goal (68%): MuJoCo 3.3.7 -> 3.3.2.
#      20 eps/task: the first 10 per task share init states with the 3.3.7 baseline
#      (paired), the extra 10 buy precision -- nas=1 showed 32/100 discordant episodes.
#   B) harness e2e goal re-run (#3), for traces with fixture/region bodies.
set -uo pipefail
cd "$(dirname "$0")/../.."
export MUJOCO_GL=egl
PY=.venvs/lerobot-mj332
OUT=experiments/repro/runs/res256_nas10_seed1000_mj332_goal20
mkdir -p "$OUT"
{
  echo "=== session $(date -Is) ==="
  echo "purpose:     goal suite at MuJoCo 3.3.2; vs res256_nas10_seed1000 goal=68% (3.3.7)"
  echo "delta:       mujoco 3.3.7 -> 3.3.2 ONLY (venv freeze diff verified: one line)"
  echo "held:        256x256, nas=10, relative, hard_reset, init_states, seed 1000"
  echo "episodes/task: 20 (first 10 paired with baseline init states)"
  echo "venv:        $PY"
  echo "lerobot_sha: $(git -C third_party/lerobot rev-parse HEAD)"
  echo "gpu_power:   $(nvidia-smi --query-gpu=enforced.power.limit,power.draw,clocks.sm,pstate --format=csv,noheader)"
  echo "ac_online:   $(cat /sys/class/power_supply/AC/online)"
  $PY/bin/python -c "
import importlib.metadata as m
for p in ['lerobot','torch','mujoco','robosuite','transformers','numpy']: print(f'{p}: {m.version(p)}')"
} >> "$OUT/provenance.txt" 2>/dev/null
echo "--- session start ---" >> "$OUT/provenance.txt"
# RESUMABLE: lerobot-eval has no resume, so run ONE TASK PER INVOCATION into its
# own directory and skip any task whose eval_info.json already exists. A task
# interrupted mid-way has no eval_info.json and simply re-runs from its start.
# Stop anytime (Ctrl-C / kill); re-run this same script to continue.
T0=$(date +%s)
for TID in 0 1 2 3 4 5 6 7 8 9; do
  TD="$OUT/task$TID"
  if [ -f "$TD/eval_info.json" ]; then echo "task $TID: done, skipping"; continue; fi
  rm -rf "$TD"; mkdir -p "$TD"
  echo "task $TID: start $(date +%T)  ac=$(cat /sys/class/power_supply/AC/online)  $(nvidia-smi --query-gpu=pstate,clocks.sm --format=csv,noheader)" | tee -a "$OUT/progress.log"
  $PY/bin/lerobot-eval \
    --policy.path=HuggingFaceVLA/smolvla_libero --policy.n_action_steps=10 \
    --env.type=libero --env.task=libero_goal --env.task_ids="[$TID]" \
    --env.observation_width=256 --env.observation_height=256 \
    --env.control_mode=relative --env.init_states=true \
    --eval.batch_size=1 --eval.n_episodes=20 --env.max_parallel_tasks=1 \
    --seed=1000 --output_dir="$TD" > "$TD/eval.log" 2>&1
  RC=$?
  echo "task $TID: rc=$RC end $(date +%T)" | tee -a "$OUT/progress.log"
  [ $RC -eq 0 ] || { echo "task $TID failed; stopping so it can be inspected"; exit $RC; }
done
echo "A wall_seconds_this_session: $(( $(date +%s) - T0 ))" >> "$OUT/provenance.txt"
echo "=== A done ==="

T1=$(date +%s)
$PY/bin/python experiments/e2e.py --suite libero_goal --tasks 8 --seeds 5 \
  --yaw-levels 0,5,10,15,20 --run-id goal_rerun_mj332 > experiments/repro/logs/goal_rerun_mj332.log 2>&1
echo "B rc=$? wall=$(( $(date +%s) - T1 ))s" | tee -a experiments/repro/logs/goal_rerun_mj332.log
echo "=== B done ==="
