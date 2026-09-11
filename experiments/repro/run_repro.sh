#!/usr/bin/env bash
# Reproduce SmolVLA's published LIBERO numbers with lerobot-eval as shipped.
#
# Deliberately NOT using vla_harness: if the number comes out wrong we must be
# able to tell "the checkpoint doesn't reproduce" from "our adapter is buggy".
# Our adapters get wired afterwards and must land on the same number.
#
# Usage:  ./run_repro.sh <n_action_steps> <seed> [suite,suite,...]
set -euo pipefail
cd "$(dirname "$0")/../.."

NAS="${1:-1}"                 # 1 = the checkpoint's shipped config
SEED="${2:-1000}"
SUITES="${3:-libero_spatial,libero_object,libero_goal,libero_10}"
OUT="experiments/repro/runs/nas${NAS}_seed${SEED}"

export MUJOCO_GL=egl
mkdir -p "$OUT"

# Provenance first — a result without it is not reproducible (PLAN.md §7.3).
{
  echo "date:        $(date -Is)"
  echo "checkpoint:  HuggingFaceVLA/smolvla_libero"
  echo "n_action_steps: $NAS"
  echo "seed:        $SEED"
  echo "suites:      $SUITES"
  echo "episodes/task: 10"
  echo "lerobot_sha: $(git -C third_party/lerobot rev-parse HEAD)"
  echo "gpu:         $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
  # Power state is part of the result, not trivia. A wall-clock number taken at
  # a 15 W battery cap and one taken at 35 W differ by ~25x on identical work
  # (independently confirmed by session `solver`: 601.7 s vs 24.0 s on the same
  # input). nvidia-smi's utilization.gpu does NOT reveal this -- it read 100%
  # throughout the throttled window. pstate and clocks.sm do.
  echo "gpu_power:   $(nvidia-smi --query-gpu=enforced.power.limit,power.draw,clocks.sm,pstate --format=csv,noheader)"
  echo "ac_online:   $(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)"
  echo "power_profile: $(powerprofilesctl get 2>/dev/null || echo unknown)"
  echo "gpu_other_procs: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')"
  .venvs/lerobot/bin/python - <<'PY'
import importlib.metadata as m
for p in ["lerobot","torch","mujoco","robosuite","hf-libero","transformers","numpy"]:
    try: print(f"{p}: {m.version(p)}")
    except Exception: print(f"{p}: MISSING")
PY
} > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"

echo "=== running: $SUITES | n_action_steps=$NAS | seed=$SEED ==="
START=$(date +%s)
.venvs/lerobot/bin/lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --policy.n_action_steps="$NAS" \
  --env.type=libero \
  --env.task="$SUITES" \
  --env.init_states=true \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1 \
  --seed="$SEED" \
  --output_dir="$OUT" 2>&1 | tee "$OUT/eval.log"
echo "wall_seconds: $(( $(date +%s) - START ))" >> "$OUT/provenance.txt"
echo "=== done -> $OUT ==="
