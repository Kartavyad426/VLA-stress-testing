#!/usr/bin/env bash
# R-022: GR00T harness parity. Resumable: re-run to continue.
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
OUT=experiments/repro/runs/groot_parity_spatial
LOG=$OUT/progress.log
mkdir -p "$OUT"
say() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
power() { echo "ac=$(cat /sys/class/power_supply/AC/online) $(nvidia-smi --query-gpu=pstate,clocks.sm --format=csv,noheader)"; }
CK=nvidia/gr00t17-lerobot-libero_spatial-640
RENAME='{"observation.images.image2": "observation.images.wrist_image"}'

if [ ! -f "$OUT/code_state/HEAD" ]; then
  mkdir -p "$OUT/code_state"; git rev-parse HEAD > "$OUT/code_state/HEAD"
  git diff > "$OUT/code_state/uncommitted.patch"
  uv pip freeze --python .venvs/groot/bin/python > "$OUT/code_state/freeze_groot.txt" 2>/dev/null
fi
say "=== session start $(power) ==="

# --- reference: lerobot-eval (bf16 load), one task per invocation --------------
for TID in 0 1 2 3 4 5 6 7 8 9; do
  TD="$OUT/lerobot_eval/task$TID"
  [ -f "$TD/eval_info.json" ] && continue
  rm -rf "$TD"; mkdir -p "$TD"
  say "reference task$TID start $(power)"
  .venvs/groot/bin/python experiments/groot_eval_bf16.py \
    --policy.path=$CK --policy.base_model_path=nvidia/GR00T-N1.7-3B \
    --policy.embodiment_tag=libero_sim --policy.device=cuda \
    --rename_map="$RENAME" \
    --env.type=libero --env.task=libero_spatial --env.task_ids="[$TID]" \
    --eval.batch_size=1 --eval.n_episodes=10 --env.max_parallel_tasks=1 \
    --seed=1000 --output_dir="$TD" > "$TD/eval.log" 2>&1
  RC=$?
  SR=$(python3 -c "import json;print(json.load(open('$TD/eval_info.json'))['overall']['pc_success'])" 2>/dev/null || echo "?")
  say "reference task$TID rc=$RC success=$SR%"
done

# --- our harness --------------------------------------------------------------
say "harness start $(power)"
.venvs/groot/bin/python experiments/harness_eval.py --checkpoint $CK --n-action-steps 16 \
  --override base_model_path=nvidia/GR00T-N1.7-3B --override embodiment_tag=libero_sim \
  --dtype bfloat16 --rename-map "$RENAME" --obs-size 360 \
  --suites libero_spatial --episodes 10 --run-id groot_harness_parity >> "$OUT/harness.log" 2>&1
say "harness done rc=$?"
say "=== complete ==="
