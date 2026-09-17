#!/usr/bin/env bash
# R-023: GR00T on basic (difficulty-1) LIBERO-Plus perturbations, libero_spatial.
# One lerobot-eval invocation per perturbation type (resumable per type).
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config
OUT=experiments/repro/runs/lplus_basic_groot_spatial
mkdir -p "$OUT"; LOG=$OUT/progress.log
say() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
if [ ! -f "$OUT/code_state/HEAD" ]; then
  mkdir -p "$OUT/code_state"; git rev-parse HEAD > "$OUT/code_state/HEAD"; git diff > "$OUT/code_state/uncommitted.patch"
  git -C third_party/LIBERO-plus rev-parse HEAD > "$OUT/code_state/libero_plus_HEAD"
  uv pip freeze --python .venvs/libero-plus/bin/python > "$OUT/code_state/freeze_libero_plus.txt" 2>/dev/null
  cp experiments/repro/lplus_basic_selection.json "$OUT/selection.json"
fi
python3 -c "import json;[print(k.replace(' ','_'), ','.join(map(str,v))) for k,v in json.load(open('$OUT/selection.json')).items()]" | while read CAT IDS; do
  TD="$OUT/$CAT"; [ -f "$TD/eval_info.json" ] && continue
  rm -rf "$TD"; mkdir -p "$TD"
  say "$CAT start ids=[$IDS] ac=$(cat /sys/class/power_supply/AC/online) $(nvidia-smi --query-gpu=pstate,clocks.sm --format=csv,noheader)"
  .venvs/libero-plus/bin/python experiments/groot_eval_bf16.py \
    --policy.path=nvidia/gr00t17-lerobot-libero_spatial-640 --policy.base_model_path=nvidia/GR00T-N1.7-3B \
    --policy.embodiment_tag=libero_sim --policy.device=cuda \
    --rename_map='{"observation.images.image2": "observation.images.wrist_image"}' \
    --env.type=libero_plus --env.task=libero_spatial --env.task_ids="[$IDS]" \
    --eval.batch_size=1 --eval.n_episodes=1 --env.max_parallel_tasks=1 \
    --seed=1000 --output_dir="$TD" > "$TD/eval.log" 2>&1
  RC=$?
  SR=$(python3 -c "import json;d=json.load(open('$TD/eval_info.json'));print(sum(sum(t['metrics']['successes']) for t in d['per_task']),'/',len(d['per_task']))" 2>/dev/null || echo "?")
  say "$CAT rc=$RC success=$SR"
done
say "=== complete ==="
