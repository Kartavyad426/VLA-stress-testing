#!/usr/bin/env bash
# Overnight queue 2026-09-18 — FAILURE HUNT. Resumable: re-run to continue.
#
# Goal for this night is volume of FAILURE traces, not a balanced design: the
# taxonomy (family rules in vla_harness/mining/classify.py) is to be adjudicated
# by hand tomorrow, and it needs many real failures across many perturbation
# kinds. So we run LIBERO-Plus levels 5 then 4 (the variants four reference
# models mostly failed), all 7 perturbation types, interleaved by type so any
# prefix of the run is type-balanced.
#
#   1. contamination A/B  (R-025)  41 hard variants, LeRobot's contaminated text
#   2. groot_fail         (R-026)  623 L5+L4 variants, GR00T, until DEADLINE_GROOT
#   3. minerva_fail       (R-027)  same ids, MINERVA, until DEADLINE_MINERVA
#                                  (skipped unless .venvs/minerva-lplus smoke-tests)
#   4. CPU: mine every run (families + predicate fire rates), summarise,
#      render HTML pages for failures.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl
OUT=experiments/repro/runs/overnight_20260918; mkdir -p "$OUT"
LOG=$OUT/progress.log
say() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
power() { echo "ac=$(cat /sys/class/power_supply/AC/online) $(nvidia-smi --query-gpu=pstate,clocks.sm --format=csv,noheader)"; }
# wall-clock deadlines, so morning always has mined + summarised results
DEADLINE_GROOT=$(date -d "05:40" +%s)
DEADLINE_MINERVA=$(date -d "07:50" +%s)
left() { local d=$1; local n=$(date +%s); echo $(( d > n ? d - n : 0 )); }

GROOT=(--checkpoint nvidia/gr00t17-lerobot-libero_spatial-640 --n-action-steps 16
       --override base_model_path=nvidia/GR00T-N1.7-3B --override embodiment_tag=libero_sim
       --dtype bfloat16 --rename-map '{"observation.images.image2": "observation.images.wrist_image"}' --obs-size 360)
MINERVA=(--checkpoint third_party/MINERVA/ckpt/t05_l1_0.54M --n-action-steps 1
         --override temporal_ensemble_coeff=0.01 --obs-size 360)
FAIL_IDS=$(cat experiments/repro/lplus_fail_ids.txt)
HARD=$(cat experiments/repro/lplus_hard_ids.txt)
# MINERVA is not language-conditioned: it looks the instruction up in a fixed
# 40-task table (TinyflowTaskToIndexStep) and raises KeyError on anything else,
# so LIBERO-Plus language rewrites CANNOT be given to it. Its arm drops them.
FAIL_IDS_NOLANG=$(cat experiments/repro/lplus_fail_ids_nolang.txt)

if [ ! -f "$OUT/code_state/HEAD" ]; then
  mkdir -p "$OUT/code_state"; git rev-parse HEAD > "$OUT/code_state/HEAD"; git diff > "$OUT/code_state/uncommitted.patch"
  git -C third_party/LIBERO-plus diff > "$OUT/code_state/libero_plus_local.patch"
fi
cp experiments/repro/lplus_fail_selection.json "$OUT/fail_selection.json"
say "=== failure-hunt queue start $(power) ==="

run () {  # name, venv, run-id, seconds-budget, extra args...
  local NAME=$1 VENV=$2 RID=$3 BUDGET=$4; shift 4
  if [ "$BUDGET" -lt 120 ]; then say "$NAME SKIPPED (no time budget left)"; return; fi
  say "$NAME start budget=${BUDGET}s $(power)"
  timeout --foreground "$BUDGET" \
  .venvs/$VENV/bin/python experiments/harness_eval.py --libero-plus \
    --suites libero_spatial --episodes 1 --run-id "$RID" "$@" >> "$OUT/$NAME.log" 2>&1
  say "$NAME done rc=$? rollouts=$(wc -l < runs/$RID/rollouts.jsonl 2>/dev/null || echo 0) $(power)"
}

# --- 1. contamination A/B (R-025) -------------------------------------------
run contamination_ab libero-plus lplus_hard_groot_rawinstr 2400 "${GROOT[@]}" --raw-instruction --tasks "$HARD"

# --- 2. GR00T failure hunt (R-026) ------------------------------------------
run groot_fail libero-plus lplus_fail_groot "$(left $DEADLINE_GROOT)" "${GROOT[@]}" --tasks "$FAIL_IDS"

# --- 3. MINERVA on the same variants (R-027) --------------------------------
if .venvs/minerva-lplus/bin/python -c "
import lerobot, libero, robosuite, mujoco
from lerobot.envs.configs import LiberoPlusEnv
print('minerva-lplus ok', mujoco.__version__)" > "$OUT/minerva_lplus_import.log" 2>&1; then
  say "minerva-lplus import OK"
  run minerva_fail minerva-lplus lplus_fail_minerva "$(left $DEADLINE_MINERVA)" "${MINERVA[@]}" --tasks "$FAIL_IDS_NOLANG"
else
  say "minerva_fail SKIPPED: .venvs/minerva-lplus does not import (see $OUT/minerva_lplus_import.log)"
fi

# --- 4. mine, summarise, render ---------------------------------------------
say "mining"
RUNS=(runs/lplus_fail_groot runs/lplus_fail_minerva runs/lplus_hard_groot_v2 runs/lplus_hard_groot_rawinstr)
EXIST=(); for r in "${RUNS[@]}"; do [ -f "$r/rollouts.jsonl" ] && EXIST+=("$r"); done
python3 experiments/mine_run.py "${EXIST[@]}" > "$OUT/mining.txt" 2>&1
say "summarising"
python3 experiments/summarize_lplus.py "${EXIST[@]}" > "$OUT/summary.txt" 2>&1
say "rendering failure pages"
timeout --foreground 3000 ./experiments/render_lplus_failures.sh runs/lplus_fail_groot viz/lplus_fail_groot 80 > "$OUT/render.log" 2>&1
say "=== complete ===  mining: $OUT/mining.txt  summary: $OUT/summary.txt"
