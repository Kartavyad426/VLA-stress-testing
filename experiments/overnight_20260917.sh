#!/usr/bin/env bash
# Overnight queue, 2026-09-17. Resumable: re-run this script to continue.
# Priority: validate the harness, then reproduce MINERVA (gate to GR00T).
#
#   0. provenance      code state (HEAD + uncommitted patch), versions, power
#   1. harness tests   camera perturbation + init-state pinning. If either FAILS,
#                      steps 2-3 are skipped (their results would be untrustworthy)
#                      but step 4 still runs -- it does not use our harness code.
#   2. harness parity  MINERVA through OUR harness, object+spatial, 10 eps/task.
#                      Compared later against step 4's first 10 eps/task
#                      (lerobot-eval sub-env i starts at init state i, same as ours).
#   3. camera sweep    MINERVA through our harness, object, fixed camera code.
#   4. MINERVA repro   authors' exact lerobot-eval command, 4 suites x 10 tasks
#                      x 50 eps, one task per invocation so it resumes per task.
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
LOG=experiments/repro/logs/overnight_20260917.log
OUT=experiments/repro/runs/overnight_20260917
mkdir -p "$OUT" experiments/repro/logs
say() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
power() { echo "ac=$(cat /sys/class/power_supply/AC/online) $(nvidia-smi --query-gpu=pstate,clocks.sm,power.draw --format=csv,noheader)"; }

# --- 0. provenance ------------------------------------------------------------
if [ ! -f "$OUT/code_state/HEAD" ]; then
  mkdir -p "$OUT/code_state"
  git rev-parse HEAD > "$OUT/code_state/HEAD"
  git diff > "$OUT/code_state/uncommitted.patch"
  git status --short > "$OUT/code_state/status.txt"
  git -C third_party/MINERVA rev-parse HEAD > "$OUT/code_state/minerva_repo_HEAD"
  for v in lerobot minerva; do
    uv pip freeze --python .venvs/$v/bin/python > "$OUT/code_state/freeze_$v.txt" 2>/dev/null
  done
  say "provenance saved (HEAD $(cat $OUT/code_state/HEAD | cut -c1-8), patch $(wc -l < $OUT/code_state/uncommitted.patch) lines)"
fi
say "=== session start  $(power) ==="

# --- 1. harness tests -----------------------------------------------------------
HARNESS_OK=1
for T in camera_perturbation_test init_state_pinning_test; do
  if .venvs/lerobot/bin/python experiments/$T.py > "$OUT/$T.log" 2>&1; then
    say "test $T: PASS"
  else
    say "test $T: FAIL -- skipping harness steps 2-3 (see $OUT/$T.log)"; HARNESS_OK=0
  fi
done
python3 experiments/phase_segmenter_test.py > "$OUT/phase_segmenter_test.log" 2>&1 \
  && say "test phase_segmenter_test: PASS" || say "test phase_segmenter_test: FAIL (mining only; harness runs continue)"

HE=(.venvs/minerva/bin/python experiments/harness_eval.py
    --checkpoint third_party/MINERVA/ckpt/t05_l1_0.54M --n-action-steps 1
    --override temporal_ensemble_coeff=0.01 --obs-size 360)

if [ "$HARNESS_OK" = 1 ]; then
  # --- 2. harness parity --------------------------------------------------------
  say "step 2 harness parity start  $(power)"
  "${HE[@]}" --suites libero_object,libero_spatial --episodes 10 \
    --run-id minerva_harness_parity >> "$OUT/step2_parity.log" 2>&1
  say "step 2 done rc=$?"

  # --- 3. camera sweep with fixed code --------------------------------------------
  say "step 3 camera sweep start  $(power)"
  "${HE[@]}" --suites libero_object --episodes 5 \
    --specs '[{}, {"camera_yaw_deg": 0}, {"camera_yaw_deg": 2}, {"camera_yaw_deg": 5}, {"camera_yaw_deg": 10}, {"camera_yaw_deg": 20}, {"camera_pitch_deg": 5}]' \
    --run-id minerva_harness_camera >> "$OUT/step3_camera.log" 2>&1
  say "step 3 done rc=$?"
fi

# --- 4. MINERVA reproduction, authors' command, per task ----------------------------
R=experiments/repro/runs/minerva_repro_full
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  for TID in 0 1 2 3 4 5 6 7 8 9; do
    TD="$R/$SUITE/task$TID"
    [ -f "$TD/eval_info.json" ] && continue
    rm -rf "$TD"; mkdir -p "$TD"
    say "step 4 $SUITE task$TID start  $(power)"
    ( cd third_party/MINERVA && ../../.venvs/minerva/bin/lerobot-eval \
        --policy.path=ckpt/t05_l1_0.54M --env.type=libero \
        --env.task=$SUITE --env.task_ids="[$TID]" \
        --policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1 \
        --eval.batch_size=5 --eval.n_episodes=50 --env.max_parallel_tasks=1 \
        --seed=1000 --output_dir="../../$TD" ) > "$TD/eval.log" 2>&1
    RC=$?
    SR=$(python3 -c "import json;d=json.load(open('$TD/eval_info.json'));print(d['overall']['pc_success'])" 2>/dev/null || echo "?")
    say "step 4 $SUITE task$TID rc=$RC success=$SR%"
  done
done
say "=== queue complete ==="
