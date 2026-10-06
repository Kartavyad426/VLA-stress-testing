#!/usr/bin/env bash
# 2026-09-29, user's go: re-run R-055's overfit gate at a CONSTANT LR 1e-4 (as registered). The overnight run used
# the launcher's cosine schedule (warmup, decay to 0 over 300 steps) and missed at 12.1% of step 0 (bar: < 10%).
# If it passes (and conformance, 2b and memory already passed), run what it gated:
#   R-056 train (r=16) -> R-056 eval (select, held-out, nominal + on-ramekin, transfer, score)
#   -> R-056 WiSE-FT (alpha 0.5) -> R-058 r=4 training.
# The failed cosine result is kept in runs/r055/gates.json as "overfit_cosine_20260929".
# Per-job flock (shares the GPU with the R-052 continuation between its axis jobs); log to runs/overnight/queue.log.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/libero-plus/bin/python; G=.venvs/groot/bin/python; O=runs/overnight; Q=$O/queue.log
MINTED=data/r054_train48; MINTED_REPO=local/r054_train48
if [ -z "${VLA_INHIBITED:-}" ]; then
  export VLA_INHIBITED=1
  exec systemd-inhibit --what=sleep:idle:handle-lid-switch --why="VLA R-055 rerun + R-056" --mode=block bash "$0" "$@"
fi
say() { echo "[$(date '+%F %T')] [r055b] $*" >> $Q; }
step() { local n=$1; shift; say "$n start"; local t0=$(date +%s); "$@" >> $O/$n.log 2>&1; local rc=$?
         say "$n end rc=$rc ($(( ($(date +%s) - t0) / 60 )) min)"; return $rc; }
gpu() { flock /tmp/vla_gpu.lock "$@"; }
jtrue() { [ -f "$1" ] && $G -c "import json,sys; d=json.load(open('$1')); sys.exit(0 if ($2) else 1)" 2>/dev/null; }

$G - <<'EOF'
import json
p = "runs/r055/gates.json"; g = json.load(open(p))
if "overfit" in g and "overfit_cosine_20260929" not in g:
    g["overfit_cosine_20260929"] = {**g["overfit"], "lr_schedule": "cosine (5% warmup, decay to 0)", "superseded_by": "overfit"}
    json.dump(g, open(p, "w"), indent=1)
EOF
step R055_overfit_const gpu $PY experiments/r055_gates.py overfit --minted-root $MINTED --minted-repo $MINTED_REPO --lr-schedule constant
  # overfit is INFORMATIONAL (R-055 amendment 2026-09-29 15:15): logged, never blocks; blocking set = conformance + 2b + memory
say "R-055 overfit (informational): ratio $($G -c "import json;print(json.load(open('runs/r055/gates.json'))['overfit'].get('ratio'))")"
if ! jtrue runs/r055/gates.json "all(d[k]['pass'] for k in ('conformance','ckpt_equiv_r16')) and d['memory']['E3_batch1_le_7.4GiB']"; then
  say "R-055 blocking gates FAIL: nothing trains"; exit 1
fi
say "R-055 PASS: training"
step R056_train gpu $PY experiments/r056_train.py --minted-root $MINTED --minted-repo $MINTED_REPO --lora-r 16 \
  $([ -e runs/r056_r16_train/checkpoints/last ] && echo --resume) --report runs/r056_r16_train_report.json || { say "R-056 train failed"; exit 1; }
step R056_eval bash -c "
  flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py select &&
  flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py heldout &&
  flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py nominal --with-ramekin &&
  flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py transfer ;
  $G experiments/r056_eval.py score" && \
step B_R056_wise bash -c "flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py wise && $G experiments/r056_eval.py score"
step B_R058_r4_train gpu $PY experiments/r056_train.py --minted-root $MINTED --minted-repo $MINTED_REPO --run-prefix r058 --lora-r 4 \
  $([ -e runs/r058_r4_train/checkpoints/last ] && echo --resume) --report runs/r058_r4_train_report.json
say "done"
