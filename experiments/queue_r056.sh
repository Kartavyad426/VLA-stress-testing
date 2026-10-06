#!/usr/bin/env bash
# 2026-09-29 15:15, user's go: R-056 train -> R-056 eval -> WiSE-FT (alpha 0.5) -> R-058 r=4 training -> R-052 resume.
# No deadline. Every GPU job takes its own flock; the whole run holds a systemd-inhibit lock. Resume-safe:
# a trained run (a checkpoint at opt_steps_total) is skipped, otherwise --resume; eval phases skip finished rollouts;
# R-052 resumes per axis from its manifests.
# Gating (R-055 amendment 2026-09-29 15:15): training needs conformance + ckpt_equiv (2b) + memory, R-055's
# registered blocking set. The overfit gate is INFORMATIONAL: logged, never blocking; its stored result stays a miss.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/libero-plus/bin/python; G=.venvs/groot/bin/python; O=runs/overnight; Q=$O/queue.log
MINTED=data/r054_train48; MINTED_REPO=local/r054_train48
if [ -z "${VLA_INHIBITED:-}" ]; then
  export VLA_INHIBITED=1
  exec systemd-inhibit --what=sleep:idle:handle-lid-switch --why="VLA R-056 chain" --mode=block bash "$0" "$@"
fi
mkdir -p $O
say() { echo "[$(date '+%F %T')] [r056] $*" >> $Q; }
step() { local n=$1; shift; say "$n start"; local t0=$(date +%s); "$@" >> $O/$n.log 2>&1; local rc=$?
         say "$n end rc=$rc ($(( ($(date +%s) - t0) / 60 )) min)"; return $rc; }
gpu() { flock /tmp/vla_gpu.lock "$@"; }
jtrue() { [ -f "$1" ] && $G -c "import json,sys; d=json.load(open('$1')); sys.exit(0 if ($2) else 1)" 2>/dev/null; }
trained() { $G -c "
import glob, json, sys
ms = [json.load(open(p)) for p in glob.glob('$1/checkpoints/*/pretrained_model/vla_train_meta.json')]
sys.exit(0 if any(m['opt_step'] >= m['opt_steps_total'] for m in ms) else 1)" 2>/dev/null; }
train() {   # train <name> <prefix> <r> <report>
  local dir=runs/$2_r$3_train
  if trained $dir; then say "$1 SKIP (already trained: $dir)"; return 0; fi
  step $1 gpu $PY experiments/r056_train.py --minted-root $MINTED --minted-repo $MINTED_REPO --run-prefix $2 --lora-r $3 \
    $([ -e $dir/checkpoints/last ] && echo --resume) --report $4
}

say "chain start (pid $$); R-055 overfit (informational): ratio $($G -c "import json;print(json.load(open('runs/r055/gates.json'))['overfit'].get('ratio'))")"
if ! jtrue runs/r055/gates.json "all(d[k]['pass'] for k in ('conformance','ckpt_equiv_r16')) and d['memory']['E3_batch1_le_7.4GiB']"; then
  say "R-055 blocking gates (conformance, 2b, memory) FAIL: nothing trains"; OK=0
else
  say "R-055 blocking gates PASS"; OK=1
fi

if [ $OK = 1 ] && train R056_train r056 16 runs/r056_r16_train_report.json && trained runs/r056_r16_train; then
  step R056_eval bash -c "
    flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py select &&
    flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py heldout &&
    flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py nominal --with-ramekin &&
    { [ -f runs/r056_r16/transfer/summary.json ] || flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py transfer; } ;
    $G experiments/r056_eval.py score" \
  && step B_R056_wise bash -c "flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py wise && $G experiments/r056_eval.py score"
else
  say "R-056 train not complete: eval and WiSE skipped"
fi
[ $OK = 1 ] && train B_R058_r4_train r058 4 runs/r058_r4_train_report.json
if [ -f runs/r052/DONE ]; then say "B_R052 SKIP (DONE)"; else step B_R052 bash experiments/queue_r052.sh; fi
say "chain end"
