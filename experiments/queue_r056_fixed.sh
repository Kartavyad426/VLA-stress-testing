#!/usr/bin/env bash
# R-056 rerun with the fixed LR schedule (user's option (b), 2026-09-29, via SFT). The first run
# (runs/r056_r16_cyclic_train) cycled its LR: lerobot_train steps the scheduler per micro-batch.
# Chain: R-056 train -> R-056 eval -> WiSE-FT (alpha 0.5) -> R-058 r=4 TRAINING -> R-052 resume. No deadline.
#
# Launch ONLY through experiments/launch_r056_fixed.sh: it starts this script as its own user unit
# (systemd-run --user --unit=vla-r056), outside the editor's scope, so an oomd kill of VS Code cannot take
# it down, and holds a systemd-inhibit sleep lock around it.
#   DRY_RUN=1 bash experiments/queue_r056_fixed.sh     # prints what it would run; touches nothing
#
# Every GPU job takes its own flock. Resume-safe: a run is "trained" only when a checkpoint at
# opt_steps_total has schedule_verified (r056_train's LR guard), otherwise training resumes from `last`;
# eval phases skip finished rollouts and refuse unverified adapters; R-052 resumes per axis.
# Gating (R-055 amendment 2026-09-29 15:15): conformance + ckpt_equiv (2b) + memory block; overfit is informational.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/libero-plus/bin/python; G=.venvs/groot/bin/python; O=runs/overnight; Q=$O/queue.log
MINTED=data/r054_train48; MINTED_REPO=local/r054_train48
DRY=${DRY_RUN:-0}
if [ "$DRY" = 1 ]; then Q=/dev/stdout; else mkdir -p $O; fi
say() { echo "[$(date '+%F %T')] [r056fix] $*" >> $Q; }
step() {   # step <name> <cmd...>: logged to $O/<name>.log
  local n=$1; shift
  if [ "$DRY" = 1 ]; then say "DRY $n: $*"; return 0; fi
  say "$n start"; local t0=$(date +%s); "$@" >> $O/$n.log 2>&1; local rc=$?
  say "$n end rc=$rc ($(( ($(date +%s) - t0) / 60 )) min)"; return $rc
}
gpu() { flock /tmp/vla_gpu.lock "$@"; }
jtrue() { [ -f "$1" ] && $G -c "import json,sys; d=json.load(open('$1')); sys.exit(0 if ($2) else 1)" 2>/dev/null; }
trained() { $G -c "
import glob, json, sys
ms = [json.load(open(p)) for p in glob.glob('$1/checkpoints/*/pretrained_model/vla_train_meta.json')]
sys.exit(0 if any(m['opt_step'] >= m['opt_steps_total'] and m.get('schedule_verified') is True for m in ms) else 1)" 2>/dev/null; }
check_trained() {   # in a dry run nothing trains, so report the check and assume it would pass
  if [ "$DRY" = 1 ]; then trained $1 && say "DRY trained($1): already true" || say "DRY trained($1): false now, assumed true after training"; return 0; fi
  trained $1
}
train() {   # train <name> <prefix> <r> <report>
  local dir=runs/$2_r$3_train
  if trained $dir; then say "$1 SKIP (already trained, schedule verified: $dir)"; return 0; fi
  step $1 gpu $PY experiments/r056_train.py --minted-root $MINTED --minted-repo $MINTED_REPO --run-prefix $2 --lora-r $3 \
    $([ -e $dir/checkpoints/last ] && echo --resume) --report $4
}

say "chain start (pid $$, cgroup $(cut -d: -f3 /proc/$$/cgroup))"
if ! jtrue runs/r055/gates.json "all(d[k]['pass'] for k in ('conformance','ckpt_equiv_r16')) and d['memory']['E3_batch1_le_7.4GiB']"; then
  say "R-055 blocking gates (conformance, 2b, memory) FAIL: nothing trains"; OK=0
else
  say "R-055 blocking gates PASS (overfit informational)"; OK=1
fi

EV="flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py"
if [ $OK = 1 ] && train R056fix_train r056 16 runs/r056_r16_train_report.json && check_trained runs/r056_r16_train; then
  step R056fix_eval bash -c "
    $EV select &&
    $EV heldout &&
    $EV nominal --with-ramekin &&
    { [ -f runs/r056_r16/transfer/summary.json ] || $EV transfer; } ;
    $G experiments/r056_eval.py score" \
  && step R056fix_wise bash -c "$EV wise && $G experiments/r056_eval.py score"
else
  say "R-056 train not complete: eval and WiSE skipped"
fi
[ $OK = 1 ] && train R058fix_r4_train r058 4 runs/r058_r4_train_report.json
if [ -f runs/r052/DONE ]; then say "R052 SKIP (DONE)"; else step R052_resume bash experiments/queue_r052.sh; fi
say "chain end"
