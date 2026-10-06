#!/usr/bin/env bash
# Continuation of queue_overnight.sh with NO deadline (user, 2026-09-29): waits for the running queue (WAIT_PID)
# to exit, then runs the whole chain to completion: R-053 .. R-056, R-057, R-056 WiSE-FT, R-058 r=4 training,
# R-052 in full (no timeout). A separate file on purpose: bash reads a running script incrementally, so the
# live queue_overnight.sh must not be edited.
#
# Every step is skip-safe: a completed step is detected from its outputs and SKIPPED (never re-run); a partial
# one resumes from its own manifest/checkpoint:
#   R053          skip if runs/r053/DONE
#   R054_mint     skip if runs/r054/yield.json has >= 100 successes (else r054_mint.py resumes by instance id)
#   R054_export   skip if both exports' meta/export.json exist (never overwrites a dataset training read)
#   R054_gates    skip if runs/r054/gates.json already holds G1/spot/G4 and G6/f0
#   R055_*        each gate skipped if its key is in runs/r055/gates.json
#   R056_train    skip if a checkpoint's vla_train_meta.json reached opt_steps_total; else --resume
#   R056_eval     select/heldout/nominal/wise skip rollouts already in the manifest (no policy load if none
#                 are left); transfer skipped if its summary.json exists; score is recomputed (cheap)
#   R057          skip if runs/r057/DONE (else every r039_run job resumes from its manifest)
#   R058_r4_train as R056_train
#   R052          skip if runs/r052/DONE (else each axis resumes from its manifest); no timeout
#
#   WAIT_PID=2421971 bash experiments/queue_overnight_continue.sh      DRY_RUN=1 prints the plan
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
WAIT_PID=${WAIT_PID:-}
export PYTHONPATH=$ROOT/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$ROOT/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/libero-plus/bin/python; G=.venvs/groot/bin/python
O=runs/overnight; Q=$O/queue.log; ST=$O/steps_continue.tsv
DRY=${DRY_RUN:-0}
MINTED=data/r054_train48; MINTED_REPO=local/r054_train48

if [ "$DRY" != 1 ] && [ -z "${VLA_INHIBITED:-}" ]; then
  export VLA_INHIBITED=1
  exec systemd-inhibit --what=sleep:idle:handle-lid-switch --why="VLA overnight queue (continuation)" --mode=block bash "$0" "$@"
fi
say() { if [ "$DRY" = 1 ]; then echo "  $*"; else echo "[$(date '+%F %T')] [cont] $*" >> $Q; fi; }
record() { [ "$DRY" = 1 ] || printf "%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" >> $ST; }
hms() { printf "%dh%02dm" $(($1 / 3600)) $(($1 % 3600 / 60)); }

if [ "$DRY" != 1 ]; then
  mkdir -p $O; : > $ST
  if [ -n "$WAIT_PID" ]; then
    say "waiting for queue_overnight.sh (pid $WAIT_PID) to exit"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  fi
  say "continuation start (no deadline)"
fi

step() {   # step <name> <cmd...>
  local name=$1; shift
  if [ "$DRY" = 1 ]; then echo "RUN   $name"; return 0; fi
  say "$name start"
  local t0=$(date +%s); "$@" >> $O/$name.log 2>&1; local rc=$?; local dt=$(( $(date +%s) - t0 ))
  say "$name end rc=$rc ($(hms $dt))"; record "$name" "$([ $rc = 0 ] && echo OK || echo FAIL)" "$rc" "$dt"
  return $rc
}
done_skip() { say "$1 SKIP (already complete: $2)"; record "$1" "SKIP_DONE" - 0; [ "$DRY" = 1 ] && echo "SKIP  $1  ($2)"; return 0; }
blocked() { say "$1 SKIPPED: $2"; record "$1" "SKIPPED($2)" - 0; [ "$DRY" = 1 ] && echo "BLOCK $1  ($2)"; return 0; }
gpu() { flock /tmp/vla_gpu.lock "$@"; }
jtrue() { [ -f "$1" ] || return 1; $G -c "import json,sys; d=json.load(open('$1')); sys.exit(0 if ($2) else 1)" 2>/dev/null; }
pass() { jtrue "$1" "all((d.get(k) or {}).get('pass') for k in $2)"; }
has() { jtrue "$1" "all(k in d for k in $2)"; }
trained() {   # trained <train dir>: a checkpoint reached its total optimizer steps
  $G -c "
import glob, json, sys
ms = [json.load(open(p)) for p in glob.glob('$1/checkpoints/*/pretrained_model/vla_train_meta.json')]
sys.exit(0 if any(m['opt_step'] >= m['opt_steps_total'] for m in ms) else 1)" 2>/dev/null; }

# 1. R-053
if [ -f runs/r053/DONE ]; then done_skip R053 runs/r053/DONE; else step R053 bash experiments/queue_r053.sh; fi

# 2. R-054
if jtrue runs/r054/yield.json "d['pooled']['successes'] >= 100"; then done_skip R054_mint "yield.json >= 100 successes"
else step R054_mint gpu $PY experiments/r054_mint.py --out runs/r054 --target 100; fi
OK54=0
if jtrue runs/r054/yield.json "d['pooled']['successes'] >= 100"; then
  if [ -f data/r054_full/meta/export.json ] && [ -f $MINTED/meta/export.json ]; then done_skip R054_export "both exports exist"; OK54=1
  else step R054_export bash -c "
    $G -m vla_harness.data.export_lerobot --records runs/r054 --out data/r054_full --repo-id local/r054_full --overwrite &&
    $G -m vla_harness.data.export_lerobot --records runs/r054 --out $MINTED --repo-id $MINTED_REPO --truncate-steps 48 --overwrite" && OK54=1; fi
elif [ "$DRY" = 1 ]; then echo "RUN   R054_export (after minting)"; OK54=1
else blocked R054_export "fewer than 100 minted successes"; fi
OK54G=0
if [ $OK54 = 1 ]; then
  if has runs/r054/gates.json "['G1_records','G1_dataset','G1_dataset_train','spot','G4','G6','f0_check']"; then done_skip R054_gates "gates.json complete"
  else step R054_gates bash -c "
    $G experiments/r054_gates.py --records runs/r054 --dataset data/r054_full --dataset-train $MINTED &&
    flock /tmp/vla_gpu.lock $PY experiments/r054_gates.py --records runs/r054 --dataset data/r054_full --g6 --f0-check"; fi
  { pass runs/r054/gates.json "['G1_records','G1_dataset','G1_dataset_train','spot','spot_train','G4']" || [ "$DRY" = 1 ]; } && OK54G=1
  say "R-054 gates G1+G4+spot: $([ $OK54G = 1 ] && echo PASS || echo FAIL)"
else blocked R054_gates "no exported dataset"; fi

# 3. R-055
OK55=0; CACHED=""
if [ $OK54G = 1 ]; then
  A55="--minted-root $MINTED --minted-repo $MINTED_REPO"
  for pair in conformance:conformance ckpt-equiv:ckpt_equiv_r16 overfit:overfit memory:memory; do
    g=${pair%%:*}; k=${pair##*:}
    if has runs/r055/gates.json "['$k']"; then done_skip R055_$k "gates.json has $k"
    else step R055_$k gpu $PY experiments/r055_gates.py $g $A55; fi
  done
  # overfit is INFORMATIONAL (R-055 amendment 2026-09-29 15:15): logged, never blocks; blocking set = conformance + 2b + memory
  say "R-055 overfit (informational): $(jtrue runs/r055/gates.json "d['overfit']['pass']" && echo pass || echo miss)"
  if pass runs/r055/gates.json "['conformance','ckpt_equiv_r16']" || [ "$DRY" = 1 ]; then
    if jtrue runs/r055/gates.json "d['memory']['E3_batch1_le_7.4GiB']" || [ "$DRY" = 1 ]; then OK55=1
    else
      if has runs/r055/gates.json "['cache_equiv']"; then done_skip R055_cache_equiv "gates.json has cache_equiv"
      else step R055_cache_equiv gpu $PY experiments/r055_gates.py cache-equiv $A55 --cached-features runs/r056_r16_cache; fi
      pass runs/r055/gates.json "['cache_equiv']" && { OK55=1; CACHED="--cached-features runs/r056_r16_cache"; }
    fi
  fi
  say "R-055: $([ $OK55 = 1 ] && echo "PASS ${CACHED:+(cached)}" || echo FAIL)"
else blocked R055 "R-054 gates did not pass"; fi

# 4. R-056
train_step() {   # train_step <name> <prefix> <r> <report>
  local name=$1 pfx=$2 r=$3 rep=$4 dir=runs/$2_r$3_train
  if trained $dir; then done_skip $name "$dir reached its final optimizer step"; return 0; fi
  local RES=""; [ -e $dir/checkpoints/last ] && RES="--resume"
  step $name gpu $PY experiments/r056_train.py --minted-root $MINTED --minted-repo $MINTED_REPO \
    --run-prefix $pfx --lora-r $r $CACHED $RES --report $rep
}
OK56T=0
if [ $OK55 = 1 ]; then train_step R056_train r056 16 runs/r056_r16_train_report.json && { trained runs/r056_r16_train || [ "$DRY" = 1 ]; } && OK56T=1
else blocked R056_train "R-055 gates did not pass"; fi
OK56E=0
if [ $OK56T = 1 ]; then
  step R056_eval bash -c "
    flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py select &&
    flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py heldout &&
    flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py nominal --with-ramekin &&
    { [ -f runs/r056_r16/transfer/summary.json ] || flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py transfer; } ;
    $G experiments/r056_eval.py score" && OK56E=1
else blocked R056_eval "no fully trained adapter"; fi

# 5. R-057
if [ -f runs/r057/DONE ]; then done_skip R057 runs/r057/DONE; else step R057 bash experiments/queue_r057.sh --reuse-r053; fi

# 6. backlog, no deadline
if [ $OK56E = 1 ]; then step B_R056_wise bash -c "
    flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py wise && $G experiments/r056_eval.py score"
else blocked B_R056_wise "R-056 eval did not complete"; fi
if [ $OK55 = 1 ]; then train_step B_R058_r4_train r058 4 runs/r058_r4_train_report.json
else blocked B_R058_r4_train "R-055 gates did not pass"; fi
if [ -f runs/r052/DONE ]; then done_skip B_R052 runs/r052/DONE; else step B_R052 bash experiments/queue_r052.sh; fi

[ "$DRY" = 1 ] && exit 0
say "continuation end"
# the summary: queue_overnight.sh's writer, over both step tables
cat $O/steps.tsv $ST > $O/steps_all.tsv 2>/dev/null
sed -n '/^\$G - <<.PYEOF./,/^PYEOF$/p' experiments/queue_overnight.sh | sed '1d;$d' \
  | sed 's#runs/overnight/steps.tsv#runs/overnight/steps_all.tsv#' > $O/.summary.py
$G $O/.summary.py > $O/SUMMARY.txt 2>&1
say "SUMMARY written: $O/SUMMARY.txt"
