#!/usr/bin/env bash
# Continuation of the R-056 fixed chain (user decision via SFT, 2026-09-29 evening): the overnight run, results in
# the morning. Outcomes are SIGNALS: every step runs and is logged, none stops a later one. Kept checks: r056_train's
# startup schedule guard and r056_eval's schedule_verified refusal (they fire only on a wrong schedule). Physical
# skips only (an eval with no checkpoint logs "skipped: no checkpoint").
#
# Launched as its own unit (vla-r056-cont) with experiments/launch_r056_fixed.sh. It does NOT edit the running
# queue_r056_fixed.sh (vla-r056). It reorders what follows R-056's eval by stopping vla-r056 at a safe point
# (experiments/r056_cut.sh: we hold the GPU lock, the chain is blocked waiting for it, no python left in its unit).
# If that point never comes, vla-r056 runs to its end and everything below still runs, skip-safe.
#
# Order after the cut: R-056 train/eval fill-in (no-ops if vla-r056 finished them) -> R-058 r=4 train + eval ->
# R-058 r=64 train (resident; on OOM: cache-equiv signal, then --cached-features) + eval -> R-056 WiSE-FT 0.5 ->
# R-052 resume -> R-054 minting extension (runs/r054_ext, towards 500 successes; resumable, harmless if cut).
# Each step: own GPU flock, stepmeter peaks (RSS tree/anon, GPU), a headline appended to
# runs/overnight/SUMMARY_r056fix.txt.   DRY_RUN=1 bash experiments/queue_r056_cont.sh  prints the plan.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/libero-plus/bin/python; G=.venvs/groot/bin/python; O=runs/overnight; Q=$O/queue.log
MINTED=data/r054_train48; MINTED_REPO=local/r054_train48
DRY=${DRY_RUN:-0}
if [ "$DRY" = 1 ]; then Q=/dev/stdout; else mkdir -p $O; fi
say() { echo "[$(date '+%F %T')] [r056cont] $*" >> $Q; }
gpu() { flock /tmp/vla_gpu.lock "$@"; }
LAST_RC=0; LAST_MIN=0
metered() {   # metered <name> <cmd...>: log to $O/<name>.log, peaks to $O/<name>.peaks.json
  local n=$1; shift
  if [ "$DRY" = 1 ]; then say "DRY $n: $*"; LAST_RC=0; LAST_MIN=0; return 0; fi
  say "$n start"; local t0=$(date +%s)
  $G experiments/stepmeter.py $O/$n.peaks.json -- "$@" >> $O/$n.log 2>&1; LAST_RC=$?
  LAST_MIN=$(( ($(date +%s) - t0) / 60 ))
  say "$n end rc=$LAST_RC ($LAST_MIN min) peaks: $($G -c "import json;d=json.load(open('$O/$n.peaks.json'));print(f\"RSS tree {d['rss_tree_max_gib']} GiB (anon {d['rss_anon_tree_max_gib']}), largest proc {d['rss_proc_max_gib']} GiB, GPU {d['gpu_used_max_gib']} GiB\")" 2>/dev/null)"
  return $LAST_RC
}
summ() { [ "$DRY" = 1 ] && { say "DRY summary: $*"; return 0; }; $G experiments/r056fix_summary.py "$@" >> $O/summary_calls.log 2>&1 || say "summary failed for $1"; }
has_ckpt() { $G -c "
import glob, json, sys
ms = [json.load(open(p)) for p in glob.glob('$1/checkpoints/*/pretrained_model/vla_train_meta.json')]
sys.exit(0 if any(m.get('schedule_verified') is True for m in ms) else 1)" 2>/dev/null; }
trained() { $G -c "
import glob, json, sys
ms = [json.load(open(p)) for p in glob.glob('$1/checkpoints/*/pretrained_model/vla_train_meta.json')]
sys.exit(0 if any(m['opt_step'] >= m['opt_steps_total'] and m.get('schedule_verified') is True for m in ms) else 1)" 2>/dev/null; }

train() {   # train <name> <prefix> <r> [extra args...]: skip if trained, resume if a checkpoint exists
  local n=$1 pre=$2 r=$3; shift 3
  local dir=runs/${pre}_r${r}_train
  if trained $dir; then say "$n SKIP (trained, schedule verified: $dir)"; summ $n train skip - --run $dir --report runs/${pre}_r${r}_train_report.json --note "already trained"; return 0; fi
  metered $n flock /tmp/vla_gpu.lock $PY experiments/r056_train.py --minted-root $MINTED --minted-repo $MINTED_REPO --run-prefix $pre --lora-r $r \
    $([ -e $dir/checkpoints/last ] && echo --resume) --report runs/${pre}_r${r}_train_report.json "$@"
  summ $n train $LAST_RC $LAST_MIN --run $dir --report runs/${pre}_r${r}_train_report.json --peaks $O/$n.peaks.json
}

evaluate() {   # evaluate <name> <train_dir> <out> <heldout|nominal extra args> -- phases, each its own flock
  local n=$1 td=$2 out=$3; shift 3
  if ! has_ckpt $td && [ "$DRY" != 1 ]; then say "$n skipped: no checkpoint (schedule-verified) in $td"
    summ $n eval skip - --run $out --note "skipped: no checkpoint"; return 0; fi
  local EV="experiments/r056_eval.py --train-dir $td --out $out $*" t0=$(date +%s) rcs=""
  metered ${n}_select flock /tmp/vla_gpu.lock $PY $EV select; rcs="$rcs select=$LAST_RC"
  metered ${n}_heldout flock /tmp/vla_gpu.lock $PY $EV heldout; rcs="$rcs heldout=$LAST_RC"
  metered ${n}_nominal flock /tmp/vla_gpu.lock $PY $EV nominal $NOMX; rcs="$rcs nominal=$LAST_RC"
  if [ -f $out/transfer/summary.json ]; then say "${n}_transfer SKIP (done)"; else metered ${n}_transfer flock /tmp/vla_gpu.lock $PY $EV transfer; rcs="$rcs transfer=$LAST_RC"; fi
  metered ${n}_score $G $EV score; rcs="$rcs score=$LAST_RC"
  summ $n eval "$LAST_RC" $(( ($(date +%s) - t0) / 60 )) --run $out --note "phases:$rcs"
}

say "continuation start (pid $$, cgroup $(cut -d: -f3 /proc/$$/cgroup))"

# 0. reorder: stop vla-r056 at the first safe point after R-056's eval (see r056_cut.sh); else wait for it to end
if [ "$DRY" = 1 ]; then say "DRY: would hold the GPU lock after R-056 eval's last GPU phase and stop vla-r056 once it blocks on it"
else
  while systemctl --user is-active --quiet vla-r056; do
    flock /tmp/vla_gpu.lock bash experiments/r056_cut.sh && break
    sleep 3
  done
  say "vla-r056 is $(systemctl --user is-active vla-r056 2>/dev/null); continuing"
fi

# 1. R-056: train fill-in (normally already trained by vla-r056) and eval fill-in (skips finished rollouts)
train R056_train r056 16
NOMX="--with-ramekin" evaluate R056_eval runs/r056_r16_train runs/r056_r16

# 2. R-058 r=4 (alpha = r): train, then eval reusing R-056's base rollouts
train R058_r4_train r058 4
NOMX="" evaluate R058_r4_eval runs/r058_r4_train runs/r058_r4 --run-prefix r058 --lora-r 4 --base-manifest runs/r056_r16/manifest.jsonl

# 3. R-058 r=64: resident; on OOM, the cache-equiv signal and the cached-features path
R64=runs/r058_r64_train; R64C=runs/r058_r64_cached_train
if trained $R64 || trained $R64C; then say "R058_r64_train SKIP (trained)"
else
  if [ ! -f $O/r058_r64_resident_oom ]; then
    train R058_r64_train r058 64
    if [ $LAST_RC != 0 ] && grep -q "OutOfMemoryError\|CUDA out of memory" $O/R058_r64_train.log 2>/dev/null; then
      touch $O/r058_r64_resident_oom; say "R058_r64_train: resident training OOMed; switching to cached features"
    fi
  fi
  if [ -f $O/r058_r64_resident_oom ]; then
    metered R058_r64_cache_equiv flock /tmp/vla_gpu.lock $PY experiments/r055_gates.py cache-equiv --minted-root $MINTED --minted-repo $MINTED_REPO \
      --lora-r 64 --cached-features runs/r058_r64_cache_equiv
    summ R058_r64_cache_equiv other $LAST_RC $LAST_MIN --peaks $O/R058_r64_cache_equiv.peaks.json \
      --note "signal: $($G -c "import json;d=json.load(open('runs/r055/gates.json')).get('cache_equiv_r64',{});print('pass',d.get('pass'),'worst_rel',d.get('worst_rel'))" 2>/dev/null)"
    metered R058_r64_cached_train flock /tmp/vla_gpu.lock $PY experiments/r056_train.py --minted-root $MINTED --minted-repo $MINTED_REPO --run-prefix r058 \
      --lora-r 64 --cached-features runs/r058_r64_cache --out $R64C $([ -e $R64C/checkpoints/last ] && echo --resume) \
      --report runs/r058_r64_cached_train_report.json
    summ R058_r64_cached_train train $LAST_RC $LAST_MIN --run $R64C --report runs/r058_r64_cached_train_report.json --peaks $O/R058_r64_cached_train.peaks.json
  fi
fi
TD64=$R64; has_ckpt $R64C && ! has_ckpt $R64 && TD64=$R64C
NOMX="" evaluate R058_r64_eval $TD64 runs/r058_r64 --run-prefix r058 --lora-r 64 --base-manifest runs/r056_r16/manifest.jsonl

# 4. R-056 WiSE-FT alpha 0.5 (held-out + nominal), exploratory
if [ -f runs/r056_r16/selection.json ] || [ "$DRY" = 1 ]; then
  metered R056_wise flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py wise; W=$LAST_RC; WM=$LAST_MIN
  metered R056_wise_score $G experiments/r056_eval.py score
  summ R056_wise wise $W $WM --run runs/r056_r16 --peaks $O/R056_wise.peaks.json
else say "R056_wise skipped: no checkpoint selection"; summ R056_wise wise skip - --run runs/r056_r16 --note "skipped: no selection"; fi

# 5. R-052 resume (per-axis flock inside queue_r052.sh)
if [ -f runs/r052/DONE ]; then say "R052 SKIP (DONE)"; summ R052 r052 skip - --note "already DONE"
else metered R052_resume bash experiments/queue_r052.sh; summ R052 r052 $LAST_RC $LAST_MIN --peaks $O/R052_resume.peaks.json; fi

# 6. data prep: R-054 minting extension towards 500 successes (400 here + the pilot's 100), 20 attempts per flock
EXT=runs/r054_ext; START=146; TARGET=400; FAILS=0; CH=0
ext_succ() { $G -c "
import json, os
p = '$EXT/manifest.jsonl'
print(sum(bool(json.loads(l).get('success')) for l in open(p) if l.strip()) if os.path.exists(p) else 0)" 2>/dev/null || echo 0; }
while [ "$DRY" != 1 ]; do
  S0=$(ext_succ); [ "$S0" -ge $TARGET ] && break
  N0=$( [ -f $EXT/manifest.jsonl ] && wc -l < $EXT/manifest.jsonl || echo 0 )
  CH=$((CH + 1))
  metered R054ext_mint flock /tmp/vla_gpu.lock $PY experiments/r054_mint.py --out $EXT --start-attempt $START --max-attempts 2000 --target $TARGET --max-new 20
  N1=$( [ -f $EXT/manifest.jsonl ] && wc -l < $EXT/manifest.jsonl || echo 0 )
  if [ $LAST_RC != 0 ] || [ "$N1" -le "$N0" ]; then FAILS=$((FAILS + 1)); say "R054ext chunk $CH: rc=$LAST_RC, no progress ($FAILS)"; [ $FAILS -ge 3 ] && break
  else FAILS=0; fi
  [ $((CH % 5)) = 0 ] && summ R054ext_mint mint $LAST_RC $LAST_MIN --run $EXT --peaks $O/R054ext_mint.peaks.json --note "progress after $CH chunks"
done
[ "$DRY" = 1 ] && say "DRY R054ext_mint: loop of: gpu $PY experiments/r054_mint.py --out $EXT --start-attempt $START --max-attempts 2000 --target $TARGET --max-new 20"
summ R054ext_mint mint $LAST_RC - --run $EXT --note "final ($CH chunks)"
say "continuation end"
