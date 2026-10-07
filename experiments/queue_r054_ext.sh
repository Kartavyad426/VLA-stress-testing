#!/usr/bin/env bash
# D19 (user 2026-10-06): the R-054 minting extension on its own, step 6 of queue_r056_cont.sh unchanged
# (runs/r054_ext, --start-attempt 146, towards TARGET successes, 20 attempts per GPU flock; three chunks in a
# row without progress stop it). Resumable and safe to stop at any point: r054_mint.py appends per attempt.
#   UNIT=vla-r054ext bash experiments/launch_r056_fixed.sh -- bash experiments/queue_r054_ext.sh
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/libero-plus/bin/python; G=.venvs/groot/bin/python; O=runs/overnight; Q=$O/queue.log
EXT=runs/r054_ext; START=146; TARGET=${TARGET:-400}; FAILS=0; CH=0
say() { echo "[$(date '+%F %T')] [r054ext] $*" >> $Q; }
ext_succ() { $G -c "
import json, os
p = '$EXT/manifest.jsonl'
print(sum(bool(json.loads(l).get('success')) for l in open(p) if l.strip()) if os.path.exists(p) else 0)" 2>/dev/null || echo 0; }
say "start: $(ext_succ) successes on record, target $TARGET"
while true; do
  S0=$(ext_succ); [ "$S0" -ge $TARGET ] && break
  N0=$( [ -f $EXT/manifest.jsonl ] && wc -l < $EXT/manifest.jsonl || echo 0 )
  CH=$((CH + 1))
  MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-14} bash experiments/d_step.sh R054ext_mint --gpu \
    $PY experiments/r054_mint.py --out $EXT --start-attempt $START --max-attempts 2000 --target $TARGET --max-new 20
  RC=$?
  N1=$( [ -f $EXT/manifest.jsonl ] && wc -l < $EXT/manifest.jsonl || echo 0 )
  if [ $RC != 0 ] || [ "$N1" -le "$N0" ]; then FAILS=$((FAILS + 1)); say "chunk $CH: rc=$RC, no progress ($FAILS)"; [ $FAILS -ge 3 ] && break
  else FAILS=0; fi
  [ $((CH % 5)) = 0 ] && say "after $CH chunks: $N1 attempts, $(ext_succ) successes"
done
say "end after $CH chunks: $(ext_succ) successes, $( [ -f $EXT/manifest.jsonl ] && wc -l < $EXT/manifest.jsonl || echo 0) attempts"
