#!/usr/bin/env bash
# R-051: robustness checks for noise, lighting, layout, texture, language; then the rest of R-047.
# Per-job flock, rc logged, later jobs run even if one fails, the map rebuilt after every job, DONE marker.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/libero-plus/bin/python; Q=runs/r051/queue.log; mkdir -p runs/r051; rm -f runs/r051/DONE
say() { echo "[$(date +%T)] $*" >> "$Q"; }
gpu() { local name="$1"; shift; say "$name start"; flock /tmp/vla_gpu.lock "$@"; local rc=$?; say "$name rc=$rc"; [ $rc = 0 ] || FAILED="$FAILED $name"; map; return $rc; }
map() {   # analysis.json for every finished rollout run, then the map from all run files
  for R in runs/r051_*; do [ -f $R/splice/manifest.jsonl ] && .venvs/groot/bin/python experiments/r039_report.py $R --out /dev/null > /dev/null 2>&1; done
  .venvs/groot/bin/python experiments/compile_map.py >> "$Q" 2>&1 || say "MAP BUILD FAILED"; }
FAILED=""
S=experiments/repro/r051_selection

# 1. reverse direction, forward 0 only (rows.jsonl is appended to, so start clean)
for R in noise light; do rm -rf runs/r051_reverse_$R
  gpu reverse_$R $PY experiments/r048_reverse.py --selection ${S}_$R.json --out runs/r051_reverse_$R > runs/r051/reverse_$R.log 2>&1; done
for R in layout texture; do rm -rf runs/r051_reverse_$R
  gpu reverse_$R $PY experiments/r044_reverse.py --selection ${S}_$R.json --out runs/r051_reverse_$R > runs/r051/reverse_$R.log 2>&1; done

# 2. noise seeds 1 and 2 (resumable from each run's manifest)
for NS in 1 2; do
  for R in noise light layout texture; do
    gpu seed${NS}_$R $PY experiments/r039_run.py --selection ${S}_$R.json --run-id r051_${R}_seed$NS --arms extended --noise-seed $NS --no-video >> runs/r051/${R}_seed$NS.log 2>&1
  done
  gpu seed${NS}_lang $PY experiments/r039_run.py --selection experiments/repro/r049_selection_lang.json --run-id r051_lang_seed$NS --arms base --noise-seed $NS --no-video >> runs/r051/lang_seed$NS.log 2>&1
done

# 3. forward-0 rescue on layout, with the no-op drive
for D in P A N; do
  gpu rescue_layout_$D $PY experiments/r039_run.py --selection ${S}_layout.json --run-id r051_layout_rescue_$D --arms extended --drive $D --drive-until 1 --noise-seed 0 --no-video >> runs/r051/layout_rescue_$D.log 2>&1
done

map
if [ -z "$FAILED" ]; then say DONE; touch runs/r051/DONE; else say "FAILED:$FAILED"; echo "$FAILED" > runs/r051/BLOCKED; fi

# 4. the rest of R-047 (resumes from its manifests); its own queue log and markers
say "R-047 start"; bash experiments/queue_r047.sh; say "R-047 rc=$?"; map; say "ALL DONE"
