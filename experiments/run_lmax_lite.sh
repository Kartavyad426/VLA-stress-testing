#!/usr/bin/env bash
# LIBERO-MAX Lite (800 pairs) for GR00T N1.7, lerobot -640 ports, Q=8, 360 px, in the agreed observability format.
# 8 invocations (4 suites x {Plus, PRO}), one resident checkpoint each, each its own d_step unit (flock
# /tmp/vla_gpu.lock, 14 GiB floor, queue.log lines, peaks). --resume makes the whole script re-runnable.
# NOT launched by its author: needs the user's and prim's GPU OK. Run under sleep inhibition:
#   systemd-inhibit --what=sleep:idle --why="LIBERO-MAX Lite" bash experiments/run_lmax_lite.sh [RUN_ID] [EXTRA_ARGS...]
# EXTRA_ARGS go to the shard, e.g. --every-step-px 256.
# Deliberately NOT in this run: multi-draw B=32 / STAC (it changes the serving noise path; it gets its own run).
set -u
cd "$(dirname "$0")/.."
RUN=${1:-lite_v1}; shift || true
M=third_party/LIBERO-MAX/benchmark/lite/libero_max_lite.json
P=.venvs/libero-plus/bin/python
S=experiments/lmax_groot_shard.py
OUT=runs/lmax/$RUN
mkdir -p "$OUT"
cat > "$OUT/README.md" <<EOF
LIBERO-MAX Lite, GR00T N1.7 (nvidia/gr00t17-lerobot-<suite>-640), Q=8 of H=40, 360 px renders, seed 195.
Runner: experiments/lmax_groot_shard.py + experiments/lmax_obs.py (UNVERIFIED until val passes).
Per <suite>_<source>/: LIBERO-MAX case files (cases/), rollouts.jsonl (schema 3.0 + case/event/goal), npz/.
Score: their aggregator over all 8 roots against the full Lite manifest (see experiments/run_lmax_lite.sh tail).
NOT in this run, by design: multi-draw B=32 / STAC. Seeded batched draws change the serving noise path and would
contaminate the reproduction; STAC gets its own run.
EOF
rc_all=0
for suite in libero_spatial libero_object libero_goal libero_10; do
  for src in plus pro; do
    if [ "$src" = pro ]; then
      ENVV=(PYTHONPATH=$PWD/third_party/LIBERO-PRO LIBERO_CONFIG_PATH=$PWD/third_party/libero-pro-config)
    else
      ENVV=()   # d_step exports the LIBERO-Plus PYTHONPATH/LIBERO_CONFIG_PATH
    fi
    bash experiments/d_step.sh "lmax_${RUN}_${suite}_${src}" --gpu env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "${ENVV[@]}" \
      $P $S $M --suite $suite --source $src --output-root "$OUT/${suite}_${src}" --resume "$@"
    rc=$?; [ $rc -ne 0 ] && rc_all=$rc
  done
done
# Aggregate (CPU): their aggregator, every root, full Lite manifest.
roots=(); for d in "$OUT"/*_plus "$OUT"/*_pro; do [ "$d" = "$OUT/libero_spatial_plus" ] || roots+=(--case-root "$d"); done
PYTHONPATH=third_party/LIBERO-MAX/src:third_party/LIBERO-MAX/scripts $P third_party/LIBERO-MAX/scripts/aggregate_cosmos_benchmark.py \
  "$OUT/libero_spatial_plus" "${roots[@]}" --manifest $M --output-dir "$OUT/summary" --query-interval 8 --require-render-qa \
  > "$OUT/aggregate.log" 2>&1 || rc_all=$?
exit $rc_all
