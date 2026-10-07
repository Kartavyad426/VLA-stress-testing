#!/usr/bin/env bash
# One metered step of the 10-06 D20/D18a/D19 run, meant to be the whole command of its own unit:
#   UNIT=vla-<name> bash experiments/launch_r056_fixed.sh -- bash experiments/d_step.sh <name> [--gpu] <cmd...>
# --gpu: the command runs under flock /tmp/vla_gpu.lock. Refuses to start if MemAvailable < $MIN_AVAIL_GIB
# (default 14). Log: runs/overnight/<name>.log; peaks: runs/overnight/<name>.peaks.json (stepmeter.py);
# start/end lines with prefix [d1006] in runs/overnight/queue.log.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
G=.venvs/groot/bin/python; O=runs/overnight; Q=$O/queue.log; mkdir -p $O
say() { echo "[$(date '+%F %T')] [d1006] $*" >> $Q; }
n=$1; shift
LOCK=(); if [ "${1:-}" = "--gpu" ]; then LOCK=(flock /tmp/vla_gpu.lock); shift; fi
avail=$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)
if awk "BEGIN{exit !($avail < ${MIN_AVAIL_GIB:-14})}"; then say "$n REFUSED: MemAvailable $avail GiB < ${MIN_AVAIL_GIB:-14}"; exit 75; fi
say "$n start (MemAvailable $avail GiB): $*"; t0=$(date +%s)
$G experiments/stepmeter.py $O/$n.peaks.json -- "${LOCK[@]}" "$@" >> $O/$n.log 2>&1; rc=$?
say "$n end rc=$rc ($(( ($(date +%s) - t0) / 60 )) min) peaks: $($G -c "import json;d=json.load(open('$O/$n.peaks.json'));print(f\"RSS tree {d['rss_tree_max_gib']} GiB (anon {d['rss_anon_tree_max_gib']}), GPU {d['gpu_used_max_gib']} GiB\")" 2>/dev/null)"
exit $rc
