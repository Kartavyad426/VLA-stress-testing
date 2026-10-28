#!/usr/bin/env bash
# implres31, 2026-10-28 (b): re-run after the D1 "far" / D5 floor / D3 wording fixes and the new C4. CPU, nice 10.
set -u
C=/home/imerit/Documents/Code/VLA-exp-real2sim-checks/experiments/real2sim_checks
V=/home/imerit/Documents/Code/VLA; PY=$V/.venvs/real2sim/bin/python; R=$V/runs/real2sim; D=$V/data/real2sim/droid; L=$V/runs/real2sim_checks/logs
export MUJOCO_GL=osmesa
run() { local tag=$1; shift; echo "$(date +%H:%M) start $tag"; nice -n 10 $PY "$@" > $L/$tag.log 2>&1; echo "$(date +%H:%M) $tag rc=$?"; }
run ic_orange_b $C/image_checks.py --scene $R/ep_orange_block_stereo --episode $D/ep_orange_block
for sc in c11:ep_c11 c1b:ep_c1b c3h:ep_c3 c9:ep_c9; do
  n=${sc%%:*}; e=${sc##*:}
  run ic_${n}_b $C/image_checks.py --scene $R/v1_dev/calibtest/$n --episode $D/$e --checks d3,d1,a2,d5 --no_composite
done
echo "$(date +%H:%M) queue done"
