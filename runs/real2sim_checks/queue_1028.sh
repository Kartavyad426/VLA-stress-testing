#!/usr/bin/env bash
# implres31, 2026-10-28: tests of experiments/real2sim_checks (worktree ../VLA-exp-real2sim-checks) on orange, its injected
# copies and the 10-27 calibration-test scenes. CPU only (osmesa), nice 10, sequential. Reads res31's scenes; writes only
# under runs/real2sim_checks/. Exploratory, UNVERIFIED.
set -u
C=/home/imerit/Documents/Code/VLA-exp-real2sim-checks/experiments/real2sim_checks
V=/home/imerit/Documents/Code/VLA
PY=$V/.venvs/real2sim/bin/python
R=$V/runs/real2sim; D=$V/data/real2sim/droid; O=$V/runs/real2sim_checks; L=$O/logs
export MUJOCO_GL=osmesa
mkdir -p $L $O/tests
run() { local tag=$1; shift; echo "$(date +%H:%M) start $tag"; nice -n 10 $PY "$@" > $L/$tag.log 2>&1; echo "$(date +%H:%M) $tag rc=$?"; }
S=$R/ep_orange_block_stereo; E=$D/ep_orange_block
run b3_orange          $C/expert_probe.py --scene $S --episode $E
$PY -c "import json; c=json.load(open('$C/expert_probe_config.json')); c['sweeps']={'bad_friction_x0.05':{'friction':0.05}}; c['video']['sweeps']=[]; json.dump(c,open('$O/tests/b3_bad_config.json','w'),indent=1)"
run b3_orange_bad      $C/expert_probe.py --scene $S --episode $E --config $O/tests/b3_bad_config.json --out $O/tests/b3_orange_bad
run moge_orange        $C/moge_check.py --scene $S --episode $E
run moge_orange_inject $C/moge_check.py --scene $S --episode $E --inject orange_block:0.02 --wrist_frames 0 --out $O/tests/moge_inject
run ic_orange_hide     $C/image_checks.py --scene $S --episode $E --checks d3,c2 --hide yellow_block --out $O/tests/ic_hide_yellow
for cp in ray_p2cm pitch_3deg side_2cm; do
  run ic_inj_$cp       $C/image_checks.py --scene $R/v1_dev/injection/$cp --episode $E --checks d3,d1,a2,d5 --no_composite --out $O/tests/inj_$cp
done
for sc in c11:ep_c11 c1b:ep_c1b c3h:ep_c3 c9:ep_c9; do
  n=${sc%%:*}; e=${sc##*:}
  run b3_$n            $C/expert_probe.py --scene $R/v1_dev/calibtest/$n --episode $D/$e --sweeps nominal
  run ic_$n            $C/image_checks.py --scene $R/v1_dev/calibtest/$n --episode $D/$e --checks d3,d1,a2,d5 --no_composite
done
echo "$(date +%H:%M) queue done"
