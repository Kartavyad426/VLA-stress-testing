#!/bin/bash
# rebuild + checks + scripted expert + make_tasks for: orange libero/cc, c3 libero/cc (CPU rendering)
S=/tmp/claude-1000/-home-imerit-Documents-Code-VLA/fadab121-7275-4659-8165-7c7a75e14b9d/scratchpad
cd /home/imerit/Documents/Code/VLA
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
P=.venvs/real2sim/bin/python
for job in "ep_orange_block_stereo libero --depth stereo" "ep_orange_block_stereo cc --depth stereo --assets cc" \
           "ep_c3 libero --anchor_frame 212" "ep_c3 cc --anchor_frame 212 --assets cc"; do
  set -- $job; scene=$1; v=$2; shift 2
  ep=ep_orange_block; [ "$scene" = ep_c3 ] && ep=ep_c3
  d=$S/runs/${scene}_$v
  echo "=== $scene $v"
  $S/scripts/run_scene.sh $scene $v "$@"
  $P experiments/real2sim/make_tasks.py --scene $d:data/real2sim/droid/$ep --variants 5 --out $S/runs/tasks_${scene}_$v > $S/runs/tasks_${scene}_$v.log 2>&1 || echo "make_tasks failed"
  grep -E "^\s*\[[0-9]+/[0-9]+\]" $S/runs/tasks_${scene}_$v.log
done
