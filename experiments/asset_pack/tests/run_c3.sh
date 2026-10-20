#!/bin/bash
S=/tmp/claude-1000/-home-imerit-Documents-Code-VLA/fadab121-7275-4659-8165-7c7a75e14b9d/scratchpad
cd /home/imerit/Documents/Code/VLA
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa REPLAY_TARGET=toy_block_2
for v in libero cc; do
  extra="--anchor_frame 212"; [ $v = cc ] && extra="--anchor_frame 212 --assets cc"
  echo "=== ep_c3 $v"
  $S/scripts/run_scene.sh ep_c3 $v $extra
  if [ $v = cc ]; then
    rm -rf $S/runs/tasks_ep_c3_cc
    .venvs/real2sim/bin/python experiments/real2sim/make_tasks.py --scene $S/runs/ep_c3_cc:data/real2sim/droid/ep_c3 --variants 5 --out $S/runs/tasks_ep_c3_cc > $S/runs/tasks_ep_c3_cc.log 2>&1 || echo "make_tasks failed"
    grep -E "^\s*\[[0-9]+/[0-9]+\]" $S/runs/tasks_ep_c3_cc.log
  fi
done
