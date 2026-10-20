#!/bin/bash
# usage: run_scene.sh <scene: ep_orange_block_stereo|ep_c3> <variant: libero|cc|...> [extra build args]  (CPU rendering only)
set -e
cd /home/imerit/Documents/Code/VLA
S=/tmp/claude-1000/-home-imerit-Documents-Code-VLA/fadab121-7275-4659-8165-7c7a75e14b9d/scratchpad
scene=$1; v=$2; shift 2
ep=ep_orange_block; [ "$scene" = ep_c3 ] && ep=ep_c3
d=$S/runs/${scene}_$v; mkdir -p $d
cp runs/real2sim/$scene/{perception.npz,perception.json,calib.json,robot_masks_motion.npz} $d/
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
P=.venvs/real2sim/bin/python
f=build_scene_test; [ "$v" = orig ] && f=build_scene_orig_test
$P $S/patch/$f.py --episode data/real2sim/droid/$ep --cam 22008760 --calib $d/calib.json "$@" --out $d > $d/build.log 2>&1
$P experiments/real2sim/replay.py --episode data/real2sim/droid/$ep --cam 22008760 --target ${REPLAY_TARGET:-orange_block} --out $d > $d/replay.log 2>&1 || echo "replay failed (see $d/replay.log)"
$P experiments/real2sim/scene_checks.py --scene $d --episode data/real2sim/droid/$ep --no_overlay 2>&1 | grep -v -i warn > $d/checks.log || true
tail -1 $d/build.log | cut -c1-300; cat $d/checks.log
