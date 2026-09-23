#!/usr/bin/env bash
# R-044: three robustness checks on the wrist-camera claim, queued behind R-041.
# Deterministic (fixed env seed 0, fixed noise seeds), resumable (the runner
# skips instances already in a run's manifest), each GPU job under its own
# flock acquisition, every step's rc logged, later steps run even if an
# earlier one fails, and a DONE marker written at the end.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/libero-plus/bin/python
Q=runs/r044/queue.log
mkdir -p runs/r044
say() { echo "[$(date +%T)] $*" >> "$Q"; }
gpu() { local name="$1"; shift; say "$name start"; flock /tmp/vla_gpu.lock "$@"; local rc=$?; say "$name rc=$rc"; return $rc; }

# 0. wait for R-041's last axis (the queue writes 'joint_radius_rad rc=' when it exits)
say "waiting for R-041"
until grep -q "joint_radius_rad rc=" runs/r041/queue.log 2>/dev/null; do sleep 60; done
say "R-041 finished: $(tail -1 runs/r041/queue.log)"

SEL=experiments/repro/r044_selection_ris.json

# smoke: the new --drive-until path on one instance, must record all arms and drive W for one forward
rm -rf runs/r044_smoke
gpu smoke $PY experiments/r039_run.py --selection $SEL --run-id r044_smoke --arms extended --drive W --drive-until 1 --limit 1 --no-video > runs/r044_smoke.log 2>&1
if ! grep -q '"drive": "W"' runs/r044_smoke/splice/manifest.jsonl 2>/dev/null; then say "SMOKE FAILED"; echo smoke > runs/r044/BLOCKED; fi

# 1. reverse direction, forward 0, no rollouts
gpu reverse $PY experiments/r044_reverse.py --selection $SEL --out runs/r044_reverse --noise-seed 0 > runs/r044_reverse.log 2>&1

# 2. noise seeds 1 and 2, extended arms, drive P (outcomes comparable to R-042's drive-P)
for NS in 1 2; do
  gpu seed$NS $PY experiments/r039_run.py --selection $SEL --run-id r044_seed$NS --arms extended --noise-seed $NS --no-video > runs/r044_seed$NS.log 2>&1
done

# 3. forward-0 rescue: drive W, A, N for forward 0 only, noise seed 0 (R-042's)
if [ ! -f runs/r044/BLOCKED ]; then
  for D in W A N; do
    gpu rescue_$D $PY experiments/r039_run.py --selection $SEL --run-id r044_rescue_$D --arms extended --drive $D --drive-until 1 --noise-seed 0 --no-video > runs/r044_rescue_$D.log 2>&1
  done
fi

# reports (CPU)
.venvs/groot/bin/python experiments/r041_report.py runs/r041 --out docs/R041_RESULTS.html >> "$Q" 2>&1
for R in r044_seed1 r044_seed2 r044_rescue_W r044_rescue_A r044_rescue_N; do
  [ -f runs/$R/splice/manifest.jsonl ] && .venvs/groot/bin/python experiments/r039_report.py runs/$R --out /dev/null >> "$Q" 2>&1
done
say "DONE"; touch runs/r044/DONE
