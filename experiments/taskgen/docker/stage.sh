#!/usr/bin/env bash
# Gather exactly what the TaskGen chat needs into a Docker build context, as hard links (no extra disk, nothing moved).
# The image keeps the repos at the same absolute paths as here, so no hard-coded path in anyone's code has to change.
#   experiments/taskgen/docker/stage.sh [ctx_dir]      then: docker build -t taskgen:cpu <ctx_dir>
set -euo pipefail
CODE=/home/imerit/Documents/Code
CTX=${1:-$CODE/VLA-exp-taskgen/runs/taskgen/docker_ctx}
rm -rf "$CTX"; mkdir -p "$CTX/root"
link() { mkdir -p "$CTX/root$(dirname "$1")"; cp -al "$1" "$CTX/root$1"; }
V=$CODE/VLA
# res31's code (read-only in the image), its CPU wrapper, the asset-pack catalogue
link $V/experiments/real2sim; link $V/experiments/asset_pack; link $V/runs/real2sim/v1_dev/py_triton_first.sh
# scenes (with their composite assets) and their DROID episodes
for s in v1/orange v1/c11 v1/c1b v1/c9 ep_orange_block_stereo; do link $V/runs/real2sim/$s; done
for e in ep_orange_block ep_c11 ep_c1b ep_c9; do link $V/data/real2sim/droid/$e; done
# robot models, asset packs used by the agent (kitchen + GSO; YCB isn't used), LaMa weights (asset builder)
link $V/data/real2sim/menagerie; link $V/data/real2sim/asset_packs/kitchen_mjcf; link $V/data/real2sim/asset_packs/gso_mjcf
link $V/data/real2sim/tools/lama
# the two LIBERO-plus assets the scenes reference (black bowl, coffee cup), not all 9.7 GB
L=$V/third_party/LIBERO-plus/libero/libero/assets
link $L/stable_scanned_objects/akita_black_bowl; link $L/new_objects/coffee_cup/ckkwmj
# TaskGen itself (code only; runs/ is made at run time)
link $CODE/VLA-exp-taskgen/experiments/taskgen
rm -rf "$CTX/root$CODE/VLA-exp-taskgen/experiments/taskgen/docker"
find "$CTX/root" -name __pycache__ -type d -prune -exec rm -rf {} +
cp -a "$(dirname "$0")"/{Dockerfile,requirements-cpu.txt,entrypoint.sh} "$CTX/"
du -sh "$CTX/root"; echo "context: $CTX"
