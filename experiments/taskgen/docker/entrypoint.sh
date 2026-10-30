#!/usr/bin/env bash
# TaskGen chat in a container: CPU only (MuJoCo offscreen rendering via OSMesa). The NIM key comes from the environment
# (docker run -e NVIDIA_NIM_API_KEY=...); without it only the 'fake' model works.
set -e
cd /home/imerit/Documents/Code/VLA
[ -n "${NVIDIA_NIM_API_KEY:-}" ] || echo "NOTE: NVIDIA_NIM_API_KEY is not set: only the 'fake' model will answer" >&2
exec runs/real2sim/v1_dev/py_triton_first.sh ../VLA-exp-taskgen/experiments/taskgen/chat/server.py --host 0.0.0.0 --port 8765 "$@"
