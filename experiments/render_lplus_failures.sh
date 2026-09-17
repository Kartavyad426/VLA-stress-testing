#!/usr/bin/env bash
# Episode video pages for every failed rollout of a LIBERO-Plus harness run.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl
RUN=${1:-runs/lplus_hard_groot_v2}
OUT=${2:-viz/lplus_failures}
MAX=${3:-60}   # cap: each page replays the episode in sim (~15-30 s)
mkdir -p "$OUT"
python3 -c "
import json
for l in open('$RUN/rollouts.jsonl'):
    r = json.loads(l)
    if r['success']: continue
    lp = r['scene_descriptor'].get('libero_plus') or {}
    cat = (lp.get('category') or 'unknown').replace(' ', '_')
    print(r['rollout_id'], cat, lp.get('difficulty_level'), r['task_id'].rsplit('task', 1)[1])
" | head -n "$MAX" | while read RID CAT LV TID; do
  F="$OUT/${CAT}_L${LV}_task${TID}_${RID}.html"
  [ -f "$F" ] && { echo "exists $F"; continue; }
  .venvs/libero-plus/bin/python experiments/visualise_episode_video.py "$RUN" "$RID" "$F" 2>/dev/null | grep -E "fidelity|wrote" | tr '\n' ' '
  echo
done
