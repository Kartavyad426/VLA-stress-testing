#!/usr/bin/env bash
# R-049: map the three remaining categories, after R-048 finishes. Per-job flock, rc logged, DONE marker.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/libero-plus/bin/python; Q=runs/r049/queue.log; SEL=experiments/repro/r049_selection_rest.json; mkdir -p runs/r049
say() { echo "[$(date +%T)] $*" >> "$Q"; }
gpu() { local name="$1"; shift; say "$name start"; flock /tmp/vla_gpu.lock "$@"; local rc=$?; say "$name rc=$rc"; return $rc; }
say "waiting for R-048"; until [ -f runs/r048/DONE ]; do sleep 60; done
# smoke: one language instance (the new paired-text path) and one texture instance (recorded path on a non-RIS category)
rm -rf runs/r049_smoke
$PY - <<'PYEOF'
import json
s=json.load(open('experiments/repro/r049_selection_rest.json'))
pick=[next(i for i in s['instances'] if i['category']=='Language Instructions'), next(i for i in s['instances'] if i['category']=='Background Textures')]
s['instances']=pick; s['n']=2; json.dump(s, open('experiments/repro/r049_selection_smoke.json','w'), indent=1)
PYEOF
gpu smoke $PY experiments/r039_run.py --selection experiments/repro/r049_selection_smoke.json --run-id r049_smoke --arms extended --no-video > runs/r049_smoke.log 2>&1
if [ "$(wc -l < runs/r049_smoke/splice/manifest.jsonl 2>/dev/null || echo 0)" -lt 2 ]; then say "SMOKE FAILED"; echo smoke > runs/r049/BLOCKED; exit 1; fi
gpu map $PY experiments/r039_run.py --selection $SEL --run-id r049 --arms extended --no-video > runs/r049/run.log 2>&1
.venvs/groot/bin/python experiments/r039_report.py runs/r049 --out /dev/null >> "$Q" 2>&1
say DONE; touch runs/r049/DONE
