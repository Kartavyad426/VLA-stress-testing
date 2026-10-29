#!/bin/bash
# res31 10-29: re-score the 20 redaction-rerun scenes with build_v1 --report_only (no rebuild, no expert probe):
# drop/scale n/a without a container, B3 0 probes = n/a, A8 'needs review' for an unbuilt grasp near a built object.
cd "$(dirname "$0")/../../.."
one() {
  S=$1; E=$2; CAM=$(.venvs/real2sim/bin/python -c "import json;print(json.load(open('$S/calib.json'))['cam'])")
  echo "$(date '+%F %T') $S"
  nice -n 19 env MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES="" .venvs/real2sim/bin/python -c "import triton, runpy, sys; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')" \
    experiments/real2sim/build_v1.py --episode "$E" --out "$S" --cam "$CAM" --report_only ${LABEL:+--label "$LABEL"} 2>&1 | grep -E "report|rc [1-9]|Error|failed|Traceback"
}
for s in "$@"; do one ${s%%:*} ${s##*:}; done
echo "$(date '+%F %T') all done"
