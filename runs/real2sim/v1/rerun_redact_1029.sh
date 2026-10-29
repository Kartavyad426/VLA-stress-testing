#!/usr/bin/env bash
# res31 10-29: rerun build_v1 with redaction (user: "redact them, make json, have a check before scripted to not check
# for these"): build_scene --redact_unmatched (no stand-in ellipsoids), grasps of never-built objects -> scene_fit
# 'unbuilt', redacted.json + gate before B3. Each scene's previous outputs are copied to <scene>/before_redact_1029/
# first. Existing camera, nouns, perception and wrist plane are reused (no NIM, CPU only). Exploratory, UNVERIFIED.
cd "$(dirname "$0")/../../.."
one() {  # <scene dir> <episode dir>
  S=$1; E=$2; B=$S/before_redact_1029
  [ -d "$B" ] || { mkdir -p "$B"; cp -r "$S"/v1_report.json "$S"/v1_report.html "$S"/scene_fit.json "$S"/scene.mjb "$S"/scene.xml "$B"/ 2>/dev/null
                   cp -r "$S"/scene_checks.json "$S"/checks "$B"/ 2>/dev/null; }
  CAM=$(.venvs/real2sim/bin/python -c "import json;print(json.load(open('$S/calib.json'))['cam'])")
  echo "$(date '+%F %T') $S (cam $CAM)"
  nice -n 19 env MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES="" .venvs/real2sim/bin/python -c "import triton, runpy, sys; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')" \
    experiments/real2sim/build_v1.py --episode "$E" --out "$S" --cam "$CAM" 2>&1 | grep -E "unbuilt|gate|report:|rc [1-9]|Error|failed"
}
for n in c00_2023-07-07-16h-16m-13s c03_2023-07-13-11h-24m-13s c05_2023-07-21-12h-23m-24s c06_2023-07-21-15h-26m-45s \
         c10_2023-07-27-08h-47m-54s c14_2023-08-17-17h-02m-12s c15_2023-08-12-18h-57m-44s c16_2023-08-12-18h-02m-23s \
         c17_2023-08-18-11h-40m-54s c18_2023-08-18-17h-06m-22s c20_2023-07-14-15h-58m-23s c21_2023-07-14-16h-58m-47s \
         c22_2023-07-14-15h-13m-54s c26_2023-09-02-14h-02m-53s; do
  one runs/real2sim/v1/phase1/$n data/real2sim/droid/phase1/$n
done
for s in orange:ep_orange_block c11:ep_c11 c1b:ep_c1b c3:ep_c3 c9:ep_c9; do
  one runs/real2sim/v1/${s%%:*} data/real2sim/droid/${s##*:}
done
echo "$(date '+%F %T') all done"
