#!/usr/bin/env bash
# UNPERTURBED control on the LIBERO-Plus stack.
#
# First attempt (14:29) failed in 13 s and it was the right failure: there IS no
# vanilla libero_spatial in .venvs/libero-plus. The fork REPLACES the libero
# package, so its task suite is the 2,402 perturbed variants and a vanilla run
# cannot even find its init states (FileNotFoundError on `*.pruned_init`).
#
# The substitute is exact rather than approximate. 390 variants have a canonical
# camera (`view_0_0_100_0_0`) and `initstate_0`, so their ONLY perturbation is
# the rewritten instruction. Force the base-scene instruction on one of those
# (--base-instruction) and every factor is at its unperturbed value: same
# scenes, same camera, same start pose, same text the policies trained on, same
# stack as R-026. One variant per base scene, so all 10 scenes are covered.
#
# Why this matters: R-022's 98/100 was measured in .venvs/groot (robosuite 1.4.0,
# hf-libero). For the 5-10 pp appearance-perturbation deltas, a 1-2 pp stack
# difference is a real fraction of the effect being read.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config MUJOCO_GL=egl
OUT=experiments/repro/runs/overnight_20260918
IDS=$(cat experiments/repro/lplus_control_ids.txt)
exec 9>/tmp/vla_gpu.lock; flock 9   # GPU mutex (see experiments/gpu_lock.sh)
echo "[$(date +%T)] unperturbed control start" | tee -a "$OUT/progress.log"
.venvs/libero-plus/bin/python experiments/harness_eval.py --libero-plus --base-instruction \
  --checkpoint nvidia/gr00t17-lerobot-libero_spatial-640 --n-action-steps 16 \
  --override base_model_path=nvidia/GR00T-N1.7-3B --override embodiment_tag=libero_sim \
  --dtype bfloat16 --rename-map '{"observation.images.image2": "observation.images.wrist_image"}' \
  --obs-size 360 --suites libero_spatial --episodes 10 --tasks "$IDS" \
  --run-id groot_control_lplus_stack >> "$OUT/baseline.log" 2>&1
RC=$?
echo "[$(date +%T)] unperturbed control done rc=$RC" | tee -a "$OUT/progress.log"
N=$(python3 -c "
import json
rs=[json.loads(l) for l in open('runs/groot_control_lplus_stack/rollouts.jsonl')]
print(f'{sum(r[\"success\"] for r in rs)}/{len(rs)}')" 2>/dev/null || echo "FAILED")
echo "[$(date +%T)] control result: $N" | tee -a "$OUT/progress.log"
touch runs/groot_control_lplus_stack/DONE
notify-send -u critical "VLA: unperturbed control done" "GR00T, canonical camera + initstate 0 + base instruction, LIBERO-Plus stack: $N"
