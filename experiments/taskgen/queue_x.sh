#!/usr/bin/env bash
# Agent v3 on the 15 extra presets (presets/v1_orange_x.json, X01-X15, NOT verified by the user), via NIM, CPU only.
# User 10-30: "start on nemotron 550b testing on that, after that oss then nemotron smaller one".
# Per model: suite (first variant x 3 prompts x 2 runs = 90 calls), then variants (all 41 variants, 1 run).
# Outputs runs/taskgen/agent_eval_v3/{suite,variants}_x/<model>/; log runs/taskgen/agent_eval_v3/queue_x.log. Exploratory, UNVERIFIED.
cd /home/imerit/Documents/Code/VLA
LOG=../VLA-exp-taskgen/runs/taskgen/agent_eval_v3/queue_x.log
mkdir -p "$(dirname "$LOG")"
for M in nvidia/nemotron-3-ultra-550b-a55b openai/gpt-oss-20b nvidia/nemotron-3.5-lightning-30b-a3b; do
  for T in suite variants; do
    R=2; [ "$T" = variants ] && R=1
    echo "$(date '+%F %T') start $M $T runs $R" >> "$LOG"
    runs/real2sim/v1_dev/py_triton_first.sh ../VLA-exp-taskgen/experiments/taskgen/eval_agent3.py --test $T --runs $R --tag x \
      --presets ../VLA-exp-taskgen/experiments/taskgen/presets/v1_orange_x.json --model $M 2>&1 | grep -v -i warning \
      > "../VLA-exp-taskgen/runs/taskgen/agent_eval_v3/${T}_x_$(echo $M | tr '/' '_').log"
    echo "$(date '+%F %T') end $M $T rc ${PIPESTATUS[0]}" >> "$LOG"
  done
done
echo "$(date '+%F %T') queue done" >> "$LOG"
