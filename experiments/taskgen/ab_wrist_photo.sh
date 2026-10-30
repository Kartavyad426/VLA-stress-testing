#!/usr/bin/env bash
# A/B: GR00T on chat turn 4 (v1/c11, "put green block into the bowl"; the expert solved it 10-30 13:09) with the wrist
# real-gripper photo ON (blend trial) vs OFF (sim gripper always), 3 episodes each, interleaved (on, off, on, off, ...).
# User 10-30: "yes ask prim and run the A/B". Each episode via run_groot.sh (GPU lock, idle check, RAM abort).
# prim's GO 10-30 13:5x: start each episode only with MemAvailable >= 14 GiB; stop the loop on any failure, no retry.
# 3 vs 3 can't settle the question: report counts, not a verdict. UNVERIFIED.
cd /home/imerit/Documents/Code/VLA
P=../VLA-exp-taskgen/runs/taskgen/chat/policy/20261029_164333_turn004_groot_133019
OUT=../VLA-exp-taskgen/runs/taskgen/ab_wrist_photo
mkdir -p $OUT
# prim's resume GO (10-30 14:0x): run the whole loop under ONE lock:
#   flock -n -E 75 /tmp/vla_gpu.lock env TASKGEN_LOCKED=1 START_C=off ab_wrist_photo.sh
# (run_groot.sh then skips its own flock); before each episode: no policy server left (30 s), GPU <= 500 MiB, RAM >= 14 GiB.
# Earlier (per-episode lock): The previous episode's policy server can still hold the inherited lock fd for
# a moment after run_groot.sh returns (13:47:10: ep1 off failed at once), so wait until no policy server is left.
wait_server_gone() { for i in $(seq 30); do pgrep -f "policy_rollout.py --serve" > /dev/null || return 0; sleep 1; done; return 1; }
START_K=${START_K:-1}; START_C=${START_C:-on}  # resume point
for k in $(seq $START_K 3); do
  for c in on off; do
    [ $k -eq $START_K ] && [ "$START_C" = off ] && [ $c = on ] && continue
    wait_server_gone || { echo "$(date '+%F %T') STOP: a policy server is still running before ep$k photo=$c" >> $OUT/ab.log; exit 3; }
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$used" -gt 500 ]; then echo "$(date '+%F %T') STOP before ep$k photo=$c: GPU busy ($used MiB)" >> $OUT/ab.log; exit 4; fi
    ram=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$ram" -lt 14336 ]; then echo "$(date '+%F %T') STOP before ep$k photo=$c: MemAvailable $ram MiB < 14 GiB (prim's condition a)" >> $OUT/ab.log; exit 2; fi
    echo "$(date '+%F %T') start ep$k photo=$c (MemAvailable $ram MiB)" >> $OUT/ab.log
    ../VLA-exp-taskgen/experiments/taskgen/run_groot.sh $P/scene_config.json $P/task_config.json $OUT/photo_${c}_ep$k 30 --wrist_gripper_photo $c > $OUT/photo_${c}_ep$k.stdout 2>&1
    rc=$?
    echo "$(date '+%F %T') end ep$k photo=$c rc $rc" >> $OUT/ab.log
    # prim's condition b: any failure (incl. "GPU busy" / "lock held") stops the loop, no retry
    if [ $rc -ne 0 ]; then echo "$(date '+%F %T') STOP: ep$k photo=$c failed (rc $rc): $(grep -m1 -i 'busy\|lock held\|error' $OUT/photo_${c}_ep$k.stdout)" >> $OUT/ab.log; exit 1; fi
  done
done
