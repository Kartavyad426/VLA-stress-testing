#!/usr/bin/env bash
# TaskGen agent-eval queue (approved by the user 10-28: "queue sounds good and start the run"; GPU via prim's gate).
# Holds /tmp/vla_gpu.lock for its whole life (prim's condition); stops every server it starts.
#   T1 smoke (9B, 1 call; the queue stops if it fails)  T2 suite (45)  T3 variants (16)  T4 robustness (24)
#   T5 = T2 + T3 on Qwen3.5-35B-A3B, only if >= 18432 MiB RAM is available (needs ~16.4 GB), else skipped.
# Servers: the user's Tau-harness scripts, unchanged (9B: reasoning off; 35B: REASONING=off set here), on :8099.
# Logs: ../VLA-exp-taskgen/runs/taskgen/agent_eval/queue.log. Run from the main checkout:
#   nohup ../VLA-exp-taskgen/experiments/taskgen/queue_agent_eval.sh &
set -uo pipefail
cd /home/imerit/Documents/Code/VLA
WT=/home/imerit/Documents/Code/VLA-exp-taskgen
OUT=$WT/runs/taskgen/agent_eval
mkdir -p "$OUT"
LOG=$OUT/queue.log
TAU=/home/imerit/Documents/Code/Tau-harness/scripts
EVAL="runs/real2sim/v1_dev/py_triton_first.sh $WT/experiments/taskgen/eval_agent.py"
PID=""

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

if [ "${TASKGEN_LOCKED:-0}" != "1" ]; then
    exec env TASKGEN_LOCKED=1 flock -n /tmp/vla_gpu.lock "$0" "$@" || { echo "GPU lock held by another job; not started" >&2; exit 1; }
fi

stop() {
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        log "stopping server pid $PID"
        kill "$PID"; wait "$PID" 2>/dev/null
    fi
    PID=""
    # the 35B server runs inside a systemd scope (no_swap.sh): make sure nothing is left on :8099
    if pgrep -f "llama-server.*--port 8099" > /dev/null; then
        log "server still on :8099 after stop; killing it"; pkill -f "llama-server.*--port 8099"; sleep 3
    fi
}
trap 'stop; log "queue exit"' EXIT

serve() {  # $1 name, rest = command
    local name=$1; shift
    if [ -e "$STOP" ]; then log "$name not started (low-memory stop)"; return 1; fi
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$used" -gt 500 ]; then log "GPU busy ($used MiB used); not starting $name"; return 1; fi
    log "starting $name: $*"
    "$@" >> "$OUT/server_$name.log" 2>&1 &
    PID=$!
    for _ in $(seq 1 120); do
        if curl -sf -m 3 http://127.0.0.1:8099/health > /dev/null 2>&1; then log "$name ready"; return 0; fi
        if ! kill -0 "$PID" 2>/dev/null; then log "$name exited before ready (see server_$name.log)"; PID=""; return 1; fi
        sleep 5
    done
    log "$name not ready after 600 s"; stop; return 1
}

run() {  # $1 test, $2 model alias
    if [ -e "$STOP" ]; then log "test $1 on $2: skipped (low-memory stop)"; return 3; fi
    log "test $1 on $2: start"
    $EVAL --test "$1" --model "$2" >> "$OUT/eval_$1_$2.log" 2>&1
    local rc=$?
    log "test $1 on $2: rc $rc"
    return $rc
}

# monitor (prim's conditions 10-28): every 10 s log RAM available, GPU used, llama-server RSS; if RAM available
# drops below 6 GiB, stop: kill the server and the eval, and mark STOP so no further test or server starts.
STOP=$OUT/STOP_LOWMEM
rm -f "$STOP"
echo "time,mem_available_mib,gpu_used_mib,server_rss_mib" > "$OUT/monitor.csv"
(
    while kill -0 $$ 2>/dev/null; do
        a=$(free -m | awk '/^Mem:/{print $7}')
        g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        r=$(ps -C llama-server -o rss= | awk '{s+=$1} END{print int(s/1024)}')
        echo "$(date '+%F %T'),$a,$g,$r" >> "$OUT/monitor.csv"
        if [ "$a" -lt 6144 ] && [ ! -e "$STOP" ]; then
            touch "$STOP"
            echo "$(date '+%F %T') MemAvailable $a MiB < 6144: stopping the queue" >> "$LOG"
            pkill -f "llama-server.*--port 8099"; pkill -f "taskgen/eval_agent.py"
        fi
        sleep 10
    done
) &

log "queue start; disk free $(df -h / | awk 'NR==2{print $4}'); RAM available $(free -m | awk '/^Mem:/{print $7}') MiB"

# ---- Qwen3.5-9B ----
if serve 9b "$TAU/serve_examiner_9b.sh"; then
    if run smoke qwen3.5-9b; then
        run suite qwen3.5-9b
        run variants qwen3.5-9b
        run robustness qwen3.5-9b
    else
        log "smoke failed on 9B: queue stops here"; exit 2
    fi
    stop
else
    log "9B not served: queue stops"; exit 1
fi

# ---- Qwen3.5-35B-A3B (experts in RAM) ----
AVAIL=$(free -m | awk '/^Mem:/{print $7}')
if [ "$AVAIL" -lt 18432 ]; then
    log "T5 skipped: RAM available $AVAIL MiB < 18432 MiB"
else
    if serve 35b env REASONING=off PARALLEL=1 CTX=61440 "$TAU/no_swap.sh" 22G -- "$TAU/serve_35b_moe_cuda.sh"; then
        if run smoke qwen3.5-35b-a3b; then
            run suite qwen3.5-35b-a3b
            run variants qwen3.5-35b-a3b
        else
            log "smoke failed on 35B"
        fi
        stop
    fi
fi
log "queue done; peak GPU $(awk -F, 'NR>1&&$3>m{m=$3}END{print m}' "$OUT/monitor.csv") MiB, peak server RSS $(awk -F, 'NR>1&&$4>m{m=$4}END{print m}' "$OUT/monitor.csv") MiB, min RAM available $(awk -F, 'NR==2{m=$2}NR>1&&$2<m{m=$2}END{print m}' "$OUT/monitor.csv") MiB"
