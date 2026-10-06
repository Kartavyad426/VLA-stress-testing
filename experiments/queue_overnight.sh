#!/usr/bin/env bash
# Overnight queue, 2026-09-28: R-053 -> R-054 -> R-055 -> R-056 (train, eval) -> R-057, then the backlog
# (R-056 WiSE-FT, R-058 r=4 training, R-052) while time remains before DEADLINE.
#
#   bash experiments/queue_overnight.sh               # launch (only on the user's go)
#   DRY_RUN=1 bash experiments/queue_overnight.sh     # print the plan and time estimates; no GPU, no writes
#   DEADLINE=09:30 | DEADLINE=<epoch s>               # default 10:00 the next morning
#
# * The whole run holds a systemd-inhibit lock on sleep, idle and lid switch (the laptop suspends on battery
#   and on lid close). Check AC power before launching.
# * Every GPU job takes flock /tmp/vla_gpu.lock separately, so another session can slip in between jobs.
#   Every step's start, end, rc and duration go to runs/overnight/queue.log (my prim reads it).
# * Gates: a failed gate skips its dependents; the next independent step still runs (R-057 depends on
#   nothing but R-053's runs, and reruns its own P/k1 if R-053's cannot be reused).
# * A main step does not START after DEADLINE (it is logged SKIP_DEADLINE). A backlog item starts only if
#   its estimate fits before DEADLINE; R-052 needs >= 30 min and is stopped at DEADLINE - 5 min, at a
#   rollout boundary (queue_r052.sh / r047_run.py SIGTERM handling), resumable.
# * runs/overnight/SUMMARY.txt at the end: per step status, rc, duration, and each scorer's headline.
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD

# --- deadline --------------------------------------------------------------------------
DEADLINE=${DEADLINE:-10:00}
if [[ "$DEADLINE" =~ ^[0-9]{9,}$ ]]; then DL=$DEADLINE
else
  DL=$(date -d "today $DEADLINE" +%s)
  [ "$DL" -le "$(date +%s)" ] && DL=$(date -d "tomorrow $DEADLINE" +%s)
fi
export DL

# --- estimates (seconds) ---------------------------------------------------------------------
EST_R053=3000        # 60 rollouts at ~45 s + controls
EST_MINT=6300        # 100 successes / ~0.75 yield x 45 s + 27 nominal recordings
EST_EXPORT=900       # two exports (full, 48-step), CPU
EST_R054_GATES=1500  # G1 decode, spot, G7 videos (CPU) + G6 on 10, f0 re-splice (GPU)
EST_R055=3900        # conformance ~5 min, 2b ~3 min, overfit 300 steps, memory 2 x 20 steps
EST_CACHE=5400       # cache-equivalence gate + precompute over ~57k frames (only if memory fails)
EST_R056_TRAIN=10000 # 2000 optimizer steps x 8 micro-batches
EST_R056_EVAL=14000  # val 4 ckpt x 10, held-out 2 x 108, nominal 2 x 27 (+3 on-ramekin), transfer
EST_R057=11000       # 240 rollouts (+60 if R-053 cannot be reused)
EST_WISE=6100        # R-056 WiSE-FT alpha 0.5: held-out 108 + nominal 27
EST_R058_TRAIN=10000 # R-058 r=4, training only
MIN_R052=1800        # R-052 starts only with >= 30 min left

export PYTHONPATH=$ROOT/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$ROOT/third_party/libero-plus-config MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/libero-plus/bin/python; G=.venvs/groot/bin/python
O=runs/overnight; Q=$O/queue.log; ST=$O/steps.tsv
DRY=${DRY_RUN:-0}
MINTED=data/r054_train48; MINTED_REPO=local/r054_train48

# --- sleep inhibition: re-exec under systemd-inhibit (not in a dry run) ------------------------
if [ "$DRY" != 1 ] && [ -z "${VLA_INHIBITED:-}" ]; then
  export VLA_INHIBITED=1
  exec systemd-inhibit --what=sleep:idle:handle-lid-switch --why="VLA overnight queue" --mode=block bash "$0" "$@"
fi

hms() { printf "%dh%02dm" $(($1 / 3600)) $(($1 % 3600 / 60)); }
PROJ=$(date +%s)          # dry run: projected clock
if [ "$DRY" = 1 ]; then
  echo "DRY RUN $(date '+%F %T')  deadline $(date -d @$DL '+%F %T')  ($(hms $((DL - $(date +%s)))) from now)"
  printf "%-26s %9s %9s  %s\n" step estimate "ends at" "gate / note"
else
  mkdir -p $O; : > $ST
  echo "[$(date '+%F %T')] overnight queue start; deadline $(date -d @$DL '+%F %T'); pid $$" >> $Q
fi
say() { [ "$DRY" = 1 ] || echo "[$(date '+%F %T')] $*" >> $Q; }
record() { [ "$DRY" = 1 ] || printf "%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" >> $ST; }

# step <name> <estimate s> <note> <cmd...>: runs cmd (its own logging to $O/<name>.log) unless past DEADLINE
step() {
  local name=$1 est=$2 note=$3; shift 3
  if [ "$DRY" = 1 ]; then
    if [ $PROJ -ge $DL ]; then
      printf "%-26s %9s %9s  %s  [would not start: past deadline]\n" "$name" "$(hms $est)" - "$note"; return 98
    fi
    PROJ=$((PROJ + est))
    printf "%-26s %9s %9s  %s%s\n" "$name" "$(hms $est)" "$(date -d @$PROJ +%H:%M)" "$note" \
      "$([ $PROJ -gt $DL ] && echo '  [starts before, RUNS PAST deadline]')"
    return 0
  fi
  if [ "$(date +%s)" -ge "$DL" ]; then say "$name SKIP_DEADLINE"; record "$name" SKIP_DEADLINE - 0; return 98; fi
  say "$name start (est $(hms $est))"
  local t0=$(date +%s); "$@" >> $O/$name.log 2>&1; local rc=$?; local dt=$(( $(date +%s) - t0 ))
  say "$name end rc=$rc ($(hms $dt))"
  record "$name" "$([ $rc = 0 ] && echo OK || echo FAIL)" "$rc" "$dt"
  return $rc
}
skip() { say "$1 SKIPPED: $2"; record "$1" "SKIPPED($2)" - 0; [ "$DRY" = 1 ] && printf "%-26s %9s %9s  skipped: %s\n" "$1" - - "$2"; return 0; }
fits() { [ "$DRY" = 1 ] && { [ $((PROJ + $1)) -le "$DL" ]; return; }; [ $(( $(date +%s) + $1 )) -le "$DL" ]; }
gpu() { flock /tmp/vla_gpu.lock "$@"; }

# jtrue <json> <python expr over d>: exit 0 iff the expression is true (dry run: assume pass)
jtrue() { [ "$DRY" = 1 ] && return 0; [ -f "$1" ] || return 1
  $G -c "import json,sys; d=json.load(open('$1')); sys.exit(0 if ($2) else 1)" 2>/dev/null; }
pass() { jtrue "$1" "all((d.get(k) or {}).get('pass') for k in $2)"; }

# =================================================================================================
# 1. R-053
step R053 $EST_R053 "queue_r053.sh + scorer" bash experiments/queue_r053.sh

# 2. R-054 minting, export
OK54=0
if step R054_mint $EST_MINT "to 100 successes, resumable" gpu $PY experiments/r054_mint.py --out runs/r054 --target 100; then
  step R054_export $EST_EXPORT "full + 48-step exports (CPU)" bash -c "
    $G -m vla_harness.data.export_lerobot --records runs/r054 --out data/r054_full --repo-id local/r054_full --overwrite &&
    $G -m vla_harness.data.export_lerobot --records runs/r054 --out $MINTED --repo-id $MINTED_REPO --truncate-steps 48 --overwrite" \
  && OK54=1
fi
# 3. R-054 gates: G1 + G4 + spot gate what follows; G6 is informational
OK54G=0
if [ $OK54 = 1 ]; then
  step R054_gates $EST_R054_GATES "G1 G4 spot [gate]; G6 G7 f0 logged" bash -c "
    $G experiments/r054_gates.py --records runs/r054 --dataset data/r054_full --dataset-train $MINTED &&
    flock /tmp/vla_gpu.lock $PY experiments/r054_gates.py --records runs/r054 --dataset data/r054_full --g6 --f0-check"
  pass runs/r054/gates.json "['G1_records','G1_dataset','G1_dataset_train','spot','spot_train','G4']" && OK54G=1
  say "R-054 gates: G1+G4+spot $([ $OK54G = 1 ] && echo PASS || echo FAIL); G6 $(jtrue runs/r054/gates.json "d['G6']['pass']" && echo pass || echo 'not passed (informational)')"
else skip R054_gates "minting or export failed"; fi

# 4. R-055 gates
OK55=0; CACHED=""
if [ $OK54G = 1 ]; then
  A55="--minted-root $MINTED --minted-repo $MINTED_REPO"
  step R055_conformance 400 "10/10 at <= 1e-3" gpu $PY experiments/r055_gates.py conformance
  step R055_ckpt_equiv 300 "gate 2b" gpu $PY experiments/r055_gates.py ckpt-equiv $A55
  step R055_overfit 1800 "loss < 10% of step 0" gpu $PY experiments/r055_gates.py overfit $A55
  step R055_memory 1400 "b1 <= 7.4 GiB (b2 readout)" gpu $PY experiments/r055_gates.py memory $A55
  # overfit is INFORMATIONAL (R-055 amendment 2026-09-29 15:15): logged, never blocks; blocking set = conformance + 2b + memory
  say "R-055 overfit (informational): $(jtrue runs/r055/gates.json "d['overfit']['pass']" && echo pass || echo miss)"
  if pass runs/r055/gates.json "['conformance','ckpt_equiv_r16']"; then
    if jtrue runs/r055/gates.json "d['memory']['E3_batch1_le_7.4GiB']"; then OK55=1
    else
      say "R-055 memory gate failed: trying the cached-feature path"
      step R055_cache_equiv $EST_CACHE "cached == uncached loss" gpu $PY experiments/r055_gates.py cache-equiv $A55 --cached-features runs/r056_r16_cache
      if pass runs/r055/gates.json "['cache_equiv']"; then OK55=1; CACHED="--cached-features runs/r056_r16_cache"; fi
    fi
  fi
  say "R-055: $([ $OK55 = 1 ] && echo "PASS ${CACHED:+(cached features)}" || echo FAIL)"
else skip R055 "R-054 gates did not pass"; fi

# 5-6. R-056
OK56T=0
if [ $OK55 = 1 ]; then
  RES=""; [ -d runs/r056_r16_train/checkpoints/last ] && RES="--resume"
  step R056_train $EST_R056_TRAIN "r=16, mb 1 x accum 8, 2000 opt steps" gpu $PY experiments/r056_train.py \
    --minted-root $MINTED --minted-repo $MINTED_REPO --lora-r 16 $CACHED $RES --report runs/r056_r16_train_report.json \
    && OK56T=1
else skip R056_train "R-055 gates did not pass"; fi
OK56E=0
if [ $OK56T = 1 ]; then
  step R056_eval $EST_R056_EVAL "select, held-out, nominal (9x3 + on-ramekin), transfer, score" bash -c "
    flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py select &&
    flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py heldout &&
    flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py nominal --with-ramekin &&
    flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py transfer ;
    $G experiments/r056_eval.py score" && OK56E=1
else skip R056_eval "no trained adapter"; fi

# 7-8. R-057 (both halves and its scorer; start pose reuses R-053's P/k1 only if the reuse check passes)
step R057 $EST_R057 "start pose + camera, --reuse-r053" bash experiments/queue_r057.sh --reuse-r053

# =================================================================================================
# Backlog, each only if it fits before DEADLINE
if [ $OK56E = 1 ] && fits $EST_WISE; then
  step B_R056_wise $EST_WISE "exploratory, alpha 0.5" bash -c "
    flock /tmp/vla_gpu.lock $PY experiments/r056_eval.py wise && $G experiments/r056_eval.py score"
else skip B_R056_wise "$([ $OK56E = 1 ] && echo 'does not fit before deadline' || echo 'R-056 eval did not complete')"; fi
if [ $OK55 = 1 ] && fits $EST_R058_TRAIN; then
  RES=""; [ -d runs/r058_r4_train/checkpoints/last ] && RES="--resume"
  step B_R058_r4_train $EST_R058_TRAIN "training only; eval tomorrow" gpu $PY experiments/r056_train.py \
    --minted-root $MINTED --minted-repo $MINTED_REPO --run-prefix r058 --lora-r 4 $CACHED $RES \
    --report runs/r058_r4_train_report.json
else skip B_R058_r4_train "$([ $OK55 = 1 ] && echo 'does not fit before deadline' || echo 'R-055 gates did not pass')"; fi
if fits $MIN_R052; then
  step B_R052 $([ "$DRY" = 1 ] && echo $(( DL - PROJ > 0 ? DL - PROJ : 0 )) || echo $(( DL - $(date +%s) ))) \
    "until DEADLINE - 5 min, resumable" env R052_UNTIL=$DL bash experiments/queue_r052.sh
else skip B_R052 "less than 30 min before deadline"; fi

# =================================================================================================
[ "$DRY" = 1 ] && { echo "projected end $(date -d @$PROJ '+%F %T'); deadline $(date -d @$DL '+%F %T')"; exit 0; }
say "queue end"
$G - <<'PYEOF' > $O/SUMMARY.txt 2>&1
import json, os
def J(p):
    try: return json.load(open(p))
    except Exception: return None
print("Overnight queue summary\n")
print(f"{'step':26s} {'status':34s} {'rc':>4s} {'duration':>9s}")
for line in open("runs/overnight/steps.tsv"):
    n, st, rc, dt = line.rstrip("\n").split("\t")
    dt = int(dt); print(f"{n:26s} {st:34s} {rc:>4s} {dt//3600}h{dt%3600//60:02d}m")
print("\nHeadlines")
s = J("runs/r053/score.json")
if s:
    p = s["pooled"]; print(f"R-053  N success {p['N_success']}/{p['n']}; N flip rate {p['N_flip_rate']}, re-run flip rate {p['rerun_flip_rate']}; "
                          f"expectations {s['expectations']}; interpretable {s['uninterpretable_checks']['interpretable']}")
y = J("runs/r054/yield.json")
if y: print("R-054  yield " + ", ".join(f"{b} {v['successes']}/{v['attempts']}" for b, v in y.items()))
g = J("runs/r054/gates.json")
if g: print("R-054  gates " + ", ".join(f"{k}={v.get('pass')}" for k, v in g.items() if isinstance(v, dict)))
g = J("runs/r055/gates.json")
if g: print("R-055  gates " + ", ".join(f"{k}={v.get('pass', v.get('E3_batch1_le_7.4GiB'))}" for k, v in g.items() if isinstance(v, dict)))
for m in ("memory",):
    if g and m in g: print(f"R-055  memory b1 {g[m]['batch1'].get('nvidia_smi_peak_gib')} GiB, b2 {g[m]['batch2'].get('nvidia_smi_peak_gib')} GiB (nvidia-smi)")
r = J("runs/r056_r16_train_report.json")
if r: print(f"R-056  training {r.get('mode')}: trainable {r.get('trainable', {}).get('trainable')}, peak {r.get('nvidia_smi_peak_gib')} GiB")
s = J("runs/r056_r16/score.json")
if s:
    print(f"R-056  held-out base {s['heldout']['base']['success']}/{s['heldout']['base']['n']} -> retrained "
          f"{s['heldout']['retrained']['success']}/{s['heldout']['retrained']['n']}; gap closed {s['gap_closed']}; "
          f"nominal base {s['nominal']['base']['success']}/{s['nominal']['base']['n']}, retrained {s['nominal']['retrained']['success']}/{s['nominal']['retrained']['n']}")
    print(f"R-056  expectations {s['expectations']}")
    if "wise" in s: print(f"R-056  WiSE-FT {s['wise']}")
s = J("runs/r057/score.json")
if s:
    for row, v in s["rows"].items():
        print(f"R-057  {row}: k* = {v['k_star']}; " + ", ".join(f"{k} {x['success']}/{x['n']}" for k, x in v["by_k"].items()))
r = J("runs/r058_r4_train_report.json")
if r: print(f"R-058  r=4 training {r.get('mode')}: trainable {r.get('trainable', {}).get('trainable')}, peak {r.get('nvidia_smi_peak_gib')} GiB")
for ax in ("camera_bench_yaw_deg", "camera_bench_scale", "camera_bench_pitch_deg"):
    p = f"runs/r052/{ax}/manifest.jsonl"
    if os.path.exists(p):
        rows = [json.loads(l) for l in open(p) if l.strip()]
        print(f"R-052  {ax}: {len(rows)} rollouts, success {sum(r['success'] for r in rows)}")
print("\nR-052 status: " + ("DONE" if os.path.exists("runs/r052/DONE") else "PARTIAL" if os.path.exists("runs/r052/PARTIAL") else "not run"))
PYEOF
say "SUMMARY written: $O/SUMMARY.txt"
