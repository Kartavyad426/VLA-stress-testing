#!/usr/bin/env bash
# Called by queue_r056_cont.sh WHILE HOLDING the GPU flock. Decides whether to stop vla-r056 (the running
# fixed chain) so that the continuation can reorder what follows R-056's eval. It stops the unit only when
#   (a) R-056's eval step has ended (or R-056 training ended incomplete), and
#   (b) the chain has since logged a new step "start", and we hold the GPU lock,
# and (c) no python process is left in the unit (checked twice, 3 s apart): the chain is blocked waiting for
# this lock (or between shell commands), so no GPU job and no writer (score.json, manifests) runs in it. Exit 0 = cut done
# (or the chain already ended); 3 = not the moment, release the lock at once.
set -u
cd "$(dirname "$0")/.."
Q=${CUT_Q:-runs/overnight/queue.log}; U=${CUT_UNIT:-vla-r056}    # overrides: test only
say() { echo "[$(date '+%F %T')] [r056cont] $*" >> $Q; }
systemctl --user is-active --quiet $U || exit 0
CG=/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/app.slice/$U.service
idle() {   # no python in the unit: only bash / flock / systemd-inhibit / sleep, i.e. waiting, not working
  local p c
  for p in $(cat $CG/cgroup.procs 2>/dev/null); do
    c=$(cat /proc/$p/comm 2>/dev/null) || continue
    case "$c" in bash|flock|systemd-inhibit|sleep) ;; *) return 1 ;; esac
  done
  return 0
}
trigger() { grep -n "\[r056fix\] \(R056fix_eval end\|R-056 train not complete\)" $Q | tail -1 | cut -d: -f1; }
T=$(trigger)
if [ -z "$T" ]; then
  # eval running and its last GPU phase (transfer) finished: hold the lock until the eval step ends
  grep -q "\[r056fix\] R056fix_eval start" $Q && [ -f runs/r056_r16/transfer/summary.json ] || exit 3
fi
for _ in $(seq 900); do                                    # up to 30 min (score runs on the CPU meanwhile)
  systemctl --user is-active --quiet $U || exit 0
  T=$(trigger)
  if [ -n "$T" ] && tail -n +$((T + 1)) $Q | grep -q "\[r056fix\] .* start$" && idle && sleep 3 && idle; then
    NEXT=$(tail -n +$((T + 1)) $Q | grep "\[r056fix\] .* start$" | head -1 | sed 's/.*\[r056fix\] //')
    say "cutting $U at '$NEXT' (blocked on the GPU lock held by the continuation); its processes:"
    systemctl --user status $U --no-pager 2>/dev/null | sed -n '/CGroup/,$p' | grep -v "^$" | head -12 | sed 's/^/    /' >> $Q
    systemctl --user stop $U
    for _ in $(seq 60); do systemctl --user is-active --quiet $U || break; sleep 1; done
    say "$U stopped ($(systemctl --user is-active $U 2>/dev/null)); the continuation runs the rest"
    exit 0
  fi
  sleep 2
done
say "cut: waited 30 min for the chain's next step after the trigger; releasing the lock"
exit 3
