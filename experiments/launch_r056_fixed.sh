#!/usr/bin/env bash
# Start experiments/queue_r056_fixed.sh as its own transient user unit, outside the editor's cgroup
# (2026-09-29 17:23: systemd-oomd killed app-code-8514.scope and the R-056 eval queue inside it), under a
# systemd-inhibit sleep/lid lock. --collect: the unit is garbage-collected even if it fails.
#   bash experiments/launch_r056_fixed.sh                 # the chain
#   UNIT=vla-noop bash experiments/launch_r056_fixed.sh -- sleep 10    # the same launch around any command
# Watch: systemctl --user status vla-r056; journalctl --user -u vla-r056; runs/overnight/queue.log
set -eu
cd "$(dirname "$0")/.."
UNIT=${UNIT:-vla-r056}
if [ "${1:-}" = "--" ]; then shift; CMD=("$@"); else CMD=(bash experiments/queue_r056_fixed.sh); fi
if systemctl --user is-active --quiet "$UNIT"; then echo "$UNIT is already running" >&2; exit 1; fi
exec systemd-run --user --unit="$UNIT" --collect --working-directory="$PWD" \
  --setenv=PATH="$PATH" --setenv=HOME="$HOME" ${HF_HOME:+--setenv=HF_HOME="$HF_HOME"} \
  --property=KillMode=control-group \
  systemd-inhibit --what=sleep:idle:handle-lid-switch --why="VLA $UNIT" --mode=block "${CMD[@]}"
