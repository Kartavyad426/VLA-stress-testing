#!/usr/bin/env bash
# GPU mutex for the experiment scripts.
#
# The scripts used to serialise on `pgrep -f harness_eval.py`. That is unsound:
# ANY process whose command line merely CONTAINS that string matches -- including
# the shell that launched the script, whose argv holds the whole script text. On
# 2026-09-18 this deadlocked the unperturbed control twice: it waited on its own
# launcher and sat idle for 25 minutes reporting nothing wrong.
#
# flock on a real file has no such ambiguity: a holder is a holder.
#   gpu_lock <name> <command...>
exec 9>/tmp/vla_gpu.lock
echo "[$(date +%T)] $1 waiting for GPU lock" >&2
flock 9
echo "[$(date +%T)] $1 holds GPU lock" >&2
NAME=$1; shift
"$@"
