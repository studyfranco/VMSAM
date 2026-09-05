#!/bin/bash
# Stop the runner BY PID FILE, verified BY ARGV POSITION. Never by substring.
#
# `pgrep -f 'merge_queue.py'` matches every command line CONTAINING the string,
# including the shell doing the matching. That cost me my own shell twice.
# My FIRST fix then did `case "$CMD" in *merge_queue.py*)` -- a substring test on
# /proc/PID/cmdline -- and killed my shell a THIRD time, for the same reason.
# A guard built against a trap reproduced the trap.
#
# argv POSITION cannot be spoofed by a shell: for the runner, argv[1] is
# merge_queue.py. For a shell whose command line merely mentions it, argv[0] is
# /bin/bash and the string lives further along.
#
# --check reports and never kills, so the guard can be controlled without pointing
# a live kill at anything.
CHECK=0
[ "$1" = "--check" ] && CHECK=1
P=$(cat "$(dirname "$0")/runs/merge_queue.pid" 2>/dev/null)
[ -z "$P" ] && { echo "no pid file -- refusing to guess"; exit 2; }
ps -p "$P" >/dev/null 2>&1 || { echo "pid $P not running (stale file)"; exit 1; }
ARGV1=$(tr '\0' '\n' < "/proc/$P/cmdline" | sed -n '2p')
if [ "$(basename "${ARGV1:-}")" != "merge_queue.py" ]; then
  echo "REFUSING: pid $P argv[1] is '${ARGV1:-<none>}', not merge_queue.py"
  exit 3
fi
if [ "$CHECK" = "1" ]; then echo "OK: pid $P is the runner (argv[1]=$ARGV1)"; exit 0; fi
kill "$P" && echo "stopped $P"
