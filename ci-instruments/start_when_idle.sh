#!/bin/bash
# Start the corpus run ONLY when the container is idle with an empty queue.
# Two runners were briefly alive against one serial worker tonight because I
# restarted without killing the first; the new runner would have attributed the
# old one's artefact to its own file. This makes "one client, one worker" a
# precondition the script enforces rather than something I remember.
cd /home/vmsam/src/VMSAM_HELP_AI/ci
export ONLY_IDS='694,695,696,697,698,699,9,126,134,141,268,70,83,91,98,5,20,21,22,23,26,27,29,31,33,44,46,47,55,56,58,61,62,65,73,79,108,109,111,113,118,129,131,132,133,136,137,138,139,140,142,143,147,148,149,150,154,155,156,158,160,161,162,163,164,167,169,173,179,214,223,229,240,242,244,269,270,279,281,282,286,291,294,295,296,316,318,319,320,352,353,375,403,406,407,422,428,431,688,691'
export OUT_FILE="runs/weekend-100.jsonl"
# ATOMIC LOCK. My first attempt scanned for other waiters with ps -- and two waiters
# started three seconds apart EACH SAW THE OTHER AND BOTH EXITED. A check-then-act
# scan cannot exclude anything: the window between the check and the act is exactly
# where the other process lives. `mkdir` either creates the directory or fails, in
# one indivisible step, and that is the whole difference between a guard and a race.
if ! mkdir runs/.waiter.lock 2>/dev/null; then
  LOCKPID=$(cat runs/.waiter.lock/pid 2>/dev/null)
  if [ -n "$LOCKPID" ] && [ -e "/proc/$LOCKPID" ]; then
    echo "REFUSING: waiter pid $LOCKPID holds the lock"; exit 1
  fi
  echo "stale lock from pid ${LOCKPID:-unknown}; reclaiming"
  rm -rf runs/.waiter.lock && mkdir runs/.waiter.lock || { echo "could not reclaim"; exit 1; }
fi
echo $$ > runs/.waiter.lock/pid
trap 'rm -rf runs/.waiter.lock' EXIT

for i in $(seq 1 240); do
  # TWO CHECKS, AND THE PID ONE IS AUTHORITATIVE.
  # A pid captured at launch is a thing you already hold; a name pattern is a thing
  # you must recall correctly under pressure -- and I have matched my own process in
  # a process query five times tonight, twice after writing down the fix. So the
  # primary check reads a pidfile and /proc, which cannot match itself. The name
  # scan stays only to catch a runner THIS script did not launch, it excludes its own
  # process tree, and a false positive there merely refuses to start -- the safe way
  # for that check to be wrong.
  if [ -f runs/runner.pid ] && [ -e "/proc/$(cat runs/runner.pid 2>/dev/null)" ]; then
    echo "REFUSING: runner pid $(cat runs/runner.pid) is still alive"; exit 1
  fi
  if [ "$(ps -eo pid,args | awk -v me=$$ -v pa=$PPID '$1!=me && $1!=pa && /merge_queue\.py/ && !/awk/' | wc -l)" != "0" ]; then
    echo "REFUSING: a runner this script did not launch is alive"; exit 1
  fi
  S=$(curl -s --max-time 15 "http://${VMSAM_TEST_HOST:-showgestionar-test}:8080/fusion" \
      | python3 -c 'import json,sys;d=json.load(sys.stdin);print(1 if d.get("status")=="idle" and not d.get("current_job") and not d.get("queue_length") else 0)' 2>/dev/null)
  if [ "$S" = "1" ]; then
    echo "container idle at attempt $i -- starting the run"
    echo $$ > runs/runner.pid          # the pid becomes the handle, before exec
    exec python3 merge_queue.py
  fi
  sleep 30
done
echo "GAVE UP: container never went idle in 2 h"; exit 2
