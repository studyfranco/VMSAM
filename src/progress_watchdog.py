"""Progress watchdogs: a step is stopped when it stops making progress, never for being slow (decision: no-budget-watchdog)."""
import subprocess
import threading
import time

import tools

# No progress at all for this long means the step is stuck. Measured 2026-10-23 on 17 replays at
# host load 60-75: the longest idle gap over 4693 watched calls was 4.0 s (fingerprint_decode),
# so 600 s is 150 times that (decision: no-budget-watchdog).
STALL_S = 600.0
POLL_S = 1.0


class step_stalled(Exception):
    """A step made no progress: no output, no new frame, a repeated search state or a structural bound passed."""

    def __init__(self, step, rule, detail=""):
        super().__init__(f"{step} made no progress ({rule}) {detail}".strip())
        self.step = step
        self.rule = rule


def process_progress(pid):
    """Return what process `pid` has read and written (CPU ticks if /proc/<pid>/io is unreadable), None once gone.

    A process looping without any input or output therefore reads as stalled."""
    try:
        with open(f"/proc/{pid}/io") as handle:
            counters = dict(line.split(":", 1) for line in handle if ":" in line)
        return int(counters["rchar"]), int(counters["wchar"])
    except (OSError, KeyError, ValueError):
        pass
    try:
        with open(f"/proc/{pid}/stat") as handle:
            fields = handle.read().rsplit(")", 1)[1].split()
        return int(fields[11]) + int(fields[12])
    except (OSError, IndexError, ValueError):
        return None


def tree_progress(pid):
    """Return `process_progress` of process `pid` and of every process it started."""
    import psutil
    try:
        pids = [pid] + [child.pid for child in psutil.Process(pid).children(recursive=True)]
    except psutil.Error:
        pids = [pid]
    return [(one, process_progress(one)) for one in pids]


class watch:
    """Call `stop()` once `progress()` returned one value for `stall_s` seconds, then raise `step_stalled` on exit.

    with watch("decode", lambda: process_progress(process.pid), process.kill): process.communicate()"""

    def __init__(self, step, progress, stop, stall_s=None, detail=""):
        self.step = step
        self.progress = progress
        self.stop = stop
        self.stall_s = STALL_S if stall_s is None else stall_s
        self.detail = detail
        self.stalled = False
        self.max_idle_s = 0.0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        last = self.progress()
        last_change = time.monotonic()
        while not self.done.wait(min(POLL_S, self.stall_s)):
            value = self.progress()
            now = time.monotonic()
            if value != last:
                last, last_change = value, now
                continue
            self.max_idle_s = max(self.max_idle_s, now - last_change)
            if now - last_change >= self.stall_s:
                self.stalled = True
                try:
                    self.stop()
                except Exception as error:                               # noqa: BLE001
                    tools.dev_log(f"watchdog: step={self.step} stop raised "
                                  f"{type(error).__name__}: {error}\n")
                return

    def __enter__(self):
        self.started = time.monotonic()
        self.thread.start()
        return self

    def __exit__(self, kind, error, trace):
        self.done.set()
        self.thread.join()
        wall = time.monotonic() - self.started
        tools.dev_log(f"watchdog: step={self.step} wall_s={round(wall, 2)} "
                      f"max_idle_s={round(self.max_idle_s, 2)} stalled={self.stalled}\n")
        if self.stalled:
            raise step_stalled(self.step, "no_progress",
                               f"nothing new for {round(self.stall_s, 1)} s after "
                               f"{round(wall, 1)} s {self.detail}".strip()) from error
        return False


class search_watch:
    """Stop a search that revisits a state or runs more iterations than its structural bound.

    search_watch("ladder", bound=3).visit((window_s, threshold))"""

    def __init__(self, step, bound):
        self.step = step
        self.bound = bound
        self.seen = set()

    def visit(self, state):
        if state in self.seen:
            raise step_stalled(self.step, "repeated_state", f"state={state!r}")
        if len(self.seen) >= self.bound:
            raise step_stalled(self.step, "iteration_bound",
                               f"{len(self.seen)} states visited, bound {self.bound}")
        self.seen.add(state)


def run(cmd, step, input=None, text=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stall_s=None, **popen):
    """`subprocess.run` with no time limit: the process is killed only when it stops reading and writing.

    run([ffmpeg, "-i", path, "-f", "null", "-"], "strict_decode").returncode"""
    stdin = popen.pop("stdin", None)
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE if input is not None else stdin,
                               stdout=stdout, stderr=stderr, text=text, **popen)
    with watch(step, lambda: process_progress(process.pid), process.kill, stall_s,
               f"cmd={cmd[0]}"):
        out, err = process.communicate(input)
    return subprocess.CompletedProcess(cmd, process.returncode, out, err)


def run_checked(cmd, step, stall_s=None):
    """Run `cmd` like `run` and raise when it exits non-zero; return (stdout, stderr, returncode).

    run_checked([mkvmerge, "-o", out, src], "mux")"""
    done = run(cmd, step, stall_s=stall_s)
    if done.returncode != 0:
        raise Exception("This cmd is in error: " + " ".join(cmd) + "\n"
                        + done.stderr.decode("utf-8", "replace") + "\n"
                        + done.stdout.decode("utf-8", "replace")
                        + "\nReturn code: " + str(done.returncode) + "\n")
    return done.stdout, done.stderr, done.returncode
