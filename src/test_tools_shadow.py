'''Tests for `tools.launch_cmdExt_with_timeout_reload_shadow` (owner, Addendum 32.11) -- run:
python3 src/test_tools_shadow.py (or pytest). A fake long command (it appends one line to a
counter file, then sleeps 20 s) hits a 1 s timeout with max_restart=1:
  * the ORIGINAL (tools.py:150-170, frozen) launches it THREE times and leaves one of them
    running as a child nobody waits on -- the defect of tools.py:166 (LAB_id108 §4);
  * the SHADOW launches it TWICE (the first run + one restart), then raises, and leaves no
    child at all: no zombie, no orphan (read from /proc), and no leaked descriptor.'''

import os
import signal
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tools  # noqa: E402

TIMEOUT_S = 1
SLEEP_S = 20


def _children():
    """(pid, state) of every process whose parent is this one, read from /proc."""
    me = os.getpid()
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as stat:
                fields = stat.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[1]) == me:
            found.append((int(name), fields[0]))
    return found


def _open_fds():
    return len(os.listdir("/proc/self/fd"))


def _run(function):
    """Launch count, raised message, children left behind and descriptor delta."""
    with tempfile.TemporaryDirectory() as tmp:
        counter = os.path.join(tmp, "launches")
        cmd = ["sh", "-c", f"echo x >> '{counter}'; exec sleep {SLEEP_S}"]
        fds_before = _open_fds()
        raised = None
        began = time.monotonic()
        try:
            function(cmd, 1, TIMEOUT_S)
        except Exception as error:
            raised = str(error)
        elapsed = time.monotonic() - began
        time.sleep(0.3)  # let an orphan's `echo` land before counting
        with open(counter) as lines:
            launches = len(lines.readlines())
        left = _children()
        fds_delta = _open_fds() - fds_before
        for pid, _ in left:  # clean up what the original leaves behind
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass
    return launches, raised, left, fds_delta, elapsed


def test_original_leaves_an_orphan():
    assert not _children(), "the test must start with no child"
    launches, raised, left, _, _ = _run(tools.launch_cmdExt_with_timeout_reload)
    assert raised and "will not be restarted" in raised, raised
    assert launches == 3, f"the original is expected to launch 3 times, launched {launches}"
    assert len(left) == 1, f"the original is expected to leave one child, left {left}"


def test_shadow_restarts_once_and_leaves_nothing():
    assert not _children(), "the test must start with no child"
    launches, raised, left, fds_delta, elapsed = _run(tools.launch_cmdExt_with_timeout_reload_shadow)
    assert raised and "will not be restarted" in raised, raised
    assert launches == 2, f"one run + one restart expected, launched {launches}"
    assert left == [], f"no zombie and no orphan expected, /proc shows {left}"
    assert fds_delta <= 0, f"{fds_delta} descriptor(s) leaked"
    assert elapsed < 2 * TIMEOUT_S + 5, f"took {elapsed:.1f} s"


def test_shadow_success_path_unchanged():
    stdout, stderror, exit_code = tools.launch_cmdExt_with_timeout_reload_shadow(["sh", "-c", "printf ok"], 1, 10)
    assert (stdout, exit_code) == (b"ok", 0)
    try:
        tools.launch_cmdExt_with_timeout_reload_shadow(["sh", "-c", "exit 3"], 1, 10)
    except Exception as error:
        assert "Return code: 3" in str(error)
    else:
        raise AssertionError("a failing command must raise, as in the original")
    assert not _children()


if __name__ == "__main__":
    tools.dev = False
    for test in (test_original_leaves_an_orphan, test_shadow_restarts_once_and_leaves_nothing,
                 test_shadow_success_path_unchanged):
        test()
        print(f"PASS {test.__name__}")
