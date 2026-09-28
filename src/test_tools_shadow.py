'''Tests for `tools.launch_cmdExt_with_timeout_reload_shadow` (owner, Addendum 32.11) -- run:
python3 src/test_tools_shadow.py (or pytest). A fake long command (it appends one line to a
counter file, then sleeps 20 s) hits a 1 s timeout with max_restart=1:
  * the ORIGINAL (tools.py:150-170, frozen) launches it THREE times and leaves one of them
    running as a child nobody waits on -- the defect of tools.py:166 (LAB_id108 §4);
  * the SHADOW launches it TWICE (the first run + one restart), then raises, and leaves no
    child at all: no zombie, no orphan (read from /proc), and no leaked descriptor.
The diagnostics (owner, 2026-09-28, "mettre les apps en mode verbose"): a fake command that
prints progress on stderr then sleeps past the timeout must be logged BEFORE the kill with
its last stderr lines, its /proc state and wchan and the file it names, then again after the
kill; a non-zero exit logs its tail with the return code and still raises; ffmpeg on a tiny
synthetic file with the injected `-loglevel verbose` still exits 0.'''

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


def _entries(event):
    return [entry for entry in tools.logs if f"tools: child {event} " in entry]


def test_timeout_logs_tail_wchan_and_file_before_the_kill():
    tools.logs = []
    with tempfile.TemporaryDirectory() as tmp:
        named = os.path.join(tmp, "input.bin")
        with open(named, "wb") as data:
            data.write(b"x" * 1234)
        script = ("for i in 1 2 3 4 5; do echo \"progress $i reading '%s'\" >&2; done; "
                  "echo partial-stdout; exec sleep %d" % (named, SLEEP_S))
        try:
            tools.launch_cmdExt_with_timeout_reload_shadow(["sh", "-c", script], 0, TIMEOUT_S)
        except Exception as error:
            assert "will not be restarted" in str(error)
        else:
            raise AssertionError("the timeout must raise")
        before, after = _entries("timeout_before_kill"), _entries("timeout_after_kill")
        assert len(before) == 1 and len(after) == 1, tools.logs
        head = before[0].splitlines()[0]
        assert "state=S" in head and "wchan=" in head, head
        wchan = head.split("wchan=")[1].split()[0]
        assert wchan not in ("gone", "unreadable", ""), head
        assert "progress 5 reading" in before[0], before[0]
        assert "stdout_bytes=15" in head, head  # "partial-stdout\n", read without blocking
        assert f"file '{named}' size=1234 mtime=" in before[0], before[0]
        assert "argv=['sh', '-c'," in before[0]
        assert "returncode=-9" in after[0].splitlines()[0], after[0]
        assert "progress 5 reading" in after[0]
    assert not _children()


def test_nonzero_exit_logs_tail_and_still_raises():
    tools.logs = []
    try:
        tools.launch_cmdExt_with_timeout_reload_shadow(
            ["sh", "-c", "echo 'decoder said no' >&2; exit 7"], 1, 10)
    except Exception as error:
        assert "decoder said no" in str(error) and "Return code: 7" in str(error)
    else:
        raise AssertionError("a failing command must raise, as in the original")
    entry = _entries("nonzero_exit")
    assert len(entry) == 1 and "returncode=7" in entry[0] and "decoder said no" in entry[0], tools.logs


def test_verbose_flags():
    v = tools.verbose_cmd_shadow
    assert v(["/usr/bin/ffmpeg", "-v", "error", "-i", "a"]) == ["/usr/bin/ffmpeg", "-loglevel", "verbose", "-hide_banner", "-i", "a"]
    assert v(["ffprobe", "-hide_banner", "-loglevel", "quiet", "x"]) == ["ffprobe", "-loglevel", "verbose", "-hide_banner", "x"]
    assert v(["mkvmerge", "-o", "o.mkv", "a.mkv"]) == ["mkvmerge", "-v", "-o", "o.mkv", "a.mkv"]
    assert v(["mkvmerge", "-i", "-F", "json", "a.mkv"]) == ["mkvmerge", "-i", "-F", "json", "a.mkv"]
    assert v(["mkvextract", "a.mkv", "tracks", "0:x"]) == ["mkvextract", "--verbose", "a.mkv", "tracks", "0:x"]
    assert v(["mkvpropedit", "--verbose", "a.mkv"]) == ["mkvpropedit", "--verbose", "a.mkv"]
    assert v(["sh", "-c", "true"]) == ["sh", "-c", "true"]


def test_ffmpeg_verbose_still_exits_zero():
    with tempfile.TemporaryDirectory() as tmp:
        tiny = os.path.join(tmp, "tiny.mka")
        tools.launch_cmdExt(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                             "-c:a", "flac", tiny])
        tools.logs = []
        stdout, stderror, exit_code = tools.launch_cmdExt_with_timeout_reload_shadow(
            ["ffmpeg", "-v", "error", "-i", tiny, "-map", "0:0", "-c", "copy", "-f", "md5", "-"], 1, 60)
        assert exit_code == 0 and stdout.startswith(b"MD5="), stdout
        assert len(stderror) > 0, "the injected -loglevel verbose must make ffmpeg explain itself"
        assert tools.logs == []


if __name__ == "__main__":
    tools.dev = False
    for test in (test_original_leaves_an_orphan, test_shadow_restarts_once_and_leaves_nothing,
                 test_shadow_success_path_unchanged, test_timeout_logs_tail_wchan_and_file_before_the_kill,
                 test_nonzero_exit_logs_tail_and_still_raises, test_verbose_flags,
                 test_ffmpeg_verbose_still_exits_zero):
        test()
        print(f"PASS {test.__name__}")
