"""Pins the dev-mode start/end lines that `tools` puts around every external command launched
through `launch_cmdExt` and `launch_cmdExt_with_timeout_reload` (owner, 2026-09-25)."""
import contextlib
import inspect
import io
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tools  # noqa: E402

START_RE = re.compile(r"^devcmd: utc=\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z START #(\d+) (\S+) \[")
END_RE = re.compile(r"^devcmd: utc=\S+Z END #(\d+) (\S+) \[.*\] seconds=\d+\.\d{3} exit=(-?\d+)$")


class DevTimedTest(unittest.TestCase):
    def setUp(self):
        self._dev = tools.dev

    def tearDown(self):
        tools.dev = self._dev
        tools.dev_phase(None)

    def _capture(self, fn, *args, **kwargs):
        buf = io.StringIO()
        exc = None
        with contextlib.redirect_stderr(buf):
            try:
                result = fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                result, exc = None, e
        return buf.getvalue().splitlines(), result, exc

    def test_dev_on_success_and_failure(self):
        tools.dev = True
        for fn in (tools.launch_cmdExt, tools.launch_cmdExt_with_timeout_reload):
            lines, result, exc = self._capture(fn, ["true"])
            self.assertIsNone(exc)
            self.assertEqual(result[2], 0)
            self.assertEqual(len(lines), 2, lines)
            s, e = START_RE.match(lines[0]), END_RE.match(lines[1])
            self.assertTrue(s and e, lines)
            self.assertEqual(s.group(1), e.group(1))
            self.assertEqual((s.group(2), e.group(2), e.group(3)), ("true", "true", "0"))

            lines, result, exc = self._capture(fn, ["sh", "-c", "exit 3"])
            self.assertIsNotNone(exc)
            self.assertIn("Return code: 3", str(exc))
            self.assertEqual(len(lines), 2, lines)
            s, e = START_RE.match(lines[0]), END_RE.match(lines[1])
            self.assertTrue(s and e, lines)
            self.assertEqual(e.group(2), "sh")
            self.assertEqual(e.group(3), "3")

    def test_counter_increases(self):
        tools.dev = True
        a, _, _ = self._capture(tools.launch_cmdExt, ["true"])
        b, _, _ = self._capture(tools.launch_cmdExt, ["true"])
        self.assertGreater(int(START_RE.match(b[0]).group(1)), int(START_RE.match(a[0]).group(1)))

    def test_signature_is_basename_only_and_truncated(self):
        long_name = "x" * 100 + ".mkv"
        sig = tools._dev_cmd_signature(["ffmpeg", "-y", "-i", "/secret/dir/" + long_name, "out.wav"])
        self.assertNotIn("/secret", sig)
        self.assertLessEqual(len(sig), tools.DEV_CMD_SIGNATURE_MAX)
        self.assertEqual(tools._dev_cmd_signature(["fpcalc", "-raw", "/a/b/c.flac"]), "c.flac")

    def test_phase_hook(self):
        tools.dev = True
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            tools.dev_phase("keep_best_audio")
            tools.launch_cmdExt(["true"])
        self.assertIn("phase=keep_best_audio", buf.getvalue().splitlines()[1])

    def test_dev_off_is_silent(self):
        tools.dev = False
        for fn in (tools.launch_cmdExt, tools.launch_cmdExt_with_timeout_reload):
            lines, result, _ = self._capture(fn, ["true"])
            self.assertEqual(lines, [])
            self.assertEqual(result[2], 0)
            lines, _, exc = self._capture(fn, ["sh", "-c", "exit 3"])
            self.assertEqual(lines, [])
            self.assertIn("Return code: 3", str(exc))

    def test_names_and_signatures_kept(self):
        self.assertEqual(tools.launch_cmdExt.__name__, "launch_cmdExt")
        self.assertEqual(tools.launch_cmdExt_with_timeout_reload.__name__,
                         "launch_cmdExt_with_timeout_reload")
        self.assertEqual(str(inspect.signature(tools.launch_cmdExt)), "(cmd)")
        self.assertEqual(str(inspect.signature(tools.launch_cmdExt_with_timeout_reload)),
                         "(cmd, max_restart=1, timeout=120)")


if __name__ == "__main__":
    unittest.main()
