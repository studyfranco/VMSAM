'''Tests for file_conformity.py -- run: python3 src/test_file_conformity.py (or pytest).

Every fixture is synthetic: numpy writes WAV tracks (seeded noise with a
syllable-like envelope, digital silence where a scenario needs it), ffmpeg
muxes them with a small grey video into an mkv under a temporary directory,
and the directory is removed at the end. Needs ffmpeg, ffprobe and fpcalc.'''

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import file_conformity as fc  # noqa: E402

SR = 16000
TOOLS = all(shutil.which(t) for t in ("ffmpeg", "ffprobe", "fpcalc"))


def _content(seconds, seed):
    '''Band-limited noise under a random 0.25 s envelope: loud enough to be
    content everywhere, structured enough for chromaprint.'''
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = rng.standard_normal(n).astype(np.float32)
    x = np.convolve(x, np.ones(4, np.float32) / 4, mode="same")
    env = np.repeat(rng.uniform(0.05, 0.4, size=n // (SR // 4) + 1), SR // 4)[:n]
    return (x * env).astype(np.float32)


def _track(segments, seed):
    '''segments: list of (seconds, 'c'|'s'); content drawn from ONE seeded
    stream so that two calls with the same seed give the same audio.'''
    full = _content(sum(s for s, _ in segments), seed)
    out, pos = [], 0
    for sec, kind in segments:
        n = int(sec * SR)
        out.append(full[pos:pos + n] if kind == "c" else np.zeros(n, np.float32))
        pos += n
    return np.concatenate(out)


def _write_wav(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def _mux(tmp, name, video_s, tracks):
    '''tracks: list of (samples, lang). Returns the mkv path.'''
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
           "-i", f"color=c=gray:s=64x48:r=5:d={video_s}"]
    for k, (x, _) in enumerate(tracks):
        wav = os.path.join(tmp, f"{name}_{k}.wav")
        _write_wav(wav, x)
        cmd += ["-i", wav]
    cmd += ["-map", "0:v"]
    for k in range(len(tracks)):
        cmd += ["-map", f"{k + 1}:a"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-g", "25", "-c:a", "flac"]
    for k, (_, lang) in enumerate(tracks):
        cmd += [f"-metadata:s:a:{k}", f"language={lang}"]
    out = os.path.join(tmp, f"{name}.mkv")
    subprocess.run(cmd + [out], check=True)
    for k in range(len(tracks)):
        os.unlink(os.path.join(tmp, f"{name}_{k}.wav"))
    return out


def _check(path, **kw):
    kw.setdefault("windows", 2)
    return fc.check_file(path, **kw)


@unittest.skipUnless(TOOLS, "ffmpeg / ffprobe / fpcalc not installed")
class SyntheticFiles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="test_conformity_")
        t = cls.tmp
        cls.healthy = _mux(t, "healthy", 400, [(_track([(400, "c")], 1), "eng"),
                                               (_track([(400, "c")], 2), "jpn")])
        # 691 shape: content, a long digital silence, content again after the video end
        cls.after_end = _mux(t, "after_end", 400, [
            (_track([(100, "c"), (320, "s"), (100, "c")], 3), "jpn"),
            (_track([(400, "c")], 4), "eng")])
        # id 110 shape: the same audio twice, tagged eng and jpn
        same = _track([(400, "c")], 5)
        cls.tag_conflict = _mux(t, "tag_conflict", 400, [(same, "eng"), (same.copy(), "jpn")])
        # 349 master shape: every audio track stops 400 s before the video
        full_a, full_b = _track([(800, "c")], 6), _track([(800, "c")], 7)
        cut = lambda x: np.concatenate([x[:400 * SR], np.zeros(400 * SR, np.float32)])
        cls.truncated = _mux(t, "truncated", 800, [(cut(full_a), "jpn"), (cut(full_b), "eng")])
        cls.untruncated = _mux(t, "untruncated", 800, [(full_a, "jpn"), (full_b, "eng")])
        # Ragnarok shape + 691 spa/por shape: a same-language pair 120 ms apart,
        # and a track that stops at 20 % of its siblings
        base = _track([(1000, "c")], 8)
        late = np.concatenate([np.zeros(int(0.120 * SR), np.float32), base])[:len(base)]
        cls.desync = _mux(t, "desync", 1000, [
            (base, "fre"), (late, "fre"), (_track([(1000, "c")], 9), "eng"),
            (_track([(200, "c"), (800, "s")], 10), "spa")])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_healthy_file_conforms(self):
        r = _check(self.healthy)
        self.assertEqual(r.names("error"), [])
        self.assertEqual(r.names("warning"), [])
        self.assertEqual(r.exit_code, 0)
        integ = [c for c in r.checks if c.name == "stream_integrity"][0]
        self.assertEqual(integ.numbers["processes_failed"], 0)
        self.assertEqual(integ.numbers["processes_total"], 4)      # 2 windows x (video, audio)
        self.assertEqual(integ.numbers["seed"], fc.path_seed(self.healthy))

    def test_content_after_video_end_and_islands(self):
        r = _check(self.after_end)
        errs = [c for c in r.checks if c.name == "content_after_video_end"]
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0].severity, "error")
        self.assertAlmostEqual(errs[0].numbers["content_last_s"], 520, delta=2)
        self.assertAlmostEqual(errs[0].numbers["video_end_s"], 400, delta=1)
        isl = [c for c in r.checks if c.name == "content_islands"][0]
        self.assertEqual(len(isl.numbers["islands"]), 2)
        self.assertIn("long_interior_silence", r.names("warning"))
        self.assertEqual(r.exit_code, 2)

    def test_tag_conflict(self):
        r = _check(self.tag_conflict)
        c = [c for c in r.checks if c.name == "audio_tag_conflict"]
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0].severity, "error")
        self.assertEqual(sorted(c[0].numbers["langs"]), ["eng", "jpn"])
        self.assertGreaterEqual(min(c[0].numbers["similarity"]), 0.99)
        # the different-content pair of the healthy file is NOT a conflict
        self.assertNotIn("audio_tag_conflict", _check(self.healthy).names())

    def test_truncated_tail_without_and_with_reference(self):
        r = _check(self.truncated)
        e = [c for c in r.checks if c.name == "content_ends_early" and c.severity == "error"]
        self.assertEqual(len(e), 1)
        self.assertAlmostEqual(e[0].numbers["early_s"], 400, delta=2)
        r = _check(self.truncated, reference=self.untruncated)
        cut = [c for c in r.checks if c.name == "master_cut_short"]
        self.assertEqual(len(cut), 1, r.summary_fr())
        self.assertAlmostEqual(cut[0].numbers["diff_s"], 400, delta=3)
        # the other way round the SHORT side is the reference: info, not error
        r = _check(self.untruncated, reference=self.truncated)
        self.assertIn("candidate_tail_missing", r.names("info"))
        self.assertNotIn("master_cut_short", r.names())
        self.assertEqual(r.exit_code, 0)

    def test_window_after_the_audio_end_runs_video_only(self):
        # the audio TRACK (its packets) stops at 300 s of an 800 s video
        path = _mux(self.tmp, "short_audio", 800, [(_track([(300, "c")], 12), "eng")])
        r = _check(path)
        integ = [c for c in r.checks if c.name == "stream_integrity"][0]
        self.assertEqual(len([st for st in integ.numbers["starts"] if st > 300]), 1)
        self.assertEqual(integ.numbers["processes_total"], 3)
        self.assertEqual(integ.severity, "info")
        self.assertEqual([c.severity for c in r.checks if c.name == "content_ends_early"], ["error"])
        os.unlink(path)

    def test_content_after_end_is_not_a_tail_cut(self):
        r = _check(self.after_end, reference=self.healthy)
        self.assertNotIn("master_cut_short", r.names())
        self.assertNotIn("candidate_tail_missing", r.names())

    def test_intertrack_desync_and_short_sibling(self):
        r = _check(self.desync)
        d = [c for c in r.checks if c.name == "intertrack_desync"]
        self.assertEqual(len(d), 1, r.summary_fr())
        self.assertEqual(d[0].severity, "warning")
        self.assertAlmostEqual(abs(d[0].numbers["lag_ms"]), 120, delta=5)
        s = [c for c in r.checks if c.name == "track_much_shorter_than_siblings"]
        self.assertEqual(len(s), 1)
        self.assertIn("spa", s[0].numbers["track"])
        self.assertIn("content_ends_early", r.names("warning"))

    def test_corrupt_stream_is_an_integrity_error(self):
        bad = os.path.join(self.tmp, "corrupt.mkv")
        shutil.copy(self.healthy, bad)
        size = os.path.getsize(bad)
        rng = np.random.default_rng(11)
        with open(bad, "r+b") as f:
            for frac in (0.2, 0.4, 0.6, 0.8):
                f.seek(int(size * frac))
                f.write(rng.integers(0, 256, 4096, dtype=np.uint8).tobytes())
        r = _check(bad, mode="full")
        integ = [c for c in r.checks if c.name == "stream_integrity"][0]
        self.assertEqual(integ.severity, "error", r.summary_fr())
        self.assertNotEqual(integ.numbers["rc"], 0)
        os.unlink(bad)

    def test_cli_smoke(self):
        p = subprocess.run([sys.executable, "-m", "file_conformity", self.after_end, "--json",
                            "--windows", "2"], cwd=HERE, capture_output=True, text=True)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("NON CONFORME", p.stdout)
        self.assertIn("content_after_video_end", p.stdout)
        js = json.loads(p.stdout[p.stdout.index("\n{") + 1:])
        self.assertEqual(js["exit_code"], 2)
        p = subprocess.run([sys.executable, "-m", "file_conformity", self.healthy,
                            "--windows", "2"], cwd=HERE, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("CONFORME", p.stdout)


class PureFunctions(unittest.TestCase):

    def test_extent_islands(self):
        ms = np.zeros(1000 * fc.BLOCKS_PER_S, np.float32)
        ms[:100 * fc.BLOCKS_PER_S] = 1e-2
        ms[200 * fc.BLOCKS_PER_S:300 * fc.BLOCKS_PER_S] = 1e-2     # gap 100 s -> 2 islands
        ms[330 * fc.BLOCKS_PER_S:340 * fc.BLOCKS_PER_S] = 1e-2     # gap 30 s -> same island
        e = fc._extent(ms)
        self.assertEqual(e["first"], 0.0)
        self.assertEqual(e["last"], 340.0)
        self.assertEqual(e["islands"], [[0.0, 100.0], [200.0, 340.0]])
        self.assertEqual(e["max_interior_gap_s"], 100.0)

    def test_extent_isolated_clicks_are_not_islands(self):
        # 691 jpn shape: content, 1 s blips inside a long digital silence, content again
        ms = np.zeros(3000 * fc.BLOCKS_PER_S, np.float32)
        ms[:400 * fc.BLOCKS_PER_S] = 1e-2
        for t in (900, 1500):
            ms[t * fc.BLOCKS_PER_S:(t + 1) * fc.BLOCKS_PER_S] = 1e-2
        ms[2500 * fc.BLOCKS_PER_S:2800 * fc.BLOCKS_PER_S] = 1e-2
        e = fc._extent(ms)
        self.assertEqual(e["islands"], [[0.0, 400.0], [2500.0, 2800.0]])
        self.assertEqual(e["clicks"], [[900.0, 901.0], [1500.0, 1501.0]])
        self.assertEqual(e["max_interior_gap_s"], 2100.0)
        self.assertEqual(e["content_s"], 700.0)

    def test_integrity_windows_seeded_and_stratified(self):
        a = fc.integrity_windows(3600, 10, 7)
        self.assertEqual(a, fc.integrity_windows(3600, 10, 7))
        self.assertNotEqual(a, fc.integrity_windows(3600, 10, 8))
        for i, s in enumerate(a):
            self.assertGreaterEqual(s, i * 360)
            self.assertLessEqual(s + fc.INTEGRITY_WINDOW_S, (i + 1) * 360 + 1e-6)
        self.assertIsNone(fc.integrity_windows(600, 10, 7))   # too short -> full decode
        self.assertEqual(fc.path_seed("/a/b.mkv"), fc.path_seed("/a/b.mkv"))
        self.assertNotEqual(fc.path_seed("/a/b.mkv"), fc.path_seed("/a/c.mkv"))

    def test_xcorr_sign(self):
        x = np.random.default_rng(1).standard_normal(SR * 10).astype(np.float32)
        late = np.concatenate([np.zeros(800, np.float32), x])[:len(x)]   # 50 ms late
        lag, corr = fc._xcorr(late, x, SR, 1.0)
        self.assertAlmostEqual(lag, 50.0, delta=0.5)
        self.assertGreater(corr, 0.95)

    def test_fp_similarity_shift(self):
        rng = np.random.default_rng(2)
        a = rng.integers(0, 2 ** 32, 500, dtype=np.uint64).astype(np.uint32)
        b = np.concatenate([rng.integers(0, 2 ** 32, 7, dtype=np.uint64).astype(np.uint32), a])
        s, sh = fc._fp_similarity(a, b, 12, 200)
        self.assertEqual((round(s, 6), sh), (1.0, 7))
        s2, _ = fc._fp_similarity(a, rng.integers(0, 2 ** 32, 500, dtype=np.uint64)
                                  .astype(np.uint32), 12, 200)
        self.assertLess(s2, 0.6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
