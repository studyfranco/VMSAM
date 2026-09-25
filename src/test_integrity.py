'''Tests for integrity.py -- run: python3 src/test_integrity.py [--quick | --real-only]
[--only NAME]. pytest is not needed (it also runs under pytest).

--quick      synthetic fixtures + the cheap cuts of the Chainsaw files (seconds).
(default)    also the REAL positives and negatives, sequentially, -threads 2, with the cost
             of every call printed: the Chainsaw VARYG AMZN E-AC-3 tracks (corrupt frame at
             4 007.968 s) and the CR files, the 691 master against a healthy jpn, the id 349
             master (four tracks cut at 3 404.6 s) against its candidate, and the 19 healthy
             masters of the c8706d01 entry check (every audio track strictly decoded, every
             other audio track's silences against the first one, the video sampled).
Every source file is READ-ONLY: fixtures and cuts go to a temporary directory, removed.'''

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import integrity as I  # noqa: E402

SR = 8000
TOOLS = all(shutil.which(t) for t in ("ffmpeg", "ffprobe"))
CSM = "/config/example/ChainSaw_Movie"
CSM_VARYG_DUAL = f"{CSM}/Chainsaw.Man.The.Movie.Reze.Arc.2025.1080p.AMZN.WEB-DL.DUAL.DDP5.1.H.264-VARYG.mkv"
CSM_VARYG_MULTI = f"{CSM}/Chainsaw.Man.The.Movie.Reze.Arc.2025.1080p.AMZN.WEB-DL.MULTi.DDP5.1.Atmos.H.264-VARYG.mkv"
CSM_VARYG_REPACK = f"{CSM}/Chainsaw.Man.The.Movie.Reze.Arc.2025.REPACK.1080p.AMZN.WEB-DL.MULTi.DDP5.1.Atmos.H.264-VARYG.mkv"
CSM_TOONS_AMZN = f"{CSM}/Chainsaw.Man.The.Movie.Reze.Arc.2025.1080p.AMZN.WEB-DL.DUAL.DDP5.1.H.264.MSubs-ToonsHub.mkv"
CSM_TOONS_CR = f"{CSM}/Chainsaw.Man.The.Movie.Reze.Arc.2025.1080p.CR.WEB-DL.DUAL.AAC2.0.H.264.MSubs-ToonsHub.mkv"
CSM_ERAI = f"{CSM}/[Erai-raws] Chainsaw Man - Reze-hen - Movie [1080p CR WEB-DL AVC AAC][MultiSub][EA0E4F22].mkv"
M691 = ("/srv/Plex/Anime/Chainsaw Man {tvdb-397934} [tvdb-397934] [tvdbid-397934]/Specials/"
        "Chainsaw Man - S00E01 - 1080p.CR.WEB-DL.MULTi.AAC2.0.H.264-VARYG.mkv")
M349 = ("/srv/Plex/Anime Fr En/Seishun Buta Yarou wa Bunny Girl Senpai no Yume wo Minai [tvdb-345596] "
        "[tvdbid-345596]/Specials/[Moozzi2] Seishun Buta Yarou wa Ransel Girl no Yume o Minai - S00E08 "
        "- (BD 1920x1080 HEVC-YUV444P10 4Audio).mkv")
C349 = ("/srv/Plex/tmp_sync/rss/merge_error/srv/Plex/Anime Fr En/Seishun Buta Yarou wa Bunny Girl "
        "Senpai no Yume wo Minai [tvdb-345596] [tvdbid-345596]/Specials/Rascal.Does.Not.Dream.of.a."
        "Knapsack.Kid.S00E08.1080p.LFTL.WEB-DL.JPN.AAC2.0.H.264.MSubs-ToonsHub.mkv")
# the 19 healthy masters of the c8706d01 entry check (a21/cert/masters_conf.tsv, h_* lines)
HEALTHY = [
    ("e202", "/home/vmsam/src/VMSAM_CORPUS/video-pairs/errid-202/master.mkv"),
    ("e232", "/home/vmsam/src/VMSAM_CORPUS/video-pairs/errid-232/master.mkv"),
    ("e70", "/home/vmsam/src/VMSAM_CORPUS/video-pairs/errid-70/master.mkv"),
    ("fo02", "/srv/Plex/Serie Fr En/Fallout {tvdb-416744} [tvdb-416744] [tvdbid-416744]/Season 1/"
             "Fallout - S01E02 - MULTi.VFi.1080p.BluRay.REMUX.AVC.TrueHD.7.1-ROMKENT.mkv"),
    ("mh134", "/srv/Plex/Anime Fr En/Mai-HiME [tvdb-83020] [tvdbid-83020]/Season 1/"
              "Mai-HiME - S01E12 - (1460x1078 HEVC 10bit FLAC).mkv"),
    ("bl156", "/srv/Plex/Anime/Bleach [tvdb-74796] [tvdbid-74796]/Season 17/"
              "[ReinForce] Bleach - S17E25 - (BDRip 1920x1080 x264 FLAC).mkv"),
    ("uu171", "/srv/Plex/Anime/Undead Unluck {tvdb-423030} [tvdb-423030] [tvdbid-423030]/Season 1/"
              "Undead Unluck - S01E09 - 1080p.DSNP.WEB-DL.AAC2.0.H.264-VARYG.mkv"),
    ("e121", "/home/vmsam/src/VMSAM_CORPUS/video-pairs/errid-121/master.mkv"),
    ("tougen07", "/srv/Plex/Anime/Tougen Anki [tvdb-449843] [tvdbid-449843]/Season 1/"
                 "TOUGEN.ANKI - S01E07 - 1080p.CR.WEB-DL.DUAL.AAC2.0.H.264-VARYG.mkv"),
    ("e100", "/home/vmsam/src/VMSAM_CORPUS/video-pairs/errid-100/master.mkv"),
    ("e213", "/home/vmsam/src/VMSAM_CORPUS/video-pairs/errid-213/master.mkv"),
    ("e222", "/home/vmsam/src/VMSAM_CORPUS/video-pairs/errid-222/master.mkv"),
    ("e352", "/home/vmsam/src/VMSAM_CORPUS/video-pairs/errid-352/master.mkv"),
    ("id101", "/srv/Plex/Anime/Lazarus [tvdb-437408] [tvdbid-437408]/Season 1/"
              "Lazarus - S01E01 - 1080p.AMZN.WEB-DL.MULTi.DDP2.0.H.264-VARYG.mkv"),
    ("id57", "/srv/Plex/Anime/Lazarus [tvdb-437408] [tvdbid-437408]/Season 1/"
             "Lazarus - S01E05 - 1080p.AMZN.WEB-DL.MULTi.DDP2.0.H.264-VARYG.mkv"),
    ("id293", "/srv/Plex/Anime Fr En/Isekai Suicide Squad {tvdb-436846} [tvdb-436846] [tvdbid-436846]/"
              "Season 1/Isekai Suicide Squad - S01E04 - 1080p.AMZN.WEB-DL.DDP2.0.H.264-VARYG.mkv"),
    ("id300", "/srv/Plex/Anime Fr En/Isekai Suicide Squad {tvdb-436846} [tvdb-436846] [tvdbid-436846]/"
              "Season 1/Isekai Suicide Squad - S01E08 - 1080p.AMZN.WEB-DL.DDP2.0.H.264-VARYG.mkv"),
    ("fo01", "/srv/Plex/Serie Fr En/Fallout {tvdb-416744} [tvdb-416744] [tvdbid-416744]/Season 1/"
             "Fallout - S01E01 - MULTi.VFi.1080p.BluRay.REMUX.AVC.TrueHD.7.1-ROMKENT.mkv"),
    ("fo03", "/srv/Plex/Serie Fr En/Fallout {tvdb-416744} [tvdb-416744] [tvdbid-416744]/Season 1/"
             "Fallout - S01E03 - MULTi.VFi.1080p.BluRay.REMUX.AVC.TrueHD.7.1-ROMKENT.mkv"),
]

# the healthy masters' measured same-file silence differences (label -> {(a, b, result)})
KNOWN_SILENCE_DIFFERENCES = {"e352": {(1, 2, "B")}}

MODE = "all"
RESULTS = []            # (case, expected, got, cost_s) printed at the end


def note(case, expected, got, cost, ok=None):
    ok = (str(expected) == str(got)) if ok is None else ok
    RESULTS.append((case, expected, got, cost, ok))
    print(f"  [{'ok' if ok else 'MISMATCH'}] {case}: expected {expected} "
          f"got {got} cost {cost:.1f} s", flush=True)


# ---------------------------------------------------------------- synthetic fixtures

def _content(seconds, seed, level=0.3):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = rng.standard_normal(n).astype(np.float32)
    x = np.convolve(x, np.ones(3, np.float32) / 3, mode="same")
    env = np.repeat(rng.uniform(0.3, 1.0, size=n // (SR // 4) + 1), SR // 4)[:n]
    return (x * env * level).astype(np.float32)


def _signal(segments, seed=1):
    '''segments: (seconds, kind): 'c' content (ONE seeded stream: the same seconds of
    content are the same samples in every track built from the same seed), 's' digital
    silence, ('q', dBFS) steady noise at that RMS level, 'k' a 1 s click.'''
    base = _content(sum(s for s, _ in segments) + 1, seed)
    out, pos = [], 0
    for sec, kind in segments:
        n = int(sec * SR)
        if kind == "c":
            out.append(base[pos:pos + n])
        elif kind == "s":
            out.append(np.zeros(n, np.float32))
        elif kind == "k":
            out.append(np.full(n, 0.2, np.float32) * np.sign(np.sin(np.arange(n) / 3.0)))
        else:
            rms = 10 ** (kind[1] / 20.0)
            out.append((np.random.default_rng(seed + 7).standard_normal(n) * rms).astype(np.float32))
        pos += n
    return np.concatenate(out)


def _write_wav(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def _mux(tmp, name, tracks, video_s=None):
    '''tracks: list of float arrays -> an mkv (flac audio), with a tiny video when video_s.'''
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    maps = []
    k = 0
    if video_s:
        cmd += ["-f", "lavfi", "-i", f"color=c=gray:s=16x16:r=2:d={video_s}"]
        maps += ["-map", "0:v"]
        k = 1
    for i, x in enumerate(tracks):
        w = os.path.join(tmp, f"{name}_{i}.wav")
        _write_wav(w, x)
        cmd += ["-i", w]
        maps += ["-map", f"{k + i}:a"]
    out = os.path.join(tmp, f"{name}.mkv")
    cmd += maps + (["-c:v", "mpeg4"] if video_s else []) + ["-c:a", "flac", out]
    subprocess.run(cmd, check=True)
    return out


@unittest.skipUnless(TOOLS, "ffmpeg/ffprobe missing")
class Synthetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="test_integrity_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def agree(self, case, a, b, delay_ms, expected, **kw):
        t = time.time()
        rep = I.silence_report(a, 0, b, 0, delay_ms, kind="audio_pos", **kw)
        got = (rep["agree"], rep["result"])
        note(case, expected, got, time.time() - t)
        self.assertEqual(got, expected, json.dumps({k: rep[k] for k in ("a_only", "b_only")},
                                                   default=str))
        self.assertEqual(I.silences_agree(a, 0, b, 0, delay_ms, kind="audio_pos", **kw), expected)
        return rep

    def test_agree_a_only_b_only_both(self):
        a = _mux(self.tmp, "int_a", [_signal([(200, "c"), (30, "s"), (370, "c")])])
        b = _mux(self.tmp, "int_b", [_signal([(600, "c")])])
        rep = self.agree("silence 30 s in A only", a, b, 0, (False, "A"))
        self.assertAlmostEqual(rep["a_only"][0]["start_s"], 200.0, delta=0.5)
        self.assertTrue(rep["a_only"][0]["digital"])
        self.agree("silence 30 s in B only", b, a, 0, (False, "B"))
        c = _mux(self.tmp, "int_c", [_signal([(200, "c"), (30, "s"), (370, "c")])])
        self.agree("the same silence in both", a, c, 0, (True, None))
        d = _mux(self.tmp, "int_d", [_signal([(400, "c"), (60, "s"), (140, "c")])])
        rep = self.agree("A silent at 200 s, B silent at 400 s", a, d, 0, (False, "AB"))
        self.assertTrue(rep["a_only"] and rep["b_only"])
        e = _mux(self.tmp, "int_e", [_signal([(200, "c"), (4, "s"), (396, "c")])])
        self.agree("silence 4 s < 5 s", e, b, 0, (True, None))

    def test_track_that_stops_plays_silence_to_its_video_end(self):
        # 349's shape: A's packets stop at 300 s, its video runs to 800 s, B plays to 800 s
        a = _mux(self.tmp, "end_a", [_signal([(300, "c")])], video_s=800)
        b = _mux(self.tmp, "end_b", [_signal([(800, "c")])], video_s=800)
        rep = self.agree("A's packets stop at 300 s of an 800 s video", a, b, 0, (False, "A"))
        self.assertAlmostEqual(rep["a_only"][0]["start_s"], 300.0, delta=0.5)
        # both stop at the same instant (credit cards on a silent video): same silence
        c = _mux(self.tmp, "end_c", [_signal([(300, "c")])], video_s=800)
        self.agree("both tracks stop at 300 s of an 800 s video", a, c, 0, (True, None))
        # past B's own file B says nothing: A's longer track is not contradicted there
        d = _mux(self.tmp, "end_d", [_signal([(300, "c")])])      # no video: plays to 300 s
        self.agree("B's file ends at 300 s, A's track silent after it", a, d, 0, (True, None))

    def test_delay_shift(self):
        a = _mux(self.tmp, "dl_a", [_signal([(100, "c"), (10, "s"), (490, "c")], seed=3)])
        b = _mux(self.tmp, "dl_b", [np.concatenate([np.zeros(7 * SR, np.float32),
                                                     _signal([(100, "c"), (10, "s"), (490, "c")], seed=3)])])
        self.agree("same silence, B 7 s late, delay +7000 ms", a, b, 7000, (True, None))
        self.agree("same pair read at delay 0", a, b, 0, (False, "AB"))
        self.agree("same pair read at delay -7000 ms (wrong sign)", a, b, -7000, (False, "AB"))

    def test_quiet_passage_is_the_same_silence(self):
        # A at -62 dBFS (silent by the -60 line), B at -58: within the 10 dB tolerance
        a = _mux(self.tmp, "q_a", [_signal([(200, "c"), (60, ("q", -62.0)), (340, "c")], seed=9)])
        b = _mux(self.tmp, "q_b", [_signal([(200, "c"), (60, ("q", -58.0)), (340, "c")], seed=9)])
        self.agree("quiet passage -62 vs -58 dBFS", a, b, 0, (True, None))

    def test_clicks_inside_a_long_silence(self):
        a = _mux(self.tmp, "ck_a", [_signal([(300, "c"), (200, "s"), (1, "k"), (200, "s"),
                                              (1, "k"), (198, "s")])])
        m = I.silence_map(a, 0, kind="audio_pos")
        self.assertEqual(len(m["clicks"]), 2)
        self.assertAlmostEqual(m["content_last_s"], 300.0, delta=0.5)
        self.assertAlmostEqual(m["last_silence_start_s"], 300.0, delta=0.5)
        note("clicks in a 600 s silence ignored", "content_last=300", f"content_last={m['content_last_s']:.0f}",
             m["cost_s"])

    def test_resolve_stream_kinds(self):
        v = _mux(self.tmp, "rs", [_signal([(30, "c")], seed=1), _signal([(30, "c")], seed=2)],
                 video_s=30)
        ids = [I.resolve_stream(v, 2)[1]["index"], I.resolve_stream(v, "2")[1]["index"],
               I.resolve_stream(v, 1, kind="audio_pos")[1]["index"],
               I.resolve_stream(v, 2, kind="typeorder")[1]["index"],
               I.resolve_stream(v, {"StreamOrder": "2"})[1]["index"]]
        self.assertEqual(ids, [2] * 5)
        with self.assertRaises(ValueError):
            I.resolve_stream(v, 7)
        with self.assertRaises(ValueError):
            I.silence_map(v, 0)            # the video stream is not audio

        class Obj:
            filePath = v
            video = {"StreamOrder": "0", "Duration": "30.000"}
        self.assertEqual(I.video_end_s(Obj()), 30.0)
        t = time.time()
        self.assertTrue(I.track_is_sound(Obj(), 1))
        note("synthetic flac track_is_sound", True, True, time.time() - t)

    def test_conversion_preserved(self):
        o = _mux(self.tmp, "cv_o", [_signal([(300, "c")], seed=11)])
        good = os.path.join(self.tmp, "cv_good.mkv")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-itsoffset", "2", "-i", o,
                        "-map", "0:a", "-c", "copy", good], check=True)
        t = time.time()
        ok, why = I.conversion_preserved(o, 0, good, 0, 2000)
        note("conversion +2000 ms copy", True, ok, time.time() - t)
        self.assertTrue(ok, why)
        ok, why = I.conversion_preserved(o, 0, good, 0, 2000, expected_duration_s=250)
        self.assertFalse(ok)
        self.assertTrue(why.startswith("conversion_duration"), why)
        # a conversion that inserts 20 s of silence at 100 s (the Chainsaw shape, small)
        x = _signal([(300, "c")], seed=11)
        bad_x = np.concatenate([x[:100 * SR], np.zeros(20 * SR, np.float32), x[100 * SR:]])
        bad = _mux(self.tmp, "cv_bad", [bad_x])
        t = time.time()
        ok, why = I.conversion_preserved(o, 0, bad, 0, 0)
        note("conversion inserting 20 s of silence", False, ok, time.time() - t)
        self.assertFalse(ok)
        self.assertIn("conversion_created_silence at 100", why)
        # without the chromaprint (fidelity=False) the silence map is the check itself
        ok, why = I.conversion_preserved(o, 0, bad, 0, 0, expected_duration_s=None, fidelity=False)
        self.assertFalse(ok)
        # a different sound of the same length: fidelity
        other = _mux(self.tmp, "cv_other", [_signal([(300, "c")], seed=12)])
        if shutil.which("fpcalc"):
            ok, why = I.conversion_preserved(o, 0, other, 0, 0)
            self.assertFalse(ok)
            self.assertTrue(why.startswith("conversion_fidelity"), why)


class StderrPatterns(unittest.TestCase):
    # lines measured on the Chainsaw window (legacy second pass, ffmpeg 9.0.2)
    BAD = [
        "[aost#0:0/eac3 @ 0x560a0d8c3180] Non-monotonic DTS; previous: 109019, current: 99980; "
        "changing to 109019. This may result in incorrect timestamps in the output file.",
        "[af#0:0 @ 0x560a0d8c4a40] Reconfiguring filter graph because audio parameters changed to "
        "48000 Hz, stereo, fltp, downmix medatata changed",
        "[eac3 @ 0x560f42c519c0] new coupling strategy must be present in block 0",
        "[aresample @ 0x55] Failed to compensate for timestamp delta of 3076.160000",
        "[eac3 @ 0x560a0d8d19c0] frame CRC mismatch",
    ]
    GOOD = [
        "Input #0, matroska,webm, from 'x.mkv':",
        "  Stream #0:1(jpn): Audio: eac3, 48000 Hz, 5.1(side), fltp, 640 kb/s",
        "size=   24133KiB time=00:01:43.12 bitrate=1917.0kbits/s speed=65.8x",
        "[sost#0:3/ass @ 0x55] Non-monotonic DTS; previous: 5, current: 4; changing to 5.",
    ]

    def test_patterns(self):
        for line in self.BAD:
            ok, lines = I.conversion_stderr_is_clean("frame=1\r" + line + "\n")
            self.assertFalse(ok, line)
            self.assertEqual(lines, [line])
        ok, lines = I.conversion_stderr_is_clean("\n".join(self.GOOD))
        self.assertTrue(ok, lines)
        self.assertEqual(I.conversion_stderr_is_clean("\n".join(self.GOOD), ignore_non_audio=False)[0],
                         False)
        self.assertEqual(I.CONVERSION_SAFE_OPTIONS, ["-xerror", "-reinit_filter", "0"])
        for p in ("Non-monotonic DTS", "Failed to compensate", "Reconfiguring filter graph",
                  "new coupling strategy", "Error while decoding"):
            self.assertIn(p, I.FATAL_CONVERSION_PATTERNS)
        note("stderr patterns (5 fatal, 4 benign)", "ok", "ok", 0.0)


# ---------------------------------------------------------------- the Chainsaw cut (cheap, real)

def _cut(src, index, start, dur, out):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(start), "-i", src,
                    "-t", str(dur), "-map", f"0:{index}", "-c", "copy", out], check=True)
    return out


@unittest.skipUnless(TOOLS and os.path.exists(CSM_VARYG_DUAL), "Chainsaw files absent")
class ChainsawCut(unittest.TestCase):
    '''A 200 s stream-copy cut around 4 007.968 s: the corrupt frame travels with the copy.'''

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="test_integrity_cut_")
        cls.bad = _cut(CSM_VARYG_DUAL, 1, 3900, 200, os.path.join(cls.tmp, "bad.mka"))
        cls.good = _cut(CSM_TOONS_AMZN, 1, 3900, 200, os.path.join(cls.tmp, "good.mka"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _legacy(self, src, out, extra_global=(), extra_input=()):
        '''mergeVideo.py:1709-1735, second pass, delay -1001.001 ms (Chainsaw H1).'''
        cmd = (["ffmpeg", "-nostdin", "-y"] + list(extra_global)
               + ["-err_detect", "crccheck+bitstream+buffer", "-threads", "2", "-vn"]
               + list(extra_input)
               + ["-i", src, "-ss", "1.001001", "-map", "0", "-c", "copy", "-copyts",
                  "-c:a:0", "eac3", "-filter:a:0", "aresample=async=1:first_pts=0",
                  "-b:a:0", "640000", "-strict", "-2", "-t", "198.9", "-max_interleave_delta", "0",
                  out])
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return p.returncode, p.stderr.decode("utf-8", "replace")

    def test_strict_decode_of_the_cut(self):
        t = time.time()
        r = I.track_check(self.bad, 0)
        note("cut VARYG DUAL jpn [3900;4100] track_is_sound", False, r["sound"], time.time() - t)
        self.assertFalse(r["sound"])
        self.assertNotEqual(r["rc"], 0)
        t = time.time()
        self.assertTrue(I.track_is_sound(self.good, 0))
        note("cut ToonsHub AMZN jpn [3900;4100] track_is_sound", True, True, time.time() - t)

    def test_legacy_conversion_and_the_safe_options(self):
        out = os.path.join(self.tmp, "legacy_bad.mkv")
        rc, err = self._legacy(self.bad, out)
        self.assertEqual(rc, 0)                               # the legacy code sees success
        clean, lines = I.conversion_stderr_is_clean(err)
        self.assertFalse(clean)
        self.assertTrue(any("Reconfiguring filter graph" in l for l in lines))
        self.assertTrue(any("Non-monotonic DTS" in l for l in lines))
        t = time.time()
        ok, why = I.conversion_preserved(self.bad, 0, out, 0, -1001.001, expected_duration_s=198.9)
        note("legacy re-encode of the corrupt cut: conversion_preserved", False, ok, time.time() - t)
        print(f"      reason: {why}")
        self.assertFalse(ok)
        self.assertIn("conversion_created_silence at", why)
        # the options: -xerror (global) and -reinit_filter 0 (input) make it FAIL
        rc_x, _ = self._legacy(self.bad, os.path.join(self.tmp, "x.mkv"), extra_global=["-xerror"])
        rc_r, err_r = self._legacy(self.bad, os.path.join(self.tmp, "r.mkv"),
                                   extra_input=["-reinit_filter", "0"])
        note(f"legacy + -xerror / + -reinit_filter 0 on the corrupt cut: rc {rc_x} / {rc_r}",
             "fails / fails", f"{'fails' if rc_x else 'rc 0'} / {'fails' if rc_r else 'rc 0'}", 0.0)
        self.assertNotEqual(rc_x, 0)
        self.assertNotEqual(rc_r, 0)
        self.assertIn("Changing audio frame properties on the fly", err_r)
        # healthy cut: legacy with BOTH options is clean, rc 0, and preserved with fidelity
        out_g = os.path.join(self.tmp, "good_out.mkv")
        rc_g, err_g = self._legacy(self.good, out_g, extra_global=["-xerror"],
                                   extra_input=["-reinit_filter", "0"])
        self.assertEqual(rc_g, 0, err_g[-500:])
        self.assertTrue(I.conversion_stderr_is_clean(err_g)[0])
        t = time.time()
        rep = I.conversion_report(self.good, 0, out_g, 0, -1001.001, expected_duration_s=198.9)
        note("healthy cut, safe options: conversion_preserved", True, rep["preserved"], time.time() - t)
        print(f"      reason: {rep['reason']} windows={rep['fidelity_windows']}")
        self.assertTrue(rep["preserved"], rep["reason"])


# ---------------------------------------------------------------- the real files (long)

def _delay_ms_between(a, ia, b, ib, max_lag_s=90.0, span_s=3000.0):
    '''Coarse delay (400 ms) between two tracks from their silence maps' level curves:
    B's instant of a sound minus A's. Test helper only.'''
    ma, mb = I.silence_map(a, ia), I.silence_map(b, ib)
    n = int(span_s / I.SILENCE_BLOCK_S)
    x = np.clip(ma["db"][:n], -80, 0) + 80
    y = np.clip(mb["db"][:n], -80, 0) + 80
    x, y = x - x.mean(), y - y.mean()
    L = int(max_lag_s / I.SILENCE_BLOCK_S)
    best = max(range(-L, L + 1), key=lambda k: float(np.dot(x[max(0, -k):len(x) - max(0, k)],
                                                             y[max(0, k):len(y) - max(0, -k)][:len(x) - abs(k)])))
    return 1000.0 * (best * I.SILENCE_BLOCK_S + (mb["start_s"] - ma["start_s"]))


@unittest.skipUnless(TOOLS and os.path.exists(CSM_VARYG_DUAL), "Chainsaw files absent")
class RealPositives(unittest.TestCase):

    def check_track(self, case, path, idx, expected):
        r = I.track_check(path, idx)
        note(case, expected, r["sound"], r["cost_s"])
        if r["error_lines"]:
            print(f"      first: {r['error_lines'][0][:150]}")
        self.assertEqual(r["sound"], expected)

    def test_chainsaw_tracks(self):
        self.check_track("691 source VARYG AMZN DUAL jpn eac3", CSM_VARYG_DUAL, 1, False)
        self.check_track("691 source VARYG AMZN DUAL eng eac3", CSM_VARYG_DUAL, 2, False)
        self.check_track("691 source VARYG AMZN MULTi eng eac3", CSM_VARYG_MULTI, 2, False)
        self.check_track("691 source VARYG AMZN REPACK eng eac3", CSM_VARYG_REPACK, 2, False)
        self.check_track("691 source VARYG AMZN REPACK jpn eac3 (lab: clean)", CSM_VARYG_REPACK, 1, True)
        self.check_track("691 source ToonsHub AMZN DUAL eng eac3", CSM_TOONS_AMZN, 2, True)
        self.check_track("691 source Erai CR jpn aac", CSM_ERAI, 1, True)
        self.check_track("691 source ToonsHub CR DUAL jpn aac", CSM_TOONS_CR, 1, True)
        self.check_track("691 source ToonsHub CR DUAL eng aac", CSM_TOONS_CR, 2, True)

    def test_691_master_silences(self):
        # ffmpeg is blind to the master (26.9.10): its jpn decodes clean ...
        self.check_track("691 master jpn eac3 #11 (ffmpeg-blind, 26.9.10)", M691, 11, True)
        # ... the silences are not: jpn digital silence 4 007 -> 10 808 s where Erai's jpn plays
        rep = I.silence_report(M691, 11, CSM_ERAI, 1, 0)
        got = (rep["agree"], rep["result"])
        note("691 master jpn#11 vs Erai jpn, delay 0", (False, "A"), got, rep["cost_s"])
        print(f"      a_only: {rep['a_only'][:2]}  b_only: {rep['b_only'][:2]}")
        self.assertEqual(got, (False, "A"))
        rep = I.silence_report(M691, 5, CSM_ERAI, 1, 0)
        got = (rep["agree"], rep["result"])
        note("691 master spa#5 (stops at 1 377 s) vs Erai jpn", (False, "A"), got, rep["cost_s"])
        print(f"      a_only: {rep['a_only'][:2]}  b_only: {rep['b_only'][:2]}")
        self.assertEqual(got, (False, "A"))
        rep = I.silence_report(CSM_TOONS_CR, 1, CSM_ERAI, 1, 0)
        got = (rep["agree"], rep["result"])
        note("healthy pair ToonsHub CR jpn vs Erai jpn", (True, None), got, rep["cost_s"])
        self.assertEqual(got, (True, None))

    def test_349(self):
        if not (os.path.exists(M349) and os.path.exists(C349)):
            self.skipTest("349 files absent")
        t = time.time()
        d = _delay_ms_between(M349, 2, C349, 1)
        print(f"      349 master jpn#2 -> candidate jpn#1 coarse delay {d:.0f} ms "
              f"({time.time() - t:.1f} s)")
        rep = I.silence_report(M349, 2, C349, 1, d)
        got = (rep["agree"], rep["result"])
        # the master's silence from 3 404.6 s to its video end is A's; the candidate's own
        # packet holes (1 259 jumps, audio up to 7.4 s: MASTER_INTEGRITY memo) may be B's too
        note("349 master jpn#2 (cut at 3 404.6 s) vs candidate jpn", "(False, 'A' | 'AB')", got,
             rep["cost_s"], ok=(not got[0] and "A" in (got[1] or "")))
        print(f"      a_only: {rep['a_only'][-2:]}  b_only: {len(rep['b_only'])} "
              f"{rep['b_only'][:3]}")
        self.assertFalse(got[0])
        self.assertIn("A", got[1])
        self.assertTrue(any(abs(x["start_s"] - 3404.6) < 2.0 for x in rep["a_only"]))
        self.check_track("349 candidate jpn aac (memo: audio clean)", C349, 1, True)
        r = I.video_check(C349)
        note("349 candidate video_is_sound (sampled)", False, r["sound"], r["cost_s"])
        print(f"      reasons: {r['reasons'][:2]}")
        self.assertFalse(r["sound"])


@unittest.skipUnless(TOOLS, "ffmpeg missing")
class RealNegatives(unittest.TestCase):
    '''The 19 healthy masters: every audio track strictly decoded, every other audio track's
    silences against the first audio track (delay 0, one container), the video sampled.'''

    def test_healthy_masters(self):
        fails = []
        for label, path in HEALTHY:
            if not os.path.exists(path):
                print(f"  [absent] {label}")
                continue
            info = I._probe(path)
            audio = [s["index"] for s in info["streams"] if s.get("codec_type") == "audio"]
            t = time.time()
            bad_tracks = []
            for idx in audio:
                r = I.track_check(path, idx)
                if not r["sound"]:
                    bad_tracks.append((idx, r["verdict"], r["error_lines"][:1]))
            cost_a = time.time() - t
            t = time.time()
            disagree = []
            for idx in audio[1:]:
                rep = I.silence_report(path, audio[0], path, idx, 0)
                if not rep["agree"]:
                    disagree.append((audio[0], idx, rep["result"],
                                     (rep["a_only"] or rep["b_only"])[:1]))
            cost_s = time.time() - t
            v = I.video_check(path)
            # MEASURED 2026-09-26: e352's fre AAC (#2) is digital silence over its last 31.9 s
            # (1 419.2 -> 1 451.1 s) where the eng FLAC (#1) plays credits music -- a real
            # difference that the plain comparison (owner 23:5x, no tail exception) reports.
            known = KNOWN_SILENCE_DIFFERENCES.get(label, set())
            disagree = [d for d in disagree if (d[0], d[1], d[2]) not in known]
            ok = not bad_tracks and not disagree and v["sound"]
            note(f"healthy {label}: {len(audio)} audio strict / {len(audio) - 1} silence pairs / video",
                 "all sound", "all sound" if ok else
                 f"tracks={bad_tracks} pairs={disagree} video={v['verdict']}:{v['reasons'][:1]}",
                 cost_a + cost_s + v["cost_s"])
            print(f"      cost: audio strict {cost_a:.1f} s, silences {cost_s:.1f} s, "
                  f"video {v['cost_s']:.1f} s")
            if not ok:
                fails.append(label)
        self.assertEqual(fails, [])


def main(argv):
    global MODE
    only = None
    if "--only" in argv:
        only = argv[argv.index("--only") + 1]
    if "--quick" in argv:
        MODE = "quick"
    elif "--real-only" in argv:
        MODE = "real"
    loader = unittest.TestLoader()
    classes = {"quick": [Synthetic, StderrPatterns, ChainsawCut],
               "real": [RealPositives, RealNegatives],
               "all": [Synthetic, StderrPatterns, ChainsawCut, RealPositives, RealNegatives]}[MODE]
    suite = unittest.TestSuite()
    for c in classes:
        for t in loader.loadTestsFromTestCase(c):
            if only is None or only in t.id():
                suite.addTest(t)
    t0 = time.time()
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"\n{len(RESULTS)} measured calls in {time.time() - t0:.0f} s:")
    for case, exp, got, cost, ok in RESULTS:
        print(f"  {'ok ' if ok else 'BAD'} {cost:7.1f} s  {case}: {got}")
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
