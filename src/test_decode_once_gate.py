'''Tests for the decode-once cache and the delivery gate's fingerprint-slice instrument
(coordinator 2026-09-28) -- run: python3 src/test_decode_once_gate.py (or pytest). Synthetic
media: a 400 s tone sequence written to a master .mkv and, in a product .mkv, (0) the same track
12 dB quieter -- a level-shifted rebuild, which must still read as the SAME content, so the intact
master track still wins and the rebuild is refused delivery -- and (1) the same track 2 s later,
which must not.'''

import os
import random
import subprocess
import sys
import tempfile
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import tools  # noqa: E402
tools.software = tools.config_loader("config.ini", "software")
import audioCorrelation  # noqa: E402
import merge_video_decode_once as once  # noqa: E402
import merge_video_repair as mvr  # noqa: E402

RATE = 22050
SECONDS = 400


class _Obj:
    def __init__(self, file_path):
        self.filePath = file_path


def _tones(seconds, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(int(0.25 * RATE)) / RATE
    notes = []
    for _ in range(int(seconds / 0.25)):
        freqs = rng.choice([220, 247, 262, 294, 330, 349, 392, 440, 494, 523, 587, 659], 3)
        notes.append(sum(np.sin(2 * np.pi * f * t) for f in freqs) / 3)
    return np.concatenate(notes).astype(np.float32)


def _wav(path, samples):
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((np.clip(samples, -1, 1) * 30000).astype(np.int16).tobytes())


def _mkv(out, wavs):
    cmd = [tools.software["ffmpeg"], "-v", "error", "-y", "-nostdin"]
    for w in wavs:
        cmd += ["-i", w]
    for i in range(len(wavs)):
        cmd += ["-map", f"{i}:a"]
    cmd += ["-c:a", "flac", out]
    subprocess.run(cmd, check=True)


def _fixture(tmp):
    base = _tones(SECONDS)
    shifted = np.concatenate([np.zeros(2 * RATE, np.float32), base[:-2 * RATE]])
    paths = {}
    for name, samples in (("m", base), ("quiet", base * 10 ** (-12 / 20)), ("late", shifted)):
        paths[name] = os.path.join(tmp, f"{name}.wav")
        _wav(paths[name], samples)
    master = os.path.join(tmp, "master.mkv")
    product = os.path.join(tmp, "product.mkv")
    _mkv(master, [paths["m"]])
    _mkv(product, [paths["quiet"], paths["late"]])
    return master, product


def test_compare_is_audiocorrelation_compare():
    rng = random.Random(3)
    x = [rng.getrandbits(32) for _ in range(300)]
    y = x[17:] + [rng.getrandbits(32) for _ in range(40)]
    span = min(len(x), len(y)) - audioCorrelation.min_overlap
    assert once.compare(x, y, span, 1) == audioCorrelation.compare(x, y, span, 1)
    fast = once.correlate_points(x, y, 37.0)
    corr = audioCorrelation.compare(x, y, span, 1)
    assert fast == audioCorrelation.get_max_corr(corr, None, None, span, int(37.0 / len(x) * 1000))


def test_level_shifted_rebuild_is_same_content_and_a_shifted_one_is_not():
    tmp = tempfile.mkdtemp(prefix="test_decode_once_")
    saved = (tools.tmpFolder, tools.logs)
    tools.tmpFolder, tools.logs = tmp, []
    try:
        master, product = _fixture(tmp)
        once.begin(tmp, "test")
        m_audio = {"Duration": str(SECONDS), "StreamOrder": 0}
        quiet = {"Duration": str(SECONDS), "StreamOrder": 0}
        late = {"Duration": str(SECONDS), "StreamOrder": 1}
        same, detail = mvr.measure_same_content(_Obj(master), m_audio, _Obj(product), quiet, tmp)
        assert same is True, detail
        same, detail = mvr.measure_same_content(_Obj(master), m_audio, _Obj(product), late, tmp)
        assert same is False, detail
        assert "2" in detail and "instrument=fingerprint_slices" in detail, detail
        once.end()
        lines = "".join(tools.logs)
        # the master's track is decoded once and read twice; each product track once
        assert lines.count("decode_once: miss kind=fingerprint file=master.mkv") == 1, lines
        assert lines.count("decode_once: hit kind=fingerprint file=master.mkv") == 1, lines
        assert lines.count("decode_once: miss kind=fingerprint file=product.mkv") == 2, lines
        assert "decode_once: summary hits=1 misses=3" in lines, lines
    finally:
        once.end()
        tools.tmpFolder, tools.logs = saved
        subprocess.run(["rm", "-rf", tmp])


def test_a_stream_that_starts_late_is_read_on_the_file_clock():
    # MEASURED Tougen S01E07: the master's ja stream starts at 1.125 s, the product's track at 0
    # carries the same programme on the master's timeline (1.125 s of silence first). Read from
    # each stream's first sample they sat 1125 ms apart and the gate kept the rebuild as
    # "another version"; on the file clock they are the same content.
    tmp = tempfile.mkdtemp(prefix="test_decode_once_late_")
    saved = (tools.tmpFolder, tools.logs)
    tools.tmpFolder, tools.logs = tmp, []
    try:
        base = _tones(SECONDS)
        late = int(1.125 * RATE)
        _wav(os.path.join(tmp, "m.wav"), base)
        _wav(os.path.join(tmp, "p.wav"), np.concatenate([np.zeros(late, np.float32), base]))
        master = os.path.join(tmp, "master_late.mkv")
        subprocess.run([tools.software["ffmpeg"], "-v", "error", "-y", "-nostdin", "-itsoffset",
                        "1.125", "-i", os.path.join(tmp, "m.wav"), "-c:a", "flac", master],
                       check=True)
        product = os.path.join(tmp, "product_zero.mkv")
        _mkv(product, [os.path.join(tmp, "p.wav")])
        start = subprocess.run([tools.software["ffprobe"], "-v", "error", "-select_streams", "a:0",
                                "-show_entries", "stream=start_time", "-of", "csv=p=0", master],
                               capture_output=True, text=True).stdout.strip()
        assert abs(float(start) - 1.125) < 0.002, start
        once.begin(tmp, "late")
        m_audio = {"Duration": str(SECONDS), "StreamOrder": 0, "ffprobe": {"start_time": start}}
        p_audio = {"Duration": str(SECONDS), "StreamOrder": 0, "ffprobe": {"start_time": "0"}}
        same, detail = mvr.measure_same_content(_Obj(master), m_audio, _Obj(product), p_audio, tmp)
        assert same is True and "delays_ms=[0]" in detail, detail
    finally:
        once.end()
        tools.tmpFolder, tools.logs = saved
        subprocess.run(["rm", "-rf", tmp])


def test_outside_a_scope_nothing_is_cached():
    saved = tools.logs
    tools.logs = []
    try:
        calls = []
        for _ in range(2):
            once.get("file_clock", __file__, 1, {"x": 1}, lambda: calls.append(1) or np.zeros(3))
        assert len(calls) == 2
        assert "".join(tools.logs).count("scope=none") == 2
    finally:
        tools.logs = saved


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
