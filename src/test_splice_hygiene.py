'''Tests for splice hygiene (ADDENDUM 23.2 / 23.3) -- run: python3 src/test_splice_hygiene.py (or
pytest). The pure rules on arrays, then one end-to-end build through ffmpeg: a candidate 4 dB
louder than the master that fills its hole -- the fill must come out at the candidate's level,
joined by 10 ms crossfades, with the exact duration.'''

import os
import subprocess
import sys
import tempfile
from decimal import Decimal

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import splice_hygiene as sh  # noqa: E402

R = sh.MEASURE_RATE


def _noise(seconds, seed, level=0.1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * R))
    x = np.convolve(x, np.ones(6) / 6, "same")
    return level * x / np.sqrt(np.mean(x * x))


def _db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))


def test_edge_gain_reads_a_flat_gain_on_coherent_material():
    x = _noise(10.0, 1)
    d, blocks = sh.edge_gain(x * 10 ** (4 / 20), x)
    assert blocks >= sh.MIN_COHERENT_BLOCKS and abs(d - 4.0) < 0.05, (d, blocks)


def test_edge_gain_ignores_different_material():
    d, blocks = sh.edge_gain(_noise(10.0, 2), _noise(10.0, 3))
    assert d is None and blocks < sh.MIN_COHERENT_BLOCKS, (d, blocks)


def test_fill_gain_rules():
    assert sh.fill_gain(0.2, 0.3)["mode"] == "none"                     # under the deadband
    assert sh.fill_gain(3.0, 3.2) == {"mode": "flat", "gain_a_db": 3.1, "gain_b_db": 3.1,
                                      "unmeasurable": False}
    assert sh.fill_gain(-3.7, -1.0)["mode"] == "ramp"                   # a dub's M&E
    assert sh.fill_gain(None, 2.0)["gain_a_db"] == 2.0                  # one edge takes the other's
    assert sh.fill_gain(None, None) == {"mode": "none", "gain_a_db": 0.0, "gain_b_db": 0.0,
                                        "unmeasurable": True}


def test_silence_is_a_hard_cut():
    sound = _noise(0.02, 4)
    assert sh.splice_join(sound, sound) == "crossfade"
    assert sh.splice_join(np.zeros(960), sound) == "hard_cut"
    assert sh.splice_join(sound, None) == "hard_cut"


def _write(path, samples):
    raw = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(R), "-ac", "1",
                    "-i", "-", "-c:a", "flac", path], input=raw, check=True)


def test_level_mismatched_fill_is_gain_aligned_and_crossfaded():
    import tools
    tools.software = {"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"}
    tools.dev = False
    import merge_video_chimeric as mvc
    a, x, b = _noise(20.0, 11), _noise(2.0, 12), _noise(20.0, 13)
    master = np.concatenate([a, x, b])
    candidate = np.concatenate([a, b]) * 10 ** (4 / 20)                # 4 dB louder, no X
    with tempfile.TemporaryDirectory() as work:
        m_path, c_path = os.path.join(work, "m.mka"), os.path.join(work, "c.mka")
        _write(m_path, master)
        _write(c_path, candidate)

        class Obj:
            def __init__(self, p):
                self.filePath = p
        pieces = [
            {"source": "candidate", "master_start_ms": Decimal(0), "master_end_ms": Decimal(20000),
             "source_start_ms": Decimal(0)},
            {"source": "master", "master_start_ms": Decimal(20000),
             "master_end_ms": Decimal(22000), "source_start_ms": Decimal(20000)},
            {"source": "candidate", "master_start_ms": Decimal(22000),
             "master_end_ms": Decimal(42000), "source_start_ms": Decimal(20000)}]
        splices = mvc.plan_splices(Obj(c_path), {"StreamOrder": 0}, Obj(m_path),
                                   {"StreamOrder": 0}, pieces, None, None)
        assert splices[1]["gain"]["mode"] == "flat", splices
        assert abs(splices[1]["gain"]["gain_a_db"] - 4.0) < 0.1, splices
        assert splices[1]["fade_left"] and splices[1]["fade_right"], splices
        graph, _pad, _heads = mvc.build_audio_filtergraph(pieces, 0, 0, R, "mono", splices=splices)
        assert graph.count("acrossfade=d=0.010:o=1:c1=tri:c2=tri") == 2, graph
        out = os.path.join(work, "o.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", c_path, "-i", m_path,
                        "-filter_complex", graph, "-map", "[aout]", "-ac", "1", out], check=True)
        done = subprocess.run(["ffmpeg", "-v", "error", "-i", out, "-f", "f32le", "-ac", "1",
                               "-ar", str(R), "-"], capture_output=True, check=True).stdout
        y = np.frombuffer(done, np.float32).astype(np.float64)
    assert abs(len(y) / R - 42.0) < 0.002, len(y) / R                   # the exact duration
    for at in (20.0, 22.0):                                             # no level step at a splice
        i = int(at * R)
        step = _db(y[i + int(0.02 * R):i + int(0.52 * R)]) - _db(y[i - int(0.52 * R):i - int(0.02 * R)])
        assert abs(step) < 0.5, (at, step)


if __name__ == "__main__":
    import tools
    tools.log_always = lambda message: None
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
