'''Tests for ADDENDUM 26.9 (`repair_orchestrator.tail_content_verdict`, `wav_content_end_s`) --
run: python3 src/test_tail_content.py (or pytest).

The decision is pure: the three shapes the owner named are written with their measured numbers.
The content-end reader runs on a synthetic WAV written here.'''

import os
import sys
import tempfile
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import repair_orchestrator as ro  # noqa: E402


def _couple(last_m, last_c, end_m, end_c, name="1x1"):
    return {"couple": name, "last_common_master_s": last_m, "last_common_candidate_s": last_c,
            "master_content_end_s": end_m, "candidate_content_end_s": end_c}


def test_id_349_master_cut_short():
    # master audio content stops at 3404.6 s of a 4494.5 s video; the candidate runs to its end
    verdict, _ = ro.tail_content_verdict([_couple(3404.0, 3404.0, 3404.6, 4494.4)], 4494.49)
    assert verdict == "master_cut_short", verdict


def test_id_691_shape_is_not_master_cut_short():
    # a corrupt master: real audio to ~4007 s, digital silence, content again at the very end
    # of a 12,799 s track over 5,997 s of video -- its content reaches the timeline's end
    verdict, _ = ro.tail_content_verdict([_couple(4007.0, 4007.0, 12799.3, 5997.4)], 5997.498)
    assert verdict is None, verdict


def test_candidate_truncated_six_minutes_is_candidate_short():
    verdict, _ = ro.tail_content_verdict([_couple(1080.0, 1080.0, 1439.9, 1080.2)], 1440.0)
    assert verdict == "candidate_short", verdict


def test_short_difference_and_disagreeing_couples_decide_nothing():
    assert ro.tail_content_verdict([_couple(1300.0, 1300.0, 1439.0, 1300.0)], 1440.0)[0] is None
    both = [_couple(3404.0, 3404.0, 3404.6, 4494.4, "1x1"),
            _couple(3404.0, 3404.0, 4494.0, 4494.4, "2x1")]
    assert ro.tail_content_verdict(both, 4494.49)[0] is None
    assert ro.tail_content_verdict([_couple(None, 1.0, 2.0, 3.0)], 10.0)[0] is None


def test_wav_content_end_excludes_trailing_silence_and_dither():
    rate = 8000
    rng = np.random.default_rng(1)
    content = rng.normal(0, 0.2, rate * 70)                   # 70 s of programme
    dither = rng.normal(0, 10 ** (-101 / 20), rate * 90)       # -101 dB, under the floor
    silence = np.zeros(rate * 30)
    samples = np.concatenate([content, dither, silence])
    with tempfile.TemporaryDirectory() as work:
        path = os.path.join(work, "t.wav")
        with wave.open(path, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(rate)
            writer.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
        end = ro.wav_content_end_s(path)
    assert end is not None and abs(end - 70.0) <= ro.CONTENT_BLOCK_S, end


if __name__ == "__main__":
    import tools
    tools.log_always = lambda message: None
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
