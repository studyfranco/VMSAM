'''Tests for the verifier's master-own-step reading -- run: python3 src/test_master_intertrack_step.py
(or pytest).

Fallout S01E01 (CASE_fallout_e01_alignment_contradicts_plan_20260925): the master's French track
steps by one frame (42.2 ms) near its end while its English track, its video and the candidate
hold one offset; the delivered French read 2.1 / 40.1 ms against the master's French and the
build was refused `alignment_contradicts_plan`. Synthetic audio at the verifier's probe rate:
two languages sharing a music-and-effects bed, the master's second language stepping 42 ms.'''

import os
import sys
from decimal import Decimal

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import merge_video_chimeric as mvc  # noqa: E402

R = mvc.verify_probe_rate
LENGTH_S = 200
STEP_AT_S = 150
STEP = int(0.042 * R)          # 336 samples = 42 ms, one frame at 23.976 fps

rng = np.random.default_rng(20260925)
_bed = rng.standard_normal(LENGTH_S * R)
_en = 0.7 * _bed + 0.5 * rng.standard_normal(LENGTH_S * R)
_fr = 0.7 * _bed + 0.5 * rng.standard_normal(LENGTH_S * R)


def _stepped(signal):
    '''`signal` whose content after STEP_AT_S arrives STEP samples early.'''
    out = signal.copy()
    cut = STEP_AT_S * R
    out[cut:len(out) - STEP] = signal[cut + STEP:]
    return out


class _Master:
    filePath = "master.mkv"
    audios = {"en": [{"StreamOrder": 1}], "fr": [{"StreamOrder": 2}]}
    audiodesc = {}


def _install(tracks):
    def read(file_path, spec, start_ms, duration_ms, rate, deadline=None):
        start = int(Decimal(str(start_ms)) * rate / 1000)
        return tracks[(file_path, spec)][start:start + int(Decimal(str(duration_ms)) * rate / 1000)]
    mvc.read_mono_samples = read


PIECES = [{"source": "candidate", "master_start_ms": Decimal("0"),
           "master_end_ms": Decimal(LENGTH_S * 1000)}]
REPORTS = [{"language": "en", "stream_order": 1}, {"language": "fr", "stream_order": 2}]


def _verify(reports=REPORTS):
    return mvc._verify_on_master_timeline("out.mkv", _Master(), reports, PIECES, 15, 300,
                                          None, None)


def test_master_step_is_read_as_the_masters():
    _install({("master.mkv", "0:1"): _en, ("master.mkv", "0:2"): _stepped(_fr),
              ("out.mkv", "0:a:0"): _en, ("out.mkv", "0:a:1"): _fr})
    results = _verify()
    fr = [r for r in results if r["language"] == "fr"][0]
    assert fr["outcome"] == "aligned", fr
    assert fr["worst_lag_ms"] <= 1.0, fr
    step = fr["master_intertrack_step"]
    assert abs(abs(step["relation_ms"][1] - step["relation_ms"][0]) - 42.0) <= 0.5, step
    assert abs(max(abs(v) for v in fr["inconsistent_against_master_track"][0]["lags_ms"]) - 42.0) <= 0.5


def test_a_product_step_is_still_refused():
    _install({("master.mkv", "0:1"): _en, ("master.mkv", "0:2"): _fr,
              ("out.mkv", "0:a:0"): _en, ("out.mkv", "0:a:1"): _stepped(_fr)})
    try:
        _verify()
    except Exception as error:
        assert getattr(error, "cause", None) == "alignment_contradicts_plan", error
        return
    raise AssertionError("a step in the delivered track was accepted")


def test_no_aligned_sibling_keeps_the_refusal():
    _install({("master.mkv", "0:2"): _stepped(_fr), ("out.mkv", "0:a:0"): _fr})
    try:
        _verify([{"language": "fr", "stream_order": 2}])
    except Exception as error:
        assert getattr(error, "cause", None) == "alignment_contradicts_plan", error
        return
    raise AssertionError("an unexplained disagreement was accepted")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
