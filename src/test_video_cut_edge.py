'''The video cut the audio agrees with pins, even when the resolver's other cut is wrong
(errid 130, 2026-09-28) -- run: python3 src/test_video_cut_edge.py (or pytest).

MEASURED errid 130 (Tougen Anki S01E03): walk step -5005.001 ms, interval [819.985, 820.125] s;
the resolver read frames 19538-19781 (814.897-825.0325 s, a 10135 ms fill), so the width check
dropped the video and the quietest instant (820.095 s) cut 67 ms after the video-pinned truth
(820.028 s, cut_check L5). The last cut minus the audio fill is that truth.'''

import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import repair_orchestrator as ro  # noqa: E402

RATE = Fraction(24000, 1001)
DOMAIN = {"master_rate": RATE, "frame_ms": float(1000 / RATE)}
QUANTUM_MS = 123.80952380952381
INTERVAL = (819.985, 820.125)
EXTRA_S = 5.005001


def _outcome(first, last, status=None):
    return {"status": status or ro.HOLE_RESOLVED, "master_start_frame": first,
            "master_end_frame": last}


def test_the_130_shape_pins_on_the_last_cut():
    at, note = ro.video_cut_instant(_outcome(19538, 19781), DOMAIN, EXTRA_S, INTERVAL, QUANTUM_MS)
    assert at is not None and abs(at - 820.0275) < 0.001, at
    assert "last cut minus the audio fill" in note and "10135" in note, note
    pinned, decision = ro.video_pin(at, INTERVAL, (819.985, 825.13), EXTRA_S,
                                    DOMAIN["frame_ms"] / 1000.0)
    assert abs(pinned - 820.028) < 0.015 and decision == "video_frame_inside_audio_interval"


def test_an_agreeing_width_offers_the_first_cut_as_before():
    # 820.0275 s is frame 19661; a 120-frame fill agrees with 5.005 s
    at, note = ro.video_cut_instant(_outcome(19661, 19781), DOMAIN, EXTRA_S, INTERVAL, QUANTUM_MS)
    assert note is None and abs(at - 19661 / float(RATE)) < 1e-9, (at, note)


def test_a_contradicting_width_never_offers_its_first_cut():
    # as before the fix: the first cut of a contradicted width is not trusted (errid 695's
    # additions would have moved by up to 222 ms)
    at, note = ro.video_cut_instant(_outcome(19661, 19900), DOMAIN, EXTRA_S, INTERVAL, QUANTUM_MS)
    assert at is None and note.startswith("video fill"), (at, note)


def test_the_last_cut_must_land_inside_the_interval_itself():
    # id 126 cut 2: interval [1220.76, 1220.749] (inverted), last cut minus fill 1220.7195 s
    at, note = ro.video_cut_instant(_outcome(29204, 29292), DOMAIN, 1.000999,
                                    (1220.76, 1220.749), QUANTUM_MS)
    assert at is None, (at, note)


def test_neither_cut_inside_the_interval_keeps_the_audio_instant():
    at, note = ro.video_cut_instant(_outcome(19538, 19900), DOMAIN, EXTRA_S, INTERVAL, QUANTUM_MS)
    assert at is None and note.startswith("video fill"), (at, note)


def test_an_addition_never_pins_on_the_last_cut():
    # no fill (extra 0): the last cut says nothing about where the candidate's excess starts
    at, note = ro.video_cut_instant(_outcome(19538, 19661), DOMAIN, 0.0, INTERVAL, QUANTUM_MS)
    assert at is None and note is not None, (at, note)


def test_an_unresolved_video_offers_nothing():
    for status in (ro.HOLE_NO_CUT_CONFIRMED, ro.HOLE_DECLINED):
        assert ro.video_cut_instant(_outcome(19538, 19781, status), DOMAIN, EXTRA_S, INTERVAL,
                                    QUANTUM_MS) == (None, None)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"{len(tests)} passed")
