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


def test_the_uu171_shape_keeps_the_audio_instant():
    # MEASURED uu171 (Undead Unluck S01E09) cp2 under b563ffb6: frames 23817-23936 (a 4963 ms
    # span, five fills) for a 1001 ms fill; last cut minus fill 997.3297 s inside the 6.3 s
    # interval pinned 0.96 s after the audio instant 996.37 s, worst lag 1.837 -> 2.837 s
    interval = (993.52, 999.784)
    at, note = ro.video_cut_instant(_outcome(23817, 23936), DOMAIN, 1.001, interval, QUANTUM_MS)
    assert at is None and note.startswith("video fill 4963.29"), (at, note)
    assert ro.video_pin(at, interval, (993.52, 1000.785), 1.001,
                        DOMAIN["frame_ms"] / 1000.0) == (None, None)


def test_the_tougen_e07_shape_keeps_the_audio_instant():
    # MEASURED Tougen Anki S01E07 under b563ffb6: frames 22687-22897 (8758.75 ms, 1.75 fills)
    # for a 5004.999 ms fill; last cut minus fill 949.9907 s inside [949.72, 950.105] pinned
    # 39 ms before the audio instant 950.03 s, ja lag 3.188 -> 3.688 s
    at, note = ro.video_cut_instant(_outcome(22687, 22897), DOMAIN, 5.004999, (949.72, 950.105),
                                    QUANTUM_MS)
    assert at is None and note.startswith("video fill 8758.75"), (at, note)


def test_two_fills_in_a_sub_frame_interval_keep_the_audio_instant():
    # MEASURED e285 (The 100 S07E14) cp2/cp3 under b563ffb6: 48-frame spans (2002 ms, two
    # fills) for 960 / 959.543 ms fills, but 5 ms intervals: the audio already places the cut
    # finer than a frame; b563ffb6 moved them by 3.6 and 1.1 ms
    for first, last, extra, interval in ((39118, 39166, 0.96, (1632.585, 1632.59)),
                                         (48339, 48387, 0.959543, (2017.1805, 2017.185))):
        at, note = ro.video_cut_instant(_outcome(first, last), DOMAIN, extra, interval, QUANTUM_MS)
        assert at is None and note.startswith("video fill 2002"), (at, note)


def test_the_130_span_in_a_wide_interval_still_pins_when_two_fills():
    # control: the same two-fill span pins wherever its last cut minus the fill is inside a
    # wider-than-a-frame interval
    at, note = ro.video_cut_instant(_outcome(19538, 19781), DOMAIN, EXTRA_S, (817.0, 823.0),
                                    QUANTUM_MS)
    assert at is not None and abs(at - 820.0275) < 0.001 and "two fills" in note, (at, note)


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
