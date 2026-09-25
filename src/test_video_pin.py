'''Tests for `repair_orchestrator.video_pin` (ADDENDUM 25: "the audio bounds, the video pins") --
run: python3 src/test_video_pin.py (or pytest). No media: the numbers are uu171's, measured.'''

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import repair_orchestrator as ro  # noqa: E402

FRAME = 1001 / 24000


def test_frame_inside_the_interval_pins():
    # uu171 on origin 8fffbf08: interval [209.885, 210.989], frame 210.9607
    at, decision = ro.video_pin(210.9607, (209.885, 210.989), (209.885, 211.99), 1.001, FRAME)
    assert (at, decision) == (210.9607, "video_frame_inside_audio_interval"), (at, decision)


def test_narrow_interval_inside_the_step_still_pins():
    # uu171 on the pre-ff14b603 fine edges: interval [211.014, 211.019] read narrower than the
    # step's own bounds [209.88, 212.02]; the frame fits the 1.001 s fill -> pinned, not the audio
    at, decision = ro.video_pin(210.9607, (211.014, 211.019), (209.88, 212.02), 1.001, FRAME)
    assert (at, decision) == (210.9607, "video_frame_inside_audio_bounds"), (at, decision)


def test_frame_the_audio_excludes_or_a_blind_video_keeps_the_audio_instant():
    # a frame whose fill would run past the step's last edge
    assert ro.video_pin(211.5, (211.014, 211.019), (209.88, 212.02), 1.001, FRAME) == (None, None)
    # a frame before the step's first edge
    assert ro.video_pin(209.0, (209.885, 210.989), (209.885, 211.99), 1.001, FRAME) == (None, None)
    # the video is blind (no cut frame)
    assert ro.video_pin(None, (209.885, 210.989), (209.885, 211.99), 1.001, FRAME) == (None, None)


def test_one_frame_tolerance():
    at, decision = ro.video_pin(210.989 + FRAME * 0.9, (209.885, 210.989), (209.885, 211.99), 1.001,
                                FRAME)
    assert decision == "video_frame_inside_audio_interval" and at == 210.989, (at, decision)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
