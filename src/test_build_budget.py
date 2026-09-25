'''Tests for the repair budget over the track build (ADDENDUM 26.3 / 26.8) -- run:
python3 src/test_build_budget.py (or pytest). The real `assemble_on_master_timeline` loop over a
twelve-track candidate, each track's build replaced by a one-second stand-in: with a budget of
2.5 s the build must stop by name within the budget + one build, name the tracks built, and
never reach the mux.'''

import os
import sys
import tempfile
import time
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tools  # noqa: E402
import merge_video_chimeric as mvc  # noqa: E402

BUILD_S = 1.0
TRACKS = 12


class _Video:
    def __init__(self, n_audio):
        self.filePath = "/synthetic/candidate.mkv"
        self.audios = {"ja": [{"StreamOrder": i + 1} for i in range(n_audio)]}
        self.audiodesc, self.commentary, self.subtitles = {}, {}, {}
        self.video = {"Duration": "60.0", "FrameRate": "23.976", "FrameRate_Mode": "CFR"}


def _run(budget_s, logs):
    piece = [{"source": "candidate", "master_start_ms": Decimal(0),
              "master_end_ms": Decimal(60000), "source_start_ms": Decimal(0)}]
    plans = {i + 1: {"pieces": piece, "extent_ms": Decimal(60000), "extent_source": "test",
                     "offset_measured": True} for i in range(TRACKS)}
    built, muxed = [], []
    real_build, real_mux = mvc.build_one_audio_track, mvc.mux_repaired_file

    def fake_build(candidate_obj, master_obj, audio, *args, **kwargs):
        time.sleep(BUILD_S)
        built.append(audio["StreamOrder"])
        return {"stream_order": audio["StreamOrder"], "speed_ratio_applied": None}

    mvc.build_one_audio_track = fake_build
    mvc.mux_repaired_file = lambda *a, **k: muxed.append(True)
    tools.log_always = lambda message: logs.append(message)
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory() as work:
            mvc.assemble_on_master_timeline(
                _Video(TRACKS), _Video(1), plans, piece, work, os.path.join(work, "o.mkv"), "",
                "test", deadline=started + budget_s)
        return None, built, muxed, time.monotonic() - started
    except mvc.chimeric_error as error:
        return error, built, muxed, time.monotonic() - started
    finally:
        mvc.build_one_audio_track, mvc.mux_repaired_file = real_build, real_mux


def test_track_build_declines_by_the_budget_within_one_build():
    logs = []
    error, built, muxed, elapsed = _run(2.5, logs)
    assert error is not None and getattr(error, "cause", None) == "repair_budget_exceeded", error
    assert elapsed <= 2.5 + BUILD_S + 0.5, elapsed                    # budget + one build
    assert 2 <= len(built) <= 4 and not muxed, (built, muxed)          # stopped, never muxed
    assert any("partial_plan cause=repair_budget_exceeded" in m and "audio:1" in m for m in logs), logs


if __name__ == "__main__":
    tools.dev = False
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
