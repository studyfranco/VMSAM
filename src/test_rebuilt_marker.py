'''Every track the repair builds carries its marker, and an unmarked rebuilt track is raced
(errid 319, 2026-09-28) -- run: python3 src/test_rebuilt_marker.py (or pytest).

MEASURED id 319 (0-saiji Start Dash Monogatari S01E04): one candidate zone over the whole master,
head and tail trims only, no splice -> ADDENDUM 5 "not chimeric" -> an empty marker -> the mux
wrote no VMSAM_FABRICATED -> the delivery gate skipped the track -> the candidate's 48 kHz jpn,
re-encoded, replaced the master's intact 44.1 kHz jpn with no race.'''

import os
import sys
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tools  # noqa: E402
import merge_video_chimeric as mvc  # noqa: E402
import merge_video_repair as mvr  # noqa: E402


def _audio(stream, rate, **extra):
    audio = {"keep": True, "Format": "AAC", "Channels": "2", "SamplingRate": rate,
             "BitRate": "128000", "StreamOrder": stream, "Language": "ja"}
    audio.update(extra)
    return audio


class _Obj:
    def __init__(self, audios):
        self.filePath = "/synthetic/x.mkv"
        self.audios = {"ja": audios}
        self.audiodesc, self.commentary = {}, {}


def test_delivered_marker_is_never_empty():
    assert mvc.delivered_marker("", None) == mvc.REBUILT_MARKER == "rebuilt"
    assert mvc.delivered_marker(None, None) == "rebuilt"
    assert mvc.delivered_marker("chimeric", None) == "chimeric"
    assert mvc.delivered_marker("", Decimal(1001) / Decimal(1000)) == "resampled:1001/1000"
    # compose_marker itself is unchanged (its readers and test_rate_arm rely on it)
    assert mvc.compose_marker("", None) == ""


def test_mux_writes_a_marker_on_every_track():
    commands = []
    tools.software.setdefault("ffmpeg", "ffmpeg")
    real = tools.launch_cmdExt_with_timeout_reload
    tools.launch_cmdExt_with_timeout_reload = lambda command, *a, **k: commands.append(command)
    try:
        mvc.mux_repaired_file(
            [{"path": "/synthetic/a0.mka", "language": "jpn", "title": None, "marker": ""},
             {"path": "/synthetic/a1.mka", "language": "eng", "title": None}],
            [{"path": "/synthetic/s0.mks", "language": "eng", "title": None, "marker": ""}],
            "/synthetic/out.mkv", "", 60, "test")
    finally:
        tools.launch_cmdExt_with_timeout_reload = real
    command = commands[0]
    tags = [command[i + 1] for i, part in enumerate(command) if part.startswith("-metadata:s:")]
    fabricated = [tag for tag in tags if tag.startswith("VMSAM_FABRICATED=")]
    assert fabricated == ["VMSAM_FABRICATED=rebuilt"] * 3, fabricated
    assert len([tag for tag in tags if tag.startswith("VMSAM=")]) == 3


def test_mark_audio_dicts_marks_every_rebuilt_track():
    repaired = _Obj([_audio(0, "48000"), _audio(1, "48000", extra={"VMSAM_FABRICATED": "chimeric"})])
    mvr.mark_audio_dicts(repaired, "")
    assert [a["fabricated"] for a in repaired.audios["ja"]] == ["rebuilt", "chimeric"]


def _gate(repaired_audio):
    master = _Obj([_audio(1, "44100")])
    repaired = _Obj([repaired_audio])
    logs = []
    real_logs = tools.logs
    tools.logs = logs
    try:
        dropped = mvr.gate_fabricated_delivery(
            repaired, master, content_probe=lambda *a: (True, "synthetic_same_content"))
    finally:
        tools.logs = real_logs
    return dropped, logs


def test_the_319_shape_loses_to_the_intact_master_track():
    # the path the product takes after the fix: marked at mux, re-probed under extra
    audio = _audio(0, "48000", extra={"VMSAM_FABRICATED": "rebuilt"})
    dropped, logs = _gate(audio)
    assert [d["cause"] for d in dropped] == ["intact_same_language_wins"], dropped
    assert audio["keep"] is False
    assert any("fabricated_dropped cause=intact_same_language_wins" in line for line in logs)


def test_an_unmarked_rebuilt_track_is_raced_never_skipped():
    # defence in depth: a track of the repaired file with no marker at all
    audio = _audio(0, "48000")
    dropped, logs = _gate(audio)
    assert [d["cause"] for d in dropped] == ["intact_same_language_wins"], dropped
    assert dropped[0]["marker"] == "rebuilt(unmarked)"
    assert audio["keep"] is False
    assert any(line.startswith("repair: unmarked_rebuilt_track") for line in logs)


def test_a_rebuilt_track_of_a_language_the_master_lacks_is_kept():
    audio = _audio(0, "48000", extra={"VMSAM_FABRICATED": "rebuilt"})
    master = _Obj([])
    repaired = _Obj([audio])
    real_logs, tools.logs = tools.logs, []
    try:
        dropped = mvr.gate_fabricated_delivery(repaired, master,
                                               content_probe=lambda *a: (True, "x"))
        logs = tools.logs
    finally:
        tools.logs = real_logs
    assert dropped == [] and audio["keep"] is True
    assert any("fabricated_kept cause=no_intact_master_track" in line for line in logs)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"{len(tests)} passed")
