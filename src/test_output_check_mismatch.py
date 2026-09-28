'''Tests for `output_check_mismatch` (CASE_output_check_mismatch_20260928, ids 44/161/709/714) --
run: python3 src/test_output_check_mismatch.py.

  * the token is in `repair_orchestrator.DECLINE_CAUSES`, classed could-not-run like the other
    refusal of OUR product (`delivery_offset_exceeds_tolerance`), and `log_measurement_class`
    no longer writes the UNVOCABULARISED line for it;
  * the container guard of `verify_output_file` adds the codec delay the produced container
    DECLARES (ffprobe `initial_padding`, Matroska CodecDelay) to the track's frame: the measured
    shape of id 714 (AAC 48 kHz, 1024 priming samples, container 1429995.328 ms against the
    master's 1429972.000, i.e. 21.333 declared delay + 1.995 last frame) is accepted, and the
    same overshoot on a track declaring no delay is still refused;
  * a real encode through the build's own path (ffmpeg `aac` into Matroska, then the mkvmerge
    chapter remux) states that delay and `probe_output_streams` reads it.'''

import os
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tools  # noqa: E402

if not getattr(tools, "software", None):
    tools.software = {"ffmpeg": "ffmpeg", "ffprobe": "ffprobe", "mkvmerge": "mkvmerge"}
if not getattr(tools, "tmpFolder", None) or tools.tmpFolder == "/tmp":
    tools.tmpFolder = tempfile.mkdtemp(prefix="test_ocm_")

import merge_video_chimeric as mvc  # noqa: E402
import repair_orchestrator as ro  # noqa: E402

MASTER_VIDEO_MS = Decimal("1429972")        # id 714's master, video DURATION tag
MASTER_CONTAINER_MS = Decimal("1429972.000")
PRODUCED_CONTAINER_MS = Decimal("1429995.328")  # the refused product, measured


def _aac_stream(initial_padding, duration_ms=PRODUCED_CONTAINER_MS):
    return {"index": 0, "codec_type": "audio", "codec_name": "aac", "sample_rate": "48000",
            "frame_rate": "0/0", "initial_padding": initial_padding, "language": "jpn",
            "duration_ms": duration_ms, "duration_source": "matroska tag"}


def _verify(streams, container_ms):
    saved = mvc.probe_output_streams
    mvc.probe_output_streams = lambda path: (streams, container_ms)
    try:
        return mvc.verify_output_file("/nonexistent/produced.mkv", MASTER_VIDEO_MS,
                                      [{"language": "jpn"}], [], 500, MASTER_CONTAINER_MS)
    finally:
        mvc.probe_output_streams = saved


def test_token_is_vocabularised_could_not_run():
    assert ro.DECLINE_CAUSES["output_check_mismatch"] == ro.CLASS_COULD_NOT_RUN
    assert ro.DECLINE_CAUSES["output_check_mismatch"] == \
        ro.DECLINE_CAUSES["delivery_offset_exceeds_tolerance"]
    since = len(tools.logs)
    ro.log_measurement_class("/c.mkv", "output_check_mismatch")
    lines = [str(line) for line in tools.logs[since:]]
    assert not [line for line in lines if "UNVOCABULARISED" in line], lines
    assert any("cause=output_check_mismatch measurement=could_not_run" in line
               for line in lines), lines


def test_tolerance_adds_the_declared_codec_delay():
    tolerance, detail = mvc.container_grid_tolerance_ms([_aac_stream(1024)])
    frame = Decimal(1024 * 1000) / Decimal(48000)
    assert tolerance == frame + frame, (tolerance, detail)   # 1024 frame + 1024 priming
    assert "audio_codec=aac" in detail and "audio_codec_delay_ms=" in detail, detail
    # A track that declares no delay gets none: the bound stays one frame.
    for padding in (0, "0", None):
        tolerance, detail = mvc.container_grid_tolerance_ms([_aac_stream(padding)])
        assert tolerance == frame, (padding, tolerance)
    # ac3: 256 declared priming samples on a 1536-sample frame.
    ac3 = dict(_aac_stream(256), codec_name="ac3")
    tolerance, _ = mvc.container_grid_tolerance_ms([ac3])
    assert tolerance == Decimal(1792 * 1000) / Decimal(48000), tolerance


def test_id714_shape_is_accepted():
    report = _verify([_aac_stream(1024)], PRODUCED_CONTAINER_MS)
    assert report["would_refuse"] is False, report["problems"]
    assert Decimal(report["container_overshoot_ms"]) == Decimal("23.328")


def test_same_overshoot_without_declared_delay_is_refused():
    try:
        _verify([_aac_stream(0)], PRODUCED_CONTAINER_MS)
    except mvc.chimeric_error as error:
        assert getattr(error, "cause", None) == "output_check_mismatch", error
        assert "runs 23.328 ms past" in str(error), error
    else:
        raise AssertionError("an undeclared 23.328 ms overshoot was accepted")


def test_overshoot_past_frame_and_delay_is_refused():
    # One full AAC frame of real extra content on top of the delay: 21.333 + 21.333 + 1.
    container = MASTER_CONTAINER_MS + Decimal("43.667")
    try:
        _verify([_aac_stream(1024, container)], container)
    except mvc.chimeric_error as error:
        assert getattr(error, "cause", None) == "output_check_mismatch", error
    else:
        raise AssertionError("an overshoot past one frame plus the declared delay was accepted")


def test_real_aac_encode_states_its_delay_and_the_probe_reads_it():
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe") and shutil.which("mkvmerge")):
        print("skip: ffmpeg/ffprobe/mkvmerge not on PATH")
        return
    saved = dict(tools.software)
    tools.software.update({"ffmpeg": shutil.which("ffmpeg"),
                           "ffprobe": shutil.which("ffprobe"),
                           "mkvmerge": shutil.which("mkvmerge")})
    work = tempfile.mkdtemp(prefix="ocm_", dir=tools.tmpFolder)
    try:
        content_ms = Decimal("30000")
        built = os.path.join(work, "built.mka")
        subprocess.run([tools.software["ffmpeg"], "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                        "-i", "sine=f=440:r=48000,atrim=end=30,aformat=channel_layouts=stereo",
                        "-c:a", "aac", "-b:a", "128k", built], check=True)
        remuxed = os.path.join(work, "remuxed.mkv")
        completed = subprocess.run([tools.software["mkvmerge"], "-q", "-o", remuxed, built],
                                   capture_output=True)
        assert completed.returncode in (0, 1), completed
        streams, container_ms = mvc.probe_output_streams(remuxed)
        audio = [s for s in streams if s["codec_type"] == "audio"]
        assert str(audio[0]["initial_padding"]) == "1024", audio
        overshoot = container_ms - content_ms
        frame = Decimal(1024 * 1000) / Decimal(48000)
        tolerance, _ = mvc.container_grid_tolerance_ms(streams)
        # The container states the priming: more than one frame past the content, and within
        # one frame plus the declared delay.
        assert frame < overshoot <= tolerance, (overshoot, tolerance)
    finally:
        tools.software.clear()
        tools.software.update(saved)
        shutil.rmtree(work, ignore_errors=True)


def main():
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok   {name}")
        except Exception as error:                                       # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {error}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
