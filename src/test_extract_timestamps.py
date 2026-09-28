'''Tests for the strict extraction's muxer timestamps -- run: python3 src/test_extract_timestamps.py

`audio_extract.extract_audio_window` decodes under `-xerror`, where a WAV muxer complaint is fatal.
A TrueHD track resampled to 44.1 kHz hands the muxer a DTS one tick backwards (MEASURED 2026-09-28,
Fallout S01E03 BD master stream 1, rc 234 at 21 s) and a sound track was condemned
`comparison_track_corrupt`. The chain now ends in `asetpts=N/SR/TB`: the muxer's timestamps are
the sample count, and the WAV is the plain extraction's, byte for byte.'''

import hashlib
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tools  # noqa: E402

if not getattr(tools, "software", None):
    tools.software = {"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"}
tools.logs = getattr(tools, "logs", [])

import audio_extract  # noqa: E402

# a 30 s stream-copy of that master's TrueHD stream (scratchpad, not in the repo): the positive
# control when present, skipped otherwise
TRUEHD_CUT = os.environ.get(
    "VMSAM_TRUEHD_DTS_CUT",
    "/tmp/claude-1000/-home-vmsam-src-VMSAM/c9c86eae-36e9-4352-8ddc-e1b9c919c2a9/scratchpad/"
    "judges2/fo03_truehd30.mka")


class _Done:
    returncode, stdout, stderr = 0, b"", b""


def _captured_command(audio_filter):
    seen = []
    real = subprocess.run
    subprocess.run = lambda cmd, **kw: seen.append(cmd) or _Done()
    try:
        audio_extract.extract_audio_window("/x.mkv", 1, 0.0, 10.0, "/tmp/never.wav", 16000,
                                           audio_filter=audio_filter)
    except Exception:                                                    # noqa: BLE001
        pass                     # the stub writes no WAV: only the command matters here
    finally:
        subprocess.run = real
    return seen[0]


def test_chain_ends_in_the_sample_count():
    cmd = _captured_command(None)
    assert cmd[cmd.index("-af") + 1] == "asetpts=N/SR/TB", cmd
    chain = "aresample=384000,asetrate=383616,aresample=48000"
    cmd = _captured_command(chain)
    assert cmd[cmd.index("-af") + 1] == chain + ",asetpts=N/SR/TB", cmd
    assert cmd.count("-af") == 1 and cmd.index("-af") > cmd.index("-i"), cmd
    assert cmd[cmd.index("-xerror"):cmd.index("-xerror") + 3] == ["-xerror", "-reinit_filter", "0"]


def _md5(path):
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


def test_wav_is_the_plain_extraction_byte_for_byte():
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "s.mka")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "sine=f=440:r=48000:d=12", "-c:a", "flac", source], check=True)
        out, plain = os.path.join(tmp, "a.wav"), os.path.join(tmp, "b.wav")
        audio_extract.extract_audio_window(source, 0, 1.0, 10.0, out, 44100)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin", "-ss", "1.000000", "-t",
                        "10.000000", "-i", source, "-map", "0:0", "-vn", "-ac", "1", "-ar",
                        "44100", "-acodec", "pcm_s16le", plain], check=True)
        assert _md5(out) == _md5(plain)


def test_truehd_backward_dts_is_not_a_corrupt_track():
    if not os.path.exists(TRUEHD_CUT):
        print(f"skip test_truehd_backward_dts_is_not_a_corrupt_track: {TRUEHD_CUT} absent")
        return
    with tempfile.TemporaryDirectory() as tmp:
        bare = subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin", "-xerror", "-i",
                               TRUEHD_CUT, "-map", "0:0", "-ac", "1", "-ar", "44100",
                               os.path.join(tmp, "bare.wav")], capture_output=True)
        assert bare.returncode != 0 and b"Non-monotonic DTS" in bare.stderr, bare.returncode
        out = os.path.join(tmp, "a.wav")
        audio_extract.extract_audio_window(TRUEHD_CUT, 0, 0.0, 29.0, out, 44100)
        assert os.path.getsize(out) > 29 * 44100 * 2


if __name__ == "__main__":
    failures = 0
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        try:
            globals()[name]()
            print(f"ok   {name}")
        except Exception as error:                                       # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"{len(names) - failures}/{len(names)} passed")
    sys.exit(1 if failures else 0)
