# -*- coding: utf-8 -*-
"""
Shared audio-extraction and stream-enumeration helpers.

extract_audio_window extracts one audio window to a mono PCM WAV with a strict
decode; streams_for lists every audio stream of a language.
"""
from os import stat as os_stat

import subprocess

import tools
import repair_log

# Input options placed before -i. "-reinit_filter 0" makes a mid-stream change of
# audio parameters fail instead of silently rebuilding the filter graph.
STRICT_DECODE_FLAGS = ["-xerror", "-reinit_filter", "0",
                       "-err_detect", "crccheck+bitstream+buffer+explode"]


def strict_decode_verdict(returncode, stderr_text):
    """Return None when the decode is sound, else the stderr lines that condemn the source.

    The source is condemned by a non-zero exit with a decoder error line, or a zero exit
    with a fatal pattern. A non-zero exit without a decoder line is not a media verdict.
    """
    import integrity
    lines = integrity.decode_error_lines(stderr_text)
    if returncode != 0 and lines:
        return lines
    clean, fatal = integrity.conversion_stderr_is_clean(stderr_text)
    return None if clean else fatal


class StrictDecodeFailed(Exception):
    """The extraction's strict decode found the source track corrupt.

    Attributes lines (first 20 offending stderr lines) and rc (ffmpeg exit status).
    """

    def __init__(self, source_path, stream_order, rc, lines):
        first = lines[0][:200] if lines else "?"
        super().__init__(f"strict decode of {source_path} stream {stream_order} failed: rc={rc} "
                         f"first=«{first}»")
        self.source_path = source_path
        self.stream_order = stream_order
        self.rc = rc
        self.lines = list(lines)[:20]


class ExtractProducedNothing(Exception):
    """ffmpeg exited 0 and produced no (or under one second of) audio.

    Typically caused by seeking past the end of the source.
    """


def extract_audio_window(source_path, stream_order, start_seconds, length_seconds, out_path,
                          sample_rate, audio_filter=None):
    """Extract an audio window of one stream to a mono 16-bit PCM WAV.

    Args:
        source_path: media file to read.
        stream_order: absolute stream index (ffmpeg map 0:<n>).
        start_seconds, length_seconds: window read from the source.
        out_path: WAV file to write.
        sample_rate: output rate; no default, so every caller passes the pair's rate.
        audio_filter: optional -af filtergraph (e.g. a speed correction); the output
            length then differs from length_seconds.

    Raises:
        StrictDecodeFailed: the decode condemns the source.
        Exception: any other non-zero ffmpeg exit.
        tools.decoder_timeout: ffmpeg exceeded its timeout.
        ExtractProducedNothing: under one second of audio was written.
    """
    cmd = [tools.software["ffmpeg"], "-v", "error", "-y", "-nostdin"] + STRICT_DECODE_FLAGS + [
           "-ss", f"{start_seconds:.6f}", "-t", f"{length_seconds:.6f}",
           "-i", source_path, "-map", f"0:{stream_order}",
           "-vn", "-ac", "1", "-ar", str(sample_rate)]
    # Timestamps from the sample count: under -xerror a resampled track's
    # non-monotonic DTS would otherwise be fatal.
    cmd.extend(["-af", (audio_filter + "," if audio_filter else "") + "asetpts=N/SR/TB"])
    cmd.extend(["-acodec", "pcm_s16le", out_path])
    tools.dev_log(f"audio_extract: extract_audio_window ffmpeg call file={source_path} "
                  f"stream_order={stream_order} out_path={out_path}"
                  + (f" audio_filter={audio_filter}" if audio_filter else "") + "\n")
    timeout = tools.decoder_timeout_for(length_seconds)
    try:
        with repair_log.announced("audio_extract", "ffmpeg", source_path,
                                  media_s=length_seconds) as call:
            done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  timeout=timeout)
            call["exit"] = done.returncode
    except subprocess.TimeoutExpired:
        raise tools.decoder_timeout("extract_audio_window", timeout,
                                    f"file={source_path} stream_order={stream_order}")
    stderr_text = done.stderr.decode("utf-8", "replace")
    corrupt = strict_decode_verdict(done.returncode, stderr_text)
    if corrupt is not None:
        tools.log_always(f"audio_extract: strict_decode_failed file={source_path} "
                         f"stream_order={stream_order} rc={done.returncode} "
                         f"lines={len(corrupt)} first=«{corrupt[0][:200]}»\n")
        raise StrictDecodeFailed(source_path, stream_order, done.returncode, corrupt)
    if done.returncode != 0:
        raise Exception("This cmd is in error: " + " ".join(cmd) + "\n"
                        + done.stderr.decode("utf-8", "replace") + "\nReturn code: "
                        + str(done.returncode) + "\n")
    # Seeking past the end exits 0 with a header-only WAV, so require one second of audio.
    _floor = sample_rate * 2          # 1 s, 16-bit mono
    try:
        _written = os_stat(out_path).st_size
    except OSError:
        raise ExtractProducedNothing("extract wrote no file at the requested position")
    if _written < _floor:
        raise ExtractProducedNothing(
            f"extract wrote {_written} bytes, under {_floor} for one second at {sample_rate} Hz")


def streams_for(video_obj, language):
    """Return the StreamOrder of every audio stream of `language` in `video_obj` ([] if none).

    All streams, since two streams of one language can sit at different offsets.
    """
    audios = getattr(video_obj, "audios", None)
    if not audios or language not in audios:
        return []
    return [entry["StreamOrder"] for entry in audios[language]
            if entry.get("StreamOrder") is not None]
