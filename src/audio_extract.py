# -*- coding: utf-8 -*-
"""
audio_extract.py -- the shared audio-extraction and stream-enumeration helpers, lifted out of
`change_point_locator.py` (2026-09-22) so they could outlive it.

The locator was removed with the switch to the orchestrator (2026-09-24,
RULING_20260922_ORCHESTRATOR_ARCHITECTURE.MD ADDENDUM 8 points 4 and 6), and with it the
re-export shims (`_extract`, `_streams_for`, `_all_audio_streams`, `_audio_duration_seconds`)
that kept its callers on the same function objects, plus `all_audio_streams` and
`audio_duration_seconds`, whose only callers were the locator and the modules that reached
into it. What remains is what `repair_orchestrator` calls: `extract_audio_window` and
`streams_for`, and the `ExtractProducedNothing` refusal the former raises.

The ffmpeg narration line was renamed from `locator: _extract ...` to
`audio_extract: extract_audio_window ...` in the same batch -- the rename this file's first
version deferred "to the locator's own removal, where the line has a new owner to be named
after". No parser reads that prefix (grep over src/ and VMSAM_HELP_AI/tools, 2026-09-24).
"""
from os import stat as os_stat

import subprocess

import tools
import repair_log

# `-reinit_filter 0` (owner 2026-09-26 01:3x, « avec -xerror -reinit_filter 0 »): an INPUT option,
# so it stands before the `-i` below -- a change of audio parameters mid-stream fails the filter
# graph instead of rebuilding it (MEASURED on the Chainsaw cut: rc 234 alone, « Changing audio
# frame properties on the fly is not supported »).
STRICT_DECODE_FLAGS = ["-xerror", "-reinit_filter", "0",
                       "-err_detect", "crccheck+bitstream+buffer+explode"]


def strict_decode_verdict(returncode, stderr_text):
    """None when the extraction's decode is sound; else the lines that condemn the source:
    rc != 0 WITH a decoder error line (`integrity.decode_error_lines`: every `-v error` line
    but the muxer's and the seek's), or rc 0 with a fatal pattern
    (`integrity.conversion_stderr_is_clean`). An rc != 0 without a decoder line is not a
    verdict on the media and keeps its old refusal below."""
    import integrity
    lines = integrity.decode_error_lines(stderr_text)
    if returncode != 0 and lines:
        return lines
    clean, fatal = integrity.conversion_stderr_is_clean(stderr_text)
    return None if clean else fatal


class StrictDecodeFailed(Exception):
    """The extraction's own decode says the SOURCE TRACK is corrupt (owner 2026-09-25 23:4x,
    « strict decode gratuit »: the WAV the prime extracts for fpcalc is decoded with
    `-xerror -err_detect crccheck+bitstream+buffer+explode`, so that decode IS the strict
    decode of `integrity.track_check` over the extracted span). Raised when ffmpeg exits
    non-zero with a decoder error line, or exits 0 with one of
    `integrity.FATAL_CONVERSION_PATTERNS` on its stderr. `lines` are the offending lines
    (first 20), `rc` ffmpeg's exit status. A subclass of Exception like the refusal below, so a
    caller that catches everything still refuses as before; the orchestrator's prime catches it
    BY NAME (`comparison_track_corrupt`, the owner's policy)."""

    def __init__(self, source_path, stream_order, rc, lines):
        first = lines[0][:200] if lines else "?"
        super().__init__(f"strict decode of {source_path} stream {stream_order} failed: rc={rc} "
                         f"first=«{first}»")
        self.source_path = source_path
        self.stream_order = stream_order
        self.rc = rc
        self.lines = list(lines)[:20]


class ExtractProducedNothing(Exception):
    """ffmpeg exited 0 and produced no audio. A type I own, so the site tally can name it.

    *** NOT a generic Exception: `_probe` already catches everything and reports
    `extract_or_correlate_raised`, which would fold this into the correlator's failures.
    This is the one failure mode `vmsam-ci` traced to a root cause, and it deserves to be
    distinguishable from a correlation that ran and failed. ***

    MOVED HERE 2026-09-22 from `change_point_locator`, which bound its own name to THIS class
    object so its probe site kept catching it; the class identity lives here alone since the
    locator's removal.
    """


def extract_audio_window(source_path, stream_order, start_seconds, length_seconds, out_path,
                          sample_rate, audio_filter=None):
    """`sample_rate` IS REQUIRED AND HAS NO DEFAULT, DELIBERATELY.

    This pinned "44100" until 2026-09-05. `mergeVideo.py:583-585` derives the pair's
    LOWER rate and clamps only when it is ABOVE 44100, so above 44100 the two agreed BY
    ACCIDENT -- the clamp landed both on the same number -- and below it this module
    UPSAMPLED one side and measured on a grid the consumer never uses.

    I had `comparison_grid = min(pair's lowest rate, 44100)` written down as a property
    of the PAIR for hours and quoted it to other agents; this extractor did not
    implement the rule I was citing. Found by vmsam-ci-build as a reading of the code.

    LIVE ON TWO FILES: ids 307 and 316 carry 32 kHz candidate streams. And
    `video.py:914-919` asserts, inside the function that computes the shared rate, that
    "a sub-44100 source ... this corpus does not contain". That is false of today's
    corpus -- reported as R21, not my module, not fixed here.

    MEASURED BEFORE CHANGING, both files both ways, paired, same probe positions:
        id 316   max |difference|  0.001 ms over 8 probes
        id 307   13 of 14 probes within 0.02 ms; ONE probe at 1.288 ms
        1.288 ms against a ~129 ms quantum is 1.0% OF ONE QUANTUM
    and the outlier is not a grid effect -- the 44100 side is out of step with ITS OWN
    NEIGHBOURS (t=20/40/80 read 0.018/0.018/-0.005), so it is one unstable correlation
    at one position.

    SO THIS IS NOT A CORRECTNESS FIX AND MUST NOT BE READ AS ONE. It is landed for
    INTERPRETABILITY: two instruments on different grids cannot validate each other --
    agreement would be luck and disagreement unattributable -- which suspended ci's
    cross-check of `offset_ms` against the pipeline's delays on exactly these two files.
    The Lead's ruling: the smallness of the delta is the argument FOR landing, because
    it resolves the risk side and leaves interpretability standing alone.

    NO DEFAULT: a missed call site must be a TypeError, not a silent return to 44100.

    ON `length_seconds` AND THE ONE-SECOND FLOOR BELOW, now that this is a SHARED entry point:
    the floor's own comment says it is chosen to sit two orders of magnitude under any window
    the LOCATOR asks for. The orchestrator asks for whole tracks, which is further above the
    floor still, so the guard keeps the same meaning for the new caller -- but a future caller
    that genuinely wants a sub-second window must change the floor deliberately rather than
    discover it as a refusal.

    ON `audio_filter`, ADDED 2026-09-22 FOR THE ORCHESTRATOR'S COMPARISON RESAMPLE, OPTIONAL AND
    DEFAULTING TO NOTHING. When it is None the command built below is BYTE-IDENTICAL to the one
    this function has always built -- which is the whole point of the default: the acceptance
    condition for the move that created this module is that a locator run stays byte-identical,
    and a parameter that changed the command even when unused would break exactly the comparison
    that proves the move was safe. It carries an ffmpeg `-af` filtergraph, and the only producer
    today is `repair_orchestrator.rate_resample_routing` (the re-prime at a confirmed factor,
    upstream of chimeric since ADDENDUM 21.6), which builds it through
    `merge_video_resample.build_speed_filter_chain` -- the pipeline's single asetrate authority,
    never a chain spelled out at a call site.

    THE OUTPUT IS NO LONGER `length_seconds` LONG WHEN A SPEED FILTER IS PASSED, AND THE CALLER
    OWNS THAT. `-ss`/`-t` sit BEFORE `-i`, so they bound what is READ from the source; a filter
    that changes the rate changes what is WRITTEN. A caller that hands a chain here and then tells
    fpcalc the input length would truncate exactly the tail the correction just restored, so the
    caller computes the corrected length from the EFFECTIVE ratio and passes that on. Said here
    because this function cannot check it: it never sees fpcalc.
    """
    # THE STRICT DECODE RIDES ON THE EXTRACTION (owner 2026-09-25 23:4x): `-xerror`,
    # `-reinit_filter 0` and the `-err_detect` flags make this decode the integrity check of
    # the span it reads, for free.
    # MEASURED on the Chainsaw VARYG AMZN cut: the corrupt E-AC-3 frame at 4 007.968 s fails
    # it (« frame CRC mismatch », rc 183); a clean cut, and 4 windows of a TrueHD stream, pass.
    # The command is therefore no longer byte-identical to the locator's (the acceptance
    # condition of the 2026-09-22 move, above): deliberately, by that order.
    cmd = [tools.software["ffmpeg"], "-v", "error", "-y", "-nostdin"] + STRICT_DECODE_FLAGS + [
           "-ss", f"{start_seconds:.6f}", "-t", f"{length_seconds:.6f}",
           "-i", source_path, "-map", f"0:{stream_order}",
           "-vn", "-ac", "1", "-ar", str(sample_rate)]
    # THE MUXER'S TIMESTAMPS ARE THE SAMPLE COUNT (`asetpts=N/SR/TB`, last in the chain): under
    # `-xerror` a muxer complaint is fatal, and a TrueHD track resampled to 44.1 kHz hands the WAV
    # muxer a DTS one tick backwards (MEASURED 2026-09-28, Fallout S01E03 BD master stream 1:
    # « Non-monotonic DTS; previous: 933339, current: 933338 », rc 234 at 21 s, a sound track
    # condemned `comparison_track_corrupt`). A muxer line is never the decoder's (26.9.12);
    # the WAV is byte-identical to the plain extraction's (md5 on 60 s of that track).
    cmd.extend(["-af", (audio_filter + "," if audio_filter else "") + "asetpts=N/SR/TB"])
    cmd.extend(["-acodec", "pcm_s16le", out_path])
    # *** THE EXIT CODE IS CHECKED AND THE OUTPUT IS NOT, AND THE FAILURE MODE IS ONE THAT
    # EXITS ZERO. `launch_cmdExt` raises on a non-zero return, so that half is covered --
    # but REPRODUCED HERE: seeking past the end of a source makes ffmpeg EXIT 0, WRITE A
    # 78-BYTE HEADER-ONLY WAV, AND SAY NOTHING ON STDERR. ffprobe then reports its duration
    # as N/A, which is `vmsam-ci`'s root cause: the per-stream files THIS FUNCTION WRITES
    # are what audio_sync chokes on, and the locator then cannot probe at all.
    # *** A COMMAND THAT SUCCEEDS IS NOT A COMMAND THAT PRODUCED SOMETHING. THE LAUNCHER CAN
    # ONLY CHECK THE CALL; ONLY THE CALLER KNOWS WHAT THE CALL WAS FOR. ***
    # IMMEDIATELY-PRE-CALL, NOT FUNCTION-ENTRY (owner's order, 2026-09-22,
    # via the Architect: `tools.launch_cmdExt` here is genuinely unbounded --
    # `Popen` + bare `communicate()`, no timeout at any layer). A line at
    # `extract_audio_window`'s own top would not name THIS call in flight if a
    # hang happens here specifically, since it runs many times per pair
    # and nothing upstream of this point can hang -- the log has to sit
    # where the block actually starts, not where the function does.
    # THE FILTER IS NAMED ONLY WHEN THERE IS ONE, so a filtered extraction can never be mistaken
    # for an untouched one by a reader of the same log (the prefix was `locator: _extract` until
    # the locator's removal, 2026-09-24).
    # A speed-corrected extraction that looked exactly like a raw one would make the single most
    # consequential fact about this call invisible.
    tools.dev_log(f"audio_extract: extract_audio_window ffmpeg call file={source_path} "
                  f"stream_order={stream_order} out_path={out_path}"
                  + (f" audio_filter={audio_filter}" if audio_filter else "") + "\n")
    # BOUNDED SINCE ADDENDUM 26.3 ("un timeout sur CHAQUE appel ffmpeg", `decoder_timeout`): the
    # launcher above was the unbounded one; the same non-zero-exit refusal is kept.
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
    # *** MY FIRST THRESHOLD WAS `size <= 44` ON THE ASSUMPTION OF A CANONICAL WAV HEADER, AND
    # IT DID NOT FIRE: ffmpeg WRITES A LARGER HEADER (LIST/INFO CHUNKS), SO THE HEADER-ONLY FILE
    # WAS 78 BYTES AND SAILED THROUGH. A guard whose threshold is wrong is a guard that runs and
    # reports nothing, which is the shape I have spent the night finding elsewhere. ***
    # ONE SECOND OF AUDIO AT THE REQUESTED RATE IS THE FLOOR. The caller only ever asks for whole
    # probe windows -- 60 s, or a tail start computed so the window fits -- so a file under one
    # second cannot be a legitimate short tail. THE NUMBER IS CHOSEN, NOT DERIVED: it is two
    # orders of magnitude below any window this module requests, which is why it cannot
    # false-refuse rather than because it is the true boundary.
    _floor = sample_rate * 2          # 1 s, 16-bit mono
    try:
        _written = os_stat(out_path).st_size
    except OSError:
        raise ExtractProducedNothing("extract wrote no file at the requested position")
    if _written < _floor:
        raise ExtractProducedNothing(
            f"extract wrote {_written} bytes, under {_floor} for one second at {sample_rate} Hz")


def streams_for(video_obj, language):
    """EVERY stream of the language, not just the first.

    The first version read `audios[language][0]` and returned one offset for the
    language, while the repair rebuilds every stream of it. Measured on error
    id 266: the candidate carries two jpn streams **27.5 ms apart**, so one of the
    two rebuilt tracks took an offset that far wrong. dev-2's post-mux verifier
    measured the same split from the produced file - 27.8 ms - independently.
    27.5 ms is 0.66 of a frame: under the quantum the merge snaps to, under
    mkvmerge's integer milliseconds, and under the verifier's 100 ms tolerance.
    It would have shipped silently.
    """
    audios = getattr(video_obj, "audios", None)
    if not audios or language not in audios:
        return []
    return [entry["StreamOrder"] for entry in audios[language]
            if entry.get("StreamOrder") is not None]
