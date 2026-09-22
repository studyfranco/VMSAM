# -*- coding: utf-8 -*-
"""
audio_extract.py -- the shared audio-extraction and stream-enumeration helpers, lifted out of
`change_point_locator.py` so they can outlive it.

WHY THIS FILE EXISTS, AND IT IS NOT A TIDY-UP. Four helpers that nothing about the probe grid
made locator-specific ended up private names inside the locator, and three OTHER modules already
reach across the underscore to call them (`zone_similarity_vector`, `banded_seed_alignment`,
`pal_speed_discriminator` -- measured, `grep -rn` over `src/`, 2026-09-22). The orchestrator
(RULING_20260922_ORCHESTRATOR_ARCHITECTURE.MD) removes the locator; removing it with these
functions still inside would take three unrelated modules down with it. So they MOVE, they do
not get copied and they do not get deleted -- `change_point_locator` keeps thin re-export shims
bound to the objects defined here, so every existing caller keeps calling the SAME function
object, not a second implementation that can drift.

WHAT IS DELIBERATELY UNCHANGED, BYTE FOR BYTE: the bodies, the thresholds, and the
`locator: _extract ffmpeg call ...` narration line. That prefix now names a module this code no
longer lives in, and it stays anyway for one stage: the acceptance condition for this move is
that a locator run on a real pair is byte-identical to the run before it, and renaming a line
that appears in every captured log would break exactly the comparison that proves the move was
safe. The rename belongs with the locator's own removal, where the line has a new owner to be
named after, not here where it would be an unmeasurable change riding along with a mechanical
one.

NEW PUBLIC NAMES, OLD PRIVATE ONES KEPT ALIVE BY THE SHIMS. A helper three modules import is
not private, and carrying the leading underscore into a module built to be shared would
enshrine the very confusion that put them behind one. The locator's `_extract`,
`_audio_duration_seconds`, `_streams_for` and `_all_audio_streams` are now aliases for
`extract_audio_window`, `audio_duration_seconds`, `streams_for` and `all_audio_streams`.
"""
from os import stat as os_stat

import tools


class ExtractProducedNothing(Exception):
    """ffmpeg exited 0 and produced no audio. A type I own, so the site tally can name it.

    *** NOT a generic Exception: `_probe` already catches everything and reports
    `extract_or_correlate_raised`, which would fold this into the correlator's failures.
    This is the one failure mode `vmsam-ci` traced to a root cause, and it deserves to be
    distinguishable from a correlation that ran and failed. ***

    MOVED HERE 2026-09-22, CLASS IDENTITY PRESERVED: `change_point_locator` binds its own
    `ExtractProducedNothing` name to THIS class object, so `except cpl.ExtractProducedNothing`
    at the locator's own probe site still catches what this module raises. A copy of the class
    would have compiled, passed every import, and caught nothing.
    """


def extract_audio_window(source_path, stream_order, start_seconds, length_seconds, out_path,
                          sample_rate):
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
    """
    cmd = [tools.software["ffmpeg"], "-v", "error", "-y", "-nostdin",
           "-ss", f"{start_seconds:.6f}", "-t", f"{length_seconds:.6f}",
           "-i", source_path, "-map", f"0:{stream_order}",
           "-vn", "-ac", "1", "-ar", str(sample_rate),
           "-acodec", "pcm_s16le", out_path]
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
    # `_extract`'s own top would not name THIS call in flight if a hang
    # happens here specifically, since `_extract` runs many times per pair
    # and nothing upstream of this point can hang -- the log has to sit
    # where the block actually starts, not where the function does.
    tools.dev_log(f"locator: _extract ffmpeg call file={source_path} "
                  f"stream_order={stream_order} out_path={out_path}\n")
    tools.launch_cmdExt(cmd)
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


def all_audio_streams(video_obj):
    """EVERY audio stream with its language, not only one language's.

    `streams_for` answers "the streams of language L". This answers "the streams",
    which is what a per-language pairing needs.
    """
    audios = getattr(video_obj, "audios", None) or {}
    out = []
    for lang, entries in audios.items():
        for entry in entries:
            order = entry.get("StreamOrder")
            if order is not None:
                out.append((order, lang))
    return sorted(out)


def audio_duration_seconds(video_obj, language):
    """The comparison language's own declared duration, off the video object's metadata.

    Returns None when the object carries no such reading. NONE MEANS "I COULD NOT MEASURE",
    never "zero" and never "the track is absent" -- the standing invariant, restated here
    because this function is now reachable from a second chain whose entry point cannot see the
    locator's own docstring saying it.
    """
    audios = getattr(video_obj, "audios", None)
    if not audios or language not in audios or not audios[language]:
        return None
    for key in ("Duration", "duration"):
        if key in audios[language][0]:
            try:
                return float(audios[language][0][key])
            except (TypeError, ValueError):
                pass
    return None


# THE OLD PRIVATE NAMES, ALIASED TO THE SAME OBJECTS. `change_point_locator` re-exports these,
# and three modules call them through it today. Aliases rather than wrappers: a wrapper would
# put a second frame in every traceback and a second place for a signature to drift.
_extract = extract_audio_window
_streams_for = streams_for
_all_audio_streams = all_audio_streams
_audio_duration_seconds = audio_duration_seconds
