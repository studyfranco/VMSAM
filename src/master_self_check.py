"""
master_self_check.py -- the master, measured against ITSELF, before any
candidate is measured against the master.

RULING_20260922_MASTER_INTERTRACK_ADMISSION.MD, with its two addenda. The gap
that ruling names, verbatim in its own words: `merge_video_repair.py`'s
admission loop never inspects the master alone, `change_point_locator.py` picks
`master_streams[0]` unconditionally with no agreement check against
`master_streams[1:]`, and the existing master-defect detectors in
`merge_video_chimeric.py` are POST-BUILD -- unreachable for this whole
population, because the locator declines first (`coverage_incomplete`).

So a master whose own same-language audio tracks disagree by 131 ms is, today,
the silent reference every candidate is judged against. Eleven files in folder
86 were refused for a defect of the MASTER and not of the candidate. This
module is the missing measurement, and it runs BEFORE the judging starts.

WHAT IT DOES NOT DO, AND THAT IS THE POINT
------------------------------------------
It returns NUMBERS AND A VERDICT. It never cuts, never writes a track, never
repairs, never touches `best_video.sameAudioMD5UseForCalculation`, and never
decides what the caller does with the verdict -- the same discipline
`change_point_locator` states in its own docstring, for the same reason.

SCOPE, AND IT IS NARROW BY OWNER CONSTRAINT (ADDENDUM 1)
--------------------------------------------------------
ONE LANGUAGE: the comparison language -- the language the pipeline generates
its delays on (`common_language_use_for_generate_delay`, derived from
`tools.special_params["original_language"]` at `mergeVideo.py:2141`; the repair
chain receives that very language from `mergeVideo.remove_not_compatible_video`
since ADDENDUM 20 -- `repair_not_compatible_videos(..., language)`). The owner's rationale, carried here so it cannot be lost in a
refactor: THE DEFECT MATTERS EXACTLY WHERE THE DELAYS ARE MEASURED. Anything
else is exploration the pipeline does not need.

Concretely: this module NEVER enumerates `master.audios.keys()`. It is handed
one language and it looks at exactly `master.audios[language]`. If THAT list
has one entry, the check is INERT -- even when another language of the same
master carries a duplicate pair. That is not an oversight to be fixed later; it
is the constraint.

IMPLEMENTATION, OFF THE OBJECT THE CALLER ALREADY HAS (ADDENDUM 2)
------------------------------------------------------------------
No re-probing of structure the video object already knows. `video.py:125-128`
builds `self.audios[language]` as a list of the mediainfo track dicts, each
carrying `StreamOrder` -- which `video.py:85` proves is the ffprobe stream
index (`ffprobe_data[int(data['StreamOrder'])]`), and which the whole file
already uses as `-map 0:<StreamOrder>` (`video.py:285`, `video.py:1023`). So
the track set, and the map argument for each track, come straight off the
object. No `ffprobe`, no `mediainfo`, no second parse.

THE INSTRUMENT
--------------
A ~30 s window, taken at the SAME file timestamp from both tracks of a pair
(they are two streams of ONE container, so the same `-ss`/`-t` is an aligned
window BY CONSTRUCTION -- there is no alignment step to get wrong), decoded to
16 kHz mono, and cross-correlated by FFT over +/-1 s. That is the instrument
the ruling names, and it is the same shape as
`pal_pitch_confirmer._pcm` (16 kHz mono through
`tools.software["ffmpeg"]` and `tools.launch_cmdExt_no_test`, decode failure
returning None rather than silence).

IT DOES NOT REPRODUCE THE LEDGER TO THE MILLISECOND, and saying so here is
cheaper than letting the next reader discover it. The ledger's folder-86
points (130.31 / 131.38 / 131.69 / 132.31 ms) are all exact INTEGER SAMPLE
COUNTS AT 48 kHz -- 6255, 6306, 6321, 6351 -- so they were taken at native
rate. None of them is an integer at 16 kHz (131.69 ms is 2107.04 samples), and
this module's 16 kHz grid plus its own window position lands E05 at 131.33 ms.
A ~0.3-0.5 ms disagreement with the ledger is the RESOLUTION GAP, not a defect,
and it is more than two orders below the 90 ms decision it feeds
(`MIN_LAG_MS_FOR_DESYNC`) -- which is also why raising the extraction to 48 kHz
to close it would buy nothing this module can spend.

NO TEMPORARY FILE IS WRITTEN. 30 s x 16 kHz x 2 bytes is 960 KB per track;
it comes back on the ffmpeg pipe, exactly as `pal_pitch_confirmer._pcm` already
takes it. `tools.tmpFolder` is therefore never touched by this module -- the
work that would have gone there does not exist. Stated explicitly because the
dispatch that ordered this module named `tools.tmpFolder` as the place for the
extraction's temporaries: there are none to place.

numpy only, no scipy. `pal_pitch_confirmer` is the repo's existing FFT
instrument and it imports numpy alone (`np.fft.rfft`, `np.correlate`); scipy is
installed but nothing in `src/` correlates through it, and this module is not
the place to open a second convention.
"""

import numpy as np

import tools
import repair_log

SR = 16000
WINDOW_SECONDS = 30.0
MAX_LAG_SECONDS = 1.0

# THE TOKEN. Fixed vocabulary, `snake_case`, and it travels with the numbers
# that produced it -- `merge_video_repair.record()` parses the cause out of a
# closed-vocabulary PREFIX (`repair: <outcome> cause=<token> for <path>:`), so
# a token that varies is not a token.
VERDICT_DESYNC = "master_intertrack_desync"

# CORRELATION FLOOR for a verdict to be allowed AT ALL. Below it the two tracks
# are not the same content, and a lag between two different contents is not a
# desync measurement -- it is a number with no referent. See
# `MIN_LAG_MS_FOR_DESYNC` for what happens to those pairs (they are logged and
# NOT verdicted).
MIN_CORRELATION_FOR_VERDICT = 0.9

# 90 ms, RE-RULED 2026-09-22 (ADDENDUM 3), AND HERE IS WHERE IT COMES FROM.
# A threshold must come from a measured gap, never from a round number that felt
# safe (`merge_video_repair.verify_tolerance_ms` carries its own basis the same
# way). This one was 60 ms for part of a day; the bidirectional trigger below
# fired during the landing validation, and the Architect moved it.
#
# THE TWO MEASURED EDGES, EACH WITH ITS n AND ITS INSTRUMENT:
#
#   NEAR EDGE   63.2 ms   The largest value this instrument returns on the only
#                         healthy-labelled pair that exists: folder 32
#                         (Tougen Anki S01) `fre` stream 3 vs stream 4, swept
#                         across seven window positions (37.1 / 48.2 / 55.6 /
#                         57.2 / 58.4 / 61.0 / 63.2 ms). n = 1 FILE, and the
#                         spread is not noise -- see point 3 below.
#                         AND THAT PAIR IS PIPELINE-UNREACHABLE IN PRODUCTION:
#                         stream 4 is titled `Descriptive` / flagged
#                         `visual_impaired`, so `video.py:118-122` routes it to
#                         `.audiodesc` and never to `.audios`, which is the only
#                         dict this module reads. The real production margin is
#                         therefore LARGER than the 26.8 ms this edge implies.
#   FAR EDGE    130.31 ms The LOWEST MEASURED defect point, folder 86
#                         (Shuumatsu no Walkure S02), 9+ points all at or above
#                         it, corr > 0.96 throughout.
#
#   90 ms is the GEOMETRIC midpoint of those two edges -- sqrt(63.2 * 130.31)
#   = 90.75, taken down to 90. The convention is stated rather than assumed
#   because it is a choice: the arithmetic midpoint would be 96.75, and on a
#   quantity whose two edges differ by a factor of ~2 the geometric mean is the
#   one that splits the RATIO evenly (1.42x clear of the near edge, 1.45x under
#   the far one) instead of favouring the larger edge.
#
# *** THE REGION 63-130 ms IS UNSAMPLED, NOT EMPTY, AND THAT DISTINCTION IS THE
# POINT. No observation of either class has ever landed in it. A threshold
# placed in a gap that nothing has probed is a DECLARED BET, not a measured
# boundary, and the triggers below are what make the bet falsifiable. ***
#
# WHAT THE FIRST BASIS GOT WRONG, KEPT HERE SO THE STALE NUMBERS CANNOT COME
# BACK. The ruling's original point 3 read "healthy point 10-11 ms (n=1, folder
# 32 fre pair)" and "defect floor 126 ms (n=9)". Re-measured during the landing
# validation:
#
#   1. `forensic/mastershort_rebuild/REPORT.md:154-161`'s 10-11 ms IS A
#      PACKET-TAIL END-TIME DELTA (`ffprobe -show_entries
#      packet=pts_time,duration_time`, last PTS plus its duration) -- NOT a
#      cross-correlation lag. Nothing had ever run THIS instrument on that pair.
#      Run at the same midpoint window production uses: 57.23 ms at corr 0.982
#      (E02), 34.33 ms at corr 0.627 (E03). So the margin under the OLD 60 ms
#      floor was 2.8 ms, not 50, and "60 ms is >5x the healthy point" was never
#      true of this instrument -- it was ~1.05x. THE 10-11 ms FIGURE DOES NOT
#      DESCRIBE THIS MEASUREMENT AND MUST NOT REAPPEAR AS IF IT DID.
#   2. 126 ms was the BOTTOM OF A CITED RANGE, not a measured point, and the
#      written points number 7, not 9. The lowest value anyone actually
#      measured is 130.31 ms -- which is what the far edge now uses.
#   3. THE FOLDER-32 PAIR DRIFTS, which an end-time delta could not have shown:
#      37.1 -> 63.2 ms MONOTONE with window position, ~26 ms gained over 1100 s,
#      a rate relation of ~1.000024 between a master's OWN two tracks. That
#      signature is a FUTURE TOKEN CANDIDATE (Addendum 3), deliberately NOT
#      verdicted in this ship: this module reports the offset its one window
#      saw, and a rate relation is objective 3's problem.
#
# THE DEFECT SIDE HELD UNDER THE SAME SCRUTINY, and that is why only the
# healthy edge moved: folder 86 E05 swept at the same seven positions returns
# 131.33-132.33 ms at corr 0.953-0.991 -- 1.0 ms of spread over 18 minutes of
# programme. That is a PROPERTY OF THE FILE, not of the window.
#
# BIDIRECTIONAL TRIGGERS, RESTATED ON THE NEW VALUE (Addendum 3; the ruling's
# point 3 in force) so this constant cannot quietly ossify:
#   * ANY observation of EITHER class landing inside 63-130 ms MOVES THIS
#     NUMBER -- a healthy pair above 90 ms and a defective pair below it are the
#     same falsification, from opposite sides. This is the trigger that already
#     fired once, on the 60 ms value, during this module's own landing.
#   * A LARGER SAMPLE FAILING TO REPRODUCE EITHER EDGE moves it too. The near
#     edge is ONE FILE of a pair the pipeline cannot even form, and everything
#     built on it inherits that; the far edge is 9+ points but all from ONE
#     SERIES, which is not a distribution either.
MIN_LAG_MS_FOR_DESYNC = 90.0


def _log(message):
    """`tools.dev_log`, one indirection, so every line this module emits
    carries the same `master_self_check: ` prefix and cannot drift between
    call sites. Same shape as the `repair: ` / `chimeric: ` / `resample: `
    prefixes `tools.dev_log`'s own docstring describes."""
    tools.dev_log(f"master_self_check: {message}\n")


def _stream_order(track):
    """The ffprobe stream index for a mediainfo audio dict, or None.

    `int()` rather than trusting the type: mediainfo hands `StreamOrder` back
    as a STRING (`video.py:85` casts it before indexing `ffprobe_data`), and an
    f-string would happily build `-map 0:None` out of a missing key."""
    raw = track.get("StreamOrder")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _track_duration_seconds(track):
    """Seconds, or None. `Duration` is a string on the mediainfo dicts
    (`video.py:405` calls `float()` on it for exactly this reason)."""
    raw = track.get("Duration")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _window_start(track_a, track_b, master_video_obj,
                  window_seconds=WINDOW_SECONDS):
    """Where the ~30 s window starts, in seconds.

    THE MIDPOINT OF THE SHORTER TRACK, minus half the window -- deterministic,
    and it is the one position in an episode that is almost never an opening
    logo, a silent lead-in, or end credits. A window is worth having only if it
    carries content in BOTH tracks; a fixed small offset would sit in the cold
    open or the sponsor card on a meaningful share of real files.

    ONE WINDOW IS ENOUGH FOR A REAL DESYNC, AND THAT IS MEASURED. Folder 86
    E05 swept at 200/400/600/734/900/1100/1300 s returns 131.33-132.33 ms --
    1.0 ms of spread over 18 minutes. A container-level track offset does not
    move with the window, so sampling it seven times buys nothing.
    BUT SEE `MIN_LAG_MS_FOR_DESYNC` POINT 3: a pair related by a RATE rather
    than an offset DOES move with the window (folder 32 E02, 37 -> 63 ms
    across the same sweep), and on such a pair this single position is a
    sample and not the answer. That is objective 3's problem, this module
    reports what its one window saw, and the log carries the position so the
    number can be rechecked without rerunning anything.

    FALLBACK TO THE OBJECT'S OWN VIDEO DURATION when neither audio track states
    one. Measured, not assumed: mediainfo DOES fill `Duration` on the audio
    tracks of the folder-86 master (1498.581 / 1498.112 s), which is the
    normal case -- but a Duration-less audio track is a real shape in streaming
    remuxes, and `master_obj.video["Duration"]` is ALREADY the timeline this
    file trusts (`merge_video_repair.get_master_timeline_ms`). Reaching for it
    is reading structure the object already carries, which is what ADDENDUM 2
    asks for; re-probing the container would not be.

    None when nothing states a duration at all: the caller then declines to
    measure rather than guessing an offset, because a window placed past the
    end of the file decodes to nothing and "nothing decoded" would be
    indistinguishable from "no signal".
    """
    durations = [d for d in (_track_duration_seconds(track_a),
                             _track_duration_seconds(track_b)) if d is not None]
    if not durations:
        video_track = getattr(master_video_obj, "video", None) or {}
        fallback = _track_duration_seconds(video_track)
        if fallback is not None:
            durations = [fallback]
    if not durations:
        return None
    shortest = min(durations)
    if shortest <= window_seconds:
        # A track shorter than the window itself: start at zero and let the
        # decoder hand back whatever exists. `_pcm` enforces its own minimum.
        return 0.0
    return max(0.0, shortest / 2.0 - window_seconds / 2.0)


def _pcm(file_path, stream_order, start_s, dur_s):
    """Decode ONE stream of ONE file to 16 kHz mono float PCM.

    None on ANY failure. Ported in shape from `pal_pitch_confirmer._pcm`, whose
    own rule is carried here because it is the one that matters: A DECODE THAT
    DID NOT HAPPEN IS NOT SILENCE. A caller that cannot tell those apart will
    read a broken container as a perfectly correlated pair.

    `-map 0:<StreamOrder>` is what makes this different from
    `pal_pitch_confirmer`'s version, and it is the entire reason this module
    exists: the defect is BETWEEN two streams of one file, so picking the
    stream is not a detail.
    """
    cmd = [tools.software["ffmpeg"], "-v", "error", "-nostdin",
           "-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}", "-i", file_path,
           "-map", f"0:{stream_order}", "-vn", "-ac", "1", "-ar", str(SR),
           "-f", "s16le", "-"]
    # BEFORE THE CALL THAT CAN HANG, NAME THE FILE. Same order, same evening,
    # same reason as `merge_video_repair.py:3435`: `tools.launch_cmdExt_no_test`
    # has NO timeout, and when a container wedges ffmpeg the last line logged is
    # the only evidence of where the process went.
    _log(f"_pcm extracting stream {stream_order} of {file_path} "
         f"at {start_s:.3f}s for {dur_s:.3f}s")
    with repair_log.announced("master_self_check", "ffmpeg", file_path) as call:
        stdout, stderror, exit_code = tools.launch_cmdExt_no_test(cmd)
        call["exit"] = exit_code
    if exit_code != 0:
        _log(f"_pcm ffmpeg exit {exit_code} on stream {stream_order} of "
             f"{file_path}: {stderror[-400:]}")
        return None
    if len(stdout) < SR * 4:
        # Under 1 s of audio (2 bytes/sample, so SR*4 bytes is 2 s). A window
        # this short cannot support a +/-1 s search at all.
        _log(f"_pcm only {len(stdout)} bytes from stream {stream_order} of "
             f"{file_path} -- too short to correlate")
        return None
    return np.frombuffer(stdout, dtype="<i2").astype(np.float64) / 32768.0


def _fft_cross_correlation(a, b, max_lag_samples):
    """Normalised cross-correlation of two equal-length signals, by FFT,
    restricted to +/-`max_lag_samples`.

    Returns `(lag_samples, correlation)` -- `lag_samples` is a FLOAT (parabolic
    sub-sample refinement, the same three-point form `pal_pitch_confirmer
    ._spectral_ratio` uses on its own peak) and is SIGNED with this convention:

        lag > 0  =>  `a`'s content arrives LATER in the file than `b`'s,
                     i.e. `a` is delayed by `lag` relative to `b`.

    Stated because a sign convention that lives only in the reader's head is a
    bug waiting for its second reader. Derivation, so it can be rechecked
    without rerunning anything: `r[k] = sum_i a[i+k]*b[i]`; if `a[i] = b[i-d]`
    then `r[k] = sum_i b[i+k-d]*b[i]`, maximal at `k = d`.

    `(None, None)` when either signal is flat -- a constant signal has no
    correlation with anything, and dividing by its zero norm would hand back a
    NaN that compares False against every threshold and so would SILENTLY read
    as "healthy".

    The FFT length is padded past `2*n` so the circular correlation carries no
    wrap-around into the lag band we read.
    """
    n = min(len(a), len(b))
    if n <= 2 * max_lag_samples:
        return None, None
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return None, None
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft))
    full = np.fft.irfft(spectrum, nfft)
    # Positive lags sit at the head, negative lags wrap to the tail. Slice both
    # ends and stitch them into one band running -max_lag .. +max_lag.
    band = np.concatenate((full[nfft - max_lag_samples:],
                           full[:max_lag_samples + 1]))
    band = band / (norm_a * norm_b)
    peak = int(np.argmax(band))
    correlation = float(band[peak])
    lag_samples = float(peak - max_lag_samples)
    if 0 < peak < len(band) - 1:
        y0, y1, y2 = band[peak - 1], band[peak], band[peak + 1]
        denominator = y0 - 2 * y1 + y2
        if abs(denominator) > 1e-12:
            lag_samples += 0.5 * float(y0 - y2) / float(denominator)
    return lag_samples, correlation


def _describe(track):
    """A track named the way a reader can act on it: the stream index the file
    actually carries, plus its title when it has one."""
    stream_order = _stream_order(track)
    label = f"stream {stream_order}" if stream_order is not None else "stream ?"
    title = track.get("Title")
    if title:
        label += f" ({title})"
    return label


def check_master_intertrack(master_video_obj, language,
                            window_seconds=WINDOW_SECONDS,
                            max_lag_seconds=MAX_LAG_SECONDS,
                            min_correlation=MIN_CORRELATION_FOR_VERDICT,
                            min_lag_ms=MIN_LAG_MS_FOR_DESYNC):
    """Does this master agree with ITSELF on `language`?

    `language` is the COMPARISON language and nothing else -- see the module
    docstring's scope section. This function reads `master_video_obj.audios
    [language]` and no other key of that dict.

    Returns a dict, always, never raises for a measurement problem:

        verdict      `master_intertrack_desync`, or None. NOT-None means
                     terminal for this (master, language)'s repair attempts.
        inert        True when nothing was measured because there was nothing
                     to compare. NO EXTRACTION RUNS on an inert call -- that is
                     the byte-identical-to-today arm and it is checkable in the
                     log by the absence of any `_pcm extracting` line.
        inert_reason the named reason, when inert.
        language     echoed back, so a stored verdict cannot be misattributed.
        pairs        one entry per same-language track pair actually measured:
                     {a, b, stream_a, stream_b, lag_ms, correlation,
                      verdict, measured}.
        worst        the pair that carries the verdict, or None.
        reason       prose for the caller's terminal line, or None.

    The thresholds are parameters ONLY so the boundary can be driven with
    literals in a test. Production calls pass none of them: a repair
    conditioned on a parameter is not a repair (`WRITE_ZONES.MD` s4), and
    nothing in `src/` reads these from configuration.
    """
    result = {"verdict": None, "inert": False, "inert_reason": None,
              "language": language, "pairs": [], "worst": None, "reason": None}

    audios = getattr(master_video_obj, "audios", None) or {}
    tracks = audios.get(language) or []

    if len(tracks) <= 1:
        # THE INERT ARM, AND IT EXITS BEFORE ANY ffmpeg EXISTS. A master with
        # one track in the comparison language has nothing to disagree with
        # itself about; a genuinely healthy file must cost exactly what it
        # costs today. ADDENDUM 1 BINDS HERE: another language of this same
        # master may well carry a pair, and we do not look, on purpose.
        result["inert"] = True
        result["inert_reason"] = ("single_track_in_comparison_language"
                                  if len(tracks) == 1
                                  else "no_track_in_comparison_language")
        _log(f"INERT ({result['inert_reason']}) language={language} "
             f"tracks={len(tracks)} master={getattr(master_video_obj, 'filePath', '?')} "
             f"-- nothing extracted")
        return result

    master_path = getattr(master_video_obj, "filePath", None)
    if master_path is None:
        result["inert"] = True
        result["inert_reason"] = "master_object_carries_no_path"
        _log(f"INERT (master_object_carries_no_path) language={language} "
             f"tracks={len(tracks)} -- nothing extracted")
        return result

    max_lag_samples = int(round(max_lag_seconds * SR))
    _log(f"language={language} master={master_path} tracks={len(tracks)} "
         f"-- {len(tracks) * (len(tracks) - 1) // 2} pair(s) to compare, "
         f"window {window_seconds:.1f}s at 16 kHz mono, "
         f"search +/-{max_lag_seconds:.1f}s")

    worst = None
    for index_a in range(len(tracks)):
        for index_b in range(index_a + 1, len(tracks)):
            track_a, track_b = tracks[index_a], tracks[index_b]
            entry = {"a": _describe(track_a), "b": _describe(track_b),
                     "stream_a": _stream_order(track_a),
                     "stream_b": _stream_order(track_b),
                     "lag_ms": None, "correlation": None,
                     "verdict": None, "measured": False}
            result["pairs"].append(entry)

            if entry["stream_a"] is None or entry["stream_b"] is None:
                _log(f"language={language} pair {entry['a']} vs {entry['b']}: "
                     f"NO MEASUREMENT -- a track carries no usable StreamOrder")
                continue

            start_s = _window_start(track_a, track_b, master_video_obj,
                                    window_seconds)
            if start_s is None:
                _log(f"language={language} pair {entry['a']} vs {entry['b']}: "
                     f"NO MEASUREMENT -- no track and no video stream states a "
                     f"duration, so the window has no defensible position")
                continue

            signal_a = _pcm(master_path, entry["stream_a"], start_s, window_seconds)
            signal_b = _pcm(master_path, entry["stream_b"], start_s, window_seconds)
            if signal_a is None or signal_b is None:
                _log(f"language={language} pair {entry['a']} vs {entry['b']}: "
                     f"NO MEASUREMENT -- extraction failed at {start_s:.3f}s "
                     f"(a={'ok' if signal_a is not None else 'failed'}, "
                     f"b={'ok' if signal_b is not None else 'failed'})")
                continue

            lag_samples, correlation = _fft_cross_correlation(
                signal_a, signal_b, max_lag_samples)
            if lag_samples is None:
                _log(f"language={language} pair {entry['a']} vs {entry['b']}: "
                     f"NO MEASUREMENT -- a window with no energy has no "
                     f"correlation, and a NaN would read as healthy")
                continue

            lag_ms = lag_samples * 1000.0 / SR
            entry["lag_ms"] = round(lag_ms, 3)
            entry["correlation"] = round(correlation, 4)
            entry["measured"] = True

            if correlation <= min_correlation:
                # POINT 4 OF THE RULING, AND THE LOG SAYS SO IN WORDS RATHER
                # THAN LEAVING A READER TO INFER IT FROM A MISSING VERDICT.
                # The E11-E15 third-track family correlates at ~0.69: those
                # tracks are not the same CONTENT, which is a different defect
                # needing a different token (`master_intertrack_discordant`, or
                # whatever it is eventually named) and a population measured
                # properly first. Verdicting them on this instrument would be
                # an unmeasured claim wearing a measured one's numbers.
                _log(f"language={language} pair {entry['a']} vs {entry['b']}: "
                     f"window@{start_s:.1f}s lag {lag_ms:.2f} ms "
                     f"corr {correlation:.4f} -- "
                     f"NO VERDICT THIS ITERATION: correlation {correlation:.4f} "
                     f"is at or below the {min_correlation} floor, so these two "
                     f"tracks are not the same content and their lag has no "
                     f"referent. A discordant-content master token is a second "
                     f"iteration, once that family is measured properly "
                     f"(RULING_20260922_MASTER_INTERTRACK_ADMISSION.MD point 4). "
                     f"Logged, not verdicted.")
                continue

            if abs(lag_ms) > min_lag_ms:
                entry["verdict"] = VERDICT_DESYNC
                _log(f"language={language} pair {entry['a']} vs {entry['b']}: "
                     f"window@{start_s:.1f}s lag {lag_ms:.2f} ms "
                     f"corr {correlation:.4f} -- "
                     f"VERDICT {VERDICT_DESYNC} (|lag| > {min_lag_ms} ms at "
                     f"corr > {min_correlation})")
                if worst is None or abs(lag_ms) > abs(worst["lag_ms"]):
                    worst = entry
            else:
                _log(f"language={language} pair {entry['a']} vs {entry['b']}: "
                     f"window@{start_s:.1f}s lag {lag_ms:.2f} ms "
                     f"corr {correlation:.4f} -- "
                     f"no verdict, |lag| within the {min_lag_ms} ms floor "
                     f"(these two tracks agree)")

    if worst is not None:
        result["verdict"] = VERDICT_DESYNC
        result["worst"] = worst
        result["reason"] = (
            f"the master disagrees with ITSELF on {language}: its {worst['a']} "
            f"and {worst['b']} carry the same content offset by "
            f"{worst['lag_ms']:.2f} ms (FFT cross-correlation of a "
            f"{window_seconds:.0f} s 16 kHz mono window, +/-"
            f"{max_lag_seconds:.1f} s search, correlation "
            f"{worst['correlation']:.4f}) -- above the {min_lag_ms:.0f} ms "
            f"floor, so every delay measured through this master on "
            f"{language} inherits an offset that depends on which of its own "
            f"tracks was picked")
        _log(f"language={language} master={master_path} "
             f"TERMINAL for this language: {result['reason']}")
    else:
        measured = sum(1 for pair in result["pairs"] if pair["measured"])
        _log(f"language={language} master={master_path} "
             f"no verdict -- {measured}/{len(result['pairs'])} pair(s) "
             f"measured, none crossed both thresholds")
    return result


# ---------------------------------------------------------------------------------------------
# CONFORMITY AT MASTER ENTRY (ADDENDUM 31, "coutures à venir"; owner on id 691: « il doit planter
# par vérification »). The CHEAP families of `file_conformity.check_file` -- container, content
# extent, inter-track coherence and tag conflicts -- run on the master before anything else; the
# sampled strict decode is NOT run (its in-pipeline use is an owner decision still pending). Any
# error-severity finding is `master_nonconformant`, naming each failed check with its numbers:
# a master that fails its own verification cannot give a candidate a timeline.
VERDICT_NONCONFORMANT = "master_nonconformant"


def check_master_conformity(master_video_obj):
    """`{verdict, failed: [{name, numbers, sentence}], seconds, warnings}` -- `verdict` is
    `master_nonconformant` or None. A master the check cannot read at all is also None here (the
    measurement did not happen; the steps after it will refuse it by their own names)."""
    import time
    import file_conformity
    started = time.monotonic()
    path = getattr(master_video_obj, "filePath", None)
    result = {"verdict": None, "failed": [], "seconds": None, "warnings": []}
    if path is None:
        return result
    _log(f"conformity: master={path} -- cheap families (no strict decode)")
    report = file_conformity.check_file(path, threads=3, integrity=False, extent="gated",
                                        workdir=getattr(tools, "tmpFolder", None),
                                        log=lambda message: _log(message))
    result["seconds"] = round(time.monotonic() - started, 1)
    result["warnings"] = report.names("warning")
    result["extent_decoded"] = report.facts.get("extent_decoded")
    failed = [{"name": c.name, "numbers": c.numbers, "sentence": c.sentence}
              for c in report.checks if c.severity == "error" and c.name != "unreadable"]
    result["failed"] = failed
    if failed:
        result["verdict"] = VERDICT_NONCONFORMANT
    return result
