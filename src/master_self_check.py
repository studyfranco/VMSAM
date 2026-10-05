"""Check the master against itself before any candidate is measured against it.

A master whose same-language audio tracks are offset from each other would make
every candidate look wrong. This module measures each same-language track pair
and returns numbers and a verdict; it never modifies a track.

Only the comparison language (the one delays are measured on) is checked; with
a single track in that language the check is inert.

Method: a ~30 s window at the same file timestamp in both streams, decoded to
16 kHz mono (precise enough against the 90 ms threshold) and cross-correlated by
FFT over +/-1 s. PCM is read from the ffmpeg pipe; no temporary file is written.
"""

import numpy as np

import tools
import repair_log

SR = 16000
WINDOW_SECONDS = 30.0
MAX_LAG_SECONDS = 1.0

# Fixed verdict token, parsed by `merge_video_repair.record()`.
VERDICT_DESYNC = "master_intertrack_desync"

# Correlation floor for any verdict. Below it the two tracks are not the same
# content and their lag has no referent: such pairs are logged, not verdicted.
MIN_CORRELATION_FOR_VERDICT = 0.9

# Desync threshold: the geometric midpoint between the largest lag seen on a
# healthy pair (~63 ms) and the smallest real defect (~130 ms).
MIN_LAG_MS_FOR_DESYNC = 90.0


def _log(message):
    """Dev-log one line with the `master_self_check: ` prefix."""
    tools.dev_log(f"master_self_check: {message}\n")


def _stream_order(track):
    """The ffprobe stream index (int) of a mediainfo audio dict, or None."""
    raw = track.get("StreamOrder")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _track_duration_seconds(track):
    """Track duration in seconds (mediainfo stores it as a string), or None."""
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
    """Start of the measurement window, in seconds.

    The window is centred on the shorter track, where content is most likely
    present in both tracks (no logos, lead-in or credits). A container-level
    offset does not depend on the position, so one window suffices. Falls back
    to the video duration when no audio track states one; None when nothing
    states a duration.
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
        # Shorter than the window: start at zero; `_pcm` enforces a minimum.
        return 0.0
    return max(0.0, shortest / 2.0 - window_seconds / 2.0)


def _pcm(file_path, stream_order, start_s, dur_s):
    """Decode one stream of a file to 16 kHz mono float PCM.

    None on any failure: a failed decode must not be read as silence.
    """
    cmd = [tools.software["ffmpeg"], "-v", "error", "-nostdin",
           "-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}", "-i", file_path,
           "-map", f"0:{stream_order}", "-vn", "-ac", "1", "-ar", str(SR),
           "-f", "s16le", "-"]
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
        # Under 2 s of audio cannot support a +/-1 s search.
        _log(f"_pcm only {len(stdout)} bytes from stream {stream_order} of "
             f"{file_path} -- too short to correlate")
        return None
    return np.frombuffer(stdout, dtype="<i2").astype(np.float64) / 32768.0


def _fft_cross_correlation(a, b, max_lag_samples):
    """Normalised FFT cross-correlation of two signals within +/-`max_lag_samples`.

    Returns `(lag_samples, correlation)`; the lag is a float (parabolic
    sub-sample refinement), positive when `a` is delayed relative to `b`.
    `(None, None)` when either signal is flat, since a NaN would read as healthy.
    The FFT is zero-padded past `2*n` to avoid circular wrap-around.
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
    # Negative lags wrap to the tail: stitch tail and head into -max..+max.
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
    """Human-readable track label: stream index plus title when present."""
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
    """Check whether the master's audio tracks in `language` agree with each other.

    Args:
        master_video_obj: the master video object (reads `audios[language]` only).
        language: the comparison language.

    Returns:
        dict with `verdict` (`master_intertrack_desync` or None), `inert` and
        `inert_reason`, `language`, `pairs` (one entry per pair), `worst` (the
        pair carrying the verdict) and `reason`. Never raises on measurement
        problems.
    """
    result = {"verdict": None, "inert": False, "inert_reason": None,
              "language": language, "pairs": [], "worst": None, "reason": None}

    audios = getattr(master_video_obj, "audios", None) or {}
    tracks = audios.get(language) or []

    if len(tracks) <= 1:
        # One track has nothing to disagree with; other languages are not checked.
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
                # Not the same content: logged, no verdict.
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
# Master conformity: the cheap `file_conformity.check_file` families (no strict decode). Any
# error-severity finding makes the master `master_nonconformant`.
VERDICT_NONCONFORMANT = "master_nonconformant"


def check_master_conformity(master_video_obj):
    """Run the cheap conformity checks on the master.

    Returns `{verdict, failed: [{name, numbers, sentence}], seconds, warnings}`;
    `verdict` is `master_nonconformant` or None (also None when unreadable).
    """
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
