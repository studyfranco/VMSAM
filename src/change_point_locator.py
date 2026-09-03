# -*- coding: utf-8 -*-
"""
change_point_locator.py — where two timelines diverge, and by how much.

Measurement half of campaign 2 objective 2. Written by `vmsam-dev-1`; the only
caller is `vmsam-dev-2`'s repair module, behind an off-by-default `config.ini`
`[features]` key. Interface agreed before either side was built:
`VMSAM_HELP_AI/dev-2/INTERFACE_dev1_dev2.md`.

It returns NUMBERS, never files. It never cuts, never writes a track, and never
touches `best_video.sameAudioMD5UseForCalculation`. The repair is dev-2's.

--------------------------------------------------------------------------------
THE SIGN CONVENTION, AS AN EQUATION SO IT CANNOT BE READ TWO WAYS

    candidate_time_ms = master_time_ms + candidate_offset_ms

To fill master position `m`, read the candidate at `m + candidate_offset_ms`.
A candidate missing content the master has gives a NEGATIVE offset.

Verified two ways before anything was built on it: derived from
`audioCorrelation.correlate`'s algebra, then measured on a constructed pair with a
known 1000 ms deletion, which read -1000. dev-2 confirmed the same convention
independently from the opposite direction.

--------------------------------------------------------------------------------
WHY THIS RE-MEASURES INSTEAD OF READING `delayFirstMethodAbort`

`SPEC_ZONE_A.MD` §1 offers the recorded delays so the correlation need not be
re-run. Measured, those numbers are RESIDUALS against a candidate that
`recreate_files_for_delay_adjuster` has already shifted by `delayUse/1000` s — an
arbitrary millisecond value, essentially never a whole multiple of the chromaprint
hop (4096/3/11025 = 123.840 ms). A fractional-hop shift misaligns the two
fingerprint grids and MANUFACTURES one-point steps:

    shift 0.000 ms (0.000 hops)   fidelity 0.968-0.975   residual +552
    shift 495.360 ms (4.000 hops) fidelity 0.967-0.974   residual    0
    shift 552.000 ms (4.457 hops) fidelity 0.918-0.936   residual  -138   <- pipeline
    shift 619.200 ms (5.000 hops) fidelity 0.967-0.974   residual -138

Three files whose logs record a one-point step measure as CONSTANT offsets when
read unshifted — including one independently verified frame-identical by eye with
a duration delta of exactly 0.000 s. So every measurement here extracts both sides
at the SAME absolute time, keeping the grids aligned.

--------------------------------------------------------------------------------
THE QUANTUM IS PER CALL, AND MILLISECONDS ARE NOT PORTABLE

`audioCorrelation.correlate` computes its own resolution per call:
`int(lengthFile / len(fingerprint_source) * 1000)`. It is NOT the chromaprint
frame, and NOT a constant 125 ms. Measured across 315 recorded refusals: 125 ms
for most, 124 ms for 27, 138 ms for three 190-second files. It is 125 ms only for
a shortest-audio duration between about 1120 s and 1824 s.

Consequence, measured on one pair at three window lengths: the same physical
offsets read 774/516, 810/540 and 900/600 ms — and 6 points and 4 points every
time. **The point count is portable; the millisecond value is not.** Both are
emitted; on disagreement, trust the points.
"""

from os import path, remove
from statistics import median

import tools
import audioCorrelation

# One chromaprint fingerprint item, seconds. frame 4096, hop frame/3, rate 11025.
CHROMAPRINT_HOP_SECONDS = 4096.0 / 3.0 / 11025.0

# Refusal thresholds. Every one comes from a measurement recorded in
# VMSAM_HELP_AI/dev-1/lab/, not from a guess.
#
# A pair with no shared content reads a flat 0.556-0.578 with delays scattered
# over 421 s and the sign flipping 8 times in 9 transitions (error id 237,
# verified different series by eye). Real same-content pairs read 0.87-0.99.
MIN_MEDIAN_FIDELITY = 0.70
MAX_DISTINCT_DELAYS = 4
MAX_SIGN_FLIPS = 2

# A one-point step has never survived direct absolute measurement: the three
# corpus files that recorded one all read constant when measured unshifted. Steps
# of two points and more reproduced on every file tested. So one point is treated
# as noise and merged into its neighbour.
MIN_STEP_POINTS = 2

# Window lengths for narrowing a change point, seconds. Two independent lengths
# must AGREE or the change point is reported unnarrowed — a disagreement means
# the instrument is not measuring what it thinks it is.
NARROW_WINDOWS = (30, 15)
NARROW_STEP_SECONDS = 5.0


def _log(message):
    if tools.dev:
        tools.logs.append(f"\t\t[change_point_locator] {message}\n")


def _extract(source_path, stream_order, start_seconds, length_seconds, out_path):
    """One mono 44.1 kHz PCM window. No loudnorm: measured to change nothing here,
    and it costs a filter pass per window."""
    cmd = [tools.software["ffmpeg"], "-v", "error", "-y", "-nostdin",
           "-ss", f"{start_seconds:.6f}", "-t", f"{length_seconds:.6f}",
           "-i", source_path, "-map", f"0:{stream_order}",
           "-vn", "-ac", "1", "-ar", "44100", "-acodec", "pcm_s16le", out_path]
    tools.launch_cmdExt(cmd)


def _correlate_at(master_path, master_stream, candidate_path, candidate_stream,
                  start_seconds, window_seconds, work_dir, tag):
    """Both sides extracted at the SAME absolute time, so the fingerprint grids
    stay aligned. Returns (fidelity, offset_points, delay_ms)."""
    master_window = path.join(work_dir, f"cpl_m_{tag}.wav")
    candidate_window = path.join(work_dir, f"cpl_c_{tag}.wav")
    try:
        _extract(master_path, master_stream, start_seconds, window_seconds, master_window)
        _extract(candidate_path, candidate_stream, start_seconds, window_seconds, candidate_window)
        return audioCorrelation.correlate(master_window, candidate_window, window_seconds)
    finally:
        for temporary in (master_window, candidate_window):
            try:
                remove(temporary)
            except OSError:
                pass


def _window_geometry(shortest_audio_seconds):
    """Reproduces video.generate_begin_and_length_by_segment and
    generate_cut_with_begin_length. Read from those functions, not assumed."""
    if shortest_audio_seconds > 540:
        begin_seconds = 120.0
        spacing = int((shortest_audio_seconds - 240) / (10 + 1))
    elif shortest_audio_seconds > 60:
        begin_seconds = 30.0
        spacing = int((shortest_audio_seconds - 45) / (10 + 1))
    else:
        begin_seconds = 0.0
        spacing = int(shortest_audio_seconds - 2 / (10 + 1))
    return begin_seconds, spacing, spacing * 2


def _stream_order_for(video_obj, language):
    audios = getattr(video_obj, "audios", None)
    if not audios or language not in audios or not audios[language]:
        return None
    return audios[language][0].get("StreamOrder")


def _audio_duration_seconds(video_obj, language):
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


def _sign_flips(values):
    non_zero = [v for v in values if v]
    return sum(1 for i in range(len(non_zero) - 1) if (non_zero[i] > 0) != (non_zero[i + 1] > 0))


def _plateaus(points_per_window):
    """Collapse the window sequence into runs, dropping runs shorter than a
    single window's worth of evidence and steps below MIN_STEP_POINTS."""
    runs = []
    for index, value in enumerate(points_per_window):
        if runs and runs[-1][0] == value:
            runs[-1][2] = index
        else:
            runs.append([value, index, index])
    merged = [runs[0]]
    for value, first, last in runs[1:]:
        if abs(value - merged[-1][0]) < MIN_STEP_POINTS:
            merged[-1][2] = last            # one-point wobble: absorb it
        else:
            merged.append([value, first, last])
    return merged


def _narrow(master_path, master_stream, candidate_path, candidate_stream,
            scan_from, scan_to, points_before, points_after, work_dir):
    """Bracket one change point. A window [p, p+W] reads the old plateau while
    more than half of it is pre-change, so last-old and first-new give
    T in (last_old + W/2, first_new + W/2). Two window lengths must agree."""
    brackets = []
    for window_seconds in NARROW_WINDOWS:
        previous = None
        last_old = None
        probe = scan_from
        while probe < scan_to:
            _, offset_points, _ = _correlate_at(
                master_path, master_stream, candidate_path, candidate_stream,
                probe, window_seconds, work_dir, f"n{int(probe)}_{window_seconds}")
            if previous is not None and -offset_points == points_after and last_old is not None:
                brackets.append((last_old + window_seconds / 2.0,
                                 probe + window_seconds / 2.0))
                break
            if -offset_points == points_before:
                last_old = probe
            previous = offset_points
            probe += NARROW_STEP_SECONDS
    if len(brackets) < 2:
        return None
    low = max(b[0] for b in brackets)
    high = min(b[1] for b in brackets)
    if low >= high:
        _log(f"window lengths disagree on a change point: {brackets}")
        return None
    return low, high


def locate_change_points(best_video, candidate_video, language, work_dir=None):
    """Locate where `candidate_video`'s timeline diverges from `best_video`'s.

    Returns the block described in INTERFACE_dev1_dev2.md, or **None**.

    None means *I could not measure* — never *the files are compatible*. dev-2
    maps it to `no_plan` and the refusal stands. Those are different answers and
    collapsing them is the mistake this campaign exists to avoid.
    """
    master_stream = _stream_order_for(best_video, language)
    candidate_stream = _stream_order_for(candidate_video, language)
    if master_stream is None or candidate_stream is None:
        _log(f"no {language} stream on one side; declining")
        return None

    master_duration = _audio_duration_seconds(best_video, language)
    candidate_duration = _audio_duration_seconds(candidate_video, language)
    if not master_duration or not candidate_duration:
        _log("audio duration unavailable; declining")
        return None

    shortest = min(master_duration, candidate_duration)
    begin_seconds, spacing, window_seconds = _window_geometry(shortest)
    if spacing <= 0:
        _log(f"degenerate geometry for {shortest:.1f}s of audio; declining")
        return None

    work_dir = work_dir or tools.tmpFolder
    master_path = best_video.filePath
    candidate_path = candidate_video.filePath

    fidelities, points, delays = [], [], []
    for index in range(10):
        fidelity, offset_points, delay_ms = _correlate_at(
            master_path, master_stream, candidate_path, candidate_stream,
            begin_seconds + index * spacing, window_seconds, work_dir, f"w{index}")
        fidelities.append(fidelity)
        points.append(-offset_points)          # points, in the stated convention
        delays.append(delay_ms)

    quantum_ms = int(round(delays[0] / points[0])) if points[0] else None
    if quantum_ms is None:
        for delay_ms, point in zip(delays, points):
            if point:
                quantum_ms = int(round(delay_ms / point))
                break
    if quantum_ms is None:
        quantum_ms = 125                        # every window read zero offset

    median_fidelity = median(fidelities)
    distinct = sorted(set(points))
    flips = _sign_flips(delays)
    _log(f"{language}: points={points} fid_median={median_fidelity:.3f} "
         f"quantum={quantum_ms}ms distinct={len(distinct)} flips={flips}")

    # --- refusals, each with a measured basis --------------------------------
    if median_fidelity < MIN_MEDIAN_FIDELITY:
        monotone = all(points[i] <= points[i + 1] for i in range(9)) or \
                   all(points[i] >= points[i + 1] for i in range(9))
        if monotone and len(distinct) > 2:
            # low fidelity with a monotone drift is a SPEED relation, which is
            # objective 3's problem and not a change point. Refusing it here on
            # fidelity alone would refuse the family the speed repair exists for.
            _log("low fidelity but monotone drift: speed relation suspected; declining")
        else:
            _log("fidelity at the floor with scattered delays: no shared content; declining")
        return None
    if len(distinct) > MAX_DISTINCT_DELAYS or flips > MAX_SIGN_FLIPS:
        _log(f"delays scattered ({len(distinct)} distinct, {flips} sign flips); declining")
        return None

    runs = _plateaus(points)
    segments = []
    change_points = []
    master_end_ms = int(round(shortest * 1000))

    if len(runs) == 1:
        segments.append({"master_start_ms": 0,
                         "master_end_ms": master_end_ms,
                         "candidate_offset_points": runs[0][0],
                         "candidate_offset_ms": runs[0][0] * quantum_ms})
    else:
        boundary_start = 0
        for run_index in range(len(runs) - 1):
            before_points = runs[run_index][0]
            after_points = runs[run_index + 1][0]
            scan_from = begin_seconds + runs[run_index][2] * spacing
            scan_to = begin_seconds + runs[run_index + 1][1] * spacing + window_seconds
            bracket = _narrow(master_path, master_stream, candidate_path,
                              candidate_stream, scan_from, scan_to,
                              before_points, after_points, work_dir)
            if bracket is None:
                # coarse fallback: the window data alone brackets to one spacing
                bracket = (scan_from + window_seconds / 2.0,
                           scan_to - window_seconds / 2.0)
                narrowed = False
            else:
                narrowed = True
            low_ms = int(round(bracket[0] * 1000))
            high_ms = int(round(bracket[1] * 1000))
            step_ms = (after_points - before_points) * quantum_ms
            # A DECREASING offset means the master holds content the candidate
            # lacks; the uncertain region must be widened by that much so the
            # segment after it is certainly past the missing span. An INCREASING
            # offset means the candidate holds extra content and the gap is the
            # bracket itself.
            gap_end_ms = high_ms + (-step_ms if step_ms < 0 else 0)
            segments.append({"master_start_ms": boundary_start,
                             "master_end_ms": low_ms,
                             "candidate_offset_points": before_points,
                             "candidate_offset_ms": before_points * quantum_ms})
            change_points.append({"bracket_low_ms": low_ms,
                                  "bracket_high_ms": high_ms,
                                  "step_points": after_points - before_points,
                                  "step_ms": step_ms,
                                  "gap_start_ms": low_ms,
                                  "gap_end_ms": gap_end_ms,
                                  "narrowed": narrowed})
            boundary_start = gap_end_ms
        segments.append({"master_start_ms": boundary_start,
                         "master_end_ms": master_end_ms,
                         "candidate_offset_points": runs[-1][0],
                         "candidate_offset_ms": runs[-1][0] * quantum_ms})

    return {"kind": "constant" if len(runs) == 1 else "piecewise_constant",
            "master_path": master_path,
            "candidate_path": candidate_path,
            "language": language,
            "quantum_ms": quantum_ms,
            "window_seconds": window_seconds,
            "spacing_seconds": spacing,
            "segments": segments,
            "change_points": change_points,
            "window_points": points,
            "window_fidelity": [round(f, 4) for f in fidelities],
            "median_fidelity": round(median_fidelity, 4)}
