'''Guards for a large negative audio delay, where the comparison windows must start later than usual.

A negative delay larger than the comparison begin means the first video holds |delay| seconds
at its head that the second one does not reach. Every window of the second file shifted by the
delay would start before 0, so both files' windows move later instead.

When the second file is also shorter by about |delay| and the first cut agrees clearly worse than
the other cuts of the same pass, that head content is absent from the second file: a
constant-delay merge would silently leave it without the second file's tracks. The pair is then
refused with `head_content_missing`, whose message carries the measurements.'''

from math import ceil
from statistics import median

# Seconds the duration gap may fall short of |delay|: the tails of two releases of one episode
# (end card, trailing black, next-episode preview) commonly differ by a few seconds.
DURATION_GAP_TOLERANCE = 10.0
# How far the first cut's fidelity must fall below the median of the other cuts of the pass.
FIRST_CUT_FIDELITY_DROP = 0.10
# Extra lead given to the picture comparison windows, which scan a few frames before the delay.
PICTURE_SCAN_HEADROOM_MS = 1000


class head_content_missing(Exception):
    """The first video has head content the shorter second one lacks."""


def begin_inside_both_files(begin_in_second, delay):
    """Return `begin_in_second`, or the first whole second after it at which a window shifted by
    `delay` ms no longer starts before 0."""
    return float(max(float(begin_in_second), ceil(-float(delay) / 1000)))


def picture_begin_inside_both_files(begin_in_second, delay):
    """Return the begin of the picture comparison windows for `delay` ms: `begin_in_second`, or
    later when the shifted windows, scanned a few frames early, would start before 0."""
    return begin_inside_both_files(begin_in_second, float(delay) - PICTURE_SCAN_HEADROOM_MS)


def language_duration(video_obj, language):
    """Return the longest audio duration of `language` in `video_obj`, in seconds, or None."""
    durations = []
    for audio in video_obj.audios.get(language, []):
        try:
            durations.append(float(audio['Duration']))
        except (KeyError, TypeError, ValueError):
            pass
    return max(durations) if durations else None


def first_cut_drop(delay_fidelity_values):
    """Return (drop, first-cut fidelity, median of the other cuts) for the audio couple of a
    correlation pass whose first cut falls least below its other cuts, or None without data.

    Args:
        delay_fidelity_values (dict): audio couple -> list of (fidelity, window, delay) per cut.
    """
    best = None
    for results in delay_fidelity_values.values():
        fidelities = [result[0] for result in results if result is not None]
        if len(fidelities) < 2 or results[0] is None:
            continue
        first = fidelities[0]
        others = median(fidelities[1:])
        if best is None or others - first < best[0]:
            best = (others - first, first, others)
    return best


def check_head_content(begin_in_second, delay, duration_1, duration_2, delay_fidelity_values,
                       name_1="the first file", name_2="the second file"):
    """Raise head_content_missing when the begin had to move for `delay` ms, the first file is
    longer by at least |delay| - DURATION_GAP_TOLERANCE seconds and the first cut's fidelity is
    at least FIRST_CUT_FIDELITY_DROP below the median of the other cuts of the pass.

    Returns:
        tuple: (begin moved, duration gap in s or None, (drop, first, median) or None), for the log.
    """
    moved = begin_inside_both_files(begin_in_second, delay) != float(begin_in_second)
    gap = None
    if duration_1 is not None and duration_2 is not None:
        gap = duration_1 - duration_2
    drop = first_cut_drop(delay_fidelity_values)
    head_seconds = -float(delay) / 1000
    if (moved and gap is not None and gap >= head_seconds - DURATION_GAP_TOLERANCE
            and drop is not None and round(drop[0], 6) >= FIRST_CUT_FIDELITY_DROP):
        raise head_content_missing(
            f"Head content missing for {name_2} against {name_1}: the second file starts "
            f"{head_seconds:.3f} s into the first (delay {float(delay):.0f} ms), the first is "
            f"{gap:.2f} s longer (the limit is {head_seconds - DURATION_GAP_TOLERANCE:.2f} s), and "
            f"after re-aligning both files the first cut's fidelity is {drop[1]:.3f} against a "
            f"median of {drop[2]:.3f} for the other cuts, a drop of {drop[0]:.3f} (the limit is "
            f"{FIRST_CUT_FIDELITY_DROP:.2f}). The first {head_seconds:.1f} s of the first file are "
            f"missing from the second: a constant-delay merge would leave them without the second "
            f"file's tracks, so the pair is refused.")
    return moved, gap, drop
