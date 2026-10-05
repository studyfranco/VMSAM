"""Picture-only shift scan: a reporting side-channel over an audio-aligned plan.

A zone is one audio offset holding over a span of the master timeline. Inside such a zone
the picture can still sit at a different frame offset for a while (the candidate's own
edit, not an audio event). This module pairs every scene cut of the master, inside a zone,
with its candidate counterpart and reads the residual between their frame offset and the
zone's own audio offset; it never changes the plan.

Reuse, not new code: `video_offset_plan.decode_scenes_and_hashes` (whole-file scene cuts and
hashes, disk-cached -- a decode already on disk for this pair is read, not repeated) and
`video_offset_plan.match_changes` (pairs a master cut with its candidate cut over a lag
range, each one confirmed by a `frame_hash.align` window on both sides of the cut, never
crossing it, plus a content check). A cut `match_changes` returns has already cleared that
side-window, margin-gated bar on its own, so a run of one confirmed cut needs no second cut
to be believed.
"""

import bisect
import time
from fractions import Fraction

import frame_compare
import repair_pool
import video_offset_plan as vop
import tools

# A zone shorter than this is not worth scanning.
MIN_ZONE_SECONDS = 30.0

# Hypothesis band around the zone's own offset that a candidate cut must fall in to be
# considered its counterpart; `match_changes` adds its own per-side alignment reach on top.
HYPOTHESIS_REACH_FRAMES = 32

# Kept off each zone edge so a boundary cut is never read as this zone's own.
ZONE_EDGE_MARGIN_S = 1.0

# Repair-budget floor: under this much remaining time the scan is skipped outright.
MIN_REMAINING_BUDGET_S = 15.0


def _round_half_up(value):
    """Round an exact rational half away from zero."""
    value = Fraction(value)
    return int(value + Fraction(1, 2)) if value >= 0 else -int(-value + Fraction(1, 2))


def _frame_of_ms(ms, rate):
    """Master-grid frame nearest an exact-millisecond instant."""
    return _round_half_up(Fraction(str(ms)) * rate / 1000)


def _nominal_lag_frames(offset_ms, rate):
    """The zone's audio offset, rounded to whole frames the way the pipeline already does
    it (`frame_compare`/`scene_anchor`'s own `_nominal_shift_frames`) -- the same rounding
    on both sides of a residual keeps a non-integer-frame offset from reading as a shift."""
    return frame_compare._nominal_shift_frames(float(offset_ms), rate.numerator, rate.denominator)


def _zone_bounds(zone, rate):
    """(first, stop) master frames of a zone, margined off its own edges, or None when short."""
    start_ms, stop_ms = zone["master_start_ms"], zone["master_end_ms"]
    if float(stop_ms - start_ms) / 1000.0 < MIN_ZONE_SECONDS:
        return None
    margin = int(round(ZONE_EDGE_MARGIN_S * rate))
    first, stop = _frame_of_ms(start_ms, rate) + margin, _frame_of_ms(stop_ms, rate) - margin
    return (first, stop) if stop > first else None


def _residuals(matched, nominal_lag):
    """(m, residual) for each matched cut whose offset departs from the zone's own."""
    return [(m, d - nominal_lag) for m, d, _dist in matched if d != nominal_lag]


def _runs(residuals, rate):
    """Group consecutive same-residual cuts into runs: (start_s, end_s, residual, n_cuts)."""
    runs, m0, last, residual0, count = [], None, None, None, 0
    for m, residual in residuals + [(None, None)]:
        if residual is not None and residual == residual0:
            count, last = count + 1, m
            continue
        if count:
            runs.append((float(m0 / rate), float((last + 1) / rate), residual0, count))
        m0, residual0, count, last = m, residual, (1 if residual is not None else 0), m
    return runs


def _unpaired(cuts, matched_marks, reach):
    """Cuts with no counterpart within `reach` of any matched mark."""
    marks = sorted(matched_marks)
    out = 0
    for c in cuts:
        i = bisect.bisect_left(marks, c)
        near = (i < len(marks) and marks[i] - c <= reach) or (i > 0 and c - marks[i - 1] <= reach)
        if not near:
            out += 1
    return out


def _log_run(candidate_path, zone_index, run):
    start_s, end_s, residual, count = run
    tools.log_always(
        f"repair: picture_only_shift zone={zone_index} "
        f"master_s=[{round(start_s, 1)}, {round(end_s, 1)}] residual_frames={residual:+d} "
        f"cuts={count} audio=continuous for {candidate_path}\n")


def _scan_zone(zone, rate, m_cuts, m_hashes, m_coloured, c_cuts, c_hashes, c_coloured,
               candidate_path):
    """Scan one zone; return (runs logged, cuts confirmed, unpaired cuts)."""
    bounds = _zone_bounds(zone, rate)
    if bounds is None:
        return 0, 0, 0
    first, stop = bounds
    nominal_lag = _nominal_lag_frames(zone["offset_ms"], rate)
    m_zone_cuts = [m for m in m_cuts if first <= m < stop]
    if not m_zone_cuts:
        return 0, 0, 0
    lo, hi = nominal_lag - HYPOTHESIS_REACH_FRAMES, nominal_lag + HYPOTHESIS_REACH_FRAMES
    matched, _ambiguous, _total = vop.match_changes(m_hashes, m_zone_cuts, c_hashes, c_cuts,
                                                     lo, hi, m_coloured, c_coloured)
    residuals = _residuals(matched, nominal_lag)
    runs = _runs(residuals, rate)
    for run in runs:
        _log_run(candidate_path, zone["zone"], run)
    unpaired_master = len(m_zone_cuts) - len(matched)
    c_zone_cuts = [c for c in c_cuts if first + nominal_lag - hi <= c < stop + nominal_lag + hi]
    unpaired_candidate = _unpaired(c_zone_cuts, (m + d for m, d, _ in matched),
                                   HYPOTHESIS_REACH_FRAMES)
    if unpaired_master or unpaired_candidate:
        tools.log_always(
            f"repair: picture_only_shift_unpaired zone={zone['zone']} "
            f"master_only={unpaired_master} candidate_only={unpaired_candidate} "
            f"for {candidate_path}\n")
    return len(runs), len(matched), unpaired_master + unpaired_candidate


def scan_zones(zones, domain, master_obj, candidate_obj, candidate_path, work_dir,
               repair_deadline):
    """Scan audio-aligned zones for a picture-only shift and log what is found.

    Logs, never acts: `zones` is read only. Skipped outright when the repair's own budget is
    nearly spent; a budget that runs out mid-scan is logged with how far the scan got. Never
    raises -- a failure here is informational, not a repair outcome.
    """
    if (repair_deadline is not None
            and repair_deadline - time.monotonic() < MIN_REMAINING_BUDGET_S):
        tools.log_always(f"repair: picture_only_shift_summary skipped=low_budget "
                         f"zones_done=0 zones_total={len(zones)} for {candidate_path}\n")
        return
    started = time.monotonic()
    try:
        master_rate, candidate_rate = domain["master_rate"], domain["candidate_rate"]
        if master_rate != candidate_rate:
            # Scene cuts are only comparable as a frame count when both files share one
            # native rate; a speed-changed candidate is left to the resampling it already got.
            tools.log_always(f"repair: picture_only_shift_summary skipped=rate_mismatch "
                             f"zones_done=0 zones_total={len(zones)} for {candidate_path}\n")
            return
        m_info, _reason = vop.probe_video(master_obj.filePath)
        c_info, _reason = vop.probe_video(candidate_obj.filePath)
        m_duration_s = m_info["duration_s"] if m_info else None
        c_duration_s = c_info["duration_s"] if c_info else None
        # Master and candidate decoded concurrently on the shared repair pool; each decode is
        # disk-cached (vop.decode_scenes_and_hashes), so a decode already on disk for this file
        # in this work_dir (e.g. from video_offset_plan's own measure_video_offset) is read once.
        (m_cuts, m_hashes, m_coloured, _mc), (c_cuts, c_hashes, c_coloured, _cc) = \
            repair_pool.run_parallel([
                lambda: vop.decode_scenes_and_hashes(
                    master_obj.filePath, float(master_rate), m_duration_s, work_dir,
                    repair_deadline),
                lambda: vop.decode_scenes_and_hashes(
                    candidate_obj.filePath, float(candidate_rate), c_duration_s, work_dir,
                    repair_deadline),
            ])
    except vop.BudgetExceeded:
        tools.log_always(f"repair: picture_only_shift_summary skipped=budget_during_decode "
                         f"zones_done=0 zones_total={len(zones)} for {candidate_path}\n")
        return
    except Exception as error:                                           # noqa: BLE001
        tools.log_always(f"repair: picture_only_shift_summary skipped=error "
                         f"({type(error).__name__}: {str(error)[:200]}) for {candidate_path}\n")
        return
    total_runs = total_cuts = total_unpaired = zones_done = 0
    partial = False
    for zone in zones:
        if repair_deadline is not None and time.monotonic() > repair_deadline:
            partial = True
            break
        runs, cuts, unpaired = _scan_zone(zone, master_rate, m_cuts, m_hashes, m_coloured,
                                          c_cuts, c_hashes, c_coloured, candidate_path)
        total_runs, total_cuts, total_unpaired = (total_runs + runs, total_cuts + cuts,
                                                   total_unpaired + unpaired)
        zones_done += 1
    tools.log_always(
        f"repair: picture_only_shift_summary zones_done={zones_done} zones_total={len(zones)} "
        f"cuts_confirmed={total_cuts} runs={total_runs} unpaired={total_unpaired} "
        f"{'partial=1 ' if partial else ''}wall_s={round(time.monotonic() - started, 1)} "
        f"for {candidate_path}\n")
