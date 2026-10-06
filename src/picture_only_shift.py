"""Picture-only shift scan: a reporting side-channel over an audio-aligned plan.

A zone is one audio offset holding over a span of the master timeline. Inside such a zone
the picture can still sit at a different frame offset for a while (the candidate's own
edit, not an audio event). This module pairs every scene cut of the master, inside a zone,
with its candidate counterpart and reads the residual between their file-clock picture lag
(decoded-index lag plus each file's own video `start_time`) and the zone's own audio offset,
against the pair's own whole-file baseline (its own constant audio/picture relation, logged
once and never refused on); only a residual that departs from that baseline by a full frame,
over at least two consecutive matched cuts, is reported. It never changes the plan.

Reuse, not new code: `video_offset_plan.decode_scenes_and_hashes` (whole-file scene cuts and
hashes, disk-cached -- a decode already on disk for this pair is read, not repeated) and
`video_offset_plan.match_changes` (pairs a master cut with its candidate cut over a lag
range, each one confirmed by a `frame_hash.align` window on both sides of the cut, never
crossing it, plus a content check). A cut `match_changes` returns has already cleared that
side-window, margin-gated bar on its own, so a run of one confirmed cut needs no second cut
to be believed.
"""

import bisect
import statistics
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


def _search_window_frames(offset_ms, rate):
    """Rounded-to-frame search window centre for `match_changes`'s lag range only.

    Never used to judge a residual: `match_changes` needs an integer frame window and already
    reaches `HYPOTHESIS_REACH_FRAMES` past it, so rounding here only widens the search, it does
    not decide whether a cut's picture lag agrees with the audio."""
    return frame_compare._nominal_shift_frames(float(offset_ms), rate.numerator, rate.denominator)


def _frame_ms(rate):
    """Exact frame duration in ms for a native rate (Fraction frames/second)."""
    return 1000.0 / float(rate)


def _video_start_ms(info):
    """A probed video's `start_time` in ms, in file-clock units (0 when unprobed)."""
    return float(info["start_s"]) * 1000.0 if info else 0.0


def _picture_ms(d, frame_ms, c_start_ms, m_start_ms):
    """File-clock picture lag of a matched cut: the decoded-index lag plus each file's own
    video `start_time` -- `match_changes`'s `d` is an index lag, not a presentation-time one."""
    return d * frame_ms + (c_start_ms - m_start_ms)


def _zone_bounds(zone, rate):
    """(first, stop) master frames of a zone, margined off its own edges, or None when short."""
    start_ms, stop_ms = zone["master_start_ms"], zone["master_end_ms"]
    if float(stop_ms - start_ms) / 1000.0 < MIN_ZONE_SECONDS:
        return None
    margin = int(round(ZONE_EDGE_MARGIN_S * rate))
    first, stop = _frame_of_ms(start_ms, rate) + margin, _frame_of_ms(stop_ms, rate) - margin
    return (first, stop) if stop > first else None


def _cut_runs(matched, rate):
    """Group consecutive matched cuts sharing the same candidate lag `d` into runs.

    Every cut in a run reads the same file-clock picture lag (`d` alone fixes it), so grouping
    by `d` is grouping by residual without ever rounding the audio offset to decide it.

    Returns:
        [{start_s, end_s, d, count}], in master cut order.
    """
    runs, m0, last, d0, count = [], None, None, None, 0
    for m, d, _dist in list(matched) + [(None, None, None)]:
        if d is not None and d == d0:
            count, last = count + 1, m
            continue
        if count:
            runs.append({"start_s": float(m0) / float(rate),
                        "end_s": float(last + 1) / float(rate), "d": d0, "count": count})
        m0, d0, count, last = m, d, (1 if d is not None else 0), m
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


def _log_run(candidate_path, zone_index, run, deviation_ms, frame_ms):
    # `residual_frames` is a display-only rounding of the full-precision `deviation_ms` the
    # decision above was made on (never the other way round) -- kept so the existing report
    # renderer, which reads this field by name, still has an integer frame count to show.
    residual_frames = int(round(deviation_ms / frame_ms))
    tools.log_always(
        f"repair: picture_only_shift zone={zone_index} "
        f"master_s=[{round(run['start_s'], 1)}, {round(run['end_s'], 1)}] "
        f"residual_frames={residual_frames:+d} deviation_ms={round(deviation_ms, 3)} "
        f"cuts={run['count']} audio=continuous for {candidate_path}\n")


def _disagreement(zone, run, deviation_ms):
    """Build one descriptive dict for a confirmed picture-only-shift run (logged, never a
    decline, per the owner's 2026-10-06 ruling that the picture is the truth for its own zone).

    Candidate bounds carry the zone's own audio offset forward (the zone is audio-continuous
    by construction); there is no audio cut and no single video cut instant here, only a
    sustained picture residual, so both cut fields stay None. `picture_shift_ms` is the run's
    residual against the pair's own baseline, never the raw file-clock lag -- a whole-file
    constant offset is not a shift.
    """
    offset_s = float(zone["offset_ms"]) / 1000.0
    return {"zone": zone["zone"], "reason": "picture_only_shift",
           "master_start_s": run["start_s"], "master_end_s": run["end_s"],
           "candidate_start_s": run["start_s"] + offset_s,
           "candidate_end_s": run["end_s"] + offset_s,
           "audio_cut_s": None, "video_cut_s": None,
           "picture_shift_ms": deviation_ms,
           "frames_compared": run["count"]}


def _zone_matches(zone, rate, m_cuts, m_hashes, m_coloured, c_cuts, c_hashes, c_coloured,
                  candidate_path):
    """Match one zone's cuts; return (m_zone_cuts, matched) or (None, None) when too short/empty."""
    bounds = _zone_bounds(zone, rate)
    if bounds is None:
        return None, None
    first, stop = bounds
    m_zone_cuts = [m for m in m_cuts if first <= m < stop]
    if not m_zone_cuts:
        return None, None
    nominal_lag = _search_window_frames(zone["offset_ms"], rate)
    lo, hi = nominal_lag - HYPOTHESIS_REACH_FRAMES, nominal_lag + HYPOTHESIS_REACH_FRAMES
    matched, _ambiguous, _total = vop.match_changes(m_hashes, m_zone_cuts, c_hashes, c_cuts,
                                                     lo, hi, m_coloured, c_coloured)
    c_zone_cuts = [c for c in c_cuts if first + nominal_lag - hi <= c < stop + nominal_lag + hi]
    unpaired_master = len(m_zone_cuts) - len(matched)
    unpaired_candidate = _unpaired(c_zone_cuts, (m + d for m, d, _ in matched),
                                   HYPOTHESIS_REACH_FRAMES)
    if unpaired_master or unpaired_candidate:
        tools.log_always(
            f"repair: picture_only_shift_unpaired zone={zone['zone']} "
            f"master_only={unpaired_master} candidate_only={unpaired_candidate} "
            f"for {candidate_path}\n")
    return m_zone_cuts, matched


def scan_zones(zones, domain, master_obj, candidate_obj, candidate_path, work_dir,
               repair_deadline, language=None):
    """Scan audio-aligned zones for a picture-only shift and log what is found.

    `zones` is read only; the scan itself never changes the plan. Skipped outright when the
    repair's own budget is nearly spent; a budget that runs out mid-scan is logged with how
    far the scan got. Never raises -- a failure here is informational, not a repair outcome.

    Returns:
        A list of `owner_judgment` zone dicts (empty when no shift is confirmed), one per
        confirmed residual run across every zone scanned: the caller declines on this,
        before any plan built from these zones is applied.
    """
    if (repair_deadline is not None
            and repair_deadline - time.monotonic() < MIN_REMAINING_BUDGET_S):
        tools.log_always(f"repair: picture_only_shift_summary skipped=low_budget "
                         f"zones_done=0 zones_total={len(zones)} for {candidate_path}\n")
        return []
    started = time.monotonic()
    try:
        master_rate, candidate_rate = domain["master_rate"], domain["candidate_rate"]
        if master_rate != candidate_rate:
            # Scene cuts are only comparable as a frame count when both files share one
            # native rate; a speed-changed candidate is left to the resampling it already got.
            tools.log_always(f"repair: picture_only_shift_summary skipped=rate_mismatch "
                             f"zones_done=0 zones_total={len(zones)} for {candidate_path}\n")
            return []
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
        return []
    except Exception as error:                                           # noqa: BLE001
        tools.log_always(f"repair: picture_only_shift_summary skipped=error "
                         f"({type(error).__name__}: {str(error)[:200]}) for {candidate_path}\n")
        return []
    frame_ms = _frame_ms(master_rate)
    m_start_ms, c_start_ms = _video_start_ms(m_info), _video_start_ms(c_info)
    total_cuts = total_unpaired = zones_done = 0
    zone_runs = []  # [(zone, [run, ...])]
    all_residual_ms = []
    partial = False
    for zone in zones:
        if repair_deadline is not None and time.monotonic() > repair_deadline:
            partial = True
            break
        m_zone_cuts, matched = _zone_matches(zone, master_rate, m_cuts, m_hashes, m_coloured,
                                             c_cuts, c_hashes, c_coloured, candidate_path)
        zones_done += 1
        if m_zone_cuts is None:
            continue
        total_cuts += len(matched)
        total_unpaired += len(m_zone_cuts) - len(matched)
        runs = _cut_runs(matched, master_rate)
        zone_runs.append((zone, runs))
        for run in runs:
            residual_ms = (_picture_ms(run["d"], frame_ms, c_start_ms, m_start_ms)
                          - float(zone["offset_ms"]))
            all_residual_ms.extend([residual_ms] * run["count"])
    # The pair's own A/V relation: a residual shared by every matched cut, over the whole file,
    # is the two releases' own constant offset (codec delay, mux, start_time convention), not a
    # picture edit -- logged once as information, and subtracted before any run is judged.
    baseline_ms = statistics.median(all_residual_ms) if all_residual_ms else 0.0
    if all_residual_ms:
        tools.log_always(
            f"repair: picture_audio_constant_offset offset_ms={round(baseline_ms, 3)} "
            f"frame_ms={round(frame_ms, 3)} cuts_sampled={len(all_residual_ms)} "
            f"for {candidate_path}\n")
    total_runs = 0
    all_disagreements = []
    for zone, runs in zone_runs:
        for run in runs:
            if run["count"] < 2:
                continue
            residual_ms = (_picture_ms(run["d"], frame_ms, c_start_ms, m_start_ms)
                          - float(zone["offset_ms"]))
            deviation_ms = residual_ms - baseline_ms
            if abs(deviation_ms) < frame_ms:
                continue
            total_runs += 1
            _log_run(candidate_path, zone["zone"], run, deviation_ms, frame_ms)
            all_disagreements.append(_disagreement(zone, run, deviation_ms))
    tools.log_always(
        f"repair: picture_only_shift_summary zones_done={zones_done} zones_total={len(zones)} "
        f"cuts_confirmed={total_cuts} runs={total_runs} unpaired={total_unpaired} "
        f"{'partial=1 ' if partial else ''}wall_s={round(time.monotonic() - started, 1)} "
        f"for {candidate_path}\n")
    # Owner ruling 2026-10-06: the picture is the truth for a confirmed run -- logged for
    # visibility (the caller logs its own summary line), never turned into an `owner_judgment`
    # decline.
    return all_disagreements
