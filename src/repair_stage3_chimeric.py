# -*- coding: utf-8 -*-
"""Stage 3 of the repair pipeline: chimeric processing, from audio holes to frame-exact cuts.

3a. holes: each couple's `banded_seed_alignment.b2_align` zones become holes (`holes_for_couple`),
    cross-checked between couples and united (`union_holes`, `cluster_holes`);
3b. the audio walk (`reference_walk`, `audio_walk`) refines each change point to the
    millisecond and gives the window searched around it;
3c. the picture decides: scene cuts and pHash anchors A and B with a forward/backward cross
    scan (`resolve_hole`, `scene_anchor`), a pixel re-read on flat frames, and the edges
    (`audio_edges`); `audio_transitions` places every cut on the master timeline.

Step logging and the shared constants live in `repair_orchestrator`; they are read at call time.
"""
from decimal import Decimal
from fractions import Fraction
import math
import statistics
import time

import banded_seed_alignment
import repair_pool
import tools


def _orch():
    import repair_orchestrator
    return repair_orchestrator


# ---------------------------------------------------------------------------
# Zones to holes
# ---------------------------------------------------------------------------

def derive_holes(zones, n_master, n_candidate, quantum_ms, candidate_quantum_ms):
    """Return the n + 1 raw gaps around n aligned zones, before classification.

    Gaps are emitted even when empty on both axes, since adjacent zones may still differ in
    offset (a cut with no slack). Point bounds are inclusive (`lo > hi` means empty);
    millisecond bounds are half-open on each axis's own quantum.
    """
    holes = []
    for gap_index in range(len(zones) + 1):
        before = zones[gap_index - 1] if gap_index > 0 else None
        after = zones[gap_index] if gap_index < len(zones) else None
        m_lo = (before[0][1] + 1) if before else 0
        m_hi = (after[0][0] - 1) if after else (n_master - 1)
        c_lo = (before[1][1] + 1) if before else 0
        c_hi = (after[1][0] - 1) if after else (n_candidate - 1)
        holes.append({
            "modality": _orch().MODALITY,
            "gap_index": gap_index,
            "touches_head": gap_index == 0,
            "touches_tail": gap_index == len(zones),
            "master_points": [m_lo, m_hi],
            "candidate_points": [c_lo, c_hi],
            "master_ms": [m_lo * quantum_ms, (m_hi + 1) * quantum_ms],
            "candidate_ms": [c_lo * candidate_quantum_ms,
                              (c_hi + 1) * candidate_quantum_ms],
            "master_span_seconds": max(0, m_hi - m_lo + 1) * quantum_ms / 1000.0,
            "candidate_span_seconds": (max(0, c_hi - c_lo + 1)
                                        * candidate_quantum_ms / 1000.0),
        })
    return holes


def classify_holes(holes, zones_detail, quantum_ms):
    """Classify each hole as head, interior, tail or spans_whole_file, with its offset step.

    The step (`offset_after - offset_before`) exists only for interior holes; None means
    "no such quantity", not zero. A hole touching both ends is `spans_whole_file` and the
    caller declines on it.
    """
    classified = []
    for hole in holes:
        entry = dict(hole)
        if hole["touches_head"] and hole["touches_tail"]:
            entry["kind"] = "spans_whole_file"
        elif hole["touches_head"]:
            entry["kind"] = "head"
        elif hole["touches_tail"]:
            entry["kind"] = "tail"
        else:
            entry["kind"] = "interior"
        entry["why_token"] = _orch().WHY_TOKEN.get(entry["kind"])
        gap_index = hole["gap_index"]
        before = zones_detail[gap_index - 1] if gap_index > 0 else None
        after_index = gap_index
        after = zones_detail[after_index] if after_index < len(zones_detail) else None
        entry["offset_before_points"] = before["offset_points"] if before else None
        entry["offset_after_points"] = after["offset_points"] if after else None
        if before is not None and after is not None:
            entry["step_points"] = after["offset_points"] - before["offset_points"]
            entry["step_ms"] = entry["step_points"] * quantum_ms
        else:
            entry["step_points"] = None
            entry["step_ms"] = None
        classified.append(entry)
    return classified


def merge_quantum_flicker(zones, zones_detail):
    """Give a zone one quantum off between two equal-offset neighbours their offset.

    Such a zone is the true offset falling between two quanta, not an edit. Works on copies.

    Returns:
        (zones, detail, merged), merged holding (master_ms, offset_before, offset_after) per
        flicker.
    """
    zones = [[list(zone[0]), list(zone[1])] for zone in zones]
    detail = [dict(entry) for entry in zones_detail]
    merged = []
    for index in range(1, len(detail) - 1):
        before, middle, after = (detail[index - 1]["offset_points"],
                                 detail[index]["offset_points"],
                                 detail[index + 1]["offset_points"])
        if before == after and abs(middle - before) == 1:
            merged.append((list(detail[index]["master_ms"]), middle, before))
            detail[index]["offset_points"] = before
    return zones, detail, merged


def coalesce_same_offset_zones(zones, zones_detail):
    """Merge adjacent zones at the same offset into one aligned region, after the flicker merge.

    After this, a hole means a change of offset rather than a stretch the aligner failed to
    match: a same-offset gap changes no plan. An edit that removes and adds the same duration
    inside one gap is invisible to any offset instrument.

    """
    zones, zones_detail, _flicker = merge_quantum_flicker(zones, zones_detail)
    out_zones, out_detail = [], []
    for zone, detail in zip(zones, zones_detail):
        if out_detail and detail["offset_points"] == out_detail[-1]["offset_points"]:
            previous_zone, previous_detail = out_zones[-1], out_detail[-1]
            previous_zone[0][1] = zone[0][1]
            previous_zone[1][1] = zone[1][1]
            for key in ("master_points", "candidate_points", "master_ms", "candidate_ms"):
                previous_detail[key][1] = detail[key][1]
            previous_detail["n_members"] += detail["n_members"]
            continue
        out_zones.append([list(zone[0]), list(zone[1])])
        detail = dict(detail)
        for key in ("master_points", "candidate_points", "master_ms", "candidate_ms"):
            detail[key] = list(detail[key])
        out_detail.append(detail)
    return out_zones, out_detail


def holes_for_couple(alignment):
    """Derive one couple's holes: flicker merge, coalesce, derive, classify, drop non-holes.

    Nearby holes are not merged here (`cluster_holes` groups them). Gaps empty on both axes
    and without a step are dropped.
    """
    _zones, _detail, flicker = merge_quantum_flicker(alignment.get("zones") or [],
                                                     alignment.get("zones_detail") or [])
    for master_ms, offset_before, offset_after in flicker:
        tools.log_always(f"repair: quantum_flicker_merged master_ms=[{round(master_ms[0], 1)}, "
                         f"{round(master_ms[1], 1)}] offset_points={offset_before}->{offset_after} "
                         f"quantum_ms={alignment.get('quantum_ms')} -- two +/-1-quantum holes, zero "
                         f"net step: one zone\n")
    zones, zones_detail = coalesce_same_offset_zones(alignment.get("zones") or [],
                                                      alignment.get("zones_detail") or [])
    if not zones:
        return []
    quantum_ms = alignment["quantum_ms"]
    candidate_quantum_ms = alignment.get("candidate_quantum_ms") or quantum_ms
    raw = derive_holes(zones, alignment["n_master"], alignment["n_candidate"],
                        quantum_ms, candidate_quantum_ms)
    classified = classify_holes(raw, zones_detail, quantum_ms)
    return [hole for hole in classified
            if hole["master_span_seconds"] > 0 or hole["candidate_span_seconds"] > 0
            or hole["step_points"]]


def edge_addition_seconds(holes):
    """Return the alignment's estimate of candidate content at the head and tail, in seconds.

    Logged only; the chimeric marker reads the fills actually written.
    """
    return sum(hole["candidate_span_seconds"] for hole in holes
               if hole["kind"] in ("head", "tail"))


def hole_on_file_clock(hole, couple, fold):
    """Move one couple's hole from its tracks' clocks to the file's clock.

    Args:
        hole: the couple's hole.
        couple: couple name, recorded in `members` for provenance.
        fold: `couple_start_delta_ms` output (`delta_ms`, `master_start_ms`,
            `candidate_start_ms`, `scale`). Offsets gain `delta_ms`, the master bracket
            `master_start_ms`, the candidate bracket `candidate_start_ms * scale`.
    """
    delta = fold["delta_ms"]
    master_shift = fold["master_start_ms"]
    candidate_shift = fold["candidate_start_ms"] * fold["scale"]
    quantum_ms = hole["quantum_ms"]
    entry = dict(hole)
    entry["master_ms"] = [hole["master_ms"][0] + master_shift,
                          hole["master_ms"][1] + master_shift]
    entry["candidate_ms"] = [hole["candidate_ms"][0] + candidate_shift,
                             hole["candidate_ms"][1] + candidate_shift]
    entry["offset_before_ms"] = (None if hole["offset_before_points"] is None
                                 else hole["offset_before_points"] * quantum_ms + delta)
    entry["offset_after_ms"] = (None if hole["offset_after_points"] is None
                                else hole["offset_after_points"] * quantum_ms + delta)
    entry["track_delay_delta_ms"] = delta
    entry["members"] = [{"couple": couple, "kind": hole["kind"],
                         "master_ms": [round(entry["master_ms"][0], 2),
                                       round(entry["master_ms"][1], 2)],
                         "step_ms": (None if hole["step_ms"] is None
                                     else round(hole["step_ms"], 3)),
                         "gap_index": hole["gap_index"]}]
    entry["offset_sources"] = [couple, couple]
    return entry


def _bounding_offsets(entry, members):
    """Set a union region's offsets from the members that bound it on the master axis.

    The step is the bounding member's own when one member gives both offsets, else the
    difference in ms rounded to the quantum.
    """
    first = min((m for m in members if m["offset_before_ms"] is not None),
                key=lambda m: m["master_ms"][0], default=None)
    last = max((m for m in members if m["offset_after_ms"] is not None),
               key=lambda m: m["master_ms"][1], default=None)
    entry["offset_before_ms"] = (None if entry["kind"] in ("head", "spans_whole_file")
                                 or first is None else first["offset_before_ms"])
    entry["offset_after_ms"] = (None if entry["kind"] in ("tail", "spans_whole_file")
                                or last is None else last["offset_after_ms"])
    entry["offset_before_points"] = (None if entry["offset_before_ms"] is None
                                     else first["offset_before_points"])
    entry["offset_after_points"] = (None if entry["offset_after_ms"] is None
                                    else last["offset_after_points"])
    entry["track_delay_delta_ms"] = (first or last or members[0])["track_delay_delta_ms"]
    entry["offset_sources"] = [None if entry["offset_before_ms"] is None
                               else first["offset_sources"][0],
                               None if entry["offset_after_ms"] is None
                               else last["offset_sources"][1]]
    if entry["offset_before_ms"] is None or entry["offset_after_ms"] is None:
        entry["step_ms"], entry["step_points"] = None, None
    elif first is last:
        entry["step_ms"], entry["step_points"] = first["step_ms"], first["step_points"]
    else:
        entry["step_ms"] = entry["offset_after_ms"] - entry["offset_before_ms"]
        entry["step_points"] = _round_half_up(
            Fraction(str(entry["step_ms"])) / Fraction(str(entry["quantum_ms"])))


def _widen(target, member):
    """Return `target` (a union region) widened to cover `member`, derived fields recomputed."""
    members = target["_members"] + [member]
    entry = dict(target)
    entry["_members"] = members
    entry["master_ms"] = [min(m["master_ms"][0] for m in members),
                          max(m["master_ms"][1] for m in members)]
    entry["candidate_ms"] = [min(m["candidate_ms"][0] for m in members),
                             max(m["candidate_ms"][1] for m in members)]
    entry["touches_head"] = any(m["touches_head"] for m in members)
    entry["touches_tail"] = any(m["touches_tail"] for m in members)
    entry["kind"] = ("spans_whole_file" if entry["touches_head"] and entry["touches_tail"]
                     else "head" if entry["touches_head"]
                     else "tail" if entry["touches_tail"] else "interior")
    entry["why_token"] = _orch().WHY_TOKEN.get(entry["kind"])
    entry["master_span_seconds"] = (entry["master_ms"][1] - entry["master_ms"][0]) / 1000.0
    entry["candidate_span_seconds"] = max(
        0.0, (entry["candidate_ms"][1] - entry["candidate_ms"][0]) / 1000.0)
    _bounding_offsets(entry, members)
    entry["members"] = [record for m in members for record in m["members"]]
    entry["couples"] = sorted({record["couple"] for record in entry["members"]})
    entry["union_of"] = len(members)
    return entry


def union_holes(per_couple_holes):
    """Unite the holes of all couples into search regions on the file's clock.

    Each hole joins the nearest region within HOLE_MERGE_WINDOW_SECONDS that holds no hole of
    its own couple (the same event seen by another couple), else starts a region. Two holes of
    one couple are never united. Overlapping regions are then merged. Regions only bound the
    search; the plan is written by the resolved frames.

    Args:
        per_couple_holes: one list of `hole_on_file_clock` holes per couple, in couple order.

    Returns:
        The regions sorted by master start; a single couple's holes come back unchanged.
    """
    union = []
    for couple_holes in per_couple_holes:
        for hole in couple_holes:
            couple = hole["members"][0]["couple"]
            best, best_gap = None, None
            for index, region in enumerate(union):
                if couple in region["couples"]:
                    continue
                gap = max(hole["master_ms"][0] - region["master_ms"][1],
                          region["master_ms"][0] - hole["master_ms"][1], 0.0)
                if gap < _orch().HOLE_MERGE_WINDOW_SECONDS * 1000.0 and (best is None or gap < best_gap):
                    best, best_gap = index, gap
            if best is None:
                union.append(dict(hole, _members=[hole], couples=[couple], union_of=1))
            else:
                union[best] = _widen(union[best], hole)
    union.sort(key=lambda region: (region["master_ms"][0], region["master_ms"][1]))
    settled = []
    for region in union:
        if settled and region["master_ms"][0] < settled[-1]["master_ms"][1]:
            merged = settled[-1]
            for member in region["_members"]:
                merged = _widen(merged, member)
            settled[-1] = merged
            continue
        settled.append(region)
    for region in settled:
        del region["_members"]
    return settled


def cluster_holes(holes):
    """Group holes closer than the resolver's reach so they share one scene-detection pass.

    Each hole keeps its own bounds and step. Holes of a multi-hole cluster get `cluster_id`,
    `cluster_window` (shared extraction window) and `scan_cache` (shared memo).

    Returns:
        The clusters, each with its member indices.
    """
    clusters = []
    for index, hole in enumerate(holes):
        if clusters and (hole["master_ms"][0] - holes[clusters[-1]["members"][-1]]["master_ms"][1]
                         < _orch().HOLE_MERGE_WINDOW_SECONDS * 1000.0):
            clusters[-1]["members"].append(index)
        else:
            clusters.append({"members": [index]})
    for number, cluster in enumerate(clusters):
        members = [holes[index] for index in cluster["members"]]
        cluster["cluster_id"] = number
        if len(members) > 1:
            offsets = [value for hole in members
                       for value in (hole["offset_before_ms"], hole["offset_after_ms"])
                       if value is not None]
            window = {"bracket_ms": (members[0]["master_ms"][0], members[-1]["master_ms"][1]),
                      "offsets_ms": (min(offsets), max(offsets))}
            cache = {}
            for hole in members:
                hole["cluster_id"] = number
                hole["cluster_window"] = window
                hole["scan_cache"] = cache
    return clusters


# ---------------------------------------------------------------------------
# Step 4: frame-exact hole resolution
# ---------------------------------------------------------------------------

def _round_half_up(value):
    """Round an exact rational half away from zero."""
    value = Fraction(value)
    return int(value + Fraction(1, 2)) if value >= 0 else -int(-value + Fraction(1, 2))


def _exact_video_rate(video_obj):
    """Return the file's frame rate as an exact rational.

    Reads MediaInfo `FrameRate_Num`/`FrameRate_Den`, then ffprobe `r_frame_rate`. Never the
    decimal FrameRate: Fraction("23.976") is not 24000/1001.

    Returns:
        (Fraction, source) or (None, reason).
    """
    video = getattr(video_obj, "video", None) or {}
    try:
        num, den = video.get("FrameRate_Num"), video.get("FrameRate_Den")
        if num not in (None, "") and den not in (None, ""):
            rate = Fraction(int(str(num)), int(str(den)))
            if rate > 0:
                return rate, "mediainfo_num_den"
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    try:
        import scene_anchor
        rate, reason = scene_anchor._probe_frame_rate(video_obj.filePath)
    except Exception as error:                                           # noqa: BLE001
        return None, f"probe_raised:{type(error).__name__}"
    if rate is None:
        return None, f"ffprobe_r_frame_rate:{reason}"
    return rate, "ffprobe_r_frame_rate"


def _video_duration_ms(video_obj):
    """Return the video stream's duration in ms as a Decimal, or None when unreadable."""
    try:
        return Decimal(str(video_obj.video["Duration"])) * Decimal("1000")
    except Exception:                                                    # noqa: BLE001
        return None


def frame_domain(master_obj, candidate_obj, speed_factor):
    """Compute the pair's frame domain: how an alignment millisecond maps to each file's frames.

    Master frames are on the master's exact grid. Candidate alignment milliseconds are
    master-equivalent time; at a speed factor r the native candidate frame is
    equivalent_ms / r on the candidate's exact rate. r is the exact rational, not the audio
    filter's effective ratio.

    Returns:
        (domain, None) or (None, reason).
    """
    master_rate, master_source = _exact_video_rate(master_obj)
    if master_rate is None:
        return None, f"master_grid_unmeasured:{master_source}"
    candidate_rate, candidate_source = _exact_video_rate(candidate_obj)
    if candidate_rate is None:
        return None, f"candidate_grid_unmeasured:{candidate_source}"
    master_timeline_ms = _video_duration_ms(master_obj)
    if master_timeline_ms is None:
        return None, "master_timeline_unmeasured"
    candidate_raw_ms = _video_duration_ms(candidate_obj)
    scale = None
    if speed_factor is not None and speed_factor != 1:
        scale = Fraction(speed_factor)
    video_ratio = candidate_rate / master_rate
    return {
        "master_rate": master_rate, "master_rate_source": master_source,
        "candidate_rate": candidate_rate, "candidate_rate_source": candidate_source,
        "time_scale": scale,
        "video_rate_ratio": video_ratio,
        "video_ratio_matches_speed_factor": (None if scale is None else video_ratio == scale),
        "master_timeline_ms": master_timeline_ms,
        "candidate_raw_duration_ms": candidate_raw_ms,
        "candidate_equivalent_duration_ms": (
            None if candidate_raw_ms is None
            else candidate_raw_ms * (Decimal(scale.numerator) / Decimal(scale.denominator)
                                     if scale is not None else Decimal(1))),
        "frame_ms": Fraction(1000) / master_rate,
    }, None


def _exact_ms_of_frame(frame, domain):
    """Return a master frame's start time in ms as a six-decimal string (None passes through)."""
    if frame is None:
        return None
    return f"{float(Fraction(frame) * 1000 / domain['master_rate']):.6f}"


def _audio_offsets(hole):
    """Return the hole's audio offsets on the file's clock, in unrounded milliseconds.

    They are precise to one fingerprint quantum only; plan application refines them.
    """
    quantum_ms = hole["quantum_ms"]
    return {
        "audio_offset_before_ms": (None if hole.get("offset_before_ms") is None
                                   else round(hole["offset_before_ms"], 3)),
        "audio_offset_after_ms": (None if hole.get("offset_after_ms") is None
                                  else round(hole["offset_after_ms"], 3)),
        "audio_offset_precision_ms": round(quantum_ms, 3),
        "track_delay_delta_ms": hole.get("track_delay_delta_ms"),
    }


def couple_start_delta_ms(master_obj, candidate_obj, language, master_stream, candidate_stream,
                          speed_factor):
    """Return the shift from track-relative offsets to the file's clock.

    A container delay puts a track's time zero at its `start_time` on the file's clock, so the
    file-clock offset is offset + candidate_start * r - master_start (r = speed factor). A
    stream without a readable start reads 0, as in the assembly's `atrim`.

    Returns:
        (delta_ms, master_start_ms, candidate_start_ms)
    """
    import merge_video_chimeric

    def audio_of(video_obj, stream):
        for entry in (getattr(video_obj, "audios", None) or {}).get(language) or []:
            if str(entry.get("StreamOrder")) == str(stream):
                return entry
        return None

    scale = _orch()._decimal(Fraction(speed_factor)) if speed_factor not in (None, 1) else Decimal(1)
    master_start = merge_video_chimeric.get_stream_start_ms(audio_of(master_obj, master_stream))
    candidate_start = merge_video_chimeric.get_stream_start_ms(
        audio_of(candidate_obj, candidate_stream))
    return candidate_start * scale - master_start, master_start, candidate_start


def _master_frame_of_ms(ms, domain):
    return _round_half_up(Fraction(str(ms)) * domain["master_rate"] / 1000)


def _candidate_native_frame(equivalent_frame, domain):
    """Convert a master-grid equivalent frame index to the candidate's native frame number."""
    if equivalent_frame is None:
        return None
    scale = domain["time_scale"] or Fraction(1)
    return _round_half_up(Fraction(equivalent_frame) / domain["master_rate"] / scale
                          * domain["candidate_rate"])


def _candidate_native_frame_of_ms(equivalent_ms, domain):
    scale = domain["time_scale"] or Fraction(1)
    return _round_half_up(Fraction(str(equivalent_ms)) / 1000 / scale * domain["candidate_rate"])


def _shift_search_frames(domain, quantum_ms):
    """Return the anchors' shift-search half-width in frames.

    One fingerprint quantum of offset uncertainty, in frames rounded up, plus one frame of
    extraction labelling.
    """
    return int(math.ceil(Fraction(str(quantum_ms)) / domain["frame_ms"])) + 1


def _declined(hole, cause_detail, evidence=None, **extra):
    outcome = {"modality": _orch().MODALITY, "status": _orch().HOLE_DECLINED, "kind": hole["kind"],
               "why_token": hole["why_token"], "cause": "hole_resolution_declined",
               "resolver_reason": cause_detail, "evidence": evidence}
    outcome.update(extra)
    return outcome


# Neighbour shifts re-checked by raw luma MAD after an anchor seats: a fade (a luma ramp)
# barely separates two adjacent lags on pHash -- `frame_hash.ALIGN_MARGIN` can pass at a
# margin as thin as 0.058 -- while the raw picture does not share that blindness.
ANCHOR_MAD_REACH = 1
# A neighbour lag must read lower by at least this many luma levels, averaged over the
# anchor window, to override the shift pHash picked.
ANCHOR_MAD_MARGIN_LEVELS = 2.0
# The raw-MAD re-check only applies where pHash is known to be blind: a fade (a luma ramp
# moving gradually over most of the window), never a window that merely spans a wide range
# because of one hard step followed by a near-flat run (a cut, not a fade: CASE id 695 --
# 183.5 171.6 171.7 171.6 169.9 ... one -11.9 step then noise of a few tenths of a level).
# A frame-to-frame step must reach this many luma levels to count as part of the ramp.
ANCHOR_FADE_MIN_STEP_LEVELS = 2.0
# ...and at least this fraction of the window's steps must clear it (gradual, not a single
# cut); the overall window must also cover at least this many levels end to end.
ANCHOR_FADE_MIN_GRADUAL_FRACTION = 0.7
ANCHOR_FADE_MIN_RANGE_LEVELS = 10.0


def _is_fade_window(master_luma):
    """True when `master_luma` (one float per frame, in frame order) reads as a fade: most
    of its frame-to-frame steps move by at least ANCHOR_FADE_MIN_STEP_LEVELS, in the same
    direction with at most one reversal (a fade-out into a fade-in, 747's shape, still
    counts), over a window spanning at least ANCHOR_FADE_MIN_RANGE_LEVELS end to end. This
    is the signature pHash is blind to (CASE_fade_anchor_shift_off_by_one) -- a single hard
    step followed by a near-flat run is a cut, not a fade, however wide its own range."""
    if len(master_luma) < 3:
        return False
    if max(master_luma) - min(master_luma) < ANCHOR_FADE_MIN_RANGE_LEVELS:
        return False
    diffs = [b - a for a, b in zip(master_luma[:-1], master_luma[1:])]
    gradual = [d for d in diffs if abs(d) >= ANCHOR_FADE_MIN_STEP_LEVELS]
    if len(gradual) < len(diffs) * ANCHOR_FADE_MIN_GRADUAL_FRACTION:
        return False
    signs = [1 if d > 0 else -1 for d in gradual]
    reversals = sum(1 for a, b in zip(signs[:-1], signs[1:]) if a != b)
    return reversals <= 1


def _anchor_mad_confirmed_shift(domain, master_obj, candidate_obj, first, count, shift):
    """Confirm one anchor's shift against its immediate neighbours on raw luma MAD, but
    only over a fade window (`_is_fade_window`) -- pHash's known blind spot. Elsewhere the
    pHash shift is kept untouched: raw MAD has no such blindness to correct, and overriding
    it there risks trading a right shift for a wrong one (CASE id 695, master luma flat at
    ~169-184, no ramp: a 1.1-level noise order was read as a confident MAD "winner").

    Reads [first, first + count) master frames against the candidate at shift -
    ANCHOR_MAD_REACH, shift and shift + ANCHOR_MAD_REACH (`_grey_frames`, the same raw
    decode `_pixel_readings` uses -- never a hand-rolled transform). Returns (shift,
    means): shift unchanged, and means None, when the window cannot be read or is not a
    fade; otherwise the neighbour with the lowest mean absolute luma difference, kept only
    when it beats shift's own mean by at least ANCHOR_MAD_MARGIN_LEVELS -- else unchanged.
    """
    import numpy as np
    import frame_compare
    shifts = [shift - ANCHOR_MAD_REACH, shift, shift + ANCHOR_MAD_REACH]
    rate = domain["master_rate"]
    scale = domain.get("time_scale")
    comparer = frame_compare.FrameComparer(
        master_obj.filePath, candidate_obj.filePath, 0.0, 1.0, rate.numerator,
        rate.denominator,
        time_scales=None if scale is None else {candidate_obj.filePath: scale})
    try:
        master = _grey_frames(comparer, master_obj.filePath, first, count, rate)
        low = first + min(shifts)
        if first < 0 or low < 0 or not len(master):
            return shift, None
        master_luma = [float(frame.mean()) for frame in master]
        if not _is_fade_window(master_luma):
            return shift, None
        candidate = _grey_frames(comparer, candidate_obj.filePath, low,
                                 len(master) + max(shifts) - min(shifts), rate)
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"repair: anchor_mad_unread ({type(error).__name__}: {error})\n")
        return shift, None
    means = {}
    for probed_shift in shifts:
        values = []
        for index in range(len(master)):
            other = index + probed_shift - min(shifts)
            if other < len(candidate):
                values.append(float(np.abs(master[index] - candidate[other]).mean()))
        if values:
            means[probed_shift] = statistics.mean(values)
    if shift not in means:
        return shift, means
    best = min(means, key=means.get)
    if best != shift and means[shift] - means[best] >= ANCHOR_MAD_MARGIN_LEVELS:
        return best, means
    return shift, means


def _anchor_shifts_mad_confirmed(result, domain, master_obj, candidate_obj):
    """Re-check each seated anchor's shift against raw luma MAD, correcting
    `before_shift_frames` / `after_shift_frames` when a neighbour reads clearly lower.

    Anchor A's window ends at its seed (the backward side); anchor B's starts at its seed
    (the forward side) -- the same windows `scene_anchor._anchor_window` builds.
    """
    import scene_anchor
    anchor_a, anchor_b = result.get("anchor_a_frame"), result.get("anchor_b_frame")
    before, after = result.get("before_shift_frames"), result.get("after_shift_frames")
    updates = {}
    if anchor_a is not None and before is not None:
        first = max(0, anchor_a - scene_anchor.ANCHOR_WINDOW_FRAMES)
        confirmed, means = _anchor_mad_confirmed_shift(
            domain, master_obj, candidate_obj, first, anchor_a - first, before)
        if confirmed != before:
            tools.log_always(
                f"repair: anchor_shift_mad_corrected side=A anchor={anchor_a} "
                f"phash_shift={before} mad_shift={confirmed} means={means} "
                f"for {candidate_obj.filePath}\n")
            updates["before_shift_frames"] = confirmed
    if anchor_b is not None and after is not None:
        confirmed, means = _anchor_mad_confirmed_shift(
            domain, master_obj, candidate_obj, anchor_b, scene_anchor.ANCHOR_WINDOW_FRAMES,
            after)
        if confirmed != after:
            tools.log_always(
                f"repair: anchor_shift_mad_corrected side=B anchor={anchor_b} "
                f"phash_shift={after} mad_shift={confirmed} means={means} "
                f"for {candidate_obj.filePath}\n")
            updates["after_shift_frames"] = confirmed
    if not updates:
        return result
    return dict(result, **updates)


def _two_anchor_call(hole, domain, master_obj, candidate_obj, low_ms, high_ms,
                     offset_before_ms, offset_after_ms, step_ms, quantum_ms, probe,
                     resolve_shift=True, accept_step_disagreement=False):
    """Run the interior resolver once as a logged step; an exception becomes a named decline."""
    import scene_anchor
    deadline = hole.get("deadline")
    if deadline is not None and time.monotonic() > deadline:
        return {"declined": True, "reason": "hole_budget_exceeded",
                "evidence": f"probe={probe} not launched: the hole's budget is spent"}
    _orch().step_launch("two_anchor", candidate=candidate_obj.filePath, probe=probe,
                resolve_shift=resolve_shift, cluster=hole.get("cluster_id"),
                shared_scan_windows_cached=(None if hole.get("scan_cache") is None
                                            else len(hole["scan_cache"])),
                bracket_ms=[round(float(low_ms), 2), round(float(high_ms), 2)],
                offset_before_ms=round(float(offset_before_ms), 3),
                offset_after_ms=round(float(offset_after_ms), 3),
                time_scale=(None if domain["time_scale"] is None
                            else f"{domain['time_scale'].numerator}/"
                                 f"{domain['time_scale'].denominator}"))
    try:
        result = scene_anchor.locate_scene_anchors(
            master_obj.filePath, candidate_obj.filePath,
            domain["master_rate"].numerator, domain["master_rate"].denominator,
            float(low_ms), float(high_ms), float(offset_before_ms), float(offset_after_ms),
            step_ms=float(step_ms), quantum_ms=float(quantum_ms),
            candidate_time_scale=domain["time_scale"],
            normalise_geometry=True, resolve_shift=resolve_shift,
            shift_search_frames=_shift_search_frames(domain, quantum_ms),
            cluster_window=hole.get("cluster_window"), scan_cache=hole.get("scan_cache"),
            deadline=deadline, accept_step_disagreement=accept_step_disagreement)
    except Exception as error:                                           # noqa: BLE001
        result = {"declined": True, "reason": f"resolver_raised:{type(error).__name__}",
                  "evidence": str(error)[:300]}
    if not result["declined"]:
        result = _anchor_shifts_mad_confirmed(result, domain, master_obj, candidate_obj)
    _orch().step_result("two_anchor", candidate=candidate_obj.filePath, probe=probe,
                declined=result["declined"], reason=result.get("reason"),
                anchor_a=result.get("anchor_a_frame"), anchor_b=result.get("anchor_b_frame"),
                before_shift=result.get("before_shift_frames"),
                after_shift=result.get("after_shift_frames"),
                nominal_before_shift=result.get("nominal_before_shift_frames"),
                nominal_after_shift=result.get("nominal_after_shift_frames"),
                forward_walk=result.get("forward_walk_frames"),
                backward_walk=result.get("backward_walk_frames"),
                sweep_crossed=result.get("sweep_crossed"), net_kind=result.get("net_kind"),
                unmatched_span_matches_before=result.get("unmatched_span_matches_before"),
                unmatched_span_matches_after=result.get("unmatched_span_matches_after"),
                step_plumbing_ok=result.get("step_plumbing_ok"),
                evidence=result.get("evidence"))
    return result


def _span_noise_reading(result):
    """Return which shift ("before", "after", "both" or None) matches most of the unmatched span.

    pHash noise inside common content can stop a sweep early; such a span matches the other
    shift on ~90 % of its frames, while truly divergent spans match either on under ~10 %, so a
    simple majority separates them. None means the span is genuinely divergent.
    """
    readings = []
    for label in ("before", "after"):
        matched, readable = result.get(f"unmatched_span_matches_{label}") or [0, 0]
        if readable and 2 * matched > readable:
            readings.append(label)
    if len(readings) == 2:
        return "both"
    return readings[0] if readings else None


# Pixel fronts are trusted only when each side shows a run of at least this many consecutive
# frames reading the same picture under its own shift (the anchor search's validation length);
# a regraded or re-filtered pair never does and keeps the pHash fronts.
PIXEL_TRUST_RUN_FRAMES = 12
# Frames read past each anchor: a fade can start before anchor A or end after anchor B, since
# pHash seats an anchor on a fade it cannot see.
PIXEL_FRONT_MARGIN_FRAMES = 96
# Widest span (frames) read at pixel level; wider ones keep the pHash fronts.
PIXEL_FRONT_MAX_FRAMES = 2400


def _pixel_readings(domain, master_obj, candidate_obj, first, count, shifts):
    """{shift: [(level difference, local fraction, master std, master level) or None] * n} for
    [first, first + n) against the candidate at each shift (`_pixel_reading`); None marks a
    candidate frame that does not exist. Returns None on any read failure."""
    rate = domain["master_rate"]
    import frame_compare
    scale = domain.get("time_scale")
    comparer = frame_compare.FrameComparer(
        master_obj.filePath, candidate_obj.filePath, 0.0, 1.0, rate.numerator,
        rate.denominator,
        time_scales=None if scale is None else {candidate_obj.filePath: scale})
    master = _grey_frames(comparer, master_obj.filePath, first, count, rate)
    low = first + min(shifts)
    if first < 0 or low < 0 or not len(master):
        return None
    candidate = _grey_frames(comparer, candidate_obj.filePath, low,
                             len(master) + max(shifts) - min(shifts), rate)
    readings = {}
    for shift in shifts:
        row = []
        for index in range(len(master)):
            other = index + shift - min(shifts)
            row.append(_pixel_reading(master[index], candidate[other])
                       if other < len(candidate) else None)
        readings[shift] = row
    return readings


def _same_flags(row, reference):
    """Per-frame "same picture" flags of one reading row (`_pixel_readings`).

    The two files' own steady level difference (a regrade, a different transfer curve, which
    varies from scene to scene) is the median over the PIXEL_STEADY_FRAMES common frames
    nearest each frame: first those of `reference` (known common), then, once more, every
    frame the first pass read as the same picture, so a shot far from `reference` is judged
    against its own steady difference. A frame differs when its local fraction reaches
    PIXEL_DIFF_FRACTION or its level leaves the steady difference by more than
    PIXEL_FADE_LEVELS, or by more than PIXEL_FADE_SHARE of the master's own level on a dark
    picture (a fade or a dim frame), never less than PIXEL_FADE_FLOOR. Flat frames on both
    sides at near levels (black against black) are the same picture.
    """
    import bisect
    import frame_hash

    def judge(known):
        common = sorted(i for i in known if row[i] is not None
                        and row[i][1] < PIXEL_DIFF_FRACTION
                        and abs(row[i][0]) <= PIXEL_DIFF_LEVELS)
        flags = []
        for index, reading in enumerate(row):
            if reading is None or reading[1] >= PIXEL_DIFF_FRACTION:
                flags.append(False)
                continue
            if (reading[2] < frame_hash.FLAT_STD and reading[4] < frame_hash.FLAT_STD
                    and abs(reading[0]) <= PIXEL_FADE_LEVELS):
                flags.append(True)
                continue
            steady = 0.0
            if common:
                at = bisect.bisect_left(common, index)
                near = common[max(0, at - PIXEL_STEADY_FRAMES):at + PIXEL_STEADY_FRAMES]
                near = sorted(near, key=lambda i: abs(i - index))[:PIXEL_STEADY_FRAMES]
                steady = statistics.median(row[i][0] for i in near)
            bound = max(PIXEL_FADE_FLOOR, min(PIXEL_FADE_LEVELS, PIXEL_FADE_SHARE * reading[3]))
            flags.append(abs(reading[0] - steady) <= bound)
        return flags

    first = judge(reference)
    return judge(sorted(set(reference) | {i for i, flag in enumerate(first) if flag}))


def _tolerant_flags(readings, shift, reference):
    """`_same_flags` at `shift`, a frame also counting as the same picture when it matches at
    shift - 1 or shift + 1: two releases can cut or fade the same shot one frame apart."""
    rows = [_same_flags(readings[shift + delta], reference) for delta in (-1, 0, 1)
            if shift + delta in readings]
    return [any(flags) for flags in zip(*rows)]


def _front_forward(same, front):
    """Forward front over `same` flags: first index of a run of PIXEL_RUN_FRAMES differing
    frames reached from `front` (moving right through same frames when `front` reads the
    same picture, then left through differing ones)."""
    count = len(same)

    def run_differs(index):
        return all(not same[i] for i in range(index, min(count, index + PIXEL_RUN_FRAMES)))
    if front < count and same[front]:
        while front < count and not run_differs(front):
            front += 1
    while front > 0 and not all(same[i] for i in range(max(0, front - PIXEL_RUN_FRAMES),
                                                       front)):
        front -= 1
    return front


def _front_backward(same, back):
    """Mirror of `_front_forward`: index just after the last differing run ending before the
    common frames that close the span."""
    reverse = list(reversed(same))
    return len(same) - _front_forward(reverse, len(same) - back)


def _longest_run(flags):
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def _pixel_refined(result, domain, master_obj, candidate_obj):
    """Re-read a two-anchor answer's fronts on pixels, or None to keep the pHash ones.

    pHash is blind to a fade or a dim frame (same structure, other brightness) and can stop on
    one odd frame inside common or black content; this check needs every frame that really
    differs. Around the anchors, each master frame is compared with the candidate under the
    before and the after shift (`_same_flags`): the forward front advances over frames that
    are the same under the before shift and moves back over differing ones; the backward front
    does the same under the after shift, both free to pass an anchor seated on a fade. Fronts
    that cross mark a span the picture cannot place; frames next to a front that are the same
    under both shifts are counted as blind runs. Never raises.
    """
    anchor_a, anchor_b = result["anchor_a_frame"], result["anchor_b_frame"]
    before, after = result["before_shift_frames"], result["after_shift_frames"]
    if anchor_b - anchor_a <= 0 or anchor_b - anchor_a > PIXEL_FRONT_MAX_FRAMES:
        return None
    first = max(0, anchor_a - PIXEL_FRONT_MARGIN_FRAMES, 1 - min(before, after))
    count = anchor_b + PIXEL_FRONT_MARGIN_FRAMES - first
    try:
        readings = _pixel_readings(domain, master_obj, candidate_obj, first, count,
                                   sorted({shift + delta for shift in (before, after)
                                           for delta in (-1, 0, 1)}))
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"repair: pixel_fronts_unread ({type(error).__name__}: {error})\n")
        return None
    if readings is None:
        return None
    count = len(readings[before])
    front = min(max(result["pre_collapse_start_master"] - first, 0), count)
    back = min(max(result["pre_collapse_end_master"] - first, 0), count)
    # The steady level difference is first read where the pHash walks found common picture,
    # kept clear of their fronts: pHash walks into a fade it cannot see.
    guard = 2 * PIXEL_STEADY_FRAMES
    same_before = _tolerant_flags(readings, before, range(0, max(1, front - guard)))
    same_after = _tolerant_flags(readings, after, range(min(count - 1, back + guard), count))
    trusted = (_longest_run(same_before[:max(front, anchor_a - first + 1)])
               >= PIXEL_TRUST_RUN_FRAMES
               and _longest_run(same_after[min(back, anchor_b - first - 1):])
               >= PIXEL_TRUST_RUN_FRAMES)
    if not trusted:
        tools.log_always(f"repair: pixel_fronts_untrusted anchors=[{anchor_a}, {anchor_b}] "
                         f"for {candidate_obj.filePath}\n")
        return None
    # A front moves until PIXEL_RUN_FRAMES consecutive frames say otherwise: one odd frame
    # (a flash, a single-frame match inside a fade) never stops it.
    front = _front_forward(same_before, front)
    back = _front_backward(same_after, back)
    crossed = back < front
    blind_before = blind_after = 0
    if not crossed:
        while front - blind_before - 1 >= 0 and same_after[front - blind_before - 1]:
            blind_before += 1
        while back + blind_after < count and same_before[back + blind_after]:
            blind_after += 1
        if after > before:
            # A candidate excess: frames opening the span that the after shift reads as the
            # same picture (or closing it under the before shift) need no fill; only the
            # removal must sit somewhere among them (rule 1, `_differing_span_outcome`).
            lead = 0
            while front + lead < back and same_after[front + lead]:
                lead += 1
            tail = 0
            while back - tail - 1 >= front + lead and same_before[back - tail - 1]:
                tail += 1
            front, back = front + lead, back - tail
            blind_before, blind_after = blind_before + lead, blind_after + tail
    front, back = first + front, first + back
    refined = dict(result, pre_collapse_start_master=front, pre_collapse_end_master=back,
                   sweep_crossed=crossed, blind_before_front_frames=blind_before,
                   blind_after_front_frames=blind_after, pixel_fronts=True,
                   span_majority_under_single_shift=False,
                   anchor_a_frame=min(anchor_a, front), anchor_b_frame=max(anchor_b, back))
    start, end = (anchor_b, anchor_b) if crossed else (front, back)
    refined.update(master_start_frame=start, master_end_frame=end,
                   candidate_start_frame=start + before, candidate_end_frame=end + after)
    tools.log_always(
        f"repair: pixel_fronts anchors=[{anchor_a}, {anchor_b}] shifts=[{before}, {after}] "
        f"phash_fronts=[{result['pre_collapse_start_master']}, "
        f"{result['pre_collapse_end_master']}] pixel_fronts=[{front}, {back}] "
        f"crossed={crossed} blind_runs=[{blind_before}, {blind_after}] "
        f"for {candidate_obj.filePath}\n")
    return refined


def _checked_two_anchor(hole, domain, master_obj, candidate_obj, low_ms, high_ms,
                        offset_before_ms, offset_after_ms, step_ms, quantum_ms, probe,
                        resolve_shift=True, accept_step_disagreement=False):
    """Run `_two_anchor_call` and reject answers whose unmatched span is really common content.

    When one shift claims the span, the front that stopped on noise is wrong, so the search is
    repeated with its bracket collapsed onto the other front; that answer must pass the same
    test. Otherwise a named decline, carrying `claimed_by` and `claimed_shift_frames` so the
    caller can test the no-cut hypothesis at that shift.
    """
    def _with_claim(declined, reading, source):
        shift = source["before_shift_frames"] if reading == "before" else source[
            "after_shift_frames"]
        return dict(declined, claimed_by=reading, claimed_shift_frames=shift,
                    claimed_span_master=[source["pre_collapse_start_master"],
                                         source["pre_collapse_end_master"]],
                    claimed_matches=source[f"unmatched_span_matches_{reading}"])

    result = _two_anchor_call(hole, domain, master_obj, candidate_obj, low_ms, high_ms,
                              offset_before_ms, offset_after_ms, step_ms, quantum_ms, probe,
                              resolve_shift=resolve_shift,
                              accept_step_disagreement=accept_step_disagreement)
    if result["declined"]:
        return result
    refined = _pixel_refined(result, domain, master_obj, candidate_obj)
    if refined is not None:
        return refined
    reading = _span_noise_reading(result)
    if reading is None:
        return result
    frame_ms = float(domain["frame_ms"])
    _orch().step_result("sweep_front_inside_common_content", candidate=candidate_obj.filePath,
                probe=probe, claimed_by=reading,
                unmatched_master_frames=[result["pre_collapse_start_master"],
                                         result["pre_collapse_end_master"]],
                matches_before=result["unmatched_span_matches_before"],
                matches_after=result["unmatched_span_matches_after"])
    if reading == "both" and result["before_shift_frames"] == result["after_shift_frames"]:
        # One shift on both sides matching the span: the walks stopped on pHash noise, so
        # nothing differs between the anchors (read as `no_cut_confirmed`).
        _orch().step_result("single_shift_claims_span", candidate=candidate_obj.filePath, probe=probe,
                    shift=result["before_shift_frames"],
                    matches=result["unmatched_span_matches_before"])
        return dict(result, span_majority_under_single_shift=True)
    if reading == "both":
        claimed = _claimed_span_reading(result)
        if claimed is not None:
            _orch().step_result("span_claimed_by_both_shifts_read", candidate=candidate_obj.filePath,
                        probe=probe, route=claimed["span_claimed_by_both_shifts"],
                        shifts=[result["before_shift_frames"], result["after_shift_frames"]],
                        fronts=[claimed["pre_collapse_start_master"],
                                claimed["pre_collapse_end_master"]],
                        blind_runs=[claimed["blind_before_front_frames"],
                                    claimed["blind_after_front_frames"]],
                        classes=result.get("anchor_span_classes"))
            return claimed
        return {"declined": True, "reason": "unmatched_span_claimed_by_both_shifts",
                "evidence": (f"span {result['pre_collapse_start_master']}-"
                             f"{result['pre_collapse_end_master']} before="
                             f"{result['unmatched_span_matches_before']} after="
                             f"{result['unmatched_span_matches_after']}")}
    front = (result["pre_collapse_start_master"] if reading == "after"
             else result["pre_collapse_end_master"])
    narrowed = _two_anchor_call(hole, domain, master_obj, candidate_obj,
                                front * frame_ms, (front + 1) * frame_ms,
                                offset_before_ms, offset_after_ms, step_ms, quantum_ms,
                                probe=f"{probe}_refront_{reading}", resolve_shift=resolve_shift,
                                accept_step_disagreement=accept_step_disagreement)
    if narrowed["declined"]:
        return _with_claim(narrowed, reading, result)
    if _span_noise_reading(narrowed) is not None:
        return _with_claim(
            {"declined": True, "reason": "sweep_front_inside_common_content",
             "evidence": (f"first span {result['pre_collapse_start_master']}-"
                          f"{result['pre_collapse_end_master']} claimed by {reading}; "
                          f"re-fronted span {narrowed['pre_collapse_start_master']}-"
                          f"{narrowed['pre_collapse_end_master']} still claimed "
                          f"(before={narrowed['unmatched_span_matches_before']} "
                          f"after={narrowed['unmatched_span_matches_after']})")},
            reading, result)
    return narrowed


def _claimed_span_reading(result):
    """Re-read a span that both shifts claim from its per-frame classes, or None.

    `anchor_span_classes` holds one class per master frame between the anchors (before-only,
    after-only, both, neither). Frames reading the same picture under neither shift are frames
    that differ (a fade against a hard cut, black against a card): they bound the fill, and
    the static frames that match both shifts next to it are blind runs where a candidate
    excess is removed (`_differing_span_outcome`). With no differing frame the span is blind:
    the gap goes where the blind-span rule puts it, between the last before-only frame and the
    first after-only one. Any after-only frame left of a before-only one is a contradiction.
    """
    classes = result.get("anchor_span_classes")
    anchor_a = result["anchor_a_frame"]
    if not classes or "?" in classes:
        return None
    last_before, first_after = classes.rfind("b"), classes.find("a")
    if first_after < 0:
        first_after = len(classes)
    differing = [index for index, value in enumerate(classes) if value == "0"]
    if differing:
        front, back = differing[0], differing[-1] + 1
        if "a" in classes[:front] or "b" in classes[back:]:
            return None
        blind_before = front - len(classes[:front].rstrip("2"))
        blind_after = len(classes[back:]) - len(classes[back:].lstrip("2"))
        front, back = anchor_a + front, anchor_a + back
        return dict(result, pre_collapse_start_master=front, pre_collapse_end_master=back,
                    master_start_frame=front, master_end_frame=back,
                    candidate_start_frame=front + result["before_shift_frames"],
                    candidate_end_frame=back + result["after_shift_frames"],
                    sweep_crossed=False, blind_before_front_frames=blind_before,
                    blind_after_front_frames=blind_after,
                    span_claimed_by_both_shifts="differing")
    if last_before >= first_after:
        return None
    return dict(result, pre_collapse_start_master=anchor_a + first_after,
                pre_collapse_end_master=anchor_a + last_before + 1, sweep_crossed=True,
                blind_before_front_frames=0, blind_after_front_frames=0,
                span_claimed_by_both_shifts="blind")


def _interior_verdict(result):
    """Classify a resolver result as no_cut_confirmed, pinned-to-ambiguous-zone, resolved or declined.

    One shift with nothing between the fronts closes the hole. With two shifts, overlapping
    fronts (on either axis) can only come from frames matching both shifts, i.e. a static
    span, which is pinned at the end of the ambiguous zone; an anchor refused as
    `anchor_ambiguous_static_span` leads to the same verdict. Anything else is a resolved cut.
    """
    if result.get("declined"):
        if (result.get("reason") == "anchor_ambiguous_static_span"
                and (result.get("anchor_a_ambiguous") or result.get("anchor_b_ambiguous"))):
            return _orch().HOLE_PINNED_TO_AMBIGUOUS_ZONE_END
        return _orch().HOLE_DECLINED
    same_shift = result["before_shift_frames"] == result["after_shift_frames"]
    if same_shift and (result["master_end_frame"] == result["master_start_frame"]
                       or result.get("span_majority_under_single_shift")):
        return _orch().HOLE_NO_CUT_CONFIRMED
    candidate_overlap = result["candidate_end_frame"] < result["candidate_start_frame"]
    if not same_shift and (result["sweep_crossed"] or candidate_overlap):
        return _orch().HOLE_PINNED_TO_AMBIGUOUS_ZONE_END
    return _orch().HOLE_RESOLVED


def _pin_point(result):
    """Locate the end of the ambiguous (static) zone, where the N-frame edit is placed in one block.

    Crossed master fronts pin at the forward front; candidate fronts overlapping by k frames
    pin at the backward front + k. Only reached for a resolved two-anchor cross-sweep whose
    fronts crossed or overlapped (`_interior_verdict`'s non-declined branch); a refused
    ambiguous anchor is handled by `_ambiguous_pin_outcome`, which delegates to
    `_blind_span_outcome` instead -- the same picture-cannot-decide case, with one anchor
    missing rather than both sweeps crossing.

    Returns:
        (pin_frame, overlap_or_width)
    """
    if result["sweep_crossed"]:
        start, end = result["pre_collapse_start_master"], result["pre_collapse_end_master"]
        return start, start - end
    overlap = result["candidate_start_frame"] - result["candidate_end_frame"]
    return result["pre_collapse_end_master"] + overlap, overlap


def _pinned_frames(result):
    """Return (master_start, master_end, candidate_start, candidate_end) for a pinned static span.

    With delta = after_shift - before_shift and pin P: an addition (delta >= 0) replaces no
    master frame; a deletion fills the master's frames [P + delta, P). Both keep the two
    shifted reads continuous on each axis, with no candidate frame read twice.
    """
    pin, _ambiguous = _pin_point(result)
    before, after = result["before_shift_frames"], result["after_shift_frames"]
    delta = after - before
    if delta >= 0:
        return pin, pin, pin + before, pin + after
    return pin + delta, pin, pin + after, pin + after


def _interior_outcome(hole, domain, result, status, refuted_proposal=None):
    grid = result["grid"]
    master_start, master_end = result["master_start_frame"], result["master_end_frame"]
    candidate_start, candidate_end = result["candidate_start_frame"], result["candidate_end_frame"]
    if status == _orch().HOLE_PINNED_TO_AMBIGUOUS_ZONE_END:
        master_start, master_end, candidate_start, candidate_end = _pinned_frames(result)
    length_master = master_end - master_start
    length_candidate = candidate_end - candidate_start
    net_kind = ("addition" if length_candidate > length_master
                else "deletion" if length_candidate < length_master
                else "still_image" if length_master == 0 else "ordinary")
    outcome = {
        "modality": _orch().MODALITY, "status": status, "kind": hole["kind"],
        "why_token": hole["why_token"], "cause": None,
        "grid": f"{grid['num']}/{grid['den']}",
        "anchor_a_frame": result["anchor_a_frame"], "anchor_b_frame": result["anchor_b_frame"],
        "master_start_frame": master_start, "master_end_frame": master_end,
        "candidate_start_frame_equivalent": candidate_start,
        "candidate_end_frame_equivalent": candidate_end,
        "candidate_start_frame": _candidate_native_frame(candidate_start, domain),
        "candidate_end_frame": _candidate_native_frame(candidate_end, domain),
        "master_start_ms": _exact_ms_of_frame(master_start, domain),
        "master_end_ms": _exact_ms_of_frame(master_end, domain),
        # Frame shifts place cuts; they are never an audio offset.
        "video_shift_ms_frame_quantised": [
            _exact_ms_of_frame(result["before_shift_frames"], domain),
            _exact_ms_of_frame(result["after_shift_frames"], domain)],
        **_audio_offsets(hole),
        "net_kind": net_kind,
        "frames_to_cut": max(0, length_candidate - length_master),
        "frames_to_fill": max(0, length_master - length_candidate),
        "sweep_master_frames": [result["master_start_frame"], result["master_end_frame"]],
        "sweep_candidate_frames_equivalent": [result["candidate_start_frame"],
                                              result["candidate_end_frame"]],
        "pre_collapse_master_frames": [result["pre_collapse_start_master"],
                                       result["pre_collapse_end_master"]],
        "before_shift_frames": result["before_shift_frames"],
        "after_shift_frames": result["after_shift_frames"],
        "nominal_before_shift_frames": result["nominal_before_shift_frames"],
        "nominal_after_shift_frames": result["nominal_after_shift_frames"],
        "forward_walk_frames": result["forward_walk_frames"],
        "backward_walk_frames": result["backward_walk_frames"],
        "span_frames": result["anchor_b_frame"] - result["anchor_a_frame"],
        "sweep_crossed": result["sweep_crossed"],
        "geometry": (result.get("geometry") or {}).get("verdict"),
        "evidence": result.get("evidence"),
    }
    if status == _orch().HOLE_PINNED_TO_AMBIGUOUS_ZONE_END:
        outcome["cause"] = "static_span_ambiguity"
        outcome["pin_frame"], outcome["ambiguous_frames"] = _pin_point(result)
    if refuted_proposal is not None:
        outcome["refuted_proposal"] = refuted_proposal
    return outcome


def _ambiguous_pin_outcome(hole, domain, result, candidate_path, walk=None):
    """Route a refused ambiguous anchor through the blind-span placement.

    One side could not be seated because its content is self-similar (a black or static zone)
    -- the same picture-cannot-decide case a blind span is, with one anchor missing instead of
    both sweeps crossing. The firm side's resolved shift and the ambiguous side's own measured
    shift (nominal when it was never seated) are both real anchor offsets, so their difference
    is exactly the width `_blind_span_outcome` computes from a crossed sweep; no audio quantity
    enters it. This only rebuilds the sweep-shaped fields the anchor search never produced (no
    cross-sweep ran, so there is no walk, no pre-collapse front) before delegating to it, so one
    function and one log line place every static span, however it was detected.
    """
    a_ambiguous, b_ambiguous = result.get("anchor_a_ambiguous"), result.get("anchor_b_ambiguous")
    anchor_a, anchor_b = result.get("anchor_a_frame"), result.get("anchor_b_frame")
    if anchor_b is None:
        # The right anchor itself was never seated -- nothing to place just before it. The
        # only position left is the ambiguous anchor's own, nearest the hole.
        ambiguous = b_ambiguous or a_ambiguous
        anchor_b = ambiguous["anchor"]
    if anchor_a is None:
        anchor_a = (a_ambiguous or {}).get("anchor", anchor_b)
    # The shift at a side that was never seated at all (no anchor, not even a nominal one
    # carried through) falls back to its own ambiguous reading's validated shift.
    before = result.get("before_shift_frames")
    if before is None:
        before = (a_ambiguous or {}).get("shift")
    after = result.get("after_shift_frames")
    if after is None:
        after = (b_ambiguous or {}).get("shift")
    blind = dict(result, anchor_a_frame=anchor_a, anchor_b_frame=anchor_b,
                before_shift_frames=before, after_shift_frames=after,
                forward_walk_frames=None, backward_walk_frames=None, sweep_crossed=True,
                # No sweep ran: the picture bounds the gap only by the anchors themselves.
                pre_collapse_start_master=anchor_b, pre_collapse_end_master=anchor_a)
    outcome = _blind_span_outcome(hole, domain, blind, candidate_path, walk=walk)
    outcome["pin_route"] = "ambiguous_anchor"
    outcome["anchor_a_ambiguous"] = a_ambiguous
    outcome["anchor_b_ambiguous"] = b_ambiguous
    return outcome


def _span_no_cut_outcome(hole, domain, master_obj, candidate_obj, low_ms, high_ms, result):
    """Test a sub-floor hole for no cut on its own frames when no anchor pair could be seated.

    Uses `scene_anchor.island_match` at the hole's `offset_before`. A majority match under one
    shift returns a `no_cut_confirmed` outcome with the refuted proposal; otherwise None.
    """
    import scene_anchor
    frame_ms = domain["frame_ms"]
    try:
        reading = scene_anchor.island_match(
            master_obj.filePath, candidate_obj.filePath,
            domain["master_rate"].numerator, domain["master_rate"].denominator,
            float(low_ms), float(high_ms), float(hole["offset_before_ms"]),
            candidate_time_scale=domain["time_scale"],
            shift_search_frames=_shift_search_frames(domain, hole["quantum_ms"]),
            scan_cache=hole.get("scan_cache"))
    except Exception as error:                                           # noqa: BLE001
        reading = {"verdict": "unreadable", "reason": f"island_match_raised:{type(error).__name__}"}
    _orch().step_result("hole_span_no_cut_test", candidate=candidate_obj.filePath,
                master_ms=[round(float(low_ms), 2), round(float(high_ms), 2)],
                audio_step_ms=round(hole["step_ms"], 3), proposal_reason=result.get("reason"),
                **{f"span_{key}": value for key, value in reading.items()})
    if reading["verdict"] != "same":
        return None
    shift = reading["shift_frames"]
    first, last = reading["master_frames"]
    outcome = {
        "modality": _orch().MODALITY, "status": _orch().HOLE_NO_CUT_CONFIRMED, "kind": hole["kind"],
        "why_token": hole["why_token"], "cause": None,
        "grid": f"{domain['master_rate'].numerator}/{domain['master_rate'].denominator}",
        "master_start_frame": last, "master_end_frame": last,
        "before_shift_frames": shift, "after_shift_frames": shift,
        **_audio_offsets(hole),
        "span_frames": last - first, "net_kind": "still_image",
        "evidence": f"span_no_cut matched={reading['matched']} readable={reading['readable']}",
        "refuted_proposal": {
            "audio_step_ms": round(hole["step_ms"], 3), "audio_step_points": hole["step_points"],
            "master_position_ms": [round(float(hole["master_ms"][0]), 2),
                                   round(float(hole["master_ms"][1]), 2)],
            "master_position_frames": [first, last],
            "proposal_reading": result.get("reason"),
            "video_probe": "span_no_cut_test", "video_single_shift": shift,
            "video_verdict": _orch().HOLE_NO_CUT_CONFIRMED,
            "proposal_shifts": [result.get("before_shift_frames"),
                                result.get("after_shift_frames")],
            "proposal_span_matches_before": None, "proposal_span_matches_after": None,
            "claimed_span_matches": None,
            "span_test_matches": [reading["matched"], reading["readable"]],
            "surviving_shift_walk_frames": [None, None],
            "surviving_shift_span_frames": last - first},
    }
    return outcome


# Audio reading of a blind-gap placement: 20 ms RMS windows on a 5 ms hop; a window under
# BLIND_SILENCE_DB is silent (the cut checker's own silence reading), and levels are floored at
# BLIND_LEVEL_FLOOR_DB before two tracks are compared.
BLIND_SILENCE_DB = -50.0
BLIND_LEVEL_FLOOR_DB = -70.0
BLIND_SILENCE_WIN_S = 0.02
BLIND_SILENCE_HOP_S = 0.005
# Placements whose audio misfit is within this fraction of the misfit's whole range tie.
BLIND_MISFIT_TIE = 0.02


def _window_levels(samples, times, shift_s, peak=False):
    """RMS (or, with `peak`, peak) level in dB, floored, of the 20 ms window starting at each
    of `times` (+ `shift_s`)."""
    import numpy as np
    import audio_walk
    rate = audio_walk.WALK_RATE
    win = int(round(BLIND_SILENCE_WIN_S * rate))
    measure = audio_walk.peak_db if peak else audio_walk.rms_db
    rows = []
    for t in times:
        start = int(round((t + shift_s) * rate))
        chunk = samples[max(0, start):max(0, start + win)]
        rows.append(measure(chunk) if len(chunk) else -200.0)
    return np.maximum(np.array(rows), BLIND_LEVEL_FLOOR_DB)


def _candidate_silent(walk, times, shift_s):
    """Whether the candidate is silent in each 20 ms window: its peak under the audibility
    floor the audio walk itself uses (`audio_walk.AUDIBLE_DB`), so a fading sting tail still
    counts as sound."""
    import audio_walk
    return _window_levels(walk["candidate"], times, shift_s, peak=True) < audio_walk.AUDIBLE_DB


def _centre_of_longest(flags, low):
    """(centre, [first, last]) of the longest True run of `flags` (index 0 = frame `low`)."""
    best, start = None, None
    for index, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if best is None or index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    if best is None:
        return None
    first, last = low + best[0], low + best[1] - 1
    return (first + last + 1) // 2, [first, last]


def _blind_span_audio_point(domain, gap, before, after, low, high, walk):
    """Right-edge frame of a gap the picture cannot place, chosen from the audio, or None.

    `low`..`high` are the right-edge frames the picture allows (`_blind_span_bounds`).
    - A candidate removal (gap > 0) goes inside the candidate's own silence: the removed
      candidate audio must be silent, whatever the master plays there.
    - A master fill (gap < 0) is located by aligning the audio inside the span: the candidate
      is split where its audio before matches the master before the fill and its audio after
      matches the master after. When the candidate gives no such evidence (silent), every
      placement ties and the fill lands in the middle of the master's own silent stretch.
    The longest run of best placements wins and the gap is centred in it. `walk` is the pair's
    audio already decoded for the walk (`reference_walk`); no new decode is made.

    Returns:
        (frame, [first, last] frames of the winning run, how), or None when `walk` is absent,
        the probe raises, or no placement qualifies.
    """
    import numpy as np
    if walk is None or gap == 0 or high < low:
        return None
    frame_s = float(domain["frame_ms"]) / 1000.0
    width = abs(gap)
    hop = BLIND_SILENCE_HOP_S
    try:
        t_lo = max(0.0, (low - width) * frame_s)
        t_hi = high * frame_s
        times = np.arange(t_lo, max(t_lo + hop, t_hi - BLIND_SILENCE_WIN_S) + 1e-9, hop)
        candidate_after = _window_levels(walk["candidate"], times, after * frame_s)
        points = range(low, high + 1)

        def index_of(seconds):
            return int(min(len(times), max(0, math.ceil((seconds - t_lo) / hop - 1e-9))))

        def last_index(seconds):
            return int(min(len(times) - 1, math.floor((seconds - t_lo) / hop + 1e-9)))
        if gap > 0:
            silent = _candidate_silent(walk, times, after * frame_s)
            flags = []
            for point in points:
                first = index_of((point - width) * frame_s)
                last = last_index(point * frame_s - BLIND_SILENCE_WIN_S)
                flags.append(last >= first and bool(silent[first:last + 1].all()))
            found = _centre_of_longest(flags, low)
            return None if found is None else found + ("candidate_silence",)
        master = _window_levels(walk["master"], times, 0.0)
        candidate_before = _window_levels(walk["candidate"], times, before * frame_s)
        misfit_before = np.concatenate([[0.0], np.cumsum(np.abs(master - candidate_before))])
        misfit_after = np.concatenate([[0.0], np.cumsum(np.abs(master - candidate_after))])
        costs = []
        for point in points:
            split = index_of((point - width) * frame_s)
            resume = index_of(point * frame_s)
            costs.append(misfit_before[split] + misfit_after[-1] - misfit_after[resume])
        costs = np.array(costs)
        spread = float(costs.max() - costs.min())
        if spread <= 1e-9:
            # No audio evidence at all: the master's own silent stretch decides.
            silent = master < BLIND_SILENCE_DB
            flags = []
            for point in points:
                first = index_of((point - width) * frame_s)
                last = last_index(point * frame_s - BLIND_SILENCE_WIN_S)
                flags.append(last >= first and bool(silent[first:last + 1].all()))
            found = _centre_of_longest(flags, low)
            return None if found is None else found + ("master_silence",)
        found = _centre_of_longest(costs <= costs.min() + BLIND_MISFIT_TIE * spread, low)
        return None if found is None else found + ("audio_alignment",)
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"repair: blind_span_audio_probe_failed "
                      f"({type(error).__name__}: {error})\n")
        return None


def _blind_span_bounds(blind, gap):
    """Right-edge frames [low, high] the picture allows for a blind gap.

    Frames before the forward front F still match under the before shift, frames from the
    backward front Bk on match under the after shift, and nothing leaves the anchors: with
    w = max(0, -gap) master frames to fill, the edge P keeps [P - w, P) inside [A, B],
    P - w <= F and P >= Bk.
    """
    width = max(0, -gap)
    anchor_a, anchor_b = blind["anchor_a_frame"], blind["anchor_b_frame"]
    front = blind.get("pre_collapse_start_master")
    back = blind.get("pre_collapse_end_master")
    front = anchor_b if front is None else front
    back = anchor_a if back is None else back
    low, high = max(back, anchor_a + width), min(front + width, anchor_b)
    if high < low:
        low, high = anchor_a + width, anchor_b
    return low, high


def _blind_span_outcome(hole, domain, blind, candidate_path, walk=None):
    """Resolve a hole the picture could not read as a blind (black/static) span.

    Both anchors seated; the cross-sweep fronts crossed (or the candidate fronts overlapped),
    which is what a held black or static picture produces -- it matches either shift, so the
    picture bounds where the gap may sit (`_blind_span_bounds`) but cannot pick a frame. The
    width is the anchors' own shift difference (in whole frames).

    The audio places the gap inside the span when it can (`_blind_span_audio_point`), else it
    goes just before the blind span's right edge (the last frame the picture allows).

    A master fill widens the master span to the gap, ending at the chosen frame; a candidate
    removal keeps a single master point. The candidate side stays a single point at the chosen
    frame under the shift that survives (`after_shift_frames`): nothing in the candidate is
    read twice and nothing of it is kept inside the removed/filled span.
    """
    before, after = blind["before_shift_frames"], blind["after_shift_frames"]
    gap = after - before
    anchor_b = blind["anchor_b_frame"]
    frame_ms = float(domain["frame_ms"])
    low, high = _blind_span_bounds(blind, gap)
    found = _blind_span_audio_point(domain, gap, before, after, low, high, walk)
    if found is not None:
        point, frames, placement = found
        placement_evidence = f"placement_frames={frames}"
    else:
        point, placement, placement_evidence = high, "right_edge", "no_shared_silence"
    if gap < 0:
        placed = dict(blind, master_start_frame=point + gap, master_end_frame=point,
                     candidate_start_frame=point + after, candidate_end_frame=point + after)
        decision = "fill_from_master"
    else:
        placed = dict(blind, master_start_frame=point, master_end_frame=point,
                     candidate_start_frame=point + before, candidate_end_frame=point + after)
        decision = "remove_from_candidate"
    tools.log_always(
        f"repair: video_undecided_blind_span anchor_a={blind['anchor_a_frame']} "
        f"anchor_b={anchor_b} before_shift_frames={before} "
        f"after_shift_frames={after} gap_frames={gap} "
        f"gap_ms={round(gap * frame_ms, 3)} allowed_frames=[{low}, {high}] "
        f"decision={decision} placement={placement} {placement_evidence} "
        f"placed_at_frame={point} for {candidate_path}\n")
    return _interior_outcome(hole, domain, placed, _orch().HOLE_RESOLVED)


def _resolve_interior(hole, domain, master_obj, candidate_obj, walk=None):
    """Resolve an interior hole with the two-anchor frame-exact search.

    The audio proposes a step, the video decides. For a step under the aligner's resolution
    floor, the video is also offered a single shift on both sides (anchor A's, then B's, or the
    zones' audio offsets when no anchor resolved); the first that seats both anchors refutes
    the step. A step at or above the floor is likewise re-tested at a shift that claims the
    unmatched span. An ambiguous anchor leads to `_ambiguous_pin_outcome`, a sub-floor decline
    to `_span_no_cut_outcome`. `walk` (the pair's own decoded audio, when the caller has it)
    is only used to place a blind span inside a shared silence (`_blind_span_outcome`).
    """
    quantum_ms = hole["quantum_ms"]
    offset_before_ms = hole["offset_before_ms"]
    offset_after_ms = hole["offset_after_ms"]
    step_ms = hole["step_ms"]
    low_ms, high_ms = hole["master_ms"]
    # A pure insertion leaves an empty master span; the resolver needs at least one frame.
    frame_ms = domain["frame_ms"]
    if high_ms - low_ms < float(frame_ms):
        high_ms = low_ms + float(frame_ms)
    result = _checked_two_anchor(hole, domain, master_obj, candidate_obj, low_ms, high_ms,
                                 offset_before_ms, offset_after_ms, step_ms, quantum_ms,
                                 probe="proposal")
    below_floor = (hole["step_points"] is not None
                   and abs(hole["step_points"]) < banded_seed_alignment.RESOLUTION_FLOOR_QUANTA)
    proposal_single_shift = (not result["declined"]
                             and result["before_shift_frames"] == result["after_shift_frames"])
    hypotheses = []
    if result["declined"] and result.get("claimed_by") in ("before", "after"):
        hypotheses.append((f"no_cut_at_claimed_{result['claimed_by']}_shift",
                           float(result["claimed_shift_frames"] * frame_ms), False))
        _orch().step_result("video_refutes_audio_step", candidate=candidate_obj.filePath,
                    audio_step_ms=round(step_ms, 3), claimed_by=result["claimed_by"],
                    claimed_shift_frames=result["claimed_shift_frames"],
                    claimed_span_master=result.get("claimed_span_master"),
                    claimed_matches=result.get("claimed_matches"),
                    next_probe="no_cut_hypothesis_at_the_claimed_shift")
    if below_floor and not proposal_single_shift:
        if result["declined"]:
            hypotheses += [("no_cut_at_offset_before", offset_before_ms, True),
                           ("no_cut_at_offset_after", offset_after_ms, True)]
        else:
            hypotheses += [
                ("no_cut_at_anchor_a_shift",
                 float(result["before_shift_frames"] * frame_ms), False),
                ("no_cut_at_anchor_b_shift",
                 float(result["after_shift_frames"] * frame_ms), False)]
    if hypotheses:
        for label, hypothesis, search in hypotheses:
            probe = _checked_two_anchor(hole, domain, master_obj, candidate_obj, low_ms,
                                        high_ms, hypothesis, hypothesis, 0.0, quantum_ms,
                                        probe=label, resolve_shift=search)
            if probe["declined"] or probe["before_shift_frames"] != probe["after_shift_frames"]:
                continue
            verdict = _interior_verdict(probe)
            if verdict != _orch().HOLE_NO_CUT_CONFIRMED and not result["declined"]:
                # One shift that still leaves differing frames does not refute a step the
                # proposal resolved; the proposal's own answer stands.
                continue
            refuted = {"audio_step_ms": round(step_ms, 3),
                       "audio_step_points": hole["step_points"],
                       "master_position_ms": [round(float(hole["master_ms"][0]), 2),
                                              round(float(hole["master_ms"][1]), 2)],
                       "master_position_frames": [
                           _master_frame_of_ms(hole["master_ms"][0], domain),
                           _master_frame_of_ms(hole["master_ms"][1], domain)],
                       "proposal_reading": (result.get("reason") if result["declined"] else
                                            f"shifts {result['before_shift_frames']}/"
                                            f"{result['after_shift_frames']} anchors "
                                            f"{result['anchor_a_frame']}/"
                                            f"{result['anchor_b_frame']}"),
                       "video_probe": label,
                       "video_single_shift": probe["before_shift_frames"],
                       "video_verdict": verdict,
                       "proposal_shifts": ([result.get("before_shift_frames"),
                                            result.get("after_shift_frames")]
                                           if not result["declined"] else
                                           [result.get("claimed_by"),
                                            result.get("claimed_shift_frames")]),
                       "proposal_span_matches_before": result.get(
                           "unmatched_span_matches_before"),
                       "proposal_span_matches_after": result.get(
                           "unmatched_span_matches_after"),
                       "claimed_span_matches": result.get("claimed_matches"),
                       "surviving_shift_walk_frames": [probe.get("forward_walk_frames"),
                                                       probe.get("backward_walk_frames")],
                       "surviving_shift_span_frames": (probe["anchor_b_frame"]
                                                       - probe["anchor_a_frame"])}
            return _interior_outcome(hole, domain, probe, verdict, refuted_proposal=refuted)
    if result["declined"]:
        # Under the resolution floor the audio step may be noise, so the span itself is first
        # asked the no-cut question (a static span matches any shift).
        if below_floor:
            span_outcome = _span_no_cut_outcome(hole, domain, master_obj, candidate_obj,
                                                low_ms, high_ms, result)
            if span_outcome is not None:
                return span_outcome
        # A sub-floor span the no-cut test could not close is still a static span: the
        # ambiguous anchor is pinned by the blind-span rule rather than declined.
        if _interior_verdict(result) == _orch().HOLE_PINNED_TO_AMBIGUOUS_ZONE_END:
            return _ambiguous_pin_outcome(hole, domain, result, candidate_obj.filePath, walk=walk)
        if not below_floor and result.get("reason") == "anchor_step_inconsistent":
            # Both anchors seated and the cross-sweep ran, but the frame gap it counted
            # disagreed with the audio's nominal step -- the usual sign of a blind span: a
            # held black or static picture matches either offset, so each walk runs clean
            # through the whole span and touches the far anchor (`sweep_crossed`). The
            # picture is never read past the anchors in that case, so the anchors' own
            # (frame-exact, scene-cut-seeded) shift difference is trusted over the nominal
            # step instead of declining.
            blind = _checked_two_anchor(hole, domain, master_obj, candidate_obj, low_ms,
                                        high_ms, offset_before_ms, offset_after_ms, step_ms,
                                        quantum_ms, probe="blind_span_anchor_gap",
                                        accept_step_disagreement=True)
            if not blind["declined"] and blind.get("sweep_crossed"):
                return _blind_span_outcome(hole, domain, blind, candidate_obj.filePath, walk=walk)
        return _declined(hole, result.get("reason"), result.get("evidence"),
                         no_cut_probe_run=bool(hypotheses))
    verdict = _interior_verdict(result)
    if verdict == _orch().HOLE_PINNED_TO_AMBIGUOUS_ZONE_END:
        # Crossed or overlapping fronts: frames match both shifts, so the picture cannot
        # place the gap; the blind-span rule places it (shared silence, else right anchor).
        outcome = _blind_span_outcome(hole, domain, result, candidate_obj.filePath, walk=walk)
        outcome["pin_route"] = "sweep_fronts"
        return outcome
    if verdict == _orch().HOLE_RESOLVED:
        return _differing_span_outcome(hole, domain, result, candidate_obj.filePath, walk=walk)
    return _interior_outcome(hole, domain, result, verdict)


def _master_quiet(walk, times):
    """Whether the master is quiet in each 20 ms window: its RMS under BLIND_SILENCE_DB, the
    level `_blind_span_audio_point` already reads as the master's own silence."""
    return _window_levels(walk["master"], times, 0.0) < BLIND_SILENCE_DB


def _audio_continuous_span(hole, domain, sweep, before, after, walk):
    """Master frames [front, back) of a differing span the audio does not already carry.

    The audio walk places the step between the hole's edges (`hole["master_ms"]`): before the
    first the candidate plays at the before offset, after the second at the after offset, so
    differing frames outside them (a frame belongs to the side holding most of it) sit in an
    audio-continuous zone. Inside, a frame at either end where the master is quiet and the
    candidate, at that side's audio offset (not the picture shift: the candidate's audio may
    lag its own picture), is silent is common silence: nothing of the master is missing there
    either. A master fill keeps at least the shift gap (before - after frames). Without
    `walk` the sweep span is returned unchanged.

    Returns:
        (front, back) master frames, front <= back.
    """
    import numpy as np
    front, back = sweep
    if walk is None or back <= front:
        return front, back
    frame_ms = float(domain["frame_ms"])
    frame_s = frame_ms / 1000.0
    low_ms, high_ms = hole["master_ms"]
    first, last = front, back
    edge_first, edge_last = int(round(low_ms / frame_ms)), int(round(high_ms / frame_ms))
    if edge_first < back and edge_last > front:
        # Edges that miss the span entirely place the step elsewhere than the picture does:
        # nothing there to bound, only the silence below is read.
        first, last = max(front, edge_first), min(back, edge_last)

    def common_silence(frame, offset_ms):
        start = frame * frame_s
        times = np.arange(start, start + frame_s - BLIND_SILENCE_WIN_S + 1e-9,
                          BLIND_SILENCE_HOP_S)
        if not len(times):
            times = np.array([start])
        return bool(_master_quiet(walk, times).all()
                    and _candidate_silent(walk, times, offset_ms / 1000.0).all())
    try:
        while last > first and common_silence(last - 1, hole["offset_after_ms"]):
            last -= 1
        while first < last and common_silence(first, hole["offset_before_ms"]):
            first += 1
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"repair: audio_continuous_probe_failed ({type(error).__name__}: "
                      f"{error})\n")
        return front, back
    need = max(0, before - after)
    if last - first < need:
        last = min(back, first + need)
        first = max(front, last - need)
    return first, last


def _differing_span_outcome(hole, domain, result, candidate_path, walk=None):
    """Resolve a cut whose sweep fronts did not cross: fill every unmatched master frame.

    Frames that differ between the files (a candidate fade, dim or black frame against master
    content, master black against candidate content, a retouched shot) are cut and filled from
    the master at any width. The unmatched master span
    [F, Bk) between the fronts is exactly those frames plus any master-only content, so it is
    the fill, and the candidate's own unmatched frames are removed.

    A candidate excess (after shift > before shift) is not tied to the fill: when frames next
    to a front match under both shifts (`blind_before_front_frames`,
    `blind_after_front_frames`), the picture cannot place the removal inside them, so rule 1
    applies there -- inside the candidate's own silence when one fits, else at the fill (the
    blind run's right edge). The chosen frame is returned as `removal_frame`.

    Differing frames whose audio is continuous with the candidate are findings only, never
    filled (`_audio_continuous_span`): the span is first bounded by the hole's own audio edges,
    when they overlap it, then its ends lose every frame of silence common to the master and
    the candidate at that side's audio offset.
    """
    before, after = result["before_shift_frames"], result["after_shift_frames"]
    gap = after - before
    sweep = (result["master_start_frame"], result["master_end_frame"])
    front, back = _audio_continuous_span(hole, domain, sweep, before, after, walk)
    if (front, back) != sweep:
        tools.log_always(
            f"repair: differing_frames_audio_continuous sweep_frames=[{sweep[0]}, {sweep[1]}] "
            f"fill_frames=[{front}, {back}] shifts=[{before}, {after}] "
            f"audio_edges_ms=[{round(hole['master_ms'][0], 2)}, "
            f"{round(hole['master_ms'][1], 2)}] decision=picture_finding_only "
            f"for {candidate_path}\n")
        result = dict(result, master_start_frame=front, master_end_frame=back,
                      candidate_start_frame=front + before, candidate_end_frame=back + after)
    outcome = _interior_outcome(hole, domain, result, _orch().HOLE_RESOLVED)
    outcome["rule2_fill_frames"] = back - front
    tools.log_always(
        f"repair: differing_span_filled anchors=[{result['anchor_a_frame']}, "
        f"{result['anchor_b_frame']}] shifts=[{before}, {after}] "
        f"master_fill_frames=[{front}, {back}] candidate_removed_frames="
        f"[{result['candidate_start_frame']}, {result['candidate_end_frame']}] "
        f"blind_runs=[{result.get('blind_before_front_frames')}, "
        f"{result.get('blind_after_front_frames')}] for {candidate_path}\n")
    if gap <= 0:
        return outcome
    runs = []
    before_run = result.get("blind_before_front_frames") or 0
    after_run = result.get("blind_after_front_frames") or 0
    if before_run:
        runs.append(("before_fill", front - before_run, front))
    if after_run:
        runs.append(("after_fill", back, back + after_run))
    best = None
    for side, low, high in runs:
        found = _blind_span_audio_point(domain, gap, before, after, low, high, walk)
        if found is not None and (best is None
                                  or found[1][1] - found[1][0] > best[2][1] - best[2][0]):
            best = (side, found[0], found[1])
    if best is None:
        return outcome
    side, point, frames = best
    if (side == "before_fill" and point == front) or (side == "after_fill" and point == back):
        return outcome
    outcome["removal_frame"] = point
    tools.log_always(
        f"repair: video_undecided_blind_span anchor_a={result['anchor_a_frame']} "
        f"anchor_b={result['anchor_b_frame']} before_shift_frames={before} "
        f"after_shift_frames={after} gap_frames={gap} "
        f"gap_ms={round(gap * float(domain['frame_ms']), 3)} allowed_frames={frames} "
        f"decision=remove_from_candidate placement=candidate_silence side={side} "
        f"placed_at_frame={point} for {candidate_path}\n")
    return outcome


def _resolve_edge(hole, domain, master_obj, candidate_obj):
    """Resolve a head or tail hole with the one-anchor search and bounded outward walk.

    Calls `scene_anchor.locate_edge_boundary`. The bracket is clamped to at least one frame
    inside the timeline; one frame is enough to tell which side is common.
    """
    import scene_anchor
    edge = hole["kind"]
    quantum_ms = hole["quantum_ms"]
    frame_ms = float(domain["frame_ms"])
    timeline_ms = float(domain["master_timeline_ms"])
    if edge == "head":
        offset_ms = hole["offset_after_ms"]
        low_ms, high_ms = 0.0, max(float(hole["master_ms"][1]), frame_ms)
    else:
        offset_ms = hole["offset_before_ms"]
        low_ms = min(float(hole["master_ms"][0]), timeline_ms - frame_ms)
        high_ms = timeline_ms
    candidate_duration_ms = domain["candidate_equivalent_duration_ms"]
    _orch().step_launch("edge_walk", candidate=candidate_obj.filePath, edge=edge,
                bracket_ms=[round(low_ms, 2), round(high_ms, 2)],
                offset_ms=round(offset_ms, 3), master_timeline_ms=timeline_ms,
                candidate_equivalent_duration_ms=(None if candidate_duration_ms is None
                                                  else float(candidate_duration_ms)))
    try:
        result = scene_anchor.locate_edge_boundary(
            master_obj.filePath, candidate_obj.filePath,
            domain["master_rate"].numerator, domain["master_rate"].denominator,
            low_ms, high_ms, float(offset_ms), edge, timeline_ms,
            None if candidate_duration_ms is None else float(candidate_duration_ms),
            quantum_ms=float(quantum_ms), candidate_time_scale=domain["time_scale"],
            shift_search_frames=_shift_search_frames(domain, quantum_ms),
            deadline=hole.get("deadline"))
    except Exception as error:                                           # noqa: BLE001
        result = {"declined": True, "reason": f"resolver_raised:{type(error).__name__}",
                  "evidence": str(error)[:300]}
    _orch().step_result("edge_walk", candidate=candidate_obj.filePath, edge=edge,
                declined=result["declined"], reason=result.get("reason"),
                termination=result.get("termination"), anchor=result.get("anchor_frame"),
                shift=result.get("shift_frames"), nominal_shift=result.get("nominal_shift_frames"),
                boundary=result.get("boundary_frame"), walked=result.get("walked_frames"),
                mismatch_run=result.get("mismatch_run"),
                max_mismatch_run=result.get("max_mismatch_run"),
                addition_frames=result.get("addition_frames"),
                evidence=result.get("evidence"))
    if result["declined"] and result.get("reason") in EDGE_PIXEL_FALLBACK_REASONS:
        shift = hole.get("picture_shift_frames")
        if shift is None:
            shift = int(round(float(offset_ms) / frame_ms))
        try:
            fallback = _pixel_edge_walk(hole, domain, master_obj, candidate_obj, shift)
        except Exception as error:                                       # noqa: BLE001
            tools.dev_log(f"repair: pixel_edge_walk_failed ({type(error).__name__}: "
                          f"{error})\n")
            fallback = None
        if fallback is not None:
            result = fallback
    if result["declined"]:
        return _declined(hole, result.get("reason"), result.get("evidence"))
    if result.get("method") != "pixel_edge_walk":
        # pHash cannot see a fade and stops on a few odd frames: the boundary is re-read on
        # pixels at the anchor's own shift, from the common side of the pHash boundary.
        boundary_ms = float(result["boundary_frame"]) * frame_ms
        bracket = [0.0, boundary_ms] if edge == "head" else [boundary_ms, timeline_ms]
        try:
            refined = _pixel_edge_walk(dict(hole, master_ms=bracket), domain, master_obj,
                                       candidate_obj, result["shift_frames"])
        except Exception as error:                                       # noqa: BLE001
            tools.dev_log(f"repair: pixel_edge_walk_failed ({type(error).__name__}: "
                          f"{error})\n")
            refined = None
        if refined is not None:
            result = dict(refined, anchor_frame=result.get("anchor_frame"),
                          anchor_side=result.get("anchor_side"),
                          anchor_n_frames=result.get("anchor_n_frames"),
                          nominal_shift_frames=result.get("nominal_shift_frames"),
                          geometry=result.get("geometry"))

    boundary = result["boundary_frame"]
    shift = result["shift_frames"]
    termination = result["termination"]
    master_last = result["master_last_frame"]
    if edge == "head":
        # Master [0, boundary) is not common; the candidate's is [0, boundary + shift).
        master_start, master_end = 0, boundary
        candidate_start_eq, candidate_end_eq = 0, boundary + shift
        master_non_common = boundary
        candidate_non_common = boundary + shift
    else:
        # Master (boundary, last] is not common. The candidate's end is known only on
        # `candidate_exhausted`; otherwise plan application reads it.
        master_start, master_end = boundary + 1, master_last + 1
        candidate_start_eq = boundary + 1 + shift
        candidate_end_eq = (boundary + 1 + shift
                            if termination == _orch().EDGE_CANDIDATE_EXHAUSTED else None)
        master_non_common = master_last - boundary
        candidate_non_common = (0 if termination == _orch().EDGE_CANDIDATE_EXHAUSTED else None)

    # Master frames added to the output (a replacement or addition; a trim adds nothing).
    added_frames = 0 if termination == _orch().EDGE_MASTER_EXHAUSTED else max(0, master_non_common)
    added_seconds = float(Fraction(added_frames) / domain["master_rate"])

    status = termination
    # Both files reach their edge at the same step: nothing to trim or add.
    if master_non_common == 0 and candidate_non_common == 0:
        status = _orch().HOLE_NO_CUT_CONFIRMED
    return {
        "modality": _orch().MODALITY, "status": status, "kind": edge, "why_token": hole["why_token"],
        "cause": None, "termination": termination, "net_kind": result["net_kind"],
        "grid": f"{result['grid']['num']}/{result['grid']['den']}",
        "anchor_frame": result["anchor_frame"], "anchor_side": result["anchor_side"],
        "anchor_n_frames": result["anchor_n_frames"],
        "shift_frames": shift, "nominal_shift_frames": result["nominal_shift_frames"],
        "boundary_frame": boundary, "walked_frames": result["walked_frames"],
        "mismatch_run": result["mismatch_run"],
        "max_mismatch_run": result["max_mismatch_run"],
        "addition_frames": result["addition_frames"], "addition_ms": result["addition_ms"],
        "master_last_frame": master_last,
        "master_start_frame": master_start, "master_end_frame": master_end,
        "candidate_start_frame_equivalent": candidate_start_eq,
        "candidate_end_frame_equivalent": candidate_end_eq,
        "candidate_start_frame": _candidate_native_frame(candidate_start_eq, domain),
        "candidate_end_frame": _candidate_native_frame(candidate_end_eq, domain),
        "edge_addition_frames": added_frames,
        "edge_addition_seconds": added_seconds,
        "master_start_ms": _exact_ms_of_frame(master_start, domain),
        "master_end_ms": _exact_ms_of_frame(master_end, domain),
        "boundary_ms": _exact_ms_of_frame(boundary, domain),
        "video_shift_ms_frame_quantised": _exact_ms_of_frame(shift, domain),
        **_audio_offsets(hole),
        "geometry": (result.get("geometry") or {}).get("verdict"),
        "evidence": result.get("evidence"),
    }


# Reasons for which a one-anchor edge search found only self-similar (static/black) content:
# the picture cannot seat an anchor there, so the edge is read on pixels at a known shift.
EDGE_PIXEL_FALLBACK_REASONS = ("edge_anchor_uninformative", "search_window_ceiling_reached")
# Frames read on the common side of an edge bracket before the walk outward begins.
EDGE_PIXEL_MARGIN_FRAMES = 48


def _pixel_edge_walk(hole, domain, master_obj, candidate_obj, shift):
    """Walk an edge on pixels at a known shift when no anchor could be seated, or None.

    A static or black edge the anchor search cannot seat is never a reason to decline. The
    shift comes from the picture next to the edge (the nearest resolved
    cut's own shift, `hole["picture_shift_frames"]`), else the zone's audio offset. From the
    common side of the bracket the walk goes outward one master frame at a time and stops on
    EDGE_WALK_SUSTAINED_MISMATCH_FRAMES differing frames (rule 2: they are filled from the
    master), or where either file ends. The common side must first show
    PIXEL_TRUST_RUN_FRAMES frames reading the same picture. Returns a
    `scene_anchor.locate_edge_boundary`-shaped answer.
    """
    import scene_anchor
    edge = hole["kind"]
    frame_ms = float(domain["frame_ms"])
    rate = domain["master_rate"]
    master_last = int(Fraction(str(domain["master_timeline_ms"])) / 1000 * rate) - 1
    if edge == "head":
        stop = min(master_last + 1,
                   int(math.ceil(float(hole["master_ms"][1]) / frame_ms)) + EDGE_PIXEL_MARGIN_FRAMES)
        first, count = 0, stop
    else:
        first = max(0, int(float(hole["master_ms"][0]) / frame_ms) - EDGE_PIXEL_MARGIN_FRAMES)
        count = master_last + 1 - first
    if count <= 0 or count > PIXEL_FRONT_MAX_FRAMES or first + shift < 0 and edge == "tail":
        return None
    read_first = max(first, -shift)
    read_first = max(read_first, 1 - shift)
    readings = _pixel_readings(domain, master_obj, candidate_obj, read_first,
                               first + count - read_first, [shift - 1, shift, shift + 1])
    if readings is None:
        return None
    row = readings[shift]
    common_side = (range(min(len(row), EDGE_PIXEL_MARGIN_FRAMES)) if edge == "tail"
                   else range(max(0, len(row) - EDGE_PIXEL_MARGIN_FRAMES), len(row)))
    same = _tolerant_flags(readings, shift, common_side)
    # Candidate frames that do not exist (before its start, after its end) read as absent.
    candidate_frames = domain.get("candidate_equivalent_duration_ms")
    candidate_last = (None if candidate_frames is None
                      else int(float(candidate_frames) / frame_ms) - 1)
    order = list(range(len(same)) if edge == "tail" else range(len(same) - 1, -1, -1))
    common = [same[i] for i in order]
    # The walk starts at the first PIXEL_TRUST_RUN_FRAMES-long run of common picture: a
    # bracket the audio opened early can still hold the last cut's other side.
    start = next((i for i in range(len(common) - PIXEL_TRUST_RUN_FRAMES + 1)
                  if all(common[i:i + PIXEL_TRUST_RUN_FRAMES])), None)
    if start is None:
        return None
    boundary, run, termination, walked, max_run = None, 0, None, 0, 0
    for index in order[start:]:
        frame = read_first + index
        if candidate_last is not None and frame + shift > candidate_last:
            termination = _orch().EDGE_CANDIDATE_EXHAUSTED
            break
        walked += 1
        if same[index]:
            boundary, run = frame, 0
            continue
        run += 1
        max_run = max(max_run, run)
        if run >= scene_anchor.EDGE_WALK_SUSTAINED_MISMATCH_FRAMES and boundary is not None:
            termination = _orch().EDGE_SUSTAINED_MISMATCH
            break
    if boundary is None:
        return None
    if termination is None:
        termination = (_orch().EDGE_CANDIDATE_EXHAUSTED if edge == "head" and read_first > 0
                       else _orch().EDGE_MASTER_EXHAUSTED)
    if edge == "head":
        addition = boundary if termination == _orch().EDGE_CANDIDATE_EXHAUSTED else None
    else:
        addition = (master_last - boundary if termination == _orch().EDGE_CANDIDATE_EXHAUSTED
                    else None)
    net_kind = {_orch().EDGE_CANDIDATE_EXHAUSTED: "master_addition",
                _orch().EDGE_MASTER_EXHAUSTED: "candidate_excess_trimmed"}.get(termination,
                                                                       "master_replacement")
    tools.log_always(
        f"repair: pixel_edge_walk edge={edge} shift_frames={shift} boundary_frame={boundary} "
        f"termination={termination} walked={walked} for {candidate_obj.filePath}\n")
    return {"declined": False, "method": "pixel_edge_walk", "edge": edge,
            "grid": {"num": rate.numerator, "den": rate.denominator},
            "anchor_frame": None, "anchor_side": None, "anchor_n_frames": None,
            "shift_frames": shift, "nominal_shift_frames": shift, "boundary_frame": boundary,
            "walked_frames": walked, "mismatch_run": run, "max_mismatch_run": max_run,
            "termination": termination, "net_kind": net_kind, "addition_frames": addition,
            "addition_ms": None if addition is None else _exact_ms_of_frame(addition, domain),
            "master_last_frame": master_last, "geometry": None,
            "evidence": f"pixel edge walk at shift {shift} (no anchor could be seated)"}


def resolve_hole(hole, master_obj, candidate_obj, work_dir, walk=None):
    """Resolve one hole to exact frames.

    Interior holes go to `_resolve_interior`, head and tail holes to `_resolve_edge`. The hole
    must carry `frame_domain` and `quantum_ms`; it gets HOLE_BUDGET_S, capped by the repair's
    deadline. `work_dir` is unused (the resolvers decode through pipes). `walk` (the pair's
    already-decoded audio, when the caller has it) is forwarded to `_resolve_interior` only,
    to place a blind span inside a shared silence.
    """
    domain = hole.get("frame_domain")
    if domain is None:
        return _declined(hole, "frame_domain_absent")
    repair_deadline = domain.get("repair_deadline")
    if repair_deadline is not None and time.monotonic() > repair_deadline:
        return _declined(hole, "repair_budget_exceeded")
    deadline = time.monotonic() + _orch().HOLE_BUDGET_S
    hole = dict(hole, deadline=(deadline if repair_deadline is None
                                else min(deadline, repair_deadline)))
    if hole["kind"] == "interior":
        outcome = _resolve_interior(hole, domain, master_obj, candidate_obj, walk=walk)
    elif hole["kind"] in ("head", "tail"):
        outcome = _resolve_edge(hole, domain, master_obj, candidate_obj)
    else:
        outcome = _declined(hole, f"no_resolver_for_kind:{hole['kind']}")
    # An over-budget hole declines even if its last call answered.
    if outcome["status"] != _orch().HOLE_DECLINED and time.monotonic() > hole["deadline"]:
        outcome = _declined(hole, "hole_budget_exceeded",
                            evidence=f"answered {outcome['status']} past the "
                                     f"{_orch().HOLE_BUDGET_S} s budget")
    outcome["audio_master_frames"] = [_master_frame_of_ms(hole["master_ms"][0], domain),
                                      _master_frame_of_ms(hole["master_ms"][1], domain)]
    outcome["audio_candidate_frames"] = [
        _candidate_native_frame_of_ms(hole["candidate_ms"][0], domain),
        _candidate_native_frame_of_ms(hole["candidate_ms"][1], domain)]
    outcome["audio_step_ms"] = (None if hole["step_ms"] is None else round(hole["step_ms"], 3))
    return outcome


def _resolve_logged(candidate_path, index, hole, domain, master_obj, candidate_obj, work_dir,
                    walk=None):
    """Run `resolve_hole` with step logging; an out-of-vocabulary status becomes a decline."""
    _orch().step_launch("resolve_hole", candidate=candidate_path, hole=index, kind=hole["kind"],
                why=hole["why_token"], master_span_s=round(hole["master_span_seconds"], 3),
                candidate_span_s=round(hole["candidate_span_seconds"], 3),
                step_ms=(round(hole["step_ms"], 1) if hole["step_ms"] is not None else None),
                offsets_ms=[None if hole.get("offset_before_ms") is None
                            else round(hole["offset_before_ms"], 3),
                            None if hole.get("offset_after_ms") is None
                            else round(hole["offset_after_ms"], 3)],
                union_of=hole.get("union_of"), cluster=hole.get("cluster_id"),
                origin=hole.get("origin", "alignment"))
    started = time.time()
    outcome = resolve_hole(dict(hole, frame_domain=domain), master_obj, candidate_obj, work_dir,
                           walk=walk)
    if outcome["status"] not in _orch().HOLE_STATUSES_WITH_FRAMES + (_orch().HOLE_DECLINED,):
        tools.log_always(f"repair: orchestrator UNVOCABULARISED hole status="
                         f"{outcome['status']} hole={index} for {candidate_path} -- "
                         f"treated as declined\n")
        outcome = dict(outcome, status=_orch().HOLE_DECLINED, cause="hole_resolution_declined",
                       resolver_reason=f"unvocabularised_status:{outcome['status']}")
    _orch().step_result("resolve_hole", candidate=candidate_path, hole=index, kind=hole["kind"],
                status=outcome["status"], cause=outcome.get("cause"),
                resolver_reason=outcome.get("resolver_reason"),
                termination=outcome.get("termination"),
                master_frames=[outcome.get("master_start_frame"),
                               outcome.get("master_end_frame")],
                master_ms=[outcome.get("master_start_ms"), outcome.get("master_end_ms")],
                audio_offsets_ms=[outcome.get("audio_offset_before_ms"),
                                  outcome.get("audio_offset_after_ms")],
                candidate_frames=[outcome.get("candidate_start_frame"),
                                  outcome.get("candidate_end_frame")],
                candidate_frames_equivalent=[
                    outcome.get("candidate_start_frame_equivalent"),
                    outcome.get("candidate_end_frame_equivalent")],
                audio_master_frames=outcome.get("audio_master_frames"),
                audio_candidate_frames=outcome.get("audio_candidate_frames"),
                audio_step_ms=outcome.get("audio_step_ms"),
                anchors=[outcome.get("anchor_a_frame", outcome.get("anchor_frame")),
                         outcome.get("anchor_b_frame")],
                shifts=[outcome.get("before_shift_frames", outcome.get("shift_frames")),
                        outcome.get("after_shift_frames")],
                walks=[outcome.get("forward_walk_frames", outcome.get("walked_frames")),
                       outcome.get("backward_walk_frames")],
                span_frames=outcome.get("span_frames"),
                net_kind=outcome.get("net_kind"),
                edge_addition_frames=outcome.get("edge_addition_frames"),
                seconds=round(time.time() - started, 2))
    # Refuted proposals and no-cut closures are decisions on the material: logged unconditionally.
    if outcome.get("refuted_proposal"):
        refuted = outcome["refuted_proposal"]
        _orch().step_result("refuted_proposal", candidate=candidate_path, hole=index, **refuted)
        tools.log_always(
            f"repair: refuted_proposal hole={index} "
            f"master_frames={refuted['master_position_frames']} "
            f"master_ms={refuted['master_position_ms']} "
            f"proposed_step_ms={refuted['audio_step_ms']} "
            f"proposed_step_points={refuted['audio_step_points']} "
            f"proposal_shifts={refuted['proposal_shifts']} "
            f"proposal_span_matches_before={refuted['proposal_span_matches_before']} "
            f"proposal_span_matches_after={refuted['proposal_span_matches_after']} "
            f"claimed_span_matches={refuted['claimed_span_matches']} "
            f"span_test_matches={refuted.get('span_test_matches')} "
            f"surviving_shift={refuted['video_single_shift']} "
            f"surviving_shift_walk_frames={refuted['surviving_shift_walk_frames']} "
            f"surviving_shift_span_frames={refuted['surviving_shift_span_frames']} "
            f"video_verdict={refuted['video_verdict']} probe={refuted['video_probe']} "
            f"for {candidate_path}\n")
    if outcome["status"] == _orch().HOLE_NO_CUT_CONFIRMED:
        tools.log_always(
            f"repair: no_cut_confirmed hole={index} kind={hole['kind']} "
            f"origin={hole.get('origin', 'alignment')} "
            f"audio_master_frames={outcome.get('audio_master_frames')} "
            f"audio_step_ms={outcome.get('audio_step_ms')} "
            f"shift={outcome.get('before_shift_frames', outcome.get('shift_frames'))} "
            f"anchors={[outcome.get('anchor_a_frame', outcome.get('anchor_frame')), outcome.get('anchor_b_frame')]} "
            f"walks={[outcome.get('forward_walk_frames', outcome.get('walked_frames')), outcome.get('backward_walk_frames')]} "
            f"for {candidate_path}\n")
    if outcome["status"] == _orch().HOLE_PINNED_TO_AMBIGUOUS_ZONE_END:
        _orch().step_result("boundary_pinned_to_ambiguous_zone_end", candidate=candidate_path,
                    hole=index, cause="static_span_ambiguity",
                    pin_route=outcome.get("pin_route", "sweep_fronts"),
                    forward_walk_frames=outcome["forward_walk_frames"],
                    backward_walk_frames=outcome["backward_walk_frames"],
                    span_frames=outcome["span_frames"],
                    ambiguous_frames=outcome["ambiguous_frames"],
                    pin_frame=outcome["pin_frame"],
                    anchor_b_frame=outcome["anchor_b_frame"],
                    audio_step_frames=outcome.get("audio_step_frames"),
                    ambiguous_shift_span_a=(outcome.get("anchor_a_ambiguous") or {}).get(
                        "ambiguous_shift_span"),
                    ambiguous_shift_span_b=(outcome.get("anchor_b_ambiguous") or {}).get(
                        "ambiguous_shift_span"),
                    fill_master_frames=[outcome["master_start_frame"],
                                        outcome["master_end_frame"]])
    return outcome



# ---------------------------------------------------------------------------
# Step 4: audio bounds, video pins
# ---------------------------------------------------------------------------

def _audio_entry(video_obj, language, stream):
    for entry in (getattr(video_obj, "audios", None) or {}).get(language) or []:
        if str(entry.get("StreamOrder")) == str(stream):
            return entry
    return None


def hole_sanity(holes, domain, candidate_path):
    """Check every hole before any scan.

    Returns:
        None, or (cause, reason) for the first impossible hole: `hole_outside_master_timeline`,
        `hole_step_exceeds_duration` (|step| longer than the shorter file) or
        `interior_hole_exceeds_budget` (wider than INTERIOR_HOLE_MAX_SPAN_S).
    """
    timeline = float(domain["master_timeline_ms"])
    frame = float(domain["frame_ms"])
    candidate = domain.get("candidate_equivalent_duration_ms")
    shorter = timeline if candidate is None else min(timeline, float(candidate))
    for index, hole in enumerate(holes):
        low, high = hole["master_ms"]
        verdict = None
        if low >= timeline or (hole["kind"] == "interior" and high > timeline + frame):
            verdict = ("hole_outside_master_timeline",
                       f"hole {index} ({hole['kind']}) spans master [{round(low, 1)}, "
                       f"{round(high, 1)}] ms beyond the master video's {round(timeline, 1)} ms")
        elif hole["step_ms"] is not None and abs(hole["step_ms"]) > shorter:
            verdict = ("hole_step_exceeds_duration",
                       f"hole {index} carries a step of {round(hole['step_ms'], 1)} ms, longer "
                       f"than the shorter file ({round(shorter, 1)} ms)")
        elif hole["kind"] == "interior" and (high - low) / 1000.0 > _orch().INTERIOR_HOLE_MAX_SPAN_S:
            verdict = ("interior_hole_exceeds_budget",
                       f"interior hole {index} spans {round((high - low) / 1000.0, 1)} s of the "
                       f"master, over the {_orch().INTERIOR_HOLE_MAX_SPAN_S} s bound")
        if verdict is not None:
            _orch().step_result("hole_sanity", candidate=candidate_path, hole=index, kind=hole["kind"],
                        master_ms=[round(low, 2), round(high, 2)], step_ms=hole["step_ms"],
                        cause=verdict[0])
            return verdict
    return None


# The resolver reasons that are a time bound, not a reading.
BUDGET_REASONS = ("repair_budget_exceeded", "hole_budget_exceeded", "decoder_timeout")


def budget_cause(outcome, domain):
    """Return the time bound that stopped this video call, or None.

    The repair's budget takes precedence over the hole's, then a decoder timeout.
    """
    reason = outcome.get("resolver_reason")
    if reason not in BUDGET_REASONS:
        return None
    deadline = domain.get("repair_deadline")
    if reason == "repair_budget_exceeded" or (deadline is not None
                                              and time.monotonic() > deadline):
        return "repair_budget_exceeded"
    return reason


def log_partial_plan(candidate_path, cause, placed):
    """Log the partial plan placed before a budget overrun; `placed` holds (what, decision, where)."""
    tools.log_always(f"repair: partial_plan cause={cause} placed={placed} for {candidate_path}\n")


def started_slices(video_s, slice_s):
    """Return the number of started `slice_s` slices in `video_s` seconds, at least one."""
    return max(1, math.ceil((video_s or 0.0) / slice_s))


def repair_budget_seconds(master_obj):
    """Return (budget_s, video_s): REPAIR_BUDGET_PER_SLICE_S per started slice, capped."""
    video_ms = _video_duration_ms(master_obj)
    video_s = None if video_ms is None else float(video_ms) / 1000.0
    slices = started_slices(video_s, _orch().REPAIR_BUDGET_SLICE_S)
    return min(slices * _orch().REPAIR_BUDGET_PER_SLICE_S, _orch().REPAIR_BUDGET_CAP_S), video_s


def max_holes_per_couple(master_obj):
    """Return the hole cap per couple: MAX_HOLES_PER_COUPLE per started HOLE_BUDGET_SLICE_S."""
    video_ms = _video_duration_ms(master_obj)
    video_s = None if video_ms is None else float(video_ms) / 1000.0
    return started_slices(video_s, _orch().HOLE_BUDGET_SLICE_S) * _orch().MAX_HOLES_PER_COUPLE


def _budget_terminal(candidate_path, step, budget_s):
    """Decline the candidate because the repair's budget ran out between two steps."""
    _orch()._plan_line("none", candidate_path, step=step, cause="repair_budget_exceeded")
    return _orch()._terminal(candidate_path, "no_plan", "repair_budget_exceeded",
                     f"the repair's {budget_s} s budget ran out after the {step} step -- a "
                     f"statement about this run's cost; declined, retried at the next run")


def _speed_chain(audio, speed_ratio, engine="asetrate"):
    """Return a track's speed filter chain at `speed_ratio`, or None when the ratio is None."""
    if speed_ratio is None:
        return None
    import merge_video_resample
    rate = (audio.get("ffprobe") or {}).get("sample_rate") or audio.get("SamplingRate")
    return merge_video_resample.build_transform_chain(int(float(rate)), speed_ratio, engine)[0]


def reference_walk(reference, holes, master_obj, candidate_obj, language, speed_ratio,
                   candidate_path, deadline=None, engine="asetrate", frame_ms=None):
    """Run the millisecond audio walk on the reference couple, on the file clock.

    Seeded by every alignment offset of the reference couple's zones and the union's holes.
    A whole-track read stopped by `deadline` re-raises (cause `repair_budget_exceeded`).
    `frame_ms` (the pair's own `domain['frame_ms']`, when measured) is passed to
    `audio_walk.change_points` so a sub-frame step can be placed in a shared silence.

    Returns:
        (walk, None) or (None, reason); the walk keeps both decoded tracks.
    """
    import audio_walk
    master_stream, candidate_stream = reference["couple"].split("x")
    master_audio = _audio_entry(master_obj, language, master_stream)
    candidate_audio = _audio_entry(candidate_obj, language, candidate_stream)
    alignment, fold = reference["alignment"], reference["fold"]
    _zones, detail = coalesce_same_offset_zones(alignment.get("zones") or [],
                                                alignment.get("zones_detail") or [])
    seeds = sorted({round(zone["offset_points"] * alignment["quantum_ms"] + fold["delta_ms"], 3)
                    for zone in detail}
                   | {round(value, 3) for hole in holes
                      for value in (hole["offset_before_ms"], hole["offset_after_ms"])
                      if value is not None})
    _orch().step_launch("audio_walk", candidate=candidate_path, couple=reference["couple"],
                seeds=seeds, window_s=audio_walk.WALK_WINDOW_S, hop_s=audio_walk.WALK_HOP_S,
                search_ms=audio_walk.WALK_SEARCH_MS)
    started = time.time()
    if master_audio is None or candidate_audio is None:
        return None, f"reference couple {reference['couple']} has no {language} audio entry"
    try:
        scale = speed_ratio if speed_ratio is not None else Decimal(1)
        # atempo pairs are walked on speech envelopes.
        envelope = engine == "atempo" and speed_ratio is not None
        master = audio_walk.read_on_file_clock(master_obj, master_audio, deadline=deadline,
                                               envelope=envelope)
        candidate = audio_walk.read_on_file_clock(
            candidate_obj, candidate_audio, _speed_chain(candidate_audio, speed_ratio, engine),
            scale, deadline=deadline, envelope=envelope)
    except Exception as error:                                           # noqa: BLE001
        if getattr(error, "cause", None) == "repair_budget_exceeded":
            raise
        return None, f"the comparison tracks could not be read ({type(error).__name__}: {error})"
    rows = audio_walk.walk(master, candidate, seeds)
    found, outliers = audio_walk.levels(rows)
    points = audio_walk.change_points(master, candidate, found, frame_ms=frame_ms)
    seconds = time.time() - started
    audio_walk.log_walk(candidate_path, rows, found, points, seconds)
    _orch().step_result("audio_walk", candidate=candidate_path, n_windows=len(rows),
                n_levels=len(found), n_outliers=len(outliers),
                levels=[(lv["t_first"], lv["t_last"], lv["off_ms"], lv["mad_ms"]) for lv in found],
                change_points=[(p["a_ms"], p["b_ms"], p["jump_ms"], p["kind"],
                                (p.get("edges") or {}).get("interval")) for p in points],
                seconds=round(seconds, 1))
    if not found:
        return None, "the walk measured no level: no window of the comparison tracks matched"
    return {"master": master, "candidate": candidate, "rows": rows, "levels": found,
            "points": points, "seeds": seeds, "master_audio": master_audio,
            "candidate_audio": candidate_audio, "master_stream": master_stream,
            "candidate_stream": candidate_stream,
            "master_audio_end_s": len(master) / audio_walk.WALK_RATE}, None


def _frame_s(frame, domain):
    return float(Fraction(frame) / domain["master_rate"])


def head_content_edge(placed_s, decision, frame_s, lead):
    """Keep the head edge from putting the candidate's silent lead-in over audible master content.

    A placement earlier than the candidate's first sound (`audio_walk.head_lead_in`) is raised
    to it. A head edge within one frame of the start carries no fill unless the lead-in leaves
    master content uncovered.

    Returns:
        (head_end_s or None, decision)
    """
    if placed_s is None:
        return None, decision
    if lead is not None and placed_s < lead["content_edge_s"]:
        placed_s, decision = lead["content_edge_s"], f"{decision}_raised_to_candidate_first_sound"
    if placed_s <= frame_s and lead is None:
        return None, decision
    return placed_s, decision


def _edge_blind_point(kind, outcome, domain, master_obj, candidate_obj, walk):
    """Rule 1 at an edge: move a head/tail boundary inside the candidate's own silence.

    Frames next to the boundary on the common side that are flat (black) on the master and
    the same picture in the candidate are a span the picture cannot place the cut in. When
    the candidate (at the edge's shift) is not silent at the walk's own boundary, the cut
    moves into the longest silent stretch of that span, at its end nearest that boundary, so
    no candidate sound is cut off over the master's fill. Returns the new boundary frame
    (head: first common frame; tail: first filled frame), or None to keep the walk's own.
    """
    import frame_hash
    import numpy as np
    shift = outcome.get("shift_frames")
    if walk is None or shift is None:
        return None
    edge = outcome["master_end_frame"] if kind == "head" else outcome["master_start_frame"]
    if kind == "head" and edge <= 0 or kind == "tail" and edge > outcome["master_last_frame"]:
        return None
    first = edge if kind == "head" else max(0, edge - PIXEL_FRONT_MARGIN_FRAMES)
    if first + shift - 1 < 0:
        return None
    readings = _pixel_readings(domain, master_obj, candidate_obj, first,
                               PIXEL_FRONT_MARGIN_FRAMES, [shift - 1, shift, shift + 1])
    if readings is None:
        return None
    row = readings[shift]
    same = _tolerant_flags(readings, shift, range(len(row)))
    order = range(len(row)) if kind == "head" else range(len(row) - 1, -1, -1)
    run = 0
    for index in order:
        if not (same[index] and row[index][2] < frame_hash.FLAT_STD):
            break
        run += 1
    if run == 0:
        return None
    low, high = (edge, edge + run) if kind == "head" else (edge - run, edge)
    frame_s = float(domain["frame_ms"]) / 1000.0
    joint = BLIND_SILENCE_WIN_S
    times = np.arange(max(0.0, low * frame_s - joint), high * frame_s + joint, BLIND_SILENCE_HOP_S)
    silent = _candidate_silent(walk, times, shift * frame_s)
    flags = []
    for point in range(low, high + 1):
        near = (times >= point * frame_s - joint - 1e-9) & (times <= point * frame_s + 1e-9)
        flags.append(bool(near.any() and silent[near].all()))
    if flags[edge - low]:
        return None
    found = _centre_of_longest(flags, low)
    if found is None:
        return None
    # The silent frame nearest the walk's own boundary: the smallest move off the picture's
    # answer that keeps the cut inside the candidate's silence.
    point = found[1][0] if kind == "head" else found[1][1]
    tools.log_always(f"repair: edge_blind_span edge={kind} walk_boundary={edge} "
                     f"blind_frames=[{low}, {high}] candidate_silence_frames={found[1]} "
                     f"placed_at_frame={point} for {candidate_obj.filePath}\n")
    return point


def audio_edges(walk, holes, domain, master_obj, candidate_obj, work_dir, candidate_path,
                shift_hints=None):
    """Place the head and tail fill edges from the audio walk, pinned to a video frame if close.

    `audio_walk.single_edge` walks outward from the first/last level at 20 ms: a witness only,
    to tell the resolver where to search and to confirm its answer in the dev log. Where the
    one-anchor walk (`scene_anchor.locate_edge_boundary`, via `_resolve_edge`) concludes, its own
    frame is delivered unconditionally -- never nudged onto the audio edge, never raised for the
    candidate's lead-in, which was itself an audio placement. Only where no such hole was raised
    for this edge (nothing flagged it as needing a frame-exact search) does the walk's own edge
    stand, with the lead-in floor still applied to it. A one-anchor walk that could not conclude
    declines by name instead of falling back to the audio's edge.

    `shift_hints` ({"head"|"tail": frames}) carries the picture shift of the nearest resolved
    cut, used when the edge's own anchor cannot be seated (`_pixel_edge_walk`).

    Returns:
        (head_end_s, tail_start_s, refusal); an edge is None when no fill is needed there.
    """
    import audio_walk
    master, candidate = walk["master"], walk["candidate"]
    first, last = walk["levels"][0], walk["levels"][-1]
    frame = float(domain["frame_ms"]) / 1000.0
    reach = audio_walk.WALK_WINDOW_S + audio_walk.WALK_HOP_S + _orch().HOLE_MERGE_WINDOW_SECONDS
    # Start inside the level's measured window: a window mid-point can lie in silence.
    window = audio_walk.WALK_WINDOW_S
    head = audio_walk.single_edge(master, candidate, first["off_ms"], first["t_first"] + window,
                                  max(0.0, first["t_first"] - reach), "head")
    if head is None and first.get("off_first_ms", first["off_ms"]) != first["off_ms"]:
        head = audio_walk.single_edge(master, candidate, first["off_first_ms"],
                                      first["t_first"] + window,
                                      max(0.0, first["t_first"] - reach), "head")
        tools.dev_log(f"repair: audio_edge head retry at local offset "
                      f"{first['off_first_ms']} {'found ' + str(head['edge_s']) if head else 'also failed, falling back to the walk window edge'} "
                      f"for {candidate_path}\n")
    tail = audio_walk.single_edge(master, candidate, last["off_ms"], last["t_last"],
                                  min(walk["master_audio_end_s"], last["t_last"] + reach), "tail")
    if tail is None and last.get("off_last_ms", last["off_ms"]) != last["off_ms"]:
        tail = audio_walk.single_edge(master, candidate, last["off_last_ms"], last["t_last"],
                                      min(walk["master_audio_end_s"], last["t_last"] + reach),
                                      "tail")
        tools.dev_log(f"repair: audio_edge tail retry at local offset "
                      f"{last['off_last_ms']} {'found ' + str(tail['edge_s']) if tail else 'also failed, falling back to the walk window edge'} "
                      f"for {candidate_path}\n")
    # Without a fine edge, the level's measured windows bound the common content.
    lead = audio_walk.head_lead_in(master, candidate, first["off_ms"], first["t_first"] + window)
    head_s = first["t_first"] if head is None else head["edge_s"]
    tail_s = (last["t_last"] + window) if tail is None else tail["edge_s"]
    edge_source = {"head": "walk_window_edge" if head is None else "audio_edge",
                   "tail": "walk_window_edge" if tail is None else "audio_edge"}
    by_kind = {hole["kind"]: (index, hole) for index, hole in enumerate(holes)
               if hole["kind"] in ("head", "tail")}
    # Head and tail are independent holes (one anchor each): their exact-frame search runs
    # together on the shared repair pool instead of one after the other.
    edge_jobs = {}
    for kind, level in (("head", first), ("tail", last)):
        if kind in by_kind:
            index, hole = by_kind[kind]
            offsets = ({"offset_after_ms": level["off_ms"]} if kind == "head"
                       else {"offset_before_ms": level["off_ms"]})
            if (shift_hints or {}).get(kind) is not None:
                offsets["picture_shift_frames"] = shift_hints[kind]
            edge_jobs[kind] = (index, dict(hole, **offsets))
    edge_outcomes = dict(zip(
        edge_jobs.keys(),
        repair_pool.run_parallel([
            (lambda k=kind, idx=index, h=hole: _resolve_logged(
                candidate_path, idx, h, domain, master_obj, candidate_obj, work_dir))
            for kind, (index, hole) in edge_jobs.items()]))) if edge_jobs else {}
    decisions, summary = {}, []
    for kind, audio_s, level in (("head", head_s, first), ("tail", tail_s, last)):
        video_s, video_concluded, decline_reason = None, False, None
        if kind in by_kind:
            outcome = edge_outcomes[kind]
            budget = budget_cause(outcome, domain)
            if budget is not None:
                log_partial_plan(candidate_path, budget,
                                 [(edge, "placed", value) for edge, value in decisions.items()]
                                 + [(kind, "stopped", audio_s)])
                return None, None, (budget, f"the video on the {kind} edge stopped on a time "
                                            f"bound ({outcome.get('evidence')}) -- the partial "
                                            f"plan is logged; declined, retried at the next run")
            if outcome["status"] in _orch().EDGE_TERMINATIONS or outcome["status"] == _orch().HOLE_NO_CUT_CONFIRMED:
                try:
                    moved = _edge_blind_point(kind, outcome, domain, master_obj, candidate_obj,
                                              walk)
                except Exception as error:                               # noqa: BLE001
                    tools.dev_log(f"repair: edge_blind_span_unread ({type(error).__name__}: "
                                  f"{error})\n")
                    moved = None
                if moved is not None:
                    outcome = dict(outcome, **({"master_end_frame": moved} if kind == "head"
                                               else {"master_start_frame": moved}))
                video_s = _frame_s(outcome["master_end_frame"] if kind == "head"
                                   else outcome["master_start_frame"], domain)
                video_concluded = True
            elif (by_kind[kind][1].get("master_span_seconds") is not None
                  and by_kind[kind][1]["master_span_seconds"] <= frame):
                # No master-exclusive runtime exists before (head) or after (tail) this
                # bracket: the candidate simply runs past the master's own end here, or starts
                # before its own beginning. There is nothing of the master's own left to search
                # for, so this is a trim at the master's own edge, not a decline.
                video_s = 0.0 if kind == "head" else float(domain["master_timeline_ms"]) / 1000.0
                video_concluded = True
            else:
                # The one-anchor walk could not establish a boundary here: a named decline,
                # never a silent fall back to the audio's own edge.
                decline_reason = outcome.get("resolver_reason")
        if decline_reason is not None:
            log_partial_plan(candidate_path, "video_edge_undetermined",
                             [(edge, "placed", value) for edge, value in decisions.items()]
                             + [(kind, "stopped", audio_s)])
            return None, None, ("video_edge_undetermined",
                                f"the one-anchor walk on the {kind} edge could not place a "
                                f"boundary nor confirm there is none ({decline_reason})")
        if video_concluded:
            placed, decision = video_s, "video_edge_boundary"
        else:
            placed, decision = audio_s, edge_source[kind]
            if kind == "head":
                placed, decision = head_content_edge(placed, decision, frame, lead)
        decisions[kind] = placed
        summary.append(f"{kind}_placed_s={placed} {kind}_decision={decision}")
        tools.dev_log(f"repair: audio_edge kind={kind} level_offset_ms={level['off_ms']} "
                         f"audio_edge_s={audio_s} video_boundary_s="
                         f"{None if video_s is None else round(video_s, 4)} placed_s={placed} "
                         f"decision={decision} for {candidate_path}\n")
    if lead is not None:
        summary.append(f"head_candidate_first_sound_s={lead['content_edge_s']} "
                       f"head_master_first_sound_s={lead['master_first_sound_s']} "
                       f"head_uncovered_master_ms={round(lead['uncovered_s'] * 1000.0, 1)} "
                       f"head_uncovered_master_db={lead['master_db']}")
    if tail is not None and tail.get("run_on_s") is not None:
        summary.append(f"tail_candidate_run_on_ms={round(tail['run_on_s'] * 1000.0, 1)}")
    tools.log_always(f"repair: audio_edges {' '.join(summary)} for {candidate_path}\n")
    head_end = decisions["head"]
    # Past the tail edge nothing proves the candidate's content is common, so the master fills
    # up to the timeline even where its own audio has already ended.
    tail_start = decisions["tail"]
    if tail_start is not None and tail_start >= float(domain["master_timeline_ms"]) / 1000.0 - frame:
        tail_start = None
    return head_end, tail_start, None


def _cp_hole(point, reference, index):
    """Build an interior hole from a walk change point: bracket = its audio edges, offsets = its levels."""
    edges = point["edges"]
    quantum_ms = reference["alignment"]["quantum_ms"]
    low = min(edges["edge_A"], edges["edge_B"]) * 1000.0
    high = max(edges["edge_A"], edges["edge_B"]) * 1000.0
    a, b = point["a_ms"], point["b_ms"]
    return {"modality": _orch().MODALITY, "kind": "interior", "why_token": _orch().WHY_TOKEN["interior"],
            "origin": "audio_walk", "touches_head": False, "touches_tail": False,
            "master_ms": [low, high], "candidate_ms": [low + a, high + b],
            "master_span_seconds": (high - low) / 1000.0,
            "candidate_span_seconds": max(0.0, (high + b - low - a) / 1000.0),
            "offset_before_ms": a, "offset_after_ms": b, "offset_before_points": None,
            "offset_after_points": None, "step_ms": b - a,
            "step_points": _round_half_up(Fraction(str(b - a)) / Fraction(str(quantum_ms))),
            "quantum_ms": quantum_ms, "track_delay_delta_ms": reference["fold"]["delta_ms"],
            "offset_sources": ["audio_walk", "audio_walk"], "union_of": 1,
            "members": [{"couple": reference["couple"], "kind": "audio_walk_change_point",
                         "master_ms": [round(low, 2), round(high, 2)],
                         "step_ms": round(b - a, 3), "change_point": index}]}


def union_hole_edges(point, union):
    """Borrow edges for a change point from the cross-verified union when the walk could not read them.

    The walk's edge probes need an NCC of 0.8, which some couples (e.g. 6 ch vs 2 ch) never
    reach. Exactly one union interior hole must overlap the change point's search span with
    the same step within the inter-couple tolerance. Its master bounds, clipped to the span,
    become the edges; the walk's levels still set step and fill.

    Returns:
        An `edges` dict (`method` = `union_hole`), or None.
    """
    import audio_walk
    before, after = point["level_before"], point["level_after"]
    span_lo = before["t_last"] - audio_walk.WALK_HOP_S
    span_hi = after["t_first"] + audio_walk.WALK_WINDOW_S + audio_walk.WALK_HOP_S
    jump = point["b_ms"] - point["a_ms"]
    matches = [hole for hole in union or []
               if hole["kind"] == "interior" and hole.get("step_ms") is not None
               and abs(hole["step_ms"] - jump) <= (_orch().INTERCOUPLE_STEP_TOLERANCE_QUANTA
                                                   * _orch().INTERCOUPLE_STEP_TOLERANCE_SLACK
                                                   * hole["quantum_ms"])
               and hole["master_ms"][0] / 1000.0 < span_hi
               and hole["master_ms"][1] / 1000.0 > span_lo]
    if len(matches) != 1:
        return None
    hole = matches[0]
    edge_a = max(hole["master_ms"][0] / 1000.0, span_lo)
    edge_b = min(hole["master_ms"][1] / 1000.0, span_hi)
    extra = max(0.0, (point["a_ms"] - point["b_ms"]) / 1000.0)
    if extra > 0:
        kind, lo, hi = "deletion", edge_a, edge_b - extra
    else:
        kind, lo, hi = "addition", edge_a, edge_b
    if hi < lo:
        return None
    return {"method": "union_hole", "a_ms": point["a_ms"], "b_ms": point["b_ms"],
            "step_ms": round(jump, 3), "status": "ok", "edge_A": round(edge_a, 4),
            "edge_B": round(edge_b, 4), "extra_s": round(extra, 6), "master_only_audible": None,
            "kind": kind, "interval": [round(lo, 4), round(hi, 4)], "feasible": True,
            "union_hole_ms": [round(hole["master_ms"][0], 2), round(hole["master_ms"][1], 2)],
            "union_step_ms": round(hole["step_ms"], 3),
            "walk_edges": point.get("edges")}


# A video-pinned span exceeding the audio fill by less than this is the candidate's own join
# material (a crossfade dip, a silence) and is cut.
REPLACEMENT_MAX_EXCESS_S = 1.0


def replacement_hole(outcome, domain, edges, slack_s):
    """Turn an infeasible deletion into a replacement when the video pins both edges.

    Both cut frames must lie inside the audio edges (with `slack_s`) and the span must exceed
    the audio fill by less than REPLACEMENT_MAX_EXCESS_S.

    Returns:
        {start_s, end_s, fill_s, cut_ms}, or None.
    """
    if outcome.get("status") != _orch().HOLE_RESOLVED:
        return None
    start_s = _frame_s(outcome["master_start_frame"], domain)
    end_s = _frame_s(outcome["master_end_frame"], domain)
    fill_s = end_s - start_s
    excess = fill_s - edges["extra_s"]
    first, last = min(edges["edge_A"], edges["edge_B"]), max(edges["edge_A"], edges["edge_B"])
    if not (first - slack_s <= start_s and end_s <= last + slack_s):
        return None
    if not (0.0 <= excess < REPLACEMENT_MAX_EXCESS_S):
        return None
    return {"start_s": start_s, "end_s": end_s, "fill_s": fill_s,
            "cut_ms": round(excess * 1000.0, 3)}


def addition_replacement(point, union, walk):
    """Detect an addition change point that is really a replacement.

    It is one when audible master content between its edges matches neither offset
    (`master_only_audible`) and lies in a union interior hole, UNLESS the candidate's own
    material already covers that span at either offset (a plain splice, not a hole).

    Returns:
        {start_s, end_s, fill_s, cut_ms, hole}, or None (a point splice).
    """
    edges = point.get("edges") or {}
    only = edges.get("master_only_audible")
    if edges.get("kind") != "addition" or not only:
        return None
    start_s, end_s = edges["edge_A"], edges["edge_B"]
    if end_s <= start_s:
        return None
    low_ms, high_ms = only[0] * 1000.0, only[1] * 1000.0
    hits = [index for index, hole in enumerate(union or [])
            if hole["kind"] == "interior"
            and hole["master_ms"][0] < high_ms and hole["master_ms"][1] > low_ms]
    if not hits:
        return None
    import audio_walk
    if audio_walk.master_only_covered(walk["master"], walk["candidate"], only,
                                      point["a_ms"], point["b_ms"]):
        return None
    fill_s = end_s - start_s
    return {"start_s": start_s, "end_s": end_s, "fill_s": fill_s,
            "cut_ms": round(fill_s * 1000.0 + point["b_ms"] - point["a_ms"], 3),
            "hole": hits[0]}


def video_pin(video_s, interval, edges, extra_s, frame_s):
    """Confirm the video's own cut frame as a witness check against the audio's bounds.

    Accepted within one frame of the walk's interval, or inside the step's own edges leaving
    room for the whole fill (the interval can read narrower than the step). The audio only
    confirms or refuses here; the delivered instant is always the video's own frame, never
    nudged onto the audio interval's edge.

    Returns:
        (at_s, decision), or (None, None) when the video is unconfirmed or blind.
    """
    if video_s is None:
        return None, None
    lo, hi = interval
    if lo - frame_s <= video_s <= hi + frame_s:
        return video_s, "video_frame_inside_audio_interval"
    first, last = min(edges), max(edges) - max(0.0, extra_s)
    if first - frame_s <= video_s <= last + frame_s:
        return video_s, "video_frame_inside_audio_bounds"
    return None, None


def video_cut_instant(outcome, domain, extra_s, interval, quantum_ms):
    """Return the instant the video offers for a change point.

    The video's own width is first read as the anchors' own shift gap -- the same convention
    as `_pinned_frames`/`_blind_span_outcome` (delta = after_shift - before_shift; delta >= 0
    is the candidate's own excess, zero master fill; delta < 0 fills -delta frames from the
    master). This is checked in every case a walk reaches here, not only when neither walk
    advanced: an advancing walk locates where a cut sits, but the raw master span it covers can
    still include master-only frames that match neither shift (a fade, an eyecatch transition,
    a dark scene), which is not the cut's width. When that gap agrees with the audio fill
    within one frame, the video's first cut is offered directly. Otherwise the raw master span
    (`master_end_frame - master_start_frame`) is tried next, at the wider tolerance a resolved
    span is read at elsewhere in this pipeline -- this is the genuine case of two fills folded
    into one span, which the anchors' own gap does not describe. A deletion whose video span
    reads two fills this way, with an audio interval wider than a frame, offers its last cut
    minus the fill, if inside the interval.

    Returns:
        (video_s or None, width_note or None)
    """
    if outcome.get("status") not in (_orch().HOLE_RESOLVED, _orch().HOLE_PINNED_TO_AMBIGUOUS_ZONE_END):
        return None, None
    frame_ms = float(domain["frame_ms"])
    start_s = _frame_s(outcome["master_start_frame"], domain)
    before, after = outcome.get("before_shift_frames"), outcome.get("after_shift_frames")
    if before is not None and after is not None:
        gap_fill_ms = max(0, before - after) * frame_ms
        if abs(gap_fill_ms - extra_s * 1000.0) <= frame_ms:
            return start_s, None
    tolerance_ms = quantum_ms + 2 * frame_ms
    video_fill_ms = (outcome["master_end_frame"] - outcome["master_start_frame"]) * frame_ms
    if abs(video_fill_ms - extra_s * 1000.0) <= tolerance_ms:
        return start_s, None
    note = f"video fill {round(video_fill_ms, 3)} ms vs audio {round(extra_s * 1000.0, 3)} ms"
    two_fills = abs(video_fill_ms - 2.0 * extra_s * 1000.0) <= tolerance_ms
    wider_than_frame = (max(interval) - min(interval)) * 1000.0 > frame_ms
    if extra_s > 0 and two_fills and wider_than_frame:
        instant = _frame_s(outcome["master_end_frame"], domain) - extra_s
        if min(interval) <= instant <= max(interval):
            return instant, (f"{note}, two fills: its last cut minus the audio fill inside the "
                             f"audio interval pins")
    return None, note


def audio_transitions(walk, reference, domain, master_obj, candidate_obj, work_dir,
                      candidate_path, language=None, union=None):
    """Place every walk change point on the master timeline.

    The video decides: its anchor and cross-sweep search (`_resolve_logged` ->
    `_resolve_interior`) gives the exact cut frames, and the audio walk is the witness that
    confirms them, within one frame at the pair's own rational rate (`video_pin`'s tolerance,
    `domain["frame_ms"]`) -- the natural unit a frame-exact cut can miss by, never `int(fps)`
    nor a fixed millisecond figure tuned on one grid. The audio still fixes the step, the fill
    width (max(0, a - b)) and the interval a video cut must fall in to be confirmed.

    Three measured outcomes, never a silent fourth: a video cut confirmed within tolerance
    wins; a video cut the audio witness does not confirm (wrong width or wrong location) is a
    disagreement, logged with both positions, and the video is followed; a video
    that could neither place a cut nor rule one out (no compatible anchor, a static/black span,
    a geometry mismatch) declines the whole plan by name (`video_cut_undetermined` or, under
    the alignment's own resolution floor, `sub_quantum_step_video_ambiguous`) instead of
    delivering the audio's own instant unverified. Only when the video itself confirms there is
    no cut (`no_cut_confirmed`) does a sub-quantum step resolve as a slip at the quietest
    instant. Unreadable edges are borrowed from the union (`union_hole_edges`). Additions with
    master-only audio in a union hole become replacements.

    Returns:
        (transitions, None) or (None, (cause, reason)).
    """
    import audio_walk
    points = [p for p in walk["points"] if p["kind"] == "change_point"]
    holes, frame = [], float(domain["frame_ms"]) / 1000.0
    for index, point in enumerate(points):
        edges = point.get("edges") or {}
        if edges.get("status") != "ok":
            fallback = union_hole_edges(point, union)
            if fallback is not None:
                tools.log_always(
                    f"repair: edges_from_union_hole change_point={index} a_ms={point['a_ms']} "
                    f"b_ms={point['b_ms']} walk_step_ms={fallback['step_ms']} "
                    f"union_step_ms={fallback['union_step_ms']} "
                    f"union_hole_ms={fallback['union_hole_ms']} "
                    f"edges_s=[{fallback['edge_A']}, {fallback['edge_B']}] "
                    f"interval_s={fallback['interval']} walk_edges={edges.get('status')} "
                    f"for {candidate_path}\n")
                point["edges"] = edges = fallback
        if edges.get("status") != "ok":
            return None, ("audio_step_unlocalised",
                          f"the walk measured a {point['jump_ms']} ms step between levels "
                          f"{point['a_ms']} and {point['b_ms']} ms (t {point['level_before']['t_last']}"
                          f" -> {point['level_after']['t_first']} s) but its edges could not be "
                          f"read at 20 ms or 100 ms, and no cross-verified union hole carries "
                          f"that step there")
        holes.append(_cp_hole(point, reference, index))
    # Nearby change points share one scene pass.
    cluster_holes(holes)
    # Every hole's exact-frame search is independent (holes under the merge window share only
    # a read-through decode cache, never a decision), so they run together on the shared pool
    # instead of one change point at a time.
    hole_outcomes = repair_pool.run_parallel([
        (lambda i=index, h=hole: _resolve_logged(candidate_path, f"change_point_{i}", h, domain,
                                                 master_obj, candidate_obj, work_dir, walk=walk))
        for index, hole in enumerate(holes)])
    transitions = []
    for index, (point, hole) in enumerate(zip(points, holes)):
        edges = point["edges"]
        lo, hi = edges["interval"]
        extra = edges["extra_s"]
        jump = point["b_ms"] - point["a_ms"]
        sub_quantum = abs(jump) < hole["quantum_ms"]
        outcome = hole_outcomes[index]
        budget = budget_cause(outcome, domain)
        if budget is not None:
            log_partial_plan(candidate_path, budget,
                             [(f"change_point_{t['change_point']}", t["decision"], t["at_s"],
                               t["fill_s"]) for t in transitions]
                             + [(f"change_point_{index}", "stopped", lo, hi)])
            return None, (budget, f"the video on change point {index} ({lo}-{hi} s) stopped on "
                                  f"a time bound ({outcome.get('evidence')}) -- the partial plan "
                                  f"is logged; declined, retried at the next run")
        status = outcome["status"]
        replaced = addition_replacement(point, union, walk)
        if replaced is not None:
            transitions.append({"at_s": replaced["start_s"], "fill_s": replaced["fill_s"],
                                "a_ms": point["a_ms"], "b_ms": point["b_ms"],
                                "decision": "replacement_hole", "interval": [lo, hi],
                                "edges": [edges["edge_A"], edges["edge_B"]],
                                "video_status": status, "video_s": None,
                                "change_point": index, "cut_ms": replaced["cut_ms"]})
            tools.log_always(
                f"repair: replacement_hole change_point={index} kind=addition "
                f"a_ms={point['a_ms']} b_ms={point['b_ms']} step_ms={round(jump, 3)} "
                f"audio_edges_s=[{edges['edge_A']}, {edges['edge_B']}] "
                f"master_only_audible_s={edges['master_only_audible']} "
                f"union_hole={replaced['hole']} "
                f"fill_ms={round(replaced['fill_s'] * 1000.0, 3)} cut_ms={replaced['cut_ms']} "
                f"video_status={status} for {candidate_path}\n")
            continue
        # A resolved video answer (rule 2: its unmatched span is the fill) is never vetoed by
        # an audio interval that cannot hold the fill; only a video that placed nothing is.
        if not edges["feasible"] and status not in (_orch().HOLE_NO_CUT_CONFIRMED, _orch().HOLE_RESOLVED,
                                                     _orch().HOLE_PINNED_TO_AMBIGUOUS_ZONE_END):
            only = edges.get("master_only_audible")
            return None, ("hole_width_contradicts_audio_step",
                          f"the {round(edges['extra_s'] * 1000, 3)} ms of master content the "
                          f"candidate lacks at {edges['edge_A']}-{edges['edge_B']} s does not "
                          f"fit between the audio edges around its audible master-only sound "
                          f"({'span ' + str(only) if only else 'no audible master-only span'}"
                          f"), and the video does not pin a replacement (status {status}, "
                          f"{outcome.get('resolver_reason') or 'no frames placed'})")
        video_s, width_note = video_cut_instant(outcome, domain, extra, (lo, hi),
                                                hole["quantum_ms"])
        video_decided = status in (_orch().HOLE_RESOLVED, _orch().HOLE_PINNED_TO_AMBIGUOUS_ZONE_END)
        if status == _orch().HOLE_DECLINED:
            # No compatible anchor, a static/black span, a geometry mismatch...: the video
            # could neither place a cut here nor rule one out. A declined video reading is
            # never delivered as an unverified audio-only cut -- the whole plan declines,
            # named and explicit.
            return None, ("sub_quantum_step_video_ambiguous" if sub_quantum
                          else "video_cut_undetermined",
                          f"the walk measured a {round(jump, 3)} ms step in [{lo}, {hi}] s and "
                          f"the video could neither place a cut nor confirm there is none "
                          f"({outcome.get('resolver_reason')})")
        if video_decided and outcome["before_shift_frames"] == outcome["after_shift_frames"]:
            # The picture holds one offset across the hole: the audio step is the audio's own
            # (a track off its picture), not an edit. Nothing is cut, added or removed.
            at, fill, decision = _frame_s(outcome["master_start_frame"], domain), 0.0, \
                "audio_only_step_picture_continuous"
            tools.log_always(
                f"repair: audio_only_step_picture_continuous change_point={index} "
                f"a_ms={point['a_ms']} b_ms={point['b_ms']} step_ms={round(jump, 3)} "
                f"picture_shift_frames={outcome['before_shift_frames']} "
                f"audio_interval_s=[{lo}, {hi}] decision=finding_only for {candidate_path}\n")
        elif video_decided:
            # The video is the source of truth: once the anchors and the pHash cross-sweep
            # resolve a cut, its own position and width (the anchors' shift gap, the same
            # convention as `_pinned_frames`/`_blind_span_outcome`) are delivered regardless
            # of what the audio witness reads -- the audio alignment exists to save a full
            # scene scan, not to veto a resolved video answer. A disagreement is only logged,
            # with both readings, never refused.
            at = _frame_s(outcome["master_start_frame"], domain)
            before, after = outcome["before_shift_frames"], outcome["after_shift_frames"]
            # Rule 2: every unmatched master frame is filled, not only the shift gap.
            fill = (max(0, before - after,
                        outcome["master_end_frame"] - outcome["master_start_frame"]) * frame
                    if status == _orch().HOLE_RESOLVED else max(0, before - after) * frame)
            pinned, pin_decision = video_pin(video_s, (lo, hi), (edges["edge_A"], edges["edge_B"]),
                                             extra, frame)
            if pinned is not None and not width_note:
                decision = "video_decides"
            else:
                tools.log_always(
                    f"repair: video_audio_disagree_followed_video change_point={index} "
                    f"video_s={round(at, 4)} video_fill_ms={round(fill * 1000.0, 3)} "
                    f"audio_interval_s=[{lo}, {hi}] audio_fill_ms={round(extra * 1000.0, 3)} "
                    f"{width_note or 'video cut outside the audio interval and edges'} "
                    f"for {candidate_path}\n")
                decision = "video_audio_disagree_followed_video"
        elif status == _orch().HOLE_NO_CUT_CONFIRMED:
            # The video itself confirms there is no cut here (a pHash-measured still span):
            # nothing is added or removed, and the placed instant is the video's own, never an
            # audio-picked one.
            at, fill, decision = _frame_s(outcome["master_start_frame"], domain), 0.0, \
                "video_no_cut_confirmed"
        else:
            # HOLE_STATUSES_WITH_FRAMES carries exactly four statuses; RESOLVED/PINNED took
            # the video_decided branch above and DECLINED returned earlier, so nothing else
            # should reach here -- but a repair never crashes a merge on an unexpected
            # internal state, so this is a named decline, not a raise.
            tools.log_always(
                f"repair: hole_status_unhandled change_point={index} status={status!r} "
                f"for {candidate_path}\n")
            return None, ("hole_status_unhandled",
                          f"an internal hole status {status!r} reached audio_transitions at "
                          f"change point {index}, outside the four statuses it handles")
        shifts_known = video_decided or status == _orch().HOLE_NO_CUT_CONFIRMED
        before = outcome.get("before_shift_frames") if shifts_known else None
        after = outcome.get("after_shift_frames") if shifts_known else None
        removal_s = None
        if (outcome.get("removal_frame") is not None and status == _orch().HOLE_RESOLVED
                and before != after):
            removal_s = _frame_s(outcome["removal_frame"], domain)
        for piece_at, piece_fill, a_ms, b_ms, piece_decision, left, right in _transition_pieces(
                at, fill, point["a_ms"], point["b_ms"], decision, before, after, removal_s):
            transitions.append({"at_s": piece_at, "fill_s": piece_fill, "a_ms": a_ms,
                                "b_ms": b_ms, "decision": piece_decision, "interval": [lo, hi],
                                "edges": [edges["edge_A"], edges["edge_B"]],
                                "video_status": status, "video_s": video_s,
                                "change_point": index, "before_shift_frames": left,
                                "after_shift_frames": right})
        tools.dev_log(
            f"repair: audio_transition "
            f"change_point={index} a_ms={point['a_ms']} b_ms={point['b_ms']} "
            f"step_ms={round(jump, 3)} at_s={at} fill_ms={round(fill * 1000.0, 3)} "
            f"interval_s=[{lo}, {hi}] audio_edges_s=[{edges['edge_A']}, {edges['edge_B']}] "
            f"video_status={status} video_s={None if video_s is None else round(video_s, 4)} "
            f"decision={decision}{' ' + width_note.replace(' ', '_') if width_note else ''} "
            f"for {candidate_path}\n")
    log_transitions_summary(transitions, candidate_path)
    return transitions, None


def _transition_pieces(at, fill, a_ms, b_ms, decision, before, after, removal_s=None):
    """Return the transitions (at, fill, a_ms, b_ms, decision, left shift, right shift) of one cut.

    A candidate excess placed apart from the fill (`removal_s`, rule 1) gives two pieces: one
    offset change at the removal, one master fill with none.
    """
    if removal_s is None:
        return [(at, fill, a_ms, b_ms, decision, before, after)]
    if removal_s > at:
        return [(at, fill, a_ms, a_ms, decision, before, before),
                (removal_s, 0.0, a_ms, b_ms, "blind_removal", before, after)]
    return [(removal_s, 0.0, a_ms, b_ms, "blind_removal", before, after),
            (at, fill, b_ms, b_ms, decision, after, after)]


# A span whose bounding anchors read picture shifts this many frames apart has another frame
# count than the master's; one frame apart is two releases cutting the same shot differently.
PICTURE_STEP_MIN_FRAMES = 2
# Frames read per bisection probe, and the width the probes narrow a picture step down to.
PICTURE_PROBE_FRAMES = 24
PICTURE_STEP_WINDOW_FRAMES = 240
# Probe moves each way around a static middle: 16 probes, 16 s at 24 fps.
PICTURE_PROBE_MOVES = 16
# Widest window handed to the frame-exact resolver, whose anchor search reads +/- 10 s.
PICTURE_STEP_MAX_FRAMES = 720


def _picture_side(domain, master_obj, candidate_obj, frame, before, after):
    """Return which shift the picture holds over PICTURE_PROBE_FRAMES from master `frame`.

    "before" or "after" when two thirds of the frames read the same picture under that shift
    and under a third under the other, or else when the share of departing pixels under the
    other shift is at least twice its own and PIXEL_DIFF_FRACTION higher (moving content
    over a static background); None otherwise (a static or black span, a misread).
    """
    readings = _pixel_readings(domain, master_obj, candidate_obj, frame,
                               PICTURE_PROBE_FRAMES, sorted({before, after}))
    if readings is None or not len(readings[before]):
        return None
    count = len(readings[before])
    same_before = sum(_same_flags(readings[before], range(count)))
    same_after = sum(_same_flags(readings[after], range(count)))
    if 3 * same_before >= 2 * count and 3 * same_after < count:
        return "before"
    if 3 * same_after >= 2 * count and 3 * same_before < count:
        return "after"
    departing = {}
    for shift in (before, after):
        values = [reading[1] for reading in readings[shift] if reading is not None]
        departing[shift] = statistics.mean(values) if values else None
    if departing[before] is None or departing[after] is None:
        return None
    low, high = sorted((departing[before], departing[after]))
    if high >= 2 * low and high - low >= PIXEL_DIFF_FRACTION:
        return "before" if departing[before] == low else "after"
    return None


def _locate_picture_step(domain, master_obj, candidate_obj, low, high, before, after):
    """Narrow master frames [low, high) by bisection to a window holding the step, or None.

    The picture holds `before` at `low` and `after` at `high`. A probe that reads neither
    shift is moved by one probe width, up to PICTURE_PROBE_MOVES times each way. The search
    stops when the window is PICTURE_STEP_WINDOW_FRAMES wide, or when the probes no longer
    narrow it (a static or black stretch holds the step); a window still wider than
    PICTURE_STEP_MAX_FRAMES is not a location.
    """
    moves = [0] + [sign * step for step in range(1, PICTURE_PROBE_MOVES + 1) for sign in (1, -1)]
    sides = {}
    while high - low > PICTURE_STEP_WINDOW_FRAMES:
        middle = (low + high) // 2
        side = None
        for move in moves:
            probe = middle + move * PICTURE_PROBE_FRAMES
            if low < probe < high - PICTURE_PROBE_FRAMES:
                if probe not in sides:
                    sides[probe] = _picture_side(domain, master_obj, candidate_obj, probe,
                                                 before, after)
                side = sides[probe]
                if side is not None:
                    break
        if side is None:
            break
        width = high - low
        if side == "before":
            low = probe
        else:
            high = probe + PICTURE_PROBE_FRAMES
        if width - (high - low) < PICTURE_PROBE_FRAMES:
            break
    if high - low > PICTURE_STEP_MAX_FRAMES:
        return None
    return low, high


def picture_steps_in_continuous_zones(transitions, walk, reference, domain, master_obj,
                                      candidate_obj, work_dir, candidate_path):
    """Apply the frame-count rule to every span the audio sees as continuous.

    Between two cuts the audio holds one offset. The anchors bounding that span give its frame
    count on each file: the picture shift at its start and at its end.
    - Same count (shifts within one frame: two releases can cut a shot one frame apart) and
      the audio aligned: nothing in the span changes; a picture difference there is a finding.
    - Another count (frames more or fewer in the candidate): that part is taken from the
      master. The step is located by a pixel bisection inside the span and resolved
      frame-exact like any hole; when the frames between the probes read either shift (a fade,
      a black or static run), the whole window between the probes is taken from the master.
      The audio offset of the span is kept on both sides.
    A step the probes cannot locate stays a finding.

    Returns:
        the transitions, with the master takes in master order.
    """
    frame_ms = float(domain["frame_ms"])
    quantum_ms = reference["alignment"]["quantum_ms"]
    added = []
    for number, (left, right) in enumerate(zip(transitions, transitions[1:])):
        before, after = left.get("after_shift_frames"), right.get("before_shift_frames")
        if before is None or after is None or abs(after - before) < PICTURE_STEP_MIN_FRAMES:
            continue
        low = _master_frame_of_ms((left["at_s"] + left["fill_s"]) * 1000.0, domain)
        high = _master_frame_of_ms(right["at_s"] * 1000.0, domain)
        window = _locate_picture_step(domain, master_obj, candidate_obj, low, high, before,
                                      after)
        finding = (f"zone_frames=[{low}, {high}] shifts=[{before}, {after}] "
                   f"for {candidate_path}\n")
        if window is None:
            tools.log_always(f"repair: frame_count_differs_unlocated {finding}")
            continue
        level = left["b_ms"]
        step_ms = (after - before) * frame_ms
        low_ms, high_ms = window[0] * frame_ms, window[1] * frame_ms
        hole = {"modality": _orch().MODALITY, "kind": "interior",
                "why_token": _orch().WHY_TOKEN["interior"], "origin": "frame_count",
                "touches_head": False, "touches_tail": False,
                "master_ms": [low_ms, high_ms], "candidate_ms": [low_ms + level, high_ms + level],
                "master_span_seconds": (high_ms - low_ms) / 1000.0,
                "candidate_span_seconds": (high_ms - low_ms) / 1000.0,
                "offset_before_ms": before * frame_ms, "offset_after_ms": after * frame_ms,
                "offset_before_points": None, "offset_after_points": None,
                "step_ms": step_ms,
                "step_points": _round_half_up(Fraction(str(step_ms)) / Fraction(str(quantum_ms))),
                "quantum_ms": quantum_ms, "track_delay_delta_ms": reference["fold"]["delta_ms"],
                "offset_sources": ["picture", "picture"], "union_of": 1,
                "members": [{"couple": reference["couple"], "kind": "frame_count",
                             "master_ms": [round(low_ms, 2), round(high_ms, 2)],
                             "step_ms": round(step_ms, 3), "change_point": None}]}
        outcome = _resolve_logged(candidate_path, f"frame_count_{number}", hole, domain,
                                  master_obj, candidate_obj, work_dir, walk=walk)
        status = outcome["status"]
        exact = (status == _orch().HOLE_RESOLVED
                 and outcome["after_shift_frames"] - outcome["before_shift_frames"]
                 == after - before)
        if exact:
            start, end = outcome["master_start_frame"], outcome["master_end_frame"]
        else:
            # The frames between the probes read either shift: the window is taken whole.
            start, end = window
        removal_s = None
        if exact and outcome.get("removal_frame") is not None:
            removal_s = _frame_s(outcome["removal_frame"], domain)
        at = _frame_s(start, domain)
        fill = max(0, before - after, end - start) * frame_ms / 1000.0
        for piece_at, piece_fill, a_ms, b_ms, decision, left_shift, right_shift in (
                _transition_pieces(at, fill, level, level, "frame_count_differs_from_master",
                                   before, after, removal_s)):
            added.append({"at_s": piece_at, "fill_s": piece_fill, "a_ms": a_ms, "b_ms": b_ms,
                          "decision": decision, "interval": None, "edges": None,
                          "video_status": status, "video_s": at,
                          "change_point": left["change_point"],
                          "before_shift_frames": left_shift, "after_shift_frames": right_shift})
        tools.log_always(
            f"repair: frame_count_differs_from_master at_s={round(at, 4)} "
            f"fill_ms={round(fill * 1000.0, 3)} master_frames=[{start}, {end}] "
            f"read={'exact' if exact else 'probe_window'} resolver_status={status} "
            f"audio_offset_ms={level} {finding}")
    if not added:
        return transitions
    transitions = sorted(transitions + added, key=lambda t: t["at_s"])
    log_transitions_summary(transitions, candidate_path)
    return transitions


def log_transitions_summary(transitions, candidate_path):
    """Log one unconditional line with every transition as `change_point:decision:at_s:fill_ms:step_ms`."""
    tools.log_always(
        f"repair: audio_transitions n={len(transitions)} transitions="
        + ",".join(f"{t['change_point']}:{t['decision']}:{t['at_s']}:"
                   f"{round(t['fill_s'] * 1000.0, 3)}:{round(t['b_ms'] - t['a_ms'], 3)}"
                   for t in transitions)
        + f" for {candidate_path}\n")


def log_holes_against_walk(holes, walk, transitions, candidate_path):
    """Cross-log the union's holes against the walk's change points and the plan's cuts.

    An interior hole with no change point in reach sits where the audio is continuous: the
    frame-count rule decides its span (`picture_steps_in_continuous_zones`). Same frame count
    -> nothing changes (`same_frame_count_no_change`); another count -> the span was taken
    from the master (`frame_count_differs_from_master`). A change point no hole reaches is
    logged as unseen by the aligner.
    """
    import audio_walk
    regions = [(p["level_before"]["t_last"] * 1000.0,
                (p["level_after"]["t_first"] + audio_walk.WALK_WINDOW_S) * 1000.0, p)
               for p in walk["points"] if p["kind"] == "change_point"]
    reach = _orch().HOLE_MERGE_WINDOW_SECONDS * 1000.0
    takes = [t["at_s"] * 1000.0 for t in transitions
             if t["decision"] == "frame_count_differs_from_master"]
    cuts = [t["at_s"] * 1000.0 for t in transitions
            if t["decision"] != "frame_count_differs_from_master"]
    seen = set()
    for index, hole in enumerate(holes):
        if hole["kind"] != "interior":
            continue
        hits = [id(p) for low, high, p in regions
                if low < hole["master_ms"][1] + reach and high > hole["master_ms"][0] - reach]
        seen.update(hits)
        if hits:
            continue
        left = max((c for c in cuts if c <= hole["master_ms"][0]), default=None)
        right = min((c for c in cuts if c >= hole["master_ms"][1]), default=None)
        taken = any((left is None or left <= at) and (right is None or at <= right)
                    for at in takes)
        tools.log_always(
            f"repair: picture_only hole={index} master_ms=[{round(hole['master_ms'][0], 2)}, "
            f"{round(hole['master_ms'][1], 2)}] b2_step_ms="
            f"{None if hole['step_ms'] is None else round(hole['step_ms'], 3)} "
            f"rule={'frame_count_differs_from_master' if taken else 'same_frame_count_no_change'} "
            f"for {candidate_path}\n")
    for low, high, point in regions:
        if id(point) not in seen:
            _orch().step_result("change_point_unseen_by_b2", candidate=candidate_path,
                        a_ms=point["a_ms"], b_ms=point["b_ms"], jump_ms=point["jump_ms"],
                        region_ms=[round(low, 1), round(high, 1)])


# Pixel reading of a frame pair (160x90 grey): the pair differs when this fraction of its
# pixels departs by more than PIXEL_DIFF_LEVELS from the pair's mean difference (other
# content, a retouch, a layout change), or when its mean difference leaves the two files' own
# steady difference by more than PIXEL_FADE_LEVELS (a fade or a dim frame). A uniform regrade
# moves the steady difference itself and an encode difference stays near 1 level: neither
# differs. Measured: a candidate fade-out reads 2.4 levels off steady on its first faded
# frame; the same picture across two encodes keeps up to 1.4 % of its pixels past 20 levels.
PIXEL_DIFF_LEVELS = 20.0
PIXEL_DIFF_FRACTION = 0.03
PIXEL_FADE_LEVELS = 2.0
# On a dark picture a fade moves the level by little: 8 % of the master's own level counts
# (measured: a candidate fade-in on a level-5 shot, 0.6 levels off steady on its last faded
# frame, 0.4 on the first steady one), never under 0.75 level (measured: faint candidate text
# over a black master frame, 0.45 off steady, is no difference).
PIXEL_FADE_SHARE = 0.08
PIXEL_FADE_FLOOR = 0.75
# Common frames whose median level difference is the steady one around a frame.
PIXEL_STEADY_FRAMES = 12
# Consecutive frames that settle a front (see `_front_forward`).
PIXEL_RUN_FRAMES = 3


def _grey_frames(comparer, path, first, count, rate):
    """`count` grey 160x90 frames of `path` from decoded frame `first` on (float array)."""
    import numpy as np
    import frame_hash
    # Same window convention as the anchor search's own extraction, so frame labels agree.
    start_s = float(Fraction(max(0, first)) / rate)
    blob = comparer._ffmpeg_raw_frames(path, start_s, float(Fraction(count) / rate))
    import frame_compare
    frames = frame_hash.frames_from_raw(blob, comparer.width, comparer.height, 3)
    # A file at another native rate is re-indexed onto the master grid, as the anchors are.
    frames, _reason = frame_compare._on_comparer_grid(comparer, path, list(frames))
    if not frames:
        return np.zeros((0, comparer.height, comparer.width))
    return np.asarray(frames, dtype=np.float64).mean(axis=-1)[:count]


def _pixel_reading(master_frame, candidate_frame):
    """(mean level difference, fraction of pixels departing from it by > PIXEL_DIFF_LEVELS,
    master std, master level, candidate std) of one grey frame pair."""
    import numpy as np
    diff = master_frame - candidate_frame
    level = float(diff.mean())
    return (level, float((np.abs(diff - level) > PIXEL_DIFF_LEVELS).mean()),
            float(master_frame.std()), float(master_frame.mean()), float(candidate_frame.std()))
