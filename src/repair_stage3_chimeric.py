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
import progress_watchdog
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
# barely separates two adjacent lags on pHash -- `frame_compare.ALIGN_MARGIN` can pass at a
# margin as thin as 0.058 -- while the raw picture does not share that blindness.
ANCHOR_MAD_REACH = 1
# On a fade, a neighbour lag must read lower by at least this many luma levels, averaged over
# the anchor window, to override the shift pHash picked.
ANCHOR_MAD_MARGIN_LEVELS = 2.0
# Elsewhere every shift is re-checked over the anchor window plus ANCHOR_MAD_BEYOND_FRAMES read
# away from its cut, and a neighbour wins when its mean is at most ANCHOR_MAD_NEIGHBOUR_RATIO of
# the shift's own. Measured on 160x90 grey frames (2997/125 master against a 24000/1001
# candidate, four 60-frame anchor windows): the right lag reads 0.25-0.41 levels (encode noise)
# and its two neighbours 1.4-10 times that. A 12-frame window on a near-static shot is not
# enough: raw MAD there ranks lags on noise.
ANCHOR_MAD_BEYOND_FRAMES = 48
ANCHOR_MAD_NEIGHBOUR_RATIO = 0.5
# An anchor whose MAD over its neighbouring shifts varies less than this share of its mean read
# no shift at all: a decisive zone majority wins over it.
ANCHOR_MAD_FLAT_SHARE = 0.01
# The raw-MAD fade rule applies where pHash is known to be blind: a fade (a luma ramp
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


def _anchor_mad_confirmed_shift(domain, master_obj, candidate_obj, first, count, shift,
                                beyond=0):
    import numpy as np
    import frame_compare
    shifts = [shift - ANCHOR_MAD_REACH, shift, shift + ANCHOR_MAD_REACH]
    rate = domain["master_rate"]
    scale = domain.get("time_scale")
    comparer = frame_compare.FrameComparer(
        master_obj.filePath, candidate_obj.filePath, 0.0, 1.0, rate.numerator,
        rate.denominator,
        time_scales=None if scale is None else {candidate_obj.filePath: scale})
    low = max(0, first + min(0, beyond))
    try:
        if first < 0 or low + min(shifts) < 0:
            return shift, None
        master = _grey_frames(comparer, master_obj.filePath, low, first + count + max(0, beyond)
                              - low, rate)
        window = range(first - low, min(len(master), first - low + count))
        if not len(window):
            return shift, None
        fade = _is_fade_window([float(master[index].mean()) for index in window])
        if not fade and (beyond == 0 or len(master) < count + abs(beyond)):
            return shift, None
        candidate = _grey_frames(comparer, candidate_obj.filePath, low + min(shifts),
                                 len(master) + max(shifts) - min(shifts), rate)
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"repair: anchor_mad_unread ({type(error).__name__}: {error})\n")
        return shift, None
    rows = window if fade else range(len(master))
    means = {}
    for probed_shift in shifts:
        values = [float(np.abs(master[index] - candidate[index + probed_shift - min(shifts)])
                        .mean())
                  for index in rows if index + probed_shift - min(shifts) < len(candidate)]
        if values:
            means[probed_shift] = statistics.mean(values)
    if shift not in means:
        return shift, means
    best = min(means, key=means.get)
    if best == shift:
        return shift, means
    if fade and means[shift] - means[best] >= ANCHOR_MAD_MARGIN_LEVELS:
        return best, means
    if not fade and means[best] <= ANCHOR_MAD_NEIGHBOUR_RATIO * means[shift]:
        return best, means
    return shift, means


def _anchor_shifts_mad_confirmed(result, domain, master_obj, candidate_obj):
    import scene_anchor
    anchor_a, anchor_b = result.get("anchor_a_frame"), result.get("anchor_b_frame")
    before, after = result.get("before_shift_frames"), result.get("after_shift_frames")
    updates, flat = {}, []
    for side, anchor, shift, key in (("A", anchor_a, before, "before_shift_frames"),
                                     ("B", anchor_b, after, "after_shift_frames")):
        if anchor is None or shift is None:
            continue
        if side == "A":
            first = max(0, anchor - scene_anchor.ANCHOR_WINDOW_FRAMES)
            window, beyond = (first, anchor - first), -ANCHOR_MAD_BEYOND_FRAMES
        else:
            window, beyond = (anchor, scene_anchor.ANCHOR_WINDOW_FRAMES), ANCHOR_MAD_BEYOND_FRAMES
        confirmed, means = _anchor_mad_confirmed_shift(domain, master_obj, candidate_obj,
                                                       window[0], window[1], shift, beyond)
        rounded = None if means is None else {k: round(v, 3) for k, v in means.items()}
        tools.dev_log(f"repair: anchor_shift_mad side={side} anchor={anchor} "
                      f"phash_shift={shift} mad_shift={confirmed} means={rounded}\n")
        if means and len(means) >= 3 and (max(means.values()) - min(means.values())
                                          < ANCHOR_MAD_FLAT_SHARE * statistics.mean(means.values())):
            flat.append("before" if side == "A" else "after")
        if confirmed != shift:
            tools.log_always(
                f"repair: anchor_shift_mad_corrected side={side} anchor={anchor} "
                f"phash_shift={shift} mad_shift={confirmed} means={rounded} "
                f"for {candidate_obj.filePath}\n")
            updates[key] = confirmed
    if flat:
        updates["anchor_flat_sides"] = flat
    if not updates:
        return result
    return dict(result, **updates)


def _two_anchor_call(hole, domain, master_obj, candidate_obj, low_ms, high_ms,
                     offset_before_ms, offset_after_ms, step_ms, quantum_ms, probe,
                     resolve_shift=True):
    """Run the interior resolver once as a logged step; an exception becomes a named decline."""
    import scene_anchor
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
            cluster_window=hole.get("cluster_window"), scan_cache=hole.get("scan_cache"))
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


# Pixel fronts are trusted only when each side shows a run of at least this many consecutive
# frames reading the same picture under its own shift (the anchor search's validation length);
# a regraded or re-filtered pair never does and keeps the pHash fronts.
PIXEL_TRUST_RUN_FRAMES = 12
# Frames read past each anchor: a fade can start before anchor A or end after anchor B, since
# pHash seats an anchor on a fade it cannot see.
PIXEL_FRONT_MARGIN_FRAMES = 96
# Widest span (frames) read at pixel level; wider ones keep the pHash fronts.
PIXEL_FRONT_MAX_FRAMES = 2400


def _pixel_readings(domain, master_obj, candidate_obj, first, count, shifts, common=None):
    """{shift: [`_pixel_reading` tuple or None] * n} for master [first, first + n) against the
    candidate at each shift; None marks a candidate frame that does not exist (before its
    start or after its end). `common` lists (master indices, shift) pairs already known to
    show the same picture: the candidate's tone curve is fitted on them (`candidate_tone_map`).
    The key "candidate_change" holds (first candidate frame read, per-frame mean luma change
    from the frame before); "master_shot" the master shot of each frame (`master_shots`).
    Returns None on any read failure."""
    import numpy as np
    rate = domain["master_rate"]
    import frame_compare
    scale = domain.get("time_scale")
    comparer = frame_compare.FrameComparer(
        master_obj.filePath, candidate_obj.filePath, 0.0, 1.0, rate.numerator,
        rate.denominator,
        time_scales=None if scale is None else {candidate_obj.filePath: scale})
    master = _grey_frames(comparer, master_obj.filePath, first, count, rate)
    if first < 0 or not len(master):
        return None
    low = max(0, first + min(shifts))
    candidate = _grey_frames(comparer, candidate_obj.filePath, low,
                             first + len(master) + max(shifts) - low, rate)

    def candidate_index(index, shift):
        other = first + index + shift - low
        return other if 0 <= other < len(candidate) else None
    pairs = [(index, candidate_index(index, shift)) for indices, shift in (common or [])
             for index in indices if 0 <= index < len(master)]
    tone = candidate_tone_map(master, candidate,
                              [(index, other) for index, other in pairs if other is not None])
    if tone is not None:
        candidate = np.interp(candidate, tone[0], tone[1])
    readings = {}
    for shift in shifts:
        row = []
        for index in range(len(master)):
            other = candidate_index(index, shift)
            row.append(None if other is None else _pixel_reading(master[index], candidate[other]))
        readings[shift] = row
    changes = (np.abs(np.diff(candidate, axis=0)).mean(axis=(1, 2)) if len(candidate) > 1
               else np.zeros(0))
    readings["candidate_change"] = (low, [0.0] + [float(value) for value in changes])
    master_changes = (np.abs(np.diff(master, axis=0)).mean(axis=(1, 2)) if len(master) > 1
                      else np.zeros(0))
    readings["master_shot"] = master_shots([0.0] + [float(value) for value in master_changes])
    return readings


# Quantiles of the tone curve fitted between two releases graded differently.
TONE_QUANTILES = 101


def candidate_tone_map(master, candidate, pairs):
    """(candidate levels, master levels) of a monotone map putting the candidate's tone curve
    on the master's, or None when the frames known common already read the same picture.

    Fitted by quantile matching over the (master index, candidate index) `pairs`, and kept only
    when it brings their median departing-pixel share under PIXEL_DIFF_FRACTION.
    """
    import numpy as np
    if not pairs:
        return None

    def median_departing(mapped):
        return statistics.median(_pixel_reading(master[index], mapped[other])[1]
                                 for index, other in pairs)
    identity = median_departing(candidate)
    if identity < PIXEL_DIFF_FRACTION:
        return None
    quantiles = np.linspace(0.0, 100.0, TONE_QUANTILES)
    source = np.percentile(np.concatenate([candidate[other].ravel() for _index, other in pairs]),
                           quantiles)
    target = np.percentile(np.concatenate([master[index].ravel() for index, _other in pairs]),
                           quantiles)
    source, kept = np.unique(source, return_index=True)
    target = np.maximum.accumulate(target[kept])
    if len(source) < 2:
        return None
    sample = {other: np.interp(candidate[other], source, target) for _index, other in pairs}
    mapped = median_departing(sample)
    tools.log_always(f"repair: pixel_tone_map frames={len(pairs)} departing_identity="
                     f"{round(identity, 4)} departing_mapped={round(mapped, 4)} "
                     f"kept={mapped < PIXEL_DIFF_FRACTION}\n")
    return (source, target) if mapped < PIXEL_DIFF_FRACTION else None


def candidate_change_at(readings, index, shift, first):
    """Mean luma change of the candidate frame read against master `first + index` at `shift`
    from the frame before it, or None when that frame or the one before was not read."""
    low, changes = readings["candidate_change"]
    other = first + index + shift - low
    return changes[other] if 0 < other < len(changes) else None


def level_bound(reading):
    """Largest level departure from the steady difference that still reads the same picture."""
    return max(PIXEL_FADE_FLOOR, min(PIXEL_FADE_LEVELS, PIXEL_FADE_SHARE * reading[3]))


def change_limit(changes):
    """Luma change above which a frame opens a new shot against `changes` around it:
    BLIND_PICTURE_CHANGE_RATIO x their median, never under PIXEL_FADE_FLOOR."""
    return max(BLIND_PICTURE_CHANGE_RATIO * statistics.median(changes) if changes else 0.0,
               PIXEL_FADE_FLOOR)


def opens_new_shot(changes, index):
    """Whether frame `index` of a per-frame mean luma change list opens a new shot against the
    PIXEL_STEADY_FRAMES frames before it (`change_limit`)."""
    if index <= 0 or index >= len(changes):
        return False
    return changes[index] > change_limit(changes[max(1, index - PIXEL_STEADY_FRAMES):index])


def master_shots(changes):
    """Shot number of each master frame from its per-frame mean luma change: a shot opens on a
    frame whose change passes `change_limit` against the PIXEL_STEADY_FRAMES frames on each
    side (a cut is a spike; a moving shot or a shot that moves more than the one before is
    not)."""
    shots, shot = [], 0
    for index in range(len(changes)):
        after = changes[index + 1:index + 1 + PIXEL_STEADY_FRAMES]
        if opens_new_shot(changes, index) and changes[index] > change_limit(after):
            shot += 1
        shots.append(shot)
    return shots


def shot_seed(row, frames):
    """First PIXEL_TRUST_RUN_FRAMES consecutive frames of one shot that read the same picture
    under a steady level (departing share under PIXEL_DIFF_FRACTION, centred correlation from
    PIXEL_FADE_CORR, every level within `level_bound` of the run's median), or []."""
    for start in range(len(frames) - PIXEL_TRUST_RUN_FRAMES + 1):
        run = frames[start:start + PIXEL_TRUST_RUN_FRAMES]
        if any(row[i] is None or row[i][1] >= PIXEL_DIFF_FRACTION
               or row[i][7] < PIXEL_FADE_CORR for i in run):
            continue
        median = statistics.median(row[i][0] for i in run)
        if all(abs(row[i][0] - median) <= level_bound(row[i]) for i in run):
            return run
    return []


def _same_flags(row, reference, shots=None):
    """Per-frame "same picture" flags of one reading row (`_pixel_readings`).

    The two files' own steady level difference (a regrade, a different transfer curve, which
    varies from shot to shot) is the median over the PIXEL_STEADY_FRAMES common frames of the
    same master shot (`shots`, one shot when None) nearest each frame. Common frames start as
    `reference` (known common) plus, in a shot holding none of them, its `shot_seed`; every
    frame read as the same picture joins them until nothing changes, so trust spreads shot by
    shot. A shot with neither borrows the nearest common frames of the others. A frame differs
    when its local fraction reaches PIXEL_DIFF_FRACTION or its level leaves the steady
    difference by more than `level_bound` (a fade or a dim frame). A dark frame (either side
    under `frame_compare.FLAT_STD`) is judged on contrast (`dark_contrast_same`); two
    noise-flat frames (black against black) are the same picture at near levels
    (decision: steady-level-per-master-shot).
    """
    import bisect
    if shots is None or len(shots) != len(row):
        shots = [0] * len(row)
    members = {}
    for index, shot in enumerate(shots):
        members.setdefault(shot, []).append(index)

    def is_common(index):
        return (row[index] is not None and row[index][1] < PIXEL_DIFF_FRACTION
                and abs(row[index][0]) <= PIXEL_DIFF_LEVELS)

    def steady_of(index, common, by_shot):
        pool = by_shot.get(shots[index]) or common
        if not pool:
            return 0.0
        at = bisect.bisect_left(pool, index)
        near = pool[max(0, at - PIXEL_STEADY_FRAMES):at + PIXEL_STEADY_FRAMES]
        near = sorted(near, key=lambda i: abs(i - index))[:PIXEL_STEADY_FRAMES]
        return statistics.median(row[i][0] for i in near)

    def judge(known):
        common = sorted(i for i in known if is_common(i))
        by_shot = {}
        for index in common:
            by_shot.setdefault(shots[index], []).append(index)
        flags = []
        for index, reading in enumerate(row):
            if reading is None or reading[1] >= PIXEL_DIFF_FRACTION:
                flags.append(False)
                continue
            contrast = dark_contrast_same(reading)
            if contrast is not None:
                flags.append(contrast)
                continue
            if (max(reading[5], reading[6]) < PIXEL_NOISE_STD
                    and abs(reading[0]) <= PIXEL_FADE_LEVELS):
                flags.append(True)
                continue
            flags.append(abs(reading[0] - steady_of(index, common, by_shot))
                         <= level_bound(reading))
        return flags

    known = {i for i in reference if 0 <= i < len(row)}
    for shot, frames in members.items():
        if not any(is_common(i) for i in frames if i in known):
            known.update(shot_seed(row, frames))
    while True:
        flags = judge(known)
        grown = known | {i for i, flag in enumerate(flags) if flag}
        if grown == known:
            return flags
        known = grown


def dark_contrast_same(reading):
    """Whether a dark frame pair shows the same picture, judged on its contrast; None when
    neither frame is dark or both are noise-flat (left to the level rules).

    A level offset measured on a brighter shot says nothing about black against the first frame
    of a fade-in, so a dark pair is the same picture only when the std ratio of the pixels that
    do not depart lies within 1 / PIXEL_DARK_STD_RATIO .. PIXEL_DARK_STD_RATIO.
    """
    import frame_compare
    low, high = sorted((reading[5], reading[6]))
    if low >= frame_compare.FLAT_STD or high < PIXEL_NOISE_STD:
        return None
    return high <= PIXEL_DARK_STD_RATIO * low


def _tolerant_flags(readings, shift, reference):
    """`_same_flags` at `shift`, a frame also counting as the same picture when it matches at
    shift - 1 or shift + 1: two releases can cut or fade the same shot one frame apart."""
    rows = [_same_flags(readings[shift + delta], reference, readings.get("master_shot"))
            for delta in (-1, 0, 1)
            if shift + delta in readings]
    return [any(flags) for flags in zip(*rows)]


def tolerance_only_at_new_shot(readings, shift, first, reference):
    """Per master frame, whether it differs at the exact shift where the candidate frame opens
    a new shot (`opens_new_shot`): the one-frame tolerance must not count it as the same."""
    exact = _same_flags(readings[shift], reference, readings.get("master_shot"))
    low, changes = readings["candidate_change"]
    return [not same and candidate_change_at(readings, index, shift, first) is not None
            and opens_new_shot(changes, first + index + shift - low)
            for index, same in enumerate(exact)]


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
    first = max(0, anchor_a - PIXEL_FRONT_MARGIN_FRAMES, -min(before, after))
    count = anchor_b + PIXEL_FRONT_MARGIN_FRAMES - first
    # The steady level difference (and a regrade's tone curve) is first read where the pHash
    # walks found common picture, kept clear of their fronts: pHash walks into a fade it
    # cannot see.
    guard = 2 * PIXEL_STEADY_FRAMES

    def common_sides(count):
        front = min(max(result["pre_collapse_start_master"] - first, 0), count)
        back = min(max(result["pre_collapse_end_master"] - first, 0), count)
        return front, back, range(0, max(1, front - guard)), range(min(count - 1, back + guard),
                                                                   count)
    _front, _back, common_before, common_after = common_sides(count)
    try:
        readings = _pixel_readings(domain, master_obj, candidate_obj, first, count,
                                   sorted({shift + delta for shift in (before, after)
                                           for delta in (-1, 0, 1)}),
                                   common=[(common_before, before), (common_after, after)])
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"repair: pixel_fronts_unread ({type(error).__name__}: {error})\n")
        return None
    if readings is None:
        return None
    count = len(readings[before])
    front, back, common_before, common_after = common_sides(count)
    same_before = _tolerant_flags(readings, before, common_before)
    same_after = _tolerant_flags(readings, after, common_after)
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
    moved = []
    if front > 0 and tolerance_only_at_new_shot(readings, before, first, common_before)[front - 1]:
        front -= 1
        moved.append(f"forward_to={first + front}")
    if back < count and tolerance_only_at_new_shot(readings, after, first, common_after)[back]:
        back += 1
        moved.append(f"backward_to={first + back}")
    if moved:
        tools.log_always(f"repair: pixel_front_exact_shift {' '.join(moved)} "
                         f"for {candidate_obj.filePath}\n")
    crossed = back < front
    gap = before - after
    phash_front, phash_back = result["pre_collapse_start_master"], result["pre_collapse_end_master"]
    if gap > 0 and phash_back - phash_front == gap and (crossed or back - front < gap):
        # A master insert lacks at least the shift gap: narrower pixel fronts come from the
        # one-frame tolerance, never from a blind span (decision: pixel-front-gap-floor).
        tools.log_always(
            f"repair: pixel_fronts_narrower_than_gap anchors=[{anchor_a}, {anchor_b}] "
            f"shifts=[{before}, {after}] phash_fronts=[{phash_front}, {phash_back}] "
            f"pixel_fronts=[{first + front}, {first + back}] crossed={crossed} kept=phash "
            f"for {candidate_obj.filePath}\n")
        return dict(result, sweep_crossed=False, pixel_fronts=False,
                    master_start_frame=phash_front, master_end_frame=phash_back,
                    candidate_start_frame=phash_front + before,
                    candidate_end_frame=phash_back + after)
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
                        resolve_shift=True):
    """Run `_two_anchor_call`, then re-read its fronts on pixels when they can be trusted."""
    result = _two_anchor_call(hole, domain, master_obj, candidate_obj, low_ms, high_ms,
                              offset_before_ms, offset_after_ms, step_ms, quantum_ms, probe,
                              resolve_shift=resolve_shift)
    if result["declined"]:
        return result
    refined = _pixel_refined(result, domain, master_obj, candidate_obj)
    return result if refined is None else refined


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
    if same_shift and result["master_end_frame"] == result["master_start_frame"]:
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
        "anchor_flat_sides": result.get("anchor_flat_sides"),
        "geometry": (result.get("geometry") or {}).get("verdict"),
        "evidence": result.get("evidence"),
    }
    if status == _orch().HOLE_PINNED_TO_AMBIGUOUS_ZONE_END:
        outcome["cause"] = "static_span_ambiguity"
        outcome["pin_frame"], outcome["ambiguous_frames"] = _pin_point(result)
    if refuted_proposal is not None:
        outcome["refuted_proposal"] = refuted_proposal
    return outcome


def _ambiguous_pin_outcome(hole, domain, result, candidate_path, walk=None, master_obj=None):
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
    outcome = _blind_span_outcome(hole, domain, blind, candidate_path, walk=walk,
                                  master_obj=master_obj)
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
FADE_LEVEL_DIFFER_DB = 6.0
FADE_GAP_SPAN_S = 20.0
FADE_GAP_MIN_FRAMES = 10


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


def _candidate_silent(walk, times, shift_s, picture_shift_s=None):
    """Whether the candidate is silent in each 20 ms window: the peak of the comparison track
    at `shift_s` and of every other track that can be delivered (`walk["silence_tracks"]`) at
    `picture_shift_s` (the offset the delivery places them with; `shift_s` when None) under
    the audibility floor the audio walk itself uses (`audio_walk.AUDIBLE_DB`), so a fading
    sting tail still counts as sound."""
    import numpy as np
    import audio_walk
    if picture_shift_s is None:
        picture_shift_s = shift_s
    tracks = walk.get("silence_tracks") or [walk["candidate"]]
    return np.logical_and.reduce([_window_levels(track, times, shift_s if index == 0
                                                 else picture_shift_s, peak=True)
                                  < audio_walk.AUDIBLE_DB for index, track in enumerate(tracks)])


def quiet_frame_in_interval(walk, domain, interval, offsets_ms):
    import numpy as np
    import audio_walk
    frame_s = float(domain["frame_ms"]) / 1000.0
    low = math.floor(min(interval) / frame_s)
    high = math.ceil((max(interval) + audio_walk.FINE_WIN_S / 2.0) / frame_s)
    flags = []
    for frame in range(low, high + 1):
        times = np.arange(frame * frame_s, (frame + 1) * frame_s - BLIND_SILENCE_WIN_S + 1e-9,
                          BLIND_SILENCE_HOP_S)
        flags.append(all(bool(_candidate_silent(walk, times, offset / 1000.0).all())
                         for offset in offsets_ms))
    found = _centre_of_longest(flags, low)
    return None if found is None else found[0]


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
    """(frame, [first, last], how) placing a blind gap from the audio: a removal in the candidate's
    silence, a fill where the candidate's audio before and after best matches the master; or None."""
    import numpy as np
    if walk is None or gap == 0 or high < low:
        return None
    frame_s = float(domain["frame_ms"]) / 1000.0
    width = abs(gap)
    hop = BLIND_SILENCE_HOP_S
    try:
        # A removal keeps one frame of silence on each side of the removed candidate span.
        margin = 1 if gap > 0 else 0
        t_lo = max(0.0, (low - width - margin) * frame_s)
        t_hi = (high + margin) * frame_s
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
                first = index_of((point - width - margin) * frame_s)
                last = last_index((point + margin) * frame_s - BLIND_SILENCE_WIN_S)
                flags.append(last >= first and bool(silent[first:last + 1].all()))
            found = _centre_of_longest(flags, low)
            return None if found is None else found + ("candidate_silence",)
        import audio_walk
        master_peak = _window_levels(walk["master"], times, 0.0, peak=True)
        if not (master_peak > audio_walk.AUDIBLE_DB).any():
            # The master is inaudible over the whole span: every placement holds each
            # candidate sound once and the misfit is noise. No audio evidence -- the caller
            # applies the picture rule (just before the right anchor).
            tools.log_always(
                f"repair: blind_span_master_inaudible frames=[{low}, {high}] "
                f"master_peak_max_db={round(float(master_peak.max()), 1)} "
                f"audible_db={audio_walk.AUDIBLE_DB}\n")
            return None
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
            return None
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


# A master frame-to-frame luma change above this many times the span's median change (and the
# noise floor) is a picture change: the first frame of the next shot after a blind run.
BLIND_PICTURE_CHANGE_RATIO = 3.0


def _right_anchor_frame(domain, master_obj, low, high, width):
    """The first master frame of the shot after a blind run, inside the allowed edges [low, high].

    The last master frame in [low, high] whose picture changes from the frame before it (mean
    luma change above BLIND_PICTURE_CHANGE_RATIO x the span's median change and
    REFUTE_NOISE_FLOOR_LEVELS) is where the blind run ends: a fill placed just before it is
    the owner rule's "just before the right anchor". None when no such change is read.
    """
    import numpy as np
    import frame_compare
    first = max(0, low - width - 1)
    rate = domain["master_rate"]
    comparer = frame_compare.FrameComparer(master_obj.filePath, master_obj.filePath, 0.0, 1.0,
                                           rate.numerator, rate.denominator)
    try:
        frames = _grey_frames(comparer, master_obj.filePath, first, high + 1 - first, rate)
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"repair: right_anchor_unread ({type(error).__name__}: {error})\n")
        return None
    if len(frames) < 2:
        return None
    changes = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    limit = max(BLIND_PICTURE_CHANGE_RATIO * float(np.median(changes)), REFUTE_NOISE_FLOOR_LEVELS)
    for index in range(len(changes) - 1, -1, -1):
        frame = first + index + 1
        if frame < low:
            break
        # The fill [frame - width, frame) must lie in the blind run itself: no picture
        # change between its frames (a moving master-only span is not a blind run).
        inside = changes[max(0, index - width + 1):index]
        if frame <= high and changes[index] > limit and not (inside > limit).any():
            return frame
    return None


def _blind_span_outcome(hole, domain, blind, candidate_path, walk=None, master_obj=None):
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
    anchor = None
    if gap < 0 and master_obj is not None:
        anchor = _right_anchor_frame(domain, master_obj, low, high, -gap)
    # A fill ends at the right anchor when that picture change is where the after shift starts
    # to read the same picture (a picture identity at the backward front); the audio places a
    # fill only when both sides are picture-blind (decision: picture-before-audio-fill).
    back = blind.get("pre_collapse_end_master")
    identity = (anchor is not None and back is not None
                and abs(anchor - back) <= PIXEL_RUN_FRAMES)
    found = None
    if not identity:
        found = _blind_span_audio_point(domain, gap, before, after, low, high, walk)
    if found is not None:
        point, frames, placement = found
        placement_evidence = f"placement_frames={frames}"
    elif anchor is not None:
        point, placement, placement_evidence = anchor, "right_anchor", "master_picture_change"
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
    outcome = _interior_outcome(hole, domain, placed, _orch().HOLE_RESOLVED)
    # The placement is not a picture reading: kept so the audio interval can bound it, and so
    # a corrected shift pair can re-place it (`audio_transitions`).
    outcome["blind_span"] = {"placement": placement, "allowed_frames": [low, high],
                             "gap_frames": gap, "point": point,
                             "run": frames if found is not None else None,
                             "blind": {key: blind.get(key) for key in (
                                 "anchor_a_frame", "anchor_b_frame",
                                 "pre_collapse_start_master", "pre_collapse_end_master")}}
    return outcome


# A master frame is unmatched when its lowest luma difference over the searched lags exceeds
# this many times the median of the span's lowest differences (and the noise floor below).
REFUTE_UNMATCHED_RATIO = 3.0
# Below this many luma levels a difference is noise, whatever the span's median.
REFUTE_NOISE_FLOOR_LEVELS = 2.0
# Lags read on each side of the shifts involved when looking for a master frame's match.
REFUTE_LAG_REACH = 3


def _single_shift_refutes(domain, master_obj, candidate_obj, result, shift):
    """Whether a no-cut reading at `shift` may overturn a proposal that measured a master insert.

    The two-shift proposal is a measurement; a single shift that merely seats both anchors
    (a content check at a given shift, which held drawings or a static shot pass at either
    lag) is not enough to overturn it. Read on raw luma (`_grey_frames`) over the anchors'
    windows and the span between them, the refutation stands only when both hold:
    - every anchor window whose proposal shift differs from `shift` prefers `shift`: more of
      its frames read clearly closer under `shift` than under the proposal's shift, at least
      twice as many (the proposal's own margin there must be beaten, and a tie -- both lags
      within the span's noise of each other -- is no evidence for either);
    - no master frame between the anchors is unmatched (its lowest difference over every lag
      near the shifts above REFUTE_UNMATCHED_RATIO x the span's median lowest difference and
      REFUTE_NOISE_FLOOR_LEVELS): a master frame the candidate never shows is an insert,
      which no single shift explains.

    Returns:
        (refutes, evidence). An unreadable span keeps the previous rule (refutes True).
    """
    import numpy as np
    import frame_compare
    import scene_anchor
    anchor_a, anchor_b = result["anchor_a_frame"], result["anchor_b_frame"]
    before, after = result["before_shift_frames"], result["after_shift_frames"]
    window = scene_anchor.ANCHOR_WINDOW_FRAMES
    first = max(0, anchor_a - window)
    stop = anchor_b + window
    if stop - first > PIXEL_FRONT_MAX_FRAMES:
        return True, f"span_unread frames={stop - first}"
    lags = list(range(min(before, after, shift) - REFUTE_LAG_REACH,
                      max(before, after, shift) + REFUTE_LAG_REACH + 1))
    rate = domain["master_rate"]
    scale = domain.get("time_scale")
    comparer = frame_compare.FrameComparer(
        master_obj.filePath, candidate_obj.filePath, 0.0, 1.0, rate.numerator,
        rate.denominator,
        time_scales=None if scale is None else {candidate_obj.filePath: scale})
    try:
        if first + lags[0] < 0:
            return True, "span_unread candidate_before_start"
        master = _grey_frames(comparer, master_obj.filePath, first, stop - first, rate)
        candidate = _grey_frames(comparer, candidate_obj.filePath, first + lags[0],
                                 stop - first + lags[-1] - lags[0], rate)
    except Exception as error:                                           # noqa: BLE001
        return True, f"span_unread ({type(error).__name__}: {error})"
    count = len(master)
    if count < stop - first or len(candidate) < count + lags[-1] - lags[0]:
        return True, f"span_unread master={count} candidate={len(candidate)}"
    mad = np.full((count, len(lags)), np.inf)
    for column, lag in enumerate(lags):
        offset = lag - lags[0]
        mad[:, column] = np.abs(master - candidate[offset:offset + count]).mean(axis=(1, 2))
    lowest = mad.min(axis=1)
    median = float(np.median(lowest))
    noise = max(median, REFUTE_NOISE_FLOOR_LEVELS / 2.0)
    limit = max(REFUTE_UNMATCHED_RATIO * median, REFUTE_NOISE_FLOOR_LEVELS)
    inside = range(anchor_a - first, min(count, anchor_b + 1 - first))
    unmatched = [first + index for index in inside if lowest[index] > limit]
    column = {lag: lags.index(lag) for lag in lags}
    votes = []
    for side, low, high, own in (("A", anchor_a - window, anchor_a, before),
                                 ("B", anchor_b, anchor_b + window, after)):
        if own == shift:
            continue
        rows = range(max(0, low - first), min(count, high - first))
        for_shift = sum(1 for row in rows
                        if mad[row, column[shift]] + noise < mad[row, column[own]])
        for_own = sum(1 for row in rows
                      if mad[row, column[own]] + noise < mad[row, column[shift]])
        votes.append((side, own, for_shift, for_own))
    windows_agree = all(for_shift > 0 and for_shift >= 2 * for_own
                        for _, _, for_shift, for_own in votes)
    evidence = (f"shift={shift} proposal_shifts=[{before}, {after}] anchors=[{anchor_a}, "
                f"{anchor_b}] window_votes={[list(vote) for vote in votes]} "
                f"median_lowest={round(median, 3)} unmatched_limit={round(limit, 3)} "
                f"unmatched_frames={unmatched[:12]}")
    return windows_agree and not unmatched, evidence


def _resolve_interior(hole, domain, master_obj, candidate_obj, walk=None):
    """Resolve an interior hole: the audio proposes a step, the anchors and cross-sweep decide;
    sub-floor steps are offered a single-shift no-cut reading, ambiguous anchors the blind rule."""
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
            if (not result["declined"]
                    and result["before_shift_frames"] > result["after_shift_frames"]):
                # A measured master insert (the candidate lacks frames: before > after) is
                # overturned only by a single shift the picture itself prefers, with no
                # master frame left unmatched. A candidate excess has no master frame to
                # protect, and frames that differ there (retouched credits) are no insert.
                refutes, gate_evidence = _single_shift_refutes(
                    domain, master_obj, candidate_obj, result, probe["before_shift_frames"])
                tools.log_always(
                    f"repair: no_cut_refutation_gate probe={label} "
                    f"decision={'refuted' if refutes else 'proposal_stands'} {gate_evidence} "
                    f"for {candidate_obj.filePath}\n")
                if not refutes:
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
                                           if not result["declined"] else None),
                       "proposal_span_matches_before": result.get(
                           "unmatched_span_matches_before"),
                       "proposal_span_matches_after": result.get(
                           "unmatched_span_matches_after"),
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
            return _ambiguous_pin_outcome(hole, domain, result, candidate_obj.filePath, walk=walk,
                                          master_obj=master_obj)
        return _declined(hole, result.get("reason"), result.get("evidence"),
                         no_cut_probe_run=bool(hypotheses))
    verdict = _interior_verdict(result)
    if verdict == _orch().HOLE_PINNED_TO_AMBIGUOUS_ZONE_END:
        # Crossed or overlapping fronts: frames match both shifts, so the picture cannot
        # place the gap; the blind-span rule places it (shared silence, else right anchor).
        outcome = _blind_span_outcome(hole, domain, result, candidate_obj.filePath, walk=walk,
                                      master_obj=master_obj)
        outcome["pin_route"] = "sweep_fronts"
        return outcome
    if verdict == _orch().HOLE_RESOLVED:
        return _differing_span_outcome(hole, domain, result, candidate_obj.filePath, walk=walk,
                                       master_obj=master_obj, candidate_obj=candidate_obj)
    return _interior_outcome(hole, domain, result, verdict)


def _master_quiet(walk, times):
    """Whether the master is quiet in each 20 ms window: its RMS under BLIND_SILENCE_DB, the
    level `_blind_span_audio_point` already reads as the master's own silence."""
    return _window_levels(walk["master"], times, 0.0) < BLIND_SILENCE_DB


def _audio_continuous_span(hole, domain, sweep, before, after, walk):
    """Master frames (front, back) of a differing span the candidate's audio does not carry, the
    sweep itself without `walk`."""
    import numpy as np
    import audio_walk
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

    def frame_levels(samples, frames, shift):
        width = int(round(frame_s * audio_walk.WALK_RATE))
        starts = [int(round((frame + shift) * frame_s * audio_walk.WALK_RATE)) for frame in frames]
        return np.array([audio_walk.rms_db(samples[max(0, start):max(0, start + width)])
                         for start in starts])

    def candidate_quieter(shift, gap_frames):
        gap_master = frame_levels(walk["master"], gap_frames, 0)
        gap_candidate = frame_levels(walk["candidate"], gap_frames, shift)
        loud = (gap_master > BLIND_SILENCE_DB) & (gap_candidate > BLIND_SILENCE_DB)
        if int(loud.sum()) < FADE_GAP_MIN_FRAMES:
            return np.zeros(back - front, dtype=bool)
        usual_gap = float(np.median(gap_candidate[loud] - gap_master[loud]))
        master = frame_levels(walk["master"], range(front, back), 0)
        relative = frame_levels(walk["candidate"], range(front, back), shift) - master - usual_gap
        return (relative <= -FADE_LEVEL_DIFFER_DB) & (master > BLIND_SILENCE_DB)
    gap_count = int(round(FADE_GAP_SPAN_S / frame_s))

    def common_silence(frame, offset_ms, shift):
        start = frame * frame_s
        times = np.arange(start, start + frame_s - BLIND_SILENCE_WIN_S + 1e-9,
                          BLIND_SILENCE_HOP_S)
        if not len(times):
            times = np.array([start])
        return bool(_master_quiet(walk, times).all()
                    and _candidate_silent(walk, times, offset_ms / 1000.0,
                                          shift * frame_s).all())
    try:
        clamped = (first, last) != (front, back)
        while last > first and common_silence(last - 1, hole["offset_after_ms"], after):
            last -= 1
        while first < last and common_silence(first, hole["offset_before_ms"], before):
            first += 1
        if clamped and first < last:
            quieter_before = candidate_quieter(before, range(max(0, front - gap_count), front))
            quieter_after = candidate_quieter(after, range(back, back + gap_count))
            while first > front and quieter_before[first - 1 - front]:
                first -= 1
            while last < back and quieter_after[last - front]:
                last += 1
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"repair: audio_continuous_probe_failed ({type(error).__name__}: "
                      f"{error})\n")
        return front, back
    need = max(0, before - after)
    if last - first < need:
        last = min(back, first + need)
        first = max(front, last - need)
    return first, last


def join_levels(walk, domain, positions, before, after):
    """Loudest 20 ms peak (dB) of every deliverable candidate track around the two joins of a
    cut at each master frame of `positions`: the frame before and after it, at the before
    shift (where the candidate stops) and at the after shift (where it resumes), the offsets
    the delivery places the tracks with."""
    import numpy as np
    frame_s = float(domain["frame_ms"]) / 1000.0
    tracks = walk.get("silence_tracks") or [walk["candidate"]]
    levels = []
    for position in positions:
        times = np.arange(max(0.0, (position - 1) * frame_s),
                          (position + 1) * frame_s - BLIND_SILENCE_WIN_S + 1e-9,
                          BLIND_SILENCE_HOP_S)
        if not len(times):
            times = np.array([max(0.0, position * frame_s)])
        levels.append(max(float(_window_levels(track, times, shift * frame_s, peak=True).max())
                          for track in tracks for shift in (before, after)))
    return levels


def collapsed_cut_frame(domain, master_obj, candidate_obj, before, after, sweep, collapsed,
                        walk, candidate_path):
    """Master frame where a differing span that common silence emptied is cut, or None.

    Frames before the cut should show the candidate's picture at the before shift and frames
    from it on at the after shift (`_same_flags`, or a fade of the same shot: centred
    correlation from PIXEL_FADE_CORR); the cut keeps the most of them. The frame the silence
    collapsed onto (`collapsed`) stays when it is one of those positions with silent joins on
    every deliverable track (`join_levels`), or when none of them has silent joins; else the
    first such position (decision: collapsed-cut-keeps-matching-frames).
    """
    low, high = sweep
    if high <= low:
        return None
    try:
        readings = _pixel_readings(domain, master_obj, candidate_obj, low, high - low,
                                   sorted({before, after}))
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"repair: collapsed_cut_unread ({type(error).__name__}: {error})\n")
        return None
    if readings is None or not len(readings[before]):
        return None
    count = len(readings[before])

    def matching(shift):
        row = readings[shift]
        return [flag or (reading is not None and reading[7] >= PIXEL_FADE_CORR
                         and min(reading[5], reading[6]) >= PIXEL_NOISE_STD)
                for flag, reading in zip(_same_flags(row, range(count),
                                                     readings.get("master_shot")), row)]
    match_before, match_after = matching(before), matching(after)
    scores = [sum(match_before[:index]) + sum(match_after[index:]) for index in range(count + 1)]
    best = [low + index for index, score in enumerate(scores) if score == max(scores)]
    first, last = best[0], best[-1]
    point, placement = first, "picture"
    if walk is not None:
        import audio_walk
        joins = join_levels(walk, domain, best, before, after)
        silent = [position for position, level in zip(best, joins)
                  if level < audio_walk.AUDIBLE_DB]
        if collapsed in silent:
            point, placement = collapsed, "silence_collapse_point"
        elif silent:
            point, placement = silent[0], "silent_joins"
        else:
            point, placement = collapsed, "audio_fallback"
    tools.log_always(
        f"repair: collapsed_span_cut sweep_frames=[{low}, {high}] shifts=[{before}, {after}] "
        f"matching_frames={max(scores)}/{count} picture_frames=[{first}, {last}] "
        f"placement={placement} placed_at_frame={point} for {candidate_path}\n")
    return point


def _differing_span_outcome(hole, domain, result, candidate_path, walk=None, master_obj=None,
                            candidate_obj=None):
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
    the candidate at that side's audio offset. A span that this leaves empty is cut where the
    picture keeps the most matching frames (`collapsed_cut_frame`).
    """
    before, after = result["before_shift_frames"], result["after_shift_frames"]
    gap = after - before
    sweep = (result["master_start_frame"], result["master_end_frame"])
    front, back = _audio_continuous_span(hole, domain, sweep, before, after, walk)
    if front == back and (front, back) != sweep and master_obj is not None:
        placed = collapsed_cut_frame(domain, master_obj, candidate_obj, before, after, sweep,
                                     front, walk, candidate_path)
        if placed is not None:
            front = back = placed
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


def _resolve_edge(hole, domain, master_obj, candidate_obj, walk=None):
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
    zone_bound_ms = hole.get("zone_bound_ms")
    zone_lag = hole.get("zone_picture_lag_frames")
    _orch().step_launch("edge_walk", candidate=candidate_obj.filePath, edge=edge,
                bracket_ms=[round(low_ms, 2), round(high_ms, 2)],
                offset_ms=round(offset_ms, 3), master_timeline_ms=timeline_ms,
                zone_bound_ms=zone_bound_ms, zone_picture_lag_frames=zone_lag,
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
            zone_bound_ms=zone_bound_ms)
    except Exception as error:                                           # noqa: BLE001
        result = {"declined": True, "reason": f"resolver_raised:{type(error).__name__}",
                  "evidence": str(error)[:300]}
    anchor_shift = None if result["declined"] else result["shift_frames"]
    if anchor_shift is not None and not (
            (zone_lag is not None and abs(anchor_shift - zone_lag) <= 1)
            or abs(anchor_shift - round(float(offset_ms) / frame_ms)) <= 1):
        # (decision: edge-search-own-zone)
        tools.log_always(
            f"repair: edge_anchor_shift_disagrees edge={edge} anchor={result.get('anchor_frame')} "
            f"shift_frames={result['shift_frames']} zone_picture_lag_frames={zone_lag} "
            f"level_offset_ms={round(float(offset_ms), 3)} for {candidate_obj.filePath}\n")
        return _declined(hole, EDGE_ANCHOR_SHIFT_DISAGREES,
                         f"anchor shift {result['shift_frames']} frames vs zone picture lag "
                         f"{zone_lag} and audio offset {offset_ms} ms")
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
        shift = zone_lag if zone_lag is not None else hole.get("picture_shift_frames")
        if shift is None:
            shift = int(round(float(offset_ms) / frame_ms))
        try:
            fallback = _pixel_edge_walk(hole, domain, master_obj, candidate_obj, shift, walk)
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
        # pixels from the common side of the pHash boundary (decision: edge-search-own-zone).
        boundary_ms = float(result["boundary_frame"]) * frame_ms
        bracket = [0.0, boundary_ms] if edge == "head" else [boundary_ms, timeline_ms]
        walk_shift = (zone_lag if zone_lag is not None and abs(zone_lag - anchor_shift) <= 1
                      else anchor_shift)
        try:
            refined = _pixel_edge_walk(dict(hole, master_ms=bracket), domain, master_obj,
                                       candidate_obj, walk_shift, walk)
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
# An edge anchor more than one frame from both its zone's picture lag and its audio offset.
EDGE_ANCHOR_SHIFT_DISAGREES = "edge_anchor_shift_disagrees"
# Frames read on the common side of an edge bracket before the walk outward begins.
EDGE_PIXEL_MARGIN_FRAMES = 48


def _pixel_edge_walk(hole, domain, master_obj, candidate_obj, shift, walk=None):
    import audio_walk
    import scene_anchor
    edge = hole["kind"]
    frame_ms = float(domain["frame_ms"])
    rate = domain["master_rate"]
    master_last = int(Fraction(str(domain["master_timeline_ms"])) / 1000 * rate) - 1
    zone_bound_ms = hole.get("zone_bound_ms")
    if edge == "head":
        stop = min(master_last + 1,
                   int(math.ceil(float(hole["master_ms"][1]) / frame_ms)) + EDGE_PIXEL_MARGIN_FRAMES)
        if zone_bound_ms is not None:
            stop = min(stop, int(math.floor(float(zone_bound_ms) / frame_ms + 1e-6)))
        first, count = 0, stop
    else:
        first = max(0, int(float(hole["master_ms"][0]) / frame_ms) - EDGE_PIXEL_MARGIN_FRAMES)
        if zone_bound_ms is not None:
            first = max(first, int(math.ceil(float(zone_bound_ms) / frame_ms - 1e-6)))
        count = master_last + 1 - first
    if count <= 0 or count > PIXEL_FRONT_MAX_FRAMES or first + shift < 0 and edge == "tail":
        return None
    # Every master frame whose candidate frame exists at the exact shift is read: a missing
    # one-frame tolerance neighbour (candidate frame -1) leaves that reading to the shift.
    read_first = max(first, -shift)
    read_count = first + count - read_first

    def common_side(length):
        return (range(min(length, EDGE_PIXEL_MARGIN_FRAMES)) if edge == "tail"
                else range(max(0, length - EDGE_PIXEL_MARGIN_FRAMES), length))
    readings = _pixel_readings(domain, master_obj, candidate_obj, read_first, read_count,
                               [shift - 1, shift, shift + 1],
                               common=[(common_side(read_count), shift)])
    if readings is None:
        return None
    row = readings[shift]
    reference = common_side(len(row))
    same = [tolerant and not new_shot for tolerant, new_shot in
            zip(_tolerant_flags(readings, shift, reference),
                tolerance_only_at_new_shot(readings, shift, read_first, reference))]
    # Candidate frames that do not exist (before its start, after its end) read as absent.
    candidate_frames = domain.get("candidate_equivalent_duration_ms")
    candidate_last = (None if candidate_frames is None
                      else int(round(Fraction(str(candidate_frames)) / 1000 * rate)) - 1)
    order = list(range(len(same)) if edge == "tail" else range(len(same) - 1, -1, -1))
    common = [same[i] for i in order]
    # The walk starts at the first PIXEL_TRUST_RUN_FRAMES-long run of common picture: a
    # bracket the audio opened early can still hold the last cut's other side.
    start = next((i for i in range(len(common) - PIXEL_TRUST_RUN_FRAMES + 1)
                  if all(common[i:i + PIXEL_TRUST_RUN_FRAMES])), None)
    if start is None:
        return None
    boundary, run, termination, walked, max_run = None, 0, None, 0, 0
    walk_shift, stepped = shift, []
    edge_offset_ms = hole.get("offset_after_ms" if edge == "head" else "offset_before_ms")
    for position, index in enumerate(order[start:], start):
        frame = read_first + index
        if candidate_last is not None and frame + walk_shift > candidate_last:
            termination = _orch().EDGE_CANDIDATE_EXHAUSTED
            break
        walked += 1
        if same[index]:
            boundary, run = frame, 0
            continue
        run += 1
        max_run = max(max_run, run)
        if run >= scene_anchor.EDGE_WALK_SUSTAINED_MISMATCH_FRAMES and boundary is not None:
            span = order[position - run + 1:position - run + 1 + PIXEL_TRUST_RUN_FRAMES]
            step = None
            for neighbour in ((shift - 1, shift + 1)
                              if walk_shift == shift and walk and edge_offset_ms is not None else ()):
                exact = _same_flags(readings[neighbour], reference, readings.get("master_shot"))
                if step is None and all(exact[i] for i in span):
                    centre_s = (read_first + min(span)) * frame_ms / 1000.0
                    measured = audio_walk.zone_offset(
                        walk["master"], walk["candidate"], centre_s - audio_walk.WALK_WINDOW_S / 2,
                        centre_s + audio_walk.WALK_WINDOW_S / 2, float(edge_offset_ms))
                    if (measured["offset_ms"] is not None
                            and abs(measured["offset_ms"] - float(edge_offset_ms)) <= frame_ms / 2.0):
                        step = (neighbour, exact)
            if step is None:
                termination = _orch().EDGE_SUSTAINED_MISMATCH
                break
            walk_shift, same = step
            stepped.append((read_first + span[0], walk_shift))
            boundary, run = frame, 0
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
        f"termination={termination} walked={walked} picture_only_steps={stepped} "
        f"for {candidate_obj.filePath}\n")
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
    """Resolve one hole to exact frames (interior: `_resolve_interior`, head/tail: `_resolve_edge`), with no time limit.

    The hole carries `frame_domain` and `quantum_ms`; `walk` places a blind interior span inside a shared silence."""
    domain = hole.get("frame_domain")
    if domain is None:
        return _declined(hole, "frame_domain_absent")
    if hole["kind"] == "interior":
        outcome = _resolve_interior(hole, domain, master_obj, candidate_obj, walk=walk)
    elif hole["kind"] in ("head", "tail"):
        outcome = _resolve_edge(hole, domain, master_obj, candidate_obj, walk=walk)
    else:
        outcome = _declined(hole, f"no_resolver_for_kind:{hole['kind']}")
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


def log_partial_plan(candidate_path, cause, placed):
    """Log the partial plan placed before a stop; `placed` holds (what, decision, where)."""
    tools.log_always(f"repair: partial_plan cause={cause} placed={placed} for {candidate_path}\n")


def started_slices(video_s, slice_s):
    """Return the number of started `slice_s` slices in `video_s` seconds, at least one."""
    return max(1, math.ceil((video_s or 0.0) / slice_s))


def max_holes_per_couple(master_obj):
    """Return the hole cap per couple: MAX_HOLES_PER_COUPLE per started HOLE_BUDGET_SLICE_S."""
    video_ms = _video_duration_ms(master_obj)
    video_s = None if video_ms is None else float(video_ms) / 1000.0
    return started_slices(video_s, _orch().HOLE_BUDGET_SLICE_S) * _orch().MAX_HOLES_PER_COUPLE


def _speed_chain(audio, speed_ratio, engine="asetrate"):
    """Return a track's speed filter chain at `speed_ratio`, or None when the ratio is None."""
    if speed_ratio is None:
        return None
    import merge_video_resample
    rate = (audio.get("ffprobe") or {}).get("sample_rate") or audio.get("SamplingRate")
    return merge_video_resample.build_transform_chain(int(float(rate)), speed_ratio, engine)[0]


def reference_walk(reference, holes, master_obj, candidate_obj, language, speed_ratio,
                   candidate_path, engine="asetrate", frame_ms=None):
    """Run the millisecond audio walk on the reference couple on the file clock: (walk, None) or (None, reason).

    Seeded by the couple's zone offsets and the holes'; a stalled read re-raises step_stalled."""
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
        master = audio_walk.read_on_file_clock(master_obj, master_audio, envelope=envelope)
        candidate = audio_walk.read_on_file_clock(
            candidate_obj, candidate_audio, _speed_chain(candidate_audio, speed_ratio, engine),
            scale, envelope=envelope)
    except Exception as error:                                           # noqa: BLE001
        if isinstance(error, progress_watchdog.step_stalled):
            raise
        return None, f"the comparison tracks could not be read ({type(error).__name__}: {error})"
    silence_tracks = [candidate] + added_language_tracks(
        master_obj, candidate_obj, candidate_audio, speed_ratio, engine, scale, envelope,
        candidate_path)
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
            "candidate_stream": candidate_stream, "silence_tracks": silence_tracks,
            "master_audio_end_s": len(master) / audio_walk.WALK_RATE}, None


def added_language_tracks(master_obj, candidate_obj, comparison_audio, speed_ratio, engine,
                          scale, envelope, candidate_path):
    """The candidate's audio tracks in a language the master lacks, read like the comparison track.

    A track that cannot be read is logged and left out; a stalled read re-raises."""
    import audio_walk
    master_languages = set((master_obj.audios or {}).keys())
    tracks = []
    for language, audios in (candidate_obj.audios or {}).items():
        if language in master_languages:
            continue
        for audio in audios:
            if audio is comparison_audio:
                continue
            try:
                tracks.append(audio_walk.read_on_file_clock(
                    candidate_obj, audio, _speed_chain(audio, speed_ratio, engine), scale,
                    envelope=envelope))
            except Exception as error:                                   # noqa: BLE001
                if isinstance(error, progress_watchdog.step_stalled):
                    raise
                tools.log_always(f"repair: added_language_track_unread language={language} "
                                 f"stream={audio.get('StreamOrder')} ({type(error).__name__}: "
                                 f"{error}) for {candidate_path}\n")
    tools.log_always(f"repair: silence_tracks added_languages={len(tracks)} "
                     f"for {candidate_path}\n")
    return tracks


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
    import frame_compare
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
        if not (same[index] and row[index][2] < frame_compare.FLAT_STD):
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


def edge_silent_join(kind, outcome, domain, walk, candidate_path):
    """Rule 2 at an edge: move the join of a master edge fill onto the candidate's silence.

    The master fills the edge's differing frames whatever their count; the candidate then
    resumes (head) or stops (tail) on its first frame, at the edge's shift, that is silent on
    every deliverable track (`_candidate_silent` over the whole frame), searched over at most
    PIXEL_STEADY_FRAMES frames past the picture's boundary. Returns the new boundary frame
    (head: first candidate frame kept; tail: first filled frame), or None to keep the
    picture's own (decision: steady-level-per-master-shot).
    """
    import numpy as np
    shift = outcome.get("shift_frames")
    if walk is None or shift is None:
        return None
    last = outcome["master_last_frame"]
    if kind == "head":
        edge = outcome["master_end_frame"]
        if edge <= 0:
            return None
        frames = range(edge, min(last + 1, edge + PIXEL_STEADY_FRAMES + 1))
    else:
        edge = outcome["master_start_frame"]
        if edge > last or edge <= 0:
            return None
        frames = [frame + 1 for frame in range(edge - 1, max(-1, edge - PIXEL_STEADY_FRAMES - 2),
                                                -1)]
    frame_s = float(domain["frame_ms"]) / 1000.0
    shift_s = shift * frame_s

    def silent(frame):
        start = frame * frame_s
        times = np.arange(start, start + frame_s - BLIND_SILENCE_WIN_S + 1e-9,
                          BLIND_SILENCE_HOP_S)
        return bool(_candidate_silent(walk, times, shift_s).all())
    for join in frames:
        if silent(join if kind == "head" else join - 1):
            break
    else:
        return None
    if join == edge:
        return None
    tools.log_always(f"repair: edge_silent_join edge={kind} picture_boundary={edge} "
                     f"shift_frames={shift} placed_at_frame={join} for {candidate_path}\n")
    return join


# Seconds read on each side of a tail level edge to tell whether both tracks are silent there.
TAIL_EDGE_SILENCE_S = 0.1


# (decision: silent-edge-span-no-fill)
def silent_edge_covered(walk, kind, offset_ms, span_s, domain, candidate_obj, candidate_path):
    import numpy as np
    import audio_walk
    import frame_count_check
    try:
        times = np.arange(span_s[0], max(span_s[0] + BLIND_SILENCE_HOP_S / 2, span_s[1]),
                          BLIND_SILENCE_HOP_S)
        offset_s = float(offset_ms) / 1000.0
        master_peak = float(_window_levels(walk["master"], times, 0.0, peak=True).max())
        candidate_peak = float(_window_levels(walk["candidate"], times, offset_s,
                                              peak=True).max())
        silent = max(master_peak, candidate_peak) < audio_walk.AUDIBLE_DB
        extent = (frame_count_check.candidate_extent_ms(domain, domain.get("time_scale"),
                                                        candidate_obj) if silent else None)
        frame_ms = float(domain["frame_ms"])
        timeline_ms = float(domain["master_timeline_ms"])
        if extent is None:
            video_on_master_ms = audio_on_master_s = None
            covered = False
        elif kind == "head":
            video_on_master_ms = float(extent[0]) - float(offset_ms)
            audio_on_master_s = -offset_s
            covered = video_on_master_ms <= frame_ms and audio_on_master_s <= 0.0
        else:
            video_on_master_ms = float(extent[1]) - float(offset_ms)
            audio_on_master_s = len(walk["candidate"]) / audio_walk.WALK_RATE - offset_s
            covered = (video_on_master_ms >= timeline_ms - frame_ms
                       and audio_on_master_s * 1000.0 >= timeline_ms)
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"repair: edge_silence_unread edge={kind} "
                      f"({type(error).__name__}: {error})\n")
        return False
    tools.log_always(
        f"repair: edge_span_silence edge={kind} span_s=[{round(span_s[0], 4)}, "
        f"{round(span_s[1], 4)}] offset_ms={offset_ms} master_peak_db={round(master_peak, 1)} "
        f"candidate_peak_db={round(candidate_peak, 1)} "
        f"candidate_video_edge_on_master_ms="
        f"{None if video_on_master_ms is None else round(video_on_master_ms, 3)} "
        f"candidate_audio_edge_on_master_s="
        f"{None if audio_on_master_s is None else round(audio_on_master_s, 3)} "
        f"master_timeline_ms={round(timeline_ms, 3)} "
        f"decision={'no_fill' if covered else 'fill_kept'} for {candidate_path}\n")
    return covered


def audio_edges(walk, holes, domain, master_obj, candidate_obj, work_dir, candidate_path,
                shift_hints=None, zones=None):
    """Return (head_end_s, tail_start_s, refusal) from the edge walks; None edge = no fill."""
    import audio_walk
    master, candidate = walk["master"], walk["candidate"]
    first, last = walk["levels"][0], walk["levels"][-1]
    frame = float(domain["frame_ms"]) / 1000.0
    timeline_s = float(domain["master_timeline_ms"]) / 1000.0
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
            zone = (zones or {}).get(kind) or {}
            offsets["zone_bound_ms"] = zone.get("bound_ms")
            offsets["zone_picture_lag_frames"] = zone.get("lag")
            edge_jobs[kind] = (index, dict(hole, **offsets))
    edge_outcomes = dict(zip(
        edge_jobs.keys(),
        repair_pool.run_parallel([
            (lambda k=kind, idx=index, h=hole: _resolve_logged(
                candidate_path, idx, h, domain, master_obj, candidate_obj, work_dir, walk=walk))
            for kind, (index, hole) in edge_jobs.items()]))) if edge_jobs else {}
    decisions, summary = {}, []
    for kind, audio_s, level in (("head", head_s, first), ("tail", tail_s, last)):
        video_s, video_concluded, decline_reason = None, False, None
        if kind in by_kind:
            outcome = edge_outcomes[kind]
            if outcome.get("resolver_reason") == "step_stalled":
                log_partial_plan(candidate_path, "step_stalled",
                                 [(edge, "placed", value) for edge, value in decisions.items()]
                                 + [(kind, "stopped", audio_s)])
                return None, None, ("step_stalled", f"the video on the {kind} edge made no "
                                                    f"progress ({outcome.get('evidence')}) -- the "
                                                    f"partial plan is logged; declined, retried at "
                                                    f"the next run")
            if outcome["status"] in _orch().EDGE_TERMINATIONS or outcome["status"] == _orch().HOLE_NO_CUT_CONFIRMED:
                try:
                    moved = _edge_blind_point(kind, outcome, domain, master_obj, candidate_obj,
                                              walk)
                except Exception as error:                               # noqa: BLE001
                    tools.dev_log(f"repair: edge_blind_span_unread ({type(error).__name__}: "
                                  f"{error})\n")
                    moved = None
                if moved is None:
                    moved = edge_silent_join(kind, outcome, domain, walk, candidate_path)
                if moved is not None:
                    outcome = dict(outcome, **({"master_end_frame": moved} if kind == "head"
                                               else {"master_start_frame": moved}))
                video_s =_frame_s(outcome["master_end_frame"] if kind == "head"
                                   else outcome["master_start_frame"], domain)
                video_concluded = True
            elif (by_kind[kind][1].get("master_span_seconds") is not None
                  and by_kind[kind][1]["master_span_seconds"] <= frame):
                # No master-exclusive runtime exists before (head) or after (tail) this
                # bracket: the candidate simply runs past the master's own end here, or starts
                # before its own beginning. There is nothing of the master's own left to search
                # for, so this is a trim at the master's own edge, not a decline.
                video_s = 0.0 if kind == "head" else timeline_s
                video_concluded = True
            elif outcome.get("resolver_reason") == EDGE_ANCHOR_SHIFT_DISAGREES:
                pass
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
            span_s = (0.0, video_s) if kind == "head" else (video_s, timeline_s)
            shift = outcome.get("shift_frames")
            picture_ms = None if shift is None else shift * frame * 1000.0
            one_offset = (picture_ms is not None
                          and abs(picture_ms - level["off_ms"]) <= frame * 1000.0)
            if one_offset and span_s[1] - span_s[0] >= frame:
                if silent_edge_covered(walk, kind, picture_ms,
                                       (span_s[0], span_s[1] - BLIND_SILENCE_WIN_S), domain,
                                       candidate_obj, candidate_path):
                    placed, decision = None, "edge_span_silent_candidate_covers"
                elif edge_source[kind] == "audio_edge" and (
                        audio_s > video_s if kind == "tail" else audio_s < video_s):
                    index = (math.ceil(audio_s / frame - 1e-6) if kind == "tail"
                             else math.floor(audio_s / frame + 1e-6))
                    placed = _frame_s(index, domain) if index > 0 else None
                    decision = "video_edge_boundary_audio_common"
        else:
            placed, decision = audio_s, edge_source[kind]
            if kind == "head":
                placed, decision = head_content_edge(placed, decision, frame, lead)
            elif decision == "audio_edge" and silent_edge_covered(
                    walk, kind, level["off_ms"],
                    (max(0.0, audio_s - TAIL_EDGE_SILENCE_S), audio_s + TAIL_EDGE_SILENCE_S),
                    domain, candidate_obj, candidate_path):
                placed, decision = None, "tail_edge_in_silence_candidate_covers_end"
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
    if tail_start is not None and tail_start >= timeline_s - frame:
        tail_start = None
    log_edge_fill_sound(walk, head_end, tail_start, candidate_path)
    return head_end, tail_start, None


def log_edge_fill_sound(walk, head_end, tail_start, candidate_path):
    """Log, as a finding, an edge fill that carries audible master sound while the candidate
    brings tracks in a language the master lacks: that sound lands in those tracks."""
    import numpy as np
    import audio_walk
    added = len(walk.get("silence_tracks") or []) - 1
    if added <= 0:
        return
    end_s = walk.get("master_audio_end_s") or 0.0
    for edge, start_s, stop_s in (("head", 0.0, head_end), ("tail", tail_start, end_s)):
        if start_s is None or stop_s is None or stop_s - start_s < BLIND_SILENCE_WIN_S:
            continue
        times = np.arange(start_s, stop_s - BLIND_SILENCE_WIN_S + 1e-9, BLIND_SILENCE_HOP_S)
        peak = float(_window_levels(walk["master"], times, 0.0, peak=True).max())
        if peak > audio_walk.AUDIBLE_DB:
            tools.log_always(f"repair: edge_fill_carries_master_sound edge={edge} "
                             f"fill_s=[{round(start_s, 4)}, {round(stop_s, 4)}] "
                             f"master_peak_db={round(peak, 1)} added_language_tracks={added} "
                             f"for {candidate_path}\n")


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


# Picture lags searched each side of a level's own lag: a candidate's audio can sit a few frames
# off its picture (measured: 2 frames on a 23.976 fps pair).
LEVEL_GAP_SHIFT_FRAMES = 3


def level_lag_reading(domain, master_obj, candidate_obj, first, count, centre):
    shifts = list(range(centre - LEVEL_GAP_SHIFT_FRAMES, centre + LEVEL_GAP_SHIFT_FRAMES + 1))
    readings = _pixel_readings(domain, master_obj, candidate_obj, first, count, shifts)
    if readings is None:
        return f"master frames {first}-{first + count} could not be read"
    flags = {shift: _same_flags(readings[shift], (), readings["master_shot"]) for shift in shifts}
    counts = {shift: sum(flags[shift]) for shift in shifts}
    shift = max(shifts, key=lambda value: (counts[value], -abs(value - centre)))
    if _longest_run(flags[shift]) < PIXEL_TRUST_RUN_FRAMES:
        return f"no lag near {centre} frames holds {PIXEL_TRUST_RUN_FRAMES} frames"
    tied = [value for value in shifts if value != shift and counts[value] == counts[shift]]
    return shift, flags[shift], counts, tied


def audio_transitions(walk, reference, domain, master_obj, candidate_obj, work_dir,
                      candidate_path, language=None, union=None):
    """Place every walk change point on the master timeline: (transitions, None) or (None, (cause, reason)).
    """
    import audio_walk
    points = [p for p in walk["points"] if p["kind"] == "change_point"]
    holes, frame = [], float(domain["frame_ms"]) / 1000.0
    hole_quantum_ms = reference["alignment"]["quantum_ms"]
    for index, point in enumerate(points):
        edges = point.get("edges") or {}
        holes.append(_cp_hole(point, reference, index) if edges.get("status") == "ok" else None)
    # Nearby change points share one scene pass.
    cluster_holes([hole for hole in holes if hole is not None])
    # Every hole's exact-frame search is independent (holes under the merge window share only
    # a read-through decode cache, never a decision), so they run together on the shared pool
    # instead of one change point at a time.
    localised = [index for index, hole in enumerate(holes) if hole is not None]
    hole_outcomes = dict(zip(localised, repair_pool.run_parallel([
        (lambda i=index, h=holes[index]: _resolve_logged(
            candidate_path, f"change_point_{i}", h, domain, master_obj, candidate_obj, work_dir,
            walk=walk))
        for index in localised])))
    transitions = []
    for index, (point, hole) in enumerate(zip(points, holes)):
        if hole is None:
            # (decision: level-gap-picture-step)
            before, after = point["level_before"], point["level_after"]
            level_ms = point["a_ms"]
            for earlier in reversed(transitions):
                if earlier["decision"] != "audio_only_step_picture_continuous":
                    break
                level_ms = earlier["a_ms"]
            frame_ms, rate = float(domain["frame_ms"]), domain["master_rate"]
            first = max(0, math.floor((before["t_last"] - audio_walk.WALK_HOP_S) * rate))
            last = math.ceil((after["t_first"] + audio_walk.WALK_WINDOW_S + audio_walk.WALK_HOP_S)
                             * rate)
            placed, sides = None, []
            if min(before["n"], after["n"]) < audio_walk.LEVEL_PERSIST:
                reading = f"a level holds fewer than {audio_walk.LEVEL_PERSIST} windows"
            elif last - first > PIXEL_FRONT_MAX_FRAMES:
                reading = f"the gap between the levels spans {last - first} frames"
            else:
                sides = [level_lag_reading(domain, master_obj, candidate_obj, first, last - first,
                                           round(level_ms / frame_ms)),
                         level_lag_reading(domain, master_obj, candidate_obj, first, last - first,
                                           round(point["b_ms"] / frame_ms))]
                reading = next((side for side in sides if isinstance(side, str)), None)
            if reading is None:
                (left, left_flags, left_counts, left_tied), (right, right_flags, right_counts,
                                                             right_tied) = sides
                common = {"a_ms": point["a_ms"], "b_ms": point["b_ms"],
                          "video_status": _orch().HOLE_RESOLVED, "before_shift_frames": left,
                          "after_shift_frames": right, "blind_span": None, "sweep_crossed": False,
                          "change_point": index}
                opening = next((at for at in range(len(right_flags) - PIXEL_TRUST_RUN_FRAMES + 1)
                                if all(right_flags[at:at + PIXEL_TRUST_RUN_FRAMES])), None)
                closing = (None if opening is None
                           else next((at for at in range(opening - 1, -1, -1) if left_flags[at]),
                                     None))
                if left == right and not left_tied and not right_tied:
                    at = _frame_s(first + (last - first) // 2, domain)
                    reading = (f"lag {left} frames on both sides (counts {left_counts} / "
                               f"{right_counts}): the picture holds one offset")
                    placed = dict(common, at_s=at, fill_s=0.0, interval=[at, at], edges=[at, at],
                                  decision="audio_only_step_picture_continuous", video_s=at,
                                  video_span_frames=None, anchor_frames=[None, None],
                                  ambiguous_sides=[])
                elif closing is None:
                    reading = f"no frame matches lag {left} before the first run at lag {right}"
                else:
                    fill_frames = opening - closing - 1
                    removed_frames = fill_frames + right - left
                    picture_step_ms = (right - left) * frame_ms
                    audio_step_ms = point["b_ms"] - level_ms
                    reading = (f"lags {left}/{right} frames (counts {left_counts} / "
                               f"{right_counts}), master frames {first + closing}|"
                               f"{first + opening}, fill {fill_frames} frames, candidate removal "
                               f"{removed_frames} frames, picture step "
                               f"{round(picture_step_ms, 3)} ms against the audio's "
                               f"{round(audio_step_ms, 3)} ms")
                    if (removed_frames >= 0 and abs(picture_step_ms - audio_step_ms)
                            <= frame_ms + hole_quantum_ms):
                        at = _frame_s(first + closing + 1, domain)
                        placed = dict(common, at_s=at, fill_s=fill_frames * frame_ms / 1000.0,
                                      interval=[at, at], decision="level_gap_picture_step",
                                      edges=[at, _frame_s(first + opening, domain)], video_s=at,
                                      video_span_frames=fill_frames,
                                      anchor_frames=[first + closing, first + opening],
                                      ambiguous_sides=[side for side, tied in (
                                          ("before", left_tied), ("after", right_tied)) if tied])
            tools.log_always(
                f"repair: level_gap_picture_step change_point={index} a_ms={point['a_ms']} "
                f"b_ms={point['b_ms']} levels_s=[{before['t_last']}, {after['t_first']}] "
                f"placed={placed is not None} {reading} for {candidate_path}\n")
            if placed is None:
                return None, ("video_cut_undetermined",
                              f"the walk measured a {point['jump_ms']} ms step between levels "
                              f"{point['a_ms']} and {point['b_ms']} ms (t {before['t_last']} -> "
                              f"{after['t_first']} s) whose audio edges could not be read, and "
                              f"the picture between the levels does not place it ({reading})")
            transitions.append(placed)
            continue
        edges = point["edges"]
        lo, hi = edges["interval"]
        extra = edges["extra_s"]
        jump = point["b_ms"] - point["a_ms"]
        sub_quantum = abs(jump) < hole["quantum_ms"]
        outcome = hole_outcomes[index]
        if outcome.get("resolver_reason") == "step_stalled":
            log_partial_plan(candidate_path, "step_stalled",
                             [(f"change_point_{t['change_point']}", t["decision"], t["at_s"],
                               t["fill_s"]) for t in transitions]
                             + [(f"change_point_{index}", "stopped", lo, hi)])
            return None, ("step_stalled", f"the video on change point {index} ({lo}-{hi} s) made "
                                          f"no progress ({outcome.get('evidence')}) -- the "
                                          f"partial plan is logged; declined, retried at the "
                                          f"next run")
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
            # Rule 2: every unmatched master frame is filled, not only the shift gap, unless the
            # candidate audio runs on across them (a picture-only difference: no effect).
            fill = (max(0, before - after,
                        outcome["master_end_frame"] - outcome["master_start_frame"]) * frame
                    if status == _orch().HOLE_RESOLVED else max(0, before - after) * frame)
            if fill > max(0, before - after) * frame and walk is not None:
                span_s = (_frame_s(outcome["master_start_frame"], domain),
                          _frame_s(outcome["master_end_frame"], domain))
                reading = audio_walk.zone_offset(walk["master"], walk["candidate"], span_s[0],
                                                 span_s[1], point["b_ms"])
                measured = reading["offset_ms"]
                share = (max((level["n"] for level in reading["levels"]), default=0)
                         / max(1, sum(reading["counts"].values())))
                continuous = (measured is not None and share >= UNMATCHED_SPAN_AUDIO_SHARE and min(
                    abs(measured - point["a_ms"]), abs(measured - point["b_ms"])) <= frame * 500.0)
                quiet = (quiet_frame_in_interval(walk, domain, (lo, hi),
                                                 (point["a_ms"], point["b_ms"]))
                         if continuous else None)
                tools.log_always(
                    f"repair: unmatched_span_audio change_point={index} span_s=[{round(span_s[0], 4)}, "
                    f"{round(span_s[1], 4)}] audio_offset_ms={measured} share={round(share, 2)} continuous={continuous} "
                    f"quiet_frame={quiet} for {candidate_path}\n")
                if continuous:
                    fill = max(0, before - after) * frame
                    if quiet is not None:
                        at = _frame_s(quiet, domain)
            pinned, pin_decision = video_pin(video_s, (lo, hi), (edges["edge_A"], edges["edge_B"]),
                                             extra, frame)
            inside = (None if pinned is not None and not width_note
                      else _blind_point_in_audio_interval(outcome, domain, (lo, hi)))
            if pinned is not None and not width_note:
                decision = "video_decides"
            elif inside is not None:
                # A blind-span placement is not a picture reading: where the picture allows
                # positions inside the audio interval, the fill starts there.
                blind = outcome["blind_span"]
                at = _frame_s(inside + min(0, blind["gap_frames"]), domain)
                fill = max(0, -blind["gap_frames"]) * frame
                tools.log_always(
                    f"repair: blind_span_inside_audio_interval change_point={index} "
                    f"placement={blind['placement']} placed_at_frame={blind['point']} "
                    f"moved_to_frame={inside} allowed_frames={blind['allowed_frames']} "
                    f"audio_interval_s=[{lo}, {hi}] at_s={round(at, 4)} for {candidate_path}\n")
                decision = "blind_span_inside_audio_interval"
            else:
                tools.log_always(
                    f"repair: video_audio_disagree_followed_video change_point={index} "
                    f"video_s={round(at, 4)} video_fill_ms={round(fill * 1000.0, 3)} "
                    f"audio_interval_s=[{lo}, {hi}] audio_fill_ms={round(extra * 1000.0, 3)} "
                    f"{width_note or 'video cut outside the audio interval and edges'} "
                    f"for {candidate_path}\n")
                decision = "video_audio_disagree_followed_video"
        elif status == _orch().HOLE_NO_CUT_CONFIRMED:
            # A still picture has no instant of its own: outside the audio interval, the junction
            # moves to where the candidate is silent at both offsets.
            at, fill, decision = _frame_s(outcome["master_start_frame"], domain), 0.0, \
                "video_no_cut_confirmed"
            quiet = (None if lo - frame <= at <= hi + frame else
                     quiet_frame_in_interval(walk, domain, (lo, hi), (point["a_ms"], point["b_ms"])))
            tools.log_always(
                f"repair: no_cut_junction change_point={index} still_edge_s={round(at, 4)} "
                f"audio_interval_s=[{lo}, {hi}] quiet_frame={quiet} for {candidate_path}\n")
            if quiet is not None:
                at = _frame_s(quiet, domain)
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
        pieces = _transition_pieces(at, fill, point["a_ms"], point["b_ms"], decision, before,
                                    after, removal_s)
        for piece_at, piece_fill, a_ms, b_ms, piece_decision, left, right in pieces:
            transitions.append({"at_s": piece_at, "fill_s": piece_fill, "a_ms": a_ms,
                                "b_ms": b_ms, "decision": piece_decision, "interval": [lo, hi],
                                "edges": [edges["edge_A"], edges["edge_B"]],
                                "video_status": status, "video_s": video_s,
                                "change_point": index, "before_shift_frames": left,
                                "after_shift_frames": right,
                                # Kept for `align_transitions_to_zone_lags` (one-piece cuts).
                                "blind_span": (outcome.get("blind_span")
                                               if len(pieces) == 1 else None),
                                "video_span_frames": (
                                    outcome["master_end_frame"] - outcome["master_start_frame"]
                                    if video_decided and len(pieces) == 1 else None),
                                **anchor_reading(outcome)})
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


# Unmatched master frames are a picture-only difference when at least this share of the audio
# windows over them reads one bounding zone's offset.
UNMATCHED_SPAN_AUDIO_SHARE = 0.8


# Blind-span placements read off the master picture itself (its next shot change): the audio
# interval does not move them.
PICTURE_PLACEMENTS = ("right_anchor",)
# An audio-alignment run of placements at most this wide that touches the walk's interval places a
# blind fill as exactly as that interval.
BLIND_RUN_EXACT_FRAMES = 3


def _blind_point_in_audio_interval(outcome, domain, interval):
    blind = outcome.get("blind_span")
    if not blind or blind["placement"] in PICTURE_PLACEMENTS or blind["gap_frames"] >= 0:
        return None
    import audio_walk
    low, high = blind["allowed_frames"]
    gap = blind["gap_frames"]
    # The interval's bounds are window starts: the instant a 20 ms window reads is its centre.
    latest = interval[1] + audio_walk.FINE_WIN_S / 2.0
    lo, hi = sorted(interval)
    frame_s = float(domain["frame_ms"]) / 1000.0
    run = blind.get("run") if blind["placement"] == "audio_alignment" else None
    if (run is not None and run[1] - run[0] + 1 <= BLIND_RUN_EXACT_FRAMES
            and any(lo - frame_s <= _frame_s(point + min(0, gap), domain) <= hi + frame_s
                    for point in range(run[0], run[1] + 1))):
        return None
    best = None
    for point in range(low - 1, high + 2):
        start_s = _frame_s(point + min(0, gap), domain)
        if not lo - frame_s <= start_s <= hi + frame_s:
            continue
        key = (start_s > latest + 1e-9, abs(latest - start_s))
        if best is None or key < best[0]:
            best = (key, point)
    return None if best is None else best[1]


def anchor_reading(outcome):
    sides = [side for side, key in (("before", "anchor_a_ambiguous"), ("after", "anchor_b_ambiguous"))
             if outcome.get(key)]
    return {"anchor_frames": [outcome.get("anchor_a_frame"), outcome.get("anchor_b_frame")],
            "sweep_crossed": bool(outcome.get("sweep_crossed")), "ambiguous_sides": sides,
            "flat_sides": outcome.get("anchor_flat_sides") or []}


# A zone majority this strong beats an anchor read on a fade, a black span or a crossed window,
# however far apart (decision: zone-majority-beats-unreliable-anchor).
ZONE_LAG_DECISIVE_EVENTS = 20
ZONE_LAG_DECISIVE_SHARE = 0.9


def zone_lag_decisive(lag, counts):
    """Whether `lag` holds at least ZONE_LAG_DECISIVE_EVENTS changes and ZONE_LAG_DECISIVE_SHARE of them.

    >>> from collections import Counter
    >>> zone_lag_decisive(43, Counter({43: 104, 48: 3, 42: 2}))
    True
    """
    if lag is None:
        return False
    total = sum(counts.values())
    return counts[lag] >= ZONE_LAG_DECISIVE_EVENTS and counts[lag] >= ZONE_LAG_DECISIVE_SHARE * total


def flat_window(master_frames):
    """Whether every grey master frame is flat (std under frame_compare.FLAT_STD): black or a plain field."""
    import frame_compare
    return len(master_frames) > 0 and all(float(frame.std()) < frame_compare.FLAT_STD
                                          for frame in master_frames)


def unreliable_anchor_cause(transition, domain, master_obj):
    import frame_compare
    import scene_anchor
    if transition.get("blind_span") is not None or transition.get("sweep_crossed"):
        return "crossed"
    if transition.get("flat_sides"):
        return "flat"
    anchor_a, anchor_b = transition.get("anchor_frames") or [None, None]
    if master_obj is None or anchor_a is None or anchor_b is None:
        return None
    width = scene_anchor.ANCHOR_WINDOW_FRAMES
    rate = domain["master_rate"]
    comparer = frame_compare.FrameComparer(master_obj.filePath, master_obj.filePath, 0.0, 1.0,
                                           rate.numerator, rate.denominator)
    for first in (max(0, anchor_a - width), anchor_b):
        try:
            frames = _grey_frames(comparer, master_obj.filePath, first, width, rate)
        except Exception as error:                                           # noqa: BLE001
            tools.dev_log(f"repair: anchor_window_unread ({type(error).__name__}: {error})\n")
            return None
        if _is_fade_window([float(frame.mean()) for frame in frames]):
            return "fade"
        if flat_window(frames):
            return "black"
    return None


def zone_lags_over_anchor(before, after, left, left_counts, right, right_counts, cause):
    """Return the (left, right) shifts a cut takes from its zones, or None to keep its anchors.

    The rule of `align_transitions_to_zone_lags`: a side takes its zone's majority when it is
    within one frame of the anchor; when the anchor was read on a fade, a black span or a
    crossed window (`cause`), a decisive majority (`zone_lag_decisive`) wins at any distance
    and a side whose zone has no majority keeps its anchor.

    >>> from collections import Counter
    >>> zone_lags_over_anchor(19, 41, 18, Counter({18: 180}), 43, Counter({43: 104}), "crossed")
    (18, 43)
    >>> zone_lags_over_anchor(19, 41, 18, Counter({18: 180}), 43, Counter({43: 104}), None) is None
    True
    """
    shifts = []
    for anchor, lag, counts in ((before, left, left_counts), (after, right, right_counts)):
        if cause is not None and zone_lag_decisive(lag, counts):
            shifts.append(lag)
        elif lag is not None and abs(lag - anchor) <= 1:
            shifts.append(lag)
        elif lag is None and cause is not None:
            shifts.append(anchor)
        else:
            return None
    decisive = cause is not None and (zone_lag_decisive(left, left_counts)
                                      or zone_lag_decisive(right, right_counts))
    if (left is None or right is None) and not decisive:
        return None
    return tuple(shifts)


def _zone_bounds_s(transitions, index, domain):
    """Master seconds (start, end) of the zones left and right of transition `index`."""
    end_s = float(domain["master_timeline_ms"]) / 1000.0
    left_start = (transitions[index - 1]["at_s"] + transitions[index - 1]["fill_s"]
                  if index > 0 else 0.0)
    right_end = transitions[index + 1]["at_s"] if index + 1 < len(transitions) else end_s
    current = transitions[index]
    return (left_start, current["at_s"]), (current["at_s"] + current["fill_s"], right_end)


def align_transitions_to_zone_lags(transitions, pairs, domain, walk, master_obj,
                                   candidate_path):
    """Bring every cut's two picture shifts to the majority lag of the zones it separates.

    `pairs` are the matched scene changes of both whole files (`zone_picture_lag`). The anchors
    of a cut read one 12-frame window on each side; where that reading is one frame off its
    zone's majority (a jittering cadence, drawings held on twos), the zone's majority wins:
    - a blind-span gap is placed again at the corrected width (audio, else the right anchor,
      then the audio interval, as `_blind_span_outcome` and `audio_transitions` do);
    - a resolved cut keeps its frames and fills at least the corrected shift gap;
    - a cut the picture read as continuous (one shift) takes the zone lag when both zones agree.
    A side whose majority is unknown, or more than one frame from the anchor, keeps the anchor,
    unless the anchor was read on a fade, a black span or a crossed window and a zone majority
    is decisive (`zone_lags_over_anchor`). Each correction is logged (`zone_lag_correction`); an
    anchor kept against both zones is marked `zone_lag_disagrees_anchor`.

    Returns:
        the transitions (updated in place where corrected).
    """
    import zone_picture_lag
    if not pairs:
        return transitions
    frame_ms = float(domain["frame_ms"])
    frame_s = frame_ms / 1000.0
    for index, transition in enumerate(transitions):
        before, after = transition.get("before_shift_frames"), transition.get("after_shift_frames")
        if before is None or after is None or transition["decision"] == "blind_removal":
            continue
        (left_lo, left_hi), (right_lo, right_hi) = _zone_bounds_s(transitions, index, domain)
        left, left_counts = zone_picture_lag.zone_majority(pairs, left_lo * 1000.0,
                                                           left_hi * 1000.0, frame_ms)
        right, right_counts = zone_picture_lag.zone_majority(pairs, right_lo * 1000.0,
                                                             right_hi * 1000.0, frame_ms)
        if (left, right) == (before, after):
            continue
        cause = None
        lags = zone_lags_over_anchor(before, after, left, left_counts, right, right_counts, None)
        if lags is None and (zone_lag_decisive(left, left_counts)
                             or zone_lag_decisive(right, right_counts)):
            cause = unreliable_anchor_cause(transition, domain, master_obj)
            lags = zone_lags_over_anchor(before, after, left, left_counts, right, right_counts,
                                         cause)
        if lags is None:
            if left is not None and right is not None:
                transition["zone_lag_disagrees_anchor"] = True
                tools.log_always(
                    f"repair: zone_lag_disagrees_anchor change_point={transition['change_point']} "
                    f"anchor_shifts=[{before}, {after}] zone_lags=[{left}, {right}] "
                    f"left_counts={dict(left_counts.most_common(3))} "
                    f"right_counts={dict(right_counts.most_common(3))} anchor_reading={cause} "
                    f"decision=anchor_kept for {candidate_path}\n")
            continue
        left, right = lags
        if (left, right) == (before, after):
            continue
        old = (transition["at_s"], transition["fill_s"])
        blind = transition.get("blind_span")
        gap = right - left
        if before == after and left != right:
            # The picture read no cut here; one frame of jitter does not make one.
            continue
        if blind is not None:
            data = dict(blind["blind"], before_shift_frames=left, after_shift_frames=right)
            low, high = _blind_span_bounds(data, gap)
            found = _blind_span_audio_point(domain, gap, left, right, low, high, walk)
            anchor = (_right_anchor_frame(domain, master_obj, low, high, -gap)
                      if found is None and gap < 0 else None)
            point, placement = ((found[0], found[2]) if found is not None
                                else (anchor, "right_anchor") if anchor is not None
                                else (high, "right_edge"))
            replaced = {"blind_span": {"placement": placement, "allowed_frames": [low, high],
                                       "gap_frames": gap, "point": point,
                                       "run": found[1] if found is not None else None,
                                       "blind": blind["blind"]}}
            interval = transition.get("interval")
            start_s = _frame_s(point + min(0, gap), domain)
            if interval is not None and not (interval[0] - frame_s <= start_s
                                             <= interval[1] + frame_s):
                inside = _blind_point_in_audio_interval(replaced, domain, interval)
                if inside is not None:
                    point, placement = inside, f"{placement}_inside_audio_interval"
                    replaced["blind_span"].update(point=point, placement=placement)
            transition["at_s"] = _frame_s(point + min(0, gap), domain)
            transition["fill_s"] = max(0, -gap) * frame_s
            transition["blind_span"] = replaced["blind_span"]
        elif transition.get("video_span_frames") is not None and before != after:
            transition["fill_s"] = max(0, left - right, transition["video_span_frames"]) * frame_s
        transition["before_shift_frames"], transition["after_shift_frames"] = left, right
        transition["ambiguous_sides"] = []
        tools.log_always(
            f"repair: zone_lag_correction change_point={transition['change_point']} "
            f"anchor_shifts=[{before}, {after}] zone_lags=[{left}, {right}] "
            f"anchor_reading={cause} "
            f"left_counts={dict(left_counts.most_common(3))} "
            f"right_counts={dict(right_counts.most_common(3))} "
            f"at_s={round(old[0], 4)}->{round(transition['at_s'], 4)} "
            f"fill_ms={round(old[1] * 1000.0, 3)}->{round(transition['fill_s'] * 1000.0, 3)} "
            f"placement={(transition.get('blind_span') or {}).get('placement')} "
            f"for {candidate_path}\n")
    return transitions


def edge_zones(transitions, pairs, domain):
    """Return the master bound (ms) and majority picture lag of the head and tail zones."""
    import zone_picture_lag
    frame_ms = float(domain["frame_ms"])
    end_ms = float(domain["master_timeline_ms"])
    head_end = min((t["at_s"] * 1000.0 for t in transitions), default=None)
    tail_start = max(((t["at_s"] + t["fill_s"]) * 1000.0 for t in transitions), default=None)
    zones = {}
    for kind, low, high, bound in (("head", 0.0, end_ms if head_end is None else head_end, head_end),
                                   ("tail", tail_start or 0.0, end_ms, tail_start)):
        lag = zone_picture_lag.zone_majority(pairs, low, high, frame_ms)[0] if pairs else None
        zones[kind] = {"bound_ms": bound, "lag": lag}
    return zones


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


def log_transitions_summary(transitions, candidate_path):
    """Log one unconditional line with every transition as `change_point:decision:at_s:fill_ms:step_ms`."""
    tools.log_always(
        f"repair: audio_transitions n={len(transitions)} transitions="
        + ",".join(f"{t['change_point']}:{t['decision']}:{t['at_s']}:"
                   f"{round(t['fill_s'] * 1000.0, 3)}:{round(t['b_ms'] - t['a_ms'], 3)}"
                   for t in transitions)
        + f" for {candidate_path}\n")


def log_holes_against_walk(holes, walk, candidate_path):
    """Log interior holes no change point reaches (no repair) and change points b2 did not see."""
    import audio_walk
    regions = [(p["level_before"]["t_last"] * 1000.0,
                (p["level_after"]["t_first"] + audio_walk.WALK_WINDOW_S) * 1000.0, p)
               for p in walk["points"] if p["kind"] == "change_point"]
    reach = _orch().HOLE_MERGE_WINDOW_SECONDS * 1000.0
    seen = set()
    for index, hole in enumerate(holes):
        if hole["kind"] != "interior":
            continue
        hits = [id(p) for low, high, p in regions
                if low < hole["master_ms"][1] + reach and high > hole["master_ms"][0] - reach]
        seen.update(hits)
        if hits:
            continue
        tools.log_always(
            f"repair: picture_only hole={index} master_ms=[{round(hole['master_ms'][0], 2)}, "
            f"{round(hole['master_ms'][1], 2)}] b2_step_ms="
            f"{None if hole['step_ms'] is None else round(hole['step_ms'], 3)} "
            f"rule=audio_continuous_no_repair for {candidate_path}\n")
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
# Dark frames (std under frame_compare.FLAT_STD) are the same picture when their contrast ratio
# stays within 1.5 (measured: a matched fade-in frame pair 1.55 / 1.80 = 1.16; black 0.15
# against a first fade frame 1.8 = 12). Under PIXEL_NOISE_STD on both sides a frame is flat
# noise (black) and the ratio means nothing.
PIXEL_DARK_STD_RATIO = 1.5
PIXEL_NOISE_STD = 1.0
# A candidate fade of the master's own shot keeps its structure: centred correlation from
# 0.95 (measured: a shared shot fading out over 23 frames reads 1.00).
PIXEL_FADE_CORR = 0.95


def _grey_frames(comparer, path, first, count, rate):
    """`count` grey 160x90 frames of `path` from decoded frame `first` on (float array)."""
    import numpy as np
    import frame_compare
    # Labelled by each frame's own pts, as the anchor search's own extraction is.
    _base, frames = frame_compare.grid_frames(comparer, path, max(0, first), count)
    if not len(frames):
        return np.zeros((0, comparer.height, comparer.width))
    return np.asarray(frames, dtype=np.float64).mean(axis=-1)[:count]


def _pixel_reading(master_frame, candidate_frame):
    """(mean level difference, fraction of pixels departing from it by > PIXEL_DIFF_LEVELS,
    master std, master level, candidate std, master std and candidate std over the pixels that
    do not depart, centred correlation) of one grey frame pair; the kept-pixel stds leave out
    a logo or a burnt-in title."""
    import numpy as np
    diff = master_frame - candidate_frame
    level = float(diff.mean())
    departing = np.abs(diff - level) > PIXEL_DIFF_LEVELS
    kept = ~departing
    if kept.any():
        master_kept, candidate_kept = float(master_frame[kept].std()), float(
            candidate_frame[kept].std())
    else:
        master_kept, candidate_kept = float(master_frame.std()), float(candidate_frame.std())
    master_std, candidate_std = float(master_frame.std()), float(candidate_frame.std())
    centred = 0.0
    if master_std > 0.0 and candidate_std > 0.0:
        centred = float(((master_frame - master_frame.mean())
                         * (candidate_frame - candidate_frame.mean())).mean()
                        / (master_std * candidate_std))
    return (level, float(departing.mean()), master_std, float(master_frame.mean()),
            candidate_std, master_kept, candidate_kept, centred)
