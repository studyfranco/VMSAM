# -*- coding: utf-8 -*-
"""Stage 4 of the repair pipeline: plan application.

`plan_geometry` lays the resolved transitions and edges on the master timeline as candidate
zones and master fills (trims, master fill, tail gap); `track_offsets` measures each track's own
offset in every zone; `track_pieces` cuts each track into pieces; `apply_plan` builds and
verifies the repaired file through `merge_video_repair.build_repaired_video_object`, which also
publishes the delivered plan.

Step logging and shared helpers live in `repair_orchestrator`; they are read at call time.
"""
from decimal import Decimal
from fractions import Fraction
from os import path
import statistics
import time

import repair_log
import tools


def _orch():
    import repair_orchestrator
    return repair_orchestrator


def plan_geometry(transitions, head_end_s, tail_start_s, domain, walk):
    """Lay the plan on the master timeline in milliseconds.

    A head fill [0, head_end), then per transition at T with fill w: the zone before ends at T,
    a master fill [T, T + w) when w > 0, the next zone starts at T + w; then a tail fill to the
    timeline's end. Each zone's offset is the median of the walk's windows wholly inside it,
    or its level when none is.

    Returns:
        (zones, fills, None), or (None, None, reason) when transitions overlap.
    """
    import audio_walk
    timeline_ms = _orch()._decimal(domain["master_timeline_ms"])
    zones, fills = [], []
    cursor = Decimal(0)
    if head_end_s is not None:
        cursor = min(Decimal(str(head_end_s)) * 1000, timeline_ms)
        if cursor > 0:
            # A head trimmed to nothing (the candidate's own first frame already lines up with
            # the master's) is not a fill -- an empty [0, 0) piece is what
            # `merge_video_chimeric`'s own shape check (`plan_piece_empty_or_inverted`) exists to
            # catch, so it is never handed down in the first place, the same way the tail fill
            # below is already skipped when it would be empty.
            fills.append({"master_start_ms": Decimal(0), "master_end_ms": cursor,
                          "reason": _orch().WHY_TOKEN["head"], "hole": "head", "status": "audio_edge"})
    fallback = [walk["levels"][0]["off_ms"]] + [t["b_ms"] for t in transitions]
    boundaries = []
    for number, transition in enumerate(transitions):
        at = Decimal(str(transition["at_s"])) * 1000
        width = Decimal(str(transition["fill_s"])) * 1000
        if at < cursor:
            return None, None, (f"transition {number} at {at} ms lies before the previous "
                                f"piece's end {cursor} ms")
        boundaries.append((cursor, at, fallback[number]))
        if width > 0:
            fills.append({"master_start_ms": at, "master_end_ms": at + width,
                          "reason": _orch().WHY_TOKEN["interior"], "hole": number,
                          "status": transition["decision"],
                          "cut_ms": transition.get("cut_ms")})
        cursor = at + width
    end = timeline_ms
    if tail_start_s is not None:
        end = min(Decimal(str(tail_start_s)) * 1000, timeline_ms)
        if end < cursor:
            return None, None, f"the tail edge {end} ms lies before the last piece's end {cursor} ms"
    boundaries.append((cursor, end, fallback[len(transitions)]))
    if tail_start_s is not None and end < timeline_ms:
        fills.append({"master_start_ms": end, "master_end_ms": timeline_ms,
                      "reason": _orch().WHY_TOKEN["tail"], "hole": "tail", "status": "audio_edge"})
    for start, stop, level in boundaries:
        if stop <= start:
            continue
        inside = [row["off"] for row in walk["rows"]
                  if row["status"] == "ok" and row["t"] * 1000 >= float(start)
                  and (row["t"] + audio_walk.WALK_WINDOW_S) * 1000 <= float(stop)]
        offset = (Decimal(str(round(float(statistics.median(inside)), 3))) if inside
                  else Decimal(str(level)))
        zones.append({"master_start_ms": start, "master_end_ms": stop, "offset_ms": offset,
                      "n_windows": len(inside), "zone": len(zones)})
    fills.sort(key=lambda fill: fill["master_start_ms"])
    return zones, fills, None


def zone_picture_offsets_from_hole_anchors(zones, transitions, domain, lag_pairs=None,
                                           start_delta_ms=0, candidate_path=""):
    """Each zone's picture offset in ms on the files' own clocks.

    A zone's lag is the majority picture lag over the scene changes inside it
    (`zone_picture_lag.zone_majority` over `lag_pairs`, both whole files' matched changes),
    used when it agrees within one frame with a bounding anchor; otherwise, and when the zone
    holds too few changes, the anchors' reading
    (`zone_anchor_lags`). Two lags each holding more than 15 % of a zone's changes are logged
    (`picture_lag_jitter`); the zone still takes the majority.

    The lag is a count of decoded frames; the offset adds the candidate's video start minus
    the master's (`start_delta_ms`), since both files' first frames sit at their own start
    times.

    Returns:
        {zone_index: {"picture_ms": Fraction, "cuts": 1, 2 or "scene_changes:<n>"}}.
    """
    import zone_picture_lag
    frame_ms = domain["frame_ms"]
    delta = Fraction(start_delta_ms)
    anchors = zone_anchor_lags(zones, transitions)
    offsets = {}
    for zone in zones:
        index = zone["zone"]
        anchor = anchors.get(index)
        lag, counts = zone_picture_lag.zone_majority(
            lag_pairs, zone["master_start_ms"], zone["master_end_ms"], frame_ms)
        spread = zone_picture_lag.jitter(counts)
        if spread is not None:
            tools.log_always(
                f"repair: picture_lag_jitter zone={index} counts="
                f"{dict(counts.most_common(4))} majority={lag} for {candidate_path}\n")
        bounds = [] if anchor is None else anchor["bounds"]
        # A zone no cut bounds keeps its audio offset, as before: only the anchors' one-shot
        # reading is replaced by the zone's own majority.
        if lag is not None and any(abs(lag - b) <= 1 for b in bounds):
            offsets[index] = {"picture_ms": Fraction(lag) * frame_ms + delta,
                              "cuts": f"scene_changes:{counts[lag]}"}
        elif anchor is not None and anchor["lag"] is not None:
            offsets[index] = {"picture_ms": Fraction(anchor["lag"]) * frame_ms + delta,
                              "cuts": anchor["cuts"]}
    return offsets


def zone_anchor_lags(zones, transitions):
    """Read each zone's picture lag (frames) off the pHash anchors already measured resolving
    its bounding change point(s), instead of a separate scan.

    `zones[k]` (`plan_geometry`'s own numbering) sits between `transitions[k-1]` (its left
    bound, when `k > 0`) and `transitions[k]` (its right bound, when `k < len(transitions)`).
    `audio_transitions` already measured, at any change point its anchors and pHash cross-sweep
    resolved, the frame shift between master and candidate on each side of it
    (`before_shift_frames`, `after_shift_frames`) -- the zone right after such a change point
    carries that change point's own `after_shift_frames`, and the zone right before it carries
    its `before_shift_frames`. Both sides are read when both bounding change points resolved
    one, and must agree to within one frame (a disagreement is itself not a measured offset --
    every change point around a zone is independent, so an inconsistency here means the zone's
    offset is not actually one picture-anchored constant, which `audio_transitions` would need
    a species of its own to express; until then this zone keeps the audio offset it already has).

    Returns:
        {zone_index: {"lag": frames, "cuts": 1 or 2, "bounds": [anchor lags]}} for every zone
        at least one bounding change point resolved a frame shift for (`bounds` also lists the
        readings of a zone whose two anchors disagree, which then has no "lag").
    """
    lags = {}
    for index, zone in enumerate(zones):
        left = transitions[index - 1].get("after_shift_frames") if index > 0 else None
        right = (transitions[index].get("before_shift_frames") if index < len(transitions)
                else None)
        bounds = [value for value in (left, right) if value is not None]
        if left is not None and right is not None:
            if abs(left - right) > 1:
                lags[zone["zone"]] = {"lag": None, "cuts": 2, "bounds": bounds}
                continue
            lags[zone["zone"]] = {"lag": left, "cuts": 2, "bounds": bounds}
        elif bounds:
            lags[zone["zone"]] = {"lag": bounds[0], "cuts": 1, "bounds": bounds}
    return lags


def written_edge_seconds(fills, master_audio_end_s):
    """Return (head_s, tail_s) of edge fill the master actually writes (tail capped at its audio end)."""
    end = Decimal(str(master_audio_end_s)) * 1000
    head = sum((f["master_end_ms"] - f["master_start_ms"]) for f in fills
               if f["reason"] == _orch().WHY_TOKEN["head"])
    tail = sum(max(Decimal(0), min(f["master_end_ms"], end) - f["master_start_ms"])
               for f in fills if f["reason"] == _orch().WHY_TOKEN["tail"])
    return float(head) / 1000.0, float(tail) / 1000.0


# Margin trimmed from each side of a zone before measuring a track's offset in it.
ZONE_EDGE_MARGIN_S = 0.5


def remeasure_at_other_levels(zones, readings, measure, searched_ms):
    """Re-measure unexplained zones at the track's own levels, then the reference's other levels.

    Without this, a track that does not take the reference's steps would be given them.
    `measure(low, high, seed)` wraps `audio_walk.zone_offset`; seeds within `searched_ms` of the
    zone's reference offset are skipped. The reading with the most windows is kept.

    Returns:
        The retries, for the log.
    """
    import audio_walk
    own = [float(r["offset_ms"]) for r in readings if r["offset_ms"] is not None]
    other = [float(zone["offset_ms"]) for zone in zones]
    retries = []
    for zone, reading in zip(zones, readings):
        if reading["offset_ms"] is not None or not own:
            continue
        low = float(zone["master_start_ms"]) / 1000.0 + ZONE_EDGE_MARGIN_S
        high = float(zone["master_end_ms"]) / 1000.0 - ZONE_EDGE_MARGIN_S
        if high - low < audio_walk.WALK_WINDOW_S:
            continue
        reference_ms = float(zone["offset_ms"])
        tried, best = [], None
        for seed in own + other:
            if abs(seed - reference_ms) <= searched_ms or any(abs(seed - s) < 1.0 for s in tried):
                continue
            tried.append(seed)
            measured = measure(low, high, seed)
            if measured["offset_ms"] is not None and (best is None
                                                      or measured["n_ok"] > best[1]["n_ok"]):
                best = (seed, measured)
        if best is None:
            continue
        seed, measured = best
        reading.update(offset_ms=Decimal(str(measured["offset_ms"])), windows=measured["n_ok"],
                       seed_ms=seed, reason=None)
        retries.append({"zone": zone["zone"], "reference_ms": reference_ms, "seed_ms": seed,
                        "offset_ms": measured["offset_ms"], "windows": measured["n_ok"]})
    return retries


def track_offsets(zones, walk, master_obj, candidate_obj, language, speed_ratio, scale,
                  deadline=None, engine="asetrate"):
    """Measure each candidate track's own offset per zone, at the millisecond.

    The reference track takes the walk's zone offsets. Other tracks are measured against the
    master track of their language (`audio_walk.zone_offset`, then `remeasure_at_other_levels`).
    Unmeasurable zones are derived from the nearest measured zone plus the reference's step,
    else inherited from the reference; every substitution is logged.

    Returns:
        (tracks, None) or (None, reason); tracks[stream_order] has `language`, `zones`,
        `start_ms`, `extent_ms`, `extent_source`, `measured`, `reference`.
    """
    import audio_walk
    import merge_video_chimeric
    reference_order = int(walk["candidate_stream"])
    audios = list(merge_video_chimeric.iterate_candidate_audios(candidate_obj))
    master_cache = {int(walk["master_stream"]): walk["master"]}
    tracks = {}
    for track_language, audio in audios:
        order = int(audio["StreamOrder"])
        start_ms, extent_ms, extent_source = _track_timing(candidate_obj, audio, scale)
        entry = {"language": track_language, "start_ms": start_ms, "extent_ms": extent_ms,
                 "extent_source": extent_source, "zones": [], "measured": False,
                 "reference": None}
        tracks[order] = entry
        if order == reference_order:
            entry["reference"] = int(walk["master_stream"])
            entry["zones"] = [{"zone": zone["zone"], "offset_ms": zone["offset_ms"],
                               "coarse_offset_ms": str(zone["offset_ms"]),
                               "source": "walk_level", "reason": None,
                               "windows": zone["n_windows"]} for zone in zones]
            entry["measured"] = True
            continue
        master_audio = merge_video_chimeric.find_master_audio_for_language(
            master_obj, track_language,
            walk["master_stream"] if track_language == language else None)
        _orch().step_launch("track_offset", candidate=candidate_obj.filePath, stream=order,
                    language=track_language,
                    master_stream=None if master_audio is None else master_audio.get("StreamOrder"))
        readings = [{"zone": zone["zone"], "offset_ms": None,
                     "coarse_offset_ms": str(zone["offset_ms"]), "reason": None, "windows": 0}
                    for zone in zones]
        entry["zones"] = readings
        if master_audio is None:
            entry["own_reason"] = f"master_carries_no_{track_language}_track"
            for reading in readings:
                reading["reason"] = entry["own_reason"]
            _orch().step_result("track_offset", candidate=candidate_obj.filePath, stream=order,
                        measured=False, reason=entry["own_reason"])
            continue
        master_order = int(master_audio["StreamOrder"])
        entry["reference"] = master_order
        try:
            envelope = engine == "atempo" and speed_ratio is not None
            if master_order not in master_cache:
                master_cache[master_order] = audio_walk.read_on_file_clock(
                    master_obj, master_audio, deadline=deadline, envelope=envelope)
            samples = audio_walk.read_on_file_clock(
                candidate_obj, audio, _orch()._speed_chain(audio, speed_ratio, engine), scale,
                deadline=deadline, envelope=envelope)
        except Exception as error:                                       # noqa: BLE001
            if getattr(error, "cause", None) == "repair_budget_exceeded":
                raise
            entry["own_reason"] = f"track_unreadable({type(error).__name__})"
            for reading in readings:
                reading["reason"] = entry["own_reason"]
            _orch().step_result("track_offset", candidate=candidate_obj.filePath, stream=order,
                        measured=False, reason=entry["own_reason"], evidence=str(error)[:200])
            continue
        for zone, reading in zip(zones, readings):
            low = float(zone["master_start_ms"]) / 1000.0 + ZONE_EDGE_MARGIN_S
            high = float(zone["master_end_ms"]) / 1000.0 - ZONE_EDGE_MARGIN_S
            if high - low < audio_walk.WALK_WINDOW_S:
                reading["reason"] = f"zone_too_short({round(high - low + 2 * ZONE_EDGE_MARGIN_S, 3)}s)"
                continue
            measured = audio_walk.zone_offset(master_cache[master_order], samples, low, high,
                                              float(zone["offset_ms"]))
            reading["windows"] = measured["n_ok"]
            if measured["offset_ms"] is None:
                reading["reason"] = f"no_window_measured({measured['counts']})"
                continue
            reading["offset_ms"] = Decimal(str(measured["offset_ms"]))
            if len(measured["levels"]) > 1:
                tools.dev_log(f"orchestrator: track {order} ({track_language}) changes inside zone "
                              f"{zone['zone']}: levels "
                              f"{[(lv['t_first'], lv['t_last'], lv['off_ms']) for lv in measured['levels']]}"
                              f" -- the dominant one is applied\n")
        for retry in remeasure_at_other_levels(
                zones, readings,
                lambda low, high, seed: audio_walk.zone_offset(
                    master_cache[master_order], samples, low, high, seed),
                audio_walk.WALK_SEARCH_MS):
            tools.log_line(f"repair: offset_remeasured stream={order} lang={track_language} "
                           f"zone={retry['zone']} reference_ms={retry['reference_ms']} "
                           f"seed_ms={retry['seed_ms']} measured_ms={retry['offset_ms']} "
                           f"windows={retry['windows']}\n")
        del samples
        entry["measured"] = any(reading["offset_ms"] is not None for reading in readings)
        _orch().step_result("track_offset", candidate=candidate_obj.filePath, stream=order,
                    language=track_language, master_stream=master_order,
                    measured_zones=sum(1 for r in readings if r["offset_ms"] is not None),
                    n_zones=len(zones),
                    offsets_ms=[None if r["offset_ms"] is None else float(r["offset_ms"])
                                for r in readings],
                    reasons=[r["reason"] for r in readings])
    del master_cache
    if reference_order not in tracks:
        return None, (f"the comparison track (stream {reference_order}) is not among the "
                      f"candidate's audio tracks")
    reference = tracks[reference_order]["zones"]
    for order, entry in tracks.items():
        for reading in entry["zones"]:
            if reading["offset_ms"] is not None:
                reading.setdefault("source", "measured")
    for order, entry in tracks.items():
        if order == reference_order:
            continue
        measured = [r for r in entry["zones"] if r.get("source") == "measured"]
        for reading in entry["zones"]:
            if reading["offset_ms"] is not None:
                continue
            if measured:
                nearest = min(measured, key=lambda r: abs(r["zone"] - reading["zone"]))
                reading["offset_ms"] = (nearest["offset_ms"]
                                        + reference[reading["zone"]]["offset_ms"]
                                        - reference[nearest["zone"]]["offset_ms"])
                reading["source"] = f"derived_by_reference_step(zone_{nearest['zone']})"
            else:
                reading["offset_ms"] = reference[reading["zone"]]["offset_ms"]
                reading["source"] = f"inherited(stream_{reference_order})"
            tools.log_line(
                f"repair: offset_substitution stream={order} lang={entry['language']} "
                f"zone={reading['zone']} own=unmeasured({reading['reason']}) "
                f"reference_ms={reference[reading['zone']]['offset_ms']} "
                f"applied_ms={reading['offset_ms']} source={reading['source']}\n")
    return tracks, None


def _track_timing(video_obj, audio, scale):
    """Return (start_ms, extent_ms, source) for a track on the plan's timeline.

    The end is read from packets, since a declared Duration can under-report by ~120 ms; the
    declared value is only a fallback.
    """
    import merge_video_chimeric
    start_ms = merge_video_chimeric.get_stream_start_ms(audio) * scale
    extent_ms, reason = merge_video_chimeric.measure_track_extent_ms(
        video_obj.filePath, int(audio["StreamOrder"]))
    source = f"packets({reason})"
    if extent_ms is None:
        declared = merge_video_chimeric.get_track_audio_length_ms(audio)
        if declared is not None:
            extent_ms = declared + merge_video_chimeric.get_stream_start_ms(audio)
            source = f"declared_duration(packets {reason})"
    if extent_ms is not None:
        extent_ms = extent_ms * scale
    return start_ms, extent_ms, source


def track_pieces(zones, fills, readings, extent_ms, timeline_ms):
    """Lay the plan for one track: master fills, and candidate zones at this track's offsets.

    A zone reading before the candidate's time zero, or past the track's real end, is clipped
    and the master fills the difference (logged as adjustments). Overlapping candidate reads at
    a splice are kept and reported.

    Returns:
        (pieces, adjustments, overlaps)
    """
    segments = ([dict(fill, source="master") for fill in fills]
                + [dict(zone, source="candidate") for zone in zones])
    segments.sort(key=lambda segment: segment["master_start_ms"])
    pieces, adjustments = [], []
    for segment in segments:
        start, end = segment["master_start_ms"], segment["master_end_ms"]
        if segment["source"] == "master":
            pieces.append({"source": "master", "master_start_ms": start, "master_end_ms": end,
                           "source_start_ms": start, "reason": segment["reason"]})
            continue
        offset = readings[segment["zone"]]["offset_ms"]
        if start + offset < 0:
            shifted = -offset
            adjustments.append({"zone": segment["zone"], "kind": "head_before_candidate_zero",
                                "master_fill_ms": str(shifted - start)})
            pieces.append({"source": "master", "master_start_ms": start,
                           "master_end_ms": min(shifted, end), "source_start_ms": start,
                           "reason": _orch().WHY_TOKEN["head"] if start == 0 else _orch().WHY_TOKEN["interior"]})
            start = min(shifted, end)
        if extent_ms is not None and end + offset > extent_ms and start < end:
            cut = min(end - start, end + offset - extent_ms)
            adjustments.append({"zone": segment["zone"], "kind": "past_track_end",
                                "master_fill_ms": str(cut)})
            pieces_tail = {"source": "master", "master_start_ms": end - cut,
                           "master_end_ms": end, "source_start_ms": end - cut,
                           "reason": (_orch().WHY_TOKEN["tail"] if end == timeline_ms
                                      else _orch().WHY_TOKEN["interior"])}
            if end - cut > start:
                pieces.append({"source": "candidate", "master_start_ms": start,
                               "master_end_ms": end - cut, "source_start_ms": start + offset,
                               "zone": segment["zone"], "reason": "zone"})
            pieces.append(pieces_tail)
            continue
        if end > start:
            pieces.append({"source": "candidate", "master_start_ms": start, "master_end_ms": end,
                           "source_start_ms": start + offset, "zone": segment["zone"],
                           "reason": "zone"})
    merged = []
    for piece in pieces:
        if (merged and piece["source"] == "master" and merged[-1]["source"] == "master"
                and merged[-1]["master_end_ms"] == piece["master_start_ms"]):
            merged[-1]["master_end_ms"] = piece["master_end_ms"]
            if piece["reason"] == _orch().WHY_TOKEN["tail"]:
                merged[-1]["reason"] = _orch().WHY_TOKEN["tail"]
            continue
        merged.append(piece)
    overlaps = []
    previous = None
    for piece in merged:
        if piece["source"] != "candidate":
            continue
        if previous is not None:
            previous_end = previous["source_start_ms"] + (previous["master_end_ms"]
                                                          - previous["master_start_ms"])
            if piece["source_start_ms"] < previous_end:
                overlaps.append({"zones": [previous["zone"], piece["zone"]],
                                 "reread_ms": str(previous_end - piece["source_start_ms"])})
        previous = piece
    return merged, adjustments, overlaps


def apply_picture_offsets(zones, tracks, zone_picture_offsets_ms, candidate_path,
                          residual_by_zone=None):
    """Override every track's per-zone offset with that zone's own measured picture offset.

    The video is the source of truth: the zone's own picture offset, read off the pHash
    anchors already measured resolving its bounding change point(s)
    (`zone_picture_offsets_from_hole_anchors`), is what every delivered piece is keyed on in
    that zone -- rebuilt audio and retimed subtitles alike, every track alike. A candidate whose
    own audio drifts from its own picture inside a zone is a finding only, logged here, never
    reflected in what is delivered. A zone whose bounding change points carry no video anchor
    keeps the audio-walk offset it already has, logged the same way.

    Mutates `zones` (so later logging of the plan's geometry reads the delivered offset) and
    every track's own zone reading in `tracks`. When `residual_by_zone` is a dict, it receives
    each such zone's signed picture_ms - audio_ms: the lag the delivery verifier is expected to
    measure on that zone's delivered audio.

    Returns:
        The largest |audio_ms - picture_ms| over every zone a picture offset was measured for
        (None when no zone had one) -- the file's own expected audio-vs-picture drift, carried
        forward so the post-build verifier (which independently cross-correlates the delivered
        audio against the master's) does not mistake this same, already-logged finding for a
        geometry defect in the comparison-language track.
    """
    worst_diff_ms = None
    for zone in zones:
        index = zone["zone"]
        audio_ms = float(zone["offset_ms"])
        measured = zone_picture_offsets_ms.get(index)
        if measured is None:
            tools.log_always(
                f"repair: picture_audio_offset_diff zone={index} audio_ms={round(audio_ms, 3)} "
                f"picture_ms=none reason=no_anchors for {candidate_path}\n")
            continue
        picture_ms = measured["picture_ms"]  # exact Fraction -- never rounded through a float
        tools.log_always(
            f"repair: picture_audio_offset_diff zone={index} audio_ms={round(audio_ms, 3)} "
            f"picture_ms={round(float(picture_ms), 3)} cuts={measured['cuts']} "
            f"for {candidate_path}\n")
        # Decimal(numerator)/Decimal(denominator) at the module's default 28-digit precision --
        # exact-rational for every value a frame count at a real-world rate produces, unlike
        # `Decimal(str(round(picture_ms, 3)))`, which would round a repeating fraction (e.g.
        # 1001/24 ms per frame) to the millisecond and discard its own frame-exactness.
        zone["offset_ms"] = Decimal(picture_ms.numerator) / Decimal(picture_ms.denominator)
        for entry in tracks.values():
            reading = entry["zones"][index]
            # Only the delivered value changes -- `source` still records whether this track's
            # own audio was actually read for this zone (the confidence bookkeeping downstream
            # reports keys on), now just no longer what the delivered offset itself is built
            # from.
            reading["offset_ms"] = zone["offset_ms"]
        diff_ms = abs(audio_ms - float(picture_ms))
        if residual_by_zone is not None:
            residual_by_zone[index] = float(picture_ms) - audio_ms
        worst_diff_ms = diff_ms if worst_diff_ms is None else max(worst_diff_ms, diff_ms)
    return worst_diff_ms


@repair_log.timed_phase("orchestrator", "apply_plan", lambda candidate_path, *a, **k: candidate_path)
def apply_plan(candidate_path, plan_spec, speed_factor, master_obj, candidate_obj, context):
    """Apply a resolved plan and build the temporary chimeric file (step 5); decides nothing.

    Steps: log geometry; set the speed ratio; measure per-track offsets (`track_offsets`);
    lay pieces per track (`track_pieces`); re-time chapters; build and verify through
    `merge_video_repair.build_repaired_video_object`; probe delivered durations; store the
    result on the repair seam and record `repaired`. Assembly refusals (`chimeric_error`)
    propagate with their own cause token.

    Args:
        plan_spec: `zones` and `fills` from `plan_geometry`, the reference `walk`, and
            `head_written_s` / `tail_written_s`.
        context: comparison language, reference streams and quantum, frame domain, rate gate
            and chimeric-tag decision.

    Returns:
        (ok, cause, reason); ok is True only when the chimeric file exists.
    """
    import merge_video_chimeric
    import merge_video_repair
    started = time.time()
    domain = context["domain"]
    language = context["language"]
    work_dir = path.join(context["work_dir"], "apply_plan")
    tools.make_dirs(work_dir)
    timeline_ms = _orch()._decimal(domain["master_timeline_ms"])
    rate_text = (f"{speed_factor.numerator}/{speed_factor.denominator}"
                 if isinstance(speed_factor, Fraction) else speed_factor)
    zones, fills = plan_spec["zones"], plan_spec["fills"]
    _orch().step_launch("apply_plan", candidate=candidate_path, n_zones=len(zones), n_fills=len(fills),
                speed_factor=rate_text)

    # ---- 1. geometry --------------------------------------------------------
    head_added = Decimal(str(round(plan_spec["head_written_s"] * 1000.0, 3)))
    tail_added = Decimal(str(round(plan_spec["tail_written_s"] * 1000.0, 3)))
    interior_filled = sum((fill["master_end_ms"] - fill["master_start_ms"]) for fill in fills
                          if fill["reason"] == _orch().WHY_TOKEN["interior"])
    _orch().step_result("plan_geometry", candidate=candidate_path,
                zones=[[float(z["master_start_ms"]), float(z["master_end_ms"])] for z in zones],
                offsets_ms=[float(z["offset_ms"]) for z in zones],
                zone_windows=[z["n_windows"] for z in zones],
                fills=[[float(f["master_start_ms"]), float(f["master_end_ms"]), f["reason"],
                        f["status"]] for f in fills])
    tools.dev_log(f"orchestrator: edge_additions head_ms={head_added} tail_ms={tail_added} "
                  f"interior_filled_ms={interior_filled} for {candidate_path}\n")
    if not zones:
        _orch().step_result("apply_plan", candidate=candidate_path, ok=False,
                    cause="plan_reads_no_candidate_content")
        return False, "plan_reads_no_candidate_content", (
            "the resolved holes leave no candidate content on the master timeline -- a plan "
            "that reads nothing from the candidate is the master, not a repair")

    # ---- 2. speed -----------------------------------------------------------
    speed_ratio = None
    scale = Decimal(1)
    if speed_factor is not None and speed_factor != 1:
        speed_ratio = _orch()._decimal(Fraction(speed_factor))
        scale = speed_ratio
    _orch().step_result("delivery_speed", candidate=candidate_path, speed_factor=rate_text,
                speed_ratio=(None if speed_ratio is None else str(speed_ratio)),
                filter=(None if speed_ratio is None
                        else (context.get("resample_routing") or {}).get("filter_name",
                                                                         "asetrate")),
                rule=("ADDENDUM_6_no_filter_without_speed_change" if speed_ratio is None
                      else "ADDENDUM_8_resample_is_restoration"))

    # ---- 3. per-track sub-frame offsets --------------------------------------
    engine = (context.get("resample_routing") or {}).get("filter_name", "asetrate")
    tracks, offset_failure = track_offsets(zones, plan_spec["walk"], master_obj, candidate_obj,
                                           language, speed_ratio, scale,
                                           deadline=domain.get("repair_deadline"), engine=engine)
    if tracks is None:
        _orch().step_result("apply_plan", candidate=candidate_path, ok=False,
                    cause="plan_offset_unmeasurable")
        return False, "plan_offset_unmeasurable", offset_failure
    picture_audio_residual_by_zone = {}
    picture_audio_worst_diff_ms = apply_picture_offsets(
        zones, tracks, plan_spec.get("zone_picture_offsets_ms") or {}, candidate_path,
        residual_by_zone=picture_audio_residual_by_zone)

    # ---- 4. pieces per track -------------------------------------------------
    track_plans = {}
    reference_pieces = None
    for order, entry in tracks.items():
        pieces, adjustments, overlaps = track_pieces(zones, fills, entry["zones"],
                                                     entry["extent_ms"], timeline_ms)
        sources = sorted({reading["source"] for reading in entry["zones"]})
        own = set(sources) <= {"measured", "walk_level"}
        track_plans[order] = {
            "pieces": pieces,
            "extent_ms": entry["extent_ms"], "extent_source": entry["extent_source"],
            "offset_measured": own,
            "borrow_reason": (None if own
                              else ",".join(sources) + (f"[{entry['own_reason']}]"
                                                        if entry.get("own_reason") else "")),
            "offset_sources": [{"zone": r["zone"], "offset_ms": str(r["offset_ms"]),
                                "source": r["source"]} for r in entry["zones"]]}
        for adjustment in adjustments:
            tools.log_line(f"repair: plan_edge_adjustment stream={order} "
                              f"zone={adjustment['zone']} kind={adjustment['kind']} "
                              f"master_fill_ms={adjustment['master_fill_ms']}\n")
        for overlap in overlaps:
            tools.log_line(f"repair: splice_reread stream={order} zones={overlap['zones']} "
                              f"reread_ms={overlap['reread_ms']} (the audio edit and the video "
                              f"cut differ by a fraction of a frame; each zone is read at its "
                              f"own measured offset)\n")
        if str(order) == str(context["candidate_stream"]):
            reference_pieces = pieces
        _orch().step_result("track_pieces", candidate=candidate_path, stream=order,
                    n_pieces=len(pieces), sources=sources,
                    pieces=[(p["source"][0], float(p["master_start_ms"]),
                             float(p["master_end_ms"]),
                             float(p["source_start_ms"])) for p in pieces])

    # ---- 5. chapters ---------------------------------------------------------
    _orch().step_launch("chapters", candidate=candidate_path)
    chapters_path, chapter_decisions = merge_video_chimeric.build_delivered_chapters(
        master_obj.filePath, candidate_obj.filePath, reference_pieces,
        speed_ratio, timeline_ms, work_dir)
    for decision in chapter_decisions:
        tools.log_line("repair: chapter " + " ".join(
            f"{key}={str(value).replace(' ', '_')}" for key, value in decision.items())
            + "\n")
    _orch().step_result("chapters", candidate=candidate_path, delivered=chapters_path is not None,
                n_decisions=len(chapter_decisions))

    # ---- 6. build ------------------------------------------------------------
    seam = getattr(candidate_obj, merge_video_repair.REPAIR_SEAM_ATTRIBUTE, None)
    job_start_utc = (seam or {}).get("job_start_utc")
    if job_start_utc is None:
        # Standalone run (no repair seam): no job start to stamp.
        job_start_utc = "unstamped(no_repair_seam_standalone_run)"
        tools.dev_log(f"orchestrator: no {merge_video_repair.REPAIR_SEAM_ATTRIBUTE} on "
                      f"{candidate_path}: standalone run, the era tag carries no job start\n")
    # The route label only: each track's tags are decided where its plan is applied
    # (`merge_video_chimeric.attach_track_tags`).
    marker = "orchestrator"
    for fill in fills:
        if fill.get("status") == "replacement_hole":
            marker = "+".join(part for part in (
                marker, f"replacement_hole:{fill['cut_ms']}/"
                        f"{round(float(fill['master_end_ms'] - fill['master_start_ms']), 3)}") if part)
    comparison_offsets = tracks[int(context["candidate_stream"])]["zones"]
    plan = {
        "kind": "orchestrator_chimeric",
        "language": language, "reference_stream": context["master_stream"],
        "reference_candidate_stream": int(context["candidate_stream"]),
        "quantum_ms": context["quantum_ms"], "master_path": master_obj.filePath,
        "decided_by": "repair_orchestrator.apply_plan",
        "segments_dropped_unusable": 0,
        "speed_margin": (context.get("sweep_gate") or {}).get("margin"),
        "speed_engine": (None if speed_ratio is None
                         else (context.get("resample_routing") or {}).get("filter_name")),
        "speed_margin_absent_reason": ("no_rate_relation" if speed_ratio is None else None),
        "segments": [{"master_start_ms": zone["master_start_ms"],
                      "master_end_ms": zone["master_end_ms"],
                      "candidate_offset_ms": comparison_offsets[zone["zone"]]["offset_ms"],
                      "candidate_offset_ms_by_stream": {
                          order: str(entry["zones"][zone["zone"]]["offset_ms"])
                          for order, entry in tracks.items()}}
                     for zone in zones],
        "track_plans": track_plans, "reference_pieces": reference_pieces,
        # Already strictly decoded at the prime; the build's corrupt-track gate skips it.
        "strictly_decoded_streams": [int(context["candidate_stream"])],
        "marker": marker, "chapters_path": chapters_path,
        "speed_ratio": speed_ratio,
        "speed_ratio_exact": (None if speed_ratio is None else rate_text),
        "rate_source": (None if speed_ratio is None else "rate_arm"),
        "resample_gate": context.get("sweep_gate"),
        "repair_deadline": domain.get("repair_deadline"),
        "picture_audio_worst_diff_ms": picture_audio_worst_diff_ms,
        "picture_audio_residual_by_zone": picture_audio_residual_by_zone,
    }
    _orch().step_launch("build", candidate=candidate_path, marker=marker,
                n_tracks=len(track_plans))
    try:
        repaired_obj, assembly = merge_video_repair.build_repaired_video_object(
            candidate_obj, master_obj, plan, path.join(tools.tmpFolder, "repair"),
            job_start_utc)
    except merge_video_chimeric.chimeric_error as error:
        cause = getattr(error, "cause", None) or "chimeric_build_refused"
        _orch().step_result("build", candidate=candidate_path, ok=False,
                    cause=cause, error=str(error)[:300],
                    seconds=round(time.time() - started, 1))
        # Every other branch of this plan returns a (False, cause, reason) decline; a build
        # refusal does the same instead of raising out of apply_plan -- a single unbuildable
        # track (or an unrecoverable plan) is a decline, never a crash of the whole repair.
        return False, cause, str(error)
    out_path = getattr(repaired_obj, "filePath", None)
    exists = bool(out_path) and path.exists(out_path)
    _orch().step_result("build", candidate=candidate_path, ok=exists, out_path=out_path,
                marker=assembly.get("marker"),
                track_markers=[r.get("marker") for r in assembly.get("audios") or []],
                verification=[(v.get("track"), v.get("outcome"), v.get("worst_lag_ms"))
                              for v in assembly.get("verification") or []],
                fabricated_dropped=len(assembly.get("fabricated_dropped") or []))
    if not exists:
        return False, "plan_application_no_file", (
            f"the build returned but the temporary chimeric file is not on disk ({out_path}) -- "
            f"no file, no repair")
    # A file that delivers nothing is still returned: the merge runs and keep_best_audio
    # decides (the master wins). `nothing_to_deliver` is informational.
    delivered_tracks = [
        (holder, language_, audio.get("StreamOrder"))
        for holder in ("audios", "commentary", "audiodesc", "subtitles")
        for language_, entries in (getattr(repaired_obj, holder, None) or {}).items()
        for audio in entries if audio.get("keep", True) is not False]
    if not delivered_tracks:
        gate = [(d.get("stream_order"), d.get("cause")) if isinstance(d, dict) else d
                for d in (assembly.get("fabricated_dropped") or [])]
        tools.log_always(f"repair: nothing_to_deliver informational=1 out_path={out_path} "
                         f"gate_dropped={gate} -- no rebuilt track passes the delivery gate; the "
                         f"merge runs and the master wins for {candidate_path}\n")

    # ---- 7. DELIVERED_DURATIONS ---------------------------------------------
    delivered = merge_video_chimeric.probe_delivered_durations(out_path)
    tools.log_line(
        f"repair: DELIVERED_DURATIONS container_ms={delivered['container_ms']} "
        f"master_video_ms={timeline_ms} video_ms=absent(the_chimeric_file_carries_no_video) "
        + " ".join(f"{stream['type']}_{stream['index']}_ms={stream['duration_ms']}"
                   for stream in delivered["streams"])
        + f" max_cue_end_ms={delivered['max_cue_end_ms']} for {candidate_path}\n")

    # ---- 8. the seam and the terminal ----------------------------------------
    if seam is not None:
        seam["repaired_obj"] = repaired_obj
        seam["assembly"] = assembly
    summary = {
        "out_path": out_path,
        "zones": [[str(z["master_start_ms"]), str(z["master_end_ms"])] for z in zones],
        "fills": [[str(f["master_start_ms"]), str(f["master_end_ms"]), f["reason"]]
                  for f in fills],
        "offsets_ms": {order: plan_["offset_sources"] for order, plan_ in track_plans.items()},
        "pieces": {order: len(plan_["pieces"]) for order, plan_ in track_plans.items()},
        "markers": {r["stream_order"]: r.get("marker") for r in assembly.get("audios") or []},
        "edge_additions_ms": {"head": str(head_added), "tail": str(tail_added),
                              "interior_filled": str(interior_filled)},
        "speed_ratio": plan["speed_ratio_exact"],
        "chapters": chapters_path is not None,
        "delivered_durations": delivered,
        "fabricated_dropped": assembly.get("fabricated_dropped"),
    }
    reason = (f"plan applied: {len(zones)} candidate zone(s), {len(fills)} master fill(s) "
              f"(head {head_added} ms, interior {interior_filled} ms, tail {tail_added} ms), "
              f"{len(assembly.get('audios') or [])} audio and "
              f"{len(assembly.get('subtitles') or [])} subtitle track(s) rebuilt, marker "
              f"'{assembly.get('marker')}', speed {rate_text}, "
              f"{len(assembly.get('fabricated_dropped') or [])} fabricated track(s) dropped by the "
              f"delivery gate, temporary chimeric file {out_path}")
    merge_video_repair.record(candidate_path, "repaired", reason, detail=summary)
    _orch().step_result("apply_plan", candidate=candidate_path, ok=True, out_path=out_path,
                seconds=round(time.time() - started, 1))
    return True, None, reason
