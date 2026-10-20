"""Picture-only merge for a pair sharing no common audio language (owner order 2026-10-19).

Chains the owner's 5 steps on top of functions that already exist and are already proven on the
audio-anchored routes -- this module adds no new measurement, only the wiring:

  1. no common audio -> compare by picture (`video.get_best_quality_video` picks the master;
     `video_offset_plan.probe_video` + `measure_video_offset` / `detect_speed_ratio` do the rest).
  2. every scene change, aligned by pHash -- entirely inside `measure_video_offset`
     /`detect_speed_ratio`, which reach pHash and geometry only through `frame_compare.py`
     (Phase A of this same change routed `video_offset_plan` there).
  3. faster frames -> resample (`detect_speed_ratio`, a named ratio); holes -> anchors and
     inserted zones, missing head/tail added (`group_video_zones`, `check_zone_compatibility`,
     `video_zone_plan`, `video_offset_plan.video_anchored_pieces`).
  4. apply the plan (`merge_video_repair.build_repaired_video_object`).
  5. add the candidate's modified tracks -- audio included, subtitles deduped against the
     master -- to the master's MD5 group (`winner.sameAudioMD5UseForCalculation`).

Dropped on purpose, against `merge_video_visual_fallback.py` (AUDIT_VISUAL_ROUTE_20261019.md):
divergent-span carving (measured harmful: ids 182/715/223/766, a wrong product or dropped
subtitle lines) and `detect_speed_ratio`'s own duplicate probe/match pipeline (kept as is, not
re-duplicated here). Kept, because they are real measured cases any design needs: geometry crop
(internal to `measure_video_offset`, nothing to redo here), the candidate start-time fold
(`VideoOffsetResult.candidate_track_delay_ms`, `video_zone_plan`'s `start_delta_ms`), subtitle
text dedup (`_drop_duplicate_candidate_subtitles`, reused unchanged), and the quality vote
(`_pick_quality_winner`, reused unchanged).

No parameter switch (WRITE_ZONES SS4): every branch below is a measured fact about the pair, not
a setting. A step that cannot be established declines, by name, and returns `video_a` unchanged
-- never a raise, never a partial product.
"""

from decimal import Decimal
from os import path

import tools
import video
import video_offset_plan
import merge_video_chimeric
import merge_video_repair
import repair_orchestrator as orch
from merge_video_visual_fallback import _pick_quality_winner, _drop_duplicate_candidate_subtitles

DECIDED_BY = "merge_video_picture.picture_merge"

CAUSE_QUALITY_VOTE_FAILED = "video_quality_vote_failed"
CAUSE_MEASUREMENT_RAISED = "video_offset_measurement_raised"
CAUSE_BUILD_NO_FILE = "plan_application_no_file"
CAUSE_BUILD_RAISED = "plan_application_raised"


def _log_decline(cause, video_a, video_b, detail=""):
    """Log the one explicit line the no-raise contract requires."""
    tools.log_always(
        f"repair: picture_merge declined cause={cause} video_a={video_a.filePath} "
        f"video_b={video_b.filePath}" + (f" {detail}" if detail else "") + "\n")


def _track_plans_one_zone(loser, offset_ms, timeline_ms):
    """Every candidate audio track, moved by the one picture offset for the whole file.

    Same shape `video_offset_plan.video_anchored_route` already builds for its own one-zone
    plan (there, the picture offset arbitrates audio that disagrees; here, there is no common
    audio to disagree in the first place -- the pieces are built the same way either way,
    `verify_video_anchored` checks a track against its OWN original, never against the master).
    """
    track_plans = {}
    for track_language, audio in merge_video_chimeric.iterate_candidate_audios(loser):
        order = int(audio["StreamOrder"])
        _, extent_ms, extent_source = orch._track_timing(loser, audio, Decimal(1))
        pieces, adjustments, _ = video_offset_plan.video_anchored_pieces(
            offset_ms, extent_ms, timeline_ms)
        for adjustment in adjustments:
            tools.log_line(f"repair: plan_edge_adjustment stream={order} zone=0 "
                           f"kind={adjustment['kind']} "
                           f"master_fill_ms={adjustment['master_fill_ms']}\n")
        track_plans[order] = {
            "pieces": pieces, "extent_ms": extent_ms, "extent_source": extent_source,
            "offset_measured": True, "borrow_reason": None,
            "offset_sources": [{"zone": 0, "offset_ms": str(offset_ms),
                                "source": "picture_anchored"}]}
        tools.log_line(f"repair: picture_audio_included stream={order} "
                       f"language={track_language} n_pieces={len(pieces)} "
                       f"for {loser.filePath}\n")
    return track_plans


def _track_plans_zoned(loser, zones, fills, timeline_ms):
    """Every candidate audio track, moved by the per-zone picture offsets.

    `readings` is the same `{zone index: offset_ms}` map used to lay the reference (video)
    pieces -- there is no separate audio measurement on this route, so every track of the
    candidate file shares the one picture-measured plan.
    """
    readings = {zone["zone"]: {"offset_ms": zone["offset_ms"]} for zone in zones}
    track_plans = {}
    for track_language, audio in merge_video_chimeric.iterate_candidate_audios(loser):
        order = int(audio["StreamOrder"])
        _, extent_ms, extent_source = orch._track_timing(loser, audio, Decimal(1))
        pieces, adjustments, overlaps = orch.track_pieces(zones, fills, readings, extent_ms,
                                                           timeline_ms)
        for adjustment in adjustments:
            tools.log_line(f"repair: plan_edge_adjustment stream={order} "
                           f"zone={adjustment['zone']} kind={adjustment['kind']} "
                           f"master_fill_ms={adjustment['master_fill_ms']}\n")
        for overlap in overlaps:
            tools.log_line(f"repair: splice_reread stream={order} zones={overlap['zones']} "
                           f"reread_ms={overlap['reread_ms']}\n")
        track_plans[order] = {
            "pieces": pieces, "extent_ms": extent_ms, "extent_source": extent_source,
            "offset_measured": True, "borrow_reason": None,
            "offset_sources": [{"zone": z["zone"], "offset_ms": str(z["offset_ms"]),
                                "source": "picture_anchored"} for z in zones]}
        tools.log_line(f"repair: picture_audio_included stream={order} "
                       f"language={track_language} n_pieces={len(pieces)} "
                       f"for {loser.filePath}\n")
    return track_plans


def _deliver(winner, loser, plan, repair_deadline, work_root, marker, summary):
    """Shared tail: build, check the file landed, stamp the repair seam, record, return."""
    seam = getattr(loser, merge_video_repair.REPAIR_SEAM_ATTRIBUTE, None)
    job_start_utc = (seam or {}).get("job_start_utc") or "unstamped(no_repair_seam_standalone_run)"
    repaired_obj, assembly = merge_video_repair.build_repaired_video_object(
        loser, winner, plan, path.join(tools.tmpFolder, "repair"), job_start_utc)
    out_path = getattr(repaired_obj, "filePath", None)
    if not out_path or not path.exists(out_path):
        raise merge_video_chimeric.chimeric_error(
            f"the build returned but the picture-merge file is not on disk ({out_path})",
            cause=CAUSE_BUILD_NO_FILE)
    if seam is not None:
        seam["repaired_obj"] = repaired_obj
        seam["assembly"] = assembly
    merge_video_repair.record(loser.filePath, "repaired",
                              f"picture merge (no common audio language): {summary}, "
                              f"marker '{marker}', master {winner.filePath} untouched, "
                              f"file {out_path}",
                              detail={"out_path": out_path, "video_anchored": plan["video_anchored"]})
    return repaired_obj


def _constant_plan(winner, loser, result, language, repair_deadline, work_root):
    """One zone for the whole file: a single constant picture offset."""
    candidate_path = loser.filePath
    shift = -result.offset_frames
    marker = f"picture_merge:{shift:+d}"
    offset_ms = orch._decimal(-result.candidate_track_delay_ms)
    timeline_ms = merge_video_chimeric.get_master_timeline_length_ms(winner)
    work_dir = path.join(work_root, merge_video_chimeric.stable_case_key(candidate_path))
    tools.make_dirs(work_dir)
    _drop_duplicate_candidate_subtitles(winner, loser, work_dir)

    track_plans = _track_plans_one_zone(loser, offset_ms, timeline_ms)
    reference_pieces, _, _ = video_offset_plan.video_anchored_pieces(offset_ms, None, timeline_ms)
    chapters_path, _ = merge_video_chimeric.build_delivered_chapters(
        winner.filePath, candidate_path, reference_pieces, None, timeline_ms, work_dir)

    plan = {
        "kind": "orchestrator_picture_merge", "language": language, "reference_stream": None,
        "quantum_ms": None, "master_path": winner.filePath, "decided_by": DECIDED_BY,
        "segments_dropped_unusable": 0, "speed_margin": None, "speed_engine": None,
        "speed_margin_absent_reason": "no_rate_relation",
        "segments": [{"master_start_ms": Decimal(0), "master_end_ms": timeline_ms,
                      "candidate_offset_ms": offset_ms,
                      "candidate_offset_ms_by_stream": {order: str(offset_ms)
                                                        for order in track_plans}}],
        "track_plans": track_plans, "reference_pieces": reference_pieces,
        "marker": marker, "chapters_path": chapters_path, "speed_ratio": None,
        "speed_ratio_exact": None, "rate_source": None, "resample_gate": None,
        "repair_deadline": repair_deadline,
        "video_anchored": {"offset_frames": result.offset_frames, "shift_frames": shift,
                           "offset_ms": offset_ms, "fps": str(result.fps),
                           "trigger": "picture_merge_no_common_language"},
    }
    return _deliver(winner, loser, plan, repair_deadline, work_root, marker,
                    f"every candidate track moved by the picture offset "
                    f"{result.offset_frames:+d} frame(s), {len(track_plans)} audio track(s) "
                    f"included")


def _speed_plan(winner, loser, speed_result, language, repair_deadline, work_root):
    """One zone, with a constant-ratio speed correction applied to every candidate track first."""
    candidate_path = loser.filePath
    shift = -speed_result.residual_frames
    ratio = speed_result.ratio
    ratio_name = speed_result.ratio_name or f"{ratio.numerator}/{ratio.denominator}"
    marker = f"picture_merge_speed:{shift:+d}@{ratio_name}"
    offset_ms = orch._decimal(-speed_result.candidate_offset_ms)
    timeline_ms = merge_video_chimeric.get_master_timeline_length_ms(winner)
    work_dir = path.join(work_root, merge_video_chimeric.stable_case_key(candidate_path))
    tools.make_dirs(work_dir)
    _drop_duplicate_candidate_subtitles(winner, loser, work_dir)

    track_plans = _track_plans_one_zone(loser, offset_ms, timeline_ms)
    reference_pieces, _, _ = video_offset_plan.video_anchored_pieces(offset_ms, None, timeline_ms)
    chapters_path, _ = merge_video_chimeric.build_delivered_chapters(
        winner.filePath, candidate_path, reference_pieces, None, timeline_ms, work_dir)

    resample_gate = {"verdict": "confirmed", "span_coverage": speed_result.span_coverage,
                     "engine": "asetrate", "cause": None}
    plan = {
        "kind": "orchestrator_picture_merge_speed", "language": language,
        "reference_stream": None, "quantum_ms": None, "master_path": winner.filePath,
        "decided_by": DECIDED_BY,
        "segments_dropped_unusable": 0, "speed_margin": None, "speed_engine": "asetrate",
        "speed_margin_absent_reason": "visual_frame_match_no_runner_up",
        "segments": [{"master_start_ms": Decimal(0), "master_end_ms": timeline_ms,
                      "candidate_offset_ms": offset_ms,
                      "candidate_offset_ms_by_stream": {order: str(offset_ms)
                                                        for order in track_plans}}],
        "track_plans": track_plans, "reference_pieces": reference_pieces,
        "marker": marker, "chapters_path": chapters_path,
        "speed_ratio": str(Decimal(ratio.numerator) / Decimal(ratio.denominator)),
        "speed_ratio_exact": f"{ratio.numerator}/{ratio.denominator}",
        "rate_source": video_offset_plan.RATE_SOURCE_VISUAL, "resample_gate": resample_gate,
        "repair_deadline": repair_deadline,
        "video_anchored": {"offset_frames": None, "residual_frames": speed_result.residual_frames,
                           "shift_frames": shift, "offset_ms": offset_ms,
                           "fps": str(speed_result.candidate_fps),
                           "master_fps": str(speed_result.master_fps),
                           "speed_ratio": str(ratio), "speed_ratio_name": ratio_name,
                           "trigger": "picture_merge_no_common_language_speed"},
    }
    return _deliver(winner, loser, plan, repair_deadline, work_root, marker,
                    f"speed ratio {ratio} ({ratio_name}), residual offset {shift:+d} frame(s), "
                    f"{len(track_plans)} audio track(s) included")


def _zoned_plan(winner, loser, result, groups, language, repair_deadline, work_root):
    """One zone per stable picture offset, holes/head/tail filled from the master."""
    candidate_path = loser.filePath
    start_delta_ms = (result.candidate_start_s - result.master_start_s) * 1000
    timeline_ms = merge_video_chimeric.get_master_timeline_length_ms(winner)
    zones, fills = video_offset_plan.video_zone_plan(groups, result.frame_ms, timeline_ms,
                                                      start_delta_ms)
    work_dir = path.join(work_root, merge_video_chimeric.stable_case_key(candidate_path))
    tools.make_dirs(work_dir)
    _drop_duplicate_candidate_subtitles(winner, loser, work_dir)

    readings = {zone["zone"]: {"offset_ms": zone["offset_ms"]} for zone in zones}
    track_plans = _track_plans_zoned(loser, zones, fills, timeline_ms)
    reference_pieces, adjustments, _ = orch.track_pieces(zones, fills, readings, None, timeline_ms)
    for adjustment in adjustments:
        tools.log_line(f"repair: plan_edge_adjustment stream=reference zone={adjustment['zone']} "
                       f"kind={adjustment['kind']} "
                       f"master_fill_ms={adjustment['master_fill_ms']}\n")
    chapters_path, _ = merge_video_chimeric.build_delivered_chapters(
        winner.filePath, candidate_path, reference_pieces, None, timeline_ms, work_dir)

    marker = f"picture_merge_zones:n={len(zones)}"
    plan = {
        "kind": "orchestrator_picture_merge_zones", "language": language,
        "reference_stream": None, "quantum_ms": None, "master_path": winner.filePath,
        "decided_by": DECIDED_BY,
        "segments_dropped_unusable": 0, "speed_margin": None, "speed_engine": None,
        "speed_margin_absent_reason": "no_rate_relation",
        "segments": [{"master_start_ms": zone["master_start_ms"],
                      "master_end_ms": zone["master_end_ms"],
                      "candidate_offset_ms": zone["offset_ms"],
                      "candidate_offset_ms_by_stream": {order: str(zone["offset_ms"])
                                                        for order in track_plans}}
                     for zone in zones],
        "track_plans": track_plans, "reference_pieces": reference_pieces,
        "marker": marker, "chapters_path": chapters_path, "speed_ratio": None,
        "speed_ratio_exact": None, "rate_source": None, "resample_gate": None,
        "repair_deadline": repair_deadline,
        "video_anchored": {"offset_frames": None, "n_zones": len(zones),
                           "offset_ms": zones[0]["offset_ms"] if zones else Decimal(0),
                           "offsets_ms": [str(zone["offset_ms"]) for zone in zones],
                           "fps": str(result.fps),
                           "trigger": "picture_merge_no_common_language_multizone"},
    }
    return _deliver(winner, loser, plan, repair_deadline, work_root, marker,
                    f"{len(zones)} picture zones, {len(track_plans)} audio track(s) included")


def picture_merge(video_a, video_b, forced_best_video, language=None, repair_deadline=None,
                  work_root=None):
    '''Merge two files sharing no common audio language by their picture alone.

    Args:
        video_a: the winner of the previous per-language rounds, or the forced video.
        video_b: the file with no audio language in common with `video_a`.
        forced_best_video: truthy when `video_a` was forced; the quality vote only runs when
            this is falsy.
        language: comparison language forwarded for head/tail fill and chapter bookkeeping only;
            may be None (there is, by construction, no common language on this route).
        repair_deadline: an optional `time.monotonic()` deadline, forwarded like
            `repair_orchestrator`'s own repair budget.
        work_root: scratch directory for the measurement cache and the build; defaults under
            `tools.tmpFolder`.

    Returns:
        The winning video object, with `video_b`'s repaired form appended to its
        `sameAudioMD5UseForCalculation` on success -- audio tracks included, subtitles deduped
        against the master, master tracks never touched (md5 unchanged). Never raises
        (`KeyboardInterrupt` / `SystemExit` excepted): any failure logs one named decline and
        returns `video_a` unchanged, no file produced.
    '''
    try:
        winner, loser = ((video_a, video_b) if forced_best_video
                         else _pick_quality_winner(video_a, video_b))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:                                           # noqa: BLE001
        _log_decline(CAUSE_QUALITY_VOTE_FAILED, video_a, video_b,
                    f"error={type(error).__name__}:{error}")
        return video_a

    work_root = work_root or path.join(tools.tmpFolder, "repair", "picture_merge")
    result, speed_result = None, None
    try:
        tools.make_dirs(work_root)
        cache_dir = path.join(work_root, "measure")
        tools.make_dirs(cache_dir)
        m_info, _ = video_offset_plan.probe_video(winner.filePath)
        c_info, _ = video_offset_plan.probe_video(loser.filePath)
        is_speed_pair = (
            m_info is not None and c_info is not None
            and m_info["r_rate"] is not None and c_info["r_rate"] is not None
            and m_info["r_rate"] == m_info["avg_rate"] and c_info["r_rate"] == c_info["avg_rate"]
            and m_info["r_rate"] != c_info["r_rate"])
        if is_speed_pair:
            speed_result = video_offset_plan.detect_speed_ratio(
                winner.filePath, loser.filePath, cache_dir, deadline=repair_deadline)
            if speed_result.status != video_offset_plan.STATUS_SPEED_OK:
                _log_decline(speed_result.status, video_a, video_b,
                            f"reason={speed_result.reason}")
                return video_a
        else:
            result = video_offset_plan.measure_video_offset(
                winner.filePath, loser.filePath, cache_dir, deadline=repair_deadline)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:                                           # noqa: BLE001
        _log_decline(CAUSE_MEASUREMENT_RAISED, video_a, video_b,
                    f"error={type(error).__name__}:{error}")
        return video_a

    if speed_result is None and result.status not in (
            video_offset_plan.STATUS_OK, video_offset_plan.STATUS_NOT_CONSTANT,
            video_offset_plan.STATUS_COVERAGE):
        _log_decline(result.status, video_a, video_b, f"reason={result.reason}")
        return video_a

    try:
        if speed_result is not None:
            repaired_obj = _speed_plan(winner, loser, speed_result, language, repair_deadline,
                                       work_root)
        elif result.status == video_offset_plan.STATUS_OK:
            repaired_obj = _constant_plan(winner, loser, result, language, repair_deadline,
                                          work_root)
        else:
            groups = video_offset_plan.group_video_zones(result.pairs or [], float(result.frame_ms))
            cause = video_offset_plan.check_zone_compatibility(groups, result.master_frames)
            if cause is not None:
                _log_decline(cause, video_a, video_b, f"n_zones={len(groups)}")
                return video_a
            tools.log_always(
                f"repair: picture_merge_zones n_zones={len(groups)} offsets="
                + ",".join(f"{d:+d}x{len(m)}" for d, m in groups)
                + f" for {loser.filePath}\n")
            repaired_obj = _zoned_plan(winner, loser, result, groups, language, repair_deadline,
                                      work_root)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:                                           # noqa: BLE001
        cause = getattr(error, "cause", None) or type(error).__name__
        _log_decline(cause, video_a, video_b, f"error={error}")
        try:
            merge_video_repair.record(loser.filePath, "declined", str(error), cause=cause)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:                                                # noqa: BLE001
            pass
        return video_a

    if repaired_obj is None:
        _log_decline(CAUSE_BUILD_NO_FILE, video_a, video_b)
        return video_a

    winner.sameAudioMD5UseForCalculation.append(repaired_obj)
    return winner
