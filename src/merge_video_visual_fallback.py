"""Visual fallback merge for a pair sharing no common audio language.

Chantier D: when the owner's per-language merge rounds are exhausted and two files have no
audio language in common, `visual_fallback_merge` is the last resort before raising -- it
compares the two pictures instead of the audio. It reuses chantier C's video-only measurement
(`video_offset_plan.measure_video_offset`, its zone grouping and compatibility guards) and the
same delivery path already proven by `repair_orchestrator.video_anchored_route` (the
video-anchored plan shape, `merge_video_repair.build_repaired_video_object`, the delivery
gates): no builder or verifier of its own.

Zero production caller yet: the owner wires the call into the frozen per-language round, inside
`mergeVideo.sync_merge_video`'s "no common language" branch.
"""

from decimal import Decimal
from os import path
from time import gmtime, strftime

import tools
import video
import video_offset_plan

# A changing video offset (chantier C's zones) passes the same before-any-plan guards as a
# constant one, but applying several zones through the common delivery path is chantier C's own
# next step (its diagnostic-only comment on `video_anchored_route`), not wired yet: declined by
# name rather than built here, unproven, against real media.
DECLINE_ZONE_APPLY_UNAVAILABLE = "video_zone_apply_not_available"

# Failure causes this module names itself, beyond the ones `video_offset_plan` already returns
# (its own status tokens, and `check_zone_compatibility`'s DECLINE_* tokens, are logged as-is).
CAUSE_QUALITY_VOTE_FAILED = "video_quality_vote_failed"
CAUSE_MEASUREMENT_RAISED = "video_offset_measurement_raised"
CAUSE_BUILD_NO_FILE = "plan_application_no_file"
CAUSE_WORK_DIR_UNAVAILABLE = "work_dir_unavailable"


def _log_decline(cause, video_a, video_b, detail=""):
    """Log the one explicit line the no-raise contract requires."""
    tools.log_always(
        f"repair: visual_fallback_merge declined cause={cause} video_a={video_a.filePath} "
        f"video_b={video_b.filePath}" + (f" {detail}" if detail else "") + "\n")


def _pick_quality_winner(video_a, video_b):
    """Return (winner, loser) by video quality, built from the same inputs `simple_merge_video`
    passes to `video.get_best_quality_video` for an unforced pair."""
    min_duration_s = video.get_shortest_video_durations([video_a, video_b])
    begin_s, length_s = video.generate_begin_and_length_by_segment(min_duration_s)
    time_by_test = strftime('%H:%M:%S', gmtime(video.generate_time_compare_video_quality(length_s)))
    begins_video = video.generate_cut_to_compare_video_quality(begin_s, begin_s, length_s)
    if video.get_best_quality_video(video_a, video_b, begins_video, time_by_test) == "1":
        return video_a, video_b
    return video_b, video_a


def _apply_constant_plan(winner, loser, result, language, repair_deadline, work_root):
    """Build and deliver the one-zone video-anchored plan; return a repaired object or None."""
    import merge_video_chimeric
    import merge_video_repair
    import repair_orchestrator as orch

    candidate_path = loser.filePath
    shift = -result.offset_frames
    marker = f"visual_fallback:{shift:+d}"
    picture_ms = -result.candidate_track_delay_ms
    offset_ms = orch._decimal(picture_ms)
    timeline_ms = merge_video_chimeric.get_master_timeline_length_ms(winner)
    work_dir = path.join(work_root, merge_video_chimeric.stable_case_key(candidate_path))
    tools.make_dirs(work_dir)

    # No candidate audio track is planned here, deliberately: this route exists only because no
    # audio language is shared with the master, so none of the loser's audio tracks has anything
    # to be verified against (`verify_video_anchored` checks a track against its own original,
    # never against the master -- a track this route cannot place in any master language is not
    # delivered unverified). Declining an unordered track_plans entry would also just decline it
    # individually below with no effect on the subtitles; dropping it here says so once, by name.
    # Subtitles carry no audio identity and ride `reference_pieces` unconditionally, below.
    track_plans = {}
    for track_language, audio in merge_video_chimeric.iterate_candidate_audios(loser):
        tools.log_line(f"repair: visual_fallback_audio_dropped stream={audio['StreamOrder']} "
                       f"language={track_language} reason=no_language_in_common_to_verify_against "
                       f"for {candidate_path}\n")

    reference_pieces, _, _ = video_offset_plan.video_anchored_pieces(offset_ms, None, timeline_ms)
    chapters_path, _ = merge_video_chimeric.build_delivered_chapters(
        winner.filePath, candidate_path, reference_pieces, None, timeline_ms, work_dir)

    seam = getattr(loser, merge_video_repair.REPAIR_SEAM_ATTRIBUTE, None)
    job_start_utc = ((seam or {}).get("job_start_utc")
                     or "unstamped(no_repair_seam_standalone_run)")

    plan = {
        "kind": "orchestrator_visual_fallback", "language": language, "reference_stream": None,
        "quantum_ms": None, "master_path": winner.filePath,
        "decided_by": "merge_video_visual_fallback.visual_fallback_merge",
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
                           "trigger": "visual_fallback_no_common_language"},
    }
    repaired_obj, assembly = merge_video_repair.build_repaired_video_object(
        loser, winner, plan, path.join(tools.tmpFolder, "repair"), job_start_utc)
    out_path = getattr(repaired_obj, "filePath", None)
    if not out_path or not path.exists(out_path):
        raise merge_video_chimeric.chimeric_error(
            f"the build returned but the visual-fallback file is not on disk ({out_path})",
            cause=CAUSE_BUILD_NO_FILE)
    if seam is not None:
        seam["repaired_obj"] = repaired_obj
        seam["assembly"] = assembly
    merge_video_repair.record(candidate_path, "repaired",
                              f"visual fallback (no common audio language): every candidate "
                              f"track moved by the picture offset {result.offset_frames:+d} "
                              f"frame(s), marker '{marker}', master {winner.filePath} untouched, "
                              f"file {out_path}",
                              detail={"out_path": out_path, "video_anchored": plan["video_anchored"]})
    return repaired_obj


def visual_fallback_merge(video_a, video_b, forced_best_video, language=None,
                          repair_deadline=None, work_root=None):
    '''Merge two files sharing no common audio language by their picture alone.

    Args:
        video_a: the winner of the owner's previous per-language rounds, or the forced video.
        video_b: the file with no audio language in common with `video_a`.
        forced_best_video: truthy when `video_a` was forced (as the caller's own
            `forced_best_video`); the video-quality vote only runs when this is falsy.
        language: comparison language forwarded to the chimeric builder for head/tail fill
            only (borrowed master audio vs. silence); may be None.
        repair_deadline: an optional `time.monotonic()` deadline, forwarded to the measurement
            and the build like `repair_orchestrator`'s own repair budget.
        work_root: scratch directory for the measurement cache and the build; defaults under
            `tools.tmpFolder`.

    `audioRules` and `dict_file_path_obj` (passed to the analogous `compare_video` /
    `get_delay_and_best_video`) are not needed here: this path never re-derives an audio delay
    or looks a path up by name, and the loser rejoins the merge the same way a zone-A repair
    does -- appended to the winner's own `sameAudioMD5UseForCalculation`, not inserted into
    `dict_file_path_obj`.

    Returns:
        The winning video object. Never raises (`KeyboardInterrupt` / `SystemExit` excepted):
        on any failure -- different content, another episode, no usable plan, a guard refusal,
        or a build/delivery error -- logs one explicit line naming the cause and returns
        `video_a` unchanged, with no chimeric file added anywhere.
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

    work_root = work_root or path.join(tools.tmpFolder, "repair", "visual_fallback")
    try:
        tools.make_dirs(work_root)
        cache_dir = path.join(work_root, "measure")
        tools.make_dirs(cache_dir)
        result = video_offset_plan.measure_video_offset(
            winner.filePath, loser.filePath, cache_dir, deadline=repair_deadline)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:                                           # noqa: BLE001
        _log_decline(CAUSE_MEASUREMENT_RAISED, video_a, video_b,
                    f"error={type(error).__name__}:{error}")
        return video_a

    if result.status not in (video_offset_plan.STATUS_OK, video_offset_plan.STATUS_NOT_CONSTANT,
                             video_offset_plan.STATUS_COVERAGE):
        _log_decline(result.status, video_a, video_b, f"reason={result.reason}")
        return video_a

    try:
        if result.status == video_offset_plan.STATUS_OK:
            repaired_obj = _apply_constant_plan(winner, loser, result, language,
                                                repair_deadline, work_root)
        else:
            groups = video_offset_plan.group_video_zones(result.pairs or [],
                                                          float(result.frame_ms))
            cause = video_offset_plan.check_zone_compatibility(groups, result.master_frames)
            if cause is not None:
                _log_decline(cause, video_a, video_b, f"n_zones={len(groups)}")
                return video_a
            _log_decline(DECLINE_ZONE_APPLY_UNAVAILABLE, video_a, video_b,
                        f"n_zones={len(groups)} offsets="
                        + ",".join(f"{d:+d}x{len(m)}" for d, m in groups))
            return video_a
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:                                           # noqa: BLE001
        import merge_video_repair
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
