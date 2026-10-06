"""Visual fallback merge for a pair sharing no common audio language.

Chantier D: when the owner's per-language merge rounds are exhausted and two files have no
audio language in common, `visual_fallback_merge` is the last resort before raising -- it
compares the two pictures instead of the audio. It reuses chantier C's video-only measurement
(`video_offset_plan.measure_video_offset`, its zone grouping and compatibility guards) and the
same delivery path already proven by `repair_orchestrator.video_anchored_route` (the
video-anchored plan shape, `merge_video_repair.build_repaired_video_object`, the delivery
gates): no builder or verifier of its own.

Speed: the two files' declared exact frame rates may differ while each side is still
individually CFR (PAL/NTSC, 1001/1000, ...) -- frames, never time, carry the acceleration, per
the owner's own rule. `video_offset_plan.detect_speed_ratio` matches scene cuts by FRAME INDEX
UNSCALED, the same plain `match_changes`/`prove_constant` chantier C already runs for a no-speed
offset: a pure declared-rate difference carries the SAME discrete frames on both sides, nothing
added or dropped, so the ratio cancels out of the candidate-minus-master correspondence exactly
(checked: `candidate_fps / (ratio_declared x master_fps) == 1`) and scaling the search by that
ratio would manufacture a drift that the file does not actually carry (replay-E defect,
corrected: measured on id 33 before its real block -- below -- was found). A confirmed ratio is
snapped to the nearest named broadcast-rate fraction (`merge_video_resample
.build_rate_ratio_vocabulary`) and carried through `merge_video_repair.speed_plan_evidence`'s
own evidence gate (`rate_source="visual_frame_match"`) so `build_repaired_video_object` applies
it to the loser's audio (exact-fraction `asetrate`) and subtitles (linear retime) before the
zone plan is built, which is then expressed once, in master time.

id 33's real block (chantier E, item 2) was never the matching math: the master and candidate
are cropped to two different frame heights (a BluRay's open-matte 1080 px against a WEB
release's cinematic 800 px crop of the SAME shot). `video_offset_plan.decode_scenes_and_hashes`
used to scale every frame to its fixed decode box with a plain `scale=W:H`, which stretches
rather than fits -- the same picture then sits at two different vertical scales on each side,
and a frame pair that should pHash identical instead read as unrelated (measured: grey distance
0.465, next to `frame_hash.SAME_FRAME_MAX` 0.07 -- indistinguishable from a wrong lag). Fixed by
fitting each side to the decode box by its OWN aspect ratio first (letterboxed/pillarboxed,
never stretched; measured on the same pair: grey distance 0.113, no longer indistinguishable
from a wrong lag by `frame_hash.align`'s own margin, the only gate this decode feeds).

Picture-divergent spans (chantier E, item 1): inside a zone whose own offset never changed, a
span can still show a different PICTURE -- redrawn animation, a different eyecatch/credits
card, or black/static on one side against content on the other. `match_changes` already
refuses to match a scene cut whose own window content distance is too high, so a divergent
span never invents a false zone of its own; `video_offset_plan.find_divergent_spans_in_zone`
(grey pHash, `frame_hash.SAME_FRAME_MAX`, run over the whole zone) finds the span itself, and
`carve_divergent_fills` splits it out of its zone as one more master fill -- the owner's rule,
"on prend du master les zones differentes" -- so the master's own frames cover it in the
delivered plan, logged once per span (master/candidate frames, width, reason). Wired into both
`_apply_constant_plan` (promoted to a multi-zone plan when a span is found inside its single
zone) and `_apply_multizone_plan`, through the shared `_apply_zoned_plan` builder.

Two guards, both logged and refused by returning `video_a` unchanged, never raised:

- `visual_content_mismatch`: too few matched scene cuts, or too little of the master's span
  covered by them, to trust any frame correspondence at all -- not the same episode. Also
  raised by the plain (no-speed) path's own `check_zone_compatibility`/coverage guards, logged
  as the status `video_offset_plan.measure_video_offset` already names.
- `visual_frame_count_mismatch`: between the first and last common scene cut the two files do
  NOT carry the same number of frames beyond what the detected ratio alone explains -- the
  matched cuts' residual is not one constant once the ratio is divided out, so some matched
  pair's own frame offset jumped mid-span (a stable-offset zone's own invariant broken): a
  telecine, or a frame-rate conversion that dropped or duplicated frames. The two rates are not
  comparable and no speed correction is applied.

Zero production caller yet: the owner wires the call into the frozen per-language round, inside
`mergeVideo.sync_merge_video`'s "no common language" branch.

Contract with `mergeVideo.generate_launch_merge_command(dict_with_video_quality_logic,
dict_file_path_obj, out_folder, common_language_use_for_generate_delay, audioRules)`, called
unchanged at the end of either flow below:

- `dict_with_video_quality_logic` stays EMPTY on this route, always (measured, replay-D3 defect
  1): a pair where one side is a raw file and the other a chimeric repair built FROM that side
  is not the `already_compared[name_a][name_b] = bool` shape `get_delay_and_best_video` builds
  for two files both still raw -- there is only ever one surviving file to a pairing here, and
  that survivor already carries the whole pairing's content (its own, plus the chimeric repair
  of whichever side lost). A call this route *declines* (returns `video_a` with nothing added),
  or whose winner CHANGES (the candidate out-votes the current winner on quality), drops the
  LOSING side -- whichever of the two did not survive -- from `dict_file_path_obj` and
  `videosObj`, the same way `remove_not_compatible_video` already does for an audio-incompatible
  pair: a raw loser left in `dict_file_path_obj` reaches `generate_merge_command_other_part`,
  which indexes it by `.delays[language]` -- a key nothing on this route ever sets, since no
  `prepare_get_delay` runs for it (measured: `KeyError` at `mergeVideo.py:1585`, replay-D3 id
  681/682). Leaving a path in `dict_file_path_obj` with no entry anywhere in
  `dict_with_video_quality_logic` is exactly what `generate_launch_merge_command`'s own
  `dict_file_path_obj.keys() - set_bad_video` already expects for a file nobody compared.
- `common_language_use_for_generate_delay` has no natural value here (no language is common):
  use `pick_common_language_use_for_generate_delay(winner)` once, after the whole loop, on the
  final surviving winner -- it is always one of the winner's own (untouched) audio languages, the
  only ones `generate_launch_merge_command`'s `keep_best_audio` and
  `generate_merge_command_other_part` can index without raising, since neither a `.delays` key
  nor an audio language is otherwise initialized on this route (no `prepare_get_delay` runs).
- A winner's `sameAudioMD5UseForCalculation` list already carries what `generate_merge_command
  _common_md5` needs per chimeric loser (`delay_same_md5_audio = Decimal('0')`, set by
  `merge_video_repair.build_repaired_video_object` itself): no further change.
- If every candidate is declined, NOTHING is built (replay-D3 defect 4, measured on ids
  237/196): `generate_launch_merge_command` is only called when at least one candidate actually
  joined. Calling it on a lone survivor with nothing appended would hand `fusion.py` what looks
  like a successful merge of a single file; `apply_production_outcome` would then delete the
  declined candidate's own file and registry row instead of leaving it in error. When nothing
  joined, control falls through to the surrounding branch's own existing "No common language"
  check and raise (`mergeVideo.py`, `audio_counts`/`most_frequent_language`) unchanged.

The owner's own pairwise round, inside `sync_merge_video`'s `if len(commonLanguages) == 0:`
branch, before its `if audio_counts[most_frequent_language] == 1:` raise:

    dict_with_video_quality_logic = {}
    # `forced_best_video` here is `sync_merge_video`'s own parameter: a file PATH (or None/""),
    # never a bool -- the forced file must become the starting winner itself (replay-D3 defect
    # 5a), not merely make every call "truthy": `visual_fallback_merge`'s own third argument is
    # a per-call bool, true only on the call where the CURRENT winner is that forced path.
    forced_path = forced_best_video or None
    winner = next((v for v in videosObj if v.filePath == forced_path), videosObj[0])
    videosObj.remove(winner)
    any_joined = False
    for candidate in videosObj:
        n_before = len(winner.sameAudioMD5UseForCalculation)
        was_forced = (winner.filePath == forced_path)
        new_winner = merge_video_visual_fallback.visual_fallback_merge(
            winner, candidate, was_forced)
        if new_winner is winner and len(winner.sameAudioMD5UseForCalculation) == n_before:
            # declined: nothing was added for `candidate` -- it never joins the merge
            del dict_file_path_obj[candidate.filePath]
            continue
        # accepted: whichever side did not survive never reaches
        # generate_launch_merge_command -- its content already rides the chimeric object
        # appended to the survivor's own sameAudioMD5UseForCalculation (defect 1, above)
        loser = candidate if new_winner is winner else winner
        del dict_file_path_obj[loser.filePath]
        winner = new_winner
        any_joined = True
    if any_joined:
        common_language_use_for_generate_delay = (
            merge_video_visual_fallback.pick_common_language_use_for_generate_delay(winner))
        generate_launch_merge_command(dict_with_video_quality_logic, dict_file_path_obj,
                                      out_folder, common_language_use_for_generate_delay,
                                      audioRules)
    # else: fall through to the surrounding branch's own "No common language" raise (defect 4)

A future `tools.force_video_comparison` mode (skips every per-language round, always compares by
picture) reduces to the same loop over every file from the very first one:

    if tools.force_video_comparison:
        dict_with_video_quality_logic, winner = force_video_comparison_merge(
            videosObj, dict_file_path_obj, forced_best_video)
        common_language_use_for_generate_delay = (
            merge_video_visual_fallback.pick_common_language_use_for_generate_delay(winner))
        generate_launch_merge_command(dict_with_video_quality_logic, dict_file_path_obj,
                                      out_folder, common_language_use_for_generate_delay,
                                      audioRules)
        return

`force_video_comparison_merge` (the owner's own future function, not part of this module) is the
same loop as above, lifted out so both call sites share it -- same empty-`dict_with_video_quality
_logic`, same per-candidate drop of whichever side lost, same `any_joined` guard before calling
`generate_launch_merge_command` -- it calls `visual_fallback_merge` for every file, not only
when `commonLanguages` is empty, and returns `(dict_with_video_quality_logic, winner)` only when
at least one candidate joined; when none did, it is this call site's own responsibility to avoid
building a file from nothing (not shown here: `tools.force_video_comparison` has no language
round to fall back into, so its own caller decides what "nothing joined" means for it).
"""

from decimal import Decimal
from os import path
from time import gmtime, strftime

import tools
import video
import video_offset_plan

# Kept for callers/tests still matching on the old token: a changing video offset now applies
# through `_apply_multizone_plan` (`video_offset_plan.video_zone_plan` + `track_pieces`) once
# `check_zone_compatibility` passes; this value is only ever logged, never returned, from here on.
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


def _extract_subtitle_text(file_path, stream_order, codec_name, work_dir, tag):
    """Extract one subtitle stream's dialogue text, ignoring every timestamp.

    Returns the non-empty plaintext lines, in cue order, or None when the stream is a bitmap
    format (no text to compare), carries no cue, or could not be extracted/parsed -- any of
    which simply skips the stream from the duplicate check, never raises.
    """
    import merge_video_chimeric
    target = merge_video_chimeric.classify_subtitle(codec_name)
    if target == "bitmap":
        return None
    out_path = path.join(work_dir, f"dedup_{tag}_{stream_order}.{target}")
    command = [tools.software["ffmpeg"], "-y", "-nostdin",
               "-analyzeduration", "1000M", "-probesize", "1000M",
               "-i", file_path, "-map", f"0:{int(stream_order)}", "-map_chapters", "-1",
               "-c:s", target, out_path]
    try:
        tools.launch_cmdExt_with_timeout_reload(command, 1, 120)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:                                                    # noqa: BLE001
        return None
    if not path.exists(out_path) or not path.getsize(out_path):
        return None
    try:
        import pysubs2
        subs = pysubs2.load(out_path)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:                                                    # noqa: BLE001
        return None
    return [event.plaintext.strip() for event in subs if event.plaintext.strip()]


def _drop_duplicate_candidate_subtitles(winner, loser, work_dir):
    '''Mark a candidate subtitle `dropped_duplicate` when its text (dialogue lines, ignoring
    timing) equals a same-language master subtitle's own.

    Subtitles ride `reference_pieces` unconditionally and are retimed onto the master clock
    before delivery, so a subtitle the master already carries in the same language reaches the
    product again as a byte-identical-text, differently-timed copy; the merge's only dedup is
    the delivered stream's MD5 (`mergeVideo.py`'s `md5_sub_already_added`), which a retimed copy
    never matches (replay-D3 defect 3, measured: 96/96 identical lines at id 681). Checked once,
    before the build, never raised: an extraction or parse failure just leaves the candidate
    track in, unflagged.
    '''
    master_text_cache = {}
    for language, subtitles in loser.subtitles.items():
        master_tracks = winner.subtitles.get(language) or []
        if not master_tracks:
            continue
        for subtitle in subtitles:
            codec = (subtitle.get("ffprobe") or {}).get("codec_name", "")
            candidate_lines = _extract_subtitle_text(
                loser.filePath, subtitle["StreamOrder"], codec, work_dir, f"cand_{language}")
            if not candidate_lines:
                continue
            for master_subtitle in master_tracks:
                key = (language, master_subtitle["StreamOrder"])
                if key not in master_text_cache:
                    master_codec = (master_subtitle.get("ffprobe") or {}).get("codec_name", "")
                    master_text_cache[key] = _extract_subtitle_text(
                        winner.filePath, master_subtitle["StreamOrder"], master_codec,
                        work_dir, f"master_{language}")
                master_lines = master_text_cache[key]
                if master_lines and master_lines == candidate_lines:
                    subtitle["dropped_duplicate"] = (
                        f"text_identical_to_master_stream_{master_subtitle['StreamOrder']}")
                    tools.log_line(
                        f"repair: visual_fallback_subtitle_dropped "
                        f"stream={subtitle['StreamOrder']} language={language} "
                        f"reason={subtitle['dropped_duplicate']} lines={len(candidate_lines)} "
                        f"for {loser.filePath}\n")
                    break


def pick_common_language_use_for_generate_delay(winner):
    '''Pick `generate_launch_merge_command`'s `common_language_use_for_generate_delay` when the
    owner's pairwise video-comparison loop leaves no audio language shared by every file.

    No correlation ever runs on this route (every offset was measured on the picture, already
    baked into each chimeric file by `video_offset_plan`/`merge_video_repair`), so nothing calls
    `prepare_get_delay` to seed `winner.delays`; `generate_merge_command_other_part` and
    `generate_launch_merge_command`'s own `keep_best_audio` call both index by this language, and
    both read it off `winner` -- the overall survivor, the one object whose own `.audios` is
    never touched by this route (only a loser's track plan is dropped, never the winner's).
    Picking one of the winner's own languages guarantees both reads succeed.

    Call this once, after the whole pairwise loop, right before `generate_launch_merge_command`
    (see the module docstring for the owner's exact call site).

    Returns:
        The chosen language, after setting `winner.delays[language] = Decimal('0')` if it was
        not already present (a prior per-language round may have left a real, measured delay
        there; this never overwrites one).
    '''
    language = (tools.special_params["original_language"]
               if tools.special_params["original_language"] in winner.audios
               else next(iter(winner.audios)))
    winner.delays.setdefault(language, Decimal('0'))
    return language


def _pick_quality_winner(video_a, video_b):
    """Return (winner, loser) by video quality, built from the same inputs `simple_merge_video`
    passes to `video.get_best_quality_video` for an unforced pair."""
    min_duration_s = video.get_shortest_video_durations([video_a, video_b])
    begin_s, length_s = video.generate_begin_and_length_by_segment(min_duration_s)
    time_by_test = strftime('%H:%M:%S', gmtime(video.generate_time_compare_video_quality(length_s)))
    begins_video = video.generate_cut_to_compare_video_quality(begin_s, begin_s, length_s)
    if video.get_best_quality_video(video_a, video_b, begins_video, time_by_test) == 1:
        return video_a, video_b
    return video_b, video_a


def _decode_hashes_for_divergence(winner, loser, result_fps, work_root, repair_deadline):
    '''Redecode (cache hit: same path/fps/work_dir `measure_video_offset` already used) the
    whole-file grey pHash of both sides, for `video_offset_plan.carve_divergent_fills`.

    Never raises: on any probe/decode failure, returns `(None, None, None)` and the caller
    skips divergent-span detection for this pair, keeping its plan exactly as measured.
    '''
    try:
        m_info, why = video_offset_plan.probe_video(winner.filePath)
        if m_info is None:
            raise RuntimeError(f"master:{why}")
        c_info, why = video_offset_plan.probe_video(loser.filePath)
        if c_info is None:
            raise RuntimeError(f"candidate:{why}")
        cache_dir = path.join(work_root, "measure")
        tools.make_dirs(cache_dir)
        _, m_hashes, _, _ = video_offset_plan.decode_scenes_and_hashes(
            winner.filePath, result_fps, m_info["duration_s"], cache_dir, repair_deadline)
        _, c_hashes, _, _ = video_offset_plan.decode_scenes_and_hashes(
            loser.filePath, result_fps, c_info["duration_s"], cache_dir, repair_deadline)
        return m_hashes, c_hashes, video_offset_plan.Fraction(1000) / result_fps
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:                                           # noqa: BLE001
        tools.log_line(f"repair: visual_fallback_divergent_skip error={type(error).__name__}:"
                       f"{error} for {loser.filePath}\n")
        return None, None, None


def _carve_divergence(winner, loser, result_fps, zones, fills, work_root, repair_deadline):
    """Run `find_divergent_spans_in_zone`/`carve_divergent_fills` over `zones`; on any failure
    (including no usable decode), return `(zones, fills)` unchanged."""
    m_hashes, c_hashes, frame_ms = _decode_hashes_for_divergence(
        winner, loser, result_fps, work_root, repair_deadline)
    if m_hashes is None:
        return zones, fills
    return video_offset_plan.carve_divergent_fills(zones, fills, m_hashes, c_hashes, frame_ms,
                                                    log=tools.log_always)


def _apply_zoned_plan(winner, loser, zones, fills, fps, trigger, language, repair_deadline,
                      work_root):
    """Build and deliver a multi-zone video-anchored plan from already-built `zones`/`fills`
    (one zone per stable picture offset, or more once `carve_divergence` has split out any
    picture-divergent span); return a repaired object or None.

    Shared by the plain constant-offset plan (promoted to one zone, split further when a
    divergent span was found inside it) and the multi-zone plan built from `group_video_zones`.
    """
    import merge_video_chimeric
    import merge_video_repair
    import repair_orchestrator as orch

    candidate_path = loser.filePath
    timeline_ms = merge_video_chimeric.get_master_timeline_length_ms(winner)
    work_dir = path.join(work_root, merge_video_chimeric.stable_case_key(candidate_path))
    tools.make_dirs(work_dir)
    _drop_duplicate_candidate_subtitles(winner, loser, work_dir)
    readings = {zone["zone"]: {"offset_ms": zone["offset_ms"]} for zone in zones}

    # Same drop as the plain constant-offset plan, same reason: no audio track survives
    # verification against a master that shares no language with it.
    track_plans = {}
    for track_language, audio in merge_video_chimeric.iterate_candidate_audios(loser):
        tools.log_line(f"repair: visual_fallback_audio_dropped stream={audio['StreamOrder']} "
                       f"language={track_language} reason=no_language_in_common_to_verify_against "
                       f"for {candidate_path}\n")

    reference_pieces, adjustments, _ = orch.track_pieces(zones, fills, readings, None, timeline_ms)
    for adjustment in adjustments:
        tools.log_line(f"repair: plan_edge_adjustment stream=reference "
                       f"zone={adjustment['zone']} kind={adjustment['kind']} "
                       f"master_fill_ms={adjustment['master_fill_ms']}\n")
    chapters_path, _ = merge_video_chimeric.build_delivered_chapters(
        winner.filePath, candidate_path, reference_pieces, None, timeline_ms, work_dir)

    seam = getattr(loser, merge_video_repair.REPAIR_SEAM_ATTRIBUTE, None)
    job_start_utc = ((seam or {}).get("job_start_utc")
                     or "unstamped(no_repair_seam_standalone_run)")
    n_divergent = sum(1 for fill in fills if fill.get("status") == "video_divergent")
    marker = f"visual_fallback_zones:n={len(zones)}" + (f":div={n_divergent}" if n_divergent
                                                        else "")

    plan = {
        "kind": "orchestrator_visual_fallback_zones", "language": language,
        "reference_stream": None, "quantum_ms": None, "master_path": winner.filePath,
        "decided_by": "merge_video_visual_fallback.visual_fallback_merge",
        "segments_dropped_unusable": 0, "speed_margin": None, "speed_engine": None,
        "speed_margin_absent_reason": "no_rate_relation",
        "segments": [{"master_start_ms": zone["master_start_ms"],
                      "master_end_ms": zone["master_end_ms"],
                      "candidate_offset_ms": zone["offset_ms"],
                      "candidate_offset_ms_by_stream": {}} for zone in zones],
        "track_plans": track_plans, "reference_pieces": reference_pieces,
        "marker": marker, "chapters_path": chapters_path, "speed_ratio": None,
        "speed_ratio_exact": None, "rate_source": None, "resample_gate": None,
        "repair_deadline": repair_deadline,
        "video_anchored": {"offset_frames": None, "n_zones": len(zones),
                           # `merge_video_repair.build_repaired_video_object` reads this key
                           # unconditionally whenever `video_anchored` is set, constant or
                           # multi-zone alike (replay-D3 defect 2, measured: every multi-zone
                           # call declined with `KeyError: 'offset_ms'`). `verify_video_anchored`
                           # takes one offset, meaningless for several zones, but harmless here:
                           # this route plans no candidate audio (`track_plans` is always empty,
                           # above), so `verify_video_anchored` iterates zero tracks and never
                           # reads this value for anything -- the first zone's offset is placed
                           # here only so the key exists.
                           "offset_ms": zones[0]["offset_ms"] if zones else Decimal(0),
                           "offsets_ms": [str(zone["offset_ms"]) for zone in zones],
                           "fps": str(fps), "n_divergent_spans": n_divergent,
                           "trigger": trigger},
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
                              f"visual fallback (no common audio language, {len(zones)} picture "
                              f"zones, {n_divergent} taken from the master): marker '{marker}', "
                              f"master {winner.filePath} untouched, file {out_path}",
                              detail={"out_path": out_path, "video_anchored": plan["video_anchored"]})
    return repaired_obj


def _apply_constant_plan(winner, loser, result, language, repair_deadline, work_root):
    """Build and deliver the one-zone video-anchored plan; return a repaired object or None.

    When a picture-divergent span (chantier E, item 1) is found inside that single zone, the
    plan is promoted to `_apply_zoned_plan`'s multi-zone shape instead: same offset throughout,
    but the divergent span's own frames come from the master.
    """
    import merge_video_chimeric
    import merge_video_repair
    import repair_orchestrator as orch

    candidate_path = loser.filePath
    shift = -result.offset_frames
    marker = f"visual_fallback:{shift:+d}"
    picture_ms = -result.candidate_track_delay_ms
    offset_ms = orch._decimal(picture_ms)
    timeline_ms = merge_video_chimeric.get_master_timeline_length_ms(winner)

    zones = [{"master_start_ms": Decimal(0), "master_end_ms": timeline_ms,
             "offset_ms": offset_ms, "offset_frames": result.offset_frames, "zone": 0}]
    zones, fills = _carve_divergence(winner, loser, result.fps, zones, [], work_root,
                                     repair_deadline)
    if len(zones) > 1 or fills:
        return _apply_zoned_plan(winner, loser, zones, fills, result.fps,
                                 "visual_fallback_no_common_language_divergent", language,
                                 repair_deadline, work_root)

    work_dir = path.join(work_root, merge_video_chimeric.stable_case_key(candidate_path))
    tools.make_dirs(work_dir)
    _drop_duplicate_candidate_subtitles(winner, loser, work_dir)

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


def _apply_speed_plan(winner, loser, speed_result, language, repair_deadline, work_root):
    """Build and deliver a one-zone video-anchored plan with a constant-ratio speed correction
    applied to every candidate track first; return a repaired object or None.

    Same shape as `_apply_constant_plan`'s one-zone plan, through the same build and delivery
    path, except the plan also carries the evidence `merge_video_repair.speed_plan_evidence`
    requires before `build_repaired_video_object` applies `speed_result.ratio` to the loser's
    audio (exact-fraction `asetrate`) and subtitles (linear retime, both already wired in
    `assemble_on_master_timeline` behind its own `speed_ratio` parameter) --
    `video_offset_plan.RATE_SOURCE_VISUAL` as the `rate_source`, a `resample_gate` built from
    this module's own matched-span coverage, `speed_ratio_exact` the snapped named fraction. The
    candidate-side offset is expressed in the candidate's OWN pre-resample clock
    (`speed_result.candidate_offset_ms`); `assemble_on_master_timeline` multiplies it by
    `speed_ratio` itself to land on the master's rescaled clock, exactly as it already does for
    `repair_orchestrator.rate_arm`'s own audio-chromaprint reading -- the plan is expressed once,
    in master time, and the speed correction is applied ahead of (not after) that geometry.
    """
    import merge_video_chimeric
    import merge_video_repair
    import repair_orchestrator as orch

    candidate_path = loser.filePath
    shift = -speed_result.residual_frames
    ratio = speed_result.ratio
    ratio_name = speed_result.ratio_name or f"{ratio.numerator}/{ratio.denominator}"
    marker = f"visual_fallback_speed:{shift:+d}@{ratio_name}"
    picture_ms = -speed_result.candidate_offset_ms
    offset_ms = orch._decimal(picture_ms)
    timeline_ms = merge_video_chimeric.get_master_timeline_length_ms(winner)
    work_dir = path.join(work_root, merge_video_chimeric.stable_case_key(candidate_path))
    tools.make_dirs(work_dir)
    _drop_duplicate_candidate_subtitles(winner, loser, work_dir)

    # Same drop as `_apply_constant_plan`, same reason: no candidate audio track survives
    # verification against a master that shares no language with it.
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

    engine = "asetrate"
    resample_gate = {"verdict": "confirmed", "span_coverage": speed_result.span_coverage,
                     "engine": engine, "cause": None}
    plan = {
        "kind": "orchestrator_visual_fallback_speed", "language": language,
        "reference_stream": None, "quantum_ms": None, "master_path": winner.filePath,
        "decided_by": "merge_video_visual_fallback.visual_fallback_merge",
        "segments_dropped_unusable": 0, "speed_margin": None, "speed_engine": engine,
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
                           "trigger": "visual_fallback_no_common_language_speed"},
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
                              f"visual fallback speed correction (no common audio language): "
                              f"ratio {ratio} ({ratio_name}), residual offset {shift:+d} "
                              f"frame(s), marker '{marker}', master {winner.filePath} "
                              f"untouched, file {out_path}",
                              detail={"out_path": out_path,
                                     "video_anchored": plan["video_anchored"]})
    return repaired_obj


def _apply_multizone_plan(winner, loser, result, groups, language, repair_deadline, work_root):
    """Build and deliver a multi-zone video-anchored plan; return a repaired object or None.

    Laid out by `video_offset_plan.video_zone_plan`: one candidate zone per stable picture
    offset, the holes between them (and before/after the first/last) filled from the master
    under the blind-span rule -- then, inside each zone, `_carve_divergence` splits out any
    picture-divergent span (chantier E, item 1) as one more master fill. Delivered by the same
    `_apply_zoned_plan` the promoted one-zone plan uses.
    """
    import merge_video_chimeric

    # Same start-time correction the constant-offset plan already folds into its own
    # `candidate_track_delay_ms` (`VideoOffsetResult`'s own property) -- a per-zone `d x frame_ms`
    # alone leaves out the two files' differing video start times (replay-D3 defect 5b/2,
    # measured: a 23 ms error on files whose candidate starts at 0.023 s rather than 0.000 s).
    start_delta_ms = (result.candidate_start_s - result.master_start_s) * 1000
    timeline_ms = merge_video_chimeric.get_master_timeline_length_ms(winner)
    zones, fills = video_offset_plan.video_zone_plan(groups, result.frame_ms, timeline_ms,
                                                      start_delta_ms)
    zones, fills = _carve_divergence(winner, loser, result.fps, zones, fills, work_root,
                                     repair_deadline)
    return _apply_zoned_plan(winner, loser, zones, fills, result.fps,
                             "visual_fallback_no_common_language_multizone", language,
                             repair_deadline, work_root)


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
    result, speed_result = None, None
    try:
        tools.make_dirs(work_root)
        cache_dir = path.join(work_root, "measure")
        tools.make_dirs(cache_dir)
        # Frames, never time: two individually-CFR files at different declared exact rates
        # cannot share one constant frame offset (`measure_video_offset`'s own `check_fps`
        # correctly refuses them for exactly that reason), but they can share one constant
        # SPEED ratio -- the owner's own case, measured by `detect_speed_ratio` instead.
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
            repaired_obj = _apply_speed_plan(winner, loser, speed_result, language,
                                             repair_deadline, work_root)
        elif result.status == video_offset_plan.STATUS_OK:
            repaired_obj = _apply_constant_plan(winner, loser, result, language,
                                                repair_deadline, work_root)
        else:
            groups = video_offset_plan.group_video_zones(result.pairs or [],
                                                          float(result.frame_ms))
            cause = video_offset_plan.check_zone_compatibility(groups, result.master_frames)
            if cause is not None:
                _log_decline(cause, video_a, video_b, f"n_zones={len(groups)}")
                return video_a
            tools.log_always(
                f"repair: visual_fallback_zones n_zones={len(groups)} offsets="
                + ",".join(f"{d:+d}x{len(m)}" for d, m in groups)
                + f" for {loser.filePath}\n")
            repaired_obj = _apply_multizone_plan(winner, loser, result, groups, language,
                                                 repair_deadline, work_root)
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
