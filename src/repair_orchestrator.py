# -*- coding: utf-8 -*-
"""Flat repair orchestrator for a candidate the compatibility check refused.

Each step is launched here and returns a result; the steps never call each other. The
pipeline has four stages:
  0. preconditions: master conformity and self-consistency, comparison track and language
     checks, audio self-contradiction (video-anchored route);
  1/2. similarity gate and resample sweep (`ensemble_similarity_gate`, `rate_arm`): when the
     native alignment is poor, every named ratio is tried and the best admissible one kept;
  3. chimeric processing: whole-track chromaprint aligned by `banded_seed_alignment.b2_align`
     gives the holes, `audio_walk` refines each edge to the millisecond, then the picture
     (`scene_anchor`: scene cuts, pHash anchors, cross scan) pins every cut to a frame;
  4. plan application (`plan_geometry`, `apply_plan`): trims, master fills and tail gap.

Entry point: `merge_video_repair.repair_not_compatible_videos` calls `repair()` once per
refused candidate. `repair()` returns True only when a plan was found and the chimeric file
was written; every False logs a `cause=<token>` with its measurement class
(`ran_conclusive_negative` or `could_not_run`). The path is sequential by design: the cost is
per-track ffmpeg extraction, and the process already runs under a `Pool`.
"""
from decimal import Decimal
from fractions import Fraction
import math
from os import path, remove
import statistics
import time

import audioCorrelation
import alignment_diagnostics
import audio_extract
import banded_seed_alignment
import progress_watchdog
import repair_log
import repair_pool
import tools
import video_offset_plan
# The stage modules. Names used here or reached through this module by other modules are
# imported by name.
from repair_stage1_resample import (ALIGNMENT_COULD_NOT_MEASURE_VERDICTS,
                                    MASTER_AXIS_COVERAGE_FLOOR, _drop_rate_wav,
                                    ensemble_similarity_gate, rate_arm, rate_resample_routing)
from repair_stage3_chimeric import (_speed_chain, _video_duration_ms,
                                    align_transitions_to_zone_lags,
                                    audio_edges, audio_transitions, classify_holes,
                                    coalesce_same_offset_zones, couple_start_delta_ms,
                                    edge_addition_seconds, edge_zones, frame_domain,
                                    hole_on_file_clock, hole_sanity, holes_for_couple,
                                    log_holes_against_walk, log_partial_plan,
                                    max_holes_per_couple, picture_steps_in_continuous_zones,
                                    reference_walk, union_holes)
from repair_stage4_plan import (ZONE_EDGE_MARGIN_S, _track_timing, apply_picture_offsets,
                                apply_plan, plan_geometry, remeasure_at_other_levels,
                                track_offsets, track_pieces, written_edge_seconds,
                                zone_picture_offsets_from_hole_anchors)

MODALITY = "repair_orchestrator"

# ---------------------------------------------------------------------------
# Derived constants: each one names what it is derived from.
# ---------------------------------------------------------------------------

# Holes closer than this share one scene-detection pass (`cluster_holes`); each hole keeps its
# own bounds and step. Equal to the frame-exact resolver's search reach, so one pass serves both.
try:
    import scene_anchor as _scene_anchor
    HOLE_MERGE_WINDOW_SECONDS = float(_scene_anchor.SCENE_SEARCH_WINDOW_SECONDS_DEFAULT)
    _HOLE_MERGE_SOURCE = "scene_anchor.SCENE_SEARCH_WINDOW_SECONDS_DEFAULT"
except Exception:                                                        # noqa: BLE001
    HOLE_MERGE_WINDOW_SECONDS = 10.0
    _HOLE_MERGE_SOURCE = "literal fallback -- scene_anchor unimportable"

# Step tolerance between two measurements of one cut (a walk step against a union hole): couples
# measuring the same cut differ by up to ~1.02 quanta, so steps are compared against 1.5 quanta.
INTERCOUPLE_STEP_TOLERANCE_QUANTA = 1
INTERCOUPLE_STEP_TOLERANCE_SLACK = 1.5

# No time limit anywhere in the repair: only the progress watchdogs stop a step
# (decision: no-budget-watchdog).
# Hole sanity bound: an interior hole wider than this is not searched (real holes stay under ~30 s).
INTERIOR_HOLE_MAX_SPAN_S = 300.0

# Cap on holes per couple, per started 30-min slice of the master's video: each hole costs a
# frame-exact search. About twice the highest count seen on real media (21), so the choice of
# comparison language never decides a refusal. Exceeding it is `could_not_run`.
MAX_HOLES_PER_COUPLE = 45
HOLE_BUDGET_SLICE_S = 1800.0

# Measurement class, logged beside every cause token.
CLASS_CONCLUSIVE = "ran_conclusive_negative"
CLASS_COULD_NOT_RUN = "could_not_run"

# Closed cause vocabulary with each token's class; an unknown token logs as `unclassified`.
DECLINE_CAUSES = {
    # step 1 (a self-contradicting master routes to `video_anchored_route`).
    # No match between the two files' scene changes proves different content:
    "video_content_mismatch": CLASS_CONCLUSIVE,
    # The chimeric build itself refused with no cause of its own (apply_plan's default):
    "chimeric_build_refused": CLASS_COULD_NOT_RUN,
    # A comparison track whose strict decode fails routes to the video when both pictures are
    # sound; when one is not (`video_unreliable`), the pair is refused with the decoder's line.
    "comparison_track_corrupt": CLASS_CONCLUSIVE,
    # No candidate track carries the master's comparison-language content, measured by
    # fingerprint on every track (`audio_tag_conflict` is a routing, not a terminal: see
    # ROUTING_SIGNALS).
    "no_common_language_after_tag_check": CLASS_CONCLUSIVE,
    # The arbiter could not run or could not conclude -- nothing proven about the pair:
    "video_fps_mismatch": CLASS_COULD_NOT_RUN,
    "video_offset_not_constant": CLASS_COULD_NOT_RUN,
    "video_offset_coverage_incomplete": CLASS_COULD_NOT_RUN,
    "video_probe_failed": CLASS_COULD_NOT_RUN,
    "video_decode_failed": CLASS_COULD_NOT_RUN,
    # The master's content ends >= 300 s before the candidate's and never resumes:
    "master_cut_short": CLASS_CONCLUSIVE,
    # Step 0: the master fails its own conformity check (error severity):
    "master_nonconformant": CLASS_CONCLUSIVE,
    # step 2
    # Low similarity is not a proven negative (a mistagged track can read 0.15):
    "similarity_unrecoverable_by_resample": CLASS_COULD_NOT_RUN,
    "rate_sweep_no_sample_rate": CLASS_COULD_NOT_RUN,
    # The rate arm had no comparison WAV or could not build the candidate's resample:
    "rate_arm_unmeasured": CLASS_COULD_NOT_RUN,
    # step 3, measurement
    "track_duration_unmeasurable": CLASS_COULD_NOT_RUN,
    "fingerprinting_raised": CLASS_COULD_NOT_RUN,
    "alignment_degenerate_input": CLASS_COULD_NOT_RUN,
    "alignment_no_anchored_runs": CLASS_COULD_NOT_RUN,
    "alignment_all_seeds_refused": CLASS_COULD_NOT_RUN,
    "alignment_segments_below_duration_floor": CLASS_COULD_NOT_RUN,
    # A structural bound on the problem size, not a measurement about the pair (see
    # MAX_HOLES_PER_COUPLE):
    "hole_count_exceeds_resolver_budget": CLASS_COULD_NOT_RUN,
    # Every couple falls under MASTER_AXIS_COVERAGE_FLOOR:
    "alignment_coverage_below_floor": CLASS_COULD_NOT_RUN,
    # The speed-corrected candidate could not be built at the confirmed factor:
    "rate_resample_unbuildable": CLASS_COULD_NOT_RUN,
    # step 4: the frame-exact search could not establish a boundary on at least one hole.
    "hole_resolution_declined": CLASS_COULD_NOT_RUN,
    # A watchdog stopped a step that made no progress, and hole sanity bounds: facts about this
    # run or the alignment, not the pair.
    "step_stalled": CLASS_COULD_NOT_RUN,
    "hole_outside_master_timeline": CLASS_COULD_NOT_RUN,
    "hole_step_exceeds_duration": CLASS_COULD_NOT_RUN,
    "interior_hole_exceeds_budget": CLASS_COULD_NOT_RUN,
    # step 5, plan application. The pair's frame grid could not be read (pair with no hole):
    "frame_domain_unmeasured": CLASS_COULD_NOT_RUN,
    # No zone of the comparison track has a sample-measurable offset:
    "plan_offset_unmeasurable": CLASS_COULD_NOT_RUN,
    # The resolved holes cover the whole master timeline: nothing is read from the candidate.
    "plan_reads_no_candidate_content": CLASS_CONCLUSIVE,
    # The build returned and the temporary chimeric file is not on disk.
    "plan_application_no_file": CLASS_COULD_NOT_RUN,
    # The audio walk could not read the comparison tracks or found no level:
    "audio_walk_unavailable": CLASS_COULD_NOT_RUN,
    # A change point whose edges the 20 ms and 100 ms profiles could not read:
    "audio_step_unlocalised": CLASS_COULD_NOT_RUN,
    # The walk's level step and its 20 ms edges disagree on one transition's width:
    "hole_width_contradicts_audio_step": CLASS_COULD_NOT_RUN,
    # A sub-quantum step the video could neither place nor rule out:
    "sub_quantum_step_video_ambiguous": CLASS_COULD_NOT_RUN,
    # A step at or above the quantum the video could neither place nor rule out (no compatible
    # anchor, a static/black span, a geometry mismatch...): the video decides, so a declined
    # reading is never delivered as an audio-only cut.
    "video_cut_undetermined": CLASS_COULD_NOT_RUN,
    # The one-anchor walk on a head/tail edge could neither place a boundary nor confirm there
    # is none: the same rule as an interior cut, never delivered as an audio-only edge.
    "video_edge_undetermined": CLASS_COULD_NOT_RUN,
    # The audio's transitions or edges do not tile the timeline (a plan defect, not the pair's):
    "audio_transitions_overlap": CLASS_COULD_NOT_RUN,
    # A hole status outside the four HOLE_STATUSES_WITH_FRAMES values reached the branch that
    # places a transition: an internal state the plan does not recognize, never delivered.
    "hole_status_unhandled": CLASS_COULD_NOT_RUN,
    # The assembly's own refusals (`merge_video_chimeric` / `merge_video_repair`). A refusal of
    # the built file measures the product, not the pair.
    "alignment_contradicts_plan": CLASS_CONCLUSIVE,
    "delivery_offset_exceeds_tolerance": CLASS_COULD_NOT_RUN,  # our product, never the pair
    # `merge_video_chimeric.verify_output_file`: the produced file does not match what was built.
    "output_check_mismatch": CLASS_COULD_NOT_RUN,
    "master_audio_complement_short": CLASS_CONCLUSIVE,
    "master_duration_sources_disagree": CLASS_CONCLUSIVE,
    "candidate_admission_window_exceeded": CLASS_CONCLUSIVE,
    "speed_transform_not_validated": CLASS_COULD_NOT_RUN,
    "plan_not_contiguous": CLASS_COULD_NOT_RUN,
    "plan_piece_empty_or_inverted": CLASS_COULD_NOT_RUN,
    "plan_end_not_master_timeline": CLASS_COULD_NOT_RUN,
    # A delivery probe landed where a compared track has no audio:
    "probe_reads_no_audio": CLASS_COULD_NOT_RUN,
    # The plan does not give back the master's frame count and the picture-only plan could not
    # replace it: a defect of our plan, not a fact about the pair.
    "frame_count_mismatch": CLASS_COULD_NOT_RUN,
    # Family tokens of CAUSE_FAMILIES (below), when raised as such.
    "boundary_unresolved": CLASS_COULD_NOT_RUN,
    "audio_unmeasurable": CLASS_COULD_NOT_RUN,
    "precondition_failed": CLASS_CONCLUSIVE,
    "plan_invariant_broken": CLASS_COULD_NOT_RUN,
    "video_route_unresolved": CLASS_COULD_NOT_RUN,
    "watchdog_stopped": CLASS_COULD_NOT_RUN,
}

# Surfaced vocabulary: detailed causes that share one meaning are reported under one family
# token, the detailed cause kept in a named field (`cause=<family> <field>=<detailed>`).
# Internal decisions still read the detailed cause.
CAUSE_FAMILIES = {
    # A step made no progress, or the problem passed a structural bound: a fact about this run,
    # not the pair.
    "watchdog_stopped": ("checkpoint", (
        "step_stalled", "hole_count_exceeds_resolver_budget", "interior_hole_exceeds_budget")),
    # The picture could not seat a frame-exact boundary.
    "boundary_unresolved": ("kind", (
        "hole_resolution_declined", "video_cut_undetermined",
        "sub_quantum_step_video_ambiguous", "video_edge_undetermined",
        "hole_status_unhandled")),
    # The audio measurement gave nothing usable.
    "audio_unmeasurable": ("stage", (
        "audio_step_unlocalised", "hole_width_contradicts_audio_step",
        "frame_domain_unmeasured", "plan_offset_unmeasurable", "alignment_degenerate_input",
        "alignment_no_anchored_runs", "alignment_all_seeds_refused",
        "alignment_segments_below_duration_floor", "alignment_coverage_below_floor",
        "audio_walk_unavailable", "track_duration_unmeasurable")),
    # The master's or the candidate's own shape rules the pair out before any comparison.
    "precondition_failed": ("reason", (
        "master_audio_complement_short", "master_duration_sources_disagree",
        "candidate_admission_window_exceeded")),
    # The picture-only route could not run or could not conclude.
    "video_route_unresolved": ("reason", (
        "video_fps_mismatch", "video_offset_not_constant", "video_offset_coverage_incomplete",
        "video_probe_failed", "video_decode_failed")),
    # Our own plan builder produced an impossible shape: a defect of the plan, not the pair.
    "plan_invariant_broken": ("kind", (
        "plan_not_contiguous", "plan_piece_empty_or_inverted", "plan_end_not_master_timeline",
        "audio_transitions_overlap", "hole_outside_master_timeline",
        "hole_step_exceeds_duration", "plan_application_no_file")),
}
_FAMILY_OF = {detailed: (family, field) for family, (field, members) in CAUSE_FAMILIES.items()
              for detailed in members}
assert all(DECLINE_CAUSES[family] == DECLINE_CAUSES[detailed]
           for detailed, (family, _field) in _FAMILY_OF.items()), (
    "repair_orchestrator: a cause family mixes measurement classes")


def surfaced_cause(cause):
    """Return (token, field, detail) for a cause as reported: its family when it has one.

    `field` and `detail` are None for a cause without a family.
    """
    if cause in _FAMILY_OF:
        family, field = _FAMILY_OF[cause]
        return family, field, cause
    return cause, None, None


# Hole-result vocabulary. The audio proposes a zone, the video decides; a hole the video crosses
# under one shift closes as `no_cut_confirmed`. Every status but `declined` carries exact frames:
#   resolved                          interior, the cut pinned on both files
#   no_cut_confirmed                  the video crossed the hole under one shift
#   boundary_pinned_to_ambiguous_zone_end
#                                     self-similar span, pinned at the end of the ambiguous zone
#   sustained_mismatch                edge: divergent content, replace
#   master_exhausted                  edge: candidate excess, trim
#   candidate_exhausted               edge: master addition
#   declined                          no boundary established; named reason
HOLE_RESOLVED = "resolved"
HOLE_NO_CUT_CONFIRMED = "no_cut_confirmed"
HOLE_PINNED_TO_AMBIGUOUS_ZONE_END = "boundary_pinned_to_ambiguous_zone_end"
EDGE_SUSTAINED_MISMATCH = "sustained_mismatch"
EDGE_MASTER_EXHAUSTED = "master_exhausted"
EDGE_CANDIDATE_EXHAUSTED = "candidate_exhausted"
HOLE_DECLINED = "declined"
EDGE_TERMINATIONS = (EDGE_SUSTAINED_MISMATCH, EDGE_MASTER_EXHAUSTED, EDGE_CANDIDATE_EXHAUSTED)
HOLE_STATUSES_WITH_FRAMES = (HOLE_RESOLVED, HOLE_NO_CUT_CONFIRMED, HOLE_PINNED_TO_AMBIGUOUS_ZONE_END
                             ) + EDGE_TERMINATIONS

# Closed set: `tools/validate_merge_plan.py` parses exactly these tokens.
WHY_TOKEN = {"head": "head_gap", "interior": "interior_bracket", "tail": "tail_gap"}


# ---------------------------------------------------------------------------
# LOGGING -- one launch line and one result line per step
# ---------------------------------------------------------------------------

def _fields(pairs):
    return " ".join(f"{key}={value}" for key, value in pairs)


def step_launch(step, **fields):
    """Log a step's launch line (dev only), before the work starts so a hang stays attributable."""
    _STEP_STARTED[step] = time.monotonic()
    tools.dev_log(f"orchestrator: launch step={step} "
                  f"{_fields(sorted(fields.items()))}\n")


# When each launched step began: its result line carries `elapsed_s`.
_STEP_STARTED = {}


def step_result(step, **fields):
    """Log a step's result line with its elapsed time; called on every exit, refusals included."""
    started = _STEP_STARTED.pop(step, None)
    if started is not None:
        fields = dict(fields, elapsed_s=round(time.monotonic() - started, 2))
    tools.dev_log(f"orchestrator: result step={step} "
                  f"{_fields(sorted(fields.items()))}\n")


def _plan_line(kind, candidate_path, **fields):
    """Write the `repair: plan <kind> ...` line that marks a job log, once per candidate run.

    `merge_plan_report.is_job_log` keys on it. Skipped when the assembly already wrote its own
    plan line (with geometry) since this run's `_PLAN_LINE_MARK`.
    """
    mark = _PLAN_LINE_MARK.get(candidate_path)
    if mark is not None and any(line.startswith("repair: plan ")
                                for entry in tools.logs[mark:]
                                for line in str(entry).splitlines()):
        tools.dev_log(f"orchestrator: plan line kind={kind} not repeated for {candidate_path} "
                      f"-- the build's plan line (with its geometry) is this run's one line\n")
        return
    if "cause" in fields:
        token, field, detail = surfaced_cause(fields["cause"])
        fields["cause"] = token
        if field:
            fields[field] = detail
    tools.log_line(f"repair: plan {kind} build={repair_log.build_sha()} orchestrator=1 "
                      f"{_fields(sorted(fields.items()))} for {candidate_path}\n")


# Index in `tools.logs` where each candidate's current `repair()` run began.
_PLAN_LINE_MARK = {}


def log_measurement_class(candidate_path, cause):
    """Log the measurement class of a cause token; an unknown token is logged as `unclassified`.

    A detailed cause is logged under its family (`surfaced_cause`) with the detail beside it.
    """
    if cause not in DECLINE_CAUSES:
        tools.log_always(f"repair: orchestrator UNVOCABULARISED cause={cause} for "
                         f"{candidate_path} -- this token is not in DECLINE_CAUSES and has no "
                         f"measurement class; add it there. The refusal below stands.\n")
    token, field, detail = surfaced_cause(cause)
    tools.log_always(f"repair: orchestrator cause={token} "
                     + (f"{field}={detail} " if field else "")
                     + f"measurement={DECLINE_CAUSES.get(cause, 'unclassified')} "
                     f"for {candidate_path}\n")


def _terminal(candidate_path, outcome, cause, reason, detail=None):
    """Record the terminal per-candidate refusal with its cause token and class; returns False.

    Routed through `merge_video_repair.record` when importable (late import avoids a cycle).
    """
    log_measurement_class(candidate_path, cause)
    try:
        import merge_video_repair
    except Exception as error:                                           # noqa: BLE001
        tools.log_always(f"repair: {outcome} cause={cause} for {candidate_path}: {reason}\n")
        tools.dev_log(f"orchestrator: merge_video_repair unimportable for the terminal "
                      f"record ({type(error).__name__}) -- the line above was written "
                      f"directly\n")
    else:
        merge_video_repair.record(candidate_path, outcome, reason, detail=detail, cause=cause)
    return False


# ---------------------------------------------------------------------------
# Couples and fingerprints
# ---------------------------------------------------------------------------

def _track_duration_seconds(video_obj, language, stream_order):
    """Return this track's own duration in seconds, or None when it cannot be read.

    Falls back to the container's duration via ffprobe when the track carries none.
    """
    audios = getattr(video_obj, "audios", None) or {}
    for entry in audios.get(language) or []:
        if entry.get("StreamOrder") != stream_order:
            continue
        for key in ("Duration", "duration"):
            if key in entry:
                try:
                    return float(entry[key])
                except (TypeError, ValueError):
                    pass
    tools.dev_log(f"orchestrator: ffprobe duration call file={video_obj.filePath} "
                  f"stream_order={stream_order}\n")
    try:
        with repair_log.announced("orchestrator", "ffprobe", video_obj.filePath) as call:
            completed = progress_watchdog.run(
                [tools.software["ffprobe"], "-v", "error", "-show_entries",
                 "format=duration", "-of", "default=nw=1:nk=1", video_obj.filePath],
                "ffprobe_duration", text=True)
            call["exit"] = completed.returncode
        return float(completed.stdout.strip())
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"orchestrator: ffprobe duration unreadable for "
                      f"{video_obj.filePath} stream_order={stream_order}: "
                      f"{type(error).__name__}\n")
        return None


def comparison_sample_rate(master_obj, candidate_obj, language):
    """Return the pair's comparison sample rate: the pair's lowest rate, capped at 44100 Hz.

    Uses `video.get_less_sampling_rate`; falls back to 44100 when the rate cannot be derived.
    """
    try:
        import video as _video
        rate = int(_video.get_less_sampling_rate(master_obj.audios[language],
                                                  candidate_obj.audios[language]))
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"orchestrator: comparison grid underivable "
                      f"({type(error).__name__}); using 44100\n")
        rate = 44100
    return 44100 if rate > 44100 else rate


def enumerate_couples(master_obj, candidate_obj, language):
    """Return every (master stream, candidate stream) pair of the comparison language.

    Each pair is aligned independently and the alignments are cross-checked.
    """
    master_streams = audio_extract.streams_for(master_obj, language)
    candidate_streams = audio_extract.streams_for(candidate_obj, language)
    return [(m, c) for m in master_streams for c in candidate_streams]


# Fingerprint quantum: Chromaprint resamples to 11025 Hz and hops 4096 // 3 = 1365 samples, so
# points are 123.8095 ms apart. `duration / len(points)` is wrong: fpcalc drops ~21 points at the
# end of the stream.
CHROMAPRINT_SAMPLE_RATE = 11025
CHROMAPRINT_HOP_SAMPLES = 4096 // 3
CHROMAPRINT_HOP_MS = CHROMAPRINT_HOP_SAMPLES * 1000.0 / CHROMAPRINT_SAMPLE_RATE


def fingerprint_track(video_obj, language, stream_order, side, work_dir, sample_rate,
                      duration_seconds, audio_filter=None, output_duration_seconds=None,
                      measures=None, keep_wav=False):
    """Fingerprint one whole track with fpcalc.

    Each track is extracted to its own full duration so a one-sided tail stays visible.
    `duration_seconds` bounds what is read from the source; with `audio_filter` (a speed
    resample) pass `output_duration_seconds`, the length of the corrected output, or fpcalc
    truncates the tail.

    Returns:
        (points, CHROMAPRINT_HOP_MS), or (None, None) when the track could not be read.
    """
    if output_duration_seconds is None:
        output_duration_seconds = duration_seconds
    wav = path.join(work_dir, f"orch_{side}_{stream_order}.wav")
    try:
        audio_extract.extract_audio_window(video_obj.filePath, stream_order, 0.0,
                                           duration_seconds, wav, sample_rate,
                                           audio_filter=audio_filter)
        with repair_log.announced("orchestrator", "fpcalc", wav) as call:
            points = audioCorrelation.calculate_fingerprints(wav, length=output_duration_seconds)
            call["exit"] = 0
        if audio_filter is None and points:
            # Cached for the delivery gate, which compares this track with every rebuilt one.
            import merge_video_decode_once
            merge_video_decode_once.note("comparison_rate", int(sample_rate))
            merge_video_decode_once.put(
                "fingerprint", video_obj.filePath, int(stream_order),
                merge_video_decode_once.fingerprint_params(sample_rate),
                {"points": list(points), "duration_s": float(output_duration_seconds)})
        if measures is not None:
            measures["content_end_s"] = wav_content_end_s(wav)
    except (progress_watchdog.step_stalled, audio_extract.StrictDecodeFailed):
        # A corrupt source is handled by the caller's policy, not as a fingerprint failure.
        raise
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"orchestrator: fingerprint_track raised on "
                      f"{video_obj.filePath} stream_order={stream_order}: "
                      f"{type(error).__name__}: {error}\n")
        return None, None
    finally:
        # `keep_wav`: the rate arm reuses this extraction and deletes it itself.
        if keep_wav and measures is not None and path.exists(wav):
            measures["wav"] = wav
            wav = None
        # Safe only because the WAV path is unique per pair, side and stream.
        try:
            if wav is not None:
                remove(wav)
        except OSError:
            pass
    if not points:
        return None, None
    return points, CHROMAPRINT_HOP_MS


# ---------------------------------------------------------------------------
# One-sided tail
# ---------------------------------------------------------------------------
# After the last common instant, content may continue on one side only. At least
# TAIL_ONE_SIDED_MIN_S of it: a short candidate is filled from the master (no size cap); a short
# master declines `master_cut_short`. The master's content is read to the end of its own track,
# since audio past its video still reaches the timeline's end.
TAIL_ONE_SIDED_MIN_S = 300.0
CONTENT_BLOCK_S = 0.1
CONTENT_READ_CHUNK_S = 60.0
# Chunks probed backwards past the video's end before calling a master overrun silent.
OVERRUN_PROBE_CHUNKS = 5


def _audible_block_end(samples, rate, offset_s):
    """End (s) of the last CONTENT_BLOCK_S block at or above audio_walk.AUDIBLE_DB, or None."""
    import numpy
    import audio_walk
    block = max(1, int(rate * CONTENT_BLOCK_S))
    count = len(samples) // block
    if not count:
        return None
    rms = numpy.sqrt(numpy.mean(samples[:count * block].reshape(count, block) ** 2, axis=1))
    loud = numpy.nonzero(20 * numpy.log10(numpy.maximum(rms, 1e-12)) >= audio_walk.AUDIBLE_DB)[0]
    return None if not len(loud) else offset_s + (int(loud[-1]) + 1) * block / rate


def wav_content_end_s(wav_path):
    """Return the end (s) of a mono 16-bit WAV's last audible block, or None.

    Read backwards in chunks; None when unreadable or silent throughout.
    """
    import numpy
    import wave
    try:
        with wave.open(wav_path, "rb") as reader:
            if reader.getsampwidth() != 2 or reader.getnchannels() != 1:
                return None
            rate, end = reader.getframerate(), reader.getnframes()
            block = max(1, int(rate * CONTENT_BLOCK_S))
            chunk = max(block, int(rate * CONTENT_READ_CHUNK_S) // block * block)
            while end > 0:
                start = max(0, end - chunk)
                reader.setpos(start)
                samples = numpy.frombuffer(reader.readframes(end - start),
                                           dtype="<i2").astype(numpy.float64) / 32768.0
                found = _audible_block_end(samples, rate, start / rate)
                if found is not None:
                    return found
                end = start
    except (OSError, EOFError, ValueError) as error:
        tools.dev_log(f"orchestrator: content_end unreadable {path.basename(wav_path)}: "
                      f"{type(error).__name__}\n")
    return None


def overrun_content_end_s(video_obj, stream_order, track_s, timeline_s):
    """Return the content end of a master track past its video's end, or None.

    Probes at most OVERRUN_PROBE_CHUNKS chunks backwards from the track's end.
    """
    import merge_video_chimeric
    rate = 8000
    end = track_s
    for _ in range(OVERRUN_PROBE_CHUNKS):
        start = max(timeline_s, end - CONTENT_READ_CHUNK_S)
        if end - start < 1.0:
            break
        try:
            samples = merge_video_chimeric.read_mono_samples(
                video_obj.filePath, f"0:{stream_order}", Decimal(str(start * 1000.0)),
                Decimal(str((end - start) * 1000.0)), rate)
        except merge_video_chimeric.chimeric_error:
            samples = None
        found = None if samples is None else _audible_block_end(samples, rate, start)
        if found is not None:
            return found
        end = start
    return None


def tail_content_verdict(couples, timeline_s):
    """Decide the one-sided-tail verdict from per-couple readings (pure).

    Args:
        couples: dicts with `couple`, `last_common_master_s`, `last_common_candidate_s`,
            `master_content_end_s`, `candidate_content_end_s`.
        timeline_s: master video duration; master content is capped at it.

    Returns:
        (verdict, per_couple): verdict is `master_cut_short`, `candidate_short` or None. All
        couples must agree; an unmeasured couple yields None.
    """
    readings = []
    for couple in couples:
        values = [couple.get(key) for key in ("last_common_master_s", "last_common_candidate_s",
                                             "master_content_end_s", "candidate_content_end_s")]
        if any(value is None for value in values):
            readings.append((couple.get("couple"), None, None, None))
            continue
        last_m, last_c, end_m, end_c = values
        master_after = max(0.0, min(end_m, timeline_s) - last_m)
        candidate_after = max(0.0, end_c - last_c)
        verdict = ("master_cut_short" if candidate_after - master_after >= TAIL_ONE_SIDED_MIN_S
                   else "candidate_short" if master_after - candidate_after >= TAIL_ONE_SIDED_MIN_S
                   else None)
        readings.append((couple.get("couple"), verdict, round(master_after, 3),
                         round(candidate_after, 3)))
    verdicts = {reading[1] for reading in readings}
    if not readings or any(reading[2] is None for reading in readings) or len(verdicts) != 1:
        return None, readings
    return verdicts.pop(), readings


def tail_decision(primed, master_obj, language, candidate_path):
    """Apply the one-sided-tail rule on the final alignment (after any rate correction).

    Returns:
        None, or ("master_cut_short", reason). A short candidate tail is only logged; the plan
        fills it as an ordinary tail hole.
    """
    timeline_ms = _video_duration_ms(master_obj)
    if timeline_ms is None:
        return None
    ends = tail_couples(primed)
    verdict, readings = tail_content_verdict(ends, float(timeline_ms) / 1000.0)
    step_result("tail_content", candidate=candidate_path, verdict=verdict, readings=readings,
                min_one_sided_s=TAIL_ONE_SIDED_MIN_S, primed_at=primed["factor_label"])
    if verdict == "candidate_short":
        tools.log_always(f"repair: candidate_short_tail readings={readings} "
                         f"factor={primed['factor_label']} -- the candidate's content ends "
                         f">= {TAIL_ONE_SIDED_MIN_S} s before the master's: an ordinary tail "
                         f"hole, filled from the master, no size cap, for "
                         f"{candidate_path}\n")
    if verdict != "master_cut_short":
        return None
    return "master_cut_short", (
        f"the master's {language} content ends before the candidate's and does not resume, "
        f"read on the alignment at factor {primed['factor_label']}: per couple (couple, verdict, "
        f"master content after the last common instant s, candidate content after it s) "
        f"{readings}; measured ends "
        + "; ".join(f"{c['couple']} master {c['master_content_end_s']} s / candidate "
                    f"{c['candidate_content_end_s']} s after the last common instant "
                    f"{c['last_common_master_s']} s" for c in ends)
        + f" of a {round(float(timeline_ms) / 1000.0, 3)} s master video -- the master cannot "
          f"give the candidate its end")


def tail_couples(primed):
    """Build the per-couple inputs of `tail_content_verdict` from the prime's measurements."""
    couples = []
    for master_stream, candidate_stream in primed["couples"]:
        name = f"{master_stream}x{candidate_stream}"
        alignment = primed["alignments"].get(name) or {}
        _zones, detail = coalesce_same_offset_zones(alignment.get("zones") or [],
                                                    alignment.get("zones_detail") or [])
        ends = primed.get("content_end") or {}
        couples.append({
            "couple": name,
            "last_common_master_s": (detail[-1]["master_ms"][1] / 1000.0 if detail else None),
            "last_common_candidate_s": (detail[-1]["candidate_ms"][1] / 1000.0
                                        if detail else None),
            "master_content_end_s": ends.get(("master", master_stream)),
            "candidate_content_end_s": ends.get(("candidate", candidate_stream))})
    return couples


# ---------------------------------------------------------------------------
# Exact numbers
# ---------------------------------------------------------------------------

def _decimal(value):
    """Convert a Fraction (or any value whose str() is exact) to a Decimal."""
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(str(value))


# ---------------------------------------------------------------------------
# Step 3: chimeric plan
# ---------------------------------------------------------------------------

@repair_log.timed_phase("orchestrator", "chimeric",
                        lambda factor, language, master_obj, candidate_obj, *a, **k:
                        candidate_obj.filePath)
def chimeric(factor, language, master_obj, candidate_obj, work_dir, primed,
             sweep_gate=None, resample_routing=None):
    """Build the chimeric plan from primed couples and apply it.

    Receives ready fingerprints and alignments (already speed-corrected at a confirmed
    factor) and never resamples; `factor` sets the frame domain's time scale. Order:
    one-sided-tail check; holes per couple (blind or low-coverage couples screened);
    cross-check; holes put on the file's clock and united; frame domain; audio walk,
    transitions and edges; `plan_geometry`; `apply_plan`.

    Returns:
        (ok, cause, reason, detail)
    """
    candidate_path = candidate_obj.filePath
    if factor is None or factor == 1:
        step_result("no_filter", candidate=candidate_path, speed_factor=factor,
                    rule="ADDENDUM_6_no_filter_without_speed_change")
    step_result("chimeric_input", candidate=candidate_path, n_couples=len(primed["couples"]),
                primed_at=primed["factor_label"],
                rule="ADDENDUM_21_6_chimeric_receives_every_couple_ready_and_never_resamples")

    # ---- one-sided tail -----------------------------------------------------
    refusal = tail_decision(primed, master_obj, language, candidate_path)
    if refusal is not None:
        return False, refusal[0], refusal[1], None

    couple_results = []
    for master_stream, candidate_stream in primed["couples"]:
        couple = f"{master_stream}x{candidate_stream}"
        alignment = primed["alignments"][couple]
        # `single_segment_no_cut` is a success; its head/tail holes may still be real.
        if alignment["verdict"] in ALIGNMENT_COULD_NOT_MEASURE_VERDICTS:
            tools.dev_log(f"orchestrator: couple {couple} could not be aligned "
                          f"({alignment['verdict']}) -- recorded, and the remaining couples "
                          f"still run: one blind track is not a verdict about the pair\n")
            continue
        # A couple under the coverage floor contributes no holes and no cross-check events.
        couple_coverage = alignment.get("master_axis_coverage_fraction")
        if couple_coverage is None or couple_coverage < MASTER_AXIS_COVERAGE_FLOOR:
            step_result("couple_screened", candidate=candidate_path, couple=couple,
                        verdict=alignment["verdict"], coverage=couple_coverage,
                        floor=MASTER_AXIS_COVERAGE_FLOOR,
                        reason="master_axis_coverage_below_floor",
                        rule="a_verdict_token_is_not_a_measurement_of_how_much_lined_up")
            continue

        holes = [dict(hole, quantum_ms=alignment["quantum_ms"])
                 for hole in holes_for_couple(alignment)]
        step_result("holes", candidate=candidate_path, couple=couple, n_holes=len(holes),
                    kinds=[hole["kind"] for hole in holes],
                    steps_ms=[(round(hole["step_ms"], 1) if hole["step_ms"] is not None
                                else None) for hole in holes],
                    master_spans_s=[round(hole["master_span_seconds"], 2) for hole in holes],
                    edge_addition_s=round(edge_addition_seconds(holes), 3))
        # Each couple may sit on differently delayed tracks: move its holes to the file's clock.
        delta_ms, master_start_ms, candidate_start_ms = couple_start_delta_ms(
            master_obj, candidate_obj, language, master_stream, candidate_stream, factor)
        scale = Fraction(factor) if factor not in (None, 1) else Fraction(1)
        fold = {"delta_ms": float(delta_ms), "master_start_ms": float(master_start_ms),
                "candidate_start_ms": float(candidate_start_ms), "scale": float(scale)}
        step_result("track_delay_fold", candidate=candidate_path, couple=couple,
                    master_start_ms=fold["master_start_ms"],
                    candidate_start_ms=fold["candidate_start_ms"], delta_ms=fold["delta_ms"],
                    rule="file_time_offset=track_offset+candidate_start*r-master_start;"
                         "file_time_position=track_position+own_start")
        couple_results.append({"couple": couple, "alignment": alignment, "holes": holes,
                               "fold": fold})

    if not couple_results:
        # The cause distinguishes "all blind" from "aligned but under the coverage floor".
        alignments = primed["alignments"]
        names = [f"{m}x{c}" for m, c in primed["couples"]]
        verdicts = {alignments[name]["verdict"] for name in names}
        measured = verdicts & set(banded_seed_alignment.MEASURED_VERDICTS)
        if measured:
            cause = "alignment_coverage_below_floor"
        else:
            cause = ("alignment_degenerate_input"
                     if banded_seed_alignment.VERDICT_DEGENERATE_INPUT in verdicts
                     else "alignment_all_seeds_refused"
                     if banded_seed_alignment.VERDICT_ALL_SEEDS_REFUSED in verdicts
                     else "alignment_segments_below_duration_floor"
                     if banded_seed_alignment.VERDICT_ALL_SEGMENTS_BELOW_DURATION_FLOOR in verdicts
                     else "alignment_no_anchored_runs")
        coverages = sorted(round(alignments[name].get("master_axis_coverage_fraction") or 0.0, 4)
                           for name in names)
        return False, cause, (
            f"no couple of {language} produced a usable alignment; the aligner reported "
            f"{sorted(verdicts)} across {len(names)} couples, covering {coverages} of the "
            f"master axis against a floor of {MASTER_AXIS_COVERAGE_FLOOR}"), None

    # Holes are the union of all couples. The reference couple (first usable one) only supplies
    # the master track the walk and `apply_plan` correlate against.
    holes = union_holes([[hole_on_file_clock(hole, record["couple"], record["fold"])
                          for hole in record["holes"]] for record in couple_results])
    for index, hole in enumerate(holes):
        step_result("union_hole", candidate=candidate_path, hole=index, kind=hole["kind"],
                    master_ms=[round(hole["master_ms"][0], 2), round(hole["master_ms"][1], 2)],
                    offsets_ms=[None if hole["offset_before_ms"] is None
                                else round(hole["offset_before_ms"], 3),
                                None if hole["offset_after_ms"] is None
                                else round(hole["offset_after_ms"], 3)],
                    step_ms=(None if hole["step_ms"] is None else round(hole["step_ms"], 3)),
                    union_of=hole["union_of"], offset_sources=hole["offset_sources"],
                    members=hole["members"])
    reference = couple_results[0]
    step_result("plan_shape", candidate=candidate_path, hole_source="union_of_all_couples",
                n_couples=len(couple_results), reference_couple=reference["couple"],
                per_couple_holes={record["couple"]: len(record["holes"])
                                  for record in couple_results},
                n_holes=len(holes), edge_addition_s=round(edge_addition_seconds(holes), 3))

    if any(hole["kind"] == "spans_whole_file" for hole in holes):
        return False, "alignment_no_anchored_runs", (
            f"after the union of "
            f"{len(couple_results)} couple(s), one hole spans the whole file -- no aligned zone "
            f"survived long enough to anchor either end, so there is no head, interior or tail "
            f"to resolve"), None

    # The hole budget applies per couple; the union may legitimately hold more regions.
    hole_budget = max_holes_per_couple(master_obj)
    over = [(record["couple"], len(record["holes"])) for record in couple_results
            if len(record["holes"]) > hole_budget]
    if over:
        return False, "hole_count_exceeds_resolver_budget", (
            f"couple(s) {over} decompose into more holes than the budget of "
            f"{hole_budget} ({MAX_HOLES_PER_COUPLE} per started {HOLE_BUDGET_SLICE_S:g} s of "
            f"master video); a pair that fragments this far is not one this instrument "
            f"has measured itself able to reconstruct"), None

    if not holes:
        step_result("holes", candidate=candidate_path, couple="union", n_holes=0,
                    verdict="audios_fully_compatible_offset_only")

    step_launch("frame_domain", candidate=candidate_path)
    domain, domain_reason = frame_domain(master_obj, candidate_obj, factor)
    step_result("frame_domain", candidate=candidate_path, reason=domain_reason,
                **({} if domain is None else {
                    "master_rate": f"{domain['master_rate'].numerator}/"
                                   f"{domain['master_rate'].denominator}",
                    "master_rate_source": domain["master_rate_source"],
                    "candidate_rate": f"{domain['candidate_rate'].numerator}/"
                                      f"{domain['candidate_rate'].denominator}",
                    "candidate_rate_source": domain["candidate_rate_source"],
                    "time_scale": (None if domain["time_scale"] is None
                                   else f"{domain['time_scale'].numerator}/"
                                        f"{domain['time_scale'].denominator}"),
                    "audio_effective_ratio": (None if resample_routing is None
                                              else resample_routing["effective_ratio_str"]),
                    "video_rate_ratio": f"{domain['video_rate_ratio'].numerator}/"
                                        f"{domain['video_rate_ratio'].denominator}",
                    "video_ratio_matches_speed_factor":
                        domain["video_ratio_matches_speed_factor"],
                    "master_timeline_ms": float(domain["master_timeline_ms"]),
                    "candidate_equivalent_duration_ms": (
                        None if domain["candidate_equivalent_duration_ms"] is None
                        else float(domain["candidate_equivalent_duration_ms"]))}))
    if domain is None and holes:
        return False, "hole_resolution_declined", (
            f"the pair decomposed into {len(holes)} hole(s) but its frame domain could not "
            f"be measured ({domain_reason}) -- a boundary that is not a frame on an exact "
            f"grid is not a boundary, so none was sought"), None
    if domain is None:
        return False, "frame_domain_unmeasured", (
            f"the pair carries no hole, but its frame domain could not be measured "
            f"({domain_reason}): the plan's timeline end and its grid are unknown, so no "
            f"piece can be placed"), None
    insane = hole_sanity(holes, domain, candidate_path)
    if insane is not None:
        return False, insane[0], insane[1], None

    # ---- audio bounds, video pins --------------------------------------------
    speed_ratio = None if factor in (None, 1) else _decimal(Fraction(factor))
    try:
        walk, walk_reason = reference_walk(reference, holes, master_obj, candidate_obj,
                                           language, speed_ratio, candidate_path,
                                           engine=(resample_routing or {}).get("filter_name",
                                                                               "asetrate"),
                                           frame_ms=float(domain["frame_ms"]))
    except progress_watchdog.step_stalled as error:
        log_partial_plan(candidate_path, "step_stalled",
                         [("holes", "b2", [(h["kind"], h["master_ms"]) for h in holes]),
                          ("audio_walk", "stalled", str(error)[:200])])
        return False, "step_stalled", (
            f"the audio walk made no progress ({error}) -- the partial plan "
            f"is logged; declined, retried at the next run"), None
    if walk is None:
        return False, "audio_walk_unavailable", (
            f"the millisecond walk on the reference couple {reference['couple']} could not "
            f"measure the pair ({walk_reason}) -- no offset, step or fill can be placed "
            f"without it"), None
    transitions, refusal = audio_transitions(walk, reference, domain, master_obj, candidate_obj,
                                             work_dir, candidate_path, language, union=holes)
    if transitions is None:
        return False, refusal[0], refusal[1], None
    # Both whole files are decoded for their scene changes once the holes are resolved (never
    # alongside them: both compete for the same decoders): each zone's picture
    # lag is the majority over its own changes (`zone_picture_lag`).
    import zone_picture_lag
    zone_lag_job = zone_picture_lag.ZoneLagJob(master_obj, candidate_obj, domain)
    frame_ms_float = float(domain["frame_ms"])
    zone_lag_pairs, zone_lag_reason = zone_lag_job.pairs(
        [t.get(key) for t in transitions for key in ("before_shift_frames", "after_shift_frames")]
        + [round(float(level["off_ms"]) / frame_ms_float) for level in walk["levels"]])
    tools.log_always(f"repair: zone_picture_lags scene_changes_matched="
                     f"{None if zone_lag_pairs is None else len(zone_lag_pairs)} "
                     f"reason={zone_lag_reason} for {candidate_path}\n")
    transitions = align_transitions_to_zone_lags(transitions, zone_lag_pairs, domain, walk,
                                                 master_obj, candidate_path)
    transitions = picture_steps_in_continuous_zones(transitions, walk, reference, domain,
                                                    master_obj, candidate_obj, work_dir,
                                                    candidate_path)
    log_holes_against_walk(holes, walk, transitions, candidate_path)
    picture_shifts = [(t["before_shift_frames"], t["after_shift_frames"]) for t in transitions
                      if t.get("before_shift_frames") is not None
                      and t.get("after_shift_frames") is not None]
    head_end_s, tail_start_s, refusal = audio_edges(
        walk, holes, domain, master_obj, candidate_obj, work_dir, candidate_path,
        shift_hints=({"head": picture_shifts[0][0], "tail": picture_shifts[-1][1]}
                     if picture_shifts else None),
        zones=edge_zones(transitions, zone_lag_pairs, domain))
    if refusal is not None:
        return False, refusal[0], refusal[1], None
    zones, fills, geometry_failure = plan_geometry(transitions, head_end_s, tail_start_s,
                                                   domain, walk)
    if zones is None:
        return False, "audio_transitions_overlap", (
            f"the audio's transitions do not tile the master timeline: {geometry_failure}"), None
    if not zones:
        return False, "plan_reads_no_candidate_content", (
            "the audio edges leave no candidate content on the master timeline -- a plan that "
            "reads nothing from the candidate is the master, not a repair"), None
    # Whole-file frame count at the delivered (picture) offsets, after the head and tail the
    # candidate's video does not reach are filled from the master; a plan that does not give
    # back the master's count is replaced by the picture-only plan, or declined.
    import frame_count_check
    start_delta_ms = zone_picture_lag.video_start_delta_ms(master_obj, candidate_obj)
    zone_picture_offsets_ms = zone_picture_offsets_from_hole_anchors(
        zones, transitions, domain, lag_pairs=zone_lag_pairs, start_delta_ms=start_delta_ms,
        candidate_path=candidate_path)
    picture_edges_ms = {index: (measured["picture_ms"], measured["picture_ms"])
                        for index, measured in zone_picture_offsets_ms.items()}
    try:
        moved = frame_count_check.close_plan_edges(
            zones, fills, frame_count_check.picture_edge_offsets(zones, walk, picture_edges_ms),
            frame_count_check.candidate_extent_ms(domain, factor, candidate_obj),
            domain["frame_ms"])
        tools.log_always(f"repair: plan_edges_closed head_ms={round(float(moved['head_ms']), 3)} "
                         f"tail_ms={round(float(moved['tail_ms']), 3)} for {candidate_path}\n")
    except Exception as error:                                           # noqa: BLE001
        tools.log_always(f"repair: plan_edges_closed skipped={type(error).__name__}:{error} "
                         f"for {candidate_path}\n")
    frame_count = frame_count_check.check_audio_plan(
        zones, fills, walk, domain, factor, master_obj, candidate_obj, candidate_path,
        picture_edges_ms=picture_edges_ms,
        zone_lags={zone["zone"]: zone_picture_lag.zone_majority(
            zone_lag_pairs, zone["master_start_ms"], zone["master_end_ms"], domain["frame_ms"])[0]
            for zone in zones},
        start_delta_ms=start_delta_ms)
    if frame_count.get("ok") is False:
        master_stream, candidate_stream = reference["couple"].split("x")
        return frame_count_check.visual_fallback(frame_count, master_obj, candidate_obj, walk,
                                                 domain, factor, {
            "language": language, "work_dir": work_dir, "domain": domain,
            "quantum_ms": reference["alignment"]["quantum_ms"],
            "master_stream": master_stream, "candidate_stream": candidate_stream,
            "resample_routing": resample_routing, "sweep_gate": sweep_gate},
            candidate_path)
    # The video is the source of truth: every zone's own picture offset is what every
    # delivered piece (rebuilt audio and retimed subtitles alike) is keyed on -- the audio
    # walk only located the zone boundaries. The pipeline only ever points a
    # scene search at a hole the audio walk already placed (no whole-file or whole-zone scan),
    # so a zone's picture offset is read off the pHash anchors already measured resolving that
    # zone's own bounding change point(s) (`audio_transitions`'s `before_shift_frames`/
    # `after_shift_frames`), never from a separate decode. A candidate whose own audio drifts
    # from its own picture inside a zone is a finding only (logged below), never reflected in
    # what is delivered; a zone whose bounding change points carry no video anchor keeps the
    # audio offset it already has, logged the same way.
    head_written_s, tail_written_s = written_edge_seconds(fills, walk["master_audio_end_s"])
    step_result("plan_shape_resolved", candidate=candidate_path,
                n_transitions=len(transitions),
                decisions=[t["decision"] for t in transitions],
                head_end_s=head_end_s, tail_start_s=tail_start_s,
                head_written_s=round(head_written_s, 3), tail_written_s=round(tail_written_s, 3))
    master_stream, candidate_stream = reference["couple"].split("x")
    ok, cause, reason = apply_plan(candidate_path, {
        "zones": zones, "fills": fills, "walk": walk, "head_written_s": head_written_s,
        "tail_written_s": tail_written_s,
        "zone_picture_offsets_ms": zone_picture_offsets_ms}, factor, master_obj, candidate_obj, {
        "language": language, "work_dir": work_dir, "domain": domain,
        "quantum_ms": reference["alignment"]["quantum_ms"],
        "master_stream": master_stream, "candidate_stream": candidate_stream,
        "resample_routing": resample_routing,
        "sweep_gate": sweep_gate})
    return ok, cause, reason, None


# ---------------------------------------------------------------------------
# Video-anchored route (`video_offset_plan`)
# ---------------------------------------------------------------------------

def _video_route_terminal(status, cause, reason, candidate_path, trigger):
    """Turn a non-fallback video-route outcome into the repair's boolean, writing the plan line."""
    if status == "repaired":
        _plan_line(video_offset_plan.VIDEO_ANCHORED_KIND, candidate_path, step="video_anchored",
                   trigger=trigger)
        return True
    _plan_line("none", candidate_path, step="video_anchored", trigger=trigger, cause=cause)
    return _terminal(candidate_path, "no_plan" if cause == "step_stalled" else "declined",
                     cause, reason)


# ---------------------------------------------------------------------------
# Corrupt comparison track
# ---------------------------------------------------------------------------
COMPARISON_TRACK_CORRUPT = "comparison_track_corrupt"


def _video_unreliable(video_obj):
    """Return (unreliable, source) for a file's picture: its `video_unreliable` tag, else `integrity.video_is_sound`.

    (None, 'step_stalled') when the probe made no progress."""
    tag = getattr(video_obj, "video_unreliable", None)
    if tag is not None:
        return bool(tag), "tag"
    import integrity
    try:
        return (not integrity.video_is_sound(video_obj)), "video_is_sound"
    except progress_watchdog.step_stalled:
        return None, "step_stalled"


def comparison_track_corrupt_route(master_obj, candidate_obj, language, primed, prime_reason,
                                   candidate_path):
    """Route a pair whose comparison track failed its strict decode at the prime.

    If both pictures are sound, the video arbitrates (video-anchored route); otherwise the pair
    declines `comparison_track_corrupt` with the decoder's line.
    """
    corrupt = primed.get("corrupt_track") or {}
    first = (corrupt.get("lines") or ["?"])[0][:200]
    evidence = {"side": corrupt.get("side"), "stream": corrupt.get("stream"),
                "rc": corrupt.get("rc"), "first": first, "source": "prime_strict_decode"}
    step_launch("comparison_track_corrupt", candidate=candidate_path, **evidence)
    videos = {side: _video_unreliable(obj)
              for side, obj in (("master", master_obj), ("candidate", candidate_obj))}
    step_result("comparison_track_corrupt", candidate=candidate_path,
                master_video_unreliable=videos["master"][0], master_source=videos["master"][1],
                candidate_video_unreliable=videos["candidate"][0],
                candidate_source=videos["candidate"][1])
    if any(unreliable is None for unreliable, _ in videos.values()):
        _plan_line("none", candidate_path, step="comparison_track_corrupt", cause="step_stalled")
        return _terminal(candidate_path, "no_plan", "step_stalled", (
            f"{prime_reason}; the video probe that decides the route made no progress "
            f"({videos}) -- a statement about the tool on this host"))
    bad = [side for side, (unreliable, _) in videos.items() if unreliable]
    if bad:
        _plan_line("none", candidate_path, step="comparison_track_corrupt",
                   cause=COMPARISON_TRACK_CORRUPT)
        return _terminal(candidate_path, "declined", COMPARISON_TRACK_CORRUPT, (
            f"{prime_reason} (the {evidence['side']} stream {evidence['stream']}, decoder: "
            f"«{first}»); the video cannot arbitrate: the {' and '.join(bad)} picture is "
            f"unreliable ({videos})"), detail={"corrupt_track": corrupt, "videos": videos})
    tools.log_always(f"repair: comparison_track_corrupt route=video_anchored "
                     f"language={language} evidence={evidence} for {candidate_path}\n")
    status, cause, reason = video_offset_plan.video_anchored_route(
        COMPARISON_TRACK_CORRUPT, evidence, master_obj, candidate_obj, language, [])
    # No audio fallback: the comparison track itself is corrupt.
    return _video_route_terminal("declined" if status == "fallback" else status, cause, reason,
                                 candidate_path, COMPARISON_TRACK_CORRUPT)


# ---------------------------------------------------------------------------
# Language tag vs content (`language_content_check`)
# ---------------------------------------------------------------------------
def _language_content_route(master_obj, candidate_obj, language, primed, work_dir, work_root,
                            master_intertrack_cache):
    """Check by content whether a candidate track carries the comparison language.

    None when it has nothing to say, else a decline or, on `audio_tag_conflict`, `repair()` re-run once."""
    import language_content_check as lcc
    candidate_path = candidate_obj.filePath
    if getattr(candidate_obj, "tag_checked", False):
        return None
    step_launch("content_check", candidate=candidate_path, language=language)
    check = lcc.content_check(master_obj, candidate_obj, language, primed, work_dir)
    best = check.get("best") or {}
    step_result("content_check", candidate=candidate_path, verdict=check["verdict"],
                best_similarity=best.get("similarity"), best_candidate=best.get("candidate"),
                best_candidate_tag=best.get("candidate_tag"), cost_s=check.get("cost_s"))
    if check["verdict"] == lcc.NO_COMMON_LANGUAGE:
        _drop_rate_wav(primed)
        _plan_line("none", candidate_path, step="content_check", cause=lcc.NO_COMMON_LANGUAGE)
        return _terminal(candidate_path, "declined", lcc.NO_COMMON_LANGUAGE, check["reason"],
                         detail={"best": best, "references": check["references"]})
    if check["verdict"] != lcc.AUDIO_TAG_CONFLICT:
        return None
    tools.log_always(f"repair: {lcc.AUDIO_TAG_CONFLICT} route=retag_and_rerun "
                     f"track={check['track']} tag={check['tag']} matched_language={language} "
                     f"master_stream={check['master_stream']} "
                     f"similarity={check['similarity']:.4f} moves={check['moves']} for "
                     f"{candidate_path}\n")
    _drop_rate_wav(primed)
    lcc.apply_correction(candidate_obj, check)
    return repair(master_obj, candidate_obj, language, work_root=work_root,
                  master_intertrack_cache=master_intertrack_cache,
                  _carried_prime={"fingerprints": primed.get("fingerprints"),
                                  "content_end": primed.get("content_end"),
                                  "sample_rate": primed.get("sample_rate"),
                                  "rate_wav": primed.pop("tag_check_wav", None)})


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class VmsamDecline(Exception):
    """Exception carrying a DECLINE_CAUSES token in `.cause`, for callers that catch Exception."""

    def __init__(self, message, cause):
        super().__init__(message)
        self.cause = cause


# Signals that route a pair rather than end it (never terminal causes).
ROUTING_SIGNALS = {
    # A candidate track tagged another language carries the comparison language: re-tag it.
    "audio_tag_conflict": "routing",
}


@repair_log.timed_phase("orchestrator", "repair",
                        lambda master_obj, candidate_obj, *a, **k: candidate_obj.filePath)
def repair(master_obj, candidate_obj, comparison_language, work_root=None,
           master_intertrack_cache=None, _carried_prime=None):
    """Repair one refused candidate against the master; returns True when a plan was applied.

    Runs `_repair` inside one `merge_video_decode_once` scope, so no track is decoded twice; a
    re-entry after a language re-tag joins the open scope.

    Args:
        master_obj, candidate_obj: the video objects.
        comparison_language: language whose tracks are aligned.
        work_root: scratch directory.
        master_intertrack_cache: caller's per-master memo (never module state, so two masters
            cannot share a verdict).
    """
    import merge_video_decode_once
    merge_video_decode_once.begin(work_root or path.join(tools.tmpFolder, "repair",
                                                         "orchestrator"),
                                  label=path.basename(candidate_obj.filePath))
    try:
        return _repair(master_obj, candidate_obj, comparison_language, work_root,
                       master_intertrack_cache, _carried_prime)
    finally:
        merge_video_decode_once.end()


def _repair(master_obj, candidate_obj, comparison_language, work_root=None,
            master_intertrack_cache=None, _carried_prime=None):
    """Run the repair steps in order; each step returns a result and none calls another.

    0. Master conformity check; failure declines.
    1. Master self-consistency; a desynchronised master routes to the video.
    2. Prime every couple; similarity gate and rate arm.
    3. Chimeric plan at the chosen factor, then plan application.
    """
    candidate_path = candidate_obj.filePath
    tools.log_always(f"orchestrator: no time budget, stall_s={progress_watchdog.STALL_S} "
                     f"for {candidate_path}\n")
    if master_intertrack_cache is None:
        master_intertrack_cache = {}
    work_dir = work_root or path.join(tools.tmpFolder, "repair", "orchestrator")
    tools.make_dirs(work_dir)
    _PLAN_LINE_MARK[candidate_path] = len(tools.logs)
    tools.dev_log(f"orchestrator: repair starting on {candidate_path} "
                  f"master={master_obj.filePath} language={comparison_language} "
                  f"hole_merge_window_s={HOLE_MERGE_WINDOW_SECONDS} "
                  f"({_HOLE_MERGE_SOURCE})\n")

    # ---- STEP 0: does the master pass its own verification? -----------------
    # Cached per master.
    conformity = master_intertrack_cache.get(("conformity", master_obj.filePath))
    if conformity is None:
        step_launch("master_conformity", candidate=candidate_path, master=master_obj.filePath)
        import master_self_check
        try:
            conformity = master_self_check.check_master_conformity(master_obj)
        except Exception as error:                                       # noqa: BLE001
            tools.log_always(f"repair: master_conformity unmeasured ({type(error).__name__}: "
                             f"{error}) -- no verdict, never healthy by default, for "
                             f"{candidate_path}\n")
            conformity = {"verdict": None, "failed": [], "seconds": None, "warnings": []}
        master_intertrack_cache[("conformity", master_obj.filePath)] = conformity
        step_result("master_conformity", candidate=candidate_path, verdict=conformity["verdict"],
                    failed=[c["name"] for c in conformity["failed"]],
                    warnings=conformity["warnings"], seconds=conformity["seconds"])
    if conformity["verdict"] is not None:
        _plan_line("none", candidate_path, step="master_conformity",
                   cause=conformity["verdict"])
        return _terminal(candidate_path, "declined", conformity["verdict"], (
            "the master fails its own verification (file_conformity, cheap families): "
            + "; ".join(f"{c['name']} {c['numbers']} -- {c['sentence']}"
                        for c in conformity["failed"])),
            detail={"master_conformity": conformity})

    # ---- STEP 1: does the master agree with itself? -------------------------
    step_launch("master_self_check", candidate=candidate_path,
                master=master_obj.filePath, language=comparison_language)
    try:
        import merge_video_repair
        verdict = merge_video_repair.master_intertrack_verdict(
            master_obj, comparison_language, master_intertrack_cache)
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"orchestrator: master_intertrack_verdict unavailable "
                      f"({type(error).__name__}: {error}) -- no verdict; None means NOT "
                      f"MEASURED, never healthy\n")
        verdict = None
    step_result("master_self_check", candidate=candidate_path,
                verdict=(verdict or {}).get("verdict") if verdict else None,
                inert=(verdict or {}).get("inert") if verdict else None)
    if verdict is not None and verdict.get("verdict") == video_offset_plan.TRIGGER_MASTER_DESYNC:
        # The master's tracks disagree with each other: the video arbitrates.
        worst = verdict.get("worst") or {}
        evidence = {"pairs": [(p.get("stream_a"), p.get("stream_b"), p.get("lag_ms"),
                               p.get("correlation")) for p in verdict.get("pairs") or []],
                    "worst_lag_ms": worst.get("lag_ms"), "source": "master_self_check"}
        tools.log_always(f"repair: master_intertrack_desync route=video_anchored "
                         f"language={comparison_language} evidence={evidence} "
                         f"for {candidate_path}\n")
        status, cause, reason = video_offset_plan.video_anchored_route(
            video_offset_plan.TRIGGER_MASTER_DESYNC, evidence, master_obj, candidate_obj, comparison_language,
            [])
        return _video_route_terminal(status, cause, reason, candidate_path,
                                     video_offset_plan.TRIGGER_MASTER_DESYNC)
    if verdict is not None and verdict.get("verdict") is not None:
        _plan_line("none", candidate_path, step="master_self_check",
                   cause=verdict["verdict"])
        return _terminal(candidate_path, "declined", verdict["verdict"], verdict["reason"],
                         detail={"verdict": verdict["verdict"],
                                 "master_intertrack": verdict})

    # ---- STEP 2: low similarity? can a resample raise it? -------------------
    # Every couple is primed once here and reused by step 3.
    factor = 1
    sweep_gate = None
    primed = {"couples": None, "fingerprints": {}, "alignments": {}, "factor_label": "1",
              "sample_rate": None}
    if (_carried_prime is not None and _carried_prime.get("sample_rate")
            == comparison_sample_rate(master_obj, candidate_obj, comparison_language)):
        # Re-run after a tag correction: reuse the fingerprints taken at the same sample rate.
        primed["fingerprints"] = dict(_carried_prime.get("fingerprints") or {})
        primed["content_end"] = dict(_carried_prime.get("content_end") or {})
        primed["sample_rate"] = _carried_prime["sample_rate"]
        if _carried_prime.get("rate_wav"):
            primed["rate_wav"] = _carried_prime["rate_wav"]
    elif _carried_prime is not None and _carried_prime.get("rate_wav"):
        _drop_rate_wav({"rate_wav": _carried_prime["rate_wav"]})
    if not enumerate_couples(master_obj, candidate_obj, comparison_language):
        # No candidate track is tagged with the comparison language: check content first.
        primed["sample_rate"] = primed.get("sample_rate") or comparison_sample_rate(
            master_obj, candidate_obj, comparison_language)
        primed["couples"] = []
        routed = _language_content_route(master_obj, candidate_obj, comparison_language,
                                         primed, work_dir, work_root, master_intertrack_cache)
        if routed is not None:
            return routed
        primed["couples"] = None
    step_launch("prime", candidate=candidate_path, language=comparison_language)
    prime_ok, prime_cause, prime_reason = prime_couples(
        master_obj, candidate_obj, comparison_language, work_dir, primed)
    couples = primed["couples"]
    step_result("prime", candidate=candidate_path, ok=prime_ok, cause=prime_cause,
                couples=couples)
    if not prime_ok and prime_cause == COMPARISON_TRACK_CORRUPT:
        _drop_rate_wav(primed)
        return comparison_track_corrupt_route(master_obj, candidate_obj, comparison_language,
                                              primed, prime_reason, candidate_path)
    if not prime_ok:
        _drop_rate_wav(primed)
        _plan_line("none", candidate_path, step="prime", cause=prime_cause)
        return _terminal(candidate_path, "no_plan", prime_cause, prime_reason)

    # ---- step 1b: audio self-contradiction -> video route --------------------
    trigger, evidence, delay_rows = video_offset_plan.detect_audio_contradiction(
        master_obj, candidate_obj, comparison_language, primed, candidate_path)
    if trigger is not None:
        status, cause, reason = video_offset_plan.video_anchored_route(
            trigger, evidence, master_obj, candidate_obj, comparison_language, delay_rows)
        if status != "fallback":
            _drop_rate_wav(primed)
            return _video_route_terminal(status, cause, reason, candidate_path, trigger)
        tools.log_always(f"repair: video_route_fallback trigger={trigger} video={cause} -- the "
                         f"offset is not one constant from head to tail (a drift or an interior "
                         f"edit): the ordinary audio path continues "
                         f"for {candidate_path}\n")

    step_launch("similarity_gate", candidate=candidate_path, n_couples=len(couples))
    should_sweep, gate_prose, observations = ensemble_similarity_gate(primed, candidate_path)
    step_result("similarity_gate", candidate=candidate_path, should_sweep=should_sweep,
                **{key: value for key, value in observations.items()})

    # ---- step 2a: rate arm ---------------------------------------------------
    # Armed by low similarity, by a fast drift on any couple (a linear drift would otherwise
    # reach the hole resolver as 100+ holes) or by declared frame rates naming a rate. The
    # drift and declared rate only choose the first finalists; alignment decides.
    import rate_direction
    engine = None
    drift_named = []
    for couple_name, alignment in primed["alignments"].items():
        drift = rate_direction.fast_drift_signature(alignment.get("zones_detail") or [],
                                                    alignment.get("quantum_ms"))
        step_result("fast_drift", candidate=candidate_path, couple=couple_name,
                    fires=drift["fires"], implied_ratio=drift["implied_ratio_fit"],
                    named=[f"{r.numerator}/{r.denominator}"
                           for r in drift["named_rate_candidates"]], reason=drift["reason"])
        drift_named += [r for r in drift["named_rate_candidates"] if r not in drift_named]
    declared = rate_direction.declared_named_ratio(master_obj.filePath, candidate_obj.filePath)
    armed_by = ("similarity_gate" if should_sweep else "fast_drift" if drift_named
                else "declared_frame_rate" if declared is not None else None)
    first_ratios = rate_direction.first_finalists(declared, drift_named)
    step_result("rate_arm_armed", candidate=candidate_path, armed_by=armed_by,
                declared=(None if declared is None
                          else f"{declared.numerator}/{declared.denominator}"),
                first_finalists=[f"{r.numerator}/{r.denominator}" for r in first_ratios])
    if armed_by is None:
        _drop_rate_wav(primed)
    else:
        step_launch("rate_arm", candidate=candidate_path, language=comparison_language,
                    armed_by=armed_by)
        winner, engine, sweep_gate, sweep_cause = rate_arm(
            primed, first_ratios, work_dir, candidate_path, videos=(master_obj, candidate_obj))
        step_result("rate_arm", candidate=candidate_path, armed_by=armed_by,
                    factor=(None if winner is None
                            else f"{winner.numerator}/{winner.denominator}"),
                    engine=engine, cause=sweep_cause, rounds=sweep_gate.get("rounds"),
                    span_coverage=sweep_gate.get("span_coverage"),
                    margin=sweep_gate.get("margin"),
                    finalists=[(row["ratio"], row["engine"], row["span_coverage"], row["zones"],
                                row["ladder"]) for row in sweep_gate["rows"]])
        # Only low similarity with every ratio refused is terminal; other arms are hints and
        # continue at factor 1 on the alignment already measured.
        terminal = (winner is None and armed_by == "similarity_gate"
                    and observations.get("gate_arm") != "rate_relation_signature"
                    and sweep_cause != "rate_arm_unity_wins")
        if terminal:
            # The native alignment is final: check for a master cut short first.
            refusal = tail_decision(primed, master_obj, comparison_language, candidate_path)
            if refusal is not None:
                _plan_line("none", candidate_path, step="rate_arm", cause=refusal[0])
                return _terminal(candidate_path, "declined", refusal[0], refusal[1])
            # Rule out a wrong language tag before concluding on low similarity.
            routed = _language_content_route(master_obj, candidate_obj, comparison_language,
                                             primed, work_dir, work_root, master_intertrack_cache)
            if routed is not None:
                return routed
            _plan_line("none", candidate_path, step="rate_arm", cause=sweep_cause)
            return _terminal(
                candidate_path, "no_plan", sweep_cause,
                f"mean similarity is low ({gate_prose}) and no named ratio raises it in either "
                f"engine: {[(r['ratio'], r['engine'], r['span_coverage']) for r in sweep_gate['rows']]}",
                detail={"resample_gate": sweep_gate})
        if winner is None:
            tools.dev_log(f"orchestrator: the rate arm ({armed_by}) found no rate for "
                          f"{candidate_path} ({sweep_cause}); the pair CONTINUES at speed_factor 1 "
                          f"on the alignment already measured\n")
        else:
            factor = winner

    # ---- step 2b: re-prime at a confirmed factor ------------------------------
    # The candidate side of every couple is speed-corrected, re-fingerprinted and re-aligned.
    resample_routing = None
    if factor != 1:
        factor_label = (f"{factor.numerator}/{factor.denominator}"
                        if isinstance(factor, Fraction) else factor)
        step_launch("rate_reprime", candidate=candidate_path, speed_factor=factor_label)
        resample_routing, reprime_cause = rate_resample_routing(
            factor, engine, master_obj, candidate_obj, comparison_language, work_dir,
            primed["sample_rate"])
        if resample_routing is None:
            step_result("rate_reprime", candidate=candidate_path, ok=False, cause=reprime_cause)
            _plan_line("none", candidate_path, step="rate_reprime", cause=reprime_cause)
            return _terminal(
                candidate_path, "no_plan", reprime_cause,
                f"the pair carries a confirmed rate relation ({factor_label}) but the "
                f"speed-corrected candidate that would let the aligner measure across it could "
                f"not be built ({reprime_cause}) -- no measurement was made at that factor")
        step_result("rate_reprime", candidate=candidate_path, ok=True,
                    side=resample_routing["side"],
                    filter=resample_routing["filter_name"],
                    requested_ratio=resample_routing["requested_ratio"],
                    effective_ratio=resample_routing["effective_ratio_str"],
                    tag_factor=resample_routing["tag_factor"],
                    source_sample_rate=resample_routing["source_sample_rate"],
                    asetrate_target=resample_routing["asetrate_target"],
                    intermediate_rate=resample_routing["intermediate_rate"],
                    filter_chain=resample_routing["filter_chain"],
                    pitch_measured_ratio=resample_routing["pitch_measured_ratio"],
                    pitch_peak=resample_routing["pitch_peak"],
                    pitch_refusal=resample_routing["pitch_refusal"],
                    pitch_window_s=resample_routing["pitch_window_seconds"],
                    pitch_test_discriminating=resample_routing["pitch_test_discriminating"],
                    pitch_tolerance_band=resample_routing["pitch_tolerance_band"],
                    inverting_case_detector=resample_routing["inverting_case_detector"],
                    inverting_case_observation=resample_routing["inverting_case_observation"],
                    rule=resample_routing["rule"])
        tools.dev_log(f"orchestrator: rate_reprime routing for {candidate_path}: "
                      f"{resample_routing['route_reason']}\n")
        prime_ok, prime_cause, prime_reason = prime_couples(
            master_obj, candidate_obj, comparison_language, work_dir, primed,
            resample_routing=resample_routing)
        if not prime_ok:
            _plan_line("none", candidate_path, step="rate_reprime", cause=prime_cause)
            return _terminal(candidate_path, "no_plan", prime_cause, prime_reason)

    # ---- step 3: chimeric ----------------------------------------------------
    step_launch("chimeric", candidate=candidate_path, language=comparison_language,
                speed_factor=(f"{factor.numerator}/{factor.denominator}"
                               if isinstance(factor, Fraction) else factor))
    ok, cause, reason, detail = chimeric(factor, comparison_language, master_obj,
                                         candidate_obj, work_dir, primed,
                                         sweep_gate=sweep_gate,
                                         resample_routing=resample_routing)
    step_result("chimeric", candidate=candidate_path, ok=ok, cause=cause)
    if ok:
        # `apply_plan` already recorded the `repaired` terminal.
        _plan_line("chimeric", candidate_path, step="chimeric",
                   speed_factor=(f"{factor.numerator}/{factor.denominator}"
                                  if isinstance(factor, Fraction) else factor))
        return True
    _plan_line("none", candidate_path, step="chimeric", cause=cause)
    return _terminal(candidate_path,
                     "declined" if cause in ("master_cut_short", "frame_count_mismatch")
                     else "no_plan",
                     cause, reason,
                     detail={"cross_verification": detail} if detail else None)


def align_fingerprints(fp_master, quantum_master, duration_master, fp_candidate,
                       quantum_candidate, duration_candidate):
    """Align two fingerprint lists with `b2_align` and its diagnostics (prime and rate arm)."""
    return alignment_diagnostics.align(
        fp_master, fp_candidate, quantum_master,
        candidate_quantum_ms=quantum_candidate,
        duration_diff_ms=abs(duration_master - duration_candidate) * 1000.0,
        signed_duration_diff_ms=(duration_candidate - duration_master) * 1000.0,
        shorter_duration_ms=min(duration_master, duration_candidate) * 1000.0)


def prime_couples(master_obj, candidate_obj, language, work_dir, primed, resample_routing=None):
    """Fingerprint and align every couple of the comparison language, filling `primed` in place.

    With `resample_routing` (re-prime at a confirmed factor) the candidate fingerprints and all
    alignments are dropped and rebuilt with the speed filter on the candidate side; master
    fingerprints are kept.

    Args:
        primed: dict with `couples`, `fingerprints`, `alignments`, `factor_label`,
            `sample_rate`; updated in place.

    Returns:
        (ok, cause, reason)
    """
    candidate_path = candidate_obj.filePath
    if primed.get("couples") is None:
        primed["couples"] = enumerate_couples(master_obj, candidate_obj, language)
        if not primed["couples"]:
            raise ValueError(f"no {language} couple on {candidate_path}: the caller guarantees "
                             f"the comparison language on both sides")
    sample_rate = primed.get("sample_rate") or comparison_sample_rate(master_obj, candidate_obj,
                                                                      language)
    primed["sample_rate"] = sample_rate
    if resample_routing is not None:
        dropped_fingerprints = sorted(key[1] for key in primed["fingerprints"]
                                      if key[0] != "master")
        primed["fingerprints"] = {key: value for key, value in primed["fingerprints"].items()
                                  if key[0] == "master"}
        dropped_alignments = sorted(primed["alignments"])
        primed["alignments"] = {}
        primed["factor_label"] = resample_routing["requested_ratio"]
        step_result("reprime", candidate=candidate_path,
                    kept_master_fingerprints=sorted(key[1] for key in primed["fingerprints"]),
                    dropped_candidate_fingerprints=dropped_fingerprints,
                    dropped_alignments=dropped_alignments,
                    reason="candidate_fingerprints_stale_under_the_confirmed_factor")
    for master_stream, candidate_stream in primed["couples"]:
        for side, video_obj, stream in (("master", master_obj, master_stream),
                                         ("candidate", candidate_obj, candidate_stream)):
            key = (side, stream)
            if key in primed["fingerprints"]:
                continue
            duration = _track_duration_seconds(video_obj, language, stream)
            if duration is None:
                return (False, "track_duration_unmeasurable",
                        f"the {side} {language} stream {stream} carries no readable duration, "
                        f"so there is no length to fingerprint it over")
            # The plan's timeline is the master video, so master audio past it is not read.
            video_ms = _video_duration_ms(master_obj) if side == "master" else None
            if video_ms is not None and duration > float(video_ms) / 1000.0:
                step_result("master_audio_overruns_video", candidate=candidate_path,
                            stream=stream, audio_s=round(duration, 3),
                            video_s=round(float(video_ms) / 1000.0, 3),
                            rule="ADDENDUM_26_2_fingerprint_stops_at_the_master_video")
                duration = float(video_ms) / 1000.0
            track_filter = (resample_routing["filter_chain"]
                            if resample_routing is not None and side == "candidate" else None)
            corrected_duration = (duration * float(resample_routing["effective_ratio"])
                                  if track_filter else duration)
            step_launch("fingerprint", candidate=candidate_path, side=side, stream=stream,
                        duration_s=round(duration, 3), sample_rate=sample_rate,
                        audio_filter=track_filter,
                        corrected_duration_s=(round(corrected_duration, 3)
                                              if track_filter else None))
            started = time.time()
            measures = {}
            keep = (resample_routing is None and side == "candidate"
                    and (master_stream, candidate_stream) == primed["couples"][0])
            try:
                points, quantum_ms = fingerprint_track(
                    video_obj, language, stream, side, work_dir, sample_rate, duration,
                    audio_filter=track_filter, output_duration_seconds=corrected_duration,
                    measures=measures, keep_wav=keep)
            except progress_watchdog.step_stalled as error:
                return (False, "step_stalled",
                        f"the {side} {language} stream {stream} extraction made no progress "
                        f"({error}) -- a statement about the tool on this host")
            except audio_extract.StrictDecodeFailed as error:
                # The fingerprint extraction is a strict decode; it failed on this track.
                primed["corrupt_track"] = {"side": side, "stream": stream, "rc": error.rc,
                                           "lines": error.lines}
                step_result("fingerprint", candidate=candidate_path, side=side, stream=stream,
                            strict_decode="failed", rc=error.rc,
                            first=(error.lines[0][:160] if error.lines else None),
                            seconds=round(time.time() - started, 2))
                return (False, COMPARISON_TRACK_CORRUPT,
                        f"the {side} {language} comparison stream {stream} fails its strict "
                        f"decode: rc={error.rc}, «{error.lines[0][:200] if error.lines else '?'}»")
            step_result("fingerprint", candidate=candidate_path, side=side, stream=stream,
                        n_points=len(points) if points else 0,
                        quantum_ms=round(quantum_ms, 4) if quantum_ms else None,
                        resampled=bool(track_filter),
                        seconds=round(time.time() - started, 2))
            if points is None:
                return (False, "fingerprinting_raised",
                        f"the {side} {language} stream {stream} could not be extracted or "
                        f"fingerprinted")
            primed["fingerprints"][key] = (points, quantum_ms, corrected_duration)
            if measures.get("wav"):
                primed["rate_wav"] = {"path": measures["wav"], "stream": stream,
                                      "duration_s": corrected_duration, "rate": sample_rate}
            content_end = measures.get("content_end_s")
            full_s = _track_duration_seconds(video_obj, language, stream)
            if side == "master" and full_s is not None and full_s > duration:
                # The track runs past the video, where the WAV stopped: read its own end.
                content_end = (overrun_content_end_s(video_obj, stream, full_s, duration)
                               or content_end)
            primed.setdefault("content_end", {})[key] = content_end
            step_result("content_end", candidate=candidate_path, side=side, stream=stream,
                        content_end_s=None if content_end is None else round(content_end, 3),
                        track_s=None if full_s is None else round(full_s, 3))

        name = f"{master_stream}x{candidate_stream}"
        fp_master, quantum_master, duration_master = primed["fingerprints"][
            ("master", master_stream)]
        fp_candidate, quantum_candidate, duration_candidate = primed["fingerprints"][
            ("candidate", candidate_stream)]
        step_launch("align", candidate=candidate_path, couple=name, n_master=len(fp_master),
                    n_candidate=len(fp_candidate))
        started = time.time()
        alignment = align_fingerprints(fp_master, quantum_master, duration_master,
                                       fp_candidate, quantum_candidate, duration_candidate)
        alignment["alignment_seconds"] = time.time() - started
        primed["alignments"][name] = alignment
        step_result("align", candidate=candidate_path, couple=name,
                    verdict=alignment["verdict"], n_zones=len(alignment.get("zones") or []),
                    n_cut_zones=len(alignment.get("cut_zones") or []),
                    overlaps_resolved=alignment.get("segments_overlap_resolved"),
                    admitted_self_evident=alignment.get("admitted_self_evident"),
                    coverage=alignment.get("master_axis_coverage_fraction"),
                    residual_fraction=alignment.get("residual_fraction"),
                    seconds=round(alignment["alignment_seconds"], 2))
    return True, None, None
