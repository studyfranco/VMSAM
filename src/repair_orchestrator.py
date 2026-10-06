# -*- coding: utf-8 -*-
"""Flat repair orchestrator for a candidate the compatibility check refused.

Each step is launched here and returns a result; the steps never call each other:
master self-check, similarity gate and rate arm (`rate_arm`), priming of every couple
(`prime_couples`), holes per couple and their union (`union_holes`), audio bounds with
video-pinned cuts (`audio_walk`, `scene_anchor`), then plan application (`apply_plan`).

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
import subprocess
import time

import audioCorrelation
import audio_extract
import banded_seed_alignment
import owner_judgment
import repair_log
import repair_pool
import tools
import video_offset_plan

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

# Inter-couple positional window: a cluster is the events one anchor search would reach.
INTERCOUPLE_POSITION_WINDOW_SECONDS = HOLE_MERGE_WINDOW_SECONDS

# Same-offset gaps absorbed by coalescing are logged; those at or above the resolver's reach are
# confirmed by the video no-cut test (below it, the gap cannot be told from the aligned zone).
ABSORBED_GAP_VIDEO_CHECK_SECONDS = HOLE_MERGE_WINDOW_SECONDS

# Island (aligned span between two clustered holes) video check: islands with at least
# `scene_anchor.MIN_VALIDATION_FRAMES` frames get the video no-cut test; shorter ones are logged
# `untestable`.
try:
    ISLAND_VIDEO_CHECK_MIN_FRAMES = int(_scene_anchor.MIN_VALIDATION_FRAMES)
except Exception:                                                        # noqa: BLE001
    ISLAND_VIDEO_CHECK_MIN_FRAMES = 3

# Inter-couple step tolerance: couples measuring the same cut differ by up to ~1.02 quanta, so
# steps are compared in milliseconds against 1.5 quanta. Positions are clustered, never compared.
INTERCOUPLE_STEP_TOLERANCE_QUANTA = 1
INTERCOUPLE_STEP_TOLERANCE_SLACK = 1.5

# Time budgets: an overrun is a named decline, never a blocked worker.
ALIGNMENT_BUDGET_S = 120.0          # one couple's b2_align (`alignment_budget_exceeded`)
HOLE_BUDGET_S = 300.0               # one hole's frame-exact search (`hole_budget_exceeded`)
# One candidate's whole repair (`repair_budget_exceeded`): 40 min per started 30-min slice of the
# master's video, capped at 4 h, so no file is declined for its length alone.
REPAIR_BUDGET_PER_SLICE_S = 2400.0
REPAIR_BUDGET_SLICE_S = 1800.0
REPAIR_BUDGET_CAP_S = 14400.0
# Hole sanity bound: an interior hole wider than this is not searched (real holes stay under ~30 s).
INTERIOR_HOLE_MAX_SPAN_S = 300.0

# Edge additions totalling under this do not tag a track chimeric (any interior splice does), so
# `keep_best_audio` does not demote a near-intact track for a few seconds of edge completion.
EDGE_ADDITION_CHIMERIC_TAG_THRESHOLD_SECONDS = 15.0

# Cap on holes per couple, per started 30-min slice of the master's video: each hole costs a
# frame-exact search. About twice the highest count seen on real media (21), so the choice of
# comparison language never decides a refusal. Exceeding it is `could_not_run`.
MAX_HOLES_PER_COUPLE = 45
HOLE_BUDGET_SLICE_S = 1800.0

# Coverage floor of the step-2 gate: `single_segment_no_cut` means "no offset step found", not
# "aligned". Healthy couples read >= 0.50, rate-mismatched or wrong-episode couples <= 0.31; 0.40
# sits in the empty band between them. Terminal when the rate arm cannot raise the coverage.
MASTER_AXIS_COVERAGE_FLOOR = 0.40

# Pitch probe window: `pal_pitch_confirmer.confirm_pitch`'s own default, the window its
# NTSC_TOLERANCE was calibrated on.
try:
    import pal_pitch_confirmer as _pal_pitch_confirmer
    PITCH_PROBE_WINDOW_SECONDS = float(
        _pal_pitch_confirmer.confirm_pitch.__defaults__[0])
    _PITCH_WINDOW_SOURCE = "pal_pitch_confirmer.confirm_pitch's own default"
except Exception:                                                        # noqa: BLE001
    PITCH_PROBE_WINDOW_SECONDS = 180.0
    _PITCH_WINDOW_SOURCE = "literal fallback -- pal_pitch_confirmer unimportable"

# Floor under a shortened pitch window (short pairs use half their usable span): a spectral ratio
# over a shorter window has no useful frequency resolution.
PITCH_PROBE_WINDOW_MINIMUM_SECONDS = 30.0

# Rate-relation arm of the step-2 gate, read by `zone_ladder_signature`: a rate relation forces a
# ladder of one-quantum steps all in one direction. The direction fraction is what separates
# (drift has a sign, editing does not); a line fit of `best_shift_trace` does not. A false
# positive costs one rate sweep and then continues on the alignment already measured.
RATE_RELATION_SLOPE_GATE_CALIBRATED = True

# Minimum rungs: a statistical-sufficiency floor for the fractions below, not a detector. At the
# smallest named deviation (1001/1000) 8 rungs need ~16.5 min of aligned span; drift fast enough to
# step 3+ quanta per segment produces no rungs (merged by OFFSET_MERGE_TOLERANCE_POINTS).
LADDER_MIN_RUNGS = 8
# Purity: a cheap conservative guard against a file that both drifts and is heavily edited.
LADDER_MIN_RUNG_FRACTION = 0.80
# Direction: the deciding condition. Drift reads ~0.94, edited pairs 0.50-0.54.
LADDER_MIN_RUNG_MONOTONE_FRACTION = 0.85

# Magnitude floor: half the smallest named deviation (|1001/1000 - 1| = 9.99e-4).
RATE_LADDER_MIN_FACTOR_DEVIATION = 5e-4

# The aligner's own "could not measure" verdicts, bound to its tuple so a renamed verdict cannot
# silently disable the gate. `all_segments_below_duration_floor` catches rate mismatches (at the
# PAL factor no fixed-offset run survives the 2 s segment floor). Checked at import: non-empty and
# disjoint from the measured verdicts.
ALIGNMENT_COULD_NOT_MEASURE_VERDICTS = banded_seed_alignment.COULD_NOT_MEASURE_VERDICTS
assert ALIGNMENT_COULD_NOT_MEASURE_VERDICTS, (
    "repair_orchestrator: banded_seed_alignment.COULD_NOT_MEASURE_VERDICTS is empty -- the "
    "step-2 similarity gate would never fire")
assert not (set(ALIGNMENT_COULD_NOT_MEASURE_VERDICTS)
            & set(banded_seed_alignment.MEASURED_VERDICTS)), (
    "repair_orchestrator: a verdict cannot be both a measurement and a could-not-measure")

# Measurement class, logged beside every cause token.
CLASS_CONCLUSIVE = "ran_conclusive_negative"
CLASS_COULD_NOT_RUN = "could_not_run"

# Closed cause vocabulary with each token's class; an unknown token logs as `unclassified`.
DECLINE_CAUSES = {
    # step 1 (a self-contradicting master routes to `video_anchored_route`).
    # No match between the two files' scene changes proves different content:
    "video_content_mismatch": CLASS_CONCLUSIVE,
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
    "intercouple_disagreement": CLASS_CONCLUSIVE,
    # A budget limit, not a measurement about the pair (see MAX_HOLES_PER_COUPLE):
    "hole_count_exceeds_resolver_budget": CLASS_COULD_NOT_RUN,
    # Every couple falls under MASTER_AXIS_COVERAGE_FLOOR:
    "alignment_coverage_below_floor": CLASS_COULD_NOT_RUN,
    # The speed-corrected candidate could not be built at the confirmed factor:
    "rate_resample_unbuildable": CLASS_COULD_NOT_RUN,
    # step 4: the frame-exact search could not establish a boundary on at least one hole.
    "hole_resolution_declined": CLASS_COULD_NOT_RUN,
    # Budgets and hole sanity bounds: facts about this run or the alignment, not the pair.
    "alignment_budget_exceeded": CLASS_COULD_NOT_RUN,
    "hole_budget_exceeded": CLASS_COULD_NOT_RUN,
    "repair_budget_exceeded": CLASS_COULD_NOT_RUN,
    "decoder_timeout": CLASS_COULD_NOT_RUN,
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
    "candidate_segment_regression": CLASS_CONCLUSIVE,
    "speed_transform_not_validated": CLASS_COULD_NOT_RUN,
    "plan_not_contiguous": CLASS_COULD_NOT_RUN,
    "plan_piece_empty_or_inverted": CLASS_COULD_NOT_RUN,
    "plan_end_not_master_timeline": CLASS_COULD_NOT_RUN,
    # A delivery probe landed where a compared track has no audio:
    "probe_reads_no_audio": CLASS_COULD_NOT_RUN,
    # The audio measures aligned but the video disagrees: a measured fact about the pair, not
    # a failure to measure -- the owner judges it, the pair is not reattempted unchanged.
    "owner_judgment_pending": CLASS_CONCLUSIVE,
}

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
    tools.log_line(f"repair: plan {kind} build={repair_log.build_sha()} orchestrator=1 "
                      f"{_fields(sorted(fields.items()))} for {candidate_path}\n")


# Index in `tools.logs` where each candidate's current `repair()` run began.
_PLAN_LINE_MARK = {}


def log_measurement_class(candidate_path, cause):
    """Log the measurement class of a cause token; an unknown token is logged as `unclassified`."""
    if cause not in DECLINE_CAUSES:
        tools.log_always(f"repair: orchestrator UNVOCABULARISED cause={cause} for "
                         f"{candidate_path} -- this token is not in DECLINE_CAUSES and has no "
                         f"measurement class; add it there. The refusal below stands.\n")
    tools.log_always(f"repair: orchestrator cause={cause} "
                     f"measurement={DECLINE_CAUSES.get(cause, 'unclassified')} "
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
            completed = subprocess.run(
                [tools.software["ffprobe"], "-v", "error", "-show_entries",
                 "format=duration", "-of", "default=nw=1:nk=1", video_obj.filePath],
                capture_output=True, text=True, timeout=120)
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
    except (tools.decoder_timeout, audio_extract.StrictDecodeFailed):
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
            "modality": MODALITY,
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
        entry["why_token"] = WHY_TOKEN.get(entry["kind"])
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
    """Merge adjacent zones at the same offset into one aligned region.

    After this, a hole means a change of offset rather than a stretch the aligner failed to
    match. Each absorbed span is kept in `absorbed_gaps` (bounds on both axes) and counted in
    `unmatched_points_inside`. An edit that removes and adds the same duration inside one gap
    is invisible to any offset instrument.
    """
    if not zones:
        return [], []
    zones, zones_detail, _flicker = merge_quantum_flicker(zones, zones_detail)
    out_zones, out_detail = [], []
    for zone, detail in zip(zones, zones_detail):
        if out_detail and detail["offset_points"] == out_detail[-1]["offset_points"]:
            previous_zone, previous_detail = out_zones[-1], out_detail[-1]
            unmatched = zone[0][0] - previous_zone[0][1] - 1
            if unmatched > 0:
                previous_detail["absorbed_gaps"].append({
                    "master_points": [previous_zone[0][1] + 1, zone[0][0] - 1],
                    "candidate_points": [previous_zone[1][1] + 1, zone[1][0] - 1],
                    "offset_points": detail["offset_points"]})
            previous_zone[0][1] = zone[0][1]
            previous_zone[1][1] = zone[1][1]
            previous_detail["master_points"][1] = detail["master_points"][1]
            previous_detail["candidate_points"][1] = detail["candidate_points"][1]
            previous_detail["master_ms"][1] = detail["master_ms"][1]
            previous_detail["candidate_ms"][1] = detail["candidate_ms"][1]
            previous_detail["n_members"] += detail["n_members"]
            previous_detail["coalesced_zones"] = previous_detail.get("coalesced_zones", 1) + 1
            previous_detail["unmatched_points_inside"] = (
                previous_detail.get("unmatched_points_inside", 0) + max(0, unmatched))
            continue
        out_zones.append([list(zone[0]), list(zone[1])])
        detail = dict(detail)
        detail["master_points"] = list(detail["master_points"])
        detail["candidate_points"] = list(detail["candidate_points"])
        detail["master_ms"] = list(detail["master_ms"])
        detail["candidate_ms"] = list(detail["candidate_ms"])
        detail["coalesced_zones"] = 1
        detail["unmatched_points_inside"] = 0
        detail["absorbed_gaps"] = []
        out_detail.append(detail)
    return out_zones, out_detail


def absorbed_gaps_for_couple(alignment):
    """Return every same-offset gap absorbed by coalescing, in track-relative milliseconds."""
    _zones, detail = coalesce_same_offset_zones(alignment.get("zones") or [],
                                                alignment.get("zones_detail") or [])
    quantum_ms = alignment["quantum_ms"]
    candidate_quantum_ms = alignment.get("candidate_quantum_ms") or quantum_ms
    gaps = []
    for zone in detail:
        for gap in zone["absorbed_gaps"]:
            m_lo, m_hi = gap["master_points"]
            c_lo, c_hi = gap["candidate_points"]
            gaps.append({
                "master_ms": [m_lo * quantum_ms, (m_hi + 1) * quantum_ms],
                "candidate_ms": [c_lo * candidate_quantum_ms, (c_hi + 1) * candidate_quantum_ms],
                "offset_points": gap["offset_points"],
                "offset_ms": gap["offset_points"] * quantum_ms})
    return gaps


def holes_for_couple(alignment):
    """Derive one couple's holes: flicker merge, coalesce, derive, classify, drop non-holes.

    Nearby holes are not merged here (`cluster_holes` groups them). Gaps empty on both axes
    and without a step are dropped.
    """
    _z, _d, flicker = merge_quantum_flicker(alignment.get("zones") or [],
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
    entry["why_token"] = WHY_TOKEN.get(entry["kind"])
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
                if gap < HOLE_MERGE_WINDOW_SECONDS * 1000.0 and (best is None or gap < best_gap):
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


def cluster_holes(holes, per_couple_alignments):
    """Group holes closer than the resolver's reach so they share one scene-detection pass.

    Each hole keeps its own bounds and step. Holes of a multi-hole cluster get `cluster_id`,
    `cluster_window` (shared extraction window) and `scan_cache` (shared memo).

    Returns:
        The clusters, each with its member indices and its islands (master span, offset and
        per-couple similarity of the aligned zone between two holes).
    """
    clusters = []
    for index, hole in enumerate(holes):
        if clusters and (hole["master_ms"][0] - holes[clusters[-1]["members"][-1]]["master_ms"][1]
                         < HOLE_MERGE_WINDOW_SECONDS * 1000.0):
            clusters[-1]["members"].append(index)
        else:
            clusters.append({"members": [index]})
    for number, cluster in enumerate(clusters):
        members = [holes[index] for index in cluster["members"]]
        cluster["cluster_id"] = number
        cluster["islands"] = []
        for left, right in zip(members, members[1:]):
            low, high = left["master_ms"][1], right["master_ms"][0]
            offset = (left["offset_after_ms"] if left["offset_after_ms"] is not None
                      else right["offset_before_ms"])
            similarity = []
            for couple, alignment, fold in per_couple_alignments:
                for detail in alignment.get("zones_detail") or []:
                    z_low = detail["master_ms"][0] + fold["master_start_ms"]
                    z_high = detail["master_ms"][1] + fold["master_start_ms"]
                    if z_low < high and z_high > low:
                        similarity.append({
                            "couple": couple,
                            "master_ms": [round(z_low, 2), round(z_high, 2)],
                            "offset_ms": round(detail["offset_points"] * alignment["quantum_ms"]
                                               + fold["delta_ms"], 3),
                            "mean_match_quality": detail.get("mean_match_quality"),
                            "mean_local_baseline": detail.get("mean_local_baseline")})
            cluster["islands"].append({"master_ms": [low, high], "offset_ms": offset,
                                       "quantum_ms": left["quantum_ms"],
                                       "track_delay_delta_ms": left["track_delay_delta_ms"],
                                       "b2_zones": similarity})
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


def walk_agreement(walk, low_ms, high_ms, offset_ms, tolerance_ms):
    """Count the audio walk's windows inside master [low_ms, high_ms).

    Returns:
        (n_agree, n_ok, n_windows): windows within `tolerance_ms` of `offset_ms`, measured
        windows, and all windows wholly inside the span.
    """
    import audio_walk
    rows = [row for row in walk["rows"]
            if row["t"] * 1000.0 >= low_ms
            and (row["t"] + audio_walk.WALK_WINDOW_S) * 1000.0 <= high_ms]
    ok = [row for row in rows if row["status"] == "ok"]
    agree = [row for row in ok if abs(row["off"] - offset_ms) <= tolerance_ms]
    return len(agree), len(ok), len(rows)


def log_absorbed_gaps(couple_results, union, walk, candidate_path):
    """Log every absorbed same-offset gap of every couple with the evidence for it.

    Gaps at or above ABSORBED_GAP_VIDEO_CHECK_SECONDS are checked against the audio walk (only
    an audio step can change the plan). Actions: under_threshold, inside_a_hole,
    confirmed_by_another_couple, confirmed_by_walk, walk_change_point_inside (resolved as a
    change point elsewhere), unverified_by_walk (too little measured). File clock throughout.
    """
    raw_zones = []
    for record in couple_results:
        alignment, fold = record["alignment"], record["fold"]
        quantum_ms = alignment["quantum_ms"]
        for detail in alignment.get("zones_detail") or []:
            raw_zones.append((record["couple"],
                              detail["master_points"][0] * quantum_ms + fold["master_start_ms"],
                              (detail["master_points"][1] + 1) * quantum_ms
                              + fold["master_start_ms"],
                              detail["offset_points"] * quantum_ms + fold["delta_ms"],
                              quantum_ms))
    changes = [(min(p["level_before"]["t_last"], p["level_after"]["t_first"]) * 1000.0,
                max(p["level_before"]["t_last"], p["level_after"]["t_first"]) * 1000.0)
               for p in walk["points"] if p["kind"] == "change_point"]
    for record in couple_results:
        fold = record["fold"]
        quantum_ms = record["alignment"]["quantum_ms"]
        for gap in absorbed_gaps_for_couple(record["alignment"]):
            low = gap["master_ms"][0] + fold["master_start_ms"]
            high = gap["master_ms"][1] + fold["master_start_ms"]
            offset = gap["offset_ms"] + fold["delta_ms"]
            span_s = (high - low) / 1000.0
            evidence = None
            if span_s < ABSORBED_GAP_VIDEO_CHECK_SECONDS:
                action = "under_threshold"
            elif any(low < hole["master_ms"][1] and high > hole["master_ms"][0]
                     for hole in union):
                action = "inside_a_hole"
            else:
                covered = sum(max(0.0, min(high, z_high) - max(low, z_low))
                              for couple, z_low, z_high, z_offset, z_quantum in raw_zones
                              if couple != record["couple"]
                              and abs(z_offset - offset) <= z_quantum)
                if 2 * covered > (high - low):
                    action = "confirmed_by_another_couple"
                elif any(c_low < high and c_high > low for c_low, c_high in changes):
                    action = "walk_change_point_inside"
                else:
                    agree, measured, windows = walk_agreement(walk, low, high, offset,
                                                              quantum_ms)
                    evidence = {"agree": agree, "ok": measured, "windows": windows}
                    action = ("confirmed_by_walk" if measured and 2 * agree > measured
                              else "unverified_by_walk")
            step_result("absorbed_gap", candidate=candidate_path, couple=record["couple"],
                        master_ms=[round(low, 2), round(high, 2)],
                        span_s=round(span_s, 3), offset_ms=round(offset, 3),
                        threshold_s=ABSORBED_GAP_VIDEO_CHECK_SECONDS, action=action,
                        walk=evidence)

def cross_verify_couples(couple_results):
    """Check that all couples agree on the steps of each interior event.

    Agreement is tested per position cluster on the signed step, never bound for bound, since
    couples place the same event up to ~8 s apart. Rules: head and tail holes are not events;
    a couple with no event in a cluster is `could_not_see`, not dissent; steps under the
    aligner's resolution floor are excluded; only a step-magnitude disagreement declines.

    Returns:
        dict with `agree`, counts, `clusters` (per-cluster verdicts) and `disagreements`.
    """
    events = []
    for record in couple_results:
        alignment = record["alignment"]
        quantum_ms = alignment["quantum_ms"]
        floor_ms = banded_seed_alignment.RESOLUTION_FLOOR_QUANTA * quantum_ms
        for hole in record["holes"]:
            # A zero-step hole is a lost stretch, not a proposed cut.
            if hole["kind"] != "interior" or not hole["step_points"]:
                continue
            events.append({
                "couple": record["couple"],
                "master_position_seconds": hole["master_ms"][0] / 1000.0,
                "step_points": hole["step_points"],
                "step_ms": hole["step_ms"],
                "quantum_ms": quantum_ms,
                "resolution_floor_ms": floor_ms,
                "residual_fraction": alignment.get("residual_fraction"),
                "master_axis_coverage_fraction": alignment.get(
                    "master_axis_coverage_fraction"),
            })

    events.sort(key=lambda e: e["master_position_seconds"])
    clusters = []
    for event in events:
        if clusters and (event["master_position_seconds"] - clusters[-1]["start_seconds"]
                          <= INTERCOUPLE_POSITION_WINDOW_SECONDS):
            clusters[-1]["events"].append(event)
        else:
            clusters.append({"start_seconds": event["master_position_seconds"],
                              "events": [event]})

    all_couples = [record["couple"] for record in couple_results]
    verdicts = []
    disagreements = []
    for index, cluster in enumerate(clusters):
        # One net step per couple per cluster: a cut and its re-add seconds apart can both
        # land in one cluster.
        by_couple = {}
        for event in cluster["events"]:
            net = by_couple.setdefault(event["couple"], dict(event, step_ms=0.0, step_points=0,
                                                             n_events=0))
            net["step_ms"] += event["step_ms"]
            net["step_points"] += event["step_points"]
            net["n_events"] += 1
        members = list(by_couple.values())
        seen = [event["couple"] for event in members]
        could_not_see = [couple for couple in all_couples if couple not in seen]
        tolerance_ms = (INTERCOUPLE_STEP_TOLERANCE_QUANTA * INTERCOUPLE_STEP_TOLERANCE_SLACK
                        * max(event["quantum_ms"] for event in members))
        # Steps under the aligner's resolution floor are not claims about a cut and cannot
        # contradict each other; they stay in the record but out of the test.
        above_floor = [event for event in members
                       if abs(event["step_ms"]) >= event["resolution_floor_ms"]]
        below_floor = [event for event in members if event not in above_floor]
        steps = [event["step_ms"] for event in (above_floor or members)]
        spread_ms = max(steps) - min(steps)
        if not above_floor:
            verdict = "below_floor_only_excluded"
        elif len(above_floor) == 1:
            verdict = "single_couple_event"
        elif spread_ms <= tolerance_ms:
            verdict = "agree"
        else:
            verdict = "disagree"
        entry = {
            "cluster_index": index,
            "master_position_seconds": round(cluster["start_seconds"], 3),
            "verdict": verdict,
            "spread_ms": round(spread_ms, 3),
            "tolerance_ms": round(tolerance_ms, 3),
            "members": members,
            "n_above_floor": len(above_floor),
            "below_floor_excluded": [(event["couple"], round(event["step_ms"], 3))
                                      for event in below_floor],
            "could_not_see": could_not_see,
        }
        verdicts.append(entry)
        if verdict == "disagree":
            disagreements.append(entry)
    return {
        "modality": MODALITY,
        "agree": not disagreements,
        "n_couples": len(couple_results),
        "n_events": len(events),
        "clusters": verdicts,
        "disagreements": disagreements,
    }


def log_cross_verification(candidate_path, report):
    """Log the cross-verification report.

    One unconditional `repair: cross_verify_summary` line, then per-cluster and per-member
    detail in the dev log.
    """
    tools.log_always(
        f"repair: cross_verify_summary agree={report['agree']} n_couples={report['n_couples']} "
        f"n_events={report['n_events']} n_clusters={len(report['clusters'])} "
        f"n_disagreements={len(report['disagreements'])} clusters="
        + ",".join(f"{c['cluster_index']}:{c['verdict']}:{c['master_position_seconds']}:"
                   f"{c['spread_ms']}" for c in report["clusters"])
        + f" for {candidate_path}\n")
    if report["agree"]:
        step_result("cross_verify", candidate=candidate_path, agree=True,
                    n_couples=report["n_couples"], n_events=report["n_events"],
                    clusters=len(report["clusters"]))
        for cluster in report["clusters"]:
            tools.dev_log(
                f"orchestrator: cross_verify cluster={cluster['cluster_index']} "
                f"verdict={cluster['verdict']} "
                f"master_position_s={cluster['master_position_seconds']} "
                f"spread_ms={cluster['spread_ms']} tolerance_ms={cluster['tolerance_ms']} "
                f"could_not_see={cluster['could_not_see']} "
                f"below_floor_excluded={cluster['below_floor_excluded']} "
                f"members={[(e['couple'], round(e['step_ms'], 1)) for e in cluster['members']]}"
                f"\n")
        return
    for cluster in report["disagreements"]:
        tools.dev_log(
            f"orchestrator: intercouple_disagreement for {candidate_path} "
            f"cluster={cluster['cluster_index']} "
            f"master_position_s={cluster['master_position_seconds']} "
            f"spread_ms={cluster['spread_ms']} tolerance_ms={cluster['tolerance_ms']} "
            f"n_above_floor={cluster['n_above_floor']} "
            f"below_floor_excluded={cluster['below_floor_excluded']} "
            f"could_not_see={cluster['could_not_see']}\n")
        for event in cluster["members"]:
            tools.dev_log(
                f"orchestrator: intercouple_disagreement_member "
                f"for {candidate_path} cluster={cluster['cluster_index']} "
                f"couple={event['couple']} "
                f"master_position_s={round(event['master_position_seconds'], 3)} "
                f"step_points={event['step_points']} "
                f"step_ms={round(event['step_ms'], 3)} "
                f"quantum_ms={round(event['quantum_ms'], 4)} "
                f"resolution_floor_ms={round(event['resolution_floor_ms'], 3)} "
                f"residual_fraction={event['residual_fraction']} "
                f"coverage={event['master_axis_coverage_fraction']}\n")


# ---------------------------------------------------------------------------
# Rate re-prime: pitch layer and speed-corrected candidate
# ---------------------------------------------------------------------------

def _pitch_probe_window(master_obj, candidate_obj, language, master_stream, candidate_stream,
                        speed_factor, work_dir, sample_rate):
    """Extract content-aligned master and candidate WAVs for the pitch layer.

    Pre-extracted with an explicit stream map, since `pal_pitch_confirmer` would otherwise read
    ffmpeg's default stream. Master instant t maps to candidate instant t / speed_factor. The
    candidate is not filtered: the pitch layer measures its original pitch.

    Returns:
        (master_wav, candidate_wav, window_seconds), or (None, None, None).
    """
    master_duration = _track_duration_seconds(master_obj, language, master_stream)
    candidate_duration = _track_duration_seconds(candidate_obj, language, candidate_stream)
    if master_duration is None or candidate_duration is None:
        return None, None, None
    ratio = float(speed_factor)
    window = PITCH_PROBE_WINDOW_SECONDS
    usable = min(master_duration, candidate_duration * ratio)
    if usable <= window:
        window = max(PITCH_PROBE_WINDOW_MINIMUM_SECONDS, usable / 2.0)
        if usable <= window:
            return None, None, None
    master_start = max(0.0, (usable - window) / 2.0)
    candidate_start = master_start / ratio
    master_wav = path.join(work_dir, f"orch_pitch_master_{master_stream}.wav")
    candidate_wav = path.join(work_dir, f"orch_pitch_candidate_{candidate_stream}.wav")
    try:
        audio_extract.extract_audio_window(master_obj.filePath, master_stream, master_start,
                                           window, master_wav, sample_rate)
        audio_extract.extract_audio_window(candidate_obj.filePath, candidate_stream,
                                           candidate_start, window / ratio, candidate_wav,
                                           sample_rate)
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"orchestrator: pitch probe windows unextractable "
                      f"({type(error).__name__}: {error}) -- the pitch layer will not be asked, "
                      f"and not asking is recorded as not asking\n")
        for temporary in (master_wav, candidate_wav):
            try:
                remove(temporary)
            except OSError:
                pass
        return None, None, None
    return master_wav, candidate_wav, window


def pitch_routing(speed_factor, master_obj, candidate_obj, language, work_dir, sample_rate):
    """Read the pitch layer for a speed correction and choose the resample filter.

    `asetrate` (undoes speed and pitch together) is routed on every reachable outcome: it
    returned the pitch to 0 cents on every real PAL and NTSC sample. The measured ratio is
    recorded for calibration; no detector exists for a source pitch-corrected at origin.

    Returns:
        A routing dict, never None; a pitch layer that could not run records its refusal.
    """
    routing = {
        "filter_name": "asetrate",
        "pitch_measured_ratio": None,
        "pitch_peak": None,
        "pitch_refusal": None,
        "pitch_window_seconds": None,
        "pitch_test_discriminating": None,
        "pitch_tolerance_band": None,
        "inverting_case_detector": "not_implemented",
        "inverting_case_observation": None,
    }
    master_streams = audio_extract.streams_for(master_obj, language)
    candidate_streams = audio_extract.streams_for(candidate_obj, language)
    if not master_streams or not candidate_streams:
        routing["pitch_refusal"] = "no_stream_to_probe"
        routing["route_reason"] = (
            "the pitch layer was not asked: there is no comparison-language stream pair to probe "
            "it on. asetrate stands as the policy default (AUDIO_SPEED_POLICY 23/23 PAL, 6/6 "
            "NTSC), and 'not asked' is recorded as not asked, never as 'pitch intact'")
        return routing
    master_wav, candidate_wav, window = _pitch_probe_window(
        master_obj, candidate_obj, language, master_streams[0], candidate_streams[0],
        speed_factor, work_dir, sample_rate)
    if master_wav is None:
        routing["pitch_refusal"] = "probe_window_unavailable"
        routing["route_reason"] = (
            "the pitch layer was not asked: no window could be taken inside both tracks. "
            "asetrate stands as the policy default, and 'not asked' is not 'pitch intact'")
        return routing
    routing["pitch_window_seconds"] = round(window, 3)
    try:
        import pal_pitch_confirmer
        tools.dev_log(f"orchestrator: calling pal_pitch_confirmer.confirm_pitch "
                      f"master_wav={master_wav} candidate_wav={candidate_wav} "
                      f"predicted_ratio={float(speed_factor)} window_seconds={window}\n")
        reading = pal_pitch_confirmer.confirm_pitch(
            master_wav, candidate_wav, 0.0, float(speed_factor), window_seconds=window)
    except Exception as error:                                           # noqa: BLE001
        routing["pitch_refusal"] = "pitch_layer_raised"
        routing["route_reason"] = (
            f"the pitch layer raised {type(error).__name__} -- the instrument did not run, which "
            f"is not a reading about the pitch. asetrate stands as the policy default")
        tools.dev_log(f"orchestrator: pal_pitch_confirmer raised "
                      f"({type(error).__name__}: {error})\n")
        return routing
    finally:
        for temporary in (master_wav, candidate_wav):
            try:
                remove(temporary)
            except OSError:
                pass
    routing["pitch_measured_ratio"] = reading.get("measured_ratio")
    routing["pitch_peak"] = reading.get("peak")
    routing["pitch_refusal"] = reading.get("refusal")
    measured = reading.get("measured_ratio")
    if measured is not None:
        # Recorded for a future inverting-case detector ("duration moved, pitch did not"
        # reads `measured` near 1.0 while `speed_factor` is far from it).
        routing["inverting_case_observation"] = {
            "measured_from_unity": round(abs(measured - 1.0), 6),
            "applied_from_unity": round(abs(float(speed_factor) - 1.0), 6),
        }
    # The pitch test discriminates only when its tolerance band (TOL_ARM * applied) excludes
    # unity, i.e. TOL_ARM * applied < |applied - 1|: true at PAL, false at NTSC.
    try:
        import pal_pitch_confirmer as _confirmer
        tolerance_arm = float(_confirmer.TOL_ARM)
    except Exception:                                                    # noqa: BLE001
        tolerance_arm = 0.0030
    applied = float(speed_factor)
    band_half_width = tolerance_arm * applied
    discriminating = band_half_width < abs(applied - 1.0)
    routing["pitch_test_discriminating"] = discriminating
    routing["pitch_tolerance_band"] = round(band_half_width, 7)
    if reading.get("refusal") is None and discriminating:
        routing["route_reason"] = (
            f"the pitch layer confirms the pitch moved with the speed (measured {measured}, "
            f"applied {applied:.7f}, peak {reading.get('peak')}; its +/-{band_half_width:.6f} "
            f"tolerance band excludes unity at this ratio, so agreement is informative): this "
            f"is the naive-speedup family, and asetrate is its exact inverse -- it undoes speed "
            f"AND pitch together")
    elif reading.get("refusal") is None:
        routing["route_reason"] = (
            f"the pitch layer did not refuse (measured {measured}, applied {applied:.7f}, peak "
            f"{reading.get('peak')}) BUT THAT IS NOT A CONFIRMATION AT THIS RATIO: its "
            f"+/-{band_half_width:.6f} tolerance band is wider than the {abs(applied - 1.0):.6f} "
            f"deviation being corrected, so the band contains unity and a completely unshifted "
            f"pitch would have passed the same test. Nothing here says the pitch moved. asetrate "
            f"stands on the policy default (AUDIO_SPEED_POLICY 23/23 PAL, 6/6 NTSC), not on this "
            f"reading")
    else:
        routing["route_reason"] = (
            f"the pitch layer returned {reading.get('refusal')} ({reading.get('reason')}). That "
            f"is NOT the inverting case and must not be read as one -- the inverting-case "
            f"detector is unimplemented in this tree (merge_video_repair:229-233), so routing "
            f"to atempo stays disabled until it is. The existing routing stands: "
            f"asetrate, on AUDIO_SPEED_POLICY's 23/23 PAL and 6/6 NTSC")
    return routing


def rate_resample_routing(speed_factor, engine, master_obj, candidate_obj, language, work_dir,
                          sample_rate):
    """Build the speed-correction routing for re-priming the candidate at a confirmed factor.

    The routing is a description, not a file: `fingerprint_track` applies its filter chain
    while extracting. Ratios are exact; `asetrate` takes an integer rate, so the effective
    ratio (carried as `effective_ratio`) differs slightly from the requested one and every
    downstream length uses the effective one. Never called at factor 1.

    Returns:
        (routing, None) or (None, cause).
    """
    source_rate = _candidate_audio_sample_rate(candidate_obj)
    if source_rate is None:
        return None, "rate_sweep_no_sample_rate"
    # Built at the source rate, not the comparison rate: a higher rate leaves a smaller
    # integer-asetrate residual.
    try:
        import merge_video_resample
        ratio_decimal = (Decimal(speed_factor.numerator) / Decimal(speed_factor.denominator)
                         if isinstance(speed_factor, Fraction) else Decimal(str(speed_factor)))
        chain, effective = merge_video_resample.build_transform_chain(
            source_rate, ratio_decimal, engine)
        intermediate = target = None
        if engine == "asetrate":
            _chain, _effective, intermediate, target = (
                merge_video_resample.build_speed_filter_chain(source_rate, ratio_decimal))
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"orchestrator: build_speed_filter_chain refused "
                      f"({type(error).__name__}: {error})\n")
        return None, "rate_resample_unbuildable"
    routing = pitch_routing(speed_factor, master_obj, candidate_obj, language, work_dir,
                            sample_rate)
    # The engine is the rate arm's measurement; the pitch reading stays as an observation.
    routing["route_reason"] = (f"engine {engine} measured by the rate arm (the finalist that "
                               f"aligned); pitch layer: {routing.get('route_reason')}")
    routing["filter_name"] = engine
    routing["inverting_case_detector"] = "rate_arm_engine_comparison"
    routing.update({
        "modality": MODALITY,
        "side": "candidate",
        "rule": "reprime_extraction_only_never_a_product_track",
        "requested_ratio": (f"{speed_factor.numerator}/{speed_factor.denominator}"
                            if isinstance(speed_factor, Fraction) else str(speed_factor)),
        "requested_ratio_value": float(speed_factor),
        "source_sample_rate": source_rate,
        "comparison_sample_rate": sample_rate,
        "filter_chain": chain,
        "intermediate_rate": intermediate,
        "asetrate_target": target,
        "effective_ratio": effective,
        "effective_ratio_str": str(effective),
        "tag_factor": merge_video_resample.format_factor(effective),
    })
    return routing, None


def _candidate_audio_sample_rate(candidate_obj):
    """Return the candidate's audio sample rate (ffprobe, then MediaInfo), or None.

    Never a default: a guessed source rate would make the filter's effective factor silently
    wrong.
    """
    for _language, audios in (getattr(candidate_obj, "audios", None) or {}).items():
        for audio in audios:
            rate = audio.get("ffprobe", {}).get("sample_rate") or audio.get("SamplingRate")
            if rate is not None:
                try:
                    return int(float(rate))
                except (TypeError, ValueError):
                    continue
    return None


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

    scale = _decimal(Fraction(speed_factor)) if speed_factor not in (None, 1) else Decimal(1)
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
    outcome = {"modality": MODALITY, "status": HOLE_DECLINED, "kind": hole["kind"],
               "why_token": hole["why_token"], "cause": "hole_resolution_declined",
               "resolver_reason": cause_detail, "evidence": evidence}
    outcome.update(extra)
    return outcome


def _two_anchor_call(hole, domain, master_obj, candidate_obj, low_ms, high_ms,
                     offset_before_ms, offset_after_ms, step_ms, quantum_ms, probe,
                     resolve_shift=True, accept_step_disagreement=False):
    """Run the interior resolver once as a logged step; an exception becomes a named decline."""
    import scene_anchor
    deadline = hole.get("deadline")
    if deadline is not None and time.monotonic() > deadline:
        return {"declined": True, "reason": "hole_budget_exceeded",
                "evidence": f"probe={probe} not launched: the hole's budget is spent"}
    step_launch("two_anchor", candidate=candidate_obj.filePath, probe=probe,
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
    step_result("two_anchor", candidate=candidate_obj.filePath, probe=probe,
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
    reading = _span_noise_reading(result)
    if reading is None:
        return result
    frame_ms = float(domain["frame_ms"])
    step_result("sweep_front_inside_common_content", candidate=candidate_obj.filePath,
                probe=probe, claimed_by=reading,
                unmatched_master_frames=[result["pre_collapse_start_master"],
                                         result["pre_collapse_end_master"]],
                matches_before=result["unmatched_span_matches_before"],
                matches_after=result["unmatched_span_matches_after"])
    if reading == "both" and result["before_shift_frames"] == result["after_shift_frames"]:
        # One shift on both sides matching the span: the walks stopped on pHash noise, so
        # nothing differs between the anchors (read as `no_cut_confirmed`).
        step_result("single_shift_claims_span", candidate=candidate_obj.filePath, probe=probe,
                    shift=result["before_shift_frames"],
                    matches=result["unmatched_span_matches_before"])
        return dict(result, span_majority_under_single_shift=True)
    if reading == "both":
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
            return HOLE_PINNED_TO_AMBIGUOUS_ZONE_END
        return HOLE_DECLINED
    same_shift = result["before_shift_frames"] == result["after_shift_frames"]
    if same_shift and (result["master_end_frame"] == result["master_start_frame"]
                       or result.get("span_majority_under_single_shift")):
        return HOLE_NO_CUT_CONFIRMED
    candidate_overlap = result["candidate_end_frame"] < result["candidate_start_frame"]
    if not same_shift and (result["sweep_crossed"] or candidate_overlap):
        return HOLE_PINNED_TO_AMBIGUOUS_ZONE_END
    return HOLE_RESOLVED


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
    if status == HOLE_PINNED_TO_AMBIGUOUS_ZONE_END:
        master_start, master_end, candidate_start, candidate_end = _pinned_frames(result)
    length_master = master_end - master_start
    length_candidate = candidate_end - candidate_start
    net_kind = ("addition" if length_candidate > length_master
                else "deletion" if length_candidate < length_master
                else "still_image" if length_master == 0 else "ordinary")
    outcome = {
        "modality": MODALITY, "status": status, "kind": hole["kind"],
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
    if status == HOLE_PINNED_TO_AMBIGUOUS_ZONE_END:
        outcome["cause"] = "static_span_ambiguity"
        outcome["pin_frame"], outcome["ambiguous_frames"] = _pin_point(result)
    if refuted_proposal is not None:
        outcome["refuted_proposal"] = refuted_proposal
    return outcome


def _ambiguous_pin_outcome(hole, domain, result, candidate_path):
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
                pre_collapse_start_master=anchor_a, pre_collapse_end_master=anchor_b)
    outcome = _blind_span_outcome(hole, domain, blind, candidate_path)
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
    step_result("hole_span_no_cut_test", candidate=candidate_obj.filePath,
                master_ms=[round(float(low_ms), 2), round(float(high_ms), 2)],
                audio_step_ms=round(hole["step_ms"], 3), proposal_reason=result.get("reason"),
                **{f"span_{key}": value for key, value in reading.items()})
    if reading["verdict"] != "same":
        return None
    shift = reading["shift_frames"]
    first, last = reading["master_frames"]
    outcome = {
        "modality": MODALITY, "status": HOLE_NO_CUT_CONFIRMED, "kind": hole["kind"],
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
            "video_verdict": HOLE_NO_CUT_CONFIRMED,
            "proposal_shifts": [result.get("before_shift_frames"),
                                result.get("after_shift_frames")],
            "proposal_span_matches_before": None, "proposal_span_matches_after": None,
            "claimed_span_matches": None,
            "span_test_matches": [reading["matched"], reading["readable"]],
            "surviving_shift_walk_frames": [None, None],
            "surviving_shift_span_frames": last - first},
    }
    return outcome


def _blind_span_outcome(hole, domain, blind, candidate_path):
    """Resolve a hole the picture could not read as a blind (black/static) span.

    Both anchors seated, scene-cut-seeded and pHash-validated; the cross-sweep from each ran
    clean through the whole span and touched the far anchor (`sweep_crossed`), which is what a
    held black or static picture produces -- it matches either offset, so neither walk ever
    finds a mismatch to stop on. The picture is never read past the anchors in that case: the
    width is the anchors' own shift difference (in whole frames), and the edit sits just before
    Anchor B's first frame.

    `locate_scene_anchors` collapses both axes to anchor B (`master_start_frame ==
    master_end_frame == anchor_b_frame`), which loses the gap whenever it must fill from the
    master (`video_cut_instant` would then read a 0 ms video width against a non-zero audio
    fill and always decline). A master fill needs the master span widened to the gap, ending at
    anchor B; a candidate removal needs no master span at all (the master is untouched), so its
    collapsed shape is already correct. Either way the candidate side stays a single point at
    anchor B's own frame, under the shift that survives (`after_shift_frames`): nothing in the
    candidate is read twice and nothing of it is kept inside the removed/filled span.
    """
    before, after = blind["before_shift_frames"], blind["after_shift_frames"]
    gap = after - before
    anchor_b = blind["anchor_b_frame"]
    frame_ms = float(domain["frame_ms"])
    if gap < 0:
        placed = dict(blind, master_start_frame=anchor_b + gap, master_end_frame=anchor_b,
                     candidate_start_frame=anchor_b + after, candidate_end_frame=anchor_b + after)
        decision = "fill_from_master"
    else:
        placed = dict(blind, master_start_frame=anchor_b, master_end_frame=anchor_b,
                     candidate_start_frame=anchor_b + before, candidate_end_frame=anchor_b + after)
        decision = "remove_from_candidate"
    tools.log_always(
        f"repair: video_undecided_blind_span anchor_a={blind['anchor_a_frame']} "
        f"anchor_b={anchor_b} before_shift_frames={before} "
        f"after_shift_frames={after} gap_frames={gap} "
        f"gap_ms={round(gap * frame_ms, 3)} "
        f"decision={decision} "
        f"placed_before_frame={anchor_b} for {candidate_path}\n")
    return _interior_outcome(hole, domain, placed, HOLE_RESOLVED)


def _resolve_interior(hole, domain, master_obj, candidate_obj):
    """Resolve an interior hole with the two-anchor frame-exact search.

    The audio proposes a step, the video decides. For a step under the aligner's resolution
    floor, the video is also offered a single shift on both sides (anchor A's, then B's, or the
    zones' audio offsets when no anchor resolved); the first that seats both anchors refutes
    the step. A step at or above the floor is likewise re-tested at a shift that claims the
    unmatched span. An ambiguous anchor leads to `_ambiguous_pin_outcome`, a sub-floor decline
    to `_span_no_cut_outcome`.
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
        step_result("video_refutes_audio_step", candidate=candidate_obj.filePath,
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
        # Under the resolution floor the audio step is noise, so an ambiguous anchor is not
        # pinned; the span itself is asked the no-cut question (a static span matches any shift).
        if below_floor:
            span_outcome = _span_no_cut_outcome(hole, domain, master_obj, candidate_obj,
                                                low_ms, high_ms, result)
            if span_outcome is not None:
                return span_outcome
        elif _interior_verdict(result) == HOLE_PINNED_TO_AMBIGUOUS_ZONE_END:
            return _ambiguous_pin_outcome(hole, domain, result, candidate_obj.filePath)
        elif result.get("reason") == "anchor_step_inconsistent":
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
                return _blind_span_outcome(hole, domain, blind, candidate_obj.filePath)
        return _declined(hole, result.get("reason"), result.get("evidence"),
                         no_cut_probe_run=bool(hypotheses))
    return _interior_outcome(hole, domain, result, _interior_verdict(result))


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
    step_launch("edge_walk", candidate=candidate_obj.filePath, edge=edge,
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
    step_result("edge_walk", candidate=candidate_obj.filePath, edge=edge,
                declined=result["declined"], reason=result.get("reason"),
                termination=result.get("termination"), anchor=result.get("anchor_frame"),
                shift=result.get("shift_frames"), nominal_shift=result.get("nominal_shift_frames"),
                boundary=result.get("boundary_frame"), walked=result.get("walked_frames"),
                mismatch_run=result.get("mismatch_run"),
                max_mismatch_run=result.get("max_mismatch_run"),
                addition_frames=result.get("addition_frames"),
                evidence=result.get("evidence"))
    if result["declined"]:
        return _declined(hole, result.get("reason"), result.get("evidence"))

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
                            if termination == EDGE_CANDIDATE_EXHAUSTED else None)
        master_non_common = master_last - boundary
        candidate_non_common = (0 if termination == EDGE_CANDIDATE_EXHAUSTED else None)

    # Master frames added to the output (a replacement or addition; a trim adds nothing).
    added_frames = 0 if termination == EDGE_MASTER_EXHAUSTED else max(0, master_non_common)
    added_seconds = float(Fraction(added_frames) / domain["master_rate"])

    status = termination
    # Both files reach their edge at the same step: nothing to trim or add.
    if master_non_common == 0 and candidate_non_common == 0:
        status = HOLE_NO_CUT_CONFIRMED
    return {
        "modality": MODALITY, "status": status, "kind": edge, "why_token": hole["why_token"],
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


def resolve_hole(hole, master_obj, candidate_obj, work_dir):
    """Resolve one hole to exact frames.

    Interior holes go to `_resolve_interior`, head and tail holes to `_resolve_edge`. The hole
    must carry `frame_domain` and `quantum_ms`; it gets HOLE_BUDGET_S, capped by the repair's
    deadline. `work_dir` is unused (the resolvers decode through pipes).

    Returns:
        An outcome dict whose `status` is in HOLE_STATUSES_WITH_FRAMES, or `declined` with a
        named `resolver_reason`.
    """
    domain = hole.get("frame_domain")
    if domain is None:
        return _declined(hole, "frame_domain_absent")
    repair_deadline = domain.get("repair_deadline")
    if repair_deadline is not None and time.monotonic() > repair_deadline:
        return _declined(hole, "repair_budget_exceeded")
    deadline = time.monotonic() + HOLE_BUDGET_S
    hole = dict(hole, deadline=(deadline if repair_deadline is None
                                else min(deadline, repair_deadline)))
    if hole["kind"] == "interior":
        outcome = _resolve_interior(hole, domain, master_obj, candidate_obj)
    elif hole["kind"] in ("head", "tail"):
        outcome = _resolve_edge(hole, domain, master_obj, candidate_obj)
    else:
        outcome = _declined(hole, f"no_resolver_for_kind:{hole['kind']}")
    # An over-budget hole declines even if its last call answered.
    if outcome["status"] != HOLE_DECLINED and time.monotonic() > hole["deadline"]:
        outcome = _declined(hole, "hole_budget_exceeded",
                            evidence=f"answered {outcome['status']} past the "
                                     f"{HOLE_BUDGET_S} s budget")
    outcome["audio_master_frames"] = [_master_frame_of_ms(hole["master_ms"][0], domain),
                                      _master_frame_of_ms(hole["master_ms"][1], domain)]
    outcome["audio_candidate_frames"] = [
        _candidate_native_frame_of_ms(hole["candidate_ms"][0], domain),
        _candidate_native_frame_of_ms(hole["candidate_ms"][1], domain)]
    outcome["audio_step_ms"] = (None if hole["step_ms"] is None else round(hole["step_ms"], 3))
    return outcome


def _resolve_logged(candidate_path, index, hole, domain, master_obj, candidate_obj, work_dir):
    """Run `resolve_hole` with step logging; an out-of-vocabulary status becomes a decline."""
    step_launch("resolve_hole", candidate=candidate_path, hole=index, kind=hole["kind"],
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
    outcome = resolve_hole(dict(hole, frame_domain=domain), master_obj, candidate_obj, work_dir)
    if outcome["status"] not in HOLE_STATUSES_WITH_FRAMES + (HOLE_DECLINED,):
        tools.log_always(f"repair: orchestrator UNVOCABULARISED hole status="
                         f"{outcome['status']} hole={index} for {candidate_path} -- "
                         f"treated as declined\n")
        outcome = dict(outcome, status=HOLE_DECLINED, cause="hole_resolution_declined",
                       resolver_reason=f"unvocabularised_status:{outcome['status']}")
    step_result("resolve_hole", candidate=candidate_path, hole=index, kind=hole["kind"],
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
        step_result("refuted_proposal", candidate=candidate_path, hole=index, **refuted)
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
    if outcome["status"] == HOLE_NO_CUT_CONFIRMED:
        tools.log_always(
            f"repair: no_cut_confirmed hole={index} kind={hole['kind']} "
            f"origin={hole.get('origin', 'alignment')} "
            f"audio_master_frames={outcome.get('audio_master_frames')} "
            f"audio_step_ms={outcome.get('audio_step_ms')} "
            f"shift={outcome.get('before_shift_frames', outcome.get('shift_frames'))} "
            f"anchors={[outcome.get('anchor_a_frame', outcome.get('anchor_frame')), outcome.get('anchor_b_frame')]} "
            f"walks={[outcome.get('forward_walk_frames', outcome.get('walked_frames')), outcome.get('backward_walk_frames')]} "
            f"for {candidate_path}\n")
    if outcome["status"] == HOLE_PINNED_TO_AMBIGUOUS_ZONE_END:
        step_result("boundary_pinned_to_ambiguous_zone_end", candidate=candidate_path,
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
        elif hole["kind"] == "interior" and (high - low) / 1000.0 > INTERIOR_HOLE_MAX_SPAN_S:
            verdict = ("interior_hole_exceeds_budget",
                       f"interior hole {index} spans {round((high - low) / 1000.0, 1)} s of the "
                       f"master, over the {INTERIOR_HOLE_MAX_SPAN_S} s bound")
        if verdict is not None:
            step_result("hole_sanity", candidate=candidate_path, hole=index, kind=hole["kind"],
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
    slices = started_slices(video_s, REPAIR_BUDGET_SLICE_S)
    return min(slices * REPAIR_BUDGET_PER_SLICE_S, REPAIR_BUDGET_CAP_S), video_s


def max_holes_per_couple(master_obj):
    """Return the hole cap per couple: MAX_HOLES_PER_COUPLE per started HOLE_BUDGET_SLICE_S."""
    video_ms = _video_duration_ms(master_obj)
    video_s = None if video_ms is None else float(video_ms) / 1000.0
    return started_slices(video_s, HOLE_BUDGET_SLICE_S) * MAX_HOLES_PER_COUPLE


def _budget_terminal(candidate_path, step, budget_s):
    """Decline the candidate because the repair's budget ran out between two steps."""
    _plan_line("none", candidate_path, step=step, cause="repair_budget_exceeded")
    return _terminal(candidate_path, "no_plan", "repair_budget_exceeded",
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
                   candidate_path, deadline=None, engine="asetrate"):
    """Run the millisecond audio walk on the reference couple, on the file clock.

    Seeded by every alignment offset of the reference couple's zones and the union's holes.
    A whole-track read stopped by `deadline` re-raises (cause `repair_budget_exceeded`).

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
    step_launch("audio_walk", candidate=candidate_path, couple=reference["couple"],
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
    points = audio_walk.change_points(master, candidate, found)
    seconds = time.time() - started
    audio_walk.log_walk(candidate_path, rows, found, points, seconds)
    step_result("audio_walk", candidate=candidate_path, n_windows=len(rows),
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


def audio_edges(walk, holes, domain, master_obj, candidate_obj, work_dir, candidate_path):
    """Place the head and tail fill edges from the audio walk, pinned to a video frame if close.

    `audio_walk.single_edge` walks outward from the first/last level at 20 ms: a witness only,
    to tell the resolver where to search and to confirm its answer in the dev log. Where the
    one-anchor walk (`scene_anchor.locate_edge_boundary`, via `_resolve_edge`) concludes, its own
    frame is delivered unconditionally -- never nudged onto the audio edge, never raised for the
    candidate's lead-in, which was itself an audio placement. Only where no such hole was raised
    for this edge (nothing flagged it as needing a frame-exact search) does the walk's own edge
    stand, with the lead-in floor still applied to it. A one-anchor walk that could not conclude
    declines by name instead of falling back to the audio's edge.

    Returns:
        (head_end_s, tail_start_s, refusal); an edge is None when no fill is needed there.
    """
    import audio_walk
    master, candidate = walk["master"], walk["candidate"]
    first, last = walk["levels"][0], walk["levels"][-1]
    frame = float(domain["frame_ms"]) / 1000.0
    reach = audio_walk.WALK_WINDOW_S + audio_walk.WALK_HOP_S + HOLE_MERGE_WINDOW_SECONDS
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
            if outcome["status"] in EDGE_TERMINATIONS or outcome["status"] == HOLE_NO_CUT_CONFIRMED:
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
    return {"modality": MODALITY, "kind": "interior", "why_token": WHY_TOKEN["interior"],
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
               and abs(hole["step_ms"] - jump) <= (INTERCOUPLE_STEP_TOLERANCE_QUANTA
                                                   * INTERCOUPLE_STEP_TOLERANCE_SLACK
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
    if outcome.get("status") != HOLE_RESOLVED:
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

    A resolved hole whose width matches the audio fill (within one quantum + two frames)
    offers its first cut. Otherwise only a deletion whose video span reads two fills, with an
    audio interval wider than a frame, offers its last cut minus the fill, if inside the
    interval.

    Returns:
        (video_s or None, width_note or None)
    """
    if outcome.get("status") not in (HOLE_RESOLVED, HOLE_PINNED_TO_AMBIGUOUS_ZONE_END):
        return None, None
    frame_ms = float(domain["frame_ms"])
    tolerance_ms = quantum_ms + 2 * frame_ms
    start_s = _frame_s(outcome["master_start_frame"], domain)
    forward_walk = outcome.get("forward_walk_frames")
    backward_walk = outcome.get("backward_walk_frames")
    if forward_walk == 0 and backward_walk == 0:
        # Neither walk advanced from its anchor: nothing inside the pair located a boundary, so
        # the raw inter-anchor master span is not a found fill -- it is only the anchors' own
        # positions. The anchors' own shift gap (after - before) is the real measured quantity,
        # and a removal (gap <= 0) has zero master fill by construction.
        video_fill_ms = max(0, outcome["after_shift_frames"]
                            - outcome["before_shift_frames"]) * frame_ms
    else:
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
    disagreement, logged and declined as `owner_judgment_pending` with both positions; a video
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
    # Nearby change points share one scene pass; islands between them are checked by the walk.
    import audio_walk
    for cluster in cluster_holes(holes, [(reference["couple"], reference["alignment"],
                                          reference["fold"])]):
        islands = []
        for island in cluster["islands"]:
            agree, measured, windows = walk_agreement(
                walk, island["master_ms"][0], island["master_ms"][1],
                island["offset_ms"], audio_walk.LEVEL_TOLERANCE_MS)
            islands.append({"master_ms": [round(island["master_ms"][0], 2),
                                          round(island["master_ms"][1], 2)],
                            "offset_ms": round(island["offset_ms"], 3),
                            "b2_zones": island["b2_zones"],
                            "walk": {"agree": agree, "ok": measured, "windows": windows}})
        step_result("cluster", candidate=candidate_path, cluster=cluster["cluster_id"],
                    members=cluster["members"], shared_scan=len(cluster["members"]) > 1,
                    islands=islands)
    # Every hole's exact-frame search is independent (holes under the merge window share only
    # a read-through decode cache, never a decision), so they run together on the shared pool
    # instead of one change point at a time.
    hole_outcomes = repair_pool.run_parallel([
        (lambda i=index, h=hole: _resolve_logged(candidate_path, f"change_point_{i}", h, domain,
                                                 master_obj, candidate_obj, work_dir))
        for index, hole in enumerate(holes)])
    transitions, disagreements = [], []
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
        if not edges["feasible"]:
            replacement = replacement_hole(outcome, domain, edges,
                                           hole["quantum_ms"] / 1000.0 + 2 * frame)
            if replacement is None:
                return None, ("hole_width_contradicts_audio_step",
                              f"the {round(edges['extra_s'] * 1000, 3)} ms of master content the "
                              f"candidate lacks at {edges['edge_A']}-{edges['edge_B']} s does not "
                              f"fit between the audio edges around its audible master-only sound "
                              f"(interval {edges['interval']}), and the video does not pin a "
                              f"replacement (status {status}, "
                              f"{outcome.get('resolver_reason') or 'frames outside the edges or an excess of 1 s or more'})")
            transitions.append({"at_s": replacement["start_s"], "fill_s": replacement["fill_s"],
                                "a_ms": point["a_ms"], "b_ms": point["b_ms"],
                                "decision": "replacement_hole", "interval": [lo, hi],
                                "edges": [edges["edge_A"], edges["edge_B"]],
                                "video_status": status, "video_s": replacement["start_s"],
                                "change_point": index, "cut_ms": replacement["cut_ms"]})
            tools.dev_log(
                f"repair: replacement_hole change_point={index} a_ms={point['a_ms']} "
                f"b_ms={point['b_ms']} step_ms={round(jump, 3)} audio_edges_s=[{edges['edge_A']}, "
                f"{edges['edge_B']}] video_edges_s=[{replacement['start_s']}, "
                f"{replacement['end_s']}] fill_ms={round(replacement['fill_s'] * 1000, 3)} "
                f"cut_ms={replacement['cut_ms']} -- the candidate's own excess is cut, the "
                f"master-only span is filled from the master for {candidate_path}\n")
            continue
        video_s, width_note = video_cut_instant(outcome, domain, extra, (lo, hi),
                                                hole["quantum_ms"])
        video_decided = status in (HOLE_RESOLVED, HOLE_PINNED_TO_AMBIGUOUS_ZONE_END)
        if status == HOLE_DECLINED:
            # No compatible anchor, a static/black span, a geometry mismatch...: the video
            # could neither place a cut here nor rule one out. A declined video reading is
            # never delivered as an unverified audio-only cut -- the whole plan declines,
            # named and explicit.
            return None, ("sub_quantum_step_video_ambiguous" if sub_quantum
                          else "video_cut_undetermined",
                          f"the walk measured a {round(jump, 3)} ms step in [{lo}, {hi}] s and "
                          f"the video could neither place a cut nor confirm there is none "
                          f"({outcome.get('resolver_reason')})")
        fill = extra
        # The witness bound is the audio edges/interval as measured, never narrowed by a
        # candidate-leak reading: the cut's own position and width come from the video's
        # anchors and pHash walk, not from audio.
        pinned, decision = video_pin(video_s, (lo, hi), (edges["edge_A"], edges["edge_B"]),
                                     extra, frame)
        if pinned is not None:
            at = pinned
            if width_note:
                decision = "video_cut_edge_pins_" + decision
        elif video_decided:
            # The video resolved a cut (or pinned a static span) but the audio witness does not
            # confirm it within tolerance (one frame, `video_pin`) -- a measured disagreement,
            # collected for the owner instead of a silent audio-only delivery.
            disagreements.append({
                "zone": index, "reason": "video_audio_disagree",
                "master_start_s": hole["master_ms"][0] / 1000.0,
                "master_end_s": hole["master_ms"][1] / 1000.0,
                "candidate_start_s": hole["candidate_ms"][0] / 1000.0,
                "candidate_end_s": hole["candidate_ms"][1] / 1000.0,
                "audio_cut_s": audio_walk.quietest_instant(walk["master"], lo, hi, extra),
                "video_cut_s": _frame_s(outcome["master_start_frame"], domain),
                "picture_shift_ms": None, "frames_compared": outcome.get("span_frames")})
            tools.dev_log(
                f"repair: video_audio_disagree change_point={index} "
                f"{width_note or 'video cut outside the audio interval and edges'} "
                f"for {candidate_path}\n")
            at = audio_walk.quietest_instant(walk["master"], lo, hi, extra)
            decision = "video_audio_disagree_pending"
        elif status == HOLE_NO_CUT_CONFIRMED:
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
        transitions.append({"at_s": at, "fill_s": fill, "a_ms": point["a_ms"],
                            "b_ms": point["b_ms"], "decision": decision, "interval": [lo, hi],
                            "edges": [edges["edge_A"], edges["edge_B"]],
                            "video_status": status, "video_s": video_s, "change_point": index})
        tools.dev_log(
            f"repair: audio_transition "
            f"change_point={index} a_ms={point['a_ms']} b_ms={point['b_ms']} "
            f"step_ms={round(jump, 3)} at_s={at} fill_ms={round(fill * 1000.0, 3)} "
            f"interval_s=[{lo}, {hi}] audio_edges_s=[{edges['edge_A']}, {edges['edge_B']}] "
            f"video_status={status} video_s={None if video_s is None else round(video_s, 4)} "
            f"decision={decision}{' ' + width_note.replace(' ', '_') if width_note else ''} "
            f"for {candidate_path}\n")
    if disagreements:
        for entry in disagreements:
            owner_judgment.log_pending(
                entry["zone"], entry["reason"], entry["master_start_s"], entry["master_end_s"],
                entry["candidate_start_s"], entry["candidate_end_s"], entry["audio_cut_s"],
                entry["video_cut_s"], entry["picture_shift_ms"], entry["frames_compared"])
        owner_judgment.log_summary(len(disagreements), language)
        return None, ("owner_judgment_pending",
                      f"{len(disagreements)} change point(s) measure an audio-aligned cut "
                      f"whose video fill width the audio does not predict -- logged for the "
                      f"owner, no cut delivered")
    log_transitions_summary(transitions, candidate_path)
    return transitions, None


def log_transitions_summary(transitions, candidate_path):
    """Log one unconditional line with every transition as `change_point:decision:at_s:fill_ms:step_ms`."""
    tools.log_always(
        f"repair: audio_transitions n={len(transitions)} transitions="
        + ",".join(f"{t['change_point']}:{t['decision']}:{t['at_s']}:"
                   f"{round(t['fill_s'] * 1000.0, 3)}:{round(t['b_ms'] - t['a_ms'], 3)}"
                   for t in transitions)
        + f" for {candidate_path}\n")


def log_holes_against_walk(holes, walk, candidate_path):
    """Cross-log the union's holes against the walk's change points.

    An interior hole with no change point in reach carries no audio step and creates no fill
    (`picture_only`); a change point no hole reaches is logged as unseen by the aligner.
    """
    import audio_walk
    regions = [(p["level_before"]["t_last"] * 1000.0,
                (p["level_after"]["t_first"] + audio_walk.WALK_WINDOW_S) * 1000.0, p)
               for p in walk["points"] if p["kind"] == "change_point"]
    reach = HOLE_MERGE_WINDOW_SECONDS * 1000.0
    seen = set()
    for index, hole in enumerate(holes):
        if hole["kind"] != "interior":
            continue
        hits = [id(p) for low, high, p in regions
                if low < hole["master_ms"][1] + reach and high > hole["master_ms"][0] - reach]
        seen.update(hits)
        if not hits:
            tools.log_always(
                f"repair: picture_only hole={index} master_ms=[{round(hole['master_ms'][0], 2)}, "
                f"{round(hole['master_ms'][1], 2)}] b2_step_ms="
                f"{None if hole['step_ms'] is None else round(hole['step_ms'], 3)} "
                f"rule=ADDENDUM_25_2_no_audio_jump_no_fill for {candidate_path}\n")
    for low, high, point in regions:
        if id(point) not in seen:
            step_result("change_point_unseen_by_b2", candidate=candidate_path,
                        a_ms=point["a_ms"], b_ms=point["b_ms"], jump_ms=point["jump_ms"],
                        region_ms=[round(low, 1), round(high, 1)])


# ---------------------------------------------------------------------------
# Step 5: plan application
# ---------------------------------------------------------------------------

def _decimal(value):
    """Convert a Fraction (or any value whose str() is exact) to a Decimal."""
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(str(value))


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
    timeline_ms = _decimal(domain["master_timeline_ms"])
    zones, fills = [], []
    cursor = Decimal(0)
    if head_end_s is not None:
        cursor = min(Decimal(str(head_end_s)) * 1000, timeline_ms)
        fills.append({"master_start_ms": Decimal(0), "master_end_ms": cursor,
                      "reason": WHY_TOKEN["head"], "hole": "head", "status": "audio_edge"})
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
                          "reason": WHY_TOKEN["interior"], "hole": number,
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
                      "reason": WHY_TOKEN["tail"], "hole": "tail", "status": "audio_edge"})
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


def written_edge_seconds(fills, master_audio_end_s):
    """Return (head_s, tail_s) of edge fill the master actually writes (tail capped at its audio end)."""
    end = Decimal(str(master_audio_end_s)) * 1000
    head = sum((f["master_end_ms"] - f["master_start_ms"]) for f in fills
               if f["reason"] == WHY_TOKEN["head"])
    tail = sum(max(Decimal(0), min(f["master_end_ms"], end) - f["master_start_ms"])
               for f in fills if f["reason"] == WHY_TOKEN["tail"])
    return float(head) / 1000.0, float(tail) / 1000.0


def tag_decision(n_splices, edge_added_s):
    """Decide whether the track is tagged chimeric.

    Any interior splice tags; otherwise written edge additions at or above
    EDGE_ADDITION_CHIMERIC_TAG_THRESHOLD_SECONDS tag.

    Returns:
        (required, reason)
    """
    if n_splices:
        return True, (f"{n_splices} interior splice(s) tag regardless of their size")
    if edge_added_s >= EDGE_ADDITION_CHIMERIC_TAG_THRESHOLD_SECONDS:
        return True, (f"edge additions written total {edge_added_s:.3f}s, at or above the "
                      f"{EDGE_ADDITION_CHIMERIC_TAG_THRESHOLD_SECONDS}s threshold")
    return False, (f"edge additions written total {edge_added_s:.3f}s, under the "
                   f"{EDGE_ADDITION_CHIMERIC_TAG_THRESHOLD_SECONDS}s threshold and no interior "
                   f"splice -- the original track with a marginal completion, not a chimera")


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
        step_launch("track_offset", candidate=candidate_obj.filePath, stream=order,
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
            step_result("track_offset", candidate=candidate_obj.filePath, stream=order,
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
                candidate_obj, audio, _speed_chain(audio, speed_ratio, engine), scale,
                deadline=deadline, envelope=envelope)
        except Exception as error:                                       # noqa: BLE001
            if getattr(error, "cause", None) == "repair_budget_exceeded":
                raise
            entry["own_reason"] = f"track_unreadable({type(error).__name__})"
            for reading in readings:
                reading["reason"] = entry["own_reason"]
            step_result("track_offset", candidate=candidate_obj.filePath, stream=order,
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
        step_result("track_offset", candidate=candidate_obj.filePath, stream=order,
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
                           "reason": WHY_TOKEN["head"] if start == 0 else WHY_TOKEN["interior"]})
            start = min(shifted, end)
        if extent_ms is not None and end + offset > extent_ms and start < end:
            cut = min(end - start, end + offset - extent_ms)
            adjustments.append({"zone": segment["zone"], "kind": "past_track_end",
                                "master_fill_ms": str(cut)})
            pieces_tail = {"source": "master", "master_start_ms": end - cut,
                           "master_end_ms": end, "source_start_ms": end - cut,
                           "reason": (WHY_TOKEN["tail"] if end == timeline_ms
                                      else WHY_TOKEN["interior"])}
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
            if piece["reason"] == WHY_TOKEN["tail"]:
                merged[-1]["reason"] = WHY_TOKEN["tail"]
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
    timeline_ms = _decimal(domain["master_timeline_ms"])
    rate_text = (f"{speed_factor.numerator}/{speed_factor.denominator}"
                 if isinstance(speed_factor, Fraction) else speed_factor)
    zones, fills = plan_spec["zones"], plan_spec["fills"]
    step_launch("apply_plan", candidate=candidate_path, n_zones=len(zones), n_fills=len(fills),
                speed_factor=rate_text, chimeric_tag=context["tagged"])

    # ---- 1. geometry --------------------------------------------------------
    head_added = Decimal(str(round(plan_spec["head_written_s"] * 1000.0, 3)))
    tail_added = Decimal(str(round(plan_spec["tail_written_s"] * 1000.0, 3)))
    interior_filled = sum((fill["master_end_ms"] - fill["master_start_ms"]) for fill in fills
                          if fill["reason"] == WHY_TOKEN["interior"])
    step_result("plan_geometry", candidate=candidate_path,
                zones=[[float(z["master_start_ms"]), float(z["master_end_ms"])] for z in zones],
                offsets_ms=[float(z["offset_ms"]) for z in zones],
                zone_windows=[z["n_windows"] for z in zones],
                fills=[[float(f["master_start_ms"]), float(f["master_end_ms"]), f["reason"],
                        f["status"]] for f in fills])
    tools.dev_log(f"orchestrator: edge_additions head_ms={head_added} tail_ms={tail_added} "
                  f"interior_filled_ms={interior_filled} chimeric_tag={context['tagged']} "
                  f"tag_reason={context['tag_reason'].replace(' ', '_')} "
                  f"for {candidate_path}\n")
    if not zones:
        step_result("apply_plan", candidate=candidate_path, ok=False,
                    cause="plan_reads_no_candidate_content")
        return False, "plan_reads_no_candidate_content", (
            "the resolved holes leave no candidate content on the master timeline -- a plan "
            "that reads nothing from the candidate is the master, not a repair")

    # ---- 2. speed -----------------------------------------------------------
    speed_ratio = None
    scale = Decimal(1)
    if speed_factor is not None and speed_factor != 1:
        speed_ratio = _decimal(Fraction(speed_factor))
        scale = speed_ratio
    step_result("delivery_speed", candidate=candidate_path, speed_factor=rate_text,
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
        step_result("apply_plan", candidate=candidate_path, ok=False,
                    cause="plan_offset_unmeasurable")
        return False, "plan_offset_unmeasurable", offset_failure

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
        step_result("track_pieces", candidate=candidate_path, stream=order,
                    n_pieces=len(pieces), sources=sources,
                    pieces=[(p["source"][0], float(p["master_start_ms"]),
                             float(p["master_end_ms"]),
                             float(p["source_start_ms"])) for p in pieces])

    # ---- 5. chapters ---------------------------------------------------------
    step_launch("chapters", candidate=candidate_path)
    chapters_path, chapter_decisions = merge_video_chimeric.build_delivered_chapters(
        master_obj.filePath, candidate_obj.filePath, reference_pieces,
        speed_ratio, timeline_ms, work_dir)
    for decision in chapter_decisions:
        tools.log_line("repair: chapter " + " ".join(
            f"{key}={str(value).replace(' ', '_')}" for key, value in decision.items())
            + "\n")
    step_result("chapters", candidate=candidate_path, delivered=chapters_path is not None,
                n_decisions=len(chapter_decisions))

    # ---- 6. build ------------------------------------------------------------
    seam = getattr(candidate_obj, merge_video_repair.REPAIR_SEAM_ATTRIBUTE, None)
    job_start_utc = (seam or {}).get("job_start_utc")
    if job_start_utc is None:
        # Standalone run (no repair seam): no job start to stamp.
        job_start_utc = "unstamped(no_repair_seam_standalone_run)"
        tools.dev_log(f"orchestrator: no {merge_video_repair.REPAIR_SEAM_ATTRIBUTE} on "
                      f"{candidate_path}: standalone run, the era tag carries no job start\n")
    marker = "chimeric" if context["tagged"] else ""
    for fill in fills:
        if fill.get("status") == "replacement_hole":
            marker = "+".join(part for part in (
                marker, f"replacement_hole:{fill['cut_ms']}/"
                        f"{round(float(fill['master_end_ms'] - fill['master_start_ms']), 3)}") if part)
    comparison_offsets = tracks[int(context["candidate_stream"])]["zones"]
    plan = {
        "kind": "orchestrator_chimeric",
        "language": language, "reference_stream": context["master_stream"],
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
    }
    step_launch("build", candidate=candidate_path, marker=marker,
                n_tracks=len(track_plans))
    try:
        repaired_obj, assembly = merge_video_repair.build_repaired_video_object(
            candidate_obj, master_obj, plan, path.join(tools.tmpFolder, "repair"),
            job_start_utc)
    except merge_video_chimeric.chimeric_error as error:
        step_result("build", candidate=candidate_path, ok=False,
                    cause=getattr(error, "cause", None), error=str(error)[:300],
                    seconds=round(time.time() - started, 1))
        raise
    out_path = getattr(repaired_obj, "filePath", None)
    exists = bool(out_path) and path.exists(out_path)
    step_result("build", candidate=candidate_path, ok=exists, out_path=out_path,
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
        "chimeric_tag": context["tagged"], "speed_ratio": plan["speed_ratio_exact"],
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
    step_result("apply_plan", candidate=candidate_path, ok=True, out_path=out_path,
                seconds=round(time.time() - started, 1))
    return True, None, reason


# ---------------------------------------------------------------------------
# Step 2: similarity gate and rate sweep
# ---------------------------------------------------------------------------

RATE_ARM_WAV_NAME = "rate_arm_{name}_{engine}.wav"


def _drop_rate_wav(primed):
    """Delete the prime's kept comparison WAV on paths that do not run the rate arm."""
    rate_wav = primed.pop("rate_wav", None)
    if rate_wav is not None:
        try:
            remove(rate_wav["path"])
        except OSError:
            pass


def _finalist_row(ratio, engine, alignment, shared_ms, seconds, candidate_path):
    """Build and log one rate finalist's reading (`rate_direction.finalist_reading`).

    Its residual-rate test is the one-quantum ladder or the fast-drift signature.
    """
    import rate_direction
    _zones, detail = coalesce_same_offset_zones(alignment.get("zones") or [],
                                                alignment.get("zones_detail") or [])
    quantum_ms = alignment.get("quantum_ms")
    ladder = (zone_ladder_signature(alignment)["is_rate_ladder"]
              or rate_direction.fast_drift_signature(alignment.get("zones_detail") or [],
                                                     quantum_ms)["fires"])
    row = {"ratio": ratio, "engine": engine,
           **rate_direction.finalist_reading(detail, quantum_ms, shared_ms, ladder),
           "fidelity": rate_direction.fidelity(alignment.get("zones_detail")),
           "point_coverage": alignment.get("master_axis_coverage_fraction"),
           "seconds": round(seconds, 1)}
    rate_direction.log_finalist(candidate_path, row)
    return row


def rate_arm(primed, first_ratios, work_dir, candidate_path, deadline=None):
    """Find which named speed ratio and engine, if any, best aligns the candidate.

    The prime's kept candidate WAV is resampled at each (ratio, engine), fingerprinted and
    aligned against the master; the prime's own alignment is the finalist at 1. Round one
    tries `first_ratios` in both engines; round two, only without a winner, every named ratio.
    The winner is `rate_direction.choose_winner`'s. The WAV is deleted on every path.

    Returns:
        (factor, engine, gate, cause): factor is an exact Fraction or None (1 won, or no
        finalist qualified), gate holds every finalist's reading and the verdict.
    """
    import merge_video_resample
    import rate_direction
    wav = primed.get("rate_wav")
    couple = primed["couples"][0]
    name = f"{couple[0]}x{couple[1]}"
    gate = {"instrument": "rate_arm", "verdict": "declined", "ratio": None, "engine": None,
            "span_coverage": None, "margin": None, "median_fidelity": None, "rows": [],
            "cause": None, "rounds": 0}
    try:
        baseline = primed["alignments"].get(name)
        fp_master, quantum_master, duration_master = primed["fingerprints"][("master", couple[0])]
        if wav is None or baseline is None:
            gate["cause"] = "rate_arm_unmeasured"
            return None, None, gate, "rate_arm_unmeasured"
        master_duration_ms = duration_master * 1000.0
        rows = [_finalist_row(Fraction(1), None, baseline,
                              min(master_duration_ms, wav["duration_s"] * 1000.0), 0.0,
                              candidate_path)]
        tried = {Fraction(1)}

        def run_round(ratios):
            for ratio in ratios:
                for engine in merge_video_resample.SPEED_ENGINES:
                    if deadline is not None and time.monotonic() > deadline:
                        return "repair_budget_exceeded"
                    started = time.time()
                    try:
                        chain, effective = merge_video_resample.build_transform_chain(
                            wav["rate"], ratio, engine)
                    except Exception as error:                           # noqa: BLE001
                        tools.dev_log(f"orchestrator: rate_arm {ratio} {engine} unbuildable "
                                      f"({type(error).__name__}: {error})\n")
                        continue
                    out = path.join(work_dir, RATE_ARM_WAV_NAME.format(
                        name=f"{ratio.numerator}_{ratio.denominator}", engine=engine))
                    command = [tools.software["ffmpeg"], "-y", "-v", "error", "-nostdin",
                               "-i", wav["path"], "-af", chain, "-ac", "1",
                               "-ar", str(int(wav["rate"])), "-acodec", "pcm_s16le", out]
                    corrected = wav["duration_s"] * float(effective)
                    try:
                        with repair_log.announced("orchestrator", "ffmpeg", wav["path"],
                                                  media_s=wav["duration_s"]) as call:
                            done = subprocess.run(command, capture_output=True,
                                                  timeout=tools.decoder_timeout_for(
                                                      wav["duration_s"]))
                            call["exit"] = done.returncode
                        if done.returncode != 0:
                            continue
                        with repair_log.announced("orchestrator", "fpcalc", out) as call:
                            points = audioCorrelation.calculate_fingerprints(
                                out, length=corrected)
                            call["exit"] = 0
                    except (subprocess.TimeoutExpired, Exception) as error:  # noqa: BLE001
                        tools.dev_log(f"orchestrator: rate_arm {ratio} {engine} unmeasured "
                                      f"({type(error).__name__})\n")
                        continue
                    finally:
                        try:
                            remove(out)
                        except OSError:
                            pass
                    if not points:
                        continue
                    alignment = align_fingerprints(fp_master, quantum_master, duration_master,
                                                   points, CHROMAPRINT_HOP_MS, corrected)
                    rows.append(_finalist_row(ratio, engine, alignment,
                                              min(master_duration_ms, corrected * 1000.0),
                                              time.time() - started, candidate_path))
                tried.add(ratio)
            return None

        rounds = [list(first_ratios),
                  [r for r in merge_video_resample.build_rate_ratio_vocabulary()]]
        winner = None
        for number, ratios in enumerate(rounds, 1):
            ratios = [r for r in ratios if r not in tried]
            if not ratios and number > 1:
                break
            gate["rounds"] = number
            stopped = run_round(ratios)
            winner = rate_direction.choose_winner(rows)
            if stopped:
                gate["cause"] = stopped
                break
            if winner is not None:
                break
        gate["rows"] = [{**row, "ratio": f"{row['ratio'].numerator}/{row['ratio'].denominator}"}
                        for row in rows]
        if winner is None:
            gate["cause"] = gate["cause"] or "similarity_unrecoverable_by_resample"
            return None, None, gate, gate["cause"]
        others = [r["span_coverage"] for r in rows if r["ratio"] != winner["ratio"]]
        gate.update({"span_coverage": winner["span_coverage"],
                     "fidelity": winner.get("fidelity"),
                     "margin": round(winner["span_coverage"] - max(others), 4) if others else None,
                     "engine": winner["engine"], "zones": winner["zones"]})
        if winner["ratio"] == 1:
            gate["cause"] = "rate_arm_unity_wins"
            return None, None, gate, "rate_arm_unity_wins"
        gate.update({"verdict": "confirmed", "ratio": winner["ratio"]})
        return winner["ratio"], winner["engine"], gate, None
    finally:
        if wav is not None:
            try:
                remove(wav["path"])
            except OSError:
                pass
        primed.pop("rate_wav", None)


def zone_offset_rate_signature(alignment):
    """Fit the aligned zones' offsets against master position, for logging only.

    Unlike the drift trace, zone offsets are absolute, so a large true offset does not blind
    the fit. The residual (not the slope) separates a rate ramp from a staircase of edits.
    Needs at least three zones (two fit any line exactly).

    Returns:
        A dict with every key present; None where nothing could be measured.
    """
    empty = {"n_zones": 0, "slope_points_per_point": None, "r_squared": None,
             "residual_rms_ms": None, "residual_max_ms": None, "span_points": None,
             "implied_total_drift_ms": None}
    detail = alignment.get("zones_detail") or []
    quantum_ms = alignment.get("quantum_ms")
    if len(detail) < 3 or not quantum_ms:
        empty["n_zones"] = len(detail)
        return empty
    xs = [(zone["master_points"][0] + zone["master_points"][1]) / 2.0 for zone in detail]
    ys = [float(zone["offset_points"]) for zone in detail]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    if ss_xx == 0:
        empty["n_zones"] = n
        return empty
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / ss_xx
    intercept = mean_y - slope * mean_x
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    ss_res = sum(residual ** 2 for residual in residuals)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    span = xs[-1] - xs[0]
    return {
        "n_zones": n,
        "slope_points_per_point": slope,
        "r_squared": (1 - ss_res / ss_tot) if ss_tot > 0 else None,
        "residual_rms_ms": ((ss_res / n) ** 0.5) * quantum_ms,
        "residual_max_ms": max(abs(residual) for residual in residuals) * quantum_ms,
        "span_points": span,
        "implied_total_drift_ms": slope * span * quantum_ms,
    }


def zone_ladder_signature(alignment):
    """Decide whether the alignment's zones form a rate ladder.

    Under a slow rate drift the aligner emits many zones separated by one-quantum steps, all in
    the same direction; edits give a few large steps in any direction. Counting rungs separates
    the two where a line fit cannot. The arm sees NTSC-scale drift on long files only: faster
    drift is merged into multi-quantum steps (PAL is caught by other gate arms).

    Conditions, all required: at least LADDER_MIN_RUNGS rungs, rung fraction >=
    LADDER_MIN_RUNG_FRACTION, one-directional fraction >= LADDER_MIN_RUNG_MONOTONE_FRACTION
    (the deciding one), and implied deviation >= RATE_LADDER_MIN_FACTOR_DEVIATION.

    Returns:
        A dict with `is_rate_ladder`, the counts it rests on and a `reason`.
    """
    detail = alignment.get("zones_detail") or []
    quantum_ms = alignment.get("quantum_ms") or 0.0
    offsets = [zone["offset_points"] for zone in detail]
    steps = [later - earlier for earlier, later in zip(offsets, offsets[1:])]
    nonzero = [step for step in steps if step != 0]
    floor = banded_seed_alignment.RESOLUTION_FLOOR_QUANTA
    rungs = [step for step in nonzero if abs(step) < floor]
    above_floor = [step for step in nonzero if abs(step) >= floor]
    span_points = (detail[-1]["master_points"][1] - detail[0]["master_points"][0]) if detail else 0
    span_minutes = span_points * quantum_ms / 60000.0 if quantum_ms else 0.0
    rung_fraction = (len(rungs) / len(nonzero)) if nonzero else None
    monotone_fraction = (max(sum(1 for step in rungs if step > 0),
                             sum(1 for step in rungs if step < 0)) / len(rungs)) if rungs else None
    # Implied ratio from total rise over run (offset = j - i, dj/di = 1/ratio), not a regression.
    total_rise = (offsets[-1] - offsets[0]) if len(offsets) >= 2 else 0
    drift_per_point = (total_rise / span_points) if span_points else 0.0
    implied_ratio = 1.0 / (1.0 + drift_per_point) if (1.0 + drift_per_point) != 0 else None
    implied_deviation = None if implied_ratio is None else abs(implied_ratio - 1.0)
    signature = {
        "n_zones": len(detail),
        "n_steps": len(steps),
        "n_steps_nonzero": len(nonzero),
        "n_rungs_subfloor": len(rungs),
        "n_steps_above_floor": len(above_floor),
        "rung_fraction": None if rung_fraction is None else round(rung_fraction, 4),
        "rung_monotone_fraction": (None if monotone_fraction is None
                                   else round(monotone_fraction, 4)),
        "zones_per_minute": round(len(detail) / span_minutes, 3) if span_minutes > 0 else None,
        "implied_speed_ratio": None if implied_ratio is None else round(implied_ratio, 7),
        "implied_factor_deviation": (None if implied_deviation is None
                                     else round(implied_deviation, 7)),
        "min_factor_deviation": RATE_LADDER_MIN_FACTOR_DEVIATION,
        "is_rate_ladder": False,
        "reason": None,
    }
    if len(rungs) < LADDER_MIN_RUNGS:
        signature["reason"] = (f"{len(rungs)} one-quantum rungs, under the {LADDER_MIN_RUNGS} "
                               f"this instrument needs before it will call a ladder a ladder")
        return signature
    if rung_fraction < LADDER_MIN_RUNG_FRACTION:
        signature["reason"] = (f"one-quantum rungs are {rung_fraction:.3f} of the "
                               f"{len(nonzero)} offset changes, under {LADDER_MIN_RUNG_FRACTION} "
                               f"-- {len(above_floor)} steps clear the resolution floor, so this "
                               f"is a file with edits in it, not a file that is drifting")
        return signature
    if monotone_fraction < LADDER_MIN_RUNG_MONOTONE_FRACTION:
        signature["reason"] = (f"the {len(rungs)} rungs are only {monotone_fraction:.3f} "
                               f"one-directional, under {LADDER_MIN_RUNG_MONOTONE_FRACTION} -- "
                               f"drift has a sign and this does not")
        return signature
    if implied_deviation is None or implied_deviation < RATE_LADDER_MIN_FACTOR_DEVIATION:
        signature["reason"] = (f"the ladder implies a speed ratio of {implied_ratio}, a deviation "
                               f"of {implied_deviation} from unity -- under "
                               f"{RATE_LADDER_MIN_FACTOR_DEVIATION}, half the smallest deviation "
                               f"any NAMED rate has, so no named rate could explain it and the "
                               f"sweep would have nothing to confirm")
        return signature
    signature["is_rate_ladder"] = True
    signature["reason"] = (f"{len(rungs)} one-quantum rungs ({rung_fraction:.3f} of all offset "
                           f"changes, {monotone_fraction:.3f} of them one-directional) over "
                           f"{span_minutes:.1f} minutes, against {len(above_floor)} steps above "
                           f"the resolution floor, implying a speed ratio of {implied_ratio}")
    return signature


def similarity_gate(alignment):
    """Decide from one couple's alignment whether to run the rate arm.

    Arms (named in `observations["gate_arm"]`):
        alignment_could_not_measure: the aligner returned a could-not-measure verdict; terminal
            if the rate arm then finds nothing.
        master_axis_coverage_below_floor: coverage under MASTER_AXIS_COVERAGE_FLOOR; same.
        rate_relation_signature: the zones form a rate ladder; non-terminal, so a false
            positive only costs one sweep.

    Returns:
        (should_sweep, reason, observations)
    """
    drift_fit = alignment.get("drift_fit") or {}
    zone_fit = zone_offset_rate_signature(alignment)
    observations = {
        "verdict": alignment.get("verdict"),
        "coverage": alignment.get("master_axis_coverage_fraction"),
        "residual_fraction": alignment.get("residual_fraction"),
        "slope_points_per_point": drift_fit.get("slope_points_per_point"),
        "r_squared": drift_fit.get("r_squared"),
        "implied_step_count": drift_fit.get("implied_step_count"),
        "trace_fit_degenerate": drift_fit.get("fit_degenerate"),
        "trace_residual_rms_ms": (None if drift_fit.get("residual_rms_ms") is None
                                  else round(drift_fit["residual_rms_ms"], 4)),
        "zone_n": zone_fit["n_zones"],
        "zone_slope_points_per_point": zone_fit["slope_points_per_point"],
        "zone_r_squared": zone_fit["r_squared"],
        "zone_residual_rms_ms": (None if zone_fit["residual_rms_ms"] is None
                                 else round(zone_fit["residual_rms_ms"], 3)),
        "zone_residual_max_ms": (None if zone_fit["residual_max_ms"] is None
                                 else round(zone_fit["residual_max_ms"], 3)),
        "zone_implied_total_drift_ms": (None if zone_fit["implied_total_drift_ms"] is None
                                        else round(zone_fit["implied_total_drift_ms"], 1)),
        "rate_arm_calibrated": RATE_RELATION_SLOPE_GATE_CALIBRATED,
    }
    if alignment.get("verdict") in ALIGNMENT_COULD_NOT_MEASURE_VERDICTS:
        observations["gate_arm"] = "alignment_could_not_measure"
        return True, f"the aligner returned {alignment['verdict']}", observations

    # Unreadable coverage (None) counts as low.
    coverage = alignment.get("master_axis_coverage_fraction")
    if coverage is None or coverage < MASTER_AXIS_COVERAGE_FLOOR:
        observations["gate_arm"] = "master_axis_coverage_below_floor"
        return True, (f"the aligner returned {alignment.get('verdict')} but its trusted zones "
                      f"cover {coverage} of the master axis, under the "
                      f"{MASTER_AXIS_COVERAGE_FLOOR} floor -- a verdict token is not a "
                      f"measurement of how much lined up"), observations

    ladder = zone_ladder_signature(alignment)
    observations.update({f"ladder_{key}": value for key, value in ladder.items()})
    if RATE_RELATION_SLOPE_GATE_CALIBRATED and ladder["is_rate_ladder"]:
        observations["gate_arm"] = "rate_relation_signature"
        return True, (f"the aligner aligned ({alignment.get('verdict')}) but its zones form a "
                      f"rate ladder: {ladder['reason']}"), observations
    observations["gate_arm"] = None
    return False, (f"the aligner anchored runs ({alignment.get('verdict')}) and its zones are "
                   f"not a rate ladder ({ladder['reason']}), so similarity is not low in the "
                   f"rate-ladder sense"), observations


def ensemble_similarity_gate(primed, candidate_path):
    """Apply `similarity_gate` to every couple and decide for the pair.

    No healthy couple: sweep on the first couple's terminal arm. A healthy couple with a rate
    ladder: sweep on the non-terminal arm. Otherwise no sweep; low couples are screened later.

    Returns:
        (should_sweep, prose, observations), like `similarity_gate`, plus per-couple arms.
    """
    readings = []
    for couple in primed["couples"]:
        name = f"{couple[0]}x{couple[1]}"
        should, prose, observations = similarity_gate(primed["alignments"][name])
        step_result("similarity_gate_couple", candidate=candidate_path, couple=name,
                    should_sweep=should, gate_arm=observations["gate_arm"],
                    coverage=observations["coverage"], verdict=observations["verdict"])
        readings.append((name, should, prose, observations))
    arms = {name: observations["gate_arm"] for name, _s, _p, observations in readings}
    healthy = [reading for reading in readings
               if reading[3]["gate_arm"] in (None, "rate_relation_signature")]
    if not healthy:
        name, should, prose, observations = readings[0]
    else:
        ladder = [reading for reading in healthy
                  if reading[3]["gate_arm"] == "rate_relation_signature"]
        name, should, prose, observations = (ladder or healthy)[0]
    observations = dict(observations, deciding_couple=name, couple_arms=arms,
                        n_healthy_couples=len(healthy))
    return should, prose, observations


# ---------------------------------------------------------------------------
# Step 3: chimeric plan
# ---------------------------------------------------------------------------

@repair_log.timed_phase("orchestrator", "chimeric",
                        lambda factor, language, master_obj, candidate_obj, *a, **k:
                        candidate_obj.filePath)
def chimeric(factor, language, master_obj, candidate_obj, work_dir, primed,
             sweep_gate=None, resample_routing=None, repair_deadline=None):
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

    step_launch("cross_verify", candidate=candidate_path, n_couples=len(couple_results))
    report = cross_verify_couples(couple_results)
    log_cross_verification(candidate_path, report)
    if not report["agree"]:
        return False, "intercouple_disagreement", (
            f"{len(report['disagreements'])} of {len(report['clusters'])} event clusters "
            f"disagree across {report['n_couples']} couples of {language}; every couple's "
            f"position, step, quantum, residual and coverage is in the report"), report

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
    domain["repair_deadline"] = repair_deadline
    insane = hole_sanity(holes, domain, candidate_path)
    if insane is not None:
        return False, insane[0], insane[1], None

    # ---- audio bounds, video pins --------------------------------------------
    speed_ratio = None if factor in (None, 1) else _decimal(Fraction(factor))
    try:
        walk, walk_reason = reference_walk(reference, holes, master_obj, candidate_obj,
                                           language, speed_ratio, candidate_path,
                                           deadline=repair_deadline,
                                           engine=(resample_routing or {}).get("filter_name",
                                                                               "asetrate"))
    except Exception as error:                                           # noqa: BLE001
        if getattr(error, "cause", None) != "repair_budget_exceeded":
            raise
        log_partial_plan(candidate_path, "repair_budget_exceeded",
                         [("holes", "b2", [(h["kind"], h["master_ms"]) for h in holes]),
                          ("audio_walk", "stopped_by_budget", str(error)[:200])])
        return False, "repair_budget_exceeded", (
            f"the repair's budget ran out during the audio walk ({error}) -- the partial plan "
            f"is logged; declined, retried at the next run"), None
    if walk is None:
        return False, "audio_walk_unavailable", (
            f"the millisecond walk on the reference couple {reference['couple']} could not "
            f"measure the pair ({walk_reason}) -- no offset, step or fill can be placed "
            f"without it"), None
    if repair_deadline is not None and time.monotonic() > repair_deadline:
        log_partial_plan(candidate_path, "repair_budget_exceeded",
                         [("audio_walk", "levels", [lv["off_ms"] for lv in walk["levels"]])])
        return False, "repair_budget_exceeded", (
            f"the repair's budget ran out after the audio walk -- the "
            f"partial plan is logged; declined, retried at the next run"), None
    log_holes_against_walk(holes, walk, candidate_path)
    log_absorbed_gaps(couple_results, holes, walk, candidate_path)
    transitions, refusal = audio_transitions(walk, reference, domain, master_obj, candidate_obj,
                                             work_dir, candidate_path, language, union=holes)
    if transitions is None:
        return False, refusal[0], refusal[1], None
    head_end_s, tail_start_s, refusal = audio_edges(walk, holes, domain, master_obj,
                                                    candidate_obj, work_dir, candidate_path)
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
    if repair_deadline is not None and time.monotonic() > repair_deadline:
        log_partial_plan(candidate_path, "repair_budget_exceeded",
                         [(f"change_point_{t['change_point']}", t["decision"], t["at_s"],
                           t["fill_s"]) for t in transitions]
                         + [("head", "placed", head_end_s), ("tail", "placed", tail_start_s)])
        return False, "repair_budget_exceeded", (
            f"the repair's budget ran out before the plan's application -- "
            f"the partial plan is logged; declined, retried at the next run"), None
    # A picture shift inside an otherwise audio-continuous zone is never acted on by widening
    # `zones`: it declines the whole repair instead (owner_judgment_pending), since nothing
    # here measures which of the audio or the video is right.
    import picture_only_shift
    picture_shifts = picture_only_shift.scan_zones(
        zones, domain, master_obj, candidate_obj, candidate_path, work_dir, repair_deadline,
        language)
    if picture_shifts:
        return False, "owner_judgment_pending", (
            f"{len(picture_shifts)} zone(s) measured as audio-continuous show the picture "
            f"itself at a different frame offset for a sustained run -- logged for the owner, "
            f"no cut delivered"), None
    head_written_s, tail_written_s = written_edge_seconds(fills, walk["master_audio_end_s"])
    tagged, tag_reason = tag_decision(len(transitions), head_written_s + tail_written_s)
    step_result("plan_shape_resolved", candidate=candidate_path,
                n_transitions=len(transitions),
                decisions=[t["decision"] for t in transitions],
                head_end_s=head_end_s, tail_start_s=tail_start_s,
                head_written_s=round(head_written_s, 3), tail_written_s=round(tail_written_s, 3),
                chimeric_tag=tagged, chimeric_tag_reason=tag_reason.replace(" ", "_"))
    master_stream, candidate_stream = reference["couple"].split("x")
    ok, cause, reason = apply_plan(candidate_path, {
        "zones": zones, "fills": fills, "walk": walk, "head_written_s": head_written_s,
        "tail_written_s": tail_written_s}, factor, master_obj, candidate_obj, {
        "language": language, "work_dir": work_dir, "domain": domain,
        "quantum_ms": reference["alignment"]["quantum_ms"],
        "master_stream": master_stream, "candidate_stream": candidate_stream,
        "resample_routing": resample_routing,
        "sweep_gate": sweep_gate, "tagged": tagged, "tag_reason": tag_reason})
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
    return _terminal(candidate_path,
                     "no_plan" if cause == "repair_budget_exceeded" else "declined",
                     cause, reason)


# ---------------------------------------------------------------------------
# Corrupt comparison track
# ---------------------------------------------------------------------------
COMPARISON_TRACK_CORRUPT = "comparison_track_corrupt"


def _video_unreliable(video_obj):
    """Return (unreliable, source) for a file's picture.

    Uses the object's `video_unreliable` tag if set, else `integrity.video_is_sound`;
    (None, "decoder_timeout") when the probe timed out.
    """
    tag = getattr(video_obj, "video_unreliable", None)
    if tag is not None:
        return bool(tag), "tag"
    import integrity
    try:
        return (not integrity.video_is_sound(video_obj)), "video_is_sound"
    except tools.decoder_timeout:
        return None, "decoder_timeout"


def comparison_track_corrupt_route(master_obj, candidate_obj, language, primed, prime_reason,
                                   candidate_path, repair_deadline):
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
        _plan_line("none", candidate_path, step="comparison_track_corrupt", cause="decoder_timeout")
        return _terminal(candidate_path, "no_plan", "decoder_timeout", (
            f"{prime_reason}; the video probe that decides the route ran past its bound "
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
        COMPARISON_TRACK_CORRUPT, evidence, master_obj, candidate_obj, language, [],
        repair_deadline)
    # No audio fallback: the comparison track itself is corrupt.
    return _video_route_terminal("declined" if status == "fallback" else status, cause, reason,
                                 candidate_path, COMPARISON_TRACK_CORRUPT)


# ---------------------------------------------------------------------------
# Language tag vs content (`language_content_check`)
# ---------------------------------------------------------------------------
def _language_content_route(master_obj, candidate_obj, language, primed, work_dir, work_root,
                            repair_deadline, master_intertrack_cache, repair_budget_s):
    """Check by content whether a candidate track carries the comparison language.

    Returns:
        None when the check has nothing to say; else the repair's result: a
        `no_common_language_after_tag_check` decline, a budget decline, or, on
        `audio_tag_conflict`, `repair()` re-run once on the re-tagged candidate within the same
        deadline.
    """
    import language_content_check as lcc
    candidate_path = candidate_obj.filePath
    if getattr(candidate_obj, "tag_checked", False):
        return None
    step_launch("content_check", candidate=candidate_path, language=language)
    check = lcc.content_check(master_obj, candidate_obj, language, primed, work_dir,
                              deadline=repair_deadline)
    best = check.get("best") or {}
    step_result("content_check", candidate=candidate_path, verdict=check["verdict"],
                best_similarity=best.get("similarity"), best_candidate=best.get("candidate"),
                best_candidate_tag=best.get("candidate_tag"), cost_s=check.get("cost_s"))
    if check["verdict"] == "repair_budget_exceeded":
        return _budget_terminal(candidate_path, "content_check", repair_budget_s)
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
                  _carried_deadline=repair_deadline,
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
           master_intertrack_cache=None, _carried_deadline=None, _carried_prime=None):
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
                       master_intertrack_cache, _carried_deadline, _carried_prime)
    finally:
        merge_video_decode_once.end()


def _repair(master_obj, candidate_obj, comparison_language, work_root=None,
            master_intertrack_cache=None, _carried_deadline=None, _carried_prime=None):
    """Run the repair steps in order; each step returns a result and none calls another.

    0. Master conformity check; failure declines.
    1. Master self-consistency; a desynchronised master routes to the video.
    2. Prime every couple; similarity gate and rate arm.
    3. Chimeric plan at the chosen factor, then plan application.
    """
    candidate_path = candidate_obj.filePath
    repair_budget_s, budget_video_s = repair_budget_seconds(master_obj)
    repair_deadline = time.monotonic() + repair_budget_s
    if _carried_deadline is not None:
        # A re-run after a language re-tag stays inside the first run's budget.
        repair_deadline = _carried_deadline
    tools.log_always(f"orchestrator: repair_budget budget_s={repair_budget_s} "
                     f"per_slice_s={REPAIR_BUDGET_PER_SLICE_S} slice_s={REPAIR_BUDGET_SLICE_S} "
                     f"cap_s={REPAIR_BUDGET_CAP_S} master_video_s={budget_video_s} "
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
            [], repair_deadline)
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
                                         primed, work_dir, work_root, repair_deadline,
                                         master_intertrack_cache, repair_budget_s)
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
                                              primed, prime_reason, candidate_path,
                                              repair_deadline)
    if not prime_ok:
        _drop_rate_wav(primed)
        _plan_line("none", candidate_path, step="prime", cause=prime_cause)
        return _terminal(candidate_path, "no_plan", prime_cause, prime_reason)
    if time.monotonic() > repair_deadline:
        _drop_rate_wav(primed)
        return _budget_terminal(candidate_path, "prime", repair_budget_s)

    # ---- step 1b: audio self-contradiction -> video route --------------------
    trigger, evidence, delay_rows = video_offset_plan.detect_audio_contradiction(
        master_obj, candidate_obj, comparison_language, primed, candidate_path)
    if trigger is not None:
        status, cause, reason = video_offset_plan.video_anchored_route(
            trigger, evidence, master_obj, candidate_obj, comparison_language, delay_rows,
            repair_deadline)
        if status != "fallback":
            _drop_rate_wav(primed)
            return _video_route_terminal(status, cause, reason, candidate_path, trigger)
        tools.log_always(f"repair: video_route_fallback trigger={trigger} video={cause} -- the "
                         f"offset is not one constant from head to tail (a drift or an interior "
                         f"edit): the ordinary audio path continues "
                         f"for {candidate_path}\n")
        if time.monotonic() > repair_deadline:
            _drop_rate_wav(primed)
            return _budget_terminal(candidate_path, "video_anchored", repair_budget_s)

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
            primed, first_ratios, work_dir, candidate_path, deadline=repair_deadline)
        step_result("rate_arm", candidate=candidate_path, armed_by=armed_by,
                    factor=(None if winner is None
                            else f"{winner.numerator}/{winner.denominator}"),
                    engine=engine, cause=sweep_cause, rounds=sweep_gate.get("rounds"),
                    span_coverage=sweep_gate.get("span_coverage"),
                    margin=sweep_gate.get("margin"),
                    finalists=[(row["ratio"], row["engine"], row["span_coverage"], row["zones"],
                                row["ladder"]) for row in sweep_gate["rows"]])
        if sweep_cause == "repair_budget_exceeded":
            log_partial_plan(candidate_path, "repair_budget_exceeded",
                             [("rate_arm", "stopped_by_budget", len(sweep_gate["rows"]))])
            return _budget_terminal(candidate_path, "rate_arm", repair_budget_s)
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
                                             primed, work_dir, work_root, repair_deadline,
                                             master_intertrack_cache, repair_budget_s)
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
    if time.monotonic() > repair_deadline:
        return _budget_terminal(candidate_path, "rate_decision", repair_budget_s)

    # ---- step 3: chimeric ----------------------------------------------------
    step_launch("chimeric", candidate=candidate_path, language=comparison_language,
                speed_factor=(f"{factor.numerator}/{factor.denominator}"
                               if isinstance(factor, Fraction) else factor))
    ok, cause, reason, detail = chimeric(factor, comparison_language, master_obj,
                                         candidate_obj, work_dir, primed,
                                         sweep_gate=sweep_gate,
                                         resample_routing=resample_routing,
                                         repair_deadline=repair_deadline)
    step_result("chimeric", candidate=candidate_path, ok=ok, cause=cause)
    if ok:
        # `apply_plan` already recorded the `repaired` terminal.
        _plan_line("chimeric", candidate_path, step="chimeric",
                   speed_factor=(f"{factor.numerator}/{factor.denominator}"
                                  if isinstance(factor, Fraction) else factor))
        return True
    _plan_line("none", candidate_path, step="chimeric", cause=cause)
    return _terminal(candidate_path, "declined" if cause == "master_cut_short" else "no_plan",
                     cause, reason,
                     detail={"cross_verification": detail} if detail else None)


def align_fingerprints(fp_master, quantum_master, duration_master, fp_candidate,
                       quantum_candidate, duration_candidate):
    """Align two fingerprint lists with `b2_align`, as both the prime and the rate arm do."""
    return banded_seed_alignment.b2_align(
        fp_master, fp_candidate, quantum_master,
        candidate_quantum_ms=quantum_candidate,
        duration_diff_ms=abs(duration_master - duration_candidate) * 1000.0,
        signed_duration_diff_ms=(duration_candidate - duration_master) * 1000.0,
        shorter_duration_ms=min(duration_master, duration_candidate) * 1000.0,
        deadline=time.monotonic() + ALIGNMENT_BUDGET_S)


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
            except tools.decoder_timeout as error:
                return (False, "decoder_timeout",
                        f"the {side} {language} stream {stream} extraction ran past its bound "
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
        if alignment["verdict"] == banded_seed_alignment.VERDICT_ALIGNMENT_BUDGET_EXCEEDED:
            step_result("align", candidate=candidate_path, couple=name,
                        verdict=alignment["verdict"],
                        seeds_extended=alignment.get("seeds_extended"),
                        seeds_total=alignment.get("seeds_total"),
                        seconds=round(alignment["alignment_seconds"], 2))
            return (False, "alignment_budget_exceeded",
                    f"couple {name} did not align within {ALIGNMENT_BUDGET_S} s: "
                    f"{alignment.get('seeds_extended')} of {alignment.get('seeds_total')} seeds "
                    f"extended ({len(fp_master)} x {len(fp_candidate)} points) -- a statement "
                    f"about this run's cost; declined, retried at the next run")
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
