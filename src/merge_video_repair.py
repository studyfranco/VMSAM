"""Repair of a file rejected by mergeVideo, at the moment it is rejected.

Entry point called from `mergeVideo.remove_not_compatible_video`: declines candidates without a
video object, runs `repair_orchestrator.repair()` on each one and attaches the repaired object to
the merge. Also holds helpers shared with the orchestrator (`record`,
`master_intertrack_verdict`, `build_repaired_video_object`, `gate_fabricated_delivery`).

The repaired video object must:

1. have `delay_same_md5_audio = Decimal('0')` (mergeVideo adds a delay to it, and the repair has
   already placed the file on the master's timeline);
2. have run `get_mediadata()` (`generate_new_file` reads `audios` and `video['Duration']`);
3. keep its temporary file, under `tools.tmpFolder/repair/`, until `mkvmerge` has run.

Outcomes: `no_plan` (nothing measured, the rejection stands), `declined` (repair refused, with a
reason), `repaired` (file attached), `failed` (the repair raised).
"""

from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from os import path
import hashlib
import json
import sys
import time

import tools
import repair_log
import video

# Verifier alignment tolerance (ms): clean products land at <= 10.5 ms, plan errors at >= 26 ms.
verify_tolerance_ms = 15

last_repair_report = []


def get_speed_margin(plan):
    """Return the winning speed hypothesis' margin as a string, or None when none was reported.

    None, not zero: a zero margin means two hypotheses tied.
    """
    margin = plan.get("speed_margin")
    return None if margin == None else str(margin)


def assemble_or_log_the_decline(logged_candidate, plan, unverified_ms, *args, **kwargs):
    """Run `assemble_on_master_timeline`; on failure, log what was done, then re-raise.

    The per-track `repair:` lines are written from the error's `partial_assembly`, and a
    DECLINED (chimeric_error) or FAILED line is always logged.
    """
    import merge_video_chimeric
    tools.dev_log(f"repair: assemble_or_log_the_decline starting "
                  f"candidate={logged_candidate.filePath}\n")
    try:
        return merge_video_chimeric.assemble_on_master_timeline(*args, **kwargs)
    except Exception as error:
        # Any exception, not only chimeric_error: tool failures are logged like declines.
        partial = getattr(error, "partial_assembly", None)
        if partial != None:
            partial["unverified_segment_ms"] = unverified_ms
            try:
                log_assembly(logged_candidate.filePath, partial, plan)
            except Exception as logging_error:
                tools.logs.append("repair: could not write the per-track log for "
                                  f"an UNDELIVERED file: {logging_error}\n")
        # Always logged: chimeric_error -> DECLINED, anything else -> FAILED.
        if isinstance(error, merge_video_chimeric.chimeric_error):
            tools.log_always(f"repair: DECLINED {error}\n")
        else:
            tools.log_always(f"repair: FAILED {type(error).__name__}: {error}\n")
        # Where the refused product lies, only when one was marked undelivered.
        marked = getattr(error, "undelivered_path", None)
        if marked != None:
            tools.logs.append(
                f"repair: undelivered state={getattr(error, 'undelivered_state', 'unnamed')} "
                f"path={marked} "
                f"in_place={getattr(error, 'undelivered_in_place', 'unreported')}"
                .rstrip() + "\n")
        raise


SPEED_EVIDENCE_INSTRUMENTS = frozenset({"rate_arm", "visual_frame_match"})
# Rate sources accepted as speed evidence: `repair_orchestrator.rate_arm` (audio chromaprint) and
# `video_offset_plan.detect_speed_ratio` (matched scene-cut frame indices, `RATE_SOURCE_VISUAL`);
# any other value is refused.

# Relative applied-vs-evidenced tolerance: half the ~1e-3 gap between the closest named rates.
SPEED_EVIDENCE_RELATIVE_TOLERANCE = Decimal("0.0005")


def speed_plan_evidence(plan, speed_ratio):
    '''Check that the plan carries the evidence making `speed_ratio` admissible.

    Requires: `speed_ratio_exact` is a named rate ratio and equals `speed_ratio` within
    SPEED_EVIDENCE_RELATIVE_TOLERANCE; `rate_source` is in SPEED_EVIDENCE_INSTRUMENTS; the
    `resample_gate` is confirmed with a numeric `span_coverage` at or above the rate arm's floor
    and the same engine as the plan. Returns `(admissible, token, prose)`.
    '''
    import merge_video_resample

    vocabulary = merge_video_resample.build_rate_ratio_vocabulary()
    exact = plan.get("speed_ratio_exact")
    if exact is None:
        return False, "speed_evidence_absent", (
            "the plan carries no speed_ratio_exact: no winning exact rational "
            "travels with this coefficient, so nothing says WHICH exact ratio "
            "was recognised or by what")
    try:
        named = Fraction(str(exact))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False, "speed_evidence_rational_unreadable", (
            f"speed_ratio_exact={exact!r} does not parse as an exact rational")
    if named not in vocabulary:
        return False, "speed_evidence_rational_not_named", (
            f"speed_ratio_exact={named} is not a member of the rate sweep's "
            f"vocabulary {[str(f) for f in vocabulary]}: an exact-looking "
            f"fraction is not a recognised broadcast rate combination")

    # Relative comparison: `speed_ratio` arrives through str(), so exact equality would fail.
    nominal = Decimal(named.numerator) / Decimal(named.denominator)
    drift = abs(Decimal(str(speed_ratio)) - nominal) / nominal
    if drift > SPEED_EVIDENCE_RELATIVE_TOLERANCE:
        return False, "speed_evidence_ratio_is_not_the_snapped_rational", (
            f"the plan would apply speed_ratio={speed_ratio} while its "
            f"evidence is for {named} ({nominal}): relative drift {drift} "
            f"exceeds {SPEED_EVIDENCE_RELATIVE_TOLERANCE}. "
            f"Evidence about one coefficient does not license another")

    instrument = plan.get("rate_source")
    if instrument not in SPEED_EVIDENCE_INSTRUMENTS:
        return False, "speed_evidence_instrument_unrecognised", (
            f"rate_source={instrument!r} is not one of "
            f"{sorted(SPEED_EVIDENCE_INSTRUMENTS)}: the deciding instrument "
            f"must be named, and an unenumerated one is refused rather than "
            f"trusted")

    gate = plan.get("resample_gate")
    if not isinstance(gate, dict):
        return False, "speed_evidence_no_gate", (
            "the plan names an instrument but carries no resample_gate: the rate arm's own "
            "result is missing, so the coefficient was never validated")
    if gate.get("verdict") != "confirmed":
        return False, "speed_evidence_gate_not_confirmed", (
            f"resample_gate verdict={gate.get('verdict')!r} cause={gate.get('cause')!r}: the "
            f"rate arm did not confirm")
    import rate_direction
    span = gate.get("span_coverage")
    if not isinstance(span, (int, float)) or isinstance(span, bool):
        return False, "speed_evidence_span_absent", (
            f"resample_gate says confirmed but span_coverage={span!r} is not a measured number")
    if span < rate_direction.RATE_ARM_MIN_SPAN_COVERAGE:
        return False, "speed_evidence_span_below_floor", (
            f"the winner's span coverage {span} is below the arm's floor "
            f"{rate_direction.RATE_ARM_MIN_SPAN_COVERAGE}")
    engine = plan.get("speed_engine")
    if engine not in merge_video_resample.SPEED_ENGINES or engine != gate.get("engine"):
        return False, "speed_evidence_engine_mismatch", (
            f"the plan would apply engine {engine!r} while the rate arm's winner aligned with "
            f"{gate.get('engine')!r}")

    return True, "speed_evidence_complete", (
        f"snapped named rational {named} (applied as {speed_ratio}, engine {engine}), winner "
        f"span coverage {span} >= {rate_direction.RATE_ARM_MIN_SPAN_COVERAGE}, deciding "
        f"instrument {instrument}")


def build_repaired_video_object(candidate_obj, master_obj, plan, work_root, job_start_utc):
    '''Build the repaired file and its video object from the orchestrator's plan.

    `plan` carries the already-built pieces (`track_plans`, `reference_pieces`, `marker`, and an
    optional speed ratio with its evidence); nothing is measured here. `job_start_utc` is passed
    to the mux tags. Returns (video object, assembly report).
    '''
    import merge_video_chimeric

    key = merge_video_chimeric.stable_case_key(candidate_obj.filePath)
    work_dir = path.join(work_root, key)
    tools.make_dirs(work_dir)
    out_path = path.join(work_root, f"{key}_repaired.mkv")

    # Logged before any work that could hang.
    tools.dev_log(f"repair: build_repaired_video_object starting "
                  f"candidate={candidate_obj.filePath} work_dir={work_dir} "
                  f"out_path={out_path}\n")

    speed_ratio = plan.get("speed_ratio")
    if speed_ratio is not None:
        # A speed transform is applied only with the evidence that validated it.
        admissible, evidence_token, evidence_prose = speed_plan_evidence(
            plan, speed_ratio)
        tools.dev_log(
            f"repair: speed evidence gate for {candidate_obj.filePath}: "
            f"speed_ratio={speed_ratio} admissible={admissible} "
            f"token={evidence_token} detail={evidence_prose}\n")
        if not admissible:
            raise merge_video_chimeric.chimeric_error(
                f"speed transform not validated for production application: "
                f"speed_ratio={speed_ratio} reached build_repaired_video_object "
                f"without the validation evidence that makes it admissible "
                f"({evidence_token}: {evidence_prose}) -- refusing rather "
                f"than applying an unevidenced transform",
                cause="speed_transform_not_validated")

    # A video-anchored plan is not on the master's audio: each delivered track is verified
    # against its own original instead of the master timeline.
    video_anchored = plan.get("video_anchored")
    try:
        repaired_obj, assembly = _build_and_gate(candidate_obj, master_obj, plan, work_dir,
                                                 out_path, job_start_utc, speed_ratio,
                                                 video_anchored)
    except Exception as error:
        log_chimeric_keep(candidate_obj, plan,
                          refused=getattr(error, "cause", None) or type(error).__name__)
        raise
    log_chimeric_keep(candidate_obj, plan, assembly, repaired_obj)
    return repaired_obj, assembly


def _build_and_gate(candidate_obj, master_obj, plan, work_dir, out_path, job_start_utc,
                    speed_ratio, video_anchored):
    """Drop corrupt tracks, assemble, verify, log, then apply the delivery gates.

    Returns (repaired video object, assembly).
    """
    import merge_video_chimeric
    dropped_corrupt = drop_corrupt_candidate_tracks(candidate_obj, plan,
                                                    deadline=plan.get("repair_deadline"))
    assembly = assemble_or_log_the_decline(
        candidate_obj, plan, Decimal("0"),
        candidate_obj, master_obj, plan["track_plans"], plan["reference_pieces"],
        work_dir, out_path, plan["marker"],
        job_start_utc=job_start_utc,
        speed_ratio=speed_ratio,
        reference_stream=plan.get("reference_stream"),
        # Fill fallback when the master lacks the track's language.
        comparison_language=plan.get("language"),
        chapters_path=plan.get("chapters_path"),
        verify=video_anchored is None, verify_tolerance_ms=verify_tolerance_ms,
        deadline=plan.get("repair_deadline"),
        speed_engine=plan.get("speed_engine") or "asetrate")

    assembly["unverified_segment_ms"] = Decimal("0")
    assembly["dropped_corrupt"] = dropped_corrupt
    if video_anchored is not None:
        verification, refusal = verify_video_anchored(
            assembly["path"], candidate_obj, assembly.get("audios") or [],
            Decimal(str(video_anchored["offset_ms"])), plan["track_plans"],
            deadline=plan.get("repair_deadline"))
        assembly["verification"] = verification
        if refusal is not None:
            try:
                log_assembly(candidate_obj.filePath, assembly, plan)
            except Exception as error:
                tools.logs.append(f"repair: could not write the per-track log: {error}\n")
            error = merge_video_chimeric.chimeric_error(refusal,
                                                        cause="delivery_offset_exceeds_tolerance")
            error.verification = verification
            raise error
    # Logged before the product is re-read, so a failed re-read still leaves the log.
    try:
        log_assembly(candidate_obj.filePath, assembly, plan)
    except Exception as error:
        tools.logs.append(f"repair: could not write the per-track log: {error}\n")

    repaired_obj = video.video(path.dirname(out_path), path.basename(out_path))
    # No audio track is required: the candidate may contribute subtitles only.
    repaired_obj.need_one_audio_track = False
    repaired_obj.get_mediadata()
    # Already on the master's timeline (see module docstring).
    repaired_obj.delay_same_md5_audio = Decimal('0')
    mark_audio_dicts(repaired_obj, assembly["marker"])
    # Fabricated-track gate; its `keep=False` is read by `generate_new_file_audio_config`.
    assembly["gate_kept"] = []
    assembly["fabricated_dropped"] = gate_fabricated_delivery(
        repaired_obj, master_obj, work_dir=work_dir, deadline=plan.get("repair_deadline"),
        kept=assembly["gate_kept"])
    # Drop delivered tracks with a silence the master lacks inside the master video's bounds.
    assembly["silence_dropped"] = gate_delivered_silences(
        repaired_obj, master_obj, plan.get("reference_stream"),
        deadline=plan.get("repair_deadline"))
    return repaired_obj, assembly


DROP_CORRUPT_JOBS = 3


def drop_corrupt_candidate_tracks(candidate_obj, plan, deadline=None):
    """Strictly decode each rebuilt candidate audio track and mark failing ones `dropped_corrupt`.

    Tracks in `plan["strictly_decoded_streams"]` are skipped; a timed-out decode keeps the
    track. `deadline` is checked before each batch of DROP_CORRUPT_JOBS; past it the repair
    declines `repair_budget_exceeded`. Returns the dropped tracks.
    """
    import concurrent.futures
    import integrity
    import merge_video_chimeric
    skip = {int(s) for s in plan.get("strictly_decoded_streams") or []}
    todo = [(language, audio) for language, audio in
            merge_video_chimeric.iterate_candidate_audios(candidate_obj)
            if int(audio["StreamOrder"]) in (plan.get("track_plans") or {})
            and int(audio["StreamOrder"]) not in skip]

    def check(item):
        try:
            return item, integrity.track_check(candidate_obj, item[1]["StreamOrder"]), None
        except Exception as error:                                       # noqa: BLE001
            return item, None, error
    dropped, checked = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=DROP_CORRUPT_JOBS) as pool:
        for start in range(0, len(todo), DROP_CORRUPT_JOBS):
            batch = todo[start:start + DROP_CORRUPT_JOBS]
            if deadline is not None and time.monotonic() > deadline:
                nxt = [f"{language}:{audio['StreamOrder']}" for language, audio in batch]
                judged = [f"{d['language']}:{d['stream_order']}" for d in dropped]
                tools.log_always(
                    f"repair: partial_plan cause=repair_budget_exceeded stage=corrupt_track_gate "
                    f"checked_so_far={checked} dropped_so_far={judged} next={nxt} -- the "
                    f"repair's budget ran out while the rebuilt tracks were "
                    f"strictly decoded\n")
                raise merge_video_chimeric.chimeric_error(
                    f"the repair's budget ran out in the corrupt-track gate, before {nxt} "
                    f"(checked so far {checked}, dropped {judged}) -- declined, retried at "
                    f"the next run",
                    cause="repair_budget_exceeded")
            for (language, audio), result, error in pool.map(check, batch):
                order = audio["StreamOrder"]
                checked.append(f"{language}:{order}")
                if result is None or result["verdict"] == "decoder_timeout":
                    tools.log_always(f"repair: track_integrity unmeasured stream={order} "
                                     f"language={language} cause="
                                     f"{'decoder_timeout' if result else type(error).__name__} -- "
                                     f"no verdict, the track stays, for {candidate_obj.filePath}\n")
                    continue
                if result["verdict"] != "corrupt":
                    continue
                first = (result["error_lines"] or ["?"])[0][:200]
                audio["dropped_corrupt"] = first
                dropped.append({"stream_order": int(order), "language": language,
                                "rc": result["rc"], "first": first,
                                "cost_s": result["cost_s"]})
                tools.log_always(f"repair: track_dropped_corrupt stream={order} "
                                 f"language={language} codec={result['codec']} rc={result['rc']} "
                                 f"first=«{first}» cost_s={result['cost_s']} for "
                                 f"{candidate_obj.filePath}\n")
    return dropped


def gate_delivered_silences(repaired_obj, master_obj, reference_stream, deadline=None,
                            report=None):
    """Drop delivered audio tracks that are silent where the master's comparison track is not.

    Each kept track is compared with `reference_stream` via
    `integrity.delivered_silence_report`. A silence entirely outside the master video's bounds,
    or a trailing one of at most `integrity.TAIL_SILENCE_TOLERANCE_S`, keeps the track; any other
    sets `keep=False`. A failed measurement keeps the track. Past `deadline` the repair declines
    `repair_budget_exceeded`. `report` replaces the measurement. Returns the dropped tracks.
    """
    import integrity
    measure = report or integrity.delivered_silence_report
    tracks = [(holder, language, audio) for holder in AUDIO_HOLDERS
              for language, audios in (getattr(repaired_obj, holder, None) or {}).items()
              for audio in audios if audio.get("keep", True)]
    if reference_stream is None:
        tools.log_always(f"repair: track_silence unmeasured cause=no_reference_stream "
                         f"tracks={len(tracks)} for {repaired_obj.filePath}\n")
        return []
    bounds = integrity.video_bounds_s(master_obj)
    dropped = []
    for holder, language, audio in tracks:
        order = audio["StreamOrder"]
        where = f"stream={order} language={language} holder={holder}"
        if deadline is not None and time.monotonic() > deadline:
            import merge_video_chimeric
            judged = [f"{d['language']}:{d['stream_order']}" for d in dropped]
            tools.log_always(
                f"repair: partial_plan cause=repair_budget_exceeded stage=silence_gate "
                f"dropped_so_far={judged} next={where} -- the repair's budget ran out while "
                f"the delivered tracks' silences were compared with the master's\n")
            raise merge_video_chimeric.chimeric_error(
                f"the repair's budget ran out in the silence gate, before {where} "
                f"(dropped so far {judged}) -- declined, retried at the next run",
                cause="repair_budget_exceeded")
        try:
            r = measure(master_obj, reference_stream, repaired_obj, order, 0,
                        video_bounds_s_=bounds)
        except Exception as error:                                       # noqa: BLE001
            tools.log_always(f"repair: track_silence unmeasured {where} "
                             f"cause={type(error).__name__}: {str(error)[:200]} -- no verdict, "
                             f"the track stays, for {repaired_obj.filePath}\n")
            continue
        spans = " ".join(f"{s['start_s']}-{s['end_s']}s[{s['rule']}]"
                         for s in r["silences"][:6])
        if r["reference_only"]:
            tools.log_always(f"repair: track_silence reference_only {where} "
                             f"master_stream={reference_stream} n={len(r['reference_only'])} "
                             f"first={r['reference_only'][0]['start_s']}-"
                             f"{r['reference_only'][0]['end_s']}s -- the master's track is "
                             f"silent where this one plays; not a reason to drop, for "
                             f"{repaired_obj.filePath}\n")
        if r["verdict"] == "kept":
            tools.log_always(f"repair: track_kept_silence {where} master_stream="
                             f"{reference_stream} video_bounds_s={r['video_bounds_s']} "
                             f"silences={spans} for {repaired_obj.filePath}\n")
        elif r["verdict"] == "dropped":
            audio["keep"] = False
            dropped.append({"stream_order": int(order), "language": language,
                            "holder": holder, "silences": r["silences"]})
            tools.log_always(f"repair: track_dropped_silence {where} master_stream="
                             f"{reference_stream} video_bounds_s={r['video_bounds_s']} "
                             f"silences={spans} for {repaired_obj.filePath}\n")
    return dropped


# Where a video-anchored track is probed: fractions of its candidate piece.
VIDEO_ANCHORED_PROBE_FRACTIONS = (0.2, 0.5, 0.8)


def verify_video_anchored(out_path, candidate_obj, audio_reports, offset_ms, track_plans,
                          deadline=None):
    '''Verify each delivered video-anchored track against its own original.

    At master time `t` the product must carry what the candidate track carries at `t +
    offset_ms`; probed at VIDEO_ANCHORED_PROBE_FRACTIONS of the longest candidate piece, within
    `verify_tolerance_ms`. Returns `(results, refusal)`, refusal None when all tracks align.
    '''
    import merge_video_chimeric as mvc
    window_ms = Decimal(str(mvc.verify_window_seconds)) * Decimal("1000")
    rate = mvc.verify_probe_rate
    results = []
    for produced_index, report in enumerate(audio_reports):
        order = report["stream_order"]
        pieces = [p for p in (track_plans.get(int(order)) or {}).get("pieces") or []
                  if p["source"] == "candidate"]
        entry = {"track": order, "language": report.get("language"),
                 "produced_index": produced_index, "reference": "candidate_original",
                 "probes": []}
        results.append(entry)
        if not pieces:
            entry.update(outcome="skipped", reason="no candidate piece on this track")
            continue
        piece = max(pieces, key=lambda p: p["master_end_ms"] - p["master_start_ms"])
        span = piece["master_end_ms"] - piece["master_start_ms"] - window_ms
        for fraction in VIDEO_ANCHORED_PROBE_FRACTIONS:
            master_ms = piece["master_start_ms"] + max(Decimal(0), span) * Decimal(str(fraction))
            original = mvc.read_mono_samples(candidate_obj.filePath, f"0:{int(order)}",
                                             master_ms + offset_ms, window_ms, rate,
                                             deadline=deadline)
            produced = mvc.read_mono_samples(out_path, f"0:a:{produced_index}", master_ms,
                                             window_ms, rate, deadline=deadline)
            if min(mvc.get_rms(original), mvc.get_rms(produced)) < mvc.verify_min_rms:
                entry["probes"].append({"master_position_ms": str(master_ms),
                                        "outcome": "no_signal"})
                continue
            lag, score = mvc.measure_lag_ms(original, produced, rate, 1000)
            entry["probes"].append({"master_position_ms": str(master_ms), "lag_ms": lag,
                                    "correlation": score, "outcome": "measured"})
        measured = [p for p in entry["probes"] if p["outcome"] == "measured"]
        if not measured:
            entry.update(outcome="skipped", reason="no probe window carried signal")
            continue
        worst = max(abs(p["lag_ms"]) for p in measured)
        entry.update(outcome="aligned" if worst <= verify_tolerance_ms else "misaligned",
                     worst_lag_ms=worst,
                     weakest_correlation=round(min(p["correlation"] for p in measured), 4),
                     probes_measured=len(measured))
        tools.log_always(f"repair: video_anchored_verify track={order} "
                         f"lang={report.get('language')} outcome={entry['outcome']} "
                         f"lags_ms={[round(p['lag_ms'], 2) for p in measured]} "
                         f"correlations={[round(p['correlation'], 4) for p in measured]} "
                         f"tolerance_ms={verify_tolerance_ms} out_path={out_path}\n")
    off = [r for r in results if r.get("outcome") == "misaligned"]
    if off:
        return results, ("a video-anchored track is not its own original moved by the picture "
                         "offset: " + "; ".join(f"track {r['track']} ({r['language']}) off by "
                                               f"{r['worst_lag_ms']:.1f} ms" for r in off))
    return results, None


def mark_audio_dicts(repaired_obj, marker):
    """Set `fabricated` on every main and audio-description track of the repaired object.

    A track's own `VMSAM_FABRICATED` tag wins; `marker` is the fallback. Commentary tracks are
    left to `gate_fabricated_delivery`.
    """
    import merge_video_chimeric
    for holder in (repaired_obj.audios, repaired_obj.audiodesc):
        for language, audios in holder.items():
            for audio in audios:
                # Every track of the repaired file is rebuilt, so never left unmarked.
                audio["fabricated"] = (fabricated_marker_of(audio) or marker
                                       or merge_video_chimeric.REBUILT_MARKER)


# Audio holders of a video object; commentary included so it is always judged.
AUDIO_HOLDERS = ("audios", "commentary", "audiodesc")


def fabricated_marker_of(audio):
    """Return the fabricated marker from the in-memory key or the `VMSAM_FABRICATED` tag.

    Empty string means an intact track.
    """
    return str(audio.get("fabricated") or
               (audio.get("extra") or {}).get("VMSAM_FABRICATED") or "")


# Same thresholds as `mergeVideo.find_differences_and_keep_best_audio`, so the gate and the
# grouping agree on what "the same track" is.
SAME_CONTENT_MEAN_FIDELITY = 0.90
SAME_CONTENT_MAX_DELAY_MS = 128


def same_content_verdict(delay_fidelity_values):
    """Apply the grouping's same-content rule to one pair of tracks.

    `delay_fidelity_values`: one (fidelity, _, delay_ms) per window. Same content means mean
    fidelity >= SAME_CONTENT_MEAN_FIDELITY and one or two delays, all under
    SAME_CONTENT_MAX_DELAY_MS. Returns (bool, mean fidelity, set of delays).
    """
    from statistics import mean
    fidelity = mean([fi[0] for fi in delay_fidelity_values])
    delays = set(fi[2] for fi in delay_fidelity_values)
    if fidelity < SAME_CONTENT_MEAN_FIDELITY:
        return False, fidelity, delays
    values = list(delays)
    if len(values) == 1:
        return abs(values[0]) < SAME_CONTENT_MAX_DELAY_MS, fidelity, delays
    if len(values) == 2:
        return (abs(values[0]) < SAME_CONTENT_MAX_DELAY_MS
                and abs(values[1]) < SAME_CONTENT_MAX_DELAY_MS), fidelity, delays
    return False, fidelity, delays


def same_content_windows(duration):
    """Return the grouping's comparison windows over `duration` seconds as (start_s, length_s)."""
    begin, length_time = video.generate_begin_and_length_by_segment(duration)
    return [(float(begin) + i * length_time, float(length_time * 2))
            for i in range(video.number_cut)]


def measure_same_content(master_obj, master_audio, repaired_obj, audio, work_dir,
                         deadline=None):
    """Tell whether a fabricated track is the same version as an intact master track.

    Uses `same_content_windows` and `same_content_verdict` on slices of each track's whole
    chromaprint fingerprint. Returns (verdict, detail); verdict None means not measured, not
    "different".
    """
    import merge_video_decode_once as once
    try:
        duration = min(float(master_audio["Duration"]), float(audio["Duration"]))
        windows = same_content_windows(duration)
        need = max(start + length for start, length in windows)
        hop_s = SAME_CONTENT_HOP_S
        m_order, f_order = int(master_audio["StreamOrder"]), int(audio["StreamOrder"])
        master = once.fingerprints(master_obj.filePath, [m_order], {m_order: need}, work_dir,
                                   deadline=deadline,
                                   pads_ms={m_order: _start_ms(master_audio)})[m_order]
        fab = once.fingerprints(repaired_obj.filePath, [f_order], {f_order: need}, work_dir,
                                deadline=deadline, pads_ms={f_order: _start_ms(audio)})[f_order]
        if master is None or fab is None:
            return None, (f"unmeasured=fingerprint_unreadable "
                          f"master={master is not None} fabricated={fab is not None}")
        values = []
        for start, length in windows:
            first = int(round(start / hop_s))
            # Point count fpcalc would return on the extracted window, so delays scale identically.
            count = int(length / hop_s) - SAME_CONTENT_FPCALC_TAIL_POINTS
            source = master["points"][first:first + count]
            target = fab["points"][first:first + count]
            if min(len(source), len(target)) <= 2 * SAME_CONTENT_MIN_OVERLAP:
                return None, (f"unmeasured=window_past_the_fingerprint start_s={start} "
                              f"points={len(source)}/{len(target)}")
            values.append(once.correlate_points(source, target, length))
        verdict, fidelity, delays = same_content_verdict(values)
        return verdict, (f"mean_fidelity={fidelity:.4f} delays_ms={sorted(delays)} "
                         f"windows={len(values)} instrument=fingerprint_slices")
    except Exception as error:                                           # noqa: BLE001
        return None, f"unmeasured={type(error).__name__}: {error}"


def _start_ms(audio):
    """Return the stream's container start time in ms, used to put fingerprints on the file clock."""
    import merge_video_chimeric
    return float(merge_video_chimeric.get_stream_start_ms(audio))


# Chromaprint hop (same as `repair_orchestrator.CHROMAPRINT_HOP_MS`) and the correlation's
# minimum overlap (`audioCorrelation.min_overlap`).
SAME_CONTENT_HOP_S = (4096 // 3) / 11025.0
SAME_CONTENT_MIN_OVERLAP = 32
# fpcalc returns floor(seconds / hop) - 21 points for a window.
SAME_CONTENT_FPCALC_TAIL_POINTS = 21
# Parallel fingerprint decodes ahead of the gate's loop.
SAME_CONTENT_PREFETCH_JOBS = 2


def _prefetch_same_content(repaired_obj, master_obj, master_intact, work_dir, deadline, say):
    """Decode ahead the fingerprints `gate_fabricated_delivery` will compare, both files at once.

    A failure here is only logged; the gate then decodes the missing tracks itself.
    """
    import merge_video_decode_once as once
    from concurrent.futures import ThreadPoolExecutor
    need = {"master": {}, "product": {}}
    pads = {"master": {}, "product": {}}
    for holder in AUDIO_HOLDERS:
        if holder == "commentary":
            continue
        for language, audios in (getattr(repaired_obj, holder, None) or {}).items():
            for audio in audios:
                if not audio.get("keep", True):
                    continue
                for intact in master_intact.get(language, []):
                    try:
                        duration = min(float(intact["Duration"]), float(audio["Duration"]))
                    except (KeyError, TypeError, ValueError):
                        continue
                    end = max(a + b for a, b in same_content_windows(duration))
                    m, f = int(intact["StreamOrder"]), int(audio["StreamOrder"])
                    need["master"][m] = max(need["master"].get(m, 0.0), end)
                    need["product"][f] = max(need["product"].get(f, 0.0), end)
                    pads["master"][m] = _start_ms(intact)
                    pads["product"][f] = _start_ms(audio)
    jobs = [(master_obj.filePath, need["master"], pads["master"]),
            (repaired_obj.filePath, need["product"], pads["product"])]
    jobs = [job for job in jobs if job[1]]
    if not jobs:
        return
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=SAME_CONTENT_PREFETCH_JOBS) as pool:
            list(pool.map(lambda job: once.fingerprints(job[0], sorted(job[1]), job[1], work_dir,
                                                        deadline=deadline, pads_ms=job[2]),
                          jobs))
    except Exception as error:                                           # noqa: BLE001
        say(f"repair: gate_prefetch failed={type(error).__name__}: {str(error)[:200]}")
    say(f"repair: gate_prefetch master_streams={sorted(need['master'])} "
        f"product_streams={sorted(need['product'])} "
        f"seconds={round(time.monotonic() - started, 2)}")


def gate_fabricated_delivery(repaired_obj, master_obj, work_dir=None,
                             content_probe=None, deadline=None, kept=None):
    """Judge every fabricated audio track of the repaired file before delivery; only sets `keep`.

    Commentary tracks are kept. Other tracks are compared with each intact master track of the
    same language (`measure_same_content`): same content (or unmeasurable) -> raced through
    `mergeVideo.keep_best_audio`, where the intact track wins; different content or no intact
    track -> compared with the intact tracks of the other languages, and dropped as a duplicate
    (`cross_lang=<declared>-><matched>`) if one measures as the same content, else kept.
    `content_probe` replaces the comparison; `kept` collects kept-track decisions; past
    `deadline` the repair declines `repair_budget_exceeded`.
    Returns the dropped tracks.
    """
    import mergeVideo
    import merge_video_chimeric
    probe = content_probe or measure_same_content
    dropped = []
    master_intact = {}
    # Opponents: intact main and audio-description tracks, never a master commentary.
    for holder in ("audios", "audiodesc"):
        for language, audios in (getattr(master_obj, holder, None) or {}).items():
            for audio in audios:
                if not fabricated_marker_of(audio):
                    master_intact.setdefault(language, []).append(audio)

    def say(line, to_stderr=False):
        tools.logs.append(line + "\n")
        if to_stderr:
            sys.stderr.write(line + "\n")

    # Only the default probe reads the prefetched fingerprints.
    if content_probe is None:
        _prefetch_same_content(repaired_obj, master_obj, master_intact, work_dir, deadline, say)

    for holder in AUDIO_HOLDERS:
        for language, audios in (getattr(repaired_obj, holder, None) or {}).items():
            for audio in audios:
                if not audio.get("keep", True):
                    continue
                marker = fabricated_marker_of(audio)
                if not marker:
                    # Unmarked but still rebuilt: race it rather than let it replace an intact track.
                    marker = f"{merge_video_chimeric.REBUILT_MARKER}(unmarked)"
                    audio["fabricated"] = marker
                    say(f"repair: unmarked_rebuilt_track lang={language} holder={holder} "
                        f"stream={audio.get('StreamOrder')} format={audio.get('Format')} "
                        f"marker={marker} reason=a track of the repaired file carried no "
                        f"VMSAM_FABRICATED; raced as rebuilt", to_stderr=True)
                where = (f"lang={language} holder={holder} "
                         f"stream={audio.get('StreamOrder')} "
                         f"format={audio.get('Format')} marker={marker}")
                if holder == "commentary":
                    carrier = []
                    if "commentary" in str(audio.get("Title", "")).lower():
                        carrier.append(f"title={audio.get('Title')}")
                    if (audio.get("properties") or {}).get("flag_commentary"):
                        carrier.append("flag_commentary=true")
                    say(f"repair: fabricated_kept cause=commentary_tagged {where} "
                        f"tagged_by={'+'.join(carrier) or 'unknown'} "
                        f"reason=a commentary is never raced against a main "
                        f"track; delivered with --commentary-flag")
                    _note_kept(kept, audio, language, holder, "commentary_tagged")
                    continue
                opponents = master_intact.get(language, [])
                past_deadline = deadline is not None and time.monotonic() > deadline
                if not len(opponents):
                    cross = (None if past_deadline else
                             _cross_language_match(probe, master_obj, master_intact, language,
                                                   repaired_obj, audio, work_dir, ()))
                    if cross is None or cross[0] is None:
                        cross_measures = " ".join(cross[2]) if cross else ""
                        say(f"repair: fabricated_kept cause=no_intact_master_track {where} "
                            f"{cross_measures + ' ' if cross_measures else ''}"
                            f"reason=the master carries no intact {language} track "
                            f"to race it against")
                        _note_kept(kept, audio, language, holder, "no_intact_master_track")
                        continue
                    _drop_duplicate(dropped, say, holder, language, audio, marker, where,
                                    cross[0], True, cross[2], cross[1])
                    continue
                if past_deadline:
                    import merge_video_chimeric
                    judged = [f"{d['language']}:{d['stream_order']}" for d in dropped]
                    tools.log_always(
                        f"repair: partial_plan cause=repair_budget_exceeded stage=delivery_gate "
                        f"dropped_so_far={judged} next={where} -- the repair's budget ran out "
                        f"while the rebuilt tracks were raced against the master's\n")
                    raise merge_video_chimeric.chimeric_error(
                        f"the repair's budget ran out in the delivery gate, before {where} "
                        f"(dropped so far {judged}) -- declined, retried at the next run",
                        cause="repair_budget_exceeded")
                lost_to = None
                measures = []
                verdicts = []
                track_started = time.monotonic()
                for intact in opponents:
                    same, detail = probe(master_obj, intact, repaired_obj, audio, work_dir)
                    verdicts.append(same)
                    measures.append(f"vs_master_stream={intact.get('StreamOrder')}"
                                    f"[same_content={same} {detail}]")
                    if same is False:
                        continue
                    rival = dict(intact)
                    rival["keep"] = True
                    mergeVideo.keep_best_audio([rival, audio], {})
                    if not audio["keep"]:
                        lost_to = (intact, same)
                        break
                cross_lang = None
                # Only a track every same-language measure called "different" is looked up
                # under the other languages: one that won a same-content race stays as it was.
                if lost_to is None and all(same is False for same in verdicts):
                    raced = tuple(str(intact.get("StreamOrder")) for intact in opponents)
                    intact, cross_lang, cross_measures = _cross_language_match(
                        probe, master_obj, master_intact, language, repaired_obj, audio,
                        work_dir, raced)
                    measures.extend(cross_measures)
                    if intact is not None:
                        lost_to = (intact, True)
                say(f"repair: gate_cost {where} opponents={len(opponents)} "
                    f"seconds={round(time.monotonic() - track_started, 2)}")
                if lost_to is None:
                    say(f"repair: fabricated_kept cause=different_version {where} "
                        f"{' '.join(measures)} reason=its fingerprint matches no "
                        f"intact {language} master track: another version, "
                        f"delivered tagged VMSAM_FABRICATED")
                    _note_kept(kept, audio, language, holder, "different_version")
                    continue
                intact, same = lost_to
                _drop_duplicate(dropped, say, holder, language, audio, marker, where, intact,
                                same, measures, cross_lang)
    return dropped


def _cross_language_match(probe, master_obj, master_intact, language, repaired_obj, audio,
                          work_dir, raced):
    """Find the master's intact track of another language with the same content as `audio`.

    `raced` lists the StreamOrders already compared; an unmeasured pair never matches.
    Returns (intact or None, "<declared>-><matched>" or None, measure strings).
    """
    seen = set(raced)
    measures = []
    for other, intacts in master_intact.items():
        if other == language:
            continue
        for intact in intacts:
            order = str(intact.get("StreamOrder"))
            if order in seen:
                continue
            seen.add(order)
            same, detail = probe(master_obj, intact, repaired_obj, audio, work_dir)
            measures.append(f"vs_master_stream={order}[lang={other} same_content={same} "
                            f"{detail}]")
            if same is True:
                return intact, f"{language}->{other}", measures
    return None, None, measures


def _drop_duplicate(dropped, say, holder, language, audio, marker, where, intact, same,
                    measures, cross_lang):
    """Record and log a rebuilt track dropped as a duplicate of the master's `intact` track."""
    entry = {"kind": "audio", "holder": holder, "language": language,
             "stream_order": audio.get("StreamOrder"),
             "format": audio.get("Format"), "marker": marker,
             "cause": "intact_same_language_wins",
             "kept_master_stream": intact.get("StreamOrder"),
             "same_content": same}
    if cross_lang:
        audio["keep"] = False
        entry["cross_lang"] = cross_lang
    dropped.append(entry)
    reason = ("same content as the master's intact track of another language: a duplicate "
              "under a wrong language tag, intact wins" if cross_lang else
              "same content (or unmeasured), raced by keep_best_audio, intact wins")
    say(f"repair: fabricated_dropped cause=intact_same_language_wins {where} "
        f"kept_master_stream={intact.get('StreamOrder')} "
        f"kept_master_format={intact.get('Format')} "
        f"{f'cross_lang={cross_lang} ' if cross_lang else ''}{' '.join(measures)} "
        f"reason={reason}", to_stderr=True)


def _note_kept(kept, audio, language, holder, cause):
    """Append a kept-track decision to `kept` when a list was given."""
    if kept is not None:
        kept.append({"stream_order": audio.get("StreamOrder"), "language": language,
                     "holder": holder, "cause": cause})


def _subtitle_decline_token(reason):
    """Map a subtitle build's refusal message to a short token."""
    text = str(reason)
    if "bitmap subtitle" in text:
        return "bitmap_subtitle_not_retimable"
    if "carries no cue at all" in text:
        return "source_carries_no_cue"
    if "every cue fell outside" in text:
        return "no_cue_on_master_timeline"
    return "build_declined"


def candidate_non_video_tracks(candidate_obj):
    """List the candidate's non-video tracks once each, audio holders first, then subtitles.

    A track can be aliased under two language keys, so tracks are deduplicated by StreamOrder.

    Returns:
        A list of (kind, holder, language, track dict).
    """
    seen, tracks = set(), []
    for kind, holders in (("audio", ("audios", "audiodesc", "commentary")),
                          ("subtitle", ("subtitles",))):
        for holder in holders:
            for language, entries in (getattr(candidate_obj, holder, None) or {}).items():
                for entry in entries:
                    order = str(entry.get("StreamOrder"))
                    if order in seen:
                        continue
                    seen.add(order)
                    tracks.append((kind, holder, language, entry))
    return tracks


def chimeric_keep_decisions(candidate_obj, plan, assembly=None, repaired_obj=None,
                            refused=None):
    """Summarise, per candidate non-video track, whether the chimeric build kept it and why.

    Reads only decisions already taken by the build and the delivery gates.

    Returns:
        A list of dicts with keys track, lang, kind, holder, kept, reason (a token such as
        source_corrupt, build_declined, repair_refused(<cause>), interior_silence,
        intact_same_language_wins(master_stream=N[,cross_lang=X->Y]) or retimed(...)),
        product_stream and plan.
    """
    track_plans = (plan or {}).get("track_plans") or {}
    assembly = assembly or {}
    audio_reports = assembly.get("audios") or []
    subtitle_reports = assembly.get("subtitles") or []
    produced = {}
    for index, report in enumerate(audio_reports):
        produced[("audio", str(report.get("stream_order")))] = (index, report)
    for index, report in enumerate(subtitle_reports):
        produced[("subtitle", str(report.get("stream_order")))] = (len(audio_reports) + index,
                                                                   report)
    refusals = {(entry.get("kind"), str(entry.get("stream_order"))): (verdict, entry)
                for verdict, entries in (("build_declined", assembly.get("declined") or []),
                                         ("build_failed", assembly.get("failed") or []))
                for entry in entries}
    gate_dropped = {str(entry.get("stream_order")): f"{entry.get('cause')}(master_stream="
                                                     f"{entry.get('kept_master_stream')}"
                                                     + (f",cross_lang={entry['cross_lang']}"
                                                        if entry.get("cross_lang") else "")
                                                     + ")"
                    for entry in assembly.get("fabricated_dropped") or []
                    if isinstance(entry, dict)}
    for entry in assembly.get("silence_dropped") or []:
        gate_dropped.setdefault(str(entry.get("stream_order")), "interior_silence")
    gate_kept = {str(entry.get("stream_order")): entry.get("cause")
                 for entry in assembly.get("gate_kept") or []}
    product_tracks = {}
    if repaired_obj is not None:
        for holder in AUDIO_HOLDERS + ("subtitles",):
            for entries in (getattr(repaired_obj, holder, None) or {}).values():
                for entry in entries:
                    product_tracks[str(entry.get("StreamOrder"))] = entry
    rows = []
    for kind, holder, language, entry in candidate_non_video_tracks(candidate_obj):
        order = str(entry.get("StreamOrder"))
        if kind == "audio":
            track_plan = track_plans.get(int(order)) if order.isdigit() else None
            how = ("no_track_plan" if track_plan is None
                   else "own_offset" if track_plan.get("offset_measured")
                   else f"borrowed_offset({str(track_plan.get('borrow_reason')).replace(' ', '_')})")
        else:
            how = "reference_pieces"
        row = {"track": order, "lang": language, "kind": kind, "holder": holder, "kept": False,
               "reason": None, "product_stream": None, "plan": how}
        rows.append(row)
        made = produced.get((kind, order))
        if made is not None:
            row["product_stream"] = str(made[0])
        if refused is not None:
            row["reason"] = f"repair_refused({refused})"
            continue
        if kind == "audio" and entry.get("dropped_corrupt"):
            row["reason"] = "source_corrupt"
            continue
        if (kind, order) in refusals:
            verdict, refusal = refusals[(kind, order)]
            row["reason"] = (_subtitle_decline_token(refusal.get("reason"))
                             if kind == "subtitle" and verdict == "build_declined" else verdict)
            continue
        if made is None:
            row["reason"] = "not_built"
            continue
        product = product_tracks.get(row["product_stream"])
        if kind == "subtitle":
            report = made[1]
            row["kept"] = product is None or product.get("keep", True) is not False
            row["reason"] = (f"retimed(cues_kept={report.get('kept_cues')},"
                             f"cues_dropped={report.get('dropped_cues')})")
            continue
        if row["product_stream"] in gate_dropped:
            row["reason"] = gate_dropped[row["product_stream"]]
            continue
        if product is not None and product.get("keep", True) is False:
            row["reason"] = "keep_false"
            continue
        row["kept"] = True
        row["reason"] = gate_kept.get(row["product_stream"]) or "unjudged"
    return rows


def log_chimeric_keep(candidate_obj, plan, assembly=None, repaired_obj=None, refused=None):
    """Log one `repair: chimeric_keep` line per candidate track plus a summary (dev mode only)."""
    if not tools.dev:
        return []
    try:
        rows = chimeric_keep_decisions(candidate_obj, plan, assembly, repaired_obj, refused)
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"repair: chimeric_keep unavailable {type(error).__name__}: {error}\n")
        return []
    for row in rows:
        tools.dev_log(f"repair: chimeric_keep track={row['track']} lang={row['lang']} "
                      f"kept={'yes' if row['kept'] else 'no'} reason={row['reason']} "
                      f"kind={row['kind']} holder={row['holder']} "
                      f"product_stream={row['product_stream']} plan={row['plan']} "
                      f"for {candidate_obj.filePath}\n")
    tools.dev_log(f"repair: chimeric_keep_summary tracks={len(rows)} "
                  f"kept={sum(1 for row in rows if row['kept'])} "
                  f"dropped={sum(1 for row in rows if not row['kept'])} "
                  f"for {candidate_obj.filePath}\n")
    return rows


def quanta(value_ms, quantum_ms):
    """Express a distance in fingerprint quanta rather than ms; None when not computable.

    The quantum varies per file, so thresholds in quanta are file-independent.
    """
    if value_ms == None or quantum_ms in (None, 0):
        return None
    try:
        return round(float(Decimal(str(value_ms)) / Decimal(str(quantum_ms))), 2)
    except Exception:
        return None


def _track_shortfall_ms(assembly, report):
    """Return how much shorter (ms) this produced track is than expected, or None if unread."""
    check = assembly.get("output_check") or {}
    expected = check.get("expected_duration_ms")
    if expected == None:
        return None
    for stream in check.get("streams") or []:
        if stream.get("codec_type") != "audio":
            continue
        if str(stream.get("language")) != str(report.get("language")):
            continue
        if stream.get("duration_ms") == None:
            return None
        return Decimal(str(expected)) - Decimal(str(stream["duration_ms"]))
    return None


def _margin_fields(plan):
    """Format the plan's speed_margin, fidelity_margin and decided_by fields for a log line.

    A missing field is written as `absent(<reason>)`.
    """
    if not plan:
        return ""
    parts = []
    margin = get_speed_margin(plan)
    if margin != None:
        parts.append(f"speed_margin={margin}")
    else:
        reason = plan.get("speed_margin_absent_reason")
        parts.append(f"speed_margin=absent({reason})" if reason != None
                     else "speed_margin=absent(not_in_plan)")
    fidelity = plan.get("fidelity_margin")
    parts.append(f"fidelity_margin={fidelity}" if fidelity != None
                 else "fidelity_margin=absent(not_in_plan)")
    decided = plan.get("decided_by")
    parts.append(f"decided_by={decided}" if decided != None
                 else "decided_by=absent(not_in_plan)")
    return (" ".join(parts) + " ") if len(parts) else ""


def _head_pad_summary(report):
    """Count the report's head-padding decisions per outcome.

    Outcomes: unmeasured, read_past (the plan already reads past a late stream start),
    none (stream starts at zero), padded (silence added).
    """
    decisions = report.get("head_decisions")
    if decisions == None:
        return ("unreported(no head_decisions on this report; expected only for "
                "assemblies predating the field)")
    if not len(decisions):
        return "no-candidate-piece"
    counts = {}
    for decision in decisions:
        counts[decision["outcome"]] = counts.get(decision["outcome"], 0) + 1
    return ",".join(f"{name}={counts[name]}" for name in sorted(counts))


def _shortfall_annotation(assembly, report):
    """Describe a track's duration loss versus its fill-source shortfall, and the unexplained rest.

    The unexplained residual is never negative; an over-count is stated in words.
    """
    lost = _track_shortfall_ms(assembly, report)
    short = report.get("fill_short_by_ms")
    if short:
        if lost == None:
            return "[FILL SOURCE SHORT BY " + str(short) + " ms; TRACK LOSS UNMEASURED]"
        residual = Decimal(str(lost)) - Decimal(str(short))
        if residual > 0:
            tail = "UNEXPLAINED " + str(residual) + " ms"
        else:
            tail = ("UNEXPLAINED 0 ms (the fill shortfall over-accounts by "
                    + str(-residual) + " ms)")
        return ("[FILL SOURCE SHORT BY " + str(short) + " ms; TRACK LOST "
                + str(lost) + " ms; " + tail + "]")
    if lost != None and lost > 0:
        return "[TRACK LOST " + str(lost) + " ms, NO SHORT FILL SOURCE -- UNEXPLAINED]"
    return ""


def _digest_of_loaded_source():
    """Return a short sha256 of this module's file, computed at import time."""
    import hashlib
    try:
        with open(__file__, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()[:12]
    except Exception:
        return "unreadable"


LOADED_SOURCE_DIGEST = _digest_of_loaded_source()


# Digest of the running sources, computed once per process.
_sources_digest_cache = None

# Files the container image ships (relative to this module's directory), so a checkout and the
# image hash the same set.
SOURCE_SCOPE = ("*.py", "gestionar_show/**/*.py", "gestionar_movie/**/*.py")


def sources_digest():
    """Return a digest of the deployed SOURCE_SCOPE sources, computed once per process.

    Returns:
        A dict with sha12 (rolled digest), files (count), scope, root and per_file entries.
    """
    global _sources_digest_cache
    if _sources_digest_cache != None:
        return _sources_digest_cache
    import glob, hashlib
    root = path.dirname(path.abspath(__file__))
    found = {}
    for pattern in SOURCE_SCOPE:
        for name in glob.glob(path.join(root, pattern), recursive=True):
            if path.isfile(name):
                found[path.relpath(name, root)] = name
    per_file, rolled = [], hashlib.sha256()
    # glob has no guaranteed order.
    for relative in sorted(found):
        try:
            with open(found[relative], "rb") as handle:
                payload = handle.read()
        except OSError as error:
            # Named rather than skipped, so it differs from a missing file.
            digest = f"unreadable({type(error).__name__})"
            rolled.update(relative.encode("utf-8") + b"\x00" + digest.encode("utf-8") + b"\n")
            per_file.append({"path": relative, "sha12": digest})
            continue
        one = hashlib.sha256(payload).hexdigest()
        rolled.update(relative.encode("utf-8") + b"\x00" + one.encode("utf-8") + b"\n")
        per_file.append({"path": relative, "sha12": one[:12], "bytes": len(payload)})
    _sources_digest_cache = {"sha12": rolled.hexdigest()[:12],
                             "files": len(per_file),
                             "scope": " + ".join(SOURCE_SCOPE),
                             "root": root,
                             "per_file": per_file}
    return _sources_digest_cache


def write_sources_manifest():
    """Write the per-file sources manifest (once per digest) and return its path, or None."""
    digest = sources_digest()
    try:
        # Content-addressed name, so a manifest can never be stale.
        target = path.join(tools.tmpFolder,
                           f"vmsam_sources_{digest['sha12']}.json")
        if not path.exists(target):
            import json as _json
            with open(target, "w") as handle:
                _json.dump({"sha12": digest["sha12"], "files": digest["files"],
                            "scope": digest["scope"], "root": digest["root"],
                            "per_file": digest["per_file"]}, handle, indent=1)
        return target
    except Exception as error:
        tools.logs.append(f"repair: the sources manifest could not be written: {error}\n")
        return None


def module_fingerprint():
    """Identify the running repair code by the import-time digests of the two repair modules."""
    parts = [f"{path.basename(__file__)}:{LOADED_SOURCE_DIGEST}"]
    try:
        import merge_video_chimeric as _chi
        parts.append(f"{path.basename(_chi.__file__)}:"
                     f"{getattr(_chi, 'LOADED_SOURCE_DIGEST', 'unreported')}")
    except Exception:
        parts.append("merge_video_chimeric.py:unimportable")
    return " ".join(parts)


def master_fill_offset(region):
    """Return a master piece's computed offset (source start minus master start); expected 0."""
    try:
        return (Decimal(str(region.get("source_start_ms")))
                - Decimal(str(region.get("master_start_ms"))))
    except Exception:
        return "unreported(bounds unreadable)"


def log_assembly(candidate_path, assembly, plan):
    """Log what was done to the file, track by track.

    Skipped, declined and failed tracks are logged with their reason, never omitted.
    """
    quantum_ms = plan.get("quantum_ms") if plan else None
    pieces = assembly.get("pieces") or []
    spans = []
    for piece in pieces:
        start = Decimal(str(piece["master_start_ms"]))
        end = Decimal(str(piece["master_end_ms"]))
        # Printed as Decimal so it matches the bound on the ADDED line exactly.
        spans.append(f"{piece['source'][0]}{start}-{end}")
    tools.logs.append(f"repair: build {module_fingerprint()}\n")
    _sources = sources_digest()
    _manifest = write_sources_manifest()
    tools.logs.append(f"repair: sources {_sources['sha12']} "
                      f"files={_sources['files']} scope={_sources['scope']} "
                      f"manifest={_manifest or 'unwritten'}\n")
    if plan and plan.get("master_path"):
        tools.logs.append(f"repair: master {plan['master_path']}\n")
    # Hash of the path string, so candidates of one master stay distinguishable.
    if candidate_path:
        import hashlib
        tools.logs.append(
            f"repair: candidate_digest "
            f"{hashlib.sha256(str(candidate_path).encode()).hexdigest()}\n")
    for index, segment in enumerate(plan.get("segments") or []):
        by_stream = segment.get("candidate_offset_ms_by_stream")
        tools.logs.append(
            f"repair: segment {index} "
            f"master={segment.get('master_start_ms')}-{segment.get('master_end_ms')} "
            f"base_offset_ms={segment.get('candidate_offset_ms')}"
            f"{'(' + str(segment['offset_origin']) + ')' if segment.get('offset_origin') else ''} "
            f"by_stream={by_stream if by_stream else 'none'}\n")

    for index, change in enumerate(plan.get("change_points") or []):
        low = change.get("bracket_low_ms")
        high = change.get("bracket_high_ms")
        width = (Decimal(str(high)) - Decimal(str(low))
                 if low != None and high != None else None)
        tools.logs.append(
            f"repair: bracket {index} low_ms={low} high_ms={high} "
            f"width_ms={width if width != None else 'unreported'} "
            # True when the position is only bounded by a whole inter-window interval.
            f"bound_only={change.get('bracket_is_bound_only')} "
            f"{'clamped_to_next=true ' if change.get('bracket_clamped_to_next') else ''}"
            f"step_ms={change.get('step_ms')} "
            f"step_points={change.get('step_points')}\n")

    dropped_note = plan.get("segments_dropped_unusable") if plan else None
    tools.logs.append(f"repair: plan {plan.get('kind') if plan else 'none'} "
                      f"build={repair_log.build_sha()} "
                      f"{'language_route=' + str(plan['language_route']).replace(' ', '_') + ' ' if plan and plan.get('language_route') else ''}"
                      f"{'dropped_segments=' + str(dropped_note) + ' ' if dropped_note else ''}"
                      f"{'dropped_segments=unreported(locator did not report it) ' if plan and 'segments_dropped_unusable' not in plan else ''}"
                      f"language={plan.get('language') if plan else None} "
                      # Picture offset d: candidate frame k+d shows master frame k.
                      f"{'video_anchored=' + format(plan['video_anchored']['offset_frames'], '+d') + ' video_shift_frames=' + format(plan['video_anchored']['shift_frames'], '+d') + ' ' if plan and plan.get('video_anchored') else ''}"
                      f"{_margin_fields(plan)}"
                      f"quantum={quantum_ms}"
                      # A quantum is only comparable with the probe window that produced it.
                      f"{'@window_s=' + str(plan['probe_window_seconds']) if plan and plan.get('probe_window_seconds') != None else ''} "
                      f"pieces={' '.join(spans)}\n")

    verification = {}
    for entry in assembly.get("verification") or []:
        verification[entry.get("track")] = entry

    verified_count = sum(1 for v in verification.values()
                         if v.get("outcome") not in (None, "skipped"))
    for report in assembly.get("audios") or []:
        checked = verification.get(report["stream_order"], {})
        worst = checked.get("worst_lag_ms")
        line = (f"repair: audio track {report['stream_order']} "
                f"lang={report['language']} "
                f"fill={report['gap_fill']}"
                f"{'/' + str(report['fill_language']) if report.get('fill_language') else ''}"
                f"{'[' + str(report['fill_title']) + ']' if report.get('fill_title') else ''}"
                # AMBIGUOUS: several master tracks in the fill language, chosen without measurement.
                f"{('(among ' + str(report['fill_choices']) + ' by measurement)' if report.get('fill_by_reference') else '(AMBIGUOUS among ' + str(report['fill_choices']) + ')') if (report.get('fill_choices') or 0) > 1 else ''}"
                f"{_shortfall_annotation(assembly, report)} "
                f"filled_ms={report['gap_filled_ms']} "
                f"silence_ms={report['silence_filled_ms']} "
                f"head_pad_ms={report['head_pad_ms']} "
                f"head_pad={_head_pad_summary(report)} "
                + ("cross_language_fill=true "
                   if (report.get("fill_language")
                       and report.get("fill_language") != report.get("language"))
                   else "")
                + (f"tool_split={';'.join(report['tool_disagreements'])} "
                   if report.get("tool_disagreements") else "")
                + 
                # head: master/<lang>, NO-HEAD, unprobed or silence.
                f"{'head=' + str(report['head_source']) + ' ' if report.get('head_source') else ''}"
                f"speed={report.get('speed_ratio_applied') if report.get('speed_ratio_applied') != None else 'none(no rate proposed by the measurement)'} "
                # BORROWED: the track uses another language's offset.
                f"offset={'measured' if report.get('offset_measured') else 'BORROWED'}"
                f"{'[' + str(report['borrow_reason']) + ']' if report.get('borrow_reason') else ''}"
                f"{'(fid ' + str(report['offset_fidelity']) + ')' if report.get('offset_fidelity') != None else ''} "
                f"verify={checked.get('outcome')}"
                f"{'(' + str(checked['reason']) + ')' if checked.get('outcome') == 'skipped' and checked.get('reason') else ''} "
                f"residual=probes={checked.get('probes_measured')} "
                f"worst={quanta(worst, quantum_ms)}q "
                f"quantum={quantum_ms}ms "
                # Weakest correlation: `aligned` alone does not rule out unrelated content.
                f"{'r_min=' + str(checked['weakest_correlation']) + ' ' if checked.get('weakest_correlation') != None else ''}"
                # Kept probes' RMS relative to `verify_min_rms`, below which probes are discarded.
                f"{'rms_over_floor=' + str(checked['rms_over_floor']) + 'x ' if checked.get('rms_over_floor') != None else ''}"
                # Index in the produced file (`stream_order` is the candidate's).
                f"produced_index={checked.get('produced_index') if checked.get('produced_index') != None else 'unknown'} "
                f"verified={verified_count}/{len(assembly.get('audios') or [])}\n")
        tools.logs.append(line)
        # One line per region: USED (from the candidate), ADDED (filled), CUT (dropped).
        for region in report.get("used_regions") or []:
            tools.logs.append(
                f"repair: USED audio track {report['stream_order']} "
                f"master {region['master_start_ms']}-{region['master_end_ms']} "
                f"candidate {region['candidate_start_ms']}-{region['candidate_end_ms']} "
                f"offset_ms={region['offset_ms']}\n")
        for region in report.get("filled_regions") or []:
            tools.logs.append(
                f"repair: ADDED audio track {report['stream_order']} "
                f"master {region['master_start_ms']}-{region['master_end_ms']} "
                f"why={region.get('reason') or 'absent(region carries no reason; cause of the absence NOT established)'} "
                f"from={region['source']}"
                f"{'/' + str(region['language']) if region.get('language') else ''}"
                f" stream={report.get('fill_stream_order') if report.get('fill_stream_order') != None else 'unknown'}"
                f" offset_ms={master_fill_offset(region)} "
                f"fill_source_class={region.get('fill_source_class') or 'unreported'}"
                f"{' frame_tier_declined_reason=' + str(region['frame_tier_declined_reason']) if region.get('frame_tier_declined_reason') else ''}"
                f"{' frame_tier_declined_evidence=' + repr(str(region['frame_tier_declined_evidence'])) if region.get('frame_tier_declined_evidence') else ''}"
                "\n")
        for region in report.get("cut_regions") or []:
            if region.get("unmeasured"):
                tools.logs.append(
                    f"repair: CUT audio track {report['stream_order']} "
                    f"candidate {region['candidate_start_ms']}-? "
                    f"where={region.get('where')} dropped_ms=UNMEASURED "
                    f"(the candidate duration was not available)\n")
                continue
            tools.logs.append(
                f"repair: CUT audio track {report['stream_order']} "
                f"candidate {region['candidate_start_ms']}-"
                f"{region['candidate_end_ms']} dropped_ms={region['dropped_ms']} "
                f"where={region.get('where')}\n")

    for report in assembly.get("subtitles") or []:
        tools.logs.append(f"repair: subtitle track {report['stream_order']} "
                          f"lang={report['language']} format={report.get('format')} "
                          f"kept_cues={report.get('kept_cues')} "
                          f"shifts_ms={report.get('shifts_applied_ms') or 'none'} "
                          f"dropped_cues={report.get('dropped_cues')}\n")

    # Skipped segments are filled from the master; logged so they differ from plan holes.
    for entry in assembly.get("dropped_segments") or []:
        tools.logs.append(
            f"repair: SKIPPED segment master {entry['master_start_ms']}-"
            f"{entry['master_end_ms']} dropped_ms={entry['dropped_ms']} "
            f"DECLINED: offset unverified (segment shorter than the "
            f"measurement's probe window); this span is filled from the master "
            f"instead of the candidate\n")

    for entry in assembly.get("declined") or []:
        tools.logs.append(f"repair: SKIPPED {entry.get('kind')} track "
                          f"{entry.get('stream_order')} DECLINED: "
                          f"{entry.get('reason')}\n")
    for entry in assembly.get("failed") or []:
        tools.logs.append(f"repair: SKIPPED {entry.get('kind')} track "
                          f"{entry.get('stream_order')} FAILED: "
                          f"{entry.get('reason')}\n")

    check = assembly.get("output_check")
    if check:
        tools.logs.append(f"repair: output file audio {check['audio_in_file']}/"
                          f"{check['audio_built']} subtitles "
                          f"{check['subtitles_in_file']}/{check['subtitles_built']} "
                          f"expected_ms={check['expected_duration_ms']} "
                          f"source={check['expected_duration_source']} "
                          f"frame_rate={assembly.get('master_frame_rate') or 'unread'}"
                          f"({assembly.get('master_frame_rate_mode') or 'mode unread'}"
                          f"{',used' if assembly.get('master_frame_rate_original') else ''}) "
                          f"{'frame_rate_original=' + str(assembly['master_frame_rate_original']) + ' ' if assembly.get('master_frame_rate_original') else ''}"
                          f"tolerance_ms={check['tolerance_ms']} "
                          f"measured={check.get('measured')} "
                          f"would_refuse={check.get('would_refuse')} "
            f"{'-- 1 WOULD HAVE BEEN DECLINED (gate inert) ' if check.get('would_refuse') and not check.get('enforcing') else ''}"
                          f"enforcing={check.get('enforcing')}\n")
        # One line per problem behind `would_refuse`.
        for problem in (check.get("problems") or []):
            tools.logs.append(f"repair: output problem {problem}\n")
        # Separates a subtitle-length container duration from a build defect.
        if check.get("container_duration_ms") != None or check.get("max_av_stream_duration_ms") != None:
            tools.logs.append(
                f"repair: output durations container_ms={check.get('container_duration_ms')} "
                f"max_av_stream_ms={check.get('max_av_stream_duration_ms')} "
                f"expected_ms={check.get('expected_duration_ms')}\n")


def decline_detail(error):
    """Extract the detail fields a decline error carries.

    `undelivered_state` is REFUSED (the gate decided against) or NOVERDICT (a tool failed
    before any verdict); `undelivered_path` is None when no artefact was marked.
    """
    return {"verification": getattr(error, "verification", None),
            "audios": getattr(error, "audios", None),
            "output_check": getattr(error, "output_check", None),
            "undelivered_state": getattr(error, "undelivered_state", None),
            "undelivered_path": getattr(error, "undelivered_path", None)}


def chimeric_cause(error):
    """Return a chimeric_error's cause token, or `(untokened_raise_site_<line>)` without one.

    The parentheses keep the sentinel from parsing as a cause (`cause=([A-Za-z0-9_]+)`); the line
    is the raise site in merge_video_chimeric.py, taken from the traceback.
    """
    cause = getattr(error, "cause", None)
    if cause != None:
        return cause
    line = None
    traceback_entry = getattr(error, "__traceback__", None)
    while traceback_entry != None:
        if traceback_entry.tb_frame.f_code.co_filename.endswith(
                "merge_video_chimeric.py"):
            line = traceback_entry.tb_lineno
        traceback_entry = traceback_entry.tb_next
    if line == None:
        return "(untokened_raise_site)"
    return f"(untokened_raise_site_{line})"


def detail_summary(detail):
    """Summarise a refusal's `detail` as key=value decision fields, without diagnostics.

    Lists are reduced to counts and checks to `present`; the full dump is logged in dev mode.
    """
    if not detail:
        return ""
    fields = []
    for key in ("plan_kind", "verdict", "plan_source", "marker",
                "undelivered_state"):
        value = detail.get(key)
        if value != None:
            fields.append(f"{key}={value}")
    for key in ("audios", "subtitles", "declined", "failed", "coarse_brackets",
                "fabricated_dropped"):
        value = detail.get(key)
        if isinstance(value, (list, tuple)):
            fields.append(f"{key}={len(value)}")
    for key in ("output_check", "verification"):
        value = detail.get(key)
        if value != None:
            fields.append(f"{key}=present")
    return " ".join(fields)


def record(candidate_path, outcome, reason, detail=None, cause=None):
    """Record and log one candidate's repair outcome; returns the report entry."""
    entry = {"candidate": candidate_path, "outcome": outcome, "reason": reason,
             "detail": detail, "cause": cause}
    last_repair_report.append(entry)
    # Format: `repair: <outcome> cause=<token> for <path>: <prose>`. The cause precedes the path
    # so a filename cannot forge one.
    head = f"repair: {outcome}"
    if cause != None:
        head += f" cause={cause}"
    tools.log_always(f"{head} for {candidate_path}: {reason}\n")
    # `repair_detail:` prefix so the outcome parser does not read it as an outcome.
    if outcome in ("declined", "failed") and detail:
        summary = detail_summary(detail)
        if summary:
            tools.logs.append(f"repair_detail: {outcome} {summary}\n")
        if tools.dev:
            # default=str: the plan carries Decimals.
            try:
                dump = json.dumps(detail, default=str, sort_keys=True)
            except Exception as error:
                dump = f"<undumpable: {type(error).__name__}: {error}>"
            tools.dev_log(f"repair_detail_verbose: {outcome} {dump}\n")
    return entry


def master_intertrack_verdict(best_video, language, cache):
    """Check whether the master's own tracks in `language` agree with each other.

    A master may carry two same-language tracks offset from each other, which would wrongly
    fail a candidate compared against only one. Verdicts are cached per language in `cache`
    (one dict per master).

    Returns:
        The `master_self_check.check_master_intertrack` verdict, or None when not measured.
    """
    if language in cache:
        return cache[language]
    verdict = None
    try:
        import master_self_check
    except Exception as error:
        tools.dev_log(f"repair: no master_self_check module: {error}\n")
    else:
        tools.dev_log(f"repair: master_intertrack_verdict starting on "
                      f"master={best_video.filePath} language={language}\n")
        try:
            verdict = master_self_check.check_master_intertrack(
                best_video, language)
        except Exception as error:
            tools.dev_log(f"repair: master_intertrack_verdict raised "
                          f"{type(error).__name__}: {error} -- no verdict, the "
                          f"chain continues unchanged\n")
            verdict = None
    cache[language] = verdict
    return verdict


def _drain_audio_pools(objs, site):
    '''Wait for pending ffmpeg audio extractions on each object (None entries skipped).

    Unawaited extractions left by mergeVideo can deadlock `Pool.terminate()` in fusion.py.
    Failures are logged, never raised.
    '''
    for obj in objs:
        if obj is None:
            continue
        try:
            obj.wait_end_ffmpeg_progress_audio()
        except Exception as error:
            tools.dev_log(f"repair: draining pending audio extraction at "
                          f"{site} raised {type(error).__name__}: {error}\n")


def retire_ffmpeg_pools(grace_seconds=300):
    '''Shut down video's ffmpeg pools with close() and join(), never terminate().

    terminate() can deadlock on `inqueue._rlock`; workers still busy after `grace_seconds` are
    SIGKILLed. The pools are left closed: any later `apply_async` raises ValueError.
    '''
    import multiprocessing.pool, video, signal, time, os
    for name in ("ffmpeg_pool_audio_convert", "ffmpeg_pool_big_job"):
        pool = getattr(video, name, None)
        if pool == None:
            continue
        try:
            pool.close()
            deadline = time.monotonic() + grace_seconds
            while (time.monotonic() < deadline
                   and any(p.is_alive() for p in pool._pool)):
                time.sleep(0.1)
            survivors = [p for p in pool._pool if p.is_alive()]
            if len(survivors):
                # A killed worker's task stays in `pool._cache`, so the handler threads would
                # never stop and join() would hang; stop both handlers before killing.
                pool._worker_handler._state = multiprocessing.pool.TERMINATE
                pool._result_handler._state = multiprocessing.pool.TERMINATE
                pool._change_notifier.put(None)
                for p in survivors:
                    tools.log_always(f"repair: retiring {name}: worker {p.pid} "
                                     f"outlived {grace_seconds}s, SIGKILL\n")
                    os.kill(p.pid, signal.SIGKILL)
            pool.join()
            if len(survivors):
                # Leftover entries belong to killed tasks; a non-empty cache makes a later
                # terminate() raise AssertionError.
                pool._cache.clear()
        except Exception as error:
            tools.dev_log(f"repair: retiring {name} raised "
                          f"{type(error).__name__}: {error}\n")


# `repair_orchestrator.repair()` returns a bool, so the repaired object travels back in a dict
# set on the candidate under this attribute: {"job_start_utc" (in), "repaired_obj" and
# "assembly" (out, set by `apply_plan`)}. It is removed after each call.
REPAIR_SEAM_ATTRIBUTE = "vmsam_repair_seam"


def _open_repair_seam(candidate_obj, job_start_utc):
    """Attach a fresh seam dict to the candidate and return it."""
    seam = {"job_start_utc": job_start_utc, "repaired_obj": None, "assembly": None}
    setattr(candidate_obj, REPAIR_SEAM_ATTRIBUTE, seam)
    return seam


def _close_repair_seam(candidate_obj):
    """Detach and return the candidate's seam dict ({} when absent)."""
    seam = getattr(candidate_obj, REPAIR_SEAM_ATTRIBUTE, None)
    if seam is not None:
        delattr(candidate_obj, REPAIR_SEAM_ATTRIBUTE)
    return seam or {}


def _terminal_cause_since(candidate_path, reported_before):
    """Return the cause last recorded for this candidate since `reported_before`, or None."""
    for entry in reversed(last_repair_report[reported_before:]):
        if entry.get("candidate") == candidate_path:
            return entry.get("cause")
    return None


def repair_not_compatible_videos(list_not_compatible_video, dict_file_path_obj,
                                 best_video, language):
    """Try to repair each rejected candidate and attach the repaired objects to the merge.

    Repaired objects are appended to `best_video.sameAudioMD5UseForCalculation`; rejected paths
    stay out of `dict_file_path_obj`.

    Args:
        list_not_compatible_video: rejected candidate paths.
        dict_file_path_obj: path -> video object.
        best_video: the master video object.
        language: the comparison language the merge measured the delay on.

    Returns:
        The list of candidate paths that were repaired and attached.
    """
    import repair_orchestrator
    import merge_video_chimeric
    del last_repair_report[:]
    work_root = path.join(tools.tmpFolder, "repair")
    tools.make_dirs(work_root)
    repaired = []
    master_intertrack_by_language = {}

    for candidate_path in list_not_compatible_video:
        job_start_utc = datetime.now(timezone.utc).isoformat()
        # Logged first so a hang's last line names the file.
        tools.dev_log(f"repair: repair_not_compatible_videos starting on "
                      f"{candidate_path}\n")
        candidate_obj = dict_file_path_obj.get(candidate_path)
        if candidate_obj == None:
            repair_orchestrator._plan_line("none", candidate_path, step="entry",
                                           cause="candidate_object_absent")
            record(candidate_path, "declined",
                   "the rejected path has no video object in dict_file_path_obj",
                   cause="candidate_object_absent")
            _drain_audio_pools((best_video, candidate_obj),
                               "candidate_object_absent")
            continue
        tools.dev_log(f"repair: comparison language={language} "
                      f"route=passed_by_the_caller(get_delay) for {candidate_path}\n")

        _open_repair_seam(candidate_obj, job_start_utc)
        reported_before = len(last_repair_report)
        try:
            ok = repair_orchestrator.repair(
                best_video, candidate_obj, language,
                work_root=path.join(work_root,
                                    merge_video_chimeric.stable_case_key(candidate_path),
                                    "orchestrator"),
                master_intertrack_cache=master_intertrack_by_language)
        except Exception as error:
            _close_repair_seam(candidate_obj)
            # Never let one candidate's exception abort the others: decoder timeouts and
            # chimeric_error are declines, anything else a failure.
            if isinstance(error, tools.decoder_timeout):
                cause = "decoder_timeout"
                repair_orchestrator.log_measurement_class(candidate_path, cause)
                record(candidate_path, "declined", str(error), cause=cause)
            elif isinstance(error, merge_video_chimeric.chimeric_error):
                cause = chimeric_cause(error)
                repair_orchestrator.log_measurement_class(candidate_path, cause)
                record(candidate_path, "declined", str(error),
                       decline_detail(error), cause=cause)
            else:
                cause = "repair_raised_unhandled"
                record(candidate_path, "failed",
                       f"{type(error).__name__}: {error}", decline_detail(error),
                       cause=cause)
            repair_orchestrator._plan_line("none", candidate_path,
                                           step="orchestrator_raised", cause=cause)
            sys.stderr.write(f"repair: {cause} for {candidate_path}: {error}\n")
            _drain_audio_pools((best_video, candidate_obj), cause)
            continue
        seam = _close_repair_seam(candidate_obj)
        if not ok:
            # Already recorded by the orchestrator; drain pending extractions on every refusal.
            cause = _terminal_cause_since(candidate_path, reported_before)
            _drain_audio_pools((best_video, candidate_obj),
                               cause or "orchestrator_declined")
            tools.log_always(f"repair: drain complete at {cause} "
                             f"for {candidate_path}\n")
            continue
        repaired_obj = seam.get("repaired_obj")
        if repaired_obj == None:
            record(candidate_path, "failed",
                   f"the orchestrator returned True but handed over no repaired "
                   f"object through `{REPAIR_SEAM_ATTRIBUTE}['repaired_obj']`: "
                   f"the seam with plan application is broken, nothing is "
                   f"attached and the refusal stands",
                   cause="repaired_object_missing")
            continue
        sys.stdout.write(f"\tRepaired {candidate_path} as "
                         f"{getattr(repaired_obj, 'filePath', repaired_obj)}\n")
        repaired.append(candidate_path)
        best_video.sameAudioMD5UseForCalculation.append(repaired_obj)
    return repaired
