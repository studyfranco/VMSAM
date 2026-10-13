# -*- coding: utf-8 -*-
"""Stage 1/2 of the repair pipeline: similarity gate and resample sweep.

When the native alignment of a couple is poor (the aligner could not measure, its zones cover
too little of the master, or they form a rate ladder), every named speed ratio is tried in both
engines (`rate_arm`) and the admissible finalist with the best fidelity is kept
(`rate_direction.choose_winner`). A confirmed ratio is then turned into the routing that
re-primes the candidate at that speed (`rate_resample_routing`, with the pitch layer as an
observation).

The step logging and the fingerprint alignment live in `repair_orchestrator`; they are read at
call time, so this module can be imported on its own.
"""
from decimal import Decimal
from fractions import Fraction
from os import path, remove
import subprocess
import time

import audioCorrelation
import audio_extract
import banded_seed_alignment
import repair_log
import tools


def _orch():
    import repair_orchestrator
    return repair_orchestrator


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
# (drift has a sign, editing does not); a line fit of the drift trace
# (`alignment_diagnostics.best_shift_trace`) does not. A false positive costs one rate sweep and
# then continues on the alignment already measured.
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
    "repair_stage1_resample: banded_seed_alignment.COULD_NOT_MEASURE_VERDICTS is empty -- the "
    "step-2 similarity gate would never fire")
assert not (set(ALIGNMENT_COULD_NOT_MEASURE_VERDICTS)
            & set(banded_seed_alignment.MEASURED_VERDICTS)), (
    "repair_stage1_resample: a verdict cannot be both a measurement and a could-not-measure")


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
    master_duration = _orch()._track_duration_seconds(master_obj, language, master_stream)
    candidate_duration = _orch()._track_duration_seconds(candidate_obj, language, candidate_stream)
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
        "modality": _orch().MODALITY,
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
    _zones, detail = _orch().coalesce_same_offset_zones(alignment.get("zones") or [],
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
                    alignment = _orch().align_fingerprints(fp_master, quantum_master, duration_master,
                                                   points, _orch().CHROMAPRINT_HOP_MS, corrected)
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
        _orch().step_result("similarity_gate_couple", candidate=candidate_path, couple=name,
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
