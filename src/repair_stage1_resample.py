# -*- coding: utf-8 -*-
"""Stage 1/2 of the repair: similarity gate, resample sweep and the re-prime routing."""
from decimal import Decimal
from fractions import Fraction
from os import path, remove
import time

import audioCorrelation
import banded_seed_alignment
import progress_watchdog
import repair_log
import tools


def _orch():
    import repair_orchestrator
    return repair_orchestrator


# Coverage floor of the step-2 gate: `single_segment_no_cut` means "no offset step found", not
# "aligned". Healthy couples read >= 0.50, rate-mismatched or wrong-episode couples <= 0.31; 0.40
# sits in the empty band between them. Terminal when the rate arm cannot raise the coverage.
MASTER_AXIS_COVERAGE_FLOOR = 0.40

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
    routing = {"filter_name": engine,
               "route_reason": f"engine {engine} measured by the rate arm (the finalist that "
                               f"aligned)"}
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


def rate_arm(primed, first_ratios, work_dir, candidate_path, videos=None):
    """Find which named speed ratio and engine, if any, best aligns the candidate.

    The prime's kept candidate WAV is resampled at each (ratio, engine), fingerprinted and
    aligned against the master; the prime's own alignment is the finalist at 1. Round one
    tries `first_ratios` in both engines; round two, only without a winner, every named ratio.
    The winner is `rate_direction.choose_winner`'s; with `videos` (master, candidate), near-tied
    finalists naming different ratios are separated by the pictures (`picture_tie_break`).
    The WAV is deleted on every path.

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
        mappings = {(Fraction(1), None): (baseline, Fraction(1))}

        def run_round(ratios):
            for ratio in ratios:
                for engine in merge_video_resample.SPEED_ENGINES:
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
                            done = progress_watchdog.run(command, "rate_arm_resample")
                            call["exit"] = done.returncode
                        if done.returncode != 0:
                            continue
                        with repair_log.announced("orchestrator", "fpcalc", out) as call:
                            points = audioCorrelation.calculate_fingerprints(
                                out, length=corrected)
                            call["exit"] = 0
                    except Exception as error:                           # noqa: BLE001
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
                    alignment = _orch().align_fingerprints(
                        fp_master, quantum_master, duration_master, points,
                        _orch().CHROMAPRINT_HOP_MS, corrected)
                    mappings[(ratio, engine)] = (alignment, Fraction(str(effective)))
                    rows.append(_finalist_row(ratio, engine, alignment,
                                              min(master_duration_ms, corrected * 1000.0),
                                              time.time() - started, candidate_path))
                tried.add(ratio)

        rounds = [list(first_ratios),
                  [r for r in merge_video_resample.build_rate_ratio_vocabulary()]]
        winner = None
        for number, ratios in enumerate(rounds, 1):
            ratios = [r for r in ratios if r not in tried]
            if not ratios and number > 1:
                break
            gate["rounds"] = number
            run_round(ratios)
            break_tie = (None if videos is None else
                         lambda tied: picture_tie_break(tied, mappings, videos, candidate_path))
            winner = rate_direction.choose_winner(rows, break_tie=break_tie)
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


def _time_mapping(alignment, effective):
    """Return master_s -> native candidate_s through the zone holding that instant, or None."""
    zones = [(zone["master_ms"][0], zone["master_ms"][1], zone["offset_ms"])
             for zone in alignment.get("zones_detail") or []]

    def mapping(master_s):
        master_ms = master_s * 1000.0
        for low, high, offset_ms in zones:
            if low <= master_ms < high:
                return float((master_ms + offset_ms) / 1000.0 / float(effective))
        return None
    return mapping


def picture_tie_break(tied, mappings, videos, candidate_path):
    """Pick among near-tied finalists of different ratios by the pictures, or return None.

    Each finalist's zones map a master instant to the candidate's own clock at its ratio;
    `frame_snap.ratio_picture_scores` reads which mapping lands on the same picture. One
    finalist per ratio (the first, in metadata order) is compared. Never raises.
    """
    import frame_snap
    master_obj, candidate_obj = videos
    by_ratio = {}
    for row in tied:
        by_ratio.setdefault(row["ratio"], row)
    labels = {f"{ratio.numerator}/{ratio.denominator}": row for ratio, row in by_ratio.items()}
    try:
        candidate_rate, _source = frame_snap.exact_rate(candidate_obj.video)
        if candidate_rate is None:
            raise ValueError("candidate frame rate unread")
        functions, span = {}, None
        for label, row in labels.items():
            alignment, effective = mappings[(row["ratio"], row["engine"])]
            functions[label] = _time_mapping(alignment, effective)
            detail = alignment.get("zones_detail") or []
            if detail:
                low, high = detail[0]["master_ms"][0] / 1000.0, detail[-1]["master_ms"][1] / 1000.0
                span = (low, high) if span is None else (max(span[0], low), min(span[1], high))
        rows, notes = frame_snap.ratio_picture_scores(
            master_obj.filePath, master_obj.video["StreamOrder"], candidate_obj.filePath,
            candidate_obj.video["StreamOrder"], candidate_rate, functions, span or (0.0, 0.0))
        label, reason, detail = frame_snap.decide_ratio(rows, list(functions))
    except Exception as error:                                           # noqa: BLE001
        label, reason, detail, notes = None, f"raised({type(error).__name__}: {error})", {}, []
    tools.log_always(f"rate_direction: picture_tie_break {reason} ratios={list(labels)} "
                     f"chosen={label} totals={detail.get('totals')} "
                     f"winners={detail.get('winners')} valid={detail.get('valid')} "
                     f"skipped={notes} for {candidate_path}\n")
    return None if label is None else labels[label]


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
    observations = {
        "verdict": alignment.get("verdict"),
        "coverage": alignment.get("master_axis_coverage_fraction"),
        "residual_fraction": alignment.get("residual_fraction"),
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
    if ladder["is_rate_ladder"]:
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
