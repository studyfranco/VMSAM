'''
rate_direction.py -- THE RATE ARM: which direction, if any, the resample goes.

RULING_20260922_ORCHESTRATOR_ARCHITECTURE.MD, ADDENDUM 15 (owner idea, measured
5/5), 19(f) and 21.4-6. Four answers, each a pure measurement or a pure
reading, none of them wired into the chain yet (see THE SEAM below):

  decide_direction          the sweep's winner, its reciprocal and 1 (plus
                            every ratio that also passed, plus any rate the
                            fast-drift reading proposes) are scored on THE
                            SAME 10 master segments; the best mean wins; 1
                            winning means NO RATE. Returns the resampled
                            candidate track of every couple at the winning
                            factor, so nothing downstream resamples again.
  fps_contradiction_warning the declared frame rates imply a direction; if it
                            and the chosen one sit on different sides of 1,
                            say so LOUDLY -- and decide nothing.
  fast_drift_signature      a PAL-sized drift (4 %) is too fast for the
                            one-quantum ladder: the aligner cuts it into
                            many-quantum steps, all of one sign, on a line.
                            That shape IS a rate; name the rates it fits.
  inverting_case            the pitch did not move although the duration did:
                            the source was pitch-corrected at origin, and the
                            correction is tempo-only (atempo / Rubber Band R3),
                            never asetrate.

WHY THE SHARED SEGMENTS DECIDE AND NOT THE SWEEP. The sweep probes each factor
on windows of its OWN (a candidate window at t/ratio), so two factors are
never compared on the same material, and a factor and its reciprocal both clear
the floor whenever the drift across one probe window is under half a quantum
(the 0.1 % family). Measured: the current sweep inverted Fallout S01E03 and
errid-202, and it cannot answer "no rate" because 1 is not in its vocabulary.
Here every finalist resamples the WHOLE candidate track, padded to the master's
length, and all are cut at the master's 10 positions -- identical windows, so
the means are comparable and 1 is a candidate like any other. Measured on the
scratch run (VMSAM_HELP_AI/architect/cases/CHAIN.md, row RATE ARM): Fallout
E01/E02/E03 -> 1001/1000, errid-70/fr -> 1001/960 over 25/24, errid-202 -> 1.

THE SEAM (the next batch wires it; repair_orchestrator.py is one-writer-owned
right now). In `repair_orchestrator.speed_factor(master_obj, candidate_obj,
language)`, after `run_speed_sweep` returned `gate`:

    factor, evidence, tracks = rate_direction.decide_direction(
        master_obj, candidate_obj, language, gate, work_dir,
        extra_ratios=drift["named_rate_candidates"] if drift["fires"] else ())
    rate_direction.fps_contradiction_warning(master_obj, candidate_obj, factor, evidence)
    return factor, gate, cause   # + evidence and tracks for chimeric (add. 21.6)

  * `factor` is an exact `Fraction`, or None for "no rate" (1 won, or nothing
    to decide). It REPLACES the sweep's winner; a disagreement is already
    logged here with both means.
  * `tracks[(master_stream_order, candidate_stream_order)]` = dict(path,
    duration_seconds, sample_rate, effective_ratio, filter) for EVERY couple
    of the language: a mono pcm_s16le WAV of that candidate stream resampled
    at `factor` over its whole length (NOT padded), at the pair's comparison
    grid rate. The caller fingerprints it with
    `audioCorrelation.calculate_fingerprints(path, length=duration_seconds)`
    and aligns it with `b2_align` like a raw track; the caller owns the files
    and deletes them. Empty when `factor` is None (the prime at factor 1 has
    the raw fingerprints already).
  * `drift` = `fast_drift_signature(alignment["zones_detail"],
    alignment["quantum_ms"])` on the prime's alignment -- its named rates go
    in as `extra_ratios` so a 4 % pair the ladder cannot see still reaches
    the finalists even when the sweep declined.
  * `inverting_case(pitch_routing_dict, factor)` is read right after
    `pitch_routing`; when it fires, `tracks` must be rebuilt with
    `filter_kind="atempo"` (decide_direction accepts it) and the delivered
    track uses `merge_video_resample.build_tempo_filter_chain`.
'''

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from fractions import Fraction
from os import path, remove
import json
import shutil
import statistics
import subprocess
import tempfile
import time

import audioCorrelation
import audio_extract
import merge_video_resample
import tools
import repair_log
import video


# ---------------------------------------------------------------------------
# NAMED CONSTANTS
# ---------------------------------------------------------------------------

# A fast drift's implied ratio must sit this close (absolute) to a named rate
# to be proposed as one (ADDENDUM 19(f): "ratio ~ named rate"). 1e-3 is the
# gap between the two closest PAL names (25/24 and 1001/960), so a reading can
# land near both -- they are then BOTH proposed and the shared segments
# separate them, which is exactly what they are for.
FAST_DRIFT_NAMED_RATE_TOLERANCE = 1e-3
# ADDENDUM 19(f), verbatim: zone fit r^2 >= 0.9999. Measured on id 57 (Lazarus
# S01E05, PAL 4.27 %): 0.9999935 over 137 zones.
FAST_DRIFT_MIN_R_SQUARED = 0.9999
# A line through a handful of zones is not evidence. Same count as the
# orchestrator's LADDER_MIN_RUNGS, for the same reason (enough steps that a
# fraction of them means something).
FAST_DRIFT_MIN_NONZERO_STEPS = 8
# Half the smallest named deviation (1001/1000): below it no named rate can
# explain the drift -- the orchestrator's RATE_LADDER_MIN_FACTOR_DEVIATION.
FAST_DRIFT_MIN_FACTOR_DEVIATION = 5e-4

# The pitch layer's own tolerance when it cannot be imported.
PITCH_TOLERANCE_FALLBACK = 0.0030

# Seconds of padding past the master's end on the scored candidate, so the last
# window never reads past a short resample (the scratch run's `m_dur + 5`).
SCORING_PAD_SECONDS = 5.0


class rate_direction_error(Exception):
    '''The instrument could not run (no stream, no duration, ffmpeg failed).
    Never a verdict about the rate.'''
    pass


def _name(ratio):
    return f"{ratio.numerator}/{ratio.denominator}"


def _as_fraction(value):
    if value is None:
        return None
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


# ---------------------------------------------------------------------------
# (1) THE DIRECTION, ON SHARED SEGMENTS
# ---------------------------------------------------------------------------

def finalists_from_sweep(sweep_gate, extra_ratios=()):
    '''Winner, its reciprocal, 1, every other ratio that passed, then the
    extras -- deduplicated, in that order. 1 is ALWAYS present (ADDENDUM 15: "le
    facteur 1 DOIT etre candidat : sans lui une paire sans taux se voit
    confirmer un taux").'''
    ordered = []

    def add(ratio):
        ratio = _as_fraction(ratio)
        if ratio is not None and ratio > 0 and ratio not in ordered:
            ordered.append(ratio)

    gate = sweep_gate or {}
    winner = gate.get("ratio") if gate.get("verdict") == "confirmed" else None
    if winner is not None:
        winner = _as_fraction(winner)
        add(winner)
        add(1 / winner)
    add(Fraction(1))
    for entry in gate.get("passing") or []:
        add(entry.get("ratio"))
    for ratio in extra_ratios or ():
        add(ratio)
        add(1 / _as_fraction(ratio))
    return ordered, winner


def _track_entry(video_obj, language, stream_order):
    for entry in (getattr(video_obj, "audios", None) or {}).get(language) or []:
        if entry.get("StreamOrder") == stream_order:
            return entry
    return None


def _ffprobe_format_duration(file_path):
    with repair_log.announced("rate_direction", "ffprobe", file_path) as call:
        completed = subprocess.run(
            [tools.software["ffprobe"], "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", file_path],
            capture_output=True, text=True, timeout=120)
        call["exit"] = completed.returncode
    return float(completed.stdout.strip())


def _track_duration(video_obj, language, stream_order):
    entry = _track_entry(video_obj, language, stream_order) or {}
    for key in ("Duration", "duration"):
        try:
            return float(entry[key])
        except (KeyError, TypeError, ValueError):
            pass
    return _ffprobe_format_duration(video_obj.filePath)


def _comparison_rate(master_obj, candidate_obj, language):
    '''The pair's comparison grid -- the orchestrator's rule, same function:
    `video.get_less_sampling_rate`, clamped to 44100 only from above.'''
    try:
        rate = int(video.get_less_sampling_rate(master_obj.audios[language],
                                                candidate_obj.audios[language]))
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"rate_direction: comparison grid underivable "
                      f"({type(error).__name__}); using 44100\n")
        rate = 44100
    return 44100 if rate > 44100 else rate


def _ffmpeg(cmd, label):
    tools.dev_log(f"rate_direction: ffmpeg {label}: {' '.join(cmd)}\n")
    source = cmd[cmd.index("-i") + 1] if "-i" in cmd else None
    with repair_log.announced("rate_direction", "ffmpeg", source) as call:
        tools.launch_cmdExt_with_timeout_reload(cmd, 1, 3600)
        call["exit"] = 0


def _decode_track(source_path, stream_order, sample_rate, out_path):
    '''One whole-track decode, mono, at the comparison grid, float so the
    resample that follows does not stack a second quantisation.'''
    _ffmpeg([tools.software["ffmpeg"], "-y", "-v", "error", "-nostdin",
             "-i", source_path, "-map", f"0:{stream_order}", "-vn", "-ac", "1",
             "-ar", str(int(sample_rate)), "-c:a", "pcm_f32le", out_path],
            f"decode stream={stream_order} file={source_path}")


def _speed_chain(ratio, sample_rate, filter_kind):
    '''(filter_string_or_None, effective_ratio_string). None at 1: no filter
    ever runs on an unchanged speed (ADDENDUM 6).'''
    if ratio == 1:
        return None, "1"
    ratio_decimal = Decimal(ratio.numerator) / Decimal(ratio.denominator)
    if filter_kind == "atempo":
        tempo = merge_video_resample.build_tempo_filter_chain(
            ratio, rubberband_binary="")          # "" = force the ffmpeg engine here
        return tempo["filter"], str(tempo["time_ratio"])
    chain, effective, _, _ = merge_video_resample.build_speed_filter_chain(
        sample_rate, ratio_decimal)
    return chain, str(effective)


def _resample_track(raw_wav, chain, sample_rate, out_path):
    cmd = [tools.software["ffmpeg"], "-y", "-v", "error", "-nostdin", "-i", raw_wav,
           "-ac", "1"]
    if chain:
        cmd += ["-af", chain]
    cmd += ["-ar", str(int(sample_rate)), "-c:a", "pcm_s16le", out_path]
    _ffmpeg(cmd, f"resample out={out_path}")


def _segment_starts(master_duration):
    '''The pipeline's own 10 windows (`generate_begin_and_length_by_segment` /
    `generate_cut_with_begin_length`), as floats: (starts, window_seconds,
    length_time).'''
    if master_duration <= 5:
        raise rate_direction_error(f"master track of {master_duration} s is too short "
                                   f"for the 10-segment test")
    begin, length_time = video.generate_begin_and_length_by_segment(master_duration)
    cuts = video.generate_cut_with_begin_length(
        begin, length_time, time.strftime('%H:%M:%S', time.gmtime(length_time * 2)))
    starts = []
    for cut in cuts:
        hms, fraction = cut[0].split(".")
        hours, minutes, seconds = (int(x) for x in hms.split(":"))
        starts.append(hours * 3600 + minutes * 60 + seconds + float("0." + fraction))
    return starts, float(length_time * 2), length_time


def _cut_segment(source_wav, start, window, final_path, pad):
    '''One normalised window, exactly as `measure_same_content` makes it
    (`video.generate_normalised_file`: extract, then EBU R128 loudnorm).'''
    codec_param = ["-c:a", "pcm_s16le", "-ac", "1"]
    tmp = final_path + ".tmp.wav"
    cmd = [tools.software["ffmpeg"], "-y", "-nostdin", "-ss", f"{start:.3f}",
           "-i", source_wav, "-vn"]
    if pad:
        # Past the end of a short resample the window is completed with
        # silence, so every finalist is scored on the same window length.
        cmd += ["-af", f"apad=whole_dur={window:.3f}"]
    cmd += codec_param + ["-t", f"{window:.3f}", tmp]
    video.generate_normalised_file(cmd, codec_param.copy(), final_path, tmp)
    return final_path


def _delay_constancy(delays):
    '''The correlate DELAY across the 10 windows: constant under the right
    ratio, sliding under the wrong one. Corroboration only, never decisive.'''
    if not delays:
        return {"distinct": 0, "spread_ms": None, "slope_ms_per_segment": None}
    xs = list(range(len(delays)))
    mean_x, mean_y = statistics.mean(xs), statistics.mean(delays)
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    slope = (sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, delays)) / ss_xx
             if ss_xx else 0.0)
    return {"distinct": len(set(delays)), "spread_ms": max(delays) - min(delays),
            "slope_ms_per_segment": round(slope, 3)}


def decide_direction(master_obj, candidate_obj, language, sweep_gate, work_dir,
                     extra_ratios=(), sample_rate=None, filter_kind="asetrate",
                     couples=None):
    '''ADDENDUM 15: the finalists are scored on the SAME 10 master segments.

    Returns `(factor_or_None, evidence, tracks)`:
      factor    exact Fraction, or None when 1 wins (no rate) or there is
                nothing to decide / the instrument could not run
                (`evidence["decision"]` says which).
      evidence  every number the decision rests on: per finalist the mean,
                median, per-segment fidelities, correlate delays and their
                constancy, the effective factor, timings; the winner, the
                runner-up, the margin, and whether the sweep was overruled.
      tracks    {(master_so, candidate_so): {path, duration_seconds,
                sample_rate, effective_ratio, filter}} -- every couple's
                candidate resampled at `factor` (see the module docstring).
    '''
    t_start = time.time()
    evidence = {"language": language, "filter_kind": filter_kind, "finalists": {},
                "decision": None, "winner": None, "runner_up": None, "margin": None,
                "sweep_winner": None, "sweep_overruled": None}
    finalists, sweep_winner = finalists_from_sweep(sweep_gate, extra_ratios)
    evidence["sweep_winner"] = None if sweep_winner is None else _name(sweep_winner)
    if len(finalists) < 2:
        evidence["decision"] = "no_rate_candidate"
        tools.logs.append(f"rate_direction: no rate candidate to test on "
                          f"{candidate_obj.filePath} (sweep verdict="
                          f"{(sweep_gate or {}).get('verdict')}, no extra ratio) -- factor None\n")
        return None, evidence, {}

    if couples is None:
        couples = [(m, c) for m in audio_extract.streams_for(master_obj, language)
                   for c in audio_extract.streams_for(candidate_obj, language)]
    if not couples:
        evidence["decision"] = "no_stream_in_language"
        return None, evidence, {}
    master_so, candidate_so = couples[0]
    rate = int(sample_rate or _comparison_rate(master_obj, candidate_obj, language))
    master_duration = _track_duration(master_obj, language, master_so)
    starts, window, length_time = _segment_starts(master_duration)
    evidence.update({"couple": [master_so, candidate_so], "sample_rate": rate,
                     "master_duration": master_duration, "segment_starts": starts,
                     "window_seconds": window})

    tools.make_dirs(work_dir)
    private = tempfile.mkdtemp(prefix="rate_dir_", dir=work_dir)
    # The resampled tracks outlive this call (the caller owns them): a
    # directory of their own per call, so two candidates sharing a work_dir
    # can never overwrite each other's files.
    tracks_dir = tempfile.mkdtemp(prefix="rate_tracks_", dir=work_dir)
    cores = max(1, int(getattr(tools, "core_to_use", 1) or 1))
    kept_tracks = {}
    try:
        # THE MASTER'S 10 SEGMENTS, EXTRACTED ONCE, reused by every finalist.
        master_raw = path.join(private, "master_raw.wav")
        candidate_raw = path.join(private, f"cand_raw_{candidate_so}.wav")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(_decode_track, master_obj.filePath, master_so, rate, master_raw),
                    pool.submit(_decode_track, candidate_obj.filePath, candidate_so, rate,
                                candidate_raw)]
            for job in jobs:
                job.result()
        with ThreadPoolExecutor(max_workers=cores) as pool:
            master_segments = list(pool.map(
                lambda item: _cut_segment(master_raw, item[1], window,
                                          path.join(private, f"m.{item[0]}.wav"), pad=False),
                enumerate(starts)))
        remove(master_raw)
        evidence["t_decode_and_master_segments"] = round(time.time() - t0, 1)

        # PHASE A -- every finalist's whole-track resample, in parallel.
        def resample(ratio):
            t = time.time()
            name = _name(ratio)
            chain, effective = _speed_chain(ratio, rate, filter_kind)
            resampled = path.join(tracks_dir, f"rate_{name.replace('/', '_')}_c{candidate_so}.wav")
            _resample_track(candidate_raw, chain, rate, resampled)
            return name, resampled, chain, effective, round(time.time() - t, 1)

        # PHASE B -- every (finalist, window) pair as one job: cut, normalise,
        # correlate against the master's window, delete. Flattened so the
        # pool stays full instead of each finalist walking its ten in series.
        def window_job(item):
            name, resampled, k = item
            fab = _cut_segment(resampled, starts[k], window,
                               path.join(private, f"f{name.replace('/', '_')}.{k}.wav"), pad=True)
            try:
                with repair_log.announced("rate_direction", "fpcalc", fab) as call:
                    measured = audioCorrelation.correlate(master_segments[k], fab,
                                                          length_time * 2)
                    call["exit"] = 0
                return name, k, measured
            finally:
                try:
                    remove(fab)
                except OSError:
                    pass

        resampled_paths, meta = {}, {}
        t_phase = time.time()
        with ThreadPoolExecutor(max_workers=min(len(finalists), cores)) as pool:
            for name, resampled, chain, effective, t_res in pool.map(resample, finalists):
                resampled_paths[name] = resampled
                meta[name] = (chain, effective, t_res)
        evidence["t_resample_phase"] = round(time.time() - t_phase, 1)
        t_phase = time.time()
        results = {name: [None] * len(starts) for name in resampled_paths}
        with ThreadPoolExecutor(max_workers=cores) as pool:
            for name, k, value in pool.map(window_job, [(name, resampled, k)
                                                        for name, resampled in resampled_paths.items()
                                                        for k in range(len(starts))]):
                results[name][k] = value
        evidence["t_window_phase"] = round(time.time() - t_phase, 1)
        for name, values in results.items():
            chain, effective, t_res = meta[name]
            fidelities = [float(v[0]) for v in values]
            delays = [v[2] for v in values]
            evidence["finalists"][name] = {
                "ratio": name, "effective_ratio": effective, "filter": chain,
                "mean": statistics.mean(fidelities),
                "median": statistics.median(fidelities),
                "per_segment": [round(x, 4) for x in fidelities],
                "delays_ms": delays, "delay_constancy": _delay_constancy(delays),
                "t_resample": t_res}

        ranked = sorted(evidence["finalists"].values(), key=lambda r: r["mean"], reverse=True)
        best, second = ranked[0], ranked[1]
        winner = Fraction(best["ratio"])
        evidence["winner"] = best["ratio"]
        evidence["runner_up"] = second["ratio"]
        evidence["margin"] = round(best["mean"] - second["mean"], 4)
        evidence["margin_over_unity"] = (None if winner == 1 else round(
            best["mean"] - evidence["finalists"]["1/1"]["mean"], 4))
        reciprocal = _name(1 / winner)
        evidence["margin_over_reciprocal"] = (
            round(best["mean"] - evidence["finalists"][reciprocal]["mean"], 4)
            if winner != 1 and reciprocal in evidence["finalists"] else None)
        evidence["sweep_overruled"] = (sweep_winner is not None and winner != sweep_winner)
        evidence["decision"] = "no_rate" if winner == 1 else "rate"

        for row in ranked:
            tools.dev_log(
                f"rate_direction: finalist {row['ratio']} mean={row['mean']:.4f} "
                f"median={row['median']:.4f} per_segment={row['per_segment']} "
                f"delays_ms={row['delays_ms']} delay_constancy={row['delay_constancy']} "
                f"effective={row['effective_ratio']} t_resample={row['t_resample']}s\n")
        tools.logs.append(
            f"rate_direction: decision={evidence['decision']} winner={best['ratio']} "
            f"(mean {best['mean']:.4f}, delays distinct="
            f"{best['delay_constancy']['distinct']}) runner_up={second['ratio']} "
            f"(mean {second['mean']:.4f}) margin={evidence['margin']} "
            f"sweep_winner={evidence['sweep_winner']}"
            + (" -- THE SHARED SEGMENTS OVERRULE THE SWEEP" if evidence["sweep_overruled"]
               else "")
            + f" finalists={[r['ratio'] + ':' + format(r['mean'], '.4f') for r in ranked]}"
            f" candidate={candidate_obj.filePath}\n")

        # THE TRACKS THE CALLER FINGERPRINTS: the winner's resample of the
        # comparison stream is already on disk; every other candidate stream of
        # the language is resampled once, here, at the same factor.
        if winner != 1:
            chain, effective = _speed_chain(winner, rate, filter_kind)
            per_stream = {candidate_so: resampled_paths.pop(best["ratio"])}
            for other in sorted({c for _, c in couples} - {candidate_so}):
                raw = path.join(private, f"cand_raw_{other}.wav")
                _decode_track(candidate_obj.filePath, other, rate, raw)
                out = path.join(tracks_dir, f"rate_{best['ratio'].replace('/', '_')}_c{other}.wav")
                _resample_track(raw, chain, rate, out)
                remove(raw)
                per_stream[other] = out
            for m_so, c_so in couples:
                kept_tracks[(m_so, c_so)] = {
                    "path": per_stream[c_so],
                    "duration_seconds": _ffprobe_format_duration(per_stream[c_so]),
                    "sample_rate": rate, "effective_ratio": effective, "filter": chain}
        for leftover in resampled_paths.values():
            try:
                remove(leftover)
            except OSError:
                pass
        if not kept_tracks:
            shutil.rmtree(tracks_dir, ignore_errors=True)
        evidence["t_wall"] = round(time.time() - t_start, 1)
        return (None if winner == 1 else winner), evidence, kept_tracks
    except Exception as error:                                           # noqa: BLE001
        # THE INSTRUMENT DID NOT RUN -- not "no rate". Named, and the caller
        # keeps the sweep's own answer (the status quo ante).
        evidence["decision"] = "unmeasured"
        evidence["error"] = f"{type(error).__name__}: {error}"
        tools.log_always(f"rate_direction: shared-segment test could not run on "
                         f"{candidate_obj.filePath}: {evidence['error']}\n")
        shutil.rmtree(tracks_dir, ignore_errors=True)
        return None, evidence, {}
    finally:
        shutil.rmtree(private, ignore_errors=True)


# ---------------------------------------------------------------------------
# (2) THE DECLARED FRAME RATES -- a warning, never a decision
# ---------------------------------------------------------------------------

def _declared_rates(file_path):
    with repair_log.announced("rate_direction", "ffprobe", file_path) as call:
        completed = subprocess.run(
            [tools.software["ffprobe"], "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate,avg_frame_rate", "-of", "json", file_path],
            capture_output=True, text=True, timeout=120)
        call["exit"] = completed.returncode
    streams = json.loads(completed.stdout or "{}").get("streams") or [{}]
    rates = {}
    for key in ("r_frame_rate", "avg_frame_rate"):
        raw = streams[0].get(key)
        try:
            value = Fraction(raw)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if value > 0:
            rates[key] = value
    return rates


def _side(value):
    return 0 if value == 1 else (1 if value > 1 else -1)


def fps_contradiction_warning(master_obj, candidate_obj, chosen, evidence=None):
    '''ADDENDUM 15: implied = candidate fps / master fps (the speed_ratio
    convention: a 25 fps candidate of a 23.976 master is 1.0427). If the chosen
    ratio (None = 1) and the implied one sit on different sides of 1, log it
    loudly with both and the means. WARN ONLY: declared rates lie (a 25 fps
    container around 23.976 content, a speed-corrected re-encode), which is why
    the measurement decides and this only flags.

    Returns a dict: fired, rates of both files, implied, chosen.'''
    report = {"fired": False, "master_rates": None, "candidate_rates": None,
              "implied": None, "chosen": None if chosen is None else _name(_as_fraction(chosen))}
    try:
        master_rates = _declared_rates(master_obj.filePath)
        candidate_rates = _declared_rates(candidate_obj.filePath)
    except Exception as error:                                           # noqa: BLE001
        report["reason"] = f"unreadable: {type(error).__name__}"
        return report
    report["master_rates"] = {k: _name(v) for k, v in master_rates.items()}
    report["candidate_rates"] = {k: _name(v) for k, v in candidate_rates.items()}
    key = next((k for k in ("r_frame_rate", "avg_frame_rate")
                if k in master_rates and k in candidate_rates), None)
    if key is None:
        report["reason"] = "no declared rate on one side"
        return report
    implied = candidate_rates[key] / master_rates[key]
    chosen_value = Fraction(1) if chosen is None else _as_fraction(chosen)
    report["implied"] = _name(implied)
    report["implied_from"] = key
    if _side(implied) != _side(chosen_value):
        report["fired"] = True
        means = ""
        if evidence and evidence.get("finalists"):
            means = " means=" + " ".join(f"{name}:{row['mean']:.4f}"
                                         for name, row in evidence["finalists"].items())
        tools.log_always(
            f"rate_direction: rate sweep direction CONTRADICTS declared frame rates -- "
            f"chosen={report['chosen'] or '1 (no rate)'} implied={report['implied']} "
            f"({key}: candidate {report['candidate_rates']} / master "
            f"{report['master_rates']}){means} -- warning only, the measurement stands; "
            f"candidate={candidate_obj.filePath}\n")
    return report


# ---------------------------------------------------------------------------
# (3) THE FAST DRIFT -- a rate the one-quantum ladder cannot see
# ---------------------------------------------------------------------------

def fast_drift_signature(zones_detail, quantum_ms):
    '''ADDENDUM 19(f): at PAL speed (4 %) the offset moves ~5 quanta per 124
    points, so the aligner never emits one-quantum rungs -- it emits steps of
    several quanta, every one of the SAME sign, and the zone offsets sit on a
    line. The orchestrator's ladder reads that as "a file with edits in it"
    (id 57: 136/136 steps negative, 128 above the resolution floor). Three
    conditions, all required:
      sign     every nonzero step between consecutive zones has one sign
      line     the zone offsets' least-squares fit reaches r^2 >= 0.9999
      named    the implied ratio lies within FAST_DRIFT_NAMED_RATE_TOLERANCE
               of a rate in the sweep's vocabulary
    The implied ratio is `1 / (1 + slope)` (offset = candidate - master point,
    speed_ratio = master / candidate duration), from the least-squares slope,
    and also from the ladder's end-to-end rise (the orchestrator's own
    reading), both reported. A named rate within tolerance of EITHER estimate is proposed.

    Returns a dict, always, with `fires` and `named_rate_candidates` (exact
    Fractions, nearest first) -- the candidates go to decide_direction as
    `extra_ratios`; the shared segments pick among them.'''
    result = {"fires": False, "reason": None, "n_zones": len(zones_detail or []),
              "n_steps_nonzero": 0, "n_positive": 0, "n_negative": 0,
              "r_squared": None, "slope_points_per_point": None,
              "implied_ratio_fit": None, "implied_ratio_rise": None,
              "named_rate_candidates": [], "named_rate_distances": {}}
    detail = zones_detail or []
    if len(detail) < 3 or not quantum_ms:
        result["reason"] = "fewer than three zones"
        return result
    offsets = [float(z["offset_points"]) for z in detail]
    steps = [b - a for a, b in zip(offsets, offsets[1:])]
    nonzero = [s for s in steps if s != 0]
    result["n_steps_nonzero"] = len(nonzero)
    result["n_positive"] = sum(1 for s in nonzero if s > 0)
    result["n_negative"] = sum(1 for s in nonzero if s < 0)
    xs = [(z["master_points"][0] + z["master_points"][1]) / 2.0 for z in detail]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(offsets) / n
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_tot = sum((y - mean_y) ** 2 for y in offsets)
    if ss_xx == 0 or ss_tot == 0:
        result["reason"] = "flat offsets: no drift"
        return result
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, offsets)) / ss_xx
    intercept = mean_y - slope * mean_x
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, offsets))
    r_squared = 1 - ss_res / ss_tot
    span = detail[-1]["master_points"][1] - detail[0]["master_points"][0]
    rise = offsets[-1] - offsets[0]
    implied_fit = 1.0 / (1.0 + slope)
    implied_rise = 1.0 / (1.0 + rise / span) if span else None
    result.update({"r_squared": r_squared, "slope_points_per_point": slope,
                   "implied_ratio_fit": round(implied_fit, 7),
                   "implied_ratio_rise": None if implied_rise is None else round(implied_rise, 7),
                   "residual_rms_ms": round((ss_res / n) ** 0.5 * quantum_ms, 3)})
    named = sorted(merge_video_resample.build_rate_ratio_vocabulary(),
                   key=lambda r: abs(float(r) - implied_fit))
    # Near EITHER estimate: on id 57 the fit reads 1.042714 (1001/960 at 5e-6)
    # and the end-to-end rise 1.042326 (25/24 at 6.6e-4) -- proposing both
    # costs one finalist and leaves the choice to the shared segments.
    estimates = [implied_fit] + ([implied_rise] if implied_rise is not None else [])
    close = [r for r in named
             if min(abs(float(r) - e) for e in estimates) <= FAST_DRIFT_NAMED_RATE_TOLERANCE]
    result["named_rate_distances"] = {_name(r): round(abs(float(r) - implied_fit), 7)
                                      for r in named[:3]}
    if len(nonzero) < FAST_DRIFT_MIN_NONZERO_STEPS:
        result["reason"] = (f"{len(nonzero)} nonzero steps, under "
                            f"{FAST_DRIFT_MIN_NONZERO_STEPS}")
    elif result["n_positive"] and result["n_negative"]:
        result["reason"] = (f"steps of both signs ({result['n_positive']} up, "
                            f"{result['n_negative']} down): edits, not a rate")
    elif r_squared < FAST_DRIFT_MIN_R_SQUARED:
        result["reason"] = f"zone fit r^2 {r_squared:.6f} under {FAST_DRIFT_MIN_R_SQUARED}"
    elif abs(implied_fit - 1.0) < FAST_DRIFT_MIN_FACTOR_DEVIATION:
        result["reason"] = f"implied ratio {implied_fit:.7f} is unity for every named rate"
    elif not close:
        result["reason"] = (f"implied ratio {implied_fit:.7f} is not within "
                            f"{FAST_DRIFT_NAMED_RATE_TOLERANCE} of a named rate "
                            f"(nearest {result['named_rate_distances']})")
    else:
        result["fires"] = True
        result["named_rate_candidates"] = close
        result["reason"] = (f"{len(nonzero)}/{len(steps)} steps one-signed, zone fit r^2 "
                            f"{r_squared:.7f}, implied ratio {implied_fit:.7f} (end-to-end "
                            f"{implied_rise:.7f}) -> named {[_name(r) for r in close]}")
    tools.dev_log(f"rate_direction: fast_drift_signature fires={result['fires']} "
                  f"{result['reason']}\n")
    return result


# ---------------------------------------------------------------------------
# (4) THE INVERTING CASE -- pitch already corrected: tempo only
# ---------------------------------------------------------------------------

def inverting_case(pitch_probe, applied_factor):
    '''ADDENDUM 19(f) / 8.1: "un candidat dont le pitch est deja corrige est le
    CAS INVERSEUR -> atempo".

    `pitch_probe`: the routing dict of `repair_orchestrator.pitch_routing`
    (`pitch_measured_ratio`, `pitch_refusal`) or a raw
    `pal_pitch_confirmer.confirm_pitch` reading (`measured_ratio`, `refusal`).

    FIRES when, all three:
      discriminating  the pitch test's band (TOL_ARM * applied) is narrower
                      than the defect |applied - 1| -- at NTSC 1.001 it never
                      is, so the inverting case is only decidable at PAL scale;
      pitch at unity  |measured - 1| <= TOL_ARM * applied;
      not the applied |measured - applied| > TOL_ARM * applied.
    A missing measurement (no peak) is "not asked", never "pitch intact".

    Returns a dict: fires, route ("atempo" | "asetrate"), the numbers, and
    `tempo_chain` (merge_video_resample.build_tempo_filter_chain) when it
    fires -- Rubber Band R3 when its CLI is installed, else ffmpeg atempo.'''
    try:
        import pal_pitch_confirmer
        tolerance = float(pal_pitch_confirmer.TOL_ARM)
    except Exception:                                                    # noqa: BLE001
        tolerance = PITCH_TOLERANCE_FALLBACK
    probe = pitch_probe or {}
    measured = probe.get("pitch_measured_ratio", probe.get("measured_ratio"))
    applied = float(_as_fraction(applied_factor)) if applied_factor is not None else 1.0
    band = tolerance * applied
    decision = {"fires": False, "route": "asetrate", "measured_ratio": measured,
                "applied_ratio": round(applied, 7), "tolerance_band": round(band, 7),
                "discriminating": band < abs(applied - 1.0),
                "pitch_refusal": probe.get("pitch_refusal", probe.get("refusal")),
                "tempo_chain": None, "reason": None}
    if measured is None:
        decision["reason"] = "no pitch measurement: not asked, asetrate stands"
    elif not decision["discriminating"]:
        decision["reason"] = (f"band {band:.6f} contains unity at applied {applied:.6f}: "
                              f"the pitch test cannot tell moved from intact here")
    elif abs(measured - 1.0) <= band and abs(measured - applied) > band:
        decision["fires"] = True
        decision["route"] = "atempo"
        tempo = merge_video_resample.build_tempo_filter_chain(_as_fraction(applied_factor))
        decision["tempo_chain"] = {k: (str(v) if isinstance(v, Decimal) else v)
                                   for k, v in tempo.items()}
        decision["reason"] = (f"pitch measured {measured} at unity (+/-{band:.6f}) while the "
                              f"duration moved by {applied:.6f}: pitch already corrected at "
                              f"origin -> tempo only ({tempo['engine']})")
    elif abs(measured - applied) <= band:
        decision["reason"] = (f"pitch moved with the speed (measured {measured}, applied "
                              f"{applied:.6f}): naive speedup, asetrate")
    else:
        decision["reason"] = (f"pitch measured {measured} is neither unity nor the applied "
                              f"{applied:.6f}: unexplained, asetrate stands (policy default)")
    tools.logs.append(f"rate_direction: inverting_case fires={decision['fires']} "
                      f"route={decision['route']} -- {decision['reason']}\n")
    return decision
