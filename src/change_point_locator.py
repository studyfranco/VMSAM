# -*- coding: utf-8 -*-
"""
change_point_locator.py — where two timelines diverge, and by how much.

Measurement half of campaign 2 objective 2. Written by `vmsam-dev-1`; the only
caller is `vmsam-dev-2`'s repair module. Interface agreed before either side was
built: `VMSAM_HELP_AI/dev-2/INTERFACE_dev1_dev2.md`.

**This module reads no configuration and is conditioned on nothing but what it
measures.** Every refusal below is a measurement result — no shared language, no
duration, too few probes carrying signal, fidelity at the floor with scattered
offsets, monotone drift (a speed relation, objective 3's problem), or every
segment unusable after clamping. The owner's ruling that a repair conditioned on
a parameter is not a repair applies here by construction rather than by removal:
there was never a flag to take out.

It returns NUMBERS, never files. It never cuts, never writes a track, and never
touches `best_video.sameAudioMD5UseForCalculation`. The repair is dev-2's.

--------------------------------------------------------------------------------
THE SIGN CONVENTION, AS AN EQUATION SO IT CANNOT BE READ TWO WAYS

    candidate_time_ms = master_time_ms + candidate_offset_ms

To fill master position `m`, read the candidate at `m + candidate_offset_ms`.
A candidate missing content the master has gives a NEGATIVE offset.

Derived from `audioCorrelation`'s algebra, then measured on a constructed pair
with a known 1000 ms deletion, which read -1000. dev-2 confirmed it independently
from the opposite direction after its own control caught a sign error.

--------------------------------------------------------------------------------
WHY THIS RE-MEASURES INSTEAD OF READING `delayFirstMethodAbort`

`SPEC_ZONE_A.MD` §1 offers the recorded delays so the correlation need not be
re-run. Measured, those are RESIDUALS against a candidate that
`recreate_files_for_delay_adjuster` has already shifted by `delayUse/1000` s — an
arbitrary value, essentially never a whole multiple of the chromaprint hop
(4096/3/11025 = 123.840 ms). A fractional-hop shift misaligns the fingerprint
grids and MANUFACTURES one-point steps: three corpus files whose logs record one
measure as constant offsets when read unshifted, including one independently
verified frame-identical by eye with a duration delta of exactly 0.000 s. So every
probe here extracts both sides at the SAME absolute time and the grids stay
aligned.

--------------------------------------------------------------------------------
WHY THE COARSE SCAN IS THE FFT, AND WHY IT COVERS THE WHOLE FILE

Two defects in the first version of this module, both found by other agents
running it on real files rather than by review, drove this design:

1. IT SCANNED ONLY THE PIPELINE'S TEN WINDOWS, which start at `begin_in_second`
   (120 s) and are `2 * spacing` long. A step is visible to that geometry only if
   some window is majority-pre while the next is majority-post — and window 0 is
   the first, so nothing can be majority-pre before it. A step is therefore
   invisible unless `T > begin + L/2`: the first ~227 s of a 24-minute episode.
   Error ids 12, 13 and 108 all carry real steps there and the module returned
   "constant" on all three.

2. CHROMAPRINT'S QUANTUM HIDES SMALL STEPS. `int(lengthFile/n_items*1000)` is
   124-142 ms depending on window length. Error id 13 has a real 89 ms step at
   ~1170 s — 0.7 of a point — which no window reading can express. Position was
   never the only condition: a step must ALSO exceed roughly half a quantum.

Hence a probe grid over the WHOLE file with NO privileged region, read with the
unquantised FFT (`audioCorrelation.second_correlation`). The head was never
special; it was only the part noticed first.

Chromaprint is kept for the one thing it is better at here: a fidelity floor. A
pair with no shared content reads a flat 0.556-0.578 with offsets scattered over
421 s and the sign flipping 8 times in 9 transitions.

--------------------------------------------------------------------------------
UNITS

The FFT is unquantised, so `candidate_offset_ms` is a real measurement rather than
a multiple of anything. `candidate_offset_points` and `quantum_ms` are emitted for
cross-checking against the pipeline's own numbers, and
`abs(points * quantum_ms - candidate_offset_ms) <= quantum_ms/2` holds by
construction.

**Never report a step as a bare millisecond figure taken at some other window
length.** One physical step measured 500, 540 and 600 ms at three window lengths,
because each carries its own quantum. Established three separate times on three
files before it was believed.
"""

from os import path, remove
from statistics import median

import tools
import json
import subprocess
import audioCorrelation

# One chromaprint fingerprint item, seconds. frame 4096, hop frame/3, rate 11025.
CHROMAPRINT_HOP_SECONDS = 4096.0 / 3.0 / 11025.0

# --- coarse scan -------------------------------------------------------------
# 60 s probes correlate at 0.86-0.97 on real different-source pairs and resolve
# an 89 ms step. The step between probe starts is shorter than the probe itself,
# so no transition can fall between two probes unobserved.
PROBE_WINDOW_SECONDS = 60.0
PROBE_STEP_SECONDS = 40.0

# Probes within this of each other are one plateau. Flat regions measured a
# 16.7 ms spread across 800 s, so 50 ms sits comfortably above the noise.
PLATEAU_TOLERANCE_MS = 50.0
# The smallest step claimed. The smallest confirmed real step in the corpus is
# 89 ms (id 13 at ~1170 s).
MIN_STEP_MS = 60.0

# --- refusal thresholds ------------------------------------------------------
MIN_MEDIAN_FIDELITY = 0.70
MAX_DISTINCT_POINTS = 4
MAX_SIGN_FLIPS = 2

# --- cross-language stream pairing ---------------------------------------
# A candidate audio stream outside the measured language used to get NO entry in
# the per-stream offset table, and dev-2's assembler fell back to the measured
# language's offset for it -- silently, carrying 14-32 ms on the two files it
# measured, and 34.62 ms on a produced file. Under its 100 ms tolerance, under a
# video frame, invisible.
#
# The fix is a per-language reference: probe each candidate stream against a
# master stream OF ITS OWN LANGUAGE. Measured on 26 corpus files / 127 stream
# pairs: same-language same-track pairs score 0.944-0.990, cross-language pairs
# 0.566-0.848, and 0 of 87 cross-language pairs reach 0.85 on the MINIMUM of two
# probe positions. At a SINGLE position cross-language reaches 0.9400 -- so the
# two-position minimum is load-bearing and a single-probe bar would accept them.
#
# 0.85 IS A CHOICE INSIDE AN OVERLAP, NOT A BOUNDARY: the highest cross-language
# min-of-two is 0.8477 and the lowest genuine-looking same-label is 0.8196. Every
# measurement is therefore reported beside its verdict, so the bar can be moved
# by someone who disagrees with it. A row saying "rejected" cannot be re-judged;
# a row saying "0.8477, rejected at 0.85" can.
MIN_PAIRING_FIDELITY = 0.85
PAIRING_POSITION_FRACTIONS = (0.35, 0.65)

# --- no-signal guard ---------------------------------------------------------
# A correlation taken where there is no signal is not a measurement. dev-2 found
# a 1.4 s near-silent window returning -170.69 ms with apparent confidence; with
# signal in the window the same region reads -0.19 ms at r=0.974. Probes far
# below the file's own median energy are dropped rather than trusted.
LOW_SIGNAL_FRACTION = 0.10

# --- refinement ---------------------------------------------------------------
# A transition is bracketed by CLEAN probes only, never by bisecting straddling
# ones. `vmsam-forensic`'s standing note: a peak-picking correlator on a window
# that spans a feature boundary returns a DISPLACED peak, not a blend, and the
# displacement is arbitrary in sign and unbounded by the grid. Measured cost of
# ignoring this: a first version bisected 60 s probes under a majority model and
# placed id 108's two transitions 19 s early and 24 s late while reporting a
# 625 ms bracket — an over-claim of about 35x, against two independent
# instruments that agreed with each other.
REFINE_WINDOW_SECONDS = 8.0
REFINE_STEP_SECONDS = 4.0


def _log(message):
    if tools.dev:
        tools.logs.append(f"\t\t[change_point_locator] {message}\n")


def _start_times_ms(source_path):
    """Container `start_time` per stream, in milliseconds, keyed by stream index.

    This is the quantity `ffmpeg -ss t -i file` silently absorbs: it seeks by
    presentation timestamp, so a stream whose first packet is stamped 1.103 s is
    entered 1.103 s later than a consumer decoding it from its first sample.

    Returns {} when ffprobe is unavailable or the field is absent, and the caller
    emits None rather than a guess -- a converter with a wrong start_time is worse
    than one that knows it cannot convert.
    """
    probe = tools.software.get("ffprobe")
    if not probe:
        ffmpeg = tools.software.get("ffmpeg", "")
        probe = ffmpeg[:-6] + "ffprobe" if ffmpeg.endswith("ffmpeg") else "ffprobe"
    try:
        completed = subprocess.run(
            [probe, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index,start_time", "-of", "json", source_path],
            capture_output=True, text=True, timeout=60)
        streams = json.loads(completed.stdout).get("streams", [])
    except Exception:                                  # noqa: BLE001 — absence is a valid answer
        return {}
    out = {}
    for entry in streams:
        try:
            out[int(entry["index"])] = round(float(entry["start_time"]) * 1000.0, 3)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _extract(source_path, stream_order, start_seconds, length_seconds, out_path):
    cmd = [tools.software["ffmpeg"], "-v", "error", "-y", "-nostdin",
           "-ss", f"{start_seconds:.6f}", "-t", f"{length_seconds:.6f}",
           "-i", source_path, "-map", f"0:{stream_order}",
           "-vn", "-ac", "1", "-ar", "44100", "-acodec", "pcm_s16le", out_path]
    tools.launch_cmdExt(cmd)


def _rms(wav_path):
    """Root mean square of the PCM payload. Deliberately stdlib-only: this runs
    inside the merge and should not pull numpy in for one number."""
    try:
        with open(wav_path, "rb") as handle:
            raw = handle.read()
    except OSError:
        return 0.0
    body = raw[44:]
    count = len(body) // 2
    if count == 0:
        return 0.0
    total = 0
    for index in range(0, count * 2, 2):
        sample = body[index] | (body[index + 1] << 8)
        if sample >= 32768:
            sample -= 65536
        total += sample * sample
    return (total / count) ** 0.5


def _probe(master_path, master_stream, candidate_path, candidate_stream,
           start_seconds, window_seconds, work_dir, tag):
    """Both sides extracted at the SAME absolute time, so the fingerprint grids
    stay aligned and no fractional-hop shift is involved.

    Returns (offset_ms, fidelity, offset_points, quantum_ms, rms) or None.
    """
    master_window = path.join(work_dir, f"cpl_m_{tag}.wav")
    candidate_window = path.join(work_dir, f"cpl_c_{tag}.wav")
    try:
        _extract(master_path, master_stream, start_seconds, window_seconds, master_window)
        _extract(candidate_path, candidate_stream, start_seconds, window_seconds, candidate_window)
        signal = _rms(master_window)
        # unquantised, and therefore the authoritative offset
        which_file, seconds = audioCorrelation.second_correlation(master_window, candidate_window)
        if path.abspath(which_file) == path.abspath(master_window):
            offset_ms = -seconds * 1000.0
        elif path.abspath(which_file) == path.abspath(candidate_window):
            offset_ms = seconds * 1000.0
        else:
            return None
        # quantised, kept only to cross-check against the pipeline's own numbers
        fidelity, points, delay_ms = audioCorrelation.correlate(
            master_window, candidate_window, window_seconds)
        quantum = int(round(delay_ms / -points)) if points else None
        return offset_ms, fidelity, -points, quantum, signal
    except Exception as error:                        # noqa: BLE001 — logged, not swallowed
        _log(f"probe at {start_seconds:.1f}s failed: {error}")
        return None
    finally:
        for temporary in (master_window, candidate_window):
            try:
                remove(temporary)
            except OSError:
                pass


def _streams_for(video_obj, language):
    """EVERY stream of the language, not just the first.

    The first version read `audios[language][0]` and returned one offset for the
    language, while the repair rebuilds every stream of it. Measured on error
    id 266: the candidate carries two jpn streams **27.5 ms apart**, so one of the
    two rebuilt tracks took an offset that far wrong. dev-2's post-mux verifier
    measured the same split from the produced file — 27.8 ms — independently.
    27.5 ms is 0.66 of a frame: under the quantum the merge snaps to, under
    mkvmerge's integer milliseconds, and under the verifier's 100 ms tolerance.
    It would have shipped silently.
    """
    audios = getattr(video_obj, "audios", None)
    if not audios or language not in audios:
        return []
    return [entry["StreamOrder"] for entry in audios[language]
            if entry.get("StreamOrder") is not None]


def _all_audio_streams(video_obj):
    """EVERY audio stream with its language, not only one language's.

    `_streams_for` answers "the streams of language L". This answers "the streams",
    which is what a per-language pairing needs.
    """
    audios = getattr(video_obj, "audios", None) or {}
    out = []
    for lang, entries in audios.items():
        for entry in entries:
            order = entry.get("StreamOrder")
            if order is not None:
                out.append((order, lang))
    return sorted(out)


def _pair_candidate_streams(best_video, candidate_video, master_path, candidate_path,
                            shortest, work_dir, runs):
    """Give every candidate audio stream a master partner OF ITS OWN LANGUAGE.

    Returns (accepted, measurements). `accepted` keys only the streams whose best
    partner cleared MIN_PAIRING_FIDELITY -- a stream ABSENT from it has no
    measurable offset and its consumer must refuse rather than borrow one.
    `measurements` carries every pairing that was probed, accepted or not, with the
    fidelity the bar was applied to.

    ABSENT, NEVER ZERO: a stream that cleared no partner gets no key. Not
    `fidelity: 0.0` -- an unknown fidelity is not a fidelity of zero, and a
    placeholder here would read downstream as a measurement.

    CROSS-LANGUAGE PAIRS ARE NOT PROBED. 0 of 87 reached the bar on the minimum of
    two positions in the population this rule was measured on, so spending probes
    on them buys nothing. IF THAT EVER CHANGES THIS IS THE LINE TO CHANGE -- the
    rule is an empirical result, not a property of audio.
    """
    master_streams = _all_audio_streams(best_video)
    candidate_streams_all = _all_audio_streams(candidate_video)
    by_language = {}
    for order, lang in master_streams:
        by_language.setdefault(lang, []).append(order)

    # PROBE AT PLATEAU CENTRES, NOT AT BLIND FRACTIONS OF DURATION.
    #
    # The first version used fixed 35 % and 65 % positions. On error id 125 the 35 %
    # position landed inside a transition region, returned 0.7880 against 0.9487 at
    # the other position, and the minimum-of-two REFUSED A FILE WHOSE STREAMS MATCH --
    # the pre-change locator measured 36 probes at fid_median 0.955 on it. A window
    # crossing a change point returns a DISPLACED PEAK, and a minimum turns one
    # displaced peak into a refusal of the whole file.
    #
    # k-of-n was the obvious repair and MEASUREMENT KILLED IT: over the same 127-pair
    # population with a third position added, 2-of-3 admits SEVEN cross-language pairs
    # that 2-of-2 refuses -- all on one music-and-effects-heavy file where one candidate
    # stream scores 0.87-0.96 against SEVEN different master languages. Relaxing to
    # 2-of-3 trades one loud false refusal for seven silent false accepts, and the
    # false accept is the direction that ships a wrong offset.
    #
    #     rule      cross-language accepted    same-label accepted
    #     2-of-2         0 of 87                    26 of 40
    #     2-of-3         7 of 87                    26 of 40
    #     3-of-3         0 of 87                    25 of 40
    #
    # So the minimum stays and the POSITIONS change. `runs` is already computed by the
    # time pairing happens, and a plateau centre is inside a segment BY CONSTRUCTION --
    # straddling becomes impossible rather than merely unlikely.
    # THE TWO LARGEST plateaus by probe count, not the first two. The first version
    # of this took runs[:2] and REGRESSED error id 114: that file has four segments,
    # so the first two centres both sit near the head and neither samples the body.
    # Its primary stream scored 0.902 at the old blind positions and fell below the
    # bar at the new ones -- a fix for one file breaking another, caught by re-running
    # the same comparison rather than by reasoning about it.
    #
    # The largest plateau is the one with the most probes agreeing, so it is both the
    # furthest from any boundary and the best-evidenced place to ask whether two
    # streams are the same recording.
    positions = []
    for run in sorted(runs, key=lambda r: -len(r["members"]))[:2]:
        centre = (run["first"] + run["last"] + PROBE_WINDOW_SECONDS) / 2.0
        positions.append(max(0.0, min(centre, shortest - PROBE_WINDOW_SECONDS)))
    while len(positions) < 2:
        # A single-plateau file has no boundary to straddle, so a blind second position
        # is safe here and nowhere else.
        fraction = PAIRING_POSITION_FRACTIONS[len(positions)]
        positions.append(max(0.0, min(shortest * fraction, shortest - PROBE_WINDOW_SECONDS)))

    accepted, measurements = {}, []
    for stream, language in candidate_streams_all:
        partners = by_language.get(language, [])
        if not partners:
            _log(f"candidate stream {stream} ({language}): the master carries no "
                 f"{language} stream, so no partner exists; no entry")
            measurements.append({"candidate_stream": stream, "language": language,
                                 "master_stream": None, "fidelity": None,
                                 "accepted": False, "reason": "no master stream of this language"})
            continue
        best = None
        for master_stream in partners:
            scores = []
            for index, centre in enumerate(positions):
                probe = _probe(master_path, master_stream, candidate_path, stream,
                               centre, PROBE_WINDOW_SECONDS, work_dir,
                               f"pair{master_stream}_{stream}_{index}")
                if probe is None:
                    scores = []
                    break
                scores.append(probe[1])
            if not scores:
                continue
            # THE MINIMUM, not the mean: a pair that agrees at one position and not
            # the other is a coincidence, and cross-language pairs reach 0.94 at a
            # single position.
            score = min(scores)
            if best is None or score > best[1]:
                best = (master_stream, score)
        if best is None:
            _log(f"candidate stream {stream} ({language}): no {language} master stream "
                 f"could be probed; no entry")
            measurements.append({"candidate_stream": stream, "language": language,
                                 "master_stream": None, "fidelity": None,
                                 "accepted": False, "reason": "every probe failed"})
            continue
        record = {"candidate_stream": stream, "language": language,
                  "master_stream": best[0], "fidelity": round(float(best[1]), 4),
                  "positions": len(positions),
                  "accepted": bool(best[1] >= MIN_PAIRING_FIDELITY)}
        if not record["accepted"]:
            record["reason"] = f"below {MIN_PAIRING_FIDELITY} on the minimum of {len(positions)} positions"
            _log(f"candidate stream {stream} ({language}): best partner master "
                 f"{best[0]} at {best[1]:.4f}, below {MIN_PAIRING_FIDELITY}; no entry")
        else:
            accepted[stream] = {"master_stream": best[0],
                                "fidelity": round(float(best[1]), 4),
                                "language": language,
                                "positions": len(positions)}
        measurements.append(record)
    return accepted, measurements


def _audio_duration_seconds(video_obj, language):
    audios = getattr(video_obj, "audios", None)
    if not audios or language not in audios or not audios[language]:
        return None
    for key in ("Duration", "duration"):
        if key in audios[language][0]:
            try:
                return float(audios[language][0][key])
            except (TypeError, ValueError):
                pass
    return None


def _sign_flips(values):
    non_zero = [v for v in values if v]
    return sum(1 for i in range(len(non_zero) - 1)
               if (non_zero[i] > 0) != (non_zero[i + 1] > 0))


def _group_plateaus(samples):
    """samples: [(start_seconds, offset_ms)] in time order -> plateau runs.

    A run extends while the next probe stays within PLATEAU_TOLERANCE_MS of the
    run's running mean. Runs separated by less than MIN_STEP_MS are then merged,
    so noise does not become a change point.
    """
    runs = []
    for start_seconds, offset_ms in samples:
        if runs and abs(offset_ms - runs[-1]["mean"]) <= PLATEAU_TOLERANCE_MS:
            run = runs[-1]
            run["members"].append((start_seconds, offset_ms))
            run["mean"] = sum(m[1] for m in run["members"]) / len(run["members"])
            run["last"] = start_seconds
        else:
            runs.append({"members": [(start_seconds, offset_ms)], "mean": offset_ms,
                         "first": start_seconds, "last": start_seconds})
    merged = runs[:1]
    for run in runs[1:]:
        if abs(run["mean"] - merged[-1]["mean"]) < MIN_STEP_MS:
            previous = merged[-1]
            previous["members"].extend(run["members"])
            previous["mean"] = sum(m[1] for m in previous["members"]) / len(previous["members"])
            previous["last"] = run["last"]
        else:
            merged.append(run)
    return merged


def _bracket_transition(master_path, master_stream, candidate_path, candidate_stream,
                        region_start, region_end, before_ms, after_ms, work_dir):
    """Bracket one transition using ONLY probes that sit cleanly on one plateau.

    A short probe [p, p+w] whose offset matches `before_ms` proves T > p + w.
    One matching `after_ms` proves T < p. Anything in between spans the boundary
    and is DISCARDED rather than interpreted — that is the whole point.

    Returns (low_seconds, high_seconds, narrowed) where `narrowed` is False when
    no clean pair was found and the coarse region is returned as a bound.
    """
    # TWO CONSECUTIVE clean probes are required to set an edge. One is not
    # enough: forensic's rule says a straddling window returns a DISPLACED peak,
    # arbitrary in sign, and a displaced peak can land inside PLATEAU_TOLERANCE_MS
    # of the wrong plateau by chance. A single such probe would push `last_before`
    # past the true transition and put wrong-plateau content into a segment --
    # which is the one path by which this module can DAMAGE a track rather than
    # merely waste master content.
    samples = []
    probe_at = max(0.0, region_start)
    while probe_at <= region_end:
        result = _probe(master_path, master_stream, candidate_path, candidate_stream,
                        probe_at, REFINE_WINDOW_SECONDS, work_dir, f"r{int(probe_at * 10)}")
        if result is None:
            side = None
        elif abs(result[0] - before_ms) <= PLATEAU_TOLERANCE_MS:
            side = "before"
        elif abs(result[0] - after_ms) <= PLATEAU_TOLERANCE_MS:
            side = "after"
        else:
            side = None                      # straddling: discarded, not interpreted
        samples.append((probe_at, side))
        probe_at += REFINE_STEP_SECONDS
    last_before = None
    first_after = None
    for index in range(1, len(samples)):
        if samples[index][1] == "before" and samples[index - 1][1] == "before":
            last_before = samples[index][0]
    for index in range(len(samples) - 1):
        if samples[index][1] == "after" and samples[index + 1][1] == "after":
            first_after = samples[index][0]
            break
    if last_before is None or first_after is None:
        return region_start, region_end + PROBE_WINDOW_SECONDS, False
    # WHAT A CLEAN PROBE ACTUALLY PROVES. The earlier version claimed more: that a
    # window [p, p+w] reading the BEFORE plateau proves T > p+w. It does not. A
    # window 75 % before and 25 % after still reads the before plateau, because
    # the majority dominates the peak. Measured on error id 266: a probe
    # [624, 632] read -1989.57 cleanly with the transition at ~630 INSIDE it.
    # Combined with the mirror claim from the next probe this asserted T > 632 AND
    # T < 628, so every bracket collapsed to bound-only and all precision was lost.
    #
    # `vmsam-forensic` has the compact form, and it had the same error in its own
    # published brackets: A CLEAN READING BOUNDS THE TRANSITION TO THE FAR EDGE OF
    # THE WINDOW, NEVER TO THE NEAR EDGE.
    #   reading BEFORE at p -> if T <= p the whole window would be after -> T > p
    #   reading AFTER at q  -> if T >= q+w the whole window would be before -> T < q+w
    return last_before, first_after + REFINE_WINDOW_SECONDS, True


def locate_change_points(best_video, candidate_video, language, work_dir=None):
    """Locate where `candidate_video`'s timeline diverges from `best_video`'s.

    Returns the block described in INTERFACE_dev1_dev2.md, or **None**.

    None means *I could not measure* — never *the files are compatible*. dev-2
    maps it to `no_plan` and the refusal stands. Those are different answers and
    collapsing them is the mistake this campaign exists to avoid.
    """
    master_streams = _streams_for(best_video, language)
    candidate_streams = _streams_for(candidate_video, language)
    if not master_streams or not candidate_streams:
        _log(f"no {language} stream on one side; declining")
        return None

    master_duration = _audio_duration_seconds(best_video, language)
    candidate_duration = _audio_duration_seconds(candidate_video, language)
    if not master_duration or not candidate_duration:
        _log("audio duration unavailable; declining")
        return None

    shortest = min(master_duration, candidate_duration)
    work_dir = work_dir or tools.tmpFolder
    master_path = best_video.filePath
    candidate_path = candidate_video.filePath
    reference_stream = master_streams[0]
    reference_start_ms = _start_times_ms(master_path).get(reference_stream)
    primary_stream = candidate_streams[0]

    # --- coarse scan: the WHOLE file, no privileged region -------------------
    # The grid stops at the last probe that fits, which leaves up to
    # PROBE_STEP_SECONDS of TAIL unscanned — measured at 39 s in the worst case.
    # That is the head blind spot again at the other end, and a trimmed tail is
    # exactly a change point there. So the last probe is anchored to the END.
    # A PROBE BEFORE THE MASTER STREAM'S OWN START READS A SPURIOUS OFFSET.
    #
    # `_probe` seeks by presentation timestamp. On a stream whose first packet is
    # stamped 1.103 s, a probe at t=0 cannot return audio from t=0 -- there is none
    # -- so it returns the stream's opening against a candidate window that really
    # does start at 0, and the offset it reports is the start_time rather than the
    # relation being measured.
    #
    # Measured on error ids 144 and 375. The head probe reported +1003 and +1090 ms
    # against a body of +22 and +1014, the run splitter read that as a change point
    # at the very start, and the resulting zero-width first segment was dropped:
    #
    #     segment 0 unusable (offset 1003 ms, master [0.0,0.0]); dropped, not declining
    #
    # So the plan began at master 100000 ms instead of 0. That is not merely a lost
    # 100 s: vmsam-dev-2 emits a master-fill piece for [0, first_segment_start), and
    # that piece is cut from the master stream at source 0, so it carries the same
    # defect one level down. On id 173, whose plan does begin at 0, none of this
    # happens.
    #
    # The cure is to start the grid where the reference stream actually begins.
    # Everything before that is a region no probe can measure.
    first_measurable = max(0.0, (reference_start_ms or 0.0) / 1000.0)
    if first_measurable > 0:
        _log(f"master reference stream begins at {first_measurable * 1000:.0f} ms; "
             f"probing starts there, not at 0")
    starts = []
    start_seconds = first_measurable
    while start_seconds + PROBE_WINDOW_SECONDS <= shortest:
        starts.append(start_seconds)
        start_seconds += PROBE_STEP_SECONDS
    tail_start = shortest - PROBE_WINDOW_SECONDS
    if tail_start > 0 and (not starts or tail_start - starts[-1] > 1.0):
        starts.append(tail_start)
    raw = []
    for index, probe_start in enumerate(starts):
        result = _probe(master_path, reference_stream, candidate_path, primary_stream,
                        probe_start, PROBE_WINDOW_SECONDS, work_dir, f"s{index}")
        if result is not None:
            raw.append((probe_start, result))
    if len(raw) < 3:
        _log(f"only {len(raw)} usable probes over {shortest:.0f}s; declining")
        return None

    # --- no-signal guard -----------------------------------------------------
    median_energy = median([r[1][4] for r in raw])
    kept = [r for r in raw if r[1][4] >= LOW_SIGNAL_FRACTION * median_energy]
    dropped = len(raw) - len(kept)
    if dropped:
        _log(f"dropped {dropped} probe(s) below {LOW_SIGNAL_FRACTION:.0%} of median energy")
    if len(kept) < 3:
        _log("too few probes carry signal; declining")
        return None

    offsets = [r[1][0] for r in kept]
    fidelities = [r[1][1] for r in kept]
    quanta = [r[1][3] for r in kept if r[1][3]]
    quantum_ms = int(median(quanta)) if quanta else 125
    median_fidelity = median(fidelities)
    distinct_points = len({r[1][2] for r in kept})
    flips = _sign_flips(offsets)
    _log(f"{language}: {len(kept)} probes over [0,{shortest:.0f}]s, "
         f"fid_median={median_fidelity:.3f}, quantum={quantum_ms}ms, "
         f"distinct_points={distinct_points}, flips={flips}")

    # --- refusals, each with a measured basis --------------------------------
    if median_fidelity < MIN_MEDIAN_FIDELITY:
        monotone = (all(offsets[i] <= offsets[i + 1] for i in range(len(offsets) - 1))
                    or all(offsets[i] >= offsets[i + 1] for i in range(len(offsets) - 1)))
        if monotone:
            # Low fidelity with a monotone drift is a SPEED relation, which is
            # objective 3's problem. Refusing on fidelity alone would refuse the
            # whole family the speed repair exists for.
            _log("fidelity low but drift monotone: speed relation suspected; declining")
        else:
            _log("fidelity at the floor with scattered offsets: no shared content; declining")
        return None
    if distinct_points > MAX_DISTINCT_POINTS and flips > MAX_SIGN_FLIPS:
        _log(f"offsets scattered ({distinct_points} distinct, {flips} flips); declining")
        return None

    runs = _group_plateaus([(r[0], r[1][0]) for r in kept])

    # --- per-stream plateau offsets -----------------------------------------
    # The transitions are shared: every stream of the language shows the same
    # staircase in the same places, and only the offsets differ. So the structure
    # is measured once and the offsets once per stream, at each plateau's centre.
    # Every candidate audio stream gets a master partner of its OWN language, so
    # the table below covers tracks outside the measured language instead of
    # leaving them to be assigned another language's offset by a consumer.
    pairing, pairing_measurements = _pair_candidate_streams(
        best_video, candidate_video, master_path, candidate_path, shortest, work_dir, runs)
    # A measured-language stream missing from the pairing is missing for one of two
    # DIFFERENT reasons and they must not be collapsed. The first version of this
    # block re-added every measured-language stream unconditionally, which put a
    # stream that had FAILED THE BAR at 0.5844 back into the table with a null
    # fidelity -- and a consumer reads PRESENCE as measurable. That is
    # "absent, never zero" violated in its other form: not a fabricated value but a
    # fabricated KEY.
    probe_failed = {m["candidate_stream"] for m in pairing_measurements
                    if m.get("reason") == "every probe failed"}
    for stream in candidate_streams:
        if stream in pairing:
            continue
        if stream in probe_failed:
            # UNCHANGED: the repair rebuilds every stream of this language, so one it
            # cannot measure at all is a refusal of the whole plan.
            _log(f"stream {stream} ({language}) could not be probed at all; declining")
            return None
        # Measurable, and not the same content as any master stream of its language.
        # No entry, and NOT a decline: the plan stays valid for the streams that do
        # match, and the consumer refuses this one rather than borrowing an offset.
        _log(f"stream {stream} ({language}) is in the measured language but matched no "
             f"master stream of it above {MIN_PAIRING_FIDELITY}; no entry, not declining")
    if primary_stream not in pairing:
        # The plan's own stream failing its own bar means the offsets the segments are
        # built from were measured against a track that does not match. That is not a
        # missing entry, it is a plan with no foundation.
        _log(f"the primary stream {primary_stream} did not clear "
             f"{MIN_PAIRING_FIDELITY} against any {language} master stream; declining")
        return None
    extra_streams = [s for s in sorted(pairing) if s not in candidate_streams]
    if extra_streams:
        _log(f"pairing adds {len(extra_streams)} stream(s) outside {language}: "
             + ", ".join(f"{s}->master {pairing[s]['master_stream']} "
                         f"({pairing[s]['language']}, fid {pairing[s]['fidelity']})"
                         for s in extra_streams))

    per_stream = []
    per_stream_fidelity = []
    dropped_streams = set()
    for run in runs:
        centre = (run["first"] + run["last"] + PROBE_WINDOW_SECONDS) / 2.0
        centre = max(0.0, min(centre, shortest - PROBE_WINDOW_SECONDS))
        by_stream = {}
        fidelity_by_stream = {}
        for stream in sorted(pairing):
            if stream in dropped_streams:
                continue
            partner = pairing[stream]["master_stream"]
            if stream == primary_stream:
                by_stream[stream] = run["mean"]
                continue
            result = _probe(master_path, partner, candidate_path, stream,
                            centre, PROBE_WINDOW_SECONDS, work_dir,
                            f"p{stream}_{int(centre)}")
            if result is None:
                if stream in candidate_streams:
                    # UNCHANGED for the measured language: a stream the repair will
                    # rebuild and cannot measure is a refusal, not a gap.
                    _log(f"stream {stream} unmeasurable at {centre:.1f}s; declining")
                    return None
                # A stream outside the measured language is dropped ENTIRELY rather
                # than measured in some segments and not others: a track placed in
                # segments 0 and 2 and missing from 1 is a gap in the middle of a
                # track, not a placement.
                _log(f"stream {stream} ({pairing[stream]['language']}) unmeasurable at "
                     f"{centre:.1f}s; dropping it from the table entirely")
                dropped_streams.add(stream)
                continue
            by_stream[stream] = result[0]
            fidelity_by_stream[stream] = round(float(result[1]), 4)
        per_stream.append(by_stream)
        per_stream_fidelity.append(fidelity_by_stream)
    if dropped_streams:
        for stream in dropped_streams:
            pairing.pop(stream, None)
            for table in per_stream:
                table.pop(stream, None)
            for table in per_stream_fidelity:
                table.pop(stream, None)

    # --- transitions, bisected ----------------------------------------------
    change_points = []
    for position in range(len(runs) - 1):
        before, after = runs[position], runs[position + 1]
        low, high, narrowed = _bracket_transition(
            master_path, reference_stream, candidate_path, primary_stream,
            before["last"], after["first"], before["mean"], after["mean"], work_dir)
        step_ms = after["mean"] - before["mean"]
        change_points.append({"bracket_low_ms": round(low * 1000.0, 2),
                              "bracket_high_ms": round(high * 1000.0, 2),
                              "bracket_is_bound_only": not narrowed,
                              "step_ms": round(step_ms, 2),
                              "step_points": int(round(step_ms / quantum_ms))})

    # A bound-only bracket returns `region_end + PROBE_WINDOW_SECONDS`, which can
    # reach past the NEXT run's first probe and swallow a whole plateau. On error
    # id 266 that dropped two segments and left 26.6 % of the timeline filled from
    # the master -- waste rather than damage, but 380 s of candidate content the
    # repair could have used. Clamp each bracket so it cannot cross the next one.
    for index in range(len(change_points) - 1):
        ceiling = change_points[index + 1]["bracket_low_ms"]
        if change_points[index]["bracket_high_ms"] > ceiling:
            change_points[index]["bracket_high_ms"] = max(
                change_points[index]["bracket_low_ms"], ceiling)
            change_points[index]["bracket_clamped_to_next"] = True

    # --- segments, with the bounds clamp ------------------------------------
    # With a negative offset the first segment must not start at master 0: it
    # would read the candidate at a negative time, and dev-2's bounds check
    # refuses the whole plan. Master [0, -offset) is a LEADING GAP filled from
    # the master, which is correct rather than a compromise. Mirror at the tail.
    segments = []
    kept_runs = set()
    dropped_segments = 0
    candidate_end_ms = candidate_duration * 1000.0
    master_end_ms = round(shortest * 1000.0, 2)
    boundary = 0.0
    for position, run in enumerate(runs):
        offset_ms = run["mean"]
        # A segment can never begin before -offset, at ANY position: the
        # candidate has nothing to give there, since candidate_time =
        # master_time + offset would be negative.
        # NOT int(round(...)). A candidate offset of +0.48 ms rounds -0.48 to 0,
        # so the segment starts at master 0 and reads the candidate at -0.48 ms —
        # out of bounds by less than a millisecond, and dev-2's Decimal bounds
        # check refuses the WHOLE PLAN. A sub-millisecond offset declining a pair
        # is not a fact about the media. These offsets are unquantised real
        # measurements, so the boundary stays fractional too. Found by dev-2 on
        # error id 8, mid-sweep, rather than by me.
        start_ms = max(boundary, max(0.0, -offset_ms))
        if position < len(runs) - 1:
            change = change_points[position]
            end_ms = change["bracket_low_ms"]
            widened = change["bracket_high_ms"]
            if change["step_ms"] < 0:
                widened = max(widened, end_ms - change["step_ms"])
            change["gap_start_ms"] = round(end_ms, 2)
            change["gap_end_ms"] = round(widened, 2)
            boundary = widened
        else:
            end_ms = min(master_end_ms, candidate_end_ms - offset_ms)
        if end_ms <= start_ms:
            # This plateau lies entirely inside the leading gap, or past the
            # candidate's end. There is nothing for the repair to place, so the
            # gap simply extends to the next usable segment — which is what the
            # comment above already says a leading gap is for.
            #
            # DROP THE SEGMENT, NOT THE PAIR. Declining here threw away four
            # good plateaus on error id 266 because the first one was unusable
            # by construction. Found by vmsam-dev-2 running this on real media:
            # a guard written for a real hazard, firing correctly, with a scope
            # one case too wide — the fourth time that shape has bitten us.
            dropped_segments += 1
            _log(f"segment {position} unusable (offset {offset_ms:.0f} ms, "
                 f"master [{start_ms},{end_ms}]); dropped, not declining")
            continue
        kept_runs.add(position)
        by_stream = per_stream[position]
        fidelity_here = per_stream_fidelity[position]
        # A SEGMENT SHORTER THAN ONE PROBE WINDOW HAS NO CLEAN PROBE IN IT.
        #
        # Every window overlapping such a segment also overlaps a transition, and
        # a peak-picking correlator on a straddling window does not return a blend
        # of the two offsets — it returns a DISPLACED PEAK, arbitrary in sign and
        # unbounded by the sampling grid. So the offset below is a number this
        # instrument produced but did not measure.
        #
        # Measured on error id 266, whose first segment spans 29 s against a 60 s
        # window: an independent video instrument put its true offset at segment
        # 2's value to within 1.3 ms, meaning the reported -819.41 ms was ~168 ms
        # wrong and the change point at ~30 s did not exist at all. The repair
        # consumed that segment and its verifier could not see the error, because
        # 29 s is 2 % of the file and six spread probes never sampled it.
        #
        # FLAGGED, NOT DROPPED. Declining the pair would throw away three
        # corroborated change points to protect one bad plateau — the same scope
        # error that dropped four good segments on this very file. The caller
        # decides whether to splice a flagged segment or fill it from the master.
        span_ms = end_ms - start_ms
        unverified = span_ms < PROBE_WINDOW_SECONDS * 1000.0
        if unverified:
            _log(f"segment {position} spans {span_ms / 1000.0:.1f}s, shorter than the "
                 f"{PROBE_WINDOW_SECONDS:.0f}s probe window: no probe in it can be "
                 f"clean, so its offset is unverified")
        segments.append({
            "master_start_ms": round(start_ms, 2),
            "master_end_ms": round(end_ms, 2),
            "master_span_ms": round(span_ms, 2),
            "candidate_offset_ms": round(offset_ms, 2),
            "candidate_offset_points": int(round(offset_ms / quantum_ms)),
            "candidate_offset_ms_by_stream": {s: round(v, 2) for s, v in by_stream.items()},
            "candidate_offset_points_by_stream": {s: int(round(v / quantum_ms))
                                                  for s, v in by_stream.items()},
            # DIAGNOSTIC, NOT A GATE. The bar is applied once per file, in
            # `candidate_stream_pairing`; gating per segment would place a track in
            # segments 0 and 2 and refuse it in 1. A stream carries no key here when
            # its offset is a plateau MEAN rather than a single probe at this centre
            # -- absent, never zero.
            "candidate_offset_fidelity_by_stream": dict(fidelity_here),
            "probes_in_segment": len(run["members"]),
            "offset_unverified": unverified,
        })

    if not segments:
        _log("every segment unusable after clamping; declining")
        return None
    if dropped_segments:
        _log(f"{dropped_segments} segment(s) dropped as unusable, {len(segments)} kept")
    # A change point is only meaningful between two segments that both survived.
    change_points = [cp for index, cp in enumerate(change_points)
                     if index in kept_runs and (index + 1) in kept_runs]

    return {"kind": "constant" if len(segments) == 1 else "piecewise_constant",
            "master_path": master_path,
            "candidate_path": candidate_path,
            "language": language,
            "reference_stream": reference_stream,
            "candidate_streams": candidate_streams,
            # ONE decision per candidate stream, made once for the file. A stream
            # ABSENT here cleared no master partner at the bar and its consumer must
            # REFUSE to place it rather than borrow another stream's offset.
            "candidate_stream_pairing": pairing,
            # Every pairing that was probed, accepted or not, with the fidelity the
            # bar was applied to -- so a bar sitting inside an overlap can be moved
            # by someone who disagrees with it.
            "candidate_stream_pairing_measurements": pairing_measurements,
            "pairing_min_fidelity": MIN_PAIRING_FIDELITY,
            "quantum_ms": quantum_ms,
            "probe_window_seconds": PROBE_WINDOW_SECONDS,
            "probe_step_seconds": PROBE_STEP_SECONDS,
            # ACTUAL coverage, not the intent. vmsam-ci measured that the
            # pipeline's own geometry never samples a median 15.9 % of a file,
            # and that a file its geometry cannot see is not declined — it is
            # never presented, and it looks like a clean constant offset. This
            # module scans [0, shortest] contiguously including an end-anchored
            # tail probe, so its coverage is stated rather than assumed.
            "scanned_seconds": [round(min(starts), 3) if starts else None,
                                round(max(starts) + PROBE_WINDOW_SECONDS, 3) if starts else None],
            # The smallest step this scan can resolve. A divergence below it
            # reads as "constant" and is INVISIBLE, not absent — so a decline or
            # a constant verdict from this module carries this floor with it.
            "step_floor_ms": MIN_STEP_MS,
            "probes_used": len(kept),
            "probes_dropped_low_signal": dropped,
            "segments": segments,
            "change_points": change_points,
            "median_fidelity": round(median_fidelity, 4),
            # "constant" means NO STEP WAS SEEN by a scan that covered
            # [0, shortest] at PROBE_WINDOW_SECONDS resolution and cannot resolve
            # a step below MIN_STEP_MS. It is NOT a warrant for applying this
            # offset as a container delay.
            # THE MASTER REFERENCE STREAM'S OWN start_time. Emitted because it
            # predicts a real failure and not because of any framing question --
            # a consumer that rebuilds a track starting at PTS 0 is misaligned from
            # the master's track by exactly this, and vmsam-dev-2's four release-32
            # declines match it to under 5 ms across three distinct values:
            # 1103.4 vs 1103.0, 1103.8 vs 1103.0, 1059.4 vs 1055.0, 887.6 vs 887.0.
            #
            # An `offset_reference` key and the master-minus-candidate difference
            # were emitted here for twenty minutes and are gone: dev-2 tested PTS
            # seek against the `atrim` its assembler uses and got 0.0 ms apart on a
            # stream with a 120 ms start_time, so there was no second frame and no
            # conversion to name. A key naming a distinction that does not exist is
            # a second thing to keep true and a second thing to get wrong.
            "master_reference_start_time_ms": reference_start_ms,
            "segments_dropped_unusable": dropped_segments,
            # Surfaced at the top level so a consumer does not have to scan the
            # segment list to discover that part of the plan is unverified.
            "segments_offset_unverified": sum(1 for seg in segments
                                              if seg["offset_unverified"]),
            "constant_floor_ms": MIN_STEP_MS if len(segments) == 1 else None}
