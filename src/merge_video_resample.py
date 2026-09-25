'''
Application d'une relation de vitesse mesuree.

`docs/AUDIO_SPEED_POLICY.MD` est une DECISION, pas une suggestion: `asetrate`,
ni `rubberband`, ni `atempo`, pour PAL comme pour NTSC. Ce module ne rejuge
rien; il applique, et il ecrit ce qu'il a applique.

CONVENTION, sous forme d'equation pour qu'elle ne se lise pas dans les deux
sens:

    speed_ratio = duree_maitre / duree_candidat

Donc PAL contre film lit 1.042709, le candidat va TROP VITE et doit etre
RALENTI: `asetrate = frequence / speed_ratio`. La faiblesse 3 du document est
exactement la: un balayage avait applique `* r` au lieu de `/ r`, atterri deux
fois plus loin, et `rubberband` avait gagne 33 fichiers sur 33 -- signe
constant, magnitudes plausibles, et rien a l'interieur de la campagne ne
pouvait le rattraper.

LE FACTEUR ECRIT DANS LE TAG EST CELUI REELLEMENT APPLIQUE. `asetrate` prend un
entier: 48000/1.042709 vaut 46033.55, donc la frequence posee est arrondie et le
facteur reellement obtenu n'est PAS celui demande. On passe par une frequence
intermediaire haute pour diviser cette erreur, on calcule le facteur effectif
exactement, et c'est lui qui part dans `VMSAM_FABRICATED`. Un tag
`resampled:1.042709` sur une piste etiree par 1.042731 est pire que pas de tag.
'''

from decimal import Decimal, getcontext
from fractions import Fraction
from os import path, remove
import statistics

import audioCorrelation
import tools
import repair_log

# `asetrate` ne prend qu'un entier. Poser la frequence cible directement a
# 48000/r arrondirait a 0.5 Hz pres, soit 1.1e-5 en relatif -- 15 ms de derive
# sur un episode de 1420 s. En remontant d'abord a 8x, la meme demi-unite pese
# huit fois moins. Le plafond existe parce qu'une frequence absurde coute du
# temps de calcul sans rien rendre.
speed_intermediate_factor = 8
speed_intermediate_rate_max = 768000


class resample_error(Exception):
    '''La relation de vitesse ne peut pas etre appliquee. Refus explicite.'''
    pass


def get_intermediate_rate(source_rate):
    return min(int(source_rate) * speed_intermediate_factor,
               speed_intermediate_rate_max)


def build_speed_filter_chain(source_rate, speed_ratio):
    '''Renvoie (chaine de filtres, facteur effectif, frequence intermediaire,
    frequence posee).

    Le facteur effectif est `intermediaire / round(intermediaire / ratio)`,
    exactement -- pas le ratio demande.
    '''
    ratio = Decimal(str(speed_ratio))
    if ratio <= 0:
        raise resample_error(f"speed ratio {ratio} is not positive")
    if ratio == 1:
        raise resample_error("speed ratio is exactly 1: nothing to apply")

    source_rate = int(source_rate)
    intermediate = get_intermediate_rate(source_rate)
    getcontext().prec = 28
    target = int((Decimal(intermediate) / ratio).to_integral_value(rounding="ROUND_HALF_EVEN"))
    if target < 1:
        raise resample_error(
            f"speed ratio {ratio} would set the sample rate to {target}")
    effective = Decimal(intermediate) / Decimal(target)
    # SOXR VERY HIGH QUALITY ON BOTH RESAMPLING STEPS (RULING_20260922 ADDENDUM
    # 18 and 21.5). `asetrate` only re-labels the rate -- it is lossless; the two
    # `aresample` steps are the only real sinc interpolations in the chain, so
    # they are where the resampler's transparency is decided. precision=28 is
    # soxr's "Very High Quality" tier, cutoff=0.91 its own 0 dB point (swr's
    # default is a 6 dB point at 0.97). ffmpeg here is built --enable-libsoxr
    # (`ffmpeg -buildconf`, 2026-09-24). The effective factor does not change:
    # it is fixed by the two INTEGER rates, not by the interpolator.
    chain = (f"aresample={intermediate}:{SOXR_RESAMPLER_OPTIONS},"
             f"asetrate={target},"
             f"aresample={source_rate}:{SOXR_RESAMPLER_OPTIONS}")
    return chain, effective, intermediate, target


SOXR_RESAMPLER_OPTIONS = "resampler=soxr:precision=28:cutoff=0.91"

# THE INVERTING CASE'S ENGINES, in preference order (ADDENDUM 18: ffmpeg's
# `rubberband` FILTER is R2 real-time, never R3 -- R3 exists only as the CLI
# from `rubberband-cli`; the container image does not carry it yet, owner
# decision pending, so ffmpeg `atempo` is the fallback that always exists).
RUBBERBAND_R3_BINARIES = ("rubberband-r3", "rubberband")


def build_tempo_filter_chain(speed_ratio, rubberband_binary=None):
    '''THE INVERTING CASE'S CORRECTION: tempo only, pitch untouched.

    Same convention as `build_speed_filter_chain` (speed_ratio = duree_maitre /
    duree_candidat), so the candidate's duration is multiplied by `speed_ratio`
    and its tempo by `1 / speed_ratio`. A candidate whose pitch was ALREADY
    corrected at the source (duration moved, pitch did not -- ADDENDUM 19(f))
    must not go through `asetrate`, which would shift a correct pitch.

    Returns a dict, never None:
      engine        "rubberband_r3" when an R3-capable binary is on PATH (or
                    passed), else "ffmpeg_atempo".
      filter        the ffmpeg `-af` string (`atempo=<1/ratio>`), for the
                    ffmpeg engine; None for rubberband.
      argv          for rubberband: the argv with "{in}" / "{out}" placeholders
                    (`-3` selects the R3 "finer" engine, `-t` is the TIME
                    ratio = `speed_ratio`); None for ffmpeg.
      tempo         1 / speed_ratio as a Decimal (the atempo coefficient).
      time_ratio    speed_ratio as a Decimal.
    '''
    import shutil
    ratio = Decimal(str(speed_ratio)) if not isinstance(speed_ratio, Fraction) else \
        Decimal(speed_ratio.numerator) / Decimal(speed_ratio.denominator)
    if ratio <= 0:
        raise resample_error(f"speed ratio {ratio} is not positive")
    if ratio == 1:
        raise resample_error("speed ratio is exactly 1: nothing to apply")
    getcontext().prec = 28
    tempo = Decimal(1) / ratio
    binary = rubberband_binary
    if binary is None:
        for name in RUBBERBAND_R3_BINARIES:
            binary = shutil.which(name)
            if binary:
                break
    if binary:
        return {"engine": "rubberband_r3", "filter": None,
                "argv": [binary, "-3", "-t", f"{ratio:.12f}", "{in}", "{out}"],
                "tempo": tempo, "time_ratio": ratio}
    # atempo takes a double in [0.5, 100]; every named broadcast ratio is well
    # inside it. Twelve decimals are far below the aligner's resolution.
    return {"engine": "ffmpeg_atempo", "filter": f"atempo={tempo:.12f}",
            "argv": None, "tempo": tempo, "time_ratio": ratio}


def format_factor(effective_ratio, digits=6):
    '''La precision ecrite dans le tag. On n'ecrit pas plus de chiffres qu'on
    n'en tient: le facteur effectif est exact en tant que rapport d'entiers,
    mais le tag est une chaine et six decimales suffisent a le distinguer de
    tout autre facteur plausible du corpus.
    '''
    quantum = Decimal(1).scaleb(-digits)
    return str(Decimal(str(effective_ratio)).quantize(quantum))


def retime_subtitle_events_by_ratio(subtitles, speed_ratio):
    '''Les sous-titres subissent le MEME coefficient que l'audio.

    `CAMPAIGN.MD` et le brief sont explicites: on prend les deux. Une piste
    audio recalee et des sous-titres laisses en place produisent un fichier qui
    a l'air correct et qui derive d'une seconde par demi-heure.
    '''
    ratio = Decimal(str(speed_ratio))
    for event in subtitles.events:
        event.start = int((Decimal(event.start) * ratio).to_integral_value())
        event.end = int((Decimal(event.end) * ratio).to_integral_value())
    return subtitles


'''
STEP 2 OF THE OWNER'S PIPELINE -- "Test Reechantillonnage (Fidelite > 0,90)"
(BRIEF.md, `CAMPAIGNS/02-dev-chimeric-resample/tools/RULINGS_IN_FORCE.md`,
row `PIPELINE_CANONICAL`, 2026-09-21). Everything below answers Q3 of that
ruling and gates the hand-off from "monotone drift suspected" (Stage 1,
`change_point_locator.py`, NOT this file) to "speed relation confirmed,
produce a plan" (`merge_video_repair.py`).

MEASURED, BEFORE WRITING A LINE OF THIS: `change_point_locator.MIN_MEDIAN_FIDELITY`
(0.70) and `RESAMPLE_FIDELITY_FLOOR` (0.90, below) are TWO GATES ON TWO
DIFFERENT QUANTITIES, not the same dial at two settings:

    MIN_MEDIAN_FIDELITY   Stage 1. Median chromaprint fidelity of the
                          ORIGINAL, UNCORRECTED pairing -- "is there enough
                          raw correlation to trust an offset measurement at
                          all". Never sees a resampled file.
    RESAMPLE_FIDELITY_FLOOR   Stage 2 (here). Median chromaprint fidelity,
                          SAME instrument (`audioCorrelation.correlate`,
                          SAME probe-window convention), between the master
                          and the WHOLE-TRACK RESAMPLED candidate -- "does
                          the discovered speed ratio actually explain the
                          drift". Never runs on the original pairing.

Real measurement, 2026-09-21, `VMSAM_CORPUS/corpus-B-pal-ntsc/curated-46`
(confirmed real-campaign PAL exemplar, eng track, `*.locator.flac` profile,
44100 Hz mono), n=6 probes of 60 s each, positions spread across the file,
`audioCorrelation.correlate` both times:

    BEFORE any resample                    median fidelity 0.6027 (range 0.586-0.643)
    AFTER whole-file resample at 25/23.976  median fidelity 0.9271 (range 0.756-0.970)

Confirms: (a) the pairing does sit below MIN_MEDIAN_FIDELITY before correction,
consistent with reaching Stage 1's `speed_relation_suspected` branch; (b) the
SAME instrument, applied to the corrected file, clears 0.90 by a wide margin.
Full script: `VMSAM_HELP_AI/dev-step2-resample/measure_resample_fidelity.py`
(the synthetic corpus case it also tries, `synth-drift-pal`, does NOT confirm
-- its `drift_pal.mka` arm measures a PITCH ratio of 0.99997 against master,
i.e. it was built by a pitch-PRESERVING time-stretch, not the asetrate-style
resample this project's own `docs/AUDIO_SPEED_POLICY.MD` mandates for real
PAL/NTSC content; applying `build_speed_filter_chain`'s asetrate correction to
a pitch-preserving fixture is undoing the WRONG transform and correctly fails
to raise fidelity. Reported as a corpus-fixture finding, not a code defect --
see the task file).
'''

RESAMPLE_FIDELITY_FLOOR = 0.90
# THE START VALUE, NOT A CEILING (Lead dispatch, 2026-09-21: "0,90 is a start
# value, not a fixed one, per his escalation order"). What grows across the
# ladder below is the AMOUNT OF EVIDENCE (probe count), never this floor --
# same shape as `scene_anchor.py`'s `VALIDATION_FRAME_LADDER` /
# `WINDOW_LADDER_GROWTH_FACTOR` (grow the search when a rung is AMBIGUOUS,
# not when it disagrees with a hoped-for answer). RAISED HERE FOR THE LEAD:
# a threshold-VALUE ladder (0.90 -> 0.92 -> ... at a hard ceiling) is also a
# reading of "escalation order" the dispatch's wording supports and this
# module does NOT implement it -- flagged in the task file, not guessed.

RESAMPLE_PROBE_WINDOW_SECONDS = 60.0
# SAME CONSTANT as `change_point_locator.PROBE_WINDOW_SECONDS` (60.0) --
# deliberately, so a fidelity number from Stage 1 and one from here are
# comparable without a unit conversion. Not imported: this module must not
# depend on `change_point_locator.py`, which `dev-step1-classify` owns.

RESAMPLE_PROBE_COUNT_LADDER = (5, 9, 15)
# GROWTH RULE. Rung 0 tries 5 probes; a median inside
# RESAMPLE_INCONCLUSIVE_BAND of the floor is AMBIGUOUS, not a verdict, and
# earns more probes at the NEXT rung -- probes already measured are reused,
# never re-run. A rung whose median sits CLEARLY above or below the floor
# returns immediately: an unambiguous answer does not need more evidence to
# stay true.
RESAMPLE_INCONCLUSIVE_BAND = 0.03
# HARD CEILING (Lead dispatch, 2026-09-21: "a hard ceiling and a named
# decline at the ceiling"). A pairing that is still inside the inconclusive
# band at the largest rung declines as `resample_fidelity_inconclusive`,
# never as a silent pass or a silent fail -- see `measure_fidelity_ladder`.


class resample_fidelity_error(Exception):
    '''Le test n'a pas pu tourner (extraction ou correlation en echec). PAS un
    refus: le refus est un verdict "below"/"inconclusive_at_ceiling" rendu par
    `test_speed_ratio_against_master`; ceci est "l'instrument n'a pas tourne",
    la distinction que BRIEF_COMMON regle 5 exige de ne jamais confondre.'''
    pass


def _ffprobe_duration_seconds(source_path):
    cmd = [tools.software["ffprobe"], "-v", "error", "-show_entries",
           "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
           source_path]
    tools.dev_log(f"resample: _ffprobe_duration_seconds starting "
                  f"file={source_path}\n")
    with repair_log.announced("resample", "ffprobe", source_path) as call:
        stdout, stderror, exitCode = tools.launch_cmdExt_no_test(cmd)
        call["exit"] = exitCode
    if exitCode != 0:
        raise resample_fidelity_error(
            f"ffprobe could not read a duration: exit {exitCode}: "
            f"{stderror.decode('utf-8', 'replace')}")
    try:
        return float(stdout.decode("utf-8").strip())
    except ValueError as error:
        raise resample_fidelity_error(f"ffprobe returned no readable duration: {error}")


def _extract_wav(source_path, start_seconds, window_seconds, out_path,
                  sample_rate, audio_filter=None):
    '''Une fenetre mono, au meme taux des deux cotes -- condition de
    `audioCorrelation.correlate`, qui compare deux listes fpcalc terme a terme.'''
    command = [tools.software["ffmpeg"], "-y", "-v", "error", "-nostdin",
               "-ss", f"{start_seconds:.3f}", "-t", f"{window_seconds:.3f}",
               "-i", source_path, "-vn", "-ac", "1", "-ar", str(int(sample_rate))]
    if audio_filter:
        command.extend(["-af", audio_filter])
    command.append(out_path)
    tools.dev_log(f"resample: _extract_wav starting file={source_path} "
                  f"start_seconds={start_seconds} window_seconds={window_seconds}\n")
    with repair_log.announced("resample", "ffmpeg", source_path) as call:
        tools.launch_cmdExt_with_timeout_reload(command, 1, 120)
        call["exit"] = 0


def measure_fidelity_ladder(fidelity_at, duration_seconds_value,
                             window_seconds=RESAMPLE_PROBE_WINDOW_SECONDS,
                             floor=RESAMPLE_FIDELITY_FLOOR,
                             probe_ladder=RESAMPLE_PROBE_COUNT_LADDER,
                             inconclusive_band=RESAMPLE_INCONCLUSIVE_BAND,
                             log_label="resample_ladder"):
    '''Marche l'echelle DE NOMBRE DE SONDES (jamais du seuil) jusqu'a un
    verdict net ou le plafond. `fidelity_at(start_seconds)` -> float ou None.

    Renvoie (verdict, median, n_mesure, rang, journal_des_rangs) avec verdict
    dans {"above", "below", "inconclusive_at_ceiling"}. Chaque rang est journalise
    VIA `tools.logs.append` (BRIEF_ADDENDUM.md #3, "Log all step... every
    attempt at every rung"), succes ou non -- `log_label` distingue quel dial
    (quelle hypothese de direction) a produit la ligne, jamais un nom de
    fichier ou un titre. Le rungs_log en valeur de retour reste, pour le
    plan produit; ce log est le TRACE PERSISTANTE que le retour seul n'est
    pas (mediation `tools.logs`, pas un `print`).
    '''
    measured = {}
    rungs_log = []
    span = duration_seconds_value - window_seconds - 10
    if span <= 0:
        tools.dev_log(
            f"resample gate [{log_label}]: rung 0 -- file too short for the "
            f"probe window, no rung attempted\n")
        rungs_log.append({"rung": 0, "n_requested": probe_ladder[0], "n_measured": 0,
                           "median": None, "reason": "file_too_short_for_window"})
        return "inconclusive_at_ceiling", None, 0, 0, rungs_log
    for rung_index, n in enumerate(probe_ladder):
        positions = [5 + span * i / max(1, n - 1) for i in range(n)] if n > 1 else [5 + span / 2]
        values = []
        for start in positions:
            key = round(start, 3)
            if key not in measured:
                measured[key] = fidelity_at(start)
            if measured[key] is not None:
                values.append(measured[key])
        if not values:
            tools.dev_log(
                f"resample gate [{log_label}]: rung {rung_index} "
                f"(n_requested={n}) -- every probe failed, no median\n")
            rungs_log.append({"rung": rung_index, "n_requested": n, "n_measured": 0,
                               "median": None})
            continue
        median = statistics.median(values)
        rungs_log.append({"rung": rung_index, "n_requested": n, "n_measured": len(values),
                           "median": round(median, 4)})
        if median >= floor + inconclusive_band or median <= floor - inconclusive_band:
            verdict = "above" if median >= floor else "below"
            tools.dev_log(
                f"resample gate [{log_label}]: rung {rung_index} "
                f"(n_requested={n}, n_measured={len(values)}) median={median:.4f} "
                f"floor={floor} -> {verdict}, unambiguous, ladder stops\n")
            return verdict, median, len(values), rung_index, rungs_log
        tools.dev_log(
            f"resample gate [{log_label}]: rung {rung_index} "
            f"(n_requested={n}, n_measured={len(values)}) median={median:.4f} "
            f"floor={floor} -- inside the +/-{inconclusive_band} inconclusive band, "
            f"widening to the next rung\n")
    last = rungs_log[-1]
    tools.dev_log(
        f"resample gate [{log_label}]: HARD CEILING reached at rung "
        f"{len(probe_ladder) - 1} (max {probe_ladder[-1]} probes), still "
        f"inconclusive (median={last.get('median')}) -> "
        f"inconclusive_at_ceiling, named decline\n")
    return "inconclusive_at_ceiling", last.get("median"), last.get("n_measured", 0), \
        len(probe_ladder) - 1, rungs_log


# ---------------------------------------------------------------------------
# THE RATE SWEEP (RULING_20260922_NO_BAND_ROUTING.MD, ADDENDUM 2 -- OWNER
# OVERRIDE). "Tester TOUTES les combinaisons de cadences."
#
# WHY A SWEEP BEATS A DERIVED FACTOR, in the owner's own terms: a derived
# factor is one hypothesis and it can be derived wrong, while the ladder is a
# MEASUREMENT that can be run against every hypothesis the world actually
# contains. The rate set below is small and closed, so "every combination" is
# sixteen ratios, not an open search -- and sixteen measured answers are
# cheaper to trust than one inferred one.
#
# AND IT SETTLES SOMETHING NO TOLERANCE CAN. Three members of the vocabulary
# sit within 0.1 % of each other (800/1001 = 0.799201, 4/5 = 0.800000,
# 1001/1250 = 0.800800), as do 25/24 and 1001/960. No snap window can
# separate those by arithmetic -- but the fidelity ladder separates them by
# measurement, which is the whole point of running it on each.
# ---------------------------------------------------------------------------

BROADCAST_RATE_SET = (Fraction(24000, 1001),   # 23.976
                      Fraction(24),
                      Fraction(25),
                      Fraction(30000, 1001),   # 29.97
                      Fraction(30))


def build_rate_ratio_vocabulary(rate_set=BROADCAST_RATE_SET):
    '''Every ordered pair's ratio over `rate_set`, deduplicated, closed under
    reciprocal, with 1 excluded. Exact `Fraction`s throughout.

    GENERATED, NOT TABULATED. The addendum lists the sixteen it expects
    (1001/1000, 1001/960, 1001/800, 5/4, 1250/1001, 1200/1001, 6/5, 25/24 and
    the inverses) and this function reproduces exactly that set from R -- a
    hand-written table of the same sixteen would be a second definition of
    one fact, and the rate SET is the thing a future owner edits.

    Ordered pairs already generate both directions, so the explicit
    reciprocal closure below is redundant TODAY. It stays because it is only
    redundant while the set is used symmetrically, and a ruling that adds a
    rate only on one side should not silently lose its inverse.
    '''
    vocabulary = set()
    for numerator in rate_set:
        for denominator in rate_set:
            if numerator == denominator:
                continue
            ratio = Fraction(numerator, denominator)
            if ratio == 1:
                continue
            vocabulary.add(ratio)
            vocabulary.add(1 / ratio)
    return tuple(sorted(vocabulary))


def _probe_fidelity_at_ratio(master_path, candidate_path, start_seconds,
                             window_seconds, work_dir, sample_rate, ratio,
                             chain, tag, master_cache):
    '''One probe of the master against the candidate CORRECTED BY `ratio`,
    without resampling the candidate's whole track.

    THE ANCHOR IS CORRECTED ARITHMETICALLY, WHICH IS THE ONLY REASON THIS IS
    ALLOWED TO BE A WINDOW. `_resample_whole_track`'s docstring records the
    defect that forced whole-track passes in the first place: applying the
    filter to a window extracted at the SAME absolute time stretches the
    window's CONTENT while leaving its START where it was, so the two sides
    drift apart as soon as the relation bites. That is a property of taking
    the window at `start_seconds` on both sides -- not of windows.

    With `ratio = master_span / candidate_span`, master instant `t` is
    candidate instant `t / ratio`, and a candidate segment of length
    `window_seconds / ratio` becomes exactly `window_seconds` once stretched.
    So this extracts the candidate at `start_seconds / ratio` for
    `window_seconds / ratio`, applies the SAME `build_speed_filter_chain` the
    whole-track path uses, and compares against the master window at
    `start_seconds`. Both the anchor and the length are corrected, so the
    grids stay aligned -- which the naive version could not do.

    THE MASTER WINDOW IS EXTRACTED ONCE AND REUSED ACROSS EVERY FACTOR
    (`master_cache`), which is the reuse the owner's cost note asks for: the
    master side does not depend on the hypothesis, so extracting it sixteen
    times would be sixteen times the same file.

    `None` on any failure -- BLANK LAW, same as `_probe_pair_fidelity`.
    '''
    cache_key = round(start_seconds, 3)
    master_wav = master_cache.get(cache_key)
    candidate_wav = path.join(work_dir, f"rs_c_{tag}.wav")
    try:
        if master_wav is None:
            master_wav = path.join(work_dir, f"rs_m_{cache_key}.wav")
            _extract_wav(master_path, start_seconds, window_seconds,
                         master_wav, sample_rate)
            master_cache[cache_key] = master_wav
        candidate_start = start_seconds / float(ratio)
        candidate_length = window_seconds / float(ratio)
        _extract_wav(candidate_path, candidate_start, candidate_length,
                     candidate_wav, sample_rate, audio_filter=chain)
        tools.dev_log(f"resample: _probe_fidelity_at_ratio calling "
                      f"audioCorrelation.correlate tag={tag} ratio={ratio} "
                      f"master_wav={master_wav} candidate_wav={candidate_wav}\n")
        with repair_log.announced("resample", "fpcalc", candidate_wav) as call:
            fidelity, points, delay_ms = audioCorrelation.correlate(
                master_wav, candidate_wav, window_seconds)
            call["exit"] = 0
        return fidelity
    except Exception as error:                          # noqa: BLE001
        # ONE PROBE OF ONE HYPOTHESIS, never the sweep. Same rule and same
        # scope as `_probe_pair_fidelity` above: a failure here is a
        # measurement about this factor at this position, not a reason to
        # take down the fifteen other factors or the merge.
        tools.dev_log(f"resample: sweep probe at {start_seconds:.1f}s "
                      f"ratio={ratio} failed: {type(error).__name__}\n")
        return None
    finally:
        try:
            remove(candidate_wav)
        except OSError:
            pass


def sweep_rate_ratios(master_path, candidate_path, sample_rate, work_dir,
                      vocabulary=None,
                      window_seconds=RESAMPLE_PROBE_WINDOW_SECONDS,
                      floor=RESAMPLE_FIDELITY_FLOOR):
    '''TEST EVERY RATE COMBINATION, let the best median fidelity win.

    Owner's design (ADDENDUM 2), and the decision rule is his, verbatim in
    substance:

      * each factor in the vocabulary is validated by the SAME
        `measure_fidelity_ladder`, against the SAME unchanged floor (0.90);
      * the factor with the BEST median among those at or above the floor
        WINS;
      * SEVERAL at or above the floor is an ANOMALY -- "ne devrait pas
        arriver" -- logged LOUDLY through `tools.log_always` with every
        passing factor and its median, and the best is still taken;
      * NONE at or above the floor means there is no rate leg at all.

    WHY "SEVERAL PASS" IS LOGGED RATHER THAN REFUSED. It is the owner's
    instruction, and it is also the right shape: two factors clearing 0.90
    means either the pair is nearly self-similar under two transforms (a real
    fact about the media, worth seeing) or the ladder is being fooled (a real
    fact about the instrument, worth seeing MORE). Refusing silently would
    destroy the evidence either way; this is the one outcome that must never
    be quiet.

    Returns the SAME dict shape `test_speed_ratio_against_master` returns --
    `verdict`/`ratio`/`median_fidelity`/`margin`/`cause`/`hypotheses` -- so
    every existing reader (`merge_video_repair.describe_resample_decline`,
    the plan's `resample_gate`) keeps working unchanged, plus `passing` and
    `vocabulary_size` which only this function can report.
    '''
    if vocabulary is None:
        vocabulary = build_rate_ratio_vocabulary()
    tools.dev_log(f"resample: sweep_rate_ratios starting master={master_path} "
                  f"candidate={candidate_path} factors={len(vocabulary)} "
                  f"floor={floor}\n")
    try:
        master_duration = _ffprobe_duration_seconds(master_path)
        candidate_duration = _ffprobe_duration_seconds(candidate_path)
    except resample_fidelity_error as error:
        return {"verdict": "declined", "ratio": None, "median_fidelity": None,
                "margin": None, "cause": "resample_master_unmeasurable",
                "reason": str(error), "hypotheses": {}, "passing": [],
                "vocabulary_size": len(vocabulary)}

    master_cache = {}
    results = {}
    try:
        for ratio in vocabulary:
            name = f"{ratio.numerator}/{ratio.denominator}"
            # A `Fraction` IS NOT ACCEPTED BY `build_speed_filter_chain`, and
            # the failure is not a polite one. That function opens with
            # `Decimal(str(speed_ratio))`, and `str(Fraction(1001, 1000))` is
            # `"1001/1000"`, which `Decimal` rejects with
            # `decimal.InvalidOperation` -- NOT a `resample_error`, so it
            # would sail past the handler below and take the whole sweep down
            # on its first factor. Measured by running this sweep locally
            # before any real pair went near it. Converted here, once, where
            # the exact rational becomes a coefficient.
            ratio_decimal = Decimal(ratio.numerator) / Decimal(ratio.denominator)
            try:
                chain, effective, _, _ = build_speed_filter_chain(
                    sample_rate, ratio_decimal)
            except resample_error as error:
                results[name] = {"verdict": "unmeasurable", "median": None,
                                 "reason": f"could not build the filter: {error}"}
                continue
            # STAY INSIDE BOTH FILES. A probe at master `t` reads the
            # candidate at `t / ratio`, so the usable master span is bounded
            # by what the candidate can supply -- never by the master alone.
            # Computed per factor because it depends on the factor.
            usable_span = min(master_duration,
                              candidate_duration * float(ratio))

            def fidelity_at(start, ratio=ratio, chain=chain, name=name):
                return _probe_fidelity_at_ratio(
                    master_path, candidate_path, start, window_seconds,
                    work_dir, sample_rate, ratio, chain,
                    tag=f"{name.replace('/', '_')}_{round(start)}",
                    master_cache=master_cache)

            verdict, median, n_used, rung, rungs_log = measure_fidelity_ladder(
                fidelity_at, usable_span, window_seconds, floor=floor,
                log_label=f"sweep {name}")
            results[name] = {"verdict": verdict, "median": median,
                             "n_used": n_used, "rung": rung, "rungs": rungs_log,
                             "effective_ratio": str(effective),
                             "requested_ratio": name,
                             "usable_span_seconds": round(usable_span, 3)}
    finally:
        for cached in master_cache.values():
            try:
                remove(cached)
            except OSError:
                pass

    passing = {name: r for name, r in results.items() if r["verdict"] == "above"}
    if not passing:
        # SAME THREE-WAY SPLIT as `test_speed_ratio_against_master`, and for
        # the same measured reason: "the instrument never ran" and "the
        # instrument ran and said no" are different answers (BRIEF_COMMON
        # rule 5) and one token cannot carry both.
        if any(r["verdict"] == "inconclusive_at_ceiling" for r in results.values()):
            cause = "resample_fidelity_inconclusive"
        elif all(r["verdict"] == "unmeasurable" for r in results.values()):
            cause = "resample_fidelity_unmeasurable"
        else:
            cause = "resample_fidelity_below_floor"
        tools.dev_log(f"resample: sweep_rate_ratios -- no factor of "
                      f"{len(vocabulary)} reached the floor {floor}; "
                      f"cause={cause}\n")
        return {"verdict": "declined", "ratio": None, "median_fidelity": None,
                "margin": None, "cause": cause, "hypotheses": results,
                "passing": [], "vocabulary_size": len(vocabulary)}

    ordered = sorted(passing.items(), key=lambda item: item[1]["median"],
                     reverse=True)
    best_name, best = ordered[0]
    winner = Fraction(best_name)
    if len(ordered) > 1:
        margin = best["median"] - ordered[1][1]["median"]
    else:
        margin = best["median"] - floor

    # WHICH CO-PASSES ARE EXPECTED AND WHICH ARE NOT (ADDENDUM 3, the
    # Architect's ruling on a finding this sweep produced on its first run).
    #
    # MEASURED, AND IT IS ARITHMETIC RATHER THAN A SURPRISE: a factor and ITS
    # OWN RECIPROCAL both clear the floor whenever the drift across one probe
    # window is under half the chromaprint quantum. At the 0.1% family that is
    # 60 ms of drift across a 60 s window against a 124-129 ms point -- so the
    # wrong direction still lines up, and BOTH directions read high. Calling
    # that an anomaly would fire the loud line on the single most ordinary
    # case this sweep has, and a warning that cries on the normal case is a
    # warning nobody reads.
    #
    # THE MARGIN IS WHAT DECIDES THERE, and it does so correctly -- measured
    # 0.0559 between 1000/1001 and 1001/1000 on a real pair, with the right
    # one on top. So that case gets ONE QUIET LINE carrying both medians and
    # the margin, which is the evidence a reader needs and no alarm.
    #
    # THE OWNER'S "ne devrait pas arriver" IS PRESERVED FOR WHAT IT MEANT:
    # two UNRELATED rationals fitting the same audio. That is what stays loud.
    reciprocal_name = f"{winner.denominator}/{winner.numerator}"
    unexpected = [(name, r) for name, r in ordered
                  if name != best_name and name != reciprocal_name]
    if unexpected:
        # LOUD. `log_always`, not `dev_log`: a line that only exists when a
        # flag is on is a line that does not exist on the run that mattered.
        tools.log_always(
            "resample: SWEEP ANOMALY -- a rate factor that is neither the "
            f"winner nor its reciprocal cleared the fidelity floor {floor}, "
            f"which should not happen: "
            + " ".join(f"{name}(median={r['median']})" for name, r in ordered)
            + f" -- unexpected: "
            + " ".join(f"{name}(median={r['median']})" for name, r in unexpected)
            + f" -- taking the best ({best_name}, median={best['median']}) "
              f"as the owner's rule directs, and recording the anomaly\n")
    elif len(ordered) > 1:
        # QUIET, AND STILL WRITTEN DOWN. Expected is not the same as
        # uninteresting: the margin is the number that did the separating and
        # a census of how big it runs is how anyone would ever learn that the
        # probe window is too short for a family.
        tools.dev_log(
            f"resample: sweep winner and its reciprocal both cleared the "
            f"floor {floor}, which is EXPECTED at this factor size (drift "
            f"across one probe window under half the chromaprint quantum): "
            + " ".join(f"{name}(median={r['median']})" for name, r in ordered)
            + f" -- the margin {round(margin, 4)} separates them and "
              f"{best_name} wins\n")
    tools.dev_log(f"resample: sweep_rate_ratios winner={best_name} "
                  f"median={best['median']} margin={round(margin, 4)} "
                  f"passing={len(ordered)}/{len(vocabulary)}\n")
    return {"verdict": "confirmed", "ratio": winner,
            "median_fidelity": best["median"], "margin": round(margin, 4),
            "cause": None, "hypotheses": results,
            "passing": [{"ratio": name, "median": r["median"]}
                        for name, r in ordered],
            "vocabulary_size": len(vocabulary)}
