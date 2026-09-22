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
from os import path, remove
import statistics
import sys

import audioCorrelation
import tools

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
    chain = (f"aresample={intermediate},asetrate={target},"
             f"aresample={source_rate}")
    return chain, effective, intermediate, target


def get_drift_after_correction_ms(effective_ratio, requested_ratio, duration_ms):
    '''Ce que la quantisation d'`asetrate` laisse comme derive, en ms.

    Se dit avant le run, pas apres: c'est le residu que la correction NE corrige
    pas, et il doit rester petit devant un cadre video (41.7 ms a 23.976 fps).
    '''
    return abs(Decimal(str(effective_ratio)) - Decimal(str(requested_ratio))) \
        * Decimal(str(duration_ms))


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


def describe(source_rate, speed_ratio, duration_ms):
    '''Ce que l'application fera, AVANT de la faire. Sert au journal et au
    rapport: annoncer le resultat attendu avant le run est ce que
    `AGENT.MD` demande.
    '''
    chain, effective, intermediate, target = build_speed_filter_chain(
        source_rate, speed_ratio)
    return {"requested_ratio": str(Decimal(str(speed_ratio))),
            "effective_ratio": str(effective),
            "tag_factor": format_factor(effective),
            "intermediate_rate": intermediate,
            "asetrate_target": target,
            "filter": chain,
            "residual_drift_ms": str(get_drift_after_correction_ms(
                effective, speed_ratio, duration_ms))}

def iter_audio_dicts(video_obj):
    """Toutes les pistes audio de l'objet, dans l'ordre du conteneur."""
    audios = []
    for holder in (video_obj.audios, video_obj.commentary, video_obj.audiodesc):
        for language, tracks in holder.items():
            for audio in tracks:
                audios.append(audio)
    return sorted(audios, key=lambda a: int(a["StreamOrder"]))


def build_resampled_candidate(candidate_obj, speed_ratio, out_path, timeout=3600):
    '''Le candidat ENTIER avec son audio deja reechantillonne, ecrit comme fichier.

    POURQUOI CETTE FONCTION EXISTE. La mesure ne peut pas produire de plan sur
    la paire d'ORIGINE: a 4.27 % le correlateur s'effondre, ce qui est
    exactement pourquoi les pas de coupe du dossier 110 sont restes invisibles
    jusqu'au 2026-09-03. L'ordre pour un fichier a relation de vitesse est donc
    reechantillonner, PUIS localiser, PUIS assembler -- et localiser demande un
    FICHIER, pas un graphe de filtres interne. `vmsam-dev-1` prend celui-ci
    comme candidat et son module ne change pas.

    Video et sous-titres sont COPIES: seul l'audio subit la relation, et le
    reechantillonnage se fait en FLAC pour ne pas ajouter une generation de
    codec a une mesure. Les etiquettes de langue survivent par `-map 0`.

    Renvoie (chemin, facteur applique, liste des pistes vues).
    '''
    tools.dev_log(f"resample: build_resampled_candidate starting "
                  f"candidate={candidate_obj.filePath} speed_ratio={speed_ratio} "
                  f"out_path={out_path}\n")
    audios = iter_audio_dicts(candidate_obj)
    if not len(audios):
        raise resample_error("the candidate carries no audio track to resample")
    command = [tools.software["ffmpeg"], "-y", "-nostdin",
               "-analyzeduration", "1000M", "-probesize", "1000M",
               "-i", candidate_obj.filePath, "-map", "0",
               "-c", "copy", "-c:a", "flac"]
    applied = None
    seen = []
    for index, audio in enumerate(audios):
        rate = audio.get("SamplingRate")
        if rate == None:
            raise resample_error(
                f"stream {audio.get('StreamOrder')} has no sampling rate: "
                f"the applied factor could not be stated for it")
        chain, effective, _, _ = build_speed_filter_chain(rate, speed_ratio)
        # Le facteur EFFECTIF depend de la frequence source, donc deux pistes a
        # des frequences differentes ne subissent pas exactement le meme
        # coefficient. On le dit plutot que d'en écrire un seul.
        if applied == None:
            applied = effective
        elif effective != applied:
            applied = None if applied == "mixed" else "mixed"
        command.extend([f"-filter:a:{index}", chain])
        seen.append({"stream_order": int(audio["StreamOrder"]),
                     "sampling_rate": str(rate),
                     "applied_factor": format_factor(effective)})
    command.append(out_path)
    tools.dev_log(f"resample: build_resampled_candidate ffmpeg mux call "
                  f"candidate={candidate_obj.filePath} out_path={out_path}\n")
    tools.launch_cmdExt_with_timeout_reload(command, 2, timeout)
    return out_path, applied, seen


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
RESAMPLE_LADDER_MAX_RUNGS = len(RESAMPLE_PROBE_COUNT_LADDER)
# HARD CEILING (Lead dispatch, 2026-09-21: "a hard ceiling and a named
# decline at the ceiling"). A pairing that is still inside the inconclusive
# band at the largest rung declines as `resample_fidelity_inconclusive`,
# never as a silent pass or a silent fail -- see `test_speed_ratio_against_master`.


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
    stdout, stderror, exitCode = tools.launch_cmdExt_no_test(cmd)
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
    tools.launch_cmdExt_with_timeout_reload(command, 1, 120)


def _resample_whole_track(source_path, chain, sample_rate, out_path, timeout=1800):
    '''Reechantillonne LA PISTE AUDIO ENTIERE, du debut a la fin -- jamais une
    fenetre isolee. MESURE EN CONSTRUISANT CETTE FONCTION: appliquer `chain` a
    un extrait pris a `start_seconds` inchange (le meme instant des deux cotes)
    laisse l'ANCRE de la fenetre non corrigee -- seul son CONTENU est etire --
    donc la fenetre continue de comparer des instants qui ne se correspondent
    plus des que la derive depasse quelques pour-cent sur la duree du fichier.
    Reechantillonner le fichier ENTIER avant de sonder replace chaque instant
    sur la grille du maitre, ce que la fenetre seule ne peut pas faire. Cela ne
    mux PAS la video: `-vn`, audio seul, donc le cout reste celui d'une
    extraction, pas d'une copie de fichier.'''
    command = [tools.software["ffmpeg"], "-y", "-v", "error", "-nostdin",
               "-i", source_path, "-vn", "-ac", "1", "-ar", str(int(sample_rate)),
               "-af", chain, out_path]
    # THE BIG ONE (measured 2026-09-22 on one real decline: two of these,
    # 12.04 s + 11.00 s, ~84% of the whole gate's 27.31 s wall-clock -- the
    # single most expensive step in the resample fidelity gate, per
    # hypothesis, unconditionally. The case is named in the private notes
    # beside the repository). File named on the way IN, not just the way
    # out.
    tools.dev_log(f"resample: _resample_whole_track starting "
                  f"file={source_path} out_path={out_path}\n")
    tools.launch_cmdExt_with_timeout_reload(command, 1, timeout)


def _probe_pair_fidelity(master_path, candidate_path, start_seconds, window_seconds,
                          work_dir, sample_rate, tag):
    '''Meme geste que `change_point_locator._probe`: memes fenetres, meme
    correlateur (`audioCorrelation.correlate`, SEULE source de fidelite dans le
    pipeline -- P6, 2026-09-21). `None` sur un echec d'extraction ou de mesure,
    JAMAIS une fidelite fabriquee -- BLANK LAW (RULINGS_IN_FORCE.md, 2026-09-21):
    une sonde qui n'a rien mesure dit qu'elle n'a rien mesure.'''
    master_wav = path.join(work_dir, f"rf_m_{tag}.wav")
    candidate_wav = path.join(work_dir, f"rf_c_{tag}.wav")
    try:
        _extract_wav(master_path, start_seconds, window_seconds, master_wav, sample_rate)
        _extract_wav(candidate_path, start_seconds, window_seconds, candidate_wav, sample_rate)
        # IMMEDIATELY-PRE-CALL (owner's order via the Lead, 2026-09-22):
        # `audioCorrelation.correlate` lives in the FROZEN module; this open
        # caller is the only lever available on it.
        tools.dev_log(f"resample: _probe_pair_fidelity calling "
                      f"audioCorrelation.correlate tag={tag} "
                      f"master_wav={master_wav} candidate_wav={candidate_wav}\n")
        fidelity, points, delay_ms = audioCorrelation.correlate(
            master_wav, candidate_wav, window_seconds)
        return fidelity
    except Exception as error:                          # noqa: BLE001 -- see below
        # LARGE ET DELIBERE, MEME REGLE QUE change_point_locator.py:1883-1889:
        # ce site sonde un fichier reechantillonne construit a l'instant, sur
        # une hypothese qui peut etre la mauvaise moitie d'une paire (r, 1/r).
        # Une panne ici est une mesure ("cette hypothese echoue"), jamais une
        # raison de faire planter tout le declin qui l'a appelee.
        if tools.dev:
            tools.logs.append(
                f"resample fidelity probe at {start_seconds:.1f}s failed: "
                f"{type(error).__name__}\n")
        return None
    finally:
        for temporary in (master_wav, candidate_wav):
            try:
                remove(temporary)
            except OSError:
                pass


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
        if tools.dev:
            tools.logs.append(
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
            if tools.dev:
                tools.logs.append(
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
            if tools.dev:
                tools.logs.append(
                    f"resample gate [{log_label}]: rung {rung_index} "
                    f"(n_requested={n}, n_measured={len(values)}) median={median:.4f} "
                    f"floor={floor} -> {verdict}, unambiguous, ladder stops\n")
            return verdict, median, len(values), rung_index, rungs_log
        if tools.dev:
            tools.logs.append(
                f"resample gate [{log_label}]: rung {rung_index} "
                f"(n_requested={n}, n_measured={len(values)}) median={median:.4f} "
                f"floor={floor} -- inside the +/-{inconclusive_band} inconclusive band, "
                f"widening to the next rung\n")
    last = rungs_log[-1]
    if tools.dev:
        tools.logs.append(
            f"resample gate [{log_label}]: HARD CEILING reached at rung "
            f"{len(probe_ladder) - 1} (max {probe_ladder[-1]} probes), still "
            f"inconclusive (median={last.get('median')}) -> "
            f"inconclusive_at_ceiling, named decline\n")
    return "inconclusive_at_ceiling", last.get("median"), last.get("n_measured", 0), \
        len(probe_ladder) - 1, rungs_log


def test_speed_ratio_against_master(master_path, candidate_path, speed_ratio,
                                     sample_rate, work_dir,
                                     window_seconds=RESAMPLE_PROBE_WINDOW_SECONDS,
                                     timeout=1800):
    '''STEP 2's gate, entry point. `master_path`/`candidate_path` are whole
    media files -- same calling convention as `pal_pitch_confirmer.confirm_pitch`
    and `pal_rate_corrected_ncc.confirm_rate_correction`, the sibling Stage-3
    confirmers this design sits beside: ffmpeg's default audio-stream pick, not
    an explicit stream index. A caller with a specific comparison stream must
    pre-extract it to a single-stream file first, same limitation those two
    modules already carry.

    Tests BOTH `speed_ratio` and its reciprocal -- this module's own docstring
    records the measured cost of getting the direction wrong ("rubberband avait
    gagne 33 fichiers sur 33"), and `merge_video_repair.check_ratio_convention`
    exists in the sibling file for exactly this risk. SPEC_ZONE_A.MD s4f: "prendre
    la meilleure transformation au-dessus de 90% ET RAPPORTER DE COMBIEN ELLE
    L'EMPORTE" -- if both hypotheses clear the floor within `RESAMPLE_INCONCLUSIVE_BAND`
    of each other, that is INDETERMINATE, not a winner, because the margin cannot
    separate them.

    Returns a dict:
        verdict     "confirmed" | "declined" | "indeterminate"
        ratio       the winning Decimal ratio (direction included), or None
        median_fidelity   the winning hypothesis's median, or None
        margin      median_fidelity - RESAMPLE_FIDELITY_FLOOR when ONE hypothesis
                    wins; the gap between the two medians when BOTH clear the
                    floor (SPEC_ZONE_A.MD s4f's "by how much it won"); never
                    None when verdict == "confirmed" (get_speed_margin's contract,
                    merge_video_repair.py)
        cause       set when verdict != "confirmed"
        hypotheses  {"direct": {...}, "reciprocal": {...}}, full ladder log per side
    '''
    tools.dev_log(f"resample: test_speed_ratio_against_master starting "
                  f"master={master_path} candidate={candidate_path} "
                  f"speed_ratio={speed_ratio}\n")
    ratio = Decimal(str(speed_ratio))
    hypotheses_ratios = {"direct": ratio, "reciprocal": Decimal(1) / ratio}
    try:
        master_duration = _ffprobe_duration_seconds(master_path)
    except resample_fidelity_error as error:
        return {"verdict": "declined", "ratio": None, "median_fidelity": None,
                "margin": None, "cause": "resample_master_unmeasurable",
                "reason": str(error), "hypotheses": {}}

    results = {}
    for name, r in hypotheses_ratios.items():
        try:
            chain, effective, _, _ = build_speed_filter_chain(sample_rate, r)
        except resample_error as error:
            # `verdict` HERE IS NOT "below" -- see the note above `winners`
            # below. Nothing was measured against the floor; the filter
            # could not even be built, which is "the instrument did not
            # run", not "the instrument ran and refused".
            results[name] = {"verdict": "unmeasurable", "median": None,
                              "reason": f"could not build the filter: {error}"}
            continue
        resampled_path = path.join(work_dir, f"resample_fidelity_{name}.wav")
        try:
            _resample_whole_track(candidate_path, chain, sample_rate, resampled_path, timeout)
        except Exception as error:                       # noqa: BLE001 -- one hypothesis, not the caller
            results[name] = {"verdict": "unmeasurable", "median": None,
                              "reason": f"whole-track resample failed: "
                                        f"{type(error).__name__}: {error}"}
            continue

        def fidelity_at(start, resampled_path=resampled_path):
            return _probe_pair_fidelity(master_path, resampled_path, start, window_seconds,
                                         work_dir, sample_rate,
                                         tag=f"{name}_{round(start)}")

        verdict, median, n_used, rung, rungs_log = measure_fidelity_ladder(
            fidelity_at, master_duration, window_seconds, log_label=name)
        results[name] = {"verdict": verdict, "median": median, "n_used": n_used,
                          "rung": rung, "rungs": rungs_log,
                          "effective_ratio": str(effective), "requested_ratio": str(r)}
        try:
            remove(resampled_path)
        except OSError:
            pass

    winners = {name: r for name, r in results.items() if r["verdict"] == "above"}
    if not winners:
        # NEITHER HYPOTHESIS CLEARS THE FLOOR. THREE SHAPES, NOT TWO, and
        # `cause` must name which one -- measured 2026-09-22 on real wave
        # output (errids e8f7bc8a55924482, 79b17a3f34c007df): the OLD code
        # folded "could not build the filter"/"whole-track resample failed"
        # into the SAME `"below"` verdict a genuine measured-low median
        # uses, so `resample_fidelity_below_floor` fired on files where NO
        # FIDELITY WAS EVER MEASURED -- the exact defect class BRIEF_COMMON
        # rule 5 names ("I could not measure" and a conclusive negative are
        # different answers), one layer inside this module.
        #
        #   any hypothesis "inconclusive_at_ceiling" (ladder ran, stayed in
        #     the +/-band at the hard ceiling)      -> resample_fidelity_inconclusive
        #     (existing, unchanged: a real, if ambiguous, measurement)
        #   EVERY hypothesis "unmeasurable" (filter build or whole-track
        #     resample failed before any probe ran) -> resample_fidelity_unmeasurable
        #     (NEW: the instrument did not run, on either hypothesis)
        #   otherwise (at least one hypothesis reached a clean, measured
        #     "below" verdict -- a real median exists in `results[name]
        #     ["median"]`)                           -> resample_fidelity_below_floor
        #     (unchanged token, now a TRUE conclusive negative every time)
        if any(r["verdict"] == "inconclusive_at_ceiling" for r in results.values()):
            cause = "resample_fidelity_inconclusive"
        elif all(r["verdict"] == "unmeasurable" for r in results.values()):
            cause = "resample_fidelity_unmeasurable"
        else:
            cause = "resample_fidelity_below_floor"
        return {"verdict": "declined", "ratio": None, "median_fidelity": None,
                "margin": None, "cause": cause, "hypotheses": results}
    if len(winners) == 2:
        medians = {name: r["median"] for name, r in winners.items()}
        best_name = max(medians, key=medians.get)
        other_name = "reciprocal" if best_name == "direct" else "direct"
        margin = medians[best_name] - medians[other_name]
        if margin < RESAMPLE_INCONCLUSIVE_BAND:
            # SPEC_ZONE_A.MD s4f, VERBATIM: "If two hypotheses sit above the
            # bar within a margin too small to separate them, that is
            # INDETERMINATE, not PAL." Read literally for the reciprocal case.
            return {"verdict": "indeterminate", "ratio": None, "median_fidelity": None,
                    "margin": round(margin, 4), "cause": "resample_hypotheses_indistinguishable",
                    "hypotheses": results}
        return {"verdict": "confirmed", "ratio": hypotheses_ratios[best_name],
                "median_fidelity": medians[best_name], "margin": round(margin, 4),
                "cause": None, "hypotheses": results}
    only_name = next(iter(winners))
    only = winners[only_name]
    return {"verdict": "confirmed", "ratio": hypotheses_ratios[only_name],
            "median_fidelity": only["median"],
            "margin": round(only["median"] - RESAMPLE_FIDELITY_FLOOR, 4),
            "cause": None, "hypotheses": results}
