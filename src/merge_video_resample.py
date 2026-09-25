'''
Application d'une relation de vitesse mesuree.

`docs/AUDIO_SPEED_POLICY.MD` est une DECISION, pas une suggestion: `asetrate`,
ni `rubberband`, pour PAL comme pour NTSC -- et, depuis l'ADDENDUM 30, `atempo`
quand le bras de taux MESURE que la hauteur a ete conservee a l'origine (le
moteur qui s'aligne gagne: ids 57/293/300). Ce module ne rejuge rien; il
applique, et il ecrit ce qu'il a applique.

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


# THE TWO SPEED ENGINES (ADDENDUM 30 / 30.5): `asetrate` undoes speed AND pitch together (the
# naive PAL/NTSC speed-up); `atempo` undoes the tempo only (a candidate whose pitch was kept at
# origin -- ids 57/293/300). The rate arm measures which one aligns; nothing chooses by default.
SPEED_ENGINES = ("asetrate", "atempo")


def build_transform_chain(source_rate, speed_ratio, engine):
    """The ONE place a speed correction becomes an ffmpeg `-af` string, for both engines:
    `(chain, effective_ratio)` with `effective_ratio` a Decimal -- asetrate's is quantised by its
    integer rate (`build_speed_filter_chain`), atempo's is the ratio itself. The ffmpeg engine
    for atempo always: the Rubber Band CLI is not a filter and never runs inside a chain."""
    if isinstance(speed_ratio, Fraction):
        speed_ratio = Decimal(speed_ratio.numerator) / Decimal(speed_ratio.denominator)
    if engine == "asetrate":
        chain, effective, _intermediate, _target = build_speed_filter_chain(
            source_rate, speed_ratio)
        return chain, effective
    if engine == "atempo":
        tempo = build_tempo_filter_chain(speed_ratio, rubberband_binary="")
        return tempo["filter"], tempo["time_ratio"]
    raise resample_error(f"unknown speed engine {engine!r}: one of {SPEED_ENGINES}")


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


# ---------------------------------------------------------------------------
# THE RATE VOCABULARY (RULING_20260922_NO_BAND_ROUTING.MD, ADDENDUM 2 -- OWNER
# OVERRIDE). "Tester TOUTES les combinaisons de cadences." Since ADDENDUM 30.5
# the combinations are measured by the rate arm (`repair_orchestrator.rate_arm`:
# each ratio in both engines, re-fingerprinted and aligned); the window sweep that
# measured them here scored 1001/960 on id 101 at 0.704 against a 0.9 floor while
# the aligned whole track covers 0.9997, and left with its only caller.
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
