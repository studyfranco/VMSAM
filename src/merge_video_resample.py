'''
Apply a measured speed relation to an audio track and its subtitles.

Two engines: asetrate (speed and pitch together, for PAL/NTSC speed-ups) and
atempo (tempo only, when the source already kept the pitch). The caller picks
the engine; this module builds the filter and reports the factor applied.

Convention: speed_ratio = master_duration / candidate_duration. A PAL candidate
against a film master reads 1.042709 and is slowed with asetrate = rate / speed_ratio.

asetrate takes an integer rate, so the applied factor differs slightly from the
requested one; that exact effective factor is what is written to the tag.
'''

from decimal import Decimal, getcontext
from fractions import Fraction


# Upsample 8x before asetrate so its integer rounding weighs eight times less;
# the cap bounds compute cost.
speed_intermediate_factor = 8
speed_intermediate_rate_max = 768000


class resample_error(Exception):
    '''Raised when a speed relation cannot be applied.'''
    pass


def get_intermediate_rate(source_rate):
    '''Return the upsampled rate used before asetrate.'''
    return min(int(source_rate) * speed_intermediate_factor,
               speed_intermediate_rate_max)


def build_speed_filter_chain(source_rate, speed_ratio):
    '''Build the asetrate speed-and-pitch filter chain.

    Returns:
        (chain, effective, intermediate, target): the -af string, the exact applied
        factor intermediate / target as a Decimal, and the two integer rates.

    Raises:
        resample_error: ratio not positive, equal to 1, or yielding an invalid rate.
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
    # asetrate only relabels the rate; the two soxr aresample steps are the only
    # real interpolations (requires ffmpeg built with libsoxr).
    chain = (f"aresample={intermediate}:{SOXR_RESAMPLER_OPTIONS},"
             f"asetrate={target},"
             f"aresample={source_rate}:{SOXR_RESAMPLER_OPTIONS}")
    return chain, effective, intermediate, target


SOXR_RESAMPLER_OPTIONS = "resampler=soxr:precision=28:cutoff=0.91"

# Rubber Band R3 CLI names, in preference order (ffmpeg's filter is R2 only);
# ffmpeg atempo is the fallback when none is installed.
RUBBERBAND_R3_BINARIES = ("rubberband-r3", "rubberband")


def build_tempo_filter_chain(speed_ratio, rubberband_binary=None):
    '''Build a tempo-only speed correction that leaves the pitch untouched.

    Used when the source already kept the pitch, where asetrate would shift it.
    Same speed_ratio convention as build_speed_filter_chain.

    Args:
        speed_ratio: master_duration / candidate_duration.
        rubberband_binary: Rubber Band path; None searches PATH, "" forces atempo.

    Returns:
        dict with engine ("rubberband_r3" or "ffmpeg_atempo"), filter (the atempo
        -af string or None), argv (Rubber Band command with {in}/{out} placeholders
        or None), tempo (1 / speed_ratio) and time_ratio (speed_ratio), as Decimals.
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
    return {"engine": "ffmpeg_atempo", "filter": f"atempo={tempo:.12f}",
            "argv": None, "tempo": tempo, "time_ratio": ratio}


SPEED_ENGINES = ("asetrate", "atempo")


def build_transform_chain(source_rate, speed_ratio, engine):
    """Build the ffmpeg -af string for a speed correction with either engine.

    atempo always uses the ffmpeg filter, since the Rubber Band CLI cannot run in a chain.

    Returns:
        (chain, effective_ratio): the filter string and the applied ratio as a Decimal.

    Raises:
        resample_error: unknown engine or unusable ratio.
    """
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
    '''Format the effective factor for the tag with `digits` decimals.'''
    quantum = Decimal(1).scaleb(-digits)
    return str(Decimal(str(effective_ratio)).quantize(quantum))


def retime_subtitle_events_by_ratio(subtitles, speed_ratio):
    '''Scale every subtitle event's start and end by the audio's speed ratio.'''
    ratio = Decimal(str(speed_ratio))
    for event in subtitles.events:
        event.start = int((Decimal(event.start) * ratio).to_integral_value())
        event.end = int((Decimal(event.end) * ratio).to_integral_value())
    return subtitles


# Every ratio between two broadcast frame rates is measured rather than derived,
# since several sit within 0.1 % of each other and only alignment tells them apart.

BROADCAST_RATE_SET = (Fraction(24000, 1001),   # 23.976
                      Fraction(24),
                      Fraction(25),
                      Fraction(30000, 1001),   # 29.97
                      Fraction(30))


def build_rate_ratio_vocabulary(rate_set=BROADCAST_RATE_SET):
    '''Return every ratio between two rates of `rate_set`, as sorted exact Fractions.

    Deduplicated, closed under reciprocal, 1 excluded.
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
