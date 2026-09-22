"""
Stage 2 of the PAL/speed family design (DESIGN_PAL_SPEED_FAMILY_20260916.MD):
DISCRIMINATOR -- names the hypothesis, never applies anything.

WHICH duration: the AUDIO stream durations of the two tracks actually
compared, never container duration (the design's own 0.000737 trap -- headers
lie; 7/20 S1 ids declare 30.000 fps yet carry the same 0.959 relation).

Band membership CLASSIFIES; it never supplies the number. The predicted
factor is always the MEASURED ratio, never the band's nominal constant
(t112's caveat: 1.042710395 vs 1001/960 differ -- hardcoding either is the
quantum defect wearing a constant).

Convention: RATIO = master_span / candidate_span (TASKS/009, already the
convention `merge_video_repair.RATIO_CONVENTION` states and `get_speed_ratio`
consumes) -- so ratio > 1 means the candidate runs FAST and must be SLOWED.

Both arms tested against literals, including corpus-pairs' three confirmed
PAL ids (errid 70/135/213) and the design's own named refutation cases (id
33's cut-not-rate duration ratio, the near-unity blind spot), in
VMSAM_HELP_AI/dev-pal/003-stage2-discriminator.MD.
"""

from decimal import Decimal
from fractions import Fraction

# PAL rate-shaped band, nominal 1001/960 (and its inverse). The design names
# the band's EXISTENCE and the rule that the predicted factor is the measured
# ratio, never this constant -- it does not state a tolerance width. Chosen
# here: the SAME 1% `merge_video_repair.CONVENTION_FREE_MARGIN` already uses
# for a related (not identical) purpose in this exact file, rather than an
# invented number. Real measured PAL ids (corpus-pairs, errid 70/135/213:
# 0.959/0.958/0.956) sit comfortably inside a 1% band around the inverse
# (960/1001 = 0.959041). FLAGGED FOR THE ARCHITECT, NOT A MEASUREMENT: if the
# true population spreads wider than corpus-pairs' three examples, 1% may be
# too narrow and would misroute a real PAL pair to "wrong-content suspicion"
# instead of the speed family.
PAL_NOMINAL = Decimal(1001) / Decimal(960)
BAND_TOLERANCE = Decimal("0.01")

# The design's own named blind spot: ratios within ~0.1% of 1 are undecidable
# by duration alone (a cut changes duration without changing rate -- id 33
# measured 1.0687 duration ratio for a genuine 1.001 rate). This family does
# not enter Stage 2's band classification at all; NTSC is out of scope by the
# same reasoning, named explicitly in the brief.
NEAR_UNITY_MARGIN = Decimal("0.001")


def duration_ratio(duration_master_s, duration_candidate_s):
    """RATIO = master / candidate, in TASKS/009's convention. Returns None
    when either duration is unusable -- absent, never a fabricated ratio."""
    if duration_master_s in (None, 0) or duration_candidate_s in (None, 0):
        return None
    try:
        return Decimal(str(duration_master_s)) / Decimal(str(duration_candidate_s))
    except Exception:
        return None


def classify_band(ratio):
    """Is `ratio` (or its reciprocal) within BAND_TOLERANCE of the PAL
    nominal constant? Band membership CLASSIFIES; the predicted factor
    returned to the caller is always the MEASURED ratio itself, never
    PAL_NOMINAL or its inverse -- t112's caveat, load-bearing.

    Returns one of:
      "near_unity"     -- undecidable by duration alone (design's blind spot)
      "pal_direct"      -- ratio itself is near PAL_NOMINAL (candidate slow)
      "pal_inverse"     -- ratio's reciprocal is near PAL_NOMINAL (candidate fast)
      "no_band"         -- outside every named band -- wrong-content suspicion
      None              -- ratio was None (could not measure at all)
    """
    if ratio is None:
        return None
    if abs(ratio - 1) <= NEAR_UNITY_MARGIN:
        return "near_unity"
    near = PAL_NOMINAL * BAND_TOLERANCE
    if abs(ratio - PAL_NOMINAL) <= near:
        return "pal_direct"
    inverse_nominal = Decimal(1) / PAL_NOMINAL
    near_inverse = inverse_nominal * BAND_TOLERANCE
    if abs(ratio - inverse_nominal) <= near_inverse:
        return "pal_inverse"
    return "no_band"


def discriminate(duration_master_s, duration_candidate_s):
    """The full Stage 2 verdict for one pair, from durations alone.

    Returns a dict: {duration_master_s, duration_candidate_s, speed_ratio,
    speed_ratio_convention, band, hypothesis}. `speed_ratio` is always the
    MEASURED ratio (never the band's nominal constant); `band` classifies,
    `hypothesis` states in words what Stage 3 must now confirm or refute --
    this function never applies anything and never routes to undo by itself.
    """
    ratio = duration_ratio(duration_master_s, duration_candidate_s)
    band = classify_band(ratio)
    if ratio is None:
        hypothesis = "could not measure: one or both audio durations unusable"
    elif band == "near_unity":
        hypothesis = ("undecidable by duration alone: ratio within "
                       f"{NEAR_UNITY_MARGIN} of 1 (a cut changes duration "
                       "without changing rate) -- out of this family's scope")
    elif band in ("pal_direct", "pal_inverse"):
        hypothesis = (f"speed relation suspected, predicted factor = "
                       f"{ratio} (the MEASURED ratio, not the band's nominal "
                       f"{PAL_NOMINAL})")
    else:
        hypothesis = ("ratio outside every named rate-shaped band: "
                       "wrong-content suspicion, route to content-mismatch "
                       "investigation, never to undo")
    return {
        "duration_master_s": duration_master_s,
        "duration_candidate_s": duration_candidate_s,
        "speed_ratio": ratio,
        "speed_ratio_convention": "master_span / candidate_span",
        "band": band,
        "hypothesis": hypothesis,
    }


def discriminate_from_videos(master_video_obj, candidate_video_obj, language):
    """`discriminate`, reading the durations off real video objects instead
    of taking bare numbers -- WHICH duration, named forever: the AUDIO
    stream durations of the two tracks actually compared, never container
    duration (the design's own 0.000737 trap).

    Reuses `change_point_locator._audio_duration_seconds` rather than
    building a second reader of the same MediaInfo convention (stream_tags
    `Duration`) -- `locate_change_points` already reads exactly this
    quantity, on exactly this convention, for exactly this reason (see its
    own top-of-file citation of `video.py`'s MediaInfo convention). A second
    reader of one measurement is a second convention, which is the defect
    `ARTEFACT_FORMATS.md` SS9c exists to name. Late, tolerant import,
    matching `merge_video_repair.get_plan_from_locator`'s own pattern: this
    module must not fail to import if the locator is not deployed.

    This is the SECOND call site for that private function (the first is
    inside `change_point_locator.locate_change_points` itself) -- per the
    Lead's standing note, a private name reused a second time is still just
    reuse; a THIRD site would be the signal to promote it, and that is a
    decision for the Lead/Architect, not a refactor to make quietly here.
    """
    try:
        import change_point_locator
    except Exception as error:
        return discriminate(None, None), f"locator_module_absent: {error}"
    duration_master_s = change_point_locator._audio_duration_seconds(
        master_video_obj, language)
    duration_candidate_s = change_point_locator._audio_duration_seconds(
        candidate_video_obj, language)
    return discriminate(duration_master_s, duration_candidate_s), None


# ---------------------------------------------------------------------------
# RATE FACTOR FROM THE OFFSET SLOPE
# (RULING_20260922_NO_BAND_ROUTING.MD point 2, implementing the instrument
# RULING_20260921_STEP1_CLASSIFIER_DESIGN.MD already put in force)
#
# WHY THIS EXISTS AT ALL. `duration_ratio` above compares TOTAL durations, so
# it cannot tell a rate difference from a content-length difference -- the
# defect the ruling names, and the one this module's own docstring already
# flagged as "id 33's cut-not-rate duration ratio". A per-window offset SLOPE
# cannot be confused that way: a bounded content insertion moves the offsets
# it covers by a STEP, while a rate difference tilts EVERY window by the same
# amount per second. So the slope is a factor; the duration ratio is a screen.
#
# NOTHING HERE READS A WINDOW LENGTH, A PROBE SPACING OR A CHROMAPRINT
# QUANTUM. The series arrives with its own `(position_seconds, delay_ms)`
# pairs and, optionally, the `window` metadata that produced it -- which is
# LOGGED and never substituted. There is deliberately no default: `125` (and
# the 124 the real id 33 series actually carries) is exactly the kind of
# stand-in the Architect's E3 ruling banned, and a series whose geometry is
# unknown says so instead of borrowing someone else's.
# ---------------------------------------------------------------------------

NAMED_RATE_RATIONALS = (
    Fraction(1001, 1000),   # NTSC film -> film        (+0.100 %)
    Fraction(1000, 1001),   # film -> NTSC film        (-0.100 %)
    Fraction(1001, 960),    # PAL family, nominal      (+4.271 %)
    Fraction(960, 1001),    # PAL family, inverse      (-4.096 %)
    Fraction(25, 24),       # 24 -> 25 straight pull-up(+4.167 %)
    Fraction(24, 25),       # 25 -> 24 straight pull-dn(-4.000 %)
)

# ARITHMETIC, NOT A TUNED MARGIN -- and it is derived from the set above
# rather than chosen. The two CLOSEST members of `NAMED_RATE_RATIONALS` are
# 1001/960 = 1.04270833 and 25/24 = 1.04166667, a relative gap of 1.0e-3. A
# snap window WIDER than half that gap could put one measurement inside two
# named rationals at once, which is not a recognition; 5e-4 is that half gap,
# so the windows meet and never overlap. `snap_to_named_rational` still
# refuses on ambiguity rather than trusting this arithmetic to hold if the
# named set ever grows -- a guard that costs nothing and survives an edit.
SNAP_RELATIVE_TOLERANCE = Decimal("0.0005")


def _named_rational_decimals():
    """The named set as exact Decimals, computed once per call site rather
    than stored: `Fraction` is the authority on what the value IS, and a
    second hand-written table of the same six numbers is the second
    convention this module already refuses to create elsewhere."""
    return [(named, Decimal(named.numerator) / Decimal(named.denominator))
            for named in NAMED_RATE_RATIONALS]


def derive_rate_factor_from_slope(delay_series, window=None,
                                  min_points=None, r_squared_min=None):
    """Linear regression over a per-window `(position_seconds, delay_ms)`
    series -> the slope-implied rate factor and its residual scatter.

    `delay_series` is any iterable of pairs (longer tuples are tolerated and
    read positionally: position first, delay second). `window` is the
    metadata that PRODUCED the series -- window length, probe spacing,
    quantum, which module measured it. It travels into the returned dict and
    into the caller's log line; it is never read as a number by this
    function, because this function does not need one.

    THE REGRESSION IS NOT MINE. `change_point_locator._robust_slope_regression`
    is the instrument RULING_20260921_STEP1_CLASSIFIER_DESIGN.MD put in
    force, it already carries the two-pass MAD outlier exclusion that keeps a
    single wild probe from defeating a clean drift, and Stage 1 already
    decides `speed_relation_suspected` with it. Re-deriving a second
    least-squares fit here would be a second instrument answering one
    question -- the defect `ARTEFACT_FORMATS.md` SS9c names. This is the
    SECOND call site of that private function (the first is inside
    `locate_change_points`); by the Lead's standing note a second reuse is
    still reuse, and a third would be the signal to promote it. Late,
    tolerant import, the same pattern `discriminate_from_videos` above uses
    and for the same reason.

    ALWAYS RETURNS A DICT -- never None, never a bare number:

        factor                    the slope-implied rate factor, or None on
                                  EVERY refusal. `factor is None` is the one
                                  test a caller needs for "no rate leg".
        refusal                   the named token when `factor` is None
        slope_seconds_per_second  the measured slope, in the units the factor
                                  is built from
        slope_ms_per_s            the same slope as the instrument reports it
        residual_scatter_ms       median |residual| over the points the
                                  regression KEPT -- the scatter the ruling
                                  asks to be returned beside the factor
        r_squared, n_used, n_excluded, n_points, span_seconds, window

    FACTOR = 1 / (1 + SLOPE), AND IT WAS `1 + SLOPE` UNTIL REAL MEDIA SAID
    OTHERWISE. *** CORRECTED 2026-09-22 ON THE id 33 END-TO-END RUN: THE
    FIRST VERSION RETURNED THE RECIPROCAL OF THIS MODULE'S OWN CONVENTION,
    AND I AM THE ONE WHO TALKED THE ARCHITECT INTO REGISTERING THAT ERROR AS
    A RULING (ADDENDUM 1's "direction correction"). ***

    The derivation, so nobody has to trust the sign again. Both sides are
    probed at the SAME absolute time, so for content sitting at master
    instant `t_m` and candidate instant `t_c`:

        offset(t_m) = t_c - t_m = t_m * (t_c/t_m - 1)
        slope       = t_c/t_m - 1                 [seconds per second]
        1 + slope   = t_c / t_m = candidate_span / master_span

    and this module's RATIO convention is `master_span / candidate_span`,
    the RECIPROCAL of that. Hence the division.

    MEASURED, id 33, three instruments against the old formula: slope
    -0.000949 s/s -> 1/(1 - 0.000949) = 1.000949 -> snaps 1001/1000, which is
    what the RATE SWEEP measured (median 0.9555, 1 of 16 factors) and what
    the PITCH layer measured (1.001086 against a predicted 1.001). The old
    `1 + slope` gave 0.99905 -> 1000/1001, the reciprocal, which would have
    stretched the candidate 0.2% THE WRONG WAY -- the exact defect
    `merge_video_repair.get_speed_ratio` names "le cas DESTRUCTEUR": near
    unity, where no bound and no tolerance can catch it.

    WHICH SIDE MUST BE CORRECTED IS STILL NOT DECIDED HERE, and since the
    owner's override (ADDENDUM 2) this factor decides nothing at all -- the
    rate SWEEP measures every rate combination against the fidelity ladder
    and the best median wins. This function's output is corroboration. That
    run is the argument FOR the override: the inferred number was wrong and
    the measured one was right.
    """
    samples = []
    for sample in delay_series or ():
        samples.append((float(sample[0]), float(sample[1])))
    result = {"factor": None, "refusal": None,
              "slope_seconds_per_second": None, "slope_ms_per_s": None,
              "residual_scatter_ms": None, "r_squared": None,
              "n_used": 0, "n_excluded": 0, "n_points": len(samples),
              "span_seconds": None, "window": window}
    if len(samples) >= 2:
        positions = [s[0] for s in samples]
        result["span_seconds"] = max(positions) - min(positions)
    try:
        import change_point_locator
    except Exception as error:
        # THE INSTRUMENT IS NOT DEPLOYED. That is my own process state and it
        # is a different answer from "the slope says no" -- BRIEF_COMMON rule
        # 5, the distinction this whole family refuses to collapse.
        result["refusal"] = "slope_instrument_absent"
        result["refusal_detail"] = f"{type(error).__name__}: {error}"
        return result
    if min_points is None:
        min_points = change_point_locator.RATE_SLOPE_MIN_POINTS
    if r_squared_min is None:
        r_squared_min = change_point_locator.RATE_SLOPE_R_SQUARED_MIN
    result["min_points"] = min_points
    result["r_squared_min"] = r_squared_min
    if len(samples) < 3:
        result["refusal"] = "slope_series_too_short"
        return result

    starts = [s[0] for s in samples]
    offsets = [s[1] for s in samples]
    fit = change_point_locator._robust_slope_regression(starts, offsets)
    if fit is None:
        # Fewer than 3 survivors, or every position identical: the instrument
        # declined to fit. Not "the slope is flat" -- there is no slope.
        result["refusal"] = "slope_regression_impossible"
        return result

    kept = [i for i in range(len(samples)) if i not in set(fit["excluded_indices"])]
    residuals = [abs(offsets[i] - (fit["slope_ms_per_s"] * starts[i]
                                   + fit["intercept_ms"])) for i in kept]
    residuals.sort()
    middle = len(residuals) // 2
    result["residual_scatter_ms"] = (
        residuals[middle] if len(residuals) % 2
        else (residuals[middle - 1] + residuals[middle]) / 2.0)
    result["slope_ms_per_s"] = fit["slope_ms_per_s"]
    result["slope_seconds_per_second"] = fit["slope_ms_per_s"] / 1000.0
    result["r_squared"] = fit["r_squared"]
    result["n_used"] = fit["n_used"]
    result["n_excluded"] = fit["n_excluded"]

    # THE TWO GUARDS ARE THE ONES ALREADY IN FORCE, at their existing values.
    # Stage 1 decides `speed_relation_suspected` on exactly this pair of
    # numbers (`change_point_locator.py`, the rate-family gate); inventing a
    # third threshold here would mean two places disagreeing about when a
    # slope is real.
    if fit["n_used"] < min_points:
        result["refusal"] = "slope_points_below_floor"
        return result
    if fit["r_squared"] < r_squared_min:
        # SCATTER-DOMINATED. The points do not lie on a line, so the line's
        # slope is not a measurement of anything -- the refusal the ruling
        # requires this guard to be able to make.
        result["refusal"] = "slope_scatter_dominated"
        return result
    # `1 + slope` IS candidate/master; this module's ratio is master/candidate.
    # See the derivation in the docstring -- this division is the whole of the
    # 2026-09-22 direction correction, and a slope of exactly -1 s/s (a
    # candidate of zero length) is the only value it cannot take, which is not
    # a media this pipeline can be handed.
    candidate_over_master = 1.0 + result["slope_seconds_per_second"]
    if candidate_over_master <= 0:
        result["refusal"] = "slope_implies_non_positive_span"
        return result
    result["factor"] = 1.0 / candidate_over_master
    return result


def snap_to_named_rational(factor, tolerance=SNAP_RELATIVE_TOLERANCE):
    """Nearest member of `NAMED_RATE_RATIONALS` within `tolerance` (RELATIVE),
    as an exact `Fraction` -- or None.

    WHY EXACT RATIONALS AND NOT THE MEASURED NUMBER. t112's caveat, still
    load-bearing above: a BAND must never supply the number. This is the
    other half of that rule and not a contradiction of it -- the measured
    slope has to RECOGNISE the rational before the rational may be used, and
    what is applied afterwards is the exact ratio the broadcast standards
    actually define, not a float that happens to be near one. A resample at
    1.0010000 and a resample at the measured 1.0010342 differ by 3.4e-6,
    which is 5 ms over a 1500 s episode -- under a frame, but it is a drift
    that accumulates for no reason when an exact value is available.

    None means "no named rational owns this number", which is a MEASUREMENT
    (the ruling's "unsnappable slope = no rate leg, go to 4"), and `None` in
    means `None` out so a refused derivation flows straight through.

    AMBIGUITY REFUSES. If two named rationals are both inside the window the
    factor has recognised neither, and returning the nearer one would be
    picking a winner the measurement did not name.
    """
    if factor is None:
        return None
    value = Decimal(str(factor))
    if value <= 0:
        return None
    window = Decimal(str(tolerance))
    inside = []
    for named, nominal in _named_rational_decimals():
        relative = abs(value - nominal) / nominal
        if relative <= window:
            inside.append((relative, named))
    if not inside:
        return None
    if len(inside) > 1:
        return None
    return inside[0][1]


def describe_rate_corroboration(snapped, master_fps, candidate_fps):
    """Do the DECLARED frame rates agree with the snapped rational? Prose for
    the log, and nothing else -- this function returns no verdict and no
    routing, by construction.

    HEADERS LIE: the standing finding this module's own docstring opens with
    (7/20 S1 ids declare 30.000 fps and carry the same 0.959 relation). So a
    declared rate may CORROBORATE a measured slope and may never decide one,
    which is why this returns a string and the caller logs it.
    """
    if snapped is None:
        return "no snapped rational to corroborate"
    if master_fps is None or candidate_fps is None:
        return (f"declared rates unavailable "
                f"(master={master_fps}, candidate={candidate_fps}): "
                f"no corroboration possible, headers are not required")
    try:
        # *** CANDIDATE OVER MASTER, NOT MASTER OVER CANDIDATE. CORRECTED
        # 2026-09-22 on the id 33 run, same defect and same day as the slope
        # inversion above. *** A span is frames DIVIDED by rate, so for the
        # same frame count `master_span / candidate_span` equals
        # `candidate_fps / master_fps` -- the fps ratio is the RECIPROCAL of
        # the duration ratio this module states its convention in. The old
        # line compared `master_fps / candidate_fps` against a
        # `master_span / candidate_span` rational, i.e. a number against its
        # own reciprocal, and on id 33 it printed `corroborates=True` for
        # 1000/1001 with a relative gap of 8e-17 -- A FALSE CORROBORATION OF
        # THE WRONG DIRECTION, wearing the most convincing number in the run.
        # A corroborator that can agree with the reciprocal of the truth is
        # worse than none: it is evidence pointing the wrong way.
        declared = Decimal(str(candidate_fps)) / Decimal(str(master_fps))
    except Exception:
        return (f"declared rates unreadable "
                f"(master={master_fps}, candidate={candidate_fps})")
    nominal = Decimal(snapped.numerator) / Decimal(snapped.denominator)
    relative = abs(declared - nominal) / nominal
    agrees = relative <= SNAP_RELATIVE_TOLERANCE
    return (f"declared master_fps={master_fps} candidate_fps={candidate_fps} "
            f"-> declared_ratio(master_span/candidate_span = "
            f"candidate_fps/master_fps)={declared} vs snapped="
            f"{snapped.numerator}/"
            f"{snapped.denominator} relative_gap={relative} "
            f"corroborates={agrees} (headers corroborate, never decide)")
