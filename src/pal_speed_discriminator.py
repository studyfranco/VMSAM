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
