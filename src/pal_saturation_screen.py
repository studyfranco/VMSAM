"""
Stage 1 of the PAL/speed family design (DESIGN_PAL_SPEED_FAMILY_20260916.MD):
"this offset is not a media fact." Classifies each probe's quantised offset as
a SATURATION ARTIFACT when it sits at the correlator's own search bound.

New module per WRITE_ZONES.MD section 4 ("new runtime modules" are open) --
the integration inside change_point_locator.py is meant to be a thin call,
kept deliberately small so it does not collide with a parallel fix landing in
the same function.

N is DERIVED from the window actually passed, never imported or hardcoded --
the same rule AGENT.MD states for the chromaprint quantum, extended here to
the correlator's search span, which moves with the window for the same reason
the quantum does. Both arms tested against literals in
VMSAM_HELP_AI/dev-pal/002-stage1-saturation-screen.MD.
"""

SATURATION_FRACTION = 0.99  # design's own constant: |points| >= 0.99*(N-32)

# MEASURED, M2 corpus run (VMSAM_HELP_AI/dev-pal/011-corpus-run-three-
# constants.MD SS3): `window_seconds / hop_seconds` OVERSTATES the real
# fpcalc fingerprint length by a FIXED item count, not a fraction of it.
# Directly measured (`audioCorrelation.calculate_fingerprints`, same ffmpeg
# extraction `_probe` uses, 44100 Hz, four different source files):
#
#   window_s   theoretical(N)   real len(fingerprint)   gap
#        8.0          64.5996                     43   21.5996
#       30.0         242.2485                    221   21.2485
#       60.0         484.4971                    463   21.4971
#       90.0         726.7456                    705   21.7456
#      180.0        1453.4912                   1432   21.4912
#
# The gap is CONSTANT across a 22x range of window sizes (21.25-21.75,
# mean 21.516), never proportional to N -- so it is a fixed startup cost in
# fpcalc's own fingerprinting, not a rate error in CHROMAPRINT_HOP_SECONDS.
# BEFORE this fix, at PRODUCTION's own window (60.0 s), the assumed bound
# was 452.5 and SATURATION_FRACTION's threshold (0.99*452.5 = 447.97) sat
# ABOVE the correlator's actual achievable maximum (463-32 = 431) -- so
# `is_saturated()` could NEVER return True at the window size production
# actually calls with, regardless of how far two files diverge. An armed
# guard that has never fired and one that CANNOT fire are the same number
# from outside; this is the second kind, found by measuring the instrument
# it watches rather than only the material fed to it.
#
# SCOPE LIMIT, NAMED SO THE NEXT READER INHERITS IT WITH THE NUMBER: measured
# at 44100 Hz only, four source files, five window sizes -- good evidence the
# gap is fixed ACROSS WINDOW SIZE, no evidence at all that it is fixed ACROSS
# SAMPLE RATE. fpcalc's startup cost is exactly the kind of quantity that
# could scale with the input rate. If the correlator is ever fed another
# rate, this constant is UNVERIFIED there -- the `125 ms` lesson in AGENT.MD
# wearing different clothes: right for the configuration it was measured in,
# silently wrong outside it, nothing downstream recomputing it.
CHROMAPRINT_FIXED_STARTUP_POINTS = 21.5


class SearchBoundUnevaluable(Exception):
    """The window is too short, at this hop and this measured fixed startup
    cost, for the search bound to be positive. Raised, never silently
    answered as True or False: a probe cannot be classified saturated OR
    clean when the instrument has no room left to search in after its own
    startup cost and `min_overlap` are both subtracted from the window's
    point count. Production's own window (60.0 s) is far above the ~6.6 s
    crossover (measured: `probe_search_bound` turns non-positive there), so
    this is a defensive refusal, not a live production path -- but "no
    caller reaches this today" is exactly the reasoning that made the
    pre-fix bound unfireable in the other direction; it is not repeated here."""


def probe_search_bound(window_seconds, hop_seconds, min_overlap):
    """N minus the correlator's own min_overlap -- the span audioCorrelation's
    `compare()` actually scans (span = len(fingerprint) - min_overlap, then
    offsets range over [-span, +span]). `N` is corrected by the MEASURED
    fixed startup cost above; the naive `window_seconds/hop_seconds` alone
    overstates the real fpcalc fingerprint length on every window tested.
    CAN BE NON-POSITIVE for a short enough window -- callers that classify a
    single probe or a population MUST check `bound_is_evaluable` first; this
    function itself only computes the number, it does not judge it."""
    n = window_seconds / hop_seconds - CHROMAPRINT_FIXED_STARTUP_POINTS
    return n - min_overlap


def bound_is_evaluable(window_seconds, hop_seconds, min_overlap):
    """False when `probe_search_bound(...)` is non-positive: at this window,
    hop and min_overlap, the fixed startup cost (plus min_overlap) consumes
    the whole point budget, so no probe here can be told apart from a
    saturated one -- neither True nor False is a safe default (see
    `SearchBoundUnevaluable`)."""
    return probe_search_bound(window_seconds, hop_seconds, min_overlap) > 0


def is_saturated(offset_points, window_seconds, hop_seconds, min_overlap,
                  fraction=SATURATION_FRACTION):
    """Per-probe rule: |offset_points| >= fraction*(N-min_overlap) means the
    correlator's answer is a search-bound artifact, not a measurement -- the
    true offset could lie anywhere beyond what this probe searched.

    Raises `SearchBoundUnevaluable` when the bound itself is non-positive --
    this function never guesses which of the two answers is the safe one."""
    bound = probe_search_bound(window_seconds, hop_seconds, min_overlap)
    if bound <= 0:
        raise SearchBoundUnevaluable(
            f"search bound {bound:.2f} <= 0 at window={window_seconds}s, "
            f"hop={hop_seconds}s, min_overlap={min_overlap}: fixed startup "
            f"cost {CHROMAPRINT_FIXED_STARTUP_POINTS} plus min_overlap "
            f"exceeds the window's own point count")
    return abs(offset_points) >= fraction * bound


def split_saturated(kept, window_seconds, hop_seconds, min_overlap,
                     fraction=SATURATION_FRACTION):
    """`kept`: the (probe_start, result) pairs already gathered, where
    result[2] is the quantised offset_points from `_probe`.

    Returns (unsaturated, saturated_points) -- `unsaturated` is the subset to
    feed every downstream aggregate; `saturated_points` carries each excluded
    probe's own value (retention rule: excluded AND counted, never silently
    dropped).

    Raises `SearchBoundUnevaluable` (checked ONCE, before the loop, not left
    to surface from inside it unlabeled) when the bound is non-positive --
    same refusal as `is_saturated`, at the population entry point."""
    if not bound_is_evaluable(window_seconds, hop_seconds, min_overlap):
        raise SearchBoundUnevaluable(
            f"search bound <= 0 at window={window_seconds}s: cannot split "
            f"{len(kept)} probes into saturated/clean")
    unsaturated = []
    saturated_points = []
    for entry in kept:
        points = entry[1][2]
        if is_saturated(points, window_seconds, hop_seconds, min_overlap, fraction):
            saturated_points.append(points)
        else:
            unsaturated.append(entry)
    return unsaturated, saturated_points


def screen_decline_detail(kept, window_seconds, hop_seconds, min_overlap,
                           fraction=SATURATION_FRACTION):
    """Population rule: a pair whose probes saturate may NEVER close on
    median_fidelity_below_floor alone. The unambiguous case this function
    decides is the strongest one: EVERY kept probe saturated, so nothing
    survives to measure a median from at all -- the caller must decline with
    `offsets_saturated_at_search_bound` before it ever reaches the fidelity
    check, exactly as it already does for `no_quantised_points`.

    A PARTIAL saturation (some probes clear the bound, some don't) is not a
    decline here: the excluded probes are dropped from the aggregates and the
    survivors flow through normally -- the population rule only forces a
    decline when there is nothing left for it to be an alternative to.

    *** THE TRIGGER BAR WAS AN OPEN DESIGN QUESTION; ARCHITECT'S RULING
    2026-09-16 RESOLVES IT INTO A MEASUREMENT, NOT YET A NUMBER. *** The design
    states the population rule without stating the trigger fraction.
    ZERO-SURVIVORS is what fires below because it is the least ambiguous
    predicate to implement, but it is also the MOST CONSERVATIVE one: a pair
    with a MAJORITY of saturated probes and a couple of clean-looking
    survivors still falls through to the normal median_fidelity check on those
    few survivors. Under the minimal-destruction north star that
    under-triggering is the WORSE failure mode -- a PAL pair that slips past
    this screen proceeds through the normal splice as a single-offset relation
    it is not. THE RULING: census the saturation fraction over the 20 real PAL
    ids AND a content-mismatch control population (mismatches also saturate --
    the known confound; the duration-ratio discriminator downstream is what
    separates the two, not this screen) and set the bar to catch 20/20 with a
    named margin, documenting both error directions. ZERO-SURVIVORS STAYS
    until that census exists -- do not raise it without the ruling that names
    the number. What changes now: `saturation_stats` below is returned on
    EVERY call, decline or not, so the census accrues from live runs instead
    of needing a dedicated campaign -- log it unconditionally at the call
    site.

    A NON-POSITIVE BOUND IS ITS OWN THIRD OUTCOME, never folded into either
    of the other two. Returning `unsaturated=kept, decline=None` here would
    silently re-create the pre-fix defect (a guard that answers "clean"
    when it cannot answer at all); returning a full decline that CLAIMS
    saturation would be a named vocabulary token this function does not
    own -- no probe was actually measured against a real bound. So this
    case gets its OWN cause, `search_bound_unevaluable`, distinguishable at
    the caller from `offsets_saturated_at_search_bound` by the `cause` key
    inside `decline_fields` -- the caller must read it, not assume the one
    hardcoded reason it always used before this existed. `unsaturated` is
    returned EMPTY in this case: nothing here was actually classified, so
    nothing should flow to a downstream aggregate as though it had been.

    Returns (unsaturated, decline_fields_or_None, saturation_stats).
    decline_fields, when not None, is a dict of the literal fields `_decline`
    needs beyond reason/measurement, PLUS a `cause` key naming which of the
    two decline shapes this is. saturation_stats is ALWAYS a dict --
    probes_kept, probes_saturated, observed_fraction (probes_saturated /
    probes_kept, the quantity the pending census reads, None when
    unevaluable), search_bound_points, threshold_fraction, and `evaluable`
    (False only in the non-positive-bound case), so an artefact this fired
    on is self-describing even if the threshold changes later.
    """
    bound = probe_search_bound(window_seconds, hop_seconds, min_overlap)
    probes_kept = len(kept)
    if bound <= 0:
        saturation_stats = {
            "probes_kept": probes_kept,
            "probes_saturated": None,
            "observed_fraction": None,
            "search_bound_points": round(bound, 1),
            "threshold_fraction": fraction,
            "evaluable": False,
        }
        return [], {
            "cause": "search_bound_unevaluable",
            "search_bound_points": round(bound, 1),
            "window_seconds": window_seconds,
            "min_overlap": min_overlap,
            "probes_kept": probes_kept,
        }, saturation_stats

    unsaturated, saturated_points = split_saturated(
        kept, window_seconds, hop_seconds, min_overlap, fraction)
    probes_saturated = len(saturated_points)
    saturation_stats = {
        "probes_kept": probes_kept,
        "probes_saturated": probes_saturated,
        "observed_fraction": round(probes_saturated / probes_kept, 4) if probes_kept else None,
        "search_bound_points": round(bound, 1),
        "threshold_fraction": fraction,
        "evaluable": True,
    }
    if unsaturated:
        return unsaturated, None, saturation_stats
    return unsaturated, {
        "cause": "offsets_saturated_at_search_bound",
        "search_bound_points": round(bound, 1),
        "saturation_fraction": fraction,
        "probes_saturated": probes_saturated,
        "probes_kept": probes_kept,
    }, saturation_stats
