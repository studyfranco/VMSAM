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


def probe_search_bound(window_seconds, hop_seconds, min_overlap):
    """N minus the correlator's own min_overlap -- the span audioCorrelation's
    `compare()` actually scans (span = len(fingerprint) - min_overlap, then
    offsets range over [-span, +span])."""
    n = window_seconds / hop_seconds
    return n - min_overlap


def is_saturated(offset_points, window_seconds, hop_seconds, min_overlap,
                  fraction=SATURATION_FRACTION):
    """Per-probe rule: |offset_points| >= fraction*(N-min_overlap) means the
    correlator's answer is a search-bound artifact, not a measurement -- the
    true offset could lie anywhere beyond what this probe searched."""
    bound = probe_search_bound(window_seconds, hop_seconds, min_overlap)
    return abs(offset_points) >= fraction * bound


def split_saturated(kept, window_seconds, hop_seconds, min_overlap,
                     fraction=SATURATION_FRACTION):
    """`kept`: the (probe_start, result) pairs already gathered, where
    result[2] is the quantised offset_points from `_probe`.

    Returns (unsaturated, saturated_points) -- `unsaturated` is the subset to
    feed every downstream aggregate; `saturated_points` carries each excluded
    probe's own value (retention rule: excluded AND counted, never silently
    dropped)."""
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

    Returns (unsaturated, decline_fields_or_None, saturation_stats).
    decline_fields, when not None, is a dict of the literal fields `_decline`
    needs beyond reason/measurement. saturation_stats is ALWAYS a dict --
    probes_kept, probes_saturated, observed_fraction (probes_saturated /
    probes_kept, the quantity the pending census reads), search_bound_points,
    and the threshold_fraction actually applied, so an artefact this fired on
    is self-describing even if the threshold changes later.
    """
    unsaturated, saturated_points = split_saturated(
        kept, window_seconds, hop_seconds, min_overlap, fraction)
    bound = probe_search_bound(window_seconds, hop_seconds, min_overlap)
    probes_kept = len(kept)
    probes_saturated = len(saturated_points)
    saturation_stats = {
        "probes_kept": probes_kept,
        "probes_saturated": probes_saturated,
        "observed_fraction": round(probes_saturated / probes_kept, 4) if probes_kept else None,
        "search_bound_points": round(bound, 1),
        "threshold_fraction": fraction,
    }
    if unsaturated:
        return unsaturated, None, saturation_stats
    return unsaturated, {
        "search_bound_points": round(bound, 1),
        "saturation_fraction": fraction,
        "probes_saturated": probes_saturated,
        "probes_kept": probes_kept,
    }, saturation_stats
