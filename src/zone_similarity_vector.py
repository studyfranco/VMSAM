"""
Stage 1 of the owner's ordered zone-detection design (SPEC_ZONE_A.MD s3c,
2026-09-21 owner order; ARCH_FRAME_ACCURATE.MD's own Stage 0-4 transposition,
same date). Build order: Lead dispatch, Architect-ratified acceptance
conditions, VMSAM_HELP_AI/dev-pal/018-stage1-vector-build.MD.

Global correlation is banned for cut-finding (WRITE_ZONES.MD P6 ruling):
`correlate()`/`compare()`/`correlation()` scan OFFSET at a fixed comparison
window and return only the winning offset's summary -- the axis this module
needs is TIME/POSITION at a (locally) fixed offset, a different computation
that reuses the same low-level primitive (017's own finding). audioCorrelation.py
is FROZEN, so the popcount/XOR primitive is REIMPLEMENTED here fresh, on an
open module, per the Lead's explicit instruction -- never imported from the
frozen module.

Reuses, never re-derives:
- `audioCorrelation.calculate_fingerprints()` -- "the sanctioned source" (s3c),
  the ONLY thing taken from the frozen module (a read, not a primitive).
- `change_point_locator._extract()` -- the same per-window audio extraction
  `_probe()` already uses (second call site, same reuse discipline
  `pal_speed_discriminator.discriminate_from_videos` already established for
  `_audio_duration_seconds`).
- `change_point_locator._group_plateaus()` / `_merge_narrow_runs()` -- THE
  OFFSETS' SOURCE, RESOLVED BY THE LEAD, TRACED NOT GUESSED (018's own
  provenance trail). This module does not compute a plateau offset itself;
  it calls the real, unmodified functions on real probe samples the caller
  supplies, and reports the flanking runs' mean/n_members/tolerance triple.
"""

import os

import audioCorrelation

# THEORETICAL constant, reimplemented fresh (audioCorrelation.py is frozen).
# Q4's own answer (017): this is NOT the same number as the measured
# `size_point_ms` below, even though both describe "one fingerprint step" --
# named separately, on purpose, never conflated.
CHROMAPRINT_HOP_MS = 4096.0 / 3.0 / 11025.0 * 1000.0  # 123.840 ms

RUPTURE_THRESHOLD = 0.50  # s3c's own number. Also the chance baseline for two
                          # unrelated 32-bit fingerprints -- see the degeneracy
                          # screen below, which exists because of this fact.
MIN_WINDOW_POINTS = 3
MAX_WINDOW_POINTS = 5
DEFAULT_WINDOW_POINTS = 3  # the tightest end of s3c's named 3-5 range --
                           # chosen for sensitivity; Q5's own derivation
                           # already shows the method's floor is in the
                           # hundreds of ms regardless, so a wider default
                           # only trades sensitivity for smoothing.


def _popcount32(value):
    return bin(value & 0xFFFFFFFF).count("1")


def compute_similarity_vector(fingerprint_a, fingerprint_b):
    """V[i] = (32 - popcount(A[i] XOR B[i])) / 32.0, index-aligned, truncated
    to the SHORTER list -- the same truncation convention `audioCorrelation.
    correlation()` already uses for its own list-length mismatch, reused for
    consistency rather than invented independently. [] if either side is
    empty."""
    n = min(len(fingerprint_a), len(fingerprint_b))
    return [(32 - _popcount32(fingerprint_a[i] ^ fingerprint_b[i])) / 32.0
            for i in range(n)]


def degeneracy_report(fingerprint_a, fingerprint_b):
    """The inverted twin of the saturation-screen degeneracy (measured this
    session, `014-saturation-positive-class.MD`): there, a degenerate
    (constant) fingerprint on BOTH sides pinned a quantised offset at the
    search bound. HERE, a degenerate fingerprint on either side can make
    V[i] artefactually FLAT -- and if both sides happen to share the SAME
    constant (silence against silence), flat AND HIGH (V[i]=1.0 throughout),
    which could read as "no rupture" or mask a real one, because 0.50 (the
    rupture threshold) is also the chance baseline for genuinely unrelated
    32-bit fingerprints (017's own finding). FLAGS, does not refuse -- a
    caller decides whether a near-fully-degenerate side invalidates the
    reading; this function states the fact."""
    distinct_a = len(set(fingerprint_a))
    distinct_b = len(set(fingerprint_b))
    return {
        "n_items_a": len(fingerprint_a),
        "n_items_b": len(fingerprint_b),
        "n_distinct_a": distinct_a,
        "n_distinct_b": distinct_b,
        "degenerate_a": distinct_a <= 1 and len(fingerprint_a) > 0,
        "degenerate_b": distinct_b <= 1 and len(fingerprint_b) > 0,
    }


def find_rupture(v, window_points=DEFAULT_WINDOW_POINTS, threshold=RUPTURE_THRESHOLD):
    """s3c: "sliding window of 3-5 points; a sharp fall below 0.50 names
    i_cut, in O(N)." The spec gives a shape, not an algorithm -- the exact
    operational definition, stated so it can be checked rather than
    inferred: a ROLLING MEAN of V over `window_points` consecutive items;
    `i_cut` is the index of the FIRST window whose rolling mean crosses from
    `>= threshold` to `< threshold` -- a threshold CROSSING, not merely the
    lowest point observed. A profile that starts below threshold and never
    rises above it first has no crossing to report: returns None
    (`no_rupture_found`), never a fabricated index 0.

    `i_cut` is the START index (into `v`) of the first window whose mean
    fell below the threshold -- i.e. "concordance begins falling at or
    after position i_cut", not the window's end."""
    if not (MIN_WINDOW_POINTS <= window_points <= MAX_WINDOW_POINTS):
        raise ValueError(f"window_points must be in [{MIN_WINDOW_POINTS},"
                         f"{MAX_WINDOW_POINTS}], got {window_points}")
    n = len(v)
    if n < window_points + 1:
        return None
    rolling = [sum(v[i:i + window_points]) / window_points
               for i in range(n - window_points + 1)]
    for i in range(1, len(rolling)):
        if rolling[i - 1] >= threshold and rolling[i] < threshold:
            return i
    return None


def _flanking_runs(runs, cut_absolute_seconds, probe_window_seconds):
    """The plateau run before the cut, the run after it, and the run
    STRADDLING it, if any -- named explicitly rather than let a straddling
    run fall through an `elif` and be silently skipped (measured defect,
    this build: skipping it let the runs on EITHER SIDE of the actual
    containing run be reported as `before`/`after`, wrong numbers carrying
    a full provenance triple that made them look measured).

    A run's covered span is `[run["first"], run["last"] +
    probe_window_seconds)` -- `run["last"]` is the START of its LAST
    MEMBER PROBE, not the run's own end (`change_point_locator.py`'s own
    convention, its plateau-end computation adds `PROBE_WINDOW_SECONDS`
    for exactly this reason); using `run["last"]` alone as an end boundary
    under-covers every run by one full probe window.

    Returns `(before_run, after_run, straddling_run)`. Exactly one of
    `(before_run OR after_run)` or `straddling_run` carries information --
    never a silent gap. When `straddling_run` is not None, `before_run` and
    `after_run` are both None: the plateau instrument measured that whole
    span as ONE consistent offset, disagreeing with V[i] about there being
    a split there at all -- a disagreement to report, not to resolve by
    picking a side.
    """
    for run in runs:
        run_start = run["first"]
        run_end = run["last"] + probe_window_seconds
        if run_start <= cut_absolute_seconds < run_end:
            return None, None, run
    before_run, after_run = None, None
    for run in runs:
        run_end = run["last"] + probe_window_seconds
        if run_end <= cut_absolute_seconds:
            before_run = run
        elif after_run is None and run["first"] >= cut_absolute_seconds:
            after_run = run
    return before_run, after_run, None


def locate_zone_by_vector(master_path, master_stream, candidate_path, candidate_stream,
                          start_seconds, window_seconds, work_dir, sample_rate,
                          samples_for_plateaus, window_points=DEFAULT_WINDOW_POINTS,
                          threshold=RUPTURE_THRESHOLD, fps_num=None, fps_den=None,
                          tag="zsv"):
    """Full Stage 1 for one probed window. `samples_for_plateaus`: the SAME
    `[(probe_start_seconds, offset_ms), ...]` shape `change_point_locator.
    _group_plateaus()` already consumes -- supplied by the caller, never
    regenerated here; this module reuses whatever probe history the caller
    already has rather than re-probing. `fps_num`/`fps_den`: the pair's own
    rational grid, carried through verbatim (never measured or invented
    here) so the emission states its grid rather than assuming one --
    ADDENDUM's own "rational grid" requirement.

    Returns a dict, always, `verdict` one of (FIVE, not four -- a rupture
    verdict must never carry a silent `None` offset, the could-not-measure-
    read-as-measured defect inverted; this build's own finding, fixed here):
      "unreliable_degenerate_input"       -- the degeneracy screen fired
                                             first; nothing past
                                             `degeneracy` is trusted.
      "window_below_detection_floor"      -- too few raw fingerprint items
                                             to run even one rolling-window
                                             comparison. NOT "no rupture
                                             found" -- that claim requires
                                             having actually evaluated.
      "no_rupture_found"                  -- evaluated; no threshold
                                             crossing. `i_cut` is None.
      "rupture_found_offsets_unavailable" -- `i_cut`/bracket ARE populated,
                                             but the plateau instrument
                                             cannot supply a clean flanking
                                             pair: either the cut sits
                                             inside a single, UN-SPLIT
                                             plateau run (`straddling_run_
                                             mean_ms` populated -- the
                                             plateau instrument disagrees
                                             with V[i] about a split
                                             existing here at all), or one
                                             side is genuinely absent
                                             (edge of the probed data).
                                             `offset_before`/`offset_after`
                                             are both None here, by
                                             contract, never a guess.
      "rupture_found"                     -- `i_cut`/bracket/BOTH offsets
                                             populated, with full
                                             provenance. The only verdict
                                             where `offset_before`/`offset_
                                             after` are non-None.

    NEVER cuts, never writes a track, never applies anything -- detect only,
    same discipline as every other Stage in this chain today.
    """
    import change_point_locator as cpl

    master_wav = os.path.join(work_dir, f"{tag}_master.wav")
    candidate_wav = os.path.join(work_dir, f"{tag}_candidate.wav")
    try:
        cpl._extract(master_path, master_stream, start_seconds, window_seconds,
                    master_wav, sample_rate)
        cpl._extract(candidate_path, candidate_stream, start_seconds, window_seconds,
                    candidate_wav, sample_rate)
        fp_master = audioCorrelation.calculate_fingerprints(master_wav, length=window_seconds)
        fp_candidate = audioCorrelation.calculate_fingerprints(candidate_wav, length=window_seconds)
    finally:
        for path in (master_wav, candidate_wav):
            try:
                os.remove(path)
            except OSError:
                pass

    degeneracy = degeneracy_report(fp_master, fp_candidate)
    v = compute_similarity_vector(fp_master, fp_candidate)
    n_items = len(v)
    window_ms = window_seconds * 1000.0
    size_point_ms = (window_ms / n_items) if n_items else None

    result = {
        "verdict": None,
        "n_items": n_items,
        "window_ms": window_ms,
        "size_point_ms": size_point_ms,                    # MEASURED, this call
        "chromaprint_hop_ms_theoretical": CHROMAPRINT_HOP_MS,  # THEORETICAL, never conflated with the line above
        "min_detectable_zone_ms": (window_points * size_point_ms
                                    if size_point_ms else None),  # declared, never silent (Q5)
        "grid_num": fps_num, "grid_den": fps_den,           # carried through verbatim, never invented
        "degeneracy": degeneracy,
        "i_cut": None,
        "start_point": None, "end_point": None,
        "bracket_low_ms": None, "bracket_high_ms": None,
        "offset_before": None, "offset_after": None,
        "straddling_run_mean_ms": None,   # present in EVERY result (hard constraint 5:
                                           # one return value, no field that only
                                           # exists conditionally), populated only
                                           # when the plateau instrument disagrees
                                           # with V[i] about there being a split here.
    }

    # ORDER MATTERS: degeneracy is checked FIRST, unconditionally -- a
    # degenerate input invalidates a floor reading built on top of it just
    # as much as a rupture reading (hard constraint 2: "every rupture
    # verdict pairs with a degeneracy check").
    if degeneracy["degenerate_a"] or degeneracy["degenerate_b"]:
        result["verdict"] = "unreliable_degenerate_input"
        return result

    if n_items < window_points + 1:
        # HARD CONSTRAINT 3: this is NOT "no rupture found" -- that claim
        # requires having actually run the rolling-window comparison at
        # least once. A window too narrow to do that COULD NOT EVALUATE,
        # and reporting it as a clean pass would be exactly the
        # could-not-measure-read-as-measured-negative defect this campaign
        # keeps finding in other modules.
        result["verdict"] = "window_below_detection_floor"
        return result

    i_cut = find_rupture(v, window_points=window_points, threshold=threshold)
    result["i_cut"] = i_cut
    if i_cut is None:
        result["verdict"] = "no_rupture_found"
        return result

    start_point, end_point = i_cut, i_cut + 1
    result["start_point"] = start_point
    result["end_point"] = end_point
    result["bracket_low_ms"] = start_point * size_point_ms
    result["bracket_high_ms"] = end_point * size_point_ms

    # --- the two offsets: REUSED from the real plateau machinery, never re-derived ---
    runs = cpl._merge_narrow_runs(cpl._group_plateaus(samples_for_plateaus))
    cut_absolute_seconds = start_seconds + (start_point * size_point_ms) / 1000.0
    before_run, after_run, straddling_run = _flanking_runs(
        runs, cut_absolute_seconds, cpl.PROBE_WINDOW_SECONDS)

    def _provenance(run):
        if run is None:
            return None
        return {"mean_ms": run["mean"], "n_members": len(run["members"]),
                "tolerance_ms": cpl.PLATEAU_TOLERANCE_MS}

    result["offset_before"] = _provenance(before_run)
    result["offset_after"] = _provenance(after_run)
    result["straddling_run_mean_ms"] = (straddling_run["mean"]
                                        if straddling_run is not None else None)

    if straddling_run is not None or before_run is None or after_run is None:
        # A RUPTURE VERDICT MUST NEVER CARRY A SILENT None OFFSET (Lead's
        # finding, this build): that is the could-not-measure-read-as-
        # measured defect, inverted -- a result that LOOKS fully populated
        # (a verdict, a bracket, a provenance-shaped dict) while actually
        # missing the very thing its own name promises. Two distinct causes
        # collapse to the SAME token here because both mean the same thing
        # to a caller: "V[i] found a rupture, but the plateau instrument
        # cannot supply a clean flanking pair for it" -- either because a
        # single, un-split run CONTAINS the cut (the plateau/scalar
        # instrument measured this whole span as one consistent offset,
        # disagreeing with V[i] about there being a split here at all --
        # not resolved by picking a side, reported instead), or because one
        # side is genuinely absent (edge of the probed data).
        result["verdict"] = "rupture_found_offsets_unavailable"
        return result

    result["verdict"] = "rupture_found"
    return result
