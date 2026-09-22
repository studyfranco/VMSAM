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

DECLARED LIMIT, not a defect to fix (Lead's ruling, 2026-09-21, on a real-
media measurement -- `VMSAM_HELP_AI/dev-step3-vector/001-stage1-vector-
escalation-ladder.MD`, errid267 section): THIS METHOD CANNOT SEE A TAIL GAP
BY CONSTRUCTION. `compute_similarity_vector` truncates `V[]` to the SHORTER
of the two fingerprint arrays -- the same convention `audioCorrelation`'s
own `correlation()` already uses for a list-length mismatch, reused for
consistency, not invented here. When a candidate's real content simply RUNS
OUT (a tail-trim), the comparison never reaches the region where it is
missing; the array quietly ends there instead of showing a concordance
drop. Measured directly on real audio with independent frame-exact ground
truth (a real tail-trim at a known boundary): the instrument found NOTHING
at the true boundary and, on a widened window, a real but UNRELATED dip
elsewhere -- confirming, not merely explaining, why head/tail brackets are
computed from `candidate_end_ms` directly and never reach this module's
per-transition call site (RULINGS_IN_FORCE row 31). Callers must not expand
this module's scope to cover a tail case; that is a different measurement,
already made, elsewhere.
"""

import os

import audioCorrelation
import tools

# THEORETICAL constant, reimplemented fresh (audioCorrelation.py is frozen).
# Q4's own answer (017): this is NOT the same number as the measured
# `size_point_ms` below, even though both describe "one fingerprint step" --
# named separately, on purpose, never conflated.
CHROMAPRINT_HOP_MS = 4096.0 / 3.0 / 11025.0 * 1000.0  # 123.840 ms

RUPTURE_THRESHOLD = 0.50  # s3c's own number. Also the chance baseline for two
                          # unrelated 32-bit fingerprints -- see the degeneracy
                          # screen below, which exists because of this fact.
                          # CONFIRMS a ramp already found by RUPTURE_ONSET_THRESHOLD
                          # below; never used alone as the bracket edge since
                          # 2026-09-21 (see find_rupture_onset_confirm).

# --- dual-threshold onset/confirm, added 2026-09-21 (dev-step3-vector) ------
# MEASURED, not assumed: the single-threshold RUPTURE_THRESHOLD crossing
# (`find_rupture`, kept below for what it actually measures) landed 1.086s
# and 3.100s PAST the true, independently-known cut on two real fixtures
# (lab/diagnose_vi.py, diagnose_vi2.py; VMSAM_HELP_AI/dev-step3-vector/
# 001-stage1-vector-escalation-ladder.MD carries the full numbers). Root
# cause, read off the raw V[i]: chromaprint's own smoothing turns a real,
# instantaneous edit into a ~1.5-3.8s roughly-monotonic RAMP, not a step;
# past the ramp, post-cut content reads as CHANCE-BASELINE NOISE (real
# consecutive values measured: 0.406, 0.531, 0.594, 0.500, 0.625 -- both
# sides of 0.50, dozens of spurious crossings across a file). "First
# crossing below 0.50" therefore fires on the first noise excursion under
# chance, not on the ramp's onset -- dev-pal's original scoping note (S3c
# build, same day) predicted exactly this before any of it was measured:
# 0.50 sits at chance baseline, so a rupture verdict there needs a
# degeneracy screen because the threshold sits exactly where noise lives.
# The prediction and the measurement are the same finding, eight hours
# apart.
#
# Architect's ruling 2026-09-21: SPEC_ZONE_A s3c's "~125 ms" bracket is the
# per-window quantum -- the INSTRUMENT'S RESOLUTION -- never a promised
# output width (CAMPAIGN.MD s2: "the economy stage, not a limit").
# CONTAINMENT of the true cut is the contract; width is reported economy,
# not silently discarded when the ramp is genuinely wider than one point.
RUPTURE_ONSET_THRESHOLD = 0.90  # measured n=2 real fixtures: catches the
                          # ramp's actual departure from the stable plateau
                          # 0.3-0.55s BEFORE the true cut on both -- tighter
                          # than the confirm-only crossing's 1.1-3.1s AFTER
                          # it, and on the correct side (early, not late).
RUPTURE_MAX_RAMP_SECONDS = 6.0  # widest onset->confirm gap measured, n=2:
                          # 3.78s (cut_reencode_across.mka, the smoother,
                          # re-encoded-across-the-join arm). This bound is
                          # ~1.6x that, not a guess -- an onset with no
                          # confirm inside it is an isolated dip, not a
                          # rupture ramp, and DECLINES named
                          # (`onset_without_confirm`) rather than emitting
                          # an unbounded bracket. Revisit as n grows past 2.
                          # Also keeps the emitted bracket well under
                          # SCENE_SEARCH_WINDOW_SECONDS (10s, ARCH_FRAME_
                          # ACCURATE.MD's own Stage-2 reach) -- checked
                          # explicitly below, not only by this margin.
SCENE_SEARCH_WINDOW_SECONDS = 10.0  # ARCH_FRAME_ACCURATE.MD Stage 2's own
                          # named constant ("scene_search_window_sec = 10"),
                          # cited not reinvented. A bracket this wide or
                          # wider does not narrow anything Stage 2 could not
                          # already reach on its own -- Architect's ruling
                          # 2026-09-21: assert bracket width < this reach,
                          # decline named rather than ship it.

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


MAX_ONSET_CANDIDATES_PER_WINDOW = 10  # named ceiling, 2026-09-21 (the Lead's
    # ruling): "onset_without_confirm" on the FIRST crossing is a negative
    # result about ONE dip, not a negative result about the window -- the
    # could-not-measure-read-as-measured-negative defect, an eighth instance
    # this campaign has found, in this criterion. Measured on real audio
    # (Shuumatsu no Walkure S01E01, Erai-Raws vs DBD-Raws BDRip, ja):
    # a single early noise dip (one point at 0.84 inside an otherwise
    # 0.91-1.0 field) consumed the ONLY candidate the old code tried, and a
    # real, clean, 5964ms transition 60-100s further into the window was
    # never reached. REJECT AND CONTINUE, bounded: an unbounded scan over a
    # noisy V[] is a hang (real V[] past a ramp is noisy by measurement,
    # see the module-level comment above RUPTURE_ONSET_THRESHOLD) -- 10 is
    # chosen as comfortably above what one genuine transition plus ordinary
    # noise should ever need (the real fixture above needed 2: one false,
    # one true) and far below where a scan becomes its own cost concern.


def find_rupture_onset_confirm(v, window_points=DEFAULT_WINDOW_POINTS,
                                onset_threshold=RUPTURE_ONSET_THRESHOLD,
                                confirm_threshold=RUPTURE_THRESHOLD,
                                max_ramp_points=None,
                                max_candidates=MAX_ONSET_CANDIDATES_PER_WINDOW):
    """Two-threshold rupture location -- see the module-level comment above
    RUPTURE_ONSET_THRESHOLD for the measurement that motivated this.
    `find_rupture` (above) is kept UNCHANGED and still measures exactly what
    its own docstring says (the single 0.50 crossing); this function does not
    replace it, it is what `locate_zone_by_vector` now calls instead for the
    EMITTED bracket, because the single crossing was measured missing the
    true cut by 1.1-3.1s on real audio.

    ONSET: a rolling-mean crossing below `onset_threshold` -- concordance
    starting to depart the stable plateau. CONFIRM: the first rolling-mean
    value at or after an onset that is below `confirm_threshold` -- the ramp
    completing into genuine divergence (chance baseline for unrelated
    fingerprints, SPEC_ZONE_A s3c's own number, unchanged). `max_ramp_points`
    bounds how far past an onset a confirm may still count: past that bound,
    that ONE onset is an isolated dip, not a rupture ramp -- REJECT IT AND
    CONTINUE to the next onset crossing, up to `max_candidates` (the Lead's
    ruling, 2026-09-21: a rejected candidate is a fact about that candidate,
    never a fact about the window -- see `MAX_ONSET_CANDIDATES_PER_WINDOW`).

    Returns `(onset_index, confirm_index, candidates_examined)`:
      (None, None, 0)      -- ZERO onset crossings anywhere in the window.
                             Caller reports `no_rupture_found` -- there was
                             nothing to examine, distinct from examining
                             something and rejecting it.
      (None, None, N>0)    -- N onset candidates were examined, in order,
                             and NONE confirmed within the bound -- either
                             `max_candidates` was reached or the window ran
                             out of rolling-mean positions to try. Caller
                             reports `onset_without_confirm` carrying N, so a
                             reader can tell "looked once" from "looked N
                             times", never a bare decline with the count
                             thrown away.
      (onset_index, confirm_index, N) -- the Nth candidate examined is the
                             one that confirmed (1-indexed count of how many
                             were tried, including this one). The caller's
                             bracket spans [onset_index, confirm_index + 1) --
                             data-driven width: a truly sharp, unsmoothed cut
                             has onset and confirm one or two points apart
                             and the bracket stays near the spec's ~125ms
                             resolution; a smoothed real edit widens it
                             honestly instead of reporting a point that
                             missed."""
    if not (MIN_WINDOW_POINTS <= window_points <= MAX_WINDOW_POINTS):
        raise ValueError(f"window_points must be in [{MIN_WINDOW_POINTS},"
                         f"{MAX_WINDOW_POINTS}], got {window_points}")
    n = len(v)
    if n < window_points + 1:
        return None, None, 0
    rolling = [sum(v[i:i + window_points]) / window_points
               for i in range(n - window_points + 1)]
    candidates_examined = 0
    search_from = 1
    while candidates_examined < max_candidates:
        onset_index = None
        for i in range(search_from, len(rolling)):
            if rolling[i - 1] >= onset_threshold and rolling[i] < onset_threshold:
                onset_index = i
                break
        if onset_index is None:
            # No FURTHER onset crossing exists past where the last
            # candidate left off -- genuinely nothing left to examine,
            # not a bound reached. Zero candidates so far means no onset
            # ever fired; N>0 means N were tried and none confirmed.
            return None, None, candidates_examined
        candidates_examined += 1
        ceiling = (len(rolling) if max_ramp_points is None
                   else min(len(rolling), onset_index + max_ramp_points))
        confirm_index = None
        for i in range(onset_index, ceiling):
            if rolling[i] < confirm_threshold:
                confirm_index = i
                break
        if confirm_index is not None:
            return onset_index, confirm_index, candidates_examined
        # REJECTED, NOT ABSENT: this candidate did not confirm -- that is a
        # fact about this dip, not about the window. Resume the search
        # strictly past this onset's own index so the same crossing is
        # never re-examined (each `while` iteration finds a DIFFERENT,
        # LATER crossing).
        search_from = onset_index + 1
    # Bound reached with candidates still possibly remaining past
    # `search_from` -- reported as examined-and-rejected, same shape as
    # running out of window, because the caller-facing claim is identical:
    # "this many were tried, none confirmed," never "there is no cut."
    return None, None, candidates_examined


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
                          tag="zsv", candidate_offset_ms=0.0):
    """Full Stage 1 for one probed window. `samples_for_plateaus`: the SAME
    `[(probe_start_seconds, offset_ms), ...]` shape `change_point_locator.
    _group_plateaus()` already consumes -- supplied by the caller, never
    regenerated here; this module reuses whatever probe history the caller
    already has rather than re-probing. `fps_num`/`fps_den`: the pair's own
    rational grid, carried through verbatim (never measured or invented
    here) so the emission states its grid rather than assuming one --
    ADDENDUM's own "rational grid" requirement.

    `candidate_offset_ms` (BASELINE_OFFSET_BLINDNESS fix, 2026-09-21, the
    Lead's own wiring, corrected by the Lead): the flanking plateau's own
    established offset (e.g. `before["mean"]`), applied to the CANDIDATE
    extraction only -- master stays on its own raw timeline, the reference
    every returned position is measured against, unchanged. `master` is
    always extracted at `start_seconds`; `candidate` is extracted at
    `start_seconds + candidate_offset_ms / 1000.0` (sign convention,
    unchanged elsewhere in this module: `candidate_time = master_time +
    offset`). Defaults to 0.0 -- existing callers that never pass it get the
    OLD (measured-broken-on-real-baselines) behaviour unchanged, so nothing
    silently starts assuming alignment that was not asked for.

    WHY THIS EXISTS, measured not argued (`VMSAM_HELP_AI/dev-step3-vector/
    001-stage1-vector-escalation-ladder.MD`, "URGENT" section): comparing
    fingerprints point-by-point (`V[i] = popcount(A[i] XOR B[i])`, no lag
    search -- that IS this method, by SPEC_ZONE_A s3c's own text) is correct
    ONLY when the baseline offset between the two sides is already ~0
    entering the call. On a real same-language pair with a real, ordinary
    1002ms sync offset and ZERO content divergence, the unshifted call
    read `mean_V=0.574` (chance baseline); shifting the candidate side by
    the known +1.002s read `mean_V=0.993`. The one synthetic fixture this
    method was validated against before this fix (`corpus-C-structural-cut/
    synth-cut/cut_concat_at_join.mka`) has baseline offset 0 BY
    CONSTRUCTION -- spliced within one continuous recording, never encoded
    as a separate release -- so the defect could not appear on it. This is
    why REAL_MEDIA_ACCEPTANCE exists as a rule and not merely a preference.

    The offsets needed were ALREADY present at every call site before this
    fix -- `samples_for_plateaus` carries exactly the `[(probe_start_seconds,
    offset_ms), ...]` history this function's own flanking-run lookup reads
    for provenance AFTER a rupture is found. The information required to
    align the comparison sat unused in the same call that performed the
    misaligned one.

    Returns a dict, always, `verdict` one of (SEVEN, not five -- two added
    2026-09-21 for the dual-threshold onset/confirm bracket, see
    `find_rupture_onset_confirm`'s own docstring and the module-level
    comment above `RUPTURE_ONSET_THRESHOLD` for the real-media measurement
    that forced this; a rupture verdict must never carry a silent `None`
    offset, the could-not-measure-read-as-measured defect inverted):
      "unreliable_degenerate_input"       -- the degeneracy screen fired
                                             first; nothing past
                                             `degeneracy` is trusted.
      "window_below_detection_floor"      -- too few raw fingerprint items
                                             to run even one rolling-window
                                             comparison. NOT "no rupture
                                             found" -- that claim requires
                                             having actually evaluated.
      "no_rupture_found"                  -- evaluated; no onset crossing.
                                             `i_cut`/`onset_index` are None.
      "onset_without_confirm"             -- an onset crossing fired but no
                                             confirm crossing followed within
                                             `RUPTURE_MAX_RAMP_SECONDS` -- an
                                             isolated dip, not a rupture
                                             ramp. `onset_index` is set,
                                             `confirm_index` and the bracket
                                             are None: NEVER a fabricated
                                             bracket from an unconfirmed dip.
      "bracket_exceeds_stage2_reach"      -- onset and confirm both fired,
                                             but the resulting bracket is
                                             `>= SCENE_SEARCH_WINDOW_SECONDS`
                                             wide -- Architect's ruling
                                             2026-09-21: a bracket that wide
                                             narrows nothing Stage 2's own
                                             +/-10s reach could not already
                                             find; decline named rather than
                                             ship a bracket that outgrew its
                                             own purpose. `bracket_low_ms`/
                                             `bracket_high_ms`/`bracket_
                                             width_ms` ARE populated here
                                             (so the width that triggered
                                             the decline is inspectable);
                                             `offset_before`/`offset_after`
                                             are not computed.
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
        # IMMEDIATELY-PRE-CALL, x4 (owner's order via the Lead, 2026-09-22):
        # `cpl._extract` logs its own ffmpeg call internally, but not this
        # module's `tag`; `calculate_fingerprints` lives in FROZEN
        # audioCorrelation.py, so this open caller is the only lever.
        tools.dev_log(f"zsv: locate_zone_by_vector extracting master tag={tag} "
                      f"file={master_path} start_seconds={start_seconds}\n")
        cpl._extract(master_path, master_stream, start_seconds, window_seconds,
                    master_wav, sample_rate)
        # BASELINE_OFFSET_BLINDNESS fix -- see this function's own docstring.
        # Master stays on its own raw timeline; candidate reads from where
        # the flanking plateau's own offset says its content actually is.
        candidate_start_seconds = start_seconds + candidate_offset_ms / 1000.0
        tools.dev_log(f"zsv: locate_zone_by_vector extracting candidate "
                      f"tag={tag} file={candidate_path} "
                      f"start_seconds={candidate_start_seconds}\n")
        cpl._extract(candidate_path, candidate_stream, candidate_start_seconds,
                    window_seconds, candidate_wav, sample_rate)
        tools.dev_log(f"zsv: locate_zone_by_vector calculating fingerprints "
                      f"tag={tag} file={master_wav}\n")
        fp_master = audioCorrelation.calculate_fingerprints(master_wav, length=window_seconds)
        tools.dev_log(f"zsv: locate_zone_by_vector calculating fingerprints "
                      f"tag={tag} file={candidate_wav}\n")
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
        "candidate_offset_ms": candidate_offset_ms,         # what alignment this call
                                             # actually used -- never a bare V[] with
                                             # no record of which timeline it was read on
        "degeneracy": degeneracy,
        "i_cut": None,                      # alias for onset_index -- kept for
                                             # callers already reading it; "the
                                             # index where local concordance
                                             # DROPS" (owner's own words) is the
                                             # onset, not the noise-floor confirm.
        "onset_index": None, "confirm_index": None,   # BOTH emitted, always,
                                             # per the Lead's ruling 2026-09-21:
                                             # never hide the pair behind a
                                             # single derived field.
        "start_point": None, "end_point": None,
        "bracket_low_ms": None, "bracket_high_ms": None,
        "bracket_width_ms": None,           # emitted always once a bracket
                                             # exists, checked against
                                             # SCENE_SEARCH_WINDOW_SECONDS
                                             # below -- economy REPORTED,
                                             # never silently promised.
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

    max_ramp_points = (int(round(RUPTURE_MAX_RAMP_SECONDS * 1000.0 / size_point_ms))
                       if size_point_ms else None)
    onset_index, confirm_index, candidates_examined = find_rupture_onset_confirm(
        v, window_points=window_points, onset_threshold=RUPTURE_ONSET_THRESHOLD,
        confirm_threshold=threshold, max_ramp_points=max_ramp_points)
    result["i_cut"] = onset_index
    result["onset_index"] = onset_index
    result["onset_candidates_examined"] = candidates_examined   # ALWAYS
                                             # present -- the Lead's ruling,
                                             # 2026-09-21: "scanned N, none
                                             # confirmed" must be
                                             # distinguishable from "found no
                                             # onset at all" (N==0), and the
                                             # count travels with the result
                                             # rather than being implied.
    if onset_index is None and candidates_examined == 0:
        result["verdict"] = "no_rupture_found"
        return result
    result["confirm_index"] = confirm_index
    if confirm_index is None:
        # candidates_examined >= 1 here (reject-and-continue exhausted every
        # candidate up to MAX_ONSET_CANDIDATES_PER_WINDOW or the window's own
        # end): each one was an isolated dip, not a rupture ramp -- a fact
        # about those candidates, never a fact claiming the window has no
        # cut. NEVER fabricate a bracket from this: the single-threshold
        # predecessor's own defect (a lone noise excursion read as the cut)
        # is exactly what pairing onset with a bounded confirm exists to
        # refuse, and stopping on the FIRST rejected candidate was the SAME
        # defect one level up (the Lead's finding, 2026-09-21, real media:
        # one early noise dip consumed the only try and a real 5964ms
        # transition further into the window was never reached).
        result["verdict"] = "onset_without_confirm"
        return result

    start_point, end_point = onset_index, confirm_index + 1
    result["start_point"] = start_point
    result["end_point"] = end_point
    result["bracket_low_ms"] = start_point * size_point_ms
    result["bracket_high_ms"] = end_point * size_point_ms
    bracket_width_ms = result["bracket_high_ms"] - result["bracket_low_ms"]
    result["bracket_width_ms"] = bracket_width_ms
    if bracket_width_ms >= SCENE_SEARCH_WINDOW_SECONDS * 1000.0:
        # Architect's ruling 2026-09-21: a bracket this wide narrows nothing
        # Stage 2's own +/-10s reach could not already find on its own --
        # decline named, width still inspectable on the returned dict.
        result["verdict"] = "bracket_exceeds_stage2_reach"
        return result

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
