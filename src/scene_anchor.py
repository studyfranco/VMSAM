# scene_anchor.py
"""
THE BIDIRECTIONAL SCENE ANCHOR PROTOCOL (owner's specification, 2026-09-21).
See CAMPAIGNS/02-dev-chimeric-resample/ARCH_FRAME_ACCURATE.MD, the section
of that name, and SPEC_ZONE_A.MD S4h's AMENDMENTS. New runtime module (S4:
"new capability goes in its own module, called from a tagged zone"), called
from merge_video_chimeric.py's open code.

AUTHORITY (Architect's ruling, 2026-09-21): once wired, this module is the
AUTHORITATIVE method for frame-refinement/validation at a bracket.
frame_compare's F1 methods (locate_bracket_boundary / locate_match_onset)
remain independent CROSS-CHECKS and are NEVER silently promoted to the
answer when this protocol declines -- a decline here means the caller's
tier declines too, full stop. The caller records F1's result, if any, as
named evidence; it must never be consumed for the boundary on a decline
here (enforced at the call site, not in this module).

DEPENDENCY, NAMED SO A REFACTOR IS FOUND BY GREP, NOT BY BREAKAGE
(Architect's ruling, 2026-09-21): this module imports frame_compare's
PRIVATE helpers (_extract_hashes, FrameComparer._popcount64) AS A LIBRARY,
unmodified -- reuse of the already-correct native-rate frame extraction and
pHash primitives (no `fps=` resampling, exact rational grid), not a
duplicate implementation. A refactor of those in frame_compare.py breaks
this module; coordinate via the Lead. No edit is made to frame_compare.py
for this mission (Architect's constraint).

TWO DIFFERENT PROPERTIES, NAMED HERE SO A FUTURE FIXTURE AUTHOR DOES NOT
CONFLATE THEM THE WAY THIS MISSION DID (Architect's instruction,
2026-09-21, "for the record"):

  CROSS-SCENE DISCRIMINATION   does scene X's content differ from scene
                               Y's? Needed for PySceneDetect seeding and
                               for `_frames_match` to tell RED from GREEN
                               at all. A fixture with distinct per-scene
                               colours/patterns has this even if every
                               frame WITHIN one scene is identical.

  WITHIN-SCENE DISTINCTIVENESS  does the content at THIS position differ
                               from the content a few frames away, WITHIN
                               THE SAME SCENE? This is what
                               `_check_anchor_distinctive` tests, and it
                               is what the owner's "mouths moving over a
                               fixed background" warns is not automatic
                               just because the SCENE is identifiable.

This module's own first fixture ("fixture 1", the RED/YELLOW/GREEN/BLUE
checkerboard) had the FIRST property and not the second -- every frame
within one of its scenes was pixel-identical by construction, so it
looked like a valid positive-path test right up until the distinctiveness
guard existed and correctly rejected it too, alongside the deliberately
degenerate fixture it was built to be an improvement over. Both are
legitimate findings, not a broken test: a guard that rejects a fixture
nobody had verified had the second property is the guard working. See
`fixtures/combined/` (dev-anchor's lab, not tracked here) for a fixture
built with BOTH properties measured, not assumed.
"""
from fractions import Fraction
from decimal import Decimal

import tools
from frame_compare import FrameComparer, _extract_hashes
from scenedetect import open_video, SceneManager, ContentDetector


SCENE_SEARCH_WINDOW_SECONDS_DEFAULT = 10.0

# ContentDetector's own default. CARRIED, NEVER TUNED ON A SINGLE SYNTHETIC
# FIXTURE (Lead's ruling, 2026-09-21): fixture 1 (a solid-colour red->yellow
# transition) measured content_val=10.0 against this threshold, while
# yellow->green (52.0) and green->blue (62.3) cleared it easily -- so this
# constant CAN miss a true hard cut. A solid-colour synthetic is
# unrepresentative in BOTH directions: real content has texture and a real
# cut usually scores far higher, but a real low-contrast transition might
# score lower than anything a synthetic fixture can construct. The protocol
# TOLERATES a missed cut BY DESIGN -- see the union-of-seeds comment in
# `_anchor_search` below -- because the scene list is only a SEED for
# anchor candidates, never itself the answer. The miss RATE on production
# material is UNMEASURED; do not lower this constant on n=1 synthetic
# evidence (same defect this campaign corrected three times in one day
# elsewhere). If a real case ever needs a specific cut caught to succeed at
# all, that case is fragile and it is a finding, not a reason to retune this.
CONTENT_DETECTOR_THRESHOLD_DEFAULT = 27.0

# ">= 3 consecutive identical frames", owner's spec, both for Anchor A/B
# validation and as the viability floor below.
MIN_VALIDATION_FRAMES = 3

# Same Hamming vocabulary as frame_compare.py's own boundary validation
# (BOUNDARY_VALIDATION_HAMMING_THRESHOLD_DEFAULT) -- a DIFFERENT constant,
# not imported, because this module's validation question ("are these two
# frames the SAME picture") is stricter than F1's ("is this boundary
# roughly where we think"): anchor frames must be near-identical, not
# merely close enough to confirm a shift.
ANCHOR_HAMMING_THRESHOLD_DEFAULT = 6


def _scene_anchor_config():
    '''`scene_search_window_sec`: config.ini [features], additions-only
    per WRITE_ZONES S3. Absent section or key -> the owner's stated
    default, never a silent zero (a zero window would make EVERY bracket
    decline `search_window_unviable`, which is not the same as "the
    feature is off" -- that gate lives at the CALL SITE via
    `scene_anchor_protocol`, not here).

    A MALFORMED VALUE IS NOT THE SAME FACT AS AN ABSENT ONE (Lead's
    finding, 2026-09-21): a non-numeric `scene_search_window_sec` (a typo
    in config.ini) used to raise `ValueError` past this function --
    caught only by the call site's own exception isolation, degrading to
    `protocol_errored`, a worse diagnosis than the typo deserves. An
    ABSENT section or key means "nobody set one, use the sensible
    default" and stays that way. A PRESENT-BUT-UNPARSEABLE value means
    "someone tried to configure this and it is wrong" and must not be
    silently swallowed into the same default -- it returns `None`, which
    the caller's own viability check already turns into a NAMED
    `search_window_unviable` decline (its evidence literally showing
    `scene_search_window_sec=None`), the honest diagnosis rather than a
    masked one.
    '''
    try:
        section = tools.config_loader(tools.config_file, "features")
    except Exception:
        return SCENE_SEARCH_WINDOW_SECONDS_DEFAULT
    raw = section.get("scene_search_window_sec")
    if raw is None:
        return SCENE_SEARCH_WINDOW_SECONDS_DEFAULT
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _exact_ms_from_frame(frame_count, fps_num, fps_den):
    '''Same pattern as merge_video_chimeric.py's own `exact_ms_from_frame`
    (Architect's ruling, 2026-09-16): Decimal, never float, never a
    rounded-for-display value consumed as an input -- the frame index
    times the grid's own exact frame duration, nothing rounded in between.
    '''
    frame_ms_exact = Decimal(1000 * fps_den) / Decimal(fps_num)
    return Decimal(frame_count) * frame_ms_exact


def _nominal_shift_frames(offset_ms, fps_num, fps_den):
    '''Same arithmetic as frame_compare._nominal_shift_frames -- not
    imported (that name is private to that module's own F1 producer and
    this is a small, self-contained computation, not a primitive worth
    coupling to).'''
    frame_ms = 1000.0 * fps_den / fps_num
    return int(round(offset_ms / frame_ms))


def _scene_cut_frames(path, start_frame, n_frames, threshold, debug=False):
    '''PySceneDetect's ContentDetector over [start_frame, start_frame+n_frames)
    of `path`, returning ABSOLUTE frame numbers (this file's own frame 0),
    never PySceneDetect's float-fps timecode -- STRICT RATIONAL-RATE
    CONSERVATION (owner, 2026-09-21): only the integer frame INDEX is
    read, everything else in this module computes ms from that index on
    the pair's own exact rational grid.

    Measured (dev-anchor, 2026-09-21): `video.seek(start_frame)` followed
    by `detect_scenes(video, duration=n_frames)` returns scene frame
    numbers ABSOLUTE to the file, not relative to the seek point --
    verified directly (seek(100), scene 0 started at frame_num=100, not 0)
    before this function was written on the strength of that measurement.

    A cut is a SCENE START (`get_scene_list()` returns contiguous
    (start,end) pairs covering the requested span, scene i's end == scene
    i+1's start) so every interior boundary is every scene-start after the
    first. Declines to an empty list on any failure (unreadable region,
    file too short, PySceneDetect exception) -- a caller with no seeds from
    this side falls back to the other side's seeds or to validating
    directly at the bracket edge; a crash here must never abort the whole
    protocol over what is, by design, only a candidate generator.
    '''
    if n_frames <= 0:
        return []
    try:
        video = open_video(path)
        if start_frame > 0:
            video.seek(start_frame)
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=threshold))
        sm.detect_scenes(video, duration=n_frames)
        scene_list = sm.get_scene_list()
        if len(scene_list) < 2:
            return []
        return [scene.frame_num for scene, _ in scene_list[1:]]
    except Exception as exc:
        if debug:
            tools.logs.append(f"scene_anchor: PySceneDetect failed on {path} "
                              f"[{start_frame},{start_frame + n_frames}): "
                              f"{type(exc).__name__}\n")
        return []


def _frames_match(m_hashes, m_base, m_frame, c_hashes, c_base, c_frame,
                  threshold):
    mi, ci = m_frame - m_base, c_frame - c_base
    if not (0 <= mi < len(m_hashes)) or not (0 <= ci < len(c_hashes)):
        return None  # unreadable, never treated as either match or mismatch
    return FrameComparer._popcount64(m_hashes[mi] ^ c_hashes[ci]) <= threshold


def _validate_anchor(m_hashes, m_base, c_hashes, c_base, m_seed, shift_frames,
                     direction, threshold):
    '''Owner's ">= 3 consecutive identical frames" check, one direction at
    a time. `direction="forward"`: validate [m_seed, m_seed+N) against the
    candidate under `shift_frames`. `direction="backward"`: validate
    [m_seed-N, m_seed) instead. Which anchor uses which direction is
    `_anchor_search`'s own call (Anchor B validates forward from its seed,
    Anchor A validates backward from its seed) -- this function only
    performs the check it is told to, in the direction it is given.
    Returns True only if EVERY one of the `MIN_VALIDATION_FRAMES` pairs is
    both readable and matching -- an unreadable frame is not a pass by
    omission.
    '''
    frames = (range(m_seed, m_seed + MIN_VALIDATION_FRAMES) if direction == "forward"
             else range(m_seed - MIN_VALIDATION_FRAMES, m_seed))
    for m_frame in frames:
        c_frame = m_frame + shift_frames
        result = _frames_match(m_hashes, m_base, m_frame, c_hashes, c_base,
                               c_frame, threshold)
        if result is not True:
            return False
    return True


def _anchor_search(m_hashes, m_base, c_hashes, c_base, seeds, shift_frames,
                   direction, threshold):
    '''Try each seed (a MASTER-coordinate frame number) in order, return
    the first that validates. `seeds` is the CALLER's responsibility to
    order by proximity to the bracket (closest first) -- this function is
    a pure "first validated seed wins", not a search-order policy.

    UNION, NOT AGREEMENT (measured on fixture 1, dev-anchor, 2026-09-21):
    seeds come from BOTH files' independent PySceneDetect runs, translated
    to master coordinates, PLUS the bracket edge itself. Requiring the two
    files' scene lists to agree on a cut before trying it as a seed would
    have failed fixture 1's own red->yellow transition, which
    ContentDetector's default threshold misses on the CANDIDATE side only
    (content_val=10.0 there) while the MASTER side's equivalent transition
    -- RED ending, in master coordinates -- still seeds it correctly. THE
    PROTOCOL'S CORRECTNESS MUST NOT DEPEND ON THE SCENE DETECTOR'S
    COMPLETENESS: the scene list is a SEED for anchor candidates; the
    pHash frame-by-frame validation in `_validate_anchor` above is what
    actually proves an anchor, never the scene detector by itself. A
    missed cut on one side costs a candidate, not the answer, as long as
    some usable seed exists nearby on EITHER side. If a real case is ever
    found where a specific cut must be detected for the protocol to
    succeed at all, that case is fragile and it is a finding, not evidence
    this design was wrong.
    Returns `(seed, None)` on success, `(None, reason)` on failure, where
    `reason` is `None` if no seed ever validated at all, or the FIRST
    rejected-as-uninformative seed's evidence if at least one did but
    failed the distinctiveness probe. A PLAIN LOCAL VARIABLE, not module
    state -- CORRECTED IN THIS REVISION (Lead's finding, 2026-09-21,
    reproduced against this module directly, not reasoned about): an
    earlier version used a module-level scratch attribute cleared only on
    the failure path, so a search that rejected an early seed as
    uninformative and then SUCCEEDED on a later one left its reason
    behind for the NEXT search's failure to pick up -- a bracket where
    nothing could even be tried would then be misreported as
    `anchor_uninformative` (a measured, content-based refutation) instead
    of `anchors_not_established` (nothing to measure), the exact fold the
    Architect's ruling names as wrong, in the direction that manufactures
    a measurement that never happened. THE INVARIANT THIS VERSION PROVES:
    a search's returned reason is produced by THAT search, and by no
    earlier one -- trivially true of a local variable, which is scoped to
    one call by the language itself and needs no cross-call bookkeeping
    to get right. The original comment's stated reason for avoiding a
    return-tuple field ("avoiding widening every caller's unpacking") did
    not survive its own revision: both call sites already unpack a
    two-tuple.
    '''
    last_uninformative_reason = None
    for seed in seeds:
        if not _validate_anchor(m_hashes, m_base, c_hashes, c_base, seed,
                                shift_frames, direction, threshold):
            continue
        distinctive, why_not = _check_anchor_distinctive(
            m_hashes, m_base, c_hashes, c_base, seed, shift_frames, direction, threshold)
        if distinctive:
            return seed, None
        # MATCHED BUT UNINFORMATIVE -- try the next seed rather than
        # accepting a match a wrong shift could have produced just as
        # easily. Remember the reason from the FIRST such seed only (the
        # evidence a caller actually wants is "why did the closest/best
        # candidate fail", not the last one tried).
        if last_uninformative_reason is None:
            last_uninformative_reason = why_not
    return None, last_uninformative_reason


ANCHOR_DISTINCTIVENESS_PROBE_FRAMES = (4, 8)
# EXCLUDE the immediate neighbourhood of the true shift -- natural
# continuity (adjacent real frames are often somewhat alike even in
# distinctive content) is not the failure mode this guard exists to
# catch; a self-similar RUN is. Probing only at well-separated deltas
# means a positive result (still matches) is a real signal, not noise
# from nearby-frame similarity.


def _check_anchor_distinctive(m_hashes, m_base, c_hashes, c_base, m_seed,
                              shift_frames, direction, threshold):
    '''ANCHOR DISTINCTIVENESS (Architect's ruling, 2026-09-21, replacing
    point (i) of the 2026-09-17 ruling): "an anchor match is evidence only
    if the local content could have refuted it." Anchors answer WHERE;
    offsets answer HOW MUCH; no offset-derived quantity can corroborate
    WHERE (see `_check_step_plumbing` below for what offset comparison
    actually is, now that it no longer claims to be this). A match inside
    a self-similar run (the owner's "mouths moving over a fixed
    background") is unfalsifiable and therefore not evidence, whatever
    hypothesis produced it.

    THE TEST: the same >= 3-frame window that validated at `shift_frames`
    is re-checked at ALTERNATIVE, well-separated shifts
    (`ANCHOR_DISTINCTIVENESS_PROBE_FRAMES` frames away, both directions).
    If it ALSO matches at ANY of those -- meaning the surrounding content
    is indifferent to which position it is read from -- the match carries
    no information about WHERE the true boundary is, and the anchor is
    UNINFORMATIVE. Uses frames already extracted into `m_hashes`/`c_hashes`
    (no new probe class, no new ffmpeg call): only the comparison shifts,
    not the extraction.

    Returns `(True, None)` when the match survives every probe (a real
    anchor), `(False, evidence_str)` on the first probe that ALSO matches
    (uninformative).
    '''
    for delta in ANCHOR_DISTINCTIVENESS_PROBE_FRAMES:
        for probe_shift in (shift_frames + delta, shift_frames - delta):
            if _validate_anchor(m_hashes, m_base, c_hashes, c_base, m_seed,
                               probe_shift, direction, threshold):
                return False, (f"seed={m_seed} also matches at shift="
                              f"{probe_shift} (true shift={shift_frames}, "
                              f"delta={probe_shift - shift_frames}) -- "
                              f"self-similar content, uninformative")
    return True, None


def _check_step_plumbing(delta_frames, frame_ms, step_ms, quantum_ms):
    '''NOT CORROBORATION. Renamed and re-documented in place (Architect's
    ruling, 2026-09-21) after the Lead proved by algebra, and I confirmed
    by direct simulation, that what this function used to call
    "corroboration" cannot corroborate anything: `delta_frames` (the net
    divergence between the two anchors' candidate positions) reduces
    EXACTLY to `after_shift - before_shift` for every possible sweep
    outcome -- the anchor placements cancel out of the arithmetic
    entirely. Verified: 20 synthetic (split_start_master, split_end_master)
    pairs spanning a wide range all produced the identical delta. In
    PRODUCTION `step_ms` is *also* built from the same two plateau means
    that produce `offset_before_ms`/`offset_after_ms`
    (`change_point_locator.py:2269` vs `:2333`/`:2426`) -- so this
    comparison is a rounding-residue check between two renderings of ONE
    quantity, not two independent instruments. Measured (Lead,
    2026-09-21, live artefacts): the comparison branch cannot fire above
    ~7.75 fps; the ABSENT-input branch below IS reachable and exercised
    in real production (15 of the quantum-bearing lines in a 6-hour
    window carry `quantum_ms=None`, counted over artefacts DEDUPLICATED
    BY CONTENT -- an earlier reading of 23 counted mirrored copies of the
    same artefact as separate cases; corrected by the Lead, same day).

    Anchors answer WHERE; offsets answer HOW MUCH; no offset-derived
    quantity can corroborate WHERE (Architect's ruling). What actually
    corroborates an anchor is `_check_anchor_distinctive`, above --
    content-based, computed from the SAME frames the anchor's own match
    was proven on, never from the offset hypotheses.

    WHAT THIS FUNCTION IS, HONESTLY: a rounding/pipe-integrity assert
    between the locator's own `step_ms` and this module's own frame-shift
    inputs -- useful for catching a caller bug (a `step_ms` belonging to
    the wrong bracket, a units mismatch) at near-zero cost, since the
    quantities are already in hand. It proves the PLUMBING agrees with
    itself, not that the anchors are correctly placed. Its result is
    reported under `anchor_step_unavailable` (input missing -- reachable,
    per the measurement above) but the former `anchor_step_uncorroborated`
    name is RETIRED: a token claiming corroboration that cannot fire is
    the exact "permanent zero read as always-succeeds" trap this
    module's own `_check_anchor_ordering` docstring warns about, aimed at
    the wrong guard.
    '''
    if step_ms is None or quantum_ms is None:
        return False, f"plumbing_check=not_available step_ms={step_ms} quantum_ms={quantum_ms}"
    delta_ms = delta_frames * frame_ms
    agrees = abs(delta_ms - step_ms) <= quantum_ms
    return agrees, (f"counted_delta={delta_frames} frames ({delta_ms:.2f} ms) vs "
                    f"locator step_ms={step_ms} quantum_ms={quantum_ms}")


def _check_anchor_ordering(anchor_a, anchor_b):
    '''Extracted to a pure, directly-testable unit (Architect's ruling,
    2026-09-21) so "fire every guard deliberately, with literals" applies
    at the BRANCH level, independent of whether an adversarial fixture
    can be built to reach it through the full search.

    Cross-sweep is only meaningful with A strictly before B -- an INVERTED
    pair (`anchor_a > anchor_b`) is not a smaller bracket, it is the
    search producing nonsense, and shipping it would be exactly the
    "silently ships a boundary nobody confirmed" failure this whole
    protocol exists to prevent.

    STRICT, NOT `>=` -- CORRECTED after reasoning through reachability
    found a real false-positive, not a hypothetical one (dev-anchor,
    2026-09-21, reasoning about guard ordering per the Lead's instruction
    to reason rather than hunt a fixture). `_frame_index` rounds ms to the
    NEAREST frame on the pair's own exact rational grid -- a bracket under
    one frame wide (a legitimate, common shape: Stage 1's own bracket can
    be that narrow when its measurement is already precise) rounds
    `bracket_low_ms` and `bracket_high_ms` to the SAME frame index, so
    `m_bracket_first == m_bracket_last` by construction, not by error.
    When that single frame's content genuinely matches under BOTH offset
    hypotheses -- an entirely ordinary, no-divergence bracket -- Anchor A
    and Anchor B both validate at that SAME seed, giving `anchor_a ==
    anchor_b`. MEASURED: a sub-frame bracket (2000.0-2001.0 ms, well
    inside fixture 1's RED scene, offset 0 both sides -- genuinely no
    divergence there) produced exactly this and the `>=` form declined it
    `cross_sweep_refuted` with `evidence="anchor_a=48 >= anchor_b=48"` --
    REFUSING A LEGITIMATE RESULT, not catching a bad one. `anchor_a ==
    anchor_b` is not nonsense; it is the zero-width degenerate case, and
    the cross-sweep below already handles it correctly on its own (both
    sweep ranges are empty, `length_master == length_candidate == 0`,
    the distinctiveness check already ran per anchor before either was
    accepted, and the plumbing check still runs unconditionally) -- so the
    fix is simply to stop excluding it here, not to add special-case
    handling downstream.

    REACHABILITY of the remaining STRICT inversion (`anchor_a > anchor_b`),
    answered in writing at the branch, not only in the task file
    (Architect's ruling, 2026-09-21): every adversarial construction tried
    so far (a swapped-hypothesis pair on static content, a falsely claimed
    jump on genuinely animated content) fails EARLIER -- `anchor_uninformative`
    or `anchors_not_established` respectively -- never by reaching this
    check with both anchors independently validated and STRICTLY swapped.
    **Answer (b) for the strict form:
    reachable in principle, no natural construction found.** A future
    census reading a permanent zero on `cross_sweep_refuted` should find
    this paragraph, not have to re-derive it: `_anchor_search`'s two calls
    are independent (backward from the bracket's low edge, forward from
    its high edge) and nothing in their construction PROVES `anchor_a <=
    anchor_b` always holds, so removing this check would trade a
    proven-present, rarely-firing guard for an unproven invariant.
    '''
    if anchor_a > anchor_b:
        return True, f"anchor_a={anchor_a} > anchor_b={anchor_b}"
    return False, None


def locate_scene_anchors(master_path, candidate_path, fps_num, fps_den,
                         bracket_low_ms, bracket_high_ms,
                         offset_before_ms, offset_after_ms,
                         step_ms=None, quantum_ms=None,
                         scene_search_window_sec=None, debug=False):
    '''Entry point for an INTERIOR bracket. Mirrors
    `frame_compare.locate_bracket_boundary`'s call shape and decline
    payload (`{"declined": True, "reason": ..., "evidence": ...}`) for
    call-site symmetry -- head/tail edges are a separate, later increment
    (reported, not silently skipped: see this mission's task file).

    `offset_before_ms`/`offset_after_ms` are the TWO HYPOTHESES the
    caller's adjacent plan segments already carry (same convention as
    `locate_bracket_boundary`) -- Stage 1's own coarse measurement of each
    side, not something this function discovers. What this function finds
    that Stage 1's coarse bracket does not: the EXACT frame each hypothesis
    stops applying at, and whether the interior gap represents an
    ordinary offset change, a deletion, an ADDITION (SPEC_ZONE_A S4h's
    insertion rule), or a still-image degenerate case.

    `step_ms`/`quantum_ms`: the locator's OWN, independently measured audio
    step for this bracket and its quantum -- same fields, same convention,
    as `frame_compare.locate_bracket_boundary`'s own parameters of the
    same name (threaded through unchanged by the caller). Used by
    `_check_step_plumbing` below -- a rounding/pipe-integrity assert, NOT
    corroboration of WHERE the anchors sit (Architect's ruling,
    2026-09-21, correcting the 2026-09-17 point (i) this module first
    implemented literally: see `_check_anchor_distinctive` for what
    actually corroborates an anchor).

    Always returns a dict, `declined` True or False, never neither.
    '''
    fps_num = int(fps_num)
    fps_den = int(fps_den)
    if fps_num <= 0 or fps_den <= 0:
        return {"declined": True, "reason": "grid_unmeasured",
               "evidence": f"fps_num={fps_num} fps_den={fps_den}"}

    if bracket_high_ms <= bracket_low_ms:
        return {"declined": True, "reason": "empty_bracket",
               "evidence": f"[{bracket_low_ms},{bracket_high_ms}] ms"}

    window_sec = (scene_search_window_sec if scene_search_window_sec is not None
                 else _scene_anchor_config())
    frame_ms = 1000.0 * fps_den / fps_num
    window_frames = int(round((window_sec * 1000.0) / frame_ms)) if window_sec else 0

    # VIABILITY FLOOR (Lead's ruling, 2026-09-21), a NAMED decline BEFORE
    # PySceneDetect is ever asked to look -- distinct from
    # `no_scene_change_in_window` ("I looked and found nothing"). Same
    # shape as this morning's `pal_saturation_screen` fix: a bound that
    # could go non-positive at a short enough setting makes a guard that
    # could never NOT fire. Below this floor the window cannot even
    # structurally hold the >= MIN_VALIDATION_FRAMES margin either
    # direction needs to validate an anchor at all.
    if window_sec is None or window_sec <= 0 or window_frames < MIN_VALIDATION_FRAMES:
        return {"declined": True, "reason": "search_window_unviable",
               "evidence": f"scene_search_window_sec={window_sec} -> "
                          f"{window_frames} frames, needs >= "
                          f"{MIN_VALIDATION_FRAMES}"}

    comparer = FrameComparer(master_path, candidate_path,
                             bracket_low_ms / 1000.0, bracket_high_ms / 1000.0,
                             fps_num, fps_den, debug=debug)
    m_bracket_first = comparer._frame_index(bracket_low_ms / 1000.0)
    m_bracket_last = comparer._frame_index(bracket_high_ms / 1000.0)

    before_shift = _nominal_shift_frames(offset_before_ms, fps_num, fps_den)
    after_shift = _nominal_shift_frames(offset_after_ms, fps_num, fps_den)

    m_win_start = max(0, m_bracket_first - window_frames)
    m_win_end = m_bracket_last + window_frames
    # Candidate window generously covers BOTH offset hypotheses plus the
    # same search margin -- this is seed generation only (see
    # `_anchor_search`'s union-not-agreement note), so generosity here
    # costs decode time, never correctness.
    c_win_start = max(0, m_win_start + min(before_shift, after_shift) - window_frames)
    c_win_end = m_win_end + max(before_shift, after_shift) + window_frames

    m_start_s = float(m_win_start * fps_den / fps_num)
    m_dur_s = float((m_win_end - m_win_start) * fps_den / fps_num)
    c_start_s = float(c_win_start * fps_den / fps_num)
    c_dur_s = float((c_win_end - c_win_start) * fps_den / fps_num)

    m_base, m_hashes = _extract_hashes(comparer, master_path, m_start_s, m_dur_s)
    c_base, c_hashes = _extract_hashes(comparer, candidate_path, c_start_s, c_dur_s)
    if not m_hashes or not c_hashes:
        return {"declined": True, "reason": "frames_unextractable",
               "evidence": f"master_frames={len(m_hashes)} "
                          f"candidate_frames={len(c_hashes)}"}

    threshold = ANCHOR_HAMMING_THRESHOLD_DEFAULT
    cd_threshold = CONTENT_DETECTOR_THRESHOLD_DEFAULT

    master_cuts = _scene_cut_frames(master_path, m_win_start,
                                    m_win_end - m_win_start, cd_threshold, debug)
    candidate_cuts = _scene_cut_frames(candidate_path, c_win_start,
                                       c_win_end - c_win_start, cd_threshold, debug)

    # Anchor A: search BACKWARD from the bracket's own low edge, closest
    # seed first. Seeds: the bracket edge itself (the common, cheap case:
    # Stage 1's bracket already sits right at the true boundary), then
    # master's own cuts before it, then candidate's cuts (translated to
    # master coordinates under the BEFORE hypothesis) before it.
    a_seeds_master = sorted(
        {m_bracket_first}
        | {f for f in master_cuts if f <= m_bracket_first}
        | {f - before_shift for f in candidate_cuts if f - before_shift <= m_bracket_first},
        reverse=True)
    anchor_a, anchor_a_reason = _anchor_search(
        m_hashes, m_base, c_hashes, c_base, a_seeds_master,
        before_shift, "backward", threshold)

    # Anchor B: search FORWARD from the bracket's own high edge, symmetric,
    # under the AFTER hypothesis.
    b_seeds_master = sorted(
        {m_bracket_last}
        | {f for f in master_cuts if f >= m_bracket_last}
        | {f - after_shift for f in candidate_cuts if f - after_shift >= m_bracket_last})
    anchor_b, anchor_b_reason = _anchor_search(
        m_hashes, m_base, c_hashes, c_base, b_seeds_master,
        after_shift, "forward", threshold)

    if anchor_a is None or anchor_b is None:
        # THREE DISTINCT FACTS, per the Architect's ruling, 2026-09-21:
        # no seed ever validated at all (`anchor_a_reason`/`anchor_b_reason`
        # is None), a seed validated but every one failed the
        # distinctiveness probe (`_anchor_search` set a reason), or one
        # side succeeded and the other did not. `anchor_uninformative`
        # names the middle case explicitly rather than folding a MEASURED,
        # content-based refutation into the same bucket as "nothing to
        # try" -- claim exactly what was measured (Architect's own
        # instruction on which token to use).
        if anchor_a_reason or anchor_b_reason:
            return {"declined": True, "reason": "anchor_uninformative",
                   "evidence": f"anchor_a={anchor_a} anchor_b={anchor_b} "
                              f"a_reason={anchor_a_reason} "
                              f"b_reason={anchor_b_reason}"}
        return {"declined": True, "reason": "anchors_not_established",
               "evidence": f"anchor_a={anchor_a} anchor_b={anchor_b} "
                          f"master_cuts={len(master_cuts)} "
                          f"candidate_cuts={len(candidate_cuts)}"}

    ordering_refuted, ordering_evidence = _check_anchor_ordering(anchor_a, anchor_b)
    if ordering_refuted:
        return {"declined": True, "reason": "cross_sweep_refuted",
               "evidence": ordering_evidence}

    # CROSS-SWEEP: forward from A under before_shift, backward from B
    # under after_shift, each capped at the other anchor so neither sweep
    # can run past its partner.
    split_start_master = anchor_a
    for m_frame in range(anchor_a, anchor_b):
        c_frame = m_frame + before_shift
        if _frames_match(m_hashes, m_base, m_frame, c_hashes, c_base, c_frame,
                         threshold) is True:
            split_start_master = m_frame + 1
        else:
            break
    split_start_candidate = split_start_master + before_shift

    split_end_master = anchor_b
    for m_frame in range(anchor_b - 1, anchor_a - 1, -1):
        c_frame = m_frame + after_shift
        if _frames_match(m_hashes, m_base, m_frame, c_hashes, c_base, c_frame,
                         threshold) is True:
            split_end_master = m_frame
        else:
            break
    split_end_candidate = split_end_master + after_shift

    if split_end_master < split_start_master:
        # The two sweeps crossed -- forward-from-A matched further into
        # the bracket than backward-from-B did, meaning there is no
        # unmatched interior left for either hypothesis to own. This is
        # the STILL-IMAGE degenerate case the owner's spec names
        # explicitly (S4.3, "the sweep may land with split_start =
        # Anchor-B position"): collapse both to the SAME point rather
        # than reporting a negative-length interior.
        split_start_master = split_end_master = anchor_b
        split_start_candidate = split_start_master + before_shift
        split_end_candidate = split_end_master + after_shift

    length_master = split_end_master - split_start_master
    length_candidate = split_end_candidate - split_start_candidate

    if length_candidate > length_master:
        net_kind = "addition"
    elif length_candidate < length_master:
        net_kind = "deletion"
    else:
        net_kind = "still_image" if length_master == 0 else "ordinary"

    # THE REAL CORROBORATION ALREADY HAPPENED, per anchor, inside
    # `_anchor_search` -> `_check_anchor_distinctive`, before either
    # anchor was accepted (Architect's ruling, 2026-09-21). What follows
    # is a SEPARATE, ADDITIONAL, low-cost plumbing check -- not a second
    # corroboration pass. Applied unconditionally (WRITE_ZONES S4: "must
    # not be conditioned on a parameter") -- an "ordinary" zero-delta
    # result is cheap to check too and gets no exemption for looking
    # harmless.
    delta_frames = length_candidate - length_master  # RULE 8's own sign: + = candidate holds more
    plumbing_ok, plumbing_evidence = _check_step_plumbing(
        delta_frames, frame_ms, step_ms, quantum_ms)
    if step_ms is None or quantum_ms is None:
        # REACHABLE IN PRODUCTION (Lead's measurement, 2026-09-21): 15 of
        # the quantum-bearing lines in a 6-hour live window carried
        # `quantum_ms=None`, over artefacts deduplicated by content --
        # a first reading said 23 by counting mirrored copies of one
        # artefact twice, corrected the same day. A path is not a case.
        # The correction does not disturb the claim: the branch is fed by
        # real input either way -- this branch is fed by real input today,
        # unlike the disagreement branch below. Producer:
        # `change_point_locator.py` omits `quantum_ms` on the plan when it
        # has no plan-level audio quantum to report for the pair (see that
        # module's own field for which paths leave it unset); traced no
        # further here -- that module is not this mission's to edit.
        return {"declined": True, "reason": "anchor_step_unavailable",
               "evidence": plumbing_evidence}
    if not plumbing_ok:
        # RETIRED NAME `anchor_step_uncorroborated` (measured dead: fires
        # only below ~7.75 fps, since `step_ms` and the offset hypotheses
        # are built from the same two plateau means in production --
        # `change_point_locator.py:2269` vs `:2333`/`:2426`). Renamed so
        # nothing downstream mistakes a plumbing disagreement for a
        # content-based refutation.
        return {"declined": True, "reason": "anchor_step_inconsistent",
               "evidence": plumbing_evidence}

    return {
        "declined": False,
        "grid": {"num": fps_num, "den": fps_den},
        "method": "scene_anchor_bidirectional",
        "anchor_a_frame": anchor_a,
        "anchor_b_frame": anchor_b,
        "master_start_frame": split_start_master,
        "master_end_frame": split_end_master,
        "candidate_start_frame": split_start_candidate,
        "candidate_end_frame": split_end_candidate,
        "net_kind": net_kind,
        "frames_to_cut": max(0, length_candidate - length_master),
        "frames_to_fill": max(0, length_master - length_candidate),
        "derived_ms": {
            "master_start_ms": f"{round(float(_exact_ms_from_frame(split_start_master, fps_num, fps_den)), 2)}",
            "master_end_ms": f"{round(float(_exact_ms_from_frame(split_end_master, fps_num, fps_den)), 2)}",
        },
        "evidence": (f"anchor_a={anchor_a} anchor_b={anchor_b} "
                    f"master_cuts={len(master_cuts)} "
                    f"candidate_cuts={len(candidate_cuts)} "
                    f"length_master={length_master} "
                    f"length_candidate={length_candidate} "
                    f"{plumbing_evidence}"),
    }
