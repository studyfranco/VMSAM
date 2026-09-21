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

# THE THRESHOLD IS A FLOOR, NOT A FIXED VALUE -- owner's order, verbatim (per
# the addendum naming this dial explicitly, 2026-09-21): "if anchor are not
# the same, ... increase the number [and lower the detector's own bar]."
# Same principle as `VALIDATION_FRAME_LADDER` below (dev-step6-phash): a
# rung that fails to produce usable anchors is not proof the boundary is
# unreachable, it is proof THIS rung's instrument was not sensitive enough.
#
# VALUES ARE MEASURED, NOT GUESSED (dev-step5-scenedetect, 2026-09-21,
# real error-tree media -- Addendum 3/4: no acceptance closes on a
# synthetic fixture, and this constant's own comment above already warned
# against tuning it on n=1 synthetic evidence):
#   27.0  the unchanged default, carried per the comment above -- never
#         retuned on a single case.
#   18.0  clears every real near-miss measured just under the default
#         (26.94 / 26.56 / 26.54, real content_val on two real episodes,
#         n=708 gap-zone frames) with margin, while staying well above the
#         typical real noise floor (median 0.65, p90 4.2 across n=2399
#         real frames sampled).
#   10.0  matches the SYNTHETIC red->yellow miss this file's own top
#         comment already documents (content_val=10.0) -- the floor beyond
#         which this module has no further documented failure case to aim
#         at.
# MEASURED, NOT ASSUMED, TO BE UNEVEN COST (dev-step5-scenedetect, widened
# sample, 2026-09-21): the seed-candidate density a lower rung admits is
# CONTENT-DEPENDENT and can spike hard on a real busy window -- one real
# 50 s window measured 32% of its frames in the [8,18) band against every
# other sampled window an order of magnitude lower (15 real windows, 5
# files, n=9592). Every admitted seed is still pHash-validated before
# acceptance (`_anchor_search`), never trusted outright, so the spike's
# cost is compute, not correctness -- but do not expect a smooth curve
# across files, and see the per-rung log line below, which now carries the
# seed count precisely so this variance is visible in production rather
# than rediscovered (Lead's ruling, 2026-09-21, on this same finding).
CONTENT_DETECTOR_THRESHOLD_LADDER = (CONTENT_DETECTOR_THRESHOLD_DEFAULT, 18.0, 10.0)

# ">= 3 consecutive identical frames", owner's spec, both for Anchor A/B
# validation and as the viability floor below.
MIN_VALIDATION_FRAMES = 3

# THE 3 IS A FLOOR, NOT A FIXED VALUE (owner, dev-step6-phash mission
# BRIEF.md, 2026-09-21, verbatim: "if anchor are not the same, ... the 3
# frames is not enought and increase the number"). Read literally: a
# candidate anchor whose MIN_VALIDATION_FRAMES-frame match is AMBIGUOUS --
# it also matches under a competing shift hypothesis, i.e. it is NOT
# uniquely "the same" -- is not thrown away outright. The window is
# WIDENED at that SAME seed and re-checked: a longer consecutive run is
# less likely to still be self-similar under a wrong hypothesis (the
# owner's "mouths moving over a fixed background" is exactly a SHORT
# self-similar run; a longer one is a stronger claim about the content).
# `_anchor_search` below walks this ladder PER SEED -- re-running both
# `_validate_anchor` and `_check_anchor_distinctive` at each rung -- before
# giving up on that seed and moving to the next one.
#
# THE RUNG VALUES AND THE CAP (13) ARE NOT THE OWNER'S. He named the
# principle (floor, not fixed) and the trigger (an ambiguous match), never
# a schedule. ATTRIBUTION (Lead's ruling, 2026-09-21, ratifying this after
# measuring it): owner gave the principle, this seat (dev-step6-phash)
# derived the numbers, the Lead ratified them -- kept explicit here so the
# record never lets the schedule quietly acquire the owner's authority.
# Each rung costs only comparisons against hashes ALREADY extracted (no
# new ffmpeg call, no new probe class -- the same reuse
# `_check_anchor_distinctive` already relies on), so growth is cheap; the
# cap is chosen because 13 frames is already ~0.5 s at 24 fps -- past that,
# content still indifferent to which shift reads it is not something a few
# more frames will resolve, and trying the next SEED costs less than
# growing this one further (see `_anchor_search`'s own union-of-seeds
# design). UNMEASURED: how often production content actually needs a rung
# past 3 -- covered by the per-rung log line `_anchor_search` now writes.
#
# ORDERING WITH THE OUTER (WINDOW) LADDER -- DELIBERATE, DO NOT REORDER
# (Lead's measurement, 2026-09-21, on dev-step4-extract's
# WINDOW_LADDER_* below): this ladder is CHEAP (re-validates hashes already
# extracted) and sits INSIDE `_anchor_search`, so it exhausts fully before
# dev-step4-extract's EXPENSIVE window-widening retry (re-extracts frames,
# re-runs PySceneDetect, doubles 10s->20s->40s) ever fires one level up.
# Worst case is bounded (5 inner rungs x 3 outer rungs = 15 anchor
# searches per bracket) precisely because the many, cheap rungs sit on
# THIS dial and the few, expensive rungs sit on the OTHER one. Swapping
# which ladder is inner and which is outer turns a bounded retry into
# something that re-extracts video on every inner rung -- the difference
# between a ladder and a hang.
VALIDATION_FRAME_LADDER = (MIN_VALIDATION_FRAMES, 4, 6, 9, 13)

# Same Hamming vocabulary as frame_compare.py's own boundary validation
# (BOUNDARY_VALIDATION_HAMMING_THRESHOLD_DEFAULT) -- a DIFFERENT constant,
# not imported, because this module's validation question ("are these two
# frames the SAME picture") is stricter than F1's ("is this boundary
# roughly where we think"): anchor frames must be near-identical, not
# merely close enough to confirm a shift.
ANCHOR_HAMMING_THRESHOLD_DEFAULT = 6

# NAMED, NOT AN ANONYMOUS REPEATED SUBTRACTION (Lead's ruling, 2026-09-21,
# on a measured finding: the candidate window used to get `window_frames`
# subtracted/added a SECOND time on top of the master window's own
# margin, with no constant marking it as deliberate). The candidate
# window carries this MANY TIMES the master's own search margin, beyond
# whatever the two offset hypotheses already require -- because a seed
# MISSED here is a decline (`anchors_not_established`), while a seed
# window that is too generous only costs decode time. The asymmetry is
# real and is NOT what the owner's diagram states (`t_cut +/- 10 s`,
# one number) -- it is deliberate, so it gets a name instead of staying
# an unmarked doubling a reader would have to reverse-engineer.
CANDIDATE_SEED_MARGIN_MULTIPLIER = 2

# THE LADDER (owner's order, 2026-09-21, verbatim: "if anchor are not the
# same, the 10 seconds is not enought ... and increase the number"). A
# single rung that fails to seed BOTH anchors is not evidence the
# boundary is unreachable -- it is evidence the search was too narrow.
# Doubling is the smallest growth law that halves the rung count needed
# to reach a given ceiling versus a smaller step, and it is not tuned on
# this module's own fixtures -- it is the same shape as
# `CANDIDATE_SEED_MARGIN_MULTIPLIER` above, re-used rather than a second,
# arbitrary constant.
WINDOW_LADDER_GROWTH_FACTOR = 2.0
# Hard ceiling (Lead's ruling, 2026-09-21: "a hard ceiling and a named
# decline at the ceiling"). Rung 0 is the configured `scene_search_window_sec`
# (10 s as committed); rung 1 doubles it (20 s); rung 2 doubles again
# (40 s). Three rungs, not an unbounded loop: every rung re-extracts and
# re-runs PySceneDetect on BOTH files, so the ceiling is a cost bound as
# much as a safety one. Raise it only against a corpus case that measured
# rung 2 as insufficient -- not pre-emptively.
WINDOW_LADDER_MAX_RUNGS = 3

# Which decline reasons mean "the window was the limiting factor" and are
# therefore worth a wider rung, versus a reason widening cannot fix
# (Architect's own vocabulary, this file's `locate_scene_anchors` decline
# payloads): `grid_unmeasured`/`empty_bracket` are input problems a wider
# window does not touch; `frames_unextractable` is an I/O failure a wider
# window only makes larger; `cross_sweep_refuted`/`anchor_step_unavailable`/
# `anchor_step_inconsistent` are established-anchor failures downstream of
# the window entirely (dev-step7-sweep's own domain, confirmed with them
# directly, 2026-09-21: their logic sits inside this same call, strictly
# after anchors are already found, and `cross_sweep_refuted` is explicitly
# EXCLUDED here so their own escalation, if the Lead rules for one, is not
# pre-empted by mine).
#
# `search_window_unviable` VS `search_window_too_narrow` -- SPLIT, NOT ONE
# TOKEN (Lead's ruling, 2026-09-21, on a real disagreement between this
# seat and dev-step7-sweep that turned out to be two people right about
# two different shapes hiding under one name): the pre-split decline site
# tested THREE conditions under ONE reason -- `window_sec is None`,
# `window_sec <= 0`, and `window_frames < MIN_VALIDATION_FRAMES` -- but
# only the third is a "the window was too small" claim. The first two are
# "the configuration cannot be interpreted as a window at all," and
# doubling does not help either: `None * WINDOW_LADDER_GROWTH_FACTOR`
# does not evaluate, `0 * WINDOW_LADDER_GROWTH_FACTOR` is still `0`, and a
# negative value only moves further from viable. So:
#   `search_window_unviable`    None or <= 0 -- TERMINAL, config problem
#   `search_window_too_narrow`  valid, positive, under the frame floor --
#                               RETRYABLE, a wider rung can plausibly clear it
# Same defect this campaign has now found five times under five names
# (could-not-measure sharing a return value with measured-and-found-
# nothing) -- here it was a decline REASON standing for two different
# claims rather than a threshold or a count.
#
# `anchors_not_established` (no seed validated at all) and
# `anchor_uninformative` (a seed validated but every one failed the
# distinctiveness probe) are retried for the same reason as
# `search_window_too_narrow`: both are exactly the shape of "nothing to
# work with in this window, or only self-similar content in it," which a
# wider window can plausibly change.
WINDOW_LADDER_RETRYABLE_REASONS = frozenset(
    {"anchors_not_established", "anchor_uninformative", "search_window_too_narrow"})


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
    first.

    Returns `(cuts, None)` on success -- `cuts` may legitimately be an
    EMPTY list, meaning the detector RAN over the window and found no
    interior scene change, a real result. Returns `(None, reason)` when
    ContentDetector itself never produced one (unreadable region, file too
    short, a PySceneDetect exception) -- `reason` is a NAMED token, never
    the bare `[]` an honest empty result also returns (dev-step5-scenedetect
    finding, 2026-09-21: "an empty result and a failed detector are
    different claims and must not share a return value" -- BRIEF_COMMON's
    fifth rule, one level down in this module). A caller still falls back
    to the bracket edge either way -- that seed is tried regardless of
    what this function returns -- so this distinction changes no anchor
    decision by itself; it exists so a decline's evidence, and any future
    census over it, can say WHICH of the two happened instead of reading a
    silent zero as "no shots" when the instrument may never have run.
    '''
    if n_frames <= 0:
        return [], None
    try:
        video = open_video(path)
        if start_frame > 0:
            video.seek(start_frame)
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=threshold))
        sm.detect_scenes(video, duration=n_frames)
        scene_list = sm.get_scene_list()
        if len(scene_list) < 2:
            return [], None
        return [scene.frame_num for scene, _ in scene_list[1:]], None
    except Exception as exc:
        reason = f"scene_detector_failed:{type(exc).__name__}"
        if debug:
            tools.logs.append(f"scene_anchor: PySceneDetect failed on {path} "
                              f"[{start_frame},{start_frame + n_frames}): "
                              f"{reason}\n")
        return None, reason


def _frames_match(m_hashes, m_base, m_frame, c_hashes, c_base, c_frame,
                  threshold):
    mi, ci = m_frame - m_base, c_frame - c_base
    if not (0 <= mi < len(m_hashes)) or not (0 <= ci < len(c_hashes)):
        return None  # unreadable, never treated as either match or mismatch
    return FrameComparer._popcount64(m_hashes[mi] ^ c_hashes[ci]) <= threshold


def _validate_anchor(m_hashes, m_base, c_hashes, c_base, m_seed, shift_frames,
                     direction, threshold, n_frames=MIN_VALIDATION_FRAMES):
    '''Owner's ">= 3 consecutive identical frames" check, one direction at
    a time. `direction="forward"`: validate [m_seed, m_seed+N) against the
    candidate under `shift_frames`. `direction="backward"`: validate
    [m_seed-N, m_seed) instead. Which anchor uses which direction is
    `_anchor_search`'s own call (Anchor B validates forward from its seed,
    Anchor A validates backward from its seed) -- this function only
    performs the check it is told to, in the direction it is given.
    Returns True only if EVERY one of the `n_frames` pairs is both readable
    and matching -- an unreadable frame is not a pass by omission.

    `n_frames` (default `MIN_VALIDATION_FRAMES`, the owner's floor): the
    RUNG this call validates at. `_anchor_search` walks
    `VALIDATION_FRAME_LADDER` through this same parameter; this function
    itself knows nothing about escalation, only about the one width it was
    asked to check -- same separation of concerns as `direction`.
    '''
    frames = (range(m_seed, m_seed + n_frames) if direction == "forward"
             else range(m_seed - n_frames, m_seed))
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
    Returns `(seed, None, n_frames_used)` on success, `(None, reason,
    n_frames_used)` on failure, where `reason` is `None` if no seed ever
    validated at all, or the FIRST rejected-as-uninformative seed's
    evidence if at least one did but failed the distinctiveness probe at
    every rung it reached. `n_frames_used` is the rung the returned
    seed/reason was produced at -- on success, the smallest rung that was
    both valid and distinctive; on an uninformative decline, the highest
    rung THAT SEED reached before either escalation exhausted the ladder
    or a wider window stopped validating at all; `None` when nothing ever
    validated even at the floor. A PLAIN LOCAL VARIABLE, not module
    state -- CORRECTED IN AN EARLIER REVISION (Lead's finding, 2026-09-21,
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
    two-tuple, and now a three-tuple (dev-step6-phash mission, 2026-09-21,
    widening the return again for the same reason -- the ladder's own rung
    is exactly the kind of fact a caller building evidence needs, and a
    module-level counter would repeat the module-state defect this
    docstring already records once).

    ESCALATION (owner: "if anchor are not the same, ... the 3 frames is
    not enought and increase the number" -- BRIEF.md, dev-step6-phash
    mission, 2026-09-21; rung schedule `VALIDATION_FRAME_LADDER`, this
    seat's own derivation, Lead-ratified, see that constant's comment).
    PER SEED, not across seeds: walk the ladder from the floor. If a rung
    fails to VALIDATE (some frame in the wider window does not match),
    STOP escalating this seed and move to the next one -- a stricter,
    longer requirement failing is not evidence a few more frames would
    help; it is evidence this seed's run is genuinely short. If a rung
    validates but is UNINFORMATIVE (matches a competing shift too, i.e.
    the anchor is not yet uniquely "the same"), widen to the next rung and
    re-run BOTH checks at the SAME seed -- a longer consecutive run is
    less likely to still be self-similar under the wrong hypothesis. The
    first rung that is both valid and distinctive wins; exhausting the
    ladder without resolving falls through to the SAME `anchor_uninformative`
    evidence as today, just possibly reached after climbing further.
    LOGGED PER RUNG (owner's order, "log all step"): every `(seed,
    n_frames)` attempt this function makes, whether it validated, and
    whether it was distinctive -- a ladder that silently succeeds at the
    floor must read identically in the log to one that never escalated,
    and a ladder that climbs all the way to the cap and still fails must
    be visible as having tried, not as having declined outright.
    '''
    last_uninformative_reason = None
    last_uninformative_n_frames = None
    for seed in seeds:
        seed_reason = None
        seed_n_frames = None
        for n_frames in VALIDATION_FRAME_LADDER:
            validated = _validate_anchor(m_hashes, m_base, c_hashes, c_base,
                                         seed, shift_frames, direction,
                                         threshold, n_frames)
            if not validated:
                tools.logs.append(
                    f"scene_anchor: anchor_rung direction={direction} "
                    f"seed={seed} n_frames={n_frames} validated=False "
                    f"distinctive=n/a\n")
                break
            distinctive, why_not = _check_anchor_distinctive(
                m_hashes, m_base, c_hashes, c_base, seed, shift_frames,
                direction, threshold, n_frames)
            tools.logs.append(
                f"scene_anchor: anchor_rung direction={direction} "
                f"seed={seed} n_frames={n_frames} validated=True "
                f"distinctive={distinctive}\n")
            if distinctive:
                return seed, None, n_frames
            # MATCHED BUT UNINFORMATIVE AT THIS RUNG -- escalate to the
            # next rung at the SAME seed before giving up on it.
            seed_reason, seed_n_frames = why_not, n_frames
        if seed_reason is not None and last_uninformative_reason is None:
            # Remember the reason (and rung) from the FIRST such SEED only
            # (unchanged from before the ladder existed) -- the evidence a
            # caller wants is "why did the closest/best candidate fail",
            # not the last one tried.
            last_uninformative_reason = seed_reason
            last_uninformative_n_frames = seed_n_frames
    return None, last_uninformative_reason, last_uninformative_n_frames


ANCHOR_DISTINCTIVENESS_PROBE_FRAMES = (4, 8)
# EXCLUDE the immediate neighbourhood of the true shift -- natural
# continuity (adjacent real frames are often somewhat alike even in
# distinctive content) is not the failure mode this guard exists to
# catch; a self-similar RUN is. Probing only at well-separated deltas
# means a positive result (still matches) is a real signal, not noise
# from nearby-frame similarity.


def _check_anchor_distinctive(m_hashes, m_base, c_hashes, c_base, m_seed,
                              shift_frames, direction, threshold,
                              n_frames=MIN_VALIDATION_FRAMES):
    '''ANCHOR DISTINCTIVENESS (Architect's ruling, 2026-09-21, replacing
    point (i) of the 2026-09-17 ruling): "an anchor match is evidence only
    if the local content could have refuted it." Anchors answer WHERE;
    offsets answer HOW MUCH; no offset-derived quantity can corroborate
    WHERE (see `_check_step_plumbing` below for what offset comparison
    actually is, now that it no longer claims to be this). A match inside
    a self-similar run (the owner's "mouths moving over a fixed
    background") is unfalsifiable and therefore not evidence, whatever
    hypothesis produced it.

    THE TEST: the same `n_frames`-wide window that validated at
    `shift_frames` is re-checked at ALTERNATIVE, well-separated shifts
    (`ANCHOR_DISTINCTIVENESS_PROBE_FRAMES` frames away, both directions),
    AT THE SAME `n_frames` (`_anchor_search`'s ladder widens both checks
    together -- escalating only the probe while leaving the original match
    at the floor would compare windows of different widths, which proves
    nothing). If it ALSO matches at ANY of those -- meaning the
    surrounding content is indifferent to which position it is read from
    -- the match carries no information about WHERE the true boundary is,
    and the anchor is UNINFORMATIVE at this rung. Uses frames already
    extracted into `m_hashes`/`c_hashes` (no new probe class, no new
    ffmpeg call): only the comparison shifts, not the extraction -- true
    at any `n_frames`, since escalating never reads outside the window
    `locate_scene_anchors` already extracted for the whole bracket search.

    Returns `(True, None)` when the match survives every probe (a real
    anchor), `(False, evidence_str)` on the first probe that ALSO matches
    (uninformative at this rung -- the caller may still escalate `n_frames`
    and retry).
    '''
    for delta in ANCHOR_DISTINCTIVENESS_PROBE_FRAMES:
        for probe_shift in (shift_frames + delta, shift_frames - delta):
            if _validate_anchor(m_hashes, m_base, c_hashes, c_base, m_seed,
                               probe_shift, direction, threshold, n_frames):
                return False, (f"seed={m_seed} also matches at shift="
                              f"{probe_shift} (true shift={shift_frames}, "
                              f"delta={probe_shift - shift_frames}, "
                              f"n_frames={n_frames}) -- self-similar "
                              f"content, uninformative")
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


# DECLINE TAXONOMY (dev-step7-sweep, 2026-09-21, per the Lead's ruling on
# the owner's ladder order: "if anchor are not the same ... increase the
# number" -- a failed rung must know whether widening could plausibly
# change its own answer, or whether it is re-running an already-fixed
# computation at N times the cost). Covers every `reason`
# `_locate_scene_anchors_at_window` can return. dev-step4-extract's own
# `WINDOW_LADDER_RETRYABLE_REASONS` above is value-identical to the
# RETRYABLE set below (confirmed directly, 2026-09-21) -- this function is
# offered as the single source of truth for whoever's ladder consumes it
# next; not force-adopted into their already-shipped loop unasked.
#
#   RETRYABLE          widening scene_search_window_sec (more readable
#                       frames, more candidate seeds) could plausibly
#                       change this outcome.
#   TERMINAL            MEASURED, not assumed, to be independent of this
#                       module's own window parameter -- see the per-reason
#                       note below. Retrying is not a second measurement.
#   CANNOT_DETERMINE    no reason of this module's own falls here today;
#                       kept so an unrecognised future reason fails
#                       EXPLICITLY rather than being silently folded into
#                       either bucket (BRIEF_COMMON rule 5's distinction,
#                       one level up: "unknown" and "known-not-retryable"
#                       are different facts).
DECLINE_RETRY_CLASS = {
    # Pre-anchor input/config defects. A wider window cannot repair a bad
    # frame-rate grid or a Stage-1 bracket that is already empty; listed
    # for completeness of this function's whole reason vocabulary.
    "grid_unmeasured": "TERMINAL",
    "empty_bracket": "TERMINAL",
    # CORRECTED (dev-step7-sweep, 2026-09-21, catching an error
    # dev-step4-extract's own read found): this single token covers THREE
    # different shapes at the decline site
    # (`_locate_scene_anchors_at_window`, `window_sec is None or
    # window_sec <= 0 or window_frames < MIN_VALIDATION_FRAMES`), and only
    # ONE of them is fixable by widening. `window_sec is None` (an
    # unparseable config value, `_scene_anchor_config`'s own contract) and
    # `window_sec <= 0` (a literal zero/negative setting) are config
    # DEFECTS, not narrow searches -- doubling `None` stays `None`,
    # doubling a non-positive number never crosses into positive. Only
    # `window_sec` being a valid, small POSITIVE value whose
    # `window_frames` still falls under the floor is genuinely retryable.
    # Marking the WHOLE token RETRYABLE (an earlier revision of this
    # table did) means a malformed-config decline burns every rung before
    # failing, AND its ceiling-reached report BURIES the sharper "someone
    # typo'd config.ini" diagnosis one level down in the evidence string
    # instead of surfacing it as the top-level reason -- the same
    # token-conflation defect BLANK LAW / INSTRUMENT SCOPE LAW name
    # elsewhere in tools/RULINGS_IN_FORCE.md. RULED (Lead, 2026-09-21):
    # split into two tokens, see `WINDOW_LADDER_RETRYABLE_REASONS`'s own
    # comment above for the full ruling text -- `search_window_unviable`
    # now means None/<=0 ONLY (config defect, TERMINAL) and
    # `search_window_too_narrow` means valid-positive-but-under-the-floor
    # (RETRYABLE). The decline site below is the code half of that ruling
    # (dev-step7-sweep, same pass: the ruling had landed in the comments
    # and the frozenset but not yet in the `if` that actually returns the
    # reason -- `search_window_too_narrow` was unreachable dead vocabulary
    # until this edit).
    "search_window_unviable": "TERMINAL",
    "search_window_too_narrow": "RETRYABLE",
    # ffmpeg produced no frames at all on one side. A wider window reads a
    # LARGER span of the SAME unreadable source (same path, same codec,
    # same failure) -- there is no reason more of the same input becomes
    # readable. TERMINAL, not RETRYABLE-but-unlikely.
    "frames_unextractable": "TERMINAL",
    # Anchor establishment: no seed validated at all, or every validated
    # seed failed the distinctiveness probe (at every rung of the OTHER,
    # inner ladder -- dev-step6-phash's VALIDATION_FRAME_LADDER already
    # widens the frame count per seed before this reason is ever reached).
    # A wider window changes BOTH the candidate seed set (more
    # PySceneDetect cuts in range) and the extracted hash range (more
    # distinctiveness probes become readable instead of returning `None`)
    # -- the two reasons the owner's order names directly ("if anchor are
    # not the same, the 10 seconds is not enough").
    "anchors_not_established": "RETRYABLE",
    "anchor_uninformative": "RETRYABLE",
    # DOWNSTREAM OF ANCHORS -- dev-step7-sweep's own domain. All three
    # PROVEN TERMINAL below, not merely judged unlikely to help:
    #
    # `cross_sweep_refuted`: fires ONLY when `_check_anchor_ordering` is
    # reached, which requires BOTH anchor_a and anchor_b to already be
    # non-None -- i.e. `_anchor_search` already returned successfully for
    # both, on its CLOSEST-seed-first, first-match-wins ordering
    # (docstring above, "Try each seed ... return the first that
    # validates"). PROVEN UNREACHABLE, not merely rare, at the sole call
    # site (grepped: one call, right here): `a_seeds_master`'s every
    # member is filtered `<= m_bracket_first` and `b_seeds_master`'s every
    # member is filtered `>= m_bracket_last` (the two seed-construction
    # blocks immediately above this function's own call site), so
    # whichever seed each search returns, `anchor_a <= m_bracket_first <=
    # m_bracket_last <= anchor_b` ALWAYS holds (`_frame_index` is
    # monotonic non-decreasing, and `bracket_high_ms > bracket_low_ms` is
    # already guaranteed by the `empty_bracket` decline above) --
    # `anchor_a > anchor_b` cannot occur given the current seed filters,
    # independent of window size, content, or anything else. CORRECTS the
    # Architect's 2026-09-21 ruling on this same branch, above ("reachable
    # in principle, no natural construction found") -- that reading did
    # not account for the seed-set filters themselves proving the bound;
    # reported to the Lead/Architect as a measured refutation, not acted
    # on here (the guard stays: it is not an invariant of
    # `_check_anchor_ordering` in isolation, only of the CURRENT seed
    # construction that feeds it, so it remains real defense against a
    # future change to that construction). Reproduced directly: the same
    # non-inversion holds under adversarial seed injection at
    # scene_search_window_sec=1.0 and =10.0 on a bracket built to try to
    # force it (dev-step7-sweep lab, T6). Production check, n=59
    # (`scene_anchor_shadow` lines, /config/output, 2026-09-21): zero
    # occurrences of `cross_sweep_refuted`, consistent with the proof.
    "cross_sweep_refuted": "TERMINAL",
    # `anchor_step_unavailable`/`anchor_step_inconsistent`: both come out
    # of `_check_step_plumbing`, which compares `delta_frames` against the
    # LOCATOR's OWN `step_ms`/`quantum_ms` -- fields this function receives
    # as opaque parameters and never computes. PROVEN (not assumed) that
    # `delta_frames` itself cannot be moved by this module's window
    # parameter: by construction, `split_start_candidate = split_start_master
    # + before_shift` and `split_end_candidate = split_end_master +
    # after_shift` ALWAYS (both the ordinary sweep and the crossed-collapse
    # branch preserve this), so
    #   delta_frames = length_candidate - length_master
    #                = (split_end_candidate - split_start_candidate)
    #                  - (split_end_master - split_start_master)
    #                = after_shift - before_shift
    # identically, for EVERY possible anchor_a/anchor_b/sweep outcome --
    # this module's own `_check_step_plumbing` docstring already proves the
    # same identity by 20-case simulation. `step_ms`/`quantum_ms` are
    # supplied unchanged by the caller (`change_point_locator.py`, not this
    # module) and are equally untouched by `scene_search_window_sec`. A
    # wider window changes NEITHER side of the comparison
    # `_check_step_plumbing` makes, so it cannot change whether it agrees.
    "anchor_step_unavailable": "TERMINAL",
    "anchor_step_inconsistent": "TERMINAL",
}


def classify_decline(reason):
    '''RETRYABLE / TERMINAL / CANNOT_DETERMINE for a `reason`
    `locate_scene_anchors` can return -- the predicate a retry ladder built
    OUTSIDE this module (dev-step4-extract's window rungs, which already
    wrap the whole call) should consult before spending another rung:
    widen and retry only on RETRYABLE; stop and decline, named, on
    anything else. Unrecognised reasons return `CANNOT_DETERMINE` rather
    than silently joining either bucket -- an unclassified reason must be
    visible as unclassified, not mistaken for a measured verdict either
    way.
    '''
    return DECLINE_RETRY_CLASS.get(reason, "CANNOT_DETERMINE")


def locate_scene_anchors(master_path, candidate_path, fps_num, fps_den,
                         bracket_low_ms, bracket_high_ms,
                         offset_before_ms, offset_after_ms,
                         step_ms=None, quantum_ms=None,
                         scene_search_window_sec=None, debug=False):
    '''PUBLIC ENTRY POINT -- unchanged call shape (Lead's ruling,
    2026-09-21). THE OUTER RUNG: resolves `scene_search_window_sec` once
    (explicit argument, else `config.ini`), same as before this refactor,
    then calls `_locate_scene_anchors_at_window` up to
    `WINDOW_LADDER_MAX_RUNGS` times, doubling the window each time
    (`WINDOW_LADDER_GROWTH_FACTOR`) -- owner's order, 2026-09-21, verbatim:
    "if anchor are not the same, the 10 seconds is not enought ... and
    increase the number."

    ORDERING WITH THE INNER (PER-SEED) LADDER -- do not invert (Lead's
    measurement, 2026-09-21, on dev-step6-phash's `VALIDATION_FRAME_LADDER`
    inside `_anchor_search`): that ladder is cheap (re-validates hashes
    already extracted) and runs to exhaustion INSIDE every single call this
    loop makes; THIS loop is expensive (re-extracts frames, re-runs
    PySceneDetect on both files) and only escalates once the cheap one has
    already given up. Worst case is bounded at
    `WINDOW_LADDER_MAX_RUNGS * len(VALIDATION_FRAME_LADDER)` anchor
    searches per bracket, not their product with anything else, because
    only these two dials retry at all (dev-step7-sweep confirmed, 2026-09-21,
    directly: their cross-sweep logic sits downstream of anchor
    establishment inside this same call and builds no outer retry of its
    own).

    RETRY BOUNDARY: only a decline whose reason is in
    `WINDOW_LADDER_RETRYABLE_REASONS` -- currently `anchors_not_established`
    (no seed validated at all) and `anchor_uninformative` (a seed validated
    but every one failed the distinctiveness probe) -- is worth a wider
    rung. Every other reason (`grid_unmeasured`, `empty_bracket`,
    `search_window_unviable`, `frames_unextractable`,
    `cross_sweep_refuted`, `anchor_step_unavailable`,
    `anchor_step_inconsistent`) is either a config/input problem a wider
    window cannot fix, or -- `cross_sweep_refuted` -- a failure downstream
    of anchors already being established, which is dev-step7-sweep's own
    domain to escalate or not (PROVISIONAL pending their own classification
    landing, per the Lead: "take their classification when it lands rather
    than freezing your own set").

    CO-ESCALATED SECOND DIAL, SAME LOOP (dev-step5-scenedetect, agreed
    directly with dev-step4-extract, 2026-09-21, per the owner's addendum
    naming "the detector threshold and the shot count" as its own dial):
    `CONTENT_DETECTOR_THRESHOLD_LADDER`, index-matched to this SAME rung
    number, lowers ContentDetector's own sensitivity alongside the widened
    window rather than owning a second loop -- the Lead's "ONE loop, not
    four" instruction applied to a dial whose cost (re-runs PySceneDetect
    on the already-decided window, no re-extraction) sits between this
    loop's (expensive: re-extracts frames) and the inner per-seed ladder's
    (free: reuses hashes already extracted). `search_window_ceiling_reached`
    keeps its name on exhaustion (Lead's ruling: one EVENT, several dials,
    is evidence -- not the same shape as a token conflating two different
    CAUSES) and its evidence now carries both final dial values.

    Exhausting every rung without a match declines
    `search_window_ceiling_reached`, named so a census can tell "the
    ladder ran and lost" from every other decline shape.

    Every rung is logged (`scene_anchor: window_ladder_rung ...`) with both
    dials, the values tried, the resulting seed count on each side (Lead's
    ruling, 2026-09-21: real content's seed density under a lowered
    threshold is measured HETEROGENEOUS -- one real window spiked to 32%
    of its frames against every other sampled window an order of
    magnitude lower -- so this is logged explicitly rather than left to be
    rediscovered from aggregate stats), and whether it matched, so a
    census can read the per-bracket cost of both dials back out without
    re-deriving it from the anchor evidence strings alone.
    '''
    window_sec = (scene_search_window_sec if scene_search_window_sec is not None
                 else _scene_anchor_config())

    result = None
    rung_cd_threshold = CONTENT_DETECTOR_THRESHOLD_LADDER[0]
    for rung in range(WINDOW_LADDER_MAX_RUNGS):
        rung_window_sec = (window_sec if window_sec is None
                           else window_sec * (WINDOW_LADDER_GROWTH_FACTOR ** rung))
        # CO-ESCALATED, NOT A SECOND LOOP (dev-step5-scenedetect, agreed
        # directly with dev-step4-extract, 2026-09-21): index-matched to
        # THIS SAME rung -- widening the window and lowering the detector's
        # own sensitivity are two answers to the same trigger ("anchors do
        # not match"), driven by the one loop that already exists. Clamped
        # rather than IndexError if the two ladders' lengths ever drift
        # apart (they are both 3 today, by agreement, not by a shared
        # constant enforcing it).
        rung_cd_threshold = CONTENT_DETECTOR_THRESHOLD_LADDER[
            min(rung, len(CONTENT_DETECTOR_THRESHOLD_LADDER) - 1)]
        result = _locate_scene_anchors_at_window(
            master_path, candidate_path, fps_num, fps_den,
            bracket_low_ms, bracket_high_ms, offset_before_ms, offset_after_ms,
            rung_window_sec, step_ms=step_ms, quantum_ms=quantum_ms,
            content_detector_threshold=rung_cd_threshold, debug=debug)
        matched = not result["declined"]
        tools.logs.append(
            f"scene_anchor: window_ladder_rung rung={rung} "
            f"dial=window_sec+cd_threshold "
            f"window_sec={rung_window_sec} cd_threshold={rung_cd_threshold} "
            f"master_seed_count={result.get('master_seed_count')} "
            f"candidate_seed_count={result.get('candidate_seed_count')} "
            f"matched={matched} "
            f"reason={result.get('reason')} evidence={result.get('evidence')}\n")
        if matched or result["reason"] not in WINDOW_LADDER_RETRYABLE_REASONS:
            return result

    return {"declined": True, "reason": "search_window_ceiling_reached",
           "evidence": f"rungs_tried={WINDOW_LADDER_MAX_RUNGS} "
                      f"base_window_sec={window_sec} "
                      f"final_window_sec={rung_window_sec} "
                      f"final_cd_threshold={rung_cd_threshold} "
                      f"final_master_seed_count={result.get('master_seed_count')} "
                      f"final_candidate_seed_count={result.get('candidate_seed_count')} "
                      f"last_reason={result.get('reason')} "
                      f"last_evidence={result.get('evidence')}"}


def _locate_scene_anchors_at_window(master_path, candidate_path, fps_num, fps_den,
                                    bracket_low_ms, bracket_high_ms,
                                    offset_before_ms, offset_after_ms,
                                    window_sec, step_ms=None, quantum_ms=None,
                                    content_detector_threshold=CONTENT_DETECTOR_THRESHOLD_DEFAULT,
                                    debug=False):
    '''ONE RUNG of `locate_scene_anchors`'s ladder (below): everything that
    mission originally did at a single, fixed `window_sec`, unchanged
    except that the candidate margin is now the NAMED
    `CANDIDATE_SEED_MARGIN_MULTIPLIER` (Lead's ruling, 2026-09-21) rather
    than an anonymous second subtraction of `window_frames`. `window_sec`
    is REQUIRED here and already concrete -- resolving it from the
    caller's argument or `config.ini`, and deciding whether it is viable
    at all, is `locate_scene_anchors`'s job, once, before any rung; a
    malformed or absent config value is not a per-rung question and must
    not be multiplied by the ladder's growth factor.

    Mirrors `frame_compare.locate_bracket_boundary`'s call shape and
    decline payload (`{"declined": True, "reason": ..., "evidence": ...}`)
    for call-site symmetry -- head/tail edges are a separate, later
    increment (reported, not silently skipped: see this mission's task
    file).

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

    `content_detector_threshold` (dev-step5-scenedetect, 2026-09-21): the
    ContentDetector sensitivity for THIS rung -- co-escalated by the
    caller's outer loop alongside `window_sec`, index-matched, agreed
    directly with dev-step4-extract (both owning dials in the SAME loop,
    no second loop of this function's own). Defaults to
    `CONTENT_DETECTOR_THRESHOLD_DEFAULT` so a direct call (a unit test, a
    caller that has not adopted the ladder) is unaffected.

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

    frame_ms = 1000.0 * fps_den / fps_num

    # CONFIGURATION VS SIZE -- SPLIT, NOT ONE TOKEN (Lead's ruling,
    # 2026-09-21, on a real disagreement this seat and dev-step7-sweep
    # each got half right: see `WINDOW_LADDER_RETRYABLE_REASONS`'s own
    # comment for the full argument). `None`/`<= 0` means the
    # configuration cannot be read as a window AT ALL -- doubling does
    # not help (`None * growth` does not evaluate, `0 * growth` is still
    # `0`, negative only moves further away) -- TERMINAL, checked and
    # returned BEFORE `window_frames` is even computed, so this branch
    # never depends on a value it has just declined.
    if window_sec is None or window_sec <= 0:
        return {"declined": True, "reason": "search_window_unviable",
               "evidence": f"scene_search_window_sec={window_sec}"}

    # STRUCTURAL FLOOR, a DIFFERENT claim (Lead's ruling, 2026-09-21), a
    # NAMED decline BEFORE PySceneDetect is ever asked to look -- distinct
    # from `no_scene_change_in_window` ("I looked and found nothing").
    # Same shape as this morning's `pal_saturation_screen` fix: a bound
    # that could go non-positive at a short enough setting makes a guard
    # that could never NOT fire. `window_sec` here is already confirmed a
    # real, positive number (the branch above returned otherwise) -- this
    # is genuinely "too small," not "not a number," and the ladder's
    # growth factor is exactly the plausible remedy: RETRYABLE, in
    # `WINDOW_LADDER_RETRYABLE_REASONS`.
    window_frames = int(round((window_sec * 1000.0) / frame_ms))
    if window_frames < MIN_VALIDATION_FRAMES:
        return {"declined": True, "reason": "search_window_too_narrow",
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
    # Candidate window generously covers BOTH offset hypotheses plus
    # CANDIDATE_SEED_MARGIN_MULTIPLIER times the search margin -- this is
    # seed generation only (see `_anchor_search`'s union-not-agreement
    # note), so generosity here costs decode time, never correctness.
    candidate_margin_frames = CANDIDATE_SEED_MARGIN_MULTIPLIER * window_frames
    c_win_start = max(0, m_bracket_first - candidate_margin_frames + min(before_shift, after_shift))
    c_win_end = m_bracket_last + candidate_margin_frames + max(before_shift, after_shift)

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
    cd_threshold = content_detector_threshold

    master_cuts, master_cuts_failed = _scene_cut_frames(
        master_path, m_win_start, m_win_end - m_win_start, cd_threshold, debug)
    candidate_cuts, candidate_cuts_failed = _scene_cut_frames(
        candidate_path, c_win_start, c_win_end - c_win_start, cd_threshold, debug)
    # `or []` here is SEED GENERATION ONLY, not a re-conflation of the
    # distinction `_scene_cut_frames` just drew: a failed side simply
    # contributes no scene-based seeds (the bracket edge is always tried
    # regardless, per that function's own contract), while
    # `master_cuts_failed`/`candidate_cuts_failed` -- the NAMED tokens --
    # travel unchanged into the evidence below, so a decline still says
    # WHICH of "no cut found" or "detector never ran" occurred on each
    # side, rather than a shared empty list erasing the difference.
    master_cuts_seeds = master_cuts or []
    candidate_cuts_seeds = candidate_cuts or []

    # Anchor A: search BACKWARD from the bracket's own low edge, closest
    # seed first. Seeds: the bracket edge itself (the common, cheap case:
    # Stage 1's bracket already sits right at the true boundary), then
    # master's own cuts before it, then candidate's cuts (translated to
    # master coordinates under the BEFORE hypothesis) before it.
    a_seeds_master = sorted(
        {m_bracket_first}
        | {f for f in master_cuts_seeds if f <= m_bracket_first}
        | {f - before_shift for f in candidate_cuts_seeds if f - before_shift <= m_bracket_first},
        reverse=True)
    anchor_a, anchor_a_reason, anchor_a_n_frames = _anchor_search(
        m_hashes, m_base, c_hashes, c_base, a_seeds_master,
        before_shift, "backward", threshold)

    # Anchor B: search FORWARD from the bracket's own high edge, symmetric,
    # under the AFTER hypothesis.
    b_seeds_master = sorted(
        {m_bracket_last}
        | {f for f in master_cuts_seeds if f >= m_bracket_last}
        | {f - after_shift for f in candidate_cuts_seeds if f - after_shift >= m_bracket_last})
    anchor_b, anchor_b_reason, anchor_b_n_frames = _anchor_search(
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
        # instruction on which token to use). `a_n_frames`/`b_n_frames`
        # (dev-step6-phash mission, 2026-09-21): the widest rung the
        # ladder reached before giving up on that side -- proves whether
        # escalation ran at all versus declined at the floor.
        if anchor_a_reason or anchor_b_reason:
            return {"declined": True, "reason": "anchor_uninformative",
                   "master_seed_count": len(master_cuts_seeds),
                   "candidate_seed_count": len(candidate_cuts_seeds),
                   "evidence": f"anchor_a={anchor_a} anchor_b={anchor_b} "
                              f"a_reason={anchor_a_reason} "
                              f"a_n_frames={anchor_a_n_frames} "
                              f"b_reason={anchor_b_reason} "
                              f"b_n_frames={anchor_b_n_frames}"}
        return {"declined": True, "reason": "anchors_not_established",
               "master_seed_count": len(master_cuts_seeds),
               "candidate_seed_count": len(candidate_cuts_seeds),
               "evidence": f"anchor_a={anchor_a} anchor_b={anchor_b} "
                          f"master_cuts={len(master_cuts_seeds)} "
                          f"master_detector_failed={master_cuts_failed} "
                          f"candidate_cuts={len(candidate_cuts_seeds)} "
                          f"candidate_detector_failed={candidate_cuts_failed}"}

    ordering_refuted, ordering_evidence = _check_anchor_ordering(anchor_a, anchor_b)
    if ordering_refuted:
        return {"declined": True, "reason": "cross_sweep_refuted",
               "master_seed_count": len(master_cuts_seeds),
               "candidate_seed_count": len(candidate_cuts_seeds),
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

    # CROSSING (dev-step7-sweep, 2026-09-21): a MEASUREMENT this function
    # used to take and then discard -- the collapse below fired on it, but
    # neither a return field nor the evidence string ever said so, so a
    # crossed bracket and a bracket that was naturally zero-width from the
    # first frame were INDISTINGUISHABLE to any caller or log reader
    # (Lead's finding, reading this block). Recorded BEFORE the collapse
    # overwrites the pre-collapse values, so the overshoot amount survives
    # into the evidence even though the shipped boundary does not use it.
    sweep_crossed = split_end_master < split_start_master
    pre_collapse_start_master = split_start_master
    pre_collapse_end_master = split_end_master
    if sweep_crossed:
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
               "master_seed_count": len(master_cuts_seeds),
               "candidate_seed_count": len(candidate_cuts_seeds),
               "evidence": plumbing_evidence}
    if not plumbing_ok:
        # RETIRED NAME `anchor_step_uncorroborated` (measured dead: fires
        # only below ~7.75 fps, since `step_ms` and the offset hypotheses
        # are built from the same two plateau means in production --
        # `change_point_locator.py:2269` vs `:2333`/`:2426`). Renamed so
        # nothing downstream mistakes a plumbing disagreement for a
        # content-based refutation.
        return {"declined": True, "reason": "anchor_step_inconsistent",
               "master_seed_count": len(master_cuts_seeds),
               "candidate_seed_count": len(candidate_cuts_seeds),
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
        # SEED-COUNT VARIANCE, MEASURED CONTENT-DEPENDENT (dev-step5-
        # scenedetect, 2026-09-21, real error-tree media): a low
        # `content_detector_threshold` rung's seed density can spike hard
        # on a genuinely busy real window (one measured 32% of frames in
        # a low band against every other sampled window an order of
        # magnitude lower) -- carried on EVERY outcome past seed
        # generation (declined or not) so this variance is visible in
        # production instead of rediscovered (Lead's ruling, same date).
        "master_seed_count": len(master_cuts_seeds),
        "candidate_seed_count": len(candidate_cuts_seeds),
        # REPORTED, NOT ONLY HANDLED (dev-step7-sweep, 2026-09-21): whether
        # the collapse above fired, and the two pre-collapse frame numbers
        # it discarded -- without this a crossed bracket and a bracket that
        # was zero-width from its first frame return an IDENTICAL payload
        # (same master_start_frame == master_end_frame), and nobody reading
        # the result, then or later, can tell which one happened.
        "sweep_crossed": sweep_crossed,
        "derived_ms": {
            "master_start_ms": f"{round(float(_exact_ms_from_frame(split_start_master, fps_num, fps_den)), 2)}",
            "master_end_ms": f"{round(float(_exact_ms_from_frame(split_end_master, fps_num, fps_den)), 2)}",
        },
        "evidence": (f"anchor_a={anchor_a} anchor_a_n_frames={anchor_a_n_frames} "
                    f"anchor_b={anchor_b} anchor_b_n_frames={anchor_b_n_frames} "
                    f"master_cuts={len(master_cuts_seeds)} "
                    f"master_detector_failed={master_cuts_failed} "
                    f"candidate_cuts={len(candidate_cuts_seeds)} "
                    f"candidate_detector_failed={candidate_cuts_failed} "
                    f"sweep_crossed={sweep_crossed}"
                    + (f" pre_collapse_forward={pre_collapse_start_master} "
                       f"pre_collapse_backward={pre_collapse_end_master}"
                       if sweep_crossed else "")
                    + f" length_master={length_master} "
                    f"length_candidate={length_candidate} "
                    f"{plumbing_evidence}"),
    }
