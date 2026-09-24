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

# SUSTAINED DISAGREEMENT, NOT A SINGLE FRAME (dev-step7-sweep, 2026-09-21,
# owner order, real-media finding). MEASURED on real production media
# (bracket step_ms=-64.55, sub-quantum, the smallest of a 4-bracket real
# sample): the forward sweep stopped at master frame 8753 on a single
# Hamming=30 mismatch, while frames 8755-9070 under the SAME before_shift
# hypothesis matched cleanly almost everywhere (re-verified against the
# code's own extraction, not asserted) -- one or two noisy pHash frames
# were read as a structural cut. Compare brackets whose divergence is
# REAL: Hamming stays saturated (30-38) for 4+ CONSECUTIVE frames at every
# genuine cut measured in the same sample. The value mirrors
# `MIN_VALIDATION_FRAMES` -- the owner's own ">= 3 consecutive frames" is
# already the project's standard for "this is not noise," applied here to
# the sweep's STOPPING decision instead of only to anchor establishment.
# Fixes the false-positive-width case (a blip inside otherwise-matching
# content); does NOT and cannot fix a genuine cut being reported wider
# than its truest extent (a different, harder problem -- the sweep still
# never re-probes past a CONFIRMED sustained divergence to look for
# matching resuming further in; that is a search-strategy question, not
# a noise-tolerance one, and stays open, reported separately).
SWEEP_SUSTAINED_MISMATCH_FRAMES = MIN_VALIDATION_FRAMES

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

# ---------------------------------------------------------------------------
# EDGE BRACKETS / SINGLE ANCHOR / pHASH-WALK
# (owner's ruling RULING_20260922_EDGE_SINGLE_ANCHOR.MD + its ADDENDUM;
#  measured spec: architect/cases/ANALYSIS_edge_single_anchor.md)
#
# Everything from here to `locate_edge_boundary` serves ONE new public entry
# point. The two-anchor path above is NOT touched: interior brackets keep two
# anchors and a bidirectional cross-sweep, as designed. An EDGE bracket is a
# different SHAPE of evidence -- content on one side only -- and the owner's
# rule is that it therefore takes ONE anchor, on the common side, and a
# frame-by-frame pHash walk outward from it.
# ---------------------------------------------------------------------------

# N -- THE SUSTAINED-MISMATCH WIDTH FOR THE WALK'S STOPPING DECISION.
# The ruling says N comes from the EXISTING ladder, "symmetric with anchor
# validation, not a new constant". The two candidates in
# `VALIDATION_FRAME_LADDER` are 3 (= `MIN_VALIDATION_FRAMES` =
# `SWEEP_SUSTAINED_MISMATCH_FRAMES`) and 4. MEASURED, three windows of real
# common content on two real pairs (2026-09-22):
#   * Undead Unluck S01E13, master frames 23-959, 937 frames of
#     confirmed-common content: median Hamming 2, longest consecutive
#     mismatch run 2 (with one isolated Hamming=32 frame at master 35 --
#     reproduced here directly, a single-frame excursion inside matching
#     content).
#   * Mai-HiME S01E11, master [1340,1388] s, 1151 frames: 99.0 % match,
#     median Hamming 2, longest consecutive mismatch run 2.
#   * Mai-HiME S01E11, immediately before the candidate's video exhausts:
#     a THREE-frame excursion (Hamming 31/31/31 at master 34405-34407 under
#     the nominal shift, reproduced here), followed by five clean frames at
#     Hamming 3, and only THEN does the candidate run out.
# So the measured noise floor on real pairs is <= 2 consecutive, and there is
# a measured 3-frame excursion sitting a handful of frames before a true
# exhaustion boundary. N=3 would have stopped ON that excursion and
# master-filled the five real candidate frames behind it; N=4 clears every
# observed excursion and carries the walk to the true boundary. N=3 is not
# WRONG -- stopping early only master-fills a little more, a precision cost,
# not a correctness one -- but "a la frame pres" is the ruling's whole point.
# `mismatch_run` AND `max_mismatch_run` are reported on every outcome so this
# choice stays MEASURABLE from production logs instead of re-litigated from
# memory (spec S5 item 7).
EDGE_WALK_SUSTAINED_MISMATCH_FRAMES = VALIDATION_FRAME_LADDER[1]

# THE WALK IS CHUNKED, AND THE CHUNK SIZE IS A MEASURED COST, NOT A GUESS.
# MEASURED `_extract_hashes` throughput (spec S3c, UU S01E13 master, 1080p
# H.264, warm): 2 s window -> 14.45 ms/frame; 8 s -> 6.61; 20 s -> 5.64;
# 40 s -> 5.13; 80 s -> 2.61 ms/frame. Derived: ~0.45 s fixed cost per ffmpeg
# call, ~2.4-2.6 ms marginal decode per frame. 80 s (~1919 frames at
# 24000/1001) is where the fixed cost has amortised; larger chunks buy little
# and cost memory and latency before the first comparison.
EDGE_WALK_CHUNK_SECONDS = 80.0

# THE SEAM BETWEEN TWO CHUNKS IS A MEASUREMENT, NOT AN ASSUMPTION.
# `_extract_hashes` labels element 0 of a decoded window with
# `comparer._frame_index(start_s)` -- the REQUESTED frame -- while ffmpeg's
# first emitted frame at `-ss T` is the first frame at or after T. The two can
# disagree by up to half a grid frame, and `_on_comparer_grid`'s own docstring
# says so in as many words ("that was true before this function existed and
# stays exactly as true after it"). MEASURED, 2026-09-22, Mai-HiME S01E11's
# candidate: extractions started at master frames 32623 and 34332 put the
# file's last video frame at index 34392, while extractions started at 33778
# and 33880 put it at 34393 -- the SAME physical frame, labelled one apart,
# purely from where the seek began. A walk that crossed a chunk boundary and
# simply trusted the new chunk's label would therefore silently gain or lose
# one frame AT the seam, and the error would land straight in
# `addition_frames`, which the ADDENDUM requires to be an EXACT count.
# So every chunk after the first OVERLAPS its predecessor by this many frames
# and its base is CORRECTED by the integer delta that makes the overlap agree.
# 48 frames is ~2 s at 24000/1001: long enough that agreement is a real claim
# about content rather than a coincidence of two similar frames, short enough
# that the overlap is a rounding error against an 80 s chunk.
EDGE_WALK_CHUNK_OVERLAP_FRAMES = 48
# The seam delta is searched over +/- this many frames. Two, because the
# defect being corrected is a HALF-FRAME labelling disagreement (so +/-1 is
# the whole of it) and one frame of margin is kept so the guard measures the
# seam rather than assuming its own bound is tight.
EDGE_WALK_SEAM_SEARCH_FRAMES = 2
# A seam is RE-ESTABLISHED only if the best delta makes this fraction of the
# overlap agree at `ANCHOR_HAMMING_THRESHOLD_DEFAULT`. Below it, the two
# chunks cannot be shown to be the same content read twice, and the walk
# DECLINES (`edge_walk_unreadable`) rather than guessing which label is right
# -- an unverifiable seam is exactly the could-not-measure /
# measured-nothing conflation this campaign has named repeatedly.
EDGE_WALK_SEAM_AGREEMENT_MIN = 0.75

# WHEN GEOMETRY MUST BE NORMALISED BEFORE THE INSTRUMENT IS BELIEVED.
# The pHash squashes every frame to 32x32, so a few pixels of difference in
# CODED size are irrelevant; what is NOT irrelevant is a difference in the
# DISPLAYED ASPECT, because that means one file carries picture where the
# other carries black bars, and the two 32x32 reductions then describe
# different pictures. TWO MEASURED POINTS bracket this constant:
#   * Mai-HiME S01E11: 1460x1078 vs 1456x1072, aspects 1.3543 vs 1.3582 --
#     0.29 % apart. Raw, un-normalised, real common content matches at mean
#     Hamming 1.35 over 500 frames (reproduced 2026-09-22). Normalising here
#     would be a change with no defect to fix.
#   * id 33 (Fallout S01E05): 1920x1080 vs 1920x800, aspects 1.778 vs 2.400
#     -- 25.9 % apart. Raw match 0/193; cropped to `1920:800:0:140`, 193/193.
# 2 % sits roughly an order of magnitude above the benign case and an order of
# magnitude below the broken one. It is a THRESHOLD ON A RATIO, not on pixels:
# a pixel count says nothing about whether the two pictures frame the same
# thing.
EDGE_GEOMETRY_ASPECT_TOLERANCE = 0.02

# THE SHIFT IS RESOLVED AT THE ANCHOR, AGAINST THE EXTRACTION THE ANCHOR WAS
# VALIDATED ON -- over the nominal shift PLUS OR MINUS this many frames.
#
# NOT A SEARCH FOR A BETTER ANSWER, A CORRECTION FOR A KNOWN INSTRUMENT
# PROPERTY, and the bound is DERIVED from the two things that can move it by
# exactly one frame each, with nothing left over:
#   1. The offset is an AUDIO measurement in milliseconds; rounding it onto
#      the video grid (`_nominal_shift_frames`) can land either side of a
#      half-frame residual. MEASURED: Undead Unluck S01E13's head offset is
#      -1017.33 ms = -24.39 frames, Mai-HiME S01E11's tail offset is
#      -807.82 ms = -19.37 frames -- both well off a frame boundary.
#   2. `_extract_hashes` labels element 0 with the frame it ASKED for, while
#      ffmpeg returns the first frame at or after that instant, so the SAME
#      picture can be labelled one index apart in two different reads.
#      MEASURED on Mai-HiME S01E11's tail (2026-09-22): over master
#      [34000,34390) the best-fitting shift is -19 for a candidate read
#      starting at master frame 33880, and -20 for one starting at 32935 --
#      the same content, the same pair, two reads, one frame apart. The
#      BOUNDARY is invariant under this (-19 with the first labelling and -20
#      with the second both put the candidate's last frame at master 34412);
#      the shift alone is not, so the shift must be re-derived per extraction
#      or the walk applies a hypothesis proven against a different labelling.
#
# WHY THE DISCRIMINATOR IS SUMMED HAMMING AND NOT THE MATCH COUNT: at a
# one-frame offset the binary `<= threshold` test DEMONSTRABLY cannot separate
# the hypotheses -- measured on Undead Unluck S01E13, master [120,250), where
# shifts -25 AND -24 both match 130/130, while their mean Hamming (2.369 vs
# 2.154) does separate them. This is the same argument
# `_check_anchor_distinctive` makes at +/-4 and +/-8, continued down to the
# range where a threshold saturates.
EDGE_SHIFT_SEARCH_FRAMES = 2

# Which edge declines a WIDER anchor window could plausibly change, same
# question `WINDOW_LADDER_RETRYABLE_REASONS` answers for the two-anchor path
# and answered the same way: "nothing validated" and "only self-similar
# content was in range" are both window-limited; everything else is not.
EDGE_WINDOW_LADDER_RETRYABLE_REASONS = frozenset(
    {"edge_anchor_not_established", "edge_anchor_uninformative",
     "search_window_too_narrow"})


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


def _parse_positive_rate(value):
    '''Exact positive rational, or `None`. SAME CONTRACT, deliberately NOT
    IMPORTED, as `merge_video_chimeric.parse_positive_rate` (Architect's
    frame-indexed contract rule 6): that module IMPORTS THIS ONE
    (`merge_video_chimeric.py`, its `import scene_anchor` inside the repair
    path), so importing it back at module level is a cycle. A blank string,
    a non-positive rate, `"0/0"`, `"inf"`/`"nan"`, a malformed `"num/den"`
    and a wrong type all collapse to the SAME `None` -- "I could not
    measure" is a property of the value, not of its absence.
    '''
    if value is None:
        return None
    try:
        rate = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return rate if rate > 0 else None


def _probe_frame_rate(path):
    '''THE FILE'S OWN frame rate, as an EXACT RATIONAL -- `(Fraction, None)`
    on success, `(None, reason)` when it could not be measured.

    WHY THIS EXISTS (defect measured 2026-09-22 on errid 5, a 23.976-fps
    master against a 29.97-fps candidate): this module is handed ONE grid
    (`fps_num`/`fps_den`, the MASTER's, resolved by the caller from
    MediaInfo through `merge_video_chimeric.resolve_master_grid`) because
    every boundary it ships is indexed in MASTER frame numbers, by ruling
    (F1). But `_scene_cut_frames` hands its `start_frame`/`n_frames` to
    PySceneDetect, which seeks and counts on the frame grid of THE FILE IT
    OPENED. Handing the candidate a frame COUNT computed on the master's
    grid therefore asks for a window inflated by exactly the ratio of the
    two rates -- measured on that pair: a 180.01 s master scan against a
    240.71 s candidate scan for what should have been the same span of
    time. A frame count is only a duration once you say whose frames.

    `r_frame_rate` is the exact rational the container declares -- the same
    source `frame_compare.py`'s own module docstring names ("r_frame_rate
    ffprobe ou FrameRate_Original"), read as the string ffprobe prints
    ("30000/1001"), never a float rounding of it. Its MediaInfo sibling is
    not reachable here: this module receives PATHS, not the media objects
    the caller resolved the master's grid from.
    '''
    try:
        cmd = [tools.software["ffprobe"], "-v", "error",
               "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
               "-of", "default=noprint_wrappers=1:nokey=1", path]
    except KeyError:
        return None, "ffprobe_not_configured"
    # IMMEDIATELY-PRE-CALL (owner's order via the Lead, 2026-09-22): every
    # external tool call says which file it is on before it can hang.
    tools.dev_log(f"scene_anchor: _probe_frame_rate calling ffprobe "
                  f"file={path}\n")
    try:
        stdout, stderror, exit_code = tools.launch_cmdExt_with_timeout_reload(
            cmd, max_restart=3, timeout=60)
    except Exception as exc:
        return None, f"ffprobe_raised:{type(exc).__name__}"
    if exit_code != 0:
        return None, f"ffprobe_exit:{exit_code}"
    lines = stdout.decode("utf-8", "replace").strip().splitlines()
    raw = lines[0].strip() if lines else ""
    rate = _parse_positive_rate(raw)
    if rate is None:
        return None, f"unparseable_r_frame_rate:{raw!r}"
    return rate, None


def _frames_at_rate(seconds, rate):
    '''An exact-rational number of SECONDS, as a frame count/index on
    `rate`. Exact throughout: `Fraction * Fraction`, rounded once at the
    end. When the product is already a whole number -- which it always is
    for the master side, whose seconds were themselves derived FROM master
    frame counts on this same grid -- the rounding is the identity, so
    routing the master through this helper changes nothing it computed
    before (that is the point: one conversion, used by both sides, that
    degenerates to the old arithmetic whenever the two rates agree).
    '''
    return int(round(Fraction(seconds) * Fraction(rate)))


def _frame_on_grid(frame, from_rate, to_rate):
    '''A frame INDEX carried from one grid to another through the instant
    it names -- `frame / from_rate` seconds, re-counted on `to_rate`.
    Exactly the identity when the two rates are equal.

    Needed because `_scene_cut_frames` returns indices on the grid of the
    file PySceneDetect opened, while every seed arithmetic downstream
    (`f - before_shift` against `m_bracket_first`) is in MASTER frame
    numbers by ruling (F1). Before this existed the candidate's own cut
    indices were consumed as if they were master frames: at 30000/1001
    against 24000/1001 that is a 25% error on the index, i.e. minutes of
    drift by the middle of an episode, and every translated seed lands in
    the wrong half of the bracket filter.
    '''
    if from_rate == to_rate:
        return frame
    return int(round(Fraction(frame) / Fraction(from_rate) * Fraction(to_rate)))


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
        # IMMEDIATELY-PRE-CALL (owner's order via the Lead, 2026-09-22): an
        # in-process PySceneDetect full decode, no timeout, no thread --
        # reached from the repair path (merge_video_chimeric.py's anchor
        # calls). Nothing bounds this call; this line is the only thing
        # that would say it was the one running during a hang.
        tools.dev_log(f"scene_anchor: _scene_cut_frames calling "
                      f"detect_scenes file={path} start_frame={start_frame} "
                      f"n_frames={n_frames}\n")
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


def locate_scene_anchors(master_path, candidate_path, fps_num, fps_den,
                         bracket_low_ms, bracket_high_ms,
                         offset_before_ms, offset_after_ms,
                         step_ms=None, quantum_ms=None,
                         scene_search_window_sec=None, debug=False,
                         candidate_time_scale=None, normalise_geometry=False,
                         resolve_shift=False,
                         shift_search_frames=EDGE_SHIFT_SEARCH_FRAMES):
    '''PUBLIC ENTRY POINT -- unchanged call shape (Lead's ruling,
    2026-09-21), plus three OPT-IN keywords added for the orchestrator's
    hole resolution (stage 4, 2026-09-24). All three default to OFF, and
    with them off every line this function runs, and every byte it returns
    except three added fields, is what it was before -- the live chain
    (`merge_video_chimeric`) passes none of them.

      `candidate_time_scale`  a rate relation `r` (Fraction): the
                              candidate's content plays `r` times faster
                              than the master's. Offsets and every second
                              handed in are then REFERENCE-EQUIVALENT
                              (master) time, which is exactly the domain the
                              orchestrator's speed-corrected audio alignment
                              measured them in; see
                              `frame_compare.FrameComparer`'s `time_scales`.
      `normalise_geometry`    run the EDGE path's geometry precondition
                              (`_resolve_geometry`, the letterbox crop) here
                              too. The edge ruling built it because a
                              geometry mismatch is indistinguishable from a
                              real divergence, and that is no less true
                              between two anchors than beside one: MEASURED,
                              errid-27 (Fallout S01E02) is 1920x1080 against
                              1920x800.
      `resolve_shift`         resolve each anchor's frame shift within
                              `EDGE_SHIFT_SEARCH_FRAMES` of its nominal one,
                              through the edge path's `_edge_anchor_search`,
                              instead of trusting the nominal shift exactly.
                              Needed when the offsets come from fingerprint
                              points: they are quantised to one fingerprint
                              hop (~124 ms, ~3 frames), so the nominal shift
                              can be 1-2 frames off -- the same two reasons
                              `EDGE_SHIFT_SEARCH_FRAMES` documents, plus a
                              coarser instrument. `shift_search_frames` is
                              that window's half-width; its default is the
                              edge constant, and a caller with a coarser
                              offset passes its own derived bound.

    THE OUTER RUNG: resolves `scene_search_window_sec` once
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

    crop_filters = {}
    geometry = None
    if normalise_geometry:
        # THE PRECONDITION, BEFORE ANY ANCHOR, ONCE PER CALL -- the edge
        # path's own function, unmodified. Its refusal is TERMINAL for the
        # same reason it is there: no window size makes two differently
        # framed pictures the same picture.
        geometry, crop_filters, geometry_reason = _resolve_geometry(
            master_path, candidate_path)
        tools.logs.append(
            f"scene_anchor: interior_geometry "
            f"master={geometry.get('master')} candidate={geometry.get('candidate')} "
            f"normalised={geometry.get('normalised')} crop={geometry.get('crop')} "
            f"verdict={geometry.get('verdict')} reason={geometry_reason}\n")
        if geometry_reason is not None:
            return {"declined": True, "reason": "geometry_unreconciled",
                    "geometry": geometry, "evidence": geometry_reason}

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
            content_detector_threshold=rung_cd_threshold, debug=debug,
            candidate_time_scale=candidate_time_scale,
            crop_filters=crop_filters, resolve_shift=resolve_shift,
            shift_search_frames=shift_search_frames)
        if geometry is not None:
            result["geometry"] = geometry
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
                                    debug=False, candidate_time_scale=None,
                                    crop_filters=None, resolve_shift=False,
                                    shift_search_frames=EDGE_SHIFT_SEARCH_FRAMES):
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
                             fps_num, fps_den, debug=debug,
                             crop_filters=crop_filters,
                             time_scales=({candidate_path: candidate_time_scale}
                                          if candidate_time_scale is not None
                                          else None))
    m_bracket_first = comparer._frame_index(bracket_low_ms / 1000.0)
    m_bracket_last = comparer._frame_index(bracket_high_ms / 1000.0)

    before_shift = _nominal_shift_frames(offset_before_ms, fps_num, fps_den)
    after_shift = _nominal_shift_frames(offset_after_ms, fps_num, fps_den)
    nominal_before_shift, nominal_after_shift = before_shift, after_shift

    m_win_start = max(0, m_bracket_first - window_frames)
    m_win_end = m_bracket_last + window_frames
    # Candidate window generously covers BOTH offset hypotheses plus
    # CANDIDATE_SEED_MARGIN_MULTIPLIER times the search margin -- this is
    # seed generation only (see `_anchor_search`'s union-not-agreement
    # note), so generosity here costs decode time, never correctness.
    candidate_margin_frames = CANDIDATE_SEED_MARGIN_MULTIPLIER * window_frames
    c_win_start = max(0, m_bracket_first - candidate_margin_frames + min(before_shift, after_shift))
    c_win_end = m_bracket_last + candidate_margin_frames + max(before_shift, after_shift)

    # THE WINDOW IS A SPAN OF TIME; A FRAME COUNT IS ONLY A SPAN OF TIME
    # ONCE YOU SAY WHOSE FRAMES (defect measured 2026-09-22 on errid 5, a
    # 23.976-fps master against a 29.97-fps candidate -- see
    # `_probe_frame_rate` for the measurement). `m_win_*`/`c_win_*` above
    # are MASTER frame numbers and stay that way: every boundary this
    # module ships is indexed on the master's grid by ruling (F1). But the
    # two consumers below do NOT read frame numbers the same way:
    #
    #   `_extract_hashes`  takes SECONDS -- already per-file correct, and
    #                      unchanged here (the same rational, floated once).
    #   `_scene_cut_frames` takes FRAME NUMBERS and gives them to
    #                      PySceneDetect, which seeks and counts on the grid
    #                      of THE FILE IT OPENED -- so the candidate's
    #                      window must be counted on the CANDIDATE's rate,
    #                      or it asks for a span inflated by the ratio of
    #                      the two rates (measured: 180.01 s of master
    #                      against 240.71 s of candidate for what was meant
    #                      to be the same window of time).
    #
    # The seconds are exact rationals, not floats, so neither side's span
    # is built on a rounded rate; each is rounded ONCE, at its own rate.
    master_rate = Fraction(fps_num, fps_den)
    m_win_start_sec = Fraction(m_win_start * fps_den, fps_num)
    m_win_span_sec = Fraction((m_win_end - m_win_start) * fps_den, fps_num)
    c_win_start_sec = Fraction(c_win_start * fps_den, fps_num)
    c_win_span_sec = Fraction((c_win_end - c_win_start) * fps_den, fps_num)

    candidate_rate, candidate_rate_reason = _probe_frame_rate(candidate_path)
    # A SPEED-CHANGED CANDIDATE IS SCANNED ON ITS CORRECTED RATE (opt-in,
    # `candidate_time_scale`): the candidate window above is in master-
    # EQUIVALENT seconds, and `native / r` frames per equivalent second is
    # exactly `native` frames per raw second -- so the scan addresses the
    # right raw span and `_frame_on_grid` carries its cuts back onto the
    # master grid through the same corrected rate. Identity when absent.
    if candidate_rate is not None and candidate_time_scale is not None:
        candidate_rate = candidate_rate / Fraction(candidate_time_scale)

    m_start_s = float(m_win_start_sec)
    m_dur_s = float(m_win_span_sec)
    c_start_s = float(c_win_start_sec)
    c_dur_s = float(c_win_span_sec)

    m_scan_start = _frames_at_rate(m_win_start_sec, master_rate)
    m_scan_frames = _frames_at_rate(m_win_span_sec, master_rate)
    if candidate_rate is None:
        c_scan_start = c_scan_frames = None
    else:
        c_scan_start = _frames_at_rate(c_win_start_sec, candidate_rate)
        c_scan_frames = _frames_at_rate(c_win_span_sec, candidate_rate)
    # BOTH RATES AND BOTH SPANS, AT THE CONVERSION -- so a future asymmetry
    # is readable from a production log instead of being rediscovered by
    # timing two scans against each other (which is how this one was
    # found). Rates as rationals, never as the float that hid the ratio.
    tools.dev_log(
        f"scene_anchor: scan_window_conversion "
        f"master_rate={master_rate.numerator}/{master_rate.denominator} "
        f"candidate_rate="
        f"{'unmeasured:' + str(candidate_rate_reason) if candidate_rate is None else str(candidate_rate.numerator) + '/' + str(candidate_rate.denominator)} "
        f"master_scan=[{m_scan_start},+{m_scan_frames}) "
        f"({float(m_win_span_sec):.3f} s) "
        f"candidate_scan=[{c_scan_start},+{c_scan_frames}) "
        f"({float(c_win_span_sec):.3f} s)\n")

    m_base, m_hashes = _extract_hashes(comparer, master_path, m_start_s, m_dur_s)
    c_base, c_hashes = _extract_hashes(comparer, candidate_path, c_start_s, c_dur_s)
    if not m_hashes or not c_hashes:
        return {"declined": True, "reason": "frames_unextractable",
               "evidence": f"master_frames={len(m_hashes)} "
                          f"candidate_frames={len(c_hashes)}"}

    threshold = ANCHOR_HAMMING_THRESHOLD_DEFAULT
    cd_threshold = content_detector_threshold

    master_cuts, master_cuts_failed = _scene_cut_frames(
        master_path, m_scan_start, m_scan_frames, cd_threshold, debug)
    if candidate_rate is None:
        # REFUSE A SCAN WE CANNOT ADDRESS, DO NOT GUESS ITS GRID. Falling
        # back to the master's rate here is precisely the defect this
        # block fixes, so the candidate simply contributes no scene seeds
        # and says why -- the SAME shape `_scene_cut_frames` already uses
        # for "the detector never ran", and the bracket edge is still
        # tried on both anchors regardless (that function's own contract),
        # so an unprobeable candidate declines nothing by itself.
        candidate_cuts = None
        candidate_cuts_failed = f"candidate_grid_unmeasured:{candidate_rate_reason}"
    else:
        candidate_cuts, candidate_cuts_failed = _scene_cut_frames(
            candidate_path, c_scan_start, c_scan_frames, cd_threshold, debug)
    # `or []` here is SEED GENERATION ONLY, not a re-conflation of the
    # distinction `_scene_cut_frames` just drew: a failed side simply
    # contributes no scene-based seeds (the bracket edge is always tried
    # regardless, per that function's own contract), while
    # `master_cuts_failed`/`candidate_cuts_failed` -- the NAMED tokens --
    # travel unchanged into the evidence below, so a decline still says
    # WHICH of "no cut found" or "detector never ran" occurred on each
    # side, rather than a shared empty list erasing the difference.
    master_cuts_seeds = master_cuts or []
    # CARRIED ONTO THE MASTER'S GRID, because that is the only coordinate
    # the seed arithmetic below speaks (`f - before_shift` compared against
    # `m_bracket_first`, both master frame numbers, F1's ruling). The cuts
    # PySceneDetect just returned are indices on the CANDIDATE's own grid --
    # now that the scan is addressed at the candidate's rate, they are the
    # right instants, and this is the one step that keeps them the right
    # NUMBERS too. Exactly the identity for a same-rate pair.
    candidate_cuts_seeds = [
        _frame_on_grid(f, candidate_rate, master_rate)
        for f in (candidate_cuts or [])]

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
    if resolve_shift:
        # OPT-IN (see `locate_scene_anchors`): the edge path's own search,
        # unmodified -- `_validate_anchor` and `_check_anchor_distinctive`
        # offered each shift within `EDGE_SHIFT_SEARCH_FRAMES` of the
        # nominal one, smallest summed Hamming wins, ties to the nominal.
        # Each anchor resolves ITS OWN hypothesis; the sweep below then runs
        # on the resolved pair, and `_check_step_plumbing` still compares
        # their difference against the caller's step.
        anchor_a, resolved, anchor_a_n_frames, anchor_a_reason = _edge_anchor_search(
            m_hashes, m_base, c_hashes, c_base, a_seeds_master,
            before_shift, "backward", threshold, search_frames=shift_search_frames)
        if anchor_a is not None:
            before_shift = resolved
    else:
        anchor_a, anchor_a_reason, anchor_a_n_frames = _anchor_search(
            m_hashes, m_base, c_hashes, c_base, a_seeds_master,
            before_shift, "backward", threshold)

    # Anchor B: search FORWARD from the bracket's own high edge, symmetric,
    # under the AFTER hypothesis.
    b_seeds_master = sorted(
        {m_bracket_last}
        | {f for f in master_cuts_seeds if f >= m_bracket_last}
        | {f - after_shift for f in candidate_cuts_seeds if f - after_shift >= m_bracket_last})
    if resolve_shift:
        anchor_b, resolved, anchor_b_n_frames, anchor_b_reason = _edge_anchor_search(
            m_hashes, m_base, c_hashes, c_base, b_seeds_master,
            after_shift, "forward", threshold, search_frames=shift_search_frames)
        if anchor_b is not None:
            after_shift = resolved
    else:
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
    # can run past its partner. STOPS ON SUSTAINED DISAGREEMENT, NOT A
    # SINGLE FRAME (dev-step7-sweep, 2026-09-21, owner order): a lone
    # mismatch inside an otherwise-matching run is tolerated -- the sweep
    # does not bank it as an advance (a genuinely unmatched frame is not
    # silently counted as matching) but it is not accepted as THE boundary
    # either until `SWEEP_SUSTAINED_MISMATCH_FRAMES` consecutive frames
    # disagree. `split_start_master`/`split_end_master` land on the LAST
    # frame actually confirmed matching, one step before whichever run of
    # mismatches triggered the stop -- identical to the old single-frame
    # rule whenever the true divergence is sustained from its first frame
    # (measured: brackets whose Hamming stays saturated for 4+ frames stop
    # in exactly the same place either way), and different only when the
    # first mismatch was noise the old rule had no way to see past.
    split_start_master = anchor_a
    consecutive_mismatches = 0
    for m_frame in range(anchor_a, anchor_b):
        c_frame = m_frame + before_shift
        if _frames_match(m_hashes, m_base, m_frame, c_hashes, c_base, c_frame,
                         threshold) is True:
            split_start_master = m_frame + 1
            consecutive_mismatches = 0
        else:
            consecutive_mismatches += 1
            if consecutive_mismatches >= SWEEP_SUSTAINED_MISMATCH_FRAMES:
                break
    split_start_candidate = split_start_master + before_shift

    split_end_master = anchor_b
    consecutive_mismatches = 0
    for m_frame in range(anchor_b - 1, anchor_a - 1, -1):
        c_frame = m_frame + after_shift
        if _frames_match(m_hashes, m_base, m_frame, c_hashes, c_base, c_frame,
                         threshold) is True:
            split_end_master = m_frame
            consecutive_mismatches = 0
        else:
            consecutive_mismatches += 1
            if consecutive_mismatches >= SWEEP_SUSTAINED_MISMATCH_FRAMES:
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
    # WHAT THE TWO FRONTS LEFT UNMATCHED, READ UNDER EACH SHIFT (orchestrator
    # stage 4, 2026-09-24; observational, additive, decides nothing here). The
    # sweep stops on `SWEEP_SUSTAINED_MISMATCH_FRAMES` consecutive mismatches,
    # which assumes pHash noise inside common content never runs that long.
    # MEASURED FALSE on errid-70 (PAL, under the true shift, master
    # 11152-14801): mismatch runs of 3 (x6), 4 (x3), 5 and 15 frames inside
    # content that matches 92-96 % -- the backward walk stopped 124 frames
    # into 3647 frames of common content. The caller can only tell a
    # divergent span from a noise stop by counting how much of the span
    # each hypothesis actually matches, from the hashes already in hand.
    span_counts = {}
    for label, span_shift in (("before", before_shift), ("after", after_shift)):
        matched = readable = 0
        for m_frame in range(pre_collapse_start_master, pre_collapse_end_master):
            verdict = _frames_match(m_hashes, m_base, m_frame, c_hashes, c_base,
                                    m_frame + span_shift, threshold)
            if verdict is None:
                continue
            readable += 1
            matched += 1 if verdict else 0
        span_counts[label] = [matched, readable]
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
        # THE TWO FRONTS AND THE SHIFTS, AS FIELDS (orchestrator stage 4,
        # 2026-09-24). They were already measured here and already in the
        # evidence STRING; a caller that has to decide on them -- ADDENDUM 3's
        # `no_cut_confirmed`, ADDENDUM 4/13's `boundary_pinned_to_ambiguous_zone_end`,
        # which must log "the lengths of both walks and the size of the span"
        # -- must not parse prose to get them. Additive: no existing key moves.
        "pre_collapse_start_master": pre_collapse_start_master,
        "pre_collapse_end_master": pre_collapse_end_master,
        "unmatched_span_matches_before": span_counts["before"],
        "unmatched_span_matches_after": span_counts["after"],
        "forward_walk_frames": pre_collapse_start_master - anchor_a,
        "backward_walk_frames": anchor_b - pre_collapse_end_master,
        "before_shift_frames": before_shift,
        "after_shift_frames": after_shift,
        "nominal_before_shift_frames": nominal_before_shift,
        "nominal_after_shift_frames": nominal_after_shift,
        "anchor_a_n_frames": anchor_a_n_frames,
        "anchor_b_n_frames": anchor_b_n_frames,
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


# ===========================================================================
# EDGE BRACKETS -- ONE ANCHOR, THEN A pHASH-WALK TO THE BOUNDARY
# ===========================================================================
#
# WHY A SEPARATE PATH AND NOT A FLAG ON THE ONE ABOVE. At an edge, the
# two-anchor protocol does not merely perform badly -- it CANNOT succeed, and
# it burns every rung of the expensive outer ladder proving it. Traced in the
# code, 2026-09-22: at a HEAD bracket `a_seeds_master` is filtered
# `<= m_bracket_first`, and with `bracket_low_ms ~ 0` that collapses to
# roughly `{0}`; `_validate_anchor(direction="backward")` then checks
# `range(m_seed - n, m_seed)`, i.e. NEGATIVE frame indices, `_frames_match`
# returns `None`, and validation fails -- correctly, because an unreadable
# frame is never a pass by omission. Anchor A can never validate at a head.
# The tail is the mirror image. So `anchor_a is None or anchor_b is None`
# always fires, the reason is RETRYABLE, and all three rungs run -- each one
# re-extracting frames and re-running PySceneDetect on BOTH files -- before
# declining `search_window_ceiling_reached`. That is precisely the "the ladder
# must not burn rungs looking for one [anchor]" the ruling forbids.


def _probe_video_geometry(path):
    '''The file's own coded `width`/`height` and pixel aspect, as an EXACT
    rational where the container declares one -- `(dict, None)` on success,
    `(None, reason)` when it could not be measured.

    Same instrument, same failure vocabulary and same immediately-pre-call
    log line as `_probe_frame_rate` above; this module receives PATHS, not
    the media objects the caller resolved the master's grid from, so ffprobe
    is the only source available here.

    `sample_aspect_ratio` is ffprobe's own "N:M" and is frequently `0:1`
    ("unknown") rather than `1:1`; both are read as square pixels, which is
    what every file measured for this ruling actually has. The value is used
    ONLY to form a display aspect for the tolerance comparison below -- never
    to scale anything.
    '''
    try:
        cmd = [tools.software["ffprobe"], "-v", "error",
               "-select_streams", "v:0",
               "-show_entries", "stream=width,height,sample_aspect_ratio",
               "-of", "default=noprint_wrappers=1:nokey=1", path]
    except KeyError:
        return None, "ffprobe_not_configured"
    # IMMEDIATELY-PRE-CALL (owner's order via the Lead, 2026-09-22): every
    # external tool call says which file it is on before it can hang.
    tools.dev_log(f"scene_anchor: _probe_video_geometry calling ffprobe "
                  f"file={path}\n")
    try:
        stdout, stderror, exit_code = tools.launch_cmdExt_with_timeout_reload(
            cmd, max_restart=3, timeout=60)
    except Exception as exc:
        return None, f"ffprobe_raised:{type(exc).__name__}"
    if exit_code != 0:
        return None, f"ffprobe_exit:{exit_code}"
    lines = [ln.strip() for ln in
             stdout.decode("utf-8", "replace").strip().splitlines()]
    if len(lines) < 2:
        return None, f"unparseable_geometry:{lines!r}"
    try:
        width, height = int(lines[0]), int(lines[1])
    except (TypeError, ValueError):
        return None, f"unparseable_geometry:{lines[:2]!r}"
    if width <= 0 or height <= 0:
        return None, f"non_positive_geometry:{width}x{height}"
    sar = Fraction(1, 1)
    if len(lines) > 2 and lines[2] not in ("", "N/A", "0:1"):
        try:
            num, den = lines[2].split(":")
            parsed = Fraction(int(num), int(den))
            if parsed > 0:
                sar = parsed
        except (TypeError, ValueError, ZeroDivisionError):
            # A malformed SAR is not an absent one, but it is also not worth a
            # decline: square pixels is what every file measured for this
            # ruling has, and the value only feeds a RATIO comparison whose
            # own tolerance is 2 %. Named in the returned dict so a reader can
            # see the assumption rather than infer it.
            sar = Fraction(1, 1)
    return {"width": width, "height": height, "sar": sar,
            "aspect": Fraction(width, height) * sar}, None


def _even(value):
    '''An even, non-negative crop offset. Chroma-subsampled pixel formats
    (yuv420, what every file in this campaign's corpus uses) cannot be cropped
    at an odd offset; ffmpeg either refuses or silently shifts. One pixel of
    framing is far below the 32x32 reduction's own resolution, so rounding
    down is free -- but it is rounded HERE, once, rather than left for ffmpeg
    to decide silently.'''
    return max(0, int(value) - (int(value) % 2))


def _resolve_geometry(master_path, candidate_path):
    '''THE GEOMETRY PRECONDITION (spec S3a, finding 2) -- run ONCE per pair,
    before any anchor or walk, because a walk that reports "sustained mismatch
    at the anchor" on a geometry mismatch is INDISTINGUISHABLE from a real
    boundary. That is the could-not-measure / measured-nothing conflation this
    campaign has now named six times, in the direction that manufactures a
    divergence nobody observed and then trims or master-fills on it.

    Returns `(geometry_dict, crop_filters, reason)`:
      * `reason is None` and `crop_filters` empty  -- the two geometries are
        already comparable (same coded size, or aspects within
        `EDGE_GEOMETRY_ASPECT_TOLERANCE`); read the files as they are.
      * `reason is None` and `crop_filters` non-empty -- normalisation was
        NEEDED and was established: the file with the SMALLER display aspect
        (the one carrying bars) is cropped, centred, to the larger aspect.
      * `reason` set -- `edge_geometry_unreconciled`'s evidence; the caller
        declines rather than walking.

    PREFERENCE ORDER IS THE SPEC'S, AND ITS FIRST BRANCH IS MEASURED: on id 33
    cropping the master to `1920:800:0:140` takes the match from 0/193 to
    193/193 at lag 15. The crop this function computes for that pair is
    exactly `1920:800:0:140` -- 1920/2.400 = 800, (1080-800)//2 = 140 -- which
    is how the branch is anchored to a measurement rather than to an
    intention.
    '''
    m_geom, m_reason = _probe_video_geometry(master_path)
    c_geom, c_reason = _probe_video_geometry(candidate_path)
    if m_geom is None or c_geom is None:
        return ({"master": None, "candidate": None, "normalised": False,
                 "crop": None},
                {},
                f"geometry_unmeasured master={m_reason} candidate={c_reason}")

    def _label(g):
        return f"{g['width']}x{g['height']}"

    base = {"master": _label(m_geom), "candidate": _label(c_geom),
            "master_aspect": f"{float(m_geom['aspect']):.4f}",
            "candidate_aspect": f"{float(c_geom['aspect']):.4f}",
            "normalised": False, "crop": None}

    if (m_geom["width"], m_geom["height"], m_geom["sar"]) == \
       (c_geom["width"], c_geom["height"], c_geom["sar"]):
        base["verdict"] = "identical"
        return base, {}, None

    a_m, a_c = m_geom["aspect"], c_geom["aspect"]
    spread = abs(a_m - a_c) / max(a_m, a_c)
    if spread <= Fraction(EDGE_GEOMETRY_ASPECT_TOLERANCE).limit_denominator(10 ** 6):
        # DIFFERENT PIXELS, SAME PICTURE. Measured on Mai-HiME S01E11
        # (1460x1078 vs 1456x1072, 0.29 % apart): real common content matches
        # at mean Hamming 1.35 over 500 frames with no normalisation at all.
        # Cropping here would change a measurement that is already correct.
        base["verdict"] = f"comparable aspect_spread={float(spread):.4f}"
        return base, {}, None

    # The file with the SMALLER display aspect is the one carrying bars; crop
    # it to the larger aspect, centred. Only ONE side is ever cropped: cropping
    # both would be two guesses where the evidence supports one.
    if a_m < a_c:
        bar_path, bar_geom, target = master_path, m_geom, a_c
        bar_side = "master"
    else:
        bar_path, bar_geom, target = candidate_path, c_geom, a_m
        bar_side = "candidate"

    width, height, sar = bar_geom["width"], bar_geom["height"], bar_geom["sar"]
    # Letterbox (bars top/bottom) -> the height is the excess. Pillarbox
    # (bars left/right) -> the width is. Try height first; if the required
    # height is not smaller than the real one, the excess is horizontal.
    new_h = int(round(Fraction(width) * sar / target))
    if 0 < new_h < height:
        crop_w, crop_h = width, new_h
        crop_x, crop_y = 0, _even((height - new_h) // 2)
    else:
        new_w = int(round(Fraction(height) * target / sar))
        if not (0 < new_w < width):
            return (base, {},
                    f"aspect_spread={float(spread):.4f} exceeds "
                    f"{EDGE_GEOMETRY_ASPECT_TOLERANCE} and no centred crop of "
                    f"{bar_side} {_label(bar_geom)} reaches "
                    f"{float(target):.4f}")
        crop_w, crop_h = new_w, height
        crop_x, crop_y = _even((width - new_w) // 2), 0

    # THE CROP IS VERIFIED AGAINST ITS OWN TARGET, not assumed to have hit it:
    # the even-offset rounding and the integer crop size both move the result,
    # and a crop that misses the aspect it was computed for must decline, not
    # ship. `crop_x`/`crop_y` are even by `_even`; `crop_w`/`crop_h` keep the
    # uncropped dimension exactly, so only the computed one can drift.
    achieved = Fraction(crop_w, crop_h) * sar
    if abs(achieved - target) / max(achieved, target) > \
            Fraction(EDGE_GEOMETRY_ASPECT_TOLERANCE).limit_denominator(10 ** 6):
        return (base, {},
                f"centred crop {crop_w}:{crop_h}:{crop_x}:{crop_y} of "
                f"{bar_side} reaches aspect {float(achieved):.4f}, not "
                f"{float(target):.4f}")

    crop = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"
    base["normalised"] = True
    base["crop"] = f"{bar_side}:{crop}"
    base["verdict"] = (f"normalised aspect_spread={float(spread):.4f} "
                       f"{bar_side} {_label(bar_geom)} -> {crop_w}x{crop_h}")
    return base, {bar_path: crop}, None


class _ChunkedFrames:
    '''ONE SIDE of the walk's frame supply, read in `EDGE_WALK_CHUNK_SECONDS`
    chunks and addressed by MASTER-GRID frame number (F1's ruling: every index
    this module ships is a master frame, and `_extract_hashes` already returns
    the candidate's hashes re-indexed onto the comparer's grid, so both sides
    speak the same numbers).

    THE WHOLE POINT OF THIS CLASS IS THE DISTINCTION THE ADDENDUM TURNS ON:
    a `None` at a CHUNK boundary must fetch the next chunk, while a `None` at
    the FILE boundary terminates the walk. Getting that backwards turns a
    chunk edge into a false `candidate_exhausted` and master-fills the rest of
    the episode with content the candidate actually had. `get()` therefore
    never returns a bare `None`: it returns a NAMED state, and "I have no
    frame there" is split into `file_end` (a fact about the file) and
    `unreadable` (a fact about this read), which are different claims.

    `declared_last_frame` is an UPPER bound only -- a ceiling past which the
    walk refuses to keep asking. It is deliberately NOT the authority on where
    the video ends: MEASURED on Mai-HiME S01E11's candidate, the container
    declares 1452.030 s while the VIDEO stream's last frame is at 1434.45 s,
    the 17.5 s difference being carried by its subtitle streams. A walk that
    trusted the container would ask for 420 frames that do not exist. What
    ends the walk is the DECODE: ffmpeg was asked for frames in that span and
    produced none, twice, from two different seek points.
    '''

    def __init__(self, comparer, path, side, fps_num, fps_den,
                 declared_last_frame, debug=False,
                 initial_base=None, initial_hashes=None,
                 chunk_seconds=EDGE_WALK_CHUNK_SECONDS):
        self.comparer = comparer
        self.path = path
        self.side = side
        self.fps_num = int(fps_num)
        self.fps_den = int(fps_den)
        self.frame_ms = 1000.0 * self.fps_den / self.fps_num
        self.chunk_frames = max(
            MIN_VALIDATION_FRAMES,
            int(round(chunk_seconds * 1000.0 / self.frame_ms)))
        self.declared_last_frame = declared_last_frame
        self.debug = debug
        # SEEDED WITH THE ANCHOR WINDOW'S OWN HASHES, AND THAT IS LOAD-BEARING,
        # NOT AN OPTIMISATION. The anchor was validated at a particular frame
        # SHIFT against a particular pair of extractions; `_extract_hashes`
        # labels element 0 with the frame it ASKED for, while ffmpeg returns
        # the first frame at or after that instant, so a DIFFERENT extraction
        # of the same span can label the same picture one index apart (see
        # `EDGE_WALK_CHUNK_OVERLAP_FRAMES` for the measurement). Starting the
        # walk on a fresh read would therefore apply a shift proven against
        # one labelling to a different one. MEASURED consequence of not doing
        # this, on Mai-HiME S01E11's tail (2026-09-22): a distinctive anchor
        # at master 34117 validated at n=13, and the walk then reported
        # `sustained_mismatch` four frames later, inside 295 frames of content
        # that matches 500/500 at the same shift. The frames are already in
        # hand; re-reading them was both slower and wrong.
        self.base = initial_base if initial_hashes else None
        self.hashes = list(initial_hashes) if initial_hashes else []
        self.chunks_read = 0
        self.seam_deltas = []
        self.file_end_source = None

    def _seconds(self, frame):
        return max(0.0, frame * self.frame_ms / 1000.0)

    def _read(self, start_frame, n_frames):
        base, hashes = _extract_hashes(
            self.comparer, self.path, self._seconds(start_frame),
            n_frames * self.frame_ms / 1000.0)
        return base, hashes

    def _seam_delta(self, new_base, new_hashes):
        '''The integer correction that makes a NEW chunk agree with the one
        already held, over their deliberate overlap. Returns
        `(delta, agreement)`, or `(None, agreement)` when no delta in
        `+/- EDGE_WALK_SEAM_SEARCH_FRAMES` reaches
        `EDGE_WALK_SEAM_AGREEMENT_MIN`.

        Not an optimisation and not defensive padding: see
        `EDGE_WALK_CHUNK_OVERLAP_FRAMES` for the measurement that made this
        necessary -- the same physical frame is labelled one index apart
        depending on where the seek began, and that one frame lands directly
        in `addition_frames`, which the ADDENDUM requires to be exact.
        '''
        # The overlap is wherever the two reads actually cover the same frame
        # numbers -- computed, not assumed to sit at one end, because a
        # forward read starts inside the held chunk while a backward read ends
        # inside it.
        lo = max(self.base, new_base)
        hi = min(self.base + len(self.hashes), new_base + len(new_hashes))
        if hi - lo > EDGE_WALK_CHUNK_OVERLAP_FRAMES:
            if new_base > self.base:
                hi = lo + EDGE_WALK_CHUNK_OVERLAP_FRAMES
            else:
                lo = hi - EDGE_WALK_CHUNK_OVERLAP_FRAMES
        best_delta, best_agree = None, 0.0
        for delta in range(-EDGE_WALK_SEAM_SEARCH_FRAMES,
                           EDGE_WALK_SEAM_SEARCH_FRAMES + 1):
            agree = total = 0
            for frame in range(lo, hi):
                old_i = frame - self.base
                new_i = frame + delta - new_base
                if not (0 <= old_i < len(self.hashes)):
                    continue
                if not (0 <= new_i < len(new_hashes)):
                    continue
                total += 1
                if FrameComparer._popcount64(
                        self.hashes[old_i] ^ new_hashes[new_i]) \
                        <= ANCHOR_HAMMING_THRESHOLD_DEFAULT:
                    agree += 1
            if total < MIN_VALIDATION_FRAMES:
                continue
            fraction = agree / total
            if fraction > best_agree:
                best_delta, best_agree = delta, fraction
        if best_delta is None or best_agree < EDGE_WALK_SEAM_AGREEMENT_MIN:
            return None, best_agree
        return best_delta, best_agree

    def get(self, frame):
        '''`(hash, "ok")`, `(None, "file_end")` or `(None, "unreadable")` for
        one MASTER-GRID frame number.'''
        if frame < 0:
            self.file_end_source = self.file_end_source or "before_frame_zero"
            return None, "file_end"
        if self.declared_last_frame is not None and frame > self.declared_last_frame:
            self.file_end_source = self.file_end_source or "declared_duration"
            return None, "file_end"
        if self.base is not None and 0 <= frame - self.base < len(self.hashes):
            return self.hashes[frame - self.base], "ok"

        # A NEW CHUNK, POSITIONED SO IT ALWAYS OVERLAPS THE ONE ALREADY HELD.
        # The walk is contiguous, so the frame asked for sits exactly one step
        # outside the held range; the new read is placed to cover it AND to
        # re-read `EDGE_WALK_CHUNK_OVERLAP_FRAMES` frames the held chunk
        # already has, which is what `_seam_delta` needs to measure the seam.
        # An earlier version aligned chunks to multiples of the chunk size --
        # tidier to read, and MEASURABLY WRONG: on Mai-HiME S01E11's tail the
        # first aligned chunk was labelled one frame apart from the anchor
        # window's own extraction, the validated shift no longer applied, and
        # the walk declared a sustained mismatch four frames past a perfectly
        # good anchor (reproduced 2026-09-22, before this form replaced it).
        want_overlap = self.base is not None
        if not want_overlap:
            read_start = max(0, frame)
            read_frames = self.chunk_frames
        elif frame > self.base:
            held_hi = self.base + len(self.hashes) - 1
            read_start = max(0, held_hi - EDGE_WALK_CHUNK_OVERLAP_FRAMES + 1)
            read_frames = self.chunk_frames + EDGE_WALK_CHUNK_OVERLAP_FRAMES
        else:
            read_start = max(0, self.base - self.chunk_frames)
            read_frames = (self.base - read_start) + EDGE_WALK_CHUNK_OVERLAP_FRAMES
        new_base, new_hashes = self._read(read_start, read_frames)
        self.chunks_read += 1
        if not new_hashes:
            # ASKED AND GOT NOTHING. Confirm from a DIFFERENT seek point
            # before calling it the end of the file -- one empty read could be
            # a seek artefact, two from different starts is a property of the
            # file. Cheap (one ffmpeg call) against the cost of being wrong,
            # which is master-filling the remainder of an episode.
            confirm_base, confirm_hashes = self._read(frame, self.chunk_frames)
            if not confirm_hashes:
                self.file_end_source = self.file_end_source or "decode_empty_twice"
                return None, "file_end"
            new_base, new_hashes = confirm_base, confirm_hashes
            want_overlap = False

        if want_overlap:
            delta, agreement = self._seam_delta(new_base, new_hashes)
            if delta is None:
                tools.logs.append(
                    f"scene_anchor: edge_walk_seam side={self.side} "
                    f"read_start={read_start} re_established=False "
                    f"best_agreement={agreement:.3f} "
                    f"min={EDGE_WALK_SEAM_AGREEMENT_MIN}\n")
                return None, "unreadable"
            self.seam_deltas.append(delta)
            new_base += delta
            tools.logs.append(
                f"scene_anchor: edge_walk_seam side={self.side} "
                f"read_start={read_start} re_established=True delta={delta} "
                f"agreement={agreement:.3f}\n")

        self.base, self.hashes = new_base, new_hashes
        if 0 <= frame - self.base < len(self.hashes):
            return self.hashes[frame - self.base], "ok"
        if frame >= self.base:
            # The chunk was read and is SHORT of the frame asked for: the
            # decoder stopped inside this span. Same confirmation rule as an
            # empty chunk -- a second read from the frame itself, and only two
            # failures make it the file's end.
            confirm_base, confirm_hashes = self._read(frame, self.chunk_frames)
            if confirm_hashes and 0 <= frame - confirm_base < len(confirm_hashes):
                self.base, self.hashes = confirm_base, confirm_hashes
                return self.hashes[frame - self.base], "ok"
            self.file_end_source = self.file_end_source or "decode_short_chunk"
            return None, "file_end"
        return None, "unreadable"


def _anchor_window_distance(m_hashes, m_base, c_hashes, c_base, m_seed,
                            shift_frames, direction, n_frames):
    '''Total Hamming distance over the SAME window `_validate_anchor` checks,
    or `None` when any pair in it is unreadable. The continuous form of that
    function's binary answer -- see `EDGE_SHIFT_SEARCH_FRAMES` for why a
    threshold cannot separate two hypotheses one frame apart and this can.'''
    frames = (range(m_seed, m_seed + n_frames) if direction == "forward"
              else range(m_seed - n_frames, m_seed))
    total = 0
    for m_frame in frames:
        mi = m_frame - m_base
        ci = m_frame + shift_frames - c_base
        if not (0 <= mi < len(m_hashes)) or not (0 <= ci < len(c_hashes)):
            return None
        total += FrameComparer._popcount64(m_hashes[mi] ^ c_hashes[ci])
    return total


def _edge_anchor_search(m_hashes, m_base, c_hashes, c_base, seeds,
                        nominal_shift, direction, threshold,
                        search_frames=EDGE_SHIFT_SEARCH_FRAMES):
    '''`_anchor_search`, but resolving the frame SHIFT at the same time as the
    seed -- the one thing the two-anchor path does not have to do, because it
    is handed two independently-measured offset hypotheses and this path is
    handed one whose rounding and whose extraction labelling can each move it
    by a frame (see `EDGE_SHIFT_SEARCH_FRAMES`).

    Per seed, per rung of `VALIDATION_FRAME_LADDER`, in that order -- the same
    nesting and the same stop rules as `_anchor_search`, whose docstring is
    the authority on why:
      * every shift in the search window is offered to `_validate_anchor`,
        UNMODIFIED, at this rung;
      * if none validates, this seed is done -- a stricter window failing is
        evidence the seed's run is short, not that more frames would help;
      * among those that do, the one with the SMALLEST summed Hamming over
        the same window wins, ties broken toward the NOMINAL shift, which is
        the hypothesis the plan actually measured;
      * that winner is then put to `_check_anchor_distinctive`, UNMODIFIED, at
        the same rung. Distinctive -> done. Not distinctive -> widen the rung
        at the same seed, exactly as the two-anchor ladder does.

    Returns `(seed, shift, n_frames_used, reason)`; `seed` is `None` on
    failure, with `reason` `None` if nothing ever validated and the first
    uninformative seed's evidence otherwise -- the same two facts, kept apart
    for the same reason.
    '''
    # `search_frames` defaults to `EDGE_SHIFT_SEARCH_FRAMES`, derived for a
    # millisecond-precise offset. A caller whose offset is COARSER passes its
    # own derived bound (orchestrator stage 4: one fingerprint quantum in
    # frames, plus the labelling frame) -- see `locate_edge_boundary`.
    shift_candidates = sorted(
        range(nominal_shift - search_frames,
              nominal_shift + search_frames + 1),
        key=lambda s: (abs(s - nominal_shift), s))
    last_uninformative_reason = None
    last_uninformative_n_frames = None
    for seed in seeds:
        seed_reason = None
        seed_n_frames = None
        for n_frames in VALIDATION_FRAME_LADDER:
            scored = []
            for shift in shift_candidates:
                if not _validate_anchor(m_hashes, m_base, c_hashes, c_base,
                                        seed, shift, direction, threshold,
                                        n_frames):
                    continue
                distance = _anchor_window_distance(
                    m_hashes, m_base, c_hashes, c_base, seed, shift,
                    direction, n_frames)
                if distance is None:
                    continue
                scored.append((distance, abs(shift - nominal_shift), shift))
            if not scored:
                tools.logs.append(
                    f"scene_anchor: edge_anchor_rung direction={direction} "
                    f"seed={seed} n_frames={n_frames} validated=False "
                    f"shift=none distinctive=n/a\n")
                break
            scored.sort()
            best_shift = scored[0][2]
            distinctive, why_not = _check_anchor_distinctive(
                m_hashes, m_base, c_hashes, c_base, seed, best_shift,
                direction, threshold, n_frames)
            tools.logs.append(
                f"scene_anchor: edge_anchor_rung direction={direction} "
                f"seed={seed} n_frames={n_frames} validated=True "
                f"shift={best_shift} nominal_shift={nominal_shift} "
                f"shift_scores={[(d, s) for d, _, s in scored]} "
                f"distinctive={distinctive}\n")
            if distinctive:
                return seed, best_shift, n_frames, None
            seed_reason, seed_n_frames = why_not, n_frames
        if seed_reason is not None and last_uninformative_reason is None:
            last_uninformative_reason = seed_reason
            last_uninformative_n_frames = seed_n_frames
    return None, None, last_uninformative_n_frames, last_uninformative_reason


def _edge_walk(master_frames, candidate_frames, first_confirmed, shift_frames,
               edge, threshold, n_sustained):
    '''THE pHASH-WALK, owner's rule 3 + the ADDENDUM's three terminations.

    From the validated anchor, step OUTWARD one master frame at a time --
    DOWN toward frame 0 at a head, UP toward the last frame at a tail --
    comparing master against candidate under the single shift the anchor was
    validated at. While the frames match, the common region extends.

    THREE TERMINATIONS, EXACTLY (ADDENDUM):
      1. `sustained_mismatch`   -- `n_sustained` CONSECUTIVE frames disagree.
      2. `master_exhausted`     -- no master frame left to compare.
      3. `candidate_exhausted`  -- no candidate frame left to compare.
    Each side's exhaustion is a FILE fact, bounded by that file's own last
    frame; `_ChunkedFrames.get` is what keeps a chunk boundary from
    impersonating one.

    `boundary_frame` is the LAST MASTER FRAME CONFIRMED MATCHING, for every
    termination and both edges -- the same quantity, so the two call sites do
    not each re-derive it with their own off-by-one.

    `first_confirmed` is the OUTERMOST frame the anchor's own validation
    already proved, and the walk's first step is the one beyond it. It is NOT
    simply the anchor: `_validate_anchor` covers `[seed, seed+n)` forward and
    `[seed-n, seed)` BACKWARD, so a tail anchor's own frame was never
    compared and a walk that started past it would bank an unverified frame
    as matching -- the caller passes `anchor` at a head and `anchor - 1` at a
    tail for exactly that reason. This also means a walk that terminates on
    its very first step still has a CONFIRMED boundary to report and never
    invents one.

    Returns a dict; `reason` is set only when the walk could not run to any
    termination at all (an unreadable chunk), in which case no boundary is
    claimed.
    '''
    step = -1 if edge == "head" else 1
    boundary_frame = first_confirmed
    walked = 0
    mismatch_run = 0
    max_mismatch_run = 0
    termination = None
    unreadable_side = None
    frame = first_confirmed

    while True:
        frame += step
        m_hash, m_state = master_frames.get(frame)
        if m_state == "file_end":
            termination = "master_exhausted"
            break
        if m_state != "ok":
            unreadable_side = "master"
            break
        c_hash, c_state = candidate_frames.get(frame + shift_frames)
        if c_state == "file_end":
            termination = "candidate_exhausted"
            break
        if c_state != "ok":
            unreadable_side = "candidate"
            break
        walked += 1
        if FrameComparer._popcount64(m_hash ^ c_hash) <= threshold:
            boundary_frame = frame
            mismatch_run = 0
        else:
            mismatch_run += 1
            if mismatch_run > max_mismatch_run:
                max_mismatch_run = mismatch_run
            if mismatch_run >= n_sustained:
                termination = "sustained_mismatch"
                break

    if termination is None:
        return {"reason": "edge_walk_unreadable",
                "evidence": (f"side={unreadable_side} frame={frame} "
                             f"walked_frames={walked} "
                             f"master_chunks={master_frames.chunks_read} "
                             f"candidate_chunks={candidate_frames.chunks_read}")}
    return {"reason": None,
            "boundary_frame": boundary_frame,
            "walked_frames": walked,
            "mismatch_run": mismatch_run,
            "max_mismatch_run": max_mismatch_run,
            "termination": termination,
            "master_end_source": master_frames.file_end_source,
            "candidate_end_source": candidate_frames.file_end_source,
            "master_chunks": master_frames.chunks_read,
            "candidate_chunks": candidate_frames.chunks_read,
            "master_seam_deltas": list(master_frames.seam_deltas),
            "candidate_seam_deltas": list(candidate_frames.seam_deltas)}


def locate_edge_boundary(master_path, candidate_path, fps_num, fps_den,
                         bracket_low_ms, bracket_high_ms, offset_ms, edge,
                         master_timeline_ms, candidate_duration_ms,
                         known_match_ms=None, step_ms=None, quantum_ms=None,
                         scene_search_window_sec=None, debug=False,
                         candidate_time_scale=None,
                         shift_search_frames=EDGE_SHIFT_SEARCH_FRAMES):
    '''PUBLIC ENTRY POINT for an EDGE bracket -- owner's ruling
    RULING_20260922_EDGE_SINGLE_ANCHOR.MD and its ADDENDUM.

    `edge` is `"head"` or `"tail"`, and the CALLER classifies: a bracket is an
    edge bracket iff `bracket["edge"] in ("head","tail")`, exactly, with NO
    distance epsilon. That field exists on `leading_bracket`/
    `trailing_bracket` only; `following_bracket` -- attached at
    `change_point_locator.py:3044-3048` ONLY between two SURVIVING segments,
    so provably with content on both sides -- never has it and can never be an
    edge. A distance epsilon from the probe grid would be WRONG and the
    constructors show why: the probe grid sets a bracket's WIDTH (a bound-only
    interior bracket is exactly `PROBE_STEP + PROBE_WINDOW = 100 000 ms`),
    never its DISTANCE from the file edge -- which is ZERO in every edge
    branch by construction (`bracket_low_ms = 0.0` at a head,
    `bracket_high_ms = master_end_ms` at a tail). A 100 000 ms epsilon would
    sweep in errid 5's interior bracket, whose high edge is 150 016 ms from
    the end and which this ruling puts explicitly out of scope.

    `master_timeline_ms` is THE MASTER'S OWN TIMELINE -- what
    `merge_video_chimeric.get_master_timeline_length_ms` reads and what
    `generate_new_file` imposes with `-t duration_best_video` -- never the
    locator's `shortest` (`change_point_locator.py:1842,2913`, which is
    `min(master, candidate)` and is therefore the CANDIDATE's duration
    whenever the master is longer). The distinction is not academic: it is the
    whole of finding 3, errid 5's 37.93 s of unbracketed master fill.

    `offset_ms` is the adjacent plan segment's own offset (candidate_time =
    master_time + offset), the SAME convention every other tier here uses. It
    NOMINATES the frame shift; the anchor RESOLVES it, within
    `EDGE_SHIFT_SEARCH_FRAMES`, against the very extraction the anchor was
    validated on -- see that constant for the two measured reasons a nominal
    shift can be one frame off and for the measurement showing the BOUNDARY is
    invariant under the correction even though the shift is not. The owner's
    "the protocol's >=3-consecutive-frame pHash validation applies to that
    single anchor unchanged" is honoured literally: `_validate_anchor` and
    `_check_anchor_distinctive` are called UNMODIFIED, just offered each
    hypothesis in turn (`_edge_anchor_search`).

    Returns the same dict SHAPE as `locate_scene_anchors` (`declined` True or
    False, never neither) so the head/tail call sites can consume it with the
    interior site's own code.

    `candidate_time_scale` (OPT-IN, orchestrator stage 4, 2026-09-24): the
    same rate relation `locate_scene_anchors` takes under that name. With it,
    `offset_ms` and `candidate_duration_ms` are master-EQUIVALENT time (the
    candidate's raw duration times `r`), and the walk reads the candidate on
    its corrected grid. Absent -> bit-identical to before.

    `shift_search_frames` (OPT-IN, same stage): the half-width of the anchor's
    shift search, default `EDGE_SHIFT_SEARCH_FRAMES`. That constant is derived
    for a MILLISECOND-precise offset (one frame of rounding, one of
    labelling). An offset read off fingerprint points is precise only to one
    fingerprint quantum (~124 ms, ~3 frames at 24000/1001) -- MEASURED on
    errid-70's head: nominal -45, true -48 (50/50 over every block from frame
    50 to 440), unreachable at +/-2 -- so such a caller passes one quantum in
    frames plus the labelling frame, by the constant's own derivation.
    '''
    fps_num = int(fps_num)
    fps_den = int(fps_den)
    if fps_num <= 0 or fps_den <= 0:
        return {"declined": True, "reason": "grid_unmeasured",
                "evidence": f"fps_num={fps_num} fps_den={fps_den}"}
    if edge not in ("head", "tail"):
        return {"declined": True, "reason": "empty_bracket",
                "evidence": f"edge={edge!r} is neither 'head' nor 'tail'"}
    if bracket_high_ms <= bracket_low_ms:
        return {"declined": True, "reason": "empty_bracket",
                "evidence": f"[{bracket_low_ms},{bracket_high_ms}] ms edge={edge}"}

    # THE PRECONDITION, BEFORE ANY ANCHOR OR WALK AND ONCE PER PAIR.
    geometry, crop_filters, geometry_reason = _resolve_geometry(
        master_path, candidate_path)
    tools.logs.append(
        f"scene_anchor: edge_geometry edge={edge} "
        f"master={geometry.get('master')} candidate={geometry.get('candidate')} "
        f"normalised={geometry.get('normalised')} crop={geometry.get('crop')} "
        f"verdict={geometry.get('verdict')} reason={geometry_reason}\n")
    if geometry_reason is not None:
        return {"declined": True, "reason": "edge_geometry_unreconciled",
                "edge": edge, "geometry": geometry,
                "evidence": geometry_reason}

    window_sec = (scene_search_window_sec if scene_search_window_sec is not None
                  else _scene_anchor_config())

    result = None
    rung_window_sec = window_sec
    rung_cd_threshold = CONTENT_DETECTOR_THRESHOLD_LADDER[0]
    for rung in range(WINDOW_LADDER_MAX_RUNGS):
        rung_window_sec = (window_sec if window_sec is None
                           else window_sec * (WINDOW_LADDER_GROWTH_FACTOR ** rung))
        rung_cd_threshold = CONTENT_DETECTOR_THRESHOLD_LADDER[
            min(rung, len(CONTENT_DETECTOR_THRESHOLD_LADDER) - 1)]
        result = _locate_edge_boundary_at_window(
            master_path, candidate_path, fps_num, fps_den,
            bracket_low_ms, bracket_high_ms, offset_ms, edge,
            master_timeline_ms, candidate_duration_ms, rung_window_sec,
            crop_filters, geometry,
            content_detector_threshold=rung_cd_threshold, debug=debug,
            candidate_time_scale=candidate_time_scale,
            shift_search_frames=shift_search_frames)
        matched = not result["declined"]
        tools.logs.append(
            f"scene_anchor: edge_window_ladder_rung edge={edge} rung={rung} "
            f"window_sec={rung_window_sec} cd_threshold={rung_cd_threshold} "
            f"master_seed_count={result.get('master_seed_count')} "
            f"candidate_seed_count={result.get('candidate_seed_count')} "
            f"matched={matched} reason={result.get('reason')} "
            f"evidence={result.get('evidence')}\n")
        if matched or result["reason"] not in EDGE_WINDOW_LADDER_RETRYABLE_REASONS:
            break
    else:
        result = {"declined": True, "reason": "search_window_ceiling_reached",
                  "edge": edge, "geometry": geometry,
                  "master_seed_count": result.get("master_seed_count"),
                  "candidate_seed_count": result.get("candidate_seed_count"),
                  "evidence": f"rungs_tried={WINDOW_LADDER_MAX_RUNGS} "
                              f"base_window_sec={window_sec} "
                              f"final_window_sec={rung_window_sec} "
                              f"final_cd_threshold={rung_cd_threshold} "
                              f"last_reason={result.get('reason')} "
                              f"last_evidence={result.get('evidence')}"}

    # A NORMALISED PAIR THAT COULD NOT SEED AN ANCHOR IS A GEOMETRY ANSWER --
    # BUT ONLY WHEN THE CROP WAS ACTUALLY TESTED AND FAILED. Spec S3a
    # preference order (ii) and the id 33 acceptance arm: the walk must never
    # report a divergence on a crop that is wrong. It used to relabel EVERY
    # establishment failure on a normalised pair, which is wrong in two cases,
    # MEASURED on errid-27's tail (2026-09-24, Fallout end credits):
    #   * `edge_anchor_uninformative` -- a seed VALIDATED (>= 3 consecutive
    #     frames under the crop, at shifts 103-108 at once). The crop is
    #     therefore demonstrably fine; what failed is distinctiveness, a
    #     CONTENT fact. Relabelling it named the crop for a static span.
    #   * `edge_anchor_not_established` with ZERO scene cuts on both sides --
    #     only the bracket-edge seed was offered, so the crop met one frame
    #     pair at most and was never exercised. "No scene cut in reach" is the
    #     cause; the crop is untested, not refuted.
    # So the relabel now requires scene-cut seeds on at least one side AND no
    # validation at any of them -- the one shape a wrong crop produces. The
    # ladder's ceiling carries its last rung's reason and counts, and is
    # judged on those.
    if result["declined"] and geometry.get("normalised"):
        reason = result.get("reason")
        if reason == "search_window_ceiling_reached":
            evidence = result.get("evidence") or ""
            last_reason = next((token for token in ("edge_anchor_not_established",
                                                    "edge_anchor_uninformative")
                                if f"last_reason={token}" in evidence), None)
        else:
            last_reason = reason
        cut_seeds = ((result.get("master_seed_count") or 0)
                     + (result.get("candidate_seed_count") or 0))
        if last_reason == "edge_anchor_not_established" and cut_seeds > 0:
            result = {**result, "reason": "edge_geometry_unreconciled",
                      "geometry": geometry,
                      "evidence": (f"normalisation {geometry.get('crop')} applied, "
                                   f"{cut_seeds} scene-cut seed(s) offered and none "
                                   f"validated under it: {reason} "
                                   f"{result.get('evidence')}")}
        elif last_reason in ("edge_anchor_not_established",
                             "edge_anchor_uninformative") and cut_seeds == 0:
            result = {**result,
                      "evidence": (f"no_scene_cut_in_reach (crop {geometry.get('crop')} "
                                   f"not refuted -- "
                                   + ("a seed validated under it"
                                      if last_reason == "edge_anchor_uninformative"
                                      else "untested")
                                   + f"): {result.get('evidence')}")}
    return result


def _locate_edge_boundary_at_window(master_path, candidate_path, fps_num, fps_den,
                                    bracket_low_ms, bracket_high_ms, offset_ms,
                                    edge, master_timeline_ms, candidate_duration_ms,
                                    window_sec, crop_filters, geometry,
                                    content_detector_threshold=CONTENT_DETECTOR_THRESHOLD_DEFAULT,
                                    debug=False, candidate_time_scale=None,
                                    shift_search_frames=EDGE_SHIFT_SEARCH_FRAMES):
    '''ONE RUNG of `locate_edge_boundary`'s ladder: establish the single
    common-side anchor at this window, then walk. Same split, same reasons and
    same per-rung logging as `_locate_scene_anchors_at_window` -- resolving
    the window and deciding whether it is viable AT ALL is the caller's job,
    once, before any rung.

    Always returns a dict, `declined` True or False, never neither.
    '''
    frame_ms = 1000.0 * fps_den / fps_num

    if window_sec is None or window_sec <= 0:
        return {"declined": True, "reason": "search_window_unviable",
                "edge": edge, "geometry": geometry,
                "evidence": f"scene_search_window_sec={window_sec}"}
    window_frames = int(round((window_sec * 1000.0) / frame_ms))
    if window_frames < MIN_VALIDATION_FRAMES:
        return {"declined": True, "reason": "search_window_too_narrow",
                "edge": edge, "geometry": geometry,
                "evidence": f"scene_search_window_sec={window_sec} -> "
                            f"{window_frames} frames, needs >= "
                            f"{MIN_VALIDATION_FRAMES}"}

    comparer = FrameComparer(master_path, candidate_path,
                             bracket_low_ms / 1000.0, bracket_high_ms / 1000.0,
                             fps_num, fps_den, debug=debug,
                             crop_filters=crop_filters,
                             time_scales=({candidate_path: candidate_time_scale}
                                          if candidate_time_scale is not None
                                          else None))
    m_bracket_first = comparer._frame_index(bracket_low_ms / 1000.0)
    m_bracket_last = comparer._frame_index(bracket_high_ms / 1000.0)
    shift_frames = _nominal_shift_frames(offset_ms, fps_num, fps_den)

    # THE ANCHOR WINDOW straddles the bracket exactly as the interior path's
    # does; what differs is the SEED FILTER below, which keeps only the common
    # side. Extracting a little of the outer side too is deliberate and costs
    # nothing: `_check_anchor_distinctive` probes at +/-4 and +/-8 frames and
    # an unreadable probe frame would silently weaken the guard.
    m_win_start = max(0, m_bracket_first - window_frames)
    m_win_end = m_bracket_last + window_frames
    candidate_margin_frames = CANDIDATE_SEED_MARGIN_MULTIPLIER * window_frames
    c_win_start = max(0, m_win_start - candidate_margin_frames + shift_frames)
    c_win_end = m_win_end + candidate_margin_frames + shift_frames

    master_rate = Fraction(fps_num, fps_den)
    m_win_start_sec = Fraction(m_win_start * fps_den, fps_num)
    m_win_span_sec = Fraction((m_win_end - m_win_start) * fps_den, fps_num)
    c_win_start_sec = Fraction(c_win_start * fps_den, fps_num)
    c_win_span_sec = Fraction((c_win_end - c_win_start) * fps_den, fps_num)

    candidate_rate, candidate_rate_reason = _probe_frame_rate(candidate_path)
    # Corrected rate for a speed-changed candidate -- same arithmetic and same
    # reason as the interior path (`_locate_scene_anchors_at_window`).
    if candidate_rate is not None and candidate_time_scale is not None:
        candidate_rate = candidate_rate / Fraction(candidate_time_scale)

    m_scan_start = _frames_at_rate(m_win_start_sec, master_rate)
    m_scan_frames = _frames_at_rate(m_win_span_sec, master_rate)
    if candidate_rate is None:
        c_scan_start = c_scan_frames = None
    else:
        c_scan_start = _frames_at_rate(c_win_start_sec, candidate_rate)
        c_scan_frames = _frames_at_rate(c_win_span_sec, candidate_rate)
    tools.dev_log(
        f"scene_anchor: edge_scan_window_conversion edge={edge} "
        f"master_rate={master_rate.numerator}/{master_rate.denominator} "
        f"candidate_rate="
        f"{'unmeasured:' + str(candidate_rate_reason) if candidate_rate is None else str(candidate_rate.numerator) + '/' + str(candidate_rate.denominator)} "
        f"master_scan=[{m_scan_start},+{m_scan_frames}) "
        f"candidate_scan=[{c_scan_start},+{c_scan_frames})\n")

    m_base, m_hashes = _extract_hashes(comparer, master_path,
                                       float(m_win_start_sec), float(m_win_span_sec))
    c_base, c_hashes = _extract_hashes(comparer, candidate_path,
                                       float(c_win_start_sec), float(c_win_span_sec))
    if not m_hashes or not c_hashes:
        return {"declined": True, "reason": "frames_unextractable",
                "edge": edge, "geometry": geometry,
                "evidence": f"master_frames={len(m_hashes)} "
                            f"candidate_frames={len(c_hashes)}"}

    threshold = ANCHOR_HAMMING_THRESHOLD_DEFAULT
    master_cuts, master_cuts_failed = _scene_cut_frames(
        master_path, m_scan_start, m_scan_frames, content_detector_threshold, debug)
    if candidate_rate is None:
        candidate_cuts = None
        candidate_cuts_failed = f"candidate_grid_unmeasured:{candidate_rate_reason}"
    else:
        candidate_cuts, candidate_cuts_failed = _scene_cut_frames(
            candidate_path, c_scan_start, c_scan_frames,
            content_detector_threshold, debug)
    master_cuts_seeds = master_cuts or []
    candidate_cuts_seeds = [_frame_on_grid(f, candidate_rate, master_rate)
                            for f in (candidate_cuts or [])]

    # SEEDS COME FROM THE COMMON SIDE, NEAREST FIRST -- AND NEVER ONLY FROM
    # THE BRACKET EDGE. That last clause is the one the measurement bit on.
    # MEASURED at Undead Unluck S01E13's head (2026-09-22): the file opens on
    # a near-static card, and master frames 20 through 28 ALL read Hamming 0
    # against the candidate's very first frame -- so the bracket-edge seed
    # VALIDATES and is nonetheless UNINFORMATIVE, matching under several
    # competing shift hypotheses at once. `_check_anchor_distinctive` is what
    # rejects it, and `_anchor_search`'s nearest-first ordering is what then
    # moves on to a real scene cut further inside the common region. File
    # edges are SYSTEMATICALLY static -- black, a logo, a fade-in -- so this
    # is the normal case at an edge, not the exception, and seeding from
    # scene cuts inside the common region is the difference between working
    # and declining.
    if edge == "head":
        seeds = sorted(
            {m_bracket_last}
            | {f for f in master_cuts_seeds if f >= m_bracket_last}
            | {f - shift_frames for f in candidate_cuts_seeds
               if f - shift_frames >= m_bracket_last})
        direction = "forward"
        anchor_side = "B"
    else:
        seeds = sorted(
            {m_bracket_first}
            | {f for f in master_cuts_seeds if f <= m_bracket_first}
            | {f - shift_frames for f in candidate_cuts_seeds
               if f - shift_frames <= m_bracket_first},
            reverse=True)
        direction = "backward"
        anchor_side = "A"

    nominal_shift_frames = shift_frames
    anchor, shift_frames, anchor_n_frames, anchor_reason = _edge_anchor_search(
        m_hashes, m_base, c_hashes, c_base, seeds, nominal_shift_frames,
        direction, threshold, search_frames=shift_search_frames)

    if anchor is None:
        # TWO DISTINCT FACTS, same split and same reasoning as the two-anchor
        # path's own `anchor_uninformative` vs `anchors_not_established`:
        # a seed validated but every one failed the distinctiveness probe is a
        # MEASURED, content-based refutation; no seed validating at all is
        # nothing to measure. Folding them would manufacture a measurement
        # that never happened.
        payload = {"declined": True, "edge": edge, "geometry": geometry,
                   "master_seed_count": len(master_cuts_seeds),
                   "candidate_seed_count": len(candidate_cuts_seeds)}
        if anchor_reason:
            return {**payload, "reason": "edge_anchor_uninformative",
                    "evidence": f"edge={edge} side={anchor_side} "
                                f"seeds={len(seeds)} "
                                f"nominal_shift={nominal_shift_frames} "
                                f"reason={anchor_reason} "
                                f"n_frames={anchor_n_frames}"}
        return {**payload, "reason": "edge_anchor_not_established",
                "evidence": f"edge={edge} side={anchor_side} "
                            f"seeds={len(seeds)} "
                            f"nominal_shift={nominal_shift_frames} "
                            f"master_cuts={len(master_cuts_seeds)} "
                            f"master_detector_failed={master_cuts_failed} "
                            f"candidate_cuts={len(candidate_cuts_seeds)} "
                            f"candidate_detector_failed={candidate_cuts_failed}"}

    # THE WALK'S OWN FRAME SUPPLY -- chunked, and bounded by each FILE's own
    # last frame, which is the ADDENDUM's own condition ("BOUNDED at read time
    # by each file's own max frame count"). The master's bound is its
    # TIMELINE, by ruling: the output has exactly that many frames. The
    # candidate's declared bound is a CEILING only, never the authority -- see
    # `_ChunkedFrames` for the measurement that separates a container duration
    # from a video stream's real last frame.
    master_last_frame = _frames_at_rate(
        Fraction(str(master_timeline_ms)) / 1000, master_rate) - 1
    candidate_last_frame_ceiling = None
    if candidate_duration_ms is not None:
        candidate_last_frame_ceiling = _frames_at_rate(
            Fraction(str(candidate_duration_ms)) / 1000, master_rate) - 1

    master_frames = _ChunkedFrames(comparer, master_path, "master",
                                   fps_num, fps_den, master_last_frame, debug,
                                   initial_base=m_base, initial_hashes=m_hashes)
    candidate_frames = _ChunkedFrames(comparer, candidate_path, "candidate",
                                      fps_num, fps_den,
                                      candidate_last_frame_ceiling, debug,
                                      initial_base=c_base, initial_hashes=c_hashes)

    # `_validate_anchor` covers `[seed, seed+n)` FORWARD (head) and
    # `[seed-n, seed)` BACKWARD (tail) -- so the outermost frame the anchor
    # actually proved is the seed itself at a head and the frame BEFORE it at
    # a tail. The walk starts from what was proven, never from what was only
    # seeded.
    first_confirmed = anchor if edge == "head" else anchor - 1
    walk = _edge_walk(master_frames, candidate_frames, first_confirmed,
                      shift_frames, edge, threshold,
                      EDGE_WALK_SUSTAINED_MISMATCH_FRAMES)
    if walk["reason"] is not None:
        return {"declined": True, "reason": walk["reason"], "edge": edge,
                "geometry": geometry,
                "master_seed_count": len(master_cuts_seeds),
                "candidate_seed_count": len(candidate_cuts_seeds),
                "evidence": f"edge={edge} anchor_frame={anchor} "
                            f"shift={shift_frames} {walk['evidence']}"}

    boundary_frame = walk["boundary_frame"]
    termination = walk["termination"]

    # THE FILL'S LENGTH IS COUNTED, NOT INFERRED (ADDENDUM outcome 3, verbatim:
    # "the number of master frames remaining past the last compared frame IS
    # the length to take from the master ... duration = frames x the exact
    # rational frame time. No probe, no estimate"). `addition_frames` is set
    # ONLY on `candidate_exhausted`, because that is the only termination that
    # names a master ADDITION: the other two name a trim (outcome 2) or a
    # replacement of divergent content (outcome 1), whose lengths the existing
    # plan machinery already derives from the boundary alone.
    addition_frames = None
    if termination == "candidate_exhausted":
        addition_frames = (boundary_frame if edge == "head"
                           else master_last_frame - boundary_frame)
        addition_frames = max(0, addition_frames)
    addition_ms = (None if addition_frames is None
                   else str(_exact_ms_from_frame(addition_frames, fps_num, fps_den)))

    if termination == "candidate_exhausted":
        net_kind = "master_addition"
    elif termination == "master_exhausted":
        net_kind = "candidate_excess_trimmed"
    else:
        # THE THIRD TOKEN, AND IT IS NOT IN THE SPEC'S TABLE -- reported as a
        # deviation rather than folded into one of the two that were. A
        # sustained mismatch is neither a pure addition (the candidate HAS
        # content there, it simply disagrees) nor a pure trim (master fill
        # does take its place on the output timeline). Naming it as either
        # would make a census of "how much master was added because the
        # candidate ran out" count content that diverged instead.
        net_kind = "master_replacement"

    evidence = (f"edge={edge} anchor={anchor} anchor_side={anchor_side} "
                f"anchor_n_frames={anchor_n_frames} shift={shift_frames} "
                f"nominal_shift={nominal_shift_frames} "
                f"boundary={boundary_frame} walked={walk['walked_frames']} "
                f"mismatch_run={walk['mismatch_run']} "
                f"max_mismatch_run={walk['max_mismatch_run']} "
                f"termination={termination} "
                f"master_end_source={walk['master_end_source']} "
                f"candidate_end_source={walk['candidate_end_source']} "
                f"master_chunks={walk['master_chunks']} "
                f"candidate_chunks={walk['candidate_chunks']} "
                f"master_seam_deltas={walk['master_seam_deltas']} "
                f"candidate_seam_deltas={walk['candidate_seam_deltas']} "
                f"master_last_frame={master_last_frame} "
                f"master_cuts={len(master_cuts_seeds)} "
                f"master_detector_failed={master_cuts_failed} "
                f"candidate_cuts={len(candidate_cuts_seeds)} "
                f"candidate_detector_failed={candidate_cuts_failed} "
                f"seeds={len(seeds)}")

    # LOGGED ONCE PER EDGE, WITH THE NUMBERS -- ruling point 5 ("locating the
    # boundary emits its verdict with numbers") and spec S5 item 7 (report
    # `termination`, `addition_frames` and `mismatch_run` on EVERY outcome, so
    # the N=3-vs-N=4 question stays measurable rather than re-litigated).
    tools.logs.append(
        f"scene_anchor: edge_walk edge={edge} anchor_frame={anchor} "
        f"anchor_n_frames={anchor_n_frames} anchor_side={anchor_side} "
        f"shift_frames={shift_frames} "
        f"nominal_shift_frames={nominal_shift_frames} "
        f"boundary_frame={boundary_frame} "
        f"walked_frames={walk['walked_frames']} "
        f"mismatch_run={walk['mismatch_run']} "
        f"max_mismatch_run={walk['max_mismatch_run']} "
        f"termination={termination} net_kind={net_kind} "
        f"addition_frames={addition_frames} addition_ms={addition_ms} "
        f"threshold={threshold} "
        f"n_sustained={EDGE_WALK_SUSTAINED_MISMATCH_FRAMES} "
        f"geometry_normalised={geometry.get('normalised')}\n")

    return {
        "declined": False,
        "method": "scene_anchor_single_edge",
        "edge": edge,
        "grid": {"num": fps_num, "den": fps_den},
        "anchor_frame": anchor,
        "anchor_n_frames": anchor_n_frames,
        "anchor_side": anchor_side,
        "shift_frames": shift_frames,
        "nominal_shift_frames": nominal_shift_frames,
        "boundary_frame": boundary_frame,
        "walked_frames": walk["walked_frames"],
        "mismatch_run": walk["mismatch_run"],
        "max_mismatch_run": walk["max_mismatch_run"],
        "termination": termination,
        "net_kind": net_kind,
        "addition_frames": addition_frames,
        "addition_ms": addition_ms,
        "master_last_frame": master_last_frame,
        "geometry": geometry,
        "master_seed_count": len(master_cuts_seeds),
        "candidate_seed_count": len(candidate_cuts_seeds),
        "derived_ms": {
            "boundary_ms": f"{round(float(_exact_ms_from_frame(boundary_frame, fps_num, fps_den)), 2)}",
        },
        "evidence": evidence,
    }
