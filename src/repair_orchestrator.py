# -*- coding: utf-8 -*-
"""
repair_orchestrator.py -- the owner's flat orchestrator (RULING_20260922_ORCHESTRATOR_
ARCHITECTURE.MD, plus its seven addenda), built through the staged switch described in
`VMSAM_HELP_AI/architect/cases/DESIGN_orchestrator_mapping.md` section 5.

WHAT THIS IS, AND WHAT IT IS NOT YET. The owner's shape is "une fonction qui agence des
resultats": each big step is LAUNCHED here, RETURNS a result, and this module arranges the
results -- the steps do not call each other. That shape is implemented. The steps behind it are
landed in the design's own order, and the ones that are not landed yet DECLINE, BY NAME, WITH A
MEASUREMENT CLASS. Nothing here returns True on work it did not do.

    landed   step 1  master self-check                 -- reuses `master_self_check`, as-is
    landed   step 2  the similarity gate + rate arm    -- `rate_arm` (ADDENDUM 30.5): every
                                                          named ratio in both engines,
                                                          re-fingerprinted and aligned
    landed   the prime (ADDENDUM 21.1)                 -- EVERY couple of the comparison
                                                          language fingerprinted (x2) and
                                                          aligned (`b2_align`) BEFORE the gate,
                                                          so the gate and the rate decision
                                                          read all of them (`prime_couples`)
    landed   the re-prime at a factor (ADDENDUM 21.6)  -- at a confirmed factor other than 1
                                                          the candidate side of every couple is
                                                          re-fingerprinted speed-corrected, UP
                                                          STREAM of chimeric, which never
                                                          resamples; the interim producer is
                                                          `rate_resample_routing` until the
                                                          rate arm's seam replaces it (see
                                                          `speed_factor`)
    landed   step 3c zones -> holes, per couple        -- this module; holes closer than the
                                                          resolver's reach CLUSTER (one shared
                                                          scene pass), never fuse (owner
                                                          2026-09-24)
    landed   step 3d the multi-couple cross-check      -- this module
    landed   step 3e the UNION of every couple's holes -- `union_holes` (ADDENDUM 21.8), on the
                                                          file's clock (container delays folded
                                                          per couple), plus the same-offset gaps
                                                          the coalescing absorbed, logged and,
                                                          above the resolver's reach, checked by
                                                          the video (ADDENDUM 21.9)
    landed   step 4  THE AUDIO BOUNDS, THE VIDEO PINS  -- ADDENDUM 25: the millisecond walk
                                                          (`audio_walk`) on the reference couple
                                                          fixes offsets, steps, fill widths and
                                                          the intervals a cut may lie in; the
                                                          video (`scene_anchor`, two anchors /
                                                          one anchor) only chooses the frame
                                                          inside them or says there is no cut
                                                          (a slip); see `audio_transitions`,
                                                          `audio_edges`
    landed   step 5  plan application                  -- `apply_plan`: the resolved frames
                                                          laid as zones and fills, each track's
                                                          own sub-frame offset, resample at the
                                                          exact rational on a rate pair, cues and
                                                          chapters re-timed, the kept assembly /
                                                          verify / fabricated-delivery gates

*** WIRED: THIS IS THE LIVE CHAIN (stage 6, 2026-09-24, owner's REMPLACEMENT DIRECT --
ADDENDUM 8 points 4 and 6). `merge_video_repair.repair_not_compatible_videos`, the zone-A entry,
calls `repair()` once per refused candidate; the legacy chain (`get_plan_from_locator`, the band
routing, `change_point_locator`) was removed in the same batch. Step 5 landed after it: the
repaired object travels back to the entry on `merge_video_repair.REPAIR_SEAM_ATTRIBUTE`, and the
`repaired` terminal is written by `apply_plan` through `record()`. ***

RETURN CONTRACT: a BOOLEAN (owner's ADDENDUM 1, point 4). True = a plan was found AND the
temporary chimeric file was created successfully. False = everything else. The boolean has NO
room for "could not measure", which the standing invariant says must never be confused with
"measured and refused" -- so that distinction lives in the TOKENS, not in the return: every
False leaves behind a `cause=<token>` whose MEASUREMENT CLASS is emitted beside it, one of

    ran_conclusive_negative   the instrument ran and returned a negative
    could_not_run             the instrument could not be run, or is not built yet

(A third class, `owner_choice`, existed for the owner's restoration deferral -- ADDENDUM 7. The
owner lifted that deferral in ADDENDUM 8, the class lost its only token, and it was retired with
it rather than left as a name nothing can emit.)

A reader who sees `False` and wants to know which of the three it was reads the token; a reader
who never looks gets a boolean that is safe either way, because all three mean "do not ship".

SEQUENTIAL, DELIBERATELY. The design's section 4 measured that this code path carries NO
threading today (`frame_compare.py` imported `Thread` and never called it -- removed as
dead with the switch),
that the cost bottleneck is per-track ffmpeg extraction rather than hole resolution, and that
the process already lives under a wedge-prone frozen `Pool.terminate()` (CASE id 6). Parallelism
is the design's stage 7 and lands AFTER the switch, so that a 317 re-score regression can be
attributed to one change and not two. Nothing in this module starts a thread or a process pool.

LOGGING, PER THE RULING: every step emits a LAUNCH line and a RESULT line through
`tools.dev_log` (gated on `tools.dev`) -- the inter-couple disagreement dump included since
ADDENDUM 21.7 ("Logs de desaccord inter-couples en dev ; record() porte la cause") -- and the
terminal per-candidate verdict goes through `merge_video_repair.record` (`log_always`), which
carries the cause token and the cross-verification report, so the ledger keeps reading one line
shape and the permanent truth is the record, not a side line.
"""
from decimal import Decimal
from fractions import Fraction
import math
from os import path, remove
import statistics
import subprocess
import time

import audioCorrelation
import audio_extract
import banded_seed_alignment
import repair_log
import tools

MODALITY = "repair_orchestrator"

# ---------------------------------------------------------------------------
# DERIVED CONSTANTS -- each one states what it is derived FROM. A constant with
# no derivation is a tuning knob, and a repair conditioned on a tuning knob is
# not a repair.
# ---------------------------------------------------------------------------

# THE <10 s RULE (ruling, chimeric step 4: "les trous separes de MOINS DE 10 s se FUSIONNENT"),
# AS THE OWNER RE-RULED IT (2026-09-24, Addendum 22 pending): "THE <10 s MERGE IS A SCHEDULING
# MERGE, NOT A DATA MERGE". Holes closer than this form a CLUSTER that shares ONE scene-detection
# pass (`cluster_holes`, `scene_anchor`'s `cluster_window` / `scan_cache`), but each hole keeps its
# own bounds and step, and the aligned ISLAND between them keeps its zone and its b2 similarity --
# fusing them erased the island and made the two walks contradict each other inside common
# content. This is NOT a free number: it is the frame-exact resolver's OWN SEARCH REACH,
# `scene_anchor.SCENE_SEARCH_WINDOW_SECONDS_DEFAULT = 10.0` -- two holes closer than that have
# OVERLAPPING +/-10 s scene searches, which is exactly why one pass serves both. Imported rather than restated
# (a restatement drifts; this campaign has already measured one doing so) with a literal fallback
# for the case where `scene_anchor` is not importable, and the fallback is logged, never silent.
try:
    import scene_anchor as _scene_anchor
    HOLE_MERGE_WINDOW_SECONDS = float(_scene_anchor.SCENE_SEARCH_WINDOW_SECONDS_DEFAULT)
    _HOLE_MERGE_SOURCE = "scene_anchor.SCENE_SEARCH_WINDOW_SECONDS_DEFAULT"
except Exception:                                                        # noqa: BLE001
    HOLE_MERGE_WINDOW_SECONDS = 10.0
    _HOLE_MERGE_SOURCE = "literal fallback -- scene_anchor unimportable"

# THE INTER-COUPLE POSITIONAL WINDOW. Same derivation, same number, and deliberately the SAME
# CONSTANT rather than a second one that happens to match: a cluster is exactly "the events one
# anchor search would reach", so it can only be the resolver's reach.
INTERCOUPLE_POSITION_WINDOW_SECONDS = HOLE_MERGE_WINDOW_SECONDS

# THE UNION OF HOLES ACROSS COUPLES (ADDENDUM 21.8; owner 2026-09-24: "a union of SEARCH
# REGIONS only") uses the SAME reach, and not a second number: a hole of one couple within the
# reach of a hole of ANOTHER couple is the same event seen twice (the cross-check measured the same
# cut up to 7.7 s apart between couples), so their spans become one search region (min start /
# max end). Two holes of the SAME couple are never united -- they are two events, and at most
# share a cluster. With a single couple the union is therefore the identity.
#
# THE ABSORBED-GAP VIDEO CHECK (ADDENDUM 21.9: gaps at the same offset are "absorbes pour le plan
# MAIS logges (span) et, au-dela d'un seuil nomme, confirmes par le test no-cut video"). The
# threshold is the SAME reach again, derived the same way: an unmatched stretch shorter than the
# resolver's +/-10 s scene search is not a search region of its own -- the anchors that would
# test it sit in the aligned content on both sides, inside one search window, and an edit that
# kept the offset unchanged across it (the only kind that CAN hide there) is below what one
# anchor pair distinguishes from the aligned zone around it. At or above the reach the stretch IS
# a region the resolver can examine by itself, so it is examined: every absorbed gap is logged
# with its span, and those at or above this are handed to the video's no-cut test.
ABSORBED_GAP_VIDEO_CHECK_SECONDS = HOLE_MERGE_WINDOW_SECONDS

# THE ISLAND VIDEO CHECK (owner ruling 2026-09-24, Addendum 22 pending: "above the named
# threshold, confirm the island by the video no-cut test"). An island is shorter than the reach
# by definition (that is what put its two holes in one cluster), so the reach cannot be its
# threshold. What bounds the test is the INSTRUMENT: `scene_anchor.MIN_VALIDATION_FRAMES`, the
# fewest frames on which the video validates anything at all. An island at or above it is
# testable and IS tested; one under it cannot be, and is logged `untestable`, never passed.
try:
    ISLAND_VIDEO_CHECK_MIN_FRAMES = int(_scene_anchor.MIN_VALIDATION_FRAMES)
except Exception:                                                        # noqa: BLE001
    ISLAND_VIDEO_CHECK_MIN_FRAMES = 3

# THE INTER-COUPLE STEP TOLERANCE, calibrated by measurement, as the owner's ADDENDUM 1 point 2
# delegated ("A CALIBRER PAR MESURE sur les paires reelles multi-pistes; depart: +/-1 point de
# quantum"). MEASURED on errid-232's five real couples over one real ~6 s editorial cut, all ten
# couple pairs: |delta step| = 0 / 2 / 124 / 126 ms = 0.00 / 0.02 / 1.00 / 1.02 quanta. So the
# owner's starting value is measured CORRECT for step magnitude -- and the observed 1.02 is why
# the comparison is written in MILLISECONDS against 1.5 quanta rather than as an integer
# `< 1` on points: a strict integer comparison would refuse the very agreement it was
# calibrated on. (The same measurement found the owner's value WRONG by two orders of magnitude
# for zone POSITION -- 0 to 216 quanta on the same event -- which is why position is clustered
# with the window above and never compared point for point.)
#
# CALIBRATED ON n = 1 PAIR WITH 1 REAL EVENT, AND THE RE-MEASUREMENT THAT WAS SUPPOSED TO FIX
# THAT IS NOW DONE AND DID NOT. This comment used to say "errid-24, errid-70, errid-84 and
# errid-99 are available in the corpus and were not run". They have now all been run -- by an
# independent tester across all 20 reachable pairs on 28 language combinations, and again here
# on errid-24/es specifically -- and NONE of them produces a second ABOVE-FLOOR couple at any
# event, so none of them adds a single calibration point. MEASURED on errid-24/es, the corpus's
# widest multi-couple case (4 couples): 40 event clusters, every one of them
# `below_floor_only_excluded`, zero above-floor events, `agree=True`. Corpus-wide the agreement
# test was reached exactly TWICE in 28 runs, both on errid-232 and both from the SAME physical
# edit -- spread 124.03 ms against a 186.06 ms tolerance on ja, 0.34 ms on en.
#
# SO THE CONSTANT STANDS AT ITS MEASURED VALUE AND ITS SAMPLE SIZE STANDS AT ONE EVENT, and that
# is now a statement about the CORPUS rather than about anyone's budget: settling this needs a
# pair with two or more couples that BOTH see the same above-floor cut, and the corpus contains
# exactly one such pair. The mechanism was separately exercised by construction -- it accepts to
# exactly 1.50 quanta inclusive, refuses from 1.51, and still refuses a constructed 2-quanta
# disagreement above the floor -- so what is untested is the threshold's placement on real
# media, not whether it works.
INTERCOUPLE_STEP_TOLERANCE_QUANTA = 1
INTERCOUPLE_STEP_TOLERANCE_SLACK = 1.5

# THE TIME BUDGETS (owner, ADDENDUM 26.3, 2026-09-25): a job that does not finish is a NAMED
# DECLINE with its measurement, never a blocked container -- the file comes back next wave. Measured
# on the 401a9f2e wave: an ordinary job takes ~7.5 min, two runaway jobs (> 1 h each) ate 60 % of
# the machine time (memo THROUGHPUT_ANALYSIS_20260925). The values are the ruling's.
ALIGNMENT_BUDGET_S = 120.0          # one couple's b2_align (`alignment_budget_exceeded`)
HOLE_BUDGET_S = 300.0               # one hole's frame-exact search (`hole_budget_exceeded`)
# One candidate's whole repair (`repair_budget_exceeded`), PROPORTIONAL to the master's video
# (ADDENDUM 26.8): 20 min per started 30-min slice, floor 20 min (60 min -> 40 min, 2 h -> 80 min).
# A budget that declines a file for its length alone is a defect, not a cap (Fallout S01E02, a
# 60-min master, legitimately approaches 20 min).
REPAIR_BUDGET_PER_SLICE_S = 1200.0
REPAIR_BUDGET_SLICE_S = 1800.0
# THE HOLE SANITY BOUND (ADDENDUM 26.2): an interior hole wider than this is not searched. Measured:
# resolved holes <= 2 s, absorbed <= 7.7 s, edge additions <= 27 s on the corpus; id 691 carried an
# interior "hole" of 6,803 s past the end of its master's video.
INTERIOR_HOLE_MAX_SPAN_S = 300.0

# THE 15-SECOND EDGE-ADDITION MARKER THRESHOLD (owner's ADDENDUM 5, cited by this constant's
# existence per that addendum's own clause (d): "le seuil est une constante nommee avec ce
# ruling en reference"). If the ONLY work done is an addition at the head and/or the tail
# totalling under this, the track is NOT tagged chimeric -- it is the original track with a
# marginal completion, a generalisation of the tail-gap pad (<1 s) that never tagged either.
# The bounds are the addendum's, not this file's: (a) ANY interior splice tags, whatever its
# size; (b) edge additions totalling >= this threshold tag; (c) `resampled:<factor>` is an
# INDEPENDENT marker that rides on a transformed track with or without a splice; (d) the added
# durations are logged per edge ALWAYS, tagged or not -- only the COMPETITIVE marker follows the
# threshold, because its real job is to stop `keep_best_audio`'s "intact wins" rule from
# demoting a near-intact track over 15 s of edge completion.
EDGE_ADDITION_CHIMERIC_TAG_THRESHOLD_SECONDS = 15.0

# A CAP ON HOLES PER COUPLE, with a named decline when exceeded. MEASURED motivation (design
# section 6.4.1): zone fragmentation varies enormously by track, so a design that resolves every
# hole would launch dozens of frame-exact searches per couple, each of them an unbounded ffmpeg
# decode. Coalescing same-offset zones is the first line of defence; this is the second.
#
# RE-DERIVED 2026-09-22 AGAINST THE FULL CORPUS, and the previous value was wrong by its own
# rule. It read 22, set as "well over twice" a maximum of 9 measured on SIX couples of TWO
# pairs. An independent tester then measured every reachable pair -- 31 couples over 20 pairs --
# and the real distribution is
#     21, 18, 18, 18, 15, 10, 9, 9, 7, 7, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0
# so the true maximum is 21 (errid-99 on fr) and the old cap left ONE HOLE of headroom above
# real media. Applying the comment's own stated rule to the real maximum gives 45, which is what
# this now is. The cap never fired in 31 couples and is not meant to: the pathological pairs it
# was imagined for do NOT reach it -- wrong-episode and wrong-series pairs produce ONE giant hole
# or fail to align, and are caught by the similarity gate or by `spans_whole_file`.
#
# AND THE SAME MEASUREMENT SHOWS WHY THE PER-LANGUAGE SPREAD MATTERS: the SAME physical pair
# reads 2 holes on `en` and 21 on `fr` (errid-99). The comparison language is chosen upstream (by
# `mergeVideo.get_delay`'s caller, passed down since ADDENDUM 20) among languages that may be
# equals -- so a cap set near real media would let that upstream choice decide whether a pair is
# refused.
#
# ITS CLASS CHANGED WITH ITS VALUE, and that is the more important half. The token used to be
# `ran_conclusive_negative` -- "the instrument ran and returned a negative about this pair". It
# is not: "too fragmented for the budget I gave myself" is a statement about THIS ORCHESTRATOR'S
# resolver budget, not about the media. errid-99 is, by the corpus's own classification, one of
# its CLEANEST edit pairs (two constant-offset plateaus, NCC 0.98-0.99), and one more hole would
# have had it recorded as a conclusive refusal. It is now `could_not_run`.
MAX_HOLES_PER_COUPLE = 45

# THE COVERAGE FLOOR -- the step-2 gate's second arm, and the fix for a BREAKING finding.
#
# WHAT WENT WRONG WITHOUT IT. The aligner's `single_segment_no_cut` means "no offset STEP was
# found". It does NOT mean "the two audios line up", and until this arm existed the gate read it
# as though it did. Measured by an independent tester on the committed tip: errid-70 (a confirmed
# PAL 25/23.976 pair whose two fr tracks are 51.87 s apart) returned that token with n_zones=0
# and coverage 0.000, the gate returned should_sweep=False, the sweep never ran, and the
# orchestrator logged `audios_fully_compatible_offset_only` -- a positive claim of compatibility
# about a pair one frame rate apart -- which also made the owner's then-standing
# `restoration_deferred` deferral (ADDENDUM 7, lifted since by ADDENDUM 8) unreachable, since that
# gate tested `speed_factor != 1`. The same pair on
# `en` took the correct path, which is what proved it was the gate and not the media.
#
# THE NUMBER IS READ OFF A BIMODAL DISTRIBUTION, NOT CHOSEN. Coverage over all 31 success-token
# couples of the 20-pair corpus:
#     0.000 errid-70/fr | 0.064 errid-121/ja | 0.067 errid-123/ja | 0.304, 0.306 errid-24/es
#     <---------------------------- EMPTY BAND ---------------------------->
#     0.501 errid-99/en | 0.587 errid-84/en | ... 26 more couples up to 1.000
# Nothing lands between 0.306 and 0.501. 0.40 sits in the middle of that gap with ~0.09 of margin
# below and ~0.10 above, and it separates all five pathological couples from all twenty-six
# healthy ones. The five are not a mixed bag: two are wrong-episode pairs (errid-121, errid-123 --
# CANNOT-HELP, where "a repair that appeared to succeed would splice the wrong episode into the
# library"), one is the PAL pair above, and two are the degraded half of errid-24's four couples.
#
# THIS ARM IS TERMINAL, unlike the rate-ladder arm below, and the ruling says why in its own
# words (its `run_speed_sweep` is the rate arm since ADDENDUM 30.5): step 2 is "similarite faible
# master<->candidat ? OUI -> run_speed_sweep : le resample peut-il la remonter ? ... None -> return, message dans tools.logs (« similarite moyenne faible,
# impossible de l'augmenter par resample »)". Low coverage IS low similarity, measured; a sweep
# that cannot raise it IS the ruling's return. That also closes the wrong-episode hazard
# independently of stage 4: errid-121 and errid-123 currently decompose into a single `head` hole
# of 1348.5 s on a ~1450 s file, which the moment a frame-exact resolver exists would drive it
# across 93 % of the wrong programme.
MASTER_AXIS_COVERAGE_FLOOR = 0.40

# THE PITCH PROBE'S WINDOW. Not a new number: `pal_pitch_confirmer.confirm_pitch` and
# `confirm_ntsc` both default to 180 s and `pal_speed_verdict` passes that default through, so
# the pitch layer is asked on the window it was calibrated on (its NTSC_TOLERANCE was measured at
# n=24 over exactly these windows). Restating a different one here would silently re-scope a
# tolerance measured somewhere else. Imported rather than retyped, with a literal fallback.
try:
    import pal_pitch_confirmer as _pal_pitch_confirmer
    PITCH_PROBE_WINDOW_SECONDS = float(
        _pal_pitch_confirmer.confirm_pitch.__defaults__[0])
    _PITCH_WINDOW_SOURCE = "pal_pitch_confirmer.confirm_pitch's own default"
except Exception:                                                        # noqa: BLE001
    PITCH_PROBE_WINDOW_SECONDS = 180.0
    _PITCH_WINDOW_SOURCE = "literal fallback -- pal_pitch_confirmer unimportable"

# THE FLOOR UNDER A SHORTENED PITCH WINDOW. A pair shorter than 180 s gets half its usable span
# rather than a refusal, but not below this: `audio_extract.extract_audio_window` refuses an
# extraction under one second of audio by construction, and a spectral ratio measured over a
# window that short carries no frequency resolution worth routing a filter on. Chosen, not
# derived, and it is a FLOOR on the instrument rather than a threshold on a verdict.
PITCH_PROBE_WINDOW_MINIMUM_SECONDS = 30.0

# THE RATE-RELATION ARM OF THE STEP-2 GATE IS NOW CALIBRATED, AND THE INSTRUMENT IS NOT THE ONE
# THE DESIGN PROPOSED. That matters enough to record both.
#
# WHAT WAS REFUTED. Design section 3.6 proposed reading `drift_fit` -- `best_shift_trace`'s slope
# with its R^2 -- as "there is a rate relation here". MEASURED, that instrument cannot carry the
# arm, for two separate reasons:
#   (a) SLOPE-WITH-HIGH-R^2 FIRES ON PAIRS THAT HAVE NO RATE RELATION. errid-202 (content edits
#       only) reads slope -0.00237 at r^2 0.790, a LARGER magnitude than the real NTSC pair's
#       +0.00037 at r^2 0.601. The bake-off dossier had already measured the same trap from the
#       other end (errid 213 fitting "ratio 1.0010739", 0.07 % from NTSC, with no relation at
#       all), and `change_point_locator:1132-1136` records it a third time independently.
#   (b) THE TRACE IS BLIND WHEN THE OFFSET IS LARGE. `best_shift_trace` starts at offset 0 and
#       re-centres +/-3 points per checkpoint; errid-70's true offset is -23 points, so all 1026
#       of its checkpoints read 0 and the fit returned slope 0.0 with residual 0.0 -- a perfect
#       straight line through a measurement that never happened. (That also produced an
#       `r_squared` of 1.0, fixed in `banded_seed_alignment.fit_trace_slope`.)
#
# WHAT REPLACED IT: `zone_ladder_signature`, which COUNTS the shape a rate relation forces on
# this aligner instead of fitting a line through it. Full measured population, 11 alignments
# through this module's own code path, 2026-09-22, blind unless marked:
#
#   pair                         verdict                            zones  rungs  purity  mono   |f-1|
#   errid-27  eng   REAL NTSC    segments_found                       110     32   0.970  0.9375 1.0e-3   <- the only positive
#   errid-70  fre   REAL PAL     all_segments_below_duration_floor      0      0    --     --    0        <- other arm
#   errid-27  de-rated 1001/1000 single_segment_no_cut                  60      3   1.000  0.667  3.3e-5
#   errid-70  de-rated 1001/960  single_segment_no_cut                   8      0   0.000  --     7.8e-4
#   errid-213 jpn   edits        single_segment_no_cut                  24      8   0.889  0.500  6.9e-4   <- nearest negative
#   errid-100 jpn   ~no edit     single_segment_no_cut                  36     12   1.000  0.500  0
#   errid-232 jpn   edits        segments_found                         66      1   0.500  1.000  4.3e-3
#   errid-202 jpn   edits        segments_found                         10      1   0.250  1.000  2.2e-3
#   errid-135 jpn   edits        segments_found                         12      1   0.333  1.000  2.6e-4
#   errid-352 jpn   edits        segments_found                         26      0   0.000  --     7.0e-4
#   errid-24  jpn   no edit      single_segment_no_cut                   1      0   --     --    0
#
# CORRECTED AFTER A LARGER SAMPLE, AND THE CORRECTION MATTERS. On the eleven alignments above,
# rung count looked like a separator (32 against a largest negative of 12). On an independent
# tester's 35 real alignments it is NOT: the largest negatives read 26 rungs at purity 1.000
# (errid-24/es) and 26 at purity 0.963 (errid-99/fr), eight alignments clear LADDER_MIN_RUNGS,
# and six of those sit at purity >= 0.96. So RUNG COUNT AND PURITY DO NOT SEPARATE AT ALL.
# `rung_monotone_fraction` ALONE does -- every negative measured sits at 0.500-0.538 against the
# positive's 0.9375 -- and it does so because drift has a direction and editing does not, which
# is a mechanism rather than a coincidence of this sample. The other conditions stay as cheap
# conservative guards and as logged observations; they are not evidence and are no longer
# described as if they were.
#
# DRIVEN END TO END THROUGH `repair()` ON THE POSITIVE, so the arm is not merely calibrated on
# paper: errid-27 blind reads `gate_arm=rate_relation_signature` (rungs 32, purity 0.9697,
# direction 0.9375, implied ratio 1.0009911 against the NTSC nominal 1.000999 -- 8e-6 apart, and
# the ladder arithmetic never saw that nominal); the sweep then confirms 1001/1000 at median
# fidelity 0.9559, the comparison resample applies asetrate=383616, and the SAME couple re-aligns
# from 110 zones at coverage 0.9160 and 32 holes to 60 zones at coverage 0.9568 and THREE holes
# (head 4.34 s -- the pair's documented constant offset of +4.34269 s -- one zero-step interior,
# and a tail carrying the documented ~175 s of excess candidate content). Before this arm, that
# pair decomposed into 32 holes and was refused on the hole budget.
#
# AND THE ARM IS SAFE TO ENABLE BECAUSE ITS REFUSAL IS NON-TERMINAL (see `similarity_gate` and
# `repair()`): a false positive costs one sweep and then continues on the alignment already
# measured, producing a byte-identical plan. It cannot turn a repairable pair into a decline.
# That is what makes n=1 on the positive side an acceptable basis for switching it on -- the
# cost of being wrong is time, not a lost repair -- and it is stated here so the sample size is
# never mistaken for more than it is.
RATE_RELATION_SLOPE_GATE_CALIBRATED = True

# THE LADDER CONSTANTS, each set between two measured populations, never on one side of one.
#
# RUNGS: a STATISTICAL-SUFFICIENCY floor, NOT a detection floor, and the difference decides the
# number. This constant exists only so that `rung_fraction` and `rung_monotone_fraction` are
# computed over enough samples to mean something -- it does not separate anything (eight of 35
# real alignments clear it, including both 26-rung negatives).
#
# IT IS ALSO ONLY THE LOWER EDGE OF A BAND WITH TWO EDGES. The SPAN bound: at the smallest named
# deviation (1001/1000) and a ~124 ms quantum, 8 rungs need 8*0.124/0.000999 = 993 s of aligned
# span, so this arm cannot see an NTSC relation on anything under ~16.5 minutes. A 24-minute
# episode yields ~11.7 rungs; the 62-minute positive yielded 32. The RATE bound, undocumented
# until an independent tester measured it and the reason this comment was wrong to stop at the
# first: `banded_seed_alignment.OFFSET_MERGE_TOLERANCE_POINTS = 3` merges runs whose offsets sit
# within 3 points, so drift fast enough to accumulate 3+ quanta before a segment ends emerges as
# steps of 3-5 quanta and produces NO RUNGS AT ALL. Measured on graded synthetic fixtures,
# 0 of 10 fired: |f-1| 0.0017 gave 4 rungs, 0.0050 gave 0, and PAL (0.0427) gave 0-3 rungs
# against 29 steps above the floor. On a 300 s fixture the two bounds cross and the band is
# empty. See `zone_ladder_signature` for the full table and the raw step sequences.
LADDER_MIN_RUNGS = 8
# PURITY: NOT A SEPARATOR, and this comment used to claim it was. On the first eleven
# alignments it read 0.970 on the positive against 0.250-0.500 on the edit pairs, which looked
# like a boundary; on 35 alignments the largest negative reads purity 1.000 at 26 rungs. It is
# retained as a cheap conservative guard against a file that both drifts and is heavily edited,
# and because it has never false-refused anything -- not as evidence. It does not reject
# errid-213 (0.889) either; direction does.
LADDER_MIN_RUNG_FRACTION = 0.80
# DIRECTION: the condition that actually decides, and on the larger sample the ONLY one that
# does. Measured 0.9375 on the positive against 0.500-0.538 on every negative, including the two
# line-fit traps the bake-off and the locator independently flagged. 0.85 sits between, nearer
# the negative side than the positive's value, so the positive keeps 0.09 of margin and the
# negatives 0.31. It rests on a mechanism -- drift has a sign, editing does not -- which is what
# makes a single positive an acceptable basis for it.
LADDER_MIN_RUNG_MONOTONE_FRACTION = 0.85

# THE MAGNITUDE FLOOR, DERIVED HERE. Half the smallest deviation in the named rate vocabulary:
# 1001/1000 is the closest named rate to unity, |f - 1| = 1/1001 = 9.99e-4, and half of it is
# 5e-4 -- the smallest deviation a real rate relation can have, split so that a ladder reading
# under it cannot be a member of the vocabulary. It was `change_point_locator.RATE_SLOPE_MIN_
# FACTOR_DEVIATION`, derived there the same way and landed on a measured false fire; the locator
# was removed with the switch (2026-09-24), so the derivation lives with its only reader.
RATE_LADDER_MIN_FACTOR_DEVIATION = 5e-4

# The aligner's own "I could not measure" vocabulary -- the step-2 gate's one calibrated arm.
# These are `banded_seed_alignment`'s tokens, read from the module rather than restated here.
ALIGNMENT_COULD_NOT_MEASURE_VERDICTS = banded_seed_alignment.COULD_NOT_MEASURE_VERDICTS
# BOUND TO THE ALIGNER'S OWN TUPLE, NOT RESTATED -- and until 2026-09-22 this WAS restated, as
# four hardcoded string literals under a comment claiming the opposite. An independent tester
# found it: a renamed verdict in `banded_seed_alignment` would have silently stopped this gate
# firing, with no error anywhere, which is exactly the failure the old comment claimed to have
# avoided. The aligner now exports the vocabulary and the assignments inside it use the same
# names, so the two cannot drift.
#
# The fifth member, `all_segments_below_duration_floor`, was added with the verdict itself and is
# the arm that catches the rate family: runs anchored and were trusted and then EVERY segment
# they built came in under `MIN_SEGMENT_DURATION_S`. That is what a rate relation looks like from
# inside the aligner -- at the PAL factor the offset moves a whole quantum every ~2.9 s, so no
# fixed-offset extension survives a 2 s floor. MEASURED on errid-70 (fr), the corpus's only real
# speed pair: 2 runs extended, 1 segment built, 1 filtered, nothing left -- and before the token
# existed the aligner called that `single_segment_no_cut` and this gate read it as "similarity is
# fine, no sweep needed" on the one pair in the corpus that cannot be measured without the sweep.
#
# BELT AND BRACES, AT IMPORT: the binding above cannot go stale, but a future edit could shrink
# the aligner's tuple without anyone noticing here. The check below states, as an executable
# sentence, the two facts this gate depends on -- the tuple is non-empty, and it is disjoint from
# the aligner's MEASURED verdicts. A blank tuple would silently disable the gate; an overlap
# would silently disable a successful alignment.
assert ALIGNMENT_COULD_NOT_MEASURE_VERDICTS, (
    "repair_orchestrator: banded_seed_alignment.COULD_NOT_MEASURE_VERDICTS is empty -- the "
    "step-2 similarity gate would never fire")
assert not (set(ALIGNMENT_COULD_NOT_MEASURE_VERDICTS)
            & set(banded_seed_alignment.MEASURED_VERDICTS)), (
    "repair_orchestrator: a verdict cannot be both a measurement and a could-not-measure")

# MEASUREMENT CLASS, emitted beside every cause token -- see the module docstring's return
# contract. The boolean cannot carry this and the ledger needs it.
CLASS_CONCLUSIVE = "ran_conclusive_negative"
CLASS_COULD_NOT_RUN = "could_not_run"

# THE CAUSE VOCABULARY, CLOSED, EACH WITH ITS CLASS. The design's section 6.2 migration replaces
# fifteen dying `change_point_locator.DECLINE_REASONS` tokens with a deliberately small set; the
# ones this stage can actually reach are here, and a token absent from this table cannot be
# emitted (`_terminal` refuses it) -- the same "a token must come from a vocabulary you own"
# discipline `change_point_locator._note_site` already carries.
DECLINE_CAUSES = {
    # step 1
    "master_intertrack_desync": CLASS_CONCLUSIVE,
    # ADDENDUM 26.9, right after the prime: the master's content ends >= 300 s before the
    # candidate's and never resumes -- measured on both tracks, a fact about the master.
    "master_cut_short": CLASS_CONCLUSIVE,
    # step 2
    # RECLASSIFIED 2026-09-25 (ADDENDUM 26 report; ids 238/248/261/259/237/110): a similarity
    # FLOOR is not a proven negative -- id 110's 0.15 was a mistagged English track measured
    # against Japanese, same episode. The pass-runners read this class as Unmergeable_Verified.
    # CLASS_CONCLUSIVE stays for proofs (no common language, different content, undecodable).
    "similarity_unrecoverable_by_resample": CLASS_COULD_NOT_RUN,
    "rate_sweep_no_sample_rate": CLASS_COULD_NOT_RUN,
    # ADDENDUM 30.5: the rate arm could not measure (no comparison WAV from the prime, or the
    # resample of the candidate could not be built) -- nothing was measured about the rate.
    "rate_arm_unmeasured": CLASS_COULD_NOT_RUN,
    # step 3, measurement
    "no_stream_for_comparison_language": CLASS_COULD_NOT_RUN,
    "track_duration_unmeasurable": CLASS_COULD_NOT_RUN,
    "fingerprinting_raised": CLASS_COULD_NOT_RUN,
    "alignment_degenerate_input": CLASS_COULD_NOT_RUN,
    "alignment_no_anchored_runs": CLASS_COULD_NOT_RUN,
    "alignment_all_seeds_refused": CLASS_COULD_NOT_RUN,
    "alignment_segments_below_duration_floor": CLASS_COULD_NOT_RUN,
    "intercouple_disagreement": CLASS_CONCLUSIVE,
    # RECLASSIFIED 2026-09-22 (see MAX_HOLES_PER_COUPLE): "more holes than my resolver budget"
    # is a fact about this orchestrator's budget, not a measurement about the pair.
    "hole_count_exceeds_resolver_budget": CLASS_COULD_NOT_RUN,
    # THE COVERAGE ARM'S OWN TOKEN, for the case where every couple falls under the floor. It is
    # could-not-run and not conclusive: the aligner returned a token, but under the floor that
    # token is not a reading about the pair -- see MASTER_AXIS_COVERAGE_FLOOR.
    "alignment_coverage_below_floor": CLASS_COULD_NOT_RUN,
    # THE RE-PRIME AT A CONFIRMED FACTOR (ADDENDUM 21.6). `comparison_resample_refused_at_unity`
    # and `comparison_resample_filter_unbuildable` lived here while chimeric resampled its own
    # comparison extraction; chimeric no longer resamples, the step they named is gone, and by
    # this table's own rule so are they. What remains is the one real refusal of the upstream
    # re-prime: a factor was confirmed but the speed-corrected candidate could not be built (the
    # exact-rational filter refused the ratio). Could-not-run -- nothing about the pair was
    # measured. (An unreadable source rate keeps `rate_sweep_no_sample_rate`, its own fact.)
    "rate_resample_unbuildable": CLASS_COULD_NOT_RUN,
    # step 4. `hole_resolution_not_implemented` was here until the stage landed and is GONE, by
    # this table's own rule (see step 3e above). Its replacement is the resolver's REAL refusal:
    # the frame-exact search could not establish a boundary on at least one hole (anchors not
    # established, geometry unreconciled, an unreadable walk, no measurable grid). Could-not-run,
    # never conclusive: a resolver that could not seed an anchor has measured nothing about the
    # pair -- the per-hole resolver reason travels in the prose and in the step log.
    "hole_resolution_declined": CLASS_COULD_NOT_RUN,
    # ADDENDUM 26: the budgets and the hole sanity bounds. A budget is a statement about THIS
    # run's cost, never about the pair (could-not-run); a hole outside the master's timeline, or
    # a step longer than either file, is a measured impossibility about the ALIGNMENT, not the
    # pair -- could-not-run too: nothing was measured about whether the pair can be repaired.
    "alignment_budget_exceeded": CLASS_COULD_NOT_RUN,
    "hole_budget_exceeded": CLASS_COULD_NOT_RUN,
    "repair_budget_exceeded": CLASS_COULD_NOT_RUN,
    "decoder_timeout": CLASS_COULD_NOT_RUN,
    "hole_outside_master_timeline": CLASS_COULD_NOT_RUN,
    "hole_step_exceeds_duration": CLASS_COULD_NOT_RUN,
    "interior_hole_exceeds_budget": CLASS_COULD_NOT_RUN,
    # step 5, plan application (landed 2026-09-24; `plan_application_not_implemented` was here
    # until then and is GONE by this table's own rule). A REFUSAL BY THE ASSEMBLY'S OWN GATES is
    # not in this table on purpose: it leaves as a `chimeric_error` carrying the token its raise
    # site set, and the entry records it -- see `apply_plan`.
    # The pair's frame grid could not be read, on a pair with NO hole (with holes the same fact
    # declines `hole_resolution_declined`, before any video is decoded).
    "frame_domain_unmeasured": CLASS_COULD_NOT_RUN,
    # The comparison track has no zone whose offset could be measured to the sample: the plan
    # has no audio offset ADDENDUM 9 point 2 lets it apply.
    "plan_offset_unmeasurable": CLASS_COULD_NOT_RUN,
    # The resolved holes cover the whole master timeline: nothing is read from the candidate.
    "plan_reads_no_candidate_content": CLASS_CONCLUSIVE,
    # The build returned and the temporary chimeric file is not on disk.
    "plan_application_no_file": CLASS_COULD_NOT_RUN,
    # ADDENDUM 25 (the audio plan). The walk could not read the comparison tracks, or found no
    # level at all:
    "audio_walk_unavailable": CLASS_COULD_NOT_RUN,
    # A change point whose edges the 20 ms and 100 ms profiles could not read:
    "audio_step_unlocalised": CLASS_COULD_NOT_RUN,
    # 25.1: the master content the candidate lacks does not fit between the audio edges -- two
    # of the walk's OWN measurements of one transition (the levels' step, the 20 ms edges)
    # disagree, before the video is asked anything: a statement about the instrument, never a
    # proof about the pair (CASE_hole_width_contradicts_audio_step_20260925: on ids 152/278/686
    # the step was exact and the 20 ms edges were wrong):
    "hole_width_contradicts_audio_step": CLASS_COULD_NOT_RUN,
    # 25.2 c: a sub-quantum step the video could neither place nor rule out:
    "sub_quantum_step_video_ambiguous": CLASS_COULD_NOT_RUN,
    # The audio's transitions or edges do not tile the timeline (a plan defect, not the pair's):
    "audio_transitions_overlap": CLASS_COULD_NOT_RUN,
    # THE ASSEMBLY'S OWN REFUSALS (B5 of the certification: a stage-5 refusal left the
    # vocabulary and the ledger could not class it). They are raised by `merge_video_chimeric` /
    # `merge_video_repair` with the token set at the raise site and recorded by the entry, which
    # now writes their measurement class through `log_measurement_class`. Measured refusals of
    # the built file are conclusive; a plan the assembly cannot lay is could-not-run.
    "alignment_contradicts_plan": CLASS_CONCLUSIVE,
    "delivery_offset_exceeds_tolerance": CLASS_COULD_NOT_RUN,  # our product, never the pair (id 126)
    "master_audio_complement_short": CLASS_CONCLUSIVE,
    "master_duration_sources_disagree": CLASS_CONCLUSIVE,
    "candidate_admission_window_exceeded": CLASS_CONCLUSIVE,
    "candidate_segment_regression": CLASS_CONCLUSIVE,
    "speed_transform_not_validated": CLASS_COULD_NOT_RUN,
    "plan_not_contiguous": CLASS_COULD_NOT_RUN,
    "plan_piece_empty_or_inverted": CLASS_COULD_NOT_RUN,
    "plan_end_not_master_timeline": CLASS_COULD_NOT_RUN,
    # A delivery probe landed where a compared track has no audio (the verifier could not
    # measure there -- not a measurement about the pair):
    "probe_reads_no_audio": CLASS_COULD_NOT_RUN,
    # RETIRED 2026-09-24: `restoration_deferred` (class `owner_choice`). The owner's ADDENDUM 8
    # lifted ADDENDUM 7's deferral -- "L'audio corrige en vitesse PEUT etre livre ... Le declin
    # `restoration_deferred` disparait de `apply_plan` et de la porte de livraison" -- so the
    # token can no longer be emitted and, by this table's own rule, is not in it.
}

# THE HOLE-RESULT VOCABULARY. `no_cut_confirmed` is FIRST CLASS (owner's ADDENDUM 3): the audio
# alignment PROPOSES a zone to examine, the video pHash DISPOSES, and a hole the walk crosses
# without dropping out is a hole that CLOSES -- not a failure. The orchestrator restores zone
# continuity across it, and if every hole closes this way the pair merges with a simple offset
# and no splice at all.
#
# THE FULL VOCABULARY, CLOSED (stage 4). Every status but `declined` carries exact frames:
#   resolved                          interior, two anchors, the cut pinned on both files
#   no_cut_confirmed                  ADDENDUM 3 -- the video crossed the hole under ONE shift
#   boundary_pinned_to_ambiguous_zone_end
#                                     ADDENDUM 4/12/13 -- a self-similar span, pinned at the
#                                     END OF THE AMBIGUOUS ZONE, cause=static_span_ambiguity
#   sustained_mismatch                EDGE ruling termination 1 -- divergent content: replace
#   master_exhausted                  EDGE ruling termination 2 -- candidate excess: trim
#   candidate_exhausted               EDGE ruling termination 3 -- master addition, COUNTED
#   declined                          the resolver could not establish a boundary; named reason
HOLE_RESOLVED = "resolved"
HOLE_NO_CUT_CONFIRMED = "no_cut_confirmed"
HOLE_PINNED_TO_AMBIGUOUS_ZONE_END = "boundary_pinned_to_ambiguous_zone_end"
EDGE_SUSTAINED_MISMATCH = "sustained_mismatch"
EDGE_MASTER_EXHAUSTED = "master_exhausted"
EDGE_CANDIDATE_EXHAUSTED = "candidate_exhausted"
HOLE_DECLINED = "declined"
EDGE_TERMINATIONS = (EDGE_SUSTAINED_MISMATCH, EDGE_MASTER_EXHAUSTED, EDGE_CANDIDATE_EXHAUSTED)
HOLE_STATUSES_WITH_FRAMES = (HOLE_RESOLVED, HOLE_NO_CUT_CONFIRMED, HOLE_PINNED_TO_AMBIGUOUS_ZONE_END
                             ) + EDGE_TERMINATIONS

# The three `why=` tokens are a CLOSED SET PINNED BY A TOOL, not a naming choice:
# `tools/validate_merge_plan.py:154-156` matches `(head_gap|interior_bracket|tail_gap|absent\()`
# and `merge_video_repair.log_assembly` emits it. The orchestrator's hole classification
# produces exactly head / interior / tail, so it reuses those three verbatim; inventing a
# fourth (`merged_hole`) would silently fall out of that tool's world.
WHY_TOKEN = {"head": "head_gap", "interior": "interior_bracket", "tail": "tail_gap"}


# ---------------------------------------------------------------------------
# LOGGING -- one launch line and one result line per step, per the ruling
# ---------------------------------------------------------------------------

def _fields(pairs):
    return " ".join(f"{key}={value}" for key, value in pairs)


def step_launch(step, **fields):
    """The ruling: "Chaque etape : un log de lancement et un log de resultat, gated tools.dev".

    The launch half is emitted BEFORE the work starts and names the file it is about, because a
    step that hangs leaves only its launch line behind -- that is the entire reason the owner
    ordered these, on a real seven-hour hang whose last log line named no file.
    """
    _STEP_STARTED[step] = time.monotonic()
    tools.dev_log(f"orchestrator: launch step={step} "
                  f"{_fields(sorted(fields.items()))}\n")


# When each launched step began (ADDENDUM 26.5): its result line carries `elapsed_s`.
_STEP_STARTED = {}


def step_result(step, **fields):
    """The result half. Emitted on EVERY exit of the step, including its refusals -- a step that
    logs only its successes is a step whose failures are invisible."""
    started = _STEP_STARTED.pop(step, None)
    if started is not None:
        fields = dict(fields, elapsed_s=round(time.monotonic() - started, 2))
    tools.dev_log(f"orchestrator: result step={step} "
                  f"{_fields(sorted(fields.items()))}\n")


def _plan_line(kind, candidate_path, **fields):
    """`repair: plan <kind> ...` -- A HARD REQUIREMENT ON THIS CHAIN, NOT A STYLE CHOICE.

    `merge_plan_report.is_job_log` (`:908`) recognises a job log by exactly one test:

        any(line.startswith("repair: plan ") for line in text.splitlines())

    So if the orchestrator stops emitting this line, every downstream reader stops recognising
    the log as a job log AT ALL -- not "reads it with fewer fields", stops seeing it. The line is
    therefore emitted on every terminal path of `repair()`, success or refusal, with
    `kind=none` when no plan was built. Unconditional (`tools.logs`).

    EXACTLY ONE PER RUN (B4, owner 2026-09-25, Yozakura-san S02E09): the build's own
    `merge_video_repair.log_assembly` writes the plan line WITH ITS GEOMETRY (pieces, quantum,
    language) whenever an assembly exists -- on success, and on a build refused with a partial
    assembly. A second, geometry-less line written after it was the one `merge_plan_report`
    kept, and every successful report lost its schematic. So this line is written only when no
    `repair: plan ` line exists yet for this candidate since its `repair()` began
    (`_PLAN_LINE_MARK`); a refusal before any build still gets it, since `is_job_log` keys on it.
    """
    mark = _PLAN_LINE_MARK.get(candidate_path)
    if mark is not None and any(line.startswith("repair: plan ")
                                for entry in tools.logs[mark:]
                                for line in str(entry).splitlines()):
        tools.dev_log(f"orchestrator: plan line kind={kind} not repeated for {candidate_path} "
                      f"-- the build's plan line (with its geometry) is this run's one line\n")
        return
    tools.log_line(f"repair: plan {kind} build={repair_log.build_sha()} orchestrator=1 "
                      f"{_fields(sorted(fields.items()))} for {candidate_path}\n")


# Where each candidate's run began in `tools.logs` (set by `repair()`, per candidate path, reset
# at every call): the window `_plan_line` searches for a plan line already written.
_PLAN_LINE_MARK = {}


def log_measurement_class(candidate_path, cause):
    """The measurement-class line beside a cause token, unconditional -- for the orchestrator's
    own terminals (`_terminal`) and for the assembly's refusals the entry records (B5). A token
    outside `DECLINE_CAUSES` is named loudly and classed `unclassified`, never borrowed."""
    if cause not in DECLINE_CAUSES:
        tools.log_always(f"repair: orchestrator UNVOCABULARISED cause={cause} for "
                         f"{candidate_path} -- this token is not in DECLINE_CAUSES and has no "
                         f"measurement class; add it there. The refusal below stands.\n")
    tools.log_always(f"repair: orchestrator cause={cause} "
                     f"measurement={DECLINE_CAUSES.get(cause, 'unclassified')} "
                     f"for {candidate_path}\n")


def _terminal(candidate_path, outcome, cause, reason, detail=None):
    """The one terminal per-candidate verdict line, and the ONLY place a cause token leaves this
    module.

    Routed through `merge_video_repair.record` when that module is importable, so the ledger
    keeps reading ONE line shape (`repair: <outcome> cause=<token> for <path>: <reason>`,
    whose fixed-vocabulary PREFIX is what makes the token unforgeable by a filename -- see that
    function's own comment) rather than a second shape this module invented. A late, tolerant
    import: the orchestrator must not become unusable standalone, and it must not create an
    import cycle with the module that will eventually call it.

    THE MEASUREMENT CLASS TRAVELS WITH THE TOKEN. The owner's boolean return cannot carry
    "could not measure" and the standing invariant forbids folding it into "measured and
    refused", so the class is emitted here, beside the cause, on its own gate-free line.
    """
    # A TOKEN FROM OUTSIDE THE VOCABULARY IS A BUG IN THE CALLER, AND IT MUST BE LOUD -- but
    # LOUD IS NOT THE SAME AS FATAL. Raising here would convert an honest refusal into a crash,
    # which is the shape this campaign's own rule refuses in the other direction ("une levee
    # dans un controle n'est pas un verdict"): a token this module has not classified is a gap
    # in THIS table, and the candidate is still refused either way. So the refusal stands, the
    # gap is named on the unconditional channel where nobody can miss it, and the measurement
    # class reads `unclassified` rather than borrowing one it was not given.
    log_measurement_class(candidate_path, cause)
    try:
        import merge_video_repair
    except Exception as error:                                           # noqa: BLE001
        tools.log_always(f"repair: {outcome} cause={cause} for {candidate_path}: {reason}\n")
        tools.dev_log(f"orchestrator: merge_video_repair unimportable for the terminal "
                      f"record ({type(error).__name__}) -- the line above was written "
                      f"directly\n")
    else:
        merge_video_repair.record(candidate_path, outcome, reason, detail=detail, cause=cause)
    return False


# ---------------------------------------------------------------------------
# COUPLES AND FINGERPRINTS -- step 3, sub-step 1
# ---------------------------------------------------------------------------

def _track_duration_seconds(video_obj, language, stream_order):
    """THIS TRACK'S OWN duration, not the language's first track's and not the container's.

    `audio_extract.audio_duration_seconds` answers for `audios[language][0]`, which is the right
    answer to a different question -- under the orchestrator each track is fingerprinted to ITS
    OWN length (see `fingerprint_track`), so each one needs its own duration. Falls back to the
    container's format duration via ffprobe when the video object carries no per-track reading.

    Returns None when nothing could be read. None IS "I could not measure", never zero.
    """
    audios = getattr(video_obj, "audios", None) or {}
    for entry in audios.get(language) or []:
        if entry.get("StreamOrder") != stream_order:
            continue
        for key in ("Duration", "duration"):
            if key in entry:
                try:
                    return float(entry[key])
                except (TypeError, ValueError):
                    pass
    # BOUNDED, unlike most of the subprocess calls on the chain this replaces. An ffprobe that
    # hangs here would hang the whole repair with nothing in the log after the launch line, which
    # is the failure this campaign spent a night tracing; the timeout makes that impossible for
    # this call at least.
    tools.dev_log(f"orchestrator: ffprobe duration call file={video_obj.filePath} "
                  f"stream_order={stream_order}\n")
    try:
        with repair_log.announced("orchestrator", "ffprobe", video_obj.filePath) as call:
            completed = subprocess.run(
                [tools.software["ffprobe"], "-v", "error", "-show_entries",
                 "format=duration", "-of", "default=nw=1:nk=1", video_obj.filePath],
                capture_output=True, text=True, timeout=120)
            call["exit"] = completed.returncode
        return float(completed.stdout.strip())
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"orchestrator: ffprobe duration unreadable for "
                      f"{video_obj.filePath} stream_order={stream_order}: "
                      f"{type(error).__name__}\n")
        return None


def comparison_sample_rate(master_obj, candidate_obj, language):
    """The pair's comparison grid, BY CALLING THE PIPELINE'S OWN FUNCTION.

    Reproduces `change_point_locator`'s derivation exactly and for its stated reason: the rule is
    `min(pair's lowest rate, 44100)` and a RESTATEMENT of it has already drifted from the thing
    it restates once in this campaign (the locator's extractor pinned 44100 while its author
    quoted the rule to other agents for hours). So this calls `video.get_less_sampling_rate` and
    applies the clamp in the clamp's own direction -- only ABOVE 44100 -- rather than restating
    either half.
    """
    try:
        import video as _video
        rate = int(_video.get_less_sampling_rate(master_obj.audios[language],
                                                  candidate_obj.audios[language]))
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"orchestrator: comparison grid underivable "
                      f"({type(error).__name__}); using 44100\n")
        rate = 44100
    return 44100 if rate > 44100 else rate


def enumerate_couples(master_obj, candidate_obj, language):
    """Every (master stream, candidate stream) pair of the comparison language.

    The ruling: "Les pistes se comparent une a une ; PLUSIEURS COUPLES possibles (meme langue,
    plusieurs pistes) = autant d'alignements independants qui se CROISENT." So this is the
    cartesian product, and it is not a hypothetical shape: MEASURED, errid-232 carries 3 jpn
    master tracks against 1 jpn candidate track and 2 eng against 1 (five couples on one pair),
    errid-100 carries 6.
    """
    master_streams = audio_extract.streams_for(master_obj, language)
    candidate_streams = audio_extract.streams_for(candidate_obj, language)
    return [(m, c) for m in master_streams for c in candidate_streams]


# THE FINGERPRINT QUANTUM IS FPCALC'S HOP, A CONSTANT OF CHROMAPRINT -- NOT duration/points.
# Chromaprint resamples every input to 11025 Hz and frames it at 4096 samples with an overlap of
# `4096 - 4096 / 3` in C++ INTEGER arithmetic, so consecutive points are 1365 samples apart:
# 1365 / 11025 s = 123.8095 ms. (The float 4096/3/11025 = 123.84 ms is NOT the hop; the integer
# division is.) MEASURED 2026-09-24 on fpcalc 1.6.1, the binary the pipeline calls: synthetic
# noise of 60 / 120 / 600 / 1200 s yields 463 / 948 / 4825 / 9671 points, slope 8.0772
# points/s = 123.806 ms per point (1365/11025 gives 8.0769), and a constant ~21.6-point deficit
# (the classifier/filter window at the end of the stream) that `duration / len(points)` used to
# spread over the whole file. Positions are `index * CHROMAPRINT_HOP_MS` from the start of the
# extraction, on every track, raw or rate-corrected.
CHROMAPRINT_SAMPLE_RATE = 11025
CHROMAPRINT_HOP_SAMPLES = 4096 // 3
CHROMAPRINT_HOP_MS = CHROMAPRINT_HOP_SAMPLES * 1000.0 / CHROMAPRINT_SAMPLE_RATE


def fingerprint_track(video_obj, language, stream_order, side, work_dir, sample_rate,
                      duration_seconds, audio_filter=None, output_duration_seconds=None,
                      measures=None, keep_wav=False):
    """One whole-file fingerprint list for ONE track. Returns `(points, quantum_ms)`, or
    `(None, None)` when the track could not be read.

    ON `audio_filter` / `output_duration_seconds` -- THE COMPARISON RESAMPLE, AND WHY THE SECOND
    ARGUMENT IS NOT OPTIONAL ONCE THE FIRST IS GIVEN. `duration_seconds` bounds what is READ from
    the source (it lands on ffmpeg's `-t`, before `-i`); a speed filter changes what is WRITTEN.
    So a corrected extraction is `duration_seconds * effective_ratio` long, and the fpcalc
    `-length` must be THAT, not the input length -- the input length would tell fpcalc to stop
    early and truncate exactly the tail the correction just restored. The quantum does not depend
    on it: see `CHROMAPRINT_HOP_MS` -- the points of a corrected extraction sit one hop apart on
    the corrected (master-equivalent) timeline, exactly as those of a raw one do on its own.
    `output_duration_seconds` defaults to `duration_seconds`, which is exactly right when there
    is no filter and never right when there is one.

    EXTRACTED TO ITS OWN FULL DURATION, AND THIS IS THE WHOLE POINT OF THE STEP.
    `banded_seed_alignment.locate_zones_by_alignment` truncates BOTH sides to
    `min(master_duration, candidate_duration)`, which is correct for the standalone diagnostic
    it was written as and WRONG here: truncating to the shorter side makes a TAIL GAP invisible
    by construction -- the candidate's extra content is simply not extracted, so no alignment
    can find it. That is the same structural blindness already documented against
    `zone_similarity_vector` ("THIS METHOD CANNOT SEE A TAIL GAP BY CONSTRUCTION", `:29-45`),
    and the orchestrator's head/tail hole resolution is precisely the case it would be blind to.
    The owner's own example is 173 s of tail on one real pair.

    Unequal point counts are fine for the aligner: `b2_align` is index-based and
    `extend_seed` bounds itself on both `len(fp_master)` and `len(fp_candidate)` independently.
    The quantum is still returned PER TRACK and carried separately (the standing per-track-quantum
    invariant): it used to be `duration / len(points)`, which differed per track (124.0636 vs
    124.0418 ms on errid-232) only because fpcalc drops a fixed ~21 trailing points, so the
    division spread that loss over the file -- 124.05 ms against the real 123.81 ms hop, a
    position error that grows to ~2 s by the end of a 20-minute file. It is now the hop itself.

    `length` IS PASSED TO fpcalc EXPLICITLY. `audioCorrelation.calculate_fingerprints` defaults
    to `length=1`, i.e. ONE SECOND of fingerprints, silently -- a caller that forgets this gets a
    plausible short list rather than an error.
    """
    if output_duration_seconds is None:
        output_duration_seconds = duration_seconds
    wav = path.join(work_dir, f"orch_{side}_{stream_order}.wav")
    try:
        audio_extract.extract_audio_window(video_obj.filePath, stream_order, 0.0,
                                           duration_seconds, wav, sample_rate,
                                           audio_filter=audio_filter)
        with repair_log.announced("orchestrator", "fpcalc", wav) as call:
            points = audioCorrelation.calculate_fingerprints(wav, length=output_duration_seconds)
            call["exit"] = 0
        if measures is not None:
            # ADDENDUM 26.9: where this track's CONTENT ends, read on the WAV already extracted
            # for the fingerprint -- no second decode.
            measures["content_end_s"] = wav_content_end_s(wav)
    except tools.decoder_timeout:
        raise
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"orchestrator: fingerprint_track raised on "
                      f"{video_obj.filePath} stream_order={stream_order}: "
                      f"{type(error).__name__}: {error}\n")
        return None, None
    finally:
        # `keep_wav` (ADDENDUM 30.5): the rate arm resamples THIS extraction instead of decoding
        # the file again; it owns the file from here and deletes it (`rate_arm`).
        if keep_wav and measures is not None and path.exists(wav):
            measures["wav"] = wav
            wav = None
        # THE CLEANUP IS ONLY CORRECT BECAUSE THE PATH IS UNIQUE. `work_dir` is per-pair and the
        # name carries the side and the stream order, so two tracks of one pair cannot overwrite
        # each other's WAV nor delete each other's file. When hole resolution is parallelised
        # (design stage 7) this naming rule has to be extended, not assumed -- the design names
        # the collision as hazard 1.
        try:
            if wav is not None:
                remove(wav)
        except OSError:
            pass
    if not points:
        return None, None
    return points, CHROMAPRINT_HOP_MS


# ---------------------------------------------------------------------------
# THE TAIL THAT ONLY ONE SIDE CARRIES -- ADDENDUM 26.9 (owner, 2026-09-25)
# ---------------------------------------------------------------------------
# After the last instant of COMMON content (the end of the couple's last b2 zone), the content
# may continue on ONE side only, uninterrupted to its end (trailing digital silence excluded).
# At least TAIL_ONE_SIDED_MIN_S of it and:
#   the CANDIDATE is short  -> not an error: the master fills the tail (a tail fill carries no
#                              size cap -- the interior-hole bound does not apply), logged
#                              `repair: candidate_short_tail`;
#   the MASTER is short     -> `master_cut_short`, both measured end instants in the prose.
# A content comparison, no duration ratio. A long gap followed by common content again is an
# interior hole (the zones say so), never this rule. The master's content is read to the end of
# its OWN track: audio past its video still counts as content reaching the timeline's end -- the
# shape of id 691 (a corrupt master: audio to 4,007 s, digital silence, content again at the very
# end of a 12,799 s track over 5,997 s of video), which this rule must not call cut short.
TAIL_ONE_SIDED_MIN_S = 300.0
CONTENT_BLOCK_S = 0.1
CONTENT_READ_CHUNK_S = 60.0
# How far back from the end of a master track that runs past its video the probe looks for
# content before calling the overrun silent (5 chunks of CONTENT_READ_CHUNK_S).
OVERRUN_PROBE_CHUNKS = 5


def _audible_block_end(samples, rate, offset_s):
    """End (s) of the last CONTENT_BLOCK_S block at or above audio_walk.AUDIBLE_DB, or None."""
    import numpy
    import audio_walk
    block = max(1, int(rate * CONTENT_BLOCK_S))
    count = len(samples) // block
    if not count:
        return None
    rms = numpy.sqrt(numpy.mean(samples[:count * block].reshape(count, block) ** 2, axis=1))
    loud = numpy.nonzero(20 * numpy.log10(numpy.maximum(rms, 1e-12)) >= audio_walk.AUDIBLE_DB)[0]
    return None if not len(loud) else offset_s + (int(loud[-1]) + 1) * block / rate


def wav_content_end_s(wav_path):
    """Where a mono 16-bit WAV's content ends, read backwards: the end of its last audible block
    (trailing digital silence and dither under the walk's audibility floor excluded). None when
    unreadable or silent throughout."""
    import numpy
    import wave
    try:
        with wave.open(wav_path, "rb") as reader:
            if reader.getsampwidth() != 2 or reader.getnchannels() != 1:
                return None
            rate, end = reader.getframerate(), reader.getnframes()
            block = max(1, int(rate * CONTENT_BLOCK_S))
            chunk = max(block, int(rate * CONTENT_READ_CHUNK_S) // block * block)
            while end > 0:
                start = max(0, end - chunk)
                reader.setpos(start)
                samples = numpy.frombuffer(reader.readframes(end - start),
                                           dtype="<i2").astype(numpy.float64) / 32768.0
                found = _audible_block_end(samples, rate, start / rate)
                if found is not None:
                    return found
                end = start
    except (OSError, EOFError, ValueError) as error:
        tools.dev_log(f"orchestrator: content_end unreadable {path.basename(wav_path)}: "
                      f"{type(error).__name__}\n")
    return None


def overrun_content_end_s(video_obj, stream_order, track_s, timeline_s):
    """For a master track longer than its video: the end of content found in its last
    OVERRUN_PROBE_CHUNKS chunks past the video (hybrid-seek reads), or None."""
    import merge_video_chimeric
    rate = 8000
    end = track_s
    for _ in range(OVERRUN_PROBE_CHUNKS):
        start = max(timeline_s, end - CONTENT_READ_CHUNK_S)
        if end - start < 1.0:
            break
        try:
            samples = merge_video_chimeric.read_mono_samples(
                video_obj.filePath, f"0:{stream_order}", Decimal(str(start * 1000.0)),
                Decimal(str((end - start) * 1000.0)), rate)
        except merge_video_chimeric.chimeric_error:
            samples = None
        found = None if samples is None else _audible_block_end(samples, rate, start)
        if found is not None:
            return found
        end = start
    return None


def tail_content_verdict(couples, timeline_s):
    """ADDENDUM 26.9's decision, pure. `couples`: one dict per couple with `couple`,
    `last_common_master_s`, `last_common_candidate_s` (the end of its last b2 zone, on each
    track's clock), `master_content_end_s`, `candidate_content_end_s`. The master's content is
    capped at `timeline_s` -- content past the video reaches the timeline's end. Returns
    `(verdict, per_couple)`: `master_cut_short`, `candidate_short` or None; every couple must
    agree, and an unmeasured couple decides nothing."""
    readings = []
    for couple in couples:
        values = [couple.get(key) for key in ("last_common_master_s", "last_common_candidate_s",
                                             "master_content_end_s", "candidate_content_end_s")]
        if any(value is None for value in values):
            readings.append((couple.get("couple"), None, None, None))
            continue
        last_m, last_c, end_m, end_c = values
        master_after = max(0.0, min(end_m, timeline_s) - last_m)
        candidate_after = max(0.0, end_c - last_c)
        verdict = ("master_cut_short" if candidate_after - master_after >= TAIL_ONE_SIDED_MIN_S
                   else "candidate_short" if master_after - candidate_after >= TAIL_ONE_SIDED_MIN_S
                   else None)
        readings.append((couple.get("couple"), verdict, round(master_after, 3),
                         round(candidate_after, 3)))
    verdicts = {reading[1] for reading in readings}
    if not readings or any(reading[2] is None for reading in readings) or len(verdicts) != 1:
        return None, readings
    return verdicts.pop(), readings


def tail_couples(primed):
    """The per-couple inputs of `tail_content_verdict`, from what the prime measured."""
    couples = []
    for master_stream, candidate_stream in primed["couples"]:
        name = f"{master_stream}x{candidate_stream}"
        alignment = primed["alignments"].get(name) or {}
        _zones, detail = coalesce_same_offset_zones(alignment.get("zones") or [],
                                                    alignment.get("zones_detail") or [])
        ends = primed.get("content_end") or {}
        couples.append({
            "couple": name,
            "last_common_master_s": (detail[-1]["master_ms"][1] / 1000.0 if detail else None),
            "last_common_candidate_s": (detail[-1]["candidate_ms"][1] / 1000.0
                                        if detail else None),
            "master_content_end_s": ends.get(("master", master_stream)),
            "candidate_content_end_s": ends.get(("candidate", candidate_stream))})
    return couples


# ---------------------------------------------------------------------------
# ZONES -> HOLES -- step 3, sub-step 4's input
# ---------------------------------------------------------------------------

def derive_holes(zones, n_master, n_candidate, quantum_ms, candidate_quantum_ms):
    """Zones -> the raw gaps between them, BEFORE classification.

    A hole is the space between two aligned zones (ruling, chimeric step 4: "un trou = l'espace
    entre deux zones alignees"), plus the two ends of the file. With `n` zones there are exactly
    `n + 1` gaps, and gap `i` is separated from gap `i+1` by exactly `zones[i]` -- the ISLAND
    between them, which `cluster_holes` keeps and logs with its b2 similarity.

    Gaps are emitted EVEN WHEN EMPTY ON BOTH AXES, at this stage, because an empty span can
    still carry a real step: two zones can be adjacent in index and still differ in offset,
    which is a cut with no slack around it. Empty-and-stepless gaps are dropped last.

    Bounds are INCLUSIVE point indices and may read `lo > hi`, which is how an empty span says
    so; the millisecond bounds are half-open on each axis's OWN quantum (see
    `banded_seed_alignment`'s zone contract for why the `+1` is there).
    """
    holes = []
    for gap_index in range(len(zones) + 1):
        before = zones[gap_index - 1] if gap_index > 0 else None
        after = zones[gap_index] if gap_index < len(zones) else None
        m_lo = (before[0][1] + 1) if before else 0
        m_hi = (after[0][0] - 1) if after else (n_master - 1)
        c_lo = (before[1][1] + 1) if before else 0
        c_hi = (after[1][0] - 1) if after else (n_candidate - 1)
        holes.append({
            "modality": MODALITY,
            "gap_index": gap_index,
            "touches_head": gap_index == 0,
            "touches_tail": gap_index == len(zones),
            "master_points": [m_lo, m_hi],
            "candidate_points": [c_lo, c_hi],
            "master_ms": [m_lo * quantum_ms, (m_hi + 1) * quantum_ms],
            "candidate_ms": [c_lo * candidate_quantum_ms,
                              (c_hi + 1) * candidate_quantum_ms],
            "master_span_seconds": max(0, m_hi - m_lo + 1) * quantum_ms / 1000.0,
            "candidate_span_seconds": (max(0, c_hi - c_lo + 1)
                                        * candidate_quantum_ms / 1000.0),
        })
    return holes


def classify_holes(holes, zones_detail, quantum_ms):
    """head / interior / tail, plus the step across each hole.

    The step is the offset difference across the hole -- `offset_after - offset_before` -- which
    only exists for an interior hole: a head hole has no zone before it and a tail hole none
    after, so their step is None, and None here means "there is no such quantity", not zero.

    A hole that touches BOTH ends spans the whole file. That is not a head hole,
    not a tail hole and not an interior one, and it does not get a `why=` token invented for it
    -- it is named `spans_whole_file` and the caller declines on it rather than guessing which
    resolver to call.
    """
    classified = []
    for hole in holes:
        entry = dict(hole)
        if hole["touches_head"] and hole["touches_tail"]:
            entry["kind"] = "spans_whole_file"
        elif hole["touches_head"]:
            entry["kind"] = "head"
        elif hole["touches_tail"]:
            entry["kind"] = "tail"
        else:
            entry["kind"] = "interior"
        entry["why_token"] = WHY_TOKEN.get(entry["kind"])
        gap_index = hole["gap_index"]
        before = zones_detail[gap_index - 1] if gap_index > 0 else None
        after_index = gap_index
        after = zones_detail[after_index] if after_index < len(zones_detail) else None
        entry["offset_before_points"] = before["offset_points"] if before else None
        entry["offset_after_points"] = after["offset_points"] if after else None
        if before is not None and after is not None:
            entry["step_points"] = after["offset_points"] - before["offset_points"]
            entry["step_ms"] = entry["step_points"] * quantum_ms
        else:
            entry["step_points"] = None
            entry["step_ms"] = None
        classified.append(entry)
    return classified


def merge_quantum_flicker(zones, zones_detail):
    """QUANTUM FLICKER (ADDENDUM 30, b2 post-rule): a zone one quantum off between two zones at
    the SAME offset is the true offset falling between two quanta (id 293: ~433 ms between 371
    and 495), not an edit -- two consecutive +/-1-quantum holes with zero net step. The middle
    zone takes its neighbours' offset (copies; the aligner's own lists are never touched), so
    the coalescing below absorbs it. Returns `(zones, detail, merged)`, `merged` one
    `(master_ms, offset_points_before, offset_points_after)` per flicker."""
    zones = [[list(zone[0]), list(zone[1])] for zone in zones]
    detail = [dict(entry) for entry in zones_detail]
    merged = []
    for index in range(1, len(detail) - 1):
        before, middle, after = (detail[index - 1]["offset_points"],
                                 detail[index]["offset_points"],
                                 detail[index + 1]["offset_points"])
        if before == after and abs(middle - before) == 1:
            merged.append((list(detail[index]["master_ms"]), middle, before))
            detail[index]["offset_points"] = before
    return zones, detail, merged


def coalesce_same_offset_zones(zones, zones_detail):
    """Adjacent zones at the SAME offset are ONE aligned region with a measurement gap in it.

    THIS IS WHAT MAKES "a hole" MEAN "a change point". Without it, every stretch the aligner
    simply failed to match becomes a hole even though the offset is identical on both sides of
    it -- there is nothing to splice across such a gap, because master and candidate stay in the
    same relation throughout. MEASURED, errid-232 jpn couple 13x1: 66 zones decompose into 40
    holes, of which THIRTY-EIGHT carry a step of exactly 0 ms. Sending those to a frame-exact
    resolver would launch 38 unbounded video searches to confirm something the audio already
    says is not a cut, and it is what pushed this couple over any sane hole budget.

    It is also the consistency the aligner already applies to ITSELF: `b2_align` builds
    `all_zones` with `if seg_a["offset_points"] == seg_b["offset_points"]: continue` -- it does
    not consider a same-offset gap a zone either. After this coalescing the orchestrator's
    interior holes are exactly `all_zones`, plus the head and the tail the gap-first view has no
    way to express. Two instruments, one answer, rather than two populations to reconcile.

    WHAT IS GIVEN UP, STATED: an edit that removes and adds the SAME duration inside one gap
    leaves the offset unchanged and is invisible here. It is equally invisible to `all_zones`,
    and to any offset instrument at all -- this coalescing loses nothing the measurement could
    have seen, and the unmatched span it absorbs is still reported through
    `master_axis_coverage_fraction` and through `unmatched_points_inside` below -- and, since
    ADDENDUM 21.9, as a LIST: every absorbed gap's own bounds on both axes are kept on the
    coalesced zone (`absorbed_gaps`), so the chimeric step can log each one with its span and
    hand the long ones to the video (`absorbed_gaps_for_couple`).
    """
    if not zones:
        return [], []
    zones, zones_detail, _flicker = merge_quantum_flicker(zones, zones_detail)
    out_zones, out_detail = [], []
    for zone, detail in zip(zones, zones_detail):
        if out_detail and detail["offset_points"] == out_detail[-1]["offset_points"]:
            previous_zone, previous_detail = out_zones[-1], out_detail[-1]
            unmatched = zone[0][0] - previous_zone[0][1] - 1
            if unmatched > 0:
                previous_detail["absorbed_gaps"].append({
                    "master_points": [previous_zone[0][1] + 1, zone[0][0] - 1],
                    "candidate_points": [previous_zone[1][1] + 1, zone[1][0] - 1],
                    "offset_points": detail["offset_points"]})
            previous_zone[0][1] = zone[0][1]
            previous_zone[1][1] = zone[1][1]
            previous_detail["master_points"][1] = detail["master_points"][1]
            previous_detail["candidate_points"][1] = detail["candidate_points"][1]
            previous_detail["master_ms"][1] = detail["master_ms"][1]
            previous_detail["candidate_ms"][1] = detail["candidate_ms"][1]
            previous_detail["n_members"] += detail["n_members"]
            previous_detail["coalesced_zones"] = previous_detail.get("coalesced_zones", 1) + 1
            previous_detail["unmatched_points_inside"] = (
                previous_detail.get("unmatched_points_inside", 0) + max(0, unmatched))
            continue
        out_zones.append([list(zone[0]), list(zone[1])])
        detail = dict(detail)
        detail["master_points"] = list(detail["master_points"])
        detail["candidate_points"] = list(detail["candidate_points"])
        detail["master_ms"] = list(detail["master_ms"])
        detail["candidate_ms"] = list(detail["candidate_ms"])
        detail["coalesced_zones"] = 1
        detail["unmatched_points_inside"] = 0
        detail["absorbed_gaps"] = []
        out_detail.append(detail)
    return out_zones, out_detail


def absorbed_gaps_for_couple(alignment):
    """Every same-offset gap `coalesce_same_offset_zones` absorbed on this couple, flattened, in
    the couple's TRACK-relative milliseconds (the caller puts them on the file's clock with the
    couple's own start delta, exactly as it does the holes). ADDENDUM 21.9: absorbed for the
    plan, never silently -- the caller logs every one with its span."""
    _zones, detail = coalesce_same_offset_zones(alignment.get("zones") or [],
                                                alignment.get("zones_detail") or [])
    quantum_ms = alignment["quantum_ms"]
    candidate_quantum_ms = alignment.get("candidate_quantum_ms") or quantum_ms
    gaps = []
    for zone in detail:
        for gap in zone["absorbed_gaps"]:
            m_lo, m_hi = gap["master_points"]
            c_lo, c_hi = gap["candidate_points"]
            gaps.append({
                "master_ms": [m_lo * quantum_ms, (m_hi + 1) * quantum_ms],
                "candidate_ms": [c_lo * candidate_quantum_ms, (c_hi + 1) * candidate_quantum_ms],
                "offset_points": gap["offset_points"],
                "offset_ms": gap["offset_points"] * quantum_ms})
    return gaps


def holes_for_couple(alignment):
    """zones -> coalesce same-offset runs -> holes -> classify -> drop the non-holes.

    Coalescing first so that a hole means a change of offset. NO <10 s MERGE here any more (owner
    2026-09-24: a scheduling merge, not a data merge): nearby holes stay separate, each with its
    own bounds and step, and `cluster_holes` groups them for one shared scene pass later.

    The final drop removes gaps that are empty on both axes AND carry no step: bookkeeping
    artefacts of "there are n+1 gaps around n zones", not places where the two timelines differ.
    """
    _z, _d, flicker = merge_quantum_flicker(alignment.get("zones") or [],
                                            alignment.get("zones_detail") or [])
    for master_ms, offset_before, offset_after in flicker:
        tools.log_always(f"repair: quantum_flicker_merged master_ms=[{round(master_ms[0], 1)}, "
                         f"{round(master_ms[1], 1)}] offset_points={offset_before}->{offset_after} "
                         f"quantum_ms={alignment.get('quantum_ms')} -- two +/-1-quantum holes, zero "
                         f"net step: one zone (ADDENDUM 30)\n")
    zones, zones_detail = coalesce_same_offset_zones(alignment.get("zones") or [],
                                                      alignment.get("zones_detail") or [])
    if not zones:
        return []
    quantum_ms = alignment["quantum_ms"]
    candidate_quantum_ms = alignment.get("candidate_quantum_ms") or quantum_ms
    raw = derive_holes(zones, alignment["n_master"], alignment["n_candidate"],
                        quantum_ms, candidate_quantum_ms)
    classified = classify_holes(raw, zones_detail, quantum_ms)
    return [hole for hole in classified
            if hole["master_span_seconds"] > 0 or hole["candidate_span_seconds"] > 0
            or hole["step_points"]]


def edge_addition_seconds(holes):
    """Total candidate content to be added at the head and the tail, in seconds.

    The b2 ESTIMATE, logged beside the holes; since ADDENDUM 25 the marker decision reads the
    edge fills the master actually WRITES (`written_edge_seconds`), never this estimate.
    """
    return sum(hole["candidate_span_seconds"] for hole in holes
               if hole["kind"] in ("head", "tail"))


def hole_on_file_clock(hole, couple, fold):
    """One couple's hole, moved from its TRACKS' clocks to the FILE's (ADDENDUM 19 b).

    `fold` is the couple's `couple_start_delta_ms` reading: `delta_ms` (candidate start x r
    minus master start), `master_start_ms`, `candidate_start_ms`, and `scale` (r, 1 at unity).
    The offsets gain `delta_ms`; the master bracket gains `master_start_ms`; the candidate
    bracket gains `candidate_start_ms * r` (its alignment milliseconds are master-equivalent).
    With every start at 0 -- the common case -- this is the identity on every number.

    The hole keeps its own step in points when both of its offsets come from this couple, and
    records where it came from (`members`), so the union can say which couple saw what."""
    delta = fold["delta_ms"]
    master_shift = fold["master_start_ms"]
    candidate_shift = fold["candidate_start_ms"] * fold["scale"]
    quantum_ms = hole["quantum_ms"]
    entry = dict(hole)
    entry["master_ms"] = [hole["master_ms"][0] + master_shift,
                          hole["master_ms"][1] + master_shift]
    entry["candidate_ms"] = [hole["candidate_ms"][0] + candidate_shift,
                             hole["candidate_ms"][1] + candidate_shift]
    entry["offset_before_ms"] = (None if hole["offset_before_points"] is None
                                 else hole["offset_before_points"] * quantum_ms + delta)
    entry["offset_after_ms"] = (None if hole["offset_after_points"] is None
                                else hole["offset_after_points"] * quantum_ms + delta)
    entry["track_delay_delta_ms"] = delta
    entry["members"] = [{"couple": couple, "kind": hole["kind"],
                         "master_ms": [round(entry["master_ms"][0], 2),
                                       round(entry["master_ms"][1], 2)],
                         "step_ms": (None if hole["step_ms"] is None
                                     else round(hole["step_ms"], 3)),
                         "gap_index": hole["gap_index"]}]
    entry["offset_sources"] = [couple, couple]
    return entry


def _bounding_offsets(entry, members):
    """A search region's offsets are the ones that BOUND it: `offset_before` from the member that
    starts first on the master axis, `offset_after` from the one that ends last (ties to the
    earlier couple). When both come from one member its step is that member's own, to the point;
    otherwise it is their difference in ms, rounded to the quantum for its point count."""
    first = min((m for m in members if m["offset_before_ms"] is not None),
                key=lambda m: m["master_ms"][0], default=None)
    last = max((m for m in members if m["offset_after_ms"] is not None),
               key=lambda m: m["master_ms"][1], default=None)
    entry["offset_before_ms"] = (None if entry["kind"] in ("head", "spans_whole_file")
                                 or first is None else first["offset_before_ms"])
    entry["offset_after_ms"] = (None if entry["kind"] in ("tail", "spans_whole_file")
                                or last is None else last["offset_after_ms"])
    entry["offset_before_points"] = (None if entry["offset_before_ms"] is None
                                     else first["offset_before_points"])
    entry["offset_after_points"] = (None if entry["offset_after_ms"] is None
                                    else last["offset_after_points"])
    entry["track_delay_delta_ms"] = (first or last or members[0])["track_delay_delta_ms"]
    entry["offset_sources"] = [None if entry["offset_before_ms"] is None
                               else first["offset_sources"][0],
                               None if entry["offset_after_ms"] is None
                               else last["offset_sources"][1]]
    if entry["offset_before_ms"] is None or entry["offset_after_ms"] is None:
        entry["step_ms"], entry["step_points"] = None, None
    elif first is last:
        entry["step_ms"], entry["step_points"] = first["step_ms"], first["step_points"]
    else:
        entry["step_ms"] = entry["offset_after_ms"] - entry["offset_before_ms"]
        entry["step_points"] = _round_half_up(
            Fraction(str(entry["step_ms"])) / Fraction(str(entry["quantum_ms"])))


def _widen(target, member):
    """`target` (a union search region) grows to cover `member`: widest bounds on both axes,
    head/tail touch, kind, spans, bounding offsets and provenance recomputed."""
    members = target["_members"] + [member]
    entry = dict(target)
    entry["_members"] = members
    entry["master_ms"] = [min(m["master_ms"][0] for m in members),
                          max(m["master_ms"][1] for m in members)]
    entry["candidate_ms"] = [min(m["candidate_ms"][0] for m in members),
                             max(m["candidate_ms"][1] for m in members)]
    entry["touches_head"] = any(m["touches_head"] for m in members)
    entry["touches_tail"] = any(m["touches_tail"] for m in members)
    entry["kind"] = ("spans_whole_file" if entry["touches_head"] and entry["touches_tail"]
                     else "head" if entry["touches_head"]
                     else "tail" if entry["touches_tail"] else "interior")
    entry["why_token"] = WHY_TOKEN.get(entry["kind"])
    entry["master_span_seconds"] = (entry["master_ms"][1] - entry["master_ms"][0]) / 1000.0
    entry["candidate_span_seconds"] = max(
        0.0, (entry["candidate_ms"][1] - entry["candidate_ms"][0]) / 1000.0)
    _bounding_offsets(entry, members)
    entry["members"] = [record for m in members for record in m["members"]]
    entry["couples"] = sorted({record["couple"] for record in entry["members"]})
    entry["union_of"] = len(members)
    return entry


def union_holes(per_couple_holes):
    """ADDENDUM 21.8 as the owner re-ruled it (2026-09-24): "The UNION ACROSS COUPLES is a union
    of SEARCH REGIONS only (min start / max end on the master axis); it never writes the plan --
    the resolved frames do". No couple pilots.

    `per_couple_holes` = every usable couple's holes on the FILE's clock (`hole_on_file_clock`),
    one list per couple, in couple order. The first couple's holes seed the union; each hole of
    each later couple joins the NEAREST union region that overlaps it or lies within the
    resolver's reach (`HOLE_MERGE_WINDOW_SECONDS`) and holds no hole of that couple yet -- the
    same event seen by another couple, whose position the cross-check measured up to 7.7 s apart
    -- or else becomes a region of its own. Two holes of ONE couple are NEVER united: they are two
    events (at most one cluster, `cluster_holes`). A region that then OVERLAPS its neighbour
    (one couple saw one long hole where another saw two separated by an island) is united with
    it -- the couple that saw no island there is evidence the island is not common -- and that
    is logged by the caller through `members`.

    The region takes the widest bounds on both axes and its bounding offsets
    (`_bounding_offsets`); per-couple provenance rides on `members` / `offset_sources` and the
    caller logs it for every region. A single couple's holes come back unchanged."""
    union = []
    for couple_holes in per_couple_holes:
        for hole in couple_holes:
            couple = hole["members"][0]["couple"]
            best, best_gap = None, None
            for index, region in enumerate(union):
                if couple in region["couples"]:
                    continue
                gap = max(hole["master_ms"][0] - region["master_ms"][1],
                          region["master_ms"][0] - hole["master_ms"][1], 0.0)
                if gap < HOLE_MERGE_WINDOW_SECONDS * 1000.0 and (best is None or gap < best_gap):
                    best, best_gap = index, gap
            if best is None:
                union.append(dict(hole, _members=[hole], couples=[couple], union_of=1))
            else:
                union[best] = _widen(union[best], hole)
    union.sort(key=lambda region: (region["master_ms"][0], region["master_ms"][1]))
    settled = []
    for region in union:
        if settled and region["master_ms"][0] < settled[-1]["master_ms"][1]:
            merged = settled[-1]
            for member in region["_members"]:
                merged = _widen(merged, member)
            settled[-1] = merged
            continue
        settled.append(region)
    for region in settled:
        del region["_members"]
    return settled


def cluster_holes(holes, per_couple_alignments):
    """THE OWNER'S SCHEDULING CLUSTER (2026-09-24, Addendum 22 pending): holes closer than the
    resolver's reach share ONE scene-detection pass, but each keeps its own bounds, its own step,
    and the ISLAND between two of them keeps its zone and its b2 similarity. The island's edges are
    the two holes' own bracket edges, which `scene_anchor` tries FIRST as anchor seeds -- so the
    island's frames are candidate anchors for both holes by construction.

    Sets, on each hole of a multi-hole cluster, `cluster_id`, `cluster_window` (the cluster's
    master bracket and offset range: `scene_anchor`'s shared extraction window) and `scan_cache`
    (one dict per cluster, the shared pass's memo). A lone hole gets none of them and is searched
    exactly as before. Returns the clusters, each with its members and its islands: the master
    span, the island's offset, and b2's similarity readings of the zone on EVERY couple that
    aligned it (`mean_match_quality`, `mean_local_baseline`), for the log."""
    clusters = []
    for index, hole in enumerate(holes):
        if clusters and (hole["master_ms"][0] - holes[clusters[-1]["members"][-1]]["master_ms"][1]
                         < HOLE_MERGE_WINDOW_SECONDS * 1000.0):
            clusters[-1]["members"].append(index)
        else:
            clusters.append({"members": [index]})
    for number, cluster in enumerate(clusters):
        members = [holes[index] for index in cluster["members"]]
        cluster["cluster_id"] = number
        cluster["islands"] = []
        for left, right in zip(members, members[1:]):
            low, high = left["master_ms"][1], right["master_ms"][0]
            offset = (left["offset_after_ms"] if left["offset_after_ms"] is not None
                      else right["offset_before_ms"])
            similarity = []
            for couple, alignment, fold in per_couple_alignments:
                for detail in alignment.get("zones_detail") or []:
                    z_low = detail["master_ms"][0] + fold["master_start_ms"]
                    z_high = detail["master_ms"][1] + fold["master_start_ms"]
                    if z_low < high and z_high > low:
                        similarity.append({
                            "couple": couple,
                            "master_ms": [round(z_low, 2), round(z_high, 2)],
                            "offset_ms": round(detail["offset_points"] * alignment["quantum_ms"]
                                               + fold["delta_ms"], 3),
                            "mean_match_quality": detail.get("mean_match_quality"),
                            "mean_local_baseline": detail.get("mean_local_baseline")})
            cluster["islands"].append({"master_ms": [low, high], "offset_ms": offset,
                                       "quantum_ms": left["quantum_ms"],
                                       "track_delay_delta_ms": left["track_delay_delta_ms"],
                                       "b2_zones": similarity})
        if len(members) > 1:
            offsets = [value for hole in members
                       for value in (hole["offset_before_ms"], hole["offset_after_ms"])
                       if value is not None]
            window = {"bracket_ms": (members[0]["master_ms"][0], members[-1]["master_ms"][1]),
                      "offsets_ms": (min(offsets), max(offsets))}
            cache = {}
            for hole in members:
                hole["cluster_id"] = number
                hole["cluster_window"] = window
                hole["scan_cache"] = cache
    return clusters


def walk_agreement(walk, low_ms, high_ms, offset_ms, tolerance_ms):
    """What the audio walk read inside master [low, high): `(n_agree, n_ok, n_windows)` --
    windows wholly inside the span, those measured (`ok`), and those whose offset lies within
    `tolerance_ms` of `offset_ms`."""
    import audio_walk
    rows = [row for row in walk["rows"]
            if row["t"] * 1000.0 >= low_ms
            and (row["t"] + audio_walk.WALK_WINDOW_S) * 1000.0 <= high_ms]
    ok = [row for row in rows if row["status"] == "ok"]
    agree = [row for row in ok if abs(row["off"] - offset_ms) <= tolerance_ms]
    return len(agree), len(ok), len(rows)


def log_absorbed_gaps(couple_results, union, walk, candidate_path):
    """ADDENDUM 21.9: the same-offset gaps are ABSORBED FOR THE PLAN, LOGGED, and above
    `ABSORBED_GAP_VIDEO_CHECK_SECONDS` confirmed -- since ADDENDUM 25 by the AUDIO WALK, not the
    video: the walk reads every window of the gap at the millisecond, and a difference of
    picture with no audio step can no longer change the plan (25.2 corollary), so only the audio
    can refute an absorption, and the walk is the audio's own instrument. Every absorbed gap of
    every couple is logged with its span and one of:

      under_threshold             logged only (see the constant's derivation)
      inside_a_hole               the union already covers that stretch
      confirmed_by_another_couple another usable couple ALIGNED most of it (more than half its
                                  master span) at the same offset, within one quantum
      confirmed_by_walk           most of the walk's measured windows inside it read the gap's
                                  offset within one quantum (the b2 offset's own precision)
      walk_change_point_inside    the walk found a level change inside it -- it is a change
                                  point of the plan like any other (resolved there, not here)
      unverified_by_walk          the walk measured too little inside it to say (silence) --
                                  absorbed, and said so

    Everything is on the file's clock, each couple with its own fold."""
    raw_zones = []
    for record in couple_results:
        alignment, fold = record["alignment"], record["fold"]
        quantum_ms = alignment["quantum_ms"]
        for detail in alignment.get("zones_detail") or []:
            raw_zones.append((record["couple"],
                              detail["master_points"][0] * quantum_ms + fold["master_start_ms"],
                              (detail["master_points"][1] + 1) * quantum_ms
                              + fold["master_start_ms"],
                              detail["offset_points"] * quantum_ms + fold["delta_ms"],
                              quantum_ms))
    changes = [(min(p["level_before"]["t_last"], p["level_after"]["t_first"]) * 1000.0,
                max(p["level_before"]["t_last"], p["level_after"]["t_first"]) * 1000.0)
               for p in walk["points"] if p["kind"] == "change_point"]
    for record in couple_results:
        fold = record["fold"]
        quantum_ms = record["alignment"]["quantum_ms"]
        for gap in absorbed_gaps_for_couple(record["alignment"]):
            low = gap["master_ms"][0] + fold["master_start_ms"]
            high = gap["master_ms"][1] + fold["master_start_ms"]
            offset = gap["offset_ms"] + fold["delta_ms"]
            span_s = (high - low) / 1000.0
            evidence = None
            if span_s < ABSORBED_GAP_VIDEO_CHECK_SECONDS:
                action = "under_threshold"
            elif any(low < hole["master_ms"][1] and high > hole["master_ms"][0]
                     for hole in union):
                action = "inside_a_hole"
            else:
                covered = sum(max(0.0, min(high, z_high) - max(low, z_low))
                              for couple, z_low, z_high, z_offset, z_quantum in raw_zones
                              if couple != record["couple"]
                              and abs(z_offset - offset) <= z_quantum)
                if 2 * covered > (high - low):
                    action = "confirmed_by_another_couple"
                elif any(c_low < high and c_high > low for c_low, c_high in changes):
                    action = "walk_change_point_inside"
                else:
                    agree, measured, windows = walk_agreement(walk, low, high, offset,
                                                              quantum_ms)
                    evidence = {"agree": agree, "ok": measured, "windows": windows}
                    action = ("confirmed_by_walk" if measured and 2 * agree > measured
                              else "unverified_by_walk")
            step_result("absorbed_gap", candidate=candidate_path, couple=record["couple"],
                        master_ms=[round(low, 2), round(high, 2)],
                        span_s=round(span_s, 3), offset_ms=round(offset, 3),
                        threshold_s=ABSORBED_GAP_VIDEO_CHECK_SECONDS, action=action,
                        walk=evidence)

def cross_verify_couples(couple_results):
    """The ruling's step 3: the couples must agree, or the pair returns with everything measured.

    EVENT-LEVEL, NOT ZONE-LEVEL, AND THE DIFFERENCE IS MEASURED. On errid-232's five real
    couples over one real ~6 s cut, the STEP MAGNITUDE agreed to 0.00-1.02 quanta while the ZONE
    POSITION of the same event disagreed by 0 to 216 quanta (0.75 s to 26.8 s), and the zone
    COUNTS did not agree at all (82/77/61/27/32 for one physical pair). So agreement is tested on
    `(cluster, signed step)` with a wide positional window, and never bound for bound.

    TWO POSITION NUMBERS APPEAR IN THIS MODULE AND THEY MEASURE DIFFERENT THINGS -- reconciled
    here because an independent tester read them as one and could not reproduce the larger. The
    26.8 s above is a RAW ZONE-BOUND disagreement, measured on `b2_align`'s zone bounds before
    anything downstream touched them; it is the number that justifies not comparing bound for
    bound. What this function actually clusters is the HOLE START position, after
    `coalesce_same_offset_zones` has joined same-offset runs (the <10 s merge used to follow) -- a
    quantity two transforms removed from the first, and much better behaved. MEASURED on it,
    corpus-wide: the largest within-cluster positional spread is 7.69 s (errid-24/es), then 7.57,
    6.95, 6.58, 5.96, 5.81, and 4.22 for errid-232's real edit; everything else is ~0. All inside
    the 10 s window, with 2.3 s of headroom. So the window is not 2.7x too small -- the two
    numbers are simply not the same measurement, and neither reading was wrong.

    The rules, each one a refusal this design would otherwise make wrongly:

    1. THE FIRST AND LAST EVENTS ARE IGNORED (the ruling: the audios may start or end earlier) --
       implemented as "a head or tail hole is not an event", since that is what first and last
       mean once holes are classified.
    2. A COUPLE THAT EMITTED NO EVENT IN A CLUSTER IS `could-not-see`, NOT A DISAGREEMENT. This
       is the could-not-measure-read-as-measured-negative defect this campaign has found eight
       times; MEASURED here, errid-232's `jpn#2` found 1 zone where its siblings found 3-5 and
       would trigger it.
    3. A BELOW-FLOOR SINGLETON IS NOT A DISAGREEMENT. An event present in exactly one couple
       whose magnitude sits under the aligner's own resolution floor
       (`RESOLUTION_FLOOR_QUANTA * quantum`) is logged and excluded. MEASURED: on errid-100, a
       pair with essentially nothing to repair (0.064 s duration difference), 5 of 6 couples read
       `single_segment_no_cut` and one read a lone 248 ms = 2-quantum step -- a literal reading
       of "the rest must be strongly identical" would DECLINE that pair for disagreement when
       there is nothing to repair. That is a new false-decline family and this rule closes it.
    4. DISAGREEMENT IS TWO OR MORE COUPLES IN ONE CLUSTER WHOSE STEPS DIFFER BY MORE THAN THE
       TOLERANCE. Anything less is not a disagreement about the world.
    5. POSITION NEVER DECLINES (owner ruling 2026-09-24, Addendum 22 pending): couples that place
       the same event at different master positions only WIDEN the union's search region
       (`union_holes`); only a MAGNITUDE disagreement declines, with everything logged -- it is
       not routed to the video unless the owner rules otherwise.

    Returns a dict with `agree` (bool), the clusters, and the per-cluster records. The CALLER
    emits the disagreement log, and the ruling says that one comes out even at `tools.dev=false`.
    """
    events = []
    for record in couple_results:
        alignment = record["alignment"]
        quantum_ms = alignment["quantum_ms"]
        floor_ms = banded_seed_alignment.RESOLUTION_FLOOR_QUANTA * quantum_ms
        for hole in record["holes"]:
            # AN EVENT IS A HOLE WITH A STEP. A hole whose offsets are the SAME on both sides
            # carries no offset difference at all -- it is a stretch the aligner lost the thread
            # on, the `no_cut_confirmed` candidate class of ADDENDUM 3 where the video is
            # expected to close the hole, and it is NOT a proposed cut. MEASURED, errid-232 jpn,
            # with zero-step holes admitted as events: couple 15x1's real +6078 ms step at
            # master 307.8 s clustered with couple 13x1's ZERO-step continuity gap at 312.9 s,
            # spread 6078 ms against a 186 ms tolerance, and the pair DECLINED
            # `intercouple_disagreement` -- a false decline manufactured by comparing an event
            # against a non-event. A couple that has no step at a position has SEEN NO CUT
            # there, which rule 2 below already has a name for (`could-not-see`), and that is
            # where this belongs.
            if hole["kind"] != "interior" or not hole["step_points"]:
                continue
            events.append({
                "couple": record["couple"],
                "master_position_seconds": hole["master_ms"][0] / 1000.0,
                "step_points": hole["step_points"],
                "step_ms": hole["step_ms"],
                "quantum_ms": quantum_ms,
                "resolution_floor_ms": floor_ms,
                "residual_fraction": alignment.get("residual_fraction"),
                "master_axis_coverage_fraction": alignment.get(
                    "master_axis_coverage_fraction"),
            })

    events.sort(key=lambda e: e["master_position_seconds"])
    clusters = []
    for event in events:
        if clusters and (event["master_position_seconds"] - clusters[-1]["start_seconds"]
                          <= INTERCOUPLE_POSITION_WINDOW_SECONDS):
            clusters[-1]["events"].append(event)
        else:
            clusters.append({"start_seconds": event["master_position_seconds"],
                              "events": [event]})

    all_couples = [record["couple"] for record in couple_results]
    verdicts = []
    disagreements = []
    for index, cluster in enumerate(clusters):
        # ONE READING PER COUPLE PER CLUSTER: its events inside the cluster SUMMED into its net
        # step. Since the owner's scheduling ruling (2026-09-24) a couple's nearby holes are no
        # longer fused, so one couple can bring two events to one cluster (a cut and its
        # re-add, seconds apart); comparing them WITH EACH OTHER would manufacture a
        # disagreement. The net step is exactly what that couple's fused hole carried before.
        by_couple = {}
        for event in cluster["events"]:
            net = by_couple.setdefault(event["couple"], dict(event, step_ms=0.0, step_points=0,
                                                             n_events=0))
            net["step_ms"] += event["step_ms"]
            net["step_points"] += event["step_points"]
            net["n_events"] += 1
        members = list(by_couple.values())
        seen = [event["couple"] for event in members]
        # RULE 2: absence is could-not-see, recorded by name so it can never be read as dissent.
        could_not_see = [couple for couple in all_couples if couple not in seen]
        tolerance_ms = (INTERCOUPLE_STEP_TOLERANCE_QUANTA * INTERCOUPLE_STEP_TOLERANCE_SLACK
                        * max(event["quantum_ms"] for event in members))
        # RULE 3: BELOW-FLOOR EVENTS ARE EXCLUDED FROM THE AGREEMENT TEST -- ALL OF THEM, not
        # only the lone ones. A step under the aligner's OWN declared resolution floor is not a
        # claim about a cut: `b2_align` itself keeps such zones out of `cut_zones`, the CLAIMED
        # cut list, and its own acceptance record says every spurious zone it produced on
        # constructed fixtures was exactly +/-1 quantum -- which is why that floor is 2 quanta.
        # Two measurements that neither instrument is willing to claim cannot contradict each
        # other, and treating them as if they could MANUFACTURES disagreements: MEASURED,
        # errid-232 jpn, couple 13x1 read +124.034 ms at master 377.1 s and couple 14x1 read
        # -124.034 ms at 379.0 s -- both exactly one quantum, both under the 248.068 ms floor,
        # both sub-claim noise -- and the pair DECLINED `intercouple_disagreement` on a spread
        # of 248 ms against a 186 ms tolerance. The design's own rule excludes the SINGLETON
        # case (errid-100's lone 2-quantum outlier); this is that rule applied to what it is
        # actually about, which is the floor and not the count.
        #
        # EXCLUDED, NEVER DISCARDED: every below-floor member stays in the record with its own
        # numbers, so the exclusion is auditable and a reader can see what was set aside.
        above_floor = [event for event in members
                       if abs(event["step_ms"]) >= event["resolution_floor_ms"]]
        below_floor = [event for event in members if event not in above_floor]
        steps = [event["step_ms"] for event in (above_floor or members)]
        spread_ms = max(steps) - min(steps)
        if not above_floor:
            verdict = "below_floor_only_excluded"
        elif len(above_floor) == 1:
            # RULE 2 again, from the other side: one couple claims, the others did not see it.
            # A claim nobody contradicts is not an agreement and it is not a disagreement.
            verdict = "single_couple_event"
        elif spread_ms <= tolerance_ms:
            verdict = "agree"
        else:
            verdict = "disagree"
        entry = {
            "cluster_index": index,
            "master_position_seconds": round(cluster["start_seconds"], 3),
            "verdict": verdict,
            "spread_ms": round(spread_ms, 3),
            "tolerance_ms": round(tolerance_ms, 3),
            "members": members,
            "n_above_floor": len(above_floor),
            "below_floor_excluded": [(event["couple"], round(event["step_ms"], 3))
                                      for event in below_floor],
            "could_not_see": could_not_see,
        }
        verdicts.append(entry)
        if verdict == "disagree":
            disagreements.append(entry)
    return {
        "modality": MODALITY,
        "agree": not disagreements,
        "n_couples": len(couple_results),
        "n_events": len(events),
        "clusters": verdicts,
        "disagreements": disagreements,
    }


def log_cross_verification(candidate_path, report):
    """The cross-verification record, per cluster, every couple's stream pair, master position,
    step in both points and ms, quantum, residual fraction and coverage -- "TOUTES LES INFOS
    EXTRAITES", not a summary of them.

    ALL OF IT IS A DEV LOG SINCE ADDENDUM 21.7 ("Logs de desaccord inter-couples en dev ;
    record() porte la cause"), which revises the ruling's original "sortent meme a
    tools.dev=false": the disagreement no longer needs a second unconditional channel, because
    the terminal `record()` of the refusal carries `cause=intercouple_disagreement` AND the
    whole report as its detail (`repair()` passes it), which is what the ledger reads.
    """
    if report["agree"]:
        step_result("cross_verify", candidate=candidate_path, agree=True,
                    n_couples=report["n_couples"], n_events=report["n_events"],
                    clusters=len(report["clusters"]))
        for cluster in report["clusters"]:
            tools.dev_log(
                f"orchestrator: cross_verify cluster={cluster['cluster_index']} "
                f"verdict={cluster['verdict']} "
                f"master_position_s={cluster['master_position_seconds']} "
                f"spread_ms={cluster['spread_ms']} tolerance_ms={cluster['tolerance_ms']} "
                f"could_not_see={cluster['could_not_see']} "
                f"below_floor_excluded={cluster['below_floor_excluded']} "
                f"members={[(e['couple'], round(e['step_ms'], 1)) for e in cluster['members']]}"
                f"\n")
        return
    for cluster in report["disagreements"]:
        tools.dev_log(
            f"orchestrator: intercouple_disagreement for {candidate_path} "
            f"cluster={cluster['cluster_index']} "
            f"master_position_s={cluster['master_position_seconds']} "
            f"spread_ms={cluster['spread_ms']} tolerance_ms={cluster['tolerance_ms']} "
            f"n_above_floor={cluster['n_above_floor']} "
            f"below_floor_excluded={cluster['below_floor_excluded']} "
            f"could_not_see={cluster['could_not_see']}\n")
        for event in cluster["members"]:
            tools.dev_log(
                f"orchestrator: intercouple_disagreement_member "
                f"for {candidate_path} cluster={cluster['cluster_index']} "
                f"couple={event['couple']} "
                f"master_position_s={round(event['master_position_seconds'], 3)} "
                f"step_points={event['step_points']} "
                f"step_ms={round(event['step_ms'], 3)} "
                f"quantum_ms={round(event['quantum_ms'], 4)} "
                f"resolution_floor_ms={round(event['resolution_floor_ms'], 3)} "
                f"residual_fraction={event['residual_fraction']} "
                f"coverage={event['master_axis_coverage_fraction']}\n")


# ---------------------------------------------------------------------------
# THE RATE RE-PRIME'S TOOLS -- the pitch layer and the speed-corrected candidate (upstream of
# chimeric, ADDENDUM 21.6)
# ---------------------------------------------------------------------------

def _pitch_probe_window(master_obj, candidate_obj, language, master_stream, candidate_stream,
                        speed_factor, work_dir, sample_rate):
    """Two single-stream WAVs of the COMPARISON language, content-aligned, for the pitch layer.

    WHY THE TRACKS ARE PRE-EXTRACTED INSTEAD OF HANDING THE CONTAINERS STRAIGHT TO
    `pal_pitch_confirmer`. That module's `_pcm` runs ffmpeg with no `-map`, so it takes ffmpeg's
    DEFAULT audio-stream pick -- the "best" stream, which is the one with the most channels.
    MEASURED on the real PAL pair this stage was validated against: the master carries fre AC-3
    2ch and eng DTS 6ch, the candidate carries fre and eng both E-AC-3 2ch, so the default pick
    reads the master's ENGLISH and the candidate's FRENCH and the pitch layer would be handed two
    different languages. The limitation is documented in that module's siblings
    (`merge_video_resample.test_speed_ratio_against_master`: "a caller with a specific comparison
    stream must pre-extract it to a single-stream file first"), so this is that pre-extraction,
    not a workaround invented here.

    AND THE TWO WINDOWS ARE TAKEN AT CORRESPONDING CONTENT, NOT AT THE SAME CLOCK TIME. With
    `speed_ratio = master_duration / candidate_duration`, master instant `t` is candidate instant
    `t / ratio` -- the same arithmetic `merge_video_resample._probe_fidelity_at_ratio` states for
    its own probes. At the PAL factor a 180 s window taken at the same clock time on both sides
    would compare content 25 s apart at its start; corrected, the residual drift inside the window
    is the window's own length times the relation, which is content the long-term spectrum
    tolerates. The candidate side is NOT filtered: the whole question the pitch layer answers is
    what this audio's pitch is BEFORE anything undoes it.

    Returns `(master_wav, candidate_wav, window_seconds)` or `(None, None, None)`.
    """
    master_duration = _track_duration_seconds(master_obj, language, master_stream)
    candidate_duration = _track_duration_seconds(candidate_obj, language, candidate_stream)
    if master_duration is None or candidate_duration is None:
        return None, None, None
    ratio = float(speed_factor)
    window = PITCH_PROBE_WINDOW_SECONDS
    # STAY INSIDE BOTH FILES, on both axes: the candidate must be able to supply the window at
    # `t / ratio`, so the usable master span is bounded by the candidate's own length times the
    # ratio -- never by the master alone.
    usable = min(master_duration, candidate_duration * ratio)
    if usable <= window:
        window = max(PITCH_PROBE_WINDOW_MINIMUM_SECONDS, usable / 2.0)
        if usable <= window:
            return None, None, None
    master_start = max(0.0, (usable - window) / 2.0)
    candidate_start = master_start / ratio
    master_wav = path.join(work_dir, f"orch_pitch_master_{master_stream}.wav")
    candidate_wav = path.join(work_dir, f"orch_pitch_candidate_{candidate_stream}.wav")
    try:
        audio_extract.extract_audio_window(master_obj.filePath, master_stream, master_start,
                                           window, master_wav, sample_rate)
        audio_extract.extract_audio_window(candidate_obj.filePath, candidate_stream,
                                           candidate_start, window / ratio, candidate_wav,
                                           sample_rate)
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"orchestrator: pitch probe windows unextractable "
                      f"({type(error).__name__}: {error}) -- the pitch layer will not be asked, "
                      f"and not asking is recorded as not asking\n")
        for temporary in (master_wav, candidate_wav):
            try:
                remove(temporary)
            except OSError:
                pass
        return None, None, None
    return master_wav, candidate_wav, window


def pitch_routing(speed_factor, master_obj, candidate_obj, language, work_dir, sample_rate):
    """WHICH FILTER, DECIDED BY THE MEASUREMENT ALREADY AVAILABLE -- never hard-coded.

    The ruling's body, verbatim: "le filtre se CHOISIT par la mesure deja disponible -- la couche
    pitch dit si la hauteur a bouge : pitch decale (speedup PAL/NTSC) -> asetrate+aresample
    (defait vitesse ET hauteur ensemble) ; pitch intact -> atempo (tempo seul, hauteur
    preservee). Jamais un choix code en dur."

    WHAT THE MEASUREMENT CAN AND CANNOT SAY TODAY, AND THE DIFFERENCE IS THE WHOLE DECISION.
    `pal_pitch_confirmer.confirm_pitch` answers ONE question: does the pitch-measured ratio agree
    with the ratio we are about to apply? Two of its three outcomes are unambiguous:

      agrees                 the pitch moved WITH the speed -- the naive-speedup family, PAL and
                             NTSC both. `asetrate` is the exact inverse and undoes both together.
      refuses, no peak       the instrument did not run on this material. NOT a verdict about the
                             pitch, and the standing invariant forbids reading it as one.

    The third -- "a bounded peak was found and it is not the applied ratio" -- is the one that
    COULD contain the inverting case (a source already pitch-corrected at origin, whose duration
    moved while its pitch did not), and it is NOT implemented anywhere in this tree:
    `merge_video_repair:229-233` carries the only mention as an explicit refusal
    (`speed_verdict_rubberband_unimplemented`). ADDENDUM 2 rules on exactly this: the EXISTING
    routing stays, and that token "se liera a ATEMPO quand il s'implementera -- jamais a
    rubberband". So this function DOES NOT INVENT THE DETECTOR. It routes `asetrate` on both
    reachable outcomes, records which one it saw, and records the number a future detector would
    need -- `measured_ratio`, and how far it sits from 1.0 against how far the applied ratio sits
    from 1.0. A measurement logged on every run is how a calibration gets its data; a detector
    guessed today is how a wrong verdict gets shipped.

    WHY `asetrate` IS THE DEFAULT AND NOT A PREFERENCE. `docs/AUDIO_SPEED_POLICY.MD` (owner
    ruling, 2026-09-01) measured it winning 23 of 23 on PAL and 6 of 6 on NTSC against real
    masters, with 16 of 16 tracks reading `same_recording` after correction; the filter bake-off
    re-measured the same at the PAL scale (`architect/cases/BAKEOFF_speed_filters.md` and the ear
    dossier's classes D and E: `02_asetrate` is the only variant that returns the pitch to 0.0
    cents, 0 ms realignment, NCC 0.995/0.971). `rubberband` lost in both measured classes and is
    never routed here.

    Returns a routing dict; never None -- a pitch layer that could not run produces a routing with
    its refusal recorded, because the filter question still has an answer.
    """
    routing = {
        "filter_name": "asetrate",
        "pitch_measured_ratio": None,
        "pitch_peak": None,
        "pitch_refusal": None,
        "pitch_window_seconds": None,
        "pitch_test_discriminating": None,
        "pitch_tolerance_band": None,
        "inverting_case_detector": "not_implemented",
        "inverting_case_observation": None,
    }
    master_streams = audio_extract.streams_for(master_obj, language)
    candidate_streams = audio_extract.streams_for(candidate_obj, language)
    if not master_streams or not candidate_streams:
        routing["pitch_refusal"] = "no_stream_to_probe"
        routing["route_reason"] = (
            "the pitch layer was not asked: there is no comparison-language stream pair to probe "
            "it on. asetrate stands as the policy default (AUDIO_SPEED_POLICY 23/23 PAL, 6/6 "
            "NTSC), and 'not asked' is recorded as not asked, never as 'pitch intact'")
        return routing
    master_wav, candidate_wav, window = _pitch_probe_window(
        master_obj, candidate_obj, language, master_streams[0], candidate_streams[0],
        speed_factor, work_dir, sample_rate)
    if master_wav is None:
        routing["pitch_refusal"] = "probe_window_unavailable"
        routing["route_reason"] = (
            "the pitch layer was not asked: no window could be taken inside both tracks. "
            "asetrate stands as the policy default, and 'not asked' is not 'pitch intact'")
        return routing
    routing["pitch_window_seconds"] = round(window, 3)
    try:
        import pal_pitch_confirmer
        tools.dev_log(f"orchestrator: calling pal_pitch_confirmer.confirm_pitch "
                      f"master_wav={master_wav} candidate_wav={candidate_wav} "
                      f"predicted_ratio={float(speed_factor)} window_seconds={window}\n")
        reading = pal_pitch_confirmer.confirm_pitch(
            master_wav, candidate_wav, 0.0, float(speed_factor), window_seconds=window)
    except Exception as error:                                           # noqa: BLE001
        routing["pitch_refusal"] = "pitch_layer_raised"
        routing["route_reason"] = (
            f"the pitch layer raised {type(error).__name__} -- the instrument did not run, which "
            f"is not a reading about the pitch. asetrate stands as the policy default")
        tools.dev_log(f"orchestrator: pal_pitch_confirmer raised "
                      f"({type(error).__name__}: {error})\n")
        return routing
    finally:
        for temporary in (master_wav, candidate_wav):
            try:
                remove(temporary)
            except OSError:
                pass
    routing["pitch_measured_ratio"] = reading.get("measured_ratio")
    routing["pitch_peak"] = reading.get("peak")
    routing["pitch_refusal"] = reading.get("refusal")
    measured = reading.get("measured_ratio")
    if measured is not None:
        # THE NUMBER THE INVERTING-CASE DETECTOR WOULD NEED, MEASURED AND NOT BRANCHED ON -- the
        # same discipline this module already applies to the uncalibrated rate-relation slope
        # arm. An inverting case is "duration moved, pitch did NOT": it would read `measured`
        # near 1.0 while `speed_factor` sits far from it. Recorded as a distance, so a future
        # calibration has a population instead of a blank.
        routing["inverting_case_observation"] = {
            "measured_from_unity": round(abs(measured - 1.0), 6),
            "applied_from_unity": round(abs(float(speed_factor) - 1.0), 6),
        }
    # *** CAN THIS TEST EVEN TELL "THE PITCH MOVED" FROM "THE PITCH DID NOT"? At some ratios it
    # cannot, and until an independent tester caught it the prose below claimed a confirmation
    # anyway. `confirm_pitch` accepts when |measured - applied| <= TOL_ARM * applied, an
    # ABSOLUTE band on the ratio. When that band is WIDER than the whole defect being corrected,
    # it contains unity -- so "no refusal" is satisfied by a measurement of NO PITCH SHIFT AT ALL
    # and carries no information. MEASURED on a forced NTSC run: pitch_measured_ratio 1.000005
    # (a shift of 5e-6) against an applied 1.000999 (9.99e-4), two hundred times larger, and the
    # old sentence still read "confirms the pitch moved with the speed".
    #
    # THE CONDITION IS DERIVED, NOT CHOSEN: the band half-width is TOL_ARM * applied and the
    # quantity being measured is |applied - 1|, so the test discriminates exactly when
    # TOL_ARM * applied < |applied - 1|. At PAL (1.042708) that is 0.003128 against 0.042708 --
    # discriminating, and errid-70's measured 1.042504 is a real confirmation. At NTSC (1.001)
    # it is 0.003003 against 0.000999 -- the band swallows the defect, and no reading from this
    # instrument at that ratio can confirm anything. THE ROUTING DOES NOT CHANGE EITHER WAY
    # (ADDENDUM 2 keeps asetrate); only the sentence changes, so that it stops asserting a
    # confirmation nobody obtained. ***
    try:
        import pal_pitch_confirmer as _confirmer
        tolerance_arm = float(_confirmer.TOL_ARM)
    except Exception:                                                    # noqa: BLE001
        tolerance_arm = 0.0030
    applied = float(speed_factor)
    band_half_width = tolerance_arm * applied
    discriminating = band_half_width < abs(applied - 1.0)
    routing["pitch_test_discriminating"] = discriminating
    routing["pitch_tolerance_band"] = round(band_half_width, 7)
    if reading.get("refusal") is None and discriminating:
        routing["route_reason"] = (
            f"the pitch layer confirms the pitch moved with the speed (measured {measured}, "
            f"applied {applied:.7f}, peak {reading.get('peak')}; its +/-{band_half_width:.6f} "
            f"tolerance band excludes unity at this ratio, so agreement is informative): this "
            f"is the naive-speedup family, and asetrate is its exact inverse -- it undoes speed "
            f"AND pitch together (ruling body, step 2)")
    elif reading.get("refusal") is None:
        routing["route_reason"] = (
            f"the pitch layer did not refuse (measured {measured}, applied {applied:.7f}, peak "
            f"{reading.get('peak')}) BUT THAT IS NOT A CONFIRMATION AT THIS RATIO: its "
            f"+/-{band_half_width:.6f} tolerance band is wider than the {abs(applied - 1.0):.6f} "
            f"deviation being corrected, so the band contains unity and a completely unshifted "
            f"pitch would have passed the same test. Nothing here says the pitch moved. asetrate "
            f"stands on the policy default (AUDIO_SPEED_POLICY 23/23 PAL, 6/6 NTSC), not on this "
            f"reading")
    else:
        routing["route_reason"] = (
            f"the pitch layer returned {reading.get('refusal')} ({reading.get('reason')}). That "
            f"is NOT the inverting case and must not be read as one -- the inverting-case "
            f"detector is unimplemented in this tree (merge_video_repair:229-233) and ADDENDUM 2 "
            f"binds it to atempo only WHEN it is implemented. The existing routing stands: "
            f"asetrate, on AUDIO_SPEED_POLICY's 23/23 PAL and 6/6 NTSC")
    return routing


def rate_resample_routing(speed_factor, engine, master_obj, candidate_obj, language, work_dir,
                          sample_rate):
    """THE SPEED-CORRECTED CANDIDATE FOR THE RE-PRIME AT A CONFIRMED FACTOR -- UPSTREAM of
    chimeric, which never resamples (ADDENDUM 21.6: "chimeric NE RESAMPLE PLUS : le sweep rend
    les empreintes + alignements de TOUS les couples au facteur gagnant ... comparison_resample
    disparait"). This was `comparison_resample`, called from inside chimeric; it is now called
    by `repair()` right after the rate decision, and its routing drives `prime_couples`'
    re-fingerprinting of every couple's candidate side.

    THE ENGINE IS THE RATE ARM'S (ADDENDUM 30.5): `rate_arm` measured which of asetrate and
    atempo aligns; this builds that engine's chain (`merge_video_resample.build_transform_chain`)
    for every couple's re-prime and records the pitch layer's reading beside it.

    Returns `(routing_or_None, cause_or_None)`. The routing is a DESCRIPTION, not a file: the
    candidate's comparison tracks are speed-corrected inside the SAME ffmpeg invocation that
    extracts them for fingerprinting (`fingerprint_track`), deleted after one pass, never a
    product track -- DELIVERING speed-corrected audio is plan application's (ADDENDUM 8).

    NEVER CALLED AT speed_factor = 1 (ADDENDUM 6: "aucun filtre de correction ne tourne JAMAIS
    sur une piste dont la vitesse n'a pas change"): `repair()` reaches it only on a confirmed
    factor other than 1, and the re-prime it feeds does not exist at 1 -- the prime at factor 1
    already holds the raw fingerprints.

    EXACT RATIONALS, NEVER FLOATS, AND THE ONE THAT MATTERS IS THE EFFECTIVE ONE.
    `merge_video_resample.build_speed_filter_chain` is the pipeline's single authority on this
    arithmetic and it is called, not reimplemented: `asetrate` takes an INTEGER, so the factor
    actually obtained is `intermediate / round(intermediate / ratio)` and is NOT the one
    requested. Both are carried on the routing -- the requested one as the exact `Fraction` the
    sweep won with, the effective one as the exact integer ratio the filter will really apply --
    and it is the EFFECTIVE one every downstream length is computed from, because it is the one
    the audio will actually have.
    """
    source_rate = _candidate_audio_sample_rate(candidate_obj)
    if source_rate is None:
        return None, "rate_sweep_no_sample_rate"
    # THE CHAIN IS BUILT AT THE CANDIDATE'S OWN SOURCE RATE, NOT AT THE COMPARISON GRID, and the
    # difference is small, measured, and free. `build_speed_filter_chain` divides the integer-
    # asetrate rounding error by running through an intermediate at 8x the rate it is given, so
    # the higher that rate, the smaller the residual. MEASURED at the PAL rational 1001/960:
    #     48000 Hz (the candidate's own)  asetrate=368272  effective 1.0427075640
    #                                     7.38e-7 relative, 0.90 ms of drift over a 1220 s track
    #     44100 Hz (the comparison grid)  asetrate=338350  effective 1.0427072558
    #                                     1.03e-6 relative, 1.26 ms over the same track
    # Both sit four orders under a 124 ms fingerprint quantum, so NEITHER would change an
    # alignment -- the source rate is chosen because it is the better of two numbers that cost
    # the same, not because the other one would have failed. The extraction's own `-ar` resamples
    # to the comparison grid afterwards, so nothing is lost by correcting on the finer grid first.
    try:
        import merge_video_resample
        ratio_decimal = (Decimal(speed_factor.numerator) / Decimal(speed_factor.denominator)
                         if isinstance(speed_factor, Fraction) else Decimal(str(speed_factor)))
        chain, effective = merge_video_resample.build_transform_chain(
            source_rate, ratio_decimal, engine)
        intermediate = target = None
        if engine == "asetrate":
            _chain, _effective, intermediate, target = (
                merge_video_resample.build_speed_filter_chain(source_rate, ratio_decimal))
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"orchestrator: build_speed_filter_chain refused "
                      f"({type(error).__name__}: {error})\n")
        return None, "rate_resample_unbuildable"
    routing = pitch_routing(speed_factor, master_obj, candidate_obj, language, work_dir,
                            sample_rate)
    # THE ENGINE IS THE RATE ARM'S MEASUREMENT (ADDENDUM 30.5): the finalist that aligned; the
    # pitch layer's reading stays on the routing as an observation.
    routing["route_reason"] = (f"engine {engine} measured by the rate arm (the finalist that "
                               f"aligned); pitch layer: {routing.get('route_reason')}")
    routing["filter_name"] = engine
    routing["inverting_case_detector"] = "rate_arm_engine_comparison"
    routing.update({
        "modality": MODALITY,
        "side": "candidate",
        "rule": "reprime_extraction_only_never_a_product_track",
        "requested_ratio": (f"{speed_factor.numerator}/{speed_factor.denominator}"
                            if isinstance(speed_factor, Fraction) else str(speed_factor)),
        "requested_ratio_value": float(speed_factor),
        "source_sample_rate": source_rate,
        "comparison_sample_rate": sample_rate,
        "filter_chain": chain,
        "intermediate_rate": intermediate,
        "asetrate_target": target,
        "effective_ratio": effective,
        "effective_ratio_str": str(effective),
        "tag_factor": merge_video_resample.format_factor(effective),
    })
    return routing, None


def _candidate_audio_sample_rate(candidate_obj):
    """The candidate's own audio sampling rate, for the filter's arithmetic.

    Reads the same two places the retired sweep's sample-rate reader read
    (ffprobe first, MediaInfo in fallback) rather than importing it: that name is in the module
    the switch will eventually delete, and a stage that is not the switch should not add a new
    dependency on it. Returns None when nothing is readable -- never a default rate, because a
    guessed source rate produces a filter whose EFFECTIVE factor is wrong in a way nothing
    downstream can detect.
    """
    for _language, audios in (getattr(candidate_obj, "audios", None) or {}).items():
        for audio in audios:
            rate = audio.get("ffprobe", {}).get("sample_rate") or audio.get("SamplingRate")
            if rate is not None:
                try:
                    return int(float(rate))
                except (TypeError, ValueError):
                    continue
    return None


# ---------------------------------------------------------------------------
# STEP 4 -- FRAME-EXACT HOLE RESOLUTION
# ---------------------------------------------------------------------------

def _round_half_up(value):
    """One rounding, on an exact rational, half away from zero -- never `int()`'s truncation."""
    value = Fraction(value)
    return int(value + Fraction(1, 2)) if value >= 0 else -int(-value + Fraction(1, 2))


def _exact_video_rate(video_obj):
    """THIS FILE'S frame rate as an EXACT RATIONAL -- `(Fraction, source)` or `(None, reason)`.

    Never `int(fps)` and never MediaInfo's decimal `FrameRate`: `Fraction("23.976")` is 2997/125,
    not 24000/1001, and the difference is one frame every 41.7 s -- 90 frames over an hour of the
    corpus's longest master. The exact source, in order: MediaInfo's own `FrameRate_Num` /
    `FrameRate_Den` (MEASURED present on the errid-70, errid-202 and errid-27 masters: 24000/1001;
    ABSENT on errid-27's candidate), then ffprobe's `r_frame_rate` through
    `scene_anchor._probe_frame_rate`, the same probe the resolvers themselves use for the
    candidate. No decimal fallback: a grid that cannot be read exactly is a grid not measured.
    """
    video = getattr(video_obj, "video", None) or {}
    try:
        num, den = video.get("FrameRate_Num"), video.get("FrameRate_Den")
        if num not in (None, "") and den not in (None, ""):
            rate = Fraction(int(str(num)), int(str(den)))
            if rate > 0:
                return rate, "mediainfo_num_den"
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    try:
        import scene_anchor
        rate, reason = scene_anchor._probe_frame_rate(video_obj.filePath)
    except Exception as error:                                           # noqa: BLE001
        return None, f"probe_raised:{type(error).__name__}"
    if rate is None:
        return None, f"ffprobe_r_frame_rate:{reason}"
    return rate, "ffprobe_r_frame_rate"


def _video_duration_ms(video_obj):
    """The VIDEO stream's own duration in ms, as the output timeline reads it -- the same
    expression as `merge_video_chimeric.get_master_timeline_length_ms` (`video["Duration"]`
    seconds, times 1000), restated rather than imported so this stage adds no dependency on the
    module stage 5 rewrites. None when unreadable, never zero."""
    try:
        return Decimal(str(video_obj.video["Duration"])) * Decimal("1000")
    except Exception:                                                    # noqa: BLE001
        return None


def frame_domain(master_obj, candidate_obj, speed_factor):
    """The pair's FRAME DOMAIN, computed once per pair and carried onto every hole.

    Returns `(domain, None)` or `(None, reason)`. The domain says how a millisecond of the
    alignment becomes a frame of each file:
      * master frames are on the master's exact grid (F1: every boundary this chain ships is a
        master frame number);
      * the candidate's alignment milliseconds are MASTER-EQUIVALENT time. At speed factor 1
        that is simply candidate time. Under the re-prime at a factor it is the candidate's raw
        time times `r` -- `fingerprint_track` extracts the corrected audio at
        `duration * effective_ratio` -- so a candidate frame's NATIVE index is
        `equivalent_ms / r` on the candidate's own exact rate.
    `r` handed to the video instruments is the sweep's EXACT rational (1001/960, 1001/1000), not
    the audio filter's effective ratio: the filter's integer `asetrate` makes it 7.4e-7 off at
    PAL (0.9 ms over 1220 s -- logged beside it, below any frame), and the VIDEO relation of a
    speed-up pair is the exact one. It is corroborated, not assumed: the candidate's native
    rate over the master's is logged against it (errid-70: 25 / (24000/1001) = 1001/960 exactly;
    errid-27: 24 / (24000/1001) = 1001/1000 exactly).
    """
    master_rate, master_source = _exact_video_rate(master_obj)
    if master_rate is None:
        return None, f"master_grid_unmeasured:{master_source}"
    candidate_rate, candidate_source = _exact_video_rate(candidate_obj)
    if candidate_rate is None:
        return None, f"candidate_grid_unmeasured:{candidate_source}"
    master_timeline_ms = _video_duration_ms(master_obj)
    if master_timeline_ms is None:
        return None, "master_timeline_unmeasured"
    candidate_raw_ms = _video_duration_ms(candidate_obj)
    scale = None
    if speed_factor is not None and speed_factor != 1:
        scale = Fraction(speed_factor)
    video_ratio = candidate_rate / master_rate
    return {
        "master_rate": master_rate, "master_rate_source": master_source,
        "candidate_rate": candidate_rate, "candidate_rate_source": candidate_source,
        "time_scale": scale,
        "video_rate_ratio": video_ratio,
        "video_ratio_matches_speed_factor": (None if scale is None else video_ratio == scale),
        "master_timeline_ms": master_timeline_ms,
        "candidate_raw_duration_ms": candidate_raw_ms,
        "candidate_equivalent_duration_ms": (
            None if candidate_raw_ms is None
            else candidate_raw_ms * (Decimal(scale.numerator) / Decimal(scale.denominator)
                                     if scale is not None else Decimal(1))),
        "frame_ms": Fraction(1000) / master_rate,
    }, None


def _exact_ms_of_frame(frame, domain):
    """A master frame index as the EXACT millisecond it starts at, on the exact rational grid --
    a string, six decimals, never a float carried into arithmetic (ADDENDUM 9: the result must
    carry the millisecond measurement beside the frame indices). None in, None out."""
    if frame is None:
        return None
    return f"{float(Fraction(frame) * 1000 / domain['master_rate']):.6f}"


def _audio_offsets(hole):
    """The AUDIO's own offsets across a hole, UNROUNDED milliseconds (ADDENDUM 9 point 2: "les
    offsets audio a la milliseconde ... jamais arrondi a la frame video"). They are the
    aligner's `offset_points * quantum_ms`, i.e. precise to ONE FINGERPRINT QUANTUM (~124 ms) --
    COARSER than a frame, which is why they are carried and flagged, not presented as the
    offset to apply: stage 5 must refine each to sub-frame milliseconds before applying it.

    ON THE FILE'S CLOCK, NOT THE TRACK'S: the aligner's offsets are track-relative (see
    `couple_start_delta_ms`), and each couple's start delta is folded in ONCE, when its holes are
    put on the file's clock (`hole_on_file_clock`), before the union and before any video is
    searched -- so `offset_before_ms` / `offset_after_ms` here are already file-time, and every
    reader downstream (the video search, the plan) receives them as they are."""
    quantum_ms = hole["quantum_ms"]
    return {
        "audio_offset_before_ms": (None if hole.get("offset_before_ms") is None
                                   else round(hole["offset_before_ms"], 3)),
        "audio_offset_after_ms": (None if hole.get("offset_after_ms") is None
                                  else round(hole["offset_after_ms"], 3)),
        "audio_offset_precision_ms": round(quantum_ms, 3),
        "track_delay_delta_ms": hole.get("track_delay_delta_ms"),
    }


def couple_start_delta_ms(master_obj, candidate_obj, language, master_stream, candidate_stream,
                          speed_factor):
    """THE FINGERPRINT OFFSETS ARE TRACK-RELATIVE; THE CUTS AND THE PLAN ARE ON THE FILE'S CLOCK.

    `fingerprint_track` extracts each track from its OWN first sample, so the aligner's offset
    is `candidate_track_time = master_track_time + offset`. A container that DELAYS a track (its
    first block at `start_time` > 0 -- mediainfo `Delay`) puts that track's time zero at
    `start_time` on the file's clock, which is the clock the video, the frame search, the
    assembly's `atrim` and the verifier all read. So on the file's clock the offset is
        offset + candidate_start * r - master_start
    (`r` = the confirmed speed factor: the candidate's comparison timeline is the corrected
    one). MEASURED, found by the ceiling analysis on Tougen E07: master ja start 1103 ms,
    candidate ja 120 ms -- every offset handed to the video search was 983 ms (~24 frames) off,
    and the edge anchors validated WRONG shifts (head boundary 192 instead of 48, tail 33857
    instead of 34547). Returns `(delta_ms, master_start_ms, candidate_start_ms)`; a stream with
    no readable `start_time` reads 0 -- the same convention the assembly's `atrim` applies to
    it, so the two stay on one clock whatever the probe returned.

    THE POSITIONS MOVE WITH THE OFFSETS (ADDENDUM 19 b, verified on every path for the
    Addendum 21 batch). A hole's master bracket is a master TRACK time too, and the video search
    reads it as a file time: `hole_on_file_clock` adds `master_start` to it (and
    `candidate_start * r` to the candidate's), so a delayed master track no longer offsets the
    search bracket by its delay either -- the offsets were folded before this batch, the brackets
    were not."""
    import merge_video_chimeric

    def audio_of(video_obj, stream):
        for entry in (getattr(video_obj, "audios", None) or {}).get(language) or []:
            if str(entry.get("StreamOrder")) == str(stream):
                return entry
        return None

    scale = _decimal(Fraction(speed_factor)) if speed_factor not in (None, 1) else Decimal(1)
    master_start = merge_video_chimeric.get_stream_start_ms(audio_of(master_obj, master_stream))
    candidate_start = merge_video_chimeric.get_stream_start_ms(
        audio_of(candidate_obj, candidate_stream))
    return candidate_start * scale - master_start, master_start, candidate_start


def _master_frame_of_ms(ms, domain):
    return _round_half_up(Fraction(str(ms)) * domain["master_rate"] / 1000)


def _candidate_native_frame(equivalent_frame, domain):
    """A master-grid EQUIVALENT frame index (what the resolvers return for the candidate) as the
    CANDIDATE FILE'S OWN frame number: equivalent seconds, divided by `r`, counted on the
    candidate's exact native rate. Exactly the identity on a same-rate pair and on an exact
    speed-up pair (errid-70: k -> k). None in, None out."""
    if equivalent_frame is None:
        return None
    scale = domain["time_scale"] or Fraction(1)
    return _round_half_up(Fraction(equivalent_frame) / domain["master_rate"] / scale
                          * domain["candidate_rate"])


def _candidate_native_frame_of_ms(equivalent_ms, domain):
    scale = domain["time_scale"] or Fraction(1)
    return _round_half_up(Fraction(str(equivalent_ms)) / 1000 / scale * domain["candidate_rate"])


def _shift_search_frames(domain, quantum_ms):
    """The anchors' shift-search half-width, DERIVED for an offset read off fingerprint points.

    `scene_anchor.EDGE_SHIFT_SEARCH_FRAMES` (2) is one frame of rounding a millisecond-precise
    offset onto the grid, plus one frame of extraction labelling. The orchestrator's offsets are
    integer fingerprint points, precise to one QUANTUM, so the rounding term becomes one quantum
    in frames, rounded up; the labelling frame is unchanged. At 24000/1001 with a 124.07 ms
    quantum that is 3 + 1 = 4. MEASURED need: errid-70's head, nominal -45 frames, true -48.
    """
    return int(math.ceil(Fraction(str(quantum_ms)) / domain["frame_ms"])) + 1


def _declined(hole, cause_detail, evidence=None, **extra):
    outcome = {"modality": MODALITY, "status": HOLE_DECLINED, "kind": hole["kind"],
               "why_token": hole["why_token"], "cause": "hole_resolution_declined",
               "resolver_reason": cause_detail, "evidence": evidence}
    outcome.update(extra)
    return outcome


def _two_anchor_call(hole, domain, master_obj, candidate_obj, low_ms, high_ms,
                     offset_before_ms, offset_after_ms, step_ms, quantum_ms, probe,
                     resolve_shift=True):
    """ONE call of the interior resolver, launched and logged as its own step. Exception-
    isolated: a resolver that raises is a named decline, never a crash of the repair (the same
    rule the live chain's own call sites carry for these two functions)."""
    import scene_anchor
    deadline = hole.get("deadline")
    if deadline is not None and time.monotonic() > deadline:
        # THE HOLE'S BUDGET IS SPENT (ADDENDUM 26.3): no further probe is launched on it.
        return {"declined": True, "reason": "hole_budget_exceeded",
                "evidence": f"probe={probe} not launched: the hole's budget is spent"}
    step_launch("two_anchor", candidate=candidate_obj.filePath, probe=probe,
                resolve_shift=resolve_shift, cluster=hole.get("cluster_id"),
                shared_scan_windows_cached=(None if hole.get("scan_cache") is None
                                            else len(hole["scan_cache"])),
                bracket_ms=[round(float(low_ms), 2), round(float(high_ms), 2)],
                offset_before_ms=round(float(offset_before_ms), 3),
                offset_after_ms=round(float(offset_after_ms), 3),
                time_scale=(None if domain["time_scale"] is None
                            else f"{domain['time_scale'].numerator}/"
                                 f"{domain['time_scale'].denominator}"))
    try:
        result = scene_anchor.locate_scene_anchors(
            master_obj.filePath, candidate_obj.filePath,
            domain["master_rate"].numerator, domain["master_rate"].denominator,
            float(low_ms), float(high_ms), float(offset_before_ms), float(offset_after_ms),
            step_ms=float(step_ms), quantum_ms=float(quantum_ms),
            candidate_time_scale=domain["time_scale"],
            normalise_geometry=True, resolve_shift=resolve_shift,
            shift_search_frames=_shift_search_frames(domain, quantum_ms),
            cluster_window=hole.get("cluster_window"), scan_cache=hole.get("scan_cache"),
            deadline=deadline)
    except Exception as error:                                           # noqa: BLE001
        result = {"declined": True, "reason": f"resolver_raised:{type(error).__name__}",
                  "evidence": str(error)[:300]}
    step_result("two_anchor", candidate=candidate_obj.filePath, probe=probe,
                declined=result["declined"], reason=result.get("reason"),
                anchor_a=result.get("anchor_a_frame"), anchor_b=result.get("anchor_b_frame"),
                before_shift=result.get("before_shift_frames"),
                after_shift=result.get("after_shift_frames"),
                nominal_before_shift=result.get("nominal_before_shift_frames"),
                nominal_after_shift=result.get("nominal_after_shift_frames"),
                forward_walk=result.get("forward_walk_frames"),
                backward_walk=result.get("backward_walk_frames"),
                sweep_crossed=result.get("sweep_crossed"), net_kind=result.get("net_kind"),
                unmatched_span_matches_before=result.get("unmatched_span_matches_before"),
                unmatched_span_matches_after=result.get("unmatched_span_matches_after"),
                evidence=result.get("evidence"))
    return result


def _span_noise_reading(result):
    """Which hypothesis, if any, MATCHES THE MAJORITY of the span the two fronts left unmatched.

    Returns "before", "after", "both" or None. None is the only reading under which the span is
    what the sweep says it is -- content that neither side's shift explains.

    WHY A MAJORITY, AND WHY IT IS NOT A TUNING KNOB. The sweep stops on a short run of
    consecutive mismatches (`scene_anchor.SWEEP_SUSTAINED_MISMATCH_FRAMES`), which is only safe
    if pHash noise inside common content never runs that long. MEASURED on errid-70 under the
    true shift -70, over 3650 frames of common content: mismatch runs of 3 (x6), 4 (x3), 5 and
    15 -- the backward walk stopped 124 frames in and reported a 3523-frame "divergent" span.
    That span matches the after-shift on 92-96 % of its frames; the genuinely divergent spans
    measured on errid-202 (holes 1 and 4) match EITHER shift on 0-10 %. "More frames of the
    span agree with a hypothesis than disagree" sits between the two with the data far on both
    sides, and it is the plain meaning of "the span is that hypothesis's content" -- not a
    number fitted to one pair.
    """
    readings = []
    for label in ("before", "after"):
        matched, readable = result.get(f"unmatched_span_matches_{label}") or [0, 0]
        if readable and 2 * matched > readable:
            readings.append(label)
    if len(readings) == 2:
        return "both"
    return readings[0] if readings else None


def _checked_two_anchor(hole, domain, master_obj, candidate_obj, low_ms, high_ms,
                        offset_before_ms, offset_after_ms, step_ms, quantum_ms, probe,
                        resolve_shift=True):
    """`_two_anchor_call`, then the majority test on what its fronts left unmatched.

    A front that stopped INSIDE common content (the span's majority matches the OTHER side's
    shift) leaves the cut at the front that did not stop on noise: when the after-shift claims
    the span, the backward walk stopped early and the cut is at the FORWARD front; when the
    before-shift claims it, the reverse. The search is then asked again with its bracket
    collapsed onto that front -- the anchors re-seat beside the cut, the walks between them are
    short -- and the second answer must pass the same test. A span both shifts claim, or a
    second answer that still fails, is a NAMED decline: never the first answer's frames, which
    are proven wrong by the resolver's own hashes.

    A DECLINE AFTER A ONE-SIDED CLAIM CARRIES THE CLAIM (ADDENDUM 19 c): `claimed_by` and the
    claiming side's RESOLVED shift (`claimed_shift_frames`), so `_resolve_interior` can put the
    no-cut hypothesis to the video at exactly that shift -- the span's majority reading under
    one shift is the video refuting the audio's step, and it is acted on, not only logged.
    """
    def _with_claim(declined, reading, source):
        shift = source["before_shift_frames"] if reading == "before" else source[
            "after_shift_frames"]
        return dict(declined, claimed_by=reading, claimed_shift_frames=shift,
                    claimed_span_master=[source["pre_collapse_start_master"],
                                         source["pre_collapse_end_master"]],
                    claimed_matches=source[f"unmatched_span_matches_{reading}"])

    result = _two_anchor_call(hole, domain, master_obj, candidate_obj, low_ms, high_ms,
                              offset_before_ms, offset_after_ms, step_ms, quantum_ms, probe,
                              resolve_shift=resolve_shift)
    if result["declined"]:
        return result
    reading = _span_noise_reading(result)
    if reading is None:
        return result
    frame_ms = float(domain["frame_ms"])
    step_result("sweep_front_inside_common_content", candidate=candidate_obj.filePath,
                probe=probe, claimed_by=reading,
                unmatched_master_frames=[result["pre_collapse_start_master"],
                                         result["pre_collapse_end_master"]],
                matches_before=result["unmatched_span_matches_before"],
                matches_after=result["unmatched_span_matches_after"])
    if reading == "both" and result["before_shift_frames"] == result["after_shift_frames"]:
        # ONE SHIFT ON BOTH SIDES AND THE SPAN'S MAJORITY MATCHES IT: "both" is then a single
        # claim, not two -- the walks stopped on pHash noise inside content the one shift
        # explains, so nothing between the anchors differs (`_interior_verdict` reads it as
        # `no_cut_confirmed`). MEASURED, id 134 hole 2: the no-cut probe at -72 seats both
        # anchors, and 197 of the 257 frames the walks left match -72.
        step_result("single_shift_claims_span", candidate=candidate_obj.filePath, probe=probe,
                    shift=result["before_shift_frames"],
                    matches=result["unmatched_span_matches_before"])
        return dict(result, span_majority_under_single_shift=True)
    if reading == "both":
        return {"declined": True, "reason": "unmatched_span_claimed_by_both_shifts",
                "evidence": (f"span {result['pre_collapse_start_master']}-"
                             f"{result['pre_collapse_end_master']} before="
                             f"{result['unmatched_span_matches_before']} after="
                             f"{result['unmatched_span_matches_after']}")}
    front = (result["pre_collapse_start_master"] if reading == "after"
             else result["pre_collapse_end_master"])
    narrowed = _two_anchor_call(hole, domain, master_obj, candidate_obj,
                                front * frame_ms, (front + 1) * frame_ms,
                                offset_before_ms, offset_after_ms, step_ms, quantum_ms,
                                probe=f"{probe}_refront_{reading}", resolve_shift=resolve_shift)
    if narrowed["declined"]:
        return _with_claim(narrowed, reading, result)
    if _span_noise_reading(narrowed) is not None:
        return _with_claim(
            {"declined": True, "reason": "sweep_front_inside_common_content",
             "evidence": (f"first span {result['pre_collapse_start_master']}-"
                          f"{result['pre_collapse_end_master']} claimed by {reading}; "
                          f"re-fronted span {narrowed['pre_collapse_start_master']}-"
                          f"{narrowed['pre_collapse_end_master']} still claimed "
                          f"(before={narrowed['unmatched_span_matches_before']} "
                          f"after={narrowed['unmatched_span_matches_after']})")},
            reading, result)
    return narrowed


def _interior_verdict(result):
    """ONE shift and nothing left between the fronts -> the hole closes (ADDENDUM 3). Two shifts
    and fronts that OVERLAP, on either file's axis -> a static span, pinned at the end of the
    ambiguous zone (ADDENDUM 4, 12, 13).
    Anything else is a cut pinned on both files.

    WHY THESE ARE THE RIGHT READINGS OF THE SWEEP. The forward walk from A only advances on
    frames that MATCH under A's shift, the backward walk from B only on frames matching under
    B's. (1) With the two shifts EQUAL, both walks test the same hypothesis, so they either both
    cross the whole span -- no frame between the anchors disagrees: no cut -- or both stop on
    the same disagreeing run, leaving a same-length replacement between them. (2) With the
    shifts DIFFERENT, the two fronts can only overlap if some frames match under BOTH shifts at
    once, and on content that moves that is impossible: one candidate frame cannot be the
    picture of two different master frames. The overlap can show on EITHER axis --
    `sweep_crossed` is the master's; a candidate interval of negative length is the candidate's
    -- and it is the ruling's own detection, "chevauchement ... des deux fronts dans un span
    auto-similaire". MEASURED on errid-202 hole 1: shift -24 holds to master 5716, -48 from
    5729, 12 master frames match neither, and past 5738 BOTH shifts match 6/6 -- a static
    span; the candidate fronts overlap by 12 frames while the master's do not cross.

    (3) THE ANCHOR ITSELF SITS IN THE STATIC ZONE (ADDENDA 12-13, read here since the Addendum
    21 batch). `scene_anchor` declines `anchor_ambiguous_static_span` when its frame-scan anchor
    validates at two or more neighbouring shifts, and names the side (`anchor_a_ambiguous` /
    `anchor_b_ambiguous`, each with its `ambiguous_shift_span`). No sweep ran, so there are no
    fronts -- but the ruling's placement does not need them: the N frames go IN ONE BLOCK at the
    RIGHT END of the ambiguous zone, and the zone's right end is known from the refused anchor
    itself (see `_pin_point`). So this decline is not a failure to find the cut; it is the
    static-span verdict, reached from the anchor side. Every other decline stays a decline.
    """
    if result.get("declined"):
        if (result.get("reason") == "anchor_ambiguous_static_span"
                and (result.get("anchor_a_ambiguous") or result.get("anchor_b_ambiguous"))):
            return HOLE_PINNED_TO_AMBIGUOUS_ZONE_END
        return HOLE_DECLINED
    same_shift = result["before_shift_frames"] == result["after_shift_frames"]
    if same_shift and (result["master_end_frame"] == result["master_start_frame"]
                       or result.get("span_majority_under_single_shift")):
        return HOLE_NO_CUT_CONFIRMED
    candidate_overlap = result["candidate_end_frame"] < result["candidate_start_frame"]
    if not same_shift and (result["sweep_crossed"] or candidate_overlap):
        return HOLE_PINNED_TO_AMBIGUOUS_ZONE_END
    return HOLE_RESOLVED


def _pin_point(result):
    """THE END OF THE AMBIGUOUS ZONE -- ADDENDUM 4 as confirmed and named by the owner in
    ADDENDA 12 and 13 (2026-09-24). Returns `(pin_frame, ambiguous_frames)`.

    The owner's vocabulary, verbatim in substance: A = the last common scene before, B = the
    first common scene after (`scene_anchor`'s anchors). THE AMBIGUOUS ZONE = the frames that
    match under BOTH shifts; its END is the right edge of those frames -- where the walk from B
    stops matching under the after-shift. The N-frame addition or removal is made IN ONE BLOCK at
    that end, never in the middle: addition => candidate_zone_end + N; removal =>
    candidate_zone_end - N. Anchor B is only the special case where the zone extends all the way
    to it (a fully static span, Bleach). The verdict is `boundary_pinned_to_ambiguous_zone_end`
    (it was `boundary_pinned_to_right_anchor` until ADDENDUM 13 fixed the vocabulary; the
    behaviour below did not change, only the name).

    The ruling's premise is that BOTH walks traverse the whole static span, so the ambiguous
    region is everything between the anchors and its right end IS anchor B. On real media the
    fronts can overlap by far less than the span, and then the literal anchor B is provably
    wrong. MEASURED on errid-70 hole 1: anchors 10936 / 14570, forward front 11155, backward
    front 11153 -- a TWO-frame overlap in a 3634-frame span. Pinning at B would put the 23-frame
    fill at 14547-14570, on content the backward walk had just proven common under its shift
    for 3417 frames, and would read the before-shift across 3400 frames proven NOT to match it;
    the measured cut is at 11131-11153 (a 22-frame mismatch run under the after-shift ends at
    11151). So the pin is the right end of the CONFLICT:
      * master fronts crossed: the conflict is [backward front, forward front) -> pin at the
        forward front;
      * candidate fronts overlapped by k frames (master fronts did not cross): the backward
        walk's first k master frames read candidate frames the forward walk also claimed ->
        pin at backward front + k.
    When both walks cross the whole span -- the ruling's own case -- the forward front IS
    anchor B and this is the ruling verbatim. errid-202 hole 1 (candidate overlap 12, backward
    front 5729) pins at 5741, which is also its anchor B.

    AN AMBIGUOUS ANCHOR (the `anchor_ambiguous_static_span` decline -- no sweep, no fronts):
      * B refused as ambiguous: B's own window is inside the static zone, so the zone reaches
        B -- the ruling's ORIGINAL case, "quand la zone ambigue s'etend jusqu'a B, l'extremite
        droite est B": pin at the refused B's frame;
      * only A refused: the frame scan walked BACKWARD from the bracket and A's window
        `[A - n, A)` is the static stretch closest to it; the frames between A and the bracket
        did not validate, so the zone ENDS at A: pin at A.
    The ambiguous size reported is the refused window's own frame count -- the stretch measured
    to match under several shifts; its full extent was not walked and is not claimed.
    """
    if result.get("declined"):
        ambiguous = result.get("anchor_b_ambiguous") or result.get("anchor_a_ambiguous")
        return ambiguous["anchor"], ambiguous["n_frames"]
    if result["sweep_crossed"]:
        start, end = result["pre_collapse_start_master"], result["pre_collapse_end_master"]
        return start, start - end
    overlap = result["candidate_start_frame"] - result["candidate_end_frame"]
    return result["pre_collapse_end_master"] + overlap, overlap


def _pinned_frames(result):
    """ADDENDUM 4/12/13's placement, written out for BOTH signs of the step, at `_pin_point`
    (the end of the ambiguous zone): "l'ajout/suppression de N frames se fait D'UN BLOC A
    L'EXTREMITE DROITE de cette zone ... ajout => timecode_droit_candidat + N ; suppression =>
    timecode_droit_candidat - N". With `delta = after_shift - before_shift` (the
    candidate's net frames, + = it holds more) and P the pin:
      * delta >= 0 (addition): nothing of the master is replaced; the candidate's `delta` extra
        frames are the ones just before P's content -- master [P, P), candidate
        [P + before, P + after). At P = anchor B (zone reaching B) this is `locate_scene_anchors`' own collapse.
      * delta < 0 (deletion): the master's `-delta` frames the candidate lacks are FILLED, and
        the fill is the frames just before P -- master [P + delta, P), candidate empty at
        P + after. The resolver's collapse would instead return master [B, B) with an INVERTED
        candidate interval, i.e. `-delta` candidate frames read twice across the pin; on an
        AUDIO product that is an audible repetition, so it is not used.
    Both forms are continuous with the two shifts: the before-shift read ends exactly where the
    after-shift read begins, on each axis.
    """
    pin, _ambiguous = _pin_point(result)
    before, after = result["before_shift_frames"], result["after_shift_frames"]
    delta = after - before
    if delta >= 0:
        return pin, pin, pin + before, pin + after
    return pin + delta, pin, pin + after, pin + after


def _interior_outcome(hole, domain, result, status, refuted_proposal=None):
    grid = result["grid"]
    master_start, master_end = result["master_start_frame"], result["master_end_frame"]
    candidate_start, candidate_end = result["candidate_start_frame"], result["candidate_end_frame"]
    if status == HOLE_PINNED_TO_AMBIGUOUS_ZONE_END:
        master_start, master_end, candidate_start, candidate_end = _pinned_frames(result)
    length_master = master_end - master_start
    length_candidate = candidate_end - candidate_start
    net_kind = ("addition" if length_candidate > length_master
                else "deletion" if length_candidate < length_master
                else "still_image" if length_master == 0 else "ordinary")
    outcome = {
        "modality": MODALITY, "status": status, "kind": hole["kind"],
        "why_token": hole["why_token"], "cause": None,
        "grid": f"{grid['num']}/{grid['den']}",
        "anchor_a_frame": result["anchor_a_frame"], "anchor_b_frame": result["anchor_b_frame"],
        "master_start_frame": master_start, "master_end_frame": master_end,
        "candidate_start_frame_equivalent": candidate_start,
        "candidate_end_frame_equivalent": candidate_end,
        "candidate_start_frame": _candidate_native_frame(candidate_start, domain),
        "candidate_end_frame": _candidate_native_frame(candidate_end, domain),
        # ADDENDUM 9 point 1: THESE ARE THE FILL BOUNDS, EXACTLY -- the tightest frames the
        # walks prove, and their exact milliseconds beside them.
        "master_start_ms": _exact_ms_of_frame(master_start, domain),
        "master_end_ms": _exact_ms_of_frame(master_end, domain),
        # The frame shifts place CUTS; they are never an audio offset (ADDENDUM 9 point 2).
        "video_shift_ms_frame_quantised": [
            _exact_ms_of_frame(result["before_shift_frames"], domain),
            _exact_ms_of_frame(result["after_shift_frames"], domain)],
        **_audio_offsets(hole),
        "net_kind": net_kind,
        "frames_to_cut": max(0, length_candidate - length_master),
        "frames_to_fill": max(0, length_master - length_candidate),
        # THE RESOLVER'S OWN FRONTS, KEPT BESIDE ANY PLACEMENT MADE FROM THEM.
        "sweep_master_frames": [result["master_start_frame"], result["master_end_frame"]],
        "sweep_candidate_frames_equivalent": [result["candidate_start_frame"],
                                              result["candidate_end_frame"]],
        "pre_collapse_master_frames": [result["pre_collapse_start_master"],
                                       result["pre_collapse_end_master"]],
        "before_shift_frames": result["before_shift_frames"],
        "after_shift_frames": result["after_shift_frames"],
        "nominal_before_shift_frames": result["nominal_before_shift_frames"],
        "nominal_after_shift_frames": result["nominal_after_shift_frames"],
        "forward_walk_frames": result["forward_walk_frames"],
        "backward_walk_frames": result["backward_walk_frames"],
        "span_frames": result["anchor_b_frame"] - result["anchor_a_frame"],
        "sweep_crossed": result["sweep_crossed"],
        "geometry": (result.get("geometry") or {}).get("verdict"),
        "evidence": result.get("evidence"),
    }
    if status == HOLE_PINNED_TO_AMBIGUOUS_ZONE_END:
        outcome["cause"] = "static_span_ambiguity"
        outcome["pin_frame"], outcome["ambiguous_frames"] = _pin_point(result)
    if refuted_proposal is not None:
        outcome["refuted_proposal"] = refuted_proposal
    return outcome


def _ambiguous_pin_outcome(hole, domain, result):
    """ADDENDA 12-13 FROM THE ANCHOR SIDE: `scene_anchor` refused an anchor as ambiguous (it
    validates at several neighbouring shifts -- it sits in a static zone). The pin is
    `_pin_point`'s, and the N frames come FROM THE AUDIO STEP, as the addenda state them
    ("ajout => timecode_droit_candidat + N ; suppression => timecode_droit_candidat - N"): the
    audio's step is the only measurement of N that exists here, since a shift read inside a
    static zone is the very thing that is ambiguous. The FIRM side's resolved shift anchors the
    pair: A firm -> after = before + N; B firm -> before = after - N; both refused -> A's own
    reading, + N.

    WHAT IS NOT CLAIMED: no sweep ran, so there are no walks -- `forward_walk_frames` and
    `backward_walk_frames` are None, logged as None, and the span reported is anchor to anchor
    where both exist. `ambiguous_shift_span` (per refused side) is logged beside them."""
    frame_ms = domain["frame_ms"]
    n_frames = _round_half_up(Fraction(str(hole["step_ms"])) / frame_ms)
    a_ambiguous, b_ambiguous = result.get("anchor_a_ambiguous"), result.get("anchor_b_ambiguous")
    if result.get("anchor_a_frame") is not None:
        before = result["before_shift_frames"]
        after = before + n_frames
    elif result.get("anchor_b_frame") is not None:
        after = result["after_shift_frames"]
        before = after - n_frames
    else:
        before = (a_ambiguous or {}).get("shift", result.get("before_shift_frames"))
        after = before + n_frames
    placed = dict(result, before_shift_frames=before, after_shift_frames=after)
    master_start, master_end, candidate_start, candidate_end = _pinned_frames(placed)
    pin, ambiguous_frames = _pin_point(placed)
    anchor_a = result.get("anchor_a_frame") or (a_ambiguous or {}).get("anchor")
    anchor_b = result.get("anchor_b_frame") or (b_ambiguous or {}).get("anchor")
    return {
        "modality": MODALITY, "status": HOLE_PINNED_TO_AMBIGUOUS_ZONE_END,
        "kind": hole["kind"], "why_token": hole["why_token"],
        "cause": "static_span_ambiguity", "pin_route": "ambiguous_anchor",
        "grid": f"{domain['master_rate'].numerator}/{domain['master_rate'].denominator}",
        "anchor_a_frame": anchor_a, "anchor_b_frame": anchor_b,
        "master_start_frame": master_start, "master_end_frame": master_end,
        "candidate_start_frame_equivalent": candidate_start,
        "candidate_end_frame_equivalent": candidate_end,
        "candidate_start_frame": _candidate_native_frame(candidate_start, domain),
        "candidate_end_frame": _candidate_native_frame(candidate_end, domain),
        "master_start_ms": _exact_ms_of_frame(master_start, domain),
        "master_end_ms": _exact_ms_of_frame(master_end, domain),
        "video_shift_ms_frame_quantised": [_exact_ms_of_frame(before, domain),
                                           _exact_ms_of_frame(after, domain)],
        **_audio_offsets(hole),
        "net_kind": ("addition" if n_frames > 0 else "deletion" if n_frames < 0
                     else "still_image"),
        "frames_to_cut": max(0, n_frames), "frames_to_fill": max(0, -n_frames),
        "before_shift_frames": before, "after_shift_frames": after,
        "nominal_before_shift_frames": result.get("nominal_before_shift_frames"),
        "nominal_after_shift_frames": result.get("nominal_after_shift_frames"),
        "audio_step_frames": n_frames,
        "forward_walk_frames": None, "backward_walk_frames": None,
        "span_frames": (None if anchor_a is None or anchor_b is None else anchor_b - anchor_a),
        "pin_frame": pin, "ambiguous_frames": ambiguous_frames,
        "anchor_a_ambiguous": a_ambiguous, "anchor_b_ambiguous": b_ambiguous,
        "evidence": result.get("evidence"),
    }


def _span_no_cut_outcome(hole, domain, master_obj, candidate_obj, low_ms, high_ms, result):
    """The no-cut test on a SUB-FLOOR hole's own frames, when no anchor pair could be seated
    (see `_resolve_interior`): `scene_anchor.island_match` over the hole's master bracket at its
    `offset_before`, the shift resolved within the quantum's reach. A majority of the frames
    matching one shift closes the hole (`no_cut_confirmed`, ADDENDUM 3), logged as a refuted
    proposal with the counts; anything else returns None and the hole declines as before."""
    import scene_anchor
    frame_ms = domain["frame_ms"]
    try:
        reading = scene_anchor.island_match(
            master_obj.filePath, candidate_obj.filePath,
            domain["master_rate"].numerator, domain["master_rate"].denominator,
            float(low_ms), float(high_ms), float(hole["offset_before_ms"]),
            candidate_time_scale=domain["time_scale"],
            shift_search_frames=_shift_search_frames(domain, hole["quantum_ms"]),
            scan_cache=hole.get("scan_cache"))
    except Exception as error:                                           # noqa: BLE001
        reading = {"verdict": "unreadable", "reason": f"island_match_raised:{type(error).__name__}"}
    step_result("hole_span_no_cut_test", candidate=candidate_obj.filePath,
                master_ms=[round(float(low_ms), 2), round(float(high_ms), 2)],
                audio_step_ms=round(hole["step_ms"], 3), proposal_reason=result.get("reason"),
                **{f"span_{key}": value for key, value in reading.items()})
    if reading["verdict"] != "same":
        return None
    shift = reading["shift_frames"]
    first, last = reading["master_frames"]
    outcome = {
        "modality": MODALITY, "status": HOLE_NO_CUT_CONFIRMED, "kind": hole["kind"],
        "why_token": hole["why_token"], "cause": None,
        "grid": f"{domain['master_rate'].numerator}/{domain['master_rate'].denominator}",
        "master_start_frame": last, "master_end_frame": last,
        "before_shift_frames": shift, "after_shift_frames": shift,
        **_audio_offsets(hole),
        "span_frames": last - first, "net_kind": "still_image",
        "evidence": f"span_no_cut matched={reading['matched']} readable={reading['readable']}",
        "refuted_proposal": {
            "audio_step_ms": round(hole["step_ms"], 3), "audio_step_points": hole["step_points"],
            "master_position_ms": [round(float(hole["master_ms"][0]), 2),
                                   round(float(hole["master_ms"][1]), 2)],
            "master_position_frames": [first, last],
            "proposal_reading": result.get("reason"),
            "video_probe": "span_no_cut_test", "video_single_shift": shift,
            "video_verdict": HOLE_NO_CUT_CONFIRMED,
            "proposal_shifts": [result.get("before_shift_frames"),
                                result.get("after_shift_frames")],
            "proposal_span_matches_before": None, "proposal_span_matches_after": None,
            "claimed_span_matches": None,
            "span_test_matches": [reading["matched"], reading["readable"]],
            "surviving_shift_walk_frames": [None, None],
            "surviving_shift_span_frames": last - first},
    }
    return outcome


def _resolve_interior(hole, domain, master_obj, candidate_obj):
    """research_exact_frame, TWO ANCHORS (ruling): scene detection +/-10 s both sides of the hole
    on master AND candidate, the nearest compatible pHash anchors outward from the hole, then the
    walks between them -- all of it `scene_anchor.locate_scene_anchors`, called with the hole's
    master bracket and the offsets of the two zones that bound it.

    THE AUDIO PROPOSES, THE VIDEO DISPOSES (ADDENDUM 3) -- AND FOR A PROPOSAL UNDER THE ALIGNER'S
    RESOLUTION FLOOR THE VIDEO IS ALWAYS ASKED THE NO-CUT QUESTION, not only when the proposal's
    own anchors fail. MEASURED on errid-202 hole 4: the audio proposes -124 ms (one quantum,
    under the two-quanta floor); the proposal's anchors DO validate, at shifts -96 and -98, on
    three frames each -- yet over the next 100 frames -96 matches 50/50 where -98 matches 7/50,
    and -96 holds from there to the tail anchor 4000 frames later. Three slow frames tolerated a
    two-frame error; the question "does ONE shift carry across this hole?" does not. So under
    the floor (step 0 included -- a zero step is still a one-quantum measurement):
      * the proposal's two shifts come back EQUAL -> its own verdict stands;
      * otherwise the video is offered a SINGLE shift on both sides, held EXACTLY (no +/-2
        search: the question is whether that shift is accepted, not which shift fits best) --
        A's resolved shift first, B's second; with no anchors resolved at all, each zone's
        audio offset with the +/-2 search, since nothing better is known. The first that seats
        BOTH anchors under one shift REFUTES the step: logged with its step, position and the
        video's verdict, and the hole takes that single-shift reading -- `no_cut_confirmed`
        when nothing differs between the fronts, `resolved` (an equal-length replacement,
        zero frames cut or filled) when a picture differs but the timeline does not.
      * none does -> the proposal's reading stands (or its decline).
    At or above the floor none of this runs by default: a step of two quanta or more is a
    measured edit -- UNLESS THE VIDEO ITSELF REFUTES IT (ADDENDUM 19 c, fix F3). When the
    proposal's walks left a span whose MAJORITY matches ONE side's shift and the re-fronted
    search could not place a cut (`_checked_two_anchor` declines with `claimed_by`), the video
    has said "this is that shift's content", whatever the step's size. MEASURED, id 134 (Mai-HiME
    12): the audio proposed +185 s; 271 of the 331 unmatched master frames matched the BEFORE
    shift. So the no-cut hypothesis is put to the video AT THAT SHIFT, held exactly, first; the
    rest proceeds as above.

    SINCE ADDENDUM 25 this resolver is asked about the AUDIO WALK's change points only (the
    walk measures the offsets at the millisecond, so no offset is carried from hole to hole any
    more): its answer places the cut inside the audio's interval, or says there is none (a slip).

    A PROPOSAL DECLINED ON AN AMBIGUOUS ANCHOR is the static-span verdict reached from the anchor
    side (`_interior_verdict`, case 3) and is pinned by `_ambiguous_pin_outcome` -- after the
    no-cut hypotheses, which would close the hole outright if one shift carried across it.
    """
    quantum_ms = hole["quantum_ms"]
    offset_before_ms = hole["offset_before_ms"]
    offset_after_ms = hole["offset_after_ms"]
    step_ms = hole["step_ms"]
    low_ms, high_ms = hole["master_ms"]
    # AN EMPTY MASTER SPAN IS A REAL SHAPE, NOT A DEGENERATE ONE: a pure insertion in the
    # candidate leaves no master content between the zones. The resolver refuses a bracket of
    # zero width (`empty_bracket`), so it is given ONE master frame -- the atomic resolution --
    # and the anchors still seed outward from it.
    frame_ms = domain["frame_ms"]
    if high_ms - low_ms < float(frame_ms):
        high_ms = low_ms + float(frame_ms)
    result = _checked_two_anchor(hole, domain, master_obj, candidate_obj, low_ms, high_ms,
                                 offset_before_ms, offset_after_ms, step_ms, quantum_ms,
                                 probe="proposal")
    below_floor = (hole["step_points"] is not None
                   and abs(hole["step_points"]) < banded_seed_alignment.RESOLUTION_FLOOR_QUANTA)
    proposal_single_shift = (not result["declined"]
                             and result["before_shift_frames"] == result["after_shift_frames"])
    hypotheses = []
    if result["declined"] and result.get("claimed_by") in ("before", "after"):
        # F3: the video's own majority reading, held exactly -- whatever the step's size.
        hypotheses.append((f"no_cut_at_claimed_{result['claimed_by']}_shift",
                           float(result["claimed_shift_frames"] * frame_ms), False))
        step_result("video_refutes_audio_step", candidate=candidate_obj.filePath,
                    audio_step_ms=round(step_ms, 3), claimed_by=result["claimed_by"],
                    claimed_shift_frames=result["claimed_shift_frames"],
                    claimed_span_master=result.get("claimed_span_master"),
                    claimed_matches=result.get("claimed_matches"),
                    next_probe="no_cut_hypothesis_at_the_claimed_shift")
    if below_floor and not proposal_single_shift:
        if result["declined"]:
            hypotheses += [("no_cut_at_offset_before", offset_before_ms, True),
                           ("no_cut_at_offset_after", offset_after_ms, True)]
        else:
            hypotheses += [
                ("no_cut_at_anchor_a_shift",
                 float(result["before_shift_frames"] * frame_ms), False),
                ("no_cut_at_anchor_b_shift",
                 float(result["after_shift_frames"] * frame_ms), False)]
    if hypotheses:
        for label, hypothesis, search in hypotheses:
            probe = _checked_two_anchor(hole, domain, master_obj, candidate_obj, low_ms,
                                        high_ms, hypothesis, hypothesis, 0.0, quantum_ms,
                                        probe=label, resolve_shift=search)
            if probe["declined"] or probe["before_shift_frames"] != probe["after_shift_frames"]:
                continue
            verdict = _interior_verdict(probe)
            refuted = {"audio_step_ms": round(step_ms, 3),
                       "audio_step_points": hole["step_points"],
                       "master_position_ms": [round(float(hole["master_ms"][0]), 2),
                                              round(float(hole["master_ms"][1]), 2)],
                       "master_position_frames": [
                           _master_frame_of_ms(hole["master_ms"][0], domain),
                           _master_frame_of_ms(hole["master_ms"][1], domain)],
                       "proposal_reading": (result.get("reason") if result["declined"] else
                                            f"shifts {result['before_shift_frames']}/"
                                            f"{result['after_shift_frames']} anchors "
                                            f"{result['anchor_a_frame']}/"
                                            f"{result['anchor_b_frame']}"),
                       "video_probe": label,
                       "video_single_shift": probe["before_shift_frames"],
                       "video_verdict": verdict,
                       # THE FRAME COUNTS UNDER EACH SHIFT (owner 2026-09-24): what the proposal's
                       # unmatched span matched under its two shifts, [matched, readable] --
                       # or, on a claimed span, the claiming side's count -- and how many frames
                       # the surviving single shift carried the walks across.
                       "proposal_shifts": ([result.get("before_shift_frames"),
                                            result.get("after_shift_frames")]
                                           if not result["declined"] else
                                           [result.get("claimed_by"),
                                            result.get("claimed_shift_frames")]),
                       "proposal_span_matches_before": result.get(
                           "unmatched_span_matches_before"),
                       "proposal_span_matches_after": result.get(
                           "unmatched_span_matches_after"),
                       "claimed_span_matches": result.get("claimed_matches"),
                       "surviving_shift_walk_frames": [probe.get("forward_walk_frames"),
                                                       probe.get("backward_walk_frames")],
                       "surviving_shift_span_frames": (probe["anchor_b_frame"]
                                                       - probe["anchor_a_frame"])}
            return _interior_outcome(hole, domain, probe, verdict, refuted_proposal=refuted)
    if result["declined"]:
        # AN AMBIGUOUS ANCHOR PINS ONLY A MEASURED EDIT. Under the aligner's resolution floor
        # the audio step is not a claim (b2 keeps such zones out of its cut list), so an N read
        # off it is noise -- MEASURED, Fallout S01E02: a +1-quantum step pinned a 3-frame
        # addition where the island right after it matched the BEFORE shift 60/60. Such a hole
        # is asked the no-cut question on its OWN frames instead (`scene_anchor.island_match`,
        # the island test: a static span matches under any shift, which is exactly why anchors
        # fail there and why the span itself can still answer).
        if below_floor:
            span_outcome = _span_no_cut_outcome(hole, domain, master_obj, candidate_obj,
                                                low_ms, high_ms, result)
            if span_outcome is not None:
                return span_outcome
        elif _interior_verdict(result) == HOLE_PINNED_TO_AMBIGUOUS_ZONE_END:
            return _ambiguous_pin_outcome(hole, domain, result)
        return _declined(hole, result.get("reason"), result.get("evidence"),
                         no_cut_probe_run=bool(hypotheses))
    return _interior_outcome(hole, domain, result, _interior_verdict(result))


def _resolve_edge(hole, domain, master_obj, candidate_obj):
    """research_exact_frame, ONE ANCHOR (ruling + EDGE_SINGLE_ANCHOR): scene detection on the
    common side of the hole, the nearest compatible anchor, then the bounded walk outward to the
    LAST COMMON FRAME, terminating in exactly one of the three named ways. All of it
    `scene_anchor.locate_edge_boundary`, geometry precondition included.

    THE BRACKET IS THE HOLE'S MASTER SPAN, CLAMPED TO ONE FRAME ON THE TIMELINE. A tail hole
    whose master span is empty (errid-70: the candidate carries 3.1 s the master does not) would
    otherwise start at or past the master's last frame; a head hole with no master content
    (errid-202's candidate starts on common content) would be zero wide. Either way the anchor
    seeds from the common side and the walk runs outward to a FILE fact, so one frame of bracket
    is all the resolver needs to know which side is common.
    """
    import scene_anchor
    edge = hole["kind"]
    quantum_ms = hole["quantum_ms"]
    frame_ms = float(domain["frame_ms"])
    timeline_ms = float(domain["master_timeline_ms"])
    if edge == "head":
        offset_ms = hole["offset_after_ms"]
        low_ms, high_ms = 0.0, max(float(hole["master_ms"][1]), frame_ms)
    else:
        offset_ms = hole["offset_before_ms"]
        low_ms = min(float(hole["master_ms"][0]), timeline_ms - frame_ms)
        high_ms = timeline_ms
    candidate_duration_ms = domain["candidate_equivalent_duration_ms"]
    step_launch("edge_walk", candidate=candidate_obj.filePath, edge=edge,
                bracket_ms=[round(low_ms, 2), round(high_ms, 2)],
                offset_ms=round(offset_ms, 3), master_timeline_ms=timeline_ms,
                candidate_equivalent_duration_ms=(None if candidate_duration_ms is None
                                                  else float(candidate_duration_ms)))
    try:
        result = scene_anchor.locate_edge_boundary(
            master_obj.filePath, candidate_obj.filePath,
            domain["master_rate"].numerator, domain["master_rate"].denominator,
            low_ms, high_ms, float(offset_ms), edge, timeline_ms,
            None if candidate_duration_ms is None else float(candidate_duration_ms),
            quantum_ms=float(quantum_ms), candidate_time_scale=domain["time_scale"],
            shift_search_frames=_shift_search_frames(domain, quantum_ms),
            deadline=hole.get("deadline"))
    except Exception as error:                                           # noqa: BLE001
        result = {"declined": True, "reason": f"resolver_raised:{type(error).__name__}",
                  "evidence": str(error)[:300]}
    step_result("edge_walk", candidate=candidate_obj.filePath, edge=edge,
                declined=result["declined"], reason=result.get("reason"),
                termination=result.get("termination"), anchor=result.get("anchor_frame"),
                shift=result.get("shift_frames"), nominal_shift=result.get("nominal_shift_frames"),
                boundary=result.get("boundary_frame"), walked=result.get("walked_frames"),
                mismatch_run=result.get("mismatch_run"),
                max_mismatch_run=result.get("max_mismatch_run"),
                addition_frames=result.get("addition_frames"),
                evidence=result.get("evidence"))
    if result["declined"]:
        return _declined(hole, result.get("reason"), result.get("evidence"))

    boundary = result["boundary_frame"]
    shift = result["shift_frames"]
    termination = result["termination"]
    master_last = result["master_last_frame"]
    if edge == "head":
        # [0, boundary) on the master is not common; the candidate's non-common head is
        # [0, boundary + shift) in equivalent frames.
        master_start, master_end = 0, boundary
        candidate_start_eq, candidate_end_eq = 0, boundary + shift
        master_non_common = boundary
        candidate_non_common = boundary + shift
    else:
        # (boundary, last] on the master is not common. The candidate's end is a FILE fact only
        # the walk's own termination can state: it is known exactly on `candidate_exhausted`
        # (the candidate stopped at the very step the master continued), and on the other two it
        # is "to the candidate's end" -- stage 5 reads that end, it is never guessed here.
        master_start, master_end = boundary + 1, master_last + 1
        candidate_start_eq = boundary + 1 + shift
        candidate_end_eq = (boundary + 1 + shift
                            if termination == EDGE_CANDIDATE_EXHAUSTED else None)
        master_non_common = master_last - boundary
        candidate_non_common = (0 if termination == EDGE_CANDIDATE_EXHAUSTED else None)

    # MASTER CONTENT PLACED ON THE OUTPUT WHERE THE CANDIDATE HAS NONE THAT IS COMMON: an
    # addition (termination 3) or a replacement (termination 1). A trim (termination 2) adds
    # nothing. Counted in frames on the exact grid -- the addendum's "no probe, no estimate".
    added_frames = 0 if termination == EDGE_MASTER_EXHAUSTED else max(0, master_non_common)
    added_seconds = float(Fraction(added_frames) / domain["master_rate"])

    status = termination
    # AN EDGE HOLE THAT CLOSES: the walk reached the file edge on BOTH files at the same step --
    # nothing to trim, nothing to add. The audio saw an edge the video says is not there
    # (typically a silent or music-only intro the fingerprint could not anchor).
    if master_non_common == 0 and candidate_non_common == 0:
        status = HOLE_NO_CUT_CONFIRMED
    return {
        "modality": MODALITY, "status": status, "kind": edge, "why_token": hole["why_token"],
        "cause": None, "termination": termination, "net_kind": result["net_kind"],
        "grid": f"{result['grid']['num']}/{result['grid']['den']}",
        "anchor_frame": result["anchor_frame"], "anchor_side": result["anchor_side"],
        "anchor_n_frames": result["anchor_n_frames"],
        "shift_frames": shift, "nominal_shift_frames": result["nominal_shift_frames"],
        "boundary_frame": boundary, "walked_frames": result["walked_frames"],
        "mismatch_run": result["mismatch_run"],
        "max_mismatch_run": result["max_mismatch_run"],
        "addition_frames": result["addition_frames"], "addition_ms": result["addition_ms"],
        "master_last_frame": master_last,
        "master_start_frame": master_start, "master_end_frame": master_end,
        "candidate_start_frame_equivalent": candidate_start_eq,
        "candidate_end_frame_equivalent": candidate_end_eq,
        "candidate_start_frame": _candidate_native_frame(candidate_start_eq, domain),
        "candidate_end_frame": _candidate_native_frame(candidate_end_eq, domain),
        "edge_addition_frames": added_frames,
        "edge_addition_seconds": added_seconds,
        "master_start_ms": _exact_ms_of_frame(master_start, domain),
        "master_end_ms": _exact_ms_of_frame(master_end, domain),
        "boundary_ms": _exact_ms_of_frame(boundary, domain),
        "video_shift_ms_frame_quantised": _exact_ms_of_frame(shift, domain),
        **_audio_offsets(hole),
        "geometry": (result.get("geometry") or {}).get("verdict"),
        "evidence": result.get("evidence"),
    }


def resolve_hole(hole, master_obj, candidate_obj, work_dir):
    """research_exact_frame for ONE hole -- the ruling's two functions, wired to the resolvers
    that already exist. Returns an outcome dict whose `status` is one of the closed vocabulary
    above (`HOLE_STATUSES_WITH_FRAMES`, or `declined` with a named `resolver_reason`).

      interior  -> `_resolve_interior`: `scene_anchor.locate_scene_anchors`, two anchors, with
                   the geometry precondition, per-anchor shift resolution and the pair's rate
                   relation switched on (all three opt-in keywords of that function; the live
                   chain passes none of them).
      head/tail -> `_resolve_edge`: `scene_anchor.locate_edge_boundary`, one anchor, the bounded
                   walk and its three named terminations.

    THE HOLE MUST CARRY ITS FRAME DOMAIN (`hole["frame_domain"]`, from `frame_domain`) and the
    couple's quantum (`hole["quantum_ms"]`): the hole's bounds are fingerprint points, and the
    ONLY conversion to milliseconds is points x that quantum -- the aligner's own
    `offset_ms = offset_points * quantum_ms` -- then milliseconds to frames on the exact rational
    grid. `work_dir` is unused today: the resolvers decode through pipes and write nothing; it
    stays in the signature because stage 7's parallel pool will need a per-hole scratch path.

    A resolver that cannot establish a boundary returns a NAMED refusal. Returning a plausible
    boundary it did not measure would be the one failure this whole campaign exists to prevent.
    """
    domain = hole.get("frame_domain")
    if domain is None:
        return _declined(hole, "frame_domain_absent")
    # THE BUDGETS (ADDENDUM 26.3): this hole gets HOLE_BUDGET_S, never past the repair's own end.
    repair_deadline = domain.get("repair_deadline")
    if repair_deadline is not None and time.monotonic() > repair_deadline:
        return _declined(hole, "repair_budget_exceeded")
    deadline = time.monotonic() + HOLE_BUDGET_S
    hole = dict(hole, deadline=(deadline if repair_deadline is None
                                else min(deadline, repair_deadline)))
    if hole["kind"] == "interior":
        outcome = _resolve_interior(hole, domain, master_obj, candidate_obj)
    elif hole["kind"] in ("head", "tail"):
        outcome = _resolve_edge(hole, domain, master_obj, candidate_obj)
    else:
        outcome = _declined(hole, f"no_resolver_for_kind:{hole['kind']}")
    # A HOLE THAT TOOK LONGER THAN ITS BUDGET DECLINES BY THAT NAME even when its last call came
    # back with an answer (ADDENDUM 26.3): the bound is on the hole's time, and a rung that ran
    # past it can only be stopped by the decoders' own timeouts, not by the check between rungs.
    if outcome["status"] != HOLE_DECLINED and time.monotonic() > hole["deadline"]:
        outcome = _declined(hole, "hole_budget_exceeded",
                            evidence=f"answered {outcome['status']} past the "
                                     f"{HOLE_BUDGET_S} s budget")
    # THE AUDIO'S BRACKET, IN FRAMES, BESIDE THE VIDEO'S ANSWER -- so a reader can see where the
    # audio proposed and where the video disposed without redoing the conversion.
    outcome["audio_master_frames"] = [_master_frame_of_ms(hole["master_ms"][0], domain),
                                      _master_frame_of_ms(hole["master_ms"][1], domain)]
    outcome["audio_candidate_frames"] = [
        _candidate_native_frame_of_ms(hole["candidate_ms"][0], domain),
        _candidate_native_frame_of_ms(hole["candidate_ms"][1], domain)]
    outcome["audio_step_ms"] = (None if hole["step_ms"] is None else round(hole["step_ms"], 3))
    return outcome


def _resolve_logged(candidate_path, index, hole, domain, master_obj, candidate_obj, work_dir):
    """`resolve_hole` for hole `index`, with its launch and result lines, the vocabulary check,
    and the two records the addenda require beside a resolution (ADDENDUM 3 point 3's refuted
    proposal; ADDENDUM 4 point 3's walks and span for a pinned boundary)."""
    step_launch("resolve_hole", candidate=candidate_path, hole=index, kind=hole["kind"],
                why=hole["why_token"], master_span_s=round(hole["master_span_seconds"], 3),
                candidate_span_s=round(hole["candidate_span_seconds"], 3),
                step_ms=(round(hole["step_ms"], 1) if hole["step_ms"] is not None else None),
                offsets_ms=[None if hole.get("offset_before_ms") is None
                            else round(hole["offset_before_ms"], 3),
                            None if hole.get("offset_after_ms") is None
                            else round(hole["offset_after_ms"], 3)],
                union_of=hole.get("union_of"), cluster=hole.get("cluster_id"),
                origin=hole.get("origin", "alignment"))
    started = time.time()
    outcome = resolve_hole(dict(hole, frame_domain=domain), master_obj, candidate_obj, work_dir)
    # A STATUS FROM OUTSIDE THE CLOSED VOCABULARY IS A RESOLVER BUG, AND IT IS NOT ALLOWED TO
    # PASS AS A RESOLUTION: it becomes a named decline, loudly, rather than a hole the plan would
    # read frames off.
    if outcome["status"] not in HOLE_STATUSES_WITH_FRAMES + (HOLE_DECLINED,):
        tools.log_always(f"repair: orchestrator UNVOCABULARISED hole status="
                         f"{outcome['status']} hole={index} for {candidate_path} -- "
                         f"treated as declined\n")
        outcome = dict(outcome, status=HOLE_DECLINED, cause="hole_resolution_declined",
                       resolver_reason=f"unvocabularised_status:{outcome['status']}")
    step_result("resolve_hole", candidate=candidate_path, hole=index, kind=hole["kind"],
                status=outcome["status"], cause=outcome.get("cause"),
                resolver_reason=outcome.get("resolver_reason"),
                termination=outcome.get("termination"),
                master_frames=[outcome.get("master_start_frame"),
                               outcome.get("master_end_frame")],
                master_ms=[outcome.get("master_start_ms"), outcome.get("master_end_ms")],
                audio_offsets_ms=[outcome.get("audio_offset_before_ms"),
                                  outcome.get("audio_offset_after_ms")],
                candidate_frames=[outcome.get("candidate_start_frame"),
                                  outcome.get("candidate_end_frame")],
                candidate_frames_equivalent=[
                    outcome.get("candidate_start_frame_equivalent"),
                    outcome.get("candidate_end_frame_equivalent")],
                audio_master_frames=outcome.get("audio_master_frames"),
                audio_candidate_frames=outcome.get("audio_candidate_frames"),
                audio_step_ms=outcome.get("audio_step_ms"),
                anchors=[outcome.get("anchor_a_frame", outcome.get("anchor_frame")),
                         outcome.get("anchor_b_frame")],
                shifts=[outcome.get("before_shift_frames", outcome.get("shift_frames")),
                        outcome.get("after_shift_frames")],
                walks=[outcome.get("forward_walk_frames", outcome.get("walked_frames")),
                       outcome.get("backward_walk_frames")],
                span_frames=outcome.get("span_frames"),
                net_kind=outcome.get("net_kind"),
                edge_addition_frames=outcome.get("edge_addition_frames"),
                seconds=round(time.time() - started, 2))
    # ADDENDUM 3, POINT 3: "un candidat audio sous le plancher que la video refute est logge
    # (candidat, position, verdict video) puis ferme" -- AND IT IS A DECISION ON THE MATERIAL, SO
    # IT IS UNCONDITIONAL (owner ruling 2026-09-24): one line per refuted proposal with the master
    # position, the proposed step (ms / points), the surviving shift and the frame counts under
    # each shift; the same for every no-cut closure. The full record stays on the dev channel.
    if outcome.get("refuted_proposal"):
        refuted = outcome["refuted_proposal"]
        step_result("refuted_proposal", candidate=candidate_path, hole=index, **refuted)
        tools.log_always(
            f"repair: refuted_proposal hole={index} "
            f"master_frames={refuted['master_position_frames']} "
            f"master_ms={refuted['master_position_ms']} "
            f"proposed_step_ms={refuted['audio_step_ms']} "
            f"proposed_step_points={refuted['audio_step_points']} "
            f"proposal_shifts={refuted['proposal_shifts']} "
            f"proposal_span_matches_before={refuted['proposal_span_matches_before']} "
            f"proposal_span_matches_after={refuted['proposal_span_matches_after']} "
            f"claimed_span_matches={refuted['claimed_span_matches']} "
            f"span_test_matches={refuted.get('span_test_matches')} "
            f"surviving_shift={refuted['video_single_shift']} "
            f"surviving_shift_walk_frames={refuted['surviving_shift_walk_frames']} "
            f"surviving_shift_span_frames={refuted['surviving_shift_span_frames']} "
            f"video_verdict={refuted['video_verdict']} probe={refuted['video_probe']} "
            f"for {candidate_path}\n")
    if outcome["status"] == HOLE_NO_CUT_CONFIRMED:
        tools.log_always(
            f"repair: no_cut_confirmed hole={index} kind={hole['kind']} "
            f"origin={hole.get('origin', 'alignment')} "
            f"audio_master_frames={outcome.get('audio_master_frames')} "
            f"audio_step_ms={outcome.get('audio_step_ms')} "
            f"shift={outcome.get('before_shift_frames', outcome.get('shift_frames'))} "
            f"anchors={[outcome.get('anchor_a_frame', outcome.get('anchor_frame')), outcome.get('anchor_b_frame')]} "
            f"walks={[outcome.get('forward_walk_frames', outcome.get('walked_frames')), outcome.get('backward_walk_frames')]} "
            f"for {candidate_path}\n")
    # ADDENDUM 4, POINT 3: "jamais un placement silencieux" -- both walks and the span.
    if outcome["status"] == HOLE_PINNED_TO_AMBIGUOUS_ZONE_END:
        step_result("boundary_pinned_to_ambiguous_zone_end", candidate=candidate_path,
                    hole=index, cause="static_span_ambiguity",
                    pin_route=outcome.get("pin_route", "sweep_fronts"),
                    forward_walk_frames=outcome["forward_walk_frames"],
                    backward_walk_frames=outcome["backward_walk_frames"],
                    span_frames=outcome["span_frames"],
                    ambiguous_frames=outcome["ambiguous_frames"],
                    pin_frame=outcome["pin_frame"],
                    anchor_b_frame=outcome["anchor_b_frame"],
                    audio_step_frames=outcome.get("audio_step_frames"),
                    ambiguous_shift_span_a=(outcome.get("anchor_a_ambiguous") or {}).get(
                        "ambiguous_shift_span"),
                    ambiguous_shift_span_b=(outcome.get("anchor_b_ambiguous") or {}).get(
                        "ambiguous_shift_span"),
                    fill_master_frames=[outcome["master_start_frame"],
                                        outcome["master_end_frame"]])
    return outcome



# ---------------------------------------------------------------------------
# STEP 4 (ADDENDUM 25) -- THE AUDIO BOUNDS, THE VIDEO PINS
# ---------------------------------------------------------------------------

def _audio_entry(video_obj, language, stream):
    for entry in (getattr(video_obj, "audios", None) or {}).get(language) or []:
        if str(entry.get("StreamOrder")) == str(stream):
            return entry
    return None


def hole_sanity(holes, domain, candidate_path):
    """ADDENDUM 26.2: a hole is checked BEFORE any scan. Returns None, or `(cause, reason)` for
    the first hole that cannot be real:
      hole_outside_master_timeline   it starts at or past the end of the master's VIDEO (the plan's
                                     timeline), or, interior, ends more than a frame past it
      hole_step_exceeds_duration     |step| longer than the shorter of the two files -- no edit
                                     removes or adds more than a whole file
      interior_hole_exceeds_budget   an interior hole wider than INTERIOR_HOLE_MAX_SPAN_S
    MEASURED, id 691: an interior hole of 6,803 s, step -6,801,847 ms, past the 5,997 s video --
    scanned for hours (memo THROUGHPUT_ANALYSIS_20260925)."""
    timeline = float(domain["master_timeline_ms"])
    frame = float(domain["frame_ms"])
    candidate = domain.get("candidate_equivalent_duration_ms")
    shorter = timeline if candidate is None else min(timeline, float(candidate))
    for index, hole in enumerate(holes):
        low, high = hole["master_ms"]
        verdict = None
        if low >= timeline or (hole["kind"] == "interior" and high > timeline + frame):
            verdict = ("hole_outside_master_timeline",
                       f"hole {index} ({hole['kind']}) spans master [{round(low, 1)}, "
                       f"{round(high, 1)}] ms beyond the master video's {round(timeline, 1)} ms")
        elif hole["step_ms"] is not None and abs(hole["step_ms"]) > shorter:
            verdict = ("hole_step_exceeds_duration",
                       f"hole {index} carries a step of {round(hole['step_ms'], 1)} ms, longer "
                       f"than the shorter file ({round(shorter, 1)} ms)")
        elif hole["kind"] == "interior" and (high - low) / 1000.0 > INTERIOR_HOLE_MAX_SPAN_S:
            verdict = ("interior_hole_exceeds_budget",
                       f"interior hole {index} spans {round((high - low) / 1000.0, 1)} s of the "
                       f"master, over the {INTERIOR_HOLE_MAX_SPAN_S} s bound")
        if verdict is not None:
            step_result("hole_sanity", candidate=candidate_path, hole=index, kind=hole["kind"],
                        master_ms=[round(low, 2), round(high, 2)], step_ms=hole["step_ms"],
                        cause=verdict[0])
            return verdict
    return None


# The resolver reasons that are a TIME BOUND, not a reading (ADDENDUM 26.3).
BUDGET_REASONS = ("repair_budget_exceeded", "hole_budget_exceeded", "decoder_timeout")


def budget_cause(outcome, domain):
    """The time bound that stopped this video call, if one did (ADDENDUM 26.3): the repair's own
    budget first (everything after it declines on it), then the hole's, then a decoder's. With
    ADDENDUM 25 the audio places a transition the video could not snap -- but a video stopped by
    a CLOCK is the job running out of time, and 26.3 declines the file on it by name."""
    reason = outcome.get("resolver_reason")
    if reason not in BUDGET_REASONS:
        return None
    deadline = domain.get("repair_deadline")
    if reason == "repair_budget_exceeded" or (deadline is not None
                                              and time.monotonic() > deadline):
        return "repair_budget_exceeded"
    return reason


def log_partial_plan(candidate_path, cause, placed):
    """ADDENDUM 26.3: "un budget depasse decline le fichier avec le plan partiel logge" -- what
    was placed before the bound, unconditional (the file's next wave reads it). `placed`: one
    (what, decision, where) per transition or edge."""
    tools.log_always(f"repair: partial_plan cause={cause} placed={placed} for {candidate_path}\n")


def repair_budget_seconds(master_obj):
    """The repair's budget for this master (ADDENDUM 26.8) and the video duration that set it:
    REPAIR_BUDGET_PER_SLICE_S per STARTED REPAIR_BUDGET_SLICE_S of the master's video, never
    below one slice's worth. An unreadable duration (None) gets the floor."""
    video_ms = _video_duration_ms(master_obj)
    video_s = None if video_ms is None else float(video_ms) / 1000.0
    slices = max(1, math.ceil((video_s or 0.0) / REPAIR_BUDGET_SLICE_S))
    return slices * REPAIR_BUDGET_PER_SLICE_S, video_s


def _budget_terminal(candidate_path, step, budget_s):
    """The repair's budget ran out between two steps (ADDENDUM 26.3)."""
    _plan_line("none", candidate_path, step=step, cause="repair_budget_exceeded")
    return _terminal(candidate_path, "no_plan", "repair_budget_exceeded",
                     f"the repair's {budget_s} s budget ran out after the {step} step -- a "
                     f"statement about this run's cost; the file comes back next wave")


def _speed_chain(audio, speed_ratio, engine="asetrate"):
    """A candidate track's speed filter at a confirmed ratio (Decimal) and engine, built by the
    assembly's own `build_transform_chain` at the track's own rate -- or None at 1 (ADDENDUM 6)."""
    if speed_ratio is None:
        return None
    import merge_video_resample
    rate = (audio.get("ffprobe") or {}).get("sample_rate") or audio.get("SamplingRate")
    return merge_video_resample.build_transform_chain(int(float(rate)), speed_ratio, engine)[0]


def reference_walk(reference, holes, master_obj, candidate_obj, language, speed_ratio,
                   candidate_path, deadline=None, engine="asetrate"):
    """THE MILLISECOND WALK ON THE REFERENCE COUPLE (ADDENDUM 25.2): the comparison track of
    the master against the candidate's, whole, on the file clock, seeded by every b2 offset
    the reference couple's zones and the union's holes carry. Returns `(walk, None)` or
    `(None, reason)`; the walk keeps the two decoded tracks for the edges and the plan. Its
    whole-track reads are bounded by the repair's budget (`deadline`); a read the budget stops
    is re-raised (`chimeric_error`, cause `repair_budget_exceeded`) for the caller to decline
    by that name."""
    import audio_walk
    master_stream, candidate_stream = reference["couple"].split("x")
    master_audio = _audio_entry(master_obj, language, master_stream)
    candidate_audio = _audio_entry(candidate_obj, language, candidate_stream)
    alignment, fold = reference["alignment"], reference["fold"]
    _zones, detail = coalesce_same_offset_zones(alignment.get("zones") or [],
                                                alignment.get("zones_detail") or [])
    seeds = sorted({round(zone["offset_points"] * alignment["quantum_ms"] + fold["delta_ms"], 3)
                    for zone in detail}
                   | {round(value, 3) for hole in holes
                      for value in (hole["offset_before_ms"], hole["offset_after_ms"])
                      if value is not None})
    step_launch("audio_walk", candidate=candidate_path, couple=reference["couple"],
                seeds=seeds, window_s=audio_walk.WALK_WINDOW_S, hop_s=audio_walk.WALK_HOP_S,
                search_ms=audio_walk.WALK_SEARCH_MS)
    started = time.time()
    if master_audio is None or candidate_audio is None:
        return None, f"reference couple {reference['couple']} has no {language} audio entry"
    try:
        scale = speed_ratio if speed_ratio is not None else Decimal(1)
        # An atempo pair is walked on both tracks' speech envelopes (ADDENDUM 30).
        envelope = engine == "atempo" and speed_ratio is not None
        master = audio_walk.read_on_file_clock(master_obj, master_audio, deadline=deadline,
                                               envelope=envelope)
        candidate = audio_walk.read_on_file_clock(
            candidate_obj, candidate_audio, _speed_chain(candidate_audio, speed_ratio, engine),
            scale, deadline=deadline, envelope=envelope)
    except Exception as error:                                           # noqa: BLE001
        if getattr(error, "cause", None) == "repair_budget_exceeded":
            raise
        return None, f"the comparison tracks could not be read ({type(error).__name__}: {error})"
    rows = audio_walk.walk(master, candidate, seeds)
    found, outliers = audio_walk.levels(rows)
    points = audio_walk.change_points(master, candidate, found)
    seconds = time.time() - started
    audio_walk.log_walk(candidate_path, rows, found, points, seconds)
    step_result("audio_walk", candidate=candidate_path, n_windows=len(rows),
                n_levels=len(found), n_outliers=len(outliers),
                levels=[(lv["t_first"], lv["t_last"], lv["off_ms"], lv["mad_ms"]) for lv in found],
                change_points=[(p["a_ms"], p["b_ms"], p["jump_ms"], p["kind"],
                                (p.get("edges") or {}).get("interval")) for p in points],
                seconds=round(seconds, 1))
    if not found:
        return None, "the walk measured no level: no window of the comparison tracks matched"
    return {"master": master, "candidate": candidate, "rows": rows, "levels": found,
            "points": points, "seeds": seeds, "master_audio": master_audio,
            "candidate_audio": candidate_audio, "master_stream": master_stream,
            "candidate_stream": candidate_stream,
            "master_audio_end_s": len(master) / audio_walk.WALK_RATE}, None


def _frame_s(frame, domain):
    return float(Fraction(frame) / domain["master_rate"])


def audio_edges(walk, holes, domain, master_obj, candidate_obj, work_dir, candidate_path):
    """HEAD AND TAIL, PLACED BY THE AUDIO (ADDENDUM 25: "l'audio borne, la video epingle").

    The head fill ends where the first level's content starts in the waveform, the tail fill
    starts where the last level's ends (`audio_walk.single_edge`, 20 ms, from inside the level
    outward, stopping at a sustained mismatch). The union's head/tail hole, when there is one,
    is still put to the video (`_resolve_edge`); its boundary REPLACES the audio edge only when
    it falls within one frame of it (the frame containing the audio instant) -- a video boundary
    elsewhere is a picture fact and loses, logged. MEASURED, Bleach S17E25: the video tail
    boundary sits at 1403.9 s, the candidate's audio keeps matching the master to ~1426 s.

    Returns `(head_end_s or None, tail_start_s or None, refusal or None)`; None = no fill on that edge (the edge
    is within one frame of the file's own start / the master timeline's end). An edge the
    20 ms / 100 ms profiles cannot read falls back to the level's measured window."""
    import audio_walk
    master, candidate = walk["master"], walk["candidate"]
    first, last = walk["levels"][0], walk["levels"][-1]
    frame = float(domain["frame_ms"]) / 1000.0
    reach = audio_walk.WALK_WINDOW_S + audio_walk.WALK_HOP_S + HOLE_MERGE_WINDOW_SECONDS
    # Anchored at the level's OWN measured window (the head's first window's end, the tail's
    # last window's start) and walked outward: a window mid-point can fall in silence past the
    # content it matched (errid-202's tail: silent from 1418.4 s, mid-point 1419.0 s).
    window = audio_walk.WALK_WINDOW_S
    head = audio_walk.single_edge(master, candidate, first["off_ms"], first["t_first"] + window,
                                  max(0.0, first["t_first"] - reach), "head")
    tail = audio_walk.single_edge(master, candidate, last["off_ms"], last["t_last"],
                                  min(walk["master_audio_end_s"], last["t_last"] + reach), "tail")
    # NO FINE EDGE READ (a quiet or re-mixed stretch on a rate pair, where neither the 20 ms
    # residual nor the 100 ms NCC clears its threshold): the level's own measured window is the
    # proven extent of common content -- the head starts no later than the first window, the
    # tail ends no earlier than the last one's end. MEASURED, Fallout S01E02 (1001/1000): no
    # fine edge at either end, level measured over [1.0, 3747.0] s.
    head_s = first["t_first"] if head is None else head["edge_s"]
    tail_s = (last["t_last"] + window) if tail is None else tail["edge_s"]
    edge_source = {"head": "walk_window_edge" if head is None else "audio_edge",
                   "tail": "walk_window_edge" if tail is None else "audio_edge"}
    by_kind = {hole["kind"]: (index, hole) for index, hole in enumerate(holes)
               if hole["kind"] in ("head", "tail")}
    decisions = {}
    for kind, audio_s, level in (("head", head_s, first), ("tail", tail_s, last)):
        video_s = None
        if kind in by_kind:
            index, hole = by_kind[kind]
            offsets = ({"offset_after_ms": level["off_ms"]} if kind == "head"
                       else {"offset_before_ms": level["off_ms"]})
            outcome = _resolve_logged(candidate_path, index, dict(hole, **offsets), domain,
                                      master_obj, candidate_obj, work_dir)
            budget = budget_cause(outcome, domain)
            if budget is not None:
                log_partial_plan(candidate_path, budget,
                                 [(edge, "placed", value) for edge, value in decisions.items()]
                                 + [(kind, "stopped", audio_s)])
                return None, None, (budget, f"the video on the {kind} edge stopped on a time "
                                            f"bound ({outcome.get('evidence')}) -- the partial "
                                            f"plan is logged; the file comes back next wave")
            if outcome["status"] in EDGE_TERMINATIONS:
                video_s = _frame_s(outcome["master_end_frame"] if kind == "head"
                                   else outcome["master_start_frame"], domain)
        placed = audio_s
        decision = edge_source[kind]
        if video_s is not None and abs(video_s - audio_s) <= frame:
            placed, decision = video_s, "video_frame_at_audio_edge"
        elif video_s is not None:
            decision = f"{edge_source[kind]}_video_boundary_elsewhere"
        decisions[kind] = placed
        tools.log_always(f"repair: audio_edge kind={kind} level_offset_ms={level['off_ms']} "
                         f"audio_edge_s={audio_s} video_boundary_s="
                         f"{None if video_s is None else round(video_s, 4)} placed_s={placed} "
                         f"decision={decision} for {candidate_path}\n")
    head_end = decisions["head"]
    if head_end is not None and head_end <= frame:
        head_end = None
    # THE TAIL FILL EXISTS WHENEVER THE AUDIO EDGE IS SHORT OF THE MASTER'S TIMELINE, even when
    # it coincides with the end of the master's own AUDIO: past the edge nothing proves the
    # candidate's content is common, so it is never read there -- the master fill writes what
    # audio the master has (nothing, past its audio end; ADDENDUM 25.6 counts only what is
    # written). MEASURED, id 134 (Mai-HiME 12): edge 1437.685 s, master audio ends 1437.69 s,
    # master video 1516.1 s -- reading the candidate to the timeline's end delivered ~78 s of
    # the candidate's own extra content and the verifier found no master audio to compare it to.
    tail_start = decisions["tail"]
    if tail_start is not None and tail_start >= float(domain["master_timeline_ms"]) / 1000.0 - frame:
        tail_start = None
    return head_end, tail_start, None


def _cp_hole(point, reference, index):
    """A walk change point as an interior hole for the video (ADDENDUM 25.2: "soumis a la
    video comme tout trou"): bracket = the audio's own edges, offsets = the two levels."""
    edges = point["edges"]
    quantum_ms = reference["alignment"]["quantum_ms"]
    low = min(edges["edge_A"], edges["edge_B"]) * 1000.0
    high = max(edges["edge_A"], edges["edge_B"]) * 1000.0
    a, b = point["a_ms"], point["b_ms"]
    return {"modality": MODALITY, "kind": "interior", "why_token": WHY_TOKEN["interior"],
            "origin": "audio_walk", "touches_head": False, "touches_tail": False,
            "master_ms": [low, high], "candidate_ms": [low + a, high + b],
            "master_span_seconds": (high - low) / 1000.0,
            "candidate_span_seconds": max(0.0, (high + b - low - a) / 1000.0),
            "offset_before_ms": a, "offset_after_ms": b, "offset_before_points": None,
            "offset_after_points": None, "step_ms": b - a,
            "step_points": _round_half_up(Fraction(str(b - a)) / Fraction(str(quantum_ms))),
            "quantum_ms": quantum_ms, "track_delay_delta_ms": reference["fold"]["delta_ms"],
            "offset_sources": ["audio_walk", "audio_walk"], "union_of": 1,
            "members": [{"couple": reference["couple"], "kind": "audio_walk_change_point",
                         "master_ms": [round(low, 2), round(high, 2)],
                         "step_ms": round(b - a, 3), "change_point": index}]}


def audio_transitions(walk, reference, domain, master_obj, candidate_obj, work_dir,
                      candidate_path):
    """EVERY WALK CHANGE POINT, PLACED (ADDENDUM 25.1-25.2). Returns `(transitions, None)` or
    `(None, (cause, reason))`.

    For each change point a -> b the AUDIO fixes the step (b - a, at the millisecond), the fill
    width (`extra` = max(0, a - b): master content the candidate lacks, never a frame count read
    off an anchor) and the interval the cut may lie in (fill start for a deletion, cut instant
    for an addition). An empty interval is `hole_width_contradicts_audio_step`. The video is
    then asked, as for any hole, and only CHOOSES INSIDE the interval:
      a cut at a frame inside the interval (+/- one frame)  -> the cut is at that frame
      no cut, and the step is under one quantum            -> a SLIP: the offset changes at the
                                                              interval's quietest instant, no
                                                              fill (25.2 b), `slip_applied`
      no cut on a step of a quantum or more, a cut outside -> the audio places it at the
      the interval, or a width the audio step contradicts    interval's quietest instant
                                                              (25.1: where the video is blind
                                                              the waveform decides), logged
      the video could not say, on a sub-quantum step       -> `sub_quantum_step_video_ambiguous`
                                                              (25.2 c)
    One unconditional line per change point: it is a decision on the material."""
    import audio_walk
    points = [p for p in walk["points"] if p["kind"] == "change_point"]
    holes, frame = [], float(domain["frame_ms"]) / 1000.0
    for index, point in enumerate(points):
        edges = point.get("edges") or {}
        if edges.get("status") != "ok":
            return None, ("audio_step_unlocalised",
                          f"the walk measured a {point['jump_ms']} ms step between levels "
                          f"{point['a_ms']} and {point['b_ms']} ms (t {point['level_before']['t_last']}"
                          f" -> {point['level_after']['t_first']} s) but its edges could not be "
                          f"read at 20 ms or 100 ms")
        if not edges["feasible"]:
            return None, ("hole_width_contradicts_audio_step",
                          f"the {round(edges['extra_s'] * 1000, 3)} ms of master content the "
                          f"candidate lacks at {edges['edge_A']}-{edges['edge_B']} s does not fit "
                          f"between the audio edges around its audible master-only sound "
                          f"(interval {edges['interval']})")
        holes.append(_cp_hole(point, reference, index))
    # CLUSTERS (owner 2026-09-24): nearby change points share one scene pass; each keeps its
    # bounds; the ISLAND between two of them is a walk level, confirmed by the walk's own
    # windows (the video no longer decides anything about an island -- 25.2 corollary).
    import audio_walk
    for cluster in cluster_holes(holes, [(reference["couple"], reference["alignment"],
                                          reference["fold"])]):
        islands = []
        for island in cluster["islands"]:
            agree, measured, windows = walk_agreement(
                walk, island["master_ms"][0], island["master_ms"][1],
                island["offset_ms"], audio_walk.LEVEL_TOLERANCE_MS)
            islands.append({"master_ms": [round(island["master_ms"][0], 2),
                                          round(island["master_ms"][1], 2)],
                            "offset_ms": round(island["offset_ms"], 3),
                            "b2_zones": island["b2_zones"],
                            "walk": {"agree": agree, "ok": measured, "windows": windows}})
        step_result("cluster", candidate=candidate_path, cluster=cluster["cluster_id"],
                    members=cluster["members"], shared_scan=len(cluster["members"]) > 1,
                    islands=islands)
    transitions = []
    for index, (point, hole) in enumerate(zip(points, holes)):
        edges = point["edges"]
        lo, hi = edges["interval"]
        extra = edges["extra_s"]
        jump = point["b_ms"] - point["a_ms"]
        sub_quantum = abs(jump) < hole["quantum_ms"]
        outcome = _resolve_logged(candidate_path, f"change_point_{index}", hole, domain,
                                  master_obj, candidate_obj, work_dir)
        budget = budget_cause(outcome, domain)
        if budget is not None:
            log_partial_plan(candidate_path, budget,
                             [(f"change_point_{t['change_point']}", t["decision"], t["at_s"],
                               t["fill_s"]) for t in transitions]
                             + [(f"change_point_{index}", "stopped", lo, hi)])
            return None, (budget, f"the video on change point {index} ({lo}-{hi} s) stopped on "
                                  f"a time bound ({outcome.get('evidence')}) -- the partial plan "
                                  f"is logged; the file comes back next wave")
        status = outcome["status"]
        video_s, width_note = None, None
        if status in (HOLE_RESOLVED, HOLE_PINNED_TO_AMBIGUOUS_ZONE_END):
            video_s = _frame_s(outcome["master_start_frame"], domain)
            video_fill_ms = (outcome["master_end_frame"] - outcome["master_start_frame"]) \
                * float(domain["frame_ms"])
            if abs(video_fill_ms - extra * 1000.0) > hole["quantum_ms"] + 2 * float(domain["frame_ms"]):
                width_note = (f"video fill {round(video_fill_ms, 3)} ms vs audio "
                              f"{round(extra * 1000.0, 3)} ms")
                video_s = None
        fill = extra
        if video_s is not None and lo - frame <= video_s <= hi + frame:
            at, decision = min(max(video_s, lo), hi), "video_frame_inside_audio_interval"
        elif status == HOLE_NO_CUT_CONFIRMED and sub_quantum:
            at = audio_walk.quietest_instant(walk["master"], lo, hi)
            fill, decision = 0.0, "slip_applied"
        elif status == HOLE_DECLINED and sub_quantum:
            return None, ("sub_quantum_step_video_ambiguous",
                          f"the walk measured a {round(jump, 3)} ms step (under one quantum) in "
                          f"[{lo}, {hi}] s and the video could neither place a cut nor confirm "
                          f"there is none ({outcome.get('resolver_reason')})")
        else:
            at = audio_walk.quietest_instant(walk["master"], lo, hi, extra)
            decision = ("audio_instant_video_width_contradicts" if width_note
                        else "audio_instant_video_no_cut" if status == HOLE_NO_CUT_CONFIRMED
                        else "audio_instant_video_declined" if status == HOLE_DECLINED
                        else "audio_instant_video_outside_interval")
        transitions.append({"at_s": at, "fill_s": fill, "a_ms": point["a_ms"],
                            "b_ms": point["b_ms"], "decision": decision, "interval": [lo, hi],
                            "edges": [edges["edge_A"], edges["edge_B"]],
                            "video_status": status, "video_s": video_s, "change_point": index})
        tools.log_always(
            f"repair: {'slip_applied' if decision == 'slip_applied' else 'audio_transition'} "
            f"change_point={index} a_ms={point['a_ms']} b_ms={point['b_ms']} "
            f"step_ms={round(jump, 3)} at_s={at} fill_ms={round(fill * 1000.0, 3)} "
            f"interval_s=[{lo}, {hi}] audio_edges_s=[{edges['edge_A']}, {edges['edge_B']}] "
            f"video_status={status} video_s={None if video_s is None else round(video_s, 4)} "
            f"decision={decision}{' ' + width_note.replace(' ', '_') if width_note else ''} "
            f"for {candidate_path}\n")
    return transitions, None


def log_holes_against_walk(holes, walk, candidate_path):
    """ADDENDUM 25.2 COROLLARY, logged per b2 hole: an interior hole of the union with no walk
    change point in its reach carries no audio step -- whatever the picture does there, it
    creates NO fill (`picture_only`, unconditional: a decision on the material). A change point
    no b2 hole reaches is a sub-quantum step the aligner could not see (logged)."""
    import audio_walk
    regions = [(p["level_before"]["t_last"] * 1000.0,
                (p["level_after"]["t_first"] + audio_walk.WALK_WINDOW_S) * 1000.0, p)
               for p in walk["points"] if p["kind"] == "change_point"]
    reach = HOLE_MERGE_WINDOW_SECONDS * 1000.0
    seen = set()
    for index, hole in enumerate(holes):
        if hole["kind"] != "interior":
            continue
        hits = [id(p) for low, high, p in regions
                if low < hole["master_ms"][1] + reach and high > hole["master_ms"][0] - reach]
        seen.update(hits)
        if not hits:
            tools.log_always(
                f"repair: picture_only hole={index} master_ms=[{round(hole['master_ms'][0], 2)}, "
                f"{round(hole['master_ms'][1], 2)}] b2_step_ms="
                f"{None if hole['step_ms'] is None else round(hole['step_ms'], 3)} "
                f"rule=ADDENDUM_25_2_no_audio_jump_no_fill for {candidate_path}\n")
    for low, high, point in regions:
        if id(point) not in seen:
            step_result("change_point_unseen_by_b2", candidate=candidate_path,
                        a_ms=point["a_ms"], b_ms=point["b_ms"], jump_ms=point["jump_ms"],
                        region_ms=[round(low, 1), round(high, 1)])


# ---------------------------------------------------------------------------
# STEP 5 -- PLAN APPLICATION. It applies the plan and nothing else (ADDENDUM 10 d).
# ---------------------------------------------------------------------------

def _decimal(value):
    """An exact `Fraction` (or anything `str()` renders exactly) as a `Decimal`, for the
    millisecond arithmetic the assembly does in `Decimal`."""
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(str(value))


def plan_geometry(transitions, head_end_s, tail_start_s, domain, walk):
    """The plan laid on the master timeline IN MILLISECONDS (ADDENDUM 25.1-25.2): the audio's
    transitions in order, each zone read from the candidate at ITS level's offset, each fill
    exactly the width the audio step measured.

      head      fill [0, head_end) from the master when the audio's head edge is past one frame
      transition at T, fill w:   the zone before ends at T; a fill [T, T + w) when w > 0 (a
                deletion); the zone after starts at T + w (an addition or a slip starts it at T)
      tail      fill [tail_start, timeline end) from the master; the master writes what audio it
                has there (ADDENDUM 25.6 counts only that, see `written_edge_seconds`)
    Each zone's `offset_ms` is the median offset of the walk's measured windows wholly inside it
    -- the walk's own level, at the millisecond (replaces the three-window majority, 25.2) --
    and the level on the zone's side of its transition when no window lies wholly inside.

    Returns `(zones, fills, None)` or `(None, None, reason)` when the transitions overlap."""
    import audio_walk
    timeline_ms = _decimal(domain["master_timeline_ms"])
    zones, fills = [], []
    cursor = Decimal(0)
    if head_end_s is not None:
        cursor = min(Decimal(str(head_end_s)) * 1000, timeline_ms)
        fills.append({"master_start_ms": Decimal(0), "master_end_ms": cursor,
                      "reason": WHY_TOKEN["head"], "hole": "head", "status": "audio_edge"})
    fallback = [walk["levels"][0]["off_ms"]] + [t["b_ms"] for t in transitions]
    boundaries = []
    for number, transition in enumerate(transitions):
        at = Decimal(str(transition["at_s"])) * 1000
        width = Decimal(str(transition["fill_s"])) * 1000
        if at < cursor:
            return None, None, (f"transition {number} at {at} ms lies before the previous "
                                f"piece's end {cursor} ms")
        boundaries.append((cursor, at, fallback[number]))
        if width > 0:
            fills.append({"master_start_ms": at, "master_end_ms": at + width,
                          "reason": WHY_TOKEN["interior"], "hole": number,
                          "status": transition["decision"]})
        cursor = at + width
    end = timeline_ms
    if tail_start_s is not None:
        end = min(Decimal(str(tail_start_s)) * 1000, timeline_ms)
        if end < cursor:
            return None, None, f"the tail edge {end} ms lies before the last piece's end {cursor} ms"
    boundaries.append((cursor, end, fallback[len(transitions)]))
    if tail_start_s is not None and end < timeline_ms:
        fills.append({"master_start_ms": end, "master_end_ms": timeline_ms,
                      "reason": WHY_TOKEN["tail"], "hole": "tail", "status": "audio_edge"})
    for start, stop, level in boundaries:
        if stop <= start:
            continue
        inside = [row["off"] for row in walk["rows"]
                  if row["status"] == "ok" and row["t"] * 1000 >= float(start)
                  and (row["t"] + audio_walk.WALK_WINDOW_S) * 1000 <= float(stop)]
        offset = (Decimal(str(round(float(statistics.median(inside)), 3))) if inside
                  else Decimal(str(level)))
        zones.append({"master_start_ms": start, "master_end_ms": stop, "offset_ms": offset,
                      "n_windows": len(inside), "zone": len(zones)})
    fills.sort(key=lambda fill: fill["master_start_ms"])
    return zones, fills, None


def written_edge_seconds(fills, master_audio_end_s):
    """ADDENDUM 25.6: an edge fill counts (log, the 15 s rule, the record) only for the part the
    master can WRITE -- a tail fill past the master's own audio end is not written (the
    container keeps the master video's length). Returns `(head_s, tail_s)`."""
    end = Decimal(str(master_audio_end_s)) * 1000
    head = sum((f["master_end_ms"] - f["master_start_ms"]) for f in fills
               if f["reason"] == WHY_TOKEN["head"])
    tail = sum(max(Decimal(0), min(f["master_end_ms"], end) - f["master_start_ms"])
               for f in fills if f["reason"] == WHY_TOKEN["tail"])
    return float(head) / 1000.0, float(tail) / 1000.0


def tag_decision(n_splices, edge_added_s):
    """ADDENDUM 5, on the audio plan: any interior splice (a fill, an addition's skip or a slip
    -- every transition is one) tags (bound a); otherwise edge additions actually written at or
    above the threshold tag (bound b). Returns `(required, reason)`."""
    if n_splices:
        return True, (f"{n_splices} interior splice(s) tag regardless of their size "
                      f"(addendum 5 bound a)")
    if edge_added_s >= EDGE_ADDITION_CHIMERIC_TAG_THRESHOLD_SECONDS:
        return True, (f"edge additions written total {edge_added_s:.3f}s, at or above the "
                      f"{EDGE_ADDITION_CHIMERIC_TAG_THRESHOLD_SECONDS}s threshold "
                      f"(addendum 5 bound b)")
    return False, (f"edge additions written total {edge_added_s:.3f}s, under the "
                   f"{EDGE_ADDITION_CHIMERIC_TAG_THRESHOLD_SECONDS}s threshold and no interior "
                   f"splice -- the original track with a marginal completion, not a chimera")


# A zone this short (plus the walk's window) holds no window of its own; its offset is derived.
ZONE_EDGE_MARGIN_S = 0.5


def remeasure_at_other_levels(zones, readings, measure, searched_ms):
    """A ZONE THE REFERENCE'S OFFSET DID NOT EXPLAIN IS SEARCHED AGAIN AT THE TRACK'S OWN LEVELS,
    then at the reference's other levels, before any offset is derived for it. `measure(low,
    high, seed)` is `audio_walk.zone_offset` over the zone's margins; a seed within `searched_ms`
    of the zone's reference offset was already searched and is skipped. The reading that measured
    the most windows is kept, with its `seed_ms`; the retries are returned for the log.

    Without it a track that does not take the reference's steps is given them: MEASURED, id 126
    (Undead Unluck S01E01, NF candidate), the jpn E-AC-3 reads +538/-464/-1464/-2465 ms against
    the master over four zones while the en/es/pt AAC dubs read -501 ms end to end; each dub was
    measured in the one zone whose reference level lay within 150 ms of -501, and the other three
    were `derived_by_reference_step` to +501/-1501/-2502 ms, delivering them 1002/1000/2001 ms
    off (verifier: 2004.4-2006.0 ms)."""
    import audio_walk
    own = [float(r["offset_ms"]) for r in readings if r["offset_ms"] is not None]
    other = [float(zone["offset_ms"]) for zone in zones]
    retries = []
    for zone, reading in zip(zones, readings):
        if reading["offset_ms"] is not None or not own:
            continue
        low = float(zone["master_start_ms"]) / 1000.0 + ZONE_EDGE_MARGIN_S
        high = float(zone["master_end_ms"]) / 1000.0 - ZONE_EDGE_MARGIN_S
        if high - low < audio_walk.WALK_WINDOW_S:
            continue
        reference_ms = float(zone["offset_ms"])
        tried, best = [], None
        for seed in own + other:
            if abs(seed - reference_ms) <= searched_ms or any(abs(seed - s) < 1.0 for s in tried):
                continue
            tried.append(seed)
            measured = measure(low, high, seed)
            if measured["offset_ms"] is not None and (best is None
                                                      or measured["n_ok"] > best[1]["n_ok"]):
                best = (seed, measured)
        if best is None:
            continue
        seed, measured = best
        reading.update(offset_ms=Decimal(str(measured["offset_ms"])), windows=measured["n_ok"],
                       seed_ms=seed, reason=None)
        retries.append({"zone": zone["zone"], "reference_ms": reference_ms, "seed_ms": seed,
                        "offset_ms": measured["offset_ms"], "windows": measured["n_ok"]})
    return retries


def track_offsets(zones, walk, master_obj, candidate_obj, language, speed_ratio, scale,
                  deadline=None, engine="asetrate"):
    """EACH TRACK ITS OWN OFFSET, PER ZONE, AT THE MILLISECOND (ADDENDUM 9 points 2 and 14;
    ADDENDUM 25.2 -- the walk replaces the three-window majority).

      the reference track   the walk's own zone offsets (`plan_geometry`), source `walk_level`
      a track whose language the master carries: `audio_walk.zone_offset` over each zone
                            (margins of `ZONE_EDGE_MARGIN_S`) seeded by the zone's reference
                            offset, against the master track of ITS language; a track that
                            shows more than one level inside a zone is logged (the dominant
                            level is used); a zone that seed does not explain is searched again
                            at the track's own measured levels, then the reference's other ones
                            (`remeasure_at_other_levels`, logged `offset_remeasured`)
      a zone of such a track too short or too quiet to measure: `derived_by_reference_step` --
                            the track's nearest measured zone moved by the REFERENCE's own step
                            between the two zones (a millisecond relation, never a frame count)
      no master track of that language, or nothing derivable: `inherited` -- the reference's
                            offset, with both values logged (never another language in silence)

    Returns `(tracks, None)` or `(None, reason)`; `tracks[stream_order]` = {"language",
    "zones": [readings], "start_ms", "extent_ms", "extent_source", "measured", "reference"}."""
    import audio_walk
    import merge_video_chimeric
    reference_order = int(walk["candidate_stream"])
    audios = list(merge_video_chimeric.iterate_candidate_audios(candidate_obj))
    master_cache = {int(walk["master_stream"]): walk["master"]}
    tracks = {}
    for track_language, audio in audios:
        order = int(audio["StreamOrder"])
        start_ms, extent_ms, extent_source = _track_timing(candidate_obj, audio, scale)
        entry = {"language": track_language, "start_ms": start_ms, "extent_ms": extent_ms,
                 "extent_source": extent_source, "zones": [], "measured": False,
                 "reference": None}
        tracks[order] = entry
        if order == reference_order:
            entry["reference"] = int(walk["master_stream"])
            entry["zones"] = [{"zone": zone["zone"], "offset_ms": zone["offset_ms"],
                               "coarse_offset_ms": str(zone["offset_ms"]),
                               "source": "walk_level", "reason": None,
                               "windows": zone["n_windows"]} for zone in zones]
            entry["measured"] = True
            continue
        master_audio = merge_video_chimeric.find_master_audio_for_language(
            master_obj, track_language,
            walk["master_stream"] if track_language == language else None)
        step_launch("track_offset", candidate=candidate_obj.filePath, stream=order,
                    language=track_language,
                    master_stream=None if master_audio is None else master_audio.get("StreamOrder"))
        readings = [{"zone": zone["zone"], "offset_ms": None,
                     "coarse_offset_ms": str(zone["offset_ms"]), "reason": None, "windows": 0}
                    for zone in zones]
        entry["zones"] = readings
        if master_audio is None:
            entry["own_reason"] = f"master_carries_no_{track_language}_track"
            for reading in readings:
                reading["reason"] = entry["own_reason"]
            step_result("track_offset", candidate=candidate_obj.filePath, stream=order,
                        measured=False, reason=entry["own_reason"])
            continue
        master_order = int(master_audio["StreamOrder"])
        entry["reference"] = master_order
        try:
            envelope = engine == "atempo" and speed_ratio is not None
            if master_order not in master_cache:
                master_cache[master_order] = audio_walk.read_on_file_clock(
                    master_obj, master_audio, deadline=deadline, envelope=envelope)
            samples = audio_walk.read_on_file_clock(
                candidate_obj, audio, _speed_chain(audio, speed_ratio, engine), scale,
                deadline=deadline, envelope=envelope)
        except Exception as error:                                       # noqa: BLE001
            if getattr(error, "cause", None) == "repair_budget_exceeded":
                raise
            entry["own_reason"] = f"track_unreadable({type(error).__name__})"
            for reading in readings:
                reading["reason"] = entry["own_reason"]
            step_result("track_offset", candidate=candidate_obj.filePath, stream=order,
                        measured=False, reason=entry["own_reason"], evidence=str(error)[:200])
            continue
        for zone, reading in zip(zones, readings):
            low = float(zone["master_start_ms"]) / 1000.0 + ZONE_EDGE_MARGIN_S
            high = float(zone["master_end_ms"]) / 1000.0 - ZONE_EDGE_MARGIN_S
            if high - low < audio_walk.WALK_WINDOW_S:
                reading["reason"] = f"zone_too_short({round(high - low + 2 * ZONE_EDGE_MARGIN_S, 3)}s)"
                continue
            measured = audio_walk.zone_offset(master_cache[master_order], samples, low, high,
                                              float(zone["offset_ms"]))
            reading["windows"] = measured["n_ok"]
            if measured["offset_ms"] is None:
                reading["reason"] = f"no_window_measured({measured['counts']})"
                continue
            reading["offset_ms"] = Decimal(str(measured["offset_ms"]))
            if len(measured["levels"]) > 1:
                tools.dev_log(f"orchestrator: track {order} ({track_language}) changes inside zone "
                              f"{zone['zone']}: levels "
                              f"{[(lv['t_first'], lv['t_last'], lv['off_ms']) for lv in measured['levels']]}"
                              f" -- the dominant one is applied\n")
        for retry in remeasure_at_other_levels(
                zones, readings,
                lambda low, high, seed: audio_walk.zone_offset(
                    master_cache[master_order], samples, low, high, seed),
                audio_walk.WALK_SEARCH_MS):
            tools.log_line(f"repair: offset_remeasured stream={order} lang={track_language} "
                           f"zone={retry['zone']} reference_ms={retry['reference_ms']} "
                           f"seed_ms={retry['seed_ms']} measured_ms={retry['offset_ms']} "
                           f"windows={retry['windows']}\n")
        del samples
        entry["measured"] = any(reading["offset_ms"] is not None for reading in readings)
        step_result("track_offset", candidate=candidate_obj.filePath, stream=order,
                    language=track_language, master_stream=master_order,
                    measured_zones=sum(1 for r in readings if r["offset_ms"] is not None),
                    n_zones=len(zones),
                    offsets_ms=[None if r["offset_ms"] is None else float(r["offset_ms"])
                                for r in readings],
                    reasons=[r["reason"] for r in readings])
    del master_cache
    if reference_order not in tracks:
        return None, (f"the comparison track (stream {reference_order}) is not among the "
                      f"candidate's audio tracks")
    reference = tracks[reference_order]["zones"]
    for order, entry in tracks.items():
        for reading in entry["zones"]:
            if reading["offset_ms"] is not None:
                reading.setdefault("source", "measured")
    for order, entry in tracks.items():
        if order == reference_order:
            continue
        measured = [r for r in entry["zones"] if r.get("source") == "measured"]
        for reading in entry["zones"]:
            if reading["offset_ms"] is not None:
                continue
            if measured:
                nearest = min(measured, key=lambda r: abs(r["zone"] - reading["zone"]))
                reading["offset_ms"] = (nearest["offset_ms"]
                                        + reference[reading["zone"]]["offset_ms"]
                                        - reference[nearest["zone"]]["offset_ms"])
                reading["source"] = f"derived_by_reference_step(zone_{nearest['zone']})"
            else:
                reading["offset_ms"] = reference[reading["zone"]]["offset_ms"]
                reading["source"] = f"inherited(stream_{reference_order})"
            # ADDENDUM 9 point 14: a substitution is logged with BOTH values -- a DECISION.
            tools.log_line(
                f"repair: offset_substitution stream={order} lang={entry['language']} "
                f"zone={reading['zone']} own=unmeasured({reading['reason']}) "
                f"reference_ms={reference[reading['zone']]['offset_ms']} "
                f"applied_ms={reading['offset_ms']} source={reading['source']}\n")
    return tracks, None


def _track_timing(video_obj, audio, scale):
    """This track's first-sample time and its real end, on the plan's timeline (the candidate's
    master-equivalent one on a rate pair), in ms. The end is read from the PACKETS
    (`measure_track_extent_ms`): a declared `Duration` under-reports its own stream by up to
    120 ms on 16 of 59 measured tracks, and a zone cut short on that number would fill real
    candidate content from the master. The declared value is the fallback, and says so."""
    import merge_video_chimeric
    start_ms = merge_video_chimeric.get_stream_start_ms(audio) * scale
    extent_ms, reason = merge_video_chimeric.measure_track_extent_ms(
        video_obj.filePath, int(audio["StreamOrder"]))
    source = f"packets({reason})"
    if extent_ms is None:
        declared = merge_video_chimeric.get_track_audio_length_ms(audio)
        if declared is not None:
            extent_ms = declared + merge_video_chimeric.get_stream_start_ms(audio)
            source = f"declared_duration(packets {reason})"
    if extent_ms is not None:
        extent_ms = extent_ms * scale
    return start_ms, extent_ms, source


def track_pieces(zones, fills, readings, extent_ms, timeline_ms):
    """The plan as ONE track reads it: master fills at the resolved frames, candidate zones at
    THIS track's own offsets. Two adjustments only, both at the file's edges and both logged,
    because they are facts of the track and not decisions about the plan: a zone that would
    read BEFORE the candidate's time zero (a sub-frame offset landing a few ms under it) starts
    where the candidate starts, and a zone that would read PAST the track's real end stops
    there; the master fills the difference, exactly as it fills any hole.

    A candidate read that goes back over the previous one at a splice -- the audio's edit and
    the video's cut differ by a fraction of a frame, measured -- is not refused: each zone is
    read at its own measured offset, which is the plan. It is RETURNED, per splice, as
    `overlaps`, so the log carries it.
    """
    segments = ([dict(fill, source="master") for fill in fills]
                + [dict(zone, source="candidate") for zone in zones])
    segments.sort(key=lambda segment: segment["master_start_ms"])
    pieces, adjustments = [], []
    for segment in segments:
        start, end = segment["master_start_ms"], segment["master_end_ms"]
        if segment["source"] == "master":
            pieces.append({"source": "master", "master_start_ms": start, "master_end_ms": end,
                           "source_start_ms": start, "reason": segment["reason"]})
            continue
        offset = readings[segment["zone"]]["offset_ms"]
        if start + offset < 0:
            shifted = -offset
            adjustments.append({"zone": segment["zone"], "kind": "head_before_candidate_zero",
                                "master_fill_ms": str(shifted - start)})
            pieces.append({"source": "master", "master_start_ms": start,
                           "master_end_ms": min(shifted, end), "source_start_ms": start,
                           "reason": WHY_TOKEN["head"] if start == 0 else WHY_TOKEN["interior"]})
            start = min(shifted, end)
        if extent_ms is not None and end + offset > extent_ms and start < end:
            cut = min(end - start, end + offset - extent_ms)
            adjustments.append({"zone": segment["zone"], "kind": "past_track_end",
                                "master_fill_ms": str(cut)})
            pieces_tail = {"source": "master", "master_start_ms": end - cut,
                           "master_end_ms": end, "source_start_ms": end - cut,
                           "reason": (WHY_TOKEN["tail"] if end == timeline_ms
                                      else WHY_TOKEN["interior"])}
            if end - cut > start:
                pieces.append({"source": "candidate", "master_start_ms": start,
                               "master_end_ms": end - cut, "source_start_ms": start + offset,
                               "zone": segment["zone"], "reason": "zone"})
            pieces.append(pieces_tail)
            continue
        if end > start:
            pieces.append({"source": "candidate", "master_start_ms": start, "master_end_ms": end,
                           "source_start_ms": start + offset, "zone": segment["zone"],
                           "reason": "zone"})
    merged = []
    for piece in pieces:
        if (merged and piece["source"] == "master" and merged[-1]["source"] == "master"
                and merged[-1]["master_end_ms"] == piece["master_start_ms"]):
            merged[-1]["master_end_ms"] = piece["master_end_ms"]
            if piece["reason"] == WHY_TOKEN["tail"]:
                merged[-1]["reason"] = WHY_TOKEN["tail"]
            continue
        merged.append(piece)
    overlaps = []
    previous = None
    for piece in merged:
        if piece["source"] != "candidate":
            continue
        if previous is not None:
            previous_end = previous["source_start_ms"] + (previous["master_end_ms"]
                                                          - previous["master_start_ms"])
            if piece["source_start_ms"] < previous_end:
                overlaps.append({"zones": [previous["zone"], piece["zone"]],
                                 "reread_ms": str(previous_end - piece["source_start_ms"])})
        previous = piece
    return merged, adjustments, overlaps


@repair_log.timed_phase("orchestrator", "apply_plan", lambda candidate_path, *a, **k: candidate_path)
def apply_plan(candidate_path, plan_spec, speed_factor, master_obj, candidate_obj, context):
    """STAGE 5 -- THE PLAN, APPLIED, AND NOTHING ELSE (ADDENDUM 10 d: "LA FONCTION QUI TRAITE LE
    PLAN NE FAIT QUE TRAITER LE PLAN -- aucune decision, aucun test d'opportunite"). Returns
    `(ok, cause, reason)`; `ok` is True only when the temporary chimeric file exists.

    WHAT IT RECEIVES (ADDENDUM 25, the audio plan): `plan_spec` = {"zones", "fills" -- from
    `plan_geometry`, in master ms, each fill exactly the audio step's width; "walk" -- the
    reference walk, whose tracks and levels the offsets reuse; "head_written_s",
    "tail_written_s" -- the edge fills the master can actually write (25.6)}. `context` carries
    the comparison language, the REFERENCE couple's streams and quantum, the frame domain, the
    sweep's gate and the ADDENDUM 5 marker decision taken on this plan.

    WHAT IT DOES, in order, each step launched and resulted in the dev log:
      1  the geometry -- logged as received; every edge addition logged (ADDENDUM 5 clause d);
      2  the speed -- at a confirmed factor other than 1 the candidate's tracks are resampled at
         the EXACT rational (asetrate behind the pitch layer's routing, ADDENDUM 8), `resampled:
         <effective factor>` on every such track; at 1 no filter exists (ADDENDUM 6);
      3  the audio offsets -- `track_offsets`: each track, each zone, at the millisecond from the
         walk (ADDENDUM 9 points 2 and 14, 25.2), never rounded to a frame;
      4  the pieces -- `track_pieces`, one set per track; the comparison track's set re-times the
         subtitles (pysubs2; the factor on the cue timecodes first on a rate pair, ADDENDUM 8
         point 3; no cue twice at a splice, a cue stopped before the next -- ADDENDUM 10 d);
      5  the chapters -- master editions as they are, candidate editions only re-timed
         (ADDENDUM 9 point 7), every decision logged;
      6  the build -- `merge_video_repair.build_repaired_video_object`: assemble, mux, the
         delivery gates (`verify_output_file`, `verify_on_master_timeline` at 15 ms -- ADDENDUM
         25.3 -- and the fabricated-delivery gate); a build that DELIVERS NOTHING (every audio
         dropped by the gate, no subtitle) is still returned -- `nothing_to_deliver` is logged,
         the merge runs and the master wins (25.7 as reversed by the owner, 2026-09-25);
      7  DELIVERED_DURATIONS -- ffprobe on the chimeric file itself, right after it exists
         (ADDENDUM 10 b); that is what the forensic verifier compares;
      8  the seam -- the repaired object on `REPAIR_SEAM_ATTRIBUTE`, and the `repaired`
         terminal written HERE through `record()` (the entry writes none).

    A REFUSAL FROM THE ASSEMBLY IS NOT CAUGHT HERE: a `chimeric_error` carries the token its
    raise site set (`delivery_timeline_misalignment`, `alignment_contradicts_plan`, ...) and the
    entry records it `declined` with that token -- translating it into a token of this module
    would erase what the gate saw. It is logged as this step's result on the way out.
    """
    import merge_video_chimeric
    import merge_video_repair
    started = time.time()
    domain = context["domain"]
    language = context["language"]
    work_dir = path.join(context["work_dir"], "apply_plan")
    tools.make_dirs(work_dir)
    timeline_ms = _decimal(domain["master_timeline_ms"])
    rate_text = (f"{speed_factor.numerator}/{speed_factor.denominator}"
                 if isinstance(speed_factor, Fraction) else speed_factor)
    zones, fills = plan_spec["zones"], plan_spec["fills"]
    step_launch("apply_plan", candidate=candidate_path, n_zones=len(zones), n_fills=len(fills),
                speed_factor=rate_text, chimeric_tag=context["tagged"])

    # ---- 1. geometry --------------------------------------------------------
    # ADDENDUM 25.6: the edge fills COUNTED are the ones the master can write.
    head_added = Decimal(str(round(plan_spec["head_written_s"] * 1000.0, 3)))
    tail_added = Decimal(str(round(plan_spec["tail_written_s"] * 1000.0, 3)))
    interior_filled = sum((fill["master_end_ms"] - fill["master_start_ms"]) for fill in fills
                          if fill["reason"] == WHY_TOKEN["interior"])
    step_result("plan_geometry", candidate=candidate_path,
                zones=[[float(z["master_start_ms"]), float(z["master_end_ms"])] for z in zones],
                offsets_ms=[float(z["offset_ms"]) for z in zones],
                zone_windows=[z["n_windows"] for z in zones],
                fills=[[float(f["master_start_ms"]), float(f["master_end_ms"]), f["reason"],
                        f["status"]] for f in fills])
    # ADDENDUM 5 clause (d), as ADDENDUM 21.10 re-scoped it: the added durations per edge and the
    # tag decision are a DEV line; the permanent record is the delivered track's tag and the
    # `repaired` record's summary below (`edge_additions_ms`, `chimeric_tag`), always written.
    tools.dev_log(f"orchestrator: edge_additions head_ms={head_added} tail_ms={tail_added} "
                  f"interior_filled_ms={interior_filled} chimeric_tag={context['tagged']} "
                  f"tag_reason={context['tag_reason'].replace(' ', '_')} "
                  f"for {candidate_path}\n")
    if not zones:
        step_result("apply_plan", candidate=candidate_path, ok=False,
                    cause="plan_reads_no_candidate_content")
        return False, "plan_reads_no_candidate_content", (
            "the resolved holes leave no candidate content on the master timeline -- a plan "
            "that reads nothing from the candidate is the master, not a repair")

    # ---- 2. speed -----------------------------------------------------------
    speed_ratio = None
    scale = Decimal(1)
    if speed_factor is not None and speed_factor != 1:
        speed_ratio = _decimal(Fraction(speed_factor))
        scale = speed_ratio
    step_result("delivery_speed", candidate=candidate_path, speed_factor=rate_text,
                speed_ratio=(None if speed_ratio is None else str(speed_ratio)),
                filter=(None if speed_ratio is None
                        else (context.get("resample_routing") or {}).get("filter_name",
                                                                         "asetrate")),
                rule=("ADDENDUM_6_no_filter_without_speed_change" if speed_ratio is None
                      else "ADDENDUM_8_resample_is_restoration"))

    # ---- 3. per-track sub-frame offsets --------------------------------------
    engine = (context.get("resample_routing") or {}).get("filter_name", "asetrate")
    tracks, offset_failure = track_offsets(zones, plan_spec["walk"], master_obj, candidate_obj,
                                           language, speed_ratio, scale,
                                           deadline=domain.get("repair_deadline"), engine=engine)
    if tracks is None:
        step_result("apply_plan", candidate=candidate_path, ok=False,
                    cause="plan_offset_unmeasurable")
        return False, "plan_offset_unmeasurable", offset_failure

    # ---- 4. pieces per track -------------------------------------------------
    track_plans = {}
    reference_pieces = None
    for order, entry in tracks.items():
        pieces, adjustments, overlaps = track_pieces(zones, fills, entry["zones"],
                                                     entry["extent_ms"], timeline_ms)
        sources = sorted({reading["source"] for reading in entry["zones"]})
        own = set(sources) <= {"measured", "walk_level"}
        track_plans[order] = {
            "pieces": pieces,
            "extent_ms": entry["extent_ms"], "extent_source": entry["extent_source"],
            "offset_measured": own,
            "borrow_reason": (None if own
                              else ",".join(sources) + (f"[{entry['own_reason']}]"
                                                        if entry.get("own_reason") else "")),
            "offset_sources": [{"zone": r["zone"], "offset_ms": str(r["offset_ms"]),
                                "source": r["source"]} for r in entry["zones"]]}
        for adjustment in adjustments:
            tools.log_line(f"repair: plan_edge_adjustment stream={order} "
                              f"zone={adjustment['zone']} kind={adjustment['kind']} "
                              f"master_fill_ms={adjustment['master_fill_ms']}\n")
        for overlap in overlaps:
            tools.log_line(f"repair: splice_reread stream={order} zones={overlap['zones']} "
                              f"reread_ms={overlap['reread_ms']} (the audio edit and the video "
                              f"cut differ by a fraction of a frame; each zone is read at its "
                              f"own measured offset)\n")
        if str(order) == str(context["candidate_stream"]):
            reference_pieces = pieces
        step_result("track_pieces", candidate=candidate_path, stream=order,
                    n_pieces=len(pieces), sources=sources,
                    pieces=[(p["source"][0], float(p["master_start_ms"]),
                             float(p["master_end_ms"]),
                             float(p["source_start_ms"])) for p in pieces])

    # ---- 5. chapters ---------------------------------------------------------
    step_launch("chapters", candidate=candidate_path)
    chapters_path, chapter_decisions = merge_video_chimeric.build_delivered_chapters(
        master_obj.filePath, candidate_obj.filePath, reference_pieces,
        speed_ratio, timeline_ms, work_dir)
    for decision in chapter_decisions:
        tools.log_line("repair: chapter " + " ".join(
            f"{key}={str(value).replace(' ', '_')}" for key, value in decision.items())
            + "\n")
    step_result("chapters", candidate=candidate_path, delivered=chapters_path is not None,
                n_decisions=len(chapter_decisions))

    # ---- 6. build ------------------------------------------------------------
    seam = getattr(candidate_obj, merge_video_repair.REPAIR_SEAM_ATTRIBUTE, None)
    job_start_utc = (seam or {}).get("job_start_utc")
    if job_start_utc is None:
        # THE SEAM IS ABSENT: the orchestrator is driven standalone. There is no job start to
        # stamp, and one is NOT invented -- the era tag says so.
        job_start_utc = "unstamped(no_repair_seam_standalone_run)"
        tools.dev_log(f"orchestrator: no {merge_video_repair.REPAIR_SEAM_ATTRIBUTE} on "
                      f"{candidate_path}: standalone run, the era tag carries no job start\n")
    marker = "chimeric" if context["tagged"] else ""
    comparison_offsets = tracks[int(context["candidate_stream"])]["zones"]
    plan = {
        "kind": "orchestrator_chimeric",
        "language": language, "reference_stream": context["master_stream"],
        "quantum_ms": context["quantum_ms"], "master_path": master_obj.filePath,
        "decided_by": "repair_orchestrator.apply_plan",
        "segments_dropped_unusable": 0,
        "speed_margin": (context.get("sweep_gate") or {}).get("margin"),
        "speed_engine": (None if speed_ratio is None
                         else (context.get("resample_routing") or {}).get("filter_name")),
        "speed_margin_absent_reason": ("no_rate_relation" if speed_ratio is None else None),
        "segments": [{"master_start_ms": zone["master_start_ms"],
                      "master_end_ms": zone["master_end_ms"],
                      "candidate_offset_ms": comparison_offsets[zone["zone"]]["offset_ms"],
                      "candidate_offset_ms_by_stream": {
                          order: str(entry["zones"][zone["zone"]]["offset_ms"])
                          for order, entry in tracks.items()}}
                     for zone in zones],
        "track_plans": track_plans, "reference_pieces": reference_pieces,
        "marker": marker, "chapters_path": chapters_path,
        "speed_ratio": speed_ratio,
        "speed_ratio_exact": (None if speed_ratio is None else rate_text),
        "rate_source": (None if speed_ratio is None else "rate_arm"),
        "resample_gate": context.get("sweep_gate"),
        # the repair's budget reaches the delivery verifier (ADDENDUM 26, report commit)
        "repair_deadline": domain.get("repair_deadline"),
    }
    step_launch("build", candidate=candidate_path, marker=marker,
                n_tracks=len(track_plans))
    try:
        repaired_obj, assembly = merge_video_repair.build_repaired_video_object(
            candidate_obj, master_obj, plan, path.join(tools.tmpFolder, "repair"),
            job_start_utc)
    except merge_video_chimeric.chimeric_error as error:
        step_result("build", candidate=candidate_path, ok=False,
                    cause=getattr(error, "cause", None), error=str(error)[:300],
                    seconds=round(time.time() - started, 1))
        raise
    out_path = getattr(repaired_obj, "filePath", None)
    exists = bool(out_path) and path.exists(out_path)
    step_result("build", candidate=candidate_path, ok=exists, out_path=out_path,
                marker=assembly.get("marker"),
                track_markers=[r.get("marker") for r in assembly.get("audios") or []],
                verification=[(v.get("track"), v.get("outcome"), v.get("worst_lag_ms"))
                              for v in assembly.get("verification") or []],
                fabricated_dropped=len(assembly.get("fabricated_dropped") or []))
    if not exists:
        return False, "plan_application_no_file", (
            f"the build returned but the temporary chimeric file is not on disk ({out_path}) -- "
            f"no file, no repair")
    # ADDENDUM 25.7 AS REVERSED BY THE OWNER (2026-09-25): "la fusion est faite meme si a la fin
    # le keep va tout enlever". The delivery gate may drop every rebuilt audio track (same content
    # as an intact master track) and the candidate may carry no subtitle (errid-232/675, Yozakura-
    # san S02E09): the repair still returns its file, the merge runs, keep_best_audio decides, and
    # a product equal to the master CLOSES the error entry -- the candidate brought nothing, which
    # is not an error. `nothing_to_deliver` is an INFORMATIONAL line, never a refusal.
    delivered_tracks = [
        (holder, language_, audio.get("StreamOrder"))
        for holder in ("audios", "commentary", "audiodesc", "subtitles")
        for language_, entries in (getattr(repaired_obj, holder, None) or {}).items()
        for audio in entries if audio.get("keep", True) is not False]
    if not delivered_tracks:
        gate = [(d.get("stream_order"), d.get("cause")) if isinstance(d, dict) else d
                for d in (assembly.get("fabricated_dropped") or [])]
        tools.log_always(f"repair: nothing_to_deliver informational=1 out_path={out_path} "
                         f"gate_dropped={gate} -- no rebuilt track passes the delivery gate; the "
                         f"merge runs and the master wins for {candidate_path}\n")

    # ---- 7. DELIVERED_DURATIONS ---------------------------------------------
    delivered = merge_video_chimeric.probe_delivered_durations(out_path)
    tools.log_line(
        f"repair: DELIVERED_DURATIONS container_ms={delivered['container_ms']} "
        f"master_video_ms={timeline_ms} video_ms=absent(the_chimeric_file_carries_no_video) "
        + " ".join(f"{stream['type']}_{stream['index']}_ms={stream['duration_ms']}"
                   for stream in delivered["streams"])
        + f" max_cue_end_ms={delivered['max_cue_end_ms']} for {candidate_path}\n")

    # ---- 8. the seam and the terminal ----------------------------------------
    if seam is not None:
        seam["repaired_obj"] = repaired_obj
        seam["assembly"] = assembly
    summary = {
        "out_path": out_path,
        "zones": [[str(z["master_start_ms"]), str(z["master_end_ms"])] for z in zones],
        "fills": [[str(f["master_start_ms"]), str(f["master_end_ms"]), f["reason"]]
                  for f in fills],
        "offsets_ms": {order: plan_["offset_sources"] for order, plan_ in track_plans.items()},
        "pieces": {order: len(plan_["pieces"]) for order, plan_ in track_plans.items()},
        "markers": {r["stream_order"]: r.get("marker") for r in assembly.get("audios") or []},
        "edge_additions_ms": {"head": str(head_added), "tail": str(tail_added),
                              "interior_filled": str(interior_filled)},
        "chimeric_tag": context["tagged"], "speed_ratio": plan["speed_ratio_exact"],
        "chapters": chapters_path is not None,
        "delivered_durations": delivered,
        "fabricated_dropped": assembly.get("fabricated_dropped"),
    }
    reason = (f"plan applied: {len(zones)} candidate zone(s), {len(fills)} master fill(s) "
              f"(head {head_added} ms, interior {interior_filled} ms, tail {tail_added} ms), "
              f"{len(assembly.get('audios') or [])} audio and "
              f"{len(assembly.get('subtitles') or [])} subtitle track(s) rebuilt, marker "
              f"'{assembly.get('marker')}', speed {rate_text}, "
              f"{len(assembly.get('fabricated_dropped') or [])} fabricated track(s) dropped by the "
              f"delivery gate, temporary chimeric file {out_path}")
    merge_video_repair.record(candidate_path, "repaired", reason, detail=summary)
    step_result("apply_plan", candidate=candidate_path, ok=True, out_path=out_path,
                seconds=round(time.time() - started, 1))
    return True, None, reason


# ---------------------------------------------------------------------------
# STEP 2 -- the similarity gate and the speed sweep
# ---------------------------------------------------------------------------

RATE_ARM_WAV_NAME = "rate_arm_{name}_{engine}.wav"


def _drop_rate_wav(primed):
    """The prime's kept comparison WAV (for the rate arm) is deleted on every path that does
    not run the arm -- the arm deletes it itself."""
    rate_wav = primed.pop("rate_wav", None)
    if rate_wav is not None:
        try:
            remove(rate_wav["path"])
        except OSError:
            pass


def _finalist_row(ratio, engine, alignment, master_duration_ms, seconds, candidate_path):
    """One finalist's reading (rate_direction.finalist_reading), its residual-rate test being the
    pipeline's two ladder readings -- the one-quantum ladder and the fast drift."""
    import rate_direction
    _zones, detail = coalesce_same_offset_zones(alignment.get("zones") or [],
                                                alignment.get("zones_detail") or [])
    quantum_ms = alignment.get("quantum_ms")
    ladder = (zone_ladder_signature(alignment)["is_rate_ladder"]
              or rate_direction.fast_drift_signature(alignment.get("zones_detail") or [],
                                                     quantum_ms)["fires"])
    row = {"ratio": ratio, "engine": engine,
           **rate_direction.finalist_reading(detail, quantum_ms, master_duration_ms, ladder),
           "point_coverage": alignment.get("master_axis_coverage_fraction"),
           "seconds": round(seconds, 1)}
    rate_direction.log_finalist(candidate_path, row)
    return row


def rate_arm(primed, first_ratios, work_dir, candidate_path, deadline=None):
    """THE RATE ARM (ADDENDUM 15/21.4/30/30.5): which named ratio, and which engine, if any.

    The candidate WAV the prime already extracted (the reference couple, `primed["rate_wav"]`)
    is resampled at each finalist (ratio, engine) through `merge_video_resample.
    build_transform_chain`, fingerprinted, and aligned against the master's fingerprints by the
    prime's own `align_fingerprints`; the prime's alignment is the finalist at 1. Round one:
    `first_ratios` (the declared frame rates and the fast drift -- rate_direction.first_finalists)
    in both engines; round two, only when round one has no winner: every named ratio in both
    engines -- "similarity_unrecoverable_by_resample survives only when every named ratio failed
    with both engines". The winner is `rate_direction.choose_winner`'s.

    Returns `(factor, engine, gate, cause)`: `factor` an exact Fraction or None (1 won, or no
    finalist qualified -- `cause` says which), `gate` the evidence (every finalist's reading;
    `verdict` confirmed / declined, `span_coverage`, `margin` over the best other ratio). The WAV
    is deleted here on every path."""
    import merge_video_resample
    import rate_direction
    wav = primed.get("rate_wav")
    couple = primed["couples"][0]
    name = f"{couple[0]}x{couple[1]}"
    gate = {"instrument": "rate_arm", "verdict": "declined", "ratio": None, "engine": None,
            "span_coverage": None, "margin": None, "median_fidelity": None, "rows": [],
            "cause": None, "rounds": 0}
    try:
        baseline = primed["alignments"].get(name)
        fp_master, quantum_master, duration_master = primed["fingerprints"][("master", couple[0])]
        if wav is None or baseline is None:
            gate["cause"] = "rate_arm_unmeasured"
            return None, None, gate, "rate_arm_unmeasured"
        master_duration_ms = duration_master * 1000.0
        rows = [_finalist_row(Fraction(1), None, baseline, master_duration_ms, 0.0,
                              candidate_path)]
        tried = {Fraction(1)}

        def run_round(ratios):
            for ratio in ratios:
                for engine in merge_video_resample.SPEED_ENGINES:
                    if deadline is not None and time.monotonic() > deadline:
                        return "repair_budget_exceeded"
                    started = time.time()
                    try:
                        chain, effective = merge_video_resample.build_transform_chain(
                            wav["rate"], ratio, engine)
                    except Exception as error:                           # noqa: BLE001
                        tools.dev_log(f"orchestrator: rate_arm {ratio} {engine} unbuildable "
                                      f"({type(error).__name__}: {error})\n")
                        continue
                    out = path.join(work_dir, RATE_ARM_WAV_NAME.format(
                        name=f"{ratio.numerator}_{ratio.denominator}", engine=engine))
                    command = [tools.software["ffmpeg"], "-y", "-v", "error", "-nostdin",
                               "-i", wav["path"], "-af", chain, "-ac", "1",
                               "-ar", str(int(wav["rate"])), "-acodec", "pcm_s16le", out]
                    corrected = wav["duration_s"] * float(effective)
                    try:
                        with repair_log.announced("orchestrator", "ffmpeg", wav["path"],
                                                  media_s=wav["duration_s"]) as call:
                            done = subprocess.run(command, capture_output=True,
                                                  timeout=tools.decoder_timeout_for(
                                                      wav["duration_s"]))
                            call["exit"] = done.returncode
                        if done.returncode != 0:
                            continue
                        with repair_log.announced("orchestrator", "fpcalc", out) as call:
                            points = audioCorrelation.calculate_fingerprints(
                                out, length=corrected)
                            call["exit"] = 0
                    except (subprocess.TimeoutExpired, Exception) as error:  # noqa: BLE001
                        tools.dev_log(f"orchestrator: rate_arm {ratio} {engine} unmeasured "
                                      f"({type(error).__name__})\n")
                        continue
                    finally:
                        try:
                            remove(out)
                        except OSError:
                            pass
                    if not points:
                        continue
                    alignment = align_fingerprints(fp_master, quantum_master, duration_master,
                                                   points, CHROMAPRINT_HOP_MS, corrected)
                    rows.append(_finalist_row(ratio, engine, alignment, master_duration_ms,
                                              time.time() - started, candidate_path))
                tried.add(ratio)
            return None

        rounds = [list(first_ratios),
                  [r for r in merge_video_resample.build_rate_ratio_vocabulary()]]
        winner = None
        for number, ratios in enumerate(rounds, 1):
            ratios = [r for r in ratios if r not in tried]
            if not ratios and number > 1:
                break
            gate["rounds"] = number
            stopped = run_round(ratios)
            winner = rate_direction.choose_winner(rows)
            if stopped:
                gate["cause"] = stopped
                break
            if winner is not None:
                break
        gate["rows"] = [{**row, "ratio": f"{row['ratio'].numerator}/{row['ratio'].denominator}"}
                        for row in rows]
        if winner is None:
            gate["cause"] = gate["cause"] or "similarity_unrecoverable_by_resample"
            return None, None, gate, gate["cause"]
        others = [r["span_coverage"] for r in rows if r["ratio"] != winner["ratio"]]
        gate.update({"span_coverage": winner["span_coverage"],
                     "margin": round(winner["span_coverage"] - max(others), 4) if others else None,
                     "engine": winner["engine"], "zones": winner["zones"]})
        if winner["ratio"] == 1:
            gate["cause"] = "rate_arm_unity_wins"
            return None, None, gate, "rate_arm_unity_wins"
        gate.update({"verdict": "confirmed", "ratio": winner["ratio"]})
        return winner["ratio"], winner["engine"], gate, None
    finally:
        if wav is not None:
            try:
                remove(wav["path"])
            except OSError:
                pass
        primed.pop("rate_wav", None)


def zone_offset_rate_signature(alignment):
    """The rate-relation reading taken off the ZONES, not off the drift trace. Measurement only.

    WHY A SECOND INSTRUMENT RATHER THAN THE ONE THE DESIGN NAMED. Design section 3.6 proposes
    reading `drift_fit` (`best_shift_trace` + `fit_trace_slope`) as "there is a rate relation
    here". MEASURED this session, that instrument is BLIND on the very pair it would be for:
    `best_shift_trace` starts at offset 0 and re-centres within +/-`TRACE_M_POINTS` (3) points per
    checkpoint, so a pair whose true offset is far outside that radius never acquires the signal
    at all. errid-70's real offset is -2.85 s = -23 points: all 1026 checkpoints read offset 0,
    and the fit obediently reported slope 0.0 with a residual of 0.0 -- a perfect straight line
    through a measurement that never happened. So the drift trace is kept and logged, and this
    reading is taken beside it from a quantity that does not depend on where the trace started:
    the ALIGNED ZONES' OWN OFFSETS, which the aligner measured absolutely.

    WHAT IT COMPUTES. Least squares of `offset_points` against the zone's master midpoint, over
    the aligned zones, weighted by nothing -- one point per zone. Then, per the bake-off's own
    ruling on this exact question (`/config/output/filter_bakeoff_ear/INDEX.md`: "Ce qui les
    denonce est le RESIDU, pas la pente ; la pente seule ment"), the RESIDUAL of that fit is
    reported in milliseconds alongside the slope. A rate relation makes the offsets march
    monotonically and the line fits them; a staircase of content edits makes them sit on plateaus
    and the line cannot.

    NOTHING BRANCHES ON THIS TODAY -- see `RATE_RELATION_SLOPE_GATE_CALIBRATED`. It is emitted on
    every run so the calibration accumulates a population instead of waiting for one.

    Returns a dict, always, every key present; `None` where there was nothing to measure.
    """
    empty = {"n_zones": 0, "slope_points_per_point": None, "r_squared": None,
             "residual_rms_ms": None, "residual_max_ms": None, "span_points": None,
             "implied_total_drift_ms": None}
    detail = alignment.get("zones_detail") or []
    quantum_ms = alignment.get("quantum_ms")
    if len(detail) < 3 or not quantum_ms:
        # TWO ZONES DEFINE A LINE EXACTLY, so a fit through them has a residual of zero by
        # construction and says nothing about whether the world is a ramp or a step. The floor
        # is three, and a pair under it reports `n_zones` with everything else None rather than
        # a residual that is an artefact of the arithmetic.
        empty["n_zones"] = len(detail)
        return empty
    xs = [(zone["master_points"][0] + zone["master_points"][1]) / 2.0 for zone in detail]
    ys = [float(zone["offset_points"]) for zone in detail]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    if ss_xx == 0:
        empty["n_zones"] = n
        return empty
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / ss_xx
    intercept = mean_y - slope * mean_x
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    ss_res = sum(residual ** 2 for residual in residuals)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    span = xs[-1] - xs[0]
    return {
        "n_zones": n,
        "slope_points_per_point": slope,
        "r_squared": (1 - ss_res / ss_tot) if ss_tot > 0 else None,
        "residual_rms_ms": ((ss_res / n) ** 0.5) * quantum_ms,
        "residual_max_ms": max(abs(residual) for residual in residuals) * quantum_ms,
        "span_points": span,
        "implied_total_drift_ms": slope * span * quantum_ms,
    }


def zone_ladder_signature(alignment):
    """IS THIS ALIGNMENT A RATE LADDER? Counting, not curve fitting.

    WHAT A RATE RELATION DOES TO THIS ALIGNER, stated as a mechanism before any number. The
    aligner extends runs at a FIXED offset. Under a rate relation the true offset never stops
    moving, so a run can only survive until the drift reaches about half a quantum, at which
    point the aligner must start a new segment one point further along. The output is therefore
    a LADDER: many zones, each separated from the next by a step of exactly one quantum, all
    in the SAME direction, for the whole length of the file. A content edit does the opposite:
    a few long zones separated by a few LARGE steps, in whichever directions the editor cut or
    added.

    WHY COUNTING AND NOT A REGRESSION. The bake-off settled that a straight-line fit cannot
    separate these two: "un escalier monotone s'ajuste aussi bien a une droite", with errid 213
    returning a credible "ratio 1.0010739" on a file that has no rate relation at all. It also
    named the reading that CAN separate them -- the residual -- but the residual is only
    available in a unit this aligner cannot deliver finely: everything here is quantised to
    ~124 ms, so the bake-off's five orders of magnitude (10 microseconds against 22 seconds)
    collapse to at best one. The LADDER STRUCTURE survives the quantisation intact, because
    inside a certain band it IS the quantisation.

    *** AND THAT BAND HAS TWO EDGES, NOT ONE. An earlier version of this docstring said "a rate
    relation is the one thing that produces a long monotone run of exactly-one-quantum steps",
    full stop. That is true only BETWEEN two bounds, and only the lower one was documented:

      LOWER (span): the drift must accumulate enough quanta across the file to make
        `LADDER_MIN_RUNGS` rungs -- at the smallest named deviation that needs ~16.5 minutes of
        aligned span (see that constant).
      UPPER (rate): the drift must be SLOW ENOUGH that a segment ends before the offset has
        moved more than one quantum. `banded_seed_alignment.OFFSET_MERGE_TOLERANCE_POINTS` is 3,
        so runs whose offsets sit within 3 points are merged into ONE segment -- and any drift
        fast enough to accumulate 3 or more quanta before a segment would naturally end emerges
        as a step of 3, 4 or 5 quanta, which `abs(step) < RESOLUTION_FLOOR_QUANTA` never counts
        as a rung.

    MEASURED by an independent tester on `corpus-B-pal-ntsc`'s graded synthetic drift fixtures,
    driven through this exact function -- 0 of 10 fired, INCLUDING PAL:
        |f-1| 0.0010  zones  3  rungs 1   |f-1| 0.0050  zones  4  rungs 0
        |f-1| 0.0017  zones  6  rungs 4   |f-1| 0.0085  zones  6  rungs 0
        |f-1| 0.0029  zones  3  rungs 0   |f-1| 0.0146  zones  9  rungs 1
        |f-1| 0.0249  zones 17  rungs 0   |f-1| 0.0427  zones 27-33 rungs 0-3, 29 above floor
    with the mechanism visible in the raw steps: at |f-1|=0.0017 the offsets walk
    [0,-1,-2,-3,-4,-4] (four rungs); at 0.0050 they walk [-3,-6,-9,-12] (steps of three, zero
    rungs); at PAL they scatter [-4,-1,-2,-1,-3,-2,-4,-5,...]. Those fixtures are 300 s long, so
    the two bounds CROSS and the band is empty -- which is why nothing fired.

    SO THE HONEST SCOPE OF THIS ARM IS "NTSC-SCALE DRIFT ON LONG FILES", NOT "the rate family".
    The one real positive (errid-27, 1001/1000 on a 62-minute file) sits squarely inside the
    band. The real PAL pair does NOT, and is caught by other arms entirely -- errid-70 reaches
    its sweep through `all_segments_below_duration_floor` on fr and
    `all_seeds_refused_by_local_baseline_guard` on en, with the coverage floor behind both. This
    arm adds reach; it is not the thing standing between a rate pair and a wrong answer. ***

    FOUR CONDITIONS, ALL REQUIRED -- BUT ONLY ONE OF THEM SEPARATES, AND SAYING SO IS THE POINT:
      rungs      enough one-quantum steps that the two FRACTIONS below mean anything. A
                 PRECONDITION for the statistic, not a separator: on 35 real alignments EIGHT
                 clear it, so it is not what keeps negatives out.
      purity     those rungs dominate the steps. ALSO NOT A SEPARATOR, measured: the largest
                 negative reads 26 rungs at purity 1.000 (errid-24/es) and another 26 at 0.963
                 (errid-99/fr), against the positive's 32 at 0.970. Kept as a cheap conservative
                 guard against a file that both drifts and is heavily edited, and because it has
                 never false-refused anything -- never as evidence.
      direction  the rungs almost all point the same way. *** THIS IS THE WHOLE DISCRIMINATION.
                 Every negative on the measured population sits at 0.500-0.538; the positive
                 sits at 0.9375; the threshold is 0.85. Drift has a sign and editing does not,
                 so this condition rests on a MECHANISM rather than on a sample -- which is the
                 only reason a single positive is an acceptable basis for it. ***
      magnitude  the implied rate deviation is large enough for some NAMED rate to explain it.

    A READER WHO TAKES THIS FOR FOUR INDEPENDENT LINES OF DEFENCE IS BEING MISLED BY THE SHAPE
    OF THE CODE, which is why the paragraph above exists. It is one line of defence with three
    cheap guards standing next to it.

    THE MAGNITUDE CONDITION IS NOT MINE AND IS NOT TUNED. It is
    `change_point_locator.RATE_SLOPE_MIN_FACTOR_DEVIATION` (5e-4), derived there as HALF the
    smallest deviation in the named rate vocabulary (1001/1000 and its reciprocal, |f-1| =
    1/1001 = 9.99e-4): a slope implying less than half of the smallest named deviation cannot be
    recognised as any named rate, so it is flatness with a good fit, not a relation. That module
    landed it on a measured case -- id 33's normalised residual fit a slope at r^2 0.57 over 64
    points implying |f-1| = 9e-7, five orders below the floor, and the gate there fired on it
    anyway until the floor was added. The locator dies in the switch; its measurement does not,
    so the constant is IMPORTED from it rather than restated, with a literal fallback.

    Returns a dict, always, `is_rate_ladder` plus every count it rests on and a `reason` in
    words. NOTHING BRANCHES ON IT unless `RATE_RELATION_SLOPE_GATE_CALIBRATED` is True.
    """
    detail = alignment.get("zones_detail") or []
    quantum_ms = alignment.get("quantum_ms") or 0.0
    offsets = [zone["offset_points"] for zone in detail]
    steps = [later - earlier for earlier, later in zip(offsets, offsets[1:])]
    nonzero = [step for step in steps if step != 0]
    floor = banded_seed_alignment.RESOLUTION_FLOOR_QUANTA
    rungs = [step for step in nonzero if abs(step) < floor]
    above_floor = [step for step in nonzero if abs(step) >= floor]
    span_points = (detail[-1]["master_points"][1] - detail[0]["master_points"][0]) if detail else 0
    span_minutes = span_points * quantum_ms / 60000.0 if quantum_ms else 0.0
    rung_fraction = (len(rungs) / len(nonzero)) if nonzero else None
    monotone_fraction = (max(sum(1 for step in rungs if step > 0),
                             sum(1 for step in rungs if step < 0)) / len(rungs)) if rungs else None
    # THE IMPLIED RATE, FROM THE LADDER ITSELF RATHER THAN FROM A REGRESSION. The offset is
    # `j - i`, so its drift per master point is `dj/di - 1`; with the candidate running at
    # `speed_ratio = master_duration / candidate_duration`, `dj/di` is `1/speed_ratio`, hence
    # `speed_ratio = 1 / (1 + drift)` and the deviation this is tested against is
    # `|speed_ratio - 1|`. The drift is taken as total rise over total run across the zones --
    # the ladder's own two ends -- not as a least-squares slope, because the whole point of
    # counting rather than fitting is that this population is a staircase and a fit through a
    # staircase is the reading the bake-off warned about.
    total_rise = (offsets[-1] - offsets[0]) if len(offsets) >= 2 else 0
    drift_per_point = (total_rise / span_points) if span_points else 0.0
    implied_ratio = 1.0 / (1.0 + drift_per_point) if (1.0 + drift_per_point) != 0 else None
    implied_deviation = None if implied_ratio is None else abs(implied_ratio - 1.0)
    signature = {
        "n_zones": len(detail),
        "n_steps": len(steps),
        "n_steps_nonzero": len(nonzero),
        "n_rungs_subfloor": len(rungs),
        "n_steps_above_floor": len(above_floor),
        "rung_fraction": None if rung_fraction is None else round(rung_fraction, 4),
        "rung_monotone_fraction": (None if monotone_fraction is None
                                   else round(monotone_fraction, 4)),
        "zones_per_minute": round(len(detail) / span_minutes, 3) if span_minutes > 0 else None,
        "implied_speed_ratio": None if implied_ratio is None else round(implied_ratio, 7),
        "implied_factor_deviation": (None if implied_deviation is None
                                     else round(implied_deviation, 7)),
        "min_factor_deviation": RATE_LADDER_MIN_FACTOR_DEVIATION,
        "is_rate_ladder": False,
        "reason": None,
    }
    if len(rungs) < LADDER_MIN_RUNGS:
        signature["reason"] = (f"{len(rungs)} one-quantum rungs, under the {LADDER_MIN_RUNGS} "
                               f"this instrument needs before it will call a ladder a ladder")
        return signature
    if rung_fraction < LADDER_MIN_RUNG_FRACTION:
        signature["reason"] = (f"one-quantum rungs are {rung_fraction:.3f} of the "
                               f"{len(nonzero)} offset changes, under {LADDER_MIN_RUNG_FRACTION} "
                               f"-- {len(above_floor)} steps clear the resolution floor, so this "
                               f"is a file with edits in it, not a file that is drifting")
        return signature
    if monotone_fraction < LADDER_MIN_RUNG_MONOTONE_FRACTION:
        signature["reason"] = (f"the {len(rungs)} rungs are only {monotone_fraction:.3f} "
                               f"one-directional, under {LADDER_MIN_RUNG_MONOTONE_FRACTION} -- "
                               f"drift has a sign and this does not")
        return signature
    if implied_deviation is None or implied_deviation < RATE_LADDER_MIN_FACTOR_DEVIATION:
        signature["reason"] = (f"the ladder implies a speed ratio of {implied_ratio}, a deviation "
                               f"of {implied_deviation} from unity -- under "
                               f"{RATE_LADDER_MIN_FACTOR_DEVIATION}, half the smallest deviation "
                               f"any NAMED rate has, so no named rate could explain it and the "
                               f"sweep would have nothing to confirm")
        return signature
    signature["is_rate_ladder"] = True
    signature["reason"] = (f"{len(rungs)} one-quantum rungs ({rung_fraction:.3f} of all offset "
                           f"changes, {monotone_fraction:.3f} of them one-directional) over "
                           f"{span_minutes:.1f} minutes, against {len(above_floor)} steps above "
                           f"the resolution floor, implying a speed ratio of {implied_ratio}")
    return signature


def similarity_gate(alignment):
    """"Similarite faible master<->candidat ?" -- answered from the alignment's OWN numbers.

    The design's section 3.6 weighs two orderings and recommends (B), align first and gate on
    what the aligner says, because the alignment costs ~4 % of the extraction both orderings
    already pay and because it yields a NAMED reason ("the aligner found no anchored run")
    instead of a bare scalar.

    TWO ARMS, AND THEY ARE NOT THE SAME KIND OF CLAIM -- which is why the return says WHICH one
    fired and the caller treats the two differently.

      `alignment_could_not_measure`  the aligner returned one of its own could-not-measure
                                     verdicts. That IS "similarite faible" in the ruling's
                                     sense: not enough agreement anywhere in the file to anchor
                                     a single run. The aligner CANNOT PROCEED, so if the sweep
                                     then finds nothing, there is nothing left to try and the
                                     pair declines.
      `rate_relation_signature`      the aligner aligned, and what it produced carries the
                                     shape of a rate relation. The aligner CAN proceed, so this
                                     arm is a SUGGESTION to ask the sweep -- and if the sweep
                                     says no, the honest continuation is the alignment we
                                     already have, NOT a refusal. See `repair()`.

    THE ASYMMETRY IS THE WHOLE SAFETY ARGUMENT FOR EVER ENABLING THE SECOND ARM. A rate arm
    whose false positive ends in a decline would trade a known good outcome for a new
    false-decline family every time it misfired on a healthy pair; a rate arm whose false
    positive costs one sweep and then carries on cannot do worse than spend time. That is what
    makes the threshold below a cost question rather than a correctness one.

    Returns `(should_sweep, reason, observations)`; `observations["gate_arm"]` names the arm.
    """
    drift_fit = alignment.get("drift_fit") or {}
    zone_fit = zone_offset_rate_signature(alignment)
    observations = {
        "verdict": alignment.get("verdict"),
        "coverage": alignment.get("master_axis_coverage_fraction"),
        "residual_fraction": alignment.get("residual_fraction"),
        "slope_points_per_point": drift_fit.get("slope_points_per_point"),
        "r_squared": drift_fit.get("r_squared"),
        "implied_step_count": drift_fit.get("implied_step_count"),
        "trace_fit_degenerate": drift_fit.get("fit_degenerate"),
        "trace_residual_rms_ms": (None if drift_fit.get("residual_rms_ms") is None
                                  else round(drift_fit["residual_rms_ms"], 4)),
        # THE ZONE-OFFSET READING, beside the trace's and never instead of it -- two instruments
        # on the same question, both logged, neither branched on until one is calibrated.
        "zone_n": zone_fit["n_zones"],
        "zone_slope_points_per_point": zone_fit["slope_points_per_point"],
        "zone_r_squared": zone_fit["r_squared"],
        "zone_residual_rms_ms": (None if zone_fit["residual_rms_ms"] is None
                                 else round(zone_fit["residual_rms_ms"], 3)),
        "zone_residual_max_ms": (None if zone_fit["residual_max_ms"] is None
                                 else round(zone_fit["residual_max_ms"], 3)),
        "zone_implied_total_drift_ms": (None if zone_fit["implied_total_drift_ms"] is None
                                        else round(zone_fit["implied_total_drift_ms"], 1)),
        "rate_arm_calibrated": RATE_RELATION_SLOPE_GATE_CALIBRATED,
    }
    if alignment.get("verdict") in ALIGNMENT_COULD_NOT_MEASURE_VERDICTS:
        observations["gate_arm"] = "alignment_could_not_measure"
        return True, f"the aligner returned {alignment['verdict']}", observations

    # ARM 2 -- COVERAGE. A success TOKEN is not a success: see MASTER_AXIS_COVERAGE_FLOOR for the
    # measured case where `single_segment_no_cut` arrived with zero zones and zero coverage on a
    # confirmed PAL pair. `None` here is could-not-measure, not zero, and is treated as low --
    # the gate's business is deciding whether to spend a sweep, and a coverage nobody could read
    # is not evidence that similarity is fine.
    #
    # SCOPE: THIS FUNCTION READS ONE COUPLE; THE PAIR'S DECISION READS THEM ALL (ADDENDUM 21.1,
    # `ensemble_similarity_gate`). A couple under the floor is still screened out, per couple, in
    # `chimeric` (`couple_screened`) -- MEASURED, errid-24 on es: couples at 0.977 / 0.306 /
    # 0.304 / 0.976 on one pair -- so a low sibling beside a healthy couple neither sends the
    # pair to the sweep nor contributes holes.
    coverage = alignment.get("master_axis_coverage_fraction")
    if coverage is None or coverage < MASTER_AXIS_COVERAGE_FLOOR:
        observations["gate_arm"] = "master_axis_coverage_below_floor"
        return True, (f"the aligner returned {alignment.get('verdict')} but its trusted zones "
                      f"cover {coverage} of the master axis, under the "
                      f"{MASTER_AXIS_COVERAGE_FLOOR} floor -- a verdict token is not a "
                      f"measurement of how much lined up"), observations

    ladder = zone_ladder_signature(alignment)
    observations.update({f"ladder_{key}": value for key, value in ladder.items()})
    if RATE_RELATION_SLOPE_GATE_CALIBRATED and ladder["is_rate_ladder"]:
        observations["gate_arm"] = "rate_relation_signature"
        return True, (f"the aligner aligned ({alignment.get('verdict')}) but its zones form a "
                      f"rate ladder: {ladder['reason']}"), observations
    observations["gate_arm"] = None
    return False, (f"the aligner anchored runs ({alignment.get('verdict')}) and its zones are "
                   f"not a rate ladder ({ladder['reason']}), so similarity is not low in the "
                   f"ruling's sense"), observations


def ensemble_similarity_gate(primed, candidate_path):
    """ADDENDUM 21.1: "le gate de similarite et la decision de taux se prennent sur l'ensemble".
    Every primed couple is read by `similarity_gate`, each reading logged; the PAIR's answer is:

      no couple is healthy (every one could not be measured or covers under the floor)
          -> sweep, on the terminal arms: nothing aligned well enough to proceed without a rate.
             The arm reported is the first couple's, as before, and every couple's is logged.
      some couple is healthy and a healthy couple's zones form a rate ladder
          -> the rate arm, on the NON-terminal `rate_relation_signature` arm (ratio 1 is one
             of its finalists: a false fire loses to 1 on the same measure).
      otherwise -> no sweep: the healthy couples carry the pair, the low ones are screened.

    "Healthy" is `similarity_gate`'s own passing arms (`None` or the ladder arm). With ONE couple
    this is exactly the old primary-couple gate. Returns `(should_sweep, prose, observations)`
    like `similarity_gate`, the observations being the deciding couple's plus the per-couple
    arms."""
    readings = []
    for couple in primed["couples"]:
        name = f"{couple[0]}x{couple[1]}"
        should, prose, observations = similarity_gate(primed["alignments"][name])
        step_result("similarity_gate_couple", candidate=candidate_path, couple=name,
                    should_sweep=should, gate_arm=observations["gate_arm"],
                    coverage=observations["coverage"], verdict=observations["verdict"])
        readings.append((name, should, prose, observations))
    arms = {name: observations["gate_arm"] for name, _s, _p, observations in readings}
    healthy = [reading for reading in readings
               if reading[3]["gate_arm"] in (None, "rate_relation_signature")]
    if not healthy:
        name, should, prose, observations = readings[0]
    else:
        ladder = [reading for reading in healthy
                  if reading[3]["gate_arm"] == "rate_relation_signature"]
        name, should, prose, observations = (ladder or healthy)[0]
    observations = dict(observations, deciding_couple=name, couple_arms=arms,
                        n_healthy_couples=len(healthy))
    return should, prose, observations


# ---------------------------------------------------------------------------
# STEP 3 -- chimeric
# ---------------------------------------------------------------------------

@repair_log.timed_phase("orchestrator", "chimeric",
                        lambda factor, language, master_obj, candidate_obj, *a, **k:
                        candidate_obj.filePath)
def chimeric(factor, language, master_obj, candidate_obj, work_dir, primed,
             sweep_gate=None, resample_routing=None, repair_deadline=None):
    """The ruling's `chimeric(speed_factor, language, master_obj, candidate_obj)`, as ADDENDUM 21
    reshaped it. Returns `(ok, cause, reason, detail)` -- the orchestrator turns that into the
    owner's boolean at one place.

    IT RECEIVES READY FINGERPRINTS AND ALIGNMENTS FOR EVERY COUPLE, AND IT NEVER RESAMPLES
    (ADDENDUM 21.6). `primed` comes from `prime_couples`: at factor 1 from the prime `repair()`
    ran before the gate, at any other factor from the re-prime `repair()` ran right after the
    rate decision (the candidate side speed-corrected there, upstream). `factor` still travels
    here because stage 4 needs it (`frame_domain`'s `time_scale` converts the resampled
    domain's positions into candidate frames) and stage 5 applies it to the delivered tracks;
    `resample_routing` is that re-prime's description, carried to the plan for its log.

    Order: per couple, zones -> holes (a couple that could not be measured, or covers under the
    floor, is screened); the multi-couple cross-check; each couple's holes put on the FILE's
    clock with its own container delays (ADDENDUM 19 b) and UNITED across couples (ADDENDUM
    21.8 -- `union_holes`, provenance logged per union hole); the frame domain; ADDENDUM 25's
    audio plan -- the millisecond walk on the reference couple (`reference_walk`), the b2 holes
    and absorbed gaps logged against it (`log_holes_against_walk`, `log_absorbed_gaps`), every
    change point placed with the video choosing inside the audio's interval
    (`audio_transitions`), head and tail placed by the audio (`audio_edges`), the geometry in
    milliseconds (`plan_geometry`); plan application.

    MEASURED BEFORE THIS SHAPE (the reason it is what it is): errid-70, the corpus's real PAL
    pair, blind, returns `all_segments_below_duration_floor` at coverage 0.000; re-primed at the
    sweep's 1001/960 the same couple returns 8 zones at coverage 0.8827 and decomposes into head
    + one interior hole of -992.5 ms + tail -- the edit the bake-off dossier documents
    independently at "environ 0.96 s" between master t=460 s and 475 s.
    """
    candidate_path = candidate_obj.filePath
    if factor is None or factor == 1:
        step_result("no_filter", candidate=candidate_path, speed_factor=factor,
                    rule="ADDENDUM_6_no_filter_without_speed_change")
    step_result("chimeric_input", candidate=candidate_path, n_couples=len(primed["couples"]),
                primed_at=primed["factor_label"],
                rule="ADDENDUM_21_6_chimeric_receives_every_couple_ready_and_never_resamples")

    couple_results = []
    for master_stream, candidate_stream in primed["couples"]:
        couple = f"{master_stream}x{candidate_stream}"
        alignment = primed["alignments"][couple]
        # THE ORCHESTRATOR BRANCHES ON `zones`, NEVER ON `cut_zones` (design section 3.4c).
        # `single_segment_no_cut` is a SUCCESS: one long aligned zone with no interior hole is a
        # fully aligned pair, and its head/tail holes may still be entirely real.
        if alignment["verdict"] in ALIGNMENT_COULD_NOT_MEASURE_VERDICTS:
            tools.dev_log(f"orchestrator: couple {couple} could not be aligned "
                          f"({alignment['verdict']}) -- recorded, and the remaining couples "
                          f"still run: one blind track is not a verdict about the pair\n")
            continue
        # THE COVERAGE FLOOR, PER COUPLE (MEASURED, errid-24 on es: 0.977 / 0.306 / 0.304 /
        # 0.976 on one pair). A couple under it is `could-not-see`: it contributes no holes and
        # no cross-check events.
        couple_coverage = alignment.get("master_axis_coverage_fraction")
        if couple_coverage is None or couple_coverage < MASTER_AXIS_COVERAGE_FLOOR:
            step_result("couple_screened", candidate=candidate_path, couple=couple,
                        verdict=alignment["verdict"], coverage=couple_coverage,
                        floor=MASTER_AXIS_COVERAGE_FLOOR,
                        reason="master_axis_coverage_below_floor",
                        rule="a_verdict_token_is_not_a_measurement_of_how_much_lined_up")
            continue

        holes = [dict(hole, quantum_ms=alignment["quantum_ms"])
                 for hole in holes_for_couple(alignment)]
        # THE STEPS TRAVEL WITH THE COUNT: a zero-step hole is the `no_cut_confirmed` candidate
        # class of ADDENDUM 3, and "9 holes" alone cannot tell that population from nine edits.
        step_result("holes", candidate=candidate_path, couple=couple, n_holes=len(holes),
                    kinds=[hole["kind"] for hole in holes],
                    steps_ms=[(round(hole["step_ms"], 1) if hole["step_ms"] is not None
                                else None) for hole in holes],
                    master_spans_s=[round(hole["master_span_seconds"], 2) for hole in holes],
                    edge_addition_s=round(edge_addition_seconds(holes), 3))
        # THE CONTAINER DELAYS, PER COUPLE, BEFORE ANY UNION AND ANY VIDEO (ADDENDUM 19 b):
        # two couples of one pair can sit on differently delayed master tracks, so each couple's
        # holes are put on the file's clock with ITS OWN delta before they are compared.
        delta_ms, master_start_ms, candidate_start_ms = couple_start_delta_ms(
            master_obj, candidate_obj, language, master_stream, candidate_stream, factor)
        scale = Fraction(factor) if factor not in (None, 1) else Fraction(1)
        fold = {"delta_ms": float(delta_ms), "master_start_ms": float(master_start_ms),
                "candidate_start_ms": float(candidate_start_ms), "scale": float(scale)}
        step_result("track_delay_fold", candidate=candidate_path, couple=couple,
                    master_start_ms=fold["master_start_ms"],
                    candidate_start_ms=fold["candidate_start_ms"], delta_ms=fold["delta_ms"],
                    rule="file_time_offset=track_offset+candidate_start*r-master_start;"
                         "file_time_position=track_position+own_start")
        couple_results.append({"couple": couple, "alignment": alignment, "holes": holes,
                               "fold": fold})

    if not couple_results:
        # EVERY couple was blind, or every couple that was not blind covered too little to be
        # read. The token says WHICH of the two, so the ledger can tell them apart.
        alignments = primed["alignments"]
        names = [f"{m}x{c}" for m, c in primed["couples"]]
        verdicts = {alignments[name]["verdict"] for name in names}
        measured = verdicts & set(banded_seed_alignment.MEASURED_VERDICTS)
        if measured:
            cause = "alignment_coverage_below_floor"
        else:
            cause = ("alignment_degenerate_input"
                     if banded_seed_alignment.VERDICT_DEGENERATE_INPUT in verdicts
                     else "alignment_all_seeds_refused"
                     if banded_seed_alignment.VERDICT_ALL_SEEDS_REFUSED in verdicts
                     else "alignment_segments_below_duration_floor"
                     if banded_seed_alignment.VERDICT_ALL_SEGMENTS_BELOW_DURATION_FLOOR in verdicts
                     else "alignment_no_anchored_runs")
        coverages = sorted(round(alignments[name].get("master_axis_coverage_fraction") or 0.0, 4)
                           for name in names)
        return False, cause, (
            f"no couple of {language} produced a usable alignment; the aligner reported "
            f"{sorted(verdicts)} across {len(names)} couples, covering {coverages} of the "
            f"master axis against a floor of {MASTER_AXIS_COVERAGE_FLOOR}"), None

    step_launch("cross_verify", candidate=candidate_path, n_couples=len(couple_results))
    report = cross_verify_couples(couple_results)
    log_cross_verification(candidate_path, report)
    if not report["agree"]:
        return False, "intercouple_disagreement", (
            f"{len(report['disagreements'])} of {len(report['clusters'])} event clusters "
            f"disagree across {report['n_couples']} couples of {language}; every couple's "
            f"position, step, quantum, residual and coverage is in the report"), report

    # THE UNION (ADDENDUM 21.8) -- no couple pilots. The REFERENCE couple below is a different
    # role: the master track `apply_plan` correlates the comparison language's candidate tracks
    # against, and the couple whose longest zone centres a hole-free plan's refinement. It is the
    # first usable couple in `enumerate_couples` order, and it is logged as that.
    holes = union_holes([[hole_on_file_clock(hole, record["couple"], record["fold"])
                          for hole in record["holes"]] for record in couple_results])
    for index, hole in enumerate(holes):
        step_result("union_hole", candidate=candidate_path, hole=index, kind=hole["kind"],
                    master_ms=[round(hole["master_ms"][0], 2), round(hole["master_ms"][1], 2)],
                    offsets_ms=[None if hole["offset_before_ms"] is None
                                else round(hole["offset_before_ms"], 3),
                                None if hole["offset_after_ms"] is None
                                else round(hole["offset_after_ms"], 3)],
                    step_ms=(None if hole["step_ms"] is None else round(hole["step_ms"], 3)),
                    union_of=hole["union_of"], offset_sources=hole["offset_sources"],
                    members=hole["members"])
    reference = couple_results[0]
    step_result("plan_shape", candidate=candidate_path, hole_source="union_of_all_couples",
                n_couples=len(couple_results), reference_couple=reference["couple"],
                per_couple_holes={record["couple"]: len(record["holes"])
                                  for record in couple_results},
                n_holes=len(holes), edge_addition_s=round(edge_addition_seconds(holes), 3))

    if any(hole["kind"] == "spans_whole_file" for hole in holes):
        return False, "alignment_no_anchored_runs", (
            f"after the union of "
            f"{len(couple_results)} couple(s), one hole spans the whole file -- no aligned zone "
            f"survived long enough to anchor either end, so there is no head, interior or tail "
            f"to resolve"), None

    # THE BUDGET IS PER COUPLE, as its measurement was (see MAX_HOLES_PER_COUPLE): the union
    # of several couples may hold more regions than any one couple saw, and that sum is not a
    # fragmentation of the pair.
    over = [(record["couple"], len(record["holes"])) for record in couple_results
            if len(record["holes"]) > MAX_HOLES_PER_COUPLE]
    if over:
        return False, "hole_count_exceeds_resolver_budget", (
            f"couple(s) {over} decompose into more holes than the budget of "
            f"{MAX_HOLES_PER_COUPLE}; a pair that fragments this far is not one this instrument "
            f"has measured itself able to reconstruct"), None

    # ADDENDUM 5's good news, named: NO hole at all -- completely compatible audios.
    if not holes:
        step_result("holes", candidate=candidate_path, couple="union", n_holes=0,
                    verdict="audios_fully_compatible_offset_only")

    # THE FRAME DOMAIN, ONCE PER PAIR, HOLES OR NOT: exact rational grids for both files, the
    # master's timeline, and the rate relation the candidate's alignment milliseconds are in.
    step_launch("frame_domain", candidate=candidate_path)
    domain, domain_reason = frame_domain(master_obj, candidate_obj, factor)
    step_result("frame_domain", candidate=candidate_path, reason=domain_reason,
                **({} if domain is None else {
                    "master_rate": f"{domain['master_rate'].numerator}/"
                                   f"{domain['master_rate'].denominator}",
                    "master_rate_source": domain["master_rate_source"],
                    "candidate_rate": f"{domain['candidate_rate'].numerator}/"
                                      f"{domain['candidate_rate'].denominator}",
                    "candidate_rate_source": domain["candidate_rate_source"],
                    "time_scale": (None if domain["time_scale"] is None
                                   else f"{domain['time_scale'].numerator}/"
                                        f"{domain['time_scale'].denominator}"),
                    "audio_effective_ratio": (None if resample_routing is None
                                              else resample_routing["effective_ratio_str"]),
                    "video_rate_ratio": f"{domain['video_rate_ratio'].numerator}/"
                                        f"{domain['video_rate_ratio'].denominator}",
                    "video_ratio_matches_speed_factor":
                        domain["video_ratio_matches_speed_factor"],
                    "master_timeline_ms": float(domain["master_timeline_ms"]),
                    "candidate_equivalent_duration_ms": (
                        None if domain["candidate_equivalent_duration_ms"] is None
                        else float(domain["candidate_equivalent_duration_ms"]))}))
    if domain is None and holes:
        return False, "hole_resolution_declined", (
            f"the pair decomposed into {len(holes)} hole(s) but its frame domain could not "
            f"be measured ({domain_reason}) -- a boundary that is not a frame on an exact "
            f"grid is not a boundary, so none was sought"), None
    if domain is None:
        return False, "frame_domain_unmeasured", (
            f"the pair carries no hole, but its frame domain could not be measured "
            f"({domain_reason}): the plan's timeline end and its grid are unknown, so no "
            f"piece can be placed"), None
    domain["repair_deadline"] = repair_deadline
    insane = hole_sanity(holes, domain, candidate_path)
    if insane is not None:
        return False, insane[0], insane[1], None

    # ---- ADDENDUM 25: THE AUDIO BOUNDS, THE VIDEO PINS ------------------------
    # The millisecond walk on the reference couple fixes every offset, every step, every fill
    # width and every interval a cut may lie in; the video only chooses inside those intervals.
    speed_ratio = None if factor in (None, 1) else _decimal(Fraction(factor))
    try:
        walk, walk_reason = reference_walk(reference, holes, master_obj, candidate_obj,
                                           language, speed_ratio, candidate_path,
                                           deadline=repair_deadline,
                                           engine=(resample_routing or {}).get("filter_name",
                                                                               "asetrate"))
    except Exception as error:                                           # noqa: BLE001
        if getattr(error, "cause", None) != "repair_budget_exceeded":
            raise
        # ADDENDUM 26 RULING (2026-09-25): THE WALK IS INSIDE THE BUDGET -- a walk the budget
        # stops declines the repair by that name, with what was placed before it.
        log_partial_plan(candidate_path, "repair_budget_exceeded",
                         [("holes", "b2", [(h["kind"], h["master_ms"]) for h in holes]),
                          ("audio_walk", "stopped_by_budget", str(error)[:200])])
        return False, "repair_budget_exceeded", (
            f"the repair's budget ran out during the audio walk ({error}) -- the partial plan "
            f"is logged; the file comes back next wave"), None
    if walk is None:
        return False, "audio_walk_unavailable", (
            f"the millisecond walk on the reference couple {reference['couple']} could not "
            f"measure the pair ({walk_reason}) -- no offset, step or fill can be placed "
            f"without it"), None
    if repair_deadline is not None and time.monotonic() > repair_deadline:
        log_partial_plan(candidate_path, "repair_budget_exceeded",
                         [("audio_walk", "levels", [lv["off_ms"] for lv in walk["levels"]])])
        return False, "repair_budget_exceeded", (
            f"the repair's budget ran out after the audio walk -- the "
            f"partial plan is logged; the file comes back next wave"), None
    log_holes_against_walk(holes, walk, candidate_path)
    log_absorbed_gaps(couple_results, holes, walk, candidate_path)
    transitions, refusal = audio_transitions(walk, reference, domain, master_obj, candidate_obj,
                                             work_dir, candidate_path)
    if transitions is None:
        return False, refusal[0], refusal[1], None
    head_end_s, tail_start_s, refusal = audio_edges(walk, holes, domain, master_obj,
                                                    candidate_obj, work_dir, candidate_path)
    if refusal is not None:
        return False, refusal[0], refusal[1], None
    zones, fills, geometry_failure = plan_geometry(transitions, head_end_s, tail_start_s,
                                                   domain, walk)
    if zones is None:
        return False, "audio_transitions_overlap", (
            f"the audio's transitions do not tile the master timeline: {geometry_failure}"), None
    if not zones:
        return False, "plan_reads_no_candidate_content", (
            "the audio edges leave no candidate content on the master timeline -- a plan that "
            "reads nothing from the candidate is the master, not a repair"), None
    if repair_deadline is not None and time.monotonic() > repair_deadline:
        log_partial_plan(candidate_path, "repair_budget_exceeded",
                         [(f"change_point_{t['change_point']}", t["decision"], t["at_s"],
                           t["fill_s"]) for t in transitions]
                         + [("head", "placed", head_end_s), ("tail", "placed", tail_start_s)])
        return False, "repair_budget_exceeded", (
            f"the repair's budget ran out before the plan's application -- "
            f"the partial plan is logged; the file comes back next wave"), None
    head_written_s, tail_written_s = written_edge_seconds(fills, walk["master_audio_end_s"])
    tagged, tag_reason = tag_decision(len(transitions), head_written_s + tail_written_s)
    step_result("plan_shape_resolved", candidate=candidate_path,
                n_transitions=len(transitions),
                decisions=[t["decision"] for t in transitions],
                head_end_s=head_end_s, tail_start_s=tail_start_s,
                head_written_s=round(head_written_s, 3), tail_written_s=round(tail_written_s, 3),
                chimeric_tag=tagged, chimeric_tag_reason=tag_reason.replace(" ", "_"))
    master_stream, candidate_stream = reference["couple"].split("x")
    ok, cause, reason = apply_plan(candidate_path, {
        "zones": zones, "fills": fills, "walk": walk, "head_written_s": head_written_s,
        "tail_written_s": tail_written_s}, factor, master_obj, candidate_obj, {
        "language": language, "work_dir": work_dir, "domain": domain,
        "quantum_ms": reference["alignment"]["quantum_ms"],
        "master_stream": master_stream, "candidate_stream": candidate_stream,
        "resample_routing": resample_routing,
        "sweep_gate": sweep_gate, "tagged": tagged, "tag_reason": tag_reason})
    return ok, cause, reason, None


# ---------------------------------------------------------------------------
# THE ORCHESTRATOR
# ---------------------------------------------------------------------------

@repair_log.timed_phase("orchestrator", "repair",
                        lambda master_obj, candidate_obj, *a, **k: candidate_obj.filePath)
def repair(master_obj, candidate_obj, comparison_language, work_root=None,
           master_intertrack_cache=None):
    """The owner's `repair(master_obj, candidate_obj, comparison_language, ...)`. Returns a BOOL.

    Flat, by construction: each step is launched here, returns a result, and this function
    arranges the results. No step calls another.

        1  Is the master desynchronised against ITSELF on the comparison language? Yes -> return.
        2  Is master<->candidate similarity low? Yes -> can a resample raise it? The sweep
           returns a FACTOR or None; None -> return.
        3  Chimeric at that factor (1 by default). A positive return with a plan -> apply it and
           return a positive gain for this file. Otherwise -> return, with the logs.

    `master_intertrack_cache` is the caller's per-master, per-language memo, passed in rather
    than held as module state for the reason the existing chain already documents: module state
    would survive between two calls on TWO DIFFERENT MASTERS and hand the first one's verdict to
    the second.
    """
    candidate_path = candidate_obj.filePath
    # THE REPAIR'S BUDGET (ADDENDUM 26.3), a `time.monotonic()` instant checked between steps and
    # handed to the hole resolver, which never runs a hole past it.
    repair_budget_s, budget_video_s = repair_budget_seconds(master_obj)
    repair_deadline = time.monotonic() + repair_budget_s
    tools.log_always(f"orchestrator: repair_budget budget_s={repair_budget_s} "
                     f"master_video_s={budget_video_s} for {candidate_path}\n")
    if master_intertrack_cache is None:
        master_intertrack_cache = {}
    work_dir = work_root or path.join(tools.tmpFolder, "repair", "orchestrator")
    tools.make_dirs(work_dir)
    _PLAN_LINE_MARK[candidate_path] = len(tools.logs)
    tools.dev_log(f"orchestrator: repair starting on {candidate_path} "
                  f"master={master_obj.filePath} language={comparison_language} "
                  f"hole_merge_window_s={HOLE_MERGE_WINDOW_SECONDS} "
                  f"({_HOLE_MERGE_SOURCE})\n")

    # ---- STEP 1: does the master agree with itself? -------------------------
    step_launch("master_self_check", candidate=candidate_path,
                master=master_obj.filePath, language=comparison_language)
    try:
        import merge_video_repair
        verdict = merge_video_repair.master_intertrack_verdict(
            master_obj, comparison_language, master_intertrack_cache)
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"orchestrator: master_intertrack_verdict unavailable "
                      f"({type(error).__name__}: {error}) -- no verdict; None means NOT "
                      f"MEASURED, never healthy\n")
        verdict = None
    step_result("master_self_check", candidate=candidate_path,
                verdict=(verdict or {}).get("verdict") if verdict else None,
                inert=(verdict or {}).get("inert") if verdict else None)
    if verdict is not None and verdict.get("verdict") is not None:
        _plan_line("none", candidate_path, step="master_self_check",
                   cause=verdict["verdict"])
        return _terminal(candidate_path, "declined", verdict["verdict"], verdict["reason"],
                         detail={"verdict": verdict["verdict"],
                                 "master_intertrack": verdict})

    # ---- STEP 2: low similarity? can a resample raise it? -------------------
    # Gated on the ALIGNER'S OWN OUTPUT (design section 3.6, ordering B), over EVERY couple
    # (ADDENDUM 21.1): all of them are primed here and handed to step 3 rather than recomputed.
    factor = 1
    sweep_gate = None
    couples = enumerate_couples(master_obj, candidate_obj, comparison_language)
    step_result("enumerate_couples", candidate=candidate_path,
                language=comparison_language, n_couples=len(couples), couples=couples)
    if not couples:
        _plan_line("none", candidate_path, step="enumerate_couples",
                   cause="no_stream_for_comparison_language")
        return _terminal(candidate_path, "no_plan", "no_stream_for_comparison_language",
                         f"neither side offers a pair of {comparison_language} audio streams "
                         f"to compare")

    primed = {"couples": couples, "fingerprints": {}, "alignments": {}, "factor_label": "1",
              "sample_rate": None}
    step_launch("prime", candidate=candidate_path, n_couples=len(couples))
    prime_ok, prime_cause, prime_reason = prime_couples(
        master_obj, candidate_obj, comparison_language, work_dir, primed)
    step_result("prime", candidate=candidate_path, ok=prime_ok, cause=prime_cause)
    if not prime_ok:
        _drop_rate_wav(primed)
        _plan_line("none", candidate_path, step="prime", cause=prime_cause)
        return _terminal(candidate_path, "no_plan", prime_cause, prime_reason)
    timeline_ms = _video_duration_ms(master_obj)
    if timeline_ms is not None:
        ends = tail_couples(primed)
        tail_verdict, readings = tail_content_verdict(ends, float(timeline_ms) / 1000.0)
        step_result("tail_content", candidate=candidate_path, verdict=tail_verdict,
                    readings=readings, min_one_sided_s=TAIL_ONE_SIDED_MIN_S)
        if tail_verdict == "master_cut_short":
            _drop_rate_wav(primed)
            _plan_line("none", candidate_path, step="tail_content", cause="master_cut_short")
            return _terminal(candidate_path, "declined", "master_cut_short", (
                f"the master's {comparison_language} content ends before the candidate's and "
                f"does not resume: per couple (couple, verdict, master content after the last "
                f"common instant s, candidate content after it s) {readings}; measured ends "
                + "; ".join(f"{c['couple']} master {c['master_content_end_s']} s / candidate "
                            f"{c['candidate_content_end_s']} s after the last common instant "
                            f"{c['last_common_master_s']} s" for c in ends)
                + f" of a {round(float(timeline_ms) / 1000.0, 3)} s master video -- the master "
                  f"cannot give the candidate its end (ADDENDUM 26.9)"))
        if tail_verdict == "candidate_short":
            tools.log_always(f"repair: candidate_short_tail readings={readings} "
                             f"-- the candidate's content ends >= {TAIL_ONE_SIDED_MIN_S} s "
                             f"before the master's: the master fills the tail, no size cap "
                             f"(ADDENDUM 26.9) for {candidate_path}\n")
    if time.monotonic() > repair_deadline:
        _drop_rate_wav(primed)
        return _budget_terminal(candidate_path, "prime", repair_budget_s)

    step_launch("similarity_gate", candidate=candidate_path, n_couples=len(couples))
    should_sweep, gate_prose, observations = ensemble_similarity_gate(primed, candidate_path)
    step_result("similarity_gate", candidate=candidate_path, should_sweep=should_sweep,
                **{key: value for key, value in observations.items()})

    # ---- STEP 2a: THE RATE ARM (ADDENDUM 15/21.4/30/30.5) --------------------
    # Armed by the similarity gate (low similarity: every named ratio in both engines is tried
    # BEFORE any low-similarity conclusion), by a fast drift on any couple (a linear drift must
    # never reach the hole resolver as 131-143 holes) or by declared frame rates that name a
    # rate (25 vs 23.976 ...). The drift and the declared rate only NAME the first finalists;
    # the alignment of each finalist decides.
    import rate_direction
    engine = None
    drift_named = []
    for couple_name, alignment in primed["alignments"].items():
        drift = rate_direction.fast_drift_signature(alignment.get("zones_detail") or [],
                                                    alignment.get("quantum_ms"))
        step_result("fast_drift", candidate=candidate_path, couple=couple_name,
                    fires=drift["fires"], implied_ratio=drift["implied_ratio_fit"],
                    named=[f"{r.numerator}/{r.denominator}"
                           for r in drift["named_rate_candidates"]], reason=drift["reason"])
        drift_named += [r for r in drift["named_rate_candidates"] if r not in drift_named]
    declared = rate_direction.declared_named_ratio(master_obj.filePath, candidate_obj.filePath)
    armed_by = ("similarity_gate" if should_sweep else "fast_drift" if drift_named
                else "declared_frame_rate" if declared is not None else None)
    first_ratios = rate_direction.first_finalists(declared, drift_named)
    step_result("rate_arm_armed", candidate=candidate_path, armed_by=armed_by,
                declared=(None if declared is None
                          else f"{declared.numerator}/{declared.denominator}"),
                first_finalists=[f"{r.numerator}/{r.denominator}" for r in first_ratios])
    if armed_by is None:
        _drop_rate_wav(primed)
    else:
        step_launch("rate_arm", candidate=candidate_path, language=comparison_language,
                    armed_by=armed_by)
        winner, engine, sweep_gate, sweep_cause = rate_arm(
            primed, first_ratios, work_dir, candidate_path, deadline=repair_deadline)
        step_result("rate_arm", candidate=candidate_path, armed_by=armed_by,
                    factor=(None if winner is None
                            else f"{winner.numerator}/{winner.denominator}"),
                    engine=engine, cause=sweep_cause, rounds=sweep_gate.get("rounds"),
                    span_coverage=sweep_gate.get("span_coverage"),
                    margin=sweep_gate.get("margin"),
                    finalists=[(row["ratio"], row["engine"], row["span_coverage"], row["zones"],
                                row["ladder"]) for row in sweep_gate["rows"]])
        if sweep_cause == "repair_budget_exceeded":
            log_partial_plan(candidate_path, "repair_budget_exceeded",
                             [("rate_arm", "stopped_by_budget", len(sweep_gate["rows"]))])
            return _budget_terminal(candidate_path, "rate_arm", repair_budget_s)
        # THE ARM SUGGESTS UNLESS THE SIMILARITY GATE ASKED: a drift, a declared rate or a
        # rate-ladder reading is a hint; their "no" continues at factor 1 on the alignment
        # already measured. Only low similarity with every named ratio refused in both engines
        # is `similarity_unrecoverable_by_resample` -- and 1 winning is never a refusal.
        terminal = (winner is None and armed_by == "similarity_gate"
                    and observations.get("gate_arm") != "rate_relation_signature"
                    and sweep_cause != "rate_arm_unity_wins")
        if terminal:
            _plan_line("none", candidate_path, step="rate_arm", cause=sweep_cause)
            return _terminal(
                candidate_path, "no_plan", sweep_cause,
                f"mean similarity is low ({gate_prose}) and no named ratio raises it in either "
                f"engine: {[(r['ratio'], r['engine'], r['span_coverage']) for r in sweep_gate['rows']]}",
                detail={"resample_gate": sweep_gate})
        if winner is None:
            tools.dev_log(f"orchestrator: the rate arm ({armed_by}) found no rate for "
                          f"{candidate_path} ({sweep_cause}); the pair CONTINUES at speed_factor 1 "
                          f"on the alignment already measured\n")
        else:
            factor = winner

    # ---- STEP 2b: the re-prime at a confirmed factor (ADDENDUM 21.6) ---------
    # Every couple's candidate side, speed-corrected, re-fingerprinted and re-aligned HERE, so
    # that chimeric receives them ready and never resamples. Never at factor 1 (ADDENDUM 6).
    resample_routing = None
    if factor != 1:
        factor_label = (f"{factor.numerator}/{factor.denominator}"
                        if isinstance(factor, Fraction) else factor)
        step_launch("rate_reprime", candidate=candidate_path, speed_factor=factor_label)
        resample_routing, reprime_cause = rate_resample_routing(
            factor, engine, master_obj, candidate_obj, comparison_language, work_dir,
            primed["sample_rate"])
        if resample_routing is None:
            step_result("rate_reprime", candidate=candidate_path, ok=False, cause=reprime_cause)
            _plan_line("none", candidate_path, step="rate_reprime", cause=reprime_cause)
            return _terminal(
                candidate_path, "no_plan", reprime_cause,
                f"the pair carries a confirmed rate relation ({factor_label}) but the "
                f"speed-corrected candidate that would let the aligner measure across it could "
                f"not be built ({reprime_cause}) -- no measurement was made at that factor")
        step_result("rate_reprime", candidate=candidate_path, ok=True,
                    side=resample_routing["side"],
                    filter=resample_routing["filter_name"],
                    requested_ratio=resample_routing["requested_ratio"],
                    effective_ratio=resample_routing["effective_ratio_str"],
                    tag_factor=resample_routing["tag_factor"],
                    source_sample_rate=resample_routing["source_sample_rate"],
                    asetrate_target=resample_routing["asetrate_target"],
                    intermediate_rate=resample_routing["intermediate_rate"],
                    filter_chain=resample_routing["filter_chain"],
                    pitch_measured_ratio=resample_routing["pitch_measured_ratio"],
                    pitch_peak=resample_routing["pitch_peak"],
                    pitch_refusal=resample_routing["pitch_refusal"],
                    pitch_window_s=resample_routing["pitch_window_seconds"],
                    pitch_test_discriminating=resample_routing["pitch_test_discriminating"],
                    pitch_tolerance_band=resample_routing["pitch_tolerance_band"],
                    inverting_case_detector=resample_routing["inverting_case_detector"],
                    inverting_case_observation=resample_routing["inverting_case_observation"],
                    rule=resample_routing["rule"])
        tools.dev_log(f"orchestrator: rate_reprime routing for {candidate_path}: "
                      f"{resample_routing['route_reason']}\n")
        prime_ok, prime_cause, prime_reason = prime_couples(
            master_obj, candidate_obj, comparison_language, work_dir, primed,
            resample_routing=resample_routing)
        if not prime_ok:
            _plan_line("none", candidate_path, step="rate_reprime", cause=prime_cause)
            return _terminal(candidate_path, "no_plan", prime_cause, prime_reason)
    if time.monotonic() > repair_deadline:
        return _budget_terminal(candidate_path, "rate_decision", repair_budget_s)

    # ---- STEP 3: chimeric ---------------------------------------------------
    step_launch("chimeric", candidate=candidate_path, language=comparison_language,
                speed_factor=(f"{factor.numerator}/{factor.denominator}"
                               if isinstance(factor, Fraction) else factor))
    ok, cause, reason, detail = chimeric(factor, comparison_language, master_obj,
                                         candidate_obj, work_dir, primed,
                                         sweep_gate=sweep_gate,
                                         resample_routing=resample_routing,
                                         repair_deadline=repair_deadline)
    step_result("chimeric", candidate=candidate_path, ok=ok, cause=cause)
    if ok:
        # THE `repaired` TERMINAL IS ALREADY WRITTEN, ONCE, by `apply_plan` through `record()`
        # -- the line that used to be emitted here as well would make one repair read as two.
        _plan_line("chimeric", candidate_path, step="chimeric",
                   speed_factor=(f"{factor.numerator}/{factor.denominator}"
                                  if isinstance(factor, Fraction) else factor))
        return True
    _plan_line("none", candidate_path, step="chimeric", cause=cause)
    return _terminal(candidate_path, "no_plan", cause, reason,
                     detail={"cross_verification": detail} if detail else None)


def align_fingerprints(fp_master, quantum_master, duration_master, fp_candidate,
                       quantum_candidate, duration_candidate):
    """`b2_align` as the prime calls it -- one place, so the rate arm's finalists are aligned
    exactly like the prime's couples (ADDENDUM 30.5: the winner is read on re-fingerprinting)."""
    return banded_seed_alignment.b2_align(
        fp_master, fp_candidate, quantum_master,
        candidate_quantum_ms=quantum_candidate,
        duration_diff_ms=abs(duration_master - duration_candidate) * 1000.0,
        signed_duration_diff_ms=(duration_candidate - duration_master) * 1000.0,
        shorter_duration_ms=min(duration_master, duration_candidate) * 1000.0,
        deadline=time.monotonic() + ALIGNMENT_BUDGET_S)


def prime_couples(master_obj, candidate_obj, language, work_dir, primed, resample_routing=None):
    """THE PRIME (ADDENDUM 21.1): every couple of the comparison language, fingerprinted on both
    sides and aligned with `b2_align`, BEFORE the similarity gate -- so the gate and the rate
    decision read all of them, and chimeric receives them ready (ADDENDUM 21.6).

    `primed` = {"couples", "fingerprints", "alignments", "factor_label", "sample_rate"} is filled
    in place. Returns `(ok, cause, reason)`.

    WITH `resample_routing` IT IS THE RE-PRIME AT A CONFIRMED FACTOR, and what it keeps is
    decided by what the correction touches: the candidate's points, point count and quantum all
    change under it, so every candidate fingerprint and EVERY alignment is dropped by name
    (`reprime` step); the MASTER fingerprints are kept, because nothing was applied to them --
    one whole-track decode per master track saved (the design measured 7-15 s each, the dominant
    cost of the step). The filter rides on the candidate side only, and the EFFECTIVE ratio sets
    the corrected length fpcalc must read (see `fingerprint_track`)."""
    candidate_path = candidate_obj.filePath
    sample_rate = primed.get("sample_rate") or comparison_sample_rate(master_obj, candidate_obj,
                                                                      language)
    primed["sample_rate"] = sample_rate
    if resample_routing is not None:
        dropped_fingerprints = sorted(key[1] for key in primed["fingerprints"]
                                      if key[0] != "master")
        primed["fingerprints"] = {key: value for key, value in primed["fingerprints"].items()
                                  if key[0] == "master"}
        dropped_alignments = sorted(primed["alignments"])
        primed["alignments"] = {}
        primed["factor_label"] = resample_routing["requested_ratio"]
        step_result("reprime", candidate=candidate_path,
                    kept_master_fingerprints=sorted(key[1] for key in primed["fingerprints"]),
                    dropped_candidate_fingerprints=dropped_fingerprints,
                    dropped_alignments=dropped_alignments,
                    reason="candidate_fingerprints_stale_under_the_confirmed_factor")
    for master_stream, candidate_stream in primed["couples"]:
        for side, video_obj, stream in (("master", master_obj, master_stream),
                                         ("candidate", candidate_obj, candidate_stream)):
            key = (side, stream)
            if key in primed["fingerprints"]:
                continue
            duration = _track_duration_seconds(video_obj, language, stream)
            if duration is None:
                return (False, "track_duration_unmeasurable",
                        f"the {side} {language} stream {stream} carries no readable duration, "
                        f"so there is no length to fingerprint it over")
            # THE MASTER'S FINGERPRINT STOPS AT ITS VIDEO (ADDENDUM 26.2): the plan's timeline IS
            # the master video, so audio past it is never read -- and fingerprinting it cost id
            # 691 (audio 12,799 s over a 5,997 s video, the programme twice) 1,037 s of alignment.
            video_ms = _video_duration_ms(master_obj) if side == "master" else None
            if video_ms is not None and duration > float(video_ms) / 1000.0:
                step_result("master_audio_overruns_video", candidate=candidate_path,
                            stream=stream, audio_s=round(duration, 3),
                            video_s=round(float(video_ms) / 1000.0, 3),
                            rule="ADDENDUM_26_2_fingerprint_stops_at_the_master_video")
                duration = float(video_ms) / 1000.0
            track_filter = (resample_routing["filter_chain"]
                            if resample_routing is not None and side == "candidate" else None)
            corrected_duration = (duration * float(resample_routing["effective_ratio"])
                                  if track_filter else duration)
            step_launch("fingerprint", candidate=candidate_path, side=side, stream=stream,
                        duration_s=round(duration, 3), sample_rate=sample_rate,
                        audio_filter=track_filter,
                        corrected_duration_s=(round(corrected_duration, 3)
                                              if track_filter else None))
            started = time.time()
            measures = {}
            keep = (resample_routing is None and side == "candidate"
                    and (master_stream, candidate_stream) == primed["couples"][0])
            try:
                points, quantum_ms = fingerprint_track(
                    video_obj, language, stream, side, work_dir, sample_rate, duration,
                    audio_filter=track_filter, output_duration_seconds=corrected_duration,
                    measures=measures, keep_wav=keep)
            except tools.decoder_timeout as error:
                return (False, "decoder_timeout",
                        f"the {side} {language} stream {stream} extraction ran past its bound "
                        f"({error}) -- a statement about the tool on this host")
            step_result("fingerprint", candidate=candidate_path, side=side, stream=stream,
                        n_points=len(points) if points else 0,
                        quantum_ms=round(quantum_ms, 4) if quantum_ms else None,
                        resampled=bool(track_filter),
                        seconds=round(time.time() - started, 2))
            if points is None:
                return (False, "fingerprinting_raised",
                        f"the {side} {language} stream {stream} could not be extracted or "
                        f"fingerprinted")
            primed["fingerprints"][key] = (points, quantum_ms, corrected_duration)
            if measures.get("wav"):
                primed["rate_wav"] = {"path": measures["wav"], "stream": stream,
                                      "duration_s": corrected_duration, "rate": sample_rate}
            content_end = measures.get("content_end_s")
            full_s = _track_duration_seconds(video_obj, language, stream)
            if side == "master" and full_s is not None and full_s > duration:
                # the track runs past the video (26.2 stopped the WAV there): its own end
                content_end = (overrun_content_end_s(video_obj, stream, full_s, duration)
                               or content_end)
            primed.setdefault("content_end", {})[key] = content_end
            step_result("content_end", candidate=candidate_path, side=side, stream=stream,
                        content_end_s=None if content_end is None else round(content_end, 3),
                        track_s=None if full_s is None else round(full_s, 3))

        name = f"{master_stream}x{candidate_stream}"
        fp_master, quantum_master, duration_master = primed["fingerprints"][
            ("master", master_stream)]
        fp_candidate, quantum_candidate, duration_candidate = primed["fingerprints"][
            ("candidate", candidate_stream)]
        step_launch("align", candidate=candidate_path, couple=name, n_master=len(fp_master),
                    n_candidate=len(fp_candidate))
        started = time.time()
        alignment = align_fingerprints(fp_master, quantum_master, duration_master,
                                       fp_candidate, quantum_candidate, duration_candidate)
        alignment["alignment_seconds"] = time.time() - started
        if alignment["verdict"] == banded_seed_alignment.VERDICT_ALIGNMENT_BUDGET_EXCEEDED:
            step_result("align", candidate=candidate_path, couple=name,
                        verdict=alignment["verdict"],
                        seeds_extended=alignment.get("seeds_extended"),
                        seeds_total=alignment.get("seeds_total"),
                        seconds=round(alignment["alignment_seconds"], 2))
            return (False, "alignment_budget_exceeded",
                    f"couple {name} did not align within {ALIGNMENT_BUDGET_S} s: "
                    f"{alignment.get('seeds_extended')} of {alignment.get('seeds_total')} seeds "
                    f"extended ({len(fp_master)} x {len(fp_candidate)} points) -- a statement "
                    f"about this run's cost, the file comes back next wave")
        primed["alignments"][name] = alignment
        step_result("align", candidate=candidate_path, couple=name,
                    verdict=alignment["verdict"], n_zones=len(alignment.get("zones") or []),
                    n_cut_zones=len(alignment.get("cut_zones") or []),
                    overlaps_resolved=alignment.get("segments_overlap_resolved"),
                    admitted_self_evident=alignment.get("admitted_self_evident"),
                    coverage=alignment.get("master_axis_coverage_fraction"),
                    residual_fraction=alignment.get("residual_fraction"),
                    seconds=round(alignment["alignment_seconds"], 2))
    return True, None, None
