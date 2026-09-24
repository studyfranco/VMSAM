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
    landed   step 2  the similarity gate + speed sweep -- reuses `merge_video_repair
                                                          .run_speed_sweep`, factor-or-None
                                                          adapted HERE (see `speed_factor`)
    landed   step 3a full-file fingerprints per track  -- `audioCorrelation` + `audio_extract`
    landed   step 3b the sequence alignment per couple -- `banded_seed_alignment.b2_align`
    landed   step 3c zones -> holes -> <10 s merge     -- this module
    landed   step 3d the multi-couple cross-check      -- this module
    landed   step 3e the comparison resample           -- `merge_video_resample
                                                          .build_speed_filter_chain` behind the
                                                          pitch layer's own reading; applied on
                                                          the candidate's comparison extraction
                                                          only -- the fingerprint WAV is deleted
                                                          after one pass; DELIVERING resampled
                                                          audio is stage 5's (ADDENDUM 8)
    landed   step 4  frame-exact hole resolution       -- `scene_anchor.locate_scene_anchors`
                                                          (interior, two anchors) and
                                                          `.locate_edge_boundary` (head/tail,
                                                          one anchor), sequential; see
                                                          `resolve_hole`
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
`tools.dev_log` (gated on `tools.dev`) -- except the inter-couple disagreement of step 3, which
the ruling says "SORTENT MEME A tools.dev=false" and which therefore goes through
`tools.log_always`, and the terminal per-candidate verdict, which goes through
`merge_video_repair.record` (already `log_always`) so the ledger keeps reading one line shape.
"""
from decimal import Decimal
from fractions import Fraction
import math
from os import path, remove
import subprocess
import time

import audioCorrelation
import audio_extract
import banded_seed_alignment
import tools

MODALITY = "repair_orchestrator"

# ---------------------------------------------------------------------------
# DERIVED CONSTANTS -- each one states what it is derived FROM. A constant with
# no derivation is a tuning knob, and a repair conditioned on a tuning knob is
# not a repair.
# ---------------------------------------------------------------------------

# THE <10 s HOLE MERGE (ruling, chimeric step 4: "les trous separes de MOINS DE 10 s se
# FUSIONNENT"). This is NOT a free number and must not be re-tuned as one: it is the frame-exact
# resolver's OWN SEARCH REACH, `scene_anchor.SCENE_SEARCH_WINDOW_SECONDS_DEFAULT = 10.0`. Two
# holes closer together than that reach have OVERLAPPING +/-10 s scene searches and can select
# the SAME scene cut as two different anchors -- i.e. the merge rule exists because the resolver
# cannot tell them apart, not because 10 s is a nice round number. Imported rather than restated
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
# decode. The <10 s merge is the first line of defence; this is the second.
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
# reads 2 holes on `en` and 21 on `fr` (errid-99). The comparison language is chosen upstream by
# `get_delay_language`, whose own comment calls the choice arbitrary among equals -- so a cap set
# near real media would let an arbitrary upstream choice decide whether a pair is refused.
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
# words: step 2 is "similarite faible master<->candidat ? OUI -> run_speed_sweep : le resample
# peut-il la remonter ? ... None -> return, message dans tools.logs (« similarite moyenne faible,
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

# THE CORROBORATION BAND -- how far the ladder's own implied ratio and the sweep's winner may
# differ in MAGNITUDE before the two stop describing the same relation. See
# `corroborate_sweep_against_ladder`: sign is the condition that decides, this one is a
# deliberately wide sanity check. The ladder's implied deviation is a ratio of integer point
# counts, so its relative precision is about `1/|total_rise|` -- 3 % on the one real positive
# (errid-27, 30 points of rise). A factor of two is fifteen to thirty times looser than that,
# which is the intent: refuse a contradiction, never an imprecision. MEASURED: the real positive
# lands at 1.008 and the forced false fire at 2.19, but the forced case is already rejected on
# sign alone, so this constant is not carrying the discrimination and must not be read as if it
# were.
LADDER_CORROBORATION_BAND = 2.0

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
    # step 2
    "similarity_unrecoverable_by_resample": CLASS_CONCLUSIVE,
    "rate_sweep_no_sample_rate": CLASS_COULD_NOT_RUN,
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
    # step 3e, the comparison resample. `comparison_resample_not_implemented` was here until the
    # stage landed and is GONE rather than left behind: this table's own rule is that a token in
    # it can be emitted, so a token nothing can emit any more is a lie about what the chain can
    # say. The two below are the real refusals that replaced it, both could-not-run -- neither is
    # a measurement about the pair.
    "comparison_resample_refused_at_unity": CLASS_COULD_NOT_RUN,
    "comparison_resample_filter_unbuildable": CLASS_COULD_NOT_RUN,
    # step 4. `hole_resolution_not_implemented` was here until the stage landed and is GONE, by
    # this table's own rule (see step 3e above). Its replacement is the resolver's REAL refusal:
    # the frame-exact search could not establish a boundary on at least one hole (anchors not
    # established, geometry unreconciled, an unreadable walk, no measurable grid). Could-not-run,
    # never conclusive: a resolver that could not seed an anchor has measured nothing about the
    # pair -- the per-hole resolver reason travels in the prose and in the step log.
    "hole_resolution_declined": CLASS_COULD_NOT_RUN,
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
    tools.dev_log(f"orchestrator: launch step={step} "
                  f"{_fields(sorted(fields.items()))}\n")


def step_result(step, **fields):
    """The result half. Emitted on EVERY exit of the step, including its refusals -- a step that
    logs only its successes is a step whose failures are invisible."""
    tools.dev_log(f"orchestrator: result step={step} "
                  f"{_fields(sorted(fields.items()))}\n")


def _plan_line(kind, candidate_path, **fields):
    """`repair: plan <kind> ...` -- A HARD REQUIREMENT ON THIS CHAIN, NOT A STYLE CHOICE.

    `merge_plan_report.is_job_log` (`:908`) recognises a job log by exactly one test:

        any(line.startswith("repair: plan ") for line in text.splitlines())

    So if the orchestrator stops emitting this line, every downstream reader stops recognising
    the log as a job log AT ALL -- not "reads it with fewer fields", stops seeing it. The line is
    therefore emitted on every terminal path of `repair()`, success or refusal, with
    `kind=none` when no plan was built. Unconditional (`tools.logs`), matching the existing
    emitter at `merge_video_repair.py:2693`, which is also unconditional.
    """
    tools.logs.append(f"repair: plan {kind} orchestrator=1 "
                      f"{_fields(sorted(fields.items()))} for {candidate_path}\n")


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
    if cause not in DECLINE_CAUSES:
        tools.log_always(f"repair: orchestrator UNVOCABULARISED cause={cause} for "
                         f"{candidate_path} -- this token is not in DECLINE_CAUSES and has no "
                         f"measurement class; add it there. The refusal below stands.\n")
    measurement_class = DECLINE_CAUSES.get(cause, "unclassified")
    tools.log_always(f"repair: orchestrator cause={cause} "
                     f"measurement={measurement_class} for {candidate_path}\n")
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
        completed = subprocess.run(
            [tools.software["ffprobe"], "-v", "error", "-show_entries",
             "format=duration", "-of", "default=nw=1:nk=1", video_obj.filePath],
            capture_output=True, text=True, timeout=120)
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
                      duration_seconds, audio_filter=None, output_duration_seconds=None):
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
        points = audioCorrelation.calculate_fingerprints(wav, length=output_duration_seconds)
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"orchestrator: fingerprint_track raised on "
                      f"{video_obj.filePath} stream_order={stream_order}: "
                      f"{type(error).__name__}: {error}\n")
        return None, None
    finally:
        # THE CLEANUP IS ONLY CORRECT BECAUSE THE PATH IS UNIQUE. `work_dir` is per-pair and the
        # name carries the side and the stream order, so two tracks of one pair cannot overwrite
        # each other's WAV nor delete each other's file. When hole resolution is parallelised
        # (design stage 7) this naming rule has to be extended, not assumed -- the design names
        # the collision as hazard 1.
        try:
            remove(wav)
        except OSError:
            pass
    if not points:
        return None, None
    return points, CHROMAPRINT_HOP_MS


# ---------------------------------------------------------------------------
# ZONES -> HOLES -- step 3, sub-step 4's input
# ---------------------------------------------------------------------------

def derive_holes(zones, n_master, n_candidate, quantum_ms, candidate_quantum_ms):
    """Zones -> the raw gaps between them, BEFORE merging and BEFORE classification.

    A hole is the space between two aligned zones (ruling, chimeric step 4: "un trou = l'espace
    entre deux zones alignees"), plus the two ends of the file. With `n` zones there are exactly
    `n + 1` gaps, and gap `i` is separated from gap `i+1` by exactly `zones[i]` -- that adjacency
    is what the merge rule below walks, so it is built explicitly here rather than recovered
    later.

    Gaps are emitted EVEN WHEN EMPTY ON BOTH AXES, at this stage, because an empty span can
    still carry a real step: two zones can be adjacent in index and still differ in offset,
    which is a cut with no slack around it. Empty-and-stepless gaps are dropped after merging,
    not before -- dropping them first would break the zone-between-gaps adjacency the merge
    needs.

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
            "merged_from": 1,
            "merged_zone_lengths_seconds": [],
        })
    return holes


def _copy_hole(hole):
    """A hole whose mutable bounds are its own, so merging cannot reach back into the raw list."""
    copy = dict(hole)
    for key in ("master_points", "candidate_points", "master_ms", "candidate_ms",
                "merged_zone_lengths_seconds"):
        copy[key] = list(hole[key])
    return copy


def merge_holes(holes, zones, quantum_ms, candidate_quantum_ms=None):
    """THE <10 s MERGE, applied to fixpoint, BEFORE classification.

    Two holes merge when the ALIGNED ZONE BETWEEN THEM is shorter than
    `HOLE_MERGE_WINDOW_SECONDS` on the master axis -- derived from the resolver's own reach, see
    that constant. The merged hole spans from the first's start to the second's end on BOTH
    axes, and merging is transitive: three holes each 4 s apart become one, which this gets for
    free by walking left to right and extending the open hole rather than pairing.

    MERGING HAPPENS BEFORE CLASSIFICATION, AND THE ORDER IS LOAD-BEARING: a merge that swallows
    the first or the last zone converts an INTERIOR hole into a head or tail hole, and a
    classification taken first would then be wrong about which resolver to call. So this
    function does not classify; `classify_holes` runs after it.

    THE MERGE IS OF SEARCH REGIONS, NOT OF EDITS (standing minimal-destruction invariant). It
    says "these two holes cannot be searched independently", never "these two edits are one
    edit" -- the resolved boundaries still come from the resolver, and nothing here widens them.
    Every merged hole records HOW MANY it swallowed and the length of each zone it crossed, so a
    reader can tell one 20 s hole from four 3 s holes 4 s apart; without that the merge would
    hide exactly the structure it was invented to handle.

    BOTH SPANS ARE RECOMPUTED AS A HOLE GROWS, ON EACH AXIS'S OWN QUANTUM. Found by the
    Architect's review of this stage and fixed here: until now only `master_span_seconds` was
    recomputed while `candidate_span_seconds` kept the FIRST swallowed fragment's value, even
    though both axes' point and millisecond bounds were extended correctly. The consequence is
    not cosmetic -- `edge_addition_seconds` sums `candidate_span_seconds` over the head and tail
    holes, and it feeds ADDENDUM 5's 15-second marker rule, so a merged edge hole would
    UNDERCOUNT the candidate content actually being added and could leave a track untagged that
    the addendum requires tagged. A track wrongly left untagged keeps its INTACT status in
    `keep_best_audio` and can beat a genuinely intact track; that is the "intact wins" invariant
    broken by an arithmetic omission, which is exactly the class of silent defect this campaign
    exists to find.

    BLAST RADIUS, MEASURED BY AN INDEPENDENT TESTER ON THE WHOLE CORPUS BEFORE THE FIX, so the
    size of what was being lost is on the record: 15 merged holes across 9 of the 20 pairs, FIVE
    of them EDGE holes -- the ones that reach ADDENDUM 5 -- under-reporting by 12 s to 1236 s,
    always in the under-reporting direction (the direction that can only fail to tag a real
    chimera, never over-tag). Three pairs decide their tag on edge additions ALONE, with no
    interior hole to rescue them through bound (a), and on two of those the deciding number was
    the corrupted one. RE-MEASURED THROUGH THIS CODE AFTER THE FIX, same pairs, same runs:
        errid-99  en  head, merged_from=2   72.945 s -> 669.031 s   (the true merged extent)
        errid-696 ja  tail, merged_from=5   71.863 s -> 345.786 s
        errid-696 ja  head, merged_from=2  119.027 s -> 131.314 s
        errid-123 ja  head, merged_from=3  117.468 s -> the pair no longer reaches holes at all
    and errid-99/en's ADDENDUM 5 line now reads "edge additions total 669.031s" where it read
    72.945 s. The tag was `True` either way on this corpus -- every margin was 5x or more -- so
    what was broken here was the RECORD that clause (d) requires always, not any decision yet
    taken. A pair whose true edge span crosses 15 s while its stale span did not would have
    turned that into a silent mis-tag.

    `candidate_quantum_ms` defaults to the master's only so that a caller that genuinely has one
    quantum need not say it twice -- the per-track-quantum invariant means the real caller
    (`holes_for_couple`) always passes the candidate's own.
    """
    if not holes:
        return []
    if candidate_quantum_ms is None:
        candidate_quantum_ms = quantum_ms
    # COPIED, NOT ALIASED: the bound lists below are mutated in place as a hole grows, and a
    # shallow `dict()` would have the merged hole and its raw source sharing the same list
    # objects -- so the "before merging" record a reader compares against would silently become
    # the "after merging" one.
    merged = [_copy_hole(holes[0])]
    for index in range(1, len(holes)):
        zone_between = zones[index - 1]
        zone_seconds = (zone_between[0][1] - zone_between[0][0] + 1) * quantum_ms / 1000.0
        current = holes[index]
        if zone_seconds < HOLE_MERGE_WINDOW_SECONDS:
            open_hole = merged[-1]
            open_hole["master_points"][1] = current["master_points"][1]
            open_hole["candidate_points"][1] = current["candidate_points"][1]
            open_hole["master_ms"][1] = current["master_ms"][1]
            open_hole["candidate_ms"][1] = current["candidate_ms"][1]
            open_hole["touches_tail"] = open_hole["touches_tail"] or current["touches_tail"]
            open_hole["merged_from"] += 1
            open_hole["merged_zone_lengths_seconds"].append(round(zone_seconds, 3))
            open_hole["master_span_seconds"] = (
                max(0, open_hole["master_points"][1] - open_hole["master_points"][0] + 1)
                * quantum_ms / 1000.0)
            open_hole["candidate_span_seconds"] = (
                max(0, open_hole["candidate_points"][1] - open_hole["candidate_points"][0] + 1)
                * candidate_quantum_ms / 1000.0)
        else:
            merged.append(_copy_hole(current))
    return merged


def classify_holes(holes, zones_detail, quantum_ms):
    """head / interior / tail, AFTER merging, plus the step across each hole.

    The step is the offset difference across the hole -- `offset_after - offset_before` -- which
    only exists for an interior hole: a head hole has no zone before it and a tail hole none
    after, so their step is None, and None here means "there is no such quantity", not zero.

    A hole that touches BOTH ends after merging spans the whole file. That is not a head hole,
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
        # After merging, the zone AFTER the hole is the one following the LAST gap swallowed.
        after_index = gap_index + hole["merged_from"] - 1
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
    `master_axis_coverage_fraction` and through `unmatched_seconds` below.
    """
    if not zones:
        return [], []
    out_zones, out_detail = [], []
    for zone, detail in zip(zones, zones_detail):
        if out_detail and detail["offset_points"] == out_detail[-1]["offset_points"]:
            previous_zone, previous_detail = out_zones[-1], out_detail[-1]
            unmatched = zone[0][0] - previous_zone[0][1] - 1
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
        out_detail.append(detail)
    return out_zones, out_detail


def holes_for_couple(alignment):
    """zones -> coalesce same-offset runs -> holes -> merge -> classify -> drop the non-holes.

    The whole sub-step, in order, and the order is the design's: coalescing first so that a hole
    means a change of offset; the <10 s merge next, because it is about the RESOLVER'S REACH and
    must see the real holes; classification last, because a merge can turn an interior hole into
    a head or tail one.

    The final drop removes gaps that are empty on both axes AND carry no step: bookkeeping
    artefacts of "there are n+1 gaps around n zones", not places where the two timelines differ.
    """
    zones, zones_detail = coalesce_same_offset_zones(alignment.get("zones") or [],
                                                      alignment.get("zones_detail") or [])
    if not zones:
        return []
    quantum_ms = alignment["quantum_ms"]
    candidate_quantum_ms = alignment.get("candidate_quantum_ms") or quantum_ms
    raw = derive_holes(zones, alignment["n_master"], alignment["n_candidate"],
                        quantum_ms, candidate_quantum_ms)
    merged = merge_holes(raw, zones, quantum_ms, candidate_quantum_ms=candidate_quantum_ms)
    classified = classify_holes(merged, zones_detail, quantum_ms)
    return [hole for hole in classified
            if hole["master_span_seconds"] > 0 or hole["candidate_span_seconds"] > 0
            or hole["step_points"]]


def edge_addition_seconds(holes):
    """Total candidate content to be added at the head and the tail, in seconds.

    Feeds ADDENDUM 5's marker decision and, per that addendum's clause (d), is LOGGED ALWAYS --
    tagged or not, the artefact must always know what was done; only the competitive marker
    follows the threshold.

    ONCE THE VIDEO HAS RESOLVED AN EDGE, ITS COUNT REPLACES THE AUDIO'S ESTIMATE (stage 4).
    Before resolution the only number available is the audio hole's candidate span. After it,
    `resolve_hole` sets `resolved_edge_addition_seconds` on the hole: the MASTER frames that
    will stand on the output timeline where the candidate has no common content -- the
    EDGE ruling's addendum, verbatim, "the number of master frames remaining past the last
    compared frame IS the length to take from the master ... duration = frames x the exact
    rational frame time". A trim (`master_exhausted`) adds nothing and counts zero; a closed
    edge counts zero. The audio estimate is kept only where no resolution happened.
    """
    total = 0.0
    for hole in holes:
        if hole["kind"] in ("head", "tail"):
            resolved = hole.get("resolved_edge_addition_seconds")
            total += hole["candidate_span_seconds"] if resolved is None else resolved
    return total


def chimeric_tag_required(holes):
    """ADDENDUM 5, the marker rule, as a single decision with its reasons attached.

    Returns `(required, reason)`. (a) ANY interior splice tags, whatever its size. (b) Edge
    additions totalling at or above the threshold tag. Otherwise the track keeps its INTACT
    status in `keep_best_audio`, which is the point: the standing "intact wins" rule must not
    demote a near-intact track over a marginal edge completion.

    The `resampled:<factor>` marker is deliberately NOT decided here -- clause (c) makes it
    INDEPENDENT of this threshold, because a resample transforms the WHOLE track with or without
    any splice.
    """
    if any(hole["kind"] == "interior" for hole in holes):
        return True, "an interior splice tags regardless of its size (addendum 5 bound a)"
    added = edge_addition_seconds(holes)
    if added >= EDGE_ADDITION_CHIMERIC_TAG_THRESHOLD_SECONDS:
        return True, (f"edge additions total {added:.3f}s, at or above the "
                      f"{EDGE_ADDITION_CHIMERIC_TAG_THRESHOLD_SECONDS}s threshold "
                      f"(addendum 5 bound b)")
    return False, (f"edge additions total {added:.3f}s, under the "
                   f"{EDGE_ADDITION_CHIMERIC_TAG_THRESHOLD_SECONDS}s threshold and no interior "
                   f"splice -- the original track with a marginal completion, not a chimera")


# ---------------------------------------------------------------------------
# THE MULTI-COUPLE CROSS-CHECK -- step 3, sub-step 3
# ---------------------------------------------------------------------------

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
    `coalesce_same_offset_zones` has joined same-offset runs and after the <10 s merge -- a
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
        members = cluster["events"]
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
    """THE ONE LOG LINE THAT IS NOT GATED. The ruling is explicit about this and only this:
    "CES INFOS-LA SORTENT MEME A tools.dev=false". So the disagreement record goes through
    `tools.log_always`, carrying, per cluster, every couple's stream pair, master position, step
    in both points and ms, quantum, residual fraction and coverage -- "TOUTES LES INFOS
    EXTRAITES", not a summary of them.

    The AGREEMENT case stays gated: the ruling's sentence is about the refusal, and putting
    every healthy pair's full cluster table on the unconditional channel would push the real
    disagreements out of the 500-line stderr window that `tools.log_always`'s own docstring
    measures.
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
        tools.log_always(
            f"repair: orchestrator intercouple_disagreement for {candidate_path} "
            f"cluster={cluster['cluster_index']} "
            f"master_position_s={cluster['master_position_seconds']} "
            f"spread_ms={cluster['spread_ms']} tolerance_ms={cluster['tolerance_ms']} "
            f"n_above_floor={cluster['n_above_floor']} "
            f"below_floor_excluded={cluster['below_floor_excluded']} "
            f"could_not_see={cluster['could_not_see']}\n")
        for event in cluster["members"]:
            tools.log_always(
                f"repair: orchestrator intercouple_disagreement_member "
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
# THE STUBS -- stages not yet landed. EACH ONE DECLINES BY NAME.
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


def comparison_resample(speed_factor, master_obj, candidate_obj, language, work_dir,
                        sample_rate=None):
    """The comparison resample: the filter that lets the ALIGNER see across a rate relation.

    Returns `(routing_or_None, cause_or_None)`. The routing is a DESCRIPTION, not a file: nothing
    is written here. The candidate's comparison tracks are speed-corrected inside the SAME ffmpeg
    invocation that already extracts them for fingerprinting (`fingerprint_track`), so the
    correction costs no extra decode, no extra temporary file and no extra generation of codec.

    WHAT THIS PRODUCES IS A COMPARISON EXTRACTION, NEVER A PRODUCT TRACK. The sweep and this
    resample serve to ALIGN and LOCATE (ADDENDUM 7 point 1, fpcalc validated by the owner). The
    corrected audio exists for the length of one fingerprint pass and is deleted by
    `fingerprint_track`'s own `finally`. DELIVERING speed-corrected audio is now AUTHORISED --
    ADDENDUM 8 lifted ADDENDUM 7's deferral: "L'audio corrige en vitesse PEUT etre livre ... La
    piste livree porte le marqueur `resampled:<facteur exact>`" -- but building that track is
    plan application (stage 5), from the source, not from this temporary extraction.

    NO FILTER AT ALL AT speed_factor = 1 (owner's ADDENDUM 6): "aucun filtre de correction ne
    tourne JAMAIS sur une piste dont la vitesse n'a pas change". That is enforced TWICE and
    deliberately: the caller does not enter this function at factor 1 or None, and this function
    refuses one anyway. A rule enforced only at the call site is a rule one new call site removes.

    EXACT RATIONALS, NEVER FLOATS, AND THE ONE THAT MATTERS IS THE EFFECTIVE ONE.
    `merge_video_resample.build_speed_filter_chain` is the pipeline's single authority on this
    arithmetic and it is called, not reimplemented: `asetrate` takes an INTEGER, so the factor
    actually obtained is `intermediate / round(intermediate / ratio)` and is NOT the one
    requested. Both are carried on the routing -- the requested one as the exact `Fraction` the
    sweep won with, the effective one as the exact integer ratio the filter will really apply --
    and it is the EFFECTIVE one every downstream length is computed from, because it is the one
    the audio will actually have.
    """
    if speed_factor is None or speed_factor == 1:
        # ADDENDUM 6, enforced here as well as at the call site. See the docstring.
        return None, "comparison_resample_refused_at_unity"
    if sample_rate is None:
        sample_rate = comparison_sample_rate(master_obj, candidate_obj, language)
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
        chain, effective, intermediate, target = merge_video_resample.build_speed_filter_chain(
            source_rate, ratio_decimal)
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"orchestrator: build_speed_filter_chain refused "
                      f"({type(error).__name__}: {error})\n")
        return None, "comparison_resample_filter_unbuildable"
    routing = pitch_routing(speed_factor, master_obj, candidate_obj, language, work_dir,
                            sample_rate)
    routing.update({
        "modality": MODALITY,
        "side": "candidate",
        "rule": "comparison_extraction_only_never_a_product_track",
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

    Reads the same two places `merge_video_repair._candidate_sample_rate_for_speed_test` reads
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
        that is simply candidate time. Under a comparison resample it is the candidate's raw
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
    `couple_start_delta_ms`), so the couple's start delta is folded in here, once, and every
    reader downstream -- the video search, the plan -- receives file-time offsets."""
    quantum_ms = hole["quantum_ms"]
    delta = _track_delay_delta(hole)
    return {
        "audio_offset_before_ms": (None if hole.get("offset_before_points") is None
                                   else round(hole["offset_before_points"] * quantum_ms
                                              + delta, 3)),
        "audio_offset_after_ms": (None if hole.get("offset_after_points") is None
                                  else round(hole["offset_after_points"] * quantum_ms
                                             + delta, 3)),
        "audio_offset_precision_ms": round(quantum_ms, 3),
        "track_delay_delta_ms": delta,
    }


def _track_delay_delta(hole):
    """The couple's start delta carried on the hole's frame domain (0.0 when none was set)."""
    return float((hole.get("frame_domain") or {}).get("track_delay_delta_ms") or 0.0)


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
    it, so the two stay on one clock whatever the probe returned."""
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
    step_launch("two_anchor", candidate=candidate_obj.filePath, probe=probe,
                resolve_shift=resolve_shift,
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
            shift_search_frames=_shift_search_frames(domain, quantum_ms))
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
    """
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
        return narrowed
    if _span_noise_reading(narrowed) is not None:
        return {"declined": True, "reason": "sweep_front_inside_common_content",
                "evidence": (f"first span {result['pre_collapse_start_master']}-"
                             f"{result['pre_collapse_end_master']} claimed by {reading}; "
                             f"re-fronted span {narrowed['pre_collapse_start_master']}-"
                             f"{narrowed['pre_collapse_end_master']} still claimed "
                             f"(before={narrowed['unmatched_span_matches_before']} "
                             f"after={narrowed['unmatched_span_matches_after']})")}
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
    """
    same_shift = result["before_shift_frames"] == result["after_shift_frames"]
    if same_shift and result["master_end_frame"] == result["master_start_frame"]:
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
    """
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
    At or above the floor none of this runs: a step of two quanta or more is a measured edit.
    """
    quantum_ms = hole["quantum_ms"]
    offset_before_ms = hole["offset_before_points"] * quantum_ms + _track_delay_delta(hole)
    offset_after_ms = hole["offset_after_points"] * quantum_ms + _track_delay_delta(hole)
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
    if below_floor and not proposal_single_shift:
        if result["declined"]:
            hypotheses = [("no_cut_at_offset_before", offset_before_ms, True),
                          ("no_cut_at_offset_after", offset_after_ms, True)]
        else:
            hypotheses = [
                ("no_cut_at_anchor_a_shift",
                 float(result["before_shift_frames"] * frame_ms), False),
                ("no_cut_at_anchor_b_shift",
                 float(result["after_shift_frames"] * frame_ms), False)]
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
                       "video_verdict": verdict}
            return _interior_outcome(hole, domain, probe, verdict, refuted_proposal=refuted)
    if result["declined"]:
        return _declined(hole, result.get("reason"), result.get("evidence"),
                         no_cut_probe_run=below_floor)
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
        offset_ms = hole["offset_after_points"] * quantum_ms + _track_delay_delta(hole)
        low_ms, high_ms = 0.0, max(float(hole["master_ms"][1]), frame_ms)
    else:
        offset_ms = hole["offset_before_points"] * quantum_ms + _track_delay_delta(hole)
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
            shift_search_frames=_shift_search_frames(domain, quantum_ms))
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
    if hole["kind"] == "interior":
        outcome = _resolve_interior(hole, domain, master_obj, candidate_obj)
    elif hole["kind"] in ("head", "tail"):
        outcome = _resolve_edge(hole, domain, master_obj, candidate_obj)
    else:
        outcome = _declined(hole, f"no_resolver_for_kind:{hole['kind']}")
    # THE AUDIO'S BRACKET, IN FRAMES, BESIDE THE VIDEO'S ANSWER -- so a reader can see where the
    # audio proposed and where the video disposed without redoing the conversion.
    outcome["audio_master_frames"] = [_master_frame_of_ms(hole["master_ms"][0], domain),
                                      _master_frame_of_ms(hole["master_ms"][1], domain)]
    outcome["audio_candidate_frames"] = [
        _candidate_native_frame_of_ms(hole["candidate_ms"][0], domain),
        _candidate_native_frame_of_ms(hole["candidate_ms"][1], domain)]
    outcome["audio_step_ms"] = (None if hole["step_ms"] is None else round(hole["step_ms"], 3))
    return outcome


# ---------------------------------------------------------------------------
# STEP 5 -- PLAN APPLICATION. It applies the plan and nothing else (ADDENDUM 10 d).
# ---------------------------------------------------------------------------

# THE REFINEMENT'S WINDOW IS THE DELIVERY GATE'S OWN: `merge_video_chimeric.verify_window_
# seconds` (20 s) at `verify_probe_rate` (8 kHz, 0.125 ms per sample). The offset is measured at
# the scale the gate will check it at, with the same instrument (`measure_lag_ms`), and never on
# a window of its own invention. Read from the module at call time, not restated.
#
# THREE WINDOWS PER ZONE, when the zone holds them: two can disagree with nothing to break the
# tie, three give a majority -- the same reasoning `choose_probe_positions` gives for probing a
# piece twice ("their DISAGREEMENT is the signal"), plus the one that decides.
REFINE_WINDOWS_PER_ZONE = 3

# THE FLOOR UNDER A SHORTENED WINDOW. A zone shorter than the gate's 20 s gets a window as long
# as the zone allows, but not below this. Chosen, not derived -- a floor on the INSTRUMENT, the
# same status as `PITCH_PROBE_WINDOW_MINIMUM_SECONDS`: a cross-correlation over less programme
# than this is a peak nobody should call a measurement. A zone under it is not measured, and the
# fallbacks below are named and logged.
REFINE_MIN_WINDOW_SECONDS = 4.0

# HOW FAR THE REFINEMENT SEARCHES, IN QUANTA OF THE COUPLE: the carried offset is precise to one
# fingerprint quantum, and a zone the video closed across (ADDENDUM 3) carries the audio's two
# readings one quantum apart -- so the truth lies within one quantum and a half of the offset the
# search is centred on. The SAME slack the inter-couple step tolerance was measured to need
# (`INTERCOUPLE_STEP_TOLERANCE_SLACK`), bound to it rather than retyped.
REFINE_SEARCH_QUANTA = INTERCOUPLE_STEP_TOLERANCE_SLACK


def _decimal(value):
    """An exact `Fraction` (or anything `str()` renders exactly) as a `Decimal`, for the
    millisecond arithmetic the assembly does in `Decimal`."""
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(str(value))


def _frame_start_ms(frame, domain):
    """A master frame index as the EXACT millisecond it starts at (Decimal)."""
    return _decimal(Fraction(frame) * 1000 / domain["master_rate"])


def plan_geometry(holes, domain, default_offset_ms):
    """The resolved holes, laid on the master timeline: the ZONES read from the candidate and
    the FILLS read from the master, in master milliseconds, at the EXACT frames stage 4 resolved.

    Every fill is exactly `[master_start_frame, master_end_frame)` of its hole and never wider
    (ADDENDUM 9 point 1): real candidate content outside it is kept. Closed holes never reach
    here (ADDENDUM 3 -- they were removed as continuity). Per hole kind:
      head      fill [0, boundary) from the master, except a trim (`master_exhausted`, which adds
                nothing -- the candidate's excess head is simply not read); the first zone starts
                at the boundary
      interior  the zone before ends at the fill's start, the zone after starts at its end; an
                addition or a same-length replacement has an EMPTY fill and the zone after reads
                the candidate past the content cut
      tail      fill [boundary + 1, timeline end) from the master, except a trim; the last zone
                ends at the boundary
    The timeline end is the master's VIDEO duration -- the reference ADDENDUM 9 point 6 names.

    Each zone carries the offset the audio measured on it (`coarse_offset_ms`, one quantum
    precise -- the refinement's centre, never the offset applied) and the frame shift the video
    measured on it (`video_shift_frames`, used only to carry a refined offset across a zone too
    short to measure; never an audio offset by itself).
    """
    timeline_ms = _decimal(domain["master_timeline_ms"])
    zones, fills = [], []
    cursor = Decimal(0)
    pending_offset, pending_shift = default_offset_ms, None
    for index, hole in enumerate(holes):
        resolution = hole["resolution"]
        status = resolution["status"]
        start_ms = min(_frame_start_ms(resolution["master_start_frame"], domain), timeline_ms)
        end_ms = min(_frame_start_ms(resolution["master_end_frame"], domain), timeline_ms)
        if hole["kind"] == "head":
            if status != EDGE_MASTER_EXHAUSTED and end_ms > 0:
                fills.append({"master_start_ms": Decimal(0), "master_end_ms": end_ms,
                              "reason": WHY_TOKEN["head"], "hole": index, "status": status})
            cursor = end_ms
            pending_offset = _decimal(resolution["audio_offset_after_ms"])
            pending_shift = resolution["shift_frames"]
            continue
        zone_end = start_ms
        if hole["kind"] == "tail" and status == EDGE_MASTER_EXHAUSTED:
            zone_end = timeline_ms
        before_offset = resolution["audio_offset_before_ms"]
        zones.append({"master_start_ms": cursor, "master_end_ms": zone_end,
                      "coarse_offset_ms": (_decimal(before_offset) if before_offset is not None
                                           else pending_offset),
                      "video_shift_frames": resolution.get("before_shift_frames",
                                                           resolution.get("shift_frames"))})
        if hole["kind"] == "tail":
            if status != EDGE_MASTER_EXHAUSTED and timeline_ms > start_ms:
                fills.append({"master_start_ms": start_ms, "master_end_ms": timeline_ms,
                              "reason": WHY_TOKEN["tail"], "hole": index, "status": status})
            cursor = timeline_ms
            pending_offset, pending_shift = None, None
            continue
        if end_ms > start_ms:
            fills.append({"master_start_ms": start_ms, "master_end_ms": end_ms,
                          "reason": WHY_TOKEN["interior"], "hole": index, "status": status})
        cursor = end_ms
        pending_offset = _decimal(resolution["audio_offset_after_ms"])
        pending_shift = resolution["after_shift_frames"]
    if cursor < timeline_ms:
        zones.append({"master_start_ms": cursor, "master_end_ms": timeline_ms,
                      "coarse_offset_ms": pending_offset, "video_shift_frames": pending_shift})
    zones = [zone for zone in zones if zone["master_end_ms"] > zone["master_start_ms"]]
    for number, zone in enumerate(zones):
        zone["zone"] = number
    return zones, fills


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


def _refine_zone(zone, master_samples, master_start_ms, candidate_samples, candidate_start_ms,
                 search_ms, agreement_ms):
    """ONE zone of ONE track: the offset the audio carries, refined to the sample by
    cross-correlation of the candidate's OWN track against the master (ADDENDUM 9 point 2).

    Windows of the gate's length, as many as the zone holds up to `REFINE_WINDOWS_PER_ZONE`,
    kept `search_ms` inside the zone on the master axis so that no window can straddle the
    zone's own boundary under the offset's uncertainty. A window is REJECTED -- never averaged
    in -- when either side is silent (the verifier's own RMS floor) or when its peak sits on the
    edge of the search (an edge peak is the search running out, not a lag). The accepted lags
    must AGREE: the largest group within `agreement_ms` (half a master frame -- the refinement
    exists to beat one frame, so two readings further apart than half of one are not the same
    reading) decides by its median; a zone whose windows all disagree is unmeasured, by name.

    Returns a dict: `offset_ms` (Decimal) or None, with every window's lag and correlation.
    """
    import merge_video_chimeric
    rate = merge_video_chimeric.verify_probe_rate
    coarse = zone["coarse_offset_ms"]
    span_ms = zone["master_end_ms"] - zone["master_start_ms"] - 2 * search_ms
    window_ms = min(Decimal(merge_video_chimeric.verify_window_seconds) * 1000, span_ms)
    reading = {"zone": zone["zone"], "coarse_offset_ms": str(coarse), "windows": [],
               "offset_ms": None, "reason": None}
    if coarse is None:
        reading["reason"] = "no_coarse_offset"
        return reading
    if window_ms < Decimal(str(REFINE_MIN_WINDOW_SECONDS)) * 1000:
        reading["reason"] = f"zone_too_short({float(span_ms + 2 * search_ms):.0f}ms)"
        return reading
    count = int(max(1, min(REFINE_WINDOWS_PER_ZONE, span_ms // window_ms)))
    free_ms = span_ms - window_ms
    starts = [zone["master_start_ms"] + search_ms
              + (free_ms * index / (count - 1) if count > 1 else free_ms / 2)
              for index in range(count)]
    n_samples = int(window_ms * rate / 1000)
    lags = []
    for start in starts:
        m_index = int(round((start - master_start_ms) * rate / 1000))
        c_index = int(round((start + coarse - candidate_start_ms) * rate / 1000))
        entry = {"master_ms": str(round(start, 1))}
        if (m_index < 0 or c_index < 0 or m_index + n_samples > len(master_samples)
                or c_index + n_samples > len(candidate_samples)):
            entry["outcome"] = "outside_track"
            reading["windows"].append(entry)
            continue
        reference = master_samples[m_index:m_index + n_samples].astype("float64")
        produced = candidate_samples[c_index:c_index + n_samples].astype("float64")
        reference = reference - reference.mean()
        produced = produced - produced.mean()
        if min(merge_video_chimeric.get_rms(reference),
               merge_video_chimeric.get_rms(produced)) < merge_video_chimeric.verify_min_rms:
            entry["outcome"] = "no_signal"
            reading["windows"].append(entry)
            continue
        lag, correlation = merge_video_chimeric.measure_lag_ms(
            reference, produced, rate, float(search_ms))
        entry.update({"lag_ms": round(lag, 3), "r": round(correlation, 4)})
        if abs(lag) >= float(search_ms) - 1000.0 / rate:
            entry["outcome"] = "peak_at_search_edge"
        else:
            entry["outcome"] = "measured"
            lags.append(lag)
        reading["windows"].append(entry)
    if not lags:
        reading["reason"] = "no_window_measured"
        return reading
    ordered = sorted(lags)
    best = []
    for low in range(len(ordered)):
        group = [lag for lag in ordered[low:] if lag - ordered[low] <= float(agreement_ms)]
        if len(group) > len(best):
            best = group
    if len(best) < 2 and len(lags) > 1:
        reading["reason"] = f"windows_disagree({[round(lag, 2) for lag in ordered]})"
        return reading
    median = best[len(best) // 2] if len(best) % 2 else (best[len(best) // 2 - 1]
                                                         + best[len(best) // 2]) / 2
    # candidate_time = master_time + offset; a window read at the coarse offset whose content
    # arrives `e` ms later returns lag -e (measured on a synthetic shift), so the offset is the
    # coarse one MINUS the lag.
    reading["offset_ms"] = coarse - Decimal(str(round(median, 3)))
    reading["n_agreeing"] = len(best)
    reading["single_window"] = len(lags) == 1
    return reading


def measure_track_offsets(zones, master_obj, candidate_obj, language, master_stream,
                          candidate_stream, speed_ratio, scale, quantum_ms, domain):
    """EACH TRACK ITS OWN OFFSET, PER ZONE, IN SUB-FRAME MILLISECONDS (ADDENDUM 9 points 2, 14).

    Every candidate audio track is correlated against the master's track of ITS OWN language
    (the comparison language against the very master stream the alignment was measured on). A
    zone that could not be measured takes, in this order and ALWAYS LOGGED:
      1. `derived_by_video_step`: the same track's nearest measured zone, moved by the frame
         step the video measured between the two zones -- a frame count stage 4 pinned exactly,
         so the carried relation stays sub-frame;
      2. `inherited`: the comparison track's offset for that zone (the track of the same
         source), with both values in the log -- never another language's offset in silence.
    The comparison track itself has nothing to inherit from: with no zone of its own measured,
    the plan has no audio offset it can apply, and says so.

    Returns `(tracks, None)` or `(None, reason)`; `tracks[stream_order]` = {"language",
    "zones": [readings], "start_ms", "extent_ms", "extent_source", "measured"}.
    """
    import merge_video_chimeric
    import merge_video_resample
    rate = merge_video_chimeric.verify_probe_rate
    search_ms = Decimal(str(REFINE_SEARCH_QUANTA)) * _decimal(quantum_ms)
    agreement_ms = _decimal(domain["frame_ms"]) / 2
    frame_ms = _decimal(domain["frame_ms"])
    audios = list(merge_video_chimeric.iterate_candidate_audios(candidate_obj))
    audios.sort(key=lambda item: str(item[1].get("StreamOrder")) != str(candidate_stream))
    master_cache = {}
    tracks = {}
    comparison_order = None
    for track_language, audio in audios:
        order = int(audio["StreamOrder"])
        if str(audio.get("StreamOrder")) == str(candidate_stream):
            comparison_order = order
        start_ms, extent_ms, extent_source = _track_timing(candidate_obj, audio, scale)
        entry = {"language": track_language, "start_ms": start_ms, "extent_ms": extent_ms,
                 "extent_source": extent_source, "zones": [], "measured": False,
                 "reference": None}
        tracks[order] = entry
        master_audio = merge_video_chimeric.find_master_audio_for_language(
            master_obj, track_language,
            master_stream if track_language == language else None)
        step_launch("track_offset", candidate=candidate_obj.filePath, stream=order,
                    language=track_language,
                    master_stream=(None if master_audio is None
                                   else master_audio.get("StreamOrder")))
        if master_audio is None:
            entry["own_reason"] = f"master_carries_no_{track_language}_track"
            entry["zones"] = [{"zone": zone["zone"], "coarse_offset_ms":
                               str(zone["coarse_offset_ms"]), "offset_ms": None,
                               "reason": entry["own_reason"], "windows": []}
                              for zone in zones]
            step_result("track_offset", candidate=candidate_obj.filePath, stream=order,
                        measured=False, reason=entry["own_reason"])
            continue
        master_order = int(master_audio["StreamOrder"])
        entry["reference"] = master_order
        try:
            if master_order not in master_cache:
                master_cache[master_order] = (
                    merge_video_chimeric.read_track_samples(master_obj.filePath, master_order,
                                                            rate),
                    merge_video_chimeric.get_stream_start_ms(master_audio))
            master_samples, master_start_ms = master_cache[master_order]
            chain = None
            if speed_ratio is not None:
                source_rate = (audio.get("ffprobe", {}).get("sample_rate")
                               or audio.get("SamplingRate"))
                chain = merge_video_resample.build_speed_filter_chain(
                    int(float(source_rate)), speed_ratio)[0]
            candidate_samples = merge_video_chimeric.read_track_samples(
                candidate_obj.filePath, order, rate, audio_filter=chain)
        except Exception as error:                                       # noqa: BLE001
            entry["own_reason"] = f"track_unreadable({type(error).__name__})"
            entry["zones"] = [{"zone": zone["zone"], "coarse_offset_ms":
                               str(zone["coarse_offset_ms"]), "offset_ms": None,
                               "reason": entry["own_reason"], "windows": []}
                              for zone in zones]
            step_result("track_offset", candidate=candidate_obj.filePath, stream=order,
                        measured=False, reason=entry["own_reason"], evidence=str(error)[:200])
            continue
        for zone in zones:
            entry["zones"].append(_refine_zone(zone, master_samples, master_start_ms,
                                               candidate_samples, start_ms, search_ms,
                                               agreement_ms))
        del candidate_samples
        entry["measured"] = any(reading["offset_ms"] is not None for reading in entry["zones"])
        step_result("track_offset", candidate=candidate_obj.filePath, stream=order,
                    language=track_language, master_stream=master_order,
                    measured_zones=sum(1 for r in entry["zones"] if r["offset_ms"] is not None),
                    n_zones=len(zones),
                    offsets_ms=[None if r["offset_ms"] is None else float(r["offset_ms"])
                                for r in entry["zones"]],
                    coarse_ms=[r["coarse_offset_ms"] for r in entry["zones"]],
                    reasons=[r["reason"] for r in entry["zones"]],
                    windows=[[(w.get("lag_ms"), w.get("r"), w["outcome"])
                              for w in r["windows"]] for r in entry["zones"]])
    del master_cache

    if comparison_order is None or not tracks[comparison_order]["measured"]:
        return None, (f"the comparison track (stream {candidate_stream}) has no zone whose offset "
                      f"could be measured to the sample: "
                      + ("it is not among the candidate's audio tracks"
                         if comparison_order is None else
                         "; ".join(f"zone {r['zone']}: {r['reason']}"
                                   for r in tracks[comparison_order]["zones"])))

    def derive(entry, zone_index):
        zone = zones[zone_index]
        measured = [r for r in entry["zones"] if r["offset_ms"] is not None
                    and r.get("source", "measured") == "measured"]
        if zone["video_shift_frames"] is None or not measured:
            return None, None
        nearest = min(measured, key=lambda r: abs(r["zone"] - zone_index))
        shift = zones[nearest["zone"]]["video_shift_frames"]
        if shift is None:
            return None, None
        return (nearest["offset_ms"] + (zone["video_shift_frames"] - shift) * frame_ms,
                nearest["zone"])

    comparison = tracks[comparison_order]
    for order in [comparison_order] + [o for o in tracks if o != comparison_order]:
        entry = tracks[order]
        for reading in entry["zones"]:
            if reading["offset_ms"] is not None:
                reading["source"] = "measured"
    for order in [comparison_order] + [o for o in tracks if o != comparison_order]:
        entry = tracks[order]
        for reading in entry["zones"]:
            if reading["offset_ms"] is not None:
                continue
            derived, from_zone = derive(entry, reading["zone"])
            if derived is not None:
                reading["offset_ms"] = derived
                reading["source"] = f"derived_by_video_step(zone_{from_zone})"
            elif order != comparison_order:
                inherited = comparison["zones"][reading["zone"]]["offset_ms"]
                reading["offset_ms"] = inherited
                reading["source"] = f"inherited(stream_{comparison_order})"
            if reading["offset_ms"] is None:
                return None, (f"stream {order} zone {reading['zone']}: no offset measured "
                              f"({reading['reason']}) and none derivable")
            # ADDENDUM 9 point 14: a substitution is logged with BOTH values -- what this track
            # could say for itself, and what it was given. A DECISION, so unconditional.
            tools.logs.append(
                f"repair: offset_substitution stream={order} lang={entry['language']} "
                f"zone={reading['zone']} own=unmeasured({reading['reason']}) "
                f"coarse_ms={reading['coarse_offset_ms']} applied_ms={reading['offset_ms']} "
                f"source={reading['source']}\n")
    return tracks, None


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


def apply_plan(candidate_path, holes, speed_factor, master_obj, candidate_obj, context):
    """STAGE 5 -- THE PLAN, APPLIED, AND NOTHING ELSE (ADDENDUM 10 d: "LA FONCTION QUI TRAITE LE
    PLAN NE FAIT QUE TRAITER LE PLAN -- aucune decision, aucun test d'opportunite"). Returns
    `(ok, cause, reason)`; `ok` is True only when the temporary chimeric file exists.

    WHAT IT RECEIVES (the stage-4 contract, `resolve_hole`): `holes` = the holes that remain
    after closures, each with its `resolution` in exact master frames. `context` carries what
    the orchestrator measured that the plan needs: the comparison language, the driving couple's
    streams, its quantum, the frame domain, the sweep's gate (a rate pair's evidence) and the
    ADDENDUM 5 marker decision taken on these very holes.

    WHAT IT DOES, in order, each step launched and resulted in the dev log:
      1  the geometry -- `plan_geometry`: zones and fills at the exact frames, fills never wider
         than the hole (ADDENDUM 9 point 1); every edge addition logged, tagged or not
         (ADDENDUM 5 clause d);
      2  the speed -- at a confirmed factor other than 1 the candidate's tracks are resampled at
         the EXACT rational (asetrate behind the pitch layer's routing, ADDENDUM 8), `resampled:
         <effective factor>` on every such track; at 1 no filter exists (ADDENDUM 6);
      3  the audio offsets -- `measure_track_offsets`: each track, each zone, to the sample,
         never rounded to a frame (ADDENDUM 9 points 2 and 14);
      4  the pieces -- `track_pieces`, one set per track; the comparison track's set re-times the
         subtitles (pysubs2; the factor on the cue timecodes first on a rate pair, ADDENDUM 8
         point 3; no cue twice at a splice, a cue stopped before the next -- ADDENDUM 10 d);
      5  the chapters -- master editions as they are, candidate editions only re-timed
         (ADDENDUM 9 point 7), every decision logged;
      6  the build -- `merge_video_repair.build_repaired_video_object`: assemble, mux, the
         delivery gates (`verify_output_file`, `verify_on_master_timeline` at its measured
         100 ms, the fabricated-delivery gate);
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
    step_launch("apply_plan", candidate=candidate_path, n_holes=len(holes),
                speed_factor=rate_text, chimeric_tag=context["tagged"])

    # ---- 1. geometry --------------------------------------------------------
    zones, fills = plan_geometry(holes, domain, context["default_offset_ms"])
    head_added = sum((fill["master_end_ms"] - fill["master_start_ms"]) for fill in fills
                     if fill["reason"] == WHY_TOKEN["head"])
    tail_added = sum((fill["master_end_ms"] - fill["master_start_ms"]) for fill in fills
                     if fill["reason"] == WHY_TOKEN["tail"])
    interior_filled = sum((fill["master_end_ms"] - fill["master_start_ms"]) for fill in fills
                          if fill["reason"] == WHY_TOKEN["interior"])
    step_result("plan_geometry", candidate=candidate_path,
                zones=[[float(z["master_start_ms"]), float(z["master_end_ms"])] for z in zones],
                coarse_offsets_ms=[None if z["coarse_offset_ms"] is None
                                   else float(z["coarse_offset_ms"]) for z in zones],
                video_shifts_frames=[z["video_shift_frames"] for z in zones],
                fills=[[float(f["master_start_ms"]), float(f["master_end_ms"]), f["reason"],
                        f["status"]] for f in fills])
    # ADDENDUM 5 clause (d): the added durations, per edge, ALWAYS -- tagged or not.
    tools.logs.append(f"repair: edge_additions head_ms={head_added} tail_ms={tail_added} "
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
    tracks, offset_failure = measure_track_offsets(
        zones, master_obj, candidate_obj, language, context["master_stream"],
        context["candidate_stream"], speed_ratio, scale, context["quantum_ms"], domain)
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
        track_plans[order] = {
            "pieces": pieces,
            "extent_ms": entry["extent_ms"], "extent_source": entry["extent_source"],
            "offset_measured": sources == ["measured"],
            "borrow_reason": (None if sources == ["measured"]
                              else ",".join(sources) + (f"[{entry['own_reason']}]"
                                                        if entry.get("own_reason") else "")),
            "offset_sources": [{"zone": r["zone"], "offset_ms": str(r["offset_ms"]),
                                "source": r["source"]} for r in entry["zones"]]}
        for adjustment in adjustments:
            tools.logs.append(f"repair: plan_edge_adjustment stream={order} "
                              f"zone={adjustment['zone']} kind={adjustment['kind']} "
                              f"master_fill_ms={adjustment['master_fill_ms']}\n")
        for overlap in overlaps:
            tools.logs.append(f"repair: splice_reread stream={order} zones={overlap['zones']} "
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
        tools.logs.append("repair: chapter " + " ".join(
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
        "rate_source": (None if speed_ratio is None else "rate_sweep"),
        "resample_gate": context.get("sweep_gate"),
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

    # ---- 7. DELIVERED_DURATIONS ---------------------------------------------
    delivered = merge_video_chimeric.probe_delivered_durations(out_path)
    tools.logs.append(
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

def speed_factor(master_obj, candidate_obj, language):
    """The ruling's step 2 return contract: A FACTOR, OR None -- plus the gate dict beside it.

    "run_speed_sweep RETOURNE un facteur de vitesse, ou None." The live
    `merge_video_repair.run_speed_sweep` returns `(gate, cause, prose)` and its single call site
    unwraps the winner one line later; the design's recommended shape is `(factor_or_None,
    gate)`, keeping the full measurement for the log. THAT ADAPTER LIVES HERE, ON THE
    ORCHESTRATOR'S SIDE, rather than as an edit to the live function's signature -- the live
    chain is still the production chain until the switch commit, and changing a signature it
    calls is a change to the deployed path, which this stage is not.

    THE TWO Nones MUST NOT COLLAPSE. "The sweep could not be started" (no readable sample rate)
    and "it ran and nothing cleared the floor" are different facts and the standing invariant
    says so; both come back as a None factor for the ORCHESTRATOR'S branch, and the CAUSE
    returned beside it keeps them apart for the log and the ledger.

    Returns `(factor_or_None, gate_or_None, cause_or_None)`.
    """
    try:
        import merge_video_repair
    except Exception as error:                                           # noqa: BLE001
        tools.dev_log(f"orchestrator: merge_video_repair unimportable for the speed sweep "
                      f"({type(error).__name__})\n")
        return None, None, "rate_sweep_no_sample_rate"
    gate, cause, _prose = merge_video_repair.run_speed_sweep(
        master_obj, candidate_obj, language)
    if gate is None:
        return None, None, cause
    winner = gate.get("ratio") if gate.get("verdict") == "confirmed" else None
    if winner is None:
        return None, gate, "similarity_unrecoverable_by_resample"
    return winner, gate, None


def corroborate_sweep_against_ladder(winner, ladder_implied_ratio):
    """TWO INSTRUMENTS OR NONE: a factor the ladder asked for must be one the ladder RECOGNISES.

    WHY THIS EXISTS, AND IT CLOSES A HOLE IN MY OWN SAFETY ARGUMENT. The rate arm was switched
    on because its refusal is non-terminal: a false fire was supposed to cost one sweep and then
    continue on the alignment already measured. An independent tester forced the arm to fire on
    two healthy pairs and measured that the promise fails in the branch that matters -- THE SWEEP
    DID NOT DECLINE, IT CONFIRMED a near-unity factor on both, so the non-terminal branch never
    ran. The real cost of a false fire was a DIFFERENT PLAN BUILT ON DELIBERATELY SPEED-ALTERED
    AUDIO: errid-202 went 6 holes -> 12, and its real, consistent -992.37 ms editorial step
    (three times over) dissolved into a scatter of -124 / -248 / -1116 ms; errid-100 went 1 hole
    -> 12 on one couple and 9 -> 15 on the other, coverage 0.9997 -> 0.9552. The premise that
    licensed n=1 enablement was false, and this is the guard that makes it true.

    THE TEST IS THE TESTER'S OWN DISCRIMINATOR, AND IT NEEDS NO NEW CONSTANT. The ladder already
    measures an `implied_speed_ratio` from the zone offsets it counted. The sweep independently
    picks a rational. On the real positive they agree: errid-27's ladder implied 1.0009911 and
    the sweep returned 1001/1000 = 1.001000 -- 8.9e-6 apart, and the ladder never saw that
    nominal. On the forced false fire they contradict each other IN SIGN: errid-202's ladder
    implied 1.0021909 (candidate slower) while the sweep returned 1000/1001 = 0.999001
    (candidate faster). A relation cannot run in both directions at once.

    TWO CONDITIONS, AND THE FIRST IS THE ONE THAT DECIDES:
      sign       `implied - 1` and `winner - 1` must share a sign. This is the discriminator the
                 measurement named, and it alone rejects the forced case.
      magnitude  the two deviations must agree within a factor of `LADDER_CORROBORATION_BAND`
                 -- a DELIBERATELY WIDE sanity check, not a precision test. The ladder's implied
                 deviation is `total_rise / span_points` over integer point counts, so its own
                 relative precision is about `1/|total_rise|`: errid-27's 30-point rise makes it
                 good to ~3 %. A factor of two is fifteen to thirty times looser than that, which
                 is the point -- this condition must refuse a CONTRADICTION and never a
                 imprecision, because refusing imprecision would throw away a correct factor.

    AN EARLIER DRAFT USED A NEAREST-MEMBER TEST INSTEAD ("the winner must be the vocabulary
    member closest to what the ladder implied") AND IT WAS WRONG IN A WAY WORTH RECORDING. The
    vocabulary contains 25/24 = 1.0416667 and 1001/960 = 1.0427083, one tenth of a percent apart,
    which is FINER than the ladder can measure; a ladder reading of 1.0421 in the PAL band would
    then have rejected the correct 1001/960 in favour of 25/24. The condition would have been
    strict exactly where the instrument is coarse. Measured while building it, not discovered
    afterwards -- and unreachable today only because R2 shows the ladder cannot fire in the PAL
    band at all, which is not a reason to have shipped it.

    A FAILED CORROBORATION IS NOT A REFUSAL OF THE PAIR. The caller discards the factor and
    proceeds at 1 on the alignment already in hand -- which is exactly the behaviour that existed
    before this arm was built, so the worst case of the guard being too strict is the status quo
    ante. That asymmetry is deliberate: over-strictness costs a rate correction the other arms
    would usually have caught anyway, while over-permissiveness rebuilds the plan on altered
    audio, which is what was measured going wrong.

    Returns a dict, always, with `corroborated` and every number it rests on.
    """
    detail = {
        "corroborated": False,
        "winner": (f"{winner.numerator}/{winner.denominator}"
                   if isinstance(winner, Fraction) else str(winner)),
        "winner_value": float(winner),
        "ladder_implied_ratio": ladder_implied_ratio,
        "sign_agrees": None,
        "magnitude_ratio": None,
        "reason": None,
    }
    if ladder_implied_ratio is None:
        detail["reason"] = ("the ladder armed the sweep but reported no implied speed ratio, so "
                            "there is nothing for the winner to corroborate against")
        return detail
    implied_deviation = ladder_implied_ratio - 1.0
    winner_deviation = float(winner) - 1.0
    detail["ladder_deviation"] = round(implied_deviation, 7)
    detail["winner_deviation"] = round(winner_deviation, 7)
    detail["sign_agrees"] = (implied_deviation > 0) == (winner_deviation > 0)
    if not detail["sign_agrees"]:
        detail["reason"] = (
            f"the two instruments contradict each other in SIGN: the ladder counted a drift "
            f"implying {ladder_implied_ratio} (deviation {implied_deviation:+.7f}) while the "
            f"sweep confirmed {detail['winner']} = {float(winner)} (deviation "
            f"{winner_deviation:+.7f}). A rate relation does not run in both directions")
        return detail
    if winner_deviation == 0 or implied_deviation == 0:
        # A ZERO DEVIATION IS NOT A RELATION AT ALL. The sweep never returns 1 (the vocabulary
        # excludes it) and the ladder's magnitude floor already refuses a flat reading, so this
        # is unreachable by construction -- and it is handled rather than divided by, because an
        # unreachable branch that raises is still a crash the day something makes it reachable.
        detail["reason"] = ("one of the two readings is exactly unity, which is not a rate "
                            "relation either instrument can be describing")
        return detail
    spread = abs(implied_deviation) / abs(winner_deviation)
    detail["magnitude_ratio"] = round(spread, 4)
    detail["magnitude_band"] = LADDER_CORROBORATION_BAND
    if spread > LADDER_CORROBORATION_BAND or spread < 1.0 / LADDER_CORROBORATION_BAND:
        detail["reason"] = (
            f"the two instruments agree on direction but not on size: the ladder implied a "
            f"deviation of {implied_deviation:+.7f} and the sweep confirmed {detail['winner']} "
            f"at {winner_deviation:+.7f}, a factor of {spread:.2f} apart, outside the "
            f"{LADDER_CORROBORATION_BAND}x band")
        return detail
    detail["corroborated"] = True
    detail["reason"] = (
        f"the ladder implied {ladder_implied_ratio} and the sweep independently confirmed "
        f"{detail['winner']} = {float(winner)}: same direction, and their deviations agree to a "
        f"factor of {spread:.3f}, inside the {LADDER_CORROBORATION_BAND}x band")
    return detail


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
    # SCOPE: THIS ARM SEES ONLY THE PRIMARY COUPLE, and that is a property of the gate, not an
    # oversight -- `similarity_gate` is called once, on `couples[0]`, because step 2 decides ONE
    # thing for the whole pair (is a sweep worth running) and the design does not fingerprint
    # every couple before answering it. MEASURED consequence, errid-24 on es: the primary covers
    # 0.977 so this arm passes, while two siblings sit at 0.306 and 0.304 -- under the floor and
    # invisible here. THE COUPLES THIS ARM CANNOT REACH ARE REFUSED ANYWAY, one layer down: the
    # per-couple screen in `chimeric` applies the SAME floor to every couple and logs each
    # exclusion as `couple_screened`, so those two contribute no holes and no cross-check
    # events. What is lost by the gate's narrow view is only the chance to skip the work, never
    # the refusal itself. Extending the gate to all couples would mean fingerprinting every
    # track before deciding whether to sweep -- the dominant cost of the whole step -- to buy a
    # decision the second screen already makes correctly.
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


# ---------------------------------------------------------------------------
# STEP 3 -- chimeric
# ---------------------------------------------------------------------------

def chimeric(factor, language, master_obj, candidate_obj, work_dir,
             primed_alignments=None, sweep_gate=None):
    """The ruling's `chimeric(speed_factor, language, master_obj, candidate_obj)`.

    Order, exactly as ruled: full-file fingerprints per file; THE SEQUENCE ALIGNMENT FUNCTION
    per couple; the multi-couple cross-check; then hole resolution. Returns `(ok, cause, reason,
    detail)` -- the orchestrator turns that into the owner's boolean at one place.

    `factor` defaults to 1 per the ruling ("Chimeric(facteur_vitesse=1 par defaut, ...)"), and
    at 1 NO FILTER RUNS AT ALL (ADDENDUM 6) -- the audio passes through untouched. That is
    stated as a branch here rather than left implicit, because "the filter happened to be a
    no-op" and "no filter ran" are different things in a pipeline that is being asked to prove
    it did not damage anything.

    `primed_alignments` carries the alignment the step-2 gate already computed for the primary
    couple, so the gate does not cost a second whole alignment of the same tracks -- EXCEPT when
    a comparison resample is applied, where every candidate-side reading in it describes audio
    that will not be used again and is dropped by name (see the `reprime` step below).

    AT ANY OTHER FACTOR THE CANDIDATE'S COMPARISON TRACKS ARE SPEED-CORRECTED ON EXTRACTION,
    inside the ffmpeg call that was going to run anyway -- no second decode, no file on disk that
    outlives one fingerprint pass, nothing that can reach a product. MEASURED end to end on the
    corpus's only real PAL pair (errid-70, fre, candidate 25 fps against a 23.976 master): blind,
    the aligner returns `all_segments_below_duration_floor` with coverage 0.000 and no zones;
    through the sweep (winner 1001/960, median fidelity 0.9774 at 5/5 probes) and this resample,
    the same couple returns 8 zones at coverage 0.8827 and decomposes into head + one interior
    hole of -992.5 ms + tail. That interior step is the content edit the bake-off dossier
    documents independently for this pair at "environ 0.96 s" between master t=460 s and 475 s
    -- two instruments, two sessions, 32 ms apart.
    """
    candidate_path = candidate_obj.filePath
    sample_rate = comparison_sample_rate(master_obj, candidate_obj, language)
    resample_routing = None
    if factor is None or factor == 1:
        step_result("no_filter", candidate=candidate_path, speed_factor=factor,
                    rule="ADDENDUM_6_no_filter_without_speed_change")
    else:
        step_launch("comparison_resample", candidate=candidate_path, speed_factor=factor)
        resample_routing, cause = comparison_resample(factor, master_obj, candidate_obj,
                                                      language, work_dir,
                                                      sample_rate=sample_rate)
        if resample_routing is None:
            step_result("comparison_resample", candidate=candidate_path, ok=False, cause=cause)
            return False, cause, (
                f"the pair carries a confirmed rate relation ({factor}) but the comparison "
                f"resample that would let the aligner measure across it could not be built "
                f"({cause}) -- without it the aligner cannot measure across the rate "
                f"relation, and no measurement was made"), None
        step_result("comparison_resample", candidate=candidate_path, ok=True,
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
                    # THE R3 VERDICT BELONGS IN THE STRUCTURED LINE, NOT ONLY IN THE PROSE. The
                    # finding was that the sentence asserted a confirmation the numbers did not
                    # support; fixing the sentence alone would leave a reader who greps the step
                    # log unable to tell an earned confirmation from a vacuous one.
                    pitch_test_discriminating=resample_routing["pitch_test_discriminating"],
                    pitch_tolerance_band=resample_routing["pitch_tolerance_band"],
                    inverting_case_detector=resample_routing["inverting_case_detector"],
                    inverting_case_observation=resample_routing["inverting_case_observation"],
                    rule=resample_routing["rule"])
        tools.dev_log(f"orchestrator: comparison_resample routing for {candidate_path}: "
                      f"{resample_routing['route_reason']}\n")

    couples = enumerate_couples(master_obj, candidate_obj, language)
    step_result("enumerate_couples", candidate=candidate_path, language=language,
                n_couples=len(couples), couples=couples)
    if not couples:
        return False, "no_stream_for_comparison_language", (
            f"neither side offers a pair of {language} audio streams to compare"), None

    fingerprints = dict(primed_alignments.get("fingerprints", {})
                        if primed_alignments else {})
    alignments = dict(primed_alignments.get("alignments", {}) if primed_alignments else {})
    if resample_routing is not None:
        # THE PRIMED WORK DESCRIBES A CANDIDATE THAT NO LONGER EXISTS, AND KEEPING ANY OF IT WOULD
        # BE THE WORST KIND OF REUSE. Step 2's gate fingerprints and aligns the primary couple
        # BLIND, on the uncorrected candidate -- that is what produced the "the aligner could not
        # measure" reading that sent us to the sweep in the first place. Every candidate-side
        # fingerprint and EVERY alignment is now stale: the candidate's points, its point count
        # and therefore its quantum all change under the correction. The MASTER side is kept, and
        # only the master side, because nothing was applied to it -- that is one whole-track
        # ffmpeg decode per master track saved (the design measured 7-15 s each, the dominant
        # cost of the entire step) without carrying forward a single number measured on audio the
        # aligner is no longer going to see.
        dropped_fingerprints = [key for key in fingerprints if key[0] != "master"]
        for key in dropped_fingerprints:
            del fingerprints[key]
        dropped_alignments = sorted(alignments)
        alignments = {}
        step_result("reprime", candidate=candidate_path,
                    kept_master_fingerprints=sorted(key[1] for key in fingerprints),
                    dropped_candidate_fingerprints=sorted(key[1]
                                                          for key in dropped_fingerprints),
                    dropped_alignments=dropped_alignments,
                    reason="blind_candidate_fingerprints_stale_under_comparison_resample")

    couple_results = []
    for master_stream, candidate_stream in couples:
        couple = f"{master_stream}x{candidate_stream}"
        for side, video_obj, stream in (("master", master_obj, master_stream),
                                         ("candidate", candidate_obj, candidate_stream)):
            key = (side, stream)
            if key in fingerprints:
                continue
            duration = _track_duration_seconds(video_obj, language, stream)
            if duration is None:
                return False, "track_duration_unmeasurable", (
                    f"the {side} {language} stream {stream} carries no readable duration, so "
                    f"there is no length to fingerprint it over"), None
            # THE FILTER RIDES ON THE CANDIDATE SIDE ONLY, and `effective_ratio` -- not the
            # requested one -- sets the corrected length. The master defines the grid; correcting
            # it too would move the reference the whole campaign measures against.
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
            points, quantum_ms = fingerprint_track(
                video_obj, language, stream, side, work_dir, sample_rate, duration,
                audio_filter=track_filter, output_duration_seconds=corrected_duration)
            step_result("fingerprint", candidate=candidate_path, side=side, stream=stream,
                        n_points=len(points) if points else 0,
                        quantum_ms=round(quantum_ms, 4) if quantum_ms else None,
                        resampled=bool(track_filter),
                        seconds=round(time.time() - started, 2))
            if points is None:
                return False, "fingerprinting_raised", (
                    f"the {side} {language} stream {stream} could not be extracted or "
                    f"fingerprinted"), None
            fingerprints[key] = (points, quantum_ms, corrected_duration)

        if couple not in alignments:
            fp_master, quantum_master, duration_master = fingerprints[("master", master_stream)]
            fp_candidate, quantum_candidate, duration_candidate = fingerprints[
                ("candidate", candidate_stream)]
            step_launch("align", candidate=candidate_path, couple=couple,
                        n_master=len(fp_master), n_candidate=len(fp_candidate))
            started = time.time()
            alignments[couple] = banded_seed_alignment.b2_align(
                fp_master, fp_candidate, quantum_master,
                candidate_quantum_ms=quantum_candidate,
                duration_diff_ms=abs(duration_master - duration_candidate) * 1000.0,
                signed_duration_diff_ms=(duration_candidate - duration_master) * 1000.0,
                shorter_duration_ms=min(duration_master, duration_candidate) * 1000.0)
            alignments[couple]["alignment_seconds"] = time.time() - started
        alignment = alignments[couple]
        step_result("align", candidate=candidate_path, couple=couple,
                    verdict=alignment["verdict"], n_zones=len(alignment.get("zones") or []),
                    n_cut_zones=len(alignment.get("cut_zones") or []),
                    overlaps_resolved=alignment.get("segments_overlap_resolved"),
                admitted_self_evident=alignment.get("admitted_self_evident"),
                    coverage=alignment.get("master_axis_coverage_fraction"),
                    residual_fraction=alignment.get("residual_fraction"),
                    seconds=round(alignment["alignment_seconds"], 2))

        # THE ORCHESTRATOR BRANCHES ON `zones`, NEVER ON `cut_zones` (design section 3.4c).
        # `single_segment_no_cut` is a SUCCESS: one long aligned zone with no interior hole is a
        # fully aligned pair, and its head/tail holes may still be entirely real -- that is
        # exactly the shape of the pair with a 4.3 s head and a 173 s tail.
        if alignment["verdict"] in ALIGNMENT_COULD_NOT_MEASURE_VERDICTS:
            tools.dev_log(f"orchestrator: couple {couple} could not be aligned "
                          f"({alignment['verdict']}) -- recorded, and the remaining couples "
                          f"still run: one blind track is not a verdict about the pair\n")
            continue

        # THE SAME COVERAGE FLOOR THE STEP-2 GATE USES, APPLIED PER COUPLE, because the gate only
        # ever sees the PRIMARY couple and a pair can carry both kinds at once. MEASURED,
        # errid-24 on es: four couples reading coverage 0.977 / 0.306 / 0.304 / 0.976 -- two
        # healthy and two under the floor, on one pair, in one language. Whichever of the four
        # happens to be first decides what the gate sees, so without this screen a healthy
        # primary lets two couples that aligned almost nothing into the hole decomposition (18
        # holes each) and into the cross-verification, where they can only add noise to an
        # agreement test about events they were never in a position to see. A couple under the
        # floor is `could-not-see`, which is what the cross-check already has a name for.
        couple_coverage = alignment.get("master_axis_coverage_fraction")
        if couple_coverage is None or couple_coverage < MASTER_AXIS_COVERAGE_FLOOR:
            step_result("couple_screened", candidate=candidate_path, couple=couple,
                        verdict=alignment["verdict"], coverage=couple_coverage,
                        floor=MASTER_AXIS_COVERAGE_FLOOR,
                        reason="master_axis_coverage_below_floor",
                        rule="a_verdict_token_is_not_a_measurement_of_how_much_lined_up")
            continue

        holes = holes_for_couple(alignment)
        # THE STEPS TRAVEL WITH THE COUNT. A hole whose step is 0 is a stretch the aligner
        # lost the thread on with NO offset change either side -- the `no_cut_confirmed`
        # candidate class of ADDENDUM 3, where the video is expected to close the hole -- and a
        # reader who sees only "9 holes" cannot tell that population from nine real edits.
        step_result("holes", candidate=candidate_path, couple=couple, n_holes=len(holes),
                    kinds=[hole["kind"] for hole in holes],
                    steps_ms=[(round(hole["step_ms"], 1) if hole["step_ms"] is not None
                                else None) for hole in holes],
                    master_spans_s=[round(hole["master_span_seconds"], 2) for hole in holes],
                    merged=[hole["merged_from"] for hole in holes],
                    merged_zone_lengths_s=[hole["merged_zone_lengths_seconds"]
                                            for hole in holes],
                    edge_addition_s=round(edge_addition_seconds(holes), 3))
        couple_results.append({"couple": couple, "alignment": alignment, "holes": holes})

    if not couple_results:
        # EVERY couple was blind, or every couple that was not blind covered too little to be
        # read. That is a property of the pair, and it gets a token that says WHICH of the two,
        # so the ledger can tell them apart -- the aligner's own vocabulary when it never
        # anchored anything, the coverage token when it anchored something too small to trust.
        verdicts = {alignments[f"{m}x{c}"]["verdict"] for m, c in couples
                    if f"{m}x{c}" in alignments}
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
        coverages = sorted(
            round(alignments[f"{m}x{c}"].get("master_axis_coverage_fraction") or 0.0, 4)
            for m, c in couples if f"{m}x{c}" in alignments)
        return False, cause, (
            f"no couple of {language} produced a usable alignment; the aligner reported "
            f"{sorted(verdicts)} across {len(couples)} couples, covering {coverages} of the "
            f"master axis against a floor of {MASTER_AXIS_COVERAGE_FLOOR}"), None

    step_launch("cross_verify", candidate=candidate_path, n_couples=len(couple_results))
    report = cross_verify_couples(couple_results)
    log_cross_verification(candidate_path, report)
    if not report["agree"]:
        return False, "intercouple_disagreement", (
            f"{len(report['disagreements'])} of {len(report['clusters'])} event clusters "
            f"disagree across {report['n_couples']} couples of {language}; every couple's "
            f"position, step, quantum, residual and coverage is in the log above"), report

    # WHICH COUPLE'S HOLES DRIVE THE PLAN -- STATED, NOT ASSUMED. The cross-check proves the
    # couples agree about the EVENTS; it does not merge their hole lists, and the design does
    # not settle which list the resolver walks (the zone counts differ by a factor of three on
    # real media for the same physical pair). The first couple is used and the choice is
    # LOGGED as provisional, so a later stage replaces a recorded decision rather than
    # discovering an accident.
    driving = couple_results[0]
    holes = driving["holes"]
    tagged, tag_reason = chimeric_tag_required(holes)
    step_result("plan_shape", candidate=candidate_path, driving_couple=driving["couple"],
                driving_choice="provisional_first_couple", n_holes=len(holes),
                edge_addition_s=round(edge_addition_seconds(holes), 3),
                chimeric_tag=tagged, chimeric_tag_reason=tag_reason.replace(" ", "_"))

    if any(hole["kind"] == "spans_whole_file" for hole in holes):
        return False, "alignment_no_anchored_runs", (
            f"after the <{HOLE_MERGE_WINDOW_SECONDS}s merge the couple {driving['couple']} "
            f"carries one hole spanning the whole file -- no aligned zone survived long enough "
            f"to anchor either end, so there is no head, interior or tail to resolve"), None

    if len(holes) > MAX_HOLES_PER_COUPLE:
        return False, "hole_count_exceeds_resolver_budget", (
            f"couple {driving['couple']} decomposes into {len(holes)} holes after the "
            f"<{HOLE_MERGE_WINDOW_SECONDS}s merge, above the budget of "
            f"{MAX_HOLES_PER_COUPLE}; resolving each one launches an unbounded frame-exact "
            f"search, and a pair that fragments this far is not one this instrument has "
            f"measured itself able to reconstruct"), None

    # ADDENDUM 5's good news, named: the alignment may find NO hole at all -- completely
    # compatible audios, a simple offset. That is not an empty result, it is the easy one.
    if not holes:
        step_result("holes", candidate=candidate_path, couple=driving["couple"],
                    n_holes=0, verdict="audios_fully_compatible_offset_only")

    # THE FRAME DOMAIN, ONCE PER PAIR: exact rational grids for both files, the master's
    # timeline, and the rate relation the candidate's alignment milliseconds are expressed in.
    # A pair whose grid cannot be read exactly cannot have a single hole resolved, so it is
    # refused here, by name, before any video is decoded.
    # ONCE PER PAIR, HOLES OR NOT: plan application needs the exact grid and the master's
    # timeline even on a pair whose every hole closed, or that had none.
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
    # THE ALIGNER'S TRACK-RELATIVE OFFSETS, PUT ON THE FILE'S CLOCK ONCE, BEFORE ANY VIDEO IS
    # SEARCHED -- see `couple_start_delta_ms`. Logged with both starts, always.
    master_stream, candidate_stream = driving["couple"].split("x")
    delta_ms, master_start_ms, candidate_start_ms = couple_start_delta_ms(
        master_obj, candidate_obj, language, master_stream, candidate_stream, factor)
    domain["track_delay_delta_ms"] = float(delta_ms)
    step_result("track_delay_fold", candidate=candidate_path, couple=driving["couple"],
                master_start_ms=float(master_start_ms),
                candidate_start_ms=float(candidate_start_ms),
                delta_ms=float(delta_ms),
                rule="file_time_offset=track_offset+candidate_start*r-master_start")

    # SEQUENTIAL, ONE HOLE AFTER THE OTHER, EVERY HOLE RESOLVED EVEN AFTER ONE DECLINES -- so the
    # log holds the whole pair's reading, not the first refusal's. Stage 7 parallelises this.
    outcomes = []
    for index, hole in enumerate(holes):
        step_launch("resolve_hole", candidate=candidate_path, hole=index, kind=hole["kind"],
                    why=hole["why_token"], master_span_s=round(hole["master_span_seconds"], 3),
                    candidate_span_s=round(hole["candidate_span_seconds"], 3),
                    step_ms=(round(hole["step_ms"], 1) if hole["step_ms"] is not None
                              else None),
                    merged_from=hole["merged_from"])
        started = time.time()
        outcome = resolve_hole(dict(hole, frame_domain=domain,
                                    quantum_ms=driving["alignment"]["quantum_ms"]),
                               master_obj, candidate_obj, work_dir)
        # A STATUS FROM OUTSIDE THE CLOSED VOCABULARY IS A RESOLVER BUG, AND IT IS NOT ALLOWED TO
        # PASS AS A RESOLUTION: it becomes a named decline, loudly, rather than a hole the plan
        # would read frames off.
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
        # (candidat, position, verdict video) puis ferme".
        if outcome.get("refuted_proposal"):
            step_result("refuted_proposal", candidate=candidate_path, hole=index,
                        **outcome["refuted_proposal"])
        # ADDENDUM 4, POINT 3: "jamais un placement silencieux" -- both walks and the span.
        if outcome["status"] == HOLE_PINNED_TO_AMBIGUOUS_ZONE_END:
            step_result("boundary_pinned_to_ambiguous_zone_end", candidate=candidate_path,
                        hole=index, cause="static_span_ambiguity",
                        forward_walk_frames=outcome["forward_walk_frames"],
                        backward_walk_frames=outcome["backward_walk_frames"],
                        span_frames=outcome["span_frames"],
                        ambiguous_frames=outcome["ambiguous_frames"],
                        pin_frame=outcome["pin_frame"],
                        anchor_b_frame=outcome["anchor_b_frame"],
                        fill_master_frames=[outcome["master_start_frame"],
                                            outcome["master_end_frame"]])
        outcomes.append((hole, outcome))

    declined = [(index, hole, outcome) for index, (hole, outcome) in enumerate(outcomes)
                if outcome["status"] == HOLE_DECLINED]
    if declined:
        return False, "hole_resolution_declined", (
            f"{len(declined)} of {len(holes)} hole(s) could not be pinned to a frame: "
            + "; ".join(f"hole {index} ({hole['kind']}) {outcome.get('resolver_reason')}"
                        for index, hole, outcome in declined)
            + " -- no boundary was invented for them, and a plan with an unresolved hole is "
              "not a plan"), None

    # ADDENDUM 3: a closed hole restores zone continuity and is NOT a failure -- it simply
    # leaves the plan. What remains is the work actually to be done, and ADDENDUM 5's marker is
    # decided on THAT, with the edges' COUNTED additions in place of the audio's estimate.
    effective = []
    for hole, outcome in outcomes:
        if outcome["status"] == HOLE_NO_CUT_CONFIRMED:
            continue
        entry = dict(hole, resolution=outcome)
        if hole["kind"] in ("head", "tail"):
            entry["resolved_edge_addition_seconds"] = outcome["edge_addition_seconds"]
        effective.append(entry)
    tagged, tag_reason = chimeric_tag_required(effective)
    step_result("plan_shape_resolved", candidate=candidate_path,
                n_holes=len(holes), n_closed=len(holes) - len(effective),
                statuses=[outcome["status"] for _hole, outcome in outcomes],
                edge_addition_s=round(edge_addition_seconds(effective), 3),
                chimeric_tag=tagged, chimeric_tag_reason=tag_reason.replace(" ", "_"))
    if holes and not effective:
        step_result("holes", candidate=candidate_path, couple=driving["couple"],
                    verdict="all_holes_no_cut_confirmed",
                    rule="ADDENDUM_3_video_disposes_pair_merges_without_splice")

    # THE COARSE OFFSET OF A PAIR WITH NO HOLE LEFT BEFORE ITS FIRST ZONE: the driving
    # alignment's own offset on its longest coalesced zone. Only the refinement's CENTRE --
    # `apply_plan` measures the applied offset itself.
    coalesced, coalesced_detail = coalesce_same_offset_zones(
        driving["alignment"].get("zones") or [], driving["alignment"].get("zones_detail") or [])
    default_offset_ms = None
    if coalesced_detail:
        longest = max(coalesced_detail,
                      key=lambda detail: detail["master_points"][1] - detail["master_points"][0])
        default_offset_ms = Decimal(str(round(
            longest["offset_points"] * driving["alignment"]["quantum_ms"]
            + domain["track_delay_delta_ms"], 6)))
    ok, cause, reason = apply_plan(candidate_path, effective, factor, master_obj, candidate_obj, {
        "language": language, "work_dir": work_dir, "domain": domain,
        "quantum_ms": driving["alignment"]["quantum_ms"],
        "master_stream": master_stream, "candidate_stream": candidate_stream,
        "default_offset_ms": default_offset_ms, "resample_routing": resample_routing,
        "sweep_gate": sweep_gate, "tagged": tagged, "tag_reason": tag_reason})
    return ok, cause, reason, None


# ---------------------------------------------------------------------------
# THE ORCHESTRATOR
# ---------------------------------------------------------------------------

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
    if master_intertrack_cache is None:
        master_intertrack_cache = {}
    work_dir = work_root or path.join(tools.tmpFolder, "repair", "orchestrator")
    tools.make_dirs(work_dir)
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
    # Gated on the ALIGNER'S OWN OUTPUT (design section 3.6, ordering B), which means the
    # primary couple is aligned here and the result is handed to step 3 rather than recomputed.
    primed = {"fingerprints": {}, "alignments": {}}
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

    primary_name = f"{couples[0][0]}x{couples[0][1]}"
    step_launch("similarity_gate", candidate=candidate_path, couple=primary_name)
    gate_ok, gate_cause, gate_reason, gate_detail = _prime_primary_couple(
        master_obj, candidate_obj, comparison_language, work_dir, couples[0], primed)
    if not gate_ok:
        _plan_line("none", candidate_path, step="similarity_gate", cause=gate_cause)
        return _terminal(candidate_path, "no_plan", gate_cause, gate_reason,
                         detail=gate_detail)

    primary = primed["alignments"][primary_name]
    should_sweep, gate_prose, observations = similarity_gate(primary)
    step_result("similarity_gate", candidate=candidate_path, should_sweep=should_sweep,
                **{key: value for key, value in observations.items()})
    if should_sweep:
        step_launch("speed_sweep", candidate=candidate_path, language=comparison_language)
        winner, sweep_gate, sweep_cause = speed_factor(
            master_obj, candidate_obj, comparison_language)
        step_result("speed_sweep", candidate=candidate_path,
                    factor=(f"{winner.numerator}/{winner.denominator}"
                             if isinstance(winner, Fraction) else winner),
                    cause=sweep_cause,
                    median_fidelity=(sweep_gate or {}).get("median_fidelity"),
                    margin=(sweep_gate or {}).get("margin"),
                    passing=(sweep_gate or {}).get("passing"))
        # THE CORROBORATION GUARD -- ONLY ON THE LADDER ARM, AND ONLY WHEN THE SWEEP CONFIRMED.
        # This is the branch the non-terminal promise forgot. A ladder-armed sweep that DECLINES
        # is handled below and always was; a ladder-armed sweep that CONFIRMS used to be accepted
        # unconditionally, and an independent tester measured what that costs on a healthy pair:
        # the factor gets applied, the candidate gets resampled, and the plan is rebuilt on
        # deliberately speed-altered audio (errid-202: 6 holes -> 12, its real -992.37 ms step
        # dissolved into scatter). See `corroborate_sweep_against_ladder`.
        #
        # SCOPE, STATED BECAUSE IT IS THE WHOLE POINT: this runs ONLY when the gate fired on
        # `rate_relation_signature`. The other two arms (`alignment_could_not_measure`,
        # `master_axis_coverage_below_floor`) are untouched, because on those the aligner
        # produced no ladder to corroborate against and the sweep is the ONLY instrument in the
        # room -- errid-70 reaches its 1001/960 through `alignment_could_not_measure` and never
        # comes near this code.
        ladder_armed = observations.get("gate_arm") == "rate_relation_signature"
        corroboration = None
        if winner is not None and ladder_armed:
            corroboration = corroborate_sweep_against_ladder(
                winner, observations.get("ladder_implied_speed_ratio"))
            step_result("sweep_corroboration", candidate=candidate_path,
                        **{key: value for key, value in corroboration.items()
                           if key != "reason"})
            tools.dev_log(f"orchestrator: sweep corroboration for {candidate_path}: "
                          f"{corroboration['reason']}\n")
            if not corroboration["corroborated"]:
                # DISCARDED, NOT DECLINED. Dropping the winner routes this into the non-terminal
                # branch below, which is exactly where a ladder-armed suggestion belongs once it
                # has failed to be corroborated: the pair continues on the alignment already
                # measured, at factor 1, and no filter ever runs (ADDENDUM 6).
                winner = None
        if winner is None and ladder_armed:
            # THE RATE ARM'S REFUSAL IS NOT TERMINAL, AND THAT ASYMMETRY IS DELIBERATE -- see
            # `similarity_gate`. This arm fired on an alignment that SUCCEEDED; the sweep was
            # asked because the zones looked like a rate ladder, and either it answered no or its
            # answer was not corroborated. Declining here would convert a suggestion into a
            # refusal and manufacture a new false-decline family every time the ladder reading
            # misfired on a healthy pair. The honest continuation is the alignment we already
            # have, at factor 1, with the reason recorded so nobody has to wonder why a sweep ran.
            step_result("speed_sweep", candidate=candidate_path,
                        arm="rate_relation_signature", terminal=False,
                        cause=(sweep_cause if corroboration is None
                               else "sweep_winner_uncorroborated_by_ladder"),
                        corroborated=(None if corroboration is None
                                      else corroboration["corroborated"]),
                        continuing="at_factor_1_with_the_blind_alignment",
                        rule="a_suggestion_that_was_refused_is_not_a_refusal_of_the_pair")
            tools.dev_log(
                f"orchestrator: the rate-ladder arm asked for a sweep on {candidate_path} and "
                + (f"the sweep declined ({sweep_cause} / "
                   f"{(sweep_gate or {}).get('cause')})" if corroboration is None
                   else f"the sweep's answer was NOT corroborated -- "
                        f"{corroboration['reason']}")
                + f"; the pair CONTINUES at speed_factor 1 on the alignment already measured "
                  f"-- this arm suggests, it does not refuse\n")
        elif winner is None:
            _plan_line("none", candidate_path, step="speed_sweep", cause=sweep_cause)
            return _terminal(
                candidate_path, "no_plan", sweep_cause,
                f"mean similarity is low ({gate_prose}) and the rate sweep could not raise it "
                f"({(sweep_gate or {}).get('cause')})", detail={"resample_gate": sweep_gate})
        else:
            factor = winner

    # ---- STEP 3: chimeric ---------------------------------------------------
    step_launch("chimeric", candidate=candidate_path, language=comparison_language,
                speed_factor=(f"{factor.numerator}/{factor.denominator}"
                               if isinstance(factor, Fraction) else factor))
    ok, cause, reason, detail = chimeric(factor, comparison_language, master_obj,
                                         candidate_obj, work_dir, primed_alignments=primed,
                                         sweep_gate=sweep_gate)
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


def _prime_primary_couple(master_obj, candidate_obj, language, work_dir, couple, primed):
    """Fingerprint and align the PRIMARY couple once, for the step-2 gate, and keep both so
    step 3 reuses them instead of paying for the same whole-file work twice.

    Returns `(ok, cause, reason, detail)`. Extracted as its own function so that the gate's
    cost is visible at the call site: this is where the two ffmpeg whole-track decodes happen,
    which the design measured as the dominant cost of the entire step (7-15 s per track, against
    0.15 s for fingerprinting and 0.6-1.4 s for the alignment itself).
    """
    candidate_path = candidate_obj.filePath
    master_stream, candidate_stream = couple
    sample_rate = comparison_sample_rate(master_obj, candidate_obj, language)
    for side, video_obj, stream in (("master", master_obj, master_stream),
                                     ("candidate", candidate_obj, candidate_stream)):
        duration = _track_duration_seconds(video_obj, language, stream)
        if duration is None:
            return (False, "track_duration_unmeasurable",
                    f"the {side} {language} stream {stream} carries no readable duration, so "
                    f"there is no length to fingerprint it over", None)
        step_launch("fingerprint", candidate=candidate_path, side=side, stream=stream,
                    duration_s=round(duration, 3), sample_rate=sample_rate)
        started = time.time()
        points, quantum_ms = fingerprint_track(video_obj, language, stream, side, work_dir,
                                               sample_rate, duration)
        step_result("fingerprint", candidate=candidate_path, side=side, stream=stream,
                    n_points=len(points) if points else 0,
                    quantum_ms=round(quantum_ms, 4) if quantum_ms else None,
                    seconds=round(time.time() - started, 2))
        if points is None:
            return (False, "fingerprinting_raised",
                    f"the {side} {language} stream {stream} could not be extracted or "
                    f"fingerprinted", None)
        primed["fingerprints"][(side, stream)] = (points, quantum_ms, duration)

    fp_master, quantum_master, duration_master = primed["fingerprints"][
        ("master", master_stream)]
    fp_candidate, quantum_candidate, duration_candidate = primed["fingerprints"][
        ("candidate", candidate_stream)]
    name = f"{master_stream}x{candidate_stream}"
    step_launch("align", candidate=candidate_path, couple=name, n_master=len(fp_master),
                n_candidate=len(fp_candidate))
    started = time.time()
    alignment = banded_seed_alignment.b2_align(
        fp_master, fp_candidate, quantum_master,
        candidate_quantum_ms=quantum_candidate,
        duration_diff_ms=abs(duration_master - duration_candidate) * 1000.0,
        signed_duration_diff_ms=(duration_candidate - duration_master) * 1000.0,
        shorter_duration_ms=min(duration_master, duration_candidate) * 1000.0)
    alignment["alignment_seconds"] = time.time() - started
    primed["alignments"][name] = alignment
    step_result("align", candidate=candidate_path, couple=name,
                verdict=alignment["verdict"], n_zones=len(alignment.get("zones") or []),
                n_cut_zones=len(alignment.get("cut_zones") or []),
                overlaps_resolved=alignment.get("segments_overlap_resolved"),
                admitted_self_evident=alignment.get("admitted_self_evident"),
                coverage=alignment.get("master_axis_coverage_fraction"),
                residual_fraction=alignment.get("residual_fraction"),
                seconds=round(alignment["alignment_seconds"], 2))
    return True, None, None, None
