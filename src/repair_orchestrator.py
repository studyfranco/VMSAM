# -*- coding: utf-8 -*-
"""
repair_orchestrator.py -- the owner's flat orchestrator (RULING_20260922_ORCHESTRATOR_
ARCHITECTURE.MD, plus its seven addenda), stage 1 of the staged switch described in
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
                                                          the candidate's extraction only, and
                                                          never delivered (ADDENDUM 7)
    STUB     step 4  frame-exact hole resolution       -- declines `hole_resolution_
                                                          not_implemented`
    STUB     step 5  plan application                  -- declines `plan_application_
                                                          not_implemented`, or
                                                          `restoration_deferred` on a
                                                          rate-family pair (addendum 7)

*** NOT WIRED INTO THE LIVE CHAIN. Nothing in `src/` calls `repair()` yet. The owner's ruling
is REMPLACEMENT DIRECT -- one switch, no cohabitation flag -- and that switch is the design's
stage 6, not this one. Until then `merge_video_repair.repair_not_compatible_videos` remains the
production entry unchanged, and this module is driven directly against real pairs. ***

RETURN CONTRACT: a BOOLEAN (owner's ADDENDUM 1, point 4). True = a plan was found AND the
temporary chimeric file was created successfully. False = everything else. The boolean has NO
room for "could not measure", which the standing invariant says must never be confused with
"measured and refused" -- so that distinction lives in the TOKENS, not in the return: every
False leaves behind a `cause=<token>` whose MEASUREMENT CLASS is emitted beside it, one of

    ran_conclusive_negative   the instrument ran and returned a negative
    could_not_run             the instrument could not be run, or is not built yet
    owner_choice              neither -- a deferral the owner ordered, cited verbatim

A reader who sees `False` and wants to know which of the three it was reads the token; a reader
who never looks gets a boolean that is safe either way, because all three mean "do not ship".

SEQUENTIAL, DELIBERATELY. The design's section 4 measured that this code path carries NO
threading today (`frame_compare.py:5` imports `Thread` and never calls it -- a dead import),
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
# about a pair one frame rate apart -- which also made the owner's `restoration_deferred`
# deferral (ADDENDUM 7) unreachable, since that gate tests `speed_factor != 1`. The same pair on
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
# RUNG COUNT separates 32 from {0,0,0,0,1,1,1,3,8,12}; MONOTONICITY separates 0.9375 from a
# nearest negative of 0.500. The two pairs that get past the rung count -- errid-213 and
# errid-100, the two the bake-off and the locator both flagged as line-fit traps -- are refused
# on direction, which is the condition that rests on a MECHANISM rather than on a sample: a rate
# relation drifts one way for the whole file, and an editor's cuts do not.
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
# number. The detection floor is set by the magnitude condition below, which scales with the
# file: requiring |f-1| >= 5e-4 already requires the total rise to be at least 5e-4 of the span,
# so a longer file needs proportionally more rungs on its own. This constant exists only so that
# `rung_fraction` and `rung_monotone_fraction` are computed over enough samples to mean
# something. 8 sits above the largest negative that is not one of the two line-fit traps (3) and
# a quarter of the way to the positive (32). CONSEQUENCE, STATED BECAUSE IT IS A REAL LIMIT: at
# the smallest named deviation (1001/1000) and a ~124 ms quantum, 8 rungs need 8*0.124/0.000999
# = 993 s of aligned span, so this arm cannot see an NTSC relation on anything under ~16.5
# minutes and does not claim to. A 24-minute episode yields ~11.7 rungs; the 62-minute pair
# measured above yielded 32.
LADDER_MIN_RUNGS = 8
# PURITY: measured 0.970 on the positive against 0.250-0.500 on the four ordinary edit pairs.
# 0.80 sits between them. It does NOT reject errid-213 (0.889) -- direction does.
LADDER_MIN_RUNG_FRACTION = 0.80
# DIRECTION: the condition that actually decides at the boundary. Measured 0.9375 on the
# positive against 0.500 on both line-fit traps. 0.85 sits between, nearer the negative side
# than the positive's value, so the positive keeps 0.09 of margin and the negatives 0.35.
LADDER_MIN_RUNG_MONOTONE_FRACTION = 0.85

# THE MAGNITUDE FLOOR, IMPORTED NOT RESTATED. `change_point_locator.RATE_SLOPE_MIN_FACTOR_
# DEVIATION` is 5e-4, derived there as HALF the smallest deviation in the named rate vocabulary
# (1001/1000, |f-1| = 1/1001 = 9.99e-4) and landed on a measured false fire. The locator dies in
# the switch and its measurement does not, so this reads the constant while the module still
# exists and falls back to the same literal -- with the fallback logged, never silent -- for the
# day it does not.
try:
    import change_point_locator as _change_point_locator
    RATE_LADDER_MIN_FACTOR_DEVIATION = float(
        _change_point_locator.RATE_SLOPE_MIN_FACTOR_DEVIATION)
    _RATE_LADDER_DEVIATION_SOURCE = "change_point_locator.RATE_SLOPE_MIN_FACTOR_DEVIATION"
except Exception:                                                        # noqa: BLE001
    RATE_LADDER_MIN_FACTOR_DEVIATION = 5e-4
    _RATE_LADDER_DEVIATION_SOURCE = "literal fallback -- change_point_locator unimportable"

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
CLASS_OWNER_CHOICE = "owner_choice"

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
    # the stubs -- honest declines for stages not yet landed
    "hole_resolution_not_implemented": CLASS_COULD_NOT_RUN,
    "plan_application_not_implemented": CLASS_COULD_NOT_RUN,
    # the owner's deferral
    "restoration_deferred": CLASS_OWNER_CHOICE,
}

# THE HOLE-RESULT VOCABULARY. `no_cut_confirmed` is FIRST CLASS (owner's ADDENDUM 3): the audio
# alignment PROPOSES a zone to examine, the video pHash DISPOSES, and a hole the walk crosses
# without dropping out is a hole that CLOSES -- not a failure. The orchestrator restores zone
# continuity across it, and if every hole closes this way the pair merges with a simple offset
# and no splice at all.
HOLE_RESOLVED = "resolved"
HOLE_NO_CUT_CONFIRMED = "no_cut_confirmed"
HOLE_NOT_IMPLEMENTED = "not_implemented"

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


def fingerprint_track(video_obj, language, stream_order, side, work_dir, sample_rate,
                      duration_seconds, audio_filter=None, output_duration_seconds=None):
    """One whole-file fingerprint list for ONE track. Returns `(points, quantum_ms)`, or
    `(None, None)` when the track could not be read.

    ON `audio_filter` / `output_duration_seconds` -- THE COMPARISON RESAMPLE, AND WHY THE SECOND
    ARGUMENT IS NOT OPTIONAL ONCE THE FIRST IS GIVEN. `duration_seconds` bounds what is READ from
    the source (it lands on ffmpeg's `-t`, before `-i`); a speed filter changes what is WRITTEN.
    So a corrected extraction is `duration_seconds * effective_ratio` long, and BOTH the fpcalc
    `-length` and the quantum must be computed from THAT, not from the input length. Passing the
    input length would (a) tell fpcalc to stop early and truncate exactly the tail the correction
    just restored, and (b) divide the real span by the wrong number and hand every consumer a
    quantum that is wrong by the rate relation -- a per-track quantum silently off by 4.27 % is
    worse than no quantum, because everything downstream would keep working and be wrong.
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
    What unequal counts DO produce is two different quanta, which is why each side's quantum is
    computed here from ITS OWN count and carried separately -- the standing per-track-quantum
    invariant, measured live on errid-232 at 124.0636 ms master vs 124.0418 ms candidate.

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
    return points, output_duration_seconds * 1000.0 / len(points)


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
    """
    total = 0.0
    for hole in holes:
        if hole["kind"] in ("head", "tail"):
            total += hole["candidate_span_seconds"]
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
    if reading.get("refusal") is None:
        routing["route_reason"] = (
            f"the pitch layer confirms the pitch moved with the speed (measured "
            f"{measured}, applied {float(speed_factor):.7f}, peak {reading.get('peak')}): this "
            f"is the naive-speedup family, and asetrate is its exact inverse -- it undoes speed "
            f"AND pitch together (ruling body, step 2)")
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

    A MEASUREMENT TOOL, AND ONLY THAT (owner's ADDENDUM 7 point 1): "la correction de vitesse
    reste un OUTIL DE MESURE : le sweep et le resample de comparaison servent a ALIGNER et
    LOCALISER". NOTHING PRODUCED HERE IS EVER SHIPPED. The corrected audio exists for the length
    of one fingerprint pass and is deleted by `fingerprint_track`'s own `finally`; the delivery
    question -- which corrected audio a product should carry -- is SUSPENDED until the owner
    validates examples himself, and `apply_plan` declines `restoration_deferred` when a
    rate-family pair reaches it.

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
        "rule": "ADDENDUM_7_measurement_tool_never_delivered",
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


def resolve_hole(hole, master_obj, candidate_obj, work_dir):
    """STUB -- design stage 4. Returns a result dict whose `status` is `not_implemented`.

    WHAT IT WILL DO, and every piece of it already exists:
      interior  -> `scene_anchor.locate_scene_anchors` (scene detection at +/-10 s both sides,
                   nearest compatible pHash-validated anchors, then `run_edge_walk` to pin the
                   boundary) -- it already returns exactly the ruling's "positions de frames
                   debut et fin EXACTES sur master et candidat".
      head/tail -> `scene_anchor.locate_edge_boundary`, the single-anchor bounded walk with its
                   three named terminations (`sustained_mismatch`, `master_exhausted`,
                   `candidate_exhausted`).

    AND IT CAN RETURN `no_cut_confirmed` (owner's ADDENDUM 3), which the orchestrator already
    handles: the anchors land, the walk crosses the candidate hole WITHOUT dropping out, so
    there is no cut -- the hole CLOSES and that is a first-class success. The hierarchy is
    graven: audio alignment PROPOSES the zones to examine, video pHash DISPOSES.

    AND `boundary_pinned_to_right_anchor` (owner's ADDENDUM 4), for a hole whose content is a
    static span where every frame matches every other at pHash: the two walks cross it at
    different speeds and the true position is undecidable inside it, so the boundary is pinned
    to the frames JUST BEFORE THE RIGHT-HAND ANCHOR -- a deterministic convention producing the
    same bytes every run, emitted as a NAMED verdict with both walk lengths and the span size,
    never as a silent placement.

    A STUB RETURNS A REFUSAL, NEVER A PLAUSIBLE ANSWER. Returning a guessed boundary here would
    be the one failure this whole campaign exists to prevent.
    """
    return {"modality": MODALITY, "status": HOLE_NOT_IMPLEMENTED,
            "kind": hole["kind"], "why_token": hole["why_token"],
            "cause": "hole_resolution_not_implemented"}


def apply_plan(candidate_path, holes, speed_factor, master_obj, candidate_obj):
    """STUB -- design stage 5. Returns `(False, cause, reason)`.

    THE RESTORATION GATE IS REAL AND LANDS HERE, NOT IN A LATER STAGE, because this is the
    DELIVERY decision and the owner's ADDENDUM 7 is about delivery: no speed-corrected audio
    ships in any product until the owner has validated examples. On a rate-family pair the
    repair delivers what the alignment permits WITHOUT restored audio -- retimed subtitles,
    untransformed candidate tracks where they align, intact master tracks (the errid-195 pattern,
    audio alignment as the vehicle for measuring the subtitle recalage). If nothing is
    deliverable on those terms, the decline is `restoration_deferred`, and that is an OWNER
    CHOICE cited in the prose, NOT a measurement failure -- which is why its measurement class is
    `owner_choice` and not `ran_conclusive_negative`.

    At `speed_factor` of 1 or None nothing was ever speed-corrected, so the gate does not apply
    and the honest decline is simply that assembly is not built yet.
    """
    if speed_factor is not None and speed_factor != 1:
        return False, "restoration_deferred", (
            "the pair is a rate-family pair and speed correction is a MEASUREMENT tool only: "
            "the owner has suspended the restoration filter question until he validates "
            "examples himself (RULING_20260922_ORCHESTRATOR_ARCHITECTURE.MD ADDENDUM 7), so no "
            "speed-corrected audio may be delivered. What the alignment permits without "
            "restored audio -- retimed subtitles, untransformed candidate tracks, intact master "
            "tracks -- is not assembled yet (design stage 5), so nothing ships for this pair. "
            "This is the owner's deferral, not a measurement failure")
    return False, "plan_application_not_implemented", (
        "the alignment and the hole decomposition completed, but plan application "
        "(normalize_segments on pre-resolved boundaries, assemble, retime, mux, verify) is "
        "design stage 5 and is not built yet -- no file was produced and none is claimed")


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
    collapse to at best one. The LADDER STRUCTURE survives the quantisation intact, because it
    IS the quantisation: a rate relation is the one thing that produces a long monotone run of
    exactly-one-quantum steps. So that is what gets counted.

    FOUR CONDITIONS, ALL REQUIRED, each answering a different way of being wrong:
      rungs      enough one-quantum steps that a handful of coincidences cannot supply them
      purity     those rungs DOMINATE the steps -- a file with three big edits and two
                 one-quantum wobbles is not a ladder
      direction  the rungs almost all point the same way -- drift has a sign, edits do not
      magnitude  the implied rate deviation is large enough for some NAMED rate to explain it

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
             primed_alignments=None):
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
                f"({cause}) -- speed correction here is a MEASUREMENT tool only (ADDENDUM 7), "
                f"and no measurement was made"), None
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

    resolved = []
    for index, hole in enumerate(holes):
        step_launch("resolve_hole", candidate=candidate_path, hole=index, kind=hole["kind"],
                    why=hole["why_token"], master_span_s=round(hole["master_span_seconds"], 3),
                    candidate_span_s=round(hole["candidate_span_seconds"], 3),
                    step_ms=(round(hole["step_ms"], 1) if hole["step_ms"] is not None
                              else None),
                    merged_from=hole["merged_from"])
        outcome = resolve_hole(hole, master_obj, candidate_obj, work_dir)
        step_result("resolve_hole", candidate=candidate_path, hole=index,
                    status=outcome["status"], cause=outcome.get("cause"))
        if outcome["status"] == HOLE_NOT_IMPLEMENTED:
            return False, outcome["cause"], (
                f"the pair decomposed cleanly into {len(holes)} hole(s) "
                f"({[h['kind'] for h in holes]}) and the couples agree, but frame-exact hole "
                f"resolution is design stage 4 and is not built yet -- no boundary was measured "
                f"and none is claimed"), None
        # ADDENDUM 3: a closed hole restores zone continuity and is NOT a failure. Unreachable
        # from the stub above, and implemented anyway because the orchestration of it is this
        # module's job, not the resolver's, and building it with the resolver would mean
        # building it under time pressure with a real pair waiting.
        if outcome["status"] != HOLE_NO_CUT_CONFIRMED:
            resolved.append((hole, outcome))
    if holes and not resolved:
        step_result("holes", candidate=candidate_path, couple=driving["couple"],
                    verdict="all_holes_no_cut_confirmed",
                    rule="ADDENDUM_3_video_disposes_pair_merges_without_splice")

    ok, cause, reason = apply_plan(candidate_path, holes, factor, master_obj, candidate_obj)
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
        if winner is None and observations.get("gate_arm") == "rate_relation_signature":
            # THE RATE ARM'S REFUSAL IS NOT TERMINAL, AND THAT ASYMMETRY IS DELIBERATE -- see
            # `similarity_gate`. This arm fired on an alignment that SUCCEEDED; the sweep was
            # asked because the zones looked like a rate ladder, and it has now answered no.
            # Declining here would convert a suggestion into a refusal and manufacture a new
            # false-decline family every time the ladder reading misfired on a healthy pair.
            # The honest continuation is the alignment we already have, at factor 1, with the
            # sweep's own cause recorded so nobody has to wonder why a sweep ran.
            step_result("speed_sweep", candidate=candidate_path,
                        arm="rate_relation_signature", terminal=False,
                        cause=sweep_cause,
                        continuing="at_factor_1_with_the_blind_alignment",
                        rule="a_suggestion_that_was_refused_is_not_a_refusal_of_the_pair")
            tools.dev_log(
                f"orchestrator: the rate-ladder arm asked for a sweep on {candidate_path} and "
                f"the sweep declined ({sweep_cause} / {(sweep_gate or {}).get('cause')}); the "
                f"pair CONTINUES at speed_factor 1 on the alignment already measured -- this "
                f"arm suggests, it does not refuse\n")
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
                                         candidate_obj, work_dir, primed_alignments=primed)
    step_result("chimeric", candidate=candidate_path, ok=ok, cause=cause)
    if ok:
        _plan_line("chimeric", candidate_path, step="chimeric",
                   speed_factor=(f"{factor.numerator}/{factor.denominator}"
                                  if isinstance(factor, Fraction) else factor))
        tools.log_always(f"repair: repaired for {candidate_path}: the orchestrator built a "
                         f"plan and produced the temporary chimeric file\n")
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
                coverage=alignment.get("master_axis_coverage_fraction"),
                residual_fraction=alignment.get("residual_fraction"),
                seconds=round(alignment["alignment_seconds"], 2))
    return True, None, None, None
