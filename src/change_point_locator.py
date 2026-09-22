# -*- coding: utf-8 -*-
"""
change_point_locator.py — where two timelines diverge, and by how much.

Measurement half of campaign 2 objective 2. Written by `vmsam-dev-1`; the only
caller is `vmsam-dev-2`'s repair module. Interface agreed before either side was
built: `VMSAM_HELP_AI/dev-2/INTERFACE_dev1_dev2.md`.

**This module reads no configuration and is conditioned on nothing but what it
measures.** Every refusal below is a measurement result — no shared language, no
duration, too few probes carrying signal, fidelity at the floor with scattered
offsets, monotone drift (a speed relation, objective 3's problem), or every
segment unusable after clamping. The owner's ruling that a repair conditioned on
a parameter is not a repair applies here by construction rather than by removal:
there was never a flag to take out.

It returns NUMBERS, never files. It never cuts, never writes a track, and never
touches `best_video.sameAudioMD5UseForCalculation`. The repair is dev-2's.

--------------------------------------------------------------------------------
THE CONTROL-FLOW RELATIONSHIP, MEASURED RATHER THAN DESCRIBED

*** IF THIS SECTION AND `PIPELINE.MD` EVER DISAGREE, MEASURE THE CODE. ***

ONE ENTRY, ONE CALL SITE. `locate_change_points` is the only name here without a
leading underscore, and the only call from anywhere in `src/` is in
`merge_video_repair.py`, anchored on this exact line rather than a line number
because A LINE NUMBER IS A PIN AND THIS FILE MOVES:

    plan = change_point_locator.locate_change_points(best_video, candidate_obj,

reached through a guarded `import change_point_locator`. Counted by AST over
`src/*.py`, not by recollection. A test that drives any other function is testing
something the consumer never calls.

**I LANDED THIS PARAGRAPH CITING `:294` AND IT WAS ALREADY STALE** -- measured
before a 52-commit fast-forward, correct at the old tip, a comment at the new one.
Corrected to an anchor in the next commit. The rule I broke is my own.

THE CONSUMER'S HALF OF `CAMPAIGN.MD`'s REFUSAL CONTRACT IS ALREADY LANDED AND
MINE IS NOT. That wrapper returns `(plan, cause)` -- `return plan, None` on
success, `return None, "locator_module_absent"` when the import fails -- and it
is waiting on a producer token this module does not yet emit. NOTE THE
CONSEQUENCE BEFORE CHANGING THIS FUNCTION'S RETURN: the call site assigns a
single value and tests `if plan != None`, so returning a tuple from here makes
that test ALWAYS TRUE AND SILENTLY BREAKS THE CONSUMER. The two halves land
together or not at all.

TWO LOG CHANNELS, AND THEY DIFFER IN THE ONLY WAY THAT MATTERS IN PRODUCTION:

    `_log`   gated on `tools.dev`  -> SILENT when the corpus runs
    `_emit`  UNGATED               -> reaches `tools.logs` always

BOTH write the same `[change_point_locator]` tag, and `merge_plan_report.py` keys
`_LOCATOR_TAG` to exactly that string. So the consumer's reader is pointed at a
channel that, in production, carries ONLY `_emit` lines. Eight of the ten refusal
paths in `locate_change_points` use `_log` alone and therefore say nothing at all
where it counts; the consumer records "no measurement available for this pair",
which is false -- the probes ran and returned a conclusive negative.

COVERAGE OF THIS MODULE'S OWN TESTS, MEASURED BY TRACER, NOT ESTIMATED:

    refusal sites (`return None`) reachable from the entry     14
    silent refusal paths ever driven by any harness             2 of 8

Aim is not coverage. A harness that calls `locate_change_points` exercises the
call tree by construction and still says almost nothing about which refusal fired.

ONE INVARIANT THIS MODULE RELIES ON AND DOES NOT STATE ANYWHERE ELSE:

    a pairing entry with `master_stream=None` NEVER carries a `fidelity`

That is true, and it is NOT enforced where it is relied upon. It holds because
`_master_streams` drops any entry whose `StreamOrder` is None before a partner
list is ever built -- roughly eighty lines above the two literals that depend on
it. Measured both ways: separating the two literals on a copy produces the
violation immediately, while feeding the shipped function a None stream id
cannot. AN EDIT TO THAT FILTER BREAKS THIS SILENTLY AND NOTHING HERE WOULD CATCH IT.

--------------------------------------------------------------------------------
THE SIGN CONVENTION, AS AN EQUATION SO IT CANNOT BE READ TWO WAYS

    candidate_time_ms = master_time_ms + candidate_offset_ms

To fill master position `m`, read the candidate at `m + candidate_offset_ms`.
A candidate missing content the master has gives a NEGATIVE offset.

Derived from `audioCorrelation`'s algebra, then measured on a constructed pair
with a known 1000 ms deletion, which read -1000. dev-2 confirmed it independently
from the opposite direction after its own control caught a sign error.

--------------------------------------------------------------------------------
WHY THIS RE-MEASURES INSTEAD OF READING `delayFirstMethodAbort`

`SPEC_ZONE_A.MD` §1 offers the recorded delays so the correlation need not be
re-run. Measured, those are RESIDUALS against a candidate that
`recreate_files_for_delay_adjuster` has already shifted by `delayUse/1000` s — an
arbitrary value, essentially never a whole multiple of the chromaprint hop
(4096/3/11025 = 123.840 ms). A fractional-hop shift misaligns the fingerprint
grids and MANUFACTURES one-point steps: three corpus files whose logs record one
measure as constant offsets when read unshifted, including one independently
verified frame-identical by eye with a duration delta of exactly 0.000 s. So every
probe here extracts both sides at the SAME absolute time and the grids stay
aligned.

--------------------------------------------------------------------------------
WHY THE COARSE SCAN IS THE FFT, AND WHY IT COVERS THE WHOLE FILE

Two defects in the first version of this module, both found by other agents
running it on real files rather than by review, drove this design:

1. IT SCANNED ONLY THE PIPELINE'S TEN WINDOWS, which start at `begin_in_second`
   (120 s) and are `2 * spacing` long. A step is visible to that geometry only if
   some window is majority-pre while the next is majority-post — and window 0 is
   the first, so nothing can be majority-pre before it. A step is therefore
   invisible unless `T > begin + L/2`: the first ~227 s of a 24-minute episode.
   Error ids 12, 13 and 108 all carry real steps there and the module returned
   "constant" on all three.

2. CHROMAPRINT'S QUANTUM HIDES SMALL STEPS. `int(lengthFile/n_items*1000)` is
   124-142 ms depending on window length. Error id 13 has a real 41.7 ms step —
   ONE VIDEO FRAME at 24.000 fps, bracketed at (1285, 1290) s — a third of a
   point, which no window reading can express. Position was never the only
   condition: a step must ALSO exceed roughly half a quantum.
   CORRECTED 2026-09-04: this paragraph previously read "a real 89 ms step at
   ~1170 s". THAT CLAIM WAS FABRICATED and was retracted by its own author the
   same day — it came from a refuted drift hypothesis about a different subject
   and was never measured on id 13. id 13's four steps are -1500.0, -833.4,
   -83.3 and -41.7 ms, all whole frames at the file's own 24.000 rate, measured
   by a dense pass and confirmed independently by `vmsam-ci`.

Hence a probe grid over the WHOLE file with NO privileged region, read with the
unquantised FFT (`audioCorrelation.second_correlation`). The head was never
special; it was only the part noticed first.

Chromaprint is kept for the one thing it is better at here: a fidelity floor. A
pair with no shared content reads a flat 0.556-0.578 with offsets scattered over
421 s and the sign flipping 8 times in 9 transitions.

--------------------------------------------------------------------------------
UNITS

The FFT is unquantised, so `candidate_offset_ms` is a real measurement rather than
a multiple of anything. `candidate_offset_points` and `quantum_ms` are RETURNED IN
THE RESULT DICT, and from 2026-09-04 also written to the log on the success path.
The channel matters: this section said "emitted", vmsam-ci read that as "logged",
went to run the cross-check against preserved logs and found zero occurrences in
every artefact it holds. It was reading the right word in the wrong channel because
the word did not name one.

`abs(points * quantum_ms - candidate_offset_ms) <= quantum_ms/2` IS NOT A CROSS-CHECK.
It is the definition of `round`, and `candidate_offset_points` is computed by `round`
at the emission site below. A digest cannot disagree with what it digests, so the
inequality can never fail and tests nothing. It is stated here so nobody spends time
verifying an identity. THE REAL CROSS-CHECK IS AGAINST THE PIPELINE'S OWN QUANTUM,
which is a different number from a different window: ours comes from fixed 60 s
probes and is constant by construction, the pipeline's from `int(lengthFile/n*1000)`
over the whole file and varies. vmsam-ci measured 129 here against 124-125 there on
four files. Both are correct and they are not each other's check.

**Never report a step as a bare millisecond figure taken at some other window
length.** One physical step measured 500, 540 and 600 ms at three window lengths,
because each carries its own quantum. Established three separate times on three
files before it was believed.
"""

from os import path, stat as os_stat, remove
from statistics import median

import hashlib
import tools
import json
import subprocess
import audioCorrelation
import pal_saturation_screen
import zone_similarity_vector

# One chromaprint fingerprint item, seconds. frame 4096, hop frame/3, rate 11025.
CHROMAPRINT_HOP_SECONDS = 4096.0 / 3.0 / 11025.0

# --- coarse scan -------------------------------------------------------------
# 60 s probes correlate at 0.86-0.97 on real different-source pairs. The step
# between probe starts is shorter than the probe itself, so no transition can
# fall between two probes unobserved.
# This comment previously claimed these probes "resolve an 89 ms step", citing a
# figure that was FABRICATED AND RETRACTED. What they resolve has not been
# measured in this configuration; see MIN_STEP_MS below.
PROBE_WINDOW_SECONDS = 60.0
PROBE_STEP_SECONDS = 40.0

# --- the two step gates -----------------------------------------------------
# THESE ARE TWO QUESTIONS, NOT ONE, AND THE ORDERING IS A STRUCTURAL CLAIM.
#
#   PLATEAU_TOLERANCE_MS  a SINGLE PROBE against the run's RUNNING MEAN.
#                         Guards against one noisy probe splitting a plateau.
#   MIN_STEP_MS           two SETTLED MEANS against each other.
#                         Guards against two real plateaus being called one.
#
# A single probe deviates more than a mean of several does, so the tolerance must
# be the LARGER of the two. MEASURED on four flat spans over three files, 60 s
# probes, no rate correction -- the configuration these constants actually gate:
#
#     max single-probe deviation from the running mean   0.217 ms
#     max standard error of a settled plateau mean       0.055 ms     ratio 3.9x
#
# THE PREVIOUS VALUES WERE 50.0 AND 60.0 -- INVERTED. With tolerance < step floor
# a step between the two starts a new run and is then merged away: the tolerance
# makes a decision the step floor overrides, and the work is done and discarded.
# Nothing in the code said which way round they belonged, which is why it could
# invert unnoticed.
#
# THE WINDOW BOTH MUST LIE IN:
#     lower   0.4 ms   measured flat-region spread, this configuration
#     upper  41.7 ms   the smallest CONFIRMED real step -- ONE VIDEO FRAME at
#                      24.000 fps, on id 13, bracketed at (1285, 1290) s
# The old values were outside it, above, and either alone DELETED that step.
#
# 5.0 ms is ~12x the noise and ~8x below the smallest confirmed step. Not nearer
# either edge: a false step gives a piecewise plan where a constant would do, a
# missed step gives a constant plan on a file that needs splicing, and both ship
# a wrong file nobody checks.
#
# AND 41.7 ms IS A FLOOR ON WHAT HAS BEEN SEEN, NOT ON WHAT EXISTS. Setting a
# constant just under it would make the smallest observed step the smallest
# OBSERVABLE one, permanently -- which is how 60.0 got here, from a figure
# ("89 ms on id 13 at ~1170 s") that was FABRICATED and retracted by its author.
#
# THE TWO ARE EQUAL TODAY BECAUSE THE ORDERING DOES NOT BITE AT THIS MAGNITUDE --
# both sit far above 0.217 and 0.055 -- NOT BECAUSE THEY ARE THE SAME QUANTITY.
# Below roughly 1 ms the distinction starts to matter and tolerance >= step floor
# must be restored explicitly.
PLATEAU_TOLERANCE_MS = 5.0
MIN_STEP_MS = 5.0

# THE ORDERING IS ENFORCED, NOT DOCUMENTED. The comment above explains WHY; this
# makes the inversion impossible rather than merely described. The previous values
# were inverted at 50.0/60.0 and NOTHING IN THE CODE SAID WHICH WAY ROUND THEY
# BELONGED -- which is why it could invert unnoticed, and a requirement living in
# prose beside the artefact rather than inside it is the shape that failed twice
# tonight elsewhere in this campaign.
assert PLATEAU_TOLERANCE_MS >= MIN_STEP_MS, (
    "tolerance compares ONE PROBE to a running mean, the step floor compares TWO "
    "SETTLED MEANS; a single probe deviates more (measured 0.217 vs 0.055 ms), so "
    "the tolerance must be the larger. They were inverted at 50.0/60.0 until "
    "2026-09-04, and a step between the two started a run that was then merged away.")

# --- refusal thresholds ------------------------------------------------------
MIN_MEDIAN_FIDELITY = 0.70
MAX_DISTINCT_POINTS = 4
MAX_SIGN_FLIPS = 2

# --- cross-language stream pairing ---------------------------------------
# A candidate audio stream outside the measured language used to get NO entry in
# the per-stream offset table, and dev-2's assembler fell back to the measured
# language's offset for it -- silently, carrying 14-32 ms on the two files it
# measured, and 34.62 ms on a produced file. Under its 100 ms tolerance, under a
# video frame, invisible.
#
# The fix is a per-language reference: probe each candidate stream against a
# master stream OF ITS OWN LANGUAGE. Measured on 26 corpus files / 127 stream
# pairs: same-language same-track pairs score 0.944-0.990, cross-language pairs
# 0.566-0.848, and 0 of 87 cross-language pairs reach 0.85 on the MINIMUM of two
# probe positions. At a SINGLE position cross-language reaches 0.9400 -- so the
# two-position minimum is load-bearing and a single-probe bar would accept them.
#
# 0.85 IS A CHOICE INSIDE AN OVERLAP, NOT A BOUNDARY: the highest cross-language
# min-of-two is 0.8477 and the lowest genuine-looking same-label is 0.8196. Every
# measurement is therefore reported beside its verdict, so the bar can be moved
# by someone who disagrees with it. A row saying "rejected" cannot be re-judged;
# a row saying "0.8477, rejected at 0.85" can.
MIN_PAIRING_FIDELITY = 0.85
# REVERTED to (0.35, 0.65) -- the >=4-position widening
# (RULING_20260921_STEP1_CLASSIFIER_DESIGN.MD SM3) is NOT the tolerance
# change it looked like. `score = min(scores)` over MORE samples can only
# fall or stay equal, never rise (Lead's catch, 2026-09-21): every
# candidate stream's pairing score can only get WORSE under more positions,
# so the bar becomes strictly HARDER to pass, and the new outer positions
# (0.2, 0.8) sit closer to file edges -- head trims, tail reels, credits --
# where a probe can legitimately land on content that does not match for
# reasons unrelated to a pairing failure. "Adds tolerance" and "adds
# strictness" are not the same change wearing different numbers. Reverted
# pending a real regression count on the 73-file population (which files
# flip from PASS to FAIL between the two/four-position geometries, by id
# and margin) -- that measurement is a real-media re-run at the same scale
# as the original acceptance run, not done in this session; the fix is
# independent of the rest of this design (the gated instrument block) and
# lands separately once the count exists.
PAIRING_POSITION_FRACTIONS = (0.35, 0.65)

# --- no-signal guard ---------------------------------------------------------
# A correlation taken where there is no signal is not a measurement. dev-2 found
# a 1.4 s near-silent window returning -170.69 ms with apparent confidence; with
# signal in the window the same region reads -0.19 ms at r=0.974. Probes far
# below the file's own median energy are dropped rather than trusted.
LOW_SIGNAL_FRACTION = 0.10

# --- refinement ---------------------------------------------------------------
# A transition is bracketed by CLEAN probes only, never by bisecting straddling
# ones. `vmsam-forensic`'s standing note: a peak-picking correlator on a window
# that spans a feature boundary returns a DISPLACED peak, not a blend, and the
# displacement is arbitrary in sign and unbounded by the grid. Measured cost of
# ignoring this: a first version bisected 60 s probes under a majority model and
# placed id 108's two transitions 19 s early and 24 s late while reporting a
# 625 ms bracket — an over-claim of about 35x, against two independent
# instruments that agreed with each other.
REFINE_WINDOW_SECONDS = 8.0
REFINE_STEP_SECONDS = 4.0


def _digest(*parts):
    """Opaque, stable, one-way identifier for a tuple of strings.

    USED FOR PATHS AND EMITTED IN THEIR PLACE. The paths themselves may never reach a
    log line this fleet quotes into a report (`WRITE_ZONES` section 8: where the input is
    free text you do not own, WITHHOLD it rather than sanitise it -- a pattern redactor
    was measured failing on a path containing spaces). A digest carries the FACT that two
    specific files were paired, which is all a consumer needs to join rows, and carries
    none of the content.
    """
    # *** A MISSING INPUT RETURNS A WORD, NEVER A HASH. `vmsam-dev-4` found the defect in
    # the coercion this replaces: `_digest(None, None)` and `_digest("", "")` produced the
    # SAME 12-hex value, so every unresolvable pair carried an IDENTICAL, REAL-LOOKING join
    # key. Not absent, not unique -- A SENTINEL WEARING THE SHAPE OF A REAL KEY. ***
    #
    # AND IT FAILS THIS MODULE'S OWN RULE HARDER THAN THE ASYMMETRY IT WAS PART OF FIXING:
    # a consumer does not see those rows as unjoinable, it sees N rows OF THE SAME PAIR, and
    # joins them silently. A census would report one pair with N refusals.
    #
    # THE DILEMMA dev-4 POSED, SETTLED BY READING `video.py`: `self.filePath` is assigned
    # UNCONDITIONALLY and BEFORE the constructor's only `raise`, so any object that survives
    # construction carries it, and for the engine chain this branch is UNREACHABLE -- m = 0.
    # **THAT IS EXACTLY WHY IT IS NOT LEFT AS A SENTINEL.** An untested cell is not a clean
    # one, and the cost of being wrong here is a silent false join in somebody else's census.
    for part in parts:
        if part is None or str(part) == "":
            # NOT `unidentified`. `vmsam-dev-4` measured that it is **12 CHARACTERS
            # LONG -- EXACTLY THE WIDTH OF A REAL DIGEST** -- so a `len(...) == 12` or a
            # truthiness check joins on it happily. My claim that it "cannot be joined on"
            # was a claim about MY OWN charset predicate, never about a consumer's:
            # section 7 one level down, the producer asserting a property of a value while
            # THE CONSUMER IS THE ONE MAKING THE CLAIM ABOUT WHAT THE BYTES MEAN.
            #
            # *** AND THE HONEST LIMIT: NO VALUE DEFEATS A TRUTHINESS TEST. A producer can
            # only make the sentinel HARDER to mistake -- wrong width, non-hex,
            # self-describing -- and then TELL THE CONSUMER. The width was free; the
            # telling is the part that actually protects `forensic`'s pair-keyed map. ***
            return "pair_unidentified"
    joined = "\x00".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8", "surrogatepass")).hexdigest()[:12]


def _build_digest():
    """Digest of THIS module's source, read AT CALL TIME.

    Deliberately not cached at import: the rule it satisfies says emit build identity at
    EMISSION time, and a value captured once cannot report that the code changed under a
    long-lived process. It costs one small file read per pair, which is nothing beside the
    ffmpeg probes this function has already run by the time it is called.

    Returns `unknown` rather than raising: a missing build id must never be the reason a
    merge fails, and `unknown` is an honest value where a stale constant would not be.
    """
    try:
        with open(__file__, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()[:12]
    except OSError:
        return "unknown"


def _log(message):
    # ROUTED THROUGH `tools.dev_log` (owner's order via the Lead, 2026-09-22):
    # this used to write ONLY to `tools.logs`, never stderr -- and
    # `tools.logs` is drained only at the END of a merge (`mergeVideo.py:872`,
    # `fusion.py:424`). A hung process never reaches either drain point, so
    # this module's ~30 per-probe lines were invisible in EXACTLY the
    # incident they exist to describe. The format string is unchanged (same
    # tab/bracket prefix every existing caller and reader already expects);
    # only the sink gained a stderr half that survives a hang.
    tools.dev_log(f"\t\t[change_point_locator] {message}\n")


def _emit(message):
    """Unconditional. Once per pair: the success line, or the refusal that replaces it.

    *** "AND NOTHING ELSE" WAS TRUE WHEN WRITTEN AND STOPPED BEING TRUE WHEN `_decline`
    STARTED CALLING THIS. The docstring kept saying it. I read this file all day and read
    past it, and only noticed when I checked WHICH SINK A REFUSAL GOES THROUGH rather than
    assuming -- the prefix is shared between the two sinks, so the output does not tell you
    which one produced it. A COMMENT THAT WAS ACCURATE WHEN WRITTEN IS NOT A COMMENT THAT IS
    ACCURATE. The per-pair limit is the real invariant and it still holds: success OR refusal,
    never both, never twice. ***

    NOT A POLICY THAT `_log` SHOULD FOLLOW. The limit is per-pair, and the reason
    lives in the SINK rather than in this module:

    `tools.logs` is consumed unconditionally -- `gestionar_show/fusion.py:412` into
    the error file, and `mergeVideo.py:845` slices `tools.logs[emitted_before_repair:]`
    into `merge_plan`, which is IN THE ARTEFACT. `mergeVideo.py:835` states that the
    slice is taken BY POSITION AND NOT BY PATTERN, deliberately, so that lines outside
    the repair vocabulary are COUNTED rather than dropped. `merge_plan_report.py:489`
    counts them into `foreign_lines`.

    So a per-probe line is not merely verbose: at 30 probes x 315 files it is 9450
    entries in a counter built to detect ONE anomaly. THE GATE PROTECTS THAT COUNTER,
    NOT THE LOG SIZE. Anything per-probe stays on `_log`.

    This line is safe to print in full because every field is a number from a
    vocabulary this module owns -- no path, no filename, nothing borrowed
    (`WRITE_ZONES.MD` section 8). vmsam-ci ran dev-4's own parser over it and the
    result carries `carries_path: False`, which is dev-4's measurement of this line
    rather than an argument about it.

    Requires the matching allowlist entry in `merge_plan_report.py` (vmsam-dev-4's
    module, NOT vmsam-dev-2's -- that file's dev-2 commit is a sync of dev-4's bytes,
    and a blame count measures who ran a command, not who owns a file). Without it
    this line is counted foreign on every successful pair. Ruled by the Lead on
    2026-09-05: both halves land in one batch, allowlist first or alongside, never
    after.

    Ungated because the value could not be read otherwise: `tools.dev` is False in
    production, so vmsam-forensic found ZERO `[change_point_locator]` lines across 43
    records and vmsam-ci zero across the corpus. The cross-check this module's UNITS
    section promises was unrunnable for three distinct reasons in sequence -- computed
    and not emitted, then emitted into a gated sink, then the gate closed in the only
    configuration that runs the corpus -- and each was invisible until the previous
    was repaired.

    STAYS UNGATED, GAINED A STDERR HALF (owner's order via the Lead,
    2026-09-22, wave 3b): routed through `tools.log_always` rather than
    `tools.dev_log` specifically so this function's own ungated-ness is
    preserved, not converted into a gate that was never here. NOTE FOR THE
    NEXT READER, NOT ACTED ON HERE: the "`tools.dev` is False in production"
    premise two paragraphs up is the Lead's own measured-STALE claim as of
    tonight (dev is ON in both containers) -- left as written, flagged
    rather than silently corrected, because deciding whether/how to amend it
    is the Lead's call, not this edit's.
    """
    tools.log_always(f"\t\t[change_point_locator] {message}\n")


DECLINE_REASONS = (
    "audio_duration_unavailable",
    # Architect's grant, 2026-09-16, TASKS/013 §3: the scan RAN, the probe-count
    # floors (too_few_usable_probes / too_few_probes_with_signal) already
    # PASSED, and the ACHIEVED coverage of the scanned region -- read from the
    # probes that SURVIVED, never the attempted grid -- has a hole inside it.
    # Distinct from every count guard above and below: those ask "how many
    # probes came back", this asks "does what came back cover the span". Always
    # paired with measurement=could_not_run, never ran_conclusive_negative --
    # a hole means the scan cannot conclude, not that it concluded negatively.
    # Detail carries the hole bounds (coverage_gap_bounds) and total
    # (coverage_gap_total_s), never only the count: TASKS/013's own lesson on
    # `primary_below_pairing_bar` -- "a row saying rejected cannot be
    # re-judged; a row saying '0.8477, rejected at 0.85' can."
    "coverage_incomplete",
    "every_probe_failed",
    "median_fidelity_below_floor",
    # Architect's grant, 2026-09-16, E3 (DESIGN_PAL_SPEED_FAMILY_20260916.MD),
    # replacing the line's old `else 125` substitution. Fires when the probe
    # count and energy floors have ALREADY PASSED (`too_few_usable_probes` /
    # `too_few_probes_with_signal` above did not fire) and EVERY KEPT probe's
    # quantised `audioCorrelation.correlate()` call returned `points == 0` --
    # `_probe`'s own guard (`quantum = ... if points else None`) means a
    # zero-offset probe carries no per-probe quantum at all, not a quantum of
    # zero. Distinct from every count guard above and below: those ask "how
    # many probes came back" or "how much energy did they carry"; this asks
    # "of the probes that came back with signal, did even one of them let the
    # quantised cross-check name a step size". Always paired with
    # measurement=could_not_run, never ran_conclusive_negative -- an all-zero
    # quantised read is could-not-measure, not measured-nothing, and the two
    # are not interchangeable (this campaign's whole argument against
    # collapsing them). Substituting a number here was worse than an ordinary
    # absence-as-value: the old fallback (125) is the PIPELINE's own quantum,
    # derived over its own file-length-dependent merge window, never this
    # module's -- this module's probes run a fixed 60 s window
    # (PROBE_WINDOW_SECONDS) and, in every real emission examined, actually
    # measure 129. Detail carries probes_kept, so a reader can tell this apart
    # from a run that never reached the energy floor at all.
    "no_quantised_points",
    "no_stream_for_language",
    "no_usable_segments",
    # E3 (Architect's grant, exercised -- DESIGN_PAL_SPEED_FAMILY_20260916.MD
    # Stage 1, "offsets_saturated_at_search_bound is PRE-AUTHORIZED, you do
    # not need to ask for it"). Fires when `pal_saturation_screen` finds
    # every kept probe's quantised offset sitting at the correlator's own
    # search bound (|points| >= 0.99*(N-32), N derived from the window
    # actually passed) -- the correlator ran out of window, not out of
    # signal. Distinct from every fidelity-based guard here: this fires
    # BEFORE fidelity is even computed on the survivors, because a saturated
    # probe's fidelity number is not evidence about the true offset at all.
    # Always could_not_run, never ran_conclusive_negative -- a search-bound
    # readout is could-not-measure, the same reasoning as
    # `no_quantised_points` one level up. THE TRIGGER BAR ITSELF (zero
    # survivors, not a softer majority rule) is an open design question,
    # not settled here -- see `pal_saturation_screen.screen_decline_detail`'s
    # own docstring and the Architect's ruling on it, 2026-09-16: the bar
    # becomes a measurement once a census exists over the 20 real PAL ids
    # plus a content-mismatch control population. Every screen pass, decline
    # or not, logs its observed saturation fraction so that census accrues
    # from live runs.
    "offsets_saturated_at_search_bound",
    "offsets_scattered",
    "primary_below_pairing_bar",
    # Architect ruling 2026-09-21, search_bound_unevaluable enumeration.
    # `pal_saturation_screen.probe_search_bound` can be non-positive at a
    # short enough window -- unreachable at production's own 60.0 s window
    # (crossover ~6.6 s) -- and `screen_decline_detail` refuses to answer
    # rather than guess "clean" or "saturated" when that happens
    # (`SearchBoundUnevaluable`, `pal_saturation_screen.py`). Enumerated so
    # the vocabulary stays complete: "enumerate when observed" would mean
    # never, by construction, since the branch is unreachable at today's
    # window, and an enumeration permanently missing a token the code can
    # provably emit is a standing lie in the exact table a census reads. A
    # vocabulary registry's completeness claim is over what the code CAN
    # say, not what it HAS said. Always could_not_run, never
    # ran_conclusive_negative -- nothing was measured at all, the same
    # reasoning as every other could_not_run entry above.
    "search_bound_unevaluable",
    "speed_relation_suspected",
    "stream_unmeasurable_at_centre",
    "too_few_probes_with_signal",
    "too_few_usable_probes",
)

DECLINE_MEASUREMENTS = ("ran_conclusive_negative", "could_not_run")


def _decline(reason, measurement, **fields):
    """The ONE emission every boundary refusal of `locate_change_points` goes through.

    CAMPAIGN.MD's refusal contract specifies a `(plan, cause)` RETURN; `vmsam-dev-4`
    added the clause that decides whether the work counts:

        "THE TOKEN MUST APPEAR IN AN EMITTED LOG LINE. A cause that reaches only an
         in-memory structure SATISFIES NOTHING."

    That clause is why this is a helper and not a return value. `_log` is gated on
    `tools.dev`, which is False in production -- so a cause carried only by the ten
    `_log` sentences below emits NOTHING in the one configuration that runs the
    library, and the consumer records "no measurement available" for every refusal.
    That is not hypothetical: this module's own `_emit` docstring records the same
    defect happening three times in sequence, each invisible until the previous was
    repaired.

    ONE SHAPE, AND THE INVARIANT IS THE PREFIX:

        declined: reason=<snake_case_token> measurement=<token> [name=value ...]

    *** `declined: reason=` NOW CATCHES 10 OF 10. IT USED TO CATCH 1 OF 3. ***
    The three emissions that existed before this change put `stream=` and `lang=`
    ahead of `reason=`, so the natural grep -- the one that names a token -- saw a
    third of the declines and a reader had no way to know. The fields did not move
    out of the line, they moved along it; every reader that resolves BY NAME
    (`WRITE_ZONES.MD` section 7, rule 1) is unaffected, and a positional reader was
    already broken by the two different field orders.

    `measurement` carries the distinction the consumer demonstrably needs and could
    not make -- `merge_video_repair.py` states it at its own call site: a conclusive
    negative filed as an absence of evidence is the substitution this module warns
    against in its own first docstring.

        ran_conclusive_negative  the probes RAN, they SUCCEEDED, and the answer is no
        could_not_run            no usable measurement was obtained at all

    PRIVACY, BY CONSTRUCTION AND NOT BY REDACTION (`WRITE_ZONES.MD` section 8): every
    value this function can emit is a number this module computed, a stream index, a
    language code, or a literal token from the vocabulary above. No path, no
    filename, no exception text, and no caller-supplied free text -- the reason is a
    literal at every call site, never an argument that reached the module from
    outside. Nothing here is sanitised, because nothing here is borrowed.

    *** A CAUSE IS NOT A ROUTING INSTRUCTION, AND THE TWO AXES ARE ORTHOGONAL. ***
    The architect's ruling, 2026-09-07, correcting the contract's own citation:
    `PIPELINE.MD` section 3 separates what the CONSUMER should do -- "this file cannot be
    helped" (a dead end) against "this file goes back to the cheap path" (a routing
    decision). `measurement` above is the PRODUCER's epistemic status, which is a
    different question. **A consumer that reads `could_not_run` and infers "dead end" has
    made exactly the confusion section 3 warns about, using a token introduced here.** The
    consumer needs both axes and MUST NOT DERIVE ONE FROM THE OTHER: `could_not_run` says
    nothing about whether the file is helpable, and `ran_conclusive_negative` does not
    make it hopeless. Nothing in this module emits a routing decision, deliberately --
    `CAMPAIGN.MD` is explicit that a token "names WHAT THE MEASUREMENT FOUND, never what
    the consumer should do".

    THE ENUMERATION ABOVE IS A GATE, NOT A LIST -- architect's ruling on the contract's
    reserved line, the whole reason it lives here rather than in a document:

        A VOCABULARY IN ITS ONLY PRODUCER CANNOT DRIFT FROM IT.
        A NEW TOKEN IS NOT EMITTABLE WITHOUT BEING ENUMERATED IN THE SAME EDIT.

    *** AND IT MARKS RATHER THAN RAISES, WHICH IS THE ONLY SAFE FAILURE HERE. *** VMSAM
    runs with nobody watching. A validator that raised would turn a mistyped token into a
    crashed merge, and one that refused to emit would DESTROY THE REFUSAL LINE -- which is
    the exact defect this whole helper exists to close. So an unenumerated value still
    emits, and carries `reason_enumerated=false` beside it: the refusal always reaches the
    sink, and the vocabulary breach is greppable in the same line. **A gate that can only
    fail by making the output NOISIER is one that cannot make the engine worse.**

    Returns `(None, reason)` so that a call site reads `return _decline(...)`: the
    emission and the refusal cannot drift apart if they are one statement, AND the
    token reaches the caller as a value rather than only as a log line.

    *** IT USED TO RETURN None, AND THAT WAS MY ERROR, NOT AN OPEN QUESTION.
    CAMPAIGN.MD lines 1153-1164 already specified `(plan, cause)` with a stable token on
    EVERY boundary return. I read dev-4's clause -- "the token must appear in an EMITTED
    LOG LINE; a cause that reaches only an in-memory structure satisfies nothing" -- AS
    EXCLUSIVE OF THE RETURN, AND BUILT THE EMISSION HALF ONLY. THEY WERE NEVER EXCLUSIVE.
    IT CAN EMIT AND RETURN, AND THE CONTRACT SAID SO IN WRITING THE WHOLE TIME.
    The consumer had no cause to state, so a real run resolved to `cause_unavailable`. ***
    """
    parts = [f"declined: reason={reason}", f"measurement={measurement}"]
    if reason not in DECLINE_REASONS:
        parts.append("reason_enumerated=false")
    if measurement not in DECLINE_MEASUREMENTS:
        parts.append("measurement_enumerated=false")
    for key in sorted(fields):
        parts.append(f"{key}={fields[key]}")
    # BUILD IDENTITY ON THE REFUSAL LINE TOO, AND IT IS NOT SYMMETRY FOR ITS OWN SAKE.
    # `vmsam-forensic` reads refusals to census which branch fired; without a build id
    # "this branch did not fire" and "this artefact predates the branch existing" are
    # THE SAME ABSENT ROW, and the outcome that disappears is always THE INSTRUMENT DID
    # NOT RUN. The success line carries it; a refusal that did not would leave exactly
    # the population a consumer most needs to date undateable.
    #
    # NO `pair=` HERE, DELIBERATELY: the two paths are not in scope at the earliest
    # refusals -- they are read after the language and duration checks -- and a field
    # present on some refusals and absent on others is worse than one absent from all,
    # because a consumer cannot tell a missing pair from an early refusal. STATED
    # RATHER THAN LEFT AS AN INCONSISTENCY SOMEBODY LATER "FIXES".
    parts.append(f"build={_build_digest()}")
    _emit(" ".join(parts))
    return (None, reason)


def _start_times_ms(source_path):
    """Container `start_time` per stream, in milliseconds, keyed by stream index.

    This is the quantity `ffmpeg -ss t -i file` silently absorbs: it seeks by
    presentation timestamp, so a stream whose first packet is stamped 1.103 s is
    entered 1.103 s later than a consumer decoding it from its first sample.

    Returns {} when ffprobe is unavailable or the field is absent, and the caller
    emits None rather than a guess -- a converter with a wrong start_time is worse
    than one that knows it cannot convert.
    """
    probe = tools.software.get("ffprobe")
    if not probe:
        ffmpeg = tools.software.get("ffmpeg", "")
        probe = ffmpeg[:-6] + "ffprobe" if ffmpeg.endswith("ffmpeg") else "ffprobe"
    try:
        completed = subprocess.run(
            [probe, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index,start_time", "-of", "json", source_path],
            capture_output=True, text=True, timeout=60)
        streams = json.loads(completed.stdout).get("streams", [])
    except Exception:                                  # noqa: BLE001 — absence is a valid answer
        return {}
    out = {}
    for entry in streams:
        try:
            out[int(entry["index"])] = round(float(entry["start_time"]) * 1000.0, 3)
        except (KeyError, TypeError, ValueError):
            continue
    return out


class ExtractProducedNothing(Exception):
    """ffmpeg exited 0 and produced no audio. A type I own, so the site tally can name it.

    *** NOT a generic Exception: `_probe` already catches everything and reports
    `extract_or_correlate_raised`, which would fold this into the correlator's failures.
    This is the one failure mode `vmsam-ci` traced to a root cause, and it deserves to be
    distinguishable from a correlation that ran and failed. ***
    """


def _extract(source_path, stream_order, start_seconds, length_seconds, out_path,
             sample_rate):
    """`sample_rate` IS REQUIRED AND HAS NO DEFAULT, DELIBERATELY.

    This pinned "44100" until 2026-09-05. `mergeVideo.py:583-585` derives the pair's
    LOWER rate and clamps only when it is ABOVE 44100, so above 44100 the two agreed BY
    ACCIDENT -- the clamp landed both on the same number -- and below it this module
    UPSAMPLED one side and measured on a grid the consumer never uses.

    I had `comparison_grid = min(pair's lowest rate, 44100)` written down as a property
    of the PAIR for hours and quoted it to other agents; this extractor did not
    implement the rule I was citing. Found by vmsam-ci-build as a reading of the code.

    LIVE ON TWO FILES: ids 307 and 316 carry 32 kHz candidate streams. And
    `video.py:914-919` asserts, inside the function that computes the shared rate, that
    "a sub-44100 source ... this corpus does not contain". That is false of today's
    corpus -- reported as R21, not my module, not fixed here.

    MEASURED BEFORE CHANGING, both files both ways, paired, same probe positions:
        id 316   max |difference|  0.001 ms over 8 probes
        id 307   13 of 14 probes within 0.02 ms; ONE probe at 1.288 ms
        1.288 ms against a ~129 ms quantum is 1.0% OF ONE QUANTUM
    and the outlier is not a grid effect -- the 44100 side is out of step with ITS OWN
    NEIGHBOURS (t=20/40/80 read 0.018/0.018/-0.005), so it is one unstable correlation
    at one position.

    SO THIS IS NOT A CORRECTNESS FIX AND MUST NOT BE READ AS ONE. It is landed for
    INTERPRETABILITY: two instruments on different grids cannot validate each other --
    agreement would be luck and disagreement unattributable -- which suspended ci's
    cross-check of `offset_ms` against the pipeline's delays on exactly these two files.
    The Lead's ruling: the smallness of the delta is the argument FOR landing, because
    it resolves the risk side and leaves interpretability standing alone.

    NO DEFAULT: a missed call site must be a TypeError, not a silent return to 44100.
    """
    cmd = [tools.software["ffmpeg"], "-v", "error", "-y", "-nostdin",
           "-ss", f"{start_seconds:.6f}", "-t", f"{length_seconds:.6f}",
           "-i", source_path, "-map", f"0:{stream_order}",
           "-vn", "-ac", "1", "-ar", str(sample_rate),
           "-acodec", "pcm_s16le", out_path]
    # *** THE EXIT CODE IS CHECKED AND THE OUTPUT IS NOT, AND THE FAILURE MODE IS ONE THAT
    # EXITS ZERO. `launch_cmdExt` raises on a non-zero return, so that half is covered --
    # but REPRODUCED HERE: seeking past the end of a source makes ffmpeg EXIT 0, WRITE A
    # 78-BYTE HEADER-ONLY WAV, AND SAY NOTHING ON STDERR. ffprobe then reports its duration
    # as N/A, which is `vmsam-ci`'s root cause: the per-stream files THIS FUNCTION WRITES
    # are what audio_sync chokes on, and the locator then cannot probe at all.
    # *** A COMMAND THAT SUCCEEDS IS NOT A COMMAND THAT PRODUCED SOMETHING. THE LAUNCHER CAN
    # ONLY CHECK THE CALL; ONLY THE CALLER KNOWS WHAT THE CALL WAS FOR. ***
    # IMMEDIATELY-PRE-CALL, NOT FUNCTION-ENTRY (owner's order, 2026-09-22,
    # via the Architect: `tools.launch_cmdExt` here is genuinely unbounded --
    # `Popen` + bare `communicate()`, no timeout at any layer). A line at
    # `_extract`'s own top would not name THIS call in flight if a hang
    # happens here specifically, since `_extract` runs many times per pair
    # and nothing upstream of this point can hang -- the log has to sit
    # where the block actually starts, not where the function does.
    tools.dev_log(f"locator: _extract ffmpeg call file={source_path} "
                  f"stream_order={stream_order} out_path={out_path}\n")
    tools.launch_cmdExt(cmd)
    # *** MY FIRST THRESHOLD WAS `size <= 44` ON THE ASSUMPTION OF A CANONICAL WAV HEADER, AND
    # IT DID NOT FIRE: ffmpeg WRITES A LARGER HEADER (LIST/INFO CHUNKS), SO THE HEADER-ONLY FILE
    # WAS 78 BYTES AND SAILED THROUGH. A guard whose threshold is wrong is a guard that runs and
    # reports nothing, which is the shape I have spent the night finding elsewhere. ***
    # ONE SECOND OF AUDIO AT THE REQUESTED RATE IS THE FLOOR. The caller only ever asks for whole
    # probe windows -- 60 s, or a tail start computed so the window fits -- so a file under one
    # second cannot be a legitimate short tail. THE NUMBER IS CHOSEN, NOT DERIVED: it is two
    # orders of magnitude below any window this module requests, which is why it cannot
    # false-refuse rather than because it is the true boundary.
    _floor = sample_rate * 2          # 1 s, 16-bit mono
    try:
        _written = os_stat(out_path).st_size
    except OSError:
        raise ExtractProducedNothing("extract wrote no file at the requested position")
    if _written < _floor:
        raise ExtractProducedNothing(
            f"extract wrote {_written} bytes, under {_floor} for one second at {sample_rate} Hz")


def _rms(wav_path):
    """Root mean square of the PCM payload. Deliberately stdlib-only: this runs
    inside the merge and should not pull numpy in for one number."""
    try:
        with open(wav_path, "rb") as handle:
            raw = handle.read()
    except OSError:
        return 0.0
    body = raw[44:]
    count = len(body) // 2
    if count == 0:
        return 0.0
    total = 0
    for index in range(0, count * 2, 2):
        sample = body[index] | (body[index + 1] << 8)
        if sample >= 32768:
            sample -= 65536
        total += sample * sample
    return (total / count) ** 0.5


def _note_site(sites, token):
    """Tally WHICH refusal site inside `_probe` fired. Closed vocabulary, two members.

    *** `vmsam-ci` ASKED FOR THIS AND IT IS THE ONE OF ITS THREE ASKS I HAD NOT BUILT: it has
    FOUR REFUSALS ON REAL MATERIAL AND CANNOT ATTRIBUTE ANY OF THEM. `probes_attempted` minus
    `probes_raw` says HOW MANY probes were lost; it does not say WHERE. ***

    THE VOCABULARY IS TWO TOKENS AND BOTH ARE MINE:
        correlator_named_neither_window -- the correlator returned a path that is neither of the
            two windows I extracted. A disagreement about identity, not a failure to measure.
        extract_or_correlate_raised     -- anything raised inside the try. Extraction, RMS, or
            either correlation call.

    *** THE EXCEPTION CLASS IS DELIBERATELY NOT A TOKEN HERE, AND THE REASON IS THIS MODULE'S OWN
    RULE: A TOKEN MUST COME FROM A VOCABULARY YOU OWN, AND EXCEPTION CLASSES ARE AN OPEN SET
    NOBODY OWNS. The class is still emitted per probe through `_log`, which is GATED -- so in
    production you get the SITE and not the CLASS. That is a real limit and it is the honest one:
    a closed tally I can promise, or an open one I cannot. ***
    m = 0 ON WHETHER TWO SITES IS THE RIGHT GRANULARITY -- `ci` first counted 14 reachable sites
    and corrected to 2 explicit plus 2 except paths; mine are the 2 that can be told apart from
    the caller without inventing a vocabulary.
    """
    if sites is not None:
        sites[token] = sites.get(token, 0) + 1


def _probe(master_path, master_stream, candidate_path, candidate_stream,
           start_seconds, window_seconds, work_dir, tag, sample_rate, sites=None):
    """Both sides extracted at the SAME absolute time, so the fingerprint grids
    stay aligned and no fractional-hop shift is involved.

    Returns (offset_ms, fidelity, offset_points, quantum_ms, rms) or None.
    """
    # THE CALLER OWNS UNIQUENESS. `tag` is unique only WITHIN a `work_dir`: every
    # caller here passes a positional tag (`f"s{index}"`, `f"b{...}"`), so two
    # concurrent probes of DIFFERENT FILE PAIRS sharing one `work_dir` write the same
    # two paths and each correlates whichever extraction won the race.
    #
    # Production is safe and it is safe BY THE CALLER, not by this line:
    # `merge_video_repair.py:509` builds `work_dir = path.join(work_root, key)` per
    # candidate, and `gestionar_show/fusion.py:337` gives a tmpFolder per folder and
    # episode. Nothing in this module enforced it and nothing said so.
    #
    # Written down because it cost twelve rows. `lab/gapcost.py` was safe only because
    # it was SEQUENTIAL; a census of mine put three workers on one tmpFolder and
    # produced a completely plausible stratification -- fidelities of 0.93 to 0.97,
    # two files declining on the scatter guard, nothing in any row saying "wrong".
    # The only symptom was `audio_sync not working` on a stderr nobody was reading.
    #
    # SO: IF YOU PARALLELISE A CALLER, GIVE EACH WORKER ITS OWN `work_dir`.
    master_window = path.join(work_dir, f"cpl_m_{tag}.wav")
    candidate_window = path.join(work_dir, f"cpl_c_{tag}.wav")
    try:
        # IMMEDIATELY-PRE-CALL, AT THIS CALLER (owner's order via the Lead,
        # 2026-09-22): `_extract` logs its own ffmpeg call from inside, but
        # that line does not carry `tag`/`start_seconds` -- the probe
        # identity a caller three hundred lines up needs to know WHICH of
        # this pair's ~30 probes is in flight, not just which file.
        tools.dev_log(f"locator: _probe extracting master tag={tag} "
                      f"start_seconds={start_seconds} file={master_path}\n")
        _extract(master_path, master_stream, start_seconds, window_seconds,
                 master_window, sample_rate)
        tools.dev_log(f"locator: _probe extracting candidate tag={tag} "
                      f"start_seconds={start_seconds} file={candidate_path}\n")
        _extract(candidate_path, candidate_stream, start_seconds, window_seconds,
                 candidate_window, sample_rate)
        signal = _rms(master_window)
        # THE OFFSET SOURCE, unquantised. Architect ruling P6, 2026-09-21
        # (audit of this module's correlation usage): `second_correlation`'s
        # OFFSET is forbidden for cut/boundary placement -- tolerated here
        # only as the coarse stage during migration to the vectorial
        # instrument that mandate names, never frame-accurate. Forensic
        # confirmed, two independent ways, that it returns exactly
        # `(file, offset)` in BOTH branches (Rust and the Python fallback),
        # and that every artefact's full unfiltered JSON dict carries exactly
        # `file` and `offset_seconds` -- zero third fields. No fidelity comes
        # from this call; do not read one out of it.
        #
        # PRIORITY SITE (owner's order via the Lead, 2026-09-22, measured
        # ceiling): `audioCorrelation.py:264` calls `audio_sync` through
        # `launch_cmdExt_with_timeout_reload(max_restart=3, timeout=28800)`
        # -- 8 hours per attempt, up to 4 attempts, a 32-hour ceiling that
        # BRACKETS tonight's observed 7-hour hang. `audioCorrelation.py` is
        # frozen; this line instruments the OPEN caller instead of touching
        # the frozen callee -- the only lever available on this call.
        tools.dev_log(f"locator: _probe calling audioCorrelation."
                      f"second_correlation tag={tag} "
                      f"master_window={master_window} "
                      f"candidate_window={candidate_window}\n")
        which_file, seconds = audioCorrelation.second_correlation(master_window, candidate_window)
        if path.abspath(which_file) == path.abspath(master_window):
            offset_ms = -seconds * 1000.0
        elif path.abspath(which_file) == path.abspath(candidate_window):
            offset_ms = seconds * 1000.0
        else:
            _note_site(sites, "correlator_named_neither_window")
            return None
        # THE ONLY SOURCE OF `fidelity` IN THIS MODULE. P6's ruling: the ban
        # forbids the named functions "for cut finding" -- `correlate()`'s
        # role here is a COMPARABILITY SCREEN (median_fidelity_below_floor,
        # the pairing bar), not cut localization, so that role is explicitly
        # TOLERATED under this corrected label. Its OFFSET (`points`,
        # `delay_ms`, read below only to derive `quantum`) remains FORBIDDEN
        # for boundary placement -- exactly the split `pal_saturation_screen`
        # already treats these `points` as: a search-bound ARTIFACT check,
        # never a location. Load-bearing by construction: nothing else in
        # this module supplies fidelity, so `correlate()` cannot simply be
        # swapped out until the vectorial instrument (which carries fidelity
        # natively, a per-point Hamming similarity vector) replaces it --
        # that recalibration is the future Stage-1 migration's scope, not
        # this module's, and not now.
        fidelity, points, delay_ms = audioCorrelation.correlate(
            master_window, candidate_window, window_seconds)
        quantum = int(round(delay_ms / -points)) if points else None
        return offset_ms, fidelity, -points, quantum, signal
    except ExtractProducedNothing:
        # ITS OWN SITE. ci traced this to a root cause; folding it into the generic
        # catch would report "something raised" for the one failure we can now name.
        _note_site(sites, "extract_produced_no_audio")
        _log(f"probe at {start_seconds:.1f}s: extract produced no audio")
        return None
    except Exception as error:                        # noqa: BLE001 — logged, not swallowed
        # THE TYPE, NEVER THE MESSAGE. `_extract` is called on `master_path` and
        # `candidate_path` two lines up, so an ffmpeg/ffprobe failure there raises with
        # THE MEDIA PATH IN ITS MESSAGE, and this line writes to `tools.logs`, which
        # `gestionar_show/fusion.py:412` puts into the error file verbatim.
        #
        # WRITE_ZONES.MD section 8: quote a log line only if EVERY FIELD IN IT COMES
        # FROM A VOCABULARY YOU OWN. An arbitrary exception message is not a vocabulary
        # anyone owns -- it is whatever the failing tool decided to say, and it commonly
        # says the input path.
        #
        # Not redacted, replaced: vmsam-dev-4's rule is that a redactor which cannot
        # guarantee its result must not run, and it measured a series title with spaces
        # surviving its own pattern redaction in this same report. So the diagnostic
        # cost is accepted -- an exception CLASS and a probe position, both ours.
        #
        # Found by vmsam-dev-4 while it was declining to implement an allowlist fix I
        # had specified wrongly. It was reading this module to check what ungating
        # `_log` would expose; the leak is on the GATED path, so it was live in every
        # dev-mode run all along, which is the configuration ci and forensic read.
        _log(f"probe at {start_seconds:.1f}s failed: {type(error).__name__}")
        _note_site(sites, "extract_or_correlate_raised")
        return None
    finally:
        for temporary in (master_window, candidate_window):
            try:
                remove(temporary)
            except OSError:
                pass


def _shared_languages(master_obj, candidate_obj):
    """Which audio languages BOTH sides carry. Language codes only, never names.

    *** WHY THIS EXISTS. `no_stream_for_language` conflates two different worlds and I
    emitted the same row for both:

        the REQUESTED language is missing, but the pair shares another one
            -> a different `language` argument would let this pair proceed.
        THE PAIR SHARES NO AUDIO LANGUAGE AT ALL
            -> NO language argument exists that would let it proceed. AUDIO IS EXHAUSTED.

    The consumer could not tell them apart from my row. The per-side stream counts say
    how many streams of the REQUESTED language each side has; they say nothing about
    whether any OTHER language is shared.

    `vmsam-arch-heir` measured the consequence on six real rejected pairs: five align on
    the video channel, and TWO OF THE FIVE SHARE NO AUDIO LANGUAGE AT ALL -- one carries
    French only, the other English and Japanese, same episode, same duration. *** AUDIO
    CORRELATION CANNOT TOUCH THOSE TWO BY CONSTRUCTION. THE VIDEO CHANNEL ALIGNS THEM. ***

    So my decline is CORRECT and it is also A FLOOR. This module's contract says None means
    "I could not measure", never "the files are compatible" -- and that stays exactly true.
    What was missing is that a reader could not tell WHICH KIND of could-not-measure it had:
    *** NOT UNMEASURABLE. UNMEASURABLE BY AUDIO. ***

    THIS FUNCTION CHANGES NO DECISION AND ADDS NO ENUMERATED TOKEN. It adds a field, so the
    allowlist ruling (both halves in one batch, never after) is not engaged. I am not
    claiming the video channel works -- that is not my measurement and arch-heir marked its
    own result RUN for comparability and m=0 for drift. I am stating the audio floor, which
    IS mine.
    """
    a = getattr(master_obj, "audios", None) or {}
    b = getattr(candidate_obj, "audios", None) or {}
    try:
        return sorted(set(a) & set(b))
    except TypeError:
        # A non-mapping `audios` is a caller defect, not a shared-language answer.
        # Return None so the field reads `unknown` rather than a confident zero --
        # ABSENT IS NOT EMPTY, and a wrong zero here would read as "audio exhausted".
        return None


def _streams_for(video_obj, language):
    """EVERY stream of the language, not just the first.

    The first version read `audios[language][0]` and returned one offset for the
    language, while the repair rebuilds every stream of it. Measured on error
    id 266: the candidate carries two jpn streams **27.5 ms apart**, so one of the
    two rebuilt tracks took an offset that far wrong. dev-2's post-mux verifier
    measured the same split from the produced file — 27.8 ms — independently.
    27.5 ms is 0.66 of a frame: under the quantum the merge snaps to, under
    mkvmerge's integer milliseconds, and under the verifier's 100 ms tolerance.
    It would have shipped silently.
    """
    audios = getattr(video_obj, "audios", None)
    if not audios or language not in audios:
        return []
    return [entry["StreamOrder"] for entry in audios[language]
            if entry.get("StreamOrder") is not None]


def _all_audio_streams(video_obj):
    """EVERY audio stream with its language, not only one language's.

    `_streams_for` answers "the streams of language L". This answers "the streams",
    which is what a per-language pairing needs.
    """
    audios = getattr(video_obj, "audios", None) or {}
    out = []
    for lang, entries in audios.items():
        for entry in entries:
            order = entry.get("StreamOrder")
            if order is not None:
                out.append((order, lang))
    return sorted(out)


def _pair_candidate_streams(best_video, candidate_video, master_path, candidate_path,
                            shortest, work_dir, runs, sample_rate, sites=None):
    """Give every candidate audio stream a master partner OF ITS OWN LANGUAGE.

    Returns (accepted, measurements). `accepted` keys only the streams whose best
    partner cleared MIN_PAIRING_FIDELITY -- a stream ABSENT from it has no
    measurable offset and its consumer must refuse rather than borrow one.
    `measurements` carries every pairing that was probed, accepted or not, with the
    fidelity the bar was applied to.

    ABSENT, NEVER ZERO: a stream that cleared no partner gets no key. Not
    `fidelity: 0.0` -- an unknown fidelity is not a fidelity of zero, and a
    placeholder here would read downstream as a measurement.

    CROSS-LANGUAGE PAIRS ARE NOT PROBED. 0 of 87 reached the bar on the minimum of
    two positions in the population this rule was measured on, so spending probes
    on them buys nothing. IF THAT EVER CHANGES THIS IS THE LINE TO CHANGE -- the
    rule is an empirical result, not a property of audio.
    """
    master_streams = _all_audio_streams(best_video)
    candidate_streams_all = _all_audio_streams(candidate_video)
    by_language = {}
    for order, lang in master_streams:
        by_language.setdefault(lang, []).append(order)

    # PROBE AT PLATEAU CENTRES, NOT AT BLIND FRACTIONS OF DURATION.
    #
    # The first version used fixed 35 % and 65 % positions. On error id 125 the 35 %
    # position landed inside a transition region, returned 0.7880 against 0.9487 at
    # the other position, and the minimum-of-two REFUSED A FILE WHOSE STREAMS MATCH --
    # the pre-change locator measured 36 probes at fid_median 0.955 on it. A window
    # crossing a change point returns a DISPLACED PEAK, and a minimum turns one
    # displaced peak into a refusal of the whole file.
    #
    # k-of-n was the obvious repair and MEASUREMENT KILLED IT: over the same 127-pair
    # population with a third position added, 2-of-3 admits SEVEN cross-language pairs
    # that 2-of-2 refuses -- all on one music-and-effects-heavy file where one candidate
    # stream scores 0.87-0.96 against SEVEN different master languages. Relaxing to
    # 2-of-3 trades one loud false refusal for seven silent false accepts, and the
    # false accept is the direction that ships a wrong offset.
    #
    #     rule      cross-language accepted    same-label accepted
    #     2-of-2         0 of 87                    26 of 40
    #     2-of-3         7 of 87                    26 of 40
    #     3-of-3         0 of 87                    25 of 40
    #
    # So the minimum stays and the POSITIONS change. `runs` is already computed by the
    # time pairing happens, and a plateau centre is inside a segment BY CONSTRUCTION --
    # straddling becomes impossible rather than merely unlikely.
    # THE TWO LARGEST plateaus by probe count, not the first two. The first version
    # of this took runs[:2] and REGRESSED error id 114: that file has four segments,
    # so the first two centres both sit near the head and neither samples the body.
    # Its primary stream scored 0.902 at the old blind positions and fell below the
    # bar at the new ones -- a fix for one file breaking another, caught by re-running
    # the same comparison rather than by reasoning about it.
    #
    # The largest plateau is the one with the most probes agreeing, so it is both the
    # furthest from any boundary and the best-evidenced place to ask whether two
    # streams are the same recording.
    #
    # TWO PROBES, MINIMUM, DRAWN FROM *INSIDE* THE DOMINANT PLATEAU -- not one from
    # the dominant plateau and one from a second, disagreeing plateau. Lead's
    # correction, 2026-09-15: `MIN_PAIRING_FIDELITY` was calibrated on 26 files /
    # 127 pairs specifically against the MINIMUM OF TWO PROBES ("the two-position
    # minimum is load-bearing" -- the comment above the constant). A single probe
    # is a DIFFERENT STATISTIC than the one the bar was measured against, so
    # deciding on one dominant-plateau probe alone -- my first version of this
    # patch -- would have made the calibration stop applying to what it gates,
    # even though it fixed id 108. Taking the minimum of two points *within* the
    # dominant plateau keeps the calibrated statistic (still two draws, still a
    # minimum) while no longer letting an honest, differently-offset SECOND
    # plateau veto a pairing the dominant plateau's own evidence otherwise
    # supports -- id 108, measured 2026-09-15: 89% of the file (31 of 35 coarse
    # probes) sits in one plateau at 0.9289; a 3-probe island 503 ms away, itself
    # perfectly clean, used to be one of the two votes and dragged the minimum to
    # 0.8319. Frame extraction confirmed the island is a real, spliceable local
    # divergence, not a measurement failure.
    #
    # NOT FULLY EQUIVALENT TO THE ORIGINAL CALIBRATION'S TWO DRAWS: two points
    # inside one plateau are LESS INDEPENDENT than two points from different
    # plateaus (same underlying alignment, same neighbourhood) -- so a
    # coincidental cross-language spike that reached ~0.94 at one position could
    # plausibly repeat at a second, nearby position more easily than at a second,
    # DISTANT one. This narrows the exposure the original calibration measured
    # (min-of-two across the whole file); it does not reproduce it and does not
    # claim to. Carry that hedge forward -- it is not settled by this change.
    #
    # RULED, not merely re-measured: Architect, 2026-09-15, under the owner's
    # delegation ("tu devrais pouvoir débloquer ce genre de chose -- investigue
    # et solutionne"). Grounds: (1) MIN_PAIRING_FIDELITY untouched, so the
    # 127-pair calibration formally still applies; (2) forensic's two-arm
    # measurement -- 14/14 commentary negatives reject under both geometries,
    # margin >=0.19 (0.573-0.618 against 0.85); 11/11 controls accept
    # (0.941-0.988); (3) cross-language material still refuses earlier, at
    # median_fidelity_below_floor, on every proxy tested. Hedges carried, not
    # softened: one show / 11 episodes; forensic's method cannot reliably
    # reproduce values in the 0.83-0.90 band. Compensating tripwire for
    # exactly that gap: any ACCEPT whose minimum lands in [0.85, 0.87] is a
    # flagged forensic review row in the sweep (not built here -- the sweep's
    # own module). Refutation condition, stated by the ruling: one measured
    # commentary or cross-language acceptance under this geometry reopens it.
    if runs:
        dominant_run = max(runs, key=lambda r: len(r["members"]))
        plateau_start = dominant_run["first"]
        plateau_end = dominant_run["last"] + PROBE_WINDOW_SECONDS
        span = plateau_end - plateau_start
        positions = [max(0.0, min(plateau_start + fraction * span, shortest - PROBE_WINDOW_SECONDS))
                    for fraction in PAIRING_POSITION_FRACTIONS]
    else:
        # No plateau at all -- the pre-existing whole-file blind fallback,
        # unchanged in shape, just no longer reached by "fewer than two
        # plateaus" (a single plateau now yields two positions from its own
        # span, above) -- only by zero.
        positions = [max(0.0, min(shortest * fraction, shortest - PROBE_WINDOW_SECONDS))
                    for fraction in PAIRING_POSITION_FRACTIONS]

    # THE SECOND-LARGEST PLATEAU, PROBED SEPARATELY, DIAGNOSTIC ONLY. Never part
    # of `scores`, never part of `min()`, never able to change `accepted`. This is
    # what makes a real local divergence VISIBLE without giving it a vote.
    minority_run = None
    if len(runs) >= 2:
        minority_run = sorted(runs, key=lambda r: -len(r["members"]))[1]
        minority_centre = (minority_run["first"] + minority_run["last"] + PROBE_WINDOW_SECONDS) / 2.0
        minority_centre = max(0.0, min(minority_centre, shortest - PROBE_WINDOW_SECONDS))

    accepted, measurements = {}, []
    for stream, language in candidate_streams_all:
        partners = by_language.get(language, [])
        if not partners:
            _log(f"candidate stream {stream} ({language}): the master carries no "
                 f"{language} stream, so no partner exists; no entry")
            measurements.append({"candidate_stream": stream, "language": language,
                                 "master_stream": None, "fidelity": None,
                                 "accepted": False, "reason": "no master stream of this language"})
            continue
        best = None
        for master_stream in partners:
            scores = []
            for index, centre in enumerate(positions):
                probe = _probe(master_path, master_stream, candidate_path, stream,
                               centre, PROBE_WINDOW_SECONDS, work_dir,
                               f"pair{master_stream}_{stream}_{index}", sample_rate, sites=sites)
                if probe is None:
                    scores = []
                    break
                scores.append(probe[1])
            if not scores:
                continue
            # THE MINIMUM, of the two within-plateau draws -- the calibrated
            # statistic, restored. See the comment above `positions`.
            score = min(scores)
            if best is None or score > best[1]:
                best = (master_stream, score, scores)
        if best is None:
            _log(f"candidate stream {stream} ({language}): no {language} master stream "
                 f"could be probed; no entry")
            measurements.append({"candidate_stream": stream, "language": language,
                                 "master_stream": None, "fidelity": None,
                                 "accepted": False, "reason": "every probe failed"})
            continue
        master_stream, score, all_scores = best
        rounded_scores = [round(float(s), 4) for s in all_scores]
        record = {"candidate_stream": stream, "language": language,
                  "master_stream": master_stream, "fidelity": round(float(score), 4),
                  "positions": len(positions), "position_scores": rounded_scores,
                  "accepted": bool(score >= MIN_PAIRING_FIDELITY)}
        # THE MINORITY PROBE, AGAINST THE SAME WINNING MASTER STREAM -- diagnostic
        # only, run whether the decision above accepted or declined, never fed
        # into `score`. A `None` probe (unmeasurable) leaves the field absent
        # rather than fabricating a value -- absent, never zero, same rule as
        # everywhere else in this module.
        if minority_run is not None:
            minority_probe = _probe(master_path, master_stream, candidate_path, stream,
                                    minority_centre, PROBE_WINDOW_SECONDS, work_dir,
                                    f"minority{master_stream}_{stream}", sample_rate, sites=sites)
            if minority_probe is not None:
                minority_score = round(float(minority_probe[1]), 4)
                record["minority_score"] = minority_score
                if minority_score < MIN_PAIRING_FIDELITY:
                    # DIAGNOSTIC, NOT A DECISION. A minority plateau below the bar
                    # is exactly the shape a real local divergence takes --
                    # recorded so the repair chain can look there, never
                    # subtracted from `accepted`.
                    record["minority_disagreement"] = True
        if not record["accepted"]:
            record["reason"] = (f"below {MIN_PAIRING_FIDELITY} on the minimum of "
                                f"{len(positions)} positions within the dominant plateau")
            _log(f"candidate stream {stream} ({language}): best partner master "
                 f"{master_stream} at {score:.4f} (dominant-plateau minimum), "
                 f"below {MIN_PAIRING_FIDELITY}; no entry (positions: {rounded_scores})")
        else:
            accepted[stream] = {"master_stream": master_stream,
                                "fidelity": round(float(score), 4),
                                "language": language,
                                "positions": len(positions),
                                "position_scores": rounded_scores}
            if record.get("minority_disagreement"):
                accepted[stream]["minority_disagreement"] = True
                accepted[stream]["minority_score"] = record["minority_score"]
        measurements.append(record)
    return accepted, measurements


def _audio_duration_seconds(video_obj, language):
    audios = getattr(video_obj, "audios", None)
    if not audios or language not in audios or not audios[language]:
        return None
    for key in ("Duration", "duration"):
        if key in audios[language][0]:
            try:
                return float(audios[language][0][key])
            except (TypeError, ValueError):
                pass
    return None


def _sign_flips(values):
    non_zero = [v for v in values if v]
    return sum(1 for i in range(len(non_zero) - 1)
               if (non_zero[i] > 0) != (non_zero[i + 1] > 0))


# --- rate-family slope instrument (RULING_20260921_STEP1_CLASSIFIER_DESIGN.MD) --
# Replaces nothing: this is instrument 2 of the classifier redesign, gated to run
# ONLY where the plateau machinery below finds a single run or cannot trust one
# (median_fidelity below floor) -- never on a file the plateau machinery has
# already split into multiple real runs. That gate is why this can use a plain
# least-squares fit rather than a statistic that must itself tell a staircase
# from a drift: measured directly (this module's own dev notes,
# VMSAM_HELP_AI/dev-step1-classify/lab/slope_regression_prototype.py), a genuine
# 3-cut splice file (errid 266) scores r_squared=0.8009 on a whole-file line fit
# -- INSIDE the range real PAL/NTSC drift scored (0.63-0.9997) -- so r_squared
# alone cannot separate a staircase from a drift. It does not have to: the
# plateau gate excludes the staircase population structurally before this ever
# runs, on the same machinery the splice family's own 7/7 result already proves.
#
# PROVISIONAL THRESHOLDS, first cut, not a census (same status as this module's
# other first-cut constants when they landed): R_SQUARED_MIN=0.50 sits below the
# weakest real drift measured so far (errid 57, PAL, r_squared=0.626) with margin,
# and MIN_POINTS_FOR_SLOPE=10 is far below every measured n_used (34, 34, 102).
# Flagged for the Architect exactly as NTSC_TOLERANCE and SATURATION_FRACTION
# were: right for the population measured, unverified beyond it.
RATE_SLOPE_R_SQUARED_MIN = 0.50
RATE_SLOPE_MIN_POINTS = 10
RATE_SLOPE_OUTLIER_MAD_K = 5.0

# fps-equality pre-filter for the pitch-layer NTSC check (dev-step2-resample's
# catch, 2026-09-21): a real conversion changes the declared frame rate, so
# two sides reporting the SAME fps cannot carry a speed relation. Smaller
# than the smallest real conversion this module targets (NTSC's own ~0.1%,
# 23.976 vs 24.000 = a 0.024 fps gap) -- 0.01 sits at ~40% of that gap,
# clear of ordinary float/MediaInfo rounding noise. Measured false positives
# this closes: two same-rate (23.976/23.976) real files where pitch
# measurement noise landed near the NTSC nominal by chance, one within five
# parts in a million of it.
FPS_EQUAL_TOLERANCE = 0.01


def _robust_slope_regression(starts, offsets, outlier_mad_k=RATE_SLOPE_OUTLIER_MAD_K):
    """Least-squares slope of offset_ms over probe_start_s, after excluding
    points whose residual from a FIRST-PASS fit exceeds `outlier_mad_k` times
    the median absolute residual (a standard robust-regression pre-filter).
    Two passes: fit once on everything, exclude by residual, refit on
    survivors -- this is what keeps a single wild probe (the correlator's own
    search bound exceeded on that one reading, measured on real PAL files in
    the same dev notes above) from defeating an otherwise-clean drift fit.

    Returns None if fewer than 3 points survive either pass -- a regression on
    2 points fits perfectly and proves nothing about consistency, and this
    function must not pretend otherwise.
    """
    n = len(starts)
    if n < 3:
        return None

    def _fit(xs, ys):
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx == 0:
            return None
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = sxy / sxx
        return slope, my - slope * mx

    first = _fit(starts, offsets)
    if first is None:
        return None
    slope0, intercept0 = first
    residuals = [o - (slope0 * s + intercept0) for s, o in zip(starts, offsets)]
    abs_res = sorted(abs(r) for r in residuals)
    mid = len(abs_res) // 2
    mad = abs_res[mid] if len(abs_res) % 2 else (abs_res[mid - 1] + abs_res[mid]) / 2.0
    if mad == 0:
        keep_idx = list(range(n))
    else:
        keep_idx = [i for i in range(n)
                    if abs(residuals[i]) <= outlier_mad_k * mad]
    if len(keep_idx) < 3:
        return None

    ks = [starts[i] for i in keep_idx]
    ko = [offsets[i] for i in keep_idx]
    refit = _fit(ks, ko)
    if refit is None:
        return None
    slope1, intercept1 = refit
    fitted = [slope1 * s + intercept1 for s in ks]
    ss_res = sum((o - f) ** 2 for o, f in zip(ko, fitted))
    my = sum(ko) / len(ko)
    ss_tot = sum((o - my) ** 2 for o in ko)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {"slope_ms_per_s": slope1, "intercept_ms": intercept1,
            "n_used": len(keep_idx), "n_excluded": n - len(keep_idx),
            "excluded_indices": sorted(set(range(n)) - set(keep_idx)),
            "r_squared": r_squared}


def _group_plateaus(samples):
    """samples: [(start_seconds, offset_ms)] in time order -> plateau runs.

    A run extends while the next probe stays within PLATEAU_TOLERANCE_MS of the
    run's running mean. Runs separated by less than MIN_STEP_MS are then merged,
    so noise does not become a change point.
    """
    runs = []
    for start_seconds, offset_ms in samples:
        if runs and abs(offset_ms - runs[-1]["mean"]) <= PLATEAU_TOLERANCE_MS:
            run = runs[-1]
            run["members"].append((start_seconds, offset_ms))
            run["mean"] = sum(m[1] for m in run["members"]) / len(run["members"])
            run["last"] = start_seconds
        else:
            runs.append({"members": [(start_seconds, offset_ms)], "mean": offset_ms,
                         "first": start_seconds, "last": start_seconds})
    merged = runs[:1]
    for run in runs[1:]:
        if abs(run["mean"] - merged[-1]["mean"]) < MIN_STEP_MS:
            previous = merged[-1]
            previous["members"].extend(run["members"])
            previous["mean"] = sum(m[1] for m in previous["members"]) / len(previous["members"])
            previous["last"] = run["last"]
        else:
            merged.append(run)
    return merged


def _merge_narrow_runs(runs):
    """The measurement-retention invariant, applied to `_group_plateaus`'s own
    output (Architect's ruling, 2026-09-15): *"A measurement, once made, is
    never silently discarded."* `_group_plateaus` already separates every run
    correctly at the level of raw probes -- id 12, measured 2026-09-15: three
    small runs at the head (four probes, offsets -642/-588/-567 ms) came out
    as three DISTINCT runs, not folded into the body's -1412.3356 ms plateau.
    What was lost was downstream: the per-segment usability filter drops any
    run whose bracketed window collapses to zero width, and it ran BEFORE
    anything compared the head against the body -- so the 845 ms difference
    between them was never evaluated, because every run that could have
    anchored one side of that comparison was, on its own, too narrow to
    survive. *"It didn't fail measuring. It failed keeping what it
    measured."*

    This function runs BEFORE the usability filter rather than repairing
    after it, so every measured run reaches the comparison stage. It does not
    replace the usability filter -- an unusable segment is still real and is
    still dropped, later, on its own merits (`WRITE_ZONES` note: "the
    usability filter is not the enemy"). It only stops SMALL runs from being
    discarded one at a time before anyone asks whether, taken together, they
    say something the survivors alone do not.

    NO NEW CONSTANT. Two thresholds, both already calibrated for a different
    question and reused for this one rather than invented:

    "NARROW" -- a run whose own probe span (`last - first`) is under
    `PROBE_WINDOW_SECONDS`. The exact test the segment-building loop already
    applies to flag `offset_unverified` a few screens down: a run this thin
    contains no probe whose window does not overlap a neighbouring run, by
    the same reasoning that comment gives for segments.

    "NOISE" -- a maximal run of consecutive narrow runs is noise, and is
    absorbed with NOTHING carried forward, only when it has a real
    (non-narrow) run flanking it on BOTH sides and those two flanking means
    agree within `MIN_STEP_MS` -- the exact statistic `_group_plateaus`'s own
    merge pass already uses to call a gap between two plateaus noise rather
    than a step. id 12's single interior probe at t=520 (offset -1433.7 ms,
    one member) sits between two runs both reading -1412.3356 ms: the
    question the Architect posed -- should one uncorroborated interior probe
    be allowed to split a plateau -- is answered by this same test, not by a
    separate rule: NO, when the two sides it would split already agree with
    each other, because at that point nothing distinguishes it from a probe
    that read the SAME plateau slightly wrong. A cluster with no flanking
    context on one side (it sits at the very start or end of the scan, id
    12's head and tail both) can never be called noise by a test that needs
    both sides -- there is nothing to compare it against, so it is kept.

    Everything narrow that is not noise is EVIDENCE: a maximal cluster of
    consecutive narrow runs is merged into ONE wider run -- the mean of every
    probe across the whole cluster, spanning its full range -- so the
    transition/segment stage downstream sees one candidate wide enough to
    keep or flag as unverified, instead of several individually too narrow
    to survive. Merging within a cluster, never across a non-narrow run:
    `PLATEAU_TOLERANCE_MS`/`MIN_STEP_MS` already decided a non-narrow run is
    its own plateau, and this function does not re-open that.

    A run this function keeps but could not disprove as noise may still be
    SUB-QUANTUM -- id 12's own head spread is ~70 ms against a ~129 ms
    chromaprint quantum on this file, below what any audio instrument can
    resolve as one step or several. Merging the cluster to one mean is the
    conservative answer to that, not a claim about how many real steps live
    inside it: the owner's own division of labour is unchanged by this
    function -- fpcalc targets a zone, pHash decides the frame, and a merged
    run here is a zone, not a frame-accurate answer.
    """
    n = len(runs)
    is_narrow = [(run["last"] - run["first"]) < PROBE_WINDOW_SECONDS for run in runs]
    result = []
    index = 0
    while index < n:
        if not is_narrow[index]:
            result.append(runs[index])
            index += 1
            continue
        cluster_end = index
        while cluster_end < n and is_narrow[cluster_end]:
            cluster_end += 1
        cluster = runs[index:cluster_end]
        before = result[-1] if result else None
        after = runs[cluster_end] if cluster_end < n else None
        if (before is not None and after is not None
                and abs(before["mean"] - after["mean"]) < MIN_STEP_MS):
            _log(f"absorbed {len(cluster)} narrow run(s) at "
                 f"[{cluster[0]['first']:.1f},{cluster[-1]['last']:.1f}]s as noise: "
                 f"flanking runs agree within {MIN_STEP_MS} ms "
                 f"({before['mean']:.1f} vs {after['mean']:.1f})")
        else:
            members = [member for run in cluster for member in run["members"]]
            merged_run = {
                "members": members,
                "mean": sum(member[1] for member in members) / len(members),
                "first": cluster[0]["first"],
                "last": cluster[-1]["last"],
            }
            _log(f"merged {len(cluster)} narrow run(s) at "
                 f"[{merged_run['first']:.1f},{merged_run['last']:.1f}]s into one "
                 f"evidence run, mean={merged_run['mean']:.1f}ms "
                 f"({len(members)} probe(s)) -- "
                 + (f"no run before it" if before is None else f"before={before['mean']:.1f}ms")
                 + ", "
                 + (f"no run after it" if after is None else f"after={after['mean']:.1f}ms"))
            result.append(merged_run)
        index = cluster_end

    # ABSORBING A NOISE CLUSTER CAN NEWLY PLACE TWO REAL PLATEAUS ADJACENT TO
    # EACH OTHER that `_group_plateaus` never compared directly, because the
    # noise run used to sit between them. id 12: runs 3 and 5 both read
    # -1412.3356 ms and disagree only in a trailing digit once run 4 (the
    # t=520 outlier) is absorbed -- without this pass they survive as two
    # separate runs with a spurious ~0 ms "change point" between them. Same
    # test, same constant, `_group_plateaus`'s own second pass, reapplied
    # because absorption can create the adjacency it was written to see.
    reconciled = result[:1]
    for run in result[1:]:
        if abs(run["mean"] - reconciled[-1]["mean"]) < MIN_STEP_MS:
            previous = reconciled[-1]
            previous["members"] = previous["members"] + run["members"]
            previous["mean"] = sum(m[1] for m in previous["members"]) / len(previous["members"])
            previous["last"] = run["last"]
        else:
            reconciled.append(run)
    return reconciled


def _discards(samples):
    """How many refine probes read NEITHER plateau and were thrown away.

    THE SIXTH COMPUTED-AND-NOT-EMITTED VALUE IN THIS MODULE. `_bracket_transition`
    already knows this -- it is the count of `side is None` in `samples` -- and
    reported only a width. A reader seeing a 20000 ms bracket could not tell whether
    the region was wide or the probes were noisy.

    It is recoverable by arithmetic and that is how I recovered it: the return is
    `(last_before, first_after + REFINE_WINDOW_SECONDS)`, so
    width = (first_after - last_before) + REFINE_WINDOW, and the gap between the two
    clean pairs is (width - REFINE_WINDOW) in units of REFINE_STEP. THE WIDTH IS A
    COUNT OF DISCARDED PROBES PLUS ONE. On vmsam-auditor's sweep, nine positions gave
    12000 (one step, the minimum possible) and two gave 20000 and 16000 -- two extra
    discards at cut 136, one at cut 140.

    I DERIVED THAT BY INVERTING AN ARITHMETIC EXPRESSION WHILE THE MODULE HELD THE
    INTEGER. Emitting it makes the 136/140 anomaly directly observable instead of
    reconstructible, and that anomaly is still unexplained: why those positions
    straddle more is NOT MEASURED and I did not guess.

    Counts the whole scan, including probes outside the clean pairs, because a
    straddling probe is a straddling probe wherever it sits -- narrowing it to the
    interval between the pairs would embed the same assumption the anomaly questions.
    """
    return sum(1 for _, side in samples if side is None)


def _bracket_transition(master_path, master_stream, candidate_path, candidate_stream,
                        region_start, region_end, before_ms, after_ms, work_dir,
                        sample_rate):
    """Bracket one transition using ONLY probes that sit cleanly on one plateau.

    A short probe [p, p+w] whose offset matches `before_ms` proves T > p + w.
    One matching `after_ms` proves T < p. Anything in between spans the boundary
    and is DISCARDED rather than interpreted — that is the whole point.

    Returns (low_seconds, high_seconds, narrowed) where `narrowed` is False when
    no clean pair was found and the coarse region is returned as a bound.
    """
    # TWO CONSECUTIVE clean probes are required to set an edge. One is not
    # enough: forensic's rule says a straddling window returns a DISPLACED peak,
    # arbitrary in sign, and a displaced peak can land inside PLATEAU_TOLERANCE_MS
    # of the wrong plateau by chance. A single such probe would push `last_before`
    # past the true transition and put wrong-plateau content into a segment --
    # which is the one path by which this module can DAMAGE a track rather than
    # merely waste master content.
    samples = []
    probe_at = max(0.0, region_start)
    # THE SCAN SEARCHED A REGION STRICTLY SMALLER THAN THE ONE IT REPORTED.
    #
    # This read `while probe_at <= region_end` until 2026-09-05, while the fallback
    # below returns `region_start, region_end + PROBE_WINDOW_SECONDS` as the interval
    # the transition can occupy. TWO STATEMENTS IN THIS FUNCTION ABOUT ONE QUANTITY,
    # disagreeing. The bound here is not a value anyone chose -- it is the one this
    # function already computes on its failure path and discards on its success path.
    #
    # It is also this module's own rule, stated below for the REFINE window and never
    # applied at the coarse scale: "reading AFTER at q -> if T >= q+w the whole window
    # would be before -> T < q+w". With w = PROBE_WINDOW_SECONDS that is exactly
    # T < after["first"] + PROBE_WINDOW, which is what the fallback returns.
    #
    # `region_end` is `after["first"]`, a COARSE probe START with a 60 s window, so a
    # coarse probe at q reads "after" whenever T < q + 60. after["first"] therefore
    # does NOT bound T above, and the scan stopped up to a full PROBE_WINDOW before
    # the transition could be. Coverage of the interval T can occupy: 40% -> 100%.
    #
    # MEASURED, vmsam-auditor's sweep and pass condition, both written before any fix
    # existed; 11 cut positions across a full PROBE_STEP with sample-exact ground truth.
    #   BASELINE  7/11 bound-only at exactly 100000; contains the true cut 11/11
    #   FIXED     0/11 bound-only, 7 of 7 narrowed to 12000; contains the cut 11/11
    #             4 of 4 already-narrow brackets IDENTICAL -- same low AND high edge
    #   CORPUS    id 266 -1000.60: 100000 -> 12000. id 266 -1000.92 and id 52
    #             -2003.84: unchanged. Synthetic and corpus counted SEPARATELY.
    #
    # THE BASELINE ALREADY CONTAINED THE CUT 11 OF 11. The old code was never wrong
    # about WHERE the transition is, only about how wide the bracket had to be. THE
    # DEFECT IS PRECISION, NOT ACCURACY -- an inaccurate locator eventually places a
    # cut somewhere absurd and is caught; an imprecise one is right every time and
    # expensive every time, and nothing in its output ever looks wrong.
    #
    # A HALF-WINDOW EXTENSION WAS TRIED FIRST AND FAILED. `+ PROBE_WINDOW/2` left id
    # 266's -1000.60 at 100000. The falsifier registered in advance -- "if it stays
    # bound-only my mechanism is wrong" -- separated the insufficient fix from this
    # one on the observable nominated beforehand.
    #
    # COST: 11 -> 26 refine probes per transition, 2.36x. Wall-clock did NOT scale
    # with it (174s->192s on one file, 158s->149s on another) and that is unexplained,
    # so the probe count is the honest cost figure and the timings are withheld.
    #
    # NOT MEASURED: multi-cut, sub-44100, real-media segment survival at scale, cost.
    # THE DEFECT IS FIXED; THE CORPUS IS UNMEASURED.
    #
    # Found by vmsam-auditor (the 7/11 sweep and "_bracket_transition is innocent --
    # the region it is handed is wrong"); mechanism refined with vmsam-dev-sandbox.
    while probe_at <= region_end + PROBE_WINDOW_SECONDS:
        result = _probe(master_path, master_stream, candidate_path, candidate_stream,
                        probe_at, REFINE_WINDOW_SECONDS, work_dir,
                        f"r{int(probe_at * 10)}", sample_rate)
        if result is None:
            side = None
        elif abs(result[0] - before_ms) <= PLATEAU_TOLERANCE_MS:
            side = "before"
        elif abs(result[0] - after_ms) <= PLATEAU_TOLERANCE_MS:
            side = "after"
        else:
            side = None                      # straddling: discarded, not interpreted
        samples.append((probe_at, side))
        probe_at += REFINE_STEP_SECONDS
    last_before = None
    first_after = None
    for index in range(1, len(samples)):
        if samples[index][1] == "before" and samples[index - 1][1] == "before":
            last_before = samples[index][0]
    for index in range(len(samples) - 1):
        if samples[index][1] == "after" and samples[index + 1][1] == "after":
            first_after = samples[index][0]
            break
    if last_before is None or first_after is None:
        return (region_start, region_end + PROBE_WINDOW_SECONDS, False,
                _discards(samples), len(samples))
    # WHAT A CLEAN PROBE ACTUALLY PROVES. The earlier version claimed more: that a
    # window [p, p+w] reading the BEFORE plateau proves T > p+w. It does not. A
    # window 75 % before and 25 % after still reads the before plateau, because
    # the majority dominates the peak. Measured on error id 266: a probe
    # [624, 632] read -1989.57 cleanly with the transition at ~630 INSIDE it.
    # Combined with the mirror claim from the next probe this asserted T > 632 AND
    # T < 628, so every bracket collapsed to bound-only and all precision was lost.
    #
    # `vmsam-forensic` has the compact form, and it had the same error in its own
    # published brackets: A CLEAN READING BOUNDS THE TRANSITION TO THE FAR EDGE OF
    # THE WINDOW, NEVER TO THE NEAR EDGE.
    #   reading BEFORE at p -> if T <= p the whole window would be after -> T > p
    #   reading AFTER at q  -> if T >= q+w the whole window would be before -> T < q+w
    return (last_before, first_after + REFINE_WINDOW_SECONDS, True,
            _discards(samples), len(samples))


def _search_edge_onset(master_path, master_stream, candidate_path, candidate_stream,
                       search_start, search_end, reference_ms, work_dir,
                       sample_rate, find="first"):
    '''One-sided onset search for a HEAD or TAIL CONSTRUCTED GAP -- Architect's
    ruling, 2026-09-16: a dropped or unverified run leaves no "before" (head)
    or "after" (tail) plateau to bisect `_bracket_transition`-style, so there
    is nothing to bisect AGAINST. This instead scans `[search_start,
    search_end]` seconds and finds where TWO CONSECUTIVE probes read
    `reference_ms` (the one plateau that DOES survive) within
    PLATEAU_TOLERANCE_MS -- the same two-clean-probes discipline
    `_bracket_transition` uses for its own `first_after`/`last_before`.

    BORROWED CONSTANT, NAMED AT THE BORROW SITE (house rule, 2026-09-16,
    after this exact function shipped a defect by not doing this): the probe
    step is `REFINE_STEP_SECONDS` (4.0s), `_bracket_transition`'s OWN
    constant. Its origin assumption: that function REFINES an
    ALREADY-NARROW locator bracket (an interior gap, typically single-digit
    seconds), where a 4s step still resolves the transition usefully. THIS
    function does something the origin never had to: a COLD search over a
    WIDE, initially-unknown range (here, up to `head_run["first"]` seconds
    from position 0). The assumption that licensed 4s AT THE ORIGIN --
    "the bracket is already narrow, so 4s is fine precision" -- does NOT
    transfer to a cold, wide search, and importing the constant unexamined
    is exactly what produced H-A3/H-TIER's second borrowed-parameter
    defect: on real E04 media, the true onset (~2732ms) fell INSIDE the
    unsampled gap between probes at t=0 and t=4, and the two-consecutive-
    match rule reported the confirmed transition at t=4 as if it were exact
    -- a claim fifteen times finer than a 4-second grid can support.

    THE FIX, and it is now this function's OWN contract, not the caller's
    obligation to remember: **the honest output of a step-S search that
    first matches at probe t is the interval [t-S, t]** (head) or
    [t, t+S] (tail) -- the true transition PROVABLY lies between the last
    non-matching probe and the first (pair-confirmed) matching one, and
    reporting anything narrower is fabricated precision. Returning the
    step alongside the onset makes this the CALLER's contract too: build
    the bracket from `[onset - step, onset]`, never from `[onset -
    2*quantum, onset + 2*quantum]` (a locator-quantum-scale window has no
    relation to what THIS search actually resolved).

    `find="first"` (HEAD): the FIRST confirmed matching pair -- content
    starts matching the surviving plateau there; `[onset-step, onset)` is
    the evidenced divergence, honestly bounded by what this step size can
    prove.
    `find="last"` (TAIL): the LAST confirmed matching pair -- content stops
    matching there; `[onset, onset+step)` mirrors the same honesty.

    Returns (onset_seconds, discards, probes, step_seconds). `onset_seconds`
    is None when no two-consecutive-match exists anywhere in the interval --
    the caller's OWN law, not this function's: "onset not found -> an
    honest bracket ships and NOTHING is filled." Never read a None as
    "onset at 0" or "onset at search_end".
    '''
    samples = []
    probe_at = max(0.0, search_start)
    while probe_at <= search_end:
        result = _probe(master_path, master_stream, candidate_path, candidate_stream,
                        probe_at, REFINE_WINDOW_SECONDS, work_dir,
                        f"edge{int(probe_at * 10)}", sample_rate)
        matches = result is not None and abs(result[0] - reference_ms) <= PLATEAU_TOLERANCE_MS
        samples.append((probe_at, matches))
        probe_at += REFINE_STEP_SECONDS
    onset = None
    if find == "first":
        for index in range(1, len(samples)):
            if samples[index][1] and samples[index - 1][1]:
                onset = samples[index - 1][0]
                break
    else:
        for index in range(len(samples) - 2, -1, -1):
            if samples[index][1] and samples[index + 1][1]:
                onset = samples[index + 1][0]
                break
    discards = sum(1 for _, matches in samples if not matches)
    return onset, discards, len(samples), REFINE_STEP_SECONDS


def _coverage_gaps(starts, window_seconds, span_start, span_end):
    """Intervals inside [span_start, span_end] that NO SURVIVING PROBE covered.

    `starts` are the starts of the probes that reached the plateau grouping --
    NEVER the attempted grid. Measured: TASKS/013 §3, a 30-probe grid losing
    probes 8-17 still reported scanned_seconds=[0.0, 1200.0] -- this function
    is what tells the two cases apart, by reading what actually survived.

    Windows overlap by PROBE_WINDOW_SECONDS - PROBE_STEP_SECONDS (20 s), so
    coverage is contiguous ONLY while adjacent probes survive; one lost probe
    opens 20 s, N consecutive open 40*N - 20.

    Returns a list of (low, high) in seconds, empty when the span is fully
    covered -- a MEASUREMENT, not a default: the caller must not read an empty
    list as "no gaps" unless this function ran.
    """
    if span_end <= span_start:
        return []
    covered = []
    for start_seconds in starts:
        low = max(float(start_seconds), span_start)
        high = min(float(start_seconds) + window_seconds, span_end)
        if high > low:
            covered.append((low, high))
    covered.sort()
    gaps = []
    cursor = span_start
    for low, high in covered:
        if low > cursor:
            gaps.append((cursor, low))
        cursor = max(cursor, high)
    if cursor < span_end:
        gaps.append((cursor, span_end))
    return gaps


def locate_change_points(best_video, candidate_video, language, work_dir=None):
    """Locate where `candidate_video`'s timeline diverges from `best_video`'s.

    Returns the block described in INTERFACE_dev1_dev2.md, or **None**.

    None means *I could not measure* — never *the files are compatible*. dev-2
    maps it to `no_plan` and the refusal stands. Those are different answers and
    collapsing them is the mistake this campaign exists to avoid.
    """
    # COMPUTED AT ENTRY, BEFORE ANY REFUSAL CAN BE TAKEN. `master_path` and
    # `candidate_path` are not bound until after the language and duration checks, so the
    # two EARLIEST refusals used to carry no join key at all.
    #
    # `vmsam-forensic` named the consequence and it is fatal on its side, not mine: an
    # early-refusal row carrying NEITHER a pair NOR an error id is UNJOINABLE TO ANY FILE
    # FROM THE ARTEFACT ALONE. For my own arm that is harmless -- I know what I fed in.
    # For a retrospective census over artefacts, it is the whole population lost.
    # *** THE ASYMMETRY I DEFENDED WAS AN ACCIDENT OF WHERE TWO LOCALS WERE ASSIGNED, NOT
    # A PROPERTY OF THE REFUSALS. The attributes were available at entry all along. ***
    pair_id = _digest(getattr(best_video, "filePath", None),
                      getattr(candidate_video, "filePath", None))
    master_streams = _streams_for(best_video, language)
    candidate_streams = _streams_for(candidate_video, language)
    if not master_streams or not candidate_streams:
        _log(f"no {language} stream on one side; declining")
        _shared = _shared_languages(best_video, candidate_video)
        return _decline("no_stream_for_language", "could_not_run", pair=pair_id,
                        lang=language,
                        master_streams=len(master_streams),
                        candidate_streams=len(candidate_streams),
                        shared_langs=("unknown" if _shared is None
                                      else ",".join(_shared) if _shared else "none"),
                        audio_exhausted=("unknown" if _shared is None
                                         else str(not _shared).lower()))

    master_duration = _audio_duration_seconds(best_video, language)
    candidate_duration = _audio_duration_seconds(candidate_video, language)
    if not master_duration or not candidate_duration:
        _log("audio duration unavailable; declining")
        return _decline("audio_duration_unavailable", "could_not_run", pair=pair_id,
                        lang=language,
                        master_duration_s=master_duration,
                        candidate_duration_s=candidate_duration)

    shortest = min(master_duration, candidate_duration)
    work_dir = work_dir or tools.tmpFolder
    master_path = best_video.filePath
    candidate_path = candidate_video.filePath
    reference_stream = master_streams[0]
    reference_start_ms = _start_times_ms(master_path).get(reference_stream)
    primary_stream = candidate_streams[0]

    # THE COMPARISON GRID IS A PROPERTY OF THE PAIR, NOT OF THIS MODULE.
    #
    # `video.get_less_sampling_rate` is the PIPELINE'S OWN function and
    # `mergeVideo.py:583-585` is the clamp it applies, so this reproduces the
    # consumer's grid BY CALLING THE SAME CODE rather than by restating the rule --
    # a restatement can drift from the thing it restates, and this one already had:
    # `_extract` pinned 44100 while I quoted `min(pair's lowest rate, 44100)` to
    # other agents for hours.
    #
    # Clamp direction matters and is easy to get backwards: the pipeline clamps only
    # when the derived rate is ABOVE 44100. Below it, the LOWER rate is used and this
    # module must not upsample.
    try:
        import video as _video
        comparison_grid_hz = int(_video.get_less_sampling_rate(
            best_video.audios[language], candidate_video.audios[language]))
    except Exception as error:                        # noqa: BLE001
        # TYPE ONLY, never the message -- section 8, same reason as the probe failure
        # below: an arbitrary exception is not a vocabulary anyone owns.
        _log(f"comparison grid underivable ({type(error).__name__}); using 44100")
        comparison_grid_hz = 44100
    if comparison_grid_hz > 44100:
        comparison_grid_hz = 44100
    _log(f"comparison grid {comparison_grid_hz} Hz "
         f"(pair-derived; clamped only above 44100)")

    # --- coarse scan: the WHOLE file, no privileged region -------------------
    # The grid stops at the last probe that fits, which leaves up to
    # PROBE_STEP_SECONDS of TAIL unscanned — measured at 39 s in the worst case.
    # That is the head blind spot again at the other end, and a trimmed tail is
    # exactly a change point there. So the last probe is anchored to the END.
    # A PROBE BEFORE THE MASTER STREAM'S OWN START READS A SPURIOUS OFFSET.
    #
    # `_probe` seeks by presentation timestamp. On a stream whose first packet is
    # stamped 1.103 s, a probe at t=0 cannot return audio from t=0 -- there is none
    # -- so it returns the stream's opening against a candidate window that really
    # does start at 0, and the offset it reports is the start_time rather than the
    # relation being measured.
    #
    # Measured on error ids 144 and 375. The head probe reported +1003 and +1090 ms
    # against a body of +22 and +1014, the run splitter read that as a change point
    # at the very start, and the resulting zero-width first segment was dropped:
    #
    #     segment 0 unusable (offset 1003 ms, master [0.0,0.0]); dropped, not declining
    #
    # So the plan began at master 100000 ms instead of 0. That is not merely a lost
    # 100 s: vmsam-dev-2 emits a master-fill piece for [0, first_segment_start), and
    # that piece is cut from the master stream at source 0, so it carries the same
    # defect one level down. On id 173, whose plan does begin at 0, none of this
    # happens.
    #
    # The cure is to start the grid where the reference stream actually begins.
    # Everything before that is a region no probe can measure.
    first_measurable = max(0.0, (reference_start_ms or 0.0) / 1000.0)
    if first_measurable > 0:
        _log(f"master reference stream begins at {first_measurable * 1000:.0f} ms; "
             f"probing starts there, not at 0")
    starts = []
    start_seconds = first_measurable
    while start_seconds + PROBE_WINDOW_SECONDS <= shortest:
        starts.append(start_seconds)
        start_seconds += PROBE_STEP_SECONDS
    tail_start = shortest - PROBE_WINDOW_SECONDS
    if tail_start > 0 and (not starts or tail_start - starts[-1] > 1.0):
        starts.append(tail_start)
    # THE PER-SITE TALLY ci ASKED FOR. `probes_attempted - probes_raw` says HOW MANY
    # probes were lost; this says WHERE. Closed vocabulary, two members, both mine.
    probe_sites = {}
    raw = []
    for index, probe_start in enumerate(starts):
        result = _probe(master_path, reference_stream, candidate_path, primary_stream,
                        probe_start, PROBE_WINDOW_SECONDS, work_dir, f"s{index}",
                        comparison_grid_hz, sites=probe_sites)
        if result is not None:
            raw.append((probe_start, result))
    if len(raw) < 3:
        _log(f"only {len(raw)} usable probes over {shortest:.0f}s; declining")
        return _decline("too_few_usable_probes", "could_not_run", pair=pair_id,
                        probes_raw=len(raw),
                        probes_attempted=len(starts),
                        probes_required=3,
                        # WHERE the lost probes were lost. Empty means every attempted probe
                        # returned -- the decline is then about the COUNT and not about failure.
                        refusal_sites=(",".join(f"{k}:{v}" for k, v in sorted(probe_sites.items()))
                                       or "none"),
                        span_s=f"{shortest:.0f}")

    # --- no-signal guard -----------------------------------------------------
    median_energy = median([r[1][4] for r in raw])
    kept = [r for r in raw if r[1][4] >= LOW_SIGNAL_FRACTION * median_energy]
    dropped = len(raw) - len(kept)
    if dropped:
        _log(f"dropped {dropped} probe(s) below {LOW_SIGNAL_FRACTION:.0%} of median energy")

    # MOVED HERE, pure arithmetic over `kept` (already built above), no new
    # probe, no decode (RULING_20260921_STEP1_CLASSIFIER_DESIGN.MD; measured
    # cheap by the Lead before landing). Was built ~90 lines further down,
    # right before the per-stream pairing section; that later site now reuses
    # this same `runs`, not a second computation. The gated rate-family
    # instrument a little further below needs to know, before the existing
    # fidelity/scatter declines run, whether the file has already resolved
    # into multiple real plateaus -- that is the only reason the timing moved.
    runs = _group_plateaus([(r[0], r[1][0]) for r in kept])
    # MEASUREMENT-RETENTION INVARIANT: every measured run reaches the
    # transition/segment stage, or is absorbed as noise against a named,
    # calibrated test -- never dropped one at a time by a filter that runs
    # before anything compares it to its neighbours. See `_merge_narrow_runs`.
    runs = _merge_narrow_runs(runs)

    if len(kept) < 3:
        _log("too few probes carry signal; declining")
        return _decline("too_few_probes_with_signal", "could_not_run", pair=pair_id,
            # *** `probes_attempted` ADDED. ci measured that the artefact cannot say whether a
            # probe was lost to an EXTRACTION FAILURE or to the ENERGY GUARD. The two are
            # recoverable only as differences:  attempted-raw = EXTRACTION,  raw-kept = GUARD.
            # This row carried kept and raw and NOT attempted, so the second subtraction was
            # available and THE FIRST WAS NOT.  *** THE VALUE WAS IN SCOPE THE WHOLE TIME --
            # not a measurement I could not make, one I never emitted. ONE MISSING VALUE MAKES
            # A MECHANISM UNRECOVERABLE, AND m ON IT WAS 0 UNTIL THIS LINE. ***
                        probes_kept=len(kept),
                        probes_raw=len(raw),
                        probes_attempted=len(starts),
                        probes_required=3,
                        signal_floor_fraction=LOW_SIGNAL_FRACTION,
                        # *** THE TALLY RODE ONE DECLINE ROW OF ELEVEN AND WAS THEREFORE NEVER OBSERVED. ci
                        # deployed it and ran five entries: `refusal_sites` appeared in ZERO. The reason is not
                        # that probes never fail -- it is that I emitted the tally ONLY on `too_few_usable_probes`,
                        # so a run whose probes fail but which still has enough to CONTINUE said nothing at all.
                        # *** THE CASE ci CARES ABOUT -- PROBES LOST WHILE THE RUN PROCEEDS -- WAS THE ONE CASE
                        # THE FIELD COULD NOT REPORT. AN INSTRUMENT PLACED ONLY ON THE FAILURE PATH CANNOT
                        # MEASURE ATTRITION THAT DOES NOT CAUSE FAILURE. *** Same dict, still in scope.
                        refusal_sites=(",".join(f"{k}:{v}" for k, v in sorted(probe_sites.items()))
                                       or "none"))

    # STAGE 1 SCREEN (DESIGN_PAL_SPEED_FAMILY_20260916.MD): "this offset is
    # not a media fact." A probe whose quantised offset sits at the
    # correlator's own search bound is not a measurement -- the correlator
    # ran out of window, not out of signal -- so it is excluded here, before
    # any of the aggregates below are built from it. Exactly what
    # `no_quantised_points` two steps below already does for an unusable
    # quantum; this is the same admission-time discipline for a different
    # unusable reading.
    #
    # SCOPED NARROWLY, AND SAID SO: this filters ONLY the refusal-decision
    # aggregates built below (offsets/fidelities/quanta/distinct_points/
    # flips). `kept` ITSELF IS UNTOUCHED -- achieved coverage (`_coverage_gaps`
    # a few hundred lines down) and plateau/run construction (`_group_plateaus`)
    # still see every probe that actually ran, saturated or not, because
    # "never scanned" and "scanned but unusable" are different facts and
    # widening the filter to those would conflate them. That is a larger,
    # different change this design does not ask for.
    screened_kept, saturation_decline, saturation_stats = (
        pal_saturation_screen.screen_decline_detail(
            kept, PROBE_WINDOW_SECONDS, CHROMAPRINT_HOP_SECONDS,
            audioCorrelation.min_overlap))
    # EMITTED UNCONDITIONALLY, DECLINE OR NOT (Architect's ruling,
    # 2026-09-16): the population census the trigger bar still needs
    # (20 real PAL ids + a content-mismatch control) accrues from live runs
    # only if every pass states its fraction, not only the ones that decline.
    _log(f"{language}: saturation screen kept={saturation_stats['probes_kept']} "
         f"saturated={saturation_stats['probes_saturated']} "
         f"observed_fraction={saturation_stats['observed_fraction']} "
         f"bound={saturation_stats['search_bound_points']} "
         f"threshold={saturation_stats['threshold_fraction']} "
         f"evaluable={saturation_stats['evaluable']}")
    if saturation_decline is not None:
        saturation_cause = saturation_decline.pop("cause", "offsets_saturated_at_search_bound")
        return _decline(saturation_cause, "could_not_run",
                        pair=pair_id, **saturation_decline)

    offsets = [r[1][0] for r in screened_kept]
    fidelities = [r[1][1] for r in screened_kept]
    quanta = [r[1][3] for r in screened_kept if r[1][3]]
    # DECLINE AT ADMISSION, NOT THREE FRAMES LATER. `:step_points` a few hundred
    # lines below (`step_ms / quantum_ms`) is the very next arithmetic use of
    # this value and it is unconditional on the success path -- so a `None`
    # here would not fail here, it would fail there, as a bare `TypeError` with
    # no reason attached. There is no line between this one and that one where
    # inserting a check would be less late. See `no_quantised_points` above for
    # what this condition actually is and why `125` was never a safe stand-in
    # for it (the Architect's ruling on the E3 mission; census in
    # VMSAM_HELP_AI/dev-pal/001-quantum-ms-census.MD).
    if not quanta:
        _log(f"{language}: {len(screened_kept)} probes kept, none carried a usable "
             f"quantised offset (points==0 on every one); declining")
        return _decline("no_quantised_points", "could_not_run", pair=pair_id,
                        probes_kept=len(screened_kept))
    quantum_ms = int(median(quanta))
    median_fidelity = median(fidelities)
    distinct_points = len({r[1][2] for r in screened_kept})
    flips = _sign_flips(offsets)
    _log(f"{language}: {len(screened_kept)} probes over [0,{shortest:.0f}]s, "
         f"fid_median={median_fidelity:.3f}, quantum={quantum_ms}ms, "
         f"distinct_points={distinct_points}, flips={flips}")

    # --- rate-family instruments (RULING_20260921_STEP1_CLASSIFIER_DESIGN.MD) ---
    # GATED, ADDITIVE, NEVER REPLACES THE EXISTING PATH: fires only where the
    # plateau machinery above (`runs`) has NOT already split this file into
    # multiple real plateaus -- a genuine splice file cannot reach this block,
    # by construction, the same machinery the splice family's own 7/7 result
    # already proves. On any decline or exception here, execution falls
    # through UNCHANGED to the monotone/scatter/pairing logic below -- same
    # discipline as the vector-locator swap-in (`zone_similarity_vector`,
    # further down this file): a refining/adding stage may decline, never
    # break the path it sits in front of.
    _rate_gate_open = len(runs) <= 1 or median_fidelity < MIN_MEDIAN_FIDELITY
    if _rate_gate_open:
        _log(f"{language}: rate-family gate open (runs={len(runs)}, "
             f"fid_median={median_fidelity:.3f}); trying pitch then slope")
        # FPS-EQUALITY PRE-FILTER (dev-step2-resample's catch, same evening):
        # a PAL/NTSC conversion CHANGES the declared frame rate -- that is
        # what the conversion IS. If both sides report the same fps, no
        # conversion happened and a speed relation is impossible BY
        # CONSTRUCTION, so the pitch check must not even be tried. Measured
        # false positives this closes: two real same-rate (23.976/23.976)
        # files where pitch measurement noise landed near-unity and one,
        # `kuroshitsuji_10`, matched the NTSC nominal by five parts in a
        # million (0.000528 from unity vs 0.000472 from the NTSC nominal --
        # essentially equidistant, not a rate recognition). Arithmetic, not
        # a tuned margin: `FPS_EQUAL_TOLERANCE` only has to be smaller than
        # the smallest real conversion this module cares about (NTSC's own
        # ~0.1%, i.e. 23.976 vs 24.000, a 0.024 fps gap) -- 0.01 sits at
        # roughly 40% of that gap, well clear of ordinary float/MediaInfo
        # rounding noise in a single declared rate. Missing fps on either
        # side does NOT skip the check -- absence of the guarding fact must
        # not be read as permission, same rule as everywhere else in this
        # module ("absent, never zero").
        _master_fps = getattr(best_video, "get_fps", lambda: None)()
        _candidate_fps = getattr(candidate_video, "get_fps", lambda: None)()
        _fps_equal = (_master_fps is not None and _candidate_fps is not None
                      and abs(_master_fps - _candidate_fps) < FPS_EQUAL_TOLERANCE)
        if _fps_equal:
            _log(f"{language}: master/candidate fps equal "
                 f"({_master_fps} vs {_candidate_fps}); no conversion possible, "
                 f"skipping the pitch-layer NTSC check")
        # Instrument 1: pitch-layer NTSC recognizer, unconditional (except for
        # the fps-equality case above), first -- NTSC_KNIFE_EDGE (in force):
        # duration cannot carry this signal, so this measures pitch directly
        # rather than inferring from duration. Local import + broad except,
        # same pattern as the PAL chain lower in this file: a new
        # instrument's failure must become a measurement, never a crashed
        # merge.
        _ntsc_result = None
        if not _fps_equal:
            try:
                import pal_pitch_confirmer
                _ntsc_probe_window = min(180.0, shortest * 0.5)
                _ntsc_probe_start = shortest * 0.3
                tools.dev_log(f"locator: calling pal_pitch_confirmer.confirm_ntsc "
                              f"master={master_path} candidate={candidate_path} "
                              f"probe_start={_ntsc_probe_start} "
                              f"probe_window={_ntsc_probe_window}\n")
                _ntsc_result = pal_pitch_confirmer.confirm_ntsc(
                    master_path, candidate_path, _ntsc_probe_start, _ntsc_probe_window)
            except Exception as error:                          # noqa: BLE001 -- see above
                _log(f"pitch-layer NTSC check errored: {type(error).__name__}")
                _ntsc_result = None
        if _ntsc_result is not None and _ntsc_result.get("matched"):
            _log(f"{language}: pitch-layer NTSC match ({_ntsc_result['matched']}, "
                 f"ratio={_ntsc_result['measured_ratio']}); declining as rate family")
            return _decline("speed_relation_suspected", "could_not_run", pair=pair_id,
                            rate_instrument="pitch_ntsc",
                            rate_instrument_matched=_ntsc_result["matched"],
                            rate_instrument_measured_ratio=_ntsc_result["measured_ratio"],
                            rate_instrument_predicted_ratio=_ntsc_result["predicted_ratio"],
                            rate_instrument_peak=_ntsc_result["peak"],
                            plateau_runs=len(runs))
        # Instrument 2: offset-slope regression over the whole per-window
        # series -- pools all windows (robust to per-window noise), is
        # sign-invariant (immune to the zero-crossing artifact in the old
        # adjacency test), and accumulates a small per-window drift into a
        # measurable whole-file slope. `screened_kept` is the same population
        # the median-fidelity/monotone checks below use -- already past the
        # saturation screen, not yet past this module's OWN outlier
        # exclusion, which `_robust_slope_regression` applies on top.
        _slope_starts = [r[0] for r in screened_kept]
        _slope_offsets = [r[1][0] for r in screened_kept]
        _slope_result = _robust_slope_regression(_slope_starts, _slope_offsets)
        if (_slope_result is not None
                and _slope_result["n_used"] >= RATE_SLOPE_MIN_POINTS
                and _slope_result["r_squared"] >= RATE_SLOPE_R_SQUARED_MIN):
            _log(f"{language}: slope regression significant "
                 f"(slope={_slope_result['slope_ms_per_s']:.4f}ms/s, "
                 f"r2={_slope_result['r_squared']:.4f}, "
                 f"n_used={_slope_result['n_used']}/{len(_slope_starts)}); "
                 f"declining as rate family")
            return _decline("speed_relation_suspected", "could_not_run", pair=pair_id,
                            rate_instrument="slope_regression",
                            rate_instrument_slope_ms_per_s=round(
                                _slope_result["slope_ms_per_s"], 4),
                            rate_instrument_r_squared=round(_slope_result["r_squared"], 4),
                            rate_instrument_n_used=_slope_result["n_used"],
                            rate_instrument_n_excluded=_slope_result["n_excluded"],
                            plateau_runs=len(runs))
        _log(f"{language}: rate-family gate open but neither instrument fired "
             f"(ntsc_matched={_ntsc_result.get('matched') if _ntsc_result else None}, "
             f"slope_r2={_slope_result['r_squared'] if _slope_result else None}); "
             f"falling through to the existing path")

    # --- refusals, each with a measured basis --------------------------------
    if median_fidelity < MIN_MEDIAN_FIDELITY:
        monotone = (all(offsets[i] <= offsets[i + 1] for i in range(len(offsets) - 1))
                    or all(offsets[i] >= offsets[i + 1] for i in range(len(offsets) - 1)))
        # E2 (Architect's finding, FINDING_median_ignores_consistency_20260916.md,
        # 2026-09-16): the MEDIAN alone cannot tell a real-but-quiet alignment
        # from a genuinely scattered one. A probe scores low fidelity for
        # reasons that have nothing to do with whether the offset is real
        # (quiet audio, a low-signal scene, a mix difference); the offsets
        # ALREADY COMPUTED ABOVE (`distinct_points`, `flips`) are evidence
        # this branch never consulted before landing here.
        #
        # WHAT ACTUALLY SEPARATES THE TWO REQUIRED CONTROLS, STATED SO
        # ADJACENCY IS NOT MISREAD AS AN ARGUMENT (Lead's correction,
        # 2026-09-16): `fidelity_max` ALONE separates errid:307 (0.9527)
        # from errid:250 (0.6266) -- 250 declines because 0.6266 < 0.70,
        # full stop, regardless of what `distinct_points`/`flips` read.
        # `distinct_points`/`flips` do NOT discriminate this pair; do not
        # read the paragraph below as claiming they do.
        #
        # `flips` is INERT on this pair specifically: `_sign_flips` reads
        # the SIGN of the raw offset, and both 307's and 250's published
        # offsets are all-positive, so `flips=0` for both regardless of
        # scatter. Present as a guard (reused from the sibling check), not
        # as a discriminator here.
        #
        # `distinct_points` earns its place on a DIFFERENT, UNTESTED hazard:
        # a scattered pair whose `fidelity_max` happens to clear the floor
        # anyway -- neither 307 nor 250 exercises this. Fired as a literal
        # (dev-subcue's ARM C, TASK_E2_MEDIAN_CONSISTENCY_DESIGN.MD):
        # 250-shaped scatter (distinct_points=35) with `fidelity_max`
        # artificially raised to 0.95 still declines, on `35 > 4` alone --
        # that is the case this term exists to catch.
        #
        # MEASURED, errid:307 (forensic, REPORT_S2_id307_fidelity_vs_
        # offset_conflict.md): 37 probes, median_fidelity 0.6039 (below
        # floor) yet fidelity_max 0.9527, and 35 of 37 offsets agree to one
        # quantum (distinct_points=3, computed by hand from the artefact's
        # own published raw list, quantum=129ms -- INFERRED, not re-run on
        # the original media). CONTROL, errid:250: fidelity_max 0.6266 --
        # never clears the floor. **`distinct_points` for errid:250 is a
        # SYNTHETIC STAND-IN built from its published aggregate statistics
        # (spread, probe count), NOT the real artefact's number -- no raw
        # per-probe log for 250 could be found (checked `/config/output`,
        # `/tmp` scratch, and three seats' docs under `VMSAM_HELP_AI/`) and
        # the real value could not be obtained.**
        #
        # Reused thresholds (`MAX_DISTINCT_POINTS`, `MAX_SIGN_FLIPS`),
        # already calibrated for the sibling `offsets_scattered` guard below
        # -- not a new number invented for this one pair; if a
        # purpose-specific threshold is wanted instead, that is the
        # Architect's call. ONLY evaluated on the non-monotone path: a
        # monotone drift already routes to `speed_relation_suspected` and
        # that ROUTING is UNCHANGED (Architect's reserved block) --
        # `consistent` is forced False whenever `monotone` is True so the
        # existing routing is never touched by this addition.
        fidelity_max = max(fidelities)
        consistent = (not monotone and distinct_points <= MAX_DISTINCT_POINTS
                     and flips <= MAX_SIGN_FLIPS and fidelity_max >= MIN_MEDIAN_FIDELITY)
        # EMPTY UNLESS `monotone`: the PAL/speed chain below only ever runs on
        # the `speed_relation_suspected` reason. A pair declining
        # `median_fidelity_below_floor` through this SAME `_decline` call
        # (the `not consistent` branch, non-monotone) must stay
        # byte-identical to before this landing -- the regression the Lead
        # named. Spread via `**pal_fields` at the call site so the two
        # reasons genuinely diverge rather than sharing a placeholder value.
        pal_fields = {}
        if monotone:
            # Low fidelity with a monotone drift is a SPEED relation, which is
            # objective 3's problem. Refusing on fidelity alone would refuse the
            # whole family the speed repair exists for.
            _log("fidelity low but drift monotone: speed relation suspected; declining")
            # PAL/NTSC SPEED CHAIN, DETECT-AND-REPORT ONLY (owner order, 2026-09-17,
            # relayed by the Lead). This is the exact site the chain was designed
            # for: production ALREADY recognises a speed relation here and ALREADY
            # declines on it -- `determine_speed_verdict` runs the discriminator and
            # both independent confirmers on the SAME pair and states what it finds,
            # AS FIELDS ON THE SAME DECLINE BELOW. It changes NOTHING about the
            # decision: the return two lines down is UNCHANGED,
            # `speed_relation_suspected`/`could_not_run`, exactly as before this
            # edit. No resample, no repair call -- a confirm that acted would be an
            # unvalidated behaviour change; a confirm that only reports is a
            # measurement, and a measurement is what was asked for.
            #
            # BROAD EXCEPTION HANDLING HERE IS DELIBERATE, NOT THE BLANKET PATTERN
            # AGENT.MD WARNS AGAINST: this is one call site, exercising a
            # brand-new, never-before-run-on-real-media chain against production
            # jobs for the first time. A failure inside it must become a measurement
            # (`pal_chain_verdict=error`), never a crashed merge -- the one thing
            # this decline site must not do is depend on an instrument nobody has
            # validated against real material yet.
            pal_fields = {"pal_chain_verdict": "error", "pal_chain_cause": "not_attempted",
                          "pal_chain_ratio": None, "pal_chain_pitch_ratio": None,
                          "pal_chain_ncc_before": None, "pal_chain_ncc_after": None,
                          "pal_chain_speed_margin": None,
                          "pal_chain_speed_margin_absent_reason": "not_attempted"}
            try:
                import pal_speed_verdict
                probe_window = min(180.0, shortest * 0.5)
                probe_start = shortest * 0.3
                tools.dev_log(f"locator: calling pal_speed_verdict."
                              f"determine_speed_verdict "
                              f"master={best_video.filePath} "
                              f"candidate={candidate_video.filePath} "
                              f"language={language} probe_start={probe_start} "
                              f"probe_window={probe_window}\n")
                pal_result = pal_speed_verdict.determine_speed_verdict(
                    best_video, candidate_video, language, probe_start, probe_window)
                disc = pal_result.get("discriminator") or {}
                pitch = pal_result.get("pitch") or {}
                ncc = pal_result.get("ncc") or {}
                pal_fields = {
                    "pal_chain_verdict": pal_result["verdict"],
                    "pal_chain_cause": pal_result.get("cause"),
                    "pal_chain_ratio": disc.get("speed_ratio"),
                    "pal_chain_pitch_ratio": pitch.get("measured_ratio"),
                    "pal_chain_ncc_before": ncc.get("ncc_before"),
                    "pal_chain_ncc_after": ncc.get("ncc_after"),
                    # P1b (VMSAM_HELP_AI/dev-pal/013-speed-margin-producer.MD):
                    # this chain's OWN margin -- ncc_after minus its own
                    # NCC_FLOOR, from pal_speed_verdict._finalize -- NEVER
                    # `plan["speed_margin"]` (a different vocabulary, a
                    # different quantity, and populating THAT one activates a
                    # real repair transform; see the file above for why this
                    # stays on this decline line instead).
                    "pal_chain_speed_margin": pal_result.get("speed_margin"),
                    "pal_chain_speed_margin_absent_reason":
                        pal_result.get("speed_margin_absent_reason"),
                }
            except Exception as error:                    # noqa: BLE001 -- see comment above
                pal_fields["pal_chain_verdict"] = "error"
                pal_fields["pal_chain_cause"] = type(error).__name__
        elif consistent:
            # MEASURABLE: the survivors agree with each other within the
            # reused scatter thresholds, and the best probe clears the floor
            # -- the low MEDIAN is explained by a few weak-but-real probes,
            # not by the alignment being false. Falls through below: no
            # decline, exactly today's behaviour when `median_fidelity`
            # itself already clears the floor.
            _log(f"{language}: median fidelity {median_fidelity:.3f} below floor but "
                 f"offsets consistent (distinct_points={distinct_points}<="
                 f"{MAX_DISTINCT_POINTS}, flips={flips}<={MAX_SIGN_FLIPS}) and "
                 f"fidelity_max={fidelity_max:.4f} clears it: measurable, not "
                 f"declining on the median alone")
        else:
            _log("fidelity at the floor with scattered offsets: no shared content; declining")
        if not consistent:
            # UNCONDITIONAL. `_log` is gated on `tools.dev`, so in production this refusal
            # emitted NOTHING and the consumer recorded "no measurement available for this
            # pair" -- which is FALSE: the probes ran, succeeded, and returned a conclusive
            # negative. Every field below is a number this module computed or a literal it
            # owns: no path, no filename, no exception text. See `THE TYPE, NEVER THE MESSAGE`.
            # *** `no_shared_content` RETIRED: A CLAIM ABOUT THE WORLD FROM AN INSTRUMENT THAT CAN
            # ONLY SPEAK ABOUT ITS OWN MEASURABILITY -- and it contradicted this module's OWN
            # contract: "None means I could not measure -- never the files are compatible."
            # I wrote that rule and emitted a token breaking it WITH THE SIGN FLIPPED.
            # `vmsam-dev-2` CONSTRUCTED the proof rather than sampling for it: one file resampled
            # from the other at the PAL constant -- SAME SOURCE, ALL CONTENT SHARED -- declines
            # here at median_fidelity 0.6012. THE TOKEN ASSERTED THEY SHARE NONE.
            # The monotone branch cannot save it: a 4.27% rate difference drifts ~20 chromaprint
            # points INSIDE one probe window, so the correlation returns noise, and *** NOISE IS
            # NOT MONOTONE -- THE GUARD FOR THE SPEED FAMILY IS DEFEATED BY THE SPEED RELATION
            # BEING LARGE ENOUGH TO DESTROY THE MEASUREMENT THAT WOULD DETECT IT. ***
            # (mechanism `vmsam-arch-aide`; constructed pair `vmsam-dev-2`.)
            # `speed_relation_suspected` STAYS -- a SUSPICION is a thing an instrument may report.
            # AND AN ARTEFACT THAT ASSERTS SOMETHING FALSE IS A DEFECT EVEN IF NO REAL FILE
            # TRIGGERS IT TODAY.
            return _decline("speed_relation_suspected" if monotone else "median_fidelity_below_floor",
                        "could_not_run", pair=pair_id,
                        median_fidelity=f"{median_fidelity:.4f}",
                        fidelity_floor=MIN_MEDIAN_FIDELITY,
                        # *** THE SPREAD OF THE SURVIVORS, NOT JUST HOW MANY SURVIVED. `vmsam-dev-2` measured
                        # that stability under widening is NECESSARY AND NOT SUFFICIENT, and saw the second
                        # half only because it PRINTED THE SPREAD INSTEAD OF COUNTING: on a real population
                        # half the probes survived AND THE SURVIVORS DISAGREED WITH EACH OTHER BY 13.7-18.8
                        # SECONDS. A COUNT WOULD HAVE SHOWN 0.44 AND HIDDEN FIFTEEN SECONDS OF DISAGREEMENT.
                        # A lag that survives widening came from the CONTENT; one that moves came from the
                        # SEARCH -- and only the spread separates them.
                        # *** ITS CONTROL REQUIREMENT, WHICH IS dev-2's AND TRAVELS WITH THE FIELD: A KNOWN-
                        # RELATED PAIR MUST READ ~0. WITHOUT THAT CONTROL A LARGE SPREAD IS UNINTERPRETABLE --
                        # IT COULD BE THE INSTRUMENT. dev-2 measured 0 ms on a pure-delay control; I HAVE NOT
                        # RUN ONE MYSELF, so this field is EMITTED AND NOT YET CALIBRATED AT THIS BENCH.
                        # m = 0 ON THE CONTROL HERE, AND THE FIELD SAYS SO RATHER THAN LOOKING MEASURED. ***
                        # *** THE MINIMUM, NOT ONLY THE MEDIAN, AND IT WAS COMPUTED AND DISCARDED. `vmsam-ci`
                        # carried `arch-heir`'s reasoning: A DECISION IS STABLE UNDER PROBE ATTRITION IFF
                        # min(per-probe fidelity) > FLOOR -- AN ORDER-STATISTIC PROPERTY NEEDING NO REPETITION,
                        # because a median cannot fall below a threshold when NO OBSERVATION is below it.
                        # *** THAT CONCLUSION IS PUBLISHED TO THE FLEET AND IT CANNOT BE CHECKED WITHOUT THIS
                        # VALUE. The list existed in memory two lines above and ONLY THE MEDIAN SURVIVED TO ANY
                        # EMISSION -- SO AN INSTRUMENTATION GAP WAS INVALIDATING AN ANALYSIS CONCLUSION, WHICH
                        # IS STRONGER THAN "IT WOULD BE USEFUL". ***
                        # AND THE CAVEAT THAT TRAVELS WITH IT, ALSO ci's: THE ORDER-STATISTIC ARGUMENT HOLDS
                        # ONLY IF LOST PROBES ARE LOST AT RANDOM. If the energy guard systematically removes
                        # QUIET passages, attrition is BIASED toward exactly where fidelity would be low and
                        # the conclusion does not apply. min alone cannot show that; the attempted/raw/kept
                        # triple is what lets a reader test it.
                        # *** AND THE MINIMUM OVER `raw`, WHICH IS THE ONE THAT ANSWERS THE BIAS QUESTION.
                        # `ci` said the property needs min over KEPT for the stability claim as stated, and that
                        # min-over-kept is EXACTLY THE STATISTIC THAT CANNOT SEE THE BIAS, because the probes
                        # the guard removed are the ones excluded from it. It proposed min over ATTEMPTED.
                        # *** MIN OVER ATTEMPTED IS NOT COMPUTABLE: a probe that fails EXTRACTION returns None
                        # and HAS NO FIDELITY AT ALL. THE QUANTITY THAT EXISTS IS MIN OVER `raw` -- everything
                        # that RETURNED, including what the energy guard then dropped -- BECAUSE `kept` IS
                        # `raw` FILTERED BY ENERGY, NOT BY FIDELITY, SO A DROPPED PROBE STILL CARRIES ONE. ***
                        # THE COMPARISON IS THE TEST: if min_raw is far below min_kept, the guard removed
                        # LOW-FIDELITY probes and the attrition is BIASED toward exactly where fidelity is low,
                        # which is the condition under which the order-statistic argument does not apply.
                        # Neither of us named this quantity; both of us named one that does not exist or cannot
                        # see. IT IS NOT MY PROPERTY TO RULE ON -- arch-heir owns the claim -- BUT THE NUMBER IT
                        # WOULD NEED IS NOW ON THE ROW INSTEAD OF BEING DISCUSSED.
                        fidelity_min_raw=f"{min(r[1][1] for r in raw):.4f}",
                        fidelity_min=f"{min(fidelities):.4f}",
                        fidelity_max=f"{max(fidelities):.4f}",
                        offset_spread_ms=f"{(max(offsets) - min(offsets)):.1f}",
                        offset_spread_control_run="no",
                        probes=len(offsets),
                        offsets_monotone=bool(monotone),
                        # THE RIDER (E2 mission): a mission touching an emission site
                        # adopts the canonical names at that site. These two fields are
                        # NEW on this call (it never consulted them before this landing),
                        # so they are born under the success line's own spellings
                        # (`:2549`-ish) rather than the older `distinct_points=`/
                        # `sign_flips=` names the sibling `offsets_scattered` decline
                        # below still uses -- no rename, no grep-compat pair needed,
                        # because there is no prior spelling on THIS call to break.
                        offset_distinct_count=distinct_points,
                        offset_sign_flips=flips,
                        # *** WHAT THIS ROW DOES NOT KNOW, STATED IN THE ROW. Below the floor,
                        # "unrelated" and "related but unmeasurable by THIS correlator" are
                        # INDISTINGUISHABLE TO ME, and the retired token picked one of them.
                        distinguishes_unrelated_from_unmeasurable="no",
                        **pal_fields)
    # NO LONGER INERT -- AND THE MECHANISM THE OLD NOTE GAVE IS REFUTED, NOT JUST ITS COUNT.
    #
    # IT SAID: "MEASURED INERT. A systematic every-7th census of a 315-record index, 45
    # files, 2026-09-05: 33 reached this line and IT FIRED ZERO TIMES. The nine files
    # scattered enough to trip it -- dp=32 fl=14, dp=32 fl=17, dp=33 fl=15, dp=32 fl=15 --
    # were already refused by the fidelity floor above, which is tested first and returns."
    #
    # MEASURED 2026-09-07, RUN, real corpus, 5 files drawn from a SPEED-MISMATCH family:
    # THIS BRANCH FIRED THREE TIMES.
    #   dp=31 fl=4 · dp=26 fl=3 · dp=21 fl=4
    # Every one PASSED the fidelity floor -- it is tested above and returns, so reaching
    # this line proves it -- and then tripped here.
    #
    # THE DISTINGUISHING VARIABLE IS SIGN FLIPS, NOT SCATTER. The census's nine carried
    # fl=14..17; mine carry fl=3..4, barely over MAX_SIGN_FLIPS. A file whose offsets flip
    # sign fifteen times has no coherent correlation left and the fidelity floor catches it
    # first, exactly as the old note said. A file that drifts across many distinct points
    # with FEW flips keeps its fidelity and arrives here. THE OLD NOTE DESCRIBED THE
    # HIGH-FLIP CASE AND GENERALISED IT TO ALL SCATTER.
    #
    # AND THE CENSUS COULD NOT HAVE SEEN THIS: it sampled every 7th record of a general
    # index -- a BASE RATE. These five were TARGETED at the one family that produces
    # low-flip drift. 0 of 33 general and 3 of 5 targeted are both true and measure
    # different things. THE BRANCH IS NOT INERT, IT IS SELECTIVE, and a base-rate sample of
    # a rare-but-clustered condition reports zero and reads as dead code.
    #
    # THIS IS WHY THE `AND` IS LOAD-BEARING AND AN `OR` WOULD BE WRONG: these three have
    # dp WELL over the max and flips barely over it. Under OR, the dp alone would refuse a
    # genuine staircase, which has many distinct points and ZERO flips.
    #
    # NOTHING ARRIVES HERE BOTH SCATTERED AND CONFIDENT.
    #
    # So the operator is not load-bearing on this population, and the AND-versus-OR
    # question is about the shape of a branch that does not fire. Recorded because I
    # argued for OR, was wrong -- a genuine five-plateau staircase has five distinct
    # points and zero flips, so OR would refuse the target population -- and because
    # two other agents spent time on its shape before anyone measured whether it runs.
    #
    # id 45 is the closest any file came: dp=4, flips=8, fidelity 0.81. It fails this
    # AND by exactly one distinct point. On n=33 that is a curiosity, not a class.
    #
    # DO NOT READ THIS AS "the guard is unnecessary". It reads as: the population that
    # would exercise it is caught earlier, and if MIN_MEDIAN_FIDELITY ever moves down,
    # this branch starts mattering and has never been exercised.
    if distinct_points > MAX_DISTINCT_POINTS and flips > MAX_SIGN_FLIPS:
        _log(f"offsets scattered ({distinct_points} distinct, {flips} flips); declining")
        return _decline("offsets_scattered", "ran_conclusive_negative", pair=pair_id,
                        distinct_points=distinct_points,
                        distinct_points_max=MAX_DISTINCT_POINTS,
                        sign_flips=flips,
                        sign_flips_max=MAX_SIGN_FLIPS)

    # --- achieved coverage, read from `kept` (SURVIVED), never `starts`
    # (ATTEMPTED) -- the predicate the Architect's grant requires, and the same
    # separation `scanned_seconds` below still gets wrong for the field, not
    # for this guard: this guard reads the honest side of that split.
    coverage_span = (min(starts), max(starts) + PROBE_WINDOW_SECONDS)
    coverage_gaps = _coverage_gaps([r[0] for r in kept], PROBE_WINDOW_SECONDS,
                                   coverage_span[0], coverage_span[1])
    if coverage_gaps:
        coverage_gap_total = round(sum(high - low for low, high in coverage_gaps), 3)
        _log(f"achieved coverage has {len(coverage_gaps)} hole(s) totalling "
             f"{coverage_gap_total:.1f}s inside the scanned span "
             f"[{coverage_span[0]:.0f},{coverage_span[1]:.0f}]s; declining")
        return _decline("coverage_incomplete", "could_not_run", pair=pair_id,
                        coverage_gaps_n=len(coverage_gaps),
                        coverage_gap_total_s=coverage_gap_total,
                        coverage_gap_bounds=(",".join(
                            f"{round(lo, 1)}-{round(hi, 1)}" for lo, hi in coverage_gaps)),
                        probes_attempted=len(starts), probes_raw=len(raw),
                        probes_kept=len(kept))
    coverage_gap_total = 0.0
    # `runs` ALREADY BUILT ABOVE, right after the no-signal guard -- moved
    # there so the gated rate-family instrument can read it before the
    # fidelity/scatter declines run (RULING_20260921_STEP1_CLASSIFIER_
    # DESIGN.MD). Not recomputed here: same `kept`, same value, one
    # computation instead of two.

    # --- per-stream plateau offsets -----------------------------------------
    # The transitions are shared: every stream of the language shows the same
    # staircase in the same places, and only the offsets differ. So the structure
    # is measured once and the offsets once per stream, at each plateau's centre.
    # Every candidate audio stream gets a master partner of its OWN language, so
    # the table below covers tracks outside the measured language instead of
    # leaving them to be assigned another language's offset by a consumer.
    # *** ci RESOLVED EVERY DECLINE BY AST AND NAMED THE MEMBER I MISSED: `every_probe_failed`
    # FIRES **12 OF 13** IN THE CORPUS, AND MY TALLY WAS ON TWO ROWS THAT HAVE FIRED **0**
    # TIMES. I wrote that an instrument on the failure path cannot see attrition that does
    # not cause failure -- CORRECT PRINCIPLE, WRONG MEMBERS: I then placed it on two failure
    # paths THAT NEVER EXECUTE. The reasoning was sound and only the membership was wrong.
    # *** AND THIS IS A **DIFFERENT** TALLY, NOT THE SAME DICT MOVED: `every_probe_failed`
    # LIVES IN THE PAIRING PHASE AND ITS PROBES ARE NOT THE MAIN LOOP'S. Emitting the main
    # loop's tally there would have named sites from probes that had nothing to do with the
    # refusal -- a plausible number attached to the wrong population. ***
    pairing_sites = {}
    pairing, pairing_measurements = _pair_candidate_streams(
        best_video, candidate_video, master_path, candidate_path, shortest, work_dir,
        runs, comparison_grid_hz, sites=pairing_sites)
    # A measured-language stream missing from the pairing is missing for one of two
    # DIFFERENT reasons and they must not be collapsed. The first version of this
    # block re-added every measured-language stream unconditionally, which put a
    # stream that had FAILED THE BAR at 0.5844 back into the table with a null
    # fidelity -- and a consumer reads PRESENCE as measurable. That is
    # "absent, never zero" violated in its other form: not a fabricated value but a
    # fabricated KEY.
    probe_failed = {m["candidate_stream"] for m in pairing_measurements
                    if m.get("reason") == "every probe failed"}
    # A REJECTED ROW MUST CARRY ITS NUMBER. The comment above MIN_PAIRING_FIDELITY says
    # the bar "is a choice inside an overlap, not a boundary", and that every
    # measurement is reported beside its verdict so someone who disagrees can move it:
    # "a row saying 'rejected' cannot be re-judged; a row saying '0.8477, rejected at
    # 0.85' can." THE DECLINE PATH DID NOT DO THAT. Both sites below named the BAR and
    # not the VALUE, and this function returns None on decline, so
    # `pairing_measurements` -- which holds every fidelity -- died with it.
    #
    # Found while certifying error 108 for vmsam-forensic: I went looking for the
    # number that caused the decline in order to state it, and there was none anywhere
    # in the log. vmsam-ci holds 53 files in NO_LOCK_IN_WINDOW and 15 in
    # UNCLASSIFIED_THIN_SUPPORT; every one is a bar someone may want to move and none
    # carries the value it was measured against.
    #
    # Same class as the success-path silence fixed earlier today, on the path where it
    # matters more: a file that produces a plan can be re-judged from the plan, a file
    # that declines leaves nothing to re-judge.
    measured_fidelity = {m["candidate_stream"]: m.get("fidelity")
                         for m in pairing_measurements}
    # THE DECLINE PATH USED TO KEEP ONLY THE MINIMUM AND LOSE THE REST --
    # `_pair_candidate_streams` always computed every position's score, but a
    # decline here only ever emitted the one number the bar was applied to.
    # Found reconstructing id 108's refusal by hand, position by position,
    # because nothing on this path had kept them: "a file that declines
    # leaves nothing to re-judge" was true of the two lines above THIS one
    # and stayed true of the number that actually decided the case. Read at
    # the SAME site as `measured_fidelity`, for the same reason.
    #
    # POSITION COUNT ONLY -- not the per-position scores. Those belong to the
    # pairing fix's own producer (`_pair_candidate_streams`'s
    # `position_scores`), which this file does not carry standalone: emitting
    # a key whose producer is absent would always read `None`, and `None`
    # means "could not measure" everywhere else in this module, never "the
    # code that would have told you was removed elsewhere in the same patch
    # split." The count alone is still real: it distinguishes a two-probe
    # minimum from a fallback geometry, live from `_pair_candidate_streams`
    # whichever selection rule is in the tree.
    measured_positions = {m["candidate_stream"]: m.get("positions")
                          for m in pairing_measurements}
    def _pairing_detail(stream):
        fid = measured_fidelity.get(stream)
        return (f"{fid} against {MIN_PAIRING_FIDELITY}" if fid is not None
                else "no fidelity measured")
    for stream in candidate_streams:
        if stream in pairing:
            continue
        if stream in probe_failed:
            # UNCHANGED: the repair rebuilds every stream of this language, so one it
            # cannot measure at all is a refusal of the whole plan.
            _log(f"stream {stream} ({language}) could not be probed at all; declining")
            return _decline("every_probe_failed", "could_not_run", pair=pair_id,
                            stream=stream, lang=language,
                            pairing_fidelity=None,
                            pairing_bar=MIN_PAIRING_FIDELITY,
                            refusal_sites=(",".join(f"{k}:{v}" for k, v in sorted(pairing_sites.items()))
                                           or "none"))
        # Measurable, and not the same content as any master stream of its language.
        # No entry, and NOT a decline: the plan stays valid for the streams that do
        # match, and the consumer refuses this one rather than borrowing an offset.
        _log(f"stream {stream} ({language}) is in the measured language but matched no "
             f"master stream of it above {MIN_PAIRING_FIDELITY}; no entry, not declining "
             f"({_pairing_detail(stream)})")
    if primary_stream not in pairing:
        # The plan's own stream failing its own bar means the offsets the segments are
        # built from were measured against a track that does not match. That is not a
        # missing entry, it is a plan with no foundation.
        _log(f"the primary stream {primary_stream} did not clear "
             f"{MIN_PAIRING_FIDELITY} against any {language} master stream; declining "
             f"({_pairing_detail(primary_stream)})")
        # ONCE PER PAIR, ON A DECISION PATH, NUMBERS ONLY -- the same shape the Lead
        # ruled on for the success line, and the same limit: nothing per-probe.
        return _decline("primary_below_pairing_bar", "ran_conclusive_negative", pair=pair_id,
                        stream=primary_stream, lang=language,
                        pairing_fidelity=measured_fidelity.get(primary_stream),
                        pairing_bar=MIN_PAIRING_FIDELITY,
                        pairing_positions=measured_positions.get(primary_stream))
    extra_streams = [s for s in sorted(pairing) if s not in candidate_streams]
    if extra_streams:
        _log(f"pairing adds {len(extra_streams)} stream(s) outside {language}: "
             + ", ".join(f"{s}->master {pairing[s]['master_stream']} "
                         f"({pairing[s]['language']}, fid {pairing[s]['fidelity']})"
                         for s in extra_streams))

    per_stream = []
    per_stream_fidelity = []
    dropped_streams = set()
    for run in runs:
        centre = (run["first"] + run["last"] + PROBE_WINDOW_SECONDS) / 2.0
        centre = max(0.0, min(centre, shortest - PROBE_WINDOW_SECONDS))
        by_stream = {}
        fidelity_by_stream = {}
        for stream in sorted(pairing):
            if stream in dropped_streams:
                continue
            partner = pairing[stream]["master_stream"]
            if stream == primary_stream:
                by_stream[stream] = run["mean"]
                continue
            result = _probe(master_path, partner, candidate_path, stream,
                            centre, PROBE_WINDOW_SECONDS, work_dir,
                            f"p{stream}_{int(centre)}", comparison_grid_hz)
            if result is None:
                if stream in candidate_streams:
                    # UNCHANGED for the measured language: a stream the repair will
                    # rebuild and cannot measure is a refusal, not a gap.
                    _log(f"stream {stream} unmeasurable at {centre:.1f}s; declining")
                    return _decline("stream_unmeasurable_at_centre", "could_not_run", pair=pair_id,
                                    stream=stream, lang=language,
                                    centre_s=f"{centre:.1f}")
                # A stream outside the measured language is dropped ENTIRELY rather
                # than measured in some segments and not others: a track placed in
                # segments 0 and 2 and missing from 1 is a gap in the middle of a
                # track, not a placement.
                _log(f"stream {stream} ({pairing[stream]['language']}) unmeasurable at "
                     f"{centre:.1f}s; dropping it from the table entirely")
                dropped_streams.add(stream)
                continue
            by_stream[stream] = result[0]
            fidelity_by_stream[stream] = round(float(result[1]), 4)
        per_stream.append(by_stream)
        per_stream_fidelity.append(fidelity_by_stream)
    if dropped_streams:
        for stream in dropped_streams:
            pairing.pop(stream, None)
            for table in per_stream:
                table.pop(stream, None)
            for table in per_stream_fidelity:
                table.pop(stream, None)

    # --- transitions, bisected ----------------------------------------------
    change_points = []
    for position in range(len(runs) - 1):
        before, after = runs[position], runs[position + 1]
        # --- STAGE 1, SPEC_ZONE_A S3c, OWNER'S ORDER 2026-09-21 --------------
        # The similarity vector locates the rupture point-by-point from the raw
        # fingerprints instead of asking the probe grid to bisect it. It runs
        # FIRST and UNCONDITIONALLY -- the owner's order the same day, "no option
        # needed to activate the pipeline": there is no flag and no fallback
        # switch, only the vector's own four named verdicts.
        #
        # WHY THIS IS NOT COSMETIC. The probe bisection leaves a bracket at the
        # SEARCH BOUND whenever it cannot land clean probes on both plateaus,
        # and that is not rare: 8 of 8 bracket readings measured across three
        # episodes of one release came back at exactly 100000 ms -- the bound
        # itself, identical every time, which is the signature of never having
        # narrowed at all rather than of narrowing badly. A rupture found by
        # V[i] hands Stage 2 a bracket one fingerprint step wide instead.
        #
        # ON A DECLINE THE OLD PATH RUNS UNCHANGED. `unreliable_degenerate_input`
        # and `window_below_detection_floor` still fall through to the probe
        # bisection exactly as before, and so does any exception -- this stage
        # is allowed to decline, never to break a merge it only refines.
        #
        # ESCALATION LADDER (dev-step3-vector, per BRIEF's own order: "when
        # the bracket does not yield ... the window and the point count are
        # the dials that widen"). MEASURED on real media, ground truth known
        # independently of this code (VMSAM_HELP_AI/dev-step3-vector/lab/
        # arm_a_real_media.py, corpus-C-structural-cut/synth-cut): the
        # NOMINAL region [before.last, after.first] read `no_rupture_found`
        # on a real, unambiguous cut, because the true divergence sat AT the
        # region's own edge -- the post-cut content the sliding window needs
        # to see the drop was simply outside the probed span. Widening
        # forward by one PROBE_WINDOW_SECONDS (content already confirmed to
        # belong to the `after` run, since that span is the extent its own
        # last member's probe window covered -- never past a neighbour's own
        # confirmed extent) turned the SAME call into a rupture found with
        # i_cut landing 0.03-0.6 s from the independently-known cut position
        # -- tighter than the old bisection's 12 s bracket on the same pair.
        # Each rung also sweeps `window_points` between the sensitivity end
        # (3, the default) and the smoothing end (5) of SPEC_ZONE_A s3c's own
        # named range, in case a marginal, noisy single-point dip needs the
        # wider average to read as a genuine crossing rather than a
        # borderline one.
        #
        # `rupture_found_offsets_unavailable` counts as a usable narrowing
        # here, not only `rupture_found`: the two verdicts differ ONLY in
        # whether the flanking-plateau lookup found a clean pair for the
        # offset_before/offset_after PROVENANCE fields -- `i_cut` and the
        # bracket itself are populated identically in both. That lookup
        # structurally straddles whenever the gap between plateaus is
        # narrower than PROBE_WINDOW_SECONDS (60 s), which is most real
        # transitions by the coarse scan's own design (PROBE_STEP_SECONDS 40
        # < PROBE_WINDOW_SECONDS 60, "so no transition can fall between two
        # probes unobserved") -- requiring strict `rupture_found` would make
        # this ladder inert on exactly the population it exists for.
        # zone_similarity_vector's own `offset_before`/`offset_after` fields
        # are not read anywhere below or downstream: `step_ms` is computed
        # from the coarse runs' own `mean` unconditionally, and
        # merge_video_chimeric.py has zero references to `offset_before_ms`/
        # `offset_after_ms` (grepped) -- so accepting the bracket without
        # that (unread) provenance costs no consumer anything.
        _zsv_verdict = None
        low = high = narrowed = discarded = probes = None
        _region_start = float(before["last"])
        _region_end = float(after["first"])
        _members = before["members"] + after["members"]
        # HELD, 2026-09-21 (dev-step3-vector, after the Lead + Architect caught a
        # containment defect neither the author nor the reviewer had checked
        # for): a real-media measurement found the bracket this ladder emits
        # does NOT reliably CONTAIN the true cut -- chromaprint's own smoothing
        # turns a real edit into a ~1.5-2s ramp, and the single-point 0.50
        # crossing this loop trusted lands in post-cut NOISE (chance baseline
        # ~0.5), not reliably at the ramp's onset. Measured 1.086s and 3.100s
        # misses on two real fixtures -- see VMSAM_HELP_AI/dev-step3-vector/
        # 001-stage1-vector-escalation-ladder.MD for the numbers and a tested
        # (not yet authorised) dual-threshold fix. EMPTY ON PURPOSE until that
        # is resolved: every rung below still RUNS and LOGS (Addendum's "log
        # all steps"), so the widening/logging machinery stays exercised and
        # visible, but NOTHING it finds is trusted for the actual bracket --
        # always falls through to `_bracket_transition` below, the same safe
        # behaviour as before this session started.
        _USABLE_ZSV_VERDICTS = ()
        # CONTAINMENT GUARD (Lead's ruling, 2026-09-21, on a real-media
        # measurement): widening buys the EXTRACTION more signal so the
        # sliding window can see a drop that straddles an edge -- it must
        # never buy the ANSWER a wider space to be found in. The plateau
        # machinery already established the transition sits in
        # [_region_start, _region_end] (the UNWIDENED, ORIGINAL gap between
        # the two confirmed runs); that is not a hypothesis this loop's own
        # widened extraction gets to revise. Measured directly, real audio,
        # a tail-case fixture with no real interior transition at all
        # (VMSAM_HELP_AI/dev-step3-vector/001-...MD, errid267 section): a
        # widened [1340,1450]s extraction found a real, well-formed-looking
        # rupture at [1346.78,1348.37]s -- 97s from where the (nonexistent,
        # in that fixture) transition would have been asked about -- and
        # NOTHING in the returned verdict distinguished it from a genuine
        # find. This guard is that distinction, made explicit: a rupture
        # whose ONSET falls outside the ORIGINAL acceptance window is not
        # the transition this call was asked to locate, whatever else it
        # is, and it is refused by name rather than shipped.
        _ACCEPTANCE_LOW, _ACCEPTANCE_HIGH = _region_start, _region_end
        _rungs_tried = []
        for _widen_s in (0.0, PROBE_WINDOW_SECONDS):
            _start = max(before["first"], _region_start - _widen_s)
            _end = min(after["last"] + PROBE_WINDOW_SECONDS, _region_end + _widen_s)
            _seconds = _end - _start
            if _seconds <= 0:
                continue
            for _wp in (zone_similarity_vector.DEFAULT_WINDOW_POINTS,
                       zone_similarity_vector.MAX_WINDOW_POINTS):
                try:
                    _zsv = zone_similarity_vector.locate_zone_by_vector(
                        master_path, reference_stream, candidate_path, primary_stream,
                        _start, _seconds, work_dir, comparison_grid_hz, _members,
                        window_points=_wp, fps_num=None, fps_den=None,
                        tag=f"zsv_{position}_w{int(_widen_s)}_p{_wp}",
                        # BASELINE_OFFSET_BLINDNESS fix (Lead's ruling,
                        # 2026-09-21): the comparison this function does is
                        # point-by-point, no lag search -- correct only when
                        # the two sides are already aligned entering the
                        # call. `before["mean"]` is the PRE-CUT plateau's
                        # own established offset, already measured by the
                        # coarse scan; anchoring the candidate extraction to
                        # it makes the pre-cut portion of the window align
                        # cleanly and lets a real divergence (or the
                        # after-plateau's own different offset) show up as
                        # a genuine drop instead of chance-level noise
                        # throughout. Measured, real media: unshifted
                        # mean_V=0.574 (chance) on a pair with zero content
                        # divergence and a 1002ms baseline; shifted by that
                        # same baseline, mean_V=0.993.
                        candidate_offset_ms=before["mean"])
                    _zsv_verdict = _zsv["verdict"]
                except Exception as error:                # noqa: BLE001 -- see above
                    _zsv_verdict = f"errored:{type(error).__name__}"
                    _zsv = None
                # POSITION, NOT ONLY THE TOKEN (the Lead's finding, 2026-09-21):
                # i_cut and the bracket already sit on `_zsv`'s own returned
                # dict and were being thrown away at the point of logging --
                # no production log line let anyone audit WHERE the vector
                # thought a rupture was, only whether it declined. Every rung
                # line below carries onset_index/candidates_examined and, once
                # a bracket exists, its absolute position -- cheap, already
                # computed, previously discarded.
                _onset_abs_str = ""
                if _zsv is not None and _zsv.get("onset_index") is not None \
                        and _zsv.get("size_point_ms"):
                    _onset_abs_str = (f",onset_abs={_start + _zsv['onset_index'] * _zsv['size_point_ms'] / 1000.0:.2f}s"
                                      f",candidates={_zsv.get('onset_candidates_examined')}")
                elif _zsv is not None:
                    _onset_abs_str = f",candidates={_zsv.get('onset_candidates_examined')}"
                if _zsv is not None and _zsv_verdict in _USABLE_ZSV_VERDICTS:
                    _onset_abs = (_start + _zsv["onset_index"]
                                  * (_zsv["size_point_ms"] or 0.0) / 1000.0)
                    if not (_ACCEPTANCE_LOW <= _onset_abs <= _ACCEPTANCE_HIGH):
                        _rungs_tried.append(
                            f"widen={_widen_s:.0f}s,points={_wp}:{_zsv_verdict}"
                            f"{_onset_abs_str}"
                            f"(REFUSED, onset {_onset_abs:.1f}s outside "
                            f"acceptance [{_ACCEPTANCE_LOW:.1f},{_ACCEPTANCE_HIGH:.1f}]"
                            f"->rupture_outside_acceptance_window)")
                        continue
                    _bracket_abs_low = _start + _zsv["bracket_low_ms"] / 1000.0
                    _bracket_abs_high = _start + _zsv["bracket_high_ms"] / 1000.0
                    _rungs_tried.append(
                        f"widen={_widen_s:.0f}s,points={_wp}:{_zsv_verdict}{_onset_abs_str}"
                        f",bracket_abs=[{_bracket_abs_low:.2f},{_bracket_abs_high:.2f}]s")
                    low = _bracket_abs_low
                    high = _bracket_abs_high
                    narrowed, discarded, probes = True, 0, 0
                    break
                _rungs_tried.append(f"widen={_widen_s:.0f}s,points={_wp}:{_zsv_verdict}{_onset_abs_str}")
            if low is not None:
                break
        _log(f"stage1 vector transition {position}: rungs=[{'; '.join(_rungs_tried)}] "
             f"region=[{_region_start:.1f},{_region_end:.1f}]s")

        if low is None:
            low, high, narrowed, discarded, probes = _bracket_transition(
                master_path, reference_stream, candidate_path, primary_stream,
                before["last"], after["first"], before["mean"], after["mean"], work_dir,
                comparison_grid_hz)
        step_ms = after["mean"] - before["mean"]
        change_points.append({"bracket_low_ms": round(low * 1000.0, 2),
                              "bracket_high_ms": round(high * 1000.0, 2),
                              "bracket_is_bound_only": not narrowed,
                              "step_ms": round(step_ms, 2),
                              "step_points": int(round(step_ms / quantum_ms)),
                              # Straddling probes thrown away while bracketing THIS
                              # transition. See `_discards`. A wide bracket with a low
                              # count means the region was wide; a wide bracket with a
                              # high count means the probes could not read either
                              # plateau, and those want different repairs.
                              "refine_discards": discarded,
                              # HOW MANY REFINE PROBES THE SCAN ACTUALLY TOOK.
                              # Added before the first container run of F4, because
                              # after that run the question about any REMAINING
                              # bound-only bracket is "was the transition outside even
                              # the EXTENDED region, or did the probes fail to read
                              # either plateau?" -- and those want different repairs.
                              # `refine_discards` answers the second. This answers the
                              # first: the probe count IS the region width in units of
                              # REFINE_STEP, so a small count means a small region.
                              # Reconstructing it afterwards would need the region
                              # bounds, which are not emitted either.
                              "refine_probes": probes})

    # A bound-only bracket returns `region_end + PROBE_WINDOW_SECONDS`, which can
    # reach past the NEXT run's first probe and swallow a whole plateau. On error
    # id 266 that dropped two segments and left 26.6 % of the timeline filled from
    # the master -- waste rather than damage, but 380 s of candidate content the
    # repair could have used. Clamp each bracket so it cannot cross the next one.
    for index in range(len(change_points) - 1):
        ceiling = change_points[index + 1]["bracket_low_ms"]
        if change_points[index]["bracket_high_ms"] > ceiling:
            change_points[index]["bracket_high_ms"] = max(
                change_points[index]["bracket_low_ms"], ceiling)
            change_points[index]["bracket_clamped_to_next"] = True

    # --- segments, with the bounds clamp ------------------------------------
    # With a negative offset the first segment must not start at master 0: it
    # would read the candidate at a negative time, and dev-2's bounds check
    # refuses the whole plan. Master [0, -offset) is a LEADING GAP filled from
    # the master, which is correct rather than a compromise. Mirror at the tail.
    segments = []
    # PARALLEL TO `segments`, NOT MERGED INTO IT: the run index each appended
    # segment came from, so a POST-PASS below can tell two segments are
    # actually ADJACENT (nothing dropped between them) before attaching the
    # bracket that separates them. `position` alone cannot do this inline --
    # the run after `position` may still be dropped later in this same loop,
    # which is only known once the loop finishes.
    segment_positions = []
    kept_runs = set()
    dropped_segments = 0
    # MEASUREMENT-RETENTION INVARIANT, the emission half: a run this filter
    # drops is a measurement, and the drop must carry its own reason in the
    # same structured stream the kept segments travel in -- not only `_log`,
    # which is gated and reaches no production artefact. Found needing this
    # first-hand: reconstructing id 12's drops took `tools.dev=True` and
    # hand instrumentation, because nothing else could show which runs were
    # dropped or why.
    dropped_segment_detail = []
    candidate_end_ms = candidate_duration * 1000.0
    master_end_ms = round(shortest * 1000.0, 2)
    boundary = 0.0
    for position, run in enumerate(runs):
        offset_ms = run["mean"]
        # A segment can never begin before -offset, at ANY position: the
        # candidate has nothing to give there, since candidate_time =
        # master_time + offset would be negative.
        # NOT int(round(...)). A candidate offset of +0.48 ms rounds -0.48 to 0,
        # so the segment starts at master 0 and reads the candidate at -0.48 ms —
        # out of bounds by less than a millisecond, and dev-2's Decimal bounds
        # check refuses the WHOLE PLAN. A sub-millisecond offset declining a pair
        # is not a fact about the media. These offsets are unquantised real
        # measurements, so the boundary stays fractional too. Found by dev-2 on
        # error id 8, mid-sweep, rather than by me.
        start_ms = max(boundary, max(0.0, -offset_ms))
        # H-A3 (2026-09-16): `widened` is computed here but NOT committed to
        # `boundary` yet -- THE EDGE IS COMMITTED BEFORE THE RUN IS DROPPED
        # was the defect, verbatim, and the fix is the ordering, not the
        # arithmetic. `widened` stays a local until this run is confirmed
        # KEPT, below the `if end_ms <= start_ms: ... continue` that decides
        # that. A run this loop is about to drop cannot leave its bracket
        # edge behind for the next run to inherit -- that edge belongs to a
        # measurement artifact that no longer exists in the output.
        widened = None
        if position < len(runs) - 1:
            change = change_points[position]
            end_ms = change["bracket_low_ms"]
            widened = change["bracket_high_ms"]
            if change["step_ms"] < 0:
                widened = max(widened, end_ms - change["step_ms"])
            change["gap_start_ms"] = round(end_ms, 2)
            change["gap_end_ms"] = round(widened, 2)
        else:
            end_ms = min(master_end_ms, candidate_end_ms - offset_ms)
        if end_ms <= start_ms:
            # This plateau lies entirely inside the leading gap, or past the
            # candidate's end. There is nothing for the repair to place, so the
            # gap simply extends to the next usable segment — which is what the
            # comment above already says a leading gap is for.
            #
            # DROP THE SEGMENT, NOT THE PAIR. Declining here threw away four
            # good plateaus on error id 266 because the first one was unusable
            # by construction. Found by vmsam-dev-2 running this on real media:
            # a guard written for a real hazard, firing correctly, with a scope
            # one case too wide — the fourth time that shape has bitten us.
            dropped_segments += 1
            dropped_segment_detail.append({
                "position": position,
                "candidate_offset_ms": round(offset_ms, 2),
                "master_start_ms": round(start_ms, 2),
                "master_end_ms": round(end_ms, 2),
                "probes_in_run": len(run["members"]),
                "reason": "bracketed window collapsed to zero width after the "
                          "leading-gap/boundary clamp",
            })
            _log(f"segment {position} unusable (offset {offset_ms:.0f} ms, "
                 f"master [{start_ms},{end_ms}]); dropped, not declining")
            continue
        # THE EDGE COMMITS ONLY NOW -- this run survived the check above, so
        # its bracket's far edge is a real boundary the NEXT run's start_ms
        # may legitimately inherit. `widened` is None on the last position
        # (no next transition to bound), which is correctly a no-op here.
        if widened is not None:
            boundary = widened
        kept_runs.add(position)
        by_stream = per_stream[position]
        fidelity_here = per_stream_fidelity[position]
        # A SEGMENT SHORTER THAN ONE PROBE WINDOW HAS NO CLEAN PROBE IN IT.
        #
        # Every window overlapping such a segment also overlaps a transition, and
        # a peak-picking correlator on a straddling window does not return a blend
        # of the two offsets — it returns a DISPLACED PEAK, arbitrary in sign and
        # unbounded by the sampling grid. So the offset below is a number this
        # instrument produced but did not measure.
        #
        # Measured on error id 266, whose first segment spans 29 s against a 60 s
        # window: an independent video instrument put its true offset at segment
        # 2's value to within 1.3 ms, meaning the reported -819.41 ms was ~168 ms
        # wrong and the change point at ~30 s did not exist at all. The repair
        # consumed that segment and its verifier could not see the error, because
        # 29 s is 2 % of the file and six spread probes never sampled it.
        #
        # FLAGGED, NOT DROPPED. Declining the pair would throw away three
        # corroborated change points to protect one bad plateau — the same scope
        # error that dropped four good segments on this very file. The caller
        # decides whether to splice a flagged segment or fill it from the master.
        span_ms = end_ms - start_ms
        unverified = span_ms < PROBE_WINDOW_SECONDS * 1000.0
        if unverified:
            _log(f"segment {position} spans {span_ms / 1000.0:.1f}s, shorter than the "
                 f"{PROBE_WINDOW_SECONDS:.0f}s probe window: no probe in it can be "
                 f"clean, so its offset is unverified")
        segments.append({
            "master_start_ms": round(start_ms, 2),
            "master_end_ms": round(end_ms, 2),
            "master_span_ms": round(span_ms, 2),
            "candidate_offset_ms": round(offset_ms, 2),
            "candidate_offset_points": int(round(offset_ms / quantum_ms)),
            "candidate_offset_ms_by_stream": {s: round(v, 2) for s, v in by_stream.items()},
            "candidate_offset_points_by_stream": {s: int(round(v / quantum_ms))
                                                  for s, v in by_stream.items()},
            # DIAGNOSTIC, NOT A GATE. The bar is applied once per file, in
            # `candidate_stream_pairing`; gating per segment would place a track in
            # segments 0 and 2 and refuse it in 1. A stream carries no key here when
            # its offset is a plateau MEAN rather than a single probe at this centre
            # -- absent, never zero.
            "candidate_offset_fidelity_by_stream": dict(fidelity_here),
            "probes_in_segment": len(run["members"]),
            "offset_unverified": unverified,
        })
        segment_positions.append(position)

    # TRANSPORT ONLY -- THIS MODULE DOES NOT REFINE ANYTHING. Attach, to the
    # segment BEFORE an interior gap, the bracket that bounds that gap
    # (`following_bracket`) -- the same dict already emitted in `change_points`,
    # copied rather than shared so a later mutation of one cannot leak into the
    # other. SPEC_ZONE_A.MD S4h ruled the refinement itself belongs in
    # `merge_video_chimeric.py`, downstream of this locator: "the comparison
    # stage produces COARSE ZONES ... it does not adopt them." This is the
    # carrier, not the refiner -- `merge_video_repair.py` (closed, WRITE_ZONES.MD
    # S4) is the only call site that has `plan` in scope on the way to
    # `assemble_on_master_timeline`, and it forwards `segments` UNCHANGED
    # (`parse_segments` copies the whole raw dict, see its own docstring on
    # field whitelists in a transport). A new key on the segment reaches the
    # open module for free; a new parameter on that closed call site would not.
    #
    # ONLY WHEN NOTHING WAS DROPPED BETWEEN THEM. `change_points[p]` is the
    # bracket between RUN p and RUN p+1; if the run right after `p` was later
    # dropped as unusable, the two SURVIVING segments either side are no
    # longer separated by that one bracket alone, and attaching it would
    # understate the gap the assembler actually has to cross.
    for index in range(len(segment_positions) - 1):
        this_position = segment_positions[index]
        next_position = segment_positions[index + 1]
        if next_position == this_position + 1 and this_position < len(change_points):
            segments[index]["following_bracket"] = dict(change_points[this_position])

    # H-TIER/H-A3 (2026-09-16), Architect's ruling. Head and tail carry NO
    # bracket today by construction -- `following_bracket` above is
    # interior-only. This fills that gap, additively (a new key on the
    # first/last segment, `leading_bracket`/`trailing_bracket`), SAME SHAPE
    # as `following_bracket` (width fields + `bracket_is_bound_only`) plus
    # `edge` and `evidence_class` -- so the chimeric-side width predicate can
    # gate the frame tier on these spans exactly like an interior one.
    # Interior brackets' own shape is UNTOUCHED (dev-locator's standard,
    # `2b36d376`: keys added, none removed, no shared key's value changed).
    #
    # GATED ON THE ARTIFACT, NEVER ON A PROXY FOR WHY IT SHOULD BE ABSENT --
    # the law the owner stated after rejecting the first version of this:
    # the only question that matters is whether a master-fill piece will
    # actually exist, not WHY one might not. `master_start_ms == 0` (head)
    # / the tail segment already reaching `master_end_ms` (tail) means NO
    # FILL WILL EXIST, so there is nothing for a bracket to bound -- that is
    # case 1, checked first, and it needs no domain reasoning at all.
    if segments:
        head_position = segment_positions[0]
        head_run = runs[head_position]
        head_offset_ms = head_run["mean"]
        head_dropped_before = any(d["position"] < head_position
                                  for d in dropped_segment_detail)
        if round(segments[0]["master_start_ms"], 6) <= 0:
            pass  # CASE 1: the head fill span has zero width -- nothing filled.
        elif not head_dropped_before and head_offset_ms < 0:
            # CASE 2, MEASURED ABSENCE: the outermost surviving plateau's own
            # offset PROVES the candidate lacks [0, B). Narrow by
            # construction -- the audio tier already predicted B; the
            # bracket exists only for the frame tier to confirm/adjust it.
            # E04's head is exactly this case (forensic: B ~ 2732ms).
            b_ms = -head_offset_ms
            low = max(0.0, b_ms - 2 * quantum_ms)
            high = b_ms + 2 * quantum_ms
            segments[0]["leading_bracket"] = {
                "bracket_low_ms": round(low, 2), "bracket_high_ms": round(high, 2),
                "bracket_is_bound_only": False, "step_ms": round(head_offset_ms, 2),
                "edge": "head", "evidence_class": "measured_absence",
                # WHERE THE CONSUMER'S `locate_match_onset` DUAL-BASELINE
                # SHOULD LOOK: a point comfortably inside the surviving
                # plateau (known to match), and master 0 (known absent by
                # this very evidence class). Computed here, not re-derived
                # downstream -- these are locator-internal quantities
                # (`head_run["first"]`) the chimeric side does not have.
                "known_match_ms": round(head_run["first"] * 1000.0, 2),
                "known_absent_ms": 0.0}
        else:
            # CASE 3, CONSTRUCTED GAP: a run before the head was dropped (no
            # offset prediction survives for this span at all), OR -- the
            # ANOMALY the artifact-gated rewrite exists to catch -- neither
            # a drop nor a negative offset explains why `master_start_ms` is
            # still > 0. Either way, no trustworthy offset covers [0,
            # head_run["first"]): search for where content starts matching
            # the plateau that DOES survive, instead of extrapolating a
            # number that was never measured there.
            anomalous = (not head_dropped_before) and head_offset_ms >= 0
            search_end_s = head_run["first"]
            onset_s, onset_discards, onset_probes, search_step_s = _search_edge_onset(
                master_path, reference_stream, candidate_path, primary_stream,
                0.0, search_end_s, head_offset_ms, work_dir, comparison_grid_hz,
                find="first")
            if onset_s is not None:
                onset_ms = onset_s * 1000.0
                search_step_ms = search_step_s * 1000.0
                # STEP-HONEST BRACKET (Architect's ruling, 2026-09-16,
                # replacing a defect this exact shape shipped): the true
                # onset provably lies between the last non-matching probe
                # and this confirmed matching one, no tighter than the
                # search's OWN step -- a quantum-scale window here
                # (`+/-2*quantum_ms`, the ORIGINAL form) fabricated
                # precision the search never measured. `low` is exact
                # (never fabricated below 0); `high` is the confirmed
                # onset itself, never widened past it.
                low = max(0.0, onset_ms - search_step_ms)
                high = onset_ms
                segments[0]["leading_bracket"] = {
                    "bracket_low_ms": round(low, 2), "bracket_high_ms": round(high, 2),
                    "bracket_is_bound_only": False, "step_ms": None,
                    "search_step_ms": round(search_step_ms, 2),
                    "edge": "head", "evidence_class": "constructed_gap",
                    "onset_anomalous": anomalous, "onset_discards": onset_discards,
                    "onset_probes": onset_probes,
                    "known_match_ms": round(head_run["first"] * 1000.0, 2),
                    "known_absent_ms": 0.0}
            else:
                # Onset not found: an HONEST, UNNARROWED bracket ships --
                # never a silent fill. bound_only=True so the chimeric-side
                # width gate still offers the whole span to the frame tier
                # rather than skipping it unexamined.
                segments[0]["leading_bracket"] = {
                    "bracket_low_ms": 0.0,
                    "bracket_high_ms": round(search_end_s * 1000.0, 2),
                    "bracket_is_bound_only": True, "step_ms": None,
                    "edge": "head", "evidence_class": "constructed_gap_onset_not_found",
                    "onset_anomalous": anomalous, "onset_discards": onset_discards,
                    "onset_probes": onset_probes,
                    "known_match_ms": round(head_run["first"] * 1000.0, 2),
                    "known_absent_ms": 0.0}
            if anomalous:
                _log(f"ANOMALY: head segment starts at "
                     f"{segments[0]['master_start_ms']} ms with neither a "
                     f"dropped run before it nor a negative offset to "
                     f"explain the gap -- named, not silently accepted")

        tail_position = segment_positions[-1]
        tail_run = runs[tail_position]
        tail_offset_ms = tail_run["mean"]
        tail_dropped_after = any(d["position"] > tail_position
                                 for d in dropped_segment_detail)
        tail_ends_at_master_end = (round(segments[-1]["master_end_ms"], 6)
                                   >= round(master_end_ms, 6))
        tail_predicted_end_ms = candidate_end_ms - tail_offset_ms
        if tail_ends_at_master_end:
            pass  # CASE 1: no tail gap -- nothing filled, nothing to bound.
        elif not tail_dropped_after and tail_predicted_end_ms <= master_end_ms:
            # CASE 2, MEASURED ABSENCE, mirrored: the candidate runs out
            # before the master does, at the boundary the plateau's own
            # offset and the two durations already predict.
            t_ms = tail_predicted_end_ms
            low = max(0.0, t_ms - 2 * quantum_ms)
            high = t_ms + 2 * quantum_ms
            segments[-1]["trailing_bracket"] = {
                "bracket_low_ms": round(low, 2), "bracket_high_ms": round(high, 2),
                "bracket_is_bound_only": False, "step_ms": round(tail_offset_ms, 2),
                "edge": "tail", "evidence_class": "measured_absence",
                "known_match_ms": round(tail_run["last"] * 1000.0, 2),
                "known_absent_ms": round(master_end_ms, 2)}
        else:
            # CASE 3, CONSTRUCTED GAP, mirrored: search FORWARD from the
            # last surviving plateau's own last supporting probe for the
            # LAST position that still matches it -- past that is the
            # evidenced divergence, exactly as undropped/negative-offset
            # would have predicted, had the run behind it survived.
            anomalous = (not tail_dropped_after) and tail_predicted_end_ms > master_end_ms
            search_start_s = tail_run["last"] + PROBE_WINDOW_SECONDS
            search_end_s = master_end_ms / 1000.0
            onset_s, onset_discards, onset_probes, search_step_s = _search_edge_onset(
                master_path, reference_stream, candidate_path, primary_stream,
                search_start_s, search_end_s, tail_offset_ms, work_dir,
                comparison_grid_hz, find="last")
            if onset_s is not None:
                onset_ms = onset_s * 1000.0
                search_step_ms = search_step_s * 1000.0
                # STEP-HONEST BRACKET, mirrored (see the head branch above
                # for the full ruling): content is CONFIRMED matching at
                # `onset_ms`; the divergence lies somewhere in the next
                # step, never claimed tighter than that.
                low = onset_ms
                high = min(master_end_ms, onset_ms + search_step_ms)
                segments[-1]["trailing_bracket"] = {
                    "bracket_low_ms": round(low, 2), "bracket_high_ms": round(high, 2),
                    "bracket_is_bound_only": False, "step_ms": None,
                    "search_step_ms": round(search_step_ms, 2),
                    "edge": "tail", "evidence_class": "constructed_gap",
                    "onset_anomalous": anomalous, "onset_discards": onset_discards,
                    "onset_probes": onset_probes,
                    "known_match_ms": round(tail_run["last"] * 1000.0, 2),
                    "known_absent_ms": round(master_end_ms, 2)}
            else:
                segments[-1]["trailing_bracket"] = {
                    "bracket_low_ms": round(search_start_s * 1000.0, 2),
                    "bracket_high_ms": round(master_end_ms, 2),
                    "bracket_is_bound_only": True, "step_ms": None,
                    "edge": "tail", "evidence_class": "constructed_gap_onset_not_found",
                    "onset_anomalous": anomalous, "onset_discards": onset_discards,
                    "onset_probes": onset_probes,
                    "known_match_ms": round(tail_run["last"] * 1000.0, 2),
                    "known_absent_ms": round(master_end_ms, 2)}
            if anomalous:
                _log(f"ANOMALY: tail segment ends at "
                     f"{segments[-1]['master_end_ms']} ms with neither a "
                     f"dropped run after it nor a predicted overrun to "
                     f"explain the gap -- named, not silently accepted")

    if not segments:
        _log("every segment unusable after clamping; declining")
        return _decline("no_usable_segments", "ran_conclusive_negative", pair=pair_id,
                        segments_kept=0,
                        segments_dropped=dropped_segments)
    if dropped_segments:
        _log(f"{dropped_segments} segment(s) dropped as unusable, {len(segments)} kept")
    # A change point is only meaningful between two segments that both survived.
    change_points = [cp for index, cp in enumerate(change_points)
                     if index in kept_runs and (index + 1) in kept_runs]

    # THE SUCCESS PATH WAS SILENT. Every decline reason above reaches the log and no
    # measurement did, so the module explained itself when it refused and said nothing
    # when it worked -- which is the wrong way round for anyone reconstructing a run
    # from artefacts. vmsam-ci found it by trying to run the cross-check the UNITS
    # section promises and having nothing on disk to run it against. `window_s` is on
    # the line because a quantum without its window is the exact error this module's
    # own docstring warns about: one physical step measured 500, 540 and 600 ms at
    # three window lengths, and `quantum=129` published alone invites the comparison
    # against a pipeline quantum measured over a different window.
    # THE FIELD BELOW WAS A LEAKED LOOP VARIABLE. `offset_ms` is assigned inside
    # `for position, run in enumerate(runs)` above and this emission is AFTER the loop,
    # so it held the LAST segment's mean and nothing said so. vmsam-ci built a
    # cross-check on it, described it correctly from five records as "the last
    # segment's base offset", and noted it coincided with the maximum magnitude only
    # because those five drift monotonically. On a non-monotone file -- id 108 reads
    # -950, -1454, -950 -- first and last are both -950 and the maximum is -1454.
    #
    # A LABEL IS NOT A MEASUREMENT. The name promised a quantity the code never chose.
    #
    # Schema agreed with vmsam-ci BEFORE the change, per WRITE_ZONES section 4: the
    # freedom to change an emitted field ends at the fields another agent's code
    # consumes. It asked for `offset_max_abs_ms` because that is what it actually
    # compares against the pipeline and neither first nor last is it on a non-monotone
    # file; and it kept `offset_monotone` as the field it would choose if forced to one,
    # because it reports whether the max-magnitude comparison is LEGITIMATE rather than
    # what it came out as.
    #
    # `offset_ms=` IS AN ALIAS FOR ONE IMAGE AND THEN GOES. ci's words: an alias that
    # never dies is a field meaning two things at once, and a later reader cannot tell
    # it was ever the accident rather than the intent. REMOVE IT once ci confirms its
    # parser reads `offset_last_ms`.
    # `segments` is non-empty here -- `if not segments: return None` above. With
    # exactly one segment the zip below is empty and `all([])` is True, so
    # `offset_monotone` reports true on a single segment. That is correct rather than
    # accidental: one segment is trivially monotone. Stated because an aggregate over
    # an empty set returns its identity element, and for a universal quantifier that
    # identity is the STRONGEST claim the function can make -- a census harness of mine
    # printed "32 of 32 monotone staircases" from `all([])` on lists that were empty
    # because of a key error. Both `all(...)` sites in this module are guarded by a
    # length floor (>= 3 probes at :760, >= 1 segment here); neither can fire empty.
    _seg_offsets = [sg["candidate_offset_ms"] for sg in segments]
    _monotone = (all(a <= b for a, b in zip(_seg_offsets, _seg_offsets[1:]))
                 or all(a >= b for a, b in zip(_seg_offsets, _seg_offsets[1:])))
    # PLAN-TIME PER-SEGMENT TABLE (`RULING_20260916_PLAN_ADMISSIBILITY_NOT_
    # TELEMETRY.MD`, Ruling 2's "ALSO" item, forensic's addition, adopted):
    # `offset_monotone` is a SCALAR verdict over the whole plan; the
    # consumer that actually needs to check admissibility (`merge_video_
    # repair.check_candidate_admissibility`) needs the EXACT per-boundary
    # facts, not a summary of them. Comma-joined into THIS `_emit` call
    # rather than a new per-segment call: segments are bounded and few
    # (2-5 observed, same population as `_gaps`/`bound_only` above), never
    # per-probe, so this does not touch `_emit`'s once-per-pair limit
    # (that limit protects a per-PROBE counter -- 30 probes x 315 files --
    # not a per-segment one).
    _seg_master_starts = [sg["master_start_ms"] for sg in segments]
    _seg_master_ends = [sg["master_end_ms"] for sg in segments]
    # GAP WIDTHS AND BOUND-ONLY COUNT, ADDED FOR vmsam-ci. `gap_start_ms` and
    # `gap_end_ms` are computed a few lines above and were LOGGED IN ZERO RECORDS --
    # ci's fourth instance tonight of computed-and-not-emitted in this module, after
    # `candidate_offset_points`, the quantum without its window, and the pairing
    # fidelity on the decline path.
    #
    # It could not run a test I designed because the quantity the test needs never
    # left the process, and it correctly refused to reconstruct `gap_end_ms` from
    # dev-2's plan -- that would be inferring my quantity from another agent's output.
    #
    # PURELY ADDITIVE, so no parser breaks: the fields ci already reads are untouched.
    # Once per pair, numbers only, and the count of change points is bounded and small
    # (2-5 observed) rather than per-probe, so the limit on `_emit` is respected.
    #
    # ci's live hypothesis needs exactly these two: a job whose change points are ALL
    # bound-only appears to lose a candidate piece, while a job with at least one
    # REFINED bracket does not. Perfect split on six jobs, which ci reports as p ~ 0.05
    # one-tailed rather than as a correlation -- a perfect split on six is the shape
    # that looks like a law and is a coin. The emission is what lets the count grow.
    _gaps = [int(round(cp["gap_end_ms"] - cp["gap_start_ms"]))
             for cp in change_points
             if cp.get("gap_start_ms") is not None and cp.get("gap_end_ms") is not None]
    _bound_only = sum(1 for cp in change_points if cp.get("bracket_is_bound_only"))
    # grid_hz: R23. vmsam-dev-4 checked all thirty job logs -- quantum, quantum_ms,
    # window_s and frame_rate are emitted and NOTHING NAMES AN AUDIO RATE OR GRID. So
    # nobody can tell from an artefact whether a measurement ran on a sub-44100 pair,
    # and two such pairs exist (ids 307 and 316, 32 kHz candidates against 48 kHz
    # masters). This is the quantity R20 changed: it was pinned at 44100 and is now
    # derived from the pair. A CHANGE NOBODY CAN SEE IN AN ARTEFACT CANNOT BE AUDITED
    # AFTER THE FACT, and this is the field that makes R23 checkable retrospectively.
    #
    # It was already computed and logged through `_log`, which is gated on tools.dev
    # and therefore absent from every production artefact -- the SIXTH
    # computed-and-not-emitted in this module. The value was there; the reader was not.
    _discard_list = [cp.get("refine_discards") for cp in change_points]
    _probe_list = [cp.get("refine_probes") for cp in change_points]
    _emit(f"grid_hz={comparison_grid_hz} "
          f"refine_probes={','.join(str(n) for n in _probe_list) if _probe_list else 'none'} "
          f"refine_discards={','.join(str(d) for d in _discard_list) if _discard_list else 'none'} "
          f"gaps_ms={','.join(str(g) for g in _gaps) if _gaps else 'none'} "
          f"bound_only={_bound_only}/{len(change_points)} "
          f"offset_first_ms={_seg_offsets[0]:.1f} offset_last_ms={_seg_offsets[-1]:.1f} "
          f"offset_max_abs_ms={max(_seg_offsets, key=abs):.1f} "
          f"offset_monotone={str(_monotone).lower()} "
          f"offset_ms={offset_ms:.1f} points={int(round(offset_ms / quantum_ms))} "
         f"quantum_ms={quantum_ms} window_s={PROBE_WINDOW_SECONDS} "
          # STAGE 1 SCREEN CENSUS FIELD (Architect's ruling, 2026-09-16): the
          # trigger bar for `offsets_saturated_at_search_bound` becomes a
          # measurement once a population exists; this is the SUCCESS-PATH
          # half of that population (a pair whose saturation, if any, did
          # not stop it from proceeding). The decline itself already carries
          # its own fields unconditionally via `_decline`; this is the other
          # half nothing was emitting. NOT YET carried by the OTHER decline
          # branches below (median_fidelity_below_floor and siblings) --
          # stated as a limit of this landing, not silently absent.
          f"saturation_kept={saturation_stats['probes_kept']} "
          f"saturation_saturated={saturation_stats['probes_saturated']} "
          f"saturation_observed_fraction={saturation_stats['observed_fraction']} "
         f"segments={len(segments)} change_points={len(change_points)} "
          # THE SEVEN BELOW ARE FOR A CONSUMER, NOT FOR THIS MODULE'S OWN DECISIONS.
          # `vmsam-forensic` holds 315 failure records and can census NEITHER the
          # low-signal population NOR the offset-scatter population, because
          # `probe`, `energy`, `silence` and `sign` appear in ZERO of them. The
          # module COMPUTES all of this, DECIDES on it, and threw it away -- the
          # same defect the refusal contract closed one level up, one level down.
          #
          # RAW COUNTS, NEVER THE VERDICT. The scatter guard is
          # `distinct_points > MAX_DISTINCT_POINTS AND flips > MAX_SIGN_FLIPS`;
          # emitting the counts lets a consumer census the DISTRIBUTION instead of
          # inheriting this module's thresholds, and a threshold in an artefact is
          # one that two modules then have to keep in step.
          #
          # AND THE STRUCTURAL BLANK, WHICH MUST NOT BE READ AS ZERO: a pair that
          # declines BEFORE the energy step emits none of these, because they are
          # not computed yet. That absence is a property of the pipeline, not a
          # gap in the emission.
          f"probe_energy_median={median_energy:.6g} "
          # AND ON THE SUCCESS LINE: probes lost on a run that SUCCEEDS are the attrition
          # arch-heir's order-statistic argument depends on being random. Silent until now.
          f"refusal_sites={','.join(f'{k}:{v}' for k, v in sorted(probe_sites.items())) or 'none'} "
          f"probes_attempted={len(starts)} "
            f"probes_raw={len(raw)} probes_kept={len(kept)} "
          f"probes_dropped_low_signal={dropped} "
          f"signal_floor_fraction={LOW_SIGNAL_FRACTION} "
          # Achieved-coverage fields on the SUCCESS line too, not only in the
          # returned plan: the two channels carry different fields and neither
          # is a superset (merge_plan_report.py:3469-3478 files plan-dict fields
          # as NO_PRODUCER reading only this channel -- correct about the log,
          # wrong about the module). Always 0/[] here: a non-empty value would
          # have declined above and never reached this line.
          f"coverage_gaps_n=0 coverage_gap_total_s=0.0 "
          # NOT `offset_distinct_points`. `points` IS ALREADY A UNIT OF TIME IN THIS
          # CODEBASE -- `audioCorrelation` returns `offset_in_points` and ONE POINT IS
          # 125 ms, the chromaprint frame -- and THIS VERY LINE prints
          # `points=offset_ms/quantum_ms` twelve fields earlier. `..._points=7` beside
          # `points=3` cannot be read: seven distinct offsets, or a spread of seven
          # chromaprint frames? Both are plausible on this line. `_count` costs three
          # characters and is the `codec_delay`/`start_time` collision this campaign has
          # already paid for once. Caught by `vmsam-forensic` reviewing the field names
          # BEFORE the first emission, which is the whole reason they were announced.
          f"offset_distinct_count={distinct_points} "
          f"offset_sign_flips={flips} "
          # THE JOIN IS MANY-TO-ONE AND THE CONSUMER FOUND IT BEFORE I DID. This line is
          # once per PAIR; `vmsam-forensic`'s census is keyed one row per ERROR ID, and
          # pairs-per-id varies. Without a pair key its reader takes whichever pair it saw
          # last, and A MEDIAN OVER PAIRS IS NOT A MEDIAN OVER IDS -- with nothing in the
          # field name to say which you computed.
          #
          # A DIGEST, NEVER THE PATHS: `WRITE_ZONES` section 8 -- carry the FACT of the
          # line, not its content. This is stable per pair, opaque, and keeps the
          # `carries_path: False` property this line was measured to have.
          #
          # I CANNOT EMIT THE ERROR ID: the locator is never told it. That is a real
          # limit of this boundary, not an omission, and closing it needs the CALLER to
          # pass one.
          f"pair={_digest(master_path, candidate_path)} "
          # THE TABLE ITSELF, comma-joined and index-aligned with each
          # other and with `segment_candidate_offset_ms` (identical to
          # `offset_first_ms`/`offset_last_ms`'s own source, `_seg_offsets`,
          # above -- not a second computation of the same value). ONE
          # boundary at a time is exactly what `check_candidate_
          # admissibility` needs and `offset_monotone` cannot give: it is
          # a wrong-signed proxy (Ruling 1) precisely because it collapses
          # this table into one bit.
          f"segment_master_start_ms={','.join(str(s) for s in _seg_master_starts)} "
          f"segment_master_end_ms={','.join(str(e) for e in _seg_master_ends)} "
          f"segment_candidate_offset_ms={','.join(str(o) for o in _seg_offsets)} "
          # BUILD IDENTITY AT EMISSION TIME, NEVER CAPTURED AT STARTUP (`AGENT.MD`): a
          # field written once at startup reports the same value whatever is running,
          # which is worse than an empty one -- an empty field is honest and a constant
          # one is a lie that survives every rebuild.
          #
          # AND IT IS WHAT MAKES THE STRUCTURAL BLANK READABLE: "never reached the energy
          # step" and "this artefact predates the emission" are THE SAME ABSENT FIELD
          # without it, and the outcome that goes missing is always THE INSTRUMENT DID NOT
          # RUN. Precedent, not a new shape: `merge_video_repair` already ships
          # `build <module>:<digest>`.
          f"build={_build_digest()}")

    # *** `(plan, cause)`: cause is None WHEN A PLAN IS RETURNED -- CAMPAIGN.MD 1153-1164.
    # The pair is returned from ONE place, as the plan is built in one place. ***
    plan = {"kind": "constant" if len(segments) == 1 else "piecewise_constant",
            "master_path": master_path,
            "candidate_path": candidate_path,
            "language": language,
            "reference_stream": reference_stream,
            "candidate_streams": candidate_streams,
            # ONE decision per candidate stream, made once for the file. A stream
            # ABSENT here cleared no master partner at the bar and its consumer must
            # REFUSE to place it rather than borrow another stream's offset.
            "candidate_stream_pairing": pairing,
            # Every pairing that was probed, accepted or not, with the fidelity the
            # bar was applied to -- so a bar sitting inside an overlap can be moved
            # by someone who disagrees with it.
            "candidate_stream_pairing_measurements": pairing_measurements,
            "pairing_min_fidelity": MIN_PAIRING_FIDELITY,
            "quantum_ms": quantum_ms,
            "probe_window_seconds": PROBE_WINDOW_SECONDS,
            "probe_step_seconds": PROBE_STEP_SECONDS,
            # STAGE 1 SCREEN CENSUS FIELDS, success-path half (see the
            # matching comment at the `_emit` line): NOT the decline's own
            # fields (those live on the `offsets_saturated_at_search_bound`
            # decline line, a different shape, deliberately not merged here
            # to avoid ARTEFACT_FORMATS.md SS9c's one-name-two-shapes defect).
            "saturation_probes_kept": saturation_stats["probes_kept"],
            "saturation_probes_saturated": saturation_stats["probes_saturated"],
            "saturation_observed_fraction": saturation_stats["observed_fraction"],
            "saturation_search_bound_points": saturation_stats["search_bound_points"],
            "saturation_threshold_fraction": saturation_stats["threshold_fraction"],
            # ACTUAL coverage, not the intent. vmsam-ci measured that the
            # pipeline's own geometry never samples a median 15.9 % of a file,
            # and that a file its geometry cannot see is not declined — it is
            # never presented, and it looks like a clean constant offset. This
            # module scans [0, shortest] contiguously including an end-anchored
            # tail probe, so its coverage is stated rather than assumed.
            #
            # THIS VALUE IS THE ATTEMPTED GRID, NOT WHAT WAS ACHIEVED -- unchanged
            # here by the Lead's correction to the Architect's grant, TASKS/013 §3,
            # because a live consumer (merge_plan_report.py) already parses it. A
            # reader who wants what actually came back wants the three fields
            # beside it instead: `coverage_measured`, `coverage_gaps_seconds`,
            # `coverage_gap_total_seconds`.
            "scanned_seconds": [round(min(starts), 3) if starts else None,
                                round(max(starts) + PROBE_WINDOW_SECONDS, 3) if starts else None],
            # The smallest step this scan can resolve. A divergence below it
            # reads as "constant" and is INVISIBLE, not absent — so a decline or
            # a constant verdict from this module carries this floor with it.
            "step_floor_ms": MIN_STEP_MS,
            # PLATEAU_TOLERANCE_MS was DEFINED AND NEVER RETURNED -- vmsam-dev-4 filed it
            # as NO_PRODUCER and it never left this module at all. Of the two constants that
            # decide whether a step survives, one reached the consumer unread and the other
            # did not reach it. A limit a consumer cannot see is not a limit it can respect.
            #
            # It is the EARLIER of the two gates: a run extends while the next probe stays
            # within this of the running mean, so a step smaller than the tolerance is
            # absorbed BEFORE step_floor_ms is ever consulted. Emitting only step_floor_ms
            # understated the floor.
            "plateau_tolerance_ms": PLATEAU_TOLERANCE_MS,
            # The EFFECTIVE floor a consumer should reason with: a step must clear the
            # plateau tolerance to become a separate run at all, and then clear the step
            # floor to survive merging. Stated so nobody has to re-derive the interaction.
            "effective_step_floor_ms": max(PLATEAU_TOLERANCE_MS, MIN_STEP_MS),
            "probes_used": len(kept),
            # ATTEMPTED reached the log line and never the plan -- the two
            # subtractions are the only way to tell WHY a probe is missing
            # (attempted-raw = EXTRACTION FAILURE, raw-kept = ENERGY GUARD,
            # :1512) and a consumer holding only the plan could compute neither.
            "probes_attempted": len(starts),
            "probes_raw": len(raw),
            "probes_dropped_low_signal": dropped,
            # What `scanned_seconds` actually is, named rather than left to its
            # name: the ATTEMPTED grid's span, unchanged by this landing at the
            # Lead's correction -- a live consumer (merge_plan_report.py)
            # already reads it and its meaning does not move.
            "scanned_seconds_basis": "attempted_probe_grid",
            # What was ACHIEVED inside that span -- the coverage_incomplete
            # guard's own predicate. Always empty/0.0 here BY CONSTRUCTION: a
            # non-empty value declines above and this line is never reached in
            # that case. `coverage_measured=True` says the instrument RAN, so
            # this absence is a measurement, never a default silently agreeing.
            "coverage_measured": True,
            "coverage_gaps_seconds": [],
            "coverage_gap_total_seconds": 0.0,
            "segments": segments,
            "change_points": change_points,
            "median_fidelity": round(median_fidelity, 4),
            # "constant" means NO STEP WAS SEEN by a scan that covered
            # [0, shortest] at PROBE_WINDOW_SECONDS resolution and cannot resolve
            # a step below MIN_STEP_MS. It is NOT a warrant for applying this
            # offset as a container delay.
            # THE MASTER REFERENCE STREAM'S OWN start_time. Emitted because it
            # predicts a real failure and not because of any framing question --
            # a consumer that rebuilds a track starting at PTS 0 is misaligned from
            # the master's track by exactly this, and vmsam-dev-2's four release-32
            # declines match it to under 5 ms across three distinct values:
            # 1103.4 vs 1103.0, 1103.8 vs 1103.0, 1059.4 vs 1055.0, 887.6 vs 887.0.
            #
            # An `offset_reference` key and the master-minus-candidate difference
            # were emitted here for twenty minutes and are gone: dev-2 tested PTS
            # seek against the `atrim` its assembler uses and got 0.0 ms apart on a
            # stream with a 120 ms start_time, so there was no second frame and no
            # conversion to name. A key naming a distinction that does not exist is
            # a second thing to keep true and a second thing to get wrong.
            "master_reference_start_time_ms": reference_start_ms,
            "segments_dropped_unusable": dropped_segments,
            "segments_dropped_unusable_detail": dropped_segment_detail,
            # Surfaced at the top level so a consumer does not have to scan the
            # segment list to discover that part of the plan is unverified.
            "segments_offset_unverified": sum(1 for seg in segments
                                              if seg["offset_unverified"]),
            "constant_floor_ms": MIN_STEP_MS if len(segments) == 1 else None}
    return (plan, None)
