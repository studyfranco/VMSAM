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
    STUB     step 3e the comparison resample           -- declines `comparison_resample_
                                                          not_implemented`
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
# CALIBRATED ON n = 1 PAIR WITH 1 REAL EVENT, AND THAT IS NOT ENOUGH. errid-24, errid-70,
# errid-84 and errid-99 are available in the corpus and were not run. This constant is honest
# about its own sample size rather than presented as settled.
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
# section 6.4.1): zone fragmentation varies enormously by track -- 82/77/61/27/32 aligned zones
# for the SAME physical pair on errid-232 -- so a design that resolves every hole would launch
# dozens of frame-exact searches per couple, each of them an unbounded ffmpeg decode. The <10 s
# merge is the first line of defence; this is the second.
#
# THE NUMBER IS SET FROM MEASUREMENT, AND IT TOOK TWO CORRECTIONS TO GET A HONEST POPULATION TO
# MEASURE. The design suggests this campaign's precedent for the same shape
# (`zone_similarity_vector.MAX_ONSET_CANDIDATES_PER_WINDOW = 10`), and 10 was tried first; it
# refused errid-202 at 11 holes and sat exactly on the boundary for another couple. The reason
# was not the cap: it was that same-offset coverage gaps were being counted as holes (see
# `coalesce_same_offset_zones`, where errid-232 jpn couple 13x1 read 40 holes of which 38 had a
# step of exactly 0). MEASURED post-merge hole counts once a hole means a change of offset:
#     errid-202 jpn  m#3xc#1    6     errid-232 jpn  m#13xc#1   4
#     errid-232 eng  m#3xc#2    4     errid-232 jpn  m#14xc#1   9
#     errid-232 eng  m#4xc#2    6     errid-232 jpn  m#15xc#1   3
# Largest observed: 9. Set to well over twice that, so a pair must fragment past anything yet
# observed before it is refused -- this stays a guard against the pathological and never
# becomes a filter on the ordinary. SAMPLE: 6 couples on 2 pairs. Under-sampled, stated as
# such, and re-measurable the moment more pairs are driven.
MAX_HOLES_PER_COUPLE = 22

# THE RATE-RELATION ARM OF THE STEP-2 GATE IS NOT CALIBRATED, AND IT IS OFF BECAUSE OF THAT.
# The design's section 3.6 recommends reading `drift_fit` (slope + R^2) as "there is a rate
# relation here" and quotes errid-202 at slope -0.00078 / r^2 0.752. MEASURED THIS SESSION on
# the two pairs available, both of which are CONTENT-EDIT pairs with no rate relation:
#     errid-202 jpn    slope -0.00237   r^2 0.790   implied_step_count 78
#     errid-232 jpn#0  slope -0.00025   r^2 0.709   implied_step_count 3
#     errid-232 eng#0  slope -0.00025   r^2 0.718   implied_step_count 3
# i.e. a threshold on slope-with-high-R^2 would fire on pairs that have no rate relation at all,
# and there is no known-rate pair in this corpus checkout to calibrate the other side of the
# boundary against. A threshold picked from one side of a boundary is a guess wearing a number.
# So the numbers are MEASURED AND LOGGED on every run and NOTHING BRANCHES ON THEM; the gate
# fires only on the arm that is unambiguous (the aligner could not align at all). Flip this to
# True in the same commit that lands the calibration, never before.
RATE_RELATION_SLOPE_GATE_CALIBRATED = False

# The aligner's own "I could not measure" vocabulary -- the step-2 gate's one calibrated arm.
# These are `banded_seed_alignment`'s tokens, read from the module rather than restated here.
ALIGNMENT_COULD_NOT_MEASURE_VERDICTS = (
    "unreliable_degenerate_input",
    "no_seeds_found",
    "no_anchored_runs",
    "all_seeds_refused_by_local_baseline_guard",
)

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
    "intercouple_disagreement": CLASS_CONCLUSIVE,
    "hole_count_exceeds_resolver_budget": CLASS_CONCLUSIVE,
    # the stubs -- honest declines for stages not yet landed
    "comparison_resample_not_implemented": CLASS_COULD_NOT_RUN,
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
                      duration_seconds):
    """One whole-file fingerprint list for ONE track. Returns `(points, quantum_ms)`, or
    `(None, None)` when the track could not be read.

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
    wav = path.join(work_dir, f"orch_{side}_{stream_order}.wav")
    try:
        audio_extract.extract_audio_window(video_obj.filePath, stream_order, 0.0,
                                           duration_seconds, wav, sample_rate)
        points = audioCorrelation.calculate_fingerprints(wav, length=duration_seconds)
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
    return points, duration_seconds * 1000.0 / len(points)


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


def merge_holes(holes, zones, quantum_ms):
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
    """
    if not holes:
        return []
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
    merged = merge_holes(raw, zones, quantum_ms)
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

def comparison_resample(speed_factor, master_obj, candidate_obj, language, work_dir):
    """STUB -- design stage 3. Returns `(None, cause)`.

    WHAT IT WILL DO. Speed correction is a MEASUREMENT tool (owner's ADDENDUM 7 point 1): the
    sweep and this resample serve ALIGNING and LOCATING, nothing else. The filter is chosen BY
    THE MEASUREMENT ALREADY AVAILABLE and never hard-coded -- the pitch layer says whether the
    pitch moved, and ADDENDUM 2 settled the routing by bench: the EXISTING routing stays
    (`asetrate` behind the pitch-shifted verdict, which is what `merge_video_resample
    .build_speed_filter_chain` already builds), and the pitch layer's currently-unimplemented
    "inverting case" token binds to `atempo` when it is implemented -- never to `rubberband`,
    which lost in both measured classes.

    AND NO FILTER RUNS AT ALL AT speed_factor = 1 (owner's ADDENDUM 6): a track whose speed did
    not change passes through INTACT, zero processing. A filter exists only to UNDO a measured
    change. The caller enforces that before ever reaching here.
    """
    return None, "comparison_resample_not_implemented"


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


def similarity_gate(alignment):
    """"Similarite faible master<->candidat ?" -- answered from the alignment's OWN numbers.

    The design's section 3.6 weighs two orderings and recommends (B), align first and gate on
    what the aligner says, because the alignment costs ~4 % of the extraction both orderings
    already pay and because it yields a NAMED reason ("the aligner found no anchored run")
    instead of a bare scalar.

    ONE ARM FIRES AND IT IS THE UNAMBIGUOUS ONE: the aligner returned one of its own
    could-not-measure verdicts. That IS "similarity is low" in the ruling's sense -- there was
    not enough agreement anywhere in the file to anchor a single run.

    THE RATE-RELATION ARM IS MEASURED AND NOT BRANCHED ON -- see
    `RATE_RELATION_SLOPE_GATE_CALIBRATED` for the measurements that refuse to support a
    threshold yet. Its numbers are returned so the caller logs them on every run, which is how
    the calibration gets its data instead of waiting for someone to go and collect it.

    Returns `(should_sweep, reason, observations)`.
    """
    drift_fit = alignment.get("drift_fit") or {}
    observations = {
        "verdict": alignment.get("verdict"),
        "coverage": alignment.get("master_axis_coverage_fraction"),
        "residual_fraction": alignment.get("residual_fraction"),
        "slope_points_per_point": drift_fit.get("slope_points_per_point"),
        "r_squared": drift_fit.get("r_squared"),
        "implied_step_count": drift_fit.get("implied_step_count"),
        "rate_arm_calibrated": RATE_RELATION_SLOPE_GATE_CALIBRATED,
    }
    if alignment.get("verdict") in ALIGNMENT_COULD_NOT_MEASURE_VERDICTS:
        return True, f"the aligner returned {alignment['verdict']}", observations
    return False, (f"the aligner anchored runs ({alignment.get('verdict')}), so similarity is "
                   f"not low in the ruling's sense; the rate-relation arm is measured and "
                   f"NOT branched on (uncalibrated)"), observations


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
    couple, so the gate does not cost a second whole alignment of the same tracks.
    """
    candidate_path = candidate_obj.filePath
    if factor is None or factor == 1:
        step_result("no_filter", candidate=candidate_path, speed_factor=factor,
                    rule="ADDENDUM_6_no_filter_without_speed_change")
    else:
        step_launch("comparison_resample", candidate=candidate_path, speed_factor=factor)
        resampled, cause = comparison_resample(factor, master_obj, candidate_obj, language,
                                               work_dir)
        step_result("comparison_resample", candidate=candidate_path, ok=False, cause=cause)
        if resampled is None:
            return False, cause, (
                f"the pair carries a confirmed rate relation ({factor}) and the comparison "
                f"resample that would let the aligner measure across it is design stage 3 and "
                f"is not built yet -- speed correction here is a MEASUREMENT tool only "
                f"(ADDENDUM 7), and no measurement was made"), None

    couples = enumerate_couples(master_obj, candidate_obj, language)
    step_result("enumerate_couples", candidate=candidate_path, language=language,
                n_couples=len(couples), couples=couples)
    if not couples:
        return False, "no_stream_for_comparison_language", (
            f"neither side offers a pair of {language} audio streams to compare"), None

    sample_rate = comparison_sample_rate(master_obj, candidate_obj, language)
    fingerprints = dict(primed_alignments.get("fingerprints", {})
                        if primed_alignments else {})
    alignments = dict(primed_alignments.get("alignments", {}) if primed_alignments else {})

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
            step_launch("fingerprint", candidate=candidate_path, side=side, stream=stream,
                        duration_s=round(duration, 3), sample_rate=sample_rate)
            started = time.time()
            points, quantum_ms = fingerprint_track(video_obj, language, stream, side,
                                                   work_dir, sample_rate, duration)
            step_result("fingerprint", candidate=candidate_path, side=side, stream=stream,
                        n_points=len(points) if points else 0,
                        quantum_ms=round(quantum_ms, 4) if quantum_ms else None,
                        seconds=round(time.time() - started, 2))
            if points is None:
                return False, "fingerprinting_raised", (
                    f"the {side} {language} stream {stream} could not be extracted or "
                    f"fingerprinted"), None
            fingerprints[key] = (points, quantum_ms, duration)

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
        # EVERY couple was blind. That is a property of the pair, and it gets the aligner's own
        # token rather than a generic one, so the ledger can tell the three apart.
        verdicts = {alignments[f"{m}x{c}"]["verdict"] for m, c in couples
                    if f"{m}x{c}" in alignments}
        cause = ("alignment_degenerate_input" if "unreliable_degenerate_input" in verdicts
                 else "alignment_all_seeds_refused"
                 if "all_seeds_refused_by_local_baseline_guard" in verdicts
                 else "alignment_no_anchored_runs")
        return False, cause, (
            f"no couple of {language} could be aligned; the aligner reported "
            f"{sorted(verdicts)} across {len(couples)} couples"), None

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
        if winner is None:
            _plan_line("none", candidate_path, step="speed_sweep", cause=sweep_cause)
            return _terminal(
                candidate_path, "no_plan", sweep_cause,
                f"mean similarity is low ({gate_prose}) and the rate sweep could not raise it "
                f"({(sweep_gate or {}).get('cause')})", detail={"resample_gate": sweep_gate})
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
