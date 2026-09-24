"""
Candidate B2 — banded k-mer seed-and-extend alignment, ported to src/ as a standalone open
module (owner order, 2026-09-21: "B2 est valide! On impose un fingerprint de tout le fichier!
Et on fait un alignement adn." — B2_ADOPTED, production integration ordered). Ported from
`VMSAM_HELP_AI/prover-stage1/lab/candidate_b2.py`, whose bench results
(`b2_results_constructed.md`) are the acceptance record this build answers to.

WIRED (2026-09-24): `repair_orchestrator` is this module's caller -- `b2_align` is the
orchestrator's alignment function (RULING_20260922_ORCHESTRATOR_ARCHITECTURE.MD, "la fonction
d'alignement de sequences"), and since the switch the orchestrator is THE repair chain. The
standalone `locate_zones_by_alignment` wrapper that existed for pre-wiring acceptance runs was
removed with the switch: it had no caller, and the orchestrator extracts each track to its own
full duration instead (see `repair_orchestrator.fingerprint_track`).

DNA-style k-mer WORD SEEDING (BLAST-style) on a REDUCED alphabet (top `B` bits of each 32-bit
chromaprint value), seed-and-extend with Hamming-tolerant extension on FULL 32-bit values,
banded by the plateau step function when the caller supplies one. WHOLE-FILE input, no
pre-windowing -- this is what lets it see a step the coarse per-window probe grid's own
geometry can miss (LOCATOR_BAN_SCOPE remains respected: seeding restricted to a caller-supplied
band is banded probing inside an already-established region, never a fresh global correlation
search for cut-finding).

ACCEPTANCE CONTRACT, three constraints entered before this was built, all already
measured/motivated in the prover's own bake-off, not guessed here:

1. B2 IS THE COARSE STAGE. Its own declared resolution floor is ~= k * quantum_ms (k=2,
   quantum ~124-142ms -> ~250-280ms). It localises to a BAND with per-segment offsets; frame-
   accurate refinement is the frame-anchor stage's job, never this module's. Every emitted
   zone/segment says so explicitly (`stage_contract`), and the reported band WIDTH is NOT the
   same quantity as this resolution floor -- see the module-level note under `MODALITY` below,
   the exact confusion this file exists to prevent from happening twice in one campaign.
2. LOCAL-BASELINE GUARD. A seed/extension match counts as evidence for a segment only if the
   region FLANKING it also matches well. Motivated directly by a same-night failure mode this
   campaign hit on real material (a wrong crossing that was the DEEPEST dip in its window but
   sat inside a broad degraded region at local baseline ~0.53 against a typical ~0.93-0.97; the
   correct crossing was shallower but the SURROUNDING region was clean). A Hamming tolerance
   loose enough for real noise seeds inside degraded regions too -- the same depth-vs-shape
   error, reborn in seed-space, is exactly what this guard exists to refuse structurally rather
   than by tuning a threshold after the fact.
3. DEGENERACY SCREEN, entry gate, before any seeding. Seeding is MORE exposed to this than a
   single scalar threshold is: flat/silent content matches everywhere, densely, confidently, and
   a screen that only ran for earlier candidates in this campaign's bake-off is not enough here.
   Reimplemented fresh (not imported): `audioCorrelation.py` is frozen and this is a fresh open
   module; `zone_similarity_vector.degeneracy_report` is another seat's open module, read for its
   logic, not imported across module ownership boundaries.

   MEASURED, 2026-09-21, not assumed either direction: the gate fires on `<= 1` distinct
   fingerprint value. Checked directly against real, genuine near-silence (a real -91dB, codec-
   noise-floor tail region, 37s of real content) AND against a pure digital-silence fixture
   (200s): BOTH read `n_distinct=1` -- the same single fingerprint value repeated throughout,
   with zero measured exceptions. So this IS a working, measured degeneracy gate for content at
   the codec noise floor, n=2 sources so far, not merely a mathematical limit case that happens
   never to fire. What remains genuinely UNTESTED, stated as such rather than assumed either
   way: LOW-LEVEL AMBIENT OR HUM CONTENT WITH REAL, NONZERO VARIATION -- quiet but not pinned at
   the hard floor, which could in principle produce a handful of distinct values (two, three,
   five) across thousands of points and pass this gate while still being the kind of degenerate-
   ish content the gate exists to catch. Nobody has yet found such a fixture in this corpus
   despite looking; if a real 100+-file sweep ever turns one up (this module's own `n_distinct`/
   `n_items` fields make that free to watch for), re-open this note rather than trusting the
   `<=1` boundary to generalise from n=2.

k, B: MEASURED from a k x B match-rate table over two real error-tree files (not guessed).
`k=2, B=12`. B=32 (no reduction) gives near-zero k>=2 seed rates even on a clean pair (0.008 on
one measured file) -- seeding on raw 32-bit values finds almost nothing to seed from. B=8 gives
high hit rates (0.45 on the same file at k=2) but the false-seed risk is worst there (most bits
discarded, most collisions by chance) -- the mandatory minimum-run anchor below exists for
exactly this reason. B=12 is the first reduction level where a clean file's k=2 rate (0.247)
still clears a usable seed density while B=16/B=32 do not (0.121/0.008); a second, independent
file's k=2/B=12 rate landed close (0.240), a workable middle rather than an overfit optimum, and
still re-measurable per file rather than assumed universal.

MODALITY (new law, 2026-09-21: "a measurement carries its modality"): every emitted
segment/zone/boundary below carries `"modality"` naming THIS instrument by a literal string a
census can key on without inferring which module produced a record from context. Distinct from
the coarse per-window probe/plateau instrument (`change_point_locator`'s own `_probe`/
`_group_plateaus`) and from the single-point onset/confirm vector (`zone_similarity_vector`) --
three different measurements over the same underlying chromaprint fingerprints, none of which
may be silently substituted for another's claim.

THE BAND-WIDTH VS RESOLUTION-FLOOR DISTINCTION, MEASURED, NOT ASSUMED (the prover's own
bake-off, answering a question raised the same night by a DIFFERENT instrument's 39-second
"containment" success that turned out to be a magnitude match, not a location one -- do not
conflate the two failure classes, they are adjacent, not identical): on four constructed
fixtures with exact-by-construction truth, reported zone widths measured 6.3-7.2s (mean 6.75s)
-- NOT the ~250ms resolution floor, and NOT tens of seconds either. The zone width is the gap
between the LAST matched point before a cut and the FIRST matched point after it, which
necessarily spans the true removed/added duration PLUS a small extension-slack margin where the
Hamming-tolerant extension does not stop instantly at the true edge (measured mean excess over
the true removed span: ~1.75s total, split across both edges). The true cut position sits close
to the NEAR edge of the band in every measured case (mean nearest-edge delta 0.9s, range
0.2-1.9s across 7 real cut events on 4 fixtures) -- it is not centred slop, it is "true span,
plus edge slack," and the truth has been inside it every time measured so far (7/7). Callers
must read `edge_slack_note` on each zone rather than treating `zone_master_time_bounds_s`'s own
width as either the resolution floor or a plausible-only-once-refined margin.
"""
import math
import statistics

# THE COMPLETE VERDICT VOCABULARY, AS CONSTANTS, BECAUSE A CONSUMER HAS TO BRANCH ON IT.
# Added 2026-09-22 on an independent tester's finding: `repair_orchestrator`'s step-2 gate
# carried the four could-not-measure tokens as HARDCODED STRING LITERALS under a comment
# claiming they were "read from the module", and this module exported nothing for them to read.
# A renamed verdict here would have silently stopped that gate from firing, with no error
# anywhere -- which is precisely the failure the comment claimed to have avoided. So the tokens
# live here, once, the assignments below use these names, and the consumer imports the TUPLE.
#
# THE SPLIT IS THE PART THAT MATTERS. The first five mean "I could not measure"; the last two
# are measurements. A caller that branches on the wrong half of this line reads a blank as a
# negative, which is the defect class this campaign keeps finding.
VERDICT_DEGENERATE_INPUT = "unreliable_degenerate_input"
VERDICT_NO_SEEDS_FOUND = "no_seeds_found"
VERDICT_NO_ANCHORED_RUNS = "no_anchored_runs"
VERDICT_ALL_SEEDS_REFUSED = "all_seeds_refused_by_local_baseline_guard"
VERDICT_ALL_SEGMENTS_BELOW_DURATION_FLOOR = "all_segments_below_duration_floor"
VERDICT_SINGLE_SEGMENT_NO_CUT = "single_segment_no_cut"
VERDICT_SEGMENTS_FOUND = "segments_found"

COULD_NOT_MEASURE_VERDICTS = (
    VERDICT_DEGENERATE_INPUT,
    VERDICT_NO_SEEDS_FOUND,
    VERDICT_NO_ANCHORED_RUNS,
    VERDICT_ALL_SEEDS_REFUSED,
    VERDICT_ALL_SEGMENTS_BELOW_DURATION_FLOOR,
)
MEASURED_VERDICTS = (VERDICT_SINGLE_SEGMENT_NO_CUT, VERDICT_SEGMENTS_FOUND)

# *** AND A MEASURED VERDICT IS NOT THE SAME AS A USABLE ONE. `VERDICT_SINGLE_SEGMENT_NO_CUT`
# says "no offset STEP was found"; it says nothing about how much of the master axis any zone
# covered. The same tester measured a confirmed PAL pair returning it at
# `master_axis_coverage_fraction` 0.0, and two wrong-episode pairs returning it at 0.064/0.067.
# A consumer must read the COVERAGE beside the verdict, never the verdict alone -- that is what
# `master_axis_coverage_fraction` is reported on every result for. ***

MODALITY = "audio_kmer_seed_b2"  # literal, stable, cited by name -- see the module docstring's
                                   # own MODALITY section. Never re-derive this string at a call
                                   # site; import and use the constant so a rename cannot drift
                                   # silently between producer and any future consumer.

K = 2
B = 12
MIN_RUN_POINTS = 5   # mandatory anchor: consecutive matching/extending points required before a
                       # seed is trusted as founding a segment. Chance-baseline similarity
                       # (~0.5) can align noise on its own, and a lone matching k-mer proves
                       # nothing by itself -- this is the same discipline this campaign's other
                       # instruments already carry under their own names (a rupture verdict
                       # pairs with a degeneracy check; a match is evidence only if local content
                       # could refute it).
HAMMING_MATCH_THRESHOLD = 0.85  # per-point similarity floor for "still matching" during
                                  # extension, evaluated on FULL 32-bit values only -- the
                                  # reduced alphabet (`B` bits) is for FINDING seeds cheaply, not
                                  # for judging whether an extension is real.
LOCAL_BASELINE_WINDOW = 15       # points, each side, for the local-baseline guard below.
LOCAL_BASELINE_MIN = 0.75        # a segment's flanking region must average at least this to be
                                   # trusted, whatever its own extension measured. Set below a
                                   # clean plateau's typical measured range (~0.93-0.97) but well
                                   # above a measured degraded region's own reading (~0.53), so it
                                   # actually discriminates rather than passing everything or
                                   # nothing -- re-verify per corpus rather than assume this
                                   # margin transfers to material this was not measured against.
LOCAL_BASELINE_SELF_EVIDENT_SECONDS = 30.0  # a run at least this long, over non-degenerate
                                   # content, is its OWN local region and is admitted without its
                                   # flanks. WHY THE FLANKS CANNOT JUDGE IT: `extend_seed` grows a
                                   # run until similarity drops, so on BIT-IDENTICAL audio (a
                                   # same-source release, typically at offset 0) the run covers the
                                   # WHOLE matching region and its flanks are, by construction, the
                                   # far side of the cut or the file edge. MEASURED 2026-09-24 on the
                                   # stage-4 constructed fixture: a 1930-point (240 s) identical
                                   # half at offset 0 read flank baseline 0.598 (left flank empty,
                                   # right flank = post-cut) and was REFUSED -- the half went
                                   # unaligned. Re-encoded pairs never showed it because noise
                                   # breaks their runs into pieces whose flanks still match. 30 s
                                   # is ~240 points: two orders of magnitude past the guard's own
                                   # 15-point window, and longer than any in-file exact repeat the
                                   # guard exists to refuse at a wrong offset (a shared music cue
                                   # carries different dialogue/effects on top and does not stay
                                   # >= 0.85 for 30 s).
LOCAL_BASELINE_SELF_EVIDENT_MIN_DISTINCT_FRACTION = 0.5  # ...and "non-degenerate" means at
                                   # least half of the run's master values are distinct. Silence
                                   # and flat tone repeat ONE value (measured on the same fixture:
                                   # a 29-point silent run at offset 0 carries 1 distinct value) and
                                   # match at every offset, which is exactly the trap the flank
                                   # guard is for; real programme audio reads ~99 % distinct
                                   # (1921 of 1930 on the fixture's identical half).
OFFSET_MERGE_TOLERANCE_POINTS = 3  # merge two extended runs into one segment when their offsets
                                     # agree within this many points (~one quantum of ordinary
                                     # per-seed measurement jitter), not only on an exact match --
                                     # an exact-offset merge was measured splitting one real
                                     # transition into dozens of "segments" from ordinary jitter
                                     # alone on a real file (neighbouring seeds measuring the SAME
                                     # transition landed at four offsets within +/-250ms of each
                                     # other and were treated as distinct without this).
MIN_SEGMENT_DURATION_S = 2.0    # a segment covering less than this is more likely a measurement-
                                   # noise fragment sitting at a real segment's own boundary than
                                   # a genuine extra segment -- filtered, and the filtering COUNT
                                   # is reported (`segments_filtered_short_fragments`) rather than
                                   # a silent drop.
RESOLUTION_FLOOR_QUANTA = 2     # a step whose magnitude is under this many quanta is classified
                                   # `below_resolution_floor` rather than `cut` -- NEVER deleted,
                                   # see `all_zones` vs `cut_zones` below. Chosen to match this
                                   # module's own declared resolution floor (k * quantum), not an
                                   # independently-tuned threshold.

TRACE_X_POINTS = 10   # owner's re-centering check interval, points. PAL-rate drift moves the
                        # best shift ~1 point per ~23 points (~3s at this quantum) -- a checkpoint
                        # spacing at or under 10-12 points keeps at least 2 checkpoints inside
                        # that span, so a drifting band cannot walk out between checks. NTSC-rate
                        # drift (~1 point per ~1000 points, ~2.15min) is comfortably sampled at
                        # this same cadence too -- one cadence, both rates covered, measured not
                        # separately tuned per rate family.
TRACE_M_POINTS = 3    # re-centering search radius around the CURRENT offset, points -- wide
                        # enough to catch one PAL-rate step per checkpoint (expected ~0.4 points)
                        # with margin, without searching so wide that a real cut's own jump gets
                        # mistaken for a sequence of small re-centers.
TRACE_MIN_SUSTAINED_WORDS = 3   # a re-center only wins if it improves the checkpoint's own
                                  # matched-word count by at least this many words, never on one
                                  # lucky point -- the owner's own floor, applied to the re-center
                                  # decision so noise is never chased.


def _popcount32(value):
    return bin(value & 0xFFFFFFFF).count("1")


def sim(a, b):
    """Per-point similarity on FULL 32-bit values -- the same primitive every fingerprint-
    comparing instrument in this campaign reimplements locally rather than importing (the frozen
    module this reduces to, `audioCorrelation.py`, cannot be touched; the sibling open module,
    `zone_similarity_vector.py`, is another seat's, read for its shape, not imported across
    ownership)."""
    return (32 - _popcount32(a ^ b)) / 32.0


def _reduce(value, bits=B):
    """Top `bits` bits of a 32-bit fingerprint value -- the REDUCED alphabet used only for
    finding seed candidates cheaply. Real similarity judgement (`sim`, above) always runs on the
    full, unreduced value; conflating the two was never the intent and is guarded structurally by
    never passing a reduced value into `sim`."""
    shift = 32 - bits
    return (value & 0xFFFFFFFF) >> shift if shift else (value & 0xFFFFFFFF)


def degeneracy_report(fingerprint_a, fingerprint_b):
    """DEGENERACY SCREEN, entry gate -- see the module docstring's acceptance condition 3. A
    side is degenerate if it carries at most one distinct value: flat/silent content matches
    everywhere, densely and confidently, which is the opposite of what a seed's presence is
    supposed to mean."""
    distinct_a = len(set(fingerprint_a))
    distinct_b = len(set(fingerprint_b))
    return {
        "n_items_a": len(fingerprint_a), "n_items_b": len(fingerprint_b),
        "n_distinct_a": distinct_a, "n_distinct_b": distinct_b,
        "degenerate_a": distinct_a <= 1 and len(fingerprint_a) > 0,
        "degenerate_b": distinct_b <= 1 and len(fingerprint_b) > 0,
    }


def build_kmer_index(fingerprint, k=K, bits=B):
    """Index: reduced k-mer tuple -> list of starting positions in `fingerprint`."""
    reduced = [_reduce(x, bits) for x in fingerprint]
    index = {}
    for i in range(len(reduced) - k + 1):
        key = tuple(reduced[i:i + k])
        index.setdefault(key, []).append(i)
    return index


def find_seeds(fp_master, fp_candidate, band=None, k=K, bits=B):
    """`band`: optional `(offset_lo_points, offset_hi_points)` restricting seed acceptance to
    `candidate_index - master_index` within that range, when the caller already has the plateau
    step function's own reach to supply. LOCATOR_BAN_SCOPE-compliant: this bands a probe inside a
    region already established elsewhere; it is never a fresh global correlation search mounted
    for cut-finding."""
    master_index = build_kmer_index(fp_master, k, bits)
    reduced_candidate = [_reduce(x, bits) for x in fp_candidate]
    seeds = []
    for j in range(len(reduced_candidate) - k + 1):
        key = tuple(reduced_candidate[j:j + k])
        for i in master_index.get(key, ()):
            offset = j - i
            if band is not None and not (band[0] <= offset <= band[1]):
                continue
            seeds.append((i, j))
    return seeds


def extend_seed(fp_master, fp_candidate, i, j, threshold=HAMMING_MATCH_THRESHOLD):
    """Extend a seed (master index `i`, candidate index `j`) left and right on FULL 32-bit values
    while per-point similarity stays `>= threshold`. Returns `(i_lo, i_hi, j_lo, j_hi, mean_sim)`
    covering the matched run, inclusive bounds."""
    i_lo = i_hi = i
    j_lo = j_hi = j
    while (i_lo > 0 and j_lo > 0
           and sim(fp_master[i_lo - 1], fp_candidate[j_lo - 1]) >= threshold):
        i_lo -= 1
        j_lo -= 1
    n_master, n_candidate = len(fp_master), len(fp_candidate)
    while (i_hi < n_master - 1 and j_hi < n_candidate - 1
           and sim(fp_master[i_hi + 1], fp_candidate[j_hi + 1]) >= threshold):
        i_hi += 1
        j_hi += 1
    offset = j - i  # this call's own offset, computed locally from the (i, j) it was handed
                      # rather than threaded in from the caller.
    sims = [sim(fp_master[x], fp_candidate[x + offset]) for x in range(i_lo, i_hi + 1)]
    return i_lo, i_hi, j_lo, j_hi, (sum(sims) / len(sims) if sims else 0.0)


def local_baseline(fp_master, fp_candidate, i_lo, i_hi, offset, window=LOCAL_BASELINE_WINDOW):
    """LOCAL-BASELINE GUARD (mandatory -- acceptance condition 2). Mean similarity in the region
    FLANKING the matched run (never the run itself), at the SAME offset: a run sitting inside an
    otherwise-degraded region reads low here even when the run's own extension looked plausible.

    SIGN CONVENTION, stated because a flipped one here breaks silently rather than loudly
    (measured directly, see the module changelog note above): the candidate index for master
    index `x` is `x + offset` -- `offset` is defined as `candidate_index - master_index`
    (`j - i`) everywhere in this module, so recovering a candidate position from a master one
    ADDS the offset, never subtracts it. A subtraction here reads every run's flanking region
    against the wrong candidate span entirely, refusing nearly every real segment on a file whose
    true mean similarity is high -- caught once, this session, by the refusal COUNT being
    implausible for what the file's own overall similarity said; kept as a comment here so it is
    not rediscovered by re-deriving the sign from scratch under time pressure."""
    n_master, n_candidate = len(fp_master), len(fp_candidate)
    left = [sim(fp_master[x], fp_candidate[x + offset])
            for x in range(max(0, i_lo - window), i_lo)
            if 0 <= x + offset < n_candidate]
    right = [sim(fp_master[x], fp_candidate[x + offset])
             for x in range(i_hi + 1, min(n_master, i_hi + 1 + window))
             if 0 <= x + offset < n_candidate]
    pool = left + right
    return (sum(pool) / len(pool)) if pool else None


def _segment_evidence(segment):
    """How much a segment is WORTH in an overlap arbitration: matched master points, discounted
    by how well they matched.

    NOT `mean_match_quality` on its own, and the difference is measured, not stylistic. A short
    run is systematically CLEANER than a long one -- it stops extending exactly where the
    content stops agreeing, so its mean is taken over its best points only. On errid-202 a 1.24s
    noise fragment sitting at offset +10674 (86x the file's real offset scale -- it is not a
    relation, it is a coincidence) read `mean_match_quality` 0.987 against the 235s real segment
    beside it at 0.958. Arbitrating on quality alone therefore hands every contested span to the
    least evidence, which is how the first version of this guard threw away 12 minutes of a
    correctly aligned file (coverage 0.9987 -> 0.4571 on that pair).

    Points times quality is the honest quantity: it asks how many points actually matched, not
    how flattering the average of the survivors is. The master axis is used for the count because
    it is the reference grid, and because a segment's two axes span the same number of points by
    construction (`j = i + offset_points`).
    """
    return (segment["i_hi"] - segment["i_lo"] + 1) * segment["mean_match_quality"]


def best_shift_trace(fp_master, fp_candidate, i_start=0, i_end=None, current_offset=0,
                      x_points=TRACE_X_POINTS, m_points=TRACE_M_POINTS,
                      min_sustained=TRACE_MIN_SUSTAINED_WORDS):
    """Owner's periodic re-centering measurement -- built in as a first-class output from the
    start, not a retrofit added after the fact (a drift corrected without being recorded would
    hide the resample family inside this module the exact way a silent-absorption defect has
    already been measured doing elsewhere in this campaign). Returns the FULL best-shift-vs-
    position trace, one entry per checkpoint, with every re-center recorded (position and delta)
    -- nothing here is absorbed quietly.

    At each checkpoint (every `x_points` master indices), tests shifts in
    `[current_offset - m_points, current_offset + m_points]` against the ORIGINAL fingerprints
    (never a running/smoothed estimate) over the next `x_points`-wide window; each shift's score
    is how many of those points match at `>= HAMMING_MATCH_THRESHOLD` -- WORD COUNT, the owner's
    own unit, not a mean-similarity average, which is what lets `min_sustained` gate the re-center
    decision on a countable quantity rather than an opaque score. Re-centers only if a different
    shift's matched-word count beats the current offset's AND clears `min_sustained`.

    Reading the trace once built: a STEP (offset jumps by many points between adjacent
    checkpoints, or several consecutive re-centers all moving the same direction by more than one
    point at once) is a candidate CUT. A steady, small, one-point-at-a-time SLOPE across many
    checkpoints is monotone drift -- the resample family, read natively off this same trace,
    never a separate mechanism requiring its own pass."""
    n_master, n_candidate = len(fp_master), len(fp_candidate)
    if i_end is None:
        i_end = n_master

    def word_score(offset, i0):
        count = 0
        for x in range(i0, min(i0 + x_points, n_master)):
            j = x + offset
            if 0 <= j < n_candidate and sim(fp_master[x], fp_candidate[j]) >= HAMMING_MATCH_THRESHOLD:
                count += 1
        return count

    trace = []
    offset = current_offset
    i = i_start
    while i < i_end:
        current_score = word_score(offset, i)
        best_offset, best_score = offset, current_score
        for delta in range(-m_points, m_points + 1):
            if delta == 0:
                continue
            candidate_offset = offset + delta
            score = word_score(candidate_offset, i)
            if score > best_score:
                best_offset, best_score = candidate_offset, score
        recentered = (best_offset != offset) and (best_score >= min_sustained)
        trace.append({
            "modality": MODALITY,
            "master_index": i,
            "offset_points_before": offset,
            "offset_points_after": best_offset if recentered else offset,
            "recentered": recentered,
            "delta_points": (best_offset - offset) if recentered else 0,
            "matched_words_at_offset": best_score if recentered else current_score,
        })
        if recentered:
            offset = best_offset
        i += x_points
    return trace


def fit_trace_slope(trace):
    """Least-squares fit of offset (points) vs master index across a `best_shift_trace` result --
    slope in points/point, plus R^2 as the fit confidence carried WITH the slope (so this reading
    and a resample classifier's own slope regression can cross-check each other rather than one
    silently standing in for the other).

    AND THE RESIDUAL, WHICH IS THE READING THAT ACTUALLY SEPARATES A RATE RELATION FROM A CUT.
    `r_squared` cannot do it and must not be asked to: it is scale-free, so a clean two-plateau
    staircase fits a line with a high R^2 exactly as a genuine rate ramp does. The bake-off
    measured that trap directly (`/config/output/filter_bakeoff_ear/INDEX.md`, "l'avertissement
    methodologique"): a naive regression over the offset series of errid 213 returns "ratio
    1.0010739", 0.07 % from the NTSC nominal, on a file with NO rate relation at all; errid 135
    and errid 352 do the same. What denounces them is the RESIDUAL, not the slope -- 0.11 s to
    22 s for the staircases against under 10 microseconds for the two real rate pairs, five
    orders of magnitude apart. "Le residu est l'instrument ; la pente seule ment."

    So the residual is returned, in POINTS (this trace's own unit), and every consumer that wants
    milliseconds multiplies by ITS OWN quantum -- `b2_align` does exactly that below, because the
    quantum is per track and this function has never been told which track it is looking at.
    `residual_rms_points` is the root-mean-square of the fit residuals; `residual_max_points` is
    the largest single one, kept beside it because one big excursion and a general scatter are
    different facts about the same series and a mean hides the first.
    """
    xs = [entry["master_index"] for entry in trace]
    ys = [entry["offset_points_after"] for entry in trace]
    n = len(xs)
    if n < 2:
        return {"slope_points_per_point": None, "r_squared": None, "n": n,
                "residual_rms_points": None, "residual_max_points": None}
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    if ss_xx == 0:
        return {"slope_points_per_point": 0.0, "r_squared": None, "n": n,
                "fit_degenerate": "single_master_index",
                "residual_rms_points": None, "residual_max_points": None,
                "implied_step_count": sum(1 for entry in trace if entry["recentered"])}
    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    ss_res = sum(residual ** 2 for residual in residuals)
    # R^2 IS UNDEFINED ON A CONSTANT SERIES, AND THIS USED TO RETURN 1.0 FOR IT. `ss_tot == 0`
    # means every checkpoint read the SAME offset, which happens in two very different worlds: a
    # genuinely constant-offset pair, and a trace that never found anything at all and sat at its
    # starting offset from end to end. MEASURED, the second: errid-70 blind (the real PAL pair,
    # whose true offset is -2.85 s = -23 points, far outside the trace's +/-3-point re-centering
    # radius) produced 1026 checkpoints all reading offset 0 -- and this function reported
    # `r_squared: 1.0`, a PERFECT FIT, about a trace that had measured nothing. 1 - 0/0 is not 1;
    # it is undefined, and the standing invariant spells undefined `None`. `fit_degenerate` names
    # which of the two worlds it was, so a reader never has to guess from a blank.
    r_squared = (1 - (ss_res / ss_tot)) if ss_tot > 0 else None
    fit = {"slope_points_per_point": slope, "r_squared": r_squared, "n": n,
           "residual_rms_points": (ss_res / n) ** 0.5,
           "residual_max_points": max(abs(residual) for residual in residuals),
           "implied_step_count": sum(1 for entry in trace if entry["recentered"])}
    if ss_tot == 0:
        fit["fit_degenerate"] = "constant_offset_series"
    return fit


def b2_align(fp_master, fp_candidate, quantum_ms, band=None, k=K, bits=B,
             min_run_points=MIN_RUN_POINTS, local_baseline_min=LOCAL_BASELINE_MIN,
             include_drift_trace=True, duration_diff_ms=None,
             signed_duration_diff_ms=None, shorter_duration_ms=None,
             candidate_quantum_ms=None):
    """Full B2 pipeline over WHOLE-FILE fingerprint lists: degeneracy screen -> seed -> extend ->
    mandatory-anchor filter -> local-baseline guard -> merge overlapping runs into segments ->
    report cut zones between segments -> (optionally) the owner's re-centering drift trace.

    Returns a dict, ALWAYS, every field named so a caller need not re-derive any of them from
    context:

      "verdict"                 one of the tokens below; None is never returned.
      "stage_contract"           this result's own resolution-floor statement, in words, cited
                                  fresh every call rather than left for the caller to remember.
      "modality"                 MODALITY, literal, so a census can key on it directly.
      "degeneracy"               the entry-gate screen's own reading, ALWAYS present even when
                                  it did not fire, per this campaign's own "a rupture verdict
                                  pairs with a degeneracy check" discipline generalised here.
      "segments"                 merged offset-family runs on the master axis, each carrying
                                  `offset_ms`, `mean_match_quality`, `mean_local_baseline`,
                                  `n_members`, `master_time_range_s`.
      "all_zones"                EVERY zone between adjacent segments whose offsets differ, INCLUDING
                                  ones classified `below_resolution_floor` or `exceeds_duration_
                                  budget` -- a measurement, once made, is never silently discarded
                                  (see `RESOLUTION_FLOOR_QUANTA` and `duration_diff_ms` below).
      "cut_zones"                the subset of `all_zones` classified `"cut"` -- the CLAIMED cut
                                  list a caller should act on; `all_zones` is the full record.
      "refused_by_local_baseline_guard"   count of extended runs the guard refused -- reported
                                  even at zero, never a field that only exists when non-zero.
      "segments_filtered_short_fragments" count of segments dropped for being under
                                  `MIN_SEGMENT_DURATION_S`, reported for the same reason. ONE
                                  NUMBER, over BOTH passes of that filter -- the one before
                                  overlap resolution and the one that catches a segment a clip
                                  shrank below the floor; the field answers "how many were
                                  dropped for being too short", and two numbers would only make
                                  a reader add them up.
      "segments_overlap_resolved" count of adjacent segment pairs whose master- and/or
                                  candidate-axis bounds interleaved before being clipped to
                                  non-overlapping -- see the OVERLAP RESOLUTION comment below.
                                  Reported even at zero, same discipline as the two counts above.
      "zones"                    THE ORCHESTRATOR'S OWN OUTPUT SHAPE (owner's ruling
                                  2026-09-22, "sortie = LA LISTE DES ZONES ALIGNEES"):
                                  `[[[m_lo, m_hi], [c_lo, c_hi]], ...]` in FINGERPRINT POINTS,
                                  inclusive bounds, strictly increasing and non-overlapping on
                                  both axes (asserted). Same objects as `segments`, re-projected
                                  -- the ALIGNED runs are first class here, and the holes are
                                  derived as the gaps BETWEEN them; `cut_zones` remains the
                                  older gap-first view of the same measurement and is unchanged.
      "zones_detail"             1:1 with `zones`, SAME INDEX, carrying each zone's offset,
                                  match quality, baseline, member count and both axes' ms
                                  bounds -- so a consumer never has to multiply a point by a
                                  quantum itself and so cannot pick the WRONG axis's quantum.
      "candidate_quantum_ms"     the candidate track's OWN quantum, when the caller supplied
                                  one; the per-track quantum is a standing invariant (measured:
                                  124.0229 ms master vs 124.0350 ms candidate on one real pair)
                                  and is never averaged into a single per-pair number here.
      "master_axis_coverage_fraction"  fraction of the master axis covered by trusted zones --
                                  the honest positive claim that replaces the probe grid's
                                  `coverage_incomplete` decline (whole-file fingerprinting
                                  cannot leave the file unobserved, but it CAN leave it
                                  unaligned, and that is a different fact worth reporting).
      "drift_trace" / "drift_fit"   present when `include_drift_trace=True` (the default) --
                                  `best_shift_trace`'s FULL trace and `fit_trace_slope`'s summary,
                                  first-class output per the owner's own instruction that this
                                  never be a hidden internal.

    `duration_diff_ms`: OPTIONAL, the caller's own `|master_duration - candidate_duration|` in
    milliseconds, measured independently of anything this function computes (typically via
    ffprobe on both original files, before either side is truncated to the shorter one for
    extraction). When supplied, it is a PHYSICAL bound on a SINGLE step considered alone: content
    removed from one side has to show up in the length, so one uncompensated step cannot
    legitimately exceed it. Zones this large are classified `"exceeds_duration_budget"` rather
    than `"cut"`, on the same "never silently discard a measurement" principle as
    `below_resolution_floor` -- see `all_zones` vs `cut_zones` above.

    READ THIS TOKEN AS "THIS STEP EXCEEDS THE LENGTH DIFFERENCE," NOT AS "THIS STEP IS
    IMPOSSIBLE" (2026-09-21 correction, measured: mixed-sign steps can individually exceed the
    budget while together summing to something well within it -- id=60 in the real sweep had 10
    of 11 steps individually classified this way with a whole-file residual of 0.07%, i.e. the
    file's overall alignment was fine). The classification also fires whenever `duration_diff_ms`
    itself is small (files of nearly equal length), where even an ordinary few-second interior
    edit technically exceeds a near-zero budget without being any kind of anomaly -- id=406,
    id=375, id=688 in the real sweep each carried exactly one such zone. `duration_diff_ms` rides
    on every zone precisely so a reader can tell the two situations apart: a zone reading
    `exceeds_duration_budget` beside `duration_diff_ms=50` is a near-zero-budget artifact; one
    beside `duration_diff_ms=3492` next to a `step_ms` two orders of magnitude larger (id=147, a
    -599111ms step) is the real thing this check exists to name. The per-zone view cannot make
    that judgment itself and does not try to -- see `residual_ms`/`residual_fraction` below for
    the complementary per-result view, which mixed-sign cancellation and near-zero budgets affect
    in the opposite direction.

    When `None` (direct `b2_align` calls with no known real duration, e.g.
    most of this module's own diagnostics), the check is skipped and every large step still
    reads `"cut"` -- callers who cannot supply the real durations get today's behaviour
    unchanged, not a silently-stricter one.

    `signed_duration_diff_ms` / `shorter_duration_ms`: OPTIONAL, both needed together to compute
    `residual_ms` / `residual_fraction` on the RESULT (not per zone -- see those fields below).
    `signed_duration_diff_ms` is `(candidate_duration - master_duration) * 1000`, SIGNED, same
    convention as `step_ms` (positive = candidate longer). `shorter_duration_ms` is
    `min(master_duration, candidate_duration) * 1000`. Measured 2026-09-21 on the real sweep: a
    PER-ZONE check (`exceeds_duration_budget` above) and a PER-RESULT check (this one) catch
    different failures and neither substitutes for the other -- Pearson r=0.035 between
    `residual_fraction` and the fraction of a file's zones classified `exceeds_duration_budget`
    across 74 real pairs, i.e. essentially no linear relationship. A file can have a near-zero
    residual while carrying multiple locally-oversized-but-mutually-cancelling steps (mixed-sign
    edits that net out), or a large residual while every individual step stayed under its own
    budget (one file, id=49, had exactly one step that individually looked fine yet the whole
    length difference went unaccounted -- see the truncation-confound note on `duration_diff_ms`
    above). Only a per-result view catches an alignment that never settled; only a per-zone view
    names which reading is impossible. Neither is computed when omitted.

    `candidate_quantum_ms`: OPTIONAL, the CANDIDATE track's own quantum in milliseconds
    (`length_seconds * 1000 / len(fp_candidate)`), which is NOT the master's. The alignment maths
    is entirely index-based and never uses it -- it exists so that `zones_detail`'s
    `candidate_ms` bounds are read on the candidate's own grid instead of borrowing the master's.
    MEASURED, 2026-09-22, on real media: errid-202 read 124.0229 ms on the master and
    124.0350 ms on the candidate; errid-232's five couples read 124.0265 / 123.9822 ms. That gap
    is 0.01 %, which is nothing at one point and ~1.4 points (~174 ms) by the end of a 24-minute
    track -- i.e. exactly the size of error the frame stage is then asked to resolve. The
    standing per-TRACK-quantum invariant says these are two numbers and stay two numbers; when
    omitted, `candidate_ms` falls back to `quantum_ms` and the result SAYS SO by carrying
    `candidate_quantum_ms: None`, rather than silently echoing the master's as if it had been
    measured on the candidate.

    Verdicts: `"unreliable_degenerate_input"` (entry gate fired, nothing past `degeneracy` is
    trusted) / `"no_seeds_found"` (seeding ran, found nothing at all) /
    `"all_seeds_refused_by_local_baseline_guard"` (seeds existed and extended, but every one sat
    in a degraded region) / `"no_anchored_runs"` (no extension ever cleared `min_run_points`) /
    `"all_segments_below_duration_floor"` (runs anchored and were trusted, but every segment they
    built was shorter than `MIN_SEGMENT_DURATION_S` -- the rate-relation signature, see the guard
    at the floor's second pass) / `"single_segment_no_cut"` (evaluated cleanly, at least one
    segment survived, no offset step found) / `"segments_found"` (at least one `cut_zones`
    entry -- the only verdict carrying a non-empty `cut_zones`).

    THE FIRST FIVE ARE ALL "I COULD NOT MEASURE", the last two are measurements. That split is
    the one a caller must branch on, and it is why the fifth exists separately: it used to be
    reported as the sixth."""
    result = {
        "verdict": None,
        "modality": MODALITY,
        "stage_contract": ("COARSE stage. Resolution floor ~= k*quantum_ms = "
                            f"{k * quantum_ms:.1f}ms. Segments/zones below are BANDS, not "
                            "frame-accurate positions -- frame-accurate refinement is the "
                            "frame-anchor stage's job, this result is its input, never its "
                            "replacement. A reported zone's own WIDTH is a DIFFERENT quantity "
                            "from this resolution floor -- see edge_slack_note on each zone."),
        "k": k, "B": bits, "quantum_ms": quantum_ms,
        "candidate_quantum_ms": candidate_quantum_ms,
        "n_master": len(fp_master), "n_candidate": len(fp_candidate),
        "degeneracy": None, "segments": None, "all_zones": None, "cut_zones": None,
        # PRESENT ON EVERY RETURN PATH, None ON THE ONES THAT NEVER GOT TO MEASURE -- the
        # orchestrator branches on `zones`, and a KeyError on an early return would turn "the
        # aligner could not measure" into a crash instead of a named verdict.
        "zones": None, "zones_detail": None, "master_axis_coverage_fraction": None,
        "refused_by_local_baseline_guard": None, "admitted_self_evident": None,
        "segments_filtered_short_fragments": None,
        "segments_overlap_resolved": None,
        "drift_trace": None, "drift_fit": None,
        "residual_ms": None, "residual_fraction": None,
    }

    degeneracy = degeneracy_report(fp_master, fp_candidate)
    result["degeneracy"] = degeneracy
    if degeneracy["degenerate_a"] or degeneracy["degenerate_b"]:
        result["verdict"] = VERDICT_DEGENERATE_INPUT
        return result

    seeds = find_seeds(fp_master, fp_candidate, band=band, k=k, bits=bits)
    if not seeds:
        result["verdict"] = VERDICT_NO_SEEDS_FOUND
        return result

    extended = []
    seen_seed_offsets = set()
    self_evident_points = math.ceil(LOCAL_BASELINE_SELF_EVIDENT_SECONDS * 1000.0 / quantum_ms)
    for i, j in seeds:
        offset = j - i
        if (i, offset) in seen_seed_offsets:
            continue
        i_lo, i_hi, j_lo, j_hi, mean_sim = extend_seed(fp_master, fp_candidate, i, j)
        run_len = i_hi - i_lo + 1
        seen_seed_offsets.add((i, offset))
        if run_len < min_run_points:
            continue  # MANDATORY ANCHOR -- a chance k-mer hit with no sustained extension is
                       # exactly the noise-alignment trap this module's own acceptance contract
                       # names; refused here, before it can ever reach a segment.
        base = local_baseline(fp_master, fp_candidate, i_lo, i_hi, offset)
        self_evident = (run_len >= self_evident_points
                        and len(set(fp_master[i_lo:i_hi + 1]))
                        >= LOCAL_BASELINE_SELF_EVIDENT_MIN_DISTINCT_FRACTION * run_len)
        extended.append({"i_lo": i_lo, "i_hi": i_hi, "j_lo": j_lo, "j_hi": j_hi,
                          "offset_points": offset, "run_len": run_len, "mean_sim": mean_sim,
                          "local_baseline": base, "self_evident": self_evident})

    # LOCAL-BASELINE GUARD -- refuse any run whose surrounding region does not itself read high,
    # even when the run's own extension looked clean. See the module docstring's acceptance
    # condition 2 and `local_baseline`'s own docstring for the measured failure this refuses.
    # A SELF-EVIDENT run (see LOCAL_BASELINE_SELF_EVIDENT_SECONDS) is admitted on its own
    # length and content: its flanks are the far side of its own boundary, not its environment.
    trusted = [run for run in extended if run["self_evident"]
               or run["local_baseline"] is None
               or run["local_baseline"] >= local_baseline_min]
    result["admitted_self_evident"] = sum(
        1 for run in extended if run["self_evident"] and run["local_baseline"] is not None
        and run["local_baseline"] < local_baseline_min)
    # COUNT ONLY -- this used to also build `refused_by_baseline = [run for run in extended if
    # run not in trusted]`, an O(len(extended) * len(trusted)) dict-equality scan of a list
    # (measured: 8.8s of a 9.7s seed+extend+baseline phase at 23,028 points -- 65,582 extended
    # runs each linearly scanned against 8,771 trusted runs; the sole downstream use of that list
    # was `len(...)`). `trusted` is built above by filtering `extended` with the guard predicate,
    # so every run in `trusted` is a run from `extended` and every run refused is exactly the
    # complement -- no run is ever duplicated or dropped by that filter, so the refused count is
    # `len(extended) - len(trusted)` by construction, with the exact same meaning the old
    # (correct but quadratic) count carried, and no list of refused runs is ever materialised.
    result["refused_by_local_baseline_guard"] = len(extended) - len(trusted)

    if not trusted:
        result["verdict"] = (VERDICT_ALL_SEEDS_REFUSED if extended
                              else VERDICT_NO_ANCHORED_RUNS)
        return result

    # Merge overlapping/adjacent runs on the master axis into segments (same offset family), with
    # PLATEAU_TOLERANCE-style clustering: offsets within OFFSET_MERGE_TOLERANCE_POINTS of the
    # segment's own running mean join it, rather than requiring an exact match -- see the module-
    # level constant's own comment for the measured cost of an exact-offset merge.
    trusted.sort(key=lambda run: run["i_lo"])
    segments = []
    for run in trusted:
        if segments and run["i_lo"] <= segments[-1]["i_hi"] + min_run_points \
                and abs(run["offset_points"] - segments[-1]["offset_points_running_mean"]) \
                <= OFFSET_MERGE_TOLERANCE_POINTS:
            seg = segments[-1]
            seg["i_hi"] = max(seg["i_hi"], run["i_hi"])
            seg["j_hi"] = max(seg["j_hi"], run["j_hi"])
            seg["members"].append(run)
            seg["offset_points_running_mean"] = statistics.mean(
                m["offset_points"] for m in seg["members"])
        else:
            segments.append({"i_lo": run["i_lo"], "i_hi": run["i_hi"], "j_lo": run["j_lo"],
                              "j_hi": run["j_hi"],
                              "offset_points_running_mean": run["offset_points"],
                              "members": [run]})

    for seg in segments:
        seg["modality"] = MODALITY
        seg["mean_match_quality"] = statistics.mean(m["mean_sim"] for m in seg["members"])
        baselines = [m["local_baseline"] for m in seg["members"] if m["local_baseline"] is not None]
        seg["mean_local_baseline"] = statistics.mean(baselines) if baselines else None
        seg["master_time_range_s"] = [seg["i_lo"] * quantum_ms / 1000.0,
                                       seg["i_hi"] * quantum_ms / 1000.0]
        seg["offset_points"] = round(seg["offset_points_running_mean"])
        seg["offset_ms"] = seg["offset_points"] * quantum_ms
        seg["n_members"] = len(seg["members"])
        del seg["members"]
        del seg["offset_points_running_mean"]

    # MINIMUM SEGMENT DURATION, stated not silent -- see the module-level constant's own comment.
    # RUNS BEFORE OVERLAP RESOLUTION, AND THAT ORDER IS THE CORRECTION'S OWN PRECONDITION.
    # MEASURED, errid-202 jpn, with the filter running after resolution instead: of the 22
    # segments this pair produces, TEN are sub-2s noise fragments, and a short fragment reads a
    # SYSTEMATICALLY HIGHER `mean_match_quality` than a long real segment (a 1.24s fragment at
    # offset +10674 -- pure noise, 86x the file's real offset scale -- measured q=0.987 against
    # the neighbouring 235s real segment's 0.958). Letting those fragments into the arbitration
    # below let them WIN it and consume their long neighbours: master-axis coverage collapsed
    # from 0.9987 to 0.4571 on that pair and the first surviving segment moved from point 8 to
    # point 6106, i.e. the first 12 minutes of a correctly-aligned file were thrown away by the
    # fix meant to make its zones well-defined. Arbitration is only meaningful between segments
    # that have already cleared the floor.
    kept_segments = [seg for seg in segments
                      if (seg["master_time_range_s"][1] - seg["master_time_range_s"][0])
                      >= MIN_SEGMENT_DURATION_S]
    segments_filtered_short_fragments = len(segments) - len(kept_segments)
    segments = kept_segments

    # OVERLAP RESOLUTION -- BLOCKING correction (design section 3.4a). Two segments come from
    # independent offset families (the merge above only joins runs that already agree on
    # offset); two runs at GENUINELY DIFFERENT offsets that happen to interleave in master index
    # start two segments whose [i_lo,i_hi] (or [j_lo,j_hi]) ranges overlap. Measured on real
    # media before this fix: errid-202 jpn 1/19 zones read `hi < lo` from 1/37 overlapping
    # adjacent segment pairs; errid-232's five couples read 0-13 overlapping pairs each. A zone
    # built from interleaved segments has an undefined "hole" downstream, so this must be
    # resolved before `all_zones` is built, never left to the caller.
    #
    # Choice of repair, and why: this module's own offset-family semantics say a segment IS an
    # offset -- one `offset_points` relates every one of its master indices to its candidate
    # indices (`j = i + offset_points`, the same convention `local_baseline`'s docstring states
    # explicitly). So a contested master (or candidate) span belongs, in full, to whichever
    # neighbouring segment's evidence is stronger -- never split point-by-point re-deciding each
    # point's own offset, which nothing here measures. The loser is clipped on ITS OWN axis pair
    # together (i and j move as one, through ITS OWN offset), so the clip can never desynchronise
    # a segment's own i<->j correspondence. The clip is sized to the MORE RESTRICTIVE of what the
    # i-axis and the j-axis independently demand, so one clip clears both axes in one step rather
    # than needing a second pass. A segment entirely consumed by a stronger neighbour is dropped
    # and counted here, not silently vanished into `segments_filtered_short_fragments`.
    #
    # "STRONGER" IS HOW MUCH EVIDENCE, NOT HOW CLEAN IT IS -- see `_segment_evidence` below for
    # the measurement that forced that definition. Quality alone systematically favours the
    # shortest segment, which is the opposite of what arbitration is for.
    segments.sort(key=lambda seg: seg["i_lo"])
    segments_overlap_resolved = 0
    idx = 0
    while idx < len(segments) - 1:
        seg_a, seg_b = segments[idx], segments[idx + 1]
        if seg_b["i_lo"] > seg_a["i_hi"] and seg_b["j_lo"] > seg_a["j_hi"]:
            idx += 1
            continue
        segments_overlap_resolved += 1
        if _segment_evidence(seg_a) >= _segment_evidence(seg_b):
            # seg_a keeps the contested span; clip seg_b forward, through seg_b's OWN offset.
            new_i_lo = max(seg_a["i_hi"] + 1, (seg_a["j_hi"] + 1) - seg_b["offset_points"])
            seg_b["i_lo"] = new_i_lo
            seg_b["j_lo"] = new_i_lo + seg_b["offset_points"]
            seg_b["master_time_range_s"][0] = new_i_lo * quantum_ms / 1000.0
            if seg_b["i_lo"] > seg_b["i_hi"] or seg_b["j_lo"] > seg_b["j_hi"]:
                segments.pop(idx + 1)  # seg_b fully consumed -- re-check (idx, idx+1) afresh.
            else:
                idx += 1
        else:
            # seg_b keeps the contested span; clip seg_a backward, through seg_a's OWN offset.
            new_i_hi = min(seg_b["i_lo"] - 1, (seg_b["j_lo"] - 1) - seg_a["offset_points"])
            seg_a["i_hi"] = new_i_hi
            seg_a["j_hi"] = new_i_hi + seg_a["offset_points"]
            seg_a["master_time_range_s"][1] = new_i_hi * quantum_ms / 1000.0
            if seg_a["i_lo"] > seg_a["i_hi"] or seg_a["j_lo"] > seg_a["j_hi"]:
                segments.pop(idx)  # seg_a fully consumed -- re-check against idx-1 if it exists.
                if idx > 0:
                    idx -= 1
            else:
                idx += 1
    result["segments_overlap_resolved"] = segments_overlap_resolved

    # THE FLOOR AGAIN, ON WHAT THE CLIPS LEFT BEHIND. A clip can shrink a segment that entered
    # the arbitration well clear of the floor to something under it (measured: errid-202's
    # 3.60s segment at offset -25 is clipped to 1.49s by its stronger neighbour). That shrunk
    # remnant is a short fragment like any other and is caught by the SAME named filter, its
    # count ADDED to the first pass's rather than reported separately -- the field means "how
    # many segments were dropped for being too short", and splitting it into two numbers would
    # make a reader reconstruct the total to answer the only question it is asked.
    kept_segments = [seg for seg in segments
                      if (seg["master_time_range_s"][1] - seg["master_time_range_s"][0])
                      >= MIN_SEGMENT_DURATION_S]
    segments_filtered_short_fragments += len(segments) - len(kept_segments)
    result["segments_filtered_short_fragments"] = segments_filtered_short_fragments
    segments = kept_segments

    # ZERO SURVIVING SEGMENTS IS NOT "ONE SEGMENT WITH NO CUT", AND UNTIL 2026-09-22 THIS
    # FUNCTION SAID IT WAS. The verdict below is derived from whether `cut_zones` is empty, and
    # `cut_zones` is built from `segments` -- so a file whose every anchored run was shorter than
    # `MIN_SEGMENT_DURATION_S` came out of here reading `single_segment_no_cut` with `zones: []`
    # and `master_axis_coverage_fraction: 0.0`. That reads to a caller as "these two audios
    # align perfectly and need no splice", which is the exact inverse of what was measured.
    #
    # MEASURED, AND IT IS THE PAIR THIS MATTERS MOST ON: errid-70, the corpus's ONLY real
    # speed-relation pair (candidate 25 fps against a 23.976 master, factor 1.0427), fre couple,
    # 10256 x 9837 points. At 4.27 % the offset moves a full quantum every ~2.9 s, so no
    # extension survives long enough to clear a 2 s floor: 2 runs extended, 2 refused by the
    # baseline guard, ONE segment built and ONE filtered as a short fragment, leaving nothing --
    # and the old verdict announced a perfectly compatible pair. Downstream, the orchestrator's
    # step-2 similarity gate reads this verdict to decide whether to run the speed sweep, so the
    # one pair in the corpus that NEEDS the sweep was the one pair certified as not needing it.
    # A could-not-measure read as a measured negative: the defect class this campaign has now
    # found nine times.
    #
    # THE TOKEN IS ITS OWN, not a reuse of `no_anchored_runs`. Runs WERE anchored here -- they
    # cleared `min_run_points` and the baseline guard -- and they were then refused by a
    # DIFFERENT, later, nameable filter. Folding the two together would hide which gate fired
    # from the only reader who can act on it, and the counts that explain it
    # (`segments_filtered_short_fragments`, `refused_by_local_baseline_guard`) are already on
    # the result beside it.
    if not segments:
        result["verdict"] = VERDICT_ALL_SEGMENTS_BELOW_DURATION_FLOOR
        result["zones"] = []
        result["zones_detail"] = []
        result["segments"] = []
        result["master_axis_coverage_fraction"] = 0.0
        return result

    # ASSERTED, not merely intended: emitted segments must be strictly increasing and non-
    # overlapping on BOTH axes, or "holes = the gaps between zones" is undefined downstream. This
    # is the module's own contract on its own output, checked at the point every field above is
    # already known to be final for this call.
    for seg_a, seg_b in zip(segments, segments[1:]):
        assert seg_a["i_hi"] < seg_b["i_lo"], (
            "banded_seed_alignment: segments must be non-overlapping on the master axis "
            f"after overlap resolution: {seg_a['i_hi']} !< {seg_b['i_lo']}")
        assert seg_a["j_hi"] < seg_b["j_lo"], (
            "banded_seed_alignment: segments must be non-overlapping on the candidate axis "
            f"after overlap resolution: {seg_a['j_hi']} !< {seg_b['j_lo']}")
    result["segments"] = segments

    # THE ORCHESTRATOR'S OUTPUT CONTRACT (owner's ruling 2026-09-22, step 2 of `chimeric`:
    # "sortie = LA LISTE DES ZONES ALIGNEES : [[[m_debut, m_fin], [c_debut, c_fin]], ...]").
    #
    # This is a RE-PROJECTION, not a second measurement: `zones[i]` is built from
    # `segments[i]` and nothing else, so no reading can disagree between the two views. The
    # transform is the whole point of the contract change -- this module was built emitting the
    # GAPS between segments (`all_zones`/`cut_zones`) as first class, and the orchestrator needs
    # the MATCHED RUNS as first class because its holes are defined as the gaps between zones
    # and its no_cut_confirmed return CLOSES a hole (which is meaningless against a gap-first
    # list). Both views ship: `cut_zones` is unchanged for every existing reader, and no caller
    # is obliged to derive one from the other.
    #
    # POINTS ARE THE PRIMARY UNIT, ms is derived and carried beside it. The ruling's own input
    # to this function is the fingerprint lists, so its output belongs in the same index space;
    # a consumer that wants time gets it already multiplied -- and, per the per-track-quantum
    # invariant, multiplied by THE RIGHT AXIS'S QUANTUM (see `candidate_quantum_ms` above).
    #
    # BOUNDS ARE INCLUSIVE in points (`i_lo`..`i_hi` is the matched run) and HALF-OPEN in ms
    # (`[i_lo*q, (i_hi+1)*q)`), because point `i_hi` is itself `q` milliseconds WIDE -- reading
    # its end as `i_hi*q` would drop the last point of every zone, a one-quantum systematic
    # shortening at every zone's right edge that would land squarely inside the frame stage's
    # search window and be invisible there.
    candidate_quantum = (candidate_quantum_ms if candidate_quantum_ms is not None
                          else quantum_ms)
    zones = []
    zones_detail = []
    for seg in segments:
        zones.append([[seg["i_lo"], seg["i_hi"]], [seg["j_lo"], seg["j_hi"]]])
        zones_detail.append({
            "modality": MODALITY,
            "master_points": [seg["i_lo"], seg["i_hi"]],
            "candidate_points": [seg["j_lo"], seg["j_hi"]],
            "master_ms": [seg["i_lo"] * quantum_ms, (seg["i_hi"] + 1) * quantum_ms],
            "candidate_ms": [seg["j_lo"] * candidate_quantum,
                              (seg["j_hi"] + 1) * candidate_quantum],
            "offset_points": seg["offset_points"],
            "offset_ms": seg["offset_ms"],
            "mean_match_quality": seg["mean_match_quality"],
            "mean_local_baseline": seg["mean_local_baseline"],
            "n_members": seg["n_members"],
        })
    result["zones"] = zones
    result["zones_detail"] = zones_detail

    # COVERAGE, STATED AS A POSITIVE CLAIM. The probe grid it replaces could leave stretches of
    # the file simply unobserved, and declined `coverage_incomplete` when it did. Whole-file
    # fingerprinting makes UNOBSERVED structurally impossible -- but UNALIGNED is a different
    # fact and still perfectly possible, so the honest replacement is the fraction of the master
    # axis that trusted zones actually cover, reported always, never a decline on its own. A
    # reader comparing two couples of the same pair uses this to tell "this track saw less"
    # apart from "this track disagreed", which is the could-not-see-vs-disagree distinction the
    # cross-verification is built on.
    if len(fp_master) > 0:
        covered_points = sum(seg["i_hi"] - seg["i_lo"] + 1 for seg in segments)
        result["master_axis_coverage_fraction"] = covered_points / len(fp_master)

    # RESOLUTION-FLOOR CLASSIFICATION: a step below this module's own declared resolution floor
    # (RESOLUTION_FLOOR_QUANTA * quantum) is NEVER deleted -- deletion teaches nobody and hides
    # the exact boundary a real small-trim fixture needs calibrated against. Classified instead:
    # below-floor zones stay in `all_zones` tagged `below_resolution_floor`, excluded only from
    # `cut_zones`, the CLAIMED-cut list.
    floor_ms = RESOLUTION_FLOOR_QUANTA * quantum_ms

    all_zones = []
    for seg_a, seg_b in zip(segments, segments[1:]):
        if seg_a["offset_points"] == seg_b["offset_points"]:
            continue
        step_ms = (seg_b["offset_points"] - seg_a["offset_points"]) * quantum_ms
        # DURATION-BUDGET CLASSIFICATION -- a physical bound, not a chosen one (2026-09-21,
        # measured on the real sweep: 24/75 whole-file pairs carried a single step exceeding
        # |master_duration - candidate_duration|, up to 418x on one pair whose sides differed by
        # under three seconds). Content removed from one side must show up in the length, so ONE
        # UNCOMPENSATED step cannot legitimately exceed that difference -- but mixed-sign steps
        # elsewhere in the same file CAN compensate, so this names "exceeds the length
        # difference," never "is impossible" on its own -- see `b2_align`'s own docstring for the
        # measured cases (id=60, id=406/375/688) that make the distinction load-bearing, and
        # `residual_ms`/`residual_fraction` for the per-result view this per-zone one cannot be.
        # Checked AFTER the resolution floor (a below-floor step is always small enough to pass
        # this regardless, so the two never actually compete) and only when the caller supplied a
        # real `duration_diff_ms`.
        if abs(step_ms) < floor_ms:
            classification = "below_resolution_floor"
        elif duration_diff_ms is not None and abs(step_ms) > duration_diff_ms:
            classification = "exceeds_duration_budget"
        else:
            classification = "cut"
        zone = {
            "modality": MODALITY,
            "zone_master_index_bounds": [seg_a["i_hi"], seg_b["i_lo"]],
            "zone_master_time_bounds_s": [seg_a["i_hi"] * quantum_ms / 1000.0,
                                           seg_b["i_lo"] * quantum_ms / 1000.0],
            "offset_before_ms": seg_a["offset_points"] * quantum_ms,
            "offset_after_ms": seg_b["offset_points"] * quantum_ms,
            "step_ms": step_ms,
            "classification": classification,
            "resolution_floor_ms": floor_ms,
            "duration_diff_ms": duration_diff_ms,
            "edge_slack_note": ("This zone's own width is the gap between the last matched point "
                                 "before it and the first matched point after -- it necessarily "
                                 "spans the true removed/added span PLUS extension slack (measured "
                                 "~1.75s total across both edges on constructed fixtures), NOT the "
                                 "resolution_floor_ms above. The true position sits close to the "
                                 "zone's NEAR edge, not centred (measured mean nearest-edge delta "
                                 "0.9s, range 0.2-1.9s, n=7 constructed cuts) -- never read this "
                                 "width as either quantity it is not."),
        }
        all_zones.append(zone)

    cut_zones = [zone for zone in all_zones if zone["classification"] == "cut"]
    result["all_zones"] = all_zones
    result["cut_zones"] = cut_zones
    result["verdict"] = VERDICT_SEGMENTS_FOUND if cut_zones else VERDICT_SINGLE_SEGMENT_NO_CUT

    # PER-RESULT SELF-CONSISTENCY (2026-09-21) -- deliberately separate from the per-zone
    # `exceeds_duration_budget` check above, and deliberately computed from the FULL segment
    # chain rather than from `cut_zones`: summing only the zones that SURVIVED the per-zone
    # filter would make this check parasitic on that filter, reporting a clean result precisely
    # when a wild zone was already excised -- exactly the case (id=147) this check exists to
    # still catch. Consecutive segment-offset differences telescope, so the total signed drift
    # from the first segment to the last equals the sum of ALL step_ms (below-floor, cut, and
    # exceeds-budget alike) without re-summing them individually. Measured Pearson r=0.035
    # between this fraction and the per-zone exceeds rate across 74 real pairs, i.e. the two
    # checks catch different failures; see this function's own docstring. Computed only when the
    # caller supplied both real-duration quantities, and only when at least two segments survived
    # (a single segment has no drift to measure and no length difference to blame on one).
    if signed_duration_diff_ms is not None and shorter_duration_ms and len(segments) >= 2:
        signed_step_sum_ms = ((segments[-1]["offset_points"] - segments[0]["offset_points"])
                               * quantum_ms)
        residual_ms = signed_step_sum_ms - signed_duration_diff_ms
        result["residual_ms"] = residual_ms
        result["residual_fraction"] = abs(residual_ms) / shorter_duration_ms

    if include_drift_trace:
        trace = best_shift_trace(fp_master, fp_candidate)
        for entry in trace:
            entry["master_time_s"] = entry["master_index"] * quantum_ms / 1000.0
        result["drift_trace"] = trace
        result["drift_fit"] = fit_trace_slope(trace)
        # THE RESIDUAL IN MILLISECONDS, ATTACHED HERE AND NOWHERE ELSE, because this is the first
        # place that knows THIS COUPLE'S quantum. `fit_trace_slope` answers in points because a
        # trace has no track attached to it; the per-track-quantum invariant means the conversion
        # can only be done by a caller that owns one. Both readings are carried: a scatter and a
        # worst excursion say different things about the same series.
        for _points_key, _ms_key in (("residual_rms_points", "residual_rms_ms"),
                                      ("residual_max_points", "residual_max_ms")):
            _value = result["drift_fit"].get(_points_key)
            result["drift_fit"][_ms_key] = None if _value is None else _value * quantum_ms

    return result
