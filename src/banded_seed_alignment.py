"""Banded k-mer seed-and-extend alignment of two whole-track chromaprint fingerprints.

`b2_align` returns the aligned zones (matched runs, each at one constant offset) between a
master and a candidate fingerprint. Seeds are k-mers on a reduced alphabet (top `B` bits of each
32-bit value); runs are extended on the full 32-bit values with a Hamming similarity threshold,
optionally inside a caller-supplied offset band.

This is a coarse stage: resolution is about k * quantum (~250-280 ms). Runs only count when their
flanks also match at the same offset (local-baseline guard), and inputs with at most one
distinct value (silence) are rejected before seeding. k=2, B=12 is the smallest reduction with a
usable seed rate on clean pairs without excessive false seeds.

A zone's width is the gap between matched points on each side of a cut, i.e. the true edit plus
extension slack; the true cut sits near one edge, not at the centre.
"""
import math
import statistics
import time

# Verdicts. The first five mean "could not measure" and must not be read as a negative result.
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
# The time budget ran out: a statement about cost, not about the pair, so in neither tuple.
VERDICT_ALIGNMENT_BUDGET_EXCEEDED = "alignment_budget_exceeded"
DEADLINE_CHECK_EVERY_SEEDS = 256

# Note: VERDICT_SINGLE_SEGMENT_NO_CUT means "no offset step found", not "the tracks align";
# always read `master_axis_coverage_fraction` beside it.

MODALITY = "audio_kmer_seed_b2"

K = 2
B = 12
# Consecutive matching points before a seed founds a run: chance similarity (~0.5) aligns noise.
MIN_RUN_POINTS = 5
# Per-point similarity floor while extending, on full 32-bit values.
HAMMING_MATCH_THRESHOLD = 0.85
LOCAL_BASELINE_WINDOW = 15
# Minimum mean flank similarity: clean plateaus read ~0.93-0.97, degraded regions ~0.5.
LOCAL_BASELINE_MIN = 0.75
# A run at least this long over non-degenerate content is admitted without its flanks: a run
# extended over identical audio ends at a cut or file edge, so its flanks never match. 30 s is
# longer than any in-file repeat.
LOCAL_BASELINE_SELF_EVIDENT_SECONDS = 30.0
# "Non-degenerate": at least half of the run's master values distinct (silence repeats one value).
LOCAL_BASELINE_SELF_EVIDENT_MIN_DISTINCT_FRACTION = 0.5
# Runs whose offsets agree within this many points (per-seed jitter) merge into one segment.
OFFSET_MERGE_TOLERANCE_POINTS = 3
# Shorter segments are noise fragments at a real segment's boundary.
MIN_SEGMENT_DURATION_S = 2.0
# A step under this many quanta is classified `below_resolution_floor` instead of `cut`.
RESOLUTION_FLOOR_QUANTA = 2

# Re-centering trace: checkpoint every X points (>= 2 per PAL-rate drift step), search radius
# M points, and a re-center needs at least this many matched words so noise is never chased.
TRACE_X_POINTS = 10
TRACE_M_POINTS = 3
TRACE_MIN_SUSTAINED_WORDS = 3


def _popcount32(value):
    return bin(value & 0xFFFFFFFF).count("1")


def sim(a, b):
    """Return the fraction of equal bits between two 32-bit fingerprint values."""
    return (32 - _popcount32(a ^ b)) / 32.0


def _reduce(value, bits=B):
    """Return the top `bits` bits of a 32-bit value (the reduced alphabet used for seeding)."""
    shift = 32 - bits
    return (value & 0xFFFFFFFF) >> shift if shift else (value & 0xFFFFFFFF)


def degeneracy_report(fingerprint_a, fingerprint_b):
    """Report distinct-value counts; a side with at most one distinct value is degenerate.

    Flat or silent content matches at every offset, so it cannot be aligned."""
    distinct_a = len(set(fingerprint_a))
    distinct_b = len(set(fingerprint_b))
    return {
        "n_items_a": len(fingerprint_a), "n_items_b": len(fingerprint_b),
        "n_distinct_a": distinct_a, "n_distinct_b": distinct_b,
        "degenerate_a": distinct_a <= 1 and len(fingerprint_a) > 0,
        "degenerate_b": distinct_b <= 1 and len(fingerprint_b) > 0,
    }


def build_kmer_index(fingerprint, k=K, bits=B):
    """Map each reduced k-mer tuple to its starting positions in `fingerprint`."""
    reduced = [_reduce(x, bits) for x in fingerprint]
    index = {}
    for i in range(len(reduced) - k + 1):
        key = tuple(reduced[i:i + k])
        index.setdefault(key, []).append(i)
    return index


def find_seeds(fp_master, fp_candidate, band=None, k=K, bits=B):
    """Return seeds (master_index, candidate_index) whose reduced k-mers are equal.

    Args:
        band: optional (offset_lo, offset_hi) in points restricting candidate - master index.
    """
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
    """Extend seed (i, j) both ways while per-point similarity stays >= threshold.

    Returns:
        (i_lo, i_hi, j_lo, j_hi, mean_sim) of the matched run, inclusive bounds.
    """
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
    offset = j - i
    sims = [sim(fp_master[x], fp_candidate[x + offset]) for x in range(i_lo, i_hi + 1)]
    return i_lo, i_hi, j_lo, j_hi, (sum(sims) / len(sims) if sims else 0.0)


def local_baseline(fp_master, fp_candidate, i_lo, i_hi, offset, window=LOCAL_BASELINE_WINDOW):
    """Return the mean similarity of the run's flanks at the same offset, or None if no flank.

    A run inside an otherwise degraded region reads low here even when the run itself is clean.
    `offset` is candidate_index - master_index throughout this module.
    """
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
    """Return a segment's weight in overlap arbitration: matched points times mean quality.

    Quality alone would favour short fragments, which are systematically cleaner than long runs.
    """
    return (segment["i_hi"] - segment["i_lo"] + 1) * segment["mean_match_quality"]


def best_shift_trace(fp_master, fp_candidate, i_start=0, i_end=None, current_offset=0,
                      x_points=TRACE_X_POINTS, m_points=TRACE_M_POINTS,
                      min_sustained=TRACE_MIN_SUSTAINED_WORDS):
    """Record the best shift at every checkpoint, re-centering on the way.

    Every `x_points` master indices, shifts within +/- `m_points` of the current offset are
    scored by how many points of the next window match; the offset moves only when another
    shift scores higher and reaches `min_sustained`. A step in the trace suggests a cut; a steady
    one-point slope suggests a rate (resample) relation.
    """
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
    """Fit offset (points) against master index over a `best_shift_trace` result.

    Returns slope, R^2 and residuals in points. The residual, not R^2, separates a rate relation
    from a cut: a staircase also fits a line with high R^2, but with large residuals.
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
    # R^2 is undefined on a constant series (a true constant offset, or a trace that never moved
    # from its start), so it reads None and `fit_degenerate` names the case.
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
             candidate_quantum_ms=None, deadline=None):
    """Align two whole-track fingerprints into constant-offset zones.

    Pipeline: degeneracy screen, seeding, extension, minimum-run anchor, local-baseline guard,
    merge into segments, short-fragment filter, overlap resolution, zones, optional drift trace.

    Args:
        quantum_ms: master fingerprint point duration.
        band: optional (offset_lo, offset_hi) in points to restrict seeding.
        duration_diff_ms: optional |master - candidate| duration; larger steps are classified
            `exceeds_duration_budget` instead of `cut`.
        signed_duration_diff_ms, shorter_duration_ms: optional; both enable `residual_ms` /
            `residual_fraction` (net step minus the real duration difference).
        candidate_quantum_ms: optional candidate point duration, used only for ms bounds.
        deadline: optional time.monotonic() instant; past it the verdict is
            `alignment_budget_exceeded`.

    Returns:
        dict with every key present (None when not measured), notably: verdict (a VERDICT_*
        constant), segments, all_zones, cut_zones, zones ([[[m_lo, m_hi], [c_lo, c_hi]], ...]
        in points, inclusive, increasing and non-overlapping), zones_detail,
        master_axis_coverage_fraction, drift_trace / drift_fit, residual_ms / residual_fraction.
    """
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
    for number, (i, j) in enumerate(seeds):
        # Seed extension is the super-linear part, so the deadline is checked here.
        if (deadline is not None and number % DEADLINE_CHECK_EVERY_SEEDS == 0
                and time.monotonic() > deadline):
            result["verdict"] = VERDICT_ALIGNMENT_BUDGET_EXCEEDED
            result["seeds_extended"] = number
            result["seeds_total"] = len(seeds)
            return result
        offset = j - i
        if (i, offset) in seen_seed_offsets:
            continue
        i_lo, i_hi, j_lo, j_hi, mean_sim = extend_seed(fp_master, fp_candidate, i, j)
        run_len = i_hi - i_lo + 1
        seen_seed_offsets.add((i, offset))
        if run_len < min_run_points:
            continue
        base = local_baseline(fp_master, fp_candidate, i_lo, i_hi, offset)
        self_evident = (run_len >= self_evident_points
                        and len(set(fp_master[i_lo:i_hi + 1]))
                        >= LOCAL_BASELINE_SELF_EVIDENT_MIN_DISTINCT_FRACTION * run_len)
        extended.append({"i_lo": i_lo, "i_hi": i_hi, "j_lo": j_lo, "j_hi": j_hi,
                          "offset_points": offset, "run_len": run_len, "mean_sim": mean_sim,
                          "local_baseline": base, "self_evident": self_evident})

    # Local-baseline guard; self-evident runs are admitted on length alone.
    trusted = [run for run in extended if run["self_evident"]
               or run["local_baseline"] is None
               or run["local_baseline"] >= local_baseline_min]
    result["admitted_self_evident"] = sum(
        1 for run in extended if run["self_evident"] and run["local_baseline"] is not None
        and run["local_baseline"] < local_baseline_min)
    result["refused_by_local_baseline_guard"] = len(extended) - len(trusted)

    if not trusted:
        result["verdict"] = (VERDICT_ALL_SEEDS_REFUSED if extended
                              else VERDICT_NO_ANCHORED_RUNS)
        return result

    # Merge adjacent runs whose offset is within tolerance of the segment's running mean.
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

    # Drop short fragments before overlap resolution, so they cannot win arbitration against
    # long real segments.
    kept_segments = [seg for seg in segments
                      if (seg["master_time_range_s"][1] - seg["master_time_range_s"][0])
                      >= MIN_SEGMENT_DURATION_S]
    segments_filtered_short_fragments = len(segments) - len(kept_segments)
    segments = kept_segments

    # Overlap resolution: interleaved runs at different offsets can yield overlapping segments.
    # The contested span goes to the neighbour with more evidence; the loser is clipped on both
    # axes through its own offset, and dropped when fully consumed.
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
            new_i_lo = max(seg_a["i_hi"] + 1, (seg_a["j_hi"] + 1) - seg_b["offset_points"])
            seg_b["i_lo"] = new_i_lo
            seg_b["j_lo"] = new_i_lo + seg_b["offset_points"]
            seg_b["master_time_range_s"][0] = new_i_lo * quantum_ms / 1000.0
            if seg_b["i_lo"] > seg_b["i_hi"] or seg_b["j_lo"] > seg_b["j_hi"]:
                segments.pop(idx + 1)
            else:
                idx += 1
        else:
            new_i_hi = min(seg_b["i_lo"] - 1, (seg_b["j_lo"] - 1) - seg_a["offset_points"])
            seg_a["i_hi"] = new_i_hi
            seg_a["j_hi"] = new_i_hi + seg_a["offset_points"]
            seg_a["master_time_range_s"][1] = new_i_hi * quantum_ms / 1000.0
            if seg_a["i_lo"] > seg_a["i_hi"] or seg_a["j_lo"] > seg_a["j_hi"]:
                segments.pop(idx)
                if idx > 0:
                    idx -= 1
            else:
                idx += 1
    result["segments_overlap_resolved"] = segments_overlap_resolved

    # Clipping can shrink a segment under the floor, so filter again.
    kept_segments = [seg for seg in segments
                      if (seg["master_time_range_s"][1] - seg["master_time_range_s"][0])
                      >= MIN_SEGMENT_DURATION_S]
    segments_filtered_short_fragments += len(segments) - len(kept_segments)
    result["segments_filtered_short_fragments"] = segments_filtered_short_fragments
    segments = kept_segments

    # No surviving segment is a could-not-measure verdict, typically a rate relation (PAL drift
    # moves a quantum every ~3 s, so no run clears the floor), not "aligned with no cut".
    if not segments:
        result["verdict"] = VERDICT_ALL_SEGMENTS_BELOW_DURATION_FLOOR
        result["zones"] = []
        result["zones_detail"] = []
        result["segments"] = []
        result["master_axis_coverage_fraction"] = 0.0
        return result

    for seg_a, seg_b in zip(segments, segments[1:]):
        assert seg_a["i_hi"] < seg_b["i_lo"], (
            "banded_seed_alignment: segments must be non-overlapping on the master axis "
            f"after overlap resolution: {seg_a['i_hi']} !< {seg_b['i_lo']}")
        assert seg_a["j_hi"] < seg_b["j_lo"], (
            "banded_seed_alignment: segments must be non-overlapping on the candidate axis "
            f"after overlap resolution: {seg_a['j_hi']} !< {seg_b['j_lo']}")
    result["segments"] = segments

    # Bounds are inclusive in points and half-open in ms ([i_lo*q, (i_hi+1)*q)), each axis on its
    # own quantum.
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

    if len(fp_master) > 0:
        covered_points = sum(seg["i_hi"] - seg["i_lo"] + 1 for seg in segments)
        result["master_axis_coverage_fraction"] = covered_points / len(fp_master)

    floor_ms = RESOLUTION_FLOOR_QUANTA * quantum_ms

    all_zones = []
    for seg_a, seg_b in zip(segments, segments[1:]):
        if seg_a["offset_points"] == seg_b["offset_points"]:
            continue
        step_ms = (seg_b["offset_points"] - seg_a["offset_points"]) * quantum_ms
        # A single uncompensated step cannot exceed the duration difference (mixed-sign steps
        # can, so this flags rather than forbids).
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

    # Net step over the full segment chain (offset differences telescope to last minus first),
    # compared with the real duration difference.
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
        for _points_key, _ms_key in (("residual_rms_points", "residual_rms_ms"),
                                      ("residual_max_points", "residual_max_ms")):
            _value = result["drift_fit"].get(_points_key)
            result["drift_fit"][_ms_key] = None if _value is None else _value * quantum_ms

    return result
