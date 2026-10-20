"""Diagnostics read from a `banded_seed_alignment.b2_align` result.

Two checks that are not part of the alignment itself:
  * the duration budget: a step larger than the files' duration difference is flagged
    `exceeds_duration_budget` instead of `cut`, and the net step is compared with the real
    duration difference (`residual_ms`, `residual_fraction`);
  * the drift trace: the best shift every few points and its line fit, a hint for the
    resample sweep (a steady slope suggests a rate relation, a staircase a cut).

`align` runs the alignment and both diagnostics, as the repair pipeline uses them.
"""
import banded_seed_alignment
from banded_seed_alignment import HAMMING_MATCH_THRESHOLD, MODALITY, sim

# Re-centering trace: checkpoint every X points (>= 2 per PAL-rate drift step), search radius
# M points, and a re-center needs at least this many matched words so noise is never chased.
TRACE_X_POINTS = 10
TRACE_M_POINTS = 3
TRACE_MIN_SUSTAINED_WORDS = 3


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


def apply_duration_budget(result, duration_diff_ms=None, signed_duration_diff_ms=None,
                          shorter_duration_ms=None):
    """Flag steps larger than the duration difference and measure the net-step residual.

    A single uncompensated step cannot exceed the duration difference (mixed-sign steps can,
    so this flags rather than forbids). Updates `result` in place: zone classifications,
    `cut_zones`, the verdict, `residual_ms` and `residual_fraction`.

    Args:
        duration_diff_ms: |master - candidate| duration, or None.
        signed_duration_diff_ms, shorter_duration_ms: both needed for the residual.
    """
    all_zones = result.get("all_zones")
    if all_zones is None:
        return result
    for zone in all_zones:
        zone["duration_diff_ms"] = duration_diff_ms
        if (zone["classification"] == "cut" and duration_diff_ms is not None
                and abs(zone["step_ms"]) > duration_diff_ms):
            zone["classification"] = "exceeds_duration_budget"
    cut_zones = [zone for zone in all_zones if zone["classification"] == "cut"]
    result["cut_zones"] = cut_zones
    result["verdict"] = (banded_seed_alignment.VERDICT_SEGMENTS_FOUND if cut_zones
                         else banded_seed_alignment.VERDICT_SINGLE_SEGMENT_NO_CUT)
    # Net step over the full segment chain (offset differences telescope to last minus first),
    # compared with the real duration difference.
    segments = result.get("segments") or []
    if signed_duration_diff_ms is not None and shorter_duration_ms and len(segments) >= 2:
        signed_step_sum_ms = ((segments[-1]["offset_points"] - segments[0]["offset_points"])
                              * result["quantum_ms"])
        residual_ms = signed_step_sum_ms - signed_duration_diff_ms
        result["residual_ms"] = residual_ms
        result["residual_fraction"] = abs(residual_ms) / shorter_duration_ms
    return result


def attach_drift_trace(result, fp_master, fp_candidate):
    """Add `drift_trace` and `drift_fit` (with ms residuals) to a completed alignment."""
    if result.get("all_zones") is None:
        return result
    quantum_ms = result["quantum_ms"]
    trace = best_shift_trace(fp_master, fp_candidate)
    for entry in trace:
        entry["master_time_s"] = entry["master_index"] * quantum_ms / 1000.0
    result["drift_trace"] = trace
    result["drift_fit"] = fit_trace_slope(trace)
    for points_key, ms_key in (("residual_rms_points", "residual_rms_ms"),
                               ("residual_max_points", "residual_max_ms")):
        value = result["drift_fit"].get(points_key)
        result["drift_fit"][ms_key] = None if value is None else value * quantum_ms
    return result


def align(fp_master, fp_candidate, quantum_ms, candidate_quantum_ms=None, duration_diff_ms=None,
          signed_duration_diff_ms=None, shorter_duration_ms=None, deadline=None):
    """Run `b2_align`, then the duration budget and the drift trace on its result."""
    result = banded_seed_alignment.b2_align(fp_master, fp_candidate, quantum_ms,
                                            candidate_quantum_ms=candidate_quantum_ms,
                                            deadline=deadline)
    apply_duration_budget(result, duration_diff_ms, signed_duration_diff_ms,
                          shorter_duration_ms)
    return attach_drift_trace(result, fp_master, fp_candidate)
