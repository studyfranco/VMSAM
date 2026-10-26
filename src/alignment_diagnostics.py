"""Duration-budget diagnostics read from a `banded_seed_alignment.b2_align` result."""
import banded_seed_alignment


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


def align(fp_master, fp_candidate, quantum_ms, candidate_quantum_ms=None, duration_diff_ms=None,
          signed_duration_diff_ms=None, shorter_duration_ms=None):
    """Run `b2_align`, then the duration budget on its result."""
    result = banded_seed_alignment.b2_align(fp_master, fp_candidate, quantum_ms,
                                            candidate_quantum_ms=candidate_quantum_ms)
    return apply_duration_budget(result, duration_diff_ms, signed_duration_diff_ms,
                                 shorter_duration_ms)
