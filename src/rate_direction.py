'''
Rate arm: pick the speed ratio and resampling engine (asetrate or atempo) for a candidate.

Every named ratio is tried in both engines before a pair is called different content, because
a candidate that needs resampling barely aligns at its native rate. The winner is the ratio
whose re-fingerprinted alignment covers nearly the whole shared span at a constant offset;
whole-track alignment is used because windowed scores under-rate a correct ratio, and both
engines are tried because Chromaprint is pitch-sensitive.

This module is the pure half (`first_finalists`, `finalist_reading`, `choose_winner`,
`fast_drift_signature`); the measuring half is `repair_orchestrator.rate_arm`.
'''
from fractions import Fraction
import json
import subprocess

import merge_video_resample
import tools
import repair_log


# ---------------------------------------------------------------------------
# NAMED CONSTANTS
# ---------------------------------------------------------------------------

# A fast drift's implied ratio must sit this close (absolute) to a named rate to be
# proposed; 1e-3 is the gap between 25/24 and 1001/960, so both may be proposed.
FAST_DRIFT_NAMED_RATE_TOLERANCE = 1e-3
# Zone-offset line fit r^2 floor; a true rate drift fits a line almost exactly.
FAST_DRIFT_MIN_R_SQUARED = 0.9999
# Minimum nonzero offset steps for the line to count as evidence (as LADDER_MIN_RUNGS).
FAST_DRIFT_MIN_NONZERO_STEPS = 8
# Half the smallest named deviation (1001/1000): below it no named rate explains the drift.
FAST_DRIFT_MIN_FACTOR_DEVIATION = 5e-4

# Winner gate: fraction of the shared timeline spanned by aligned zones. Span, not point
# density, because atempo output fingerprints sparsely even when correctly matched.
RATE_ARM_MIN_SPAN_COVERAGE = 0.9
# Span tie threshold (unused by choose_winner, kept for callers).
RATE_ARM_SPAN_TIE = 0.01
# Among finalists passing the gate, the highest fidelity wins (two rates can span the same
# timeline while only one matches it well); fidelities this close are a tie, broken by
# asetrate first, then fewer zones, then the ratio nearer 1.
RATE_ARM_FIDELITY_TIE = 0.005
# A declared frame-rate ratio names a rate when it lies this close (relative) to a named one.
DECLARED_RATE_TOLERANCE = 1e-4


def _name(ratio):
    return f"{ratio.numerator}/{ratio.denominator}"


# ---------------------------------------------------------------------------
# (1) WHAT TO TRY FIRST
# ---------------------------------------------------------------------------

def _declared_rates(file_path):
    """Return the first video stream's declared r_frame_rate/avg_frame_rate as Fractions."""
    with repair_log.announced("rate_direction", "ffprobe", file_path) as call:
        completed = subprocess.run(
            [tools.software["ffprobe"], "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate,avg_frame_rate", "-of", "json", file_path],
            capture_output=True, text=True, timeout=120)
        call["exit"] = completed.returncode
    streams = json.loads(completed.stdout or "{}").get("streams") or [{}]
    rates = {}
    for key in ("r_frame_rate", "avg_frame_rate"):
        raw = streams[0].get(key)
        try:
            value = Fraction(raw)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if value > 0:
            rates[key] = value
    return rates


def declared_named_ratio(master_path, candidate_path):
    """Return the named rate implied by the declared frame rates (candidate / master), or None.

    None for equal rates, an unreadable side, or a ratio no named rate explains. Only a hint
    for the first finalists: declared rates can be wrong, the alignment decides.
    """
    try:
        master_rates = _declared_rates(master_path)
        candidate_rates = _declared_rates(candidate_path)
    except Exception:                                                    # noqa: BLE001
        return None
    key = next((k for k in ("r_frame_rate", "avg_frame_rate")
                if k in master_rates and k in candidate_rates), None)
    if key is None:
        return None
    implied = candidate_rates[key] / master_rates[key]
    if implied == 1:
        return None
    for named in merge_video_resample.build_rate_ratio_vocabulary():
        if abs(float(implied) / float(named) - 1.0) <= DECLARED_RATE_TOLERANCE:
            return named
    return None


def first_finalists(declared, drift_named):
    """Return the first-round ratios: the declared and drift-named rates, each with its reciprocal.

    Empty when nothing names a rate; the full vocabulary is then tried as a second round.
    """
    ordered = []
    for ratio in ([declared] if declared is not None else []) + list(drift_named or ()):
        for value in (ratio, 1 / ratio):
            if value != 1 and value not in ordered:
                ordered.append(value)
    return ordered


# ---------------------------------------------------------------------------
# (2) READING ONE FINALIST, AND THE WINNER
# ---------------------------------------------------------------------------

def finalist_reading(detail, quantum_ms, shared_ms, ladder):
    """Summarise one finalist's alignment: span coverage, zone count, offset spread, ladder flag.

    `detail` is the coalesced zone list, `ladder` whether the zones still form a rate ladder
    (a wrong ratio). `shared_ms` is the shorter of the master and the candidate at this rate,
    so a candidate that ends early is judged on what it carries.
    """
    span = sum(max(0.0, z["master_ms"][1] - z["master_ms"][0]) for z in detail)
    offsets = [z["offset_points"] * quantum_ms for z in detail]
    return {"span_coverage": round(min(1.0, span / shared_ms), 4) if shared_ms else 0.0,
            "zones": len(detail),
            "offset_spread_ms": round(max(offsets) - min(offsets), 1) if offsets else None,
            "ladder": bool(ladder)}


def fidelity(zones_detail):
    """Return the length-weighted mean `mean_match_quality` of the aligner's uncoalesced zones.

    0.0 when no zone carries a quality.
    """
    total, weight = 0.0, 0
    for zone in zones_detail or ():
        quality = zone.get("mean_match_quality")
        if quality is None:
            continue
        points = zone["master_points"][1] - zone["master_points"][0] + 1
        total += float(quality) * points
        weight += points
    return round(total / weight, 4) if weight else 0.0


def choose_winner(rows):
    """Return the winning finalist row, or None when none passes the gate.

    Each row holds `ratio`, `engine` (None at ratio 1), `fidelity` and `finalist_reading`'s keys.
    Gate: not a ladder and span coverage >= RATE_ARM_MIN_SPAN_COVERAGE. Highest fidelity wins;
    within RATE_ARM_FIDELITY_TIE, asetrate first, then fidelity, fewer zones, ratio nearer 1.
    """
    admissible = [r for r in rows if not r["ladder"]
                  and r["span_coverage"] >= RATE_ARM_MIN_SPAN_COVERAGE]
    if not admissible:
        return None
    best = max(r.get("fidelity", 0.0) for r in admissible)
    tied = [r for r in admissible if best - r.get("fidelity", 0.0) <= RATE_ARM_FIDELITY_TIE]
    return min(tied, key=lambda r: (0 if r["engine"] in (None, "asetrate") else 1,
                                    -r.get("fidelity", 0.0), r["zones"],
                                    abs(float(r["ratio"]) - 1.0)))


def log_finalist(candidate_path, row):
    tools.log_line(f"rate_direction: finalist ratio={_name(row['ratio'])} engine={row['engine']} "
                   f"span_coverage={row['span_coverage']} fidelity={row.get('fidelity')} "
                   f"zones={row['zones']} "
                   f"offset_spread_ms={row['offset_spread_ms']} ladder={row['ladder']} "
                   f"point_coverage={row.get('point_coverage')} seconds={row.get('seconds')} "
                   f"for {candidate_path}\n")


# ---------------------------------------------------------------------------
# (3) THE FAST DRIFT -- a rate the one-quantum ladder cannot see
# ---------------------------------------------------------------------------

def fast_drift_signature(zones_detail, quantum_ms):
    '''Detect a large constant speed drift (e.g. PAL 4 %) that the one-quantum ladder misses.

    At such speeds the aligner emits multi-quantum offset steps of one sign lying on a line.
    Three conditions, all required:
      sign     every nonzero step between consecutive zones has one sign
      line     the zone offsets' least-squares fit reaches r^2 >= 0.9999
      named    the implied ratio lies within FAST_DRIFT_NAMED_RATE_TOLERANCE
               of a rate in the sweep's vocabulary
    The implied ratio `1 / (1 + slope)` is computed from the least-squares slope and from the
    end-to-end rise; a named rate within tolerance of either is proposed.

    Returns a dict with `fires`, `reason` and `named_rate_candidates` (Fractions, nearest first).'''
    result = {"fires": False, "reason": None, "n_zones": len(zones_detail or []),
              "n_steps_nonzero": 0, "n_positive": 0, "n_negative": 0,
              "r_squared": None, "slope_points_per_point": None,
              "implied_ratio_fit": None, "implied_ratio_rise": None,
              "named_rate_candidates": [], "named_rate_distances": {}}
    detail = zones_detail or []
    if len(detail) < 3 or not quantum_ms:
        result["reason"] = "fewer than three zones"
        return result
    offsets = [float(z["offset_points"]) for z in detail]
    steps = [b - a for a, b in zip(offsets, offsets[1:])]
    nonzero = [s for s in steps if s != 0]
    result["n_steps_nonzero"] = len(nonzero)
    result["n_positive"] = sum(1 for s in nonzero if s > 0)
    result["n_negative"] = sum(1 for s in nonzero if s < 0)
    xs = [(z["master_points"][0] + z["master_points"][1]) / 2.0 for z in detail]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(offsets) / n
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_tot = sum((y - mean_y) ** 2 for y in offsets)
    if ss_xx == 0 or ss_tot == 0:
        result["reason"] = "flat offsets: no drift"
        return result
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, offsets)) / ss_xx
    intercept = mean_y - slope * mean_x
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, offsets))
    r_squared = 1 - ss_res / ss_tot
    span = detail[-1]["master_points"][1] - detail[0]["master_points"][0]
    rise = offsets[-1] - offsets[0]
    implied_fit = 1.0 / (1.0 + slope)
    implied_rise = 1.0 / (1.0 + rise / span) if span else None
    result.update({"r_squared": r_squared, "slope_points_per_point": slope,
                   "implied_ratio_fit": round(implied_fit, 7),
                   "implied_ratio_rise": None if implied_rise is None else round(implied_rise, 7),
                   "residual_rms_ms": round((ss_res / n) ** 0.5 * quantum_ms, 3)})
    named = sorted(merge_video_resample.build_rate_ratio_vocabulary(),
                   key=lambda r: abs(float(r) - implied_fit))
    # The two estimates can name different close rates; propose both and let alignment decide.
    estimates = [implied_fit] + ([implied_rise] if implied_rise is not None else [])
    close = [r for r in named
             if min(abs(float(r) - e) for e in estimates) <= FAST_DRIFT_NAMED_RATE_TOLERANCE]
    result["named_rate_distances"] = {_name(r): round(abs(float(r) - implied_fit), 7)
                                      for r in named[:3]}
    if len(nonzero) < FAST_DRIFT_MIN_NONZERO_STEPS:
        result["reason"] = (f"{len(nonzero)} nonzero steps, under "
                            f"{FAST_DRIFT_MIN_NONZERO_STEPS}")
    elif result["n_positive"] and result["n_negative"]:
        result["reason"] = (f"steps of both signs ({result['n_positive']} up, "
                            f"{result['n_negative']} down): edits, not a rate")
    elif r_squared < FAST_DRIFT_MIN_R_SQUARED:
        result["reason"] = f"zone fit r^2 {r_squared:.6f} under {FAST_DRIFT_MIN_R_SQUARED}"
    elif abs(implied_fit - 1.0) < FAST_DRIFT_MIN_FACTOR_DEVIATION:
        result["reason"] = f"implied ratio {implied_fit:.7f} is unity for every named rate"
    elif not close:
        result["reason"] = (f"implied ratio {implied_fit:.7f} is not within "
                            f"{FAST_DRIFT_NAMED_RATE_TOLERANCE} of a named rate "
                            f"(nearest {result['named_rate_distances']})")
    else:
        result["fires"] = True
        result["named_rate_candidates"] = close
        result["reason"] = (f"{len(nonzero)}/{len(steps)} steps one-signed, zone fit r^2 "
                            f"{r_squared:.7f}, implied ratio {implied_fit:.7f} (end-to-end "
                            f"{implied_rise:.7f}) -> named {[_name(r) for r in close]}")
    tools.dev_log(f"rate_direction: fast_drift_signature fires={result['fires']} "
                  f"{result['reason']}\n")
    return result
