'''
rate_direction.py -- THE RATE ARM: which speed relation, and which engine, if any.

RULING_20260922_ORCHESTRATOR_ARCHITECTURE.MD, ADDENDUM 15, 19(f), 21.4-6, 30 and 30.5 (owner
2026-09-25): "the named ratios (asetrate AND atempo ...) must be tried BEFORE any 'different
content' / low-similarity conclusion; a candidate that needs resampling looks like different
content at native rate (id 101: 0.68 % -> 99.97 % after asetrate), and a linear drift measured at
native rate is only a hint, not the diagnosis -- the winning ratio is the one that yields
near-full coverage with a constant offset after re-fingerprinting."

This module holds the pure half: which ratios to try first (`first_finalists`: what the declared
frame rates and the fast-drift reading name), how a finalist's alignment reads
(`finalist_reading`), and which finalist wins (`choose_winner`). The measuring half -- resample
the candidate WAV the prime already extracted, fingerprint it, align it with the prime's own
`b2_align` -- is `repair_orchestrator.rate_arm`, which owns the aligner.

WHY THE ALIGNMENT AND NOT THE SWEEP'S WINDOWS. MEASURED on id 101 (Lazarus S01E01, CH4 AAC at
25 fps against a 23.976 DDP master): the pipeline's 16-ratio sweep probed 1001/960 and scored it
0.704 on its 60 s windows (the fifteen others 0.575-0.603) -- clearly the best, and refused by the
sweep's absolute 0.9 floor, which suits same-source pairs, not a broadcast re-encode. The same
ratio re-primed on the whole track aligns 0.9997 of the master in ONE zone at a constant offset.
And the engine is measured, not assumed: chromaprint is pitch-sensitive, so the wrong engine does
not align (id 101: asetrate 0.9997, atempo 0.009; id 57: atempo one zone 2.6-1405.2 s at +3219 ms,
asetrate 0.003).
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

# A fast drift's implied ratio must sit this close (absolute) to a named rate
# to be proposed as one (ADDENDUM 19(f): "ratio ~ named rate"). 1e-3 is the
# gap between the two closest PAL names (25/24 and 1001/960), so a reading can
# land near both -- they are then BOTH proposed and the shared segments
# separate them, which is exactly what they are for.
FAST_DRIFT_NAMED_RATE_TOLERANCE = 1e-3
# ADDENDUM 19(f), verbatim: zone fit r^2 >= 0.9999. Measured on id 57 (Lazarus
# S01E05, PAL 4.27 %): 0.9999935 over 137 zones.
FAST_DRIFT_MIN_R_SQUARED = 0.9999
# A line through a handful of zones is not evidence. Same count as the
# orchestrator's LADDER_MIN_RUNGS, for the same reason (enough steps that a
# fraction of them means something).
FAST_DRIFT_MIN_NONZERO_STEPS = 8
# Half the smallest named deviation (1001/1000): below it no named rate can
# explain the drift -- the orchestrator's RATE_LADDER_MIN_FACTOR_DEVIATION.
FAST_DRIFT_MIN_FACTOR_DEVIATION = 5e-4

# THE WINNER'S EVIDENCE (ADDENDUM 30.5: "near-full coverage with a constant offset"). SPAN, not
# point density: under atempo the fingerprints of a correct match are sparser (id 57: points 0.533,
# span of its one zone 0.97 of the master), so the fraction of the master TIMELINE the aligned
# zones cover is the reading. 0.9 is the sweep's own floor, carried to the measure that decides.
RATE_ARM_MIN_SPAN_COVERAGE = 0.9
# Two finalists this close in span are tied: asetrate first (AUDIO_SPEED_POLICY's default, 23/23
# PAL and 6/6 NTSC -- at an NTSC ratio the two engines' 0.1 % pitch difference does not move a
# fingerprint, so a tie there is no evidence for atempo: MEASURED Fallout S01E03, asetrate 0.9972
# over 6 zones against atempo 0.9965 over 4, and an atempo winner the verifier then refused),
# then fewer zones, then the ratio nearer 1.
RATE_ARM_SPAN_TIE = 0.01
# A declared frame-rate ratio names a rate when it lies this close (relative) to a named one.
DECLARED_RATE_TOLERANCE = 1e-4


def _name(ratio):
    return f"{ratio.numerator}/{ratio.denominator}"


# ---------------------------------------------------------------------------
# (1) WHAT TO TRY FIRST
# ---------------------------------------------------------------------------

def _declared_rates(file_path):
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
    """The named rate the declared frame rates imply (candidate / master, the speed_ratio
    convention: a 25 fps candidate of a 23.976 master is 1001/960), or None -- equal rates, an
    unreadable side, or a ratio no named rate explains. A HINT for the first finalists, never a
    decision: declared rates lie, the alignment decides."""
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
    """The first round (ADDENDUM 30): the declared rate and every rate the fast drift names,
    each with its reciprocal -- the direction is measured, not assumed. Empty when nothing names
    a rate; the full vocabulary is the second round."""
    ordered = []
    for ratio in ([declared] if declared is not None else []) + list(drift_named or ()):
        for value in (ratio, 1 / ratio):
            if value != 1 and value not in ordered:
                ordered.append(value)
    return ordered


# ---------------------------------------------------------------------------
# (2) READING ONE FINALIST, AND THE WINNER
# ---------------------------------------------------------------------------

def finalist_reading(detail, quantum_ms, master_duration_ms, ladder):
    """One finalist's alignment as the rule reads it: `detail` the coalesced zones (the prime's
    `coalesce_same_offset_zones`), `ladder` whether the zones form a rate ladder (the
    orchestrator's `zone_ladder_signature` -- a residual rate: the wrong ratio)."""
    span = sum(max(0.0, z["master_ms"][1] - z["master_ms"][0]) for z in detail)
    offsets = [z["offset_points"] * quantum_ms for z in detail]
    return {"span_coverage": round(span / master_duration_ms, 4) if master_duration_ms else 0.0,
            "zones": len(detail),
            "offset_spread_ms": round(max(offsets) - min(offsets), 1) if offsets else None,
            "ladder": bool(ladder)}


def choose_winner(rows):
    """`rows`: one dict per finalist -- `ratio` (Fraction, 1 for the prime itself), `engine`
    (None at 1) and `finalist_reading`'s keys. The winner is the highest span coverage among the
    finalists that are not a ladder and reach RATE_ARM_MIN_SPAN_COVERAGE; within
    RATE_ARM_SPAN_TIE of it, asetrate, then fewer zones, then the ratio nearer 1. None when no
    finalist qualifies."""
    admissible = [r for r in rows if not r["ladder"]
                  and r["span_coverage"] >= RATE_ARM_MIN_SPAN_COVERAGE]
    if not admissible:
        return None
    best = max(r["span_coverage"] for r in admissible)
    tied = [r for r in admissible if best - r["span_coverage"] <= RATE_ARM_SPAN_TIE]
    return min(tied, key=lambda r: (0 if r["engine"] in (None, "asetrate") else 1, r["zones"],
                                    abs(float(r["ratio"]) - 1.0)))


def log_finalist(candidate_path, row):
    tools.log_line(f"rate_direction: finalist ratio={_name(row['ratio'])} engine={row['engine']} "
                   f"span_coverage={row['span_coverage']} zones={row['zones']} "
                   f"offset_spread_ms={row['offset_spread_ms']} ladder={row['ladder']} "
                   f"point_coverage={row.get('point_coverage')} seconds={row.get('seconds')} "
                   f"for {candidate_path}\n")


# ---------------------------------------------------------------------------
# (3) THE FAST DRIFT -- a rate the one-quantum ladder cannot see
# ---------------------------------------------------------------------------

def fast_drift_signature(zones_detail, quantum_ms):
    '''ADDENDUM 19(f): at PAL speed (4 %) the offset moves ~5 quanta per 124
    points, so the aligner never emits one-quantum rungs -- it emits steps of
    several quanta, every one of the SAME sign, and the zone offsets sit on a
    line. The orchestrator's ladder reads that as "a file with edits in it"
    (id 57: 136/136 steps negative, 128 above the resolution floor). Three
    conditions, all required:
      sign     every nonzero step between consecutive zones has one sign
      line     the zone offsets' least-squares fit reaches r^2 >= 0.9999
      named    the implied ratio lies within FAST_DRIFT_NAMED_RATE_TOLERANCE
               of a rate in the sweep's vocabulary
    The implied ratio is `1 / (1 + slope)` (offset = candidate - master point,
    speed_ratio = master / candidate duration), from the least-squares slope,
    and also from the ladder's end-to-end rise (the orchestrator's own
    reading), both reported. A named rate within tolerance of EITHER estimate is proposed.

    Returns a dict, always, with `fires` and `named_rate_candidates` (exact
    Fractions, nearest first) -- the candidates are the rate arm's first finalists
    (`first_finalists`); their re-fingerprinted alignments pick among them.'''
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
    # Near EITHER estimate: on id 57 the fit reads 1.042714 (1001/960 at 5e-6)
    # and the end-to-end rise 1.042326 (25/24 at 6.6e-4) -- proposing both
    # costs one finalist and leaves the choice to the shared segments.
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
