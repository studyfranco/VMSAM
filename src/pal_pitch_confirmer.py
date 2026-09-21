"""
Stage 3 confirmer 1 (PITCH) of the PAL/speed family design
(DESIGN_PAL_SPEED_FAMILY_20260916.MD).

Ported from VMSAM_HELP_AI/tools/t119_pal_pitch_discriminator.py (dev-2, lab
instrument, sealed acceptance in lab/SEALED_t119_pal_two_arm.md) -- the core
algorithm (`pcm`/`logspec`/`ratio` there) only. The TSV-driven lab harness,
its hardcoded table path, and its sealed control battery are NOT carried
over: production takes real file paths and a real predicted ratio as
arguments, never a table row. Binaries go through `tools.software`, per
AGENT.MD -- the lab version's literal "ffmpeg" is a lab convenience, not a
production one.

A true resample shifts pitch by the ratio; a time-stretch preserves it.
Duration ratio alone cannot separate the two -- this confirmer owes nothing
to a duration header at all, which is the entire reason Stage 3 exists as a
second, independent instrument rather than trusting Stage 2 alone.

Refusal token per the standing ruling: a pitch-based refusal is
`no_pitch_relation`, never `no_rate_relation`.
"""

import math
import numpy as np
import tools

SR = 16000
NFFT = 8192
FMIN, FMAX, NLOG = 120.0, 5200.0, 3000
TOL_ARM = 0.0030  # t119's own sealed tolerance, ported verbatim


def _pcm(path, start_s, dur_s, audio_filter=None):
    """Decode mono PCM. None on ANY failure -- a decode that did not happen
    is not silence, and the caller must keep the distinction (t119's own
    rule, ported)."""
    cmd = [tools.software["ffmpeg"], "-v", "error", "-nostdin",
           "-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}", "-i", path,
           "-vn", "-ac", "1", "-ar", str(SR)]
    if audio_filter:
        cmd += ["-af", audio_filter]
    cmd += ["-f", "s16le", "-"]
    stdout, stderror, exit_code = tools.launch_cmdExt_no_test(cmd)
    if exit_code != 0 or len(stdout) < SR * 4:
        return None
    return np.frombuffer(stdout, dtype="<i2").astype(np.float64) / 32768.0


def _logspec(x):
    """Average magnitude spectrum on a CONSTANT-dlog(f) grid, so a frequency
    SCALING becomes a SHIFT. None if the material has no usable broadband
    structure."""
    if x is None or len(x) < NFFT * 4:
        return None
    n = (len(x) // NFFT) * NFFT
    fr = x[:n].reshape(-1, NFFT) * np.hanning(NFFT)
    mag = np.abs(np.fft.rfft(fr, axis=1)).mean(axis=0)
    f = np.fft.rfftfreq(NFFT, 1.0 / SR)
    lg = np.linspace(math.log(FMIN), math.log(FMAX), NLOG)
    s = np.interp(np.exp(lg), f, mag)
    if not np.isfinite(s).all() or s.max() <= 0:
        return None
    s = np.log(s + 1e-12)
    s = s - s.mean()
    # remove the broad spectral tilt: codec/level dependent, would dominate
    s = s - np.polyval(np.polyfit(np.arange(NLOG), s, 3), np.arange(NLOG))
    nrm = np.linalg.norm(s)
    if nrm < 1e-9:
        return None
    return s / nrm


def _spectral_ratio(a, b, max_pct=8.0):
    """Scale factor k with candidate spectrum b relative to master a. None
    if no clear peak."""
    if a is None or b is None:
        return None, None
    dlog = (math.log(FMAX) - math.log(FMIN)) / (NLOG - 1)
    lim = int(math.log(1 + max_pct / 100.0) / dlog) + 2
    cc = np.correlate(b, a, mode="full")
    mid = len(a) - 1
    seg = cc[mid - lim: mid + lim + 1]
    if len(seg) < 3:
        return None, None
    j = int(np.argmax(seg))
    if j == 0 or j == len(seg) - 1:
        return None, None  # peak at the edge: unbounded
    y0, y1, y2 = seg[j - 1], seg[j], seg[j + 1]
    den = (y0 - 2 * y1 + y2)
    sub = 0.0 if abs(den) < 1e-12 else 0.5 * (y0 - y2) / den
    shift = (j + sub) - lim
    return math.exp(shift * dlog), float(y1)


def measure_pitch_ratio(master_path, candidate_path, start_seconds, window_seconds=180.0):
    """Returns (ratio_or_None, peak_correlation_or_None)."""
    a = _logspec(_pcm(master_path, start_seconds, window_seconds))
    b = _logspec(_pcm(candidate_path, start_seconds, window_seconds))
    return _spectral_ratio(a, b)


NTSC_NOMINAL = 1001.0 / 1000.0

# PROVISIONAL, M3 (RULING_20260921_NTSC_KNIFE_EDGE.MD;
# VMSAM_HELP_AI/dev-pal/012-ntsc-knife-edge.MD): duration cannot carry this
# signal -- a real cut confounds it by 40-60x its own size (measured on the
# 8 census ids: duration ratio 0.935-0.955, nowhere near 1.001) -- so this
# tolerance is for an UNCONDITIONAL pitch measurement, never a duration-
# derived prediction. Measured from n=24 (3 probe positions x 8 real
# production episodes, folder:164): min 1.000775, max 1.001189, mean
# 1.001031, median 1.001048, stdev 9.79e-5, max deviation from the
# theoretical nominal 0.000225. 0.0005 is ~2.2x that max deviation and ~5x
# the stdev -- a defensible margin, not the measured value itself, same
# shape as `pal_saturation_screen.CHROMAPRINT_FIXED_STARTUP_POINTS`'s
# sample-rate limit. SCOPE LIMIT, same reason: all 8 ids are ONE series
# (one release pair's encode characteristics), not 8 independent
# observations -- if the true population spreads wider than this one
# series, 0.0005 may be too narrow, flagged for the Architect exactly as
# the earlier constants were.
NTSC_TOLERANCE = 0.0005


def confirm_ntsc(master_path, candidate_path, start_seconds, window_seconds=180.0,
                  tolerance=NTSC_TOLERANCE):
    """Unconditional NTSC recognizer (M3): measures pitch directly and checks
    it against the NTSC nominal and its reciprocal -- NEVER against a
    duration-derived prediction, because M3 measured that duration cannot
    carry this signal when a real cut confounds it (`pal_speed_discriminator`
    routes these pairs to `pal_inverse` or `no_band`, both wrong, by
    coincidence of where the cut lands the ratio).

    Mirrors `pal_speed_discriminator.classify_band`'s own direct/inverse
    symmetry for PAL, at the NTSC nominal instead.

    Returns {measured_ratio, peak, matched, predicted_ratio, refusal,
    reason}. `matched` is "ntsc_direct", "ntsc_inverse", or None.
    `predicted_ratio`, when not None, is the CONFIRMED ratio a caller should
    build an undo filter from -- never the discriminator's own duration
    ratio, which this function's whole reason to exist is to bypass.
    `refusal`, when not None, is `no_ntsc_relation` -- never forces, same
    discipline as `confirm_pitch`."""
    k, peak = measure_pitch_ratio(master_path, candidate_path, start_seconds, window_seconds)
    if k is None:
        return {"measured_ratio": None, "peak": None, "matched": None,
                "predicted_ratio": None, "refusal": "no_ntsc_relation",
                "reason": "no bounded spectral peak in the correlation"}
    inverse_nominal = 1.0 / NTSC_NOMINAL
    if abs(k - NTSC_NOMINAL) <= tolerance * NTSC_NOMINAL:
        return {"measured_ratio": round(k, 6), "peak": round(peak, 4), "matched": "ntsc_direct",
                "predicted_ratio": NTSC_NOMINAL, "refusal": None, "reason": None}
    if abs(k - inverse_nominal) <= tolerance * inverse_nominal:
        return {"measured_ratio": round(k, 6), "peak": round(peak, 4), "matched": "ntsc_inverse",
                "predicted_ratio": inverse_nominal, "refusal": None, "reason": None}
    return {"measured_ratio": round(k, 6), "peak": round(peak, 4), "matched": None,
            "predicted_ratio": None, "refusal": "no_ntsc_relation",
            "reason": (f"pitch-measured ratio {k:.6f} matches neither the NTSC nominal "
                       f"{NTSC_NOMINAL:.6f} nor its reciprocal {inverse_nominal:.6f} "
                       f"within {tolerance}")}


def confirm_pitch(master_path, candidate_path, start_seconds, predicted_ratio,
                   window_seconds=180.0, tolerance=TOL_ARM):
    """Stage 3, confirmer 1. Compares the pitch-measured ratio against
    Stage 2's predicted (duration-measured) ratio.

    Returns {measured_ratio, peak, agrees, refusal, reason}. `refusal`, when
    not None, is the standing token `no_pitch_relation` -- never
    `no_rate_relation`, the design's own naming rule -- with a stated reason.
    Never forces: disagreement is a refusal, not a downgrade of the
    prediction.
    """
    k, peak = measure_pitch_ratio(master_path, candidate_path, start_seconds, window_seconds)
    if k is None:
        return {"measured_ratio": None, "peak": None, "agrees": False,
                "refusal": "no_pitch_relation",
                "reason": "no bounded spectral peak in the correlation"}
    agrees = abs(k - predicted_ratio) <= tolerance * predicted_ratio
    if not agrees:
        return {"measured_ratio": round(k, 6), "peak": round(peak, 4), "agrees": False,
                "refusal": "no_pitch_relation",
                "reason": (f"pitch-measured ratio {k:.6f} does not match the "
                           f"duration-predicted {predicted_ratio:.6f} within "
                           f"{tolerance * 100:.2f}%")}
    return {"measured_ratio": round(k, 6), "peak": round(peak, 4), "agrees": True,
            "refusal": None, "reason": None}
