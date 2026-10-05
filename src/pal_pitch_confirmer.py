"""
Confirm a speed (PAL-type) relation between two audio tracks from their pitch.

A true resample shifts pitch by the speed ratio while a time-stretch preserves it,
which the duration ratio alone cannot tell apart. The ratio is measured from the
spectrum only: average log-frequency magnitude spectra of master and candidate over
the same window, where a frequency scaling becomes a shift found by cross-correlation.
"""

import math
import numpy as np
import tools

SR = 16000
NFFT = 8192
FMIN, FMAX, NLOG = 120.0, 5200.0, 3000
TOL_ARM = 0.0030  # relative tolerance on the ratio


def _pcm(path, start_s, dur_s, audio_filter=None):
    """Decode a window to mono float PCM; None on any failure (never silence)."""
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
    """Return the normalised average magnitude spectrum on a log-frequency grid.

    On this grid a frequency scaling becomes a shift. None if the signal has no
    usable broadband structure.
    """
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
    # Remove the broad spectral tilt: codec/level dependent, it would dominate.
    s = s - np.polyval(np.polyfit(np.arange(NLOG), s, 3), np.arange(NLOG))
    nrm = np.linalg.norm(s)
    if nrm < 1e-9:
        return None
    return s / nrm


def _spectral_ratio(a, b, max_pct=8.0):
    """Return (scale factor of spectrum b relative to a, peak correlation).

    (None, None) when there is no bounded peak within +/- max_pct percent.
    """
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
        return None, None  # peak at the search edge: unbounded
    y0, y1, y2 = seg[j - 1], seg[j], seg[j + 1]
    den = (y0 - 2 * y1 + y2)
    sub = 0.0 if abs(den) < 1e-12 else 0.5 * (y0 - y2) / den
    shift = (j + sub) - lim
    return math.exp(shift * dlog), float(y1)


def measure_pitch_ratio(master_path, candidate_path, start_seconds, window_seconds=180.0):
    """Measure the candidate/master pitch ratio over one window.

    Returns:
        (ratio, peak correlation), or (None, None) when no ratio is measurable.
    """
    a = _logspec(_pcm(master_path, start_seconds, window_seconds))
    b = _logspec(_pcm(candidate_path, start_seconds, window_seconds))
    return _spectral_ratio(a, b)


def confirm_pitch(master_path, candidate_path, start_seconds, predicted_ratio,
                   window_seconds=180.0, tolerance=TOL_ARM):
    """Check the pitch-measured ratio against a duration-predicted ratio.

    Returns:
        dict with measured_ratio, peak, agrees, refusal and reason; refusal is
        "no_pitch_relation" when no ratio is measured or it disagrees beyond tolerance.
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
