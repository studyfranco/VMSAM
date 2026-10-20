"""Gain alignment and splice joins for fill pieces written into a candidate track.

A fill piece is master material that covers a hole in a candidate track.

* Gain: at each splice, BS.1770 K-weighted loudness of 400 ms blocks (100 ms hop) is compared
  between the receiving track and the fill source over SPLICE_MEASURE_S of common material.
  Only blocks carrying the same material count (NCC >= BLOCK_MIN_NCC within +/- BLOCK_LAG_S,
  both above BLOCK_MIN_LUFS); the gain is the median difference, zeroed inside
  GAIN_DEADBAND_DB. Edges that disagree get a linear dB ramp (a dub's M&E is not a flat gain);
  an unmeasurable edge borrows the other's value. No loudnorm/dynaudnorm: they alter dynamics.
* Join: a 10 ms triangular crossfade, the fill extended by 10 ms so durations stay exact; a
  side in digital silence gets a hard cut (a fade would blunt the following attack).

Pure functions: this module measures arrays and decides; merge_video_chimeric reads and builds.
"""
import numpy as np

from audio_walk import _ncc_curve

MEASURE_RATE = 48000
SPLICE_MEASURE_S = 10.0
BLOCK_S = 0.4
BLOCK_HOP_S = 0.1
BLOCK_MIN_NCC = 0.9
BLOCK_LAG_S = 0.05
BLOCK_MIN_LUFS = -60.0
MIN_COHERENT_BLOCKS = 5
GAIN_DEADBAND_DB = 0.5
FADE_S = 0.010
SILENCE_DBFS = -90.0

# BS.1770-4 K-weighting at 48 kHz: the high-shelf "pre-filter" then the RLB high-pass.
_SHELF_B = (1.53512485958697, -2.69169618940638, 1.19839281085285)
_SHELF_A = (1.0, -1.69065929318241, 0.73248077421585)
_RLB_B = (1.0, -2.0, 1.0)
_RLB_A = (1.0, -1.99004745483398, 0.99007225036621)


def k_weight(x):
    """Apply BS.1770 K-weighting (48 kHz coefficients) to a signal."""
    from scipy.signal import lfilter
    return lfilter(_RLB_B, _RLB_A, lfilter(_SHELF_B, _SHELF_A, np.asarray(x, np.float64)))


def rms_dbfs(x):
    """RMS level of a signal in dBFS (-200.0 for an empty signal)."""
    x = np.asarray(x, np.float64)
    return -200.0 if not len(x) else 20 * np.log10(max(float(np.sqrt(np.mean(x * x))), 1e-10))


def _ncc_max(ref, seg):
    """Max normalised cross-correlation of ref sliding over seg; 0.0 when ref is flat."""
    curve = _ncc_curve(ref, seg)
    return 0.0 if curve is None else float(curve.max())


def edge_gain(receiving, source, rate=MEASURE_RATE):
    """Measure the loudness difference between the receiving track and the fill source.

    Both arrays cover the same master span; only blocks where both carry the same material
    are used.

    Returns:
        (d_db, n_coherent_blocks); d_db is the median receiving-minus-source difference in
        LUFS, or None with fewer than MIN_COHERENT_BLOCKS coherent blocks.
    """
    n = min(len(receiving), len(source))
    receiving = np.asarray(receiving[:n], np.float64)
    source = np.asarray(source[:n], np.float64)
    kr, ks = k_weight(receiving), k_weight(source)
    block, hop, lag = int(BLOCK_S * rate), int(BLOCK_HOP_S * rate), int(BLOCK_LAG_S * rate)
    diffs = []
    for start in range(lag, n - block - lag + 1, hop):
        lr = -0.691 + 10 * np.log10(max(float(np.mean(kr[start:start + block] ** 2)), 1e-20))
        ls = -0.691 + 10 * np.log10(max(float(np.mean(ks[start:start + block] ** 2)), 1e-20))
        if lr < BLOCK_MIN_LUFS or ls < BLOCK_MIN_LUFS:
            continue
        if _ncc_max(source[start:start + block], receiving[start - lag:start + block + lag]) \
                < BLOCK_MIN_NCC:
            continue
        diffs.append(lr - ls)
    if len(diffs) < MIN_COHERENT_BLOCKS:
        return None, len(diffs)
    return float(np.median(diffs)), len(diffs)


def fill_gain(d_a, d_b):
    """Decide a fill piece's gain from its two edge readings (None: unmeasured or no edge).

    Returns:
        dict with mode ("none", "flat" or "ramp"), gain_a_db, gain_b_db and unmeasurable.
    """
    if d_a is None and d_b is None:
        return {"mode": "none", "gain_a_db": 0.0, "gain_b_db": 0.0, "unmeasurable": True}
    d_a = d_b if d_a is None else d_a
    d_b = d_a if d_b is None else d_b
    if abs(d_a - d_b) > GAIN_DEADBAND_DB:
        return {"mode": "ramp", "gain_a_db": round(d_a, 3), "gain_b_db": round(d_b, 3),
                "unmeasurable": False}
    flat = (d_a + d_b) / 2.0
    if abs(flat) < GAIN_DEADBAND_DB:
        return {"mode": "none", "gain_a_db": 0.0, "gain_b_db": 0.0, "unmeasurable": False}
    return {"mode": "flat", "gain_a_db": round(flat, 3), "gain_b_db": round(flat, 3),
            "unmeasurable": False}


def splice_join(receiving_edge, source_edge):
    """Return "crossfade" when both sides carry sound around the splice, else "hard_cut".

    A side in digital silence (below SILENCE_DBFS) or not read gives a hard cut.
    """
    if receiving_edge is None or source_edge is None or not len(receiving_edge) \
            or not len(source_edge):
        return "hard_cut"
    if rms_dbfs(receiving_edge) < SILENCE_DBFS or rms_dbfs(source_edge) < SILENCE_DBFS:
        return "hard_cut"
    return "crossfade"


def volume_filter(gain, duration_s):
    """Return the ffmpeg volume filter suffix for a piece of duration_s seconds, or ''."""
    if gain["mode"] == "flat":
        return f",volume={gain['gain_a_db']:.3f}dB"
    if gain["mode"] == "ramp":
        a, b = gain["gain_a_db"], gain["gain_b_db"]
        return (f",volume='pow(10\\,({a:.3f}+({b - a:.3f})*t/{duration_s:.6f})/20)'"
                f":eval=frame")
    return ""
