"""splice_hygiene.py -- the gain and the join of a fill piece in a CANDIDATE track (ADDENDUM 23.2 /
23.3, owner 2026-09-24; memo architect/drafts/SPLICE_HYGIENE_RESEARCH_20260924.md).

A fill piece (master material written into a candidate track to cover a hole) is:
  * GAIN-ALIGNED on the track that receives it: at each splice, BS.1770 loudness of 400 ms blocks
    (hop 100 ms, K-weighted) over the SPLICE_MEASURE_S of common material next to it, the
    receiving candidate track at its own offset against the fill source at master time; only the
    blocks where both carry the same material count (waveform NCC >= BLOCK_MIN_NCC within a
    +/- BLOCK_LAG_S search, both above BLOCK_MIN_LUFS); d = median(L_receiving - L_source);
    |d| < GAIN_DEADBAND_DB -> 0. Two edges more than GAIN_DEADBAND_DB apart (a dub: its M&E is
    not a flat gain) -> a linear ramp in dB from one to the other; an edge without
    MIN_COHERENT_BLOCKS coherent blocks takes the other's d; neither -> 0 dB and
    `fill_gain_unmeasurable`. Never loudnorm / dynaudnorm (they change the dynamics).
  * JOINED by a 10 ms triangular crossfade (`acrossfade=d=0.010:o=1:c1=tri:c2=tri`), the fill
    taking 10 ms MORE on that side so the duration is exact; a side that is digital silence
    (< SILENCE_DBFS around the splice) is a hard cut, no fade, no margin (a fade there blunts
    the content's own attack -- errid-70's head).
The candidate's own pieces are never touched; the master's tracks are never touched (general law).
This module is pure: it measures arrays and decides; `merge_video_chimeric` reads and builds.
"""
import numpy as np

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
    from scipy.signal import lfilter
    return lfilter(_RLB_B, _RLB_A, lfilter(_SHELF_B, _SHELF_A, np.asarray(x, np.float64)))


def rms_dbfs(x):
    x = np.asarray(x, np.float64)
    return -200.0 if not len(x) else 20 * np.log10(max(float(np.sqrt(np.mean(x * x))), 1e-10))


def _ncc_max(ref, seg):
    """Max normalised cross-correlation of `ref` sliding over `seg` (len(seg) >= len(ref))."""
    n, m = len(ref), len(seg)
    ref = ref - ref.mean()
    norm = np.sqrt((ref ** 2).sum())
    if norm < 1e-9:
        return 0.0
    size = 1 << int(np.ceil(np.log2(n + m)))
    corr = np.fft.irfft(np.fft.rfft(seg, size) * np.conj(np.fft.rfft(ref, size)), size)[:m - n + 1]
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    cs2 = np.concatenate([[0.0], np.cumsum(seg * seg)])
    s1 = cs[n:] - cs[:-n]
    var = np.maximum((cs2[n:] - cs2[:-n]) - s1 * s1 / n, 1e-12)
    return float((corr / (norm * np.sqrt(var))).max())


def edge_gain(receiving, source, rate=MEASURE_RATE):
    """(d_db or None, n_coherent_blocks): the receiving track's level minus the fill source's,
    over the blocks where both carry the same material. Both arrays cover the same master span."""
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
    """The piece's gain from its two edges' readings (None = unmeasured / no such edge):
    `{"mode": none|flat|ramp, "gain_a_db", "gain_b_db", "unmeasurable"}`."""
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
    """`crossfade` when both sides carry sound over +/- FADE_S around the splice, else
    `hard_cut` (digital silence on either side, or a side not read)."""
    if receiving_edge is None or source_edge is None or not len(receiving_edge) \
            or not len(source_edge):
        return "hard_cut"
    if rms_dbfs(receiving_edge) < SILENCE_DBFS or rms_dbfs(source_edge) < SILENCE_DBFS:
        return "hard_cut"
    return "crossfade"


def volume_filter(gain, duration_s):
    """The ffmpeg `volume` for a piece of `duration_s` seconds (its margins included), or ''."""
    if gain["mode"] == "flat":
        return f",volume={gain['gain_a_db']:.3f}dB"
    if gain["mode"] == "ramp":
        a, b = gain["gain_a_db"], gain["gain_b_db"]
        return (f",volume='pow(10\\,({a:.3f}+({b - a:.3f})*t/{duration_s:.6f})/20)'"
                f":eval=frame")
    return ""
