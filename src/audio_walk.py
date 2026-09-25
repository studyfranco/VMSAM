# -*- coding: utf-8 -*-
"""
audio_walk.py -- THE MILLISECOND WALK IN THE WAVEFORM (RULING_20260922_ORCHESTRATOR_ARCHITECTURE.MD,
ADDENDUM 25 points 1 and 2; experiment architect/drafts/MS_WALK_EXPERIMENT_20260925.md, prototype
scratchpad/mswalk/ms_walk.py, "HYPOTHESE TENUE 5/5").

WHY IT EXISTS. The b2 aligner sees at one fingerprint quantum (123.8 ms); a delivery is judged at
one millisecond. The independent certification of stages 4+5 measured what falls between the
two: sub-quantum steps inside a zone outvoted and shipped (B2: 66.75 / 33.375 / 25.02 ms), fills
placed at the picture's frame ~1 s from the audio edit (B3), and a fill 8.76 s wide for a 5.005 s
insert (B1). This module measures the audio itself: a sliding normalised cross-correlation over
the WHOLE comparison track, seeded by the b2 zones' offsets, then the change points between its
constant-offset levels, then their edges at 20 ms.

CONVENTION: every track is read mono at `WALK_RATE` on its FILE clock (the stream's container
`start_time` prepended as zeros, times the rate factor on a rate pair), and an offset `o` means
`candidate(t + o) == master(t)`, in ms -- the file-time offset the orchestrator carries.

WHAT IT RETURNS, NEVER DECIDES. Levels, change points, edges, intervals and the quietest instant
in an interval. The orchestrator decides what to do with them (the video picks the frame inside
an interval; a slip; a fill of exactly the measured width).

Parameters are the memo's retained ones, each named below with its measurement.
"""
from decimal import Decimal

import numpy as np

import tools

# THE RATE. 16 kHz: the experiment's rate, at which the fine 20 ms edges are 320 samples and the
# sub-sample parabolic peak resolves to ~0.01 ms. Every number in the memo was measured at it.
WALK_RATE = 16000

# THE WALK: window 2 s, hop 1 s, search +/-150 ms around each seed (memo section 3: 0 false jumps
# on 5 pairs from 0.5 to 10 s windows; W=2/H=1 costs 29-59 s per hour of audio; +/-150 ms covers
# the b2 quantum's uncertainty on each seed with one quantum of margin).
WALK_WINDOW_S = 2.0
WALK_HOP_S = 1.0
WALK_SEARCH_MS = 150.0
# A window's master content under this level is UNMEASURABLE -- no offset is read there (memo
# section 1.2). This is the WALK's floor only; the fine edges read down to digital silence.
WALK_SILENT_DB = -55.0
# A measured window: NCC at least this, and its second peak (more than 3 ms away) under
# PEAK_AMBIGUITY of the main one; otherwise `unmatched` / `ambiguous`, never an offset.
MATCH_NCC = 0.5
PEAK_AMBIGUITY = 0.92
# LEVELS: a window joins a level within LEVEL_TOLERANCE_MS of its running median; a new level
# must hold LEVEL_PERSIST consecutive windows or its windows are outliers.
LEVEL_TOLERANCE_MS = 5.0
LEVEL_PERSIST = 2
# THE CHANGE POINT (ADDENDUM 25.2, owner): a jump of at least this between levels. Measured: the
# clean products land at <= 10.5 ms, the real sub-quantum steps at 25-67 ms (certification).
JUMP_MS = 15.0
# COARSE LOCALISATION: fixed-lag 0.4 s windows every 10 ms around the change, a side is claimed
# at NCC >= 0.8 with a 0.2 margin over the other.
SHORT_WIN_S, SHORT_HOP_S, SHORT_THRESHOLD, SHORT_MARGIN = 0.4, 0.01, 0.8, 0.2
# FINE EDGES: residual of a gain-fitted 20 ms window every 5 ms (memo 1.4). Only DIGITAL silence
# is unmeasurable at this scale (a -55 dB floor pulled the Tougen edges 0.1-0.3 s inward, memo
# 4.2); a residual floor over FINE_FLOOR_MAX means the waveforms are not sample-identical (a rate
# pair) and the edges fall back to 100 ms NCC windows.
FINE_WIN_S, FINE_HOP_S, FINE_THRESHOLD = 0.02, 0.005, 0.25
FINE_SILENT_DB = -100.0
FINE_FLOOR_MAX = 0.1
# The master-only sound a fill MUST cover (20 ms window level, memo 1.4).
AUDIBLE_DB = -60.0


# ---------------------------------------------------------------------------------------------
# DECODE
# ---------------------------------------------------------------------------------------------

# THE SPEECH ENVELOPE (ADDENDUM 30): under `atempo` (WSOLA) the waveform is rebuilt from overlapped
# grains, so waveform NCC loses its grip while the speech envelope survives. MEASURED on id 57 (fr,
# atempo 1001/960, 13 positions, 8 s windows): envelope 3245.5-3248.9 ms at corr 0.82-0.98;
# waveform 3242-3258 ms at corr 0.17-0.48 (the walk found 71 matched windows of 1451, the plan
# landed 21.1 ms off). Band 300-1800 Hz, 5 ms RMS, linearly interpolated back to the reading's
# rate so every instrument downstream reads it like a waveform (their NCC removes the mean).
ENVELOPE_BAND_HZ = (300.0, 1800.0)
ENVELOPE_HOP_S = 0.005


class EnvelopeSignal(np.ndarray):
    """A track read as its speech envelope (`read_on_file_clock(..., envelope=True)`). Its type
    is how `_search` knows the correlation peak is broad: the second peak is sought outside the
    whole main lobe, not outside +/-3 ms (a waveform's peak is a fraction of a period wide, an
    envelope's is tens of ms -- MEASURED id 300: 1382 of 1443 windows read `ambiguous` on the
    +/-3 ms guard)."""


def speech_envelope(samples, rate):
    """`samples` (mono, at `rate`) -> its speech envelope, same length and rate."""
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(samples, np.float64)
    hop = max(1, int(round(rate * ENVELOPE_HOP_S)))
    count = len(x) // hop
    if count < 2:
        return np.zeros(len(x), np.float32)
    sos = butter(4, ENVELOPE_BAND_HZ, btype="band", fs=rate, output="sos")
    band = sosfiltfilt(sos, x)
    rms = np.sqrt(np.mean(band[:count * hop].reshape(count, hop) ** 2, axis=1))
    centres = (np.arange(count) + 0.5) * hop
    return np.interp(np.arange(len(x)), centres, rms).astype(np.float32)


def read_on_file_clock(video_obj, audio, audio_filter=None, scale=Decimal(1), deadline=None,
                       envelope=False):
    """One track, mono float32 at WALK_RATE, on the FILE clock: `merge_video_chimeric.
    read_track_samples` (bounded, logged, the assembly's own convention) with the stream's
    container start_time -- times `scale` on a rate pair, as the assembly reads it -- prepended
    as zeros (a negative start drops samples). `deadline`: the repair's budget bounds the read
    (`read_track_samples`). `envelope`: the track's speech envelope instead of its waveform (an
    atempo pair -- see `speech_envelope`)."""
    import merge_video_chimeric
    samples = merge_video_chimeric.read_track_samples(
        video_obj.filePath, int(audio["StreamOrder"]), WALK_RATE, audio_filter=audio_filter,
        deadline=deadline)
    if envelope:
        samples = speech_envelope(samples, WALK_RATE)
    start_ms = merge_video_chimeric.get_stream_start_ms(audio) * Decimal(scale)
    pad = int(round(float(start_ms) * WALK_RATE / 1000.0))
    if pad > 0:
        samples = np.concatenate([np.zeros(pad, np.float32), samples])
    elif pad < 0:
        samples = samples[-pad:]
    return samples.view(EnvelopeSignal) if envelope else samples


def rms_db(x):
    if len(x) == 0:
        return -200.0
    value = float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))
    return 20 * np.log10(max(value, 1e-10))


# ---------------------------------------------------------------------------------------------
# CROSS-CORRELATION
# ---------------------------------------------------------------------------------------------

def _ncc_curve(ref, seg):
    """NCC of `ref` (n samples) at every lag inside `seg` (m >= n): m - n + 1 values."""
    n, m = len(ref), len(seg)
    ref = ref - ref.mean()
    norm = np.sqrt((ref ** 2).sum())
    if norm < 1e-9:
        return None
    size = 1 << int(np.ceil(np.log2(n + m)))
    corr = np.fft.irfft(np.fft.rfft(seg, size) * np.conj(np.fft.rfft(ref, size)), size)[:m - n + 1]
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    cs2 = np.concatenate([[0.0], np.cumsum(seg * seg)])
    s1 = cs[n:] - cs[:-n]
    var = np.maximum((cs2[n:] - cs2[:-n]) - s1 * s1 / n, 1e-12)
    return corr / (norm * np.sqrt(var))


def _peak(curve):
    """Sub-sample parabolic peak: (fractional index, value, integer index)."""
    k = int(np.argmax(curve))
    frac = 0.0
    if 0 < k < len(curve) - 1:
        a, b, c = curve[k - 1], curve[k], curve[k + 1]
        d = a - 2 * b + c
        if abs(d) > 1e-12:
            frac = 0.5 * (a - c) / d
    return k + frac, float(curve[k]), k


def _search(m, c, t, off_ms, win_s, search_ms):
    """Best offset for master window [t, t + win) within off_ms +/- search_ms, or None."""
    n = int(win_s * WALK_RATE)
    i0 = int(round(t * WALK_RATE))
    span = int(search_ms * WALK_RATE / 1000)
    j0 = i0 + int(round(off_ms * WALK_RATE / 1000)) - span
    lo, hi = max(j0, 0), min(j0 + n + 2 * span, len(c))
    if hi - lo < n + 2:
        return None
    seg = c[lo:hi].astype(np.float64)
    if rms_db(seg) < WALK_SILENT_DB:
        return {"cand_silent": True}
    curve = _ncc_curve(m[i0:i0 + n].astype(np.float64), seg)
    if curve is None:
        return None
    kf, value, k = _peak(curve)
    guard = int(0.003 * WALK_RATE)
    low, high = max(k - guard, 0), k + guard + 1
    if isinstance(m, EnvelopeSignal):
        # the whole main lobe: out to the first local minimum on each side
        low, high = k, k + 1
        while low > 0 and curve[low - 1] <= curve[low]:
            low -= 1
        while high < len(curve) and curve[high] <= curve[high - 1]:
            high += 1
    rest = np.concatenate([curve[:low], curve[high:]])
    second = float(rest.max()) if len(rest) else 0.0
    return {"off": (lo + kf - i0) * 1000.0 / WALK_RATE, "ncc": value, "second": second}


def _status(result):
    if result["ncc"] < MATCH_NCC:
        return "unmatched"
    if result["second"] > PEAK_AMBIGUITY * result["ncc"]:
        return "ambiguous"
    return "ok"


# ---------------------------------------------------------------------------------------------
# THE WALK
# ---------------------------------------------------------------------------------------------

# A GAP BETWEEN TWO MEASURED OFFSETS IS SEARCHED WHERE ITS CONTENT CAN ONLY BE: after the walk,
# every run of audible unmeasured windows between two `ok` windows whose offsets differ by at
# least JUMP_MS -- and before the first / after the last `ok` window -- is searched for common
# content inside the CANDIDATE span those two windows leave between them (an edit keeps the order
# of what both files carry). A deletion leaves no candidate span and an addition no master gap, so
# this searches only where both sides carry something between two levels. The offsets it finds
# are then walked like seeds over the whole gap (+/-WALK_SEARCH_MS, same NCC and ambiguity rules).
# MEASURED, id 111 (Aldnoah.Zero S01E23, BD master against a web candidate): the b2 seeds were
# -91007 and -164374 ms; master 660-664 s reads at -90383.7 ms (NCC 0.65-0.71, 623 ms outside
# either seed's search), so the walk fused a +623 ms addition (candidate silence at the Part A/B
# break) and a -73990 ms deletion (master 674.3-748.8 s) into one "-73367 ms step" over a 92 s
# gap whose edges could not be read. At most GAP_PROBE_WORK_S candidate-seconds are searched per
# gap (one probe per window while the span is short, fewer on a long one), and a span over
# GAP_PROBE_MAX_S is not searched (ADDENDUM 26.2: an interior hole over 300 s declines anyway).
GAP_PROBE_MAX_S = 300.0
GAP_PROBE_WORK_S = 2400.0
GAP_PROBE_MIN_PROBES = 8


def _gaps(rows, win_s):
    """(first row, end row, candidate span lo, hi) of every gap the walk left between offsets."""
    ok = [i for i, row in enumerate(rows) if row["status"] == "ok"]
    if not ok:
        return []
    out = [(0, ok[0], 0.0, rows[ok[0]]["t"] + win_s + rows[ok[0]]["off"] / 1000.0)]
    for p, q in zip(ok, ok[1:]):
        if q - p > 1 and abs(rows[q]["off"] - rows[p]["off"]) >= JUMP_MS:
            out.append((p + 1, q, rows[p]["t"] + rows[p]["off"] / 1000.0,
                        rows[q]["t"] + win_s + rows[q]["off"] / 1000.0))
    out.append((ok[-1] + 1, len(rows), rows[ok[-1]]["t"] + rows[ok[-1]]["off"] / 1000.0, None))
    return out


def _probe_gaps(m, c, rows, win_s, search_ms):
    cand_end = len(c) / WALK_RATE
    for first, end, lo, hi in _gaps(rows, win_s):
        hi = cand_end if hi is None else min(hi, cand_end)
        lo = max(lo, 0.0)
        todo = [i for i in range(first, end) if rows[i]["status"] != "silent"]
        if not todo or hi - lo - win_s <= 0 or hi - lo > GAP_PROBE_MAX_S:
            continue
        count = min(len(todo), max(GAP_PROBE_MIN_PROBES, int(GAP_PROBE_WORK_S / (hi - lo))))
        found = []
        for k in np.unique(np.linspace(0, len(todo) - 1, count).round().astype(int)):
            t = rows[todo[k]]["t"]
            low_ms, high_ms = (lo - t) * 1000.0, (hi - win_s - t) * 1000.0
            result = _search(m, c, t, (low_ms + high_ms) / 2.0, win_s,
                             (high_ms - low_ms) / 2.0)
            if result is None or result.get("cand_silent") or _status(result) != "ok":
                continue
            if all(abs(result["off"] - seed) >= LEVEL_TOLERANCE_MS for seed in found):
                found.append(result["off"])
        for i in todo if found else []:
            t, best = rows[i]["t"], None
            for seed in found:
                result = _search(m, c, t, seed, win_s, search_ms)
                if result is None or result.get("cand_silent") or _status(result) != "ok":
                    continue
                start = t + result["off"] / 1000.0
                if not lo - search_ms / 1000.0 <= start <= hi - win_s + search_ms / 1000.0:
                    continue
                if best is None or result["ncc"] > best["ncc"]:
                    best = result
            if best is not None:
                rows[i].update(off=round(best["off"], 3), ncc=round(best["ncc"], 4),
                               second=round(best["second"], 4), seed="gap_probe", status="ok")
    return rows


def walk(m, c, seeds, win_s=WALK_WINDOW_S, hop_s=WALK_HOP_S, search_ms=WALK_SEARCH_MS,
         t0=0.0, t1=None, probe_gaps=True):
    """Every window of the master: its offset (sub-sample), NCC, second peak and status.

    CONTINUITY FIRST, THEN THE SEEDS (a deviation from the prototype, measured): each window is
    first searched around the offset the previous measured window read, and that reading is
    kept while it is `ok`; only a window the current offset no longer explains is searched
    around every seed, best NCC wins. MEASURED, id 134 (Mai-HiME 12): best-NCC-over-seeds jumps
    to +181889 ms for 88 windows (NCC 0.998, the candidate carries that audio twice) where the
    picture continues under -3097 ms. A real edit makes the current offset STOP matching, so
    continuity costs nothing at a real change point; a sub-quantum step moves the peak inside
    the continuity search (+/-150 ms) and is read as the new value, never smoothed.
    `probe_gaps`: then search the gaps between offsets where their content can only be
    (`_probe_gaps`, GAP_PROBE_MAX_S)."""
    end = len(m) / WALK_RATE if t1 is None else min(t1, len(m) / WALK_RATE)
    last = end - win_s
    rows = []
    current = None
    t = t0
    while t <= last:
        i0 = int(round(t * WALK_RATE))
        db = rms_db(m[i0:i0 + int(win_s * WALK_RATE)])
        row = {"t": round(t, 3), "db": round(db, 1)}
        if db < WALK_SILENT_DB:
            row["status"] = "silent"
            rows.append(row)
            t += hop_s
            continue
        best = None
        if current is not None:
            kept = _search(m, c, t, current, win_s, search_ms)
            if kept is not None and not kept.get("cand_silent") and _status(kept) == "ok":
                best = dict(kept, seed="continuity")
        cand_silent = 0
        if best is None:
            for index, seed in enumerate(seeds):
                result = _search(m, c, t, seed, win_s, search_ms)
                if result is None:
                    continue
                if result.get("cand_silent"):
                    cand_silent += 1
                    continue
                if best is None or result["ncc"] > best["ncc"]:
                    best = dict(result, seed=index)
        if best is None:
            row["status"] = "cand_silent" if cand_silent else "out_of_range"
        else:
            row.update(off=round(best["off"], 3), ncc=round(best["ncc"], 4),
                       second=round(best["second"], 4), seed=best["seed"],
                       status=_status(best))
            if row["status"] == "ok":
                current = best["off"]
        rows.append(row)
        t += hop_s
    return _probe_gaps(m, c, rows, win_s, search_ms) if probe_gaps else rows


def levels(rows):
    """Constant-offset levels over the `ok` windows: [{t_first, t_last, n, off_ms, mad_ms,
    ncc_med}], plus the times of the outlier windows (a would-be level that did not persist)."""
    ok = [row for row in rows if row["status"] == "ok"]
    found, outliers = [], []
    index = 0
    current = None
    while index < len(ok):
        row = ok[index]
        if current is not None and abs(row["off"] - np.median(current["offs"][-9:])) \
                < LEVEL_TOLERANCE_MS:
            current["offs"].append(row["off"])
            current["ts"].append(row["t"])
            current["nccs"].append(row["ncc"])
            index += 1
            continue
        run = [row]
        j = index + 1
        while j < len(ok) and len(run) < LEVEL_PERSIST and \
                abs(ok[j]["off"] - row["off"]) < LEVEL_TOLERANCE_MS:
            run.append(ok[j])
            j += 1
        if len(run) >= LEVEL_PERSIST or (current is None and j >= len(ok)):
            current = {"offs": [x["off"] for x in run], "ts": [x["t"] for x in run],
                       "nccs": [x["ncc"] for x in run]}
            found.append(current)
            index = j
        else:
            outliers.append(row["t"])
            index += 1
    out = []
    for level in found:
        offs = np.array(level["offs"])
        median = float(np.median(offs))
        out.append({"t_first": level["ts"][0], "t_last": level["ts"][-1], "n": len(offs),
                    "off_ms": round(median, 3),
                    "mad_ms": round(float(np.median(np.abs(offs - median))), 3),
                    "ncc_med": round(float(np.median(level["nccs"])), 4),
                    # THE LOCAL OFFSET AT EACH END (ADDENDUM 25.9.2): where the edge probes read
                    "off_first_ms": round(float(np.median(offs[:EDGE_LOCAL_WINDOWS])), 3),
                    "off_last_ms": round(float(np.median(offs[-EDGE_LOCAL_WINDOWS:])), 3)})
    merged = []
    for level in out:
        if merged and abs(level["off_ms"] - merged[-1]["off_ms"]) < LEVEL_TOLERANCE_MS:
            merged[-1]["t_last"] = level["t_last"]
            merged[-1]["n"] += level["n"]
            merged[-1]["off_last_ms"] = level["off_last_ms"]
        else:
            merged.append(level)
    return merged, outliers


# ---------------------------------------------------------------------------------------------
# LOCALISATION
# ---------------------------------------------------------------------------------------------

def _fixed_ncc(m, c, t, off_ms, win_s, micro_ms=1.0):
    n = int(win_s * WALK_RATE)
    i0 = int(round(t * WALK_RATE))
    span = int(micro_ms * WALK_RATE / 1000)
    j0 = i0 + int(round(off_ms * WALK_RATE / 1000)) - span
    if i0 < 0 or i0 + n > len(m) or j0 < 0 or j0 + n + 2 * span > len(c):
        return None, -200.0
    seg = c[j0:j0 + n + 2 * span].astype(np.float64)
    curve = _ncc_curve(m[i0:i0 + n].astype(np.float64), seg)
    return (None if curve is None else float(curve.max())), rms_db(seg)


# THE EDGE PROBES FALL BACK TO THE LOCAL OFFSET (ADDENDUM 25.9.2): inside a level the offset
# wanders a few ms (within LEVEL_TOLERANCE_MS), so a fixed-lag probe at the level MEDIAN can miss an
# edge the local offset reads. The local offset is the median of the EDGE_LOCAL_WINDOWS windows
# nearest the edge, searched +/- EDGE_LOCAL_SEARCH_MS by the coarse probe; the level median stays
# the step and the fill that are delivered. MEDIAN FIRST, LOCAL ONLY WHEN THE MEDIAN CANNOT READ
# THE EDGE -- MEASURED, id 111: CP3 (-90383.7 -> -164374.192 ms, local -164371.3 at the after-
# level's start) is unmeasurable at the median (0.4 s NCC -0.2...0.07) and reads 673.52 / 748.865 s
# at the local offsets; but CP2 (+623 ms) reads 656.945 / 661.325 s at the median and is
# unmeasurable at its before-level's local offset (-91009.2, 2 ms off); and on errid-222 a local
# offset 0.001 ms from the median moved an edge by 15 ms (a threshold flip). The median path is
# unchanged wherever it reads.
EDGE_LOCAL_WINDOWS = 5
EDGE_LOCAL_SEARCH_MS = 5.0


def coarse_edges(m, c, a, b, t_lo, t_hi, micro_ms=1.0):
    """Edges of an a -> b change inside master [t_lo, t_hi] from 0.4 s fixed-lag windows:
    (edge_A, edge_B) or None. Only a SEED for `fine_edges`, never an edge by itself (memo 4.1:
    coarse edges are biased inward by 0.1-0.3 s)."""
    profile = []
    t = t_lo
    while t + SHORT_WIN_S <= t_hi:
        i0 = int(round(t * WALK_RATE))
        mdb = rms_db(m[i0:i0 + int(SHORT_WIN_S * WALK_RATE)])
        na, _ = _fixed_ncc(m, c, t, a, SHORT_WIN_S, micro_ms)
        nb, _ = _fixed_ncc(m, c, t, b, SHORT_WIN_S, micro_ms)
        profile.append((t, mdb, na, nb))
        t += SHORT_HOP_S

    def claims(p, mine, other):
        return (p[1] >= WALK_SILENT_DB and mine is not None and mine >= SHORT_THRESHOLD
                and (other is None or mine - other >= SHORT_MARGIN))
    a_idx = [i for i, p in enumerate(profile) if claims(p, p[2], p[3])]
    b_idx = [i for i, p in enumerate(profile) if claims(p, p[3], p[2])]
    if not a_idx or not b_idx:
        return None
    first_b = b_idx[0]
    last_a = ([i for i in a_idx if i < first_b] or a_idx)[-1]
    return profile[last_a][0] + SHORT_WIN_S, profile[first_b][0]


def _shifted(c, t0, t1, off_ms):
    """Candidate samples for master [t0, t1) read at off_ms, fractional delay by FFT."""
    pos = t0 * WALK_RATE + off_ms * WALK_RATE / 1000.0
    i = int(np.floor(pos))
    frac = pos - i
    n = int(round((t1 - t0) * WALK_RATE))
    pad = 64
    lo = i - pad
    seg = c[max(lo, 0):i + n + pad].astype(np.float64)
    if lo < 0:
        seg = np.concatenate([np.zeros(-lo), seg])
    if len(seg) < n + 2 * pad:
        seg = np.concatenate([seg, np.zeros(n + 2 * pad - len(seg))])
    size = len(seg)
    freq = np.fft.rfftfreq(size)
    seg = np.fft.irfft(np.fft.rfft(seg) * np.exp(2j * np.pi * freq * frac), size)
    return seg[pad:pad + n]


# THE GAIN OF A FINE WINDOW IS READ ON ONE SIDE OF IT, never across it: each 20 ms window gets
# two gains, one fitted over the FINE_GAIN_S ending at the window's end, one over the FINE_GAIN_S
# starting at its start, and keeps the smaller residual. A gain fitted over a span centred on the
# window straddles a splice for the FINE_GAIN_S / 2 next to it and is dragged by the other side's
# content. MEASURED, id 152 (Dragon Ball Daima S01E20, change point -9.062 -> -4304.753 ms): the
# centred gain held the residual under offset b at 0.26-0.32 over master 986.01-986.09 s (just
# over FINE_THRESHOLD, read as master-only sound) where a gain read on the steady side gives
# 0.004-0.06; edge_B moved 80 ms outward and the 4295.691 ms hole "did not fit" a 4380 ms edge
# gap -- truth by 1 s xcorr on both sides: offsets exact to 0.001 ms up to the edges, edge gap
# 4300 ms. A window whose own content does not match keeps a residual near 1 or above under ANY
# scalar gain, so the one-sided fit cannot turn a mismatch into a match.
FINE_GAIN_S = 0.4


def _residuals(m, c, offsets, t0, t1):
    """Per 20 ms window (hop 5 ms): times, master dB, and the gain-fitted residual ratio under
    each offset of `offsets` (the gain read on one side of the window -- FINE_GAIN_S)."""
    times, mdb, residuals, _scale_free = _fits(m, c, offsets, t0, t1)
    return times, mdb, residuals


def _fits(m, c, offsets, t0, t1):
    """`_residuals`, plus per offset the SCALE-FREE residual of each window: its gain fitted on
    the window alone, unclipped (1 - the window's uncentred NCC squared) -- the one reading that
    follows a fade (see `fine_edges`)."""
    start = int(round(t0 * WALK_RATE))
    x = m[start:start + int(round((t1 - t0) * WALK_RATE))].astype(np.float64)
    w, h = int(FINE_WIN_S * WALK_RATE), int(FINE_HOP_S * WALK_RATE)
    span = int(FINE_GAIN_S * WALK_RATE)

    def cumulative(v):
        return np.concatenate([[0.0], np.cumsum(v)])
    starts = np.arange(0, max(len(x) - w + 1, 0), h)
    ends = starts + w
    xx = cumulative(x * x)
    energy = xx[ends] - xx[starts]
    back_lo, forth_hi = np.maximum(ends - span, 0), np.minimum(starts + span, len(x))
    residuals, scale_free = [], []
    for off in offsets:
        cs = _shifted(c, t0, t1, off)[:len(x)]
        xc, cc = cumulative(x * cs), cumulative(cs * cs)
        in_xc, in_cc = xc[ends] - xc[starts], cc[ends] - cc[starts]
        best = None
        for lo, hi in ((back_lo, ends), (starts, forth_hi)):
            gain = np.clip((xc[hi] - xc[lo]) / np.maximum(cc[hi] - cc[lo], 1e-12), 0.25, 4.0)
            residual = np.maximum(energy - 2 * gain * in_xc + gain * gain * in_cc, 0.0)
            best = residual if best is None else np.minimum(best, residual)
        residuals.append(best / np.maximum(energy, 1e-12))
        scale_free.append(np.clip(1.0 - in_xc * in_xc / np.maximum(energy * in_cc, 1e-30),
                                  0.0, 1.0))
    mdb = 10 * np.log10(np.maximum(energy / w, 1e-20))
    return t0 + starts / WALK_RATE, mdb, residuals, scale_free


def _ncc_profile(m, c, offsets, t0, t1, win_s=0.1):
    times = np.arange(t0, t1 - win_s, FINE_HOP_S)
    mdb = np.array([rms_db(m[int(round(t * WALK_RATE)):int(round(t * WALK_RATE))
                             + int(win_s * WALK_RATE)]) for t in times])
    nccs = [np.array([(_fixed_ncc(m, c, t, off, win_s, 0.5)[0] or 0.0) for t in times])
            for off in offsets]
    return times, mdb, nccs


def fine_edges(m, c, a, b, coarse_a, coarse_b, margin=0.6, probe_a=None, probe_b=None):
    """The sharp edges of an a -> b change (memo 1.4): edge_A = the last master instant read at
    a, edge_B = the first read at b; `extra_s` = max(0, a - b) of master content the candidate
    lacks; for a deletion the feasible fill-start interval (fill [F, F + extra] with F >= A,
    F + extra <= B, covering the audible master-only sound), for an addition the cut-instant
    interval. `status` is `ok` or `edge_unmeasurable`. `probe_a` / `probe_b`: the offsets the
    probes read at (the levels' local offsets at this edge, ADDENDUM 25.9.2); `a` / `b` stay the
    levels' medians, which set the step and the fill."""
    probe_a = a if probe_a is None else probe_a
    probe_b = b if probe_b is None else probe_b
    t0 = min(coarse_a, coarse_b) - margin
    t1 = max(coarse_a, coarse_b) + margin
    times, mdb, (ra, rb), (fa, fb) = _fits(m, c, (probe_a, probe_b), t0, t1)
    audible = mdb >= FINE_SILENT_DB
    pre = audible & (times < min(coarse_a, coarse_b) - 0.45)
    post = audible & (times > max(coarse_a, coarse_b) + 0.05)
    floor_a = float(np.median(ra[pre])) if pre.any() else 1.0
    floor_b = float(np.median(rb[post])) if post.any() else 1.0
    method = "residual20ms"
    if max(floor_a, floor_b) <= FINE_FLOOR_MAX:
        win = FINE_WIN_S
        is_a = audible & (ra < FINE_THRESHOLD) & (rb > 2 * ra)
        is_b = audible & (rb < FINE_THRESHOLD) & (ra > 2 * rb)
        good_a, good_b = ra < FINE_THRESHOLD, rb < FINE_THRESHOLD
        follows_a = audible & (fa < FINE_THRESHOLD) & (fb > 2 * fa)
        follows_b = audible & (fb < FINE_THRESHOLD) & (fa > 2 * fb)
    else:
        method, win = "ncc100ms", 0.1
        times, mdb, (na, nb) = _ncc_profile(m, c, (probe_a, probe_b), t0, t1, win)
        audible = mdb >= FINE_SILENT_DB
        is_a = audible & (na >= 0.8) & (na - nb >= 0.2)
        is_b = audible & (nb >= 0.8) & (nb - na >= 0.2)
        good_a, good_b = na >= 0.8, nb >= 0.8
        follows_a = follows_b = np.zeros(len(times), bool)       # NCC is already scale-free
    result = {"method": method, "floor_a": round(floor_a, 4), "floor_b": round(floor_b, 4),
              "a_ms": a, "b_ms": b, "step_ms": round(b - a, 3)}
    near = min(coarse_a, coarse_b) - 0.2
    b_idx = np.where(is_b & (times >= near))[0]
    first_b = None
    for i in b_idx:
        if (is_b[i:i + 8] | ~audible[i:i + 8]).all() and is_b[i:i + 8].sum() >= 4:
            first_b = i
            break
    if first_b is None and len(b_idx):
        first_b = b_idx[0]
    a_idx = np.where(is_a[:first_b])[0] if first_b is not None else []
    if first_b is None or not len(a_idx):
        result["status"] = "edge_unmeasurable"
        return result
    # edge_A CLOSES A SUSTAINED RUN, as edge_B opens one: the last window read at `a` whose 8
    # windows back are read at `a` or silent, at least 4 of them read. MEASURED, id 278 (The 100
    # S07E06, change point -4839.25 -> -6842.125 ms): offset a holds to the sample up to 1498.62 s
    # (residual 0.000), then two isolated 20 ms windows of a fading master (-35 dB, residual
    # 0.09 / 0.21) at 1499.325-1499.33 s put edge_A 0.73 s late and the 2002.875 ms hole "did not
    # fit" a 1280 ms edge gap; the sustained edge gives 2005 ms (step + 2.1 ms).
    last_a = a_idx[-1]
    for i in a_idx[::-1]:
        window = slice(max(i - 7, 0), i + 1)
        if (is_a[window] | ~audible[window]).all() and is_a[window].sum() >= 4:
            last_a = i
            break
    # A FADE IS STILL THE SAME SOUND: from each sustained edge, the edge moves outward over the
    # CONTIGUOUS windows that still read at its own offset once the gain is fitted on the window
    # alone (`_fits`, scale-free) -- a master fading to or from silence changes its gain faster
    # than any FINE_GAIN_S fit follows. MEASURED, id 686 (Yozakura-san S02E11, -8025.457 ->
    # -9026.456 ms): the master fades from -43 dB at 1086.0 s to digital silence at 1086.50 s
    # and back in from 1087.53 s; its 20 ms windows keep an uncentred NCC of 0.96-1.00 with the
    # candidate under offset a down to -86 dB, while the fitted residual crossed FINE_THRESHOLD
    # at -55 dB (1086.22 s): edge_A 0.25 s early, edge_B 0.22 s late, the fade tails read as 1530
    # ms of "master-only" sound for a 1001 ms step. Only the edges move: the windows BETWEEN them
    # keep the fitted test, so master-only sound is still found as before (the unclipped
    # in-window gain alone reads a mismatch as a match on up to 0.75 % of windows -- measured on
    # wrong offsets of 152/278/686 -- the fitted one-sided law on at most 0.07 %).
    while last_a + 1 < first_b and follows_a[last_a + 1]:
        last_a += 1
    while first_b - 1 > last_a and follows_b[first_b - 1]:
        first_b -= 1
    edge_a = float(times[last_a] + win)
    edge_b = float(times[first_b])
    extra = max(0.0, (a - b) / 1000.0)
    between = ((times >= min(edge_a, edge_b)) & (times + win <= max(edge_a, edge_b))
               & (mdb >= AUDIBLE_DB) & ~good_a & ~good_b)
    only = times[between]
    result.update(status="ok", edge_A=round(edge_a, 4), edge_B=round(edge_b, 4),
                  extra_s=round(extra, 6),
                  master_only_audible=([round(float(only.min()), 3),
                                        round(float(only.max() + win), 3)] if len(only) else None))
    if extra > 0:
        lo, hi = edge_a, edge_b - extra
        if edge_b - edge_a < extra:
            # BOTH OFFSETS EXPLAIN THE SPAN: the edges are closer than the step when the master
            # around the cut repeats itself at the step's lag (silence, a hum, a decay tail), so
            # it reads at a up to edge_A AND at b from edge_B. Any fill start F with a holding
            # up to F and b from F + extra is then the same splice: F in [edge_B - extra,
            # edge_A], cut back to where each offset really holds over it (a window that reads
            # at neither offset and is audible ends the span -- a stray late edge_A cannot open
            # it). MEASURED, id 104 (Lazarus S01E02, -20.0 -> -186.83 ms): master 809.96-810.23 s
            # at -87 to -98 dB reads at a (residual 0.000) and at b (0.078) at once, edge_A
            # 810.25 > edge_B 810.235, and the 166.83 ms hole "did not fit" a -15 ms edge gap.
            lo, hi = edge_b - extra, edge_a
            holds_a = good_a | (mdb < AUDIBLE_DB)
            holds_b = good_b | (mdb < AUDIBLE_DB)
            broken_a = (times >= lo) & (times + win <= hi) & ~holds_a
            if broken_a.any():
                hi = min(hi, float(times[broken_a].min()))
            broken_b = (times >= lo + extra) & (times + win <= hi + extra) & ~holds_b
            if broken_b.any():
                lo = max(lo, float(times[broken_b].max() + win) - extra)
        if len(only):
            lo = max(lo, float(only.max() + win) - extra)
            hi = min(hi, float(only.min()))
        result["kind"] = "deletion"
        result["interval"] = [round(lo, 4), round(hi, 4)]
        result["feasible"] = bool(hi >= lo - win)
    else:
        result["kind"] = "addition"
        result["interval"] = [round(min(edge_a, edge_b), 4), round(max(edge_a, edge_b), 4)]
        result["feasible"] = True
    return result


# THE EDGE WALK STOPS ON A SUSTAINED MISMATCH, not on the first mismatching 20 ms: an audible
# stretch shorter than half the walk's hop cannot have produced the level the edge closes (the
# walk would have measured across it), so half a hop of audible non-matching audio is where
# the common content provably ends.
EDGE_SUSTAINED_MISMATCH_S = WALK_HOP_S / 2.0


def single_edge(m, c, off, anchor_s, limit_s, side):
    """ONE edge under ONE offset, for the file's head or tail (ADDENDUM 25: the audio edge
    decides where the candidate's common content ends). From `anchor_s` -- the outer end of the
    level's first (head) / last (tail) measured window, walked inward-to-outward so the window's
    matching content is met first -- walk 20 ms windows toward `limit_s` (forward for 'tail', backward
    for 'head') while they read at `off` or are digitally silent, and stop at the first
    `EDGE_SUSTAINED_MISMATCH_S` of audible mismatch. Returns the end (tail) / start (head) of the
    last matching window, or None when nothing matches. Falls back to 100 ms NCC windows like
    `fine_edges` when the waveforms are not sample-identical (a rate pair)."""
    t0, t1 = min(anchor_s, limit_s), max(anchor_s, limit_s)
    times, mdb, (res,) = _residuals(m, c, (off,), t0, t1)
    audible = mdb >= FINE_SILENT_DB
    win = FINE_WIN_S
    good = audible & (res < FINE_THRESHOLD)
    method = "residual20ms"
    floor = float(np.median(res[good])) if good.any() else 1.0
    if floor > FINE_FLOOR_MAX:
        method, win = "ncc100ms", 0.1
        times, mdb, (nccs,) = _ncc_profile(m, c, (off,), t0, t1, win)
        audible = mdb >= FINE_SILENT_DB
        good = audible & (nccs >= 0.8)
    if not len(times):
        return None
    order = range(len(times)) if side == "tail" else range(len(times) - 1, -1, -1)
    start = int(np.argmin(np.abs(times - anchor_s)))
    order = [i for i in order if (i >= start if side == "tail" else i <= start)]
    limit = int(round(EDGE_SUSTAINED_MISMATCH_S / FINE_HOP_S))
    last_good, bad_run = None, 0
    for i in order:
        if good[i]:
            last_good, bad_run = i, 0
        elif audible[i]:
            bad_run += 1
            if bad_run >= limit:
                break
    if last_good is None:
        return None
    # ACROSS THE MASTER'S OWN SILENCE: past the last matching window, master windows under
    # AUDIBLE_DB carry nothing a fill must cover (the memo's own fill rule: only master sound at
    # or above it must be covered), and the candidate can supply that span itself -- so the edge
    # moves over them, up to the candidate's own extent. MEASURED, errid-202: the master sits at
    # -101 +/- 1 dB (dither) over [0.1, 3.4] s; stopping at the first sound put the head edge at
    # 3.42 s where the candidate's own start (1.018 s) and the video (1.001 s) agree -- a 2.4 s
    # master fill of silence counted as an edge addition. (-100 dB, the digital-silence floor of
    # the MATCHING test, is crossed by single 20 ms windows of that dither, so it cannot bound
    # this extension.)
    quiet = mdb < AUDIBLE_DB
    cand_end = len(c) / WALK_RATE
    step = 1 if side == "tail" else -1
    i = last_good
    while 0 <= i + step < len(times) and quiet[i + step] and not good[i + step]:
        t = times[i + step] + (win if side == "tail" else 0.0)
        if not 0.0 <= t + off / 1000.0 <= cand_end:
            break
        i += step
    edge = float(times[i] + win) if side == "tail" else float(times[i])
    return {"edge_s": round(edge, 4), "method": method, "floor": round(floor, 4),
            "silence_extended_s": round(abs(edge - (float(times[last_good] + win)
                                                    if side == "tail"
                                                    else float(times[last_good]))), 4)}


def quietest_instant(m, lo_s, hi_s, extra_s=0.0):
    """The instant F in [lo, hi] where the master is quietest at the splice points F and
    F + extra (20 ms windows, 5 ms steps) -- ADDENDUM 25.1: "sinon le raccord se pose au point de
    plus faible energie de l'intervalle, logge avec l'intervalle"."""
    if hi_s <= lo_s:
        return lo_s
    best, best_energy = lo_s, None
    w = int(FINE_WIN_S * WALK_RATE)
    t = lo_s
    while t <= hi_s + 1e-9:
        energy = 0.0
        for point in (t, t + extra_s):
            i0 = max(0, int(round(point * WALK_RATE)) - w // 2)
            chunk = m[i0:i0 + w].astype(np.float64)
            energy += float((chunk ** 2).mean()) if len(chunk) else 0.0
        if best_energy is None or energy < best_energy:
            best, best_energy = t, energy
        t += FINE_HOP_S
    return round(best, 4)


# THE CANDIDATE'S OWN SOUND BETWEEN THE EDGES (errid 695, CASE_id695_splice_in_dialogue_20260925).
# A transition at F plays the candidate under offset a over [edge_A, F) and under offset b over
# [F + extra, edge_B): past its own edge each offset reads the candidate's OWN material, which
# the master does not carry. Where that material is silent any F is the same splice; where it is
# speech, F decides how much unsynced speech the product plays. The master is no judge of it:
# MEASURED id 695 (Bleach TYBW S17E41, change point +10005.456 -> +22142.581 ms), the master is
# digital silence (-116 to -118 dB) over the whole interval [168.515, 169.585] s, so its
# "quietest instant" was dither and fell at 169.275 s, 380 ms into candidate speech under a
# (candidate 178.9-179.6 s at -33 dB) -- cut mid-sentence. A 20 ms window is a LEAK when the
# candidate under its offset is audible (AUDIBLE_DB, the fill rule's own floor) and does not
# read as the master there (gain-fitted residual >= FINE_THRESHOLD).
# The guard keeps the cut one fine window away from the leak's first/last window, so the 10 ms
# crossfade or a millisecond of rounding never lets its attack through.
LEAK_GUARD_S = FINE_WIN_S


def _candidate_db(c, times, t0, t1, off_ms, win):
    """Candidate level (dB) of each fine window at `times`, read under `off_ms`."""
    cs = _shifted(c, t0, t1, off_ms).astype(np.float64)
    cc = np.concatenate([[0.0], np.cumsum(cs * cs)])
    w = int(round(win * WALK_RATE))
    starts = np.clip(np.round((times - t0) * WALK_RATE).astype(int), 0, max(len(cs) - w, 0))
    energy = (cc[np.minimum(starts + w, len(cs))] - cc[starts]) / max(w, 1)
    return 10 * np.log10(np.maximum(energy, 1e-20))


def leak_bounds(m, c, a_ms, b_ms, edge_a, edge_b, lo_s, hi_s, extra_s=0.0):
    """Where a transition a -> b may sit without playing the candidate's own audible material
    (see LEAK_GUARD_S). `lo_s`/`hi_s` bound F (the fill start of a deletion, the cut instant of
    an addition). Returns a dict:
      interval    [lo', hi'] inside [lo, hi]: every F there plays no leak window
      clean       True when that interval is not empty ("a silent boundary exists")
      narrowed    True when it is strictly narrower than [lo, hi]
      leak_a_s    first leak window start under a past edge_A (None: none)
      leak_b_s    last leak window end under b before edge_B (None: none)
      min_leak_s  the F in [lo, hi] playing the least leaked energy (for `clean` False)
    Measures, never decides."""
    win = FINE_WIN_S
    t0 = min(edge_a, lo_s) - win
    t1 = max(edge_b, hi_s + extra_s) + win
    # A speech ENVELOPE (an atempo pair) has no level in dB to hold against AUDIBLE_DB: the
    # bounds stand as the walk gave them.
    if t1 - t0 < 2 * win or isinstance(m, EnvelopeSignal) or isinstance(c, EnvelopeSignal):
        return {"interval": [lo_s, hi_s], "clean": True, "narrowed": False, "leak_a_s": None,
                "leak_b_s": None, "min_leak_s": lo_s}
    times, _mdb, (ra, rb), _sf = _fits(m, c, (a_ms, b_ms), t0, t1)
    cdb_a = _candidate_db(c, times, t0, t1, a_ms, win)
    cdb_b = _candidate_db(c, times, t0, t1, b_ms, win)
    # Only the windows a transition inside [lo, hi] can play: under a [edge_A, hi), under b
    # (lo + extra, edge_B].
    leak_a = ((cdb_a >= AUDIBLE_DB) & (ra >= FINE_THRESHOLD) & (times >= edge_a - 1e-9)
              & (times < hi_s))
    leak_b = ((cdb_b >= AUDIBLE_DB) & (rb >= FINE_THRESHOLD) & (times + win <= edge_b + 1e-9)
              & (times + win > lo_s + extra_s))
    leak_a_s = float(times[leak_a].min()) if leak_a.any() else None
    leak_b_s = float(times[leak_b].max() + win) if leak_b.any() else None
    clo = lo_s if leak_b_s is None else max(lo_s, leak_b_s + LEAK_GUARD_S - extra_s)
    chi = hi_s if leak_a_s is None else min(hi_s, leak_a_s - LEAK_GUARD_S)
    clean = clo <= chi + 1e-9
    # THE LEAST LEAK when no F is clean: leaked energy under a over [edge_A, F) plus under b
    # over [F + extra, edge_B), on the 5 ms grid.
    ea = np.where(leak_a, 10 ** (cdb_a / 10), 0.0)
    eb = np.where(leak_b, 10 ** (cdb_b / 10), 0.0)
    best, best_cost = lo_s, None
    f = lo_s
    while f <= hi_s + 1e-9:
        cost = float(ea[times < f].sum()) + float(eb[times + win > f + extra_s].sum())
        if best_cost is None or cost < best_cost:
            best, best_cost = f, cost
        f += FINE_HOP_S
    return {"interval": [round(clo, 4), round(chi, 4)] if clean else [lo_s, hi_s],
            "clean": bool(clean),
            "narrowed": bool(clean and (clo > lo_s + 1e-9 or chi < hi_s - 1e-9)),
            "leak_a_s": None if leak_a_s is None else round(leak_a_s, 4),
            "leak_b_s": None if leak_b_s is None else round(leak_b_s, 4),
            "min_leak_s": round(best, 4)}


# A NON-REFERENCE TRACK'S ZONE OFFSET IS A MEDIAN, NOT A PLAN: at most this many windows per zone
# (the hop widens on a long zone). The reference track, which fixes the plan, keeps the full walk;
# a median over 120 windows of a level whose MAD is <= 0.15 ms (memo section 3) is far below one
# millisecond, and a one-hour track then costs 120 windows instead of 3600.
ZONE_OFFSET_MAX_WINDOWS = 120


def zone_offset(m, c, t0, t1, seed):
    """THIS TRACK's own offset over master [t0, t1) (stage 5, ADDENDUM 25.2 -- replaces the
    three-window majority): the walk restricted to the zone with the zone's reference offset
    as its only seed, its levels, the dominant one's median. Returns a dict: `offset_ms` (None
    when no window measured), `n_ok`, and the levels -- more than one level means this track
    changes inside a zone the reference track did not, which is logged by the caller."""
    hop = max(WALK_HOP_S, (t1 - t0 - WALK_WINDOW_S) / ZONE_OFFSET_MAX_WINDOWS)
    rows = walk(m, c, [seed], hop_s=hop, t0=t0, t1=t1, probe_gaps=False)
    found, _outliers = levels(rows)
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    if not found:
        return {"offset_ms": None, "n_ok": 0, "levels": [], "counts": counts}
    dominant = max(found, key=lambda level: level["n"])
    return {"offset_ms": dominant["off_ms"], "n_ok": sum(level["n"] for level in found),
            "levels": found, "counts": counts}


def change_points(m, c, found):
    """Each pair of consecutive levels at least JUMP_MS apart, localised: coarse edges from the
    0.4 s profile, then `fine_edges`. A pair closer than JUMP_MS is `sub_threshold` and is not a
    change point (its levels stay one zone)."""
    points = []
    for before, after in zip(found, found[1:]):
        jump = after["off_ms"] - before["off_ms"]
        point = {"a_ms": before["off_ms"], "b_ms": after["off_ms"], "jump_ms": round(jump, 3),
                 "level_before": before, "level_after": after}
        if abs(jump) < JUMP_MS:
            point["kind"] = "sub_threshold"
            points.append(point)
            continue
        point["kind"] = "change_point"
        # ONE MORE WALK HOP OF EACH LEVEL: a level holds over its measured windows, and a pair
        # whose common content reads at a 2 s NCC of 0.6-0.7 (a remixed web candidate) may claim
        # none of the 0.4 s windows inside the two bounding ones. MEASURED, id 111 (-91007.2 ->
        # -90383.7 ms, +623 ms of candidate silence at the Part A/B break): the first 0.4 s claim
        # of the after-level is at 661.83 s, 0.23 s past [656, 662]; with one hop more the edges
        # read 656.945 / 661.325 s around the master's digital silence (658.2-659.2 s).
        t_lo = before["t_last"] - WALK_HOP_S
        t_hi = after["t_first"] + WALK_WINDOW_S + WALK_HOP_S
        coarse = coarse_edges(m, c, before["off_ms"], after["off_ms"], t_lo, t_hi)
        edges = (fine_edges(m, c, before["off_ms"], after["off_ms"], *coarse)
                 if coarse is not None else None)
        if edges is None or edges["status"] != "ok":
            probe_a = before.get("off_last_ms", before["off_ms"])
            probe_b = after.get("off_first_ms", after["off_ms"])
            local = coarse_edges(m, c, probe_a, probe_b, t_lo, t_hi,
                                 micro_ms=EDGE_LOCAL_SEARCH_MS)
            if local is not None:
                retried = fine_edges(m, c, before["off_ms"], after["off_ms"], *local,
                                     probe_a=probe_a, probe_b=probe_b)
                if retried["status"] == "ok" or edges is None:
                    edges = dict(retried, probe_ms=[probe_a, probe_b])
        point["edges"] = (edges if edges is not None
                          else {"status": "edge_unmeasurable", "stage": "coarse"})
        points.append(point)
    return points


def log_walk(candidate_path, rows, found, points, seconds):
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    tools.dev_log(f"audio_walk: windows={len(rows)} counts={counts} seconds={round(seconds, 1)} "
                  f"levels={[(lv['t_first'], lv['t_last'], lv['off_ms'], lv['mad_ms']) for lv in found]} "
                  f"for {candidate_path}\n")
    for point in points:
        tools.dev_log(f"audio_walk: {point['kind']} a_ms={point['a_ms']} b_ms={point['b_ms']} "
                      f"jump_ms={point['jump_ms']} edges={point.get('edges')} "
                      f"for {candidate_path}\n")
