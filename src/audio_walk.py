# -*- coding: utf-8 -*-
"""Millisecond-resolution audio offset walk between a master and a candidate track.

The fingerprint aligner resolves one quantum (123.8 ms); this module refines it on the waveform:
a sliding normalised cross-correlation seeded by the aligner's zone offsets, the constant-offset
levels it reads, the change points between levels, and their edges at 20 ms.

Convention: every track is read mono at `WALK_RATE` on its file clock (the container
`start_time` prepended as zeros, scaled on a rate pair); an offset `o` in ms means
`candidate(t + o) == master(t)`.

The module only measures (levels, change points, edges, intervals, quietest instants); the
caller decides what to do with them.
"""
from decimal import Decimal

import numpy as np

import tools

# 16 kHz: a 20 ms window is 320 samples and the parabolic peak resolves ~0.01 ms; all thresholds
# below are calibrated at this rate.
WALK_RATE = 16000

# Window 2 s, hop 1 s: no false jumps from 0.5 to 10 s windows, at ~30-60 s per hour of audio.
# +/-185.5 ms (measured, real media: a true middle offset 185.5 ms from its aligner seed) is the
# worst seed error observed; +/-200 ms keeps a margin over it rather than sitting on the exact
# measured figure.
WALK_WINDOW_S = 2.0
WALK_HOP_S = 1.0
WALK_SEARCH_MS = 200.0
# A window whose master content is under this level is unmeasurable: no offset is read there.
# This is the walk's floor only; the fine edges read down to digital silence.
WALK_SILENT_DB = -55.0
# A measured window: NCC at least this, and its second peak (more than 3 ms away) under
# PEAK_AMBIGUITY of the main one; otherwise `unmatched` / `ambiguous`, never an offset.
MATCH_NCC = 0.5
PEAK_AMBIGUITY = 0.92
# LEVELS: a window joins a level within LEVEL_TOLERANCE_MS of its running median; a new level
# must hold LEVEL_PERSIST consecutive windows or its windows are outliers. Measured: a 2-window
# run (1 s of real content) read a spurious level from a window pair that did not belong to a
# real offset change, which `change_points` then read as a false step; 3 windows still accepts
# every genuine level observed and asks for one more confirming window before trusting a new one.
LEVEL_TOLERANCE_MS = 5.0
LEVEL_PERSIST = 3
# A change point is a jump of at least this between levels: in-level wander stays under ~10 ms,
# real sub-quantum steps are 25 ms and more.
JUMP_MS = 15.0
# Coarse localisation: fixed-lag 0.4 s windows every 10 ms around the change; a side is claimed
# at NCC >= 0.8 with a 0.2 margin over the other.
SHORT_WIN_S, SHORT_HOP_S, SHORT_THRESHOLD, SHORT_MARGIN = 0.4, 0.01, 0.8, 0.2
# Fine edges: residual of a gain-fitted 20 ms window every 5 ms. Only digital silence is
# unmeasurable (a -55 dB floor pulls edges inward); a residual floor over FINE_FLOOR_MAX means
# the waveforms are not sample-identical (a rate pair) and edges fall back to 100 ms NCC windows.
FINE_WIN_S, FINE_HOP_S, FINE_THRESHOLD = 0.02, 0.005, 0.25
FINE_SILENT_DB = -100.0
FINE_FLOOR_MAX = 0.1
# Level (20 ms window) at or above which master-only sound must be covered by a fill.
AUDIBLE_DB = -60.0
# A master-only-audible span the candidate already matches at this NCC (searched, not the
# gain-fitted residual test) is not really master-only: same threshold as the coarse side claim.
MASTER_ONLY_COVERED_NCC = SHORT_THRESHOLD
# The matched offset must beat the other by this much (fine_edges' own ncc100ms disambiguation),
# so a short span searched widely cannot read as covered by chance at both offsets alike.
MASTER_ONLY_COVERED_MARGIN = 0.2
# A span counts as covered only once at least this fraction of it reads a match: a wide span's
# natural boundary into the next level (one window's width) must not look like real coverage.
MASTER_ONLY_COVERED_FRACTION = 0.05


# ---------------------------------------------------------------------------------------------
# DECODE
# ---------------------------------------------------------------------------------------------

# Speech envelope: `atempo` (WSOLA) rebuilds the waveform from overlapped grains, so waveform
# NCC fails (corr < 0.5) while the envelope still correlates (0.8-0.98). Band 300-1800 Hz, 5 ms
# RMS, interpolated back to the reading's rate so downstream code reads it like a waveform.
ENVELOPE_BAND_HZ = (300.0, 1800.0)
ENVELOPE_HOP_S = 0.005


class EnvelopeSignal(np.ndarray):
    """A track read as its speech envelope (`read_on_file_clock(..., envelope=True)`).

    The type tells `_search` the correlation peak is broad (tens of ms), so the second peak is
    sought outside the whole main lobe instead of outside +/-3 ms."""


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
    """Read one track as mono float32 at WALK_RATE on the file clock.

    The stream's container start_time (times `scale` on a rate pair) is prepended as zeros; a
    negative start drops samples. `deadline` bounds the decode; `envelope` returns the speech
    envelope instead of the waveform. Results are cached per repair
    (`merge_video_decode_once.file_clock`) and shared read-only."""
    import merge_video_decode_once
    return merge_video_decode_once.file_clock(
        video_obj, audio, lambda: _read_on_file_clock(video_obj, audio, audio_filter, scale,
                                                      deadline, envelope),
        audio_filter=audio_filter, scale=scale, envelope=envelope)


def _read_on_file_clock(video_obj, audio, audio_filter, scale, deadline, envelope):
    """Uncached decode behind `read_on_file_clock`."""
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
    """Return the RMS level of `x` in dB (-200 for an empty array)."""
    if len(x) == 0:
        return -200.0
    value = float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))
    return 20 * np.log10(max(value, 1e-10))


def peak_db(x):
    """Return the peak level of `x` in dB (-200 for an empty array).

    Unlike `rms_db`, a peak is not diluted by surrounding silence, so it stays
    the right measure of "is this sample audible" over a span that mixes a
    short loud stretch with a much longer quiet one.
    """
    if len(x) == 0:
        return -200.0
    value = float(np.max(np.abs(np.asarray(x, np.float64))))
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

# Gap probing: each run of audible unmeasured windows between two `ok` windows whose offsets
# differ by at least JUMP_MS (and before the first / after the last `ok` window) is searched for
# common content inside the candidate span those windows bound, since an edit keeps content order.
# Offsets found are walked like seeds over the gap. This catches an offset outside every seed's
# search window, which would otherwise be fused with a neighbour. At most GAP_PROBE_WORK_S
# candidate-seconds are searched per gap; spans over GAP_PROBE_MAX_S are skipped. Measured: a
# real 482 s unmatched run (above the former 300 s cap, so never probed) hid the offset that
# change_points needed; 600 s keeps a margin over it.
GAP_PROBE_MAX_S = 600.0
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
         t0=0.0, t1=None, probe_gaps=True, _reseeded=False):
    """Return one row per master window: offset (sub-sample), NCC, second peak and status.

    Each window is searched first around the previous `ok` offset (continuity), and only when
    that fails around every seed, best NCC winning; continuity avoids jumping to a repeated copy
    of the audio. `probe_gaps` then runs `_probe_gaps` on the unmeasured gaps.

    When no seed covers the true head offset, the first seeded window can lock onto any other
    `ok` peak (a repetitive-music secondary peak included), and continuity then holds that wrong
    level until `_probe_gaps` finds the true offset -- but only in the rows still unmeasured, so
    an already-`ok` wrong level is never revisited. If the probe finds an offset farther than
    `search_ms` from every seed, the whole walk is re-run once with it added, which lets
    continuity reach the true offset from its own start; `_reseeded` bounds this to one extra
    pass."""
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
    if not probe_gaps:
        return rows
    probed = _probe_gaps(m, c, rows, win_s, search_ms)
    if _reseeded:
        return probed
    found_offsets = {row["off"] for row in probed if row.get("seed") == "gap_probe"}
    new_seeds = [off for off in found_offsets
                if all(abs(off - seed) >= search_ms for seed in seeds)]
    if not new_seeds:
        return probed
    return walk(m, c, list(seeds) + new_seeds, win_s, hop_s, search_ms, t0, t1,
               probe_gaps=True, _reseeded=True)


def levels(rows):
    """Group the `ok` windows into constant-offset levels.

    Returns (levels, outlier_times); each level is {t_first, t_last, n, off_ms, mad_ms, ncc_med,
    off_first_ms, off_last_ms}."""
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
                    # local offset at each end, where the edge probes read (EDGE_LOCAL_WINDOWS)
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


# Edge probes fall back to the local offset: inside a level the offset wanders a few ms, so a
# probe at the level median can miss an edge. The local offset is the median of the
# EDGE_LOCAL_WINDOWS windows nearest the edge, searched +/- EDGE_LOCAL_SEARCH_MS. The median is
# tried first (some edges read only there); the level median always sets the step and the fill.
EDGE_LOCAL_WINDOWS = 5
EDGE_LOCAL_SEARCH_MS = 5.0


def coarse_edges(m, c, a, b, t_lo, t_hi, micro_ms=1.0):
    """Locate an a -> b change in master [t_lo, t_hi] with 0.4 s fixed-lag windows.

    Returns (edge_A, edge_B) or None. Only a seed for `fine_edges`: coarse edges are biased
    inward by 0.1-0.3 s."""
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


# Each fine window's gain is fitted on one side of it, never across it: once over the
# FINE_GAIN_S ending at the window's end, once over the FINE_GAIN_S starting at its start, keeping
# the smaller residual. A centred fit straddles a nearby splice and pushes the residual over
# FINE_THRESHOLD; a true mismatch stays near 1 under any scalar gain.
FINE_GAIN_S = 0.4


def _residuals(m, c, offsets, t0, t1):
    """Per 20 ms window (5 ms hop): times, master dB, and the one-sided gain-fitted residual
    ratio under each offset in `offsets`."""
    times, mdb, residuals, _scale_free = _fits(m, c, offsets, t0, t1)
    return times, mdb, residuals


def _fits(m, c, offsets, t0, t1):
    """Like `_residuals`, plus per offset the scale-free residual of each window.

    The scale-free residual (1 - uncentred NCC squared) fits the gain on the window alone, so it
    follows a fade."""
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


def master_only_covered(m, c, only_s, a_ms, b_ms, win_s=0.1, hop_s=0.02, search_ms=WALK_SEARCH_MS):
    """True when the candidate's own audio already covers `only_s` at either offset.

    Samples a 0.1 s window (fine_edges' own ncc100ms fallback width) every `hop_s` across
    `only_s`: a plain NCC search (`_search`, as the walk already uses) is scale-free, so it
    still matches genuine content that fine_edges' gain-fitted good_a/good_b test can miss next
    to true silence. A sample counts as covered only when its offset clears
    MASTER_ONLY_COVERED_NCC and beats the other offset by MASTER_ONLY_COVERED_MARGIN (fine_edges'
    own disambiguation), and the span counts as covered only once at least a
    MASTER_ONLY_COVERED_FRACTION share of samples do: a wide span's one-window sliver into the
    next level, at its very edge, must not alone read as real coverage.
    """
    lo_s, hi_s = only_s
    if hi_s <= lo_s:
        return False
    samples = covered = 0
    t = lo_s
    while t < hi_s:
        ra = _search(m, c, t, a_ms, win_s, search_ms)
        rb = _search(m, c, t, b_ms, win_s, search_ms)
        na = 0.0 if ra is None or ra.get("cand_silent") else float(ra.get("ncc", 0.0))
        nb = 0.0 if rb is None or rb.get("cand_silent") else float(rb.get("ncc", 0.0))
        if ((na >= MASTER_ONLY_COVERED_NCC and na - nb >= MASTER_ONLY_COVERED_MARGIN)
                or (nb >= MASTER_ONLY_COVERED_NCC and nb - na >= MASTER_ONLY_COVERED_MARGIN)):
            covered += 1
        samples += 1
        t += hop_s
    return samples > 0 and covered / samples >= MASTER_ONLY_COVERED_FRACTION


def fine_edges(m, c, a, b, coarse_a, coarse_b, margin=0.6, probe_a=None, probe_b=None):
    """Measure the sharp edges of an a -> b change.

    edge_A is the last master instant read at a, edge_B the first read at b; `extra_s` =
    max(0, a - b) is master content the candidate lacks. `interval` is the feasible fill-start
    interval for a deletion, the cut-instant interval for an addition. `status` is `ok` or
    `edge_unmeasurable`. `probe_a` / `probe_b` are the offsets probed (local offsets at the
    edge); `a` / `b` stay the level medians, which set the step."""
    probe_a = a if probe_a is None else probe_a
    probe_b = b if probe_b is None else probe_b
    # A deletion's b level cannot start before coarse_A + the step, so its residual floor is read
    # past that instant (coarse_B may sit inside the master-only span).
    b_from = max(coarse_a, coarse_b)
    if a > b:
        b_from = max(b_from, coarse_a + (a - b) / 1000.0)
    t0 = min(coarse_a, coarse_b) - margin
    t1 = b_from + margin
    times, mdb, (ra, rb), (fa, fb) = _fits(m, c, (probe_a, probe_b), t0, t1)
    audible = mdb >= FINE_SILENT_DB
    pre = audible & (times < min(coarse_a, coarse_b) - 0.45)
    post = audible & (times > b_from + 0.05)
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
    # edge_A closes a sustained run (8 windows read at `a` or silent, at least 4 read), as edge_B
    # opens one; isolated matches in a fading master would otherwise push edge_A late.
    last_a = a_idx[-1]
    for i in a_idx[::-1]:
        window = slice(max(i - 7, 0), i + 1)
        if (is_a[window] | ~audible[window]).all() and is_a[window].sum() >= 4:
            last_a = i
            break
    # A fade is the same sound: each edge extends outward over contiguous windows that still
    # match under the scale-free residual, since a fade changes gain faster than a FINE_GAIN_S
    # fit follows. Only the edges move; the windows between them keep the stricter fitted test.
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
            # Edges closer than the step: the master repeats at the step's lag (silence, hum,
            # decay), so any F in [edge_B - extra, edge_A] is the same splice, narrowed to where
            # each offset really holds (an audible window matching neither ends the span).
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


# The edge walk stops on a sustained mismatch: an audible mismatch shorter than half the walk's
# hop could not have ended the level, so half a hop marks where common content ends.
EDGE_SUSTAINED_MISMATCH_S = WALK_HOP_S / 2.0


def single_edge(m, c, off, anchor_s, limit_s, side):
    """Find where the candidate's common content ends at the file's head or tail under `off`.

    Walks 20 ms windows from `anchor_s` (the level's outermost measured window) toward `limit_s`
    while they match or are silent, stopping after `EDGE_SUSTAINED_MISMATCH_S` of audible
    mismatch, then extends over master silence where the candidate is quiet too, and at the
    tail over the candidate's run-on past the master's last sample (`tail_run_on`). Returns
    {edge_s, method, floor, silence_extended_s, run_on_s} or None. Falls back to 100 ms NCC
    windows on a rate pair."""
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
    # Extend over the master's own silence (under AUDIBLE_DB, not FINE_SILENT_DB, which dither
    # crosses), but only where the candidate is quiet too: candidate sound over a silent master
    # is the candidate's own content, never common content.
    quiet = (mdb < AUDIBLE_DB) & (_candidate_db(c, times, t0, t1, off, win) < AUDIBLE_DB)
    cand_end = len(c) / WALK_RATE
    step = 1 if side == "tail" else -1
    i = last_good
    while 0 <= i + step < len(times) and quiet[i + step] and not good[i + step]:
        t = times[i + step] + (win if side == "tail" else 0.0)
        if not 0.0 <= t + off / 1000.0 <= cand_end:
            break
        i += step
    edge = float(times[i] + win) if side == "tail" else float(times[i])
    extended = round(abs(edge - (float(times[last_good] + win) if side == "tail"
                                 else float(times[last_good]))), 4)
    run_on = tail_run_on(m, c, off, edge) if side == "tail" else None
    return {"edge_s": round(edge if run_on is None else run_on, 4), "method": method,
            "floor": round(floor, 4), "silence_extended_s": extended,
            "run_on_s": None if run_on is None else round(run_on - edge, 4)}


# Where the master's track ends mid-sound, the candidate's same sound runs on past it. It is
# common content up to its last sample above digital silence when it decays (never louder than
# the candidate over the last RUN_ON_REFERENCE_S under the master) and falls silent for a fine
# window within EDGE_SUSTAINED_MISMATCH_S; anything longer or louder is the candidate's own.
RUN_ON_REFERENCE_S = 0.1


def tail_run_on(m, c, off, edge_s):
    """Extend a tail edge that reached the end of the master's samples over the candidate's run-on.

    Returns the master-time instant just after the candidate's last sample above digital
    silence, or None when the edge stops before the master's end, the candidate runs on louder
    or longer than the rule allows, or on an envelope pair."""
    if isinstance(c, EnvelopeSignal) or isinstance(m, EnvelopeSignal):
        return None
    master_end = len(m) / WALK_RATE
    if edge_s < master_end - FINE_WIN_S:
        return None
    shift = off / 1000.0
    start = int(round((master_end + shift) * WALK_RATE))
    w = int(round(FINE_WIN_S * WALK_RATE))
    if start - w < 0 or start >= len(c):
        return None
    budget = int(round(EDGE_SUSTAINED_MISMATCH_S * WALK_RATE))
    seg = np.asarray(c[start:start + budget + w], np.float64)
    if start + len(seg) >= len(c):
        # The candidate's own end counts as silence.
        seg = np.concatenate([seg, np.zeros(w)])
    sound = np.concatenate([[0], np.cumsum(np.abs(seg) >= DIGITAL_SILENCE_AMPLITUDE)])
    silent = np.flatnonzero(sound[w:] - sound[:-w] == 0)
    silent = silent[silent <= budget]
    if not len(silent) or silent[0] == 0:
        return None
    heard = np.flatnonzero(np.abs(seg[:silent[0]]) >= DIGITAL_SILENCE_AMPLITUDE)
    end = int(heard[-1]) + 1
    hop = int(round(FINE_HOP_S * WALK_RATE))

    def loudest(x):
        return max(rms_db(x[k:k + w]) for k in range(0, max(len(x) - w, 0) + 1, hop))
    reference = c[max(0, start - int(round(RUN_ON_REFERENCE_S * WALK_RATE))):start]
    if loudest(seg[:end]) > loudest(reference):
        return None
    return max(edge_s, (start + end) / WALK_RATE - shift)


# The candidate's content begins at its first sound, not at its file start: a sample under the
# digital-silence floor of the matching test (FINE_SILENT_DB) is no content; the master's first
# sound is its first sample at AUDIBLE_DB, the level a fill must cover.
DIGITAL_SILENCE_AMPLITUDE = 10.0 ** (FINE_SILENT_DB / 20.0)
AUDIBLE_AMPLITUDE = 10.0 ** (AUDIBLE_DB / 20.0)


def head_lead_in(m, c, off, limit_s):
    """Measure master content left uncovered by a candidate's digitally silent lead-in.

    Under the head offset `off`, returns {content_edge_s, master_first_sound_s, uncovered_s,
    master_db} in master time when the master has any sample at AUDIBLE_DB or louder before the
    candidate's first non-silent sample; else None (also on an envelope pair). `master_db` is a
    peak, not an average: a short loud stretch inside a much longer uncovered span must still be
    filled, and an average over the whole span would dilute it under AUDIBLE_DB. Searched up to
    `limit_s`."""
    if isinstance(c, EnvelopeSignal) or isinstance(m, EnvelopeSignal):
        return None
    shift = off / 1000.0
    head = np.flatnonzero(np.abs(c[:max(0, int(round((limit_s + shift) * WALK_RATE)))])
                          >= DIGITAL_SILENCE_AMPLITUDE)
    if not len(head):
        return None
    content_s = head[0] / WALK_RATE - shift
    if content_s <= 0.0:
        return None
    span = m[:int(np.ceil(content_s * WALK_RATE))]
    loud = np.flatnonzero(np.abs(span) >= AUDIBLE_AMPLITUDE)
    if not len(loud):
        return None
    # `loud` already holds only samples at or above AUDIBLE_AMPLITUDE: no extra threshold
    # check needed, and no minimum duration -- any genuine uncovered sound must be filled.
    return {"content_edge_s": round(float(content_s), 4),
            "master_first_sound_s": round(float(loud[0]) / WALK_RATE, 4),
            "uncovered_s": round(float(content_s - loud[0] / WALK_RATE), 4),
            "master_db": round(float(peak_db(span[loud])), 1)}


def quietest_instant(m, lo_s, hi_s, extra_s=0.0):
    """Return the F in [lo, hi] where the master is quietest at F and F + extra (20 ms windows,
    5 ms steps)."""
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


def _candidate_db(c, times, t0, t1, off_ms, win):
    """Candidate PEAK level (dB) of each fine window at `times`, read under `off_ms`.

    A peak, not an RMS average: a decaying sting (logo, sting) crosses the audibility floor on
    its mean energy well before its last audible sample, and extending a head/tail edge past
    that point leaves candidate-only sound delivered over a digitally silent master (measured:
    ids 26/29, 88 / 173 ms of candidate sting audible at -49 to -59 dBFS peak over the master's
    digital floor). The audibility rule (`AUDIBLE_DB`) is itself a peak threshold, so the quiet
    test this feeds must use the same measure on both sides.
    """
    cs = _shifted(c, t0, t1, off_ms).astype(np.float64)
    w = int(round(win * WALK_RATE))
    starts = np.clip(np.round((times - t0) * WALK_RATE).astype(int), 0, max(len(cs) - w, 0))
    ends = np.minimum(starts + w, len(cs))
    peaks = np.array([float(np.max(np.abs(cs[s:e]))) if e > s else 0.0
                      for s, e in zip(starts, ends)])
    return 20 * np.log10(np.maximum(peaks, 1e-10))


# A non-reference track's zone offset is a median over at most this many windows (the hop widens
# on long zones); with in-level MAD around 0.15 ms that stays well under one millisecond.
ZONE_OFFSET_MAX_WINDOWS = 120


def zone_offset(m, c, t0, t1, seed):
    """Measure this track's offset over master [t0, t1), seeded by the zone's reference offset.

    Returns {offset_ms (dominant level median, None if nothing measured), n_ok, levels, counts};
    more than one level means the track changes inside the zone."""
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
    """Classify and localise each pair of consecutive levels.

    A pair is a `change_point` when both the median jump and the local jump at the boundary
    (`off_last_ms` before, `off_first_ms` after) reach JUMP_MS, otherwise `sub_threshold` (a
    drifting offset has no step to find). Change points get coarse then fine edges."""
    points = []
    for before, after in zip(found, found[1:]):
        jump = after["off_ms"] - before["off_ms"]
        local_jump = (after.get("off_first_ms", after["off_ms"])
                      - before.get("off_last_ms", before["off_ms"]))
        point = {"a_ms": before["off_ms"], "b_ms": after["off_ms"], "jump_ms": round(jump, 3),
                 "local_jump_ms": round(local_jump, 3),
                 "level_before": before, "level_after": after}
        if abs(jump) < JUMP_MS or abs(local_jump) < JUMP_MS:
            point["kind"] = "sub_threshold"
            points.append(point)
            continue
        point["kind"] = "change_point"
        # Search one extra walk hop on each side: on weakly matching (remixed) audio the 0.4 s
        # claims can fall just outside the bounding windows.
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
    """Write the walk's window counts, levels and change points to the dev log."""
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    tools.dev_log(f"audio_walk: windows={len(rows)} counts={counts} seconds={round(seconds, 1)} "
                  f"levels={[(lv['t_first'], lv['t_last'], lv['off_ms'], lv['mad_ms']) for lv in found]} "
                  f"for {candidate_path}\n")
    for point in points:
        tools.dev_log(f"audio_walk: {point['kind']} a_ms={point['a_ms']} b_ms={point['b_ms']} "
                      f"jump_ms={point['jump_ms']} local_jump_ms={point.get('local_jump_ms')} "
                      f"edges={point.get('edges')} "
                      f"for {candidate_path}\n")
