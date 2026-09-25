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

def read_on_file_clock(video_obj, audio, audio_filter=None, scale=Decimal(1)):
    """One track, mono float32 at WALK_RATE, on the FILE clock: `merge_video_chimeric.
    read_track_samples` (bounded, logged, the assembly's own convention) with the stream's
    container start_time -- times `scale` on a rate pair, as the assembly reads it -- prepended
    as zeros (a negative start drops samples)."""
    import merge_video_chimeric
    samples = merge_video_chimeric.read_track_samples(
        video_obj.filePath, int(audio["StreamOrder"]), WALK_RATE, audio_filter=audio_filter)
    start_ms = merge_video_chimeric.get_stream_start_ms(audio) * Decimal(scale)
    pad = int(round(float(start_ms) * WALK_RATE / 1000.0))
    if pad > 0:
        return np.concatenate([np.zeros(pad, np.float32), samples])
    return samples[-pad:] if pad < 0 else samples


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
    rest = np.concatenate([curve[:max(k - guard, 0)], curve[k + guard + 1:]])
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

def walk(m, c, seeds, win_s=WALK_WINDOW_S, hop_s=WALK_HOP_S, search_ms=WALK_SEARCH_MS,
         t0=0.0, t1=None):
    """Every window of the master: its offset (sub-sample), NCC, second peak and status.

    CONTINUITY FIRST, THEN THE SEEDS (a deviation from the prototype, measured): each window is
    first searched around the offset the previous measured window read, and that reading is
    kept while it is `ok`; only a window the current offset no longer explains is searched
    around every seed, best NCC wins. MEASURED, id 134 (Mai-HiME 12): best-NCC-over-seeds jumps
    to +181889 ms for 88 windows (NCC 0.998, the candidate carries that audio twice) where the
    picture continues under -3097 ms. A real edit makes the current offset STOP matching, so
    continuity costs nothing at a real change point; a sub-quantum step moves the peak inside
    the continuity search (+/-150 ms) and is read as the new value, never smoothed."""
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
    return rows


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
                    "ncc_med": round(float(np.median(level["nccs"])), 4)})
    merged = []
    for level in out:
        if merged and abs(level["off_ms"] - merged[-1]["off_ms"]) < LEVEL_TOLERANCE_MS:
            merged[-1]["t_last"] = level["t_last"]
            merged[-1]["n"] += level["n"]
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


def coarse_edges(m, c, a, b, t_lo, t_hi):
    """Edges of an a -> b change inside master [t_lo, t_hi] from 0.4 s fixed-lag windows:
    (edge_A, edge_B) or None. Only a SEED for `fine_edges`, never an edge by itself (memo 4.1:
    coarse edges are biased inward by 0.1-0.3 s)."""
    profile = []
    t = t_lo
    while t + SHORT_WIN_S <= t_hi:
        i0 = int(round(t * WALK_RATE))
        mdb = rms_db(m[i0:i0 + int(SHORT_WIN_S * WALK_RATE)])
        na, _ = _fixed_ncc(m, c, t, a, SHORT_WIN_S)
        nb, _ = _fixed_ncc(m, c, t, b, SHORT_WIN_S)
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


def _residuals(m, c, offsets, t0, t1):
    """Per 20 ms window (hop 5 ms): times, master dB, and the gain-fitted residual ratio under
    each offset of `offsets`."""
    start = int(round(t0 * WALK_RATE))
    x = m[start:start + int(round((t1 - t0) * WALK_RATE))].astype(np.float64)
    w, h = int(FINE_WIN_S * WALK_RATE), int(FINE_HOP_S * WALK_RATE)
    kernel = np.ones(w)
    local = np.ones(int(0.4 * WALK_RATE))

    def smooth(v):
        return np.convolve(v, kernel, "valid")[::h]
    energy = smooth(x * x)
    residuals = []
    for off in offsets:
        cs = _shifted(c, t0, t1, off)[:len(x)]
        gain = np.convolve(x * cs, local, "same") / np.maximum(
            np.convolve(cs * cs, local, "same"), 1e-12)
        gain = np.clip(gain, 0.25, 4.0)
        residuals.append(smooth((x - gain * cs) ** 2) / np.maximum(energy, 1e-12))
    mdb = 10 * np.log10(np.maximum(energy / w, 1e-20))
    return t0 + np.arange(len(energy)) * FINE_HOP_S, mdb, residuals


def _ncc_profile(m, c, offsets, t0, t1, win_s=0.1):
    times = np.arange(t0, t1 - win_s, FINE_HOP_S)
    mdb = np.array([rms_db(m[int(round(t * WALK_RATE)):int(round(t * WALK_RATE))
                             + int(win_s * WALK_RATE)]) for t in times])
    nccs = [np.array([(_fixed_ncc(m, c, t, off, win_s, 0.5)[0] or 0.0) for t in times])
            for off in offsets]
    return times, mdb, nccs


def fine_edges(m, c, a, b, coarse_a, coarse_b, margin=0.6):
    """The sharp edges of an a -> b change (memo 1.4): edge_A = the last master instant read at
    a, edge_B = the first read at b; `extra_s` = max(0, a - b) of master content the candidate
    lacks; for a deletion the feasible fill-start interval (fill [F, F + extra] with F >= A,
    F + extra <= B, covering the audible master-only sound), for an addition the cut-instant
    interval. `status` is `ok` or `edge_unmeasurable`."""
    t0 = min(coarse_a, coarse_b) - margin
    t1 = max(coarse_a, coarse_b) + margin
    times, mdb, (ra, rb) = _residuals(m, c, (a, b), t0, t1)
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
    else:
        method, win = "ncc100ms", 0.1
        times, mdb, (na, nb) = _ncc_profile(m, c, (a, b), t0, t1, win)
        audible = mdb >= FINE_SILENT_DB
        is_a = audible & (na >= 0.8) & (na - nb >= 0.2)
        is_b = audible & (nb >= 0.8) & (nb - na >= 0.2)
        good_a, good_b = na >= 0.8, nb >= 0.8
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
    edge_a = float(times[a_idx[-1]] + win)
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
    rows = walk(m, c, [seed], hop_s=hop, t0=t0, t1=t1)
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
        coarse = coarse_edges(m, c, before["off_ms"], after["off_ms"], before["t_last"],
                              after["t_first"] + WALK_WINDOW_S)
        if coarse is None:
            point["edges"] = {"status": "edge_unmeasurable", "stage": "coarse"}
        else:
            point["edges"] = fine_edges(m, c, before["off_ms"], after["off_ms"], *coarse)
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
