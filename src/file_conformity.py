"""Conformity check of a media file (master, candidate or VMSAM output).

check_file(path) measures one file and returns a Report; it never modifies the
file. CLI: python3 -m file_conformity <file> [--full] [--json] [--reference F]
Exit code: 0 conforming, 1 warnings only, 2 at least one error, 3 unreadable.

Check families:
  1. Container: declared vs measured ends, start times, flags, tags, chapters.
  2. Content extent: per-track audio envelope, first/last sound, silent islands.
  3. Coherence: fingerprint similarity and lag between audio tracks.
  4. Stream integrity: strict decode of seeded windows, or the whole file.
  5. Reference (optional): where the common audio content ends on each side.

Silence is -60 dBFS on the loudest 10 ms block of each second (digital silence
is ~-121 dB). Sampled integrity can miss a single-point defect; mode='full' is
exhaustive.
"""

import argparse
import concurrent.futures
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from dataclasses import dataclass, field, asdict

import numpy as np

# ---------------------------------------------------------------- constants

ENV_SR = 8000
ENV_BLOCK = 80                # 10 ms at 8 kHz
BLOCKS_PER_S = ENV_SR // ENV_BLOCK
SILENCE_DB = -60.0                # digital silence is ~-121 dB
ISLAND_GAP_S = 60
CLICK_S = 3
TAIL_RULE_S = 300.0               # a gap over 5 min running to the end is a cut
EARLY_INFO_S = 30.0
AFTER_VIDEO_WARNING_S = 10.0
AFTER_VIDEO_ERROR_S = 60.0        # healthy files end audio < ~45 s after the video
LONG_INTERIOR_SILENCE_S = 300.0
SIBLING_RATIO = 0.5               # damaged tracks stop near 25 %, healthy ones above 90 %
DURATION_MISMATCH_S = 2.0
DURATION_MISMATCH_FRAC = 0.005
START_OFFSET_INFO_S = 0.05
START_OFFSET_WARNING_S = 10.0
SUB_AFTER_VIDEO_S = 5.0

WIN_SR = 16000
FP_WINDOW_S = 60.0
XCORR_WINDOW_S = 30.0
XCORR_MAX_LAG_S = 1.0
XCORR_MIN_CORR = 0.5
COHERENCE_FRACTIONS = (0.3, 0.5, 0.7)   # away from opening and ending themes
FP_SAME_CONTENT = 0.90            # fingerprint similarity of same content
FP_REFERENCE_MATCH = 0.75
FP_ITEM_S = 0.1238            # chromaprint item period (4096 / 11025 / 3)
DESYNC_INFO_MS = 30.0
DESYNC_WARNING_MS = 70.0          # healthy same-language pairs stay under ~63 ms

INTEGRITY_WINDOW_S = 120.0
INTEGRITY_WINDOWS = 10
INTEGRITY_THREADS = 2
STRICT_FLAGS = ["-xerror", "-err_detect", "crccheck+bitstream+buffer+explode"]
IGNORED_ERROR_SUBSTRINGS = ("non monotonically increasing dts",      # muxer warning, not corruption
                            "Last message repeated")

SEVERITIES = ("error", "warning", "info")


# ---------------------------------------------------------------- report

@dataclass
class Check:
    """One check result: name, severity, French sentence and its numbers."""

    name: str
    severity: str
    sentence: str
    numbers: dict = field(default_factory=dict)


@dataclass
class Report:
    """All checks of one file, with timings and measured facts."""

    path: str
    mode: str
    reference: str = None
    checks: list = field(default_factory=list)
    wall_seconds: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)

    def add(self, name, severity, sentence, **numbers):
        """Append a check; extra keywords are its numbers."""
        assert severity in SEVERITIES, severity
        self.checks.append(Check(name, severity, sentence, _jsonable(numbers)))

    def count(self, severity):
        return sum(1 for c in self.checks if c.severity == severity)

    @property
    def exit_code(self):
        """Return 3 unreadable, 2 errors, 1 warnings, else 0."""
        if any(c.name == "unreadable" for c in self.checks):
            return 3
        if self.count("error"):
            return 2
        if self.count("warning"):
            return 1
        return 0

    @property
    def conforming(self):
        """Return True when no check is a warning or an error."""
        return self.exit_code == 0

    def names(self, severity=None):
        return [c.name for c in self.checks if severity is None or c.severity == severity]

    def to_dict(self):
        """Return the report as a JSON-ready dict with exit code and counts."""
        d = _jsonable(asdict(self))
        d["exit_code"] = self.exit_code
        d["counts"] = {s: self.count(s) for s in SEVERITIES}
        return d

    def to_json(self, **kw):
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)

    def summary_fr(self):
        """Return a French human-readable summary of the report."""
        label = {"error": "ERREUR", "warning": "AVERTISSEMENT", "info": "info"}
        verdict = {0: "CONFORME", 1: "CONFORME AVEC AVERTISSEMENTS",
                   2: "NON CONFORME", 3: "ILLISIBLE"}[self.exit_code]
        lines = [f"Fichier : {self.path}",
                 f"Mode : {'complet' if self.mode == 'full' else 'échantillonné'}"
                 + (f" ; référence : {self.reference}" if self.reference else ""),
                 f"Verdict : {verdict} ({self.count('error')} erreur(s), "
                 f"{self.count('warning')} avertissement(s), {self.count('info')} info(s)) "
                 f"en {self.wall_seconds.get('total', 0):.0f} s"]
        order = {s: i for i, s in enumerate(SEVERITIES)}
        for c in sorted(self.checks, key=lambda c: order[c.severity]):
            lines.append(f"  [{label[c.severity]}] {c.name} : {c.sentence}")
        return "\n".join(lines)


def _jsonable(o):
    """Convert numpy scalars and non-finite floats to JSON-safe values."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float):
        return None if not math.isfinite(o) else round(o, 3)
    return o


def _fr(x, nd=1):
    """Format a number the French way (12 799,4)."""
    if x is None:
        return "?"
    s = f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")
    return s


# ---------------------------------------------------------------- tools

def _bin(name):
    return shutil.which(name) or name


def _run(cmd, timeout=None):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           stdin=subprocess.DEVNULL, timeout=timeout)
        return p.returncode, p.stdout, p.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as e:
        return -9, e.stdout or b"", "timeout after %s s" % timeout


def _parse_hms(s):
    try:
        h, m, sec = str(s).strip().split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    except (ValueError, AttributeError):
        return None


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _tag(stream, key):
    """Return a stream tag, case-insensitive, also matching KEY-<lang> variants."""
    for k, v in (stream.get("tags") or {}).items():
        if k.upper() == key or k.upper().startswith(key + "-"):
            return v
    return None


def probe(path):
    """Return ffprobe's streams, format and chapters as a dict.

    Raises:
        RuntimeError: ffprobe failed.
    """
    rc, out, err = _run([_bin("ffprobe"), "-v", "error", "-show_streams", "-show_format",
                         "-show_chapters", "-of", "json", path], timeout=300)
    if rc != 0:
        raise RuntimeError(f"ffprobe rc={rc}: {err.strip()[:300]}")
    return json.loads(out.decode("utf-8", "replace"))


def _track_label(s):
    lang = _tag(s, "LANGUAGE") or "und"
    return f"{s.get('codec_type', '?')[0]}#{s['index']}({lang},{s.get('codec_name', '?')})"


# ---------------------------------------------------------------- packet ends

def _packet_scan(path, select=None, interval=None, gaps=False, timeout=None):
    """Scan packets for each stream's end and, optionally, its timestamp jumps.

    Returns:
        (ends, jumps): ends maps stream index to last pts + duration; jumps maps
        it to (count, max, first) of jumps over 1 s when `gaps` is set.
    """
    cmd = [_bin("ffprobe"), "-v", "error"]
    if select is not None:
        cmd += ["-select_streams", str(select)]
    if interval:
        cmd += ["-read_intervals", interval]
    cmd += ["-show_entries", "packet=stream_index,pts_time,dts_time,duration_time",
            "-of", "csv=p=0", path]
    ends, last, jumps = {}, {}, {}
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL)
    t0 = time.time()
    for raw in p.stdout:
        parts = raw.decode("ascii", "replace").strip().split(",")
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        t = _num(parts[1]) if parts[1] not in ("", "N/A") else _num(parts[2])
        if t is None:
            continue
        d = _num(parts[3]) or 0.0
        e = t + d
        if e > ends.get(idx, -1e18):
            ends[idx] = e
        if gaps:
            prev = last.get(idx)
            if prev is not None and t - prev > 1.0:
                n, mx, first = jumps.get(idx, (0, 0.0, t))
                jumps[idx] = (n + 1, max(mx, t - prev), first)
            last[idx] = max(e, prev or -1e18)
        if timeout and time.time() - t0 > timeout:
            p.kill()
            break
    p.stdout.close()
    p.wait()
    return ends, jumps


# ---------------------------------------------------------------- envelopes

def _envelopes(path, audio_positions, threads, workdir, log):
    """Decode every audio track once to a 10 ms mean-square envelope.

    One ffmpeg writes 8 kHz mono to one fifo per track; gaps are filled from
    timestamps (aresample async).

    Returns:
        ({audio position: float32 array}, ffmpeg return code, stderr).
    """
    fifos = []
    for k in audio_positions:
        f = os.path.join(workdir, f"env_{k}.fifo")
        os.mkfifo(f)
        fifos.append(f)
    results = {}

    def reader(k, f):
        chunks, rest = [], b""
        step = ENV_BLOCK * 2 * 1000
        with open(f, "rb", buffering=0) as fh:
            while True:
                b = fh.read(1 << 17)
                if not b:
                    break
                rest += b
                n = (len(rest) // (ENV_BLOCK * 2)) * (ENV_BLOCK * 2)
                if n >= step or (n and len(b) == 0):
                    x = np.frombuffer(rest[:n], dtype="<i2").astype(np.float32) / 32768.0
                    chunks.append((x.reshape(-1, ENV_BLOCK) ** 2).mean(axis=1))
                    rest = rest[n:]
        n = (len(rest) // (ENV_BLOCK * 2)) * (ENV_BLOCK * 2)
        if n:
            x = np.frombuffer(rest[:n], dtype="<i2").astype(np.float32) / 32768.0
            chunks.append((x.reshape(-1, ENV_BLOCK) ** 2).mean(axis=1))
        results[k] = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    threads_r = [threading.Thread(target=reader, args=(k, f), daemon=True)
                 for k, f in zip(audio_positions, fifos)]
    for t in threads_r:
        t.start()
    cmd = [_bin("ffmpeg"), "-nostdin", "-v", "error", "-threads", str(threads), "-i", path]
    for k, f in zip(audio_positions, fifos):
        cmd += ["-map", f"0:a:{k}", "-af", f"aresample={ENV_SR}:async=1:first_pts=0",
                "-ac", "1", "-c:a", "pcm_s16le", "-f", "s16le", "-y", f]
    errf = os.path.join(workdir, "env.err")
    with open(errf, "wb") as ef:
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=ef, stdin=subprocess.DEVNULL)
    for t, f in zip(threads_r, fifos):
        t.join(timeout=5)
        if t.is_alive():             # ffmpeg died before opening this fifo: unblock the reader
            try:
                fd = os.open(f, os.O_WRONLY | os.O_NONBLOCK)
                os.close(fd)
            except OSError:
                pass
            t.join(timeout=30)
    for f in fifos:
        try:
            os.unlink(f)
        except OSError:
            pass
    with open(errf, "rb") as ef:
        err = ef.read().decode("utf-8", "replace")
    if p.returncode != 0:
        log(f"conformity: envelope decode rc={p.returncode}: {err.strip()[-300:]}")
    return results, p.returncode, err


def _extent(ms):
    """Summarize per-second content of a 10 ms mean-square envelope.

    Returns:
        dict of length, first / last content second, content and silent totals,
        islands, longest interior gap, clicks and median level.
    """
    n_s = len(ms) // BLOCKS_PER_S
    length_s = len(ms) / BLOCKS_PER_S
    if n_s == 0:
        return dict(length_s=length_s, first=None, last=None, content_s=0.0,
                    silent_s=length_s, islands=[], max_interior_gap_s=0.0, clicks=[], level_db=None)
    peak = ms[:n_s * BLOCKS_PER_S].reshape(n_s, BLOCKS_PER_S).max(axis=1)
    db = 10.0 * np.log10(np.maximum(peak, 1e-14))
    content = db > SILENCE_DB
    idx = np.flatnonzero(content)
    if idx.size == 0:
        return dict(length_s=length_s, first=None, last=None, content_s=0.0,
                    silent_s=length_s, islands=[], max_interior_gap_s=0.0, clicks=[],
                    level_db=float(np.max(db)))
    brk = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[brk + 1]))
    ends = np.concatenate((idx[brk], [idx[-1]])) + 1
    gaps = starts[1:] - ends[:-1]
    islands, cur = [], [int(starts[0]), int(ends[0])]
    for s, e, g in zip(starts[1:], ends[1:], gaps):
        if g >= ISLAND_GAP_S:
            islands.append(cur)
            cur = [int(s), int(e)]
        else:
            cur[1] = int(e)
    islands.append(cur)
    # Clicks: a run shorter than CLICK_S between two silences of >= ISLAND_GAP_S
    # (isolated blips inside digital silence) is counted, not kept as an island.
    clicks = [isl for isl in islands if isl[1] - isl[0] < CLICK_S]
    kept = [isl for isl in islands if isl[1] - isl[0] >= CLICK_S] or islands
    if len(kept) < len(islands):
        islands = kept
        idx = np.concatenate([np.arange(x, y) for x, y in islands])
    else:
        clicks = []
    interior =[islands[i + 1][0] - islands[i][1] for i in range(len(islands) - 1)]
    return dict(length_s=length_s, first=float(idx[0]), last=float(idx[-1] + 1),
                content_s=float(idx.size), silent_s=float(length_s - idx.size),
                islands=[[float(a), float(b)] for a, b in islands],
                max_interior_gap_s=float(max(interior) if interior else 0.0),
                clicks=[[float(a), float(b)] for a, b in clicks],
                level_db=float(np.median(db[content])))


# ---------------------------------------------------------------- windows

def _extract_windows(path, audio_positions, start, dur, workdir, tag, threads):
    """Extract one window of every audio track to a 16 kHz mono wav (one ffmpeg)."""
    outs = {k: os.path.join(workdir, f"w_{tag}_{k}.wav") for k in audio_positions}
    cmd = [_bin("ffmpeg"), "-nostdin", "-v", "error", "-threads", str(threads),
           "-ss", f"{max(0.0, start):.3f}", "-t", f"{dur:.3f}", "-i", path]
    for k in audio_positions:
        cmd += ["-map", f"0:a:{k}", "-af", f"aresample={WIN_SR}:async=1", "-ac", "1",
                "-c:a", "pcm_s16le", "-y", outs[k]]
    _run(cmd, timeout=600)
    res = {}
    for k, f in outs.items():
        if os.path.exists(f) and os.path.getsize(f) > 44 + WIN_SR:
            with open(f, "rb") as fh:
                data = fh.read()
            # skip the RIFF header: find the 'data' chunk
            i = data.find(b"data")
            x = np.frombuffer(data[i + 8:] if i >= 0 else data[44:], dtype="<i2")
            x = x.astype(np.float32) / 32768.0
            res[k] = (f, x)
    return res


def _fpcalc(wav, length):
    """Return the raw chromaprint fingerprint of a wav, or None."""
    rc, out, _ = _run([_bin("fpcalc"), "-raw", "-length", str(int(length) + 1), wav], timeout=120)
    if rc != 0:
        return None
    for line in out.decode("ascii", "replace").splitlines():
        if line.startswith("FINGERPRINT="):
            try:
                return np.array([int(v) for v in line[12:].split(",") if v], dtype=np.uint32)
            except ValueError:
                return None
    return None


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _fp_similarity(a, b, max_shift, min_overlap):
    """Return the best fingerprint similarity over shifts, and its shift.

    Similarity is mean(1 - popcount(a[i] ^ b[i+s]) / 32).

    Returns:
        (similarity, shift_items), or (None, None) when the overlap is too short.
    """
    if a is None or b is None or len(a) < min_overlap or len(b) < min_overlap:
        return None, None
    best, best_s = -1.0, 0
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            x, y = a[: max(0, len(b) - s)], b[s: s + len(a)]
        else:
            x, y = a[-s: -s + len(b)], b[: max(0, len(a) + s)]
        n = min(len(x), len(y))
        if n < min_overlap:
            continue
        v = np.bitwise_xor(x[:n], y[:n])
        bits = _POP8[v.view(np.uint8)].reshape(-1, 4).sum(axis=1)
        sim = 1.0 - float(bits.mean()) / 32.0
        if sim > best:
            best, best_s = sim, s
    return (best, best_s) if best >= 0 else (None, None)


def _xcorr(a, b, sr, max_lag_s):
    """Return (lag_ms, corr) of the normalised cross-correlation peak.

    A positive lag means the same sound arrives later in `a` than in `b`.
    """
    n = min(len(a), len(b))
    if n < sr * 2:
        return None, None
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    na, nb = float(np.sqrt((a * a).sum())), float(np.sqrt((b * b).sum()))
    if na < 1e-6 or nb < 1e-6:
        return None, None
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft)
    L = int(max_lag_s * sr)
    seg = np.concatenate((c[-L:], c[: L + 1]))
    i = int(np.argmax(seg))
    lag = i - L
    frac = 0.0
    if 0 < i < len(seg) - 1:
        y0, y1, y2 = seg[i - 1], seg[i], seg[i + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            frac = 0.5 * (y0 - y2) / den
    return (lag + frac) * 1000.0 / sr, float(seg[i] / (na * nb))


def _rms_db(x):
    """Return the RMS level of samples in dB."""
    if x is None or len(x) == 0:
        return -200.0
    return 10.0 * math.log10(max(float((x.astype(np.float64) ** 2).mean()), 1e-20))


# ---------------------------------------------------------------- integrity

def _strict_decode(path, start, dur, threads, timeout, maps=None):
    """Strict-decode one window (or the whole file); return rc, timing and error lines.

    Args:
        maps: explicit stream indices. Map only streams with packets in the
            window, otherwise ffmpeg never reaches -t and decodes to EOF.
    """
    cmd = [_bin("ffmpeg"), "-nostdin", "-v", "error"] + STRICT_FLAGS + ["-threads", str(threads)]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}"]
    cmd += ["-i", path]
    if maps is None:
        cmd += ["-map", "0:v:0?", "-map", "0:a?"]
    else:
        for m in maps:
            cmd += ["-map", f"0:{m}"]
    cmd += ["-f", "null", "-"]
    t0 = time.time()
    rc, _, err = _run(cmd, timeout=timeout)
    lines = [l for l in err.splitlines()
             if l.strip() and not any(s in l for s in IGNORED_ERROR_SUBSTRINGS)]
    return dict(start=start, dur=dur, maps=maps, rc=rc, seconds=round(time.time() - t0, 1),
                first=lines[0][:300] if lines else None,
                last=lines[-1][:300] if lines else None, n_lines=len(lines))


def integrity_windows(duration, k, seed, window=INTEGRITY_WINDOW_S):
    """Return k seeded window starts, one drawn uniformly in each of k equal strata.

    Returns None when the file is shorter than k windows.
    """
    if duration is None or duration <= k * window:
        return None
    rng = random.Random(seed)
    stratum = duration / k
    return [round(i * stratum + rng.uniform(0.0, max(0.0, stratum - window)), 3)
            for i in range(k)]


# ---------------------------------------------------------------- analysis

class _Analysis:
    """Measurements of one file, used for both the checked file and the reference."""

    def __init__(self, path, mode, threads, workdir, log, tag):
        self.path, self.mode, self.threads, self.workdir, self.log = path, mode, threads, workdir, log
        self.tag = tag
        self.info = probe(path)
        self.streams = self.info.get("streams", [])
        self.format = self.info.get("format", {})
        self.fmt_dur = _num(self.format.get("duration"))
        self.audio = [s for s in self.streams if s.get("codec_type") == "audio"]
        self.video = [s for s in self.streams if s.get("codec_type") == "video"
                      and not (s.get("disposition") or {}).get("attached_pic")]
        self.subs = [s for s in self.streams if s.get("codec_type") == "subtitle"]
        self.ends = {}          # stream index -> (end_s, source)
        self.jumps = {}
        self.env = {}
        self.ext = {}
        self.windows = {}       # fraction -> {apos: (wav, samples)}
        self.fps = {}           # fraction -> {apos: fingerprint}
        self.env_rc = 0

    def lang(self, apos):
        return (_tag(self.audio[apos], "LANGUAGE") or "und").lower()

    def label(self, apos):
        return _track_label(self.audio[apos])

    def declared_end(self, s):
        """Return (end, source, start_time) declared for a stream."""
        d = _parse_hms(_tag(s, "DURATION"))
        src = "tag DURATION"
        if d is None:
            d, src = _num(s.get("duration")), "stream.duration"
        if d is None:
            d, src = self.fmt_dur, "format.duration"
        start = _num(s.get("start_time")) or 0.0
        return d, src, start

    def measure_ends(self):
        """Measure each stream's last packet end (full scan or windows near the end)."""
        t0 = time.time()
        if self.mode == "full":
            ends, self.jumps = _packet_scan(self.path, gaps=True)
            for i, e in ends.items():
                self.ends[i] = (e, "packets (full scan)")
        else:
            if self.fmt_dur:
                ends, _ = _packet_scan(self.path, interval=f"{max(0, self.fmt_dur - 90):.3f}%")
                for i, e in ends.items():
                    self.ends[i] = (e, "packets (tail window)")
            # Remaining streams: scan around their declared ends, grouping ends
            # within 300 s into one scan (files can carry many tracks).
            pending = []
            for s in self.streams:
                i = s["index"]
                if i in self.ends or s.get("codec_type") not in ("video", "audio", "subtitle"):
                    continue
                if (s.get("disposition") or {}).get("attached_pic"):
                    continue
                d, _, _ = self.declared_end(s)
                if d is None:
                    continue
                back = 600 if s.get("codec_type") == "subtitle" else 90
                pending.append((d, back, i))
            pending.sort()
            clusters = []
            for d, back, i in pending:
                if clusters and d - clusters[-1][0][0] <= 300:
                    clusters[-1].append((d, back, i))
                else:
                    clusters.append([(d, back, i)])
            for cl in clusters:
                lo = max(0.0, min(d - b for d, b, _ in cl))
                hi = max(d for d, _, _ in cl) + 120
                ends, _ = _packet_scan(self.path, interval=f"{lo:.3f}%{hi:.3f}")
                for _, _, i in cl:
                    if i in ends:
                        self.ends[i] = (ends[i], "packets (window at declared end)")
        self.log(f"conformity[{self.tag}]: packet ends measured in {time.time() - t0:.1f} s")

    def video_end(self):
        """Return the first video stream's measured (else declared) end."""
        if not self.video:
            return None
        v = self.video[0]
        if v["index"] in self.ends:
            return self.ends[v["index"]][0]
        return self.declared_end(v)[0]

    def measure_envelopes(self):
        """Decode the audio envelopes and compute each track's content extent."""
        if not self.audio:
            return
        t0 = time.time()
        self.env, self.env_rc, _ = _envelopes(self.path, list(range(len(self.audio))),
                                              self.threads, self.workdir, self.log)
        for k, ms in self.env.items():
            self.ext[k] = _extent(ms)
        self.log(f"conformity[{self.tag}]: {len(self.audio)} audio envelope(s) in "
                 f"{time.time() - t0:.1f} s")

    def measure_windows(self, fractions, span):
        """Extract audio windows at fractions of `span` seconds and fingerprint them."""
        if not self.audio or not span:
            return
        t0 = time.time()
        for fr in fractions:
            start = fr * span - FP_WINDOW_S / 2
            w = _extract_windows(self.path, list(range(len(self.audio))), start, FP_WINDOW_S,
                                 self.workdir, f"{self.tag}_{fr}", self.threads)
            self.windows[fr] = w
            self.fps[fr] = {}
            for k, (wav, x) in w.items():
                self.fps[fr][k] = _fpcalc(wav, FP_WINDOW_S) if _rms_db(x) > -50 else None
                try:
                    os.unlink(wav)
                except OSError:
                    pass
        self.log(f"conformity[{self.tag}]: coherence windows at {list(fractions)} in "
                 f"{time.time() - t0:.1f} s")


# ---------------------------------------------------------------- checks

def _check_container(a, rep):
    """Family 1: container metadata, track ends, flags, duplicates and chapters."""
    vend = a.video_end()
    rep.facts["video_end_s"] = vend
    rep.facts["format_duration_s"] = a.fmt_dur
    rep.facts["tracks"] = []
    if not a.video:
        rep.add("no_video", "warning", "Le fichier ne contient aucune piste vidéo.")
    if not a.audio:
        rep.add("no_audio", "error", "Le fichier ne contient aucune piste audio.")
    vstart = _num(a.video[0].get("start_time")) if a.video else 0.0
    for s in a.streams:
        ct = s.get("codec_type")
        if ct not in ("video", "audio", "subtitle") or (s.get("disposition") or {}).get("attached_pic"):
            continue
        dec, src, start = a.declared_end(s)
        meas, msrc = a.ends.get(s["index"], (None, None))
        if ct == "audio":
            k = a.audio.index(s)
            if k in a.ext and a.ext[k]["length_s"] > 0:
                meas, msrc = a.ext[k]["length_s"], "decoded samples"
        lab = _track_label(s)
        row = dict(track=lab, index=s["index"], codec=s.get("codec_name"),
                   lang=_tag(s, "LANGUAGE") or "und", channels=s.get("channels"),
                   declared_end_s=dec, declared_source=src, measured_end_s=meas,
                   measured_source=msrc, start_time_s=start,
                   default=(s.get("disposition") or {}).get("default"),
                   forced=(s.get("disposition") or {}).get("forced"))
        rep.facts["tracks"].append(_jsonable(row))
        if dec is not None and meas is not None and ct != "subtitle":
            diff = meas - dec
            if abs(diff) > max(DURATION_MISMATCH_S, DURATION_MISMATCH_FRAC * max(dec, meas)):
                rep.add("declared_duration_mismatch", "warning",
                        f"La piste {lab} déclare {_fr(dec)} s ({src}) mais sa mesure donne "
                        f"{_fr(meas)} s ({msrc}), écart {_fr(diff)} s.",
                        track=lab, declared_s=dec, measured_s=meas, diff_s=diff)
        if ct != "video" and vstart is not None and start is not None:
            off = start - vstart
            if abs(off) >= START_OFFSET_WARNING_S:
                rep.add("start_time_offset", "warning",
                        f"La piste {lab} commence {_fr(off, 3)} s après le début de la vidéo.",
                        track=lab, offset_s=off)
            elif abs(off) >= START_OFFSET_INFO_S and ct == "audio":
                rep.add("start_time_offset", "info",
                        f"La piste {lab} a un start_time décalé de {_fr(off * 1000, 0)} ms "
                        f"par rapport à la vidéo.", track=lab, offset_s=off)
        if (_tag(s, "LANGUAGE") or "und").lower() in ("und", "") and ct in ("audio", "subtitle"):
            rep.add("language_undetermined", "info",
                    f"La piste {lab} n'a pas de langue déclarée (« und »).", track=lab)
        if ct == "audio" and (s.get("disposition") or {}).get("forced"):
            rep.add("audio_forced_flag", "info", f"La piste audio {lab} porte le drapeau « forcé ».",
                    track=lab)
        if ct == "subtitle" and vend is not None and meas is not None \
                and meas > vend + SUB_AFTER_VIDEO_S:
            late_s = meas - vend
            rep.add("subtitle_after_video_end",
                    "warning" if late_s > AFTER_VIDEO_ERROR_S else "info",
                    f"Les sous-titres {lab} continuent jusqu'à {_fr(meas)} s, "
                    f"{_fr(meas - vend)} s après la fin de la vidéo ({_fr(vend)} s).",
                    track=lab, last_cue_end_s=meas, video_end_s=vend)
    defaults = [s for s in a.audio if (s.get("disposition") or {}).get("default")]
    if a.audio and not defaults:
        rep.add("no_default_audio", "info", "Aucune piste audio n'est marquée par défaut.")
    elif len(defaults) > 1:
        rep.add("several_default_audio", "info",
                f"{len(defaults)} pistes audio sont marquées par défaut ; le lecteur choisira "
                f"la première.", tracks=[_track_label(s) for s in defaults])
    seen = {}
    for k, s in enumerate(a.audio):
        dur = a.ext.get(k, {}).get("length_s") or a.declared_end(s)[0] or 0
        key = (s.get("codec_name"), a.lang(k), s.get("channels"))
        for k2, dur2 in seen.get(key, []):
            if abs(dur - dur2) < 0.5:
                rep.add("duplicate_track_metadata", "info",
                        f"Les pistes {a.label(k2)} et {a.label(k)} ont le même codec, la même "
                        f"langue, le même nombre de canaux et la même durée.",
                        tracks=[a.label(k2), a.label(k)])
        seen.setdefault(key, []).append((k, dur))
    chaps = a.info.get("chapters") or []
    if vend is not None:
        late = [c for c in chaps if (_num(c.get("start_time")) or 0) > vend + 1.0]
        if late:
            rep.add("chapter_after_video_end", "warning",
                    f"{len(late)} chapitre(s) commencent après la fin de la vidéo ({_fr(vend)} s), "
                    f"le dernier à {_fr(_num(late[-1].get('start_time')))} s.",
                    count=len(late), video_end_s=vend,
                    last_start_s=_num(late[-1].get("start_time")))
        elif chaps and (_num(chaps[-1].get("end_time")) or 0) > vend + AFTER_VIDEO_ERROR_S:
            rep.add("chapter_after_video_end", "warning",
                    f"Le dernier chapitre finit à {_fr(_num(chaps[-1].get('end_time')))} s, "
                    f"après la fin de la vidéo ({_fr(vend)} s).",
                    end_s=_num(chaps[-1].get("end_time")), video_end_s=vend)
    for i, (n, mx, first) in sorted(a.jumps.items()):
        s = next((x for x in a.streams if x["index"] == i), None)
        if s is None or s.get("codec_type") not in ("audio", "video"):
            continue
        rep.add("timestamp_gaps", "warning",
                f"La piste {_track_label(s)} a {n} saut(s) de timestamps de plus d'une seconde "
                f"(max {_fr(mx)} s, le premier à {_fr(first)} s).",
                track=_track_label(s), count=n, max_s=mx, first_at_s=first)


def _check_extent(a, rep):
    """Family 2: silent tracks, islands, content after or before the video end."""
    vend = a.video_end()
    lasts = {k: e["last"] for k, e in a.ext.items() if e["last"] is not None}
    rep.facts["audio_extents"] = {a.label(k): e for k, e in a.ext.items()}
    early = {}
    for k, e in sorted(a.ext.items()):
        lab = a.label(k)
        if e["last"] is None:
            rep.add("audio_track_silent", "error",
                    f"La piste {lab} ne contient que du silence numérique "
                    f"(aucune seconde au-dessus de {SILENCE_DB:.0f} dB).", track=lab,
                    length_s=e["length_s"])
            continue
        if len(e["islands"]) > 1:
            rep.add("content_islands", "info",
                    f"La piste {lab} a {len(e['islands'])} îlots de contenu séparés par au moins "
                    f"{ISLAND_GAP_S} s de silence : "
                    + ", ".join(f"[{_fr(x, 0)} ; {_fr(y, 0)}]" for x, y in e["islands"][:6]) + ".",
                    track=lab, islands=e["islands"])
        if e["max_interior_gap_s"] >= LONG_INTERIOR_SILENCE_S:
            rep.add("long_interior_silence", "warning",
                    f"La piste {lab} contient un silence numérique intérieur de "
                    f"{_fr(e['max_interior_gap_s'], 0)} s, puis du contenu reprend.",
                    track=lab, gap_s=e["max_interior_gap_s"], islands=e["islands"])
        if vend is None:
            continue
        after = e["last"] - vend
        if after > AFTER_VIDEO_ERROR_S or after > AFTER_VIDEO_WARNING_S:
            after_content = sum(max(0.0, y - max(x, vend)) for x, y in e["islands"])
            sev = "error" if after > AFTER_VIDEO_ERROR_S else "warning"
            rep.add("content_after_video_end", sev,
                    f"La piste {lab} a du contenu audio jusqu'à {_fr(e['last'])} s alors que la "
                    f"vidéo finit à {_fr(vend)} s ({_fr(after)} s après la dernière image)"
                    + (f" ; le dernier bloc [{_fr(e['islands'][-1][0], 0)} ; "
                       f"{_fr(e['islands'][-1][1], 0)}] suit {_fr(e['islands'][-1][0] - e['islands'][-2][1], 0)} s "
                       f"de silence numérique." if len(e["islands"]) > 1 else "."),
                    track=lab, content_last_s=e["last"], video_end_s=vend, after_s=after,
                    content_after_video_s=after_content, islands=e["islands"])
        gap = vend - e["last"]
        if gap > EARLY_INFO_S:
            early[k] = gap
    if vend is not None and early:
        all_early = len(early) == len(lasts) and all(g > TAIL_RULE_S for g in early.values())
        small = {k: g for k, g in early.items() if g <= TAIL_RULE_S}
        for k, gap in early.items():
            lab = a.label(k)
            if gap > TAIL_RULE_S and not all_early:
                rep.add("content_ends_early", "warning",
                        f"Le contenu de la piste {lab} s'arrête à {_fr(lasts[k])} s, "
                        f"{_fr(gap, 0)} s avant la fin de la vidéo ({_fr(vend)} s).",
                        track=lab, content_last_s=lasts[k], video_end_s=vend, early_s=gap)
        if small:
            labs = [a.label(k) for k in small]
            rep.add("content_ends_early", "info",
                    f"Le contenu audio de {len(small)} piste(s) s'arrête entre "
                    f"{_fr(min(small.values()), 0)} et {_fr(max(small.values()), 0)} s avant la fin "
                    f"de la vidéo ({_fr(vend)} s), sous le seuil de {TAIL_RULE_S:.0f} s.",
                    tracks=labs, early_s=[small[k] for k in small], video_end_s=vend)
        if all_early:
            ends = sorted(lasts.values())
            rep.add("content_ends_early", "error",
                    f"Toutes les pistes audio ({len(lasts)}) s'arrêtent "
                    + (f"à {_fr(ends[0])} s" if ends[-1] - ends[0] < 1.5 else
                       f"entre {_fr(ends[0])} s et {_fr(ends[-1])} s")
                    + f" alors que la vidéo continue jusqu'à {_fr(vend)} s "
                    f"({_fr(vend - ends[-1], 0)} s sans son) : signature d'une coupure.",
                    content_last_s=ends, video_end_s=vend, early_s=vend - ends[-1],
                    tracks=[a.label(k) for k in lasts])
    if len(lasts) >= 2:
        for k, last in lasts.items():
            others = [v for j, v in lasts.items() if j != k]
            med = statistics.median(others)
            if last < SIBLING_RATIO * med and med - last > TAIL_RULE_S:
                rep.add("track_much_shorter_than_siblings", "warning",
                        f"La piste {a.label(k)} s'arrête à {_fr(last)} s, soit "
                        f"{100 * last / med:.0f} % de la médiane des autres pistes audio "
                        f"({_fr(med)} s).",
                        track=a.label(k), content_last_s=last, siblings_median_s=med,
                        ratio=last / med)


def _check_coherence(a, rep):
    """Family 3: tag conflicts, duplicates and desync between audio track pairs."""
    n = len(a.audio)
    if n < 2:
        return
    fracs = sorted(a.fps)
    for i in range(n):
        for j in range(i + 1, n):
            li, lj = a.lang(i), a.lang(j)
            sims, lags = [], []
            for fr in fracs:
                fi, fj = a.fps[fr].get(i), a.fps[fr].get(j)
                s, sh = _fp_similarity(fi, fj, max_shift=12, min_overlap=200)
                if s is not None:
                    sims.append(round(s, 4))
                if li == lj and i in a.windows[fr] and j in a.windows[fr]:
                    xi, xj = a.windows[fr][i][1], a.windows[fr][j][1]
                    mid = len(xi) // 2
                    half = int(XCORR_WINDOW_S * WIN_SR / 2)
                    lag, corr = _xcorr(xi[max(0, mid - half): mid + half],
                                       xj[max(0, mid - half): mid + half], WIN_SR, XCORR_MAX_LAG_S)
                    if lag is not None:
                        lags.append((fr, round(lag, 1), round(corr, 3)))
            pair = f"{a.label(i)} / {a.label(j)}"
            rep.facts.setdefault("pairs", {})[pair] = dict(fp_similarity=sims, xcorr=lags)
            same =len(sims) >= 2 and min(sims) >= FP_SAME_CONTENT
            if same and li != lj:
                sev = "warning" if "und" in (li, lj) else "error"
                rep.add("audio_tag_conflict", sev,
                        f"Les pistes {pair} ont le même contenu (similarité d'empreinte "
                        f"{min(sims):.3f} sur {len(sims)} fenêtres) mais des langues différentes "
                        f"({li} / {lj}) : un des deux tags est faux.",
                        tracks=[a.label(i), a.label(j)], langs=[li, lj], similarity=sims)
            elif same and li == lj:
                d = abs((a.ext.get(i, {}).get("length_s") or 0) - (a.ext.get(j, {}).get("length_s") or 0))
                same_codec = (a.audio[i].get("codec_name") == a.audio[j].get("codec_name")
                              and a.audio[i].get("channels") == a.audio[j].get("channels"))
                if same_codec and d < 0.5 and min(sims) >= 0.99:
                    rep.add("duplicate_audio_track", "warning",
                            f"Les pistes {pair} sont des doublons : même langue, même codec, même "
                            f"durée et même contenu (similarité {min(sims):.3f}).",
                            tracks=[a.label(i), a.label(j)], similarity=sims)
            elif li == lj and li != "und" and len(sims) >= 2 and max(sims) < 0.75:
                rep.add("same_tag_different_content", "info",
                        f"Les pistes {pair} portent la même langue ({li}) mais leur contenu diffère "
                        f"(similarité max {max(sims):.3f}) : commentaire, version différente ou "
                        f"tag faux.", tracks=[a.label(i), a.label(j)], similarity=sims)
            good = [l for l in lags if l[2] >= XCORR_MIN_CORR]
            if li == lj and len(good) >= 2:
                med = statistics.median(l[1] for l in good)
                nums = dict(tracks=[a.label(i), a.label(j)], lag_ms=med, windows=lags)
                if abs(med) >= DESYNC_WARNING_MS:
                    rep.add("intertrack_desync", "warning",
                            f"Les pistes {pair} (même langue {li}) sont décalées de {med:+.0f} ms "
                            f"(médiane de {len(good)} fenêtres : "
                            + ", ".join(f"{l[1]:+.0f}" for l in good) + " ms).", **nums)
                elif abs(med) >= DESYNC_INFO_MS:
                    rep.add("intertrack_desync", "info",
                            f"Les pistes {pair} (même langue {li}) diffèrent de {med:+.0f} ms, "
                            f"sous le seuil de {DESYNC_WARNING_MS:.0f} ms.", **nums)
                else:
                    rep.facts.setdefault("intertrack_lags_ms", {})[pair] = med


def path_seed(path):
    """Return a stable seed from the path (Python's hash() is salted per run)."""
    return zlib.crc32(os.path.abspath(path).encode("utf-8", "surrogateescape"))


def _check_integrity(a, rep, mode, windows, seed, threads, jobs, log):
    """Strict-decode integrity check (family 4).

    Sampled mode decodes K seeded 120 s windows over the video's packet length,
    each as two processes (video only, audio only): a combined process past an
    audio end would decode to EOF.
    """
    t0 = time.time()
    vend = a.video_end() or a.fmt_dur
    if seed is None:
        seed = path_seed(a.path)
    starts = integrity_windows(vend, windows, seed) if mode == "sampled" else None

    def stream_end(s):
        k = a.audio.index(s) if s in a.audio else None
        if k is not None and a.ext.get(k, {}).get("length_s"):
            return a.ext[k]["length_s"]
        return a.ends.get(s["index"], (None,))[0] or a.declared_end(s)[0] or vend

    if starts is None:
        runs = [_strict_decode(a.path, None, None, threads, timeout=None)]
        how = "décodage strict complet"
    else:
        tasks = []
        for st in starts:
            if a.video and (stream_end(a.video[0]) or 0) > st + 1.0:
                tasks.append((st, [a.video[0]["index"]]))
            alive = [s["index"] for s in a.audio if (stream_end(s) or 0) > st + 1.0]
            if alive:
                tasks.append((st, alive))
        n_jobs = jobs if jobs is not None else len(tasks)
        log(f"conformity: integrity windows seed={seed} k={windows} starts={starts} "
            f"processes={len(tasks)} jobs={n_jobs} threads={threads}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, n_jobs)) as ex:
            runs = list(ex.map(lambda t: _strict_decode(a.path, t[0], INTEGRITY_WINDOW_S, threads,
                                                        timeout=1800, maps=t[1]), tasks))
        how = (f"décodage strict de {len(starts)} fenêtres de {INTEGRITY_WINDOW_S:.0f} s "
               f"({len(runs)} processus vidéo/audio, graine {seed})")
    # Fail on any decoder message, not only rc: corrupt files rarely hit a fatal packet.
    failed = [r for r in runs if r["rc"] != 0 or r["n_lines"] > 0]
    rep.wall_seconds["integrity"] = round(time.time() - t0, 1)
    nums = dict(mode=mode, windows_total=len(starts) if starts else 1,
                processes_total=len(runs), processes_failed=len(failed), seed=seed,
                starts=starts, runs=runs)
    if failed:
        f = failed[0]
        rep.add("stream_integrity", "error",
                f"Le {how} signale des erreurs sur {len(failed)}/{len(runs)} : rc={f['rc']}"
                + (f" à partir de {_fr(f['start'])} s" if f["start"] is not None else "")
                + f" ; premier message : « {f['first'] or '?'} ».",
                rc=f["rc"], first_error=f["first"], last_error=f["last"],
                failed_starts=sorted({r["start"] for r in failed if r["start"] is not None}), **nums)
    else:
        rep.add("stream_integrity", "info",
                f"Le {how} ne rapporte aucune erreur (rc=0).", **nums)


def _check_reference(a, r, rep):
    """Apply the tail rule (TAIL_RULE_S): `a` is the checked file, `r` the reference."""
    # Pair tracks by fingerprint; the shift search tolerates +/-20 s of delay.
    best = None
    for fr in sorted(set(a.fps) & set(r.fps)):
        for i, fi in a.fps[fr].items():
            for j, fj in r.fps[fr].items():
                s, sh = _fp_similarity(fi, fj, max_shift=int(20 / FP_ITEM_S), min_overlap=200)
                if s is None:
                    continue
                score = s + (0.05 if a.lang(i) == r.lang(j) else 0.0)
                if best is None or score > best[0]:
                    best = (score, s, i, j, sh, fr)
    if best is None or best[1] < FP_REFERENCE_MATCH:
        rep.add("no_common_audio_with_reference", "warning",
                "Aucune piste audio du fichier ne concorde avec une piste de la référence "
                f"(meilleure similarité {best[1]:.3f})." if best else
                "Aucune piste audio comparable avec la référence.",
                best_similarity=best[1] if best else None)
        return
    _, sim, i, j, shift, fr = best
    # ref_time = self_time + d  (fingerprint b[k+shift] matches a[k])
    d = shift * FP_ITEM_S
    ea, er = a.ext.get(i, {}), r.ext.get(j, {})
    va, vr = a.video_end(), r.video_end()
    nums = dict(track=a.label(i), reference_track=r.label(j), similarity=sim, delay_s=d,
                content_last_s=ea.get("last"), reference_content_last_s=er.get("last"),
                video_end_s=va, reference_video_end_s=vr)
    if ea.get("last") is None or er.get("last") is None:
        return
    # Audio past its own video is incoherent (reported elsewhere), not a cut.
    def past_video(x, v):
        return v is not None and any(e.get("last") is not None and e["last"] > v + AFTER_VIDEO_ERROR_S
                                     for e in x.ext.values())
    if past_video(a, va) or past_video(r, vr):
        rep.add("tail_comparison_not_applicable", "info",
                "La comparaison des fins avec la référence n'a pas de sens : une piste audio a du "
                "contenu après la fin de sa vidéo (voir content_after_video_end), le fichier est "
                "incohérent et non coupé.", **nums)
        return
    end_a, end_r = ea["last"], er["last"] - d        # both on the checked file's timeline
    diff = end_r - end_a
    nums.update(end_self_s=end_a, end_reference_on_self_timeline_s=end_r, diff_s=diff)
    if abs(diff) <= TAIL_RULE_S:
        rep.add("tail_matches_reference", "info",
                f"La fin du contenu audio concorde avec la référence (écart {_fr(diff)} s).", **nums)
        return
    short_end = min(end_a, end_r)
    start = short_end - 75.0
    wa = _extract_windows(a.path, [i], start, FP_WINDOW_S, a.workdir, "tail_a", a.threads)
    wr = _extract_windows(r.path, [j], start + d - 20.0, FP_WINDOW_S + 40.0, r.workdir,
                          "tail_r", r.threads)
    fa = _fpcalc(wa[i][0], FP_WINDOW_S) if i in wa else None
    fb = _fpcalc(wr[j][0], FP_WINDOW_S + 40) if j in wr else None
    for w in list(wa.values()) + list(wr.values()):
        try:
            os.unlink(w[0])
        except OSError:
            pass
    # fa[k] ~ fb[k + 20/FP_ITEM_S]
    s_tail, _ = _fp_similarity(fa, fb, max_shift=int(40 / FP_ITEM_S), min_overlap=200)
    nums["tail_similarity"] = s_tail
    if s_tail is None or s_tail < FP_REFERENCE_MATCH:
        rep.add("tail_comparison_inconclusive", "info",
                f"Les fins diffèrent de {_fr(abs(diff), 0)} s mais le contenu commun juste avant la "
                f"fin la plus courte n'est pas confirmé (similarité "
                f"{'?' if s_tail is None else f'{s_tail:.3f}'}).", **nums)
        return
    # A cut stops every track; if a sibling reaches the reference's end, only this track is short.
    file_last = max(e["last"] for e in a.ext.values() if e.get("last") is not None)
    nums["file_content_last_s"] = file_last
    if diff > 0 and end_r - file_last <= TAIL_RULE_S:
        rep.add("track_tail_shorter_than_reference", "warning",
                f"La piste {a.label(i)} s'arrête à {_fr(end_a)} s alors que la référence continue "
                f"jusqu'à {_fr(end_r)} s, mais d'autres pistes de ce fichier vont jusqu'à "
                f"{_fr(file_last)} s : piste abîmée, pas un fichier coupé.", **nums)
    elif diff > 0:
        rep.add("master_cut_short", "error",
                f"Le contenu audio s'arrête à {_fr(end_a)} s alors que la référence continue "
                f"{_fr(diff, 0)} s de plus, jusqu'à la fin ({_fr(end_r)} s sur l'échelle de ce "
                f"fichier) : ce fichier est coupé.", **nums)
    else:
        rep.add("candidate_tail_missing", "info",
                f"La référence s'arrête {_fr(-diff, 0)} s avant ce fichier ({_fr(end_r)} s contre "
                f"{_fr(end_a)} s) : sa queue manque, ce fichier peut la fournir.", **nums)


# ---------------------------------------------------------------- entry point

def extent_gate(a, video_end):
    """Return the reasons to decode the content extent; [] when nothing points at it.

    A reason is an audio track whose packets end more than EARLY_INFO_S before
    or AFTER_VIDEO_WARNING_S after the video end, or that is silent in every
    coherence window.
    """
    reasons = []
    for k, s in enumerate(a.audio):
        end = a.ends.get(s["index"], (None, None))[0]
        if end is None:
            end = a.declared_end(s)[0]
        if end is None or video_end is None:
            reasons.append(f"{a.label(k)}: end unmeasured")
        elif end < video_end - EARLY_INFO_S:
            reasons.append(f"{a.label(k)}: packets end {round(video_end - end, 1)} s early")
        elif end > video_end + AFTER_VIDEO_WARNING_S:
            reasons.append(f"{a.label(k)}: packets end {round(end - video_end, 1)} s late")
        windows = [w[k][1] for w in a.windows.values() if k in w]
        if windows and all(_rms_db(x) <= SILENCE_DB for x in windows):
            reasons.append(f"{a.label(k)}: silent in every coherence window")
    return reasons


def check_file(path, *, mode="sampled", reference=None, threads=3, log=None,
               windows=INTEGRITY_WINDOWS, seed=None, jobs=None, workdir=None,
               integrity_threads=INTEGRITY_THREADS, integrity=True, extent="always"):
    """Measure one media file and return a `Report`.

    Args:
        mode: 'sampled' (strict decode of `windows` seeded 2-minute windows) or
            'full' (whole-file strict decode and full packet scan).
        reference: optional other file of the same episode; adds the tail comparison.
        threads: ffmpeg threads for envelope, window and full decodes.
        windows: number of sampled integrity windows.
        seed: integrity sampling seed; None uses a stable hash of the path.
        jobs: sampled integrity processes run at once; None runs all.
        log: callable(str) or logging.Logger; None is silent.
        workdir: parent directory for temporary files (removed afterwards).
        integrity_threads: ffmpeg threads per sampled integrity process.
        integrity: False skips the strict decode (family 4).
        extent: 'always' decodes every audio envelope (family 2); 'gated' only
            when `extent_gate` finds a reason (the decode is most of the cost).

    Raises:
        ValueError: unknown mode.
    """
    if mode not in ("sampled", "full"):
        raise ValueError(f"mode must be 'sampled' or 'full', not {mode!r}")
    if log is None:
        logf = lambda m: None
    elif hasattr(log, "info"):
        logf = log.info
    else:
        logf = log
    t0 = time.time()
    rep = Report(path=path, mode=mode, reference=reference)
    tmp = tempfile.mkdtemp(prefix="conformity_", dir=workdir)
    try:
        try:
            a = _Analysis(path, mode, threads, tmp, logf, "file")
        except (RuntimeError, ValueError, OSError) as e:
            rep.add("unreadable", "error", f"ffprobe ne peut pas lire le fichier : {e}")
            return rep
        t = time.time()
        a.measure_ends()
        vend = a.video_end() or a.fmt_dur
        t_windows = time.time()
        a.measure_windows(COHERENCE_FRACTIONS, vend)
        rep.wall_seconds["coherence"] = round(time.time() - t_windows, 1)
        reasons = extent_gate(a, vend) if extent == "gated" else ["always"]
        rep.facts["extent_decoded"] = reasons
        if reasons:
            a.measure_envelopes()
            if a.env_rc != 0:
                rep.add("audio_decode_failed", "warning",
                        f"Le décodage audio (non strict) s'est terminé avec rc={a.env_rc}.",
                        rc=a.env_rc)
        rep.wall_seconds["extent"] = round(time.time() - t - rep.wall_seconds["coherence"], 1)
        _check_container(a, rep)
        _check_extent(a, rep)
        _check_coherence(a, rep)
        if integrity:
            _check_integrity(a, rep, mode, windows, seed,
                             threads if mode == "full" else integrity_threads, jobs, logf)
        if reference:
            t = time.time()
            try:
                r = _Analysis(reference, "sampled", threads, tmp, logf, "ref")
                r.measure_ends()
                r.measure_envelopes()
                r.measure_windows(COHERENCE_FRACTIONS, vend)
                _check_reference(a, r, rep)
            except (RuntimeError, ValueError, OSError) as e:
                rep.add("reference_unreadable", "warning",
                        f"La référence n'a pas pu être lue : {e}")
            rep.wall_seconds["reference"] = round(time.time() - t, 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        rep.wall_seconds["total"] = round(time.time() - t0, 1)
    logf(f"conformity: {path}: exit={rep.exit_code} errors={rep.names('error')} "
         f"warnings={rep.names('warning')} in {rep.wall_seconds['total']} s")
    return rep


def main(argv=None):
    """Run the CLI and return the report's exit code."""
    ap = argparse.ArgumentParser(
        prog="python3 -m file_conformity",
        description="Contrôle de conformité d'un fichier vidéo (conteneur, étendue du contenu "
                    "audio, cohérence entre pistes, intégrité des flux). Code de sortie : "
                    "0 conforme, 1 avertissements, 2 erreurs, 3 illisible.")
    ap.add_argument("file")
    ap.add_argument("--full", action="store_true", help="décodage strict du fichier entier")
    ap.add_argument("--json", action="store_true", help="imprimer aussi le rapport JSON")
    ap.add_argument("--reference", help="autre fichier du même épisode (comparaison des fins)")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--windows", type=int, default=INTEGRITY_WINDOWS)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="défaut : hash du chemin")
    ap.add_argument("--verbose", action="store_true")
    ns = ap.parse_args(argv)
    log = (lambda m: print(m, file=sys.stderr)) if ns.verbose else None
    rep = check_file(ns.file, mode="full" if ns.full else "sampled", reference=ns.reference,
                     threads=ns.threads, log=log, windows=ns.windows, seed=ns.seed, jobs=ns.jobs)
    print(rep.summary_fr())
    if ns.json:
        print(rep.to_json(indent=1))
    return rep.exit_code


if __name__ == "__main__":
    sys.exit(main())
