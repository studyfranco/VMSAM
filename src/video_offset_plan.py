'''video_offset_plan.py -- a master that contradicts itself is aligned by its pictures.

ADDENDUM 27 of RULING_20260922_ORCHESTRATOR_ARCHITECTURE (owner, 2026-09-25): when the
comparison-language tracks of the MASTER disagree with each other (`master_intertrack_desync`:
each couple gives a CONSTANT delay, but not the same one), the audio cannot say where the
candidate sits. The video can: both files carry the same pictures, so matched scene changes give
one offset IN FRAMES, and that offset is applied to every candidate track. This is not chimeric
work: no interior hole, no interior fill -- one constant frame offset, proved constant from the
head of the file to its tail, and additions only at the edges.

Two public entry points, both pure of any pipeline state:

    measure_video_offset(master_path, candidate_path, work_dir, log) -> VideoOffsetResult
    build_candidate_plan(result, master_obj_like, candidate_obj_like) -> dict

CONVENTION (the one every number below uses): `offset_frames = d` means candidate frame
`k + d` shows master frame `k`. A candidate track placed on the master timeline is therefore
delayed by `candidate_track_delay_ms = (master_video_start - candidate_video_start) - d x frame`
(exact rationals, never a rounded float consumed as an input). The track marker
`video_anchored:<+-N>` carries N = -d, the number of frames every candidate track is moved by.

MEASUREMENT (ADDENDUM 27.3, with ADDENDUM 26.4's "one decode per window, shared"):
  1. precondition -- both videos CFR (r_frame_rate == avg_frame_rate, exact Fraction) at the SAME
     exact rate; otherwise `fps_mismatch`, named, and nothing is decoded;
  2. each video is decoded ONCE, whole (`-threads 2`, ADDENDUM 27.8), to 128x72 BGR; that single stream feeds
     PySceneDetect's ContentDetector frame by frame AND the 64x36 grayscale pHash of every frame
     (memo MS_WALK_EXPERIMENT_20260925: decoding is the cost, so the pHash of the 10 frames either
     side of a change is read from the same decode instead of a second seek-and-decode per
     change). Results are cached in `work_dir` per file (a master reused by several candidates is
     decoded once);
  3. every change far enough from the file's ends owns a 20-frame signature (10 before, 10 after);
  4. for every master change, every candidate change inside the search bound is tried at its own
     index and +-1 frame; the best 20-frame mean Hamming distance below
     `PAIR_MEAN_HAMMING_MAX`, and clearly better than any other offset, is one vote; the offset is
     the value with the most votes (ties: the lower mean distance);
  5. constancy: the votes of each third of the master must elect the SAME offset, every third
     must hold `MIN_PAIRS_PER_THIRD` pairs at that offset, no run of consecutive pairs may agree
     on another offset, and the fitted trend of the offsets across the file must stay under one
     frame -- otherwise `video_offset_not_constant` (a real chimeric case or a drift) or
     `video_offset_coverage_incomplete`, both named with their per-third numbers.

The gate is the measurement (WRITE_ZONES section 4): nothing here is conditioned on a parameter.
'''
from collections import Counter
from fractions import Fraction
import datetime
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import tools

# --------------------------------------------------------------------------------------------
# Named constants
# --------------------------------------------------------------------------------------------

CACHE_VERSION = 2
CACHE_DIRNAME = "video_offset_cache"

# The decode: one pass per file. 128x72 BGR is what ContentDetector reads (PySceneDetect's own
# SceneManager downscales 1080p to ~274 px before its detectors; its HSV means are scale-free
# to first order), and the 64x36 grayscale the pHash reads is its 2x2 area mean.
DECODE_WIDTH = 128
DECODE_HEIGHT = 72
# ADDENDUM 27.8: the scene detection runs inside the repair's budget, bounded by `-threads 2`.
DECODE_THREADS = 2
BATCH_FRAMES = 512
CONCURRENT_DECODES = 2

# ContentDetector's own default, the same value scene_anchor.CONTENT_DETECTOR_THRESHOLD_DEFAULT
# carries (not imported: that module pulls the whole anchor machinery for one number).
SCENE_THRESHOLD = 27.0

# Frames either side of a change (owner: "10 frames de part et d'autre").
SIGNATURE_HALF = 10

# A pair of changes is the same change when the 20 frames around them differ, on average, by at
# most this many bits of a 64-bit pHash. MEASURED 2026-09-25 (host load 50-65), true pair = the
# change at its proved offset, wrong = the best partner more than 1 frame away from it:
#   Tougen S01E07 CR vs NF (independent encodes): true n=421 p50 1.0 p99 4.74 max 6.0;
#     wrong n=421 min 7.4 p5 18.2 p50 25.0 (2 wrong <= 8.0, 0 <= 6.0);
#   Kuroshitsuji S02E12 BD HEVC vs web H.264: matched n=325 p50 2.0 p95 3.38 max 6.1;
#   Ragnarok lots (6 pairs, NF re-encodes): wrong minima 4.0 / 6.2 / 13.0 / 14.5; 5 of 2 471
#     changes have a wrong partner <= 8.0 -- single stray votes, never REGIME_RUN_MIN in a row.
# 8.0 keeps every true pair measured with a 1.9-bit margin; the few wrong partners below it are
# outvoted by construction (one vote each, and the ambiguity margin below).
PAIR_MEAN_HAMMING_MAX = 8.0

# A vote is kept only when no other offset (more than 1 frame away) comes within this many bits
# of the winner: a change into black, a fade or a repeated shot matches many offsets equally.
AMBIGUITY_MARGIN_BITS = 2.0

# The search bound when no audio delay is supplied: +-SEARCH_BOUND_DEFAULT_S, widened to the
# duration difference plus SEARCH_BOUND_DURATION_MARGIN_S when the two files differ more.
# With audio delay hints: the largest |hint| plus SEARCH_BOUND_HINT_MARGIN_S.
SEARCH_BOUND_DEFAULT_S = 120
SEARCH_BOUND_DURATION_MARGIN_S = 30
SEARCH_BOUND_HINT_MARGIN_S = 30

# Constancy proof.
MIN_PAIRS_PER_THIRD = 3
REGIME_RUN_MIN = 3
TREND_MAX_FRAMES = 1.0
TREND_WINDOW_FRAMES = 2

STATUS_OK = "video_anchored"
STATUS_FPS_MISMATCH = "fps_mismatch"
STATUS_NOT_CONSTANT = "video_offset_not_constant"
STATUS_COVERAGE = "video_offset_coverage_incomplete"
STATUS_UNMATCHED = "video_offset_unmatched"
STATUS_PROBE_FAILED = "video_probe_failed"
STATUS_DECODE_FAILED = "video_decode_failed"
STATUS_DECODER_TIMEOUT = "decoder_timeout"
# ADDENDUM 27.8 / 26.3: the decode was stopped by the REPAIR's deadline, not by its own bound.
STATUS_BUDGET = "repair_budget_exceeded"

LOG_PREFIX = "video_offset: "


# --------------------------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------------------------

class VideoOffsetResult:
    '''Everything `measure_video_offset` established. `status == STATUS_OK` is the only state in
    which `offset_frames` may be consumed; every other status is a named decline.'''

    def __init__(self, status, **fields):
        self.status = status
        self.reason = fields.pop("reason", None)
        self.fps = fields.pop("fps", None)                   # Fraction, the shared exact rate
        self.offset_frames = fields.pop("offset_frames", None)
        self.master_frames = fields.pop("master_frames", None)
        self.candidate_frames = fields.pop("candidate_frames", None)
        self.master_start_s = fields.pop("master_start_s", Fraction(0))
        self.candidate_start_s = fields.pop("candidate_start_s", Fraction(0))
        self.master_scenes = fields.pop("master_scenes", None)
        self.candidate_scenes = fields.pop("candidate_scenes", None)
        self.paired = fields.pop("paired", 0)
        self.matched = fields.pop("matched", 0)
        self.ambiguous = fields.pop("ambiguous", 0)
        self.total = fields.pop("total", 0)
        self.mean_hamming = fields.pop("mean_hamming", None)
        self.covered_frames = fields.pop("covered_frames", None)   # (first, last) master frame
        self.thirds = fields.pop("thirds", None)               # [(mode, votes_at_mode, pairs_at_d)]
        self.regimes = fields.pop("regimes", None)
        self.trend_frames = fields.pop("trend_frames", None)
        self.pairs = fields.pop("pairs", None)                 # [(m_cut, d, dist)] matched
        self.wall_s = fields.pop("wall_s", None)
        self.extra = fields

    @property
    def frame_ms(self):
        return None if self.fps is None else Fraction(1000) / self.fps

    @property
    def offset_ms(self):
        '''d x frame duration, exact.'''
        if self.offset_frames is None or self.fps is None:
            return None
        return self.offset_frames * self.frame_ms

    @property
    def candidate_track_delay_ms(self):
        '''The delay every candidate track receives to sit on the master timeline, exact.'''
        if self.offset_ms is None:
            return None
        return (self.master_start_s - self.candidate_start_s) * 1000 - self.offset_ms

    def head_tail(self):
        '''(head_add, tail_add, head_trim, tail_trim) in master-grid frames.'''
        d, n_m, n_c = self.offset_frames, self.master_frames, self.candidate_frames
        if d is None or n_m is None or n_c is None:
            return None
        return (max(0, -d), max(0, n_m - (n_c - d)), max(0, d), max(0, (n_c - d) - n_m))

    def as_dict(self):
        out = {k: v for k, v in self.__dict__.items() if k not in ("pairs", "extra")}
        out.update(self.extra)
        for key in ("fps", "master_start_s", "candidate_start_s"):
            if out.get(key) is not None:
                out[key] = str(out[key])
        out["offset_ms"] = _ms_str(self.offset_ms)
        out["candidate_track_delay_ms"] = _ms_str(self.candidate_track_delay_ms)
        return out


def _ms_str(value):
    return None if value is None else f"{float(value):.6f}"


# --------------------------------------------------------------------------------------------
# Probe
# --------------------------------------------------------------------------------------------

def _ffprobe():
    return tools.software.get("ffprobe", "ffprobe")


def _ffmpeg():
    return tools.software.get("ffmpeg", "ffmpeg")


def _rate(value):
    try:
        rate = Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def probe_video(path):
    '''`(info, None)` or `(None, reason)`; info = r_rate, avg_rate (Fraction), start_s (Fraction),
    duration_s (float or None).'''
    cmd = [_ffprobe(), "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate,avg_frame_rate,start_time:format=duration",
           "-of", "json", path]
    try:
        done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    except subprocess.TimeoutExpired:
        return None, "ffprobe_timeout"
    if done.returncode != 0:
        return None, f"ffprobe_exit:{done.returncode}"
    try:
        data = json.loads(done.stdout.decode("utf-8", "replace"))
        stream = data["streams"][0]
    except (ValueError, KeyError, IndexError):
        return None, "ffprobe_no_video_stream"
    r_rate, avg_rate = _rate(stream.get("r_frame_rate")), _rate(stream.get("avg_frame_rate"))
    try:
        start = Fraction(str(stream.get("start_time", "0")))
    except (ValueError, ZeroDivisionError):
        start = Fraction(0)
    try:
        duration = float(data.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    return {"r_rate": r_rate, "avg_rate": avg_rate, "start_s": start, "duration_s": duration}, None


def check_fps(m_info, c_info):
    '''The ADDENDUM 27.1 precondition: `(Fraction, None)` or `(None, reason)`.'''
    for side, info in (("master", m_info), ("candidate", c_info)):
        if info["r_rate"] is None or info["avg_rate"] is None:
            return None, f"{side}_rate_unmeasured"
        if info["r_rate"] != info["avg_rate"]:
            return None, (f"{side}_not_cfr:r={info['r_rate']},avg={info['avg_rate']}")
    if m_info["r_rate"] != c_info["r_rate"]:
        return None, f"rates_differ:{m_info['r_rate']}!={c_info['r_rate']}"
    return m_info["r_rate"], None


# --------------------------------------------------------------------------------------------
# Decode once: scene changes + per-frame pHash
# --------------------------------------------------------------------------------------------

_BIT_WEIGHTS = (np.uint64(1) << np.arange(64, dtype=np.uint64))
_DCT_BASES = {}


def _dct_rows(n, rows=8):
    '''The first `rows` rows of the orthonormal DCT-II matrix of size n (what
    `scipy.fft.dctn(norm="ortho")` applies), cached: the pHash only reads the top-left 8x8, so two
    small products replace a full 2-D transform (equal to it within 1e-11, measured).'''
    if (n, rows) not in _DCT_BASES:
        k = np.arange(rows)[:, None]
        i = np.arange(n)[None, :]
        basis = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
        basis[0] /= np.sqrt(2.0)
        _DCT_BASES[(n, rows)] = basis.astype(np.float64)
    return _DCT_BASES[(n, rows)]


def phash64(gray_frames):
    '''64-bit DCT pHash of each frame of a `(k, h, w)` array: 2-D DCT-II (orthonormal), top-left
    8x8, bits set where the coefficient exceeds the median of the 63 non-DC ones.'''
    frames = np.asarray(gray_frames, dtype=np.float64)
    _, h, w = frames.shape
    coeffs = np.einsum("ai,kij,bj->kab", _dct_rows(h), frames, _dct_rows(w),
                       optimize=True).reshape(len(frames), 64)
    median = np.median(coeffs[:, 1:], axis=1, keepdims=True)
    bits = (coeffs > median).astype(np.uint64)
    return (bits * _BIT_WEIGHTS).sum(axis=1, dtype=np.uint64)


def _gray_64x36(bgr):
    '''(k, 72, 128, 3) uint8 BGR -> (k, 36, 64) uint8 luma (BT.601), 2x2 area mean -- OpenCV
    per frame (PySceneDetect's own dependency): measured 35x faster than the numpy float path.'''
    import cv2
    return np.stack([cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                (DECODE_WIDTH // 2, DECODE_HEIGHT // 2),
                                interpolation=cv2.INTER_AREA) for frame in bgr])


def _cache_path(work_dir, path):
    st = os.stat(path)
    key = json.dumps([os.path.realpath(path), st.st_size, st.st_mtime_ns, CACHE_VERSION,
                      SCENE_THRESHOLD, DECODE_WIDTH, DECODE_HEIGHT])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    return os.path.join(work_dir, CACHE_DIRNAME, f"{digest}.npz")


class BudgetExceeded(Exception):
    '''The decode was stopped because the repair's deadline (a `time.monotonic()` instant) came
    first -- a statement about this run's cost, never about the media.'''


def decode_scenes_and_hashes(path, fps, duration_s, work_dir, deadline=None):
    '''`(cuts, hashes, from_cache)` for the whole video: `cuts` = ContentDetector scene starts
    (absolute frame indices of the decoded stream), `hashes` = uint64 pHash per decoded frame.
    Raises `tools.decoder_timeout` past the ADDENDUM 26.3 bound, `BudgetExceeded` when the
    repair's `deadline` comes first, RuntimeError on a failed decode.
    '''
    cache = _cache_path(work_dir, path) if work_dir else None
    if cache and os.path.exists(cache):
        with np.load(cache) as data:
            return [int(x) for x in data["cuts"]], data["hashes"].copy(), True

    from scenedetect import ContentDetector, FrameTimecode  # heavy import, only when decoding

    frame_bytes = DECODE_WIDTH * DECODE_HEIGHT * 3
    cmd = [_ffmpeg(), "-v", "error", "-nostdin", "-threads", str(DECODE_THREADS), "-i", path,
           "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough",
           "-vf", f"scale={DECODE_WIDTH}:{DECODE_HEIGHT}:flags=area", "-pix_fmt", "bgr24",
           "-f", "rawvideo", "pipe:1"]
    timeout = tools.decoder_timeout_for(duration_s or 0.0)
    budget_bound = False
    if deadline is not None:
        left = deadline - time.monotonic()
        if left <= 0:
            raise BudgetExceeded(f"no budget left before the decode of {path}")
        if left < timeout:
            timeout, budget_bound = left, True
    tools.dev_log(f"{LOG_PREFIX}decode starting file={path} timeout_s={round(timeout, 1)} "
                  f"bound_by={'repair_budget' if budget_bound else 'decoder_bound'}\n")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    fired = []

    def _kill():
        fired.append(True)
        try:
            proc.kill()
        except OSError:
            pass

    timer = threading.Timer(timeout, _kill)
    timer.start()
    stderr_chunks = []
    drain = threading.Thread(target=lambda: stderr_chunks.append(proc.stderr.read()), daemon=True)
    drain.start()
    detector = ContentDetector(threshold=SCENE_THRESHOLD)
    cuts, hash_parts, index = [], [], 0
    try:
        while True:
            blob = proc.stdout.read(frame_bytes * BATCH_FRAMES)
            if not blob:
                break
            usable = len(blob) - len(blob) % frame_bytes
            frames = np.frombuffer(blob[:usable], dtype=np.uint8).reshape(
                -1, DECODE_HEIGHT, DECODE_WIDTH, 3)
            for frame in frames:
                for tc in detector.process_frame(FrameTimecode(index, fps=fps), frame):
                    cuts.append(int(tc.frame_num))
                index += 1
            hash_parts.append(phash64(_gray_64x36(frames)))
            if usable != len(blob):
                break
        if index:
            for tc in detector.post_process(FrameTimecode(index - 1, fps=fps)):
                cuts.append(int(tc.frame_num))
        proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    drain.join(timeout=5)
    if fired and budget_bound:
        raise BudgetExceeded(f"the repair's deadline stopped the scene decode of {path} "
                             f"after {round(timeout, 1)} s")
    if fired:
        raise tools.decoder_timeout("ffmpeg_scene_decode", timeout, f"file={path}")
    if proc.returncode != 0 or index == 0:
        err = b"".join(c for c in stderr_chunks if c).decode("utf-8", "replace")[-300:]
        raise RuntimeError(f"decode_exit:{proc.returncode} frames={index} {err.strip()}")
    hashes = np.concatenate(hash_parts) if hash_parts else np.zeros(0, dtype=np.uint64)
    cuts = sorted(set(c for c in cuts if 0 < c < len(hashes)))
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = cache + ".tmp.npz"
        np.savez(tmp, cuts=np.asarray(cuts, dtype=np.int64), hashes=hashes)
        os.replace(tmp, cache)
    return cuts, hashes, False


# --------------------------------------------------------------------------------------------
# Matching and constancy -- pure functions over scene lists and hash arrays (unit-testable)
# --------------------------------------------------------------------------------------------

def _signature_ok(cut, n):
    return cut - SIGNATURE_HALF >= 0 and cut + SIGNATURE_HALF <= n


def match_changes(m_hashes, m_cuts, c_hashes, c_cuts, lo, hi):
    '''For each master change, the best candidate offset `d` in [lo, hi] (tried at every candidate
    change's own index and +-1 frame). Returns `(matched, ambiguous, total)`: `matched` is a list
    of `(m_cut, d, mean_hamming)` sorted by m_cut, one per master change that found a clear partner
    below `PAIR_MEAN_HAMMING_MAX`; `ambiguous` counts the changes whose partner was not distinct.
    '''
    m_hashes = np.asarray(m_hashes, dtype=np.uint64)
    c_hashes = np.asarray(c_hashes, dtype=np.uint64)
    n_m, n_c = len(m_hashes), len(c_hashes)
    c_cuts = np.asarray(sorted(c for c in c_cuts if _signature_ok(c, n_c)), dtype=np.int64)
    span = np.arange(-SIGNATURE_HALF, SIGNATURE_HALF, dtype=np.int64)
    matched, ambiguous, total = [], 0, 0
    for m in sorted(m_cuts):
        if not _signature_ok(m, n_m):
            continue
        total += 1
        left = np.searchsorted(c_cuts, m + lo - 1, side="left")
        right = np.searchsorted(c_cuts, m + hi + 1, side="right")
        near = c_cuts[left:right]
        if len(near) == 0:
            continue
        ds = np.unique(np.concatenate([near - m - 1, near - m, near - m + 1]))
        ds = ds[(ds >= lo) & (ds <= hi)]
        ds = ds[(m + ds - SIGNATURE_HALF >= 0) & (m + ds + SIGNATURE_HALF <= n_c)]
        if len(ds) == 0:
            continue
        sig = m_hashes[m + span]
        cand = c_hashes[(m + ds)[:, None] + span[None, :]]
        dist = np.bitwise_count(np.bitwise_xor(cand, sig[None, :])).mean(axis=1)
        order = np.argsort(dist, kind="stable")
        best = order[0]
        if dist[best] > PAIR_MEAN_HAMMING_MAX:
            continue
        rivals = [i for i in order[1:] if abs(int(ds[i]) - int(ds[best])) > 1]
        if rivals and dist[rivals[0]] <= dist[best] + AMBIGUITY_MARGIN_BITS:
            ambiguous += 1
            continue
        matched.append((int(m), int(ds[best]), float(dist[best])))
    return matched, ambiguous, total


def _mode(values_dists):
    '''(value, count) of the most frequent value; ties -> lower mean distance.'''
    counts = Counter(v for v, _ in values_dists)
    if not counts:
        return None, 0
    sums = Counter()
    for v, dist in values_dists:
        sums[v] += dist
    best = min(counts, key=lambda v: (-counts[v], sums[v] / counts[v]))
    return best, counts[best]


def prove_constant(matched, n_master):
    '''The ADDENDUM 27.3 proof over the matched pairs. Returns a dict: status (STATUS_OK /
    STATUS_NOT_CONSTANT / STATUS_COVERAGE / STATUS_UNMATCHED), d, paired, mean_hamming, thirds,
    regimes, trend_frames, covered.'''
    out = {"d": None, "paired": 0, "mean_hamming": None, "thirds": [], "regimes": [],
           "trend_frames": None, "covered": None}
    d, votes = _mode([(dd, dist) for _, dd, dist in matched])
    if d is None or votes < MIN_PAIRS_PER_THIRD:
        out["status"] = STATUS_UNMATCHED
        out["d"] = d
        out["paired"] = votes
        return out
    agreeing = [(m, dist) for m, dd, dist in matched if dd == d]
    out["d"], out["paired"] = d, len(agreeing)
    out["mean_hamming"] = sum(dist for _, dist in agreeing) / len(agreeing)
    out["covered"] = (agreeing[0][0], agreeing[-1][0])
    bounds = [0, n_master // 3, (2 * n_master) // 3, n_master]
    for i in range(3):
        inside = [(dd, dist) for m, dd, dist in matched if bounds[i] <= m < bounds[i + 1]]
        mode, count = _mode(inside)
        at_d = sum(1 for dd, _ in inside if dd == d)
        out["thirds"].append((mode, count, at_d))
    # runs of consecutive matched pairs agreeing on another offset
    run_d, run = None, []
    for m, dd, _ in matched + [(None, None, None)]:
        if dd is not None and dd == run_d:
            run.append(m)
            continue
        if run_d is not None and run_d != d and len(run) >= REGIME_RUN_MIN:
            out["regimes"].append((run_d, run[0], run[-1], len(run)))
        run_d, run = dd, [m]
    # trend of the offsets across the file (pairs within TREND_WINDOW_FRAMES of d)
    near = [(m, dd) for m, dd, _ in matched if abs(dd - d) <= TREND_WINDOW_FRAMES]
    if len(near) >= 2 and len({m for m, _ in near}) >= 2:
        xs = np.array([m for m, _ in near], dtype=np.float64)
        ys = np.array([dd for _, dd in near], dtype=np.float64)
        slope = np.polyfit(xs, ys, 1)[0]
        out["trend_frames"] = float(slope * n_master)
    else:
        out["trend_frames"] = 0.0
    if (any(mode is not None and mode != d for mode, _, _ in out["thirds"])
            or out["regimes"] or abs(out["trend_frames"]) >= TREND_MAX_FRAMES):
        out["status"] = STATUS_NOT_CONSTANT
    elif any(at_d < MIN_PAIRS_PER_THIRD for _, _, at_d in out["thirds"]):
        out["status"] = STATUS_COVERAGE
    else:
        out["status"] = STATUS_OK
    return out


def search_bound_frames(fps, m_duration_s, c_duration_s, audio_delay_hints_ms=None):
    '''(lo, hi) in frames. Hints (ms, any sign) bound it to max|hint| + margin; otherwise the
    default bound, widened to the duration difference plus a margin.'''
    if audio_delay_hints_ms:
        seconds = max(abs(float(h)) for h in audio_delay_hints_ms) / 1000.0 \
            + SEARCH_BOUND_HINT_MARGIN_S
    else:
        seconds = SEARCH_BOUND_DEFAULT_S
        if m_duration_s is not None and c_duration_s is not None:
            seconds = max(seconds, abs(m_duration_s - c_duration_s)
                          + SEARCH_BOUND_DURATION_MARGIN_S)
    frames = int(Fraction(seconds).limit_denominator(1000) * fps) + 1
    return -frames, frames


# --------------------------------------------------------------------------------------------
# Entry point 1: the measurement
# --------------------------------------------------------------------------------------------

def _emit(log, line):
    now = datetime.datetime.now(datetime.timezone.utc)
    stamped = (f"{LOG_PREFIX}utc={now.strftime('%Y-%m-%dT%H:%M:%S')}."
               f"{now.microsecond // 1000:03d}Z {line}\n")
    if callable(log):
        log(stamped)
    else:
        tools.log_always(stamped)


def _log_line(result, bound):
    fps = result.fps
    parts = [f"status={result.status}"]
    if result.reason:
        parts.append(f"reason={result.reason}")
    parts.append(f"fps={fps}" if fps is not None else "fps=?")
    if result.master_scenes is not None:
        parts.append(f"scenes m={result.master_scenes} c={result.candidate_scenes}")
        parts.append(f"paired={result.paired}/{result.total} matched={result.matched} "
                     f"ambiguous={result.ambiguous}")
    if result.offset_frames is not None:
        parts.append(f"d={result.offset_frames:+d}fr ({_ms_str(result.offset_ms)} ms) "
                     f"track_delay_ms={_ms_str(result.candidate_track_delay_ms)}")
    if result.mean_hamming is not None:
        parts.append(f"mean_hamming={result.mean_hamming:.2f}/{PAIR_MEAN_HAMMING_MAX}")
    if result.covered_frames is not None and fps is not None:
        a, b = result.covered_frames
        parts.append(f"covered=[{float(a / fps):.1f},{float(b / fps):.1f}]s "
                     f"of {float(result.master_frames / fps):.1f}s")
    if result.thirds:
        parts.append("thirds=" + ",".join(
            f"{'?' if mode is None else f'{mode:+d}'}:{at}" for mode, _, at in result.thirds))
    if result.regimes:
        parts.append("regimes=" + ",".join(f"{dd:+d}@[{a},{b}]x{n}"
                                           for dd, a, b, n in result.regimes))
    if result.trend_frames is not None:
        parts.append(f"trend={result.trend_frames:+.2f}fr")
    ht = result.head_tail() if result.status == STATUS_OK else None
    if ht:
        parts.append(f"plan head_add={ht[0]}fr tail_add={ht[1]}fr head_trim={ht[2]}fr "
                     f"tail_trim={ht[3]}fr")
    if bound:
        parts.append(f"bound=[{bound[0]},{bound[1]}]fr")
    if result.wall_s is not None:
        parts.append(f"wall={result.wall_s:.1f}s")
    return " ".join(parts)


def measure_video_offset(master_path, candidate_path, work_dir, log=None,
                         audio_delay_hints_ms=None, deadline=None):
    '''The ADDENDUM 27 measurement. Never raises for a media reason: every failure is a named
    status on the returned `VideoOffsetResult`, and exactly one `log_always` line says it.
    `deadline` (ADDENDUM 27.8): the repair's `time.monotonic()` instant; a decode it stops
    returns `STATUS_BUDGET`.'''
    t0 = time.monotonic()
    bound = None

    def done(result):
        result.wall_s = time.monotonic() - t0
        _emit(log, _log_line(result, bound))
        return result

    m_info, why = probe_video(master_path)
    if m_info is None:
        return done(VideoOffsetResult(STATUS_PROBE_FAILED, reason=f"master:{why}"))
    c_info, why = probe_video(candidate_path)
    if c_info is None:
        return done(VideoOffsetResult(STATUS_PROBE_FAILED, reason=f"candidate:{why}"))
    fps, why = check_fps(m_info, c_info)
    if fps is None:
        return done(VideoOffsetResult(STATUS_FPS_MISMATCH, reason=why))

    common = {"fps": fps, "master_start_s": m_info["start_s"],
              "candidate_start_s": c_info["start_s"]}
    # The two decodes run side by side (each `-threads 2`): decoding is the whole cost (memo
    # MS_WALK_EXPERIMENT_20260925) and the two files are independent -- measured on Ragnarok
    # S02E14 at host load 55-65: 283 s one after the other.
    with ThreadPoolExecutor(max_workers=CONCURRENT_DECODES) as pool:
        futures = [pool.submit(decode_scenes_and_hashes, path, fps, info["duration_s"], work_dir,
                               deadline)
                   for path, info in ((master_path, m_info), (candidate_path, c_info))]
        errors, outcomes = [], []
        for future in futures:
            try:
                outcomes.append(future.result())
            except BudgetExceeded as exc:
                errors.append((STATUS_BUDGET, str(exc)))
            except tools.decoder_timeout as exc:
                errors.append((STATUS_DECODER_TIMEOUT, str(exc)))
            except (RuntimeError, OSError) as exc:
                errors.append((STATUS_DECODE_FAILED, str(exc)[:200]))
    if errors:
        errors.sort(key=lambda error: error[0] != STATUS_BUDGET)
        return done(VideoOffsetResult(errors[0][0], reason=errors[0][1], **common))
    (m_cuts, m_hashes, m_cached), (c_cuts, c_hashes, c_cached) = outcomes

    common.update(master_frames=len(m_hashes), candidate_frames=len(c_hashes),
                  master_scenes=len(m_cuts), candidate_scenes=len(c_cuts),
                  master_cached=m_cached, candidate_cached=c_cached)
    bound = search_bound_frames(fps, m_info["duration_s"], c_info["duration_s"],
                                audio_delay_hints_ms)
    matched, ambiguous, total = match_changes(m_hashes, m_cuts, c_hashes, c_cuts, *bound)
    proof = prove_constant(matched, len(m_hashes))
    status = proof["status"]
    result = VideoOffsetResult(
        status, offset_frames=proof["d"] if status == STATUS_OK else None,
        paired=proof["paired"], matched=len(matched), ambiguous=ambiguous, total=total,
        mean_hamming=proof["mean_hamming"], covered_frames=proof["covered"],
        thirds=proof["thirds"], regimes=proof["regimes"], trend_frames=proof["trend_frames"],
        pairs=matched, best_d=proof["d"], **common)
    return done(result)


# --------------------------------------------------------------------------------------------
# Entry point 2: the plan
# --------------------------------------------------------------------------------------------

def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _tracks(obj, kind):
    by_lang = _get(obj, kind) or {}
    for lang, tracks in by_lang.items():
        for track in tracks or []:
            yield lang, track


def build_candidate_plan(result, master_obj_like, candidate_obj_like, comparison_language=None):
    '''The ADDENDUM 27.4 plan. Master tracks untouched; EVERY candidate track (audio,
    commentary, audio description, subtitles) and the candidate's chapters shifted by the one
    video offset; additions only at the head and tail, where the candidate is shorter than the
    master, filled from the master's comparison-language audio when it has one, else silence
    (owner: those zones are noise or silence, language does not matter).

    `*_obj_like`: a `video.video` or any object/dict carrying `filePath` and the `audios` /
    `commentary` / `audiodesc` / `subtitles` dicts (language -> list of track dicts).
    '''
    if result is None or result.status != STATUS_OK:
        return {"status": "declined",
                "cause": None if result is None else result.status,
                "reason": None if result is None else result.reason}
    shift = -result.offset_frames
    delay = result.candidate_track_delay_ms
    marker = f"video_anchored:{shift:+d}"
    frame_ms = result.frame_ms
    head_add, tail_add, head_trim, tail_trim = result.head_tail()
    lang = comparison_language or tools.special_params.get("original_language")
    master_audios = _get(master_obj_like, "audios") or {}
    if lang and master_audios.get(lang):
        track = master_audios[lang][0]
        fill = {"source": "master_audio", "language": lang,
                "stream": track.get("StreamOrder") if isinstance(track, dict) else None}
    else:
        fill = {"source": "silence", "language": None, "stream": None}

    tracks = []
    for kind in ("audios", "commentary", "audiodesc", "subtitles"):
        for tlang, track in _tracks(candidate_obj_like, kind):
            tracks.append({"kind": kind, "language": tlang,
                           "stream": track.get("StreamOrder") if isinstance(track, dict) else None,
                           "shift_frames": shift, "delay_ms": _ms_str(delay),
                           "marker": marker})

    def edge(frames, trim):
        return {"add_frames": frames, "add_ms": _ms_str(frames * frame_ms),
                "fill": dict(fill) if frames else None,
                "trim_frames": trim, "trim_ms": _ms_str(trim * frame_ms)}

    return {
        "status": STATUS_OK,
        "marker": marker,
        "master": {"path": _get(master_obj_like, "filePath"), "tracks": "untouched"},
        "candidate": {"path": _get(candidate_obj_like, "filePath"),
                      "offset_frames": result.offset_frames,
                      "shift_frames": shift,
                      "delay_ms": _ms_str(delay),
                      "delay_ms_exact": str(delay),
                      "tracks": tracks,
                      "chapters": {"shift_frames": shift, "delay_ms": _ms_str(delay)}},
        "head": edge(head_add, head_trim),
        "tail": edge(tail_add, tail_trim),
        "interior": "none",
        "fps": str(result.fps),
    }


# --------------------------------------------------------------------------------------------
# Entry point 3 (ADDENDUM 27.8): the detection and the route, called by `repair_orchestrator`
# --------------------------------------------------------------------------------------------

class _Orchestrator:
    """`repair_orchestrator`, imported on first use: it imports this module's route at call
    time, and this module must stay importable (and testable) without it."""

    def __getattr__(self, name):
        import repair_orchestrator
        return getattr(repair_orchestrator, name)


orch = _Orchestrator()

# Owner, 2026-09-25 14:1x: "Comme nous ne savons pas qui est le bon, je pense qu'il faut fallback
# sur la comparaison video." Every audio contradiction leads to ONE fallback, the video:
#   (i)   the master's comparison-language tracks disagree among themselves
#         (`master_intertrack_desync`: master_self_check at STEP 1, or two couples sharing one
#         candidate track more than one frame apart);
#   (ii)  the candidate's own comparison-language tracks disagree among themselves (two couples
#         sharing one master track more than one frame apart -- Erai: AAC 0, E-AC-3 -84.4 ms);
#   (iii) the audio delay and the picture offset differ by more than one frame
#         (`audio_video_offset_disagree`, frame_snap).
# Nobody knows which audio is right, so NO audio is elected and disagreeing couples are NEVER
# averaged: the video says whether these are the same files and whether the delay is constant,
# and the candidate's tracks all follow its picture (27.4 under the general law -- the master is
# never modified; each candidate track keeps its own relation to its own picture).

TRIGGER_MASTER_DESYNC = "master_intertrack_desync"          # (i)
TRIGGER_CANDIDATE_DESYNC = "candidate_intertrack_desync"    # (ii)
TRIGGER_PICTURE_DISAGREE = "audio_video_offset_disagree"    # (iii)
VIDEO_ANCHORED_KIND = "orchestrator_video_anchored"
# The per-couple delay instrument is master_self_check's own (the "same measurement on the
# candidate side" the ruling asks for): a 30 s 16 kHz mono window, FFT cross-correlation with a
# parabolic peak, +-1 s around the couple's coarse offset, a couple read only above its
# correlation floor. ONE instant for every couple (the middle of the master video), so a drift
# or an edit elsewhere in the file moves every couple alike and cannot open a gap between them.
COUPLE_DELAY_WINDOW_S = 30.0
COUPLE_DELAY_SEARCH_S = 1.0
COUPLE_DELAY_POSITION = 0.5


def _video_frame_ms(video_obj):
    """One frame of this video in ms (the exact rational rate when readable, else the decimal
    `FrameRate`), or None."""
    try:
        import frame_snap
        rate, _ = frame_snap.exact_rate(video_obj.video)
        if rate is not None:
            return Fraction(1000) / Fraction(rate)
        return Fraction(1000) / Fraction(str(video_obj.video["FrameRate"]))
    except Exception:                                                    # noqa: BLE001
        return None


def coarse_offsets_from_prime(primed, master_obj, candidate_obj, language, at_s):
    """Each measured couple's offset ON THE FILE'S CLOCK (container delays folded,
    `couple_start_delta_ms`) at master instant `at_s`, from the zone of its alignment that holds
    that instant -- `{couple: ms}`; a couple blind, under the coverage floor, or with no zone at
    that instant is absent (it is not a reading)."""
    out = {}
    for master_stream, candidate_stream in primed["couples"]:
        name = f"{master_stream}x{candidate_stream}"
        alignment = primed["alignments"].get(name) or {}
        if alignment.get("verdict") in orch.ALIGNMENT_COULD_NOT_MEASURE_VERDICTS:
            continue
        coverage = alignment.get("master_axis_coverage_fraction")
        if coverage is None or coverage < orch.MASTER_AXIS_COVERAGE_FLOOR:
            continue
        delta_ms, master_start_ms, _ = orch.couple_start_delta_ms(
            master_obj, candidate_obj, language, master_stream, candidate_stream, 1)
        track_ms = at_s * 1000.0 - float(master_start_ms)
        zone = next((z for z in alignment.get("zones_detail") or []
                     if z["master_ms"][0] <= track_ms < z["master_ms"][1]), None)
        if zone is not None:
            out[name] = float(zone["offset_ms"]) + float(delta_ms)
    return out


def couple_fine_delays(master_obj, candidate_obj, couples, coarse_ms, at_s):
    """EVERY couple's own audio delay, at the millisecond, on the file's clock: candidate file
    time = master file time + delay. One row per couple, measured or not (`reason` says why
    not); never a mean."""
    import master_self_check
    rows = []
    for master_stream, candidate_stream in couples:
        name = f"{master_stream}x{candidate_stream}"
        coarse = coarse_ms.get(name)
        row = {"couple": name, "master_stream": master_stream,
               "candidate_stream": candidate_stream,
               "coarse_ms": None if coarse is None else round(float(coarse), 3),
               "delay_ms": None, "correlation": None, "at_s": round(at_s, 3), "reason": None}
        rows.append(row)
        if coarse is None:
            row["reason"] = "no_coarse_offset_at_this_instant"
            continue
        master_at = round(at_s, 3)
        candidate_at = round(at_s + float(coarse) / 1000.0, 3)
        if candidate_at < 0:
            row["reason"] = "window_before_candidate_zero"
            continue
        candidate_pcm = master_self_check._pcm(candidate_obj.filePath, candidate_stream,
                                               candidate_at, COUPLE_DELAY_WINDOW_S)
        master_pcm = master_self_check._pcm(master_obj.filePath, master_stream, master_at,
                                            COUPLE_DELAY_WINDOW_S)
        if candidate_pcm is None or master_pcm is None:
            row["reason"] = "extraction_failed"
            continue
        lag, correlation = master_self_check._fft_cross_correlation(
            candidate_pcm, master_pcm, int(COUPLE_DELAY_SEARCH_S * master_self_check.SR))
        if lag is None:
            row["reason"] = "window_without_energy"
            continue
        row["correlation"] = round(correlation, 4)
        if correlation <= master_self_check.MIN_CORRELATION_FOR_VERDICT:
            row["reason"] = (f"correlation_at_or_below_"
                             f"{master_self_check.MIN_CORRELATION_FOR_VERDICT}")
            continue
        row["delay_ms"] = round((candidate_at - master_at) * 1000.0
                                + lag * 1000.0 / master_self_check.SR, 3)
    return rows


def _rows_text(rows):
    return "[" + ",".join(
        f"{r['couple']}:{r['delay_ms'] if r['delay_ms'] is not None else 'unmeasured'}"
        f"@{r['correlation']}" + (f"({r['reason']})" if r["reason"] else "")
        for r in rows) + "]"


def contradiction_among_couples(rows, frame_ms):
    """(ii) and (i)-by-couples, from the rows alone: couples sharing ONE master track more than
    one frame apart are the CANDIDATE contradicting itself; couples sharing ONE candidate track
    more than one frame apart are the MASTER contradicting itself. Returns `(trigger, evidence)`
    or `(None, None)`."""
    if frame_ms is None:
        return None, None
    measured = [r for r in rows if r["delay_ms"] is not None]
    for trigger, key in ((TRIGGER_CANDIDATE_DESYNC, "master_stream"),
                         (TRIGGER_MASTER_DESYNC, "candidate_stream")):
        groups = {}
        for row in measured:
            groups.setdefault(row[key], []).append(row)
        for shared, group in groups.items():
            values = [r["delay_ms"] for r in group]
            if len(values) > 1 and max(values) - min(values) > float(frame_ms):
                return trigger, {"shared_" + key: shared,
                                 "delays_ms": {r["couple"]: r["delay_ms"] for r in group},
                                 "spread_ms": round(max(values) - min(values), 3),
                                 "frame_ms": round(float(frame_ms), 3), "source": "couples"}
    return None, None


def detect_audio_contradiction(master_obj, candidate_obj, language, primed, candidate_path):
    """After the prime: (ii), (i) by couples, then (iii). Returns `(trigger, evidence, rows)`;
    `trigger` None means the audios tell one story and the ordinary path continues unchanged."""
    import frame_snap
    video_ms = orch._video_duration_ms(master_obj)
    if video_ms is None:
        orch.step_result("audio_contradiction", candidate=candidate_path, measured=False,
                    reason="master_video_duration_unread")
        return None, None, []
    at_s = float(video_ms) / 1000.0 * COUPLE_DELAY_POSITION
    coarse = coarse_offsets_from_prime(primed, master_obj, candidate_obj, language, at_s)
    if not coarse:
        orch.step_result("audio_contradiction", candidate=candidate_path, measured=False,
                    reason="no_couple_holds_the_instant", at_s=round(at_s, 3))
        return None, None, []
    orch.step_launch("audio_contradiction", candidate=candidate_path, n_couples=len(coarse))
    rows = couple_fine_delays(master_obj, candidate_obj, primed["couples"], coarse, at_s)
    frame_ms = _video_frame_ms(master_obj)
    trigger, evidence = contradiction_among_couples(rows, frame_ms)
    if trigger is None:
        measured = [r for r in rows if r["delay_ms"] is not None]
        if measured:
            signal = (frame_snap.disagreement(master_obj.filePath, candidate_obj.filePath)
                      or frame_snap.disagreement(candidate_obj.filePath, master_obj.filePath))
            probe_reason = "recorded_by_adjust_delay_to_frame" if signal else None
            if signal is None:
                signal, probe_reason = frame_snap.probe_disagreement(
                    master_obj, candidate_obj, measured[0]["delay_ms"])
            if signal is not None:
                trigger, evidence = TRIGGER_PICTURE_DISAGREE, dict(signal, probe=probe_reason)
            else:
                evidence = {"picture_probe": probe_reason}
    tools.log_always(f"repair: audio_contradiction trigger={trigger} language={language} "
                     f"couples={_rows_text(rows)} frame_ms="
                     f"{None if frame_ms is None else round(float(frame_ms), 3)} "
                     f"evidence={evidence} for {candidate_path}\n")
    orch.step_result("audio_contradiction", candidate=candidate_path, trigger=trigger,
                n_measured=sum(1 for r in rows if r["delay_ms"] is not None))
    return trigger, evidence, rows


# What the video's statuses become (ADDENDUM 27.8 point 2). `fallback` exists only for a
# contradiction the audio couples may still carry (ii / iii): an offset that changes along the
# file is a drift or an interior edit, "voie chimerique ordinaire si les couples audio le
# permettent" -- for (i) the couples themselves disagree, so the same statuses decline.
def video_status_cause(status, trigger):
    vop = sys.modules[__name__]
    if status in (vop.STATUS_NOT_CONSTANT, vop.STATUS_COVERAGE):
        return ("declined" if trigger == TRIGGER_MASTER_DESYNC else "fallback"), status
    return "declined", {
        vop.STATUS_FPS_MISMATCH: "video_fps_mismatch",
        vop.STATUS_UNMATCHED: "video_content_mismatch",
        vop.STATUS_PROBE_FAILED: "video_probe_failed",
        vop.STATUS_DECODE_FAILED: "video_decode_failed",
        vop.STATUS_DECODER_TIMEOUT: "decoder_timeout",
        vop.STATUS_BUDGET: "repair_budget_exceeded",
    }.get(status, "video_decode_failed")


def video_anchored_pieces(offset_ms, extent_ms, timeline_ms):
    """ONE candidate track on the master timeline at the picture's offset (27.4): one zone
    `[0, timeline)` read at `master + offset_ms` (file clock), the head filled where the track
    would read before the candidate's zero, the tail filled past the track's own end -- the
    only additions. `track_pieces` does exactly that for one zone and no fill."""
    zone = {"master_start_ms": Decimal(0), "master_end_ms": timeline_ms,
            "offset_ms": offset_ms, "n_windows": 0, "zone": 0}
    return orch.track_pieces([zone], [], [{"zone": 0, "offset_ms": offset_ms}], extent_ms,
                        timeline_ms)


def video_anchored_route(trigger, evidence, master_obj, candidate_obj, language, rows,
                         repair_deadline):
    """THE VIDEO-ANCHORED ROUTE (ADDENDUM 27.8 point 3). Returns `(status, cause, reason)`,
    status in repaired / declined / fallback. The master is never touched; every candidate
    track (audio, commentary, audio description, subtitles, chapters) is moved by the picture
    offset, each keeping its own relation to its own picture; additions at the head and tail
    only, from the master or silence, with no language requirement."""
    import merge_video_chimeric
    import merge_video_repair
    video_offset_plan = sys.modules[__name__]
    candidate_path = candidate_obj.filePath
    started = time.time()
    orch.step_launch("video_anchored", candidate=candidate_path, trigger=trigger)
    hints = [r["delay_ms"] for r in rows if r["delay_ms"] is not None] or None
    cache_root = os.path.join(tools.tmpFolder, "repair")
    tools.make_dirs(cache_root)
    result = video_offset_plan.measure_video_offset(
        master_obj.filePath, candidate_path, cache_root, audio_delay_hints_ms=hints,
        deadline=repair_deadline)
    numbers = (f"fps={result.fps} scenes master={result.master_scenes} "
               f"candidate={result.candidate_scenes} paired={result.paired}/{result.total} "
               f"matched={result.matched} ambiguous={result.ambiguous} "
               f"mean_hamming={None if result.mean_hamming is None else round(result.mean_hamming, 2)} "
               f"covered_frames={result.covered_frames} thirds={result.thirds} "
               f"regimes={result.regimes} trend_frames="
               f"{None if result.trend_frames is None else round(result.trend_frames, 3)} "
               f"best_d={result.extra.get('best_d')} wall_s="
               f"{None if result.wall_s is None else round(result.wall_s, 1)}")
    if result.status != video_offset_plan.STATUS_OK:
        status, cause = video_status_cause(result.status, trigger)
        reason = (f"the audios contradict each other ({trigger}: {evidence}; per couple "
                  f"{_rows_text(rows)}) and the video could not arbitrate: {result.status}"
                  f"({result.reason}) -- {numbers}")
        tools.log_always(f"repair: video_anchored_route trigger={trigger} status={status} "
                         f"video={result.status} cause={cause} couples={_rows_text(rows)} "
                         f"{numbers} for {candidate_path}\n")
        orch.step_result("video_anchored", candidate=candidate_path, status=status, cause=cause)
        return status, cause, reason

    frame_ms = result.frame_ms
    picture_ms = -result.candidate_track_delay_ms           # candidate file = master file + this
    if not any(r["delay_ms"] is not None for r in rows):
        # (i) from STEP 1 has no prime: every couple is read here, around the picture's delay.
        couples = orch.enumerate_couples(master_obj, candidate_obj, language)
        at_s = float(result.master_frames / result.fps) * COUPLE_DELAY_POSITION
        rows = couple_fine_delays(master_obj, candidate_obj, couples,
                                  {f"{m}x{c}": float(picture_ms) for m, c in couples}, at_s)
    gaps = {r["couple"]: round(r["delay_ms"] - float(picture_ms), 3)
            for r in rows if r["delay_ms"] is not None}
    for row in rows:
        if row["delay_ms"] is not None and abs(gaps[row["couple"]]) > float(frame_ms):
            # the analysis signal of 27.7/27.8 (iii), wherever it fires -- never a decision here
            tools.log_always(
                f"repair: audio_video_offset_disagree couple={row['couple']} "
                f"audio_ms={row['delay_ms']} audio_frames="
                f"{round(row['delay_ms'] / float(frame_ms), 2)} "
                f"picture_frames={result.offset_frames:+d} picture_ms={float(picture_ms):.3f} "
                f"gap_ms={gaps[row['couple']]} for {candidate_path}\n")
    shift = -result.offset_frames
    marker = f"video_anchored:{shift:+d}"
    tools.log_always(f"repair: video_anchored_route trigger={trigger} status=anchored "
                     f"couples={_rows_text(rows)} picture_offset_frames={result.offset_frames:+d} "
                     f"picture_delay_ms={float(picture_ms):.3f} gaps_ms={gaps} "
                     f"constant=yes same_files=yes marker={marker} {numbers} "
                     f"for {candidate_path}\n")
    if repair_deadline is not None and time.monotonic() > repair_deadline:
        orch.log_partial_plan(candidate_path, "repair_budget_exceeded",
                         [("video_anchored", "measured", result.offset_frames)])
        orch.step_result("video_anchored", candidate=candidate_path, status="declined",
                    cause="repair_budget_exceeded")
        return "declined", "repair_budget_exceeded", (
            "the repair's budget ran out after the video measurement -- the partial plan is "
            "logged; the file comes back next wave")

    # ---- the plan: one zone per track at the picture's offset -----------------
    work_dir = os.path.join(tools.tmpFolder, "repair", "video_anchored",
                         merge_video_chimeric.stable_case_key(candidate_path))
    tools.make_dirs(work_dir)
    timeline_ms = merge_video_chimeric.get_master_timeline_length_ms(master_obj)
    offset_ms = orch._decimal(picture_ms)
    track_plans = {}
    for track_language, audio in merge_video_chimeric.iterate_candidate_audios(candidate_obj):
        order = int(audio["StreamOrder"])
        _, extent_ms, extent_source = orch._track_timing(candidate_obj, audio, Decimal(1))
        pieces, adjustments, _ = video_anchored_pieces(offset_ms, extent_ms, timeline_ms)
        for adjustment in adjustments:
            tools.log_line(f"repair: plan_edge_adjustment stream={order} zone=0 "
                           f"kind={adjustment['kind']} "
                           f"master_fill_ms={adjustment['master_fill_ms']}\n")
        track_plans[order] = {
            "pieces": pieces, "extent_ms": extent_ms, "extent_source": extent_source,
            "offset_measured": True, "borrow_reason": None,
            "offset_sources": [{"zone": 0, "offset_ms": str(offset_ms),
                                "source": "video_anchored"}]}
        orch.step_result("track_pieces", candidate=candidate_path, stream=order,
                    language=track_language, n_pieces=len(pieces),
                    pieces=[(p["source"][0], float(p["master_start_ms"]),
                             float(p["master_end_ms"]), float(p["source_start_ms"]))
                            for p in pieces])
    reference_pieces, _, _ = video_anchored_pieces(offset_ms, None, timeline_ms)
    chapters_path, chapter_decisions = merge_video_chimeric.build_delivered_chapters(
        master_obj.filePath, candidate_path, reference_pieces, None, timeline_ms, work_dir)
    for decision in chapter_decisions:
        tools.log_line("repair: chapter " + " ".join(
            f"{key}={str(value).replace(' ', '_')}" for key, value in decision.items()) + "\n")
    master_tracks = (getattr(master_obj, "audios", None) or {}).get(language) or []
    reference_stream = master_tracks[0].get("StreamOrder") if master_tracks else None
    seam = getattr(candidate_obj, merge_video_repair.REPAIR_SEAM_ATTRIBUTE, None)
    job_start_utc = ((seam or {}).get("job_start_utc")
                     or "unstamped(no_repair_seam_standalone_run)")
    plan = {
        "kind": VIDEO_ANCHORED_KIND, "language": language, "reference_stream": reference_stream,
        "quantum_ms": None, "master_path": master_obj.filePath,
        "decided_by": "repair_orchestrator.video_anchored_route",
        "segments_dropped_unusable": 0, "speed_margin": None, "speed_engine": None,
        "speed_margin_absent_reason": "no_rate_relation",
        "segments": [{"master_start_ms": Decimal(0), "master_end_ms": timeline_ms,
                      "candidate_offset_ms": offset_ms,
                      "candidate_offset_ms_by_stream": {order: str(offset_ms)
                                                        for order in track_plans}}],
        "track_plans": track_plans, "reference_pieces": reference_pieces,
        "marker": marker, "chapters_path": chapters_path, "speed_ratio": None,
        "speed_ratio_exact": None, "rate_source": None, "resample_gate": None,
        "repair_deadline": repair_deadline,
        "video_anchored": {"offset_frames": result.offset_frames, "shift_frames": shift,
                           "offset_ms": offset_ms, "fps": str(result.fps),
                           "trigger": trigger},
    }
    orch.step_launch("build", candidate=candidate_path, marker=marker, n_tracks=len(track_plans))
    repaired_obj, assembly = merge_video_repair.build_repaired_video_object(
        candidate_obj, master_obj, plan, os.path.join(tools.tmpFolder, "repair"), job_start_utc)
    out_path = getattr(repaired_obj, "filePath", None)
    exists = bool(out_path) and os.path.exists(out_path)
    orch.step_result("build", candidate=candidate_path, ok=exists, out_path=out_path,
                marker=assembly.get("marker"),
                verification=[(v.get("track"), v.get("outcome"), v.get("worst_lag_ms"))
                              for v in assembly.get("verification") or []])
    if not exists:
        return "declined", "plan_application_no_file", (
            f"the build returned but the video-anchored file is not on disk ({out_path})")
    delivered = merge_video_chimeric.probe_delivered_durations(out_path)
    tools.log_line(
        f"repair: DELIVERED_DURATIONS container_ms={delivered['container_ms']} "
        f"master_video_ms={timeline_ms} video_ms=absent(the_repaired_file_carries_no_video) "
        + " ".join(f"{stream['type']}_{stream['index']}_ms={stream['duration_ms']}"
                   for stream in delivered["streams"])
        + f" max_cue_end_ms={delivered['max_cue_end_ms']} for {candidate_path}\n")
    if seam is not None:
        seam["repaired_obj"] = repaired_obj
        seam["assembly"] = assembly
    reason = (f"video-anchored ({trigger}): every candidate track moved by the picture offset "
              f"{result.offset_frames:+d} frame(s) = {float(picture_ms):.3f} ms on the file "
              f"clock, marker '{marker}', {len(assembly.get('audios') or [])} audio and "
              f"{len(assembly.get('subtitles') or [])} subtitle track(s) rebuilt, the master "
              f"untouched, per-couple audio delays {_rows_text(rows)}, file {out_path}")
    merge_video_repair.record(candidate_path, "repaired", reason, detail={
        "out_path": out_path, "video_anchored": plan["video_anchored"],
        "couples": rows, "gaps_ms": gaps,
        "markers": {r["stream_order"]: r.get("marker") for r in assembly.get("audios") or []},
        "delivered_durations": delivered,
        "fabricated_dropped": assembly.get("fabricated_dropped")})
    orch.step_result("video_anchored", candidate=candidate_path, status="repaired", out_path=out_path,
                seconds=round(time.time() - started, 1))
    return "repaired", None, reason
