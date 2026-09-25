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
  2. each video is decoded ONCE, whole (`-threads 3`), to 128x72 BGR; that single stream feeds
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
import threading
import time
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
DECODE_THREADS = 3
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


def decode_scenes_and_hashes(path, fps, duration_s, work_dir):
    '''`(cuts, hashes, from_cache)` for the whole video: `cuts` = ContentDetector scene starts
    (absolute frame indices of the decoded stream), `hashes` = uint64 pHash per decoded frame.
    Raises `tools.decoder_timeout` past the ADDENDUM 26.3 bound, RuntimeError on a failed decode.
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
    tools.dev_log(f"{LOG_PREFIX}decode starting file={path} timeout_s={timeout}\n")
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
                         audio_delay_hints_ms=None):
    '''The ADDENDUM 27 measurement. Never raises for a media reason: every failure is a named
    status on the returned `VideoOffsetResult`, and exactly one `log_always` line says it.'''
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
    # The two decodes run side by side (each `-threads 3`): decoding is the whole cost (memo
    # MS_WALK_EXPERIMENT_20260925) and the two files are independent -- measured on Ragnarok
    # S02E14 at host load 55-65: 283 s one after the other.
    with ThreadPoolExecutor(max_workers=CONCURRENT_DECODES) as pool:
        futures = [pool.submit(decode_scenes_and_hashes, path, fps, info["duration_s"], work_dir)
                   for path, info in ((master_path, m_info), (candidate_path, c_info))]
        errors, outcomes = [], []
        for future in futures:
            try:
                outcomes.append(future.result())
            except tools.decoder_timeout as exc:
                errors.append((STATUS_DECODER_TIMEOUT, str(exc)))
            except (RuntimeError, OSError) as exc:
                errors.append((STATUS_DECODE_FAILED, str(exc)[:200]))
    if errors:
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
