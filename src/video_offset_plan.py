'''Align a candidate to a self-contradicting master by matching their pictures.

When the master's comparison-language tracks disagree with each other, the audio cannot
place the candidate. Both files carry the same pictures, so matched scene changes give one
constant offset in frames, applied to every candidate track; additions happen only at the
head and tail.

Entry points:

    measure_video_offset(master_path, candidate_path, work_dir, log) -> VideoOffsetResult
    build_candidate_plan(result, master_obj_like, candidate_obj_like) -> dict

Convention: `offset_frames = d` means candidate frame `k + d` shows master frame `k`. A
candidate track is delayed by
`candidate_track_delay_ms = (master_video_start - candidate_video_start) - d x frame`
(exact rationals). The marker `video_anchored:<+-N>` carries N = -d.

Measurement:
  1. both videos must be CFR at the same exact rate, otherwise `fps_mismatch`;
  2. each video is decoded once to 128x72, feeding PySceneDetect's ContentDetector and a
     per-frame grey pHash (`frame_hash`), plus Cb/Cr hashes next to each change; results are
     cached per file in `work_dir`;
  3. each master change has two windows of up to `SIDE_FRAMES` frames, one entirely before it
     and one entirely after it (never across another change);
  4. each window is aligned (`frame_hash.align`) against the candidate around the candidate
     changes within the search bound; the change votes when the sides with a clear minimum
     agree, that offset sits within one frame of a candidate change, and their content
     distance is under `PAIR_CONTENT_MAX`; the offset is the most-voted value (ties: lower
     mean distance);
  5. constancy: each third of the master must elect the same offset with at least
     `MIN_PAIRS_PER_THIRD` pairs, no run of pairs may agree on another offset, and the fitted
     trend must stay under one frame -- otherwise `video_offset_not_constant` or
     `video_offset_coverage_incomplete`.
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
import numpy as np

import frame_hash
import repair_pool
import tools

# --------------------------------------------------------------------------------------------
# Named constants
# --------------------------------------------------------------------------------------------

CACHE_VERSION = 3
CACHE_DIRNAME = "video_offset_cache"

# One decode per file at 128x72 BGR for ContentDetector (its HSV means are nearly
# scale-free); the pHash reads the same pictures.
DECODE_WIDTH = 128
DECODE_HEIGHT = 72
# Decoder threads per file, kept low to fit the repair's budget.
DECODE_THREADS = 2
BATCH_FRAMES = 512

# ContentDetector's default threshold.
SCENE_THRESHOLD = 27.0

# Window on each side of a change, and the shortest side kept when another change is
# closer than that.
SIDE_FRAMES = 12
SIDE_MIN_FRAMES = 6

# Lags scanned on each side of every candidate-change hypothesis; the lags around the
# hypothesis give the margin gate its rivals (a static or repeated shot ties them).
LAG_REACH = 12

# Colour hashes are kept for frames this close to a change (a side window at +-1 frame).
COLOUR_REACH = SIDE_FRAMES + 1

# A side shows the same content as the candidate when its mean `frame_hash.content_distance`
# is at most this (fraction of the bits).
PAIR_CONTENT_MAX = frame_hash.SAME_CONTENT_MAX

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
# The decode was stopped by the repair's deadline, not by its own bound.
STATUS_BUDGET = "repair_budget_exceeded"

LOG_PREFIX = "video_offset: "


# --------------------------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------------------------

class VideoOffsetResult:
    '''Result of `measure_video_offset`.

    `offset_frames` is valid only when `status == STATUS_OK`; any other status is a named decline.
    '''

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
        self.mean_distance = fields.pop("mean_distance", None)
        self.covered_frames = fields.pop("covered_frames", None)   # (first, last) master frame
        self.thirds = fields.pop("thirds", None)               # [(mode, votes_at_mode, pairs_at_d)]
        self.regimes = fields.pop("regimes", None)
        self.trend_frames = fields.pop("trend_frames", None)
        self.pairs = fields.pop("pairs", None)                 # [(m_cut, d, dist)] matched
        self.wall_s = fields.pop("wall_s", None)
        self.extra = fields

    @property
    def frame_ms(self):
        """Duration of one frame in ms, exact, or None."""
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
        """Return a JSON-friendly dict of the result."""
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
    '''Precondition: both videos CFR at the same exact rate. `(Fraction, None)` or
    `(None, reason)`.'''
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

def _colour_rows(cuts, first, count):
    '''Indices in [first, first + count) within COLOUR_REACH of a change.'''
    if not cuts or count <= 0:
        return np.zeros(0, dtype=np.int64)
    rows = np.arange(first, first + count, dtype=np.int64)
    marks = np.asarray(sorted(cuts), dtype=np.int64)
    pos = np.searchsorted(marks, rows)
    after = marks[np.minimum(pos, len(marks) - 1)]
    before = marks[np.maximum(pos - 1, 0)]
    near = (np.abs(after - rows) <= COLOUR_REACH) | (np.abs(rows - before) <= COLOUR_REACH)
    return rows[near]


def _cache_path(work_dir, path):
    """Return the cache file path of a video's decode, keyed by file identity and settings."""
    st = os.stat(path)
    key = json.dumps([os.path.realpath(path), st.st_size, st.st_mtime_ns, CACHE_VERSION,
                      SCENE_THRESHOLD, DECODE_WIDTH, DECODE_HEIGHT])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    return os.path.join(work_dir, CACHE_DIRNAME, f"{digest}.npz")


class BudgetExceeded(Exception):
    '''The decode was stopped by the repair's `time.monotonic()` deadline; says nothing about the media.'''


def decode_scenes_and_hashes(path, fps, duration_s, work_dir, deadline=None):
    '''Decode a whole video once and return `(cuts, hashes, coloured, from_cache)`.

    `cuts` are ContentDetector scene starts (frame indices); `hashes` is a FrameHashes of every
    frame whose colour rows are filled only where `coloured` (bool per frame) is set, near the
    changes. Raises `tools.decoder_timeout`, `BudgetExceeded` when `deadline` comes first, or
    RuntimeError on a failed decode.
    '''
    cache = _cache_path(work_dir, path) if work_dir else None
    if cache and os.path.exists(cache):
        with np.load(cache) as data:
            n = len(data["grey"])
            colour = np.zeros((n,) + data["colour"].shape[1:], dtype=np.uint64)
            coloured = np.zeros(n, dtype=bool)
            colour[data["colour_rows"]] = data["colour"]
            coloured[data["colour_rows"]] = True
            hashes = frame_hash.FrameHashes(data["grey"].copy(), colour, data["std"].copy())
            return [int(x) for x in data["cuts"]], hashes, coloured, True

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
    # The last two batches stay in memory: a change is reported a few frames after it
    # happens, and its colour rows are hashed once it is known.
    held, colour = [], {}

    def _colour_near_changes():
        for first, rgb in held:
            for row in _colour_rows(cuts, first, len(rgb)):
                if int(row) not in colour:
                    one = frame_hash.hash_frames(rgb[row - first:row - first + 1], colour=True)
                    colour[int(row)] = one.colour[0]

    try:
        while True:
            blob = proc.stdout.read(frame_bytes * BATCH_FRAMES)
            if not blob:
                break
            usable = len(blob) - len(blob) % frame_bytes
            frames = np.frombuffer(blob[:usable], dtype=np.uint8).reshape(
                -1, DECODE_HEIGHT, DECODE_WIDTH, 3)
            first = index
            for frame in frames:
                for tc in detector.process_frame(FrameTimecode(index, fps=fps), frame):
                    cuts.append(int(tc.frame_num))
                index += 1
            rgb = frames[..., ::-1]
            hash_parts.append(frame_hash.hash_frames(rgb))
            held = (held + [(first, rgb)])[-2:]
            _colour_near_changes()
            if usable != len(blob):
                break
        if index:
            for tc in detector.post_process(FrameTimecode(index - 1, fps=fps)):
                cuts.append(int(tc.frame_num))
            _colour_near_changes()
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
    grey = frame_hash.FrameHashes.concat(hash_parts)
    cuts = sorted(set(c for c in cuts if 0 < c < len(grey)))
    rows = np.asarray(sorted(colour), dtype=np.int64)
    words = frame_hash.COLOUR_BITS // 64
    colour_rows = (np.stack([colour[int(r)] for r in rows]) if len(rows)
                   else np.zeros((0, 2, words), dtype=np.uint64))
    full = np.zeros((len(grey), 2, words), dtype=np.uint64)
    coloured = np.zeros(len(grey), dtype=bool)
    full[rows] = colour_rows
    coloured[rows] = True
    hashes = frame_hash.FrameHashes(grey.grey, full, grey.std)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = cache + ".tmp.npz"
        np.savez(tmp, cuts=np.asarray(cuts, dtype=np.int64), grey=grey.grey, std=grey.std,
                 colour_rows=rows, colour=colour_rows)
        os.replace(tmp, cache)
    return cuts, hashes, coloured, False


# --------------------------------------------------------------------------------------------
# Matching and constancy -- pure functions over scene lists and hash arrays
# --------------------------------------------------------------------------------------------

def _sides(m, m_cuts_sorted, n_m):
    '''Master windows `(name, first, length)` before and after change `m`, each kept off
    the neighbouring changes; sides shorter than SIDE_MIN_FRAMES are left out.'''
    pos = int(np.searchsorted(m_cuts_sorted, m))
    prev_cut = int(m_cuts_sorted[pos - 1]) if pos > 0 else 0
    next_cut = int(m_cuts_sorted[pos + 1]) if pos + 1 < len(m_cuts_sorted) else n_m
    out = []
    before = max(prev_cut, m - SIDE_FRAMES)
    if m - before >= SIDE_MIN_FRAMES:
        out.append(("before", before, m - before))
    after = min(next_cut, m + SIDE_FRAMES)
    if after - m >= SIDE_MIN_FRAMES:
        out.append(("after", m, after - m))
    return out


def match_changes(m_hashes, m_cuts, c_hashes, c_cuts, lo, hi, m_coloured=None,
                  c_coloured=None):
    '''Find, for each master change, the candidate offset `d` in [lo, hi].

    Args:
        m_hashes, c_hashes: FrameHashes of the whole files (colour needed near changes).
        m_coloured, c_coloured: optional bool per frame, where the colour rows are valid.

    Returns:
        `(matched, ambiguous, total)`: `matched` lists `(m_cut, d, mean content distance)`
        sorted by m_cut; `ambiguous` counts changes no side could place.
    '''
    n_m = len(m_hashes)
    m_sorted = np.asarray(sorted(set(m_cuts)), dtype=np.int64)
    c_sorted = np.asarray(sorted(set(c_cuts)), dtype=np.int64)
    reach = np.arange(-LAG_REACH, LAG_REACH + 1, dtype=np.int64)
    matched, ambiguous, total = [], 0, 0
    for m in m_sorted:
        m = int(m)
        sides = _sides(m, m_sorted, n_m)
        if not sides:
            continue
        total += 1
        left = np.searchsorted(c_sorted, m + lo - 1, side="left")
        right = np.searchsorted(c_sorted, m + hi + 1, side="right")
        hyps = c_sorted[left:right] - m
        if len(hyps) == 0:
            continue
        lags = np.unique((hyps[:, None] + reach[None, :]).ravel())
        lags = lags[(lags >= lo) & (lags <= hi)]
        on_cut = set(int(x) for x in np.unique((hyps[:, None] + np.arange(-1, 2)).ravel()))
        votes = []
        for _, first, length in sides:
            side = m_hashes[first:first + length]
            alignment = frame_hash.align(side, c_hashes, lags, start=first)
            if alignment.ok:
                votes.append((alignment.lag, first, length))
        if not votes:
            ambiguous += 1
            continue
        d = votes[0][0]
        # Sides at different offsets mark a boundary at this change; a lag away from every
        # candidate change does not confirm it. Neither votes.
        if any(lag != d for lag, _, _ in votes) or d not in on_cut:
            continue
        dists = []
        for _, first, length in votes:
            rows = np.arange(first, first + length)
            if ((m_coloured is not None and not m_coloured[rows].all())
                    or (c_coloured is not None and not c_coloured[rows + d].all())):
                break
            dists.append(frame_hash.window_content_distance(m_hashes[first:first + length],
                                                            c_hashes, first + d))
        if len(dists) != len(votes) or not max(dists) <= PAIR_CONTENT_MAX:
            continue
        matched.append((m, d, float(np.mean(dists))))
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
    '''Check that the matched pairs prove one constant offset (module docstring, step 5).

    Returns a dict: status, d, paired, mean_distance, thirds, regimes, trend_frames, covered.
    '''
    out = {"d": None, "paired": 0, "mean_distance": None, "thirds": [], "regimes": [],
           "trend_frames": None, "covered": None}
    d, votes = _mode([(dd, dist) for _, dd, dist in matched])
    if d is None or votes < MIN_PAIRS_PER_THIRD:
        out["status"] = STATUS_UNMATCHED
        out["d"] = d
        out["paired"] = votes
        return out
    agreeing = [(m, dist) for m, dd, dist in matched if dd == d]
    out["d"], out["paired"] = d, len(agreeing)
    out["mean_distance"] = sum(dist for _, dist in agreeing) / len(agreeing)
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
    '''Return the offset search bound (lo, hi) in frames.

    With audio delay hints: max|hint| + margin; otherwise the default, widened to the duration
    difference plus a margin.
    '''
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
    """Write one timestamped line to `log`, or to `tools.log_always` when it is not callable."""
    now = datetime.datetime.now(datetime.timezone.utc)
    stamped = (f"{LOG_PREFIX}utc={now.strftime('%Y-%m-%dT%H:%M:%S')}."
               f"{now.microsecond // 1000:03d}Z {line}\n")
    if callable(log):
        log(stamped)
    else:
        tools.log_always(stamped)


def _log_line(result, bound):
    """Return the one-line summary of a measurement."""
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
    if result.mean_distance is not None:
        parts.append(f"mean_distance={result.mean_distance:.3f}/{PAIR_CONTENT_MAX}")
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
    '''Measure the constant frame offset between two videos.

    Never raises for a media reason: every failure is a named status on the returned
    `VideoOffsetResult`, logged on one line. A decode stopped by `deadline` returns `STATUS_BUDGET`.
    '''
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
    # Both files are decoded concurrently on the shared repair pool: decoding dominates the cost.
    pool = repair_pool.get_pool()
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
    (m_cuts, m_hashes, m_coloured, m_cached), (c_cuts, c_hashes, c_coloured, c_cached) = outcomes

    common.update(master_frames=len(m_hashes), candidate_frames=len(c_hashes),
                  master_scenes=len(m_cuts), candidate_scenes=len(c_cuts),
                  master_cached=m_cached, candidate_cached=c_cached)
    bound = search_bound_frames(fps, m_info["duration_s"], c_info["duration_s"],
                                audio_delay_hints_ms)
    matched, ambiguous, total = match_changes(m_hashes, m_cuts, c_hashes, c_cuts, *bound,
                                              m_coloured=m_coloured, c_coloured=c_coloured)
    proof = prove_constant(matched, len(m_hashes))
    status = proof["status"]
    result = VideoOffsetResult(
        status, offset_frames=proof["d"] if status == STATUS_OK else None,
        paired=proof["paired"], matched=len(matched), ambiguous=ambiguous, total=total,
        mean_distance=proof["mean_distance"], covered_frames=proof["covered"],
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
    '''Build the video-anchored plan.

    Master tracks are untouched; every candidate track and the chapters are shifted by the
    video offset. Head and tail gaps are filled from the master's comparison-language audio
    when available, else silence.

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
# Entry point 3: the detection and the route, called by `repair_orchestrator`
# --------------------------------------------------------------------------------------------

class _Orchestrator:
    """Lazy proxy to `repair_orchestrator`, avoiding a circular import."""

    def __getattr__(self, name):
        import repair_orchestrator
        return getattr(repair_orchestrator, name)


orch = _Orchestrator()

# When the audio alignment contradicts itself, the video arbitrates via scene-change matching:
#   (i)   the master's comparison-language tracks disagree among themselves;
#   (ii)  the candidate's comparison-language tracks disagree by more than one frame;
#   (iii) the audio delay and the picture offset differ by more than one frame.
# Disagreeing couples are never averaged; every candidate track follows the candidate's
# picture, and the master is never modified.

TRIGGER_MASTER_DESYNC = "master_intertrack_desync"          # (i)
TRIGGER_CANDIDATE_DESYNC = "candidate_intertrack_desync"    # (ii)
TRIGGER_PICTURE_DISAGREE = "audio_video_offset_disagree"    # (iii)
VIDEO_ANCHORED_KIND = "orchestrator_video_anchored"
# Per-couple delay: master_self_check's 30 s window cross-correlation, +-1 s around the coarse
# offset, read at one instant (mid-file) for every couple so a drift affects them alike.
COUPLE_DELAY_WINDOW_S = 30.0
COUPLE_DELAY_SEARCH_S = 1.0
COUPLE_DELAY_POSITION = 0.5


def _video_frame_ms(video_obj):
    """Return one frame duration in ms (exact rate when readable, else `FrameRate`), or None."""
    try:
        import frame_snap
        rate, _ = frame_snap.exact_rate(video_obj.video)
        if rate is not None:
            return Fraction(1000) / Fraction(rate)
        return Fraction(1000) / Fraction(str(video_obj.video["FrameRate"]))
    except Exception:                                                    # noqa: BLE001
        return None


def coarse_offsets_from_prime(primed, master_obj, candidate_obj, language, at_s):
    """Return `{couple: ms}`, each couple's coarse offset on the file clock at master instant `at_s`.

    Couples that could not be measured, are under the coverage floor, or have no zone at that
    instant are omitted.
    """
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
    """Measure each couple's audio delay in ms on the file clock (candidate = master + delay).

    One row per couple; an unmeasured row carries its `reason`.
    """
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
    """Detect triggers (ii) and (i) from the couple rows; return `(trigger, evidence)` or `(None, None)`.

    Couples sharing one master track more than one frame apart mean the candidate disagrees;
    couples sharing one candidate track more than `master_self_check.MIN_LAG_MS_FOR_DESYNC`
    apart mean the master disagrees (same floor as master_self_check).
    """
    import master_self_check
    if frame_ms is None:
        return None, None
    measured = [r for r in rows if r["delay_ms"] is not None]
    for trigger, key, floor_ms in (
            (TRIGGER_CANDIDATE_DESYNC, "master_stream", float(frame_ms)),
            (TRIGGER_MASTER_DESYNC, "candidate_stream",
             float(master_self_check.MIN_LAG_MS_FOR_DESYNC))):
        groups = {}
        for row in measured:
            groups.setdefault(row[key], []).append(row)
        for shared, group in groups.items():
            values = [r["delay_ms"] for r in group]
            if len(values) > 1 and max(values) - min(values) > floor_ms:
                return trigger, {"shared_" + key: shared,
                                 "delays_ms": {r["couple"]: r["delay_ms"] for r in group},
                                 "spread_ms": round(max(values) - min(values), 3),
                                 "frame_ms": round(float(frame_ms), 3),
                                 "floor_ms": round(floor_ms, 3), "source": "couples"}
    return None, None


def detect_audio_contradiction(master_obj, candidate_obj, language, primed, candidate_path):
    """Check triggers (ii), (i) and (iii) after the prime; return `(trigger, evidence, rows)`.

    `trigger` None means the audios agree and the ordinary path continues.
    """
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


def video_status_cause(status, trigger):
    """Map a video status to (route status, cause).

    A non-constant offset falls back to the chimeric route for triggers (ii)/(iii)
    but declines for (i), where the couples themselves disagree.
    """
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
    """Place one candidate track on the master timeline at the picture's offset.

    One zone `[0, timeline)` read at `master + offset_ms`; head and tail are filled where the
    track has no content.
    """
    zone = {"master_start_ms": Decimal(0), "master_end_ms": timeline_ms,
            "offset_ms": offset_ms, "n_windows": 0, "zone": 0}
    return orch.track_pieces([zone], [], [{"zone": 0, "offset_ms": offset_ms}], extent_ms,
                        timeline_ms)


def video_anchored_route(trigger, evidence, master_obj, candidate_obj, language, rows,
                         repair_deadline):
    """Run the video-anchored route; return `(status, cause, reason)`.

    Status is repaired, declined or fallback. Every candidate track and the chapters are moved
    by the picture offset; the master is never touched; additions happen only at head and tail.
    """
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
               f"mean_distance={None if result.mean_distance is None else round(result.mean_distance, 3)} "
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
        # Trigger (i) from master_self_check has no prime: read every couple around the picture's delay.
        couples = orch.enumerate_couples(master_obj, candidate_obj, language)
        at_s = float(result.master_frames / result.fps) * COUPLE_DELAY_POSITION
        rows = couple_fine_delays(master_obj, candidate_obj, couples,
                                  {f"{m}x{c}": float(picture_ms) for m, c in couples}, at_s)
    gaps = {r["couple"]: round(r["delay_ms"] - float(picture_ms), 3)
            for r in rows if r["delay_ms"] is not None}
    for row in rows:
        if row["delay_ms"] is not None and abs(gaps[row["couple"]]) > float(frame_ms):
            # Signal (iii) is logged only, never acted on here.
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
            "logged; declined, retried at the next run")

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
