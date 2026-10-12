'''Check the audio delay against the pictures at the audio window bounds.

At each bound P (first-file time) both files are decoded over P +- WINDOW_S, the second
shifted by the rounded delay. Scene changes are matched by pHash on each side of each
change, never across one; a pairing is kept only when an adjacent matched change shows the
same interval on both files. Each kept change gives the picture offset, and its residual
against the rounded audio delay, in frames of the best video:
  0 -> delay kept; 1 -> delay moved to the picture's frame; 2 or more -> exception.
With different frame rates the second file is put on the first's grid by nearest pts, so
offsets jitter by one frame: only the 2-or-more check applies.
Delay convention: the second file shows at t + delay (ms) what the first shows at t.
'''

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, getcontext
from fractions import Fraction
from statistics import median
import re
import subprocess
import time

import numpy as np

import frame_hash
import tools

WINDOW_S = 5.0
# Lags scanned around the expected one, and the longest side kept on each side of a change.
REACH = 12
SIDE_MIN_FRAMES = 6
SCENE_THRESHOLD = 27.0
MIN_SCENE_FRAMES = 6
MIN_CHANGES = 3
CONSENSUS = 0.8
# Same rate within this ratio: absorbs millisecond container rates, keeps 1001/1000 apart.
RATE_TOLERANCE = Fraction(1, 10000)
THREADS = 2
POINT_WORKERS = 3
WIDTH, HEIGHT = frame_hash.FRAME_WIDTH, frame_hash.FRAME_HEIGHT
_PTS_RE = re.compile(r"pts_time:\s*(-?[0-9.]+)")


def _parse_rate(text):
    try:
        num, den = str(text).split("/", 1)
        rate = Fraction(int(num), int(den))
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def exact_rate(track):
    '''Exact frame rate from ffprobe r_frame_rate, else MediaInfo FrameRate_Num/Den, else None.'''
    rate = _parse_rate((track.get("ffprobe") or {}).get("r_frame_rate"))
    if rate is None and track.get("FrameRate_Num") and track.get("FrameRate_Den"):
        rate = _parse_rate(f"{track['FrameRate_Num']}/{track['FrameRate_Den']}")
    return rate


def same_rate(rate_1, rate_2):
    return abs(rate_1 / rate_2 - 1) <= RATE_TOLERANCE


def decode(path, stream_order, start_s, duration_s):
    '''Small RGB frames (fit + pad) of [start_s, start_s + duration_s] and their times (s).'''
    start_s = max(0.0, start_s)
    scale = (f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease:flags=area,"
             f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2")
    cmd = [tools.software["ffmpeg"], "-hide_banner", "-nostdin", "-threads", str(THREADS),
           "-ss", f"{start_s:.6f}", "-t", f"{duration_s:.6f}", "-i", path,
           "-map", f"0:{stream_order}", "-an", "-sn", "-vf", f"showinfo,{scale},format=rgb24",
           # passthrough: the output holds exactly the frames showinfo lists
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    frames = frame_hash.frames_from_raw(done.stdout, WIDTH, HEIGHT)
    pts = [float(x) for x in _PTS_RE.findall(done.stderr.decode("utf-8", "replace"))]
    count = min(len(frames), len(pts))
    return frames[:count], np.array(pts[:count]) + start_s


def on_grid(frames, times, frame_s):
    '''Re-index frames onto a frame_s grid from their first time, nearest frame per instant.'''
    grid = np.arange(times[0], times[-1] + frame_s / 2.0, frame_s)
    idx = np.clip(np.searchsorted(times, grid), 1, len(times) - 1)
    idx -= (grid - times[idx - 1]) < (times[idx] - grid)
    return frames[idx], times[idx]


def scene_changes(frames):
    '''Indices of the frames that start a new shot (ContentDetector).'''
    from scenedetect import ContentDetector, FrameTimecode
    detector = ContentDetector(threshold=SCENE_THRESHOLD, min_scene_len=MIN_SCENE_FRAMES)
    cuts = []
    for index, frame in enumerate(frames):
        # ContentDetector reads BGR
        for tc in detector.process_frame(FrameTimecode(index, fps=24.0),
                                         np.ascontiguousarray(frame[..., ::-1])):
            cuts.append(int(tc.frame_num))
    return [c for c in cuts if c > 0]


def _sides(m, m_cuts_sorted, n_m):
    '''Windows (first, length) before and after change m, kept off the neighbouring changes.'''
    pos = int(np.searchsorted(m_cuts_sorted, m))
    prev_cut = int(m_cuts_sorted[pos - 1]) if pos > 0 else 0
    next_cut = int(m_cuts_sorted[pos + 1]) if pos + 1 < len(m_cuts_sorted) else n_m
    out = []
    before = max(prev_cut, m - REACH)
    if m - before >= SIDE_MIN_FRAMES:
        out.append((before, m - before))
    after = min(next_cut, m + REACH)
    if after - m >= SIDE_MIN_FRAMES:
        out.append((m, after - m))
    return out


def match_changes(m_hashes, m_cuts, c_hashes, c_cuts, lo, hi):
    '''For each first-file change m, the offset d in [lo, hi] at which the second file shows
    both of its sides; d must land within one frame of a second-file change and the content
    distance stay under SAME_CONTENT_MAX. Returns [(m, d, distance)] sorted by m.'''
    m_sorted = np.asarray(sorted(set(m_cuts)), dtype=np.int64)
    c_sorted = np.asarray(sorted(set(c_cuts)), dtype=np.int64)
    reach = np.arange(-REACH, REACH + 1, dtype=np.int64)
    matched = []
    for m in m_sorted:
        m = int(m)
        sides = _sides(m, m_sorted, len(m_hashes))
        hyps = c_sorted[(c_sorted >= m + lo - 1) & (c_sorted <= m + hi + 1)] - m
        if not sides or len(hyps) == 0:
            continue
        lags = np.unique((hyps[:, None] + reach[None, :]).ravel())
        lags = lags[(lags >= lo) & (lags <= hi)]
        on_cut = set(int(x) for x in (hyps[:, None] + np.arange(-1, 2)).ravel())
        votes = [frame_hash.align(m_hashes[first:first + length], c_hashes, lags, start=first)
                 for first, length in sides]
        if not all(v.ok for v in votes) or len({v.lag for v in votes}) != 1:
            continue
        d = votes[0].lag
        if d not in on_cut:
            continue
        dists = [frame_hash.window_content_distance(m_hashes[first:first + length], c_hashes, first + d)
                 for first, length in sides]
        if max(dists) <= frame_hash.SAME_CONTENT_MAX:
            matched.append((m, d, float(np.mean(dists))))
    return matched


def adjacent_consistent(matched, tolerance):
    '''Keep the matched changes whose interval to an adjacent matched change is the same on
    both files within `tolerance` frames; a lone or inconsistent pairing is a false pair.'''
    return [cur for i, cur in enumerate(matched)
            if any(abs(cur[1] - nb[1]) <= tolerance for nb in matched[max(0, i - 1):i] + matched[i + 1:i + 2])]


def measure_point(path_1, stream_1, path_2, stream_2, point_s, delay_s, frame_s, same):
    '''Picture offsets (ms) of the kept scene changes around one point, and a short note.'''
    reach_s = REACH * frame_s
    frames_1, times_1 = decode(path_1, stream_1, point_s - WINDOW_S, 2 * WINDOW_S)
    frames_2, times_2 = decode(path_2, stream_2, point_s - WINDOW_S + delay_s - reach_s,
                               2 * (WINDOW_S + reach_s))
    if len(frames_1) < 2 * REACH or len(frames_2) < 2 * REACH:
        return [], "decode"
    if not same:
        frames_2, times_2 = on_grid(frames_2, times_2, frame_s)
    cuts_1 = scene_changes(frames_1)
    if not cuts_1:
        return [], "no_change"
    expected = round((times_1[0] + delay_s - times_2[0]) / frame_s)
    matched = match_changes(frame_hash.hash_frames(frames_1, colour=True), cuts_1,
                            frame_hash.hash_frames(frames_2, colour=True), scene_changes(frames_2),
                            expected - REACH, expected + REACH)
    kept = adjacent_consistent(matched, 0 if same else 1)
    return ([float(times_2[m + d] - times_1[m]) * 1000.0 for m, d, _ in kept],
            f"kept {len(kept)}/matched {len(matched)}/changes {len(cuts_1)}")


def snap_for_merge(video_obj_1, video_obj_2, best_video_obj, delay, points_s):
    '''Return the delay to round to a frame, moved by one frame when the pictures say so.

    Raises:
        Exception when the pictures sit two or more frames from the audio delay.
    '''
    delay = Decimal(delay)
    getcontext().prec = 10
    frame_ms = Decimal('1000.0') / Decimal(best_video_obj.video["FrameRate"])
    rounded = round(delay / frame_ms)
    audio_ms = float(rounded * frame_ms)
    started = time.monotonic()
    try:
        track_1, track_2 = video_obj_1.video, video_obj_2.video
        rate_1, rate_2 = exact_rate(track_1), exact_rate(track_2)
        same = same_rate(rate_1, rate_2)
        frame_s = float(1 / rate_1)
        margin_s = WINDOW_S + REACH * frame_s
        dur_1, dur_2 = float(track_1["Duration"]), float(track_2["Duration"])
        points = [p for p in points_s if p - WINDOW_S >= 0 and p + WINDOW_S <= dur_1
                  and p + audio_ms / 1000.0 - margin_s >= 0 and p + audio_ms / 1000.0 + margin_s <= dur_2]
        with ThreadPoolExecutor(POINT_WORKERS) as pool:
            measures = list(pool.map(lambda p: measure_point(
                video_obj_1.filePath, track_1["StreamOrder"], video_obj_2.filePath,
                track_2["StreamOrder"], p, audio_ms / 1000.0, frame_s, same), points))
    except Exception as exc:
        tools.logs.append(f"\t\tframe_snap declined: {type(exc).__name__}: {exc}; plain rounding\n")
        return delay
    residuals, notes, far = [], [], []
    for point, (offsets, note) in zip(points, measures):
        found = [(o - audio_ms) / float(frame_ms) for o in offsets]
        found = [round(r) for r in found] if same else [round(r, 2) for r in found]
        residuals += found
        notes.append(f"{point:.0f}s {found} ({note})")
        if found and abs(round(median(found))) >= 2:
            far.append(f"{point:.0f}s")
    verdict, shift = "too_few_changes", 0
    if len(residuals) >= MIN_CHANGES:
        shift = round(median(residuals))
        agree = sum(1 for r in residuals if abs(r - shift) <= (0.5 if same else 1))
        verdict = "no_consensus" if agree < CONSENSUS * len(residuals) else "picture"
    summary = (f"audio delay {delay} ms = frame {rounded} of {frame_ms:.3f} ms, rates {rate_1} vs {rate_2}; "
               f"picture minus audio, in frames, per point: {'; '.join(notes)}; "
               f"{len(points_s) - len(points)} point(s) outside a file; "
               f"files {video_obj_1.filePath} | {video_obj_2.filePath}; {time.monotonic() - started:.1f} s")
    if far:
        raise Exception(f"frame_snap: at {', '.join(far)} the pictures sit two or more frames from "
                        f"the audio delay, more than a rounding: abnormal pair. {summary}")
    if verdict == "picture" and shift and same:
        tools.logs.append(f"\t\tframe_snap corrected {shift:+d} frame: {summary}\n")
        return Decimal((rounded + shift) * frame_ms)
    if verdict == "picture":
        verdict = "confirmed" if shift == 0 else "within_rate_jitter"
    tools.logs.append(f"\t\tframe_snap {verdict} {shift:+d}: {summary}\n")
    return delay
