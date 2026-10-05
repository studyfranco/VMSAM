'''Snap an audio delay to the video frame the pictures agree on.

Plain rounding, `round(delay / frame_ms)`, can land one frame off when the
audio delay sits near a half frame; this module checks the pictures instead.

It only moves the rounded frame by -1, 0 or +1, and only when both videos are
constant-frame-rate at the same rate (within RATE_TOLERANCE). At K positions
over the common span it decodes 2G frames of the first video from `t`, keeps the
first G of them that hold no scene cut (so no window straddles a cut), and
decodes G + 4 frames of the second around that group's time + delay (small RGB,
one ffmpeg call each). Each candidate offset (rounded frame -2..+2) is a sliding
G-frame window of the second group, scored by `frame_hash.align` (mean grey
pHash distance; a position counts only when its minimum is clear of the lags two
frames away). -1/0/+1 vote; +-2 is only a guard, logged and never selected.

Decision (every outcome on ONE `log_always` line, `frame_snap ...`):
  * the offset with the lowest total over the valid positions, IF it wins at a
    strict majority of them AND beats the runner-up's total by more than
    `MARGIN_MIN` (relative) -> chosen;
  * winners that change monotonically along the file (e.g. -1 early, +1 late)
    are not a constant delay -> `frame_snap_drift`, rounding kept (the rate arm
    owns drift);
  * otherwise -> `frame_snap_no_consensus`, rounding kept.
Anything this module cannot establish (other rate, VFR, unreadable rate, too few
usable positions, a decoder that fails or runs out of budget, any exception)
keeps the plain rounding, bit for bit, and says why.

Delay convention: the second video shows at `t + delay` (ms) what the first
shows at `t`.

Entry points: `snap_for_merge` (from `adjust_delay_to_frame`) and
`probe_disagreement` / `disagreement` (repair path).
'''

from decimal import Decimal, getcontext
from fractions import Fraction
import subprocess
import re
import threading
import time

import numpy as np
from PIL import Image

import frame_hash
import tools

# -- tunables ----------------------------------------------------------------
POSITIONS = 8            # K: positions spread over the common span
GROUP_FRAMES = 12        # G: consecutive frames compared at each position
REF_DECODE_FRAMES = 2 * GROUP_FRAMES   # first-video frames decoded to find a cut-free group
EDGE_FRACTION = 0.05     # the first/last 5 % of the common span are never used
VOTE_OFFSETS = (-1, 0, 1)
GUARD_OFFSETS = (-2, 2)  # logged, never selected
REACH = 2                # frames of the second video decoded on each side
WIDTH, HEIGHT = frame_hash.FRAME_WIDTH, frame_hash.FRAME_HEIGHT
THREADS = 3
# Group statistics (black, static) are read on a 64x36 area-averaged picture.
STATS_SIZE = (64, 36)
# A group whose mean inter-frame difference (grey levels, 0-255, on the 64x36
# picture) is below this cannot tell one frame from its neighbour.
STATIC_MEAN_DIFF = 0.3
# A frame darker than this mean AND flatter than BLACK_STD is black.
BLACK_MEAN = 18.0
BLACK_STD = 6.0
# ContentDetector threshold for a cut inside a first-video group (its default).
GROUP_CUT_THRESHOLD = 27.0
# A position whose best window is this far (`frame_hash.content_distance`,
# fraction of the bits, mean over the group) shows other content: the delay does
# not hold there (a step, a recap cut), so it votes for nothing.
UNMATCHED_DISTANCE = frame_hash.SAME_CONTENT_MAX
# Required relative margin of the winner's total over the runner-up's (right winners: median
# 0.84, 0.4 % under 0.20; wrong winners: median 0.18).
MARGIN_MIN = 0.20
# Fewest valid positions for a decision (wrong decisions: 0.12 % at 5, 0.19 % at 4).
MIN_VALID_POSITIONS = 5
# Two rates are "the same" when |a/b - 1| <= this: it absorbs millisecond
# container timestamps (18965/791 vs 24000/1001) but keeps 1001/1000 pairs
# apart, which belong to the rate-change path.
RATE_TOLERANCE = Fraction(1, 10000)
BUDGET_S = 20.0          # wall-clock budget for one pair

# Last `audio_video_offset_disagree` signal per {(first_path, second_path): dict},
# written by `snap_for_merge`/`probe_disagreement`, read through `disagreement`.
_SIGNALS = {}
_SIGNALS_LOCK = threading.Lock()


def disagreement(first_path, second_path):
    '''Return the last `audio_video_offset_disagree` signal recorded for this pair.

    Returns:
        dict with audio_frames/audio_ms, picture_frames/picture_ms, votes, valid,
        positions_s and fps, or None when no disagreement was found.
    '''
    with _SIGNALS_LOCK:
        return _SIGNALS.get((first_path, second_path))


_PTS_RE = re.compile(r"pts_time:\s*(-?[0-9.]+)")


# -- exact frame rate --------------------------------------------------------

def _parse_rate(value):
    if value is None:
        return None
    try:
        text = str(value).strip()
        if "/" in text:
            num, den = text.split("/", 1)
            rate = Fraction(int(num), int(den))
        else:
            return None
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def exact_rate(video_track):
    '''Return the stream's exact frame rate.

    Reads MediaInfo FrameRate_Num/Den, then ffprobe r_frame_rate; never the
    rounded decimal `FrameRate`, which hides 1001/1000 differences.

    Returns:
        (Fraction, source) on success, (None, "no_exact_rate") otherwise.
    '''
    num, den = video_track.get("FrameRate_Num"), video_track.get("FrameRate_Den")
    if num not in (None, "") and den not in (None, ""):
        rate = _parse_rate(f"{num}/{den}")
        if rate is not None:
            return rate, "mediainfo"
    probe = video_track.get("ffprobe") or {}
    rate = _parse_rate(probe.get("r_frame_rate"))
    if rate is not None:
        return rate, "ffprobe"
    return None, "no_exact_rate"


def same_rate(rate_1, rate_2):
    '''True when both rates are known and equal within RATE_TOLERANCE.'''
    return (rate_1 is not None and rate_2 is not None
            and abs(Fraction(rate_1) / Fraction(rate_2) - 1) <= RATE_TOLERANCE)


def _duration_s(video_track):
    for key in ("Duration",):
        try:
            value = float(video_track[key])
            if value > 0:
                return value
        except (KeyError, TypeError, ValueError):
            pass
    try:
        return float(video_track["FrameCount"]) / float(video_track["FrameRate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


# -- decoding ----------------------------------------------------------------

def decode_group(path, stream_order, start_s, n_frames):
    '''Decode `n_frames` small RGB frames from `start_s`.

    Frame times are read from showinfo (`start_s + pts_time`), not assumed,
    so an inexact seek shifts the times rather than the comparison.

    Returns:
        (frames array of shape (n, HEIGHT, WIDTH, 3) uint8, times array in seconds).
    '''
    start_s = max(0.0, float(start_s))
    cmd = [tools.software["ffmpeg"], "-hide_banner", "-nostdin", "-threads", str(THREADS),
           "-ss", f"{start_s:.6f}", "-i", path, "-map", f"0:{stream_order}",
           "-frames:v", str(n_frames), "-an", "-sn",
           "-vf", f"showinfo,scale={WIDTH}:{HEIGHT}:flags=area,format=rgb24",
           # passthrough: the raw output must hold exactly the frames showinfo lists
           # (the default constant-rate output may duplicate the first one)
           "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    timeout = tools.decoder_timeout_for(n_frames / 10.0)
    try:
        done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise tools.decoder_timeout("frame_snap_decode", timeout, f"file={path} start_s={start_s}")
    frames = frame_hash.frames_from_raw(done.stdout, WIDTH, HEIGHT, 3)
    pts = [float(x) for x in _PTS_RE.findall(done.stderr.decode("utf-8", "replace"))]
    count = min(len(frames), len(pts))
    times = np.array([start_s + p for p in pts[:count]], dtype=np.float64)
    return frames[:count], times


# -- group checks ------------------------------------------------------------

def _stats_frames(frames):
    '''Grey 64x36 area-averaged float copies of RGB (or grey) frames.'''
    grey = frame_hash.grey_frames(frames) if np.ndim(frames) == 4 else np.asarray(frames, np.uint8)
    if len(grey) == 0:
        return np.zeros((0, STATS_SIZE[1], STATS_SIZE[0]), np.float32)
    return np.stack([np.asarray(Image.fromarray(g, "L").resize(STATS_SIZE, Image.BOX), np.float32)
                     for g in grey])


def group_cuts(frames):
    '''Indices of the frames of a run of RGB frames that start a new shot (ContentDetector).'''
    from scenedetect import ContentDetector, FrameTimecode
    detector = ContentDetector(threshold=GROUP_CUT_THRESHOLD, min_scene_len=1)
    cuts = []
    for index, frame in enumerate(frames):
        # ContentDetector reads BGR
        for tc in detector.process_frame(FrameTimecode(index, fps=24.0),
                                         np.ascontiguousarray(frame[..., ::-1])):
            cuts.append(int(tc.frame_num))
    return cuts


def cut_free_start(frames, length=GROUP_FRAMES):
    '''Start of the first run of `length` frames that holds no scene cut, or None.'''
    cuts = group_cuts(frames)
    for start in range(0, len(frames) - length + 1):
        if not any(start < cut < start + length for cut in cuts):
            return start
    return None


def group_is_usable(frames):
    '''Check that a group is long enough, not black and not static.

    Args:
        frames: (n, h, w, 3) RGB frames.

    Returns:
        (True, "") or (False, reason).
    '''
    if len(frames) < GROUP_FRAMES:
        return False, f"short:{len(frames)}"
    stats = _stats_frames(frames)
    means = stats.reshape(len(stats), -1).mean(axis=1)
    stds = stats.reshape(len(stats), -1).std(axis=1)
    if np.sum((means < BLACK_MEAN) & (stds < BLACK_STD)) > len(stats) // 2:
        return False, "black"
    motion = float(np.mean(np.abs(np.diff(stats, axis=0))))
    if motion < STATIC_MEAN_DIFF:
        return False, f"static:{motion:.2f}"
    return True, ""


# -- distance ----------------------------------------------------------------

def score_offsets(ref_hashes, ref_times, other_hashes, other_times, base_frames, frame_s):
    '''Score offsets -2..+2 at one position with `frame_hash.align`.

    Offset `o` is the G-frame window of the second group starting at
    `anchor + REACH + o`, where `anchor` is located from the decoded frame
    times. Offsets whose window is incomplete are not scored.

    Args:
        ref_hashes, other_hashes: FrameHashes of the two groups.

    Returns:
        ({offset: mean grey distance}, Alignment or None when the groups cannot
        be paired by their times).
    '''
    if len(other_times) == 0 or len(ref_times) == 0:
        return {}, None
    target = ref_times[0] + (base_frames - REACH) * frame_s
    anchor = int(np.argmin(np.abs(other_times - target)))
    if abs(other_times[anchor] - target) > frame_s / 2.0:
        return {}, None
    alignment = frame_hash.align(ref_hashes, other_hashes, VOTE_OFFSETS + GUARD_OFFSETS,
                                 start=anchor + REACH)
    return dict(alignment.curve), alignment


# -- decision ----------------------------------------------------------------

def is_drift(winners):
    '''True when per-position winners change monotonically (not a constant delay).

    At least two positions must differ from the most common winner.
    '''
    if len(set(winners)) < 2:
        return False
    rising = all(a <= b for a, b in zip(winners, winners[1:]))
    falling = all(a >= b for a, b in zip(winners, winners[1:]))
    if not (rising or falling):
        return False
    mode = max(set(winners), key=winners.count)
    return sum(1 for w in winners if w != mode) >= 2


def decide(per_position):
    '''Choose the frame offset from per-position scores.

    Args:
        per_position: {offset: score} dicts in time order.

    Returns:
        (offset or None, reason, detail dict).
    '''
    rows = [s for s in per_position if all(o in s for o in VOTE_OFFSETS)]
    detail = {"valid": len(rows)}
    if len(rows) < MIN_VALID_POSITIONS:
        return None, "frame_snap_too_few_positions", detail
    winners = [min(VOTE_OFFSETS, key=lambda o: s[o]) for s in rows]
    totals = {o: sum(s[o] for s in rows) for o in VOTE_OFFSETS}
    ranked = sorted(VOTE_OFFSETS, key=lambda o: totals[o])
    best, runner = ranked[0], ranked[1]
    margin = (totals[runner] - totals[best]) / totals[runner] if totals[runner] > 0 else 0.0
    votes = winners.count(best)
    detail.update(winners=winners, totals=totals, margin=margin, votes=votes)
    guard = {o: sum(s[o] for s in rows if o in s) / max(1, sum(1 for s in rows if o in s))
             for o in GUARD_OFFSETS}
    detail["guard"] = guard
    if is_drift(winners):
        return None, "frame_snap_drift", detail
    if votes * 2 <= len(rows) or margin <= MARGIN_MIN:
        return None, "frame_snap_no_consensus", detail
    best_mean = totals[best] / len(rows)
    beaten = [o for o, g in guard.items() if sum(1 for s in rows if o in s) and g < best_mean]
    if beaten:
        # The pictures put the delay two or more frames away: never applied,
        # but returned as a structured signal for the repair path.
        picture = min(beaten, key=lambda o: guard[o])
        full = [min((o for o in VOTE_OFFSETS + GUARD_OFFSETS if o in s), key=lambda o: s[o]) for s in rows]
        detail.update(picture_offset=picture, picture_votes=full.count(picture), picture_winners=full)
        return None, "audio_video_offset_disagree", detail
    return best, "frame_snap_chosen", detail


def positions(first_duration_s, second_duration_s, delay_s):
    '''Candidate instants of the first video, K slots over the common span.

    Edges (EDGE_FRACTION) are excluded. Each slot offers four instants, tried
    in turn until one holds a usable group, since animation often holds frames.
    '''
    low = max(0.0, -delay_s)
    high = min(first_duration_s, second_duration_s - delay_s)
    span = high - low
    if span <= 0:
        return []
    low, high = low + span * EDGE_FRACTION, high - span * EDGE_FRACTION
    slot = (high - low) / POSITIONS
    return [tuple(low + (i + f) * slot for f in (0.5, 0.25, 0.75, 0.125)) for i in range(POSITIONS)]


def measure(first_path, first_stream, second_path, second_stream, rate, delay_ms,
            first_duration_s, second_duration_s, budget_s=BUDGET_S):
    '''Score every position.

    Returns:
        (base_frames, rows of per-offset scores, skip notes, elapsed seconds).
    '''
    frame_s = float(1 / rate)
    delay_s = float(delay_ms) / 1000.0
    base_frames = int(round(Fraction(str(delay_ms)) / 1000 * rate))
    started = time.monotonic()
    rows, notes = [], []
    for instants in positions(first_duration_s, second_duration_s, delay_s):
        for t in instants:
            if time.monotonic() - started > budget_s:
                notes.append("budget")
                return base_frames, rows, notes, time.monotonic() - started
            # decode starts half a frame early so the frame AT t is the first one
            frames, times = decode_group(first_path, first_stream, t - frame_s / 2.0,
                                         REF_DECODE_FRAMES)
            start = cut_free_start(frames) if len(frames) >= GROUP_FRAMES else 0
            if start is None:
                notes.append(f"{t:.1f}:cut")
                continue
            ref_frames, ref_times = frames[start:start + GROUP_FRAMES], times[start:start + GROUP_FRAMES]
            usable, why = group_is_usable(ref_frames)
            if not usable:
                notes.append(f"{t:.1f}:{why}")
                continue
            other_start = ref_times[0] + (base_frames - REACH) * frame_s - frame_s / 2.0
            other_frames, other_times = decode_group(second_path, second_stream, other_start,
                                                     GROUP_FRAMES + 2 * REACH)
            ref_hashes = frame_hash.hash_frames(ref_frames, colour=True)
            other_hashes = frame_hash.hash_frames(other_frames, colour=True)
            scores, alignment = score_offsets(ref_hashes, ref_times, other_hashes, other_times,
                                              base_frames, frame_s)
            if not all(o in scores for o in VOTE_OFFSETS):
                notes.append(f"{t:.1f}:unpaired")
                continue
            if alignment.best is None:
                notes.append(f"{t:.1f}:{alignment.reason}")
                continue
            content = frame_hash.window_content_distance(ref_hashes, other_hashes,
                                                         alignment.start + alignment.best)
            if not content <= UNMATCHED_DISTANCE:
                notes.append(f"{t:.1f}:unmatched:{content:.3f}")
                break
            if not alignment.ok:
                notes.append(f"{t:.1f}:{alignment.reason}")
                continue
            scores["t"] = t
            rows.append(scores)
            break
    return base_frames, rows, notes, time.monotonic() - started


def _fmt_scores(rows):
    return "[" + "; ".join(f"{r['t']:.0f}s " + ",".join(f"{o:+d}={r[o]:.4f}" for o in sorted(k for k in r if k != "t"))
                           for r in rows) + "]"


def snap_for_merge(video_obj_1, video_obj_2, best_video_obj, delay):
    '''Return the delay to round to a frame, corrected by the pictures if needed.

    Args:
        video_obj_1, video_obj_2: the two videos; delay is from 1 to 2.
        best_video_obj: the video whose FrameRate defines the rounding grid.
        delay: audio delay in ms.

    Returns:
        Decimal: `delay` unchanged, or `(rounded + offset) * frame_ms` when the
        pictures choose a neighbouring frame. Never raises.
    '''
    delay = Decimal(delay)
    try:
        track_1, track_2 = video_obj_1.video, video_obj_2.video
        modes = (track_1.get("FrameRate_Mode"), track_2.get("FrameRate_Mode"))
        if modes != ("CFR", "CFR"):
            tools.log_always(f"frame_snap declined: not both CFR (modes={modes}); plain rounding\n")
            return delay
        (rate_1, src_1), (rate_2, src_2) = exact_rate(track_1), exact_rate(track_2)
        if not same_rate(rate_1, rate_2):
            tools.log_always(f"frame_snap declined: frame rates differ or unreadable "
                             f"({rate_1} from {src_1} vs {rate_2} from {src_2}); plain rounding\n")
            return delay
        dur_1, dur_2 = _duration_s(track_1), _duration_s(track_2)
        if not dur_1 or not dur_2:
            tools.log_always(f"frame_snap declined: duration unread ({dur_1}, {dur_2}); plain rounding\n")
            return delay
        base, rows, notes, elapsed = measure(video_obj_1.filePath, track_1["StreamOrder"],
                                             video_obj_2.filePath, track_2["StreamOrder"],
                                             rate_1, delay, dur_1, dur_2)
        offset, reason, detail = decide(rows)
        getcontext().prec = 10
        frame_ms = Decimal('1000.0') / Decimal(best_video_obj.video["FrameRate"])
        rounded = round(delay / frame_ms)
        if base != rounded:
            # Exact-rate and decimal-rate rounding disagree: offsets are not comparable.
            offset, reason = None, f"frame_snap_base_mismatch(scan {base})"
        chosen = offset if offset is not None else 0
        signal = None
        if reason == "audio_video_offset_disagree":
            picture_frames = rounded + detail["picture_offset"]
            signal = {"signal": "audio_video_offset_disagree", "fps": str(rate_1),
                      "audio_frames": int(rounded), "audio_ms": float(delay),
                      "picture_frames": int(picture_frames),
                      "picture_ms": float(picture_frames * frame_ms),
                      "votes": detail["picture_votes"], "valid": detail["valid"],
                      "positions_s": [round(r["t"], 1) for r in rows],
                      "picture_winners": detail["picture_winners"]}
            tools.log_always(f"frame_snap audio_video_offset_disagree: audio {signal['audio_frames']} frames "
                             f"({signal['audio_ms']:.2f} ms) vs picture {signal['picture_frames']} frames "
                             f"({signal['picture_ms']:.2f} ms), picture wins {signal['votes']}/{signal['valid']} "
                             f"positions at {signal['positions_s']} s; files {video_obj_1.filePath} | "
                             f"{video_obj_2.filePath}; rounding kept (audio)\n")
        with _SIGNALS_LOCK:
            if signal is None:
                _SIGNALS.pop((video_obj_1.filePath, video_obj_2.filePath), None)
            else:
                _SIGNALS[(video_obj_1.filePath, video_obj_2.filePath)] = signal
        final = Decimal((rounded + chosen) * frame_ms) if chosen else delay
        tools.log_always(
            f"frame_snap {reason}: fps={rate_1} ({src_1}) vs {rate_2} ({src_2}) audio_delay_ms={delay} rounded_frame={rounded} "
            f"winners={detail.get('winners')} votes={detail.get('votes')}/{detail.get('valid')} "
            f"totals={ {o: round(v, 3) for o, v in detail.get('totals', {}).items()} } "
            f"margin={round(detail.get('margin', 0.0), 3)} "
            f"guard(+-2)={ {o: round(v, 3) for o, v in detail.get('guard', {}).items()} } "
            f"distance={_fmt_scores(rows)} skipped={notes} chosen_offset={chosen:+d} "
            f"final_delay_ms={Decimal((rounded + chosen) * frame_ms)} elapsed_s={elapsed:.1f}\n")
        return final
    except Exception as exc:
        tools.log_always(f"frame_snap declined: {type(exc).__name__}: {exc}; plain rounding\n")
        return delay


def probe_disagreement(video_obj_1, video_obj_2, delay_ms, budget_s=BUDGET_S):
    '''Check whether the pictures sit more than one frame from this audio delay.

    Same measurement as `snap_for_merge`, for pairs that reach the repair path
    without going through `adjust_delay_to_frame`.

    Returns:
        (signal, reason): the `audio_video_offset_disagree` dict (also recorded
        for `disagreement()`) or None, and the decision or decline reason.
        Never raises and never changes the delay.
    '''
    try:
        track_1, track_2 = video_obj_1.video, video_obj_2.video
        modes = (track_1.get("FrameRate_Mode"), track_2.get("FrameRate_Mode"))
        if modes != ("CFR", "CFR"):
            return None, f"not_both_cfr{modes}"
        (rate_1, _), (rate_2, _) = exact_rate(track_1), exact_rate(track_2)
        if not same_rate(rate_1, rate_2):
            return None, f"rates_differ({rate_1},{rate_2})"
        dur_1, dur_2 = _duration_s(track_1), _duration_s(track_2)
        if not dur_1 or not dur_2:
            return None, "duration_unread"
        delay = Decimal(str(delay_ms))
        base, rows, notes, elapsed = measure(video_obj_1.filePath, track_1["StreamOrder"],
                                             video_obj_2.filePath, track_2["StreamOrder"],
                                             rate_1, delay, dur_1, dur_2, budget_s=budget_s)
        offset, reason, detail = decide(rows)
        frame_ms = Decimal(1000) * Decimal(rate_1.denominator) / Decimal(rate_1.numerator)
        signal = None
        if reason == "audio_video_offset_disagree":
            picture_frames = base + detail["picture_offset"]
            signal = {"signal": "audio_video_offset_disagree", "fps": str(rate_1),
                      "audio_frames": int(base), "audio_ms": float(delay),
                      "picture_frames": int(picture_frames),
                      "picture_ms": float(picture_frames * frame_ms),
                      "votes": detail["picture_votes"], "valid": detail["valid"],
                      "positions_s": [round(r["t"], 1) for r in rows],
                      "picture_winners": detail["picture_winners"], "source": "repair_probe"}
        with _SIGNALS_LOCK:
            if signal is None:
                _SIGNALS.pop((video_obj_1.filePath, video_obj_2.filePath), None)
            else:
                _SIGNALS[(video_obj_1.filePath, video_obj_2.filePath)] = signal
        tools.log_always(
            f"frame_snap repair_probe {reason}: fps={rate_1} audio_delay_ms={delay} "
            f"base_frame={base} winners={detail.get('winners')} "
            f"votes={detail.get('votes')}/{detail.get('valid')} "
            f"guard(+-2)={ {o: round(v, 3) for o, v in detail.get('guard', {}).items()} } "
            f"skipped={notes} elapsed_s={elapsed:.1f} files {video_obj_1.filePath} | "
            f"{video_obj_2.filePath}\n")
        return signal, reason
    except Exception as exc:                                             # noqa: BLE001
        tools.log_always(f"frame_snap repair_probe declined: {type(exc).__name__}: {exc}\n")
        return None, f"raised({type(exc).__name__})"
