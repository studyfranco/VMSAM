'''frame_snap.py -- the frame the pictures agree on, for `adjust_delay_to_frame`.

At the end of the second correlation step the audio delay is snapped to a whole
video frame. Until now that snap was a pure rounding, `round(delay / frame_ms)`:
when the audio delay sits near a half frame the rounding can land one frame off,
and nothing looked at the pictures to say so (owner request, 2026-09-25).

This module only ever moves the rounded frame by -1, 0 or +1, and only when both
videos are constant-frame-rate at the same rate (within RATE_TOLERANCE). It looks at the
pictures at K positions spread over the span both files cover: a group of G
consecutive frames of the first video at `t`, and G + 4 frames of the second
around `t + delay`, both decoded once (one ffmpeg call per file per position),
downscaled to 64x36 grey. Each candidate offset (the rounded frame plus -2..+2)
is one sliding G-frame window of the second group, scored by the mean pHash
Hamming distance (64-bit DCT hash, owner's choice 2026-09-25) to the first
group. -1/0/+1 vote; +-2 is only a guard, logged and never selected.

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
keeps today's rounding, bit for bit, and says why.

Delay convention (the one `get_good_frame` and both call sites use): the second
video shows at `t + delay` what the first shows at `t`, delay in milliseconds.

The only caller is `mergeVideo.compare_video.adjust_delay_to_frame` (open zone,
WRITE_ZONES.MD section 2), through `snap_for_merge`.
'''

from decimal import Decimal, getcontext
from fractions import Fraction
import subprocess
import re
import threading
import time

import numpy as np

import tools

# -- tunables, each one measured or justified in the commit that set it ------
POSITIONS = 8            # K: positions spread over the common span
GROUP_FRAMES = 12        # G: consecutive frames compared at each position
EDGE_FRACTION = 0.05     # the first/last 5 % of the common span are never used
VOTE_OFFSETS = (-1, 0, 1)
GUARD_OFFSETS = (-2, 2)  # logged, never selected
REACH = 2                # frames of the second video decoded on each side
WIDTH, HEIGHT = 64, 36
THREADS = 3
# A group whose mean inter-frame difference (grey levels, 0-255, on the 64x36
# picture) is below this cannot tell one frame from its neighbour. Measured on
# the owner's lots: at 0.38 the right frame still wins (Hamming 0.38 against
# 1.9-2.1 at +-1), at 0.16-0.25 it no longer does (0.9 against 1.2).
STATIC_MEAN_DIFF = 0.3
# A frame darker than this mean AND flatter than BLACK_STD is black.
BLACK_MEAN = 18.0
BLACK_STD = 6.0
# A position where even the best of the five windows is this far (Hamming
# bits out of 64) matches nothing: the delay does not hold there (a step, a
# recap cut), so it votes for nothing. Unrelated pictures sit near 32; the
# same picture re-encoded, 0-3 (measured, commit message).
UNMATCHED_HAMMING = 12.0
# Required relative margin of the winner's total over the runner-up's.
MARGIN_MIN = 0.10
MIN_VALID_POSITIONS = 5
# Two rates are "the same" when |a/b - 1| <= this. 18965/791 (a container that
# stores ms timestamps, ToonsHub) against 24000/1001 is 1.8e-6 apart: the same
# 23.976 stream. 1001/1000 rate pairs (25 vs 25000/1001, 24 vs 24000/1001) are
# 1e-3 apart and stay different; that is the rate arm's case, not this one.
RATE_TOLERANCE = Fraction(1, 10000)
BUDGET_S = 20.0          # wall-clock budget for one pair

# Addendum 27.7: the last `audio_video_offset_disagree` signal per pair of
# files, {(first_path, second_path): dict}. Written by `snap_for_merge`, read
# by the repair seam through `disagreement(first_path, second_path)`.
_SIGNALS = {}
_SIGNALS_LOCK = threading.Lock()


def disagreement(first_path, second_path):
    '''The `audio_video_offset_disagree` signal recorded for this pair (as a
    dict: audio_frames/audio_ms, picture_frames/picture_ms, votes, valid,
    positions, fps), or None when the last snap on it found no such
    disagreement.'''
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
    '''The stream's frame rate as an exact Fraction, or (None, reason).

    MediaInfo's FrameRate_Num/FrameRate_Den first, then ffprobe's
    r_frame_rate (carried on the track as `ffprobe`). Never the rounded
    decimal `FrameRate` ("23.976"): two files "at 23.976" may be at
    24000/1001 and 23976/1000, which drift a frame every ~17 minutes.'''
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
    '''`n_frames` frames from `start_s`, 64x36 grey, and each frame's time.

    One ffmpeg call, `-ss` before `-i` (keyframe seek, then an exact decode
    to `start_s`). The time of every frame is READ from showinfo, not
    assumed: `start_s + pts_time`, so a seek that lands one frame later than
    asked moves the time, never the comparison.'''
    start_s = max(0.0, float(start_s))
    cmd = [tools.software["ffmpeg"], "-hide_banner", "-nostdin", "-threads", str(THREADS),
           "-ss", f"{start_s:.6f}", "-i", path, "-map", f"0:{stream_order}",
           "-frames:v", str(n_frames), "-an", "-sn",
           "-vf", f"showinfo,scale={WIDTH}:{HEIGHT}:flags=area,format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    timeout = tools.decoder_timeout_for(n_frames / 10.0)
    try:
        done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise tools.decoder_timeout("frame_snap_decode", timeout, f"file={path} start_s={start_s}")
    size = WIDTH * HEIGHT
    count = len(done.stdout) // size
    frames = np.frombuffer(done.stdout[:count * size], dtype=np.uint8).reshape(count, HEIGHT, WIDTH).astype(np.float32)
    pts = [float(x) for x in _PTS_RE.findall(done.stderr.decode("utf-8", "replace"))]
    count = min(count, len(pts))
    times = np.array([start_s + p for p in pts[:count]], dtype=np.float64)
    return frames[:count], times


# -- distance ----------------------------------------------------------------

def _dct_matrix(n):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0, :] /= np.sqrt(2.0)
    return m


_DCT_H = _dct_matrix(HEIGHT)
_DCT_W = _dct_matrix(WIDTH)


def phash(frame):
    '''64-bit perceptual hash of one 64x36 grey frame: 2-D DCT, the 8x8
    lowest frequencies, each bit = coefficient above their median (DC left
    out of the median). The same construction as `frame_compare`'s pHash,
    on this module's decode size.'''
    coeffs = (_DCT_H @ frame @ _DCT_W.T)[:8, :8].flatten()
    median = np.median(coeffs[1:])
    bits = coeffs > median
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a, b):
    return (a ^ b).bit_count()


def group_is_usable(frames):
    '''(True, "") or (False, reason) for a group of the first video.'''
    if len(frames) < GROUP_FRAMES:
        return False, f"short:{len(frames)}"
    means = frames.reshape(len(frames), -1).mean(axis=1)
    stds = frames.reshape(len(frames), -1).std(axis=1)
    if np.sum((means < BLACK_MEAN) & (stds < BLACK_STD)) > len(frames) // 2:
        return False, "black"
    motion = float(np.mean(np.abs(np.diff(frames, axis=0))))
    if motion < STATIC_MEAN_DIFF:
        return False, f"static:{motion:.2f}"
    return True, ""


def score_offsets(ref_hashes, ref_times, other_hashes, other_times, base_frames, frame_s):
    '''{offset: mean Hamming distance} at one position, for -2..+2.

    The second video's group is G + 2*REACH consecutive frames decoded from
    `t + (base - REACH) frames`; offset `o` is the G-frame window starting at
    index `anchor + REACH + o` -- windows [0..11] (-2), [1..12] (-1),
    [2..13] (0), [3..14] (+1), [4..15] (+2). `anchor` is READ from the
    decoded frame times (the frame nearest to where window -2 must start),
    never assumed, so a seek that lands one frame late shifts the anchor and
    not the answer; an offset whose window is not whole is not scored.'''
    scores = {}
    if len(other_times) == 0 or len(ref_times) == 0:
        return scores
    target = ref_times[0] + (base_frames - REACH) * frame_s
    anchor = int(np.argmin(np.abs(other_times - target)))
    if abs(other_times[anchor] - target) > frame_s / 2.0:
        return scores
    g = len(ref_hashes)
    for offset in VOTE_OFFSETS + GUARD_OFFSETS:
        first = anchor + REACH + offset
        window = other_hashes[first:first + g]
        if first < 0 or len(window) < g:
            continue
        scores[offset] = float(np.mean([hamming(x, y) for x, y in zip(ref_hashes, window)]))
    return scores


# -- decision ----------------------------------------------------------------

def is_drift(winners):
    '''Winners in time order that change monotonically, with at least two
    positions away from the most common value: not a constant delay.'''
    if len(set(winners)) < 2:
        return False
    rising = all(a <= b for a, b in zip(winners, winners[1:]))
    falling = all(a >= b for a, b in zip(winners, winners[1:]))
    if not (rising or falling):
        return False
    mode = max(set(winners), key=winners.count)
    return sum(1 for w in winners if w != mode) >= 2


def decide(per_position):
    '''`per_position`: list, in time order, of {offset: score} dicts.

    Returns (offset or None, reason, detail dict).'''
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
        # The picture says the audio delay is two frames or more away (Addendum
        # 27.7). This module never moves that far -- the rounding stays -- but
        # the disagreement is a finding about the pair, not noise: it is
        # returned as a structured signal for the repair seam to route.
        picture = min(beaten, key=lambda o: guard[o])
        full = [min((o for o in VOTE_OFFSETS + GUARD_OFFSETS if o in s), key=lambda o: s[o]) for s in rows]
        detail.update(picture_offset=picture, picture_votes=full.count(picture), picture_winners=full)
        return None, "audio_video_offset_disagree", detail
    return best, "frame_snap_chosen", detail


def positions(first_duration_s, second_duration_s, delay_s):
    '''K instants of the first video, evenly spread over the span both files
    cover, first/last EDGE_FRACTION excluded; each slot is tried at four
    instants in turn until one holds a usable (moving, not black) group --
    held-frame animation leaves most half-second groups static.'''
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
    '''Score every position. Returns (base_frames, rows, notes, elapsed_s).'''
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
            ref_frames, ref_times = decode_group(first_path, first_stream, t - frame_s / 2.0, GROUP_FRAMES)
            usable, why = group_is_usable(ref_frames)
            if not usable:
                notes.append(f"{t:.1f}:{why}")
                continue
            other_start = t + (base_frames - REACH) * frame_s - frame_s / 2.0
            other_frames, other_times = decode_group(second_path, second_stream, other_start,
                                                     GROUP_FRAMES + 2 * REACH)
            scores = score_offsets([phash(f) for f in ref_frames], ref_times,
                                   [phash(f) for f in other_frames], other_times, base_frames, frame_s)
            if not all(o in scores for o in VOTE_OFFSETS):
                notes.append(f"{t:.1f}:unpaired")
                continue
            if min(scores.values()) > UNMATCHED_HAMMING:
                notes.append(f"{t:.1f}:unmatched:{min(scores.values()):.1f}")
                break
            scores["t"] = t
            rows.append(scores)
            break
    return base_frames, rows, notes, time.monotonic() - started


def _fmt_scores(rows):
    return "[" + "; ".join(f"{r['t']:.0f}s " + ",".join(f"{o:+d}={r[o]:.3f}" for o in sorted(k for k in r if k != "t"))
                           for r in rows) + "]"


def snap_for_merge(video_obj_1, video_obj_2, best_video_obj, delay):
    '''The call `adjust_delay_to_frame` makes before its rounding.

    Returns the delay to round: `delay` itself (today's behaviour, unchanged)
    unless the pictures chose a neighbour of the rounded frame, in which case
    exactly `(rounded + offset) * frame_ms`, so the rounding that follows
    lands on it. `frame_ms` is computed the way the rounding computes it.'''
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
            # The scan's base frame (exact rational) and the rounding's (decimal
            # FrameRate) disagree: the offsets were measured from another frame.
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
            f"hamming={_fmt_scores(rows)} skipped={notes} chosen_offset={chosen:+d} "
            f"final_delay_ms={Decimal((rounded + chosen) * frame_ms)} elapsed_s={elapsed:.1f}\n")
        return final
    except Exception as exc:
        tools.log_always(f"frame_snap declined: {type(exc).__name__}: {exc}; plain rounding\n")
        return delay
