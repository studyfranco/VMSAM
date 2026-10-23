"""Frame extraction, comparison and alignment of a time window, on an exact frame grid.

Comparisons, distances and alignment of frames -- the only place any of these three live, per
caller (`video_offset_plan`, `scene_anchor`, `frame_snap`, `merge_video_*`...) comparing or
aligning frames does it through the names here, never by calling `frame_hash` directly.
`frame_hash` is the only place that computes the underlying imagehash pHash; this module imports
it for that (`hash_frames`, `frame_hashes`, `frames_from_raw`, `grey_frames`) and for nothing
else -- `frame_hash` itself imports nothing from here, so there is no cycle.

pHash is unstable on a frame that is almost entirely black or almost entirely white: most of its
DCT coefficients sit at the median, so two copies of the same frame can read far apart.
`frame_distance` is the one frame-to-frame comparison: on such a frame it decides from a small
grey thumbnail instead (`pixel_distance`), stored only for frames near that case.
"""

from fractions import Fraction
from sys import stderr
import numpy as np
import frame_hash
import progress_watchdog
import tools
import repair_log

# A span where neither side is flagged flat is trusted only below this luma standard deviation
# (0-255): below it, a hash's bits are noise, so the frame never takes part in an alignment.
FLAT_STD = 3.0

# Weight of the mean Cb/Cr distance in `content_distance`.
COLOUR_WEIGHT = 0.5

# An alignment is kept only when its minimum beats every lag two or more frames away by this
# fraction of the hash bits, to reject ambiguous matches while rarely rejecting a correct one.
ALIGN_MARGIN = 0.01

# Fewest usable (non-flat) reference frames an alignment accepts.
ALIGN_MIN_FRAMES = 6

# Same content: mean `content_distance` of a window of 6 or more frames at or below this.
SAME_CONTENT_MAX = 0.37

# Same picture: grey `distance` of one frame pair at or below this. Chosen to keep most
# same-frame pairs while still separating them from a neighbouring frame or another scene.
SAME_FRAME_MAX = 0.07

# A luma pixel is "extreme" at or below `frame_hash.DARK_LEVEL` or at or above
# `frame_hash.LIGHT_LEVEL`; from this fraction a frame is near-uniform and `frame_distance`
# decides by pixels instead, since pHash reads unreliably far apart on near-uniform frames that
# are actually identical.
UNIFORM_FRACTION = 0.75
# Luma std floor below which the z-score would otherwise amplify a thumbnail's own encode noise
# on a very flat (e.g. near-black) frame.
PIXEL_STD_FLOOR = 12.0
# Same picture by pixels: `pixel_distance` at or below this. Keeps identical near-uniform frames
# apart from frames that differ, including a redrawn shot of the same scene.
SAME_PIXEL_MAX = 0.005


def hamming(a, b):
    '''Differing bits between packed hashes (words on the last axis); broadcasts.

    frame_hashes arguments compare their grey hashes.
    '''
    a = a.grey if isinstance(a, frame_hash.frame_hashes) else a
    b = b.grey if isinstance(b, frame_hash.frame_hashes) else b
    return np.bitwise_count(np.bitwise_xor(a, b)).sum(axis=-1, dtype=np.int64)


def distance(a, b):
    '''Per-frame raw grey pHash distance of two equal-length frame_hashes, as a fraction of the
    bits. Comparisons go through `frame_distance`.'''
    return hamming(a.grey, b.grey) / a.bits


def pixel_distance(a, b):
    '''(1 - Pearson correlation) / 2 of grey thumbnails (..., h, w), each z-scored by its own
    mean and std (std floored at PIXEL_STD_FLOOR): 0 same, about 0.5 unrelated, 1 inverted.'''
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    axes = (-2, -1)
    za = (a - a.mean(axis=axes, keepdims=True)) / np.maximum(a.std(axis=axes, keepdims=True),
                                                             PIXEL_STD_FLOOR)
    zb = (b - b.mean(axis=axes, keepdims=True)) / np.maximum(b.std(axis=axes, keepdims=True),
                                                             PIXEL_STD_FLOOR)
    return np.square(za - zb).mean(axis=axes) / 4.0


def _pixel_rows(a_extreme, a_index, b_extreme, b_index):
    '''Pairs decided by pixels: either side near-uniform, both carrying a thumbnail.'''
    return (((a_extreme >= UNIFORM_FRACTION) | (b_extreme >= UNIFORM_FRACTION))
            & (a_index >= 0) & (b_index >= 0))


def _pixel_as_bits(pixel):
    '''A `pixel_distance` on the pHash bit-fraction scale, piecewise linear: 0 stays 0,
    SAME_PIXEL_MAX lands on SAME_FRAME_MAX, 0.5 (unrelated) on 0.5, capped there.'''
    pixel = np.asarray(pixel, dtype=np.float64)
    low = pixel * (SAME_FRAME_MAX / SAME_PIXEL_MAX)
    high = SAME_FRAME_MAX + (pixel - SAME_PIXEL_MAX) * ((0.5 - SAME_FRAME_MAX)
                                                        / (0.5 - SAME_PIXEL_MAX))
    return np.minimum(np.where(pixel <= SAME_PIXEL_MAX, low, high), 0.5)


def frame_distance(a, b):
    '''Per-frame distance of two equal-length frame_hashes on the grey pHash scale.

    The grey pHash fraction of differing bits, except where either frame is near-uniform
    (`extreme` >= UNIFORM_FRACTION) and both carry a thumbnail: there `pixel_distance`
    decides, rescaled by `_pixel_as_bits` so every pHash threshold keeps its meaning.
    '''
    dist = hamming(a.grey, b.grey) / a.bits
    rows = _pixel_rows(a.extreme, a.thumb_index, b.extreme, b.thumb_index)
    if rows.any():
        dist = np.asarray(dist, dtype=np.float64)
        pixel = pixel_distance(a.thumbs[a.thumb_index[rows]], b.thumbs[b.thumb_index[rows]])
        dist[rows] = _pixel_as_bits(pixel)
    return dist


def same_picture(a, b):
    '''Per-frame verdicts (bool array) that two equal-length frame_hashes show the same picture.'''
    return frame_distance(a, b) <= SAME_FRAME_MAX


def content_distance(a, b):
    '''Per-frame content distance of two equal-length frame_hashes, as a fraction of the bits.

    (grey + COLOUR_WEIGHT x mean(Cb, Cr)) / (1 + COLOUR_WEIGHT), each part as a fraction of
    its own bits; `frame_distance` alone where it decides by pixels. Both sides need colour.
    '''
    if a.colour is None or b.colour is None:
        raise ValueError("content_distance needs colour hashes on both sides")
    grey = hamming(a.grey, b.grey) / a.bits
    chroma = hamming(a.colour, b.colour).mean(axis=-1) / (a.colour.shape[-1] * 64)
    content = (grey + COLOUR_WEIGHT * chroma) / (1.0 + COLOUR_WEIGHT)
    rows = _pixel_rows(a.extreme, a.thumb_index, b.extreme, b.thumb_index)
    if rows.any():
        # chroma hashes of a near-uniform frame are as unstable as its grey one
        content = np.asarray(content, dtype=np.float64)
        content[rows] = frame_distance(a[np.nonzero(rows)[0]], b[np.nonzero(rows)[0]])
    return content


def window_content_distance(ref, other, offset=0):
    '''Mean `content_distance` of `ref` against `other[offset : offset + len(ref)]`.

    Flat reference frames are left out. NaN when the window leaves `other` or no frame
    is usable.
    '''
    if offset < 0 or offset + len(ref) > len(other):
        return float("nan")
    rows = np.nonzero(ref.std >= FLAT_STD)[0]
    if len(rows) == 0:
        return float("nan")
    return float(np.mean(content_distance(ref[rows], other[rows + offset])))


# -- alignment -------------------------------------------------------------------------------

class alignment:
    '''Result of `align`.

    Attributes:
        lag: the chosen lag, or None when refused.
        reason: "ok", or why it was refused: "short", "flat", "unreadable", "tie",
            "no_rival", "ambiguous".
        curve: {lag: mean grey distance} over the readable lags.
        best: the lag of the minimum (also set when refused, when there is one).
        distance: the minimum mean distance.
        margin: second-best distance two or more lags away minus the minimum.
        rival: the lag of that second best.
        frames: reference frames used (non-flat).
        start: the `start` the lags are counted from.
    '''

    def __init__(self, reason, curve=None, best=None, margin=None, rival=None, frames=0,
                 start=0):
        self.reason = reason
        self.start = start
        self.curve = curve or {}
        self.best = best
        self.lag = best if reason == "ok" else None
        self.distance = None if best is None else self.curve[best]
        self.margin = margin
        self.rival = rival
        self.frames = frames

    @property
    def ok(self):
        return self.reason == "ok"

    def __repr__(self):
        margin = "None" if self.margin is None else f"{self.margin:.4f}"
        return (f"alignment(lag={self.lag} reason={self.reason} best={self.best} "
                f"margin={margin} rival={self.rival} frames={self.frames})")


def distance_curve(ref, other, lags, start=0):
    '''Mean `frame_distance` of `ref` against `other[start + lag :]`, for each lag.

    Flat reference frames (std < FLAT_STD) are left out, the same rows for every lag.

    Returns:
        (curve, used): curve is a float array aligned with `lags`, NaN where the window
        leaves `other`; used is the number of reference frames compared.
    '''
    lags = np.asarray(list(lags), dtype=np.int64)
    n = len(ref)
    rows = np.nonzero(ref.std >= FLAT_STD)[0]
    curve = np.full(len(lags), np.nan)
    if len(rows) == 0 or len(lags) == 0:
        return curve, len(rows)
    first = start + lags
    readable = (first >= 0) & (first + n <= len(other))
    if readable.any():
        idx = first[readable][:, None] + rows[None, :]
        dist = hamming(other.grey[idx], ref.grey[rows][None, :, :]) / ref.bits
        ref_index = np.broadcast_to(ref.thumb_index[rows][None, :], idx.shape)
        pixel = _pixel_rows(np.broadcast_to(ref.extreme[rows][None, :], idx.shape), ref_index,
                            other.extreme[idx], other.thumb_index[idx])
        if pixel.any():
            dist[pixel] = _pixel_as_bits(pixel_distance(ref.thumbs[ref_index[pixel]],
                                                        other.thumbs[other.thumb_index[idx][pixel]]))
        curve[readable] = dist.mean(axis=1)
    return curve, len(rows)


def align(ref, other, lags, start=0, margin=ALIGN_MARGIN, min_frames=ALIGN_MIN_FRAMES):
    '''Find the lag at which `other` shows the frames of `ref`.

    Lag `l` compares ref[i] with other[start + l + i]. The minimum of the mean-distance
    curve is kept only when it is unique and beats every lag two or more frames away by
    `margin` (fraction of the bits); adjacent lags are not gated, the argmin decides.

    Returns:
        alignment.
    '''
    lags = sorted({int(lag) for lag in lags})
    if len(ref) < min_frames:
        return alignment("short", frames=len(ref), start=start)
    values, used = distance_curve(ref, other, lags, start)
    if used < min_frames:
        return alignment("flat", frames=used, start=start)
    curve = {lag: float(v) for lag, v in zip(lags, values) if np.isfinite(v)}
    if not curve:
        return alignment("unreadable", frames=used, start=start)
    best = min(curve, key=lambda lag: (curve[lag], lag))
    low = curve[best]
    if sum(1 for v in curve.values() if v == low) > 1:
        return alignment("tie", curve, best, frames=used, start=start)
    rivals = {lag: v for lag, v in curve.items() if abs(lag - best) >= 2}
    if not rivals:
        return alignment("no_rival", curve, best, frames=used, start=start)
    rival = min(rivals, key=lambda lag: (rivals[lag], lag))
    gap = rivals[rival] - low
    return alignment("ok" if gap >= margin else "ambiguous", curve, best, gap, rival, used,
                     start)


class FrameComparer:
    """Compare frames of two videos inside a time window.

    Frames are decoded as small RGB pictures at the native rate and hashed by
    `frame_hash` (grey and colour). The frame rate must be an exact rational
    (`fps_num`/`fps_den`); every frame index is on that grid.

    Args:
        ref_path, tgt_path: the two files.
        start_sec, end_sec: the time window.
        fps_num, fps_den: exact frame rate of the comparison grid.
        crop_filters: optional {path: "crop=w:h:x:y"} applied before scaling.
        time_scales: optional {path: Fraction r}, r = speed relative to the reference.

    Attributes:
        mask_box: optional (y0, y1, x0, x1) region of the hashed frame blanked on both
            files before hashing (a static overlay only one file carries); None by default.

    Raises:
        ValueError: on a non-positive rate or time scale.
    """

    def __init__(self, ref_path, tgt_path, start_sec, end_sec,
                 fps_num, fps_den,
                 band_width_sec=2.0, max_search_sec=5.0, debug=False,
                 scene_threshold=0.30, crop_filters=None, time_scales=None):
        self.ref_path = ref_path
        self.tgt_path = tgt_path
        self.start_sec = float(start_sec)
        self.end_sec = float(end_sec)

        fps_num = int(fps_num)
        fps_den = int(fps_den)
        if fps_num <= 0 or fps_den <= 0:
            # No default rate: a non-positive rate means it was not measured.
            raise ValueError(
                f"FrameComparer requires an exact positive frame-rate rational, "
                f"got fps_num={fps_num} fps_den={fps_den}")
        self.fps_num = fps_num
        self.fps_den = fps_den
        self.fps_frac = Fraction(fps_num, fps_den)
        # Float only for ffmpeg arguments and display.
        self.fps = float(self.fps_frac)

        self.band_width = max(1, self._round_frac(Fraction(band_width_sec).limit_denominator(10**6) * self.fps_frac))
        self.max_search_frames = max(8, self._round_frac(Fraction(max_search_sec).limit_denominator(10**6) * self.fps_frac))
        self.width = frame_hash.FRAME_WIDTH
        self.height = frame_hash.FRAME_HEIGHT
        self.debug = debug
        self.scene_threshold = float(scene_threshold)
        self.crop_filters = dict(crop_filters) if crop_filters else {}
        # Seconds given for a scaled path are reference-equivalent: seek and
        # duration are divided by `r`.
        self.time_scales = {}
        for scaled_path, scale in (time_scales or {}).items():
            if scale is None:
                continue
            scale = Fraction(scale)
            if scale <= 0:
                raise ValueError(f"FrameComparer time scale must be positive, "
                                 f"got {scale} for {scaled_path}")
            if scale != 1:
                self.time_scales[scaled_path] = scale
        self.mask_box = None

    @staticmethod
    def _round_frac(frac: Fraction) -> int:
        """Round half up on the exact rational (`int()` would truncate)."""
        return int(frac + Fraction(1, 2))

    def _frame_index(self, seconds: float) -> int:
        """Absolute frame index at `seconds` on this object's exact grid."""
        return self._round_frac(Fraction(seconds).limit_denominator(10**9) * self.fps_frac)

    def _ffmpeg_raw_frames(self, path, start_sec, dur_sec):
        """Decode a window as raw small RGB frames at the native rate (bytes)."""
        # No `fps=` filter: it would duplicate or drop frames.
        ffmpeg = tools.software["ffmpeg"]
        w, h = self.width, self.height
        # Crop before scaling so black bars do not contaminate the hash.
        crop = self.crop_filters.get(path)
        scale = f"scale={w}:{h}:flags=area,format=rgb24"
        vf = scale if not crop else f"{crop},{scale}"
        scale = self.time_scales.get(path)
        if scale is not None:
            start_sec = float(Fraction(start_sec).limit_denominator(10**9) / scale)
            dur_sec = float(Fraction(dur_sec).limit_denominator(10**9) / scale)
        # A window of no length is never decoded: `-t 0.0` means "no limit" to
        # ffmpeg, so a clamped negative window would decode the whole file.
        if dur_sec <= 0:
            tools.dev_log(f"frame_compare: _ffmpeg_raw_frames window_empty file={path} "
                          f"start_sec={start_sec} dur_sec={dur_sec} -- not decoded\n")
            return b""
        cmd = [
            ffmpeg, "-v", "error", "-nostdin",
            "-ss", f"{start_sec}",
            "-t", f"{dur_sec}",
            "-i", path,
            "-vf", vf,
            # passthrough: the default constant-rate output may duplicate a frame
            "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
        ]
        tools.dev_log(f"frame_compare: _ffmpeg_raw_frames starting file={path} "
                      f"start_sec={start_sec} dur_sec={dur_sec}\n")
        with repair_log.announced("frame_compare", "ffmpeg", path, media_s=dur_sec) as call:
            done = progress_watchdog.run(cmd, "ffmpeg_raw_frames")
            call["exit"] = done.returncode
        stdout, stderr_out, rc = done.stdout, done.stderr, done.returncode
        if rc not in (0,):
            if self.debug:
                stderr.write(f"[frame_compare] ffmpeg returned {rc}, partial data used\n")
        return stdout

    def _hash_frames(self, blob_bytes):
        """Grey and colour hashes of each frame in a raw rgb24 buffer (frame_hashes)."""
        frames = frame_hash.frames_from_raw(blob_bytes, self.width, self.height, 3)
        if self.mask_box is not None:
            frames = mask_frames(frames, self.mask_box)
        hashes = frame_hash.hash_frames(frames, colour=True)
        if self.debug:
            stderr.write(f"[frame_compare] hashed frames: {len(hashes)}\n")
        return hashes


def mask_frames(frames, box):
    '''Return a copy of (n, h, w[, 3]) frames with the (y0, y1, x0, x1) region set to zero.'''
    y0, y1, x0, x1 = box
    out = np.array(frames, copy=True)
    out[:, y0:y1, x0:x1] = 0
    return out


def _nominal_shift_frames(offset_ms, fps_num, fps_den):
    frame_ms = 1000.0 * fps_den / fps_num
    return int(round(offset_ms / frame_ms))


# Native decode rate vs label grid: `_ffmpeg_raw_frames` decodes at the file's
# own rate, but indices are on the comparer's grid. When the rates differ, each
# decoded series is re-indexed onto the grid (`_on_comparer_grid`) so element
# `k` is the frame playing `k` grid frames into the window.

# Native rate per path; only successes are cached, so a transient ffprobe
# failure is retried.
_NATIVE_RATE_CACHE = {}


def parse_positive_rate(value):
    '''Parse a frame rate to an exact positive Fraction, or None.

    Shared rate normaliser (also used by `scene_anchor` and
    `merge_video_chimeric`). Blank, non-positive, non-finite or unparseable
    values return None.
    '''
    if value is None:
        return None
    try:
        rate = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return rate if rate > 0 else None


def _native_frame_rate(path):
    '''Native decode rate of the file from ffprobe `r_frame_rate`.

    Returns `(Fraction, None)` on success, `(None, reason)` otherwise.
    '''
    cached = _NATIVE_RATE_CACHE.get(path)
    if cached is not None:
        return cached, None
    try:
        cmd = [tools.software["ffprobe"], "-v", "error",
               "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
               "-of", "default=noprint_wrappers=1:nokey=1", path]
    except KeyError:
        return None, "ffprobe_not_configured"
    tools.dev_log(f"frame_compare: _native_frame_rate calling ffprobe "
                  f"file={path}\n")
    try:
        with repair_log.announced("frame_compare", "ffprobe", path) as call:
            stdout, stderror, exit_code = progress_watchdog.run_checked(cmd, "ffprobe_frame_rate")
            call["exit"] = exit_code
    except Exception as exc:
        return None, f"ffprobe_raised:{type(exc).__name__}"
    if exit_code != 0:
        return None, f"ffprobe_exit:{exit_code}"
    lines = stdout.decode("utf-8", "replace").strip().splitlines()
    raw = lines[0].strip() if lines else ""
    rate = parse_positive_rate(raw)
    if rate is None:
        return None, f"unparseable_r_frame_rate:{raw!r}"
    _NATIVE_RATE_CACHE[path] = rate
    return rate, None


def _on_comparer_grid(comparer, path, values):
    '''Re-index a natively decoded per-frame series onto the comparer's grid.

    Element `k` is taken from native element `round(k * native_rate / grid_rate)`
    on window-relative indices (an instant-based form would add `base`'s own
    rounding residue). Returns `(values_on_grid, None)`, or `(None, reason)`
    when the file's rate cannot be measured.
    '''
    if not len(values):
        return values, None
    native_rate, reason = _native_frame_rate(path)
    if native_rate is None:
        return None, reason
    grid_rate = comparer.fps_frac
    # A speed-changed path uses its corrected grid `grid_rate * r`.
    scale = getattr(comparer, "time_scales", {}).get(path)
    if scale is not None:
        grid_rate = grid_rate * scale
    if native_rate == grid_rate:
        return values, None
    ratio = native_rate / grid_rate
    n_native = len(values)
    picks = []
    k = 0
    while True:
        j = FrameComparer._round_frac(Fraction(k) * ratio)
        if j >= n_native:
            break
        picks.append(j)
        k += 1
    if isinstance(values, frame_hash.frame_hashes):
        return values[np.asarray(picks, dtype=np.int64)], None
    return [values[j] for j in picks], None


def _extract_hashes(comparer, path, start_s, dur_s):
    """Return (base frame index, frame_hashes on the comparer's grid) for a window.

    The hashes are empty when the window is empty or the file's rate is unknown.
    """
    start_s = max(0.0, start_s)
    if dur_s <= 0:
        return comparer._frame_index(start_s), frame_hash.frame_hashes.empty(colour=True)
    blob = comparer._ffmpeg_raw_frames(path, start_s, dur_s)
    hashes = comparer._hash_frames(blob)
    base = comparer._frame_index(start_s)
    on_grid, reason = _on_comparer_grid(comparer, path, hashes)
    if on_grid is None:
        # Never guess the grid: an empty series makes every caller decline.
        tools.dev_log(f"frame_compare: _extract_hashes declining "
                      f"file={path} reason=native_rate_unmeasured:{reason} "
                      f"start_sec={start_s} decoded_frames={len(hashes)}\n")
        return base, frame_hash.frame_hashes.empty(colour=True)
    return base, on_grid
