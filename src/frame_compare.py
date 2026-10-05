"""Frame extraction and hashing of a time window, on an exact frame grid."""

from fractions import Fraction
import subprocess
from sys import stderr
import numpy as np
import frame_hash
import tools
import repair_log

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
        timeout = tools.decoder_timeout_for(dur_sec)
        tools.dev_log(f"frame_compare: _ffmpeg_raw_frames starting file={path} "
                      f"start_sec={start_sec} dur_sec={dur_sec} timeout_s={timeout}\n")
        try:
            with repair_log.announced("frame_compare", "ffmpeg", path, media_s=dur_sec) as call:
                done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      timeout=timeout)
                call["exit"] = done.returncode
        except subprocess.TimeoutExpired:
            raise tools.decoder_timeout("ffmpeg_raw_frames", timeout,
                                        f"file={path} start_sec={start_sec} dur_sec={dur_sec}")
        stdout, stderr_out, rc = done.stdout, done.stderr, done.returncode
        if rc not in (0,):
            if self.debug:
                stderr.write(f"[frame_compare] ffmpeg returned {rc}, partial data used\n")
        return stdout

    def _hash_frames(self, blob_bytes):
        """Grey and colour hashes of each frame in a raw rgb24 buffer (FrameHashes)."""
        frames = frame_hash.frames_from_raw(blob_bytes, self.width, self.height, 3)
        hashes = frame_hash.hash_frames(frames, colour=True)
        if self.debug:
            stderr.write(f"[frame_compare] hashed frames: {len(hashes)}\n")
        return hashes


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
            stdout, stderror, exit_code = tools.launch_cmdExt_with_timeout_reload(
                cmd, max_restart=3, timeout=60)
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
    if isinstance(values, frame_hash.FrameHashes):
        return values[np.asarray(picks, dtype=np.int64)], None
    return [values[j] for j in picks], None


def _extract_hashes(comparer, path, start_s, dur_s):
    """Return (base frame index, FrameHashes on the comparer's grid) for a window.

    The hashes are empty when the window is empty or the file's rate is unknown.
    """
    start_s = max(0.0, start_s)
    if dur_s <= 0:
        return comparer._frame_index(start_s), frame_hash.FrameHashes.empty(colour=True)
    blob = comparer._ffmpeg_raw_frames(path, start_s, dur_s)
    hashes = comparer._hash_frames(blob)
    base = comparer._frame_index(start_s)
    on_grid, reason = _on_comparer_grid(comparer, path, hashes)
    if on_grid is None:
        # Never guess the grid: an empty series makes every caller decline.
        tools.dev_log(f"frame_compare: _extract_hashes declining "
                      f"file={path} reason=native_rate_unmeasured:{reason} "
                      f"start_sec={start_s} decoded_frames={len(hashes)}\n")
        return base, frame_hash.FrameHashes.empty(colour=True)
    return base, on_grid
