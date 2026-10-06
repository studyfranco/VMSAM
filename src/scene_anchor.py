"""Frame-accurate refinement and validation of a bracket (a cut between master and candidate).

Scene cuts from PySceneDetect seed anchor candidates; pHash comparison validates them. This is
the authoritative boundary method for merge_video_chimeric; when it declines, the caller
declines too (frame_compare's locators are cross-checks only).

Frames come from frame_compare's `_extract_hashes` (native-rate extraction); hashes,
distances and alignment from `frame_hash`.

An anchor window lies entirely on one side of its seed and never holds a scene cut. It needs
both cross-scene discrimination (the same content as the candidate) and within-scene
distinctiveness (`frame_hash.align`'s margin over lags a few frames away); solid-colour or
static shots have only the first and are refused.
"""
import bisect
from fractions import Fraction
from decimal import Decimal
import threading
import time

import numpy as np

import frame_hash
import tools
import repair_log
from frame_compare import FrameComparer, _extract_hashes, parse_positive_rate
from scenedetect import open_video, SceneManager, ContentDetector


SCENE_SEARCH_WINDOW_SECONDS_DEFAULT = 10.0

# ContentDetector's default. The scene list only seeds candidates, so a missed cut is
# tolerated; lower rungs of CONTENT_DETECTOR_THRESHOLD_LADDER cover weak cuts.
CONTENT_DETECTOR_THRESHOLD_DEFAULT = 27.0

# Threshold ladder lowered alongside the window ladder: a rung without usable anchors was not
# sensitive enough. 18.0 clears near-misses just under 27 while staying far above the noise
# floor (median ~0.65); 10.0 catches weak solid-colour cuts. Lower rungs may yield many seeds;
# each is still pHash-validated, so this costs compute, not correctness.
CONTENT_DETECTOR_THRESHOLD_LADDER = (CONTENT_DETECTOR_THRESHOLD_DEFAULT, 18.0, 10.0)

# Minimum consecutive frames for a sweep stop and window viability.
MIN_VALIDATION_FRAMES = 3

# Anchor window: at most ANCHOR_WINDOW_FRAMES master frames on one side of the seed, cut short
# at the nearest scene cut; a window under ANCHOR_WINDOW_MIN_FRAMES is not used.
ANCHOR_WINDOW_FRAMES = 12
ANCHOR_WINDOW_MIN_FRAMES = frame_hash.ALIGN_MIN_FRAMES

# Lags scanned on each side of the nominal shift: they give the margin gate its rivals (a
# self-similar shot matches several of them).
ANCHOR_LAG_REACH = 8

# The sweep stops only after this many consecutive mismatches: isolated pHash noise frames
# occur inside common content, while real cuts stay saturated for several frames.
SWEEP_SUSTAINED_MISMATCH_FRAMES = MIN_VALIDATION_FRAMES

# Extra candidate-window margin relative to the master's: a missed seed is a decline, while
# a wider window only costs decode time.
CANDIDATE_SEED_MARGIN_MULTIPLIER = 2

# Outer window ladder: each rung multiplies the search window (10 s -> 20 s -> 40 s by
# default). Every rung re-extracts frames and re-runs PySceneDetect, so the rung count is
# also a cost bound.
WINDOW_LADDER_GROWTH_FACTOR = 2.0
WINDOW_LADDER_MAX_RUNGS = 3

# Decline reasons that a wider window may fix. `search_window_unviable` (None or <= 0) is a
# config error and terminal; `search_window_too_narrow` (positive but under the frame floor)
# is retryable. Input problems, I/O failures and failures after anchors were found are not
# retried.
WINDOW_LADDER_RETRYABLE_REASONS = frozenset(
    {"anchors_not_established", "anchor_uninformative", "search_window_too_narrow",
     # A wider rung may seat a firm scene-seed anchor; if the last rung still ends here,
     # `locate_scene_anchors` keeps this reason.
     "anchor_ambiguous_static_span"})

# A frame-scan anchor refused because a lag this close to its minimum is as good marks a
# static, ambiguous span.
FRAME_SCAN_AMBIGUITY_FRAMES = 3

# ---------------------------------------------------------------------------
# Edge brackets: one anchor on the common side, then a frame-by-frame pHash walk outward.
# Interior brackets use two anchors and a bidirectional cross-sweep instead.
# ---------------------------------------------------------------------------

# Sustained-mismatch width that stops the edge walk. Common content shows mismatch excursions
# of up to 3 frames, so 4 avoids stopping early and master-filling real candidate frames.
EDGE_WALK_SUSTAINED_MISMATCH_FRAMES = 4

# Walk chunk length: frame extraction costs ~0.45 s per ffmpeg call plus ~2.5 ms/frame, so
# 80 s chunks amortise the fixed cost without excessive memory or latency.
EDGE_WALK_CHUNK_SECONDS = 80.0

# Chunk overlap used to re-align chunk seams. `_extract_hashes` labels element 0 with the
# requested frame, but ffmpeg seeks to the first frame at or after T, so labels can differ by
# one frame between seeks. Each chunk overlaps its predecessor and its base is corrected by
# the delta that makes the overlap agree, keeping frame counts exact.
EDGE_WALK_CHUNK_OVERLAP_FRAMES = 48
# Seam delta search range: +/-1 labelling error plus one frame of margin.
EDGE_WALK_SEAM_SEARCH_FRAMES = 2

# Relative displayed-aspect difference above which geometry is normalised before pHash.
# Frames are downsized to a fixed small size, so small coded-size differences are
# irrelevant, but differing aspects (picture vs black bars) break matching.
EDGE_GEOMETRY_ASPECT_TOLERANCE = 0.02

# Shift search range around the nominal shift, resolved per extraction at the anchor. It
# absorbs rounding of the audio offset onto the video grid and the one-frame labelling
# ambiguity of `_extract_hashes`. Shifts are ranked by the alignment's distance curve, since
# match counts cannot separate one-frame-apart hypotheses.
EDGE_SHIFT_SEARCH_FRAMES = 2

# Edge decline reasons a wider anchor window may fix.
EDGE_WINDOW_LADDER_RETRYABLE_REASONS = frozenset(
    {"edge_anchor_not_established", "edge_anchor_uninformative",
     "search_window_too_narrow"})


def _scene_anchor_config():
    '''Return scene_search_window_sec from config.ini [features].

    Missing section or key -> SCENE_SEARCH_WINDOW_SECONDS_DEFAULT; an unparseable value ->
    None, which the caller reports as search_window_unviable rather than masking the typo.
    '''
    try:
        section = tools.config_loader(tools.config_file, "features")
    except Exception:
        return SCENE_SEARCH_WINDOW_SECONDS_DEFAULT
    raw = section.get("scene_search_window_sec")
    if raw is None:
        return SCENE_SEARCH_WINDOW_SECONDS_DEFAULT
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _exact_ms_from_frame(frame_count, fps_num, fps_den):
    '''Frame index -> exact ms as Decimal, with no float rounding.'''
    frame_ms_exact = Decimal(1000 * fps_den) / Decimal(fps_num)
    return Decimal(frame_count) * frame_ms_exact


def _nominal_shift_frames(offset_ms, fps_num, fps_den):
    '''Offset in ms -> nearest whole-frame shift on the given grid.'''
    frame_ms = 1000.0 * fps_den / fps_num
    return int(round(offset_ms / frame_ms))


_MEDIA_DURATION_S = {}


def _media_duration_s(path):
    """Container duration in seconds via ffprobe (memoised per path), or None."""
    if path not in _MEDIA_DURATION_S:
        value = None
        try:
            with repair_log.announced("scene_anchor", "ffprobe", path) as call:
                stdout, _stderr, _code = tools.launch_cmdExt_with_timeout_reload(
                    [tools.software["ffprobe"], "-v", "error", "-show_entries",
                     "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
                    max_restart=1, timeout=60)
                call["exit"] = _code
            value = float(stdout.decode().strip())
        except Exception:                                                # noqa: BLE001
            value = None
        _MEDIA_DURATION_S[path] = value
    return _MEDIA_DURATION_S[path]


def _clamp_to_file(start, end, duration_s, fps_num, fps_den, scale=None):
    """Clamp a [start, end) window of master-grid frames to a file of duration_s seconds.

    scale converts the file's seconds on a rate pair. Without a duration the window is
    unchanged.
    """
    start = max(0, start)
    if duration_s is not None:
        seconds = Fraction(str(duration_s)) * (Fraction(scale) if scale is not None else 1)
        end = min(end, int(seconds * Fraction(fps_num, fps_den)))
    return start, end


def _probe_frame_rate(path):
    '''Return the file's video frame rate as (Fraction, None), or (None, reason).

    PySceneDetect counts frames on the opened file's own grid, so candidate windows must be
    converted from the master grid using this rate. Reads ffprobe r_frame_rate as an exact
    rational string.
    '''
    try:
        cmd = [tools.software["ffprobe"], "-v", "error",
               "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
               "-of", "default=noprint_wrappers=1:nokey=1", path]
    except KeyError:
        return None, "ffprobe_not_configured"
    tools.dev_log(f"scene_anchor: _probe_frame_rate calling ffprobe "
                  f"file={path}\n")
    try:
        with repair_log.announced("scene_anchor", "ffprobe", path) as call:
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
    return rate, None


def _frames_at_rate(seconds, rate):
    '''Convert exact-rational seconds to a frame count on rate, rounding once.'''
    return int(round(Fraction(seconds) * Fraction(rate)))


def _frame_on_grid(frame, from_rate, to_rate):
    '''Carry a frame index from one grid to another through the instant it names.

    Scene cut indices are on the opened file's grid, while seed arithmetic uses master
    frame numbers. Identity when the rates are equal.
    '''
    if from_rate == to_rate:
        return frame
    return int(round(Fraction(frame) / Fraction(from_rate) * Fraction(to_rate)))


def _scene_cut_frames(path, start_frame, n_frames, threshold, debug=False):
    '''Run ContentDetector over [start_frame, start_frame+n_frames) of path.

    Returns (cuts, None) with absolute frame numbers of interior scene starts (possibly
    empty), or (None, reason) when the detector failed. Only integer frame indices are read,
    never float timecodes.
    '''
    if n_frames <= 0:
        return [], None
    # PySceneDetect runs in-process, so a timer calls `SceneManager.stop()`; a timer stop raises
    # `tools.decoder_timeout`. The bound assumes 24 fps, the slowest expected rate.
    timeout = tools.decoder_timeout_for(n_frames / 24.0)
    fired = []
    try:
        # `open_video`'s default ("opencv") backend hands AV1 to OpenCV's own FFmpeg build,
        # which picks a hardware-only AV1 decoder on this host and reads 0 frames without
        # raising (measured: "Your platform doesn't support hardware accelerated AV1 decoding",
        # `video.frame_number` stays 0, every call silently returns "no scene"). "pyav" opens
        # with PyAV's own software decode and reads every frame on both the AV1 master and
        # candidate measured.
        video = open_video(path, backend="pyav")
        if start_frame > 0:
            video.seek(start_frame)
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=threshold))
        timer = threading.Timer(timeout, lambda: (fired.append(True), sm.stop()))
        tools.dev_log(f"scene_anchor: _scene_cut_frames calling "
                      f"detect_scenes file={path} start_frame={start_frame} "
                      f"n_frames={n_frames} timeout_s={timeout}\n")
        timer.start()
        try:
            with repair_log.announced("scene_anchor", "pyscenedetect", path,
                                      media_s=n_frames / 24.0) as call:
                sm.detect_scenes(video, duration=n_frames)
                call["exit"] = "stopped_by_timer" if fired else 0
        finally:
            timer.cancel()
        if fired:
            raise tools.decoder_timeout("pyscenedetect", timeout,
                                        f"file={path} start_frame={start_frame} "
                                        f"n_frames={n_frames}")
        if video.frame_number == 0:
            # A 0-frame read is a decoder failure, never "no cut": returning `([], None)` here
            # would read as a legitimate scene-less clip and feed a false "no anchor" decline
            # downstream instead of the named, distinguishable cause.
            return None, "scene_detector_blind_decode"
        scene_list = sm.get_scene_list()
        if len(scene_list) < 2:
            return [], None
        return [scene.frame_num for scene, _ in scene_list[1:]], None
    except tools.decoder_timeout:
        raise
    except Exception as exc:
        reason = f"scene_detector_failed:{type(exc).__name__}"
        if debug:
            tools.log_line(f"scene_anchor: PySceneDetect failed on {path} "
                              f"[{start_frame},{start_frame + n_frames}): "
                              f"{reason}\n")
        return None, reason


def _frames_match(m_hashes, m_base, m_frame, c_hashes, c_base, c_frame):
    '''Whether a master frame and a candidate frame show the same picture.

    Returns None when either frame is outside its extracted range (never treated as a
    match or a mismatch).
    '''
    mi, ci = m_frame - m_base, c_frame - c_base
    if not (0 <= mi < len(m_hashes)) or not (0 <= ci < len(c_hashes)):
        return None
    return bool(frame_hash.same_picture(m_hashes[mi], c_hashes[ci])[0])


def _flat_vs_content_mismatch(m_hashes, m_base, m_frame, c_hashes, c_base, c_frame):
    '''A mismatching frame pair where exactly one side is flat (black/near-black).

    The owner's rule (2026-10-06): a candidate fade or dim frame against a master black
    frame, or a black candidate frame against master content, is a frame that genuinely
    differs -- never a pHash noise reading to tolerate, whatever its run length. Returns
    False when either frame is outside its extracted range (handled by the mismatch count
    itself) or when both sides agree on flatness (two different noise readings of the same
    held black/static picture, not a content difference).
    '''
    mi, ci = m_frame - m_base, c_frame - c_base
    if not (0 <= mi < len(m_hashes)) or not (0 <= ci < len(c_hashes)):
        return False
    return bool(m_hashes.std[mi] < frame_hash.FLAT_STD) != bool(c_hashes.std[ci] < frame_hash.FLAT_STD)


def _span_matches(m_hashes, m_base, c_hashes, c_base, first, stop, shift):
    '''(matched, readable) frame counts of master [first, stop) against the candidate at shift.'''
    m_idx = np.arange(first, stop) - m_base
    c_idx = m_idx + m_base + shift - c_base
    ok = (m_idx >= 0) & (m_idx < len(m_hashes)) & (c_idx >= 0) & (c_idx < len(c_hashes))
    if not ok.any():
        return 0, 0
    same = frame_hash.same_picture(m_hashes[m_idx[ok]], c_hashes[c_idx[ok]])
    return int(same.sum()), int(ok.sum())


def _anchor_window(seed, direction, boundaries, m_base, m_len):
    '''Master frames [first, stop) of the anchor window at seed, or None.

    "backward" ends at the seed (Anchor A side), "forward" starts at it (Anchor B side). The
    window holds at most ANCHOR_WINDOW_FRAMES frames and stops at the nearest boundary (a
    scene cut of either file), so it never holds two shots; under
    ANCHOR_WINDOW_MIN_FRAMES frames it is not used.
    '''
    pos = bisect.bisect_right(boundaries, seed)
    if direction == "forward":
        first = seed
        stop = seed + ANCHOR_WINDOW_FRAMES
        if pos < len(boundaries):
            stop = min(stop, boundaries[pos])
    else:
        stop = seed
        first = seed - ANCHOR_WINDOW_FRAMES
        before = bisect.bisect_left(boundaries, seed)
        if before > 0:
            first = max(first, boundaries[before - 1])
    first, stop = max(first, m_base), min(stop, m_base + m_len)
    if stop - first < ANCHOR_WINDOW_MIN_FRAMES:
        return None
    return first, stop


def _try_anchor(m_hashes, m_base, c_hashes, c_base, seed, nominal_shift, direction,
                boundaries, search_frames):
    '''Align the anchor window at seed against the candidate around nominal_shift.

    Lags nominal +/- max(search_frames, ANCHOR_LAG_REACH) are scanned by
    `frame_hash.align`; the window anchors when the minimum passes the margin gate, lies
    within search_frames of nominal (one frame when search_frames is 0: the shift is then
    kept, as a frame-exact offset only needs the content confirmed) and shows the same
    content (SAME_CONTENT_MAX).

    Returns:
        dict: verdict ("anchor", "unreadable", "mismatch" or "uninformative"), window,
        alignment, content, and for "uninformative" the near lags (`static_lags`).
    '''
    window = _anchor_window(seed, direction, boundaries, m_base, len(m_hashes))
    if window is None:
        return {"verdict": "unreadable", "window": None, "alignment": None, "content": None}
    first, stop = window
    ref = m_hashes[first - m_base:stop - m_base]
    reach = max(search_frames, ANCHOR_LAG_REACH)
    alignment = frame_hash.align(ref, c_hashes,
                                 range(nominal_shift - reach, nominal_shift + reach + 1),
                                 start=first - c_base)
    out = {"window": window, "alignment": alignment, "content": None}
    if alignment.best is None:
        return {**out, "verdict": "unreadable"}
    content = frame_hash.window_content_distance(ref, c_hashes,
                                                 first - c_base + alignment.best)
    out["content"] = content
    if not content <= frame_hash.SAME_CONTENT_MAX:
        return {**out, "verdict": "mismatch"}
    tolerance = max(search_frames, 1)
    if not alignment.ok:
        low = alignment.curve[alignment.best]
        near = sorted(lag for lag, value in alignment.curve.items()
                      if value - low < frame_hash.ALIGN_MARGIN)
        if any(abs(lag - nominal_shift) <= tolerance for lag in near):
            # the shot looks the same at the nominal shift and at others: self-similar
            return {**out, "verdict": "uninformative", "static_lags": near}
        return {**out, "verdict": "mismatch"}
    if abs(alignment.best - nominal_shift) > tolerance:
        return {**out, "verdict": "mismatch"}
    return {**out, "verdict": "anchor"}


def _anchor_search(m_hashes, m_base, c_hashes, c_base, seeds, nominal_shift, direction,
                   boundaries, search_frames=0, log_tag="", log_rungs=True,
                   static_reach=None):
    '''Return the first seed (master frame number) whose window anchors.

    Seeds are the union of both files' scene cuts (in master coordinates) plus the bracket
    edge, ordered by the caller. search_frames 0 keeps the nominal shift; a wider value
    resolves the shift within it.

    With static_reach set (frame scan), the first seed that shows the same content but is
    refused because a lag within static_reach of its minimum is as good ends the search as
    an ambiguous static span.

    Returns:
        (seed, shift, n_frames, reason, static): seed None on failure, with reason None if
        nothing showed the same content, else the first uninformative seed's evidence;
        static is the ambiguous-span dict or None.
    '''
    first_reason = first_n_frames = None
    for seed in seeds:
        tried = _try_anchor(m_hashes, m_base, c_hashes, c_base, seed, nominal_shift,
                            direction, boundaries, search_frames)
        verdict, alignment, window = tried["verdict"], tried["alignment"], tried["window"]
        n_frames = None if window is None else window[1] - window[0]
        if log_rungs:
            tools.log_line(
                f"scene_anchor: anchor_window direction={direction} seed={seed} "
                f"window={window} verdict={verdict} nominal_shift={nominal_shift} "
                f"best={None if alignment is None else alignment.best} "
                f"reason={None if alignment is None else alignment.reason} "
                f"margin={None if alignment is None or alignment.margin is None else round(alignment.margin, 4)} "
                f"content={None if tried['content'] is None else round(tried['content'], 4)}"
                f"{log_tag}\n")
        if verdict == "anchor":
            return seed, alignment.best if search_frames else nominal_shift, n_frames, None, None
        if verdict != "uninformative":
            continue
        static = [lag for lag in tried["static_lags"]
                  if abs(lag - nominal_shift) <= (static_reach or 0)]
        if static_reach is not None and len(static) >= 2 and static[-1] - static[0] >= 2:
            return None, nominal_shift, n_frames, None, {
                "anchor": seed, "shift": nominal_shift, "n_frames": n_frames,
                "ambiguous_shift_span": [static[0], static[-1]], "validating_shifts": static}
        if first_reason is None:
            # Keep only the first such seed's evidence: the closest candidate.
            first_reason = (f"seed={seed} window={window} best_shift={alignment.best} "
                            f"also near at shift={alignment.rival} "
                            f"(margin={alignment.margin}, {alignment.reason}) -- "
                            f"self-similar content, uninformative")
            first_n_frames = n_frames
    return None, None, first_n_frames, first_reason, None


def _check_step_plumbing(delta_frames, frame_ms, step_ms, quantum_ms):
    '''Consistency check between the anchors' frame delta and the locator's step_ms.

    Both derive from the same offsets, so this only catches plumbing errors (wrong bracket,
    unit mismatch); it does not corroborate anchor placement. Missing inputs return
    (False, "plumbing_check=not_available ...").
    '''
    if step_ms is None or quantum_ms is None:
        return False, f"plumbing_check=not_available step_ms={step_ms} quantum_ms={quantum_ms}"
    delta_ms = delta_frames * frame_ms
    # One audio quantum plus two frames: each of the two shifts is rounded to a whole frame.
    tolerance_ms = quantum_ms + 2 * frame_ms
    agrees = abs(delta_ms - step_ms) <= tolerance_ms
    return agrees, (f"counted_delta={delta_frames} frames ({delta_ms:.2f} ms) vs "
                    f"locator step_ms={step_ms} quantum_ms={quantum_ms} "
                    f"step_tolerance=quantum+2frames ({tolerance_ms:.2f} ms)")


def _check_anchor_ordering(anchor_a, anchor_b):
    '''Return (True, evidence) when anchor A lies after anchor B, else (False, None).

    Strict >: a sub-frame bracket legitimately gives anchor_a == anchor_b, which the
    cross-sweep handles as zero width.
    '''
    if anchor_a > anchor_b:
        return True, f"anchor_a={anchor_a} > anchor_b={anchor_b}"
    return False, None


def _frame_scan_anchor(m_hashes, m_base, c_hashes, c_base, scene_seeds, bracket_frame, side,
                       shift, boundaries, search_frames):
    '''Fallback anchor search for static shots, run only after every scene seed failed.

    Every master frame outward from the bracket edge over already-extracted hashes is a
    seed, with the side's one-sided window (never across a cut). The first seed that shows
    the same content but cannot be told from a lag within FRAME_SCAN_AMBIGUITY_FRAMES is an
    ambiguous static span and ends the search.

    Returns (seed, shift, n_frames, reason, ambiguous).
    '''
    one_sided = "backward" if side == "A" else "forward"
    tried = set(scene_seeds)
    if side == "A":
        scan = [f for f in range(bracket_frame - 1, m_base + ANCHOR_WINDOW_MIN_FRAMES - 1, -1)
                if f not in tried]
    else:
        scan = [f for f in range(bracket_frame + 1,
                                 m_base + len(m_hashes) - ANCHOR_WINDOW_MIN_FRAMES + 1)
                if f not in tried]
    seed, found_shift, n_frames, reason, ambiguous = _anchor_search(
        m_hashes, m_base, c_hashes, c_base, scan, shift, one_sided, boundaries,
        search_frames=search_frames, log_rungs=False,
        static_reach=FRAME_SCAN_AMBIGUITY_FRAMES)
    if ambiguous is not None:
        ambiguous["side"] = side
    tools.log_line(
        f"scene_anchor: static_shot_fallback seed_source=frame_scan side={side} "
        f"direction={one_sided} bracket_frame={bracket_frame} "
        f"frames_scanned_max={len(scan)} "
        f"scan_span=[{scan[-1] if scan else None},{scan[0] if scan else None}] "
        f"nominal_shift={shift} anchor={seed} "
        f"shift={found_shift if seed is not None else None} n_frames={n_frames} "
        f"distance_frames={None if seed is None else abs(bracket_frame - seed)} "
        + (f"accepted=False ambiguous_shift_span={ambiguous['ambiguous_shift_span']} "
           f"validating_shifts={ambiguous['validating_shifts']} "
           if ambiguous is not None else "")
        + f"reason={reason}\n")
    if ambiguous is not None:
        return None, shift, n_frames, None, ambiguous
    if seed is not None:
        return seed, found_shift, n_frames, None, None
    return None, shift, n_frames, reason, None


def island_match(master_path, candidate_path, fps_num, fps_den, low_ms, high_ms, offset_ms,
                 candidate_time_scale=None, shift_search_frames=EDGE_SHIFT_SEARCH_FRAMES,
                 scan_cache=None):
    '''Check whether an island (aligned zone between two holes) matches the candidate.

    Every master frame of [low_ms, high_ms) is compared with the candidate at the nominal
    shift +/- shift_search_frames; the best shift (ties: closest to nominal) is kept.
    Verdict by majority of readable frames, since pHash noise in common content affects only
    a few frames: "same", "differs", or "unreadable" (no verdict). Decodes through
    scan_cache when given.
    '''
    geometry, crop_filters, geometry_reason = _scan_memo(
        scan_cache, ("geometry", master_path, candidate_path),
        lambda: _resolve_geometry(master_path, candidate_path))
    if geometry_reason is not None:
        return {"verdict": "unreadable", "reason": f"geometry_unreconciled:{geometry_reason}"}
    comparer = FrameComparer(master_path, candidate_path, low_ms / 1000.0, high_ms / 1000.0,
                             fps_num, fps_den, crop_filters=crop_filters,
                             time_scales=({candidate_path: candidate_time_scale}
                                          if candidate_time_scale is not None else None))
    first = comparer._frame_index(low_ms / 1000.0)
    last = comparer._frame_index(high_ms / 1000.0)
    nominal = _nominal_shift_frames(offset_ms, fps_num, fps_den)
    frame_s = fps_den / fps_num
    m_start, m_dur = first * frame_s, (last - first) * frame_s
    c_start = (first + nominal - shift_search_frames) * frame_s
    c_dur = (last - first + 2 * shift_search_frames) * frame_s
    geometry_key = repr(sorted((crop_filters or {}).items()))
    m_base, m_hashes = _scan_memo(
        scan_cache, ("hashes", master_path, m_start, m_dur, geometry_key),
        lambda: _extract_hashes(comparer, master_path, m_start, m_dur))
    c_base, c_hashes = _scan_memo(
        scan_cache, ("hashes", candidate_path, c_start, c_dur, geometry_key,
                     str(candidate_time_scale)),
        lambda: _extract_hashes(comparer, candidate_path, c_start, c_dur))
    best = None
    for shift in range(nominal - shift_search_frames, nominal + shift_search_frames + 1):
        matched, readable = _span_matches(m_hashes, m_base, c_hashes, c_base, first, last, shift)
        if readable and (best is None or matched > best[1]
                         or (matched == best[1] and abs(shift - nominal) < abs(best[0] - nominal))):
            best = (shift, matched, readable)
    if best is None:
        return {"verdict": "unreadable", "reason": "no_frame_readable",
                "master_frames": [first, last], "nominal_shift_frames": nominal}
    shift, matched, readable = best
    return {"verdict": "same" if 2 * matched > readable else "differs",
            "shift_frames": shift, "nominal_shift_frames": nominal,
            "matched": matched, "readable": readable, "master_frames": [first, last],
            "reason": None}


def _scan_memo(cache, key, compute):
    """Memoise compute() under key in a cluster's shared scan cache (no cache: compute)."""
    if cache is None:
        return compute()
    if key not in cache:
        cache[key] = compute()
    return cache[key]


def locate_scene_anchors(master_path, candidate_path, fps_num, fps_den,
                         bracket_low_ms, bracket_high_ms,
                         offset_before_ms, offset_after_ms,
                         step_ms=None, quantum_ms=None,
                         scene_search_window_sec=None, debug=False,
                         candidate_time_scale=None, normalise_geometry=False,
                         resolve_shift=False,
                         shift_search_frames=EDGE_SHIFT_SEARCH_FRAMES,
                         cluster_window=None, scan_cache=None, deadline=None,
                         accept_step_disagreement=False):
    '''Locate Anchor A (before) and Anchor B (after) an interior bracket and cross-sweep.

    Optional keywords (all off by default):
      candidate_time_scale  rate relation r (Fraction): the candidate plays r times faster;
                            offsets and seconds are then in master-equivalent time.
      normalise_geometry    resolve letterbox crops first (_resolve_geometry).
      resolve_shift         re-resolve each anchor's shift within shift_search_frames
                            of nominal, for offsets quantised coarser than a frame.
      cluster_window /      shared scan window for a cluster of nearby holes;
      scan_cache            scan_cache memoises decoded hashes and cuts per window.
      deadline              time.monotonic() limit checked between rungs
                            (declines hole_budget_exceeded).
      accept_step_disagreement  both anchors were found and the cross-sweep ran (an
                            ambiguous span never reaches this gate), but the frame gap it
                            counted does not match the caller's nominal step within
                            tolerance -- deliver the counted gap anyway, flagged
                            (`step_plumbing_ok`), instead of declining
                            `anchor_step_inconsistent`. Still declines when an anchor
                            itself was never seated, or when the anchors' own order is
                            refused.

    Runs _locate_scene_anchors_at_window up to WINDOW_LADDER_MAX_RUNGS times, growing the
    window by WINDOW_LADDER_GROWTH_FACTOR and lowering the detector threshold per
    CONTENT_DETECTOR_THRESHOLD_LADDER, but only for WINDOW_LADDER_RETRYABLE_REASONS.
    Exhaustion declines search_window_ceiling_reached, unless the last rung ended
    anchor_ambiguous_static_span. Each rung is logged.
    '''
    window_sec = (scene_search_window_sec if scene_search_window_sec is not None
                 else _scene_anchor_config())

    crop_filters = {}
    geometry = None
    if normalise_geometry:
        # A geometry refusal is terminal: no window size reconciles different framings.
        geometry, crop_filters, geometry_reason = _resolve_geometry(
            master_path, candidate_path)
        tools.log_line(
            f"scene_anchor: interior_geometry "
            f"master={geometry.get('master')} candidate={geometry.get('candidate')} "
            f"normalised={geometry.get('normalised')} crop={geometry.get('crop')} "
            f"verdict={geometry.get('verdict')} reason={geometry_reason}\n")
        if geometry_reason is not None:
            return {"declined": True, "reason": "geometry_unreconciled",
                    "geometry": geometry, "evidence": geometry_reason}

    result = None
    rung_cd_threshold = CONTENT_DETECTOR_THRESHOLD_LADDER[0]
    for rung in range(WINDOW_LADDER_MAX_RUNGS):
        if deadline is not None and time.monotonic() > deadline:
            return {"declined": True, "reason": "hole_budget_exceeded",
                    "evidence": f"rungs_run={rung} last_reason="
                                f"{(result or {}).get('reason')}"}
        rung_window_sec = (window_sec if window_sec is None
                           else window_sec * (WINDOW_LADDER_GROWTH_FACTOR ** rung))
        # Clamped in case the two ladders' lengths differ.
        rung_cd_threshold = CONTENT_DETECTOR_THRESHOLD_LADDER[
            min(rung, len(CONTENT_DETECTOR_THRESHOLD_LADDER) - 1)]
        try:
            result = _locate_scene_anchors_at_window(
                master_path, candidate_path, fps_num, fps_den,
                bracket_low_ms, bracket_high_ms, offset_before_ms, offset_after_ms,
                rung_window_sec, step_ms=step_ms, quantum_ms=quantum_ms,
                cluster_window=cluster_window, scan_cache=scan_cache,
                content_detector_threshold=rung_cd_threshold, debug=debug,
                candidate_time_scale=candidate_time_scale,
                crop_filters=crop_filters, resolve_shift=resolve_shift,
                shift_search_frames=shift_search_frames,
                accept_step_disagreement=accept_step_disagreement)
        except tools.decoder_timeout as error:
            return {"declined": True, "reason": "decoder_timeout",
                    "evidence": f"rung={rung} {error}"}
        if geometry is not None:
            result["geometry"] = geometry
        matched = not result["declined"]
        tools.log_line(
            f"scene_anchor: window_ladder_rung rung={rung} "
            f"dial=window_sec+cd_threshold "
            f"window_sec={rung_window_sec} cd_threshold={rung_cd_threshold} "
            f"master_seed_count={result.get('master_seed_count')} "
            f"candidate_seed_count={result.get('candidate_seed_count')} "
            f"matched={matched} "
            f"reason={result.get('reason')} evidence={result.get('evidence')}\n")
        if matched or result["reason"] not in WINDOW_LADDER_RETRYABLE_REASONS:
            return result

    if result.get("reason") == "anchor_ambiguous_static_span":
        # The last rung's ambiguous frame-scan anchor is the cause, not the ceiling.
        result = dict(result)
        result["evidence"] = (f"rungs_tried={WINDOW_LADDER_MAX_RUNGS} "
                              f"final_window_sec={rung_window_sec} "
                              f"{result.get('evidence')}")
        return result

    return {"declined": True, "reason": "search_window_ceiling_reached",
           "evidence": f"rungs_tried={WINDOW_LADDER_MAX_RUNGS} "
                      f"base_window_sec={window_sec} "
                      f"final_window_sec={rung_window_sec} "
                      f"final_cd_threshold={rung_cd_threshold} "
                      f"final_master_seed_count={result.get('master_seed_count')} "
                      f"final_candidate_seed_count={result.get('candidate_seed_count')} "
                      f"last_reason={result.get('reason')} "
                      f"last_evidence={result.get('evidence')}"}


def _locate_scene_anchors_at_window(master_path, candidate_path, fps_num, fps_den,
                                    bracket_low_ms, bracket_high_ms,
                                    offset_before_ms, offset_after_ms,
                                    window_sec, step_ms=None, quantum_ms=None,
                                    content_detector_threshold=CONTENT_DETECTOR_THRESHOLD_DEFAULT,
                                    debug=False, candidate_time_scale=None,
                                    crop_filters=None, resolve_shift=False,
                                    shift_search_frames=EDGE_SHIFT_SEARCH_FRAMES,
                                    cluster_window=None, scan_cache=None,
                                    accept_step_disagreement=False):
    '''Run one rung of locate_scene_anchors' window ladder at a concrete window_sec.

    offset_before_ms/offset_after_ms are the offset hypotheses on each side; this finds
    the exact frame where each stops applying and classifies the gap (offset change,
    deletion, addition, still image). step_ms/quantum_ms feed only
    _check_step_plumbing, whose disagreement is fatal unless accept_step_disagreement
    is set, in which case the counted gap is returned anyway with step_plumbing_ok=False.
    Returns a dict with declined True or False (and reason, evidence when declined).
    Head/tail edges use locate_edge_boundary instead.
    '''
    fps_num = int(fps_num)
    fps_den = int(fps_den)
    if fps_num <= 0 or fps_den <= 0:
        return {"declined": True, "reason": "grid_unmeasured",
               "evidence": f"fps_num={fps_num} fps_den={fps_den}"}

    if bracket_high_ms <= bracket_low_ms:
        return {"declined": True, "reason": "empty_bracket",
               "evidence": f"[{bracket_low_ms},{bracket_high_ms}] ms"}

    frame_ms = 1000.0 * fps_den / fps_num

    # None or <= 0 is a config error that widening cannot fix: terminal.
    if window_sec is None or window_sec <= 0:
        return {"declined": True, "reason": "search_window_unviable",
               "evidence": f"scene_search_window_sec={window_sec}"}

    # A positive window under MIN_VALIDATION_FRAMES frames is retryable at a wider rung.
    window_frames = int(round((window_sec * 1000.0) / frame_ms))
    if window_frames < MIN_VALIDATION_FRAMES:
        return {"declined": True, "reason": "search_window_too_narrow",
               "evidence": f"scene_search_window_sec={window_sec} -> "
                          f"{window_frames} frames, needs >= "
                          f"{MIN_VALIDATION_FRAMES}"}

    comparer = FrameComparer(master_path, candidate_path,
                             bracket_low_ms / 1000.0, bracket_high_ms / 1000.0,
                             fps_num, fps_den, debug=debug,
                             crop_filters=crop_filters,
                             time_scales=({candidate_path: candidate_time_scale}
                                          if candidate_time_scale is not None
                                          else None))
    m_bracket_first = comparer._frame_index(bracket_low_ms / 1000.0)
    m_bracket_last = comparer._frame_index(bracket_high_ms / 1000.0)

    before_shift = _nominal_shift_frames(offset_before_ms, fps_num, fps_den)
    after_shift = _nominal_shift_frames(offset_after_ms, fps_num, fps_den)
    nominal_before_shift, nominal_after_shift = before_shift, after_shift

    # Windows are built around this bracket and its shifts, or around the whole cluster's
    # bracket and shift range for a shared pass.
    span_first, span_last = m_bracket_first, m_bracket_last
    low_shift, high_shift = min(before_shift, after_shift), max(before_shift, after_shift)
    if cluster_window is not None:
        span_first = min(span_first,
                         comparer._frame_index(cluster_window["bracket_ms"][0] / 1000.0))
        span_last = max(span_last,
                        comparer._frame_index(cluster_window["bracket_ms"][1] / 1000.0))
        low_shift = min(low_shift, _nominal_shift_frames(cluster_window["offsets_ms"][0],
                                                         fps_num, fps_den))
        high_shift = max(high_shift, _nominal_shift_frames(cluster_window["offsets_ms"][1],
                                                           fps_num, fps_den))
    m_win_start = max(0, span_first - window_frames)
    m_win_end = span_last + window_frames
    # The candidate window covers both offset hypotheses plus an extra margin; it only
    # generates seeds, so generosity costs decode time, not correctness.
    candidate_margin_frames = CANDIDATE_SEED_MARGIN_MULTIPLIER * window_frames
    c_win_start = max(0, span_first - candidate_margin_frames + low_shift)
    c_win_end = span_last + candidate_margin_frames + high_shift
    # Clamp both windows to their files: a negative length would decode the whole file.
    m_win_start, m_win_end = _clamp_to_file(m_win_start, m_win_end,
                                            _media_duration_s(master_path), fps_num, fps_den)
    c_win_start, c_win_end = _clamp_to_file(c_win_start, c_win_end,
                                            _media_duration_s(candidate_path), fps_num, fps_den,
                                            candidate_time_scale)
    if m_win_end <= m_win_start or c_win_end <= c_win_start:
        return {"declined": True, "reason": "candidate_window_empty",
                "evidence": f"master_window=[{m_win_start},{m_win_end}) "
                            f"candidate_window=[{c_win_start},{c_win_end}) "
                            f"before_shift={before_shift} after_shift={after_shift}"}

    # Window bounds are master frame numbers. `_extract_hashes` takes seconds, but
    # `_scene_cut_frames` takes frame numbers on the opened file's own grid, so the candidate
    # scan is counted at the candidate's rate. Seconds are exact rationals, rounded once per
    # side.
    master_rate = Fraction(fps_num, fps_den)
    m_win_start_sec = Fraction(m_win_start * fps_den, fps_num)
    m_win_span_sec = Fraction((m_win_end - m_win_start) * fps_den, fps_num)
    c_win_start_sec = Fraction(c_win_start * fps_den, fps_num)
    c_win_span_sec = Fraction((c_win_end - c_win_start) * fps_den, fps_num)

    candidate_rate, candidate_rate_reason = _probe_frame_rate(candidate_path)
    # A speed-changed candidate is scanned at its corrected rate, so the window maps to the
    # right raw span and cuts map back to the master grid.
    if candidate_rate is not None and candidate_time_scale is not None:
        candidate_rate = candidate_rate / Fraction(candidate_time_scale)

    m_start_s = float(m_win_start_sec)
    m_dur_s = float(m_win_span_sec)
    c_start_s = float(c_win_start_sec)
    c_dur_s = float(c_win_span_sec)

    m_scan_start = _frames_at_rate(m_win_start_sec, master_rate)
    m_scan_frames = _frames_at_rate(m_win_span_sec, master_rate)
    if candidate_rate is None:
        c_scan_start = c_scan_frames = None
    else:
        c_scan_start = _frames_at_rate(c_win_start_sec, candidate_rate)
        c_scan_frames = _frames_at_rate(c_win_span_sec, candidate_rate)
    tools.dev_log(
        f"scene_anchor: scan_window_conversion "
        f"master_rate={master_rate.numerator}/{master_rate.denominator} "
        f"candidate_rate="
        f"{'unmeasured:' + str(candidate_rate_reason) if candidate_rate is None else str(candidate_rate.numerator) + '/' + str(candidate_rate.denominator)} "
        f"master_scan=[{m_scan_start},+{m_scan_frames}) "
        f"({float(m_win_span_sec):.3f} s) "
        f"candidate_scan=[{c_scan_start},+{c_scan_frames}) "
        f"({float(c_win_span_sec):.3f} s)\n")

    geometry_key = repr(sorted((crop_filters or {}).items()))
    m_base, m_hashes = _scan_memo(
        scan_cache, ("hashes", master_path, m_start_s, m_dur_s, geometry_key),
        lambda: _extract_hashes(comparer, master_path, m_start_s, m_dur_s))
    c_base, c_hashes = _scan_memo(
        scan_cache, ("hashes", candidate_path, c_start_s, c_dur_s, geometry_key,
                     str(candidate_time_scale)),
        lambda: _extract_hashes(comparer, candidate_path, c_start_s, c_dur_s))
    if not m_hashes or not c_hashes:
        return {"declined": True, "reason": "frames_unextractable",
               "evidence": f"master_frames={len(m_hashes)} "
                          f"candidate_frames={len(c_hashes)}"}

    cd_threshold = content_detector_threshold

    master_cuts, master_cuts_failed = _scan_memo(
        scan_cache, ("cuts", master_path, m_scan_start, m_scan_frames, cd_threshold),
        lambda: _scene_cut_frames(master_path, m_scan_start, m_scan_frames, cd_threshold,
                                  debug))
    if candidate_rate is None:
        # Unknown candidate grid: contribute no candidate seeds rather than guess the rate.
        # The bracket edge is still tried.
        candidate_cuts = None
        candidate_cuts_failed = f"candidate_grid_unmeasured:{candidate_rate_reason}"
    else:
        candidate_cuts, candidate_cuts_failed = _scan_memo(
            scan_cache, ("cuts", candidate_path, c_scan_start, c_scan_frames, cd_threshold),
            lambda: _scene_cut_frames(candidate_path, c_scan_start, c_scan_frames,
                                      cd_threshold, debug))
    # A failed side contributes no seeds; the `*_cuts_failed` tokens keep the cause in the
    # evidence.
    master_cuts_seeds = master_cuts or []
    candidate_cuts_seeds = [
        _frame_on_grid(f, candidate_rate, master_rate)
        for f in (candidate_cuts or [])]

    # Anchor A seeds, closest first: the bracket's low edge, master cuts before it, then
    # candidate cuts (translated under the BEFORE hypothesis) before it.
    a_seeds_master = sorted(
        {m_bracket_first}
        | {f for f in master_cuts_seeds if f <= m_bracket_first}
        | {f - before_shift for f in candidate_cuts_seeds if f - before_shift <= m_bracket_first},
        reverse=True)
    # Each anchor resolves its own shift when asked; `_check_step_plumbing` then compares
    # their difference with the caller's step.
    search = shift_search_frames if resolve_shift else 0
    a_boundaries = sorted(set(master_cuts_seeds)
                          | {f - before_shift for f in candidate_cuts_seeds})
    anchor_a, resolved, anchor_a_n_frames, anchor_a_reason, _ = _anchor_search(
        m_hashes, m_base, c_hashes, c_base, a_seeds_master,
        before_shift, "backward", a_boundaries, search_frames=search)
    if anchor_a is not None:
        before_shift = resolved
    anchor_a_ambiguous = anchor_b_ambiguous = None
    if anchor_a is None:
        anchor_a, fallback_shift, fallback_n_frames, fallback_reason, anchor_a_ambiguous = \
            _frame_scan_anchor(
                m_hashes, m_base, c_hashes, c_base, a_seeds_master,
                m_bracket_first, "A", before_shift, a_boundaries, search)
        if anchor_a is not None:
            before_shift, anchor_a_n_frames = fallback_shift, fallback_n_frames
            anchor_a_reason = None
        elif anchor_a_reason is None and fallback_reason is not None:
            anchor_a_reason, anchor_a_n_frames = fallback_reason, fallback_n_frames

    b_seeds_master = sorted(
        {m_bracket_last}
        | {f for f in master_cuts_seeds if f >= m_bracket_last}
        | {f - after_shift for f in candidate_cuts_seeds if f - after_shift >= m_bracket_last})
    b_boundaries = sorted(set(master_cuts_seeds)
                          | {f - after_shift for f in candidate_cuts_seeds})
    anchor_b, resolved, anchor_b_n_frames, anchor_b_reason, _ = _anchor_search(
        m_hashes, m_base, c_hashes, c_base, b_seeds_master,
        after_shift, "forward", b_boundaries, search_frames=search)
    if anchor_b is not None:
        after_shift = resolved
    if anchor_b is None:
        anchor_b, fallback_shift, fallback_n_frames, fallback_reason, anchor_b_ambiguous = \
            _frame_scan_anchor(
                m_hashes, m_base, c_hashes, c_base, b_seeds_master,
                m_bracket_last, "B", after_shift, b_boundaries, search)
        if anchor_b is not None:
            after_shift, anchor_b_n_frames = fallback_shift, fallback_n_frames
            anchor_b_reason = None
        elif anchor_b_reason is None and fallback_reason is not None:
            anchor_b_reason, anchor_b_n_frames = fallback_reason, fallback_n_frames

    if anchor_a is None or anchor_b is None:
        if anchor_a_ambiguous is not None or anchor_b_ambiguous is not None:
            # The frame-scan anchor validates at several neighbouring shifts (static zone).
            # Callers read the flags and the firm side's anchor/shift from the fields below.
            return {"declined": True, "reason": "anchor_ambiguous_static_span",
                   "master_seed_count": len(master_cuts_seeds),
                   "candidate_seed_count": len(candidate_cuts_seeds),
                   "anchor_a_ambiguous": anchor_a_ambiguous,
                   "anchor_b_ambiguous": anchor_b_ambiguous,
                   "anchor_a_frame": anchor_a, "anchor_b_frame": anchor_b,
                   "before_shift_frames": before_shift,
                   "after_shift_frames": after_shift,
                   "nominal_before_shift_frames": nominal_before_shift,
                   "nominal_after_shift_frames": nominal_after_shift,
                   "grid": {"num": fps_num, "den": fps_den},
                   "evidence": f"anchor_a={anchor_a} anchor_b={anchor_b} "
                              f"a_ambiguous={anchor_a_ambiguous} "
                              f"b_ambiguous={anchor_b_ambiguous} "
                              f"a_reason={anchor_a_reason} "
                              f"b_reason={anchor_b_reason}"}
        if anchor_a_reason or anchor_b_reason:
            return {"declined": True, "reason": "anchor_uninformative",
                   "master_seed_count": len(master_cuts_seeds),
                   "candidate_seed_count": len(candidate_cuts_seeds),
                   "evidence": f"anchor_a={anchor_a} anchor_b={anchor_b} "
                              f"a_reason={anchor_a_reason} "
                              f"a_n_frames={anchor_a_n_frames} "
                              f"b_reason={anchor_b_reason} "
                              f"b_n_frames={anchor_b_n_frames}"}
        return {"declined": True, "reason": "anchors_not_established",
               "master_seed_count": len(master_cuts_seeds),
               "candidate_seed_count": len(candidate_cuts_seeds),
               "evidence": f"anchor_a={anchor_a} anchor_b={anchor_b} "
                          f"master_cuts={len(master_cuts_seeds)} "
                          f"master_detector_failed={master_cuts_failed} "
                          f"candidate_cuts={len(candidate_cuts_seeds)} "
                          f"candidate_detector_failed={candidate_cuts_failed}"}

    ordering_refuted, ordering_evidence = _check_anchor_ordering(anchor_a, anchor_b)
    if ordering_refuted:
        return {"declined": True, "reason": "cross_sweep_refuted",
               "master_seed_count": len(master_cuts_seeds),
               "candidate_seed_count": len(candidate_cuts_seeds),
               "evidence": ordering_evidence}

    # Cross-sweep: forward from A under before_shift, backward from B under after_shift,
    # each capped at the other anchor. A sweep stops only after
    # SWEEP_SUSTAINED_MISMATCH_FRAMES consecutive mismatches and lands on the last
    # matching frame -- except a flat-vs-content mismatch (owner's rule 2: a candidate
    # fade/black frame against the master's black/content, never read by "how black" it
    # is), which ends the walk's own extension at once, however short the run, instead of
    # being absorbed back into the matched span when ordinary frames resume matching.
    split_start_master = anchor_a
    consecutive_mismatches = 0
    for m_frame in range(anchor_a, anchor_b):
        c_frame = m_frame + before_shift
        if _frames_match(m_hashes, m_base, m_frame, c_hashes, c_base, c_frame) is True:
            split_start_master = m_frame + 1
            consecutive_mismatches = 0
        else:
            consecutive_mismatches += 1
            if (consecutive_mismatches >= SWEEP_SUSTAINED_MISMATCH_FRAMES
                    or _flat_vs_content_mismatch(m_hashes, m_base, m_frame,
                                                 c_hashes, c_base, c_frame)):
                break
    split_start_candidate = split_start_master + before_shift

    split_end_master = anchor_b
    consecutive_mismatches = 0
    for m_frame in range(anchor_b - 1, anchor_a - 1, -1):
        c_frame = m_frame + after_shift
        if _frames_match(m_hashes, m_base, m_frame, c_hashes, c_base, c_frame) is True:
            split_end_master = m_frame
            consecutive_mismatches = 0
        else:
            consecutive_mismatches += 1
            if (consecutive_mismatches >= SWEEP_SUSTAINED_MISMATCH_FRAMES
                    or _flat_vs_content_mismatch(m_hashes, m_base, m_frame,
                                                 c_hashes, c_base, c_frame)):
                break
    split_end_candidate = split_end_master + after_shift

    # Keep the pre-collapse fronts so a crossed bracket is distinguishable from a
    # zero-width one.
    sweep_crossed = split_end_master < split_start_master
    pre_collapse_start_master = split_start_master
    pre_collapse_end_master = split_end_master
    # Per-hypothesis match counts over the unmatched span (observational): pHash noise can
    # outrun the sustained-mismatch rule, so callers use these to tell divergence from noise.
    span_counts = {}
    for label, span_shift in (("before", before_shift), ("after", after_shift)):
        span_counts[label] = list(_span_matches(
            m_hashes, m_base, c_hashes, c_base, pre_collapse_start_master,
            pre_collapse_end_master, span_shift))
    if sweep_crossed:
        # Still-image degenerate case: the sweeps crossed, so collapse both to anchor B
        # instead of reporting a negative-length interior.
        split_start_master = split_end_master = anchor_b
        split_start_candidate = split_start_master + before_shift
        split_end_candidate = split_end_master + after_shift

    length_master = split_end_master - split_start_master
    length_candidate = split_end_candidate - split_start_candidate

    if length_candidate > length_master:
        net_kind = "addition"
    elif length_candidate < length_master:
        net_kind = "deletion"
    else:
        net_kind = "still_image" if length_master == 0 else "ordinary"

    delta_frames = length_candidate - length_master  # + = candidate holds more
    plumbing_ok, plumbing_evidence = _check_step_plumbing(
        delta_frames, frame_ms, step_ms, quantum_ms)
    if step_ms is None or quantum_ms is None:
        return {"declined": True, "reason": "anchor_step_unavailable",
               "master_seed_count": len(master_cuts_seeds),
               "candidate_seed_count": len(candidate_cuts_seeds),
               "evidence": plumbing_evidence}
    if not plumbing_ok and not accept_step_disagreement:
        # Fires only on a plumbing or units bug, never on content.
        return {"declined": True, "reason": "anchor_step_inconsistent",
               "master_seed_count": len(master_cuts_seeds),
               "candidate_seed_count": len(candidate_cuts_seeds),
               "evidence": plumbing_evidence}

    return {
        "declined": False,
        "step_plumbing_ok": plumbing_ok,
        "step_plumbing_evidence": plumbing_evidence,
        "grid": {"num": fps_num, "den": fps_den},
        "method": "scene_anchor_bidirectional",
        "anchor_a_frame": anchor_a,
        "anchor_b_frame": anchor_b,
        "master_start_frame": split_start_master,
        "master_end_frame": split_end_master,
        "candidate_start_frame": split_start_candidate,
        "candidate_end_frame": split_end_candidate,
        "net_kind": net_kind,
        "frames_to_cut": max(0, length_candidate - length_master),
        "frames_to_fill": max(0, length_master - length_candidate),
        "master_seed_count": len(master_cuts_seeds),
        "candidate_seed_count": len(candidate_cuts_seeds),
        "sweep_crossed": sweep_crossed,
        "pre_collapse_start_master": pre_collapse_start_master,
        "pre_collapse_end_master": pre_collapse_end_master,
        "unmatched_span_matches_before": span_counts["before"],
        "unmatched_span_matches_after": span_counts["after"],
        "forward_walk_frames": pre_collapse_start_master - anchor_a,
        "backward_walk_frames": anchor_b - pre_collapse_end_master,
        "before_shift_frames": before_shift,
        "after_shift_frames": after_shift,
        "nominal_before_shift_frames": nominal_before_shift,
        "nominal_after_shift_frames": nominal_after_shift,
        "anchor_a_n_frames": anchor_a_n_frames,
        "anchor_b_n_frames": anchor_b_n_frames,
        "derived_ms": {
            "master_start_ms": f"{round(float(_exact_ms_from_frame(split_start_master, fps_num, fps_den)), 2)}",
            "master_end_ms": f"{round(float(_exact_ms_from_frame(split_end_master, fps_num, fps_den)), 2)}",
        },
        "evidence": (f"anchor_a={anchor_a} anchor_a_n_frames={anchor_a_n_frames} "
                    f"anchor_b={anchor_b} anchor_b_n_frames={anchor_b_n_frames} "
                    f"master_cuts={len(master_cuts_seeds)} "
                    f"master_detector_failed={master_cuts_failed} "
                    f"candidate_cuts={len(candidate_cuts_seeds)} "
                    f"candidate_detector_failed={candidate_cuts_failed} "
                    f"sweep_crossed={sweep_crossed}"
                    + (f" pre_collapse_forward={pre_collapse_start_master} "
                       f"pre_collapse_backward={pre_collapse_end_master}"
                       if sweep_crossed else "")
                    + f" length_master={length_master} "
                    f"length_candidate={length_candidate} "
                    f"{plumbing_evidence}"),
    }


# ===========================================================================
# Edge brackets: one anchor, then a pHash walk to the boundary
# ===========================================================================
# The two-anchor protocol cannot work at an edge: at the head, Anchor A's seeds collapse
# to frame ~0 and backward validation reads negative indices (the tail mirrors this).


def _probe_video_geometry(path):
    '''Probe the file's coded width, height and pixel aspect with ffprobe.

    An unknown sample aspect ratio ("0:1") is read as square pixels; the value only feeds
    the aspect tolerance comparison.

    Returns:
        (dict with width, height, sar, aspect; None) on success, (None, reason) otherwise.
    '''
    try:
        cmd = [tools.software["ffprobe"], "-v", "error",
               "-select_streams", "v:0",
               "-show_entries", "stream=width,height,sample_aspect_ratio",
               "-of", "default=noprint_wrappers=1:nokey=1", path]
    except KeyError:
        return None, "ffprobe_not_configured"
    tools.dev_log(f"scene_anchor: _probe_video_geometry calling ffprobe "
                  f"file={path}\n")
    try:
        with repair_log.announced("scene_anchor", "ffprobe", path) as call:
            stdout, stderror, exit_code = tools.launch_cmdExt_with_timeout_reload(
                cmd, max_restart=3, timeout=60)
            call["exit"] = exit_code
    except Exception as exc:
        return None, f"ffprobe_raised:{type(exc).__name__}"
    if exit_code != 0:
        return None, f"ffprobe_exit:{exit_code}"
    lines = [ln.strip() for ln in
             stdout.decode("utf-8", "replace").strip().splitlines()]
    if len(lines) < 2:
        return None, f"unparseable_geometry:{lines!r}"
    try:
        width, height = int(lines[0]), int(lines[1])
    except (TypeError, ValueError):
        return None, f"unparseable_geometry:{lines[:2]!r}"
    if width <= 0 or height <= 0:
        return None, f"non_positive_geometry:{width}x{height}"
    sar = Fraction(1, 1)
    if len(lines) > 2 and lines[2] not in ("", "N/A", "0:1"):
        try:
            num, den = lines[2].split(":")
            parsed = Fraction(int(num), int(den))
            if parsed > 0:
                sar = parsed
        except (TypeError, ValueError, ZeroDivisionError):
            # Assume square pixels: SAR only feeds a ratio comparison with 2 % tolerance.
            sar = Fraction(1, 1)
    return {"width": width, "height": height, "sar": sar,
            "aspect": Fraction(width, height) * sar}, None


def _even(value):
    '''Round a crop offset down to an even, non-negative value (required by yuv420).'''
    return max(0, int(value) - (int(value) % 2))


def _resolve_geometry(master_path, candidate_path):
    '''Make the two files' pictures comparable before any anchor or walk.

    A geometry mismatch would read as a sustained mismatch indistinguishable from a real
    boundary. When the display aspects differ beyond EDGE_GEOMETRY_ASPECT_TOLERANCE, the
    file with the smaller aspect (the one carrying bars) is centre-cropped to the larger.

    Returns:
        (geometry_dict, crop_filters, reason): crop_filters maps a path to its ffmpeg crop
        (empty when none is needed); reason is set when the pair cannot be reconciled.
    '''
    m_geom, m_reason = _probe_video_geometry(master_path)
    c_geom, c_reason = _probe_video_geometry(candidate_path)
    if m_geom is None or c_geom is None:
        return ({"master": None, "candidate": None, "normalised": False,
                 "crop": None},
                {},
                f"geometry_unmeasured master={m_reason} candidate={c_reason}")

    def _label(g):
        return f"{g['width']}x{g['height']}"

    base = {"master": _label(m_geom), "candidate": _label(c_geom),
            "master_aspect": f"{float(m_geom['aspect']):.4f}",
            "candidate_aspect": f"{float(c_geom['aspect']):.4f}",
            "normalised": False, "crop": None}

    if (m_geom["width"], m_geom["height"], m_geom["sar"]) == \
       (c_geom["width"], c_geom["height"], c_geom["sar"]):
        base["verdict"] = "identical"
        return base, {}, None

    a_m, a_c = m_geom["aspect"], c_geom["aspect"]
    spread = abs(a_m - a_c) / max(a_m, a_c)
    if spread <= Fraction(EDGE_GEOMETRY_ASPECT_TOLERANCE).limit_denominator(10 ** 6):
        # Same picture with different pixel counts; cropping would only degrade matching.
        base["verdict"] = f"comparable aspect_spread={float(spread):.4f}"
        return base, {}, None

    # Crop only the file carrying bars, to the other's aspect, centred.
    if a_m < a_c:
        bar_path, bar_geom, target = master_path, m_geom, a_c
        bar_side = "master"
    else:
        bar_path, bar_geom, target = candidate_path, c_geom, a_m
        bar_side = "candidate"

    width, height, sar = bar_geom["width"], bar_geom["height"], bar_geom["sar"]
    # Letterbox: the height is in excess; otherwise pillarbox: the width is.
    new_h = int(round(Fraction(width) * sar / target))
    if 0 < new_h < height:
        crop_w, crop_h = width, new_h
        crop_x, crop_y = 0, _even((height - new_h) // 2)
    else:
        new_w = int(round(Fraction(height) * target / sar))
        if not (0 < new_w < width):
            return (base, {},
                    f"aspect_spread={float(spread):.4f} exceeds "
                    f"{EDGE_GEOMETRY_ASPECT_TOLERANCE} and no centred crop of "
                    f"{bar_side} {_label(bar_geom)} reaches "
                    f"{float(target):.4f}")
        crop_w, crop_h = new_w, height
        crop_x, crop_y = _even((width - new_w) // 2), 0

    # Even-offset rounding and integer sizes can move the aspect: verify the result.
    achieved = Fraction(crop_w, crop_h) * sar
    if abs(achieved - target) / max(achieved, target) > \
            Fraction(EDGE_GEOMETRY_ASPECT_TOLERANCE).limit_denominator(10 ** 6):
        return (base, {},
                f"centred crop {crop_w}:{crop_h}:{crop_x}:{crop_y} of "
                f"{bar_side} reaches aspect {float(achieved):.4f}, not "
                f"{float(target):.4f}")

    crop = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"
    base["normalised"] = True
    base["crop"] = f"{bar_side}:{crop}"
    base["verdict"] = (f"normalised aspect_spread={float(spread):.4f} "
                       f"{bar_side} {_label(bar_geom)} -> {crop_w}x{crop_h}")
    return base, {bar_path: crop}, None


class _ChunkedFrames:
    '''One side of the edge walk's frame supply, read in chunks, indexed by master frame.

    get() distinguishes a chunk boundary (fetch the next chunk) from the file's end, so a
    chunk edge never reads as candidate exhaustion. The declared duration is only an upper
    bound (containers may declare more than the video stream holds); the end of the file is
    confirmed by two empty decodes from different seek points.
    '''

    def __init__(self, comparer, path, side, fps_num, fps_den,
                 declared_last_frame, debug=False,
                 initial_base=None, initial_hashes=None,
                 chunk_seconds=EDGE_WALK_CHUNK_SECONDS):
        self.comparer = comparer
        self.path = path
        self.side = side
        self.fps_num = int(fps_num)
        self.fps_den = int(fps_den)
        self.frame_ms = 1000.0 * self.fps_den / self.fps_num
        self.chunk_frames = max(
            MIN_VALIDATION_FRAMES,
            int(round(chunk_seconds * 1000.0 / self.frame_ms)))
        self.declared_last_frame = declared_last_frame
        self.debug = debug
        # Seeding with the anchor window's hashes is required for correctness: a fresh
        # extraction may label the same picture one index apart, invalidating the shift.
        self.base = initial_base if initial_hashes else None
        self.hashes = initial_hashes if initial_hashes else frame_hash.FrameHashes.empty(colour=True)
        self.chunks_read = 0
        self.seam_deltas = []
        self.file_end_source = None

    def _seconds(self, frame):
        return max(0.0, frame * self.frame_ms / 1000.0)

    def _read(self, start_frame, n_frames):
        base, hashes = _extract_hashes(
            self.comparer, self.path, self._seconds(start_frame),
            n_frames * self.frame_ms / 1000.0)
        return base, hashes

    def _seam_delta(self, new_base, new_hashes):
        '''Find the label correction that makes a new chunk agree with the held one.

        The held overlap is aligned against the new chunk over
        +/-EDGE_WALK_SEAM_SEARCH_FRAMES. A static overlap (every delta shows the same
        picture) keeps the decoded labels (delta 0).

        Returns:
            (delta, content distance at it), or (None, evidence) when the overlap shows
            other content or cannot be placed.
        '''
        # The overlap is computed: a forward read starts inside the held chunk, a backward
        # read ends inside it.
        lo = max(self.base, new_base)
        hi = min(self.base + len(self.hashes), new_base + len(new_hashes))
        if hi - lo > EDGE_WALK_CHUNK_OVERLAP_FRAMES:
            if new_base > self.base:
                hi = lo + EDGE_WALK_CHUNK_OVERLAP_FRAMES
            else:
                lo = hi - EDGE_WALK_CHUNK_OVERLAP_FRAMES
        if hi - lo < frame_hash.ALIGN_MIN_FRAMES:
            return None, f"overlap={hi - lo}"
        held = self.hashes[lo - self.base:hi - self.base]
        reach = EDGE_WALK_SEAM_SEARCH_FRAMES
        alignment = frame_hash.align(held, new_hashes, range(-reach, reach + 1),
                                     start=lo - new_base)
        delta = alignment.lag if alignment.ok else (0 if alignment.best is not None else None)
        if delta is None:
            return None, alignment.reason
        content = frame_hash.window_content_distance(held, new_hashes, lo - new_base + delta)
        if not content <= frame_hash.SAME_CONTENT_MAX:
            return None, f"{alignment.reason} content={content:.4f}"
        return delta, round(content, 4)

    def get(self, frame):
        '''Return (hash, "ok"), (None, "file_end") or (None, "unreadable") for a master frame.'''
        if frame < 0:
            self.file_end_source = self.file_end_source or "before_frame_zero"
            return None, "file_end"
        if self.declared_last_frame is not None and frame > self.declared_last_frame:
            self.file_end_source = self.file_end_source or "declared_duration"
            return None, "file_end"
        if self.base is not None and 0 <= frame - self.base < len(self.hashes):
            return self.hashes[frame - self.base], "ok"

        # Read the next chunk so it overlaps the held one by EDGE_WALK_CHUNK_OVERLAP_FRAMES,
        # letting `_seam_delta` re-align labels across the seam.
        want_overlap = self.base is not None
        if not want_overlap:
            read_start = max(0, frame)
            read_frames = self.chunk_frames
        elif frame > self.base:
            held_hi = self.base + len(self.hashes) - 1
            read_start = max(0, held_hi - EDGE_WALK_CHUNK_OVERLAP_FRAMES + 1)
            read_frames = self.chunk_frames + EDGE_WALK_CHUNK_OVERLAP_FRAMES
        else:
            read_start = max(0, self.base - self.chunk_frames)
            read_frames = (self.base - read_start) + EDGE_WALK_CHUNK_OVERLAP_FRAMES
        new_base, new_hashes = self._read(read_start, read_frames)
        self.chunks_read += 1
        if not new_hashes:
            # One empty read may be a seek artefact; confirm from another seek point.
            confirm_base, confirm_hashes = self._read(frame, self.chunk_frames)
            if not confirm_hashes:
                self.file_end_source = self.file_end_source or "decode_empty_twice"
                return None, "file_end"
            new_base, new_hashes = confirm_base, confirm_hashes
            want_overlap = False

        if want_overlap:
            delta, evidence = self._seam_delta(new_base, new_hashes)
            if delta is None:
                tools.log_line(
                    f"scene_anchor: edge_walk_seam side={self.side} "
                    f"read_start={read_start} re_established=False "
                    f"evidence={evidence}\n")
                return None, "unreadable"
            self.seam_deltas.append(delta)
            new_base += delta
            tools.log_line(
                f"scene_anchor: edge_walk_seam side={self.side} "
                f"read_start={read_start} re_established=True delta={delta} "
                f"content={evidence}\n")

        self.base, self.hashes = new_base, new_hashes
        if 0 <= frame - self.base < len(self.hashes):
            return self.hashes[frame - self.base], "ok"
        if frame >= self.base:
            # Short chunk: confirm with a second read before declaring the file's end.
            confirm_base, confirm_hashes = self._read(frame, self.chunk_frames)
            if confirm_hashes and 0 <= frame - confirm_base < len(confirm_hashes):
                self.base, self.hashes = confirm_base, confirm_hashes
                return self.hashes[frame - self.base], "ok"
            self.file_end_source = self.file_end_source or "decode_short_chunk"
            return None, "file_end"
        return None, "unreadable"


def _edge_walk(master_frames, candidate_frames, first_confirmed, shift_frames,
               edge, n_sustained):
    '''Walk outward from a validated anchor, one master frame at a time, to the edge.

    Compares master and candidate under the anchor's shift and stops on
    "sustained_mismatch" (n_sustained consecutive mismatches), "master_exhausted" or
    "candidate_exhausted". first_confirmed is the outermost frame validation already
    proved (anchor at a head, anchor - 1 at a tail).

    Returns:
        dict with boundary_frame (last master frame confirmed matching) and walk statistics,
        or with reason "edge_walk_unreadable" when a chunk could not be read.
    '''
    step = -1 if edge == "head" else 1
    boundary_frame = first_confirmed
    walked = 0
    mismatch_run = 0
    max_mismatch_run = 0
    termination = None
    unreadable_side = None
    frame = first_confirmed

    while True:
        frame += step
        m_hash, m_state = master_frames.get(frame)
        if m_state == "file_end":
            termination = "master_exhausted"
            break
        if m_state != "ok":
            unreadable_side = "master"
            break
        c_hash, c_state = candidate_frames.get(frame + shift_frames)
        if c_state == "file_end":
            termination = "candidate_exhausted"
            break
        if c_state != "ok":
            unreadable_side = "candidate"
            break
        walked += 1
        if frame_hash.same_picture(m_hash, c_hash)[0]:
            boundary_frame = frame
            mismatch_run = 0
        else:
            mismatch_run += 1
            if mismatch_run > max_mismatch_run:
                max_mismatch_run = mismatch_run
            if mismatch_run >= n_sustained:
                termination = "sustained_mismatch"
                break

    if termination is None:
        return {"reason": "edge_walk_unreadable",
                "evidence": (f"side={unreadable_side} frame={frame} "
                             f"walked_frames={walked} "
                             f"master_chunks={master_frames.chunks_read} "
                             f"candidate_chunks={candidate_frames.chunks_read}")}
    return {"reason": None,
            "boundary_frame": boundary_frame,
            "walked_frames": walked,
            "mismatch_run": mismatch_run,
            "max_mismatch_run": max_mismatch_run,
            "termination": termination,
            "master_end_source": master_frames.file_end_source,
            "candidate_end_source": candidate_frames.file_end_source,
            "master_chunks": master_frames.chunks_read,
            "candidate_chunks": candidate_frames.chunks_read,
            "master_seam_deltas": list(master_frames.seam_deltas),
            "candidate_seam_deltas": list(candidate_frames.seam_deltas)}


def locate_edge_boundary(master_path, candidate_path, fps_num, fps_den,
                         bracket_low_ms, bracket_high_ms, offset_ms, edge,
                         master_timeline_ms, candidate_duration_ms,
                         known_match_ms=None, step_ms=None, quantum_ms=None,
                         scene_search_window_sec=None, debug=False,
                         candidate_time_scale=None,
                         shift_search_frames=EDGE_SHIFT_SEARCH_FRAMES, deadline=None):
    '''Locate an edge (head or tail) boundary: one anchor on the common side, then a walk.

    Args:
        edge: "head" or "tail", as classified by the caller from the bracket's edge field.
        master_timeline_ms: the master's own timeline length (not min(master, candidate)).
        offset_ms: the adjacent segment's offset (candidate_time = master_time + offset);
            it nominates the shift, which the anchor resolves within shift_search_frames.
        candidate_time_scale: optional rate relation, as in locate_scene_anchors.
        shift_search_frames: shift search half-width; callers with offsets quantised to a
            fingerprint hop pass a wider bound.
        deadline: optional time.monotonic() limit checked between rungs.

    Returns:
        The same dict shape as locate_scene_anchors (declined True or False).
    '''
    fps_num = int(fps_num)
    fps_den = int(fps_den)
    if fps_num <= 0 or fps_den <= 0:
        return {"declined": True, "reason": "grid_unmeasured",
                "evidence": f"fps_num={fps_num} fps_den={fps_den}"}
    if edge not in ("head", "tail"):
        return {"declined": True, "reason": "empty_bracket",
                "evidence": f"edge={edge!r} is neither 'head' nor 'tail'"}
    if bracket_high_ms <= bracket_low_ms:
        return {"declined": True, "reason": "empty_bracket",
                "evidence": f"[{bracket_low_ms},{bracket_high_ms}] ms edge={edge}"}

    geometry, crop_filters, geometry_reason = _resolve_geometry(
        master_path, candidate_path)
    tools.log_line(
        f"scene_anchor: edge_geometry edge={edge} "
        f"master={geometry.get('master')} candidate={geometry.get('candidate')} "
        f"normalised={geometry.get('normalised')} crop={geometry.get('crop')} "
        f"verdict={geometry.get('verdict')} reason={geometry_reason}\n")
    if geometry_reason is not None:
        return {"declined": True, "reason": "edge_geometry_unreconciled",
                "edge": edge, "geometry": geometry,
                "evidence": geometry_reason}

    window_sec = (scene_search_window_sec if scene_search_window_sec is not None
                  else _scene_anchor_config())

    result = None
    rung_window_sec = window_sec
    rung_cd_threshold = CONTENT_DETECTOR_THRESHOLD_LADDER[0]
    for rung in range(WINDOW_LADDER_MAX_RUNGS):
        if deadline is not None and time.monotonic() > deadline:
            return {"declined": True, "reason": "hole_budget_exceeded", "edge": edge,
                    "geometry": geometry,
                    "evidence": f"rungs_run={rung} last_reason="
                                f"{(result or {}).get('reason')}"}
        rung_window_sec = (window_sec if window_sec is None
                           else window_sec * (WINDOW_LADDER_GROWTH_FACTOR ** rung))
        rung_cd_threshold = CONTENT_DETECTOR_THRESHOLD_LADDER[
            min(rung, len(CONTENT_DETECTOR_THRESHOLD_LADDER) - 1)]
        try:
            result = _locate_edge_boundary_at_window(
                master_path, candidate_path, fps_num, fps_den,
                bracket_low_ms, bracket_high_ms, offset_ms, edge,
                master_timeline_ms, candidate_duration_ms, rung_window_sec,
                crop_filters, geometry,
                content_detector_threshold=rung_cd_threshold, debug=debug,
                candidate_time_scale=candidate_time_scale,
                shift_search_frames=shift_search_frames)
        except tools.decoder_timeout as error:
            return {"declined": True, "reason": "decoder_timeout", "edge": edge,
                    "geometry": geometry, "evidence": f"rung={rung} {error}"}
        matched = not result["declined"]
        tools.log_line(
            f"scene_anchor: edge_window_ladder_rung edge={edge} rung={rung} "
            f"window_sec={rung_window_sec} cd_threshold={rung_cd_threshold} "
            f"master_seed_count={result.get('master_seed_count')} "
            f"candidate_seed_count={result.get('candidate_seed_count')} "
            f"matched={matched} reason={result.get('reason')} "
            f"evidence={result.get('evidence')}\n")
        if matched or result["reason"] not in EDGE_WINDOW_LADDER_RETRYABLE_REASONS:
            break
    else:
        result = {"declined": True, "reason": "search_window_ceiling_reached",
                  "edge": edge, "geometry": geometry,
                  "master_seed_count": result.get("master_seed_count"),
                  "candidate_seed_count": result.get("candidate_seed_count"),
                  "evidence": f"rungs_tried={WINDOW_LADDER_MAX_RUNGS} "
                              f"base_window_sec={window_sec} "
                              f"final_window_sec={rung_window_sec} "
                              f"final_cd_threshold={rung_cd_threshold} "
                              f"last_reason={result.get('reason')} "
                              f"last_evidence={result.get('evidence')}"}

    # With a normalising crop, relabel the decline as a geometry failure only when scene-cut
    # seeds were offered and none validated (the shape a wrong crop produces). A validated
    # but non-distinctive seed, or no cuts at all, leaves the crop unrefuted.
    if result["declined"] and geometry.get("normalised"):
        reason = result.get("reason")
        if reason == "search_window_ceiling_reached":
            evidence = result.get("evidence") or ""
            last_reason = next((token for token in ("edge_anchor_not_established",
                                                    "edge_anchor_uninformative")
                                if f"last_reason={token}" in evidence), None)
        else:
            last_reason = reason
        cut_seeds = ((result.get("master_seed_count") or 0)
                     + (result.get("candidate_seed_count") or 0))
        if last_reason == "edge_anchor_not_established" and cut_seeds > 0:
            result = {**result, "reason": "edge_geometry_unreconciled",
                      "geometry": geometry,
                      "evidence": (f"normalisation {geometry.get('crop')} applied, "
                                   f"{cut_seeds} scene-cut seed(s) offered and none "
                                   f"validated under it: {reason} "
                                   f"{result.get('evidence')}")}
        elif last_reason in ("edge_anchor_not_established",
                             "edge_anchor_uninformative") and cut_seeds == 0:
            result = {**result,
                      "evidence": (f"no_scene_cut_in_reach (crop {geometry.get('crop')} "
                                   f"not refuted -- "
                                   + ("a seed validated under it"
                                      if last_reason == "edge_anchor_uninformative"
                                      else "untested")
                                   + f"): {result.get('evidence')}")}
    return result


def _locate_edge_boundary_at_window(master_path, candidate_path, fps_num, fps_den,
                                    bracket_low_ms, bracket_high_ms, offset_ms,
                                    edge, master_timeline_ms, candidate_duration_ms,
                                    window_sec, crop_filters, geometry,
                                    content_detector_threshold=CONTENT_DETECTOR_THRESHOLD_DEFAULT,
                                    debug=False, candidate_time_scale=None,
                                    shift_search_frames=EDGE_SHIFT_SEARCH_FRAMES):
    '''Run one rung of locate_edge_boundary's ladder: seat the common-side anchor, then walk.

    Returns:
        dict with declined True or False, with the same decline reasons as
        _locate_scene_anchors_at_window.
    '''
    frame_ms = 1000.0 * fps_den / fps_num

    if window_sec is None or window_sec <= 0:
        return {"declined": True, "reason": "search_window_unviable",
                "edge": edge, "geometry": geometry,
                "evidence": f"scene_search_window_sec={window_sec}"}
    window_frames = int(round((window_sec * 1000.0) / frame_ms))
    if window_frames < MIN_VALIDATION_FRAMES:
        return {"declined": True, "reason": "search_window_too_narrow",
                "edge": edge, "geometry": geometry,
                "evidence": f"scene_search_window_sec={window_sec} -> "
                            f"{window_frames} frames, needs >= "
                            f"{MIN_VALIDATION_FRAMES}"}

    comparer = FrameComparer(master_path, candidate_path,
                             bracket_low_ms / 1000.0, bracket_high_ms / 1000.0,
                             fps_num, fps_den, debug=debug,
                             crop_filters=crop_filters,
                             time_scales=({candidate_path: candidate_time_scale}
                                          if candidate_time_scale is not None
                                          else None))
    m_bracket_first = comparer._frame_index(bracket_low_ms / 1000.0)
    m_bracket_last = comparer._frame_index(bracket_high_ms / 1000.0)
    shift_frames = _nominal_shift_frames(offset_ms, fps_num, fps_den)

    # The window also covers the outer side so the lags scanned around the shift stay
    # readable; only the seed filter restricts to the common side.
    m_win_start = max(0, m_bracket_first - window_frames)
    m_win_end = m_bracket_last + window_frames
    candidate_margin_frames = CANDIDATE_SEED_MARGIN_MULTIPLIER * window_frames
    c_win_start = max(0, m_win_start - candidate_margin_frames + shift_frames)
    c_win_end = m_win_end + candidate_margin_frames + shift_frames
    # Clamp both windows to their files: a negative length would decode the whole file.
    m_win_start, m_win_end = _clamp_to_file(
        m_win_start, m_win_end, float(master_timeline_ms) / 1000.0, fps_num, fps_den)
    c_win_start, c_win_end = _clamp_to_file(
        c_win_start, c_win_end,
        None if candidate_duration_ms is None else float(candidate_duration_ms) / 1000.0,
        fps_num, fps_den)
    if m_win_end <= m_win_start or c_win_end <= c_win_start:
        return {"declined": True, "reason": "candidate_window_empty", "edge": edge,
                "evidence": f"master_window=[{m_win_start},{m_win_end}) "
                            f"candidate_window=[{c_win_start},{c_win_end}) "
                            f"shift={shift_frames}"}

    master_rate = Fraction(fps_num, fps_den)
    m_win_start_sec = Fraction(m_win_start * fps_den, fps_num)
    m_win_span_sec = Fraction((m_win_end - m_win_start) * fps_den, fps_num)
    c_win_start_sec = Fraction(c_win_start * fps_den, fps_num)
    c_win_span_sec = Fraction((c_win_end - c_win_start) * fps_den, fps_num)

    candidate_rate, candidate_rate_reason = _probe_frame_rate(candidate_path)
    # Speed-changed candidate: scan at its corrected rate, as on the interior path.
    if candidate_rate is not None and candidate_time_scale is not None:
        candidate_rate = candidate_rate / Fraction(candidate_time_scale)

    m_scan_start = _frames_at_rate(m_win_start_sec, master_rate)
    m_scan_frames = _frames_at_rate(m_win_span_sec, master_rate)
    if candidate_rate is None:
        c_scan_start = c_scan_frames = None
    else:
        c_scan_start = _frames_at_rate(c_win_start_sec, candidate_rate)
        c_scan_frames = _frames_at_rate(c_win_span_sec, candidate_rate)
    tools.dev_log(
        f"scene_anchor: edge_scan_window_conversion edge={edge} "
        f"master_rate={master_rate.numerator}/{master_rate.denominator} "
        f"candidate_rate="
        f"{'unmeasured:' + str(candidate_rate_reason) if candidate_rate is None else str(candidate_rate.numerator) + '/' + str(candidate_rate.denominator)} "
        f"master_scan=[{m_scan_start},+{m_scan_frames}) "
        f"candidate_scan=[{c_scan_start},+{c_scan_frames})\n")

    m_base, m_hashes = _extract_hashes(comparer, master_path,
                                       float(m_win_start_sec), float(m_win_span_sec))
    c_base, c_hashes = _extract_hashes(comparer, candidate_path,
                                       float(c_win_start_sec), float(c_win_span_sec))
    if not m_hashes or not c_hashes:
        return {"declined": True, "reason": "frames_unextractable",
                "edge": edge, "geometry": geometry,
                "evidence": f"master_frames={len(m_hashes)} "
                            f"candidate_frames={len(c_hashes)}"}

    master_cuts, master_cuts_failed = _scene_cut_frames(
        master_path, m_scan_start, m_scan_frames, content_detector_threshold, debug)
    if candidate_rate is None:
        candidate_cuts = None
        candidate_cuts_failed = f"candidate_grid_unmeasured:{candidate_rate_reason}"
    else:
        candidate_cuts, candidate_cuts_failed = _scene_cut_frames(
            candidate_path, c_scan_start, c_scan_frames,
            content_detector_threshold, debug)
    master_cuts_seeds = master_cuts or []
    candidate_cuts_seeds = [_frame_on_grid(f, candidate_rate, master_rate)
                            for f in (candidate_cuts or [])]

    # Seeds on the common side, nearest first. File edges are often static (black, logo,
    # fade), so the bracket-edge seed tends to be non-distinctive and real scene cuts are
    # needed.
    if edge == "head":
        seeds = sorted(
            {m_bracket_last}
            | {f for f in master_cuts_seeds if f >= m_bracket_last}
            | {f - shift_frames for f in candidate_cuts_seeds
               if f - shift_frames >= m_bracket_last})
        direction = "forward"
        anchor_side = "B"
    else:
        seeds = sorted(
            {m_bracket_first}
            | {f for f in master_cuts_seeds if f <= m_bracket_first}
            | {f - shift_frames for f in candidate_cuts_seeds
               if f - shift_frames <= m_bracket_first},
            reverse=True)
        direction = "backward"
        anchor_side = "A"

    nominal_shift_frames = shift_frames
    boundaries = sorted(set(master_cuts_seeds)
                        | {f - shift_frames for f in candidate_cuts_seeds})
    anchor, shift_frames, anchor_n_frames, anchor_reason, _ = _anchor_search(
        m_hashes, m_base, c_hashes, c_base, seeds, nominal_shift_frames,
        direction, boundaries, search_frames=shift_search_frames)

    if anchor is None:
        payload = {"declined": True, "edge": edge, "geometry": geometry,
                   "master_seed_count": len(master_cuts_seeds),
                   "candidate_seed_count": len(candidate_cuts_seeds)}
        if anchor_reason:
            return {**payload, "reason": "edge_anchor_uninformative",
                    "evidence": f"edge={edge} side={anchor_side} "
                                f"seeds={len(seeds)} "
                                f"nominal_shift={nominal_shift_frames} "
                                f"reason={anchor_reason} "
                                f"n_frames={anchor_n_frames}"}
        return {**payload, "reason": "edge_anchor_not_established",
                "evidence": f"edge={edge} side={anchor_side} "
                            f"seeds={len(seeds)} "
                            f"nominal_shift={nominal_shift_frames} "
                            f"master_cuts={len(master_cuts_seeds)} "
                            f"master_detector_failed={master_cuts_failed} "
                            f"candidate_cuts={len(candidate_cuts_seeds)} "
                            f"candidate_detector_failed={candidate_cuts_failed}"}

    # The master is bounded by its timeline; the candidate's declared length is only a
    # ceiling (see _ChunkedFrames).
    master_last_frame = _frames_at_rate(
        Fraction(str(master_timeline_ms)) / 1000, master_rate) - 1
    candidate_last_frame_ceiling = None
    if candidate_duration_ms is not None:
        candidate_last_frame_ceiling = _frames_at_rate(
            Fraction(str(candidate_duration_ms)) / 1000, master_rate) - 1

    master_frames = _ChunkedFrames(comparer, master_path, "master",
                                   fps_num, fps_den, master_last_frame, debug,
                                   initial_base=m_base, initial_hashes=m_hashes)
    candidate_frames = _ChunkedFrames(comparer, candidate_path, "candidate",
                                      fps_num, fps_den,
                                      candidate_last_frame_ceiling, debug,
                                      initial_base=c_base, initial_hashes=c_hashes)

    # Backward validation covers [seed-n, seed): at a tail the outermost proven frame is
    # seed - 1.
    first_confirmed = anchor if edge == "head" else anchor - 1
    walk = _edge_walk(master_frames, candidate_frames, first_confirmed,
                      shift_frames, edge, EDGE_WALK_SUSTAINED_MISMATCH_FRAMES)
    if walk["reason"] is not None:
        return {"declined": True, "reason": walk["reason"], "edge": edge,
                "geometry": geometry,
                "master_seed_count": len(master_cuts_seeds),
                "candidate_seed_count": len(candidate_cuts_seeds),
                "evidence": f"edge={edge} anchor_frame={anchor} "
                            f"shift={shift_frames} {walk['evidence']}"}

    boundary_frame = walk["boundary_frame"]
    termination = walk["termination"]

    # Only candidate exhaustion means a master addition; its length is the count of master
    # frames beyond the boundary.
    addition_frames = None
    if termination == "candidate_exhausted":
        addition_frames = (boundary_frame if edge == "head"
                           else master_last_frame - boundary_frame)
        addition_frames = max(0, addition_frames)
    addition_ms = (None if addition_frames is None
                   else str(_exact_ms_from_frame(addition_frames, fps_num, fps_den)))

    if termination == "candidate_exhausted":
        net_kind = "master_addition"
    elif termination == "master_exhausted":
        net_kind = "candidate_excess_trimmed"
    else:
        # The candidate has content there but it disagrees: master content replaces it.
        net_kind = "master_replacement"

    evidence = (f"edge={edge} anchor={anchor} anchor_side={anchor_side} "
                f"anchor_n_frames={anchor_n_frames} shift={shift_frames} "
                f"nominal_shift={nominal_shift_frames} "
                f"boundary={boundary_frame} walked={walk['walked_frames']} "
                f"mismatch_run={walk['mismatch_run']} "
                f"max_mismatch_run={walk['max_mismatch_run']} "
                f"termination={termination} "
                f"master_end_source={walk['master_end_source']} "
                f"candidate_end_source={walk['candidate_end_source']} "
                f"master_chunks={walk['master_chunks']} "
                f"candidate_chunks={walk['candidate_chunks']} "
                f"master_seam_deltas={walk['master_seam_deltas']} "
                f"candidate_seam_deltas={walk['candidate_seam_deltas']} "
                f"master_last_frame={master_last_frame} "
                f"master_cuts={len(master_cuts_seeds)} "
                f"master_detector_failed={master_cuts_failed} "
                f"candidate_cuts={len(candidate_cuts_seeds)} "
                f"candidate_detector_failed={candidate_cuts_failed} "
                f"seeds={len(seeds)}")

    tools.log_line(
        f"scene_anchor: edge_walk edge={edge} anchor_frame={anchor} "
        f"anchor_n_frames={anchor_n_frames} anchor_side={anchor_side} "
        f"shift_frames={shift_frames} "
        f"nominal_shift_frames={nominal_shift_frames} "
        f"boundary_frame={boundary_frame} "
        f"walked_frames={walk['walked_frames']} "
        f"mismatch_run={walk['mismatch_run']} "
        f"max_mismatch_run={walk['max_mismatch_run']} "
        f"termination={termination} net_kind={net_kind} "
        f"addition_frames={addition_frames} addition_ms={addition_ms} "
        f"same_frame_max={frame_hash.SAME_FRAME_MAX} "
        f"n_sustained={EDGE_WALK_SUSTAINED_MISMATCH_FRAMES} "
        f"geometry_normalised={geometry.get('normalised')}\n")

    return {
        "declined": False,
        "method": "scene_anchor_single_edge",
        "edge": edge,
        "grid": {"num": fps_num, "den": fps_den},
        "anchor_frame": anchor,
        "anchor_n_frames": anchor_n_frames,
        "anchor_side": anchor_side,
        "shift_frames": shift_frames,
        "nominal_shift_frames": nominal_shift_frames,
        "boundary_frame": boundary_frame,
        "walked_frames": walk["walked_frames"],
        "mismatch_run": walk["mismatch_run"],
        "max_mismatch_run": walk["max_mismatch_run"],
        "termination": termination,
        "net_kind": net_kind,
        "addition_frames": addition_frames,
        "addition_ms": addition_ms,
        "master_last_frame": master_last_frame,
        "geometry": geometry,
        "master_seed_count": len(master_cuts_seeds),
        "candidate_seed_count": len(candidate_cuts_seeds),
        "derived_ms": {
            "boundary_ms": f"{round(float(_exact_ms_from_frame(boundary_frame, fps_num, fps_den)), 2)}",
        },
        "evidence": evidence,
    }
