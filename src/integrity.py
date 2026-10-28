"""Track integrity checks: strict decodes, silence maps and conversion stderr."""

import concurrent.futures
import math
from collections import OrderedDict
from decimal import Decimal
import os
import re
import subprocess
import threading
import time
import zlib

import numpy as np

import file_conformity as fc
import progress_watchdog

try:
    import tools
except Exception:          # the module stays importable without the pipeline's config
    tools = None

# ---------------------------------------------------------------- constants

STRICT_FLAGS = list(fc.STRICT_FLAGS)
THREADS = 2
VIDEO_WINDOWS = fc.INTEGRITY_WINDOWS
WINDOW_S = fc.INTEGRITY_WINDOW_S
# Error lines that are not corruption: muxer timestamp warnings and the demuxer's
# complaint when seeking into a file with no keyframe index.
IGNORED_LINE_SUBSTRINGS = tuple(fc.IGNORED_ERROR_SUBSTRINGS) + (
    "Non-monotonic DTS", "keyframes not correctly marked")

ENV_SR = 8000
SILENCE_BLOCK_S = 0.4
SILENCE_DB = fc.SILENCE_DB                    # dBFS RMS per 400 ms block
DIGITAL_DB = -90.0
SILENCE_MIN_S = 2.0
# A span map (`silence_map(span_ms=...)`) cuts the span into about SPAN_BLOCKS blocks.
SPAN_BLOCKS = 10
SPAN_BLOCK_MIN_S = 0.01
CLICK_S = fc.CLICK_S
CLICK_GAP_S = fc.ISLAND_GAP_S
CONTENT_DB = -50.0                            # "plays sound": 10 dB above the silence line
BAD_SILENCE_S = 5.0

FRAME_COUNT_TOL = 0.001
FRAME_COUNT_MIN_FRAMES = 2
FPS_TOL = 0.02

# Options every conversion should carry. `-xerror` is global; `-reinit_filter 0` is an input
# option and goes before the `-i` of the input whose audio is filtered.
CONVERSION_SAFE_OPTIONS = ["-xerror", "-reinit_filter", "0"]
FATAL_CONVERSION_PATTERNS = (
    "Non-monotonic DTS",               # muxer rewrote timestamps
    "Failed to compensate",            # swresample could not absorb a timestamp jump
    "Reconfiguring filter graph",      # audio parameters changed mid-stream
    "new coupling strategy",           # the corrupt E-AC-3 frame itself
    "Error while decoding",
    "frame CRC mismatch",              # the same frame under -err_detect crccheck
    "Error submitting packet to decoder",   # what -xerror stops on
    "Changing audio frame properties on the fly",   # -reinit_filter 0 refusing
)
# Fatal patterns on subtitle / video output streams concern cue order, not audio.
NON_AUDIO_STREAM_PREFIXES = ("[sost#", "[vost#")

_CACHE_MAX = 64
_probe_cache, _map_cache, _check_cache = OrderedDict(), OrderedDict(), OrderedDict()
_cache_lock = threading.Lock()

class IntegrityError(Exception):
    pass

def test_video_file_have_no_big_error(file_to_test):
    try:
        tools.launch_cmdExt_with_timeout_reload([tools.software["ffmpeg"], "-xerror", "-err_detect", "crccheck+bitstream+buffer",
                            "-analyzeduration", "1000M", "-probesize", "1000M", "-threads", str(tools.core_to_use),
                            "-i", file_to_test, "-map", "0:v?", "-map", "0:a?", "-map", "0:s?", "-c:s", "copy", "-f", "null", "-"], 2, 1200)
    except Exception as e:
        raise IntegrityError(f"{file_to_test} does not decode cleanly: {e}") from e

# ---------------------------------------------------------------- helpers

def _log(message, always=False):
    """Log one line through tools under the merge's log lock: worker threads call it."""
    line = f"integrity: {message}\n"
    if tools is None:
        return
    import mergeVideo
    with mergeVideo.errors_merge_lock:
        (tools.log_always if always else tools.dev_log)(line)


def _bin(name):
    """Path of an external tool, from tools.software only (a missing name raises KeyError)."""
    return tools.software[name]




def _path_of(video_obj):
    if isinstance(video_obj, (str, os.PathLike)):
        return os.fspath(video_obj)
    return video_obj.filePath


def _file_key(path):
    """Return the cache key (path, size, mtime) of a file."""
    st = os.stat(path)
    return (os.path.abspath(path), st.st_size, st.st_mtime_ns)


def probe(path):
    """Return the cached ffprobe result of a file."""
    key = _file_key(path)
    with _cache_lock:
        if key in _probe_cache:
            _probe_cache.move_to_end(key)
            return _probe_cache[key]
    info = fc.probe(path)
    with _cache_lock:
        if len(_probe_cache) >= _CACHE_MAX:
            _probe_cache.popitem(last=False)
        _probe_cache[key] = info
    return info


def resolve_stream(video_obj, stream_id, kind="stream_order"):
    """Return (path, ffprobe stream dict) for a stream identifier.

    Args:
        stream_id: stream index (int, str) or a track dict with 'StreamOrder'.
        kind: 'stream_order' (absolute index), 'audio_pos' (0-based among audio
            streams) or 'typeorder' (mediainfo's 1-based audio order).

    Raises:
        ValueError: unknown kind or no such stream.
    """
    path = _path_of(video_obj)
    streams = probe(path).get("streams", [])
    if isinstance(stream_id, dict):
        if "StreamOrder" not in stream_id:
            raise ValueError(f"track dict without 'StreamOrder': {sorted(stream_id)[:8]}")
        stream_id, kind = stream_id["StreamOrder"], "stream_order"
    try:
        n = int(str(stream_id).strip())
    except ValueError:
        raise ValueError(f"stream_id {stream_id!r} is not an integer stream index") from None
    if kind == "stream_order":
        hit = [s for s in streams if s.get("index") == n]
    elif kind in ("audio_pos", "typeorder"):
        audio = [s for s in streams if s.get("codec_type") == "audio"]
        pos = n if kind == "audio_pos" else n - 1
        hit = [audio[pos]] if 0 <= pos < len(audio) else []
    else:
        raise ValueError(f"kind must be 'stream_order', 'audio_pos' or 'typeorder', not {kind!r}")
    if not hit:
        raise ValueError(f"{path}: no stream {stream_id!r} (kind={kind})")
    return path, hit[0]


def _declared_end(info, s):
    """Return a stream's declared end: DURATION tag, stream duration, else format duration."""
    d = fc._parse_hms(fc._tag(s, "DURATION"))
    if d is None:
        d = fc._num(s.get("duration"))
    if d is None:
        d = fc._num(info.get("format", {}).get("duration"))
    return d


def _video_stream(info):
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic"):
            return s
    return None


def video_end_s(video_obj):
    """Return the video's end in seconds, or None without video.

    Uses the object's mediainfo video Duration, else the first video stream's
    declared end (tag DURATION first, since format.duration can be far off).
    """
    v = getattr(video_obj, "video", None)
    if isinstance(v, dict) and fc._num(v.get("Duration")):
        return float(v["Duration"])
    info = probe(_path_of(video_obj))
    vs = _video_stream(info)
    return _declared_end(info, vs) if vs else None


def _fr(x, nd=1):
    return "?" if x is None else f"{x:.{nd}f}"


def _run_capture(cmd):
    """Run a command and return (rc, stderr_text, stalled, seconds)."""
    t0 = time.time()
    try:
        p = progress_watchdog.run(cmd, "strict_decode", stdout=subprocess.DEVNULL,
                                  stdin=subprocess.DEVNULL)
        return p.returncode, p.stderr.decode("utf-8", "replace"), False, time.time() - t0
    except progress_watchdog.step_stalled as stall:
        return -9, str(stall), True, time.time() - t0


def _error_lines(stderr_text):
    return [l.strip() for l in stderr_text.splitlines()
            if l.strip() and not any(s in l for s in IGNORED_LINE_SUBSTRINGS)]


def decode_error_lines(stderr_text):
    """Return the ffmpeg stderr lines that count against a strict decode.

    Lines matching IGNORED_LINE_SUBSTRINGS are dropped.
    """
    return _error_lines(stderr_text or "")


# ---------------------------------------------------------------- strict decode

def _strict(path, index, start, dur, threads):
    """Strict-decode one stream (optionally one window) and return its result dict."""
    cmd = [_bin("ffmpeg"), "-nostdin", "-hide_banner", "-v", "error"] + STRICT_FLAGS + [
        "-threads", str(threads)]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}"]
    cmd += ["-i", path, "-map", f"0:{index}", "-f", "null", "-"]
    rc, err, stalled, sec = _run_capture(cmd)
    lines = [] if stalled else _error_lines(err)
    return dict(start=start, dur=dur, rc=rc, stalled=stalled, seconds=round(sec, 1),
                n_lines=len(lines), lines=lines[:20])


def track_check(video_obj, stream_id, full=None, *, kind="stream_order", threads=THREADS,
                windows=VIDEO_WINDOWS, seed=None, jobs=None):
    """Strict-decode one stream (audio whole, video by seeded windows; whole results cached) and return its numbers.

    `verdict` is 'sound', 'corrupt' or 'step_stalled' (never cached); `full` None picks the per-type default."""
    t0 = time.time()
    path, s = resolve_stream(video_obj, stream_id, kind)
    info = probe(path)
    idx, ctype = s["index"], s.get("codec_type")
    media = _declared_end(info, s) or 0.0
    whole = full if full is not None else (ctype != "video")
    key = _file_key(path) + (idx, "strict_full") if whole else None
    with _cache_lock:
        if key is not None and key in _check_cache:
            _check_cache.move_to_end(key)
            return dict(_check_cache[key], cached=True)
    starts = None
    if not whole:
        if seed is None:
            seed = fc.path_seed(path)
        starts = fc.integrity_windows(media, windows, seed, WINDOW_S)
    if starts is None:
        mode = "full"
        runs = [_strict(path, idx, None, None, threads)]
    else:
        mode = "sampled"
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs or len(starts))) as ex:
            runs = list(ex.map(lambda st: _strict(path, idx, st, WINDOW_S, threads), starts))
    failed = [r for r in runs if not r["stalled"] and (r["rc"] != 0 or r["n_lines"] > 0)]
    stalled = [r for r in runs if r["stalled"]]
    if failed:
        verdict, sound = "corrupt", False
    elif stalled:
        verdict, sound = "step_stalled", None
    else:
        verdict, sound = "sound", True
    first = failed[0] if failed else (runs[0] if runs else {})
    lines = [l for r in failed for l in r["lines"]][:20]
    res = dict(path=path, stream_index=idx, codec_type=ctype, codec=s.get("codec_name"),
               lang=fc._tag(s, "LANGUAGE") or "und", mode=mode, sound=sound, verdict=verdict,
               rc=first.get("rc"), error_lines=lines,
               n_error_lines=sum(r["n_lines"] for r in failed),
               windows=[{k: v for k, v in r.items() if k != "lines"} for r in runs],
               failed_starts=[r["start"] for r in failed if r["start"] is not None],
               seed=seed, media_s=media, cost_s=round(time.time() - t0, 1))
    msg = (f"track_check path={path} stream={idx} codec={res['codec']} lang={res['lang']} "
           f"mode={mode} verdict={verdict} rc={res['rc']} error_lines={res['n_error_lines']} "
           f"processes={len(runs)} failed={len(failed)} stalled={len(stalled)} seed={seed} "
           f"cost_s={res['cost_s']}" + (f" first=«{lines[0][:200]}»" if lines else ""))
    _log(msg, always=(verdict != "sound"))
    if key is not None and verdict != "step_stalled":
        with _cache_lock:
            if len(_check_cache) >= _CACHE_MAX:
                _check_cache.popitem(last=False)
            _check_cache[key] = res
    return res


# ---------------------------------------------------------------- silence maps

def _decode_blocks(path, index, timeline, threads, span_s=None, block_s=SILENCE_BLOCK_S):
    """Decode one stream (or `span_s`) to 8 kHz mono `block_s` mean-square blocks: (blocks, n_samples, rc, stderr_tail).

    'timestamps' fills gaps from packet times (aresample async=1); 'samples' concatenates, as mkvmerge re-times."""
    af = f"aresample={ENV_SR}:async=1" if timeline == "timestamps" else f"aresample={ENV_SR}"
    seek = ([] if span_s is None else
            ["-ss", f"{span_s[0]:.6f}", "-t", f"{span_s[1]:.6f}"])
    cmd = [_bin("ffmpeg"), "-nostdin", "-hide_banner", "-v", "error", "-threads", str(threads),
           *seek, "-i", path, "-map", f"0:{index}", "-af", af, "-ac", "1",
           "-c:a", "pcm_s16le", "-f", "s16le", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         stdin=subprocess.DEVNULL)
    err_chunks = []
    t_err = threading.Thread(target=lambda: err_chunks.append(p.stderr.read()), daemon=True)
    t_err.start()
    block = max(1, int(round(ENV_SR * block_s)))
    bbytes = block * 2
    pcm_buffer = bytearray((1 << 18) + bbytes)
    pcm_view = memoryview(pcm_buffer)
    pcm_filled = 0
    out, n = [], 0
    try:
        with progress_watchdog.watch("silence_map_decode",
                                     lambda: (progress_watchdog.process_progress(p.pid), n),
                                     p.kill, detail=f"file={path} stream={index}"):
            read_size = p.stdout.readinto(pcm_view[pcm_filled:])
            while read_size:
                pcm_filled += read_size
                m = (pcm_filled // bbytes) * bbytes
                if m:
                    x = np.frombuffer(pcm_buffer, dtype="<i2", count=m // 2).astype(np.float64) / 32768.0
                    out.append((x.reshape(-1, block) ** 2).mean(axis=1))
                    n += m // 2
                    pcm_view[:pcm_filled - m] = pcm_view[m:pcm_filled]
                    pcm_filled -= m
                read_size = p.stdout.readinto(pcm_view[pcm_filled:])
            if pcm_filled >= 2:
                x = np.frombuffer(pcm_buffer, dtype="<i2", count=pcm_filled // 2).astype(np.float64) / 32768.0
                out.append(np.array([(x ** 2).mean()]))
                n += x.size
    finally:
        p.stdout.close()
        p.wait()
        t_err.join(timeout=5)
    err = b"".join(c for c in err_chunks if c).decode("utf-8", "replace")
    ms = np.concatenate(out) if out else np.zeros(0)
    return ms, n, p.returncode, err.strip()[-400:]


def span_block_s(length_s):
    """Block length for a span of `length_s` seconds: about ten blocks, 10 ms to 400 ms."""
    return min(SILENCE_BLOCK_S, max(SPAN_BLOCK_MIN_S, float(length_s) / SPAN_BLOCKS))


def silence_map(video_obj, stream_id, *, kind="stream_order", timeline="timestamps",
                threads=THREADS, span_ms=None):
    """Return the silence map (>= 2 s at or under SILENCE_DB; digital under DIGITAL_DB) of one audio track, cached.

    `span_ms` maps only that span in scaled blocks; raises ValueError on a non-audio stream, step_stalled on a stall."""
    if timeline not in ("timestamps", "samples"):
        raise ValueError(f"timeline must be 'timestamps' or 'samples', not {timeline!r}")
    path, s = resolve_stream(video_obj, stream_id, kind)
    if s.get("codec_type") != "audio":
        raise ValueError(f"{path} stream {s['index']} is {s.get('codec_type')}, not audio")
    span_s = None
    block_s = SILENCE_BLOCK_S
    if span_ms is not None:
        span_s = (max(0.0, float(span_ms[0]) / 1000.0), float(span_ms[1]) / 1000.0)
        if span_s[1] <= 0:
            raise ValueError(f"span_ms must have a positive length, not {span_ms!r}")
        block_s = span_block_s(span_s[1])
    key = _file_key(path) + (s["index"], timeline, span_s)
    with _cache_lock:
        if key in _map_cache:
            _map_cache.move_to_end(key)
            return _map_cache[key]
    t0 = time.time()
    ms, n, rc, err = _decode_blocks(path, s["index"], timeline, threads, span_s, block_s)
    if span_s:
        start = span_s[0]
    else:
        # ffprobe gives start_time 0 to a stream whose first packet lies beyond its probe window (an audio track muxed 21.5 s late).
        stdout, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload([tools.software["ffprobe"], "-v", "error", "-select_streams", str(s["index"]),
                                                                              "-read_intervals", "%+#1", "-show_entries", "packet=pts_time",
                                                                              "-of", "csv=p=0", path], 2, 300)
        start = fc._num(stdout.decode("utf-8", errors="ignore").strip().split("\n")[0].strip(","))
        if start is None:
            start = fc._num(s.get("start_time")) or 0.0
    db = 10.0 * np.log10(np.maximum(ms, 1e-14)) if ms.size else np.zeros(0)
    m = dict(path=path, stream_index=s["index"], codec=s.get("codec_name"),
             lang=fc._tag(s, "LANGUAGE") or "und", timeline=timeline, start_s=start,
             length_s=n / ENV_SR, end_s=start + n / ENV_SR, block_s=block_s,
             rc=rc, stderr_tail=err if rc != 0 else "", db=db,
             max_db=round(float(np.max(db)), 1) if db.size else None)
    m.update(_spans(db, start, block_s,
                    min(SILENCE_MIN_S, span_s[1]) if span_s else SILENCE_MIN_S))
    m["cost_s"] = round(time.time() - t0, 1)
    span_text = (f" span_ms=[{span_ms[0]};+{span_ms[1]}] block_ms={round(block_s * 1000, 1)}"
                 if span_s else "")
    _log(f"silence_map path={path} stream={s['index']} timeline={timeline}{span_text} "
         f"length_s={m['length_s']:.3f} content=[{_fr(m['content_first_s'])};"
         f"{_fr(m['content_last_s'])}] silences={len(m['silences'])} max_db={m['max_db']} "
         f"silence_db={SILENCE_DB} rc={rc} cost_s={m['cost_s']}", always=span_s is not None)
    with _cache_lock:
        if len(_map_cache) >= _CACHE_MAX:
            _map_cache.popitem(last=False)
        _map_cache[key] = m
    return m


def _runs(mask):
    """Return the [i0, i1) runs of True in a boolean array."""
    if not mask.size:
        return []
    d = np.diff(np.pad(mask.astype(np.int8), 1))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _spans(db, start, b=SILENCE_BLOCK_S, silence_min_s=SILENCE_MIN_S):
    """Derive silences, clicks and content bounds from per-block dB levels (`b` s blocks)."""
    content = db > SILENCE_DB
    # Clicks: content runs < CLICK_S between silences >= CLICK_GAP_S are not content.
    runs = _runs(content)
    clicks = []
    for k, (i0, i1) in enumerate(runs):
        if (i1 - i0) * b >= CLICK_S:
            continue
        before = i0 - (runs[k - 1][1] if k else 0)
        after = (runs[k + 1][0] if k + 1 < len(runs) else len(db)) - i1
        if before * b >= CLICK_GAP_S and (after * b >= CLICK_GAP_S or k + 1 == len(runs)):
            clicks.append([round(float(start + i0 * b), 1), round(float(start + i1 * b), 1)])
            content[i0:i1] = False
    silences = []
    for i0, i1 in _runs(~content):
        if (i1 - i0) * b >= silence_min_s - 1e-9:
            silences.append([round(float(start + i0 * b), 2), round(float(start + i1 * b), 2),
                             bool(np.max(db[i0:i1]) <= DIGITAL_DB)])
    idx = np.flatnonzero(content)
    first = float(start + idx[0] * b) if idx.size else None
    last = float(start + (idx[-1] + 1) * b) if idx.size else None
    end = float(start + len(db) * b)
    trailing = [x for x in silences if x[1] >= end - 1e-6]
    return dict(silences=silences, clicks=clicks, content_first_s=first, content_last_s=last,
                last_silence_start_s=trailing[0][0] if trailing else end,
                content_mask=content)


def _levels_at(m, times):
    """dB of `m` at absolute instants of its own timeline; NaN where it has no data."""
    i = np.floor((times - m["start_s"]) / m["block_s"]).astype(np.int64)
    ok = (i >= 0) & (i < len(m["db"]))
    out = np.full(times.shape, np.nan)
    out[ok] = m["db"][i[ok]]
    return out


def _play_silences(m, play_end):
    """Return the map's silences plus the silence the file plays where the track has no packet.

    Adds [0, first sample) and [last sample, play_end); adjacent spans are merged.
    """
    spans = [[a, e, dg] for a, e, dg in m["silences"]]
    if m["start_s"] > 0:
        spans.insert(0, [0.0, m["start_s"], True])
    if play_end is not None and play_end > m["end_s"]:
        spans.append([m["end_s"], play_end, True])
    spans.sort()
    out = []
    for a, e, dg in spans:
        if out and a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], e)
            out[-1][2] = out[-1][2] and dg
        else:
            out.append([a, e, dg])
    return out


def _unmatched(x, y, shift_s, x_play_end, y_play_end):
    """Return the silences of `x` (>= BAD_SILENCE_S) that `y` does not have.

    A silence is unmatched when `y` (instant = x's + shift_s) plays sound above
    CONTENT_DB without a break for >= BAD_SILENCE_S. `y` is read over
    [0, max(its end, y_play_end)); beyond that it cannot contradict.
    """
    out = []
    b = x["block_s"]
    y_end = max(y["end_s"], y_play_end or 0.0)
    need = int(math.ceil(BAD_SILENCE_S / b - 1e-9))
    for a, e, digital in _play_silences(x, x_play_end):
        if e - a < BAD_SILENCE_S:
            continue
        t = np.arange(a + b / 2, e, b)
        ty = t + shift_s
        known = (ty >= 0) & (ty < y_end)
        ly = np.nan_to_num(_levels_at(y, ty), nan=-200.0)
        loud = known & (ly > CONTENT_DB)
        runs = [(i0, i1) for i0, i1 in _runs(loud) if i1 - i0 >= need]
        if not runs:
            continue
        longest = max(i1 - i0 for i0, i1 in runs)
        out.append(dict(start_s=round(float(a), 2), end_s=round(float(e), 2),
                        length_s=round(float(e - a), 2), digital=bool(digital),
                        other_sound_s=round(float(loud.sum() * b), 1),
                        other_longest_run_s=round(float(longest * b), 1),
                        other_sound_from_s=round(float(t[runs[0][0]]), 2)))
    return out


def silence_report(video_1, stream_a, video_2, stream_b, delay_ms=0, *, kind="stream_order",
                   threads=THREADS):
    """Compare two tracks' silence maps; the numbers behind `silences_agree`.

    Args:
        delay_ms: B's instant of a sound minus A's instant of the same sound.

    Returns:
        dict with agree, result ('A', 'B', 'AB' or None), a, b, a_only (A's
        silences B lacks), b_only, delay_ms and cost_s.
    """
    t0 = time.time()
    ma = silence_map(video_1, stream_a, kind=kind, threads=threads)
    mb = silence_map(video_2, stream_b, kind=kind, threads=threads)
    d = float(delay_ms) / 1000.0
    va, vb = video_end_s(video_1), video_end_s(video_2)
    a_only = _unmatched(ma, mb, d, va, vb)
    b_only = _unmatched(mb, ma, -d, vb, va)
    result = ("AB" if a_only and b_only else "A" if a_only else "B" if b_only else None)

    def summary(m, v):
        return dict(path=m["path"], stream_index=m["stream_index"], codec=m["codec"],
                    lang=m["lang"], start_s=m["start_s"], length_s=round(m["length_s"], 2),
                    end_s=m["end_s"], content_first_s=m["content_first_s"],
                    content_last_s=m["content_last_s"],
                    silences_ge_5s=[x for x in _play_silences(m, v)
                                    if x[1] - x[0] >= BAD_SILENCE_S][:30],
                    clicks=m["clicks"][:10], video_end_s=v, map_cost_s=m["cost_s"])
    rep = dict(agree=result is None, result=result, delay_ms=float(delay_ms),
               a=summary(ma, va), b=summary(mb, vb), a_only=a_only, b_only=b_only,
               cost_s=round(time.time() - t0, 1))
    first = lambda L: (f"{L[0]['start_s']}-{L[0]['end_s']}s(other sound {L[0]['other_sound_s']} s)"
                       if L else "-")
    _log(f"silences_agree A={ma['path']}#{ma['stream_index']} B={mb['path']}#{mb['stream_index']} "
         f"delay_ms={float(delay_ms):.1f} agree={rep['agree']} result={result} "
         f"a_only={len(a_only)}:{first(a_only)} b_only={len(b_only)}:{first(b_only)} "
         f"cost_s={rep['cost_s']}", always=result is not None)
    return rep


# ---------------------------------------------------------------- video

def _frame_probe(path, index, start, n=12):
    """Return (width, height, timestamp) of n frames from `start`."""
    cmd = [_bin("ffprobe"), "-v", "error", "-select_streams", str(index), "-read_intervals",
           f"{start:.3f}%+#{n}", "-show_entries", "frame=width,height,best_effort_timestamp_time",
           "-of", "csv=p=0", path]
    rc, out, err = fc._run(cmd)
    rows = []
    for line in out.decode("ascii", "replace").splitlines():
        p = [v for v in line.strip().split(",") if v != ""]
        if len(p) >= 3:
            try:
                rows.append((int(p[0]), int(p[1]), float(p[2])))
            except ValueError:
                pass
    return rows


def _rate(x):
    """Parse an 'a/b' frame rate; None when invalid."""
    try:
        a, b = str(x).split("/")
        return float(a) / float(b) if float(b) else None
    except (ValueError, TypeError):
        return None


def _video_consistency(video_obj, path, vs, starts):
    """Frame count vs duration x fps, and resolution / frame period at every window start."""
    info = probe(path)
    mi = getattr(video_obj, "video", None) if not isinstance(video_obj, (str, os.PathLike)) else None
    mi = mi if isinstance(mi, dict) else {}
    reasons, facts = [], {}
    r_rate, a_rate = _rate(vs.get("r_frame_rate")), _rate(vs.get("avg_frame_rate"))
    cfr = (mi.get("FrameRate_Mode") == "CFR") if mi.get("FrameRate_Mode") else (
        r_rate is not None and a_rate is not None and abs(r_rate - a_rate) <= 1e-3 * r_rate)
    fps = a_rate or r_rate or fc._num(mi.get("FrameRate"))
    frames = fc._num(fc._tag(vs, "NUMBER_OF_FRAMES")) or fc._num(vs.get("nb_frames")) \
        or fc._num(mi.get("FrameCount"))
    dur = _declared_end(info, vs) or fc._num(mi.get("Duration"))
    facts.update(cfr=cfr, fps=fps, frames=frames, duration_s=dur, width=vs.get("width"),
                 height=vs.get("height"))
    if cfr and fps and frames and dur:
        expected = dur * fps
        tol = max(FRAME_COUNT_TOL * expected, FRAME_COUNT_MIN_FRAMES)
        facts["frames_expected"] = round(expected, 1)
        if abs(frames - expected) > tol:
            reasons.append(f"frame_count_mismatch frames={int(frames)} expected={expected:.0f} "
                           f"(duration {dur:.3f} s x {fps:.4f} fps, tol {tol:.0f})")
    for k in ("Width", "Height"):
        if mi.get(k) and vs.get(k.lower()) and int(float(mi[k])) != int(vs[k.lower()]):
            reasons.append(f"header_{k.lower()}_mismatch mediainfo={mi[k]} ffprobe={vs[k.lower()]}")
    seen = []
    for st in (starts or [0.0]):
        rows = _frame_probe(path, vs["index"], st)
        if not rows:
            continue
        sizes = {(w, h) for w, h, _ in rows}
        ts = sorted(t for _, _, t in rows)
        per = float(np.median(np.diff(ts))) if len(ts) >= 3 else None
        seen.append(dict(start=st, sizes=sorted(sizes), period_s=per))
        if any((w, h) != (vs.get("width"), vs.get("height")) for w, h in sizes):
            reasons.append(f"resolution_changes at {st:.1f} s: {sorted(sizes)} vs header "
                           f"{vs.get('width')}x{vs.get('height')}")
        if cfr and fps and per and abs(per * fps - 1.0) > FPS_TOL:
            reasons.append(f"frame_rate_changes at {st:.1f} s: period {per * 1000:.2f} ms vs "
                           f"{1000 / fps:.2f} ms")
    facts["windows"] = seen
    return reasons, facts


def video_check(video_obj, full=False, *, windows=VIDEO_WINDOWS, seed=None, jobs=None,
                threads=THREADS):
    """Check the video stream: strict decode plus frame count, resolution and period.

    Returns:
        dict with sound, verdict, reasons, integrity (track_check dict),
        consistency and cost_s.

    Raises:
        ValueError: the file has no video stream.
    """
    t0 = time.time()
    path = _path_of(video_obj)
    info = probe(path)
    v = getattr(video_obj, "video", None) if not isinstance(video_obj, (str, os.PathLike)) else None
    if isinstance(v, dict) and "StreamOrder" in v:
        _, vs = resolve_stream(path, v["StreamOrder"])
    else:
        vs = _video_stream(info)
    if vs is None:
        raise ValueError(f"{path}: no video stream")
    integ = track_check(path, vs["index"], full=bool(full), windows=windows, seed=seed,
                        jobs=jobs, threads=threads)
    starts = [w["start"] for w in integ["windows"] if w["start"] is not None]
    reasons, facts = _video_consistency(video_obj, path, vs, starts)
    if integ["verdict"] == "corrupt":
        reasons.insert(0, f"strict_decode rc={integ['rc']} first=«"
                          f"{(integ['error_lines'] or ['?'])[0][:160]}»")
    verdict = "corrupt" if reasons else ("step_stalled" if integ["verdict"] == "step_stalled"
                                         else "sound")
    res = dict(path=path, stream_index=vs["index"], sound={"sound": True, "corrupt": False}.get(verdict),
               verdict=verdict, reasons=reasons, integrity=integ, consistency=facts,
               cost_s=round(time.time() - t0, 1))
    _log(f"video_check path={path} stream={vs['index']} verdict={verdict} reasons={reasons[:3]} "
         f"cost_s={res['cost_s']}", always=verdict != "sound")
    return res


def video_is_sound(video_obj, full=False, **kw):
    """Return True when the video decodes cleanly and its frame properties are consistent.

    Raises progress_watchdog.step_stalled when a window made no progress and nothing failed."""
    r = video_check(video_obj, full, **kw)
    if r["verdict"] == "step_stalled":
        raise progress_watchdog.step_stalled("strict_video_probe", "no_progress",
                                             f"file={r['path']}")
    return r["sound"]


# ---------------------------------------------------------------- conversion

# ---------------------------------------------------------------- conversion stderr

def conversion_stderr_is_clean(stderr_text, *, ignore_non_audio=True):
    """Return (True, []) when stderr has none of FATAL_CONVERSION_PATTERNS, else (False, lines).

    Carriage returns split lines too (ffmpeg progress). With ignore_non_audio,
    lines of subtitle / video output streams are ignored.
    """
    bad = []
    for line in re.split(r"[\r\n]+", stderr_text or ""):
        if not line.strip():
            continue
        if not any(p in line for p in FATAL_CONVERSION_PATTERNS):
            continue
        if ignore_non_audio and any(q in line for q in NON_AUDIO_STREAM_PREFIXES):
            continue
        bad.append(line.strip()[:300])
    if bad:
        _log(f"conversion_stderr fatal_lines={len(bad)} first=«{bad[0][:200]}»", always=True)
    return (not bad, bad[:50])


def check_delay_retention_sub(new_track, original_tracks, original_video_path, delay_to_put, sentence):
    if tools.dev:
        expected_delay = max(Decimal(0), Decimal(str(original_tracks.get('Delay', '0'))) + Decimal(str(delay_to_put))/Decimal(1000))
        tools.dev_log(f"[FFmpeg DEBUG] {sentence} {original_video_path} track {original_tracks['StreamOrder']}: delay {original_tracks.get('Delay', '0')} + {delay_to_put} ms -> {new_track.get('Delay', '0')}\n")
        if abs(expected_delay - Decimal(str(new_track.get('Delay', '0')))) > Decimal("0.001"):
            tools.dev_log(f"[FFmpeg WARN] {sentence} Delay retention for {original_video_path} track {original_tracks['StreamOrder']}: original delay {original_tracks.get('Delay', '0')}, delay put {delay_to_put} ms, expected delay {expected_delay}, new delay {new_track.get('Delay', '0')}\n")
