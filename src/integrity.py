"""Silence maps of audio tracks, and which of two aligned tracks holds silences the other lacks.

Public checks:
  silence_map             the silences of one audio track
  silence_report          the silences each of two aligned tracks has that the other lacks

stream_id is the absolute ffprobe stream index (mediainfo StreamOrder); pass
kind="audio_pos" (0-based, 0:a:N) or kind="typeorder" (1-based) otherwise.
delay_ms is B's instant of a sound minus A's.
"""

import math
import os
import shutil
import subprocess
import threading
import time

import numpy as np

import file_conformity as fc

try:
    import tools
except Exception:          # the module stays importable without the pipeline's config
    tools = None

# ---------------------------------------------------------------- constants

THREADS = 2
ENV_SR = 8000
SILENCE_BLOCK_S = 0.4
SILENCE_DB = fc.SILENCE_DB                    # dBFS RMS per 400 ms block
DIGITAL_DB = -90.0
SILENCE_MIN_S = 2.0
CLICK_S = fc.CLICK_S
CLICK_GAP_S = fc.ISLAND_GAP_S
CONTENT_DB = -50.0                            # "plays sound": 10 dB above the silence line
BAD_SILENCE_S = 5.0

_CACHE_MAX = 64
_probe_cache, _map_cache = {}, {}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------- helpers

def _log(message, always=False):
    line = f"integrity: {message}\n"
    if tools is None:
        return
    log = getattr(tools, "log_always" if always else "dev_log", None)
    if log is not None:
        log(line)
    elif always or getattr(tools, "dev", False):
        tools.logs.append(line)


def _bin(name):
    if tools is not None and getattr(tools, "software", None) and tools.software.get(name):
        return tools.software[name]
    return shutil.which(name) or name


def _bound(media_s):
    """Return the decoder timeout for a given media length in seconds."""
    if tools is not None and hasattr(tools, "decoder_timeout_for"):
        return tools.decoder_timeout_for(media_s or 0.0)
    return max(120.0, float(media_s or 0.0))


def _timeout_error(seconds, detail):
    if tools is not None and hasattr(tools, "decoder_timeout"):
        return tools.decoder_timeout("ffmpeg", seconds, detail)
    return TimeoutError(f"ffmpeg ran past its {round(seconds, 1)} s bound {detail}")


def _path_of(video_obj):
    if isinstance(video_obj, (str, os.PathLike)):
        return os.fspath(video_obj)
    return video_obj.filePath


def _file_key(path):
    """Return the cache key (path, size, mtime) of a file."""
    st = os.stat(path)
    return (os.path.abspath(path), st.st_size, st.st_mtime_ns)


def _probe(path):
    """Return the cached ffprobe result of a file."""
    key = _file_key(path)
    with _cache_lock:
        if key in _probe_cache:
            return _probe_cache[key]
    info = fc.probe(path)
    with _cache_lock:
        if len(_probe_cache) >= _CACHE_MAX:
            _probe_cache.clear()
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
    streams = _probe(path).get("streams", [])
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
    info = _probe(_path_of(video_obj))
    vs = _video_stream(info)
    return _declared_end(info, vs) if vs else None


def _fr(x, nd=1):
    return "?" if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------- silence maps

def _decode_blocks(path, index, timeline, threads, timeout):
    """Decode one stream to 8 kHz mono, reduced on the fly to 400 ms mean-square blocks.

    The 'timestamps' timeline fills gaps from packet timestamps (aresample
    async=1, without first_pts, which would pad from 0 on a filter rebuild);
    'samples' concatenates decoded samples, as mkvmerge re-times a track.

    Returns:
        (mean-square blocks, n_samples, rc, stderr_tail, timed_out).
    """
    af = f"aresample={ENV_SR}:async=1" if timeline == "timestamps" else f"aresample={ENV_SR}"
    cmd = [_bin("ffmpeg"), "-nostdin", "-hide_banner", "-v", "error", "-threads", str(threads),
           "-i", path, "-map", f"0:{index}", "-af", af, "-ac", "1",
           "-c:a", "pcm_s16le", "-f", "s16le", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         stdin=subprocess.DEVNULL)
    err_chunks = []
    t_err = threading.Thread(target=lambda: err_chunks.append(p.stderr.read()), daemon=True)
    t_err.start()
    killed = []

    def _kill():
        killed.append(True)
        p.kill()
    timer = threading.Timer(timeout, _kill)
    timer.start()
    block = int(ENV_SR * SILENCE_BLOCK_S)
    bbytes = block * 2
    rest = b""
    out, n = [], 0
    try:
        while True:
            b = p.stdout.read(1 << 18)
            if not b:
                break
            rest += b
            m = (len(rest) // bbytes) * bbytes
            if m:
                x = np.frombuffer(rest[:m], dtype="<i2").astype(np.float64) / 32768.0
                out.append((x.reshape(-1, block) ** 2).mean(axis=1))
                n += m // 2
                rest = rest[m:]
        if len(rest) >= 2:
            x = np.frombuffer(rest[: len(rest) // 2 * 2], dtype="<i2").astype(np.float64) / 32768.0
            out.append(np.array([(x ** 2).mean()]))
            n += x.size
    finally:
        p.stdout.close()
        p.wait()
        timer.cancel()
        t_err.join(timeout=5)
    err = b"".join(c for c in err_chunks if c).decode("utf-8", "replace")
    ms = np.concatenate(out) if out else np.zeros(0)
    return ms, n, p.returncode, err.strip()[-400:], bool(killed)


def silence_map(video_obj, stream_id, *, kind="stream_order", timeline="timestamps",
                threads=THREADS):
    """Return the silence map of one audio track (cached per file, stream and timeline).

    A silence is >= 2 s of blocks at or under SILENCE_DB, flagged digital when
    no block exceeds DIGITAL_DB.

    Returns:
        dict with path, stream_index, timeline, start_s, length_s, end_s,
        block_s, silences [[start, end, digital]], clicks, content_first_s,
        content_last_s, last_silence_start_s, rc, cost_s and db (per block).

    Raises:
        ValueError: the stream is not audio.
        tools.decoder_timeout: the decode ran past its bound.
    """
    if timeline not in ("timestamps", "samples"):
        raise ValueError(f"timeline must be 'timestamps' or 'samples', not {timeline!r}")
    path, s = resolve_stream(video_obj, stream_id, kind)
    if s.get("codec_type") != "audio":
        raise ValueError(f"{path} stream {s['index']} is {s.get('codec_type')}, not audio")
    key = _file_key(path) + (s["index"], timeline)
    with _cache_lock:
        if key in _map_cache:
            return _map_cache[key]
    t0 = time.time()
    info = _probe(path)
    media = _declared_end(info, s) or 0.0
    ms, n, rc, err, timed_out = _decode_blocks(path, s["index"], timeline, threads,
                                               _bound(max(media, 1.0) * 1.5))
    if timed_out:
        raise _timeout_error(_bound(max(media, 1.0) * 1.5), f"silence map of {path} stream {s['index']}")
    start = fc._num(s.get("start_time")) or 0.0
    db = 10.0 * np.log10(np.maximum(ms, 1e-14)) if ms.size else np.zeros(0)
    m = dict(path=path, stream_index=s["index"], codec=s.get("codec_name"),
             lang=fc._tag(s, "LANGUAGE") or "und", timeline=timeline, start_s=start,
             length_s=n / ENV_SR, end_s=start + n / ENV_SR, block_s=SILENCE_BLOCK_S,
             rc=rc, stderr_tail=err if rc != 0 else "", db=db)
    m.update(_spans(db, start))
    m["cost_s"] = round(time.time() - t0, 1)
    _log(f"silence_map path={path} stream={s['index']} timeline={timeline} "
         f"length_s={m['length_s']:.1f} content=[{_fr(m['content_first_s'])};"
         f"{_fr(m['content_last_s'])}] silences={len(m['silences'])} rc={rc} cost_s={m['cost_s']}")
    with _cache_lock:
        if len(_map_cache) >= _CACHE_MAX:
            _map_cache.clear()
        _map_cache[key] = m
    return m


def _runs(mask):
    """Return the [i0, i1) runs of True in a boolean array."""
    if not mask.size:
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _spans(db, start):
    """Derive silences, clicks and content bounds from per-block dB levels."""
    b = SILENCE_BLOCK_S
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
        if (i1 - i0) * b >= SILENCE_MIN_S:
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

