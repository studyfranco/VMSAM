"""Silence maps of audio tracks, and the silences one of two aligned tracks has that the other lacks."""

import json
import math
import os
import subprocess
import threading
import time

import numpy as np

import tools

THREADS = 2
ENV_SR = 8000
SILENCE_BLOCK_S = 0.4
SILENCE_DB = -60.0
DIGITAL_DB = -90.0
SILENCE_MIN_S = 2.0
CLICK_S = 3
CLICK_GAP_S = 60
CONTENT_DB = -50.0
BAD_SILENCE_S = 5.0

CACHE_MAX = 64
probe_cache, map_cache = {}, {}
cache_lock = threading.Lock()

def test_video_file_have_no_big_error(file_to_test):
    tools.launch_cmdExt_with_timeout_reload([tools.software["ffmpeg"], "-err_detect", "crccheck+bitstream+buffer",
                        "-analyzeduration", "1000M", "-probesize", "1000M", "-threads", "2",
                        "-i", file_to_test, "-map", "0", "-f", "null", "-c", "copy", "-"], 2, 1200)

def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def stream_tag(stream, key):
    for name, value in (stream.get("tags") or {}).items():
        if name.upper() == key or name.upper().startswith(key + "-"):
            return value
    return None


def probe(file_path):
    stat = os.stat(file_path)
    key = (os.path.abspath(file_path), stat.st_size, stat.st_mtime_ns)
    with cache_lock:
        if key in probe_cache:
            return probe_cache[key]
    try:
        result = subprocess.run([tools.software["ffprobe"], "-v", "error", "-show_streams", "-show_format",
                                 "-show_chapters", "-of", "json", file_path],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, timeout=300)
        code, out, err = result.returncode, result.stdout, result.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as e:
        code, out, err = -9, e.stdout or b"", "timeout after 300 s"
    if code != 0:
        raise RuntimeError(f"ffprobe rc={code}: {err.strip()[:300]}")
    info = json.loads(out.decode("utf-8", "replace"))
    with cache_lock:
        if len(probe_cache) >= CACHE_MAX:
            probe_cache.clear()
        probe_cache[key] = info
    return info


def declared_end(info, stream):
    try:
        hours, minutes, seconds = str(stream_tag(stream, "DURATION")).strip().split(":")
        end = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (ValueError, AttributeError):
        end = None
    if end is None:
        end = number(stream.get("duration"))
    if end is None:
        end = number(info.get("format", {}).get("duration"))
    return end


def video_end_s(file_path):
    info = probe(file_path)
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video" and not (stream.get("disposition") or {}).get("attached_pic"):
            return declared_end(info, stream)
    return None


def silence_map(file_path, stream_id):
    info = probe(file_path)
    try:
        index = int(str(stream_id).strip())
    except ValueError:
        raise ValueError(f"stream_id {stream_id!r} is not an integer stream index") from None
    hit = [stream for stream in info.get("streams", []) if stream.get("index") == index]
    if not hit:
        raise ValueError(f"{file_path}: no stream {stream_id!r} (kind=stream_order)")
    stream = hit[0]
    if stream.get("codec_type") != "audio":
        raise ValueError(f"{file_path} stream {stream['index']} is {stream.get('codec_type')}, not audio")
    stat = os.stat(file_path)
    key = (os.path.abspath(file_path), stat.st_size, stat.st_mtime_ns, stream["index"], "timestamps")
    with cache_lock:
        if key in map_cache:
            return map_cache[key]
    started = time.time()
    timeout = max(120.0, max(declared_end(info, stream) or 0.0, 1.0) * 1.5)
    process = subprocess.Popen([tools.software["ffmpeg"], "-nostdin", "-hide_banner", "-v", "error", "-threads", str(THREADS),
                                "-i", file_path, "-map", f"0:{stream['index']}", "-af", f"aresample={ENV_SR}:async=1",
                                "-ac", "1", "-c:a", "pcm_s16le", "-f", "s16le", "-"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    err_chunks = []
    err_reader = threading.Thread(target=lambda: err_chunks.append(process.stderr.read()), daemon=True)
    err_reader.start()
    killed = []

    def kill():
        killed.append(True)
        process.kill()
    timer = threading.Timer(timeout, kill)
    timer.start()
    block = int(ENV_SR * SILENCE_BLOCK_S)
    block_bytes = block * 2
    rest = b""
    mean_squares, samples = [], 0
    try:
        while True:
            chunk = process.stdout.read(1 << 18)
            if not chunk:
                break
            rest += chunk
            usable = (len(rest) // block_bytes) * block_bytes
            if usable:
                x = np.frombuffer(rest[:usable], dtype="<i2").astype(np.float64) / 32768.0
                mean_squares.append((x.reshape(-1, block) ** 2).mean(axis=1))
                samples += usable // 2
                rest = rest[usable:]
        if len(rest) >= 2:
            x = np.frombuffer(rest[: len(rest) // 2 * 2], dtype="<i2").astype(np.float64) / 32768.0
            mean_squares.append(np.array([(x ** 2).mean()]))
            samples += x.size
    finally:
        process.stdout.close()
        process.wait()
        timer.cancel()
        err_reader.join(timeout=5)
    err = b"".join(c for c in err_chunks if c).decode("utf-8", "replace").strip()[-400:]
    ms = np.concatenate(mean_squares) if mean_squares else np.zeros(0)
    if killed:
        raise TimeoutError(f"ffmpeg ran past its {round(timeout, 1)} s bound silence map of {file_path} stream {stream['index']}")
    start = number(stream.get("start_time")) or 0.0
    db = 10.0 * np.log10(np.maximum(ms, 1e-14)) if ms.size else np.zeros(0)
    m = dict(path=file_path, stream_index=stream["index"], codec=stream.get("codec_name"),
             lang=stream_tag(stream, "LANGUAGE") or "und", timeline="timestamps", start_s=start,
             length_s=samples / ENV_SR, end_s=start + samples / ENV_SR, block_s=SILENCE_BLOCK_S,
             rc=process.returncode, stderr_tail=err if process.returncode != 0 else "", db=db)

    content = db > SILENCE_DB
    content_runs = runs(content)
    clicks = []
    for k, (i0, i1) in enumerate(content_runs):
        if (i1 - i0) * SILENCE_BLOCK_S >= CLICK_S:
            continue
        before = i0 - (content_runs[k - 1][1] if k else 0)
        after = (content_runs[k + 1][0] if k + 1 < len(content_runs) else len(db)) - i1
        if before * SILENCE_BLOCK_S >= CLICK_GAP_S and (after * SILENCE_BLOCK_S >= CLICK_GAP_S or k + 1 == len(content_runs)):
            clicks.append([round(float(start + i0 * SILENCE_BLOCK_S), 1), round(float(start + i1 * SILENCE_BLOCK_S), 1)])
            content[i0:i1] = False
    silences = []
    for i0, i1 in runs(~content):
        if (i1 - i0) * SILENCE_BLOCK_S >= SILENCE_MIN_S:
            silences.append([round(float(start + i0 * SILENCE_BLOCK_S), 2), round(float(start + i1 * SILENCE_BLOCK_S), 2),
                             bool(np.max(db[i0:i1]) <= DIGITAL_DB)])
    idx = np.flatnonzero(content)
    end = float(start + len(db) * SILENCE_BLOCK_S)
    trailing = [x for x in silences if x[1] >= end - 1e-6]
    m.update(silences=silences, clicks=clicks,
             content_first_s=float(start + idx[0] * SILENCE_BLOCK_S) if idx.size else None,
             content_last_s=float(start + (idx[-1] + 1) * SILENCE_BLOCK_S) if idx.size else None,
             last_silence_start_s=trailing[0][0] if trailing else end, content_mask=content)
    m["cost_s"] = round(time.time() - started, 1)
    first, last = ("?" if value is None else f"{value:.1f}" for value in (m["content_first_s"], m["content_last_s"]))
    tools.dev_log(f"integrity: silence_map path={file_path} stream={stream['index']} timeline=timestamps "
                  f"length_s={m['length_s']:.1f} content=[{first};{last}] silences={len(silences)} "
                  f"rc={process.returncode} cost_s={m['cost_s']}\n")
    with cache_lock:
        if len(map_cache) >= CACHE_MAX:
            map_cache.clear()
        map_cache[key] = m
    return m


def runs(mask):
    if not mask.size:
        return []
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def play_silences(m, play_end):
    spans = [[a, e, digital] for a, e, digital in m["silences"]]
    if m["start_s"] > 0:
        spans.insert(0, [0.0, m["start_s"], True])
    if play_end is not None and play_end > m["end_s"]:
        spans.append([m["end_s"], play_end, True])
    spans.sort()
    merged = []
    for a, e, digital in spans:
        if merged and a <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2] = merged[-1][2] and digital
        else:
            merged.append([a, e, digital])
    return merged


def unmatched(x, y, shift_s, x_play_end, y_play_end):
    found = []
    b = x["block_s"]
    y_end = max(y["end_s"], y_play_end or 0.0)
    need = int(math.ceil(BAD_SILENCE_S / b - 1e-9))
    for a, e, digital in play_silences(x, x_play_end):
        if e - a < BAD_SILENCE_S:
            continue
        t = np.arange(a + b / 2, e, b)
        ty = t + shift_s
        known = (ty >= 0) & (ty < y_end)
        i = np.floor((ty - y["start_s"]) / y["block_s"]).astype(np.int64)
        has_data = (i >= 0) & (i < len(y["db"]))
        levels = np.full(ty.shape, np.nan)
        levels[has_data] = y["db"][i[has_data]]
        loud = known & (np.nan_to_num(levels, nan=-200.0) > CONTENT_DB)
        loud_runs = [(i0, i1) for i0, i1 in runs(loud) if i1 - i0 >= need]
        if not loud_runs:
            continue
        longest = max(i1 - i0 for i0, i1 in loud_runs)
        found.append(dict(start_s=round(float(a), 2), end_s=round(float(e), 2),
                          length_s=round(float(e - a), 2), digital=bool(digital),
                          other_sound_s=round(float(loud.sum() * b), 1),
                          other_longest_run_s=round(float(longest * b), 1),
                          other_sound_from_s=round(float(t[loud_runs[0][0]]), 2)))
    return found


def silence_report(path_a, stream_a, path_b, stream_b, delay_ms=0):
    """Compare two audio streams' silence maps, delay_ms being B's instant of a sound minus A's; b_only lists B's extra silences."""
    started = time.time()
    ma = silence_map(path_a, stream_a)
    mb = silence_map(path_b, stream_b)
    shift = float(delay_ms) / 1000.0
    va, vb = video_end_s(path_a), video_end_s(path_b)
    a_only = unmatched(ma, mb, shift, va, vb)
    b_only = unmatched(mb, ma, -shift, vb, va)
    result = ("AB" if a_only and b_only else "A" if a_only else "B" if b_only else None)
    sides = [dict(path=m["path"], stream_index=m["stream_index"], codec=m["codec"],
                  lang=m["lang"], start_s=m["start_s"], length_s=round(m["length_s"], 2),
                  end_s=m["end_s"], content_first_s=m["content_first_s"],
                  content_last_s=m["content_last_s"],
                  silences_ge_5s=[x for x in play_silences(m, v) if x[1] - x[0] >= BAD_SILENCE_S][:30],
                  clicks=m["clicks"][:10], video_end_s=v, map_cost_s=m["cost_s"])
             for m, v in ((ma, va), (mb, vb))]
    report = dict(agree=result is None, result=result, delay_ms=float(delay_ms), a=sides[0], b=sides[1],
                  a_only=a_only, b_only=b_only, cost_s=round(time.time() - started, 1))
    first = lambda spans: (f"{spans[0]['start_s']}-{spans[0]['end_s']}s(other sound {spans[0]['other_sound_s']} s)"
                           if spans else "-")
    line = (f"integrity: silences_agree A={ma['path']}#{ma['stream_index']} B={mb['path']}#{mb['stream_index']} "
            f"delay_ms={float(delay_ms):.1f} agree={report['agree']} result={result} "
            f"a_only={len(a_only)}:{first(a_only)} b_only={len(b_only)}:{first(b_only)} "
            f"cost_s={report['cost_s']}\n")
    if result is None:
        tools.dev_log(line)
    else:
        tools.logs.append(line)
    return report
