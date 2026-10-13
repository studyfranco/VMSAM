'''Per-repair cache so each track is decoded once for the whole repair.

Entries are keyed by `(kind, realpath, size, mtime_ns, stream, params)`:
  * `fingerprint` -- a whole-track chromaprint list (mono, at the comparison rate); one ffmpeg
                     pass per file for all requested streams, fpcalc two at a time;
  * `file_clock`  -- a track read on the file clock by `audio_walk.read_on_file_clock`.
Every request logs a `decode_once: hit|miss` line; `end()` logs the totals.

Outside a scope (`begin` not called) nothing is cached and every request is a
`miss scope=none`.
'''
import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import tools

LOG_PREFIX = "decode_once: "
DECODE_THREADS = 2          # ffmpeg -threads
FPCALC_JOBS = 2             # fingerprints computed two at a time
DEFAULT_RATE = 44100        # the comparison grid's clamp (`repair_orchestrator.comparison_sample_rate`)

_LOCK = threading.RLock()
_SCOPE = None


def begin(work_dir, label=""):
    '''Open the cache scope, or join the one already open (nested calls keep its decodes).

    Every `begin` must be paired with an `end`.'''
    global _SCOPE
    with _LOCK:
        if _SCOPE is not None:
            _SCOPE["depth"] += 1
            return
        _SCOPE = {"dir": os.path.join(work_dir, "decode_once"), "mem": {}, "busy": {},
                  "hits": 0, "misses": 0, "saved_s": 0.0, "cost_s": {}, "label": label,
                  "notes": {}, "depth": 1}
        os.makedirs(_SCOPE["dir"], exist_ok=True)


def end():
    '''Leave the scope; the outermost `end` logs the totals and releases the cache.'''
    global _SCOPE
    with _LOCK:
        if _SCOPE is not None and _SCOPE["depth"] > 1:
            _SCOPE["depth"] -= 1
            return
        scope, _SCOPE = _SCOPE, None
    if scope is None:
        return
    tools.log_line(f"{LOG_PREFIX}summary hits={scope['hits']} misses={scope['misses']} "
                   f"saved_s={round(scope['saved_s'], 1)} label={scope['label']}\n")
    try:
        for name in os.listdir(scope["dir"]):
            os.remove(os.path.join(scope["dir"], name))
        os.rmdir(scope["dir"])
    except OSError:
        pass


def note(name, value):
    '''Store a value in the scope for later requests (e.g. the comparison rate).'''
    with _LOCK:
        if _SCOPE is not None:
            _SCOPE["notes"][name] = value


def noted(name, default=None):
    '''Return a value stored by `note`, or `default`.'''
    with _LOCK:
        return default if _SCOPE is None else _SCOPE["notes"].get(name, default)


def _key(kind, file_path, stream, params):
    st = os.stat(file_path)
    raw = json.dumps([kind, os.path.realpath(file_path), st.st_size, st.st_mtime_ns, str(stream),
                      sorted((str(k), str(v)) for k, v in (params or {}).items())])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def _say(outcome, kind, file_path, stream, params, cost_s=None, extra=""):
    tools.log_line(f"{LOG_PREFIX}{outcome} kind={kind} file={os.path.basename(file_path)} "
                   f"stream={stream} params={json.dumps(params, sort_keys=True, default=str)}"
                   + (f" cost_s={round(cost_s, 2)}" if cost_s is not None else "")
                   + (f" {extra}" if extra else "") + "\n")


def lookup(kind, file_path, stream, params, valid=None):
    '''Return the cached value, or None; a value rejected by `valid(value)` is a miss.

    Misses are logged by the producer's `put`, not here.'''
    with _LOCK:
        if _SCOPE is None:
            return None
        key = _key(kind, file_path, stream, params)
        value = _SCOPE["mem"].get(key)
        if value is None:
            return None
        if valid is not None and not valid(value):
            return None
        _SCOPE["hits"] += 1
        _SCOPE["saved_s"] += _SCOPE["cost_s"].get(key, 0.0)
    _say("hit", kind, file_path, stream, params)
    return value


def put(kind, file_path, stream, params, value, cost_s=None):
    '''Store a decoded value (logged as a `miss`).'''
    with _LOCK:
        if _SCOPE is None:
            _say("miss", kind, file_path, stream, params, cost_s, "scope=none")
            return
        key = _key(kind, file_path, stream, params)
        _SCOPE["mem"][key] = value
        _SCOPE["cost_s"][key] = float(cost_s or 0.0)
        _SCOPE["misses"] += 1
    _say("miss", kind, file_path, stream, params, cost_s)


def get(kind, file_path, stream, params, produce, valid=None):
    '''Return the cached value, else run `produce()` once; concurrent callers wait for it.'''
    value = lookup(kind, file_path, stream, params, valid)
    if value is not None:
        return value
    with _LOCK:
        in_scope = _SCOPE is not None
        key = _key(kind, file_path, stream, params) if in_scope else None
        event = _SCOPE["busy"].get(key) if in_scope else None
        mine = in_scope and event is None
        if mine:
            event = _SCOPE["busy"][key] = threading.Event()
    if in_scope and not mine:
        event.wait()
        value = lookup(kind, file_path, stream, params, valid)
        if value is not None:
            return value
    try:
        started = time.monotonic()
        value = produce()
        put(kind, file_path, stream, params, value, time.monotonic() - started)
        return value
    finally:
        if mine:
            with _LOCK:
                if _SCOPE is not None:
                    _SCOPE["busy"].pop(key, None)
            event.set()


# ---------------------------------------------------------------------------------------------
# Whole-track fingerprints: one ffmpeg pass per file
# ---------------------------------------------------------------------------------------------

def fingerprint_params(rate, pad_ms=0):
    """Cache params of a whole-track fingerprint: rate, no filter, and the stream start_time.

    `pad_ms` is needed because a fingerprint is on the file clock only when the stream starts at 0.
    """
    return {"rate": int(rate), "filter": None, "pad_ms": round(float(pad_ms), 3)}


def _covers(need_s):
    return lambda value: value.get("duration_s", 0.0) + 1e-6 >= need_s


def fingerprints(file_path, streams, need_s, work_dir, rate=None, deadline=None, pads_ms=None):
    '''Whole-track fingerprints of several streams of one file, on the file clock.

    Uncached streams are decoded in one ffmpeg pass and fingerprinted two at a time.

    Args:
        file_path: the media file.
        streams: stream indices to fingerprint.
        need_s: {stream: minimum seconds covered}.
        work_dir: directory for temporary WAVs.
        rate: sample rate (default: the scope's comparison rate, else DEFAULT_RATE).
        deadline: monotonic time after which nothing new is decoded.
        pads_ms: {stream: container start_time}, padded with silence or trimmed.

    Returns:
        {stream: {"points": [...], "duration_s": float} or None when unreadable}.
    '''
    import audioCorrelation
    rate = int(rate or noted("comparison_rate", DEFAULT_RATE))
    pads_ms = {stream: float((pads_ms or {}).get(stream) or 0.0) for stream in streams}
    params_of = {stream: fingerprint_params(rate, pads_ms[stream]) for stream in streams}
    out, todo = {}, []
    for stream in streams:
        value = lookup("fingerprint", file_path, stream, params_of[stream],
                       _covers(need_s[stream]))
        if value is not None:
            out[stream] = value
        else:
            todo.append(stream)
    if not todo:
        return out
    if deadline is not None and time.monotonic() > deadline:
        for stream in todo:
            out[stream] = None
        return out
    private = os.path.join(work_dir or tools.tmpFolder, f"decode_once_{os.getpid()}_"
                           f"{threading.get_ident()}_{int(time.monotonic() * 1000)}")
    os.makedirs(private, exist_ok=True)
    wavs = {stream: os.path.join(private, f"s{stream}.wav") for stream in todo}
    length = max(need_s[s] for s in todo)
    cmd = [tools.software["ffmpeg"], "-v", "error", "-y", "-nostdin", "-threads",
           str(DECODE_THREADS), "-i", file_path]
    for stream in todo:
        cmd += ["-map", f"0:{stream}", "-vn", "-ac", "1", "-ar", str(rate)]
        pad = pads_ms[stream]
        if pad > 0.5:
            cmd += ["-af", f"adelay=delays={pad:.3f}:all=1"]
        elif pad < -0.5:
            cmd += ["-af", f"atrim=start={-pad / 1000.0:.6f},asetpts=PTS-STARTPTS"]
        cmd += ["-acodec", "pcm_s16le", wavs[stream]]
    timeout = tools.decoder_timeout_for(length * max(1, len(todo)))
    if deadline is not None:
        timeout = max(1.0, min(timeout, deadline - time.monotonic()))
    started = time.monotonic()
    try:
        tools.dev_log(f"{LOG_PREFIX}ffmpeg pass file={file_path} streams={todo} rate={rate}\n")
        done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout)
        decoded = done.returncode == 0
        decode_error = (done.stderr or b"").decode("utf-8", "replace").strip()[-200:]
    except subprocess.TimeoutExpired:
        decoded, decode_error = False, f"decoder_timeout after {round(timeout, 1)} s"
    decode_s = time.monotonic() - started

    def one(stream):
        t0 = time.monotonic()
        wav = wavs[stream]
        try:
            if not decoded or not os.path.exists(wav) or os.path.getsize(wav) < rate * 2:
                return stream, None, 0.0
            import wave
            with wave.open(wav, "rb") as handle:
                seconds = handle.getnframes() / float(handle.getframerate())
            points = audioCorrelation.calculate_fingerprints(wav, length=int(seconds) + 1)
            return stream, {"points": points, "duration_s": seconds}, time.monotonic() - t0
        except Exception:                                                # noqa: BLE001
            return stream, None, time.monotonic() - t0
        finally:
            try:
                os.remove(wav)
            except OSError:
                pass
    try:
        with ThreadPoolExecutor(max_workers=FPCALC_JOBS) as pool:
            for stream, value, fp_s in pool.map(one, todo):
                if value is None:
                    _say("miss", "fingerprint", file_path, stream, params_of[stream], None,
                         f"unreadable={decode_error or 'no_wav'}")
                    out[stream] = None
                    continue
                put("fingerprint", file_path, stream, params_of[stream], value,
                    decode_s / len(todo) + fp_s)
                out[stream] = value
    finally:
        try:
            os.rmdir(private)
        except OSError:
            pass
    return out


# ---------------------------------------------------------------------------------------------
# Fingerprint cross-correlation: audioCorrelation.compare, vectorised
# ---------------------------------------------------------------------------------------------

def compare(listx, listy, span, step=1):
    '''Vectorised `audioCorrelation.compare` producing identical floats.

    For each offset in [-span, span]: `sum(32 - popcount(x ^ y)) / n / 32` over the overlap;
    None when the overlap is under `audioCorrelation.min_overlap`.'''
    import audioCorrelation
    x = np.asarray(listx, dtype=np.int64)
    y = np.asarray(listy, dtype=np.int64)
    if (x < 0).any() or (y < 0).any() or (x >= 2 ** 32).any() or (y >= 2 ** 32).any():
        return audioCorrelation.compare(list(listx), list(listy), span, step)
    x, y = x.astype(np.uint32), y.astype(np.uint32)
    if span > min(len(x), len(y)):
        raise Exception('span >= sample size: %i >= %i\n' % (span, min(len(x), len(y))))
    out = []
    for offset in range(-span, span + 1, step):
        if offset > 0:
            a = x[offset:]
            b = y[:len(a)]
        elif offset < 0:
            b = y[-offset:]
            a = x[:len(b)]
        else:
            a, b = x, y
        n = min(len(a), len(b))
        if n < audioCorrelation.min_overlap:
            out.append(None)
            continue
        ones = int(np.bitwise_count(np.bitwise_xor(a[:n], b[:n])).sum())
        covariance = (32 * n - ones) / float(n)
        out.append(covariance / 32)
    return out


def correlate_points(source, target, length_s):
    '''`audioCorrelation.correlate` on two fingerprint lists already computed.'''
    import audioCorrelation
    if len(source) != len(target):
        span = min(len(source), len(target)) - audioCorrelation.min_overlap
    else:
        span = len(target) - audioCorrelation.min_overlap
    corr = compare(source, target, span, audioCorrelation.step)
    return audioCorrelation.get_max_corr(corr, None, None, span,
                                         int(length_s / len(source) * 1000))


# ---------------------------------------------------------------------------------------------
# File-clock reads
# ---------------------------------------------------------------------------------------------

def file_clock(video_obj, audio, read, audio_filter=None, scale=None, envelope=False):
    '''Cached `audio_walk.read_on_file_clock` result, one `read()` per parameter set.

    The returned array is read-only; copy it before modifying.'''
    params = {"filter": audio_filter, "scale": str(scale), "envelope": bool(envelope)}
    value = get("file_clock", video_obj.filePath, int(audio["StreamOrder"]), params, read)
    try:
        value.flags.writeable = False
    except (AttributeError, ValueError):
        pass
    return value
