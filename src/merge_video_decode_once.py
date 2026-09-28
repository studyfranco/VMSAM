'''merge_video_decode_once.py -- ONE DECODE PER TRACK FOR THE WHOLE REPAIR.

Item (d) of the 2026-09-24 plan, made urgent by the delivery gate (coordinator 2026-09-28): on
Ragnarok S02E14 (id 23, 345a28b0) the 59 track builds took 4.6 min and the fabricated-track gate
compared 10 of 14 rebuilt audio tracks in 16 min before the budget stopped it -- every comparison
re-extracted the master's intact track in ten output-seek windows (each decoded from the file's
start), normalised them in two more ffmpeg passes each, and cross-correlated the fingerprints in a
pure-Python O(n^2) loop.

WHAT THIS MODULE OWNS. A per-repair cache of what is decoded from a track, keyed by
`(kind, realpath, size, mtime_ns, stream, params)`:
  * `fingerprint`  -- a whole-track chromaprint list (mono, at the pair's comparison rate): the
                      prime's own extraction registers its tracks (`put`), the delivery gate asks
                      for every track it compares (`fingerprints`, one ffmpeg pass per FILE for
                      all its requested streams, fpcalc two at a time);
  * `file_clock`   -- a track read on the file clock by `audio_walk.read_on_file_clock` (the
                      walk and the track-level remeasure read the same comparison tracks).
Every request logs one `decode_once: hit|miss` line (`tools.log_line`, unconditional in the job
log) so a pass can count the decodes saved; `end()` logs the scope's totals.

OUTSIDE A SCOPE (`begin` not called: a unit test, a caller outside `repair()`), nothing is cached
and every request is a `miss scope=none` -- the behaviour is the uncached one, never an error.
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
DECODE_THREADS = 2          # ffmpeg -threads, as every decode of this chain (ADDENDUM 27.8)
FPCALC_JOBS = 2             # fingerprints computed two at a time
DEFAULT_RATE = 44100        # the comparison grid's clamp (`repair_orchestrator.comparison_sample_rate`)

_LOCK = threading.RLock()
_SCOPE = None


def begin(work_dir, label=""):
    '''Opens the repair's scope (one per `repair()` call), or joins the one already open: a
    `repair()` that re-enters itself (the re-tagged comparison language) keeps its decodes. Every
    `begin` is paired with an `end` in a `finally`, so the depth always returns to zero.'''
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
    '''Leaves the scope; the outermost `end` closes it: its totals on one line, its memory
    released.'''
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
    '''A fact the scope carries for later requests (the pair's comparison rate).'''
    with _LOCK:
        if _SCOPE is not None:
            _SCOPE["notes"][name] = value


def noted(name, default=None):
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
    '''The cached value, or None (a miss is NOT logged here -- the producer's `put` logs it).
    `valid(value)`: a cached value it rejects is a miss (`reason=invalid`).'''
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
    '''Registers what a producer decoded (logged as the `miss` that paid for it).'''
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
    '''`lookup`, else `produce()` once -- two threads asking for the same key wait for the one
    decode instead of running two.'''
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
# WHOLE-TRACK FINGERPRINTS -- one ffmpeg pass per file for every stream requested
# ---------------------------------------------------------------------------------------------

def fingerprint_params(rate, pad_ms=0):
    """The key of a whole-track fingerprint: the rate, no filter, and the stream's own start on
    the file clock (`pad_ms`, its container start_time): a fingerprint read from the stream's
    first sample is on the FILE clock only when that start is 0 -- MEASURED Tougen S01E07, the
    master's ja stream starts at 1.125 s and a stream-relative read put every window 1125 ms
    off the product's."""
    return {"rate": int(rate), "filter": None, "pad_ms": round(float(pad_ms), 3)}


def _covers(need_s):
    return lambda value: value.get("duration_s", 0.0) + 1e-6 >= need_s


def fingerprints(file_path, streams, need_s, work_dir, rate=None, deadline=None, pads_ms=None):
    '''`{stream: {"points": [...], "duration_s": float}}` for every stream of ONE file, whole
    (at least `need_s` seconds, `{stream: seconds}`), mono at `rate` (the scope's noted
    comparison rate, else DEFAULT_RATE), ON THE FILE CLOCK: each stream's `pads_ms[stream]` (its
    container start_time) is prepended as silence, or trimmed when negative, so two files'
    fingerprints index the same instants. What the scope already holds is a hit; the rest is
    decoded in ONE ffmpeg pass (`-threads 2`, one WAV per stream) and fingerprinted two at a
    time. A stream that could not be read maps to None -- the caller reads that as "not
    measured", never as a verdict.'''
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
# THE FINGERPRINT CROSS-CORRELATION -- audioCorrelation.compare, vectorised, same numbers
# ---------------------------------------------------------------------------------------------

def compare(listx, listy, span, step=1):
    '''`audioCorrelation.compare` over numpy: for each offset in [-span, span], the overlapped
    lists truncated to the shorter one, `sum(32 - popcount(x ^ y)) / len / 32`. The same integer
    sum and the same two divisions as the Python loop, so the same floats and the same argmax.
    Offsets whose overlap is under `audioCorrelation.min_overlap` read None, as there.'''
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
    '''`audioCorrelation.correlate` on two fingerprint lists already in hand: the same span rule,
    the same `get_max_corr`, the same point size `int(length / len(source) * 1000)`.'''
    import audioCorrelation
    if len(source) != len(target):
        span = min(len(source), len(target)) - audioCorrelation.min_overlap
    else:
        span = len(target) - audioCorrelation.min_overlap
    corr = compare(source, target, span, audioCorrelation.step)
    return audioCorrelation.get_max_corr(corr, None, None, span,
                                         int(length_s / len(source) * 1000))


# ---------------------------------------------------------------------------------------------
# THE FILE-CLOCK READS OF THE WALK
# ---------------------------------------------------------------------------------------------

def file_clock(video_obj, audio, read, audio_filter=None, scale=None, envelope=False):
    '''`audio_walk.read_on_file_clock`'s result through the scope: `read()` runs once per
    (file, stream, filter, scale, envelope). The array is shared read-only -- a caller that
    needs to change it copies it.'''
    params = {"filter": audio_filter, "scale": str(scale), "envelope": bool(envelope)}
    value = get("file_clock", video_obj.filePath, int(audio["StreamOrder"]), params, read)
    try:
        value.flags.writeable = False
    except (AttributeError, ValueError):
        pass
    return value
