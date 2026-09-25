"""
integrity.py -- does a track still hold what it claims, and does a conversion
keep it?

Owner, 2026-09-25 23:1x (ADDENDUM 32.8, after the Chainsaw Man lab): integrity
checks spread through the pipeline at five hook points in the owner's own
files, which the owner wires himself. This module is what they call. It
MEASURES and ANSWERS; it never writes a track, never modifies a file, never
decides what the caller does with the answer.

THE FUNCTIONS THE OWNER CALLS
-----------------------------
  track_is_sound(video_obj, stream_id, full=None) -> bool
      « je donne un objet video et le streamID à tester, elle renvoie true si la
      piste est saine, false sinon ; juste un test ffmpeg qui valide
      l'intégrité d'une piste ». Its numbers: `track_check(...) -> dict`.
  silences_agree(video_1, stream_a, video_2, stream_b, delay_ms=0)
      -> (True, None) | (False, 'A' | 'B' | 'AB')
      « elle renvoie (true, None) ou (false, 'A' ou 'B') sur la piste qui a des
      silences pas bons » -- 'AB' when each track has silences the other does
      not have (owner 23:5x). Its numbers: `silence_report(...) -> dict`.
  video_is_sound(video_obj) -> bool          (numbers: `video_check`)
      For the hook at video-object creation -> the owner's tag
      `video_unreliable`.
  conversion_preserved(original_video_obj, original_stream, produced_path,
                       produced_stream, delay_ms, expected_duration_s=None)
      -> (bool, str)
      After a conversion: same end, no NEW silence, same sound.
  conversion_stderr_is_clean(stderr_text) -> (bool, list[str])
      and `CONVERSION_SAFE_OPTIONS` / `FATAL_CONVERSION_PATTERNS`.

WHAT `stream_id` IS
-------------------
The index the owner's video object uses for its streams: `StreamOrder`, the
mediainfo key every track dict of `video.audios[lang]` / `.video` carries.
video.py proves it is the ffprobe/ffmpeg stream index -- `get_mediadata` looks
up `ffprobe_data[int(data['StreamOrder'])]` (video.py:85) and every extraction
maps `"-map", "0:"+str(audio['StreamOrder'])` (video.py:285, :300). So by
default `stream_id` is that ABSOLUTE index (int or its string), and a track
dict itself is accepted (its 'StreamOrder' is read). It is NOT the position in
`video.audios[lang]` (a list per language) nor `@typeorder` (1-based among one
type, and absent when the file has a single track of that type). When a caller
holds one of those, it says so: `kind="audio_pos"` (0-based among the file's
audio streams, ffmpeg's `0:a:N`) or `kind="typeorder"` (mediainfo's 1-based
`@typeorder` of an audio track). `video_obj` may also be a plain path.

THE MEASUREMENTS AND THEIR BASIS
--------------------------------
STRICT DECODE (MASTER_INTEGRITY_FFMPEG_20260925.md, ADDENDUM 26.9.10/.12):
  `ffmpeg -xerror -err_detect crccheck+bitstream+buffer+explode -threads 2
  -i F -map 0:N -f null -`. A track FAILS on rc != 0 OR on any error-level
  line that is not a muxer line (the muxer's « non monotonically increasing
  dts » and « Non-monotonic DTS » are ignored: measured false positives on the
  healthy 224c / 216c). Audio: the WHOLE track, always (26.9.10 accepted for
  these tracks by the owner, 32.8(b)) -- the Chainsaw corruption is ONE E-AC-3
  frame at 4 007.968 s that a sample would miss. Video: 10 stratified 2-minute
  windows, seed = crc32 of the path (logged), all at once (26.9.12; measured 0
  false positive on 58 probes / 20 files, 5/5 on the 349 candidate); `full=True`
  decodes all of it. Every process is bounded by max(120 s, 1 s per second of
  media) (ADDENDUM 26.3, `tools.decoder_timeout_for`); a process that runs
  past it is a statement about the host, not the media: `track_check` says
  `verdict='decoder_timeout'`, `track_is_sound` raises `tools.decoder_timeout`
  (never a False the caller would read as « corrupt »).
SILENCE MAP: the track decoded once to 8 kHz mono, 400 ms blocks; a block is
  silent at <= -60 dBFS RMS (the owner's silencedetect level, file_conformity
  SILENCE_DB; digital silence measured at -121 dB in 691), a SILENCE is a run
  of silent blocks lasting >= 2 s, flagged `digital` when no block rises above
  -90 dB. Clicks are file_conformity's: a content run shorter than 3 s between
  two silences of >= 60 s is not content (691 jpn: 1 s blips every ~931 s
  inside its 6 802 s of digital silence).
  Two timelines. `timestamps` (default) places the samples by their packet
  timestamps (`aresample=async=1`, gaps filled) -- the real timeline of a
  source, the one the pipeline's delays live on. `samples` concatenates the
  decoded samples, as mkvmerge re-times a track from its frames -- the timeline
  a CONVERSION PRODUCT really has. MEASURED on the Chainsaw window: the legacy
  re-encode's output decodes to 198.9 s by timestamps and 308.0 s by samples,
  and mkvmerge makes it 308.0 s.
  `first_pts=0` is NOT used by the measuring decode: MEASURED, the corrupt
  E-AC-3 frame rebuilds the filter graph and a fresh `first_pts=0` resampler
  pads from 0 to the frame -- the very defect under test, 17.97 s inserted into
  a 40 s window -- whereas `async=1` alone rebuilds without padding (39.996 s
  out for 40 s in). The stream's own start_time places block 0.
SILENCE COMPARISON (`silence_report`; owner 2026-09-25 23:5x: « c'est juste une
  comparaison des silences que je veux, pas plus, pas d'autre intelligence »):
  the two silence maps, B shifted by the delay onto A's timeline, compared. A
  silence of >= 5 s in one track is a SILENCE THE OTHER DOES NOT HAVE when, at
  the corresponding instants, the other plays sound WITHOUT A BREAK for >= 5 s
  (the same threshold both ways). "Plays sound" = louder than -50 dBFS: the
  maps are compared with 10 dB of tolerance around the -60 line, so a quiet
  passage at -62 in one track and -58 in the other is the same silence, not a
  disagreement. What a file plays is read up to the end of its own video (or of
  the track, whichever is later): after a track's last packet the file plays
  silence (349's master: four tracks stop at 3 404.6 s, the video runs to
  4 494.5 s), beyond that the file says nothing and cannot contradict.
  -> (True, None) the maps agree; (False, 'A') only A has silences B does not
  have; (False, 'B') only B; (False, 'AB') each has silences the other does
  not have -- the ambiguous case, the caller decides. No tail rule, no credits
  exception, no ranking here: those stay in file_conformity / the repair path.
`delay_ms` IN `silences_agree`: B's instant of a sound minus A's instant of the
  same sound -- the value `mergeVideo.compare_video.first_delay_test` computes
  between video_obj_1 (A) and video_obj_2 (B) (`correlate` returns
  -offset*size, i.e. t2 - t1; `recreate_files_for_delay_adjuster` cuts video 2
  at begin + delay). The mkvmerge-style delay the pipeline stores in
  `video_obj_2.delays[lang]` is its NEGATION (mergeVideo.py:669).
`delay_ms` IN `conversion_preserved`: the mkvmerge-style `delay_to_put` of
  `generate_new_file`: the produced track's instant = the original's + delay.
CONVERSION (owner 2026-09-25 23:4x: the content comparison is the PRIMARY
  post-conversion check): (1) the produced track's end -- the sum of its frame
  durations (how mkvmerge re-times it) and its last packet, one ffprobe pass,
  no decode -- within 500 ms of the original's end + delay, bounded by
  `expected_duration_s` -> `conversion_duration` otherwise; (2) chromaprint
  similarity of three 60 s windows (produced vs original shifted by the delay,
  windows where the original is silence skipped), median >= 0.98 ->
  `conversion_fidelity <x>` otherwise. MEASURED: the legacy eac3 re-encode of a
  healthy window at -1001 ms scores 0.9945-0.9969. (3) The silence maps are the
  EXPLANATION when (1) or (2) fails, and the check itself when fpcalc is
  absent: a silence >= 5 s in the product that the shifted original does not
  have (the comparison above) -> `conversion_created_silence at <s>`.
CONVERSION OPTIONS (MEASURED on the 200 s Chainsaw window around 4 007.968 s,
  with the legacy second-pass command of mergeVideo.py:1709-1735 at delay
  -1001 ms): the legacy command exits 0, logs « Reconfiguring filter graph
  because audio parameters changed » twice and 3 409 « Non-monotonic DTS »,
  and writes a track of 308.0 s for 198.9 s; with `-xerror` (before `-i`: the
  legacy `-err_detect crccheck` already flags « frame CRC mismatch ») it exits
  183, with `-reinit_filter 0` alone it exits 234 (« Changing audio frame
  properties on the fly is not supported »). On the healthy ToonsHub window at
  the same place, both options leave rc 0 and a clean stderr.

Nothing here is wired: the owner calls it (INTEGRITY_HOOKS_PROPOSAL_20260925).
"""

import concurrent.futures
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import zlib

import numpy as np

import file_conformity as fc

try:
    import tools
except Exception:          # the module stays importable without the pipeline's config
    tools = None

# ---------------------------------------------------------------- constants

STRICT_FLAGS = list(fc.STRICT_FLAGS)          # -xerror -err_detect crccheck+bitstream+buffer+explode
THREADS = 2                                   # 26.9.12: 2 threads per process, measured
VIDEO_WINDOWS = fc.INTEGRITY_WINDOWS          # 10
WINDOW_S = fc.INTEGRITY_WINDOW_S              # 120 s
# error-level lines that are not the decoder's: the muxer's (measured false positives on the
# healthy 224c / 216c) and the demuxer's complaint on a SEEK into a file whose index marks no
# keyframe (MEASURED 2026-09-25 on Fate UBW 08 TrueHD: « File is broken, keyframes not
# correctly marked! » on 4/4 windows of a stream whose full strict decode is clean, rc 0)
IGNORED_LINE_SUBSTRINGS = tuple(fc.IGNORED_ERROR_SUBSTRINGS) + (
    "Non-monotonic DTS", "keyframes not correctly marked")

ENV_SR = 8000
SILENCE_BLOCK_S = 0.4
SILENCE_DB = fc.SILENCE_DB                    # -60 dBFS RMS per 400 ms block
DIGITAL_DB = -90.0
SILENCE_MIN_S = 2.0
CLICK_S = fc.CLICK_S                          # 3 s
CLICK_GAP_S = fc.ISLAND_GAP_S                 # 60 s
CONTENT_DB = -50.0                            # "plays sound": 10 dB above the silence line
BAD_SILENCE_S = 5.0

CONVERSION_DURATION_TOL_S = 0.5
FIDELITY_MIN = 0.98
FIDELITY_WINDOW_S = 60.0
FIDELITY_FRACTIONS = (0.25, 0.5, 0.75)
FRAME_COUNT_TOL = 0.001                       # nb_frames vs duration x fps
FRAME_COUNT_MIN_FRAMES = 2
FPS_TOL = 0.02

# The options every conversion should carry (see the module docstring for the measure).
# `-xerror` is global; `-reinit_filter 0` is an INPUT option: it goes before the `-i` of the
# input whose audio is filtered.
CONVERSION_SAFE_OPTIONS = ["-xerror", "-reinit_filter", "0"]
FATAL_CONVERSION_PATTERNS = (
    "Non-monotonic DTS",               # muxer rewrote timestamps (Chainsaw H1: 125 219 per track)
    "Failed to compensate",            # swresample could not absorb a timestamp jump
    "Reconfiguring filter graph",      # audio parameters changed mid-stream: the 4 007 s cause
    "new coupling strategy",           # the corrupt E-AC-3 frame itself
    "Error while decoding",
    "frame CRC mismatch",              # measured: the same frame under -err_detect crccheck
    "Error submitting packet to decoder",   # measured: what -xerror stops on
    "Changing audio frame properties on the fly",   # measured: -reinit_filter 0 refusing
)
# a Non-monotonic DTS on a SUBTITLE output stream (`[sost#...]`) is a cue order matter, not audio
NON_AUDIO_STREAM_PREFIXES = ("[sost#", "[vost#")

_CACHE_MAX = 64
_probe_cache, _map_cache, _check_cache = {}, {}, {}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------- helpers

def _log(message, always=False):
    line = f"integrity: {message}\n"
    if tools is None:
        return
    (tools.log_always if always else tools.dev_log)(line)


def _bin(name):
    if tools is not None and getattr(tools, "software", None) and tools.software.get(name):
        return tools.software[name]
    return shutil.which(name) or name


def _bound(media_s):
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
    st = os.stat(path)
    return (os.path.abspath(path), st.st_size, st.st_mtime_ns)


def _probe(path):
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
    """-> (path, ffprobe stream dict). See the module docstring for `kind`."""
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
    """The video's end: the object's mediainfo video Duration when it has one, else the
    declared end (tag DURATION first -- 691's format.duration is 12 799 s for a 5 997.5 s
    video) of the first video stream. None without video."""
    v = getattr(video_obj, "video", None)
    if isinstance(v, dict) and fc._num(v.get("Duration")):
        return float(v["Duration"])
    info = _probe(_path_of(video_obj))
    vs = _video_stream(info)
    return _declared_end(info, vs) if vs else None


def _fr(x, nd=1):
    return "?" if x is None else f"{x:.{nd}f}"


def _run_capture(cmd, timeout):
    """(rc, stderr_text, timed_out, seconds)."""
    t0 = time.time()
    try:
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           stdin=subprocess.DEVNULL, timeout=timeout)
        return p.returncode, p.stderr.decode("utf-8", "replace"), False, time.time() - t0
    except subprocess.TimeoutExpired as e:
        err = (e.stderr or b"").decode("utf-8", "replace")
        return -9, err, True, time.time() - t0


def _error_lines(stderr_text):
    return [l.strip() for l in stderr_text.splitlines()
            if l.strip() and not any(s in l for s in IGNORED_LINE_SUBSTRINGS)]


def decode_error_lines(stderr_text):
    """The lines of an `-v error` ffmpeg stderr that count against a strict decode (every
    error-level line but the muxer's and the seek's, see IGNORED_LINE_SUBSTRINGS)."""
    return _error_lines(stderr_text or "")


# ---------------------------------------------------------------- (A) strict decode

def _strict(path, index, start, dur, threads, timeout):
    cmd = [_bin("ffmpeg"), "-nostdin", "-hide_banner", "-v", "error"] + STRICT_FLAGS + [
        "-threads", str(threads)]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}"]
    cmd += ["-i", path, "-map", f"0:{index}", "-f", "null", "-"]
    rc, err, timed_out, sec = _run_capture(cmd, timeout)
    lines = _error_lines(err)
    return dict(start=start, dur=dur, rc=rc, timeout=timed_out, bound_s=round(timeout, 1),
                seconds=round(sec, 1), n_lines=len(lines), lines=lines[:20])


def track_check(video_obj, stream_id, full=None, *, kind="stream_order", threads=THREADS,
                windows=VIDEO_WINDOWS, seed=None, jobs=None):
    """The strict decode of ONE stream, with its numbers.

    full   None = the default per type: an AUDIO track is decoded whole, a VIDEO track by
           `windows` stratified 2-minute windows; True = whole; False = windows.
    -> dict(path, stream_index, codec_type, codec, mode, sound (True/False/None),
            verdict ('sound' | 'corrupt' | 'decoder_timeout'), rc, error_lines, n_error_lines,
            windows (one entry per process), seed, cost_s). A whole-track result is cached per
    (file, size, mtime, stream) in this process: `cached=True` on a reuse."""
    t0 = time.time()
    path, s = resolve_stream(video_obj, stream_id, kind)
    info = _probe(path)
    idx, ctype = s["index"], s.get("codec_type")
    media = _declared_end(info, s) or 0.0
    whole = full if full is not None else (ctype != "video")
    key = _file_key(path) + (idx, "strict_full") if whole else None
    with _cache_lock:
        if key is not None and key in _check_cache:
            return dict(_check_cache[key], cached=True)
    starts = None
    if not whole:
        if seed is None:
            seed = fc.path_seed(path)
        starts = fc.integrity_windows(media, windows, seed, WINDOW_S)
    if starts is None:
        mode = "full"
        runs = [_strict(path, idx, None, None, threads, _bound(media))]
    else:
        mode = "sampled"
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs or len(starts))) as ex:
            runs = list(ex.map(lambda st: _strict(path, idx, st, WINDOW_S, threads, _bound(WINDOW_S)),
                               starts))
    failed = [r for r in runs if not r["timeout"] and (r["rc"] != 0 or r["n_lines"] > 0)]
    timed = [r for r in runs if r["timeout"]]
    if failed:
        verdict, sound = "corrupt", False
    elif timed:
        verdict, sound = "decoder_timeout", None
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
           f"processes={len(runs)} failed={len(failed)} timeouts={len(timed)} seed={seed} "
           f"cost_s={res['cost_s']}" + (f" first=«{lines[0][:200]}»" if lines else ""))
    _log(msg, always=(verdict != "sound"))
    if key is not None and verdict != "decoder_timeout":
        with _cache_lock:
            if len(_check_cache) >= _CACHE_MAX:
                _check_cache.clear()
            _check_cache[key] = res
    return res


def track_is_sound(video_obj, stream_id, full=None, **kw):
    """True when ffmpeg's strict decode of the track reports nothing, False when it reports a
    decoder error or a non-zero exit. Raises `tools.decoder_timeout` when a decode ran past
    its bound -- a statement about the host, never a False. Keywords: see `track_check`."""
    r = track_check(video_obj, stream_id, full, **kw)
    if r["verdict"] == "decoder_timeout":
        w = next(x for x in r["windows"] if x["timeout"])
        raise _timeout_error(w["bound_s"], f"strict decode of {r['path']} stream {r['stream_index']}")
    return r["sound"]


# ---------------------------------------------------------------- silence maps

def _decode_blocks(path, index, timeline, threads, timeout):
    """Decode one stream to 8 kHz mono, reduced on the fly to 400 ms mean-square blocks.
    -> (ms blocks np.float64, n_samples, rc, stderr_tail, timed_out)."""
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
    """The silences of one track. -> dict(path, stream_index, timeline, start_s, length_s,
    end_s, block_s, silences [[start, end, digital]], clicks, content_first_s,
    content_last_s, last_silence_start_s, rc, cost_s, db (np array, one value per block)).
    Cached per (file, size, mtime, stream, timeline) in this process."""
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
    """[(i0, i1)) runs of True."""
    if not mask.size:
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _spans(db, start):
    b = SILENCE_BLOCK_S
    content = db > SILENCE_DB
    # clicks: a content run < CLICK_S between two silences >= CLICK_GAP_S (or a file edge)
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
    """The map's silences plus what the FILE plays where the track has no packet: silence
    from 0 to the track's first sample and from its last sample to `play_end` (the file's
    video end). Adjacent spans merged."""
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
    """The silences of `x` (>= BAD_SILENCE_S) that `y` does not have: at the corresponding
    instants (y's instant = x's + shift_s) `y` plays sound (> CONTENT_DB) WITHOUT A BREAK for
    >= BAD_SILENCE_S -- five seconds of the other track sounding inside five seconds or more
    of this one's silence. `y` is read over [0, max(its end, y_play_end)): no packet there =
    silence; beyond, it says nothing."""
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
    """The numbers behind `silences_agree`. `delay_ms`: B's instant of a sound minus A's
    instant of the same sound (the pipeline's first_delay_test delay between video_obj_1 = A
    and video_obj_2 = B). -> dict(agree, result ('A' | 'B' | 'AB' | None), a, b, a_only
    (A's silences B does not have), b_only, delay_ms, cost_s)."""
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
                    content_first_s=m["content_first_s"], content_last_s=m["content_last_s"],
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


def silences_agree(video_1, stream_a, video_2, stream_b, delay_ms=0, **kw):
    """(True, None) when the two silence maps agree (B shifted by delay_ms); (False, 'A') when
    only A has silences (>= 5 s) that B does not have, (False, 'B') when only B has,
    (False, 'AB') when each has. Keywords and numbers: `silence_report`."""
    r = silence_report(video_1, stream_a, video_2, stream_b, delay_ms, **kw)
    return (r["agree"], r["result"])


# ---------------------------------------------------------------- (C) the video

def _frame_probe(path, index, start, n=12, timeout=120):
    cmd = [_bin("ffprobe"), "-v", "error", "-select_streams", str(index), "-read_intervals",
           f"{start:.3f}%+#{n}", "-show_entries", "frame=width,height,best_effort_timestamp_time",
           "-of", "csv=p=0", path]
    rc, out, err = fc._run(cmd, timeout=timeout)
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
    try:
        a, b = str(x).split("/")
        return float(a) / float(b) if float(b) else None
    except (ValueError, TypeError):
        return None


def _video_consistency(video_obj, path, vs, starts):
    """Frame count vs duration x fps, and resolution / frame period at every window start."""
    info = _probe(path)
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
    """The numbers behind `video_is_sound`: the strict decode of the video stream (sampled
    by default, 26.9.12) and the consistency of its frame count, resolution and frame period.
    -> dict(sound, verdict, reasons, integrity (track_check dict), consistency, cost_s)."""
    t0 = time.time()
    path = _path_of(video_obj)
    info = _probe(path)
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
    verdict = "corrupt" if reasons else ("decoder_timeout" if integ["verdict"] == "decoder_timeout"
                                         else "sound")
    res = dict(path=path, stream_index=vs["index"], sound={"sound": True, "corrupt": False}.get(verdict),
               verdict=verdict, reasons=reasons, integrity=integ, consistency=facts,
               cost_s=round(time.time() - t0, 1))
    _log(f"video_check path={path} stream={vs['index']} verdict={verdict} reasons={reasons[:3]} "
         f"cost_s={res['cost_s']}", always=verdict != "sound")
    return res


def video_is_sound(video_obj, full=False, **kw):
    """True when the video stream decodes strictly without a line on its sampled windows (or
    whole with full=True) AND its frame count, resolution and frame period are consistent.
    Raises `tools.decoder_timeout` when a window ran past its bound and nothing failed."""
    r = video_check(video_obj, full, **kw)
    if r["verdict"] == "decoder_timeout":
        raise _timeout_error(_bound(WINDOW_S), f"strict video probe of {r['path']}")
    return r["sound"]


# ---------------------------------------------------------------- (D) after a conversion

def _packet_extent(path, index, timeout):
    """One pass over the stream's packets, no decode: (first pts, last pts+duration, sum of
    durations). The SUM is the length mkvmerge gives the track when it re-times it from its
    frames (Chainsaw: the legacy product's packets end at 198.9 s, their durations sum to
    308.0 s, mkvmerge writes 308.0 s)."""
    cmd = [_bin("ffprobe"), "-v", "error", "-select_streams", str(index), "-show_entries",
           "packet=pts_time,duration_time", "-of", "csv=p=0", path]
    rc, out, err = fc._run(cmd, timeout=timeout)
    first, last, total = None, None, 0.0
    for line in out.decode("ascii", "replace").splitlines():
        p = line.strip().split(",")
        if len(p) < 2:
            continue
        t, du = fc._num(p[0]), fc._num(p[1]) or 0.0
        total += du
        if t is None:
            continue
        first = t if first is None else min(first, t)
        last = t + du if last is None else max(last, t + du)
    return first, last, total


def _wav(path, index, start_abs, dur, out, threads=THREADS):
    fmt_start = fc._num(_probe(path).get("format", {}).get("start_time")) or 0.0
    ss = max(0.0, start_abs - fmt_start)
    cmd = [_bin("ffmpeg"), "-nostdin", "-v", "error", "-y", "-threads", str(threads),
           "-ss", f"{ss:.3f}", "-t", f"{dur:.3f}", "-i", path, "-map", f"0:{index}",
           "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out]
    rc, _, _ = fc._run(cmd, timeout=_bound(dur))
    return rc == 0 and os.path.exists(out) and os.path.getsize(out) > 44 + 16000


def _fidelity(orig_path, oi, prod_path, pi, d, lo, hi, m_orig):
    """Chromaprint similarity of 60 s windows of the product and the original shifted by d."""
    if not shutil.which(_bin("fpcalc")) and not os.path.exists(_bin("fpcalc")):
        return None, []
    tmp = tempfile.mkdtemp(prefix="integrity_fp_")
    sims = []
    try:
        span = hi - lo
        if span < FIDELITY_WINDOW_S + 1:
            return None, []
        for fr in FIDELITY_FRACTIONS:
            t = lo + fr * (span - FIDELITY_WINDOW_S)
            # skip a window that is silence in the original: chromaprint of silence is noise
            lv = _levels_at(m_orig, np.arange(t - d, t - d + FIDELITY_WINDOW_S, SILENCE_BLOCK_S))
            if np.nanmean(lv > CONTENT_DB) < 0.5:
                continue
            wp, wo = os.path.join(tmp, f"p{fr}.wav"), os.path.join(tmp, f"o{fr}.wav")
            if not (_wav(prod_path, pi, t, FIDELITY_WINDOW_S, wp)
                    and _wav(orig_path, oi, t - d, FIDELITY_WINDOW_S, wo)):
                continue
            fa, fb = fc._fpcalc(wp, FIDELITY_WINDOW_S), fc._fpcalc(wo, FIDELITY_WINDOW_S)
            s, sh = fc._fp_similarity(fa, fb, max_shift=4, min_overlap=100)
            if s is not None:
                sims.append(dict(at_s=round(t, 1), similarity=round(s, 4), shift_items=sh))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not sims:
        return None, sims
    return float(np.median([x["similarity"] for x in sims])), sims


def conversion_report(original_video_obj, original_stream, produced_path, produced_stream,
                      delay_ms, expected_duration_s=None, *, kind="stream_order",
                      produced_kind="stream_order", fidelity=True, threads=THREADS):
    """The numbers behind `conversion_preserved`. `delay_ms`: the produced track's instant =
    the original's + delay_ms (mkvmerge-style `delay_to_put`)."""
    t0 = time.time()
    op, os_ = resolve_stream(original_video_obj, original_stream, kind)
    pp, ps = resolve_stream(produced_path, produced_stream, produced_kind)
    d = float(delay_ms) / 1000.0
    # PRIMARY (owner 2026-09-25 23:4x, adopted): the duration bound and the content
    # comparison by chromaprint; the silence maps are the EXPLANATION when one fails, and the
    # check itself only when fpcalc is absent.
    o_first, o_last, _ = _packet_extent(op, os_["index"], _bound(_declared_end(_probe(op), os_)))
    p_first, p_last, p_sum = _packet_extent(pp, ps["index"], _bound(_declared_end(_probe(pp), ps)))
    o_end = o_last if o_last is not None else (_declared_end(_probe(op), os_) or 0.0)
    expected = o_end + d
    if expected_duration_s is not None:
        expected = min(expected, float(expected_duration_s))
    p_start = p_first if p_first is not None else 0.0
    p_frames_end = p_start + p_sum
    failures, explanation = [], []
    if abs(p_frames_end - expected) > CONVERSION_DURATION_TOL_S:
        failures.append(f"conversion_duration {p_frames_end:.3f} s expected {expected:.3f} s "
                        f"(sum of the frames, as mkvmerge re-times the track)")
    if p_last is not None and abs(p_last - expected) > CONVERSION_DURATION_TOL_S:
        failures.append(f"conversion_duration {p_last:.3f} s expected {expected:.3f} s "
                        f"(last packet)")
    sim, sims, mo = None, [], None
    have_fpcalc = bool(shutil.which(_bin("fpcalc")) or os.path.exists(_bin("fpcalc")))
    if fidelity and have_fpcalc and not failures:
        mo = silence_map(op, os_["index"], timeline="timestamps", threads=threads)
        lo = max(p_start, mo["start_s"] + d)
        hi = min(p_frames_end, mo["end_s"] + d)
        sim, sims = _fidelity(op, os_["index"], pp, ps["index"], d, lo, hi, mo)
        if sim is not None and sim < FIDELITY_MIN:
            failures.append(f"conversion_fidelity {sim:.4f}")
    created = []
    if failures or not have_fpcalc or not fidelity or sim is None:
        mo = mo or silence_map(op, os_["index"], timeline="timestamps", threads=threads)
        mp = silence_map(pp, ps["index"], timeline="samples", threads=threads)
        created = _unmatched(mp, mo, -d, None, None)
        if created:
            c = created[0]
            line = (f"conversion_created_silence at {c['start_s']:.1f} s ({c['length_s']:.1f} s "
                    f"of silence where the original plays {c['other_sound_s']:.1f} s of sound)")
            (explanation if failures else failures).append(line)
    reason = "; ".join(failures + explanation) if failures else (
        f"conversion_preserved end {p_frames_end:.3f} s expected {expected:.3f} s"
        + (f", fidelity {sim:.4f}" if sim is not None else ", fidelity unmeasured (silence map)"))
    rep = dict(preserved=not failures, reason=reason, failures=failures, explanation=explanation,
               original=dict(path=op, stream_index=os_["index"], end_s=o_end),
               produced=dict(path=pp, stream_index=ps["index"], start_s=p_start,
                             end_frames_s=p_frames_end, end_last_packet_s=p_last),
               expected_end_s=expected, delay_ms=float(delay_ms), created_silences=created,
               fidelity=sim, fidelity_windows=sims, cost_s=round(time.time() - t0, 1))
    _log(f"conversion_preserved original={op}#{os_['index']} produced={pp}#{ps['index']} "
         f"delay_ms={float(delay_ms):.3f} preserved={rep['preserved']} reason={rep['reason']} "
         f"cost_s={rep['cost_s']}", always=not rep["preserved"])
    return rep


def conversion_preserved(original_video_obj, original_stream, produced_path, produced_stream,
                         delay_ms, expected_duration_s=None, **kw):
    """(True, 'conversion_preserved ...') when the produced track ends where the original +
    delay ends (within 500 ms, bounded by expected_duration_s) and sounds the same as the
    shifted original (fpcalc median >= 0.98; without fpcalc: no silence >= 5 s the shifted
    original does not have); else (False, 'conversion_duration ...' | 'conversion_fidelity
    <x>' | 'conversion_created_silence at <s> ...'), the failed primary check first, then the
    silence that explains it. Numbers: `conversion_report`."""
    r = conversion_report(original_video_obj, original_stream, produced_path, produced_stream,
                          delay_ms, expected_duration_s, **kw)
    return r["preserved"], r["reason"]


def track_facts(video_obj, stream_id, **kw):
    """What `generate_new_file` stores in data_to_save for a track: its end, size, start,
    content end and the start of its last (trailing) silence, from the silence map."""
    path, s = resolve_stream(video_obj, stream_id, kw.pop("kind", "stream_order"))
    m = silence_map(path, s["index"], **kw)
    size = fc._num(fc._tag(s, "NUMBER_OF_BYTES"))
    return dict(original_duration=round(m["end_s"], 3), original_start=round(m["start_s"], 3),
                original_size=int(size) if size else None,
                original_content_last=m["content_last_s"],
                original_last_silence_start=round(m["last_silence_start_s"], 3))


# ---------------------------------------------------------------- (E) the conversion's stderr

def conversion_stderr_is_clean(stderr_text, *, ignore_non_audio=True):
    """(True, []) when the conversion's stderr carries none of FATAL_CONVERSION_PATTERNS;
    else (False, the offending lines, first 50). ffmpeg writes progress with carriage
    returns: they split lines too. ignore_non_audio: a pattern on a subtitle / video output
    stream line (`[sost#` / `[vost#`) does not count."""
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
