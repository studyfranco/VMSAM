"""Probe helpers of media files: ffprobe in JSON, tags, numbers and packet spans."""

import json
import math
import shutil
import subprocess


def _bin(name):
    return shutil.which(name) or name


def _run(cmd, timeout=None):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           stdin=subprocess.DEVNULL, timeout=timeout)
        return p.returncode, p.stdout, p.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as e:
        return -9, e.stdout or b"", "timeout after %s s" % timeout


def _parse_hms(s):
    try:
        h, m, sec = str(s).strip().split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    except (ValueError, AttributeError):
        return None


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _tag(stream, key):
    """Return a stream tag, case-insensitive, also matching KEY-<lang> variants."""
    for k, v in (stream.get("tags") or {}).items():
        if k.upper() == key or k.upper().startswith(key + "-"):
            return v
    return None


def probe(path):
    """Return ffprobe's streams, format and chapters as a dict.

    Raises:
        RuntimeError: ffprobe failed.
    """
    rc, out, err = _run([_bin("ffprobe"), "-v", "error", "-show_streams", "-show_format",
                         "-show_chapters", "-of", "json", path], timeout=300)
    if rc != 0:
        raise RuntimeError(f"ffprobe rc={rc}: {err.strip()[:300]}")
    return json.loads(out.decode("utf-8", "replace"))


def packet_spans(path):
    """Return {stream index: (first packet time, end of the last packet, its duration)}."""
    rc, out, err = _run([_bin("ffprobe"), "-v", "error", "-show_entries",
                         "packet=stream_index,pts_time,duration_time", "-of", "csv=p=0", path])
    if rc != 0:
        raise RuntimeError(f"ffprobe rc={rc}: {err.strip()[:300]}")
    spans = {}
    for line in out.decode("ascii", "replace").splitlines():
        index, pts, duration = (line.split(",") + ["", ""])[:3]
        start, length = _num(pts), _num(duration) or 0.0
        if start is None:
            continue
        first, end, last = spans.get(int(index), (start, start + length, length))
        if start + length >= end:
            end, last = start + length, length
        spans[int(index)] = (min(first, start), end, last)
    return spans
