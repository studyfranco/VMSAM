"""Conformity of media files: probe helpers, packet spans, and the refusal of a broken merge input."""

import json
import math
import shutil
import subprocess

SILENCE_DB = -60.0                # digital silence is ~-121 dB
ISLAND_GAP_S = 60
CLICK_S = 3
TAIL_RULE_S = 300.0               # a gap over 5 min running to the end is a cut
AFTER_VIDEO_ERROR_S = 60.0        # healthy files end audio < ~45 s after the video


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


class ConformityError(Exception):
    """A merge input is cut or overlong: its audio cannot be aligned on its video."""


def require_conformity(path):
    """Raise ConformityError when an audio track runs a minute past the video, or every one stops
    five minutes before it. A single track shorter than the video is common and valid.
    """
    streams = probe(path).get("streams", [])
    video = [s["index"] for s in streams if s.get("codec_type") == "video"
             and not (s.get("disposition") or {}).get("attached_pic")]
    audio = [s["index"] for s in streams if s.get("codec_type") == "audio"]
    if not video or not audio:
        return
    spans = packet_spans(path)
    video_end = spans[video[0]][1]
    ends = {i: spans[i][1] for i in audio if i in spans}
    late = [f"audio track {i} ends at {e:.3f} s" for i, e in ends.items() if e > video_end + AFTER_VIDEO_ERROR_S]
    if ends and all(e < video_end - TAIL_RULE_S for e in ends.values()):
        late.append(f"every audio track ends by {max(ends.values()):.3f} s")
    if late:
        raise ConformityError(f"{path}: {'; '.join(late)}, the video ends at {video_end:.3f} s")
