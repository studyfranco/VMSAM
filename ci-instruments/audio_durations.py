#!/usr/bin/env python3
"""Re-probe every pair on the AUDIO axis, not the container's.

PRIVATE — reads real media paths. Never inside the repository.

Why this replaces sweep_durations.py's numbers
-----------------------------------------------
That script used `format=duration`, which is the CONTAINER duration, and in
Matroska the container duration is set by the longest stream — usually video.
dev-1 measured the consequence on error id 237:

    master : video 1501.250 s | audio eng/jpn/chi ALL 1436.977 s
    error  : audio jpn 1438.040 s

So the container delta is -63.21 s and the AUDIO delta is +1.06 s. The window
geometry is driven from audio (`get_shortest_audio_durations`), so audio is the
axis that matters, and the container figure is a property of that master's video
rather than of the pair. Every duration I quoted before this was on the wrong axis.

Matroska rarely fills the per-stream `duration` field and puts the value in a
`DURATION` tag instead, so this tries three sources and RECORDS WHICH ONE ANSWERED.
A container-derived fallback is kept and labelled rather than silently mixed in
with real audio durations — a reader with two outcomes describing a world with
three loses the third.

It also computes the speed ratio, because a low correlation does not distinguish
"different content" from "speed mismatch": docs/AUDIO_SPEED_POLICY.MD measured
uncorrected PAL-against-film pairs at fidelity 0.558-0.562, which is the same floor
a no-shared-content pair sits on. The separator available here is the duration
RATIO — a PAL/film relation is 25/23.976 = 1.042709.
"""

import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

RUNS = "/home/vmsam/src/VMSAM_HELP_AI/ci/runs"
PAL = 25.0 / 23.976            # 1.042709
TOL = 0.004                    # ratio tolerance for calling a speed relation


def _hms(s):
    m = re.match(r"(\d+):(\d+):(\d+(?:\.\d+)?)", s or "")
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def audio_duration(path):
    """(seconds, source). source is 'stream', 'tag' or 'container'."""
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index,duration:stream_tags=DURATION",
             "-of", "json", path], capture_output=True, text=True, timeout=120)
        st = json.loads(p.stdout or "{}").get("streams", [])
        vals, src = [], None
        for s in st:
            d = s.get("duration")
            if d and d != "N/A":
                vals.append(float(d))
                src = src or "stream"
                continue
            t = (s.get("tags") or {})
            t = t.get("DURATION") or t.get("duration")
            v = _hms(t) if t else None
            if v:
                vals.append(v)
                src = src or "tag"
        if vals:
            # shortest audio is what drives the window geometry
            return min(vals), src
        q = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path], capture_output=True, text=True, timeout=120)
        v = q.stdout.strip()
        return (float(v), "container") if v and v != "N/A" else (None, None)
    except Exception:
        return None, None


def main():
    rows = json.load(open(f"{RUNS}/sweep-durations.json"))
    paths = sorted({r["file_path"] for r in rows} |
                   {r["master_path"] for r in rows if r.get("master_path")})
    print(f"probing audio duration of {len(paths)} files...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=12) as ex:
        res = dict(zip(paths, ex.map(audio_duration, paths)))

    out = []
    for r in rows:
        e, es = res.get(r["file_path"], (None, None))
        m, ms = res.get(r["master_path"], (None, None)) if r.get("master_path") \
            else (None, None)
        n = dict(r)
        n["err_audio_s"], n["err_src"] = e, es
        n["mst_audio_s"], n["mst_src"] = m, ms
        if e and m:
            n["audio_delta_s"] = round(e - m, 3)
            n["ratio"] = round(m / e, 6)          # master / candidate
            rr = n["ratio"]
            n["speed_candidate"] = bool(
                abs(rr - PAL) < TOL or abs(rr - 1.0 / PAL) < TOL)
            n["container_derived"] = (es == "container" or ms == "container")
        else:
            n["audio_delta_s"] = n["ratio"] = None
            n["speed_candidate"] = None
            n["container_derived"] = None
        out.append(n)

    json.dump(out, open(f"{RUNS}/audio-durations.json", "w"), indent=1)
    src = {}
    for r in out:
        src[(r["err_src"], r["mst_src"])] = src.get((r["err_src"], r["mst_src"]), 0) + 1
    print("duration sources (err, mst):", src, file=sys.stderr)
    print(f"container-derived on either side: "
          f"{sum(1 for r in out if r['container_derived'])}", file=sys.stderr)
    print(f"speed candidates (ratio ~{PAL:.6f}): "
          f"{sum(1 for r in out if r['speed_candidate'])}", file=sys.stderr)
    print(f"wrote {RUNS}/audio-durations.json", file=sys.stderr)


if __name__ == "__main__":
    main()
