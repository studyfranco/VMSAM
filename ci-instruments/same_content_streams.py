#!/usr/bin/env python3
"""Pass 2 — compare EVERY audio stream pair, not just the first of each.

PRIVATE — reads real media paths. Never inside the repository.

Why this pass exists, and it was measured not guessed
-----------------------------------------------------
`fpcalc` fingerprints the FIRST audio stream of a file. Pass 1 therefore compared
whatever stream happened to be first on each side, and on error id 349 that was a
Japanese original against an ENGLISH DUB: peak 0.0849, which pass 1 called
"candidate-different" for a pair that is the same work.

Measured on that pair:

    candidate : 1 stream  -- jpn
    master    : 4 streams -- eng, jpn, jpn, jpn   <- fpcalc took eng

That is the "weak correlation / differently mixed dub" family, not misfiling, and
it is exactly the false positive that would inflate a misfiling count.

So this pass re-scores only the pairs pass 1 could not call, taking the BEST peak
over all (candidate stream x master stream) combinations. A dub-versus-original
pair then finds its matching language and scores high; a genuinely different
programme has no stream pair that correlates.

Cost is why it is a second pass: this extracts audio, pass 1 does not.

Still a screen, not a verdict
-----------------------------
A high peak is conclusive that the two share content. A low peak across every
stream pair is a CANDIDATE for misfiling, to be confirmed by looking at frames.
Music-only stretches, heavy re-encodes and commentary tracks can all depress a
peak without the content differing.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from same_content import LENGTH, MIN_OVERLAP, peak

TMP = "/tmp/claude-1000/ci-streams"


def audio_streams(path):
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index:stream_tags=language",
             "-of", "json", path], capture_output=True, text=True, timeout=120)
        d = json.loads(p.stdout or "{}")
        return [(i, s.get("tags", {}).get("language", "?"))
                for i, s in enumerate(d.get("streams", []))]
    except Exception:
        return []


def fingerprint_stream(path, n):
    """Fingerprint audio stream n. Extracts to a temp wav, then deletes it.

    /tmp is RAM shared by four agents, so the wav is removed in a finally block
    whatever happens.
    """
    os.makedirs(TMP, exist_ok=True)
    fd, wav = tempfile.mkstemp(suffix=".wav", dir=TMP)
    os.close(fd)
    try:
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-t", str(LENGTH), "-i", path,
             "-map", f"0:a:{n}", "-ac", "1", "-ar", "22050",
             "-c:a", "pcm_s16le", wav],
            capture_output=True, timeout=600)
        if p.returncode != 0 or not os.path.getsize(wav):
            return None
        q = subprocess.run(["fpcalc", "-raw", "-length", str(LENGTH), wav],
                           capture_output=True, text=True, timeout=300)
        for line in q.stdout.splitlines():
            if line.startswith("FINGERPRINT="):
                return [int(x) for x in line[12:].split(",") if x]
        return None
    except Exception:
        return None
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass


def all_fps(path, cap=4):
    st = audio_streams(path)[:cap]
    out = []
    for n, lang in st:
        fp = fingerprint_stream(path, n)
        if fp:
            out.append((lang, fp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--same", type=float, default=0.30)
    ap.add_argument("--diff", type=float, default=0.15)
    a = ap.parse_args()

    pairs = json.load(open(a.pairs))
    paths = sorted({p["file_path"] for p in pairs} |
                   {p["master_path"] for p in pairs if p.get("master_path")})
    print(f"per-stream fingerprinting {len(paths)} files...",
          file=sys.stderr, flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        fps = dict(zip(paths, ex.map(all_fps, paths)))

    out = []
    for i, p in enumerate(pairs, 1):
        A = fps.get(p["file_path"]) or []
        B = fps.get(p.get("master_path")) or []
        best, where = None, None
        for la, fa in A:
            for lb, fb in B:
                s = peak(fa, fb)
                if s is not None and (best is None or s > best):
                    best, where = s, f"{la}->{lb}"
        r = dict(p)
        r["peak_streams"] = None if best is None else round(best, 4)
        r["best_pair"] = where
        r["n_streams"] = f"{len(A)}x{len(B)}"
        r["verdict_streams"] = ("unreadable" if best is None else
                                "same-content" if best >= a.same else
                                "candidate-different" if best < a.diff
                                else "ambiguous")
        out.append(r)
        print(f"[{i}/{len(pairs)}] id={p.get('id')} peak={r['peak_streams']} "
              f"({where}) {r['verdict_streams']}", file=sys.stderr, flush=True)

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
