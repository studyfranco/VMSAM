#!/usr/bin/env python3
"""Do these two files contain the same programme? A screen, not a verdict.

PRIVATE — reads real media paths. Never inside the repository.

Why this exists
---------------
The Lead asked whether the misfiling found in one folder occurs in the other 37.
Two cheaper tests were tried first and both are wrong for this question:

* **by filename** — folders legitimately mix title conventions for the SAME work
  (a romaji-named folder holding an English-titled release, or a Chinese-titled
  one). Screening on names flags all of those. Measured on this corpus: it would
  be mostly false positives.
* **by duration** — measured on folder 103, the misfiled files and the correctly
  named ones show the SAME delta against the master (~-129 s on episode 23, both
  families). Duration mismatch has several causes and "different programme" is
  only one of them; the largest failure family in this campaign, different cuts,
  produces it too.

So this compares **content**: perceptual hashes of frames sampled across both
files, matched any-to-any so that a constant offset does not matter.

What it reports, and what it does not
-------------------------------------
`score` = the median, over the candidate's sampled frames, of the smallest
Hamming distance to ANY master frame (64-bit pHash, so 0..64).

* same programme  -> most candidate frames find a near-identical master frame
  somewhere, so the median is low, and an offset does not change that.
* different programme -> no frame matches anything, so the median is high.

**This is a screen.** A low score is strong evidence of shared content; a high
score is a CANDIDATE for misfiling that must then be verified by looking at the
frames. It is not a verdict, and it is not a delay measurement — it says nothing
about alignment, only about whether the two files are the same work.

Validated against two cases whose answer was established by eye FIRST:
id 6 (same content, frame-identical at 700 s) and id 237 (different series).
Both answers must be produced before any sweep result is believed.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

N_FRAMES = 20
LO, HI = 0.08, 0.92          # sample the middle, avoiding logos and end cards
SIZE = 32


def duration(path):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True, timeout=120)
    v = p.stdout.strip()
    return float(v) if v and v != "N/A" else None


def _frame(path, t):
    """One greyscale SIZE*SIZE frame at t seconds, or None."""
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", path,
         "-frames:v", "1", "-vf", f"scale={SIZE}:{SIZE}", "-pix_fmt", "gray",
         "-f", "rawvideo", "-"],
        capture_output=True, timeout=180)
    if len(p.stdout) != SIZE * SIZE:
        return None
    return np.frombuffer(p.stdout, dtype=np.uint8).reshape(SIZE, SIZE).astype(float)


def _phash(img):
    """64-bit DCT perceptual hash as a boolean array."""
    from scipy.fftpack import dct
    d = dct(dct(img, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    flat = d.flatten()[1:]                      # drop DC
    return flat > np.median(flat)


def hashes(path, n=N_FRAMES):
    dur = duration(path)
    if not dur or dur < 30:
        return None
    ts = [dur * (LO + (HI - LO) * i / (n - 1)) for i in range(n)]
    with ThreadPoolExecutor(max_workers=6) as ex:
        frames = list(ex.map(lambda t: _frame(path, t), ts))
    hs = [_phash(f) for f in frames if f is not None]
    return hs or None


def compare(cand_hashes, mast_hashes):
    """Median over candidate frames of the min Hamming distance to any master frame."""
    if not cand_hashes or not mast_hashes:
        return None
    C = np.array(cand_hashes)
    M = np.array(mast_hashes)
    # pairwise Hamming: (nC, nM)
    d = (C[:, None, :] != M[None, :, :]).sum(axis=2)
    return float(np.median(d.min(axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True,
                    help="JSON list of {id, file_path, master_path}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    pairs = json.load(open(a.pairs))
    cache = {}

    def get(path):
        if path not in cache:
            cache[path] = hashes(path)
        return cache[path]

    out = []
    for i, p in enumerate(pairs, 1):
        ch = get(p["file_path"])
        mh = get(p["master_path"]) if p.get("master_path") else None
        s = compare(ch, mh)
        rec = dict(p)
        rec["score"] = s
        # three outcomes, not two: None means the instrument could not run here
        rec["verdict"] = ("unreadable" if s is None else
                          "same-content" if s <= 12 else
                          "candidate-different" if s >= 22 else "ambiguous")
        out.append(rec)
        print(f"[{i}/{len(pairs)}] id={p.get('id')} score={s} {rec['verdict']}",
              file=sys.stderr, flush=True)

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
