#!/usr/bin/env python3
"""Are these two files the same programme? Chromaprint peak over all offsets.

PRIVATE — reads real media paths. Never inside the repository.

The question this answers, and the two it does not
--------------------------------------------------
Answers: *is there any offset at which these two audio tracks agree?* That is the
same-work question, and it is offset-blind by construction.

Does NOT answer: what the delay is (the peak lag here is coarse and unvalidated —
use the pipeline for that), nor whether a pair is repairable.

Why the two cheaper screens were abandoned, both measured on this corpus
-----------------------------------------------------------------------
* **filenames** — folders legitimately mix title conventions for one work
  (romaji folder, English-titled release; Chinese-titled release). Screening on
  names is mostly false positives here.
* **durations** — in folder 103 the misfiled files and the correctly named ones
  show the SAME delta against the master (~-129 s on episode 23, both families).
  Duration mismatch has several causes and "different programme" is only one.
* **sparse frame pHash** — tried, and REFUTED against a known case: it called
  three pairs "different" that are plainly the same work under English vs romaji
  titles. Cause: 20 frames over ~22 min is 66 s apart, so with any offset the
  sampled frames never coincide. Its positive control had a ZERO offset, which is
  exactly the case that cannot expose the flaw. Recorded because it is the
  campaign's own lesson — a guard validated only on the easy case.

Chromaprint has none of that problem: it compares dense fingerprints at every
lag, so an offset is found rather than tripped over. It is also the instrument
the pipeline already trusts.

Known limitation, stated rather than hidden
-------------------------------------------
`fpcalc` reads the FIRST audio stream of each file. A same-content pair whose
first streams are different languages (an original against a dub) can therefore
score low and be flagged. That is the "weak correlation" family, not misfiling.
So a low peak is a CANDIDATE to be verified by looking at frames, never a verdict.
A high peak is conclusive the other way: different programmes do not correlate.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.signal import fftconvolve

LENGTH = 900          # seconds of audio to fingerprint, from the start
MIN_OVERLAP = 200     # points that must overlap for a lag to be scored


def fingerprint(path, length=LENGTH):
    """Chromaprint raw fingerprint as a list of ints, or None if it could not run."""
    try:
        p = subprocess.run(["fpcalc", "-raw", "-length", str(length), path],
                           capture_output=True, text=True, timeout=300)
        if p.returncode != 0:
            return None
        for line in p.stdout.splitlines():
            if line.startswith("FINGERPRINT="):
                return [int(x) for x in line[12:].split(",") if x]
    except Exception:
        return None
    return None


def _bits(fp):
    a = np.array(fp, dtype=np.uint32)
    b = ((a[:, None] >> np.arange(32)) & 1).astype(np.float32)
    return b * 2.0 - 1.0                      # +-1 so agreement is a dot product


def peak(fp_a, fp_b, min_overlap=MIN_OVERLAP):
    """Best bit-agreement over all lags, in [-1, 1]. 0 is chance, 1 identical."""
    if not fp_a or not fp_b:
        return None
    A, B = _bits(fp_a), _bits(fp_b)
    na, nb = len(A), len(B)
    if min(na, nb) < min_overlap:
        return None
    corr = fftconvolve(A, B[::-1, :], mode="full", axes=0).sum(axis=1)
    lags = np.arange(-(nb - 1), na)
    overlap = np.minimum(np.minimum(na, nb),
                         np.minimum(na - lags, nb + lags)).astype(float)
    valid = overlap >= min_overlap
    if not valid.any():
        return None
    score = corr[valid] / (32.0 * overlap[valid])
    return float(score.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--same", type=float, default=0.30,
                    help="peak at or above this = same content")
    ap.add_argument("--diff", type=float, default=0.15,
                    help="peak below this = candidate different content")
    a = ap.parse_args()

    pairs = json.load(open(a.pairs))
    paths = sorted({p["file_path"] for p in pairs} |
                   {p["master_path"] for p in pairs if p.get("master_path")})
    print(f"fingerprinting {len(paths)} files...", file=sys.stderr, flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        fps = dict(zip(paths, ex.map(fingerprint, paths)))
    bad = [p for p, v in fps.items() if not v]
    print(f"  fingerprinted {len(paths)-len(bad)}, failed {len(bad)}",
          file=sys.stderr, flush=True)

    out = []
    for i, p in enumerate(pairs, 1):
        s = peak(fps.get(p["file_path"]), fps.get(p.get("master_path")))
        r = dict(p)
        r["peak"] = None if s is None else round(s, 4)
        # three outcomes, not two -- "could not run" is its own answer
        r["verdict"] = ("unreadable" if s is None else
                        "same-content" if s >= a.same else
                        "candidate-different" if s < a.diff else "ambiguous")
        out.append(r)
        if i % 25 == 0 or i == len(pairs):
            print(f"  scored {i}/{len(pairs)}", file=sys.stderr, flush=True)

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
