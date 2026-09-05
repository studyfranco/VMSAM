#!/usr/bin/env python3
"""Offset profile across a WHOLE file, at sub-quantum resolution.

PRIVATE — reads real media paths. Never inside the repository.

What this is for
----------------
Two shapes the stage-1 record cannot show and the head sweep cannot reach:

  CLASS B  an island: the offset departs and RETURNS, anywhere in the file.
           Every stage-1 window spans both transitions, so every window reads
           constant. Demonstrated on one file (departs ~56 s, returns ~147 s).
  CLASS C  a sub-quantum step: below ~62 ms, so it moves no window's reading
           anywhere, at any position. Never sampled by anything.

The head sweep bounds neither: its grid is 20 s probes over 10-250 s at a 150 ms
threshold, so it cannot see the body, an island narrower than its spacing, or
anything sub-quantum. Those limits are its author's, stated, not inferred.

Reporting rules, learned the hard way today
-------------------------------------------
**The full deviant sequence is printed with positions — never a head, never a
count.** Another agent's sweep printed `deviant[:3]` and the return leg of an
island fell off the end, so a shape present in the data twice over was erased by
the summary. A cross-check cannot catch that, because a cross-check compares
measurements and a summary is not one.

**The label names the PATTERN, not the fact of deviation.** `flat`, `step`,
`island`, `scatter`. A label reading "deviates from body" cannot distinguish a
step from an island, which is exactly how the island was missed.

> A summary that cannot express a shape will not report it, and nothing
> downstream will ever know the shape was there.

What it does NOT do
-------------------
It measures offsets; it does not decide repairability. And a file whose two sides
share no content produces meaningless offsets — those are excluded upstream by
family, not re-tested here.
"""

import json
from redact import safe_text as _safe
import subprocess
import sys

import numpy as np
from scipy.signal import fftconvolve

SR = 8000
PROBE = 20.0          # seconds per probe
MAXLAG = 3.0          # seconds of lag searched
MIN_CORR = 0.25       # below this the probe is unreadable, not deviant


def pcm(path, stream, ss, dur):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{ss:.3f}", "-t", f"{dur:.3f}",
         "-i", path, "-map", f"0:a:{stream}", "-ac", "1", "-ar", str(SR),
         "-c:a", "pcm_s16le", "-f", "s16le", "-"],
        capture_output=True, timeout=300)
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float64)


def offset_at(mpath, ms, cpath, cs, t, dur=PROBE, maxlag=MAXLAG, base_ms=0.0):
    """(lag_ms, corr). Searches base_ms +- maxlag, not 0 +- maxlag."""
    M = pcm(mpath, ms, t, dur)
    C = pcm(cpath, cs, t - base_ms / 1000.0, dur)
    n = min(len(M), len(C))
    ml = int(maxlag * SR)
    if n <= ml + SR:                      # guard: maxlag must fit inside the probe
        return None, None
    M, C = M[:n] - M[:n].mean(), C[:n] - C[:n].mean()
    d = np.linalg.norm(M) * np.linalg.norm(C)
    if d == 0:
        return None, None
    corr = fftconvolve(M, C[::-1], mode="full")
    mid = len(C) - 1
    seg = corr[mid - ml:mid + ml + 1] / d
    j = int(np.argmax(seg))
    return (np.arange(-ml, ml + 1)[j] / SR * 1000.0) + base_ms, float(seg[j])


def classify(seq, quantum_ms):
    """seq = [(t, lag, corr)]. Returns (label, detail). Pattern, not deviation.

    TWO DEFECTS FIXED after a false positive on a real file, caught by inspecting a
    positive rather than counting it:

    1. COVERAGE WAS SILENT. Unreadable probes were dropped and the label computed on
       whatever remained, so a file readable only in its first third was labelled from
       that third with nothing saying so. Coverage is now reported and a file whose
       readable probes do not span most of its duration is labelled `partial-coverage`,
       which is a THIRD outcome and not a clean one.

    2. A FIXED CORRELATION FLOOR IS NOT ENOUGH. On that file, twelve probes read 0.76-0.97
       and the rest collapsed to 0.01-0.15 -- and ONE stray probe at 0.41 cleared the fixed
       0.25 floor, deviated by 58 ms from the others, and was labelled a sub-quantum step.
       A probe at 0.41 surrounded by 0.02 is not a measurement, it is a spurious peak.
       A deviant probe must now also be strong RELATIVE TO THAT FILE'S OWN baseline.
    """
    readable = [(t, l, c) for t, l, c in seq if l is not None and c >= MIN_CORR]
    if len(readable) < 4:
        return "unreadable", f"{len(readable)} readable probes of {len(seq)}"
    span_all = seq[-1][0] - seq[0][0]
    span_ok = readable[-1][0] - readable[0][0]
    frac = len(readable) / len(seq)
    med_c = float(np.median([c for _, _, c in readable]))
    # a deviant probe must be at least half as strong as this file's own median
    floor = max(MIN_CORR, 0.5 * med_c)
    good = [(t, l) for t, l, c in readable if c >= floor]
    if len(good) < 4:
        return "partial-coverage", (f"{len(good)} probes above this file's own floor "
                                    f"{floor:.2f}, of {len(seq)}")
    if frac < 0.7 or span_ok < 0.7 * span_all:
        return "partial-coverage", (f"{len(readable)}/{len(seq)} probes readable, "
                                    f"covering {span_ok:.0f} s of {span_all:.0f} s")
    lags = [l for _, l in good]
    base = float(np.median(lags))
    dev = [i for i, (_, l) in enumerate(good) if abs(l - base) > 25.0]
    if not dev:
        return "flat", f"median {base:.1f} ms, spread {max(lags)-min(lags):.1f} ms"
    runs, cur = [], [dev[0]]
    for a, b in zip(dev, dev[1:]):
        (cur.append(b) if b == a + 1 else (runs.append(cur), cur := [b]))
    runs.append(cur)
    touches_end = any(r[0] == 0 or r[-1] == len(good) - 1 for r in runs)
    span = max(abs(l - base) for _, l in [good[i] for i in dev])
    sub = span < quantum_ms / 2
    if len(runs) == 1 and not touches_end:
        return ("island-subquantum" if sub else "island"), \
               f"probes {good[runs[0][0]][0]:.0f}-{good[runs[0][-1]][0]:.0f} s, depth {span:.1f} ms"
    if len(runs) == 1:
        return ("step-subquantum" if sub else "step"), f"depth {span:.1f} ms"
    return "scatter", f"{len(runs)} deviant runs, max depth {span:.1f} ms"


def baseline_span(rec):
    """Baseline search half-width, SIZED FROM THE RECORD'S OWN DELAYS.

    THE FOURTH SPAN DEFECT IN THIS FILE, and the same family as the first three.
    The baseline search was a fixed +-60 s. Measured over the full corpus:

        recorded |delay| < 10 s   n=222   did-not-run  4 %
        recorded |delay| >= 60 s  n= 53   did-not-run 98 %
        52 of the 72 did-not-run rows are files whose offset lies OUTSIDE the search

    A fixed span cannot find an offset larger than itself, and the record says how
    large it is before the search starts. The fine window was fixed at +-3 s until a
    peer's positive controls showed 37 of 100 files exceeded it; this is the same
    lesson one stage earlier, in the same function I fixed then.
    """
    v = [abs(x) for p in rec.get("per_pair", {}).values() for x in p["delays"]]
    if not v:
        return 60.0
    return float(min(300.0, max(60.0, max(v) / 1000.0 * 1.4 + 5.0)))


def find_baseline(rec, maxlag=None, dur=120.0):
    if maxlag is None:
        maxlag = baseline_span(rec)
    # the guard needs the probe to contain its own search: n > ml + SR
    dur = max(dur, (maxlag + 5.0) * 2.2)
    """One wide, long probe to locate the pair's gross offset before fine probing.

    DEFECT THIS FIXES, and it invalidated 54 files of a run: the fine probes searched
    +-3000 ms around ZERO. Another agent hypothesised from my own garbage output that
    my span was the cause, and named a file to test it on. Measured: at +-3 s that
    file reads 2395 / 667 / -2375 / 123 ms at correlations 0.03-0.17; at +-15 s it
    reads 6032 / 7046 / 8059 / 8059 at 0.40-0.69. The offsets were real and outside
    my window.

    64 of 100 sampled files carry a recorded delay beyond +-3000 ms, so the scan was
    reporting the edge of its own search on most of its frame. The symptom -- a clean
    region then garbage -- is indistinguishable from a real structural break, which is
    exactly how it survived inspection on the first file I checked.

    Searching relative to a measured baseline is also far cheaper than widening the
    fine search: one long probe, then short ones.
    DEFECT FIXED: a SINGLE baseline probe. On one file it landed at correlation
    0.1436 -- a non-answer -- and the fine probes then searched around a wrong
    baseline, so 22 of 30 came back unreadable and the file was labelled
    `partial-coverage`. That label reads as a property of the MEDIA and was in fact a
    property of the INSTRUMENT. The same conflation I fixed one level up, one level
    down, found the same way: by inspecting a result instead of counting it.

    Now: try several positions, keep the strongest. If none is confident, say so with
    its own label rather than letting the failure masquerade as thin coverage.
    """
    d = min(rec["err_audio_s"], rec["mst_audio_s"])
    best = (0.0, 0.0)
    for frac in (0.25, 0.5, 0.10, 0.75):
        t = max(2.0, min(frac * d, d - dur - 2.0))
        if t < 2.0:
            continue
        lag, corr = offset_at(rec["master_path"], rec["ms"], rec["file_path"],
                              rec["cs"], t, dur=dur, maxlag=maxlag)
        if lag is not None and corr > best[1]:
            best = (lag, corr)
        if best[1] >= 0.40:          # confident enough, stop paying for more
            break
    return best


def fine_window(rec):
    """Fine-probe half-width, SIZED FROM THE RECORD'S OWN STEP SPAN.

    THE THIRD SPAN DEFECT, and it is the same family as the first two. I fixed the
    BASELINE search (+-60 s) and left the FINE window at a fixed +-3 s. A step
    LARGER than 3 s therefore cannot be measured: probes on the far plateau fall
    outside the window and come back unreadable, or worse, land on a spurious peak
    and produce a DEPTH THAT LOOKS LIKE A MEASUREMENT.

    Found by two positive controls another agent supplied -- files with a ~5 s step
    established by a pHash video probe and a change-point locator sharing no code
    with each other or with me. My scan returned `partial-coverage` on BOTH. A
    positive control is the only thing that could have found this: on unknown files
    the output was self-consistent and wrong.

    37 of the 100 sampled files carry a record step span exceeding 3 s. Measured
    consequence on six of them: depths understated by 2000-3100 ms, clustering near
    2000 ms -- which is a CEILING OF MINE, not a property of the media, and it is
    what my own quantum-artefact check was reading as `16.01 quanta`.
    """
    span = 0.0
    for p in rec.get("per_pair", {}).values():
        span = max(span, max(p["delays"]) - min(p["delays"]))
    # 1.3x headroom: the baseline may sit on either plateau, and the step must fit
    # with room for the peak to be resolved rather than clipped at the edge.
    return min(15.0, max(MAXLAG, span / 1000.0 * 1.3 + 1.0))


def scan(rec, n_probes=10):
    dur = min(rec["err_audio_s"], rec["mst_audio_s"])
    fine = fine_window(rec)
    # the guard in offset_at needs maxlag to fit inside the probe: n > ml + SR
    probe = max(PROBE, (fine + 2.0) * 2.0)
    # NO PRIVILEGED REGIONS. An earlier version started at 6 % of duration and
    # ended at 94 %, and on the one file whose answer was known it MISSED the
    # island's leading edge (island 56-147 s; first probe landed at 85 s), so a
    # deviant run that began before the grid read as a step touching the end.
    # Two agents' sweeps have now failed the same way -- one never probing the
    # tail, mine never probing the head -- each having inherited a start offset
    # from somewhere. Cover the file.
    lo, hi = 2.0, max(2.0, dur - probe - 2.0)
    base, base_corr = find_baseline(rec)
    if base_corr < 0.25:
        return {"id": rec["id"], "label": "baseline-not-found",
                "detail": f"strongest baseline probe only {base_corr:.3f} -- INSTRUMENT "
                          f"DID NOT RUN, not a property of the media",
                "baseline_ms": round(base, 1), "baseline_corr": round(base_corr, 4),
                "sequence": []}
    ts = [lo + (hi - lo) * i / (n_probes - 1) for i in range(n_probes)]
    seq = []
    for t in ts:
        lag, corr = offset_at(rec["master_path"], rec["ms"],
                              rec["file_path"], rec["cs"], t,
                              dur=probe, maxlag=fine, base_ms=base)
        seq.append((t, lag, corr))
    label, detail = classify(seq, rec.get("quantum_ms", 125.0))

    # ESCALATE ON POOR COVERAGE, rather than trusting the record's span.
    #
    # The window above is sized from the RECORD -- and the record is the very thing
    # whose blind spots this scan exists to bound. On id 266 the record shows a
    # 2000 ms span where a locator and a video probe agree the true displacement is
    # 3171 ms: the record UNDERSTATES it by exactly what its geometry could not see.
    # My window then inherits that blindness. That is the defect I wrote down in
    # NOTE_census.md -- "an independent instrument aimed by a dependent one is not
    # independent" -- and then built a fourth instance of.
    #
    # Poor coverage is the OBSERVABLE symptom of a window too small, so escalate on
    # the symptom, which owes the record nothing. Cost is paid only where it fails.
    tries = 0
    while label in ("partial-coverage", "unreadable") and fine < 12.0 and tries < 1:
        tries += 1
        fine = min(12.0, fine * 2.5)
        probe = max(PROBE, (fine + 2.0) * 2.0)
        hi2 = max(lo, dur - probe - 2.0)
        ts = [lo + (hi2 - lo) * i / (n_probes - 1) for i in range(n_probes)]
        seq = [(t,) + tuple(offset_at(rec["master_path"], rec["ms"],
                                      rec["file_path"], rec["cs"], t,
                                      dur=probe, maxlag=fine, base_ms=base))
               for t in ts]
        label, detail = classify(seq, rec.get("quantum_ms", 125.0))
        detail += f" [escalated to +-{fine:.1f} s]"
    return {"id": rec["id"], "label": label, "detail": detail,
            "fine_maxlag_s": round(fine, 2), "probe_s": round(probe, 1),
            "baseline_ms": round(base, 1), "baseline_corr": round(base_corr, 4),
            # FULL sequence, never truncated -- see module docstring
            "sequence": [(round(t, 1), None if l is None else round(l, 1),
                          None if c is None else round(c, 4)) for t, l, c in seq]}


if __name__ == "__main__":
    recs = json.load(open(sys.argv[1]))
    out = []
    for i, r in enumerate(recs, 1):
        try:
            res = scan(r)
        except Exception as e:
            res = {"id": r["id"], "label": "instrument-failed", "detail": _safe(e),
                   "sequence": []}
        out.append(res)
        print(f"[{i}/{len(recs)}] id={res['id']} {res['label']}: {res['detail']}",
              file=sys.stderr, flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(f"wrote {sys.argv[2]}", file=sys.stderr)
