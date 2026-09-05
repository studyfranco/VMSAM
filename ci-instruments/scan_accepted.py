#!/usr/bin/env python3
"""Measure the population the pipeline ACCEPTED, on same-language track pairs.

THE GAP THIS ADDRESSES. The error corpus is defined by an inclusion criterion that
excludes sub-quantum steps by construction, so no blind-spot figure derived from it
covers the files that shipped. vmsam-forensic: "the interesting population has always
been the successes."

WHY SAME-LANGUAGE ONLY. A positive control with a planted 40 ms step:
    cross-language pair : max error 73.5 ms  -- NOISE EXCEEDS A SUB-QUANTUM SIGNAL
    same-language pair  : max error  0.0 ms  -- 8 of 8 probes exact, at corr 0.058
Two dubs share only a music-and-effects bed, giving a broad ambiguous peak. Two
encodes of ONE mix give a sharp peak whose magnitude is low but whose LOCATION is
exact. The 0.25 correlation floor is right for the first case and wrong for the
second, so it is not applied here -- CONSISTENCY ACROSS PROBES is the acceptance
rule instead, and a file whose probes disagree is reported as such.

SAFEGUARDS, each paid for deliberately:
  - the ASK is written first, so the denominator survives the run
  - COULD NOT MEASURE is a value, never a silent skip
  - incremental write, so a kill loses nothing
  - units stated: FILES throughout, one file per folder
"""
import json, os, subprocess, sys, collections
from redact import safe_text as _safe
import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, '.')
import scan_hidden as S
from scan_hidden import pcm, SR

MIN_PROMINENCE = 10.0     # CHOSEN, not derived -- see the note below


def probe(path, a, b, t, dur=60.0, maxlag=5.0):
    """(lag_ms, magnitude, prominence).

    ACCEPTANCE IS PROMINENCE, NOT MAGNITUDE, AND THE REASON IS MEASURED.
    Two encodes of one mix give a SHARP peak whose height may be low; two different
    recordings give a BROAD peak. Magnitude cannot tell them apart -- a control
    reading known-exact at height 0.218 sits below the 0.25 floor, while the six
    false STEP rows in the first run were deviant probes at height 0.01-0.03.

    Measured on those six, which are artefacts with a known answer:
        deviant (artefact) probes : prominence 4.9, 5.3, 6.5
        agreeing probes           : prominence 3.7 to 147.1, mostly > 19
    NO CLEAN SEPARATION -- two agreeing probes fall below a deviant one. So this
    rejects some valid readings, which costs COVERAGE and not CORRECTNESS, and the
    threshold below is a constant I chose on a sample of six files.
    """
    M = pcm(path, a, t, dur); C = pcm(path, b, t, dur)
    n = min(len(M), len(C)); ml = int(maxlag * SR)
    if n <= ml + SR: return None, None, None
    M, C = M[:n] - M[:n].mean(), C[:n] - C[:n].mean()
    d = np.linalg.norm(M) * np.linalg.norm(C)
    if d == 0: return None, None, None
    corr = fftconvolve(M, C[::-1], mode="full"); mid = len(C) - 1
    seg = corr[mid - ml:mid + ml + 1] / d
    pk = int(np.argmax(seg)); h = float(seg[pk])
    w = int(0.050 * SR)
    mask = np.ones(len(seg), bool); mask[max(0, pk - w):pk + w] = False
    bg = seg[mask]
    sd = float(np.std(bg)) if bg.size else 0.0
    lag = (np.arange(-ml, ml + 1)[pk] / SR * 1000.0)
    return float(lag), h, (h / sd if sd > 0 else None)

PROBES = (200.0, 450.0, 700.0, 950.0, 1200.0)
STEP_MS = 25.0          # a spread above this is a step, not jitter

def _run():
    fol = json.load(open('runs/folders.json'))['folders']
    dests = [f['destination_path'] for f in fol
             if f.get('destination_path') and os.path.isdir(f['destination_path'])]

    out = open('runs/accepted-scan2.jsonl', 'a')
    out.write(json.dumps({"ask": "one file per accepted destination folder with a "
                                 "same-language audio pair",
                          "folders_offered": len(dests),
                          "probes": list(PROBES), "step_ms": STEP_MS, "min_prominence": MIN_PROMINENCE}) + "\n")
    out.flush()

    n_probed = n_pair = 0
    for d in dests:
        try:
            files = sorted(x for x in os.listdir(d) if x.endswith('.mkv'))
        except Exception as e:
            out.write(json.dumps({"folder": d[-12:], "result": "COULD_NOT_MEASURE",
                                  "why": "listdir: " + _safe(e)}) + "\n"); continue
        if not files:
            out.write(json.dumps({"folder": d[-12:], "result": "COULD_NOT_MEASURE",
                                  "why": "no mkv"}) + "\n"); continue
        p = os.path.join(d, files[0])
        try:
            q = subprocess.run(["ffprobe","-v","error","-select_streams","a",
                "-show_entries","stream_tags=language","-of","csv=p=0",p],
                capture_output=True, timeout=45)
            langs = [x.strip() for x in q.stdout.decode().split() if x.strip()]
        except Exception as e:
            out.write(json.dumps({"folder": d[-12:], "result": "COULD_NOT_MEASURE",
                                  "why": "ffprobe: " + _safe(e)}) + "\n"); continue
        n_probed += 1
        c = collections.Counter(langs)
        dup = [k for k, v in c.items() if v > 1]
        if not dup:
            out.write(json.dumps({"folder": d[-12:], "result": "NO_SAME_LANGUAGE_PAIR",
                                  "n_tracks": len(langs)}) + "\n"); continue
        n_pair += 1
        lang = dup[0]
        idx = [i for i, l in enumerate(langs) if l == lang][:2]
        lags = []
        for t in PROBES:
            try:
                lag, mag, pr = probe(p, idx[0], idx[1], t)
            except Exception:
                lag, mag, pr = None, None, None
            lags.append((t, None if lag is None else round(lag, 2),
                         None if mag is None else round(mag, 4),
                         None if pr is None else round(pr, 1)))
        got = [l for _, l, _, pr in lags
               if l is not None and pr is not None and pr >= MIN_PROMINENCE]
        if len(got) < 3:
            rec = {"folder": d[-12:], "lang": lang, "result": "COULD_NOT_MEASURE",
                   "why": f"only {len(got)} probes cleared prominence {MIN_PROMINENCE}", "lags": lags}
        else:
            spread = max(got) - min(got)
            rec = {"folder": d[-12:], "lang": lang,
                   "result": "STEP" if spread > STEP_MS else "CONSTANT",
                   "spread_ms": round(spread, 2), "median_ms": round(sorted(got)[len(got)//2], 2),
                   "lags": lags}
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(f"[{n_pair}] {rec['result']:<20} spread="
              f"{rec.get('spread_ms','-')}", file=sys.stderr, flush=True)
    out.write(json.dumps({"summary": True, "folders_offered": len(dests),
                          "probed": n_probed, "with_same_language_pair": n_pair}) + "\n")
    out.close()
    print(f"DONE  offered {len(dests)}  probed {n_probed}  pairs {n_pair}", file=sys.stderr)


if __name__ == '__main__':
    _run()
