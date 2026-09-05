#!/usr/bin/env python3
"""Is there a W_min below which the correlator itself fails, independent of stretch?

dev-3: if W_min >= ceiling/(r-1) the usable band is EMPTY and the file is structurally
unmeasurable by this instrument -- not badly measured, unmeasurable. Predicted for
id 70, which peaks at W=5 s and is WORSE at W=2 s.

THE CONTROL dev-3 DID NOT ASK FOR: run the same short windows on id 83, which IS
monotone. If id 83 also falls at 0.5 s, W_min is a property of the correlator and
applies to everything; if only id 70 falls, it is a property of that file. "id 70 falls"
alone cannot distinguish those.

The coarse lag is removed first, so maxlag can be tiny -- 50 ms -- which is what makes
sub-second windows possible at all. My earlier sweep could not go below 2 s because it
carried a 0.5 s maxlag that does not fit inside a 1 s window.
"""
import subprocess, sys, json
import numpy as np
SR = 8000

def pcm(path, stream, ss, dur, rate_mult=1.0):
    ar = int(round(SR * rate_mult))
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{ss:.3f}", "-t", f"{dur:.3f}",
                        "-i", path, "-map", f"0:a:{stream}", "-ac", "1", "-ar", str(ar),
                        "-c:a", "pcm_s16le", "-f", "s16le", "-"], capture_output=True, timeout=300)
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float64)

def corr(M, C, maxlag_s, min_samples=1000):
    n = min(len(M), len(C))
    if n < min_samples: return None, None, n
    M, C = M[:n] - M[:n].mean(), C[:n] - C[:n].mean()
    ml = int(maxlag_s * SR)
    if ml >= n // 2: ml = max(1, n // 4)
    f = np.fft.rfft(M, 2*n); g = np.fft.rfft(C, 2*n)
    cc = np.fft.irfft(f*np.conj(g), 2*n)
    cc = np.concatenate((cc[-ml:], cc[:ml+1]))
    d = (np.linalg.norm(M)*np.linalg.norm(C)) or 1.0
    cc /= d
    j = int(np.argmax(cc))
    return np.arange(-ml, ml+1)[j]/SR*1000.0, float(cc[j]), n

recs = {r['id']: r for r in json.load(open('runs/audio-durations.json'))}
MULT = 25.0/(24000/1001)
def coarse(r, MULT):
    """Find the lag on a window big enough to contain it, BEFORE sweeping.
    A 0.25 s window cannot search +-3 s; that error cost me two false negatives."""
    M = pcm(r['master_path'], 0, 30.0, 25.0)
    C = pcm(r['file_path'], 0, 30.0, 25.0/MULT, rate_mult=MULT)
    return corr(M, C, 8.0, min_samples=8000)
print(f"  {'id':>5}{'window_s':>10}{'samples':>9}{'stretch_ms':>12}{'corr':>9}{'lag_ms':>9}")
for i in [int(x) for x in sys.argv[1:]]:
    r = recs[i]
    base, bc, _ = coarse(r, MULT)
    if base is None or bc is None or bc < 0.15:
        print(f'  {i:>5}   NO COARSE LAG (best {bc if bc is not None else float("nan"):.4f}) -- does not carry the transform; excluded')
        print(); continue
    rows=[]
    for W in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0):
        M = pcm(r['master_path'], 0, 30.0, W)
        cs = 30.0 - (base/1000.0)/MULT
        C = pcm(r['file_path'], 0, max(cs, 0.0), W/MULT, rate_mult=MULT)
        lag, c, n = corr(M, C, 0.05)
        cs_ = f"{c:.4f}" if c is not None else "TOO SHORT"
        lg = f"{lag:.1f}" if lag is not None else "-"
        rows.append((W, c, lag))
        print(f"  {i:>5}{W:>10.2f}{n:>9}{W*1000*(MULT-1):>12.0f}{cs_:>9}{lg:>9}")
    # A LAG THAT DISAGREES WITH ITS NEIGHBOURS IDENTIFIES A SPURIOUS PEAK regardless
    # of the correlation value. dev-3's rule, and it costs nothing because both columns
    # are already recorded. id 70's W=2 s reading was caught this way: -11.2 ms against
    # neighbours at +0.2.
    good=[l for _,c,l in rows if l is not None]
    med=sorted(good)[len(good)//2] if good else 0.0
    bad=[(W,c,l) for W,c,l in rows if l is not None and abs(l-med)>2.0]
    for W,c,l in bad:
        print(f"        SPURIOUS PEAK at W={W}: lag {l:+.1f} ms against median {med:+.1f} -- corr {c:.4f} DISCARDED")
    clean=[(W,c) for W,c,l in rows if c is not None and (l is None or abs(l-med)<=2.0)]
    # THE CURVE IS UNIMODAL, NOT MONOTONE, so "the window where it crosses the bar" is
    # not well defined. The first version took max(above) and min(below) and printed
    # "bar crossed between W=5.00s and W=0.25s" on id 46 -- endpoints REVERSED, a
    # confident interval bracketing nothing. It encoded the monotone model it was
    # supposed to be testing, in the same edit where I added the lag control that
    # disproved it.
    # Report the PEAK and the band that clears the bar. Both are defined either way.
    if clean:
        pw, pc = max(clean, key=lambda x: x[1])
        # DO NOT ASSUME THE CLEARING SET IS CONTIGUOUS. "from min(ok) to max(ok)" is
        # the SAME defect as the monotone "bar crossed between" line it replaced, one
        # iteration later and harder to see because the output looks sane: a file that
        # clears at 0.25 and 3.0 but fails at 1.0 would be reported as clearing
        # throughout. dev-3 warned about exactly this while I was writing it.
        # The shape is NOT established -- unimodal holds for 1 of 4 files -- so the
        # set is printed, and a gap is named.
        ok = [W for W, c in clean if c >= 0.90]
        allw = [W for W, _ in clean]
        contiguous = ok and all(W in ok for W in allw
                                if min(ok) <= W <= max(ok))
        if not ok:
            band = "NEVER clears 0.90 at any window tested"
        elif contiguous:
            band = f"clears 0.90 from W={min(ok):.2f}s to W={max(ok):.2f}s"
        else:
            gaps = [W for W in allw if min(ok) <= W <= max(ok) and W not in ok]
            band = (f"clears 0.90 at W in {[round(w,2) for w in ok]} "
                    f"-- NOT CONTIGUOUS, fails inside the range at {[round(w,2) for w in gaps]}")
        print(f"        peak {pc:.4f} at W={pw:.2f}s "
              f"({pw*1000*(MULT-1):.0f} ms stretch);  {band}")
        if ok and len(ok) < len(clean):
            print(f"        -> BAND, not a ceiling: {len(clean)-len(ok)} of {len(clean)} "
                  f"clean windows fail, on BOTH sides of the peak where applicable")
    print()


