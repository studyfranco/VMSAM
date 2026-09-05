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
BASE = {70: 3171.9, 83: 3175.0, 46: 3133.9, 91: 3218.2, 98: 3171.9, 51: 3176.0}
print(f"  {'id':>5}{'window_s':>10}{'samples':>9}{'stretch_ms':>12}{'corr':>9}{'lag_ms':>9}")
for i in [int(x) for x in sys.argv[1:]]:
    r = recs[i]; base = BASE[i]
    for W in (0.25, 0.5, 1.0, 2.0, 5.0):
        M = pcm(r['master_path'], 0, 30.0, W)
        cs = 30.0 - (base/1000.0)/MULT
        C = pcm(r['file_path'], 0, max(cs, 0.0), W/MULT, rate_mult=MULT)
        lag, c, n = corr(M, C, 0.05)
        cs_ = f"{c:.4f}" if c is not None else "TOO SHORT"
        lg = f"{lag:.1f}" if lag is not None else "-"
        print(f"  {i:>5}{W:>10.2f}{n:>9}{W*1000*(MULT-1):>12.0f}{cs_:>9}{lg:>9}")
    print()
