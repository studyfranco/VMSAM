#!/usr/bin/env python3
"""Is the 0.49 ceiling my WINDOWING or the FILES?

dev-3: over 25 s a 4.271 % relation stretches content by 1068 ms, so a residual
mis-span caps the correlation. §4f's acceptance bar is 90 % audio similarity and my
compensated readings are 0.475-0.495 -- a factor of two below it. One of the bar and
the measurement is wrong, and stage 1 cannot be built until someone says which.

DISCRIMINATOR: vary the window length at fixed compensation.
  correlation RISES as the window shrinks  -> residual mis-span; the bar is reachable
  correlation FLAT across window lengths   -> a content limit; the bar is not
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

def corr(M, C, maxlag_s):
    n = min(len(M), len(C))
    if n < SR: return None, None
    M, C = M[:n] - M[:n].mean(), C[:n] - C[:n].mean()
    ml = min(int(maxlag_s * SR), n - SR)
    if ml < 1: return None, None
    f = np.fft.rfft(M, 2*n); g = np.fft.rfft(C, 2*n)
    cc = np.fft.irfft(f*np.conj(g), 2*n)
    cc = np.concatenate((cc[-ml:], cc[:ml+1]))
    d = (np.linalg.norm(M)*np.linalg.norm(C)) or 1.0
    cc /= d
    j = int(np.argmax(cc))
    return np.arange(-ml, ml+1)[j]/SR*1000.0, float(cc[j])

recs = {r['id']: r for r in json.load(open('runs/audio-durations.json'))}
fr = {}
for l in open('runs/framerate-pairs.jsonl'):
    r = json.loads(l); fr[r['id']] = (r.get('master_fps'), r.get('err_fps'))
MULT = 25.0/(24000/1001)
print(f"  compensation x{MULT:.10f}   (SPEC_ZONE_A §4f names 1.04270833333)")
print(f"\n  {'id':>5}{'window_s':>10}{'stretch_ms':>12}{'corr':>9}{'lag_ms':>10}")
for i in [int(x) for x in sys.argv[1:]]:
    r = recs[i]
    for W in (2.0, 5.0, 10.0, 25.0):
        M = pcm(r['master_path'], 0, 30.0, W)
        C = pcm(r['file_path'], 0, 30.0, W/MULT, rate_mult=MULT)
        lag, c = corr(M, C, min(8.0, W/3))
        print(f"  {i:>5}{W:>10.1f}{W*1000*(MULT-1):>12.0f}"
              f"{('%.4f'%c) if c is not None else 'n/a':>9}{('%.1f'%lag) if lag is not None else '-':>10}")
    print()
