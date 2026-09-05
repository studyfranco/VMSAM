#!/usr/bin/env python3
"""Does a declared-25.0 pair correlate ONCE THE RATE IS COMPENSATED?

dev-3's hypothesis: folder 110's REFUSED_NO_PLAN is CORRECT because the content really
carries a 25 against 24000/1001 relation, which blows the <127 ms window after 2.97 s.

MY EARLIER TEST OF IT WAS INVALID -- I measured eleven EQUAL-RATE files and reported
them as evidence about 25.0-declared files. Retracted.

The real obstacle: all 13 files declared 23.976 vs 25.0 come back `baseline-not-found`
with correlation 0.004-0.013. My correlator searches ONE lag over a 20 s window, and a
4.27 % speed difference smears that window by ~854 ms -- so no-peak is exactly what a
REAL rate relation would produce. The instrument cannot tell "rate-related" from
"unrelated"; it returns no baseline for both.

THE DISCRIMINATOR: resample the candidate by 25/23.976 and correlate again.
  peak appears  -> the content relation is REAL, the refusals are correct
  still nothing -> the audio is unrelated for some other reason

CONTROL IN BOTH DIRECTIONS, because a resample that always helps proves nothing:
  - the SAME compensation applied to an EQUAL-RATE pair must make it WORSE
  - the uncompensated reading is taken on the same window, same tracks
"""
import subprocess, sys, json
import numpy as np
SR = 8000

def pcm(path, stream, ss, dur, rate_mult=1.0):
    """rate_mult != 1 resamples: we ASK for SR*mult and label it SR, which stretches
    or compresses the timeline by exactly mult."""
    ar = int(round(SR * rate_mult))
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{ss:.3f}", "-t", f"{dur:.3f}",
         "-i", path, "-map", f"0:a:{stream}", "-ac", "1", "-ar", str(ar),
         "-c:a", "pcm_s16le", "-f", "s16le", "-"],
        capture_output=True, timeout=300)
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float64)

def corr_at(M, C, maxlag_s):
    n = min(len(M), len(C))
    if n < SR * 2:
        return None, None
    M, C = M[:n], C[:n]
    M = M - M.mean(); C = C - C.mean()
    ml = int(maxlag_s * SR)
    if n <= ml + SR:
        return None, None
    f = np.fft.rfft(M, 2 * n); g = np.fft.rfft(C, 2 * n)
    cc = np.fft.irfft(f * np.conj(g), 2 * n)
    cc = np.concatenate((cc[-ml:], cc[:ml + 1]))
    d = (np.linalg.norm(M) * np.linalg.norm(C)) or 1.0
    cc = cc / d
    j = int(np.argmax(cc))
    return (np.arange(-ml, ml + 1)[j] / SR * 1000.0), float(cc[j])

def main():
    ids = [int(x) for x in sys.argv[1:]]
    recs = {r['id']: r for r in json.load(open('runs/audio-durations.json'))}
    fr = {}
    for l in open('runs/framerate-pairs.jsonl'):
        r = json.loads(l); fr[r['id']] = (r.get('master_fps'), r.get('err_fps'))
    for i in ids:
        r = recs.get(i)
        if not r:
            continue
        m, e = fr.get(i, (None, None))
        mult = (e / m) if (m and e) else 1.0
        M = pcm(r['master_path'], 0, 400.0, 25.0)
        best = None
        for name, mu in (("uncompensated", 1.0), (f"compensated x{mult:.5f}", mult),
                         ("inverse", 1.0 / mult)):
            C = pcm(r['file_path'], 0, 400.0, 25.0 * (1.0 / mu if mu != 1.0 else 1.0), rate_mult=mu)
            lag, c = corr_at(M, C, 4.0)
            print(f"  id {i:>4} declared {m} vs {e}   {name:<22} "
                  f"corr={('%.4f' % c) if c is not None else 'n/a':>8}  lag={('%.1f' % lag) if lag is not None else '-':>9} ms")
        print()

main()
