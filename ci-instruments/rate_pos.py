#!/usr/bin/env python3
"""Is the curve a property of WINDOW LENGTH or of WHERE THE SWEEP STARTED?

dev-3's confound, and it undercuts everything downstream: my sweep used ONE start and
varied only length, so different lengths cover DIFFERENT CONTENT. A 0.5 s window landing
on a quiet passage dips; longer windows average over it. Length and content are
confounded BY CONSTRUCTION, and every band, ceiling and intersection I have reported
inherits it.

  dip MOVES or vanishes across starts -> content-local; the curve is a property of
                                         WHERE I started, and "the shape" is not a shape
  dip STAYS at the same window length -> a genuine window-length effect

The coarse lag is re-derived AT EACH START, because a file with change points has a
different offset at t=600 than at t=30 and reusing one lag would manufacture a dip.
"""
import subprocess, sys, json
import numpy as np
SR = 8000

def pcm(path, stream, ss, dur, rate_mult=1.0):
    ar = int(round(SR*rate_mult))
    p = subprocess.run(["ffmpeg","-v","error","-ss",f"{ss:.3f}","-t",f"{dur:.3f}",
                        "-i",path,"-map",f"0:a:{stream}","-ac","1","-ar",str(ar),
                        "-c:a","pcm_s16le","-f","s16le","-"],capture_output=True,timeout=300)
    return np.frombuffer(p.stdout,dtype=np.int16).astype(np.float64)

def corr(M,C,maxlag_s,min_samples=1000):
    n=min(len(M),len(C))
    if n<min_samples: return None,None
    M,C=M[:n]-M[:n].mean(),C[:n]-C[:n].mean()
    ml=int(maxlag_s*SR)
    if ml>=n//2: ml=max(1,n//4)
    f=np.fft.rfft(M,2*n); g=np.fft.rfft(C,2*n)
    cc=np.fft.irfft(f*np.conj(g),2*n)
    cc=np.concatenate((cc[-ml:],cc[:ml+1]))
    d=(np.linalg.norm(M)*np.linalg.norm(C)) or 1.0
    cc/=d; j=int(np.argmax(cc))
    return np.arange(-ml,ml+1)[j]/SR*1000.0, float(cc[j])

recs={r['id']:r for r in json.load(open('runs/audio-durations.json'))}
MULT=25.0/(24000/1001)
i=int(sys.argv[1]); starts=[float(x) for x in sys.argv[2:]]
r=recs[i]
print(f"  id {i}   compensation x{MULT:.8f}")
# THE OFFSET GROWS WITH POSITION. At 42.708 ms/s, a start 570 s later than the anchor
# sits ~24 s further out -- so a fixed +-8 s coarse search finds NOTHING at t=600 and
# reports "does not carry the transform". That is the FOURTH time in this thread I have
# produced a false negative from a search window too narrow to contain the answer.
# So: PREDICT the offset from the anchor and search narrowly around the prediction.
anchor_T, anchor_base = None, None
for T in starts:
    if anchor_base is None:
        M0=pcm(r['master_path'],0,T,25.0)
        C0=pcm(r['file_path'],0,T,25.0/MULT,rate_mult=MULT)
        base,bc=corr(M0,C0,8.0,min_samples=8000)
        if base is not None and bc is not None and bc>=0.15:
            anchor_T, anchor_base = T, base
    else:
        pred = anchor_base + (T-anchor_T)*1000.0*(MULT-1.0)
        # read the candidate already shifted by the PREDICTION, then search +-2 s
        cs0 = T - (pred/1000.0)/MULT
        M0=pcm(r['master_path'],0,T,25.0)
        C0=pcm(r['file_path'],0,max(cs0,0.0),25.0/MULT,rate_mult=MULT)
        d,bc=corr(M0,C0,2.0,min_samples=8000)
        base = (pred + d) if d is not None else None
        print(f"    start t={T:>6.0f}s  predicted {pred:+.0f} ms from the anchor, "
              f"residual {d if d is not None else float('nan'):+.1f} ms")
    if base is None or bc is None or bc<0.15:
        print(f"    start t={T:>6.0f}s   NO COARSE LAG (best {bc}) -- skipped"); continue
    row=[]
    for W in (0.25,0.5,1.0,2.0,3.0,5.0):
        M=pcm(r['master_path'],0,T,W)
        cs=T-(base/1000.0)/MULT
        C=pcm(r['file_path'],0,max(cs,0.0),W/MULT,rate_mult=MULT)
        lag,c=corr(M,C,0.05)
        row.append((W,c,lag))
    cs_=" ".join(f"{c:.4f}" if c is not None else "  n/a " for _,c,_ in row)
    lg=" ".join(f"{l:+.2f}" if l is not None else "  -  " for _,_,l in row)
    dips=[W for k,(W,c,_) in enumerate(row) if c is not None and 0<k<len(row)-1
          and row[k-1][1] is not None and row[k+1][1] is not None
          and c < row[k-1][1] and c < row[k+1][1]]
    print(f"    start t={T:>6.0f}s  base {base:+8.1f} ms ({bc:.3f})  corr: {cs_}")
    print(f"                      lags: {lg}")
    print(f"                      local dips at W = {dips if dips else 'NONE'}")
print("\n  windows: 0.25  0.50  1.00  2.00  3.00  5.00 s")
