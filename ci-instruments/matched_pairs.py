#!/usr/bin/env python3
"""folder 110's matched pairs: same master, same episode, two candidates.

dev-1 found this structure in my own corpus index -- 11 episodes, each with TWO
candidate records against ONE master, one declared 25.0 and its partner 23.976. It is
a within-release matched design and it was there all evening.

It replaces my earlier BETWEEN-release discrimination (11 folder-110 carriers against
one folder-54 non-carrier), where everything differing between two releases was
confounded with the transform being tested.

WINDOW NOTE: 25 s carries 1068 ms of residual PAL stretch, so the carrier column reads
~0.45 rather than the ~0.95 a short window gives. Both columns use the same window, so
the COMPARISON is unaffected -- but these are not the headline magnitudes.
"""
import json, subprocess, sys, os, datetime
import numpy as np
SR=8000; PAL=25.0/(24000/1001)
OUT='runs/matched-pairs.json'

def pcm(path,stream,ss,dur,mult=1.0):
    ar=int(round(SR*mult))
    p=subprocess.run(["ffmpeg","-v","error","-ss",f"{ss:.3f}","-t",f"{dur:.3f}","-i",path,
        "-map",f"0:a:{stream}","-ac","1","-ar",str(ar),"-c:a","pcm_s16le","-f","s16le","-"],
        capture_output=True,timeout=600)
    return np.frombuffer(p.stdout,dtype=np.int16).astype(np.float64)

def corr(M,C,ml_s):
    n=min(len(M),len(C))
    if n<4000: return None
    M,C=M[:n]-M[:n].mean(),C[:n]-C[:n].mean()
    ml=int(ml_s*SR)
    if ml>=n//2: ml=n//2-1
    f=np.fft.rfft(M,2*n); g=np.fft.rfft(C,2*n)
    cc=np.fft.irfft(f*np.conj(g),2*n)
    cc=np.concatenate((cc[-ml:],cc[:ml+1]))
    d=(np.linalg.norm(M)*np.linalg.norm(C)) or 1.0
    return float(np.max(cc/d))

recs={r['id']:r for r in json.load(open('runs/audio-durations.json'))}
fr={}
for l in open('runs/framerate-pairs.jsonl'):
    q=json.loads(l); fr[q['id']]=(q.get('master_fps'),q.get('err_fps'))
PAIRS={1:(46,47),2:(51,52),3:(59,60),4:(63,64),5:(70,72),6:(77,78),
       7:(83,84),8:(87,88),9:(91,92),10:(95,96),11:(98,99)}
rows=[]
print(f"  {'ep':>3} {'id':>4} {'declared':>16} {'raw':>9} {'PAL':>9}  reading")
for ep,(a,b) in PAIRS.items():
    for i in (a,b):
        r=recs.get(i)
        if not r: continue
        m,e=fr.get(i,(None,None))
        M=pcm(r['master_path'],0,30.0,25.0)
        raw=corr(M,pcm(r['file_path'],0,30.0,25.0),8.0)
        pal=corr(M,pcm(r['file_path'],0,30.0,25.0/PAL,mult=PAL),8.0)
        raw=raw or 0.0; pal=pal or 0.0
        which="PAL carrier" if pal>raw else "equal-rate partner"
        rows.append({"episode":ep,"id":i,"master_fps":m,"err_fps":e,
                     "raw_r1":round(raw,4),"pal_1_042708":round(pal,4),
                     "reading":which,"window_s":25.0,"probe_start_s":30.0,
                     "master_a":0,"cand_a":0})
        print(f"  {ep:>3} {i:>4} {str(m)+' vs '+str(e):>16} {raw:>9.4f} {pal:>9.4f}  {which}")
json.dump({"produced_utc":datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
           "design":"within-release matched pairs, same master per episode, found by dev-1",
           "LIMIT_one_release":"folder 110 only. Removes the between-release confound; says "
                               "NOTHING about how common the relation is in the library.",
           "LIMIT_window":"25 s carries 1068 ms of residual PAL stretch; short windows give "
                          "0.93-0.99 on the same carriers. Comparison unaffected.",
           "rows":rows}, open(OUT,'w'), indent=1)
print(f"\n  WROTE {OUT}: {len(rows)} rows")
