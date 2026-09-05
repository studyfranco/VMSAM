#!/usr/bin/env python3
"""Does a file CARRY its declared rate relation? Generic in the ratio.

dev-2 asked this of five files it calls "the PAL population". They are not PAL: all
five are declared 23.976 vs 24.0, a ratio of 1.001001 and a drift of 1.001 ms/s
against PAL's 42.708 -- a factor of 43. The transform under test is therefore a
different one, and a probe hardcoded to the PAL ratio would have answered a question
nobody asked.

Method, per dev-3's description of it: compensate one side by the DECLARED transform,
remove the coarse lag, correlate at several short windows, and check each lag against
its neighbours. A carrier came back >=0.93 with a residual under ~1 ms; the
non-carrier id 57 came back 0.02-0.05 at every window.

REPORTED AS THREE OUTCOMES: CARRIES, DOES NOT CARRY, or COULD NOT MEASURE -- the last
being an instrument statement, not a file one.
"""
import subprocess, sys, json
import numpy as np
SR = 8000

def pcm(path, stream, ss, dur, rate_mult=1.0):
    ar = int(round(SR*rate_mult))
    p = subprocess.run(["ffmpeg","-v","error","-ss",f"{ss:.3f}","-t",f"{dur:.3f}",
                        "-i",path,"-map",f"0:a:{stream}","-ac","1","-ar",str(ar),
                        "-c:a","pcm_s16le","-f","s16le","-"],capture_output=True,timeout=600)
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


import os, datetime
TRACE = 'runs/probe-trace.jsonl'

def trace(**kw):
    """A KILLED PROBE MUST LEAVE A TRACE. dev-2 lost a 44-minute TrueHD run to a kill
    and could not tell it from a file never attempted -- absent-versus-zero in the
    harness rather than in a field. A `started` row is written and flushed BEFORE the
    work, so a row with no matching `finished` is a timeout, not a silence.
    """
    kw['utc'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    with open(TRACE, 'a') as fh:
        fh.write(json.dumps(kw) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

def astreams(p):
    q = subprocess.run(["ffprobe","-v","error","-select_streams","a","-show_entries",
                        "stream=index:stream_tags=language","-of","csv=p=0",p],
                       capture_output=True, timeout=45)
    out=[]
    for line in q.stdout.decode().split():
        parts=line.split(',')
        if parts and parts[0].isdigit():
            out.append(parts[1] if len(parts)>1 else None)
    return out

def shared_pairs(M, C):
    """(master_a_index, cand_a_index, lang) for every SHARED language.

    THIS PROBE HARDCODED a:0 ON BOTH SIDES AND I DOCUMENTED THAT EXACT DEFECT IN
    run_full.py MONTHS-EQUIVALENT AGO: "the first version hardcoded stream 0 on both
    sides. Measured: on 25 randomly sampled pairs, stream 0 carries a DIFFERENT language
    on the two sides for 18 of them -- 72 % OF THE CORPUS WOULD BE CORRELATED ACROSS
    LANGUAGES."

    I fixed it there and reintroduced it here. Every "DOES NOT CARRY" verdict this probe
    emitted was a cross-language comparison: id 57 was cze against eng, id 101 cze
    against eng, ids 288-290 eng against tha/jpn. Czech does not correlate with English
    and that is not a finding about a transform.
    """
    ml, cl = astreams(M), astreams(C)
    out=[]
    for lg in dict.fromkeys(x for x in cl if x):
        if lg in ml:
            out.append((ml.index(lg), cl.index(lg), lg))
    return out

recs={r['id']:r for r in json.load(open('runs/audio-durations.json'))}
fr={}
for l in open('runs/framerate-pairs.jsonl'):
    r=json.loads(l); fr[r['id']]=(r.get('master_fps'),r.get('err_fps'))
for i in [int(x) for x in sys.argv[1:]]:
    r=recs[i]; m,e=fr[i]; MULT=e/m
    trace(event="started", id=i, probe="carry_test", ratio=round(MULT,6))
    print(f"  id {i:>4}  declared {m} vs {e}   ratio {MULT:.6f}   drift {(MULT-1)*1000:.3f} ms/s")
    pairs = shared_pairs(r['master_path'], r['file_path'])
    if not pairs:
        print("        NO SHARED LANGUAGE -- not measurable, and not a statement about the transform\n")
        trace(event="finished", id=i, probe="carry_test", ratio=round(MULT,6),
              verdict="NO SHARED LANGUAGE"); continue
    print(f"        shared languages: {[(a,b,l) for a,b,l in pairs]}")
    mi, ci, lang = pairs[0]
    M0=pcm(r['master_path'],mi,30.0,25.0)
    C0=pcm(r['file_path'],ci,30.0,25.0/MULT,rate_mult=MULT)
    base,bc=corr(M0,C0,8.0,min_samples=8000)
    if base is None or bc is None:
        print("        COULD NOT MEASURE -- probe returned nothing\n"); continue
    print(f"        coarse: lag {base:+.1f} ms at corr {bc:.4f}")
    rows=[]
    for W in (0.5,1.0,2.0,5.0):
        M=pcm(r['master_path'],mi,30.0,W)
        cs=30.0-(base/1000.0)/MULT
        C=pcm(r['file_path'],ci,max(cs,0.0),W/MULT,rate_mult=MULT)
        lag,c=corr(M,C,0.05)
        rows.append((W,c,lag))
        print(f"        W={W:>4.2f}s  corr {c if c is not None else float('nan'):.4f}  lag {lag if lag is not None else float('nan'):+.2f} ms")
    best=max((c for _,c,_ in rows if c is not None), default=0.0)
    good=[l for _,_,l in rows if l is not None]
    med=sorted(good)[len(good)//2] if good else None
    spread=(max(good)-min(good)) if good else None
    # THE COMPENSATION RATIO IS A PARAMETER OF THE MEASUREMENT AND BELONGS ON THE ROW.
    # A fidelity of 0.03 means nothing without knowing which transform produced it, and
    # neither does 0.99. Running this probe with the PAL ratio against a true 1.001001
    # returns near-zero on a file that DOES carry its relation -- correctly-working code
    # answering a different question, and agreeing with itself while it does.
    # So no verdict is emitted without the ratio inside it.
    if best>=0.93 and spread is not None and spread<=2.0:
        v=f"CARRIES x{MULT:.6f} (best {best:.4f}, lag spread {spread:.2f} ms)"
    elif best<0.20:
        v=f"DOES NOT CARRY x{MULT:.6f} (best {best:.4f}) -- an id-57 FOR THIS TRANSFORM ONLY"
    else:
        v=f"AMBIGUOUS under x{MULT:.6f} (best {best:.4f})"
    print(f"        -> {v}\n")
    trace(event="finished", id=i, probe="carry_test", ratio=round(MULT,6),
          verdict=v.split(" (")[0], best=round(best,4),
          # what a NEGATIVE excludes, per dev-3: residual stretch from a wrong
          # transform is (r_applied - r_true) x W, so a near-zero at W=0.25 s
          # only excludes transforms within ~0.24 of the one applied.
          negative_excludes_ratios_within=0.24 if best<0.20 else None)
