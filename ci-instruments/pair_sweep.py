#!/usr/bin/env python3
"""How many corpus files are NOT A REPAIR PROBLEM because the pair is wrong?

dev-3's denominator correction: every "the repair fixes X of N" statistic has an N
inflated by files no repair can fix. They need a different master, not a better
algorithm, and counting them as failures makes the repair look worse than it is.

METHOD -- a test with NO ALIGNMENT ASSUMPTION, which is why it settles what every
aligned test left open: scan one master window against the WHOLE candidate.
    a peak somewhere  -> the content is present; whatever is wrong is alignment or rate
    nothing anywhere  -> different work, or a different dub. No transform repairs it.

CONTROL REQUIREMENT, learned three times tonight: A CONTROL MUST POSSESS THE PROPERTY
THE QUESTION TESTS. id 52 works here because it correlates RAW; id 83 does not, because
it is PAL and cannot correlate uncompensated at all. Naming the required property
before choosing the file is the step that catches this.

SHARED LANGUAGE ONLY. Comparing a:0 to a:0 compares Czech to English on this corpus --
my own documented defect, reintroduced in a new instrument tonight.
"""
import json, subprocess, sys, os, datetime
import numpy as np
SR = 8000
TRACE = 'runs/pair-sweep-trace.jsonl'

def trace(**kw):
    kw['utc'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    with open(TRACE, 'a') as fh:
        fh.write(json.dumps(kw) + "\n"); fh.flush(); os.fsync(fh.fileno())

def astreams(p):
    q = subprocess.run(["ffprobe","-v","error","-select_streams","a","-show_entries",
                        "stream=index:stream_tags=language","-of","csv=p=0",p],
                       capture_output=True, timeout=60)
    out=[]
    for line in q.stdout.decode().split():
        pr=line.split(',')
        if pr and pr[0].isdigit(): out.append(pr[1] if len(pr)>1 else None)
    return out

def pcm(path, stream, ss, dur, mult=1.0):
    ar=int(round(SR*mult))
    p = subprocess.run(["ffmpeg","-v","error","-ss",f"{ss:.3f}","-t",f"{dur:.3f}","-i",path,
                        "-map",f"0:a:{stream}","-ac","1","-ar",str(ar),
                        "-c:a","pcm_s16le","-f","s16le","-"], capture_output=True, timeout=600)
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float64)

def corr_lag(M, C, ml_s):
    """(lag_ms, corr) -- the lag is needed before a short window can be placed."""
    n=min(len(M),len(C))
    if n<2000: return None,None
    M,C=M[:n]-M[:n].mean(),C[:n]-C[:n].mean()
    ml=int(ml_s*SR)
    if ml>=n//2: ml=max(1,n//4)
    f=np.fft.rfft(M,2*n); g=np.fft.rfft(C,2*n)
    cc=np.fft.irfft(f*np.conj(g),2*n)
    cc=np.concatenate((cc[-ml:],cc[:ml+1]))
    d=(np.linalg.norm(M)*np.linalg.norm(C)) or 1.0
    cc/=d; j=int(np.argmax(cc))
    return np.arange(-ml,ml+1)[j]/SR*1000.0, float(cc[j])

def peak(M, C):
    n=min(len(M),len(C))
    if n<4000: return 0.0
    M,C=M[:n]-M[:n].mean(),C[:n]-C[:n].mean()
    f=np.fft.rfft(M,2*n); g=np.fft.rfft(C,2*n)
    cc=np.fft.irfft(f*np.conj(g),2*n)
    d=(np.linalg.norm(M)*np.linalg.norm(C)) or 1.0
    return float(np.max(np.abs(cc)))/d

def main():
    recs={r['id']:r for r in json.load(open('runs/audio-durations.json'))}
    global fr
    fr={}
    for l in open('runs/framerate-pairs.jsonl'):
        q=json.loads(l); fr[q['id']]=(q.get('master_fps'), q.get('err_fps'))
    ids=[int(x) for x in sys.argv[1:]]
    for i in ids:
        r=recs.get(i)
        if not r: continue
        trace(event="started", id=i, probe="pair_sweep")
        try:
            ml,cl=astreams(r['master_path']),astreams(r['file_path'])
        except Exception as e:
            print(f"  id {i:>4}  PROBE FAILED {str(e)[:40]}"); 
            trace(event="finished", id=i, verdict="PROBE_FAILED"); continue
        sh=[(ml.index(l),cl.index(l),l) for l in dict.fromkeys(x for x in cl if x) if l in ml]
        if not sh:
            print(f"  id {i:>4}  NO SHARED LANGUAGE -- not measurable by this method")
            trace(event="finished", id=i, verdict="NO_SHARED_LANGUAGE"); continue
        mi,ci,lang=sh[0]
        mdur=r.get('mst_dur') or 1400.0
        T=min(300.0, max(60.0, mdur*0.25))
        M=pcm(r['master_path'],mi,T,20.0)

        # A RAW SCAN CANNOT FIND SPEED-SHIFTED CONTENT. The first version scanned
        # uncompensated only, and of its first 8 verdicts SIX WERE KNOWN CARRIERS --
        # four called WRONG PAIR at 0.02-0.06, files I had measured carrying PAL at
        # 0.93-0.99. dev-3 predicted exactly this: every failure mode of this
        # instrument points at WRONG PAIR, so a broken sweep manufactures the finding
        # it was built to test, and it agrees with the result already in hand.
        #
        # So: TRY THE TRANSFORMS FIRST. Only a file that no transform and no offset
        # can explain is a wrong pair.
        PAL = 25.0/(24000/1001)
        NTSC = 24.0/(24000/1001)
        m_fps, e_fps = fr.get(i, (None, None))
        ratios = [1.0, PAL, 1.0/PAL, NTSC, 1.0/NTSC]
        if m_fps and e_fps:
            ratios.append(e_fps/m_fps)
        # TWO STAGES, because a 25 s window at PAL carries 1068 ms of residual stretch
        # and reads ~0.45 even on a file that scores 0.97 properly. The first version
        # scored id 46 at 0.345 -- just under threshold -- and called a known carrier a
        # WRONG PAIR. Coarse-correlate to FIND the lag, remove it, then score on a
        # SHORT window where the stretch is small. This is carry_test's method and I
        # should have reused it rather than writing a second, weaker one.
        # PROBE THE TRANSFORM NEAR THE START. The offset ACCUMULATES: at PAL the lag at
        # t=300 is ~15 s, outside any maxlag I would set, so a coarse search there finds
        # nothing and reports 0.000 on a known carrier. That is the SIXTH time tonight a
        # search window too narrow for the elapsed distance produced a confident
        # negative. T0 is small so the accumulated offset is small; the OFFSET SCAN
        # below still uses the later T, where it belongs.
        T0 = 30.0
        best_r = (None, 0.0)
        for mu in ratios:
            Mx = pcm(r['master_path'], mi, T0, 25.0)
            Cx = pcm(r['file_path'], ci, T0, 25.0/mu, mult=mu)
            lag, c0 = corr_lag(Mx, Cx, 8.0)
            if lag is None or c0 is None or c0 < 0.10:
                continue
            cs2 = T0 - (lag/1000.0)/mu
            M2 = pcm(r['master_path'], mi, T0, 1.0)
            C2 = pcm(r['file_path'], ci, max(cs2, 0.0), 1.0/mu, mult=mu)
            _, p = corr_lag(M2, C2, 0.05)
            p = p or 0.0
            if p > best_r[1]: best_r = (mu, p)
            if p > 0.5: break
        if best_r[1] >= 0.35:
            v = "CONTENT PRESENT"
            print(f"  id {i:>4} lang={lang:<4} transform x{best_r[0]:.6f} gives {best_r[1]:.4f}   -> {v}")
            trace(event="finished", id=i, probe="pair_sweep", lang=lang,
                  best=round(best_r[1],4), via="transform", ratio=round(best_r[0],6), verdict=v)
            continue

        # No transform at this position. Now the offset scan, which catches a pure
        # displacement with no rate change.
        best=(None,0.0)
        WIN=20.0
        step=int(WIN/2)
        for cs in range(0,int((r.get('err_dur') or mdur))-int(WIN),step):
            p=peak(M,pcm(r['file_path'],ci,float(cs),WIN))
            if p>best[1]: best=(cs,p)
            if p>0.5: break
        v = "CONTENT PRESENT" if best[1]>=0.35 else ("WRONG PAIR" if best[1]<0.15 else "AMBIGUOUS")
        print(f"  id {i:>4} lang={lang:<4} no transform (best {best_r[1]:.3f}); "
              f"scan best {best[1]:.4f} at t={best[0]}s   -> {v}")
        trace(event="finished", id=i, probe="pair_sweep", lang=lang,
              best=round(best[1],4), at_s=best[0], verdict=v)

main()
