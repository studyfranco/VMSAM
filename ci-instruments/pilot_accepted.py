#!/usr/bin/env python3
"""PILOT: does the population the pipeline ACCEPTED carry desync it never recorded?

vmsam-forensic established that the error corpus is defined, by construction, as
files whose record step span is at least TWO quanta -- minimum observed 250 ms,
zero files below. The reconciliation window is one quantum. So a file carrying only
a sub-quantum step RECONCILED, WAS ACCEPTED, AND SHIPPED, and cannot appear in the
corpus. Every blind-spot figure the campaign has published is conditioned on a file
having failed loudly enough to be recorded.

The candidate is consumed by a successful merge, so no pair survives. But a merged
file carries several audio tracks, and A STEP BETWEEN TWO TRACKS OF ONE SHIPPED FILE
IS A DESYNC THAT SHIPPED -- measurable with no candidate at all.

READ-ONLY. Pilot scale: a few files, to find out whether the measurement works before
anyone commits to a sweep.
"""
import json, os, random, subprocess, sys
sys.path.insert(0, '.')
import scan_hidden as S

fol = json.load(open('runs/folders.json'))['folders']
dests = [f['destination_path'] for f in fol if f.get('destination_path')
         and os.path.isdir(f['destination_path'])]
random.seed(20260903)
random.shuffle(dests)

def alangs(p):
    q = subprocess.run(["ffprobe","-v","error","-select_streams","a","-show_entries",
        "stream=index:stream_tags=language","-of","csv=p=0",p], capture_output=True, timeout=60)
    return [l.split(',') for l in q.stdout.decode().strip().split("\n") if l]

done = 0
for d in dests:
    if done >= 6: break
    try: files = sorted(x for x in os.listdir(d) if x.endswith('.mkv'))
    except Exception: continue
    if not files: continue
    p = os.path.join(d, files[0])
    st = alangs(p)
    if len(st) < 2: continue
    done += 1
    # correlate track 0 against track 1 at several positions. A CONSTANT offset is
    # fine -- it may be intentional. A STEP between positions is a shipped desync.
    lags = []
    for frac in (0.15, 0.35, 0.55, 0.75, 0.90):
        dur = 1400.0
        t = max(30.0, frac * dur)
        lag, c = S.offset_at(p, 0, p, 1, t, dur=60.0, maxlag=5.0)
        lags.append((round(t), None if lag is None else round(lag,1),
                     None if c is None else round(c,3)))
    good = [l for _, l, c in lags if l is not None and c is not None and c >= 0.25]
    spread = (max(good)-min(good)) if len(good) >= 2 else None
    verdict = ("CONSTANT" if spread is not None and spread <= 25 else
               "STEP -- shipped desync" if spread is not None else
               "not measurable (tracks do not correlate)")
    print(f"  {len(st)} tracks | accepted n={len(good)}/5 | "
          f"spread={spread if spread is None else round(spread,1)} ms | {verdict}",
          file=sys.stderr, flush=True)
    print(f"     {lags}", file=sys.stderr, flush=True)
print("PILOT DONE", file=sys.stderr)
