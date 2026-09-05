#!/usr/bin/env python3
"""Find a master whose audio does NOT start at PTS 0.

dev-2's assembler ends every piece with asetpts=PTS-STARTPTS, so a produced track
starts at 0. If the master's track also starts at 0 the two agree by accident, and
the test cannot fail -- which is exactly what id 150 is: every stream 0.000000.

THE CONTROL HAS TO BE A FILE WHERE THE TWO TIMELINES COULD DISAGREE.
"""
import json, subprocess, sys
ad=json.load(open('runs/audio-durations.json'))
seen=set(); hits=[]
for n,r in enumerate(ad,1):
    m=r['master_path']
    if m in seen: continue
    seen.add(m)
    try:
        q=subprocess.run(["ffprobe","-v","error","-select_streams","a","-show_entries",
            "stream=index,start_time","-of","csv=p=0",m],capture_output=True,timeout=45)
        vals=[x for x in q.stdout.decode().split() if x]
    except Exception:
        continue
    nz=[v for v in vals if v.split(',')[-1] not in ('0.000000','N/A','0')]
    if nz:
        hits.append({"id":r['id'],"start_times":nz[:4]})
        json.dump(hits,open('runs/nonzero-start.json','w'),indent=1)
        print(f"[{n}] id={r['id']} NON-ZERO start_time: {nz[:3]}",file=sys.stderr,flush=True)
print(f"DONE: {len(hits)} masters with a non-zero audio start_time",file=sys.stderr)
