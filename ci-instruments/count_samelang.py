#!/usr/bin/env python3
"""Count SAME-LANGUAGE audio track pairs in the accepted population.

WHY. Inter-track correlation is the only route into the pipeline's accepted output,
and the positive control showed it fails on CROSS-language pairs: noise floor ~73 ms
against a sub-quantum signal below 62 ms. Two dubs share only a music-and-effects bed.
A SAME-LANGUAGE pair is not capped that way -- both tracks carry the same dialogue --
so if such pairs exist in quantity, the route is open on them and closed elsewhere.

This counts. It does not measure. Read-only.
"""
import json, os, subprocess, sys, collections
fol=json.load(open('runs/folders.json'))['folders']
dests=[f['destination_path'] for f in fol if f.get('destination_path') and os.path.isdir(f['destination_path'])]
attempted=0; probed=0; failed=[]; withdup=0; tally=collections.Counter(); examples=[]
for d in dests:
    try: files=sorted(x for x in os.listdir(d) if x.endswith('.mkv'))
    except Exception as e: failed.append(('listdir', str(e)[:30])); continue
    if not files: continue
    p=os.path.join(d, files[0]); attempted+=1
    try:
        q=subprocess.run(["ffprobe","-v","error","-select_streams","a","-show_entries",
            "stream_tags=language","-of","csv=p=0",p],capture_output=True,timeout=45)
        langs=[x.strip() for x in q.stdout.decode().split() if x.strip()]
    except Exception as e:
        failed.append((os.path.basename(d)[:1], str(e)[:30])); continue
    if not langs: failed.append((os.path.basename(d)[:1],'no audio list')); continue
    probed+=1
    c=collections.Counter(langs)
    dups={k:v for k,v in c.items() if v>1}
    tally[len(langs)]+=1
    if dups:
        withdup+=1
        if len(examples)<8: examples.append((len(langs), dups))
print(f"  folders attempted        {attempted}")
print(f"  probed successfully      {probed}")
print(f"  COULD NOT MEASURE        {len(failed)}   {failed[:4]}")
print(f"  with a SAME-LANGUAGE pair {withdup}   = {withdup/probed:.1%} of probed" if probed else "")
print(f"  examples (n_tracks, duplicated langs): {examples}")
