#!/usr/bin/env python3
"""Re-scan the files whose offset lay outside the old fixed +-60 s baseline search."""
import json, os, subprocess, sys
sys.path.insert(0, '.')
import scan_hidden as S
from run_full import matched_pair
ad={r['id']:r for r in json.load(open('runs/audio-durations.json'))}
rec={r['id']:r for r in json.load(open('runs/log-matrices2.json'))}
ids=json.load(open('runs/redo-ids.json'))
out=open('runs/full-out.jsonl','a')
for n,i in enumerate(ids,1):
    d=ad[i]; pp=rec.get(i,{}).get('per_pair',{})
    mp=matched_pair(d['master_path'], d['file_path'])
    if mp is None:
        rec_out={'id':i,'label':'no-shared-language','detail':'COULD NOT MEASURE','sequence':[]}
    else:
        ms,cs,lang=mp
        r={'id':i,'master_path':d['master_path'],'file_path':d['file_path'],'ms':ms,'cs':cs,
           'err_audio_s':d['err_audio_s'],'mst_audio_s':d['mst_audio_s'],'per_pair':pp}
        try: rec_out=S.scan(r,n_probes=30)
        except Exception as e: rec_out={'id':i,'label':'instrument-failed','detail':str(e)[:90],'sequence':[]}
        rec_out['track_pair']={'lang':lang,'master_a':ms,'candidate_a':cs}
    rec_out['rescanned_wide_baseline']=True
    out.write(json.dumps(rec_out)+"\n"); out.flush()
    print(f"[{n}/{len(ids)}] id={i} {rec_out['label']}: {rec_out.get('detail','')[:50]}",
          file=sys.stderr, flush=True)
print("REDO DONE", file=sys.stderr)
