#!/usr/bin/env python3
"""Condition (2): does a file whose RECORD is blind carry a step only I can see?

Frame: record span <= 1 quantum AND language-asymmetric. Aimed by the RECORD's
blind spot deliberately -- I am hunting ONE exemplar to run through the repair,
not computing a rate. It cannot touch the 2.3 % bound.
"""
import json, subprocess, sys
sys.path.insert(0,'.')
import scan_hidden as S
from hunt_subquantum import matched_pair

ad={r['id']:r for r in json.load(open('runs/audio-durations.json'))}
rec={r['id']:r for r in json.load(open('runs/log-matrices2.json'))}
IDS=[45,172,271,21,7,15,19,11]
for i in IDS:
    d=ad.get(i)
    if not d: print(f"  id {i}: no duration record",file=sys.stderr,flush=True); continue
    mp=matched_pair(d['master_path'],d['file_path'])
    if mp is None:
        print(f"  id {i}: NO SHARED LANGUAGE -- COULD NOT MEASURE",file=sys.stderr,flush=True); continue
    ms,cs,lang=mp
    r={'id':i,'master_path':d['master_path'],'file_path':d['file_path'],'ms':ms,'cs':cs,
       'err_audio_s':d['err_audio_s'],'mst_audio_s':d['mst_audio_s'],
       'per_pair':rec.get(i,{}).get('per_pair',{})}
    try: res=S.scan(r,n_probes=30)
    except Exception as e: res={'id':i,'label':'instrument-failed','detail':str(e)[:60]}
    print(f"  id {i:<5} [{lang} a:{ms}/a:{cs}] {res['label']:<20} {res.get('detail','')[:55]}",
          file=sys.stderr,flush=True)
print("HUNT3 DONE",file=sys.stderr)
