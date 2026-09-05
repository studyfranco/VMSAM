#!/usr/bin/env python3
"""Corpus-wide subtitle cue census -- the population that can reach ZERO BYTES.

WHY: an all-dropped subtitle saves as an empty event list. A .ass keeps its headers
and survives; A .srt HAS NOTHING TO KEEP AND BECOMES ZERO BYTES, and a zero-byte
input kills the whole mux AFTER every audio track has been built. One file came
within FOUR CUES of it. Four cues is not a freak, it is a population.

READ-ONLY AND CONTAINER-FREE. Counts packets per subtitle stream in the SOURCE, so
it does not wait on the merge queue and cannot contend with it for the worker.

TWO DISTINCT AT-RISK CLASSES, and conflating them would hide the second:
  (a) FEW CUES  -- a handful of cues, all of which could fall outside the kept
                   pieces. Risk depends on the retime plan.
  (b) ZERO CUES AT SOURCE -- the track is ALREADY empty before any retiming.
                   No cue is dropped, so a guard phrased as "N dropped, 0 kept"
                   would not fire. Found on the first file sampled.
"""
import json, os, subprocess, sys, time
from redact import safe_text as _safe
from collections import Counter

ERRS = json.load(open('runs/errors-7b83af4.json'))['incompatible_files']
done = set()
if os.path.exists('runs/cue-census.jsonl'):
    for l in open('runs/cue-census.jsonl'):
        try: done.add(json.loads(l).get('id'))
        except Exception: pass
out = open('runs/cue-census.jsonl', 'a')

for n, e in enumerate(ERRS, 1):
    if e['id'] in done: continue
    p = e['file_path']
    try:
        s = subprocess.run(['ffprobe','-v','error','-select_streams','s','-show_entries',
                            'stream=index,codec_name:stream_tags=language','-of','json',p],
                           capture_output=True, timeout=300)
        streams = json.loads(s.stdout.decode() or '{}').get('streams', [])
        k = subprocess.run(['ffprobe','-v','error','-select_streams','s','-show_packets',
                            '-of','compact=p=0:nk=1','-show_entries','packet=stream_index',p],
                           capture_output=True, timeout=900)
        c = Counter(l.strip() for l in k.stdout.decode().splitlines() if l.strip())
    except Exception as ex:
        out.write(json.dumps({"id": e['id'], "error": _safe(ex)}) + "\n"); out.flush(); continue
    tr = [{"codec": st.get('codec_name'),
           "lang": (st.get('tags') or {}).get('language'),
           "cues": int(c.get(str(st.get('index')), 0))} for st in streams]
    out.write(json.dumps({"id": e['id'], "n_sub": len(tr), "tracks": tr}) + "\n"); out.flush()
    if n % 40 == 0:
        print(f"  {n}/{len(ERRS)}", file=sys.stderr, flush=True)
print("CENSUS DONE", file=sys.stderr)
