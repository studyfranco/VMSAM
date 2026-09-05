#!/usr/bin/env python3
"""Narrow the 21:21 series by TRACK STRUCTURE, not by duration.

The signature is "7 audio and 24 subtitle tracks REBUILT". A file needing seven
audio tracks rebuilt must carry at least seven. This is a NARROWING, NOT AN
IDENTIFICATION -- the output is a shortlist to put in front of the owner, whose
observation it is, because nobody in the fleet holds the original run: every
occurrence is a downstream citation of one sentence the owner wrote in prose.

Read-only, stream counts only, no packet counting, no container.
"""
import json, os, subprocess, sys
from redact import safe_text as _safe
from collections import Counter

errs = json.load(open('runs/errors-7b83af4.json'))['incompatible_files']
out = open('runs/track-structure.jsonl', 'a')
done = set()
if os.path.exists('runs/track-structure.jsonl'):
    for l in open('runs/track-structure.jsonl'):
        try: done.add(json.loads(l)['id'])
        except Exception: pass

for n, e in enumerate(errs, 1):
    if e['id'] in done: continue
    try:
        r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                            'stream=index,codec_type', '-of', 'json', e['file_path']],
                           capture_output=True, timeout=180)
        st = json.loads(r.stdout.decode() or '{}').get('streams', [])
        c = Counter(s.get('codec_type') for s in st)
        out.write(json.dumps({"id": e['id'], "audio": c.get('audio', 0),
                              "subtitle": c.get('subtitle', 0),
                              "video": c.get('video', 0)}) + "\n")
    except Exception as ex:
        out.write(json.dumps({"id": e['id'], "error": _safe(ex)}) + "\n")
    out.flush()
    if n % 60 == 0: print(f"  {n}/{len(errs)}", file=sys.stderr, flush=True)
print("STRUCTURE DONE", file=sys.stderr)
