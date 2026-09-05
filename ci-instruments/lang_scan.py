#!/usr/bin/env python3
"""Find pairs whose candidate carries an audio language the master lacks.
Those are the only files where keep_best_audio has no master track to compete
against, so they discriminate 'dropped in per-language competition' from
'never reached the merge'."""
import json, subprocess, sys
ap=json.load(open('runs/all-pairs.json'))
def langs(p):
    try:
        q=subprocess.run(["ffprobe","-v","error","-select_streams","a",
            "-show_entries","stream_tags=language","-of","csv=p=0",p],
            capture_output=True,timeout=45)
        return set(x.strip() for x in q.stdout.decode().split() if x.strip())
    except Exception: return set()
out=[]
for n,r in enumerate(ap,1):
    ml=langs(r['master_path'])
    if not ml: continue
    cl=langs(r['file_path'])
    if not cl: continue
    extra=cl-ml
    if extra:
        out.append({"id":r['id'],"folder":r['folder_id'],"extra":sorted(extra)})
        json.dump(out,open('runs/lang-asymmetry.json','w'),indent=1)
        print(f"[{n}/{len(ap)}] id={r['id']} folder={r['folder_id']} extra={sorted(extra)}",
              file=sys.stderr,flush=True)
print("LANGSCAN DONE",file=sys.stderr)
