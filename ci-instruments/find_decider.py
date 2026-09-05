#!/usr/bin/env python3
"""Find a file whose BORROWED track cannot be French.

forensic has three refuted artefacts, all with a short `fre` track, and one clean
artefact that is the only one with NO fre track. With n=4 and fre always in the
borrowed role, "French" and "the secondary-source track" are not separable. ONE FILE
WHOSE BORROWED TRACK IS ANOTHER LANGUAGE DECIDES IT.

The borrow happens when the candidate needs a language the PRIMARY master source does
not supply. So the decisive population is: masters carrying two or more languages
where NONE of them is fre -- there the borrowed track is necessarily something else.

Read-only, language tags only, no container.
"""
import json, os, subprocess, sys

errs = {e['id']: e['file_path'] for e in json.load(open('runs/errors-7b83af4.json'))['incompatible_files']}
ad = {r['id']: r for r in json.load(open('runs/audio-durations.json'))}

def langs(p):
    if not p or not os.path.exists(p): return None
    try:
        r = subprocess.run(['ffprobe','-v','error','-select_streams','a','-show_entries',
                            'stream_tags=language','-of','json',p], capture_output=True, timeout=120)
        st = json.loads(r.stdout.decode() or '{}').get('streams', [])
        return sorted({(s.get('tags') or {}).get('language') for s in st if (s.get('tags') or {}).get('language')})
    except Exception:
        return None

out = open('runs/decider-candidates.jsonl', 'a')
done = set()
if os.path.exists('runs/decider-candidates.jsonl'):
    for l in open('runs/decider-candidates.jsonl'):
        try: done.add(json.loads(l)['id'])
        except Exception: pass

for n, i in enumerate(sorted(errs), 1):
    if i in done: continue
    ml = langs((ad.get(i) or {}).get('master_path'))
    el = langs(errs[i])
    if ml is None or el is None:
        out.write(json.dumps({"id": i, "error": "unreadable"}) + "\n"); out.flush(); continue
    shared = sorted(set(ml) & set(el))
    out.write(json.dumps({"id": i, "master_langs": ml, "err_langs": el,
                          "shared": shared, "master_multi": len(ml) > 1,
                          "master_has_fre": 'fre' in ml,
                          "DECISIVE": len(ml) > 1 and 'fre' not in ml and len(shared) > 1}) + "\n")
    out.flush()
    if n % 60 == 0: print(f"  {n}/{len(errs)}", file=sys.stderr, flush=True)
print("DONE", file=sys.stderr)
