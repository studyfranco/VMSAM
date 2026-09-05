#!/usr/bin/env python3
"""The join dev-2's fix needs a delivered artefact for.

Conditions, from dev-2: a master with a nonzero start_time IN A LANGUAGE THE
CANDIDATE ALSO CARRIES, and that language must NOT be the delay language --
because only the delay language goes through keep_best_audio at 1888. Every other
shared language goes through find_differences_and_keep_best_audio at 1896, which
partitions compatible/not-compatible FIRST, so an incompatible repaired track is
in its own group and is not competing.

Delay language is taken as the folder's original_language (ja for 283 of 295).
"""
import json, subprocess, sys
nz={r['id'] for r in json.load(open('runs/nonzero-start.json'))}
ad={r['id']:r for r in json.load(open('runs/audio-durations.json'))}
fol={f['id']:f for f in json.load(open('runs/folders.json'))['folders']}
L3={'ja':'jpn','zh':'chi','en':'eng'}

def streams(p):
    q=subprocess.run(["ffprobe","-v","error","-select_streams","a","-show_entries",
        "stream=index,start_time:stream_tags=language","-of","json",p],
        capture_output=True,timeout=60)
    try: return json.loads(q.stdout.decode()).get('streams',[])
    except Exception: return []

hits=[]
for n,i in enumerate(sorted(nz),1):
    d=ad.get(i)
    if not d: continue
    ms=streams(d['master_path']); cs=streams(d['file_path'])
    if not ms or not cs: continue
    clangs={s.get('tags',{}).get('language') for s in cs}
    delay=L3.get(fol.get(d.get('folder_id'),{}).get('original_language','ja'),'jpn')
    for s in ms:
        st=s.get('start_time')
        lang=s.get('tags',{}).get('language')
        if st in (None,'0.000000','N/A'): continue
        if lang and lang in clangs and lang != delay:
            hits.append({"id":i,"lang":lang,"start_time":st,"delay_lang":delay})
            print(f"  id {i:<5} lang={lang} start_time={st} (delay lang={delay})  <-- CANDIDATE",
                  file=sys.stderr,flush=True)
            break
    json.dump(hits,open('runs/devfix-candidates.json','w'),indent=1)
print(f"DONE: {len(hits)} files satisfy nonzero start on a shared NON-delay language",
      file=sys.stderr)
