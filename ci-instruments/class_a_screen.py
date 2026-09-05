#!/usr/bin/env python3
"""Which of my non-whole-step files can test dev-1's question at all?

CLASS A: the candidate has DUPLICATE same-language tracks and BOTH correlate.
         Only these can distinguish "the EDIT is off-grid" from "this STREAM is".
CLASS B: one track per language. Can only ever return "two languages agree", which
         is what id 52 and id 134 already gave twice.

id 52 LOOKED class A and collapsed to class B: its second `eng` returned fid 0.11/0.07,
which is no signal. So duplication alone is not the criterion -- BOTH MEMBERS MUST
CORRELATE. Screening on duplication without correlation would have sent me to run id 52
again and call the result a null.

Cheap: two probes per duplicated language, one window, no full sequence.
"""
import json, subprocess, sys, collections
sys.path.insert(0, '.')
import scan_hidden as S

MIN_FID = 0.7          # dev-1's bar. Stated, not implicit.

def astreams(p):
    q = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=index:stream_tags=language",
                        "-of", "csv=p=0", p], capture_output=True, timeout=45)
    out = []
    for line in q.stdout.decode().split():
        parts = line.split(',')
        if parts and parts[0].isdigit():
            out.append(parts[1] if len(parts) > 1 else None)
    return out          # relative order == a:N index

def main(ids, t):
    rec = {r['id']: r for r in json.load(open('runs/audio-durations.json'))}
    folder = {r['id']: r.get('folder_id') for r in json.load(open('runs/audio-durations.json'))}
    for i in ids:
        r = rec.get(i)
        if not r:
            continue
        M, C = r['master_path'], r['file_path']
        try:
            ml, cl = astreams(M), astreams(C)
        except Exception as e:
            print(f"  id {i:>4}: ffprobe failed {str(e)[:40]}")
            continue
        dup = [lg for lg, n in collections.Counter(cl).items() if lg and n > 1]
        if not dup:
            print(f"  id {i:>4} folder {str(folder.get(i)):>4}: CLASS B -- no duplicated candidate language")
            continue
        for lg in dup:
            if lg not in ml:
                print(f"  id {i:>4} folder {str(folder.get(i)):>4}: dup {lg} but master lacks it -- unusable")
                continue
            mi = ml.index(lg)
            cis = [n for n, x in enumerate(cl) if x == lg]
            fids = []
            for ci in cis:
                try:
                    _, fid = S.offset_at(M, mi, C, ci, t, dur=20.0, maxlag=4.0)
                except Exception:
                    fid = None
                fids.append((ci, fid))
            ok = [f for _, f in fids if f is not None and f >= MIN_FID]
            cls = "CLASS A" if len(ok) >= 2 else "class B (a duplicate did not correlate)"
            shown = "  ".join(f"a:{c}={('%.3f' % f) if f is not None else 'FAIL'}" for c, f in fids)
            print(f"  id {i:>4} folder {str(folder.get(i)):>4}: dup={lg}  {shown}   -> {cls}")

main([int(x) for x in sys.argv[1:-1]], float(sys.argv[-1]))
