#!/usr/bin/env python3
"""Declared video frame rate of each compared pair: error file against its master.

DECIDES: whether adjust_delay_to_frame's second snap is a no-op. It snaps to the
BEST-QUALITY video's rate, while get_good_frame snapped to the MINIMUM of the two.
Equal rates -> harmless. Mixed rates -> the delay moves off the frame the PSNR scan
chose, by up to 16.7 ms and growing.

Read-only, r_frame_rate only, no container.
"""
import json, os, subprocess, sys
from fractions import Fraction

errs = {e['id']: e['file_path'] for e in json.load(open('runs/errors-7b83af4.json'))['incompatible_files']}
ad = {r['id']: r for r in json.load(open('runs/audio-durations.json'))}
out = open('runs/framerate-pairs.jsonl', 'a')
done = set()
if os.path.exists('runs/framerate-pairs.jsonl'):
    for l in open('runs/framerate-pairs.jsonl'):
        try: done.add(json.loads(l)['id'])
        except Exception: pass

def rate(p):
    if not p or not os.path.exists(p): return None
    try:
        r = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                            'stream=r_frame_rate,avg_frame_rate','-of','json',p],
                           capture_output=True, timeout=120)
        s = (json.loads(r.stdout.decode() or '{}').get('streams') or [{}])[0]
        v = s.get('r_frame_rate') or s.get('avg_frame_rate')
        if not v or v == '0/0': return None
        return round(float(Fraction(v)), 3)
    except Exception:
        return None

for n, i in enumerate(sorted(errs), 1):
    if i in done: continue
    e = rate(errs[i]); m = rate((ad.get(i) or {}).get('master_path'))
    out.write(json.dumps({"id": i, "err_fps": e, "master_fps": m,
                          "equal": (None if (e is None or m is None) else abs(e - m) < 0.001)}) + "\n")
    out.flush()
    if n % 60 == 0: print(f"  {n}/{len(errs)}", file=sys.stderr, flush=True)
print("DONE", file=sys.stderr)
