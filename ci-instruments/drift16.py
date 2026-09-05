#!/usr/bin/env python3
"""Re-measure the drift file at 16 positions, testing vmsam-forensic's FROZEN
prediction: a step of roughly 10-15 ms between t=700 and t=950, with the two segment
slopes AGREEING with each other once there are 5+ points a side.

Its reasoning, which I could not have produced: at n=5 my LS slope (+27.10) and my
endpoint slope (+25.00) disagree by 7.7 %, which for a pure line with symmetric noise
they should not; and the residuals about the single line are +0.75, +0.60, -4.80,
+4.80, -1.35 -- ONE ADJACENT EQUAL-AND-OPPOSITE PAIR CARRYING 94 % OF THE SUM OF
SQUARES, which is what a step in the middle of a linear fit produces and is not how
noise arranges itself.

The residual pattern is computed from the SINGLE-LINE fit alone and is therefore
independent of any break search. The SSE ratio and the segment slopes at n=5 are NOT
evidence -- a break after point 3 leaves two points on one side, which fit exactly.
"""
import json, os, subprocess, sys, collections
import numpy as np
sys.path.insert(0, '.')
from scan_accepted import probe          # now import-safe: body is behind __main__

# LOCATE BY LAG SIGNATURE, NOT BY RANK. My first attempt selected on
# (lang == 'fre' and spread > 700) and got one of the SIX FALSE STEPS instead --
# the drift file's spread is 25.0, not 778. A rank-based locator picks whatever is
# top of a list that has since changed underneath it.
# LOCATE BY folder_id -- OPAQUE AND UNIQUE. My scan rows key on the last 12
# characters of the folder path, chosen for privacy: 295 folders collapse to 194
# keys, and 167 folders share one. That identifier cannot map a row back to a file
# for most of the corpus, which is why the first re-measurement opened the wrong
# folder. folders.json has carried a unique numeric id the whole time.
FOLDER_ID = 32
fol = json.load(open('runs/folders.json'))['folders']
d = next(f['destination_path'] for f in fol if f['id'] == FOLDER_ID)
print(f"TARGET: folder_id {FOLDER_ID}", file=sys.stderr)
p = os.path.join(d, sorted(x for x in os.listdir(d) if x.endswith('.mkv'))[0])
q = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                    "-show_entries", "stream_tags=language", "-of", "csv=p=0", p],
                   capture_output=True, timeout=45)
langs = [x.strip() for x in q.stdout.decode().split() if x.strip()]
dup = [k for k, v in collections.Counter(langs).items() if v > 1]
idx = [i for i, l in enumerate(langs) if l == dup[0]][:2]
print(f"TRACK PAIR MEASURED: lang={dup[0]}, a:{idx[0]} vs a:{idx[1]}   "
      f"(the 7th field, which people forget)", file=sys.stderr, flush=True)
print(f"TIMELINE: positions are seconds in the MERGED OUTPUT FILE", file=sys.stderr, flush=True)

pts = []
for k in range(16):
    t = 100.0 + k * 90.0
    lag, mag, pr = probe(p, idx[0], idx[1], t)
    ok = pr is not None and pr >= 10.0
    if ok:
        pts.append((t, lag))
    print(f"  t={t:>6.0f}  lag={'None' if lag is None else round(lag,2):>9}  "
          f"prom={'n/a' if pr is None else round(pr,1):>7}{'' if ok else '  rejected'}",
          file=sys.stderr, flush=True)
json.dump(pts, open('runs/drift16-points.json', 'w'))  # append not needed: derived
print(f"\naccepted {len(pts)} of 16", file=sys.stderr)
