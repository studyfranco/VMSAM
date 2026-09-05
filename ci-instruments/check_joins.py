#!/usr/bin/env python3
"""Does any number I have PUBLISHED rest on a failed join?

dev-2 printed `n_parents 0` from a join against a table lacking the key -- every
lookup returned None and it emitted a well-formed zero. I did the same three hours
earlier: a folder join against a file with no `folder_id` reported "254 rows span
1 folder", which is the most alarming possible finding and was an artefact.

THE DETECTOR IS ONE LINE: a resolve rate of 0 % is not a distribution, it is a
failed join. A LOW rate is a partial join and also worth naming. Only a HIGH rate
is a measurement.

This checks the artefacts I have filed, not the code that made them -- because the
question is whether a published number is an artefact, not whether a script could
produce one.
"""
import json, sys


def dup_keys(name, ids):
    """A DIFFERENT FAILURE FROM A FAILED JOIN, AND THE ONE THAT BIT.

    check_joins caught zero failed joins among my published artefacts and I filed a
    count anyway that was wrong in both terms: `6 of 14` counted ROWS where the unit
    is the id. ids 128 and 694 each appeared in two run files -- because of a redeploy
    I performed -- and I counted my own re-run twice.

    A join can resolve 100 % and the count still be wrong, because coverage says every
    key found a value and says NOTHING about how many times each key appeared.
    """
    import collections
    c = collections.Counter(ids)
    dups = {k: n for k, n in c.items() if n > 1}
    verdict = "ok" if not dups else f"DUPLICATE KEYS -- count rows != count ids"
    print(f"  {name:<44} {len(ids):>4} rows {len(c):>4} ids   {verdict}")
    if dups:
        print(f"       repeated: {dict(list(dups.items())[:6])}")
    return len(dups)

def rate(name, values):
    n = len(values)
    resolved = sum(1 for v in values if v is not None)
    pct = 100.0 * resolved / n if n else 0.0
    verdict = ("FAILED JOIN" if pct == 0 else
               "PARTIAL -- name it" if pct < 95 else "ok")
    print(f"  {name:<44} {resolved:>4}/{n:<4} {pct:5.1f}%  {verdict}")
    return pct

bad = 0
# 1. the language sweep -- folder_id joined from audio-durations
d = json.load(open('runs/lang-disjoint.json'))
eps = [e for v in d.values() for e in v]
bad += rate("lang-disjoint: master_langs present", [e.get('master_langs') for e in eps]) == 0
bad += rate("lang-disjoint: cand_langs present", [e.get('cand_langs') for e in eps]) == 0

# 2. matched pairs -- fps joined from framerate-pairs
m = json.load(open('runs/matched-pairs.json'))['rows']
bad += rate("matched-pairs: master_fps resolved", [r.get('master_fps') for r in m]) == 0
bad += rate("matched-pairs: err_fps resolved", [r.get('err_fps') for r in m]) == 0

# 3. the window sweep
w = json.load(open('runs/rate-window-sweep.json'))['rows']
bad += rate("rate-window-sweep: corr resolved", [r.get('corr') for r in w]) == 0
bad += rate("rate-window-sweep: ratio present", [r.get('compensation_ratio') for r in w]) == 0

# 4. the pair sweep trace
tr = [json.loads(l) for l in open('runs/pair-sweep-trace.jsonl')]
fin = [r for r in tr if r.get('event') == 'finished']
bad += rate("pair-sweep: verdict present", [r.get('verdict') for r in fin]) == 0

# 5. the run itself -- folder join used in every clustering statement I filed
recs = json.load(open('runs/audio-durations.json'))
folder = {r['id']: r.get('folder_id') for r in recs}
rows = []
for l in open('runs/full-out.jsonl'):
    try: rows.append(json.loads(l))
    except Exception: pass
ids = [r['id'] for r in rows if isinstance(r, dict) and 'id' in r]
bad += rate("full-out -> folder_id join (clustering)", [folder.get(i) for i in ids]) == 0

print()
# DUPLICATE-KEY CHECK on the artefacts whose COUNTS I have published
import glob
wr_ids = []
for f in glob.glob('runs/*.jsonl'):
    for l in open(f):
        try: d = json.loads(l)
        except Exception: continue
        if isinstance(d, dict) and 'would_refuse' in d and 'id' in d:
            wr_ids.append(d['id'])
dups = dup_keys("would_refuse population (the gate figure)", wr_ids)

print()
print(f"  failed joins among published artefacts: {bad}")
print(f"  duplicate-key populations: {dups}")
bad += dups
sys.exit(1 if bad else 0)
