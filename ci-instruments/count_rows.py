#!/usr/bin/env python3
"""Count VERDICT ROWS, not lines containing the word "verdict".

`grep -c '"verdict"'` counted a CORRECTION row whose nested text mentions the field,
and reported 7 where 6 rows exist. A substring match on a JSON file is not a field
test -- the same defect as a verdict field that measured a file extension, in a shell
one-liner instead of a schema. Every count I quote should come through here.
"""
import json, sys
from collections import Counter

def rows(path):
    out = []
    for line in open(path):
        try: d = json.loads(line)
        except Exception: continue
        if isinstance(d.get('verdict'), str) and isinstance(d.get('id'), int):
            out.append(d)
    return out

if __name__ == '__main__':
    for p in sys.argv[1:]:
        r = rows(p)
        c = Counter(x['verdict'] for x in r)
        # COUNT NON-ROWS BY STRUCTURE, NOT BY KEYWORD. My first version grepped for
        # 'correction'/'suspended'/'void' and MISSED THE ASK HEADER, so it reported 2
        # where 3 exist -- a keyword scan inside the very tool written because a
        # keyword scan is not a field test. Anything that is not a verdict row IS a
        # non-row record, by definition, and needs no vocabulary.
        total = sum(1 for line in open(p) if line.strip())
        n_corr = total - len(r)
        print(f"  {p}")
        print(f"    verdict rows: {len(r)}   {dict(c)}")
        print(f"    non-row records (corrections, voids, headers): {n_corr}")
