#!/usr/bin/env python3
"""Owner rule: >5 lines, twice, same purpose -> a function.

THE TRIGGER IS MECHANICAL, THE VERDICT IS NOT. This finds candidates by structure;
every hit needs the counter-rule applied by a human eye:

  RESEMBLANCE IS NOT PURPOSE. The test is "would a change to one be WRONG TO LEAVE
  OUT of the other", not "do these look alike".

Comments and blank lines are stripped before comparison, because two blocks that
differ only in their comments are the same code -- and two blocks with identical
comments and different code are the more dangerous case, which this would MISS.
That limit is stated rather than hidden.
"""
import sys, os, hashlib, collections, re

def norm(path):
    out = []
    for n, raw in enumerate(open(path, errors='replace'), 1):
        s = raw.split('#')[0].rstrip() if not raw.lstrip().startswith('#') else ''
        if s.strip():
            out.append((n, re.sub(r'\s+', ' ', s.strip())))
    return out

MIN = 5
def main(root):
    files = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs
             if f.endswith('.py') and 'dup_blocks' not in f]
    idx = collections.defaultdict(list)
    for p in files:
        L = norm(p)
        for i in range(len(L) - MIN + 1):
            blk = tuple(t for _, t in L[i:i + MIN])
            if len(set(blk)) < 3:          # skip runs of near-identical filler
                continue
            idx[hashlib.sha256('\n'.join(blk).encode()).hexdigest()].append(
                (p, L[i][0], blk))
    hits = {k: v for k, v in idx.items() if len({(p) for p, _, _ in v}) >= 2
            or len(v) >= 2}
    print(f"files scanned: {len(files)}   duplicated {MIN}-line blocks: {len(hits)}")
    shown = 0
    for k, v in sorted(hits.items(), key=lambda x: -len(x[1])):
        locs = [f"{os.path.basename(p)}:{ln}" for p, ln, _ in v]
        if len(set(p for p, _, _ in v)) < 2:
            continue                       # same-file repeats: report separately
        shown += 1
        if shown > 8: break
        print(f"\n  x{len(v)}  {' | '.join(locs)}")
        for line in v[0][2]:
            print(f"      {line[:96]}")
    same = [v for v in hits.values() if len(set(p for p, _, _ in v)) == 1]
    print(f"\n  blocks duplicated WITHIN one file: {len(same)}")

main(sys.argv[1] if len(sys.argv) > 1 else '.')
