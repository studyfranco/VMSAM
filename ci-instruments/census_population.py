#!/usr/bin/env python3
"""The canonical record set for any census of container behaviour.

`vmsam-forensic` named the shape: MY POPULATION CONTAINED MY OWN REDACTED COPIES OF THE
SAME RECORDS -- the instrument's output feeding back into the instrument's input. It
reported REFUSED 4 where there were two jobs, and it is a shape only I can hit, because
I am the one who makes copies.

    /config/output/KEEP/*.error          THE CONTAINER'S OWN RECORD -- canonical
    runs/decline-<id>.log.redacted       MY REDACTED COPY OF ONE -- derived

A derived copy is usable when the canonical record is ABSENT (it covers a job whose
error record was never preserved), and NEVER IN ADDITION to it. So the rule is not
"exclude mine" -- that would lose coverage -- it is:

    CANONICAL FIRST. A DERIVED COPY EARNS A PLACE ONLY BY COVERING A JOB NOTHING
    ELSE COVERS, AND IT SAYS SO.

Deduplication is by the record's CONTENT identity, not its filename: `candidate_digest`
where the image emits it, the plan-piece string otherwise, and the record's own name
where neither exists. THE LAST CASE MAY OVER-COUNT, WHICH IS THE SURVIVABLE DIRECTION --
an inflated count gets questioned; a silent omission is indistinguishable from a clean
corpus. My first deduplicator keyed on `candidate_digest` alone and SKIPPED what it
could not key, turning a double count into a disappearance and making the output look
cleaner.
"""
import glob, hashlib, os, re


def _key(txt, name):
    cd = re.search(r'repair: candidate_digest (\w+)', txt)
    if cd:
        return cd.group(1)[:12], 'candidate_digest'
    pl = re.search(r'repair: plan [^\n]*?pieces=([^\n]*)', txt)
    if pl:
        return hashlib.sha256(pl.group(1).encode()).hexdigest()[:12], 'plan-string'
    # NO VALID IDENTITY EXISTS FOR THESE, AND I TRIED TO INVENT ONE.
    # 49 records carry neither a digest nor a plan (declines that never planned). Keying
    # on the filename may over-count when a KEEP/*.error and a runs/decline-* cover one
    # job -- so I built a key meant to SURVIVE REDACTION: the sorted multiset of numeric
    # tokens, since redaction replaces paths and not numbers.
    #
    # IT MERGED FOUR DISTINCT JOBS. decline-20, -21, -22 and -23 have four distinct
    # contents, four distinct error ids, and one numeric signature. The "bound" it
    # produced -- "over-counts by at most 7" -- came from a key that UNDER-COUNTS, so
    # the number was meaningless and I nearly published it.
    #
    # So the honest state: for these 49 there is NO measurable duplication bound, and
    # keying on the filename is the CONSERVATIVE choice. Over-counting gets questioned;
    # merging distinct jobs is invisible and reads as a cleaner corpus.
    return 'unkeyed:' + name, 'NO KEY -- keyed on filename; may over-count, bound NOT MEASURABLE'


def records(include_derived=True):
    """[(path, key, key_source, is_derived)], canonical first, one entry per job."""
    canonical = sorted(glob.glob('/config/output/KEEP/*.log') +
                       glob.glob('/config/output/KEEP/*.error'))
    derived = sorted(glob.glob('runs/decline-*.log.redacted'))
    out, seen = [], {}
    for path in canonical:
        txt = open(path, encoding='utf-8', errors='replace').read()
        k, how = _key(txt, os.path.basename(path))
        if k in seen:
            continue
        seen[k] = path
        out.append((path, k, how, False))
    if include_derived:
        for path in derived:
            txt = open(path, encoding='utf-8', errors='replace').read()
            k, how = _key(txt, os.path.basename(path))
            if k in seen:
                continue          # the container's own record already covers this job
            seen[k] = path
            out.append((path, k, how, True))
    return out


if __name__ == '__main__':
    rows = records()
    can = [r for r in rows if not r[3]]
    der = [r for r in rows if r[3]]
    raw = len(glob.glob('/config/output/KEEP/*.log')) + \
          len(glob.glob('/config/output/KEEP/*.error')) + \
          len(glob.glob('runs/decline-*.log.redacted'))
    print(f"  raw files on disk                      {raw}")
    print(f"  DISTINCT JOBS                          {len(rows)}")
    print(f"    from the container's own records     {len(can)}")
    print(f"    from my derived copies, adding cover {len(der)}")
    import collections
    print(f"  key sources: {dict(collections.Counter(r[2] for r in rows))}")
    if der:
        print("\n  derived copies covering a job nothing else covers:")
        for p, k, how, _ in der[:8]:
            print(f"    {os.path.basename(p)}  key={how}")
