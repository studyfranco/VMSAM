#!/usr/bin/env python3
"""Capture-time size against settled size, for artefacts linked while still being written.

THE HARD LINK IS NOT THE PROBLEM -- it is the whole point. Linking early is what keeps
the artefact when the runner deletes its source, and the file completes ON THE SAME
INODE, so the preserved copy is whole regardless.

WHAT IS WRONG IS THE RECORD. `9c22d7d987840f1f.mkv` was linked at 23:25:44Z with the
producer still writing: `size_at_capture` reads 1 224 802 304 for a file that completed
at 1 610 057 582. The size was TRUE AT THAT INSTANT and is not the artefact's size.

WHY NOT JUST WAIT FOR STABILITY BEFORE LINKING: because the runner deletes its outputs,
and a preserver that waits is a preserver that sometimes loses the file. THE LINK MUST
BE EAGER AND THE MEASUREMENT MUST BE PATIENT -- they are different jobs and they were
sharing one moment.

WHY A SIDE TABLE RATHER THAN A COLUMN: I migrated two ledger schemas tonight and one of
them took a live append straight through the migration because my kill filter matched
nothing. A third migration to carry a value that only applies to a handful of rows is a
larger risk than the thing it records. Additive, atomic, no schema touched.

    ANNOTATE A TRUE READING, REPLACE A MIS-TYPED ONE.
`vmsam-forensic` normalised nine `duration_s` values in place and was right to: those
were the WRONG QUANTITY. Mine is a correct measurement of a transient state, so it
stays and gains a settled reading beside it.
"""
import csv, os, sys, time

LEDGER = '/config/output/KEEP/ci-preservation-ledger.tsv'
OUT = '/config/output/KEEP/ci-settled-sizes.tsv'
STABLE_S = 2.0


def settled(path):
    """Two reads, one file. Returns (size, stable) -- and NEVER a size it has not
    confirmed twice, because an unstable read is exactly the defect being recorded."""
    try:
        a = os.stat(path)
    except OSError:
        return None, False
    time.sleep(STABLE_S)
    try:
        b = os.stat(path)
    except OSError:
        return None, False
    return b.st_size, (a.st_size == b.st_size and a.st_mtime == b.st_mtime)


def main():
    rows = list(csv.DictReader(open(LEDGER), delimiter='\t'))
    out = []
    for r in rows:
        sz = r.get('size_at_capture')
        if not (sz or '').isdigit():
            continue
        p = '/config/output/KEEP/' + r['keep_name']
        if not os.path.exists(p):
            continue
        # CHEAP READ FIRST. The two-read stability check costs 2 s, and running it on
        # every row cost 98 x 2 s and timed out -- a settle check that never settles.
        # Only a row whose CURRENT size already differs from its capture is a candidate,
        # and there is exactly one of those.
        try:
            quick = os.path.getsize(p)
        except OSError:
            continue
        if quick == int(sz):
            continue
        now, stable = settled(p)
        if now is None:
            continue
        out.append({'keep_name': r['keep_name'],
                    'size_at_capture': sz,
                    'size_settled': now if stable else '',
                    'delta': now - int(sz) if stable else '',
                    'stable': 'yes' if stable else 'NO -- still changing, not recorded',
                    'captured_utc': r.get('captured_utc', '')})
    lines = ["# CAPTURE-TIME SIZE vs SETTLED SIZE. Written by ci.",
             "# An artefact hard-linked while its producer was still writing has a TRUE",
             "# size_at_capture that is not the artefact's size. The link is eager on",
             "# purpose -- the runner deletes its outputs -- so the fix is a second",
             "# reading, not a later link.",
             "# `size_settled` is EMPTY where two reads 2 s apart disagreed: an unstable",
             "# value is the defect, not a measurement of it.",
             "keep_name\tsize_at_capture\tsize_settled\tdelta\tstable\tcaptured_utc"]
    for o in out:
        lines.append('\t'.join(str(o[k]) for k in
                     ('keep_name', 'size_at_capture', 'size_settled', 'delta',
                      'stable', 'captured_utc')))
    tmp = OUT + '.tmp'
    open(tmp, 'w').write('\n'.join(lines) + '\n')
    os.replace(tmp, OUT)
    print(f"  {len(rows)} ledger rows checked, {len(out)} with a capture/settled mismatch")
    for o in out:
        print(f"    {o['keep_name']:26} {o['size_at_capture']} -> {o['size_settled']} "
              f"({o['delta']:+d} bytes)" if o['stable'] == 'yes' else
              f"    {o['keep_name']:26} UNSTABLE, not recorded")
    return 0


if __name__ == '__main__':
    sys.exit(main())
