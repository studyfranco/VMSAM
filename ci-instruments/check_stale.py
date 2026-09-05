#!/usr/bin/env python3
"""Has a preserved artefact CHANGED since it was preserved?

A HARD LINK IS NOT A SNAPSHOT. The preserver links rather than copies, so the KEEP
copy shares an inode with the original -- and anything that rewrites the original in
place rewrites the preserved copy too. Identity survives; content need not.

Rows written before 2026-09-04T18:3xZ carry no size/mtime, so their staleness is
UNKNOWN, not false. Absent is not zero -- fourth time today.
"""
import csv, os, sys

L = '/config/output/KEEP/ci-preservation-ledger.tsv'
K = '/config/output/KEEP'

def main():
    rows = [r for r in csv.DictReader(open(L), delimiter='\t')
            if r.get('keep_name') and not r['keep_name'].startswith('#')]
    n_ok = n_stale = n_unknown = n_gone = 0
    for r in rows:
        p = os.path.join(K, r['keep_name'])
        if not os.path.exists(p):
            n_gone += 1
            print(f"  GONE      {r['keep_name']}")
            continue
        rec_size = (r.get('size_at_capture') or '').strip()
        if not rec_size:
            n_unknown += 1
            continue
        cur = os.path.getsize(p)
        if str(cur) != rec_size:
            n_stale += 1
            print(f"  STALE     {r['keep_name']}  captured {rec_size} -> now {cur}  "
                  f"(delta {cur - int(rec_size):+d} bytes)")
        else:
            n_ok += 1
    print(f"\n  unchanged {n_ok}   STALE {n_stale}   unknown (pre-dates the field) {n_unknown}   gone {n_gone}")
    # exit 0 clean, 1 stale found, 2 nothing checkable -- three outcomes, by name
    if n_stale:
        return 1
    if n_ok == 0:
        print("  NOTHING WAS CHECKABLE -- every row pre-dates size recording.")
        return 2
    return 0

sys.exit(main())
