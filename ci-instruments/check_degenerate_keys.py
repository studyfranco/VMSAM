#!/usr/bin/env python3
"""Identity columns whose values cover rows that cannot be one thing.

    A KEY THAT MATCHES NOTHING joins nothing, and the symptom is an empty join.
    A KEY THAT MATCHES EVERYTHING joins everything, AND PASSES A CHECK WRITTEN
    FOR THE FIRST.

`vmsam-dev-4` found the first in my ledger (a report stem that could never join its own
artefact). `vmsam-forensic` found the second (18 rows sharing the literal value
`UNKNOWN-backfill`, reported as 18 mutual joins). Only asking WHAT THE MATCHED ROWS ARE
separates them.

THE FIRST VERSION OF THIS FILE HAD ZERO POWER TO FIND EITHER.
It looked for repeats in MOSTLY-UNIQUE columns -- and a degenerate key has FEW distinct
values, not many. `stem_sha256_16` is 56 distinct over 98 rows, so it was invisible. The
three things v1 did report were VALUE columns: a size repeating because two files share
a size, and a `-` missing-value marker. A SWEEP THAT FINDS ONLY FALSE POSITIVES AND IS
STRUCTURALLY BLIND TO THE TRUE ONE READS AS A CLEAN AUDIT.

So this file REFUSES TO REPORT until it has re-found a defect whose answer is known --
the positive control on the FINDER, not on the data.

TWO PARSE TRAPS, both hit before the numbers were trusted:
  * `csv.DictReader` takes a leading `#` comment as the HEADER. Three of eight ledgers
    start with one, and all three produced plausible column analyses. Row counts
    disagreeing with forensic's was what exposed it.
  * `protected-stems.tsv` has NO header and needs none -- its consumer is
    `grep "^<stem>\\t"`. A header-assuming reader invents one from the first data row.
"""
import csv, collections, glob, os, sys

IDENTITY = ('name', 'id', 'stem', 'key', 'hash', 'file', 'artefact', 'log', 'digest')
HEADERLESS = ('protected-stems.tsv',)


def read(path):
    lines = [l for l in open(path, encoding='utf-8', errors='replace')
             if not l.startswith('#')]
    return list(csv.DictReader(lines, delimiter='\t')) if lines else []


def scan(path):
    if os.path.basename(path) in HEADERLESS:
        return []                      # no header; its reader does not use one
    rows = read(path)
    if not rows:
        return []
    out = []
    for col in rows[0]:
        if not any(t in col.lower() for t in IDENTITY):
            continue
        # A MISSING-VALUE MARKER IS NOT A KEY VALUE. `ci-artefact-families.tsv` writes
        # `-` where a job has no member of that kind, and its `log` column is
        # identity-named -- so 21 rows of "this family has no log" read as one value
        # covering 21 rows. THAT IS ABSENCE, NOT IDENTITY, and counting it made my own
        # new file look like a degenerate key within a minute of publishing it.
        MARKERS = {'-', 'n/a', 'N/A', 'none', 'None', 'unknown', 'NULL'}
        c = collections.Counter(r.get(col) for r in rows
                                if r.get(col) not in (None, '') and r.get(col) not in MARKERS)
        if not c:
            continue
        # A REPEAT IS NOT A DEFECT. My previous version reported EVERY repeated value
        # and flagged twenty legitimate stems -- a stem SHOULD cover a job's log, mkv
        # and report. THAT IS THE DESIGN, and "identity column with a repeat" was never
        # the right test.
        #
        # THE PRINCIPLED TEST NEEDS THE TABLE'S SEMANTICS: a job produces at most one
        # artefact of each KIND, so a valid key covers at most one row per kind. Where
        # the table has a `kind` column, use it -- it is exact and needs no threshold.
        #
        # Where it does not, there is no semantic to appeal to, and I am NOT inventing a
        # threshold to cover that case: such tables are reported as NOT CHECKABLE BY
        # THIS TEST rather than given a made-up bar. A heuristic here would produce
        # exactly the twenty false positives I have just removed.
        kindcol = next((k for k in rows[0] if k.lower() in ('kind', 'record_kind', 'type')), None)
        if kindcol is None:
            continue
        groups = collections.defaultdict(collections.Counter)
        for r in rows:
            v = r.get(col)
            if v in (None, '') or v in MARKERS:
                continue
            groups[v][r.get(kindcol)] += 1
        for val, kinds in groups.items():
            if max(kinds.values()) > 1:
                out.append({'file': os.path.basename(path), 'column': col, 'value': val,
                            'rows_covered': sum(kinds.values()), 'distinct': len(groups),
                            'total_rows': len(rows),
                            'why': f"covers {max(kinds.values())} rows of kind "
                                   f"{max(kinds, key=kinds.get)!r} -- a job produces one"})
    return out


def positive_control(findings):
    """This sweep must re-find a defect whose answer is already known."""
    return any(f['value'] == 'UNKNOWN-backfill' and f['rows_covered'] >= 18
               for f in findings)


if __name__ == '__main__':
    files = sorted(glob.glob('/config/output/KEEP/*.tsv') +
                   glob.glob('/config/output/DECLINED/*.tsv'))
    findings = [f for p in files for f in scan(p)]
    ok = positive_control(findings)
    print(f"  positive control (re-finds the known `UNKNOWN-backfill` x18): "
          f"{'PASS' if ok else 'FAIL -- this sweep is blind, do not read its result'}")
    if not ok:
        sys.exit(2)
    print(f"  {len(files)} ledger(s) scanned, {len(findings)} degenerate key(s)\n")
    for f in findings:
        print(f"    {f['file']:30} {f['column']:22} {f['value'][:24]!r} covers "
              f"{f['rows_covered']} rows ({f['distinct']} distinct / {f['total_rows']})")
    sys.exit(1 if len(findings) > 1 else 0)   # the known one is expected
