#!/usr/bin/env python3
"""Publish which preserved files belong to one job, for consumers who are not me.

Every file in KEEP is named `sha256(ITS OWN source path)[:16]`, so an artefact and its
log SHARE NO PART OF THEIR NAMES. The only binding is the `stem_sha256_16` column of
`ci-preservation-ledger.tsv`.

`vmsam-forensic` searched for a log "under its keep_name or its stem", found none, and
filed a verdict as UNVERIFIABLE for want of a log THAT WAS ON DISK ALL ALONG --
`ff0330f3d3d59793.log`, stem `9f40ea826af07f8e`. Their search was correct and could not
have worked.

    A BINDING THAT ONLY ITS AUTHOR CAN RESOLVE IS NOT PUBLISHED.

`vmsam-dev-4` hit the same wall earlier: they wanted a log, got none, and I fixed the
STEM DERIVATION and published `ci-report-bindings.tsv` FOR REPORTS ONLY. I fixed the
case in front of me and not the class, and it cost a second agent a wrong verdict.

DEGENERATE STEMS ARE EXCLUDED, NOT GUESSED. Five older `.mkv` carry `UNKNOWN-backfill`,
which matches everything and therefore identifies nothing; a family built from it would
be an invention. They are the genuine coverage limit and they are absent from this file
rather than wrong in it.
"""
import csv, collections, os, sys

LEDGER = '/config/output/KEEP/ci-preservation-ledger.tsv'
BINDINGS = '/config/output/KEEP/ci-report-bindings.tsv'
OUT = '/config/output/KEEP/ci-artefact-families.tsv'


def real(k):
    return bool(k) and len(k) == 16 and all(c in '0123456789abcdef' for c in k)


def build():
    rows = list(csv.DictReader(open(LEDGER), delimiter='\t'))
    fix = {}
    if os.path.exists(BINDINGS):
        for b in csv.DictReader(open(BINDINGS), delimiter='\t'):
            fix[b['report_keep_name']] = b['correct_stem_sha256_16']
    fam = collections.defaultdict(dict)
    for r in rows:
        k = fix.get(r['keep_name'], r['stem_sha256_16'])
        if not real(k):
            continue
        fam[k][r['kind']] = r['keep_name']
    return fam


def undelivered_families():
    """The refusal zone. ITS BINDING IS SELF-EVIDENCING AND NEEDS NO LEDGER.

    A refusal's `.error` record names its artefact outright:

        repair: undelivered state=REFUSED durable=True path=/config/output/undelivered/...

    `vmsam-forensic` pointed out that my families table was scoped to KEEP while the
    refused artefacts live in `undelivered/` -- A ZONE BOUNDARY, not a coverage gap, and
    mine to close because the evidence is in a file I already hold.
    """
    import re, glob
    out = {}
    for rec in glob.glob('/config/output/KEEP/*.error'):
        txt = open(rec, encoding='utf-8', errors='replace').read()
        m = re.search(r'undelivered state=\w+[^\n]*? path=(/config/output/undelivered/\S+)', txt)
        if not m:
            continue
        art = m.group(1)
        out[os.path.basename(rec)] = {'artefact': art,
                                      'exists': os.path.exists(art),
                                      'record': os.path.basename(rec)}
    return out


def write(fam):
    lines = ["# WHICH FILES BELONG TO ONE JOB. Written by ci for consumers who should not",
             "# have to read ci's ledger. Each file is named sha256(ITS OWN source path),",
             "# so a .mkv and its .log share NO part of their names -- the only binding is",
             "# the stem, and until now it lived only in ci-preservation-ledger.tsv.",
             "# Stems that match everything (`UNKNOWN-backfill`) identify nothing and are",
             "# EXCLUDED rather than guessed at.",
             "stem\tmkv\tlog\terror\tmerge_plan_report"]
    n = 0
    for k, v in sorted(fam.items()):
        if not (v.get('mkv') or v.get('error')):
            continue
        lines.append('\t'.join([k, v.get('mkv', '-'), v.get('log', '-'),
                                v.get('error', '-'), v.get('merge_plan_report', '-')]))
        n += 1
    und = undelivered_families()
    if und:
        lines.append('# --- the undelivered (refused) zone: binding taken from the record\'s')
        lines.append('# --- own `undelivered ... path=` line, which names its artefact outright.')
        for rec, v in sorted(und.items()):
            # A REAL PER-ROW KEY, not the sentinel `zone:undelivered`. dev-2's store
            # nests each refusal under `stable_case_key(candidate_path)` -- so the path
            # itself carries a unique identifier and I was throwing it away for a label
            # that covered every row in the zone.
            case = os.path.basename(os.path.dirname(v['artefact'])) or 'unkeyed'
            lines.append('\t'.join(['case:' + case,
                                    os.path.basename(v['artefact']) + ('' if v['exists'] else ' (GONE)'),
                                    '-', rec, '-']))
    # THE FIVE I CANNOT RESOLVE, POINTED AT RATHER THAN IMPORTED.
    # Their stem is `UNKNOWN-backfill`, which matches everything and identifies nothing.
    # `vmsam-forensic` has four of them paired BY INODE IDENTITY AT CAPTURE TIME -- a
    # filesystem fact I did not record and cannot reproduce now -- and the fifth marked
    # UNPROVABLE, which stays unprovable.
    #
    # Copying their values in would make them look like something I derived. A POINTER
    # KEEPS THE BASIS WITH THE CLAIM.
    degen = sorted({r['keep_name'] for r in csv.DictReader(open(LEDGER), delimiter='\t')
                    if r['kind'] == 'mkv' and not real(r['stem_sha256_16'])
                    and not r['keep_name'].startswith('frozen/')})
    if degen:
        lines.append('# --- NOT RESOLVABLE HERE: stem is `UNKNOWN-backfill`, which matches')
        lines.append('# --- everything. vmsam-forensic pairs four of these by INODE IDENTITY')
        lines.append('# --- in KEEP/pairing-ledger.tsv; one is marked UNPROVABLE there.')
        lines.append('# --- Referenced, NOT imported -- the basis belongs with the claim.')
        # AS COMMENTS, NOT ROWS. A row whose key is `UNRESOLVABLE-HERE` is not a row --
        # it is a footnote wearing a table's shape, and my own degenerate-key sweep
        # flagged it within a minute, correctly. A ROW WITH NO KEY IS NOT A ROW.
        for d in degen:
            lines.append('# UNRESOLVABLE HERE: ' + d + '  -- see pairing-ledger.tsv')
    tmp = OUT + '.tmp'
    open(tmp, 'w').write('\n'.join(lines) + '\n')
    os.replace(tmp, OUT)      # atomic: a consumer never sees a half-written table
    return n


if __name__ == '__main__':
    fam = build()
    n = write(fam)
    mk = sum(1 for v in fam.values() if v.get('mkv'))
    wl = sum(1 for v in fam.values() if v.get('mkv') and v.get('log'))
    print(f"  {n} families published to {OUT}")
    print(f"  produced .mkv with a log: {wl} of {mk}")
    sys.exit(0)
