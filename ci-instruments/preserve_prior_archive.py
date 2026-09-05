#!/usr/bin/env python3
"""Hard-link the pre-existing error archive, and keep it OUT of this run's ledger.

131 `.error` files sit in the container's own `error_log_archive`, every one with
nlink=1 -- ONE COPY -- from a two-hour window on 2026-08-30, five days before this run.
They are outside both sweep roots, so nothing of mine deletes them and nothing of mine
protects them either. A hard link costs no data blocks, so the only real question was
whether they belong in the run's ledger.

THEY DO NOT, AND PUTTING THEM THERE WOULD HAVE BEEN THE EASY WRONG MOVE.
This run's ledger answers "what did this run produce, and can each artefact be joined to
its siblings". Folding in 131 records from a different run would have moved `error`
coverage from 0/24 to 0/155 and made the denominator describe two populations at once.
    A LEDGER THAT COVERS TWO RUNS ANSWERS QUESTIONS ABOUT NEITHER.

So: separate zone, separate table, provenance stated. The stem is a hash of the source
basename, exactly as the main preserver does it, because these basenames were not
written by me and I have not proved what they contain.
"""
import hashlib, os, sys, datetime

SRC = '/config/output/vmsam_agent/error_log_archive'
DST = '/config/output/KEEP/prior-archive'
TABLE = os.path.join(DST, 'prior-error-archive.tsv')


def main():
    if not os.path.isdir(SRC):
        print(f"  {SRC}: ABSENT -- nothing to do, and that is not the same as zero files")
        return 0
    os.makedirs(DST, exist_ok=True)
    rows, linked, already, failed = [], 0, 0, 0
    for fn in sorted(os.listdir(SRC)):
        if not fn.endswith('.error'):
            continue
        src = os.path.join(SRC, fn)
        try:
            st = os.stat(src)
        except OSError:
            failed += 1
            continue
        stem = hashlib.sha256(fn.encode()).hexdigest()[:16]
        dst = os.path.join(DST, stem + '.error')
        if os.path.exists(dst):
            already += 1
        else:
            try:
                os.link(src, dst)
                linked += 1
            except OSError as e:
                print(f"    LINK FAILED {stem}: {e.__class__.__name__}")
                failed += 1
                continue
        st2 = os.stat(dst)
        rows.append((stem + '.error', 'prior_run_error_log', str(st2.st_ino),
                     str(st2.st_dev), str(st2.st_nlink), str(st2.st_size),
                     datetime.datetime.fromtimestamp(st2.st_mtime, datetime.UTC)
                     .strftime('%Y-%m-%dT%H:%M:%SZ')))
    hdr = ("# PRIOR-RUN ERROR ARCHIVE. NOT part of this run's ledger and NOT part of any\n"
           "# denominator in `status.py`. Source: the container's own error_log_archive,\n"
           "# all files nlink=1 before this ran. Stems are sha256:16 of the SOURCE\n"
           "# BASENAME -- those names were not written by me and are not reproduced here.\n"
           "keep_name\tkind\tinode\tdev\tnlink_at_capture\tsize\tsource_mtime_utc\n")
    tmp = TABLE + '.tmp'
    with open(tmp, 'w') as fh:
        fh.write(hdr)
        for r in rows:
            fh.write('\t'.join(r) + '\n')
    os.replace(tmp, TABLE)
    print(f"  {len(rows)} archived error log(s): {linked} newly linked, {already} already "
          f"present, {failed} failed")
    print(f"  table: {TABLE}")
    nl1 = sum(1 for fn in os.listdir(SRC)
              if fn.endswith('.error') and os.stat(os.path.join(SRC, fn)).st_nlink == 1)
    print(f"  sources still at nlink=1 (unprotected): {nl1}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
