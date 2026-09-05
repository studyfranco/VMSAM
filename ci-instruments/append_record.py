#!/usr/bin/env python3
"""Append a record to a run file with a timestamp READ, never typed, and never local.

TWO records in the current run file claim times AFTER the file's own last write:
`2026-09-05T00:37Z` and `2026-09-05T23:10Z`, against a last write of
`2026-09-04T23:26:15Z`. Both are impossible for lines inside that file.

The machine is CEST (+0200) and I typed the LOCAL clock with a `Z` on it:
  * 00:37 local on 09-05  ->  22:37Z on 09-04   (time carried over, tz dropped)
  * 01:10 local on 09-05  ->  23:10Z on 09-04   (TIME right, DATE rolled with local)

That second one is the nastier shape: the hour is correct UTC and only the DATE is
wrong, so it passes every eyeball check and lands the record 23 hours in the future.

    A TIMESTAMP I TYPE IS A MEASUREMENT OF NOTHING.

ATOMICITY: the live runner appends to the same file. A single `write()` to an O_APPEND
fd is atomic only up to PIPE_BUF (4096 bytes), so a long record can interleave with the
runner's and corrupt both lines. This refuses to write a record that cannot be written
atomically rather than hoping.
"""
import json, os, sys, datetime

PIPE_BUF = 4096


def append(path, rec):
    rec = dict(rec)
    rec['at_utc'] = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
    line = (json.dumps(rec, ensure_ascii=False) + '\n').encode('utf-8')
    if len(line) > PIPE_BUF:
        raise ValueError(f"record is {len(line)} bytes; over PIPE_BUF ({PIPE_BUF}) it "
                         f"can interleave with the live runner's append and corrupt "
                         f"both lines. Split it or shorten it -- do not widen this.")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)
    return rec['at_utc'], len(line)


if __name__ == '__main__':
    p = sys.argv[1]
    rec = json.load(sys.stdin)
    t, n = append(p, rec)
    print(f"  appended at {t}, {n} bytes (atomic: {n} <= {PIPE_BUF})")
