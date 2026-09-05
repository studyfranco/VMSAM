#!/usr/bin/env python3
"""Derive the OUTCOME from a saved decline log. A separate instrument on purpose.

WHY THIS EXISTS: targeted_probe.py's `verdict` field never measured an outcome. It
measured WHICH FILE EXTENSION APPEARED -- .mkv meant merged, anything else meant
"DECLINED". But a subprocess failure and an alignment refusal BOTH write .log.error,
so the field collapsed two opposite outcomes into one word. dev-2 named the third
branch ("failed, not declined") that the field could not express; this reads it back
out of the saved logs, so no container time is spent recovering it.

TOTAL CLASSIFIER: every log lands in a named class, and UNCLASSIFIED is reported BY
NAME rather than folded into a default. A classifier with a silent else-branch is a
classifier that reports its own blind spot as a result.
"""
import glob, json, os, re, sys
from redact import redact_checked, BARE

def classify(t):
    top = [l for l in t.splitlines() if 'Error processing file' in l]
    msg = top[0] if top else ''
    frames = re.findall(r'line \d+, in (\w+)', t)
    ffmpeg_sub = 'This cmd is in error' in t
    repair_failed = 'repair: failed' in t
    if 'This cmd is in error' in msg:
        return 'FAILED_SUBPROCESS', 'top-level exception is a failed command'
    if 'useless to merge' in msg or 'remove_not_compatible_video' in frames:
        # the repair may still have failed BENEATH this -- recorded separately,
        # because "declined because repair failed" and "declined on alignment"
        # are different stories with the same top line.
        return ('DECLINED_AFTER_REPAIR_FAILED' if repair_failed else 'DECLINED_ALIGNMENT',
                'dropped as not-compatible; ' + ('repair failed first' if repair_failed else 'no repair failure in log'))
    return 'UNCLASSIFIED', (msg[:120] or 'no top-level Error processing line')

rows = []
for f in sorted(glob.glob('runs/decline-*.log.redacted')):
    i = int(re.search(r'decline-(\d+)', f).group(1))
    t = open(f, encoding='utf-8', errors='replace').read()
    # SCRUB BEFORE READ. The probe process still running was started under the weak
    # redactor; nothing gets classified, printed or sent until it passes the guard.
    for pat, rep in BARE: t = re.sub(pat, rep, t)
    t = redact_checked(t)
    open(f, 'w').write(t)
    cls, why = classify(t)
    d = [l.strip()[:160] for l in t.splitlines() if 'first_delay_test' in l]
    rows.append({"id": i, "outcome_class": cls, "why": why,
                 "repair_attempted": 'repair:' in t, "repair_failed": 'repair: failed' in t,
                 "ffmpeg_invalid_input": 'Invalid data found when processing input' in t,
                 "n_duration_na": t.count('Duration: N/A'),
                 "first_delay_test": (d[0] if d else None)})
for r in rows:
    print(f"  id {r['id']:<5} {r['outcome_class']:<30} repair_failed={r['repair_failed']!s:<5} N/A={r['n_duration_na']}")
u = [r['id'] for r in rows if r['outcome_class'] == 'UNCLASSIFIED']
print(f"  -- {len(rows)} logs; UNCLASSIFIED: {u if u else 'none'}")
open('runs/decline-classes.json', 'w').write(json.dumps(rows, indent=1))
