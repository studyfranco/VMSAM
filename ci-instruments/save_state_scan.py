#!/usr/bin/env python3
"""Gate on committing the ci workspace: does any file carry something that must not travel?

REPORTS FILE AND COUNT, NEVER THE MATCHING TEXT. A scanner that prints what it found
publishes the thing it was written to protect -- into a terminal, a transcript, and
whatever reads them. `REDACT BEFORE THE EXCERPT TRAVELS, NOT AFTER` applies to the
scanner's own output first.

Two independent sources of "sensitive", because they fail differently:
  1. `redact.py` -- the patterns already trusted for artefacts (media roots, filenames,
     episode codes, catalogue ids).
  2. the credential file's ACTUAL VALUES, read at scan time and never printed. A pattern
     cannot know a secret; only the secret knows itself.
"""
import os, re, sys, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import redact

CREDS = '/home/vmsam/src/hook_executor_parameters'


def secret_values():
    """Values from the credential file. Returned for MATCHING ONLY -- never printed,
    never logged, never put in a message. Short values are dropped: a 3-character
    'secret' matches everything and would make every file look poisoned."""
    vals = []
    try:
        for line in open(CREDS, encoding='utf-8', errors='replace'):
            if '=' in line and not line.strip().startswith('#'):
                v = line.split('=', 1)[1].strip().strip('"\'')
                if len(v) >= 8:
                    vals.append(v)
    except OSError:
        return None          # NOT an empty list -- absent is not zero
    return vals


def scan(path, secrets):
    hits = {}
    try:
        txt = open(path, encoding='utf-8', errors='replace').read()
    except (OSError, UnicodeError):
        return {'UNREADABLE': 1}
    for line in txt.splitlines():
        if redact.redact(line) != line:
            hits['redactor-would-change'] = hits.get('redactor-would-change', 0) + 1
    if secrets is None:
        hits['CREDENTIALS-UNCHECKED'] = 1
    else:
        for v in secrets:
            if v in txt:
                hits['CREDENTIAL-VALUE'] = hits.get('CREDENTIAL-VALUE', 0) + 1
    # session links and the API host are named in the standing rule explicitly
    if re.search(r'claude\.ai/code/session|session_[A-Za-z0-9]{12,}', txt):
        hits['session-link'] = len(re.findall(r'session_[A-Za-z0-9]{12,}', txt)) or 1
    return hits


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    secrets = secret_values()
    print(f"  credential file: {'READ' if secrets is not None else 'ABSENT -- files NOT cleared'}"
          f"{f' ({len(secrets)} values, none shown)' if secrets else ''}")
    bad, clean, n = [], 0, 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ('.git', '__pycache__', 'runs')]
        for fn in sorted(filenames):
            if fn.endswith(('.pyc', '.mkv', '.bak')):
                continue
            p = os.path.join(dirpath, fn)
            n += 1
            h = scan(p, secrets)
            if h:
                bad.append((os.path.relpath(p, root), h))
            else:
                clean += 1
    print(f"  {n} file(s) scanned, {clean} clean, {len(bad)} needing a decision\n")
    for rel, h in bad:
        print(f"    {rel:44} " + ', '.join(f"{k}x{v}" for k, v in sorted(h.items())))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
