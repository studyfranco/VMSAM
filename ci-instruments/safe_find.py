#!/usr/bin/env python3
"""`find` whose output cannot carry a library path, because I just proved mine can.

I ran `find ... -printf '%h\\n' | sed 's|/config/output|<out>|'` and it printed a show
title and two catalogue ids into a terminal. The sed masked the PREFIX and left the
media path underneath it completely intact. A prefix substitution is not redaction: it
renames the safe part.

    REDACT BEFORE THE EXCERPT TRAVELS, NOT AFTER -- and a shell pipeline is travel.

Every path here goes through `redact.redact_checked`, which RAISES rather than returning
a half-cleaned string, so a path this module cannot make safe is reported as a refusal
instead of being printed hopefully.
"""
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import redact


def safe(p):
    try:
        return redact.redact_checked(p)
    except ValueError:
        return "<REFUSED: could not be safely rendered>"


def walk(root, suffix):
    for dp, dn, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(suffix):
                yield os.path.join(dp, fn)


def main():
    root, suffix = sys.argv[1], sys.argv[2]
    byd = collections.Counter()
    sizes = []
    for p in walk(root, suffix):
        byd[os.path.dirname(p)] += 1
        try:
            sizes.append(os.path.getsize(p))
        except OSError:
            pass
    print(f"  {sum(byd.values())} file(s) matching *{suffix} under the given root\n")
    for d, n in byd.most_common():
        print(f"    {n:5}  {safe(d)}")
    if sizes:
        sizes.sort()
        print(f"\n    size: n={len(sizes)} min={sizes[0]} "
              f"median={sizes[len(sizes)//2]} max={sizes[-1]} "
              f"zero-byte={sum(1 for s in sizes if s == 0)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
