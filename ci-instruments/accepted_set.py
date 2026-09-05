#!/usr/bin/env python3
"""The accepted set across run files -- refusing to aggregate incompatible vocabularies.

`vmsam-ci-pair` asked for "the ids your ledgers record as FULL_LENGTH". The obvious
implementation is to scan `runs/*.jsonl` for `verdict == 'FULL_LENGTH'`. That is wrong,
and quietly:

    pair-sweep-trace.jsonl   AMBIGUOUS . CONTENT PRESENT . WRONG PAIR
    probe-trace.jsonl        CARRIES x1.001001 . DOES NOT CARRY x1.042709
    merge-queue.jsonl        FULL_LENGTH . DECLINED . COULD_NOT_MEASURE

THE FIELD NAME IS SHARED AND THE QUESTION IS NOT. id 33 reads `CONTENT PRESENT`,
`CARRIES x1.001001` and `COULD_NOT_MEASURE` in three files -- three different questions,
not three answers to one. Scanning for one token across all of them merges a content
check, a framerate probe and a delivery outcome into a single count that looks fine.

    A SHARED FIELD NAME IS NOT A SHARED QUESTION.

So: classify each FILE by whether its whole verdict vocabulary is a subset of the
delivery vocabulary, and aggregate only within that class. A file using an unknown token
is reported as OTHER and excluded -- never silently included because one of its rows
happened to match.

AND ACCEPTED IS A PROPERTY OF (id, build), NOT OF id. Four ids are DECLINED under one
build and FULL_LENGTH under another. Collapsing them to one row per id asserts something
no run measured, so disagreements are reported separately and never resolved by picking
the newer file -- `looks better` is not a measurement.
"""
import collections, glob, json, os, sys

DELIVERY = {'FULL_LENGTH', 'DECLINED', 'COULD_NOT_MEASURE', 'TRUNCATED'}


def classify(paths):
    delivery, other, unreadable = [], [], []
    for p in paths:
        vocab = set()
        try:
            for l in open(p):
                if not l.strip():
                    continue
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                if 'id' in r and 'verdict' in r:
                    vocab.add(r['verdict'])
        except OSError:
            unreadable.append(p)
            continue
        if not vocab:
            continue
        (delivery if vocab <= DELIVERY else other).append((p, sorted(vocab)))
    return delivery, other, unreadable


def main():
    paths = [p for p in sorted(glob.glob('runs/*.jsonl')) if '-VOID-' not in p]
    voided = [p for p in sorted(glob.glob('runs/*.jsonl')) if '-VOID-' in p]
    delivery, other, unreadable = classify(paths)
    print(f"  {len(voided)} file(s) excluded by a -VOID- name (a name is not a proof; "
          f"these were withdrawn deliberately)")
    print(f"  {len(delivery)} delivery-vocabulary file(s), {len(other)} other-vocabulary, "
          f"{len(unreadable)} unreadable\n")
    for p, v in other:
        print(f"    EXCLUDED (other question): {os.path.basename(p):42} {v[:3]}")
    byid = collections.defaultdict(list)
    for p, _ in delivery:
        for l in open(p):
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if 'id' in r and 'verdict' in r:
                byid[r['id']].append((os.path.basename(p), r['verdict']))
    accepted = sorted(i for i, v in byid.items() if any(x[1] == 'FULL_LENGTH' for x in v))
    disagree = {i: v for i, v in byid.items() if len({x[1] for x in v}) > 1}
    print(f"\n  ids with a delivery verdict : {len(byid)}")
    print(f"  ids ever FULL_LENGTH        : {len(accepted)}")
    print(f"  ids DISAGREEING across runs : {len(disagree)}"
          f"  <- NOT resolved here, on purpose")
    for i, v in sorted(disagree.items()):
        print(f"    id {i:>4}: " + ' | '.join(f"{a.replace('.jsonl','')}={b}" for a, b in v))
    print(f"\n  accepted set:\n    {accepted}")
    print("\n  IMAGE: no row in any run file carries a per-row image read, so every "
          "file's\n  label is a startup cache and is true only if no recreate happened "
          "mid-run.\n  Verified for one run only -- see row_provenance.py. "
          "`targeted-probe.jsonl`\n  carries TWO labels, so that file demonstrably "
          "spans two builds.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
