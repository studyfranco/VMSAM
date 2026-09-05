#!/usr/bin/env python3
"""Cross-check someone else's (candidate, master) pairs against the pipeline's own.

WHY THIS IS WORTH RUNNING: `vmsam-dev-3`'s population answer rests on an EPISODE-TOKEN
MATCHER whose error rate is unmeasured -- its one known error was caught BY EYE, from an
absurd duration ratio. A mispairing between two episodes of similar length leaves no
signature in its table at all.

My pairing comes from a different source entirely: `runs/audio-durations.json`, which
records the master the PIPELINE resolved for each error id -- its own directory layout,
not a parse of filenames. Two independent derivations of one relation, free to disagree.

THREE OUTCOMES, and the third is not a failure:
    AGREE        my index pairs the same candidate with the same master
    DISAGREE     my index pairs that candidate with a DIFFERENT master
    UNRESOLVABLE that candidate is not in my index at all -- ITS OWN BUCKET

POSITIVE CONTROL ON THE FINDER, NOT ON THE DATA. `vmsam-aide`'s finding, from three
instances across three agents: a query built on an unverified assumption about structure
returns a clean, confident, well-formed ZERO -- and zero invites "so there aren't any"
where every other value invites "is that right?". So this file REFUSES TO REPORT until
it has demonstrated it can produce each of the three outcomes on cases whose answer is
known. A checker that has never disagreed with anything has not been shown to be able to.
"""
import json, os, sys

IDX = 'runs/audio-durations.json'
rows = json.load(open(IDX))
by_cand = {r['file_path']: r for r in rows if r.get('file_path')}

def classify(candidate_path, master_path):
    r = by_cand.get(candidate_path)
    if r is None:
        return 'UNRESOLVABLE', None
    mine = r.get('master_path')
    if mine is None:
        return 'UNRESOLVABLE', None
    return ('AGREE' if os.path.normpath(mine) == os.path.normpath(master_path)
            else 'DISAGREE'), mine

def self_test():
    """Must produce all three outcomes on known cases, or the checker is not usable."""
    got = {}
    a, b = rows[0], rows[1]
    got['AGREE'] = classify(a['file_path'], a['master_path'])[0] == 'AGREE'
    # a DELIBERATELY WRONG pair: candidate of a, master of b
    got['DISAGREE'] = classify(a['file_path'], b['master_path'])[0] == 'DISAGREE'
    got['UNRESOLVABLE'] = classify('/nonexistent/candidate.mkv', a['master_path'])[0] == 'UNRESOLVABLE'
    for k, v in got.items():
        print(f"  self-test {k:14} {'ok' if v else 'FAIL'}")
    if a['master_path'] == b['master_path']:
        print("  self-test DISAGREE  INVALID -- rows 0 and 1 share a master; pick different rows")
        return False
    return all(got.values())

if __name__ == '__main__':
    print("SELF-TEST FIRST -- a checker that cannot demonstrate all three outcomes")
    print("does not get to report a count.\n")
    ok = self_test()
    print(f"\n  usable: {ok}")
    if not ok:
        sys.exit(2)
    if len(sys.argv) > 1:
        pairs = json.load(open(sys.argv[1]))
        tally = {}
        dis = []
        for p in pairs:
            v, mine = classify(p.get('candidate') or p.get('file_path'),
                               p.get('master') or p.get('master_path'))
            tally[v] = tally.get(v, 0) + 1
            if v == 'DISAGREE':
                dis.append(p.get('id', p.get('candidate')))
        print("\n RESULT:", tally)
        print("  DISAGREEING ids:", dis)
    else:
        print("\n  no pair file given; self-test only.")
