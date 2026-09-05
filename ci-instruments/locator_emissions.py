#!/usr/bin/env python3
"""Read the locator's own emission out of an artefact, and check it against itself.

`vmsam-dev-1`'s `_emit` reaches production at `dev=False` as of image 5690f313 -- the
first time this quantity has ever left the module. Before that it was computed three
times and logged zero times.

THE FIELD I FIRST PARSED WAS A LEAKED LOOP VARIABLE. `offset_ms` was assigned inside
`for position, run in enumerate(runs)` and emitted AFTER the loop, so it held the LAST
segment's mean and nothing was named for it. My five records were five confirmations of
an accident, not a sample. dev-1 replaced it:

    offset_first_ms · offset_last_ms · offset_max_abs_ms · offset_monotone
    offset_ms=  ALIAS FOR ONE IMAGE ONLY, removed once this parser reads offset_last_ms

`offset_monotone` is the one that earns its place: it reports whether comparing
max-magnitude offsets against the pipeline is a LEGITIMATE comparison, without me
reconstructing the segment list from dev-2's plan line -- a quantity depending on two
agents' emissions staying in step is not one I can rely on.

WHAT THIS DELIBERATELY DOES NOT DO: compare the offset against the pipeline's delays.
That comparison needs the pipeline's quantum, which is `int(lengthFile/len(fingerprint)
*1000)` PER CALL -- a property of the call, not the file. A gcd over the delays recovers
it only when they are many and independent; over two values it returned 1000 ms against
a real 124-125, and I nearly published a disagreement against it. THE CROSS-CHECK WAITS
FOR THE PIPELINE TO EMIT THOSE TWO NUMBERS.
"""
import re, glob, json, sys

FIELDS = ('offset_first_ms', 'offset_last_ms', 'offset_max_abs_ms', 'offset_monotone',
          'offset_ms', 'points', 'quantum_ms', 'window_s', 'segments', 'change_points',
          'pairing_fidelity', 'pairing_bar')

def parse(txt):
    out = []
    for line in txt.splitlines():
        if '[change_point_locator]' not in line or '=' not in line:
            continue
        kv = dict(re.findall(r'(\w+)=(-?[\d.]+|true|false)', line))
        if not any(k in kv for k in ('offset_last_ms', 'offset_ms', 'pairing_fidelity')):
            continue
        rec = {k: kv[k] for k in FIELDS if k in kv}
        # THE ALIAS, AND WHICH SOURCE ANSWERED. Absent is not the same as equal.
        if 'offset_last_ms' in rec:
            rec['_offset_source'] = 'offset_last_ms'
        elif 'offset_ms' in rec:
            rec['_offset_source'] = 'offset_ms (ALIAS -- image predates the rename)'
        out.append(rec)
    return out

def internal_check(rec):
    """change_points == segments - 1. ONE EMISSION, ONE AGENT, NO PLAN LINE.

    I FIRST CLAIMED SOMETHING ELSE AND CALLED IT THIS. What I told dev-1 was "internal
    to one emission, needs no second instrument" was `segments` against DEV-2's plan
    piece count -- A CROSS-AGENT COMPARISON, and it holds on 3 of 6 distinct jobs. They
    repeated it back as the check they would keep, and neither of us noticed while both
    were being careful about exactly that.

    THIS ONE IS GENUINELY INTERNAL, AND IT IS STILL NOT AN IDENTITY. dev-1 worked it
    from the filter rather than from my records:

        change_points = [cp for i, cp in enumerate(change_points)
                         if i in kept_runs and (i+1) in kept_runs]

    That counts ADJACENT PAIRS in kept_runs, which equals len(kept_runs)-1 only when
    kept_runs is CONTIGUOUS. A dropped END segment preserves it; A DROPPED MIDDLE
    SEGMENT REMOVES TWO CHANGE POINTS AND ONE SEGMENT and it breaks.

    So a failure here means "a middle segment was dropped", NOT "the emission is
    broken" -- and having that in advance is the difference between reading a result
    and diagnosing one live.
    """
    if 'segments' not in rec or 'change_points' not in rec:
        return None
    segs = int(float(rec['segments']))
    cps = int(float(rec['change_points']))
    return {'segments': segs, 'change_points': cps, 'holds': cps == segs - 1,
            'if_it_fails': 'a MIDDLE segment was dropped -- not an emission defect'}


def vs_plan(txt, rec):
    """segments against dev-2's candidate-piece count. A CROSS-AGENT COMPARISON.

    Reported separately and never as self-consistency. 3 of 6 distinct jobs match;
    3 are short by exactly one. dev-1's leading-gap hypothesis is REFUTED -- two of the
    three short jobs have no leading gap, and one job WITH a leading gap matches.
    WHICH SURVIVING SEGMENTS BECOME CANDIDATE PIECES IS NOT ESTABLISHED.
    """
    m = re.search(r'repair: plan [^\n]*?pieces=([^\n]*)', txt)
    if not m or 'segments' not in rec:
        return None
    toks = m.group(1).split()
    cand = sum(1 for t in toks if t.startswith('c'))
    segs = int(float(rec['segments']))
    return {'plan_candidate_pieces': cand, 'segments': segs, 'delta': segs - cand,
            'leading_gap': bool(toks and toks[0].startswith('m'))}

if __name__ == '__main__':
    # ONE ENTRY PER JOB, NOT PER RECORD. My own redacted copies live beside the
    # container's records, so a raw glob counted two jobs as four. `vmsam-forensic`
    # caught it: the instrument's output feeding back into the instrument's input.
    import census_population
    files = [p for p, _k, _how, _der in census_population.records()]
    rows, agree, disagree, nocheck = [], 0, 0, 0
    for f in files:
        txt = open(f, encoding='utf-8', errors='replace').read()
        for rec in parse(txt):
            sc = internal_check(rec)
            vp = vs_plan(txt, rec)
            rows.append({'record': f.split('/')[-1], **rec,
                         'internal_check': sc, 'vs_plan': vp})
            if sc is None: nocheck += 1
            elif sc['holds']: agree += 1
            else: disagree += 1
    print(f"{len(files)} records scanned, {len(rows)} emission(s)\n")
    for r in rows:
        src = r.get('_offset_source', '-')
        o = r.get('offset_last_ms', r.get('offset_ms', '-'))
        mono = r.get('offset_monotone', 'NOT EMITTED')
        print(f"  {r['record'][:24]:26} offset_last={o:>10}  max_abs={r.get('offset_max_abs_ms','-'):>10} "
              f" monotone={mono:<12} segs={r.get('segments','-')}/{r.get('change_points','-')}")
        print(f"  {'':26} source: {src}")
    print(f"\n  INTERNAL (change_points == segments-1)  holds {agree}  FAILS {disagree}  n/a {nocheck}")
    print(f"    a failure here means A MIDDLE SEGMENT WAS DROPPED, not a broken emission")
    print(f"  parser reads offset_last_ms on: {sum(1 for r in rows if 'offset_last_ms' in r)} of {len(rows)}")
    json.dump(rows, open('runs/locator-emissions.json', 'w'), indent=1)
    print("  written runs/locator-emissions.json")
    sys.exit(1 if disagree else 0)
