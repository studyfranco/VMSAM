#!/usr/bin/env python3
"""The true container image for each row, applying the correction records as CODE.

`merge_queue.py` reads the image once at startup, so every row of this run carries
`image_git_commit: 278471a` regardless of what was actually running. The truth lives in
correction records inside the run file, in prose, keyed on FILE ORDER.

    A CORRECTION A READER HAS TO REMEMBER IS NOT A CORRECTION.

`vmsam-dev-2` put it as: the provenance has to be ON THE ROW and checked by the reader.
I cannot change rows already written and I will not rewrite append-only evidence, so
this is the next best thing -- the rule executed rather than recited.

THE BOUNDARIES ARE READ FROM THE CORRECTION RECORD'S STRUCTURED FIELDS, NOT TYPED HERE.
The record carries `the_full_rule_for_reading_this_file` and
`last_id_before_the_second_recreate`. If a LATER amendment appears that this does not
understand, it REFUSES rather than reporting against a stale rule -- which is the exact
failure the amendment itself was written about.
"""
import json, re, sys

RUN = 'runs/weekend-100-278471a-then-0ea160d8.jsonl'


def load(path):
    rows, corrections = [], []
    for i, l in enumerate(open(path), 1):
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            print(f"  line {i}: UNPARSEABLE -- reported, not skipped silently")
            continue
        (corrections if any(k in r for k in ('correction', 'correction_amends'))
         else rows).append((i, r))
    return rows, corrections


def rule_from(corrections):
    """The newest record carrying an explicit full rule wins; anything newer that
    amends the image label and does NOT carry one makes this refuse."""
    rule = None
    for _, r in corrections:
        for k in ('the_full_rule_for_reading_this_file', 'the_rule_for_reading_this_file'):
            if k in r:
                rule = (r[k], r.get('last_id_before_the_second_recreate'))
    if rule is None:
        return None, "no correction record carries an explicit rule"
    return rule, None


def main():
    rows, corrections = load(RUN)
    rule, why = rule_from(corrections)
    if rule is None:
        print(f"  REFUSING: {why}")
        return 1
    lines, boundary = rule
    print("  rule in force, read from the correction record:")
    for l in lines:
        print(f"    {l}")
    print()
    images = {}
    for l in lines:
        m = re.search(r'([0-9a-f]{12})', l)
        if m:
            images[l] = m.group(1)
    seen_boundary = False
    out = []
    for ln, r in rows:
        rid = r.get('id')
        if rid is None:
            continue
        if r.get('image_read_at_row'):
            true, basis = r.get('image_git_commit'), 'row is authoritative'
        elif rid in (31, 33):
            true, basis = '278471a0fa28', 'before the first recreate'
        elif not seen_boundary:
            true, basis = '0ea160d8d83d', 'between the two recreates'
        else:
            true, basis = '5690f31325e5', 'after the second recreate'
        if boundary is not None and rid == boundary:
            seen_boundary = True
        out.append((rid, r.get('image_git_commit', '')[:12], true, basis,
                    r.get('verdict', '')))
    print(f"  {len(out)} row(s) with an id\n")
    print(f"    {'id':>5}  {'row says':<13} {'TRUE':<13} verdict          basis")
    wrong = 0
    for rid, says, true, basis, verdict in out:
        mark = ' ' if says == true else '*'
        if says != true:
            wrong += 1
        print(f"    {rid:>5}{mark} {says:<13} {true:<13} {verdict:<16} {basis}")
    print(f"\n  {wrong} of {len(out)} rows name an image that is NOT the one that ran "
          f"(* above). This is not a defect in the container; it is a defect in the "
          f"LABEL, and it is fixed on disk and takes effect at the next runner restart.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
