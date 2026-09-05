#!/usr/bin/env python3
"""Which commits have a PRODUCED FILE behind them, and which do not.

THE OWNER'S STANDARD: a modification is not valid until the test container, on the
latest image, has produced a file -- and that file must then be compared and validated.
The campaign discovered ten promoted commits with nothing in the container behind them
and nobody noticed for a day. That was not cheating; it was an undefined standard.

THREE STATES, NAMED PER COMMIT, NEVER A COUNT. A count cannot answer "is this commit
one of them", which is the question anyone actually has -- the same defect as a report
giving 'folder 103 (26)' instead of the 26 ids.

    PRODUCED_ROW    the commit is in an image that produced at least one artefact row
    LOADED_NO_ROW   the container ran that image; no row carries it
    NEVER_LOADED    no image the container has loaded contains this commit

LOADED_NO_ROW IS NOT A FAILURE. It means the evidence does not exist YET. Reporting it
as absence-of-evidence rather than evidence-of-absence is the whole point of separating
it from NEVER_LOADED.

This does not validate anything. It says whether the container was ever asked.
"""
import glob, json, os, subprocess, sys

CLONE = '/home/vmsam/src/VMSAM_WIP/ci'

def images_with_rows():
    """image sha -> number of rows carrying it, from every run file."""
    out = {}
    for p in glob.glob('runs/*.jsonl'):
        try:
            for line in open(p):
                d = json.loads(line)
                img = d.get('image_git_commit')
                if img and d.get('verdict'):
                    out[img] = out.get(img, 0) + 1
        except Exception:
            pass
    return out

def images_loaded():
    """Every image the container has been observed on, from the deploy record."""
    seen = set()
    for p in glob.glob('runs/DEPLOY-*.md'):
        base = os.path.basename(p)
        part = base.split('-')[1] if '-' in base else ''
        if part: seen.add(part)
    for img in images_with_rows():
        seen.add(img)
    return seen

def image_contains_commit(commit, image):
    """Is `commit` an ancestor of `image` -- i.e. did that image carry this change?

    THE PARAMETERS USED TO BE NAMED (img, commit) AND CALLED AS (commit, image), so
    the two were swapped inside and every answer was ancestry BACKWARDS. It reported
    every commit as PRODUCED_ROW, including one produced "via" its own ancestor, which
    is the only reason I looked: the output was absurd rather than merely wrong. A
    parameter-order bug returns confident answers of the right TYPE, so nothing but a
    control catches it. Renamed so the call site reads as the question.
    """
    try:
        r = subprocess.run(['git', '-C', CLONE, 'merge-base', '--is-ancestor', commit, image],
                           capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return None


def _self_check():
    """Ancestry is directional; a checker that cannot tell the directions apart is
    worse than none. Known-true and known-false, from this repo, before any use."""
    older, newer = 'e139e1b', '7527ee6'
    ok = (image_contains_commit(older, newer) is True and
          image_contains_commit(newer, older) is False)
    if not ok:
        raise SystemExit("  ANCESTRY SELF-CHECK FAILED -- refusing to report")
    return True

if __name__ == '__main__':
    commits = sys.argv[1:]
    if not commits:
        r = subprocess.run(['git', '-C', CLONE, 'log', '--format=%h', '-20', 'origin/dev-AI'],
                           capture_output=True, timeout=30)
        commits = r.stdout.decode().split()
    _self_check()
    rows = images_with_rows()
    loaded = images_loaded()
    print(f"  images that produced rows: "
          f"{ {k[:12]: v for k, v in rows.items()} }")
    print(f"  images observed loaded    : {sorted(x[:12] for x in loaded)}")
    print()
    print("  commit    state            detail")
    tally = {}
    for c in commits:
        with_row = [i for i in rows if image_contains_commit(c, i)]
        in_loaded = [i for i in loaded if image_contains_commit(c, i)]
        if with_row:
            state, detail = 'PRODUCED_ROW', f"{sum(rows[i] for i in with_row)} row(s) via {with_row[0][:12]}"
        elif in_loaded:
            state, detail = 'LOADED_NO_ROW', f"in {in_loaded[0][:12]}; no row yet -- evidence absent, not negative"
        else:
            state, detail = 'NEVER_LOADED', 'no image the container loaded contains it'
        tally[state] = tally.get(state, 0) + 1
        print(f"  {c:<9} {state:<16} {detail}")
    print()
    print(f"  {tally}")
    sys.exit(1 if tally.get('NEVER_LOADED') else 0)
