#!/usr/bin/env python3
"""Read the chromaprint sweep and say what it does and does not establish.

PRIVATE — prints real titles. Never inside the repository.

Includes an internal validation the sweep can FAIL. Folder 103 is the only folder
whose composition is known independently: 26 entries name themselves as another
series and 23 name themselves correctly, and ONE of the 26 (id 237) was confirmed
different by eye. If the sweep is measuring what it claims to, those two
populations must separate. If they do not, the sweep is wrong and the number it
produces is not reportable — that is the point of running the check.
"""

import json
import os
import sys
from collections import Counter, defaultdict

RUNS = "/home/vmsam/src/VMSAM_HELP_AI/ci/runs"
# THE SERIES TOKEN IS NOT IN THIS FILE, AND MUST NOT BE.
#
# It was a module-level constant holding a REAL SERIES TITLE in mixed script, and
# this file is TRACKED -- the owner's hardest rule is no media filename and no
# episode title in any tracked file, ever. I put it here myself and reported it
# against myself; this is the repair.
#
# It is read at run time from the private notes BESIDE the repository, which is the
# established pattern for instance material: reference by PATH, never inline.
# `AGENT.MD` -- "cite the opaque id and write 'details in the private notes beside
# the repository'".
#
# REDACTION IS CORRECT GOING FORWARD AND DOES NOTHING BACKWARD: the value is already
# in the published history and only a rewrite removes it. That is the owner's call.
_SERIES_TOKENS = "/home/vmsam/src/VMSAM_HELP_AI/private/ci-series-tokens.json"


def _series_token(name):
    """Fail LOUDLY if the private store is missing -- a silent empty string would
    make every `token in basename` test FALSE and the sweep would report a clean
    zero over material it never matched. That is a zero from a broken search."""
    import json as _json
    with open(_SERIES_TOKENS, encoding="utf-8") as fh:
        token = _json.load(fh)["tokens"][name]
    if not token:
        raise ValueError(f"series token {name!r} is empty -- refusing to sweep")
    return token


OTOME = _series_token("OTOME")


def bucket(p):
    if p is None:
        return "unreadable"
    for lo, name in ((0.05, "<0.05"), (0.10, "0.05-0.10"), (0.15, "0.10-0.15"),
                     (0.20, "0.15-0.20"), (0.30, "0.20-0.30"), (0.45, "0.30-0.45")):
        if p < lo:
            return name
    return ">=0.45"


def main():
    R = json.load(open(f"{RUNS}/pass1-chromaprint.json"))
    dur = {r["id"]: r for r in json.load(open(f"{RUNS}/sweep-durations.json"))}
    print(f"rows: {len(R)}\n")

    print("=== peak distribution (bit agreement; ~0 chance, 1 identical) ===")
    b = Counter(bucket(r["peak"]) for r in R)
    for k in ("<0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", "0.20-0.30",
              "0.30-0.45", ">=0.45", "unreadable"):
        if b[k]:
            print(f"  {k:>10} : {b[k]:3d}  {'#' * min(60, b[k])}")

    print("\n=== verdicts ===")
    for k, v in Counter(r["verdict"] for r in R).most_common():
        print(f"  {k:<22}: {v}")

    # ---- internal validation on the one folder whose composition is known ----
    print("\n=== INTERNAL VALIDATION — folder 103, known composition ===")
    f103 = [r for r in R if r["folder_id"] == 103]
    grp = defaultdict(list)
    for r in f103:
        named = "names-other-series" if OTOME in os.path.basename(r["file_path"]) \
            else "names-this-series"
        grp[named].append(r)
    ok = True
    for name in ("names-other-series", "names-this-series"):
        rows = grp[name]
        peaks = [r["peak"] for r in rows if r["peak"] is not None]
        if not peaks:
            continue
        vc = Counter(r["verdict"] for r in rows)
        print(f"  {name:<20} n={len(rows):2d}  peak min={min(peaks):.3f} "
              f"max={max(peaks):.3f} median={sorted(peaks)[len(peaks)//2]:.3f}")
        print(f"      verdicts: {dict(vc)}")
    a = [r["peak"] for r in grp["names-other-series"] if r["peak"] is not None]
    c = [r["peak"] for r in grp["names-this-series"] if r["peak"] is not None]
    if a and c:
        if max(a) < min(c):
            print(f"  SEPARATED cleanly: other-series all < {max(a):.3f} "
                  f"< {min(c):.3f} <= this-series")
        else:
            ok = False
            print(f"  NOT SEPARATED: other-series max {max(a):.3f} overlaps "
                  f"this-series min {min(c):.3f}")
            print("  ==> the sweep does not reproduce the one composition known "
                  "independently. Do not report a corpus number from it.")

    # ---- candidates ----
    cand = sorted([r for r in R if r["verdict"] == "candidate-different"],
                  key=lambda r: (r["folder_id"], r["episode_number"]))
    amb = [r for r in R if r["verdict"] == "ambiguous"]
    print(f"\n=== candidates (need eye verification): {len(cand)}  "
          f"| ambiguous: {len(amb)} ===")
    per = Counter(r["folder_id"] for r in cand)
    tot = Counter(r["folder_id"] for r in R)
    for fid, n in per.most_common():
        eps = sorted({r["episode_number"] for r in cand if r["folder_id"] == fid})
        print(f"  folder {fid:>3}: {n:2d} of {tot[fid]:2d} entries | "
              f"{len(eps)} distinct episodes")

    print("\n=== same-content ids, by folder (real sync failures -> exemplars) ===")
    same = [r for r in R if r["verdict"] == "same-content"]
    bysame = defaultdict(list)
    for r in same:
        bysame[r["folder_id"]].append(r)
    for fid in sorted(bysame, key=lambda x: -len(bysame[x])):
        rows = bysame[fid]
        ds = [dur[r["id"]]["delta_s"] for r in rows
              if dur.get(r["id"], {}).get("delta_s") is not None]
        small = [r["id"] for r in rows
                 if dur.get(r["id"], {}).get("delta_s") is not None
                 and 0 < abs(dur[r["id"]]["delta_s"]) <= 30]
        print(f"  folder {fid:>3}: {len(rows):2d} same-content"
              + (f" | delta range {min(ds):+.1f}..{max(ds):+.1f}s" if ds else "")
              + (f" | small-delta ids {small[:8]}" if small else ""))

    json.dump({"candidates": [r["id"] for r in cand],
               "ambiguous": [r["id"] for r in amb],
               "same_content": [r["id"] for r in same],
               "validation_separated": ok},
              open(f"{RUNS}/sweep-summary.json", "w"), indent=1)
    print(f"\nwrote {RUNS}/sweep-summary.json")


if __name__ == "__main__":
    main()
