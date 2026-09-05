#!/usr/bin/env python3
"""Choose 100 files to merge unattended over a weekend.

NOBODY WILL BE WATCHING. That is the whole design constraint: every file must be
chosen so its ARTEFACT answers something without anyone present to steer, and the
reason must travel in the row rather than in my memory. A hundred files picked by id
order would produce a hundred rows and no answers.

Strata are OPEN QUESTIONS, not conveniences. Each file carries why it was chosen.

EXCLUDED BY CONSTRUCTION: forensic's CANNOT HELP set -- 52 files that cannot succeed
whatever the code does. Running them would spend a fifth of the weekend proving
something already established, and their rows would dilute every rate computed later.
They are excluded HERE and still tagged in any row that ever mentions them.
"""
import hashlib, json, os
from collections import defaultdict
from cannot_help import build as ch_build
from out_of_contract import MARKS as OOC

errs = {e['id']: e['file_path'] for e in json.load(open('runs/errors-7b83af4.json'))['incompatible_files']}
CH = ch_build(errs)
fold = lambda i: hashlib.sha256(os.path.dirname(errs[i]).encode()).hexdigest()[:10]

lang = {}
for l in open('runs/decider-candidates.jsonl'):
    d = json.loads(l)
    if 'shared' in d: lang[d['id']] = d

cues = {}
for l in open('runs/cue-census.jsonl'):
    d = json.loads(l)
    if 'tracks' in d:
        srt = [t['cues'] for t in d['tracks'] if t['codec'] in ('subrip', 'srt')]
        cues[d['id']] = min(srt) if srt else None

done = set()
for p in ('runs/series-aa3a311bd0.jsonl', 'runs/merge-queue.jsonl',
          'runs/merge-queue-bee9af4-partial.jsonl', 'runs/targeted-probe.jsonl'):
    try:
        for l in open(p):
            d = json.loads(l)
            if isinstance(d.get('verdict'), str) and isinstance(d.get('id'), int):
                done.add(d['id'])
    except Exception: pass

SEASON = [114,117,120,122,124,127,131,136,142,150,156,161,168,694,695,696,697,698,699]
DECIDERS = [9,126,134,141,268]
PAL = [70,83,91,98]

picked, why = [], {}
def take(i, reason):
    if i in picked or i not in errs or CH.get(i): return
    picked.append(i); why[i] = reason

# 1. the owner's priority, and it contains the frozen control group
for i in SEASON:
    if i not in done:
        take(i, "season aa3a311bd0 -- owner priority; 694-699 are the FROZEN CONTROL: one "
                "shared language, nothing to borrow, forensic predicts >=1 short track")
# 2. the language deciders
for i in DECIDERS:
    take(i, "multi-language master with NO fre -- a borrowed track here cannot be French; "
            "runs on the SHIPPING path, which no parameter sweep can substitute for")
# 3. PAL rate, out of contract
for i in PAL:
    take(i, "PAL ~4.27% rate mismatch. OPEN: does the repair compensate the rate? A low "
            "r_min here means 'rate not handled', NOT 'wrong programme' -- same row, "
            "different finding. verifier_out_of_contract travels on the row")
# 4. the zero-byte subtitle risk, smallest cue counts first
# CAPPED PER FOLDER. Uncapped, this stratum put 37 of 100 into TWO folders -- because
# tiny-cue tracks are a property of particular releases, which is the census finding
# itself. Testing the risk does not require every instance of it: five files from a
# folder exercise the same code path as twenty, and the other fifteen slots buy
# folders I would otherwise never see.
CUE_CAP = 5
cue_per = defaultdict(int)
for i, c in sorted([(i, c) for i, c in cues.items() if c is not None and c <= 10],
                   key=lambda x: x[1]):
    if i in picked or i not in errs or CH.get(i) or i in done: continue
    if cue_per[fold(i)] >= CUE_CAP: continue
    before = len(picked)
    take(i, f"subrip track with only {c} cue(s) at source -- the population where 'every "
            f"cue falls outside the kept pieces' is live. One observed track survived on ONE cue")
    if len(picked) > before: cue_per[fold(i)] += 1
# 5. fill to 100 maximising DISTINCT FOLDERS -- the folder is the independent unit and a
#    hundred files from six folders would be six observations wearing a hundred hats.
per = defaultdict(list)
for i in errs:
    if i not in picked and not CH.get(i) and i not in done: per[fold(i)].append(i)
# PER-FOLDER CAP. My first pass gave ONE FOLDER 23 OF THE 100 -- the exact clustering
# defect I have spent the day insisting on: a hundred files from a few folders are a
# few observations wearing a hundred hats. Capped so the weekend buys FOLDERS, which
# are the independent unit, rather than episodes of the same release.
CAP = 6
cur = defaultdict(int)
for i in picked: cur[fold(i)] += 1
order = sorted(per, key=lambda f: cur[f])
r = 0
while len(picked) < 100 and any(len(per[f]) > r for f in order):
    for f in order:
        if len(picked) >= 100: break
        if cur[f] >= CAP: continue
        if len(per[f]) > r:
            before = len(picked)
            take(per[f][r], f"folder spread (cap {CAP}/folder) -- folders are the independent unit")
            if len(picked) > before: cur[f] += 1
    r += 1

# ORDER OF THE ASK. A queue's tail is what an interruption eats, so the order encodes
# what must survive a truncated weekend. The six-file control goes FIRST: it is the only
# unselected measurement aimed at the merge path on the shipping image with no parameter
# varied, and a peer has frozen a prediction against it. Everything else is recoverable
# by running more files; that one is not substitutable by any sweep.
PRIORITY = [694,695,696,697,698,699] + DECIDERS + PAL
picked.sort(key=lambda i: (PRIORITY.index(i) if i in PRIORITY else 999, i))

fc = defaultdict(int)
for i in picked: fc[fold(i)] += 1
print(f"  SELECTED {len(picked)} files across {len(fc)} of 38 folders")
print(f"  largest single folder contributes {max(fc.values())}")
print(f"  cannot_help excluded: {len([i for i in errs if CH.get(i)])}")
print(f"  already-run excluded: {len(done)}")
json.dump({"ids": picked, "why": {str(k): v for k, v in why.items()},
           "folders": len(fc)}, open('runs/weekend-selection.json', 'w'), indent=1)
print(f"  written to runs/weekend-selection.json")
