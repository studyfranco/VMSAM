#!/usr/bin/env python3
"""How many of the 315 are repairable at all? A read, not a merge run.

THE QUESTION (Lead, 2026-09-04): a zero produced from a population with no
repairable members is not a pipeline failure, and it is indistinguishable from one
until someone measures the population.

WHY THE OBVIOUS FEATURES DO NOT WORK, MEASURED RATHER THAN ASSUMED:

  SIGN FLIPS ARE USELESS IN MY DATA. dev-1's mispair signature counts sign changes
  in CHROMAPRINT'S ABSOLUTE delays. My sweep reports offsets RELATIVE TO A BASELINE
  it already locked, so both known-measurable files show 0 flips in 29 transitions
  and so would a mispair. Reusing their threshold on my numbers is the reused-control
  trap: the instrument must possess the property the question tests.

  FIDELITY MEDIAN IS NOT A CLASSIFIER. id 31 got a six-piece plan and produced a
  file -- and its median correlation is 0.179, with 2 of 30 probes above 0.5. Any
  coverage-fraction rule discards a file the container demonstrably repaired.

THE FEATURE THAT SEPARATES BOTH KNOWN CASES IS `corr_max`, and it has a mechanism:
two files sharing content must align somewhere, even if they align nowhere else.
Coverage then decides whether a PLAN is possible -- a different question from
whether the pair is the same programme.

    id 31  corr_max 0.9197   plan built, file produced   MEASURABLE
    id 44  corr_max 0.9997   plan built, file produced   MEASURABLE
    id 46  NO PROBES AT ALL  locator: no measurement     MISPAIR

GROUND TRUTH IS n=2 MEASURABLE AND n=1 MISPAIR. That is thin and the threshold is
reported with its sensitivity curve rather than as a line.

THE CENSORING, WHICH IS THE HONEST CORE OF THE ANSWER: `fine_maxlag_s` is per file
and is 3.0 s for 93 of them. id 46's real offsets are +/-180 s. A pair whose true
offset exceeds its own search window CANNOT LOCK, and lands in `baseline-not-found`
looking exactly like a pair with nothing in common. So NO_LOCK_IN_WINDOW is its own
bucket and is NEVER counted as a mispair.
"""
import json, statistics, sys
from collections import Counter

rows = [json.loads(l) for l in open('runs/full-out.jsonl')]
by_id = {x['id']: x for x in rows if isinstance(x, dict) and 'id' in x}

THR = 0.90   # grounded on id 31 (0.9197), the WEAKER of the two known-measurable

buckets, feat = Counter(), {}
for i, x in by_id.items():
    seq = x.get('sequence') or []
    lab = x.get('label')
    # SOME PROBES CARRY A NULL CORRELATION -- a probe that could not be read.
    # Dropping them is right; treating null as 0.0 would invent a measurement.
    cors = [s[2] for s in seq if s[2] is not None]
    cmax = max(cors) if cors else None
    offs = [s[1] for s in seq if s[1] is not None]
    feat[i] = {"label": lab, "n_probes": len(seq), "n_readable": len(cors), "corr_max": cmax,
               "span_ms": (max(offs) - min(offs)) if offs else None,
               "maxlag_s": x.get('fine_maxlag_s')}
    if lab == 'no-shared-language':
        b = 'NO_SHARED_LANGUAGE'          # repair impossible, unrelated reason
    elif not cors:
        b = 'NO_LOCK_IN_WINDOW'           # mispair OR beyond the window -- UNDECIDED
    elif cmax is not None and cmax >= THR:
        b = 'LOOKS_MEASURABLE'
    else:
        b = 'CANNOT_CLASSIFY'             # probes exist, none coherent
    feat[i]['bucket'] = b
    buckets[b] += 1

n = len(by_id)
print(f"POPULATION STRATIFICATION -- {n} files, no merges run\n")
for k, v in buckets.most_common():
    print(f"  {k:22} {v:>4}   {v/n:5.1%}")

print(f"\n=== THRESHOLD SENSITIVITY (corr_max), because one line is not a result ===")
for t in (0.99, 0.97, 0.95, 0.92, 0.90, 0.85, 0.80, 0.70, 0.50):
    m = sum(1 for f in feat.values()
            if f['bucket'] in ('LOOKS_MEASURABLE', 'CANNOT_CLASSIFY')
            and f['corr_max'] is not None and f['corr_max'] >= t)
    flag = '   <- id 31, the weaker known-measurable' if t == 0.92 else ''
    print(f"  corr_max >= {t:.2f}   {m:>4} measurable   {m/n:5.1%}{flag}")

print("\n=== the two known-measurable files land where? ===")
for i in (31, 44, 46):
    f = feat.get(i)
    print(f"  id {i:>3}  {f['bucket']:<20} corr_max={f['corr_max']}  n_probes={f['n_probes']}")

json.dump({"threshold": THR, "buckets": dict(buckets), "n": n,
           "per_id": {str(k): v for k, v in feat.items()}},
          open('runs/population-strata.json', 'w'), indent=1)
print("\nwritten: runs/population-strata.json")
