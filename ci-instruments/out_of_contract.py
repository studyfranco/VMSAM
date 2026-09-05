#!/usr/bin/env python3
"""Rows where a PASS FROM THE VERIFIER DOES NOT MEAN WHAT A PASS USUALLY MEANS.

Distinct from cannot_help.py, and the distinction is the point:
  cannot_help       the REPAIR cannot succeed -- forensic, from structure and metadata
  out_of_contract   the repair may well succeed; THE VERIFIER CANNOT JUDGE IT

These five sit in the REPAIRABLE population. dev-1 measured every stream pair and
found none reaching 0.70 in any language combination -- its reading is that the
candidate is not that programme. If that is right, the repair aligns one programme to
another, and the verifier compares each produced track against a master track it
SHARES NO CONTENT WITH. A correlation against unrelated audio still returns a number.
The verifier's job is to reject a bad alignment, not to notice the two files are
unrelated: that case was never in its contract.

NOT AN EXCLUSION. Excluding five files from a 315 sweep is a silent cap, and these are
the only observations anyone has of a case nobody understands. They RUN, and the row
says the verdict came from an instrument that cannot detect the failure mode at issue.

A HIGH VERIFIER SCORE ON THESE IS A FINDING, NOT A SUCCESS -- it would be an alignment
scored between two different programmes, which is the measurement that settles whether
the correlator or the classifier is wrong.

Corroboration for the class, not for these five: id 123 was independently flagged by
BOTH forensic (wrong episode, from metadata and structure) and dev-1 (no stream pair
reaching 0.70, from audio fidelity) -- different inputs, not two routes through one.
These five are the silent ones: flagged by the correlator, absent from forensic's list.
"""
# THE REASON ON FOUR OF THESE WAS REFUTED AND HAS BEEN REPLACED, NOT DELETED.
# The original text said "may be a different programme". forensic ran the control the
# original evidence lacked -- a NEGATIVE control -- and the attribution flipped:
#
#     id  91  TEST         peak 0.9590  prominence 0.1100  SMOOTH HUMP, max/min 1.49
#     id  98  TEST         peak 0.9582  prominence 0.1138  SMOOTH HUMP
#     id 289  POSITIVE     peak 0.9590  prominence 0.3622  SMOOTH HUMP, max/min 1.50
#     id 248  NEGATIVE     peak 0.9630  prominence 0.0052  NO SHAPE, ragged
#
# THE SHAPE IS THE ANSWER, NOT THE HEIGHT. A genuinely unrelated file returns 0.0052
# with no curve; the four return a smooth hump peaking on the PAL rate, 21x above the
# negative control and structurally identical to the positive one. THEY ARE THAT
# PROGRAMME, at 4.27 % slow. Both prior measurements were right all along: "no pair
# reaches 0.70" is exactly what a 4.27 % rate mismatch produces.
#
# A MARK WHOSE STATED REASON IS REFUTED IS WORSE THAN NO MARK -- it is a wrong
# explanation in the artefact. But the mark itself is not obviously wrong, for a
# DIFFERENT reason nobody has settled: if the repair does not compensate the rate, the
# produced track is rate-mismatched against the master and the verifier's correlation
# is destroyed the same way. A LOW r_min ON THESE WOULD THEN MEAN "RATE NOT HANDLED",
# NOT "WRONG PROGRAMME" -- two different findings that produce the same row.
RATE = ("PAL rate mismatch, ~4.27 % slow: the content IS this programme (forensic, "
        "rate-compensated peak 0.958-0.959, smooth hump, 21x the negative control). "
        "OPEN: whether the repair compensates the rate. If it does not, r_min is "
        "destroyed by the rate and NOT by wrong content -- same row, different finding.")
CONTENT = ("content mismatch confirmed by two methods sharing no input: forensic "
           "(metadata and structure, median prominence <= 0.007 over 11 trial rates) "
           "and dev-1 (no stream pair reaching 0.70 in any language combination).")
MARKS = {70: RATE, 83: RATE, 91: RATE, 98: RATE, 248: CONTENT}
WHY = ("verifier out of contract: it can reject a bad alignment, it cannot tell you "
       "WHY the correlation is low. A pass here is not a verified repair, and a "
       "failure here does not identify its own cause.")

def mark(i):
    return {"verifier_out_of_contract": True, "out_of_contract_reason": MARKS[i],
            "out_of_contract_note": WHY} if i in MARKS else {}

if __name__ == '__main__':
    import json, os
    errs = {e['id']: e['file_path'] for e in json.load(open('runs/errors-7b83af4.json'))['incompatible_files']}
    from cannot_help import build
    ch = build(errs)
    print(f"  marked ids: {sorted(MARKS)}")
    for i in sorted(MARKS):
        print(f"    id {i:<5} in corpus={i in errs}  cannot_help={ch.get(i)}  "
              f"-> {'REPAIRABLE population, marked not excluded' if not ch.get(i) else 'ALSO cannot_help'}")
