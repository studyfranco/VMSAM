# 015 — Route the outlier-window refusals through the chimeric path, and let it rule

Owner ruling, 2026-09-14. The stage-2 frozen check (`mergeVideo.py:582-594`)
stays UNTOUCHED. The four rare cases it refuses on a single outlier window
while nine of ten windows agree to ~2 ms — **ids 12, 13, 45, 108** — are settled
empirically instead:

## The experiment

When the chimeric repair improvement is ready, run these four files through it
and record what it finds, on BOTH instruments:
- audio: does the chromaprint per-point pass locate any divergent zone?
- images: does the 50-frame pHash validation see any frame difference?

Two outcomes, both informative:
- **No divergence found (audio + images)** → the stage-2 refusal was
  unjustified for this class; THEN the robust-statistic amendment to the frozen
  check becomes an evidence-backed owner decision, not a bet.
- **A real divergence found** (the owner expects 1-2 frames) → the refusal was
  right, and these files are ordinary repair customers.

## The requirement this places on the repair path

The repair must treat **"no difference found" as a first-class verdict**, not
an error and not a crash: a file routed to repair in which the locator finds no
divergent zone must come back saying exactly that. This is the same contract as
TASKS/013 §3 — an empty result must be an explicit verdict, distinct from
"could not measure".

## Context (do not reload elsewhere)

Full analysis: `VMSAM_CORPUS/docs/CORPUS_MISFILING.md`, grep "single outlier
window". Units nuance: the 0.004 constant is seconds-squared (sigma ~63 ms, not
4 ms) — carried there too. These four files stay OUT of the repaired/damaged/
declined denominators until this ticket rules (REPAIR_TARGETING.md's field-on-
the-row doctrine).
