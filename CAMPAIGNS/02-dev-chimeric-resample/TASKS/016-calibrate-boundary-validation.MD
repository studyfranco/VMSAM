# 016 — Calibrate the boundary validation: window length, similarity floor, or nearest-cut

Owner delegation, 2026-09-15: "50 frames" was arbitrary (~2 s). The owner does
NOT know how many frames the check needs, nor what minimum similarity to
require, nor whether the window should instead extend to the nearest SCENE CHANGE
(owner's clarification 2026-09-15: "changement de frame" means scene cut,
not the adjacent frame) in both directions. He delegated the numbers to
measurement — Architect and agents decide, with evidence.

## The question, split in three

1. **Window length.** How many frames on each side of a repair boundary must
   match before we call the boundary clean? Measure on corpus cases with known
   frame ground truth (TASKS/014): what is the shortest window that still
   catches every misplaced-boundary case the corpus contains, and what does
   each extra frame cost in sweep time?
2. **Similarity floor.** pHash gives a Hamming distance per frame pair. The
   existing calibration measured separation ~47x with 0/24 false positives at
   threshold 12 (see FINDINGS / the campaign journal). Re-derive the floor ON
   REPAIR BOUNDARIES specifically: encode artefacts at a splice point may
   depress similarity on genuinely identical frames. Freeze the threshold
   BEFORE the validation run, from a calibration set that is not the test set.
3. **The alternative: nearest-scene-change anchoring.** Instead of a fixed
   window, extend the check to the nearest scene change on each side —
   the window becomes content-defined. Evaluate: does it beat the fixed window
   on the corpus (fewer false accepts/rejects), and what does it cost when the
   nearest cut is minutes away (a bound is needed)?

## Method constraints

- Target-first: for each candidate rule, state the expected accept/reject
  outcome per corpus case BEFORE running it, and what would refute the rule.
- Both error directions measured: a rule must reject a deliberately
  mis-shifted boundary (positive control) AND accept a known-clean one.
- The chosen rule lands as config values in `config.ini` (never hardcoded),
  and the numbers with their calibration commands go to FINDINGS via the Lead.

## Until this task rules

The 50-frame / threshold-12 pair stands as the placeholder wherever a boundary
check is needed; every use must cite it as "placeholder pending TASKS/016".
