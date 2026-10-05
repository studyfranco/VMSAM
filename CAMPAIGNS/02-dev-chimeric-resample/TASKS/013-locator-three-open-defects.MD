# 013 — Three open locator defects salvaged from the retired labs

Opened by the Architect, 2026-09-14, from the dev-1/dev-3 lab notes before their
deletion. The three may be one defect seen from three angles — establishing that
is part of the task. Nothing here is fixed; nothing here has a test in the tree.

## 1. ALIAS DEFECT (dev-1, unresolved)

On periodic signal the peak and the second peak of the correlation can tie
exactly, and the tie-break then picks the wrong one. Consequence: a confidently
wrong change point on repetitive content (openings, loops).

## 2. MIS-MODELLING DEFECT (dev-1, unresolved)

A speed *ramp* is rendered as a simple "constant" relation. The refusal the
docstring promises (`speed_relation_suspected`) never fires on the tested band.
Consequence: a file needing resample is treated as splice-only, and the repair
built on it is wrong by construction.

## 3. D-T15 — empty return without an explicit decline (dev-3, confirmed by an independent agent)

`locate_change_points` can return `[]` without declining: "no change found" and
"I could not measure" are the same value. Measured 0/8 valid measurements on a
real 16-pair corpus. This violates the standing rule that `None`/decline means
"could not measure" and is never a compatibility claim.

## Material

- Reproduction pair for §3's family: `VMSAM_CORPUS/corpus-language-divergence/curated-316_jpn-nearmiss/`
  (locator finds change points, audio correlation fails on every tested window).
- Ground-truth trio for §2's discrimination: `VMSAM_CORPUS/corpus-synthetic-ground-truth/rate-vs-splice-trio/`.

## Acceptance

For each defect: a failing test first (target built by hand), then either a fix
or a documented decline path — and for §3, a return type where "nothing found"
and "could not measure" are different values. State what would refute the fix
before running it.
