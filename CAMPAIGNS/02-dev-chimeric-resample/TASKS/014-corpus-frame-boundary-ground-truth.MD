# 014 — Annotate the corpus with expected boundaries in master frame numbers

Owner ruling 2026-09-14 (FINDINGS.MD): every corpus case's ground truth carries
the expected divergent-zone boundary as a MASTER VIDEO FRAME NUMBER — the unit
`OWNER_SPEC_FRAME_REPAIR.MD` makes invariant under resample. Audio-only
divergences are bounded in master video frames too.

Scope, in order:
1. Synthetic cases (`VMSAM_CORPUS/corpus-synthetic-ground-truth/`, the built
   B/C/D cases): the exact truth is known by construction — convert each
   recorded cut/step position to the master's frame index and add it to
   `VMSAM_CORPUS/INDEX.md` beside the existing temporal truth. State the frame
   rate used for the conversion on every line.
2. Real pairs (curated-*): annotation requires the pHash frame instrument to
   measure the boundary first — blocked on the frame_compare path being usable
   (see TASKS/013 and the four defects in ARCH_FRAME_ACCURATE.MD). Record each
   as "frame boundary: not yet measured" rather than leaving the field absent.

Acceptance: every INDEX.md entry either carries a frame-number boundary with its
conversion stated, or the explicit "not yet measured" marker — no silent gaps.
