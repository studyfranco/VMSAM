# ARCHITECT RULING — preconditions are probed at admission, not at delivery
# Raised by the Lead 2026-09-15 (dev-frame defect-4 review). DRAFT in the
# Architect's lab; travels to FINDINGS on the next authorised governance gesture.
# The CODE it directs is src/ open-zone work and needs no governance gesture.

## The observed shape

`assemble_on_master_timeline` probes the master's `FrameRate_Mode` /
`FrameRate_Original` at the END of the function — after the mux completed,
after verification ran, after `out_path` carries a delivered-looking name. The
refusal therefore needs the whole `mark_output` + REFUSED-rename +
`partial_assembly` machinery, whose only purpose is to un-say what the
filesystem already said.

## The ruling, in one line with its consequence

**A precondition is probed at admission, before any work begins; a refusal must
cost a probe, not a mux.** The entire REFUSED-rename apparatus for this gate is
scaffolding that the placement made necessary — move the probe to the function's
entry and that scaffolding is not "handled better", it is DELETED for this path.
`FrameRate_Mode` is a property of the INPUT: nothing about assembling makes it
more knowable at the end than at the start.

## Why this composes with the F1 contract, which is the deeper reason

Under `RULING_20260915_FRAME_INDEXED_BOUNDARY.MD`, the segments payload that
reaches the assembler already carries a MANDATORY exact-rational `grid` — and a
producer that cannot state the grid has already DECLINED upstream, at the
locator. So an unmeasured grid **cannot legitimately reach the assembler at
all** once F1 is implemented end to end. The assembler's check is then input
validation at entry ("is this payload well-formed?"), which is an admission
probe by definition. The post-mux gate becomes dead-by-construction; at most a
cheap entry assertion remains as defence in depth. Do not grow the late gate —
replace it.

## The empty-string finding is contract rule 6, and it generalises

The guard tested `frame_rate_original is None` while the field can arrive as
`""`. Rule 6 names the class: **"unmeasured" is a property of the VALUE, not of
its absence** — None, empty, blank, and anything unparseable as a positive
exact rational are all the same answer, "I could not measure". The durable fix
is one normaliser, used everywhere a rate enters: parse to an exact rational or
return None — so every consumer downstream faces exactly two cases, and the
guard's predicate collapses to `grid is None`. Ten call sites each doing their
own string-shape reasoning is how the next `""` gets through.

## Scope

Directed at the seat's current unstaged work via the Lead: (1) normaliser +
predicate on the value; (2) probe moved to admission; (3) the REFUSED-rename
path for this gate removed, not mirrored. Lands on the NORMAL path at exit 0.
The transitional case — assembler called with a legacy no-grid payload before
the locator implements F1 — DECLINES at entry with the same named reason.
