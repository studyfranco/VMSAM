# CAMPAIGN 2 — chimeric resample, restarted 2026-09-14

The roadmap for this round. Written by the Architect under the owner's rulings
of 2026-09-07/14; the Lead Dev sequences it. The pre-reset roadmap is archived
as `CAMPAIGN_ARCHIVE_PRE_RESET.MD` — consult its rulings, follow none of its
plan.

## 1. The need, in one paragraph

VMSAM merges several releases of one episode into one best file. Files it
refuses because no single delay fits (different edits, trimmed edges, PAL/NTSC
rate, weak correlation) pile up in the error tree. This campaign makes the
pipeline repair them **automatically, with generic code** — resample first when
a rate relation exists, then splice — and re-enter them into the normal merge
cycle, indistinguishable from files that never left it.

## 2. The requirement (frozen — `OWNER_SPEC_FRAME_REPAIR.MD` + addendum)

- Repair **only the divergent zone, never the whole comparison zone**, to the
  frame.
- Chromaprint's per-point vector **targets** the zone (~125 ms — the economy
  stage, not a limit); pHash **validates that the 50 frames before and after
  each cut are identical** between master and candidate.
- Boundaries are indexed in **master video frame numbers** — invariant under
  resample; audio-only divergences are bounded in master frames too.
- Resample before locate before splice. Never route through
  `cross_correlation`. `None`/empty means "could not measure", and **"no
  difference found" is a first-class repair verdict** (TASKS/013 §3, 015).

## 3. In place at restart (measured 2026-09-15)

- **Deploy chain verified end to end**: push → CI image → hook → `/health`
  reports the pushed SHA (~30 s after the hook; wait ~10 min after the push).
- **Corpus**: `VMSAM_CORPUS/` (16 GB), engineered + real cases by family, one
  authority `INDEX.md`; the id-level denominator in
  `VMSAM_CORPUS/docs/REPAIR_TARGETING.md` (attempt-splice 222 ·
  attempt-resample 28 · do-not 10 · cannot-help 52 · unknown 3).
- **Three campaign modules exist** (`merge_video_chimeric.py` open;
  `repair`/`resample` closed pending the owner — RULING_20260908). A
  frame-rounding step exists (`adjust_delay_to_frame`, whole function open).
- Rules, checker (35/35), tools with `TOOLS.MD`, and tickets 000-015.

## 4. The end condition — generic, and evaluable

The campaign is DONE when one automated sweep, run by the CI agent **in the
test container carrying the latest pushed code**, over **every file in the
error tree**, produces a row per file such that:

1. Every **ATTEMPT-class** file merges automatically and its product passes
   validation: the target-first comparison, and the 50-frame pHash check at
   every repair boundary.
2. Every **CANNOT-HELP / DO-NOT-class** file is refused **with its named
   reason** carried as a field on its row — never filtered out upstream.
3. **The code is generic**: no corpus id, no per-file constant, no filename
   special-case anywhere in `src/` (`grep` over `src/` for corpus/error ids
   must return nothing; tunables live in `config.ini`).
4. A person — the owner — has looked at the sweep table and said yes. This
   conjunction is irreducible by design.

Rows, not summaries: the three numbers (repaired / damaged / declined) are
computed over the blessed denominator, and a file with no row is a sweep
defect, not a success.

**Both parameters are ruled (owner, 2026-09-15):**
- *Denominator*: the content of `/errors` **at the instant of the final
  validation** — deliberately not frozen, so the owner may add test files at
  any time. Classified by the REPAIR_TARGETING method re-run at that date.
- *Validation of "passes"*: the **Forensic agent builds the expected file by
  hand** — with its own vision and its own tools, built from its own
  calculations, **never with the pipeline under test** — and the software's
  product is compared against that hand-built target. Bit-exactness is not
  required. The boundary check (pHash window) uses a length and a similarity
  threshold **calibrated by TASKS/016** — the 50-frame figure was a ~2 s
  placeholder, and the owner has delegated the real numbers to measurement.

## 5. The stages

| # | what | proves | carrier |
|---|---|---|---|
| 0 | Reread TASKS/005-010 against §2; rewrite what contradicts it | the specs and the requirement agree | Lead + Architect |
| 1 | Fix the frame-accuracy blockers: the four defects of `ARCH_FRAME_ACCURATE.MD` + the three locator defects | pHash path usable; empty ≠ undecided | TASKS/013 |
| 2 | Annotate corpus ground truth in master frame numbers | every case has a checkable expected boundary | TASKS/014 |
| 3 | Repair chain complete with "no difference found" verdict; run the four outlier-window ids through it | the chain rules on the stage-2 question | TASKS/015 |
| 4 | Corpus sweep in the test container: every corpus case through `POST /fusion`, product vs ground truth | generic code on known truth | CI agent |
| 5 | Error-tree sweep (§4 end condition) and the three numbers | the campaign's claim | CI + Forensic |

Stages 1-3 can interleave; 4 gates 5. Every stage lands through the normal
promotion cycle (dev proves → Lead validates → push → container).

## 6. Standing constraints (pointers, not restatements)

Write zones: `WRITE_ZONES.MD` (chimeric open; repair/resample await the owner).
Findings synthesis: Architect + Lead only, `FINDINGS.MD`. Privacy: opaque ids
everywhere tracked; output paths are carriers. Flags: `config.ini
[features]`, off by default. Instruments and their refuted alternatives:
`VMSAM_CORPUS/docs/`.
