# 003 — Re-establish the failure-family proportions from zero

**Agent:** `vmsam-forensic`. **State:** open. **First report of the campaign.**

## Goal

`CAMPAIGN.MD` carries a table of five families with shares — 57 / 18 / 11 / 10 / 2 —
and says in its own words that they are campaign 1's, unverified, over a corpus of
315 that has since grown. **Until this report lands, any figure taken from that
table into another document is a figure with no measurement behind it.**

Re-diagnose from zero. Do not start from campaign 1's classification, do not read
its per-file verdicts before forming your own, and say so if you did.

Deliver into `VMSAM_HELP_AI/forensic/`:

1. the corpus as it is **today** — `GET /errors` returned **315** entries over
   **38** folder ids on 2026-09-03; an identical total to campaign 1 does not
   prove an identical set, so check the ids;
2. a family per error id, with the evidence that put it there;
3. the proportions, with what you could **not** classify counted separately and
   never folded into a family;
4. for each family, **two or three exemplar ids** the developers can work on.

## Standing rules

- **Ids only, never filenames.** Reports live outside the repository, but the
  habit is what keeps the repository clean.
- **Never code, never repository files.** Your output is what the others reason
  from.
- **Keep doing it as new errors arrive.** The corpus grows.
- `unclassifiable` and *the instrument did not run* are **different answers**.
  `RESTART.MD` §3.3: 31 of 45 `unverifiable` records in campaign 1 were an ffmpeg
  that was not configured, written where "this file is unverifiable" is read.

## What changed

No repository code and no repository files beyond this one and `TASKS/004`. Deliverables are
in the private notes beside the repository.

## What was measured

**Corpus, today: 315 error files, 38 folder ids.** Identical to campaign 1's set — 315/315,
symmetric difference empty, zero folder or episode mismatches. **It has not grown.**
*Hedge: this proves the record set is identical, not that the media is unchanged.*

**Reproducing command** (from `VMSAM_HELP_AI/forensic/`, details in the private notes):
`scripts/reproduce_table.sh` — corpus → parse the `.log.error` beside each file → ffprobe both
sides → classify → PAL split → later refinements → print. `--from-scratch` re-fetches and
re-probes (~25 min); without it, seconds from stored intermediates.

| family | n | share | campaign 1 (unverified) |
| :--- | ---: | ---: | ---: |
| different cuts | 188 | 59.7 % | 57 % |
| trimmed edges | 32 | 10.2 % | 18 % |
| content mismatch — wrong series | 27 | 8.6 % | *no row* |
| speed mismatch — PAL 25/23.976 | 20 | 6.3 % | *11 % for speed* |
| inter-track divergence — master side | 11 | 3.5 % | *no row* |
| speed mismatch — NTSC 1000/1001 | 8 | 2.5 % | |
| no shared language | 7 | 2.2 % | 2 % |
| per-language divergence — constant | 4 | 1.3 % | *no row* |
| defective master (audio truncated) | 3 | 1.0 % | *no row* |
| constant offset — refused on a short-file quantum | 3 | 1.0 % | *no row* |
| content mismatch — wrong episode | 2 | 0.6 % | *no row* |
| per-language divergence — one language steps | 2 | 0.6 % | *no row* |
| weak correlation | 2 | 0.6 % | 10 % |
| *instrument did not run* | 2 | 0.6 % | |
| **candidate substantially incomplete** | 1 | 0.3 % | *no row* |
| constant offset — refused anyway | 1 | 0.3 % | |
| *unclassifiable* | 1 | 0.3 % | |
| no shared language — TAGS LIE | 1 | 0.3 % | *no row* |

**Unexplained: 5 of 315 (1.6 %)** — 112, 693 (instrument did not run); 196, 307 (partial
overlap); 45 (unclassifiable).

**FINDING for the owner — `CAMPAIGN.MD`'s table is wrong and it is frozen, so this is a
finding and not an edit. Both dev agents are steering by it.**

1. **`different cuts` is 188 (59.7 %), not 57 %.** Direction confirmed, magnitude higher.
2. **`content mismatch — wrong series` is 27 files (8.6 %) and has no row in that table.**
   No repair is meaningful on them.
3. **`weak correlation` was not a family.** It began at 53 files and ends at **2**. The other
   51 were four unrelated causes: 20 PAL speed relations, 27 wrong-series, 2 wrong-episode.
   `CAMPAIGN.MD` gives that row 10 %. "No confident peak" is a symptom.
4. **Speed is both relations**, 20 PAL and 8 NTSC. That row's prose says PAL while its rate
   ("~1 s per half-hour") is the NTSC figure.
5. **`trimmed edges` at 10.2 % is a floor and neither figure is defensible** — see blocked.

### Detector rewritten in quanta — two of my own findings WITHDRAWN

My change-point threshold was 150 ms, which sits **between one quantum (124 ms) and two (248)**.
On a file whose offsets alternate between those, the 2-quantum probes read deviant and the
1-quantum ones clean; because they alternate the deviants come out scattered and bounded on both
sides — **exactly my island signature**. The threshold measured the quantisation, not the media.
Rewritten to express deviation in **quanta per file**, plus `vmsam-ci`'s rule that a deviant
probe must reach half the file's own median prominence. Re-run over the 32 trimmed edges:
**24 STRUCTURE, 8 rounding only.**

- **id 74 — "ISLAND IN TAIL" WITHDRAWN.** Quanta {0, 1, 2}. What survives is a one-quantum step
  at t ≈ 880 s, then rounding. **`vmsam-ci`'s class B has ONE known member, id 108, not two.**
- **id 97 — "TAIL STEP" WITHDRAWN.** Quanta {0, 2}.
- **Twelve further files flipped to STRUCTURE and I do NOT claim them.** The sweeps differ in
  scope — tail-only versus whole-file. Checked: **0 of 12 runs reach the tail; 12 of 12 sit
  outside it.** Structure the tail sweep never looked at, not a threshold improvement.

**Net: the rewrite removed two findings and added none.**

**51 of 51 of my non-trivial deviations land on integer quanta — expected for me, an artefact
signature for ci.** My offsets are `k·spa − pb·spb` with integer indices, quantised by
construction; ci's FFT lags are continuous at 0.125 ms, so identical clustering means the
opposite thing. Both of us read each other's depth tables.

### Two files moved on `vmsam-dev-2`'s evidence

- **id 691 → defective master.** dev-2 measured a 53 % short candidate. The master's container
  claims 12799.4 s while its **audio ends at 5997.5 s**; the candidate is 6015.5 s — within
  **0.3 %** of the master's real audio. The shortfall is in the master's container, not the
  candidate. Its last two stage-1 delays (−824352, −1044576 ms) are windows past the audio end.
- **id 49 → candidate substantially incomplete**, a new row. 918.4 s against a master with
  1452.4 s of real audio and **no** audio gap — **36.8 %, 534 s, genuinely absent**. Stage-1
  places windows over `min(duration)`, spanning 120–791 s, so **the missing 534 s is never
  sampled**: the record shows a tidy 375 ms step at fidelity 0.955 and says nothing about the
  third of the programme that is gone. dev-2 measured **43.9 % master fill** on it, the highest
  in the corpus against a median of 8.4 % — a file at 1.2 % fill and one at 43.9 % are both
  "repaired" and are not the same result.

## What was dropped

- **The duration ratio as a general speed indicator.** Correct for PAL (4.27 %, large against
  content noise), useless for NTSC (0.1 %). I withdrew it wholesale on NTSC evidence and that
  over-generalisation cost 19 PAL files, which `vmsam-ci` caught.
- **Any criterion resting on a correlation peak between two suspect tracks** — the two traps
  in `RESTART.MD` §4. Replaced by differential measurement against a common third signal.
- **Treating "no confident peak" as a family.**
- **Reading a one-quantum difference as a measurement.** Two values on a 125 ms grid differ by
  "250 ms" only to ±125; it is a bound.

## What stayed blocked

- **`trimmed edges` = 32 is a floor. The HEAD mechanism is now bounded; the others are not.**
  Sweep of a seeded random 40 from 254 eligible, design and acceptance conditions fixed first:
  **zero head steps and ONE ISLAND** — id 108's offset departs from the body value between
  t=30–50 s, holds ~498 ms, and **returns** between t=130–150 s, so head and body end at the
  same offset. `vmsam-ci` bisected it independently to 56 s and 147 s. I first reported it as a
  head step: I had printed only the first three deviant probes, truncating the list before the
  return leg, and my outcome label could not distinguish a step from an island. Island rate
  **1/38 = 2.6 %**, and that one had **already been caught by another route**.
  **The island is the shape no instrument here sees** — every stage-1 window spans both
  transitions, so every window reads constant and the file looks like a clean constant offset. Net-new head steps missed by every
  route: **0 of 38**, one-sided 95 % upper bound 7.6 % → **≤ 19 files of 254**. So this
  mechanism is real but not materially distorting the count.
  My stated thresholds were >= 10 % and <= 2 %; 2.6 % falls between them and neither branch
  fires cleanly, which is a defect in how I set them, not a result.
  **The other blind spots remain unbounded**: my detector has a 150 ms threshold and tests one
  language pair per file, and `vmsam-ci` measured that a step under ~62 ms is invisible anywhere
  at any position with a median 15.9 % of every file never sampled. This sweep says nothing
  about those.
- **Why 5 files were refused.** 112 and 693 have no measurement at all.
- **The intra-master gap on 10 of the 11** rests on the log's bound, not a measurement.
- **Visual ground truth.** I have verified zero files by eye; all of it is ci's.
