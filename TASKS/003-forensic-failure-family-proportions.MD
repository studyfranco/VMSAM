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
| different cuts | 190 | 60.3 % | 57 % |
| trimmed edges | 32 | 10.2 % | 18 % |
| content mismatch — wrong series | 27 | 8.6 % | *no row* |
| speed mismatch — PAL 25/23.976 | 20 | 6.3 % | *11 % for speed* |
| inter-track divergence — master side | 11 | 3.5 % | *no row* |
| speed mismatch — NTSC 1000/1001 | 8 | 2.5 % | |
| no shared language | 7 | 2.2 % | 2 % |
| per-language divergence — constant | 4 | 1.3 % | *no row* |
| constant offset — refused on a short-file quantum | 3 | 1.0 % | *no row* |
| defective master (audio truncated) | 2 | 0.6 % | *no row* |
| content mismatch — wrong episode | 2 | 0.6 % | *no row* |
| per-language divergence — one language steps | 2 | 0.6 % | *no row* |
| weak correlation | 2 | 0.6 % | 10 % |
| *instrument did not run* | 2 | 0.6 % | |
| constant offset — refused anyway | 1 | 0.3 % | |
| *unclassifiable* | 1 | 0.3 % | |
| no shared language — TAGS LIE | 1 | 0.3 % | *no row* |

**Unexplained: 5 of 315 (1.6 %)** — 112, 693 (instrument did not run); 196, 307 (partial
overlap); 45 (unclassifiable).

**FINDING for the owner — `CAMPAIGN.MD`'s table is wrong and it is frozen, so this is a
finding and not an edit. Both dev agents are steering by it.**

1. **`different cuts` is 190 (60.3 %), not 57 %.** Direction confirmed, magnitude higher.
2. **`content mismatch — wrong series` is 27 files (8.6 %) and has no row in that table.**
   No repair is meaningful on them.
3. **`weak correlation` was not a family.** It began at 53 files and ends at **2**. The other
   51 were four unrelated causes: 20 PAL speed relations, 27 wrong-series, 2 wrong-episode.
   `CAMPAIGN.MD` gives that row 10 %. "No confident peak" is a symptom.
4. **Speed is both relations**, 20 PAL and 8 NTSC. That row's prose says PAL while its rate
   ("~1 s per half-hour") is the NTSC figure.
5. **`trimmed edges` at 10.2 % is a floor and neither figure is defensible** — see blocked.

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

- **`trimmed edges` cannot be bounded from the record.** Stage-1 windows begin at 120 s and run
  2L, so a step is invisible until `T > 120 + L/2` (~174–227 s), and `vmsam-ci` showed the head
  is a special case: the geometry cannot see a step below its ~194 s resolution *anywhere*.
  My own profiler has a 150 ms threshold and misses steps ci found. A sweep to bound it is
  running; its own limits are declared in advance.
- **Why 5 files were refused.** 112 and 693 have no measurement at all.
- **The intra-master gap on 10 of the 11** rests on the log's bound, not a measurement.
- **Visual ground truth.** I have verified zero files by eye; all of it is ci's.
