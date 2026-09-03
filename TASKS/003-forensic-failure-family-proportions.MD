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
*(fill as you go)*

## What was measured
*(fill as you go)*

## What was dropped
*(fill as you go)*

## What stayed blocked
*(fill as you go)*
