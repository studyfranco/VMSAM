# 004 — Inventory: error files carrying two audio tracks that should be identical but are out of sync

**Agent:** `vmsam-forensic`. **State:** open. **Campaign objective 1.**

## Goal

List every error file that carries **two audio tracks that should be identical and
are not in sync**. Deliver the list and **stop there**: the owner re-downloads
those sources first. The question *does the current code still reproduce it, and
what would fix it* has no answer until that has happened, and answering it early
is work spent on files that are about to be replaced.

**Ids only. Never filenames.**

## What "should be identical" has to mean here

State your criterion before you apply it, and state what would put a pair in the
list wrongly. Campaign 1 left two traps on exactly this question
(`CAMPAIGNS/01-dev-AI/RESTART.MD` §4):

- `SAME_RECORDING_PEAK = 0.98` does not order these populations. Same source
  across a lossy transcode read **0.869–0.963**; different sources over a shared
  music bed read up to **0.987**.
- Two signals with **no sample in common**, sharing only an amplitude envelope,
  read **0.979**. Two dubs of one programme have exactly that structure.

A correlation peak alone will not do it. Say what you used instead.

## Deliverable

`VMSAM_HELP_AI/forensic/` — the id list, the criterion, the evidence per id, and
the ids you considered and rejected with why.

## What changed / measured / dropped / blocked
*(fill as you go)*
