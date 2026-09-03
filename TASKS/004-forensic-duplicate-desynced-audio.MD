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

## What changed

Nothing in this repository. The deliverable is
`reports/TASK_004_DUPLICATE_DESYNC.md` in the private notes beside the repository,
with the criterion, the per-id evidence, and the ids considered and rejected.

## The criterion, fixed before it was applied

A file is listed only when **two of its own audio tracks, each compared against the
same reference track on the other file, both yield a stable delay, and those two
delays differ by more than one measurement quantum.** Concretely, on the stage-1
matrix: two keys sharing the same index on the other side (`i1-j` and `i2-j`), both
with mean fidelity ≥ 0.85 across all ten windows, both stable to within one quantum,
and their two delays more than one quantum apart.

Both suspect tracks are necessarily of the **same language**: `extract_audio_in_part`
iterates `self.audios[language]` for the single common language chosen
(`video.py:275`), so matrix index `i` is the *i*-th compatible track of that one
language. Two tracks of one language in one release are exactly the tracks that
"should be identical".

**Why this survives the two traps in `CAMPAIGNS/01-dev-AI/RESTART.MD` §4.** Both traps
are about believing a correlation peak *between the two suspect tracks* — same source
across a lossy transcode reads 0.869–0.963 while different sources over a shared music
bed read 0.987, and two signals with no sample in common but a shared envelope read
0.979. **This measurement never computes that peak.** It is differential: each track is
correlated against a common third signal and what is compared is the two *offsets*.
Two dubs sharing only an envelope are not listed if they land at the same offset,
whatever they would score against each other. Fidelity ≥ 0.85 decides only whether a
track has a readable offset at all, never that two tracks are the same recording.

## What would put a file on the list wrongly

1. A commentary or audio-description track — it carries the programme audio underneath
   so it correlates, and it may legitimately sit at a different offset.
2. **Attribution to the wrong side.** Keys are `i-j` with `i` indexing the first path
   in the log's `... for <p1> and <p2>`. If the error file is `p2`, varying `i`
   describes the **master's** tracks. This is not hypothetical — it is what the data
   here actually contains.
3. Two same-language tracks with genuinely different content (a different edit or mix).
4. A quantum recovered wrongly; it is taken per pair as `|delay| / |offset_points|`.

## What was measured

| outcome | n |
| :--- | ---: |
| error files carrying >= 2 same-language audio tracks — the population at risk | **77** |
| of those, files where the pipeline measured >= 2 of *their own* tracks against a shared reference | **21** |
| **LISTED** — the error file's own two tracks disagree by more than one quantum | **0** |
| rejected — measured, tracks agree within one quantum (in sync) | 21 |
| **COULD NOT MEASURE** — at risk, but the pipeline never measured two of their tracks | **56** |

**The answer is not "zero".** It is zero of the 21 measured, and 56 not measured.
Recording those 56 as "no affected files" would be `RESTART.MD` §3.3's error exactly —
an instrument that did not run, written where a finding about the file is read.

**What the same measurement found instead: 11 library files.** Eleven error files'
**masters** each carry two same-language tracks, stable across all ten windows at
fidelity ~0.92, differing by exactly **250 ms** — error ids 6, 7, 10, 11, 14, 15, 16,
17, 18, 19, 24, all one show, one per episode. That is the defect this task hunts, on
the master side, in files already published in the library. It is also the measured
cause of those files' refusal: two stable delays 250 ms apart leave
`len(delay_detected) == 2` beyond the `< 127` reconciliation window at
`mergeVideo.py:315`. They are outside this task's literal scope, so they are reported
rather than listed; whether they join the re-download set is the owner's call.

## What was dropped

- **Any criterion resting on the similarity between the two suspect tracks.** The two
  campaign-1 traps show that quantity does not order these populations, so it is not
  used anywhere — not as a threshold, not as a tie-break.
- **Treating "no two tracks measured" as "not affected".** Kept as a third outcome
  instead, which is what produced the 56.

## What stayed blocked

- **The 56 at-risk files the pipeline never measured that way.** A re-measurement with
  an independent instrument, same differential logic, is running and unfinished. It can
  only move files from COULD NOT MEASURE into LISTED or in-sync; it cannot change the 21.
- **Whether the 250 ms on those 11 masters is in the media or in the extraction.** An
  encoder-delay difference between the two tracks would produce exactly this, and a
  figure identical across eleven files of one show is what an artefact looks like.
  **I have not excluded it**, and it is the first thing to test after the re-download.
- **Everything downstream of the list.** Per the task, the inventory stops here: the
  owner re-downloads these sources first, so whether the current code still reproduces
  any of it has no answer yet.
