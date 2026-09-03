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

Nothing in this repository. `vmsam-forensic` writes no code and no repository files
by charter; the deliverables are reports in the private notes beside the repository
(`reports/CRITERIA.md`, `reports/TASK_003_FAMILY_PROPORTIONS.md`, `reports/FINDINGS.md`,
and a per-id table `data/family_per_id.csv`).

## What was measured

**Method.** All 315 files classified from the pipeline's own refusal record — the
`.log.error` beside each error file, present for **315/315**. From it the per-window
`(correlation, offset_points, delay_ms)` triples are recovered per track pair.
Criteria were written down **before** the classifier ran, and campaign 1's `Class`,
`Step`, `Final Status` and `Detail` columns were **never read**; only its id, folder
and episode columns were extracted, mechanically, to compare the corpus.

**The corpus has not grown.** Its id set is identical to campaign 1's: 315/315,
symmetric difference empty, zero folder or episode mismatches. `CAMPAIGN.MD` and the
task brief both state new errors keep arriving; measured, none have.

**Proportions, 315 files, every file accounted for:**

**CORRECTED** after `vmsam-ci` refuted the first version's misfiled-content count with a
duration-ratio test, verified independently here. 19 files I had called `weak correlation`
are PAL speed mismatches.

| family | n | share | campaign 1 (unverified) |
| :--- | ---: | ---: | ---: |
| different cuts | 189 | 60.0 % | 57 % |
| weak correlation | 34 | 10.8 % | 10 % |
| trimmed edges | 31 | 9.8 % | 18 % |
| **constant offset — refused anyway** (not in `CAMPAIGN.MD`'s table) | 24 | 7.6 % | — |
| speed mismatch — **PAL** 25/23.976 | 19 | 6.0 % | *(11 % for speed overall)* |
| speed mismatch — **NTSC** 1000/1001 | 8 | 2.5 % | |
| no shared language | 7 | 2.2 % | 2 % |
| *instrument did not run — no measurement exists* | 2 | 0.6 % | — |
| *unclassifiable — measured, shape fits nothing* | 1 | 0.3 % | — |

Speed, both relations, is 27 files = 8.6 %. Four of campaign 1's five figures now agree
closely; `trimmed edges` (9.8 % against 18 %) is the outstanding gap and I have not
investigated it.

The last two lines are counted separately and are never folded into a family.

**Three results that change how the table should be read.**

1. **Both speed relations are present: 8 NTSC (0.098–0.102 %) and 19 PAL (4.271 %).**
   The first version of this task file said "no file in the corpus drifts at PAL's
   4.271 %", with the hedge that my method structurally could not see one. **The hedge
   was right and I failed to act on it.** `vmsam-ci` caught it with the duration ratio —
   one division per file. The 19 PAL files are ids 46, 51, 59, 63, 70, 77, 83, 87, 91,
   95, 98 (folder 110), 288, 289, 290, 293, 299, 300, 302 (folder 172), 101 (folder 54).
   The test needs **both** conditions: ratio within 0.001 of 1.042708 **and** the
   correlation actually collapsed — ids 135 and 190 have PAL-like ratios but correlate
   at 0.94 and 0.91, which an uncorrected speed relation cannot do, so they stay
   `different cuts`.
   I also **withdrew the duration ratio as a speed indicator** in the first version, on
   the strength of the NTSC files where duration difference is extra content. That
   observation was right; generalising it was not, and it deleted the one cheap
   instrument that would have caught the PAL population. Correct scope: usable for PAL
   (4.27 %, large against the few-percent content noise), useless for NTSC (0.1 %, far
   below it).
2. **A sixth population, 24 files (7.6 %), has a constant offset and was refused
   anyway** — delay spread ≤ 1 quantum across all ten windows at fidelity ≥ 0.85, 14
   of them with spread exactly 0 and fidelity to 0.995. For 11 of them the cause is
   measured: two of the **master's** same-language tracks are stable 250 ms apart, so
   `delay_detected` ends with two values beyond the `< 127` reconciliation window at
   `mergeVideo.py:315`. The candidate is refused for a defect in the file it is
   compared against. Error ids 6, 7, 10, 11, 14, 15, 16, 17, 18, 19, 24.
3. **`weak correlation` is not one phenomenon, and at least 41 files (13.0 %) share no
   measurable audio with their master.** `vmsam-ci` measured that error id 237 is a
   different television series from its folder's master. **ci's hedges, carried: ONE file
   verified visually, 25 attributed from the release name without being opened, 37
   folders unchecked, no claim about campaign 1.**
   - Folder 103, independently and without reading a filename, splits 26 / 23 in my
     classification — 26 at fidelity 0.564–0.590 against a 0.50 chance floor, 23 with
     real structure at 0.90–0.97. That is ci's partition from a method sharing no input
     with ci's.
   - Folder 110 — one of ci's 37 unchecked folders — **repeats the pattern, and this one
     is measured by me**: all 11 of its weak files read 0 of 12 confident probes at
     maximum prominence 0.033–0.056 against a whole-master search. Each is `[fre, eng]`
     and ~53 s shorter than its master, alternating with non-weak files that match.
     The innocent explanation is excluded: re-running ids 46, 51, 91 on the **English**
     track as well gives 0/12 on both languages, and a different French dub of the same
     episode would still share its music bed (campaign 1 measured that case at 0.987).
   - Totals: 26 (folder 103, ci's attribution) + 11 (folder 110) + 4 (other folders)
     = **41 files, 13.0 %**, with no measurable shared audio. A further 8 are marginal
     and **cannot be separated from noise**, and 4 show genuine partial overlap that is
     something else again.
   - **My own hedges: I have verified ZERO files visually** — my instrument measures
     audio only; the 0.10 confidence threshold was calibrated on two files and the wider
     sweep shows noise reaching 0.141, so it is too low and the 8 marginal files sit
     inside its error; and the 26 in folder 103 have not been through my prominence
     instrument, only through the fidelity partition.
   The consequence: `weak correlation` at 16.8 % must not be quoted as a single family.
   A repair addressed to it would be addressed to nothing.

   *(superseded note)* **`weak correlation` contains 26 files that are not a sync failure at all.**
   `vmsam-ci` measured that error id 237 is a different television series from its
   folder's master. Independently, and without reading any filename, folder id 103
   splits 26 / 23 in my classification: 26 at fidelity 0.564–0.590 (the 0.50 chance
   floor) and 23 with real correlation structure — the same partition ci reached from
   release names and video frames. **ci's hedges stand**: ci verified one file
   visually, attributed the other 25 without opening them, and left 37 folders
   unchecked. I am not yet reporting a misfiled-content proportion of my own.

**The measurement quantum is not 125 ms.** `audioCorrelation.get_max_corr` uses
`int(lengthFile/n_points*1000)`, computed per call; recovered from the logs as 124 in
one file and 125 in others. `vmsam-dev-1` independently measures the range as 124–142
ms, exactly 125 only when the shortest audio runs 18.7–30.4 minutes; that hedge is
carried as dev-1 stated it. Every tolerance in this work is expressed in quanta
recovered per file, never in a fixed millisecond figure.

**Exemplars**, screened against the gates `vmsam-dev-2` stated — genuine sync problem
(fidelity ≥ 0.85), a language shared with the master, the master carrying it, and no
`dts` in any shared-language track:

| family | exemplar ids | note |
| :--- | :--- | :--- |
| different cuts | **266**, **428**, **212** (then 202) | 266 is a clean 3-plateau staircase, 1000 ms per step, fidelity 0.938 |
| trimmed edges | **267**, **69**, **76** | single step at the start; 76 carries 8 languages |
| speed mismatch — NTSC | **33**, **31** | both 1000/1001; only 2 of 8 pass dev-2's gates |
| speed mismatch — PAL | **46**, **91**, **288** | 46 and 91 verified visually by ci; these carry **no usable delay measurement** — the ratio is the evidence, not a peak |
| constant offset — refused anyway | **22**, **23**, **21** | ten identical zero delays at fidelity ≈ 0.99 |
| weak correlation | **none usable** | nothing in it is a demonstrable sync problem |

Usable-per-family counts: different cuts 186/189, trimmed edges 28/31, constant offset
24/24, speed mismatch 2/8, weak correlation **0/53**, no shared language 0/7.

**Independently verified, 27 files.** A second instrument — one fingerprint of the
whole track per side, sliding a 60 s chunk against the *entire* master fingerprint at
12 probe positions of my choosing, so a different grid and a different search span from
the pipeline's ten fixed windows. Sample: a seeded random 10 drawn before their verdicts
were looked at, plus 17 targeted at the rare families.

First pass scored **21/27 = 78 %**, below the 80 % I had committed to. Inspecting all
six disagreements rather than adjusting anything found a defect in **my own tool**: a
linear ramp on every curve, caused by computing seconds-per-point as
`duration / len(fingerprint)` per side, which drifts wherever the two sides round
differently. Confirmed quantitatively — predicted ramp against observed: id 67 -538.8 vs
-517.6, id 163 +92.7 vs +91.8, id 266 +269.8 vs +251.9 ms/probe; every file with an spp
ratio of exactly 0 shows no ramp. Removing a diagnosed instrument error is not moving a
threshold, so the corrected figure is reported, but the 78 % is recorded too.

**Agreement after correction: 26/27 = 96 %.** The single disagreement, error id 5, has a
stated cause: its two edge probes (at 5 % and 95 % of the file) land on content with no
counterpart in the master given the ~16 s offset; interior probes 1-10 give the same
offset as the log to within 190 ms.

Three things this established:
- the shape families are in the media, not in the window placement or in the parser —
  step positions and magnitudes reproduce under a different grid (log 250 ms -> measured
  232-244; log 625 -> 620);
- **the speed family is confirmed as NTSC by both instruments.** All eight have an spp
  ratio of exactly 0, so their drift is not the artefact: independent 0.0998-0.1075 %
  against log-based 0.0982-0.1022 %, R2 >= 0.998 on both;
- a **prominence** discriminator (best similarity minus the best outside a neighbourhood
  of the peak) separates shared content from none with no overlap: 25 of 27 files score
  0.367-0.448 at 11-12 of 12 confident probes, while the two `weak correlation` files
  score **0.004-0.009 at 0/12**. One of the two is error id 237, the file `vmsam-ci`
  verified visually as a different series — so the instrument gives the right answer on
  the one case with independent ground truth.

Hedge: the sample holds only **two** `weak correlation` files and both are folder 103,
so the discriminator is calibrated on one confirmed positive, not on a population.

## What was dropped

- **Corroborating a speed slope with the container duration ratio.** It was in my
  written criteria before I measured. It failed: the eight speed files' durations
  differ by 4.7–6.9 % while their content drifts by 0.1 %, because the duration
  difference is extra content, not speed — which `docs/AUDIO_SPEED_POLICY.MD` line 137
  already says. Withdrawn because it failed, not adjusted until it passed.
- **Treating a correlation peak as evidence of anything on its own.** Chance for this
  statistic is 0.50, not 0, so a "0.6 correlation" is noise. Every threshold here is
  stated against that floor.
- **Moving the `constant offset` fidelity bar from 0.85 to 0.80** to absorb error id
  45, the single unclassifiable file. One file is not worth a fitted threshold; it is
  reported as a boundary case instead.

## What stayed blocked

- **Whether any PAL-rate file exists in this corpus.** The pipeline's own comparison
  windows are too long to hold correlation across a 4.27 % drift, so the question
  cannot be answered from its logs at all. A short-window re-measurement of all 53
  `weak correlation` files is the test; it is running and unfinished.
- **The corpus-wide misfiled-content count.** Corroborated in folder 103 only; 37
  folders unchecked by ci and not yet swept by me.
- **Why 13 of the 24 `constant offset` files were refused.** Only 11 have the
  two-track mechanism shown; the other 13 have a single measured pair and share only
  the symptom.
- **What ffmpeg failed on for error ids 112 and 693.** Both files' media probe fine
  under ffprobe 9.0.1 here, so the failure was in the pipeline's environment rather
  than necessarily in the file — but which, I have not shown. Recorded as *the
  instrument did not run*, never as *these files are unverifiable*.
